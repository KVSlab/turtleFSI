"""Build and run a case-level monolithic POD-Galerkin ROM.

This wrapper hides the lower-level POD-Galerkin scripts behind one workflow:

1. Find or accept ``dvp_snapshots.npz``.
2. Build a monolithic POD basis.
3. Optionally run one FEniCS-projected step or a rollout.
4. For Ballarin/Rozza-style runs, build and solve a reduced operator bundle.

The default scheme is the current turtleFSI monolithic projection path. The
Ballarin/Rozza 2016 path uses block displacement, velocity, pressure, and
supremizer spaces plus an offline-online reduced operator bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from turtleFSI.rom._shared.pod import relative_l2_error
from turtleFSI.rom._pod_galerkin import build_monolithic_pod_basis
from turtleFSI.rom._pod_galerkin.monolithic_pod import (
    project_monolithic_snapshots,
    reconstruct_monolithic_state,
    save_monolithic_pod_basis,
    train_monolithic_pod_basis,
)


def snapshot_path_from_run(run_dir: Path) -> Path:
    return run_dir / "ROM" / "dvp_snapshots.npz"


def default_output_dir(snapshots_path: Path) -> Path:
    if snapshots_path.name == "dvp_snapshots.npz" and snapshots_path.parent.name == "ROM":
        return snapshots_path.parent / "pod_galerkin"
    return snapshots_path.parent / "pod_galerkin"


def infer_problem_name(snapshots_path: Path) -> str | None:
    summary_path = snapshots_path.with_name("dvp_snapshots_summary.json")
    if not summary_path.exists():
        return None
    with open(summary_path, encoding="ascii") as handle:
        summary = json.load(handle)
    problem = summary.get("problem")
    return str(problem) if problem else None


def snapshot_count(snapshots_path: Path) -> int:
    dataset = np.load(snapshots_path)
    if "dvp" in dataset.files:
        return int(np.asarray(dataset["dvp"]).shape[1])
    snapshot_lengths = [
        int(np.asarray(dataset[name]).shape[1])
        for name in dataset.files
        if np.asarray(dataset[name]).ndim == 2 and not name.endswith("_dofs")
    ]
    if not snapshot_lengths:
        raise ValueError(f"No snapshot matrices found in {snapshots_path}")
    return min(snapshot_lengths)


def _gauge_values(values: np.ndarray, gauge: str) -> np.ndarray:
    if gauge == "none":
        return values
    if gauge == "zero-mean":
        return values - values.mean(axis=0, keepdims=True)
    raise ValueError(f"Unknown pressure gauge {gauge!r}")


def _validation_field_metrics(
    snapshots_path: Path,
    rollout_path: Path,
    rollout_summary: dict,
    pressure_gauge: str,
    pressure_dof_key: str,
    output_path: Path,
) -> dict:
    snapshots = np.load(snapshots_path)
    rollout = np.load(rollout_path)
    if "dvp" not in snapshots.files:
        raise KeyError("Validation field metrics require a 'dvp' snapshot matrix.")
    if "full" not in rollout.files:
        raise KeyError("Validation field metrics require rollout full-state history.")

    rollout_start = int(rollout_summary["start_index"])
    full = np.asarray(rollout["full"], dtype=float)
    start_index = rollout_start + 1
    stop_index = rollout_start + full.shape[1] - 1
    if stop_index < start_index:
        raise ValueError("Rollout must contain at least one predicted state.")

    pressure_key = pressure_dof_key
    if pressure_key == "auto":
        for candidate in ("fluid_pressure_dofs", "pressure_active_dofs", "pressure_dofs"):
            if candidate in snapshots.files:
                pressure_key = candidate
                break
    field_dof_keys = {
        "displacement": "displacement_dofs",
        "velocity": "velocity_dofs",
        "pressure": pressure_key,
    }
    metrics = {
        "snapshots": str(snapshots_path),
        "rollout": str(rollout_path),
        "train_indices": [0, int(rollout_start)],
        "test_indices": [int(start_index), int(stop_index)],
        "reference_relative_error": rollout_summary.get("reference_relative_error"),
        "field_relative_errors": {},
        "per_index": {},
    }

    for index in range(start_index, stop_index + 1):
        metrics["per_index"][str(index)] = {}

    for field, dof_key in field_dof_keys.items():
        if dof_key not in snapshots.files:
            continue
        dofs = np.asarray(snapshots[dof_key], dtype=np.int64)
        true_values = np.asarray(snapshots["dvp"][dofs, start_index : stop_index + 1], dtype=float)
        rom_values = np.asarray(full[dofs, 1 : stop_index - rollout_start + 1], dtype=float)
        if field == "pressure":
            true_values = _gauge_values(true_values, pressure_gauge)
            rom_values = _gauge_values(rom_values, pressure_gauge)
        metrics["field_relative_errors"][field] = relative_l2_error(true_values, rom_values)
        for column, index in enumerate(range(start_index, stop_index + 1)):
            metrics["per_index"][str(index)][field] = relative_l2_error(
                true_values[:, column],
                rom_values[:, column],
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="ascii") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics


def _build_standard_basis(
    snapshots_path: Path,
    output_dir: Path,
    energy_threshold: float,
    max_modes: int | None,
    start_index: int,
    stop_index: int | None,
) -> dict:
    dataset = np.load(snapshots_path)
    fields = ["dvp"] if "dvp" in dataset.files else [
        name
        for name in dataset.files
        if np.asarray(dataset[name]).ndim == 2 and not name.endswith("_dofs")
    ]
    blocks = {
        name: np.asarray(dataset[name], dtype=float)[:, start_index:stop_index]
        for name in fields
    }
    basis, snapshots = train_monolithic_pod_basis(
        blocks=blocks,
        field_order=fields,
        energy_threshold=energy_threshold,
        max_modes=max_modes,
    )
    coefficients = project_monolithic_snapshots(snapshots, basis)
    reconstruction = reconstruct_monolithic_state(coefficients, basis)
    reconstruction_error = relative_l2_error(snapshots, reconstruction)
    summary = save_monolithic_pod_basis(
        output_dir=output_dir,
        basis=basis,
        snapshots_path=str(snapshots_path),
        reconstruction_error=reconstruction_error,
    )
    np.savez_compressed(
        output_dir / "monolithic_pod_projection.npz",
        coefficients=coefficients,
        reconstruction=reconstruction,
    )
    return summary


def build_basis(args: argparse.Namespace, snapshots_path: Path, basis_dir: Path) -> dict:
    basis_start_index = args.basis_start_index
    basis_stop_index = args.basis_stop_index
    if basis_start_index is None:
        if args.action in {"basis", "step", "rollout", "validate-rollout"}:
            basis_start_index = args.start_index
        else:
            basis_start_index = 0
    if basis_stop_index is None:
        if args.action in {"basis", "step", "rollout", "validate-rollout"}:
            basis_stop_index = args.stop_index
        else:
            basis_stop_index = None

    if args.blocked_dvp:
        # Reuse the existing blocked-DVP implementation without exposing it as
        # the primary user workflow.
        argv = [
            "build_monolithic_pod_basis",
            "--snapshots",
            str(snapshots_path),
            "--output-dir",
            str(basis_dir),
            "--blocked-dvp",
            "--energy-threshold",
            str(args.energy_threshold),
        ]
        if args.max_modes is not None:
            argv.extend(["--max-modes", str(args.max_modes)])
        if args.displacement_modes is not None:
            argv.extend(["--displacement-modes", str(args.displacement_modes)])
        if args.velocity_modes is not None:
            argv.extend(["--velocity-modes", str(args.velocity_modes)])
        if args.pressure_modes is not None:
            argv.extend(["--pressure-modes", str(args.pressure_modes)])
        if args.supremizer_modes is not None:
            argv.extend(["--supremizer-modes", str(args.supremizer_modes)])
        argv.extend(["--pressure-treatment", args.pressure_treatment])
        argv.extend(["--pressure-dof-key", args.pressure_dof_key])
        argv.extend(["--pressure-gauge", args.pressure_gauge])
        argv.extend(["--start-index", str(basis_start_index)])
        if basis_stop_index is not None:
            argv.extend(["--stop-index", str(basis_stop_index)])

        old_argv = sys.argv
        try:
            sys.argv = argv
            build_monolithic_pod_basis.main()
        finally:
            sys.argv = old_argv
        with open(basis_dir / "monolithic_pod_basis_summary.json", encoding="ascii") as handle:
            return json.load(handle)

    return _build_standard_basis(
        snapshots_path=snapshots_path,
        output_dir=basis_dir,
        energy_threshold=args.energy_threshold,
        max_modes=args.max_modes,
        start_index=basis_start_index,
        stop_index=basis_stop_index,
    )


def run_module(module: str, arguments: list[str]) -> None:
    command = [sys.executable, "-m", module, *arguments]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="Simulation run directory containing ROM/dvp_snapshots.npz.")
    source.add_argument("--snapshots", help="Path to dvp_snapshots.npz.")
    parser.add_argument(
        "--preset",
        choices=("none", "ballarin-validation"),
        default="none",
        help=(
            "Apply a named workflow preset. 'ballarin-validation' selects the "
            "blocked DVP, supremizer, fluid-pressure, equilibrated validation "
            "rollout path while still allowing explicit non-default flags to override it."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to RUN/ROM/pod_galerkin.",
    )
    parser.add_argument(
        "--problem",
        default=None,
        help="turtleFSI problem name. Required for --action step or rollout if not in snapshot summary.",
    )
    parser.add_argument(
        "--action",
        choices=(
            "basis",
            "step",
            "rollout",
            "validate-rollout",
            "build-operators",
            "operator-solve",
            "build-operator-series",
            "operator-series-solve",
        ),
        default="basis",
        help=(
            "Build only a basis, run a projected step/rollout, build a Ballarin "
            "operator bundle or series, solve saved reduced operators, or run "
            "an 80/20 validation rollout with plots and metrics."
        ),
    )
    parser.add_argument(
        "--scheme",
        choices=("monolithic", "ballarin-monolithic"),
        default="monolithic",
        help=(
            "'monolithic' uses the current projected turtleFSI residual/Jacobian. "
            "'ballarin-monolithic' uses the Ballarin/Rozza 2016-style "
            "block/supremizer/offline-online formulation."
        ),
    )
    parser.add_argument(
        "--pressure-treatment",
        choices=("none", "supremizer"),
        default="none",
        help="Pressure stabilization strategy. Ballarin/Rozza 2016 uses supremizer enrichment.",
    )
    parser.add_argument("--blocked-dvp", action="store_true")
    parser.add_argument("--energy-threshold", type=float, default=0.999)
    parser.add_argument(
        "--exact-modes",
        action="store_true",
        help=(
            "Use requested mode counts as exact caps by setting the POD energy "
            "threshold to 1.0 during basis construction. The retained count can "
            "still be lower if the snapshot rank or supremizer orthogonalization "
            "does not provide enough independent modes."
        ),
    )
    parser.add_argument("--max-modes", type=int, default=None)
    parser.add_argument("--displacement-modes", type=int, default=None)
    parser.add_argument("--velocity-modes", type=int, default=None)
    parser.add_argument("--pressure-modes", type=int, default=None)
    parser.add_argument("--supremizer-modes", type=int, default=None)
    parser.add_argument(
        "--pressure-dof-key",
        default="auto",
        help=(
            "Snapshot array used for pressure POD DoFs. 'auto' uses "
            "fluid_pressure_dofs/pressure_active_dofs when present, otherwise pressure_dofs."
        ),
    )
    parser.add_argument(
        "--pressure-gauge",
        choices=("none", "zero-mean"),
        default="none",
        help="Remove the per-snapshot pressure mean before building pressure POD modes.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int, default=None)
    parser.add_argument(
        "--basis-start-index",
        type=int,
        default=None,
        help=(
            "Optional first snapshot used to train the POD basis. Defaults to "
            "the action's start-index for basis/step/rollout and 0 for "
            "offline-online operator actions."
        ),
    )
    parser.add_argument(
        "--basis-stop-index",
        type=int,
        default=None,
        help=(
            "Optional exclusive stop snapshot used to train the POD basis. "
            "Defaults to the action's stop-index for basis/step/rollout and "
            "all snapshots for offline-online operator actions."
        ),
    )
    parser.add_argument("--target-index", type=int, default=-1)
    parser.add_argument(
        "--reference-index",
        type=int,
        default=0,
        help="Snapshot index used to linearize the Ballarin reduced operator bundle.",
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--no-line-search", action="store_true")
    parser.add_argument(
        "--jacobian-update-frequency",
        type=int,
        default=1,
        help=(
            "Update the projected Jacobian every N Newton corrections. "
            "Use 1 for exact Newton; larger values reuse the projected Jacobian."
        ),
    )
    parser.add_argument(
        "--retry-with-fresh-jacobian",
        action="store_true",
        help="If line search fails, rebuild the projected Jacobian and retry.",
    )
    parser.add_argument(
        "--operator-bundle",
        default=None,
        help=(
            "Reduced operator bundle path. For --action build-operators this is "
            "the output path; for --action operator-solve this is the input path. "
            "Defaults to OUTPUT_DIR/operators/reduced_operators.npz for build-operators."
        ),
    )
    parser.add_argument(
        "--operator-series-dir",
        default=None,
        help=(
            "Directory for --action build-operator-series. Defaults to "
            "OUTPUT_DIR/operator_series."
        ),
    )
    parser.add_argument(
        "--operator-series-manifest",
        default=None,
        help="operator_series_manifest.json path for --action operator-series-solve.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Training fraction used by --action validate-rollout.",
    )
    parser.add_argument(
        "--skip-validation-plots",
        action="store_true",
        help="For --action validate-rollout, skip slice/surface animation generation.",
    )
    parser.add_argument(
        "--surface-validation-plots",
        action="store_true",
        help="For --action validate-rollout, also write 3D exterior surface animations.",
    )
    parser.add_argument(
        "--plot-fields",
        nargs="+",
        choices=("displacement", "velocity", "pressure"),
        default=["displacement", "velocity", "pressure"],
        help="Fields plotted by --action validate-rollout.",
    )
    parser.add_argument("--plot-fps", type=int, default=2)
    parser.add_argument("--slice-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--slice-value", type=float, default=None)
    parser.add_argument("--slice-tolerance", type=float, default=None)
    parser.add_argument("--plot-component", default="magnitude")
    parser.add_argument("--plot-levels", type=int, default=40)
    parser.add_argument("--plot-elev", type=float, default=18.0)
    parser.add_argument("--plot-azim", type=float, default=-65.0)
    parser.add_argument(
        "--linear-solver",
        choices=("solve", "lstsq", "equilibrated-lstsq"),
        default="solve",
    )
    parser.add_argument("--linear-rcond", type=float, default=None)
    parser.add_argument("--equilibration-iterations", type=int, default=8)
    return parser.parse_args()


def apply_workflow_preset(args: argparse.Namespace) -> dict[str, object]:
    """Apply conservative defaults for common POD-Galerkin workflows."""

    if args.preset == "none":
        return {}
    if args.preset != "ballarin-validation":
        raise ValueError(f"Unknown workflow preset {args.preset!r}")

    applied: dict[str, object] = {}

    def set_if_current(name: str, current_value, preset_value) -> None:
        if getattr(args, name) == current_value:
            setattr(args, name, preset_value)
            applied[name] = preset_value

    set_if_current("action", "basis", "validate-rollout")
    set_if_current("scheme", "monolithic", "monolithic")
    if not args.blocked_dvp:
        args.blocked_dvp = True
        applied["blocked_dvp"] = True
    set_if_current("pressure_treatment", "none", "supremizer")
    set_if_current("pressure_dof_key", "auto", "fluid_pressure_dofs")
    set_if_current("pressure_gauge", "none", "zero-mean")
    set_if_current("linear_solver", "solve", "equilibrated-lstsq")
    if args.linear_rcond is None:
        args.linear_rcond = 1e-12
        applied["linear_rcond"] = 1e-12
    if not args.exact_modes:
        args.exact_modes = True
        applied["exact_modes"] = True
    return applied


def main() -> None:
    args = parse_args()
    preset_options = apply_workflow_preset(args)
    if args.exact_modes:
        args.energy_threshold = 1.0
    ballarin_actions = {
        "build-operators",
        "operator-solve",
        "build-operator-series",
        "operator-series-solve",
    }
    if args.scheme == "ballarin-monolithic" and args.action not in ballarin_actions:
        raise NotImplementedError(
            "Ballarin/Rozza 2016 mode is implemented through --action "
            "build-operators, operator-solve, build-operator-series, and "
            "operator-series-solve. Use the default monolithic scheme for "
            "direct FEniCS-projected step or rollout runs."
        )
    if args.action in ballarin_actions and args.scheme != "ballarin-monolithic":
        raise ValueError(f"--action {args.action} requires --scheme ballarin-monolithic")
    if args.action == "operator-solve" and args.operator_bundle is None:
        raise ValueError("--action operator-solve requires --operator-bundle")
    if args.action == "operator-series-solve" and args.operator_series_manifest is None:
        raise ValueError("--action operator-series-solve requires --operator-series-manifest")
    if args.pressure_treatment == "supremizer" and not args.blocked_dvp:
        raise ValueError("--pressure-treatment supremizer requires --blocked-dvp")
    if args.action == "validate-rollout" and not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1 for --action validate-rollout")
    snapshots_path = (
        Path(args.snapshots)
        if args.snapshots
        else snapshot_path_from_run(Path(args.run_dir))
    )
    if not snapshots_path.exists():
        raise FileNotFoundError(
            f"DVP snapshot bundle does not exist: {snapshots_path}\n"
            "Run the full simulation with rom_export_dvp enabled, then retry."
        )

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(snapshots_path)
    validation_split = None
    if args.action == "validate-rollout":
        total_snapshots = snapshot_count(snapshots_path)
        train_stop_index = int(np.ceil(args.train_fraction * total_snapshots))
        train_stop_index = max(1, min(total_snapshots - 1, train_stop_index))
        args.basis_start_index = 0 if args.basis_start_index is None else args.basis_start_index
        args.basis_stop_index = train_stop_index if args.basis_stop_index is None else args.basis_stop_index
        args.start_index = train_stop_index - 1
        args.num_steps = total_snapshots - train_stop_index if args.num_steps is None else args.num_steps
        validation_split = {
            "total_snapshots": int(total_snapshots),
            "train_fraction": float(args.train_fraction),
            "basis_start_index": int(args.basis_start_index),
            "basis_stop_index": int(args.basis_stop_index),
            "rollout_start_index": int(args.start_index),
            "num_steps": int(args.num_steps),
            "test_start_index": int(args.start_index + 1),
            "test_stop_index": int(args.start_index + args.num_steps),
        }
    basis_dir = output_dir / "basis"
    if args.action == "operator-series-solve":
        with open(args.operator_series_manifest, encoding="ascii") as handle:
            manifest = json.load(handle)
        basis_dir = Path(manifest["basis"]).parent
        summary_path = Path(manifest["basis"]).with_name("monolithic_pod_basis_summary.json")
        with open(summary_path, encoding="ascii") as handle:
            summary = json.load(handle)
    else:
        summary = build_basis(args, snapshots_path, basis_dir)

    workflow_summary = {
        "snapshots": str(snapshots_path),
        "output_dir": str(output_dir),
        "basis_dir": str(basis_dir),
        "basis": summary,
        "action": args.action,
        "preset": args.preset,
        "preset_options": preset_options,
    }
    if validation_split is not None:
        workflow_summary["validation_split"] = validation_split
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.action == "build-operators":
        problem = args.problem or infer_problem_name(snapshots_path)
        if problem is None:
            raise ValueError("--problem is required when it cannot be inferred from the snapshot summary")
        operator_bundle = (
            Path(args.operator_bundle)
            if args.operator_bundle
            else output_dir / "operators" / "reduced_operators.npz"
        )
        arguments = [
            "--problem",
            problem,
            "--basis",
            str(basis_dir / "monolithic_pod_basis.npz"),
            "--snapshots",
            str(snapshots_path),
            "--output",
            str(operator_bundle),
            "--reference-index",
            str(args.reference_index),
        ]
        if args.dt is not None:
            arguments.extend(["--dt", str(args.dt)])
        if args.theta is not None:
            arguments.extend(["--theta", str(args.theta)])
        run_module(
            "turtleFSI.rom._pod_galerkin.build_ballarin_operator_bundle",
            arguments,
        )
        workflow_summary["operator_bundle"] = str(operator_bundle)
        workflow_summary["reference_index"] = int(args.reference_index)

    if args.action == "build-operator-series":
        problem = args.problem or infer_problem_name(snapshots_path)
        if problem is None:
            raise ValueError("--problem is required when it cannot be inferred from the snapshot summary")
        operator_series_dir = (
            Path(args.operator_series_dir)
            if args.operator_series_dir
            else output_dir / "operator_series"
        )
        arguments = [
            "--problem",
            problem,
            "--basis",
            str(basis_dir / "monolithic_pod_basis.npz"),
            "--snapshots",
            str(snapshots_path),
            "--output-dir",
            str(operator_series_dir),
            "--start-index",
            str(args.start_index),
            "--stride",
            str(args.stride),
        ]
        if args.stop_index is not None:
            arguments.extend(["--stop-index", str(args.stop_index)])
        if args.dt is not None:
            arguments.extend(["--dt", str(args.dt)])
        if args.theta is not None:
            arguments.extend(["--theta", str(args.theta)])
        run_module(
            "turtleFSI.rom._pod_galerkin.build_ballarin_operator_series",
            arguments,
        )
        workflow_summary["operator_series_dir"] = str(operator_series_dir)

    if args.action == "operator-solve":
        action_dir = output_dir / "operator_solve"
        arguments = [
            "--basis",
            str(basis_dir / "monolithic_pod_basis.npz"),
            "--operators",
            str(args.operator_bundle),
            "--output-dir",
            str(action_dir),
            "--linear-solver",
            args.linear_solver,
            "--equilibration-iterations",
            str(args.equilibration_iterations),
        ]
        if args.linear_rcond is not None:
            arguments.extend(["--linear-rcond", str(args.linear_rcond)])
        run_module(
            "turtleFSI.rom._pod_galerkin.run_ballarin_operator_solve",
            arguments,
        )
        if args.linear_rcond is not None:
            workflow_summary["linear_rcond"] = args.linear_rcond
        workflow_summary["operator_solve_dir"] = str(action_dir)

    if args.action == "operator-series-solve":
        action_dir = output_dir / "operator_series_solve"
        arguments = [
            "--basis",
            str(basis_dir / "monolithic_pod_basis.npz"),
            "--manifest",
            str(args.operator_series_manifest),
            "--snapshots",
            str(snapshots_path),
            "--output-dir",
            str(action_dir),
            "--linear-solver",
            args.linear_solver,
            "--equilibration-iterations",
            str(args.equilibration_iterations),
        ]
        if args.linear_rcond is not None:
            arguments.extend(["--linear-rcond", str(args.linear_rcond)])
        run_module(
            "turtleFSI.rom._pod_galerkin.run_ballarin_operator_series_solve",
            arguments,
        )
        workflow_summary["operator_series_solve_dir"] = str(action_dir)

    if args.action in {"step", "rollout", "validate-rollout"}:
        problem = args.problem or infer_problem_name(snapshots_path)
        if problem is None:
            raise ValueError("--problem is required when it cannot be inferred from the snapshot summary")
        common = [
            "--problem",
            problem,
            "--basis",
            str(basis_dir / "monolithic_pod_basis.npz"),
            "--snapshots",
            str(snapshots_path),
            "--atol",
            str(args.atol),
            "--rtol",
            str(args.rtol),
            "--max-iterations",
            str(args.max_iterations),
            "--damping",
            str(args.damping),
            "--jacobian-update-frequency",
            str(args.jacobian_update_frequency),
            "--linear-solver",
            args.linear_solver,
            "--pressure-gauge",
            args.pressure_gauge,
        ]
        if args.linear_rcond is not None:
            common.extend(["--linear-rcond", str(args.linear_rcond)])
        if args.dt is not None:
            common.extend(["--dt", str(args.dt)])
        if args.theta is not None:
            common.extend(["--theta", str(args.theta)])
        if args.no_line_search:
            common.append("--no-line-search")
        if args.retry_with_fresh_jacobian:
            common.append("--retry-with-fresh-jacobian")

        if args.action == "step":
            action_dir = output_dir / "step"
            run_module(
                "turtleFSI.rom._pod_galerkin.run_fenics_pod_galerkin_step",
                [
                    *common,
                    "--target-index",
                    str(args.target_index),
                    "--output-dir",
                    str(action_dir),
                ],
            )
            workflow_summary["step_dir"] = str(action_dir)
        else:
            if args.num_steps is None:
                raise ValueError("--num-steps is required for --action rollout")
            action_dir = output_dir / ("validation_rollout" if args.action == "validate-rollout" else "rollout")
            run_module(
                "turtleFSI.rom._pod_galerkin.run_fenics_pod_galerkin_rollout",
                [
                    *common,
                    "--start-index",
                    str(args.start_index),
                    "--num-steps",
                    str(args.num_steps),
                    "--output-dir",
                    str(action_dir),
                ],
            )
            workflow_summary["rollout_dir"] = str(action_dir)
            if args.action == "validate-rollout":
                rollout_summary_path = action_dir / "fenics_pod_galerkin_rollout_summary.json"
                with open(rollout_summary_path, encoding="ascii") as handle:
                    rollout_summary = json.load(handle)
                metrics_path = action_dir / "field_error_metrics.json"
                metrics = _validation_field_metrics(
                    snapshots_path=snapshots_path,
                    rollout_path=action_dir / "fenics_pod_galerkin_rollout.npz",
                    rollout_summary=rollout_summary,
                    pressure_gauge=args.pressure_gauge,
                    pressure_dof_key=args.pressure_dof_key,
                    output_path=metrics_path,
                )
                workflow_summary["validation_metrics"] = str(metrics_path)
                workflow_summary["validation_field_relative_errors"] = metrics["field_relative_errors"]
                workflow_summary["validation_reference_relative_error"] = metrics["reference_relative_error"]
                if not args.skip_validation_plots:
                    plot_common = [
                        "--rollout",
                        str(action_dir / "fenics_pod_galerkin_rollout.npz"),
                        "--summary",
                        str(rollout_summary_path),
                        "--snapshots",
                        str(snapshots_path),
                        "--problem",
                        problem,
                        "--start-index",
                        str(args.start_index),
                        "--stop-index",
                        str(args.start_index + args.num_steps),
                        "--train-stop-index",
                        str(args.start_index + 1),
                        "--pressure-gauge",
                        args.pressure_gauge,
                        "--fields",
                        *args.plot_fields,
                        "--fps",
                        str(args.plot_fps),
                    ]
                    slice_dir = output_dir / "validation_slice_animations"
                    run_module(
                        "turtleFSI.cases.plots.plot_fenics_pod_galerkin_rollout",
                        [
                            *plot_common,
                            "--output-dir",
                            str(slice_dir),
                            "--slice-animation",
                            "--slice-axis",
                            args.slice_axis,
                            "--component",
                            args.plot_component,
                            "--levels",
                            str(args.plot_levels),
                            *(
                                ["--slice-value", str(args.slice_value)]
                                if args.slice_value is not None
                                else []
                            ),
                            *(
                                ["--slice-tolerance", str(args.slice_tolerance)]
                                if args.slice_tolerance is not None
                                else []
                            ),
                        ],
                    )
                    workflow_summary["validation_slice_animation_dir"] = str(slice_dir)
                    if args.surface_validation_plots:
                        surface_dir = output_dir / "validation_surface_animations"
                        run_module(
                            "turtleFSI.cases.plots.plot_fenics_pod_galerkin_rollout",
                            [
                                *plot_common,
                                "--output-dir",
                                str(surface_dir),
                                "--surface-animation",
                                "--elev",
                                str(args.plot_elev),
                                "--azim",
                                str(args.plot_azim),
                            ],
                        )
                        workflow_summary["validation_surface_animation_dir"] = str(surface_dir)

    with open(output_dir / "case_pod_galerkin_summary.json", "w", encoding="ascii") as handle:
        json.dump(workflow_summary, handle, indent=2, sort_keys=True)

    print(f"Wrote case POD-Galerkin workflow to {output_dir}")
    print(json.dumps(workflow_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
