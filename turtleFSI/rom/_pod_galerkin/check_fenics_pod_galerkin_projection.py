"""Check FEniCS residual/Jacobian projection for a monolithic POD basis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod_galerkin.ballarin_monolithic import block_basis_from_monolithic
from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem
from turtleFSI.rom._pod_galerkin.monolithic_pod import load_monolithic_pod_basis
from turtleFSI.rom._pod_galerkin.problem_overrides import parse_problem_overrides, replay_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="TF_csm", help="turtleFSI problem name.")
    parser.add_argument("--basis", required=True, help="monolithic_pod_basis.npz path.")
    parser.add_argument("--snapshots", required=True, help="Exact dvp_snapshots.npz path.")
    parser.add_argument("--snapshot-index", type=int, default=-1, help="Current snapshot index.")
    parser.add_argument("--previous-index", type=int, default=None, help="Previous snapshot index.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--dt", type=float, default=None, help="Override dt.")
    parser.add_argument("--theta", type=float, default=None, help="Override theta.")
    parser.add_argument(
        "--check-equation-3",
        action="store_true",
        help="Also compare summed Table I/Eq. 3 component blocks against Phi.T J Phi.",
    )
    parser.add_argument(
        "--paper-forms",
        action="store_true",
        help="Override compatible options for Ballarin/Rozza Section 2.3/Table I forms.",
    )
    parser.add_argument(
        "--problem-override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Additional problem namespace overrides.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    basis = load_monolithic_pod_basis(Path(args.basis))
    snapshots = np.load(args.snapshots)
    dvp = np.asarray(snapshots["dvp"], dtype=float)
    times = np.asarray(snapshots["time"], dtype=float) if "time" in snapshots.files else None

    current_index = args.snapshot_index
    if current_index < 0:
        current_index = dvp.shape[1] + current_index
    previous_index = args.previous_index
    if previous_index is None:
        previous_index = max(0, current_index - 1)
    elif previous_index < 0:
        previous_index = dvp.shape[1] + previous_index

    explicit_overrides = parse_problem_overrides(args.problem_override)
    if args.paper_forms:
        explicit_overrides.update(
            extrapolation="laplace",
            extrapolation_sub_type="paper",
            material_model="BallarinRozzaLinearElastic",
            solid_properties=[],
        )
    if args.dt is not None:
        explicit_overrides["dt"] = args.dt
    elif times is not None and current_index != previous_index:
        explicit_overrides["dt"] = float(times[current_index] - times[previous_index])
    if args.theta is not None:
        explicit_overrides["theta"] = args.theta

    overrides = replay_overrides(Path(args.snapshots), explicit_overrides=explicit_overrides)
    problem = FEniCSMonolithicProblem.build(args.problem, overrides=overrides)
    current_state = dvp[:, current_index]
    previous_state = dvp[:, previous_index]
    problem.set_previous_state(previous_state)

    residual = problem.projected_residual(basis.pod_basis.basis, current_state)
    jacobian = problem.projected_jacobian(basis.pod_basis.basis, current_state)
    equation_3_check = None
    if args.check_equation_3:
        block_basis = block_basis_from_monolithic(basis)
        equation_3_check = problem.check_equation_3_projection(block_basis, current_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "residual": residual,
        "jacobian": jacobian,
    }
    if equation_3_check is not None:
        arrays.update(
            equation_3_operator=equation_3_check.equation_3_operator,
            equation_3_difference=equation_3_check.difference,
            equation_3_projected_jacobian=equation_3_check.projected_jacobian,
        )
    np.savez_compressed(output_dir / "fenics_projected_assembly.npz", **arrays)
    summary = {
        "problem": args.problem,
        "basis": str(args.basis),
        "snapshots": str(args.snapshots),
        "snapshot_index": int(current_index),
        "previous_index": int(previous_index),
        "dt": float(overrides.get("dt", problem.namespace["dt"])),
        "num_full_dofs": int(problem.vector_size),
        "num_modes": int(basis.num_modes),
        "projected_residual_norm": float(np.linalg.norm(residual)),
        "projected_jacobian_shape": list(jacobian.shape),
        "projected_jacobian_condition": float(np.linalg.cond(jacobian))
        if jacobian.size > 0
        else None,
        "equation_3_checked": equation_3_check is not None,
        "paper_forms": bool(args.paper_forms),
    }
    if equation_3_check is not None:
        summary.update(
            equation_3_operator_shape=list(equation_3_check.equation_3_operator.shape),
            equation_3_difference_norm=equation_3_check.difference_norm,
            equation_3_relative_difference_norm=equation_3_check.relative_difference_norm,
            equation_3_component_count=len(equation_3_check.component_projection.component_blocks),
        )
    with open(output_dir / "fenics_projected_assembly_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote FEniCS projected assembly check to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
