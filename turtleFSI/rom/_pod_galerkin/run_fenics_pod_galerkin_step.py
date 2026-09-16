"""Run one FEniCS-projected monolithic POD-Galerkin Newton step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem
from turtleFSI.rom._pod_galerkin.monolithic_pod import load_monolithic_pod_basis
from turtleFSI.rom._pod_galerkin.monolithic_pod import split_reduced_coefficients
from turtleFSI.rom._pod_galerkin.problem_overrides import (
    parse_problem_overrides,
    replay_overrides,
)
from turtleFSI.rom._shared.pod import relative_l2_error
from turtleFSI.rom._pod_galerkin.reduced_solvers import reduced_newton_solve_projected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="TF_csm", help="turtleFSI problem name.")
    parser.add_argument("--basis", required=True, help="monolithic_pod_basis.npz path.")
    parser.add_argument("--snapshots", required=True, help="Exact dvp_snapshots.npz path.")
    parser.add_argument("--target-index", type=int, default=-1, help="Snapshot index to solve for.")
    parser.add_argument("--previous-index", type=int, default=None, help="Previous time-level snapshot index.")
    parser.add_argument(
        "--initial-index",
        type=int,
        default=None,
        help="Snapshot index used as initial reduced guess. Defaults to previous-index.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument(
        "--problem-override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Additional problem namespace overrides passed to the FEniCS problem builder.",
    )
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--no-line-search", action="store_true")
    parser.add_argument("--min-damping", type=float, default=1e-6)
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
        help="If line search fails, rebuild the projected Jacobian and retry instead of stopping immediately.",
    )
    parser.add_argument(
        "--linear-solver",
        choices=("solve", "lstsq", "equilibrated-lstsq"),
        default="lstsq",
        help=(
            "Reduced linear correction solver. 'lstsq' and "
            "'equilibrated-lstsq' are safer for pressure-saddle-point systems."
        ),
    )
    parser.add_argument(
        "--linear-rcond",
        type=float,
        default=1e-12,
        help="Relative singular-value cutoff used when --linear-solver=lstsq.",
    )
    parser.add_argument(
        "--no-enforce-dirichlet",
        action="store_true",
        help="Do not constrain reduced increments or reconstructed states to current Dirichlet BCs.",
    )
    parser.add_argument(
        "--pressure-gauge",
        choices=("none", "zero-mean"),
        default="none",
        help="Apply the same pressure gauge used when building the pressure POD basis.",
    )
    parser.add_argument(
        "--pressure-error-dof-key",
        default="auto",
        help=(
            "Snapshot DoF key used for pressure error reporting. 'auto' uses "
            "fluid_pressure_dofs/pressure_active_dofs when present, otherwise the basis pressure DoFs."
        ),
    )
    return parser.parse_args()


def _normalize_index(index: int, size: int) -> int:
    return size + index if index < 0 else index


def _pressure_field(basis):
    for field in basis.fields:
        if field.name == "pressure":
            return field
    return None


def _pressure_error_dofs(data, basis, requested: str) -> np.ndarray | None:
    if requested != "auto":
        if requested not in data.files:
            raise KeyError(f"pressure error DoF key '{requested}' not found in snapshot bundle")
        return np.asarray(data[requested], dtype=np.int64)
    for key in ("fluid_pressure_dofs", "pressure_active_dofs"):
        if key in data.files:
            return np.asarray(data[key], dtype=np.int64)
    field = _pressure_field(basis)
    if field is not None and field.dofs is not None:
        return np.asarray(field.dofs, dtype=np.int64)
    if "pressure_dofs" in data.files:
        return np.asarray(data["pressure_dofs"], dtype=np.int64)
    return None


def _gauge_pressure_values(values: np.ndarray, gauge: str) -> np.ndarray:
    if gauge == "none":
        return values
    if gauge == "zero-mean":
        return values - values.mean(axis=0, keepdims=True)
    raise ValueError(f"Unknown pressure gauge {gauge!r}")


def _field_errors(
    data,
    basis,
    target_state,
    initial_state,
    projection_state,
    result_state,
    pressure_gauge: str,
    pressure_error_dof_key: str,
):
    errors = {}
    for name in ("displacement", "velocity", "pressure"):
        if name == "pressure":
            dofs = _pressure_error_dofs(data, basis, pressure_error_dof_key)
        else:
            key = f"{name}_dofs"
            dofs = np.asarray(data[key], dtype=np.int64) if key in data.files else None
        if dofs is None:
            continue
        target = target_state[dofs]
        initial = initial_state[dofs]
        projection = projection_state[dofs]
        result = result_state[dofs]
        if name == "pressure":
            target = _gauge_pressure_values(target, pressure_gauge)
            initial = _gauge_pressure_values(initial, pressure_gauge)
            projection = _gauge_pressure_values(projection, pressure_gauge)
            result = _gauge_pressure_values(result, pressure_gauge)
        errors[name] = {
            "num_dofs": int(dofs.size),
            "initial_relative_error": relative_l2_error(target, initial),
            "projection_relative_error": relative_l2_error(target, projection),
            "rom_relative_error": relative_l2_error(target, result),
        }
    return errors


def main() -> None:
    args = parse_args()
    basis = load_monolithic_pod_basis(Path(args.basis))
    data = np.load(args.snapshots)
    dvp = np.asarray(data["dvp"], dtype=float)
    times = np.asarray(data["time"], dtype=float) if "time" in data.files else None

    target_index = _normalize_index(args.target_index, dvp.shape[1])
    previous_index = (
        max(0, target_index - 1)
        if args.previous_index is None
        else _normalize_index(args.previous_index, dvp.shape[1])
    )
    previous_previous_index = max(0, previous_index - 1)
    initial_index = (
        previous_index
        if args.initial_index is None
        else _normalize_index(args.initial_index, dvp.shape[1])
    )

    overrides = replay_overrides(
        Path(args.snapshots),
        explicit_overrides=parse_problem_overrides(args.problem_override),
    )
    if args.dt is not None:
        overrides["dt"] = args.dt
    elif times is not None and target_index != previous_index:
        overrides["dt"] = float(times[target_index] - times[previous_index])
    if args.theta is not None:
        overrides["theta"] = args.theta

    problem = FEniCSMonolithicProblem.build(args.problem, overrides=overrides)
    problem.set_previous_previous_state(dvp[:, previous_previous_index])
    problem.set_previous_state(dvp[:, previous_index])
    if times is not None:
        problem.set_time(float(times[target_index]))

    mean = basis.pod_basis.mean
    raw_phi = basis.pod_basis.basis
    pressure_field = _pressure_field(basis)
    enforce_dirichlet = not args.no_enforce_dirichlet
    if enforce_dirichlet:
        phi = problem.constrain_increment_basis(raw_phi)
        free_dofs = problem.free_dofs()
    else:
        phi = raw_phi
        free_dofs = np.arange(problem.vector_size, dtype=np.int64)

    def project_state(state):
        values = np.asarray(state, dtype=float).reshape(-1, 1)
        if args.pressure_gauge != "none" and pressure_field is not None and pressure_field.dofs is not None:
            values = values.copy()
            dofs = pressure_field.dofs
            values[dofs, :] = _gauge_pressure_values(values[dofs, :], args.pressure_gauge)
        lhs = phi[free_dofs, :]
        rhs = values[free_dofs, :] - mean[free_dofs, :]
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    initial_coefficients = project_state(dvp[:, initial_index])

    def reconstruct(coefficients):
        state = mean + phi @ np.asarray(coefficients, dtype=float).reshape(-1, 1)
        if enforce_dirichlet:
            return problem.apply_state_bcs(state[:, 0]).reshape(-1, 1)
        return state

    def projected_residual(coefficients):
        state = reconstruct(coefficients)
        return problem.projected_residual(phi, state[:, 0])

    def projected_jacobian(coefficients):
        state = reconstruct(coefficients)
        return problem.projected_jacobian(phi, state[:, 0])

    target_state = dvp[:, target_index].reshape(-1, 1)
    if args.pressure_gauge != "none" and pressure_field is not None and pressure_field.dofs is not None:
        target_state = target_state.copy()
        target_state[pressure_field.dofs, :] = _gauge_pressure_values(
            target_state[pressure_field.dofs, :],
            args.pressure_gauge,
        )
    initial_state = reconstruct(initial_coefficients)
    target_coefficients = project_state(target_state)
    target_projection = reconstruct(target_coefficients)
    initial_projected_residual = projected_residual(initial_coefficients)
    target_projected_residual = projected_residual(target_coefficients)
    exact_target_projected_residual = problem.projected_residual(phi, target_state[:, 0])

    result = reduced_newton_solve_projected(
        initial_coefficients=initial_coefficients,
        projected_residual_callback=projected_residual,
        projected_jacobian_callback=projected_jacobian,
        reconstruct_callback=reconstruct,
        atol=args.atol,
        rtol=args.rtol,
        max_iterations=args.max_iterations,
        damping=args.damping,
        line_search=not args.no_line_search,
        min_damping=args.min_damping,
        jacobian_update_frequency=args.jacobian_update_frequency,
        reuse_jacobian_after_line_search_failure=args.retry_with_fresh_jacobian,
        linear_solver=args.linear_solver,
        linear_rcond=args.linear_rcond,
    )

    final_projected_residual = projected_residual(result.coefficients)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    block_coefficients = split_reduced_coefficients(result.coefficients, basis.fields)
    np.savez_compressed(
        output_dir / "fenics_pod_galerkin_step.npz",
        coefficients=result.coefficients,
        full_state=result.full_state,
        target_state=target_state,
        residual_norms=np.asarray(result.residual_norms, dtype=float),
        correction_norms=np.asarray(result.correction_norms, dtype=float),
        **{
            f"{name}_coefficients": values
            for name, values in block_coefficients.items()
        },
    )
    summary = {
        "problem": args.problem,
        "basis": str(args.basis),
        "snapshots": str(args.snapshots),
        "target_index": int(target_index),
        "previous_index": int(previous_index),
        "previous_previous_index": int(previous_previous_index),
        "initial_index": int(initial_index),
        "dt": float(overrides.get("dt", problem.namespace["dt"])),
        "num_full_dofs": int(problem.vector_size),
        "num_modes": int(basis.num_modes),
        "coefficient_blocks": {
            field.name: {
                "start": int(field.start),
                "stop": int(field.stop),
                "num_modes": int(field.num_modes),
                "num_state_dofs": int(field.dofs.size) if field.dofs is not None else None,
            }
            for field in basis.fields
        },
        "dirichlet_enforced": bool(enforce_dirichlet),
        "num_dirichlet_dofs": int(problem.dirichlet_dofs().size) if enforce_dirichlet else 0,
        "converged": bool(result.converged),
        "termination_reason": result.termination_reason,
        "iterations": int(result.iterations),
        "initial_residual_norm": float(result.residual_norms[0])
        if result.residual_norms
        else None,
        "final_residual_norm": float(result.residual_norms[-1])
        if result.residual_norms
        else None,
        "max_correction_norm_observed": max(result.correction_norms)
        if result.correction_norms
        else None,
        "line_search": not args.no_line_search,
        "min_damping": float(args.min_damping),
        "jacobian_update_frequency": int(args.jacobian_update_frequency),
        "retry_with_fresh_jacobian": bool(args.retry_with_fresh_jacobian),
        "linear_solver": args.linear_solver,
        "linear_rcond": args.linear_rcond,
        "pressure_gauge": args.pressure_gauge,
        "pressure_error_dof_key": args.pressure_error_dof_key,
        "jacobian_evaluations": int(result.jacobian_evaluations),
        "jacobian_update_iterations": list(result.jacobian_update_iterations),
        "jacobian_condition_numbers": list(result.jacobian_condition_numbers),
        "initial_projected_residual_norm": float(np.linalg.norm(initial_projected_residual)),
        "target_projected_residual_norm": float(np.linalg.norm(target_projected_residual)),
        "exact_target_projected_residual_norm": float(np.linalg.norm(exact_target_projected_residual)),
        "final_projected_residual_norm": float(np.linalg.norm(final_projected_residual)),
        "initial_relative_error_to_target_snapshot": relative_l2_error(target_state, initial_state),
        "target_projection_relative_error": relative_l2_error(target_state, target_projection),
        "relative_error_to_target_snapshot": relative_l2_error(target_state, result.full_state),
        "field_relative_errors": _field_errors(
            data=data,
            basis=basis,
            target_state=target_state,
            initial_state=initial_state,
            projection_state=target_projection,
            result_state=result.full_state,
            pressure_gauge=args.pressure_gauge,
            pressure_error_dof_key=args.pressure_error_dof_key,
        ),
    }
    with open(output_dir / "fenics_pod_galerkin_step_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote FEniCS POD-Galerkin step to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
