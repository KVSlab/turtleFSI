"""Run a multi-step FEniCS-projected monolithic POD-Galerkin rollout."""

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
    parser.add_argument("--snapshots", required=True, help="Exact dvp_snapshots.npz path for initial/reference data.")
    parser.add_argument("--start-index", type=int, default=0, help="Initial snapshot index.")
    parser.add_argument("--num-steps", type=int, required=True, help="Number of ROM time steps.")
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
        help="Relative singular-value cutoff used by least-squares correction solvers.",
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
    return parser.parse_args()


def _normalize_index(index: int, size: int) -> int:
    return size + index if index < 0 else index


def _pressure_field(basis):
    for field in basis.fields:
        if field.name == "pressure":
            return field
    return None


def _gauge_pressure_values(values: np.ndarray, gauge: str) -> np.ndarray:
    if gauge == "none":
        return values
    if gauge == "zero-mean":
        return values - values.mean(axis=0, keepdims=True)
    raise ValueError(f"Unknown pressure gauge {gauge!r}")


def _apply_pressure_gauge_to_state(state: np.ndarray, basis, gauge: str) -> np.ndarray:
    values = np.asarray(state, dtype=float).reshape(-1, 1).copy()
    pressure_field = _pressure_field(basis)
    if gauge != "none" and pressure_field is not None and pressure_field.dofs is not None:
        dofs = pressure_field.dofs
        values[dofs, :] = _gauge_pressure_values(values[dofs, :], gauge)
    return values


def main() -> None:
    args = parse_args()
    basis = load_monolithic_pod_basis(Path(args.basis))
    data = np.load(args.snapshots)
    dvp = np.asarray(data["dvp"], dtype=float)
    times = np.asarray(data["time"], dtype=float) if "time" in data.files else None

    start_index = _normalize_index(args.start_index, dvp.shape[1])
    if start_index < 0 or start_index >= dvp.shape[1]:
        raise IndexError(f"start_index {start_index} is outside snapshot range")
    if args.num_steps < 1:
        raise ValueError("--num-steps must be positive")

    if args.dt is not None:
        dt = args.dt
    elif times is not None and start_index + 1 < times.size:
        dt = float(times[start_index + 1] - times[start_index])
    else:
        raise ValueError("Provide --dt when snapshot times do not define a next time step")

    overrides = replay_overrides(
        Path(args.snapshots),
        explicit_overrides=parse_problem_overrides(args.problem_override),
    )
    overrides["dt"] = dt
    if args.theta is not None:
        overrides["theta"] = args.theta

    problem = FEniCSMonolithicProblem.build(args.problem, overrides=overrides)
    mean = basis.pod_basis.mean
    raw_phi = basis.pod_basis.basis
    enforce_dirichlet = not args.no_enforce_dirichlet
    if enforce_dirichlet:
        phi = problem.constrain_increment_basis(raw_phi)
        free_dofs = problem.free_dofs()
    else:
        phi = raw_phi
        free_dofs = np.arange(problem.vector_size, dtype=np.int64)

    def project_state(state):
        values = _apply_pressure_gauge_to_state(state, basis, args.pressure_gauge)
        lhs = phi[free_dofs, :]
        rhs = values[free_dofs, :] - mean[free_dofs, :]
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    previous_previous_full = _apply_pressure_gauge_to_state(
        dvp[:, max(0, start_index - 1)],
        basis,
        args.pressure_gauge,
    )
    current_full = _apply_pressure_gauge_to_state(
        dvp[:, start_index],
        basis,
        args.pressure_gauge,
    )
    current_coefficients = project_state(current_full)

    full_history = [current_full[:, 0].copy()]
    coefficient_history = [current_coefficients[:, 0].copy()]
    step_summaries = []

    for step in range(args.num_steps):
        problem.set_previous_previous_state(previous_previous_full[:, 0])
        problem.set_previous_state(current_full[:, 0])
        if times is not None and start_index + step + 1 < times.shape[0]:
            problem.set_time(float(times[start_index + step + 1]))

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

        result = reduced_newton_solve_projected(
            initial_coefficients=current_coefficients,
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

        previous_previous_full = current_full
        current_coefficients = result.coefficients
        current_full = result.full_state
        coefficient_history.append(current_coefficients[:, 0].copy())
        full_history.append(current_full[:, 0].copy())
        step_summary = {
            "step": int(step + 1),
            "converged": bool(result.converged),
            "termination_reason": result.termination_reason,
            "iterations": int(result.iterations),
            "initial_residual_norm": (
                float(result.residual_norms[0]) if result.residual_norms else None
            ),
            "final_residual_norm": (
                float(result.residual_norms[-1]) if result.residual_norms else None
            ),
            "max_correction_norm_observed": (
                max(result.correction_norms) if result.correction_norms else None
            ),
            "jacobian_evaluations": int(result.jacobian_evaluations),
            "jacobian_update_iterations": list(result.jacobian_update_iterations),
        }
        step_summaries.append(step_summary)

    full = np.asarray(full_history, dtype=float).T
    coefficients = np.asarray(coefficient_history, dtype=float).T
    block_coefficients = split_reduced_coefficients(coefficients, basis.fields)

    reference_error = None
    reference_end_index = start_index + full.shape[1]
    reference = dvp[:, start_index: min(reference_end_index, dvp.shape[1])].copy()
    if args.pressure_gauge != "none":
        pressure_field = _pressure_field(basis)
        if pressure_field is not None and pressure_field.dofs is not None:
            dofs = pressure_field.dofs
            reference[dofs, :] = _gauge_pressure_values(
                reference[dofs, :],
                args.pressure_gauge,
            )
    if reference_end_index <= dvp.shape[1]:
        reference_error = relative_l2_error(reference, full)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "fenics_pod_galerkin_rollout.npz",
        coefficients=coefficients,
        full=full,
        reference=reference,
        time=times[start_index: min(reference_end_index, times.shape[0])]
        if times is not None
        else np.arange(full.shape[1], dtype=float) * dt,
        **{
            f"{name}_coefficients": values
            for name, values in block_coefficients.items()
        },
    )

    summary = {
        "problem": args.problem,
        "basis": str(args.basis),
        "snapshots": str(args.snapshots),
        "start_index": int(start_index),
        "num_steps": int(args.num_steps),
        "dt": float(dt),
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
        "all_steps_converged": bool(all(step["converged"] for step in step_summaries)),
        "line_search": not args.no_line_search,
        "min_damping": float(args.min_damping),
        "jacobian_update_frequency": int(args.jacobian_update_frequency),
        "retry_with_fresh_jacobian": bool(args.retry_with_fresh_jacobian),
        "linear_solver": args.linear_solver,
        "linear_rcond": args.linear_rcond,
        "pressure_gauge": args.pressure_gauge,
        "total_jacobian_evaluations": int(
            sum(step["jacobian_evaluations"] for step in step_summaries)
        ),
        "reference_relative_error": reference_error,
        "steps": step_summaries,
    }
    with open(output_dir / "fenics_pod_galerkin_rollout_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote FEniCS POD-Galerkin rollout to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
