"""Solve a saved Ballarin-style reduced operator bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod_galerkin.ballarin_monolithic import (
    block_basis_from_monolithic,
    reconstruct_ballarin_state,
    split_ballarin_state,
)
from turtleFSI.rom._pod_galerkin.monolithic_pod import load_monolithic_pod_basis
from turtleFSI.rom._pod_galerkin.offline_online import (
    load_reduced_operator_bundle,
    matrix_condition_number,
    solve_equilibrated_lstsq,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", required=True, help="monolithic_pod_basis.npz path.")
    parser.add_argument("--operators", required=True, help="Saved reduced operator bundle .npz.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--linear-solver",
        choices=("solve", "lstsq", "equilibrated-lstsq"),
        default="solve",
    )
    parser.add_argument("--linear-rcond", type=float, default=None)
    parser.add_argument("--equilibration-iterations", type=int, default=8)
    return parser.parse_args()


def _solve(args: argparse.Namespace, operator: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, dict]:
    diagnostics = {"raw_condition_number": matrix_condition_number(operator)}
    if args.linear_solver == "equilibrated-lstsq":
        result = solve_equilibrated_lstsq(
            operator,
            rhs,
            rcond=args.linear_rcond,
            iterations=args.equilibration_iterations,
        )
        diagnostics.update(
            {
                "scaled_condition_number": result.scaled_condition_number,
                "row_scaling_min": float(result.row_scaling.min()),
                "row_scaling_max": float(result.row_scaling.max()),
                "column_scaling_min": float(result.column_scaling.min()),
                "column_scaling_max": float(result.column_scaling.max()),
            }
        )
        return result.coefficients, diagnostics
    if args.linear_solver == "lstsq":
        return np.linalg.lstsq(operator, rhs, rcond=args.linear_rcond)[0], diagnostics
    return np.linalg.solve(operator, rhs), diagnostics


def main() -> None:
    args = parse_args()
    monolithic_basis = load_monolithic_pod_basis(Path(args.basis))
    block_basis = block_basis_from_monolithic(monolithic_basis)
    bundle = load_reduced_operator_bundle(Path(args.operators))
    coefficients, solve_diagnostics = _solve(args, bundle.operator, bundle.rhs)
    state = split_ballarin_state(coefficients, block_basis)
    full_state = reconstruct_ballarin_state(state, block_basis)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "ballarin_operator_solution.npz",
        coefficients=coefficients,
        full_state=full_state,
        displacement_coefficients=state.displacement,
        velocity_coefficients=state.velocity,
        pressure_coefficients=state.pressure,
        **(
            {"supremizer_coefficients": state.supremizer}
            if state.supremizer is not None
            else {}
        ),
    )
    summary = {
        "basis": str(args.basis),
        "operators": str(args.operators),
        "output_dir": str(output_dir),
        "block_mode_counts": bundle.block_mode_counts,
        "operator_metadata": bundle.metadata,
        "equation_6_matrix": "J_N",
        "equation_6_rhs": "rhs_N",
        "online_requires_fenics_assembly": (
            bool(bundle.metadata.get("online_requires_fenics_assembly"))
            if bundle.metadata is not None
            and "online_requires_fenics_assembly" in bundle.metadata
            else False
        ),
        "num_reduced_dofs": int(coefficients.shape[0]),
        "num_full_dofs": int(full_state.shape[0]),
        "residual_norm": float(np.linalg.norm(bundle.J_N @ coefficients - bundle.rhs_N)),
        "linear_solver": args.linear_solver,
        "linear_rcond": args.linear_rcond,
        "equilibration_iterations": args.equilibration_iterations,
        **solve_diagnostics,
    }
    with open(output_dir / "ballarin_operator_solution_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
