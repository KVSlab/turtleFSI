"""Solve a saved Ballarin-style reduced operator series."""

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
from turtleFSI.rom._shared.pod import relative_l2_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", required=True, help="monolithic_pod_basis.npz path.")
    parser.add_argument("--manifest", required=True, help="operator_series_manifest.json path.")
    parser.add_argument("--snapshots", default=None, help="Optional dvp_snapshots.npz for validation.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--linear-solver",
        choices=("solve", "lstsq", "equilibrated-lstsq"),
        default="solve",
    )
    parser.add_argument("--linear-rcond", type=float, default=None)
    parser.add_argument("--equilibration-iterations", type=int, default=8)
    return parser.parse_args()


def _solve(
    operator: np.ndarray,
    rhs: np.ndarray,
    solver: str,
    rcond: float | None,
    equilibration_iterations: int,
) -> tuple[np.ndarray, dict[str, float]]:
    diagnostics = {
        "raw_condition_number": matrix_condition_number(operator),
    }
    if solver == "equilibrated-lstsq":
        result = solve_equilibrated_lstsq(
            operator,
            rhs,
            rcond=rcond,
            iterations=equilibration_iterations,
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
    if solver == "lstsq":
        return np.linalg.lstsq(operator, rhs, rcond=rcond)[0], diagnostics
    return np.linalg.solve(operator, rhs), diagnostics


def _field_errors(full_state: np.ndarray, target: np.ndarray, fields) -> dict[str, float]:
    errors = {}
    for field in fields:
        if field.name == "supremizer":
            continue
        dofs = field.dofs
        if dofs is None:
            continue
        errors[field.name] = relative_l2_error(target[dofs, :], full_state[dofs, :])
    return errors


def _pressure_gauge_from_basis_summary(basis_path: Path) -> str:
    summary_path = basis_path.with_name("monolithic_pod_basis_summary.json")
    if not summary_path.exists():
        return "none"
    with open(summary_path, encoding="ascii") as handle:
        summary = json.load(handle)
    return str(summary.get("pressure_gauge", "none"))


def _validation_target(
    target: np.ndarray,
    fields,
    pressure_gauge: str,
) -> np.ndarray:
    adjusted = np.asarray(target, dtype=float).copy()
    if pressure_gauge == "none":
        return adjusted
    if pressure_gauge != "zero-mean":
        raise ValueError(f"Unknown pressure gauge {pressure_gauge!r}")
    pressure_field = next((field for field in fields if field.name == "pressure"), None)
    if pressure_field is not None and pressure_field.dofs is not None:
        dofs = pressure_field.dofs
        adjusted[dofs, :] = adjusted[dofs, :] - adjusted[dofs, :].mean(
            axis=0,
            keepdims=True,
        )
    return adjusted


def main() -> None:
    args = parse_args()
    basis_path = Path(args.basis)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    monolithic_basis = load_monolithic_pod_basis(basis_path)
    block_basis = block_basis_from_monolithic(monolithic_basis)
    pressure_gauge = _pressure_gauge_from_basis_summary(basis_path)
    with open(manifest_path, encoding="ascii") as handle:
        manifest = json.load(handle)

    validation = None
    if args.snapshots is not None:
        validation = np.load(args.snapshots)
        dvp = np.asarray(validation["dvp"], dtype=float)
    else:
        dvp = None

    full_states = []
    coefficients = []
    summaries = []
    for item in manifest["bundles"]:
        bundle = load_reduced_operator_bundle(Path(item["path"]))
        coeff, solve_diagnostics = _solve(
            bundle.operator,
            bundle.rhs,
            args.linear_solver,
            args.linear_rcond,
            args.equilibration_iterations,
        )
        state = split_ballarin_state(coeff, block_basis)
        full_state = reconstruct_ballarin_state(state, block_basis)
        reference_index = int(item["reference_index"])
        summary = {
            "reference_index": reference_index,
            "operator": item["path"],
            "operator_metadata": bundle.metadata,
            "equation_6_matrix": "J_N",
            "equation_6_rhs": "rhs_N",
            "online_requires_fenics_assembly": (
                bool(bundle.metadata.get("online_requires_fenics_assembly"))
                if bundle.metadata is not None
                and "online_requires_fenics_assembly" in bundle.metadata
                else False
            ),
            "residual_norm": float(np.linalg.norm(bundle.J_N @ coeff - bundle.rhs_N)),
            "reduced_operator_condition_number": item.get("reduced_operator_condition_number"),
            **solve_diagnostics,
        }
        if dvp is not None and 0 <= reference_index < dvp.shape[1]:
            target = dvp[:, reference_index].reshape(-1, 1)
            target = _validation_target(
                target=target,
                fields=monolithic_basis.fields,
                pressure_gauge=pressure_gauge,
            )
            validation_full_state = _validation_target(
                target=full_state,
                fields=monolithic_basis.fields,
                pressure_gauge=pressure_gauge,
            )
            summary["relative_error_to_reference_snapshot"] = relative_l2_error(
                target,
                validation_full_state,
            )
            summary["field_relative_errors"] = _field_errors(
                full_state=validation_full_state,
                target=target,
                fields=monolithic_basis.fields,
            )
        coefficients.append(coeff[:, 0].copy())
        full_states.append(full_state[:, 0].copy())
        summaries.append(summary)

    coefficient_array = np.asarray(coefficients, dtype=float).T
    full_array = np.asarray(full_states, dtype=float).T
    np.savez_compressed(
        output_dir / "ballarin_operator_series_solution.npz",
        coefficients=coefficient_array,
        full_states=full_array,
        reference_indices=np.asarray(
            [entry["reference_index"] for entry in summaries],
            dtype=np.int64,
        ),
    )
    result_summary = {
        "basis": str(basis_path),
        "manifest": str(manifest_path),
        "snapshots": args.snapshots,
        "output_dir": str(output_dir),
        "linear_solver": args.linear_solver,
        "linear_rcond": args.linear_rcond,
        "equilibration_iterations": args.equilibration_iterations,
        "pressure_gauge": pressure_gauge,
        "num_solutions": int(len(summaries)),
        "solutions": summaries,
    }
    with open(output_dir / "ballarin_operator_series_solution_summary.json", "w", encoding="ascii") as handle:
        json.dump(result_summary, handle, indent=2, sort_keys=True)
    print(json.dumps(result_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
