"""Build a reduced Ballarin-style operator bundle from turtleFSI FEniCS forms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from turtleFSI.rom._pod_galerkin.ballarin_monolithic import block_basis_from_monolithic
from turtleFSI.rom._pod_galerkin.monolithic_pod import load_monolithic_pod_basis
from turtleFSI.rom._pod_galerkin.offline_online import (
    ReducedLinearOperatorBlocks,
    ReducedLinearOperators,
    equilibrate_linear_system,
    matrix_condition_number,
    rhs_N_from_equation6_linearization,
    save_reduced_operator_bundle,
)
from turtleFSI.rom._pod_galerkin.problem_overrides import (
    parse_problem_overrides,
    replay_overrides,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, help="turtleFSI problem name.")
    parser.add_argument("--basis", required=True, help="monolithic_pod_basis.npz path.")
    parser.add_argument("--snapshots", required=True, help="dvp_snapshots.npz path.")
    parser.add_argument("--output", required=True, help="Output reduced_operators.npz path.")
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--equation-6-blocks", action="store_true",
                        help="Project and save each equation (4) operator; verify and use their equation (6) sum.")
    parser.add_argument(
        "--problem-override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Additional problem namespace overrides passed to the FEniCS problem builder.",
    )
    parser.add_argument(
        "--paper-forms",
        action="store_true",
        help="Override compatible options for Ballarin/Rozza Section 2.3/Table I forms.",
    )
    return parser.parse_args()


def _normalize_index(index: int, size: int) -> int:
    return size + index if index < 0 else index


def _project_matrix_action(problem, matrix, phi: np.ndarray, vector: np.ndarray) -> np.ndarray:
    product = problem._matrix_vector_product(matrix, vector)
    return phi.T @ product.reshape(-1, 1)




def _named_basis_blocks(block_basis) -> dict[str, np.ndarray]:
    blocks = {
        "displacement": block_basis.displacement,
        "velocity": block_basis.velocity,
        "pressure": block_basis.pressure,
    }
    if block_basis.supremizer is not None:
        blocks["supremizer"] = block_basis.supremizer
    return {
        name: blocks[name]
        for name in ("displacement", "velocity", "supremizer", "pressure")
        if name in blocks and blocks[name].shape[1] > 0
    }


def _split_reduced_rhs_by_blocks(rhs: np.ndarray, block_mode_counts: dict[str, int]) -> dict[str, np.ndarray]:
    vector = np.asarray(rhs, dtype=float).reshape(-1, 1)
    blocks = {}
    cursor = 0
    for name, count in block_mode_counts.items():
        blocks[name] = vector[cursor : cursor + count, :]
        cursor += count
    if cursor != vector.shape[0]:
        raise ValueError("rhs length does not match explicit block sizes")
    return blocks


def _project_matrix_action_blocks(problem, matrix, block_basis, rhs_N: np.ndarray) -> ReducedLinearOperatorBlocks:
    named_basis = _named_basis_blocks(block_basis)
    block_names = tuple(named_basis.keys())
    blocks = {}
    for row_name, row_basis in named_basis.items():
        for column_name, column_basis in named_basis.items():
            columns = []
            for column in range(column_basis.shape[1]):
                columns.append(
                    _project_matrix_action(problem, matrix, row_basis, column_basis[:, column])[:, 0]
                )
            blocks[(row_name, column_name)] = np.asarray(columns, dtype=float).T
    return ReducedLinearOperatorBlocks(
        block_names=block_names,
        blocks=blocks,
        rhs_blocks=_split_reduced_rhs_by_blocks(
            rhs_N,
            {name: named_basis[name].shape[1] for name in block_names},
        ),
    )


def _operator_diagnostics(operator: np.ndarray, blocks: ReducedLinearOperatorBlocks) -> dict[str, Any]:
    matrix = np.asarray(operator, dtype=float)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    condition_number = matrix_condition_number(matrix)
    row_norms = np.linalg.norm(matrix, axis=1)
    column_norms = np.linalg.norm(matrix, axis=0)
    tiny = 1e-14
    try:
        equilibrated, _, _, _ = equilibrate_linear_system(matrix, np.zeros((matrix.shape[0], 1)), iterations=8)
        equilibrated_condition_number = matrix_condition_number(equilibrated)
    except ValueError:
        equilibrated_condition_number = condition_number
    return {
        "condition_number": float(condition_number),
        "equilibrated_condition_number": float(equilibrated_condition_number),
        "singular_values": [float(value) for value in singular_values],
        "smallest_singular_value": float(singular_values[-1]) if singular_values.size else 0.0,
        "largest_singular_value": float(singular_values[0]) if singular_values.size else 0.0,
        "row_norms": [float(value) for value in row_norms],
        "column_norms": [float(value) for value in column_norms],
        "zero_row_indices": [int(index) for index in np.flatnonzero(row_norms <= tiny)],
        "zero_column_indices": [int(index) for index in np.flatnonzero(column_norms <= tiny)],
        "block_norms": {
            f"{row}__{column}": float(np.linalg.norm(blocks.blocks[(row, column)]))
            for row in blocks.block_names
            for column in blocks.block_names
        },
    }


def _component_projection_is_usable(
    component_operator: np.ndarray,
    aggregate_operator: np.ndarray,
    diagnostics: dict[str, Any],
    relative_difference: float,
    tolerance: float = 1e-4,
) -> bool:
    if diagnostics["zero_row_indices"] or diagnostics["zero_column_indices"]:
        return False
    if not np.isfinite(diagnostics["condition_number"]):
        return False
    if relative_difference > tolerance:
        return False
    return True


def _pressure_gauge_from_basis_summary(basis_path: Path) -> str:
    summary_path = basis_path.with_name("monolithic_pod_basis_summary.json")
    if not summary_path.exists():
        return "none"
    with open(summary_path, encoding="ascii") as handle:
        summary = json.load(handle)
    return str(summary.get("pressure_gauge", "none"))


def _apply_pressure_gauge_to_state(
    state: np.ndarray,
    monolithic_basis,
    pressure_gauge: str,
) -> np.ndarray:
    adjusted = np.asarray(state, dtype=float).reshape(-1, 1).copy()
    if pressure_gauge == "none":
        return adjusted
    if pressure_gauge != "zero-mean":
        raise ValueError(f"Unknown pressure gauge {pressure_gauge!r}")
    pressure_field = next(
        (field for field in monolithic_basis.fields if field.name == "pressure"),
        None,
    )
    if pressure_field is not None and pressure_field.dofs is not None:
        dofs = pressure_field.dofs
        adjusted[dofs, :] = adjusted[dofs, :] - adjusted[dofs, :].mean(
            axis=0,
            keepdims=True,
        )
    return adjusted


def build_reduced_operator_bundle(
    problem,
    monolithic_basis,
    dvp: np.ndarray,
    times: np.ndarray | None,
    reference_index: int,
    output: Path,
    problem_name: str,
    basis_path: Path,
    snapshots_path: Path,
    theta: float | None = None,
    equation_6_blocks: bool = False,
) -> dict[str, Any]:
    """Build and save one reduced operator bundle from an initialized problem."""

    block_basis = block_basis_from_monolithic(monolithic_basis)
    phi = block_basis.combined_basis
    mean = block_basis.mean
    pressure_gauge = _pressure_gauge_from_basis_summary(basis_path)

    reference_index = _normalize_index(reference_index, dvp.shape[1])
    if reference_index < 0 or reference_index >= dvp.shape[1]:
        raise IndexError(f"reference-index {reference_index} is outside snapshot range")

    previous_index = max(0, reference_index - 1)
    previous_previous_index = max(0, previous_index - 1)
    reference_state = _apply_pressure_gauge_to_state(
        dvp[:, reference_index],
        monolithic_basis,
        pressure_gauge,
    )
    previous_previous_state = _apply_pressure_gauge_to_state(
        dvp[:, previous_previous_index],
        monolithic_basis,
        pressure_gauge,
    )
    previous_state = _apply_pressure_gauge_to_state(
        dvp[:, previous_index],
        monolithic_basis,
        pressure_gauge,
    )
    problem.set_previous_previous_state(previous_previous_state[:, 0])
    problem.set_previous_state(previous_state[:, 0])
    if times is not None:
        problem.set_time(float(times[reference_index]))
    problem.set_state(reference_state[:, 0])

    residual = problem.residual(reference_state[:, 0]).reshape(-1, 1)
    R_N_ref = phi.T @ residual
    J_ref = problem.assemble_jacobian()
    mean_shift = mean[:, 0] - reference_state[:, 0]
    J_N_mean_shift = _project_matrix_action(problem, J_ref, phi, mean_shift)
    rhs_N = rhs_N_from_equation6_linearization(R_N_ref, J_N_mean_shift)

    # Build the reduced Eq. (3)/(4) system by assembling and projecting the
    # named Table I component blocks. Also build the aggregate projection of the
    # full FEniCS Jacobian, which is the safe Eq. (6) fallback whenever the
    # visible component path is incomplete for the selected problem setup.
    equation_3_projection = problem.projected_equation_3_component_blocks(
        block_basis,
        reference_state[:, 0],
    )
    component_linear_operator_blocks = ReducedLinearOperatorBlocks(
        block_names=equation_3_projection.block_names,
        blocks=equation_3_projection.blocks,
        rhs_blocks=_split_reduced_rhs_by_blocks(
            rhs_N,
            {
                name: equation_3_projection.blocks[(name, name)].shape[0]
                for name in equation_3_projection.block_names
            },
        ),
    )
    component_J_N = component_linear_operator_blocks.as_operator()
    aggregate_linear_operator_blocks = _project_matrix_action_blocks(problem, J_ref, block_basis, rhs_N)
    aggregate_J_N = aggregate_linear_operator_blocks.as_operator()
    equation_3_difference = component_J_N - aggregate_J_N
    aggregate_norm = np.linalg.norm(aggregate_J_N)
    equation_3_relative_difference = (
        float(np.linalg.norm(equation_3_difference) / aggregate_norm)
        if aggregate_norm > 0.0
        else (0.0 if np.linalg.norm(equation_3_difference) == 0.0 else float("inf"))
    )
    component_diagnostics = _operator_diagnostics(
        component_J_N,
        component_linear_operator_blocks,
    )
    aggregate_diagnostics = _operator_diagnostics(
        aggregate_J_N,
        aggregate_linear_operator_blocks,
    )
    use_component_operator = _component_projection_is_usable(
        component_J_N,
        aggregate_J_N,
        component_diagnostics,
        equation_3_relative_difference,
    )
    if use_component_operator:
        linear_operator_blocks = component_linear_operator_blocks
        J_N = component_J_N
        operator_source = "equation_3_component_blocks"
        operator_diagnostics = component_diagnostics
        operator_selection_reason = "component projection has no zero rows/columns and matches aggregate projection"
    else:
        linear_operator_blocks = aggregate_linear_operator_blocks
        J_N = aggregate_J_N
        operator_source = "aggregate_projected_jacobian"
        operator_diagnostics = aggregate_diagnostics
        operator_selection_reason = "component projection is incomplete or disagrees with aggregate projection"

    named_equation_6 = None
    if equation_6_blocks:
        named_equation_6 = problem.projected_equation_6(
            block_basis, reference_state[:, 0], constrain_basis=False,
        )
        J_N = named_equation_6["J_N"]
        names = tuple(linear_operator_blocks.block_names)
        counts = linear_operator_blocks.block_mode_counts
        offsets = np.cumsum([0] + [counts[name] for name in names])
        slices = {name: slice(offsets[i], offsets[i + 1]) for i, name in enumerate(names)}
        linear_operator_blocks = ReducedLinearOperatorBlocks(
            block_names=names,
            blocks={(row, col): J_N[slices[row], slices[col]] for row in names for col in names},
            rhs_blocks=linear_operator_blocks.rhs_blocks,
        )
        operator_source = "equation_6_named_operators"
        operator_selection_reason = "Named equation (4) projections match full Jacobian; boundary correction added separately"
        use_component_operator = True
        operator_diagnostics = _operator_diagnostics(J_N, linear_operator_blocks)

    operators = ReducedLinearOperators(
        basis=block_basis,
        operator=J_N,
        rhs=rhs_N,
        full_operator_shape=(problem.vector_size, problem.vector_size),
    )
    condition_number = operator_diagnostics["condition_number"]
    metadata = {
        "problem": problem_name,
        "basis": str(basis_path),
        "basis_summary": str(basis_path.with_name("monolithic_pod_basis_summary.json")),
        "snapshots": str(snapshots_path),
        "assembly_stage": "offline",
        "online_requires_fenics_assembly": False,
        "offline_assembled_quantities": [
            "R_N_ref",
            "equation_3_J_N_blocks",
            "equation_3_J_N",
            "aggregate_J_N_diagnostic",
            "J_N_mean_shift",
            "rhs_N_blocks",
            "rhs_N",
        ],
        "explicit_block_names": list(linear_operator_blocks.block_names),
        "reference_index": int(reference_index),
        "previous_index": int(previous_index),
        "previous_previous_index": int(previous_previous_index),
        "dt": float(problem.namespace["dt"]),
        "theta": theta,
        "linearization": "R(y_ref) + J(y_ref) (y - y_ref)",
        "operator_source": operator_source,
        "operator_selection_reason": operator_selection_reason,
        "used_component_operator": bool(use_component_operator),
        "equation_3_component_count": len(equation_3_projection.component_blocks),
        "equation_3_component_block_count": sum(
            len(blocks) for blocks in equation_3_projection.component_blocks.values()
        ),
        "equation_3_vs_aggregate_difference_norm": float(np.linalg.norm(equation_3_difference)),
        "equation_3_vs_aggregate_relative_difference_norm": equation_3_relative_difference,
        "component_operator_diagnostics": component_diagnostics,
        "aggregate_operator_diagnostics": aggregate_diagnostics,
        "selected_operator_diagnostics": operator_diagnostics,
        "R_N_ref_norm": float(np.linalg.norm(R_N_ref)),
        "J_N_condition_number": condition_number,
        "projected_residual_norm": float(np.linalg.norm(R_N_ref)),
        "reduced_operator_condition_number": condition_number,
        "reduced_operator_equilibrated_condition_number": operator_diagnostics["equilibrated_condition_number"],
        "J_N_singular_values": operator_diagnostics["singular_values"],
        "J_N_zero_row_indices": operator_diagnostics["zero_row_indices"],
        "J_N_zero_column_indices": operator_diagnostics["zero_column_indices"],
        "num_reduced_dofs": int(phi.shape[1]),
        "num_full_dofs": int(problem.vector_size),
        "pressure_gauge": pressure_gauge,
    }
    if named_equation_6 is not None:
        metadata["equation_6_operator_names"] = list(named_equation_6)
        metadata["equation_6_physical_difference_norm"] = float(np.linalg.norm(named_equation_6["equation_6_difference_N"]))
        metadata["equation_6_boundary_correction_norm"] = float(np.linalg.norm(named_equation_6["equation_6_boundary_correction_N"]))
    save_reduced_operator_bundle(
        output,
        operators,
        projected_residual_at_reference=R_N_ref,
        projected_jacobian_mean_shift=J_N_mean_shift,
        linear_operator_blocks=linear_operator_blocks,
        component_operator_blocks=equation_3_projection.component_blocks,
        metadata=metadata,
        named_equation_6_operators=named_equation_6,
    )
    summary_path = output.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="ascii") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata


def main() -> None:
    args = parse_args()
    # Import lazily so non-FEniCS unit tests can import the package.
    from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem

    basis_path = Path(args.basis)
    snapshots_path = Path(args.snapshots)
    monolithic_basis = load_monolithic_pod_basis(basis_path)
    data = np.load(snapshots_path)
    dvp = np.asarray(data["dvp"], dtype=float)
    times = np.asarray(data["time"], dtype=float) if "time" in data.files else None
    reference_index = _normalize_index(args.reference_index, dvp.shape[1])

    if args.dt is not None:
        dt = args.dt
    elif times is not None and reference_index > 0:
        dt = float(times[reference_index] - times[reference_index - 1])
    elif times is not None and times.shape[0] > 1:
        dt = float(times[1] - times[0])
    else:
        raise ValueError("Provide --dt when snapshot times cannot define a time step")

    overrides = replay_overrides(
        snapshots_path,
        explicit_overrides=parse_problem_overrides(args.problem_override),
    )
    overrides["dt"] = dt
    if args.theta is not None:
        overrides["theta"] = args.theta
    problem = FEniCSMonolithicProblem.build(args.problem, overrides=overrides)

    metadata = build_reduced_operator_bundle(
        problem=problem,
        monolithic_basis=monolithic_basis,
        dvp=dvp,
        times=times,
        reference_index=reference_index,
        output=Path(args.output),
        problem_name=args.problem,
        basis_path=basis_path,
        snapshots_path=snapshots_path,
        theta=args.theta,
        equation_6_blocks=args.equation_6_blocks,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
