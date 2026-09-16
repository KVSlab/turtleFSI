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
    parser.add_argument(
        "--problem-override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Additional problem namespace overrides passed to the FEniCS problem builder.",
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

    # Build the reduced Eq. (4) system through named blocks first, then
    # recover the dense matrix used by the current online solver.
    linear_operator_blocks = _project_matrix_action_blocks(problem, J_ref, block_basis, rhs_N)
    J_N = linear_operator_blocks.as_operator()

    operators = ReducedLinearOperators(
        basis=block_basis,
        operator=J_N,
        rhs=rhs_N,
        full_operator_shape=(problem.vector_size, problem.vector_size),
    )
    singular_values = np.linalg.svd(J_N, compute_uv=False)
    smallest = singular_values[-1] if singular_values.size else 0.0
    condition_number = (
        float(singular_values[0] / smallest)
        if smallest > 0.0
        else float("inf")
    )
    metadata = {
        "problem": problem_name,
        "basis": str(basis_path),
        "snapshots": str(snapshots_path),
        "assembly_stage": "offline",
        "online_requires_fenics_assembly": False,
        "offline_assembled_quantities": [
            "R_N_ref",
            "J_N_blocks",
            "J_N",
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
        "R_N_ref_norm": float(np.linalg.norm(R_N_ref)),
        "J_N_condition_number": condition_number,
        "projected_residual_norm": float(np.linalg.norm(R_N_ref)),
        "reduced_operator_condition_number": condition_number,
        "num_reduced_dofs": int(phi.shape[1]),
        "num_full_dofs": int(problem.vector_size),
        "pressure_gauge": pressure_gauge,
    }
    save_reduced_operator_bundle(
        output,
        operators,
        projected_residual_at_reference=R_N_ref,
        projected_jacobian_mean_shift=J_N_mean_shift,
        linear_operator_blocks=linear_operator_blocks,
        metadata=metadata,
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
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
