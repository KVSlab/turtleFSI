"""Offline-online reduced operators for monolithic POD-Galerkin ROMs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from turtleFSI.rom._pod_galerkin.ballarin_monolithic import (
    BallarinBlockBasis,
    reconstruct_ballarin_state,
    split_ballarin_state,
)


@dataclass(frozen=True)
class ReducedLinearOperators:
    """Reduced affine residual pieces ``A_N a - f_N``."""

    basis: BallarinBlockBasis
    operator: np.ndarray
    rhs: np.ndarray
    full_operator_shape: tuple[int, int]

    @property
    def J_N(self) -> np.ndarray:
        """Paper-style alias for the reduced Eq. (6) Jacobian/operator."""

        return self.operator

    @property
    def rhs_N(self) -> np.ndarray:
        """Paper-style alias for the reduced Eq. (6) right-hand side."""

        return self.rhs


@dataclass(frozen=True)
class ReducedOperatorBundle:
    """Saved online-ready reduced operators."""

    operator: np.ndarray
    rhs: np.ndarray
    block_mode_counts: dict[str, int]
    full_operator_shape: tuple[int, int] | None = None
    quadratic_tensor: np.ndarray | None = None
    projected_residual_at_reference: np.ndarray | None = None
    projected_jacobian_mean_shift: np.ndarray | None = None
    linear_operator_blocks: ReducedLinearOperatorBlocks | None = None
    metadata: dict[str, Any] | None = None

    @property
    def J_N(self) -> np.ndarray:
        """Paper-style alias for the reduced Eq. (6) Jacobian/operator."""

        return self.operator

    @property
    def rhs_N(self) -> np.ndarray:
        """Paper-style alias for the reduced Eq. (6) right-hand side."""

        return self.rhs

    @property
    def R_N_ref(self) -> np.ndarray | None:
        """Paper-style alias for the projected residual at the reference state."""

        return self.projected_residual_at_reference

    @property
    def J_N_mean_shift(self) -> np.ndarray | None:
        """Paper-style alias for ``J_N`` applied to the mean/reference shift."""

        return self.projected_jacobian_mean_shift




@dataclass(frozen=True)
class ReducedLinearOperatorBlocks:
    """Named reduced blocks for the linearized Ballarin Eq. (4) system.

    ``blocks[(row, column)]`` stores ``Phi_row.T A Phi_column`` and
    ``rhs_blocks[row]`` stores that row block of the reduced right-hand side.
    ``block_names`` defines the row/column ordering used to recover the dense
    monolithic reduced system.
    """

    block_names: tuple[str, ...]
    blocks: dict[tuple[str, str], np.ndarray]
    rhs_blocks: dict[str, np.ndarray]

    @property
    def block_mode_counts(self) -> dict[str, int]:
        return {
            name: int(self.rhs_blocks[name].shape[0])
            for name in self.block_names
        }

    def as_operator(self) -> np.ndarray:
        rows = []
        for row_name in self.block_names:
            rows.append(
                np.hstack([self.blocks[(row_name, column_name)] for column_name in self.block_names])
            )
        return np.vstack(rows)

    def as_rhs(self) -> np.ndarray:
        return np.vstack([self.rhs_blocks[name] for name in self.block_names])

@dataclass(frozen=True)
class EquilibratedLinearSolve:
    """Solution and diagnostics for row/column equilibrated reduced solves."""

    coefficients: np.ndarray
    scaled_operator: np.ndarray
    scaled_rhs: np.ndarray
    row_scaling: np.ndarray
    column_scaling: np.ndarray
    scaled_condition_number: float


def matrix_condition_number(operator: np.ndarray) -> float:
    """Return the dense 2-norm condition number from singular values."""

    values = np.linalg.svd(np.asarray(operator, dtype=float), compute_uv=False)
    if values.size == 0:
        return 0.0
    smallest = values[-1]
    if smallest <= 0.0:
        return float("inf")
    return float(values[0] / smallest)


def equilibrate_linear_system(
    operator: np.ndarray,
    rhs: np.ndarray,
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale rows and columns to reduce reduced-system magnitude imbalance.

    Returns ``A_s, b_s, row_scaling, column_scaling`` where
    ``A_s = D_r A D_c`` and ``b_s = D_r b``.  If ``A_s z = b_s`` is solved,
    the original coefficients are ``a = D_c z``.
    """

    if iterations < 1:
        raise ValueError(f"iterations must be positive, got {iterations}")
    scaled_operator = np.asarray(operator, dtype=float).copy()
    scaled_rhs = np.asarray(rhs, dtype=float).reshape(-1, 1).copy()
    if scaled_operator.ndim != 2 or scaled_operator.shape[0] != scaled_operator.shape[1]:
        raise ValueError(f"operator must be square, got {scaled_operator.shape}")
    if scaled_rhs.shape != (scaled_operator.shape[0], 1):
        raise ValueError(
            f"rhs shape {scaled_rhs.shape} does not match operator {scaled_operator.shape}"
        )

    row_scaling = np.ones(scaled_operator.shape[0], dtype=float)
    column_scaling = np.ones(scaled_operator.shape[1], dtype=float)
    tiny = np.finfo(float).tiny
    for _ in range(iterations):
        row_norms = np.linalg.norm(scaled_operator, axis=1)
        row_update = np.where(row_norms > tiny, 1.0 / row_norms, 1.0)
        scaled_operator = row_update[:, None] * scaled_operator
        scaled_rhs = row_update[:, None] * scaled_rhs
        row_scaling *= row_update

        column_norms = np.linalg.norm(scaled_operator, axis=0)
        column_update = np.where(column_norms > tiny, 1.0 / column_norms, 1.0)
        scaled_operator = scaled_operator * column_update[None, :]
        column_scaling *= column_update

    return scaled_operator, scaled_rhs, row_scaling, column_scaling


def solve_equilibrated_lstsq(
    operator: np.ndarray,
    rhs: np.ndarray,
    rcond: float | None = None,
    iterations: int = 8,
) -> EquilibratedLinearSolve:
    """Solve ``A a = b`` through row/column equilibrated least squares."""

    scaled_operator, scaled_rhs, row_scaling, column_scaling = equilibrate_linear_system(
        operator,
        rhs,
        iterations=iterations,
    )
    scaled_coefficients = np.linalg.lstsq(scaled_operator, scaled_rhs, rcond=rcond)[0]
    coefficients = column_scaling[:, None] * scaled_coefficients
    return EquilibratedLinearSolve(
        coefficients=coefficients,
        scaled_operator=scaled_operator,
        scaled_rhs=scaled_rhs,
        row_scaling=row_scaling,
        column_scaling=column_scaling,
        scaled_condition_number=matrix_condition_number(scaled_operator),
    )




def _ballarin_named_basis_blocks(basis: BallarinBlockBasis) -> dict[str, np.ndarray]:
    blocks = {
        "displacement": basis.displacement,
        "velocity": basis.velocity,
        "pressure": basis.pressure,
    }
    if basis.supremizer is not None:
        blocks["supremizer"] = basis.supremizer
    return {
        name: blocks[name]
        for name in ("displacement", "velocity", "supremizer", "pressure")
        if name in blocks and blocks[name].shape[1] > 0
    }


def project_linear_operator_blocks(
    full_operator: np.ndarray,
    full_rhs: np.ndarray,
    basis: BallarinBlockBasis,
) -> ReducedLinearOperatorBlocks:
    """Project ``A y = f`` into explicit named reduced block matrices."""

    operator = np.asarray(full_operator, dtype=float)
    rhs = np.asarray(full_rhs, dtype=float).reshape(-1, 1)
    mean = basis.mean
    named_basis = _ballarin_named_basis_blocks(basis)
    block_names = tuple(named_basis.keys())
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError(f"full_operator must be square, got {operator.shape}")
    if rhs.shape[0] != operator.shape[0] or mean.shape[0] != operator.shape[0]:
        raise ValueError("operator, rhs, and basis dimensions do not match")

    blocks = {
        (row_name, column_name): row_basis.T @ operator @ column_basis
        for row_name, row_basis in named_basis.items()
        for column_name, column_basis in named_basis.items()
    }
    reduced_rhs = {
        row_name: row_basis.T @ (rhs - operator @ mean)
        for row_name, row_basis in named_basis.items()
    }
    return ReducedLinearOperatorBlocks(
        block_names=block_names,
        blocks=blocks,
        rhs_blocks=reduced_rhs,
    )


def reduced_blocks_from_dense(
    operator: np.ndarray,
    rhs: np.ndarray,
    block_mode_counts: dict[str, int],
) -> ReducedLinearOperatorBlocks:
    """Split a dense reduced system into named blocks by coefficient counts."""

    matrix = np.asarray(operator, dtype=float)
    vector = np.asarray(rhs, dtype=float).reshape(-1, 1)
    block_names = tuple(name for name, count in block_mode_counts.items() if count > 0)
    total = sum(int(block_mode_counts[name]) for name in block_names)
    if matrix.shape != (total, total) or vector.shape != (total, 1):
        raise ValueError("operator/rhs shapes do not match block_mode_counts")

    offsets = {}
    cursor = 0
    for name in block_names:
        count = int(block_mode_counts[name])
        offsets[name] = slice(cursor, cursor + count)
        cursor += count
    return ReducedLinearOperatorBlocks(
        block_names=block_names,
        blocks={
            (row_name, column_name): matrix[offsets[row_name], offsets[column_name]]
            for row_name in block_names
            for column_name in block_names
        },
        rhs_blocks={name: vector[offsets[name], :] for name in block_names},
    )


def project_linear_operator(
    full_operator: np.ndarray,
    full_rhs: np.ndarray,
    basis: BallarinBlockBasis,
) -> ReducedLinearOperators:
    """Project ``A y = f`` to ``Phi.T A Phi a = Phi.T (f - A mean)``."""

    operator = np.asarray(full_operator, dtype=float)
    rhs = np.asarray(full_rhs, dtype=float).reshape(-1, 1)
    phi = basis.combined_basis
    mean = basis.mean
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError(f"full_operator must be square, got {operator.shape}")
    if operator.shape[0] != phi.shape[0] or rhs.shape[0] != phi.shape[0]:
        raise ValueError("operator, rhs, and basis dimensions do not match")
    return ReducedLinearOperators(
        basis=basis,
        operator=phi.T @ operator @ phi,
        rhs=phi.T @ (rhs - operator @ mean),
        full_operator_shape=operator.shape,
    )


def reduced_linear_residual(
    coefficients: np.ndarray,
    operators: ReducedLinearOperators,
) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=float)
    if coeff.ndim == 1:
        coeff = coeff[:, None]
    return operators.operator @ coeff - operators.rhs


def reduced_linear_jacobian(operators: ReducedLinearOperators) -> np.ndarray:
    return operators.operator


def affine_rhs_from_linearized_residual(
    projected_residual_at_reference: np.ndarray,
    projected_jacobian_mean_shift: np.ndarray,
) -> np.ndarray:
    """Build reduced RHS for a residual linearized around a reference state.

    For ``R(y) ≈ R(y_ref) + J_ref (y - y_ref)`` and
    ``y = mean + Phi a``, the reduced system is

    ``Phi.T J_ref Phi a = -Phi.T R(y_ref) - Phi.T J_ref (mean - y_ref)``.
    """

    residual = np.asarray(projected_residual_at_reference, dtype=float).reshape(-1, 1)
    shift = np.asarray(projected_jacobian_mean_shift, dtype=float).reshape(-1, 1)
    if residual.shape != shift.shape:
        raise ValueError(
            f"projected residual shape {residual.shape} does not match shift {shift.shape}"
        )
    return -residual - shift


def rhs_N_from_equation6_linearization(
    R_N_ref: np.ndarray,
    J_N_mean_shift: np.ndarray,
) -> np.ndarray:
    """Paper-style name for the linearized Eq. (6) reduced right-hand side."""

    return affine_rhs_from_linearized_residual(R_N_ref, J_N_mean_shift)


def solve_reduced_linear_system(
    operators: ReducedLinearOperators,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the reduced affine linear system and reconstruct full state."""

    coefficients = np.linalg.solve(operators.operator, operators.rhs)
    state = split_ballarin_state(coefficients, operators.basis)
    full_state = reconstruct_ballarin_state(state, operators.basis)
    return coefficients, full_state


def _mode_counts_from_basis(basis: BallarinBlockBasis) -> dict[str, int]:
    return {
        key: value
        for key, value in basis.num_modes.items()
        if value > 0
    }


def save_reduced_operator_bundle(
    path: Path,
    operators: ReducedLinearOperators,
    quadratic_tensor: np.ndarray | None = None,
    projected_residual_at_reference: np.ndarray | None = None,
    projected_jacobian_mean_shift: np.ndarray | None = None,
    linear_operator_blocks: ReducedLinearOperatorBlocks | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save reduced operators for online use without full FEniCS assembly."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "operator": operators.operator,
        "rhs": operators.rhs,
        "J_N": operators.J_N,
        "rhs_N": operators.rhs_N,
        "full_operator_shape": np.asarray(operators.full_operator_shape, dtype=np.int64),
        "block_names": np.asarray(list(_mode_counts_from_basis(operators.basis).keys())),
        "block_mode_counts": np.asarray(
            list(_mode_counts_from_basis(operators.basis).values()),
            dtype=np.int64,
        ),
    }
    if linear_operator_blocks is not None:
        arrays["explicit_block_names"] = np.asarray(linear_operator_blocks.block_names)
        for row_name, column_name in linear_operator_blocks.blocks:
            arrays[f"J_N__{row_name}__{column_name}"] = linear_operator_blocks.blocks[(row_name, column_name)]
        for row_name, rhs_block in linear_operator_blocks.rhs_blocks.items():
            arrays[f"rhs_N__{row_name}"] = rhs_block

    if quadratic_tensor is not None:
        arrays["quadratic_tensor"] = np.asarray(quadratic_tensor, dtype=float)
    if projected_residual_at_reference is not None:
        arrays["projected_residual_at_reference"] = np.asarray(
            projected_residual_at_reference,
            dtype=float,
        ).reshape(-1, 1)
        arrays["R_N_ref"] = arrays["projected_residual_at_reference"]
    if projected_jacobian_mean_shift is not None:
        arrays["projected_jacobian_mean_shift"] = np.asarray(
            projected_jacobian_mean_shift,
            dtype=float,
        ).reshape(-1, 1)
        arrays["J_N_mean_shift"] = arrays["projected_jacobian_mean_shift"]
    if metadata:
        arrays["metadata_json"] = np.asarray([json.dumps(metadata, sort_keys=True)])
    np.savez_compressed(path, **arrays)


def load_reduced_operator_bundle(path: Path) -> ReducedOperatorBundle:
    data = np.load(path, allow_pickle=True)
    names = [str(name) for name in data["block_names"]]
    counts = [int(value) for value in data["block_mode_counts"]]
    full_shape = tuple(int(value) for value in data["full_operator_shape"])
    metadata = None
    if "metadata_json" in data.files:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).reshape(-1)[0]))
    explicit_blocks = None
    if "explicit_block_names" in data.files:
        explicit_names = tuple(str(name) for name in data["explicit_block_names"])
        explicit_blocks = ReducedLinearOperatorBlocks(
            block_names=explicit_names,
            blocks={
                (row_name, column_name): np.asarray(
                    data[f"J_N__{row_name}__{column_name}"],
                    dtype=float,
                )
                for row_name in explicit_names
                for column_name in explicit_names
            },
            rhs_blocks={
                row_name: np.asarray(data[f"rhs_N__{row_name}"], dtype=float)
                for row_name in explicit_names
            },
        )

    return ReducedOperatorBundle(
        operator=np.asarray(data["J_N"] if "J_N" in data.files else data["operator"], dtype=float),
        rhs=np.asarray(data["rhs_N"] if "rhs_N" in data.files else data["rhs"], dtype=float),
        block_mode_counts=dict(zip(names, counts)),
        full_operator_shape=full_shape,
        quadratic_tensor=np.asarray(data["quadratic_tensor"], dtype=float)
        if "quadratic_tensor" in data.files
        else None,
        projected_residual_at_reference=np.asarray(
            data["R_N_ref"] if "R_N_ref" in data.files else data["projected_residual_at_reference"],
            dtype=float,
        )
        if "R_N_ref" in data.files or "projected_residual_at_reference" in data.files
        else None,
        projected_jacobian_mean_shift=np.asarray(
            data["J_N_mean_shift"]
            if "J_N_mean_shift" in data.files
            else data["projected_jacobian_mean_shift"],
            dtype=float,
        )
        if "J_N_mean_shift" in data.files or "projected_jacobian_mean_shift" in data.files
        else None,
        linear_operator_blocks=explicit_blocks,
        metadata=metadata,
    )


def solve_reduced_operator_bundle(bundle: ReducedOperatorBundle) -> np.ndarray:
    """Solve a saved reduced linear operator bundle in reduced coordinates."""

    return np.linalg.solve(bundle.operator, bundle.rhs)


def project_quadratic_tensor(
    full_tensor: np.ndarray,
    basis: BallarinBlockBasis,
) -> np.ndarray:
    """Project a full quadratic tensor to reduced coordinates.

    ``full_tensor[i, j, k]`` represents the contribution to full residual row
    ``i`` from ``y_j * y_k``. The returned tensor satisfies:

    ``C_N[:, a, b] = Phi.T C(Phi_a, Phi_b)``.
    """

    tensor = np.asarray(full_tensor, dtype=float)
    phi = basis.combined_basis
    if tensor.ndim != 3 or tensor.shape != (phi.shape[0], phi.shape[0], phi.shape[0]):
        raise ValueError(
            f"full_tensor must have shape {(phi.shape[0], phi.shape[0], phi.shape[0])}, "
            f"got {tensor.shape}"
        )
    return np.einsum("ir,ijk,ja,kb->rab", phi, tensor, phi, phi)


def evaluate_reduced_quadratic(
    reduced_tensor: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=float).reshape(-1)
    return np.einsum("rab,a,b->r", reduced_tensor, coeff, coeff).reshape(-1, 1)
