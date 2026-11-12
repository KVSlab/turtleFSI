"""Block reduced state helpers for Ballarin/Rozza-style monolithic ROMs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from turtleFSI.rom._pod_galerkin.monolithic_pod import MonolithicPODBasis


BALLARIN_BLOCK_ORDER = ("displacement", "velocity", "supremizer", "pressure")


@dataclass(frozen=True)
class BallarinBlockBasis:
    """Reduced displacement, velocity/supremizer, and pressure basis blocks."""

    displacement: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    mean: np.ndarray
    supremizer: np.ndarray | None = None

    @property
    def velocity_enriched(self) -> np.ndarray:
        if self.supremizer is None:
            return self.velocity
        return np.hstack([self.velocity, self.supremizer])

    @property
    def num_modes(self) -> dict[str, int]:
        return {
            "displacement": int(self.displacement.shape[1]),
            "velocity": int(self.velocity.shape[1]),
            "supremizer": 0 if self.supremizer is None else int(self.supremizer.shape[1]),
            "pressure": int(self.pressure.shape[1]),
        }

    @property
    def combined_basis(self) -> np.ndarray:
        return np.hstack([self.displacement, self.velocity_enriched, self.pressure])


@dataclass(frozen=True)
class BallarinState:
    """Reduced coefficients grouped as d_N, u_N/s_N, and p_N."""

    displacement: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    supremizer: np.ndarray | None = None

    def as_vector(self) -> np.ndarray:
        blocks = [self.displacement, self.velocity]
        if self.supremizer is not None:
            blocks.append(self.supremizer)
        blocks.append(self.pressure)
        return np.vstack([_column(block) for block in blocks])


def _column(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] != 1:
        raise ValueError(f"expected a coefficient column, got {array.shape}")
    return array


def block_basis_from_monolithic(basis: MonolithicPODBasis) -> BallarinBlockBasis:
    """Extract Ballarin/Rozza blocks from a blocked DVP basis."""

    blocks = {field.name: basis.pod_basis.basis[:, field.coefficient_slice] for field in basis.fields}
    missing = [name for name in ("displacement", "velocity", "pressure") if name not in blocks]
    if missing:
        raise ValueError(
            "Ballarin block basis requires blocked DVP fields; missing "
            f"{missing}. Build with --blocked-dvp first."
        )
    return BallarinBlockBasis(
        displacement=blocks["displacement"],
        velocity=blocks["velocity"],
        supremizer=blocks.get("supremizer"),
        pressure=blocks["pressure"],
        mean=basis.pod_basis.mean,
    )


def split_ballarin_state(coefficients: np.ndarray, block_basis: BallarinBlockBasis) -> BallarinState:
    """Split a combined coefficient vector into Ballarin block coefficients."""

    values = _column(coefficients)
    cursor = 0
    counts = block_basis.num_modes
    displacement = values[cursor : cursor + counts["displacement"]]
    cursor += counts["displacement"]
    velocity = values[cursor : cursor + counts["velocity"]]
    cursor += counts["velocity"]
    supremizer = None
    if counts["supremizer"]:
        supremizer = values[cursor : cursor + counts["supremizer"]]
        cursor += counts["supremizer"]
    pressure = values[cursor : cursor + counts["pressure"]]
    cursor += counts["pressure"]
    if cursor != values.shape[0]:
        raise ValueError(
            f"coefficient vector has {values.shape[0]} rows but block basis uses {cursor}"
        )
    return BallarinState(
        displacement=displacement,
        velocity=velocity,
        supremizer=supremizer,
        pressure=pressure,
    )


def reconstruct_ballarin_state(state: BallarinState, block_basis: BallarinBlockBasis) -> np.ndarray:
    """Reconstruct the full mixed DVP vector from block coefficients."""

    return block_basis.mean + block_basis.combined_basis @ state.as_vector()

