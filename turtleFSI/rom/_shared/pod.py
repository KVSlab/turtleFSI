"""POD utilities shared by ROM workflows.

The functions in this module operate on plain snapshot matrices with shape
``(num_dofs, num_snapshots)``. They intentionally avoid FEniCS dependencies so
they can be used by both offline exporters and small reduced-model tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PODBasis:
    """Mean field and orthonormal POD modes for one snapshot block."""

    mean: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    energy_ratio: np.ndarray

    @property
    def num_modes(self) -> int:
        return int(self.basis.shape[1])


def ensure_snapshot_matrix(name: str, snapshots: np.ndarray) -> np.ndarray:
    values = np.asarray(snapshots, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D snapshot matrix with shape "
            f"(num_dofs, num_snapshots); got {values.shape}"
        )
    if values.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one snapshot")
    return values


def select_num_modes(
    energy_ratio: np.ndarray,
    energy_threshold: float,
    max_modes: int | None = None,
) -> int:
    if not 0.0 < energy_threshold <= 1.0:
        raise ValueError(
            f"energy_threshold must be in (0, 1], got {energy_threshold}"
        )
    mode_count = int(np.searchsorted(energy_ratio, energy_threshold, side="left") + 1)
    if max_modes is not None:
        if max_modes < 1:
            raise ValueError(f"max_modes must be positive, got {max_modes}")
        mode_count = min(mode_count, max_modes)
    return max(1, mode_count)


def compute_pod_basis(
    snapshots: np.ndarray,
    energy_threshold: float = 0.999,
    max_modes: int | None = None,
    center: bool = True,
) -> PODBasis:
    values = ensure_snapshot_matrix("snapshots", snapshots)
    mean = values.mean(axis=1, keepdims=True) if center else np.zeros((values.shape[0], 1))
    centered = values - mean
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)

    energy = singular_values**2
    if np.allclose(energy.sum(), 0.0):
        basis = left[:, :1]
        return PODBasis(
            mean=mean,
            basis=basis,
            singular_values=singular_values[:1],
            energy_ratio=np.ones(1, dtype=float),
        )

    energy_ratio = np.cumsum(energy) / energy.sum()
    num_modes = select_num_modes(energy_ratio, energy_threshold, max_modes)
    return PODBasis(
        mean=mean,
        basis=left[:, :num_modes],
        singular_values=singular_values[:num_modes],
        energy_ratio=energy_ratio[:num_modes],
    )


def project_to_basis(snapshots: np.ndarray, pod_basis: PODBasis) -> np.ndarray:
    values = ensure_snapshot_matrix("snapshots", snapshots)
    if values.shape[0] != pod_basis.basis.shape[0]:
        raise ValueError(
            f"snapshot DoF count {values.shape[0]} does not match basis DoF count "
            f"{pod_basis.basis.shape[0]}"
        )
    return pod_basis.basis.T @ (values - pod_basis.mean)


def reconstruct_from_basis(coefficients: np.ndarray, pod_basis: PODBasis) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=float)
    if coeff.ndim == 1:
        coeff = coeff[:, None]
    if coeff.ndim != 2:
        raise ValueError(f"coefficients must be 1D or 2D, got {coeff.shape}")
    if coeff.shape[0] != pod_basis.basis.shape[1]:
        raise ValueError(
            f"coefficient count {coeff.shape[0]} does not match basis mode count "
            f"{pod_basis.basis.shape[1]}"
        )
    return pod_basis.mean + pod_basis.basis @ coeff


def relative_l2_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    residual = np.linalg.norm(ref - pred)
    baseline = np.linalg.norm(ref)
    if np.isclose(baseline, 0.0):
        return float(residual)
    return float(residual / baseline)
