"""Monolithic POD spaces for mixed FSI state vectors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from turtleFSI.rom._shared.pod import PODBasis, compute_pod_basis, project_to_basis, reconstruct_from_basis


@dataclass(frozen=True)
class FieldLayout:
    name: str
    start: int
    stop: int
    dofs: np.ndarray | None = None

    @property
    def size(self) -> int:
        if self.dofs is not None:
            return int(self.dofs.size)
        return self.stop - self.start

    @property
    def num_modes(self) -> int:
        if self.dofs is None:
            return self.stop - self.start
        return self.stop - self.start

    @property
    def coefficient_slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class MonolithicPODBasis:
    pod_basis: PODBasis
    fields: tuple[FieldLayout, ...]

    @property
    def num_dofs(self) -> int:
        return int(self.pod_basis.basis.shape[0])

    @property
    def num_modes(self) -> int:
        return int(self.pod_basis.basis.shape[1])


def _ensure_consistent_snapshot_count(blocks: dict[str, np.ndarray]) -> int:
    snapshot_count = None
    for name, values in blocks.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"{name} must be 2D, got {array.shape}")
        if snapshot_count is None:
            snapshot_count = array.shape[1]
        elif array.shape[1] != snapshot_count:
            raise ValueError(
                f"{name} has {array.shape[1]} snapshots, expected {snapshot_count}"
            )
    if snapshot_count is None:
        raise ValueError("at least one snapshot block is required")
    return int(snapshot_count)


def concatenate_snapshot_blocks(
    blocks: dict[str, np.ndarray],
    field_order: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, tuple[FieldLayout, ...]]:
    if not field_order:
        raise ValueError("field_order must not be empty")
    missing = [name for name in field_order if name not in blocks]
    if missing:
        raise KeyError(f"missing snapshot blocks: {missing}")

    _ensure_consistent_snapshot_count({name: blocks[name] for name in field_order})
    matrices = []
    fields = []
    cursor = 0
    for name in field_order:
        values = np.asarray(blocks[name], dtype=float)
        matrices.append(values)
        fields.append(FieldLayout(name=name, start=cursor, stop=cursor + values.shape[0]))
        cursor += values.shape[0]
    return np.vstack(matrices), tuple(fields)


def split_state_vector(
    state: np.ndarray,
    fields: tuple[FieldLayout, ...],
) -> dict[str, np.ndarray]:
    values = np.asarray(state, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"state must be 1D or 2D, got {values.shape}")
    result = {}
    for field in fields:
        if field.dofs is not None:
            result[field.name] = values[field.dofs, :]
        else:
            result[field.name] = values[field.start : field.stop]
    return result


def split_reduced_coefficients(
    coefficients: np.ndarray,
    fields: tuple[FieldLayout, ...],
) -> dict[str, np.ndarray]:
    values = np.asarray(coefficients, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"coefficients must be 1D or 2D, got {values.shape}")
    return {
        field.name: values[field.coefficient_slice, :]
        for field in fields
    }


def train_monolithic_pod_basis(
    blocks: dict[str, np.ndarray],
    field_order: list[str] | tuple[str, ...],
    energy_threshold: float = 0.999,
    max_modes: int | None = None,
) -> tuple[MonolithicPODBasis, np.ndarray]:
    snapshots, fields = concatenate_snapshot_blocks(blocks, field_order)
    basis = compute_pod_basis(
        snapshots,
        energy_threshold=energy_threshold,
        max_modes=max_modes,
        center=True,
    )
    return MonolithicPODBasis(pod_basis=basis, fields=fields), snapshots


def project_monolithic_snapshots(
    snapshots: np.ndarray,
    basis: MonolithicPODBasis,
) -> np.ndarray:
    return project_to_basis(snapshots, basis.pod_basis)


def reconstruct_monolithic_state(
    coefficients: np.ndarray,
    basis: MonolithicPODBasis,
) -> np.ndarray:
    return reconstruct_from_basis(coefficients, basis.pod_basis)


def reconstruct_monolithic_fields(
    coefficients: np.ndarray,
    basis: MonolithicPODBasis,
) -> dict[str, np.ndarray]:
    state = reconstruct_monolithic_state(coefficients, basis)
    return split_state_vector(state, basis.fields)


def reconstruct_monolithic_field_coefficients(
    coefficients: np.ndarray,
    basis: MonolithicPODBasis,
) -> dict[str, np.ndarray]:
    return split_reduced_coefficients(coefficients, basis.fields)


def save_monolithic_pod_basis(
    output_dir: Path,
    basis: MonolithicPODBasis,
    snapshots_path: str,
    reconstruction_error: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "mean": basis.pod_basis.mean,
        "basis": basis.pod_basis.basis,
        "singular_values": basis.pod_basis.singular_values,
        "energy_ratio": basis.pod_basis.energy_ratio,
        "field_names": np.asarray([field.name for field in basis.fields]),
        "field_offsets": np.asarray(
            [[field.start, field.stop] for field in basis.fields],
            dtype=np.int64,
        ),
    }
    for field in basis.fields:
        if field.dofs is not None:
            arrays[f"{field.name}_dofs"] = np.asarray(field.dofs, dtype=np.int64)
    np.savez_compressed(output_dir / "monolithic_pod_basis.npz", **arrays)

    summary = {
        "snapshots_path": snapshots_path,
        "num_dofs": basis.num_dofs,
        "num_modes": basis.num_modes,
        "retained_energy": float(basis.pod_basis.energy_ratio[-1]),
        "reconstruction_relative_error": float(reconstruction_error),
        "fields": {
            field.name: {
                "coefficient_start": int(field.start),
                "coefficient_stop": int(field.stop),
                "num_modes": int(field.num_modes),
                "size": int(field.size),
                "dof_key": f"{field.name}_dofs" if field.dofs is not None else None,
                "num_state_dofs": int(field.dofs.size) if field.dofs is not None else None,
            }
            for field in basis.fields
        },
    }
    with open(output_dir / "monolithic_pod_basis_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def load_monolithic_pod_basis(path: Path) -> MonolithicPODBasis:
    data = np.load(path)
    names = [str(name) for name in data["field_names"]]
    offsets = np.asarray(data["field_offsets"], dtype=np.int64)
    fields = tuple(
        FieldLayout(
            name=name,
            start=int(offset[0]),
            stop=int(offset[1]),
            dofs=np.asarray(data[f"{name}_dofs"], dtype=np.int64)
            if f"{name}_dofs" in data.files
            else None,
        )
        for name, offset in zip(names, offsets)
    )
    pod_basis = PODBasis(
        mean=np.asarray(data["mean"], dtype=float),
        basis=np.asarray(data["basis"], dtype=float),
        singular_values=np.asarray(data["singular_values"], dtype=float),
        energy_ratio=np.asarray(data["energy_ratio"], dtype=float),
    )
    return MonolithicPODBasis(pod_basis=pod_basis, fields=fields)
