"""Offline ROM builder for turtleFSI snapshot arrays.

This script is intentionally separate from the intrusive FEniCS solver. It
expects plain NumPy snapshot matrices exported from a full-order simulation and
builds a first ROM based on:

1. POD compression for each state block.
2. A global linear latent dynamics model.

The resulting model is suitable as a practical first ROM baseline and keeps the
training workflow independent from the monolithic solver.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FIELD_ALIASES = {
    "U": ("U", "fluid_velocity", "u"),
    "Ds": ("Ds", "solid_displacement", "ds"),
    "Vs": ("Vs", "solid_velocity", "vs"),
    "Df": ("Df", "fluid_mesh_displacement", "df"),
}


@dataclass
class SnapshotBlock:
    name: str
    snapshots: np.ndarray
    mean: np.ndarray
    basis: np.ndarray
    latent: np.ndarray
    singular_values: np.ndarray
    energy_ratio: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshots",
        required=True,
        help="Input .npz file with snapshot matrices such as U, Ds, Vs, Df and optional time.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/generated/tf_fsi_rom",
        help="Directory where the ROM model files will be written.",
    )
    parser.add_argument(
        "--mesh-map",
        default="paper/generated/tf_fsi_mesh_map/tf_fsi_mesh_map.npz",
        help="Optional mesh-map .npz used for metadata validation.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["U", "Ds", "Vs"],
        help="Snapshot blocks to include. Supported names: U, Ds, Vs, Df.",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.999,
        help="Cumulative POD energy threshold used to pick basis size.",
    )
    parser.add_argument(
        "--max-modes",
        type=int,
        default=None,
        help="Optional upper bound on POD modes per field.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-10,
        help="Ridge regularization for the latent linear system fit.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of transitions used for training. The rest are used for validation.",
    )
    return parser.parse_args()


def _resolve_field_name(dataset: np.lib.npyio.NpzFile, requested_name: str) -> str:
    aliases = FIELD_ALIASES.get(requested_name, (requested_name,))
    for alias in aliases:
        if alias in dataset.files:
            return alias
    raise KeyError(
        f"Could not find snapshots for '{requested_name}'. Checked aliases: {aliases}"
    )


def _ensure_2d_snapshot_matrix(name: str, array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"Snapshot block '{name}' must be 2D with shape (num_dofs, num_snapshots); "
            f"got {values.shape}"
        )
    if values.shape[1] < 2:
        raise ValueError(
            f"Snapshot block '{name}' needs at least 2 snapshots; got {values.shape[1]}"
        )
    return values


def select_num_modes(energy_ratio: np.ndarray, threshold: float, max_modes: int | None) -> int:
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"energy threshold must be in (0, 1], got {threshold}")
    mode_count = int(np.searchsorted(energy_ratio, threshold, side="left") + 1)
    if max_modes is not None:
        if max_modes < 1:
            raise ValueError(f"max_modes must be positive, got {max_modes}")
        mode_count = min(mode_count, max_modes)
    return mode_count


def compute_pod_block(
    name: str,
    snapshots: np.ndarray,
    energy_threshold: float,
    max_modes: int | None,
) -> SnapshotBlock:
    mean = snapshots.mean(axis=1, keepdims=True)
    centered = snapshots - mean
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)

    energy = singular_values ** 2
    if np.allclose(energy.sum(), 0.0):
        basis = left[:, :1]
        latent = basis.T @ centered
        energy_ratio = np.ones(1, dtype=float)
        singular_values = singular_values[:1]
    else:
        energy_ratio = np.cumsum(energy) / energy.sum()
        num_modes = select_num_modes(energy_ratio, energy_threshold, max_modes)
        basis = left[:, :num_modes]
        latent = basis.T @ centered
        energy_ratio = energy_ratio[:num_modes]
        singular_values = singular_values[:num_modes]

    return SnapshotBlock(
        name=name,
        snapshots=snapshots,
        mean=mean,
        basis=basis,
        latent=latent,
        singular_values=singular_values,
        energy_ratio=energy_ratio,
    )


def fit_latent_linear_dynamics(
    latent_state: np.ndarray,
    ridge: float,
    train_fraction: float,
) -> dict[str, Any]:
    if latent_state.ndim != 2:
        raise ValueError(f"latent state must be 2D, got {latent_state.shape}")
    if latent_state.shape[1] < 3:
        raise ValueError(
            "Need at least 3 latent snapshots to fit and validate a transition model."
        )
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be in (0, 1), got {train_fraction}"
        )

    x_all = latent_state[:, :-1]
    y_all = latent_state[:, 1:]
    transition_count = x_all.shape[1]
    train_count = max(1, min(transition_count - 1, int(np.floor(train_fraction * transition_count))))

    x_train = x_all[:, :train_count]
    y_train = y_all[:, :train_count]
    x_val = x_all[:, train_count:]
    y_val = y_all[:, train_count:]

    augmented = np.vstack([x_train, np.ones((1, x_train.shape[1]))])
    if ridge < 0.0:
        raise ValueError(f"ridge must be non-negative, got {ridge}")
    left, singular_values, right_t = np.linalg.svd(augmented, full_matrices=False)
    if np.isclose(ridge, 0.0):
        inverse_weights = np.divide(
            1.0,
            singular_values,
            out=np.zeros_like(singular_values),
            where=singular_values > np.finfo(float).eps,
        )
    else:
        inverse_weights = singular_values / (singular_values**2 + ridge)
    coeff = y_train @ right_t.T @ np.diag(inverse_weights) @ left.T

    operator = coeff[:, :-1]
    bias = coeff[:, -1:]

    one_step_train = operator @ x_train + bias
    train_rel = relative_l2_error(y_train, one_step_train)

    result: dict[str, Any] = {
        "A": operator,
        "b": bias,
        "train_relative_error": train_rel,
        "num_train_transitions": train_count,
        "num_validation_transitions": int(x_val.shape[1]),
    }

    if x_val.shape[1] > 0:
        one_step_val = operator @ x_val + bias
        rollout_val = rollout_latent_dynamics(
            operator=operator,
            bias=bias,
            initial_state=x_val[:, :1],
            num_steps=x_val.shape[1],
        )
        result["validation_one_step_relative_error"] = relative_l2_error(y_val, one_step_val)
        result["validation_rollout_relative_error"] = relative_l2_error(y_val, rollout_val)
    else:
        result["validation_one_step_relative_error"] = None
        result["validation_rollout_relative_error"] = None

    return result


def rollout_latent_dynamics(
    operator: np.ndarray,
    bias: np.ndarray,
    initial_state: np.ndarray,
    num_steps: int,
) -> np.ndarray:
    current = np.asarray(initial_state, dtype=float)
    states = []
    for _ in range(num_steps):
        current = operator @ current + bias
        states.append(current[:, 0])
    return np.asarray(states, dtype=float).T


def relative_l2_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    residual = np.linalg.norm(reference - prediction)
    baseline = np.linalg.norm(reference)
    if np.isclose(baseline, 0.0):
        return float(residual)
    return float(residual / baseline)


def load_mesh_map(mesh_map_path: Path | None) -> dict[str, Any] | None:
    if mesh_map_path is None or not mesh_map_path.exists():
        return None

    mesh_map = np.load(mesh_map_path)
    return {
        "path": str(mesh_map_path),
        "num_vertices": int(mesh_map["coordinates"].shape[0]),
        "num_cells": int(mesh_map["cells"].shape[0]),
        "num_fluid_nodes": int(mesh_map["fluid_nodes"].size),
        "num_solid_nodes": int(mesh_map["solid_nodes"].size),
        "num_interface_nodes": int(mesh_map["interface_nodes"].size),
        "num_near_interface_fluid_nodes": int(mesh_map["near_interface_fluid_nodes"].size),
        "num_near_interface_solid_nodes": int(mesh_map["near_interface_solid_nodes"].size),
    }


def load_snapshot_blocks(
    snapshots_path: Path,
    field_names: list[str],
    energy_threshold: float,
    max_modes: int | None,
) -> tuple[list[SnapshotBlock], np.ndarray | None]:
    dataset = np.load(snapshots_path)
    blocks = []
    snapshot_count = None

    for field_name in field_names:
        resolved_name = _resolve_field_name(dataset, field_name)
        snapshots = _ensure_2d_snapshot_matrix(field_name, dataset[resolved_name])
        if snapshot_count is None:
            snapshot_count = snapshots.shape[1]
        elif snapshots.shape[1] != snapshot_count:
            raise ValueError(
                f"All snapshot blocks must have the same number of snapshots. "
                f"'{field_name}' has {snapshots.shape[1]}, expected {snapshot_count}."
            )
        blocks.append(
            compute_pod_block(
                name=field_name,
                snapshots=snapshots,
                energy_threshold=energy_threshold,
                max_modes=max_modes,
            )
        )

    time = None
    if "time" in dataset.files:
        time = np.asarray(dataset["time"], dtype=float)
        if time.ndim != 1 or time.shape[0] != snapshot_count:
            raise ValueError(
                f"time must be 1D with length {snapshot_count}, got {time.shape}"
            )

    return blocks, time


def reconstruct_snapshot_blocks(blocks: list[SnapshotBlock]) -> dict[str, np.ndarray]:
    reconstructed = {}
    for block in blocks:
        reconstructed[block.name] = block.mean + block.basis @ block.latent
    return reconstructed


def train_rom(
    snapshots_path: Path,
    output_dir: Path,
    field_names: list[str],
    energy_threshold: float,
    max_modes: int | None,
    ridge: float,
    train_fraction: float,
    mesh_map_path: Path | None,
) -> dict[str, Any]:
    blocks, time = load_snapshot_blocks(
        snapshots_path=snapshots_path,
        field_names=field_names,
        energy_threshold=energy_threshold,
        max_modes=max_modes,
    )

    latent_state = np.vstack([block.latent for block in blocks])
    dynamics = fit_latent_linear_dynamics(
        latent_state=latent_state,
        ridge=ridge,
        train_fraction=train_fraction,
    )

    mesh_map_meta = load_mesh_map(mesh_map_path)
    reconstructed = reconstruct_snapshot_blocks(blocks)

    output_dir.mkdir(parents=True, exist_ok=True)

    model_arrays: dict[str, np.ndarray] = {
        "A": dynamics["A"],
        "b": dynamics["b"],
    }
    if time is not None:
        model_arrays["time"] = time

    summary_fields = {}
    for block in blocks:
        model_arrays[f"{block.name}_mean"] = block.mean
        model_arrays[f"{block.name}_basis"] = block.basis
        model_arrays[f"{block.name}_singular_values"] = block.singular_values
        model_arrays[f"{block.name}_latent"] = block.latent
        model_arrays[f"{block.name}_reconstruction"] = reconstructed[block.name]

        summary_fields[block.name] = {
            "num_dofs": int(block.snapshots.shape[0]),
            "num_snapshots": int(block.snapshots.shape[1]),
            "num_modes": int(block.basis.shape[1]),
            "retained_energy": float(block.energy_ratio[-1]),
            "reconstruction_relative_error": relative_l2_error(
                block.snapshots, reconstructed[block.name]
            ),
        }

    np.savez_compressed(output_dir / "rom_model.npz", **model_arrays)

    summary = {
        "snapshots_path": str(snapshots_path),
        "mesh_map": mesh_map_meta,
        "fields": summary_fields,
        "latent_dimension": int(latent_state.shape[0]),
        "num_snapshots": int(latent_state.shape[1]),
        "ridge": float(ridge),
        "train_fraction": float(train_fraction),
        "train_relative_error": dynamics["train_relative_error"],
        "validation_one_step_relative_error": dynamics["validation_one_step_relative_error"],
        "validation_rollout_relative_error": dynamics["validation_rollout_relative_error"],
        "notes": [
            "This is an offline POD + linear latent ROM baseline.",
            "It does not yet infer the adjacency-restricted operators described in the ROM plan.",
            "Snapshot export from the FEniCS mixed spaces is still a separate missing step in turtleFSI.",
        ],
    }

    with open(output_dir / "rom_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return summary


def main() -> None:
    args = parse_args()
    summary = train_rom(
        snapshots_path=Path(args.snapshots),
        output_dir=Path(args.output_dir),
        field_names=args.fields,
        energy_threshold=args.energy_threshold,
        max_modes=args.max_modes,
        ridge=args.ridge,
        train_fraction=args.train_fraction,
        mesh_map_path=Path(args.mesh_map) if args.mesh_map else None,
    )
    print(f"Wrote ROM model to {args.output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
