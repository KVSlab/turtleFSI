"""Common helpers for ROM plotting scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = ("U", "Ds", "Vs", "Df")


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.tri as mtri  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "These plotting scripts require 'matplotlib'. Install it in the turtle env first."
        ) from exc
    return plt, mtri


def load_pipeline_artifacts(work_dir: Path) -> tuple[np.lib.npyio.NpzFile, np.lib.npyio.NpzFile, dict[str, Any]]:
    snapshots = np.load(work_dir / "tf_fsi_snapshots.npz")
    model = np.load(work_dir / "model" / "rom_model.npz")
    with open(work_dir / "model" / "rom_summary.json", "r", encoding="ascii") as handle:
        summary = json.load(handle)
    return snapshots, model, summary


def train_transition_count(num_snapshots: int, train_fraction: float) -> int:
    transition_count = num_snapshots - 1
    return max(1, min(transition_count - 1, int(np.floor(train_fraction * transition_count))))


def split_latent_state(model: np.lib.npyio.NpzFile, fields: list[str]) -> dict[str, np.ndarray]:
    parts = {}
    cursor = 0
    latent = []
    for field in fields:
        field_latent = model[f"{field}_latent"]
        parts[field] = field_latent
        latent.append(field_latent)
        cursor += field_latent.shape[0]
    return parts


def stack_latent_state(model: np.lib.npyio.NpzFile, fields: list[str]) -> np.ndarray:
    return np.vstack([model[f"{field}_latent"] for field in fields])


def reconstruct_field(model: np.lib.npyio.NpzFile, field: str, latent: np.ndarray | None = None) -> np.ndarray:
    basis = model[f"{field}_basis"]
    mean = model[f"{field}_mean"]
    if latent is None:
        latent = model[f"{field}_latent"]
    return mean + basis @ latent


def rollout_fields(
    model: np.lib.npyio.NpzFile,
    summary: dict[str, Any],
    fields: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    time = model["time"] if "time" in model.files else None
    if time is None:
        raise ValueError("ROM model does not include time values.")

    num_snapshots = int(time.shape[0])
    train_count = train_transition_count(num_snapshots, float(summary["train_fraction"]))

    full_latent = stack_latent_state(model, fields)
    operator = model["A"]
    bias = model["b"]

    current = full_latent[:, train_count : train_count + 1]
    predicted = []
    for _ in range(num_snapshots - train_count - 1):
        current = operator @ current + bias
        predicted.append(current[:, 0])
    predicted_latent = np.asarray(predicted, dtype=float).T

    blocks = {}
    cursor = 0
    for field in fields:
        rows = model[f"{field}_basis"].shape[1]
        blocks[field] = predicted_latent[cursor : cursor + rows]
        cursor += rows

    return time, blocks


def fluid_triangulation(mesh_map_path: Path):
    plt, mtri = _require_matplotlib()
    del plt
    mesh_map = np.load(mesh_map_path)
    coords = mesh_map["coordinates"]
    cells = mesh_map["cells"]
    cell_regions = mesh_map["cell_regions"]
    fluid_cells = cells[cell_regions == 1]
    return coords, fluid_cells, mtri.Triangulation(coords[:, 0], coords[:, 1], fluid_cells)


def expand_component_field(flat_values: np.ndarray, node_ids: np.ndarray, num_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.full(num_vertices, np.nan, dtype=float)
    y_values = np.full(num_vertices, np.nan, dtype=float)
    x_values[node_ids] = flat_values[0::2]
    y_values[node_ids] = flat_values[1::2]
    return x_values, y_values


def select_tip_node(mesh_map_path: Path, node_ids: np.ndarray) -> tuple[int, np.ndarray]:
    mesh_map = np.load(mesh_map_path)
    coords = mesh_map["coordinates"][node_ids]
    x_max = np.max(coords[:, 0])
    y_max = np.max(coords[:, 1])
    target = np.array([x_max, y_max], dtype=float)
    local_index = int(np.argmin(np.linalg.norm(coords - target, axis=1)))
    return local_index, coords[local_index]

