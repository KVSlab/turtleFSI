"""Adjacency helpers for structured non-intrusive ROMs."""

from __future__ import annotations

import numpy as np


def unpack_csr_rows(offsets: np.ndarray, indices: np.ndarray) -> list[np.ndarray]:
    row_offsets = np.asarray(offsets, dtype=np.int64)
    row_indices = np.asarray(indices, dtype=np.int32)
    if row_offsets.ndim != 1:
        raise ValueError("offsets must be one-dimensional")
    if row_offsets[0] != 0 or np.any(np.diff(row_offsets) < 0):
        raise ValueError("offsets must be a monotone CSR offset array starting at 0")
    if row_offsets[-1] != row_indices.size:
        raise ValueError("last offset must equal the number of indices")
    return [
        row_indices[row_offsets[i] : row_offsets[i + 1]]
        for i in range(row_offsets.size - 1)
    ]


def expand_node_neighborhoods(
    node_neighbors: list[np.ndarray],
    depth: int = 1,
    include_self: bool = True,
) -> list[np.ndarray]:
    if depth < 0:
        raise ValueError(f"depth must be non-negative, got {depth}")

    expanded = []
    for node in range(len(node_neighbors)):
        visited = {node} if include_self else set()
        frontier = {node}
        for _ in range(depth):
            next_frontier = set()
            for current in frontier:
                for neighbor in node_neighbors[current]:
                    value = int(neighbor)
                    if value not in visited:
                        visited.add(value)
                        next_frontier.add(value)
            frontier = next_frontier
        expanded.append(np.asarray(sorted(visited), dtype=np.int32))
    return expanded


def nodes_to_vector_dofs(nodes: np.ndarray, value_dim: int = 2) -> np.ndarray:
    node_ids = np.asarray(nodes, dtype=np.int64)
    if value_dim < 1:
        raise ValueError(f"value_dim must be positive, got {value_dim}")
    dofs = node_ids[:, None] * value_dim + np.arange(value_dim, dtype=np.int64)[None, :]
    return dofs.reshape(-1).astype(np.int32)


def load_mesh_neighborhoods(
    mesh_map_path: str,
    depth: int = 1,
    include_self: bool = True,
) -> list[np.ndarray]:
    mesh_map = np.load(mesh_map_path)
    node_neighbors = unpack_csr_rows(
        mesh_map["vertex_neighbor_offsets"],
        mesh_map["vertex_neighbor_indices"],
    )
    return expand_node_neighborhoods(
        node_neighbors,
        depth=depth,
        include_self=include_self,
    )
