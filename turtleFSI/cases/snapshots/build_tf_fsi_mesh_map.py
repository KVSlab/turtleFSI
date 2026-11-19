"""Build a mesh-based ROM map for the TF_fsi benchmark.

This script does not require FEniCS. It reconstructs the TF_fsi mesh topology
from the packaged XML mesh, classifies cells/nodes into fluid and solid
regions using the benchmark geometry, and derives interface and adjacency
metadata needed for the next ROM preprocessing steps.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


TF_FSI_GEOMETRY = {
    "R": 0.05,
    "H": 0.41,
    "L": 2.5,
    "f_L": 0.35,
    "f_H": 0.02,
    "c_x": 0.2,
    "c_y": 0.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh",
        default="turtleFSI/mesh/TF_fsi.xml.gz",
        help="Path to the packaged TF_fsi XML mesh.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/generated/tf_fsi_mesh_map",
        help="Directory where the ROM map files will be written.",
    )
    return parser.parse_args()


def load_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with gzip.open(mesh_path, "rt") as handle:
        root = ET.parse(handle).getroot()

    mesh_node = root.find("mesh")
    if mesh_node is None:
        raise ValueError(f"Could not find <mesh> in {mesh_path}")

    vertices_node = mesh_node.find("vertices")
    cells_node = mesh_node.find("cells")
    if vertices_node is None or cells_node is None:
        raise ValueError(f"Missing <vertices> or <cells> in {mesh_path}")

    vertex_count = int(vertices_node.attrib["size"])
    cell_count = int(cells_node.attrib["size"])

    coordinates = np.zeros((vertex_count, 2), dtype=float)
    for vertex in vertices_node:
        idx = int(vertex.attrib["index"])
        coordinates[idx, 0] = float(vertex.attrib["x"])
        coordinates[idx, 1] = float(vertex.attrib["y"])

    cells = np.zeros((cell_count, 3), dtype=np.int32)
    for cell in cells_node:
        idx = int(cell.attrib["index"])
        cells[idx, 0] = int(cell.attrib["v0"])
        cells[idx, 1] = int(cell.attrib["v1"])
        cells[idx, 2] = int(cell.attrib["v2"])

    return coordinates, cells


def classify_cells(coordinates: np.ndarray, cells: np.ndarray) -> np.ndarray:
    geom = TF_FSI_GEOMETRY
    centroids = coordinates[cells].mean(axis=1)
    solid = (
        (centroids[:, 1] >= geom["c_y"] - geom["f_H"] / 2.0)
        & (centroids[:, 1] <= geom["c_y"] + geom["f_H"] / 2.0)
        & (centroids[:, 0] >= geom["c_x"])
        & (centroids[:, 0] <= geom["c_x"] + geom["R"] + geom["f_L"])
    )
    cell_regions = np.ones(cells.shape[0], dtype=np.int8)
    cell_regions[solid] = 2
    return cell_regions


def build_topology(
    coordinates: np.ndarray, cells: np.ndarray, cell_regions: np.ndarray
) -> dict[str, np.ndarray]:
    vertex_to_cells: list[list[int]] = [[] for _ in range(coordinates.shape[0])]
    vertex_neighbors: list[set[int]] = [set() for _ in range(coordinates.shape[0])]
    edge_to_cells: dict[tuple[int, int], list[int]] = defaultdict(list)

    for cell_index, (a, b, c) in enumerate(cells):
        verts = (int(a), int(b), int(c))
        for vertex in verts:
            vertex_to_cells[vertex].append(cell_index)
        for i, j in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((int(i), int(j))))
            edge_to_cells[edge].append(cell_index)
            vertex_neighbors[edge[0]].add(edge[1])
            vertex_neighbors[edge[1]].add(edge[0])

    node_regions = np.zeros(coordinates.shape[0], dtype=np.int8)
    interface_nodes = []
    fluid_nodes = []
    solid_nodes = []
    for vertex, incident_cells in enumerate(vertex_to_cells):
        incident_regions = {int(cell_regions[cell]) for cell in incident_cells}
        if incident_regions == {1}:
            node_regions[vertex] = 1
            fluid_nodes.append(vertex)
        elif incident_regions == {2}:
            node_regions[vertex] = 2
            solid_nodes.append(vertex)
        elif incident_regions == {1, 2}:
            node_regions[vertex] = 3
            interface_nodes.append(vertex)
        else:
            raise ValueError(
                f"Unexpected region combination {incident_regions} at vertex {vertex}"
            )

    interface_edges = []
    for edge, incident_cells in edge_to_cells.items():
        incident_regions = {int(cell_regions[cell]) for cell in incident_cells}
        if incident_regions == {1, 2}:
            interface_edges.append(edge)

    interface_edge_array = np.asarray(interface_edges, dtype=np.int32)

    ring1_fluid = set()
    ring1_solid = set()
    for node in interface_nodes:
        for neighbor in vertex_neighbors[node]:
            region = node_regions[neighbor]
            if region == 1:
                ring1_fluid.add(neighbor)
            elif region == 2:
                ring1_solid.add(neighbor)

    return {
        "node_regions": node_regions,
        "fluid_nodes": np.asarray(fluid_nodes, dtype=np.int32),
        "solid_nodes": np.asarray(solid_nodes, dtype=np.int32),
        "interface_nodes": np.asarray(interface_nodes, dtype=np.int32),
        "interface_edges": interface_edge_array,
        "near_interface_fluid_nodes": np.asarray(sorted(ring1_fluid), dtype=np.int32),
        "near_interface_solid_nodes": np.asarray(sorted(ring1_solid), dtype=np.int32),
        "vertex_neighbor_offsets": pack_index_lists(vertex_neighbors)[0],
        "vertex_neighbor_indices": pack_index_lists(vertex_neighbors)[1],
        "vertex_to_cell_offsets": pack_index_lists(vertex_to_cells)[0],
        "vertex_to_cell_indices": pack_index_lists(vertex_to_cells)[1],
    }


def pack_index_lists(index_lists: list[list[int]] | list[set[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(index_lists) + 1, dtype=np.int64)
    values = []
    cursor = 0
    for i, indices in enumerate(index_lists):
        row = sorted(indices)
        cursor += len(row)
        offsets[i + 1] = cursor
        values.extend(row)
    return offsets, np.asarray(values, dtype=np.int32)


def edge_lengths(coordinates: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros(0, dtype=float)
    pts0 = coordinates[edges[:, 0]]
    pts1 = coordinates[edges[:, 1]]
    return np.linalg.norm(pts1 - pts0, axis=1)


def write_outputs(
    output_dir: Path,
    coordinates: np.ndarray,
    cells: np.ndarray,
    cell_regions: np.ndarray,
    topology: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "tf_fsi_mesh_map.npz",
        coordinates=coordinates,
        cells=cells,
        cell_regions=cell_regions,
        node_regions=topology["node_regions"],
        fluid_nodes=topology["fluid_nodes"],
        solid_nodes=topology["solid_nodes"],
        interface_nodes=topology["interface_nodes"],
        interface_edges=topology["interface_edges"],
        interface_edge_lengths=edge_lengths(coordinates, topology["interface_edges"]),
        near_interface_fluid_nodes=topology["near_interface_fluid_nodes"],
        near_interface_solid_nodes=topology["near_interface_solid_nodes"],
        vertex_neighbor_offsets=topology["vertex_neighbor_offsets"],
        vertex_neighbor_indices=topology["vertex_neighbor_indices"],
        vertex_to_cell_offsets=topology["vertex_to_cell_offsets"],
        vertex_to_cell_indices=topology["vertex_to_cell_indices"],
    )

    summary = {
        "mesh_path": "turtleFSI/mesh/TF_fsi.xml.gz",
        "geometry": TF_FSI_GEOMETRY,
        "cell_region_labels": {"1": "fluid", "2": "solid"},
        "node_region_labels": {
            "1": "fluid_only",
            "2": "solid_only",
            "3": "interface",
        },
        "counts": {
            "num_vertices": int(coordinates.shape[0]),
            "num_cells": int(cells.shape[0]),
            "num_fluid_cells": int(np.sum(cell_regions == 1)),
            "num_solid_cells": int(np.sum(cell_regions == 2)),
            "num_fluid_only_nodes": int(topology["fluid_nodes"].size),
            "num_solid_only_nodes": int(topology["solid_nodes"].size),
            "num_interface_nodes": int(topology["interface_nodes"].size),
            "num_interface_edges": int(topology["interface_edges"].shape[0]),
            "num_near_interface_fluid_nodes": int(
                topology["near_interface_fluid_nodes"].size
            ),
            "num_near_interface_solid_nodes": int(
                topology["near_interface_solid_nodes"].size
            ),
        },
        "notes": [
            "This is a mesh-level map, not an exact FEniCS DoF map.",
            "It is suitable for adjacency construction and ROM preprocessing.",
            "Exact mixed-space DoF indices still require dolfin/FEniCS.",
        ],
    }

    with open(output_dir / "tf_fsi_mesh_map_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    mesh_path = Path(args.mesh)
    output_dir = Path(args.output_dir)

    coordinates, cells = load_mesh(mesh_path)
    cell_regions = classify_cells(coordinates, cells)
    topology = build_topology(coordinates, cells, cell_regions)
    write_outputs(output_dir, coordinates, cells, cell_regions, topology)

    print(f"Wrote ROM mesh map to {output_dir}")


if __name__ == "__main__":
    main()
