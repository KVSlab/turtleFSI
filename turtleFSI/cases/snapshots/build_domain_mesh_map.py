"""Build a ROM mesh map from saved turtleFSI domain markers.

This builder reads ``Mesh/mesh.h5`` files written by turtleFSI and derives the
fluid, solid, interface, and adjacency node sets from the cell domain markers.
Unlike ``build_tf_fsi_mesh_map.py``, it does not assume a particular benchmark
geometry or spatial dimension.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from turtleFSI.cases.snapshots.build_tf_fsi_mesh_map import pack_index_lists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-h5",
        default=None,
        help="Path to a turtleFSI Mesh/mesh.h5 file containing /mesh and /domains.",
    )
    parser.add_argument(
        "--results-folder",
        default=None,
        help=(
            "Path to a turtleFSI result folder. When provided, defaults to "
            "RESULTS/Mesh/mesh.h5 and reads dx_f_id/dx_s_id from "
            "RESULTS/Checkpoint/default_variables.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the mesh-map .npz and summary JSON will be written.",
    )
    parser.add_argument(
        "--fluid-id",
        nargs="+",
        type=int,
        default=None,
        help="Domain marker id(s) treated as fluid cells. Defaults to dx_f_id or 1.",
    )
    parser.add_argument(
        "--solid-id",
        nargs="+",
        type=int,
        default=None,
        help="Domain marker id(s) treated as solid cells. Defaults to dx_s_id or 2.",
    )
    return parser.parse_args()


def _as_id_list(value: Any, fallback: list[int]) -> list[int]:
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _load_default_variables(results_folder: Path | None) -> dict[str, Any]:
    if results_folder is None:
        return {}
    metadata_path = results_folder / "Checkpoint" / "default_variables.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="ascii") as handle:
        return json.load(handle)


def resolve_cli_paths_and_ids(args: argparse.Namespace) -> tuple[Path, Path, set[int], set[int]]:
    results_folder = Path(args.results_folder) if args.results_folder is not None else None
    defaults = _load_default_variables(results_folder)

    if args.mesh_h5 is not None:
        mesh_h5_path = Path(args.mesh_h5)
    elif results_folder is not None:
        mesh_h5_path = results_folder / "Mesh" / "mesh.h5"
    else:
        raise ValueError("Provide either --results-folder or --mesh-h5.")

    if not mesh_h5_path.exists():
        raise FileNotFoundError(
            f"Mesh file does not exist: {mesh_h5_path}. "
            "Pass a real turtleFSI result folder with --results-folder, or a real mesh file with --mesh-h5."
        )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif results_folder is not None:
        output_dir = results_folder / "ROM" / "domain_mesh_map"
    else:
        output_dir = mesh_h5_path.parent / "domain_mesh_map"

    fluid_ids = set(
        _as_id_list(args.fluid_id, _as_id_list(defaults.get("dx_f_id"), [1]))
    )
    solid_ids = set(
        _as_id_list(args.solid_id, _as_id_list(defaults.get("dx_s_id"), [2]))
    )
    return mesh_h5_path, output_dir, fluid_ids, solid_ids


def load_marked_mesh(mesh_h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from dolfin import HDF5File, Mesh, MeshFunction
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "build_domain_mesh_map.py requires dolfin/FEniCS to read turtleFSI mesh.h5 files."
        ) from exc

    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), str(mesh_h5_path), "r") as hdf:
        hdf.read(mesh, "/mesh", False)
        domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
        hdf.read(domains, "/domains")

    coordinates = np.asarray(mesh.coordinates(), dtype=float)
    cells = np.asarray(mesh.cells(), dtype=np.int32)
    domain_values = np.asarray(domains.array(), dtype=np.int64)
    if cells.shape[0] != domain_values.shape[0]:
        raise ValueError(
            f"Cell/domain marker count mismatch: {cells.shape[0]} cells, "
            f"{domain_values.shape[0]} domain values"
        )
    return coordinates, cells, domain_values


def classify_domain_markers(
    domain_values: np.ndarray,
    fluid_ids: set[int],
    solid_ids: set[int],
) -> np.ndarray:
    cell_regions = np.zeros(domain_values.shape[0], dtype=np.int8)
    fluid_mask = np.isin(domain_values, list(fluid_ids))
    solid_mask = np.isin(domain_values, list(solid_ids))
    overlap = fluid_mask & solid_mask
    if np.any(overlap):
        overlapping_ids = sorted(set(domain_values[overlap].astype(int)))
        raise ValueError(f"Domain ids cannot be both fluid and solid: {overlapping_ids}")
    cell_regions[fluid_mask] = 1
    cell_regions[solid_mask] = 2
    missing = cell_regions == 0
    if np.any(missing):
        unknown = sorted(set(domain_values[missing].astype(int)))
        raise ValueError(
            "Every cell must be classified as fluid or solid. "
            f"Unclassified domain marker ids: {unknown}"
        )
    return cell_regions


def build_topology(coordinates: np.ndarray, cells: np.ndarray, cell_regions: np.ndarray) -> dict[str, np.ndarray]:
    vertex_to_cells: list[list[int]] = [[] for _ in range(coordinates.shape[0])]
    vertex_neighbors: list[set[int]] = [set() for _ in range(coordinates.shape[0])]
    facet_to_cells: dict[tuple[int, ...], list[int]] = defaultdict(list)

    vertices_per_cell = cells.shape[1]
    facet_size = vertices_per_cell - 1
    if facet_size < 1:
        raise ValueError(f"Unsupported cell connectivity shape: {cells.shape}")

    for cell_index, cell in enumerate(cells):
        verts = tuple(int(vertex) for vertex in cell)
        for vertex in verts:
            vertex_to_cells[vertex].append(cell_index)
        for i, j in combinations(verts, 2):
            vertex_neighbors[i].add(j)
            vertex_neighbors[j].add(i)
        for facet in combinations(verts, facet_size):
            facet_to_cells[tuple(sorted(facet))].append(cell_index)

    node_regions = np.zeros(coordinates.shape[0], dtype=np.int8)
    fluid_nodes = []
    solid_nodes = []
    interface_nodes = []
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
        elif not incident_regions:
            raise ValueError(f"Vertex {vertex} is not attached to any cell")
        else:
            raise ValueError(
                f"Unexpected region combination {incident_regions} at vertex {vertex}"
            )

    interface_facets = []
    for facet, incident_cells in facet_to_cells.items():
        incident_regions = {int(cell_regions[cell]) for cell in incident_cells}
        if incident_regions == {1, 2}:
            interface_facets.append(facet)

    ring1_fluid = set()
    ring1_solid = set()
    for node in interface_nodes:
        for neighbor in vertex_neighbors[node]:
            region = node_regions[neighbor]
            if region == 1:
                ring1_fluid.add(neighbor)
            elif region == 2:
                ring1_solid.add(neighbor)

    interface_facets_array = np.asarray(interface_facets, dtype=np.int32)
    if interface_facets_array.size == 0:
        interface_facets_array = np.zeros((0, facet_size), dtype=np.int32)

    return {
        "node_regions": node_regions,
        "fluid_nodes": np.asarray(fluid_nodes, dtype=np.int32),
        "solid_nodes": np.asarray(solid_nodes, dtype=np.int32),
        "interface_nodes": np.asarray(interface_nodes, dtype=np.int32),
        "interface_facets": interface_facets_array,
        "near_interface_fluid_nodes": np.asarray(sorted(ring1_fluid), dtype=np.int32),
        "near_interface_solid_nodes": np.asarray(sorted(ring1_solid), dtype=np.int32),
        "vertex_neighbor_offsets": pack_index_lists(vertex_neighbors)[0],
        "vertex_neighbor_indices": pack_index_lists(vertex_neighbors)[1],
        "vertex_to_cell_offsets": pack_index_lists(vertex_to_cells)[0],
        "vertex_to_cell_indices": pack_index_lists(vertex_to_cells)[1],
    }


def _interface_edges_for_2d(interface_facets: np.ndarray) -> np.ndarray:
    if interface_facets.ndim != 2 or interface_facets.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.int32)
    return interface_facets


def edge_lengths(coordinates: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros(0, dtype=float)
    pts0 = coordinates[edges[:, 0]]
    pts1 = coordinates[edges[:, 1]]
    return np.linalg.norm(pts1 - pts0, axis=1)


def write_outputs(
    output_dir: Path,
    mesh_h5_path: Path,
    coordinates: np.ndarray,
    cells: np.ndarray,
    domain_values: np.ndarray,
    cell_regions: np.ndarray,
    topology: dict[str, np.ndarray],
    fluid_ids: set[int],
    solid_ids: set[int],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    interface_edges = _interface_edges_for_2d(topology["interface_facets"])
    output_path = output_dir / "domain_mesh_map.npz"
    np.savez_compressed(
        output_path,
        coordinates=coordinates,
        cells=cells,
        domain_values=domain_values,
        cell_regions=cell_regions,
        node_regions=topology["node_regions"],
        fluid_nodes=topology["fluid_nodes"],
        solid_nodes=topology["solid_nodes"],
        interface_nodes=topology["interface_nodes"],
        interface_facets=topology["interface_facets"],
        interface_edges=interface_edges,
        interface_edge_lengths=edge_lengths(coordinates, interface_edges),
        near_interface_fluid_nodes=topology["near_interface_fluid_nodes"],
        near_interface_solid_nodes=topology["near_interface_solid_nodes"],
        vertex_neighbor_offsets=topology["vertex_neighbor_offsets"],
        vertex_neighbor_indices=topology["vertex_neighbor_indices"],
        vertex_to_cell_offsets=topology["vertex_to_cell_offsets"],
        vertex_to_cell_indices=topology["vertex_to_cell_indices"],
    )

    summary = {
        "mesh_h5_path": str(mesh_h5_path),
        "output_path": str(output_path),
        "geometric_dimension": int(coordinates.shape[1]),
        "vertices_per_cell": int(cells.shape[1]),
        "fluid_ids": sorted(fluid_ids),
        "solid_ids": sorted(solid_ids),
        "counts": {
            "num_vertices": int(coordinates.shape[0]),
            "num_cells": int(cells.shape[0]),
            "num_fluid_cells": int(np.sum(cell_regions == 1)),
            "num_solid_cells": int(np.sum(cell_regions == 2)),
            "num_fluid_only_nodes": int(topology["fluid_nodes"].size),
            "num_solid_only_nodes": int(topology["solid_nodes"].size),
            "num_interface_nodes": int(topology["interface_nodes"].size),
            "num_interface_facets": int(topology["interface_facets"].shape[0]),
            "num_near_interface_fluid_nodes": int(topology["near_interface_fluid_nodes"].size),
            "num_near_interface_solid_nodes": int(topology["near_interface_solid_nodes"].size),
        },
        "notes": [
            "This map is built from saved domain markers, not benchmark geometry.",
            "cell_regions use 1 for fluid and 2 for solid.",
            "node_regions use 1 for fluid-only, 2 for solid-only, and 3 for interface.",
        ],
    }
    with open(output_dir / "domain_mesh_map_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def build_domain_mesh_map(
    mesh_h5_path: Path,
    output_dir: Path,
    fluid_ids: set[int],
    solid_ids: set[int],
) -> dict[str, Any]:
    coordinates, cells, domain_values = load_marked_mesh(mesh_h5_path)
    cell_regions = classify_domain_markers(domain_values, fluid_ids, solid_ids)
    topology = build_topology(coordinates, cells, cell_regions)
    return write_outputs(
        output_dir=output_dir,
        mesh_h5_path=mesh_h5_path,
        coordinates=coordinates,
        cells=cells,
        domain_values=domain_values,
        cell_regions=cell_regions,
        topology=topology,
        fluid_ids=fluid_ids,
        solid_ids=solid_ids,
    )


def main() -> None:
    args = parse_args()
    mesh_h5_path, output_dir, fluid_ids, solid_ids = resolve_cli_paths_and_ids(args)
    summary = build_domain_mesh_map(
        mesh_h5_path=mesh_h5_path,
        output_dir=output_dir,
        fluid_ids=fluid_ids,
        solid_ids=solid_ids,
    )
    print(f"Wrote ROM domain mesh map to {summary['output_path']}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
