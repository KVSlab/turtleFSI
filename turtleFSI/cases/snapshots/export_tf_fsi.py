"""Export ROM-ready snapshot matrices from TF_fsi visualization outputs.

This is an offline exporter. It reads the XDMF/HDF5 visualization files written
by turtleFSI, uses the precomputed mesh map to split the nodal fields into
fluid and solid subsets, and writes a compact `.npz` bundle consumable by the
offline ROM builder.

The exported arrays are mesh-level nodal quantities, not exact mixed-space
FEniCS DoF vectors.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class TimeSeriesSpec:
    xdmf_path: Path
    h5_path: Path
    dataset_paths: list[str]
    times: np.ndarray
    attribute_name: str
    attribute_type: str
    dimensions: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-folder",
        default="turtleFSI/problems/TF_fsi_results/1",
        help="Path to a turtleFSI TF_fsi results directory containing Visualization/.",
    )
    parser.add_argument(
        "--mesh-map",
        default="paper/generated/tf_fsi_mesh_map/tf_fsi_mesh_map.npz",
        help="Mesh-map .npz produced by build_tf_fsi_mesh_map.py.",
    )
    parser.add_argument(
        "--output",
        default="paper/generated/tf_fsi_rom/tf_fsi_snapshots.npz",
        help="Output .npz file with ROM-ready snapshot matrices.",
    )
    parser.add_argument(
        "--include-pressure",
        action="store_true",
        help="Also export fluid/interface nodal pressure snapshots as P.",
    )
    return parser.parse_args()


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This exporter requires 'h5py' to read turtleFSI HDF5 visualization files."
        ) from exc
    return h5py


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_hdf_dataitem_text(dataitem: ET.Element) -> tuple[str, str]:
    text = (dataitem.text or "").strip()
    if ":" not in text:
        raise ValueError(f"Expected HDF5 reference in DataItem, got '{text}'")
    file_part, dataset_part = text.split(":", 1)
    return file_part, dataset_part


def parse_xdmf_timeseries(xdmf_path: Path) -> TimeSeriesSpec:
    tree = ET.parse(xdmf_path)
    root = tree.getroot()

    collection = None
    for element in root.iter():
        if _strip_namespace(element.tag) == "Grid" and element.attrib.get("CollectionType") == "Temporal":
            collection = element
            break
    if collection is None:
        raise ValueError(f"Could not find temporal Grid in {xdmf_path}")

    grids = [child for child in collection if _strip_namespace(child.tag) == "Grid"]
    if not grids:
        raise ValueError(f"No timestep grids found in {xdmf_path}")

    dataset_paths = []
    times = []
    attribute_name = None
    attribute_type = None
    dimensions = None
    h5_file_name = None

    for grid in grids:
        time_node = next((child for child in grid if _strip_namespace(child.tag) == "Time"), None)
        attr_node = next((child for child in grid if _strip_namespace(child.tag) == "Attribute"), None)
        if time_node is None or attr_node is None:
            continue

        data_item = next((child for child in attr_node if _strip_namespace(child.tag) == "DataItem"), None)
        if data_item is None:
            raise ValueError(f"Attribute in {xdmf_path} is missing a DataItem")

        file_name, dataset_path = _parse_hdf_dataitem_text(data_item)
        if h5_file_name is None:
            h5_file_name = file_name
        elif h5_file_name != file_name:
            raise ValueError(f"Mixed HDF5 files in one XDMF series are not supported: {xdmf_path}")

        dims_text = data_item.attrib.get("Dimensions", "").strip()
        if dimensions is None:
            dimensions = tuple(int(value) for value in dims_text.split())
            attribute_name = attr_node.attrib.get("Name", "")
            attribute_type = attr_node.attrib.get("AttributeType", "")

        dataset_paths.append(dataset_path)
        times.append(float(time_node.attrib["Value"]))

    if h5_file_name is None or dimensions is None or attribute_name is None or attribute_type is None:
        raise ValueError(f"Failed to parse attribute metadata from {xdmf_path}")

    return TimeSeriesSpec(
        xdmf_path=xdmf_path,
        h5_path=xdmf_path.parent / h5_file_name,
        dataset_paths=dataset_paths,
        times=np.asarray(times, dtype=float),
        attribute_name=attribute_name,
        attribute_type=attribute_type,
        dimensions=dimensions,
    )


def load_timeseries_array(spec: TimeSeriesSpec) -> np.ndarray:
    h5py = _require_h5py()
    frames = []
    with h5py.File(spec.h5_path, "r") as handle:
        for dataset_path in spec.dataset_paths:
            frames.append(np.asarray(handle[dataset_path], dtype=float))
    return np.stack(frames, axis=0)


def _load_node_sets(mesh_map_path: Path) -> dict[str, np.ndarray]:
    mesh_map = np.load(mesh_map_path)
    fluid_interface = np.sort(
        np.concatenate([mesh_map["fluid_nodes"], mesh_map["interface_nodes"]]).astype(np.int32)
    )
    solid_interface = np.sort(
        np.concatenate([mesh_map["solid_nodes"], mesh_map["interface_nodes"]]).astype(np.int32)
    )
    return {
        "fluid_interface_nodes": fluid_interface,
        "solid_interface_nodes": solid_interface,
        "interface_nodes": mesh_map["interface_nodes"].astype(np.int32),
        "fluid_nodes": mesh_map["fluid_nodes"].astype(np.int32),
        "solid_nodes": mesh_map["solid_nodes"].astype(np.int32),
    }


def flatten_vector_snapshots(values: np.ndarray, node_indices: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            f"Expected vector time series with shape (steps, nodes, components), got {array.shape}"
        )
    selected = array[:, node_indices, :]
    num_snapshots = selected.shape[0]
    return selected.transpose(1, 2, 0).reshape(
        node_indices.size * selected.shape[2], num_snapshots
    )


def flatten_scalar_snapshots(values: np.ndarray, node_indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(values[:, node_indices], dtype=float)
    if selected.ndim == 3:
        selected = selected[..., 0]
    return selected.T


def _validate_series_match(reference: TimeSeriesSpec, other: TimeSeriesSpec) -> None:
    if reference.dataset_paths and len(reference.dataset_paths) != len(other.dataset_paths):
        raise ValueError(
            f"Snapshot count mismatch between {reference.xdmf_path.name} and {other.xdmf_path.name}"
        )
    if not np.allclose(reference.times, other.times):
        raise ValueError(
            f"Time values mismatch between {reference.xdmf_path.name} and {other.xdmf_path.name}"
        )


def export_snapshots(
    results_folder: Path,
    mesh_map_path: Path,
    output_path: Path,
    include_pressure: bool,
) -> dict[str, Any]:
    visualization = results_folder / "Visualization"
    velocity_spec = parse_xdmf_timeseries(visualization / "velocity.xdmf")
    displacement_spec = parse_xdmf_timeseries(visualization / "displacement.xdmf")
    _validate_series_match(velocity_spec, displacement_spec)

    velocity = load_timeseries_array(velocity_spec)
    displacement = load_timeseries_array(displacement_spec)
    if velocity.ndim != 3:
        raise ValueError(
            f"Expected velocity snapshots with shape (steps, nodes, components), got {velocity.shape}"
        )
    if displacement.ndim != 3:
        raise ValueError(
            f"Expected displacement snapshots with shape (steps, nodes, components), got {displacement.shape}"
        )

    node_sets = _load_node_sets(mesh_map_path)
    fluid_nodes = node_sets["fluid_interface_nodes"]
    solid_nodes = node_sets["solid_interface_nodes"]

    arrays: dict[str, np.ndarray] = {
        "time": velocity_spec.times,
        "U": flatten_vector_snapshots(velocity, fluid_nodes),
        "Df": flatten_vector_snapshots(displacement, fluid_nodes),
        "Ds": flatten_vector_snapshots(displacement, solid_nodes),
        "Vs": flatten_vector_snapshots(velocity, solid_nodes),
        "fluid_interface_nodes": fluid_nodes,
        "solid_interface_nodes": solid_nodes,
        "interface_nodes": node_sets["interface_nodes"],
        "fluid_only_nodes": node_sets["fluid_nodes"],
        "solid_only_nodes": node_sets["solid_nodes"],
    }

    if include_pressure:
        pressure_spec = parse_xdmf_timeseries(visualization / "pressure.xdmf")
        _validate_series_match(velocity_spec, pressure_spec)
        pressure = load_timeseries_array(pressure_spec)
        arrays["P"] = flatten_scalar_snapshots(pressure, fluid_nodes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)

    summary = {
        "results_folder": str(results_folder),
        "mesh_map_path": str(mesh_map_path),
        "output_path": str(output_path),
        "num_snapshots": int(velocity_spec.times.size),
        "time_start": float(velocity_spec.times[0]),
        "time_end": float(velocity_spec.times[-1]),
        "U_shape": list(arrays["U"].shape),
        "Df_shape": list(arrays["Df"].shape),
        "Ds_shape": list(arrays["Ds"].shape),
        "Vs_shape": list(arrays["Vs"].shape),
        "velocity_components": int(velocity.shape[2]),
        "displacement_components": int(displacement.shape[2]),
        "includes_pressure": bool(include_pressure),
        "notes": [
            "These are nodal visualization snapshots, not exact mixed-space FEniCS DoF vectors.",
            "Vector fields include all components present in the XDMF/HDF5 data.",
            "Fluid blocks include fluid-only plus interface nodes.",
            "Solid blocks include solid-only plus interface nodes.",
        ],
    }

    with open(output_path.with_suffix(".summary.json"), "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return summary


def main() -> None:
    args = parse_args()
    summary = export_snapshots(
        results_folder=Path(args.results_folder),
        mesh_map_path=Path(args.mesh_map),
        output_path=Path(args.output),
        include_pressure=args.include_pressure,
    )
    print(f"Wrote ROM snapshots to {args.output}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
