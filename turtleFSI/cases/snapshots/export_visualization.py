"""Export generic ROM snapshot matrices from turtleFSI visualization output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from turtleFSI.cases.snapshots.export_tf_fsi import (
    _parse_hdf_dataitem_text,
    _require_h5py,
    _strip_namespace,
    _validate_series_match,
    load_timeseries_array,
    parse_xdmf_timeseries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-folder",
        required=True,
        help="Path to a turtleFSI results directory containing Visualization/.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz file with ROM-ready snapshot matrices.",
    )
    parser.add_argument(
        "--include-pressure",
        action="store_true",
        help="Also export nodal pressure snapshots as P.",
    )
    return parser.parse_args()


def flatten_vector_components(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"Expected vector time series with shape (steps, nodes, components), got {array.shape}")
    num_snapshots = array.shape[0]
    return array.transpose(1, 2, 0).reshape(array.shape[1] * array.shape[2], num_snapshots)


def flatten_scalar_components(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"Expected scalar time series with shape (steps, nodes), got {array.shape}")
    return array.T


def load_mesh_arrays(xdmf_path: Path) -> dict[str, np.ndarray]:
    h5py = _require_h5py()
    import xml.etree.ElementTree as ET

    tree = ET.parse(xdmf_path)
    root = tree.getroot()
    first_grid = None
    for element in root.iter():
        if _strip_namespace(element.tag) == "Grid" and element.attrib.get("GridType") == "Uniform":
            first_grid = element
            break
    if first_grid is None:
        return {}

    arrays: dict[str, np.ndarray] = {}
    for child in first_grid:
        tag = _strip_namespace(child.tag)
        key = None
        if tag == "Geometry":
            key = "coordinates"
        elif tag == "Topology":
            key = "topology"
        if key is None:
            continue
        data_item = next((node for node in child if _strip_namespace(node.tag) == "DataItem"), None)
        if data_item is None:
            continue
        file_name, dataset_path = _parse_hdf_dataitem_text(data_item)
        with h5py.File(xdmf_path.parent / file_name, "r") as handle:
            arrays[key] = np.asarray(handle[dataset_path])
    return arrays


def export_visualization_snapshots(
    results_folder: Path,
    output_path: Path,
    include_pressure: bool,
) -> dict[str, Any]:
    visualization = results_folder / "Visualization"
    velocity_path = visualization / "velocity.xdmf"
    displacement_path = visualization / "displacement.xdmf"
    pressure_path = visualization / "pressure.xdmf"
    missing = [path for path in (velocity_path, displacement_path) if not path.exists()]
    if include_pressure and not pressure_path.exists():
        missing.append(pressure_path)
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Could not find required turtleFSI visualization files.\n"
            f"results_folder: {results_folder}\n"
            "Expected files:\n"
            f"{missing_text}\n"
            "If you used a shell variable such as $RESULTS, check that it is set "
            "in the same shell before running this command."
        )

    velocity_spec = parse_xdmf_timeseries(velocity_path)
    displacement_spec = parse_xdmf_timeseries(displacement_path)
    _validate_series_match(velocity_spec, displacement_spec)

    velocity = load_timeseries_array(velocity_spec)
    displacement = load_timeseries_array(displacement_spec)

    arrays: dict[str, np.ndarray] = {
        "time": velocity_spec.times,
        "U": flatten_vector_components(velocity),
        "D": flatten_vector_components(displacement),
    }
    arrays.update(load_mesh_arrays(visualization / "velocity.xdmf"))

    if include_pressure:
        pressure_spec = parse_xdmf_timeseries(pressure_path)
        _validate_series_match(velocity_spec, pressure_spec)
        arrays["P"] = flatten_scalar_components(load_timeseries_array(pressure_spec))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)

    summary = {
        "results_folder": str(results_folder),
        "output_path": str(output_path),
        "num_snapshots": int(velocity_spec.times.size),
        "time_start": float(velocity_spec.times[0]),
        "time_end": float(velocity_spec.times[-1]),
        "U_shape": list(arrays["U"].shape),
        "D_shape": list(arrays["D"].shape),
        "includes_pressure": bool(include_pressure),
        "notes": [
            "These are nodal visualization snapshots, not exact mixed-space FEniCS DoF vectors.",
            "Vector fields include all components present in the XDMF/HDF5 data.",
            "Use exact DVP snapshots for intrusive POD-Galerkin ROMs.",
        ],
    }
    if "P" in arrays:
        summary["P_shape"] = list(arrays["P"].shape)
    if "coordinates" in arrays:
        summary["coordinates_shape"] = list(arrays["coordinates"].shape)
    if "topology" in arrays:
        summary["topology_shape"] = list(arrays["topology"].shape)

    with open(output_path.with_suffix(".summary.json"), "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> None:
    args = parse_args()
    summary = export_visualization_snapshots(
        results_folder=Path(args.results_folder),
        output_path=Path(args.output),
        include_pressure=args.include_pressure,
    )
    print(f"Wrote visualization ROM snapshots to {args.output}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
