"""Export generic non-intrusive ROM fields to VTK for 3D visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from turtleFSI.rom._shared.plot_common import train_transition_count


VTK_TETRA = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="Generic snapshot .npz with coordinates/topology and U/D/P.")
    parser.add_argument("--model-dir", required=True, help="Directory containing rom_model.npz and rom_summary.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for .vtu files and the .pvd time collection.")
    parser.add_argument("--fields", nargs="+", default=["U", "D", "P"], help="ROM field order used during training.")
    parser.add_argument(
        "--export-fields",
        nargs="+",
        default=["U", "D", "P"],
        help="Fields to write to VTK.",
    )
    return parser.parse_args()


def _load_summary(model_dir: Path) -> dict:
    with open(model_dir / "rom_summary.json", "r", encoding="ascii") as handle:
        return json.load(handle)


def _rollout_latent(model: np.lib.npyio.NpzFile, summary: dict, fields: list[str]) -> tuple[int, dict[str, np.ndarray]]:
    time = model["time"]
    train_count = train_transition_count(int(time.shape[0]), float(summary["train_fraction"]))
    latent = np.vstack([model[f"{field}_latent"] for field in fields])
    current = latent[:, train_count : train_count + 1]
    predicted = []
    for _ in range(time.shape[0] - train_count - 1):
        current = model["A"] @ current + model["b"]
        predicted.append(current[:, 0])
    predicted_latent = np.asarray(predicted, dtype=float).T

    blocks = {}
    cursor = 0
    for field in fields:
        rows = model[f"{field}_basis"].shape[1]
        blocks[field] = predicted_latent[cursor : cursor + rows]
        cursor += rows
    return train_count, blocks


def _reconstruct(model: np.lib.npyio.NpzFile, field: str, latent: np.ndarray) -> np.ndarray:
    return model[f"{field}_mean"] + model[f"{field}_basis"] @ latent


def _rom_series(model: np.lib.npyio.NpzFile, summary: dict, fields: list[str], field: str) -> np.ndarray:
    train_count, rollout = _rollout_latent(model, summary, fields)
    series = _reconstruct(model, field, model[f"{field}_latent"])
    if rollout[field].shape[1] > 0:
        predicted = _reconstruct(model, field, rollout[field])
        series[:, train_count + 1 :] = predicted
    return series


def _field_to_nodes(values: np.ndarray, num_nodes: int) -> tuple[np.ndarray, int]:
    if values.shape[0] == num_nodes:
        return values.reshape(num_nodes, 1), 1
    if values.shape[0] % num_nodes != 0:
        raise ValueError(f"Field row count {values.shape[0]} is not divisible by node count {num_nodes}.")
    components = values.shape[0] // num_nodes
    return values.reshape(num_nodes, components), components


def _array_text(values: np.ndarray) -> str:
    flat = np.asarray(values).reshape(-1)
    if np.issubdtype(flat.dtype, np.integer):
        return " ".join(str(int(value)) for value in flat)
    return " ".join(f"{float(value):.16g}" for value in flat)


def _data_array(name: str, values: np.ndarray, vtk_type: str, components: int | None = None) -> str:
    component_attr = "" if components is None or components == 1 else f' NumberOfComponents="{components}"'
    return (
        f'        <DataArray type="{vtk_type}" Name="{escape(name)}"{component_attr} format="ascii">\n'
        f"          {_array_text(values)}\n"
        "        </DataArray>\n"
    )


def _write_vtu(
    path: Path,
    coordinates: np.ndarray,
    topology: np.ndarray,
    point_arrays: dict[str, tuple[np.ndarray, int]],
) -> None:
    num_points = coordinates.shape[0]
    num_cells = topology.shape[0]
    connectivity = topology.astype(np.int64)
    offsets = np.arange(1, num_cells + 1, dtype=np.int64) * connectivity.shape[1]
    if connectivity.shape[1] != 4:
        raise ValueError("This exporter currently expects tetrahedral topology with 4 nodes per cell.")
    cell_types = np.full(num_cells, VTK_TETRA, dtype=np.uint8)

    parts = [
        '<?xml version="1.0"?>\n',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n',
        "  <UnstructuredGrid>\n",
        f'    <Piece NumberOfPoints="{num_points}" NumberOfCells="{num_cells}">\n',
        "      <PointData>\n",
    ]
    for name, (values, components) in point_arrays.items():
        parts.append(_data_array(name, values, "Float64", components))
    parts.extend(
        [
            "      </PointData>\n",
            "      <CellData>\n",
            "      </CellData>\n",
            "      <Points>\n",
            _data_array("Points", coordinates, "Float64", 3),
            "      </Points>\n",
            "      <Cells>\n",
            _data_array("connectivity", connectivity, "Int64"),
            _data_array("offsets", offsets, "Int64"),
            _data_array("types", cell_types, "UInt8"),
            "      </Cells>\n",
            "    </Piece>\n",
            "  </UnstructuredGrid>\n",
            "</VTKFile>\n",
        ]
    )
    path.write_text("".join(parts), encoding="ascii")


def _write_pvd(path: Path, files: list[Path], time: np.ndarray) -> None:
    lines = [
        '<?xml version="1.0"?>\n',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n',
        "  <Collection>\n",
    ]
    for file_path, value in zip(files, time):
        lines.append(
            f'    <DataSet timestep="{float(value):.16g}" group="" part="0" file="{escape(file_path.name)}"/>\n'
        )
    lines.extend(["  </Collection>\n", "</VTKFile>\n"])
    path.write_text("".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    snapshots = np.load(args.snapshots)
    if "coordinates" not in snapshots.files or "topology" not in snapshots.files:
        raise KeyError(
            "The snapshots file needs coordinates and topology. Rerun "
            "export_visualization.py with the updated exporter."
        )

    model_dir = Path(args.model_dir)
    model = np.load(model_dir / "rom_model.npz")
    summary = _load_summary(model_dir)
    coordinates = np.asarray(snapshots["coordinates"], dtype=float)
    topology = np.asarray(snapshots["topology"], dtype=np.int64)
    time = np.asarray(snapshots["time"], dtype=float)
    num_nodes = coordinates.shape[0]

    rom_by_field = {
        field: _rom_series(model, summary, args.fields, field)
        for field in args.export_fields
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtu_files = []
    for index in range(time.shape[0]):
        arrays: dict[str, tuple[np.ndarray, int]] = {}
        for field in args.export_fields:
            true_values, components = _field_to_nodes(np.asarray(snapshots[field], dtype=float)[:, index], num_nodes)
            rom_values, _ = _field_to_nodes(rom_by_field[field][:, index], num_nodes)
            arrays[f"{field}_FSI"] = (true_values, components)
            arrays[f"{field}_ROM"] = (rom_values, components)
            arrays[f"{field}_error"] = (rom_values - true_values, components)
        vtu_path = output_dir / f"rom_fields_{index:04d}.vtu"
        _write_vtu(vtu_path, coordinates, topology, arrays)
        vtu_files.append(vtu_path)

    pvd_path = output_dir / "rom_fields.pvd"
    _write_pvd(pvd_path, vtu_files, time)
    print(f"Wrote VTK time series to {pvd_path}")


if __name__ == "__main__":
    main()
