"""Plot component-wise FOM-vs-ROM time series for U, D, and P."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._shared.plot_common import _require_matplotlib, train_transition_count


FIELD_TITLES = {"U": "Fluid velocity", "D": "Solid displacement", "P": "Pressure"}
COMPONENT_LABELS = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="Generic snapshot .npz with U/D/P arrays.")
    parser.add_argument("--model-dir", required=True, help="Directory containing rom_model.npz and rom_summary.json.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--fields", nargs="+", default=["U", "D", "P"], help="ROM field order used during training.")
    parser.add_argument("--plot-fields", nargs="+", default=["U", "D", "P"], help="Fields to plot.")
    parser.add_argument("--title", default=None, help="Figure title. Defaults to a FOM-vs-ROM title.")
    parser.add_argument("--case-name", default="Test Cylinder", help="Case name shown in the figure title.")
    parser.add_argument("--fom-label", default="FOM reference", help="Legend label for the full-order result.")
    parser.add_argument("--rom-label", default="POD ROM prediction", help="Legend label for the ROM result.")
    parser.add_argument("--node-index", type=int, default=None, help="Visualization node index to plot.")
    parser.add_argument(
        "--target",
        nargs="+",
        type=float,
        default=None,
        help="Select nearest node to this coordinate. Use 2 or 3 values.",
    )
    return parser.parse_args()


def _load_summary(model_dir: Path) -> dict:
    with open(model_dir / "rom_summary.json", "r", encoding="ascii") as handle:
        return json.load(handle)


def _resolve_model_dir(model_dir: Path, snapshots_path: Path) -> Path:
    candidates = [
        model_dir,
        model_dir / "model",
        model_dir.parent if model_dir.name == "model" else model_dir.parent / "model",
        snapshots_path.parent / "model",
    ]
    for candidate in candidates:
        if (candidate / "rom_model.npz").exists() and (candidate / "rom_summary.json").exists():
            return candidate
    command = (
        "python -m turtleFSI.rom._pod.build "
        f"--snapshots {snapshots_path} "
        f"--output-dir {snapshots_path.parent / 'model'} "
        "--fields U D P --max-modes 20 --train-fraction 0.8"
    )
    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find rom_model.npz and rom_summary.json for the POD baseline.\n"
        f"Checked:\n  {checked}\n\n"
        "Build the model first with:\n"
        f"  {command}"
    )


def _component_count(values: np.ndarray, num_nodes: int) -> int:
    if values.shape[0] == num_nodes:
        return 1
    if values.shape[0] % num_nodes != 0:
        raise ValueError(f"{values.shape[0]} rows is not divisible by {num_nodes} nodes")
    return int(values.shape[0] // num_nodes)


def _field_node_count(values: np.ndarray, coordinate_count: int | None) -> int:
    if coordinate_count is not None and values.shape[0] % coordinate_count == 0:
        return coordinate_count
    if values.shape[0] % 3 == 0:
        return values.shape[0] // 3
    return values.shape[0]


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


def _rom_series(model: np.lib.npyio.NpzFile, summary: dict, fields: list[str], field: str) -> tuple[np.ndarray, int]:
    train_count, rollout = _rollout_latent(model, summary, fields)
    series = model[f"{field}_mean"] + model[f"{field}_basis"] @ model[f"{field}_latent"]
    if rollout[field].shape[1] > 0:
        predicted = model[f"{field}_mean"] + model[f"{field}_basis"] @ rollout[field]
        series[:, train_count + 1 :] = predicted
    return series, train_count


def _select_field_node(
    values: np.ndarray,
    coordinates: np.ndarray | None,
    field_name: str,
    num_nodes: int,
    component_count: int,
    node_index: int | None,
    target: list[float] | None,
) -> tuple[int, np.ndarray | None]:
    if node_index is not None:
        selected = int(node_index)
    elif target is not None:
        if coordinates is None:
            raise ValueError(
                f"--target cannot be used for field {field_name}, because its snapshots do not "
                "match the exported visualization coordinates."
            )
        target_values = np.asarray(target, dtype=float)
        coords = coordinates[:, : target_values.size]
        selected = int(np.argmin(np.linalg.norm(coords - target_values, axis=1)))
    else:
        reshaped = values.reshape(num_nodes, component_count, values.shape[1])
        if component_count == 1:
            motion = np.abs(reshaped[:, 0, :] - reshaped[:, 0, :1]).max(axis=1)
        else:
            motion = np.linalg.norm(reshaped - reshaped[:, :, :1], axis=1).max(axis=1)
        selected = int(np.argmax(motion))

    if selected < 0 or selected >= num_nodes:
        raise IndexError(f"node index {selected} is outside [0, {num_nodes})")
    coord = coordinates[selected] if coordinates is not None else None
    return selected, coord


def _component_label(index: int, count: int) -> str:
    if count == 1:
        return "scalar"
    if index < len(COMPONENT_LABELS):
        return COMPONENT_LABELS[index]
    return str(index)


def main() -> None:
    args = parse_args()
    snapshots_path = Path(args.snapshots)
    snapshots = np.load(snapshots_path)
    model_dir = _resolve_model_dir(Path(args.model_dir), snapshots_path)
    model = np.load(model_dir / "rom_model.npz")
    summary = _load_summary(model_dir)

    coordinates = np.asarray(snapshots["coordinates"], dtype=float) if "coordinates" in snapshots.files else None
    coordinate_count = None if coordinates is None else int(coordinates.shape[0])

    plot_fields = [field for field in args.plot_fields if field in snapshots.files]
    if not plot_fields:
        raise ValueError("No requested plot fields are present in snapshots")

    time = np.asarray(snapshots["time"], dtype=float)

    series = {}
    component_counts = {}
    node_counts = {}
    selected_nodes = {}
    selected_coords = {}
    train_count = 0
    for field in plot_fields:
        true_values = np.asarray(snapshots[field], dtype=float)
        field_num_nodes = _field_node_count(true_values, coordinate_count)
        field_coordinates = coordinates if coordinate_count == field_num_nodes else None
        component_counts[field] = _component_count(true_values, field_num_nodes)
        node_counts[field] = field_num_nodes
        selected_nodes[field], selected_coords[field] = _select_field_node(
            values=true_values,
            coordinates=field_coordinates,
            field_name=field,
            num_nodes=field_num_nodes,
            component_count=component_counts[field],
            node_index=args.node_index,
            target=args.target,
        )
        rom_values, train_count = _rom_series(model, summary, args.fields, field)
        series[field] = (true_values, rom_values)

    row_specs = [
        (field, component)
        for field in plot_fields
        for component in range(component_counts[field])
    ]

    plt, _ = _require_matplotlib()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig_height = max(3.0, 2.3 * len(row_specs))
    fig, axes = plt.subplots(len(row_specs), 1, figsize=(11, fig_height), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (field, component) in zip(axes, row_specs):
        true_values, rom_values = series[field]
        count = component_counts[field]
        node_index = selected_nodes[field]
        row = node_index * count + component
        label = _component_label(component, count)
        ax.plot(time, true_values[row], color="#1f77b4", linewidth=1.8, label=args.fom_label)
        ax.plot(time, rom_values[row], color="#d62728", linewidth=1.8, linestyle="--", label=args.rom_label)
        ax.axvline(time[train_count], color="black", linestyle=":", linewidth=1.2)
        ax.set_ylabel(f"{FIELD_TITLES.get(field, field)} {label}\nnode {node_index}")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, loc="best")

    axes[-1].set_xlabel("Time [s]")
    title = args.title or f"{args.case_name}: FOM reference vs POD ROM prediction component time series"
    field_nodes = ", ".join(f"{FIELD_TITLES.get(field, field)} node {selected_nodes[field]}" for field in plot_fields)
    coord_fields = [
        f"{FIELD_TITLES.get(field, field)} near ({', '.join(f'{value:.6g}' for value in selected_coords[field])})"
        for field in plot_fields
        if selected_coords[field] is not None
    ]
    coord_text = "" if not coord_fields else "\n" + "; ".join(coord_fields)
    fig.suptitle(f"{title}\n{field_nodes}{coord_text}")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote all-field component time-series plot to {output}")


if __name__ == "__main__":
    main()
