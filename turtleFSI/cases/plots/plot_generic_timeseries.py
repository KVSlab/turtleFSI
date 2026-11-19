"""Plot FSI-vs-ROM time series for generic non-intrusive ROM snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._shared.plot_common import _require_matplotlib, train_transition_count


COMPONENT_LABELS = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="Generic snapshot .npz with U/D/P arrays.")
    parser.add_argument("--model-dir", required=True, help="Directory containing rom_model.npz and rom_summary.json.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--fields", nargs="+", default=["U", "D", "P"], help="ROM field order used during training.")
    parser.add_argument("--plot-field", default="U", help="Field to plot, usually U or D.")
    parser.add_argument("--components", nargs="+", type=int, default=[0, 1], help="Component indices to plot.")
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


def _reconstruct_field(model: np.lib.npyio.NpzFile, field: str, latent: np.ndarray) -> np.ndarray:
    return model[f"{field}_mean"] + model[f"{field}_basis"] @ latent


def _component_count(snapshots: np.ndarray, coordinates: np.ndarray | None) -> int:
    if coordinates is not None and coordinates.ndim == 2 and coordinates.shape[0] > 0:
        return int(snapshots.shape[0] // coordinates.shape[0])
    if snapshots.shape[0] % 3 == 0:
        return 3
    if snapshots.shape[0] % 2 == 0:
        return 2
    return 1


def _select_node(
    snapshots: np.lib.npyio.NpzFile,
    plot_values: np.ndarray,
    component_count: int,
    node_index: int | None,
    target: list[float] | None,
) -> tuple[int, np.ndarray | None]:
    coordinates = np.asarray(snapshots["coordinates"], dtype=float) if "coordinates" in snapshots.files else None
    num_nodes = plot_values.shape[0] // component_count

    if node_index is not None:
        selected = int(node_index)
    elif target is not None:
        if coordinates is None:
            raise ValueError("--target requires coordinates in the snapshots file. Rerun the generic exporter.")
        target_values = np.asarray(target, dtype=float)
        coords = coordinates[:, : target_values.size]
        selected = int(np.argmin(np.linalg.norm(coords - target_values, axis=1)))
    elif "D" in snapshots.files:
        displacement = np.asarray(snapshots["D"], dtype=float)
        disp_components = _component_count(displacement, coordinates)
        reshaped = displacement.reshape(num_nodes, disp_components, displacement.shape[1])
        motion = np.linalg.norm(reshaped - reshaped[:, :, :1], axis=1).max(axis=1)
        selected = int(np.argmax(motion))
    else:
        selected = 0

    if selected < 0 or selected >= num_nodes:
        raise IndexError(f"node index {selected} is outside [0, {num_nodes})")
    coord = coordinates[selected] if coordinates is not None else None
    return selected, coord


def main() -> None:
    args = parse_args()
    snapshots = np.load(args.snapshots)
    model_dir = Path(args.model_dir)
    model = np.load(model_dir / "rom_model.npz")
    summary = _load_summary(model_dir)

    if args.plot_field not in snapshots.files:
        raise KeyError(f"{args.plot_field} is not in {args.snapshots}")

    true_values = np.asarray(snapshots[args.plot_field], dtype=float)
    coordinates = np.asarray(snapshots["coordinates"], dtype=float) if "coordinates" in snapshots.files else None
    component_count = _component_count(true_values, coordinates)
    node_index, coord = _select_node(
        snapshots=snapshots,
        plot_values=true_values,
        component_count=component_count,
        node_index=args.node_index,
        target=args.target,
    )

    train_count, rollout = _rollout_latent(model, summary, args.fields)
    pred_values = _reconstruct_field(model, args.plot_field, rollout[args.plot_field])
    time = np.asarray(snapshots["time"], dtype=float)
    pred_time = time[train_count + 1 :]

    plt, _ = _require_matplotlib()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(args.components), 1, figsize=(10, 3.2 * len(args.components)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, component in zip(axes, args.components):
        if component < 0 or component >= component_count:
            raise IndexError(f"component {component} is outside [0, {component_count})")
        row = node_index * component_count + component
        label = COMPONENT_LABELS[component] if component < len(COMPONENT_LABELS) else str(component)
        ax.plot(time, true_values[row], color="#1b4965", linewidth=2, label="FSI")
        ax.plot(pred_time, pred_values[row], color="#c1121f", linewidth=2, linestyle="--", label="ROM")
        ax.axvline(time[train_count], color="black", linestyle=":", linewidth=1.5, label="Training cutoff")
        ax.set_ylabel(f"{args.plot_field} {label}")
        ax.legend(frameon=False)

    axes[-1].set_xlabel("Time [s]")
    coord_text = "" if coord is None else f" near ({', '.join(f'{value:.6g}' for value in coord)})"
    fig.suptitle(f"{args.plot_field} comparison at node {node_index}{coord_text}")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
