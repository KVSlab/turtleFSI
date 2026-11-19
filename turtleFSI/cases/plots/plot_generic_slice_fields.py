"""Plot or animate slice-contour fields for generic 3D non-intrusive ROMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._shared.plot_common import _require_matplotlib, train_transition_count


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
COMPONENT_LABELS = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="Generic snapshot .npz with coordinates and U/D/P arrays.")
    parser.add_argument("--model-dir", required=True, help="Directory containing rom_model.npz and rom_summary.json.")
    parser.add_argument("--output", required=True, help="Output .png or .gif path.")
    parser.add_argument("--fields", nargs="+", default=["U", "D", "P"], help="ROM field order used during training.")
    parser.add_argument("--plot-field", default="U", help="Field to plot, usually U, D, or P.")
    parser.add_argument("--component", type=int, default=0, help="Component index: 0=x, 1=y, 2=z.")
    parser.add_argument("--slice-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--slice-value", type=float, default=None, help="Slice coordinate. Defaults to mesh midpoint.")
    parser.add_argument(
        "--slice-tolerance",
        type=float,
        default=None,
        help="Half-thickness of nodes selected around the slice. Defaults to 5% of axis range.",
    )
    parser.add_argument("--time", type=float, default=None, help="Target time for static PNG output.")
    parser.add_argument("--snapshot-index", type=int, default=-1, help="Snapshot index for static PNG output.")
    parser.add_argument("--animate", action="store_true", help="Write an animated GIF over all snapshots.")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--levels", type=int, default=30)
    return parser.parse_args()


def _load_summary(model_dir: Path) -> dict:
    with open(model_dir / "rom_summary.json", "r", encoding="ascii") as handle:
        return json.load(handle)


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _component_count(values: np.ndarray, num_nodes: int) -> int:
    if values.shape[0] % num_nodes != 0:
        raise ValueError(f"Field row count {values.shape[0]} is not divisible by {num_nodes} mesh nodes.")
    return int(values.shape[0] // num_nodes)


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


def _rom_series(model: np.lib.npyio.NpzFile, summary: dict, fields: list[str], field: str) -> tuple[np.ndarray, int]:
    train_count, rollout = _rollout_latent(model, summary, fields)
    reconstruction = _reconstruct(model, field, model[f"{field}_latent"])
    values = reconstruction.copy()
    if rollout[field].shape[1] > 0:
        predicted = _reconstruct(model, field, rollout[field])
        values[:, train_count + 1 :] = predicted
    return values, train_count


def _slice_indices(
    coordinates: np.ndarray,
    axis: str,
    slice_value: float | None,
    slice_tolerance: float | None,
) -> tuple[np.ndarray, float, float]:
    axis_index = AXIS_TO_INDEX[axis]
    values = coordinates[:, axis_index]
    if slice_value is None:
        slice_value = 0.5 * (float(values.min()) + float(values.max()))
    if slice_tolerance is None:
        slice_tolerance = 0.05 * float(values.max() - values.min())
    indices = np.flatnonzero(np.abs(values - slice_value) <= slice_tolerance)
    if indices.size < 4:
        nearest = np.argsort(np.abs(values - slice_value))[: max(4, min(80, values.size))]
        indices = np.asarray(nearest, dtype=np.int64)
    return indices, float(slice_value), float(slice_tolerance)


def _project_coordinates(coordinates: np.ndarray, axis: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    if axis == "x":
        return coordinates[:, 1], coordinates[:, 2], "y", "z"
    if axis == "y":
        return coordinates[:, 0], coordinates[:, 2], "x", "z"
    return coordinates[:, 0], coordinates[:, 1], "x", "y"


def _make_plot(
    fig,
    axes,
    triangulation,
    true_values: np.ndarray,
    rom_values: np.ndarray,
    limits: tuple[float, float],
    title: str,
    labels: tuple[str, str],
    levels: int,
) -> None:
    for ax in axes:
        ax.clear()
    plots = ((axes[0], true_values, "FSI"), (axes[1], rom_values, "ROM"))
    for ax, values, name in plots:
        contour = ax.tricontourf(
            triangulation,
            values,
            levels=levels,
            cmap="RdBu_r",
            vmin=limits[0],
            vmax=limits[1],
        )
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    fig.suptitle(title)
    return contour


def main() -> None:
    args = parse_args()
    snapshots = np.load(args.snapshots)
    if "coordinates" not in snapshots.files:
        raise KeyError(
            "The snapshots file does not contain coordinates. Rerun "
            "export_visualization.py with the updated exporter."
        )

    model_dir = Path(args.model_dir)
    model = np.load(model_dir / "rom_model.npz")
    summary = _load_summary(model_dir)
    time = np.asarray(snapshots["time"], dtype=float)
    coordinates = np.asarray(snapshots["coordinates"], dtype=float)
    true_matrix = np.asarray(snapshots[args.plot_field], dtype=float)
    component_count = _component_count(true_matrix, coordinates.shape[0])
    if args.component < 0 or args.component >= component_count:
        raise IndexError(f"component {args.component} is outside [0, {component_count})")

    rom_matrix, train_count = _rom_series(model, summary, args.fields, args.plot_field)
    component_label = COMPONENT_LABELS[args.component] if args.component < len(COMPONENT_LABELS) else str(args.component)

    slice_nodes, slice_value, slice_tolerance = _slice_indices(
        coordinates=coordinates,
        axis=args.slice_axis,
        slice_value=args.slice_value,
        slice_tolerance=args.slice_tolerance,
    )
    slice_coordinates = coordinates[slice_nodes]
    plot_x, plot_y, x_label, y_label = _project_coordinates(slice_coordinates, args.slice_axis)

    plt, mtri = _require_matplotlib()
    triangulation = mtri.Triangulation(plot_x, plot_y)

    true_component = true_matrix[args.component::component_count][slice_nodes]
    rom_component = rom_matrix[args.component::component_count][slice_nodes]
    limits = (
        float(min(true_component.min(), rom_component.min())),
        float(max(true_component.max(), rom_component.max())),
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    def draw(index: int):
        phase = "POD reconstruction" if index <= train_count else "ROM rollout"
        title = (
            f"{args.plot_field}{component_label} slice comparison at t = {time[index]:.6g} s "
            f"({phase}; {args.slice_axis}={slice_value:.6g} +/- {slice_tolerance:.3g})"
        )
        contour = _make_plot(
            fig=fig,
            axes=axes,
            triangulation=triangulation,
            true_values=true_component[:, index],
            rom_values=rom_component[:, index],
            limits=limits,
            title=title,
            labels=(x_label, y_label),
            levels=args.levels,
        )
        return contour

    if args.animate:
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Animated GIF output requires matplotlib animation support and pillow.") from exc

        contour = draw(0)
        colorbar = fig.colorbar(contour, ax=axes, shrink=0.82)
        colorbar.set_label(f"{args.plot_field}{component_label}")

        def update(index: int):
            draw(index)
            return []

        animation = FuncAnimation(fig, update, frames=range(time.shape[0]), interval=1000 / args.fps, blit=False)
        animation.save(output, writer=PillowWriter(fps=args.fps))
    else:
        index = _nearest_index(time, args.time) if args.time is not None else args.snapshot_index
        if index < 0:
            index = time.shape[0] + index
        if index < 0 or index >= time.shape[0]:
            raise IndexError(f"snapshot index {index} is outside [0, {time.shape[0]})")
        contour = draw(index)
        colorbar = fig.colorbar(contour, ax=axes, shrink=0.82)
        colorbar.set_label(f"{args.plot_field}{component_label}")
        fig.savefig(output, dpi=220, bbox_inches="tight")
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
