"""Plot FSI-vs-ROM fields on the exterior 3D mesh surface."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from turtleFSI.rom._shared.plot_common import _require_matplotlib, train_transition_count


COMPONENT_LABELS = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="Generic snapshot .npz with coordinates/topology and U/D/P.")
    parser.add_argument("--model-dir", required=True, help="Directory containing rom_model.npz and rom_summary.json.")
    parser.add_argument("--output", required=True, help="Output .png or .gif path.")
    parser.add_argument("--fields", nargs="+", default=["U", "D", "P"], help="ROM field order used during training.")
    parser.add_argument("--plot-field", default="U", help="Field to plot, usually U, D, or P.")
    parser.add_argument(
        "--component",
        default="magnitude",
        help="Component to color by: 0, 1, 2, x, y, z, or magnitude.",
    )
    parser.add_argument("--time", type=float, default=None, help="Target time for static PNG output.")
    parser.add_argument("--snapshot-index", type=int, default=-1, help="Snapshot index for static PNG output.")
    parser.add_argument("--animate", action="store_true", help="Write an animated GIF over all snapshots.")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--deform", action="store_true", help="Show FSI/ROM surfaces at their displaced coordinates.")
    parser.add_argument("--deformation-scale", type=float, default=1.0)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    return parser.parse_args()


def _load_summary(model_dir: Path) -> dict:
    with open(model_dir / "rom_summary.json", "r", encoding="ascii") as handle:
        return json.load(handle)


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _component_count(values: np.ndarray, num_nodes: int) -> int:
    if values.shape[0] == num_nodes:
        return 1
    if values.shape[0] % num_nodes != 0:
        raise ValueError(f"Field row count {values.shape[0]} is not divisible by {num_nodes} mesh nodes.")
    return int(values.shape[0] // num_nodes)


def _parse_component(component: str, component_count: int) -> int | None:
    if component == "magnitude":
        return None
    aliases = {"x": 0, "y": 1, "z": 2}
    index = aliases.get(component, None)
    if index is None:
        index = int(component)
    if index < 0 or index >= component_count:
        raise IndexError(f"component {component} is outside [0, {component_count})")
    return index


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
    series = _reconstruct(model, field, model[f"{field}_latent"])
    if rollout[field].shape[1] > 0:
        predicted = _reconstruct(model, field, rollout[field])
        series[:, train_count + 1 :] = predicted
    return series, train_count


def _boundary_faces(tetrahedra: np.ndarray) -> np.ndarray:
    face_to_count: dict[tuple[int, int, int], int] = defaultdict(int)
    for tet in np.asarray(tetrahedra, dtype=np.int64):
        faces = (
            (tet[0], tet[1], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]),
            (tet[1], tet[2], tet[3]),
        )
        for face in faces:
            face_to_count[tuple(sorted(int(node) for node in face))] += 1
    return np.asarray([face for face, count in face_to_count.items() if count == 1], dtype=np.int64)


def _field_nodes(values: np.ndarray, num_nodes: int) -> np.ndarray:
    components = _component_count(values, num_nodes)
    return values.reshape(num_nodes, components)


def _color_values(nodes: np.ndarray, component: int | None) -> np.ndarray:
    if component is None:
        return np.linalg.norm(nodes, axis=1) if nodes.shape[1] > 1 else np.abs(nodes[:, 0])
    return nodes[:, component]


def _surface_limits(*coordinates: np.ndarray) -> tuple[np.ndarray, float]:
    stacked = np.vstack(coordinates)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    return center, radius


def _apply_axis_limits(ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_surface(
    ax,
    coordinates: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    title: str,
    limits: tuple[float, float],
    cmap,
    norm,
    elev: float,
    azim: float,
    center: np.ndarray,
    radius: float,
) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.clear()
    face_values = values[faces].mean(axis=1)
    colors = cmap(norm(face_values))
    triangles = coordinates[faces]
    collection = Poly3DCollection(
        triangles,
        facecolors=colors,
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
    )
    ax.add_collection3d(collection)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    _apply_axis_limits(ax, center, radius)


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
    faces = _boundary_faces(topology)
    time = np.asarray(snapshots["time"], dtype=float)
    true_matrix = np.asarray(snapshots[args.plot_field], dtype=float)
    rom_matrix, train_count = _rom_series(model, summary, args.fields, args.plot_field)

    num_nodes = coordinates.shape[0]
    component_count = _component_count(true_matrix, num_nodes)
    component = _parse_component(str(args.component), component_count)
    component_name = "magnitude" if component is None else COMPONENT_LABELS[component]

    true_by_time = true_matrix.reshape(num_nodes, component_count, time.shape[0])
    rom_by_time = rom_matrix.reshape(num_nodes, component_count, time.shape[0])

    true_color_all = np.column_stack([
        _color_values(true_by_time[:, :, index], component) for index in range(time.shape[0])
    ])
    rom_color_all = np.column_stack([
        _color_values(rom_by_time[:, :, index], component) for index in range(time.shape[0])
    ])
    limits = (float(min(true_color_all.min(), rom_color_all.min())), float(max(true_color_all.max(), rom_color_all.max())))

    displacement_true = np.asarray(snapshots["D"], dtype=float).reshape(num_nodes, 3, time.shape[0])
    displacement_rom, _ = _rom_series(model, summary, args.fields, "D")
    displacement_rom = displacement_rom.reshape(num_nodes, 3, time.shape[0])

    plt, _ = _require_matplotlib()
    from matplotlib import colors as mcolors
    from matplotlib.cm import ScalarMappable

    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=limits[0], vmax=limits[1])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 5.5), constrained_layout=True)
    axes = (fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2, projection="3d"))

    def frame_coordinates(index: int) -> tuple[np.ndarray, np.ndarray]:
        if not args.deform:
            return coordinates, coordinates
        return (
            coordinates + args.deformation_scale * displacement_true[:, :, index],
            coordinates + args.deformation_scale * displacement_rom[:, :, index],
        )

    def draw(index: int) -> None:
        true_coords, rom_coords = frame_coordinates(index)
        center, radius = _surface_limits(true_coords, rom_coords)
        phase = "POD reconstruction" if index <= train_count else "ROM rollout"
        _draw_surface(
            axes[0],
            true_coords,
            faces,
            true_color_all[:, index],
            "FSI",
            limits,
            cmap,
            norm,
            args.elev,
            args.azim,
            center,
            radius,
        )
        _draw_surface(
            axes[1],
            rom_coords,
            faces,
            rom_color_all[:, index],
            "ROM",
            limits,
            cmap,
            norm,
            args.elev,
            args.azim,
            center,
            radius,
        )
        fig.suptitle(
            f"{args.plot_field} {component_name} on 3D surface at t = {time[index]:.6g} s "
            f"({phase}; training cutoff snapshot {train_count})"
        )

    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.75)
    colorbar.set_label(f"{args.plot_field} {component_name}")

    if args.animate:
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Animated GIF output requires matplotlib animation support and pillow.") from exc

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
        draw(index)
        fig.savefig(output, dpi=220, bbox_inches="tight")

    print(f"Wrote 3D surface plot to {output}")


if __name__ == "__main__":
    main()
