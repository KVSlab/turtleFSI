"""Plot FOM-vs-ROM 3D surface comparisons for U, D, and P together."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._shared.plot_common import _require_matplotlib, train_transition_count


FIELD_TITLES = {"U": "Fluid velocity", "D": "Solid displacement", "P": "Pressure"}
COMPONENT_TITLES = {"magnitude": "magnitude", "x": "x-component", "y": "y-component", "z": "z-component"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fields", nargs="+", default=["U", "D", "P"], help="ROM field order used during training.")
    parser.add_argument("--plot-fields", nargs="+", default=["U", "D", "P"], help="Fields to display.")
    parser.add_argument("--component", default="magnitude", help="x/y/z/0/1/2 or magnitude")
    parser.add_argument("--title", default=None, help="Figure title. Defaults to a FOM-vs-ROM title.")
    parser.add_argument("--case-name", default="Test Cylinder", help="Case name shown in the figure title.")
    parser.add_argument("--fom-label", default="FOM reference", help="Column title for the full-order result.")
    parser.add_argument("--rom-label", default="POD ROM prediction", help="Column title for the ROM result.")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--snapshot-index", type=int, default=-1)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth snapshot when writing an animation.")
    parser.add_argument("--deform", action="store_true")
    parser.add_argument("--deformation-scale", type=float, default=1.0)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-60.0)
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


def _boundary_faces(tetrahedra: np.ndarray) -> np.ndarray:
    face_count: dict[tuple[int, int, int], int] = defaultdict(int)
    for tet in np.asarray(tetrahedra, dtype=np.int64):
        for face in (
            (tet[0], tet[1], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]),
            (tet[1], tet[2], tet[3]),
        ):
            face_count[tuple(sorted(int(node) for node in face))] += 1
    return np.asarray([face for face, count in face_count.items() if count == 1], dtype=np.int64)


def _component_count(values: np.ndarray, num_nodes: int) -> int:
    if values.shape[0] == num_nodes:
        return 1
    if values.shape[0] % num_nodes != 0:
        raise ValueError(f"{values.shape[0]} rows is not divisible by {num_nodes} nodes")
    return int(values.shape[0] // num_nodes)


def _component_index(component: str, count: int) -> int | None:
    if count == 1:
        return 0
    if component == "magnitude":
        return None
    aliases = {"x": 0, "y": 1, "z": 2}
    index = aliases.get(component, None)
    if index is None:
        index = int(component)
    if index < 0 or index >= count:
        raise IndexError(f"component {component} outside [0, {count})")
    return index


def _color_values(nodes: np.ndarray, component: int | None) -> np.ndarray:
    if component is None:
        return np.linalg.norm(nodes, axis=1) if nodes.shape[1] > 1 else nodes[:, 0]
    return nodes[:, component]


def _component_title(component: str, count: int) -> str:
    if count == 1:
        return "scalar value"
    return COMPONENT_TITLES.get(component, f"component {component}")


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


def _limits(*values: np.ndarray) -> tuple[float, float]:
    stacked = np.concatenate([value.reshape(-1) for value in values])
    low = float(stacked.min())
    high = float(stacked.max())
    if np.isclose(low, high):
        pad = max(abs(low), 1.0) * 1e-8
        return low - pad, high + pad
    return low, high


def _axis_limits(*coordinates: np.ndarray) -> tuple[np.ndarray, float]:
    stacked = np.vstack(coordinates)
    center = 0.5 * (stacked.min(axis=0) + stacked.max(axis=0))
    radius = 0.5 * float(np.max(stacked.max(axis=0) - stacked.min(axis=0)))
    return center, radius


def _draw(ax, coordinates, faces, values, title, cmap, norm, poly_collection, elev, azim, center, radius):
    ax.clear()
    face_values = values[faces].mean(axis=1)
    ax.add_collection3d(
        poly_collection(
            coordinates[faces],
            facecolors=cmap(norm(face_values)),
            edgecolors="none",
            linewidths=0.0,
            antialiased=False,
        )
    )
    ax.set_title(title)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)


def main() -> None:
    args = parse_args()
    snapshots_path = Path(args.snapshots)
    snapshots = np.load(snapshots_path)
    model_dir = _resolve_model_dir(Path(args.model_dir), snapshots_path)
    model = np.load(model_dir / "rom_model.npz")
    summary = _load_summary(model_dir)

    coordinates = np.asarray(snapshots["coordinates"], dtype=float)
    topology = np.asarray(snapshots["topology"], dtype=np.int64)
    time = np.asarray(snapshots["time"], dtype=float)
    faces = _boundary_faces(topology)
    num_nodes = coordinates.shape[0]

    plot_fields = [field for field in args.plot_fields if field in snapshots.files]
    if not plot_fields:
        raise ValueError("No requested fields are present in snapshots")

    true_series = {field: np.asarray(snapshots[field], dtype=float) for field in plot_fields}
    rom_series = {}
    train_count = 0
    for field in plot_fields:
        rom_series[field], train_count = _rom_series(model, summary, args.fields, field)

    component_by_field = {}
    component_title_by_field = {}
    for field in plot_fields:
        count = _component_count(true_series[field], num_nodes)
        component_by_field[field] = _component_index(args.component, count)
        component_title_by_field[field] = _component_title(str(args.component), count)
    true_colors = {}
    rom_colors = {}
    limits = {}
    for field in plot_fields:
        count = _component_count(true_series[field], num_nodes)
        true_nodes = true_series[field].reshape(num_nodes, count, time.shape[0])
        rom_nodes = rom_series[field].reshape(num_nodes, count, time.shape[0])
        true_colors[field] = np.column_stack(
            [_color_values(true_nodes[:, :, index], component_by_field[field]) for index in range(time.shape[0])]
        )
        rom_colors[field] = np.column_stack(
            [_color_values(rom_nodes[:, :, index], component_by_field[field]) for index in range(time.shape[0])]
        )
        limits[field] = _limits(true_colors[field], rom_colors[field])

    d_true = true_series["D"].reshape(num_nodes, 3, time.shape[0]) if "D" in true_series else None
    d_rom = rom_series["D"].reshape(num_nodes, 3, time.shape[0]) if "D" in rom_series else None

    plt, _ = _require_matplotlib()
    from matplotlib import colors as mcolors
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.animation import FuncAnimation, PillowWriter

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 4 * len(plot_fields)), constrained_layout=True)
    axes = np.empty((len(plot_fields), 2), dtype=object)
    for row in range(len(plot_fields)):
        axes[row, 0] = fig.add_subplot(len(plot_fields), 2, 2 * row + 1, projection="3d")
        axes[row, 1] = fig.add_subplot(len(plot_fields), 2, 2 * row + 2, projection="3d")

    cmap = plt.get_cmap("RdBu_r")
    norms = {field: mcolors.Normalize(*limits[field]) for field in plot_fields}

    def frame_coords(index: int):
        if not args.deform or d_true is None or d_rom is None:
            return coordinates, coordinates
        return (
            coordinates + args.deformation_scale * d_true[:, :, index],
            coordinates + args.deformation_scale * d_rom[:, :, index],
        )

    def draw_frame(index: int):
        fom_coords, rom_coords = frame_coords(index)
        center, radius = _axis_limits(fom_coords, rom_coords)
        phase = "POD reconstruction" if index <= train_count else "ROM rollout"
        for row, field in enumerate(plot_fields):
            title = FIELD_TITLES.get(field, field)
            quantity = component_title_by_field[field]
            _draw(
                axes[row, 0],
                fom_coords,
                faces,
                true_colors[field][:, index],
                f"{args.fom_label}: {title} {quantity}",
                cmap,
                norms[field],
                Poly3DCollection,
                args.elev,
                args.azim,
                center,
                radius,
            )
            _draw(
                axes[row, 1],
                rom_coords,
                faces,
                rom_colors[field][:, index],
                f"{args.rom_label}: {title} {quantity}",
                cmap,
                norms[field],
                Poly3DCollection,
                args.elev,
                args.azim,
                center,
                radius,
            )
        title = args.title or f"{args.case_name}: FOM reference vs POD ROM prediction"
        fig.suptitle(f"{title}\nt = {time[index]:.6g} s, {phase}")
        return []

    for row, field in enumerate(plot_fields):
        colorbar = fig.colorbar(ScalarMappable(norm=norms[field], cmap=cmap), ax=axes[row, :], shrink=0.75)
        colorbar.set_label(f"{FIELD_TITLES.get(field, field)} {component_title_by_field[field]}")

    if args.animate:
        if args.frame_stride < 1:
            raise ValueError(f"--frame-stride must be positive, got {args.frame_stride}")
        frames = list(range(0, time.shape[0], args.frame_stride))
        if frames[-1] != time.shape[0] - 1:
            frames.append(time.shape[0] - 1)
        animation = FuncAnimation(fig, draw_frame, frames=frames, interval=1000 / args.fps, blit=False)
        animation.save(output, writer=PillowWriter(fps=args.fps))
    else:
        index = args.snapshot_index if args.snapshot_index >= 0 else time.shape[0] + args.snapshot_index
        draw_frame(index)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote 3D all-field comparison to {output}")


if __name__ == "__main__":
    main()
