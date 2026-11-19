"""Plot diagnostics and surface animations for a FEniCS POD-Galerkin rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.cases.plots.plot_pod_galerkin_validation_3d import (
    FIELD_TO_SUBSPACE,
    _axis_limits,
    _boundary_faces,
    _draw_surface,
    _gauge_pressure,
    _surface_values,
    _vertex_field,
)
from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem
from turtleFSI.rom._pod_galerkin.problem_overrides import replay_overrides


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        from matplotlib.animation import PillowWriter
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("This plotting script requires matplotlib.") from exc
    return plt, mcolors, PillowWriter, Poly3DCollection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True, help="fenics_pod_galerkin_rollout.npz path.")
    parser.add_argument("--summary", default=None, help="fenics_pod_galerkin_rollout_summary.json path.")
    parser.add_argument("--snapshots", default=None, help="DVP snapshot bundle for FOM comparison.")
    parser.add_argument("--problem", default=None, help="turtleFSI problem name. Inferred from summary when possible.")
    parser.add_argument("--output", default=None, help="Optional component-diagnostics PNG path.")
    parser.add_argument("--output-dir", default=None, help="Output directory for surface animations.")
    parser.add_argument("--components", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--surface-animation",
        action="store_true",
        help="Write FOM/POD-Galerkin/error 3D surface GIFs from the rollout.",
    )
    parser.add_argument(
        "--slice-animation",
        action="store_true",
        help="Write FOM/POD-Galerkin/error slice-contour GIFs from the rollout.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=("displacement", "velocity", "pressure"),
        default=["displacement", "velocity", "pressure"],
    )
    parser.add_argument("--start-index", type=int, default=None, help="First global snapshot index to animate.")
    parser.add_argument("--stop-index", type=int, default=None, help="Last global snapshot index to animate, inclusive.")
    parser.add_argument("--train-stop-index", type=int, default=None)
    parser.add_argument("--pressure-gauge", choices=("none", "zero-mean"), default="zero-mean")
    parser.add_argument("--component", default="magnitude", help="x/y/z/0/1/2 or magnitude.")
    parser.add_argument("--slice-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--slice-value", type=float, default=None)
    parser.add_argument("--slice-tolerance", type=float, default=None)
    parser.add_argument("--levels", type=int, default=40)
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=-65.0)
    parser.add_argument("--fps", type=int, default=2)
    return parser.parse_args()


def _load_summary(args: argparse.Namespace) -> dict:
    summary_path = Path(args.summary) if args.summary else Path(args.rollout).with_name(
        "fenics_pod_galerkin_rollout_summary.json"
    )
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="ascii") as handle:
        return json.load(handle)


def _plot_component_diagnostics(output: Path, rollout: np.lib.npyio.NpzFile, components: list[int]) -> None:
    plt, _, _, _ = _require_matplotlib()
    full = np.asarray(rollout["full"], dtype=float)
    reference = np.asarray(rollout["reference"], dtype=float) if "reference" in rollout.files else None

    selected = [component for component in components if component < full.shape[0]]
    if not selected:
        raise ValueError("No requested component index is inside the rollout vector")

    fig, axes = plt.subplots(len(selected), 1, figsize=(9, 3 * len(selected)), squeeze=False)
    steps = np.arange(full.shape[1])
    for axis, component in zip(axes[:, 0], selected):
        axis.plot(steps, full[component], marker="o", label="POD-Galerkin", linewidth=1.7)
        if reference is not None and reference.size and component < reference.shape[0]:
            ref_steps = np.arange(reference.shape[1])
            axis.plot(ref_steps, reference[component], "--", label="FOM reference", linewidth=1.4)
        axis.set_ylabel(f"state[{component}]")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1, 0].set_xlabel("time step")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")


def _global_index_bounds(args: argparse.Namespace, summary: dict, rollout: np.lib.npyio.NpzFile) -> tuple[int, int, int]:
    rollout_start = int(summary.get("start_index", 0))
    rollout_stop = rollout_start + np.asarray(rollout["full"]).shape[1] - 1
    start_index = rollout_start if args.start_index is None else int(args.start_index)
    stop_index = rollout_stop if args.stop_index is None else int(args.stop_index)
    if start_index < rollout_start or stop_index > rollout_stop or start_index > stop_index:
        raise IndexError(
            f"Requested indices {start_index}-{stop_index} are outside rollout range "
            f"{rollout_start}-{rollout_stop}"
        )
    return rollout_start, start_index, stop_index


def _surface_limits(true_values: list[np.ndarray], rom_values: list[np.ndarray]) -> tuple[float, float]:
    stacked = np.concatenate([*true_values, *rom_values])
    low = float(stacked.min())
    high = float(stacked.max())
    if np.isclose(low, high):
        high = low + 1.0
    return low, high


def _parse_component(component: str, component_count: int) -> int | None:
    if component == "magnitude":
        return None
    aliases = {"x": 0, "y": 1, "z": 2}
    index = aliases.get(component)
    if index is None:
        index = int(component)
    if index < 0 or index >= component_count:
        raise IndexError(f"component {component} is outside [0, {component_count})")
    return int(index)


def _color_values(values: np.ndarray, component: int | None) -> np.ndarray:
    if component is None:
        return np.linalg.norm(values, axis=1) if values.shape[1] > 1 else values[:, 0]
    return values[:, component]


def _field_vertex_values(
    problem: FEniCSMonolithicProblem,
    state: np.ndarray,
    field: str,
    pressure_gauge: str,
    component: str,
) -> tuple[np.ndarray, str]:
    values = _vertex_field(problem, state, FIELD_TO_SUBSPACE[field])
    if field == "pressure":
        if pressure_gauge != "none":
            values = _gauge_pressure(values, pressure_gauge)
        return values[:, 0], "pressure"
    component_index = _parse_component(component, values.shape[1])
    component_label = "magnitude" if component_index is None else ("xyz"[component_index] if component_index < 3 else str(component_index))
    return _color_values(values, component_index), component_label


def _slice_indices(
    coordinates: np.ndarray,
    axis: str,
    slice_value: float | None,
    slice_tolerance: float | None,
) -> tuple[np.ndarray, float, float]:
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    values = coordinates[:, axis_index]
    if slice_value is None:
        slice_value = 0.5 * (float(values.min()) + float(values.max()))
    if slice_tolerance is None:
        slice_tolerance = 0.03 * float(values.max() - values.min())
    indices = np.flatnonzero(np.abs(values - slice_value) <= slice_tolerance)
    if indices.size < 12:
        nearest_count = max(12, min(160, values.size))
        indices = np.argsort(np.abs(values - slice_value))[:nearest_count]
    return np.asarray(indices, dtype=np.int64), float(slice_value), float(slice_tolerance)


def _project_slice_coordinates(coordinates: np.ndarray, axis: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    if axis == "x":
        return coordinates[:, 1], coordinates[:, 2], "y", "z"
    if axis == "y":
        return coordinates[:, 0], coordinates[:, 2], "x", "z"
    return coordinates[:, 0], coordinates[:, 1], "x", "y"


def _deduplicate_projected_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    slice_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rounded = np.round(np.column_stack([x_values, y_values]), decimals=12)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return x_values[unique_indices], y_values[unique_indices], slice_nodes[unique_indices]


def _draw_slice(ax, triangulation, values, title, cmap, norm, levels, labels):
    ax.clear()
    contour = ax.tricontourf(triangulation, values, levels=levels, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    return contour


def _animate_surface_field(
    output: Path,
    problem: FEniCSMonolithicProblem,
    coordinates: np.ndarray,
    faces: np.ndarray,
    field: str,
    snapshots: np.lib.npyio.NpzFile,
    rollout: np.lib.npyio.NpzFile,
    rollout_start: int,
    start_index: int,
    stop_index: int,
    train_stop_index: int | None,
    pressure_gauge: str,
    elev: float,
    azim: float,
    fps: int,
) -> None:
    plt, mcolors, pillow_writer, poly_collection = _require_matplotlib()
    indices = list(range(start_index, stop_index + 1))
    time = np.asarray(snapshots["time"], dtype=float) if "time" in snapshots.files else None
    full = np.asarray(rollout["full"], dtype=float)

    true_surface_values = []
    rom_surface_values = []
    for index in indices:
        rollout_col = index - rollout_start
        true_surface_values.append(_surface_values(problem, snapshots["dvp"][:, index], field, pressure_gauge))
        rom_surface_values.append(_surface_values(problem, full[:, rollout_col], field, pressure_gauge))

    vmin, vmax = _surface_limits(true_surface_values, rom_surface_values)
    error_max = float(max(np.max(np.abs(rom - true)) for true, rom in zip(true_surface_values, rom_surface_values)))
    if np.isclose(error_max, 0.0):
        error_max = 1.0

    cmap = plt.get_cmap("viridis")
    error_cmap = plt.get_cmap("magma")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    error_norm = mcolors.Normalize(vmin=0.0, vmax=error_max)
    center, radius = _axis_limits(coordinates)

    fig = plt.figure(figsize=(15, 5.2), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, panel + 1, projection="3d") for panel in range(3)]
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[:2], shrink=0.65, label=field)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=error_norm, cmap=error_cmap),
        ax=axes[2],
        shrink=0.65,
        label=f"{field} absolute error",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = pillow_writer(fps=fps)
    with writer.saving(fig, str(output), dpi=150):
        for frame, index in enumerate(indices):
            true_values = true_surface_values[frame]
            rom_values = rom_surface_values[frame]
            error_values = np.abs(rom_values - true_values)
            for axis in axes:
                axis.clear()
            _draw_surface(
                axes[0],
                coordinates,
                faces,
                true_values,
                "true FOM",
                cmap,
                norm,
                poly_collection,
                center,
                radius,
                elev,
                azim,
            )
            _draw_surface(
                axes[1],
                coordinates,
                faces,
                rom_values,
                "POD-Galerkin rollout",
                cmap,
                norm,
                poly_collection,
                center,
                radius,
                elev,
                azim,
            )
            _draw_surface(
                axes[2],
                coordinates,
                faces,
                error_values,
                "absolute error",
                error_cmap,
                error_norm,
                poly_collection,
                center,
                radius,
                elev,
                azim,
            )
            phase = "training" if train_stop_index is not None and index < train_stop_index else "testing"
            time_label = f", t={float(time[index]):.3f}s" if time is not None and index < time.shape[0] else ""
            fig.suptitle(f"{field}: {phase}, index {index}{time_label}", fontsize=12)
            writer.grab_frame()
    plt.close(fig)
    print(f"Wrote {output}")


def _write_surface_animations(args: argparse.Namespace, summary: dict, rollout: np.lib.npyio.NpzFile) -> None:
    if args.snapshots is None:
        raise ValueError("--snapshots is required for --surface-animation")
    if args.output_dir is None:
        raise ValueError("--output-dir is required for --surface-animation")
    problem_name = args.problem or summary.get("problem")
    if problem_name is None:
        raise ValueError("--problem is required when it cannot be inferred from rollout summary")

    snapshots_path = Path(args.snapshots)
    snapshots = np.load(snapshots_path)
    rollout_start, start_index, stop_index = _global_index_bounds(args, summary, rollout)
    overrides = replay_overrides(snapshots_path, explicit_overrides={"verbose": False})
    problem = FEniCSMonolithicProblem.build(str(problem_name), overrides=overrides)
    mesh = problem.namespace["mesh"]
    coordinates = np.asarray(mesh.coordinates(), dtype=float)
    faces = _boundary_faces(np.asarray(mesh.cells(), dtype=np.int64))

    output_dir = Path(args.output_dir)
    train_stop_index = args.train_stop_index
    for field in args.fields:
        _animate_surface_field(
            output_dir / f"{field}_fom_pod_galerkin_error_surface_indices{start_index}_{stop_index}.gif",
            problem,
            coordinates,
            faces,
            field,
            snapshots,
            rollout,
            rollout_start,
            start_index,
            stop_index,
            train_stop_index,
            args.pressure_gauge,
            args.elev,
            args.azim,
            args.fps,
        )


def _animate_slice_field(
    output: Path,
    problem: FEniCSMonolithicProblem,
    coordinates: np.ndarray,
    slice_nodes: np.ndarray,
    plot_x: np.ndarray,
    plot_y: np.ndarray,
    labels: tuple[str, str],
    field: str,
    snapshots: np.lib.npyio.NpzFile,
    rollout: np.lib.npyio.NpzFile,
    rollout_start: int,
    start_index: int,
    stop_index: int,
    train_stop_index: int | None,
    pressure_gauge: str,
    component: str,
    slice_axis: str,
    slice_value: float,
    slice_tolerance: float,
    levels: int,
    fps: int,
) -> None:
    plt, mcolors, pillow_writer, _ = _require_matplotlib()
    import matplotlib.tri as mtri

    indices = list(range(start_index, stop_index + 1))
    time = np.asarray(snapshots["time"], dtype=float) if "time" in snapshots.files else None
    full = np.asarray(rollout["full"], dtype=float)
    triangulation = mtri.Triangulation(plot_x, plot_y)

    true_values = []
    rom_values = []
    component_label = "magnitude"
    for index in indices:
        rollout_col = index - rollout_start
        true_frame, component_label = _field_vertex_values(
            problem,
            snapshots["dvp"][:, index],
            field,
            pressure_gauge,
            component,
        )
        rom_frame, component_label = _field_vertex_values(
            problem,
            full[:, rollout_col],
            field,
            pressure_gauge,
            component,
        )
        true_values.append(true_frame[slice_nodes])
        rom_values.append(rom_frame[slice_nodes])

    vmin, vmax = _surface_limits(true_values, rom_values)
    error_max = float(max(np.max(np.abs(rom - true)) for true, rom in zip(true_values, rom_values)))
    if np.isclose(error_max, 0.0):
        error_max = 1.0

    cmap = plt.get_cmap("viridis")
    error_cmap = plt.get_cmap("magma")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    error_norm = mcolors.Normalize(vmin=0.0, vmax=error_max)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[:2], shrink=0.82, label=f"{field} {component_label}")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=error_norm, cmap=error_cmap),
        ax=axes[2],
        shrink=0.82,
        label=f"{field} {component_label} absolute error",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = pillow_writer(fps=fps)
    with writer.saving(fig, str(output), dpi=150):
        for frame, index in enumerate(indices):
            error_values = np.abs(rom_values[frame] - true_values[frame])
            _draw_slice(axes[0], triangulation, true_values[frame], "true FOM", cmap, norm, levels, labels)
            _draw_slice(axes[1], triangulation, rom_values[frame], "POD-Galerkin rollout", cmap, norm, levels, labels)
            _draw_slice(axes[2], triangulation, error_values, "absolute error", error_cmap, error_norm, levels, labels)
            phase = "training" if train_stop_index is not None and index < train_stop_index else "testing"
            time_label = f", t={float(time[index]):.3f}s" if time is not None and index < time.shape[0] else ""
            fig.suptitle(
                f"{field} {component_label}: {phase}, index {index}{time_label}; "
                f"{slice_axis}={slice_value:.6g} +/- {slice_tolerance:.3g}",
                fontsize=12,
            )
            writer.grab_frame()
    plt.close(fig)
    print(f"Wrote {output}")


def _write_slice_animations(args: argparse.Namespace, summary: dict, rollout: np.lib.npyio.NpzFile) -> None:
    if args.snapshots is None:
        raise ValueError("--snapshots is required for --slice-animation")
    if args.output_dir is None:
        raise ValueError("--output-dir is required for --slice-animation")
    problem_name = args.problem or summary.get("problem")
    if problem_name is None:
        raise ValueError("--problem is required when it cannot be inferred from rollout summary")

    snapshots_path = Path(args.snapshots)
    snapshots = np.load(snapshots_path)
    rollout_start, start_index, stop_index = _global_index_bounds(args, summary, rollout)
    overrides = replay_overrides(snapshots_path, explicit_overrides={"verbose": False})
    problem = FEniCSMonolithicProblem.build(str(problem_name), overrides=overrides)
    mesh = problem.namespace["mesh"]
    coordinates = np.asarray(mesh.coordinates(), dtype=float)
    slice_nodes, slice_value, slice_tolerance = _slice_indices(
        coordinates,
        args.slice_axis,
        args.slice_value,
        args.slice_tolerance,
    )
    plot_x, plot_y, x_label, y_label = _project_slice_coordinates(coordinates[slice_nodes], args.slice_axis)
    plot_x, plot_y, slice_nodes = _deduplicate_projected_points(plot_x, plot_y, slice_nodes)

    output_dir = Path(args.output_dir)
    for field in args.fields:
        component_suffix = "pressure" if field == "pressure" else args.component
        _animate_slice_field(
            output_dir
            / (
                f"{field}_{component_suffix}_fom_pod_galerkin_error_slice_"
                f"{args.slice_axis}_indices{start_index}_{stop_index}.gif"
            ),
            problem,
            coordinates,
            slice_nodes,
            plot_x,
            plot_y,
            (x_label, y_label),
            field,
            snapshots,
            rollout,
            rollout_start,
            start_index,
            stop_index,
            args.train_stop_index,
            args.pressure_gauge,
            args.component,
            args.slice_axis,
            slice_value,
            slice_tolerance,
            args.levels,
            args.fps,
        )


def main() -> None:
    args = parse_args()
    rollout = np.load(args.rollout)
    summary = _load_summary(args)

    if args.output is not None:
        _plot_component_diagnostics(Path(args.output), rollout, args.components)

    if args.surface_animation:
        _write_surface_animations(args, summary, rollout)

    if args.slice_animation:
        _write_slice_animations(args, summary, rollout)

    if args.output is None and not args.surface_animation and not args.slice_animation:
        raise ValueError("Provide --output, --surface-animation, --slice-animation, or a combination.")


if __name__ == "__main__":
    main()
