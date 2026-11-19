"""Plot POD-Galerkin validation fields on surfaces and over train/test time."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem


FIELD_TO_SUBSPACE = {"displacement": 0, "velocity": 1, "pressure": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--step", required=True, help="Main fenics_pod_galerkin_step.npz path.")
    parser.add_argument("--summary", required=True, help="Main step summary json path.")
    parser.add_argument("--snapshots", required=True, help="DVP snapshot bundle.")
    parser.add_argument("--basis", required=True, help="monolithic_pod_basis.npz path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-stop-index", type=int, required=True)
    parser.add_argument(
        "--prediction-dir-pattern",
        default=None,
        help="Optional pattern for test step dirs, e.g. /path/step_target{index}_fluid_pressure_gauge.",
    )
    parser.add_argument("--pressure-gauge", choices=("none", "zero-mean"), default="zero-mean")
    parser.add_argument("--pressure-dof-key", default="fluid_pressure_dofs")
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=-65.0)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument(
        "--problem-override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Problem builder overrides in KEY=VALUE form.",
    )
    return parser.parse_args()


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.animation import PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    return plt, mcolors, PillowWriter, Poly3DCollection


def _parse_override_value(value: str):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_problem_overrides(entries: list[str]) -> dict[str, object]:
    overrides = {"verbose": False}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Override must have KEY=VALUE form, got {entry!r}")
        key, value = entry.split("=", 1)
        overrides[key] = _parse_override_value(value)
    return overrides


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="ascii") as handle:
        return json.load(handle)


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


def _gauge_pressure(values: np.ndarray, gauge: str) -> np.ndarray:
    if gauge == "none":
        return values
    if gauge == "zero-mean":
        return values - values.mean(axis=0, keepdims=True)
    raise ValueError(f"Unknown pressure gauge {gauge!r}")


def _vertex_field(problem: FEniCSMonolithicProblem, state: np.ndarray, subspace: int) -> np.ndarray:
    problem.set_state(np.asarray(state, dtype=float).reshape(-1))
    mesh = problem.namespace["mesh"]
    field = problem.dvp_["n"].sub(subspace, deepcopy=True)
    raw = np.asarray(field.compute_vertex_values(mesh), dtype=float)
    num_vertices = mesh.num_vertices()
    if raw.size == num_vertices:
        return raw.reshape(num_vertices, 1)
    if raw.size % num_vertices != 0:
        raise ValueError(f"Unexpected vertex field size {raw.size} for {num_vertices} vertices")
    return raw.reshape(raw.size // num_vertices, num_vertices).T


def _surface_values(
    problem: FEniCSMonolithicProblem,
    state: np.ndarray,
    field: str,
    pressure_gauge: str = "zero-mean",
) -> np.ndarray:
    values = _vertex_field(problem, state, FIELD_TO_SUBSPACE[field])
    if field == "pressure":
        return _gauge_pressure(values, pressure_gauge)[:, 0]
    return np.linalg.norm(values, axis=1)


def _axis_limits(coordinates: np.ndarray) -> tuple[np.ndarray, float]:
    mins = coordinates.min(axis=0)
    maxs = coordinates.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    return center, radius


def _set_3d_limits(ax, center: np.ndarray, radius: float):
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _draw_surface(ax, coordinates, faces, values, title, cmap, norm, poly_collection, center, radius, elev, azim):
    face_values = values[faces].mean(axis=1)
    collection = poly_collection(
        coordinates[faces],
        facecolors=cmap(norm(face_values)),
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
    )
    ax.add_collection3d(collection)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    _set_3d_limits(ax, center, radius)
    return collection


def _project_state(
    state: np.ndarray,
    phi: np.ndarray,
    mean: np.ndarray,
    pressure_dofs: np.ndarray,
    pressure_gauge: str,
) -> np.ndarray:
    values = np.asarray(state, dtype=float).reshape(-1).copy()
    if pressure_gauge != "none":
        values[pressure_dofs] = _gauge_pressure(values[pressure_dofs, None], pressure_gauge)[:, 0]
    coeff = phi.T @ (values - mean)
    return mean + phi @ coeff


def _prediction_path(pattern: str | None, index: int) -> Path | None:
    if pattern is None:
        return None
    path = Path(pattern.format(index=index)) / "fenics_pod_galerkin_step.npz"
    return path if path.exists() else None


def _model_state_for_index(
    index: int,
    train_stop_index: int,
    snapshots: np.lib.npyio.NpzFile,
    phi: np.ndarray,
    mean: np.ndarray,
    pressure_dofs: np.ndarray,
    pressure_gauge: str,
    prediction_dir_pattern: str | None,
) -> tuple[np.ndarray, str]:
    if index < train_stop_index:
        return snapshots["dvp"][:, index], "training data used for POD basis"
    if index >= train_stop_index:
        prediction = _prediction_path(prediction_dir_pattern, index)
        if prediction is not None:
            return np.load(prediction)["full_state"].reshape(-1), "POD-Galerkin prediction"
    projected = _project_state(
        snapshots["dvp"][:, index],
        phi=phi,
        mean=mean,
        pressure_dofs=pressure_dofs,
        pressure_gauge=pressure_gauge,
    )
    return projected, "POD projection fallback: missing POD-Galerkin prediction"


def _field_norm(state: np.ndarray, dofs: np.ndarray, field: str, pressure_gauge: str) -> float:
    values = np.asarray(state, dtype=float).reshape(-1)[dofs].reshape(-1, 1)
    if field == "pressure":
        values = _gauge_pressure(values, pressure_gauge)
    return float(np.linalg.norm(values))


def _relative_error(true_state, rom_state, dofs, field, pressure_gauge) -> float:
    true_values = true_state[dofs].reshape(-1, 1)
    rom_values = rom_state[dofs].reshape(-1, 1)
    if field == "pressure":
        true_values = _gauge_pressure(true_values, pressure_gauge)
        rom_values = _gauge_pressure(rom_values, pressure_gauge)
    return float(np.linalg.norm(true_values - rom_values) / np.linalg.norm(true_values))


def _surface_limits_for_field(problem, snapshots, phi, mean, pressure_dofs, pressure_gauge, field, train_stop_index, pattern):
    values = []
    for index in range(snapshots["dvp"].shape[1]):
        true_state = snapshots["dvp"][:, index]
        rom_state, _ = _model_state_for_index(index, train_stop_index, snapshots, phi, mean, pressure_dofs, pressure_gauge, pattern)
        values.append(_surface_values(problem, true_state, field, pressure_gauge))
        values.append(_surface_values(problem, rom_state, field, pressure_gauge))
    stacked = np.concatenate(values)
    vmin = float(stacked.min())
    vmax = float(stacked.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def _plot_static_surface(
    output: Path,
    problem: FEniCSMonolithicProblem,
    coordinates: np.ndarray,
    faces: np.ndarray,
    field: str,
    true_state: np.ndarray,
    rom_state: np.ndarray,
    time_value: float,
    target_index: int,
    train_stop_index: int,
    pressure_gauge: str,
    elev: float,
    azim: float,
):
    plt, mcolors, _, poly_collection = _require_matplotlib()
    true_values = _surface_values(problem, true_state, field, pressure_gauge)
    rom_values = _surface_values(problem, rom_state, field, pressure_gauge)
    error_values = np.abs(rom_values - true_values)
    vmin = float(min(true_values.min(), rom_values.min()))
    vmax = float(max(true_values.max(), rom_values.max()))
    emax = float(error_values.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    if np.isclose(emax, 0.0):
        emax = 1.0

    cmap = plt.get_cmap("viridis")
    err_cmap = plt.get_cmap("magma")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    err_norm = mcolors.Normalize(vmin=0.0, vmax=emax)
    center, radius = _axis_limits(coordinates)
    fig = plt.figure(figsize=(15, 5.2), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    panels = [
        ("true FOM", true_values, cmap, norm),
        ("POD-Galerkin model", rom_values, cmap, norm),
        ("absolute error", error_values, err_cmap, err_norm),
    ]
    for ax, (title, values, panel_cmap, panel_norm) in zip(axes, panels):
        _draw_surface(ax, coordinates, faces, values, title, panel_cmap, panel_norm, poly_collection, center, radius, elev, azim)
        fig.colorbar(plt.cm.ScalarMappable(norm=panel_norm, cmap=panel_cmap), ax=ax, shrink=0.65)
    fig.suptitle(
        f"{field} surface at t={time_value:.3f}s, target index {target_index}; "
        f"training 0-{train_stop_index - 1}, testing {train_stop_index}+",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _animate_field(
    output: Path,
    problem: FEniCSMonolithicProblem,
    coordinates: np.ndarray,
    faces: np.ndarray,
    field: str,
    snapshots: np.lib.npyio.NpzFile,
    phi: np.ndarray,
    mean: np.ndarray,
    pressure_dofs: np.ndarray,
    pressure_gauge: str,
    train_stop_index: int,
    prediction_dir_pattern: str | None,
    elev: float,
    azim: float,
    fps: int,
):
    plt, mcolors, pillow_writer, poly_collection = _require_matplotlib()
    time = np.asarray(snapshots["time"], dtype=float)
    vmin, vmax = _surface_limits_for_field(
        problem,
        snapshots,
        phi,
        mean,
        pressure_dofs,
        pressure_gauge,
        field,
        train_stop_index,
        prediction_dir_pattern,
    )
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    center, radius = _axis_limits(coordinates)
    fig = plt.figure(figsize=(10.5, 5.2), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.7, label=field)
    writer = pillow_writer(fps=fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(output), dpi=150):
        for index, time_value in enumerate(time):
            true_state = snapshots["dvp"][:, index]
            rom_state, rom_label = _model_state_for_index(
                index,
                train_stop_index,
                snapshots,
                phi,
                mean,
                pressure_dofs,
                pressure_gauge,
                prediction_dir_pattern,
            )
            phase = "TRAINING" if index < train_stop_index else "TESTING"
            for ax in axes:
                ax.clear()
            _draw_surface(
                axes[0],
                coordinates,
                faces,
                _surface_values(problem, true_state, field, pressure_gauge),
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
                _surface_values(problem, rom_state, field, pressure_gauge),
                rom_label,
                cmap,
                norm,
                poly_collection,
                center,
                radius,
                elev,
                azim,
            )
            fig.suptitle(f"{field}: {phase}, index {index}, t={time_value:.3f}s", fontsize=12)
            writer.grab_frame()
    plt.close(fig)


def _collect_behavior(snapshots, pressure_dofs, train_stop_index, pattern, pressure_gauge):
    time = np.asarray(snapshots["time"], dtype=float)
    field_dofs = {
        "displacement": np.asarray(snapshots["displacement_dofs"], dtype=np.int64),
        "velocity": np.asarray(snapshots["velocity_dofs"], dtype=np.int64),
        "pressure": pressure_dofs,
    }
    behavior = {}
    for field in ("displacement", "velocity", "pressure"):
        fom_norms = []
        pg_times = []
        pg_norms = []
        pg_errors = []
        dofs = field_dofs[field]
        for index, time_value in enumerate(time):
            true_state = np.asarray(snapshots["dvp"][:, index], dtype=float).copy()
            fom_norms.append(_field_norm(true_state, dofs, field, pressure_gauge))
            if index >= train_stop_index:
                prediction = _prediction_path(pattern, index)
                if prediction is not None:
                    pg_state = np.load(prediction)["full_state"].reshape(-1)
                    pg_times.append(time_value)
                    pg_norms.append(_field_norm(pg_state, dofs, field, pressure_gauge))
                    pg_errors.append(_relative_error(true_state, pg_state, dofs, field, pressure_gauge))
        behavior[field] = {
            "fom_norms": fom_norms,
            "pg_times": pg_times,
            "pg_norms": pg_norms,
            "pg_errors": pg_errors,
        }
    return time, behavior


def _shade_train_test(ax, time: np.ndarray, train_stop_index: int):
    ax.axvspan(time[0], time[train_stop_index - 1], color="#d9ead3", alpha=0.35, label="training data")
    ax.axvspan(time[train_stop_index], time[-1], color="#fce5cd", alpha=0.35, label="testing window")
    ax.axvline(time[train_stop_index], color="#444444", linestyle=":", linewidth=1.2, label="test starts")


def _plot_behavior(output, snapshots, pressure_dofs, train_stop_index, pattern, pressure_gauge):
    plt, _, _, _ = _require_matplotlib()
    time, behavior = _collect_behavior(snapshots, pressure_dofs, train_stop_index, pattern, pressure_gauge)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for ax, field in zip(axes, ("displacement", "velocity", "pressure")):
        data = behavior[field]
        _shade_train_test(ax, time, train_stop_index)
        ax.plot(time, data["fom_norms"], color="#1f4e79", linewidth=2.0, label="true FOM snapshots")
        ax.plot(data["pg_times"], data["pg_norms"], color="#cc0000", marker="o", linewidth=2.0, label="POD-Galerkin prediction")
        ax.set_ylabel(f"{field} norm")
        ax.set_title(field)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", ncols=2, fontsize=8)
    fig.suptitle(
        f"Field norms: training indices 0-{train_stop_index - 1}, "
        f"testing indices {train_stop_index}-{time.shape[0] - 1}",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_errors(output, snapshots, pressure_dofs, train_stop_index, pattern, pressure_gauge):
    plt, _, _, _ = _require_matplotlib()
    time, behavior = _collect_behavior(snapshots, pressure_dofs, train_stop_index, pattern, pressure_gauge)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.8), sharex=True, constrained_layout=True)
    for ax, field in zip(axes, ("displacement", "velocity", "pressure")):
        data = behavior[field]
        _shade_train_test(ax, time, train_stop_index)
        ax.plot(
            data["pg_times"],
            data["pg_errors"],
            color="#e69138",
            marker="x",
            linewidth=1.8,
            label="POD-Galerkin relative error",
        )
        ax.set_ylabel(f"{field} rel. error")
        ax.set_title(field)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", ncols=2, fontsize=8)
    fig.suptitle(
        f"POD-Galerkin testing errors: training indices 0-{train_stop_index - 1}, "
        f"testing indices {train_stop_index}-{time.shape[0] - 1}",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary = _load_json(Path(args.summary))
    snapshots = np.load(args.snapshots)
    basis = np.load(args.basis)
    step_data = np.load(args.step)
    phi = np.asarray(basis["basis"], dtype=float)
    mean = np.asarray(basis["mean"], dtype=float).reshape(-1)
    basis_pressure_dofs = np.asarray(basis["pressure_dofs"], dtype=np.int64)
    pressure_dofs = np.asarray(snapshots[args.pressure_dof_key], dtype=np.int64)
    target_index = int(summary["target_index"])
    time = np.asarray(snapshots["time"], dtype=float)

    problem = FEniCSMonolithicProblem.build(args.problem, overrides=_parse_problem_overrides(args.problem_override))
    mesh = problem.namespace["mesh"]
    coordinates = np.asarray(mesh.coordinates(), dtype=float)
    faces = _boundary_faces(np.asarray(mesh.cells(), dtype=np.int64))

    true_state = np.asarray(step_data["target_state"], dtype=float).reshape(-1)
    rom_state = np.asarray(step_data["full_state"], dtype=float).reshape(-1)
    for field in ("displacement", "velocity", "pressure"):
        _plot_static_surface(
            output_dir / f"{field}_surface_true_vs_rom_3d.png",
            problem,
            coordinates,
            faces,
            field,
            true_state,
            rom_state,
            float(time[target_index]),
            target_index,
            args.train_stop_index,
            args.pressure_gauge,
            args.elev,
            args.azim,
        )
        _animate_field(
            output_dir / f"{field}_surface_training_testing.gif",
            problem,
            coordinates,
            faces,
            field,
            snapshots,
            phi,
            mean,
            basis_pressure_dofs,
            args.pressure_gauge,
            args.train_stop_index,
            args.prediction_dir_pattern,
            args.elev,
            args.azim,
            args.fps,
        )

    _plot_behavior(
        output_dir / "field_norms_training_testing_behavior.png",
        snapshots,
        pressure_dofs,
        args.train_stop_index,
        args.prediction_dir_pattern,
        args.pressure_gauge,
    )
    _plot_errors(
        output_dir / "field_relative_errors_testing.png",
        snapshots,
        pressure_dofs,
        args.train_stop_index,
        args.prediction_dir_pattern,
        args.pressure_gauge,
    )
    print(f"Wrote validation plots and animations to {output_dir}")


if __name__ == "__main__":
    main()
