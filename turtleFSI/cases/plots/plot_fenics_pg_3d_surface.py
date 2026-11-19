"""Plot FOM-vs-POD-Galerkin fields on a 3D exterior mesh surface."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem


FIELD_TO_SUBSPACE = {
    "displacement": 0,
    "D": 0,
    "d": 0,
    "velocity": 1,
    "U": 1,
    "u": 1,
    "pressure": 2,
    "P": 2,
    "p": 2,
}
COMPONENT_LABELS = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="problem_aneu", help="turtleFSI problem name.")
    parser.add_argument("--input", required=True, help="fenics_pod_galerkin_step.npz path.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--field", default="velocity", help="velocity, displacement, or pressure.")
    parser.add_argument("--component", default="magnitude", help="x/y/z/0/1/2 or magnitude.")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--deform", action="store_true", help="Plot on displaced coordinates.")
    parser.add_argument("--deformation-scale", type=float, default=1.0)
    parser.add_argument(
        "--separate-color-scales",
        action="store_true",
        help="Use independent FOM/ROM color limits instead of one shared comparison scale.",
    )
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    return parser.parse_args()


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("This plotting script requires matplotlib.") from exc
    return plt, mcolors, ScalarMappable, Poly3DCollection


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


def _state_vector(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    values = np.asarray(data[key], dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    raise ValueError(f"{key} must be a vector or one-column array, got {values.shape}")


def _vertex_field(problem: FEniCSMonolithicProblem, state: np.ndarray, subspace: int) -> np.ndarray:
    problem.set_state(state)
    mesh = problem.namespace["mesh"]
    field = problem.dvp_["n"].sub(subspace, deepcopy=True)
    raw = np.asarray(field.compute_vertex_values(mesh), dtype=float)
    num_vertices = mesh.num_vertices()
    if raw.size == num_vertices:
        return raw.reshape(num_vertices, 1)
    if raw.size % num_vertices != 0:
        raise ValueError(f"Unexpected vertex field size {raw.size} for {num_vertices} vertices")
    components = raw.size // num_vertices
    return raw.reshape(components, num_vertices).T


def _color_values(values: np.ndarray, component: int | None) -> np.ndarray:
    if component is None:
        return np.linalg.norm(values, axis=1) if values.shape[1] > 1 else np.abs(values[:, 0])
    return values[:, component]


def _surface_limits(*coordinates: np.ndarray) -> tuple[np.ndarray, float]:
    stacked = np.vstack(coordinates)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    return center, radius


def _draw_surface(
    ax,
    coordinates: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap,
    norm,
    poly_collection,
    elev: float,
    azim: float,
    center: np.ndarray,
    radius: float,
    label: str,
) -> None:
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
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=elev, azim=azim)
    ax.text2D(0.02, 0.96, label, transform=ax.transAxes)


def _limits(values: np.ndarray) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high):
        pad = max(abs(low), 1.0) * 1e-8
        low -= pad
        high += pad
    return low, high


def main() -> None:
    args = parse_args()
    if args.field not in FIELD_TO_SUBSPACE:
        raise KeyError(f"Unknown field {args.field}. Use velocity, displacement, or pressure.")

    overrides = {"verbose": False}
    if args.dt is not None:
        overrides["dt"] = args.dt
    if args.theta is not None:
        overrides["theta"] = args.theta
    problem = FEniCSMonolithicProblem.build(args.problem, overrides=overrides)

    data = np.load(args.input)
    rom_state = _state_vector(data, "full_state")
    target_state = _state_vector(data, "target_state")

    mesh = problem.namespace["mesh"]
    coordinates = np.asarray(mesh.coordinates(), dtype=float)
    cells = np.asarray(mesh.cells(), dtype=np.int64)
    faces = _boundary_faces(cells)

    subspace = FIELD_TO_SUBSPACE[args.field]
    target_values = _vertex_field(problem, target_state, subspace)
    rom_values = _vertex_field(problem, rom_state, subspace)
    component = _parse_component(str(args.component), target_values.shape[1])
    component_name = "magnitude" if component is None else COMPONENT_LABELS[component]

    target_color = _color_values(target_values, component)
    rom_color = _color_values(rom_values, component)
    shared_limits = (
        float(min(target_color.min(), rom_color.min())),
        float(max(target_color.max(), rom_color.max())),
    )
    target_limits = _limits(target_color)
    rom_limits = _limits(rom_color)
    if not args.separate_color_scales:
        target_limits = _limits(np.asarray(shared_limits))
        rom_limits = target_limits

    target_coordinates = coordinates
    rom_coordinates = coordinates
    if args.deform:
        target_d = _vertex_field(problem, target_state, FIELD_TO_SUBSPACE["displacement"])
        rom_d = _vertex_field(problem, rom_state, FIELD_TO_SUBSPACE["displacement"])
        target_coordinates = coordinates + args.deformation_scale * target_d[:, : coordinates.shape[1]]
        rom_coordinates = coordinates + args.deformation_scale * rom_d[:, : coordinates.shape[1]]

    center, radius = _surface_limits(target_coordinates, rom_coordinates)
    plt, mcolors, scalar_mappable, poly_collection = _require_matplotlib()
    cmap = plt.get_cmap("RdBu_r")
    target_norm = mcolors.Normalize(vmin=target_limits[0], vmax=target_limits[1])
    rom_norm = mcolors.Normalize(vmin=rom_limits[0], vmax=rom_limits[1])

    fig = plt.figure(figsize=(11, 5.5), constrained_layout=True)
    axes = (fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2, projection="3d"))
    _draw_surface(
        axes[0],
        target_coordinates,
        faces,
        target_color,
        "FOM target",
        cmap,
        target_norm,
        poly_collection,
        args.elev,
        args.azim,
        center,
        radius,
        f"range [{target_limits[0]:.3e}, {target_limits[1]:.3e}]",
    )
    _draw_surface(
        axes[1],
        rom_coordinates,
        faces,
        rom_color,
        "POD-Galerkin",
        cmap,
        rom_norm,
        poly_collection,
        args.elev,
        args.azim,
        center,
        radius,
        f"range [{rom_limits[0]:.3e}, {rom_limits[1]:.3e}]",
    )
    if args.separate_color_scales:
        cbar0 = fig.colorbar(scalar_mappable(norm=target_norm, cmap=cmap), ax=axes[0], shrink=0.75)
        cbar0.set_label(f"FOM {args.field} {component_name}")
        cbar1 = fig.colorbar(scalar_mappable(norm=rom_norm, cmap=cmap), ax=axes[1], shrink=0.75)
        cbar1.set_label(f"ROM {args.field} {component_name}")
    else:
        colorbar = fig.colorbar(scalar_mappable(norm=target_norm, cmap=cmap), ax=axes, shrink=0.75)
        colorbar.set_label(f"{args.field} {component_name}")
    fig.suptitle(f"FOM vs POD-Galerkin {args.field} {component_name}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote 3D POD-Galerkin surface comparison to {output}")


if __name__ == "__main__":
    main()
