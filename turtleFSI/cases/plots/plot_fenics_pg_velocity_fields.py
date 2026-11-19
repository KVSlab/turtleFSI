"""Plot velocity fields from FEniCS POD-Galerkin mixed-state outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("This plotting script requires matplotlib.") from exc
    return plt, mtri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="TF_fsi", help="turtleFSI problem name.")
    parser.add_argument(
        "--input",
        required=True,
        help="POD-Galerkin step/rollout .npz containing full_state/full and optional target/reference.",
    )
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument(
        "--state-key",
        default=None,
        help="State array key. Defaults to full_state for step files or full for rollout files.",
    )
    parser.add_argument(
        "--reference-key",
        default=None,
        help="Reference array key. Defaults to target_state or reference when available.",
    )
    parser.add_argument(
        "--state-index",
        type=int,
        default=-1,
        help="Column index for trajectory arrays.",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Problem dt override.")
    parser.add_argument("--theta", type=float, default=None, help="Problem theta override.")
    return parser.parse_args()


def _select_key(data: np.lib.npyio.NpzFile, requested: str | None, candidates: tuple[str, ...]) -> str | None:
    if requested is not None:
        if requested not in data.files:
            raise KeyError(f"Could not find key '{requested}' in {data.files}")
        return requested
    for key in candidates:
        if key in data.files:
            return key
    return None


def _select_state(array: np.ndarray, index: int) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        raise ValueError(f"state array must be 1D or 2D, got {values.shape}")
    column = values.shape[1] + index if index < 0 else index
    if column < 0 or column >= values.shape[1]:
        raise IndexError(f"state-index {index} is outside trajectory with {values.shape[1]} columns")
    return values[:, column]


def _selected_column(array: np.ndarray, index: int) -> int:
    values = np.asarray(array)
    if values.ndim == 1:
        return 0
    column = values.shape[1] + index if index < 0 else index
    if column < 0 or column >= values.shape[1]:
        raise IndexError(f"state-index {index} is outside trajectory with {values.shape[1]} columns")
    return int(column)


def _mesh_arrays(problem: FEniCSMonolithicProblem) -> tuple[np.ndarray, np.ndarray]:
    mesh = problem.namespace["mesh"]
    domains = problem.namespace["domains"]
    coordinates = np.asarray(mesh.coordinates(), dtype=float)
    cells = np.asarray(mesh.cells(), dtype=np.int64)
    domain_values = np.asarray(domains.array(), dtype=np.int64)
    fluid_ids = set(int(value) for value in problem.namespace.get("dx_f_id_list", []))
    if fluid_ids:
        mask = np.asarray([int(value) in fluid_ids for value in domain_values], dtype=bool)
        if np.any(mask):
            cells = cells[mask]
    return coordinates, cells


def _velocity_vertex_components(
    problem: FEniCSMonolithicProblem,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    problem.set_state(state)
    mesh = problem.namespace["mesh"]
    velocity = problem.dvp_["n"].sub(1, deepcopy=True)
    raw = np.asarray(velocity.compute_vertex_values(mesh), dtype=float)
    num_vertices = mesh.num_vertices()
    geometric_dim = mesh.geometry().dim()
    if raw.size < 2 * num_vertices:
        raise ValueError(
            f"Expected at least {2 * num_vertices} vertex values for velocity, got {raw.size}"
        )
    if raw.size == geometric_dim * num_vertices:
        components = raw.reshape(geometric_dim, num_vertices)
        return components[0], components[1]
    return raw[:num_vertices], raw[num_vertices : 2 * num_vertices]


def _limits(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    if values.size == 0:
        return 0.0, 1.0
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high):
        pad = max(abs(low), 1.0) * 1e-6
        return low - pad, high + pad
    return low, high


def main() -> None:
    args = parse_args()
    plt, mtri = _require_matplotlib()
    data = np.load(args.input)
    state_key = _select_key(data, args.state_key, ("full_state", "full"))
    if state_key is None:
        raise KeyError("Input must contain full_state or full, or pass --state-key")
    reference_key = _select_key(data, args.reference_key, ("target_state", "reference"))

    rom_state = _select_state(data[state_key], args.state_index)
    selected_column = _selected_column(data[state_key], args.state_index)
    reference_state = None
    if reference_key is not None and data[reference_key].size:
        reference_state = _select_state(data[reference_key], args.state_index)

    overrides = {"verbose": False, "dt": args.dt}
    if args.theta is not None:
        overrides["theta"] = args.theta
    problem = FEniCSMonolithicProblem.build(args.problem, overrides=overrides)
    coordinates, cells = _mesh_arrays(problem)
    triangulation = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1], cells)

    rom_x, rom_y = _velocity_vertex_components(problem, rom_state)
    if reference_state is not None:
        ref_x, ref_y = _velocity_vertex_components(problem, reference_state)
    else:
        ref_x = np.zeros_like(rom_x)
        ref_y = np.zeros_like(rom_y)

    x_limits = _limits(ref_x, rom_x)
    y_limits = _limits(ref_y, rom_y)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if reference_state is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        plots = [
            (axes[0], rom_x, x_limits, "POD-Galerkin streamwise velocity"),
            (axes[1], rom_y, y_limits, "POD-Galerkin transverse velocity"),
        ]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        plots = [
            (axes[0, 0], ref_x, x_limits, "FOM streamwise velocity"),
            (axes[0, 1], rom_x, x_limits, "POD-Galerkin streamwise velocity"),
            (axes[1, 0], ref_y, y_limits, "FOM transverse velocity"),
            (axes[1, 1], rom_y, y_limits, "POD-Galerkin transverse velocity"),
        ]

    for axis, field, limits, title in plots:
        contour = axis.tricontourf(
            triangulation,
            field,
            levels=30,
            cmap="RdBu_r",
            vmin=limits[0],
            vmax=limits[1],
        )
        fig.colorbar(contour, ax=axis, shrink=0.85)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    if "time" in data.files:
        times = np.asarray(data["time"], dtype=float)
        if selected_column < times.size:
            fig.suptitle(f"FEniCS POD-Galerkin velocity, step {selected_column}, t = {times[selected_column]:.4f} s")
        else:
            fig.suptitle(f"FEniCS POD-Galerkin velocity, step {selected_column}")
    else:
        fig.suptitle(f"FEniCS POD-Galerkin velocity, step {selected_column}")

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
