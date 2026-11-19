"""Create a Figure-4-style 2x2 velocity field comparison plot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from turtleFSI.rom._shared.plot_common import (
    _require_matplotlib,
    expand_component_field,
    fluid_triangulation,
    load_pipeline_artifacts,
    reconstruct_field,
    rollout_fields,
    train_transition_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="paper/generated/tf_fsi_rom")
    parser.add_argument("--mesh-map", default="paper/generated/tf_fsi_mesh_map/tf_fsi_mesh_map.npz")
    parser.add_argument("--time", type=float, default=3.5, help="Target plot time.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    return parser.parse_args()


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    output = Path(args.output) if args.output else work_dir / "plots" / "velocity_fields.png"

    plt, _ = _require_matplotlib()
    snapshots, model, summary = load_pipeline_artifacts(work_dir)
    coords, fluid_cells, triangulation = fluid_triangulation(Path(args.mesh_map))
    del fluid_cells

    time = snapshots["time"]
    train_count = train_transition_count(len(time), float(summary["train_fraction"]))
    snap_idx = nearest_index(time, args.time)

    true_u = snapshots["U"][:, snap_idx]
    fluid_nodes = snapshots["fluid_interface_nodes"]

    if snap_idx <= train_count:
        rom_u = reconstruct_field(model, "U")[:, snap_idx]
    else:
        _, rollout = rollout_fields(model, summary, ["U", "Ds", "Vs"])
        pred_idx = snap_idx - (train_count + 1)
        if pred_idx < 0 or pred_idx >= rollout["U"].shape[1]:
            raise ValueError(f"Requested time {args.time} is outside the available rollout interval.")
        rom_u = reconstruct_field(model, "U", rollout["U"])[:, pred_idx]

    true_x, true_y = expand_component_field(true_u, fluid_nodes, coords.shape[0])
    rom_x, rom_y = expand_component_field(rom_u, fluid_nodes, coords.shape[0])

    x_limits = (np.nanmin([true_x, rom_x]), np.nanmax([true_x, rom_x]))
    y_limits = (np.nanmin([true_y, rom_y]), np.nanmax([true_y, rom_y]))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plots = [
        (axes[0, 0], true_x, x_limits, "FSI streamwise"),
        (axes[0, 1], rom_x, x_limits, "ROM streamwise"),
        (axes[1, 0], true_y, y_limits, "FSI transverse"),
        (axes[1, 1], rom_y, y_limits, "ROM transverse"),
    ]

    for ax, field, limits, title in plots:
        contour = ax.tricontourf(triangulation, field, levels=30, cmap="RdBu_r", vmin=limits[0], vmax=limits[1])
        fig.colorbar(contour, ax=ax, shrink=0.85)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(f"Velocity field comparison at t = {time[snap_idx]:.3f} s")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
