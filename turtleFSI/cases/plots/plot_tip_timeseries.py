"""Create a Figure-5-style tail-tip velocity timeseries plot for the ROM."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from turtleFSI.rom._shared.plot_common import (
    _require_matplotlib,
    load_pipeline_artifacts,
    rollout_fields,
    select_tip_node,
    train_transition_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="paper/generated/tf_fsi_rom")
    parser.add_argument("--mesh-map", default="paper/generated/tf_fsi_mesh_map/tf_fsi_mesh_map.npz")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    output = Path(args.output) if args.output else work_dir / "plots" / "tip_timeseries.png"

    plt, _ = _require_matplotlib()
    snapshots, model, summary = load_pipeline_artifacts(work_dir)

    solid_nodes = snapshots["solid_interface_nodes"]
    local_tip, tip_coord = select_tip_node(Path(args.mesh_map), solid_nodes)
    time = snapshots["time"]
    train_count = train_transition_count(len(time), float(summary["train_fraction"]))

    true_vs = snapshots["Vs"]
    true_x = true_vs[2 * local_tip]
    true_y = true_vs[2 * local_tip + 1]

    _, rollout = rollout_fields(model, summary, ["U", "Ds", "Vs"])
    pred_vs = model["Vs_mean"] + model["Vs_basis"] @ rollout["Vs"]
    pred_time = time[train_count + 1 :]
    pred_x = pred_vs[2 * local_tip]
    pred_y = pred_vs[2 * local_tip + 1]

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(time, true_x, color="#1b4965", linewidth=2, label="FSI")
    axes[0].plot(pred_time, pred_x, color="#c1121f", linewidth=2, linestyle="--", label="ROM")
    axes[0].axvline(time[train_count], color="black", linestyle=":", linewidth=1.5, label="Training cutoff")
    axes[0].set_ylabel("Streamwise velocity")
    axes[0].legend(frameon=False)

    axes[1].plot(time, true_y, color="#1b4965", linewidth=2, label="FSI")
    axes[1].plot(pred_time, pred_y, color="#c1121f", linewidth=2, linestyle="--", label="ROM")
    axes[1].axvline(time[train_count], color="black", linestyle=":", linewidth=1.5)
    axes[1].set_ylabel("Transverse velocity")
    axes[1].set_xlabel("Time [s]")

    fig.suptitle(
        f"Tail-tip velocity comparison at node {solid_nodes[local_tip]} near ({tip_coord[0]:.3f}, {tip_coord[1]:.3f})"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
