"""Create a Figure-6-style ROM dimension error heatmap."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from turtleFSI.rom._pod.build import fit_latent_linear_dynamics, relative_l2_error
from turtleFSI.rom._shared.plot_common import _require_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", default="paper/generated/tf_fsi_rom/tf_fsi_snapshots.npz")
    parser.add_argument("--output", default="paper/generated/tf_fsi_rom/plots/rom_error_grid.png")
    parser.add_argument("--max-rf", type=int, default=18)
    parser.add_argument("--max-rs", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--ridge", type=float, default=1e-10)
    return parser.parse_args()


def pod(snapshots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = snapshots.mean(axis=1, keepdims=True)
    left, _, _ = np.linalg.svd(snapshots - mean, full_matrices=False)
    return mean, left


def reconstruct(mean: np.ndarray, basis: np.ndarray, latent: np.ndarray) -> np.ndarray:
    return mean + basis @ latent


def main() -> None:
    args = parse_args()
    plt, _ = _require_matplotlib()

    data = np.load(Path(args.snapshots))
    u = data["U"]
    ds = data["Ds"]
    vs = data["Vs"]

    u_mean, u_left = pod(u)
    ds_mean, ds_left = pod(ds)
    vs_mean, vs_left = pod(vs)

    max_rf = min(args.max_rf, u_left.shape[1])
    max_rs = min(args.max_rs, ds_left.shape[1], vs_left.shape[1])
    errors = np.full((max_rs, max_rf), np.nan, dtype=float)

    num_snapshots = u.shape[1]
    transition_count = num_snapshots - 1
    train_count = max(1, min(transition_count - 1, int(np.floor(args.train_fraction * transition_count))))

    for rs in range(1, max_rs + 1):
        ds_basis = ds_left[:, :rs]
        vs_basis = vs_left[:, :rs]
        ds_latent = ds_basis.T @ (ds - ds_mean)
        vs_latent = vs_basis.T @ (vs - vs_mean)

        for rf in range(1, max_rf + 1):
            u_basis = u_left[:, :rf]
            u_latent = u_basis.T @ (u - u_mean)
            latent = np.vstack([u_latent, ds_latent, vs_latent])
            dynamics = fit_latent_linear_dynamics(latent, args.ridge, args.train_fraction)

            current = latent[:, train_count : train_count + 1]
            predicted = []
            for _ in range(num_snapshots - train_count - 1):
                current = dynamics["A"] @ current + dynamics["b"]
                predicted.append(current[:, 0])
            predicted = np.asarray(predicted, dtype=float).T

            u_pred_latent = predicted[:rf]
            u_pred = reconstruct(u_mean, u_basis, u_pred_latent)
            u_true = u[:, train_count + 1 :]
            errors[rs - 1, rf - 1] = relative_l2_error(u_true, u_pred)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(errors, origin="lower", cmap="YlOrRd", aspect="auto")
    fig.colorbar(image, ax=ax, label="Validation rollout error")
    ax.set_xlabel(r"Fluid modes $r_f$")
    ax.set_ylabel(r"Solid modes $r_s$")
    ax.set_xticks(np.arange(max_rf))
    ax.set_xticklabels(np.arange(1, max_rf + 1))
    ax.set_yticks(np.arange(max_rs))
    ax.set_yticklabels(np.arange(1, max_rs + 1))
    ax.set_title("ROM error grid")

    for rs in range(max_rs):
        for rf in range(max_rf):
            ax.text(rf, rs, f"{errors[rs, rf]:.03f}", ha="center", va="center", fontsize=7, color="black")

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
