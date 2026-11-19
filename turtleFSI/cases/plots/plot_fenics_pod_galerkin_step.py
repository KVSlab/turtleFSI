"""Plot diagnostics for one FEniCS POD-Galerkin step."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("This plotting script requires matplotlib.") from exc
    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, help="fenics_pod_galerkin_step.npz path.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=1500,
        help="Maximum number of DoF values to draw in value/error plots.",
    )
    return parser.parse_args()


def _sample_indices(size: int, max_points: int) -> np.ndarray:
    if size <= max_points:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, max_points).astype(np.int64)


def main() -> None:
    args = parse_args()
    plt = _require_matplotlib()
    data = np.load(args.step)

    rom = np.asarray(data["full_state"], dtype=float).reshape(-1)
    target = np.asarray(data["target_state"], dtype=float).reshape(-1)
    residual_norms = np.asarray(data["residual_norms"], dtype=float)
    correction_norms = np.asarray(data["correction_norms"], dtype=float)

    if rom.shape != target.shape:
        raise ValueError(f"ROM and target shapes differ: {rom.shape} vs {target.shape}")

    error = rom - target
    sampled = _sample_indices(rom.size, args.max_points)
    relative_error = np.linalg.norm(error) / max(np.linalg.norm(target), np.finfo(float).eps)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].semilogy(np.arange(residual_norms.size), residual_norms, marker="o", linewidth=1.8)
    axes[0, 0].set_title("Reduced Residual Convergence")
    axes[0, 0].set_xlabel("Newton iteration")
    axes[0, 0].set_ylabel("||Phi.T R||")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(sampled, target[sampled], label="FOM target", linewidth=1.3)
    axes[0, 1].plot(sampled, rom[sampled], "--", label="POD-Galerkin", linewidth=1.3)
    axes[0, 1].set_title("State Values")
    axes[0, 1].set_xlabel("mixed DoF index")
    axes[0, 1].set_ylabel("value")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(sampled, np.abs(error[sampled]), linewidth=1.3)
    axes[1, 0].set_title(f"Absolute Error, relative L2 = {relative_error:.3e}")
    axes[1, 0].set_xlabel("mixed DoF index")
    axes[1, 0].set_ylabel("|ROM - target|")
    axes[1, 0].grid(True, alpha=0.3)

    if correction_norms.size > 0:
        axes[1, 1].semilogy(
            np.arange(correction_norms.size),
            correction_norms,
            marker="o",
            linewidth=1.8,
        )
    axes[1, 1].set_title("Reduced Newton Correction")
    axes[1, 1].set_xlabel("Newton iteration")
    axes[1, 1].set_ylabel("||delta a||")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
