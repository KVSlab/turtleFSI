"""Plot monolithic POD and POD-Galerkin diagnostic figures."""

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
        raise ModuleNotFoundError(
            "This plotting script requires matplotlib."
        ) from exc
    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    basis = subparsers.add_parser("basis", help="Plot POD singular values and cumulative energy.")
    basis.add_argument("--basis", required=True, help="monolithic_pod_basis.npz path.")
    basis.add_argument("--output", required=True, help="Output PNG path.")

    rollout = subparsers.add_parser("rollout", help="Plot Galerkin rollout components.")
    rollout.add_argument("--rollout", required=True, help="monolithic_pod_galerkin_rollout.npz path.")
    rollout.add_argument("--output", required=True, help="Output PNG path.")
    rollout.add_argument("--reference", default=None, help="Optional reference .npy/.npz trajectory.")
    rollout.add_argument(
        "--components",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Full-state component indices to plot.",
    )
    return parser.parse_args()


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if "trajectory" in data.files:
            return np.asarray(data["trajectory"], dtype=float)
        if "full" in data.files:
            return np.asarray(data["full"], dtype=float)
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]], dtype=float)
        raise KeyError(f"Could not infer array from {path}; expected trajectory or full")
    return np.asarray(np.load(path), dtype=float)


def plot_basis(basis_path: Path, output_path: Path) -> None:
    plt = _require_matplotlib()
    data = np.load(basis_path)
    singular_values = np.asarray(data["singular_values"], dtype=float)
    energy_ratio = np.asarray(data["energy_ratio"], dtype=float)
    modes = np.arange(1, singular_values.size + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(modes, singular_values, marker="o", linewidth=1.5)
    axes[0].set_xlabel("POD mode")
    axes[0].set_ylabel("Singular value")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(modes, energy_ratio, marker="o", linewidth=1.5)
    axes[1].set_xlabel("POD mode")
    axes[1].set_ylabel("Cumulative energy")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rollout(
    rollout_path: Path,
    output_path: Path,
    reference_path: Path | None,
    components: list[int],
) -> None:
    plt = _require_matplotlib()
    rollout = np.load(rollout_path)
    full = np.asarray(rollout["full"], dtype=float)
    reference = _load_array(reference_path) if reference_path is not None else None
    if reference is not None and reference.ndim == 1:
        reference = reference[:, None]

    fig, axes = plt.subplots(len(components), 1, figsize=(8, 3 * len(components)), squeeze=False)
    time = np.arange(full.shape[1])
    for axis, component in zip(axes[:, 0], components):
        axis.plot(time, full[component], label="POD-Galerkin", linewidth=1.8)
        if reference is not None and component < reference.shape[0]:
            ref_time = np.arange(min(reference.shape[1], full.shape[1]))
            axis.plot(
                ref_time,
                reference[component, : ref_time.size],
                "--",
                label="Reference",
                linewidth=1.4,
            )
        axis.set_ylabel(f"state[{component}]")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1, 0].set_xlabel("step")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.command == "basis":
        plot_basis(Path(args.basis), Path(args.output))
    elif args.command == "rollout":
        plot_rollout(
            rollout_path=Path(args.rollout),
            output_path=Path(args.output),
            reference_path=Path(args.reference) if args.reference else None,
            components=args.components,
        )


if __name__ == "__main__":
    main()
