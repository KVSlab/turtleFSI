"""Test POD-Galerkin projection on an assembled linear system.

The operator file must contain:

``M``: full-order mass matrix
``K``: full-order stiffness/operator matrix
``f``: full-order force vector

The snapshot file must contain the requested field, for example ``Ds``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._shared.pod import relative_l2_error
from turtleFSI.rom._pod_galerkin.reduced_solvers import backward_euler, train_linear_pod_galerkin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="Input .npz snapshot bundle.")
    parser.add_argument("--field", default="Ds", help="Snapshot field to reduce.")
    parser.add_argument("--operators", required=True, help=".npz containing M, K, and f arrays.")
    parser.add_argument("--output-dir", required=True, help="Directory for POD-Galerkin outputs.")
    parser.add_argument("--dt", type=float, required=True, help="Time step for backward Euler rollout.")
    parser.add_argument("--num-steps", type=int, default=None, help="Rollout steps.")
    parser.add_argument("--energy-threshold", type=float, default=0.999)
    parser.add_argument("--max-modes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots_npz = np.load(args.snapshots)
    operators = np.load(args.operators)
    snapshots = np.asarray(snapshots_npz[args.field], dtype=float)

    model = train_linear_pod_galerkin(
        snapshots=snapshots,
        mass=operators["M"],
        stiffness=operators["K"],
        force=operators["f"],
        energy_threshold=args.energy_threshold,
        max_modes=args.max_modes,
    )
    num_steps = args.num_steps if args.num_steps is not None else snapshots.shape[1] - 1
    reduced, reconstructed = backward_euler(
        model=model,
        initial_full_state=snapshots[:, 0],
        dt=args.dt,
        num_steps=num_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "pod_galerkin_linear.npz",
        basis=model.basis.basis,
        mean=model.basis.mean,
        singular_values=model.basis.singular_values,
        energy_ratio=model.basis.energy_ratio,
        Mr=model.mass_r,
        Kr=model.stiffness_r,
        fr=model.force_r,
        reduced=reduced,
        reconstruction=reconstructed,
    )

    summary = {
        "snapshots": str(args.snapshots),
        "field": args.field,
        "operators": str(args.operators),
        "dt": float(args.dt),
        "num_steps": int(num_steps),
        "num_modes": int(model.basis.num_modes),
        "retained_energy": float(model.basis.energy_ratio[-1]),
        "reconstruction_relative_error": relative_l2_error(
            snapshots[:, : reconstructed.shape[1]],
            reconstructed,
        )
        if reconstructed.shape[1] <= snapshots.shape[1]
        else None,
    }
    with open(output_dir / "pod_galerkin_linear_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote POD-Galerkin output to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
