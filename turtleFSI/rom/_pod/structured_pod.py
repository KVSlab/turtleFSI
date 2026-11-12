"""Train and roll out a POD-reduced structured non-intrusive FSI ROM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod.infer import fit_default_fsi_blocks
from turtleFSI.rom._shared.pod import (
    compute_pod_basis,
    project_to_basis,
    reconstruct_from_basis,
    relative_l2_error,
)
from turtleFSI.rom._pod.simulate import rollout_structured_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="Input .npz with U, Ds, Vs and optional Df.")
    parser.add_argument("--output-dir", required=True, help="Directory for POD-structured ROM outputs.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Ridge regularization.")
    parser.add_argument("--energy-threshold", type=float, default=0.999, help="POD retained energy.")
    parser.add_argument("--max-modes", type=int, default=30, help="Maximum POD modes per field.")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Rollout steps. Defaults to the snapshot transition count.",
    )
    return parser.parse_args()


def _load_snapshot_blocks(path: Path) -> dict[str, np.ndarray]:
    dataset = np.load(path)
    snapshots = {
        name: np.asarray(dataset[name], dtype=float)
        for name in ("U", "Ds", "Vs", "Df")
        if name in dataset.files
    }
    if not {"U", "Ds", "Vs"}.issubset(snapshots):
        raise KeyError("snapshots must contain U, Ds, and Vs")
    return snapshots


def main() -> None:
    args = parse_args()
    snapshots = _load_snapshot_blocks(Path(args.snapshots))

    bases = {
        name: compute_pod_basis(
            values,
            energy_threshold=args.energy_threshold,
            max_modes=args.max_modes,
            center=True,
        )
        for name, values in snapshots.items()
    }
    latent_snapshots = {
        name: project_to_basis(values, bases[name])
        for name, values in snapshots.items()
    }

    models = fit_default_fsi_blocks(latent_snapshots, ridge=args.ridge)
    snapshot_count = next(iter(snapshots.values())).shape[1]
    num_steps = args.num_steps if args.num_steps is not None else snapshot_count - 1
    latent_rollout = rollout_structured_blocks(
        models=models,
        initial_state={name: values[:, 0] for name, values in latent_snapshots.items()},
        num_steps=num_steps,
    )
    reconstructed = {
        name: reconstruct_from_basis(coefficients, bases[name])
        for name, coefficients in latent_rollout.items()
        if name in bases
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    summary = {
        "snapshots": str(args.snapshots),
        "ridge": float(args.ridge),
        "energy_threshold": float(args.energy_threshold),
        "max_modes": args.max_modes,
        "num_steps": int(num_steps),
        "fields": {},
        "blocks": {},
        "rollout_relative_error": {},
        "finite_rollout": {},
    }

    for name, basis in bases.items():
        arrays[f"{name}_mean"] = basis.mean
        arrays[f"{name}_basis"] = basis.basis
        arrays[f"{name}_singular_values"] = basis.singular_values
        arrays[f"{name}_latent"] = latent_snapshots[name]
        summary["fields"][name] = {
            "num_dofs": int(snapshots[name].shape[0]),
            "num_snapshots": int(snapshots[name].shape[1]),
            "num_modes": int(basis.num_modes),
            "retained_energy": float(basis.energy_ratio[-1]),
            "pod_reconstruction_relative_error": relative_l2_error(
                snapshots[name],
                reconstruct_from_basis(latent_snapshots[name], basis),
            ),
        }

    for name, model in models.items():
        arrays[f"{name}_operator"] = model.operator
        arrays[f"{name}_bias"] = model.bias
        summary["blocks"][name] = {
            "input_fields": list(model.input_fields),
            "output_field": model.output_field,
            "train_relative_error": model.train_relative_error,
            "operator_shape": list(model.operator.shape),
        }

    for name, values in latent_rollout.items():
        arrays[f"{name}_latent_rollout"] = values
    for name, values in reconstructed.items():
        arrays[f"{name}_rollout"] = values
        summary["finite_rollout"][name] = bool(np.all(np.isfinite(values)))
        if name in snapshots and values.shape[1] <= snapshots[name].shape[1]:
            summary["rollout_relative_error"][name] = relative_l2_error(
                snapshots[name][:, : values.shape[1]],
                values,
            )

    np.savez_compressed(output_dir / "pod_structured_rom.npz", **arrays)
    with open(output_dir / "pod_structured_rom_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote POD-structured ROM to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
