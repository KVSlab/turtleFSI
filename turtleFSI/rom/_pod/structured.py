"""Train and roll out a structured non-intrusive FSI ROM from snapshot arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod.infer import fit_default_fsi_blocks
from turtleFSI.rom._pod.simulate import rollout_structured_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="Input .npz with U, Ds, Vs and optional Df.")
    parser.add_argument("--output-dir", required=True, help="Directory for structured ROM outputs.")
    parser.add_argument("--ridge", type=float, default=1e-10, help="Ridge regularization.")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Rollout steps. Defaults to the snapshot transition count.",
    )
    return parser.parse_args()


def _relative_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    residual = np.linalg.norm(reference - prediction)
    baseline = np.linalg.norm(reference)
    if np.isclose(baseline, 0.0):
        return float(residual)
    return float(residual / baseline)


def main() -> None:
    args = parse_args()
    dataset = np.load(args.snapshots)
    snapshots = {
        name: np.asarray(dataset[name], dtype=float)
        for name in ("U", "Ds", "Vs", "Df")
        if name in dataset.files
    }
    if not {"U", "Ds", "Vs"}.issubset(snapshots):
        raise KeyError("snapshots must contain U, Ds, and Vs")

    models = fit_default_fsi_blocks(snapshots, ridge=args.ridge)
    snapshot_count = next(iter(snapshots.values())).shape[1]
    num_steps = args.num_steps if args.num_steps is not None else snapshot_count - 1
    rollout = rollout_structured_blocks(
        models=models,
        initial_state={name: values[:, 0] for name, values in snapshots.items()},
        num_steps=num_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    summary = {
        "snapshots": str(args.snapshots),
        "ridge": float(args.ridge),
        "num_steps": int(num_steps),
        "blocks": {},
        "rollout_relative_error": {},
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
    for name, values in rollout.items():
        arrays[f"{name}_rollout"] = values
        if name in snapshots and values.shape[1] <= snapshots[name].shape[1]:
            summary["rollout_relative_error"][name] = _relative_l2(
                snapshots[name][:, : values.shape[1]],
                values,
            )

    np.savez_compressed(output_dir / "structured_rom.npz", **arrays)
    with open(output_dir / "structured_rom_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote structured ROM to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
