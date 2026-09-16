"""Train a case-level POD/latent ROM from a snapshot bundle.

This is the clean user-facing wrapper for non-intrusive ROM training. It works
with any ``.npz`` snapshot bundle whose fields are 2D arrays with shape
``(num_dofs, num_snapshots)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod.build import train_rom


METADATA_KEYS = {
    "time",
    "counter",
    "cells",
    "coordinates",
    "field_names",
    "field_offsets",
}


def infer_snapshot_fields(snapshots_path: Path) -> list[str]:
    """Return 2D snapshot fields with a shared snapshot count."""

    dataset = np.load(snapshots_path)
    candidates: list[tuple[str, int]] = []
    for name in dataset.files:
        if name in METADATA_KEYS or name.endswith("_dofs"):
            continue
        values = np.asarray(dataset[name])
        if values.ndim == 2 and values.shape[1] >= 2:
            candidates.append((name, int(values.shape[1])))

    if not candidates:
        raise ValueError(f"No 2D snapshot fields found in {snapshots_path}")

    counts = {}
    for _, count in candidates:
        counts[count] = counts.get(count, 0) + 1
    snapshot_count = max(counts, key=counts.get)
    return [name for name, count in candidates if count == snapshot_count]


def default_output_dir(snapshots_path: Path) -> Path:
    if snapshots_path.name == "dvp_snapshots.npz" and snapshots_path.parent.name == "ROM":
        return snapshots_path.parent / "data_fit_rom"
    return snapshots_path.parent / "rom_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="Input .npz snapshot bundle.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults next to the snapshot bundle.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help="Snapshot fields to include. Defaults to all compatible 2D fields.",
    )
    parser.add_argument("--mesh-map", default=None, help="Optional mesh-map .npz.")
    parser.add_argument("--energy-threshold", type=float, default=0.999)
    parser.add_argument("--max-modes", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=1e-10)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots_path = Path(args.snapshots)
    if not snapshots_path.exists():
        raise FileNotFoundError(f"Snapshot bundle does not exist: {snapshots_path}")

    fields = args.fields if args.fields is not None else infer_snapshot_fields(snapshots_path)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(snapshots_path)
    summary = train_rom(
        snapshots_path=snapshots_path,
        output_dir=output_dir,
        field_names=fields,
        energy_threshold=args.energy_threshold,
        max_modes=args.max_modes,
        ridge=args.ridge,
        train_fraction=args.train_fraction,
        mesh_map_path=Path(args.mesh_map) if args.mesh_map else None,
    )
    summary["selected_fields"] = fields
    with open(output_dir / "case_rom_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote case ROM to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
