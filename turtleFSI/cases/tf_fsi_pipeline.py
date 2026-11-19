"""Run the offline TF_fsi ROM export and training pipeline in one command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from turtleFSI.rom._pod.build import train_rom
from turtleFSI.cases.snapshots.export_tf_fsi import export_snapshots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-folder",
        default="turtleFSI/problems/TF_fsi_results/1",
        help="Path to a TF_fsi results directory containing Visualization/.",
    )
    parser.add_argument(
        "--mesh-map",
        default="paper/generated/tf_fsi_mesh_map/tf_fsi_mesh_map.npz",
        help="Mesh-map .npz produced by build_tf_fsi_mesh_map.py.",
    )
    parser.add_argument(
        "--work-dir",
        default="paper/generated/tf_fsi_rom",
        help="Directory where exported snapshots and the ROM model will be written.",
    )
    parser.add_argument(
        "--include-pressure",
        action="store_true",
        help="Also export fluid/interface pressure snapshots as P.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["U", "Ds", "Vs"],
        help="Snapshot blocks to include in the ROM. Supported names: U, Ds, Vs, Df.",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.999,
        help="Cumulative POD energy threshold used to pick basis size.",
    )
    parser.add_argument(
        "--max-modes",
        type=int,
        default=None,
        help="Optional upper bound on POD modes per field.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-10,
        help="Ridge regularization for the latent linear system fit.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of transitions used for training. The rest are used for validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    snapshots_path = work_dir / "tf_fsi_snapshots.npz"
    model_dir = work_dir / "model"

    export_summary = export_snapshots(
        results_folder=Path(args.results_folder),
        mesh_map_path=Path(args.mesh_map),
        output_path=snapshots_path,
        include_pressure=args.include_pressure,
    )
    rom_summary = train_rom(
        snapshots_path=snapshots_path,
        output_dir=model_dir,
        field_names=args.fields,
        energy_threshold=args.energy_threshold,
        max_modes=args.max_modes,
        ridge=args.ridge,
        train_fraction=args.train_fraction,
        mesh_map_path=Path(args.mesh_map),
    )

    summary = {
        "export": export_summary,
        "rom": rom_summary,
    }
    with open(work_dir / "pipeline_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote snapshots to {snapshots_path}")
    print(f"Wrote ROM model to {model_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
