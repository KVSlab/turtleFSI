"""Annotate an existing DVP snapshot bundle with active fluid pressure DoFs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, help="turtleFSI problem name.")
    parser.add_argument("--input", required=True, help="Input dvp_snapshots.npz bundle.")
    parser.add_argument("--output", required=True, help="Output annotated .npz bundle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    problem = FEniCSMonolithicProblem.build(args.problem, overrides={"verbose": False})
    fluid_pressure_dofs = problem.fluid_pressure_dofs()
    data = np.load(input_path)
    arrays = {name: data[name] for name in data.files}
    arrays["fluid_pressure_dofs"] = fluid_pressure_dofs
    arrays["pressure_active_dofs"] = fluid_pressure_dofs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    print(
        f"Wrote {output_path} with {fluid_pressure_dofs.size} active fluid pressure DoFs"
    )


if __name__ == "__main__":
    main()
