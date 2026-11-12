"""Check FEniCS residual/Jacobian projection for a monolithic POD basis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem
from turtleFSI.rom._pod_galerkin.monolithic_pod import load_monolithic_pod_basis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="TF_csm", help="turtleFSI problem name.")
    parser.add_argument("--basis", required=True, help="monolithic_pod_basis.npz path.")
    parser.add_argument("--snapshots", required=True, help="Exact dvp_snapshots.npz path.")
    parser.add_argument("--snapshot-index", type=int, default=-1, help="Current snapshot index.")
    parser.add_argument("--previous-index", type=int, default=None, help="Previous snapshot index.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--dt", type=float, default=None, help="Override dt.")
    parser.add_argument("--theta", type=float, default=None, help="Override theta.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    basis = load_monolithic_pod_basis(Path(args.basis))
    snapshots = np.load(args.snapshots)
    dvp = np.asarray(snapshots["dvp"], dtype=float)
    times = np.asarray(snapshots["time"], dtype=float) if "time" in snapshots.files else None

    current_index = args.snapshot_index
    if current_index < 0:
        current_index = dvp.shape[1] + current_index
    previous_index = args.previous_index
    if previous_index is None:
        previous_index = max(0, current_index - 1)
    elif previous_index < 0:
        previous_index = dvp.shape[1] + previous_index

    overrides = {"verbose": False}
    if args.dt is not None:
        overrides["dt"] = args.dt
    elif times is not None and current_index != previous_index:
        overrides["dt"] = float(times[current_index] - times[previous_index])
    if args.theta is not None:
        overrides["theta"] = args.theta

    problem = FEniCSMonolithicProblem.build(args.problem, overrides=overrides)
    current_state = dvp[:, current_index]
    previous_state = dvp[:, previous_index]
    problem.set_previous_state(previous_state)

    residual = problem.projected_residual(basis.pod_basis.basis, current_state)
    jacobian = problem.projected_jacobian(basis.pod_basis.basis, current_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "fenics_projected_assembly.npz",
        residual=residual,
        jacobian=jacobian,
    )
    summary = {
        "problem": args.problem,
        "basis": str(args.basis),
        "snapshots": str(args.snapshots),
        "snapshot_index": int(current_index),
        "previous_index": int(previous_index),
        "dt": float(overrides.get("dt", problem.namespace["dt"])),
        "num_full_dofs": int(problem.vector_size),
        "num_modes": int(basis.num_modes),
        "projected_residual_norm": float(np.linalg.norm(residual)),
        "projected_jacobian_shape": list(jacobian.shape),
        "projected_jacobian_condition": float(np.linalg.cond(jacobian))
        if jacobian.size > 0
        else None,
    }
    with open(output_dir / "fenics_projected_assembly_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote FEniCS projected assembly check to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
