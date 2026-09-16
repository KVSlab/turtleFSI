"""Build a series of Ballarin-style reduced operator bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod_galerkin.build_ballarin_operator_bundle import (
    _normalize_index,
    build_reduced_operator_bundle,
)
from turtleFSI.rom._pod_galerkin.monolithic_pod import load_monolithic_pod_basis
from turtleFSI.rom._pod_galerkin.problem_overrides import (
    parse_problem_overrides,
    replay_overrides,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, help="turtleFSI problem name.")
    parser.add_argument("--basis", required=True, help="monolithic_pod_basis.npz path.")
    parser.add_argument("--snapshots", required=True, help="dvp_snapshots.npz path.")
    parser.add_argument("--output-dir", required=True, help="Directory for operator bundles.")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--stop-index", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument(
        "--problem-override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Additional problem namespace overrides passed to the FEniCS problem builder.",
    )
    return parser.parse_args()


def _selected_indices(start: int, stop: int | None, stride: int, count: int) -> list[int]:
    if stride < 1:
        raise ValueError("--stride must be positive")
    start = _normalize_index(start, count)
    stop_value = count if stop is None else _normalize_index(stop, count)
    if start < 0 or start >= count:
        raise IndexError(f"start-index {start} is outside snapshot range")
    if stop_value < 0 or stop_value > count:
        raise IndexError(f"stop-index {stop_value} is outside snapshot range")
    if start >= stop_value:
        raise ValueError("start-index must be smaller than stop-index")
    return list(range(start, stop_value, stride))


def _dt_from_snapshots(args: argparse.Namespace, times: np.ndarray | None, indices: list[int]) -> float:
    if args.dt is not None:
        return float(args.dt)
    if times is None or times.shape[0] < 2:
        raise ValueError("Provide --dt when snapshot times cannot define a time step")
    index = indices[0]
    if index > 0:
        return float(times[index] - times[index - 1])
    return float(times[1] - times[0])


def main() -> None:
    args = parse_args()
    from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem

    basis_path = Path(args.basis)
    snapshots_path = Path(args.snapshots)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    monolithic_basis = load_monolithic_pod_basis(basis_path)
    data = np.load(snapshots_path)
    dvp = np.asarray(data["dvp"], dtype=float)
    times = np.asarray(data["time"], dtype=float) if "time" in data.files else None
    indices = _selected_indices(args.start_index, args.stop_index, args.stride, dvp.shape[1])
    dt = _dt_from_snapshots(args, times, indices)

    overrides = replay_overrides(
        snapshots_path,
        explicit_overrides=parse_problem_overrides(args.problem_override),
    )
    overrides["dt"] = dt
    if args.theta is not None:
        overrides["theta"] = args.theta
    problem = FEniCSMonolithicProblem.build(args.problem, overrides=overrides)

    bundles = []
    for reference_index in indices:
        bundle_path = output_dir / f"reduced_operators_ref{reference_index:04d}.npz"
        metadata = build_reduced_operator_bundle(
            problem=problem,
            monolithic_basis=monolithic_basis,
            dvp=dvp,
            times=times,
            reference_index=reference_index,
            output=bundle_path,
            problem_name=args.problem,
            basis_path=basis_path,
            snapshots_path=snapshots_path,
            theta=args.theta,
        )
        bundles.append({"path": str(bundle_path), **metadata})

    manifest = {
        "problem": args.problem,
        "basis": str(basis_path),
        "snapshots": str(snapshots_path),
        "output_dir": str(output_dir),
        "start_index": int(indices[0]),
        "stop_index": int(indices[-1] + 1),
        "stride": int(args.stride),
        "num_bundles": int(len(bundles)),
        "bundles": bundles,
    }
    manifest_path = output_dir / "operator_series_manifest.json"
    with open(manifest_path, "w", encoding="ascii") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
