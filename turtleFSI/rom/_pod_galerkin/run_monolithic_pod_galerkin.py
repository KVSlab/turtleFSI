"""Run a monolithic POD-Galerkin rollout for a supplied full-space operator.

This command is the runnable algebraic version of monolithic POD-Galerkin. It
uses one POD basis for the concatenated FSI state and solves projected implicit
Euler equations.

The operator file must contain:

``M``: full mass matrix
``K``: full linear operator/stiffness matrix
``f``: optional force vector

For nonlinear FEniCS FSI, the same reduced Newton core is used, but ``M``/``K``
are replaced by callbacks that assemble turtleFSI's full residual and Jacobian.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod_galerkin.monolithic_pod import load_monolithic_pod_basis
from turtleFSI.rom._shared.pod import relative_l2_error
from turtleFSI.rom._pod_galerkin.reduced_solvers import implicit_euler_nonlinear_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", required=True, help="monolithic_pod_basis.npz path.")
    parser.add_argument("--operators", required=True, help=".npz containing M, K, optional f.")
    parser.add_argument("--initial-state", required=True, help=".npy or .npz array for initial full state.")
    parser.add_argument("--reference", default=None, help="Optional .npy/.npz reference trajectory.")
    parser.add_argument("--output-dir", required=True, help="Directory for Galerkin outputs.")
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--max-newton-iterations", type=int, default=25)
    return parser.parse_args()


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if "state" in data.files:
            return np.asarray(data["state"], dtype=float)
        if "trajectory" in data.files:
            return np.asarray(data["trajectory"], dtype=float)
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]], dtype=float)
        raise KeyError(
            f"{path} contains multiple arrays; use key 'state' or 'trajectory'"
        )
    return np.asarray(np.load(path), dtype=float)


def main() -> None:
    args = parse_args()
    basis = load_monolithic_pod_basis(Path(args.basis))
    operators = np.load(args.operators)
    initial_state = _load_array(Path(args.initial_state)).reshape(-1)
    force = operators["f"] if "f" in operators.files else None

    reduced, full, newton_results = implicit_euler_nonlinear_rollout(
        basis=basis.pod_basis,
        mass=operators["M"],
        operator=operators["K"],
        operator_jacobian=operators["K"],
        initial_full_state=initial_state,
        dt=args.dt,
        num_steps=args.num_steps,
        forcing=force,
        atol=args.atol,
        rtol=args.rtol,
        max_newton_iterations=args.max_newton_iterations,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "monolithic_pod_galerkin_rollout.npz",
        reduced=reduced,
        full=full,
    )

    reference_error = None
    if args.reference:
        reference = _load_array(Path(args.reference))
        if reference.ndim == 1:
            reference = reference[:, None]
        reference_error = relative_l2_error(reference[:, : full.shape[1]], full)

    summary = {
        "basis": str(args.basis),
        "operators": str(args.operators),
        "initial_state": str(args.initial_state),
        "reference": args.reference,
        "dt": float(args.dt),
        "num_steps": int(args.num_steps),
        "num_modes": int(basis.num_modes),
        "reference_relative_error": reference_error,
        "all_newton_converged": bool(all(result.converged for result in newton_results)),
        "newton_iterations": [int(result.iterations) for result in newton_results],
        "final_reduced_residual_norms": [
            float(result.residual_norms[-1]) for result in newton_results
        ],
    }
    with open(output_dir / "monolithic_pod_galerkin_summary.json", "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote monolithic POD-Galerkin rollout to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
