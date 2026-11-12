"""Build one monolithic POD basis from named FSI snapshot blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod_galerkin.monolithic_pod import (
    FieldLayout,
    MonolithicPODBasis,
    project_monolithic_snapshots,
    reconstruct_monolithic_state,
    save_monolithic_pod_basis,
    train_monolithic_pod_basis,
)
from turtleFSI.rom._shared.pod import PODBasis, compute_pod_basis, relative_l2_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="Input .npz snapshot bundle.")
    parser.add_argument("--output-dir", required=True, help="Directory for monolithic POD outputs.")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help=(
            "Field blocks to concatenate into the monolithic state. "
            "Defaults to 'dvp' for exact DVP bundles, otherwise Df U Ds Vs."
        ),
    )
    parser.add_argument("--energy-threshold", type=float, default=0.999)
    parser.add_argument("--max-modes", type=int, default=None)
    parser.add_argument(
        "--blocked-dvp",
        action="store_true",
        help=(
            "For exact DVP bundles, build separate displacement, velocity, and "
            "pressure POD bases and embed them in the mixed vector."
        ),
    )
    parser.add_argument("--displacement-modes", type=int, default=None)
    parser.add_argument("--velocity-modes", type=int, default=None)
    parser.add_argument("--pressure-modes", type=int, default=None)
    parser.add_argument("--supremizer-modes", type=int, default=None)
    parser.add_argument(
        "--pressure-treatment",
        choices=("none", "supremizer"),
        default="none",
        help="Use supremizer snapshots to enrich the velocity reduced space.",
    )
    parser.add_argument(
        "--pressure-dof-key",
        default="auto",
        help=(
            "Snapshot array used for pressure POD DoFs. 'auto' uses "
            "fluid_pressure_dofs/pressure_active_dofs when present, otherwise pressure_dofs."
        ),
    )
    parser.add_argument(
        "--pressure-gauge",
        choices=("none", "zero-mean"),
        default="none",
        help="Remove the per-snapshot pressure mean before building pressure POD modes.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First snapshot column included in the POD basis.",
    )
    parser.add_argument(
        "--stop-index",
        type=int,
        default=None,
        help="One-past-last snapshot column included in the POD basis.",
    )
    return parser.parse_args()


def _field_mode_cap(args: argparse.Namespace, name: str) -> int | None:
    specific = {
        "displacement": args.displacement_modes,
        "velocity": args.velocity_modes,
        "pressure": args.pressure_modes,
        "supremizer": args.supremizer_modes,
    }[name]
    return args.max_modes if specific is None else specific


def _field_energy_threshold(
    args: argparse.Namespace,
    name: str,
    energy_threshold: float,
) -> float:
    if _field_mode_cap(args, name) is not None:
        return 1.0
    return energy_threshold


def _load_supremizer_snapshots(
    dataset: np.lib.npyio.NpzFile,
    total_dofs: int,
    velocity_dofs: np.ndarray,
    start_index: int,
    stop_index: int | None,
) -> np.ndarray:
    for key in (
        "supremizer",
        "supremizers",
        "supremizer_snapshots",
        "velocity_supremizer",
        "velocity_supremizer_snapshots",
    ):
        if key not in dataset.files:
            continue
        values = np.asarray(dataset[key], dtype=float)[:, start_index:stop_index]
        if values.shape[0] == total_dofs:
            return values[velocity_dofs, :]
        if values.shape[0] == velocity_dofs.size:
            return values
        raise ValueError(
            f"Supremizer snapshots '{key}' have {values.shape[0]} rows, expected "
            f"either total DVP size {total_dofs} or velocity block size {velocity_dofs.size}"
        )
    raise KeyError(
        "--pressure-treatment supremizer requires one of these arrays in the "
        "snapshot bundle: supremizer, supremizers, supremizer_snapshots, "
        "velocity_supremizer, velocity_supremizer_snapshots"
    )


def _pressure_dof_key(dataset: np.lib.npyio.NpzFile, requested: str) -> str:
    if requested != "auto":
        if requested not in dataset.files:
            raise KeyError(f"pressure DoF key '{requested}' not found in snapshot bundle")
        return requested
    for key in ("fluid_pressure_dofs", "pressure_active_dofs", "pressure_dofs"):
        if key in dataset.files:
            return key
    raise KeyError("--blocked-dvp requires a pressure DoF array")


def _apply_pressure_gauge(snapshots: np.ndarray, gauge: str) -> np.ndarray:
    if gauge == "none":
        return snapshots
    if gauge == "zero-mean":
        return snapshots - snapshots.mean(axis=0, keepdims=True)
    raise ValueError(f"Unknown pressure gauge {gauge!r}")


def _orthogonalize_against(
    vectors: np.ndarray,
    existing_basis: np.ndarray,
    max_modes: int | None,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Return orthonormal vectors after removing components in an existing basis."""

    candidates = np.asarray(vectors, dtype=float)
    if candidates.ndim != 2:
        raise ValueError(f"vectors must be 2D, got {candidates.shape}")
    existing = np.asarray(existing_basis, dtype=float)
    if existing.ndim != 2 or existing.shape[0] != candidates.shape[0]:
        raise ValueError("existing_basis and vectors must have matching row counts")
    if existing.shape[1]:
        candidates = candidates - existing @ (existing.T @ candidates)
    if not candidates.shape[1]:
        return candidates
    q, r = np.linalg.qr(candidates, mode="reduced")
    diagonal = np.abs(np.diag(r))
    keep = diagonal > tolerance
    q = q[:, keep]
    if max_modes is not None:
        q = q[:, :max_modes]
    return q


def _build_blocked_dvp_basis(
    dataset: np.lib.npyio.NpzFile,
    start_index: int,
    stop_index: int | None,
    energy_threshold: float,
    args: argparse.Namespace,
) -> tuple[MonolithicPODBasis, np.ndarray, dict[str, float]]:
    if "dvp" not in dataset.files:
        raise KeyError("--blocked-dvp requires a bundle containing 'dvp'")

    dvp = np.asarray(dataset["dvp"], dtype=float)[:, start_index:stop_index]
    total_dofs = dvp.shape[0]
    pressure_key = _pressure_dof_key(dataset, getattr(args, "pressure_dof_key", "auto"))
    field_dof_keys = {
        "displacement": "displacement_dofs",
        "velocity": "velocity_dofs",
        "pressure": pressure_key,
    }
    field_names = ("displacement", "velocity", "pressure")
    missing = [key for key in field_dof_keys.values() if key not in dataset.files]
    if missing:
        raise KeyError(f"--blocked-dvp requires field DoF arrays; missing {missing}")

    training_dvp = np.zeros_like(dvp)
    mean = np.zeros((total_dofs, 1), dtype=float)
    basis_blocks = []
    fields = []
    singular_values = []
    energy_ratio = []
    reconstruction_errors = {}
    mode_cursor = 0

    for name in field_names:
        dofs = np.asarray(dataset[field_dof_keys[name]], dtype=np.int64)
        snapshots = dvp[dofs, :]
        if name == "pressure":
            snapshots = _apply_pressure_gauge(
                snapshots,
                getattr(args, "pressure_gauge", "none"),
            )
        training_dvp[dofs, :] = snapshots
        field_basis = compute_pod_basis(
            snapshots,
            energy_threshold=_field_energy_threshold(args, name, energy_threshold),
            max_modes=_field_mode_cap(args, name),
            center=True,
        )
        embedded = np.zeros((total_dofs, field_basis.basis.shape[1]), dtype=float)
        embedded[dofs, :] = field_basis.basis
        mean[dofs, :] = field_basis.mean
        basis_blocks.append(embedded)
        singular_values.extend(float(value) for value in field_basis.singular_values)
        energy_ratio.extend(float(value) for value in field_basis.energy_ratio)
        fields.append(
            FieldLayout(
                name=name,
                start=mode_cursor,
                stop=mode_cursor + field_basis.basis.shape[1],
                dofs=dofs,
            )
        )
        reconstructed = field_basis.mean + field_basis.basis @ (
            field_basis.basis.T @ (snapshots - field_basis.mean)
        )
        reconstruction_errors[name] = relative_l2_error(snapshots, reconstructed)
        mode_cursor += field_basis.basis.shape[1]

        if name == "velocity" and args.pressure_treatment == "supremizer":
            velocity_basis = field_basis.basis
            supremizer_snapshots = _load_supremizer_snapshots(
                dataset=dataset,
                total_dofs=total_dofs,
                velocity_dofs=dofs,
                start_index=start_index,
                stop_index=stop_index,
            )
            supremizer_basis = compute_pod_basis(
                supremizer_snapshots,
                energy_threshold=_field_energy_threshold(args, "supremizer", energy_threshold),
                max_modes=_field_mode_cap(args, "supremizer"),
                center=True,
            )
            enriched_supremizer_basis = _orthogonalize_against(
                vectors=supremizer_basis.basis,
                existing_basis=velocity_basis,
                max_modes=_field_mode_cap(args, "supremizer"),
            )
            embedded_supremizer = np.zeros(
                (total_dofs, enriched_supremizer_basis.shape[1]),
                dtype=float,
            )
            embedded_supremizer[dofs, :] = enriched_supremizer_basis
            basis_blocks.append(embedded_supremizer)
            singular_values.extend(float(value) for value in supremizer_basis.singular_values)
            energy_ratio.extend(float(value) for value in supremizer_basis.energy_ratio)
            fields.append(
                FieldLayout(
                    name="supremizer",
                    start=mode_cursor,
                    stop=mode_cursor + enriched_supremizer_basis.shape[1],
                    dofs=dofs,
                )
            )
            supremizer_basis_for_error = (
                enriched_supremizer_basis
                if enriched_supremizer_basis.shape[1]
                else np.zeros((supremizer_snapshots.shape[0], 0), dtype=float)
            )
            reconstructed_supremizer = supremizer_basis.mean + supremizer_basis_for_error @ (
                supremizer_basis_for_error.T @ (
                    supremizer_snapshots - supremizer_basis.mean
                )
            )
            reconstruction_errors["supremizer"] = relative_l2_error(
                supremizer_snapshots,
                reconstructed_supremizer,
            )
            mode_cursor += enriched_supremizer_basis.shape[1]

    basis = np.hstack(basis_blocks)
    pod_basis = PODBasis(
        mean=mean,
        basis=basis,
        singular_values=np.asarray(singular_values, dtype=float),
        energy_ratio=np.asarray(energy_ratio, dtype=float),
    )
    return MonolithicPODBasis(pod_basis=pod_basis, fields=tuple(fields)), training_dvp, reconstruction_errors


def main() -> None:
    args = parse_args()
    snapshots_path = Path(args.snapshots)
    if not snapshots_path.exists():
        raise FileNotFoundError(
            f"Snapshot bundle does not exist: {snapshots_path}\n"
            "For POD-Galerkin, run the full simulation with --rom-export-dvp true, "
            "then point --snapshots at RESULTS/ROM/dvp_snapshots.npz. "
            "If you used $RESULTS, check that it points at the run with DVP export enabled."
        )
    dataset = np.load(snapshots_path)
    blocked_reconstruction_errors = None
    if args.blocked_dvp:
        basis, snapshots, blocked_reconstruction_errors = _build_blocked_dvp_basis(
            dataset=dataset,
            start_index=args.start_index,
            stop_index=args.stop_index,
            energy_threshold=args.energy_threshold,
            args=args,
        )
    else:
        fields = args.fields
        if fields is None:
            fields = ["dvp"] if "dvp" in dataset.files else ["Df", "U", "Ds", "Vs"]
        blocks = {
            name: np.asarray(dataset[name], dtype=float)[:, args.start_index : args.stop_index]
            for name in fields
        }

        basis, snapshots = train_monolithic_pod_basis(
            blocks=blocks,
            field_order=fields,
            energy_threshold=args.energy_threshold,
            max_modes=args.max_modes,
        )
    coefficients = project_monolithic_snapshots(snapshots, basis)
    reconstruction = reconstruct_monolithic_state(coefficients, basis)
    reconstruction_error = relative_l2_error(snapshots, reconstruction)

    output_dir = Path(args.output_dir)
    summary = save_monolithic_pod_basis(
        output_dir=output_dir,
        basis=basis,
        snapshots_path=str(snapshots_path),
        reconstruction_error=reconstruction_error,
    )
    if blocked_reconstruction_errors is not None:
        summary["blocked_dvp"] = True
        summary["pressure_treatment"] = args.pressure_treatment
        summary["pressure_dof_key"] = (
            _pressure_dof_key(dataset, getattr(args, "pressure_dof_key", "auto"))
            if args.blocked_dvp
            else None
        )
        summary["pressure_gauge"] = getattr(args, "pressure_gauge", "none")
        summary["supremizer_enriched"] = args.pressure_treatment == "supremizer"
        summary["field_reconstruction_relative_errors"] = blocked_reconstruction_errors
        with open(output_dir / "monolithic_pod_basis_summary.json", "w", encoding="ascii") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    np.savez_compressed(
        output_dir / "monolithic_pod_projection.npz",
        coefficients=coefficients,
        reconstruction=reconstruction,
    )

    print(f"Wrote monolithic POD basis to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
