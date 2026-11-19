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
    parser.add_argument(
        "--snapshots", required=True, nargs="+",
        help="Input .npz bundles, one per training parameter in order (multiple require --blocked-dvp).",
    )
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
        help="First time-snapshot column included from each parameter bundle.",
    )
    parser.add_argument(
        "--stop-index",
        type=int,
        default=None,
        help="One-past-last time-snapshot column included from each parameter bundle.",
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
        "velocity_supremizer, velocity_supremizer_snapshots. "
        "First run python -m turtleFSI.rom.compute_supremizers --run-dir RUN "
        "and pass the resulting dvp_supremizers.npz to --snapshots."
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


class _SnapshotArrays(dict):
    """In-memory arrays with the same key interface as an npz bundle."""

    @property
    def files(self):
        return list(self)


def _assemble_parameter_snapshots(paths, args):
    """Build Si_d, Si_u, Si_p, Si_s over time, then S_* across parameters.

    Each input bundle represents one parameter mu_i on the same reference mesh
    and FE space. Input order defines i; column order within a bundle defines
    time order. Matrices are raw snapshots, before gauge adjustment/POD centering.
    """
    blocks = {name: [] for name in ("d", "u", "p", "s")}
    arrays = _SnapshotArrays()
    exported = {}
    mixed_blocks = []
    parameters = []
    offsets = [0]
    reference_dofs = None
    total_dofs = None
    has_supremizers = None
    supremizer_keys = (
        "supremizer", "supremizers", "supremizer_snapshots",
        "velocity_supremizer", "velocity_supremizer_snapshots",
    )
    for i, path in enumerate(paths, start=1):
        with np.load(path, allow_pickle=False) as dataset:
            dvp = np.asarray(dataset["dvp"], dtype=float)
            if dvp.ndim != 2 or not np.isfinite(dvp).all():
                raise ValueError(f"{path}: dvp must be a finite 2D snapshot matrix")
            pressure_key = _pressure_dof_key(dataset, args.pressure_dof_key)
            dof_keys = ("displacement_dofs", "velocity_dofs", pressure_key)
            dofs = [np.asarray(dataset[key]) for key in dof_keys]
            for key, indices in zip(dof_keys, dofs):
                if (indices.ndim != 1 or indices.size == 0
                        or indices.dtype.kind not in "iu"
                        or np.any(indices < 0) or np.any(indices >= dvp.shape[0])
                        or np.unique(indices).size != indices.size):
                    raise ValueError(f"{path}: invalid DoF array {key}")
            if reference_dofs is None:
                reference_dofs = dofs
                total_dofs = dvp.shape[0]
                for key in dataset.files:
                    if key.endswith("_dofs"):
                        arrays[key] = np.asarray(dataset[key])
            elif dvp.shape[0] != total_dofs or any(
                not np.array_equal(actual, expected)
                for actual, expected in zip(dofs, reference_dofs)
            ):
                raise ValueError(f"{path}: parameter bundles must have identical field DoF ordering and mixed size")

            columns = np.arange(dvp.shape[1])[args.start_index:args.stop_index]
            if columns.size == 0:
                raise ValueError(f"{path}: training window contains no snapshots")
            dvp_i = dvp[:, columns]
            displacement_dofs, velocity_dofs, pressure_dofs = dofs
            Si_d = dvp_i[displacement_dofs, :]
            Si_u = dvp_i[velocity_dofs, :]
            Si_p = dvp_i[pressure_dofs, :]
            local = {"d": Si_d, "u": Si_u, "p": Si_p}

            present = any(key in dataset.files for key in supremizer_keys)
            if args.pressure_treatment == "supremizer" or present:
                Si_s = _load_supremizer_snapshots(dataset, total_dofs, velocity_dofs, 0, None)
                if Si_s.shape[1] != dvp.shape[1] or not np.isfinite(Si_s).all():
                    raise ValueError(f"{path}: supremizers must be finite and match the DVP snapshot columns")
                Si_s = Si_s[:, columns]
                local["s"] = Si_s
            if has_supremizers is not None and present != has_supremizers:
                raise ValueError("Supremizer snapshots must be present in all parameter bundles or none")
            has_supremizers = present
            for name, matrix in local.items():
                blocks[name].append(matrix)
                exported[f"S{i}_{name}"] = matrix
            exported[f"snapshot_indices_{i}"] = columns
            if "time" in dataset.files:
                times = np.asarray(dataset["time"])
                if times.shape != (dvp.shape[1],):
                    raise ValueError(f"{path}: time must have one entry per DVP snapshot")
                exported[f"time_{i}"] = times[columns]
            mixed_blocks.append(dvp_i)
            offsets.append(offsets[-1] + columns.size)
            parameters.append({
                "index": i, "source": str(path),
                "column_start": int(offsets[-2]), "column_stop": int(offsets[-1]),
                "matrices": {f"S{i}_{name}": list(matrix.shape) for name, matrix in local.items()},
            })

    # S_d = [S1_d | S2_d | ...], and likewise for velocity, pressure, supremizers.
    S_d = np.hstack(blocks["d"])
    S_u = np.hstack(blocks["u"])
    S_p = np.hstack(blocks["p"])
    arrays.update(S_d=S_d, S_u=S_u, S_p=S_p, dvp=np.hstack(mixed_blocks))
    if blocks["s"]:
        S_s = np.hstack(blocks["s"])
        arrays.update(S_s=S_s, supremizer_snapshots=S_s)
    exported.update({key: arrays[key] for key in ("S_d", "S_u", "S_p", "S_s") if key in arrays})
    exported.update({key: value for key, value in arrays.items() if key.endswith("_dofs")})
    exported["parameter_column_offsets"] = np.asarray(offsets, dtype=np.int64)
    exported["parameter_sources"] = np.asarray([str(path) for path in paths])
    return arrays, exported, parameters


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
    matrix_names = {"displacement": "S_d", "velocity": "S_u", "pressure": "S_p"}
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
        matrix_name = matrix_names[name]
        snapshots = (
            np.asarray(dataset[matrix_name])[:, start_index:stop_index]
            if matrix_name in dataset.files else dvp[dofs, :]
        )
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
            supremizer_snapshots = (
                np.asarray(dataset["S_s"])[:, start_index:stop_index]
                if "S_s" in dataset.files else _load_supremizer_snapshots(
                    dataset=dataset,
                    total_dofs=total_dofs,
                    velocity_dofs=dofs,
                    start_index=start_index,
                    stop_index=stop_index,
                )
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


def _retained_projection_energy(snapshots, reconstruction):
    """Centered snapshot variance captured by the actual reconstructed basis.

    For blocked POD, concatenated per-field cumulative SVD energies are not
    a cumulative energy curve for the combined basis. Measure its projection.
    This uses the Euclidean coefficient norm, not a physical energy norm.
    """
    centered = snapshots - snapshots.mean(axis=1, keepdims=True)
    total = float(np.sum(centered ** 2))
    error = float(np.sum((snapshots - reconstruction) ** 2))
    if total == 0.0:
        return 1.0 if error == 0.0 else 0.0
    return float(1.0 - error / total)


def _section_3_1_summary(dataset, basis, args, start_index, stop_index, reconstruction_errors=None):
    """Return Ballarin/Rozza Section 3.1 POD-space metadata."""

    def matrix_info(name, key, dof_key=None):
        if key not in dataset.files:
            return {"present": False, "snapshot_key": key, "dof_key": dof_key}
        values = np.asarray(dataset[key])
        if values.ndim == 2:
            if dof_key is not None and dof_key in dataset.files and values.shape[0] != np.asarray(dataset[dof_key]).size:
                values = values[np.asarray(dataset[dof_key], dtype=np.int64), :]
            sliced = values[:, start_index:stop_index]
        else:
            sliced = values
        info = {
            "present": True,
            "snapshot_key": key,
            "shape": [int(dim) for dim in sliced.shape],
            "dof_key": dof_key,
        }
        if dof_key is not None and dof_key in dataset.files:
            info["num_dofs"] = int(np.asarray(dataset[dof_key]).size)
        return info

    pressure_key = None
    if getattr(args, "blocked_dvp", False):
        try:
            pressure_key = _pressure_dof_key(dataset, getattr(args, "pressure_dof_key", "auto"))
        except KeyError:
            pressure_key = getattr(args, "pressure_dof_key", "auto")

    fields = {field.name: field for field in basis.fields}
    mode_counts = {name: int(field.num_modes) for name, field in fields.items()}
    velocity_modes = mode_counts.get("velocity", 0)
    supremizer_modes = mode_counts.get("supremizer", 0)

    summary = {
        "paper_section": "3.1",
        "description": "POD-Galerkin monolithic reduced spaces from global displacement, global velocity, fluid pressure, and optional supremizer snapshots.",
        "training_window": {
            "start_index": int(start_index),
            "stop_index": None if stop_index is None else int(stop_index),
        },
        "snapshot_matrices": {},
        "reduced_spaces": {
            "E_N": {"source": "global displacement POD modes", "num_modes": mode_counts.get("displacement", 0)},
            "V_N": {
                "source": "global velocity POD modes enriched with supremizer modes",
                "velocity_modes": velocity_modes,
                "supremizer_modes": supremizer_modes,
                "num_modes": velocity_modes + supremizer_modes,
            },
            "Q_N": {"source": "fluid pressure POD modes", "num_modes": mode_counts.get("pressure", 0)},
        },
        "basis_matrices": {
            "Z_d": {"field": "displacement", "shape": [int(basis.num_dofs), mode_counts.get("displacement", 0)]},
            "Z_u_s": {"fields": ["velocity", "supremizer"], "shape": [int(basis.num_dofs), velocity_modes + supremizer_modes]},
            "Z_p": {"field": "pressure", "shape": [int(basis.num_dofs), mode_counts.get("pressure", 0)]},
        },
        "supremizer_problem": {
            "equation": "X s = B_F^T p",
            "snapshot_key": "supremizer_snapshots",
            "present": "supremizer" in fields,
            "embedded_in": "velocity",
        },
        "lifting": {
            "implemented": False,
            "note": "The basis builder currently uses exported DVP snapshots directly. Explicit lifting-function subtraction for inhomogeneous boundary data remains a future Section 3.1 refinement.",
        },
    }

    if getattr(args, "blocked_dvp", False) and "dvp" in dataset.files:
        summary["snapshot_matrices"] = {
            "S_d": matrix_info("S_d", "dvp", "displacement_dofs"),
            "S_u": matrix_info("S_u", "dvp", "velocity_dofs"),
            "S_p": matrix_info("S_p", "dvp", pressure_key),
            "S_s": matrix_info("S_s", "supremizer_snapshots", "velocity_dofs"),
        }
        for matrix_name, field_name in (("S_d", "displacement"), ("S_u", "velocity"), ("S_p", "pressure"), ("S_s", "supremizer")):
            if reconstruction_errors and field_name in reconstruction_errors:
                summary["snapshot_matrices"][matrix_name]["projection_relative_error"] = float(reconstruction_errors[field_name])
    else:
        summary["snapshot_matrices"] = {
            field.name: matrix_info(field.name, field.name)
            for field in basis.fields
        }

    return summary


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.snapshots]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Snapshot bundle does not exist: {path}\n"
                "For POD-Galerkin, run the full simulation with --rom-export-dvp true, "
                "then point --snapshots at RESULTS/ROM/dvp_snapshots.npz. "
                "If you used $RESULTS, check that it points at the run with DVP export enabled."
            )
    if len(paths) > 1 and not args.blocked_dvp:
        raise ValueError("Multiple parameter bundles require --blocked-dvp")
    snapshots_path = paths[0]
    blocked_reconstruction_errors = None
    explicit_matrices = None
    start_index, stop_index = args.start_index, args.stop_index
    if args.blocked_dvp:
        dataset, explicit_matrices, parameter_blocks = _assemble_parameter_snapshots(paths, args)
        # The requested time window has already been applied to every mu_i.
        start_index, stop_index = 0, None
        basis, snapshots, blocked_reconstruction_errors = _build_blocked_dvp_basis(
            dataset=dataset,
            start_index=start_index,
            stop_index=stop_index,
            energy_threshold=args.energy_threshold,
            args=args,
        )
    else:
        with np.load(snapshots_path, allow_pickle=False) as bundle:
            dataset = _SnapshotArrays({key: bundle[key] for key in bundle.files})
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
    section_3_1 = _section_3_1_summary(
        dataset=dataset,
        basis=basis,
        args=args,
        start_index=start_index,
        stop_index=stop_index,
        reconstruction_errors=blocked_reconstruction_errors,
    )
    summary["section_3_1"] = section_3_1
    if args.blocked_dvp:
        for name, info in summary["named_basis_matrices"].items():
            section_3_1["basis_matrices"].setdefault(name, {}).update(info)
    if explicit_matrices is not None:
        matrices_path = output_dir / "snapshot_matrices.npz"
        np.savez_compressed(matrices_path, **explicit_matrices)
        summary["snapshots_paths"] = [str(path) for path in paths]
        section_3_1["training_window"] = {
            "start_index": args.start_index, "stop_index": args.stop_index,
            "applies_to": "each parameter bundle",
        }
        section_3_1["parameter_blocks"] = parameter_blocks
        section_3_1["num_training_parameters"] = len(paths)
        section_3_1["snapshot_matrices_path"] = str(matrices_path)
        section_3_1["snapshot_matrix_convention"] = (
            "Si_field contains time columns for input parameter i (1-based); "
            "S_field concatenates Si_field in input order. Saved matrices precede "
            "pressure gauge adjustment and POD centering. Inputs must share the "
            "same reference mesh and FE space; matching DoF indices alone cannot verify geometry."
        )
        for name, info in section_3_1["snapshot_matrices"].items():
            if name in explicit_matrices:
                info["snapshot_key"] = name
                info["matrix_file"] = str(matrices_path)
    if blocked_reconstruction_errors is not None:
        summary["retained_energy"] = _retained_projection_energy(snapshots, reconstruction)
        summary["field_retained_energy"] = {
            field.name: _retained_projection_energy(
                snapshots[field.dofs, :], reconstruction[field.dofs, :]
            )
            for field in basis.fields if field.name != "supremizer"
        }
        summary["metric_definitions"] = {
            "retained_energy": "1 - ||S - reconstructed_S||_F^2 / ||S - mean(S)||_F^2; Euclidean coefficient variance, not physical energy",
            "reconstruction_relative_error": "||S - reconstructed_S||_F / ||S||_F; training snapshots, mixed units across fields",
            "field_retained_energy": "Same centered-variance fraction measured separately on each physical field, including any velocity enrichment",
        }
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
