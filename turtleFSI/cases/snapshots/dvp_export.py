"""Exact mixed-space DoF snapshot export for POD-Galerkin workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from dolfin import MPI
except ModuleNotFoundError:
    MPI = None


def _mpi4py_comm(comm):
    try:
        return comm.tompi4py()
    except AttributeError:
        return None


def gather_vector_to_root(vector) -> np.ndarray | None:
    """Gather a distributed FEniCS vector into global DoF order on rank 0."""

    if MPI is None:
        raise ModuleNotFoundError("dvp_export requires dolfin for vector gathering")

    comm = vector.mpi_comm()
    mpi_comm = _mpi4py_comm(comm)
    local_values = np.asarray(vector.get_local(), dtype=float)
    start, stop = vector.local_range()
    global_size = int(vector.size())

    if mpi_comm is None:
        if MPI.rank(comm) == 0:
            values = np.zeros(global_size, dtype=float)
            values[start:stop] = local_values
            return values
        return None

    gathered = mpi_comm.gather((int(start), int(stop), local_values), root=0)
    if mpi_comm.Get_rank() != 0:
        return None

    values = np.zeros(global_size, dtype=float)
    for row_start, row_stop, row_values in gathered:
        values[row_start:row_stop] = row_values
    return values


def _subspace_dofs(DVP, index: int) -> np.ndarray:
    try:
        return np.asarray(DVP.sub(index).dofmap().dofs(), dtype=np.int64)
    except Exception:
        return np.zeros(0, dtype=np.int64)


def active_pressure_dofs(DVP, domains, fluid_domain_ids) -> np.ndarray:
    """Return mixed pressure DoFs attached to cells in the fluid domain."""

    from dolfin import cells, vertex_to_dof_map

    if fluid_domain_ids is None:
        return np.zeros(0, dtype=np.int64)
    if not isinstance(fluid_domain_ids, (list, tuple)):
        fluid_ids = {int(fluid_domain_ids)}
    else:
        fluid_ids = {int(value) for value in fluid_domain_ids}

    pressure_space, collapsed_to_mixed = DVP.sub(2).collapse(collapsed_dofs=True)
    vertex_to_pressure_dof = vertex_to_dof_map(pressure_space)
    active_collapsed_dofs: set[int] = set()
    for cell in cells(pressure_space.mesh()):
        if int(domains[cell.index()]) not in fluid_ids:
            continue
        for vertex in cell.entities(0):
            active_collapsed_dofs.add(int(vertex_to_pressure_dof[vertex]))

    return np.asarray(
        sorted(int(collapsed_to_mixed[dof]) for dof in active_collapsed_dofs),
        dtype=np.int64,
    )


def _field_names(DVP) -> list[str]:
    names = ["displacement", "velocity", "pressure"]
    count = DVP.num_sub_spaces()
    if count > 3:
        names.extend(f"auxiliary_{i}" for i in range(3, count))
    return names[:count]


def _write_snapshot_bundle(
    output_path: Path,
    summary_path: Path,
    snapshots: list[np.ndarray],
    times: list[float],
    counters: list[int],
    DVP,
    metadata: dict[str, Any],
    supremizer_snapshots: list[np.ndarray] | None = None,
    active_pressure_dofs_array: np.ndarray | None = None,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "dvp": np.asarray(snapshots, dtype=float).T,
        "time": np.asarray(times, dtype=float),
        "counter": np.asarray(counters, dtype=np.int64),
    }
    if supremizer_snapshots:
        arrays["supremizer_snapshots"] = np.asarray(
            supremizer_snapshots,
            dtype=float,
        ).T
    if active_pressure_dofs_array is not None and active_pressure_dofs_array.size:
        arrays["fluid_pressure_dofs"] = np.asarray(active_pressure_dofs_array, dtype=np.int64)
        arrays["pressure_active_dofs"] = np.asarray(active_pressure_dofs_array, dtype=np.int64)

    fields = {}
    for index, name in enumerate(_field_names(DVP)):
        dofs = _subspace_dofs(DVP, index)
        arrays[f"{name}_dofs"] = dofs
        fields[name] = {
            "subspace_index": index,
            "num_dofs": int(dofs.size),
            "dof_key": f"{name}_dofs",
        }

    np.savez_compressed(output_path, **arrays)

    summary = {
        **metadata,
        "output_path": str(output_path),
        "num_snapshots": int(len(snapshots)),
        "num_total_dofs": int(arrays["dvp"].shape[0]),
        "time_start": float(times[0]) if times else None,
        "time_end": float(times[-1]) if times else None,
        "fields": fields,
        "includes_supremizer_snapshots": bool(supremizer_snapshots),
        "supremizer_snapshot_key": "supremizer_snapshots"
        if supremizer_snapshots
        else None,
        "notes": [
            "dvp contains exact mixed FEniCS DoF vectors in global vector order.",
            "Field DoF arrays map each named subspace into the mixed vector.",
            "fluid_pressure_dofs maps pressure DoFs on cells where the fluid pressure equation is assembled.",
            "supremizer_snapshots contains velocity-space supremizer vectors on velocity DoFs when enabled.",
            "Supremizers solve a global velocity Riesz problem with the fluid ALE incompressibility adjoint as in Ballarin/Rozza.",
            "This export is intended for intrusive POD-Galerkin basis construction.",
        ],
    }
    with open(summary_path, "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def save_dvp_rom_snapshot(
    dvp_,
    DVP,
    t,
    counter,
    results_folder,
    problem,
    extrapolation,
    rom_export_folder="ROM",
    rom_dvp_snapshots=None,
    rom_dvp_times=None,
    rom_dvp_counters=None,
    **namespace,
) -> dict[str, Any]:
    """Save the current mixed solution vector for POD-Galerkin training.

    The function accumulates snapshots in memory and rewrites a compressed
    ``dvp_snapshots.npz`` bundle on rank 0. This keeps the first implementation
    simple and robust for moderate snapshot counts.
    """

    if MPI is None:
        raise ModuleNotFoundError("save_dvp_rom_snapshot requires dolfin")

    snapshots = [] if rom_dvp_snapshots is None else rom_dvp_snapshots
    times = [] if rom_dvp_times is None else rom_dvp_times
    counters = [] if rom_dvp_counters is None else rom_dvp_counters
    global_vector = gather_vector_to_root(dvp_["n"].vector())
    if MPI.rank(MPI.comm_world) == 0:
        output_dir = Path(results_folder) / rom_export_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshots.append(global_vector)
        times.append(float(t))
        counters.append(int(counter))
        active_pressure = None
        if "domains" in namespace and "dx_f_id_list" in namespace:
            active_pressure = active_pressure_dofs(
                DVP=DVP,
                domains=namespace["domains"],
                fluid_domain_ids=namespace["dx_f_id_list"],
            )
        _write_snapshot_bundle(
            output_path=output_dir / "dvp_snapshots.npz",
            summary_path=output_dir / "dvp_snapshots_summary.json",
            snapshots=snapshots,
            times=times,
            counters=counters,
            DVP=DVP,
            metadata={
                "mpi_size": int(MPI.size(MPI.comm_world)),
                "problem": str(problem),
                "extrapolation": str(extrapolation),
                "results_folder": str(results_folder),
                **({"time_stepping": "adaptive_convergence",
                    "snapshot_interval": namespace["adaptive_output_interval"],
                    "snapshot_start": namespace["adaptive_output_start"]}
                   if namespace.get("adaptive_time_step", False) else {}),
            },
            active_pressure_dofs_array=active_pressure,
        )

    return {
        "rom_dvp_snapshots": snapshots,
        "rom_dvp_times": times,
        "rom_dvp_counters": counters,
    }
