"""Compute velocity supremizer snapshots offline: X s_i = B_F(d_i)^T p_i."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.cases.snapshots.dvp_export import gather_vector_to_root, _subspace_dofs


def _iter_measures(measures):
    if measures is None:
        return []
    if isinstance(measures, dict):
        return [measures[key] for key in sorted(measures)]
    if isinstance(measures, (list, tuple)):
        return list(measures)
    return [measures]

# equation 5: supremizer enrichment

def compute_velocity_supremizer_snapshot(
    DVP,
    p_current,
    dx_measure,
    fluid_dx_measures=None,
    d_current=None,
    shape_displacement=None,
):
    """Solve the auxiliary velocity supremizer problem for the current pressure.

    The exported vector is restricted to the velocity subspace DoFs. The ROM
    basis builder embeds it back into the mixed DVP vector on ``velocity_dofs``.

    This follows the Ballarin/Rozza monolithic FSI construction:

    ``X s = B_F(d)^T p``.

    The Riesz operator ``X`` is assembled on the global velocity space so the
    resulting supremizer is continuous through the fluid-solid interface, while
    the right-hand side uses only the fluid incompressibility operator. For ALE
    FSI runs, ``B_F(d)`` uses ``JK tr(G^-1 F^-1 grad(u))``, matching the
    transformed full fluid residual. Shape fields are read from the saved run.
    """

    from dolfin import (
        Function,
        FunctionAssigner,
        TestFunction,
        TrialFunction,
        div,
        grad,
        inner,
        inv,
        solve,
    )
    from turtleFSI.modules.common import F_, J_, G_, K_
    from dolfin import tr

    V = DVP.sub(1).collapse()
    s_trial = TrialFunction(V)
    v_test = TestFunction(V)
    s = Function(V)
    pressure = p_current
    fluid_measures = _iter_measures(fluid_dx_measures) or [dx_measure]
    B_F_T_p = 0
    for fluid_dx in fluid_measures:
        if d_current is None:
            B_F_T_p -= pressure * div(v_test) * fluid_dx
        else:
            B_F_T_p -= (
                pressure * K_(d_current, shape_displacement) * J_(d_current, shape_displacement)
                * tr(inv(G_(d_current, shape_displacement)) * inv(F_(d_current, shape_displacement))
                     * grad(v_test)) * fluid_dx
            )
    X = inner(s_trial, v_test) * dx_measure + inner(grad(s_trial), grad(v_test)) * dx_measure
    solve(X == B_F_T_p, s)

    mixed_supremizer = Function(DVP)
    assigner = FunctionAssigner(DVP.sub(1), V)
    assigner.assign(mixed_supremizer.sub(1), s)
    gathered = gather_vector_to_root(mixed_supremizer.vector())
    if gathered is None:
        return None
    velocity_dofs = _subspace_dofs(DVP, 1)
    return gathered[velocity_dofs]


def compute_bundle(run_dir: Path, input_path: Path, output_path: Path) -> Path:
    """Enrich a serial DVP bundle using the run's saved reference mesh and settings."""
    from dolfin import (
        MPI, Mesh, HDF5File, MeshFunction, VectorElement, FiniteElement,
        MixedElement, FunctionSpace, Function, Measure, split,
    )

    if MPI.size(MPI.comm_world) != 1:
        raise ValueError("Offline supremizer construction currently requires one MPI rank")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Use a separate output path to preserve the original snapshots")
    if output_path.suffix != ".npz":
        raise ValueError("Output must have a .npz extension")
    summary_path = input_path.with_name(input_path.stem + "_summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if summary.get("mpi_size", 1) != 1:
        raise ValueError("Parallel snapshot DoF ordering is not supported by this serial command")
    settings = json.loads((run_dir / "Checkpoint/default_variables.json").read_text())
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), str(run_dir / "Mesh/mesh.h5"), "r") as handle:
        handle.read(mesh, "/mesh", False)
        domains = MeshFunction("size_t", mesh, mesh.topology().dim())
        handle.read(domains, "/domains")
    cell = mesh.ufl_cell()
    de = VectorElement("CG", cell, int(settings["d_deg"]))
    ve = VectorElement("CG", cell, int(settings["v_deg"]))
    pe = FiniteElement("CG", cell, int(settings["p_deg"]))
    elements = [de, ve, pe]
    if settings["extrapolation"] == "biharmonic":
        elements.append(de)
    space = FunctionSpace(mesh, MixedElement(elements))
    with np.load(input_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    states = arrays["dvp"]
    if states.ndim != 2 or states.shape[0] != space.dim() or states.shape[1] == 0:
        raise ValueError("Snapshot dimensions do not match the saved mixed function space")
    if not np.isfinite(states).all():
        raise ValueError("Snapshots contain nonfinite values")
    for i, name in enumerate(("displacement", "velocity", "pressure")):
        if not np.array_equal(arrays[f"{name}_dofs"], _subspace_dofs(space, i)):
            raise ValueError(f"Saved {name} DoF ordering does not match the reconstructed space")
    dx = Measure("dx", domain=mesh, subdomain_data=domains)
    properties = settings.get("fluid_properties")
    ids = [item["dx_f_id"] for item in properties] if properties else settings["dx_f_id"]
    if not isinstance(ids, (list, tuple)):
        ids = [ids]
    if not ids:
        raise ValueError("No fluid domains configured")
    state = Function(space)
    shape = None
    shape_path = run_dir / "Mesh/shape_displacement.h5"
    if settings.get("problem") == "TF_fsi_reference" or shape_path.exists():
        from dolfin import VectorFunctionSpace
        shape = Function(VectorFunctionSpace(mesh, "CG", 1))
        with HDF5File(mesh.mpi_comm(), str(shape_path), "r") as handle:
            handle.read(shape, "/shape_displacement")
    fields = split(state)
    supremizers = np.empty((arrays["velocity_dofs"].size, states.shape[1]))
    for index in range(states.shape[1]):
        state.vector().set_local(states[:, index])
        state.vector().apply("insert")
        supremizers[:, index] = compute_velocity_supremizer_snapshot(
            space, fields[2], dx, [dx(int(value)) for value in ids], fields[0], shape,
        )
        print(f"Computed supremizer {index + 1}/{states.shape[1]}", flush=True)
    arrays["supremizer_snapshots"] = supremizers
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the expensive result only after all solves succeed.
    temporary = output_path.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    summary.update({
        "output_path": str(output_path),
        "source_snapshots": str(input_path),
        "includes_supremizer_snapshots": True,
        "supremizer_snapshot_key": "supremizer_snapshots",
        "supremizer_equation": "X s_i = B_F(d_i)^T p_i",
        "supremizer_computation": "offline",
        "num_snapshots": int(states.shape[1]),
    })
    output_path.with_name(output_path.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Defaults to RUN/ROM/dvp_snapshots.npz")
    parser.add_argument("--output", type=Path, help="Defaults to RUN/ROM/dvp_supremizers.npz")
    args = parser.parse_args()
    output = compute_bundle(
        args.run_dir,
        args.input or args.run_dir / "ROM/dvp_snapshots.npz",
        args.output or args.run_dir / "ROM/dvp_supremizers.npz",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
