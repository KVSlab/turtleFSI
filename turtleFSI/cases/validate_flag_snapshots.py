"""Verify the three serial CG2/CG2/CG1 reference-flag smoke-test exports.

Usage: python -m turtleFSI.cases.validate_flag_snapshots /path/to/flag-geometry-tests.ID
Reads the run artifacts without modifying them. This checks representation and
finite snapshot values, not time accuracy or ROM operator correctness.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.cases.flag_geometry import TEST_CASES, signed_double_area


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, description):
    require(np.array_equal(actual, expected), description + " differs")


def validate(run_directory, cases=None):
    from dolfin import (Mesh, MeshFunction, HDF5File, MPI, FunctionSpace,
                        MixedElement, VectorElement, FiniteElement,
                        VectorFunctionSpace, Function, facets)
    from turtleFSI.cases.snapshots.dvp_export import active_pressure_dofs

    require(MPI.size(MPI.comm_world) == 1, "Run this checker on one MPI rank")
    root = Path(run_directory) / "results"
    baseline = None
    reports = {}
    for name, (radius, length) in (TEST_CASES if cases is None else cases).items():
        folder = root / name
        try:
            with np.load(folder / "Mesh/geometry_map.npz", allow_pickle=False) as archive:
                geometry = {key: archive[key] for key in archive.files}
            with np.load(folder / "ROM/dvp_snapshots.npz", allow_pickle=False) as archive:
                snapshots = {key: archive[key] for key in archive.files}
            summary = json.loads((folder / "ROM/dvp_snapshots_summary.json").read_text())
            report = json.loads((folder / "Mesh/geometry_map.json").read_text())
            require(summary["problem"] == "TF_fsi_reference", "Not a reference-domain run")
            require(summary["mpi_size"] == 1, "Expected serial snapshots")
            require(report["radius"] == radius and report["flag_length"] == length,
                    "Unexpected geometry parameters")

            mesh = Mesh()
            with HDF5File(MPI.comm_world, str(folder / "Mesh/mesh.h5"), "r") as hdf:
                hdf.read(mesh, "/mesh", False)
                domains = MeshFunction("size_t", mesh, 2)
                boundaries = MeshFunction("size_t", mesh, 1)
                hdf.read(domains, "/domains")
                hdf.read(boundaries, "/boundaries")
            same(mesh.coordinates(), geometry["reference_coordinates"], "Saved reference coordinates")
            same(mesh.cells(), geometry["cells"], "Saved connectivity")
            same(domains.array(), geometry["cell_markers"], "Saved cell markers")
            same(boundaries.array(), geometry["facet_markers"], "Saved facet markers")
            mesh.init(1, 2)
            interface_count = 0
            for facet in facets(mesh):
                labels = set(domains.array()[facet.entities(2)])
                require((labels == {1, 2}) == (boundaries[facet] == 5),
                        "Interface markers disagree with fluid/solid adjacency")
                interface_count += int(boundaries[facet] == 5)
            require(interface_count > 0, "No fluid-solid interface")

            cell = mesh.ufl_cell()
            space = FunctionSpace(mesh, MixedElement([VectorElement("CG", cell, 2),
                                  VectorElement("CG", cell, 2), FiniteElement("CG", cell, 1)]))
            signature = {key: geometry[key] for key in
                         ("reference_coordinates", "cells", "cell_markers", "facet_markers")}
            signature["cell_dofs"] = np.array([space.dofmap().cell_dofs(i)
                                               for i in range(mesh.num_cells())])
            for index, field in enumerate(("displacement", "velocity", "pressure")):
                key = field + "_dofs"
                same(snapshots[key], space.sub(index).dofmap().dofs(), key)
                signature[key] = snapshots[key]
            active = active_pressure_dofs(space, domains, [1])
            for key in ("fluid_pressure_dofs", "pressure_active_dofs"):
                same(snapshots[key], active, key)
                signature[key] = snapshots[key]
            values, times = snapshots["dvp"], snapshots["time"]
            require(times.ndim == 1 and len(times) > 0, "No snapshot times")
            require(values.shape == (space.dim(), len(times)), "Snapshot matrix has wrong dimensions")
            require(np.isfinite(values).all() and np.isfinite(times).all(), "Non-finite snapshots")
            require(np.all(np.diff(times) > 0), "Times are not strictly increasing")
            require(snapshots["counter"].shape == times.shape and
                    np.all(np.diff(snapshots["counter"]) > 0), "Invalid snapshot counters")
            require(summary["num_snapshots"] == len(times) and
                    summary["num_total_dofs"] == space.dim(), "Summary dimensions disagree")
            signature.update(time=times, counter=snapshots["counter"])
            if baseline is None:
                baseline = signature
            else:
                for key, expected in baseline.items():
                    same(signature[key], expected, "Cross-case " + key)

            shape = Function(VectorFunctionSpace(mesh, "CG", 1))
            with HDF5File(MPI.comm_world, str(folder / "Mesh/shape_displacement.h5"), "r") as hdf:
                hdf.read(shape, "/shape_displacement")
            shift = shape.compute_vertex_values(mesh).reshape(2, -1).T
            target = geometry["mapped_coordinates"]
            require(np.isfinite(target).all() and np.isfinite(shift).all(), "Non-finite shape map")
            np.testing.assert_allclose(mesh.coordinates() + shift, target, rtol=0, atol=1e-12)
            old, new = mesh.coordinates()[mesh.cells()], target[mesh.cells()]
            B0 = np.stack((old[:, 1]-old[:, 0], old[:, 2]-old[:, 0]), axis=2)
            B = np.stack((new[:, 1]-new[:, 0], new[:, 2]-new[:, 0]), axis=2)
            G = B @ np.linalg.inv(B0)
            K = np.linalg.det(G)
            np.testing.assert_allclose(geometry["G"], G, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(geometry["K"], K, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(K, signed_double_area(target, mesh.cells()) /
                                       signed_double_area(mesh.coordinates(), mesh.cells()),
                                       rtol=1e-11, atol=1e-12)
            require(K.min() >= .2, "Initial map fails the area-ratio threshold")
            if radius == .05 and length == .35:
                np.testing.assert_allclose(shift, 0, rtol=0, atol=1e-14)
            else:
                require(np.linalg.norm(shift, axis=1).max() > 1e-8, "Variant map is identity")
            circle = np.unique(np.concatenate([f.entities(0) for f in facets(mesh)
                                               if boundaries[f] in (6, 7)]))
            np.testing.assert_allclose(np.linalg.norm(target[circle]-[.2, .2], axis=1),
                                       radius, rtol=0, atol=1e-9)
            solid = np.unique(mesh.cells()[domains.array() == 2])
            np.testing.assert_allclose(target[solid, 1], mesh.coordinates()[solid, 1], rtol=0, atol=1e-12)
            np.testing.assert_allclose(target[solid, 0].max(), .2 + radius + length, rtol=0, atol=1e-12)
            reports[name] = dict(dofs=space.dim(), snapshots=len(times),
                                 time_start=float(times[0]), time_end=float(times[-1]),
                                 min_K=float(K.min()),
                                 max_shape_shift=float(np.linalg.norm(shift, axis=1).max()))
            print("PASS %s: %d DoFs, %d snapshots, t=%g..%g, min K=%.6g" %
                  (name, space.dim(), len(times), times[0], times[-1], K.min()))
        except (ValueError, AssertionError, KeyError, OSError, RuntimeError) as error:
            raise ValueError("%s: %s" % (name, error)) from error
    print("PASS: common reference meshes, reconstructed cell DoF maps, exported field indices, "
          "time grids, finite snapshots, and parameter-specific shape maps.")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    try:
        validate(args.run_directory)
    except (ValueError, OSError) as error:
        parser.exit(1, "FAIL: %s\n" % error)


if __name__ == "__main__":
    main()
