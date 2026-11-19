"""Validate flag geometries and common DoF maps without a time simulation."""

import argparse
import json
from pathlib import Path

import numpy as np
from turtleFSI.cases.flag_geometry import TEST_CASES, build_flag_mesh


def validate(output_dir, cases=None):
    from dolfin import FunctionSpace, MixedElement, VectorElement, FiniteElement, facets

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline = None
    reports = {}
    for name, (radius, length) in (TEST_CASES if cases is None else cases).items():
        mesh, domains, boundaries, report, reference = build_flag_mesh(radius, length)
        cell = mesh.ufl_cell()
        space = FunctionSpace(mesh, MixedElement([
            VectorElement("CG", cell, 2), VectorElement("CG", cell, 2), FiniteElement("CG", cell, 1),
        ]))
        signature = [mesh.cells().copy(), domains.array().copy(), boundaries.array().copy(),
                     np.array([space.dofmap().cell_dofs(i) for i in range(mesh.num_cells())]),
                     reference]
        if baseline is None:
            baseline = signature
        for expected, actual in zip(baseline, signature):
            np.testing.assert_array_equal(actual, expected)
        # Confirm physical geometry independently of the mapping construction.
        circle = np.unique(np.concatenate([f.entities(0) for f in facets(mesh) if boundaries[f] in (6, 7)]))
        radial_error = np.abs(np.linalg.norm(mesh.coordinates()[circle] - [0.2, 0.2], axis=1) - radius)
        if radial_error.max() > 1e-9:
            raise ValueError("Circle boundary vertices miss the requested radius")
        wall = np.unique(np.concatenate([f.entities(0) for f in facets(mesh) if boundaries[f] in (2, 3, 4)]))
        np.testing.assert_array_equal(mesh.coordinates()[wall], reference[wall])
        solid = np.unique(mesh.cells()[domains.array() == 2])
        np.testing.assert_allclose(mesh.coordinates()[solid, 1], reference[solid, 1], atol=1e-12, rtol=0)
        np.testing.assert_allclose(mesh.coordinates()[solid, 0].max(), 0.2 + radius + length, atol=1e-12, rtol=0)
        report.update(num_dofs=space.dim(), displacement_dofs=space.sub(0).dim(),
                      velocity_dofs=space.sub(1).dim(), pressure_dofs=space.sub(2).dim(),
                      topology_tags_and_cell_dofmaps_equal=True,
                      max_circle_vertex_radius_error=float(radial_error.max()))
        np.savez_compressed(output / (name + ".npz"), reference_coordinates=reference,
                            mapped_coordinates=mesh.coordinates(), cells=mesh.cells(),
                            cell_markers=domains.array(), facet_markers=boundaries.array(),
                            cell_dofs=signature[-1])
        reports[name] = report
    (output / "validation.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radii", nargs="+", type=float,
                        help="Circle radii; used with --lengths to validate their Cartesian product")
    parser.add_argument("--lengths", nargs="+", type=float, help="Flag lengths in metres")
    args = parser.parse_args()
    if (args.radii is None) != (args.lengths is None):
        parser.error("Supply both --radii and --lengths")
    cases = None if args.radii is None else {
        "radius_%02d_length_%02d" % (i, j): (radius, length)
        for i, radius in enumerate(args.radii) for j, length in enumerate(args.lengths)}
    validate(args.output_dir, cases)


if __name__ == "__main__":
    main()
