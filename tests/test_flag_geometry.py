"""Geometry, labeling and DoF identity on the actual flag mesh."""

import numpy as np
import pytest

pytest.importorskip("dolfin")

from turtleFSI.cases.flag_geometry import build_flag_mesh, REFERENCE, LEGACY_MAPPING
from turtleFSI.cases.validate_flag_geometry import validate
from turtleFSI.problems import TF_fsi


def test_baseline_identity():
    original, domains, boundaries = TF_fsi.get_mesh_domain_and_boundaries(**REFERENCE)
    mapped, new_domains, new_boundaries, report, reference = build_flag_mesh()
    np.testing.assert_array_equal(mapped.coordinates(), original.coordinates())
    np.testing.assert_array_equal(reference, original.coordinates())
    np.testing.assert_array_equal(domains.array(), new_domains.array())
    np.testing.assert_array_equal(boundaries.array(), new_boundaries.array())
    assert report["min_cell_area_ratio"] == 1.0


def test_three_geometries(tmp_path):
    reports = validate(tmp_path)
    assert {r["num_dofs"] for r in reports.values()} == {21749}
    for report in reports.values():
        assert report["min_cell_area_ratio"] >= 0.2
        assert report["min_triangle_quality"] >= 0.02
    assert reports["longer_flag"]["max_vertex_shift"] > 0
    assert reports["larger_circle"]["max_vertex_shift"] > 0


@pytest.mark.parametrize("radius,length", [(0.001, 0.35), (0.3, 0.35), (0.05, -0.1), (0.05, 3), (float("nan"), 0.35)])
def test_invalid_parameters(radius, length):
    with pytest.raises(ValueError):
        build_flag_mesh(radius, length)


def test_excessive_deformation_is_rejected():
    # A tip inside the channel alone is insufficient: interior cells invert.
    with pytest.raises(ValueError, match="Rejected mesh"):
        build_flag_mesh(0.05, 1.8)


def test_large_parameter_grid(tmp_path):
    cases = {"r%d_l%d" % (i, j): (radius, length)
             for i, radius in enumerate(np.linspace(.05, .075, 5))
             for j, length in enumerate(np.linspace(.35, .5, 5))}
    reports = validate(tmp_path, cases)
    assert len(reports) == 25
    assert {r["num_dofs"] for r in reports.values()} == {21749}
    assert min(r["min_cell_area_ratio"] for r in reports.values()) >= .2
    assert min(r["min_triangle_quality"] for r in reports.values()) >= .02


def test_legacy_map_still_available():
    _, _, _, report, _ = build_flag_mesh(.05, .3675, mapping=LEGACY_MAPPING)
    assert report["min_cell_area_ratio"] == pytest.approx(.685773, abs=1e-6)
    with pytest.raises(ValueError, match="Rejected mesh"):
        build_flag_mesh(.05, .5, mapping=LEGACY_MAPPING)
