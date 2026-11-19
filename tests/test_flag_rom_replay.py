import json
import numpy as np
import pytest


@pytest.mark.parametrize("mapping", ["harmonic_v1", "radius_then_length_v2"])
@pytest.mark.parametrize("source", ["reference_run_dir", "restart_folder"])
def test_saved_geometry_mapping_version(mapping, source, tmp_path):
    df = pytest.importorskip("dolfin")
    from turtleFSI.cases.flag_geometry import build_flag_reference
    from turtleFSI.problems.TF_fsi_reference import setup_reference_geometry
    data = build_flag_reference(.05, .3675, mapping=mapping)
    folder = tmp_path / "Mesh"
    folder.mkdir()
    report = dict(data["report"])
    if mapping == "harmonic_v1":
        del report["mapping_version"]  # Historic artifacts have no version.
    (folder / "geometry_map.json").write_text(json.dumps(report))
    np.savez(folder / "geometry_map.npz",
             reference_coordinates=data["reference_coordinates"],
             mapped_coordinates=data["mapped_coordinates"], cells=data["mesh"].cells(),
             cell_markers=data["domains"].array(), facet_markers=data["boundaries"].array())
    with df.HDF5File(data["mesh"].mpi_comm(), str(folder / "shape_displacement.h5"), "w") as hdf:
        hdf.write(data["shape_displacement"], "/shape_displacement")
    result = setup_reference_geometry(data["mesh"], data["domains"], data["boundaries"],
                                      .05, .3675, "laplace", "paper",
                                      rom_replay=True,
                                      **{source: tmp_path if source == "reference_run_dir"
                                         else tmp_path / "Checkpoint"})
    np.testing.assert_allclose(result["shape_displacement"].vector().get_local(),
                               data["shape_displacement"].vector().get_local(), atol=1e-12)


def test_relocated_checkpoint(tmp_path):
    from turtleFSI.rom._pod_galerkin.problem_overrides import replay_overrides
    (tmp_path / "Checkpoint").mkdir()
    (tmp_path / "ROM").mkdir()
    (tmp_path / "Mesh").mkdir()
    (tmp_path / "Checkpoint/default_variables.json").write_text(json.dumps({"dt": .01, "R": .075}))
    (tmp_path / "ROM/dvp_snapshots_summary.json").write_text(json.dumps({"results_folder": "/missing/aero/path"}))
    (tmp_path / "Mesh/shape_displacement.h5").touch()
    settings = replay_overrides(tmp_path / "ROM/dvp_snapshots.npz")
    assert settings["dt"] == .01
    assert settings["R"] == .075
    assert settings["reference_run_dir"] == str(tmp_path)


@pytest.mark.parametrize("radius", [.05, .0525])
def test_reference_geometry_enters_rom_forms(radius, tmp_path):
    df = pytest.importorskip("dolfin")
    from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem
    from turtleFSI.modules.common import K_
    problem = FEniCSMonolithicProblem.build("TF_fsi_reference", overrides={
        "R": radius, "d_deg": 2, "v_deg": 2, "p_deg": 1,
        "results_folder": tmp_path, "verbose": False})
    ns = problem.namespace
    shape = ns["shape_displacement"]
    magnitude = np.linalg.norm(shape.vector().get_local())
    assert (magnitude == 0) if radius == .05 else (magnitude > 0)
    assert not list(tmp_path.iterdir()), "ROM replay must not write simulation artifacts"
    assert shape in ns["F"].coefficients()
    assert df.assemble(K_(shape, shape)*df.dx) > 0
