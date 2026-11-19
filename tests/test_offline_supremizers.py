"""Regression coverage for moving supremizer solves out of snapshot export."""
import json

import numpy as np
import pytest

from turtleFSI.rom.compute_supremizers import compute_bundle


@pytest.fixture
def snapshot_run(tmp_path):
    df = pytest.importorskip("dolfin")
    mesh = df.UnitSquareMesh(2, 2)
    domains = df.MeshFunction("size_t", mesh, 2, 1)
    (tmp_path / "Mesh").mkdir()
    (tmp_path / "Checkpoint").mkdir()
    with df.HDF5File(mesh.mpi_comm(), str(tmp_path / "Mesh/mesh.h5"), "w") as f:
        f.write(mesh, "/mesh")
        f.write(domains, "/domains")
    (tmp_path / "Checkpoint/default_variables.json").write_text(json.dumps({
        "d_deg": 2, "v_deg": 2, "p_deg": 1, "extrapolation": "laplace",
        "dx_f_id": 1,
    }))
    cell = mesh.ufl_cell()
    space = df.FunctionSpace(mesh, df.MixedElement([
        df.VectorElement("CG", cell, 2), df.VectorElement("CG", cell, 2),
        df.FiniteElement("CG", cell, 1),
    ]))
    maps = {f"{name}_dofs": np.asarray(space.sub(i).dofmap().dofs())
            for i, name in enumerate(("displacement", "velocity", "pressure"))}
    states = np.zeros((space.dim(), 2))
    states[maps["pressure_dofs"], 1] = 1.0
    source = tmp_path / "snapshots.npz"
    np.savez(source, dvp=states, time=[0.0, 0.1], counter=[0, 1], **maps)
    return tmp_path, source


def test_offline_supremizers_preserve_snapshots_and_respond_to_pressure(snapshot_run):
    run, source = snapshot_run
    original_bytes = source.read_bytes()
    output = compute_bundle(run, source, run / "enriched.npz")
    assert source.read_bytes() == original_bytes
    with np.load(source) as old, np.load(output) as new:
        for key in old.files:
            np.testing.assert_array_equal(old[key], new[key])
        s = new["supremizer_snapshots"]
        assert s.shape == (old["velocity_dofs"].size, 2)
        np.testing.assert_allclose(s[:, 0], 0, atol=1e-14)
        assert np.linalg.norm(s[:, 1]) > 0
    summary = json.loads(output.with_name("enriched_summary.json").read_text())
    assert summary["supremizer_computation"] == "offline"
    assert summary["num_snapshots"] == 2


def test_offline_supremizers_reject_overwriting_source(snapshot_run):
    run, source = snapshot_run
    with pytest.raises(ValueError, match="separate output"):
        compute_bundle(run, source, source)


def test_offline_supremizers_reject_wrong_dof_order(snapshot_run):
    run, source = snapshot_run
    with np.load(source) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["velocity_dofs"] = arrays["velocity_dofs"][::-1]
    np.savez(source, **arrays)
    with pytest.raises(ValueError, match="DoF ordering"):
        compute_bundle(run, source, run / "enriched.npz")
    assert not (run / "enriched.npz").exists()


def test_supremizer_uses_shape_jacobian():
    df = pytest.importorskip("dolfin")
    from turtleFSI.rom.compute_supremizers import compute_velocity_supremizer_snapshot
    mesh = df.UnitSquareMesh(2, 2)
    cell = mesh.ufl_cell()
    W = df.FunctionSpace(mesh, df.MixedElement([df.VectorElement("CG", cell, 2),
                         df.VectorElement("CG", cell, 2), df.FiniteElement("CG", cell, 1)]))
    d = df.Function(df.VectorFunctionSpace(mesh, "CG", 2))
    g = df.interpolate(df.Expression((".3*x[0]", ".1*x[1]"), degree=1),
                       df.VectorFunctionSpace(mesh, "CG", 1))
    measure = df.Measure("dx", domain=mesh)
    actual = compute_velocity_supremizer_snapshot(W, df.Constant(1), measure,
                                                  d_current=d, shape_displacement=g)
    V = W.sub(1).collapse()
    s, v = df.TrialFunction(V), df.TestFunction(V)
    expected = df.Function(V)
    df.solve((df.inner(s, v)+df.inner(df.grad(s), df.grad(v)))*measure ==
             -(1.1*v[0].dx(0)+1.3*v[1].dx(1))*measure, expected)
    mixed = df.Function(W)
    df.FunctionAssigner(W.sub(1), V).assign(mixed.sub(1), expected)
    np.testing.assert_allclose(actual, mixed.vector().get_local()[W.sub(1).dofmap().dofs()], atol=1e-12)

