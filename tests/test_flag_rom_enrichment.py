from types import SimpleNamespace
import numpy as np
import pytest


def test_kinematic_enrichment_preserves_span_and_adds_only_independent_velocity_modes():
    df = pytest.importorskip("dolfin")
    from turtleFSI.cases.validate_flag_rom import enrich_velocity_with_displacement
    mesh = df.UnitSquareMesh(2, 2)
    element = df.VectorElement("CG", mesh.ufl_cell(), 2)
    W = df.FunctionSpace(mesh, df.MixedElement([element, element, df.FiniteElement("CG", mesh.ufl_cell(), 1)]))
    D, dm = W.sub(0).collapse(collapsed_dofs=True)
    V, vm = W.sub(1).collapse(collapsed_dofs=True)
    di = np.array([dm[i] for i in range(D.dim())])
    vi = np.array([vm[i] for i in range(V.dim())])
    rng = np.random.default_rng(42)
    phi = np.zeros((W.dim(), 4))
    phi[di, :2] = np.linalg.qr(rng.normal(size=(D.dim(), 2)))[0]
    phi[vi, 2:] = np.linalg.qr(rng.normal(size=(V.dim(), 2)))[0]
    problem = SimpleNamespace(DVP=W, constrain_increment_basis=lambda x: x.copy())
    basis = SimpleNamespace(fields=[SimpleNamespace(name="displacement", start=0, stop=2)])
    enriched, count = enrich_velocity_with_displacement(problem, basis, phi)
    assert count == 2
    np.testing.assert_array_equal(enriched[:, :4], phi)
    velocity = enriched[vi, 2:]
    target = phi[di, :2]
    np.testing.assert_allclose(velocity@np.linalg.lstsq(velocity, target, rcond=None)[0], target, atol=1e-12)
    nonvelocity = np.setdiff1d(np.arange(W.dim()), vi)
    np.testing.assert_array_equal(enriched[nonvelocity, 4:], 0)
    _, count = enrich_velocity_with_displacement(problem, basis, enriched)
    assert count == 0
