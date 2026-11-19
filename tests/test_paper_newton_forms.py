"""Regression checks for the Ballarin/Rozza equations through Section 2.4."""
import numpy as np
import pytest

pytest.importorskip("dolfin")
from dolfin import (
    Constant, Expression, FacetNormal, FiniteElement, Function, FunctionSpace,
    Measure, MeshFunction, MixedElement, TestFunctions, TrialFunction,
    UnitSquareMesh, VectorFunctionSpace, action, assemble, cells, facets, grad,
    interpolate, inv, near, split, tr,
)
from turtleFSI.modules.common import F_, G_, J_, K_
from turtleFSI.modules.fluid import fluid_setup
from turtleFSI.modules.solid import solid_setup
from turtleFSI.modules.laplace import extrapolate_setup
from turtleFSI.modules.newton_block_forms import build_equation_3_blocks
from turtleFSI.modules.newtonsolver import solver_setup
from turtleFSI.modules.equation_4 import build_equation_4_forms, assemble_equation_4
from dolfin import derivative

CP = {"quadrature_degree": 6}


@pytest.mark.parametrize("shape,force,interface", [
    (True, True, True), (False, False, True), (True, False, False),
])
def test_equation_4_named_assembly(case, shape, force, interface):
    base, _, setup = case
    ns = setup(shape_displacement=base["shape_displacement"] if shape else None,
               fluid_body_force=base["fluid_body_force"] if force else None,
               fsi_id=3 if interface else [])
    forms = build_equation_4_forms(**ns)
    assembled = assemble_equation_4(forms, CP)
    W, state = ns["DVP"], ns["dvp_"]["n"]
    expected = matrix(derivative(ns["F"], state, TrialFunction(W)))
    assert_close(assembled["jacobian"].array(), expected)
    assert_close(sum(block.array() for block in assembled["blocks"].values()), expected)
    # Each named block must act only on its declared test/trial fields.
    indices = {key: W.sub(i).dofmap().dofs() for i, key in enumerate("dup")}
    for name, block in assembled["blocks"].items():
        values = block.array()
        supported = np.zeros_like(values)
        ix = np.ix_(indices[name[2]], indices[name[3]])
        supported[ix] = values[ix]
        assert_close(values, supported)
    assert (assembled["operators"]["F_F_d"] is not None) == force
    # Verify the actual Newton action, not just equivalence of two form sums.
    h = direction(W)
    assert_close(vector(action(forms["jacobian"], h)),
                 directional_fd(ns["F"], state, h), rtol=2.e-8)
    # Forms must not freeze the state-dependent matrices at construction time.
    state.vector()[:] = 1.1 * state.vector().get_local()
    assert_close(matrix(forms["jacobian"]),
                 matrix(derivative(ns["F"], state, TrialFunction(W))))


@pytest.mark.parametrize("overrides", [
    {"theta": .5}, {"extrapolation_sub_type": "constant"},
    {"solid_properties": [{"material_model": "StVenantKirchhoff"}]},
])
def test_equation_4_rejects_nonpaper_models(case, overrides):
    base, _, _ = case
    ns = dict(base, **overrides)
    with pytest.raises(ValueError, match="Equation \\(4\\)"):
        build_equation_4_forms(**ns)


@pytest.mark.parametrize("constrain", [True, False])
def test_equation_6_named_projection_and_state_refresh(case, constrain, tmp_path):
    from dolfin import DirichletBC
    from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem
    from turtleFSI.rom._pod_galerkin.ballarin_monolithic import BallarinBlockBasis
    from turtleFSI.rom._pod_galerkin.offline_online import (
        ReducedLinearOperators, save_reduced_operator_bundle, load_reduced_operator_bundle,
    )
    _, _, setup = case
    ns = setup()
    W, state = ns["DVP"], ns["dvp_"]["n"]
    ns.update(J=derivative(ns["F"], state, TrialFunction(W)),
              compiler_parameters=CP,
              bcs=[DirichletBC(W.sub(1), Constant((0., 0.)), "on_boundary")])
    problem = FEniCSMonolithicProblem(ns)
    rng = np.random.default_rng(46)
    def block(index, count):
        result = np.zeros((W.dim(), count))
        dofs = W.sub(index).dofmap().dofs()
        result[dofs] = rng.normal(size=(len(dofs), count))
        return result
    basis = BallarinBlockBasis(block(0, 2), block(1, 2), block(2, 1),
                               np.zeros((W.dim(), 1)), block(1, 1))
    result = problem.projected_equation_6(basis, constrain_basis=constrain)
    Z = problem.constrain_increment_basis(basis.combined_basis) if constrain else basis.combined_basis
    Zu = Z[:, 2:5]
    AF = matrix(build_equation_4_forms(**ns)["operators"]["A_F"])
    assert_close(result["A_F_N"], Zu.T @ AF @ Zu)
    assert_close(result["J_N"], problem.projected_jacobian(Z))
    assert_close(result["equation_6_physical_J_N"], Z.T @ matrix(ns["J"]) @ Z)
    assert result["B_F_p_N"].shape == (3, 1)
    assert result["J_ud_N"].shape == (3, 2)
    path = tmp_path / "equation6.npz"
    operators = ReducedLinearOperators(basis, result["J_N"], np.zeros((6, 1)), (W.dim(), W.dim()))
    save_reduced_operator_bundle(path, operators, named_equation_6_operators=result)
    loaded = load_reduced_operator_bundle(path)
    assert_close(loaded.named_equation_6_operators["A_F_N"], result["A_F_N"])
    assert_close(loaded.J_N, result["J_N"])
    if not constrain:
        from turtleFSI.rom._shared.pod import PODBasis
        from turtleFSI.rom._pod_galerkin.monolithic_pod import MonolithicPODBasis, FieldLayout
        from turtleFSI.rom._pod_galerkin.build_ballarin_operator_bundle import build_reduced_operator_bundle
        ns["dt"] = .2
        ns["dvp_"]["n-2"] = Function(W)
        monolithic = MonolithicPODBasis(
            PODBasis(basis.mean, basis.combined_basis, np.ones(6), np.ones(6)),
            tuple(FieldLayout(name, start, stop, np.array(W.sub(index).dofmap().dofs()))
                  for name, start, stop, index in (
                      ("displacement", 0, 2, 0), ("velocity", 2, 4, 1),
                      ("supremizer", 4, 5, 1), ("pressure", 5, 6, 2))),
        )
        reference = state.vector().get_local().copy()
        summary = build_reduced_operator_bundle(
            problem, monolithic, reference[:, None], None, 0, path, "test",
            tmp_path / "basis.npz", tmp_path / "snapshots.npz",
            theta=1., equation_6_blocks=True,
        )
        assert summary["operator_source"] == "equation_6_named_operators"
        loaded = load_reduced_operator_bundle(path)
        assert_close(loaded.J_N, loaded.linear_operator_blocks.as_operator())
        assert_close(loaded.named_equation_6_operators["A_F_N"], result["A_F_N"])
    state.vector()[:] = state.vector().get_local() * 1.1
    refreshed = problem.projected_equation_6(basis, constrain_basis=constrain)
    assert not np.allclose(refreshed["A_F_N"], result["A_F_N"])
    assert_close(refreshed["J_N"], problem.projected_jacobian(Z))
    problem.J = 1.1 * ns["J"]
    with pytest.raises(ValueError, match="disagree"):
        problem.projected_equation_6(basis, constrain_basis=constrain)


def matrix(form):
    return assemble(form, form_compiler_parameters=CP).array()


def vector(form):
    return assemble(form, form_compiler_parameters=CP).get_local()


def assert_close(actual, expected, rtol=1.e-10):
    error = np.linalg.norm(actual - expected)
    assert error <= 1.e-12 + rtol * np.linalg.norm(expected), error


@pytest.fixture
def case():
    mesh = UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(mesh, "CG", 1)
    ve = V.ufl_element()
    W = FunctionSpace(mesh, MixedElement([ve, ve, FiniteElement("CG", mesh.ufl_cell(), 1)]))
    current, previous = Function(W), Function(W)
    # Continuous displacement with different gradients on either side of x=.5.
    current.interpolate(Expression((
        "0.1*x[0]+0.03*abs(x[0]-0.5)+0.04*x[1]", "0.02*x[0]+0.07*x[1]",
        "0.2+0.3*x[0]-0.1*x[1]", "0.1+0.2*x[0]+0.4*x[1]", "0.5+0.1*x[0]",
    ), degree=1))
    d, u, p = split(current)
    d_old, u_old, p_old = split(previous)
    phi, psi, q = TestFunctions(W)
    shape = interpolate(Expression(("0.2*x[0]+0.3*x[1]", "0.1*x[0]-0.1*x[1]"), degree=1), V)
    domains = MeshFunction("size_t", mesh, mesh.topology().dim(), 2)
    for cell in cells(mesh):
        if cell.midpoint().x() < .5:
            domains[cell] = 1
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    for facet in facets(mesh):
        if near(facet.midpoint().x(), .5):
            boundaries[facet] = 3
    dx = Measure("dx", domain=mesh, subdomain_data=domains)
    dS = Measure("dS", domain=mesh, subdomain_data=boundaries)
    base = dict(
        DVP=W, dvp_={"n": current, "n-1": previous},
        d_={"n": d, "n-1": d_old}, v_={"n": u, "n-1": u_old}, p_={"n": p, "n-1": p_old},
        phi=phi, psi=psi, gamma=q, dx_f=[dx(1)], dx_f_id_list=[1], dx_s=[dx(2)], dx_s_id_list=[2],
        fluid_properties=[dict(rho_f=1.3, mu_f=.7)],
        solid_properties=[dict(rho_s=2., mu_s=3., lambda_s=4., material_model="LinearElastic")],
        k=Constant(.2), theta=1., shape_displacement=shape, fluid_body_force=Constant((.2, -.1)),
        solid_body_force=None, extrapolation_sub_type="paper", mesh=mesh, gravity=None,
        robin_bc=False, k_s=[], c_s=[], ds_s=[], ds_s_ext_id_list=[], dS=dS, n=FacetNormal(mesh), fsi_id=3,
    )

    def setup(**overrides):
        ns = dict(base)
        ns.update(overrides)
        ns.update(fluid_setup(**ns))
        ns.update(solid_setup(**ns))
        ns.update(extrapolate_setup(**ns))
        ns.update(build_equation_3_blocks(**ns))
        ns["F"] = sum(ns[key] for key in (
            "F_fluid_linear", "F_fluid_nonlinear", "F_solid_linear", "F_solid_nonlinear"))
        return ns

    return base, dx, setup


def directional_fd(form, state, direction):
    initial = state.vector().get_local().copy()
    delta = direction.vector().get_local()
    epsilon = 1.e-6
    try:
        state.vector()[:] = initial + epsilon * delta
        plus = vector(form)
        state.vector()[:] = initial - epsilon * delta
        minus = vector(form)
    finally:
        state.vector()[:] = initial
    return (plus - minus) / (2 * epsilon)


def direction(W):
    h = Function(W)
    h.vector()[:] = .05 * np.random.default_rng(941).normal(size=W.dim())
    return h


@pytest.mark.parametrize("identity_map", [False, True])
def test_continuity_residual_and_derivative(case, identity_map):
    _, _, setup = case
    ns = setup(**({"shape_displacement": None} if identity_map else {}))
    d, u, shape = ns["d_"]["n"], ns["v_"]["n"], ns["shape_displacement"]
    # Chain rule for x_current = T(x) + d(x): H = G + grad(d).
    H = G_(d, shape) + grad(d)
    expected = K_(d, shape) * J_(d, shape) * tr(grad(u) * inv(H)) * ns["gamma"] * ns["dx_f"][0]
    actual = ns["fluid_paper_terms"]["b_f_continuity"]
    assert_close(vector(actual), vector(expected))
    hand = sum(ns["equation_3_component_blocks"]["fluid"]["fluid_continuity"].values())
    state, W = ns["dvp_"]["n"], ns["DVP"]
    assert_close(matrix(hand), matrix(derivative(actual, state, TrialFunction(W))))
    h = direction(W)
    assert_close(vector(action(hand, h)), directional_fd(expected, state, h), rtol=2.e-8)


def test_equation_3_total_and_solid_velocity_coupling(case):
    _, _, setup = case
    ns = setup()
    W, state = ns["DVP"], ns["dvp_"]["n"]
    expected = matrix(derivative(ns["F"], state, TrialFunction(W)))
    actual = matrix(ns["equation_3_jacobian"])
    assert_close(actual, expected)
    rows, cols = W.sub(0).dofmap().dofs(), W.sub(1).dofmap().dofs()
    assert np.linalg.norm(actual[np.ix_(rows, cols)]) > 1.e-3
    h = direction(W)
    assert_close(vector(action(ns["equation_3_jacobian"], h)), directional_fd(ns["F"], state, h), rtol=2.e-8)


@pytest.mark.parametrize("fluid_id", [1, 2])
def test_interface_uses_physical_fluid_side(case, fluid_id):
    _, dx, setup = case
    ns = setup(dx_f=[dx(fluid_id)], dx_f_id_list=[fluid_id], dx_s=[dx(3-fluid_id)], dx_s_id_list=[3-fluid_id])
    # Independently compute the constant flux for the piecewise affine field.
    G = np.array([[1.2, .3], [.1, .9]])
    D = np.array([[.1 + (-.03 if fluid_id == 1 else .03), .04], [.02, .07]])
    Gi = np.linalg.inv(G)
    J = np.linalg.det(np.eye(2) + D @ Gi)
    outward = np.array([1. if fluid_id == 1 else -1., 0.])
    flux = D @ Gi @ Gi.T @ outward / J
    probe = interpolate(Constant((1., 0., 0., 0., 0.)), ns["DVP"])
    interface = ns["extension_paper_terms"]["a_e_df"]
    actual = assemble(action(interface, probe), form_compiler_parameters=CP)
    assert actual == pytest.approx(-flux[0], rel=1.e-12, abs=1.e-12)
    hand = ns["equation_3_component_blocks"]["extension"]["extension_fluid"]["delta_d_a_e_df"]
    assert_close(matrix(hand), matrix(derivative(interface, ns["dvp_"]["n"], TrialFunction(ns["DVP"]))))
    # Both routes must agree even if the block builder constructs its own indicator.
    ns.pop("fluid_indicator")
    rebuilt = build_equation_3_blocks(**ns)
    assert_close(matrix(hand), matrix(rebuilt["equation_3_component_blocks"]["extension"]["extension_fluid"]["delta_d_a_e_df"]))


@pytest.mark.parametrize("overrides", [{"fsi_id": None}, {"dS": None}, {"n": None}])
def test_paper_interface_missing_inputs_are_rejected(case, overrides):
    _, _, setup = case
    with pytest.raises(ValueError, match="Paper extension requires"):
        setup(**overrides)


def test_paper_interface_requires_cell_markers(case):
    base, _, setup = case
    with pytest.raises(ValueError, match="cell markers"):
        setup(dx_f=[Measure("dx", domain=base["mesh"])])


def test_explicit_no_interface_and_nonpaper_mode(case):
    _, _, setup = case
    ns = setup(fsi_id=[], dS=None, n=None)
    assert ns["extension_paper_terms"]["a_e_df"] == 0
    ns = setup(extrapolation_sub_type="constant", fsi_id=None, dS=None, n=None)
    assert ns["fluid_indicator"] is None


@pytest.mark.parametrize("theta,force,change_previous", [
    (1., (.2, -.1), False), (1., None, False), (.5, None, True),
])
def test_production_jacobian_reassembly(case, theta, force, change_previous):
    _, _, setup = case
    ns = setup(theta=theta, fluid_body_force=None if force is None else Constant(force))
    solver = solver_setup(**ns, up_sol=None, compiler_parameters=CP)
    state = ns["dvp_"]["n"]
    cached = solver["A_pre"].array().copy()
    if change_previous:
        ns["dvp_"]["n-1"].vector()[:] = state.vector().get_local()
    else:
        state.vector()[:] = 1.5 * state.vector().get_local()
    # Exercise the actual in-place assembly path, not just addition of dense
    # arrays: the allocated sparsity must contain every nonlinear coupling.
    assembled = solver["A"]
    assemble(solver["J_nonlinear"], tensor=assembled,
             form_compiler_parameters=CP, keep_diagonal=True)
    assembled.axpy(1., solver["A_pre"], False)
    production = assembled.array()
    exact = matrix(derivative(solver["F"], state, TrialFunction(ns["DVP"])))
    assert_close(production, exact)
    assert_close(cached, solver["A_pre"].array())


@pytest.mark.parametrize("term,keys", [
    ("m_f", ["delta_d_m_f", "delta_u_m_f"]),
    ("a_f", ["delta_d_a_f", "delta_u_a_f"]),
    ("b_f_momentum", ["delta_d_b_f", "delta_p_b_f"]),
    ("c_f", ["delta_d_c_f", "delta_u_c_f"]),
    ("c_f_ALE", ["delta_d_c_f_ALE", "delta_u_c_f_ALE"]),
    ("f_f", ["delta_d_f_f"]),
])
def test_fluid_momentum_derivatives(case, term, keys):
    _, _, setup = case
    ns = setup()
    blocks = ns["equation_3_component_blocks"]["fluid"]["fluid_momentum"]
    hand = sum(blocks[key] for key in keys)
    residual = ns["fluid_paper_terms"][term]
    W, state = ns["DVP"], ns["dvp_"]["n"]
    assert_close(matrix(hand), matrix(derivative(residual, state, TrialFunction(W))))
    h = direction(W)
    assert_close(vector(action(hand, h)), directional_fd(residual, state, h), rtol=2.e-8)
