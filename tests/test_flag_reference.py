"""Independent change-of-variables checks for the flag reference map."""
import numpy as np
import pytest

pytest.importorskip("dolfin")
from dolfin import (FunctionSpace, VectorFunctionSpace, Function, TrialFunction,
                    TestFunction, Mesh, assemble, dx, grad, inner, inv, cells)
from turtleFSI.cases.flag_geometry import build_flag_reference, TEST_CASES
from turtleFSI.modules.common import F_


@pytest.mark.parametrize("parameters", list(TEST_CASES.values()) + [(.075, .5)])
def test_reference_map_and_change_of_variables(parameters):
    data = build_flag_reference(*parameters)
    mesh, g = data["mesh"], data["shape_displacement"]
    np.testing.assert_array_equal(mesh.coordinates(), data["reference_coordinates"])
    mapped = Mesh(mesh)
    mapped.coordinates()[:] = data["mapped_coordinates"]
    np.testing.assert_allclose(g.compute_vertex_values(mesh).reshape(2, -1).T,
                               mapped.coordinates() - mesh.coordinates(), atol=1e-14)
    G, K = data["shape_G"], data["shape_K"]
    assert data["cell_K"].min() >= .2
    # Compare UFL derivatives against independently computed affine Jacobians.
    # Integrating with DG0 tests avoids point evaluation of UFL derivatives.
    Q = FunctionSpace(mesh, "DG", 0)
    q = TestFunction(Q)
    volumes = assemble(q * dx).get_local()
    for i in range(2):
        for j in range(2):
            values = assemble(G[i, j] * q * dx).get_local() / volumes
            expected = np.empty_like(values)
            for cell in cells(mesh):
                expected[Q.dofmap().cell_dofs(cell.index())[0]] = data["cell_G"][cell.index(), i, j]
            np.testing.assert_allclose(values, expected, atol=1e-12)
    # Mass and diffusion operators must match a direct mapped-domain assembly,
    # for both the P1 geometry and the P2 solution basis.
    for degree in (1, 2):
        V, Vm = FunctionSpace(mesh, "CG", degree), FunctionSpace(mapped, "CG", degree)
        for cell in cells(mesh):
            np.testing.assert_array_equal(V.dofmap().cell_dofs(cell.index()),
                                          Vm.dofmap().cell_dofs(cell.index()))
        u, v = TrialFunction(V), TestFunction(V)
        um, vm = TrialFunction(Vm), TestFunction(Vm)
        a = K * (u*v + inner(inv(G).T * grad(u), inv(G).T * grad(v))) * dx
        am = (um*vm + inner(grad(um), grad(vm))) * dx
        # Sparse matrix difference avoids allocating dense flag-sized matrices.
        A, B = assemble(a), assemble(am)
        A.axpy(-1, B, True)
        assert A.norm("frobenius") / B.norm("frobenius") < 1e-11
    zero = Function(VectorFunctionSpace(mesh, "CG", 2))
    from dolfin import Identity
    assert assemble(inner(F_(zero, g)-Identity(2), F_(zero, g)-Identity(2))*dx) < 1e-24


@pytest.mark.parametrize("parameters", [TEST_CASES["baseline"], TEST_CASES["longer_flag"], (.075, .5)])
def test_reference_and_mapped_fsi_steps(parameters):
    """Compare two timesteps, including the same paper extension diffusivity."""
    from dolfin import (MeshFunction, MixedElement, VectorElement, FiniteElement,
                        split, TestFunctions, Constant, Measure, FacetNormal,
                        derivative, solve)
    from turtleFSI.modules.fluid import fluid_setup
    from turtleFSI.modules.solid import solid_setup
    from turtleFSI.modules.laplace import extrapolate_setup
    from turtleFSI.modules.common import J_, paper_fluid_indicator
    from turtleFSI.problems.TF_fsi import create_bcs

    data = build_flag_reference(*parameters)
    reference = data["mesh"]
    histories = []
    for use_reference in (True, False):
        mesh = reference if use_reference else Mesh(reference)
        if not use_reference:
            mesh.coordinates()[:] = data["mapped_coordinates"]
            mesh.bounding_box_tree().build(mesh)
        domains = MeshFunction("size_t", mesh, 2)
        domains.array()[:] = data["domains"].array()
        boundaries = MeshFunction("size_t", mesh, 1)
        boundaries.array()[:] = data["boundaries"].array()
        cell = mesh.ufl_cell()
        W = FunctionSpace(mesh, MixedElement([VectorElement("CG", cell, 2),
                         VectorElement("CG", cell, 2), FiniteElement("CG", cell, 1)]))
        current, previous = Function(W), Function(W)
        d, v, p = split(current)
        d0, v0, p0 = split(previous)
        phi, psi, q = TestFunctions(W)
        volume = Measure("dx", domain=mesh, subdomain_data=domains)
        surface = Measure("dS", domain=mesh, subdomain_data=boundaries)
        normal = FacetNormal(mesh)
        ns = dict(mesh=mesh, d_={"n": d, "n-1": d0}, v_={"n": v, "n-1": v0},
                  p_={"n": p, "n-1": p0}, phi=phi, psi=psi, gamma=q,
                  dx_f=[volume(1)], dx_f_id_list=[1], dx_s=[volume(2)], dx_s_id_list=[2],
                  fluid_properties=[dict(rho_f=1000., mu_f=1.)],
                  solid_properties=[dict(rho_s=1000., mu_s=2.e6, lambda_s=4.e6,
                                         material_model="LinearElastic")],
                  k=Constant(.01), theta=1., gravity=None, robin_bc=False,
                  k_s=[], c_s=[], ds_s=[], ds_s_ext_id_list=[],
                  extrapolation_sub_type="paper", dS=surface, n=normal, fsi_id=5,
                  shape_displacement=data["shape_displacement"] if use_reference else None)
        ns.update(fluid_setup(**ns))
        ns.update(solid_setup(**ns))
        if use_reference:
            ns.update(extrapolate_setup(**ns))
            extension = 0
        else:
            # The paper defines diffusivity as 1/(J*K), even when assembled
            # directly on the parameter-specific initial domain. Keep that K.
            Q = FunctionSpace(mesh, "DG", 0)
            K = Function(Q)
            values = K.vector().get_local()
            for c in cells(mesh):
                values[Q.dofmap().cell_dofs(c.index())[0]] = data["cell_K"][c.index()]
            K.vector().set_local(values)
            K.vector().apply("insert")
            stress = grad(d) / (J_(d) * K)
            extension = (inner((d-d0)/ns["k"], phi) + inner(stress, grad(phi))) * volume(1)
            fluid = paper_fluid_indicator([volume(1)], [1])
            extension -= sum(fluid(side) * inner(stress(side)*normal(side), phi(side))
                             for side in ("+", "-")) * surface(5)
        residual = extension + sum(ns[key] for key in
                                   ("F_fluid_linear", "F_fluid_nonlinear",
                                    "F_solid_linear", "F_solid_nonlinear"))
        bc_data = create_bcs(W, 2, 2., .41, boundaries, "paper")
        samples = []
        for t in (.01, .02):
            bc_data["inlet"].update(t)
            solve(residual == 0, current, bc_data["bcs"],
                  J=derivative(residual, current, TrialFunction(W)),
                  form_compiler_parameters={"quadrature_degree": 5},
                  solver_parameters={"newton_solver": {"absolute_tolerance": 1e-10,
                                     "relative_tolerance": 1e-10, "maximum_iterations": 15}})
            samples.append(current.vector().get_local().copy())
            previous.assign(current)
        histories.append(samples)
    np.testing.assert_allclose(histories[0], histories[1], rtol=2e-7, atol=2e-9)
