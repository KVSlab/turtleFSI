"""Rebase a reference-flag ROM without changing its physical ALE equation."""
import numpy as np
from dolfin import FunctionSpace, Function, cells, grad, inner, derivative, TrialFunction
from turtleFSI.modules.common import J_, paper_fluid_indicator
from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem


def build_identity_problem(run, overrides, reference):
    problem = FEniCSMonolithicProblem.build('TF_fsi_rebased', overrides=overrides)
    ns = problem.namespace
    # Same topology is insufficient: verify every cell's mixed DoF ordering.
    for cell in cells(ns['mesh']):
        np.testing.assert_array_equal(
            problem.DVP.dofmap().cell_dofs(cell.index()),
            reference.DVP.dofmap().cell_dofs(cell.index()))
    np.testing.assert_array_equal(problem.dirichlet_dofs(), reference.dirichlet_dofs())
    np.testing.assert_array_equal(problem.fluid_pressure_dofs(), reference.fluid_pressure_dofs())
    # Rebasings change G to I, but the original paper extension has diffusivity
    # 1/(J*K_original). Dropping K_original would change the ALE equation.
    Q = FunctionSpace(ns['mesh'], 'DG', 0)
    K = Function(Q)
    values = K.vector().get_local()
    with np.load(run / 'Mesh/geometry_map.npz') as data:
        for cell in cells(ns['mesh']):
            values[Q.dofmap().cell_dofs(cell.index())[0]] = data['K'][cell.index()]
    K.vector().set_local(values)
    K.vector().apply('insert')
    d, d0, phi = ns['d_']['n'], ns['d_']['n-1'], ns['phi']
    stress = grad(d) / (J_(d) * K)
    extension = sum((inner((d-d0)/ns['k'], phi) + inner(stress, grad(phi))) * ns['dx_f'][i]
                    for i in range(len(ns['dx_f_id_list'])))
    indicator = paper_fluid_indicator(ns['dx_f'], ns['dx_f_id_list'])
    markers = ns['fsi_id'] if isinstance(ns['fsi_id'], (list, tuple)) else [ns['fsi_id']]
    flux = sum(indicator(side)*inner(stress(side)*ns['n'](side), phi(side)) for side in ('+', '-'))
    extension -= sum(flux * ns['dS'](marker) for marker in markers)
    original = ns['extension_paper_terms']['extension_relation_f'] + ns['extension_paper_terms']['a_e_df']
    problem.F = ns['F'] = problem.F - original + extension
    problem.J = ns['J'] = derivative(problem.F, problem.dvp_['n'], TrialFunction(problem.DVP))
    ns['identity_check_extension_K'] = K
    return problem


def compare_operators(reference, identity, phi, states, times, first, dt):
    """Compare at identical states, including a nonzero reduced perturbation."""
    records = []
    for index in sorted(set((first, min(first+1, len(times)-1), len(times)-1))):
        previous = states[:, index]
        trial = previous + phi @ (np.sin(np.arange(phi.shape[1])+1)*1e-5)
        for problem in (reference, identity):
            problem.set_previous_state(previous)
            problem.set_previous_previous_state(previous)
            problem.set_time(float(times[index]+dt))
        np.testing.assert_allclose(reference.apply_state_bcs(trial), identity.apply_state_bcs(trial), atol=1e-12)
        R = [p.residual(trial) for p in (reference, identity)]
        A = [p.projected_jacobian(phi, trial) for p in (reference, identity)]
        def difference(a, b):
            return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), 1e-30))
        record = dict(time=float(times[index]+dt), residual_relative_difference=difference(*R),
                      projected_jacobian_relative_difference=difference(*A))
        records.append(record)
        if max(record['residual_relative_difference'], record['projected_jacobian_relative_difference']) > 1e-8:
            raise ValueError('Identity-map operator equivalence failed: %s' % record)
    return records
