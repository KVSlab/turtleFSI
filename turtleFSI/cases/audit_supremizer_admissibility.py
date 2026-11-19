"""Check Eq. (5) before and after imposing the ROM velocity constraints.

Performs auxiliary linear Riesz solves only; no POD training or time rollout.
Does not modify the production supremizer construction.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--basis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--time', type=float, default=10.01)
    args = parser.parse_args()
    from dolfin import (TrialFunction, TestFunction, Function, FunctionAssigner,
                        DirichletBC, Constant, assemble, inner, grad, inv, tr, solve)
    from turtleFSI.modules.common import F_, G_, J_, K_
    from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem
    from turtleFSI.rom._pod_galerkin.problem_overrides import replay_overrides
    from turtleFSI.rom._pod_galerkin.monolithic_pod import load_monolithic_pod_basis
    source = args.run_dir/'ROM/dvp_snapshots.npz'
    problem = FEniCSMonolithicProblem.build('TF_fsi_reference', overrides=replay_overrides(source))
    with np.load(source) as data:
        index = int(np.argmin(abs(data['time']-args.time)))
        if abs(data['time'][index]-args.time) > 1e-8:
            raise ValueError('Time must be a saved snapshot')
        problem.set_state(data['dvp'][:, index])
        velocity_dofs = data['velocity_dofs'].copy()
    ns = problem.namespace
    V, collapse_map = problem.DVP.sub(1).collapse(collapsed_dofs=True)
    mixed_indices = np.array([collapse_map[i] for i in range(V.dim())])
    fixed = np.flatnonzero(np.isin(mixed_indices, problem.dirichlet_dofs()))
    free = np.setdiff1d(np.arange(V.dim()), fixed)
    # TF_fsi velocity tests vanish on walls, inlet, cylinder and clamped root.
    # The fluid-solid interface and outlet are not fixed velocity boundaries.
    bcs = [DirichletBC(V, Constant((0., 0.)), ns['boundaries'], marker) for marker in (2, 3, 6, 7)]
    bc_dofs = np.array(sorted({i for bc in bcs for i in bc.get_boundary_values()}))
    np.testing.assert_array_equal(bc_dofs, fixed)
    trial, test = TrialFunction(V), TestFunction(V)
    a = (inner(trial, test)+inner(grad(trial), grad(test)))*ns['dx']
    X = assemble(a)
    d, pressure, shape = ns['d_']['n'], ns['p_']['n'], ns.get('shape_displacement')
    form = sum(-pressure*K_(d, shape)*J_(d, shape)*
               tr(inv(G_(d, shape))*inv(F_(d, shape))*grad(test))*ns['dx_f'][i]
               for i in range(len(ns['dx_f_id_list'])))
    b = assemble(form)
    unconstrained, admissible = Function(V), Function(V)
    solve(X, unconstrained.vector(), b)
    A, rhs = X.copy(), b.copy()
    for bc in bcs:
        bc.apply(A, rhs)
    solve(A, admissible.vector(), rhs)
    clipped = Function(V)
    clipped.assign(unconstrained)
    for bc in bcs:
        bc.apply(clipped.vector())

    def metrics(f):
        values = f.vector().get_local()
        residual = (X*f.vector()-b).get_local()
        return dict(boundary_coefficient_norm=float(np.linalg.norm(values[fixed])),
                    free_equation_relative_residual=float(np.linalg.norm(residual[free])/
                                                         max(np.linalg.norm(b.get_local()[free]), 1e-30)))

    def h1norm(values):
        f = Function(V)
        f.vector().set_local(values)
        f.vector().apply('insert')
        return float(np.sqrt(max(f.vector().inner(X*f.vector()), 0.)))

    difference = clipped.vector().get_local()-admissible.vector().get_local()
    report = dict(run_dir=str(args.run_dir.resolve()), time=args.time,
                  fixed_velocity_dofs=len(fixed), global_velocity_dofs=V.dim(),
                  unconstrained=metrics(unconstrained), clipped_after_solve=metrics(clipped),
                  constrained_solve=metrics(admissible),
                  clipped_vs_constrained_relative_H1=h1norm(difference)/h1norm(admissible.vector().get_local()),
                  interpretation='Clipping an unconstrained solution is not a constrained Riesz solve.')
    # Verify the production snapshot solve has really been reproduced.
    cached = args.basis.parent.parent/'training_supremizers_0.npz'
    if cached.exists():
        with np.load(cached) as data:
            j = int(np.argmin(abs(data['time']-args.time)))
            if abs(data['time'][j]-args.time) < 1e-8:
                mixed = Function(problem.DVP)
                FunctionAssigner(problem.DVP.sub(1), V).assign(mixed.sub(1), unconstrained)
                actual = mixed.vector().get_local()[velocity_dofs]
                saved = data['supremizer_snapshots'][:, j]
                report['recomputed_vs_cached_relative_error'] = float(np.linalg.norm(actual-saved)/max(np.linalg.norm(saved),1e-30))
    basis = load_monolithic_pod_basis(args.basis)
    field = next(f for f in basis.fields if f.name == 'supremizer')
    raw = basis.pod_basis.basis[:, field.start:field.stop]
    constrained = problem.constrain_increment_basis(raw)
    report['supremizer_modes_boundary_squared_fraction'] = (
        np.sum(raw[mixed_indices[fixed]]**2, axis=0)/np.sum(raw**2, axis=0)).tolist()
    report['constrained_enrichment_gram_eigenvalues'] = np.linalg.eigvalsh(constrained.T@constrained).tolist()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'admissibility.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
