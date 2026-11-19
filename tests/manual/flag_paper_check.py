"""Short integration check using TF_fsi geometry and the paper's unsteady model."""
from turtleFSI.problems.TF_fsi import *
from turtleFSI.problems import TF_fsi as benchmark
from pathlib import Path
import json
import csv


def set_problem_parameters(default_variables, **namespace):
    values = benchmark.set_problem_parameters(default_variables, **namespace)
    values.update(T=.5, dt=.01, theta=1., material_model='LinearElastic',
                  extrapolation='laplace', extrapolation_sub_type='paper', fsi_id=5,
                  recompute=1, recompute_tstep=1, atol=1.e-8, rtol=1.e-9, max_it=30,
                  folder='results/manual/flag-paper-check', save_step=5, checkpoint_step=50, verbose=True)
    return values


def initiate(DVP, mesh, d_, dx, **namespace):
    values = benchmark.initiate(**namespace)
    previous = Function(DVP)
    # For CG2 displacement in 2D, det(I+grad(d)) is cellwise quadratic.
    # DG2 projection gives values at its interpolation nodes (sampled extrema).
    VJ = FunctionSpace(mesh, 'DG', 2)
    j_trial, j_test = TrialFunction(VJ), TestFunction(VJ)
    j_value = Function(VJ)
    j_solver = LocalSolver(inner(j_trial,j_test)*dx, inner(J_(d_['n']),j_test)*dx)
    j_solver.factorize()
    values.update(check_previous=previous, check_history=[], check_j=j_value,
                  check_j_solver=j_solver)
    return values


def pre_solve(t, inlet, dvp_, check_previous, **namespace):
    inlet.update(t)
    check_previous.assign(dvp_['n-1'])


def post_solve(t, dvp_, F, bcs, coord, check_previous, check_history, check_j,
               check_j_solver, results_folder, DVP, mesh, compiler_parameters,
               J_nonlinear, A_pre, equation_3_jacobian, **namespace):
    # monolithic.py advances histories before post_solve. Temporarily restore
    # the old state to evaluate the residual for the step just completed.
    dvp_['n-1'].assign(check_previous)
    residual = assemble(F, form_compiler_parameters=compiler_parameters)
    for bc in bcs:
        bc.apply(residual, dvp_['n'].vector())
    residual_norm = residual.norm('l2')
    if not check_history:
        production = assemble(J_nonlinear, form_compiler_parameters=compiler_parameters)
        production.axpy(1., A_pre, False)
        paper = assemble(equation_3_jacobian, form_compiler_parameters=compiler_parameters)
        reference_norm = production.norm('frobenius')
        paper.axpy(-1., production, False)
        relative_difference = paper.norm('frobenius') / reference_norm
        print('PAPER_JACOBIAN_RELATIVE_DIFFERENCE', relative_difference, flush=True)
        if MPI.rank(mesh.mpi_comm()) == 0:
            Path(results_folder, 'jacobian_check.json').write_text(json.dumps({
                'relative_frobenius_difference':relative_difference,
                'cells':mesh.num_cells(), 'mixed_dofs':DVP.dim(),
            }, indent=2))
        assert relative_difference < 1.e-10
    dvp_['n-1'].assign(dvp_['n'])
    displacement = dvp_['n'].sub(0, deepcopy=True)
    tip = peval(displacement, coord)
    check_j_solver.solve_local_rhs(check_j)
    local_j = check_j.vector().get_local()
    jmin = MPI.min(mesh.mpi_comm(), float(local_j.min()))
    jmax = MPI.max(mesh.mpi_comm(), float(local_j.max()))
    state = dvp_['n'].vector().get_local()
    finite = bool(np.all(np.isfinite(state)))
    row = dict(time=float(t), tip_dx=float(tip[0]), tip_dy=float(tip[1]),
               residual_l2=float(residual_norm), sampled_J_min=jmin, sampled_J_max=jmax,
               finite_state=finite)
    check_history.append(row)
    if MPI.rank(mesh.mpi_comm()) == 0:
        target = Path(results_folder, 'diagnostics.csv')
        with target.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader();writer.writerows(check_history)
        print('FLAG_CHECK',json.dumps(row),flush=True)
    assert finite and jmin > 0, 'Nonfinite state or nonpositive sampled mapping Jacobian'
    assert residual_norm < 1.e-6, 'Post-update residual exceeds integration-check threshold'


def finished(results_folder, check_history, **namespace):
    if MPI.rank(MPI.comm_world) != 0:
        return
    summary = dict(
        status='passed', steps=len(check_history), final_time=check_history[-1]['time'],
        max_post_update_residual=max(row['residual_l2'] for row in check_history),
        min_sampled_J=min(row['sampled_J_min'] for row in check_history),
        max_sampled_J=max(row['sampled_J_max'] for row in check_history),
        final_tip_dx=check_history[-1]['tip_dx'], final_tip_dy=check_history[-1]['tip_dy'],
        max_abs_tip_dy=max(abs(row['tip_dy']) for row in check_history),
        model='TF_fsi geometry; linear elasticity; paper Laplace extension; theta=1',
        scope='Short startup integration check; not validation against published FSI3 values',
    )
    Path(results_folder, 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('FLAG_CHECK_SUMMARY',json.dumps(summary),flush=True)
