# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import assemble, derivative, TrialFunction, Matrix, norm, MPI


def solver_setup(F_fluid_linear, F_fluid_nonlinear, F_solid_linear, F_solid_nonlinear,
                 DVP, dvp_, up_sol, compiler_parameters, **namespace):
    """
    Pre-assemble the system of equations for the Jacobian matrix for the newton solver
    """
    F_lin = F_fluid_linear + F_solid_linear
    F_nonlin = F_solid_nonlinear + F_fluid_nonlinear
    F = F_lin + F_nonlin

    chi = TrialFunction(DVP)
    J_linear = derivative(F_lin, dvp_["n"], chi)
    J_nonlinear = derivative(F_nonlin, dvp_["n"], chi)

    A_pre = assemble(J_linear, form_compiler_parameters=compiler_parameters,
                     keep_diagonal=True)
    A = Matrix(A_pre)
    b = None

    # Option not available in FEniCS 2018.1.0
    # up_sol.parameters['reuse_factorization'] = True

    return dict(F=F, J_nonlinear=J_nonlinear, A_pre=A_pre, A=A, b=b, up_sol=up_sol)


def newtonsolver(F, J_nonlinear, A_pre, A, b, bcs, lmbda, recompute, recompute_tstep, compiler_parameters,
                 dvp_, up_sol, dvp_res, rtol, atol, max_it, counter, verbose, **namespace):
    """
    Solve the non-linear system of equations with Newton iterations scheme.
    There are two ways to control the reuse do the Jacobian matrix:
        - reused over iteration steps and re-evaluated at every "recompute" iteration steps. (faster, but may impact the rate of convergence)
        - reused over time steps and re-evaluated at every "recompute_tstep" time steps. (advanced option, "recompute_tstep=1" is preferred)
    """
    # Initial values
    Iter = 0
    residual = 10**8
    rel_res = 10**8

    # Capture if residual increases from last iteration
    last_rel_res = residual
    last_residual = rel_res

    while rel_res > rtol and residual > atol and Iter < max_it:
        # Check if recompute Jacobian from 'recompute_tstep' (time step)
        recompute_for_timestep = Iter == 0 and ((counter-1) % recompute_tstep == 0)

        # Check if recompute Jacobian from 'recompute' (iteration)
        recompute_frequency = Iter > 0 and Iter % recompute == 0

        # Recompute Jacobian due to increased residual
        recompute_residual = Iter > 0 and  (last_rel_res < rel_res or last_residual < residual)

        if recompute_for_timestep or recompute_frequency or recompute_residual:
            if MPI.rank(MPI.comm_world) == 0 and verbose:
                print("Compute Jacobian matrix")
            A = assemble(J_nonlinear, tensor=A,
                         form_compiler_parameters=compiler_parameters,
                         keep_diagonal=True)
            A.axpy(1.0, A_pre, True)
            A.ident_zeros()
            [bc.apply(A) for bc in bcs]
            up_sol.set_operator(A)

        # Compute right hand side
        b = assemble(-F, tensor=b)

        # Reset residuals
        last_rel_res = rel_res
        last_residual = residual

        # Apply boundary conditions and solve
        [bc.apply(b, dvp_["n"].vector()) for bc in bcs]
        up_sol.solve(dvp_res.vector(), b)
        dvp_["n"].vector().axpy(lmbda, dvp_res.vector())
        [bc.apply(dvp_["n"].vector()) for bc in bcs]

        # Check residual
        rel_res = norm(dvp_res, 'l2')
        residual = b.norm('l2')
        if rel_res > 1E20 or residual > 1E20:
            raise RuntimeError("Error: The simulation has diverged during the Newton solve.")

        if MPI.rank(MPI.comm_world) == 0 and verbose:
            print("Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) "
                  % (Iter, residual, atol, rel_res, rtol))
        Iter += 1

    return dict(up_sol=up_sol, A=A)
