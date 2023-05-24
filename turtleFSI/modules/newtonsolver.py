# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import assemble, derivative, TrialFunction, Matrix, norm, MPI, PETScOptions

PETScOptions.set("mat_mumps_icntl_4", 1) # If negatvie or zero, MUMPS will suppress diagnositc printining, statistics, and warning messages. 
PETScOptions.set("mat_mumps_icntl_14", 400) # allocate more memory to mumps


def solver_setup(F_fluid_linear, F_fluid_nonlinear, F_solid_linear, F_solid_nonlinear,
                 DVP, dvp_, up_sol, compiler_parameters, **namespace):
    """
    Pre-assemble the system of equations for the Jacobian matrix for the Newton solver
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

    return dict(F=F, J_nonlinear=J_nonlinear, A_pre=A_pre, A=A, b=b, up_sol=up_sol)


def newtonsolver(F, J_nonlinear, A_pre, A, b, bcs, lmbda, recompute, recompute_tstep, compiler_parameters,
                 dvp_, up_sol, dvp_res, rtol, atol, max_it, counter, first_step_num, verbose, **namespace):
    """
    Solve the non-linear system of equations with Newton scheme. The standard is to compute the Jacobian
    every time step, however this is computationally costly. We have therefore added two parameters for
    re-computing only every 'recompute' iteration, or for every 'recompute_tstep' time step. Setting 'recompute'
    to != 1 is faster, but can impact the convergence rate. Altering 'recompute_tstep' is considered an advanced option,
    and should be used with care.
    """
    # Initial values
    iter = 0
    residual = 10**8
    rel_res = 10**8

    # Capture if residual increases from last iteration
    last_rel_res = residual
    last_residual = rel_res
    up_sol.set_operator(A)
    while rel_res > rtol and residual > atol and iter < max_it:
        # Check if recompute Jacobian from 'recompute_tstep' (time step)
        recompute_for_timestep = iter == 0 and (counter % recompute_tstep == 0)

        # Check if recompute Jacobian from 'recompute' (iteration)
        recompute_frequency = iter > 0 and iter % recompute == 0

        # Recompute Jacobian due to increased residual
        recompute_residual = iter > 0 and last_residual < residual

        # Recompute Jacobian on first step of simulation (important if restart is used)
        recompute_initialize = iter == 0 and counter == first_step_num

        if recompute_for_timestep or recompute_frequency or recompute_residual or recompute_initialize:
            if MPI.rank(MPI.comm_world) == 0 and verbose:
                print("Compute Jacobian matrix")
            # Assemble non-linear part of Jacobian, keep sparsity pattern (keep_diagonal=True)
            # Here, we assume that A is already assembled with the linear part of the Jacobian, and not None type
            assemble(J_nonlinear, tensor=A,
                         form_compiler_parameters=compiler_parameters,
                         keep_diagonal=True)
            # Add non-linear and linear part of Jacobian
            A.axpy(1.0, A_pre, True)
            # Insert ones on diagonal to make sure the matrix is non-singular (related to solid pressure being zero)
            A.ident_zeros()
            [bc.apply(A) for bc in bcs]

        # Aseemble right hand side vector
        if b is None:
            b = assemble(-F)
        else:
            assemble(-F, tensor=b)

        # Apply boundary conditions before solve
        [bc.apply(b, dvp_["n"].vector()) for bc in bcs]
        # Solve the linear system A * x = b where A is the Jacobian matrix, x is the Newton increment and b is the -residual
        up_sol.solve(dvp_res.vector(), b)
        # Update solution using the Newton increment
        dvp_["n"].vector().axpy(lmbda, dvp_res.vector())
        # After adding the residual to the solution, we need to re-apply the boundary conditions
        # because the residual (dvp_res.vector) is not guaranteed to be zero on the boundary
        [bc.apply(dvp_["n"].vector()) for bc in bcs]

        # Reset residuals
        last_residual = residual
        last_rel_res = rel_res

        # Check residual
        residual = b.norm('l2')
        rel_res = norm(dvp_res, 'l2')
        if rel_res > 1E20 or residual > 1E20:
            raise RuntimeError("Error: The simulation has diverged during the Newton solve.")

        if MPI.rank(MPI.comm_world) == 0 and verbose:
            print("Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) "
                  % (iter, residual, atol, rel_res, rtol))
        iter += 1

    return dict(up_sol=up_sol, A=A)
