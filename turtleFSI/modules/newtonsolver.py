# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import assemble, derivative, TrialFunction, Matrix, norm, MPI

def solver_setup(F_fluid_linear, F_fluid_nonlinear, F_solid_linear, F_solid_nonlinear,
                 DVP, dvp_, up_sol, **monolithic):
    F_lin = F_fluid_linear + F_solid_linear
    F_nonlin = F_solid_nonlinear + F_fluid_nonlinear
    F = F_lin + F_nonlin

    chi = TrialFunction(DVP)
    J_linear = derivative(F_lin, dvp_["n"], chi)
    J_nonlinear = derivative(F_nonlin, dvp_["n"], chi)

    A_pre = assemble(J_linear, form_compiler_parameters=compiler_parameters)
    A = Matrix(A_pre)
    b = None

    up_sol.parameters['reuse_factorization'] = True

    return dict(F=F, J_nonlinear=J_nonlinear, A_pre=A_pre, A=A, b=b, up_sol=up_sol)


def newtonsolver(F, J_nonlinear, A_pre, A, b, bcs, lmbda, recompute,
                 dvp_, up_sol, dvp_res, rtol, atol, max_it, T, t, **monolithic):
    # Initial values
    Iter = 0
    residual = 10**8
    rel_res = 10**8

    # Capture if residual increases from last iteration
    last_rel_res = residual
    last_residual = rel_res

    while rel_res > rtol and residual > atol and Iter < max_it:
        # Check if recompute Jacobian
        if Iter % recompute == 0 or (last_rel_res < rel_res and last_residual < residual
                                     and last_residual < rel_res):
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

        if MPI.rank(MPI.comm_world) == 0:
            print("Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) "
                  % (Iter, residual, atol, rel_res, rtol))
        Iter += 1

    return {}
