from dolfin import TrialFunction, derivative, assemble, MPI, mpi_comm_world, norm

"""Implementation of Newton-Raphson method.

The implementation of the naive Newton-Raphson solver serves as a starting point for solving the the 
full FSI variational problem. The second order convergence of Newton-Raphson, 
is traded against time consuming assembly of the jacobian of F, where F is the FSI variational form residual.
The naive.py module also serves as a skeleton, for implementation of a general newtonsolver in turtleFSI.

For high resolution mesh, higher order elements, or a combination, the Newton-Raphson method
will often increase computational time and resources tremendously. Hence, Quasi-Newton methods can be helpful 
for decreasing computational time. Some examples of Quasi-Newton methods are provided in the current folder.

"""

def solver_setup(F_fluid_linear, F_fluid_nonlinear,
                 F_solid_linear, F_solid_nonlinear, DVP, dvp_, **kwargs) -> dict:

    """Solver setup called once, prior solving time solver.

    Args:
        F_fluid_linear: The linear parts of the fluid variational form.
        F_fluid_nonlinear: The non-linear parts of the fluid variational form.
        F_solid_linear: The non-linear parts of the solid variational form.
        F_solid_nonlinear: The non-linear parts of the solid variational form.
        DVP: MixedVectorSpace for deformation, velocity, and pressure.
        dvp_: MixedVectorSpace function for time-step "n", which is defined as the solution
        the system is converging against. (n-1 is the prior solution)
        **kwargs:

    Returns:
        Dict containing full FSI variational form residual, and its Jacobian.

    """

    F_lin = F_fluid_linear + F_solid_linear
    F_nonlin = F_fluid_nonlinear + F_solid_nonlinear
    F = F_lin + F_nonlin

    chi = TrialFunction(DVP)
    Jac = derivative(F, dvp_["n"], chi)

    return dict(F=F, Jac=Jac)


def newtonsolver(F, Jac, bcs, dvp_, lu_solver, dvp_res,
                 rtol, atol, max_it, **kwargs):
    """Newtonsolver called at each time-step in monolithic.py.

    Args:
        F: Residual of the full FSI variational formulation.
        Jac: Jacobian of residual F
        bcs: List of boundary conditions defined in problem file.
        dvp_: MixedFunctionSpace for deformation, velocity, and pressure.
        lu_solver: LUSolver class used for solving Newton-Raphson method.
        dvp_res: The solution vector in Newton-Raphson method Ax=b.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        max_it: Maximum iterations for Newton-Raphson method.
        **kwargs:

    """
    Iter = 0
    residual = 1
    rel_res = residual
    lmbda = 1

    while rel_res > rtol and residual > atol and Iter < max_it:
        A = assemble(Jac)
        A.ident_zeros()
        b = assemble(-F)

        [bc.apply(A, b, dvp_["n"].vector()) for bc in bcs]
        lu_solver.solve(A, dvp_res.vector(), b)
        dvp_["n"].vector().axpy(lmbda, dvp_res.vector())
        [bc.apply(dvp_["n"].vector()) for bc in bcs]
        rel_res = norm(dvp_res, 'l2')
        residual = b.norm('l2')

        if MPI.rank(mpi_comm_world()) == 0:
            print("Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) "
                  % (Iter, residual, atol, rel_res, rtol))
        Iter += 1

    return dict()
