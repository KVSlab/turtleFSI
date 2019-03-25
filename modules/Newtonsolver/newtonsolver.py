from dolfin import *
from numpy import isnan
#import numpy as np


def solver_setup(F_fluid_linear, F_fluid_nonlinear,
                 F_solid_linear, F_solid_nonlinear, DVP, dvp_, **monolithic):

    print("solver_setup")
    F_lin = F_fluid_linear + F_solid_linear
    F_nonlin = F_fluid_nonlinear + F_solid_nonlinear
    F = F_lin + F_nonlin

    chi = TrialFunction(DVP)
    J_linear = derivative(F_lin, dvp_["n"], chi)
    J_nonlinear = derivative(F_nonlin, dvp_["n"], chi)

    #A_pre = assemble(J_linear)
    A_pre = assemble(J_linear, form_compiler_parameters={"optimize": True, "quadrature_degree": 4})
    A = Matrix(A_pre)
    b = None

    return dict(F=F, J_nonlinear=J_nonlinear, A_pre=A_pre, A=A, b=b)


def newtonsolver(F, J_nonlinear, A_pre, A, b, bcs, DVP,
                 dvp_, up_sol, dvp_res, rtol, atol, max_it, T, t, **monolithic):
    Iter = 0
    residual = 1
    rel_res = residual
    lmbda = 1

    while rel_res > rtol and residual > atol and Iter < max_it:

        A = assemble(J_nonlinear, tensor=A, keep_diagonal=True, form_compiler_parameters={"optimize": True, "quadrature_degree": 4})
        A.axpy(1.0, A_pre, True)

        A.ident_zeros()
        b = assemble(-F, tensor=b, form_compiler_parameters={"quadrature_degree": 4})
        [bc.apply(A, b, dvp_["n"].vector()) for bc in bcs]

        up_sol.solve(A, dvp_res.vector(), b)

        dvp_["n"].vector().axpy(lmbda, dvp_res.vector())
        [bc.apply(dvp_["n"].vector()) for bc in bcs]

        rel_res = norm(dvp_res, 'l2')
        residual = b.norm('l2')
        if isnan(rel_res) or isnan(residual):
            print("type rel_res: ", type(rel_res))
            t = T*T

        if MPI.rank(mpi_comm_world()) == 0:
            print("Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) "
                  % (Iter, residual, atol, rel_res, rtol))
        Iter += 1

    return dict(t=t)
