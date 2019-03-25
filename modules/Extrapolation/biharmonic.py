from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI
#from semi_implicit import *


def extrapolate_setup(F_fluid_linear, extype, mesh_file, d_, w_, phi, gamma, beta, dx_f, **semimp_namespace):
    def F_(U):
        return Identity(len(U)) + grad(U)

    def J_(U):
        return det(F_(U))

    alfa_u = 0.01
    F_ext1 = alfa_u*inner(w_["n"], beta)*dx_f - alfa_u*inner(grad(d_["n"]), grad(beta))*dx_f
    F_ext2 = alfa_u*inner(grad(w_["n"]), grad(phi))*dx_f
    F_fluid_linear += F_ext1 + F_ext2

    return dict(F_fluid_linear=F_fluid_linear)
