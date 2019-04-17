from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI
#from semi_implicit import *


def extrapolate_setup(d_, phi, dx_f, F_fluid_linear, **semimp_namespace):
    alfa = 1
    F_extrapolate =  alfa*inner(grad(d_["n"]), grad(phi))*dx_f
    F_fluid_linear += F_extrapolate

    return dict(F_fluid_linear=F_fluid_linear)
