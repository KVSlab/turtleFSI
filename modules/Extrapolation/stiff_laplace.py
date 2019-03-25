from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI
#from semi_implicit import *


def extrapolate_setup(d_, mesh_file, phi, dx_f, F_fluid_linear, F_solid_linear, **semimp_namespace):
    alfa = 1e-2
    h = mesh_file.hmin()
    F_extrapolate =  alfa*h*h*inner(grad(d_["n"]), grad(phi))*dx_f
    F_solid_linear += F_extrapolate

    return dict(F_fluid_linear=F_fluid_linear, F_solid_linear=F_solid_linear)
