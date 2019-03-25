from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI
#from semi_implicit import *

def F_(d):
    return Identity(len(d))+grad(d)
def J_(d):
    return det(F_(d))

def extrapolate_setup(d_, phi, dx_f,  F_fluid_nonlinear, **semimp_namespace):

    F_extrapolate =  inner((1./J_(d_["n"])) * grad(d_["n"]), grad(phi))*dx_f
    F_fluid_nonlinear += F_extrapolate

    return dict(F_fluid_nonlinear = F_fluid_nonlinear)
