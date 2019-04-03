from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI, CellVolume


def extrapolate_setup(F_fluid_nonlinear, mesh_file, d_, phi, gamma, dx_f, **semimp_namespace):
    alfa = 0.01*(mesh_file.hmin())**2
    F_extrapolate = inner(alfa*grad(d_["n"]), grad(phi))*dx_f
    F_fluid_nonlinear += F_extrapolate
    return dict(F_fluid_nonlinear=F_fluid_nonlinear)
