from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI, CellVolume


def extrapolate_setup(F_fluid_nonlinear, extype, mesh_file, d_, phi, gamma, dx_f, **semimp_namespace):
    alfa = 1.0
    F_extrapolate = alfa*inner(grad(d_["n"]), grad(phi))*dx_f
    # Try nonlinear
    #F_fluid_linear += F_extrapolate
    F_fluid_nonlinear += F_extrapolate

    return dict(F_fluid_nonlinear=F_fluid_nonlinear)