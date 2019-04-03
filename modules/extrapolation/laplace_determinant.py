from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI, CellVolume


def extrapolate_setup(F_fluid_nonlinear, d_, phi, dx_f, **monolithic):
    def F_(U):
        return Identity(len(U)) + grad(U)

    def J_(U):
        return det(F_(U))

    alfa = 1./(J_(d_["n"]))

    F_extrapolate = inner(alfa*grad(d_["n"]), grad(phi))*dx_f

    F_fluid_nonlinear += F_extrapolate

    return dict(F_fluid_nonlinear=F_fluid_nonlinear)
