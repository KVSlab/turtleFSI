from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI

def extrapolate_setup(F_fluid_linear, ds, n, d_, w_, phi, beta, dx_f, **monolithic):
    def F_(U):
        return Identity(len(U)) + grad(U)

    alfa_u = 0.1
    F_ext1 = alfa_u*inner(w_["n"], beta)*dx_f - alfa_u*inner(grad(d_["n"]), grad(beta))*dx_f\
    + alfa_u*inner(grad(d_["n"])*n, beta)*ds(2) \
    + alfa_u*inner(grad(d_["n"])*n, beta)*ds(3) \
    + alfa_u*inner(grad(d_["n"])*n, beta)*ds(4) \
    + alfa_u*inner(grad(d_["n"])*n, beta)*ds(6)

    F_ext2 = alfa_u*inner(grad(w_["n"]), grad(phi))*dx_f \
        - alfa_u*inner(grad(w_["n"])*n, phi)*ds(2) \
        - alfa_u*inner(grad(w_["n"])*n, phi)*ds(3) \
        - alfa_u*inner(grad(w_["n"])*n, phi)*ds(4) \
        - alfa_u*inner(grad(w_["n"])*n, phi)*ds(6)

    F_fluid_linear += F_ext1 + F_ext2

    return dict(F_fluid_linear=F_fluid_linear)
