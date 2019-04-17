"""
Solves XXX

"""

from dolfin import inner, grad, div


def extrapolate_setup(F_fluid_linear, extype, mesh_file, d_, w_, phi, gamma, beta, dx_f,
                      **namespace):
    alfa_u = 0.01
    F_ext1 = alfa_u*inner(w_["n"], beta)*dx_f - alfa_u*inner(grad(d_["n"]), grad(beta))*dx_f
    F_ext2 = alfa_u*inner(grad(w_["n"]), grad(phi))*dx_f
    F_fluid_linear += F_ext1 + F_ext2

    return dict(F_fluid_linear=F_fluid_linear)
