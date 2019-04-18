# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Solves XXX
TODO
"""

from dolfin import inner, grad, div


def extrapolate_setup(F_fluid_linear, extrapolation_sub_type, d_, w_, phi, beta, dx_f,
                      ds, n, bc_ids, **namespace):
    alfa_u = 0.01
    F_ext1 = alfa_u*inner(w_["n"], beta)*dx_f - alfa_u*inner(grad(d_["n"]),
                                                             grad(beta))*dx_f
    F_ext2 = alfa_u*inner(grad(w_["n"]), grad(phi))*dx_f

    if extrapolation_sub_type == "bc2":
        for i in bc_ids:
            F_ext1 += alfa_u*inner(grad(d_["n"])*n, beta)*ds(i)
            F_ext2 -= alfa_u*inner(grad(w_["n"])*n, phi)*ds(i)

    F_fluid_linear += F_ext1 + F_ext2

    return dict(F_fluid_linear=F_fluid_linear)
