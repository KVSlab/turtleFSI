# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import inner, grad


def extrapolate_setup(F_fluid_linear, extrapolation_sub_type, d_, w_, phi, beta, dx_f,
                      dx_f_id, ds, n, bc_ids, **namespace):
    """
    Biharmonic lifting operator. Should be used for large deformations.

    alpha * laplace^2(d) = 0   in the fluid domain

    By introducing w = - grad(d) we obtain the equivalent system of equations:
        w = - alpha * laplace(d)
        - alpha * grad(w) = 0

    Two types of boundary conditions can be setup for this problem:

        - "constrained_disp" with conditions only on (d):
            d(d)/dn = 0    on the fluid boundaries other than FSI interface
            d = solid_def  on the FSI interface

        - "constrained_disp_vel" with conditions on (d) and (w):
            d(d(x))/dn = 0 and d(w(x))/dn = 0   on the inlet and outlet fluid boundaries
            d(d(y))/dn = 0 and d(w(y))/dn = 0   on the FSI interface
    """

    alpha_u = 0.01
    F_ext1 = 0
    F_ext2 = 0
    for fluid_region in range(len(dx_f_id)): # for all fluid regions
        F_ext1 += alpha_u * inner(w_["n"], beta) * dx_f[fluid_region] - alpha_u * inner(grad(d_["n"]), grad(beta)) * dx_f[fluid_region]
        F_ext2 += alpha_u * inner(grad(w_["n"]), grad(phi)) * dx_f[fluid_region]

    if extrapolation_sub_type == "constrained_disp_vel":
        for i in bc_ids:
            F_ext1 += alpha_u*inner(grad(d_["n"]) * n, beta) * ds(i)
            F_ext2 -= alpha_u*inner(grad(w_["n"]) * n, phi) * ds(i)

    F_fluid_linear += F_ext1 + F_ext2

    return dict(F_fluid_linear=F_fluid_linear)
