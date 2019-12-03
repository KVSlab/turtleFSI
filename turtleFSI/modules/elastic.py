# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from turtleFSI.modules import *
from dolfin import CellVolume, inner, grad, inv


def extrapolate_setup(F_fluid_linear, mesh, d_, phi, gamma, dx_f, **namespace):
    """
    Elastic lifting operator solving the equation of linear elasticity.

    div(sigma(d)) = 0   in the fluid domain
    d = 0               on the fluid boundaries other than FSI interface
    d = solid_d         on the FSI interface
    """
    E_y = 1.0 / CellVolume(mesh)
    nu = 0.25
    alpha_lam = nu * E_y / ((1.0 + nu) * (1.0 - 2.0 * nu))
    alpha_mu = E_y / (2.0 * (1.0 + nu))
    F_extrapolate = inner(J_(d_["n"]) * S_linear(d_["n"], alpha_mu, alpha_lam) *
                          inv(F_(d_["n"])).T, grad(phi))*dx_f

    F_fluid_linear += F_extrapolate

    return dict(F_fluid_linear=F_fluid_linear)
