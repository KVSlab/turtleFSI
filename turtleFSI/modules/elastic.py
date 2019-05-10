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

    References:

    Slyngstad, Andreas Strøm. Verification and Validation of a Monolithic
        Fluid-Structure Interaction Solver in FEniCS. A comparison of mesh lifting
        operators. MS thesis. 2017.

    Gjertsen, Sebastian. Development of a Verified and Validated Computational
        Framework for Fluid-Structure Interaction: Investigating Lifting Operators
        and Numerical Stability. MS thesis. 2017.
    """
    E_y = 1./CellVolume(mesh)
    nu = 0.25
    alfa_lam = nu*E_y / ((1. + nu)*(1. - 2.*nu))
    alfa_mu = E_y/(2.*(1. + nu))
    F_extrapolate = inner(J_(d_["n"]) * S_linear(d_["n"], alfa_mu, alfa_lam) *
                          inv(F_(d_["n"])).T, grad(phi))*dx_f

    F_fluid_linear += F_extrapolate

    return dict(F_fluid_linear=F_fluid_linear)
