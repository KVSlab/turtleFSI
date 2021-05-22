# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from turtleFSI.modules import *
from dolfin import Constant, inner, grad


def solid_setup(d_, v_, phi, psi, dx_s, mu_s, rho_s, lambda_s, k, theta,
                gravity, **namespace):
    """
    ALE formulation (theta-scheme) of the non-linear elastic problem:

    dv/dt - f + div(sigma) = 0   with v = d(d)/dt
    """

    # From the equation defined above we have to include the equation v - d(d)/dt = 0. This
    # ensures both that the variable d and v is well defined in the solid equation, but also
    # that there is continuity of the velocity at the boundary. Since this is imposed weakly
    # we 'make this extra important' by multiplying with a large number delta.
    delta = 1E7

    # Theta scheme constants
    theta0 = Constant(theta)
    theta1 = Constant(1 - theta)

    # Temporal term and convection
    F_solid_linear = (rho_s/k * inner(v_["n"] - v_["n-1"], psi)*dx_s
                      + delta * rho_s * (1 / k) * inner(d_["n"] - d_["n-1"], phi) * dx_s
                      - delta * rho_s * inner(theta0 * v_["n"] + theta1 * v_["n-1"], phi) * dx_s)

    # Gravity
    if gravity is not None:
        F_solid_linear -= inner(Constant((0, -gravity * rho_s)), psi)*dx_s

    # Stress
    F_solid_nonlinear = theta0 * inner(Piola1(d_["n"], lambda_s, mu_s), grad(psi)) * dx_s
    F_solid_linear += theta1 * inner(Piola1(d_["n-1"], lambda_s, mu_s), grad(psi)) * dx_s

    return dict(F_solid_linear=F_solid_linear, F_solid_nonlinear=F_solid_nonlinear)
