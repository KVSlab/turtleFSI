# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

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

    References:

    Slyngstad, Andreas Strøm. Verification and Validation of a Monolithic
        Fluid-Structure Interaction Solver in FEniCS. A comparison of mesh lifting
        operators. MS thesis. 2017.

    Gjertsen, Sebastian. Development of a Verified and Validated Computational
        Framework for Fluid-Structure Interaction: Investigating Lifting Operators
        and Numerical Stability. MS thesis. 2017.
    """

    delta = 1E10
    theta0 = Constant(theta)
    theta1 = Constant(1 - theta)

    # Temporal term and convection
    F_solid_linear = (rho_s/k*inner(v_["n"] - v_["n-1"], psi)*dx_s
                      + delta*(1/k)*inner(d_["n"] - d_["n-1"], phi)*dx_s
                      - delta*inner(theta0*v_["n"] + theta1*v_["n-1"], phi)*dx_s)

    # Gravity
    if gravity is not None:
        F_solid_linear -= inner(Constant((0, -gravity*rho_s)), psi)*dx_s

    # Stress
    F_solid_nonlinear = inner(Piola1(theta0*d_["n"], lambda_s, mu_s), grad(psi))*dx_s
    F_solid_linear += inner(Piola1(theta1*d_["n-1"], lambda_s, mu_s), grad(psi))*dx_s

    return dict(F_solid_linear=F_solid_linear, F_solid_nonlinear=F_solid_nonlinear)
