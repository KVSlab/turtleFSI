# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Solves the equation

du/dt - w + PI(u) = 0

or something like that. Explain the numerics.
"""

from turtleFSI.modules.common import *
from dolfin import Constant, inner, grad


def structure_setup(d_, v_, p_, phi, psi, gamma, dS, mu_f, n, gravity, dx_s, dx_f, mu_s,
                    rho_s, lamda_s, k, mesh_file, theta, g, **namespace):
    delta = 1E10
    theta0 = Constant(Constant)
    theta1 = Constant(1 - theta)

    # Temporal term and convection
    F_solid_linear = (rho_s/k*inner(v_["n"] - v_["n-1"], psi)*dx_s
                     + delta*(1/k)*inner(d_["n"] - d_["n-1"], phi)*dx_s
                     - delta*inner(theta0*v_["n"] + theta1*v_["n-1"], phi)*dx_s)

    # Gravity
    if gravity:
        F_solid_linear -= inner(Constant(0, -g*rho_s), psi)*dx_s

    # Stress
    # FIXME: Correct to split when Piola1 is non linear?
    #F_solid_nonlinear = inner(Piola1(theta0*d_["n"] + theta1*d_["n-1"], lambda_s, mu_s),
    #                          grad(psi))*dx_s
    F_solid_nonlinear = inner(Piola1(theta0*d_["n"], lamda_s, mu_s), grad(psi))*dx_s
    F_solid_linear = inner(Piola1(theta1*d_["n-1"], lambda_s, mu_s), grad(psi))*dx_s

    return dict(F_solid_linear=F_solid_linear, F_solid_nonlinear=F_solid_nonlinear)
