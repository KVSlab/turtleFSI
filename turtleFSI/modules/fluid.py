# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from turtleFSI.modules import *
from dolfin import Constant, inner, inv, grad, div


def fluid_setup(v_, p_, d_, psi, gamma, dx_f, dx_f_id_list, fluid_properties, k, theta, **namespace):
    """
    ALE formulation (theta-scheme) of the incompressible Navier-Stokes flow problem:

        du/dt + u * grad(u - w) = grad(p) + nu * div(grad(u))
        div(u) = 0
    """

    theta0 = Constant(theta)
    theta1 = Constant(1 - theta)

    F_fluid_linear = 0
    F_fluid_nonlinear = 0

    for fluid_region in range(len(dx_f_id_list)):
        rho_f = fluid_properties[fluid_region]["rho_f"]
        mu_f = fluid_properties[fluid_region]["mu_f"]

        # Note that we here split the equation into a linear and nonlinear part for faster
        # computation of the Jacobian matrix.
    
        # Temporal derivative
        F_fluid_nonlinear += rho_f / k * inner(J_(d_["n"]) * theta0 * (v_["n"] - v_["n-1"]), psi) * dx_f[fluid_region]
        F_fluid_linear += rho_f / k * inner(J_(d_["n-1"]) * theta1 * (v_["n"] - v_["n-1"]), psi) * dx_f[fluid_region]
    
        # Convection
        F_fluid_nonlinear += theta0 * rho_f * inner(grad(v_["n"]) * inv(F_(d_["n"])) *
                                                    J_(d_["n"]) * v_["n"], psi) * dx_f[fluid_region]
        F_fluid_linear += theta1 * rho_f * inner(grad(v_["n-1"]) * inv(F_(d_["n-1"])) *
                                                 J_(d_["n-1"]) * v_["n-1"], psi) * dx_f[fluid_region]
    
        # Stress from pressure
        F_fluid_nonlinear += inner(J_(d_["n"]) * sigma_f_p(p_["n"], d_["n"]) *
                                   inv(F_(d_["n"])).T, grad(psi)) * dx_f[fluid_region]
    
        # Stress from velocity
        F_fluid_nonlinear += theta0 * inner(J_(d_["n"]) * sigma_f_u(v_["n"], d_["n"], mu_f) *
                                            inv(F_(d_["n"])).T, grad(psi)) * dx_f[fluid_region]
        F_fluid_linear += theta1 * inner(J_(d_["n-1"]) * sigma_f_u(v_["n-1"], d_["n-1"], mu_f)
                                         * inv(F_(d_["n-1"])).T, grad(psi)) * dx_f[fluid_region]
    
        # Divergence free term
        F_fluid_nonlinear += inner(div(J_(d_["n"]) * inv(F_(d_["n"])) * v_["n"]), gamma) * dx_f[fluid_region]
    
        # ALE term
        F_fluid_nonlinear -= rho_f / k * inner(J_(d_["n"]) * grad(v_["n"]) * inv(F_(d_["n"])) *
                                           (d_["n"] - d_["n-1"]), psi) * dx_f[fluid_region]

    return dict(F_fluid_linear=F_fluid_linear, F_fluid_nonlinear=F_fluid_nonlinear)
