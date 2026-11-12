# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from turtleFSI.modules import *
from dolfin import Constant, inner, inv, grad, div

# sets up the fluid equations on the fluid subdomain:
# ALE Navier-Stokes plus incompressibility. It adds terms for fluid inertia,
# convection, pressure stress, viscous stress, divergence-free constraint,
# and the ALE correction from moving mesh displacement.

def fluid_setup(v_, p_, d_, psi, gamma, dx_f, dx_f_id_list, fluid_properties, k, theta,
                shape_displacement=None, fluid_body_force=None, **namespace):
    """
    ALE formulation (theta-scheme) of the incompressible Navier-Stokes flow problem:

        du/dt + u * grad(u - w) = grad(p) + nu * div(grad(u))
        div(u) = 0
    """

    theta0 = Constant(theta)
    theta1 = Constant(1 - theta)

    F_fluid_linear = 0
    F_fluid_nonlinear = 0
    fluid_paper_terms = {
        "m_f": 0,
        "a_f": 0,
        "b_f_momentum": 0,
        "c_f": 0,
        "c_f_ALE": 0,
        "f_f": 0,
        "b_f_continuity": 0,
    }

    for fluid_region in range(len(dx_f_id_list)):
        rho_f = fluid_properties[fluid_region]["rho_f"]
        mu_f = fluid_properties[fluid_region]["mu_f"]

        # Note that we here split the equation into a linear and nonlinear part for faster
        # computation of the Jacobian matrix.
    
        G_shape = G_(d_["n"], shape_displacement)
        K_shape = K_(d_["n"], shape_displacement)
        G_shape_inv = inv(G_shape)

        # section 2.3 A
        # fluid time derivative: m_f
        m_f_nonlinear = rho_f / k * inner(K_shape * J_(d_["n"], shape_displacement) *
                                          theta0 * (v_["n"] - v_["n-1"]), psi) * dx_f[fluid_region]
        m_f_linear = rho_f / k * inner(K_shape * J_(d_["n-1"], shape_displacement) *
                                       theta1 * (v_["n"] - v_["n-1"]), psi) * dx_f[fluid_region]

        F_fluid_nonlinear += m_f_nonlinear
        F_fluid_linear += m_f_linear
        fluid_paper_terms["m_f"] += m_f_nonlinear + m_f_linear

        # fluid stiffness: a_f
        a_f_nonlinear = theta0 * inner(K_shape * J_(d_["n"], shape_displacement) *
                                       sigma_f_u(v_["n"], d_["n"], mu_f, shape_displacement) *
                                            inv(F_(d_["n"], shape_displacement)).T *
                                       G_shape_inv.T, grad(psi)) * dx_f[fluid_region]
        a_f_linear = theta1 * inner(K_shape * J_(d_["n-1"], shape_displacement) *
                                    sigma_f_u(v_["n-1"], d_["n-1"], mu_f, shape_displacement)
                                         * inv(F_(d_["n-1"], shape_displacement)).T *
                                    G_shape_inv.T, grad(psi)) * dx_f[fluid_region]

        F_fluid_nonlinear += a_f_nonlinear
        F_fluid_linear += a_f_linear
        fluid_paper_terms["a_f"] += a_f_nonlinear + a_f_linear

        # fluid incompressibility: b_f: pressure
        b_f_nonlinear = inner(K_shape * J_(d_["n"], shape_displacement) *
                              sigma_f_p(p_["n"], d_["n"]) * inv(F_(d_["n"], shape_displacement)).T *
                              G_shape_inv.T, grad(psi)) * dx_f[fluid_region]
        F_fluid_nonlinear += b_f_nonlinear
        fluid_paper_terms["b_f_momentum"] += b_f_nonlinear

        # fluid advection: c_f
        c_f_nonlinear =  theta0 * rho_f * inner(
            K_shape * J_(d_["n"], shape_displacement) * grad(v_["n"]) * G_shape_inv *
            inv(F_(d_["n"], shape_displacement)) * v_["n"], psi) * dx_f[fluid_region]
        c_f_linear = theta1 * rho_f * inner(
            K_shape * J_(d_["n-1"], shape_displacement) * grad(v_["n-1"]) * G_shape_inv *
            inv(F_(d_["n-1"], shape_displacement)) * v_["n-1"], psi) * dx_f[fluid_region]

        F_fluid_nonlinear += c_f_nonlinear
        F_fluid_linear += c_f_linear
        fluid_paper_terms["c_f"] += c_f_nonlinear + c_f_linear

        # fluid ALE: c_f_ALE
        c_f_ALE = rho_f / k * inner(K_shape * J_(d_["n"], shape_displacement) * grad(v_["n"])
                                    * G_shape_inv * inv(F_(d_["n"], shape_displacement)) *
                                    (d_["n"] - d_["n-1"]), psi) * dx_f[fluid_region]

        F_fluid_nonlinear -= c_f_ALE
        fluid_paper_terms["c_f_ALE"] -= c_f_ALE


        # fluid forcing term: f_f
        body_force_f = vector_body_force(region_value(fluid_body_force, fluid_region), get_dimension(v_["n"]))
        if body_force_f is not None:
            f_f = inner(K_shape * J_(d_["n"], shape_displacement) *
                        rho_f * body_force_f, psi) * dx_f[fluid_region]
            F_fluid_linear -= f_f
            fluid_paper_terms["f_f"] -= f_f

        # section 2.3 B
        # fluid incompressibility: b_f: divergence free term
        b_f_continuity = inner(div(K_shape * J_(d_["n"], shape_displacement) *
                                   inv(F_(d_["n"], shape_displacement)) *
                                   G_shape_inv * v_["n"]), gamma) * dx_f[fluid_region]
        F_fluid_nonlinear += b_f_continuity
        fluid_paper_terms["b_f_continuity"] += b_f_continuity

    return dict(F_fluid_linear=F_fluid_linear, F_fluid_nonlinear=F_fluid_nonlinear,
                fluid_paper_terms=fluid_paper_terms)
