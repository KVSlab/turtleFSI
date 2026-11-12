# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from collections.abc import Iterable

from turtleFSI.modules import *
from turtleFSI.problems import info_blue
from dolfin import Constant, inner, grad, MPI

# ets up the solid mechanics equations on the solid subdomain.
# It adds solid inertia, internal elastic stress through Piola1(...),
# optional viscoelastic stress, optional gravity, optional Robin boundary support,
# and the kinematic coupling relation tying solid displacement updates to solid velocity.

def solid_setup(d_, v_, phi, psi, dx_s, ds_s, dx_s_id_list, ds_s_ext_id_list, solid_properties, k, theta,
                gravity, mesh, robin_bc, k_s, c_s, shape_displacement=None, solid_body_force=None,
                extrapolation_sub_type="constant", **namespace):

    # DB added gravity in 3d functionality and multi material capability 16/3/21
    #
    #

    """
    ALE formulation (theta-scheme) of the non-linear elastic problem:
    dv/dt - f + div(sigma) = 0   with v = d(d)/dt
    """

    # From the equation defined above we have to include the equation v - d(d)/dt = 0. This
    # ensures both that the variable d and v is well defined in the solid equation, but also
    # that there is continuity of the velocity at the boundary. Since this is imposed weakly
    # we 'make this extra important' by multiplying with a large number delta.

    delta = 1.0E7

    # Theta scheme constants
    theta0 = Constant(theta)
    theta1 = Constant(1 - theta)

    F_solid_linear = 0
    F_solid_nonlinear = 0
    solid_paper_terms = {
        "m_s": 0,
        "m_e_s": 0,
        "f_e_s": 0,
        "extension_relation_s": 0,
        "a_s": 0,
        "f_s": 0,
    }

    for solid_region in range(len(dx_s_id_list)):
        rho_s = solid_properties[solid_region]["rho_s"]
        G_shape = G_(d_["n"], shape_displacement)
        K_shape = K_(d_["n"], shape_displacement)
        G_shape_inv = inv(G_shape)

        # structure time derivative: m_s
        m_s_linear = K_shape * rho_s/k * inner(v_["n"] - v_["n-1"], psi)*dx_s[solid_region]
        F_solid_linear += m_s_linear
        solid_paper_terms["m_s"] += m_s_linear

        # extension structure relations: m_e_s, f_e_s
        if extrapolation_sub_type == "paper":

            # extension structure time derivative: m_e_s
            m_e_s = K_shape * (1 / k) * inner(d_["n"] - d_["n-1"], phi) * dx_s[solid_region]

            # extension structure forcing term: f_e_s
            f_e_s = K_shape * inner(theta0 * v_["n"] + theta1 * v_["n-1"], phi) * dx_s[solid_region]

            extension_relation_s = m_e_s - f_e_s
            F_solid_linear += extension_relation_s
            solid_paper_terms["m_e_s"] += m_e_s
            solid_paper_terms["f_e_s"] += f_e_s
            solid_paper_terms["extension_relation_s"] += extension_relation_s

        else:
            # default turtle temporal term and convection, imposed weakly, so we add delta constant large

            # computes derivative of displacement
            F_solid_linear += (K_shape * delta * rho_s * (1 / k) *
                               inner(d_["n"] - d_["n-1"], phi) * dx_s[solid_region]
            # subtracts velocity to enforce continuity
                              - K_shape * delta * rho_s * inner(theta0 * v_["n"] + theta1 * v_["n-1"], phi) * dx_s[solid_region]) # Maybe we could add viscoelasticity with v["n"] term instead of (1 / k) * inner(d_["n"] - d_["n-1"], phi)

        # does not appear in paper / this is optional turtleFSI extension for models

        # Viscoelasticity (rate dependant portion of the stress)
        if "viscoelasticity" in solid_properties[solid_region]:
            if solid_properties[solid_region]["viscoelasticity"] == "Form1": # This version (using velocity directly) seems to work best. 
                F_solid_nonlinear += theta0 * inner(K_shape * F_(d_["n"], shape_displacement)*Svisc_D(v_["n"], solid_properties[solid_region]) * G_shape_inv.T, grad(psi)) * dx_s[solid_region]
                F_solid_linear += theta1 * inner(K_shape * F_(d_["n-1"], shape_displacement)*Svisc_D(v_["n-1"], solid_properties[solid_region]) * G_shape_inv.T, grad(psi)) * dx_s[solid_region]
            elif solid_properties[solid_region]["viscoelasticity"] == "Form2": # This version (using displacements to calculate linearized derivative) doesnt work as well.
                # (1/k) can come outside because all operators are linear. (v_[n]+v_[n-1])/2 = (d_[n]-d_[n-1])/k    where k is the timestep.
                F_solid_nonlinear += (1/k) * inner(K_shape * F_(d_["n"], shape_displacement)*theta0*Svisc_D(d_["n"] - d_["n-1"], solid_properties[solid_region]) * G_shape_inv.T, grad(psi)) * dx_s[solid_region]
                F_solid_linear += (1/k) * inner(K_shape * F_(d_["n-1"], shape_displacement)*theta1*Svisc_D(d_["n"] - d_["n-1"], solid_properties[solid_region]) * G_shape_inv.T, grad(psi)) * dx_s[solid_region]
            else:
                if MPI.rank(MPI.comm_world) == 0:
                    print("Invalid/No entry for viscoelasticity, assuming no viscoelasticity.")
        else:
            if MPI.rank(MPI.comm_world) == 0:
                print("No entry for viscoelasticity, assuming no viscoelasticity.")

        # Stress (Note that if viscoelasticity is used, Piola1() is no longer the total stress,
        # it is the non-rate dependant (elastic) component of the stress)

        # structure stiffness: a_s
        a_s_linear = theta0 * inner(K_shape * Piola1(d_["n"], solid_properties[solid_region], shape_displacement)
                                            * G_shape_inv.T, grad(psi)) * dx_s[solid_region]
        a_s_nonlinear = theta1 * inner(K_shape * Piola1(d_["n-1"], solid_properties[solid_region], shape_displacement)
                                         * G_shape_inv.T, grad(psi)) * dx_s[solid_region]
        F_solid_nonlinear += a_s_nonlinear
        F_solid_linear += a_s_linear
        solid_paper_terms["a_s"] += a_s_nonlinear + a_s_linear

        # structure forcing term: f_s
        body_force_s = vector_body_force(region_value(solid_body_force, solid_region), mesh.geometry().dim())
        if body_force_s is not None:
            f_s_linear = inner(K_shape * rho_s * body_force_s, psi) * dx_s[solid_region]
            F_solid_linear -= f_s_linear
            solid_paper_terms["f_s"] -= f_s_linear

        # does not appear in paper / this is turtleFSI additional extension
        # Backward-compatible gravity shortcut, y direction only
        if gravity is not None and mesh.geometry().dim() == 2:
            F_solid_linear -= inner(K_shape * Constant((0, -gravity * rho_s)), psi)*dx_s[solid_region] 
        elif gravity is not None and mesh.geometry().dim() == 3:
            F_solid_linear -= inner(K_shape * Constant((0, -gravity * rho_s,0)), psi)*dx_s[solid_region] 
            
    # does not appear explicitly in paper / this is turtleFSI extension
    # Robin BC
    """
    The derivation comes from the eq.(9) in the followling paper:
    Moireau, P., Xiao, N., Astorino, M. et al. External tissue support and fluid–structure simulation in blood flows. 
    Biomech Model Mechanobiol 11, 1–18 (2012). https://doi.org/10.1007/s10237-011-0289-z
    """

    if robin_bc:
        info_blue("Robin BC is used for the solid domain.")
        assert isinstance(k_s, Iterable), "k_s should be an iterable (e.g., list, tuple, etc.)."
        assert isinstance(c_s, Iterable), "c_s should be an iterable (e.g., list, tuple, etc.)."
        assert len(k_s) == len(c_s) == len(ds_s_ext_id_list), "k_s, c_s, and ds_s_ext_id_list should have the same length."
        for solid_boundaries in range(len(ds_s_ext_id_list)):
            if MPI.rank(MPI.comm_world) == 0:
                print(f"solid_boundaries: {solid_boundaries}, ds_s_ext_id_list: {ds_s_ext_id_list[solid_boundaries]}")
                print(f"k_s: {k_s[solid_boundaries]}, c_s: {c_s[solid_boundaries]}")
            F_solid_linear += theta0 * inner((k_s[solid_boundaries] * d_["n"] + c_s[solid_boundaries] * v_["n"]), psi)*ds_s[solid_boundaries]
            F_solid_linear += theta1 * inner((k_s[solid_boundaries] * d_["n-1"] + c_s[solid_boundaries] * v_["n-1"]), psi)*ds_s[solid_boundaries]
            

    return dict(F_solid_linear=F_solid_linear, F_solid_nonlinear=F_solid_nonlinear,
                solid_paper_terms=solid_paper_terms)
