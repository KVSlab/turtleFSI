# File under GNU GPL (v3) licence, see LICENSE file for details.

"""Helpers for paper-style Newton block forms.

The production solver can keep using the aggregate UFL derivative. These helpers
expose the same linearization term-by-term so equation block (3) from
Ballarin/Rozza can be inspected, assembled, or projected later.
"""

from dolfin import Constant, TrialFunction, derivative, grad, inner, inv, split, tr

from turtleFSI.modules.common import (
    F_,
    G_,
    J_,
    K_,
    Piola1,
    get_dimension,
    region_value,
    sigma_f_p,
    sigma_f_u,
    vector_body_force,
)


def _is_zero_form(form):
    return isinstance(form, (int, float)) and form == 0


def _add_form(total, form):
    if _is_zero_form(form):
        return total
    if _is_zero_form(total):
        return form
    return total + form


def _differentiate_terms(terms, dvp, chi):
    differentiated = {}
    total = 0
    for name, form in terms.items():
        if _is_zero_form(form):
            differentiated[name] = 0
            continue
        block = derivative(form, dvp, chi)
        differentiated[name] = block
        total = _add_form(total, block)
    return differentiated, total


def _build_fluid_component_blocks(chi_parts, namespace):
    if not all(key in namespace for key in ("d_", "v_", "p_", "psi", "gamma", "dx_f", "dx_f_id_list", "fluid_properties", "k", "theta")):
        return {}

    delta_d, delta_u, delta_p = chi_parts[:3]
    d_ = namespace["d_"]
    v_ = namespace["v_"]
    p_ = namespace["p_"]
    psi = namespace["psi"]
    gamma = namespace["gamma"]
    dx_f = namespace["dx_f"]
    dx_f_id_list = namespace["dx_f_id_list"]
    fluid_properties = namespace["fluid_properties"]
    k = namespace["k"]
    theta0 = Constant(namespace["theta"])
    theta1 = Constant(1 - namespace["theta"])
    shape_displacement = namespace.get("shape_displacement")

    blocks = {
        "fluid_momentum": {},
        "fluid_continuity": {},
    }

    for fluid_region in range(len(dx_f_id_list)):
        rho_f = fluid_properties[fluid_region]["rho_f"]
        mu_f = fluid_properties[fluid_region]["mu_f"]
        G_shape = G_(d_["n"], shape_displacement)
        G_shape_inv = inv(G_shape)
        K_shape = K_(d_["n"], shape_displacement)
        F_inv = inv(F_(d_["n"], shape_displacement))
        A = G_shape_inv * F_inv
        B = F_inv.T * G_shape_inv.T
        J_current = J_(d_["n"], shape_displacement)
        J_previous = J_(d_["n-1"], shape_displacement)
        dJ_trace = tr(A * grad(delta_d))
        current_velocity_difference = v_["n"] - v_["n-1"]
        mesh_velocity_difference = d_["n"] - d_["n-1"]
        transformed_velocity_gradient = grad(v_["n"]) * A
        transformed_velocity_strain = transformed_velocity_gradient + B * grad(v_["n"]).T

        delta_d_m_f = rho_f / k * inner(
            K_shape * theta0 * J_current * dJ_trace * current_velocity_difference,
            psi,
        ) * dx_f[fluid_region]
        delta_u_m_f = rho_f / k * inner(
            K_shape * (theta0 * J_current + theta1 * J_previous)
            * delta_u, psi, ) * dx_f[fluid_region]

        blocks["fluid_momentum"]["delta_d_m_f"] = _add_form(
            blocks["fluid_momentum"].get("delta_d_m_f", 0), delta_d_m_f
        )
        blocks["fluid_momentum"]["delta_u_m_f"] = _add_form(
            blocks["fluid_momentum"].get("delta_u_m_f", 0), delta_u_m_f
        )

        delta_d_a_f = theta0 * inner(
            K_shape * J_current * mu_f * (
                dJ_trace * transformed_velocity_strain * B
                - grad(v_["n"]) * A * grad(delta_d) * A * B
                - B * grad(delta_d).T * B * grad(v_["n"]).T * B
                - grad(v_["n"]) * A * B * grad(delta_d).T * B
                - B * grad(v_["n"]).T * B * grad(delta_d).T * B
            ),
            grad(psi),
        ) * dx_f[fluid_region]
        delta_u_a_f = theta0 * inner(
            K_shape
            * J_current
            * sigma_f_u(delta_u, d_["n"], mu_f, shape_displacement)
            * B,
            grad(psi),
        ) * dx_f[fluid_region]
        blocks["fluid_momentum"]["delta_d_a_f"] = _add_form(
            blocks["fluid_momentum"].get("delta_d_a_f", 0), delta_d_a_f
        )
        blocks["fluid_momentum"]["delta_u_a_f"] = _add_form(
            blocks["fluid_momentum"].get("delta_u_a_f", 0), delta_u_a_f
        )

        delta_d_b_f_momentum = inner(
            K_shape
            * J_current
            * (
                -p_["n"] * dJ_trace * B
                + p_["n"] * B * grad(delta_d).T * B
            ),
            grad(psi),
        ) * dx_f[fluid_region]
        delta_p_b_f = inner(
            K_shape
            * J_current
            * sigma_f_p(delta_p, d_["n"])
            * B,
            grad(psi),
        ) * dx_f[fluid_region]
        blocks["fluid_momentum"]["delta_d_b_f"] = _add_form(
            blocks["fluid_momentum"].get("delta_d_b_f", 0), delta_d_b_f_momentum
        )
        blocks["fluid_momentum"]["delta_p_b_f"] = _add_form(
            blocks["fluid_momentum"].get("delta_p_b_f", 0), delta_p_b_f
        )

        delta_d_c_f = theta0 * rho_f * inner(
            K_shape
            * J_current
            * (
                dJ_trace * transformed_velocity_gradient * v_["n"]
                - grad(v_["n"]) * A * grad(delta_d) * A * v_["n"]
            ),
            psi,
        ) * dx_f[fluid_region]
        delta_u_c_f = theta0 * rho_f * inner(
            K_shape
            * J_current
            * (
                grad(delta_u) * A * v_["n"]
                + transformed_velocity_gradient * delta_u
            ),
            psi,
        ) * dx_f[fluid_region]
        blocks["fluid_momentum"]["delta_d_c_f"] = _add_form(
            blocks["fluid_momentum"].get("delta_d_c_f", 0), delta_d_c_f
        )
        blocks["fluid_momentum"]["delta_u_c_f"] = _add_form(
            blocks["fluid_momentum"].get("delta_u_c_f", 0), delta_u_c_f
        )

        delta_d_c_f_ALE = -(rho_f / k) * inner(
            K_shape
            * J_current
            * (
                dJ_trace * transformed_velocity_gradient * mesh_velocity_difference
                - grad(v_["n"]) * A * grad(delta_d) * A * mesh_velocity_difference
                + transformed_velocity_gradient * delta_d
            ),
            psi,
        ) * dx_f[fluid_region]
        delta_u_c_f_ALE = -(rho_f / k) * inner(
            K_shape
            * J_current
            * grad(delta_u)
            * A
            * mesh_velocity_difference,
            psi,
        ) * dx_f[fluid_region]
        blocks["fluid_momentum"]["delta_d_c_f_ALE"] = _add_form(
            blocks["fluid_momentum"].get("delta_d_c_f_ALE", 0), delta_d_c_f_ALE
        )
        blocks["fluid_momentum"]["delta_u_c_f_ALE"] = _add_form(
            blocks["fluid_momentum"].get("delta_u_c_f_ALE", 0), delta_u_c_f_ALE
        )

        body_force_f = namespace.get("fluid_body_force")
        force = vector_body_force(region_value(body_force_f, fluid_region), get_dimension(v_["n"]))
        if force is not None:
            delta_d_f_f = -inner(
                K_shape * J_current * rho_f * dJ_trace * force,
                psi,
            ) * dx_f[fluid_region]
            blocks["fluid_momentum"]["delta_d_f_f"] = _add_form(
                blocks["fluid_momentum"].get("delta_d_f_f", 0), delta_d_f_f
            )

        delta_d_b_f_continuity = (
            K_shape
            * J_current
            * (
                dJ_trace * tr(A * grad(v_["n"]))
                - tr(A * grad(delta_d) * A * grad(v_["n"]))
            )
            * gamma
            * dx_f[fluid_region]
        )
        delta_u_b_f_continuity = inner(
            K_shape * J_current * tr(A * grad(delta_u)),
            gamma,
        ) * dx_f[fluid_region]
        blocks["fluid_continuity"]["delta_d_b_f"] = _add_form(
            blocks["fluid_continuity"].get("delta_d_b_f", 0), delta_d_b_f_continuity
        )
        blocks["fluid_continuity"]["delta_u_b_f"] = _add_form(
            blocks["fluid_continuity"].get("delta_u_b_f", 0), delta_u_b_f_continuity
        )

    return blocks


def _build_extension_component_blocks(chi_parts, namespace):
    if not all(key in namespace for key in ("d_", "phi", "dx_f", "dx_f_id_list", "k")):
        return {}

    delta_d = chi_parts[0]
    d_ = namespace["d_"]
    phi = namespace["phi"]
    dx_f = namespace["dx_f"]
    dx_f_id_list = namespace["dx_f_id_list"]
    k = namespace["k"]
    shape_displacement = namespace.get("shape_displacement")

    blocks = {
        "extension_fluid": {},
    }
    if namespace.get("extrapolation_sub_type") != "paper":
        return blocks

    for fluid_region in range(len(dx_f_id_list)):
        G_shape = G_(d_["n"], shape_displacement)
        G_shape_inv = inv(G_shape)
        F_inv = inv(F_(d_["n"], shape_displacement))
        A = G_shape_inv * F_inv
        J_current = J_(d_["n"], shape_displacement)
        K_shape = K_(d_["n"], shape_displacement)
        dJ_trace = tr(A * grad(delta_d))
        GG_T = G_shape_inv * G_shape_inv.T

        delta_d_m_e_f = K_shape * (1 / k) * inner(delta_d, phi) * dx_f[fluid_region]
        delta_d_a_e_f = inner(
            (1 / J_current) * (-dJ_trace * grad(d_["n"]) * GG_T + grad(delta_d) * GG_T),
            grad(phi),
        ) * dx_f[fluid_region]
        blocks["extension_fluid"]["delta_d_m_e_f"] = _add_form(
            blocks["extension_fluid"].get("delta_d_m_e_f", 0), delta_d_m_e_f
        )
        blocks["extension_fluid"]["delta_d_a_e_f"] = _add_form(
            blocks["extension_fluid"].get("delta_d_a_e_f", 0), delta_d_a_e_f
        )

    if all(key in namespace for key in ("dS", "n", "fsi_id")):
        dS = namespace.get("dS")
        n = namespace.get("n")
        fsi_id = namespace.get("fsi_id")
        if dS is not None and n is not None and fsi_id is not None:
            interface_ids = fsi_id if isinstance(fsi_id, list) else [fsi_id]
            d_plus = d_["n"]("+")
            delta_d_plus = delta_d("+")
            phi_plus = phi("+")
            shape_plus = None if shape_displacement is None else shape_displacement("+")
            G_plus_inv = inv(G_(d_plus, shape_plus))
            F_plus_inv = inv(F_(d_plus, shape_plus))
            A_plus = G_plus_inv * F_plus_inv
            J_plus = J_(d_plus, shape_plus)
            dJ_trace_plus = tr(A_plus * grad(delta_d_plus))
            GG_T_plus = G_plus_inv * G_plus_inv.T
            positive_flux_derivative = (
                (1 / J_plus)
                * (-dJ_trace_plus * grad(d_plus) * GG_T_plus + grad(delta_d_plus) * GG_T_plus)
                * n("+")
            )
            for marker in interface_ids:
                # The residual contains ``-a_e_df``; store the derivative with that sign.
                delta_d_a_e_df = -inner(positive_flux_derivative, phi_plus) * dS(marker)
                blocks["extension_fluid"]["delta_d_a_e_df"] = _add_form(
                    blocks["extension_fluid"].get("delta_d_a_e_df", 0), delta_d_a_e_df
                )

    return blocks


def _build_solid_component_blocks(chi_parts, namespace):
    if not all(key in namespace for key in ("d_", "v_", "phi", "psi", "dx_s", "dx_s_id_list", "solid_properties", "k", "theta")):
        return {}

    delta_d, delta_u = chi_parts[:2]
    d_ = namespace["d_"]
    psi = namespace["psi"]
    phi = namespace["phi"]
    dx_s = namespace["dx_s"]
    dx_s_id_list = namespace["dx_s_id_list"]
    solid_properties = namespace["solid_properties"]
    k = namespace["k"]
    theta0 = Constant(namespace["theta"])
    theta1 = Constant(1 - namespace["theta"])
    del theta1
    shape_displacement = namespace.get("shape_displacement")

    blocks = {
        "solid_momentum": {},
        "extension": {},
    }

    for solid_region in range(len(dx_s_id_list)):
        rho_s = solid_properties[solid_region]["rho_s"]
        G_shape = G_(d_["n"], shape_displacement)
        K_shape = K_(d_["n"], shape_displacement)
        G_shape_inv = inv(G_shape)

        delta_u_m_s = K_shape * rho_s / k * inner(delta_u, psi) * dx_s[solid_region]
        blocks["solid_momentum"]["delta_u_m_s"] = _add_form(
            blocks["solid_momentum"].get("delta_u_m_s", 0), delta_u_m_s
        )

        material_model = solid_properties[solid_region].get("material_model")
        if material_model in ("LinearElastic", "BallarinRozzaLinearElastic"):
            delta_d_a_s = theta0 * inner(
                K_shape
                * Piola1(delta_d, solid_properties[solid_region], shape_displacement)
                * G_shape_inv.T,
                grad(psi),
            ) * dx_s[solid_region]
            blocks["solid_momentum"]["delta_d_a_s_linear_elastic"] = _add_form(
                blocks["solid_momentum"].get("delta_d_a_s_linear_elastic", 0), delta_d_a_s
            )

        if namespace.get("extrapolation_sub_type") == "paper":
            delta_d_m_e_s = K_shape * (1 / k) * inner(delta_d, phi) * dx_s[solid_region]
            delta_u_f_e_s = K_shape * theta0 * inner(delta_u, phi) * dx_s[solid_region]
            blocks["extension"]["delta_d_m_e_s"] = _add_form(
                blocks["extension"].get("delta_d_m_e_s", 0), delta_d_m_e_s
            )
            blocks["extension"]["delta_u_f_e_s"] = _add_form(
                blocks["extension"].get("delta_u_f_e_s", 0), delta_u_f_e_s
            )
            blocks["extension"]["delta_d_m_e_s_minus_delta_u_f_e_s"] = _add_form(
                blocks["extension"].get("delta_d_m_e_s_minus_delta_u_f_e_s", 0),
                delta_d_m_e_s - delta_u_f_e_s,
            )

    return blocks


def build_equation_3_component_blocks(DVP, **namespace):
    """Build visible equation (3) pieces split by increment component.

    These are hand-written UFL forms for the clearest paper blocks, such as
    ``delta_u_m_f`` and ``delta_u_a_f``. The aggregate derivative remains the
    source of truth for the complete Jacobian.
    """
    chi = TrialFunction(DVP)
    chi_parts = split(chi)
    return {
        "fluid": _build_fluid_component_blocks(chi_parts, namespace),
        "solid": _build_solid_component_blocks(chi_parts, namespace),
        "extension": _build_extension_component_blocks(chi_parts, namespace),
    }


def build_equation_3_blocks(
    DVP,
    dvp_,
    fluid_paper_terms=None,
    solid_paper_terms=None,
    extension_paper_terms=None,
    **namespace,
):
    """Build term-by-term Newton derivatives for the paper's equation (3).

    Each returned entry is the directional derivative of one named residual term
    with respect to the full mixed unknown ``dvp_["n"]`` in direction ``chi``.
    For example, ``fluid["m_f"]`` contains all increment contributions coming
    from the fluid mass residual, including the paper's delta-d and delta-u parts.
    """
    chi = TrialFunction(DVP)
    current_state = dvp_["n"]

    fluid_blocks, fluid_total = _differentiate_terms(fluid_paper_terms or {}, current_state, chi)
    solid_blocks, solid_total = _differentiate_terms(solid_paper_terms or {}, current_state, chi)
    extension_blocks, extension_total = _differentiate_terms(
        extension_paper_terms or {}, current_state, chi
    )

    total = 0
    total = _add_form(total, fluid_total)
    total = _add_form(total, solid_total)
    total = _add_form(total, extension_total)

    return {
        "equation_3_chi": chi,
        "equation_3_blocks": {
            "fluid": fluid_blocks,
            "solid": solid_blocks,
            "extension": extension_blocks,
        },
        "equation_3_component_blocks": build_equation_3_component_blocks(DVP, **namespace),
        "equation_3_jacobian": total,
    }
