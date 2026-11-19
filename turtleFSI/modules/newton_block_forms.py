# File under GNU GPL (v3) licence, see LICENSE file for details.

"""Helpers for paper-style Newton block forms.

The production solver can keep using the aggregate UFL derivative. These helpers
expose the same linearization term-by-term so equation block (3) from
Ballarin/Rozza can be inspected, assembled, or projected later.

Equation block (4) in the paper is the algebraic Newton system obtained after
assembling these component derivatives on the high-fidelity basis. The ROM-side
projection of that system lives in ``turtleFSI.rom._pod_galerkin``.

See paper/GALERKIN_EQUATION_AUDIT.md for mapping, sign, and summation conventions.
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
    paper_fluid_indicator,
    paper_interface_ids,
)


def _is_zero_form(form):
    return isinstance(form, (int, float)) and form == 0


def _add_form(total, form):
    if _is_zero_form(form):
        return total
    if _is_zero_form(total):
        return form
    return total + form


def _differentiate_terms(terms, dvp, chi, aliases=(), signs=None):
    """Keep named derivatives, but sum unique residual terms with their signs."""
    signs = signs or {}
    differentiated = {}
    total = 0
    for name, form in terms.items():
        if _is_zero_form(form):
            differentiated[name] = 0
            continue
        block = derivative(form, dvp, chi)
        differentiated[name] = block
        if name not in aliases:
            total = _add_form(total, signs.get(name, 1) * block)
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

# Table 1: expressions
        # fluid time derivatives: m_f
        # m_f derivative w.r.t displacement
        delta_d_m_f = rho_f / k * inner(
            K_shape * theta0 * J_current * dJ_trace * current_velocity_difference,
            psi,
        ) * dx_f[fluid_region]

        # m_f derivative w.r.t velocity
        delta_u_m_f = rho_f / k * inner(
            K_shape * (theta0 * J_current + theta1 * J_previous)
            * delta_u, psi, ) * dx_f[fluid_region]

        blocks["fluid_momentum"]["delta_d_m_f"] = _add_form(
            blocks["fluid_momentum"].get("delta_d_m_f", 0), delta_d_m_f
        )
        blocks["fluid_momentum"]["delta_u_m_f"] = _add_form(
            blocks["fluid_momentum"].get("delta_u_m_f", 0), delta_u_m_f
        )

        # fluid stiffness derivatives: a_f
        # a_f derivative w.r.t displacement
        delta_d_a_f = theta0 * inner(
            K_shape * J_current * mu_f * (
                dJ_trace * transformed_velocity_strain * B # terms 1 & 2
                - grad(v_["n"]) * A * grad(delta_d) * A * B # term 3
                - B * grad(delta_d).T * B * grad(v_["n"]).T * B # term 4
                - grad(v_["n"]) * A * B * grad(delta_d).T * B # term 5
                - B * grad(v_["n"]).T * B * grad(delta_d).T * B # term 6
            ),
            grad(psi),
        ) * dx_f[fluid_region]

        # a_f derivative w.r.t velocity
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

        # fluid incompressibility derivatives: b_f: pressure
        # b_f derivative w.r.t displacement
        delta_d_b_f_momentum = inner(
            K_shape
            * J_current
            * (
                -p_["n"] * dJ_trace * B
                + p_["n"] * B * grad(delta_d).T * B
            ),
            grad(psi),
        ) * dx_f[fluid_region]

        # b_f derivative w.r.t pressure
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

        # fluid advection derivatives: c_f
        # c_f derivative w.r.t displacement
        delta_d_c_f = theta0 * rho_f * inner(
            K_shape
            * J_current
            * (
                dJ_trace * transformed_velocity_gradient * v_["n"]
                - grad(v_["n"]) * A * grad(delta_d) * A * v_["n"]
            ),
            psi,
        ) * dx_f[fluid_region]

        # c_f derivative w.r.t velocity
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

        # fluid ALE derivatives: c_f_ALE
        # c_f_ALE derivative w.r.t displacement
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

        # c_f_ALE derivative w.r.t velocity
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
            # fluid forcing term derivatives: f_f
            # f_f derivative w.r.t displacement
            delta_d_f_f = -inner(
                K_shape * J_current * rho_f * dJ_trace * force,
                psi,
            ) * dx_f[fluid_region]

            blocks["fluid_momentum"]["delta_d_f_f"] = _add_form(
                blocks["fluid_momentum"].get("delta_d_f_f", 0), delta_d_f_f
            )

        # fluid incompressibility derivatives: b_f: divergence free term
        # b_f_continuity derivative w.r.t displacement
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

        # b_f_continuity derivative w.r.t velocity
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

        # extension fluid time derivative: m_e_f
        # m_e_f derivative w.r.t displacement
        delta_d_m_e_f = K_shape * (1 / k) * inner(delta_d, phi) * dx_f[fluid_region]

        # extension fluid stiffness derivatives: a_e_f
        # a_e_f derivative w.r.t displacement
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

    dS, n = namespace.get("dS"), namespace.get("n")
    interface_ids = paper_interface_ids(namespace.get("fsi_id"), dS, n)
    if interface_ids:
        fluid_indicator = namespace.get("fluid_indicator")
        if fluid_indicator is None:
            fluid_indicator = paper_fluid_indicator(dx_f, dx_f_id_list)
        interface_derivative = 0
        for side in ("+", "-"):
            d_side = d_["n"](side)
            delta_d_side = delta_d(side)
            shape_side = None if shape_displacement is None else shape_displacement(side)
            G_side_inv = inv(G_(d_side, shape_side))
            A_side = G_side_inv * inv(F_(d_side, shape_side))
            J_side = J_(d_side, shape_side)
            dJ_trace_side = tr(A_side * grad(delta_d_side))
            GG_T_side = G_side_inv * G_side_inv.T
            flux_derivative = (
                (1 / J_side)
                * (-dJ_trace_side * grad(d_side) * GG_T_side
                   + grad(delta_d_side) * GG_T_side) * n(side)
            )
            interface_derivative += fluid_indicator(side) * inner(flux_derivative, phi(side))
        for marker in interface_ids:
            # The residual contains -a_e_df; use the same fluid trace and sign.
            delta_d_a_e_df = -interface_derivative * dS(marker)
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

    Each named entry is differentiated with respect to the full mixed unknown
    ``dvp_["n"]`` in direction ``chi``. Inspection aliases remain available, and
    ``f_e_s`` retains its positive named-form convention. Only the returned
    ``equation_3_jacobian`` sums unique paper terms with their residual signs;
    summing the inspection dictionaries directly is not equivalent.
    For example, ``fluid["m_f"]`` contains all increment contributions coming
    from the fluid mass residual, including the paper's delta-d and delta-u parts.
    """
    chi = TrialFunction(DVP)
    current_state = dvp_["n"]

    fluid_blocks, fluid_total = _differentiate_terms(fluid_paper_terms or {}, current_state, chi)
    # Combined relations are inspection aliases, not additional equations.
    # f_e_s is a positive named form, but enters the residual with a minus sign.
    solid_blocks, solid_total = _differentiate_terms(
        solid_paper_terms or {}, current_state, chi,
        aliases=("extension_relation_s",), signs={"f_e_s": -1},
    )
    extension_blocks, extension_total = _differentiate_terms(
        extension_paper_terms or {}, current_state, chi,
        aliases=("extension_relation_f",),
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
