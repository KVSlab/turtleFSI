"""Inspectable Ballarin/Rozza (2016), equation (4), in mixed (d, u, p) order.

Call ``build_equation_4_forms(**namespace)`` after fluid/solid/extension setup,
then ``assemble_equation_4(forms)`` at the Newton state you want to inspect.
These are the unmodified variational matrices, before Dirichlet row replacement
or inactive-pressure diagonal repair. All operators are embedded in DVP space;
their nonzero rows/columns identify their physical block. No dense conversion
or extra matrix assembly is added to the production Newton loop.

F/S/E superscripts mean fluid/solid/extension. A suffix _d, _u, or _p means
directional differentiation, not multiplication by the state. In particular
C_F_u is the TWO-term Newton derivative of convection, not the frozen
convection operator C_F. Mass matrices here already include 1 / dt.
"""

from dolfin import assemble

from turtleFSI.modules.newton_block_forms import build_equation_3_component_blocks


def build_equation_4_forms(DVP, **namespace):
    """Return named operators and the seven nonzero mixed Jacobian blocks.

    This explicitly represents the paper's unsteady linear-elastic model with
    implicit Euler. Other material models and theta schemes must use the full
    production residual/Jacobian instead. Extra problem-specific loads/supports
    are not included here; compare against production J when using such terms.
    Forms retain references to the current/previous Functions: assemble again
    after changing either state. Missing required component forms raise errors.
    """
    if namespace.get("extrapolation_sub_type") != "paper":
        raise ValueError("Equation (4) requires extrapolation_sub_type='paper'.")
    if float(namespace["theta"]) != 1.0:
        raise ValueError("Equation (4) uses implicit Euler: theta=1.")
    if not namespace.get("solid_properties") or any(
        prop.get("material_model") not in ("LinearElastic", "BallarinRozzaLinearElastic")
        for prop in namespace["solid_properties"]
    ):
        raise ValueError("Equation (4) requires the paper's linear-elastic solid.")

    components = build_equation_3_component_blocks(DVP, **namespace)
    fluid = components["fluid"]["fluid_momentum"]
    continuity = components["fluid"]["fluid_continuity"]
    solid = components["solid"]["solid_momentum"]
    extension_f = components["extension"]["extension_fluid"]
    extension_s = components["solid"]["extension"]

    # Table I -> equation (4): each bilinear form becomes a matrix on assembly.
    M_F = fluid["delta_u_m_f"]
    M_F_d = fluid["delta_d_m_f"]
    A_F = fluid["delta_u_a_f"]
    A_F_d = fluid["delta_d_a_f"]
    B_F_p = fluid["delta_p_b_f"]
    B_F_d_momentum = fluid["delta_d_b_f"]
    C_F_u = fluid["delta_u_c_f"]
    C_F_d = fluid["delta_d_c_f"]
    # Existing component forms already carry their residual minus signs.
    # Undo those signs so the sums below read exactly as the printed equation.
    C_ALE_F_u = -fluid["delta_u_c_f_ALE"]
    C_ALE_F_d = -fluid["delta_d_c_f_ALE"]
    F_F_d = -fluid.get("delta_d_f_f", 0)
    B_F_u = continuity["delta_u_b_f"]
    B_F_d_continuity = continuity["delta_d_b_f"]
    M_S = solid["delta_u_m_s"]
    A_S = solid["delta_d_a_s_linear_elastic"]
    M_E_F = extension_f["delta_d_m_e_f"]
    # A^{E,F}_{,d} includes volume stiffness MINUS fluid-side interface flux.
    A_E_F_d = extension_f["delta_d_a_e_f"] + extension_f.get("delta_d_a_e_df", 0)
    M_E_S = extension_s["delta_d_m_e_s"]
    F_E_S = extension_s["delta_u_f_e_s"]

    operators = dict(
        M_F=M_F, M_F_d=M_F_d, A_F=A_F, A_F_d=A_F_d,
        B_F_p=B_F_p, B_F_d_momentum=B_F_d_momentum,
        C_F_u=C_F_u, C_F_d=C_F_d,
        C_ALE_F_u=C_ALE_F_u, C_ALE_F_d=C_ALE_F_d, F_F_d=F_F_d,
        B_F_u=B_F_u, B_F_d_continuity=B_F_d_continuity,
        M_S=M_S, A_S=A_S, M_E_F=M_E_F, A_E_F_d=A_E_F_d,
        M_E_S=M_E_S, F_E_S=F_E_S,
    )

    # Equation (4), regrouped into global displacement/velocity/pressure fields:
    # [ J_dd  J_du   0   ] [delta_d]     [R_extension ]
    # [ J_ud  J_uu  J_up ] [delta_u] = - [R_momentum  ]
    # [ J_pd  J_pu   0   ] [delta_p]     [R_continuity]
    J_dd = M_E_F + A_E_F_d + M_E_S
    J_du = -F_E_S
    J_ud = M_F_d + A_F_d + B_F_d_momentum + C_F_d - C_ALE_F_d - F_F_d + A_S
    J_uu = M_F + A_F + C_F_u - C_ALE_F_u + M_S
    J_up = B_F_p
    J_pd = B_F_d_continuity
    J_pu = B_F_u
    blocks = dict(J_dd=J_dd, J_du=J_du, J_ud=J_ud, J_uu=J_uu,
                  J_up=J_up, J_pd=J_pd, J_pu=J_pu)
    return dict(operators=operators, blocks=blocks,
                jacobian=J_dd + J_du + J_ud + J_uu + J_up + J_pd + J_pu)


def assemble_equation_4(forms, compiler_parameters=None):
    """Assemble sparse DVP-sized matrices at the current Newton state.

    ``result['operators']['M_F']`` is the fluid mass matrix; ``result['jacobian']``
    is their signed block sum. An absent optional body-force derivative is None.
    Call only for diagnostics/export: assembling every term costs extra work.
    Boundary conditions must be applied to the combined system, not separately
    to each physical contribution (which would duplicate boundary diagonals).
    """
    def matrix(form):
        if isinstance(form, (int, float)) and form == 0:
            return None
        return assemble(form, form_compiler_parameters=compiler_parameters or {},
                        keep_diagonal=True)

    return dict(
        operators={name: matrix(form) for name, form in forms["operators"].items()},
        blocks={name: matrix(form) for name, form in forms["blocks"].items()},
        jacobian=matrix(forms["jacobian"]),
    )
