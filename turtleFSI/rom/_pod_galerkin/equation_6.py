"""Explicit reduced equation (6), in (d, enriched u, p) order.

Operators use the equation_4.py sign convention and already contain time-step
factors. No extra mass scaling or convection linearization is applied here.
"""

import numpy as np


def project_equation_6(operators, Z_d, Z_u_s, Z_p, project):
    """Project assembled equation (4) operators with a sparse-action callback.

    ``project(A, left, right)`` computes ``left.T @ A @ right``.
    ``None`` represents an absent optional full-order contribution.
    """
    def reduced(name, left, right):
        matrix = operators[name]
        if matrix is None:
            return np.zeros((left.shape[1], right.shape[1]))
        return project(matrix, left, right)

    M_F_N = reduced("M_F", Z_u_s, Z_u_s)
    A_F_N = reduced("A_F", Z_u_s, Z_u_s)  # Z_u_s.T @ A_F @ Z_u_s
    C_F_u_N = reduced("C_F_u", Z_u_s, Z_u_s)
    C_ALE_F_u_N = reduced("C_ALE_F_u", Z_u_s, Z_u_s)
    M_S_N = reduced("M_S", Z_u_s, Z_u_s)
    M_F_d_N = reduced("M_F_d", Z_u_s, Z_d)
    A_F_d_N = reduced("A_F_d", Z_u_s, Z_d)
    B_F_d_momentum_N = reduced("B_F_d_momentum", Z_u_s, Z_d)
    C_F_d_N = reduced("C_F_d", Z_u_s, Z_d)
    C_ALE_F_d_N = reduced("C_ALE_F_d", Z_u_s, Z_d)
    F_F_d_N = reduced("F_F_d", Z_u_s, Z_d)
    A_S_N = reduced("A_S", Z_u_s, Z_d)
    B_F_p_N = reduced("B_F_p", Z_u_s, Z_p)
    B_F_u_N = reduced("B_F_u", Z_p, Z_u_s)
    B_F_d_continuity_N = reduced("B_F_d_continuity", Z_p, Z_d)
    M_E_F_N = reduced("M_E_F", Z_d, Z_d)
    A_E_F_d_N = reduced("A_E_F_d", Z_d, Z_d)
    M_E_S_N = reduced("M_E_S", Z_d, Z_d)
    F_E_S_N = reduced("F_E_S", Z_d, Z_u_s)

    named = dict(
        M_F_N=M_F_N, A_F_N=A_F_N, C_F_u_N=C_F_u_N,
        C_ALE_F_u_N=C_ALE_F_u_N, M_S_N=M_S_N,
        M_F_d_N=M_F_d_N, A_F_d_N=A_F_d_N, B_F_d_momentum_N=B_F_d_momentum_N,
        C_F_d_N=C_F_d_N, C_ALE_F_d_N=C_ALE_F_d_N, F_F_d_N=F_F_d_N,
        A_S_N=A_S_N, B_F_p_N=B_F_p_N, B_F_u_N=B_F_u_N,
        B_F_d_continuity_N=B_F_d_continuity_N,
        M_E_F_N=M_E_F_N, A_E_F_d_N=A_E_F_d_N, M_E_S_N=M_E_S_N, F_E_S_N=F_E_S_N,
    )
    J_dd_N = M_E_F_N + A_E_F_d_N + M_E_S_N
    J_du_N = -F_E_S_N
    J_ud_N = M_F_d_N + A_F_d_N + B_F_d_momentum_N + C_F_d_N - C_ALE_F_d_N - F_F_d_N + A_S_N
    J_uu_N = M_F_N + A_F_N + C_F_u_N - C_ALE_F_u_N + M_S_N
    J_up_N = B_F_p_N
    J_pd_N = B_F_d_continuity_N
    J_pu_N = B_F_u_N
    nd, np_ = Z_d.shape[1], Z_p.shape[1]
    J_dp_N = np.zeros((nd, np_))
    J_pp_N = np.zeros((np_, np_))
    J_N = np.block([
        [J_dd_N, J_du_N, J_dp_N],
        [J_ud_N, J_uu_N, J_up_N],
        [J_pd_N, J_pu_N, J_pp_N],
    ])
    named.update(J_dd_N=J_dd_N, J_du_N=J_du_N, J_dp_N=J_dp_N,
                 J_ud_N=J_ud_N, J_uu_N=J_uu_N, J_up_N=J_up_N,
                 J_pd_N=J_pd_N, J_pu_N=J_pu_N, J_pp_N=J_pp_N, J_N=J_N)
    return named
