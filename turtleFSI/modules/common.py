# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Common functions used in the variational formulations for the variational forms
of the mesh lifting equations, fluid equations and the structure equation.
"""

from dolfin import grad, det, Identity, tr, inv


def F_(d):
    """
    Deformation gradient tensor
    """
    return Identity(len(d)) + grad(d)


def J_(d):
    """
    Determinant of the deformation gradient
    """
    return det(F_(d))


def eps(d):
    """
    Infinitesimal strain tensor
    """
    return 0.5 * (grad(d) * inv(F_(d)) + inv(F_(d)).T * grad(d).T)


def sigma_f_u(u, d, mu_f):
    """
    Deviatoric component of the Cauchy stress tensor (fluid problem)
    """
    return mu_f * (grad(u) * inv(F_(d)) + inv(F_(d)).T * grad(u).T)


def sigma_f_p(p, u):
    """
    Hydrostatic component of the Cauchy stress tensor (fluid problem)
    """
    return -p * Identity(len(u))


def sigma(u, p, d, mu_f):
    """
    Cauchy stress tensor (fluid problem)
    """
    return sigma_f_u(u, d, mu_f) + sigma_f_p(p, u)


def E(U):
    """
    Green-Lagrange strain tensor
    """
    return 0.5*(F_(U).T*F_(U) - Identity(len(U)))


def S(U, lambda_s, mu_s):
    """
    Second Piola-Kirchhoff Stress (solid problem - Saint Venant-Kirchhoff materials)
    """
    I = Identity(len(U))
    return 2*mu_s*E(U) + lambda_s*tr(E(U))*I


def Piola1(U, lambda_s, mu_s):
    """
    First Piola-Kirchhoff Stress (solid problem)
    """
    return F_(U)*S(U, lambda_s, mu_s)


def S_linear(U, alfa_mu, alfa_lam):
    """
    Second Piola-Kirchhoff Stress (mesh problem - Linear Elastic materials)
    """
    return alfa_lam * tr(eps(d)) * Identity(len(d)) + 2.0 * alfa_mu * eps(d)
