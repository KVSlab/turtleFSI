# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Solves xxx
TODO
"""

from turtleFSI.modules import *
from dolfin import CellVolume, inner, grad, inv


def extrapolate_setup(F_fluid_linear, mesh, d_, phi, gamma, dx_f, **namespace):
    E_y = 1./CellVolume(mesh)
    nu = 0.25
    alfa_lam = nu*E_y / ((1. + nu)*(1. - 2.*nu))
    alfa_mu = E_y/(2.*(1. + nu))
    F_extrapolate = inner(J_(d_["n"]) * STVK(d_["n"], alfa_mu, alfa_lam) *
                          inv(F_(d_["n"])).T, grad(phi))*dx_f

    F_fluid_linear += F_extrapolate

    return dict(F_fluid_linear=F_fluid_linear)
