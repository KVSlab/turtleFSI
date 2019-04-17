# TODO: Add header

"""
The alpha_CN.py implements the equation:

    div(alpha(u) * grad(u)) = 0

Where alpha is 1 / J(u), which physically represents one over
volume or area. This is to make the smaller cells stiffer.
"""

from turtleFSI.modules.common import *
from dolfin import inner, grad, det, Identity


def extrapolate_setup(d_, phi, dx_f,  F_fluid_nonlinear, **semimp_namespace):
    F_extrapolate = inner((1./J_(d_["n"])) * grad(d_["n"]), grad(phi))*dx_f
    F_fluid_nonlinear += F_extrapolate

    return dict(F_fluid_nonlinear = F_fluid_nonlinear)
