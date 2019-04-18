# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Solves the equation

bla bla

TODO
"""

from dolfin import inner, inv, grad, CellVolume
from turtleFSI.modules import *

def extrapolate_setup(F_fluid_linear, extrapolation_sub_type, mesh, d_, phi,
                      dx_f, **namespace):
    if extrapolation_sub_type == "volume_change":
        alfa = 1./(J_(d_["n"]))
    elif extrapolation_sub_type == "volume":
        alfa = 1./CellVolume(mesh)
    elif extrapolation_sub_type == "small_constant":
        alfa = 0.01*(mesh.hmin())**2
    elif extrapolation_sub_type == "constant":
        alfa = 1.0
    else:
        raise RuntimeError("Could not find extrapolation method {}".format(extype))

    F_extrapolate = alfa*inner(grad(d_["n"]), grad(phi))*dx_f
    F_fluid_linear += F_extrapolate

    return dict(F_fluid_linear=F_fluid_linear)
