# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import Constant, inner


def fluid_setup(psi, phi, dx_f, mesh, **namespace):
    """No contribution from the fluid, for when solving only the structure equation."""

    F_fluid_linear = inner(Constant(tuple([0]*mesh.geometry().dim())), psi)*dx_f
    F_fluid_nonlinear = inner(Constant(tuple([0]*mesh.geometry().dim())), phi)*dx_f

    return dict(F_fluid_linear=F_fluid_linear, F_fluid_nonlinear=F_fluid_nonlinear)
