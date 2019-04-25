# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import Constant, inner


def solid_setup(psi, phi, dx_s, mesh, **namespace):
    """No contribution from the structure, for when solving only the fluid equation."""
    F_solid_linear = inner(Constant(tuple([0]*mesh.geometry().dim())), psi)*dx_s
    F_solid_nonlinear = inner(Constant(tuple([0]*mesh.geometry().dim())), phi)*dx_s

    return dict(F_solid_linear=F_solid_linear, F_solid_nonlinear=F_solid_nonlinear)
