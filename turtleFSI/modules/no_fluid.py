# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import Constant, inner


def fluid_setup(psi, phi, dx_f,dx_f_id, mesh, **namespace):
    F_fluid_linear = 0
    F_fluid_nonlinear = 0
    for fluid_region in range(len(dx_f_id)):
        """No contribution from the fluid, for when solving only the structure equation."""
        F_fluid_linear += inner(Constant(tuple([0] * mesh.geometry().dim())), psi) * dx_f[fluid_region]
        F_fluid_nonlinear += inner(Constant(tuple([0] * mesh.geometry().dim())), phi) * dx_f[fluid_region]

    return dict(F_fluid_linear=F_fluid_linear, F_fluid_nonlinear=F_fluid_nonlinear)
