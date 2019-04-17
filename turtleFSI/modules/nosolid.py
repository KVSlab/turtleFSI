# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Solve xxx
"""

from dolfin import Constant, inner


def fluid_setup(psi, dx_s, ds, **namespace):
	F_fluid_linear = inner(Constant((0, 0)), psi)*dx_s
	F_fluid_nonlinear = inner(Constant((0, 0)), phi)*dx_s

	return dict(F_fluid_linear = F_fluid_linear, F_fluid_nonlinear = F_fluid_nonlinear)
