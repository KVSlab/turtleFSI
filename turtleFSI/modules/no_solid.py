# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Solve xxx
"""

from dolfin import Constant, inner


def solid_setup(psi, phi, dx_s, **namespace):
	F_solid_linear = inner(Constant((0, 0)), psi)*dx_s
	F_solid_nonlinear = inner(Constant((0, 0)), phi)*dx_s

	return dict(F_solid_linear=F_solid_linear, F_solid_nonlinear=F_solid_nonlinear)
