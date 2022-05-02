# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import inner, inv, grad, CellVolume
from turtleFSI.modules import *


def extrapolate_setup(F_fluid_linear, extrapolation_sub_type, mesh, d_, phi,
                      dx_f, dx_f_id_list, **namespace):
    """
    Laplace lifting operator. Can be used for small to moderate deformations.
    The diffusion parameter "alfa", which is specified by "extrapolation_sub_type",
    can be used to control the deformation field from the wall boundaries to the
    fluid domain. "alfa" is assumed constant within elements.

    - alfa * laplace(d) = 0   in the fluid domain
    d = 0                     on the fluid boundaries other than FSI interface
    d = solid_def             on the FSI interface

    References:

    Slyngstad, Andreas Strøm. Verification and Validation of a Monolithic
        Fluid-Structure Interaction Solver in FEniCS. A comparison of mesh lifting
        operators. MS thesis. 2017.

    Gjertsen, Sebastian. Development of a Verified and Validated Computational
        Framework for Fluid-Structure Interaction: Investigating Lifting Operators
        and Numerical Stability. MS thesis. 2017.
    """

    if extrapolation_sub_type == "volume_change":
        alfa = 1.0 / (J_(d_["n"]))
    elif extrapolation_sub_type == "volume":
        alfa = 1.0 / CellVolume(mesh)
    elif extrapolation_sub_type == "small_constant":
        alfa = 0.01 * (mesh.hmin())**2
    elif extrapolation_sub_type == "constant":
        alfa = 1.0
    else:
        raise RuntimeError("Could not find extrapolation method {}".format(extrapolation_sub_type))

    for fluid_region in range(len(dx_f_id_list)): # for all fluid regions
        F_extrapolate = alfa * inner(grad(d_["n"]), grad(phi)) * dx_f[fluid_region]
        F_fluid_linear += F_extrapolate

    return dict(F_fluid_linear=F_fluid_linear)
