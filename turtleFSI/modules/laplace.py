# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import inner, inv, grad, CellVolume
from turtleFSI.modules import *


def extrapolate_setup(F_fluid_linear, extrapolation_sub_type, mesh, d_, phi,
                      dx_f, dx_f_id_list, shape_displacement=None, F_fluid_nonlinear=0,
                      k=None, dS=None, n=None, fsi_id=None, **namespace):
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

    extension_paper_terms = {
        "m_e_f": 0,
        "a_e_f": 0,
        "a_e_df": 0,
        "extension_relation_f": 0,
    }

    if extrapolation_sub_type == "volume_change":
        alfa = 1.0 / (J_(d_["n"]))
    elif extrapolation_sub_type == "volume":
        alfa = 1.0 / CellVolume(mesh)
    elif extrapolation_sub_type == "small_constant":
        alfa = 0.01 * (mesh.hmin())**2
    elif extrapolation_sub_type in ("constant", "paper"):
        alfa = 1.0
    else:
        raise RuntimeError("Could not find extrapolation method {}".format(extrapolation_sub_type))

    for fluid_region in range(len(dx_f_id_list)): # for all fluid regions
        if extrapolation_sub_type == "paper":
            if k is None:
                raise RuntimeError("The paper extension formulation requires the timestep constant k")
            G_shape_inv = inv(G_(d_["n"], shape_displacement))

            # extension fluid time derivative: m_e_f
            m_e_f = K_(d_["n"], shape_displacement) * (1 / k) * inner(d_["n"] - d_["n-1"], phi)

            # extension fluid stiffness: a_e_f
            a_e_f = inner(
                    K_(d_["n"], shape_displacement) * sigma_ext(d_["n"], shape_displacement) * G_shape_inv.T,
                    grad(phi), )

            F_extrapolate = (m_e_f + a_e_f) * dx_f[fluid_region]
            F_fluid_nonlinear += F_extrapolate
            extension_paper_terms["m_e_f"] += m_e_f * dx_f[fluid_region]
            extension_paper_terms["a_e_f"] += a_e_f * dx_f[fluid_region]
            extension_paper_terms["extension_relation_f"] += F_extrapolate

        else:
            # not part of paper / default turtle FSI extension
            F_extrapolate = alfa * inner(grad(d_["n"]), grad(phi)) * dx_f[fluid_region]
            F_fluid_linear += F_extrapolate

    if extrapolation_sub_type == "paper" and fsi_id is not None and dS is not None and n is not None:
        interface_ids = fsi_id if isinstance(fsi_id, list) else [fsi_id]
        d_plus = d_["n"]("+")
        phi_plus = phi("+")
        shape_plus = None if shape_displacement is None else shape_displacement("+")
        G_plus_inv = inv(G_(d_plus, shape_plus))
        extension_flux = K_(d_plus, shape_plus) * sigma_ext(d_plus, shape_plus) * G_plus_inv.T * n("+")

        # extension fluid interface stiffness: a_e_df
        for marker in interface_ids:
            a_e_df = inner(extension_flux, phi_plus) * dS(marker)
            F_fluid_nonlinear -= a_e_df
            extension_paper_terms["a_e_df"] -= a_e_df

    return dict(F_fluid_linear=F_fluid_linear, F_fluid_nonlinear=F_fluid_nonlinear,
                extension_paper_terms=extension_paper_terms)
