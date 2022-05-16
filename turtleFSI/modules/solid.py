# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from turtleFSI.modules import *
from dolfin import Constant, inner, grad


def solid_setup(d_, v_, phi, psi, dx_s, dx_s_id_list, solid_properties, k, theta,
                gravity, mesh, **namespace):

    # DB added gravity in 3d functionality and multi material capability 16/3/21
    #
    #

    """
    ALE formulation (theta-scheme) of the non-linear elastic problem:
    dv/dt - f + div(sigma) = 0   with v = d(d)/dt
    """

    # From the equation defined above we have to include the equation v - d(d)/dt = 0. This
    # ensures both that the variable d and v is well defined in the solid equation, but also
    # that there is continuity of the velocity at the boundary. Since this is imposed weakly
    # we 'make this extra important' by multiplying with a large number delta.

    delta = 1E7

    # Theta scheme constants
    theta0 = Constant(theta)
    theta1 = Constant(1 - theta)

    F_solid_linear = 0
    F_solid_nonlinear = 0
    for solid_region in range(len(dx_s_id_list)):
        rho_s = solid_properties[solid_region]["rho_s"]

        # Temporal term and convection 
        F_solid_linear += (rho_s/k * inner(v_["n"] - v_["n-1"], psi)*dx_s[solid_region]
                          + delta * rho_s * (1 / k) * inner(d_["n"] - d_["n-1"], phi) * dx_s[solid_region]
                          - delta * rho_s * inner(theta0 * v_["n"] + theta1 * v_["n-1"], phi) * dx_s[solid_region]) # Maybe we could add viscoelasticity with v["n"] term instead of (1 / k) * inner(d_["n"] - d_["n-1"], phi) 
        
        if solid_properties[solid_region]["viscoelasticity"] == None:
            pass
        elif solid_properties[solid_region]["viscoelasticity"] == "Form1":
            F_solid_nonlinear += (1/k) * inner(F_(d_["n"])*theta0*Svisc(d_["n"] - d_["n-1"], solid_properties[solid_region]), grad(psi)) * dx_s[solid_region]
            F_solid_linear += (1/k) * inner(F_(d_["n-1"])*theta1*Svisc(d_["n"] - d_["n-1"], solid_properties[solid_region]), grad(psi)) * dx_s[solid_region]
        elif solid_properties[solid_region]["viscoelasticity"] == "Form2":
            F_solid_linear += (1/k) * inner(Piola1visc(d_["n"] - d_["n-1"], solid_properties[solid_region]), grad(psi)) * dx_s[solid_region] # First attempt at viscoelastic term. 
        else:
            print("invalid entry for viscoelasticity")

        # Stress
        F_solid_nonlinear += theta0 * inner(Piola1(d_["n"], solid_properties[solid_region]), grad(psi)) * dx_s[solid_region]
        F_solid_linear += theta1 * inner(Piola1(d_["n-1"], solid_properties[solid_region]), grad(psi)) * dx_s[solid_region]
        # Gravity
        if gravity is not None and mesh.geometry().dim() == 2:
            F_solid_linear -= inner(Constant((0, -gravity * rho_s)), psi)*dx_s[solid_region] 
        elif gravity is not None and mesh.geometry().dim() == 3:
            F_solid_linear -= inner(Constant((0, -gravity * rho_s,0)), psi)*dx_s[solid_region] 

    return dict(F_solid_linear=F_solid_linear, F_solid_nonlinear=F_solid_nonlinear)
