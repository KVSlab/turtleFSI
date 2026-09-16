# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Common functions used in the variational formulations for the variational forms
of the mesh lifting equations, fluid equations and the structure equation.
"""

from dolfin import grad, det, Identity, tr, inv, inner, as_tensor, as_vector, derivative, variable, Constant
import ufl  # ufl module

def get_dimension(u):

    # Get dimension of tensor
    try:
        from ufl.domain import find_geometric_dimension

        dim = find_geometric_dimension(u)

    except:

        try:
            dim = len(u)
        except:
            dim = 3

    return dim

def F_(d, shape_displacement=None):
    """
    Physical deformation gradient tensor pulled back to the reference domain.

    With no shape map this is the original turtleFSI F = I + grad(d). With a
    shape map T(x; mu), the paper's definition is F = I + grad(d) G^{-1}.
    """
    return Identity(get_dimension(d)) + grad(d) * inv(G_(d, shape_displacement))

def L_(v):
    """
    spatial velocity gradient tensor
    """
    return grad(v)

def D_(v):
    """
    Rate of deformation tensor
    """    
    return 0.5*(L_(v) + L_(v).T)

def J_(d, shape_displacement=None):
    """
    Determinant of the physical deformation gradient.
    """
    return det(F_(d, shape_displacement))


def G_(u, shape_displacement=None):
    """
    Shape-parametrization Jacobian matrix.

    Defaults to the identity map. Passing a shape displacement field represents
    T(x; mu) = x + shape_displacement(x; mu), so G = I + grad(shape_displacement).
    """
    if shape_displacement is None:
        return Identity(get_dimension(u))
    return Identity(get_dimension(shape_displacement)) + grad(shape_displacement)


def K_(u, shape_displacement=None):
    """
    Determinant of the shape-parametrization Jacobian matrix.
    """
    return det(G_(u, shape_displacement))


def region_value(value, region):
    """
    Return a global or per-region parameter value.
    """
    if isinstance(value, dict):
        return value.get(region)
    if isinstance(value, (list, tuple)) and value and not isinstance(value[0], (int, float)):
        return value[region]
    return value


def vector_body_force(value, dim):
    """
    Convert a vector body-force value into a UFL-compatible object.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return Constant(tuple(value))
    return value

# extension: harmonic extension: sigma_ext_d_f
def sigma_ext(d, shape_displacement=None):
    """
    Harmonic extension tensor from Ballarin/Rozza Section 2.3(D).
    """
    G_inv = inv(G_(d, shape_displacement))
    return (1.0 / (J_(d, shape_displacement) * K_(d, shape_displacement))) * grad(d) * G_inv



def eps(d, shape_displacement=None):
    """
    Infinitesimal strain tensor.
    """
    F = F_(d, shape_displacement)
    G_inv = inv(G_(d, shape_displacement))
    grad_d = grad(d) * G_inv
    return 0.5 * (grad_d * inv(F) + inv(F).T * grad_d.T)

# fluid: incompressible Newtonian fluid: fluid viscous stress
def sigma_f_u(u, d, mu_f, shape_displacement=None):
    """
    Deviatoric component of the Cauchy stress tensor (fluid problem).
    """
    G_inv = inv(G_(u, shape_displacement))
    transformed_gradient = grad(u) * G_inv * inv(F_(d, shape_displacement))
    return mu_f * (transformed_gradient + transformed_gradient.T)

# fluid: incompressible Newtonian fluid: pressure stress
def sigma_f_p(p, u):
    """
    Hydrostatic component of the Cauchy stress tensor (fluid problem)
    """
    return -p * Identity(get_dimension(u))


def sigma(u, p, d, mu_f, shape_displacement=None):
    """
    Cauchy stress tensor (fluid problem)
    """
    return sigma_f_u(u, d, mu_f, shape_displacement) + sigma_f_p(p, u)


def E(d, shape_displacement=None):
    """
    Green-Lagrange strain tensor.
    """
    F = F_(d, shape_displacement)
    return 0.5*(F.T*F - Identity(get_dimension(d)))


def S(d, solid_properties, shape_displacement=None):
    """
    Second Piola-Kirchhoff Stress (solid problem)
    """
    F = F_(d, shape_displacement)
    S__ = inv(F)*Piola1(d, solid_properties, shape_displacement)

    return S__

def Piola1(d, solid_properties, shape_displacement=None):
    """
    First Piola-Kirchhoff Stress (solid problem)
    """
    if solid_properties["material_model"] in ("LinearElastic", "BallarinRozzaLinearElastic"):
        I = Identity(get_dimension(d))
        lambda_s = solid_properties["lambda_s"]
        mu_s = solid_properties["mu_s"]

        # structure: compressible linear elastic material: e_s
        strain = 0.5 * (grad(d) * inv(G_(d, shape_displacement))
                        + inv(G_(d, shape_displacement)).T * grad(d).T)

        # structure compressible linear elastic material: P_s_d
        P = lambda_s * tr(strain) * I + 2 * mu_s * strain

    # default turtleFSI extensions: 
    elif solid_properties["material_model"] == "StVenantKirchoff":
        I = Identity(get_dimension(d)) # Identity matrix
        lambda_s = solid_properties["lambda_s"]
        mu_s = solid_properties["mu_s"]
        E_solid = E(d, shape_displacement)
        S_svk = 2*mu_s*E_solid + lambda_s*tr(E_solid)*I  # Calculate First Piola Kirchoff Stress with Explicit form of St. Venant Kirchoff model
        P = F_(d, shape_displacement)*S_svk  # Convert to First Piola-Kirchoff Stress
    else: 
        # ["StVenantKirchoff",""StVenantKirchoffEnergy","NeoHookean","MooneyRivlin","Gent"]
        F = ufl.variable(F_(d, shape_displacement))  # Note that dolfin and ufl "variable" are different.
        if solid_properties["material_model"] == "StVenantKirchoffEnergy":
            W = W_St_Venant_Kirchoff(F, solid_properties["lambda_s"], solid_properties["mu_s"])
        elif solid_properties["material_model"] == "NeoHookean":
            W = W_Neo_Hookean(F, solid_properties["lambda_s"], solid_properties["mu_s"])  
        elif solid_properties["material_model"] == "MooneyRivlin":
            W = W_Mooney_Rivlin(F, solid_properties["lambda_s"], solid_properties["mu_s"], solid_properties["C01"], solid_properties["C10"], solid_properties["C11"]) 
        elif solid_properties["material_model"] == "Gent":
            W = W_Gent(F, solid_properties["mu_s"], solid_properties["Jm"])  
        else:
            if MPI.rank(MPI.comm_world) == 0:
                print('Invalid entry for material_model, choose from ["StVenantKirchoff",""StVenantKirchoffEnergy","NeoHookean","MooneyRivlin","Gent"]')
        
        P = ufl.diff(W, F) # First Piola-Kirchoff Stress for compressible hyperelastic material (https://en.wikipedia.org/wiki/Hyperelastic_material)
    
    return P

def Svisc_D(v, solid_properties):
    """
    Second Piola-Kirchhoff Stress, viscoelastic component.
    Assumed the same form as normal SVK model, but using the strain rate tensor (D) instead of green-lagrange strain (E)
    """
    I = Identity(get_dimension(v)) # Identity matrix
    mu_visc_s = solid_properties["mu_visc_s"]  # viscoelastic material constant
    lambda_visc_s = solid_properties["lambda_visc_s"] # viscoelastic material constant
    S_svk = mu_visc_s*tr(D_(v))*I +  2*lambda_visc_s*D_(v) # Viscoelastic equation based on St. Venant Kirchoff model

    return S_svk

def S_linear(d, alfa_mu, alfa_lam, shape_displacement=None):
    """
    Second Piola-Kirchhoff Stress (mesh problem - Linear Elastic materials)
    """
    strain = eps(d, shape_displacement)
    return alfa_lam * tr(strain) * Identity(get_dimension(d)) + 2.0 * alfa_mu * strain


def get_eig(T):
########################################################################
# Method for the analytical calculation of eigenvalues for 3D-Problems #
# from: https://fenicsproject.discourse.group/t/hyperelastic-model-problems-on-plotting-stresses/3130/6
########################################################################
    '''
    Analytically calculate eigenvalues for a three-dimensional tensor T with a
    characteristic polynomial equation of the form

                lambda**3 - I1*lambda**2 + I2*lambda - I3 = 0   .

    Since the characteristic polynomial is in its normal form , the eigenvalues
    can be determined using Cardanos formula. This algorithm is based on:
    "Efficient numerical diagonalization of hermitian 3 × 3 matrices" by
    J. Kopp (equations: 21-34, with coefficients: c2=-I1, c1=I2, c0=-I3).

    NOTE:
    The method implemented here, implicitly assumes that the polynomial has
    only real roots, since imaginary ones should not occur in this use case.

    In order to ensure eigenvalues with algebraic multiplicity of 1, the idea
    of numerical perturbations is adopted from "Computation of isotropic tensor
    functions" by C. Miehe (1993). Since direct comparisons with conditionals
    have proven to be very slow, not the eigenvalues but the coefficients
    occuring during the calculation of them are perturbated to get distinct
    values.
    '''

    # determine perturbation from tolerance
    tol = 1e-8
    pert = 2*tol

    # get required invariants
    I1 = tr(T)                                                               # trace of tensor
    I2 = 0.5*(tr(T)**2-inner(T,T))                                        # 2nd invariant of tensor
    I3 = det(T)                                                              # determinant of tensor

    # determine terms p and q according to the paper
    # -> Follow the argumentation within the paper, to see why p must be
    # -> positive. Additionally ensure non-zero denominators to avoid problems
    # -> during the automatic differentiation
    p = I1**2 - 3*I2                                                            # preliminary value for p
    p = ufl.conditional(ufl.lt(p,tol),abs(p)+pert,p)                            # add numerical perturbation to p, if close to zero; ensure positiveness of p
    q = 27/2*I3 + I1**3 - 9/2*I1*I2                                             # preliminary value for q
    q = ufl.conditional(ufl.lt(abs(q),tol),q+ufl.sign(q)*pert,q)                # add numerical perturbation (with sign) to value of q, if close to zero

    # determine angle phi for calculation of roots
    phiNom2 =  27*( 1/4*I2**2*(p-I2) + I3*(27/4*I3-q) )                         # preliminary value for squared nominator of expression for angle phi
    phiNom2 = ufl.conditional(ufl.lt(phiNom2,tol),abs(phiNom2)+pert,phiNom2)    # add numerical perturbation to ensure non-zero nominator expression for angle phi
    phi = 1/3*ufl.atan_2(ufl.sqrt(phiNom2),q)                                   # calculate angle phi

    # calculate polynomial roots
    lambda1 = 1/3*(ufl.sqrt(p)*2*ufl.cos(phi)+I1)
    lambda2 = 1/3*(-ufl.sqrt(p)*(ufl.cos(phi)+ufl.sqrt(3)*ufl.sin(phi))+I1)
    lambda3 = 1/3*(-ufl.sqrt(p)*(ufl.cos(phi)-ufl.sqrt(3)*ufl.sin(phi))+I1)
    

    # return polynomial roots (eigenvalues)
    #eig = as_tensor([[lambda1 ,0 ,0],[0 ,lambda2 ,0],[0 ,0 ,lambda3]])

    return lambda1, lambda2, lambda3 

"""
The following functions starting with W are strain energy density functions for hyperelastic materials. 
"""

def W_St_Venant_Kirchoff(F, lambda_s, mu_s):
    """
    Strain energy density, St. Venant Kirchoff Material
    """

    E_ = 0.5*(F.T*F - Identity(get_dimension(F)))
    J = det(F)

    W = lambda_s / 2 * (tr(E_) ** 2) + mu_s * tr(E_*E_)

    return W


def W_Neo_Hookean(F, lambda_s, mu_s):
    """
    Strain energy density, Neo-Hookean Material (Compressible)
    """
    C1 = mu_s/2
    D1 = lambda_s/2
    C = F.T * F  # Right cauchy-green strain tensor
    I1 = tr(C)
    J = det(F)

    W = C1*(I1 - get_dimension(F) - 2*ufl.ln(J)) + D1*(J-1)**2

    return W

def W_Gent(F, mu_s, Jm):
    """
    Strain energy density, Compressible Gent Material
    As described in https://www.researchgate.net/profile/Aflah-Elouneg/publication/353259552_An_open-source_FEniCS-based_framework_for_hyperelastic_parameter_estimation_from_noisy_full-field_data_Application_to_heterogeneous_soft_tissues/links/6124e7c71e95fe241af14697/An-open-source-FEniCS-based-framework-for-hyperelastic-parameter-estimation-from-noisy-full-field-data-Application-to-heterogeneous-soft-tissues.pdf?origin=publication_detail
    "An open-source FEniCS-based framework for hyperelastic parameter estimation from noisy full-field data: Application to heterogeneous soft tissues"
    """

    B = F*F.T  # Left cauchy-green strain tensor
    I1 = tr(B)
    J = det(F)

    W = -(mu_s/2)*( Jm * ufl.ln( 1 - (I1-3)/Jm ) + 2*ufl.ln(J) )

    return W

def W_Mooney_Rivlin(F, lambda_s, mu_s, C01, C10, C11):
    """
    Strain energy density, Compressible Mooney-Rivlin Material
    following: https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid
    """
    J = det(F)
    K = lambda_s + 2*mu_s/3          # Compute bulk modulus from lambda and mu
    D1 = 2/K                         # D1 is calculated from the Bulk Modulus
    B = F*F.T                        # Left cauchy-green strain tensor
    I1 = tr(B)                       # 1st Invariant
    I2 = 0.5*(tr(B)**2-tr(B * B))    # 2nd invariant
    Ibar1 = (J**(-2/3))*I1           
    Ibar2 = (J**(-4/3))*I2

    # Strain energy density function for 3 term Compressible Mooney-Rivlin Model
    W = C01*(Ibar2-3) + C10*(Ibar1-3) + C11*(Ibar2-3)*(Ibar1-3) + (1/D1)*(J-1)**2   
    
    return W
