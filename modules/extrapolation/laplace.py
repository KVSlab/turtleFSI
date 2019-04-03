from dolfin import inner, grad

"""Implementation of the laplace mesh lifting operator.

In FSI simulations, the fluid mesh cells must conform to the deformation of the solid domain. If the fluid domain cells
are poorly constructed, or fluid domain is acted upon by medium/large deformations, the FSI simulations might diverge
due to crossing fluid mesh cells.

One strategy to avoid entanglement is to solve an additional equation for "stiffening" the fluid mesh, often
refered to as a mesh lifting operator. The overall goal of a mesh lifting operator, is 
smoothing out the solid deformation of the fluid throughout the fluid domain, and not around the interface between
fluid and solid domain.



"""

def extrapolate_setup(F_fluid_nonlinear, d_, phi, dx_f, **monolithic):
    """

    Args:
        F_fluid_nonlinear: Non-linear part of the fluid variational form.
        d_: Deformation function at time-step "n"
        phi: TestFunction
        dx_f: Fluid domain of the total computational domain.
        **monolithic:

    Returns:
        Non-linear fluid variational formulation with mesh lifting operator.

    """
    F_extrapolate = inner(grad(d_["n"]), grad(phi))*dx_f
    F_fluid_nonlinear += F_extrapolate

    return dict(F_fluid_nonlinear=F_fluid_nonlinear)