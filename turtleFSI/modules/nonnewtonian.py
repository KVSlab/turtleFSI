# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Non-Newtonian fluid rheology (modified Cross model).

The effective viscosity is stored in a single DG1 field shared by all fluid regions. Each fluid
region uses its own rheology: the modified Cross model with its own parameters, or a constant
(Newtonian) viscosity mu_f. The field is updated explicitly once per time step from the velocity
and displacement of the previous time steps.
"""

import numpy as np
import dolfin as dl
from turtleFSI.modules.common import F_

CROSS_PARAMETERS = ("mu_0", "mu_inf", "lmbda", "n", "a")


def compute_shear_rate(v, d):
    """
    Shear rate sqrt(2 D:D) in the ALE frame, where D is the symmetric part of the
    spatial velocity gradient grad(v) * inv(F(d)).
    """
    l = dl.grad(v) * dl.inv(F_(d))
    D = 0.5 * (l + l.T)
    return dl.sqrt(2 * dl.inner(D, D))


def modified_cross_model(shear_rate, mu_0, mu_inf, lmbda, n, a):
    """
    Modified Cross model:

        mu = mu_inf + (mu_0 - mu_inf) / (1 + (lmbda * shear_rate)**n)**a

    Args:
        mu_0: Zero-shear viscosity [Pa.s]
        mu_inf: Infinite-shear viscosity [Pa.s]
        lmbda: Time constant [s]
        n, a: Dimensionless model exponents
    """
    return mu_inf + (mu_0 - mu_inf) / (1 + (lmbda * shear_rate)**n)**a


def get_fluid_rheology(fluid_properties, nonnewtonian=False, cross_params=None):
    """
    Get the rheology of each fluid region.

    A fluid region is non-Newtonian if fluid_properties[region]["nonnewtonian"] is True or, when
    that key is missing, if the global flag nonnewtonian is True. Its model parameters are taken
    from fluid_properties[region]["cross_params"] or, when that key is missing, from cross_params.

    Args:
        fluid_properties: List of dictionaries with the properties of each fluid region
        nonnewtonian: Default flag for the regions that do not set "nonnewtonian"
        cross_params: Default modified Cross parameters for the regions that do not set "cross_params"

    Returns:
        List with, for each fluid region, a dictionary with its modified Cross parameters,
        or None if the region is Newtonian.
    """
    rheology = []
    for region, properties in enumerate(fluid_properties):
        if not properties.get("nonnewtonian", nonnewtonian):
            rheology.append(None)
            continue

        params = properties.get("cross_params", cross_params)
        if params is None:
            raise ValueError("Fluid region {} is non-Newtonian, but no cross_params are given.".format(region))
        missing = [key for key in CROSS_PARAMETERS if key not in params]
        if missing:
            raise ValueError("cross_params of fluid region {} is missing {}.".format(region, missing))
        rheology.append({key: params[key] for key in CROSS_PARAMETERS})

    return rheology


class EffectiveViscosity(dl.Function):
    """
    Effective viscosity of the fluid, as one DG1 field for all fluid regions.

    It is used as the viscosity in the fluid stress term, and must be updated once per time step,
    before the solve (e.g. in pre_solve), by calling update(). The update is explicit and second
    order in time:

        mu^n = 3/2 * mu(v^{n-1}, d^{n-1}) - 1/2 * mu(v^{n-2}, d^{n-2})

    The first update (also after a restart) only uses the previous time step (first order).
    The new values are projected onto DG1 region by region, so every cell uses the model of its own
    fluid region, and then clipped to [mu_inf, mu_0] of that region. In Newtonian regions the field
    is equal to mu_f. In the solid domain the field is zero and is never used.
    """

    def __init__(self, mesh, domains, v_, d_, dx_f, fluid_properties, rheology):
        """
        Args:
            mesh: Mesh of the whole domain (fluid and solid)
            domains: MeshFunction with the domain markers
            v_, d_: Dictionaries with the velocity and displacement at the time steps "n-1" and "n-2"
            dx_f: Dictionary with the measure of each fluid region
            fluid_properties: List of dictionaries with the properties of each fluid region
            rheology: Output of get_fluid_rheology
        """
        super().__init__(dl.FunctionSpace(mesh, "DG", 1))
        self.rename("Viscosity", "mu")

        V = self.function_space()
        w = dl.TestFunction(V)
        dx = dl.Measure("dx", domain=mesh, subdomain_data=domains)

        # The mass matrix is assembled on the whole mesh (fluid and solid) once
        self._mass = dl.assemble(dl.TrialFunction(V) * w * dx)
        self._solver = dl.KrylovSolver("cg", "jacobi")
        self._solver.set_operator(self._mass)
        self._solver.parameters["relative_tolerance"] = 1e-12
        self._solver.parameters["absolute_tolerance"] = 1e-15
        self._solver.parameters["nonzero_initial_guess"] = True

        # Weights of the time extrapolation, set in update()
        self._weight_1 = dl.Constant(1.0)
        self._weight_2 = dl.Constant(0.0)
        self._first_update = True

        # Right-hand sides of the projection, region by region
        rhs, initial, lower, upper = 0, 0, 0, 0
        for region, params in enumerate(rheology):
            mu_f = dl.Constant(fluid_properties[region]["mu_f"])
            if params is None:
                mu_new, mu_min, mu_max = mu_f, mu_f, mu_f
            else:
                mu_1 = modified_cross_model(compute_shear_rate(v_["n-1"], d_["n-1"]), **params)
                mu_2 = modified_cross_model(compute_shear_rate(v_["n-2"], d_["n-2"]), **params)
                mu_new = self._weight_1 * mu_1 + self._weight_2 * mu_2
                mu_min, mu_max = dl.Constant(params["mu_inf"]), dl.Constant(params["mu_0"])

            rhs += mu_new * w * dx_f[region]
            initial += mu_f * w * dx_f[region]
            lower += mu_min * w * dx_f[region]
            upper += mu_max * w * dx_f[region]

        self._rhs = rhs
        self._rhs_vector = None

        # Bounds for the clipping, per degree of freedom
        self._lower = self._project(lower).get_local()
        self._upper = self._project(upper).get_local()

        # Initial value: mu_f in every fluid region
        self.vector().set_local(self._project(initial).get_local())
        self.vector().apply("insert")

    def _project(self, form):
        """L2 projection onto DG1 of a linear form of the type value * w * dx_f[region]."""
        b = dl.assemble(form)
        x = self.vector().copy()
        x.zero()
        self._solver.solve(x, b)
        return x

    def update(self):
        """Update the effective viscosity from the solution at the previous time steps."""
        if self._first_update:
            self._weight_1.assign(dl.Constant(1.0))
            self._weight_2.assign(dl.Constant(0.0))
            self._first_update = False
        else:
            self._weight_1.assign(dl.Constant(1.5))
            self._weight_2.assign(dl.Constant(-0.5))

        if self._rhs_vector is None:
            self._rhs_vector = dl.assemble(self._rhs)
        else:
            dl.assemble(self._rhs, tensor=self._rhs_vector)
        self._solver.solve(self.vector(), self._rhs_vector)

        # Clip to the bounds of each fluid region
        values = self.vector().get_local()
        np.clip(values, self._lower, self._upper, out=values)
        self.vector().set_local(values)
        self.vector().apply("insert")
