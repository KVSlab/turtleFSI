# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

import pytest
import numpy as np
from dolfin import (UnitCubeMesh, MeshFunction, Measure, VectorFunctionSpace, Function, Expression,
                    interpolate, assemble, cells)

from turtleFSI.modules.nonnewtonian import get_fluid_rheology, modified_cross_model, EffectiveViscosity

CROSS_PARAMS = {"mu_0": 0.056, "mu_inf": 0.00345, "lmbda": 3.736, "n": 2.406, "a": 0.254}
MU_F_REGION_1 = 3.45e-3
MU_F_REGION_2 = 3.5e-2


def _setup(shear_rate_1, shear_rate_2):
    """
    Unit cube with two fluid regions and a solid: x < 0.5 is the non-Newtonian fluid region (id 1),
    0.5 < x < 0.75 is the Newtonian fluid region (id 2) and x > 0.75 is the solid (id 3).
    The velocity is a simple shear flow v = (shear_rate * y, 0, 0), so the shear rate is constant.
    """
    mesh = UnitCubeMesh(4, 4, 4)
    domains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for cell in cells(mesh):
        x = cell.midpoint().x()
        domains[cell] = 1 if x < 0.5 else (2 if x < 0.75 else 3)

    dx = Measure("dx", domain=mesh, subdomain_data=domains)
    dx_f = {0: dx(1), 1: dx(2)}

    V = VectorFunctionSpace(mesh, "CG", 1)
    v_ = {"n-1": interpolate(Expression(("g*x[1]", "0", "0"), g=shear_rate_1, degree=1), V),
          "n-2": interpolate(Expression(("g*x[1]", "0", "0"), g=shear_rate_2, degree=1), V)}
    d_ = {"n-1": Function(V), "n-2": Function(V)}

    fluid_properties = [{"dx_f_id": 1, "rho_f": 1.0E3, "mu_f": MU_F_REGION_1},
                        {"dx_f_id": 2, "rho_f": 1.0E3, "mu_f": MU_F_REGION_2, "nonnewtonian": False}]
    rheology = get_fluid_rheology(fluid_properties, nonnewtonian=True, cross_params=CROSS_PARAMS)
    mu_eff = EffectiveViscosity(mesh, domains, v_, d_, dx_f, fluid_properties, rheology)

    return mu_eff, dx


def _mean(mu_eff, dx, domain_id):
    return assemble(mu_eff * dx(domain_id)) / assemble(1.0 * dx(domain_id))


def test_get_fluid_rheology():
    fluid_properties = [{"mu_f": 1.0}, {"mu_f": 1.0, "nonnewtonian": False}, {"mu_f": 1.0, "cross_params": {**CROSS_PARAMS, "n": 1.0}}]

    rheology = get_fluid_rheology(fluid_properties, nonnewtonian=True, cross_params=CROSS_PARAMS)
    assert rheology[0] == CROSS_PARAMS
    assert rheology[1] is None
    assert rheology[2]["n"] == 1.0

    assert get_fluid_rheology(fluid_properties) == [None, None, None]

    with pytest.raises(ValueError):
        get_fluid_rheology([{"mu_f": 1.0}], nonnewtonian=True)
    with pytest.raises(ValueError):
        get_fluid_rheology([{"mu_f": 1.0}], nonnewtonian=True, cross_params={"mu_0": 0.056})


def test_effective_viscosity_update():
    shear_rate_1, shear_rate_2 = 50.0, 10.0
    mu_eff, dx = _setup(shear_rate_1, shear_rate_2)
    mu_1 = modified_cross_model(shear_rate_1, **CROSS_PARAMS)
    mu_2 = modified_cross_model(shear_rate_2, **CROSS_PARAMS)

    # Initial value: mu_f in the fluid regions, zero in the solid
    assert np.isclose(_mean(mu_eff, dx, 1), MU_F_REGION_1)
    assert np.isclose(_mean(mu_eff, dx, 2), MU_F_REGION_2)
    assert np.isclose(_mean(mu_eff, dx, 3), 0.0)

    # First update: first order in time (previous time step only)
    mu_eff.update()
    assert np.isclose(_mean(mu_eff, dx, 1), mu_1)
    assert np.isclose(_mean(mu_eff, dx, 2), MU_F_REGION_2)
    assert np.isclose(_mean(mu_eff, dx, 3), 0.0)

    # Following updates: second order extrapolation in time
    mu_eff.update()
    assert np.isclose(_mean(mu_eff, dx, 1), 1.5 * mu_1 - 0.5 * mu_2)
    assert np.isclose(_mean(mu_eff, dx, 2), MU_F_REGION_2)
    assert np.isclose(_mean(mu_eff, dx, 3), 0.0)


def test_effective_viscosity_clipping():
    # The extrapolation 1.5 * mu(1000) - 0.5 * mu(0) is below mu_inf and must be clipped to mu_inf
    mu_eff, dx = _setup(1000.0, 0.0)
    mu_eff.update()
    mu_eff.update()
    assert np.isclose(_mean(mu_eff, dx, 1), CROSS_PARAMS["mu_inf"])
    assert np.isclose(_mean(mu_eff, dx, 2), MU_F_REGION_2)
    assert mu_eff.vector().min() >= 0.0
    assert mu_eff.vector().max() <= CROSS_PARAMS["mu_0"]
