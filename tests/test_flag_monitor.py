import numpy as np
import pytest

from turtleFSI.cases.flag_monitor import quadratic_minimum


def test_quadratic_extrema():
    # Interior minimum, edge minimum, vertex minimum, constant, and saddle.
    coefficients = [[.13, -.4, -.6, 1, 0, 1],
                    [.09, -.6, 1, 1, 0, 0],
                    [1, -2, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, -1]]
    np.testing.assert_allclose(quadratic_minimum(coefficients), [0, 0, -1, 1, -1], atol=1e-14)


def test_flag_monitor_tracks_deformation():
    pytest.importorskip("dolfin")
    from dolfin import UnitSquareMesh, VectorFunctionSpace, interpolate, Expression
    from turtleFSI.cases.flag_monitor import FlagJacobianMonitor
    mesh = UnitSquareMesh(2, 2)
    displacement = interpolate(Expression(("0", "0"), degree=2), VectorFunctionSpace(mesh, "CG", 2))
    shape = interpolate(Expression((".2*x[0]", ".1*x[1]"), degree=1), VectorFunctionSpace(mesh, "CG", 1))
    monitor = FlagJacobianMonitor(mesh, displacement, shape)
    np.testing.assert_allclose(monitor.minimum(), [1, 1.32], atol=1e-12)
    displacement.interpolate(Expression(("-2*x[0]*x[0]", "0"), degree=2))
    np.testing.assert_allclose(monitor.minimum(), [1-4/1.2, (1-4/1.2)*1.32], atol=1e-11)
