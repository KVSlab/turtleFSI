"""Controller and rejected-trial transaction tests without a FEniCS solve."""
from types import SimpleNamespace

import numpy as np
import pytest

from turtleFSI.modules.adaptive import AdaptiveController, advance


def test_output_grid_survives_reductions_and_growth():
    controller = AdaptiveController(.01, .0001, .01, .01, .05, .21)
    t, outputs, steps = 0., [], []
    controller.reject(.01)
    while t < controller.end - 1e-12:
        dt = controller.step(t)
        assert dt > 0
        steps.append(dt)
        t += dt
        if controller.accept(t, 3):
            outputs.append(t)
    np.testing.assert_allclose(outputs, [.01, .06, .11, .16, .21], atol=1e-12)
    assert len(set(steps)) > 1
    assert t == pytest.approx(.21)


class Vector:
    def __init__(self, value):
        self.value = np.array([value], dtype=float)

    def get_local(self):
        return self.value.copy()

    def set_local(self, value):
        self.value[:] = value

    def apply(self, mode):
        pass

    def zero(self):
        self.value[:] = 0


class Field:
    def __init__(self, value):
        self.v = Vector(value)

    def vector(self):
        return self.v


def namespace(tmp_path, solver, monitor):
    inlet = SimpleNamespace(t=0.)
    k = SimpleNamespace(value=.01)
    k.assign = lambda v: setattr(k, 'value', v)
    def pre_solve(t, **kwargs):
        inlet.t = t
    return dict(t=0., dt=.01, k=k, inlet=inlet, counter=7,
                dvp_={key: Field(i) for i, key in enumerate(['n', 'n-1', 'n-2'])},
                dvp_res=Field(0), results_folder=tmp_path, pre_solve=pre_solve,
                newtonsolver=solver, jacobian_monitor=monitor, b=None)


@pytest.mark.parametrize('failure', ['exception', 'iteration_limit', 'inversion'])
def test_rejected_trial_restores_history_and_inlet(tmp_path, failure):
    calls = []
    def solver(dvp_, dt, t, recompute_tstep, **kwargs):
        assert recompute_tstep == 1
        np.testing.assert_array_equal([dvp_[k].vector().value[0] for k in ['n','n-1','n-2']], [0,1,2])
        calls.append((t,dt))
        for field in dvp_.values():
            field.vector().set_local([99])
        if len(calls) == 1 and failure == 'exception':
            raise RuntimeError('test divergence')
        return dict(newton_iterations=4, newton_converged=not (len(calls)==1 and failure=='iteration_limit'))
    monitor = SimpleNamespace(minimum=lambda: (-1, -1) if len(calls)==1 and failure=='inversion' else (1,1))
    ns = namespace(tmp_path, solver, monitor)
    controller = AdaptiveController(.01,.001,.01,.01,.05,1)
    assert not advance(ns,controller)
    assert calls == [(0.01,0.01),(0.005,0.005)]
    assert ns['t'] == ns['inlet'].t == ns['k'].value == .005
    assert ns['counter'] == 7
    assert controller.output_index == 0
    assert 'False' in (tmp_path/'adaptive_steps.csv').read_text()


def test_minimum_failure_restores_last_accepted_state(tmp_path):
    def solver(dvp_, **kwargs):
        dvp_['n'].vector().set_local([float('nan')])
        raise RuntimeError('failure')
    ns = namespace(tmp_path, solver, None)
    controller = AdaptiveController(.01,.005,.01,.01,.05,1)
    with pytest.raises(RuntimeError, match='minimum dt'):
        advance(ns,controller)
    assert ns['t'] == ns['inlet'].t == 0
    assert ns['dt'] == ns['k'].value == .01
    np.testing.assert_array_equal(ns['dvp_']['n'].vector().get_local(),[0])


@pytest.mark.parametrize('values', [(.01,.02,.01,.01,.05,1),(.01,.001,.01,0,.05,1),(.01,.001,.01,.01,float('nan'),1)])
def test_invalid_settings(values):
    with pytest.raises(ValueError):
        AdaptiveController(*values)


@pytest.mark.parametrize('override', [dict(theta=.5), dict(restart_folder='old/Checkpoint'),
                                     dict(adaptive_dt_max=0), dict(problem='TF_fsi')])
def test_unsupported_run_rejected_before_output(override):
    pytest.importorskip('dolfin')
    from turtleFSI.modules.adaptive import configure
    ns = dict(problem='TF_fsi_reference', theta=1., d_deg=2,
              extrapolation='laplace', extrapolation_sub_type='paper',
              solver='newtonsolver', fluid='fluid', solid='solid',
              restart_folder=None, t=0., dt=.01, adaptive_dt_min=.001,
              adaptive_dt_max=None, adaptive_output_start=.01,
              adaptive_output_interval=.05, T=1.)
    ns.update(override)
    with pytest.raises(ValueError):
        configure(ns)


def test_real_newton_reports_exhaustion_and_updated_residual():
    pytest.importorskip('dolfin')
    from dolfin import (UnitSquareMesh, FunctionSpace, Function, TestFunction,
                        derivative, dx, assemble, LUSolver)
    from turtleFSI.modules.newtonsolver import newtonsolver
    mesh = UnitSquareMesh(1, 1)
    space = FunctionSpace(mesh, 'R', 0)
    u = Function(space)
    u.vector()[:] = 1
    F = (u*u-2)*TestFunction(space)*dx
    J = derivative(F, u)
    A = assemble(J)
    arguments = dict(F=F, J=J, J_nonlinear=J, A_pre=A, A=A, b=None,
                     bcs=[], lmbda=1., recompute=1, recompute_tstep=1,
                     compiler_parameters={}, dvp_={'n':u}, up_sol=LUSolver(),
                     dvp_res=Function(space), rtol=1e-7, atol=1e-7, max_it=1,
                     counter=0, first_step_num=0, verbose=False,
                     adaptive_time_step=True)
    result = newtonsolver(**arguments)
    assert not result['newton_converged']
    assert result['newton_residual'] == pytest.approx(.25)
    arguments.update(result, max_it=10)
    result = newtonsolver(**arguments)
    assert result['newton_converged']
    assert u.vector().get_local()[0] == pytest.approx(2**.5)
