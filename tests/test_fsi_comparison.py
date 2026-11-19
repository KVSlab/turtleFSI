import json
from pathlib import Path
import sys

import numpy as np
import pytest
from turtleFSI.cases.plots.compare_fsi import load_inputs, relative_errors


def make_case(tmp_path, times=(2., 2.1, 2.3)):
    run = tmp_path/'new-flag'
    (run/'Checkpoint').mkdir(parents=True)
    (run/'Checkpoint/default_variables.json').write_text('{}')
    path = run/'analysis/validation_rollout/fenics_pod_galerkin_rollout.npz'
    path.parent.mkdir(parents=True)
    np.savez(path, reference=np.ones((6,len(times))), full=np.ones((6,len(times))), time=times)
    return run, path


def test_discovery_and_arbitrary_times(tmp_path):
    run, path = make_case(tmp_path)
    loaded = load_inputs(run)
    assert loaded[1] == path
    np.testing.assert_array_equal(loaded[5], [2.,2.1,2.3])
    assert loaded[6] == {}


def test_explicit_rollout_outside_run(tmp_path):
    run, path = make_case(tmp_path)
    destination = tmp_path/'external.npz'
    path.rename(destination)
    assert load_inputs(run, destination)[1] == destination


def test_ambiguous_rollouts_require_selection(tmp_path):
    run, path = make_case(tmp_path)
    second = run/'analysis/rollout'/path.name
    second.parent.mkdir()
    second.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='Supply --rollout'):
        load_inputs(run)


@pytest.mark.parametrize('times', [[1.], [1.,1.], [2.,1.], [1.,np.nan]])
def test_reject_bad_time_arrays(tmp_path, times):
    run, _ = make_case(tmp_path, times)
    with pytest.raises(ValueError, match='strictly increasing'):
        load_inputs(run)


@pytest.mark.parametrize('rom', [np.zeros((5,3)), np.full((6,3), np.nan)])
def test_reject_invalid_state_arrays(tmp_path, rom):
    run, path = make_case(tmp_path)
    np.savez(path, reference=np.ones((6,3)), full=rom, time=[1.,2.,3.])
    with pytest.raises(ValueError):
        load_inputs(run)


def test_zero_reference_norm_and_roundoff():
    values = relative_errors(np.array([4.,0.,1.,-1e-20]), np.array([16.,0.,0.,1.]))
    assert values[0] == .5
    assert np.isnan(values[1:3]).all()
    assert values[3] == 0


def test_flag_validation_routes_to_physical_plots(tmp_path, monkeypatch):
    from turtleFSI.rom import pod_galerkin
    run = tmp_path/'flag'
    snapshots = run/'ROM/dvp_snapshots.npz'
    snapshots.parent.mkdir(parents=True)
    data = np.arange(30.).reshape(6,5)
    np.savez(snapshots, dvp=data, time=np.arange(5.), displacement_dofs=[0,1],
             velocity_dofs=[2,3], pressure_dofs=[4,5])
    calls = []
    def fake_run(module, arguments):
        calls.append((module, arguments))
        if module.endswith('run_fenics_pod_galerkin_rollout'):
            out = Path(arguments[arguments.index('--output-dir')+1]); out.mkdir(parents=True)
            np.savez(out/'fenics_pod_galerkin_rollout.npz', reference=data[:,3:], full=data[:,3:], time=[3.,4.])
            (out/'fenics_pod_galerkin_rollout_summary.json').write_text(json.dumps({'start_index':3}))
    monkeypatch.setattr(pod_galerkin, 'run_module', fake_run)
    monkeypatch.setattr(sys, 'argv', ['pod', '--run-dir', str(run), '--problem', 'TF_fsi',
                                    '--action', 'validate-rollout', '--output-dir', str(run/'analysis')])
    pod_galerkin.main()
    assert calls[-1][0] == 'turtleFSI.cases.plots.compare_fsi'
    args = calls[-1][1]
    assert args[args.index('--run-dir')+1] == str(run)
