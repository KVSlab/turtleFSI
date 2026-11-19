from types import SimpleNamespace
import numpy as np
import pytest
from turtleFSI.cases.validate_flag_rom import evaluation_windows


def options(**updates):
    args = dict(train_end=12., prediction_only=True, prediction_start=12.01,
                prediction_end=13.01, reproduction_start=2.01)
    args.update(updates)
    return SimpleNamespace(**args)


def test_training_cutoff_need_not_be_saved_time():
    times = .01 + .05*np.arange(601)
    assert evaluation_windows(times, options()) == (("prediction", 240, 260),)
    assert np.count_nonzero(times <= 12.) == 240
    assert times[239] < 12 < times[240]


def test_original_two_window_protocol():
    times = .01 + .05*np.arange(601)
    args = options(train_end=3.01, prediction_only=False, prediction_start=None,
                   prediction_end=3.51)
    assert evaluation_windows(times, args) == (("reproduction", 40, 60), ("prediction", 60, 70))


def test_reproduction_inside_training_window():
    args = options(train_end=14., prediction_only=False, prediction_start=None,
                   reproduction_only=True, reproduction_start=10.01, prediction_end=13.96)
    assert evaluation_windows(.01 + .05*np.arange(601), args) == (("reproduction", 200, 279),)
    args.prediction_end = 14.01
    with pytest.raises(ValueError):
        evaluation_windows(.01 + .05*np.arange(601), args)


@pytest.mark.parametrize('updates', [dict(prediction_start=11.96),
    dict(prediction_start=12.), dict(prediction_end=12.01), dict(train_end=float('nan')),
    dict(prediction_only=False)])
def test_invalid_windows_rejected(updates):
    with pytest.raises(ValueError):
        evaluation_windows(.01 + .05*np.arange(601), options(**updates))
