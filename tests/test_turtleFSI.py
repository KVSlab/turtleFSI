# Co (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

import pytest
import numpy as np
from os import system


def compare(one, two):
    if one < 1e-7 or two < 1e-7:
        return '{:0.5e}'.format(one) == '{:0.5e}'.format(two)
    else:
        return '{:0.6e}'.format(one) == '{:0.6e}'.format(two)


def test_cfd():
    cmd = ("turtleFSI --problem TF_cfd -dt 0.01 -T 0.05 --verbose True" +
           " --new-arguments folder=tmp")
    d = system(cmd)

    drag = np.loadtxt("tmp/Drag.txt")[-1]
    lift = np.loadtxt("tmp/Lift.txt")[-1]
    drag_reference = 2.5637554331614054
    lift_reference = -0.02078995609237899

    assert compare(drag, drag_reference)
    assert compare(lift, lift_reference)


def test_csm():
    cmd = ("turtleFSI --problem TF_csm -dt 0.01 -T 0.05 --verbose True" +
           " --new-arguments folder=tmp")
    d = system(cmd)

    distance_x = np.loadtxt("tmp/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/dis_y.txt")[-1]
    distance_x_reference = -6.13487990897633e-06
    distance_y_reference = -3.9398599897576816e-05

    assert compare(distance_x, distance_x_reference)
    assert compare(distance_y, distance_y_reference)


def test_fsi():
    cmd = ("turtleFSI --problem TF_fsi -dt 0.01 -T 0.05 --verbose True --theta 0.51" +
           " --new-arguments folder=tmp")
    d = system(cmd)

    drag = np.loadtxt("tmp/Drag.txt")[-1]
    lift = np.loadtxt("tmp/Lift.txt")[-1]
    distance_x = np.loadtxt("tmp/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/dis_y.txt")[-1]
    distance_x_reference = -3.0193475393178104e-06
    distance_y_reference = -2.6614039656203487e-08
    drag_reference = 2.4729291261050355
    lift_reference = -0.003952155852968209

    assert compare(distance_x, distance_x_reference)
    assert compare(distance_y, distance_y_reference)
    assert compare(drag, drag_reference)
    assert compare(lift, lift_reference)


@pytest.mark.parametrize("extrapolation_sub_type", ["volume", "volume_change",
                                                    "constant", "small_constant"])
def test_laplace(extrapolation_sub_type):
    cmd = ("turtleFSI --problem TF_fsi -dt 0.01 -T 0.05 --verbose True --theta 0.51"
           " --new-arguments folder=tmp")
    d = system(cmd)

    drag = np.loadtxt("tmp/Drag.txt")[-1]
    lift = np.loadtxt("tmp/Lift.txt")[-1]
    distance_x = np.loadtxt("tmp/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/dis_y.txt")[-1]
    distance_x_reference = -3.0193475393178104e-06
    distance_y_reference = -2.6614039656203487e-08
    drag_reference = 2.4729291261050355
    lift_reference = -0.003952155852968209

    assert compare(distance_x, distance_x_reference)
    assert compare(distance_y, distance_y_reference)
    assert compare(drag, drag_reference)
    assert compare(lift, lift_reference)


@pytest.mark.parametrize("extrapolation_sub_type", ["bc1", "bc2"])
def test_biharmonic(extrapolation_sub_type):
    cmd = ("turtleFSI --problem TF_fsi -dt 0.01 -T 0.05 --verbose True --theta 0.51" +
           "--extrapolation biharmonic --extrapolation_sub_type" +
           " --new-arguments folder=tmp {}".format(extrapolation_sub_type))
    d = system(cmd)

    drag = np.loadtxt("tmp/Drag.txt")[-1]
    lift = np.loadtxt("tmp/Lift.txt")[-1]
    distance_x = np.loadtxt("tmp/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/dis_y.txt")[-1]
    distance_x_reference = -3.0193475393178104e-06
    distance_y_reference = -2.6614039656203487e-08
    drag_reference = 2.4729291261050355
    lift_reference = -0.003952155852968209

    assert compare(distance_x, distance_x_reference)
    assert compare(distance_y, distance_y_reference)
    assert compare(drag, drag_reference)
    assert compare(lift, lift_reference)


def test_elastic():
    cmd = ("turtleFSI --problem TF_fsi -dt 0.01 -T 0.05 --verbose True --theta 0.51" +
           " --extrapolation elastic --new-arguments folder=tmp")
    d = system(cmd)

    drag = np.loadtxt("tmp/Drag.txt")[-1]
    lift = np.loadtxt("tmp/Lift.txt")[-1]
    distance_x = np.loadtxt("tmp/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/dis_y.txt")[-1]
    distance_x_reference = -3.0193475393178104e-06
    distance_y_reference = -2.6614039656203487e-08
    drag_reference = 2.4729291261050355
    lift_reference = -0.003952155852968209

    assert compare(distance_x, distance_x_reference)
    assert compare(distance_y, distance_y_reference)
    assert compare(drag, drag_reference)
    assert compare(lift, lift_reference)
