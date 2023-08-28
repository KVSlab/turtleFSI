# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

import pytest
import numpy as np
from pathlib import Path
import subprocess


def test_cfd():
    cmd = ("turtleFSI --problem TF_cfd -dt 0.01 -T 0.05 --verbose True" +
           " --folder tmp --sub-folder 1")
    subprocess.run(cmd, shell=True, check=True)

    drag = np.loadtxt(Path.cwd().joinpath("tmp/1/Drag.txt"))[-1]
    lift = np.loadtxt(Path.cwd().joinpath("tmp/1/Lift.txt"))[-1]
    drag_reference = 4.503203576965564
    lift_reference = -0.03790359084395478

    assert np.isclose(drag, drag_reference)
    assert np.isclose(lift, lift_reference)


def test_csm():
    cmd = ("turtleFSI --problem TF_csm -dt 0.01 -T 0.05 --verbose True" +
           " --folder tmp --sub-folder 2")
    subprocess.run(cmd, shell=True, check=True)

    distance_x = np.loadtxt("tmp/2/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/2/dis_y.txt")[-1]
    distance_x_reference = -3.313014369394714527e-05
    distance_y_reference = -3.770127311444726199e-03

    assert np.isclose(distance_x, distance_x_reference)
    assert np.isclose(distance_y, distance_y_reference)


@pytest.mark.parametrize("num_p", [1, 2])
def test_fsi(num_p):
    cmd = ("mpirun -np {} turtleFSI --problem TF_fsi -dt 0.01 -T 0.05 --verbose True" +
           " --theta 0.51 --folder tmp --sub-folder 3")
    subprocess.run(cmd.format(num_p), shell=True, check=True)

    drag = np.loadtxt("tmp/3/Drag.txt")[-1]
    lift = np.loadtxt("tmp/3/Lift.txt")[-1]
    distance_x = np.loadtxt("tmp/3/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/3/dis_y.txt")[-1]
    distance_x_reference = -6.896013956339182e-06
    distance_y_reference = 1.876355330341896e-09
    drag_reference = 4.407481239804155
    lift_reference = -0.005404703556977697

    assert np.isclose(distance_x, distance_x_reference)
    assert np.isclose(distance_y, distance_y_reference)
    assert np.isclose(drag, drag_reference)
    assert np.isclose(lift, lift_reference)


@pytest.mark.parametrize("extrapolation_sub_type", ["volume", "volume_change",
                                                    "constant", "small_constant"])
def test_laplace(extrapolation_sub_type):
    cmd = ("turtleFSI --problem TF_fsi -dt 0.01 -T 0.05 --verbose True --theta 0.51" +
           " --extrapolation laplace --extrapolation-sub-type {}" +
           " --folder tmp --sub-folder 4")
    subprocess.run(cmd.format(extrapolation_sub_type), shell=True, check=True)
    drag = np.loadtxt("tmp/4/Drag.txt")[-1]
    lift = np.loadtxt("tmp/4/Lift.txt")[-1]
    distance_x = np.loadtxt("tmp/4/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/4/dis_y.txt")[-1]
    distance_x_reference = -6.896013956339182e-06
    distance_y_reference = 1.876355330341896e-09
    drag_reference = 4.407481239804155
    lift_reference = -0.005404703556977697

    assert np.isclose(distance_x, distance_x_reference)
    assert np.isclose(distance_y, distance_y_reference)
    assert np.isclose(drag, drag_reference)
    assert np.isclose(lift, lift_reference, rtol=1e-4)


@pytest.mark.parametrize("extrapolation_sub_type", 
                        ["constrained_disp", "constrained_disp_vel"])
def test_biharmonic(extrapolation_sub_type):
    cmd = ("turtleFSI --problem TF_fsi -dt 0.01 -T 0.05 --verbose True --theta 0.51" +
           " --extrapolation biharmonic --extrapolation-sub-type {}" + 
           " --folder tmp --sub-folder 5")
    subprocess.run(cmd.format(extrapolation_sub_type), shell=True, check=True)

    drag = np.loadtxt("tmp/5/Drag.txt")[-1]
    lift = np.loadtxt("tmp/5/Lift.txt")[-1]
    distance_x = np.loadtxt("tmp/5/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/5/dis_y.txt")[-1]
    distance_x_reference = -6.896013956339182e-06
    distance_y_reference = 1.876355330341896e-09
    drag_reference = 4.407481239804155
    lift_reference = -0.005404703556977697

    assert np.isclose(distance_x, distance_x_reference)
    assert np.isclose(distance_y, distance_y_reference)
    assert np.isclose(drag, drag_reference)
    assert np.isclose(lift, lift_reference)


def test_elastic():
    cmd = ("turtleFSI --problem TF_fsi -dt 0.01 -T 0.05 --verbose True --theta 0.51" +
           " -e elastic -et constant --folder tmp --sub-folder 6")
    subprocess.run(cmd, shell=True, check=True)

    drag = np.loadtxt("tmp/6/Drag.txt")[-1]
    lift = np.loadtxt("tmp/6/Lift.txt")[-1]
    distance_x = np.loadtxt("tmp/6/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/6/dis_y.txt")[-1]
    distance_x_reference = -6.896144755254494e-06
    distance_y_reference = 1.868651990487361e-09
    drag_reference = 4.407488867909029
    lift_reference = -0.005404616050528832

    assert np.isclose(distance_x, distance_x_reference)
    assert np.isclose(distance_y, distance_y_reference)
    assert np.isclose(drag, drag_reference)
    assert np.isclose(lift, lift_reference)


def test_save_deg2():
    """simple test if the save_deg 2 works"""
    cmd = ("turtleFSI --problem TF_fsi -dt 0.01 -T 0.05 --theta 0.51 --save-deg 2" +
           " --save-step 1 --folder tmp --sub-folder 7")
    subprocess.run(cmd, shell=True, check=True)

    d_path = Path.cwd().joinpath("tmp/7/Visualization/displacement.xdmf")
    v_path = Path.cwd().joinpath("tmp/7/Visualization/velocity.xdmf")
    p_path = Path.cwd().joinpath("tmp/7/Visualization/pressure.xdmf")

    assert d_path.is_file()
    assert v_path.is_file()
    assert p_path.is_file()