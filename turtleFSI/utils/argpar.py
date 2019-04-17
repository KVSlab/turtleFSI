# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

import argparse
from argparse import RawTextHelpFormatter


def parse():
    parser = argparse.ArgumentParser(description="TODO")

    group = parser.add_argument_group('Parameters')

    # Define solver, numerics, and problem file
    group.add_argument("-p", "--problem", help="Name of problem file to solver",
                       default="TF_fsi")
    group.add_argument("-f", "--fluid-variation", default="thetaCN",
                       choices=["adamCN", "nofluid", "thetaCN", "thetaCN2"],
                       help="Name of file defining variational form for fluid",
                       default="thetaCN")
    group.add_argument("-s", "--solidvar",
                       choices=["CN_mixed", "csm", "implicit", "surfint", "thetaCN",
                                "thetaCN2"],
                       help="Set variationalform for solid", default="thetaCN")
    group.add_argument("-extravar", help="Set variationalform for extrapolation", default="alfa")
    group.add_argument("-solver", help="Set type of solver to be used", default="newtonsolver")

    # For theta scheme
    group.add_argument("-theta", type=float,  help="Theta Parameter for ex/imp/CN", default=0.5)

    # Meshing lifting operator
    group.add_argument("-extype", help="Extrapolation constant (const, smallc, det)", default="const")
    group.add_argument("-bitype", help="Different BC for extrapol (bc1, bc2)", default="bc1")

    # Solver settings
    atol = 1e-7
    rtol = 1e-7
    max_it = 50
    lmbda = 1.0

    # Set spatial and temporal resolution
    group.add_argument("-dt", type=float, help="Set timestep", default=None)
    group.add_argument("-T", type=float, help="Set end time", default=None)
    group.add_argument("-p_deg", type=int, help="Set degree of pressure", default=1)
    group.add_argument("-v_deg", type=int, help="Set degree of velocity", default=2)
    group.add_argument("-d_deg", type=int, help="Set degree of deformation", default=2)
    group.add_argument("-r", "--refiner", type=int, help="Mesh-refiner using built-in FEniCS method refine(Mesh)")

    # Postprocessing
    group.add_argument("-tag", help="tag name for storing file", default="tag")

    # FIXME: parse unspecificed arguments
    args = parser.parse_args()

    # TODO: Check if problem file exists

    return args
