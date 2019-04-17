# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Parsing provided on the command line. Although the list of parameters is rather extensive,
the user is free to add new once on the commandline. These will stil be parsed and added.
For instance a problem specific parameter like inlet flow rate could also be overwritten
from the command line options. Also, there are options like setting the absolute and
relative tolerance which is not added here, but with default values from __init__.py in
problems. For instane to overwrite the absolute tolerance simply add '--atol 1e-8' to the
command line call, or set it explicitly in the problem file.
"""

import argparse
from argparse import RawTextHelpFormatter


def str2bool(boolean):
    """Convert a string to boolean.

    Args:
        boolean (str): Input string.

    Returns:
        return (bool): Converted string.
    """
    if boolean.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif boolean.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def restricted_float(x):
    """ Make sure that float is between 0.0 and 1.0

    Args:
        x (float): Input argument
    """
    x = float(x)
    if x <= 0.0 or x >= 1.0:
        raise ArgumentTypeError("{} not in range [0.0, 1.0]".format(x))
    return x


def parse():
    parser = argparse.ArgumentParser(description="TODO: Full description of the entire turtleFSI")

    # Define solver, numerics, and problem file
    parser.add_argument("-p", "--problem", help="Name of problem file to solver",
                       default="TF_fsi")

    # Turn on / off fluid, solid, and extrapolation
    parser.add_argument("-nf", "--no-fluid", type=str2bool, default=False,
                       help="Turn off fluid and only solve the structure equation")
    parser.add_argument("-ns", "--no-solid", type=str2bool, default=False,
                       help="Turn off solid and only solve the fluid equation")
    parser.add_argument("-ne", "--no-extrapolation", type=str2bool, default=False,
                       help="Turn off extrapolation of the mesh movement into the fluid" +
                            "domain")

    # Set extrapolation
    parser.add_argument("-e", "--extrapolation", type=str, default="laplace",
                       choices=["laplace", "biharmonic", "linear-elasticity"],
                       help="Set approach for extrapolating the deformation into the fluid" +
                             "domain")
    ## TODO: Add subtypes
    # Meshing lifting operator
    parser.add_argument("-extype", help="Extrapolation constant (const, smallc, det)", default="const")
    parser.add_argument("-bitype", help="Different BC for extrapol (bc1, bc2)", default="bc1")

    # Solver settings
    parser.add_argument("-solver", help="Set type of solver to be used", default="newtonsolver")

    # For theta scheme
    parser.add_argument("-theta", type=float,  help="Theta Parameter for ex/imp/CN", default=0.5)

    # Set spatial and temporal resolution
    parser.add_argument("-dt", type=float, help="Set timestep", default=None)
    parser.add_argument("-T", type=float, help="Set end time", default=None)
    parser.add_argument("-p_deg", type=int, help="Set degree of pressure", default=1)
    parser.add_argument("-v_deg", type=int, help="Set degree of velocity", default=2)
    parser.add_argument("-d_deg", type=int, help="Set degree of deformation", default=2)
    parser.add_argument("-r", "--refiner", type=int, help="Mesh-refiner using built-in FEniCS method refine(Mesh)")

    # Postprocessing
    parser.add_argument("-tag", help="tag name for storing file", default="tag")

    # FIXME: parse unspecificed arguments
    args = parser.parse_args()

    # TODO: Check if problem file exists

    return args
