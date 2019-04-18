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
        raise argparse.ArgumentTypeError('Boolean value expected, not {}.'.format(boolean))


def restricted_float(x):
    """ Make sure that float is between 0.0 and 1.0

    Args:
        x (float): Input argument
    """
    x = float(x)
    if x <= 0.0 or x >= 1.0:
        raise argparse.ArgumentTypeError("{} not in range [0.0, 1.0]".format(x))
    return x


def parse():
    parser = argparse.ArgumentParser(description="TODO: Full description of the entire turtleFSI")

    # Define solver, numerics, and problem file
    parser.add_argument("-p", "--problem", type=str, default=None,
                        help="Name of problem file to solve. Could either be loced in the" +
                        " turtleFSI repository (TF_cfd, TF_csm, TF_fsi, turtle_demo) or it" +
                        " could be a problem file you have created locally.")

    # Set fluid, solid, and extrapolation
    parser.add_argument("-f", "--fluid", type=str, default=None,
                        choices=["fluid", "no-fluid"],
                        help="Turn off fluid and only solve the structure equation")
    parser.add_argument("-s", "--solid", type=str, default=None,
                        choices=["solid", "no-solid"],
                        help="Turn off solid and only solve the fluid equation")
    parser.add_argument("-e", "--extrapolation", type=str, default=None,
                        choices=["laplace", "linear", "biharmonic", "no-extrapolation"],
                        help="Set approach for extrapolating the deformation into the fluid" +
                            "domain")
    parser.add_argument("-et", "--extrapolation-sub-type", type=str, default=None,
                        choices=["constant", "small-constant", "volume", "bc1", "bc2"],
                        help="Set the sub type of the extrapolation method. TODO")

    # Solver settings
    parser.add_argument("--solver", type=str, default=None,
                        choices=["newtonsolver", "newtonsolver_naive"],
                        help="Chooce between a newtonsolver with multiple options or a" +
                             " naive solver withot any")

    # For theta scheme
    parser.add_argument("-theta", type=restricted_float, default=None,
                        help="Theta parameter for ex/imp/CN, TODO")

    # Verbose
    parser.add_argument("-v", "--verbose", type=str2bool, default=None,
                        help="Turn on/off verbose printing")

    # Set spatial and temporal resolution
    parser.add_argument("-dt", type=float, help="Set timestep", default=None)
    parser.add_argument("-T", type=float, help="Set end time", default=None)
    parser.add_argument("-p_deg", type=int, help="Set degree of pressure", default=None)
    parser.add_argument("-v_deg", type=int, help="Set degree of velocity", default=None)
    parser.add_argument("-d_deg", type=int, help="Set degree of deformation", default=None)
    parser.add_argument("-r", "--refiner", type=int, help="Mesh-refiner using built-in" +
                        " FEniCS method refine(Mesh)")

    # FIXME: parse unspecificed arguments
    args = parser.parse_args()

    return args
