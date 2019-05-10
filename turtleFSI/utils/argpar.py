# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Parsing provided on the command line. Although the list of parameters is rather extensive,
the user is free to add new ones on the commandline. These will still be parsed and added.
For instance, the variable "folder", used to specified the results folder, can be set by
simply adding '--new-argument folder=TF_fsi_results' to the command line call or
by adding (folder=TF_fsi_results) as a variable in the "set_problem_parameters" function
of the problem file.
NOTE: Any command line argument will overwrite the variable value given in the problem
file "set_problem_parameters". Any variable set in the problem file "set_problem_parameters"
will overwrite the default value defined in the problems/__init__.py file.
"""

import argparse
import string


class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            try:
                # Int
                if set(v).issubset(set(string.digits+"-")):
                    my_dict[k] = int(v)

                # Float
                elif set(v).issubset(set(string.digits+".eE+-")):
                    my_dict[k] = float(v)

                # Boolean
                elif v.lower in ["true", "false"]:
                    my_dict[k] = bool(v.capitalize())

                # String
                else:
                    my_dict[k] = v
            except:
                # String
                my_dict[k] = v

        setattr(namespace, self.dest, my_dict)


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

    parser = argparse.ArgumentParser(description=(
        "turtleFSI is an open source Fluid-Structure Interaction (FSI) solver written in Python "
        + "and built upon the FEniCS finite element library. The purpose of turtleFSI is to "
        + "provide a user friendly and numerically robust monolithic FSI solver able to handle "
        + "problems characterized by large deformation. turtleFSI benefites from the state-of-the-art "
        + "parrallel computing features available from the FEniCS library and can be executed with "
        + "MPI on large computing resources."))

    # Define solver, numerics, and problem file
    parser.add_argument("-p", "--problem", type=str, default="turtle_demo", metavar="Problem",
                        help="Name of problem file to solve. Could either be located in the" +
                        " turtleFSI repository (TF_cfd, TF_csm, TF_fsi, turtle_demo) or it" +
                        " could be a problem file you have created locally.")
    parser.add_argument("-t", "--theta", type=restricted_float, default=None,
                        metavar="Theta",
                        help="Setting temporal integration. " +
                        "(theta=0 : first order explicit forward Euler scheme) " +
                        "(theta=1 : first order implicit backward Euler scheme) " +
                        "(theta=0.5 : second-order Crank-Nicolson scheme) " +
                        "(theta=0.5+dt : gives a better long-term numerical stability" +
                        " while keeping the second order accuracy of the Crank-Nicolson scheme)")

    # Set fluid, solid, and extrapolation
    parser.add_argument("-f", "--fluid", type=str, default=None,
                        choices=["fluid", "no-fluid"], metavar="Fluid",
                        help="Turn off fluid and only solve the solid problem")
    parser.add_argument("-s", "--solid", type=str, default=None, metavar="Solid",
                        choices=["solid", "no-solid"],
                        help="Turn off solid and only solve the fluid problem")
    parser.add_argument("-e", "--extrapolation", type=str, default=None,
                        metavar="Extrapolation method",
                        choices=["laplace", "elastic", "biharmonic", "no-extrapolation"],
                        help="Set approach for extrapolating the deformation into the fluid" +
                        "domain")
    parser.add_argument("-et", "--extrapolation-sub-type", type=str,
                        metavar="Extrapolation sub type", default=None,
                        choices=["constant", "small_constant", "volume", "volume_change",
                                 "bc1", "bc2"],
                        help="Set the sub type of the extrapolation method. TODO")
    parser.add_argument("--bc_ids", nargs="+", type=int, default=None, metavar="ID list",
                        help="List of boundary ids for the weak formulation of the" +
                        " biharmonic mesh lifting operator with 'bc2'")

    # Meterial settings / physical constants
    parser.add_argument("--Um", type=float, default=None,
                        help="Maximum velocity at inlet")
    parser.add_argument("--rho_f", type=float, default=None,
                        help="Density of the fluid")
    parser.add_argument("--mu_f", type=float, default=None,
                        help="Fluid dynamic viscosity")
    parser.add_argument("--rho_s", type=float, default=None,
                        help="Density of the solid")
    parser.add_argument("--mu_s", type=float, default=None,
                        help="Shear modulus in the solid")
    parser.add_argument("--nu_s", type=float, default=None,
                        help="Poisson ratio in the solid")
    parser.add_argument("--lambda_s", type=float, default=None,
                        help="Young's modulus in the solid")
    parser.add_argument("--gravity", type=float, default=None,
                        help="Gravitational force on the solid")

    # Domain settings
    parser.add_argument("--dx-f-id", type=int, default=None,
                        help="Domain id of the fluid domain")
    parser.add_argument("--dx-s-id", type=int, default=None,
                        help="Domain id of the solid domain")

    # Solver settings
    parser.add_argument("--linear-solver", type=str, default=None,
                        help="Choose the linear solver for each Newton iteration," +
                        " to see a complete list run list_linear_solvers()")
    parser.add_argument("--atol", type=float, default=None,
                        metavar="Absolute tolerance",
                        help="The absolute error tolerance for the Newton iterations")
    parser.add_argument("--rtol", type=float, default=None,
                        metavar="Relative tolerance",
                        help="The relative error tolerance for the Newton iterations")
    parser.add_argument("--max_it", type=int, default=None,
                        metavar="Maximum iterations",
                        help="Maximum number of iterations in the Newton solver")
    parser.add_argument("--lmbda", type=restricted_float, default=None,
                        metavar="Relaxation factor",
                        help="Relaxation factor in the Netwon solver")
    parser.add_argument("--recompute", type=int, default=None,
                        metavar="Recompute Jacobian",
                        help="How often to recompute the Jacobian.")
    parser.add_argument("--compiler-parameters", dest="compiler_parameters",
                        action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL",
                        help="Update the defaul values of the compiler arguments" +
                        " by providing a key=value, e.g. optimize=False. You can provide" +
                        " multiple key=value pairs seperated by a whitespace",
                        default=None)

    # Output settings
    parser.add_argument("-v", "--verbose", type=str2bool, default=None,
                        help="Turn on/off verbose printing")
    parser.add_argument("--loglevel", type=int, default=None,
                        help="Set FEniCS loglevel")
    parser.add_argument("--save-step", type=int, default=None,
                        help="Saving frequency of the files defined in the problem file")

    # Set spatial and temporal resolution
    parser.add_argument("-dt", metavar="Time step", type=float,
                        help="Set timestep, dt", default=None)
    parser.add_argument("-T", type=float, metavar="End time",
                        help="Set end time", default=None)
    parser.add_argument("--p_deg", metavar="Pressure degree", type=int,
                        help="Set degree of pressure", default=None)
    parser.add_argument("--v_deg", metavar="Velocity degree", type=int,
                        help="Set degree of velocity", default=None)
    parser.add_argument("--d_deg", metavar="Deformation degree", type=int,
                        help="Set degree of deformation", default=None)

    # Add the posibility pass unspecificed arguments
    parser.add_argument("--new-arguments", action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL",
                        help="Add any non-defined argument where the value is a string," +
                        " by providing a key=value, e.g. folder=TF_fsi_results. You can provide" +
                        " multiple key=value pairs seperated by a whitespace",
                        default=None)

    # Parse arguments
    args = parser.parse_args()

    # Add unspecificed arguments
    if args.new_arguments is not None:
        args.__dict__.update(args.new_arguments)
        args.__dict__.pop("new_arguments")

    # Update the default values and then set the entire dictionary to be the inpute from
    # arparse
    if args.compiler_parameters is not None:
        from turtleFSI.problems import default_variables
        default_variables["compiler_parameters"].update(args.compiler_parameters)
        args.compiler_parameters = default_variables["compiler_parameters"]

    return args
