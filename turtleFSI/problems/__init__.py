# Co (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Define all common variables. Can be overwritten by defining in problem file or on the
commandline.
"""

from dolfin import parameters

_compiler_parameters = dict(parameters["form_compiler"])
_compiler_parameters.update({"quadrature_degree": 4,
                             "optimize": True}),
default_variables = dict(
              # Temporal settings
              dt = 0.001,
              theta = 0.501,
              T = 5,

              # Spatial settings
              v_deg = 2,
              p_deg = 1,
              d_deg = 2,

              # Domain settings
              dx_f_id = 1,
              dx_s_id = 2,

              # Meterial settings
              rho_f = 1.0E3,
              mu_f = 1.0,
              rho_s = 1.0E3,
              mu_s = 5.0E4,
              nu_s =  0.45,
              lambda_s = 4.5E5,
              gravity = None,

              # Variational formulations
              fluid = "fluid",
              solid = "solid",
              extrapolation = "laplace",            # biharmonic, elastic, no_extrapolation
              extrapolation_sub_type = "constant",  # small_constant, volume, constant,
                                                    # bc1, bc2
              bc_ids = [],                          # List of ids for weak form and bc2

              # Solver settings
              linear_solver = "mumps",  # use list_linear_solvers() to check alternatives
              solver = "newtonsolver",  # newtonsolver_naive
              atol = 1e-7,
              rtol = 1e-7,
              max_it = 50,
              lmbda = 1.0,
              recompute = 5,
              compiler_parameters=_compiler_parameters,

              # Output settings
              loglevel = 40,    # Log level from FEniCS
              verbose = False,  # Turn on/off verbose printing
              save_step = 1)    # Save file frequency


def set_problem_parameters(**namespace):
    """
    TODO: Short explenation of what should happen in this function
    """

    return {}


def get_mesh_domain_and_boundaries(**namespace):
    """
    TODO: Short explenation of what should happen in this function
    """

    raise NotImplementedError("You need to define the mesh, domains and boundaries" +
                              "of the problem")


def initiate(**namespace):
    """
    TODO: Short explenation of what should happen in this function
    """

    return {}


def create_bcs(**namespace):
    """
    TODO: Short explenation of what should happen in this function
    """

    return {}


def pre_solve(**namespace):
    """
    TODO: Short explenation of what should happen in this function
    """

    pass


def after_solve(**namespace):
    """
    TODO: Short explenation of what should happen in this function
    """

    pass


def post_process(**namespace):
    """
    TODO: Short explenation of what should happen in this function
    """

    pass
