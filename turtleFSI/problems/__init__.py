# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Define all common variables. Can be overwritten by defining in problem file or on the
commandline.
"""

from dolfin import parameters

_compiler_parameters = parameters["form_compiler"]
default_variables = dict(
              # Temporal settings
              dt = 0.001,
              tetha = 0.501,
              T = 5,

              # Spatial settings
              refine = None,
              v_deg = 2,
              p_deg = 1,
              d_def = 2,

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
              extrapolation = "laplace",            # biharmonic, linear, no_extrapolation
              extrapolation_sub_type = "constant",  # small_constant, volume, constant,
                                                    # bc1, bc2
              gravity = None,

              # Solver settings
              linear_solver = "mumps",
              atol = 1e-7,
              rtol = 1e-7,
              max_it = 50,
              lmbda = 1.0,
              recompute = 5,
              compiler_parameters=_compiler_parameters.update({"quadrature_degree": 4,
                                                               "optimize": True}),

              # FEniCS settings
              loglevel = 40,

              # Post-processing and storing settings
              step = 1,
              checkpoint = 1)


def set_problem_parameters():
    return {}


def get_mesh_domain_and_boundaries():
    return None, None, None


def initiate():
    return {}


def create_bcs():
    return {}


def pre_solve():
    return {}


def after_solve():
    return {}


def post_process():
    return {}
