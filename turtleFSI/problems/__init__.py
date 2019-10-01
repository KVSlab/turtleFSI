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
    dt=0.001,  # timestep
    theta=0.501,  # temporal integration
                  # (theta=0 : first order explicit forward Euler scheme)
                  # (theta=1 : first order implicit backward Euler scheme)
                  # (theta=0.5 : second-order Crank-Nicolson scheme)
                  # (theta=0.5+dt : gives a better long-term numerical stability)
    T=1,  # end time

    # Spatial settings
    v_deg=2,  # velocity degree
    p_deg=1,  # pressure degree
    d_deg=2,  # solid deformation degree

    # Domain settings
    dx_f_id=1,  # Domain id of the fluid domain
    dx_s_id=2,  # Domain id of the solid domain

    # Meterial settings
    rho_f=1.0E3,  # Density of the fluid
    mu_f=1.0,  # Fluid dynamic viscosity
    rho_s=1.0E3,  # Density of the solid
    mu_s=5.0E4,  # Shear modulus in the solid
    nu_s=0.45,  # Poisson ratio
    lambda_s=4.5E5,  # Young's modulus in the solid
    gravity=None,  # Gravitational force on the solid
    Um=0.8,  # Maximum velocity at inlet

    # Variational formulations
    fluid="fluid",  # ["fluid", "no-fluid"] Turn off fluid and only solve the solid problem
    solid="solid",  # ["solid", "no-solid"] Turn off solid and only solve the fluid problem
    extrapolation="laplace",  # laplace, elastic, biharmonic, no-extrapolation
    extrapolation_sub_type="constant",  # small_constant, volume, constant, bc1, bc2
    bc_ids=[],  # List of ids for weak form of biharmonic mesh lifting operator with 'bc2'

    # Solver settings
    linear_solver="mumps",  # use list_linear_solvers() to check alternatives
    solver="newtonsolver",  # newtonsolver
    atol=1e-7,  # absolute error tolerance for the Newton iterations
    rtol=1e-7,  # relative error tolerance for the Newton iterations
    max_it=50,  # maximum number of Newton iterations
    lmbda=1.0,  # (>0-1.0) Cst relaxation factor for the Newton solution update
    recompute=5,  # recompute the Jacobian after "recompute" Newton iterations
    recompute_tstep=1,  # recompute the Jacobian after "recompute_tstep" time steps (advanced option: =1 is preferred)
    compiler_parameters=_compiler_parameters,  # Update the defaul values of the compiler arguments (FEniCS)

    # Output settings
    loglevel=20,    # Log level from FEniCS
    verbose=True,  # Turn on/off verbose printing
    save_step=1)    # Save file frequency


def set_problem_parameters(**namespace):
    """
    Set values to the problem variables. Overwrite the default values present
    in the __init__.py file, but will be overwritten by any argument passed in
    the command line.
    """

    return {}


def get_mesh_domain_and_boundaries(**namespace):
    """
    Import mesh files and create the Mesh() and MeshFunction() defining the mesh
    geometry, fluid/solid domains, and boundaries.
    """

    raise NotImplementedError("You need to define the mesh, domains and boundaries" +
                              "of the problem")


def initiate(**namespace):
    """
    Initiate any variables or data files before entering the time loop of the simulation.
    """

    return {}


def create_bcs(**namespace):
    """
    Define the boundary conditions of the problem to be solved.
    """

    return {}


def pre_solve(**namespace):
    """
    Function called iteratively within the time loop of the simulation before
    solving the problem. Use to update boundary conditions (for instance, time
    variable inflow velocity expression).
    """
    pass


def post_solve(**namespace):
    """
    Function called iteratively within the time loop of the simulation after
    solving the problem. Use to save data.
    """
    pass


def finished(**namespace):
    """
    Function called once at the end of the time loop.
    """

    pass
