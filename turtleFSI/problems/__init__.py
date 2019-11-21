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
import pickle

_compiler_parameters = dict(parameters["form_compiler"])
_compiler_parameters.update({"quadrature_degree": 4, "optimize": True})

default_variables = dict(
    # Temporal settings
    dt=0.001,     # timestep
    theta=0.501,  # temporal integration
                  # (theta=0 : first order explicit forward Euler scheme)
                  # (theta=1 : first order implicit backward Euler scheme)
                  # (theta=0.5 : second-order Crank-Nicolson scheme)
                  # (theta=0.5+dt : gives a better long-term numerical stability)
    T=1,          # end time
    t=0,          # start at time t
    counter=0,    # timestep (should be 0 unless restart_path is not None)

    # Spatial settings
    v_deg=2,  # velocity degree
    p_deg=1,  # pressure degree
    d_deg=2,  # solid deformation degree

    # Domain settings
    dx_f_id=1,  # Domain id of the fluid domain
    dx_s_id=2,  # Domain id of the solid domain

    # Meterial settings
    rho_f=1.0E3,     # Density of the fluid
    mu_f=1.0,        # Fluid dynamic viscosity
    rho_s=1.0E3,     # Density of the solid
    mu_s=5.0E4,      # Shear modulus or 2nd Lame Coef. for the solid
    nu_s=0.45,       # Poisson ratio
    lambda_s=4.5E5,  # 1st Lame Coef. for the solid
    gravity=None,    # Gravitational force on the solid
    Um=0.8,          # Maximum velocity at inlet

    # Variational formulations
    fluid="fluid",                      # ["fluid", "no-fluid"] Turn off fluid and only solve the solid problem
    solid="solid",                      # ["solid", "no-solid"] Turn off solid and only solve the fluid problem
    extrapolation="laplace",            # laplace, elastic, biharmonic, no-extrapolation
    extrapolation_sub_type="constant",  # small_constant, volume, constant, constrained_disp, constrained_disp_vel
    bc_ids=[],                          # List of ids for weak form of biharmonic mesh lifting operator with 'constrained_disp_vel'

    # Solver settings
    linear_solver="mumps",  # use list_linear_solvers() to check alternatives
    solver="newtonsolver",  # newtonsolver
    atol=1e-7,              # absolute error tolerance for the Newton iterations
    rtol=1e-7,              # relative error tolerance for the Newton iterations
    max_it=50,              # maximum number of Newton iterations
    lmbda=1.0,              # [0, 1.0] Cst relaxation factor for the Newton solution update
    recompute=5,            # recompute the Jacobian after "recompute" Newton iterations
    recompute_tstep=1,      # recompute the Jacobian after "recompute_tstep" time steps (advanced option: =1 is preferred)
    compiler_parameters=_compiler_parameters,  # Update the defaul values of the compiler arguments (FEniCS)

    # Output settings
    loglevel=20,          # Log level from FEniCS
    verbose=True,         # Turn on/off verbose printing
    save_step=1,          # Save file frequency
    checkpoint_step=500   # Checkpoint frequency
    folder="results",     # Folder to store results and checkpoint files
    sub_folder=None)      # The unique name of the sub directory under folder where the results are stored


def create_folders(folder, sub_folder, **namespace):
    """Manage paths for where to store the checkpoint and visualizations"""
    # Get path to sub folder for this simulation
    path = Path.cwd() / folder
    if sub_folder is not None:
        path.joinpath(sub_folder)
    else:
        if path.glob(*) == []:
            path.joinpath("1")
        else:
            number = max([int(i) for i in path.glob(*) if i.isdigit()])
            path.joinpath(str(number))

    # Folders for visualization and checkpointing
    checkpoint_folder = path.joinpath("Checkpoint")
    visualization_folder = path.joinpoint("Visualization")
    checkpoint_folder.mkdir(parents=True, exists_ok=True)
    visualization_folder.mkdir(parents=True, exists_ok=True)

    return dict(checkpoint_folder=checkpoint_folder,
                visualization_folder=visualization_folder)


def checkpoint(dvp_, default_variables, checkpoint_folder, **namespace):
    """Utility function for storing the current parameters and the last two timesteps to
    restart from later"""
    # Only update variables that exists in default_variables
    default_variables.update((k, namespace[k]) for k in (default_variables.keys() & namespace.keys()))

    # Dump default parameters
    pickle.dump(default_variables, checkpoint_folder.joinpath("default_parameters.pickle"))

    # Dump physical fields
    d1 = dvp_["n-1"].sub(0, deepcopy=True)
    v1 = dvp_["n-1"].sub(1, deepcopy=True)
    p1 = dvp_["n-1"].sub(2, deepcopy=True)
    d2 = dvp_["n-2"].sub(0, deepcopy=True)
    v2 = dvp_["n-2"].sub(1, deepcopy=True)
    p2 = dvp_["n-2"].sub(2, deepcopy=True)

    # Create temporary checkpoint files
    for name, field in [('d1', d1), ('d2', d2), ('v1', v1), ('v2', v2), ('p1', p1), ('p2', p2)]:
        checkpoint = XDMFFile(MPI.comm_world, checkpoint_folder.joinpath("tmp_" + name + ".xdmf"))
        checkpoint.write_checkpoint(field)
        checkpoint.close()

    # Move to correct checkpoint name
    MPI.barrier()
    if MPI.rank == 0:
        for name in ['d1', 'd2', 'v1', 'v2', 'p1', 'p2']:
            new_name = checkpoint_folder.joinpath("checkpoint_" + name + ".xdmf")
            if new_name.exists():
                checkpoint_folder.joinpath("tmp_" + name + ".xdmf").replace(new_name)
            else:
                checkpoint_folder.joinpath("tmp_" + name + ".xdmf").rename(new_name)


def save_files_visualization(visualization_folder, dvp_, t, **namespace):
    # Files for storing results
    u_file = XDMFFile(MPI.comm_world, visualization_folder.joinpath("velocity.xdmf"))
    d_file = XDMFFile(MPI.comm_world, visualization_folder.joinpath("d.xdmf"))
    p_file = XDMFFile(MPI.comm_world, visualization_folder.joinpath("pressure.xdmf"))
    for tmp_t in [u_file, d_file, p_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["rewrite_function_mesh"] = False

    # Split function
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    # Name function
    d.rename("Deformation", "d")
    v.rename("Velocity", "v")
    p.rename("Pressure", "p")

    # Write results
    d_file.write(d, t)
    u_file.write(v, t)
    p_file.write(p, t)


def start_from_checkpoint(dvp_, restart_folder, **namespace):
    d1 = dvp_["n-1"].sub(0, deepcopy=True)
    d2 = dvp_["n-2"].sub(0, deepcopy=True)
    v1 = dvp_["n-1"].sub(1, deepcopy=True)
    v2 = dvp_["n-2"].sub(1, deepcopy=True)
    p1 = dvp_["n-1"].sub(2, deepcopy=True)
    p2 = dvp_["n-2"].sub(2, deepcopy=True)

    restart_folder = Path(restart_folder).joinpath("Checkpoint")
    for name, field in [('d1', d1), ('d2', d2), ('v1', v1), ('v2', v2), ('p1', p1), ('p2', p2)]:
        f = XDMFFile((MPI.comm_world, restart_folder.joinpath("checkpoint_" + name + ".xdmf")))
        f.read_checkpoint(d1)
        f.close()


def set_problem_parameters(**namespace):
    """
    Set values to the problem variables. Overwrite the default values present
    in the __init__.py file, but will be overwritten by any argument passed in
    the command line.
    """

    return default_variables


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
