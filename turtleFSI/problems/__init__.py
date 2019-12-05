# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Define all common variables. Can be overwritten by defining in problem file or on the
commandline.
"""

from dolfin import parameters, XDMFFile, MPI
import pickle
from pathlib import Path

_compiler_parameters = dict(parameters["form_compiler"])
_compiler_parameters.update({"quadrature_degree": 4, "optimize": True})

default_variables = dict(
    # Temporal settings
    dt=0.001,        # time step
    theta=0.501,     # temporal integration
                     # (theta=0 : first order explicit forward Euler scheme)
                     # (theta=1 : first order implicit backward Euler scheme)
                     # (theta=0.5 : second-order Crank-Nicolson scheme)
                     # (theta=0.5+dt : gives a better long-term numerical stability)
    T=1,             # end time
    t=0,             # start at time t
    counter=0,       # time step (should be 0 unless restart_path is not None)

    # Spatial settings
    v_deg=2,         # velocity degree
    p_deg=1,         # pressure degree
    d_deg=2,         # solid deformation degree

    # Domain settings
    dx_f_id=1,       # Domain id of the fluid domain
    dx_s_id=2,       # Domain id of the solid domain

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
    fluid="fluid",                             # ["fluid", "no_fluid"] Turn off fluid and only solve the solid problem
    solid="solid",                             # ["solid", "no_solid"] Turn off solid and only solve the fluid problem
    extrapolation="laplace",                   # laplace, elastic, biharmonic, no_extrapolation
    extrapolation_sub_type="constant",         # small_constant, volume, constant, constrained_disp, constrained_disp_vel
    bc_ids=[],                                 # List of ids for weak form of biharmonic mesh lifting operator with 'constrained_disp_vel'

    # Solver settings
    linear_solver="mumps",                     # use list_linear_solvers() to check alternatives
    solver="newtonsolver",                     # newtonsolver, there is currently no other choices.
    atol=1e-7,                                 # absolute error tolerance for the Newton iterations
    rtol=1e-7,                                 # relative error tolerance for the Newton iterations
    max_it=50,                                 # maximum number of Newton iterations
    lmbda=1.0,                                 # [0, 1.0] Cst relaxation factor for the Newton solution update
    recompute=5,                               # recompute the Jacobian after "recompute" Newton iterations
    recompute_tstep=1,                         # recompute the Jacobian after "recompute_tstep" time steps (advanced option: =1 is preferred)
    compiler_parameters=_compiler_parameters,  # Update the default values of the compiler arguments (FEniCS)

    # Output settings
    loglevel=20,                               # Log level from FEniCS
    verbose=True,                              # Turn on/off verbose printing
    save_step=10,                              # Save file frequency
    checkpoint_step=500,                       # Checkpoint frequency
    folder="results",                          # Folder to store results and checkpoint files
    sub_folder=None,                           # The unique name of the sub directory under folder where the results are stored
    restart_folder=None)                       # Path to a potential restart folder


def create_folders(folder, sub_folder, restart_folder, **namespace):
    """Manage paths for where to store the checkpoint and visualizations"""

    if restart_folder is None:
        # Get path to sub folder for this simulation
        path = Path.cwd() / folder
        if sub_folder is not None:
            path = path.joinpath(sub_folder)
        else:
            if not [int(str(i.name)) for i in path.glob("*") if str(i.name).isdigit()]:
                path = path.joinpath("1")
            else:
                number = max([int(str(i.name)) for i in path.glob("*") if str(i.name).isdigit()])
                path = path.joinpath(str(number + 1))
    else:
        path = restart_folder

    MPI.barrier(MPI.comm_world)

    if not path.joinpath("Checkpoint").exists() and restart_folder is not None:
        raise NotADirectoryError(("The restart folder: {} does not have a sub folder 'Checkpoint' where we can"
                                  " restart the simulation from.").format(restart_folder))

    # Folders for visualization and checkpointing
    checkpoint_folder = path.joinpath("Checkpoint")
    visualization_folder = path.joinpath("Visualization")

    if MPI.rank(MPI.comm_world) == 0:
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        visualization_folder.mkdir(parents=True, exist_ok=True)

    return dict(checkpoint_folder=checkpoint_folder,
                visualization_folder=visualization_folder,
                results_folder=path)


def checkpoint(dvp_, default_variables, checkpoint_folder, mesh, **namespace):
    """Utility function for storing the current parameters and the last two time steps"""
    # Only update variables that exists in default_variables
    default_variables.update((k, namespace[k]) for k in (default_variables.keys() & namespace.keys()))

    # Dump default parameters
    if MPI.rank(MPI.comm_world) == 0:
        with open(str(checkpoint_folder.joinpath("default_variables.pickle")), "bw") as f:
            pickle.dump(default_variables, f)

    # Dump physical fields
    fields = _get_fields(dvp_, mesh)

    # Write fields to temporary file to avoid corruption of existing checkpoint
    for name, field in fields:
        checkpoint_path = str(checkpoint_folder.joinpath("tmp_" + name + ".xdmf"))
        with XDMFFile(MPI.comm_world, checkpoint_path) as f:
            f.write_checkpoint(field, name)

    # Move to correct checkpoint name
    MPI.barrier(MPI.comm_world)
    if MPI.rank(MPI.comm_world) == 0:
        for name, _ in fields:
            for suffix in [".h5", ".xdmf"]:
                new_name = checkpoint_folder.joinpath("checkpoint_" + name + suffix)
                if new_name.exists():
                    checkpoint_folder.joinpath("tmp_" + name + suffix).replace(new_name)
                else:
                    checkpoint_folder.joinpath("tmp_" + name + suffix).rename(new_name)

            # Rename link in xdmf file
            with open(new_name, "r") as f:
                text = f.read().replace("tmp_", "checkpoint_")

            with open(new_name, "w") as f:
                f.write(text)


def save_files_visualization(visualization_folder, dvp_, t, **namespace):
    # Files for storing results
    if not "d_file" in namespace.keys():
        d_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("displacement.xdmf")))
        v_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("velocity.xdmf")))
        p_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("pressure.xdmf")))
        for tmp_t in [d_file, v_file, p_file]:
            tmp_t.parameters["flush_output"] = True
            tmp_t.parameters["rewrite_function_mesh"] = False
        return_dict = dict(v_file=v_file, d_file=d_file, p_file=p_file)
        namespace.update(return_dict)
    else:
        return_dict = {}

    # Split function
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    # Name function
    d.rename("Displacement", "d")
    v.rename("Velocity", "v")
    p.rename("Pressure", "p")

    # Write results
    namespace["d_file"].write(d, t)
    namespace["v_file"].write(v, t)
    namespace["p_file"].write(p, t)

    return return_dict


def start_from_checkpoint(dvp_, restart_folder, mesh, **namespace):
    """Restart simulation from a previous simulation by by setting restart_folder"""
    # Dump physical fields
    fields = _get_fields(dvp_, mesh)

    for name, field in fields:
        checkpoint_path = str(restart_folder.joinpath("Checkpoint", "checkpoint_" + name + ".xdmf"))
        with XDMFFile(MPI.comm_world, checkpoint_path) as f:
            f.read_checkpoint(field, name)


def _get_fields(dvp_, mesh):
    d1 = dvp_["n-1"].sub(0, deepcopy=True)
    d2 = dvp_["n-2"].sub(0, deepcopy=True)
    v1 = dvp_["n-1"].sub(1, deepcopy=True)
    v2 = dvp_["n-2"].sub(1, deepcopy=True)
    p1 = dvp_["n-1"].sub(2, deepcopy=True)
    p2 = dvp_["n-2"].sub(2, deepcopy=True)
    fields = [('d1', d1), ('d2', d2), ('v1', v1), ('v2', v2), ('p1', p1), ('p2', p2)]

    if len(dvp_["n-1"]) == mesh.geometric_dimension() * 3 + 1:
        w1 = dvp_["n-1"].sub(3, deepcopy=True)
        w2 = dvp_["n-2"].sub(3, deepcopy=True)
        fields += [('w1', w1), ('w2', w2)]

    return fields


def print_information(counter, t, T, dt, timer, previous_t, verbose, **namespace):
    """Print information about the time step and solver time"""
    elapsed_time = timer.elapsed()[0] - previous_t
    if verbose:
        txt = "Solved for timestep {:d}, t = {:2.04f} in {:3.1f} s"
        txt = txt.format(counter, t, elapsed_time)
        print(txt)
    else:
        j = counter / int(T/dt + 1)
        txt = "Progress: [{:<20s}] {:2.1f}%, last solve took {:3.1f} s"
        txt = txt.format('='*int(20*j-1)+">", 100*j, elapsed_time)
        print(txt, end="\r")

    return timer.elapsed()[0]


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
