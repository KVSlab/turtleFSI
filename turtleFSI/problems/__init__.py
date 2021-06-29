# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Define all common variables. Can be overwritten by defining in problem file or on the
commandline.
"""

from dolfin import parameters, XDMFFile, MPI, assign, Mesh, refine, project, VectorElement, FiniteElement,PETScDMCollection, FunctionSpace, Function
import pickle
from pathlib import Path
from xml.etree import ElementTree as ET

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
    save_deg=1,                                # Degree of the functions saved for visualisation '1' '2' '3' etc... (high value can slow down simulation significantly!)
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

    if "Checkpoint" in path.__str__():
        path = path.parent

    if not path.joinpath("Checkpoint").exists() and restart_folder is not None:
        raise NotADirectoryError(("The restart folder: {} does not have a sub folder 'Checkpoint' where we can"
                                  " restart the simulation from.").format(restart_folder))

    # Folders for visualization and checkpointing
    checkpoint_folder = path.joinpath("Checkpoint")
    visualization_folder = path.joinpath("Visualization")

    # Check if there exists previous visualization files, if so move and change name
    if list(visualization_folder.glob("*")) != []:
        # Get number of run(s)
        a = list(visualization_folder.glob("velocity*.h5"))
        b = [int(i.__str__().split("_")[-1].split(".")[0]) for i in a if "_" in i.name.__str__()]
        run_number = 1 if len(b) == 0 else max(b) + 1

        if MPI.rank(MPI.comm_world) == 0:
            for name in ["displacement", "velocity", "pressure"]:
                if not visualization_folder.joinpath(name + ".h5").exists():
                    continue
                for suffix in [".h5", ".xdmf"]:
                    new_name = visualization_folder.joinpath(name + "_run_" + str(run_number) + suffix)
                    tmp_path = visualization_folder.joinpath(name + suffix)
                    tmp_path.rename(new_name)

                # Rename link in xdmf file
                with open(new_name) as f:
                    text = f.read().replace(name + ".h5", new_name.name.__str__().replace(".xdmf", ".h5"))

                with open(new_name, "w") as f:
                    f.write(text)
    else:
        run_number = 0

    if MPI.rank(MPI.comm_world) == 0:
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        visualization_folder.mkdir(parents=True, exist_ok=True)

    return dict(checkpoint_folder=checkpoint_folder,
                visualization_folder=visualization_folder,
                results_folder=path, run_number=run_number)


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


def save_files_visualization(visualization_folder, dvp_, t, save_deg, v_deg, p_deg, mesh, domains, **namespace):
    # Files for storing results
    if not "d_file" in namespace.keys():
        d_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("displacement.xdmf")))
        v_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("velocity.xdmf")))
        p_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("pressure.xdmf")))
        for tmp_t in [d_file, v_file, p_file]:
            tmp_t.parameters["flush_output"] = True
            tmp_t.parameters["rewrite_function_mesh"] = False

        if save_deg > 1:
            print('save deg > 1 selected...')
            print('save deg > 1 selected...')
            
            # Create function space for d, v and p
            dve = VectorElement('CG', mesh.ufl_cell(), v_deg)
            pe = FiniteElement('CG', mesh.ufl_cell(), p_deg)
            FSdv = FunctionSpace(mesh, dve)   # Higher degree FunctionSpace for d and v
            FSp= FunctionSpace(mesh, pe)     # Higher degree FunctionSpace for p

            # Copy mesh
            mesh_viz = Mesh(mesh)

            for i in range(save_deg-1):
                mesh_viz = refine(mesh_viz)  # refine the mesh
                domains_viz = adapt(domains,mesh_viz)  # refine the domains (so we can output domain IDs of refined mesh)

            # Create visualization function space for d, v and p
            dve_viz = VectorElement('CG', mesh_viz.ufl_cell(), 1)
            pe_viz = FiniteElement('CG', mesh_viz.ufl_cell(), 1)
            FSdv_viz = FunctionSpace(mesh_viz, dve_viz)   # Visualisation FunctionSpace for d and v
            FSp_viz = FunctionSpace(mesh_viz, pe_viz)     # Visualisation FunctionSpace for p

            # Create lower-order function for visualization on refined mesh
            d_viz = Function(FSdv_viz)
            v_viz = Function(FSdv_viz)
            p_viz = Function(FSp_viz)
    
            # Create a transfer matrix between higher degree and lower degree (visualization) function spaces
            dv_trans = PETScDMCollection.create_transfer_matrix(FSdv,FSdv_viz)
            p_trans = PETScDMCollection.create_transfer_matrix(FSp,FSp_viz)

            return_dict = dict(v_file=v_file, d_file=d_file, p_file=p_file, d_viz=d_viz,v_viz=v_viz, p_viz=p_viz, dv_trans=dv_trans, p_trans=p_trans, mesh_viz=mesh_viz, domains_viz=domains_viz)

        else:
            return_dict = dict(v_file=v_file, d_file=d_file, p_file=p_file)

        namespace.update(return_dict)

    else:
        return_dict = {}

    # Split function
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    if save_deg > 1: # To save higher-order nodes

        # Interpolate by using the transfer matrix between higher degree and lower degree (visualization) function spaces
        namespace["d_viz"].vector()[:] = namespace["dv_trans"]*d.vector()
        namespace["v_viz"].vector()[:] = namespace["dv_trans"]*v.vector()
        namespace["p_viz"].vector()[:] = namespace["p_trans"]*p.vector()

        write_solution(namespace["d_viz"], namespace["v_viz"], namespace["p_viz"], namespace["d_file"], namespace["v_file"], namespace["p_file"], t) # Write results

    else: # To save only the corner nodes

        write_solution(d, v, p, namespace["d_file"], namespace["v_file"], namespace["p_file"], t) # Write results

    return return_dict

def write_solution(d, v, p, d_file, v_file, p_file, t):
    # Name functions
    d.rename("Displacement", "d")
    v.rename("Velocity", "v")
    p.rename("Pressure", "p") 

    # Write results
    d_file.write(d, t)
    v_file.write(v, t)
    p_file.write(p, t)

def start_from_checkpoint(dvp_, restart_folder, mesh, **namespace):
    """Restart simulation from a previous simulation by by setting restart_folder"""
    # Dump physical fields
    fields = _get_fields(dvp_, mesh)

    for name, field in fields:
        checkpoint_path = str(restart_folder.joinpath("checkpoint_" + name + ".xdmf"))
        with XDMFFile(MPI.comm_world, checkpoint_path) as f:
            f.read_checkpoint(field, name)

    assign(dvp_["n-1"].sub(0), fields[0][1])  # update d_["n-1"] to checkpoint d_["n-1"]
    assign(dvp_["n-1"].sub(1), fields[1][1])  # update v_["n-1"] to checkpoint v_["n-1"]
    assign(dvp_["n-1"].sub(2), fields[2][1])  # update p_["n-1"] to checkpoint p_["n-1"]
    assign(dvp_["n"].sub(0), fields[0][1])    # update d_["n-1"] to checkpoint d_["n-1"]
    assign(dvp_["n"].sub(1), fields[1][1])    # update v_["n-1"] to checkpoint v_["n-1"]
    assign(dvp_["n"].sub(2), fields[2][1])    # update p_["n-1"] to checkpoint p_["n-1"]


def _get_fields(dvp_, mesh):
    d1 = dvp_["n-1"].sub(0, deepcopy=True)
    v1 = dvp_["n-1"].sub(1, deepcopy=True)
    p1 = dvp_["n-1"].sub(2, deepcopy=True)
    fields = [('d1', d1), ('v1', v1), ('p1', p1)]

    if len(dvp_["n-1"]) == mesh.geometric_dimension() * 3 + 1:
        w1 = dvp_["n-1"].sub(3, deepcopy=True)
        fields += [('w1', w1)]

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
        txt = txt.format('=' * int(20*j-1) + ">", 100 * j, elapsed_time)
        print(txt, end='\r')

    return timer.elapsed()[0]


def merge_visualization_files(visualization_folder, **namesapce):
    # Gather files
    xdmf_files = list(visualization_folder.glob("*.xdmf"))
    xdmf_displacement = [f for f in xdmf_files if "displacement" in f.__str__()]
    xdmf_velocity = [f for f in xdmf_files if "velocity" in f.__str__()]
    xdmf_pressure = [f for f in xdmf_files if "pressure" in f.__str__()]

    # Merge files
    for files in [xdmf_displacement, xdmf_velocity, xdmf_pressure]:
        if len(files) > 1:
            merge_xml_files(files)


def merge_xml_files(files):
    # Get first timestep and trees
    first_timesteps = []
    trees = []
    for f in files:
        trees.append(ET.parse(f))
        root = trees[-1].getroot()
        first_timesteps.append(float(root[0][0][0][2].attrib["Value"]))

    # Index valued sort (bypass numpy dependency)
    first_timestep_sorted = sorted(first_timesteps)
    indexes = [first_timesteps.index(i) for i in first_timestep_sorted]

    # Get last timestep of first tree
    base_tree = trees[indexes[0]]
    last_node = base_tree.getroot()[0][0][-1]
    ind = 1 if len(last_node.getchildren()) == 3 else 2
    last_timestep = float(last_node[ind].attrib["Value"])

    # Append
    for index in indexes[1:]:
        tree = trees[index]
        for node in tree.getroot()[0][0].getchildren():
            ind = 1 if len(node.getchildren()) == 3 else 2
            if last_timestep < float(node[ind].attrib["Value"]):
                base_tree.getroot()[0][0].append(node)
                last_timestep = float(node[ind].attrib["Value"])

    # Seperate xdmf files
    new_file = [f for f in files if "_" not in f.name.__str__()]
    old_files = [f for f in files if "_" in f.name.__str__()]

    # Write new xdmf file
    base_tree.write(new_file[0], xml_declaration=True)

    # Delete xdmf file
    [f.unlink() for f in old_files]


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
