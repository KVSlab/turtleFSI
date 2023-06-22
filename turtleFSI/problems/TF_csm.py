# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""Problem file for running the "CSM" benchmarks in [1]. The problem is beam under load.

[1] Turek, Stefan, and Jaroslav Hron. "Proposal for numerical benchmarking of fluid-structure interaction
between an elastic object and laminar incompressible flow." Fluid-structure interaction.
Springer, Berlin, Heidelberg, 2006. 371-385."""

from dolfin import *
import numpy as np
from os import path
from mpi4py import MPI as pyMPI

from turtleFSI.problems import *


def set_problem_parameters(default_variables, **namespace):
    # Parameters
    default_variables.update(dict(
        # Temporal variables
        T=10,          # End time [s]
        dt=0.01,       # Time step [s]
        theta=0.51,     # Temporal scheme

        # Physical constants
        rho_s=1.0e3,   # Solid density[kg/m3]
        mu_s=0.5e6,    # Shear modulus, 2nd Lame Coef. CSM3: 0.5E6 [Pa]
        nu_s=0.4,      # Solid Poisson ratio [-]
        gravity=2.0,   # Gravitational force [m/s^2]
        lambda_s=2e6,  # Solid 1st Lame Coef. [Pa]

        # Problem specific
        dx_f_id=0,     # Id of the fluid domain
        dx_s_id=1,     # Id of the solid domain
        folder="TF_csm_results",          # Folder to store the results
        fluid="no_fluid",                 # Do not solve for the fluid
        extrapolation="no_extrapolation",  # No displacement to extrapolate

        # Geometric variables
        R=0.05,        # Radius of the circle
        c_x=0.2,       # Center of the circle x-direction
        c_y=0.2,       # Center of the circle y-direction
        f_L=0.35))     # Length of the flag

    return default_variables


def get_mesh_domain_and_boundaries(c_x, c_y, R, **namespace):
    # Read mesh
    mesh = Mesh(path.join(path.dirname(path.abspath(__file__)), "..", "mesh", "TF_csm.xml.gz"))
    mesh = refine(mesh)

    # Mark boundaries
    Barwall = AutoSubDomain(lambda x: ((x[0] - c_x)**2 + (x[1] - c_y)**2 < R**2 + DOLFIN_EPS * 1e5))
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundaries.set_all(0)
    Barwall.mark(boundaries, 1)

    # Mark domain
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(1)

    return mesh, domains, boundaries


def initiate(f_L, R, c_x, c_y, **namespace):
    # Coordinate for sampling statistics
    coord = [c_x + R + f_L, c_y]

    # Lists to hold results
    displacement_x_list = []
    displacement_y_list = []
    time_list = []

    return dict(displacement_x_list=displacement_x_list, displacement_y_list=displacement_y_list,
                time_list=time_list, coord=coord)


def create_bcs(DVP, boundaries, **namespace):
    # Clamp on the left hand side
    u_barwall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 1)
    v_barwall = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 1)

    return dict(bcs=[u_barwall, v_barwall])

################################################################################
# the function mpi4py_comm and peval are used to overcome FEniCS limitation of
# evaluating functions at a given mesh point in parallel.
# https://fenicsproject.discourse.group/t/problem-with-evaluation-at-a-point-in
# -parallel/1188


def mpi4py_comm(comm):
    '''Get mpi4py communicator'''
    try:
        return comm.tompi4py()
    except AttributeError:
        return comm


def peval(f, x):
    '''Parallel synced eval'''
    try:
        yloc = f(x)
    except RuntimeError:
        yloc = np.inf*np.ones(f.value_shape())

    comm = mpi4py_comm(f.function_space().mesh().mpi_comm())
    yglob = np.zeros_like(yloc)
    comm.Allreduce(yloc, yglob, op=pyMPI.MIN)

    return yglob
################################################################################


def post_solve(t, dvp_, coord, displacement_x_list, displacement_y_list, time_list, verbose, **namespace):
    # Add time
    time_list.append(t)

    # Add displacement
    d = dvp_["n"].sub(0, deepcopy=True)
    d_eval = peval(d, coord)
    dsx = d_eval[0]
    dsy = d_eval[1]
    displacement_x_list.append(dsx)
    displacement_y_list.append(dsy)

    if MPI.rank(MPI.comm_world) == 0 and verbose:
        print("Distance x: {:e}".format(dsx))
        print("Distance y: {:e}".format(dsy))


def finished(results_folder, displacement_x_list, displacement_y_list, time_list, **namespace):
    # Store results when the computation is finished
    if MPI.rank(MPI.comm_world) == 0:
        np.savetxt(path.join(results_folder, 'Time.txt'), time_list, delimiter=',')
        np.savetxt(path.join(results_folder, 'dis_x.txt'), displacement_x_list, delimiter=',')
        np.savetxt(path.join(results_folder, 'dis_y.txt'), displacement_y_list, delimiter=',')