# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import *
import numpy as np
from os import path

from turtleFSI.problems import *

def set_problem_parameters(args, default_variables, **namespace):
    # Parameters
    default_variables.update(dict(
            # Temporal variables
            T = 30,          # End time [s]
            dt = 0.01,       # Time step [s]
            theta = 0.5,     # Temporal scheme

            # Physical constants
            rho_f = 1.0e3,   # Fluid density [kg/m3]
            mu_f = 1.0,      # Fluid dynamic viscosity [Pa.s]
            rho_s = 1.0e3,   # Solid density[kg/m3]
            mu_s = 0.5e6,    # Shear modulus, 2nd Lame Coef. CSM3: 0.5E6 [Pa]
            nu_s = 0.4,      # Solid Poisson ratio [-]
            gravity = 2.0,   # Gravitational force [m/s^2]
            lambda_s = 2e6,  # Solid Young's modulus [Pa]

            # Problem specific
            dx_f_id = 0,     # Id of the fluid domain
            dx_s_id = 1,     # Id of the solid domain
            folder = "TF_csm_results",          # Folder to store the results
            fluid = "no_fluid",                 # Do not solve for the fluid
            extrapolation = "no_extrapolation", # No displacement to extrapolate

            # Geometric variables
            R = 0.05,        # Radius of the circle
            c_x = 0.2,       # Center of the circle x-direction
            c_y = 0.2,       # Center of the circle y-direction
            f_L = 0.35))     # Length of the flag

    return default_variables


def get_mesh_domain_and_boundaries(mesh, c_x, c_y, R, **namespace):
    # Read mesh
    mesh = Mesh(path.join("mesh", "TF_csm.xml.gz"))
    mesh = refine(mesh)

    # Mark boundaries
    Barwall = AutoSubDomain(lambda x: ((x[0] - c_x)**2 + (x[1] - c_y)**2
                                       < R**2 + DOLFIN_EPS*1e5))
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundaries.set_all(0)
    Barwall.mark(boundaries, 1)

    # Mark domain
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(1)

    return mesh, domains, boundaries


def initiate(folder, mesh, dvp_, f_L, R, c_x, **namespace):
    # Files for storeing results
    u_file = XDMFFile(MPI.comm_world, path.join(folder, "velocity.xdmf"))
    d_file = XDMFFile(MPI.comm_world, path.join(folder, "d.xdmf"))
    for tmp_t in [u_file, d_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["rewrite_function_mesh"] = False

    # Store initial condition
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    d_file.write(d)
    u_file.write(v)

    # Coord to sample
    for coord in mesh.coordinates():
        if coord[0] == c_x + R + f_L and (c_y - 0.001 <= coord[1] <= c_y + 0.001):
            break

    # Lists to hold results
    dis_x = []
    dis_y = []
    Time_list = []

    return dict(u_file=u_file, d_file=d_file, dis_x=dis_x, dis_y=dis_y,
                Time_list=Time_list, coord=coord)


def create_bcs(DVP, boundaries, **namespace):
    # Clamp on the left hand side
    u_barwall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 1)

    return dict(bcs=[u_barwall])


def after_solve(t, dvp_, coord, dis_x, dis_y, counter, u_file, d_file, save_step,
                Time_list, verbose, **namespace):
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)

    if counter % save_step == 0:
        d = dvp_["n"].sub(0, deepcopy=True)
        v = dvp_["n"].sub(1, deepcopy=True)
        d_file.write(d, t)
        u_file.write(v, t)

    Time_list.append(t)
    dsx = d(coord)[0]
    dsy = d(coord)[1]
    dis_x.append(dsx)
    dis_y.append(dsy)

    if MPI.rank(MPI.comm_world) == 0 and verbose:
        print("dis_x | dis_y : %g %g " % (dsx, dsy))


def post_process(folder, dis_x, dis_y, Time_list, **namespace):
    if MPI.rank(MPI.comm_world) == 0:
        np.savetxt(path.join(folder, 'Time.txt'), Time_list, delimiter=',')
        np.savetxt(path.join(folder, 'dis_x.txt'), dis_x, delimiter=',')
        np.savetxt(path.join(folder, 'dis_y.txt'), dis_y, delimiter=',')
