# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

""" TODO: Add a general description of and mathematics of the problems that is solved"""

from dolfin import *
import numpy as np
from os import path
from turtleFSI.problems import *


def set_problem_parameters(args, default_variables, **namespace):
    """TODO"""
    # Overwrite default values
    default_variables.update(dict(
                       T = 100,          # End time [s]
                       dt = 0.001,       # Time step [s]
                       rho_f = 1.0E3,    # Fluid density [kg/m3]
                       mu_f = 1.0,       # Fluid dynamic viscosity [Pa.s]
                       rho_s = 1.0E3,    # Solid density [kg/m3]
                       mu_s = 5.0E4,     # Solid shear modulus or 2nd Lame Coef. [Pa]
                       lambda_s = 4.5E5, # Solid Young's modulus [Pa]
                       nu_s = 0.45,      # Solid Poisson ratio [-]
                       dx_f_id = 1,      # ID of marker in the fluid domain
                       dx_s_id = 2))     # ID of marker in the solid domain

    # Overwrite problem specific values with those from commandline
    default_variables.update(args.__dict__)

    return default_variables


def get_mesh_domain_and_boundaries(args, **namespace):
    """Function for reading or defining the mesh, domains, and boundaries of
    the problem of interest.

    Args:
        pars (dict):

    Returns:
        mesh ():
        domains ():
        boundaries ():
    """
    #rel_path = path.dirname(path.abspath(__file__))
    mesh_folder = path.join("mesh", "turtle_demo")

    # In this example, the mesh and markers are stored in the 3 following files
    mesh_path = path.join(mesh_folder, "turtle_mesh.xdmf")  # mesh geometry
    domains_marker_path = path.join(mesh_folder, "mc.xdmf")      # marker over the elements (domains)
    boundaries_marker_path = path.join(mesh_folder, "mf.xdmf")   # markers of the segments (boundaries)

    # "mesh" collects the mesh geometry of the entire domain (fluid + solid).
    # In this example, we import a mesh stored in a .xdmf file, but other formats
    # are supported such as .xml files.
    mesh = Mesh()
    xdmf = XDMFFile(MPI.comm_world, mesh_path)
    xdmf.read(mesh)

    # "domains" collects the element markers of the fluid domain (marked as 1)
    # and the solid domain (marked as 2).
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    xdmf = XDMFFile(MPI.comm_world, domains_marker_path)
    xdmf.read(domains)

    # "boundaries" collects the boundary markers that are used to apply the
    # Dirichlet boundary conditions on both the fluid and solid domains.
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    xdmf = XDMFFile(MPI.comm_world, boundaries_marker_path)
    xdmf.read(boundaries)

    return mesh, domains, boundaries


class Inlet(Expression):
    def __init__(self, **kwargs):
        self.t = 0.0
        self.t_ramp = 1.0  # time to linearly ramp-up the inlet velocity
        self.Um = 0.8      # Max. velocity inlet [m/s]

    def update(self, t):
        self.t = t
        if self.t < self.t_ramp:
            self.value = self.Um * np.abs(np.cos(self.t*np.pi)-1)
        else:
            self.value = np.max([self.Um/5, self.Um * np.abs(np.cos(self.t*np.pi)-1)])

    def eval(self, value, x):
        value[0] = self.value
        value[1] = 0

    def value_shape(self):
        return (2,)


def create_bcs(DVP, boundaries, args, **namespace):
    if MPI.rank(MPI.comm_world) == 0:
        print("Create bcs")

    inlet = Inlet(degree=v_deg)
    noslip = ((0.0, 0.0))

    # Fluid velocity conditions
    u_inlet = DirichletBC(DVP.sub(1), inlet, boundaries, 14)
    u_bot = DirichletBC(DVP.sub(1).sub(1), (0.0), boundaries, 11)
    u_top = DirichletBC(DVP.sub(1).sub(1), (0.0), boundaries, 13)
    u_head_tail = DirichletBC(DVP.sub(1), noslip, boundaries, 15)

    # Pressure Conditions
    p_outlet = DirichletBC(DVP.sub(2), (0.0), boundaries, 12)

    bcs = [u_bot, u_top, u_inlet, p_outlet,  u_head_tail]

    if args.bitype == "bc1":
        d_inlet = DirichletBC(DVP.sub(0), noslip, boundaries, 14)
        d_bot = DirichletBC(DVP.sub(0), noslip, boundaries, 11)
        d_top = DirichletBC(DVP.sub(0), noslip, boundaries, 13)
        d_outlet = DirichletBC(DVP.sub(0), noslip, boundaries, 12)
        d_head_tail = DirichletBC(DVP.sub(0), noslip, boundaries, 15)
        for i in [d_bot, d_top, d_outlet, d_inlet, d_head_tail]:
            bcs.append(i)

    if args.bitype == "bc2":
        w_inlet = DirichletBC(DVP.sub(0), noslip, boundaries, 14)
        w_bot = DirichletBC(DVP.sub(0), noslip, boundaries, 11)
        w_top = DirichletBC(DVP.sub(0), noslip, boundaries, 13)
        w_outlet = DirichletBC(DVP.sub(0), noslip, boundaries, 12)
        w_head_tail = DirichletBC(DVP.sub(0), noslip, boundaries, 15)

        d_inlet = DirichletBC(DVP.sub(0), noslip, boundaries, 14)
        d_bot = DirichletBC(DVP.sub(0), noslip, boundaries, 11)
        d_top = DirichletBC(DVP.sub(0), noslip, boundaries, 13)
        d_outlet = DirichletBC(DVP.sub(0), noslip, boundaries, 12)
        d_head_tail = DirichletBC(DVP.sub(0), noslip, boundaries, 15)

        for i in [w_bot, w_top, w_outlet, w_inlet, w_head_tail,
                  d_bot, d_top, d_outlet, d_inlet, d_head_tail]:
            bcs.append(i)

    return dict(bcs=bcs, inlet=inlet)


def initiate(dvp_, **namespace):
    path = "results/turtle_demo/"

    # Files for storing results
    u_file = XDMFFile(MPI.comm_world, path + "/velocity.xdmf")
    d_file = XDMFFile(MPI.comm_world, path + "/d.xdmf")
    p_file = XDMFFile(MPI.comm_world, path + "/pressure.xdmf")
    for tmp_t in [u_file, d_file, p_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["rewrite_function_mesh"] = False

    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)
    d_file.write(d, 0.0)
    u_file.write(v, 0.0)
    p_file.write(p, 0.0)

    return dict(u_file=u_file, d_file=d_file, p_file=p_file, path=path)


def pre_solve(t, inlet, **namespace):
    """TODO"""
    inlet.update(t)


def after_solve(t, dvp_, counter, u_file, p_file, d_file, **namespace):
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    if counter % step == 0:
        d = dvp_["n"].sub(0, deepcopy=True)
        v = dvp_["n"].sub(1, deepcopy=True)
        p = dvp_["n"].sub(2, deepcopy=True)
        p_file.write(p, t)
        d_file.write(d, t)
        u_file.write(v, t)

    return {}
