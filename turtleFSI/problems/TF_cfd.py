# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import *
import numpy as np
from os import path

from turtleFSI.problems import *
from turtleFSI.modules import *


def set_problem_parameters(args, default_variables, **namespace):
    # Overwrite or add new variables to 'default_variables'
    default_variables.update(dict(
        # Temporal variables
        T=30,                     # End time [s]
        dt=0.01,                  # Time step [s]
        theta=0.5,                # Temporal scheme

        # Physical constants
        rho_f=1.0E3,              # Fluid density [kg/m3]
        mu_f=1.0,                 # Fluid dynamic viscosity [Pa.s]
        Um=2.0,                   # Max. velocity inlet (CDF3: 2.0) [m/s]

        # Problem specific
        folder="TF_cfd_results",  # Name of the results fulter
        solid="no_solid",         # Do not solve for the solid
        extrapolation="no_extrapolation",  # No displacement to extrapolate

        # Geometric variables
        H=0.41,                   # Total height
        L=2.5))                   # Length of domain

    return default_variables


def get_mesh_domain_and_boundaries(L, H, **namespace):
    mesh = Mesh(path.join(path.dirname(path.abspath(__file__)), "..", "mesh",
                          "TF_cfd.xml.gz"))
    mesh = refine(mesh)

    # Define the boundaries
    Inlet = AutoSubDomain(lambda x: near(x[0], 0))
    Outlet = AutoSubDomain(lambda x: near(x[0], L))
    Walls = AutoSubDomain(lambda x: near(x[1], 0) or near(x[1], H))

    # Mark the boundaries
    Allboundaries = DomainBoundary()
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundaries.set_all(0)
    Allboundaries.mark(boundaries, 4)  # Circle and flag
    Inlet.mark(boundaries, 1)
    Walls.mark(boundaries, 2)
    Outlet.mark(boundaries, 3)

    # Define the domain
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(1)

    return mesh, domains, boundaries


def initiate(dvp_, folder, **namespace):
    # Create files for storing results
    u_file = XDMFFile(MPI.comm_world, path.join(folder, "velocity.xdmf"))
    d_file = XDMFFile(MPI.comm_world, path.join(folder, "d.xdmf"))
    p_file = XDMFFile(MPI.comm_world, path.join(folder, "pressure.xdmf"))
    for tmp_t in [u_file, d_file, p_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["rewrite_function_mesh"] = False

    # Store initial conditions
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)
    d_file.write(d)
    u_file.write(v)
    p_file.write(p)

    # Lists to hold displacement, forces, and time
    Drag_list = []
    Lift_list = []
    Time_list = []

    return dict(u_file=u_file, d_file=d_file, p_file=p_file, Drag_list=Drag_list,
                Lift_list=Lift_list, Time_list=Time_list)


class Inlet(UserExpression):
    def __init__(self, Um, H, **kwargs):
        self.Um = Um * 1.5
        self.H = H
        self.factor = 0
        super().__init__(**kwargs)

    def update(self, t):
        if t < 2:
            self.factor = 0.5 * (1 - np.cos(t * np.pi / 2)) * self.Um
        else:
            self.factor = self.Um

    def eval(self, value, x):
        value[0] = self.factor * x[1] * (self.H - x[1]) / (self.H / 2.0)**2
        value[1] = 0

    def value_shape(self):
        return (2,)


def create_bcs(DVP, dvp_, Um, H, v_deg, boundaries, extrapolation_sub_type, **namespace):
    # Create inlet expression
    inlet = Inlet(Um, H, degree=v_deg)

    # Fluid velocity conditions
    u_inlet = DirichletBC(DVP.sub(1), inlet, boundaries, 1)
    u_wall = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 2)
    u_flag = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 4)

    # Pressure Conditions
    p_out = DirichletBC(DVP.sub(2), 0, boundaries, 3)

    return dict(bcs=[u_wall, u_flag, u_inlet, p_out], inlet=inlet)


def pre_solve(t, inlet, **namespace):
    """Update boundary conditions"""
    inlet.update(t)
    return {}


def post_solve(t, dvp_, n, Drag_list, Lift_list, Time_list, save_step, counter, u_file,
                p_file, d_file, mu_f, verbose, ds, **namespace):
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    if counter % save_step == 0:
        p_file.write(p, t)
        d_file.write(d, t)
        u_file.write(v, t)

    force = dot(sigma(v, p, d, mu_f), n)
    Drag_list.append(-assemble(force[0]*ds(4)))
    Lift_list.append(-assemble(force[1]*ds(4)))
    Time_list.append(t)

    if MPI.rank(MPI.comm_world) == 0 and verbose:
        print("Drag:", Drag_list[-1])
        print("Lift:", Lift_list[-1])

    return {}


def finished(Drag_list, Lift_list, Time_list, folder, **namespace):
    if MPI.rank(MPI.comm_world) == 0:
        np.savetxt(path.join(folder, 'Lift.txt'), Lift_list, delimiter=',')
        np.savetxt(path.join(folder, 'Drag.txt'), Drag_list, delimiter=',')
        np.savetxt(path.join(folder, 'Time.txt'), Time_list, delimiter=',')
