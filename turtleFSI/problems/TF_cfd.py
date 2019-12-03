# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""Problem file for running the "CFD" benchmarks in [1]. The problem is a channel flow
with a circle and a flag attached to it. For the CFD problem both the circle and flag is rigid.

[1] Turek, Stefan, and Jaroslav Hron. "Proposal for numerical benchmarking of fluid-structure interaction
between an elastic object and laminar incompressible flow." Fluid-structure interaction.
Springer, Berlin, Heidelberg, 2006. 371-385."""

from dolfin import *
import numpy as np
from os import path

from turtleFSI.problems import *
from turtleFSI.modules import *


def set_problem_parameters(default_variables, **namespace):
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
        folder="TF_cfd_results",  # Name of the results folder
        solid="no_solid",         # Do not solve for the solid
        extrapolation="no_extrapolation",  # No displacement to extrapolate

        # Geometric variables
        H=0.41,                   # Total height
        L=2.5))                   # Length of domain

    return default_variables


def get_mesh_domain_and_boundaries(L, H, **namespace):
    # Load and refine mesh
    mesh = Mesh(path.join(path.dirname(path.abspath(__file__)), "..", "mesh", "TF_cfd.xml.gz"))
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


def initiate(**namespace):
    # Lists to hold displacement, forces, and time
    drag_list = []
    lift_list = []
    time_list = []

    return dict(drag_list=drag_list, lift_list=lift_list, time_list=time_list)


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


def create_bcs(DVP, Um, H, v_deg, boundaries, **namespace):
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


def post_solve(t, dvp_, n, drag_list, lift_list, time_list, mu_f, verbose, ds, **namespace):
    # Get deformation, velocity, and pressure
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    # Compute forces
    force = dot(sigma(v, p, d, mu_f), n)
    drag_list.append(-assemble(force[0]*ds(4)))
    lift_list.append(-assemble(force[1]*ds(4)))
    time_list.append(t)

    # Print results
    if MPI.rank(MPI.comm_world) == 0 and verbose:
        print("Drag: {:e}".format(drag_list[-1]))
        print("Lift: {:e}".format(lift_list[-1]))


def finished(drag_list, lift_list, time_list, results_folder, **namespace):
    # Store results when the computation is finished
    if MPI.rank(MPI.comm_world) == 0:
        np.savetxt(path.join(results_folder, 'Lift.txt'), lift_list, delimiter=',')
        np.savetxt(path.join(results_folder, 'Drag.txt'), drag_list, delimiter=',')
        np.savetxt(path.join(results_folder, 'Time.txt'), time_list, delimiter=',')
