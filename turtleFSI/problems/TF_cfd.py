# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import *
import numpy as np
from os import path

from turtleFSI.problems import *
from turtleFSI.modules import *

# Parameters
def set_problem_parameters(args, default_variables, **namespace):
    # Overwrite default values
    default_variables.update(dict(
            v_deg = 2,                  # Velocity degree
            p_deg = 1,                  # Pressure degree
            d_deg = 2,                  # Deformation degree
            T = 30,                     # End time [s]
            dt = 0.01,                  # Time step [s]
            rho_f = 1.0E3,              # Fluid density [kg/m3]
            mu_f = 1.0,                 # Fluid dynamic viscosity [Pa.s]
            rho_s = Constant(10.0E3),   # Solid density [kg/m3]
            mu_s = Constant(0.5E6),     # Solid shear modulus or 2nd Lame Coef. [Pa]
            nu_s = Constant(0.4),       # Solid Poisson ratio [-]
            lambda_s = 1e6,             # Solid Young's modulus [Pa]
            Um = 1.0,                   # Max. velocity inlet (CFD1:0.2, CFD2:1.0, CDF3:2.0) [m/s]
            D = 0.1,                    # Turek flag specific
            H = 0.41,                   # Turek flag specific
            L = 2.5,                    # Turek flag specific
            folder = "TF_cfd_results",  # Name of the results fulter
            solid = "no_solid",         # Do not solve for the solid
            extrapolation = "no_extrapolation")) # No displacement to extrapolate

    # Have to use list since we are changing the dictionary
    return default_variables


def get_mesh_domain_and_boundaries(args, **namespace):
    mesh = Mesh(path.join("mesh", "base0.xml"))

    # Define the boundaries
    Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 0))
    Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0], 2.5)))
    Wall = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.41) or near(x[1], 0)))
    Bar = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.21)) or near(x[1], 0.19)
                        or near(x[0], 0.6))
    Circle = AutoSubDomain(lambda x: "on_boundary" and (((x[0] - 0.2) * (x[0] - 0.2) +
                                                         (x[1] - 0.2) * (x[1] - 0.2) <
                                                         0.0505 * 0.0505)))
    Barwall = AutoSubDomain(lambda x: "on_boundary" and (((x[0] - 0.2) * (x[0] - 0.2) +
                                                          (x[1] - 0.2) * (x[1] - 0.2) <
                                                          0.0505 * 0.0505) and x[1] >=
                                                         0.19 and x[1] <= 0.21 and x[0] >
                                                         0.2))

    # Mark the boundaries
    Allboundaries = DomainBoundary()
    boundaries = MeshFunction("size_t", mesh, 1)
    boundaries.set_all(0)
    Allboundaries.mark(boundaries, 1)
    Wall.mark(boundaries, 2)
    Inlet.mark(boundaries, 3)
    Outlet.mark(boundaries, 4)
    Bar.mark(boundaries, 5)
    Circle.mark(boundaries, 6)
    Barwall.mark(boundaries, 7)

    # Define the domain
    Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24 <= x[0] <= 0.6)
    domains = MeshFunction("size_t", mesh, 2)
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
    dis_x = []
    dis_y = []
    Drag_list = []
    Lift_list = []
    Time_list = []

    return dict(u_file=u_file, d_file=d_file, p_file=p_file, dis_x=dis_x, dis_y=dis_y,
                Drag_list=Drag_list, Lift_list=Lift_list, Time_list=Time_list)


class Inlet(UserExpression):
    def __init__(self, Um, H, **kwargs):
        self.Um = Um
        self.H = H
        self.factor = 0
        super().__init__(**kwargs)

    def update(self, t):
        if t < 2:
            self.factor = 0.5 * (1 - np.cos(t * np.pi / 2)) * 1.5 * self.Um

    def eval(self, value, x):
        value[0] = self.factor * x[1] * (self.H - x[1]) / ((self.H / 2.0)**2)
        value[1] = 0

    def value_shape(self):
        return (2,)


def create_bcs(DVP, dvp_, Um, H, v_deg, boundaries, extrapolation_sub_type, **namespace):
    # Create inlet expression
    inlet = Inlet(Um, H, degree=v_deg)

    # Fluid velocity conditions
    u_inlet = DirichletBC(DVP.sub(1), inlet, boundaries, 3)
    u_wall = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 2)
    u_circ = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 6)  # No slip on geometry in fluid
    u_barwall = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 7)  # No slip on geometry in fluid

    # Pressure Conditions
    p_out = DirichletBC(DVP.sub(2), 0, boundaries, 4)

    # Assemble boundary conditions
    bcs = [u_wall, u_inlet, u_circ, u_barwall, p_out]

    return dict(bcs=bcs, inlet=inlet)


def pre_solve(t, inlet, **semimp_namespace):
    """Update boundary conditions"""
    inlet.update(t)


def after_solve(t, dvp_, n, Drag_list, Lift_list, Time_list, save_step, counter, u_file,
                p_file, d_file, mu_f, **namespace):
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    if counter % save_step == 0:
        d = dvp_["n"].sub(0, deepcopy=True)
        v = dvp_["n"].sub(1, deepcopy=True)
        p = dvp_["n"].sub(2, deepcopy=True)
        p_file.write(p, t)
        d_file.write(d, t)
        u_file.write(v, t)

    Dr = -assemble((sigma(v, p, d, mu_f)*n)[0]*ds(6))
    Li = -assemble((sigma(v, p, d, mu_f)*n)[1]*ds(6))
    Dr += -assemble((sigma(v("+"), p("+"), d("+"), mu_f)*n("+"))[0]*dS(5))
    Li += -assemble((sigma(v("+"), p("+"), d("+"), mu_f)*n("+"))[1]*dS(5))
    Drag_list.append(Dr)
    Lift_list.append(Li)
    Time_list.append(t)


def post_process(Det_list, Drag_list, Lift_list, Time_list, folder, **namespace):
    if MPI.rank(MPI.comm_world) == 0:
        np.savetxt(path.join(folder, 'Lift.txt'), Lift_list, delimiter=',')
        np.savetxt(path.join(folder, 'Drag.txt'), Drag_list, delimiter=',')
        np.savetxt(path.join(folder, 'Time.txt'), Time_list, delimiter=',')
