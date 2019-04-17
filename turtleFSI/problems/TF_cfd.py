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
            folder = "TF_cfd_results")) # Name of the results fulter

    # Have to use list since we are changing the dictionary
    return default_variables


def get_mesh_domain_and_boundaries(args, **namespace):
    mesh_file = Mesh(path.join("mesh", "base0.xml"))

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
    boundaries = MeshFunction("size_t", mesh_file, 1)
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
    domains = MeshFunction("size_t", mesh_file, 2)
    domains.set_all(1)

    return mesh_file, domains, boundaries


class Inlet(Expression):
    def __init__(self, Um, **kwargs):
        self.Um = Um
        self.value = 0

    def update(self, t):
        if t < 2:
            self.value = (0.5 * (1 - np.cos(t * np.pi / 2)) * 1.5 * self.Um * x[1]
                          * (H - x[1]) / ((H / 2.0)**2))

    def eval(self, value, x):
        value[0] = self.value
        value[1] = 0

    def value_shape(self):
        return (2,)


def create_bcs(DVP, args, dvp_, n, k, Um, H, boundaries, inlet, **semimp_namespace):
    inlet = Inlet(Um, degree=v_deg)
    print("Create bcs")
    # Fluid velocity conditions
    u_inlet = DirichletBC(DVP.sub(1), inlet, boundaries, 3)
    u_wall = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 2)
    u_circ = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 6)  # No slip on geometry in fluid
    u_barwall = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 7)  # No slip on geometry in fluid

    # Pressure Conditions
    p_out = DirichletBC(DVP.sub(2), 0, boundaries, 4)

    # Assemble boundary conditions
    bcs = [u_wall, u_inlet, u_circ, u_barwall,
           p_out]

    # if DVP.num_sub_spaces() == 4:
    if args.bitype == "bc1":
        d_wall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 2)
        d_inlet = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 3)
        d_outlet = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 4)
        d_circle = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 6)
        d_barwall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 7)  # No slip on geometry in fluid
        for i in [d_wall, d_inlet, d_outlet, d_circle, d_barwall]:
            bcs.append(i)

    if args.bitype == "bc2":
        w_wall = DirichletBC(DVP.sub(0).sub(1), (0.0), boundaries, 2)
        w_inlet = DirichletBC(DVP.sub(0).sub(0), (0.0), boundaries, 3)
        w_outlet = DirichletBC(DVP.sub(0).sub(0), (0.0), boundaries, 4)
        w_circle = DirichletBC(DVP.sub(0).sub(1), (0.0), boundaries, 6)
        w_barwall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 7)  # No slip on geometry in fluid

        d_wall = DirichletBC(DVP.sub(0).sub(1), (0.0), boundaries, 2)
        d_inlet = DirichletBC(DVP.sub(0).sub(0), (0.0), boundaries, 3)
        d_outlet = DirichletBC(DVP.sub(0).sub(0), (0.0), boundaries, 4)
        d_circle = DirichletBC(DVP.sub(0).sub(1), (0.0), boundaries, 6)
        d_barwall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 7)

        for i in [w_wall, w_inlet, w_outlet, w_circle, w_barwall,
                  d_wall, d_inlet, d_outlet, d_circle, d_barwall]:
            bcs.append(i)

    return dict(bcs=bcs, inlet=inlet)


def initiate(P, v_deg, d_deg, p_deg, dt, theta, dvp_, args, Det_list, refi, mesh_file,
             mesh_name, **semimp_namespace):
    #exva = args.extravar
    #extype = args.extype
    #bitype = args.bitype

    #newfolder = 
    #if args.extravar == "alfa":
    #    path = "results/TF_CFD/%(exva)s_%(extype)s/dt-%(dt)g_theta-%(theta)g/%(mesh_name)s_refine_%(refi)d_v_deg_%(v_deg)s_d_deg_%(d_deg)s_p_deg_%(p_deg)s" % vars()
    #if args.extravar == "biharmonic" or args.extravar == "laplace" or args.extravar == "elastic":
    #    path = "results/TF_CFD/%(exva)s_%(bitype)s/dt-%(dt)g_theta-%(theta)g/%(mesh_name)s_refine_%(refi)d_v_deg_%(v_deg)s_d_deg_%(d_deg)s_p_deg_%(p_deg)s" % vars()

    u_file = XDMFFile(mpi_comm_world(), path.join("results", "TF_CFD", "velocity.xdmf"))
    d_file = XDMFFile(mpi_comm_world(), path.join("results", "TF_CFD", "d.xdmf"))
    p_file = XDMFFile(mpi_comm_world(), path.join("results", "TF_CFD", "pressure.xdmf"))
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

    dis_x = []
    dis_y = []
    Drag_list = []
    Lift_list = []
    Time_list = []

    return dict(u_file=u_file, d_file=d_file, p_file=p_file, newfolder=newfolder)



def pre_solve(t, inlet, **semimp_namespace):
    """Update boundary conditions"""
    inlet.update(t)


def after_solve(t, P, DVP, dvp_, n, coord, dis_x, dis_y, Drag_list, Lift_list, Det_list,
                counter, dvp_file, u_file, p_file, d_file, path, **namespace):
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

    Dr = -assemble((sigma_f_new(v, p, d, mu_f)*n)[0]*ds(6))
    Li = -assemble((sigma_f_new(v, p, d, mu_f)*n)[1]*ds(6))
    Dr += -assemble((sigma_f_new(v("+"), p("+"), d("+"), mu_f)*n("+"))[0]*dS(5))
    Li += -assemble((sigma_f_new(v("+"), p("+"), d("+"), mu_f)*n("+"))[1]*dS(5))
    Drag_list.append(Dr)
    Lift_list.append(Li)
    Time_list.append(t)

    dsx = d(coord)[0]
    dsy = d(coord)[1]
    dis_x.append(dsx)
    dis_y.append(dsy)

    return {}


def post_process(path, T, dt, Det_list, dis_x, dis_y, Drag_list, Lift_list, Time_list,
                 args, simtime, v_deg, p_deg, d_deg, dvp_file, **semimp_namespace):
    if MPI.rank(mpi_comm_world()) == 0:
        np.savetxt(path + '/Min_J.txt', Det_list, delimiter=',')
        np.savetxt(path + '/Lift.txt', Lift_list, delimiter=',')
        np.savetxt(path + '/Drag.txt', Drag_list, delimiter=',')
        np.savetxt(path + '/Time.txt', Time_list, delimiter=',')
        np.savetxt(path + '/dis_x.txt', dis_x, delimiter=',')
        np.savetxt(path + '/dis_y.txt', dis_y, delimiter=',')

    return {}
