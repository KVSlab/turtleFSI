# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import *
import numpy as np
from os import path

from turtleFSI.problems import *
from turtleFSI.modules import *


def set_problem_parameters(default_variables, **namespace):
    # Overwrite or add new variables to 'default_variables'
    default_variables.update(dict(
        # Temporal variables
        T=30,                         # End time [s]
        dt=0.01,                      # Time step [s]
        theta=0.51,                    # Temporal scheme

        # Physical constants ('FSI 3')
        Um=2.0,                       # Max. velocity inlet, CDF3: 2.0 [m/s]
        rho_f=1.0e3,                  # Fluid density [kg/m3]
        mu_f=1.0,                     # Fluid dynamic viscosity [Pa.s]
        rho_s=1.0e3,                  # Solid density[kg/m3]
        nu_s=0.4,                     # Solid Poisson ratio [-]
        mu_s=2.0e6,                   # Shear modulus, CSM3: 0.5E6 [Pa]
        lambda_s=4e6,                 # Solid 1st Lame Coef. [Pa]

        # Problem specific
        folder="TF_fsi_results",      # Name of the results folder
        extrapolation="biharmonic",   # No displacement to extrapolate
        extrapolation_sub_type="bc2",  # Biharmonic type
        bc_ids=[2, 3, 4, 6],          # Ids for extrapolation weak form

        # Solver settings
        recompute=1,

        # Geometric variables
        R=0.05,                       # Radius of the circle
        H=0.41,                       # Total height
        L=2.5,                        # Length of domain
        f_L=0.35,                     # Length of the flag
        f_H=0.02,                     # Height of the flag
        c_x=0.2,                      # Center of the circle x-direction
        c_y=0.2))                     # Center of the circle y-direction

    #from IPython import embed; embed()
    default_variables["compiler_parameters"].update({"quadrature_degree": 5})
    return default_variables


def get_mesh_domain_and_boundaries(R, H, L, f_L, f_H, c_x, c_y, **namespace):
    # Read mesh
    mesh = Mesh(path.join(path.dirname(path.abspath(__file__)), "..", "mesh",
                          "TF_fsi.xml.gz"))

    # Define boundaries
    Inlet = AutoSubDomain(lambda x: near(x[0], 0))
    Outlet = AutoSubDomain(lambda x: (near(x[0], L)))
    Wall = AutoSubDomain(lambda x: (near(x[1], H) or near(x[1], 0)))
    Bar = AutoSubDomain(lambda x: (near(x[1], c_y + f_H / 2) or
                                   near(x[1], c_y - f_H / 2) or
                                   near(x[0], c_x + R + f_L)))

    def circle(x): return (x[0] - c_x)**2 + (x[1] - c_y)**2 < R**2 + DOLFIN_EPS*1e5
    Circle = AutoSubDomain(circle)
    Barwall = AutoSubDomain(lambda x: circle(x) and
                            x[1] >= c_y - f_H / 2 and
                            x[1] <= c_y + f_H / 2 and x[0] > c_x)

    # Mark boundaries
    Allboundaries = DomainBoundary()
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundaries.set_all(0)
    Allboundaries.mark(boundaries, 1)
    Wall.mark(boundaries, 2)
    Inlet.mark(boundaries, 3)
    Outlet.mark(boundaries, 4)
    Bar.mark(boundaries, 5)
    Circle.mark(boundaries, 6)
    Barwall.mark(boundaries, 7)

    # Define and mark domains
    Bar_area = AutoSubDomain(lambda x: c_y - f_H / 2 <= x[1] <= c_y + f_H / 2 and
                             c_x <= x[0] <= c_x + R + f_L)
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(1)
    Bar_area.mark(domains, 2)

    return mesh, domains, boundaries


def initiate(dvp_, mesh, folder, c_x, c_y, R, f_L, **namespace):
    # Files for storing results
    u_file = XDMFFile(MPI.comm_world, path.join(folder, "velocity.xdmf"))
    d_file = XDMFFile(MPI.comm_world, path.join(folder, "d.xdmf"))
    p_file = XDMFFile(MPI.comm_world, path.join(folder, "pressure.xdmf"))
    for tmp_t in [u_file, d_file, p_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["rewrite_function_mesh"] = False

    # Store initial condition
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)
    d_file.write(d)
    u_file.write(v)
    p_file.write(p)

    # Coord to sample
    for coord in mesh.coordinates():
        if coord[0] == c_x + R + f_L and (c_y - 0.001 <= coord[1] <= c_y + 0.001):
            break

    # Lists to hold results
    dis_x = []
    dis_y = []
    Drag_list = []
    Lift_list = []
    Time_list = []

    return dict(u_file=u_file, d_file=d_file, p_file=p_file, dis_x=dis_x,
                dis_y=dis_y, Drag_list=Drag_list, Lift_list=Lift_list,
                Time_list=Time_list, coord=coord)


class Inlet(UserExpression):
    def __init__(self, Um, H, **kwargs):
        self.Um = 1.5 * Um
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


def create_bcs(DVP, v_deg, Um, H, boundaries, extrapolation_sub_type, **namespace):
    inlet = Inlet(Um, H, degree=v_deg)

    # Fluid velocity conditions
    u_inlet = DirichletBC(DVP.sub(1), inlet, boundaries, 3)
    u_wall = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 2)
    u_circ = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 6)
    u_barwall = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 7)

    # Pressure Conditions
    p_out = DirichletBC(DVP.sub(2), 0, boundaries, 4)

    # Assemble boundary conditions
    bcs = [u_wall, u_inlet, u_circ, u_barwall, p_out]

    # Boundary conditions on the displacement / extrapolation
    if extrapolation_sub_type != "bc2":
        d_wall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 2)
        d_inlet = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 3)
        d_outlet = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 4)
        d_circle = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 6)
        d_barwall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 7)
        for i in [d_wall, d_inlet, d_outlet, d_circle, d_barwall]:
            bcs.append(i)

    else:
        w_wall = DirichletBC(DVP.sub(3).sub(1), (0.0), boundaries, 2)
        w_inlet = DirichletBC(DVP.sub(3).sub(0), (0.0), boundaries, 3)
        w_outlet = DirichletBC(DVP.sub(3).sub(0), (0.0), boundaries, 4)
        w_circle = DirichletBC(DVP.sub(3), ((0.0, 0.0)), boundaries, 6)
        w_barwall = DirichletBC(DVP.sub(3), ((0.0, 0.0)), boundaries, 7)

        d_wall = DirichletBC(DVP.sub(0).sub(1), (0.0), boundaries, 2)
        d_inlet = DirichletBC(DVP.sub(0).sub(0), (0.0), boundaries, 3)
        d_outlet = DirichletBC(DVP.sub(0).sub(0), (0.0), boundaries, 4)
        d_circle = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 6)
        d_barwall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 7)

        for i in [w_wall, w_inlet, w_outlet, w_circle, w_barwall,
                  d_wall, d_inlet, d_outlet, d_circle, d_barwall]:
            bcs.append(i)

    return dict(bcs=bcs, inlet=inlet)


def pre_solve(t, inlet, **namespace):
    """Update boundary conditions"""
    inlet.update(t)
    return {}


def post_solve(t, DVP, dvp_, coord, dis_x, dis_y, Drag_list, Lift_list, mu_f, n,
               counter, u_file, p_file, d_file, verbose, save_step, Time_list, ds, dS,
               **namespace):
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    if counter % save_step == 0:
        p_file.write(p, t)
        d_file.write(d, t)
        u_file.write(v, t)

    # Compute drag and lift
    Dr = -assemble((sigma(v, p, d, mu_f)*n)[0]*ds(6))
    Li = -assemble((sigma(v, p, d, mu_f)*n)[1]*ds(6))
    Dr += -assemble((sigma(v("+"), p("+"), d("+"), mu_f)*n("+"))[0]*dS(5))
    Li += -assemble((sigma(v("+"), p("+"), d("+"), mu_f)*n("+"))[1]*dS(5))

    # Append results
    Drag_list.append(Dr)
    Lift_list.append(Li)
    Time_list.append(t)
    dis_x.append(d(coord)[0])
    dis_y.append(d(coord)[1])

    # Print
    if MPI.rank(MPI.comm_world) == 0 and verbose:
        print("Distance x: {:e}".format(dis_x[-1]))
        print("Distance y: {:e}".format(dis_y[-1]))
        print("Drag: {:e}", Drag_list[-1])
        print("Lift: {:e}", Lift_list[-1])

    return {}


def finished(folder, dis_x, dis_y, Drag_list, Lift_list, Time_list,
             **namespace):
    if MPI.rank(MPI.comm_world) == 0:
        np.savetxt(path.join(folder, 'Lift.txt'), Lift_list, delimiter=',')
        np.savetxt(path.join(folder, 'Drag.txt'), Drag_list, delimiter=',')
        np.savetxt(path.join(folder, 'Time.txt'), Time_list, delimiter=',')
        np.savetxt(path.join(folder, 'dis_x.txt'), dis_x, delimiter=',')
        np.savetxt(path.join(folder, 'dis_y.txt'), dis_y, delimiter=',')
