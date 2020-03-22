# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""Problem file for running the method of manufactured solution for the fluid
part of the equations. The manufactured solution is adapted from J. L. Guermond et al. 2005
"""

from dolfin import *
import numpy as np
from os import path

from turtleFSI.problems import *
from turtleFSI.modules import *


def set_problem_parameters(default_variables, **namespace):
    # Overwrite or add new variables to 'default_variables'
    default_variables.update(dict(
        # Temporal variables
        T=0.00005,                # End time [s]
        dt=0.00001,               # Time step [s]
        theta=0.5,                # Temporal scheme

        # Physical constants
        rho_s=1.0,                # Fluid density [kg/m3]
        mu_s=1.0,                 # Fluid dynamic viscosity [Pa.s]
        nu_s=1.0,
        lambda_s=1.0,

        # Problem specific
        folder="MMS_solid",       # Name of the results folder
        fluid="no_fluid",         # Do not solve for the solid
        extrapolation="no_extrapolation",  # No displacement to extrapolate

        # Geometric variables
        N=40,                     # Mesh resolution

        # MMS, From J. L. Guermond et al. 2005
        eps = 1e-4,
        dx_mms = "pi * cos(t_e) * sin(2 * x[1]) * sin(x[0]) * sin(x[0]) + eps + x[1] / pi",
        dy_mms = "- pi * cos(t_e) * sin(2 * x[0]) * sin(x[1]) * sin(x[1]) + eps",
        ux_mms = "- pi * sin(t_e) * sin(2 * x[1]) * sin(x[0]) * sin(x[0])",
        uy_mms = "pi * sin(t_e) * sin(2 * x[0]) * sin(x[1]) * sin(x[1])"))

    return default_variables


def get_mesh_domain_and_boundaries(N, **namespace):
    # Load and refine mesh
    mesh = RectangleMesh(Point(0, 0), Point(np.pi, np.pi), N, N)

    # Mark the boundaries
    Allboundaries = DomainBoundary()
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundaries.set_all(2)

    # Define the domain
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(2)

    return mesh, domains, boundaries


def initiate(F_solid_linear, theta, dx_mms, dy_mms, dt, lambda_s, mu_s,
             d_deg, eps, mesh, psi, dx_s, dvp_, **namespace):
    # Exact solution expressions
    dx_e = Expression(dx_mms, eps=eps, degree=v_deg, t_e=0)
    dy_e = Expression(dy_mms, eps=eps, degree=v_deg, t_e=0)
    t_e_n = Constant(0.0)
    t_e_n_1 = Constant(-dt)

    # Add F to variational formulation
    x = SpatialCoordinate(mesh)
    for th, t_n in [(theta, "t_e_n"), ((1 - theta), "t_e_n_1")]:
        d_vec = as_vector([eval(dx_mms.replace("t_e", t_n)),
                           eval(dy_mms.replace("t_e", t_n))])
        t_n = eval(t_n)

        f_tmp = (diff(u_vec, t_n), t_n)
                + div(Piola1(d_["n"], lambda_s, mu_s))

        F_solid_linear -= th * inner(f_tmp, psi)*dx_f

    # Set manufactured solution as initial condition for n-1 (t = 0)
    assign(dvp_["n-1"].sub(0).sub(0), project(ux_e, dvp_["n"].sub(1).sub(0).function_space().collapse()))
    assign(dvp_["n-1"].sub(0).sub(1), project(uy_e, dvp_["n"].sub(1).sub(1).function_space().collapse()))

    return dict(ux_e=ux_e, uy_e=uy_e, t_e_n=t_e_n, t_e_n_1=t_e_n_1,
                F_solid_linear=F_solid_linear)


def create_bcs(DVP, ux_e, uy_e, p_e, boundaries, **namespace):
    # Fluid velocity conditions
    bc_d = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 1)
    bc_ux = DirichletBC(DVP.sub(1).sub(0), ux_e, boundaries, 1)
    bc_uy = DirichletBC(DVP.sub(1).sub(1), uy_e, boundaries, 1)
    bc_p = DirichletBC(DVP.sub(2), p_e, boundaries, 1)

    return dict(bcs=[bc_d, bc_ux, bc_uy, bc_p])


def pre_solve(t_e_n, t_e_n_1, t, ux_e, uy_e, p_e, dt, **namespace):
    """Update boundary conditions"""
    t_e_n.assign(t)
    t_e_n_1.assign(t - dt)

    ux_e.t_e = t
    uy_e.t_e = t
    p_e.t_e = t


def finished(ux_e, uy_e, p_e, dvp_, T, dt, mesh, **namespace):
    # Store results when the computation is finished
    # FIXME
    V = FunctionSpace(mesh, "CG", 5)
    ux = interpolate(ux_e, V) #dvp_["n"].sub(1).sub(0).function_space().collapse())
    uy = interpolate(uy_e, V) #dvp_["n"].sub(1).sub(1).function_space().collapse())
    p = interpolate(p_e, V) #dvp_["n"].sub(2).function_space().collapse())

    print(" ")
    print("T      {0:.10e}".format(T))
    print("dt     {0:.10e}".format(dt))
    print("dx     {0:.10e}".format(mesh.hmin()))
    print("L2 norm (ux-uxmms) {0:.10e}".format(errornorm(ux, dvp_["n"].sub(1).sub(0), norm_type="l2",
                                                        degree_rise=5)))
    print("L2 norm (uy-uymms) {0:.10e}".format(errornorm(uy, dvp_["n"].sub(1).sub(1), norm_type="l2",
                                                        degree_rise=5)))
    print("L2 norm (p-pmms)   {0:.10e}".format(errornorm(p, dvp_["n"].sub(2), norm_type="l2",
                                                        degree_rise=5)))
    print(" ")
