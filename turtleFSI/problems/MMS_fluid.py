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
        theta=0.50001,            # Temporal scheme

        # Physical constants
        rho_f=1.0,                # Fluid density [kg/m3]
        mu_f=1.0,                 # Fluid dynamic viscosity [Pa.s]

        # Problem specific
        folder="MMS_fluid",       # Name of the results folder
        solid="no_solid",         # Do not solve for the solid
        extrapolation="no_extrapolation",  # No displacement to extrapolate

        # Geometric variables
        N=40,                     # Mesh resolution

        # MMS, From J. L. Guermond et al. 2005
        eps = 1e-4,
        #ux_mms = "pi * exp(3*t_e) * sin(2 * x[1]*x[1]) * sin(x[0]) * sin(x[0]) + eps + x[1] / pi + t_e + pow(t_e, 2)",
        #uy_mms = "- pi * exp(3*t_e) * sin(2 * x[0]*x[0]) * sin(x[1]) * sin(x[1]) + eps + t_e + pow(t_e, 2)",
        #p_mms = "exp(3*t_e) * cos(x[0]) * sin(x[1]) + eps + t_e + pow(t_e, 2)"))
        ux_mms = "exp(3*t_e) + t_e + t_e * t_e * t_e + sin(t_e) + x[1] + x[0]",
        uy_mms = "exp(3*t_e) + t_e + t_e * t_e * t_e + sin(t_e) - x[1] - x[0]",
        p_mms = "3*exp(3*t_e) + t_e + t_e * t_e * t_e + sin(t_e)"))

    return default_variables


def get_mesh_domain_and_boundaries(N, **namespace):
    # Load and refine mesh
    mesh = RectangleMesh(Point(0, 0), Point(np.pi, np.pi), N, N)

    # Mark the boundaries
    Allboundaries = DomainBoundary()
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundaries.set_all(1)

    # Define the domain
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(1)

    return mesh, domains, boundaries


def initiate(F_fluid_nonlinear, theta, ux_mms, uy_mms, p_mms, dt,
             v_deg, p_deg, eps, mesh, mu_f, psi, dx_f, dvp_, **namespace):
    # Exact solution expressions
    ux_e = Expression(ux_mms, eps=eps, degree=v_deg, t_e=0)
    uy_e = Expression(uy_mms, eps=eps, degree=v_deg, t_e=0)
    p_e = Expression(p_mms, eps=eps, degree=p_deg, t_e=0)
    t_e_n = Constant(0.0)
    t_e_n_1 = Constant(-dt)
    fixme_variabel = 0

    # Add F to variational formulation
    x = SpatialCoordinate(mesh)
    factor = 1
    for th, t_n in [(theta*factor, "t_e_n"), ((1 - theta)*factor, "t_e_n_1")]:
        u_vec = as_vector([eval(ux_mms.replace("t_e", t_n)),
                           eval(uy_mms.replace("t_e", t_n))])
        p_ = eval(p_mms.replace("t_e", t_n))
        t_n = eval(t_n)

        f_tmp = (diff(u_vec, t_n)
                + dot(u_vec, nabla_grad(u_vec))
                + div(p_ * Identity(2))
                - mu_f*div(grad(u_vec)))

        # Add term to equation
        if th != 0:
            F_fluid_nonlinear -= Constant(th) * inner(f_tmp, psi)*dx_f

        if t_n.name() == "f_31":
            print("Tetha 1:", th)
            if th != 0:
                fixme_variabel += th * inner(f_tmp, psi)*dx_f
        else:
            print("Tetha 2:", th)
            if th != 0:
                fixme_variabel += th * inner(f_tmp, psi)*dx_f

    # Set manufactured solution as initial condition for n-1 (t = 0)
    assign(dvp_["n-1"].sub(1).sub(0), project(ux_e, dvp_["n"].sub(1).sub(0).function_space().collapse()))
    assign(dvp_["n-1"].sub(1).sub(1), project(uy_e, dvp_["n"].sub(1).sub(1).function_space().collapse()))
    assign(dvp_["n-1"].sub(2), project(p_e, dvp_["n"].sub(2).function_space().collapse()))

    return dict(ux_e=ux_e, uy_e=uy_e, p_e=p_e, t_e_n=t_e_n, t_e_n_1=t_e_n_1,
                F_fluid_nonlinear=F_fluid_nonlinear,
                fixme_variabel=fixme_variabel)


def create_bcs(DVP, ux_e, uy_e, p_e, boundaries, **namespace):
    # Fluid velocity conditions
    bc_d = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 1)
    bc_ux = DirichletBC(DVP.sub(1).sub(0), ux_e, boundaries, 1)
    bc_uy = DirichletBC(DVP.sub(1).sub(1), uy_e, boundaries, 1)
    bc_p = DirichletBC(DVP.sub(2), p_e, boundaries, 1)

    return dict(bcs=[bc_d, bc_ux, bc_uy, bc_p])


def pre_solve(t_e_n, t_e_n_1, t, ux_e, uy_e, p_e, dt, **namespace):
    """Update boundary conditions"""
    print("f", np.sum(assemble(namespace["fixme_variabel"]).get_local()))
    t_e_n.assign(t)
    t_e_n_1.assign(t - dt)

    ux_e.t_e = t
    uy_e.t_e = t
    p_e.t_e = t

"""
def post_solve(ux_e, uy_e, p_e, dvp_, T, t, dt, mesh, **namespace):
    V = FunctionSpace(mesh, "CG", 5)
    ux = interpolate(ux_e, V) #dvp_["n"].sub(1).sub(0).function_space().collapse())
    uy = interpolate(uy_e, V) #dvp_["n"].sub(1).sub(1).function_space().collapse())
    p = interpolate(p_e, V) #dvp_["n"].sub(2).function_space().collapse())

    ux_t = Function(V)

    print(" ")
    print("t      {0:.10e}".format(t))
    #print("dt     {0:.10e}".format(dt))
    #print("dx     {0:.10e}".format(mesh.hmin()))
    print("L2 norm (ux-uxmms) {0:.10e}".format(errornorm(ux, dvp_["n"].sub(1).sub(0), norm_type="l2",
                                                         degree_rise=5)))
    print("L2 norm (uy-uymms) {0:.10e}".format(errornorm(uy, dvp_["n"].sub(1).sub(1), norm_type="l2",
                                                         degree_rise=5)))
    print("L2 norm (p-pmms)   {0:.10e}".format(errornorm(p, dvp_["n"].sub(2), norm_type="l2",
                                                         degree_rise=5)))

    print("L2 norm (ux) {0:.10e}".format(errornorm(ux_t, dvp_["n"].sub(1).sub(0), norm_type="l2",
                                                   degree_rise=5)))
    print("L2 norm (uy) {0:.10e}".format(errornorm(ux_t, dvp_["n"].sub(1).sub(1), norm_type="l2",
                                                   degree_rise=5)))
    print("L2 norm (p)  {0:.10e}".format(errornorm(ux_t, dvp_["n"].sub(2), norm_type="l2",
                                                   degree_rise=5)))
    print("L2 norm (ux-mms) {0:.10e}".format(errornorm(ux_t, ux, norm_type="l2",
                                                   degree_rise=5)))
    print("L2 norm (uy-mms) {0:.10e}".format(errornorm(ux_t, uy, norm_type="l2",
                                                   degree_rise=5)))
    print("L2 norm (p-mms)  {0:.10e}".format(errornorm(ux_t, p, norm_type="l2",
                                                   degree_rise=5)))


    print(" ")
"""

def finished(ux_e, uy_e, p_e, dvp_, T, dt, mesh, **namespace):
    # Store results when the computation is finished
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
