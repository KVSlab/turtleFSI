# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
Demo file to illustrate the essential features needed to set up a problem file
using turtleFSI. The following problem simulate the deformation of an elastic
turtle immersed in a pulsative flow. The turtle's head and tail are kept still
while the rest of the body, wings and legs are free to move with the flow.
The inlet flow (left to right) is initially gradually increased to a maximum value
to, then, fluctuates as a sine function with time.

The fluid flow is approximated by solving the incompressible Navier-Stokes equations.
The elastic deformation of the turtle is solved assuming a nonlinear elastic
Saint Venant-Kirchhoff constitutive model.

Note: this setup has no aim to reproduce any realistic or physical problem.
"""

from dolfin import *
import numpy as np
from os import path
from turtleFSI.problems import *


def set_problem_parameters(default_variables, **namespace):
    # Overwrite default values
    default_variables.update(dict(
        T=15,                          # End time [s]
        dt=0.005,                      # Time step [s]
        theta=0.505,                   # Theta value (0.5 + dt), shifted Crank-Nicolson scheme
        Um=1.0,                        # Max. velocity inlet [m/s]
        rho_f=1.0E3,                   # Fluid density [kg/m3]
        mu_f=1.0,                      # Fluid dynamic viscosity [Pa.s]
        rho_s=1.0E3,                   # Solid density [kg/m3]
        mu_s=5.0E4,                    # Solid shear modulus or 2nd Lame Coef. [Pa]
        lambda_s=4.5E5,                # Solid 1st Lame Coef. [Pa]
        nu_s=0.45,                     # Solid Poisson ratio [-]
        dx_f_id=1,                     # ID of marker in the fluid domain
        dx_s_id=2,                     # ID of marker in the solid domain
        extrapolation="biharmonic",    # Laplace, elastic, biharmonic, no-extrapolation
        extrapolation_sub_type="bc1",  # ["constant", "small_constant", "volume", "volume_change", "bc1", "bc2"]
        recompute=15,                  # Recompute the Jacobian matrix every "recompute" Newton iterations
        folder="turtle_demo_results"),  # Mame of the folder to save the data
        save_step=1                    # Frequency of data saving
    )

    return default_variables


def get_mesh_domain_and_boundaries(args, **namespace):

    mesh_folder = path.join(path.dirname(path.abspath(__file__)), "..", "mesh", "turtle_demo")

    # In this example, the mesh and markers are stored in the 3 following files
    mesh_path = path.join(mesh_folder, "turtle_mesh.xdmf")     # mesh geometry
    domains_marker_path = path.join(mesh_folder, "mc.xdmf")    # marker over the elements (domains)
    boundaries_marker_path = path.join(mesh_folder, "mf.xdmf")  # markers of the segments (boundaries)

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
    # Marker values ranging from 11 to 15.
    mesh_collection = MeshValueCollection("size_t", mesh, mesh.geometry().dim() - 1)
    xdmf = XDMFFile(MPI.comm_world, boundaries_marker_path)
    xdmf.read(mesh_collection)
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mesh_collection)

    return mesh, domains, boundaries


class Inlet(UserExpression):
    def __init__(self, Um, **kwargs):
        self.t = 0.0
        self.t_ramp = 0.5  # time to ramp-up to max inlet velocity (from 0 to Um)
        self.Um = Um       # Max. velocity inlet [m/s]
        super().__init__(**kwargs)

    def update(self, t):
        self.t = t
        if self.t < self.t_ramp:
            self.value = self.Um * np.abs(np.cos(self.t/self.t_ramp*np.pi)-1)/2  # ramp-up the inlet velocity
            # print(self.value)
        else:
            Um_min = self.Um/6  # lower velocity during oscillations
            self.value = (self.Um-Um_min) * np.abs(np.cos(self.t/self.t_ramp*np.pi)-1)/2 + Um_min
            # print(self.value)

    def eval(self, value, x):
        value[0] = self.value
        value[1] = 0

    def value_shape(self):
        return (2,)


def create_bcs(DVP, boundaries, Um, v_deg, extrapolation_sub_type, **namespace):
    if MPI.rank(MPI.comm_world) == 0:
        print("Create bcs")

    inlet = Inlet(Um, degree=v_deg)
    noslip = ((0.0, 0.0))

    # Segments indices (make sure of the consistency with the boundary file)
    bottom_id = 11             # segments at the bottom of the model
    outlet_id = 12             # segments at the outlet (right wall) of the model
    top_id = 13                # segments at the top (right wall) of the model
    inlet_id = 14              # segments at the inlet (left wall) of the model
    turtle_head_tail_id = 15   # segments along the head and tail of the turtle

    # Fluid velocity boundary conditions
    u_inlet = DirichletBC(DVP.sub(1), inlet, boundaries, inlet_id)
    u_bot = DirichletBC(DVP.sub(1).sub(1), (0.0), boundaries, bottom_id)
    u_top = DirichletBC(DVP.sub(1).sub(1), (0.0), boundaries, top_id)
    u_head_tail = DirichletBC(DVP.sub(1), noslip, boundaries, turtle_head_tail_id)

    # Pressure boundary conditions
    p_outlet = DirichletBC(DVP.sub(2), (0.0), boundaries, outlet_id)

    bcs = [u_bot, u_top, u_inlet, p_outlet,  u_head_tail]

    # Mesh uplifting boundary conditions
    d_inlet = DirichletBC(DVP.sub(0), noslip, boundaries, inlet_id)
    d_bot = DirichletBC(DVP.sub(0), noslip, boundaries, bottom_id)
    d_top = DirichletBC(DVP.sub(0), noslip, boundaries, top_id)
    d_outlet = DirichletBC(DVP.sub(0), noslip, boundaries, outlet_id)
    d_head_tail = DirichletBC(DVP.sub(0), noslip, boundaries, turtle_head_tail_id)

    for i in [d_bot, d_top, d_outlet, d_inlet, d_head_tail]:
        bcs.append(i)

    return dict(bcs=bcs, inlet=inlet)


def initiate(dvp_, folder, **namespace):
    # Files for storing results
    u_file = XDMFFile(MPI.comm_world, path.join(folder, "velocity.xdmf"))
    d_file = XDMFFile(MPI.comm_world, path.join(folder, "d.xdmf"))
    p_file = XDMFFile(MPI.comm_world, path.join(folder, "pressure.xdmf"))
    for tmp_t in [u_file, d_file, p_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["rewrite_function_mesh"] = False

    # Extract the variables to save
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    # Save the data to the simulation time=0.0
    d_file.write(d, 0.0)
    u_file.write(v, 0.0)
    p_file.write(p, 0.0)

    return dict(u_file=u_file, d_file=d_file, p_file=p_file)


def pre_solve(t, inlet, **namespace):
    # Update the time variable used for the inlet boundary condition
    inlet.update(t)
    return {}


def post_solve(t, dvp_, counter, u_file, p_file, d_file, save_step, **namespace):
    if counter % save_step == 0:
        d = dvp_["n"].sub(0, deepcopy=True)
        v = dvp_["n"].sub(1, deepcopy=True)
        p = dvp_["n"].sub(2, deepcopy=True)
        p_file.write(p, t)
        d_file.write(d, t)
        u_file.write(v, t)
    return {}
