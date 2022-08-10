# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""Problem file for running the "CSM" benchmarks in [1]. The problem is beam under load.

[1] Turek, Stefan, and Jaroslav Hron. "Proposal for numerical benchmarking of fluid-structure interaction
between an elastic object and laminar incompressible flow." Fluid-structure interaction.
Springer, Berlin, Heidelberg, 2006. 371-385."""

from dolfin import *
import numpy as np
from os import path
import stress_strain as StrStr

from turtleFSI.problems import *

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. Doesnt affect the speed
parameters["reorder_dofs_serial"] = False

def set_problem_parameters(default_variables, **namespace):
    # Overwrite default values
    E_s_val = 1E6  # Young modulus (Pa)
    nu_s_val = 0.45
    mu_s_val = E_s_val / (2 * (1 + nu_s_val))  # 0.345E6
    lambda_s_val = nu_s_val * 2. * mu_s_val / (1. - 2. * nu_s_val)

    
    default_variables.update(dict(
        # Temporal variables
        T=0.2,          # End time [s]
        dt=0.0005,       # Time step [s]
        checkpoint_step=1000, # Checkpoint frequency
        theta=0.50,     # Temporal scheme
        save_step=1,

        # Physical constants
        rho_f=1.0e3,   # Fluid density [kg/m3]
        mu_f=1.0,      # Fluid dynamic viscosity [Pa.s]

        rho_s=1.0E3,    # Solid density [kg/m3]
        mu_s=mu_s_val,     # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,      # Solid Poisson ratio [-]
        lambda_s=lambda_s_val,  # Solid 1st Lame Coef. [Pa]
        material_model="StVenantKirchoffEnergy",
        gravity=None,   # Gravitational force [m/s**2]

        # Problem specific
        dx_f_id=0,     # Id of the fluid domain
        dx_s_id=1,     # Id of the solid domain
        folder="Simple_Shear_SVK_Energy",          # Folder to store the results
        fluid="no_fluid",                 # Do not solve for the fluid
        extrapolation="no_extrapolation",  # No displacement to extrapolate
        solid_vel=0.1, # this is the velocity of the wall with prescribed displacement

        # Geometric variables
        leftEnd=0.001,
        rightEnd=0.099))   

    return default_variables


def get_mesh_domain_and_boundaries(leftEnd, rightEnd, **namespace):
    # Read mesh
    negEnd=0.001,
    posEnd=0.099
    a=Point(0.0, 0.0, 0.0)
    b=Point(0.1, 0.1, 0.1)
    mesh = BoxMesh(a, b, 6, 6, 6)
    #mesh = BoxMesh(a, b, 1, 1, 1)

    #print(dir(mesh))
    # Mark boundaries
    Lwall = AutoSubDomain(lambda x: (x[0]< negEnd))
    Rwall = AutoSubDomain(lambda x: (x[0]> posEnd))
    sideY = AutoSubDomain(lambda x: (x[1] < negEnd or x[1] > posEnd))
    sideZ = AutoSubDomain(lambda x: (x[2] < negEnd or x[2] > posEnd))

    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundaries.set_all(0)
    Lwall.mark(boundaries, 1)
    Rwall.mark(boundaries, 2)
    sideY.mark(boundaries, 3)
    sideZ.mark(boundaries, 4)


    # Mark domain
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(1)

    return mesh, domains, boundaries

class PrescribedDisp(UserExpression):
    def __init__(self, solid_vel, **kwargs):
        self.solid_vel = solid_vel
        self.factor = 0

        super().__init__(**kwargs)

    def update(self, t):
        self.factor = t * self.solid_vel
        print('displacement = ', self.factor)

    def eval(self, value,x):
        value[0] = 0
        value[1] = self.factor * x[0]/0.1
        value[2] = 0
        #print('eval',x)

    def value_shape(self):
        return (3,)


def create_bcs(DVP,d_deg,solid_vel, boundaries, **namespace):
    # Clamp on the left hand side
    u_lwall = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, 1)
    u_sideZ = DirichletBC(DVP.sub(0).sub(2), ((0.0)), boundaries, 4) # no out of plane deformation
    d_t = PrescribedDisp(solid_vel,degree=d_deg)
    u_sideY = DirichletBC(DVP.sub(0), d_t, boundaries, 3) # keep wall rigid
    u_rwall = DirichletBC(DVP.sub(0), d_t, boundaries, 2) # keep right wall rigid

    bcs = [u_lwall, u_rwall,u_sideY,u_sideZ]

    return dict(bcs=bcs,d_t=d_t)

def pre_solve(t, d_t, **namespace):
    """Update boundary conditions"""
    d_t.update(t)


def post_solve(t, dvp_, verbose,counter,save_step, visualization_folder,material_parameters, mesh,dx_s,  **namespace):

    if counter % save_step == 0:

        return_dict=StrStr.calculate_stress_strain(t, dvp_, verbose, visualization_folder,material_parameters, mesh,dx_s, **namespace)

        return return_dict