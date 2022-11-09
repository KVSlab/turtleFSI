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
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. 

def set_problem_parameters(default_variables, **namespace):
    # Overwrite default values
    E_s_val = 1E6  # Young modulus (Pa)
    nu_s_val = 0.45
    mu_s_val = E_s_val / (2 * (1 + nu_s_val))  # 0.345E6
    lambda_s_val = nu_s_val * 2. * mu_s_val / (1. - 2. * nu_s_val)

    
    default_variables.update(dict(
        # Temporal variables
        T=0.3,          # End time [s]
        dt=0.001,       # Time step [s]
        checkpoint_step=1000, # Checkpoint frequency
        theta=0.5,     # Temporal scheme
        save_step=1,

        # Physical constants
        rho_f=1.0e3,   # Fluid density [kg/m3]
        mu_f=1.0,      # Fluid dynamic viscosity [Pa.s]

        rho_s=1.0E3,    # Solid density [kg/m3]
        mu_s=mu_s_val,     # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,      # Solid Poisson ratio [-]
        lambda_s=lambda_s_val,  # Solid 1st Lame Coef. [Pa]
        gravity=None,   # Gravitational force [m/s**2]
        #solid_properties={"nu_visc_s":1e3},
        #solid_properties={"dx_s_id":1,"material_model":"StVenantKirchoff","rho_s":1.0E3,"mu_s":mu_s_val,"lambda_s":lambda_s_val,"nu_visc_s":1e5, "delta_visc_s":1e6},
        #solid_properties={"dx_s_id":1,"material_model":"StVenantKirchoff","rho_s":1.0E3,"mu_s":mu_s_val,"lambda_s":lambda_s_val,"nu_visc_s":1e5, "delta_visc_s":1e6},
        #solid_properties={"dx_s_id":1,"material_model":"StVenantKirchoff","rho_s":1.0E3,"mu_s":mu_s_val,"lambda_s":lambda_s_val,"viscoelasticity":"None","nu_visc_s":0.01, "delta_visc_s":0.1},
        solid_properties={"dx_s_id":1,"material_model":"StVenantKirchoff","rho_s":1.0E3,"mu_s":mu_s_val,"lambda_s":lambda_s_val,"viscoelasticity":"Form1","nu_visc_s":1e4, "delta_visc_s":1e4},

        # Problem specific
        dx_f_id=0,     # Id of the fluid domain
        dx_s_id=1,     # Id of the solid domain
        folder="Uniaxial_Tension_StVenantKirchoff_Relax_Visc",          # Folder to store the results
        fluid="no_fluid",                 # Do not solve for the fluid
        extrapolation="no_extrapolation",  # No displacement to extrapolate
        solid_vel=0.5, # this is the velocity of the wall with prescribed displacement

        # Geometric variables
        leftEnd=0.001,
        rightEnd=0.099))   

    return default_variables


def get_mesh_domain_and_boundaries(leftEnd, rightEnd, **namespace):
    # Read mesh

    a=Point(0.0, 0.0, 0.0)
    b=Point(0.1, 0.1, 0.1)
    mesh = BoxMesh(a, b, 3, 3, 3)

    #print(dir(mesh))
    # Mark boundaries
    Lwall = AutoSubDomain(lambda x: (x[0]< leftEnd))
    Rwall = AutoSubDomain(lambda x: (x[0]> rightEnd))
    u_sideY = AutoSubDomain(lambda x: (x[1] < 0.001))
    u_sideZ = AutoSubDomain(lambda x: (x[2] < 0.001))

    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)

    boundaries.set_all(0)

    Lwall.mark(boundaries, 1)
    Rwall.mark(boundaries, 2)
    u_sideY.mark(boundaries, 3)
    u_sideZ.mark(boundaries, 4)

    # Mark domain
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(1)

    return mesh, domains, boundaries,

class PrescribedDisp(UserExpression):
    def __init__(self, solid_vel, **kwargs):
        self.solid_vel = solid_vel
        self.factor = 0

        super().__init__(**kwargs)

    def update(self, t):
        if t<0.05:
            self.factor = t * self.solid_vel
        else:
            self.factor = 0.05 * self.solid_vel
        print('displacement = ', self.factor)

    def eval(self, value,x):
        value[0] = self.factor



def create_bcs(DVP,d_deg,solid_vel, boundaries, **namespace):
    # Sliding contact on 3 sides
    u_lwallX = DirichletBC(DVP.sub(0).sub(0), ((0.0)), boundaries, 1)
    u_CornerY = DirichletBC(DVP.sub(0).sub(1), ((0.0)), boundaries, 3)
    u_CornerZ = DirichletBC(DVP.sub(0).sub(2), ((0.0)), boundaries, 4)

    # Displacement on the right hand side (unconstrained in Y and Z)
    d_t = PrescribedDisp(solid_vel,degree=d_deg)
    u_rwall = DirichletBC(DVP.sub(0).sub(0), d_t, boundaries, 2)


    bcs = [u_lwallX, u_rwall,u_CornerY,u_CornerZ]

    return dict(bcs=bcs,d_t=d_t)

def pre_solve(t, d_t, **namespace):
    """Update boundary conditions"""
    d_t.update(t)

def post_solve(t, dvp_, verbose,counter,save_step, visualization_folder,solid_properties, mesh,dx_s,dt,  **namespace):

    if counter % save_step == 0:

        return_dict=StrStr.calculate_stress_strain(t, dvp_, verbose, visualization_folder,solid_properties[0], mesh, dx_s[0],dt, **namespace)

        return return_dict