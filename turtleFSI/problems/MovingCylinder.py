import os

from dolfin import *
from turtleFSI.problems import *
from turtleFSI.modules import *
import numpy as np

"""
This problem is used to test the fluid solver with a moving domain.
There is no fluid-solid interaction, but the cylinder is moving in the fluid, hence fluid is affected by the motion of the cylinder.
The file is based on the code developed by Henrik A. Kjeldsberg with the following version:
https://github.com/KVSlab/OasisMove/blob/c19d0982576aa4a472b325ade8032a25bf60629d/src/oasismove/problems/NSfracStep/MovingCylinder.py
Since dt is set to be small, you will most likely need to run the simulation on a cluster to get the results in a meaningful way.
"""


comm = MPI.comm_world


def set_problem_parameters(default_variables, **namespace):
    """
    Problem file for running CFD simulation for the oscillating cylinder in a rectangular 2D domain, as described
    by Blackburn and Henderson [1].The cylinder is prescribed an oscillatory motion and is placed in a free stream,
    with a diameter of D cm. The kinetmatic viscosity is computed from the free stream velocity of 1 m/s for a Reynolds
    number of 500, which is well above the critical value for vortex shedding. Moreover, the oscillation is mainly
    controlled by the amplitude ratio A_ratio, the Strouhal number St, and the frequency ratio F. 

    [1] Blackburn, H. M., & Henderson, R. D. (1999). A study of two-dimensional flow past an oscillating cylinder.
    Journal of Fluid Mechanics, 385, 255-286.
    """
    D=0.1  # Diameter in [m]
    Re=500  # Reynolds number
    u_inf=1.0  # Free-stream flow velocity in [m/s]
    rho_f = 1000
    mu_f = rho_f * u_inf * D / Re
    factor = 1 / 2 * rho_f * u_inf ** 2 * D
    # Default parameters
    default_variables.update(dict(
        # Geometrical parameters
        Re=Re,  # Reynolds number
        D=D,  # Diameter in [m]
        u_inf=u_inf,  # Free-stream flow velocity in [m/s]
        A_ratio=0.25,  # Amplitude ratio
        St=0.2280,  # Strouhal number
        F_r=1.0,  # Frequency ratio
        factor=factor,
        # Simulation parameters
        T=5,  # End time
        dt=0.000125,  # Time step
        
        folder="results_moving_cylinder",
        # fluid parameters
        rho_f=rho_f,
        mu_f=mu_f,

        # solid parameters
        solid="no_solid",

        # Extraplotation parameters
        extrapolation="laplace",
        extrapolation_sub_type="constant",

        # Solver parameters
        theta=0.500125, # shifted Crank-Nicolson, theta = 0.5 + dt
        
        checkpoint_step=500,
        save_step = 100,
        recompute=25,
        recompute_tstep=100,
        d_deg=1,
        v_deg=1,
        p_deg=1))

    return default_variables


def get_mesh_domain_and_boundaries(D, **namespace):
    # Import mesh
    mesh = Mesh()
    with XDMFFile(MPI.comm_world, "mesh/MovingCylinder/mesh4.xdmf") as infile:
        infile.read(mesh)
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(1)
    Allboundaries = DomainBoundary()
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    Allboundaries.mark(boundaries, 0)

    # Reference domain from Blackburn & Henderson [1]
    # Cylinder centered in (0,0)
    H = 30 * D / 2  # Height
    L1 = -10 * D  # Length
    L2 = 52 * D  # Length

    # Mark geometry
    inlet = AutoSubDomain(lambda x, b: b and x[0] <= L1 + DOLFIN_EPS)
    walls = AutoSubDomain(lambda x, b: b and (near(x[1], -H) or near(x[1], H)))
    circle = AutoSubDomain(lambda x, b: b and (-H / 2 <= x[1] <= H / 2) and (L1 / 2 <= x[0] <= L2 / 2))
    outlet = AutoSubDomain(lambda x, b: b and (x[0] > L2 - DOLFIN_EPS * 1000))

    inlet.mark(boundaries, 1)
    walls.mark(boundaries, 2)
    circle.mark(boundaries, 3)
    outlet.mark(boundaries, 4)
    
    return mesh, domains, boundaries

class MovingCylinder(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = 0
        self.y_max = 0
        self.f_o = 0

    def eval(self, value, x):
        value[0] = 0
        value[1] = self.y_max * sin(2 * pi * self.f_o * self.t)

    def value_shape(self):
        return (2,)

def create_bcs(DVP, D, u_inf, St, F_r, A_ratio, boundaries, **namespace):
    info_red("Creating boundary conditions")

    f_v = St * u_inf / D  # Fixed-cylinder vortex shredding frequency
    f_o = F_r * f_v  # Frequency of harmonic oscillation
    y_max = A_ratio * D  # Max displacement (Amplitude)
    if MPI.rank(comm) == 0:
        print("Frequency is %.4f" % f_o)
        print("Amplitude is %.4f " % y_max)

    # cylinder_motion = MovingCylinder(t=0, f_o=f_o, y_max=y_max)
    cylinder_motion = MovingCylinder()
    cylinder_motion.f_o = f_o
    cylinder_motion.y_max = y_max
    # Define boundary conditions for the velocity and the pressure
    bcu_inlet = DirichletBC(DVP.sub(1), Constant((u_inf, 0)), boundaries, 1)
    bcu_wall = DirichletBC(DVP.sub(1), Constant((u_inf, 0)), boundaries, 2)
    bcu_circle = DirichletBC(DVP.sub(1), Constant((0, 0)), boundaries, 3)
    bcp_outlet = DirichletBC(DVP.sub(2), Constant(0), boundaries, 4)

    bcd_inlet = DirichletBC(DVP.sub(0), Constant((0, 0)), boundaries, 1)
    bcd_wall = DirichletBC(DVP.sub(0), Constant((0, 0)), boundaries, 2)
    bcd_circle = DirichletBC(DVP.sub(0), cylinder_motion, boundaries, 3)
    bcd_outlet = DirichletBC(DVP.sub(0), Constant((0, 0)), boundaries, 4)

    bcs = [bcu_inlet, bcu_wall, bcu_circle, bcp_outlet, bcd_inlet, bcd_wall, bcd_circle, bcd_outlet]

    return dict(bcs=bcs, cylinder_motion=cylinder_motion)

def initiate(**namespace):
    # Lists to hold displacement, forces, and time
    drag_list = []
    lift_list = []
    time_list = []

    return dict(drag_list=drag_list, lift_list=lift_list, time_list=time_list)

def pre_solve(cylinder_motion, t,  boundaries, **namespace):
    cylinder_motion.t = t
    ds_circle = Measure("ds", domain=boundaries.mesh(), subdomain_data=boundaries, subdomain_id=3)
    return dict(cylinder_motion=cylinder_motion, ds_circle=ds_circle)

def post_solve(t, n, dvp_, results_folder, drag_list, lift_list, time_list, factor, mu_f, ds_circle, **namespace):
    # Compute drag and lift coefficients
    
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    # Compute forces
    force = dot(sigma(v, p, d, mu_f), n)   
    drag_list.append(-assemble(force[0]*ds_circle))
    lift_list.append(-assemble(force[1]*ds_circle))
    time_list.append(t)

    # Store forces to file
    if MPI.rank(MPI.comm_world) == 0:
        drag_coeff = drag_list[-1]/ factor
        lift_coeff = lift_list[-1] / factor
        data = [t, drag_coeff, lift_coeff]
        
        data_path = os.path.join(results_folder, "forces.txt")

        with open(data_path, "ab") as f:
            np.savetxt(f, data, fmt=" %.16f ", newline=' ')
            f.write(b'\n')