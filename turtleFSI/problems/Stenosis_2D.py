from dolfin import *
from turtleFSI.problems import *
import numpy as np

# The "ghost_mode" has to do with the assembly of form containing the facet normals n('+') within interior boundaries (dS). For 3D mesh the value should be "shared_vertex", for 2D mesh "shared_facet", the default value is "none".
parameters["ghost_mode"] = "shared_facet" #2D mesh case
_compiler_parameters = dict(parameters["form_compiler"])

"""
2D Stenosis case intended to test RobinBC
Mesh can be found in the follwoing link:
https://drive.google.com/drive/folders/1roV_iE_16Q847AQ_0tEsznIT-6EICX4o?usp=sharing
"""

def set_problem_parameters(default_variables, **namespace):
    # set problem parameters values
    E_s_val = 100E6                            # Young modulus (elasticity) [Pa] Increased a lot for the 2D case
    nu_s_val = 0.45                            # Poisson ratio (compressibility)
    mu_s_val = E_s_val / (2 * (1 + nu_s_val))  # Shear modulus
    lambda_s_val = nu_s_val * 2. * mu_s_val / (1. - 2. * nu_s_val)

    # define and set problem variables values
    default_variables.update(dict(
        T=0.5,                               # Simulation end time
        dt=0.001,                            # Time step size
        theta=0.501,                         # Theta scheme (implicit/explicit time stepping): 0.5 + dt
        atol=1e-7,                           # Absolute tolerance in the Newton solver
        rtol=1e-7,                           # Relative tolerance in the Newton solver
        mesh_file="stenosis",                # Mesh file name
        robin_bc=True,                       # Robin boundary condition
        inlet_id=1,                          # inlet id 
        outlet_id1=2,                        # outlet id
        inlet_s_id=3,                        # inlet solid id
        outlet_s_id=4,                       # outlet solid id
        fsi_id=5,                            # fsi Interface 
        outer_wall_id=6,                     # outer surface / external id 
        dx_f_id=7,                           # ID of marker in the fluid domain
        dx_s_id=8,                           # ID of marker in the solid domain
        ds_s_id=[4,6],                       # IDs of solid external boundaries for Robin BC (external wall + solid outlet)
        rho_f=1.025E3,                       # Fluid density [kg/m3]
        mu_f=3.5E-3,                         # Fluid dynamic viscosity [Pa.s]
        rho_s=1.0E3,                         # Solid density [kg/m3]
        mu_s=mu_s_val,                       # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,                       # Solid Poisson ratio [-]
        lambda_s=lambda_s_val,               # Solid 1rst Lamé coef. [Pa]
        k_s = 1.0E8,                         # elastic response necesary for RobinBC
        c_s = 1.0E2,                         # viscoelastic response necesary for RobinBC
        u_max= 0.75,                         # max inlet flow velocity value [m/s]
        p_val= 5000,                         # inner pressure for initialisation [Pa]
        vel_t_ramp= 0.2,                     # time for velocity ramp 
        p_t_ramp_start = 0.2,                # pressure ramp start time
        p_t_ramp_end = 0.4,                  # pressure ramp end time
        extrapolation="laplace",             # laplace, elastic, biharmonic, no-extrapolation
        extrapolation_sub_type="constant",   # constant, small_constant, volume, volume_change, bc1, bc2
        recompute=30,                        # Number of iterations before recompute Jacobian. 
        recompute_tstep=10,                  # Number of time steps before recompute Jacobian. 
        save_step=1,                         # Save frequency of files for visualisation
        folder="stenosis_2d",              # Folder where the results will be stored
        checkpoint_step=50,                  # checkpoint frequency
        kill_time=100000,                    # in seconds, after this time start dumping checkpoints every timestep
        save_deg=1                           # Default could be 1. 1 saves the nodal values only while 2 takes full advantage of the mide side nodes available in the P2 solution. P2 for nice visualisations
        # probe point(s)
    ))

    return default_variables


def get_mesh_domain_and_boundaries(folder, **namespace):
    # Import mesh file
    mesh = Mesh()
    with XDMFFile("mesh/Stenosis_2D/mesh_correct_id.xdmf") as infile:
        infile.read(mesh)
    # Rescale the mesh coordinated from [mm] to [m]
    x = mesh.coordinates()
    scaling_factor = 0.001  # from mm to m
    x[:, :] *= scaling_factor
    mesh.bounding_box_tree().build(mesh)

    # Import mesh boundaries
    boundaries = MeshValueCollection("size_t", mesh, 1) 
    with XDMFFile("mesh/Stenosis_2D/facet_mesh_correct_id.xdmf") as infile:
        infile.read(boundaries, "name_to_read")

    boundaries = cpp.mesh.MeshFunctionSizet(mesh, boundaries)
    
    # Define mesh domains
    domains = MeshValueCollection("size_t", mesh, 2) 
    with XDMFFile("mesh/Stenosis_2D/mesh_correct_id.xdmf") as infile:
        infile.read(domains, "name_to_read")

    domains = cpp.mesh.MeshFunctionSizet(mesh, domains)

    info_blue("Obtained mesh, domains and boundaries.")

    # Print pvd files for domains and boundaries
    # ff = File("boundaries.pvd")
    # ff << boundaries 
    # ff = File("domains.pvd")
    # ff << domains

    return mesh, domains, boundaries


# Define velocity inlet parabolic profile
class VelInPara(UserExpression):
    def __init__(self, t, vel_t_ramp, u_max, n, dsi, mesh, **kwargs):
        self.t = t
        self.t_ramp = vel_t_ramp
        self.u_max = u_max
        self.n = n # normal direction
        self.dsi = dsi # surface integral element
        self.d = mesh.geometry().dim()
        self.x = SpatialCoordinate(mesh)
        # Compute area of boundary tessellation by integrating 1.0 over all facets
        self.H = assemble(Constant(1.0, name="one")*self.dsi)
        # Compute barycentre by integrating x components over all facets
        self.c = [assemble(self.x[i]*self.dsi) / self.H for i in range(self.d)]
        # Compute radius by taking max radius of boundary points
        self.r = self.H / 2
        super().__init__(**kwargs)

    def update(self, t):
        self.t = t
    
    def eval(self, value, x):
        #Define the velocity ramp
        if self.t < self.t_ramp:
            interp_PA = self.u_max*(-0.5*np.cos((pi/(self.t_ramp))*(self.t)) + 0.5)   # Velocity initialisation with sigmoid
        else:
            interp_PA = self.u_max

        # Define the parabola
        r2 = (x[0]-self.c[0])**2 + (x[1]-self.c[1])**2  # radius**2
        fact_r = 1 - (r2/self.r**2)

        value[0] = -self.n[0] * (interp_PA) *fact_r  # *self.t # x-values
        value[1] = -self.n[1] * (interp_PA) *fact_r  # *self.t # y-values

    def value_shape(self):
        return (2,)


# Define the pressure ramp
class InnerP(UserExpression):
    def __init__(self, t, p_val, p_t_ramp_start, p_t_ramp_end, **kwargs):
        self.t = t 
        self.p_val = p_val
        self.p_t_ramp_start = p_t_ramp_start
        self.p_t_ramp_end = p_t_ramp_end
        super().__init__(**kwargs)

    def eval(self, value, x):
        if self.t < self.p_t_ramp_start:
            value[0] = 0.0
        elif self.t < self.p_t_ramp_end:
            value[0] = self.p_val*(-0.5*np.cos((pi/(self.p_t_ramp_end - self.p_t_ramp_start))*(self.t - self.p_t_ramp_start)) + 0.5) # Pressure initialisation with sigmoid
        else:
            value[0] = self.p_val

    def value_shape(self):
        return ()


# Create boundary conditions
def create_bcs(dvp_, DVP, mesh, boundaries, domains, mu_f, fsi_id, outlet_id1, inlet_id, inlet_s_id, outlet_s_id, psi, F_solid_linear, u_max, vel_t_ramp, p_t_ramp_start, p_t_ramp_end, p_val, p_deg,  v_deg, **namespace):
    
    info_red("Create bcs")

    # Assign InnerP on the reference domain (FSI interface)
    p_out_bc_val = InnerP(t=0.0, p_val=p_val, p_t_ramp_start=p_t_ramp_start, p_t_ramp_end=p_t_ramp_end, degree=p_deg)

    dSS = Measure("dS", domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    F_solid_linear += p_out_bc_val * inner(n('+'), psi('+'))*dSS(fsi_id)  

    # Fluid velocity BCs
    dsi = ds(inlet_id, domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    ndim = mesh.geometry().dim()
    ni = np.array([assemble(n[i]*dsi) for i in range(ndim)])
    n_len = np.sqrt(sum([ni[i]**2 for i in range(ndim)]))  
    normal = ni/n_len
    # Parabolic profile
    u_inflow_exp = VelInPara(t=0.0, vel_t_ramp=vel_t_ramp, u_max=u_max, n=normal, dsi=dsi, mesh=mesh, degree=v_deg)
    u_inlet = DirichletBC(DVP.sub(1), u_inflow_exp, boundaries, inlet_id)              # Impose the parabolic inlet velocity at the inlet
    u_inlet_s = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, inlet_s_id)          # velocity = 0 at inlet solid id
    
    # Solid Displacement BCs
    d_inlet = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, inlet_id)              # displacement = 0 at inlet id
    d_inlet_s = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, inlet_s_id)          # displacement = 0 at inlet solid id
    
    # Assemble boundary conditions
    bcs = [u_inlet, d_inlet, u_inlet_s, d_inlet_s] 

    return dict(bcs=bcs, u_inflow_exp=u_inflow_exp, p_out_bc_val=p_out_bc_val, F_solid_linear=F_solid_linear)


def pre_solve(t, u_inflow_exp, p_out_bc_val, **namespace):
    # Update the time variable used for the inlet boundary condition
    u_inflow_exp.update(t)
    p_out_bc_val.t = t
    return dict(u_inflow_exp=u_inflow_exp, p_out_bc_val=p_out_bc_val)