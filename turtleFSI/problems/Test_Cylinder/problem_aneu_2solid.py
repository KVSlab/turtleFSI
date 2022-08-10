from dolfin import *
import os
from turtleFSI.problems import *
import numpy as np
from os import path, makedirs, getcwd

# BM will look at the IDs. Oringinal drawing :) Discuss drawing as well...
# Thangam will clean up terminology, innerP, prestress and that everything works. Make sure that the code prints logically and chronologically. BCs in sep file@
# David B, add stress code and xdmf for viz

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. 
parameters["form_compiler"]["optimize"] = True
# The "ghost_mode" has to do with the assembly of form containing the facet
# normals n('+') within interior boundaries (dS). for 3D mesh the value should
# be "shared_vertex", for 2D mesh "shared_facet", the default value is "none"
parameters["ghost_mode"] = "shared_vertex" 
_compiler_parameters = dict(parameters["form_compiler"])


def set_problem_parameters(default_variables, **namespace):
    # Overwrite default values
    E_s_val = 1E6  # Young modulus (Pa)
    nu_s_val = 0.45
    mu_s_val = E_s_val / (2 * (1 + nu_s_val))  # 0.345E6
    lambda_s_val = nu_s_val * 2. * mu_s_val / (1. - 2. * nu_s_val)

    E_s2_val = 1E7  # Young modulus (Pa)
    nu_s2_val = 0.45
    mu_s2_val = E_s2_val / (2 * (1 + nu_s2_val))  # 0.345E6
    lambda_s2_val = nu_s2_val * 2. * mu_s2_val / (1. - 2. * nu_s2_val)

    default_variables.update(dict(
        T=0.02, # Simulation end time
        dt=0.001, # Timne step size
        theta=0.501, # Theta scheme (implicit/explicit time stepping)
        atol=1e-5, # Absolute tolerance in the Newton solver
        rtol=1e-5,# Relative tolerance in the Newton solver
        mesh_file="artery_coarse_rescaled", # duh
        inlet_id=2,  # inlet id 
        outlet_id1=3,  # outlet nb
        inlet_outlet_s_id=11,  # also the "rigid wall" id for the stucture problem. MB: Clarify :)
        fsi_id=22,  # fsi Interface 
        rigid_id=11,  # "rigid wall" id for the fluid and mesh problem
        outer_wall_id=33,  # outer surface / external id 
        rho_f=1.025E3,    # Fluid density [kg/m3]
        mu_f=3.5E-3,       # Fluid dynamic viscosity [Pa.s]

        rho_s=[1.0E3,1.0E3],    # Solid density [kg/m3]
        mu_s=[mu_s_val,mu_s2_val],     # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=[nu_s_val,nu_s2_val],      # Solid Poisson ratio [-]
        lambda_s=[lambda_s_val,lambda_s2_val],  # Solid 1st Lame Coef. [Pa]

        dx_f_id=1,      # ID of marker in the fluid domain. When reading the mesh, the fuid domain is assigned with a 1. Old crap :)
        dx_s_id=[2,3],      # ID of marker in the solid domain
        extrapolation="laplace",  # laplace, elastic, biharmonic, no-extrapolation
        # ["constant", "small_constant", "volume", "volume_change", "bc1", "bc2"]
        extrapolation_sub_type="constant",
        recompute=3000, # Number of iterations before recompute Jacobian. 
        recompute_tstep=1, # Number of time steps before recompute Jacobian. 
        save_step=1, # Save frequency of files for visualisation
        folder="Stiff_Half",# duh 
        checkpoint_step=50, # duh
        kill_time=100000, # in seconds, after this time start dumping checkpoints every timestep
        save_deg=1 # Default could be 1. 1 saves the nodal values only while 2 takes full advantage of the mide side nodes available in the P2 solution. P2 f0r nice visualisations :)
        # probe point(s)
    ))

    return default_variables


def get_mesh_domain_and_boundaries(mesh_file, fsi_id, rigid_id, outer_wall_id, folder, **namespace):
    print("Obtaining mesh, domains and boundaries...")
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), "mesh/" + mesh_file + ".h5", "r")
    hdf.read(mesh, "/mesh", False)
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains")

    # Only consider FSI in domain within this cylindrical region # Corrected sphere for the rescaled artery!
    center_y = -6.299931555986404e-06
    extent_FSI = 0.01 # This makes the whole cylinder deformable
    min_y=center_y-extent_FSI
    max_y=center_y+extent_FSI
    print("The FSI region extends from y = {} to y = {}".format(min_y, max_y))

    i = 0
    for submesh_facet in facets(mesh):
        idx_facet = boundaries.array()[i]
        if idx_facet == fsi_id or idx_facet == outer_wall_id:
            mid = submesh_facet.midpoint()
            mid_y = mid.y()
            if mid_y < min_y or mid_y > max_y:
                boundaries.array()[i] = rigid_id
        i += 1

    # Make the middle of the cylinder stiffer
    print("The stiffer region extends from y > {}".format(center_y))

    i = 0
    for submesh_cell in cells(mesh):
        idx_cell = domains.array()[i]
        if idx_cell == 2:
            mid = submesh_cell.midpoint()
            mid_y = mid.y()
            if mid_y > center_y:
                domains.array()[i] = 3 # 3 is our stiffer material
        i += 1


    print("Obtained mesh, domains and boundaries.")
    ff = File("domains/domains_solid.pvd")
    ff << domains
    ff = File("domains/boundaries.pvd")
    ff << boundaries
    return mesh, domains, boundaries


class StokesPrb():

    def __init__(self, mesh, domains, boundaries, dsi, inlet_id, outlet_id1, normal, mu_f):

        self.mesh = mesh
        self.ve = VectorElement('CG', self.mesh.ufl_cell(), 2)
        self.pe = FiniteElement('CG', self.mesh.ufl_cell(), 1)
        self.Elem = MixedElement([self.ve, self.pe])
        self.VP = FunctionSpace(self.mesh, self.Elem)

        self.u_inflow_exp = VelInPara(t=0.0, t_ramp=0.0, n=normal, dsi=dsi, mesh=mesh, degree=3)
        self.u_inlet = DirichletBC(self.VP.sub(0), self.u_inflow_exp, boundaries, inlet_id)
        self.u_fsi1 = DirichletBC(self.VP.sub(0), ((0.0, 0.0, 0.0)), boundaries, 11)
        self.u_fsi2 = DirichletBC(self.VP.sub(0), ((0.0, 0.0, 0.0)), boundaries, 22)
        self.u_fsi3 = DirichletBC(self.VP.sub(0), ((0.0, 0.0, 0.0)), boundaries, 33)

        self.Pout_val = Constant(0.0)

        self.bcs = [self.u_inlet, self.u_fsi1, self.u_fsi2, self.u_fsi3]

        self.ds = Measure("ds", domain=self.mesh, subdomain_data=boundaries)
        self.n = FacetNormal(self.mesh)
        # Define variational problem
        (self.u, self.p) = TrialFunctions(self.VP)
        (self.v, self.q) = TestFunctions(self.VP)
        f = Constant((0.0, 0.0, 0.0))
        self.a = (mu_f*inner(grad(self.u), grad(self.v))*dx
                  - self.p*div(self.v)*dx
                  + self.q*div(self.u)*dx)
        self.L = (inner(f, self.v)*dx
                  - self.Pout_val * inner(self.n, self.v)*self.ds(outlet_id1))

    def solve(self):

        # Solution vector
        self.w = Function(self.VP)
        # Assemble system and Solve
        A, B = assemble_system(self.a, self.L, self.bcs)
        #solve(A, self.w.vector(), B, "minres", "amg")
        loc_sol = LUSolver(Matrix(), "mumps")
        loc_sol.solve(A, self.w.vector(), B)
        # Get sub-functions
        self.u, self.p = self.w.split(deepcopy=True)

        return self.u, self.p


class VelInPara(UserExpression):
    def __init__(self, t, t_ramp, n, dsi, mesh, **kwargs):
        self.t = t
        self.t1 = t_ramp
        self.n = n
        self.dsi = dsi
        self.d = mesh.geometry().dim()
        self.x = SpatialCoordinate(mesh)
        # Compute area of boundary tesselation by integrating 1.0 over all facets
        self.A = assemble(Constant(1.0, name="one")*self.dsi)
        # Compute barycenter by integrating x components over all facets
        self.c = [assemble(self.x[i]*self.dsi) / self.A for i in range(self.d)]
        # Compute radius by taking max radius of boundary points
        self.r = np.sqrt(self.A / np.pi)
        super().__init__(**kwargs)

    def update(self, t):
        self.t = t
        
    def eval(self, value, x):

        if self.t < 0.1:
            interp_PA = self.t*(0.74/0.1) # 0.77 m/s mmHg rise in 0.1s
        else:
            interp_PA = self.t*0.9666667 + 0.64333333 # lower slope

        r2 = (x[0]-self.c[0])**2 + (x[1]-self.c[1])**2 + (x[2]-self.c[2])**2  # radius**2
        fact_r = 1 - (r2/self.r**2)

        value[0] = -self.n[0] * (interp_PA) *fact_r  # *self.t
        value[1] = -self.n[1] * (interp_PA) *fact_r  # *self.t
        value[2] = -self.n[2] * (interp_PA) *fact_r  # *self.t

    def value_shape(self):
        return (3,)

class InnerP(UserExpression):
    def __init__(self, t, **kwargs):
        self.t = t     
        super().__init__(**kwargs)

    def eval(self, value, x):
        
        if self.t < 0.1:
            value[0] = self.t*(10932.4/0.1) # 9333 mmHg rise in 0.1s
        else:
            value[0] = self.t*10023.84667 + 9930.01533 # lower slope

    def value_shape(self):
        return ()


def create_bcs(dvp_, DVP, mesh, boundaries, domains, mu_f,
               fsi_id, outlet_id1, inlet_id, inlet_outlet_s_id,
               rigid_id, psi, F_solid_linear, **namespace):

    if MPI.rank(MPI.comm_world) == 0:
        print("Create bcs")

    # Define the pressure condition necessary to create the variational form (Neumann BCs)

    p_out_bc_val = InnerP(t=0.0, degree=2)

    dSS = Measure("dS", domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    F_solid_linear += p_out_bc_val * inner(n('+'), psi('+'))*dSS(fsi_id)  # defined on the reference domain

    # Fluid velocity BCs
    dsi = ds(inlet_id, domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    ndim = mesh.geometry().dim()
    ni = np.array([assemble(n[i]*dsi) for i in range(ndim)])
    n_len = np.sqrt(sum([ni[i]**2 for i in range(ndim)]))  # Should always be 1!?
    normal = ni/n_len

    # new Parabolic profile
    u_inflow_exp = VelInPara(t=0.0, t_ramp=0.0, n=normal, dsi=dsi, mesh=mesh, degree=3)
    u_inlet = DirichletBC(DVP.sub(1), u_inflow_exp, boundaries, inlet_id)
    u_inlet_s = DirichletBC(DVP.sub(1), ((0.0, 0.0, 0.0)), boundaries, inlet_outlet_s_id)

    # Solid Displacement BCs
    d_inlet = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, inlet_id)
    d_inlet_s = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, inlet_outlet_s_id)
    d_rigid = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, rigid_id)

    # Assemble boundary conditions
    bcs = [u_inlet, d_inlet,  u_inlet_s, d_inlet_s, d_rigid]

    # Solve Stokes problem
    Stokes = StokesPrb(mesh, domains, boundaries, dsi, inlet_id, outlet_id1, normal, mu_f)
    sol = Stokes.solve()
    assign(dvp_["n"].sub(1), sol[0])
    assign(dvp_["n"].sub(2), sol[1])
    assign(dvp_["n-1"].sub(1), sol[0])
    assign(dvp_["n-1"].sub(2), sol[1])
    assign(dvp_["n-2"].sub(1), sol[0])
    assign(dvp_["n-2"].sub(2), sol[1])

    return dict(bcs=bcs, u_inflow_exp=u_inflow_exp, p_out_bc_val=p_out_bc_val,
                F_solid_linear=F_solid_linear)


def pre_solve(t, u_inflow_exp, p_out_bc_val, **namespace):
    # Update the time variable used for the inlet boundary condition
    u_inflow_exp.update(t)
    p_out_bc_val.t = t
    return dict(u_inflow_exp=u_inflow_exp, p_out_bc_val=p_out_bc_val)

