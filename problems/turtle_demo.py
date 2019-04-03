# TurtleFSI is build on the FEniCS Finite Elements Library project and needs to
# be executer in an environment where FEniCS is installed.


# IMPORT:
# We first need to import the finite element library from FEniCS by importing
# the module "dolfin".
from dolfin import *
import numpy as np

# READ THE MESH

# In this example, the mesh and markers are stored in the 3 following files
mesh_file_loc = "problems/mesh/turtle_demo/turtle_mesh.xdmf"  # mesh geometry
domains_marker_loc = "problems/mesh/turtle_demo/mc.xdmf"  # marker over the elements (domains)
boundaries_marker_loc = "problems/mesh/turtle_demo/mf.xdmf"  # markers of the segments (boundaries)

# "mesh_file" collects the mesh geometry of the entire domain (fluid + solid). In this example, we import a mesh
# stored in a .xdmf file, but other mesh file are supported such as .xml files for instance (see commented lines).
mesh_file = Mesh()
xdmf = XDMFFile(mesh_file_loc)
xdmf.read(mesh_file)
# mesh_file = Mesh("YOUR_MESH.xml") # alternative mesh file format (see FEniCS documentation for more details)

# "domains" collects the element markers of the fluid domain (marked as 1) and the solid domain (marked as 2).
domains = MeshFunction("size_t", mesh_file, mesh_file.geometry().dim())
xdmf = XDMFFile(domains_marker_loc)
xdmf.read(domains)
# domains = MeshFunction("size_t", mesh_file, mesh_file.dim(), "YOUR_DOM_MARKERS.xml") # alternative

# "boundaries" collects the boundary markers that are used to apply the Dirichlet boundary conditions on both the
# fluid and solid domains.
boundaries = MeshFunction("size_t", mesh_file, mesh_file.geometry().dim()-1)
xdmf = XDMFFile(boundaries_marker_loc)
xdmf.read(boundaries)
# boundaries = MeshFunction("size_t", mesh_file, mesh_file.dim()-1, "YOUR_BOUND_MARKERS.xml") # alternative


# PARAMETERS
common = {"mesh": mesh_file,
          "v_deg": 2,  # Velocity degree
          "p_deg": 1,  # Pressure degree
          "d_deg": 2,  # Deformation degree
          "T": 100,  # End time [s]
          "dt": 0.001,  # Time step [s]
          "rho_f": 1.0E3,  # Fluid density [kg/m3]
          "mu_f": 1.0,  # Fluid dynamic viscosity [Pa.s]
          "rho_s": 1.0E3,  # Solid density [kg/m3]
          "mu_s": 5.0E4,  # Solid shear modulus or 2nd Lame Coef. [Pa]
          "lamda_s": 4.5E5,  # Solid Young's modulus [Pa]
          "nu_s": 0.45,  # Solid Poisson ratio [-]
          "step": 1,  # save every step
          "checkpoint": 0}  # checkpoint every step
vars().update(common)
lamda_s = nu_s*2*mu_s/(1 - 2.*nu_s)  # Solid Young's modulus [Pa]


# BOUNDARIES

ds = Measure("ds", subdomain_data=boundaries)
dS = Measure("dS", subdomain_data=boundaries)
n = FacetNormal(mesh_file)

dx = Measure("dx", subdomain_data=domains)
dx_f = dx(1, subdomain_data=domains)
dx_s = dx(2, subdomain_data=domains)


class Inlet(Expression):
    def __init__(self, **kwargs):
        self.t = 0.0
        self.t_ramp = 1.0  # time to linearly ramp-up the inlet velocity
        self.Um = 0.8  # Max. velocity inlet [m/s]

    def eval(self, value, x):
        if self.t < self.t_ramp:
            value[0] = self.Um * np.abs(np.cos(self.t*np.pi)-1)
            value[1] = 0
        else:
            value[0] = np.max([self.Um/5, self.Um * np.abs(np.cos(self.t*np.pi)-1)])
            value[1] = 0

    def value_shape(self):
        return (2,)


inlet = Inlet(degree=v_deg)


def create_bcs(DVP, inlet, boundaries, args,  **semimp_namespace):
    print("Create bcs")

    noslip = ((0.0, 0.0))

    # Fluid velocity conditions
    u_inlet = DirichletBC(DVP.sub(1), inlet, boundaries, 14)
    u_bot = DirichletBC(DVP.sub(1).sub(1), (0.0), boundaries, 11)
    u_top = DirichletBC(DVP.sub(1).sub(1), (0.0), boundaries, 13)
    u_head_tail = DirichletBC(DVP.sub(1), noslip, boundaries, 15)

    # Pressure Conditions
    p_outlet = DirichletBC(DVP.sub(2), (0.0), boundaries, 12)

    bcs = [u_bot, u_top, u_inlet, p_outlet,  u_head_tail]

    if args.bitype == "bc1":
        d_inlet = DirichletBC(DVP.sub(0), noslip, boundaries, 14)
        d_bot = DirichletBC(DVP.sub(0), noslip, boundaries, 11)
        d_top = DirichletBC(DVP.sub(0), noslip, boundaries, 13)
        d_outlet = DirichletBC(DVP.sub(0), noslip, boundaries, 12)
        d_head_tail = DirichletBC(DVP.sub(0), noslip, boundaries, 15)
        for i in [d_bot, d_top, d_outlet, d_inlet, d_head_tail]:
            bcs.append(i)

    if args.bitype == "bc2":
        w_inlet = DirichletBC(DVP.sub(0), noslip, boundaries, 14)
        w_bot = DirichletBC(DVP.sub(0), noslip, boundaries, 11)
        w_top = DirichletBC(DVP.sub(0), noslip, boundaries, 13)
        w_outlet = DirichletBC(DVP.sub(0), noslip, boundaries, 12)
        w_head_tail = DirichletBC(DVP.sub(0), noslip, boundaries, 15)

        d_inlet = DirichletBC(DVP.sub(0), noslip, boundaries, 14)
        d_bot = DirichletBC(DVP.sub(0), noslip, boundaries, 11)
        d_top = DirichletBC(DVP.sub(0), noslip, boundaries, 13)
        d_outlet = DirichletBC(DVP.sub(0), noslip, boundaries, 12)
        d_head_tail = DirichletBC(DVP.sub(0), noslip, boundaries, 15)

        for i in [w_bot, w_top, w_outlet, w_inlet, w_head_tail,
                  d_bot, d_top, d_outlet, d_inlet, d_head_tail]:
            bcs.append(i)

    return dict(bcs=bcs, inlet=inlet)


def pre_solve(t, inlet, **semimp_namespace):

    inlet.t = t

    return dict(inlet=inlet)


def initiate(dvp_, **semimp_namespace):

    path = "results/turtle_demo/"

    u_file = XDMFFile(mpi_comm_world(), path + "/velocity.xdmf")
    d_file = XDMFFile(mpi_comm_world(), path + "/d.xdmf")
    p_file = XDMFFile(mpi_comm_world(), path + "/pressure.xdmf")
    for tmp_t in [u_file, d_file, p_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["rewrite_function_mesh"] = False
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)
    d_file.write(d, 0.0)
    u_file.write(v, 0.0)
    p_file.write(p, 0.0)

    return dict(u_file=u_file, d_file=d_file, p_file=p_file, path=path)


def after_solve(t, dvp_, counter, u_file, p_file, d_file, **semimp_namespace):

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

    return {}


def post_process(**semimp_namespace):

    return {}
