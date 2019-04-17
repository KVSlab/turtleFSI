# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
TODO
This module implements a generic form of the fractional step method for
solving the incompressible Navier-Stokes equations. There are several
possible implementations of the pressure correction and the more low-level
details are chosen at run-time and imported from any one of:

  solvers/NSfracStep/IPCS_ABCN.py    # Implicit convection
  solvers/NSfracStep/IPCS_ABE.py     # Explicit convection
  solvers/NSfracStep/IPCS.py         # Naive implict convection
  solvers/NSfracStep/BDFPC.py        # Naive Backwards Differencing IPCS in rotational form
  solvers/NSfracStep/BDFPC_Fast.py   # Fast Backwards Differencing IPCS in rotational form
  solvers/NSfracStep/Chorin.py       # Naive

The naive solvers are very simple and not optimized. They are intended
for validation of the other optimized versions. The fractional step method
can be used both non-iteratively or with iterations over the pressure-
velocity system.

The velocity vector is segregated, and we use three (in 3D) scalar
velocity components.

Each new problem needs to implement a new problem module to be placed in
the problems/NSfracStep folder. From the problems module one needs to import
a mesh and a control dictionary called NS_parameters. See
problems/NSfracStep/__init__.py for all possible parameters.
"""

import importlib
from dolfin import *

from turtleFSI.utils import *

# Get user input
args = parse()

# Import the problem
# TODO: Look for problem file locally as well
importlib.import_module('problems.{}'.format(args.problems))

# Get problem specific parameters and mesh
vars().update(set_problem_parameters(args))
mesh, domains, boundaries = get_mesh_domain_and_boundaries(**vars())

# Refine mesh
if args.refiner != None:
    for i in range(args.refiner):
        mesh = refine(mesh)

# Import variationalform and solver
importlib.import_module('turtleFSI.modules.{}'.format(args.fluidvar))
importlib.import_module('turtleFSI.modules.{}'.format(args.solidvar))
importlib.import_module('turtleFSI.modules.{}'.format(args.extravar))
importlib.import_module('turtleFSI.modules.{}'.format(args.solver))

# Control FEniCS output
set_log_level(args.loglevel)

# Finite Elements
de = VectorElement('CG', mesh_file.ufl_cell(), d_deg)
ve = VectorElement('CG', mesh_file.ufl_cell(), v_deg)
pe = FiniteElement('CG', mesh_file.ufl_cell(), p_deg)

# Define coefficients
k = Constant(dt)
n = FacetNormal(mesh_file)

# Define function space
if "biharmonic" in args.extravar:
    Elem = MixedElement([de, ve, pe])
else:
    Elem = MixedElement([de, ve, pe, de])

DVP = FunctionSpace(mesh_file, Elem)

# Create functions
dvp_ = {}
d_ = {}
v_ = {}
p_ = {}
w_ = {}

for time in ["n", "n-1", "n-2", "n-3"]:
    dvp = Function(DVP)
    dvp_[time] = dvp
    dvp_list = split(dvp)

    d_[time] = dvp_list[0]
    v_[time] = dvp_list[1]
    p_[time] = dvp_list[2]
    if "biharmonic" in args.extravar:
        w_[time] = w

if "biharmonic" in args.extravar:
    phi, psi, gamma = TestFunctions(DVP)
else:
    phi, psi, gamma, beta = TestFunctions(DVP)

# Differentials
ds = Measure("ds", subdomain_data=boundaries)
dS = Measure("dS", subdomain_data=boundaries)
dx = Measure("dx", subdomain_data=domains)

# Domains
dx_f = dx(dx_f_id, subdomain_data=domains)
dx_s = dx(dx_s_id, subdomain_data=domains)

# Solver
# TODO: Add solver type in args
up_sol = LUSolver('mumps')

# Get variation formulations
vars().update(fluid_setup(**vars()))
vars().update(structure_setup(**vars()))
vars().update(extrapolate_setup(**vars()))

# Set up Newton solver
vars().update(solver_setup(**vars()))

# Any pre-processing before the simulation
vars().update(initiate(**vars()))

# Create boundary conditions
vars().update(create_bcs(**vars()))

# Functions for residuals
dvp_res = Function(DVP)
chi = TrialFunction(DVP)

t = 0
counter = 0
tic()
while t <= T + dt / 10:
    t += dt

    if MPI.rank(mpi_comm_world()) == 0:
        print("Solving for timestep {:d}".format(t // dt))

    # Pre solve hook
    pre_solve(**vars())

    # Solve
    vars().update(newtonsolver(**vars()))

    # Update vectors
    times = ["n-2", "n-1", "n"]
    for i, t_tmp in enumerate(times[:-1]):
        dvp_[t_tmp].vector().zero()
        dvp_[t_tmp].vector().axpy(1, dvp_[times[i+1]].vector())

    # After solve hook
    vars().update(after_solve(**vars()))
    counter += 1

simtime = toc()
print("Total Simulation time %g" % simtime)

# Post-processing of simulation
post_process(**vars())
