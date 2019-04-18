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

from dolfin import *
from turtleFSI.utils import *

# Get user input
args = parse()

# Import the problem
# TODO: Look for problem file locally as well
exec("from turtleFSI.problems.{} import *".format(args.problem))

# Get problem specific parameters
vars().update(set_problem_parameters(**vars()))

# Update variables from commandline
for key, item in list(args.__dict__.items()):
    if item is None:
        args.__dict__.pop(key)
vars().update(args.__dict__)

# Get mesh information
mesh, domains, boundaries = get_mesh_domain_and_boundaries(**vars())

# Control FEniCS output
set_log_level(loglevel)

# Finite Elements
de = VectorElement('CG', mesh.ufl_cell(), d_deg)
ve = VectorElement('CG', mesh.ufl_cell(), v_deg)
pe = FiniteElement('CG', mesh.ufl_cell(), p_deg)

# Define coefficients
k = Constant(dt)
n = FacetNormal(mesh)

# Define function space
if "biharmonic" in extrapolation:
    Elem = MixedElement([de, ve, pe])
else:
    Elem = MixedElement([de, ve, pe, de])

DVP = FunctionSpace(mesh, Elem)

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
    if "biharmonic" in extrapolation:
        w_[time] = w

if "biharmonic" in extrapolation:
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

# Define solver
# Adding the Matrix() argument is a FEniCS 2018.1.0 hack
up_sol = LUSolver(Matrix(), linear_solver)

# Get variation formulations
exec("from turtleFSI.modules.{} import fluid_setup".format(fluid))
vars().update(fluid_setup(**vars()))
exec("from turtleFSI.modules.{} import solid_setup".format(solid))
vars().update(solid_setup(**vars()))
exec("from turtleFSI.modules.{} import extrapolate_setup".format(extrapolation))
vars().update(extrapolate_setup(**vars()))


# Set up Newton solver
exec("from turtleFSI.modules.{} import solver_setup, newtonsolver".format(solver))
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
timer = Timer("Total simulation time")
timer.start()
while t <= T + dt / 10:
    counter += 1
    t += dt

    if MPI.rank(MPI.comm_world) == 0:
        txt = "Solving for timestep {:6d}, time {:2.04f}".format(counter, t)
        if verbose:
            print(txt)
        else:
            print(txt, end="\r")

    # Pre solve hook
    pre_solve(**vars())

    # Solve
    newtonsolver(**vars())

    # Update vectors
    times = ["n-2", "n-1", "n"]
    for i, t_tmp in enumerate(times[:-1]):
        dvp_[t_tmp].vector().zero()
        dvp_[t_tmp].vector().axpy(1, dvp_[times[i+1]].vector())

    # After solve hook
    after_solve(**vars())

timer.stop()
if MPI.rank(MPI.comm_world) == 0:
    print("Total simulation time {0:f}".format(total_timer.elapsed()[0]))

# Post-processing of simulation
post_process(**vars())
