# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
This module implements the monolithic Fluid-Structure Interaction (FSI) solver
used in the turtleFSI package.
"""

from dolfin import *
from turtleFSI.utils import *
import os
import sys

# Get user input
args = parse()

# Import the problem
if os.path.isfile(os.path.abspath(args.problem+'.py')):
    exec("from {} import *".format(args.problem))
else:
    try:
        exec("from turtleFSI.problems.{} import *".format(args.problem))
    except:
        raise ImportError("""Can not find the problem file. Make sure that the
        problem file is specified in the current directory or in the solver
        turtleFSI/problems/... directory.""")

# Get problem specific parameters
vars().update(set_problem_parameters(**vars()))

# Update variables from commandline
for key, value in list(args.__dict__.items()):
    if value is None:
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
if extrapolation == "biharmonic":
    Elem = MixedElement([de, ve, pe, de])
else:
    Elem = MixedElement([de, ve, pe])

DVP = FunctionSpace(mesh, Elem)

# Create functions
dvp_ = {}
d_ = {}
v_ = {}
p_ = {}
w_ = {}

for time in ["n", "n-1", "n-2"]:
    dvp = Function(DVP)
    dvp_[time] = dvp
    dvp_list = split(dvp)

    d_[time] = dvp_list[0]
    v_[time] = dvp_list[1]
    p_[time] = dvp_list[2]
    if extrapolation == "biharmonic":
        w_[time] = dvp_list[3]

if extrapolation == "biharmonic":
    phi, psi, gamma, beta = TestFunctions(DVP)
else:
    phi, psi, gamma = TestFunctions(DVP)

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

# Any pre-processing before the simulation
vars().update(initiate(**vars()))

# Create boundary conditions
vars().update(create_bcs(**vars()))

# Set up Newton solver
exec("from turtleFSI.modules.{} import solver_setup, newtonsolver".format(solver))
vars().update(solver_setup(**vars()))


# Functions for residuals
dvp_res = Function(DVP)
chi = TrialFunction(DVP)

t = 0
counter = 0
timer = Timer("Total simulation time")
timer.start()
last_t = 0.0
while t <= T + dt / 10:
    counter += 1
    t += dt

    if MPI.rank(MPI.comm_world) == 0:
        txt = "Solving for timestep {:d}, time {:2.04f}".format(counter, t)
        if verbose:
            print(txt)
        else:
            print(txt, end="\r")

    # Pre solve hook
    vars().update({} if pre_solve(**vars()) is None else pre_solve(**vars()))

    # Solve
    vars().update(newtonsolver(**vars()))

    # Update vectors
    times = ["n-2", "n-1", "n"]
    for i, t_tmp in enumerate(times[:-1]):
        dvp_[t_tmp].vector().zero()
        dvp_[t_tmp].vector().axpy(1, dvp_[times[i+1]].vector())

    # After solve hook
    vars().update({} if post_solve(**vars()) is None else post_solve(**vars()))

    if MPI.rank(MPI.comm_world) == 0:
        last_n = timer.elapsed()[0]
        txt = "Elapsed time: {0:f}".format(last_n-last_t)
        last_t = last_n
        if verbose:
            print(txt)
        else:
            print(txt, end="\r")

timer.stop()
if MPI.rank(MPI.comm_world) == 0:
    print("Total simulation time {0:f}".format(timer.elapsed()[0]))

# Post-processing of simulation
finished(**vars())
