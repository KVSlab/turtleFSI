# Copyright (c) XXXX, XXXX.
# See LICENSE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import *
from time import time as epoch_time

# Get user input
from modules.utils.argpar import *
args = parse()

# Mesh refiner
exec("from problems.%s import *" % args.problem)

if args.refiner != None:
    for i in range(args.refiner):
        mesh = refine(mesh)

update_variables = {}

# Update argparser input, due to how files are made in problemfiles
for key in args.__dict__:
    if args.__dict__[key] != None:
        update_variables[key] = args.__dict__[key]

vars().update(update_variables)


# Import variationalform and solver
print(args.solver)
exec("from modules.fluidvariation.%s import *" % args.fluidvar)
exec("from modules.structurevariation.%s import *" % args.solidvar)
exec("from modules.extrapolation.%s import *" % args.extravar)
exec("from modules.newtonsolver.%s import *" % args.solver)
# Silence FEniCS output
# set_log_active(False)

# Domains
D = VectorFunctionSpace(mesh_file, "CG", d_deg)
V = VectorFunctionSpace(mesh_file, "CG", v_deg)
P = FunctionSpace(mesh_file, "CG", p_deg)

de = VectorElement('CG', mesh_file.ufl_cell(), d_deg)
ve = VectorElement('CG', mesh_file.ufl_cell(), v_deg)
pe = FiniteElement('CG', mesh_file.ufl_cell(), p_deg)

# Define coefficients
k = Constant(dt)
n = FacetNormal(mesh_file)
#nu = Constant(mu_f/rho_f)


if args.extravar == "biharmonic" or args.extravar == "biharmonic2":
    print("Biharmonic")
    Elem = MixedElement([de, ve, pe, de])
    DVP = FunctionSpace(mesh_file, Elem)

    dvp_ = {}
    d_ = {}
    v_ = {}
    p_ = {}
    w_ = {}

    for time in ["n", "n-1", "n-2"]:
        dvp = Function(DVP)
        dvp_[time] = dvp
        d, v, p, w = split(dvp)

        d_[time] = d
        v_[time] = v
        p_[time] = p
        w_[time] = w

    phi, psi, gamma, beta = TestFunctions(DVP)

else:
    Elem = MixedElement([de, ve, pe])
    DVP = FunctionSpace(mesh_file, Elem)
    # Create functions

    dvp_ = {}
    d_ = {}
    v_ = {}
    p_ = {}

    for time in ["n", "n-1", "n-2", "n-3"]:
        dvp = Function(DVP)
        dvp_[time] = dvp
        d, v, p = split(dvp)

        d_[time] = d
        v_[time] = v
        p_[time] = p

    phi, psi, gamma = TestFunctions(DVP)


# Solvers
#lu_solver = LUSolver()
up_sol = LUSolver('mumps')

vars().update(fluid_setup(**vars()))
vars().update(structure_setup(**vars()))
vars().update(extrapolate_setup(**vars()))
vars().update(solver_setup(**vars()))
vars().update(initiate(**vars()))
vars().update(create_bcs(**vars()))

atol = 1e-7
rtol = 1e-7
max_it = 50
lmbda = 1.0

dvp_res = Function(DVP)
chi = TrialFunction(DVP)

counter = 0
t = 0
t_start = epoch_time()
while t <= T:
    t += dt

    if MPI.rank(mpi_comm_world()) == 0:
        print("Solving for timestep %g" % t)

    pre_solve(**vars())
    vars().update(newtonsolver(**vars()))

    times = ["n-2", "n-1", "n"]
    for i, t_tmp in enumerate(times[:-1]):
        dvp_[t_tmp].vector().zero()
        dvp_[t_tmp].vector().axpy(1, dvp_[times[i+1]].vector())
    vars().update(after_solve(**vars()))
    counter += 1
simtime = epoch_time() - t_start
post_process(**vars())
