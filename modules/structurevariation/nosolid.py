from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI

def structure_setup(d_, v_, p_, phi, psi, gamma, dS, mu_f, n,\
            dx_s, dx_f, mu_s, rho_s, lamda_s, k, mesh_file, **semimp_namespace):

	F_solid_linear = inner(Constant((0,0)), psi)*dx_f
	F_solid_nonlinear = inner(Constant((0,0)), phi)*dx_f
	return dict(F_solid_linear = F_solid_linear, F_solid_nonlinear = F_solid_nonlinear)