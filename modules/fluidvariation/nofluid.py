from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, nabla_grad


def fluid_setup(v_, p_, d_, n, phi, psi, gamma, dx_f, ds, dS, mu_f, rho_f, k, v_deg, **semimp_namespace):

	F_fluid_linear = inner(Constant((0,0)), psi)*dx_f

	F_fluid_nonlinear = inner(Constant((0,0)), phi)*dx_f

	return dict(F_fluid_linear = F_fluid_linear, F_fluid_nonlinear = F_fluid_nonlinear)
