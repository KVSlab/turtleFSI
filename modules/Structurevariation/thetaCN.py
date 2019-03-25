from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI
#from semi_implicit import *

def F_(U):
	return Identity(len(U)) + grad(U)

def J_(U):
	return det(F_(U))

def E(U):
	return 0.5*(F_(U).T*F_(U) - Identity(len(U)))

def S(U,lamda_s,mu_s):
    I = Identity(len(U))
    return 2*mu_s*E(U) + lamda_s*tr(E(U))*I

def Piola1(U,lamda_s,mu_s):
	return F_(U)*S(U,lamda_s,mu_s)

def A_E(d, v,  lamda_s, mu_s, rho_s, delta, psi, phi, dx_s):
	return inner(Piola1(d, lamda_s, mu_s), grad(psi))*dx_s \
		   - delta*rho_s*inner(v, phi)*dx_s


def structure_setup(d_, v_, p_, phi, psi, gamma, dS, mu_f, n,\
            dx_s, dx_f, mu_s, rho_s, lamda_s, k, mesh_file, theta, **semimp_namespace):

	delta = 1E10
	"""
	A_T =  rho_s/k*inner(v_["n"] - v_["n-1"], psi)*dx_s \
	    + delta*(rho_s/k)*inner(d_["n"] - d_["n-1"], phi)*dx_s

	F_solid_nonlinear = theta*A_E(d_["n"], v_["n"], lamda_s, mu_s, rho_s, delta, psi, phi, dx_s) \
					  +  (1 - theta)*A_E(d_["n-1"], v_["n-1"], lamda_s, mu_s, rho_s, delta, psi, phi, dx_s)

	F_solid_linear = A_T
	"""

	F_solid_linear = rho_s/k*inner(v_["n"] - v_["n-1"], psi)*dx_s \
	               + delta*(1/k)*inner(d_["n"] - d_["n-1"], phi)*dx_s \
   				   - delta*inner(Constant(theta)*v_["n"] + Constant(1 - theta)*v_["n-1"], phi)*dx_s

	F_solid_nonlinear = inner(Piola1(Constant(theta)*d_["n"] + Constant(1 - theta)*d_["n-1"], lamda_s, mu_s), grad(psi))*dx_s

	return dict(F_solid_linear = F_solid_linear, F_solid_nonlinear = F_solid_nonlinear)
