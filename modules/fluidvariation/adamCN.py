from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, nabla_grad


def F_(U):
	return Identity(len(U)) + grad(U)

def J_(U):
	return det(F_(U))

def sigma_f_u(u,d,mu_f):
    return  mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)

def sigma_f_p(p, u):
    return -p*Identity(len(u))

def A_E(J, v, d, rho_f, mu_f, psi, dx_f):
    return rho_f*inner(J*grad(v)*inv(F_(d))*v, psi)*dx_f \
        + inner(J*sigma_f_u(v, d, mu_f)*inv(F_(d)).T, grad(psi))*dx_f


def fluid_setup(v_, p_, d_, n, psi, gamma, dx_f, ds, mu_f, rho_f, k, dt, v_deg, theta, **semimp_namespace):

    J_theta = theta*J_(d_["n"]) + (1 - theta)*J_(d_["n-1"])
    F_fluid_linear = rho_f/k*inner(J_theta*(v_["n"] - v_["n-1"]), psi)*dx_f

    F_fluid_nonlinear =  rho_f*inner((Constant(theta)*J_(d_["n"])*grad(v_["n"])*inv(F_(d_["n"])) + \
                         Constant(1 - theta)*J_(d_["n-1"])*grad(v_["n-1"])*inv(F_(d_["n-1"]))) * \
                         (0.5*(3*v_["n-1"] - v_["n-2"]) - \
                         0.5/k*((3*d_["n-1"] - d_["n-2"]) - (3*d_["n-2"] - d_["n-3"]))), psi)*dx_f

   # F_fluid_nonlinear =  rho_f*inner((Constant(theta)*J_(d_["n"])*grad(v_["n"])*inv(F_(d_["n"])) + \
    #                     Constant(1 - theta)*J_(d_["n-1"])*grad(v_["n-1"])*inv(F_(d_["n-1"]))) * \
     #                    (0.5*(3*v_["n-1"] - v_["n-2"]) -\
     #                    (d_["n"]-d_["n-1"])/k), psi)*dx_f


    F_fluid_nonlinear += inner(J_(d_["n"])*sigma_f_p(p_["n"], d_["n"])*inv(F_(d_["n"])).T, grad(psi))*dx_f
    F_fluid_nonlinear += Constant(theta)*inner(J_(d_["n"])*sigma_f_u(v_["n"], d_["n"], mu_f)*inv(F_(d_["n"])).T, grad(psi))*dx_f
    F_fluid_nonlinear += Constant(1 - theta)*inner(J_(d_["n-1"])*sigma_f_u(v_["n-1"], d_["n-1"], mu_f)*inv(F_(d_["n-1"])).T, grad(psi))*dx_f
    F_fluid_nonlinear += Constant(theta)*inner(div(J_(d_["n"])*inv(F_(d_["n"]))*v_["n"]), gamma)*dx_f
    F_fluid_nonlinear += Constant(1 - theta)*inner(div(J_(d_["n-1"])*inv(F_(d_["n-1"]))*v_["n-1"]), gamma)*dx_f
    #F_fluid_nonlinear +=inner(div(J_(d_["n"])*inv(F_(d_["n"]))*v_["n"]), gamma)*dx_f

    return dict(F_fluid_linear = F_fluid_linear, F_fluid_nonlinear = F_fluid_nonlinear)
