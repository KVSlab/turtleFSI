from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
    solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
    MPI


def F_(d):
    return Identity(len(d)) + grad(d)


def J_(d):
    return det(F_(d))


def E(d):
    return 0.5*(F_(d).T*F_(d) - Identity(len(d)))


def S(d, lamda_s, mu_s):
    I = Identity(len(d))
    return 2*mu_s*E(d) + lamda_s*tr(E(d))*I


def Piola1(d, lamda_s, mu_s):
    return F_(d)*S(d, lamda_s, mu_s)


def A_E(d, v,  lamda_s, mu_s, rho_s, delta, psi, phi, dx_s):
    return inner(Piola1(d, lamda_s, mu_s), grad(psi))*dx_s \
        - delta*rho_s*inner(v, phi)*dx_s


def structure_setup(d_, v_, phi, psi, dx_s, mu_s, rho_s, lamda_s, k, theta, **monolithic):

    delta = 1E10
    F_solid_linear = rho_s/k*inner(v_["n"] - v_["n-1"], psi)*dx_s \
                   + delta*(1/k)*inner(d_["n"] - d_["n-1"], phi)*dx_s \
                   - delta*inner(Constant(theta)*v_["n"] + Constant(1 - theta)*v_["n-1"], phi)*dx_s

    F_solid_nonlinear = inner(Piola1(Constant(theta)*d_["n"] + Constant(1 - theta)*d_["n-1"], lamda_s, mu_s), grad(psi))*dx_s

    return dict(F_solid_linear=F_solid_linear, F_solid_nonlinear=F_solid_nonlinear)
