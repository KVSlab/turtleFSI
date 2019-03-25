from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI, CellVolume
#from semi_implicit import *


def extrapolate_setup(F_fluid_linear, extype, mesh_file, d_, phi, gamma, dx_f, **semimp_namespace):
    def F_(U):
        return Identity(len(U)) + grad(U)

    def J_(U):
        return det(F_(U))

    def eps(U):
        return 0.5*(grad(U)*inv(F_(U)) + inv(F_(U)).T*grad(U).T)

    def STVK(U, alfa_mu, alfa_lam):
        return alfa_lam*tr(eps(U))*Identity(len(U)) + 2.0*alfa_mu*eps(U)
        #return F_(U)*(alfa_lam*tr(eps(U))*Identity(len(U)) + 2.0*alfa_mu*eps(U))

    alfa = 1.0 # holder value if linear is chosen
    if extype == "det":
        #alfa = inv(J_(d_["n"]))
        alfa = 1./(J_(d_["n"]))
    if extype == "smallconst":
        alfa = 0.01*(mesh_file.hmin())**2
    if extype == "const":
        alfa = 1.0

    F_extrapolate = alfa*inner(grad(d_["n"]), grad(phi))*dx_f

    if extype == "linear":
        hmin = mesh_file.hmin()
        #E_y =  1./(J_(d_["n"]))
        #nu = -0.2 #(-1, 0.5)
        E_y = 1./CellVolume(mesh_file)
        nu = 0.25
        alfa_lam = nu*E_y / ((1. + nu)*(1. - 2.*nu))
        alfa_mu = E_y/(2.*(1. + nu))
        #alfa_lam = hmin*hmin ; alfa_mu = hmin*hmin
        F_extrapolate = inner(J_(d_["n"])*STVK(d_["n"],alfa_mu,alfa_lam)*inv(F_(d_["n"])).T, grad(phi))*dx_f
        #F_extrapolate = inner(STVK(d_["n"],alfa_mu,alfa_lam) , grad(phi))*dx_f

    F_fluid_linear += F_extrapolate

    return dict(F_fluid_linear=F_fluid_linear)
