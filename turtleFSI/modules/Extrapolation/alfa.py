from dolfin import inner, inv, grad, CellVolume


def extrapolate_setup(F_fluid_linear, extype, mesh_file, d_, phi, gamma, dx_f, **semimp_namespace):
    if extype == "linear":
        hmin = mesh_file.hmin()
        E_y = 1./CellVolume(mesh_file)
        nu = 0.25
        alfa_lam = nu*E_y / ((1. + nu)*(1. - 2.*nu))
        alfa_mu = E_y/(2.*(1. + nu))
        F_extrapolate = inner(J_(d_["n"]) * STVK(d_["n"], alfa_mu, alfa_lam) *
                              inv(F_(d_["n"])).T, grad(phi))*dx_f

    else:
        if extype == "det":
            alfa = 1./(J_(d_["n"]))
        elif extype == "smallconst":
            alfa = 0.01*(mesh_file.hmin())**2
        elif extype == "const":
            alfa = 1.0
        else:
            raise RuntimeError("Could not find extrapolation method {}".format(extype))

        F_extrapolate = alfa*inner(grad(d_["n"]), grad(phi))*dx_f

    F_fluid_linear += F_extrapolate

    return dict(F_fluid_linear=F_fluid_linear)
