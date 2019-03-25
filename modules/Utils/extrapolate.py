from fenics import *
def extrapolate(d_exact, d_sol, chi, eta):
    bc_d = DirichletBC(V, d_exact, "on_boundary")
    a = -inner(grad(chi), grad(eta))*dx
    L = inner(Constant((0, 0)), eta)*dx
    solve(a == L, d_exp, bc_d)
    return d_exp
