"""Cellwise deformation minima for triangular CG2 displacement / CG1 shape."""
import numpy as np


def quadratic_minimum(coefficients):
    """Minimum of a+b*r+c*s+d*r²+e*r*s+f*s² on r,s>=0, r+s<=1."""
    a, b, c, d, e, f = np.asarray(coefficients).T
    result = np.minimum.reduce([a, a+b+d, a+c+f])
    # Each edge is A+B*t+C*t², 0<=t<=1.
    for A, B, C in ((a, b, d), (a, c, f), (a+c+f, b-c+e-2*f, d-e+f)):
        t = np.divide(-B, 2*C, out=np.zeros_like(B), where=C > 0)
        valid = (C > 0) & (t > 0) & (t < 1)
        result = np.minimum(result, np.where(valid, A+B*t+C*t*t, np.inf))
    determinant = 4*d*f-e*e
    positive = (d > 0) & (determinant > 0)
    r = np.divide(e*c-2*f*b, determinant, out=np.zeros_like(a), where=positive)
    s = np.divide(e*b-2*d*c, determinant, out=np.zeros_like(a), where=positive)
    valid = positive & (r > 0) & (s > 0) & (r+s < 1)
    value = a+b*r+c*s+d*r*r+e*r*s+f*s*s
    return np.minimum(result, np.where(valid, value, np.inf))


class FlagJacobianMonitor:
    """Recover the quadratic J exactly up to FE assembly roundoff per cell.

    Includes edge and interior extrema, not just quadrature-point samples.
    This tests local orientation; it does not detect global self-contact.
    """
    def __init__(self, mesh, displacement, shape):
        from dolfin import (FunctionSpace, Function, TrialFunction, TestFunction,
                            LocalSolver, dx)
        from turtleFSI.modules.common import J_, K_
        if displacement.ufl_element().degree() != 2 or shape.ufl_element().degree() != 1:
            raise ValueError("Flag Jacobian monitor requires CG2 displacement and CG1 shape")
        Q = FunctionSpace(mesh, "DG", 2)
        self.value = Function(Q)
        u, v = TrialFunction(Q), TestFunction(Q)
        measure = dx(domain=mesh, metadata={"quadrature_degree": 4})
        self.solver = LocalSolver(u*v*measure, J_(displacement, shape)*v*measure)
        self.solver.factorize()
        self.dofs = np.array([Q.dofmap().cell_dofs(i) for i in range(mesh.num_cells())])
        coordinates = Q.tabulate_dof_coordinates().reshape(-1, 2)[self.dofs]
        vertices = mesh.coordinates()[mesh.cells()]
        B = np.stack((vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]), axis=2)
        rs = np.einsum("cij,ckj->cki", np.linalg.inv(B), coordinates-vertices[:, None, 0])
        r, s = rs[:, :, 0], rs[:, :, 1]
        self.inverse = np.linalg.inv(np.stack((np.ones_like(r), r, s, r*r, r*s, s*s), axis=2))
        from dolfin import project
        K = project(K_(shape, shape), FunctionSpace(mesh, "DG", 0))
        self.K = np.array([K.vector()[K.function_space().dofmap().cell_dofs(i)[0]]
                           for i in range(mesh.num_cells())])

    def minimum(self):
        self.solver.solve_local_rhs(self.value)
        values = self.value.vector().get_local()[self.dofs]
        if not np.isfinite(values).all():
            raise ValueError("Non-finite deformation Jacobian")
        coefficients = np.einsum("cij,cj->ci", self.inverse, values)
        minima = quadratic_minimum(coefficients)
        return float(minima.min()), float((minima*self.K).min())
