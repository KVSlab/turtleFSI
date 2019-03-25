from dolfin import *
import numpy as np


def extrapolate_setup(F_fluid_linear, extype, mesh_file, d_, phi, gamma, dx_f, P, **semimp_namespace):
    def F_(U):
        return Identity(len(U)) + grad(U)

    def J_(U):
        return det(F_(U))

    def eps(U):
        return 0.5*(grad(U)*inv(F_(U)) + inv(F_(U)).T*grad(U).T)

    def STVK(U, alfa_mu, alfa_lam):
        return alfa_lam*tr(eps(U))*Identity(len(U)) + 2.0*alfa_mu*eps(U)
        # return F_(U)*(alfa_lam*tr(eps(U))*Identity(len(U)) + 2.0*alfa_mu*eps(U))

    Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24 <= x[0] <= 0.6)  # only the "flag" or "bar"
    #domains = CellFunction("size_t", mesh)
    # domains.set_all(1)
    # Bar_area.mark(domains, 2) #Overwrites structure domain
    cell_domains = CellFunction('size_t', mesh_file, 0)
    solid = '&&'.join(['((0.24 - TOL < x[0]) && (x[0] < 0.6 + TOL))',
                       '((0.19 - TOL < x[1]) && (x[1] < 0.21 + TOL))'])
    solid = CompiledSubDomain(solid, TOL=DOLFIN_EPS)
    # Int so that solid point distance to fluid is 0
    distance_f = VertexFunction('double', mesh_file, 1)
    solid.mark(distance_f, 0)

    # Fluid vertices
    fluid_vertex_ids = np.where(distance_f.get_local() > 0.02)[0]

    # Represent solid as its own mesh for ditance queries
    solid.mark(cell_domains, 1)
    solid_mesh = SubMesh(mesh_file, cell_domains, 1)
    tree = solid_mesh.bounding_box_tree()

    # Fill
    for vertex_id in fluid_vertex_ids:
        vertex = Vertex(mesh_file, vertex_id)
        _, dist = tree.compute_closest_entity(vertex.point())
        distance_f[vertex] = dist

    # Build representation in a CG1 Function
    #V = FunctionSpace(mesh_file, 'CG', 1)
    alfa = Function(P)
    transform = dof_to_vertex_map(P)
    data = distance_f.get_local()[transform]
    alfa.vector().set_local(data)
    alfa.vector().apply('insert')
    hmin = mesh_file.hmin()
    #E_y =  1./(J_(d_["n"]))
    # nu = -0.2 #(-1, 0.5)
    #E_y = 1./CellVolume(mesh_file)
    #nu = 0.25
    E_y = 1./alfa
    nu = 0.1
    alfa_lam = nu*E_y / ((1. + nu)*(1. - 2.*nu))
    alfa_mu = E_y/(2.*(1. + nu))
    #alfa_lam = hmin*hmin ; alfa_mu = hmin*hmin
    F_extrapolate = inner(J_(d_["n"])*STVK(d_["n"], alfa_mu, alfa_lam)*inv(F_(d_["n"])).T, grad(phi))*dx_f
    #F_extrapolate = inner(STVK(d_["n"],alfa_mu,alfa_lam) , grad(phi))*dx_f

    F_fluid_linear += F_extrapolate

    return dict(F_fluid_linear=F_fluid_linear)
