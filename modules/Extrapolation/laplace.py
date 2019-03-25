from dolfin import *
import numpy as np
#from semi_implicit import *


def extrapolate_setup(F_fluid_linear, P, mesh_file, d_, phi, dx_f, **semimp_namespace):

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
    #plot(f, interactive=True)

    F_extrapolate = inner(1./alfa*grad(d_["n"]), grad(phi))*dx_f
    F_fluid_linear += F_extrapolate

    return dict(F_fluid_linear=F_fluid_linear)
