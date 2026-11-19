"""One-time shape map for the serial Turek flag mesh (not its ALE motion).

Circle boundary vertices follow an angular map; the curved flag attachment
keeps its height. A harmonic radius deformation is followed by an explicit
horizontal length stretch with a wall taper. No mesh entities are recreated.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


REFERENCE = dict(R=0.05, H=0.41, L=2.5, f_L=0.35, f_H=0.02, c_x=0.2, c_y=0.2)
DEFAULT_MAPPING = "radius_then_length_v2"
LEGACY_MAPPING = "harmonic_v1"
TEST_CASES = {
    "baseline": (0.05, 0.35),
    "longer_flag": (0.05, 0.3675),
    "larger_circle": (0.0525, 0.35),
}


def signed_double_area(x, cells):
    p = x[cells]
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def triangle_quality(x, cells):
    """4 sqrt(3) area / sum(edge_length**2), one for equilateral triangles."""
    p = x[cells]
    edges = p - np.roll(p, 1, axis=1)
    return 2 * np.sqrt(3) * np.abs(signed_double_area(x, cells)) / np.sum(edges**2, axis=(1, 2))


def _harmonic_extension(x, cells, fixed, displacement):
    """Assemble a scalar P1 Laplace matrix; solve both coordinate components."""
    p = x[cells]
    area2 = signed_double_area(x, cells)
    if np.any(np.abs(area2) < 1e-16):
        raise ValueError("Degenerate reference triangle")
    gradients = np.stack([
        np.stack([p[:, 1, 1] - p[:, 2, 1], p[:, 2, 0] - p[:, 1, 0]], axis=1),
        np.stack([p[:, 2, 1] - p[:, 0, 1], p[:, 0, 0] - p[:, 2, 0]], axis=1),
        np.stack([p[:, 0, 1] - p[:, 1, 1], p[:, 1, 0] - p[:, 0, 0]], axis=1),
    ], axis=1) / area2[:, None, None]
    local = np.einsum("cik,cjk->cij", gradients, gradients) * np.abs(area2[:, None, None]) / 2
    rows = np.broadcast_to(cells[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(cells[:, None, :], local.shape).ravel()
    stiffness = coo_matrix((local.ravel(), (rows, cols)), shape=(len(x), len(x))).tocsr()
    free = np.flatnonzero(~fixed)
    known = np.flatnonzero(fixed)
    if len(free):
        displacement[free] = spsolve(
            stiffness[free][:, free].tocsc(),
            -stiffness[free][:, known] @ displacement[known],
        )
    return displacement


def map_flag_vertices(x, cells, solid_vertices, circle_vertices, wall_vertices,
                      radius, flag_length, min_area_ratio=0.2, min_quality=0.02,
                      mapping=DEFAULT_MAPPING):
    """Map the baseline mesh, rejecting invalid or poorly shaped output cells."""
    if not np.isfinite([radius, flag_length]).all():
        raise ValueError("Geometry parameters must be finite")
    if mapping not in (DEFAULT_MAPPING, LEGACY_MAPPING):
        raise ValueError("Unknown flag geometry mapping: %s" % mapping)
    r0, length0 = REFERENCE["R"], REFERENCE["f_L"]
    cx, cy, half = REFERENCE["c_x"], REFERENCE["c_y"], REFERENCE["f_H"] / 2
    tip0, tip = cx + r0 + length0, cx + radius + flag_length
    if not half < radius < min(cx, cy, REFERENCE["H"] - cy):
        raise ValueError("Circle must fit inside the channel and exceed the flag half-height")
    if flag_length <= 0 or tip >= REFERENCE["L"]:
        raise ValueError("Flag length must be positive and its tip upstream of the outlet")
    x = np.asarray(x, dtype=float)
    def boundary_target(radius, flag_length):
        target = x.copy()
        tip = cx + radius + flag_length
        # The exterior circle arc excludes the flag root. Move its endpoints to
        # the new circle/flag intersections while keeping flag thickness fixed.
        theta = np.mod(np.arctan2(x[circle_vertices, 1] - cy, x[circle_vertices, 0] - cx), 2 * np.pi)
        alpha0, alpha = np.arcsin(half / r0), np.arcsin(half / radius)
        mapped_theta = alpha + (theta - alpha0) * (2 * np.pi - 2 * alpha) / (2 * np.pi - 2 * alpha0)
        target[circle_vertices] = np.column_stack([
            cx + radius * np.cos(mapped_theta), cy + radius * np.sin(mapped_theta),
        ])
        # Root x(y) is curved. Preserve y, attach to the new circle, and map
        # the reference tip to cx + radius + flag_length.
        y = x[solid_vertices, 1] - cy
        root0 = cx + np.sqrt(r0**2 - y**2)
        root = cx + np.sqrt(radius**2 - y**2)
        fraction = (x[solid_vertices, 0] - root0) / (tip0 - root0)
        target[solid_vertices, 0] = root + fraction * (tip - root)
        target[solid_vertices, 1] = x[solid_vertices, 1]
        target[wall_vertices] = x[wall_vertices]
        return target

    fixed = np.zeros(len(x), dtype=bool)
    fixed[np.concatenate([solid_vertices, circle_vertices, wall_vertices])] = True
    target = x.copy()
    steps = 0
    if radius != r0 or flag_length != length0:
        if mapping == LEGACY_MAPPING:
            target = x + _harmonic_extension(x, cells, fixed, boundary_target(radius, flag_length) - x)
            steps = 1
        else:
            # Separate radius and length: a harmonic radius-only map followed
            # by an explicit horizontal stretch. This distributes tip motion
            # through the downstream channel instead of compressing tip cells.
            target = x + _harmonic_extension(x, cells, fixed,
                                             boundary_target(radius, length0) - x)
            radius_ratio = signed_double_area(target, cells) / signed_double_area(x, cells)
            if not np.isfinite(target).all() or np.any(radius_ratio <= 0):
                raise ValueError("Rejected mesh: invalid radius deformation")
            y = target[:, 1]
            anchor = cx + np.sqrt(radius**2 - np.minimum((y-cy)**2, half**2))
            old_tip = cx + radius + length0
            longitudinal = np.where(target[:, 0] <= old_tip,
                                    np.maximum(0, (target[:, 0]-anchor)/(old_tip-anchor)),
                                    (REFERENCE["L"]-target[:, 0])/(REFERENCE["L"]-old_tip))
            taper = np.minimum(1, np.minimum(y/(cy-half),
                                           (REFERENCE["H"]-y)/(REFERENCE["H"]-cy-half)))
            target[:, 0] += (flag_length-length0) * longitudinal * taper
            target[fixed] = boundary_target(radius, flag_length)[fixed]
            steps = 2

    ratio = signed_double_area(target, cells) / signed_double_area(x, cells)
    quality = triangle_quality(target, cells)
    if not np.isfinite(target).all() or not np.isfinite(ratio).all():
        raise ValueError("Non-finite geometry map")
    if ratio.min() < min_area_ratio or quality.min() < min_quality:
        raise ValueError("Rejected mesh: min area ratio %.6g, min quality %.6g" % (ratio.min(), quality.min()))
    return target, {
        "radius": float(radius), "flag_length": float(flag_length),
        "geometry_steps": steps,
        "mapping_version": mapping,
        "tip_reference": [float(tip), cy],
        "min_cell_area_ratio": float(ratio.min()),
        "max_cell_area_ratio": float(ratio.max()),
        "min_triangle_quality": float(quality.min()),
        "max_vertex_shift": float(np.linalg.norm(target - x, axis=1).max()),
    }


def build_flag_mesh(radius=0.05, flag_length=0.35, baseline=None, mapping=DEFAULT_MAPPING):
    """Return mapped mesh, retained tags, report, and common reference vertices.

    Deliberately serial: run independent geometries concurrently, not one
    geometry distributed over MPI ranks. This keeps vertex/DoF identity simple.
    """
    from dolfin import MPI, facets
    from turtleFSI.problems import TF_fsi

    if MPI.size(MPI.comm_world) != 1:
        raise ValueError("Geometry mapping currently requires one MPI rank per case")
    mesh, domains, boundaries = (TF_fsi.get_mesh_domain_and_boundaries(**REFERENCE)
                                 if baseline is None else baseline)
    mesh.init(1, 2)
    reference = mesh.coordinates().copy()
    cells = mesh.cells().copy()
    solid = np.unique(cells[domains.array() == 2])
    circle, walls, interface = [], [], []
    interface_count = 0
    for facet in facets(mesh):
        tag = int(boundaries[facet])
        vertices = facet.entities(0)
        neighbors = facet.entities(2)
        labels = set(int(domains.array()[i]) for i in neighbors)
        actual_interface = labels == {1, 2}
        if actual_interface != (tag == 5):
            raise ValueError("FSI facet marker does not match fluid/solid adjacency")
        if actual_interface:
            interface_count += 1
            interface.extend(vertices)
        if tag == 6:
            circle.extend(vertices)
        if tag in (2, 3, 4):
            walls.extend(vertices)
    if not interface_count or not len(solid) or not circle or not walls:
        raise ValueError("Missing baseline geometry entities")
    target, report = map_flag_vertices(
        reference, cells, solid, np.unique(circle), np.unique(walls), radius, flag_length,
        mapping=mapping,
    )
    mesh.coordinates()[:] = target
    mesh.bounding_box_tree().build(mesh)
    report.update(num_vertices=mesh.num_vertices(), num_cells=mesh.num_cells(),
                  interface_facets=interface_count, interface_vertices=len(np.unique(interface)),
                  mapping=("harmonic radius deformation followed by tapered horizontal length stretch"
                           if mapping == DEFAULT_MAPPING else "legacy harmonic fluid extension"),
                  configuration="parameter-specific initial mesh; physical displacement starts at zero")
    return mesh, domains, boundaries, report, reference


def build_flag_reference(radius=0.05, flag_length=0.35, baseline=None, mapping=DEFAULT_MAPPING):
    """Build T=X+g on the unchanged reference mesh, with cellwise G and K.

    ``baseline`` may supply a fresh baseline mesh and its tags. Its coordinates
    are restored before returning; no parameter-dependent FE space is created.
    Physical displacement is a separate field, initially zero.
    """
    from dolfin import Function, VectorFunctionSpace, vertex_to_dof_map, SpatialCoordinate
    from turtleFSI.modules.common import G_, K_

    mesh, domains, boundaries, report, reference = build_flag_mesh(
        radius, flag_length, baseline=baseline, mapping=mapping)
    mapped = mesh.coordinates().copy()
    mesh.coordinates()[:] = reference
    mesh.bounding_box_tree().build(mesh)
    space = VectorFunctionSpace(mesh, "CG", 1)
    shape = Function(space, name="shape_displacement")
    values = shape.vector().get_local()
    values[vertex_to_dof_map(space)] = (mapped - reference).ravel()
    shape.vector().set_local(values)
    shape.vector().apply("insert")
    old = reference[mesh.cells()]
    new = mapped[mesh.cells()]
    B0 = np.stack([old[:, 1] - old[:, 0], old[:, 2] - old[:, 0]], axis=2)
    B = np.stack([new[:, 1] - new[:, 0], new[:, 2] - new[:, 0]], axis=2)
    cell_G = B @ np.linalg.inv(B0)
    cell_K = np.linalg.det(cell_G)
    report["configuration"] = "common reference mesh; current coordinates are T + physical displacement"
    return dict(mesh=mesh, domains=domains, boundaries=boundaries, report=report,
                reference_coordinates=reference, mapped_coordinates=mapped,
                shape_displacement=shape, shape_map=SpatialCoordinate(mesh) + shape,
                shape_G=G_(shape, shape), shape_K=K_(shape, shape),
                cell_G=cell_G, cell_K=cell_K)
