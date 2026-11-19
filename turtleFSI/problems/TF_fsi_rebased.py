"""Diagnostic replay on the saved parameter-specific initial mesh.

The fixed shape map is identity. Only use through the identity-map ROM check,
which retains the original ALE extension diffusivity separately.
"""
from pathlib import Path
import numpy as np
from dolfin import Mesh, MeshFunction, HDF5File, Function, VectorFunctionSpace
from turtleFSI.problems.TF_fsi_geometry import *


def get_mesh_domain_and_boundaries(reference_run_dir, **namespace):
    source = Path(reference_run_dir) / 'Mesh'
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), str(source / 'mesh.h5'), 'r') as handle:
        handle.read(mesh, '/mesh', False)
        domains = MeshFunction('size_t', mesh, 2)
        boundaries = MeshFunction('size_t', mesh, 1)
        handle.read(domains, '/domains')
        handle.read(boundaries, '/boundaries')
    with np.load(source / 'geometry_map.npz') as data:
        np.testing.assert_array_equal(mesh.cells(), data['cells'])
        np.testing.assert_allclose(mesh.coordinates(), data['reference_coordinates'], atol=1e-12)
        mesh.coordinates()[:] = data['mapped_coordinates']
    mesh.bounding_box_tree().build(mesh)
    return mesh, domains, boundaries


def setup_reference_geometry(mesh, **namespace):
    return dict(shape_displacement=Function(VectorFunctionSpace(mesh, 'CG', 1)))
