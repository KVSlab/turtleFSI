"""Flag/circle geometry variants on a shared topology; one MPI rank per case."""

from turtleFSI.problems.TF_fsi import *
from turtleFSI.problems import TF_fsi as _base
from turtleFSI.cases.flag_geometry import REFERENCE, build_flag_mesh


def set_problem_parameters(default_variables, **namespace):
    values = _base.set_problem_parameters(default_variables, **namespace)
    values.update(fsi_id=5, extrapolation="laplace", extrapolation_sub_type="paper",
                  theta=1.0, material_model="LinearElastic")
    return values


def get_mesh_domain_and_boundaries(R, f_L, H, L, f_H, c_x, c_y, results_folder=None, **namespace):
    import json
    from pathlib import Path

    for name, value in dict(H=H, L=L, f_H=f_H, c_x=c_x, c_y=c_y).items():
        if value != REFERENCE[name]:
            raise ValueError("This map varies only R and f_L; keep %s=%s" % (name, REFERENCE[name]))
    mesh, domains, boundaries, report, reference = build_flag_mesh(R, f_L)
    if results_folder is not None:
        folder = Path(results_folder) / "Mesh"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "geometry_map.json").write_text(json.dumps(report, indent=2))
        np.savez_compressed(folder / "geometry_map.npz", reference_coordinates=reference,
                            mapped_coordinates=mesh.coordinates(), cells=mesh.cells(),
                            cell_markers=domains.array(), facet_markers=boundaries.array())
    print("Geometry map:", report)
    return mesh, domains, boundaries
