"""Flag FSI on a common reference mesh: x_current = X + g_mu + d."""

import json
from pathlib import Path

from turtleFSI.problems.TF_fsi_geometry import *
from turtleFSI.problems import TF_fsi as _base
from turtleFSI.cases.flag_geometry import REFERENCE, build_flag_reference, DEFAULT_MAPPING, LEGACY_MAPPING
from turtleFSI.modules.common import F_, G_, J_, K_, paper_fluid_indicator


def get_mesh_domain_and_boundaries(H, L, f_H, c_x, c_y, **namespace):
    for name, value in dict(H=H, L=L, f_H=f_H, c_x=c_x, c_y=c_y).items():
        if value != REFERENCE[name]:
            raise ValueError("Reference flag map varies only R and f_L")
    return _base.get_mesh_domain_and_boundaries(**REFERENCE)


def setup_reference_geometry(mesh, domains, boundaries, R, f_L,
                             extrapolation, extrapolation_sub_type,
                             results_folder=None, reference_run_dir=None, rom_replay=False,
                             restart_folder=None, **namespace):
    if extrapolation != "laplace" or extrapolation_sub_type != "paper":
        raise ValueError("Reference flag case requires laplace/paper extension")
    # Old snapshots must retain their original pullback, even after the default
    # construction changes. Unversioned artifacts were produced by v1.
    if reference_run_dir is None and restart_folder is not None:
        checkpoint = Path(restart_folder)
        reference_run_dir = checkpoint.parent if checkpoint.name == "Checkpoint" else checkpoint
    mapping = DEFAULT_MAPPING
    if reference_run_dir is not None:
        report_path = Path(reference_run_dir) / "Mesh" / "geometry_map.json"
        mapping = json.loads(report_path.read_text()).get("mapping_version", LEGACY_MAPPING)
    geometry = build_flag_reference(R, f_L, baseline=(mesh, domains, boundaries), mapping=mapping)
    if reference_run_dir is not None:
        source = Path(reference_run_dir) / "Mesh"
        with np.load(source / "geometry_map.npz", allow_pickle=False) as saved:
            for key in ("reference_coordinates", "mapped_coordinates"):
                np.testing.assert_allclose(saved[key], geometry[key], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(saved["cells"], mesh.cells())
            np.testing.assert_array_equal(saved["cell_markers"], domains.array())
            np.testing.assert_array_equal(saved["facet_markers"], boundaries.array())
        shape = geometry["shape_displacement"]
        expected = shape.vector().get_local().copy()
        with HDF5File(mesh.mpi_comm(), str(source / "shape_displacement.h5"), "r") as hdf:
            hdf.read(shape, "/shape_displacement")
        np.testing.assert_allclose(shape.vector().get_local(), expected, rtol=0, atol=1e-12)
    if results_folder is not None and not rom_replay:
        folder = Path(results_folder) / "Mesh"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "geometry_map.json").write_text(json.dumps(geometry["report"], indent=2))
        np.savez_compressed(folder / "geometry_map.npz",
                            reference_coordinates=geometry["reference_coordinates"],
                            mapped_coordinates=geometry["mapped_coordinates"],
                            cells=mesh.cells(), cell_markers=domains.array(),
                            facet_markers=boundaries.array(),
                            G=geometry["cell_G"], K=geometry["cell_K"])
        with HDF5File(mesh.mpi_comm(), str(folder / "shape_displacement.h5"), "w") as hdf:
            hdf.write(geometry["shape_displacement"], "/shape_displacement")
    return {key: geometry[key] for key in
            ("shape_displacement", "shape_map", "shape_G", "shape_K")}


def initiate(**namespace):
    # Probe coordinates belong to the computational reference mesh.
    args = dict(namespace)
    args.update({key: REFERENCE[key] for key in ("c_x", "c_y", "R", "f_L")})
    if namespace.get("rom_replay"):
        return _base.initiate(**args)
    from turtleFSI.cases.flag_monitor import FlagJacobianMonitor
    result = _base.initiate(**args)
    result["jacobian_monitor"] = FlagJacobianMonitor(
        namespace["mesh"], namespace["dvp_"]["n"].sub(0), namespace["shape_displacement"])
    history = Path(namespace["results_folder"]) / "reference_history.csv"
    if not namespace.get("restart_folder") or not history.exists():
        history.write_text("time,tip_dx,tip_dy,drag,lift,min_J,min_JK\n")
    result["reference_history_path"] = history
    return result


def post_solve(t, dvp_, coord, displacement_x_list, displacement_y_list,
               drag_list, lift_list, time_list, mu_f, n, ds, dS,
               dx_f, dx_f_id_list, shape_displacement, jacobian_monitor,
               reference_history_path, **namespace):
    min_J, min_JK = jacobian_monitor.minimum()
    if min_J <= 0 or min_JK <= 0:
        with reference_history_path.open("a") as handle:
            handle.write("%g,nan,nan,nan,nan,%.16g,%.16g\n" % (t, min_J, min_JK))
        raise RuntimeError("Non-positive deformation Jacobian at t=%g: J=%g, JK=%g" %
                           (t, min_J, min_JK))
    d, v, p = dvp_["n"].split(deepcopy=True)
    g = shape_displacement
    # Piola traction: current stress integrated using reference surface measure.
    P = (K_(d, g) * J_(d, g) * sigma(v, p, d, mu_f, g)
         * inv(F_(d, g)).T * inv(G_(d, g)).T)
    indicator = paper_fluid_indicator(dx_f, dx_f_id_list)
    traction = sum(indicator(side) * P(side) * n(side) for side in ("+", "-"))
    force = [-assemble((P * n)[i] * ds(6) + traction[i] * dS(5)) for i in range(2)]
    drag_list.append(force[0])
    lift_list.append(force[1])
    time_list.append(t)
    displacement = peval(d, coord)
    displacement_x_list.append(displacement[0])
    displacement_y_list.append(displacement[1])
    row = [t, displacement[0], displacement[1], force[0], force[1], min_J, min_JK]
    with reference_history_path.open("a") as handle:
        handle.write(",".join("%.16g" % value for value in row) + "\n")
    if not np.isfinite(row).all():
        raise RuntimeError("Non-finite reference diagnostics at t=%g" % t)
    print("Reference monitor: t=%g min J=%.6g min JK=%.6g tip=(%.6g, %.6g)" %
          (t, min_J, min_JK, displacement[0], displacement[1]), flush=True)
