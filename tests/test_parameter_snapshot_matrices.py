"""Time/parameter ordering and CLI coverage for explicit paper snapshot matrices."""

import argparse
import json
import sys

import numpy as np
import pytest

from turtleFSI.rom._pod_galerkin import build_monolithic_pod_basis as builder


def _bundle(path, offset=0, ntime=4, velocity_dofs=None, supremizer_columns=None):
    dvp = np.arange(6 * ntime, dtype=float).reshape(6, ntime) + offset
    ns = ntime if supremizer_columns is None else supremizer_columns
    supremizer = np.arange(2 * ns, dtype=float).reshape(2, ns) + offset + 100
    np.savez_compressed(
        path, dvp=dvp, time=np.arange(ntime) * 0.1,
        displacement_dofs=np.array([0, 3]),
        velocity_dofs=np.array([1, 4] if velocity_dofs is None else velocity_dofs),
        fluid_pressure_dofs=np.array([2, 5]),
        supremizer_snapshots=supremizer,
    )
    return dvp, supremizer


def _args():
    return argparse.Namespace(
        start_index=1, stop_index=3, pressure_dof_key="auto",
        pressure_treatment="supremizer",
    )


def test_parameter_blocks_preserve_time_and_parameter_order(tmp_path):
    paths = [tmp_path / "mu1.npz", tmp_path / "mu2.npz"]
    first, s_first = _bundle(paths[0])
    second, s_second = _bundle(paths[1], offset=1000, ntime=5)
    dataset, matrices, metadata = builder._assemble_parameter_snapshots(paths, _args())
    for name, rows in (("d", [0, 3]), ("u", [1, 4]), ("p", [2, 5])):
        np.testing.assert_array_equal(matrices[f"S1_{name}"], first[rows, 1:3])
        np.testing.assert_array_equal(matrices[f"S2_{name}"], second[rows, 1:3])
        np.testing.assert_array_equal(
            dataset[f"S_{name}"], np.hstack([first[rows, 1:3], second[rows, 1:3]])
        )
    np.testing.assert_array_equal(matrices["S_s"], np.hstack([s_first[:, 1:3], s_second[:, 1:3]]))
    np.testing.assert_array_equal(matrices["parameter_column_offsets"], [0, 2, 4])
    np.testing.assert_array_equal(matrices["snapshot_indices_2"], [1, 2])
    np.testing.assert_allclose(matrices["time_2"], [0.1, 0.2])
    assert metadata[1]["column_start"] == 2


@pytest.mark.parametrize("invalid", ["dofs", "supremizer_columns", "empty_window"])
def test_parameter_bundles_reject_misalignment(tmp_path, invalid):
    paths = [tmp_path / "mu1.npz", tmp_path / "mu2.npz"]
    _bundle(paths[0])
    options = {"velocity_dofs": [4, 1]} if invalid == "dofs" else {}
    if invalid == "supremizer_columns":
        options["supremizer_columns"] = 3
    _bundle(paths[1], **options)
    args = _args()
    if invalid == "empty_window":
        args.start_index = 10
    with pytest.raises(ValueError):
        builder._assemble_parameter_snapshots(paths, args)


def test_cli_saves_explicit_matrices_used_for_pod(tmp_path, monkeypatch):
    paths = [tmp_path / "mu1.npz", tmp_path / "mu2.npz"]
    first, _ = _bundle(paths[0])
    second, _ = _bundle(paths[1], offset=1000)
    output = tmp_path / "basis"
    monkeypatch.setattr(sys, "argv", [
        "build_monolithic_pod_basis", "--snapshots", *map(str, paths),
        "--output-dir", str(output), "--blocked-dvp",
        "--pressure-treatment", "supremizer", "--start-index", "1",
        "--stop-index", "3", "--energy-threshold", "1.0",
    ])
    builder.main()
    with np.load(output / "snapshot_matrices.npz") as matrices:
        expected = np.hstack([first[[0, 3], 1:3], second[[0, 3], 1:3]])
        np.testing.assert_array_equal(matrices["S_d"], expected)
        assert matrices["S_s"].shape == (2, 4)
    with np.load(output / "monolithic_pod_projection.npz") as projection:
        np.testing.assert_allclose(projection["reconstruction"][[0, 3]], expected, atol=1e-10)
    with np.load(output / "monolithic_pod_basis.npz") as saved:
        np.testing.assert_array_equal(saved["Z_u_s"], np.hstack([saved["Z_u"], saved["Z_s"]]))
        np.testing.assert_array_equal(
            saved["basis"], np.hstack([saved["Z_d"], saved["Z_u_s"], saved["Z_p"]])
        )
        for key, rows in (("Z_d", [0, 3]), ("Z_u", [1, 4]), ("Z_s", [1, 4]), ("Z_p", [2, 5])):
            other_rows = np.setdiff1d(np.arange(6), rows)
            assert np.all(saved[key][other_rows] == 0)
    summary = json.loads((output / "monolithic_pod_basis_summary.json").read_text())
    section = summary["section_3_1"]
    assert section["num_training_parameters"] == 2
    assert section["snapshot_matrices"]["S_d"]["shape"] == [2, 4]
    assert section["snapshot_matrices"]["S_d"]["snapshot_key"] == "S_d"
    assert section["training_window"]["start_index"] == 1
