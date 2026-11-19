import json

import numpy as np
import sys

from turtleFSI.rom._pod_galerkin import build_monolithic_pod_basis
from turtleFSI.rom._pod_galerkin.monolithic_pod import (
    load_monolithic_pod_basis,
    reconstruct_monolithic_fields,
    reconstruct_monolithic_field_coefficients,
    train_monolithic_pod_basis,
)
from turtleFSI.rom._shared.pod import project_to_basis, relative_l2_error
from turtleFSI.rom._pod_galerkin.reduced_solvers import NonlinearGalerkinProblem, reduced_newton_solve


def test_monolithic_pod_concatenates_and_splits_fields():
    time = np.linspace(0.0, 1.0, 6)
    blocks = {
        "d": np.vstack([time, time**2]),
        "v": np.vstack([1.0 - time]),
    }

    basis, snapshots = train_monolithic_pod_basis(
        blocks=blocks,
        field_order=["d", "v"],
        energy_threshold=1.0,
        max_modes=3,
    )
    coefficients = project_to_basis(snapshots, basis.pod_basis)
    fields = reconstruct_monolithic_fields(coefficients, basis)

    assert [field.name for field in basis.fields] == ["d", "v"]
    assert fields["d"].shape == blocks["d"].shape
    assert fields["v"].shape == blocks["v"].shape
    assert relative_l2_error(blocks["d"], fields["d"]) < 1e-12
    assert relative_l2_error(blocks["v"], fields["v"]) < 1e-12


def test_reduced_newton_solves_projected_nonlinear_residual():
    target = np.array([1.0, -2.0, 0.5])
    basis, _ = train_monolithic_pod_basis(
        blocks={"state": np.column_stack([target, 0.5 * target, 1.5 * target])},
        field_order=["state"],
        energy_threshold=1.0,
        max_modes=1,
    )

    def residual(y):
        return y**3 - target**3

    def jacobian(y):
        return np.diag(3.0 * y**2)

    problem = NonlinearGalerkinProblem(
        basis=basis.pod_basis,
        residual=residual,
        jacobian=jacobian,
    )
    initial = np.array([[0.25]])
    result = reduced_newton_solve(
        problem,
        initial_coefficients=initial,
        atol=1e-11,
        rtol=1e-11,
        max_iterations=20,
    )

    assert result.converged
    assert np.linalg.norm(basis.pod_basis.basis.T @ residual(result.full_state[:, 0])) < 1e-10


def test_build_monolithic_pod_basis_accepts_dvp_bundle(tmp_path, monkeypatch):
    snapshots = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 4.0, 8.0],
            [0.5, 0.25, 0.125, 0.0],
        ],
        dtype=float,
    )
    snapshots_path = tmp_path / "dvp_snapshots.npz"
    output_dir = tmp_path / "basis"
    np.savez_compressed(snapshots_path, dvp=snapshots, time=np.arange(snapshots.shape[1]))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_monolithic_pod_basis",
            "--snapshots",
            str(snapshots_path),
            "--output-dir",
            str(output_dir),
            "--energy-threshold",
            "1.0",
        ],
    )
    build_monolithic_pod_basis.main()

    basis = load_monolithic_pod_basis(output_dir / "monolithic_pod_basis.npz")
    assert [field.name for field in basis.fields] == ["dvp"]
    assert basis.num_dofs == snapshots.shape[0]


def test_build_blocked_dvp_basis_uses_separate_field_spaces(tmp_path, monkeypatch):
    snapshots = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 9.0, 8.0, 7.0],
            [2.0, 4.0, 8.0, 16.0],
            [5.0, 5.5, 6.0, 6.5],
            [100.0, 110.0, 120.0, 130.0],
        ],
        dtype=float,
    )
    snapshots_path = tmp_path / "dvp_snapshots.npz"
    output_dir = tmp_path / "blocked_basis"
    np.savez_compressed(
        snapshots_path,
        dvp=snapshots,
        displacement_dofs=np.array([0, 3], dtype=np.int64),
        velocity_dofs=np.array([1, 4], dtype=np.int64),
        pressure_dofs=np.array([2, 5], dtype=np.int64),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_monolithic_pod_basis",
            "--snapshots",
            str(snapshots_path),
            "--output-dir",
            str(output_dir),
            "--blocked-dvp",
            "--energy-threshold",
            "1.0",
            "--displacement-modes",
            "1",
            "--velocity-modes",
            "1",
            "--pressure-modes",
            "1",
        ],
    )
    build_monolithic_pod_basis.main()

    basis = load_monolithic_pod_basis(output_dir / "monolithic_pod_basis.npz")
    assert [field.name for field in basis.fields] == ["displacement", "velocity", "pressure"]
    assert basis.num_modes == 3
    assert basis.fields[0].dofs.tolist() == [0, 3]
    assert basis.fields[1].dofs.tolist() == [1, 4]
    assert basis.fields[2].dofs.tolist() == [2, 5]
    assert np.allclose(basis.pod_basis.basis[[1, 2, 4, 5], 0], 0.0)
    assert np.allclose(basis.pod_basis.basis[[0, 2, 3, 5], 1], 0.0)
    assert np.allclose(basis.pod_basis.basis[[0, 1, 3, 4], 2], 0.0)

    coefficients = basis.pod_basis.basis.T @ (snapshots - basis.pod_basis.mean)
    fields = reconstruct_monolithic_fields(coefficients, basis)
    coefficient_blocks = reconstruct_monolithic_field_coefficients(coefficients, basis)

    assert fields["displacement"].shape == (2, snapshots.shape[1])
    assert fields["velocity"].shape == (2, snapshots.shape[1])
    assert fields["pressure"].shape == (2, snapshots.shape[1])
    assert coefficient_blocks["displacement"].shape == (1, snapshots.shape[1])
    assert coefficient_blocks["velocity"].shape == (1, snapshots.shape[1])
    assert coefficient_blocks["pressure"].shape == (1, snapshots.shape[1])

    residual = np.arange(snapshots.shape[0], dtype=float).reshape(-1, 1)
    projected = basis.pod_basis.basis.T @ residual
    manual = np.vstack(
        [
            basis.pod_basis.basis[:, 0:1].T @ residual,
            basis.pod_basis.basis[:, 1:2].T @ residual,
            basis.pod_basis.basis[:, 2:3].T @ residual,
        ]
    )
    assert np.allclose(projected, manual)


def test_build_blocked_dvp_basis_adds_supremizer_enrichment(tmp_path, monkeypatch):
    snapshots = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 9.0, 8.0, 7.0],
            [2.0, 4.0, 8.0, 16.0],
            [5.0, 5.5, 6.0, 6.5],
            [100.0, 110.0, 120.0, 130.0],
        ],
        dtype=float,
    )
    supremizer = np.array(
        [
            [0.0, 0.5, 1.0, 1.5],
            [2.0, 1.0, 0.5, 0.25],
        ],
        dtype=float,
    )
    snapshots_path = tmp_path / "dvp_snapshots.npz"
    output_dir = tmp_path / "supremizer_basis"
    np.savez_compressed(
        snapshots_path,
        dvp=snapshots,
        displacement_dofs=np.array([0, 3], dtype=np.int64),
        velocity_dofs=np.array([1, 4], dtype=np.int64),
        pressure_dofs=np.array([2, 5], dtype=np.int64),
        supremizer_snapshots=supremizer,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_monolithic_pod_basis",
            "--snapshots",
            str(snapshots_path),
            "--output-dir",
            str(output_dir),
            "--blocked-dvp",
            "--pressure-treatment",
            "supremizer",
            "--energy-threshold",
            "1.0",
            "--displacement-modes",
            "1",
            "--velocity-modes",
            "1",
            "--pressure-modes",
            "1",
            "--supremizer-modes",
            "1",
        ],
    )
    build_monolithic_pod_basis.main()

    basis = load_monolithic_pod_basis(output_dir / "monolithic_pod_basis.npz")
    assert [field.name for field in basis.fields] == [
        "displacement",
        "velocity",
        "supremizer",
        "pressure",
    ]
    assert basis.num_modes == 4
    assert basis.fields[2].dofs.tolist() == [1, 4]
    assert np.allclose(basis.pod_basis.basis[[0, 2, 3, 5], 2], 0.0)
    gram = basis.pod_basis.basis.T @ basis.pod_basis.basis
    assert np.allclose(gram, np.eye(basis.num_modes), atol=1e-12)


def test_blocked_dvp_basis_summary_reports_section_3_1_spaces(tmp_path, monkeypatch):
    snapshots = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 9.0, 8.0, 7.0],
            [2.0, 4.0, 8.0, 16.0],
            [5.0, 5.5, 6.0, 6.5],
            [100.0, 110.0, 120.0, 130.0],
        ],
        dtype=float,
    )
    supremizer = np.array(
        [
            [0.0, 0.5, 1.0, 1.5],
            [2.0, 1.0, 0.5, 0.25],
        ],
        dtype=float,
    )
    snapshots_path = tmp_path / "dvp_snapshots.npz"
    output_dir = tmp_path / "section_3_1_basis"
    np.savez_compressed(
        snapshots_path,
        dvp=snapshots,
        displacement_dofs=np.array([0, 3], dtype=np.int64),
        velocity_dofs=np.array([1, 4], dtype=np.int64),
        fluid_pressure_dofs=np.array([2, 5], dtype=np.int64),
        supremizer_snapshots=supremizer,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_monolithic_pod_basis",
            "--snapshots",
            str(snapshots_path),
            "--output-dir",
            str(output_dir),
            "--blocked-dvp",
            "--pressure-treatment",
            "supremizer",
            "--pressure-dof-key",
            "fluid_pressure_dofs",
            "--energy-threshold",
            "1.0",
            "--displacement-modes",
            "1",
            "--velocity-modes",
            "1",
            "--pressure-modes",
            "1",
            "--supremizer-modes",
            "1",
        ],
    )
    build_monolithic_pod_basis.main()

    summary = json.loads(
        (output_dir / "monolithic_pod_basis_summary.json").read_text(encoding="ascii")
    )
    section = summary["section_3_1"]
    assert section["paper_section"] == "3.1"
    assert section["snapshot_matrices"]["S_d"]["num_dofs"] == 2
    assert section["snapshot_matrices"]["S_d"]["shape"] == [2, 4]
    assert section["snapshot_matrices"]["S_u"]["num_dofs"] == 2
    assert section["snapshot_matrices"]["S_u"]["shape"] == [2, 4]
    assert section["snapshot_matrices"]["S_p"]["dof_key"] == "fluid_pressure_dofs"
    assert section["snapshot_matrices"]["S_s"]["present"]
    assert section["reduced_spaces"]["E_N"]["num_modes"] == 1
    assert section["reduced_spaces"]["V_N"]["velocity_modes"] == 1
    assert section["reduced_spaces"]["V_N"]["supremizer_modes"] == 1
    assert section["reduced_spaces"]["V_N"]["num_modes"] == 2
    assert section["reduced_spaces"]["Q_N"]["num_modes"] == 1
    assert section["basis_matrices"]["Z_u_s"]["shape"] == [6, 2]
    assert section["supremizer_problem"]["equation"] == "X s = B_F^T p"
    assert section["supremizer_problem"]["embedded_in"] == "velocity"


def test_build_blocked_dvp_basis_can_use_active_gauged_pressure_dofs(tmp_path, monkeypatch):
    snapshots = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 7.0, 9.0, 11.0],
            [2.0, 4.0, 8.0, 16.0],
            [5.0, 5.5, 6.0, 6.5],
            [100.0, 110.0, 120.0, 130.0],
        ],
        dtype=float,
    )
    snapshots_path = tmp_path / "dvp_snapshots.npz"
    output_dir = tmp_path / "active_pressure_basis"
    np.savez_compressed(
        snapshots_path,
        dvp=snapshots,
        displacement_dofs=np.array([0, 3], dtype=np.int64),
        velocity_dofs=np.array([1, 4], dtype=np.int64),
        pressure_dofs=np.array([2, 5], dtype=np.int64),
        fluid_pressure_dofs=np.array([2], dtype=np.int64),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_monolithic_pod_basis",
            "--snapshots",
            str(snapshots_path),
            "--output-dir",
            str(output_dir),
            "--blocked-dvp",
            "--pressure-dof-key",
            "fluid_pressure_dofs",
            "--pressure-gauge",
            "zero-mean",
            "--energy-threshold",
            "1.0",
            "--displacement-modes",
            "1",
            "--velocity-modes",
            "1",
            "--pressure-modes",
            "1",
        ],
    )
    build_monolithic_pod_basis.main()

    basis = load_monolithic_pod_basis(output_dir / "monolithic_pod_basis.npz")
    assert [field.name for field in basis.fields] == ["displacement", "velocity", "pressure"]
    assert basis.fields[2].dofs.tolist() == [2]
    assert np.allclose(basis.pod_basis.mean[2], 0.0)
    assert np.allclose(basis.pod_basis.basis[[0, 1, 3, 4, 5], 2], 0.0)
