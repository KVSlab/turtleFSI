import json

import numpy as np

from turtleFSI.rom._pod_galerkin.ballarin_monolithic import (
    block_basis_from_monolithic,
    reconstruct_ballarin_state,
    split_ballarin_state,
)
from turtleFSI.rom._pod_galerkin.build_monolithic_pod_basis import _build_blocked_dvp_basis
from turtleFSI.rom._pod_galerkin.monolithic_pod import save_monolithic_pod_basis
from turtleFSI.rom._pod_galerkin.offline_online import (
    ReducedLinearOperators,
    affine_rhs_from_linearized_residual,
    equilibrate_linear_system,
    evaluate_reduced_quadratic,
    load_reduced_operator_bundle,
    matrix_condition_number,
    project_linear_operator,
    project_linear_operator_blocks,
    project_quadratic_tensor,
    reduced_linear_jacobian,
    reduced_linear_residual,
    save_reduced_operator_bundle,
    solve_equilibrated_lstsq,
    solve_reduced_linear_system,
    solve_reduced_operator_bundle,
    rhs_N_from_equation6_linearization,
)
from turtleFSI.rom._pod_galerkin.problem_overrides import replay_overrides
from turtleFSI.rom._pod_galerkin import run_ballarin_operator_solve
from turtleFSI.rom._pod_galerkin import run_ballarin_operator_series_solve
from turtleFSI.rom import pod_galerkin


def _blocked_monolithic_basis(tmp_path):
    snapshots_path = tmp_path / "snapshots.npz"
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
    np.savez_compressed(
        snapshots_path,
        dvp=snapshots,
        displacement_dofs=np.array([0, 3], dtype=np.int64),
        velocity_dofs=np.array([1, 4], dtype=np.int64),
        pressure_dofs=np.array([2, 5], dtype=np.int64),
        supremizer_snapshots=np.array(
            [[0.0, 0.5, 1.0, 1.5], [2.0, 1.0, 0.5, 0.25]],
            dtype=float,
        ),
    )
    args = type(
        "Args",
        (),
        {
            "displacement_modes": 1,
            "velocity_modes": 1,
            "pressure_modes": 1,
            "supremizer_modes": 1,
            "max_modes": None,
            "pressure_treatment": "supremizer",
        },
    )()
    basis, _, _ = _build_blocked_dvp_basis(
        dataset=np.load(snapshots_path),
        start_index=0,
        stop_index=None,
        energy_threshold=1.0,
        args=args,
    )
    return basis


def _blocked_basis(tmp_path):
    return block_basis_from_monolithic(_blocked_monolithic_basis(tmp_path))


def test_replay_overrides_loads_fom_checkpoint_from_snapshot_summary(tmp_path):
    run_dir = tmp_path / "run" / "case001"
    rom_dir = run_dir / "ROM"
    checkpoint_dir = run_dir / "Checkpoint"
    rom_dir.mkdir(parents=True)
    checkpoint_dir.mkdir()
    snapshots = rom_dir / "dvp_snapshots.npz"
    np.savez_compressed(snapshots, dvp=np.ones((2, 2)))
    (rom_dir / "dvp_snapshots_summary.json").write_text(
        json.dumps({"results_folder": str(run_dir)}),
        encoding="ascii",
    )
    (checkpoint_dir / "default_variables.json").write_text(
        json.dumps({"mesh_path": "/tmp/mesh.h5", "dt": 0.1, "verbose": True}),
        encoding="ascii",
    )

    overrides = replay_overrides(snapshots, explicit_overrides={"dt": 0.01})

    assert overrides["mesh_path"] == "/tmp/mesh.h5"
    assert overrides["dt"] == 0.01
    assert overrides["verbose"] is False


def test_ballarin_block_state_reconstructs_from_named_coefficients(tmp_path):
    basis = _blocked_basis(tmp_path)
    coefficients = np.arange(1, basis.combined_basis.shape[1] + 1, dtype=float)

    state = split_ballarin_state(coefficients, basis)
    full = reconstruct_ballarin_state(state, basis)

    assert state.displacement.shape == (1, 1)
    assert state.velocity.shape == (1, 1)
    assert state.supremizer.shape == (1, 1)
    assert state.pressure.shape == (1, 1)
    assert np.allclose(full, basis.mean + basis.combined_basis @ coefficients[:, None])




def test_explicit_reduced_operator_blocks_match_monolithic_projection(tmp_path):
    basis = _blocked_basis(tmp_path)
    phi = basis.combined_basis
    operator = np.arange(1, phi.shape[0] * phi.shape[0] + 1, dtype=float).reshape(
        phi.shape[0],
        phi.shape[0],
    )
    rhs = np.arange(1, phi.shape[0] + 1, dtype=float)

    reduced = project_linear_operator(operator, rhs, basis)
    blocks = project_linear_operator_blocks(operator, rhs, basis)

    assert blocks.block_names == ("displacement", "velocity", "supremizer", "pressure")
    assert np.allclose(blocks.as_operator(), reduced.J_N)
    assert np.allclose(blocks.as_rhs(), reduced.rhs_N)
    assert blocks.blocks[("displacement", "pressure")].shape == (1, 1)


def test_reduced_operator_bundle_roundtrips_explicit_blocks(tmp_path):
    basis = _blocked_basis(tmp_path)
    phi = basis.combined_basis
    operator = np.diag(np.arange(2, phi.shape[0] + 2, dtype=float))
    rhs = np.arange(1, phi.shape[0] + 1, dtype=float)
    reduced = project_linear_operator(operator, rhs, basis)
    blocks = project_linear_operator_blocks(operator, rhs, basis)
    bundle_path = tmp_path / "reduced_operators.npz"

    save_reduced_operator_bundle(bundle_path, reduced, linear_operator_blocks=blocks)
    bundle = load_reduced_operator_bundle(bundle_path)

    assert bundle.linear_operator_blocks is not None
    assert bundle.linear_operator_blocks.block_mode_counts == blocks.block_mode_counts
    assert np.allclose(bundle.linear_operator_blocks.as_operator(), bundle.J_N)
    assert np.allclose(bundle.linear_operator_blocks.as_rhs(), bundle.rhs_N)


def test_offline_linear_projection_matches_full_projection(tmp_path):
    basis = _blocked_basis(tmp_path)
    phi = basis.combined_basis
    operator = np.diag(np.arange(1, phi.shape[0] + 1, dtype=float))
    rhs = np.arange(phi.shape[0], dtype=float)
    reduced = project_linear_operator(operator, rhs, basis)
    coefficients = np.arange(1, phi.shape[1] + 1, dtype=float).reshape(-1, 1)

    residual = reduced_linear_residual(coefficients, reduced)
    full_state = basis.mean + phi @ coefficients
    expected = phi.T @ (operator @ full_state - rhs[:, None])

    assert np.allclose(residual, expected)
    assert np.allclose(reduced_linear_jacobian(reduced), phi.T @ operator @ phi)


def test_affine_rhs_from_linearized_residual_matches_reference_formula():
    projected_residual = np.array([[2.0], [-1.0], [0.5]])
    projected_shift = np.array([[0.25], [3.0], [-0.5]])

    rhs = affine_rhs_from_linearized_residual(projected_residual, projected_shift)
    rhs_N = rhs_N_from_equation6_linearization(projected_residual, projected_shift)

    assert np.allclose(rhs, -projected_residual - projected_shift)
    assert np.allclose(rhs_N, rhs)


def test_reduced_operator_bundle_solves_without_full_operator(tmp_path):
    basis = _blocked_basis(tmp_path)
    phi = basis.combined_basis
    operator = np.diag(np.arange(2, phi.shape[0] + 2, dtype=float))
    rhs = np.arange(1, phi.shape[0] + 1, dtype=float)
    reduced = project_linear_operator(operator, rhs, basis)

    coefficients, full_state = solve_reduced_linear_system(reduced)
    bundle_path = tmp_path / "reduced_operators.npz"
    projected_residual = np.arange(phi.shape[1], dtype=float).reshape(-1, 1)
    projected_shift = -projected_residual
    save_reduced_operator_bundle(
        bundle_path,
        reduced,
        projected_residual_at_reference=projected_residual,
        projected_jacobian_mean_shift=projected_shift,
        metadata={
            "assembly_stage": "offline",
            "online_requires_fenics_assembly": False,
        },
    )
    bundle = load_reduced_operator_bundle(bundle_path)
    loaded_coefficients = solve_reduced_operator_bundle(bundle)

    assert np.allclose(coefficients, loaded_coefficients)
    assert np.allclose(reduced.operator @ coefficients, reduced.rhs)
    assert np.allclose(bundle.J_N, bundle.operator)
    assert np.allclose(bundle.rhs_N, bundle.rhs)
    assert np.allclose(bundle.R_N_ref, projected_residual)
    assert np.allclose(bundle.J_N_mean_shift, projected_shift)
    assert np.allclose(bundle.projected_residual_at_reference, projected_residual)
    assert np.allclose(bundle.projected_jacobian_mean_shift, projected_shift)
    assert bundle.metadata["assembly_stage"] == "offline"
    assert bundle.metadata["online_requires_fenics_assembly"] is False
    assert full_state.shape == (phi.shape[0], 1)
    assert bundle.block_mode_counts == {
        "displacement": 1,
        "velocity": 1,
        "supremizer": 1,
        "pressure": 1,
    }


def test_equilibrated_lstsq_improves_ill_scaled_reduced_system():
    operator = np.array(
        [
            [1.0e8, 2.0e5, 0.0],
            [3.0e-4, 2.0e-7, 1.0e-7],
            [0.0, 4.0e-6, 3.0e-6],
        ],
        dtype=float,
    )
    expected = np.array([[1.5], [-2.0], [0.25]])
    rhs = operator @ expected

    scaled_operator, _, _, _ = equilibrate_linear_system(operator, rhs)
    result = solve_equilibrated_lstsq(operator, rhs, rcond=1e-14)

    assert matrix_condition_number(scaled_operator) < matrix_condition_number(operator)
    assert result.scaled_condition_number < matrix_condition_number(operator)
    assert np.allclose(result.coefficients, expected)


def test_ballarin_operator_solve_command_writes_solution(tmp_path, monkeypatch):
    monolithic_basis = _blocked_monolithic_basis(tmp_path)
    basis = block_basis_from_monolithic(monolithic_basis)
    phi = basis.combined_basis
    operator = np.diag(np.arange(2, phi.shape[0] + 2, dtype=float))
    rhs = np.arange(1, phi.shape[0] + 1, dtype=float)
    reduced = project_linear_operator(operator, rhs, basis)
    basis_dir = tmp_path / "basis"
    save_monolithic_pod_basis(
        output_dir=basis_dir,
        basis=monolithic_basis,
        snapshots_path="synthetic",
        reconstruction_error=0.0,
    )
    operators_path = tmp_path / "operators.npz"
    save_reduced_operator_bundle(
        operators_path,
        reduced,
        metadata={
            "assembly_stage": "offline",
            "online_requires_fenics_assembly": False,
        },
    )
    output_dir = tmp_path / "online"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ballarin_operator_solve",
            "--basis",
            str(basis_dir / "monolithic_pod_basis.npz"),
            "--operators",
            str(operators_path),
            "--output-dir",
            str(output_dir),
            "--linear-solver",
            "equilibrated-lstsq",
            "--linear-rcond",
            "1e-14",
        ],
    )

    run_ballarin_operator_solve.main()

    data = np.load(output_dir / "ballarin_operator_solution.npz")
    assert data["coefficients"].shape == (4, 1)
    assert data["supremizer_coefficients"].shape == (1, 1)
    summary = json.loads(
        (output_dir / "ballarin_operator_solution_summary.json").read_text(
            encoding="ascii"
        )
    )
    assert summary["linear_solver"] == "equilibrated-lstsq"
    assert summary["equation_6_matrix"] == "J_N"
    assert summary["equation_6_rhs"] == "rhs_N"
    assert summary["online_requires_fenics_assembly"] is False
    assert summary["operator_metadata"]["assembly_stage"] == "offline"
    assert "scaled_condition_number" in summary


def test_ballarin_operator_series_solve_command_writes_validation(tmp_path, monkeypatch):
    monolithic_basis = _blocked_monolithic_basis(tmp_path)
    basis = block_basis_from_monolithic(monolithic_basis)
    phi = basis.combined_basis
    operator = np.diag(np.arange(2, phi.shape[0] + 2, dtype=float))
    rhs = np.arange(1, phi.shape[0] + 1, dtype=float)
    reduced = project_linear_operator(operator, rhs, basis)
    basis_dir = tmp_path / "basis"
    save_monolithic_pod_basis(
        output_dir=basis_dir,
        basis=monolithic_basis,
        snapshots_path="synthetic",
        reconstruction_error=0.0,
    )
    operators_dir = tmp_path / "operators"
    operators_dir.mkdir()
    bundle_paths = []
    for reference_index in (1, 2):
        path = operators_dir / f"reduced_operators_ref{reference_index:04d}.npz"
        save_reduced_operator_bundle(
            path,
            reduced,
            metadata={
                "assembly_stage": "offline",
                "online_requires_fenics_assembly": False,
            },
        )
        bundle_paths.append(path)
    manifest = {
        "bundles": [
            {
                "path": str(path),
                "reference_index": reference_index,
                "reduced_operator_condition_number": 12.0,
            }
            for path, reference_index in zip(bundle_paths, (1, 2))
        ]
    }
    manifest_path = operators_dir / "operator_series_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")
    output_dir = tmp_path / "online_series"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ballarin_operator_series_solve",
            "--basis",
            str(basis_dir / "monolithic_pod_basis.npz"),
            "--manifest",
            str(manifest_path),
            "--snapshots",
            str(tmp_path / "snapshots.npz"),
            "--output-dir",
            str(output_dir),
            "--linear-solver",
            "equilibrated-lstsq",
            "--linear-rcond",
            "1e-14",
        ],
    )

    run_ballarin_operator_series_solve.main()

    data = np.load(output_dir / "ballarin_operator_series_solution.npz")
    assert data["coefficients"].shape == (4, 2)
    assert data["reference_indices"].tolist() == [1, 2]
    summary = json.loads(
        (output_dir / "ballarin_operator_series_solution_summary.json").read_text(
            encoding="ascii"
        )
    )
    assert summary["num_solutions"] == 2
    assert summary["linear_solver"] == "equilibrated-lstsq"
    assert summary["solutions"][0]["equation_6_matrix"] == "J_N"
    assert summary["solutions"][0]["equation_6_rhs"] == "rhs_N"
    assert summary["solutions"][0]["online_requires_fenics_assembly"] is False
    assert summary["solutions"][0]["operator_metadata"]["assembly_stage"] == "offline"
    assert "scaled_condition_number" in summary["solutions"][0]
    assert "field_relative_errors" in summary["solutions"][0]


def test_ballarin_operator_series_validation_respects_pressure_gauge(tmp_path, monkeypatch):
    monolithic_basis = _blocked_monolithic_basis(tmp_path)
    basis = block_basis_from_monolithic(monolithic_basis)
    reduced = ReducedLinearOperators(
        basis=basis,
        operator=np.eye(basis.combined_basis.shape[1]),
        rhs=np.zeros((basis.combined_basis.shape[1], 1)),
        full_operator_shape=(basis.combined_basis.shape[0], basis.combined_basis.shape[0]),
    )
    basis_dir = tmp_path / "basis"
    summary = save_monolithic_pod_basis(
        output_dir=basis_dir,
        basis=monolithic_basis,
        snapshots_path="synthetic",
        reconstruction_error=0.0,
    )
    summary["pressure_gauge"] = "zero-mean"
    (basis_dir / "monolithic_pod_basis_summary.json").write_text(
        json.dumps(summary),
        encoding="ascii",
    )
    operators_dir = tmp_path / "operators"
    operators_dir.mkdir()
    operator_path = operators_dir / "reduced_operators_ref0001.npz"
    save_reduced_operator_bundle(operator_path, reduced)
    manifest_path = operators_dir / "operator_series_manifest.json"
    manifest_path.write_text(
        json.dumps({"bundles": [{"path": str(operator_path), "reference_index": 1}]}),
        encoding="ascii",
    )
    snapshot = monolithic_basis.pod_basis.mean.copy()
    pressure_dofs = next(field.dofs for field in monolithic_basis.fields if field.name == "pressure")
    snapshot[pressure_dofs, :] += 42.0
    np.savez_compressed(
        tmp_path / "snapshots.npz",
        dvp=np.hstack([snapshot, snapshot]),
    )
    output_dir = tmp_path / "online_series"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ballarin_operator_series_solve",
            "--basis",
            str(basis_dir / "monolithic_pod_basis.npz"),
            "--manifest",
            str(manifest_path),
            "--snapshots",
            str(tmp_path / "snapshots.npz"),
            "--output-dir",
            str(output_dir),
        ],
    )

    run_ballarin_operator_series_solve.main()

    summary = json.loads(
        (output_dir / "ballarin_operator_series_solution_summary.json").read_text(
            encoding="ascii"
        )
    )
    assert summary["pressure_gauge"] == "zero-mean"
    assert summary["solutions"][0]["field_relative_errors"]["pressure"] == 0.0


def test_public_pod_galerkin_runs_ballarin_operator_solve(tmp_path, monkeypatch):
    monolithic_basis = _blocked_monolithic_basis(tmp_path)
    basis = block_basis_from_monolithic(monolithic_basis)
    phi = basis.combined_basis
    operator = np.diag(np.arange(2, phi.shape[0] + 2, dtype=float))
    rhs = np.arange(1, phi.shape[0] + 1, dtype=float)
    reduced = project_linear_operator(operator, rhs, basis)
    operators_path = tmp_path / "operators.npz"
    save_reduced_operator_bundle(operators_path, reduced)
    output_dir = tmp_path / "workflow"

    monkeypatch.setattr(
        "sys.argv",
        [
            "pod_galerkin",
            "--snapshots",
            str(tmp_path / "snapshots.npz"),
            "--output-dir",
            str(output_dir),
            "--blocked-dvp",
            "--pressure-treatment",
            "supremizer",
            "--displacement-modes",
            "1",
            "--velocity-modes",
            "1",
            "--supremizer-modes",
            "1",
            "--pressure-modes",
            "1",
            "--scheme",
            "ballarin-monolithic",
            "--action",
            "operator-solve",
            "--operator-bundle",
            str(operators_path),
        ],
    )

    pod_galerkin.main()

    assert (output_dir / "basis" / "monolithic_pod_basis.npz").exists()
    assert (
        output_dir
        / "operator_solve"
        / "ballarin_operator_solution.npz"
    ).exists()


def test_public_pod_galerkin_runs_ballarin_operator_build(tmp_path, monkeypatch):
    _blocked_monolithic_basis(tmp_path)
    output_dir = tmp_path / "workflow"
    calls = []

    def fake_run_module(module, arguments):
        calls.append((module, arguments))

    monkeypatch.setattr(pod_galerkin, "run_module", fake_run_module)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pod_galerkin",
            "--snapshots",
            str(tmp_path / "snapshots.npz"),
            "--output-dir",
            str(output_dir),
            "--problem",
            "TF_cfd",
            "--blocked-dvp",
            "--pressure-treatment",
            "supremizer",
            "--displacement-modes",
            "1",
            "--velocity-modes",
            "1",
            "--supremizer-modes",
            "1",
            "--pressure-modes",
            "1",
            "--scheme",
            "ballarin-monolithic",
            "--action",
            "build-operators",
            "--reference-index",
            "2",
            "--dt",
            "0.01",
        ],
    )

    pod_galerkin.main()

    assert calls
    module, arguments = calls[0]
    assert module == "turtleFSI.rom._pod_galerkin.build_ballarin_operator_bundle"
    assert "--output" in arguments
    assert str(output_dir / "operators" / "reduced_operators.npz") in arguments
    assert "--reference-index" in arguments
    assert "2" in arguments


def test_public_pod_galerkin_runs_ballarin_operator_series_build(tmp_path, monkeypatch):
    _blocked_monolithic_basis(tmp_path)
    output_dir = tmp_path / "workflow"
    calls = []

    def fake_run_module(module, arguments):
        calls.append((module, arguments))

    monkeypatch.setattr(pod_galerkin, "run_module", fake_run_module)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pod_galerkin",
            "--snapshots",
            str(tmp_path / "snapshots.npz"),
            "--output-dir",
            str(output_dir),
            "--problem",
            "TF_cfd",
            "--blocked-dvp",
            "--pressure-treatment",
            "supremizer",
            "--pressure-dof-key",
            "pressure_dofs",
            "--pressure-gauge",
            "zero-mean",
            "--displacement-modes",
            "1",
            "--velocity-modes",
            "1",
            "--supremizer-modes",
            "1",
            "--pressure-modes",
            "1",
            "--scheme",
            "ballarin-monolithic",
            "--action",
            "build-operator-series",
            "--start-index",
            "1",
            "--stop-index",
            "3",
            "--stride",
            "1",
            "--dt",
            "0.01",
        ],
    )

    pod_galerkin.main()

    assert calls
    module, arguments = calls[0]
    assert module == "turtleFSI.rom._pod_galerkin.build_ballarin_operator_series"
    assert "--output-dir" in arguments
    assert str(output_dir / "operator_series") in arguments
    with open(output_dir / "basis" / "monolithic_pod_basis_summary.json", encoding="ascii") as handle:
        summary = json.load(handle)
    assert summary["pressure_gauge"] == "zero-mean"


def test_public_pod_galerkin_operator_series_solve_uses_manifest_basis(tmp_path, monkeypatch):
    basis = _blocked_monolithic_basis(tmp_path)
    manifest_basis_dir = tmp_path / "manifest_basis"
    save_monolithic_pod_basis(
        output_dir=manifest_basis_dir,
        basis=basis,
        snapshots_path=str(tmp_path / "snapshots.npz"),
        reconstruction_error=0.0,
    )
    manifest_path = tmp_path / "operator_series_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "basis": str(manifest_basis_dir / "monolithic_pod_basis.npz"),
                "bundles": [],
            }
        ),
        encoding="ascii",
    )
    output_dir = tmp_path / "workflow"
    calls = []

    def fake_run_module(module, arguments):
        calls.append((module, arguments))

    monkeypatch.setattr(pod_galerkin, "run_module", fake_run_module)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pod_galerkin",
            "--snapshots",
            str(tmp_path / "snapshots.npz"),
            "--output-dir",
            str(output_dir),
            "--blocked-dvp",
            "--pressure-treatment",
            "supremizer",
            "--scheme",
            "ballarin-monolithic",
            "--action",
            "operator-series-solve",
            "--operator-series-manifest",
            str(manifest_path),
            "--linear-solver",
            "lstsq",
        ],
    )

    pod_galerkin.main()

    assert calls
    module, arguments = calls[0]
    assert module == "turtleFSI.rom._pod_galerkin.run_ballarin_operator_series_solve"
    assert "--basis" in arguments
    assert str(manifest_basis_dir / "monolithic_pod_basis.npz") in arguments
    assert not (output_dir / "basis").exists()


def test_public_pod_galerkin_rollout_passes_pressure_and_linear_solver(tmp_path, monkeypatch):
    _blocked_monolithic_basis(tmp_path)
    output_dir = tmp_path / "workflow"
    calls = []

    def fake_run_module(module, arguments):
        calls.append((module, arguments))

    monkeypatch.setattr(pod_galerkin, "run_module", fake_run_module)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pod_galerkin",
            "--snapshots",
            str(tmp_path / "snapshots.npz"),
            "--output-dir",
            str(output_dir),
            "--problem",
            "TF_cfd",
            "--blocked-dvp",
            "--pressure-treatment",
            "supremizer",
            "--pressure-dof-key",
            "pressure_dofs",
            "--pressure-gauge",
            "none",
            "--displacement-modes",
            "1",
            "--velocity-modes",
            "1",
            "--supremizer-modes",
            "1",
            "--pressure-modes",
            "1",
            "--action",
            "rollout",
            "--num-steps",
            "1",
            "--linear-solver",
            "equilibrated-lstsq",
            "--linear-rcond",
            "1e-12",
        ],
    )

    pod_galerkin.main()

    assert calls
    module, arguments = calls[0]
    assert module == "turtleFSI.rom._pod_galerkin.run_fenics_pod_galerkin_rollout"
    assert "--pressure-gauge" in arguments
    assert "none" in arguments
    assert "--linear-solver" in arguments
    assert "equilibrated-lstsq" in arguments
    assert "--linear-rcond" in arguments
    assert "1e-12" in arguments


def test_offline_quadratic_projection_matches_full_tensor(tmp_path):
    basis = _blocked_basis(tmp_path)
    phi = basis.combined_basis
    tensor = np.zeros((phi.shape[0], phi.shape[0], phi.shape[0]), dtype=float)
    for row in range(phi.shape[0]):
        tensor[row, row, row] = row + 1.0
    reduced_tensor = project_quadratic_tensor(tensor, basis)
    coefficients = np.arange(1, phi.shape[1] + 1, dtype=float)

    reduced_value = evaluate_reduced_quadratic(reduced_tensor, coefficients)
    full_increment = phi @ coefficients[:, None]
    full_value = np.einsum(
        "ijk,j,k->i",
        tensor,
        full_increment[:, 0],
        full_increment[:, 0],
    ).reshape(-1, 1)

    assert np.allclose(reduced_value, phi.T @ full_value)
