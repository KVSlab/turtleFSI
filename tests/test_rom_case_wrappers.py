import json
from pathlib import Path
import sys

import numpy as np
import pytest

from turtleFSI.rom import pod, pod_galerkin


def test_case_rom_infers_snapshot_fields(tmp_path):
    snapshots = tmp_path / "snapshots.npz"
    np.savez_compressed(
        snapshots,
        time=np.arange(4),
        counter=np.arange(4),
        displacement_dofs=np.array([0, 1]),
        U=np.ones((3, 4)),
        Ds=np.arange(8, dtype=float).reshape(2, 4),
        not_snapshots=np.ones((2, 1)),
    )

    assert pod.infer_snapshot_fields(snapshots) == ["U", "Ds"]


def test_case_pod_galerkin_builds_basis_from_run_dir(tmp_path, monkeypatch):
    run_dir = tmp_path / "case" / "simulations" / "tiny_training" / "1"
    rom_dir = run_dir / "ROM"
    rom_dir.mkdir(parents=True)
    snapshots = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 8.0, 16.0],
        ],
        dtype=float,
    )
    np.savez_compressed(
        rom_dir / "dvp_snapshots.npz",
        dvp=snapshots,
        time=np.arange(snapshots.shape[1], dtype=float),
    )
    with open(rom_dir / "dvp_snapshots_summary.json", "w", encoding="ascii") as handle:
        json.dump({"problem": "TF_fsi"}, handle)

    output_dir = tmp_path / "pod_galerkin"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pod_galerkin",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--energy-threshold",
            "1.0",
        ],
    )
    pod_galerkin.main()

    assert (output_dir / "basis" / "monolithic_pod_basis.npz").exists()
    assert (output_dir / "case_pod_galerkin_summary.json").exists()


def test_exact_ballarin_mode_is_explicitly_not_implemented(tmp_path, monkeypatch):
    snapshots = tmp_path / "dvp_snapshots.npz"
    np.savez_compressed(snapshots, dvp=np.ones((3, 3)), time=np.arange(3))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pod_galerkin",
            "--snapshots",
            str(snapshots),
            "--scheme",
            "ballarin-monolithic",
        ],
    )

    with pytest.raises(NotImplementedError, match="Ballarin/Rozza 2016"):
        pod_galerkin.main()


def test_supremizer_mode_requires_blocked_dvp(tmp_path, monkeypatch):
    snapshots = tmp_path / "dvp_snapshots.npz"
    np.savez_compressed(snapshots, dvp=np.ones((3, 3)), time=np.arange(3))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pod_galerkin",
            "--snapshots",
            str(snapshots),
            "--pressure-treatment",
            "supremizer",
        ],
    )

    with pytest.raises(ValueError, match="requires --blocked-dvp"):
        pod_galerkin.main()


def test_case_pod_galerkin_validate_rollout_builds_metrics(tmp_path, monkeypatch):
    snapshots_path = tmp_path / "dvp_snapshots.npz"
    snapshots = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
        ],
        dtype=float,
    )
    np.savez_compressed(
        snapshots_path,
        dvp=snapshots,
        time=np.arange(snapshots.shape[1], dtype=float),
        displacement_dofs=np.array([0, 1], dtype=np.int64),
        velocity_dofs=np.array([2, 3], dtype=np.int64),
        pressure_dofs=np.array([4, 5], dtype=np.int64),
    )
    output_dir = tmp_path / "validation"
    calls = []

    def fake_run_module(module, arguments):
        calls.append((module, arguments))
        if module != "turtleFSI.rom._pod_galerkin.run_fenics_pod_galerkin_rollout":
            return
        action_dir = Path(arguments[arguments.index("--output-dir") + 1])
        action_dir.mkdir(parents=True)
        start = int(arguments[arguments.index("--start-index") + 1])
        num_steps = int(arguments[arguments.index("--num-steps") + 1])
        full = snapshots[:, start : start + num_steps + 1]
        np.savez_compressed(
            action_dir / "fenics_pod_galerkin_rollout.npz",
            full=full,
            reference=full,
            time=np.arange(full.shape[1], dtype=float),
        )
        (action_dir / "fenics_pod_galerkin_rollout_summary.json").write_text(
            json.dumps(
                {
                    "start_index": start,
                    "num_steps": num_steps,
                    "reference_relative_error": 0.0,
                }
            ),
            encoding="ascii",
        )

    monkeypatch.setattr(pod_galerkin, "run_module", fake_run_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pod_galerkin",
            "--snapshots",
            str(snapshots_path),
            "--output-dir",
            str(output_dir),
            "--problem",
            "TF_cfd",
            "--action",
            "validate-rollout",
            "--train-fraction",
            "0.61",
            "--energy-threshold",
            "1.0",
            "--skip-validation-plots",
        ],
    )

    pod_galerkin.main()

    module, arguments = calls[0]
    assert module == "turtleFSI.rom._pod_galerkin.run_fenics_pod_galerkin_rollout"
    assert arguments[arguments.index("--start-index") + 1] == "3"
    assert arguments[arguments.index("--num-steps") + 1] == "1"
    metrics = json.loads(
        (output_dir / "validation_rollout" / "field_error_metrics.json").read_text(
            encoding="ascii"
        )
    )
    assert metrics["train_indices"] == [0, 3]
    assert metrics["test_indices"] == [4, 4]
    assert metrics["field_relative_errors"]["velocity"] == 0.0


def test_case_pod_galerkin_validate_rollout_runs_slice_plotter(tmp_path, monkeypatch):
    snapshots_path = tmp_path / "dvp_snapshots.npz"
    snapshots = np.arange(30, dtype=float).reshape(6, 5)
    np.savez_compressed(
        snapshots_path,
        dvp=snapshots,
        time=np.arange(snapshots.shape[1], dtype=float),
        displacement_dofs=np.array([0, 1], dtype=np.int64),
        velocity_dofs=np.array([2, 3], dtype=np.int64),
        pressure_dofs=np.array([4, 5], dtype=np.int64),
    )
    output_dir = tmp_path / "validation"
    calls = []

    def fake_run_module(module, arguments):
        calls.append((module, arguments))
        if module == "turtleFSI.rom._pod_galerkin.run_fenics_pod_galerkin_rollout":
            action_dir = Path(arguments[arguments.index("--output-dir") + 1])
            action_dir.mkdir(parents=True)
            start = int(arguments[arguments.index("--start-index") + 1])
            num_steps = int(arguments[arguments.index("--num-steps") + 1])
            full = snapshots[:, start : start + num_steps + 1]
            np.savez_compressed(action_dir / "fenics_pod_galerkin_rollout.npz", full=full)
            (action_dir / "fenics_pod_galerkin_rollout_summary.json").write_text(
                json.dumps({"start_index": start, "num_steps": num_steps}),
                encoding="ascii",
            )

    monkeypatch.setattr(pod_galerkin, "run_module", fake_run_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pod_galerkin",
            "--snapshots",
            str(snapshots_path),
            "--output-dir",
            str(output_dir),
            "--problem",
            "TF_cfd",
            "--action",
            "validate-rollout",
            "--train-fraction",
            "0.6",
            "--energy-threshold",
            "1.0",
            "--slice-axis",
            "y",
            "--slice-value",
            "0.0",
            "--slice-tolerance",
            "0.1",
        ],
    )

    pod_galerkin.main()

    plot_calls = [
        arguments
        for module, arguments in calls
        if module == "turtleFSI.cases.plots.plot_fenics_pod_galerkin_rollout"
    ]
    assert plot_calls
    assert "--slice-animation" in plot_calls[0]
    assert "--surface-animation" not in plot_calls[0]
    assert any("validation_slice_animations" in argument for argument in plot_calls[0])


def test_case_pod_galerkin_exact_modes_forces_energy_threshold(tmp_path, monkeypatch):
    snapshots_path = tmp_path / "dvp_snapshots.npz"
    snapshots = np.array(
        [
            [0.0, 1.0, 0.0, 2.0],
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 2.0, 1.0, 0.0],
            [2.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    np.savez_compressed(
        snapshots_path,
        dvp=snapshots,
        displacement_dofs=np.array([0, 1], dtype=np.int64),
        velocity_dofs=np.array([2, 3], dtype=np.int64),
        pressure_dofs=np.array([4, 5], dtype=np.int64),
    )
    output_dir = tmp_path / "exact_modes"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pod_galerkin",
            "--snapshots",
            str(snapshots_path),
            "--output-dir",
            str(output_dir),
            "--blocked-dvp",
            "--exact-modes",
            "--displacement-modes",
            "2",
            "--velocity-modes",
            "2",
            "--pressure-modes",
            "2",
        ],
    )

    pod_galerkin.main()

    summary = json.loads(
        (output_dir / "basis" / "monolithic_pod_basis_summary.json").read_text(
            encoding="ascii"
        )
    )
    assert summary["fields"]["displacement"]["num_modes"] == 2
    assert summary["fields"]["velocity"]["num_modes"] == 2
    assert summary["fields"]["pressure"]["num_modes"] == 2
