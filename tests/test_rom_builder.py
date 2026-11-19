import json
from pathlib import Path

import numpy as np

from turtleFSI.rom._pod.build import train_rom


def test_train_rom_builds_offline_model(tmp_path: Path):
    snapshots_path = tmp_path / "snapshots.npz"
    output_dir = tmp_path / "rom"

    steps = 12
    time = np.linspace(0.0, 1.1, steps)

    a = np.array([[0.92, 0.05], [-0.03, 0.88]])
    z = np.zeros((2, steps))
    z[:, 0] = np.array([1.0, -0.4])
    for i in range(steps - 1):
        z[:, i + 1] = a @ z[:, i]

    u_obs = np.array(
        [
            [1.0, 0.0],
            [0.1, 1.0],
            [0.5, -0.2],
            [-0.4, 0.3],
        ]
    ) @ z
    ds_obs = np.array([[0.7, -0.1], [0.2, 0.6], [-0.3, 0.5]]) @ z
    vs_obs = np.array([[0.4, 0.2], [-0.1, 0.8]]) @ z

    np.savez_compressed(
        snapshots_path,
        U=u_obs,
        Ds=ds_obs,
        Vs=vs_obs,
        time=time,
    )

    summary = train_rom(
        snapshots_path=snapshots_path,
        output_dir=output_dir,
        field_names=["U", "Ds", "Vs"],
        energy_threshold=0.999999,
        max_modes=None,
        ridge=1e-12,
        train_fraction=0.75,
        mesh_map_path=None,
    )

    model_path = output_dir / "rom_model.npz"
    summary_path = output_dir / "rom_summary.json"

    assert model_path.is_file()
    assert summary_path.is_file()
    assert summary["latent_dimension"] > 0
    assert summary["validation_one_step_relative_error"] < 1e-8

    model = np.load(model_path)
    assert "A" in model.files
    assert "U_basis" in model.files
    assert model["U_reconstruction"].shape == u_obs.shape

    summary_json = json.loads(summary_path.read_text(encoding="ascii"))
    assert summary_json["fields"]["U"]["num_dofs"] == u_obs.shape[0]
