import numpy as np

from turtleFSI.rom._pod.infer import fit_default_fsi_blocks
from turtleFSI.rom._pod.simulate import rollout_structured_blocks


def test_structured_block_rom_rolls_out_linear_fsi_surrogate():
    steps = 10
    U = np.zeros((2, steps))
    Ds = np.zeros((2, steps))
    Vs = np.zeros((2, steps))
    Df = np.zeros((2, steps))

    U[:, 0] = [1.0, -0.2]
    Ds[:, 0] = [0.1, -0.1]
    Vs[:, 0] = [0.0, 0.2]
    mesh_map = np.array([[0.4, 0.1], [-0.2, 0.3]])
    for i in range(steps - 1):
        Df[:, i] = mesh_map @ Ds[:, i]
        U[:, i + 1] = np.array([[0.8, 0.1], [-0.1, 0.7]]) @ U[:, i] + 0.2 * Df[:, i]
        Ds[:, i + 1] = Ds[:, i] + 0.1 * Vs[:, i]
        Vs[:, i + 1] = (
            np.array([[0.9, 0.0], [0.0, 0.85]]) @ Vs[:, i]
            - 0.3 * Ds[:, i]
            + 0.05 * U[:, i]
        )
    Df[:, -1] = mesh_map @ Ds[:, -1]

    snapshots = {"U": U, "Ds": Ds, "Vs": Vs, "Df": Df}
    models = fit_default_fsi_blocks(snapshots, ridge=0.0)
    rollout = rollout_structured_blocks(
        models=models,
        initial_state={name: values[:, 0] for name, values in snapshots.items()},
        num_steps=steps - 1,
    )

    assert set(models) == {"U", "Ds", "Vs", "Df"}
    assert models["U"].train_relative_error < 1e-8
    assert models["Df"].train_relative_error < 1e-10
    assert np.linalg.norm(rollout["U"] - U) / np.linalg.norm(U) < 1e-6
    assert np.linalg.norm(rollout["Ds"] - Ds) / np.linalg.norm(Ds) < 1e-5
    assert np.linalg.norm(rollout["Df"] - Df) / np.linalg.norm(Df) < 1e-5
