import numpy as np

from turtleFSI.rom._pod_galerkin.reduced_solvers import (
    backward_euler,
    reduced_newton_solve_projected,
    train_linear_pod_galerkin,
)


def test_linear_pod_galerkin_matches_full_backward_euler_in_snapshot_subspace():
    mass = np.eye(2)
    stiffness = np.array([[2.0, 0.0], [0.0, 5.0]])
    force = np.array([0.0, 0.0])
    dt = 0.1
    steps = 8

    lhs = mass / dt + stiffness
    y = np.array([[1.0], [-0.5]])
    snapshots = [y[:, 0].copy()]
    for _ in range(steps):
        y = np.linalg.solve(lhs, mass @ y / dt)
        snapshots.append(y[:, 0].copy())
    snapshot_matrix = np.asarray(snapshots, dtype=float).T

    model = train_linear_pod_galerkin(
        snapshots=snapshot_matrix,
        mass=mass,
        stiffness=stiffness,
        force=force,
        energy_threshold=1.0,
        max_modes=2,
    )
    _, rom_full = backward_euler(
        model=model,
        initial_full_state=snapshot_matrix[:, 0],
        dt=dt,
        num_steps=steps,
    )

    assert model.mass_r.shape == (2, 2)
    assert np.allclose(rom_full, snapshot_matrix, atol=1e-12)


def test_projected_newton_line_search_rejects_residual_increase():
    def residual(coefficients):
        return np.asarray(coefficients, dtype=float).reshape(1, 1)

    def wrong_sign_jacobian(coefficients):
        return np.array([[-1.0]])

    def reconstruct(coefficients):
        return np.asarray(coefficients, dtype=float).reshape(1, 1)

    result = reduced_newton_solve_projected(
        initial_coefficients=np.array([[1.0]]),
        projected_residual_callback=residual,
        projected_jacobian_callback=wrong_sign_jacobian,
        reconstruct_callback=reconstruct,
        max_iterations=3,
        damping=1.0,
        min_damping=0.25,
    )

    assert not result.converged
    assert result.residual_norms == [1.0]
    assert np.allclose(result.full_state, np.array([[1.0]]))


def test_projected_newton_reuses_jacobian_by_frequency():
    jacobian_calls = 0

    def residual(coefficients):
        value = float(np.asarray(coefficients).reshape(-1)[0])
        return np.array([[value - 1.0]])

    def jacobian(coefficients):
        nonlocal jacobian_calls
        jacobian_calls += 1
        return np.array([[1.0]])

    def reconstruct(coefficients):
        return np.asarray(coefficients, dtype=float).reshape(1, 1)

    result = reduced_newton_solve_projected(
        initial_coefficients=np.array([[0.0]]),
        projected_residual_callback=residual,
        projected_jacobian_callback=jacobian,
        reconstruct_callback=reconstruct,
        max_iterations=5,
        jacobian_update_frequency=10,
        line_search=False,
    )

    assert result.converged
    assert jacobian_calls == 1
    assert result.jacobian_evaluations == 1
    assert result.jacobian_update_iterations == (0,)
    assert np.allclose(result.coefficients, np.array([[1.0]]))
