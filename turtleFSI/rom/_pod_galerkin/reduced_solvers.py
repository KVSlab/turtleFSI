"""Reduced POD-Galerkin solver utilities.

This module implements the algebraic core needed to test POD-Galerkin on
linearized or already assembled systems. For nonlinear monolithic FSI, the same
projection pattern applies to residual/Jacobian callbacks:

``R_r(a) = Phi.T @ R(y_mean + Phi @ a)``
``J_r(a) = Phi.T @ J(y_mean + Phi @ a) @ Phi``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from turtleFSI.rom._pod_galerkin.offline_online import solve_equilibrated_lstsq
from turtleFSI.rom._shared.pod import PODBasis, compute_pod_basis, reconstruct_from_basis


@dataclass(frozen=True)
class LinearGalerkinModel:
    basis: PODBasis
    mass_r: np.ndarray
    stiffness_r: np.ndarray
    force_r: np.ndarray
    mass_full: np.ndarray
    stiffness_full: np.ndarray
    force_full: np.ndarray


@dataclass(frozen=True)
class ReducedNewtonResult:
    coefficients: np.ndarray
    full_state: np.ndarray
    converged: bool
    iterations: int
    residual_norms: list[float]
    correction_norms: list[float]
    jacobian_evaluations: int = 0
    jacobian_update_iterations: tuple[int, ...] = ()
    jacobian_condition_numbers: tuple[float, ...] = ()
    termination_reason: str = ""


@dataclass(frozen=True)
class NonlinearGalerkinProblem:
    """Callbacks for a residual-projected monolithic POD-Galerkin problem.

    ``residual`` and ``jacobian`` are evaluated in the full-order state space.
    The reduced solver reconstructs ``y = mean + Phi @ a`` and solves:

    ``Phi.T @ residual(y) = 0``
    ``Phi.T @ jacobian(y) @ Phi delta_a = -Phi.T @ residual(y)``

    This is the intrusive POD-Galerkin interface needed by a future FEniCS
    adapter. In a FEniCS implementation, these callbacks assemble the same
    residual and Jacobian forms used by the full monolithic solver.
    """

    basis: PODBasis
    residual: Callable[[np.ndarray], np.ndarray]
    jacobian: Callable[[np.ndarray], np.ndarray]


def project_linear_system(
    mass: np.ndarray,
    stiffness: np.ndarray,
    force: np.ndarray,
    basis: PODBasis,
) -> LinearGalerkinModel:
    m = np.asarray(mass, dtype=float)
    k = np.asarray(stiffness, dtype=float)
    f = np.asarray(force, dtype=float).reshape(-1, 1)
    phi = basis.basis

    if m.shape != k.shape or m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("mass and stiffness must be square matrices with matching shape")
    if phi.shape[0] != m.shape[0] or f.shape[0] != m.shape[0]:
        raise ValueError("basis and force dimensions must match the full system")

    return LinearGalerkinModel(
        basis=basis,
        mass_r=phi.T @ m @ phi,
        stiffness_r=phi.T @ k @ phi,
        force_r=phi.T @ (f - k @ basis.mean),
        mass_full=m,
        stiffness_full=k,
        force_full=f,
    )


def backward_euler(
    model: LinearGalerkinModel,
    initial_full_state: np.ndarray,
    dt: float,
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    y0 = np.asarray(initial_full_state, dtype=float).reshape(-1, 1)
    if y0.shape[0] != model.basis.basis.shape[0]:
        raise ValueError("initial state dimension does not match basis")

    a = model.basis.basis.T @ (y0 - model.basis.mean)
    lhs = model.mass_r / dt + model.stiffness_r
    reduced = [a[:, 0].copy()]
    full = [reconstruct_from_basis(a, model.basis)[:, 0]]

    for _ in range(num_steps):
        rhs = model.mass_r @ a / dt + model.force_r
        a = np.linalg.solve(lhs, rhs)
        reduced.append(a[:, 0].copy())
        full.append(reconstruct_from_basis(a, model.basis)[:, 0])

    return np.asarray(reduced, dtype=float).T, np.asarray(full, dtype=float).T


def train_linear_pod_galerkin(
    snapshots: np.ndarray,
    mass: np.ndarray,
    stiffness: np.ndarray,
    force: np.ndarray,
    energy_threshold: float = 0.999,
    max_modes: int | None = None,
) -> LinearGalerkinModel:
    basis = compute_pod_basis(
        snapshots,
        energy_threshold=energy_threshold,
        max_modes=max_modes,
        center=True,
    )
    return project_linear_system(mass, stiffness, force, basis)


def projected_residual(
    coefficients: np.ndarray,
    basis: PODBasis,
    residual_callback,
) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=float).reshape(-1, 1)
    state = reconstruct_from_basis(coeff, basis)
    residual = np.asarray(residual_callback(state[:, 0]), dtype=float).reshape(-1, 1)
    return basis.basis.T @ residual


def projected_jacobian(
    coefficients: np.ndarray,
    basis: PODBasis,
    jacobian_callback,
) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=float).reshape(-1, 1)
    state = reconstruct_from_basis(coeff, basis)
    jacobian = np.asarray(jacobian_callback(state[:, 0]), dtype=float)
    return basis.basis.T @ jacobian @ basis.basis


def reduced_newton_solve(
    problem: NonlinearGalerkinProblem,
    initial_coefficients: np.ndarray,
    atol: float = 1e-10,
    rtol: float = 1e-10,
    max_iterations: int = 25,
    damping: float = 1.0,
) -> ReducedNewtonResult:
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}")
    if not 0.0 < damping <= 1.0:
        raise ValueError(f"damping must be in (0, 1], got {damping}")

    coeff = np.asarray(initial_coefficients, dtype=float).reshape(-1, 1)
    if coeff.shape[0] != problem.basis.num_modes:
        raise ValueError(
            f"initial coefficient count {coeff.shape[0]} does not match basis "
            f"mode count {problem.basis.num_modes}"
        )

    residual_norms: list[float] = []
    correction_norms: list[float] = []
    initial_residual_norm = None
    converged = False
    termination_reason = "max_iterations"

    for iteration in range(max_iterations):
        full_state = reconstruct_from_basis(coeff, problem.basis)
        residual_full = np.asarray(problem.residual(full_state[:, 0]), dtype=float).reshape(-1, 1)
        residual_reduced = problem.basis.basis.T @ residual_full
        residual_norm = float(np.linalg.norm(residual_reduced))
        residual_norms.append(residual_norm)
        if initial_residual_norm is None:
            initial_residual_norm = max(residual_norm, np.finfo(float).eps)

        if residual_norm <= atol or residual_norm / initial_residual_norm <= rtol:
            converged = True
            termination_reason = "converged"
            break

        jacobian_full = np.asarray(problem.jacobian(full_state[:, 0]), dtype=float)
        if jacobian_full.ndim != 2:
            raise ValueError(f"jacobian callback must return a 2D matrix, got {jacobian_full.shape}")
        jacobian_reduced = problem.basis.basis.T @ jacobian_full @ problem.basis.basis
        correction = np.linalg.solve(jacobian_reduced, -residual_reduced)
        correction_norms.append(float(np.linalg.norm(correction)))
        coeff = coeff + damping * correction

    full_state = reconstruct_from_basis(coeff, problem.basis)
    return ReducedNewtonResult(
        coefficients=coeff,
        full_state=full_state,
        converged=converged,
        iterations=len(residual_norms) - 1 if converged else len(residual_norms),
        residual_norms=residual_norms,
        correction_norms=correction_norms,
        termination_reason=termination_reason,
    )


def reduced_newton_solve_projected(
    initial_coefficients: np.ndarray,
    projected_residual_callback,
    projected_jacobian_callback,
    reconstruct_callback,
    atol: float = 1e-10,
    rtol: float = 1e-10,
    max_iterations: int = 25,
    damping: float = 1.0,
    line_search: bool = True,
    min_damping: float = 1e-6,
    jacobian_update_frequency: int = 1,
    reuse_jacobian_after_line_search_failure: bool = False,
    linear_solver: str = "solve",
    linear_rcond: float | None = None,
) -> ReducedNewtonResult:
    """Solve a reduced Newton problem from already projected callbacks.

    This variant is intended for FEniCS adapters that can compute ``Phi.T R``
    and ``Phi.T J Phi`` directly, avoiding construction of a dense full
    Jacobian in Python.
    """

    if max_iterations < 1:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}")
    if not 0.0 < damping <= 1.0:
        raise ValueError(f"damping must be in (0, 1], got {damping}")
    if min_damping <= 0.0 or min_damping > damping:
        raise ValueError(
            f"min_damping must be positive and <= damping; got min_damping={min_damping}, damping={damping}"
        )
    if jacobian_update_frequency < 1:
        raise ValueError(
            f"jacobian_update_frequency must be positive, got {jacobian_update_frequency}"
        )
    if linear_solver not in {"solve", "lstsq", "equilibrated-lstsq"}:
        raise ValueError(
            "linear_solver must be 'solve', 'lstsq', or "
            f"'equilibrated-lstsq', got {linear_solver!r}"
        )
    if linear_rcond is not None and linear_rcond < 0.0:
        raise ValueError(f"linear_rcond must be non-negative, got {linear_rcond}")

    coeff = np.asarray(initial_coefficients, dtype=float).reshape(-1, 1)
    residual_norms: list[float] = []
    correction_norms: list[float] = []
    initial_residual_norm = None
    converged = False
    termination_reason = "max_iterations"
    jacobian_reduced = None
    jacobian_evaluations = 0
    jacobian_update_iterations: list[int] = []
    jacobian_condition_numbers: list[float] = []

    for iteration in range(max_iterations):
        residual_reduced = np.asarray(projected_residual_callback(coeff), dtype=float).reshape(-1, 1)
        residual_norm = float(np.linalg.norm(residual_reduced))
        residual_norms.append(residual_norm)
        if initial_residual_norm is None:
            initial_residual_norm = max(residual_norm, np.finfo(float).eps)

        if residual_norm <= atol or residual_norm / initial_residual_norm <= rtol:
            converged = True
            termination_reason = "converged"
            break

        should_update_jacobian = (
            jacobian_reduced is None
            or iteration % jacobian_update_frequency == 0
        )
        if should_update_jacobian:
            jacobian_reduced = np.asarray(projected_jacobian_callback(coeff), dtype=float)
            jacobian_evaluations += 1
            jacobian_update_iterations.append(iteration)
            singular_values = np.linalg.svd(jacobian_reduced, compute_uv=False)
            smallest = singular_values[-1] if singular_values.size else 0.0
            if smallest > 0.0:
                jacobian_condition_numbers.append(float(singular_values[0] / smallest))
            else:
                jacobian_condition_numbers.append(float("inf"))
        if linear_solver == "equilibrated-lstsq":
            correction = solve_equilibrated_lstsq(
                jacobian_reduced,
                -residual_reduced,
                rcond=linear_rcond,
            ).coefficients
        elif linear_solver == "lstsq":
            correction = np.linalg.lstsq(
                jacobian_reduced,
                -residual_reduced,
                rcond=linear_rcond,
            )[0]
        else:
            correction = np.linalg.solve(jacobian_reduced, -residual_reduced)
        correction_norms.append(float(np.linalg.norm(correction)))
        if not line_search:
            coeff = coeff + damping * correction
            continue

        step = damping
        accepted = False
        while step >= min_damping:
            trial_coeff = coeff + step * correction
            trial_residual = np.asarray(
                projected_residual_callback(trial_coeff), dtype=float
            ).reshape(-1, 1)
            trial_norm = float(np.linalg.norm(trial_residual))
            if trial_norm < residual_norm:
                coeff = trial_coeff
                accepted = True
                break
            step *= 0.5
        if not accepted:
            if not reuse_jacobian_after_line_search_failure:
                termination_reason = "line_search_failed"
                break
            jacobian_reduced = None

    full_state = np.asarray(reconstruct_callback(coeff), dtype=float)
    if full_state.ndim == 1:
        full_state = full_state[:, None]
    return ReducedNewtonResult(
        coefficients=coeff,
        full_state=full_state,
        converged=converged,
        iterations=len(residual_norms) - 1 if converged else len(residual_norms),
        residual_norms=residual_norms,
        correction_norms=correction_norms,
        jacobian_evaluations=jacobian_evaluations,
        jacobian_update_iterations=tuple(jacobian_update_iterations),
        jacobian_condition_numbers=tuple(jacobian_condition_numbers),
        termination_reason=termination_reason,
    )


def implicit_euler_residual_factory(
    mass: np.ndarray,
    operator,
    previous_state: np.ndarray,
    dt: float,
    forcing: np.ndarray | None = None,
):
    """Build a full-space implicit Euler residual callback.

    The residual is:

    ``M (y - y_prev) / dt + operator(y) - forcing = 0``

    ``operator`` may be either a matrix or a callable returning a vector.
    """

    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    m = np.asarray(mass, dtype=float)
    y_prev = np.asarray(previous_state, dtype=float).reshape(-1, 1)
    f = (
        np.zeros_like(y_prev)
        if forcing is None
        else np.asarray(forcing, dtype=float).reshape(-1, 1)
    )
    if callable(operator):
        return lambda y: (
            m @ (np.asarray(y, dtype=float).reshape(-1, 1) - y_prev) / dt
            + np.asarray(operator(y), dtype=float).reshape(-1, 1)
            - f
        )[:, 0]

    k = np.asarray(operator, dtype=float)
    return lambda y: (
        m @ (np.asarray(y, dtype=float).reshape(-1, 1) - y_prev) / dt
        + k @ np.asarray(y, dtype=float).reshape(-1, 1)
        - f
    )[:, 0]


def implicit_euler_jacobian_factory(
    mass: np.ndarray,
    operator_jacobian,
    dt: float,
):
    """Build a full-space implicit Euler Jacobian callback."""

    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    m = np.asarray(mass, dtype=float)
    if callable(operator_jacobian):
        return lambda y: m / dt + np.asarray(operator_jacobian(y), dtype=float)
    k = np.asarray(operator_jacobian, dtype=float)
    return lambda y: m / dt + k


def implicit_euler_nonlinear_rollout(
    basis: PODBasis,
    mass: np.ndarray,
    operator,
    operator_jacobian,
    initial_full_state: np.ndarray,
    dt: float,
    num_steps: int,
    forcing=None,
    atol: float = 1e-10,
    rtol: float = 1e-10,
    max_newton_iterations: int = 25,
) -> tuple[np.ndarray, np.ndarray, list[ReducedNewtonResult]]:
    """Roll out a residual-projected monolithic POD-Galerkin model.

    This is a generic nonlinear test driver. For full FSI, replace the residual
    and Jacobian factories with callbacks that assemble turtleFSI's FEniCS forms.
    """

    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    current_full = np.asarray(initial_full_state, dtype=float).reshape(-1, 1)
    coeff = basis.basis.T @ (current_full - basis.mean)
    reduced_history = [coeff[:, 0].copy()]
    full_history = [current_full[:, 0].copy()]
    results: list[ReducedNewtonResult] = []

    for step in range(num_steps):
        step_forcing = forcing(step) if callable(forcing) else forcing
        residual = implicit_euler_residual_factory(
            mass=mass,
            operator=operator,
            previous_state=current_full[:, 0],
            dt=dt,
            forcing=step_forcing,
        )
        jacobian = implicit_euler_jacobian_factory(
            mass=mass,
            operator_jacobian=operator_jacobian,
            dt=dt,
        )
        problem = NonlinearGalerkinProblem(
            basis=basis,
            residual=residual,
            jacobian=jacobian,
        )
        result = reduced_newton_solve(
            problem,
            initial_coefficients=coeff,
            atol=atol,
            rtol=rtol,
            max_iterations=max_newton_iterations,
        )
        results.append(result)
        coeff = result.coefficients
        current_full = result.full_state
        reduced_history.append(coeff[:, 0].copy())
        full_history.append(current_full[:, 0].copy())

    return (
        np.asarray(reduced_history, dtype=float).T,
        np.asarray(full_history, dtype=float).T,
        results,
    )
