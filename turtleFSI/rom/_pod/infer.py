"""Structured non-intrusive operator inference for ROM preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LinearBlockModel:
    """One fitted transition block ``Y ~= A X + b``."""

    input_fields: tuple[str, ...]
    output_field: str
    operator: np.ndarray
    bias: np.ndarray
    train_relative_error: float


def _as_snapshot_matrix(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {array.shape}")
    if array.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two snapshots")
    return array


def _relative_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    residual = np.linalg.norm(reference - prediction)
    baseline = np.linalg.norm(reference)
    if np.isclose(baseline, 0.0):
        return float(residual)
    return float(residual / baseline)


def fit_ridge_map(x: np.ndarray, y: np.ndarray, ridge: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.asarray(x, dtype=float)
    outputs = np.asarray(y, dtype=float)
    if inputs.ndim != 2 or outputs.ndim != 2:
        raise ValueError("x and y must be 2D matrices")
    if inputs.shape[1] != outputs.shape[1]:
        raise ValueError(
            f"x and y must have the same sample count, got {inputs.shape[1]} and "
            f"{outputs.shape[1]}"
        )
    if ridge < 0.0:
        raise ValueError(f"ridge must be non-negative, got {ridge}")
    augmented = np.vstack([inputs, np.ones((1, inputs.shape[1]))])
    left, singular_values, right_t = np.linalg.svd(augmented, full_matrices=False)
    if np.isclose(ridge, 0.0):
        inverse_weights = np.divide(
            1.0,
            singular_values,
            out=np.zeros_like(singular_values),
            where=singular_values > np.finfo(float).eps,
        )
    else:
        inverse_weights = singular_values / (singular_values**2 + ridge)
    coeff = outputs @ right_t.T @ np.diag(inverse_weights) @ left.T
    return coeff[:, :-1], coeff[:, -1:]


def fit_linear_transition_block(
    snapshots: dict[str, np.ndarray],
    input_fields: Iterable[str],
    output_field: str,
    ridge: float = 1e-10,
) -> LinearBlockModel:
    input_names = tuple(input_fields)
    if not input_names:
        raise ValueError("at least one input field is required")

    matrices = {name: _as_snapshot_matrix(name, values) for name, values in snapshots.items()}
    sample_count = matrices[output_field].shape[1]
    for name in input_names:
        if matrices[name].shape[1] != sample_count:
            raise ValueError(f"{name} snapshot count does not match {output_field}")

    x = np.vstack([matrices[name][:, :-1] for name in input_names])
    y = matrices[output_field][:, 1:]
    operator, bias = fit_ridge_map(x, y, ridge=ridge)
    prediction = operator @ x + bias
    return LinearBlockModel(
        input_fields=input_names,
        output_field=output_field,
        operator=operator,
        bias=bias,
        train_relative_error=_relative_l2(y, prediction),
    )


def fit_algebraic_block(
    snapshots: dict[str, np.ndarray],
    input_fields: Iterable[str],
    output_field: str,
    ridge: float = 1e-10,
    skip_first: bool = False,
) -> LinearBlockModel:
    input_names = tuple(input_fields)
    if not input_names:
        raise ValueError("at least one input field is required")

    matrices = {name: _as_snapshot_matrix(name, values) for name, values in snapshots.items()}
    start = 1 if skip_first else 0
    x = np.vstack([matrices[name][:, start:] for name in input_names])
    y = matrices[output_field][:, start:]
    operator, bias = fit_ridge_map(x, y, ridge=ridge)
    prediction = operator @ x + bias
    return LinearBlockModel(
        input_fields=input_names,
        output_field=output_field,
        operator=operator,
        bias=bias,
        train_relative_error=_relative_l2(y, prediction),
    )


def fit_default_fsi_blocks(
    snapshots: dict[str, np.ndarray],
    ridge: float = 1e-10,
) -> dict[str, LinearBlockModel]:
    """Fit a first structured FSI transition model from snapshot blocks.

    The equations are intentionally simple and explicit:

    ``U[n+1]  ~= f(U[n], Df[n])``
    ``Ds[n+1] ~= f(Ds[n], Vs[n])``
    ``Vs[n+1] ~= f(Vs[n], Ds[n], U[n])``
    ``Df[n+1] ~= f(Ds[n+1])`` when ``Df`` is available.

    This is still non-intrusive, but unlike the baseline it keeps named FSI
    state blocks separated so later adjacency-restricted rows can replace the
    dense maps without changing the simulator interface.
    """

    required = {"U", "Ds", "Vs"}
    missing = sorted(required - snapshots.keys())
    if missing:
        raise KeyError(f"missing required snapshot blocks: {missing}")

    blocks = {
        "U": fit_linear_transition_block(
            snapshots,
            input_fields=("U", "Df") if "Df" in snapshots else ("U",),
            output_field="U",
            ridge=ridge,
        ),
        "Ds": fit_linear_transition_block(
            snapshots,
            input_fields=("Ds", "Vs"),
            output_field="Ds",
            ridge=ridge,
        ),
        "Vs": fit_linear_transition_block(
            snapshots,
            input_fields=("Vs", "Ds", "U"),
            output_field="Vs",
            ridge=ridge,
        ),
    }
    if "Df" in snapshots:
        blocks["Df"] = fit_algebraic_block(
            snapshots,
            input_fields=("Ds",),
            output_field="Df",
            ridge=ridge,
            skip_first=True,
        )
    return blocks


def predict_block(model: LinearBlockModel, state: dict[str, np.ndarray]) -> np.ndarray:
    inputs = []
    for name in model.input_fields:
        if name not in state:
            raise KeyError(f"state is missing input field '{name}'")
        value = np.asarray(state[name], dtype=float)
        inputs.append(value.reshape(-1, 1))
    return model.operator @ np.vstack(inputs) + model.bias
