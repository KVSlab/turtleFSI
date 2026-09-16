"""Time integration helpers for structured non-intrusive ROM blocks."""

from __future__ import annotations

import numpy as np

from turtleFSI.rom._pod.infer import LinearBlockModel, predict_block


def rollout_structured_blocks(
    models: dict[str, LinearBlockModel],
    initial_state: dict[str, np.ndarray],
    num_steps: int,
) -> dict[str, np.ndarray]:
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    state = {
        name: np.asarray(value, dtype=float).reshape(-1, 1)
        for name, value in initial_state.items()
    }
    history = {name: [value[:, 0].copy()] for name, value in state.items()}

    for _ in range(num_steps):
        next_state = dict(state)
        if "U" in models:
            next_state["U"] = predict_block(models["U"], state)
        if "Ds" in models:
            next_state["Ds"] = predict_block(models["Ds"], state)
        if "Vs" in models:
            next_state["Vs"] = predict_block(models["Vs"], state)
        if "Df" in models:
            next_state["Df"] = predict_block(models["Df"], next_state)

        state = next_state
        for name, value in state.items():
            history.setdefault(name, []).append(value[:, 0].copy())

    return {
        name: np.asarray(values, dtype=float).T
        for name, values in history.items()
    }
