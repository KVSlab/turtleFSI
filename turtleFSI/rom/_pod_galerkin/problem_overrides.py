"""Problem override helpers for replaying FOM snapshot cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_override_value(value: str):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_problem_overrides(entries: list[str]) -> dict[str, object]:
    overrides = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Problem override must have KEY=VALUE form, got {entry!r}")
        key, value = entry.split("=", 1)
        if not key:
            raise ValueError(f"Problem override key is empty in {entry!r}")
        overrides[key] = parse_override_value(value)
    return overrides


def checkpoint_overrides_from_snapshots(snapshots_path: Path) -> dict[str, Any]:
    """Load FOM problem settings saved next to a DVP snapshot bundle.

    VaSP-backed local problem wrappers often have unusable package defaults for
    paths such as ``mesh_path``. The snapshot summary points back to the FOM run
    directory, whose checkpoint contains the exact settings used to generate the
    snapshots.
    """

    summary_path = snapshots_path.with_name("dvp_snapshots_summary.json")
    if not summary_path.exists():
        return {}
    with open(summary_path, encoding="ascii") as handle:
        summary = json.load(handle)
    results_folder = summary.get("results_folder")
    if not results_folder:
        return {}
    checkpoint_path = Path(results_folder) / "Checkpoint" / "default_variables.json"
    if not checkpoint_path.exists():
        return {}
    with open(checkpoint_path, encoding="ascii") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Checkpoint variables must be a JSON object: {checkpoint_path}")
    return values


def replay_overrides(
    snapshots_path: Path,
    explicit_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return checkpoint-derived overrides updated by explicit values."""

    overrides = checkpoint_overrides_from_snapshots(snapshots_path)
    overrides["verbose"] = False
    if explicit_overrides:
        overrides.update(explicit_overrides)
    return overrides
