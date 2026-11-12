"""Reduced models for turtleFSI.

Start with :mod:`turtleFSI.rom.pod` for snapshot/data-fit models or
:mod:`turtleFSI.rom.pod_galerkin` for projected POD-Galerkin models.
"""

from __future__ import annotations

import importlib
import sys


_ALIASES = {
    "common": "turtleFSI.rom._shared",
    "shared": "turtleFSI.rom._shared",
    "data_fit": "turtleFSI.rom._pod",
    "non_intrusive": "turtleFSI.rom._pod",
    "galerkin": "turtleFSI.rom.pod_galerkin",
    "galerkin_core": "turtleFSI.rom._pod_galerkin",
    "intrusive": "turtleFSI.rom._pod_galerkin",
    "plots": "turtleFSI.cases.plots",
    "plotting": "turtleFSI.cases.plots",
    "snapshots": "turtleFSI.cases.snapshots",
    "pod_rom": "turtleFSI.rom.pod",
    "pod_galerkin_rom": "turtleFSI.rom.pod_galerkin",
    "run_case_rom": "turtleFSI.rom.pod",
    "run_case_pod_galerkin": "turtleFSI.rom.pod_galerkin",
    "non_intrusive.build_rom": "turtleFSI.rom._pod.build",
    "non_intrusive.run_tf_fsi_rom_pipeline": "turtleFSI.cases.tf_fsi_pipeline",
    "non_intrusive.run_pod_structured_fsi_rom": "turtleFSI.rom._pod.structured_pod",
    "non_intrusive.run_structured_fsi_rom": "turtleFSI.rom._pod.structured",
    "data_fit.tf_fsi_pipeline": "turtleFSI.cases.tf_fsi_pipeline",
    "plotting.plot_rom_error_grid": "turtleFSI.cases.plots.plot_error_grid",
    "snapshots.export_tf_fsi_rom_snapshots": "turtleFSI.cases.snapshots.export_tf_fsi",
    "snapshots.export_visualization_rom_snapshots": "turtleFSI.cases.snapshots.export_visualization",
}


def _install_aliases() -> None:
    for old, new in _ALIASES.items():
        old_name = f"{__name__}.{old}"
        if old_name not in sys.modules:
            sys.modules[old_name] = importlib.import_module(new)


_install_aliases()
