"""Check energy reporting independently of concatenated field SVD metadata."""

import numpy as np
import pytest

from turtleFSI.rom._pod_galerkin.build_monolithic_pod_basis import _retained_projection_energy


def test_blocked_energy_uses_all_fields_and_removes_mean():
    # First field has variance 2 and is discarded; second has variance 8 and
    # is retained exactly. The last field alone would misleadingly report 1.
    snapshots = np.array([[99., 101.], [198., 202.]])
    reconstruction = np.array([[100., 100.], [198., 202.]])
    assert _retained_projection_energy(snapshots, reconstruction) == pytest.approx(.8)
    assert _retained_projection_energy(snapshots[0:1], reconstruction[0:1]) == pytest.approx(0.)
    assert _retained_projection_energy(snapshots[1:2], reconstruction[1:2]) == pytest.approx(1.)


def test_constant_snapshots_have_no_discarded_variance():
    snapshots = np.ones((3, 2))
    assert _retained_projection_energy(snapshots, snapshots) == 1.
