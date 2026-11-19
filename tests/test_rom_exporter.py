from pathlib import Path

import numpy as np

from turtleFSI.cases.snapshots.export_tf_fsi import (
    flatten_scalar_snapshots,
    flatten_vector_snapshots,
    parse_xdmf_timeseries,
)
from turtleFSI.cases.snapshots.export_visualization import (
    flatten_scalar_components,
    flatten_vector_components,
)
from turtleFSI.cases.snapshots.build_domain_mesh_map import (
    build_topology,
    classify_domain_markers,
    resolve_cli_paths_and_ids,
)
from turtleFSI.cases.snapshots.dvp_export import _write_snapshot_bundle


class _FakeDofMap:
    def __init__(self, dofs):
        self._dofs = dofs

    def dofs(self):
        return self._dofs


class _FakeSubspace:
    def __init__(self, dofs):
        self._dofs = dofs

    def dofmap(self):
        return _FakeDofMap(self._dofs)


class _FakeDVP:
    def __init__(self, subspace_dofs):
        self._subspace_dofs = subspace_dofs

    def num_sub_spaces(self):
        return len(self._subspace_dofs)

    def sub(self, index):
        return _FakeSubspace(self._subspace_dofs[index])


def test_flatten_vector_snapshots_orders_all_components_by_node():
    values = np.array(
        [
            [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]],
            [[4.0, 40.0, 400.0], [5.0, 50.0, 500.0], [6.0, 60.0, 600.0]],
        ]
    )
    out = flatten_vector_snapshots(values, np.array([2, 0], dtype=np.int32))
    expected = np.array(
        [
            [3.0, 6.0],
            [30.0, 60.0],
            [300.0, 600.0],
            [1.0, 4.0],
            [10.0, 40.0],
            [100.0, 400.0],
        ]
    )
    assert np.allclose(out, expected)


def test_flatten_scalar_snapshots_transposes_to_dofs_by_time():
    values = np.array(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )
    out = flatten_scalar_snapshots(values, np.array([1, 2], dtype=np.int32))
    expected = np.array([[2.0, 5.0], [3.0, 6.0]])
    assert np.allclose(out, expected)


def test_flatten_vector_components_keeps_all_3d_components():
    values = np.array(
        [
            [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]],
            [[3.0, 30.0, 300.0], [4.0, 40.0, 400.0]],
        ]
    )
    out = flatten_vector_components(values)
    expected = np.array(
        [
            [1.0, 3.0],
            [10.0, 30.0],
            [100.0, 300.0],
            [2.0, 4.0],
            [20.0, 40.0],
            [200.0, 400.0],
        ]
    )
    assert np.allclose(out, expected)


def test_flatten_scalar_components_removes_singleton_component_axis():
    values = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    out = flatten_scalar_components(values)
    expected = np.array([[1.0, 3.0], [2.0, 4.0]])
    assert np.allclose(out, expected)


def test_parse_xdmf_timeseries_reads_hdf_paths_and_times(tmp_path: Path):
    xdmf = tmp_path / "velocity.xdmf"
    xdmf.write_text(
        """<?xml version="1.0"?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="TimeSeries_Velocity" GridType="Collection" CollectionType="Temporal">
      <Grid Name="mesh" GridType="Uniform">
        <Time Value="0.1" />
        <Attribute Name="Velocity" AttributeType="Vector" Center="Node">
          <DataItem Dimensions="4 3" Format="HDF">velocity.h5:/VisualisationVector/0</DataItem>
        </Attribute>
      </Grid>
      <Grid>
        <Time Value="0.2" />
        <Attribute Name="Velocity" AttributeType="Vector" Center="Node">
          <DataItem Dimensions="4 3" Format="HDF">velocity.h5:/VisualisationVector/1</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>
""",
        encoding="ascii",
    )

    spec = parse_xdmf_timeseries(xdmf)
    assert spec.h5_path == tmp_path / "velocity.h5"
    assert spec.dataset_paths == ["/VisualisationVector/0", "/VisualisationVector/1"]
    assert np.allclose(spec.times, np.array([0.1, 0.2]))
    assert spec.attribute_name == "Velocity"


def test_domain_mesh_map_builds_interface_nodes_for_3d_cells():
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    cells = np.array(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        dtype=np.int32,
    )
    cell_regions = classify_domain_markers(
        np.array([10, 20], dtype=np.int64),
        fluid_ids={10},
        solid_ids={20},
    )

    topology = build_topology(coordinates, cells, cell_regions)

    assert topology["fluid_nodes"].tolist() == [0]
    assert topology["solid_nodes"].tolist() == [4]
    assert topology["interface_nodes"].tolist() == [1, 2, 3]
    assert topology["interface_facets"].tolist() == [[1, 2, 3]]


def test_domain_mesh_map_cli_resolves_results_folder_defaults(tmp_path: Path):
    results = tmp_path / "run" / "1"
    mesh_dir = results / "Mesh"
    checkpoint_dir = results / "Checkpoint"
    mesh_dir.mkdir(parents=True)
    checkpoint_dir.mkdir()
    mesh_path = mesh_dir / "mesh.h5"
    mesh_path.write_bytes(b"placeholder")
    (checkpoint_dir / "default_variables.json").write_text(
        '{"dx_f_id": 7, "dx_s_id": 9}',
        encoding="ascii",
    )

    args = type(
        "Args",
        (),
        {
            "results_folder": str(results),
            "mesh_h5": None,
            "output_dir": None,
            "fluid_id": None,
            "solid_id": None,
        },
    )()

    mesh_h5_path, output_dir, fluid_ids, solid_ids = resolve_cli_paths_and_ids(args)

    assert mesh_h5_path == mesh_path
    assert output_dir == results / "ROM" / "domain_mesh_map"
    assert fluid_ids == {7}
    assert solid_ids == {9}


def test_dvp_bundle_writes_supremizer_snapshots(tmp_path: Path):
    output = tmp_path / "dvp_snapshots.npz"
    summary = tmp_path / "dvp_snapshots_summary.json"
    dvp = _FakeDVP(
        [
            np.array([0, 3], dtype=np.int64),
            np.array([1, 4], dtype=np.int64),
            np.array([2, 5], dtype=np.int64),
        ]
    )

    _write_snapshot_bundle(
        output_path=output,
        summary_path=summary,
        snapshots=[
            np.arange(6, dtype=float),
            np.arange(6, dtype=float) + 10.0,
        ],
        supremizer_snapshots=[
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ],
        times=[0.1, 0.2],
        counters=[1, 2],
        DVP=dvp,
        metadata={"problem": "fake", "extrapolation": "none"},
    )

    data = np.load(output)
    assert data["dvp"].shape == (6, 2)
    assert data["supremizer_snapshots"].shape == (2, 2)
    assert data["velocity_dofs"].tolist() == [1, 4]
    assert '"includes_supremizer_snapshots": true' in summary.read_text(encoding="ascii")
