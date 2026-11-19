import numpy as np

from turtleFSI.rom._pod.adjacency import (
    expand_node_neighborhoods,
    nodes_to_vector_dofs,
    unpack_csr_rows,
)


def test_expand_node_neighborhoods_and_vector_dofs():
    offsets = np.array([0, 1, 3, 4], dtype=np.int64)
    indices = np.array([1, 0, 2, 1], dtype=np.int32)
    rows = unpack_csr_rows(offsets, indices)
    neighborhoods = expand_node_neighborhoods(rows, depth=1, include_self=True)

    assert [row.tolist() for row in neighborhoods] == [[0, 1], [0, 1, 2], [1, 2]]
    assert nodes_to_vector_dofs(np.array([2, 0], dtype=np.int32), value_dim=2).tolist() == [
        4,
        5,
        0,
        1,
    ]
