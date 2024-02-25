import aann
import time

import numpy as np
import pandas as pd

from scipy.spatial import Delaunay, KDTree


def test_random():
    N = 100000
    x = Delaunay(np.random.rand(N, 3))
    y = Delaunay(np.random.rand(N, 3))

    start_time = time.time()
    d, i = aann.all_nearest_neighbours(x, y)
    end_time = time.time()

    print("Distance to nearest neighbor for each point in x:", d)
    print("Index of nearest neighbor for each point in x:", i)
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    assert d.shape == (N,)
    assert i.shape == (N,)

    assert np.all(i >= 0)
    assert np.all(i < N)


def test_neurons():
    # Load two example neurons as point clouds
    x = pd.read_csv(
        "/Users/philipps/Downloads/720575940638426064.swc",
        comment="#",
        header=None,
        sep=" ",
    )[[2, 3, 4]].values
    y = pd.read_csv(
        "/Users/philipps/Downloads/720575940613656978.swc",
        comment="#",
        header=None,
        sep=" ",
    )[[2, 3, 4]].values

    print(f"Testing neurons with {x.shape[0]:,} and {y.shape[0]:,} points")

    N = x.shape[0]

    start_time = time.time()
    x = Delaunay(x)
    y = Delaunay(y)
    end_time = time.time()
    print(f"Generating Delaunay: {end_time - start_time:.2f} seconds")

    # Run one warm-up
    _ = aann.all_nearest_neighbours(x, y)

    start_time = time.time()
    d, i = aann.all_nearest_neighbours(x, y)
    end_time = time.time()

    print("Distance to nearest neighbor for each point in x:", d)
    print("Index of nearest neighbor for each point in x:", i)
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    assert d.shape == (N,)
    assert i.shape == (N,)

    assert np.all(i >= 0)
    assert np.all(i < N)


def test_neurons_kdtree():
    # Load two example neurons as point clouds
    x = pd.read_csv(
        "/Users/philipps/Downloads/720575940638426064.swc",
        comment="#",
        header=None,
        sep=" ",
    )[[2, 3, 4]].values
    y = pd.read_csv(
        "/Users/philipps/Downloads/720575940613656978.swc",
        comment="#",
        header=None,
        sep=" ",
    )[[2, 3, 4]].values

    N = x.shape[0]

    start_time = time.time()
    tree = KDTree(y)
    end_time = time.time()
    print(f"Generating KDTree: {end_time - start_time:.2f} seconds")

    # Run one warm-up
    _ = tree.query(x)

    start_time = time.time()
    d, i = tree.query(x)
    end_time = time.time()

    print("Distance to nearest neighbor for each point in x:", d)
    print("Index of nearest neighbor for each point in x:", i)
    print(f"Tree query time: {end_time - start_time:.2f} seconds")


def _neuron_to_fake_delaunay(swc):
    points = swc[[2, 3, 4]].values.astype(np.float64, copy=False)
    parents = swc[6].values
    nodes = swc[0].values
    indices = []
    neighbors = []
    for i, n in enumerate(nodes):
        # Start
        indices.append(len(neighbors))
        # Add this node's parent as neighbor
        if parents[i] >= 0:
            neighbors.append(parents[i])
        # Add all children as neighbors
        neighbors += list(np.where(parents == n)[0])
        # End
        indices.append(len(neighbors))
    return np.array(indices, dtype=np.uint64), np.array(neighbors, dtype=np.uint64), points

"""
def test_neurons_no_delaunay():
    # Load two example neurons as point clouds
    swc1 = pd.read_csv(
        "/Users/philipps/Downloads/720575940638426064.swc",
        comment="#",
        header=None,
        sep=" ",
    )
    ind1, neigh1, points1 = _neuron_to_fake_delaunay(swc1)
    swc2 = pd.read_csv(
        "/Users/philipps/Downloads/720575940613656978.swc",
        comment="#",
        header=None,
        sep=" ",
    )
    ind2, neigh2, points2 = _neuron_to_fake_delaunay(swc2)

    start_time = time.time()
    d, i = aann._aann.all_nearest_neighbours(
        points1,
        ind1,
        neigh1,
        points2,
        ind2,
        neigh2
    )
    end_time = time.time()

    print("Distance to nearest neighbor for each point in x:", d)
    print("Index of nearest neighbor for each point in x:", i)
    print(f"Execution time: {end_time - start_time:.2f} seconds")
"""

if __name__ == "__main__":
    test_neurons()
    test_neurons_kdtree()
