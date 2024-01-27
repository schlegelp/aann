import aann
import time

import numpy as np
import pandas as pd

from scipy.spatial import Delaunay
import time


def test_random():
    N = 10000
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

    N = x.shape[0]

    start_time = time.time()
    x = Delaunay(x)
    y = Delaunay(y)
    end_time = time.time()
    print(f"Generating Delaunay: {end_time - start_time:.2f} seconds")

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
