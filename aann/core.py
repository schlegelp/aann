import numpy as np

from scipy.spatial import Delaunay

from . import _aann

__all__ = ["all_nearest_neighbours"]


def all_nearest_neighbours(x, y):
    """For all points in x find the nearest neighbor among points in z.

    Parameters
    ----------
    x :    (N, 3) array | scipy Delaunay
            3d points cloud or precomputed delaunay triangulation.
    y:     (M, 3) array | scipy Delaunay
            3d points cloud or precomputed delaunay triangulation.

    Returns
    -------
    d :     (N, ) array of floats
            Distances to nearest neighbors for each point in x.
    i :     (N, ) array of ints
            Indices into y of nearest neighbors for each point in x.

    """
    if not isinstance(x, Delaunay):
        _assert_cloud_3d(x)
        x = x.astype(np.float64, copy=False)

        # Calculate delaunay triangulation
        x = Delaunay(x, qhull_options="QJ")
    else:
        _assert_cloud_3d(x.points)

    if not isinstance(y, Delaunay):
        _assert_cloud_3d(y)
        y = y.astype(np.float64, copy=False)

        # Calculate delaunay triangulation
        y = Delaunay(y, qhull_options="QJ")
    else:
        _assert_cloud_3d(y.points)

    return _aann.all_nearest_neighbours(
        x.points,
        x.vertex_neighbor_vertices[0].astype(np.uint64, copy=False),
        x.vertex_neighbor_vertices[1].astype(np.uint64, copy=False),
        y.points,
        y.vertex_neighbor_vertices[0].astype(np.uint64, copy=False),
        y.vertex_neighbor_vertices[1].astype(np.uint64, copy=False),
    )


def _assert_cloud_3d(x):
    """Check if `x` is 3d point cloud."""
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2
    assert x.shape[1] == 3
