import os

from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shull

from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import csr_matrix

from . import _aann

__all__ = [
    "AANN",
    "prepare_many",
    "all_by_all",
]

# Precomputed-triangulation types accepted anywhere a raw cloud is: scipy's
# ``Delaunay`` and shull's ``Delaunay`` (``shull.Delaunay3d`` subclasses it).
# Both expose the same ``.points`` and ``.vertex_neighbor_vertices`` interface,
# so they are handled identically.
_DELAUNAY_TYPES = (Delaunay, shull.Delaunay)


# A neighbourhood graph ready to hand to the Rust search. ``perm`` is the point
# permutation applied by cache-locality reordering (new index -> original
# index), or ``None`` if not reordered; it is used to map results back to the
# caller's original point order. Private: :class:`AANN` is the public handle.
_PreparedGraph = namedtuple(
    "_PreparedGraph", ["points", "indptr", "indices", "perm", "dtype"]
)


class AANN:
    """A prepared neighbourhood-graph index -- scipy ``cKDTree`` style.

    ``index = aann.AANN(target); d, i = index.query(x)`` is the entry point for a
    nearest-neighbour search. The target's neighbourhood graph is built (and its
    coordinates SIMD-packed inside Rust) **once** at construction and reused
    across every :meth:`query`, so querying the same target repeatedly -- one
    target vs many query clouds, or an all-by-all -- avoids rebuilding it.

    Unlike a KD-tree, aann is a *cloud-vs-cloud* method: the points passed to
    :meth:`query` are themselves triangulated into a graph, and the warm-started
    descent walks that graph so each query starts near its neighbour's answer.
    Querying a handful of scattered, unrelated points therefore gains nothing
    from the warm start (each is a cold descent) -- pass a coherent cloud.

    Parameters
    ----------
    target :  (M, 3) array | scipy.spatial.Delaunay | shull.Delaunay
              The reference cloud to search against. A precomputed Delaunay
              (either backend) is used directly.
    graph :   "delaunay" | "knn"
              Neighbourhood graph used for the greedy descent:

              - "delaunay" (default): the Delaunay triangulation. Its edges span
                the space, so the descent reliably reaches the true nearest
                neighbour even for query points far off the target cloud.
              - "knn": a symmetric k-nearest-neighbour graph. Much faster to
                build, but its edges only span the local point spacing -- fine
                when the query points lie *within* the target's distribution
                (self-join, space-filling data), but poor for joins between
                spatially separated clouds (e.g. two neurons), where the descent
                cannot cross empty gaps. Use the default "delaunay" for that.

              A precomputed ``Delaunay`` target is always used directly.
    graph_k : int
              Neighbours per node when ``graph="knn"`` (ignored otherwise, and
              not to be confused with ``k`` in :meth:`query`).
    reorder : bool
              If True (default), renumber points into Morton (Z-order) so that
              spatially-near -- and therefore graph-near -- points are contiguous
              in memory. The descent's scattered lookups then hit nearby cache
              lines (~1.5-2.5x faster on large, cache-spilling clouds). It costs
              an O(N log N) reorder up front, so it pays off across repeated
              queries (break-even is a handful); for a single query it is a mild
              net loss. Results from e.g. :meth:`query` are automatically mapped
              back to the original point order!
    dtype :   None | "float32" | "float64" | np.dtype
              Search precision. The descent is bound by memory bandwidth on the
              scattered coordinate lookups, so "float32" roughly halves the bytes
              moved per point and is meaningfully faster on large clouds, at the
              cost of some precision (~7 significant digits). ``None`` (default)
              infers it from ``target`` (float32 only when the target is float32).
              A scipy ``Delaunay`` always carries float64 points; a shull
              ``Delaunay`` keeps the dtype it was built from.
    backend : "shull" | "scipy"
              Delaunay implementation used to triangulate a raw point cloud when
              ``graph="delaunay"``. "shull" (default) uses the ``shull`` package
              (a Rust sweep-hull triangulator, ~2-2.4x faster to build); "scipy"
              uses Qhull (which joggles coincident points apart via ``QJ``).
              Both handle exact-duplicate / coincident points correctly.
              Ignored for ``graph="knn"`` and for precomputed Delaunay input
              (which is used directly, whichever backend produced it).

    Attributes
    ----------
    data :  (M, 3) array
            The target points (in the search dtype, original order).
    n :     int
            Number of target points.
    dtype : np.dtype
            Search precision (float32/float64).
    """

    def __init__(self, target, graph="delaunay", graph_k=16, reorder=True,
                 dtype=None, backend="shull"):
        g = _prepare(
            target, graph=graph, graph_k=graph_k, reorder=reorder,
            dtype=dtype, backend=backend,
        )
        self.dtype = g.dtype
        self.n = g.points.shape[0]
        self._perm = g.perm
        if g.dtype == np.dtype(np.float32):
            self._rust = _aann.PreparedF32(g.points, g.indptr, g.indices)
        else:
            self._rust = _aann.PreparedF64(g.points, g.indptr, g.indices)

    def query(self, x, k=1, distance_upper_bound=None, ef=None,
              graph="delaunay", graph_k=16, backend="shull"):
        """For each point of ``x`` find its k nearest neighbour(s) in the target.

        Parameters
        ----------
        x :     (N, 3) array | scipy.spatial.Delaunay | shull.Delaunay | AANN
                The query cloud. A raw array is triangulated on the fly (per
                ``graph`` / ``graph_k`` / ``backend``); a precomputed Delaunay
                (scipy or shull) is used directly; another ``AANN`` is used
                directly (its dtype must match this one), which also avoids
                re-packing it -- the fast path an all-by-all uses.
        k :     int
                Number of neighbours to return per query point (as in
                ``scipy.spatial.cKDTree.query``). ``k=1`` (default) runs the
                exact greedy descent and returns ``(N,)`` arrays. ``k>1`` runs a
                best-first graph search and returns ``(N, k)`` arrays, each row
                sorted by ascending distance; results are *approximate* (see
                ``ef``). Rows are padded with ``d=inf`` / ``i=len(target)`` if
                fewer than k neighbours are reachable (k > ``len(target)`` or a
                disconnected target graph).
        distance_upper_bound : None | float
                If given, only report a neighbour within this distance; query
                points whose nearest neighbour is farther get ``d=inf`` and
                ``i=len(target)`` -- the same missing-neighbour convention as
                ``scipy.spatial.cKDTree.query``, so hits are ``i < len(target)``.
                ``None`` (default) and ``inf`` disable the bound. This filters the
                results only: the descent has no geometric lower bound to prune
                against, so unlike a KD-tree the bound does not speed it up.
        ef :    None | int
                Search breadth for ``k>1`` (ignored for ``k=1``): the best-first
                search keeps the ``ef >= k`` closest candidates and returns the
                top k, so larger values trade speed for recall. ``None`` (default)
                uses ``ef = k``, which recovers 100% of the exact k-NN on Delaunay
                graphs and >=99% on knn graphs in benchmarks. Clamped to
                ``[k, len(target)]``.
        graph, graph_k, backend :
                How a raw-array / ``Delaunay`` ``x`` is triangulated (see the
                constructor). Ignored when ``x`` is an ``AANN``.

        Returns
        -------
        d :     (N,) or (N, k) array of floats
                Distances to the nearest neighbours, ascending along the last
                axis for k>1. ``inf`` where no neighbour qualifies.
        i :     (N,) or (N, k) array of ints
                Indices into the target points. ``len(target)`` where no
                neighbour qualifies.
        """
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")

        if isinstance(x, AANN):
            if x.dtype != self.dtype:
                raise ValueError(
                    "query AANN dtype must match this AANN's dtype "
                    f"({x.dtype} vs {self.dtype})"
                )
            if k == 1:
                d, i = self._rust.query_prepared(x._rust)
            else:
                ef_eff = _effective_ef(ef, k, self.n)
                d, i = self._rust.query_prepared_k(x._rust, k, ef_eff)
            gx_perm = x._perm
        else:
            gx = _build_prepared(x, graph, graph_k, self.dtype, backend)
            if k == 1:
                d, i = self._rust.query(gx.points, gx.indptr, gx.indices)
            else:
                ef_eff = _effective_ef(ef, k, self.n)
                d, i = self._rust.query_k(gx.points, gx.indptr, gx.indices, k, ef_eff)
            gx_perm = gx.perm

        return _finalize_results(
            d, i, gx_perm, self._perm, self.n, k, distance_upper_bound
        )

    @property
    def data(self):
        pts = self._rust.data
        if self._perm is not None:
            # Rust holds points in internal (Morton) order; echo the caller's
            # original order, as scipy's cKDTree.data does.
            out = np.empty_like(pts)
            out[self._perm] = pts
            return out
        return pts

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"AANN(n={self.n}, dtype={self.dtype})"


def prepare_many(clouds, graph="delaunay", graph_k=16, reorder=True, dtype=None,
                 backend="shull", workers=None):
    """Build many :class:`AANN` indices in parallel across a thread pool.

    The build-phase companion to :func:`all_by_all`, and equivalent to
    ``[AANN(c, ...) for c in clouds]`` but parallel. Both Delaunay backends
    (scipy Qhull and shull) release the GIL during triangulation, as do the Rust
    adjacency builder and the numpy/scipy reorder step, so graph construction
    scales across cores -- useful for all-by-all workloads where the per-cloud
    build would otherwise dominate.

    Parameters
    ----------
    clouds :  sequence of (N, 3) array | scipy.spatial.Delaunay | shull.Delaunay
              Point clouds (or precomputed triangulations) to build indices for.
    graph, graph_k, reorder, dtype, backend :
              Forwarded to :class:`AANN` for every cloud.
    workers : int | None
              Thread-pool size; ``None`` uses ``os.cpu_count()``.

    Returns
    -------
    list of AANN
        One index per input cloud, in the same order. Feed straight into
        :func:`all_by_all`.
    """
    clouds = list(clouds)
    if workers is None:
        workers = os.cpu_count() or 1

    def _one(cloud):
        return AANN(
            cloud, graph=graph, graph_k=graph_k, reorder=reorder, dtype=dtype,
            backend=backend,
        )

    if workers <= 1 or len(clouds) <= 1:
        return [_one(c) for c in clouds]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, clouds))


def all_by_all(anns, pairs=None, workers=None, k=1, distance_upper_bound=None,
               ef=None):
    """Nearest-neighbour join for many :class:`AANN` pairs, in parallel.

    The Rust search releases the GIL, so independent pairs run concurrently on a
    thread pool -- near-linear speedup with cores for an all-by-all workload.
    Because every operand is a prepared :class:`AANN` (points already SIMD-packed
    in Rust), no packing happens per pair.

    Parameters
    ----------
    anns :    sequence of AANN
              The operands, indexed by ``pairs``. Build them once with
              :class:`AANN` / :func:`prepare_many` and reuse across every pair
              each appears in. Only ``AANN`` instances are accepted.
    pairs :   iterable of (i, j) | None
              Which ordered pairs ``(query, target)`` to compute -- for pair
              ``(i, j)`` the nearest neighbour in ``anns[j]`` is found for each
              point of ``anns[i]``. ``None`` (default) means every ``i != j`` (a
              full all-by-all).
    workers : int | None
              Thread-pool size; ``None`` uses ``os.cpu_count()``.
    k :       int
              Number of neighbours per query point (``(N, k)`` results for k>1).
    distance_upper_bound : None | float
              Report ``inf`` distance / index ``len(target)`` for query points
              with no neighbour within the bound.
    ef :      None | int
              Search breadth for k>1 (see :meth:`AANN.query`).

    Returns
    -------
    list of (distances, indices)
        One ``(d, i)`` result per entry in ``pairs``, in the same order.
    """
    anns = list(anns)
    for a in anns:
        if not isinstance(a, AANN):
            raise TypeError(
                "all_by_all expects AANN instances; build them with AANN(...) "
                f"or prepare_many(...). Got {type(a).__name__}."
            )
    if anns and any(a.dtype != anns[0].dtype for a in anns):
        raise ValueError("all AANN operands must share a dtype")

    if pairs is None:
        n = len(anns)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    else:
        pairs = [tuple(p) for p in pairs]

    if workers is None:
        workers = os.cpu_count() or 1

    def _one(pair):
        i, j = pair
        return anns[j].query(
            anns[i], k=k, distance_upper_bound=distance_upper_bound, ef=ef,
        )

    if workers <= 1 or len(pairs) <= 1:
        return [_one(p) for p in pairs]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, pairs))


def _prepare(cloud, graph="delaunay", graph_k=16, reorder=True, dtype=None,
             backend="shull"):
    """Build a (optionally reordered) ``_PreparedGraph`` from a cloud/Delaunay.

    The graph-building core behind :class:`AANN`; see it for the parameters.
    """
    if dtype is None:
        dtype = _cloud_dtype(cloud)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"dtype must be float32 or float64, got {dtype}")

    points, indptr, indices = _build_graph(cloud, graph, graph_k, backend)
    points = points.astype(dtype, copy=False)

    perm = None
    if reorder:
        perm = _morton_perm(points)
        points, indptr, indices = _reorder(points, indptr, indices, perm)

    return _PreparedGraph(points, indptr, indices, perm, dtype)


def _effective_ef(ef, k, n_y):
    """Resolve the k>1 search breadth: user value or default, clamped to [k, n_y].

    ``ef = k`` is the default: in recall sweeps (uniform, clustered, manifold,
    anisotropic and neuron clouds; k = 2..16) it recovered 100% of the exact
    k-NN on Delaunay graphs and >= 99.3% on knn graphs -- the warm-started
    best-first search expands every candidate closer than the current kth best
    anyway, so extra breadth buys little here while costing ~linearly in time.
    """
    if ef is None:
        ef = k
    return max(1, min(max(int(ef), k), n_y))


def _finalize_results(d, i, gx_perm, gy_perm, n_y, k, distance_upper_bound):
    """Map raw Rust search output back to the caller's point order and apply
    the distance cutoff. Shared by every :meth:`AANN.query` path so the (fiddly)
    marker-safe permutation handling lives once.

    ``gx_perm`` / ``gy_perm`` are the query/target Morton permutations (or
    ``None``); ``n_y`` is the target point count (the missing-neighbour marker).
    """
    # Undo a target reorder so indices refer to y's original order.
    if gy_perm is not None:
        if k == 1:
            i = gy_perm[i]  # target indices -> y's original order
        else:
            # k>1 rows may carry the (inf, len(y)) padding marker; extend the
            # lookup so the marker maps to itself.
            perm_ext = np.append(gy_perm, n_y)
            i = perm_ext[i]
    # Undo a query reorder so rows are in x's original order.
    if gx_perm is not None:
        d_out = np.empty_like(d)
        i_out = np.empty_like(i)
        d_out[gx_perm] = d  # query results -> x's original order
        i_out[gx_perm] = i
        d, i = d_out, i_out

    # Applied after the perm remap so the missing marker (an out-of-range
    # index, matching scipy's convention) never flows through the fancy
    # indexing above. The descent itself cannot use the bound -- see docstring.
    if distance_upper_bound is not None and distance_upper_bound < np.inf:
        miss = d > distance_upper_bound
        d[miss] = np.inf
        i[miss] = n_y

    return d, i


def _cloud_dtype(obj):
    """float32 if the cloud is float32, else float64 (Delaunay is always f64)."""
    if isinstance(obj, _DELAUNAY_TYPES):
        dt = obj.points.dtype
    else:
        dt = np.asarray(obj).dtype
    return np.float32 if np.dtype(dt) == np.float32 else np.float64


def _build_prepared(cloud, graph, graph_k, dtype, backend="shull"):
    """Build a (non-reordered) ``_PreparedGraph`` from a raw cloud/Delaunay."""
    points, indptr, indices = _build_graph(cloud, graph, graph_k, backend)
    return _PreparedGraph(
        points.astype(dtype, copy=False), indptr, indices, None, np.dtype(dtype)
    )


def _morton_perm(points):
    """Argsort into Morton (Z-order) so near points are contiguous in memory."""
    def spread(v):
        v = v.astype(np.uint64) & np.uint64(0x1FFFFF)  # 21 bits / axis -> 63 bits
        v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
        v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
        v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
        return v

    p = np.asarray(points, dtype=np.float64)
    mn = p.min(0)
    span = p.max(0) - mn
    span[span == 0] = 1.0
    q = ((p - mn) / span * (2 ** 21 - 1)).astype(np.uint64)
    key = spread(q[:, 0]) | (spread(q[:, 1]) << np.uint64(1)) | (spread(q[:, 2]) << np.uint64(2))
    return np.argsort(key, kind="stable")


def _reorder(points, indptr, indices, perm):
    """Relabel a CSR graph by ``perm`` (new index -> original index)."""
    n = len(points)
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr.astype(np.int64)))
    a = csr_matrix(
        (np.ones(len(indices), np.int8), (rows, indices.astype(np.int64))),
        shape=(n, n),
    )
    a = a[perm][:, perm]  # symmetric permutation
    a.sort_indices()
    return (
        points[perm],
        a.indptr.astype(np.uint64, copy=False),
        a.indices.astype(np.uint64, copy=False),
    )


def _build_graph(obj, graph, graph_k, backend="shull"):
    """Return ``(points, indptr, indices)`` for the neighbourhood graph.

    ``indptr``/``indices`` are a CSR adjacency: the neighbours of vertex ``v``
    are ``indices[indptr[v]:indptr[v + 1]]``.
    """
    # A precomputed Delaunay (scipy or shull) is always used directly: both
    # expose the same vertex-adjacency CSR via ``vertex_neighbor_vertices``.
    if isinstance(obj, _DELAUNAY_TYPES):
        points = np.asarray(obj.points)
        _assert_cloud_3d(points)
        indptr, indices = obj.vertex_neighbor_vertices
        return (
            points.astype(np.float64, copy=False),
            np.asarray(indptr).astype(np.uint64, copy=False),
            np.asarray(indices).astype(np.uint64, copy=False),
        )

    _assert_cloud_3d(obj)
    points = np.ascontiguousarray(obj, dtype=np.float64)

    if graph == "delaunay":
        if backend == "shull":
            return _delaunay_shull(points)
        if backend != "scipy":
            raise ValueError(
                f"Unknown backend {backend!r}; expected 'scipy' or 'shull'."
            )
        d = Delaunay(points, qhull_options="QJ")
        indptr, indices = d.vertex_neighbor_vertices
        return (
            d.points.astype(np.float64, copy=False),
            indptr.astype(np.uint64, copy=False),
            indices.astype(np.uint64, copy=False),
        )
    elif graph == "knn":
        return _knn_graph(points, graph_k)
    else:
        raise ValueError(
            f"Unknown graph type {graph!r}; expected 'delaunay' or 'knn'."
        )


def _delaunay_shull(points):
    """Delaunay neighbourhood graph via the ``shull`` package.

    shull's 3D triangulation is ~3x faster than scipy's Qhull; the tetrahedra
    are turned into a vertex-adjacency CSR by the GIL-free Rust helper
    ``_aann.graph_from_simplices``. Exact-duplicate points are dropped from the
    triangulation by shull (leaving them as isolated vertices), so we reconnect
    each dropped point to its kept first occurrence -- reported by shull's
    ``coplanar`` attribute -- giving the same duplicate-robustness as the scipy
    backend.
    """
    d = shull.Delaunay3d(points)
    simplices = np.ascontiguousarray(d.simplices, dtype=np.uint64)
    indptr, indices = _aann.graph_from_simplices(simplices, len(points))
    indptr = np.asarray(indptr, dtype=np.uint64)
    indices = np.asarray(indices, dtype=np.uint64)

    coplanar = np.asarray(d.coplanar)
    if len(coplanar) == 0:
        return (points, indptr, indices)

    # Duplicates were dropped: stitch each dropped point (col 0) to its kept
    # first occurrence (col 2) with a symmetric edge, then rebuild the CSR (same
    # csr_matrix idiom as ``_knn_graph``/``_reorder``). One edge to the identical
    # twin is enough for the descent to reach the correct answer.
    n = len(points)
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr.astype(np.int64)))
    cols = indices.astype(np.int64)
    dropped = coplanar[:, 0].astype(np.int64)
    rep = coplanar[:, 2].astype(np.int64)
    r = np.concatenate([rows, cols, dropped, rep])
    c = np.concatenate([cols, rows, rep, dropped])
    g = csr_matrix((np.ones(r.size, dtype=np.int8), (r, c)), shape=(n, n))
    g.sum_duplicates()
    g.sort_indices()
    return (
        points,
        g.indptr.astype(np.uint64, copy=False),
        g.indices.astype(np.uint64, copy=False),
    )


def _knn_graph(points, k):
    """Symmetric k-nearest-neighbour graph as ``(points, indptr, indices)``."""
    n = len(points)
    k_eff = min(k, n - 1)
    # Query k_eff + 1 because the first neighbour returned is the point itself.
    _, idx = cKDTree(points).query(points, k=k_eff + 1)
    idx = np.asarray(idx).reshape(n, -1)

    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.reshape(-1)
    keep = rows != cols  # drop self-loops robustly (handles duplicate points)
    rows, cols = rows[keep], cols[keep]

    # Symmetrise: an edge exists if either endpoint lists the other.
    r = np.concatenate([rows, cols])
    c = np.concatenate([cols, rows])
    g = csr_matrix((np.ones(r.size, dtype=np.int8), (r, c)), shape=(n, n))
    g.sum_duplicates()
    g.sort_indices()

    return (
        points,
        g.indptr.astype(np.uint64, copy=False),
        g.indices.astype(np.uint64, copy=False),
    )


def _assert_cloud_3d(x):
    """Check if `x` is 3d point cloud."""
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2
    assert x.shape[1] == 3
