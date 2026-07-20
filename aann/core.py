import os

from collections import namedtuple, defaultdict
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
    "all_by_all_grouped",
]

# Precomputed-triangulation types accepted anywhere a raw cloud is: scipy's
# ``Delaunay`` and shull's ``Delaunay`` (``shull.Delaunay3d`` subclasses it).
# Both expose the same ``.points`` and ``.vertex_neighbor_vertices`` interface,
# so they are handled identically.
_DELAUNAY_TYPES = (Delaunay, shull.Delaunay)

# dtype of the CSR neighbourhood graph (``indptr``/``indices``) handed to Rust.
# The Rust side stores the adjacency as ``u32``, and ``PyReadonlyArray`` matches
# dtype exactly -- no implicit cast -- so anything else raises ``TypeError`` at
# the boundary rather than converting silently. This is also the cheaper cast on
# our side: scipy hands back ``int32`` from both ``vertex_neighbor_vertices`` and
# ``csr_matrix.indices``, so widening to 64 bits used to double the array only to
# have Rust index a vertex set that never approaches 2**32.
#
# Consequence: a single cloud is capped at 2**32 - 1 points. The largest neuron
# we have seen is ~14k, and shull's triangulator itself refuses at u32::MAX.
_CSR_DTYPE = np.uint32


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
        # Axis-aligned bounding box of the target (search dtype, order-independent
        # so the Morton reorder does not matter). Cached for the box-to-box
        # short-circuit a bounded query can take before triangulating a raw
        # cloud. Empty target -> (inf, -inf), which prunes everything.
        if self.n:
            self._bbox = (g.points.min(0), g.points.max(0))
        else:
            inf = np.full(3, np.inf, dtype=g.dtype)
            self._bbox = (inf, -inf)
        if g.dtype == np.dtype(np.float32):
            self._rust = _aann.PreparedF32(g.points, g.indptr, g.indices)
        else:
            self._rust = _aann.PreparedF64(g.points, g.indptr, g.indices)

    def query(self, x, k=1, distance_upper_bound=None, ef=None,
              graph="delaunay", graph_k=16, backend="shull",
              blocked=False, block=8):
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
                ``None`` (default) and ``inf`` disable the bound. The target
                cloud's bounding box lets the search skip individual query points
                -- or the whole cloud in one test -- that provably fall outside
                the bound, so unlike the unbounded search a tight bound can also
                make it *faster* on well-separated clouds.
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
        blocked : bool
                Opt into the *blocked/batched* descent (experimental). It processes
                query points in blocks so that query points sharing a target vertex
                have that vertex's neighbours gathered once and reused -- roughly
                ~1.4x faster on the descent, memory-bandwidth bound. Distances are
                **identical** to the default path (k=1 is exact on a Delaunay graph
                regardless of the descent's start); indices likewise, except in the
                pathological tie case noted under :func:`all_by_all_grouped`. Only
                takes effect on the fast
                prepared-vs-prepared ``k=1`` path (``x`` is an ``AANN``); it is
                silently ignored for ``k>1``, raw/``Delaunay`` ``x`` -- those use the
                standard path. ``distance_upper_bound`` is supported. Default False.

                **Assumes the query points are stored in Morton-coherent order** --
                the block carries a single warm seed, so it relies on consecutive
                query points being spatial neighbours. ``AANN`` builds indices with
                cache-locality reordering by default (``reorder=True``), so this
                holds here. A caller that keeps points in their original order (e.g.
                to align external per-point data) can get a *worse-than-1x* result
                from a poor warm start; for that case Morton-sort the queries first,
                or use :func:`all_by_all_grouped` (which sorts internally).
        block : int
                Block size for ``blocked`` (query points per batch). Ignored unless
                ``blocked``. Larger blocks give more gather reuse but share one warm
                seed across more points; the useful range is ~8-32 (peak ~8-16) and
                is data- and order-dependent. Default 8. (For a whole all-by-all the
                grouped path
                :func:`all_by_all_grouped`, which Morton-sorts across all queries,
                favours larger blocks ~32-128.)

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

        # Normalise the bound: None / inf disable it. When finite it is passed to
        # Rust, which prunes via the target's bounding box and marks out-of-range
        # points as misses; ``_finalize_results`` re-applies it as a safety net.
        ub = None
        if distance_upper_bound is not None and distance_upper_bound < np.inf:
            ub = float(distance_upper_bound)

        if isinstance(x, AANN):
            if x.dtype != self.dtype:
                raise ValueError(
                    "query AANN dtype must match this AANN's dtype "
                    f"({x.dtype} vs {self.dtype})"
                )
            if k == 1:
                # Blocked/batched descent (opt-in prototype): reuses each target
                # vertex's neighbour gather across a block of query points. Supports
                # distance_upper_bound; only the prepared-vs-prepared k=1 path is
                # blocked, every other case uses the standard path (no regression).
                if blocked:
                    d, i = self._rust.query_prepared_blocked(x._rust, block, ub)
                else:
                    d, i = self._rust.query_prepared(x._rust, ub)
            else:
                ef_eff = _effective_ef(ef, k, self.n)
                d, i = self._rust.query_prepared_k(x._rust, k, ef_eff, ub)
            gx_perm = x._perm
        else:
            # Box-to-box short-circuit *before* triangulating a raw cloud: if the
            # whole query cloud is out of range there is no neighbour to find, so
            # skip the (costly) graph build entirely. Rust repeats this test for
            # already-prepared operands, where no triangulation is at stake.
            if ub is not None:
                sc = _box_box_short_circuit(x, self._bbox, ub, k, self.n, self.dtype)
                if sc is not None:
                    return sc
            gx = _build_prepared(x, graph, graph_k, self.dtype, backend)
            if k == 1:
                d, i = self._rust.query(gx.points, gx.indptr, gx.indices, ub)
            else:
                ef_eff = _effective_ef(ef, k, self.n)
                d, i = self._rust.query_k(gx.points, gx.indptr, gx.indices, k, ef_eff, ub)
            gx_perm = gx.perm

        return _finalize_results(d, i, gx_perm, self._perm, self.n)

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
               ef=None, blocked=False, block=8):
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
    blocked : bool
              Use the blocked/batched descent for each pair (see the ``blocked``
              argument of :meth:`AANN.query`): ~1.4x faster on the descent, exact,
              ``k=1`` only. Default False. For a bigger, structural speedup on a
              *whole* all-by-all prefer :func:`all_by_all_grouped`, which shares
              gathers across all query neurons of a target and parallelises more
              coarsely.
    block :   int
              Block size for ``blocked`` (default 8; useful range ~8-32, see
              :meth:`AANN.query`).

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
            blocked=blocked, block=block,
        )

    if workers <= 1 or len(pairs) <= 1:
        return [_one(p) for p in pairs]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, pairs))


def all_by_all_grouped(anns, pairs=None, workers=None, distance_upper_bound=None,
                       block=32):
    """Grouped all-by-all (experimental; k=1 only).

    Equivalent output to :func:`all_by_all` (``k=1``) -- one ``(distances, indices)``
    per entry in ``pairs``, same order -- but organised so that **all** query clouds'
    points are concatenated and Morton-sorted **once** into a single grouped query
    set, which is then descended against **each target** in one blocked pass.
    Spatially-adjacent query points from *different* neurons fall in the same block
    and share a single gather of each target vertex's neighbours -- the cross-pair
    reuse that a pair-at-a-time loop (and the hardware cache) cannot exploit.

    The key is that the concatenation + Morton sort are **target-independent**, so
    they run a single time up front (in Rust, GIL released) rather than once per
    target; each per-target call then only descends + finalizes (mapping back to
    each operand's original point/index order, also in Rust). Python only slices the
    result. Distances are **identical** to :func:`all_by_all`; indices are identical
    wherever the nearest neighbour is unique, and both paths resolve exact-distance
    ties the same way in all but pathological cases (see Notes).

    Parameters
    ----------
    anns, pairs, workers, distance_upper_bound :
              As in :func:`all_by_all`. (No ``k``/``ef``: this is a ``k=1`` path.)
    block :   int
              Blocked-descent block size (default 32; see :meth:`AANN.query`). The
              sweet spot here is ~32-128 and is data-dependent (larger clouds favour
              larger blocks); ~1.95-2.05x single-thread and ~4x multi-thread over
              :func:`all_by_all` were measured at block=32 on dense neuron data.

    Returns
    -------
    list of (distances, indices)
        One ``(d, i)`` per entry in ``pairs``, in the same order. Distances match
        ``all_by_all(..., k=1)`` exactly; indices match except at rare exact-distance
        ties (see Notes).

    Notes
    -----
    This is the fastest path for a large multi-threaded all-by-all, but it does
    **not** scale identically to :func:`all_by_all` -- keep these in mind:

    - **k=1 only.** For ``k>1`` use :func:`all_by_all`.
    - **Descends the whole query set per target.** Each target is queried against
      *all* clouds' points (the shared grouped set), then only the requested pairs
      are sliced out. For the default (``pairs=None``, every ``i != j``) this costs
      only the self-pair extra (~1/N). But for a **sparse custom** ``pairs`` -- few
      query clouds per target -- it over-computes; there, ``distance_upper_bound``
      keeps far points cheap (per-point box prune), or prefer
      ``all_by_all(..., blocked=True)``.
    - **Memory.** The shared grouped set is one Morton-sorted copy of *all* query
      points (``O(total points)``), built once; each in-flight target additionally
      allocates full-length output buffers (``O(total points)``). With ``workers``
      targets running that is ``workers × O(total points)``. Cap ``workers`` if
      memory-bound.
    - **Parallelism is over distinct targets, not pairs.** Great scaling when there
      are many, similarly-connected targets (coarse GIL-free tasks; the per-pair
      :func:`all_by_all` scales poorly here due to task-dispatch/GIL overhead). But
      if ``pairs`` involve few distinct targets it under-uses cores, and a few
      *hub* targets (near many more neurons than the rest) become straggler tasks.
    - **The speedup is data-dependent.** The cross-pair gather reuse is largest when
      many query neurons densely overlap each target; on spatially sparse data it
      shrinks toward the plain blocked descent. Single-threaded, the plain
      ``all_by_all(..., blocked=True)`` is simpler and uses less memory for a similar
      gain; this path's big win is multi-threaded.
    - **Does not avoid the O(N²) pairs list.** Like :func:`all_by_all` it still needs
      ``pairs`` enumerated; ``pairs=None`` at very large N is infeasible for both --
      pre-filter to spatially-near pairs first.
    - **Exact-distance ties are near-deterministic, not guaranteed.** When a query
      point is exactly equidistant from several target points, the descent returns
      the one with the **lowest target index**. Ties are common on grid-quantised
      clouds (e.g. resampled skeletons) and rare on continuous data. This makes the
      returned index independent of *which clouds were passed in* for effectively
      all real inputs -- worth knowing, because the grouped set is built from **all**
      of ``anns`` (not just those named in ``pairs``), so the Morton order, and hence
      each descent's start vertex, shifts when you hand in a different set of
      clouds. A residual case survives: two equidistant targets are only compared if
      the descent actually visits both, which is not guaranteed when they are not
      adjacent in the target's graph. Measured at ~0.02% of tied query points on
      random grid-quantised clouds. If you need a bit-stable index across differing
      cloud sets, do not rely on this.
    """
    anns = list(anns)
    for a in anns:
        if not isinstance(a, AANN):
            raise TypeError(
                "all_by_all_grouped expects AANN instances; build them with "
                f"AANN(...) or prepare_many(...). Got {type(a).__name__}."
            )
    if anns and any(a.dtype != anns[0].dtype for a in anns):
        raise ValueError("all AANN operands must share a dtype")

    n = len(anns)
    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    else:
        pairs = [tuple(p) for p in pairs]
    if workers is None:
        workers = os.cpu_count() or 1

    ub = None
    if distance_upper_bound is not None and distance_upper_bound < np.inf:
        ub = float(distance_upper_bound)

    results = [None] * len(pairs)
    if not pairs:
        return results

    # Each cloud's new->original permutation as int64 (or None), computed ONCE and
    # reused across every target it queries (a cloud is a query for many targets).
    perms = [None if a._perm is None else np.ascontiguousarray(a._perm, dtype=np.int64)
             for a in anns]

    # Build the grouped query handle ONCE for the whole all-by-all: concatenate
    # every cloud's points and Morton-sort them a single time in Rust. This work
    # is target-independent, so hoisting it out of the per-target loop is what
    # keeps the grouped descent fast -- re-sorting per target used to cost ~half
    # the gain. `gq` is frozen/read-only, shared across the thread pool below.
    grouped_cls = (_aann.GroupedQueriesF32 if anns[0].dtype == np.dtype(np.float32)
                   else _aann.GroupedQueriesF64)
    gq = grouped_cls.prepare([a._rust for a in anns], perms)
    offsets = np.asarray(gq.offsets)  # (n + 1,); cloud i -> [offsets[i], offsets[i+1])

    # Group the requested pairs by target neuron j.
    by_target = defaultdict(list)  # j -> list of (pair_index, query_index i)
    for pidx, (i, j) in enumerate(pairs):
        by_target[j].append((pidx, i))

    def process_target(j):
        # Descend the whole (already Morton-sorted) grouped query set against
        # target j once and finalize to per-cloud ORIGINAL point/index order in
        # Rust (GIL released). Then slice out the requested query clouds via the
        # shared offsets. (The full set includes the self-pair and any clouds not
        # paired with j; those blocks are simply not sliced.)
        d_flat, i_flat = anns[j]._rust.query_grouped(gq, perms[j], block, ub)
        d_flat = np.asarray(d_flat)
        i_flat = np.asarray(i_flat)
        for (pidx, i) in by_target[j]:
            a, b = offsets[i], offsets[i + 1]
            results[pidx] = (d_flat[a:b].copy(), i_flat[a:b].copy())

    targets = list(by_target)
    if workers <= 1 or len(targets) <= 1:
        for j in targets:
            process_target(j)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(process_target, targets))
    return results


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


def _finalize_results(d, i, gx_perm, gy_perm, n_y):
    """Map raw Rust search output back to the caller's point order.

    Rust already applies ``distance_upper_bound`` -- out-of-range results carry
    the miss marker ``(inf, n_y)`` -- so this only undoes the Morton reorders
    (marker-safely). ``gx_perm`` / ``gy_perm`` are the query/target permutations
    (or ``None``); ``n_y`` is the target point count / missing-neighbour marker.
    """
    # Undo a target reorder so indices refer to y's original order. Extend the
    # permutation with the miss marker (index n_y) mapping to itself: k>1 rows
    # carry it as padding, and a bounded k=1 search emits it too, so the lookup
    # must stay in bounds for both.
    if gy_perm is not None:
        perm_ext = np.append(gy_perm, n_y)
        i = perm_ext[i]
    # Undo a query reorder so rows are in x's original order.
    if gx_perm is not None:
        d_out = np.empty_like(d)
        i_out = np.empty_like(i)
        d_out[gx_perm] = d  # query results -> x's original order
        i_out[gx_perm] = i
        d, i = d_out, i_out

    return d, i


def _all_miss(n_query, k, n_target, dtype):
    """Result arrays for a query where every point misses: ``inf`` distances and
    the ``len(target)`` index marker (scipy's convention), shaped and typed to
    match :func:`_finalize_results` output.
    """
    if k == 1:
        shape = (n_query,)
    else:
        shape = (n_query, k)
    return (
        np.full(shape, np.inf, dtype=dtype),
        np.full(shape, n_target, dtype=np.uint64),
    )


def _box_box_min_dist2(amin, amax, bmin, bmax):
    """Squared minimum distance between two axis-aligned boxes (0 if they
    overlap). Computed in float64 so the (search-dtype) box corners -- which
    bound their clouds exactly -- give a separation with no extra rounding, and
    thus never prune a cloud that has a real neighbour in range.
    """
    amax = np.asarray(amax, dtype=np.float64)
    amin = np.asarray(amin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)
    bmin = np.asarray(bmin, dtype=np.float64)
    gap = np.maximum(np.maximum(bmin - amax, amin - bmax), 0.0)
    return float(gap @ gap)


def _box_box_short_circuit(x, target_bbox, ub, k, n_target, dtype):
    """All-miss result if the raw query cloud ``x`` is wholly beyond ``ub`` from
    the target box, else ``None`` (fall through to the normal search). Boxes are
    taken in the search dtype -- the same points the descent uses -- so the
    decision matches Rust's per-point pruning and can never drop a real hit.
    """
    if isinstance(x, _DELAUNAY_TYPES):
        pts = np.asarray(x.points)
    else:
        pts = np.asarray(x)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None  # malformed; let the normal path raise as usual
    pts = pts.astype(dtype, copy=False)
    if _box_box_min_dist2(pts.min(0), pts.max(0), target_bbox[0], target_bbox[1]) > ub * ub:
        return _all_miss(len(pts), k, n_target, dtype)
    return None


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
        a.indptr.astype(_CSR_DTYPE, copy=False),
        a.indices.astype(_CSR_DTYPE, copy=False),
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
            np.asarray(indptr).astype(_CSR_DTYPE, copy=False),
            np.asarray(indices).astype(_CSR_DTYPE, copy=False),
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
            indptr.astype(_CSR_DTYPE, copy=False),
            indices.astype(_CSR_DTYPE, copy=False),
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
    indptr = np.asarray(indptr, dtype=_CSR_DTYPE)
    indices = np.asarray(indices, dtype=_CSR_DTYPE)

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
        g.indptr.astype(_CSR_DTYPE, copy=False),
        g.indices.astype(_CSR_DTYPE, copy=False),
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
        g.indptr.astype(_CSR_DTYPE, copy=False),
        g.indices.astype(_CSR_DTYPE, copy=False),
    )


def _assert_cloud_3d(x):
    """Check if `x` is 3d point cloud."""
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2
    assert x.shape[1] == 3
