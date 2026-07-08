import aann
import shull
import time

import numpy as np
import pandas as pd

from scipy.spatial import Delaunay, KDTree


def test_random():
    N = 100000
    x = Delaunay(np.random.rand(N, 3))
    y = Delaunay(np.random.rand(N, 3))

    start_time = time.time()
    d, i = aann.AANN(y, reorder=False).query(x)
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
    index = aann.AANN(y)
    end_time = time.time()
    print(f"Generating Delaunay / index: {end_time - start_time:.2f} seconds")

    # Run one warm-up
    _ = index.query(x)

    start_time = time.time()
    d, i = index.query(x)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")

    assert d.shape == (N,)
    assert i.shape == (N,)

    assert np.all(i >= 0)
    assert np.all(i < N)


def test_dtype_f32():
    # Both float32 and float64 search paths are supported.
    N = 5000
    a = np.random.rand(N, 3)
    b = np.random.rand(N, 3)

    d32, i32 = aann.AANN(b, dtype="float32").query(a)
    d64, i64 = aann.AANN(b, dtype="float64").query(a)

    assert d32.dtype == np.float32
    assert d64.dtype == np.float64
    assert i32.shape == (N,) and i64.shape == (N,)
    assert np.all(i32 < N) and np.all(i32 >= 0)

    # On uniform data the two precisions should almost always agree.
    assert np.mean(i32 == i64) > 0.99

    # dtype is inferred from the target when not given:
    d_inf, _ = aann.AANN(b.astype(np.float32)).query(a)
    assert d_inf.dtype == np.float32  # float32 target -> float32 search
    d_def, _ = aann.AANN(b).query(a)
    assert d_def.dtype == np.float64  # float64 target -> float64 search


def test_knn_graph():
    # The k-NN graph option is exact-ish for a self-join / space-filling data:
    # every query point lies within the target cloud's distribution.
    N = 5000
    a = np.random.rand(N, 3)
    b = np.random.rand(N, 3)

    index = aann.AANN(b, graph="knn", graph_k=16, reorder=False)
    d, i = index.query(a, graph="knn", graph_k=16)

    assert d.shape == (N,)
    assert i.shape == (N,)
    assert np.all(i >= 0)
    assert np.all(i < N)

    # Compare against the exact nearest neighbours from a KDTree. On uniform
    # data the k-NN-graph descent should recover the vast majority exactly.
    _, i_true = KDTree(b).query(a)
    recall = np.mean(i == i_true)
    print(f"k-NN graph recall vs exact KDTree: {recall * 100:.1f}%")
    assert recall > 0.9


def test_delaunay_input():
    # A precomputed triangulation -- scipy OR shull -- is accepted directly as
    # the target and/or the query, interchangeably.
    N = 3000
    rng = np.random.default_rng(4)
    a = rng.random((N, 3))
    b = rng.random((N, 3))
    _, it = KDTree(b).query(a)

    variants = {
        "array": lambda p: p,
        "scipy": lambda p: Delaunay(p, qhull_options="QJ"),
        "shull": lambda p: shull.Delaunay(p),
        "shull3d": lambda p: shull.Delaunay3d(p),
    }
    for tname, tmake in variants.items():
        for qname, qmake in variants.items():
            d, i = aann.AANN(tmake(b), reorder=False).query(qmake(a))
            assert d.shape == (N,) and i.shape == (N,)
            recall = np.mean(i == it)
            assert recall > 0.99, f"target={tname} query={qname}: recall {recall}"

    # A scipy Delaunay always carries float64 points; a shull Delaunay keeps
    # whatever dtype it was built from, so the inferred search precision follows.
    assert aann.AANN(Delaunay(b.astype(np.float32))).dtype == np.float64
    assert aann.AANN(shull.Delaunay(b.astype(np.float32))).dtype == np.float32
    assert aann.AANN(shull.Delaunay(b.astype(np.float64))).dtype == np.float64

    # prepare_many / all_by_all accept shull triangulations too.
    anns = aann.prepare_many([shull.Delaunay(a), shull.Delaunay(b)], reorder=False)
    (d, i), = aann.all_by_all(anns, pairs=[(0, 1)])
    assert np.mean(i == it) > 0.99


def test_reorder():
    # reorder=True renumbers points for cache locality; results must be mapped
    # back to the original point order transparently.
    N = 5000
    a = np.random.rand(N, 3)
    b = np.random.rand(N, 3)
    _, i_true = KDTree(b).query(a)

    index = aann.AANN(b, reorder=True)
    d, i = index.query(a)
    assert d.shape == (N,) and i.shape == (N,)
    assert np.all(i < N) and np.all(i >= 0)
    assert np.mean(i == i_true) > 0.99  # recall survives the round-trip remap

    # reorder composes with float32
    df, i_f = aann.AANN(b, reorder=True, dtype="float32").query(a)
    assert df.dtype == np.float32
    assert np.mean(i_f == i_true) > 0.99

    # a reordered query operand (AANN) also round-trips through both perms
    _, iq = index.query(aann.AANN(a, reorder=True))
    assert np.mean(iq == i_true) > 0.99


def test_k_neighbors():
    # k>1 runs the best-first search and returns (N, k) scipy-style results.
    N = 5000
    rng = np.random.default_rng(3)
    a = rng.random((N, 3))
    b = rng.random((N, 3))
    K = 4
    index = aann.AANN(b, reorder=False)

    d1, i1 = index.query(a)  # k=1 default: (N,) arrays
    assert d1.shape == (N,) and i1.shape == (N,)

    d, i = index.query(a, k=K)
    assert d.shape == (N, K) and i.shape == (N, K)
    assert np.all(np.diff(d, axis=1) >= 0)  # rows sorted ascending
    assert np.all(i < N)  # no padding markers on a connected Delaunay graph

    # Recall vs the exact KDTree answer (set-based, per row).
    _, it = KDTree(b).query(a, k=K)
    recall = np.mean(i == it)  # rows sorted -> direct elementwise compare
    print(f"k={K} recall vs exact KDTree: {recall * 100:.2f}%")
    assert recall > 0.95

    # Invalid k
    for bad in (0, -1, 1.5, "knn"):
        try:
            index.query(a, k=bad)
            assert False, f"k={bad!r} should raise"
        except ValueError:
            pass

    # float32 path
    d32, i32 = aann.AANN(b, reorder=False, dtype="float32").query(a, k=K)
    assert d32.dtype == np.float32 and d32.shape == (N, K)
    assert np.mean(i32 == it) > 0.95

    # Composes with distance_upper_bound: elementwise 2-D filter.
    bound = float(np.median(d[:, -1]))
    db, ib = index.query(a, k=K, distance_upper_bound=bound)
    hit = d <= bound
    assert np.array_equal(db[hit], d[hit]) and np.array_equal(ib[hit], i[hit])
    assert np.all(np.isinf(db[~hit])) and np.all(ib[~hit] == N)

    # Composes with reorder=True: marker-safe remap on both sides.
    dr, ir = aann.AANN(b, reorder=True).query(a, k=K)
    assert np.mean(ir == it) > 0.95  # recall survives the round-trip remap
    assert np.all(np.diff(dr, axis=1) >= 0)

    # all_by_all forwards k (and ef).
    anns = aann.prepare_many([a, b], reorder=False)
    (dab, iab), = aann.all_by_all(anns, pairs=[(0, 1)], k=K)
    assert np.array_equal(dab, d) and np.array_equal(iab, i)

    # Larger ef must not lower recall.
    dhi, ihi = index.query(a, k=K, ef=64)
    assert np.mean(ihi == it) >= recall

    # k >= len(target): rows padded with inf / len(target).
    small = rng.random((6, 3))
    dp, ip = aann.AANN(small, reorder=False).query(a[:100], k=8)
    assert dp.shape == (100, 8) and ip.shape == (100, 8)
    assert np.all(np.isinf(dp[:, 6:])) and np.all(ip[:, 6:] == 6)
    assert np.all(ip[:, :6] < 6)  # the whole tiny cloud is found...
    assert np.all(np.isfinite(dp[:, :6]))  # ...with real distances


def test_distance_upper_bound():
    # The bound filters results scipy-style: misses get d = inf, i = len(target).
    N = 5000
    rng = np.random.default_rng(2)
    a = rng.random((N, 3))
    b = rng.random((N, 3))
    index = aann.AANN(b, reorder=False)

    d0, i0 = index.query(a)
    bound = float(np.median(d0))

    d, i = index.query(a, distance_upper_bound=bound)
    hit = d0 <= bound
    assert np.array_equal(d[hit], d0[hit])  # hits are untouched
    assert np.array_equal(i[hit], i0[hit])
    assert np.all(np.isinf(d[~hit]))
    assert np.all(i[~hit] == N)  # scipy-style missing marker
    assert i.dtype == i0.dtype  # marker does not force a dtype change

    # None and inf both disable the bound.
    for ub in (None, np.inf):
        d1, i1 = index.query(a, distance_upper_bound=ub)
        assert np.array_equal(d1, d0) and np.array_equal(i1, i0)

    # float32 path: inf survives the reduced precision.
    d32, i32 = aann.AANN(b, reorder=False, dtype="float32").query(
        a, distance_upper_bound=bound
    )
    assert d32.dtype == np.float32
    miss32 = i32 == N
    assert np.all(np.isinf(d32[miss32]))
    assert np.mean(miss32) > 0.3  # ~half the points miss a median bound

    # Composes with reorder=True: marker applied after the remap.
    dr, ir = aann.AANN(b, reorder=True).query(a, distance_upper_bound=bound)
    assert np.array_equal(dr[hit], d0[hit])
    assert np.array_equal(ir[hit], i0[hit])
    assert np.all(np.isinf(dr[~hit])) and np.all(ir[~hit] == N)

    # all_by_all forwards the bound and matches direct calls.
    anns = aann.prepare_many([a, b], reorder=False)
    (dab, iab), = aann.all_by_all(anns, pairs=[(0, 1)], distance_upper_bound=bound)
    assert np.array_equal(dab, d) and np.array_equal(iab, i)

    # Two well-separated clouds: a small bound misses everywhere, a large
    # bound hits everywhere.
    far = aann.AANN(b + 10.0, reorder=False)
    d_far, i_far = far.query(a, distance_upper_bound=1.0)
    assert np.all(np.isinf(d_far)) and np.all(i_far == N)
    d_near, i_near = far.query(a, distance_upper_bound=100.0)
    assert np.all(np.isfinite(d_near)) and np.all(i_near < N)


def test_all_by_all():
    # Parallel all-by-all (GIL released in Rust) must match serial exactly.
    rng = np.random.default_rng(0)
    clouds = [rng.random((int(rng.integers(800, 2500)), 3)) for _ in range(6)]
    anns = aann.prepare_many(clouds, reorder=False)
    pairs = [(i, j) for i in range(len(anns)) for j in range(len(anns)) if i != j]

    serial = aann.all_by_all(anns, pairs=pairs, workers=1)
    parallel = aann.all_by_all(anns, pairs=pairs, workers=4)

    assert len(serial) == len(parallel) == len(pairs)
    for (ds, isx), (dp, ipx) in zip(serial, parallel):
        assert np.array_equal(isx, ipx)
        assert np.allclose(ds, dp)

    # results are aligned with `pairs` and match a direct query
    (i, j), (d, idx) = pairs[3], parallel[3]
    d0, i0 = anns[j].query(anns[i])
    assert np.array_equal(idx, i0) and np.allclose(d, d0)


def test_all_by_all_rejects_non_aann():
    # all_by_all is AANN-only, and requires a single shared dtype.
    rng = np.random.default_rng(5)
    a = rng.random((500, 3))
    b = rng.random((600, 3))

    try:
        aann.all_by_all([a, b])  # raw arrays
        assert False, "should reject non-AANN operands"
    except TypeError:
        pass

    try:
        aann.all_by_all(
            [aann.AANN(a, dtype="float32"), aann.AANN(b, dtype="float64")]
        )
        assert False, "should reject mixed dtype"
    except ValueError:
        pass


def test_prepare_many():
    # Parallel build must produce indices identical to a serial build.
    rng = np.random.default_rng(1)
    clouds = [rng.random((int(rng.integers(800, 2500)), 3)) for _ in range(6)]

    serial = [aann.AANN(c, reorder=True) for c in clouds]
    parallel = aann.prepare_many(clouds, reorder=True, workers=4)
    assert len(parallel) == len(clouds)
    assert all(isinstance(t, aann.AANN) for t in parallel)

    q = clouds[0]
    for gs, gp in zip(serial, parallel):
        ds, is_ = gs.query(q)
        dp, ip = gp.query(q)
        assert np.array_equal(is_, ip)
        assert np.allclose(ds, dp)


def test_query_aann_operand():
    # An AANN passed as the query cloud (the zero-copy all-by-all path) matches
    # both a raw-array query and the exact KDTree.
    rng = np.random.default_rng(7)
    a = rng.random((3000, 3))
    b = rng.random((4000, 3))
    _, it = KDTree(b).query(a)
    _, itk = KDTree(b).query(a, k=4)

    index = aann.AANN(b, reorder=False)
    qa = aann.AANN(a, reorder=False)

    d_raw, i_raw = index.query(a)
    d_ann, i_ann = index.query(qa)
    # Same underlying query graph -> identical to the raw-array path.
    assert np.array_equal(i_raw, i_ann) and np.allclose(d_raw, d_ann)
    assert np.mean(i_ann == it) > 0.99

    dk, ik = index.query(qa, k=4)
    assert ik.shape == (3000, 4)
    assert np.mean(ik == itk) > 0.95

    # dtype mismatch between two AANNs is rejected.
    try:
        aann.AANN(b, dtype="float32").query(aann.AANN(a, dtype="float64"))
        assert False, "dtype mismatch should raise"
    except ValueError:
        pass


def test_build_once_query_many():
    # One index, several different query clouds; plus the data/n/repr surface.
    rng = np.random.default_rng(8)
    b = rng.random((4000, 3))
    index = aann.AANN(b)

    assert len(index) == 4000 and index.n == 4000
    assert repr(index).startswith("AANN(")
    assert np.allclose(index.data, b)  # .data echoes the target (original order)

    for seed in range(3):
        c = np.random.default_rng(seed).random((1500, 3))
        d, i = index.query(c)
        _, it = KDTree(b).query(c)
        assert np.mean(i == it) > 0.99


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


def test_duplicate_query_points():
    # Exact-duplicate query points are dropped from the shull triangulation and
    # would become isolated graph vertices; they must still get the correct
    # nearest neighbour (previously the k=1 shull path silently returned
    # (d=0, i=0) for them). Reconnecting them via shull's ``coplanar`` makes both
    # backends agree with the exact KDTree answer.
    N = 2000
    rng = np.random.default_rng(7)
    target = rng.random((N, 3))
    q = rng.random((200, 3))
    # A few exact duplicates, including two copies of one point and a copy of the
    # walk's seed vertex 0, to exercise the multi-row / representative-is-0 cases.
    q[10] = q[3]
    q[20] = q[3]
    q[4] = q[0]
    q = np.ascontiguousarray(q)

    d_true, i_true = KDTree(target).query(q, k=1)

    for backend in ("shull", "scipy"):
        for reorder in (False, True):
            d, i = aann.AANN(target, reorder=reorder).query(q, k=1, backend=backend)
            assert np.allclose(d, d_true), f"{backend} reorder={reorder} distances"
            assert np.array_equal(i, i_true), f"{backend} reorder={reorder} indices"

    # k>1 (best-first) path stays correct with duplicate query points too.
    K = 4
    d4_true, _ = KDTree(target).query(q, k=K)
    d4, i4 = aann.AANN(target, reorder=False).query(q, k=K, backend="shull")
    assert d4.shape == (len(q), K)
    assert np.allclose(d4, d4_true)

    # float32 search precision (the stitch runs on this branch too).
    d32, i32 = aann.AANN(target.astype(np.float32)).query(q, k=1, backend="shull")
    assert d32.dtype == np.float32
    assert np.array_equal(i32, i_true)


def test_duplicate_target_points():
    # Exact-duplicate *target* points are benign (the kept copy covers the same
    # location); nearest-neighbour distances must still match the exact answer,
    # and the returned index must be one of the coincident pair.
    N = 2000
    rng = np.random.default_rng(11)
    target = rng.random((N, 3))
    target[7] = target[2]  # coincident pair {2, 7}
    target = np.ascontiguousarray(target)
    q = rng.random((200, 3))

    d_true, _ = KDTree(target).query(q, k=1)
    d, i = aann.AANN(target).query(q, k=1, backend="shull")
    assert np.allclose(d, d_true)
    # Wherever the exact NN is the duplicated location, either index is valid.
    hit_dup = np.isin(i, [2, 7])
    assert np.all(np.isin(i[hit_dup], [2, 7]))


if __name__ == "__main__":
    # test_neurons()
    # test_neurons_kdtree()
    test_random()
