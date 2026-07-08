"""bench.py -- when does aann's prepared-index build pay off vs scipy's KDTree?

Uniform random 3D point clouds only. aann builds a *costly* neighbourhood-graph
index per cloud (a Delaunay triangulation) but then answers each query with a
fast warm-started graph descent; scipy's KDTree is the opposite -- cheap to build,
slower to query. So aann only wins once an index is *reused* enough times to
amortise its build. The all-by-all join is the canonical case: every cloud's
index is built once and reused across all the pairs it appears in.

The benchmark is single-threaded so it compares the algorithms, not the thread
pools (aann additionally releases the GIL and scales across cores, which only
widens its lead on a real all-by-all).

Run:
    python bench.py                              # N=5000 pts/cloud, float64
    python bench.py --dtype float32 --n-points 10000
"""
import argparse
import time

import numpy as np
from scipy.spatial import KDTree

import aann


def best(fn, reps):
    """Best-of-`reps` wall-clock seconds for calling `fn`."""
    t = float("inf")
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - s)
    return t


def uniform_clouds(n, npts, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.random((npts, 3)) for _ in range(n)]


def per_op_costs(npts, dtype, reps=7):
    """Build- and query-time for one cloud, for scipy and aann."""
    rng = np.random.default_rng(0)
    a, b = rng.random((npts, 3)), rng.random((npts, 3))
    kt = KDTree(b)
    ia, ib = aann.AANN(a, dtype=dtype), aann.AANN(b, dtype=dtype)
    return {
        "scipy_build": best(lambda: KDTree(b), reps),
        "scipy_query": best(lambda: kt.query(a, workers=1), reps),
        "aann_build": best(lambda: aann.AANN(b, dtype=dtype), reps),
        # both operands prepared -> pack-free descent (the all-by-all per-pair cost)
        "aann_query": best(lambda: ib.query(ia), reps),
    }


def all_by_all_times(clouds, dtype, reps=2):
    """(#pairs, (scipy_build, scipy_query), (aann_build, aann_query)) seconds."""
    n = len(clouds)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    trees = [KDTree(c) for c in clouds]

    def scipy_query():
        for i, j in pairs:
            trees[j].query(clouds[i], workers=1)

    idx = aann.prepare_many(clouds, dtype=dtype, workers=1)

    tb_s = best(lambda: [KDTree(c) for c in clouds], reps)
    tq_s = best(scipy_query, reps)
    tb_a = best(lambda: aann.prepare_many(clouds, dtype=dtype, workers=1), reps)
    tq_a = best(lambda: aann.all_by_all(idx, workers=1), reps)
    return len(pairs), (tb_s, tq_s), (tb_a, tq_a)


def recall(npts, dtype):
    """aann's k=1 recall vs the exact KDTree answer on uniform data."""
    rng = np.random.default_rng(123)
    a, b = rng.random((npts, 3)), rng.random((npts, 3))
    _, it = KDTree(b).query(a)
    _, ii = aann.AANN(b, dtype=dtype).query(a)
    return float(np.mean(ii == it))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-points", type=int, default=5000, help="points per cloud")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[2, 4, 6, 8, 10, 12, 16, 20],
                    help="cloud counts n to sweep for the all-by-all")
    args = ap.parse_args()
    N, dt = args.n_points, args.dtype

    print(f"aann vs scipy.spatial.KDTree  |  uniform 3D  |  N={N} pts/cloud  |  "
          f"dtype={dt}  |  single-thread\n")

    # -- Why: the two libraries trade build cost for query cost --------------
    c = per_op_costs(N, dt)
    ms = lambda s: f"{s * 1e3:8.2f}"
    print("Per-operation cost (best of 7):")
    print(f"  {'':16}{'build':>10}{'query':>10}")
    print(f"  {'scipy KDTree':16}{ms(c['scipy_build'])}{ms(c['scipy_query'])}  ms")
    print(f"  {'aann index':16}{ms(c['aann_build'])}{ms(c['aann_query'])}  ms")
    build_x = c["aann_build"] / c["scipy_build"]
    query_x = c["scipy_query"] / c["aann_query"]
    breakeven = (c["aann_build"] - c["scipy_build"]) / (c["scipy_query"] - c["aann_query"])
    print(f"  -> aann's index costs ~{build_x:.0f}x more to build but queries "
          f"~{query_x:.0f}x faster")
    print(f"  -> break-even at ~{breakeven:.0f} reuses of a prepared index\n")

    # -- The tipping point: all-by-all total time vs number of clouds --------
    print("All-by-all over n uniform clouds (total = build + all n*(n-1) queries):")
    print(f"  {'n':>3}{'pairs':>8}{'scipy_ms':>11}{'aann_ms':>10}"
          f"{'aann speedup':>14}  winner")
    crossover = None
    for n in args.sizes:
        clouds = uniform_clouds(n, N)
        npairs, (tb_s, tq_s), (tb_a, tq_a) = all_by_all_times(clouds, dt)
        tot_s, tot_a = tb_s + tq_s, tb_a + tq_a
        winner = "aann" if tot_a < tot_s else "scipy"
        if crossover is None and tot_a < tot_s:
            crossover = n
        print(f"  {n:>3}{npairs:>8}{tot_s * 1e3:>11.1f}{tot_a * 1e3:>10.1f}"
              f"{tot_s / tot_a:>13.2f}x  {winner}")

    print()
    if crossover is not None:
        print(f"Tipping point: aann's total wall-clock overtakes scipy at "
              f"n = {crossover} clouds -- i.e. once each index is reused across "
              f"enough pairs to clear the ~{breakeven:.0f}-reuse break-even above.")
    else:
        print("No crossover in the swept range -- scipy stays ahead here.")
    print(f"Recall (aann k=1 vs exact KDTree, uniform data): "
          f"{recall(N, dt) * 100:.2f}%")


if __name__ == "__main__":
    main()
