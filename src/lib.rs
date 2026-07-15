//! Approximate all-nearest-neighbour search over neighbourhood graphs.
//!
//! Given two 3D point clouds, each equipped with a neighbourhood graph in CSR
//! form (e.g. the vertex adjacency of a Delaunay triangulation, see
//! [`graph_from_simplices`]), this crate finds for every point of the query
//! cloud its (approximate) nearest neighbour(s) in the target cloud via a
//! warm-started greedy graph descent. Distances are SIMD-accelerated via the
//! [`wide`] crate, so the crate builds on stable Rust.
//!
//! The Python bindings live behind the non-default `python` cargo feature;
//! with default features this is a pure-Rust library.
//!
//! ```
//! use aann::{graph_from_simplices, PreparedF64};
//! use aann::ndarray::array;
//!
//! // Target cloud: the 4 corners of the unit tetrahedron, fully connected
//! // by a single Delaunay simplex.
//! let points = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
//! let (indptr, indices) = graph_from_simplices(array![[0u64, 1, 2, 3]].view(), 4);
//! let target = PreparedF64::new(points.view(), indptr.view(), indices.view());
//!
//! // Query cloud with its own neighbourhood graph (here: two points linked
//! // to each other).
//! let queries = array![[0.1, 0.0, 0.0], [0.9, 0.1, 0.0]];
//! let (qptr, qidx) = (array![0usize, 1, 2], array![1usize, 0]);
//! let (dists, idxs) = target.query(queries.view(), qptr.view(), qidx.view(), None);
//! assert_eq!(idxs.to_vec(), vec![0, 1]);
//! assert!((dists[0] - 0.1).abs() < 1e-12);
//! ```
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use wide::{f32x4, f64x4};
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// The version of `ndarray` this crate's API is built against, re-exported so
/// consumers don't have to pin a matching version themselves.
pub use ndarray;

/// Reusable per-thread scratch for the k=1 all-nearest-neighbour descent
/// (`search_into_f64` / `search_into_f32` and the `query_*_into` methods).
///
/// It holds only vertex indices and generation stamps -- nothing tied to the
/// coordinate type -- so one `Workspace` serves both the f32 and f64 pipelines.
/// Recycle one per worker thread across an all-by-all: together with
/// caller-owned output `Vec`s (see [`PreparedF64::query_prepared_into`]) it makes
/// a warm query allocation-free, avoiding the per-pair churn of allocating the
/// DFS stack and visited buffer on every call.
pub struct Workspace {
    /// DFS stack over the query graph: `(start-vertex-in-target, vertex-in-query)`.
    stack: Vec<(usize, usize)>,
    /// Generation stamps indexed by query vertex; `visited[v] == gen` marks `v`
    /// seen in the current search. Avoids re-zeroing a `bool` array per call
    /// (the same trick the k>1 `search_k_*` path already uses for its target
    /// visited set).
    visited: Vec<u32>,
    gen: u32,
}

impl Workspace {
    /// An empty workspace; its buffers grow to fit on first use.
    pub fn new() -> Self {
        Workspace { stack: Vec::new(), visited: Vec::new(), gen: 0 }
    }

    /// A workspace pre-sized for a query cloud of `n` points.
    pub fn with_capacity(n: usize) -> Self {
        Workspace { stack: Vec::with_capacity(n), visited: vec![0u32; n], gen: 0 }
    }

    /// Ready the workspace for a search over `n_x` query vertices and return the
    /// generation to stamp visited nodes with this call. O(1) amortised: the
    /// `visited` array is zero-filled only when it first grows past `n_x` or
    /// when the `u32` generation counter wraps (~every 4 billion searches).
    #[inline]
    fn begin(&mut self, n_x: usize) -> u32 {
        if self.visited.len() < n_x {
            self.visited.resize(n_x, 0);
        }
        self.stack.clear();
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.visited.fill(0);
            self.gen = 1;
        }
        self.gen
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a full nearest-neighbour search pipeline for one coordinate type.
///
/// The search is identical for `f32` and `f64` coordinates; only the scalar and
/// SIMD types differ. Using `f32` halves the bytes moved per (scattered)
/// neighbour lookup, which is the bottleneck of the descent, at the cost of some
/// precision. We therefore monomorphise both via this macro.
///
/// Parameters:
///  `$t`        scalar coordinate type (`f32` / `f64`)
///  `$simd`     4-lane SIMD vector for `$t` (`f32x4` / `f64x4`)
///  `$nbhd`     generated neighbourhood-graph struct name
///  `$dist`     generated squared-distance helper name
///  `$neigh`    generated neighbour-slice helper name
///  `$find`     generated single nearest-neighbour search name
///  `$search`   generated all-nearest-neighbours search name
///  `$search_into` generated buffer-reuse all-nearest-neighbours search name
///  `$hitem`    generated (distance, index) heap-item struct name
///  `$findk`    generated k>1 best-first search name
///  `$searchk`  generated k>1 all-nearest-neighbours search name
///  `$pack`     generated SIMD point-packing helper name
///  `$prepared` generated owned, pre-packed graph struct name
///  `$box_dist2` generated point-to-box squared-distance helper name
///  `$box_box_dist2` generated box-to-box squared-min-distance helper name
///  `$bbox_of`  generated point-cloud bounding-box helper name
///  `$resolve_prune` generated bound-resolution helper name
///  `$search_pruned` generated allocating bound-aware k=1 search name
macro_rules! impl_ann_for {
    ($t:ty, $simd:ty, $nbhd:ident, $dist:ident, $neigh:ident, $find:ident, $search:ident,
     $search_into:ident, $hitem:ident, $findk:ident, $searchk:ident, $pack:ident, $prepared:ident,
     $box_dist2:ident, $box_box_dist2:ident, $bbox_of:ident, $resolve_prune:ident, $search_pruned:ident,
     $search_blocked:ident, $grouped:ident) => {
        /// A neighborhood graph over `$t` coordinates.
        ///
        /// `indices`/`neighbors` are zero-copy CSR views into the caller's
        /// arrays: the neighbours of vertex `k` are
        /// `neighbors[indices[k]..indices[k+1]]`.
        /// `points_simd` holds points pre-packed as `[x, y, z, 0.0]` for SIMD
        /// distances (see `$pack`), as a `Cow`: freshly packed callers pass
        /// `Owned`, while a persistent `$prepared` passes `Borrowed` so it packs
        /// once at construction and reuses it across every query instead of
        /// repacking on each call.
        pub struct $nbhd<'a> {
            indices: ArrayView1<'a, usize>,
            neighbors: ArrayView1<'a, usize>,
            points_simd: Cow<'a, [$simd]>,
        }

        impl<'a> $nbhd<'a> {
            pub fn new(
                points_simd: Cow<'a, [$simd]>,
                indices: ArrayView1<'a, usize>,
                neighbors: ArrayView1<'a, usize>,
            ) -> $nbhd<'a> {
                $nbhd { indices, neighbors, points_simd }
            }
        }

        /// Pack an `(N, 3)` point array into SIMD `[x, y, z, 0.0]` lanes.
        pub fn $pack(points: ArrayView2<$t>) -> Vec<$simd> {
            let mut packed: Vec<$simd> = Vec::with_capacity(points.nrows());
            for p in points.outer_iter() {
                packed.push(<$simd>::from([p[0], p[1], p[2], 0.0]));
            }
            packed
        }

        /// Squared euclidean distance between two packed points.
        #[inline]
        pub fn $dist(a: &$simd, b: &$simd) -> $t {
            let diff = *a - *b;
            (diff * diff).reduce_add()
        }

        /// Squared euclidean distance from packed point `p` to the axis-aligned
        /// box `[bmin, bmax]` (0 when `p` lies inside). Because the box contains
        /// every target point, this is a lower bound on `p`'s distance to any of
        /// them, so `$box_dist2(..) > ub²` proves no target lies within `ub` --
        /// the geometric hook the greedy descent otherwise lacks. The packed pad
        /// lane is 0 in `p`, `bmin` and `bmax`, so it contributes nothing.
        #[inline]
        pub fn $box_dist2(bmin: &$simd, bmax: &$simd, p: &$simd) -> $t {
            let zero = <$simd>::from([0.0 as $t; 4]);
            let below = (*bmin - *p).max(zero); // lo - p where p < lo, else 0
            let above = (*p - *bmax).max(zero); // p - hi where p > hi, else 0
            let d = below + above;              // per axis at most one is nonzero
            (d * d).reduce_add()
        }

        /// Squared euclidean distance between two axis-aligned boxes (0 when they
        /// overlap). A lower bound on the distance between *any* point of box A
        /// and *any* point of box B, so `$box_box_dist2(..) > ub²` proves no
        /// pair is within `ub` -- lets a whole separated query cloud be skipped
        /// in one test.
        #[inline]
        pub fn $box_box_dist2(amin: &$simd, amax: &$simd, bmin: &$simd, bmax: &$simd) -> $t {
            let zero = <$simd>::from([0.0 as $t; 4]);
            // Per axis the gap is max(bmin - amax, amin - bmax, 0): positive only
            // when the boxes are disjoint on that axis, in the direction of the
            // gap; 0 when they overlap.
            let gap = (*bmin - *amax).max(*amin - *bmax).max(zero);
            (gap * gap).reduce_add()
        }

        /// Axis-aligned bounding box of a packed point cloud as `(min, max)`.
        /// `pts` must be non-empty. The pad lane is 0 in every packed point, so
        /// it stays 0 in both corners.
        #[inline]
        fn $bbox_of(pts: &[$simd]) -> ($simd, $simd) {
            let mut mn = pts[0];
            let mut mx = pts[0];
            for p in &pts[1..] {
                mn = mn.min(*p);
                mx = mx.max(*p);
            }
            (mn, mx)
        }

        /// Resolve a `distance_upper_bound` request into what the search walk
        /// needs, doing the geometry once per call. `prune` is
        /// `(target_min, target_max, bound, query_box?)`; the query box is passed
        /// in for prepared operands (so an all-by-all never re-folds it) and
        /// folded from `xpts` otherwise. Returns:
        ///  - `None`: no bound in effect.
        ///  - `Some(None)`: the whole query cloud is out of range -> all misses.
        ///  - `Some(Some((min, max, ub2, check_points)))`: the target box, the
        ///    squared bound, and whether the per-point box test can ever fire.
        ///    `check_points` is false when the query cloud sits within the target
        ///    box grown by the bound (e.g. two overlapping clouds): no point can
        ///    be ruled out by the box, so the walk skips that test and relies on
        ///    marking neighbours that come back beyond the bound.
        #[inline]
        fn $resolve_prune(
            prune: Option<($simd, $simd, $t, Option<($simd, $simd)>)>,
            xpts: &[$simd],
            n_x: usize,
        ) -> Option<Option<($simd, $simd, $t, bool)>> {
            let (tmin, tmax, u, qbox) = prune?;
            if n_x == 0 {
                return Some(None); // nothing to search
            }
            let ub2 = u * u;
            let (qmin, qmax) = match qbox {
                Some(b) => b,
                None => $bbox_of(xpts),
            };
            if $box_box_dist2(&qmin, &qmax, &tmin, &tmax) > ub2 {
                return Some(None); // whole cloud out of range
            }
            let (qmn, qmx) = (qmin.to_array(), qmax.to_array());
            let (tmn, tmx) = (tmin.to_array(), tmax.to_array());
            let check_points = (0..3).any(|a| qmn[a] < tmn[a] - u || qmx[a] > tmx[a] + u);
            Some(Some((tmin, tmax, ub2, check_points)))
        }

        /// Neighbour indices of `vertex`, as a zero-copy slice.
        #[inline(always)]
        fn $neigh<'a>(x: &$nbhd<'a>, vertex: usize) -> ArrayView1<'a, usize> {
            x.neighbors.slice_move(s![x.indices[vertex]..x.indices[vertex + 1]])
        }

        /// Approximate nearest neighbour of `p` among the points in `y`, via a
        /// greedy descent that starts at `start`. The descent only moves to a
        /// strictly-closer node, so the running distance is always the smallest
        /// seen and an already-examined node can never be re-selected -- hence no
        /// explicit "visited" set is needed.
        pub fn $find(y: &$nbhd, p: &$simd, start: usize) -> ($t, usize) {
            // Hoist the CSR/point access to raw slices once per descent, so the
            // inner loop indexes plain slices instead of rebuilding an ndarray
            // view (`$neigh`'s `slice_move(s![..])`) on every visited vertex.
            let pts: &[$simd] = &y.points_simd;
            let indptr: &[usize] = y.indices.as_slice().expect("contiguous CSR indptr");
            let neigh: &[usize] = y.neighbors.as_slice().expect("contiguous CSR neighbours");

            let mut vertex: usize = start;
            let mut d = $dist(&pts[vertex], p);

            loop {
                let mut vert_new = false;
                for &n in &neigh[indptr[vertex]..indptr[vertex + 1]] {
                    let d_new = $dist(&pts[n], p);
                    if d_new < d {
                        d = d_new;
                        vert_new = true;
                        vertex = n;
                    }
                }
                if !vert_new {
                    break;
                }
            }
            (d.sqrt(), vertex)
        }

        /// PROTOTYPE -- blocked/batched k=1 all-nearest-neighbour descent.
        ///
        /// Same exact greedy descent as `$find`, but query points are processed in
        /// contiguous blocks of `block` (they are in Morton order, so a block is
        /// spatially coherent). Within a block the descent runs in lockstep: each
        /// round, block members sharing the same current target vertex have that
        /// vertex's neighbour coordinates **gathered once** into a small local
        /// buffer and reused for every one of them -- turning the scattered
        /// `points_simd[n]` fetches (aann's memory bottleneck) into one gather per
        /// (vertex, round) instead of one per (vertex, query point).
        ///
        /// `prune = Some((tmin, tmax, ub, qbox?))` applies a `distance_upper_bound`
        /// exactly as `$search_into` does (whole-cloud short-circuit, per-point box
        /// prune, after-descent miss marking `(inf, |y|)`); `None` is unbounded.
        /// Results are identical to the sequential path point-for-point (k=1 is exact
        /// on a Delaunay graph regardless of the descent's start vertex); only the
        /// *order* of gathers differs. Additive prototype -- `$search_into` untouched.
        pub fn $search_blocked(
            x: &$nbhd,
            y: &$nbhd,
            dists: &mut Vec<$t>,
            idx: &mut Vec<usize>,
            block: usize,
            prune: Option<($simd, $simd, $t, Option<($simd, $simd)>)>,
        ) {
            let n_x = x.indices.len() - 1;
            let n_y = y.points_simd.len();
            dists.resize(n_x, 0.0);
            idx.resize(n_x, 0);
            if n_x == 0 {
                return;
            }
            if n_y == 0 {
                dists.iter_mut().for_each(|d| *d = <$t>::INFINITY);
                idx.iter_mut().for_each(|i| *i = 0);
                return;
            }
            let xpts: &[$simd] = &x.points_simd;
            let ypts: &[$simd] = &y.points_simd;
            let yindptr: &[usize] = y.indices.as_slice().expect("contiguous CSR indptr");
            let yneigh: &[usize] = y.neighbors.as_slice().expect("contiguous CSR neighbours");

            // Resolve the bound once (see `$resolve_prune`): None -> unbounded;
            // Some(None) -> whole query cloud out of range (all misses);
            // Some(Some((bmin, bmax, ub2, check_points))) -> squared bound plus
            // whether the per-point box test can ever fire.
            let prune2 = match $resolve_prune(prune, xpts, n_x) {
                None => None,
                Some(None) => {
                    dists.iter_mut().for_each(|d| *d = <$t>::INFINITY);
                    idx.iter_mut().for_each(|i| *i = n_y);
                    return;
                }
                Some(Some(p)) => Some(p),
            };

            let block = block.max(1);
            // Per-block descent state (recycled across blocks). A pruned member is
            // flagged with the `usize::MAX` sentinel in `cur` and never descends.
            let mut cur: Vec<usize> = vec![0; block];
            let mut curd: Vec<$t> = vec![0.0; block];
            let mut done: Vec<bool> = vec![false; block];
            let mut order: Vec<usize> = Vec::with_capacity(block);
            // Gather buffer for one vertex's neighbour coordinates (reused).
            let mut coords: Vec<$simd> = Vec::with_capacity(32);
            let mut nbrs: Vec<usize> = Vec::with_capacity(32);

            let mut seed = 0usize; // warm start carried across blocks
            let mut b0 = 0usize;
            while b0 < n_x {
                let b1 = (b0 + block).min(n_x);
                let bs = b1 - b0;
                let mut remaining = 0usize;
                for k in 0..bs {
                    let gp = b0 + k;
                    // Per-point box prune: skip the descent for a query point beyond
                    // the bound from the whole target box (flag with the sentinel).
                    let pruned = match prune2 {
                        Some((bmin, bmax, ub2, true)) => $box_dist2(&bmin, &bmax, &xpts[gp]) > ub2,
                        _ => false,
                    };
                    if pruned {
                        cur[k] = usize::MAX;
                        done[k] = true;
                    } else {
                        cur[k] = seed;
                        curd[k] = $dist(&ypts[seed], &xpts[gp]);
                        done[k] = false;
                        remaining += 1;
                    }
                }
                while remaining > 0 {
                    // Group unconverged members by their current target vertex so
                    // members at the same vertex are handled together (one gather).
                    order.clear();
                    for k in 0..bs {
                        if !done[k] {
                            order.push(k);
                        }
                    }
                    order.sort_unstable_by_key(|&k| cur[k]);
                    let mut gi = 0;
                    while gi < order.len() {
                        let v = cur[order[gi]];
                        // Gather vertex v's neighbour coordinates ONCE.
                        coords.clear();
                        nbrs.clear();
                        for &nn in &yneigh[yindptr[v]..yindptr[v + 1]] {
                            coords.push(ypts[nn]);
                            nbrs.push(nn);
                        }
                        // Advance every block member currently at v from `coords`.
                        let mut gj = gi;
                        while gj < order.len() && cur[order[gj]] == v {
                            let k = order[gj];
                            let p = &xpts[b0 + k];
                            let mut best = curd[k];
                            let mut bestj = usize::MAX;
                            for (j, c) in coords.iter().enumerate() {
                                let dn = $dist(c, p);
                                if dn < best {
                                    best = dn;
                                    bestj = j;
                                }
                            }
                            if bestj != usize::MAX {
                                cur[k] = nbrs[bestj];
                                curd[k] = best;
                            } else {
                                done[k] = true;
                                remaining -= 1;
                            }
                            gj += 1;
                        }
                        gi = gj;
                    }
                }
                // Write results; mark misses (pruned, or NN beyond the bound) as
                // (inf, |y|). Advance the warm seed only onto a genuine in-range hit
                // (a miss keeps the previous seed, mirroring `$search_into`).
                for k in 0..bs {
                    let gp = b0 + k;
                    if cur[k] == usize::MAX {
                        dists[gp] = <$t>::INFINITY;
                        idx[gp] = n_y;
                        continue;
                    }
                    let miss = match prune2 {
                        Some((_, _, ub2, _)) => curd[k] > ub2,
                        None => false,
                    };
                    if miss {
                        dists[gp] = <$t>::INFINITY;
                        idx[gp] = n_y;
                    } else {
                        dists[gp] = curd[k].sqrt();
                        idx[gp] = cur[k];
                        seed = cur[k];
                    }
                }
                b0 = b1;
            }
        }

        /// Buffer-reuse form of `$search`: writes each query point's nearest
        /// neighbour into the caller-owned `dists`/`idx` (both resized to the
        /// query cloud size) and recycles the DFS scratch held in `ws`. With a
        /// warm `ws` and warm output buffers this performs no heap allocation --
        /// the hot path for an all-by-all inner loop (see
        /// `$prepared::query_prepared_into`). Behaviour is identical to
        /// `$search`; only the buffers' provenance differs.
        pub fn $search_into(
            x: &$nbhd,
            y: &$nbhd,
            ws: &mut Workspace,
            dists: &mut Vec<$t>,
            idx: &mut Vec<usize>,
            prune: Option<($simd, $simd, $t, Option<($simd, $simd)>)>,
        ) {
            let n_x = x.indices.len() - 1;
            let n_y = y.points_simd.len();
            // Exactly n_x slots: this grows (every slot is overwritten below, so
            // the fill value is irrelevant) or truncates a larger buffer from a
            // previous pair, so `&dists[..]` never exposes a stale tail.
            dists.resize(n_x, 0.0);
            idx.resize(n_x, 0);

            let xpts: &[$simd] = &x.points_simd;

            // Resolve `distance_upper_bound` into what the walk needs: the squared
            // bound plus `check_points` (whether the per-point box test can fire).
            // A wholly out-of-range cloud is reported as all-misses here without
            // touching the descent. `prune2 = (bmin, bmax, ub2, check_points)`.
            let prune2 = match $resolve_prune(prune, xpts, n_x) {
                None => None,
                Some(None) => {
                    dists.iter_mut().for_each(|d| *d = <$t>::INFINITY);
                    idx.iter_mut().for_each(|i| *i = n_y);
                    return;
                }
                Some(Some(p)) => Some(p),
            };

            // Depth-first walk of `x`'s graph, each query warm-starting from the
            // previous nearest neighbour. Stack holds (start-in-y, vertex-in-x).
            // Restart on any unvisited root so a disconnected query graph (e.g.
            // an isolated point) is still fully searched instead of left at the
            // zero-initialised default; `seed` carries a warm start across
            // components, mirroring the k>1 `$searchk`. `visited` is generation-
            // stamped so a reused workspace needs no per-call re-zeroing.
            let gen = ws.begin(n_x);
            let Workspace { stack, visited, .. } = ws;
            let mut seed: usize = 0;

            // Raw slices over `x`'s CSR adjacency (see `$find`).
            let xindptr: &[usize] = x.indices.as_slice().expect("contiguous CSR indptr");
            let xneigh: &[usize] = x.neighbors.as_slice().expect("contiguous CSR neighbours");

            // Two monomorphic walks so the common (unbounded / no-box-prune) path
            // carries zero per-point bound code -- the descent is only a few hops
            // on a warm cloud, so a per-point branch would be a large relative
            // cost. Only the straddle case (`check_points`) pays the inline box
            // test, which earns its keep by skipping whole descents.
            match prune2 {
                Some((bmin, bmax, ub2, true)) => {
                    for root in 0..n_x {
                        if visited[root] == gen {
                            continue;
                        }
                        visited[root] = gen;
                        stack.push((seed, root));

                        while let Some((start, v)) = stack.pop() {
                            // Skip the descent for a point beyond the bound from
                            // the whole target box; mark its neighbour a miss if
                            // it still comes back too far. A miss keeps `seed` and
                            // seeds children from `start`, so the warm start
                            // survives a run of pruned points.
                            let (d, ix) = if $box_dist2(&bmin, &bmax, &xpts[v]) > ub2 {
                                (<$t>::INFINITY, n_y)
                            } else {
                                let (d, ix) = $find(y, &xpts[v], start);
                                if d * d > ub2 { (<$t>::INFINITY, n_y) } else { (d, ix) }
                            };
                            dists[v] = d;
                            idx[v] = ix;
                            let child_start = if ix == n_y { start } else { seed = ix; ix };

                            for &m in &xneigh[xindptr[v]..xindptr[v + 1]] {
                                if visited[m] != gen {
                                    visited[m] = gen;
                                    stack.push((child_start, m));
                                }
                            }
                        }
                    }
                }
                _ => {
                    // Plain unbounded descent (identical to the no-bound case).
                    for root in 0..n_x {
                        if visited[root] == gen {
                            continue;
                        }
                        visited[root] = gen;
                        stack.push((seed, root));

                        while let Some((start, v)) = stack.pop() {
                            let (d, ix) = $find(y, &xpts[v], start);
                            dists[v] = d;
                            idx[v] = ix;
                            seed = ix;

                            for &m in &xneigh[xindptr[v]..xindptr[v + 1]] {
                                if visited[m] != gen {
                                    visited[m] = gen;
                                    stack.push((ix, m));
                                }
                            }
                        }
                    }
                    // check_points == false: nothing could be box-pruned, so the
                    // walk ran unbounded; mark neighbours beyond the bound in one
                    // cheap linear pass (sequential and branch-predictable, far
                    // cheaper than a test inside the pointer-chasing descent).
                    if let Some((_, _, ub2, false)) = prune2 {
                        for v in 0..n_x {
                            if dists[v] * dists[v] > ub2 {
                                dists[v] = <$t>::INFINITY;
                                idx[v] = n_y;
                            }
                        }
                    }
                }
            }
        }

        /// The full all-nearest-neighbours search over two graphs: for each
        /// point in `x`, find its nearest neighbour among the points in `y`.
        /// Allocates the output arrays and scratch fresh; for an all-by-all
        /// where buffers can be recycled, prefer `$search_into` /
        /// `$prepared::query_prepared_into`.
        pub fn $search(x: &$nbhd, y: &$nbhd) -> (Array1<$t>, Array1<usize>) {
            $search_pruned(x, y, None)
        }

        /// Bound-aware `$search`: `prune = Some((target_min, target_max, ub,
        /// query_box?))` reports the miss marker (inf, |y|) for every query point
        /// with no `y` neighbour within `ub`, using the target box to skip the
        /// descent where it provably can't help (see `$search_into` /
        /// `$resolve_prune`). `None` is plain `$search`.
        pub fn $search_pruned(
            x: &$nbhd,
            y: &$nbhd,
            prune: Option<($simd, $simd, $t, Option<($simd, $simd)>)>,
        ) -> (Array1<$t>, Array1<usize>) {
            let n_x = x.indices.len() - 1;
            let mut ws = Workspace::with_capacity(n_x);
            let mut dists: Vec<$t> = Vec::with_capacity(n_x);
            let mut idx: Vec<usize> = Vec::with_capacity(n_x);
            $search_into(x, y, &mut ws, &mut dists, &mut idx, prune);
            (Array1::from(dists), Array1::from(idx))
        }

        /// A (squared distance, index) pair with a total order on the distance
        /// (`total_cmp`), so it can live in a `BinaryHeap` despite the float key.
        #[derive(Clone, Copy)]
        struct $hitem {
            d: $t,
            ix: usize,
        }

        impl PartialEq for $hitem {
            fn eq(&self, other: &Self) -> bool {
                self.d.total_cmp(&other.d) == std::cmp::Ordering::Equal
            }
        }
        impl Eq for $hitem {}
        impl PartialOrd for $hitem {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for $hitem {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.d.total_cmp(&other.d)
            }
        }

        /// The k>1 analogue of the greedy descent: a best-first search over
        /// `y`'s graph (the HNSW search layer). Keeps the `ef` closest points
        /// seen in `res` (sorted ascending by squared distance) and stops once
        /// the closest unexpanded candidate is farther than the worst of them.
        ///
        /// Unlike the k=1 descent this revisits-checks nodes, so `visited`
        /// stamps nodes with the query's generation `gen` -- an O(1) reset per
        /// query (the array is zero-filled only when the u32 counter wraps).
        /// All buffers are caller-owned and reused across queries.
        fn $findk(
            y: &$nbhd,
            p: &$simd,
            start: usize,
            ef: usize,
            visited: &mut [u32],
            gen: u32,
            cand: &mut BinaryHeap<Reverse<$hitem>>,
            res: &mut Vec<$hitem>,
        ) {
            cand.clear();
            res.clear();

            let d0 = $dist(&y.points_simd[start], p);
            visited[start] = gen;
            cand.push(Reverse($hitem { d: d0, ix: start }));
            res.push($hitem { d: d0, ix: start });

            while let Some(Reverse(c)) = cand.pop() {
                if res.len() == ef && c.d > res[res.len() - 1].d {
                    break;
                }
                for n in $neigh(y, c.ix) {
                    let n = *n;
                    if visited[n] == gen {
                        continue;
                    }
                    visited[n] = gen;
                    let d = $dist(&y.points_simd[n], p);
                    if res.len() < ef || d < res[res.len() - 1].d {
                        cand.push(Reverse($hitem { d, ix: n }));
                        // Insertion into the small sorted buffer: for ef <~ 64
                        // this beats a second heap.
                        let pos = res.partition_point(|r| r.d <= d);
                        if res.len() == ef {
                            res.pop();
                        }
                        res.insert(pos, $hitem { d, ix: n });
                    }
                }
            }
        }

        /// The k>1 all-nearest-neighbours search: the same DFS walk and
        /// warm-start chaining as the k=1 `$search`, but each query runs the
        /// best-first `$findk` and is seeded with the previous query's best
        /// hit. Rows are (inf, |y|)-padded when the search exhausts with fewer
        /// than `k` results (disconnected graph or k > |y|); the outer loop
        /// restarts the walk on every unvisited `x` vertex, so disconnected
        /// query graphs are covered too.
        pub fn $searchk(
            x: &$nbhd,
            y: &$nbhd,
            k: usize,
            ef: usize,
            prune: Option<($simd, $simd, $t, Option<($simd, $simd)>)>,
        ) -> (Array2<$t>, Array2<usize>) {
            let n_x = x.indices.len() - 1;
            let n_y = y.points_simd.len();
            let mut distances = Array2::<$t>::from_elem((n_x, k), <$t>::INFINITY);
            let mut indices = Array2::<usize>::from_elem((n_x, k), n_y);

            // `distance_upper_bound` (see `$search_into` / `$resolve_prune`).
            // Rows already hold the miss marker (inf, |y|), so pruning leaves them
            // untouched. Heap items carry *squared* distances, so we compare
            // against `ub2` directly and only `sqrt` the ones we keep.
            let prune2 = match $resolve_prune(prune, &x.points_simd, n_x) {
                None => None,
                Some(None) => return (distances, indices), // whole cloud out of range
                Some(Some(p)) => Some(p),
            };

            // Per-search scratch, reused across all queries.
            let mut visited: Vec<u32> = vec![0; n_y];
            let mut gen: u32 = 0;
            let mut cand: BinaryHeap<Reverse<$hitem>> = BinaryHeap::with_capacity(4 * ef);
            let mut res: Vec<$hitem> = Vec::with_capacity(ef);

            let mut stack: Vec<(usize, usize)> = Vec::with_capacity(n_x);
            let mut visited_x = vec![false; n_x];
            // Warm start for the next query: the best hit of the previous one
            // (also carried across components as a "somewhere reasonable" seed).
            let mut seed: usize = 0;

            for root in 0..n_x {
                if visited_x[root] {
                    continue;
                }
                visited_x[root] = true;
                stack.push((seed, root));

                while let Some((start, v)) = stack.pop() {
                    // Whole target box beyond the bound (only possible when
                    // `check_points`): leave the row a miss and keep the warm
                    // `seed` (do not advance it off a pruned point).
                    if let Some((bmin, bmax, ub2, check_points)) = prune2 {
                        if check_points && $box_dist2(&bmin, &bmax, &x.points_simd[v]) > ub2 {
                            for n2 in $neigh(x, v) {
                                let m = *n2;
                                if !visited_x[m] {
                                    visited_x[m] = true;
                                    stack.push((seed, m));
                                }
                            }
                            continue;
                        }
                    }

                    gen = gen.wrapping_add(1);
                    if gen == 0 {
                        visited.fill(0);
                        gen = 1;
                    }
                    $findk(y, &x.points_simd[v], start, ef, &mut visited, gen, &mut cand, &mut res);
                    let mut hit = false;
                    for (j, item) in res.iter().take(k).enumerate() {
                        // Results are ascending, so once one exceeds the bound the
                        // rest do too -- leave those slots as the (inf, |y|) marker.
                        if let Some((_, _, ub2, _)) = prune2 {
                            if item.d > ub2 {
                                break;
                            }
                        }
                        distances[[v, j]] = item.d.sqrt();
                        indices[[v, j]] = item.ix;
                        hit = true;
                    }
                    if hit {
                        seed = res[0].ix; // advance the warm start only on a real hit
                    }

                    for n2 in $neigh(x, v) {
                        let m = *n2;
                        if !visited_x[m] {
                            visited_x[m] = true;
                            stack.push((seed, m));
                        }
                    }
                }
            }

            (distances, indices)
        }

        /// A prepared neighbourhood graph that **owns** its SIMD-packed points
        /// and CSR adjacency, so the packing is done once at construction and
        /// reused across every `query`. This is the persistent counterpart to
        /// the per-call `$search`: building `let tree = $prepared::new(...)`
        /// once and calling `tree.query(...)` many times (e.g. one target vs
        /// many query clouds, or an all-by-all) avoids repacking the target on
        /// each call.
        ///
        /// Note: the query points passed to `query`/`query_k` still carry their
        /// own neighbourhood graph -- the warm-started descent needs it -- so
        /// this is a cloud-vs-cloud tool, not a scattered-point KD-tree lookup.
        pub struct $prepared {
            points_simd: Vec<$simd>,
            indptr: Vec<usize>,
            indices: Vec<usize>,
            n: usize,
            /// Axis-aligned bounding box of the target points, computed once here
            /// so a bounded `query` can skip descents the box rules out (see
            /// `$search_into`). For an empty target it is `(inf, -inf)`, which
            /// prunes everything -- correct, since there are no neighbours.
            bbox_min: $simd,
            bbox_max: $simd,
        }

        impl $prepared {
            pub fn new(
                points: ArrayView2<$t>,
                indptr: ArrayView1<usize>,
                indices: ArrayView1<usize>,
            ) -> Self {
                let n = points.nrows();
                let points_simd = $pack(points);
                let (bbox_min, bbox_max) = if points_simd.is_empty() {
                    (
                        <$simd>::from([<$t>::INFINITY; 4]),
                        <$simd>::from([<$t>::NEG_INFINITY; 4]),
                    )
                } else {
                    $bbox_of(&points_simd)
                };
                $prepared {
                    points_simd,
                    indptr: indptr.to_vec(),
                    indices: indices.to_vec(),
                    n,
                    bbox_min,
                    bbox_max,
                }
            }

            /// Build the pruning tuple for a bounded search: the target box, the
            /// raw `distance_upper_bound`, and -- for a prepared query operand --
            /// its precomputed bounding box (`qbox`), so an all-by-all never
            /// re-folds it. `None` ub means an unbounded search.
            #[inline]
            fn prune(
                &self,
                ub: Option<$t>,
                qbox: Option<($simd, $simd)>,
            ) -> Option<($simd, $simd, $t, Option<($simd, $simd)>)> {
                ub.map(|u| (self.bbox_min, self.bbox_max, u, qbox))
            }

            /// Borrow this prepared graph as a `$nbhd` view (no packing, no
            /// copying -- the `Cow::Borrowed` fast path).
            pub fn as_graph(&self) -> $nbhd<'_> {
                $nbhd::new(
                    Cow::Borrowed(self.points_simd.as_slice()),
                    ArrayView1::from(self.indptr.as_slice()),
                    ArrayView1::from(self.indices.as_slice()),
                )
            }

            /// Nearest neighbour in the prepared target for each point of the
            /// query cloud `x` (given as its own CSR graph). Mirrors `$search`
            /// but reuses `self`'s pre-packed points. `ub = Some(d)` applies a
            /// `distance_upper_bound`: query points with no neighbour within `d`
            /// report the miss marker (inf, |target|), and the target box lets
            /// the search skip descents it rules out.
            pub fn query(
                &self,
                x_points: ArrayView2<$t>,
                x_indptr: ArrayView1<usize>,
                x_indices: ArrayView1<usize>,
                ub: Option<$t>,
            ) -> (Array1<$t>, Array1<usize>) {
                let x = $nbhd::new(Cow::Owned($pack(x_points)), x_indptr, x_indices);
                $search_pruned(&x, &self.as_graph(), self.prune(ub, None))
            }

            /// The k>1 variant of `query` (best-first search, `ef` breadth);
            /// `ub` applies a `distance_upper_bound` (see `query`).
            pub fn query_k(
                &self,
                x_points: ArrayView2<$t>,
                x_indptr: ArrayView1<usize>,
                x_indices: ArrayView1<usize>,
                k: usize,
                ef: usize,
                ub: Option<$t>,
            ) -> (Array2<$t>, Array2<usize>) {
                let x = $nbhd::new(Cow::Owned($pack(x_points)), x_indptr, x_indices);
                $searchk(&x, &self.as_graph(), k, ef, self.prune(ub, None))
            }

            /// Like `query`, but the query cloud is another prepared graph.
            /// Both operands are already SIMD-packed, so *nothing* is packed for
            /// this call (both views are `Cow::Borrowed`) -- this is the fast
            /// path for an all-by-all, where every graph is a persistent operand.
            /// `ub` applies a `distance_upper_bound` (see `query`).
            pub fn query_prepared(
                &self,
                other: &Self,
                ub: Option<$t>,
            ) -> (Array1<$t>, Array1<usize>) {
                let qbox = Some((other.bbox_min, other.bbox_max));
                $search_pruned(&other.as_graph(), &self.as_graph(), self.prune(ub, qbox))
            }

            /// The k>1 variant of `query_prepared`; `ub` applies a
            /// `distance_upper_bound` (see `query`).
            pub fn query_prepared_k(
                &self,
                other: &Self,
                k: usize,
                ef: usize,
                ub: Option<$t>,
            ) -> (Array2<$t>, Array2<usize>) {
                let qbox = Some((other.bbox_min, other.bbox_max));
                $searchk(&other.as_graph(), &self.as_graph(), k, ef, self.prune(ub, qbox))
            }

            /// Buffer-reuse form of `query_prepared`: both operands are already
            /// SIMD-packed, so with a warm `ws` and warm `dists`/`idx` this does
            /// no heap allocation -- the fast path for an all-by-all inner loop.
            /// `dists`/`idx` are resized to the query cloud (`other`) size; read
            /// `&dists[..]` / `&idx[..]` after the call. Results are identical to
            /// `query_prepared` (same operand order: `other` is the query, `self`
            /// the target).
            pub fn query_prepared_into(
                &self,
                other: &Self,
                ws: &mut Workspace,
                dists: &mut Vec<$t>,
                idx: &mut Vec<usize>,
                ub: Option<$t>,
            ) {
                let qbox = Some((other.bbox_min, other.bbox_max));
                $search_into(&other.as_graph(), &self.as_graph(), ws, dists, idx, self.prune(ub, qbox));
            }

            /// PROTOTYPE: blocked/batched counterpart to `query_prepared` (see
            /// `$search_blocked`). `ub` applies a `distance_upper_bound` just like
            /// `query_prepared`; results match it point-for-point. `block` is the
            /// query-point block size (gather-reuse granularity).
            pub fn query_prepared_blocked(
                &self,
                other: &Self,
                block: usize,
                ub: Option<$t>,
            ) -> (Array1<$t>, Array1<usize>) {
                let qbox = Some((other.bbox_min, other.bbox_max));
                let mut dists: Vec<$t> = Vec::new();
                let mut idx: Vec<usize> = Vec::new();
                $search_blocked(
                    &other.as_graph(), &self.as_graph(), &mut dists, &mut idx, block,
                    self.prune(ub, qbox),
                );
                (Array1::from(dists), Array1::from(idx))
            }

            /// Blocked descent of a prepared grouped query set (see `$grouped`)
            /// against this target. The concat + Morton sort were done ONCE in
            /// `$grouped::prepare` and are reused across every target, so the
            /// target-independent prep is never repeated per target -- the fix for
            /// the per-target re-sort that used to halve the grouped win.
            ///
            /// The whole grouped set is descended (rely on `ub`'s per-point box
            /// prune to keep far points cheap); the caller slices out the pairs it
            /// wants via `$grouped::cloud_span`. `ub` is as in `query_prepared`.
            ///
            /// `finalize = true` (aann's own use): un-sort Morton -> concatenated
            /// order, map to each cloud's original point order (via the perms stored
            /// in `gq`) and target indices to original (via `target_perm`); returns
            /// flat `(dists, idx)` in concatenated per-cloud ORIGINAL order with
            /// ORIGINAL target indices -- the caller only slices. `finalize = false`:
            /// return the raw results in `gq`'s Morton order with this target's
            /// INTERNAL indices; the caller maps them via `gq.perm()` (skips the
            /// per-target scatter -- useful when the consumer iterates points anyway).
            pub fn query_grouped(
                &self,
                gq: &$grouped,
                target_perm: Option<&[i64]>,
                block: usize,
                ub: Option<$t>,
                finalize: bool,
            ) -> (Array1<$t>, Array1<usize>) {
                let n_y = self.points_simd.len();
                let n = gq.sorted.len();
                if n == 0 {
                    return (Array1::from(Vec::<$t>::new()), Array1::from(Vec::<usize>::new()));
                }
                // Descend the (already Morton-sorted) grouped points once. Dummy
                // query CSR: the blocked descent walks points in order, never a
                // query adjacency. Borrow the sorted points (no per-target copy).
                let indptr: Vec<usize> = vec![0usize; n + 1];
                let neighbors: Vec<usize> = Vec::new();
                let x = $nbhd::new(
                    Cow::Borrowed(gq.sorted.as_slice()),
                    ArrayView1::from(indptr.as_slice()),
                    ArrayView1::from(neighbors.as_slice()),
                );
                let mut ds: Vec<$t> = Vec::new();
                let mut is_: Vec<usize> = Vec::new();
                $search_blocked(&x, &self.as_graph(), &mut ds, &mut is_, block, self.prune(ub, None));

                if !finalize {
                    // Raw Morton-order results with target-internal indices. The
                    // caller un-sorts via `gq.perm()` and maps target indices itself.
                    return (Array1::from(ds), Array1::from(is_));
                }

                // Un-sort Morton -> concatenated (per-cloud internal) order.
                let mut dist_int: Vec<$t> = vec![0.0 as $t; n];
                let mut idx_int: Vec<usize> = vec![0usize; n];
                for (k, &orig) in gq.perm.iter().enumerate() {
                    dist_int[orig] = ds[k];
                    idx_int[orig] = is_[k];
                }
                // Finalize: per cloud, scatter internal->original point order (query
                // side) and map target indices internal->original (mirrors the old
                // Python `_finalize_results`, run here with the GIL released).
                let mut dist_out: Vec<$t> = vec![0.0 as $t; n];
                let mut idx_out: Vec<usize> = vec![0usize; n];
                for c in 0..gq.offsets.len() - 1 {
                    let off = gq.offsets[c];
                    let nc = gq.offsets[c + 1] - off;
                    let qperm = gq.query_perms.get(c).and_then(|o| o.as_deref());
                    for k in 0..nc {
                        let src = off + k;
                        // k-th internal query point -> its original index within cloud c
                        let dst = off + match qperm {
                            Some(p) => p[k] as usize,
                            None => k,
                        };
                        dist_out[dst] = dist_int[src];
                        let ti = idx_int[src];
                        idx_out[dst] = match target_perm {
                            Some(tp) if ti < n_y => tp[ti] as usize,
                            Some(_) => n_y, // miss marker maps to itself
                            None => ti,
                        };
                    }
                }
                (Array1::from(dist_out), Array1::from(idx_out))
            }


            /// Buffer-reuse form of `query`: recycles the output buffers, the
            /// DFS scratch `ws`, and the SIMD-pack scratch `pack`. The query
            /// cloud must still be packed (into the reused `pack` buffer, which
            /// is a separate argument rather than part of `Workspace` because it
            /// is coordinate-type specific). Results are identical to `query`.
            #[allow(clippy::too_many_arguments)]
            pub fn query_into(
                &self,
                x_points: ArrayView2<$t>,
                x_indptr: ArrayView1<usize>,
                x_indices: ArrayView1<usize>,
                ws: &mut Workspace,
                pack: &mut Vec<$simd>,
                dists: &mut Vec<$t>,
                idx: &mut Vec<usize>,
                ub: Option<$t>,
            ) {
                pack.clear();
                pack.reserve(x_points.nrows());
                for p in x_points.outer_iter() {
                    pack.push(<$simd>::from([p[0], p[1], p[2], 0.0]));
                }
                let x = $nbhd::new(Cow::Borrowed(pack.as_slice()), x_indptr, x_indices);
                $search_into(&x, &self.as_graph(), ws, dists, idx, self.prune(ub, None));
            }

            /// Number of points in the prepared target.
            pub fn n(&self) -> usize {
                self.n
            }

            /// The target points as an `(n, 3)` array (pad lane dropped).
            pub fn points(&self) -> Array2<$t> {
                let mut arr = Array2::<$t>::zeros((self.n, 3));
                for (i, p) in self.points_simd.iter().enumerate() {
                    let a = p.to_array();
                    arr[[i, 0]] = a[0];
                    arr[[i, 1]] = a[1];
                    arr[[i, 2]] = a[2];
                }
                arr
            }
        }

        /// A reusable, Morton-sorted concatenation of many query clouds' points,
        /// built ONCE for a whole all-by-all and descended against each target with
        /// `$prepared::query_grouped`. The concat + Morton sort are target-
        /// independent, so hoisting them here (instead of redoing them per target)
        /// is what lets the grouped descent keep its ~2x single-thread / ~4x multi-
        /// thread gain rather than spending ~half its time re-sorting per target.
        ///
        /// Read-only after construction, so one instance can be shared across
        /// threads while the GIL is released.
        pub struct $grouped {
            /// Morton-sorted concatenated query points (across all clouds).
            sorted: Vec<$simd>,
            /// sorted position -> concatenated (pre-sort) index.
            perm: Vec<usize>,
            /// cloud `c` occupies `[offsets[c], offsets[c + 1])` in concatenated order.
            offsets: Vec<usize>,
            /// each cloud's new->original point permutation (or `None` if the cloud
            /// was not reordered); used by the `finalize` step of `query_grouped`.
            query_perms: Vec<Option<Vec<i64>>>,
        }

        impl $grouped {
            /// Concatenate every cloud's internal points and Morton-sort the whole
            /// set ONCE. `query_perms[c]` is cloud `c`'s new->original permutation
            /// (`None` = identity / not reordered); its length must match `queries`.
            /// Target-independent: build once per all-by-all, reuse for every target.
            pub fn prepare(queries: &[&$prepared], query_perms: &[Option<&[i64]>]) -> Self {
                let n_clouds = queries.len();
                let mut offsets: Vec<usize> = Vec::with_capacity(n_clouds + 1);
                offsets.push(0);
                let total: usize = queries.iter().map(|q| q.points_simd.len()).sum();
                let mut all: Vec<$simd> = Vec::with_capacity(total);
                for q in queries {
                    all.extend_from_slice(&q.points_simd);
                    offsets.push(all.len());
                }
                let n = all.len();
                let (sorted, perm) = if n == 0 {
                    (Vec::new(), Vec::new())
                } else {
                    // Morton-sort the combined points (interleaves the clouds
                    // spatially so co-located points from different clouds share
                    // a block, and thus one target-vertex gather).
                    let coords: Vec<[f64; 3]> = all
                        .iter()
                        .map(|p| {
                            let a = p.to_array();
                            [a[0] as f64, a[1] as f64, a[2] as f64]
                        })
                        .collect();
                    let perm = morton_perm(&coords);
                    let sorted: Vec<$simd> = perm.iter().map(|&i| all[i]).collect();
                    (sorted, perm)
                };
                let query_perms_owned: Vec<Option<Vec<i64>>> =
                    query_perms.iter().map(|o| o.map(|s| s.to_vec())).collect();
                $grouped { sorted, perm, offsets, query_perms: query_perms_owned }
            }

            /// Number of concatenated query points.
            pub fn n_points(&self) -> usize {
                self.sorted.len()
            }

            /// Number of query clouds.
            pub fn n_clouds(&self) -> usize {
                self.offsets.len() - 1
            }

            /// Cloud `c`'s span `[start, end)` in the concatenated per-cloud order
            /// that `query_grouped(finalize = true)` returns -- the caller slices
            /// each pair's result out with this.
            pub fn cloud_span(&self, c: usize) -> (usize, usize) {
                (self.offsets[c], self.offsets[c + 1])
            }

            /// The `(n_clouds + 1,)` cloud offsets: cloud `c`'s span is
            /// `[offsets[c], offsets[c + 1])`. Exposed so a caller can slice out
            /// each pair's block from the flat `query_grouped(finalize = true)`
            /// result without re-deriving the concatenation order.
            pub fn offsets(&self) -> &[usize] {
                &self.offsets
            }

            /// The sorted-position -> concatenated-index permutation, for a caller
            /// that uses `query_grouped(finalize = false)` and un-sorts itself.
            pub fn perm(&self) -> &[usize] {
                &self.perm
            }
        }
    };
}

impl_ann_for!(f64, f64x4, NeighborhoodF64, euclidean_distance_f64, get_neighbours_f64, find_nn_f64, search_f64,
              search_into_f64, HeapItemF64, find_knn_f64, search_k_f64, pack_points_f64, PreparedF64,
              box_dist2_f64, box_box_dist2_f64, bbox_of_f64, resolve_prune_f64, search_pruned_f64,
              search_blocked_f64, GroupedQueriesF64);
// f32 variant: half the memory traffic per point, some precision loss.
impl_ann_for!(f32, f32x4, NeighborhoodF32, euclidean_distance_f32, get_neighbours_f32, find_nn_f32, search_f32,
              search_into_f32, HeapItemF32, find_knn_f32, search_k_f32, pack_points_f32, PreparedF32,
              box_dist2_f32, box_box_dist2_f32, bbox_of_f32, resolve_prune_f32, search_pruned_f32,
              search_blocked_f32, GroupedQueriesF32);

/// Spread the low 21 bits of `v` to every 3rd bit (Morton/Z-order interleave for
/// one axis). Matches the bit magic in the Python `_morton_perm`, so the Rust and
/// Python spatial sorts agree (they need not, but consistency is convenient).
#[inline]
fn spread21(v: u64) -> u64 {
    let mut v = v & 0x1F_FFFF;
    v = (v | (v << 32)) & 0x1F00000000FFFF;
    v = (v | (v << 16)) & 0x1F0000FF0000FF;
    v = (v | (v << 8)) & 0x100F00F00F00F00F;
    v = (v | (v << 4)) & 0x10C30C30C30C30C3;
    v = (v | (v << 2)) & 0x1249249249249249;
    v
}

/// Argsort points into Morton (Z-order) so spatially-near points are adjacent.
/// Returns `perm` with `perm[k]` = original index of the k-th point in Z-order.
/// Used by the per-target grouped descent to interleave many query clouds so
/// co-located points (from different clouds) share a block -- and thus one gather.
pub fn morton_perm(pts: &[[f64; 3]]) -> Vec<usize> {
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    let mut mn = [f64::INFINITY; 3];
    let mut mx = [f64::NEG_INFINITY; 3];
    for p in pts {
        for a in 0..3 {
            mn[a] = mn[a].min(p[a]);
            mx[a] = mx[a].max(p[a]);
        }
    }
    let mut span = [0.0f64; 3];
    for a in 0..3 {
        span[a] = mx[a] - mn[a];
        if span[a] == 0.0 {
            span[a] = 1.0;
        }
    }
    let scale = ((1u64 << 21) - 1) as f64;
    let mut codes: Vec<(u64, usize)> = Vec::with_capacity(n);
    for (i, p) in pts.iter().enumerate() {
        let mut code = 0u64;
        for a in 0..3 {
            let q = ((((p[a] - mn[a]) / span[a]) * scale) as u64) & 0x1F_FFFF;
            code |= spread21(q) << a;
        }
        codes.push((code, i));
    }
    codes.sort_unstable_by_key(|&(c, _)| c);
    codes.into_iter().map(|(_, i)| i).collect()
}

/// Build a vertex-adjacency CSR graph from Delaunay tetrahedra.
///
/// Each tetrahedron contributes its 6 vertex pairs as undirected edges. Returns
/// `(indptr, indices)` where the neighbours of vertex `v` are
/// `indices[indptr[v]..indptr[v + 1]]` -- the same layout scipy's
/// `vertex_neighbor_vertices` produces, so it drops straight into the search.
///
/// Any vertex in `0..n_points` not referenced by a tetrahedron (e.g. an exact
/// duplicate point dropped by the triangulator) is left with no neighbours.
pub fn graph_from_simplices(
    simplices: ArrayView2<u64>,
    n_points: usize,
) -> (Array1<usize>, Array1<usize>) {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_points];
    for tet in simplices.outer_iter() {
        let v = [
            tet[0] as usize,
            tet[1] as usize,
            tet[2] as usize,
            tet[3] as usize,
        ];
        for a in 0..4 {
            for b in (a + 1)..4 {
                adj[v[a]].push(v[b]);
                adj[v[b]].push(v[a]);
            }
        }
    }

    let mut indptr: Vec<usize> = Vec::with_capacity(n_points + 1);
    indptr.push(0);
    let mut indices: Vec<usize> = Vec::new();
    for nb in adj.iter_mut() {
        nb.sort_unstable();
        nb.dedup();
        indices.extend_from_slice(nb);
        indptr.push(indices.len());
    }
    (Array1::from(indptr), Array1::from(indices))
}

#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The 4 corners of the unit tetrahedron plus their fully-connected CSR
    /// graph (from the single Delaunay simplex covering all of them).
    fn tetrahedron() -> (Array2<f64>, Array1<usize>, Array1<usize>) {
        let points = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let (indptr, indices) = graph_from_simplices(array![[0u64, 1, 2, 3]].view(), 4);
        (points, indptr, indices)
    }

    #[test]
    fn csr_from_simplices() {
        let (_, indptr, indices) = tetrahedron();
        // Fully connected: every vertex has the other 3 as sorted neighbours.
        assert_eq!(indptr.to_vec(), vec![0, 3, 6, 9, 12]);
        assert_eq!(
            indices.to_vec(),
            vec![1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]
        );
    }

    #[test]
    fn one_nn_unit_tetrahedron() {
        let (points, indptr, indices) = tetrahedron();
        // Queries: each corner nudged +0.01 in x, sharing the same CSR graph.
        let mut queries = points.clone();
        for mut row in queries.outer_iter_mut() {
            row[0] += 0.01;
        }

        let x = NeighborhoodF64::new(
            Cow::Owned(pack_points_f64(queries.view())),
            indptr.view(),
            indices.view(),
        );
        let y = NeighborhoodF64::new(
            Cow::Owned(pack_points_f64(points.view())),
            indptr.view(),
            indices.view(),
        );
        let (dists, idxs) = search_f64(&x, &y);
        assert_eq!(idxs.to_vec(), vec![0, 1, 2, 3]);
        for d in dists.iter() {
            assert!((d - 0.01).abs() < 1e-12);
        }

        // Same result through the owned/prepared API.
        let target = PreparedF64::new(points.view(), indptr.view(), indices.view());
        assert_eq!(target.n(), 4);
        let (dists2, idxs2) = target.query(queries.view(), indptr.view(), indices.view(), None);
        assert_eq!(idxs2.to_vec(), idxs.to_vec());
        assert_eq!(dists2.to_vec(), dists.to_vec());

        // And prepared-vs-prepared (the pack-free path).
        let qprep = PreparedF64::new(queries.view(), indptr.view(), indices.view());
        let (dists3, idxs3) = target.query_prepared(&qprep, None);
        assert_eq!(idxs3.to_vec(), idxs.to_vec());
        assert_eq!(dists3.to_vec(), dists.to_vec());
    }

    #[test]
    fn knn_sorted_and_padded() {
        let (points, indptr, indices) = tetrahedron();
        let query = array![[0.01, 0.0, 0.0]];
        // Single query point: trivial one-vertex graph with no neighbours.
        let (qptr, qidx) = (array![0usize, 0], Array1::<usize>::zeros(0));

        let target = PreparedF64::new(points.view(), indptr.view(), indices.view());
        let (dists, idxs) = target.query_k(query.view(), qptr.view(), qidx.view(), 2, 4, None);
        // Nearest = corner 0 (d=0.01), second = corner 1 (d=0.99), ascending.
        assert_eq!(idxs.row(0).to_vec(), vec![0, 1]);
        assert!((dists[[0, 0]] - 0.01).abs() < 1e-12);
        assert!((dists[[0, 1]] - 0.99).abs() < 1e-12);
        assert!(dists[[0, 0]] < dists[[0, 1]]);

        // k > |y|: missing entries are (inf, |y|)-padded.
        let (dists, idxs) = target.query_k(query.view(), qptr.view(), qidx.view(), 6, 8, None);
        assert_eq!(idxs.row(0).to_vec(), vec![0, 1, 2, 3, 4, 4]);
        assert!(dists[[0, 4]].is_infinite() && dists[[0, 5]].is_infinite());
    }

    #[test]
    fn f32_instantiation_matches() {
        let (points, indptr, indices) = tetrahedron();
        let points = points.mapv(|v| v as f32);
        let mut queries = points.clone();
        for mut row in queries.outer_iter_mut() {
            row[0] += 0.01;
        }

        let target = PreparedF32::new(points.view(), indptr.view(), indices.view());
        let qprep = PreparedF32::new(queries.view(), indptr.view(), indices.view());
        let (dists, idxs) = target.query_prepared(&qprep, None);
        assert_eq!(idxs.to_vec(), vec![0, 1, 2, 3]);
        for d in dists.iter() {
            assert!((d - 0.01).abs() < 1e-6);
        }
    }

    // ---- deterministic fixtures for the buffer-reuse + oracle tests --------

    /// A deterministic pseudo-random point cloud in the unit cube (LCG, so no
    /// `rand` dependency and fully reproducible across runs/platforms).
    fn lcg_cloud(n: usize, seed: u64) -> Array2<f64> {
        let mut s = seed | 1;
        let mut pts = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            for c in 0..3 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                pts[[i, c]] = (s >> 11) as f64 / (1u64 << 53) as f64;
            }
        }
        pts
    }

    /// Complete graph on `n` vertices as CSR (every vertex neighbours all
    /// others). A superset of any Delaunay graph, so the greedy descent is
    /// provably exact on it -- the dependency-free 100%-recall oracle graph.
    fn complete_graph_csr(n: usize) -> (Array1<usize>, Array1<usize>) {
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::with_capacity(n * n.saturating_sub(1));
        indptr.push(0);
        for v in 0..n {
            for u in 0..n {
                if u != v {
                    indices.push(u);
                }
            }
            indptr.push(indices.len());
        }
        (Array1::from(indptr), Array1::from(indices))
    }

    /// Sparse ring-lattice graph: each vertex links to +/-1, +/-2 (mod n).
    /// Forces multi-hop descents that exercise the warm-start chaining (needs
    /// `n >= 5` to stay simple/self-loop-free).
    fn ring_graph_csr(n: usize) -> (Array1<usize>, Array1<usize>) {
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        indptr.push(0);
        for v in 0..n {
            let mut nb = vec![
                (v + n - 2) % n,
                (v + n - 1) % n,
                (v + 1) % n,
                (v + 2) % n,
            ];
            nb.sort_unstable();
            nb.dedup();
            nb.retain(|&u| u != v);
            indices.extend_from_slice(&nb);
            indptr.push(indices.len());
        }
        (Array1::from(indptr), Array1::from(indices))
    }

    /// Exact nearest neighbour by O(n) scan: `(argmin index, sqrt distance)`.
    fn brute_nn(pts: &Array2<f64>, q: [f64; 3]) -> (usize, f64) {
        let mut best_ix = 0usize;
        let mut best_d2 = f64::INFINITY;
        for (i, row) in pts.outer_iter().enumerate() {
            let (dx, dy, dz) = (row[0] - q[0], row[1] - q[1], row[2] - q[2]);
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_d2 {
                best_d2 = d2;
                best_ix = i;
            }
        }
        (best_ix, best_d2.sqrt())
    }

    #[test]
    fn query_prepared_into_matches_query_prepared() {
        // (a) Tetrahedron: the buffer-reuse path is bit-for-bit identical to the
        // allocating `query_prepared`.
        let (points, indptr, indices) = tetrahedron();
        let mut queries = points.clone();
        for mut row in queries.outer_iter_mut() {
            row[0] += 0.01;
        }
        let target = PreparedF64::new(points.view(), indptr.view(), indices.view());
        let qprep = PreparedF64::new(queries.view(), indptr.view(), indices.view());
        let (d0, i0) = target.query_prepared(&qprep, None);

        let mut ws = Workspace::new();
        let (mut d, mut i) = (Vec::new(), Vec::new());
        target.query_prepared_into(&qprep, &mut ws, &mut d, &mut i, None);
        assert_eq!(d0.to_vec(), d);
        assert_eq!(i0.to_vec(), i);

        // (b) Larger clouds, reusing the SAME ws/buffers across a big-then-small
        // query: the small result must match its own fresh `query_prepared` with
        // no stale tail leaking from the bigger prior call.
        let pts_t = lcg_cloud(250, 1);
        let (tptr, tidx) = complete_graph_csr(250);
        let tgt = PreparedF64::new(pts_t.view(), tptr.view(), tidx.view());

        let pts_big = lcg_cloud(300, 2);
        let (bptr, bidx) = ring_graph_csr(300);
        let q_big = PreparedF64::new(pts_big.view(), bptr.view(), bidx.view());

        let pts_small = lcg_cloud(120, 3);
        let (sptr, sidx) = ring_graph_csr(120);
        let q_small = PreparedF64::new(pts_small.view(), sptr.view(), sidx.view());

        let (db0, ib0) = tgt.query_prepared(&q_big, None);
        let (ds0, is0) = tgt.query_prepared(&q_small, None);

        tgt.query_prepared_into(&q_big, &mut ws, &mut d, &mut i, None);
        assert_eq!(db0.to_vec(), d);
        assert_eq!(ib0.to_vec(), i);

        tgt.query_prepared_into(&q_small, &mut ws, &mut d, &mut i, None);
        assert_eq!(d.len(), 120);
        assert_eq!(i.len(), 120);
        assert_eq!(ds0.to_vec(), d);
        assert_eq!(is0.to_vec(), i);
    }

    #[test]
    fn query_prepared_into_matches_f32() {
        // The single `Workspace` type drives the f32 pipeline too.
        let (points, indptr, indices) = tetrahedron();
        let points = points.mapv(|v| v as f32);
        let mut queries = points.clone();
        for mut row in queries.outer_iter_mut() {
            row[0] += 0.01;
        }
        let target = PreparedF32::new(points.view(), indptr.view(), indices.view());
        let qprep = PreparedF32::new(queries.view(), indptr.view(), indices.view());
        let (d0, i0) = target.query_prepared(&qprep, None);

        let mut ws = Workspace::new();
        let (mut d, mut i) = (Vec::new(), Vec::new());
        target.query_prepared_into(&qprep, &mut ws, &mut d, &mut i, None);
        assert_eq!(d0.to_vec(), d);
        assert_eq!(i0.to_vec(), i);
    }

    #[test]
    fn query_into_matches_query() {
        let (points, indptr, indices) = tetrahedron();
        let mut queries = points.clone();
        for mut row in queries.outer_iter_mut() {
            row[0] += 0.01;
        }
        let target = PreparedF64::new(points.view(), indptr.view(), indices.view());
        let (d0, i0) = target.query(queries.view(), indptr.view(), indices.view(), None);

        let mut ws = Workspace::new();
        let (mut pack, mut d, mut i) = (Vec::new(), Vec::new(), Vec::new());
        target.query_into(
            queries.view(),
            indptr.view(),
            indices.view(),
            &mut ws,
            &mut pack,
            &mut d,
            &mut i,
            None,
        );
        assert_eq!(d0.to_vec(), d);
        assert_eq!(i0.to_vec(), i);
    }

    #[test]
    fn descent_is_exact_on_complete_graph() {
        // On a complete target graph the greedy descent examines every vertex
        // from any start, so it returns the exact nearest neighbour -- a
        // dependency-free 100%-recall check (complete graph is a superset of any
        // Delaunay graph, so this bounds the genuine-Delaunay recall from below).
        let pts_t = lcg_cloud(300, 10);
        let (tptr, tidx) = complete_graph_csr(300);
        let tgt = PreparedF64::new(pts_t.view(), tptr.view(), tidx.view());

        let pts_q = lcg_cloud(200, 20);
        let (qptr, qidx) = ring_graph_csr(200);
        let qry = PreparedF64::new(pts_q.view(), qptr.view(), qidx.view());

        // Fresh allocating path and buffer-reuse path must both be exact.
        let (d, idx) = tgt.query_prepared(&qry, None);
        let mut ws = Workspace::new();
        let (mut d2, mut i2) = (Vec::new(), Vec::new());
        tgt.query_prepared_into(&qry, &mut ws, &mut d2, &mut i2, None);
        assert_eq!(idx.to_vec(), i2);
        assert_eq!(d.to_vec(), d2);

        for v in 0..pts_q.nrows() {
            let q = [pts_q[[v, 0]], pts_q[[v, 1]], pts_q[[v, 2]]];
            let (truth_ix, truth_d) = brute_nn(&pts_t, q);
            assert_eq!(idx[v], truth_ix, "vertex {v}: descent picked a non-optimal NN");
            assert!((d[v] - truth_d).abs() < 1e-9, "vertex {v}: distance mismatch");
        }
    }

    #[test]
    fn bounded_prunes_separated_clouds() {
        // Target in the unit cube; query shifted far away (+10 per axis), so the
        // two bounding boxes are ~15.6 apart. A tight `distance_upper_bound`
        // reports every query point as a miss via the box-to-box short-circuit,
        // while a loose one recovers exactly the unbounded nearest neighbours.
        let pts_t = lcg_cloud(200, 7);
        let (tptr, tidx) = complete_graph_csr(200);
        let tgt = PreparedF64::new(pts_t.view(), tptr.view(), tidx.view());

        let mut pts_q = lcg_cloud(150, 8);
        pts_q += 10.0;
        let (qptr, qidx) = ring_graph_csr(150);
        let qry = PreparedF64::new(pts_q.view(), qptr.view(), qidx.view());

        // Tight bound: all misses (inf, |target|).
        let (d, idx) = tgt.query_prepared(&qry, Some(5.0));
        assert!(d.iter().all(|v| v.is_infinite()));
        assert!(idx.iter().all(|&v| v == tgt.n()));

        // Loose bound: bit-identical to the unbounded search.
        let (d0, i0) = tgt.query_prepared(&qry, None);
        let (db, ib) = tgt.query_prepared(&qry, Some(1000.0));
        assert_eq!(d0.to_vec(), db.to_vec());
        assert_eq!(i0.to_vec(), ib.to_vec());
    }

    #[test]
    fn bounded_matches_bruteforce() {
        // Complete target graph -> exact NN. A query cloud that straddles the
        // target box (spans [0,2] vs the target's [0,1]) with a mid-range bound:
        // the box-to-box short-circuit can't fire, `check_points` is true, so
        // this exercises *both* the per-point box prune (points far outside the
        // box) and the after-descent marking (in-box points whose NN is still
        // too far). The bounded result must equal brute force filtered by the
        // bound: hits keep their NN, everything else reports (inf, |target|).
        let pts_t = lcg_cloud(300, 11);
        let (tptr, tidx) = complete_graph_csr(300);
        let tgt = PreparedF64::new(pts_t.view(), tptr.view(), tidx.view());

        let mut pts_q = lcg_cloud(200, 12);
        pts_q *= 2.0; // spans [0,2], so it extends beyond the target box by > bound
        let (qptr, qidx) = ring_graph_csr(200);
        let qry = PreparedF64::new(pts_q.view(), qptr.view(), qidx.view());

        let bound = 0.1_f64;
        let (d, idx) = tgt.query_prepared(&qry, Some(bound));

        let (mut n_hits, mut n_miss) = (0usize, 0usize);
        for v in 0..pts_q.nrows() {
            let q = [pts_q[[v, 0]], pts_q[[v, 1]], pts_q[[v, 2]]];
            let (truth_ix, truth_d) = brute_nn(&pts_t, q);
            if truth_d <= bound {
                assert_eq!(idx[v], truth_ix, "vertex {v}: bounded hit picked wrong NN");
                assert!((d[v] - truth_d).abs() < 1e-9, "vertex {v}: distance mismatch");
                n_hits += 1;
            } else {
                assert!(d[v].is_infinite(), "vertex {v}: nearest is beyond the bound, expected a miss");
                assert_eq!(idx[v], tgt.n(), "vertex {v}: miss marker");
                n_miss += 1;
            }
        }
        // The bound should split the cloud, so both branches above are exercised.
        assert!(n_hits > 0 && n_miss > 0, "expected both hits and misses (got {n_hits} hits, {n_miss} misses)");
    }

    #[test]
    fn bounded_k_prunes_and_marks() {
        // The bound flows through the k>1 best-first path too.
        let pts_t = lcg_cloud(120, 21);
        let (tptr, tidx) = complete_graph_csr(120);
        let tgt = PreparedF64::new(pts_t.view(), tptr.view(), tidx.view());

        let mut pts_q = lcg_cloud(60, 22);
        pts_q += 10.0; // separated cloud
        let (qptr, qidx) = ring_graph_csr(60);
        let qry = PreparedF64::new(pts_q.view(), qptr.view(), qidx.view());

        let k = 3;
        // Tight bound: every row is fully (inf, |target|)-marked.
        let (d, idx) = tgt.query_prepared_k(&qry, k, k, Some(2.0));
        assert!(d.iter().all(|v| v.is_infinite()));
        assert!(idx.iter().all(|&v| v == tgt.n()));

        // Loose bound: bit-identical to the unbounded k>1 search.
        let (d0, i0) = tgt.query_prepared_k(&qry, k, k, None);
        let (db, ib) = tgt.query_prepared_k(&qry, k, k, Some(1000.0));
        assert_eq!(d0, db);
        assert_eq!(i0, ib);
    }
}
