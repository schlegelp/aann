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
//! let (dists, idxs) = target.query(queries.view(), qptr.view(), qidx.view());
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
///  `$hitem`    generated (distance, index) heap-item struct name
///  `$findk`    generated k>1 best-first search name
///  `$searchk`  generated k>1 all-nearest-neighbours search name
///  `$pack`     generated SIMD point-packing helper name
///  `$prepared` generated owned, pre-packed graph struct name
macro_rules! impl_ann_for {
    ($t:ty, $simd:ty, $nbhd:ident, $dist:ident, $neigh:ident, $find:ident, $search:ident,
     $hitem:ident, $findk:ident, $searchk:ident, $pack:ident, $prepared:ident) => {
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
            let mut vertex: usize = start;
            let mut d = $dist(&y.points_simd[vertex], p);

            loop {
                let neighbors = $neigh(y, vertex);
                let mut vert_new = false;
                for n in neighbors {
                    let n = *n;
                    let d_new = $dist(&y.points_simd[n], p);
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

        /// The full all-nearest-neighbours search over two graphs: for each
        /// point in `x`, find its nearest neighbour among the points in `y`.
        pub fn $search(x: &$nbhd, y: &$nbhd) -> (Array1<$t>, Array1<usize>) {
            let n_x = x.indices.len() - 1;
            let mut distances = Array1::<$t>::zeros(n_x);
            let mut indices = Array1::<usize>::zeros(n_x);

            // Depth-first walk of `x`'s graph, each query warm-starting from the
            // previous nearest neighbour. Stack holds (start-in-y, vertex-in-x).
            // Restart on any unvisited root so a disconnected query graph (e.g.
            // an isolated point) is still fully searched instead of left at the
            // zero-initialised default; `seed` carries a warm start across
            // components, mirroring the k>1 `$searchk`.
            let mut stack: Vec<(usize, usize)> = Vec::with_capacity(n_x);
            let mut visited_x = vec![false; n_x];
            let mut seed: usize = 0;

            let mut d;
            let mut ix;
            for root in 0..n_x {
                if visited_x[root] {
                    continue;
                }
                visited_x[root] = true;
                stack.push((seed, root));

                while let Some((start, v)) = stack.pop() {
                    (d, ix) = $find(y, &x.points_simd[v], start);
                    distances[v] = d;
                    indices[v] = ix;
                    seed = ix;

                    for n2 in $neigh(x, v) {
                        let m = *n2;
                        if !visited_x[m] {
                            visited_x[m] = true;
                            stack.push((ix, m));
                        }
                    }
                }
            }

            (distances, indices)
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
        pub fn $searchk(x: &$nbhd, y: &$nbhd, k: usize, ef: usize) -> (Array2<$t>, Array2<usize>) {
            let n_x = x.indices.len() - 1;
            let n_y = y.points_simd.len();
            let mut distances = Array2::<$t>::from_elem((n_x, k), <$t>::INFINITY);
            let mut indices = Array2::<usize>::from_elem((n_x, k), n_y);

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
                    gen = gen.wrapping_add(1);
                    if gen == 0 {
                        visited.fill(0);
                        gen = 1;
                    }
                    $findk(y, &x.points_simd[v], start, ef, &mut visited, gen, &mut cand, &mut res);
                    for (j, item) in res.iter().take(k).enumerate() {
                        distances[[v, j]] = item.d.sqrt();
                        indices[[v, j]] = item.ix;
                    }
                    seed = res[0].ix;

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
        }

        impl $prepared {
            pub fn new(
                points: ArrayView2<$t>,
                indptr: ArrayView1<usize>,
                indices: ArrayView1<usize>,
            ) -> Self {
                let n = points.nrows();
                $prepared {
                    points_simd: $pack(points),
                    indptr: indptr.to_vec(),
                    indices: indices.to_vec(),
                    n,
                }
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
            /// but reuses `self`'s pre-packed points.
            pub fn query(
                &self,
                x_points: ArrayView2<$t>,
                x_indptr: ArrayView1<usize>,
                x_indices: ArrayView1<usize>,
            ) -> (Array1<$t>, Array1<usize>) {
                let x = $nbhd::new(Cow::Owned($pack(x_points)), x_indptr, x_indices);
                $search(&x, &self.as_graph())
            }

            /// The k>1 variant of `query` (best-first search, `ef` breadth).
            pub fn query_k(
                &self,
                x_points: ArrayView2<$t>,
                x_indptr: ArrayView1<usize>,
                x_indices: ArrayView1<usize>,
                k: usize,
                ef: usize,
            ) -> (Array2<$t>, Array2<usize>) {
                let x = $nbhd::new(Cow::Owned($pack(x_points)), x_indptr, x_indices);
                $searchk(&x, &self.as_graph(), k, ef)
            }

            /// Like `query`, but the query cloud is another prepared graph.
            /// Both operands are already SIMD-packed, so *nothing* is packed for
            /// this call (both views are `Cow::Borrowed`) -- this is the fast
            /// path for an all-by-all, where every graph is a persistent operand.
            pub fn query_prepared(&self, other: &Self) -> (Array1<$t>, Array1<usize>) {
                $search(&other.as_graph(), &self.as_graph())
            }

            /// The k>1 variant of `query_prepared`.
            pub fn query_prepared_k(
                &self,
                other: &Self,
                k: usize,
                ef: usize,
            ) -> (Array2<$t>, Array2<usize>) {
                $searchk(&other.as_graph(), &self.as_graph(), k, ef)
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
    };
}

impl_ann_for!(f64, f64x4, NeighborhoodF64, euclidean_distance_f64, get_neighbours_f64, find_nn_f64, search_f64,
              HeapItemF64, find_knn_f64, search_k_f64, pack_points_f64, PreparedF64);
// f32 variant: half the memory traffic per point, some precision loss.
impl_ann_for!(f32, f32x4, NeighborhoodF32, euclidean_distance_f32, get_neighbours_f32, find_nn_f32, search_f32,
              HeapItemF32, find_knn_f32, search_k_f32, pack_points_f32, PreparedF32);

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
        let (dists2, idxs2) = target.query(queries.view(), indptr.view(), indices.view());
        assert_eq!(idxs2.to_vec(), idxs.to_vec());
        assert_eq!(dists2.to_vec(), dists.to_vec());

        // And prepared-vs-prepared (the pack-free path).
        let qprep = PreparedF64::new(queries.view(), indptr.view(), indices.view());
        let (dists3, idxs3) = target.query_prepared(&qprep);
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
        let (dists, idxs) = target.query_k(query.view(), qptr.view(), qidx.view(), 2, 4);
        // Nearest = corner 0 (d=0.01), second = corner 1 (d=0.99), ascending.
        assert_eq!(idxs.row(0).to_vec(), vec![0, 1]);
        assert!((dists[[0, 0]] - 0.01).abs() < 1e-12);
        assert!((dists[[0, 1]] - 0.99).abs() < 1e-12);
        assert!(dists[[0, 0]] < dists[[0, 1]]);

        // k > |y|: missing entries are (inf, |y|)-padded.
        let (dists, idxs) = target.query_k(query.view(), qptr.view(), qidx.view(), 6, 8);
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
        let (dists, idxs) = target.query_prepared(&qprep);
        assert_eq!(idxs.to_vec(), vec![0, 1, 2, 3]);
        for d in dists.iter() {
            assert!((d - 0.01).abs() < 1e-6);
        }
    }
}
