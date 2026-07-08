#![feature(portable_simd)]

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use core::simd::prelude::{f32x4, f64x4};
use std::simd::num::SimdFloat;
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Generate a full nearest-neighbour search pipeline for one coordinate type.
///
/// The search is identical for `f32` and `f64` coordinates; only the scalar and
/// SIMD types differ. Using `f32` halves the bytes moved per (scattered)
/// neighbour lookup, which is the bottleneck of the descent, at the cost of some
/// precision. We therefore monomorphise both via this macro.
///
/// Parameters:
///  `$t`      scalar coordinate type (`f32` / `f64`)
///  `$simd`   4-lane SIMD vector for `$t` (`f32x4` / `f64x4`)
///  `$nbhd`   generated neighbourhood-graph struct name
///  `$dist`   generated squared-distance helper name
///  `$neigh`  generated neighbour-slice helper name
///  `$find`   generated single nearest-neighbour search name
///  `$pyfn`   generated `#[pyfunction]` name exposed to Python
///  `$hitem`  generated (distance, index) heap-item struct name
///  `$findk`  generated k>1 best-first search name
///  `$pyfnk`  generated k>1 `#[pyfunction]` name exposed to Python
macro_rules! impl_ann_for {
    ($t:ty, $simd:ty, $nbhd:ident, $dist:ident, $neigh:ident, $find:ident, $search:ident, $pyfn:ident,
     $hitem:ident, $findk:ident, $searchk:ident, $pyfnk:ident, $pack:ident, $prepared:ident) => {
        /// A neighborhood graph over `$t` coordinates.
        ///
        /// `indices`/`neighbors` are zero-copy CSR views into the numpy inputs:
        /// the neighbours of vertex `k` are `neighbors[indices[k]..indices[k+1]]`.
        /// `points_simd` holds points pre-packed as `[x, y, z, 0.0]` for SIMD
        /// distances (see `$pack`), as a `Cow`: freshly packed callers pass
        /// `Owned`, while a persistent `$prepared` passes `Borrowed` so it packs
        /// once at construction and reuses it across every query instead of
        /// repacking on each call.
        struct $nbhd<'a> {
            indices: ArrayView1<'a, usize>,
            neighbors: ArrayView1<'a, usize>,
            points_simd: Cow<'a, [$simd]>,
        }

        impl<'a> $nbhd<'a> {
            fn new(
                points_simd: Cow<'a, [$simd]>,
                indices: ArrayView1<'a, usize>,
                neighbors: ArrayView1<'a, usize>,
            ) -> $nbhd<'a> {
                $nbhd { indices, neighbors, points_simd }
            }
        }

        /// Pack an `(N, 3)` point array into SIMD `[x, y, z, 0.0]` lanes.
        fn $pack(points: ArrayView2<$t>) -> Vec<$simd> {
            let mut packed: Vec<$simd> = Vec::with_capacity(points.nrows());
            for p in points.outer_iter() {
                packed.push(<$simd>::from_array([p[0], p[1], p[2], 0.0]));
            }
            packed
        }

        /// Squared euclidean distance between two packed points.
        #[inline]
        fn $dist(a: &$simd, b: &$simd) -> $t {
            let diff = a - b;
            (diff * diff).reduce_sum()
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
        fn $find(y: &$nbhd, p: &$simd, start: usize) -> ($t, usize) {
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

        /// The full all-nearest-neighbours search over two graphs. Pure Rust
        /// (no Python), so it can run with the GIL released.
        fn $search(x: &$nbhd, y: &$nbhd) -> (Array1<$t>, Array1<usize>) {
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
        fn $searchk(x: &$nbhd, y: &$nbhd, k: usize, ef: usize) -> (Array2<$t>, Array2<usize>) {
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

        /// Find, for each point in `x`, the nearest neighbour among points in `y`
        /// (both given as neighbourhood graphs). The heavy search runs with the
        /// GIL released, so independent calls (e.g. an all-by-all join driven
        /// from a Python thread pool) execute in parallel across cores.
        #[pyfunction]
        fn $pyfn<'py>(
            py: Python<'py>,
            x_points: PyReadonlyArray2<$t>,
            x_indices: PyReadonlyArray1<usize>,
            x_neighbors: PyReadonlyArray1<usize>,
            y_points: PyReadonlyArray2<$t>,
            y_indices: PyReadonlyArray1<usize>,
            y_neighbors: PyReadonlyArray1<usize>,
        ) -> PyResult<(Bound<'py, PyArray1<$t>>, Bound<'py, PyArray1<usize>>)> {
            // Extract zero-copy views under the GIL (cheap), then release the GIL
            // for the whole search -- graph packing and descent are pure Rust.
            let xp = x_points.as_array();
            let xi = x_indices.as_array();
            let xn = x_neighbors.as_array();
            let yp = y_points.as_array();
            let yi = y_indices.as_array();
            let yn = y_neighbors.as_array();

            let (distances, indices) = py.detach(move || {
                let x = $nbhd::new(Cow::Owned($pack(xp)), xi, xn);
                let y = $nbhd::new(Cow::Owned($pack(yp)), yi, yn);
                $search(&x, &y)
            });

            Ok((distances.into_pyarray(py), indices.into_pyarray(py)))
        }

        /// The k>1 variant: for each point in `x` find the `k` nearest
        /// neighbours among the points in `y`, exploring `ef >= k` candidates
        /// per query (recall/speed trade-off). Returns (N, k) arrays with rows
        /// sorted ascending by distance; missing entries (search exhausted) are
        /// (inf, |y|). Runs with the GIL released, like the k=1 variant.
        #[pyfunction]
        fn $pyfnk<'py>(
            py: Python<'py>,
            x_points: PyReadonlyArray2<$t>,
            x_indices: PyReadonlyArray1<usize>,
            x_neighbors: PyReadonlyArray1<usize>,
            y_points: PyReadonlyArray2<$t>,
            y_indices: PyReadonlyArray1<usize>,
            y_neighbors: PyReadonlyArray1<usize>,
            k: usize,
            ef: usize,
        ) -> PyResult<(Bound<'py, PyArray2<$t>>, Bound<'py, PyArray2<usize>>)> {
            let xp = x_points.as_array();
            let xi = x_indices.as_array();
            let xn = x_neighbors.as_array();
            let yp = y_points.as_array();
            let yi = y_indices.as_array();
            let yn = y_neighbors.as_array();

            let (distances, indices) = py.detach(move || {
                let x = $nbhd::new(Cow::Owned($pack(xp)), xi, xn);
                let y = $nbhd::new(Cow::Owned($pack(yp)), yi, yn);
                $searchk(&x, &y, k, ef)
            });

            Ok((distances.into_pyarray(py), indices.into_pyarray(py)))
        }

        /// A prepared neighbourhood graph that **owns** its SIMD-packed points
        /// and CSR adjacency, so the packing is done once at construction and
        /// reused across every `query`. This is the persistent counterpart to
        /// the per-call `$pyfn`: building `tree = $prepared(...)` once and
        /// calling `tree.query(...)` many times (e.g. one target vs many query
        /// clouds, or an all-by-all) avoids repacking the target on each call.
        ///
        /// Note: the query points passed to `query`/`query_k` still carry their
        /// own neighbourhood graph -- the warm-started descent needs it -- so
        /// this is a cloud-vs-cloud tool, not a scattered-point KD-tree lookup.
        #[pyclass]
        struct $prepared {
            points_simd: Vec<$simd>,
            indptr: Vec<usize>,
            indices: Vec<usize>,
            n: usize,
        }

        #[pymethods]
        impl $prepared {
            #[new]
            fn new(
                points: PyReadonlyArray2<$t>,
                indptr: PyReadonlyArray1<usize>,
                indices: PyReadonlyArray1<usize>,
            ) -> Self {
                let pts = points.as_array();
                let n = pts.nrows();
                $prepared {
                    points_simd: $pack(pts),
                    indptr: indptr.as_array().to_vec(),
                    indices: indices.as_array().to_vec(),
                    n,
                }
            }

            /// Nearest neighbour in the prepared target for each point of the
            /// query cloud `x` (given as its own CSR graph). Mirrors `$pyfn`
            /// but reuses `self`'s pre-packed points. GIL released for the
            /// search; `self`'s owned buffers are `Send`.
            fn query<'py>(
                &self,
                py: Python<'py>,
                x_points: PyReadonlyArray2<$t>,
                x_indptr: PyReadonlyArray1<usize>,
                x_indices: PyReadonlyArray1<usize>,
            ) -> PyResult<(Bound<'py, PyArray1<$t>>, Bound<'py, PyArray1<usize>>)> {
                let xp = x_points.as_array();
                let xi = x_indptr.as_array();
                let xn = x_indices.as_array();

                let (distances, indices) = py.detach(move || {
                    let x = $nbhd::new(Cow::Owned($pack(xp)), xi, xn);
                    let y = $nbhd::new(
                        Cow::Borrowed(self.points_simd.as_slice()),
                        ArrayView1::from(self.indptr.as_slice()),
                        ArrayView1::from(self.indices.as_slice()),
                    );
                    $search(&x, &y)
                });

                Ok((distances.into_pyarray(py), indices.into_pyarray(py)))
            }

            /// The k>1 variant of `query` (best-first search, `ef` breadth).
            fn query_k<'py>(
                &self,
                py: Python<'py>,
                x_points: PyReadonlyArray2<$t>,
                x_indptr: PyReadonlyArray1<usize>,
                x_indices: PyReadonlyArray1<usize>,
                k: usize,
                ef: usize,
            ) -> PyResult<(Bound<'py, PyArray2<$t>>, Bound<'py, PyArray2<usize>>)> {
                let xp = x_points.as_array();
                let xi = x_indptr.as_array();
                let xn = x_indices.as_array();

                let (distances, indices) = py.detach(move || {
                    let x = $nbhd::new(Cow::Owned($pack(xp)), xi, xn);
                    let y = $nbhd::new(
                        Cow::Borrowed(self.points_simd.as_slice()),
                        ArrayView1::from(self.indptr.as_slice()),
                        ArrayView1::from(self.indices.as_slice()),
                    );
                    $searchk(&x, &y, k, ef)
                });

                Ok((distances.into_pyarray(py), indices.into_pyarray(py)))
            }

            /// Like `query`, but the query cloud is another prepared graph.
            /// Both operands are already SIMD-packed, so *nothing* is packed for
            /// this call (both views are `Cow::Borrowed`) -- this is the fast
            /// path for an all-by-all, where every graph is a persistent operand.
            fn query_prepared<'py>(
                &self,
                py: Python<'py>,
                other: PyRef<'py, $prepared>,
            ) -> PyResult<(Bound<'py, PyArray1<$t>>, Bound<'py, PyArray1<usize>>)> {
                let q: &$prepared = &other;
                let (distances, indices) = py.detach(move || {
                    let x = $nbhd::new(
                        Cow::Borrowed(q.points_simd.as_slice()),
                        ArrayView1::from(q.indptr.as_slice()),
                        ArrayView1::from(q.indices.as_slice()),
                    );
                    let y = $nbhd::new(
                        Cow::Borrowed(self.points_simd.as_slice()),
                        ArrayView1::from(self.indptr.as_slice()),
                        ArrayView1::from(self.indices.as_slice()),
                    );
                    $search(&x, &y)
                });

                Ok((distances.into_pyarray(py), indices.into_pyarray(py)))
            }

            /// The k>1 variant of `query_prepared`.
            fn query_prepared_k<'py>(
                &self,
                py: Python<'py>,
                other: PyRef<'py, $prepared>,
                k: usize,
                ef: usize,
            ) -> PyResult<(Bound<'py, PyArray2<$t>>, Bound<'py, PyArray2<usize>>)> {
                let q: &$prepared = &other;
                let (distances, indices) = py.detach(move || {
                    let x = $nbhd::new(
                        Cow::Borrowed(q.points_simd.as_slice()),
                        ArrayView1::from(q.indptr.as_slice()),
                        ArrayView1::from(q.indices.as_slice()),
                    );
                    let y = $nbhd::new(
                        Cow::Borrowed(self.points_simd.as_slice()),
                        ArrayView1::from(self.indptr.as_slice()),
                        ArrayView1::from(self.indices.as_slice()),
                    );
                    $searchk(&x, &y, k, ef)
                });

                Ok((distances.into_pyarray(py), indices.into_pyarray(py)))
            }

            /// Number of points in the prepared target.
            #[getter]
            fn n(&self) -> usize {
                self.n
            }

            /// The target points as an `(n, 3)` array (pad lane dropped).
            #[getter]
            fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<$t>> {
                let mut arr = Array2::<$t>::zeros((self.n, 3));
                for (i, p) in self.points_simd.iter().enumerate() {
                    let a = p.to_array();
                    arr[[i, 0]] = a[0];
                    arr[[i, 1]] = a[1];
                    arr[[i, 2]] = a[2];
                }
                arr.into_pyarray(py)
            }

            fn __repr__(&self) -> String {
                format!("{}(n={})", stringify!($prepared), self.n)
            }
        }
    };
}

// f64 keeps the original `all_nearest_neighbours` name (backward compatible).
impl_ann_for!(f64, f64x4, NeighboorhoodF64, euclidean_distance_f64, get_neighbours_f64, find_nn_f64, search_f64, all_nearest_neighbours,
              HeapItemF64, find_knn_f64, search_k_f64, all_nearest_neighbours_k, pack_points_f64, PreparedF64);
// f32 variant: half the memory traffic per point, some precision loss.
impl_ann_for!(f32, f32x4, NeighboorhoodF32, euclidean_distance_f32, get_neighbours_f32, find_nn_f32, search_f32, all_nearest_neighbours_f32,
              HeapItemF32, find_knn_f32, search_k_f32, all_nearest_neighbours_k_f32, pack_points_f32, PreparedF32);

/// Build a vertex-adjacency CSR graph from Delaunay tetrahedra.
///
/// Each tetrahedron contributes its 6 vertex pairs as undirected edges. Returns
/// `(indptr, indices)` where the neighbours of vertex `v` are
/// `indices[indptr[v]..indptr[v + 1]]` -- the same layout scipy's
/// `vertex_neighbor_vertices` produces, so it drops straight into the search.
///
/// Runs with the GIL released, so builds can be parallelised from Python. Any
/// vertex in `0..n_points` not referenced by a tetrahedron (e.g. an exact
/// duplicate point dropped by the triangulator) is left with no neighbours.
#[pyfunction]
fn graph_from_simplices<'py>(
    py: Python<'py>,
    simplices: PyReadonlyArray2<u64>,
    n_points: usize,
) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>)> {
    let s = simplices.as_array();

    let (indptr, indices) = py.detach(move || {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_points];
        for tet in s.outer_iter() {
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
    });

    Ok((indptr.into_pyarray(py), indices.into_pyarray(py)))
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_aann")]
fn aann(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(all_nearest_neighbours, m)?)?;
    m.add_function(wrap_pyfunction!(all_nearest_neighbours_f32, m)?)?;
    m.add_function(wrap_pyfunction!(all_nearest_neighbours_k, m)?)?;
    m.add_function(wrap_pyfunction!(all_nearest_neighbours_k_f32, m)?)?;
    m.add_function(wrap_pyfunction!(graph_from_simplices, m)?)?;
    m.add_class::<PreparedF64>()?;
    m.add_class::<PreparedF32>()?;
    Ok(())
}
