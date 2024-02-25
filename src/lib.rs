#![feature(portable_simd)]

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use core::simd::prelude::f64x4;
use std::simd::num::SimdFloat;



/// A neighborhood graph.
///
/// Arguments:
///  points: (N, 3) Array of points in the triangulation
///  indices: (N+1,) Array of indices into `neighbors` for each vertex
///  neighbors: (M,) Array of indices into `points` for each vertex
///
/// To get the neighbors of point `i``, we need need use:
/// neighbors[indices[i]:indices[i+1]]
struct Neighboorhood {
    // The (N, 3) points in the triangulation
    points: Array2<f64>,
    // The indices of neighboring vertices of vertex k are
    // neighbors[indices[k]:indices[k+1]].
    indices: Array1<usize>,
    neighbors: Array1<usize>,
    points_simd: Vec<f64x4>,
}

impl Neighboorhood {
    fn new(points: Array2<f64>, indices: Array1<usize>, neighbors: Array1<usize>) -> Neighboorhood {
        let mut points_simd: Vec<f64x4> = vec![];
        for p in points.outer_iter() {
            points_simd.push(f64x4::from_array([p[0], p[1], p[2], 0.0]));
        }

        Neighboorhood {points, indices, neighbors, points_simd}
    }
}

/// Find all nearest neighbors between two point clouds using neighboorhood graphs.
///
/// Arguments:
///  x_points: (N, 3) Array of points in the first point cloud
///  x_indices: (N+1,) Array of indices into `x_neighbors` for each vertex
///  x_neighbors: (M,) Array of indices into `x_points` for each vertex
///  y_points: (L, 3) Array of points in the second point cloud
///  y_indices: (L+1,) Array of indices into `y_neighbors` for each vertex
///  y_neighbors: (O,) Array of indices into `y_points` for each vertex
#[pyfunction]
fn all_nearest_neighbours<'py>(
    py: Python<'py>,
    x_points: PyReadonlyArray2<f64>,
    x_indices: PyReadonlyArray1<usize>,
    x_neighbors: PyReadonlyArray1<usize>,
    y_points: PyReadonlyArray2<f64>,
    y_indices: PyReadonlyArray1<usize>,
    y_neighbors: PyReadonlyArray1<usize>,
    ) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<usize>>)> {
    // Convert inputs to Delaunay structs
    let x = Neighboorhood::new(x_points.to_owned_array(), x_indices.to_owned_array(), x_neighbors.to_owned_array());
    let y = Neighboorhood::new(y_points.to_owned_array(), y_indices.to_owned_array(), y_neighbors.to_owned_array());

    // Prepare outputs
    let mut distances = Array1::zeros(x_indices.len() - 1);
    let mut indices = Array1::zeros(x_indices.len() - 1);

    // Start with the first vertex in `x`
    let vertex: usize = 0;

    // Find the nearest neighbor of the first vertex in `x` in `y`
    let mut d: f64;
    let mut ix: usize;
    let mut visited_y = vec![false; y.points.nrows()];
    (d, ix) = find_nearest_neighbour_simd(&y, &x.points_simd[vertex], 0, &mut visited_y);
    distances[vertex] = d;
    indices[vertex] = ix;

    // Our stack will contain a vertex in `y` and a vertex in `x` that
    // we expect to be in the vicinity of `y`'s nearest neighbor
    // Note: we know our stack will never be bigger than the number of
    // vertices in `x`. This avoids reallocations.
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(x.indices.len());
    let mut visited_x = Array1::from_elem(x.indices.len(), false);
    for n in get_neighbours(&x, vertex) {
        let n = n.clone();
        stack.push((ix, n));
        visited_x[n] = true;
    }

    let mut current_ix: usize;
    let mut n: usize;
    while stack.len() > 0 {
        // Pop the last element from the stack
        (current_ix, n) = stack.pop().unwrap();
        // For some reason we have to clone the current_ix here
        let new_ix = current_ix;

        // Find the nearest neighbor of the current vertex in `y`
        //(d, ix) = find_nearest_neighbour(&y, &x.points.row(n), new_ix, &mut visited_y);
        (d, ix) = find_nearest_neighbour_simd(&y, &x.points_simd[n], new_ix, &mut visited_y);
        distances[n] = d;
        indices[n] = ix;

        for n2 in get_neighbours(&x, n) {
            n = n2.clone();
            if !visited_x[n] {
                visited_x[n] = true;
                stack.push((ix, n));
            }
        }

        //println!("Stack length: {}", stack.len());
    }

    // Turn the vectors into numpy arrays
    Ok((distances.into_pyarray(py).to_owned(), indices.into_pyarray(py).to_owned()))

}

macro_rules! assert_equal_len {
    ($a:ident, $b: ident) => {
        assert!($a.len() == $b.len(),
                "add_assign: dimension mismatch: {:?} += {:?}",
                ($a.len(),),
                ($b.len(),));
    }
}

/// Calculate the squared euclidean distance between two points.
fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    assert_equal_len!(a, b);
    //(a - b).mapv(|x| x.powi(2)).sum()
    // For some reason this is faster than the above:
    let mut d: f64 = 0.0;
    for (aa, bb) in a.iter().zip(b.iter()) {
        d += (aa - bb).powi(2);
    }
    d
}

fn euclidean_distance_simd(a: &f64x4, b: &f64x4) -> f64 {
    let diff = a - b;
    (diff * diff).reduce_sum()
}

/// Calculate the squared euclidean distance between a point and a set of points.
fn euclidean_distance_to_point(p: &ArrayView1<f64>, points: &ArrayView2<f64>) -> Array1<f64> {
    let diff = points - p;
    let diff_squared = &diff * &diff;
    let distances_squared: Array1<f64> = diff_squared.sum_axis(Axis(1));
    //distances_squared.mapv(f64::sqrt)
    distances_squared

}

/// Find the approximate nearest neighbor of point p among the points in y.
///
/// Arguments:
///  y: Neighboorhood graph of the points to search in
///  p: (N, 3) Point to find the nearest neighbor in `y` for
///  start: Index of a point in y where to start the search from
///  visited: Array of bools to keep track of which points we have already visited
///
/// Returns:
///  d: The distance between p and the nearest neighbor
///  vertex: The index of the nearest neighbor
fn find_nearest_neighbour(y: &Neighboorhood, p: &ArrayView1<f64>, start: usize, visited: &mut Array1<bool>) -> (f64, usize) {
    let mut vertex: usize = start;

    // Track which points we have already been visited
    visited.fill(false); // reset the visited array
    visited[vertex] = true;

    // Get the distance between p and the starting point
    let mut d = euclidean_distance(&y.points.row(vertex), &p);

    let mut neighbors: ArrayView1<usize>;
    let mut vert_new: bool;
    let mut d_new: f64;
    let mut p2: ArrayView1<f64>;
    loop {
        // Get the neighbours of the current vertex
        neighbors = get_neighbours(&y, vertex);
        // let coordinates = y.points.select(Axis(0), &neighbors.to_vec());

        // Figure out if any of the neighbors is closer to p than the current vertex
        // If so, update the distance and the current vertex
        vert_new = false;
        for n in neighbors {
            let n = n.clone();  // This is necessary because `neighbors` is a unowned slice
            assert!(n < visited.len());
            if !visited[n] {
                // No matter what happens we can mark this vertex as visited
                visited[n] = true;

                // d_new = euclidean_distance(&y.points.row(n), &p);

                // Incrementally build the distance - that way we can stop early
                // if we know that the current vertex is farther away than the
                // closest vertex we have found so far
                p2 = y.points.row(n);
                d_new = (p[0] - p2[0]).powi(2);
                if d_new > d {
                    continue;
                }
                d_new += (p[1] - p2[1]).powi(2);
                if d_new > d {
                    continue;
                }
                d_new += (p[2] - p2[2]).powi(2);
                if d_new > d {
                    continue;
                }

                // If we got to here, we know that the current vertex is closer
                // than the closest vertex we have found so far
                d = d_new;
                vert_new = true;
                vertex = n;
            }
        }
        // If none of the neighbours is closer than the current vertex, we are done
        if !vert_new {
            break;
        }
        }
    (d.sqrt(), vertex)
    }

/// Find the approximate nearest neighbor of point p among the points in y.
///
/// This version uses SIMD to calculate the distances between the points.
///
/// Arguments:
///  y: Neighboorhood graph of the points to search in
///  p: (N, 3) Point to find the nearest neighbor in `y` for
///  start: Index of a point in y where to start the search from
///  visited: Array of bools to keep track of which points we have already visited
///
/// Returns:
///  d: The distance between p and the nearest neighbor
///  vertex: The index of the nearest neighbor
fn find_nearest_neighbour_simd(y: &Neighboorhood, p: &f64x4, start: usize, visited: &mut Vec<bool>) -> (f64, usize) {
    let mut vertex: usize = start;

    // Track which points have already been visited
    visited.fill(false); // reset the visited array
    visited[vertex] = true;

    // Get the distance between p and the starting point
    let mut d = euclidean_distance_simd(&y.points_simd[vertex], &p);

    let mut neighbors: ArrayView1<usize>;
    let mut vert_new: bool;
    let mut d_new: f64;
    loop {
        // Get the neighbours of the current vertex
        neighbors = get_neighbours(&y, vertex);
        // let coordinates = y.points.select(Axis(0), &neighbors.to_vec());

        // Figure out if any of the neighbors is closer to p than the current vertex
        // If so, update the distance and the current vertex
        vert_new = false;
        for n in neighbors {
            let n = n.clone();  // This is necessary because `neighbors` is a unowned slice
            assert!(n < visited.len());
            if !visited[n] {
                // No matter what happens we can mark this vertex as visited
                visited[n] = true;

                // Calculate the distance
                d_new = euclidean_distance_simd(&y.points_simd[n], &p);
                // Check if the new distance is smaller than the current distance
                if d_new < d {
                    d = d_new;
                    vert_new = true;
                    vertex = n;
                }
            }
        }
        // If none of the neighbours is closer than the current vertex, we are done
        if !vert_new {
            break;
        }
        }
    (d.sqrt(), vertex)
    }

/// Find the neighbours for `vertex` in delaunay triangulation `x`.
///
/// Arguments:
///  x: Delaunay triangulation
///  vertex: Index of the vertex to find the neighbours for
///
/// Returns:
///  Array of indices of the neighbours of `vertex`
#[inline(always)]
fn get_neighbours(
    x: &Neighboorhood,
    vertex: usize
    ) -> ArrayView1<usize> {
    x.neighbors.slice(s![x.indices[vertex]..x.indices[vertex + 1]])
    }

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_aann")]
fn aann(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(all_nearest_neighbours, m)?)?;
    Ok(())
}
