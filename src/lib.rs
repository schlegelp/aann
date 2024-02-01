use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::prelude::Python;

/// A Delaunay triangulation.
///
/// Arguments:
///  points: (N, 3) Array of points in the triangulation
///  indices: (N+1,) Array of indices into `neighbors` for each vertex
///  neighbors: (M,) Array of indices into `points` for each vertex
///
/// To get the neighbors of point `i``, we need need use:
/// neighbors[indices[i]:indices[i+1]]
struct Delaunay {
    // The (N, 3) points in the triangulation
    points: Array2<f64>,
    // The indices of neighboring vertices of vertex k are
    // neighbors[indices[k]:indices[k+1]].
    indices: Array1<usize>,
    neighbors: Array1<usize>,
}

/// Find all nearest neighbors between two point clouds using Delaunay triangulations.
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
    let x = Delaunay {
        points: x_points.to_owned_array(),
        indices: x_indices.to_owned_array(),
        neighbors: x_neighbors.to_owned_array(),
    };
    let y = Delaunay {
        points: y_points.to_owned_array(),
        indices: y_indices.to_owned_array(),
        neighbors: y_neighbors.to_owned_array(),
    };

    // Prepare outputs
    let mut distances = Array1::zeros(x_indices.len() - 1);
    let mut indices = Array1::zeros(x_indices.len() - 1);

    // Start with the first vertex in `x`
    let vertex: usize = 0;

    // Find the nearest neighbor of the first vertex in `x` in `y`
    let mut d: f64;
    let mut ix: usize;
    (d, ix) = _find_nearest_neighbour(&y, &x.points.row(vertex), 0);
    distances[vertex] = d;
    indices[vertex] = ix;

    // Our stack will contain a vertex in `y` and a vertex in `x` that
    // we expect to be in the vicinity of `y`'s nearest neighbor
    let mut stack: Vec<(usize, usize)> = vec![];
    let mut visited = Array1::from_elem(x.indices.len(), false);
    for n in get_neighbours(&x, vertex) {
        stack.push((ix, n));
        visited[n] = true;
    }
    // let mut stack = vec![(ix, get_neighbours(&x, vertex))];
    let mut current_ix: usize;
    let mut n: usize;
    while stack.len() > 0 {
        // Pop the last element from the stack
        (current_ix, n) = stack.pop().unwrap();
        // For some reason we have to clone the current_ix here
        let new_ix = current_ix;

        // Find the nearest neighbor of the current vertex in `y`
        (d, ix) = _find_nearest_neighbour(&y, &x.points.row(n), new_ix);
        distances[n] = d;
        indices[n] = ix;

        for n in get_neighbours(&x, n) {
            if !visited[n] {
                visited[n] = true;
                stack.push((ix, n));
            }
        }

        //println!("Stack length: {}", stack.len());
    }

    // Turn the vectors into numpy arrays
    let distances_py: Py<PyArray1<f64>> = distances.into_pyarray(py).to_owned();
    let indices_py: Py<PyArray1<usize>> = indices.into_pyarray(py).to_owned();

    Ok((distances_py, indices_py))

}

/// Calculate the squared euclidean distance between two points.
fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    //(a - b).mapv(|x| x.powi(2)).sum()
    // For some reason this is faster than the above:
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
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
///  y: Delaunay triangulation of the points to search in
///  p: (N, 3) Point to find the nearest neighbor in `y` for
///  start: Index of a point in y where to start the search from
///
/// Returns:
///  d: The distance between p and the nearest neighbor
///  vertex: The index of the nearest neighbor
fn _find_nearest_neighbour(y: &Delaunay, p: &ArrayView1<f64>, start: usize) -> (f64, usize) {
    let mut vertex: usize = start;

    // Track which points we have already visited
    let mut visited = Array1::from_elem(y.indices.len(), false);
    visited[vertex] = true;

    // Get the distance between p and the starting point
    let mut d = euclidean_distance(&y.points.row(vertex), &p);

    let mut neighbors: Array1<usize>;
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
            if !visited[n] {
                d_new = euclidean_distance(&y.points.row(n), &p);
                // No matter what happens we can mark this vertex as visited
                visited[n] = true;
                // Check if this vertex is closer than the current vertex
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
fn get_neighbours(
    x: &Delaunay,
    vertex: usize
    ) -> Array1<usize> {
    let ix1 = x.indices[vertex];
    let ix2 = x.indices[vertex + 1];
    x.neighbors.slice(s![ix1..ix2]).to_owned()
    }

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_aann")]
fn aann(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(all_nearest_neighbours, m)?)?;
    Ok(())
}
