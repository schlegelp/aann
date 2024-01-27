use ndarray::{Array1, Array2, ArrayView1, s};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::prelude::Python;

struct Delaunay {
    points: Array2<f64>,
    indices: Array1<usize>,
    neighbors: Array1<usize>,
}

/// Find all nearest neighbors between two point clouds using Delaunay triangulations.
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
    let d: f64;
    let ix: usize;
    (d, ix) = _find_nearest_neighbour(&y, &x.points.row(vertex), 0);
    distances[vertex] = d;
    indices[vertex] = ix;

    // Our stack will contain a vertex in `y` and vertices in `x` that
    // we expect to be in the vicinity
    let mut stack = vec![(ix, get_neighbours(&x, vertex))];
    while stack.len() > 0 {
        // Pop the last element from the stack
        let (current_ix, neighbors) = stack.pop().unwrap().clone();
        // For some reason we have to clone the current_ix here
        let new_ix = current_ix.clone();
        for n in neighbors {
            // Skip if we already found the nearest neighbor for this vertex
            if distances[n] > 0.0 {
                continue;
            }
            // Find the nearest neighbor of the current vertex in `y`
            let (d, ix) = _find_nearest_neighbour(&y, &x.points.row(n), new_ix);
            distances[n] = d;
            indices[n] = ix;

            // Add the current vertex in `y` and the neighbours of the current vertex in `x`
            stack.push((ix, get_neighbours(&x, n)));
        }
        //println!("Stack length: {}", stack.len());
    }

    // Turn the vectors into numpy arrays
    let distances_py: Py<PyArray1<f64>> = distances.into_pyarray(py).to_owned();
    let indices_py: Py<PyArray1<usize>> = indices.into_pyarray(py).to_owned();

    Ok((distances_py, indices_py))

}

fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    // Note that we are using squared distances here
    (a - b).mapv(|x| x.powi(2)).sum()
}

/// Find the approximate nearest neighbor of point p among the points in y.
/// Arguments:
///   y: Delaunay triangulation of the points to search in
///   p: (N, 3) Point to search for
///   start: Index of the point in y to start the search from
fn _find_nearest_neighbour(y: &Delaunay, p: &ArrayView1<f64>, start: usize) -> (f64, usize) {
    let mut vertex: usize = start;

    // Track which points we have already visited
    let mut visited = Array1::from_elem(y.indices.len(), false);
    visited[vertex] = true;

    // Get the distance between p and the starting point
    let mut d = euclidean_distance(&y.points.row(vertex), &p);

    loop {
        // Get the neighbours of the current vertex
        let neighbors = get_neighbours(&y, vertex);

        // Figure out if any of the neighbors is closer to p than the current vertex
        // If so, update the distance and the current vertex
        let mut vert_new: Option<usize> = None;
        for n in neighbors {
            if !visited[n] {
                let d2 = euclidean_distance(&y.points.row(n), &p);
                // No matter what happens we can mark this vertex as visited
                visited[n] = true;
                // Check if this vertex is closer than the current vertex
                if d2 < d {
                    d = d2;
                    vert_new = Some(n);
                }
            }
        }
        // If none of the neighbours is closer than the current vertex, we are done
        if vert_new.is_none() {
            break;
        }
        // Otherwise, update the current vertex and loop again
        vertex = vert_new.unwrap();
        }
    (d.sqrt(), vertex)
    }

/// Find the neighbours for vertex in delaunay triangulation x.
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
