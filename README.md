# [WIP] aann
**A**pproximate **a**ll **n**earest-**n**eighbor search using Delaunay graphs.

Based on the paper ALL NEAREST NEIGHBOUR CALCULATION BASED ON
DELAUNAY GRAPHS, Soudani & Karami, 2018 (arXiv)
([link](https://arxiv.org/abs/1802.09594)).

### Problem

Given two point clouds `Q` and `P`, for each point `q` in `Q` find its nearest neighbor `p` among the points in `P`.

### Solution
1. Calculate Delaunay graphs for both point clouds.
2. Start with a random vertex `q` in `Q` and traverse `P` using an A* search to find its nearest neighbor `p`.
3. Move to a vertex adjacent to `q` and search `P` for its nearest neighbor using `p` as the start. Since we start the search where we have already established spatial proximity the A* search should finish quickly.
4. Rinse-repeat until we found nearest neighbors for all points in `Q`.

## Benchmarks
TODO

## Build
1. `cd` into directory
2. Activate virtual environment: `source .venv/bin/activate`
3. Run `maturin develop` (use `maturin build --release` to build wheel)

## Usage

```python
import aann
import numpy as np

N = 10000
x = Delaunay(np.random.rand(N, 3))
y = Delaunay(np.random.rand(N, 3))

distances, indices = aann.all_nearest_neighbours(x, y)
```

## Test
First make sure `pytest` and `pandas` are installed:
```
pip install pytest -U
```

Then run the test-suite like so:
```
pytest --verbose -s
```

Note that unless you compiled with `maturin develop --release` the timings will
be much slower (up to 10x) than in a release build.
