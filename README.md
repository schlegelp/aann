# [WIP] aann
**A**pproximate **a**ll **n**earest-**n**eighbor ("_aann_") search using neighborhood graphs. Implemented in Rust with Python bindings.

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

## TODOs
- [ ] generalize to N-dimensions (currently only 3D)
- [ ] implement k-nearest neighbors (currently only 1)
- [ ] use SIMD (singe instruction multiple data) for distance calculations
- [ ] test various neighborhood graphs
- [ ] see if we can immplement additional parameters (e.g. `distance_upper_bound` or maybe a `distance_lower_bound` if that's useful)
- [ ] implement alternative distance metrics (currently only Euclidean)
- [ ] benchmarks

## Usage

```python
import aann
import numpy as np

N = 10000
x = Delaunay(np.random.rand(N, 3))
y = Delaunay(np.random.rand(N, 3))

distances, indices = aann.all_nearest_neighbours(x, y)
```

## Build
1. `cd` into directory
2. Activate virtual environment: `source .venv/bin/activate`
3. Run `maturin build --release` to build a wheel or use `maturin develop` to compile and install in development mode

### SIMD
`aann` makes use of `core::simd` module which means:

1. You need to use the nightly build:

    ```bash
    # Install and update nightly
    rustup install nightly
    rustup update nightly
    # Make sure you are in project directory
    cd aann
    # Tell the project to use nightly
    rustup override set nightly
    ```
2. By default, only the oldest SIMD extension `ssse2` is enabled during compilation. It is very likely that your processor supports newer extensions such as `avx2` or even `avx512f`. To check what's supported run:
    ```bash
    $ cargo install cargo-simd-detect --force
    $ cargo simd-detect
    extension       width                   available       enabled
    sse2            128-bit/16-bytes        true            true
    avx2            256-bit/32-bytes        true            false
    avx512f         512-bit/64-bytes        true            false
    ```
    You can tell the compiler to use newer extensions by setting rust flags:
    ```bash
    # To activate a specific extension
    export RUSTFLAGS="-C target-feature=+avx2"

    # Alternatively to activate all available extensions
    export RUSTFLAGS="-C target-cpu=native"
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

