# TSNE-PSO

TSNE-PSO is a C/C++ implementation with Python bindings for dimensionality reduction.

## What it provides

- A C core library (`tsne_pso_core`)
- A Python extension module (`tsne_pso`)
- A minimal C++ wrapper header in `src/cpp/`

## Build

This project uses CMake. Python bindings are built via CMake as well.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Python

Install in editable mode (builds the extension):

```bash
python3 -m pip install -e .
```

Basic usage:

```python
from tsne_pso import TSNE_PSO
import numpy as np

X = np.random.randn(1000, 50)
model = TSNE_PSO(
    n_components=2,
    perplexity=30.0,
    n_particles=50,
    max_iter=250,
    random_state=42,
    n_jobs=4,
)
Y = model.fit_transform(X)
```

## Benchmarks

Run the local benchmark script:

```bash
python3 examples/benchmark.py --dataset digits --max-samples 1000 --models tsne_pso,sklearn_tsne,pca
```


