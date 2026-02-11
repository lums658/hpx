# Getting Started

Welcome to HPXPy! This section will help you get up and running quickly.

## Installation

### Prerequisites

- Python 3.9 or later
- HPX library (built and installed)
- NumPy

### Quick Install

```bash
pip install hpxpy
```

### Building from Source

HPXPy builds against an **installed** HPX library. You point CMake at the HPX
install prefix, then install hpxpy into your Python environment:

```bash
# Ensure Python has build deps
pip install numpy pybind11

# Build and install HPXPy
cd hpx/python
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/usr/local/hpx ..
cmake --build . -j8
cmake --install .

# Set HPX library path and test
source setup_env.sh
python -c "import hpxpy as hpx; hpx.init(); print('HPXPy works!'); hpx.finalize()"
```

See **[Building from Source](building)** for detailed instructions including:
- Building HPX itself
- HPXPy build and install
- GPU support (CUDA/SYCL)
- Troubleshooting

## Quickstart

```python
import hpxpy as hpx

# Always initialize the HPX runtime first
hpx.init()

# Create arrays
arr = hpx.arange(1000)
zeros = hpx.zeros((10, 10))
ones = hpx.ones(100)

# Parallel algorithms
total = hpx.sum(arr)
doubled = arr * 2  # Element-wise operations

# Convert to/from NumPy
import numpy as np
np_arr = arr.to_numpy()
hpx_arr = hpx.from_numpy(np_arr)

# Always finalize when done
hpx.finalize()
```

## Core Concepts

### HPX Runtime

HPXPy requires the HPX runtime to be initialized before use:

```python
hpx.init()      # Start HPX runtime
# ... use HPXPy ...
hpx.finalize()  # Clean shutdown
```

### Execution Policies

Control how algorithms execute:

- `hpx.seq` - Sequential execution
- `hpx.par` - Parallel execution
- `hpx.par_unseq` - Parallel + vectorized (default)

The default policy is `par_unseq`, configurable at build time via the
`HPXPY_DEFAULT_EXECUTION_POLICY` CMake option.

```python
# Explicit sequential execution for debugging
result = hpx.reduce(arr, policy=hpx.seq)
```

### Device Selection

Arrays can be created on different devices:

```python
cpu_arr = hpx.zeros(1000, device='cpu')   # CPU (default)
gpu_arr = hpx.zeros(1000, device='gpu')   # CUDA GPU
sycl_arr = hpx.zeros(1000, device='sycl') # SYCL GPU
auto_arr = hpx.zeros(1000, device='auto') # Best available
```

## Next Steps

- **[Tutorials](../tutorials/index)** - Step-by-step notebooks
- **[User Guide](../user_guide/index)** - Detailed feature documentation
- **[API Reference](../api/index)** - Complete API documentation
- **[Cloud Deployment](deployment)** - Docker, AWS, GCP deployment guides

## Related Documentation

- **[HPX Documentation](https://hpx-docs.stellar-group.org/)** - Full HPX C++ documentation
- **[HPX Tutorials](https://hpx-docs.stellar-group.org/latest/html/manual/getting_started.html)** - HPX getting started guide

```{toctree}
:maxdepth: 1
:hidden:

building
deployment
```
