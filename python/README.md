# HPXPy

High-performance distributed NumPy-like arrays powered by [HPX](https://hpx.stellar-group.org/).

## Overview

HPXPy provides a NumPy-compatible interface for parallel and distributed array computing using the HPX C++ runtime system. It enables Python users to leverage HPX's parallel algorithms and distributed computing capabilities with familiar NumPy syntax.

## Features

- **NumPy-compatible API**: Familiar array creation and manipulation functions
- **Parallel execution**: Automatic parallelization using HPX's execution policies
- **Distributed computing**: Scale from single node to clusters
- **Zero-copy interoperability**: Efficient data exchange with NumPy
- **GPU support**: Optional CUDA/SYCL acceleration

## Quick Start

```python
import hpxpy as hpx

# Initialize the HPX runtime
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

# Clean shutdown
hpx.finalize()
```

## Building from Source (Development)

HPXPy can be built and used directly from the HPX build directory without requiring HPX to be installed system-wide.

### Prerequisites

#### System Dependencies

| Dependency | Version | Required | Notes |
|------------|---------|----------|-------|
| CMake | >= 3.18 | Yes | Build system |
| C++ Compiler | GCC 9+, Clang 10+, Apple Clang 12+ | Yes | C++17 support required |
| Python | >= 3.9 | Yes | With development headers |
| Boost | >= 1.71 | Yes | Required by HPX |
| hwloc | any | Recommended | Hardware locality detection |

#### Python Dependencies

Install with `pip install -r requirements.txt`:

| Package | Version | Required | Purpose |
|---------|---------|----------|---------|
| numpy | >= 1.20.0 | Yes | Array operations, NumPy interop |
| pybind11 | >= 2.10.0 | Yes | C++/Python bindings |
| cmake | >= 3.18 | Yes | Python-accessible CMake |
| pytest | >= 7.0.0 | Testing | Running test suite |
| jupyter | >= 1.0.0 | Tutorials | Running notebooks |
| matplotlib | >= 3.5.0 | Tutorials | Visualization |
| scipy | >= 1.7.0 | Optional | Scientific examples |

#### GPU Dependencies (Optional)

| Backend | Requirements |
|---------|--------------|
| CUDA | CUDA Toolkit 11.0+, HPX built with `-DHPX_WITH_CUDA=ON` |
| SYCL | Intel oneAPI or compatible SYCL runtime, HPX built with `-DHPX_WITH_SYCL=ON` |

**macOS:**
```bash
brew install cmake boost hwloc python
pip install -r requirements.txt
```

**Ubuntu/Debian:**
```bash
sudo apt install cmake libboost-all-dev libhwloc-dev python3-dev python3-pip
pip install -r requirements.txt
```

### Build Instructions

```bash
# Clone and build HPX
git clone https://github.com/STEllAR-GROUP/hpx.git
cd hpx
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DHPX_WITH_FETCH_ASIO=ON -DHPX_WITH_EXAMPLES=OFF -DHPX_WITH_TESTS=OFF
make -j8

# Set up Python virtual environment (required for CMake to find Python/NumPy)
cd ../python
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pybind11

# Build HPXPy
mkdir build && cd build
cmake .. -DHPX_DIR=../../build/lib/cmake/HPX
make -j8

# Set up environment
cat > setup_env.sh << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPX_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate virtual environment if it exists
if [[ -f "$HPX_ROOT/python/.venv/bin/activate" ]]; then
    source "$HPX_ROOT/python/.venv/bin/activate"
fi

export HPX_BUILD="$HPX_ROOT/build"
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="$HPX_BUILD/lib:$DYLD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$HPX_BUILD/lib:$LD_LIBRARY_PATH"
fi
export PYTHONPATH="$SCRIPT_DIR:$HPX_ROOT/python:$PYTHONPATH"
echo "HPXPy environment configured"
EOF
chmod +x setup_env.sh

# Test
source setup_env.sh
python3 -c "import hpxpy as hpx; hpx.init(); print(hpx.sum(hpx.arange(100))); hpx.finalize()"
```

### GPU Support (Optional)

For CUDA support (requires HPX built with `HPX_WITH_CUDA=ON`):
```bash
cmake .. -DHPXPY_WITH_CUDA=ON -DHPX_DIR=../../build/lib/cmake/HPX
```

For SYCL support (requires HPX built with `HPX_WITH_SYCL=ON`):
```bash
cmake .. -DHPXPY_WITH_SYCL=ON -DHPX_DIR=../../build/lib/cmake/HPX
```

## Tutorials

The `tutorials/` directory contains Jupyter notebooks demonstrating HPXPy features, organized by implementation phase:

| Tutorial | Phase | Topics |
|----------|-------|--------|
| 01_getting_started | 1-3 | Runtime, arrays, operators |
| 03_math_and_algorithms | 4-5 | Math functions, sorting, scans, random |
| 04_slicing | 6 | Array slicing with start:stop:step |
| 05_reshape | 7 | Reshape, flatten, ravel, transpose |
| 06_gpu_acceleration | 8 | CUDA and SYCL GPU support |
| 07_distributed_computing | 9 | Collectives, distributed arrays |
| 08_benchmarks | - | Performance comparisons |

To run tutorials:
```bash
source build/setup_env.sh
jupyter lab tutorials/
```

## Examples

The `examples/` directory contains application notebooks demonstrating real-world use cases:

| Example | Description |
|---------|-------------|
| monte_carlo | Monte Carlo methods (moved from tutorials) |
| black_scholes | Options pricing simulation |
| image_processing | Image statistics with HPXPy |
| monte_carlo_pi | Classic Pi estimation |
| heat_diffusion | 1D heat equation simulation |
| kmeans_clustering | K-Means clustering with HPXPy |
| parallel_integration | Numerical integration benchmark |
| distributed_analytics | IoT sensor network analytics |
| distributed_reduction | SPMD pattern demonstration |
| distribution_demo | Distribution policies |
| multi_locality | Multi-process execution |
| scalability | Scalability benchmarks |

To run examples:
```bash
source build/setup_env.sh
jupyter lab examples/
```

## Running Tests

```bash
cd build
source setup_env.sh
pytest ../tests -v
```

## Documentation

- **[HPXPy Documentation](docs/)** - Full documentation (build with Sphinx)
- **[Getting Started](docs/source/getting_started/)** - Installation and quick start
- **[Tutorials](tutorials/)** - Interactive Jupyter notebooks
- **[API Reference](docs/source/api/)** - Complete API documentation

## License

HPXPy is distributed under the Boost Software License, Version 1.0.
See [LICENSE_1_0.txt](../LICENSE_1_0.txt) for details.
