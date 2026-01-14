# HPXPy Tutorials

Interactive Jupyter notebooks for learning HPXPy, organized by implementation phase.

## Tutorial Overview

| # | Notebook | Phase | Topics |
|---|----------|-------|--------|
| 01 | [01_getting_started](01_getting_started.ipynb) | 1-3 | Runtime, arrays, operators |
| 03 | [03_math_and_algorithms](03_math_and_algorithms.ipynb) | 4-5 | Math functions, sorting, scans, random |
| 04 | [04_slicing](04_slicing.ipynb) | 6 | Array slicing with start:stop:step |
| 05 | [05_reshape](05_reshape.ipynb) | 7 | Reshape, flatten, ravel, transpose |
| 06 | [06_gpu_acceleration](06_gpu_acceleration.ipynb) | 8 | CUDA and SYCL GPU support |
| 07 | [07_distributed_computing](07_distributed_computing.ipynb) | 9 | Collectives, distributed arrays |
| 08 | [08_benchmarks](08_benchmarks.ipynb) | - | Performance comparisons |

## Phase Reference

| Phase | Features Introduced |
|-------|-------------------|
| 1-2 | Runtime management, array creation (zeros, ones, arange) |
| 3 | Arithmetic operators (+, -, *, /), comparisons (<, >, ==) |
| 4 | Math functions (sin, cos, exp, log, sqrt) |
| 5 | Reductions (sum, prod, min, max), sorting, scans |
| 6 | Array slicing (arr[start:stop:step]) |
| 7 | Reshape, flatten, ravel, transpose |
| 8 | GPU acceleration (CUDA, SYCL) |
| 9 | Distributed arrays, collective operations |

## Prerequisites

1. HPXPy installed and built
2. JupyterLab or Jupyter Notebook

## Running the Tutorials

```bash
cd /path/to/hpx/python

# Activate virtual environment
source .venv/bin/activate

# Install Jupyter if needed
pip install jupyterlab

# Start JupyterLab
jupyter lab tutorials/
```

## Tutorial Order

The tutorials build on each other:

1. **Getting Started** - Learn basics: runtime, arrays, operators
2. **Math and Algorithms** - Math functions, sorting, reductions
3. **Slicing** - Extract portions of arrays
4. **Reshape** - Change array dimensions
5. **GPU Acceleration** - Leverage CUDA/SYCL GPUs
6. **Distributed Computing** - Scale across multiple nodes
7. **Benchmarks** - Compare performance with NumPy

## Extracting Python Code

To extract Python code from a notebook:

```bash
# Using nbconvert
jupyter nbconvert --to script tutorials/01_getting_started.ipynb

# Or using Python
python -c "
import json
with open('tutorials/01_getting_started.ipynb') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        print(''.join(cell['source']))
        print()
"
```

## Tips

- Run cells in order (some depend on previous cells)
- The HPX runtime must be initialized before using HPXPy functions
- Use `hpx.finalize()` or the `with hpx.runtime():` context manager
- Each notebook handles its own runtime lifecycle
