# Tutorials

Interactive Jupyter notebooks to learn HPXPy step by step.

## Learning Path

::::{grid} 1
:gutter: 3

:::{grid-item-card} 1. Getting Started
:link: 01_getting_started
:link-type: doc

**Level:** Beginner | **Time:** 15 min

Learn the basics: HPX runtime, array creation, and simple operations.

Topics covered:
- Initializing HPX runtime
- Creating arrays (zeros, ones, arange)
- Basic array operations
- NumPy interoperability
:::

:::{grid-item-card} 2. Parallel Algorithms
:link: 02_parallel_algorithms
:link-type: doc

**Level:** Beginner | **Time:** 20 min

Master HPXPy's parallel algorithms for fast array processing.

Topics covered:
- Math functions (sqrt, exp, sin, etc.)
- Sorting algorithms
- Scan operations (cumsum, cumprod)
- Random number generation
- Execution policies
:::

:::{grid-item-card} 3. Math and Algorithms
:link: 03_math_and_algorithms
:link-type: doc

**Level:** Intermediate | **Time:** 25 min

Deep dive into mathematical operations and parallel algorithms.

Topics covered:
- Element-wise math functions
- Reduction operations
- Custom parallel operations
:::

:::{grid-item-card} 4. Slicing
:link: 04_slicing
:link-type: doc

**Level:** Beginner | **Time:** 15 min

Master array slicing operations.

Topics covered:
- Basic slicing (start:stop:step)
- Negative indices
- NumPy compatibility
:::

:::{grid-item-card} 5. Reshape Operations
:link: 05_reshape
:link-type: doc

**Level:** Beginner | **Time:** 15 min

Learn array shape manipulation.

Topics covered:
- Reshape arrays
- Flatten and ravel
- Broadcasting basics
:::

:::{grid-item-card} 6. GPU Acceleration
:link: 06_gpu_acceleration
:link-type: doc

**Level:** Advanced | **Time:** 30 min

Accelerate computations with CUDA and SYCL GPUs.

Topics covered:
- Device detection and selection
- GPU array creation
- Host-device transfers
- Async operations with futures
:::

:::{grid-item-card} 7. Distributed Computing
:link: 07_distributed_computing
:link-type: doc

**Level:** Advanced | **Time:** 30 min

Scale across multiple nodes with distributed arrays.

Topics covered:
- Multi-locality concepts
- Distributed array creation
- Collective operations
- Launching multi-node jobs
:::

:::{grid-item-card} 8. Benchmarks
:link: 08_benchmarks
:link-type: doc

**Level:** Intermediate | **Time:** 20 min

Measure and compare HPXPy performance.

Topics covered:
- Benchmarking methodology
- NumPy comparison
- Scaling with array size
- Performance analysis
:::

:::{grid-item-card} 9. Advanced Algorithms
:link: 09_advanced_algorithms
:link-type: doc

**Level:** Advanced | **Time:** 30 min

Master advanced HPX parallel algorithms.

Topics covered:
- Scan algorithms (inclusive/exclusive)
- Transform-reduce operations
- Set operations (union, intersection)
- Selection algorithms (median, percentile)
- Grouped reductions (reduce_by_key)
:::

::::

## Running the Tutorials

### Option 1: JupyterLab

```bash
cd hpx/python/tutorials
pip install jupyterlab
jupyter lab
```

### Option 2: VS Code

Open the tutorials folder in VS Code with the Jupyter extension installed.

### Option 3: Google Colab

Upload notebooks to Google Colab (note: HPX must be installed in the Colab environment).

## Prerequisites

Before starting the tutorials, ensure you have:

1. HPXPy installed (see [Installation](../getting_started/index.md))
2. Jupyter or JupyterLab installed
3. NumPy and Matplotlib for visualization

```bash
pip install jupyterlab numpy matplotlib
```

## Related Resources

- **[HPX Tutorials](https://hpx-docs.stellar-group.org/latest/html/manual/getting_started.html)** - HPX C++ tutorials
- **[NumPy Tutorials](https://numpy.org/doc/stable/user/absolute_beginners.html)** - NumPy basics

```{toctree}
:maxdepth: 1
:hidden:

01_getting_started
02_parallel_algorithms
03_math_and_algorithms
04_slicing
05_reshape
06_gpu_acceleration
07_distributed_computing
08_benchmarks
09_advanced_algorithms
```
