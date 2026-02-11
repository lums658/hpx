# API Reference

Complete API documentation for HPXPy.

```{note}
This API reference includes full function signatures and examples.
When HPXPy is installed, additional documentation is auto-generated from docstrings.
```

## Runtime Management

Functions for initializing and managing the HPX runtime.

### `init`

```python
hpx.init(num_threads: int = None, config: list[str] = None) -> None
```

Initialize the HPX runtime system.

**Parameters:**
- `num_threads` - Number of worker threads. If None, uses all available cores.
- `config` - Additional HPX configuration options, e.g., `["--hpx:threads=4"]`

**Example:**
```python
import hpxpy as hpx

# Initialize with 4 threads
hpx.init(num_threads=4)

# ... perform operations ...

hpx.finalize()
```

### `finalize`

```python
hpx.finalize() -> None
```

Shut down the HPX runtime and release resources.

### `is_running`

```python
hpx.is_running() -> bool
```

Check if the HPX runtime is currently active.

### `runtime`

```python
@contextmanager
hpx.runtime(num_threads: int = None)
```

Context manager for automatic runtime management.

**Example:**
```python
with hpx.runtime(num_threads=8):
    arr = hpx.arange(1000000)
    print(hpx.sum(arr))
# Runtime automatically finalized
```

### `num_threads`

```python
hpx.num_threads() -> int
```

Return the number of HPX worker threads.

### `num_localities`

```python
hpx.num_localities() -> int
```

Return the number of localities (nodes) in distributed mode.

### `locality_id`

```python
hpx.locality_id() -> int
```

Return the ID of the current locality.

---

## Array Creation

Functions for creating HPXPy arrays.

### `array`

```python
hpx.array(data, dtype=None, device: str = 'cpu') -> ndarray
```

Create an array from existing data.

**Parameters:**
- `data` - Input data (list, tuple, numpy array, or scalar)
- `dtype` - NumPy dtype for the array. If None, inferred from data.
- `device` - Target device: `'cpu'`, `'gpu'`, `'sycl'`, or `'auto'`

**Example:**
```python
a = hpx.array([1.0, 2.0, 3.0])
b = hpx.array([[1, 2], [3, 4]], dtype='float64')
```

### `zeros`

```python
hpx.zeros(shape, dtype='float64', device: str = 'cpu') -> ndarray
```

Create an array of zeros.

**Parameters:**
- `shape` - Shape of the array (int or tuple)
- `dtype` - Data type (default: float64)
- `device` - Target device

**Example:**
```python
a = hpx.zeros(1000)              # 1D array
b = hpx.zeros((100, 100))        # 2D array
c = hpx.zeros(1000, device='gpu') # GPU array
```

### `ones`

```python
hpx.ones(shape, dtype='float64', device: str = 'cpu') -> ndarray
```

Create an array of ones.

### `empty`

```python
hpx.empty(shape, dtype='float64', device: str = 'cpu') -> ndarray
```

Create an uninitialized array.

### `full`

```python
hpx.full(shape, fill_value, dtype=None, device: str = 'cpu') -> ndarray
```

Create an array filled with a specified value.

### `arange`

```python
hpx.arange(start=None, stop=None, step=1, dtype='float64') -> ndarray
```

Create an array with evenly spaced values within a range.

**Parameters:**
- `start` - Start of interval (default: 0)
- `stop` - End of interval (exclusive)
- `step` - Spacing between values (default: 1)
- `dtype` - Output array data type

**Example:**
```python
a = hpx.arange(10)           # [0, 1, 2, ..., 9]
b = hpx.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
c = hpx.arange(10, 0, -1)    # [10, 9, 8, ..., 1]
```

### `linspace`

```python
hpx.linspace(start, stop, num=50, dtype='float64') -> ndarray
```

Create an array with evenly spaced values over an interval.

**Parameters:**
- `start` - Start value
- `stop` - End value (inclusive)
- `num` - Number of samples (default: 50)
- `dtype` - Output data type

**Example:**
```python
a = hpx.linspace(0, 1, 100)  # 100 points from 0 to 1
```

### `from_numpy`

```python
hpx.from_numpy(arr: np.ndarray, device: str = 'cpu') -> ndarray
```

Create an HPXPy array from a NumPy array.

---

## Reduction Operations

Parallel reduction functions with execution policy support. The default policy is `'par_unseq'` (configurable via CMake `HPXPY_DEFAULT_EXECUTION_POLICY`).

### `sum`

```python
hpx.sum(arr, axis=None, dtype=None, keepdims=False, policy='par_unseq') -> ndarray | scalar
```

Sum of array elements.

**Parameters:**
- `arr` - Input array
- `axis` - Axis along which to sum. If None, sum all elements.
- `dtype` - Type to use for accumulation
- `keepdims` - Keep reduced dimensions with size 1
- `policy` - Execution policy: `'seq'`, `'par'`, or `'par_unseq'`

**Example:**
```python
arr = hpx.arange(1000000)
total = hpx.sum(arr)                    # Parallel (default)
total = hpx.sum(arr, policy='seq')      # Sequential (deterministic)
```

### `prod`

```python
hpx.prod(arr, axis=None, dtype=None, keepdims=False, policy='par_unseq') -> ndarray | scalar
```

Product of array elements.

### `min`

```python
hpx.min(arr, axis=None, keepdims=False, policy='par_unseq') -> ndarray | scalar
```

Minimum value.

### `max`

```python
hpx.max(arr, axis=None, keepdims=False, policy='par_unseq') -> ndarray | scalar
```

Maximum value.

### `mean`

```python
hpx.mean(arr, axis=None, dtype=None, keepdims=False) -> ndarray | scalar
```

Arithmetic mean.

### `std`

```python
hpx.std(arr, axis=None, dtype=None, ddof=0, keepdims=False) -> ndarray | scalar
```

Standard deviation.

### `var`

```python
hpx.var(arr, axis=None, dtype=None, ddof=0, keepdims=False) -> ndarray | scalar
```

Variance.

### `reduce`

```python
hpx.reduce(arr, op='add', init=None, policy='par_unseq') -> scalar
```

General parallel reduction with custom operation.

**Parameters:**
- `arr` - Input array
- `op` - Reduction operation: `'add'`, `'mul'`, `'min'`, `'max'`
- `init` - Initial value
- `policy` - Execution policy

### `reduce_deterministic`

```python
hpx.reduce_deterministic(arr, op='add', init=None) -> scalar
```

Deterministic reduction (sequential, reproducible floating-point results).

---

## Sorting and Searching

### `sort`

```python
hpx.sort(arr, axis=-1, policy='par_unseq') -> ndarray
```

Return a sorted copy of an array.

**Parameters:**
- `arr` - Input array
- `axis` - Axis along which to sort (default: -1, last axis)
- `policy` - Execution policy

### `stable_sort`

```python
hpx.stable_sort(arr) -> ndarray
```

Stable sort preserving order of equal elements.

### `argsort`

```python
hpx.argsort(arr) -> ndarray
```

Return indices that would sort an array.

```{note}
Currently only supports float64 arrays.
```

### `count`

```python
hpx.count(arr, value, policy='par_unseq') -> int
```

Count occurrences of a value.

### `searchsorted`

```python
hpx.searchsorted(arr, values) -> ndarray
```

Find insertion points in sorted array.

### `nonzero`

```python
hpx.nonzero(arr) -> ndarray
```

Return indices of non-zero elements.

### `argmin`

```python
hpx.argmin(arr, axis=None) -> int | ndarray
```

Index of minimum value.

### `argmax`

```python
hpx.argmax(arr, axis=None) -> int | ndarray
```

Index of maximum value.

---

## Math Functions

Element-wise mathematical operations (parallel by default).

### Basic Math

| Function | Description |
|----------|-------------|
| `sqrt(x)` | Square root |
| `square(x)` | Element-wise square |
| `abs(x)` | Absolute value |
| `sign(x)` | Sign function (-1, 0, or 1) |
| `power(x, y)` | Element-wise power |
| `clip(x, a_min, a_max)` | Clip values to range |

### Exponential and Logarithmic

| Function | Description |
|----------|-------------|
| `exp(x)` | Exponential (e^x) |
| `exp2(x)` | Base-2 exponential (2^x) |
| `log(x)` | Natural logarithm |
| `log2(x)` | Base-2 logarithm |
| `log10(x)` | Base-10 logarithm |

### Trigonometric

| Function | Description |
|----------|-------------|
| `sin(x)` | Sine |
| `cos(x)` | Cosine |
| `tan(x)` | Tangent |
| `arcsin(x)` | Inverse sine |
| `arccos(x)` | Inverse cosine |
| `arctan(x)` | Inverse tangent |

### Hyperbolic

| Function | Description |
|----------|-------------|
| `sinh(x)` | Hyperbolic sine |
| `cosh(x)` | Hyperbolic cosine |
| `tanh(x)` | Hyperbolic tangent |

### Rounding

| Function | Description |
|----------|-------------|
| `floor(x)` | Round down to integer |
| `ceil(x)` | Round up to integer |
| `trunc(x)` | Truncate to integer |

---

## Scan Operations

Cumulative (prefix) operations.

### `cumsum`

```python
hpx.cumsum(arr, axis=None) -> ndarray
```

Cumulative sum.

**Example:**
```python
arr = hpx.array([1, 2, 3, 4, 5])
hpx.cumsum(arr)  # [1, 3, 6, 10, 15]
```

### `cumprod`

```python
hpx.cumprod(arr, axis=None) -> ndarray
```

Cumulative product.

### `inclusive_scan`

```python
hpx.inclusive_scan(arr, op='add') -> ndarray
```

Inclusive scan with operation.

**Parameters:**
- `arr` - Input array
- `op` - Operation: `'add'`, `'mul'`

### `exclusive_scan`

```python
hpx.exclusive_scan(arr, init, op='add') -> ndarray
```

Exclusive scan with initial value.

**Parameters:**
- `arr` - Input array
- `init` - Initial value (appears as first element)
- `op` - Operation: `'add'`, `'mul'`

---

## Advanced Algorithms

### `transform_reduce`

```python
hpx.transform_reduce(arr, transform_op, reduce_op) -> scalar
```

Combine transform and reduction.

**Parameters:**
- `arr` - Input array
- `transform_op` - Transform: `'square'`, `'abs'`, `'identity'`
- `reduce_op` - Reduction: `'add'`, `'mul'`, `'min'`, `'max'`

**Example:**
```python
# Sum of squares
hpx.transform_reduce(arr, 'square', 'add')

# Sum of absolute values
hpx.transform_reduce(arr, 'abs', 'add')
```

### `reduce_by_key`

```python
hpx.reduce_by_key(keys, values, op='add') -> tuple[ndarray, ndarray]
```

Reduce values by key groups.

**Returns:** (unique_keys, reduced_values)

### `nth_element`

```python
hpx.nth_element(arr, n) -> scalar
```

Find nth smallest element.

### `median`

```python
hpx.median(arr) -> scalar
```

Compute the median.

### `percentile`

```python
hpx.percentile(arr, q) -> scalar
```

Compute the qth percentile.

---

## Array Manipulation

### `flip`

```python
hpx.flip(arr) -> ndarray
```

Reverse the array.

### `roll`

```python
hpx.roll(arr, shift) -> ndarray
```

Roll array elements.

### `partition`

```python
hpx.partition(arr, pivot) -> ndarray
```

Partition array around pivot value.

### `array_equal`

```python
hpx.array_equal(a, b) -> bool
```

Check if two arrays are equal.

### `diff`

```python
hpx.diff(arr, n=1) -> ndarray
```

Calculate the n-th discrete difference.

---

## Set Operations

Operations on sorted arrays.

| Function | Description |
|----------|-------------|
| `unique(arr)` | Unique elements |
| `setdiff1d(a, b)` | Elements in a but not in b |
| `intersect1d(a, b)` | Common elements |
| `union1d(a, b)` | Union of elements |
| `setxor1d(a, b)` | Symmetric difference |
| `includes(a, b)` | Check if a includes all of b |
| `isin(arr, test)` | Test membership |
| `merge_sorted(a, b)` | Merge two sorted arrays |

---

## Element-wise Operations

### `maximum`

```python
hpx.maximum(a, b) -> ndarray
```

Element-wise maximum of two arrays.

### `minimum`

```python
hpx.minimum(a, b) -> ndarray
```

Element-wise minimum of two arrays.

### `where`

```python
hpx.where(condition, x, y) -> ndarray
```

Select from x where condition is true, else from y.

**Example:**
```python
arr = hpx.arange(10)
result = hpx.where(arr > 5, arr * 10, arr)
# [0, 1, 2, 3, 4, 5, 60, 70, 80, 90]
```

---

## Random Module (`hpx.random`)

Parallel random number generation.

### `seed`

```python
hpx.random.seed(seed: int) -> None
```

Set random seed for reproducibility.

### `rand`

```python
hpx.random.rand(*shape) -> ndarray
```

Uniform random values in [0, 1).

### `randn`

```python
hpx.random.randn(*shape) -> ndarray
```

Standard normal distribution.

### `randint`

```python
hpx.random.randint(low, high=None, size=None) -> ndarray
```

Random integers.

### `uniform`

```python
hpx.random.uniform(low=0.0, high=1.0, size=None) -> ndarray
```

Uniform distribution over [low, high).

---

## Execution Policies (`hpx.execution`)

Control algorithm parallelization. The default for all reduction and sorting operations is `'par_unseq'`, configurable at build time via CMake `HPXPY_DEFAULT_EXECUTION_POLICY`.

| Policy | String | Description |
|--------|--------|-------------|
| `hpx.execution.seq` | `'seq'` | Sequential execution (deterministic FP order) |
| `hpx.execution.par` | `'par'` | Parallel execution (multi-threaded) |
| `hpx.execution.par_unseq` | `'par_unseq'` | Parallel + vectorized (SIMD) — **default** |

**Example:**
```python
# Uses default policy (par_unseq) — parallel
result = hpx.sum(arr)

# Explicit sequential for deterministic floating-point results
result = hpx.sum(arr, policy='seq')
```

---

## GPU Module (`hpx.gpu`)

CUDA GPU support via HPX cuda_executor.

### Device Management

| Function | Description |
|----------|-------------|
| `is_available()` | Check CUDA availability |
| `device_count()` | Number of GPUs |
| `get_devices()` | List all GPU info |
| `get_device(id)` | Get specific GPU info |
| `current_device()` | Current GPU ID |
| `set_device(id)` | Set current GPU |
| `synchronize()` | Synchronize GPU |
| `memory_info()` | Get memory usage |

### Array Creation

```python
hpx.gpu.zeros(shape, dtype='float64') -> ndarray
hpx.gpu.ones(shape, dtype='float64') -> ndarray
hpx.gpu.full(shape, value, dtype='float64') -> ndarray
hpx.gpu.from_numpy(arr) -> ndarray
```

### Async Operations

| Function | Description |
|----------|-------------|
| `enable_async()` | Enable HPX CUDA polling |
| `disable_async()` | Disable polling |
| `is_async_enabled()` | Check polling state |
| `AsyncContext` | Context manager for async ops |

---

## SYCL Module (`hpx.sycl`)

Cross-platform GPU support via HPX sycl_executor.

Same interface as GPU module for Intel, AMD, and Apple GPUs.

---

## Distributed Module

### PartitionedArray Creation

Distributed arrays backed by `hpx::partitioned_vector`. Data is automatically partitioned across localities.

```python
hpx.partitioned_zeros(n) -> PartitionedArray
hpx.partitioned_ones(n) -> PartitionedArray
hpx.partitioned_full(n, value) -> PartitionedArray
hpx.partitioned_arange(start, stop) -> PartitionedArray
hpx.partitioned_from_numpy(arr) -> PartitionedArray
```

### Distributed Reductions

Reduce across all partitions (and all localities in multi-locality mode).

| Function | Description |
|----------|-------------|
| `distributed_sum(arr)` | Sum across all localities |
| `distributed_mean(arr)` | Mean across all localities |
| `distributed_min(arr)` | Min across all localities |
| `distributed_max(arr)` | Max across all localities |
| `distributed_var(arr)` | Variance across all localities |
| `distributed_std(arr)` | Std deviation across all localities |

### Collective Operations

Communication primitives for SPMD (Single Program, Multiple Data) patterns.

| Function | Description |
|----------|-------------|
| `all_reduce(arr, op='sum')` | Reduce across all localities, result on all |
| `broadcast(arr, root=0)` | Send data from root to all localities |
| `gather(arr, root=0)` | Collect data from all localities to root |
| `scatter(arr, root=0)` | Distribute chunks from root to all |
| `barrier(name)` | Synchronization barrier |

---

## Launcher Module (`hpx.launcher`)

Multi-locality job launching via TCP parcelport.

### `launch_localities`

```python
hpx.launcher.launch_localities(
    script: str,
    num_localities: int = 2,
    threads_per_locality: int = 0,
    host: str = "localhost",
    verbose: bool = False,
) -> list[subprocess.Popen]
```

Launch a Python script across multiple HPX localities. Each locality runs as a separate process connected via TCP.

### `init_from_args`

```python
hpx.launcher.init_from_args(args=None) -> None
```

Initialize HPX runtime from command-line arguments. Parses HPX arguments after `--` separator. Used inside worker scripts launched by `launch_localities()`.

---

## See Also

- **[HPX C++ API](https://hpx-docs.stellar-group.org/latest/html/api.html)** - Full HPX API reference
- **[NumPy API](https://numpy.org/doc/stable/reference/)** - NumPy reference (compatible interface)
