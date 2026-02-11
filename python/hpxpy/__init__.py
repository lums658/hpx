# HPXPy - High Performance Python Arrays powered by HPX
#
# SPDX-License-Identifier: BSL-1.0

"""
HPXPy: High-performance distributed NumPy-like arrays powered by HPX.

HPXPy provides a NumPy-compatible interface for parallel and distributed
array computing using the HPX C++ runtime system.

Basic usage::

    import hpxpy as hpx

    # Initialize the HPX runtime
    hpx.init()

    # Create arrays
    a = hpx.zeros((1000,))
    b = hpx.ones((1000,))

    # Perform parallel operations
    result = hpx.sum(a + b)

    # Clean up
    hpx.finalize()

Or using the context manager::

    import hpxpy as hpx

    with hpx.runtime():
        a = hpx.arange(1000000)
        print(hpx.sum(a))
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    # Version info
    "__version__",
    # Runtime management
    "init",
    "finalize",
    "is_running",
    "runtime",
    "num_threads",
    "num_localities",
    "locality_id",
    # Array creation
    "array",
    "zeros",
    "ones",
    "empty",
    "full",
    "arange",
    "linspace",
    "from_numpy",
    # Array class
    "ndarray",
    # Reduction algorithms
    "sum",
    "prod",
    "min",
    "max",
    "mean",
    "std",
    "var",
    # Sorting
    "sort",
    "argsort",
    "count",
    # Math functions
    "sqrt",
    "square",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "floor",
    "ceil",
    "trunc",
    "abs",
    "sign",
    # Scan operations
    "cumsum",
    "cumprod",
    # HPX algorithm exposures
    "inclusive_scan",
    "exclusive_scan",
    "transform_reduce",
    "reduce_by_key",
    "reduce",
    "reduce_deterministic",
    # Additional algorithms
    "any",
    "all",
    "argmin",
    "argmax",
    "diff",
    "unique",
    # Tier 2 algorithms
    "array_equal",
    "flip",
    "stable_sort",
    "roll",
    "nonzero",
    "nth_element",
    "median",
    "percentile",
    # Tier 3 algorithms
    "merge_sorted",
    "setdiff1d",
    "intersect1d",
    "union1d",
    "setxor1d",
    "includes",
    "isin",
    "searchsorted",
    "partition",
    # Element-wise functions
    "maximum",
    "minimum",
    "clip",
    "power",
    "where",
    # Random submodule
    "random",
    # Execution policies
    "execution",
    # Distribution (Phase 3)
    "distribution",
    # Collective operations (Phase 4)
    "collectives",
    "all_reduce",
    "broadcast",
    "gather",
    "scatter",
    "barrier",
    # Distributed arrays (Phase 4)
    "DistributionPolicy",
    "distributed_zeros",
    "distributed_ones",
    "distributed_full",
    "distributed_from_numpy",
    # Partitioned arrays (Phase 11) - HPX partitioned_vector
    "PartitionedArray",
    "partitioned_zeros",
    "partitioned_ones",
    "partitioned_full",
    "partitioned_arange",
    "partitioned_from_numpy",
    "distributed_sum",
    "distributed_mean",
    "distributed_min",
    "distributed_max",
    "distributed_var",
    "distributed_std",
    "distributed_prod",
    # Multi-locality launcher (Phase 4)
    "launcher",
    # GPU support (Phase 5)
    "gpu",
    # SYCL support (Phase 5) - HPX-native backend for Intel/AMD/Apple GPUs
    "sycl",
]

# Import the compiled extension module
try:
    from hpxpy._core import (
        # Runtime
        init as _init,
        finalize as _finalize,
        is_running,
        num_threads,
        num_localities,
        locality_id,
        # Array class
        ndarray,
        # Array creation
        _zeros,
        _ones,
        _empty,
        _arange,
        _array_from_numpy,
        # Algorithms
        _sum,
        _sum_axis,
        _prod,
        _min,
        _max,
        _sort,
        _count,
        # New HPX algorithm exposures
        _any,
        _all,
        _argmin,
        _argmax,
        _argsort,
        _diff,
        _unique,
        _inclusive_scan,
        _exclusive_scan,
        _transform_reduce,
        _reduce_by_key,
        # Tier 2 algorithms
        _array_equal,
        _flip,
        _stable_sort,
        _rotate,
        _nonzero,
        _nth_element,
        _median,
        _percentile,
        # Tier 3 algorithms
        _merge,
        _setdiff1d,
        _intersect1d,
        _union1d,
        _setxor1d,
        _includes,
        _isin,
        _searchsorted,
        _partition,
        # Math functions
        _sqrt,
        _square,
        _exp,
        _exp2,
        _log,
        _log2,
        _log10,
        _sin,
        _cos,
        _tan,
        _arcsin,
        _arccos,
        _arctan,
        _sinh,
        _cosh,
        _tanh,
        _floor,
        _ceil,
        _trunc,
        _abs,
        _sign,
        # Scan operations
        _cumsum,
        _cumprod,
        # Element-wise functions
        _maximum,
        _minimum,
        _clip,
        _power,
        _where,
        # Random submodule
        random as _random_module,
        # Execution module
        execution,
        # Distribution module (Phase 3)
        distribution,
        # Collectives module (Phase 4)
        collectives,
        all_reduce as _all_reduce,
        broadcast as _broadcast,
        gather as _gather,
        scatter as _scatter,
        barrier as _barrier,
        # Distributed arrays (Phase 4)
        distributed_zeros,
        distributed_ones,
        distributed_full,
        distributed_from_numpy,
        # Partitioned arrays (Phase 11) - HPX partitioned_vector
        PartitionedArray,
        partitioned_zeros,
        partitioned_ones,
        partitioned_full,
        partitioned_arange,
        partitioned_from_numpy,
        distributed_sum,
        distributed_mean,
        distributed_min,
        distributed_max,
        distributed_var,
        distributed_std,
        distributed_prod,
    )
    # DistributionPolicy is in the distribution submodule
    from hpxpy._core.distribution import DistributionPolicy

    _HPX_AVAILABLE = True

    # Default execution policy, set by CMake (HPXPY_DEFAULT_EXECUTION_POLICY)
    from hpxpy._core import default_execution_policy as _DEFAULT_POLICY
except ImportError as e:
    _HPX_AVAILABLE = False
    _IMPORT_ERROR = str(e)
    _DEFAULT_POLICY = "par_unseq"  # fallback


def _check_available() -> None:
    """Check if HPX extension is available."""
    if not _HPX_AVAILABLE:
        raise ImportError(
            f"HPXPy extension module not available: {_IMPORT_ERROR}\n"
            "Make sure HPX is installed and the extension was built correctly."
        )


# -----------------------------------------------------------------------------
# Runtime Management
# -----------------------------------------------------------------------------


class runtime:
    """Context manager for HPX runtime initialization.

    Example::

        with hpx.runtime(num_threads=4):
            # HPX operations here
            result = hpx.sum(hpx.arange(1000))
        # Runtime automatically finalized
    """

    def __init__(self, num_threads: int | None = None, config: list[str] | None = None):
        """Initialize the runtime context.

        Parameters
        ----------
        num_threads : int, optional
            Number of OS threads to use. Defaults to hardware concurrency.
        config : list of str, optional
            Additional HPX configuration options.
        """
        self.num_threads = num_threads
        self.config = config

    def __enter__(self) -> "runtime":
        init(num_threads=self.num_threads, config=self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        finalize()
        return False


def init(num_threads: int | None = None, config: list[str] | None = None) -> None:
    """Initialize the HPX runtime system.

    This must be called before any HPX operations. Alternatively, use the
    ``runtime`` context manager.

    Parameters
    ----------
    num_threads : int, optional
        Number of OS threads to use. If not specified, uses the number
        of hardware threads available.
    config : list of str, optional
        Additional HPX configuration options, e.g.,
        ``["--hpx:threads=4", "--hpx:localities=2"]``

    Raises
    ------
    RuntimeError
        If the runtime is already initialized.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init(num_threads=4)
    >>> # ... do work ...
    >>> hpx.finalize()
    """
    _check_available()
    _init(num_threads=num_threads, config=config or [])


def finalize() -> None:
    """Finalize the HPX runtime system.

    This should be called when HPX operations are complete. After calling
    this, HPX operations cannot be performed until ``init`` is called again.

    Raises
    ------
    RuntimeError
        If the runtime is not initialized.
    """
    _check_available()
    _finalize()


# -----------------------------------------------------------------------------
# Device Selection
# -----------------------------------------------------------------------------


def _resolve_device(device) -> tuple:
    """Resolve device specification to (backend, device_id).

    Parameters
    ----------
    device : str or int or None
        Device specification:
        - None or 'cpu': Use CPU backend
        - 'gpu' or 'cuda': Use CUDA GPU if available, else raise error
        - 'sycl': Use SYCL GPU if available, else raise error
        - 'auto': Use best available (CUDA > SYCL > CPU)
        - int: Use specific GPU device ID (CUDA preferred, then SYCL)

    Returns
    -------
    tuple
        (backend, device_id) where backend is 'cpu', 'gpu', or 'sycl'
    """
    # Lazy imports to avoid circular dependency
    from hpxpy import gpu as _gpu_mod
    from hpxpy import sycl as _sycl_mod

    if device is None or device == 'cpu':
        return ('cpu', None)

    if device == 'auto':
        # Auto-select: prefer CUDA > SYCL > CPU
        # Both use HPX executors for proper integration
        if _gpu_mod.is_available():
            return ('gpu', 0)
        if _sycl_mod.is_available():
            return ('sycl', 0)
        return ('cpu', None)

    if device == 'gpu' or device == 'cuda':
        # Explicit CUDA GPU request
        if not _gpu_mod.is_available():
            raise RuntimeError(
                "CUDA GPU requested but CUDA is not available. "
                "Use device='auto' for automatic fallback, device='sycl' for SYCL GPUs, "
                "or device='cpu' for CPU."
            )
        return ('gpu', 0)

    if device == 'sycl':
        # Explicit SYCL GPU request (works on Intel, AMD, Apple Silicon via AdaptiveCpp)
        if not _sycl_mod.is_available():
            raise RuntimeError(
                "SYCL requested but not available. "
                "SYCL requires HPX built with HPX_WITH_SYCL=ON and a SYCL compiler. "
                "Use device='auto' for automatic fallback or device='cpu' for CPU."
            )
        return ('sycl', 0)

    if isinstance(device, int):
        # Specific GPU device ID - try CUDA first, then SYCL
        if _gpu_mod.is_available():
            if device >= _gpu_mod.device_count():
                raise ValueError(f"Invalid CUDA device ID {device}. Available: 0-{_gpu_mod.device_count()-1}")
            return ('gpu', device)
        if _sycl_mod.is_available():
            if device >= _sycl_mod.device_count():
                raise ValueError(f"Invalid SYCL device ID {device}. Available: 0-{_sycl_mod.device_count()-1}")
            return ('sycl', device)
        raise RuntimeError(f"GPU device {device} requested but no GPU backend is available.")

    raise ValueError(
        f"Invalid device specification: {device!r}. "
        "Use 'cpu', 'gpu', 'cuda', 'sycl', 'auto', or an integer GPU device ID."
    )


# -----------------------------------------------------------------------------
# Array Creation Functions
# -----------------------------------------------------------------------------


def array(data, dtype=None, copy: bool = True, device=None):
    """Create an HPXPy array from existing data.

    Parameters
    ----------
    data : array_like
        Input data (list, tuple, or NumPy array).
    dtype : numpy.dtype, optional
        Desired data type. If not specified, inferred from data.
    copy : bool, default True
        If True, always copy the data. If False, try to share memory
        when possible (only works with contiguous NumPy arrays).
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        A new HPXPy array on the specified device.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> a = hpx.array([1, 2, 3, 4, 5])
    >>> a = hpx.array(numpy_array, copy=False)  # Zero-copy if possible
    >>> a = hpx.array([1, 2, 3], device='auto')  # GPU if available
    """
    _check_available()
    import numpy as np

    np_array = np.asarray(data, dtype=dtype)
    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        from hpxpy import gpu as _gpu_mod
        return _gpu_mod.from_numpy(np_array, device=device_id)
    if backend == 'sycl':
        from hpxpy import sycl as _sycl_mod
        return _sycl_mod.from_numpy(np_array, device=device_id)
    return _array_from_numpy(np_array, copy=copy)


def from_numpy(arr, copy: bool = False, device=None):
    """Create an HPXPy array from a NumPy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input NumPy array.
    copy : bool, default False
        If False, try to share memory with the input array (zero-copy).
        If True, always make a copy. Only applies to CPU arrays.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        A new HPXPy array on the specified device.

    Notes
    -----
    Zero-copy is only possible for CPU arrays when the input is C-contiguous.
    GPU arrays always copy data.
    """
    _check_available()
    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        from hpxpy import gpu as _gpu_mod
        return _gpu_mod.from_numpy(arr, device=device_id)
    if backend == 'sycl':
        from hpxpy import sycl as _sycl_mod
        return _sycl_mod.from_numpy(arr, device=device_id)
    return _array_from_numpy(arr, copy=copy)


def zeros(shape, dtype=None, device=None):
    """Create an array filled with zeros.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array.
    dtype : numpy.dtype, optional
        Data type. Default is float64.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        Array of zeros with the given shape and dtype.

    Examples
    --------
    >>> a = hpx.zeros((10, 10))
    >>> b = hpx.zeros(1000, dtype=np.int32)
    >>> c = hpx.zeros(1000000, device='auto')  # GPU if available
    """
    _check_available()
    import numpy as np

    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)

    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        from hpxpy import gpu as _gpu_mod
        return _gpu_mod.zeros(list(shape), device=device_id)
    if backend == 'sycl':
        from hpxpy import sycl as _sycl_mod
        return _sycl_mod.zeros(list(shape), device=device_id)
    return _zeros(shape, dtype)


def ones(shape, dtype=None, device=None):
    """Create an array filled with ones.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array.
    dtype : numpy.dtype, optional
        Data type. Default is float64.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        Array of ones with the given shape and dtype.
    """
    _check_available()
    import numpy as np

    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)

    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        from hpxpy import gpu as _gpu_mod
        return _gpu_mod.ones(list(shape), device=device_id)
    if backend == 'sycl':
        from hpxpy import sycl as _sycl_mod
        return _sycl_mod.ones(list(shape), device=device_id)
    return _ones(shape, dtype)


def empty(shape, dtype=None, device=None):
    """Create an uninitialized array.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array.
    dtype : numpy.dtype, optional
        Data type. Default is float64.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        Uninitialized array with the given shape and dtype.

    Notes
    -----
    The values in the array are not initialized and may contain
    arbitrary data. Use ``zeros`` or ``ones`` if you need initialized values.
    Note: GPU 'empty' currently returns zero-filled array.
    """
    _check_available()
    import numpy as np

    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)

    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        # GPU arrays don't have true 'empty' - use zeros for safety
        from hpxpy import gpu as _gpu_mod
        return _gpu_mod.zeros(list(shape), device=device_id)
    if backend == 'sycl':
        # SYCL arrays don't have true 'empty' - use zeros for safety
        from hpxpy import sycl as _sycl_mod
        return _sycl_mod.zeros(list(shape), device=device_id)
    return _empty(shape, dtype)


def arange(start, stop=None, step=1, dtype=None, device=None):
    """Create an array with evenly spaced values.

    Parameters
    ----------
    start : number
        Start of interval (or stop if stop is None).
    stop : number, optional
        End of interval (exclusive).
    step : number, default 1
        Spacing between values.
    dtype : numpy.dtype, optional
        Data type. Inferred from arguments if not specified.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        Array of evenly spaced values.

    Examples
    --------
    >>> hpx.arange(5)           # [0, 1, 2, 3, 4]
    >>> hpx.arange(1, 5)        # [1, 2, 3, 4]
    >>> hpx.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
    >>> hpx.arange(1000000, device='auto')  # GPU if available
    """
    _check_available()
    import numpy as np

    if stop is None:
        stop = start
        start = 0

    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        # Create on CPU first, then transfer
        from hpxpy import gpu as _gpu_mod
        np_arr = np.arange(start, stop, step, dtype=dtype)
        return _gpu_mod.from_numpy(np_arr, device=device_id)
    if backend == 'sycl':
        # Create on CPU first, then transfer
        from hpxpy import sycl as _sycl_mod
        np_arr = np.arange(start, stop, step, dtype=dtype)
        return _sycl_mod.from_numpy(np_arr, device=device_id)
    return _arange(start, stop, step, dtype)


def linspace(start, stop, num: int = 50, dtype=None, device=None):
    """Create an array with evenly spaced values over an interval.

    Parameters
    ----------
    start : number
        Start of interval.
    stop : number
        End of interval (inclusive).
    num : int, default 50
        Number of values to generate.
    dtype : numpy.dtype, optional
        Data type. Default is float64.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        Array of evenly spaced values.
    """
    _check_available()
    import numpy as np

    np_arr = np.linspace(start, stop, num, dtype=dtype)
    return from_numpy(np_arr, copy=True, device=device)


def full(shape, fill_value, dtype=None, device=None):
    """Create an array filled with a specified value.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array.
    fill_value : scalar
        Fill value.
    dtype : numpy.dtype, optional
        Data type. Default is float64.
    device : str or int, optional
        Device to create array on:
        - None or 'cpu': CPU (default)
        - 'gpu': GPU (error if unavailable)
        - 'auto': GPU if available, else CPU
        - int: Specific GPU device ID

    Returns
    -------
    ndarray or GPUArray
        Array filled with the specified value.

    Examples
    --------
    >>> a = hpx.full((10, 10), 3.14)
    >>> b = hpx.full(1000, 42, device='auto')
    """
    _check_available()
    import numpy as np

    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)

    backend, device_id = _resolve_device(device)

    if backend == 'gpu':
        from hpxpy import gpu as _gpu_mod
        return _gpu_mod.full(list(shape), float(fill_value), device=device_id)
    if backend == 'sycl':
        from hpxpy import sycl as _sycl_mod
        return _sycl_mod.full(list(shape), float(fill_value), device=device_id)

    # CPU: create with numpy and convert
    np_arr = np.full(shape, fill_value, dtype=dtype)
    return from_numpy(np_arr, copy=True)


# -----------------------------------------------------------------------------
# Reduction Algorithms
# -----------------------------------------------------------------------------


def sum(arr, axis=None, dtype=None, keepdims: bool = False, policy: str = _DEFAULT_POLICY):
    """Sum of array elements.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to sum. Default is to sum all elements.
    dtype : numpy.dtype, optional
        Type to use for accumulation.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq" (sequential), "par" (parallel),
        or "par_unseq" (parallel + SIMD). Use "par" for large arrays to
        enable multi-threaded execution. Note: "par" may produce
        non-deterministic floating-point results due to reduction order.

    Returns
    -------
    ndarray or scalar
        Sum of elements.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.arange(10)
    >>> hpx.sum(arr)
    45.0
    >>> hpx.sum(arr, policy="par")  # Multi-threaded for large arrays
    45.0
    >>> hpx.finalize()
    """
    _check_available()
    if axis is None:
        if keepdims:
            # Full reduction with keepdims: reduce all, then reshape
            import numpy as np
            result = _sum(arr, policy)
            shape = tuple(1 for _ in arr.shape)
            return from_numpy(np.array(result, dtype=arr.dtype).reshape(shape))
        return _sum(arr, policy)
    elif isinstance(axis, int):
        # Single axis reduction: use native HPX implementation
        return _sum_axis(arr, axis, keepdims)
    else:
        # Multiple axes: reduce one at a time
        result = arr
        # Sort axes in descending order to maintain valid indices
        for ax in sorted(axis, reverse=True):
            result = _sum_axis(result, ax, keepdims)
        return result


def prod(arr, axis=None, dtype=None, keepdims: bool = False, policy: str = _DEFAULT_POLICY):
    """Product of array elements.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to compute product.
    dtype : numpy.dtype, optional
        Type to use for accumulation.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq" (sequential), "par" (parallel),
        or "par_unseq" (parallel + SIMD).

    Returns
    -------
    ndarray or scalar
        Product of elements.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([1.0, 2.0, 3.0, 4.0])
    >>> hpx.prod(arr)
    24.0
    >>> hpx.finalize()
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter for prod() not yet implemented")
    if keepdims:
        raise NotImplementedError("keepdims parameter for prod() not yet implemented")
    return _prod(arr, policy)


def min(arr, axis=None, keepdims: bool = False, policy: str = _DEFAULT_POLICY):
    """Minimum of array elements.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to find minimum.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq" (sequential), "par" (parallel),
        or "par_unseq" (parallel + SIMD).

    Returns
    -------
    ndarray or scalar
        Minimum value.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([3.0, 1.0, 4.0, 1.0, 5.0])
    >>> hpx.min(arr)
    1.0
    >>> hpx.finalize()
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter for min() not yet implemented")
    if keepdims:
        raise NotImplementedError("keepdims parameter for min() not yet implemented")
    return _min(arr, policy)


def max(arr, axis=None, keepdims: bool = False, policy: str = _DEFAULT_POLICY):
    """Maximum of array elements.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to find maximum.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq" (sequential), "par" (parallel),
        or "par_unseq" (parallel + SIMD).

    Returns
    -------
    ndarray or scalar
        Maximum value.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([3.0, 1.0, 4.0, 1.0, 5.0])
    >>> hpx.max(arr)
    5.0
    >>> hpx.finalize()
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter for max() not yet implemented")
    if keepdims:
        raise NotImplementedError("keepdims parameter for max() not yet implemented")
    return _max(arr, policy)


def mean(arr, axis=None, dtype=None, keepdims: bool = False, policy: str = _DEFAULT_POLICY):
    """Compute the arithmetic mean.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to compute mean.
    dtype : numpy.dtype, optional
        Type to use for computation.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq" (sequential), "par" (parallel),
        or "par_unseq" (parallel + SIMD).

    Returns
    -------
    ndarray or scalar
        Mean of elements.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported in Phase 1")
    return _sum(arr, policy) / arr.size


def std(arr, axis=None, dtype=None, ddof: int = 0, keepdims: bool = False, policy: str = _DEFAULT_POLICY):
    """Compute the standard deviation.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to compute std.
    dtype : numpy.dtype, optional
        Type to use for computation.
    ddof : int, default 0
        Delta degrees of freedom.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq" (sequential), "par" (parallel),
        or "par_unseq" (parallel + SIMD).

    Returns
    -------
    ndarray or scalar
        Standard deviation.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported in Phase 1")
    return var(arr, ddof=ddof, policy=policy) ** 0.5


def var(arr, axis=None, dtype=None, ddof: int = 0, keepdims: bool = False, policy: str = _DEFAULT_POLICY):
    """Compute the variance.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to compute variance.
    dtype : numpy.dtype, optional
        Type to use for computation.
    ddof : int, default 0
        Delta degrees of freedom.
    keepdims : bool, default False
        If True, retain reduced dimensions with size 1.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq" (sequential), "par" (parallel),
        or "par_unseq" (parallel + SIMD).

    Returns
    -------
    ndarray or scalar
        Variance.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported in Phase 1")
    # Compute variance using pure HPX operations
    # Uses two-pass algorithm for numerical stability: Var(X) = E[(X - mean)^2]
    n = arr.size
    mean_val = _sum(arr, policy) / n
    # Element-wise operations use parallel execution automatically
    diff = arr - mean_val        # HPX parallel element-wise subtract
    squared = diff * diff        # HPX parallel element-wise multiply
    sum_sq = _sum(squared, policy)  # HPX reduction with specified policy
    return sum_sq / (n - ddof)


# -----------------------------------------------------------------------------
# Sorting Algorithms
# -----------------------------------------------------------------------------


def sort(arr, axis: int = -1, policy: str = _DEFAULT_POLICY) -> ndarray:
    """Sort an array.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : int, default -1
        Axis along which to sort. Default is -1 (last axis).
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq" (sequential), "par" (parallel),
        or "par_unseq" (parallel + SIMD).

    Returns
    -------
    ndarray
        Sorted array.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([3.0, 1.0, 4.0, 1.0, 5.0])
    >>> hpx.sort(arr)
    array([1., 1., 3., 4., 5.])
    >>> hpx.sort(arr, policy="par")  # Multi-threaded for large arrays
    array([1., 1., 3., 4., 5.])
    >>> hpx.finalize()
    """
    _check_available()
    # Phase 1: only support 1D arrays
    if arr.ndim != 1:
        raise NotImplementedError("Multi-dimensional sort not yet supported in Phase 1")
    return _sort(arr, policy)


def argsort(arr, axis: int = -1, policy: str = _DEFAULT_POLICY) -> ndarray:
    """Return indices that would sort an array.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : int, default -1
        Axis along which to sort. Currently only -1 (last axis) is supported for 1D.
    policy : str, optional
        Execution policy: "seq" (sequential), "par" (parallel),
        or "par_unseq" (parallel unsequenced).

    Returns
    -------
    ndarray
        Array of indices that sort the input.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([3.0, 1.0, 4.0, 1.0, 5.0])
    >>> indices = hpx.argsort(arr)
    >>> indices
    array([1, 3, 0, 2, 4])
    >>> arr[indices]  # Sorted array
    array([1., 1., 3., 4., 5.])
    >>> hpx.finalize()
    """
    _check_available()
    if arr.ndim != 1:
        # Multi-dimensional: fall back to NumPy for now
        import numpy as np
        np_arr = arr.to_numpy()
        indices = np.argsort(np_arr, axis=axis)
        return from_numpy(indices, copy=True)

    # 1D: use HPX parallel sort
    return _argsort(arr, policy)


def count(arr, value, policy: str = _DEFAULT_POLICY) -> int:
    """Count occurrences of a value.

    Parameters
    ----------
    arr : ndarray
        Input array.
    value : scalar
        Value to count.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq" (sequential), "par" (parallel),
        or "par_unseq" (parallel + SIMD).

    Returns
    -------
    int
        Number of occurrences.
    """
    _check_available()
    return _count(arr, value, policy)


# -----------------------------------------------------------------------------
# New HPX Algorithm Exposures
# -----------------------------------------------------------------------------


def any(arr, axis=None, policy: str = _DEFAULT_POLICY) -> bool:
    """Check if any element is truthy (non-zero).

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None
        Not yet supported, must be None.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    bool
        True if any element is non-zero.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported")
    return _any(arr, policy)


def all(arr, axis=None, policy: str = _DEFAULT_POLICY) -> bool:
    """Check if all elements are truthy (non-zero).

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None
        Not yet supported, must be None.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    bool
        True if all elements are non-zero.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported")
    return _all(arr, policy)


def argmin(arr, axis=None, policy: str = _DEFAULT_POLICY):
    """Return index of minimum element.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None
        Not yet supported, must be None.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    int
        Index of minimum element.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported")
    return _argmin(arr, policy)


def argmax(arr, axis=None, policy: str = _DEFAULT_POLICY):
    """Return index of maximum element.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : None
        Not yet supported, must be None.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    int
        Index of maximum element.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported")
    return _argmax(arr, policy)


def diff(arr, n: int = 1, axis: int = -1, policy: str = _DEFAULT_POLICY) -> ndarray:
    """Calculate n-th discrete difference along given axis.

    Parameters
    ----------
    arr : ndarray
        Input array.
    n : int, default 1
        Number of times values are differenced.
    axis : int, default -1
        Axis along which to compute difference. Only -1 (last axis) supported.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    ndarray
        Array of differences with shape reduced by n along axis.
    """
    _check_available()
    if axis != -1 and axis != arr.ndim - 1:
        raise NotImplementedError("Only last axis (-1) supported")
    if n != 1:
        raise NotImplementedError("Only n=1 supported")
    return _diff(arr, policy)


def unique(arr, policy: str = _DEFAULT_POLICY) -> ndarray:
    """Return sorted unique elements.

    Parameters
    ----------
    arr : ndarray
        Input array.
    policy : str, default "par_unseq" (set by CMake)
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    ndarray
        Sorted unique elements.
    """
    _check_available()
    return _unique(arr, policy)


def inclusive_scan(arr, op: str = "add", policy: str = "par") -> ndarray:
    """Inclusive scan (running accumulation).

    This is a direct exposure of HPX's inclusive_scan algorithm.

    Parameters
    ----------
    arr : ndarray
        Input array.
    op : str, default "add"
        Binary operation: "add", "mul", "min", "max".
    policy : str, default "par_unseq"
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    ndarray
        Scanned array where result[i] = op(arr[0], ..., arr[i]).

    Examples
    --------
    >>> arr = hpx.arange(5)
    >>> hpx.inclusive_scan(arr, "add")  # cumsum: [0, 1, 3, 6, 10]
    >>> hpx.inclusive_scan(arr, "mul")  # cumprod: [0, 0, 0, 0, 0]
    >>> hpx.inclusive_scan(arr, "max")  # running max: [0, 1, 2, 3, 4]
    """
    _check_available()
    return _inclusive_scan(arr, op, policy)


def exclusive_scan(arr, init, op: str = "add", policy: str = "par") -> ndarray:
    """Exclusive scan (running accumulation excluding current element).

    This is a direct exposure of HPX's exclusive_scan algorithm.

    Parameters
    ----------
    arr : ndarray
        Input array.
    init : scalar
        Initial value.
    op : str, default "add"
        Binary operation: "add", "mul".
    policy : str, default "par_unseq"
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    ndarray
        Scanned array where result[i] = op(init, arr[0], ..., arr[i-1]).

    Examples
    --------
    >>> arr = hpx.arange(5)
    >>> hpx.exclusive_scan(arr, 0, "add")  # [0, 0, 1, 3, 6]
    """
    _check_available()
    return _exclusive_scan(arr, init, op, policy)


def transform_reduce(arr, transform: str, reduce: str = "add", policy: str = "par"):
    """Fused transform and reduce in one pass.

    This is a direct exposure of HPX's transform_reduce algorithm, which
    combines a unary transform with a binary reduction in a single pass
    over the data for better performance.

    Parameters
    ----------
    arr : ndarray
        Input array.
    transform : str
        Unary transform: "square", "abs", "negate", "identity".
    reduce : str, default "add"
        Binary reduction: "add", "mul", "min", "max".
    policy : str, default "par_unseq"
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    scalar
        Result of reduce(transform(arr[0]), transform(arr[1]), ...).

    Examples
    --------
    >>> arr = hpx.arange(5)
    >>> hpx.transform_reduce(arr, "square", "add")  # sum of squares
    >>> hpx.transform_reduce(arr, "abs", "add")     # sum of absolute values
    >>> hpx.transform_reduce(arr, "identity", "max") # max element
    """
    _check_available()
    return _transform_reduce(arr, transform, reduce, policy)


def reduce_by_key(
    keys: ndarray,
    values: ndarray,
    reduce_op: str = "add",
    policy: str = "par",
) -> tuple:
    """Reduce values grouped by keys.

    Performs a reduction operation on values that share the same key.
    Returns the unique keys and their corresponding reduced values.

    Parameters
    ----------
    keys : ndarray
        Key array (float64, 1D).
    values : ndarray
        Value array (float64, 1D, same size as keys).
    reduce_op : str, default "add"
        Reduction operation: "add", "mul", "min", "max".
    policy : str, default "par_unseq"
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    tuple[ndarray, ndarray]
        (unique_keys, reduced_values) where unique_keys are sorted.

    Examples
    --------
    >>> keys = hpx.array([1.0, 2.0, 1.0, 2.0, 1.0])
    >>> values = hpx.array([10.0, 20.0, 30.0, 40.0, 50.0])
    >>> unique_keys, reduced_values = hpx.reduce_by_key(keys, values, "add")
    >>> # unique_keys = [1.0, 2.0]
    >>> # reduced_values = [90.0, 60.0]  (1: 10+30+50=90, 2: 20+40=60)
    """
    _check_available()
    return _reduce_by_key(keys, values, reduce_op, policy)


def reduce(arr, op: str = "add", policy: str = "par"):
    """General reduction with custom binary operation.

    Parameters
    ----------
    arr : ndarray
        Input array (float64).
    op : str, default "add"
        Binary operation: "add", "mul", "min", "max".
    policy : str, default "par_unseq"
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    scalar
        Result of op(op(op(arr[0], arr[1]), arr[2]), ...).

    Examples
    --------
    >>> arr = hpx.arange(5)
    >>> hpx.reduce(arr, "add")  # sum: 0+1+2+3+4=10
    >>> hpx.reduce(arr, "mul")  # product (NOTE: 0 in array!)
    >>> hpx.reduce(arr, "max")  # max element

    Note
    ----
    This is a convenience wrapper around transform_reduce with identity.
    """
    _check_available()
    return _transform_reduce(arr, "identity", op, policy)


def reduce_deterministic(arr, op: str = "add"):
    """Deterministic reduction with reproducible floating-point results.

    Performs reduction in sequential order to guarantee identical results
    across different runs. This is slower than parallel reduction but
    necessary when exact reproducibility is required.

    Parameters
    ----------
    arr : ndarray
        Input array (float64).
    op : str, default "add"
        Binary operation: "add", "mul", "min", "max".

    Returns
    -------
    scalar
        Result of op(op(op(arr[0], arr[1]), arr[2]), ...).

    Examples
    --------
    >>> arr = hpx.array([1e20, 1.0, -1e20, 1.0])
    >>> hpx.reduce(arr, "add", policy="par")  # May give 2.0 or 0.0
    >>> hpx.reduce_deterministic(arr, "add")  # Always same result

    Note
    ----
    Parallel floating-point reductions can produce different results due
    to non-associativity of floating-point addition. This function uses
    sequential execution to guarantee reproducibility at the cost of
    parallelism.
    """
    _check_available()
    return _transform_reduce(arr, "identity", op, "seq")


# -----------------------------------------------------------------------------
# Tier 2 Algorithms
# -----------------------------------------------------------------------------


def array_equal(arr1, arr2, policy: str = "par") -> bool:
    """True if two arrays have the same shape and elements.

    Parameters
    ----------
    arr1, arr2 : ndarray
        Input arrays to compare (float64).
    policy : str, default "par_unseq"
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    bool
        True if arrays are equal.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> a = hpx.array([1.0, 2.0, 3.0])
    >>> b = hpx.array([1.0, 2.0, 3.0])
    >>> hpx.array_equal(a, b)
    True
    >>> c = hpx.array([1.0, 2.0, 4.0])
    >>> hpx.array_equal(a, c)
    False
    >>> hpx.finalize()
    """
    _check_available()
    return _array_equal(arr1, arr2, policy)


def flip(arr, policy: str = "par") -> ndarray:
    """Reverse the order of elements in array.

    Parameters
    ----------
    arr : ndarray
        Input array (1D float64).
    policy : str, default "par_unseq"
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    ndarray
        Reversed array.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> hpx.flip(arr)
    array([5., 4., 3., 2., 1.])
    >>> hpx.finalize()
    """
    _check_available()
    return _flip(arr, policy)


def stable_sort(arr, policy: str = "par") -> ndarray:
    """Sort array with stable ordering (preserves relative order of equals).

    Unlike regular sort, stable_sort guarantees that elements with equal values
    maintain their original relative order.

    Parameters
    ----------
    arr : ndarray
        Input array (1D float64).
    policy : str, default "par_unseq"
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    ndarray
        Sorted array.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([3.0, 1.0, 4.0, 1.0, 5.0])
    >>> hpx.stable_sort(arr)
    array([1., 1., 3., 4., 5.])
    >>> hpx.finalize()
    """
    _check_available()
    return _stable_sort(arr, policy)


def roll(arr, shift: int, policy: str = "par") -> ndarray:
    """Roll array elements along first axis.

    Elements that roll beyond the end appear at the beginning.

    Parameters
    ----------
    arr : ndarray
        Input array (1D float64).
    shift : int
        Number of positions to roll. Positive shifts elements left (rotate left).
    policy : str, default "par_unseq"
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    ndarray
        Rolled array.

    Note
    ----
    Uses HPX rotate for parallel rotation.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> hpx.roll(arr, 2)  # Rotate left by 2
    array([3., 4., 5., 1., 2.])
    >>> hpx.roll(arr, -1)  # Rotate right by 1
    array([5., 1., 2., 3., 4.])
    >>> hpx.finalize()
    """
    _check_available()
    return _rotate(arr, shift, policy)


def nonzero(arr) -> ndarray:
    """Return indices of non-zero elements.

    Parameters
    ----------
    arr : ndarray
        Input array (1D float64).

    Returns
    -------
    ndarray
        Array of indices where arr is non-zero (int64).

    Note
    ----
    Currently sequential to maintain deterministic ordering.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
    >>> hpx.nonzero(arr)
    array([1, 3, 5])
    >>> hpx.finalize()
    """
    _check_available()
    return _nonzero(arr, "seq")


def nth_element(arr, n: int, policy: str = "par"):
    """Find the nth element as if array were sorted.

    Uses partial sort (std::nth_element) which is O(n) average complexity,
    much faster than full sorting for finding a single order statistic.

    Parameters
    ----------
    arr : ndarray
        Input array (1D float64).
    n : int
        Index of element to find (0-indexed, negative indices supported).
    policy : str, default "par_unseq"
        Execution policy (currently uses std::nth_element).

    Returns
    -------
    scalar
        The nth smallest element.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
    >>> hpx.nth_element(arr, 0)  # Minimum
    1.0
    >>> hpx.nth_element(arr, 3)  # 4th smallest (0-indexed)
    3.0
    >>> hpx.nth_element(arr, -1)  # Maximum
    9.0
    >>> hpx.finalize()
    """
    _check_available()
    return _nth_element(arr, n, policy)


def median(arr):
    """Compute the median of array elements.

    Uses nth_element for O(n) average complexity, making it faster than
    sorting-based median implementations for large arrays.

    Parameters
    ----------
    arr : ndarray
        Input array (1D float64).

    Returns
    -------
    scalar
        Median value. For even-length arrays, returns average of middle two.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([1.0, 3.0, 5.0, 7.0, 9.0])
    >>> hpx.median(arr)
    5.0
    >>> arr2 = hpx.array([1.0, 2.0, 3.0, 4.0])
    >>> hpx.median(arr2)  # Average of 2.0 and 3.0
    2.5
    >>> hpx.finalize()
    """
    _check_available()
    return _median(arr)


def percentile(arr, q: float):
    """Compute the q-th percentile of array elements.

    Parameters
    ----------
    arr : ndarray
        Input array (1D float64).
    q : float
        Percentile to compute, in range [0, 100].

    Returns
    -------
    scalar
        Percentile value with linear interpolation (numpy default method).

    Examples
    --------
    >>> arr = hpx.arange(100)
    >>> hpx.percentile(arr, 50)  # Same as median
    >>> hpx.percentile(arr, 25)  # First quartile
    >>> hpx.percentile(arr, 75)  # Third quartile
    """
    _check_available()
    return _percentile(arr, q)


# -----------------------------------------------------------------------------
# Tier 3 Algorithms
# -----------------------------------------------------------------------------


def merge_sorted(arr1, arr2, policy: str = "par") -> ndarray:
    """Merge two sorted arrays into a single sorted array.

    Both input arrays must be sorted. The result is a single sorted array
    containing all elements from both inputs.

    Parameters
    ----------
    arr1, arr2 : ndarray
        Sorted input arrays (1D float64).
    policy : str, default "par_unseq"
        Execution policy: "seq", "par", or "par_unseq".

    Returns
    -------
    ndarray
        Merged sorted array.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> a = hpx.array([1.0, 3.0, 5.0])
    >>> b = hpx.array([2.0, 4.0, 6.0])
    >>> hpx.merge_sorted(a, b)
    array([1., 2., 3., 4., 5., 6.])
    >>> hpx.finalize()
    """
    _check_available()
    return _merge(arr1, arr2, policy)


def setdiff1d(arr1, arr2, policy: str = "par") -> ndarray:
    """Set difference: elements in arr1 but not in arr2.

    Parameters
    ----------
    arr1, arr2 : ndarray
        Input arrays (1D float64).
    policy : str, default "par_unseq"
        Execution policy.

    Returns
    -------
    ndarray
        Unique values in arr1 that are not in arr2, sorted.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> a = hpx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> b = hpx.array([2.0, 4.0])
    >>> hpx.setdiff1d(a, b)  # Elements in a but not in b
    array([1., 3., 5.])
    >>> hpx.finalize()
    """
    _check_available()
    return _setdiff1d(arr1, arr2, policy)


def intersect1d(arr1, arr2, policy: str = "par") -> ndarray:
    """Set intersection: unique elements in both arr1 and arr2.

    Parameters
    ----------
    arr1, arr2 : ndarray
        Input arrays (1D float64).
    policy : str, default "par_unseq"
        Execution policy.

    Returns
    -------
    ndarray
        Sorted unique values present in both arrays.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> a = hpx.array([1.0, 2.0, 3.0, 4.0])
    >>> b = hpx.array([2.0, 4.0, 5.0, 6.0])
    >>> hpx.intersect1d(a, b)  # Elements in both a and b
    array([2., 4.])
    >>> hpx.finalize()
    """
    _check_available()
    return _intersect1d(arr1, arr2, policy)


def union1d(arr1, arr2, policy: str = "par") -> ndarray:
    """Set union: unique elements in arr1 or arr2 (or both).

    Parameters
    ----------
    arr1, arr2 : ndarray
        Input arrays (1D float64).
    policy : str, default "par_unseq"
        Execution policy.

    Returns
    -------
    ndarray
        Sorted unique values from either array.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> a = hpx.array([1.0, 2.0, 3.0])
    >>> b = hpx.array([2.0, 4.0, 5.0])
    >>> hpx.union1d(a, b)  # All unique elements from a and b
    array([1., 2., 3., 4., 5.])
    >>> hpx.finalize()
    """
    _check_available()
    return _union1d(arr1, arr2, policy)


def setxor1d(arr1, arr2, policy: str = "par") -> ndarray:
    """Set symmetric difference: elements in arr1 or arr2 but not both.

    Parameters
    ----------
    arr1, arr2 : ndarray
        Input arrays (1D float64).
    policy : str, default "par_unseq"
        Execution policy.

    Returns
    -------
    ndarray
        Sorted unique values in exactly one of the arrays.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> a = hpx.array([1.0, 2.0, 3.0])
    >>> b = hpx.array([2.0, 3.0, 4.0])
    >>> hpx.setxor1d(a, b)  # Elements in a or b, but not both
    array([1., 4.])
    >>> hpx.finalize()
    """
    _check_available()
    return _setxor1d(arr1, arr2, policy)


def includes(arr1, arr2, policy: str = "par") -> bool:
    """Test if arr1 contains all elements of arr2.

    Parameters
    ----------
    arr1, arr2 : ndarray
        Input arrays (1D float64).
    policy : str, default "par_unseq"
        Execution policy.

    Returns
    -------
    bool
        True if every element in arr2 is also in arr1.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> a = hpx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> b = hpx.array([2.0, 4.0])
    >>> hpx.includes(a, b)  # Does a contain all elements of b?
    True
    >>> c = hpx.array([2.0, 6.0])
    >>> hpx.includes(a, c)  # Does a contain all elements of c?
    False
    >>> hpx.finalize()
    """
    _check_available()
    return _includes(arr1, arr2, policy)


def isin(arr, test_arr) -> ndarray:
    """Test membership of arr elements in test_arr.

    Parameters
    ----------
    arr : ndarray
        Input array to check (1D float64).
    test_arr : ndarray
        Values to check against (1D float64).

    Returns
    -------
    ndarray
        Boolean array of same shape as arr, True where element is in test_arr.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> test = hpx.array([2.0, 4.0])
    >>> hpx.isin(arr, test)  # Which elements of arr are in test?
    array([False, True, False, True, False])
    >>> hpx.finalize()
    """
    _check_available()
    return _isin(arr, test_arr)


def searchsorted(arr, values, side: str = "left") -> ndarray:
    """Find indices where values should be inserted in sorted arr.

    Uses binary search to find insertion points efficiently (O(log n) per value).

    Parameters
    ----------
    arr : ndarray
        Sorted input array (1D float64).
    values : ndarray
        Values to insert (1D float64).
    side : str, default "left"
        "left": first suitable position, "right": last suitable position.

    Returns
    -------
    ndarray
        Indices (int64) where values should be inserted.

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> values = hpx.array([2.5, 0.5, 5.5])
    >>> hpx.searchsorted(arr, values)  # Where to insert each value
    array([2, 0, 5])
    >>> hpx.searchsorted(arr, values, side="right")
    array([2, 0, 5])
    >>> hpx.finalize()
    """
    _check_available()
    return _searchsorted(arr, values, side)


def partition(arr, pivot: float) -> ndarray:
    """Partition array around a pivot value.

    Rearranges elements so that all elements less than pivot come before
    elements greater than or equal to pivot. Useful for implementing quicksort
    variants or filtering data.

    Parameters
    ----------
    arr : ndarray
        Input array (1D float64).
    pivot : float
        Pivot value.

    Returns
    -------
    ndarray
        Partitioned array (elements < pivot come first).

    Examples
    --------
    >>> import hpxpy as hpx
    >>> hpx.init()
    >>> arr = hpx.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
    >>> hpx.partition(arr, 4.0)  # Elements < 4 come first
    array([3., 1., 1., 2., ...])  # Order within partitions may vary
    >>> hpx.finalize()
    """
    _check_available()
    return _partition(arr, pivot)


# -----------------------------------------------------------------------------
# Math Functions (Phase 2)
# -----------------------------------------------------------------------------


def sqrt(arr) -> ndarray:
    """Element-wise square root."""
    _check_available()
    return _sqrt(arr)


def square(arr) -> ndarray:
    """Element-wise square."""
    _check_available()
    return _square(arr)


def exp(arr) -> ndarray:
    """Element-wise exponential."""
    _check_available()
    return _exp(arr)


def exp2(arr) -> ndarray:
    """Element-wise 2**x."""
    _check_available()
    return _exp2(arr)


def log(arr) -> ndarray:
    """Element-wise natural logarithm."""
    _check_available()
    return _log(arr)


def log2(arr) -> ndarray:
    """Element-wise base-2 logarithm."""
    _check_available()
    return _log2(arr)


def log10(arr) -> ndarray:
    """Element-wise base-10 logarithm."""
    _check_available()
    return _log10(arr)


def sin(arr) -> ndarray:
    """Element-wise sine."""
    _check_available()
    return _sin(arr)


def cos(arr) -> ndarray:
    """Element-wise cosine."""
    _check_available()
    return _cos(arr)


def tan(arr) -> ndarray:
    """Element-wise tangent."""
    _check_available()
    return _tan(arr)


def arcsin(arr) -> ndarray:
    """Element-wise inverse sine."""
    _check_available()
    return _arcsin(arr)


def arccos(arr) -> ndarray:
    """Element-wise inverse cosine."""
    _check_available()
    return _arccos(arr)


def arctan(arr) -> ndarray:
    """Element-wise inverse tangent."""
    _check_available()
    return _arctan(arr)


def sinh(arr) -> ndarray:
    """Element-wise hyperbolic sine."""
    _check_available()
    return _sinh(arr)


def cosh(arr) -> ndarray:
    """Element-wise hyperbolic cosine."""
    _check_available()
    return _cosh(arr)


def tanh(arr) -> ndarray:
    """Element-wise hyperbolic tangent."""
    _check_available()
    return _tanh(arr)


def floor(arr) -> ndarray:
    """Element-wise floor."""
    _check_available()
    return _floor(arr)


def ceil(arr) -> ndarray:
    """Element-wise ceiling."""
    _check_available()
    return _ceil(arr)


def trunc(arr) -> ndarray:
    """Element-wise truncation toward zero."""
    _check_available()
    return _trunc(arr)


def abs(arr) -> ndarray:
    """Element-wise absolute value."""
    _check_available()
    return _abs(arr)


def sign(arr) -> ndarray:
    """Element-wise sign indicator (-1, 0, or 1)."""
    _check_available()
    return _sign(arr)


# -----------------------------------------------------------------------------
# Scan Operations (Phase 2)
# -----------------------------------------------------------------------------


def cumsum(arr, axis=None) -> ndarray:
    """Cumulative sum of array elements.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : int, optional
        Axis along which to compute cumulative sum.
        If None, compute over flattened array.

    Returns
    -------
    ndarray
        Cumulative sum.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported")
    return _cumsum(arr)


def cumprod(arr, axis=None) -> ndarray:
    """Cumulative product of array elements.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : int, optional
        Axis along which to compute cumulative product.
        If None, compute over flattened array.

    Returns
    -------
    ndarray
        Cumulative product.
    """
    _check_available()
    if axis is not None:
        raise NotImplementedError("axis parameter not yet supported")
    return _cumprod(arr)


# -----------------------------------------------------------------------------
# Element-wise Functions (Phase 2)
# -----------------------------------------------------------------------------


def maximum(a, b) -> ndarray:
    """Element-wise maximum of two arrays.

    Parameters
    ----------
    a : ndarray
        First input array.
    b : ndarray or scalar
        Second input array or scalar.

    Returns
    -------
    ndarray
        Element-wise maximum.
    """
    _check_available()
    return _maximum(a, b)


def minimum(a, b) -> ndarray:
    """Element-wise minimum of two arrays.

    Parameters
    ----------
    a : ndarray
        First input array.
    b : ndarray or scalar
        Second input array or scalar.

    Returns
    -------
    ndarray
        Element-wise minimum.
    """
    _check_available()
    return _minimum(a, b)


def clip(arr, a_min, a_max) -> ndarray:
    """Clip (limit) the values in an array.

    Parameters
    ----------
    arr : ndarray
        Input array.
    a_min : scalar
        Minimum value.
    a_max : scalar
        Maximum value.

    Returns
    -------
    ndarray
        Clipped array.
    """
    _check_available()
    return _clip(arr, a_min, a_max)


def power(arr, exponent) -> ndarray:
    """Element-wise power.

    Parameters
    ----------
    arr : ndarray
        Base array.
    exponent : ndarray or scalar
        Exponent array or scalar.

    Returns
    -------
    ndarray
        arr ** exponent element-wise.
    """
    _check_available()
    return _power(arr, exponent)


def where(condition, x, y) -> ndarray:
    """Return elements chosen from x or y depending on condition.

    Parameters
    ----------
    condition : ndarray of bool
        Where True, yield x, otherwise yield y.
    x : ndarray
        Values to use where condition is True.
    y : ndarray
        Values to use where condition is False.

    Returns
    -------
    ndarray
        Array with elements from x where condition is True, else from y.
    """
    _check_available()
    return _where(condition, x, y)


# -----------------------------------------------------------------------------
# Random Number Generation (Phase 2)
# -----------------------------------------------------------------------------


class random:
    """Random number generation module.

    Provides NumPy-like random number generation functions.
    """

    @staticmethod
    def seed(s: int) -> None:
        """Seed the random number generator.

        Parameters
        ----------
        s : int
            Seed value.
        """
        _check_available()
        _random_module.seed(s)

    @staticmethod
    def uniform(low: float = 0.0, high: float = 1.0, size=None) -> ndarray:
        """Draw samples from a uniform distribution.

        Parameters
        ----------
        low : float, default 0.0
            Lower boundary.
        high : float, default 1.0
            Upper boundary.
        size : int or tuple of ints, optional
            Output shape. Default is a single value.

        Returns
        -------
        ndarray
            Samples from uniform distribution.
        """
        _check_available()
        if size is None:
            size = (1,)
        elif isinstance(size, int):
            size = (size,)
        return _random_module._uniform(low, high, size)

    @staticmethod
    def randn(*shape) -> ndarray:
        """Return samples from the standard normal distribution.

        Parameters
        ----------
        *shape : ints
            Shape of the output array.

        Returns
        -------
        ndarray
            Samples from standard normal distribution.

        Examples
        --------
        >>> hpx.random.randn(3, 4)  # 3x4 array of standard normal samples
        """
        _check_available()
        if len(shape) == 0:
            shape = (1,)
        return _random_module._randn(list(shape))

    @staticmethod
    def randint(low: int, high: int = None, size=None) -> ndarray:
        """Return random integers from low (inclusive) to high (exclusive).

        Parameters
        ----------
        low : int
            Lowest integer (or highest if high is None).
        high : int, optional
            One above the highest integer.
        size : int or tuple of ints, optional
            Output shape.

        Returns
        -------
        ndarray
            Random integers.
        """
        _check_available()
        if high is None:
            high = low
            low = 0
        if size is None:
            size = (1,)
        elif isinstance(size, int):
            size = (size,)
        return _random_module._randint(low, high, size)

    @staticmethod
    def rand(*shape) -> ndarray:
        """Random values in a given shape from uniform [0, 1).

        Parameters
        ----------
        *shape : ints
            Shape of the output array.

        Returns
        -------
        ndarray
            Random values.

        Examples
        --------
        >>> hpx.random.rand(3, 4)  # 3x4 array of uniform random values
        """
        _check_available()
        if len(shape) == 0:
            shape = (1,)
        return _random_module._rand(list(shape))


# -----------------------------------------------------------------------------
# Collective Operations (Phase 4)
# -----------------------------------------------------------------------------


def all_reduce(arr, op: str = 'sum'):
    """Combine values from all localities using a reduction operation.

    Each locality contributes its local array, and all localities receive
    the combined result. In single-locality mode, returns the input unchanged.

    Parameters
    ----------
    arr : ndarray
        Local array to contribute.
    op : str, optional
        Reduction operation: 'sum', 'prod', 'min', 'max'. Default: 'sum'.

    Returns
    -------
    ndarray
        Combined result (same on all localities).

    Examples
    --------
    >>> local_sum = hpx.from_numpy(np.array([1.0, 2.0, 3.0]))
    >>> global_sum = hpx.all_reduce(local_sum, op='sum')
    """
    _check_available()
    from hpxpy._core import ReduceOp
    op_map = {
        'sum': ReduceOp.sum,
        'prod': ReduceOp.prod,
        'min': ReduceOp.min,
        'max': ReduceOp.max,
    }
    if op not in op_map:
        raise ValueError(f"Unknown reduction operation: {op}. Use one of: {list(op_map.keys())}")
    return _all_reduce(arr, op_map[op])


def broadcast(arr, root: int = 0):
    """Broadcast array from root locality to all localities.

    In single-locality mode, returns a copy of the input.

    Parameters
    ----------
    arr : ndarray
        Array to broadcast (only used on root locality).
    root : int, optional
        Locality ID to broadcast from. Default: 0.

    Returns
    -------
    ndarray
        Broadcasted array (same on all localities).

    Examples
    --------
    >>> data = hpx.from_numpy(np.array([1.0, 2.0, 3.0]))
    >>> shared = hpx.broadcast(data, root=0)
    """
    _check_available()
    return _broadcast(arr, root)


def gather(arr, root: int = 0):
    """Gather arrays from all localities to root.

    In single-locality mode, returns a list containing just the input.

    Parameters
    ----------
    arr : ndarray
        Local array to contribute.
    root : int, optional
        Locality ID to gather to. Default: 0.

    Returns
    -------
    list
        List of numpy arrays from all localities (only valid on root).

    Examples
    --------
    >>> local_data = hpx.from_numpy(np.array([1.0, 2.0, 3.0]))
    >>> all_data = hpx.gather(local_data, root=0)
    >>> print(len(all_data))  # Number of localities
    """
    _check_available()
    return _gather(arr, root)


def scatter(arr, root: int = 0):
    """Scatter array from root to all localities.

    The array on root is divided evenly among all localities.
    In single-locality mode, returns a copy of the input.

    Parameters
    ----------
    arr : ndarray
        Array to scatter (only used on root).
    root : int, optional
        Locality ID to scatter from. Default: 0.

    Returns
    -------
    ndarray
        This locality's portion of the scattered array.

    Examples
    --------
    >>> full_data = hpx.arange(1000)
    >>> local_chunk = hpx.scatter(full_data, root=0)
    """
    _check_available()
    return _scatter(arr, root)


def barrier(name: str = "hpxpy_barrier"):
    """Synchronize all localities.

    All localities wait until everyone reaches the barrier.
    In single-locality mode, this is a no-op.

    Parameters
    ----------
    name : str, optional
        Name for the barrier. Default: "hpxpy_barrier".

    Examples
    --------
    >>> # Ensure all localities have finished computation
    >>> hpx.barrier()
    """
    _check_available()
    _barrier(name)


# -----------------------------------------------------------------------------
# Multi-Locality Launcher (Phase 4)
# -----------------------------------------------------------------------------

# Import the launcher module for multi-locality support
from hpxpy import launcher


# -----------------------------------------------------------------------------
# GPU Support (Phase 5)
# -----------------------------------------------------------------------------

# Import the GPU module for CUDA/GPU support (via HPX cuda_executor)
from hpxpy import gpu

# Import the SYCL module for cross-platform GPU support (via HPX sycl_executor)
# Supports: Intel GPUs (oneAPI), AMD GPUs (HIP), Apple Silicon (AdaptiveCpp Metal)
from hpxpy import sycl
