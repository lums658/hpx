# HPXPy Future Work

Deferred features and enhancements for future phases.

## Phase 7: Array Indexing & Assignment (COMPLETED)

### Integer Indexing
- [x] `arr[5]` - Get single element
- [x] `arr[-1]` - Negative index support
- [x] Return scalar for 1D, sub-array view for ND

### __setitem__ Assignment
- [x] `arr[0] = 42` - Scalar assignment
- [x] `arr[1:4] = 7` - Slice assignment with scalar broadcast
- [x] `arr[1:4] = other_arr` - Slice assignment with array

## Phase 8: Multi-dimensional Slicing (COMPLETED)

### Multi-dimensional Slicing
- [x] `arr[1:3, 2:5]` - Multi-axis slicing
- [x] Tuple handling in __getitem__
- [x] Multi-axis __setitem__ support
- [x] Mixed integer/slice indices (e.g., `arr[1, 2:5]`)
- [x] Negative indices in multi-dim

### Advanced Shape Operations (Deferred)
- Transpose support
- Squeeze/expand_dims

## Phase 9: Broadcasting (COMPLETED)

### Binary Operation Broadcasting
- [x] NumPy-compatible broadcasting for binary operations
- [x] Shape compatibility checking (`broadcast_shapes()`, `can_broadcast()`)
- [x] Broadcasting for all arithmetic operators (+, -, *, /, **)
- [x] Broadcasting for comparison operators (==, !=, <, <=, >, >=)
- [x] Fast path for same-shape arrays (no broadcasting overhead)

### Tutorials Added
- [x] Monte Carlo simulations (08_monte_carlo.ipynb)
- [x] Black-Scholes option pricing (09_black_scholes.ipynb)
- [x] Image processing (10_image_processing.ipynb)

### Advanced Indexing (Deferred)
- Boolean mask indexing: `arr[arr > 5]`
- Fancy indexing: `arr[[1, 3, 5]]`
- Integer array indexing

## Phase 11: True Distribution (COMPLETED)

### partitioned_vector Integration
- [x] `PartitionedArray` class using HPX `partitioned_vector<double>`
- [x] Factory functions: `partitioned_zeros`, `partitioned_ones`, `partitioned_full`, `partitioned_arange`
- [x] `partitioned_from_numpy` for creating from NumPy arrays
- [x] Auto-distribution across all available HPX localities

### Distributed Algorithms
- [x] Distributed sum with `hpx::reduce` (segmented algorithm)
- [x] Distributed product with `hpx::reduce`
- [x] Distributed min/max with `hpx::min_element/max_element`
- [x] Distributed mean, var, std
- [x] Element-wise operations: `add_scalar`, `multiply_scalar`, `fill`

### Free Functions for Reductions
- [x] `distributed_sum`, `distributed_mean`, `distributed_min`, `distributed_max`
- [x] `distributed_var`, `distributed_std`, `distributed_prod`

### Tests
- [x] 32 new partitioned_array tests (408 total tests passing)

## Phase 10: GPU-Native Kernels (COMPLETED)

### CUDA-Native Reductions (Thrust)
- [x] `gpu.sum()` - GPU-native sum using `thrust::reduce`
- [x] `gpu.prod()` - GPU-native product using `thrust::reduce`
- [x] `gpu.min()` - GPU-native min using `thrust::min_element`
- [x] `gpu.max()` - GPU-native max using `thrust::max_element`
- [x] `gpu.mean()` - GPU-native mean (sum/size)
- [x] `gpu.var()` - GPU-native variance using `thrust::transform_reduce`
- [x] `gpu.std()` - GPU-native standard deviation

### SYCL-Native Reductions
- [x] `sycl.sum()` - GPU-native sum using SYCL reduction API
- [x] `sycl.prod()` - GPU-native product using SYCL reduction API
- [x] `sycl.min()` - GPU-native min using SYCL reduction API
- [x] `sycl.max()` - GPU-native max using SYCL reduction API
- [x] `sycl.mean()` - GPU-native mean
- [x] `sycl.var()` - GPU-native variance
- [x] `sycl.std()` - GPU-native standard deviation

### Tests
- [x] 14 new GPU reduction tests (7 CUDA + 7 SYCL)
- [x] All tests pass (408 passed, 68 skipped on non-GPU systems)

## Future Enhancements (Deferred)

### Custom Parallel Transform (High Priority)

Expose HPX `for_each`/`transform` to allow user-defined element-wise operations without Python callback overhead.

**Challenge**: Python functions require GIL, which serializes parallelism.

**Implementation Options** (in order of recommendation):
- [ ] **Expression strings** - Parse math expressions in C++: `hpx.transform(arr, "x * 2 + sin(x)")`. Simple, covers 80% of use cases, no dependencies.
- [ ] **Lazy expression trees** - Build expression graph in Python, execute in C++: `expr = x * 2 + hpx.sin(x); expr.evaluate()`. More flexible, similar to TensorFlow/Dask.
- [ ] **Numba integration** - Accept Numba-compiled functions: `hpx.transform(arr, numba_func)`. Most flexible, requires Numba dependency.
- [ ] **Custom kernel registration** - Compile C++ kernels once, reuse: `hpx.register_kernel("my_op", "x * 2")`.

**Note**: Composition of existing ops already works in parallel: `arr * 2 + 1`, `hpx.sin(arr) + hpx.cos(arr)`.

### HPX Algorithms to Expose in Python

HPX provides 72 parallel algorithms. The following are NOT yet exposed but have high value:

**Tier 1 - High Priority:**
- [ ] `all_any_none` → `any()`, `all()` - Parallel boolean reductions
- [ ] `minmax_element` → `argmin()`, `argmax()` - Find index of min/max
- [ ] `adjacent_difference` → `diff()` - Parallel differences
- [ ] `unique` → `unique()` - Parallel unique elements
- [ ] `sort_by_key` → Fix `argsort()` - Currently uses NumPy fallback
- [ ] `reduce` (custom op) → `hpx.reduce(arr, op)` - General reduction with custom binary op
- [ ] `reduce_deterministic` → `hpx.reduce_deterministic()` - Deterministic FP reduction order
- [ ] `inclusive_scan` → `hpx.inclusive_scan()`, `cumsum` - Running accumulation (expose directly)
- [ ] `exclusive_scan` → `hpx.exclusive_scan()` - Prefix sum excluding current element
- [ ] `transform_reduce` → `hpx.transform_reduce()` - Fused transform+reduce in one pass
- [ ] `reduce_by_key` → `hpx.reduce_by_key()` - Grouped reductions (enables histogram)

**Tier 2 - Medium Priority:**
- [ ] `equal` → `array_equal()` - Parallel array comparison
- [ ] `find` → `nonzero()`, `argwhere()` - Parallel search
- [ ] `reverse` → `flip()`, `flipud()`, `fliplr()` - Parallel reversal
- [ ] `rotate` → `roll()` - Parallel rotation
- [ ] `partial_sort` / `nth_element` → `median()`, `percentile()` - O(n) selection
- [ ] `stable_sort` → `stable_sort()` - Order-preserving sort

**Tier 3 - Lower Priority:**
- [ ] `merge` → `concatenate()` (for sorted arrays)
- [ ] `set_difference` → `setdiff1d()`
- [ ] `set_intersection` → `intersect1d()`
- [ ] `set_union` → `union1d()`
- [ ] `set_symmetric_difference` → `setxor1d()`
- [ ] `includes` → `isin()`, `in1d()` - Membership testing
- [ ] `search` → `searchsorted()` - Binary search
- [ ] `partition` → `partition()` - Parallel partitioning

### NumPy Functions Needing HPX Equivalents

**Tier 1 - Very High Parallel Benefit:**
- [ ] `dot` / `matmul` - Matrix multiplication (massive parallel benefit)
- [ ] `any` / `all` - Boolean reductions (use HPX all_any_none)
- [ ] `argmin` / `argmax` - Index of extrema (use HPX minmax_element)
- [ ] `diff` - Differences (use HPX adjacent_difference)
- [ ] `unique` - Unique elements (use HPX unique)
- [ ] `reduce` - General ufunc.reduce (use HPX reduce with custom op)
- [ ] `accumulate` - General ufunc.accumulate (use HPX inclusive_scan with custom op)

**Tier 2 - Good Parallel Benefit:**
- [ ] `median` - Median value (use HPX nth_element)
- [ ] `percentile` / `quantile` - Percentiles (use HPX partial_sort)
- [ ] `histogram` - Histogram bins (use HPX reduce_by_key)
- [ ] `searchsorted` - Binary search (use HPX search)
- [ ] `gradient` - Numerical gradient (use HPX adjacent_difference)
- [ ] `convolve` - Convolution (use HPX transform + custom)

**Tier 3 - Moderate Parallel Benefit:**
- [ ] `flip` / `flipud` / `fliplr` - Reverse array (use HPX reverse)
- [ ] `roll` - Shift elements (use HPX rotate)
- [ ] `tile` / `repeat` - Repeat array (use HPX transform)
- [ ] `intersect1d` / `union1d` - Set operations (use HPX set_*)
- [ ] `isin` / `in1d` - Membership test (use HPX find)

### Known NumPy Fallbacks to Fix
- [ ] `argsort()` - Currently delegates to numpy (use HPX sort_by_key)

### GPU Transforms
- Custom CUDA/SYCL kernels for transforms
- Multi-GPU support with HPX distribution

### Advanced Shape Operations
- Transpose support
- Squeeze/expand_dims

### Advanced Indexing
- Boolean mask indexing: `arr[arr > 5]`
- Fancy indexing: `arr[[1, 3, 5]]`
- Integer array indexing
