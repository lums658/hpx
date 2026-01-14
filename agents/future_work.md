# HPXPy Future Work

Deferred features and enhancements for future phases.

## Known Compiler Issues

### Apple Clang 17.0.0 Crashes with hpx::any_of / hpx::all_of

**Status**: ACTIVE BUG - Workaround in place

**Symptom**: Compiler segfaults (Segmentation fault: 11) when compiling code that uses `hpx::any_of` or `hpx::all_of` directly with lambda predicates.

**Affected Code**:
```cpp
// This crashes Apple clang 17.0.0:
hpx::any_of(hpx::execution::par, data, data + size,
    [](double val) { return val != 0.0; });
```

**Workaround**: Use `hpx::count_if` instead, which provides equivalent functionality:
```cpp
// This works:
auto count = hpx::count_if(hpx::execution::par, data, data + size,
    [](double val) { return val != 0.0; });
return count > 0;  // any_of semantics
```

**Location**: [algorithm_bindings.cpp:615-670](python/src/bindings/algorithm_bindings.cpp#L615)

**Impact**: `any()` and `all()` functions work correctly but use count-based implementation instead of early-exit algorithms. Performance may be slightly worse for large arrays where the result could be determined early.

**TODO**: Test with newer clang versions or GCC to see if direct `hpx::any_of`/`hpx::all_of` can be used.

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
- [x] `all_any_none` → `any()`, `all()` - Parallel boolean reductions *(implemented via count_if due to clang bug)*
- [x] `minmax_element` → `argmin()`, `argmax()` - Find index of min/max
- [x] `adjacent_difference` → `diff()` - Parallel differences
- [x] `unique` → `unique()` - Parallel unique elements
- [x] `sort_by_key` → `argsort()` - Uses HPX sort with value-based comparator
- [x] `reduce` (custom op) → `hpx.reduce(arr, op)` - Wraps transform_reduce with identity
- [x] `reduce_deterministic` → `hpx.reduce_deterministic()` - Sequential reduction for reproducibility
- [x] `inclusive_scan` → `hpx.inclusive_scan()`, `cumsum` - Running accumulation (expose directly)
- [x] `exclusive_scan` → `hpx.exclusive_scan()` - Prefix sum excluding current element
- [x] `transform_reduce` → `hpx.transform_reduce()` - Fused transform+reduce in one pass
- [x] `reduce_by_key` → `hpx.reduce_by_key()` - Grouped reductions (enables histogram)

**Tier 2 - Medium Priority:**
- [x] `equal` → `array_equal()` - Parallel array comparison using hpx::equal
- [x] `find` → `nonzero()` - Find non-zero element indices
- [x] `reverse` → `flip()` - Parallel reversal using hpx::reverse
- [x] `rotate` → `roll()` - Parallel rotation using hpx::rotate
- [x] `partial_sort` / `nth_element` → `median()`, `percentile()` - O(n) selection
- [x] `stable_sort` → `stable_sort()` - Order-preserving sort using hpx::stable_sort

**Tier 3 - Lower Priority:**
- [x] `merge` → `merge_sorted()` - Merge two sorted arrays using hpx::merge
- [x] `set_difference` → `setdiff1d()` - Set difference using hpx::set_difference
- [x] `set_intersection` → `intersect1d()` - Set intersection using hpx::set_intersection
- [x] `set_union` → `union1d()` - Set union using hpx::set_union
- [x] `set_symmetric_difference` → `setxor1d()` - Symmetric difference
- [x] `includes` → `includes()` - Test if arr1 contains all elements of arr2
- [x] `isin` → `isin()` - Membership testing with binary search
- [x] `search` → `searchsorted()` - Binary search using lower_bound/upper_bound
- [x] `partition` → `partition()` - Partition array around pivot

### NumPy Functions Needing HPX Equivalents

**Tier 1 - Very High Parallel Benefit:**
- [ ] `dot` / `matmul` - Matrix multiplication (massive parallel benefit)
- [x] `any` / `all` - Boolean reductions (use HPX count_if - workaround for clang bug)
- [x] `argmin` / `argmax` - Index of extrema (use HPX min/max_element)
- [x] `diff` - Differences (use HPX adjacent_difference)
- [x] `unique` - Unique elements (use HPX unique)
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
- [x] `argsort()` - Fixed! Now uses HPX sort with value-based comparator for 1D arrays

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
