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

## Phase 10: GPU-Native Kernels

### GPU-Native Reductions (Phase 5 deferred)
- Thrust-based GPU sum/reduce for CUDA
- SYCL parallel STL reductions
- Currently falls back to CPU

### GPU Transforms
- Custom CUDA/SYCL kernels for transforms
- Multi-GPU support with HPX distribution
