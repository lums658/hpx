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

## Phase 8: Multi-dimensional Slicing

### Multi-dimensional Slicing
- `arr[1:3, 2:5]` - Multi-axis slicing
- Requires tuple handling in __getitem__
- Multi-axis __setitem__ support

### Advanced Shape Operations
- Transpose support
- Squeeze/expand_dims

## Phase 9: Broadcasting

### Binary Operation Broadcasting
- NumPy-compatible broadcasting for binary operations
- Shape compatibility checking
- Requires significant operator refactoring

### Advanced Indexing
- Boolean mask indexing: `arr[arr > 5]`
- Fancy indexing: `arr[[1, 3, 5]]`
- Integer array indexing

## Phase 11: True Distribution

### partitioned_vector Integration
- Connect distributed_array to HPX partitioned_vector
- True multi-locality data distribution

### Distributed Algorithms
- Parallel algorithms on distributed arrays
- Cross-locality communication

## Phase 10: GPU-Native Kernels

### GPU-Native Reductions (Phase 5 deferred)
- Thrust-based GPU sum/reduce for CUDA
- SYCL parallel STL reductions
- Currently falls back to CPU

### GPU Transforms
- Custom CUDA/SYCL kernels for transforms
- Multi-GPU support with HPX distribution
