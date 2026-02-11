// HPXPy - Algorithm bindings
//
// SPDX-License-Identifier: BSL-1.0

#include "ndarray.hpp"
#include "operators.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/numeric.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>

namespace py = pybind11;

namespace hpxpy {

// Dispatch helper for typed operations
template<typename Func>
auto dispatch_dtype(ndarray const& arr, Func&& func) {
    char kind = arr.dtype().kind();
    auto itemsize = arr.dtype().itemsize();

    if (kind == 'f') {  // floating point
        if (itemsize == 8) {
            return func(static_cast<double const*>(arr.data()), arr.size());
        } else if (itemsize == 4) {
            return func(static_cast<float const*>(arr.data()), arr.size());
        }
    } else if (kind == 'i') {  // signed integer
        if (itemsize == 8) {
            return func(static_cast<int64_t const*>(arr.data()), arr.size());
        } else if (itemsize == 4) {
            return func(static_cast<int32_t const*>(arr.data()), arr.size());
        }
    } else if (kind == 'u') {  // unsigned integer
        if (itemsize == 8) {
            return func(static_cast<uint64_t const*>(arr.data()), arr.size());
        } else if (itemsize == 4) {
            return func(static_cast<uint32_t const*>(arr.data()), arr.size());
        }
    }

    throw std::runtime_error("Unsupported dtype: " +
        py::str(arr.dtype()).cast<std::string>());
}

// Helper to dispatch execution based on policy string
// Returns the result of calling func with the appropriate execution policy
template<typename Func>
auto with_execution_policy(std::string const& policy, Func&& func) {
    if (policy == "par") {
        return func(hpx::execution::par);
    } else if (policy == "par_unseq") {
        return func(hpx::execution::par_unseq);
    } else if (policy == "seq") {
        return func(hpx::execution::seq);
    } else {
        throw std::invalid_argument(
            "Invalid execution policy: '" + policy + "'. "
            "Valid policies are: 'seq', 'par', 'par_unseq'");
    }
}

// Helper: compute linear index from multi-dimensional indices
inline py::ssize_t compute_linear_index(
    std::vector<py::ssize_t> const& indices,
    std::vector<py::ssize_t> const& strides,
    py::ssize_t itemsize)
{
    py::ssize_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += indices[i] * strides[i];
    }
    return offset / itemsize;  // Convert byte offset to element offset
}

// Helper: compute output shape after reducing along axis
inline std::vector<py::ssize_t> compute_reduced_shape(
    std::vector<py::ssize_t> const& shape,
    int axis,
    bool keepdims)
{
    std::vector<py::ssize_t> result;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int>(i) == axis) {
            if (keepdims) {
                result.push_back(1);
            }
        } else {
            result.push_back(shape[i]);
        }
    }
    return result;
}

// Sum reduction (full array)
// policy: "seq" (deterministic), "par" (parallel), "par_unseq" (parallel + SIMD, default via CMake)
py::object sum(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->size() == 0) {
        return py::cast(0.0);
    }

    // dispatch_dtype needs GIL to access arr->dtype()
    return dispatch_dtype(*arr, [&policy](auto const* data, py::ssize_t size) -> py::object {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(data)>>;

        // Release GIL during HPX computation
        T result;
        {
            py::gil_scoped_release release;
            result = with_execution_policy(policy, [&](auto exec) {
                return hpx::reduce(exec, data, data + size, T{0});
            });
        }

        return py::cast(result);
    });
}

// Sum reduction along axis
std::shared_ptr<ndarray> sum_axis(std::shared_ptr<ndarray> arr, int axis, bool keepdims) {
    auto const& shape = arr->shape();
    int ndim = static_cast<int>(shape.size());

    // Handle negative axis
    if (axis < 0) {
        axis += ndim;
    }
    if (axis < 0 || axis >= ndim) {
        throw std::runtime_error("axis " + std::to_string(axis) +
            " is out of bounds for array of dimension " + std::to_string(ndim));
    }

    // Compute output shape
    auto out_shape = compute_reduced_shape(shape, axis, keepdims);
    auto result = std::make_shared<ndarray>(out_shape, arr->dtype());

    char kind = arr->dtype().kind();
    auto itemsize = arr->dtype().itemsize();
    auto const& strides = arr->strides();

    // Compute sizes for iteration
    py::ssize_t reduce_size = shape[axis];
    py::ssize_t reduce_stride = strides[axis];

    // Compute number of output elements
    py::ssize_t out_size = 1;
    for (auto d : out_shape) out_size *= d;

    if (out_size == 0 || reduce_size == 0) {
        return result;
    }

    // Build iteration indices excluding the reduction axis
    std::vector<py::ssize_t> outer_shape;
    std::vector<py::ssize_t> outer_strides;
    for (int i = 0; i < ndim; ++i) {
        if (i != axis) {
            outer_shape.push_back(shape[i]);
            outer_strides.push_back(strides[i]);
        }
    }

    py::gil_scoped_release release;

    // Perform reduction based on dtype
    if (kind == 'f' && itemsize == 8) {
        double const* src = arr->typed_data<double>();
        double* dst = result->typed_data<double>();

        hpx::experimental::for_loop(hpx::execution::par, 0, out_size, [&](py::ssize_t out_idx) {
            // Convert flat output index to multi-dimensional index
            std::vector<py::ssize_t> outer_indices(outer_shape.size());
            py::ssize_t remaining = out_idx;
            for (int i = static_cast<int>(outer_shape.size()) - 1; i >= 0; --i) {
                outer_indices[i] = remaining % outer_shape[i];
                remaining /= outer_shape[i];
            }

            // Compute base offset in source array
            py::ssize_t base_offset = 0;
            for (size_t i = 0; i < outer_indices.size(); ++i) {
                base_offset += outer_indices[i] * outer_strides[i];
            }
            base_offset /= itemsize;

            // Sum along the reduction axis
            double sum = 0.0;
            py::ssize_t reduce_step = reduce_stride / itemsize;
            for (py::ssize_t r = 0; r < reduce_size; ++r) {
                sum += src[base_offset + r * reduce_step];
            }
            dst[out_idx] = sum;
        });
    } else if (kind == 'f' && itemsize == 4) {
        float const* src = arr->typed_data<float>();
        float* dst = result->typed_data<float>();

        hpx::experimental::for_loop(hpx::execution::par, 0, out_size, [&](py::ssize_t out_idx) {
            std::vector<py::ssize_t> outer_indices(outer_shape.size());
            py::ssize_t remaining = out_idx;
            for (int i = static_cast<int>(outer_shape.size()) - 1; i >= 0; --i) {
                outer_indices[i] = remaining % outer_shape[i];
                remaining /= outer_shape[i];
            }

            py::ssize_t base_offset = 0;
            for (size_t i = 0; i < outer_indices.size(); ++i) {
                base_offset += outer_indices[i] * outer_strides[i];
            }
            base_offset /= itemsize;

            float sum = 0.0f;
            py::ssize_t reduce_step = reduce_stride / itemsize;
            for (py::ssize_t r = 0; r < reduce_size; ++r) {
                sum += src[base_offset + r * reduce_step];
            }
            dst[out_idx] = sum;
        });
    } else if (kind == 'i' && itemsize == 8) {
        int64_t const* src = arr->typed_data<int64_t>();
        int64_t* dst = result->typed_data<int64_t>();

        hpx::experimental::for_loop(hpx::execution::par, 0, out_size, [&](py::ssize_t out_idx) {
            std::vector<py::ssize_t> outer_indices(outer_shape.size());
            py::ssize_t remaining = out_idx;
            for (int i = static_cast<int>(outer_shape.size()) - 1; i >= 0; --i) {
                outer_indices[i] = remaining % outer_shape[i];
                remaining /= outer_shape[i];
            }

            py::ssize_t base_offset = 0;
            for (size_t i = 0; i < outer_indices.size(); ++i) {
                base_offset += outer_indices[i] * outer_strides[i];
            }
            base_offset /= itemsize;

            int64_t sum = 0;
            py::ssize_t reduce_step = reduce_stride / itemsize;
            for (py::ssize_t r = 0; r < reduce_size; ++r) {
                sum += src[base_offset + r * reduce_step];
            }
            dst[out_idx] = sum;
        });
    } else if (kind == 'i' && itemsize == 4) {
        int32_t const* src = arr->typed_data<int32_t>();
        int32_t* dst = result->typed_data<int32_t>();

        hpx::experimental::for_loop(hpx::execution::par, 0, out_size, [&](py::ssize_t out_idx) {
            std::vector<py::ssize_t> outer_indices(outer_shape.size());
            py::ssize_t remaining = out_idx;
            for (int i = static_cast<int>(outer_shape.size()) - 1; i >= 0; --i) {
                outer_indices[i] = remaining % outer_shape[i];
                remaining /= outer_shape[i];
            }

            py::ssize_t base_offset = 0;
            for (size_t i = 0; i < outer_indices.size(); ++i) {
                base_offset += outer_indices[i] * outer_strides[i];
            }
            base_offset /= itemsize;

            int32_t sum = 0;
            py::ssize_t reduce_step = reduce_stride / itemsize;
            for (py::ssize_t r = 0; r < reduce_size; ++r) {
                sum += src[base_offset + r * reduce_step];
            }
            dst[out_idx] = sum;
        });
    } else {
        py::gil_scoped_acquire acquire;
        throw std::runtime_error("Unsupported dtype for sum_axis");
    }

    return result;
}

// Product reduction
// policy: "seq" (deterministic), "par" (parallel), "par_unseq" (parallel + SIMD, default via CMake)
py::object prod(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->size() == 0) {
        return py::cast(1.0);
    }

    // dispatch_dtype needs GIL to access arr->dtype()
    return dispatch_dtype(*arr, [&policy](auto const* data, py::ssize_t size) -> py::object {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(data)>>;

        // Release GIL during HPX computation
        T result;
        {
            py::gil_scoped_release release;
            result = with_execution_policy(policy, [&](auto exec) {
                return hpx::reduce(exec, data, data + size, T{1}, std::multiplies<T>{});
            });
        }

        return py::cast(result);
    });
}

// Minimum reduction
// policy: "seq" (deterministic), "par" (parallel), "par_unseq" (parallel + SIMD, default via CMake)
py::object min(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->size() == 0) {
        throw std::runtime_error("min() arg is an empty sequence");
    }

    // dispatch_dtype needs GIL to access arr->dtype()
    return dispatch_dtype(*arr, [&policy](auto const* data, py::ssize_t size) -> py::object {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(data)>>;

        // Release GIL during HPX computation
        // Note: par_unseq triggers a __restrict__ bug in HPX's unseq/loop.hpp
        // with GCC 15 for min/max_element, so we use par instead.
        T const* result;
        {
            py::gil_scoped_release release;
            if (policy == "par" || policy == "par_unseq") {
                result = hpx::min_element(hpx::execution::par, data, data + size);
            } else {
                result = hpx::min_element(hpx::execution::seq, data, data + size);
            }
        }

        return py::cast(*result);
    });
}

// Maximum reduction
// policy: "seq" (deterministic), "par" (parallel), "par_unseq" (parallel + SIMD, default via CMake)
py::object max(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->size() == 0) {
        throw std::runtime_error("max() arg is an empty sequence");
    }

    // dispatch_dtype needs GIL to access arr->dtype()
    return dispatch_dtype(*arr, [&policy](auto const* data, py::ssize_t size) -> py::object {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(data)>>;

        // Release GIL during HPX computation
        T const* result;
        {
            py::gil_scoped_release release;
            if (policy == "par" || policy == "par_unseq") {
                result = hpx::max_element(hpx::execution::par, data, data + size);
            } else {
                result = hpx::max_element(hpx::execution::seq, data, data + size);
            }
        }

        return py::cast(*result);
    });
}

// Sort (returns new sorted array)
// policy: "seq" (deterministic), "par" (parallel), "par_unseq" (parallel + SIMD, default via CMake)
std::shared_ptr<ndarray> sort(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->ndim() != 1) {
        throw std::runtime_error("sort only supports 1D arrays in Phase 1");
    }

    // Create a copy for sorting
    auto result = std::make_shared<ndarray>(arr->shape(), arr->dtype());
    std::memcpy(result->data(), arr->data(), arr->nbytes());

    // Cache dtype info before releasing GIL
    char kind = arr->dtype().kind();
    auto itemsize = arr->dtype().itemsize();
    py::ssize_t size = arr->size();

    // Release GIL during sort
    py::gil_scoped_release release;

    // Helper to sort with execution policy
    auto do_sort = [&policy, size](auto* ptr) {
        with_execution_policy(policy, [&](auto exec) {
            hpx::sort(exec, ptr, ptr + size);
            return 0;  // Dummy return for with_execution_policy
        });
    };

    if (kind == 'f') {
        if (itemsize == 8) {
            do_sort(static_cast<double*>(result->data()));
        } else if (itemsize == 4) {
            do_sort(static_cast<float*>(result->data()));
        }
    } else if (kind == 'i') {
        if (itemsize == 8) {
            do_sort(static_cast<int64_t*>(result->data()));
        } else if (itemsize == 4) {
            do_sort(static_cast<int32_t*>(result->data()));
        }
    } else if (kind == 'u') {
        if (itemsize == 8) {
            do_sort(static_cast<uint64_t*>(result->data()));
        } else if (itemsize == 4) {
            do_sort(static_cast<uint32_t*>(result->data()));
        }
    } else {
        py::gil_scoped_acquire acquire;
        throw std::runtime_error("Unsupported dtype for sort");
    }

    return result;
}

// Count occurrences of a value
// policy: "seq" (deterministic), "par" (parallel), "par_unseq" (parallel + SIMD, default via CMake)
py::ssize_t count(std::shared_ptr<ndarray> arr, py::object value, std::string const& policy = "seq") {
    char kind = arr->dtype().kind();
    auto itemsize = arr->dtype().itemsize();

    // Helper to count with execution policy
    auto do_count = [&policy, size = arr->size()](auto const* ptr, auto val) {
        return with_execution_policy(policy, [&](auto exec) {
            return hpx::count(exec, ptr, ptr + size, val);
        });
    };

    py::gil_scoped_release release;

    if (kind == 'f') {
        if (itemsize == 8) {
            py::gil_scoped_acquire acquire;
            double val = value.cast<double>();
            py::gil_scoped_release release2;
            return do_count(static_cast<double const*>(arr->data()), val);
        } else if (itemsize == 4) {
            py::gil_scoped_acquire acquire;
            float val = value.cast<float>();
            py::gil_scoped_release release2;
            return do_count(static_cast<float const*>(arr->data()), val);
        }
    } else if (kind == 'i') {
        if (itemsize == 8) {
            py::gil_scoped_acquire acquire;
            int64_t val = value.cast<int64_t>();
            py::gil_scoped_release release2;
            return do_count(static_cast<int64_t const*>(arr->data()), val);
        } else if (itemsize == 4) {
            py::gil_scoped_acquire acquire;
            int32_t val = value.cast<int32_t>();
            py::gil_scoped_release release2;
            return do_count(static_cast<int32_t const*>(arr->data()), val);
        }
    }

    py::gil_scoped_acquire acquire;
    throw std::runtime_error("Unsupported dtype for count");
}

// Math functions using operators.hpp
std::shared_ptr<ndarray> sqrt_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Sqrt{});
}

std::shared_ptr<ndarray> square_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Square{});
}

std::shared_ptr<ndarray> exp_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Exp{});
}

std::shared_ptr<ndarray> exp2_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Exp2{});
}

std::shared_ptr<ndarray> log_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Log{});
}

std::shared_ptr<ndarray> log2_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Log2{});
}

std::shared_ptr<ndarray> log10_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Log10{});
}

std::shared_ptr<ndarray> sin_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Sin{});
}

std::shared_ptr<ndarray> cos_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Cos{});
}

std::shared_ptr<ndarray> tan_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Tan{});
}

std::shared_ptr<ndarray> arcsin_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Asin{});
}

std::shared_ptr<ndarray> arccos_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Acos{});
}

std::shared_ptr<ndarray> arctan_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Atan{});
}

std::shared_ptr<ndarray> sinh_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Sinh{});
}

std::shared_ptr<ndarray> cosh_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Cosh{});
}

std::shared_ptr<ndarray> tanh_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Tanh{});
}

std::shared_ptr<ndarray> floor_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Floor{});
}

std::shared_ptr<ndarray> ceil_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Ceil{});
}

std::shared_ptr<ndarray> trunc_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Trunc{});
}

std::shared_ptr<ndarray> abs_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Abs{});
}

std::shared_ptr<ndarray> sign_arr(std::shared_ptr<ndarray> arr) {
    return ops::dispatch_unary(*arr, ops::Sign{});
}

// Cumulative sum (scan)
std::shared_ptr<ndarray> cumsum(std::shared_ptr<ndarray> arr) {
    auto result = std::make_shared<ndarray>(arr->shape(), arr->dtype());

    char kind = arr->dtype().kind();
    auto itemsize = arr->dtype().itemsize();

    py::gil_scoped_release release;

    if (kind == 'f') {
        if (itemsize == 8) {
            auto const* src = arr->typed_data<double>();
            auto* dst = result->typed_data<double>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst);
        } else if (itemsize == 4) {
            auto const* src = arr->typed_data<float>();
            auto* dst = result->typed_data<float>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst);
        }
    } else if (kind == 'i') {
        if (itemsize == 8) {
            auto const* src = arr->typed_data<int64_t>();
            auto* dst = result->typed_data<int64_t>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst);
        } else if (itemsize == 4) {
            auto const* src = arr->typed_data<int32_t>();
            auto* dst = result->typed_data<int32_t>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst);
        }
    } else {
        py::gil_scoped_acquire acquire;
        throw std::runtime_error("Unsupported dtype for cumsum");
    }

    return result;
}

// Cumulative product (scan)
std::shared_ptr<ndarray> cumprod(std::shared_ptr<ndarray> arr) {
    auto result = std::make_shared<ndarray>(arr->shape(), arr->dtype());

    char kind = arr->dtype().kind();
    auto itemsize = arr->dtype().itemsize();

    py::gil_scoped_release release;

    if (kind == 'f') {
        if (itemsize == 8) {
            auto const* src = arr->typed_data<double>();
            auto* dst = result->typed_data<double>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst,
                std::multiplies<double>{});
        } else if (itemsize == 4) {
            auto const* src = arr->typed_data<float>();
            auto* dst = result->typed_data<float>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst,
                std::multiplies<float>{});
        }
    } else if (kind == 'i') {
        if (itemsize == 8) {
            auto const* src = arr->typed_data<int64_t>();
            auto* dst = result->typed_data<int64_t>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst,
                std::multiplies<int64_t>{});
        } else if (itemsize == 4) {
            auto const* src = arr->typed_data<int32_t>();
            auto* dst = result->typed_data<int32_t>();
            hpx::inclusive_scan(hpx::execution::par, src, src + arr->size(), dst,
                std::multiplies<int32_t>{});
        }
    } else {
        py::gil_scoped_acquire acquire;
        throw std::runtime_error("Unsupported dtype for cumprod");
    }

    return result;
}

// =============================================================================
// New HPX Algorithm Exposures
// Simplified implementations for float64 only to avoid compiler issues
// =============================================================================

// Note: hpx::any_of and hpx::all_of directly crash Apple clang 17.0.0,
// so we use hpx::count_if as a workaround which provides equivalent functionality.

// any_of - Check if any element is truthy (non-zero)
// Implemented using hpx::count to avoid compiler crash with hpx::any_of
bool any_of(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("any currently only supports float64");
    }
    double const* data = arr->typed_data<double>();
    py::ssize_t size = arr->size();

    py::gil_scoped_release release;
    // Count non-zero elements - any() is true if count > 0
    std::ptrdiff_t count;
    if (policy == "par") {
        count = hpx::count_if(hpx::execution::par, data, data + size,
            [](double val) { return val != 0.0; });
    } else if (policy == "par_unseq") {
        count = hpx::count_if(hpx::execution::par_unseq, data, data + size,
            [](double val) { return val != 0.0; });
    } else {
        count = hpx::count_if(hpx::execution::seq, data, data + size,
            [](double val) { return val != 0.0; });
    }
    return count > 0;
}

// all_of - Check if all elements are truthy (non-zero)
// Implemented using hpx::count to avoid compiler crash with hpx::all_of
bool all_of(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("all currently only supports float64");
    }
    double const* data = arr->typed_data<double>();
    py::ssize_t size = arr->size();

    py::gil_scoped_release release;
    // Count zero elements - all() is true if count == 0
    std::ptrdiff_t count;
    if (policy == "par") {
        count = hpx::count_if(hpx::execution::par, data, data + size,
            [](double val) { return val == 0.0; });
    } else if (policy == "par_unseq") {
        count = hpx::count_if(hpx::execution::par_unseq, data, data + size,
            [](double val) { return val == 0.0; });
    } else {
        count = hpx::count_if(hpx::execution::seq, data, data + size,
            [](double val) { return val == 0.0; });
    }
    return count == 0;
}

// argmin - Return index of minimum element
py::ssize_t argmin(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("argmin currently only supports float64");
    }
    double const* data = arr->typed_data<double>();
    py::ssize_t size = arr->size();

    py::gil_scoped_release release;
    double const* result;
    // Note: par_unseq triggers a __restrict__ bug in HPX's unseq/loop.hpp
    // with GCC 15 for min/max_element, so we fall back to par.
    if (policy == "par" || policy == "par_unseq") {
        result = hpx::min_element(hpx::execution::par, data, data + size);
    } else {
        result = hpx::min_element(hpx::execution::seq, data, data + size);
    }
    return static_cast<py::ssize_t>(result - data);
}

// argmax - Return index of maximum element
py::ssize_t argmax(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("argmax currently only supports float64");
    }
    double const* data = arr->typed_data<double>();
    py::ssize_t size = arr->size();

    py::gil_scoped_release release;
    double const* result;
    if (policy == "par" || policy == "par_unseq") {
        result = hpx::max_element(hpx::execution::par, data, data + size);
    } else {
        result = hpx::max_element(hpx::execution::seq, data, data + size);
    }
    return static_cast<py::ssize_t>(result - data);
}

// argsort - Return indices that would sort the array
// Uses HPX sort with custom comparator instead of NumPy fallback
std::shared_ptr<ndarray> argsort(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->ndim() != 1) {
        throw std::runtime_error("argsort only supports 1D arrays");
    }
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("argsort currently only supports float64");
    }

    py::ssize_t n = arr->size();
    double const* data = arr->typed_data<double>();

    // Create indices array [0, 1, 2, ..., n-1]
    std::vector<py::ssize_t> shape = {n};
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<int64_t>());
    int64_t* indices = result->typed_data<int64_t>();

    // Initialize indices
    for (py::ssize_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    {
        py::gil_scoped_release release;

        // Sort indices by comparing the values they point to
        auto compare = [data](int64_t a, int64_t b) {
            return data[a] < data[b];
        };

        if (policy == "par") {
            hpx::sort(hpx::execution::par, indices, indices + n, compare);
        } else if (policy == "par_unseq") {
            hpx::sort(hpx::execution::par_unseq, indices, indices + n, compare);
        } else {
            hpx::sort(hpx::execution::seq, indices, indices + n, compare);
        }
    }

    return result;
}

// diff - Compute n-th discrete difference using adjacent_difference
// Simplified for float64 only
std::shared_ptr<ndarray> diff(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("diff currently only supports float64");
    }
    if (arr->size() < 2) {
        std::vector<py::ssize_t> shape = {0};
        return std::make_shared<ndarray>(shape, arr->dtype());
    }

    std::vector<py::ssize_t> shape = {arr->size() - 1};
    auto result = std::make_shared<ndarray>(shape, arr->dtype());
    double const* src = arr->typed_data<double>();
    double* dst = result->typed_data<double>();
    py::ssize_t n = arr->size();

    py::gil_scoped_release release;

    std::vector<double> temp(n);
    if (policy == "par") {
        hpx::adjacent_difference(hpx::execution::par, src, src + n, temp.begin());
    } else {
        hpx::adjacent_difference(hpx::execution::seq, src, src + n, temp.begin());
    }
    std::copy(temp.begin() + 1, temp.end(), dst);

    return result;
}

// unique - Return sorted unique elements
// Simplified for float64 only
std::shared_ptr<ndarray> unique(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("unique currently only supports float64");
    }

    auto sorted = sort(arr, policy);
    double* data = sorted->typed_data<double>();
    py::ssize_t n = sorted->size();

    double* end;
    {
        py::gil_scoped_release release;

        if (policy == "par") {
            end = hpx::unique(hpx::execution::par, data, data + n);
        } else {
            end = hpx::unique(hpx::execution::seq, data, data + n);
        }
    }
    // GIL is re-acquired here automatically

    py::ssize_t new_size = end - data;
    std::vector<py::ssize_t> shape = {new_size};
    // Create dtype directly for float64
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    std::memcpy(result->data(), sorted->data(), new_size * sizeof(double));

    return result;
}

// inclusive_scan - Running accumulation (like cumsum but more general)
// Simplified for float64 only
std::shared_ptr<ndarray> inclusive_scan(
    std::shared_ptr<ndarray> arr,
    std::string const& op = "add",
    std::string const& policy = "par"
) {
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("inclusive_scan currently only supports float64");
    }

    auto result = std::make_shared<ndarray>(arr->shape(), arr->dtype());
    double const* src = arr->typed_data<double>();
    double* dst = result->typed_data<double>();
    py::ssize_t n = arr->size();

    py::gil_scoped_release release;

    if (op == "add") {
        if (policy == "par") {
            hpx::inclusive_scan(hpx::execution::par, src, src + n, dst, std::plus<double>{});
        } else {
            hpx::inclusive_scan(hpx::execution::seq, src, src + n, dst, std::plus<double>{});
        }
    } else if (op == "mul") {
        if (policy == "par") {
            hpx::inclusive_scan(hpx::execution::par, src, src + n, dst, std::multiplies<double>{});
        } else {
            hpx::inclusive_scan(hpx::execution::seq, src, src + n, dst, std::multiplies<double>{});
        }
    } else {
        py::gil_scoped_acquire acquire;
        throw std::invalid_argument("Unsupported op for inclusive_scan: " + op);
    }

    return result;
}

// exclusive_scan - Running accumulation excluding current element
// Simplified for float64 only
std::shared_ptr<ndarray> exclusive_scan(
    std::shared_ptr<ndarray> arr,
    py::object init_val,
    std::string const& op = "add",
    std::string const& policy = "par"
) {
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("exclusive_scan currently only supports float64");
    }

    auto result = std::make_shared<ndarray>(arr->shape(), arr->dtype());
    double const* src = arr->typed_data<double>();
    double* dst = result->typed_data<double>();
    py::ssize_t n = arr->size();
    double init = init_val.cast<double>();

    py::gil_scoped_release release;

    if (op == "add") {
        if (policy == "par") {
            hpx::exclusive_scan(hpx::execution::par, src, src + n, dst, init, std::plus<double>{});
        } else {
            hpx::exclusive_scan(hpx::execution::seq, src, src + n, dst, init, std::plus<double>{});
        }
    } else if (op == "mul") {
        if (policy == "par") {
            hpx::exclusive_scan(hpx::execution::par, src, src + n, dst, init, std::multiplies<double>{});
        } else {
            hpx::exclusive_scan(hpx::execution::seq, src, src + n, dst, init, std::multiplies<double>{});
        }
    } else {
        py::gil_scoped_acquire acquire;
        throw std::invalid_argument("Unsupported op for exclusive_scan: " + op);
    }

    return result;
}

// transform_reduce - Fused transform and reduce in one pass
// Fully explicit instantiation to avoid compiler crash
py::object transform_reduce(
    std::shared_ptr<ndarray> arr,
    std::string const& transform_op,
    std::string const& reduce_op = "add",
    std::string const& policy = "par"
) {
    char kind = arr->dtype().kind();
    auto itemsize = arr->dtype().itemsize();

    if (kind != 'f' || itemsize != 8) {
        throw std::runtime_error("transform_reduce currently only supports float64");
    }

    double const* data = arr->typed_data<double>();
    py::ssize_t size = arr->size();

    py::gil_scoped_release release;

    double init = (reduce_op == "mul") ? 1.0 :
                  (reduce_op == "min") ? std::numeric_limits<double>::max() :
                  (reduce_op == "max") ? std::numeric_limits<double>::lowest() : 0.0;

    double result = 0.0;

    // Sum of squares: transform_op="square", reduce_op="add"
    if (transform_op == "square" && reduce_op == "add") {
        auto transform = [](double x) { return x * x; };
        if (policy == "par") {
            result = hpx::transform_reduce(hpx::execution::par, data, data + size, init,
                std::plus<double>{}, transform);
        } else if (policy == "par_unseq") {
            result = hpx::transform_reduce(hpx::execution::par_unseq, data, data + size, init,
                std::plus<double>{}, transform);
        } else {
            result = hpx::transform_reduce(hpx::execution::seq, data, data + size, init,
                std::plus<double>{}, transform);
        }
    }
    // Sum of absolute values: transform_op="abs", reduce_op="add"
    else if (transform_op == "abs" && reduce_op == "add") {
        auto transform = [](double x) { return std::abs(x); };
        if (policy == "par") {
            result = hpx::transform_reduce(hpx::execution::par, data, data + size, init,
                std::plus<double>{}, transform);
        } else if (policy == "par_unseq") {
            result = hpx::transform_reduce(hpx::execution::par_unseq, data, data + size, init,
                std::plus<double>{}, transform);
        } else {
            result = hpx::transform_reduce(hpx::execution::seq, data, data + size, init,
                std::plus<double>{}, transform);
        }
    }
    // Simple sum: transform_op="identity", reduce_op="add"
    else if (transform_op == "identity" && reduce_op == "add") {
        auto transform = [](double x) { return x; };
        if (policy == "par") {
            result = hpx::transform_reduce(hpx::execution::par, data, data + size, init,
                std::plus<double>{}, transform);
        } else if (policy == "par_unseq") {
            result = hpx::transform_reduce(hpx::execution::par_unseq, data, data + size, init,
                std::plus<double>{}, transform);
        } else {
            result = hpx::transform_reduce(hpx::execution::seq, data, data + size, init,
                std::plus<double>{}, transform);
        }
    }
    // Max element: transform_op="identity", reduce_op="max"
    else if (transform_op == "identity" && reduce_op == "max") {
        auto transform = [](double x) { return x; };
        auto reduce = [](double a, double b) { return std::max(a, b); };
        if (policy == "par") {
            result = hpx::transform_reduce(hpx::execution::par, data, data + size, init,
                reduce, transform);
        } else if (policy == "par_unseq") {
            result = hpx::transform_reduce(hpx::execution::par_unseq, data, data + size, init,
                reduce, transform);
        } else {
            result = hpx::transform_reduce(hpx::execution::seq, data, data + size, init,
                reduce, transform);
        }
    }
    // Min element: transform_op="identity", reduce_op="min"
    else if (transform_op == "identity" && reduce_op == "min") {
        auto transform = [](double x) { return x; };
        auto reduce = [](double a, double b) { return std::min(a, b); };
        if (policy == "par") {
            result = hpx::transform_reduce(hpx::execution::par, data, data + size, init,
                reduce, transform);
        } else if (policy == "par_unseq") {
            result = hpx::transform_reduce(hpx::execution::par_unseq, data, data + size, init,
                reduce, transform);
        } else {
            result = hpx::transform_reduce(hpx::execution::seq, data, data + size, init,
                reduce, transform);
        }
    }
    // Product: transform_op="identity", reduce_op="mul"
    else if (transform_op == "identity" && reduce_op == "mul") {
        auto transform = [](double x) { return x; };
        if (policy == "par") {
            result = hpx::transform_reduce(hpx::execution::par, data, data + size, 1.0,
                std::multiplies<double>{}, transform);
        } else if (policy == "par_unseq") {
            result = hpx::transform_reduce(hpx::execution::par_unseq, data, data + size, 1.0,
                std::multiplies<double>{}, transform);
        } else {
            result = hpx::transform_reduce(hpx::execution::seq, data, data + size, 1.0,
                std::multiplies<double>{}, transform);
        }
    }
    else {
        py::gil_scoped_acquire acquire;
        throw std::invalid_argument("Unsupported transform/reduce combination");
    }

    py::gil_scoped_acquire acquire;
    return py::cast(result);
}

// reduce_by_key - Group by key and reduce values
// Returns (unique_keys, reduced_values)
// Useful for histogram-like operations
std::tuple<std::shared_ptr<ndarray>, std::shared_ptr<ndarray>> reduce_by_key(
    std::shared_ptr<ndarray> keys,
    std::shared_ptr<ndarray> values,
    std::string const& reduce_op = "add",
    std::string const& policy = "par"
) {
    if (keys->ndim() != 1 || values->ndim() != 1) {
        throw std::runtime_error("reduce_by_key only supports 1D arrays");
    }
    if (keys->size() != values->size()) {
        throw std::runtime_error("keys and values must have the same size");
    }

    // Currently only support float64 keys and float64 values
    if (keys->dtype().kind() != 'f' || keys->dtype().itemsize() != 8) {
        throw std::runtime_error("reduce_by_key currently only supports float64 keys");
    }
    if (values->dtype().kind() != 'f' || values->dtype().itemsize() != 8) {
        throw std::runtime_error("reduce_by_key currently only supports float64 values");
    }

    py::ssize_t n = keys->size();
    double const* keys_data = keys->typed_data<double>();
    double const* values_data = values->typed_data<double>();

    // Create sorted copies (reduce_by_key requires sorted input)
    std::vector<double> sorted_keys(keys_data, keys_data + n);
    std::vector<double> sorted_values(values_data, values_data + n);

    // Sort by key (stable sort to preserve relative order of values with same key)
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
        indices[i] = i;
    }

    {
        py::gil_scoped_release release;

        // Sort indices by key
        if (policy == "par") {
            hpx::sort(hpx::execution::par, indices.begin(), indices.end(),
                [&keys_data](size_t a, size_t b) { return keys_data[a] < keys_data[b]; });
        } else {
            hpx::sort(hpx::execution::seq, indices.begin(), indices.end(),
                [&keys_data](size_t a, size_t b) { return keys_data[a] < keys_data[b]; });
        }

        // Reorder keys and values by sorted indices
        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            sorted_keys[i] = keys_data[indices[i]];
            sorted_values[i] = values_data[indices[i]];
        }
    }

    // Now perform the reduction by key
    std::vector<double> result_keys;
    std::vector<double> result_values;
    result_keys.reserve(n);
    result_values.reserve(n);

    if (n > 0) {
        double current_key = sorted_keys[0];
        double current_value = sorted_values[0];

        for (py::ssize_t i = 1; i < n; ++i) {
            if (sorted_keys[i] == current_key) {
                // Same key - reduce
                if (reduce_op == "add") {
                    current_value += sorted_values[i];
                } else if (reduce_op == "mul") {
                    current_value *= sorted_values[i];
                } else if (reduce_op == "min") {
                    current_value = std::min(current_value, sorted_values[i]);
                } else if (reduce_op == "max") {
                    current_value = std::max(current_value, sorted_values[i]);
                }
            } else {
                // New key - save previous and start new
                result_keys.push_back(current_key);
                result_values.push_back(current_value);
                current_key = sorted_keys[i];
                current_value = sorted_values[i];
            }
        }
        // Don't forget the last group
        result_keys.push_back(current_key);
        result_values.push_back(current_value);
    }

    // Create output arrays
    py::ssize_t result_n = static_cast<py::ssize_t>(result_keys.size());
    std::vector<py::ssize_t> shape = {result_n};

    auto out_keys = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    auto out_values = std::make_shared<ndarray>(shape, py::dtype::of<double>());

    std::memcpy(out_keys->data(), result_keys.data(), result_n * sizeof(double));
    std::memcpy(out_values->data(), result_values.data(), result_n * sizeof(double));

    return std::make_tuple(out_keys, out_values);
}

// -----------------------------------------------------------------------------
// Tier 2 Algorithms
// -----------------------------------------------------------------------------

// array_equal - Parallel array comparison using hpx::equal
bool array_equal(
    std::shared_ptr<ndarray> arr1,
    std::shared_ptr<ndarray> arr2,
    std::string const& policy = "par"
) {
    // Check shape match
    if (arr1->shape() != arr2->shape()) {
        return false;
    }

    // Currently only support float64
    if (arr1->dtype().kind() != 'f' || arr1->dtype().itemsize() != 8) {
        throw std::runtime_error("array_equal currently only supports float64");
    }
    if (arr2->dtype().kind() != 'f' || arr2->dtype().itemsize() != 8) {
        throw std::runtime_error("array_equal currently only supports float64");
    }

    py::ssize_t n = arr1->size();
    double const* data1 = arr1->typed_data<double>();
    double const* data2 = arr2->typed_data<double>();

    bool result;
    {
        py::gil_scoped_release release;

        if (policy == "par") {
            result = hpx::equal(hpx::execution::par, data1, data1 + n, data2);
        } else if (policy == "par_unseq") {
            result = hpx::equal(hpx::execution::par_unseq, data1, data1 + n, data2);
        } else {
            result = hpx::equal(hpx::execution::seq, data1, data1 + n, data2);
        }
    }

    return result;
}

// flip - Reverse array elements using hpx::reverse (returns new array)
std::shared_ptr<ndarray> flip(
    std::shared_ptr<ndarray> arr,
    std::string const& policy = "par"
) {
    if (arr->ndim() != 1) {
        throw std::runtime_error("flip currently only supports 1D arrays");
    }
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("flip currently only supports float64");
    }

    py::ssize_t n = arr->size();
    double const* data = arr->typed_data<double>();

    // Create output array (copy first, then reverse)
    std::vector<py::ssize_t> shape = {n};
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    double* out_data = result->typed_data<double>();

    // Copy data
    std::memcpy(out_data, data, n * sizeof(double));

    {
        py::gil_scoped_release release;

        if (policy == "par") {
            hpx::reverse(hpx::execution::par, out_data, out_data + n);
        } else if (policy == "par_unseq") {
            hpx::reverse(hpx::execution::par_unseq, out_data, out_data + n);
        } else {
            hpx::reverse(hpx::execution::seq, out_data, out_data + n);
        }
    }

    return result;
}

// stable_sort - Order-preserving sort using hpx::stable_sort
std::shared_ptr<ndarray> stable_sort(
    std::shared_ptr<ndarray> arr,
    std::string const& policy = "par"
) {
    if (arr->ndim() != 1) {
        throw std::runtime_error("stable_sort currently only supports 1D arrays");
    }
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("stable_sort currently only supports float64");
    }

    py::ssize_t n = arr->size();
    double const* data = arr->typed_data<double>();

    // Create output array (copy first, then sort)
    std::vector<py::ssize_t> shape = {n};
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    double* out_data = result->typed_data<double>();

    std::memcpy(out_data, data, n * sizeof(double));

    {
        py::gil_scoped_release release;

        if (policy == "par") {
            hpx::stable_sort(hpx::execution::par, out_data, out_data + n);
        } else if (policy == "par_unseq") {
            hpx::stable_sort(hpx::execution::par_unseq, out_data, out_data + n);
        } else {
            hpx::stable_sort(hpx::execution::seq, out_data, out_data + n);
        }
    }

    return result;
}

// rotate - Rotate array elements using hpx::rotate
std::shared_ptr<ndarray> rotate(
    std::shared_ptr<ndarray> arr,
    py::ssize_t shift,
    std::string const& policy = "par"
) {
    if (arr->ndim() != 1) {
        throw std::runtime_error("rotate currently only supports 1D arrays");
    }
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("rotate currently only supports float64");
    }

    py::ssize_t n = arr->size();
    if (n == 0) {
        return std::make_shared<ndarray>(std::vector<py::ssize_t>{0}, py::dtype::of<double>());
    }

    double const* data = arr->typed_data<double>();

    // Create output array (copy first, then rotate)
    std::vector<py::ssize_t> shape = {n};
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    double* out_data = result->typed_data<double>();

    std::memcpy(out_data, data, n * sizeof(double));

    // Normalize shift to [0, n)
    // Note: numpy.roll shifts right for positive shift, we shift left
    // To match numpy semantics: roll(arr, 2) moves elements left by 2
    shift = shift % n;
    if (shift < 0) shift += n;

    // hpx::rotate rotates left - for numpy compatibility we need to adjust
    // numpy.roll(arr, k) -> rotate left by (n - k)
    py::ssize_t rotate_point = shift;  // How many positions to rotate left

    {
        py::gil_scoped_release release;

        if (rotate_point > 0 && rotate_point < n) {
            if (policy == "par") {
                hpx::rotate(hpx::execution::par, out_data, out_data + rotate_point, out_data + n);
            } else if (policy == "par_unseq") {
                hpx::rotate(hpx::execution::par_unseq, out_data, out_data + rotate_point, out_data + n);
            } else {
                hpx::rotate(hpx::execution::seq, out_data, out_data + rotate_point, out_data + n);
            }
        }
    }

    return result;
}

// nonzero - Find indices of non-zero elements using hpx::find_if
std::shared_ptr<ndarray> nonzero(
    std::shared_ptr<ndarray> arr,
    std::string const& policy = "seq"  // Sequential for deterministic ordering
) {
    if (arr->ndim() != 1) {
        throw std::runtime_error("nonzero currently only supports 1D arrays");
    }
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("nonzero currently only supports float64");
    }

    py::ssize_t n = arr->size();
    double const* data = arr->typed_data<double>();

    // Find all non-zero indices
    std::vector<int64_t> indices;
    indices.reserve(n);  // Worst case

    {
        py::gil_scoped_release release;

        // Note: hpx::find_if returns iterator to first match, we need all matches
        // Use sequential scan for now (parallel version would need atomic ops)
        for (py::ssize_t i = 0; i < n; ++i) {
            if (data[i] != 0.0) {
                indices.push_back(i);
            }
        }
    }

    // Create output array
    py::ssize_t count = static_cast<py::ssize_t>(indices.size());
    std::vector<py::ssize_t> shape = {count};
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<int64_t>());

    if (count > 0) {
        std::memcpy(result->data(), indices.data(), count * sizeof(int64_t));
    }

    return result;
}

// nth_element - Partial sort to find nth element (useful for median/percentile)
double nth_element(
    std::shared_ptr<ndarray> arr,
    py::ssize_t n_idx,
    std::string const& policy = "par"
) {
    if (arr->ndim() != 1) {
        throw std::runtime_error("nth_element currently only supports 1D arrays");
    }
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("nth_element currently only supports float64");
    }

    py::ssize_t size = arr->size();
    if (size == 0) {
        throw std::runtime_error("nth_element: empty array");
    }
    if (n_idx < 0) n_idx += size;
    if (n_idx < 0 || n_idx >= size) {
        throw std::out_of_range("nth_element: index out of range");
    }

    double const* data = arr->typed_data<double>();

    // Create copy for partial sorting
    std::vector<double> copy(data, data + size);

    double result;
    {
        py::gil_scoped_release release;

        // nth_element puts the nth element in sorted position
        // All elements before it are <= and all after are >=
        std::nth_element(copy.begin(), copy.begin() + n_idx, copy.end());
        result = copy[n_idx];
    }

    return result;
}

// median - Find median using nth_element
double median(
    std::shared_ptr<ndarray> arr
) {
    if (arr->ndim() != 1) {
        throw std::runtime_error("median currently only supports 1D arrays");
    }
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("median currently only supports float64");
    }

    py::ssize_t n = arr->size();
    if (n == 0) {
        throw std::runtime_error("median: empty array");
    }

    double const* data = arr->typed_data<double>();
    std::vector<double> copy(data, data + n);

    double result;
    {
        py::gil_scoped_release release;

        if (n % 2 == 1) {
            // Odd length: return middle element
            py::ssize_t mid = n / 2;
            std::nth_element(copy.begin(), copy.begin() + mid, copy.end());
            result = copy[mid];
        } else {
            // Even length: return average of two middle elements
            py::ssize_t mid1 = n / 2 - 1;
            py::ssize_t mid2 = n / 2;
            std::nth_element(copy.begin(), copy.begin() + mid1, copy.end());
            double val1 = copy[mid1];
            std::nth_element(copy.begin() + mid1 + 1, copy.begin() + mid2, copy.end());
            double val2 = copy[mid2];
            result = (val1 + val2) / 2.0;
        }
    }

    return result;
}

// percentile - Find percentile value
double percentile(
    std::shared_ptr<ndarray> arr,
    double q  // Percentile in [0, 100]
) {
    if (arr->ndim() != 1) {
        throw std::runtime_error("percentile currently only supports 1D arrays");
    }
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("percentile currently only supports float64");
    }
    if (q < 0.0 || q > 100.0) {
        throw std::invalid_argument("percentile: q must be in [0, 100]");
    }

    py::ssize_t n = arr->size();
    if (n == 0) {
        throw std::runtime_error("percentile: empty array");
    }

    double const* data = arr->typed_data<double>();
    std::vector<double> copy(data, data + n);

    double result;
    {
        py::gil_scoped_release release;

        // Linear interpolation method (same as numpy default)
        double idx = (q / 100.0) * (n - 1);
        py::ssize_t lo = static_cast<py::ssize_t>(std::floor(idx));
        py::ssize_t hi = static_cast<py::ssize_t>(std::ceil(idx));
        double frac = idx - lo;

        if (lo == hi) {
            std::nth_element(copy.begin(), copy.begin() + lo, copy.end());
            result = copy[lo];
        } else {
            std::nth_element(copy.begin(), copy.begin() + lo, copy.end());
            double val_lo = copy[lo];
            std::nth_element(copy.begin() + lo + 1, copy.begin() + hi, copy.end());
            double val_hi = copy[hi];
            result = val_lo + frac * (val_hi - val_lo);
        }
    }

    return result;
}

// -----------------------------------------------------------------------------
// Tier 3 Algorithms
// -----------------------------------------------------------------------------

// merge - Merge two sorted arrays using hpx::merge
std::shared_ptr<ndarray> merge(
    std::shared_ptr<ndarray> arr1,
    std::shared_ptr<ndarray> arr2,
    std::string const& policy = "par"
) {
    if (arr1->ndim() != 1 || arr2->ndim() != 1) {
        throw std::runtime_error("merge currently only supports 1D arrays");
    }
    if (arr1->dtype().kind() != 'f' || arr1->dtype().itemsize() != 8) {
        throw std::runtime_error("merge currently only supports float64");
    }
    if (arr2->dtype().kind() != 'f' || arr2->dtype().itemsize() != 8) {
        throw std::runtime_error("merge currently only supports float64");
    }

    py::ssize_t n1 = arr1->size();
    py::ssize_t n2 = arr2->size();
    double const* data1 = arr1->typed_data<double>();
    double const* data2 = arr2->typed_data<double>();

    // Create output array
    std::vector<py::ssize_t> shape = {n1 + n2};
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    double* out_data = result->typed_data<double>();

    {
        py::gil_scoped_release release;

        if (policy == "par") {
            hpx::merge(hpx::execution::par, data1, data1 + n1, data2, data2 + n2, out_data);
        } else if (policy == "par_unseq") {
            hpx::merge(hpx::execution::par_unseq, data1, data1 + n1, data2, data2 + n2, out_data);
        } else {
            hpx::merge(hpx::execution::seq, data1, data1 + n1, data2, data2 + n2, out_data);
        }
    }

    return result;
}

// set_difference - Elements in arr1 but not in arr2
std::shared_ptr<ndarray> set_difference(
    std::shared_ptr<ndarray> arr1,
    std::shared_ptr<ndarray> arr2,
    std::string const& policy = "par"
) {
    if (arr1->ndim() != 1 || arr2->ndim() != 1) {
        throw std::runtime_error("set_difference only supports 1D arrays");
    }
    if (arr1->dtype().kind() != 'f' || arr1->dtype().itemsize() != 8) {
        throw std::runtime_error("set_difference only supports float64");
    }
    if (arr2->dtype().kind() != 'f' || arr2->dtype().itemsize() != 8) {
        throw std::runtime_error("set_difference only supports float64");
    }

    py::ssize_t n1 = arr1->size();
    py::ssize_t n2 = arr2->size();
    double const* data1 = arr1->typed_data<double>();
    double const* data2 = arr2->typed_data<double>();

    // Sort copies first (set operations require sorted input)
    std::vector<double> sorted1(data1, data1 + n1);
    std::vector<double> sorted2(data2, data2 + n2);
    std::vector<double> result_vec(n1);  // Max possible size

    double* result_end;
    {
        py::gil_scoped_release release;

        std::sort(sorted1.begin(), sorted1.end());
        std::sort(sorted2.begin(), sorted2.end());

        if (policy == "par") {
            result_end = hpx::set_difference(hpx::execution::par,
                sorted1.begin(), sorted1.end(),
                sorted2.begin(), sorted2.end(),
                result_vec.data());
        } else {
            result_end = hpx::set_difference(hpx::execution::seq,
                sorted1.begin(), sorted1.end(),
                sorted2.begin(), sorted2.end(),
                result_vec.data());
        }
    }

    py::ssize_t result_n = result_end - result_vec.data();
    std::vector<py::ssize_t> shape = {result_n};
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    if (result_n > 0) {
        std::memcpy(result->data(), result_vec.data(), result_n * sizeof(double));
    }

    return result;
}

// set_intersection - Elements in both arr1 and arr2
std::shared_ptr<ndarray> set_intersection(
    std::shared_ptr<ndarray> arr1,
    std::shared_ptr<ndarray> arr2,
    std::string const& policy = "par"
) {
    if (arr1->ndim() != 1 || arr2->ndim() != 1) {
        throw std::runtime_error("set_intersection only supports 1D arrays");
    }
    if (arr1->dtype().kind() != 'f' || arr1->dtype().itemsize() != 8) {
        throw std::runtime_error("set_intersection only supports float64");
    }
    if (arr2->dtype().kind() != 'f' || arr2->dtype().itemsize() != 8) {
        throw std::runtime_error("set_intersection only supports float64");
    }

    py::ssize_t n1 = arr1->size();
    py::ssize_t n2 = arr2->size();
    double const* data1 = arr1->typed_data<double>();
    double const* data2 = arr2->typed_data<double>();

    std::vector<double> sorted1(data1, data1 + n1);
    std::vector<double> sorted2(data2, data2 + n2);
    std::vector<double> result_vec(std::min(n1, n2));

    double* result_end;
    {
        py::gil_scoped_release release;

        std::sort(sorted1.begin(), sorted1.end());
        std::sort(sorted2.begin(), sorted2.end());

        if (policy == "par") {
            result_end = hpx::set_intersection(hpx::execution::par,
                sorted1.begin(), sorted1.end(),
                sorted2.begin(), sorted2.end(),
                result_vec.data());
        } else {
            result_end = hpx::set_intersection(hpx::execution::seq,
                sorted1.begin(), sorted1.end(),
                sorted2.begin(), sorted2.end(),
                result_vec.data());
        }
    }

    py::ssize_t result_n = result_end - result_vec.data();
    std::vector<py::ssize_t> shape = {result_n};
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    if (result_n > 0) {
        std::memcpy(result->data(), result_vec.data(), result_n * sizeof(double));
    }

    return result;
}

// set_union - Elements in arr1 or arr2 (or both)
std::shared_ptr<ndarray> set_union(
    std::shared_ptr<ndarray> arr1,
    std::shared_ptr<ndarray> arr2,
    std::string const& policy = "par"
) {
    if (arr1->ndim() != 1 || arr2->ndim() != 1) {
        throw std::runtime_error("set_union only supports 1D arrays");
    }
    if (arr1->dtype().kind() != 'f' || arr1->dtype().itemsize() != 8) {
        throw std::runtime_error("set_union only supports float64");
    }
    if (arr2->dtype().kind() != 'f' || arr2->dtype().itemsize() != 8) {
        throw std::runtime_error("set_union only supports float64");
    }

    py::ssize_t n1 = arr1->size();
    py::ssize_t n2 = arr2->size();
    double const* data1 = arr1->typed_data<double>();
    double const* data2 = arr2->typed_data<double>();

    std::vector<double> sorted1(data1, data1 + n1);
    std::vector<double> sorted2(data2, data2 + n2);
    std::vector<double> result_vec(n1 + n2);

    double* result_end;
    {
        py::gil_scoped_release release;

        std::sort(sorted1.begin(), sorted1.end());
        std::sort(sorted2.begin(), sorted2.end());

        if (policy == "par") {
            result_end = hpx::set_union(hpx::execution::par,
                sorted1.begin(), sorted1.end(),
                sorted2.begin(), sorted2.end(),
                result_vec.data());
        } else {
            result_end = hpx::set_union(hpx::execution::seq,
                sorted1.begin(), sorted1.end(),
                sorted2.begin(), sorted2.end(),
                result_vec.data());
        }
    }

    py::ssize_t result_n = result_end - result_vec.data();
    std::vector<py::ssize_t> shape = {result_n};
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    if (result_n > 0) {
        std::memcpy(result->data(), result_vec.data(), result_n * sizeof(double));
    }

    return result;
}

// set_symmetric_difference - Elements in arr1 or arr2 but not both
std::shared_ptr<ndarray> set_symmetric_difference(
    std::shared_ptr<ndarray> arr1,
    std::shared_ptr<ndarray> arr2,
    std::string const& policy = "par"
) {
    if (arr1->ndim() != 1 || arr2->ndim() != 1) {
        throw std::runtime_error("set_symmetric_difference only supports 1D arrays");
    }
    if (arr1->dtype().kind() != 'f' || arr1->dtype().itemsize() != 8) {
        throw std::runtime_error("set_symmetric_difference only supports float64");
    }
    if (arr2->dtype().kind() != 'f' || arr2->dtype().itemsize() != 8) {
        throw std::runtime_error("set_symmetric_difference only supports float64");
    }

    py::ssize_t n1 = arr1->size();
    py::ssize_t n2 = arr2->size();
    double const* data1 = arr1->typed_data<double>();
    double const* data2 = arr2->typed_data<double>();

    std::vector<double> sorted1(data1, data1 + n1);
    std::vector<double> sorted2(data2, data2 + n2);
    std::vector<double> result_vec(n1 + n2);

    double* result_end;
    {
        py::gil_scoped_release release;

        std::sort(sorted1.begin(), sorted1.end());
        std::sort(sorted2.begin(), sorted2.end());

        if (policy == "par") {
            result_end = hpx::set_symmetric_difference(hpx::execution::par,
                sorted1.begin(), sorted1.end(),
                sorted2.begin(), sorted2.end(),
                result_vec.data());
        } else {
            result_end = hpx::set_symmetric_difference(hpx::execution::seq,
                sorted1.begin(), sorted1.end(),
                sorted2.begin(), sorted2.end(),
                result_vec.data());
        }
    }

    py::ssize_t result_n = result_end - result_vec.data();
    std::vector<py::ssize_t> shape = {result_n};
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    if (result_n > 0) {
        std::memcpy(result->data(), result_vec.data(), result_n * sizeof(double));
    }

    return result;
}

// includes - Test if arr1 contains all elements of arr2
bool includes(
    std::shared_ptr<ndarray> arr1,
    std::shared_ptr<ndarray> arr2,
    std::string const& policy = "par"
) {
    if (arr1->ndim() != 1 || arr2->ndim() != 1) {
        throw std::runtime_error("includes only supports 1D arrays");
    }
    if (arr1->dtype().kind() != 'f' || arr1->dtype().itemsize() != 8) {
        throw std::runtime_error("includes only supports float64");
    }
    if (arr2->dtype().kind() != 'f' || arr2->dtype().itemsize() != 8) {
        throw std::runtime_error("includes only supports float64");
    }

    py::ssize_t n1 = arr1->size();
    py::ssize_t n2 = arr2->size();
    double const* data1 = arr1->typed_data<double>();
    double const* data2 = arr2->typed_data<double>();

    std::vector<double> sorted1(data1, data1 + n1);
    std::vector<double> sorted2(data2, data2 + n2);

    bool result;
    {
        py::gil_scoped_release release;

        std::sort(sorted1.begin(), sorted1.end());
        std::sort(sorted2.begin(), sorted2.end());

        if (policy == "par") {
            result = hpx::includes(hpx::execution::par,
                sorted1.begin(), sorted1.end(),
                sorted2.begin(), sorted2.end());
        } else {
            result = hpx::includes(hpx::execution::seq,
                sorted1.begin(), sorted1.end(),
                sorted2.begin(), sorted2.end());
        }
    }

    return result;
}

// isin - Test membership of arr elements in test_arr
std::shared_ptr<ndarray> isin(
    std::shared_ptr<ndarray> arr,
    std::shared_ptr<ndarray> test_arr
) {
    if (arr->ndim() != 1 || test_arr->ndim() != 1) {
        throw std::runtime_error("isin only supports 1D arrays");
    }
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("isin only supports float64");
    }
    if (test_arr->dtype().kind() != 'f' || test_arr->dtype().itemsize() != 8) {
        throw std::runtime_error("isin only supports float64");
    }

    py::ssize_t n = arr->size();
    py::ssize_t n_test = test_arr->size();
    double const* data = arr->typed_data<double>();
    double const* test_data = test_arr->typed_data<double>();

    // Sort test array for binary search
    std::vector<double> sorted_test(test_data, test_data + n_test);
    std::sort(sorted_test.begin(), sorted_test.end());

    // Create boolean result array
    std::vector<py::ssize_t> shape = {n};
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<bool>());
    bool* out_data = result->typed_data<bool>();

    {
        py::gil_scoped_release release;

        // Check each element using binary search
        for (py::ssize_t i = 0; i < n; ++i) {
            out_data[i] = std::binary_search(sorted_test.begin(), sorted_test.end(), data[i]);
        }
    }

    return result;
}

// searchsorted - Find indices where elements should be inserted
std::shared_ptr<ndarray> searchsorted(
    std::shared_ptr<ndarray> arr,     // Sorted array
    std::shared_ptr<ndarray> values,  // Values to insert
    std::string const& side = "left"  // "left" or "right"
) {
    if (arr->ndim() != 1 || values->ndim() != 1) {
        throw std::runtime_error("searchsorted only supports 1D arrays");
    }
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("searchsorted only supports float64");
    }
    if (values->dtype().kind() != 'f' || values->dtype().itemsize() != 8) {
        throw std::runtime_error("searchsorted only supports float64");
    }

    py::ssize_t n = arr->size();
    py::ssize_t n_vals = values->size();
    double const* data = arr->typed_data<double>();
    double const* vals = values->typed_data<double>();

    std::vector<py::ssize_t> shape = {n_vals};
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<int64_t>());
    int64_t* out_data = result->typed_data<int64_t>();

    bool use_left = (side == "left");

    {
        py::gil_scoped_release release;

        for (py::ssize_t i = 0; i < n_vals; ++i) {
            double val = vals[i];
            if (use_left) {
                // Find first position where val <= arr[pos]
                auto it = std::lower_bound(data, data + n, val);
                out_data[i] = it - data;
            } else {
                // Find first position where val < arr[pos]
                auto it = std::upper_bound(data, data + n, val);
                out_data[i] = it - data;
            }
        }
    }

    return result;
}

// partition - Partition array around pivot
std::shared_ptr<ndarray> partition_arr(
    std::shared_ptr<ndarray> arr,
    double pivot
) {
    if (arr->ndim() != 1) {
        throw std::runtime_error("partition only supports 1D arrays");
    }
    if (arr->dtype().kind() != 'f' || arr->dtype().itemsize() != 8) {
        throw std::runtime_error("partition only supports float64");
    }

    py::ssize_t n = arr->size();
    double const* data = arr->typed_data<double>();

    std::vector<py::ssize_t> shape = {n};
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    double* out_data = result->typed_data<double>();

    std::memcpy(out_data, data, n * sizeof(double));

    {
        py::gil_scoped_release release;

        std::partition(out_data, out_data + n, [pivot](double x) { return x < pivot; });
    }

    return result;
}

// Random number generation
namespace random {

// Thread-local random engine
thread_local std::mt19937_64 rng{std::random_device{}()};

std::shared_ptr<ndarray> uniform(double low, double high, std::vector<py::ssize_t> shape) {
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    auto* data = result->typed_data<double>();
    py::ssize_t n = result->size();

    std::uniform_real_distribution<double> dist(low, high);

    // Generate random numbers (parallel generation with thread-local RNG)
    for (py::ssize_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }

    return result;
}

std::shared_ptr<ndarray> randn(std::vector<py::ssize_t> shape) {
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<double>());
    auto* data = result->typed_data<double>();
    py::ssize_t n = result->size();

    std::normal_distribution<double> dist(0.0, 1.0);

    for (py::ssize_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }

    return result;
}

std::shared_ptr<ndarray> randint(int64_t low, int64_t high, std::vector<py::ssize_t> shape) {
    auto result = std::make_shared<ndarray>(shape, py::dtype::of<int64_t>());
    auto* data = result->typed_data<int64_t>();
    py::ssize_t n = result->size();

    std::uniform_int_distribution<int64_t> dist(low, high - 1);  // high is exclusive

    for (py::ssize_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }

    return result;
}

std::shared_ptr<ndarray> rand(std::vector<py::ssize_t> shape) {
    return uniform(0.0, 1.0, shape);
}

void seed(uint64_t s) {
    rng.seed(s);
}

}  // namespace random

// Helper structs for operations (at namespace scope for template compatibility)
struct MaxOp {
    template<typename T> T operator()(T x, T y) const { return x > y ? x : y; }
};

struct MinOp {
    template<typename T> T operator()(T x, T y) const { return x < y ? x : y; }
};

struct ClipOp {
    double min_v, max_v;
    template<typename T> T operator()(T x) const {
        if (x < static_cast<T>(min_v)) return static_cast<T>(min_v);
        if (x > static_cast<T>(max_v)) return static_cast<T>(max_v);
        return x;
    }
};

// Element-wise maximum of two arrays
std::shared_ptr<ndarray> maximum(std::shared_ptr<ndarray> a, py::object b) {
    return ops::dispatch_binary_py(*a, b, MaxOp{});
}

// Element-wise minimum of two arrays
std::shared_ptr<ndarray> minimum(std::shared_ptr<ndarray> a, py::object b) {
    return ops::dispatch_binary_py(*a, b, MinOp{});
}

// Clip values to range
std::shared_ptr<ndarray> clip(std::shared_ptr<ndarray> arr, py::object a_min, py::object a_max) {
    double min_val = a_min.cast<double>();
    double max_val = a_max.cast<double>();
    return ops::dispatch_unary(*arr, ClipOp{min_val, max_val});
}

// Power function with scalar exponent
std::shared_ptr<ndarray> power(std::shared_ptr<ndarray> arr, py::object exponent) {
    return ops::dispatch_binary_py(*arr, exponent, ops::Pow{});
}

// where - conditional selection
std::shared_ptr<ndarray> where_arr(std::shared_ptr<ndarray> condition,
                                    std::shared_ptr<ndarray> x,
                                    std::shared_ptr<ndarray> y) {
    if (!ops::shapes_equal(condition->shape(), x->shape()) ||
        !ops::shapes_equal(condition->shape(), y->shape())) {
        throw std::runtime_error("Shapes must match for where()");
    }

    auto result = std::make_shared<ndarray>(x->shape(), x->dtype());

    char kind = x->dtype().kind();
    auto itemsize = x->dtype().itemsize();
    bool const* cond = condition->typed_data<bool>();

    py::gil_scoped_release release;

    if (kind == 'f') {
        if (itemsize == 8) {
            double const* x_ptr = x->typed_data<double>();
            double const* y_ptr = y->typed_data<double>();
            double* r_ptr = result->typed_data<double>();
            for (py::ssize_t i = 0; i < result->size(); ++i) {
                r_ptr[i] = cond[i] ? x_ptr[i] : y_ptr[i];
            }
        } else if (itemsize == 4) {
            float const* x_ptr = x->typed_data<float>();
            float const* y_ptr = y->typed_data<float>();
            float* r_ptr = result->typed_data<float>();
            for (py::ssize_t i = 0; i < result->size(); ++i) {
                r_ptr[i] = cond[i] ? x_ptr[i] : y_ptr[i];
            }
        }
    } else if (kind == 'i') {
        if (itemsize == 8) {
            int64_t const* x_ptr = x->typed_data<int64_t>();
            int64_t const* y_ptr = y->typed_data<int64_t>();
            int64_t* r_ptr = result->typed_data<int64_t>();
            for (py::ssize_t i = 0; i < result->size(); ++i) {
                r_ptr[i] = cond[i] ? x_ptr[i] : y_ptr[i];
            }
        } else if (itemsize == 4) {
            int32_t const* x_ptr = x->typed_data<int32_t>();
            int32_t const* y_ptr = y->typed_data<int32_t>();
            int32_t* r_ptr = result->typed_data<int32_t>();
            for (py::ssize_t i = 0; i < result->size(); ++i) {
                r_ptr[i] = cond[i] ? x_ptr[i] : y_ptr[i];
            }
        }
    } else {
        py::gil_scoped_acquire acquire;
        throw std::runtime_error("Unsupported dtype for where");
    }

    return result;
}

}  // namespace hpxpy

void bind_algorithms(py::module_& m) {
    m.def("_sum", &hpxpy::sum,
        py::arg("arr"),
        py::arg("policy") = "seq",
        R"pbdoc(
            Sum of array elements.

            Parameters
            ----------
            arr : ndarray
                Input array.
            policy : str
                Execution policy: "seq" (sequential, default), "par" (parallel),
                or "par_unseq" (parallel + SIMD).

            Returns
            -------
            scalar
                Sum of all elements.
        )pbdoc");

    m.def("_sum_axis", &hpxpy::sum_axis,
        py::arg("arr"),
        py::arg("axis"),
        py::arg("keepdims") = false,
        R"pbdoc(
            Sum of array elements along an axis.

            Parameters
            ----------
            arr : ndarray
                Input array.
            axis : int
                Axis along which to sum.
            keepdims : bool
                If True, retain reduced dimensions with size 1.

            Returns
            -------
            ndarray
                Sum along the specified axis.
        )pbdoc");

    m.def("_prod", &hpxpy::prod,
        py::arg("arr"),
        py::arg("policy") = "seq",
        "Product of array elements. policy: 'seq', 'par', or 'par_unseq'.");

    m.def("_min", &hpxpy::min,
        py::arg("arr"),
        py::arg("policy") = "seq",
        "Minimum of array elements. policy: 'seq', 'par', or 'par_unseq'.");

    m.def("_max", &hpxpy::max,
        py::arg("arr"),
        py::arg("policy") = "seq",
        "Maximum of array elements. policy: 'seq', 'par', or 'par_unseq'.");

    m.def("_sort", &hpxpy::sort,
        py::arg("arr"),
        py::arg("policy") = "seq",
        "Return a sorted copy of the array. policy: 'seq', 'par', or 'par_unseq'.");

    m.def("_count", &hpxpy::count,
        py::arg("arr"), py::arg("value"),
        py::arg("policy") = "seq",
        "Count occurrences of a value. policy: 'seq', 'par', or 'par_unseq'.");

    // Math functions
    m.def("_sqrt", &hpxpy::sqrt_arr, py::arg("arr"), "Element-wise square root.");
    m.def("_square", &hpxpy::square_arr, py::arg("arr"), "Element-wise square.");
    m.def("_exp", &hpxpy::exp_arr, py::arg("arr"), "Element-wise exponential.");
    m.def("_exp2", &hpxpy::exp2_arr, py::arg("arr"), "Element-wise 2**x.");
    m.def("_log", &hpxpy::log_arr, py::arg("arr"), "Element-wise natural logarithm.");
    m.def("_log2", &hpxpy::log2_arr, py::arg("arr"), "Element-wise base-2 logarithm.");
    m.def("_log10", &hpxpy::log10_arr, py::arg("arr"), "Element-wise base-10 logarithm.");
    m.def("_sin", &hpxpy::sin_arr, py::arg("arr"), "Element-wise sine.");
    m.def("_cos", &hpxpy::cos_arr, py::arg("arr"), "Element-wise cosine.");
    m.def("_tan", &hpxpy::tan_arr, py::arg("arr"), "Element-wise tangent.");
    m.def("_arcsin", &hpxpy::arcsin_arr, py::arg("arr"), "Element-wise inverse sine.");
    m.def("_arccos", &hpxpy::arccos_arr, py::arg("arr"), "Element-wise inverse cosine.");
    m.def("_arctan", &hpxpy::arctan_arr, py::arg("arr"), "Element-wise inverse tangent.");
    m.def("_sinh", &hpxpy::sinh_arr, py::arg("arr"), "Element-wise hyperbolic sine.");
    m.def("_cosh", &hpxpy::cosh_arr, py::arg("arr"), "Element-wise hyperbolic cosine.");
    m.def("_tanh", &hpxpy::tanh_arr, py::arg("arr"), "Element-wise hyperbolic tangent.");
    m.def("_floor", &hpxpy::floor_arr, py::arg("arr"), "Element-wise floor.");
    m.def("_ceil", &hpxpy::ceil_arr, py::arg("arr"), "Element-wise ceiling.");
    m.def("_trunc", &hpxpy::trunc_arr, py::arg("arr"), "Element-wise truncation.");
    m.def("_abs", &hpxpy::abs_arr, py::arg("arr"), "Element-wise absolute value.");
    m.def("_sign", &hpxpy::sign_arr, py::arg("arr"), "Element-wise sign.");

    // Scan operations
    m.def("_cumsum", &hpxpy::cumsum, py::arg("arr"), "Cumulative sum.");
    m.def("_cumprod", &hpxpy::cumprod, py::arg("arr"), "Cumulative product.");

    // Additional functions
    m.def("_maximum", &hpxpy::maximum, py::arg("a"), py::arg("b"),
        "Element-wise maximum of two arrays.");
    m.def("_minimum", &hpxpy::minimum, py::arg("a"), py::arg("b"),
        "Element-wise minimum of two arrays.");
    m.def("_clip", &hpxpy::clip, py::arg("arr"), py::arg("a_min"), py::arg("a_max"),
        "Clip values to a range.");
    m.def("_power", &hpxpy::power, py::arg("arr"), py::arg("exponent"),
        "Element-wise power.");
    m.def("_where", &hpxpy::where_arr, py::arg("condition"), py::arg("x"), py::arg("y"),
        "Return elements chosen from x or y depending on condition.");

    // New HPX algorithm exposures
    m.def("_any", &hpxpy::any_of,
        py::arg("arr"),
        py::arg("policy") = "seq",
        "Check if any element is truthy (non-zero). policy: 'seq', 'par', 'par_unseq'.");

    m.def("_all", &hpxpy::all_of,
        py::arg("arr"),
        py::arg("policy") = "seq",
        "Check if all elements are truthy (non-zero). policy: 'seq', 'par', 'par_unseq'.");

    m.def("_argmin", &hpxpy::argmin,
        py::arg("arr"),
        py::arg("policy") = "seq",
        "Return index of minimum element. policy: 'seq', 'par', 'par_unseq'.");

    m.def("_argmax", &hpxpy::argmax,
        py::arg("arr"),
        py::arg("policy") = "seq",
        "Return index of maximum element. policy: 'seq', 'par', 'par_unseq'.");

    m.def("_argsort", &hpxpy::argsort,
        py::arg("arr"),
        py::arg("policy") = "seq",
        "Return indices that would sort the array. policy: 'seq', 'par', 'par_unseq'.");

    m.def("_diff", &hpxpy::diff,
        py::arg("arr"),
        py::arg("policy") = "seq",
        "Compute n-th discrete difference (adjacent_difference). policy: 'seq', 'par', 'par_unseq'.");

    m.def("_unique", &hpxpy::unique,
        py::arg("arr"),
        py::arg("policy") = "seq",
        "Return sorted unique elements. policy: 'seq', 'par', 'par_unseq'.");

    m.def("_inclusive_scan", &hpxpy::inclusive_scan,
        py::arg("arr"),
        py::arg("op") = "add",
        py::arg("policy") = "par",
        R"pbdoc(
            Inclusive scan (running accumulation).

            Parameters
            ----------
            arr : ndarray
                Input array.
            op : str
                Binary operation: "add", "mul", "min", "max".
            policy : str
                Execution policy: "seq", "par", "par_unseq".

            Returns
            -------
            ndarray
                Scanned array where result[i] = op(arr[0], arr[1], ..., arr[i]).
        )pbdoc");

    m.def("_exclusive_scan", &hpxpy::exclusive_scan,
        py::arg("arr"),
        py::arg("init"),
        py::arg("op") = "add",
        py::arg("policy") = "par",
        R"pbdoc(
            Exclusive scan (running accumulation excluding current element).

            Parameters
            ----------
            arr : ndarray
                Input array.
            init : scalar
                Initial value.
            op : str
                Binary operation: "add", "mul".
            policy : str
                Execution policy: "seq", "par", "par_unseq".

            Returns
            -------
            ndarray
                Scanned array where result[i] = op(init, arr[0], arr[1], ..., arr[i-1]).
        )pbdoc");

    m.def("_transform_reduce", &hpxpy::transform_reduce,
        py::arg("arr"),
        py::arg("transform"),
        py::arg("reduce") = "add",
        py::arg("policy") = "par",
        R"pbdoc(
            Fused transform and reduce in one pass.

            Parameters
            ----------
            arr : ndarray
                Input array.
            transform : str
                Unary transform: "square", "abs", "negate", "identity".
            reduce : str
                Binary reduction: "add", "mul", "min", "max".
            policy : str
                Execution policy: "seq", "par", "par_unseq".

            Returns
            -------
            scalar
                Result of reduce(transform(arr[0]), transform(arr[1]), ...).

            Examples
            --------
            Sum of squares: transform_reduce(arr, "square", "add")
            Sum of absolute values: transform_reduce(arr, "abs", "add")
        )pbdoc");

    m.def("_reduce_by_key", &hpxpy::reduce_by_key,
        py::arg("keys"),
        py::arg("values"),
        py::arg("reduce_op") = "add",
        py::arg("policy") = "par",
        R"pbdoc(
            Reduce values grouped by keys.

            Parameters
            ----------
            keys : ndarray
                Key array (float64, 1D).
            values : ndarray
                Value array (float64, 1D, same size as keys).
            reduce_op : str
                Reduction operation: "add", "mul", "min", "max".
            policy : str
                Execution policy: "seq", "par", "par_unseq".

            Returns
            -------
            tuple[ndarray, ndarray]
                (unique_keys, reduced_values)

            Examples
            --------
            >>> keys = hpx.array([1, 2, 1, 2, 1])
            >>> values = hpx.array([10, 20, 30, 40, 50])
            >>> uk, rv = hpx.reduce_by_key(keys, values, "add")
            >>> # uk = [1, 2], rv = [90, 60]  (1: 10+30+50, 2: 20+40)
        )pbdoc");

    // Tier 2 Algorithms
    m.def("_array_equal", &hpxpy::array_equal,
        py::arg("arr1"),
        py::arg("arr2"),
        py::arg("policy") = "par",
        "True if two arrays have the same shape and elements.");

    m.def("_flip", &hpxpy::flip,
        py::arg("arr"),
        py::arg("policy") = "par",
        "Reverse array elements using hpx::reverse.");

    m.def("_stable_sort", &hpxpy::stable_sort,
        py::arg("arr"),
        py::arg("policy") = "par",
        "Stable sort preserving relative order of equal elements.");

    m.def("_rotate", &hpxpy::rotate,
        py::arg("arr"),
        py::arg("shift"),
        py::arg("policy") = "par",
        "Rotate array elements left by shift positions (like std::rotate).");

    m.def("_nonzero", &hpxpy::nonzero,
        py::arg("arr"),
        py::arg("policy") = "seq",
        "Return indices of non-zero elements.");

    m.def("_nth_element", &hpxpy::nth_element,
        py::arg("arr"),
        py::arg("n"),
        py::arg("policy") = "par",
        "Find the nth element (0-indexed) as if array were sorted.");

    m.def("_median", &hpxpy::median,
        py::arg("arr"),
        "Find median using nth_element (O(n) average).");

    m.def("_percentile", &hpxpy::percentile,
        py::arg("arr"),
        py::arg("q"),
        "Find percentile value (q in [0, 100]) with linear interpolation.");

    // Tier 3 Algorithms
    m.def("_merge", &hpxpy::merge,
        py::arg("arr1"),
        py::arg("arr2"),
        py::arg("policy") = "par",
        "Merge two sorted arrays.");

    m.def("_setdiff1d", &hpxpy::set_difference,
        py::arg("arr1"),
        py::arg("arr2"),
        py::arg("policy") = "par",
        "Set difference: elements in arr1 but not in arr2.");

    m.def("_intersect1d", &hpxpy::set_intersection,
        py::arg("arr1"),
        py::arg("arr2"),
        py::arg("policy") = "par",
        "Set intersection: elements in both arr1 and arr2.");

    m.def("_union1d", &hpxpy::set_union,
        py::arg("arr1"),
        py::arg("arr2"),
        py::arg("policy") = "par",
        "Set union: elements in arr1 or arr2 (or both).");

    m.def("_setxor1d", &hpxpy::set_symmetric_difference,
        py::arg("arr1"),
        py::arg("arr2"),
        py::arg("policy") = "par",
        "Set symmetric difference: elements in arr1 or arr2 but not both.");

    m.def("_includes", &hpxpy::includes,
        py::arg("arr1"),
        py::arg("arr2"),
        py::arg("policy") = "par",
        "Test if arr1 contains all elements of arr2.");

    m.def("_isin", &hpxpy::isin,
        py::arg("arr"),
        py::arg("test_arr"),
        "Test membership of arr elements in test_arr.");

    m.def("_searchsorted", &hpxpy::searchsorted,
        py::arg("arr"),
        py::arg("values"),
        py::arg("side") = "left",
        "Find indices where values should be inserted in sorted arr.");

    m.def("_partition", &hpxpy::partition_arr,
        py::arg("arr"),
        py::arg("pivot"),
        "Partition array around pivot value.");

    // Random submodule
    auto random = m.def_submodule("random", "Random number generation.");
    random.def("_uniform", &hpxpy::random::uniform,
        py::arg("low"), py::arg("high"), py::arg("shape"),
        "Uniform distribution over [low, high).");
    random.def("_randn", &hpxpy::random::randn, py::arg("shape"),
        "Standard normal distribution.");
    random.def("_randint", &hpxpy::random::randint,
        py::arg("low"), py::arg("high"), py::arg("shape"),
        "Random integers from [low, high).");
    random.def("_rand", &hpxpy::random::rand, py::arg("shape"),
        "Uniform distribution over [0, 1).");
    random.def("seed", &hpxpy::random::seed, py::arg("seed"),
        "Seed the random number generator.");
}

void bind_execution(py::module_& m) {
    // Create execution submodule
    auto exec = m.def_submodule("execution",
        "Execution policies for controlling parallel execution.");

    // Phase 2: Full execution policy support
    py::class_<hpx::execution::sequenced_policy>(exec, "sequenced_policy",
        "Sequential execution policy.")
        .def(py::init<>());

    py::class_<hpx::execution::parallel_policy>(exec, "parallel_policy",
        "Parallel execution policy.")
        .def(py::init<>());

    py::class_<hpx::execution::parallel_unsequenced_policy>(exec, "parallel_unsequenced_policy",
        "Parallel unsequenced execution policy.")
        .def(py::init<>());

    exec.attr("seq") = hpx::execution::seq;
    exec.attr("par") = hpx::execution::par;
    exec.attr("par_unseq") = hpx::execution::par_unseq;
}
