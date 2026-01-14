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
// policy: "seq" (default, deterministic), "par" (parallel), "par_unseq" (parallel + SIMD)
py::object sum(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->size() == 0) {
        return py::cast(0.0);
    }

    // Release GIL during computation
    py::gil_scoped_release release;

    return dispatch_dtype(*arr, [&policy](auto const* data, py::ssize_t size) -> py::object {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(data)>>;

        T result = with_execution_policy(policy, [&](auto exec) {
            return hpx::reduce(exec, data, data + size, T{0});
        });

        py::gil_scoped_acquire acquire;
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
// policy: "seq" (default, deterministic), "par" (parallel), "par_unseq" (parallel + SIMD)
py::object prod(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->size() == 0) {
        return py::cast(1.0);
    }

    py::gil_scoped_release release;

    return dispatch_dtype(*arr, [&policy](auto const* data, py::ssize_t size) -> py::object {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(data)>>;

        T result = with_execution_policy(policy, [&](auto exec) {
            return hpx::reduce(exec, data, data + size, T{1}, std::multiplies<T>{});
        });

        py::gil_scoped_acquire acquire;
        return py::cast(result);
    });
}

// Minimum reduction
// policy: "seq" (default), "par" (parallel), "par_unseq" (parallel + SIMD)
py::object min(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->size() == 0) {
        throw std::runtime_error("min() arg is an empty sequence");
    }

    py::gil_scoped_release release;

    return dispatch_dtype(*arr, [&policy](auto const* data, py::ssize_t size) -> py::object {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(data)>>;

        auto result = with_execution_policy(policy, [&](auto exec) {
            return hpx::min_element(exec, data, data + size);
        });

        py::gil_scoped_acquire acquire;
        return py::cast(*result);
    });
}

// Maximum reduction
// policy: "seq" (default), "par" (parallel), "par_unseq" (parallel + SIMD)
py::object max(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->size() == 0) {
        throw std::runtime_error("max() arg is an empty sequence");
    }

    py::gil_scoped_release release;

    return dispatch_dtype(*arr, [&policy](auto const* data, py::ssize_t size) -> py::object {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(data)>>;

        auto result = with_execution_policy(policy, [&](auto exec) {
            return hpx::max_element(exec, data, data + size);
        });

        py::gil_scoped_acquire acquire;
        return py::cast(*result);
    });
}

// Sort (returns new sorted array)
// policy: "seq" (default), "par" (parallel), "par_unseq" (parallel + SIMD)
std::shared_ptr<ndarray> sort(std::shared_ptr<ndarray> arr, std::string const& policy = "seq") {
    if (arr->ndim() != 1) {
        throw std::runtime_error("sort only supports 1D arrays in Phase 1");
    }

    // Create a copy for sorting
    auto result = std::make_shared<ndarray>(arr->shape(), arr->dtype());
    std::memcpy(result->data(), arr->data(), arr->nbytes());

    // Release GIL during sort
    py::gil_scoped_release release;

    char kind = arr->dtype().kind();
    auto itemsize = arr->dtype().itemsize();

    // Helper to sort with execution policy
    auto do_sort = [&policy, size = arr->size()](auto* ptr) {
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
// policy: "seq" (default), "par" (parallel), "par_unseq" (parallel + SIMD)
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
