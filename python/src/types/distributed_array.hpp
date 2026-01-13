// HPXPy - Distribution utilities
//
// SPDX-License-Identifier: BSL-1.0
//
// This header provides distribution policy definitions and utilities
// for distributed array support. In single-locality mode, arrays are
// stored locally with distribution metadata. In multi-locality mode,
// arrays use HPX partitioned_vector for actual distribution.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <hpx/hpx.hpp>
#include <hpx/include/partitioned_vector_predef.hpp>
#include <hpx/include/parallel_reduce.hpp>
#include <hpx/include/parallel_transform.hpp>
#include <hpx/include/parallel_minmax.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>

namespace py = pybind11;

namespace hpxpy {

// Distribution policy types
enum class DistributionPolicy {
    None,       // Local array (no distribution)
    Block,      // Block distribution (contiguous chunks)
    Cyclic      // Cyclic distribution (round-robin)
};

// Convert string to distribution policy
inline DistributionPolicy parse_distribution(py::object const& dist) {
    if (dist.is_none()) {
        return DistributionPolicy::None;
    }

    if (py::isinstance<py::str>(dist)) {
        std::string s = dist.cast<std::string>();
        if (s == "block") return DistributionPolicy::Block;
        if (s == "cyclic") return DistributionPolicy::Cyclic;
        if (s == "none" || s == "local") return DistributionPolicy::None;
        throw std::invalid_argument("Unknown distribution policy: " + s);
    }

    // Could be an enum value
    if (py::isinstance<DistributionPolicy>(dist)) {
        return dist.cast<DistributionPolicy>();
    }

    throw std::invalid_argument("Invalid distribution type");
}

// Distribution info that can be attached to arrays
struct DistributionInfo {
    DistributionPolicy policy = DistributionPolicy::None;
    std::size_t num_partitions = 1;
    std::size_t chunk_size = 0;  // 0 means automatic
    std::uint32_t locality_id = 0;

    bool is_distributed() const {
        return policy != DistributionPolicy::None && num_partitions > 1;
    }
};

// Get current locality ID (for introspection)
inline std::uint32_t get_current_locality() {
    return hpx::get_locality_id();
}

// Get number of localities
inline std::size_t get_num_localities() {
    return hpx::get_num_localities(hpx::launch::sync);
}

// Distributed array class
// In single-locality mode, this stores data locally with distribution metadata.
// The API is designed to work seamlessly when extended to multi-locality.
template<typename T>
class distributed_array {
public:
    using value_type = T;

    // Default constructor
    distributed_array()
        : size_(0)
        , num_partitions_(1)
        , locality_id_(0)
        , policy_(DistributionPolicy::None)
    {}

    // Create with shape and distribution
    distributed_array(
        std::vector<py::ssize_t> shape,
        DistributionPolicy policy = DistributionPolicy::None,
        std::size_t num_partitions = 0)
        : shape_(std::move(shape))
        , policy_(policy)
    {
        // Compute total size
        size_ = 1;
        for (auto dim : shape_) {
            size_ *= dim;
        }

        // Determine number of partitions
        std::size_t num_locs = get_num_localities();
        if (num_partitions == 0) {
            // Default: one partition per locality
            num_partitions_ = (policy == DistributionPolicy::None) ? 1 : num_locs;
        } else {
            num_partitions_ = num_partitions;
        }

        // Allocate local storage
        data_.resize(static_cast<std::size_t>(size_));

        locality_id_ = get_current_locality();
    }

    // Create from local data
    distributed_array(
        std::vector<T> const& local_data,
        std::vector<py::ssize_t> shape,
        DistributionPolicy policy = DistributionPolicy::None)
        : shape_(std::move(shape))
        , policy_(policy)
        , size_(static_cast<py::ssize_t>(local_data.size()))
        , num_partitions_(1)
        , locality_id_(get_current_locality())
        , data_(local_data)
    {
    }

    // Properties
    std::vector<py::ssize_t> const& shape() const { return shape_; }
    py::ssize_t size() const { return size_; }
    py::ssize_t ndim() const { return static_cast<py::ssize_t>(shape_.size()); }
    DistributionPolicy policy() const { return policy_; }
    std::size_t num_partitions() const { return num_partitions_; }
    std::uint32_t locality_id() const { return locality_id_; }

    bool is_distributed() const {
        return policy_ != DistributionPolicy::None &&
               num_partitions_ > 1 &&
               get_num_localities() > 1;
    }

    // Get partition info
    DistributionInfo get_distribution_info() const {
        std::size_t chunk = (num_partitions_ > 0) ?
            static_cast<std::size_t>(size_) / num_partitions_ : static_cast<std::size_t>(size_);
        return DistributionInfo{
            policy_,
            num_partitions_,
            chunk,
            locality_id_
        };
    }

    // Access underlying data
    std::vector<T>& data() { return data_; }
    std::vector<T> const& data() const { return data_; }

    // Convert to local numpy array
    py::array_t<T> to_numpy() const {
        py::array_t<T> result(shape_);
        std::memcpy(result.mutable_data(), data_.data(),
                   data_.size() * sizeof(T));
        return result;
    }

    // Fill with value
    void fill(T value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Element access
    T& operator[](std::size_t i) { return data_[i]; }
    T const& operator[](std::size_t i) const { return data_[i]; }

private:
    std::vector<py::ssize_t> shape_;
    DistributionPolicy policy_ = DistributionPolicy::None;
    py::ssize_t size_ = 0;
    std::size_t num_partitions_ = 1;
    std::uint32_t locality_id_ = 0;
    std::vector<T> data_;
};

// Type aliases for common types
using distributed_array_f64 = distributed_array<double>;
using distributed_array_f32 = distributed_array<float>;
using distributed_array_i64 = distributed_array<std::int64_t>;
using distributed_array_i32 = distributed_array<std::int32_t>;

// ============================================================================
// Phase 11: True Distribution with HPX partitioned_vector
// ============================================================================

// Partitioned array class using HPX partitioned_vector
// This provides true multi-locality distribution when running on multiple nodes.
// Uses the predefined double type from HPX for maximum compatibility.
class partitioned_array {
public:
    using value_type = double;
    using vector_type = hpx::partitioned_vector<double>;

    // Default constructor
    partitioned_array()
        : size_(0)
        , is_initialized_(false)
    {}

    // Create with size and fill value, distributed across all localities
    partitioned_array(std::size_t size, double fill_value = 0.0)
        : size_(size)
        , is_initialized_(true)
    {
        auto localities = hpx::find_all_localities();
        data_ = std::make_shared<vector_type>(
            size, fill_value, hpx::container_layout(localities));
    }

    // Create with size and custom layout
    partitioned_array(std::size_t size, double fill_value,
                      std::vector<hpx::id_type> const& localities)
        : size_(size)
        , is_initialized_(true)
    {
        data_ = std::make_shared<vector_type>(
            size, fill_value, hpx::container_layout(localities));
    }

    // Create from local numpy array, distributing data across localities
    static partitioned_array from_numpy(py::array_t<double> arr) {
        partitioned_array result;
        result.size_ = static_cast<std::size_t>(arr.size());
        result.is_initialized_ = true;

        auto localities = hpx::find_all_localities();

        // Create partitioned vector with default values
        result.data_ = std::make_shared<vector_type>(
            result.size_, 0.0, hpx::container_layout(localities));

        // Copy data from numpy array
        // In multi-locality, this copies to the local partitions first
        auto* src = arr.data();
        std::size_t idx = 0;
        for (auto it = result.data_->begin(); it != result.data_->end(); ++it, ++idx) {
            *it = src[idx];
        }

        return result;
    }

    // Properties
    std::size_t size() const { return size_; }
    bool is_initialized() const { return is_initialized_; }

    std::size_t num_localities() const {
        return hpx::get_num_localities(hpx::launch::sync);
    }

    std::uint32_t locality_id() const {
        return hpx::get_locality_id();
    }

    // Check if actually distributed (multiple localities)
    bool is_distributed() const {
        return is_initialized_ && num_localities() > 1;
    }

    // Get number of partitions
    std::size_t num_partitions() const {
        if (!is_initialized_ || !data_) return 0;
        return data_->partitions().size();
    }

    // Access underlying partitioned_vector (for advanced use)
    vector_type& vector() { return *data_; }
    vector_type const& vector() const { return *data_; }

    // Convert to local numpy array (gathers all data to calling locality)
    py::array_t<double> to_numpy() const {
        if (!is_initialized_ || !data_) {
            return py::array_t<double>(0);
        }

        py::array_t<double> result(size_);
        auto* dst = result.mutable_data();

        std::size_t idx = 0;
        for (auto it = data_->begin(); it != data_->end(); ++it, ++idx) {
            dst[idx] = *it;
        }

        return result;
    }

    // Fill with value (uses simple iteration - works in single/multi-locality)
    void fill(double value) {
        if (!is_initialized_ || !data_) return;

        for (auto it = data_->begin(); it != data_->end(); ++it) {
            *it = value;
        }
    }

    // ========================================================================
    // Distributed Reduction Operations
    // These work on the partitioned_vector. In multi-locality mode, HPX
    // automatically distributes the computation. We use std functors which
    // are serializable, unlike lambdas.
    // ========================================================================

    // Distributed sum - parallel reduction across all localities
    double sum() const {
        if (!is_initialized_ || !data_ || size_ == 0) return 0.0;

        return hpx::reduce(hpx::execution::par,
            data_->begin(), data_->end(), 0.0, std::plus<double>());
    }

    // Distributed mean
    double mean() const {
        if (!is_initialized_ || !data_ || size_ == 0) return 0.0;

        double total = sum();
        return total / static_cast<double>(size_);
    }

    // Distributed min - uses min directly, not lambda
    double min() const {
        if (!is_initialized_ || !data_ || size_ == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Use min_element instead of reduce with lambda
        auto it = hpx::min_element(hpx::execution::par,
            data_->begin(), data_->end());
        return *it;
    }

    // Distributed max - uses max directly, not lambda
    double max() const {
        if (!is_initialized_ || !data_ || size_ == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Use max_element instead of reduce with lambda
        auto it = hpx::max_element(hpx::execution::par,
            data_->begin(), data_->end());
        return *it;
    }

    // Distributed variance (two-pass algorithm)
    // Note: In single-locality mode, this computes correctly.
    // For multi-locality, a more sophisticated algorithm would be needed.
    double var() const {
        if (!is_initialized_ || !data_ || size_ == 0) return 0.0;

        double m = mean();
        double sum_sq = 0.0;

        // Simple iteration for variance calculation
        for (auto it = data_->begin(); it != data_->end(); ++it) {
            double diff = *it - m;
            sum_sq += diff * diff;
        }

        return sum_sq / static_cast<double>(size_);
    }

    // Distributed standard deviation
    double std() const {
        return std::sqrt(var());
    }

    // Distributed product
    double prod() const {
        if (!is_initialized_ || !data_ || size_ == 0) return 1.0;

        return hpx::reduce(hpx::execution::par,
            data_->begin(), data_->end(), 1.0, std::multiplies<double>());
    }

    // ========================================================================
    // Element-wise Operations (in-place)
    // Use simple iteration since segmented transform requires serializable funcs
    // ========================================================================

    // Add scalar to all elements
    void add_scalar(double value) {
        if (!is_initialized_ || !data_) return;

        for (auto it = data_->begin(); it != data_->end(); ++it) {
            *it = *it + value;
        }
    }

    // Multiply all elements by scalar
    void multiply_scalar(double value) {
        if (!is_initialized_ || !data_) return;

        for (auto it = data_->begin(); it != data_->end(); ++it) {
            *it = *it * value;
        }
    }

private:
    std::size_t size_ = 0;
    bool is_initialized_ = false;
    std::shared_ptr<vector_type> data_;
};

// ============================================================================
// Distributed Utility Functions
// ============================================================================

// Create a partitioned array filled with zeros
inline partitioned_array partitioned_zeros(std::size_t size) {
    return partitioned_array(size, 0.0);
}

// Create a partitioned array filled with ones
inline partitioned_array partitioned_ones(std::size_t size) {
    return partitioned_array(size, 1.0);
}

// Create a partitioned array filled with a value
inline partitioned_array partitioned_full(std::size_t size, double value) {
    return partitioned_array(size, value);
}

// Create a partitioned array with a range of values [0, size)
inline partitioned_array partitioned_arange(std::size_t size) {
    partitioned_array result(size, 0.0);

    // Fill with indices
    std::size_t idx = 0;
    for (auto it = result.vector().begin(); it != result.vector().end(); ++it, ++idx) {
        *it = static_cast<double>(idx);
    }

    return result;
}

// ============================================================================
// Distributed Reduction Functions (free functions)
// ============================================================================

inline double distributed_sum(partitioned_array const& arr) {
    return arr.sum();
}

inline double distributed_mean(partitioned_array const& arr) {
    return arr.mean();
}

inline double distributed_min(partitioned_array const& arr) {
    return arr.min();
}

inline double distributed_max(partitioned_array const& arr) {
    return arr.max();
}

inline double distributed_var(partitioned_array const& arr) {
    return arr.var();
}

inline double distributed_std(partitioned_array const& arr) {
    return arr.std();
}

inline double distributed_prod(partitioned_array const& arr) {
    return arr.prod();
}

}  // namespace hpxpy
