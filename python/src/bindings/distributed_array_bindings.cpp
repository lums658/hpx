// HPXPy - Distributed Array Bindings
//
// SPDX-License-Identifier: BSL-1.0
//
// Python bindings for distributed arrays.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../types/distributed_array.hpp"

namespace py = pybind11;

namespace hpxpy {

// Helper to create distributed array from numpy
template<typename T>
distributed_array<T> from_numpy_distributed(
    py::array_t<T> arr,
    py::object distribution,
    std::size_t /*num_partitions*/)
{
    auto policy = parse_distribution(distribution);

    // Get shape
    std::vector<py::ssize_t> shape(arr.ndim());
    for (py::ssize_t i = 0; i < arr.ndim(); ++i) {
        shape[i] = arr.shape(i);
    }

    // Copy data to vector
    std::vector<T> data(arr.data(), arr.data() + arr.size());

    // Create distributed array
    return distributed_array<T>(data, shape, policy);
}

// Helper to create zeros
template<typename T>
distributed_array<T> zeros_distributed(
    std::vector<py::ssize_t> shape,
    py::object distribution,
    std::size_t num_partitions)
{
    auto policy = parse_distribution(distribution);
    distributed_array<T> arr(shape, policy, num_partitions);
    arr.fill(T{0});
    return arr;
}

// Helper to create ones
template<typename T>
distributed_array<T> ones_distributed(
    std::vector<py::ssize_t> shape,
    py::object distribution,
    std::size_t num_partitions)
{
    auto policy = parse_distribution(distribution);
    distributed_array<T> arr(shape, policy, num_partitions);
    arr.fill(T{1});
    return arr;
}

// Helper to create full
template<typename T>
distributed_array<T> full_distributed(
    std::vector<py::ssize_t> shape,
    T value,
    py::object distribution,
    std::size_t num_partitions)
{
    auto policy = parse_distribution(distribution);
    distributed_array<T> arr(shape, policy, num_partitions);
    arr.fill(value);
    return arr;
}

// Bind a specific distributed array type
template<typename T>
void bind_distributed_array_type(py::module_& m, const char* name) {
    using DA = distributed_array<T>;

    py::class_<DA>(m, name)
        .def(py::init<std::vector<py::ssize_t>, DistributionPolicy, std::size_t>(),
             py::arg("shape"),
             py::arg("policy") = DistributionPolicy::None,
             py::arg("num_partitions") = 0,
             "Create a distributed array with given shape and distribution")

        // Properties
        .def_property_readonly("shape", &DA::shape,
            "Shape of the array")
        .def_property_readonly("size", &DA::size,
            "Total number of elements")
        .def_property_readonly("ndim", &DA::ndim,
            "Number of dimensions")
        .def_property_readonly("policy", &DA::policy,
            "Distribution policy")
        .def_property_readonly("num_partitions", &DA::num_partitions,
            "Number of partitions")
        .def_property_readonly("locality_id", &DA::locality_id,
            "ID of the locality that created this array")

        // Methods
        .def("is_distributed", &DA::is_distributed,
            "Check if array is actually distributed across localities")
        .def("to_numpy", &DA::to_numpy,
            "Convert to numpy array (gathers all data if distributed)")
        .def("fill", &DA::fill,
            py::arg("value"),
            "Fill array with a value")
        .def("get_distribution_info", &DA::get_distribution_info,
            "Get distribution information")

        // Repr
        .def("__repr__", [name](DA const& arr) {
            std::ostringstream oss;
            oss << name << "(shape=[";
            auto const& shape = arr.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << shape[i];
            }
            oss << "], ";

            switch (arr.policy()) {
                case DistributionPolicy::None:
                    oss << "distribution='none'";
                    break;
                case DistributionPolicy::Block:
                    oss << "distribution='block'";
                    break;
                case DistributionPolicy::Cyclic:
                    oss << "distribution='cyclic'";
                    break;
            }
            oss << ", partitions=" << arr.num_partitions();
            if (arr.is_distributed()) {
                oss << ", distributed=True";
            }
            oss << ")";
            return oss.str();
        });
}

void register_distributed_array_bindings(py::module_& m) {
    // Note: DistributionPolicy enum is already registered in array_bindings.cpp
    // in the 'distribution' submodule. Don't re-register it here.

    // Distribution info struct
    py::class_<DistributionInfo>(m, "DistributionInfo",
        "Information about array distribution")
        .def_readonly("policy", &DistributionInfo::policy)
        .def_readonly("num_partitions", &DistributionInfo::num_partitions)
        .def_readonly("chunk_size", &DistributionInfo::chunk_size)
        .def_readonly("locality_id", &DistributionInfo::locality_id)
        .def("is_distributed", &DistributionInfo::is_distributed);

    // Bind distributed array types
    bind_distributed_array_type<double>(m, "DistributedArrayF64");
    bind_distributed_array_type<float>(m, "DistributedArrayF32");
    bind_distributed_array_type<std::int64_t>(m, "DistributedArrayI64");
    bind_distributed_array_type<std::int32_t>(m, "DistributedArrayI32");

    // Factory functions for distributed arrays
    m.def("distributed_zeros",
        [](std::vector<py::ssize_t> shape, py::object distribution, std::size_t num_partitions) {
            return zeros_distributed<double>(shape, distribution, num_partitions);
        },
        py::arg("shape"),
        py::arg("distribution") = py::none(),
        py::arg("num_partitions") = 0,
        "Create a distributed array filled with zeros");

    m.def("distributed_ones",
        [](std::vector<py::ssize_t> shape, py::object distribution, std::size_t num_partitions) {
            return ones_distributed<double>(shape, distribution, num_partitions);
        },
        py::arg("shape"),
        py::arg("distribution") = py::none(),
        py::arg("num_partitions") = 0,
        "Create a distributed array filled with ones");

    m.def("distributed_full",
        [](std::vector<py::ssize_t> shape, double value, py::object distribution, std::size_t num_partitions) {
            return full_distributed<double>(shape, value, distribution, num_partitions);
        },
        py::arg("shape"),
        py::arg("value"),
        py::arg("distribution") = py::none(),
        py::arg("num_partitions") = 0,
        "Create a distributed array filled with a value");

    m.def("distributed_from_numpy",
        [](py::array_t<double> arr, py::object distribution, std::size_t num_partitions) {
            return from_numpy_distributed<double>(arr, distribution, num_partitions);
        },
        py::arg("arr"),
        py::arg("distribution") = py::none(),
        py::arg("num_partitions") = 0,
        "Create a distributed array from a numpy array");

    // ========================================================================
    // Phase 11: Partitioned Array with HPX partitioned_vector
    // ========================================================================

    py::class_<partitioned_array>(m, "PartitionedArray",
        R"doc(
        Distributed array using HPX partitioned_vector.

        This class provides true multi-locality distribution when running
        on multiple HPX localities (nodes). Data is automatically partitioned
        across all available localities.

        In single-locality mode, the array behaves like a local array but
        with the same API, allowing code to be written once and run on
        any number of nodes.

        Examples
        --------
        >>> arr = hpx.partitioned_zeros(1000000)
        >>> print(f"Sum: {arr.sum()}")
        >>> print(f"Distributed: {arr.is_distributed()}")
        )doc")

        // Constructors
        .def(py::init<std::size_t, double>(),
             py::arg("size"),
             py::arg("fill_value") = 0.0,
             "Create a partitioned array with given size and fill value")

        // Properties
        .def_property_readonly("size", &partitioned_array::size,
            "Total number of elements across all localities")
        .def_property_readonly("num_localities", &partitioned_array::num_localities,
            "Number of HPX localities (nodes)")
        .def_property_readonly("locality_id", &partitioned_array::locality_id,
            "ID of the current locality")
        .def_property_readonly("num_partitions", &partitioned_array::num_partitions,
            "Number of partitions in the distributed array")

        // Methods
        .def("is_initialized", &partitioned_array::is_initialized,
            "Check if array has been initialized")
        .def("is_distributed", &partitioned_array::is_distributed,
            "Check if array is distributed across multiple localities")
        .def("to_numpy", &partitioned_array::to_numpy,
            "Convert to numpy array (gathers all data to calling locality)")
        .def("fill", &partitioned_array::fill,
            py::arg("value"),
            "Fill all elements with a value (parallel across localities)")

        // Reduction operations
        .def("sum", &partitioned_array::sum,
            "Compute sum of all elements (distributed reduction)")
        .def("mean", &partitioned_array::mean,
            "Compute mean of all elements (distributed reduction)")
        .def("min", &partitioned_array::min,
            "Find minimum element (distributed reduction)")
        .def("max", &partitioned_array::max,
            "Find maximum element (distributed reduction)")
        .def("var", &partitioned_array::var,
            "Compute variance of all elements (distributed)")
        .def("std", &partitioned_array::std,
            "Compute standard deviation of all elements (distributed)")
        .def("prod", &partitioned_array::prod,
            "Compute product of all elements (distributed reduction)")

        // Element-wise operations
        .def("add_scalar", &partitioned_array::add_scalar,
            py::arg("value"),
            "Add scalar to all elements (parallel across localities)")
        .def("multiply_scalar", &partitioned_array::multiply_scalar,
            py::arg("value"),
            "Multiply all elements by scalar (parallel across localities)")

        // String representation
        .def("__repr__", [](partitioned_array const& arr) {
            std::ostringstream oss;
            oss << "PartitionedArray(size=" << arr.size()
                << ", localities=" << arr.num_localities()
                << ", partitions=" << arr.num_partitions();
            if (arr.is_distributed()) {
                oss << ", distributed=True";
            }
            oss << ")";
            return oss.str();
        });

    // Factory functions for partitioned arrays
    m.def("partitioned_zeros", &partitioned_zeros,
        py::arg("size"),
        R"doc(
        Create a partitioned array filled with zeros.

        The array is distributed across all available HPX localities
        using block distribution.

        Parameters
        ----------
        size : int
            Total number of elements

        Returns
        -------
        PartitionedArray
            A distributed array filled with zeros
        )doc");

    m.def("partitioned_ones", &partitioned_ones,
        py::arg("size"),
        R"doc(
        Create a partitioned array filled with ones.

        The array is distributed across all available HPX localities
        using block distribution.

        Parameters
        ----------
        size : int
            Total number of elements

        Returns
        -------
        PartitionedArray
            A distributed array filled with ones
        )doc");

    m.def("partitioned_full", &partitioned_full,
        py::arg("size"),
        py::arg("value"),
        R"doc(
        Create a partitioned array filled with a specified value.

        Parameters
        ----------
        size : int
            Total number of elements
        value : float
            Value to fill the array with

        Returns
        -------
        PartitionedArray
            A distributed array filled with the specified value
        )doc");

    m.def("partitioned_arange", &partitioned_arange,
        py::arg("size"),
        R"doc(
        Create a partitioned array with values [0, 1, 2, ..., size-1].

        Parameters
        ----------
        size : int
            Total number of elements

        Returns
        -------
        PartitionedArray
            A distributed array with sequential values
        )doc");

    m.def("partitioned_from_numpy", &partitioned_array::from_numpy,
        py::arg("arr"),
        R"doc(
        Create a partitioned array from a numpy array.

        The data is distributed across all available HPX localities.

        Parameters
        ----------
        arr : numpy.ndarray
            Source array (must be float64)

        Returns
        -------
        PartitionedArray
            A distributed array containing the data
        )doc");

    // Distributed reduction functions (free functions)
    m.def("distributed_sum", &distributed_sum,
        py::arg("arr"),
        "Compute sum of partitioned array (distributed reduction)");

    m.def("distributed_mean", &distributed_mean,
        py::arg("arr"),
        "Compute mean of partitioned array (distributed reduction)");

    m.def("distributed_min", &distributed_min,
        py::arg("arr"),
        "Find minimum element of partitioned array (distributed reduction)");

    m.def("distributed_max", &distributed_max,
        py::arg("arr"),
        "Find maximum element of partitioned array (distributed reduction)");

    m.def("distributed_var", &distributed_var,
        py::arg("arr"),
        "Compute variance of partitioned array (distributed)");

    m.def("distributed_std", &distributed_std,
        py::arg("arr"),
        "Compute standard deviation of partitioned array (distributed)");

    m.def("distributed_prod", &distributed_prod,
        py::arg("arr"),
        "Compute product of partitioned array (distributed reduction)");
}

}  // namespace hpxpy
