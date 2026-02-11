// HPXPy - Collective Operations Bindings
//
// SPDX-License-Identifier: BSL-1.0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <hpx/hpx.hpp>
#include <hpx/collectives/all_reduce.hpp>
#include <hpx/collectives/broadcast.hpp>
#include <hpx/collectives/gather.hpp>
#include <hpx/collectives/scatter.hpp>
#include <hpx/collectives/barrier.hpp>

#include "ndarray.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace hpxpy {

// ============================================================================
// Reduction operation types
// ============================================================================

enum class ReduceOp {
    Sum,
    Prod,
    Min,
    Max
};

// ============================================================================
// Serializable element-wise reduction functors for std::vector<double>
// HPX collectives require serializable operations.
// ============================================================================

struct vector_sum_op {
    std::vector<double> operator()(
        std::vector<double> const& a, std::vector<double> const& b) const
    {
        std::vector<double> r(a.size());
        for (std::size_t i = 0; i < a.size(); ++i)
            r[i] = a[i] + b[i];
        return r;
    }

    template <typename Archive>
    void serialize(Archive&, unsigned) {}
};

struct vector_prod_op {
    std::vector<double> operator()(
        std::vector<double> const& a, std::vector<double> const& b) const
    {
        std::vector<double> r(a.size());
        for (std::size_t i = 0; i < a.size(); ++i)
            r[i] = a[i] * b[i];
        return r;
    }

    template <typename Archive>
    void serialize(Archive&, unsigned) {}
};

struct vector_min_op {
    std::vector<double> operator()(
        std::vector<double> const& a, std::vector<double> const& b) const
    {
        std::vector<double> r(a.size());
        for (std::size_t i = 0; i < a.size(); ++i)
            r[i] = a[i] < b[i] ? a[i] : b[i];
        return r;
    }

    template <typename Archive>
    void serialize(Archive&, unsigned) {}
};

struct vector_max_op {
    std::vector<double> operator()(
        std::vector<double> const& a, std::vector<double> const& b) const
    {
        std::vector<double> r(a.size());
        for (std::size_t i = 0; i < a.size(); ++i)
            r[i] = a[i] > b[i] ? a[i] : b[i];
        return r;
    }

    template <typename Archive>
    void serialize(Archive&, unsigned) {}
};

// ============================================================================
// Generation counters — one per collective type.
// All localities must call the same collectives in the same order (SPMD contract)
// so the counters stay synchronized across processes.
// ============================================================================

static std::atomic<std::size_t> all_reduce_gen{0};
static std::atomic<std::size_t> broadcast_gen{0};
static std::atomic<std::size_t> gather_gen{0};
static std::atomic<std::size_t> scatter_gen{0};

// ============================================================================
// Helper: convert ndarray to std::vector<double>
// ============================================================================

static std::vector<double> ndarray_to_vector(ndarray const& arr) {
    py::array_t<double> np_arr = arr.to_numpy().cast<py::array_t<double>>();
    auto buf = np_arr.unchecked<1>();
    std::vector<double> vec(static_cast<std::size_t>(buf.size()));
    for (py::ssize_t i = 0; i < buf.size(); ++i) {
        vec[static_cast<std::size_t>(i)] = buf(i);
    }
    return vec;
}

// ============================================================================
// Helper: convert std::vector<double> to ndarray
// ============================================================================

static ndarray vector_to_ndarray(std::vector<double> const& vec) {
    py::array_t<double> np_arr(static_cast<py::ssize_t>(vec.size()));
    auto buf = np_arr.mutable_unchecked<1>();
    for (std::size_t i = 0; i < vec.size(); ++i) {
        buf(static_cast<py::ssize_t>(i)) = vec[i];
    }
    return ndarray(np_arr, false);
}

// ============================================================================
// Locality introspection
// ============================================================================

std::uint32_t get_num_localities_impl() {
    return hpx::get_num_localities(hpx::launch::sync);
}

std::uint32_t get_locality_id_impl() {
    return hpx::get_locality_id();
}

// ============================================================================
// Barrier synchronization
// ============================================================================

void barrier_impl(const std::string& name) {
    if (get_num_localities_impl() > 1) {
        // Must run on an HPX thread — the main Python thread (after hpx::start)
        // is not an HPX thread, and barrier wait suspends the calling thread
        // using HPX's scheduler, which requires an HPX thread context.
        auto f = hpx::async([&name]() {
            hpx::distributed::barrier b(name);
            b.wait();
        });
        f.get();
    }
}

// ============================================================================
// All-reduce: element-wise reduction across all localities
// ============================================================================

ndarray all_reduce_impl(ndarray const& arr, ReduceOp op) {
    std::uint32_t num_locs = get_num_localities_impl();

    if (num_locs == 1) {
        py::array np_arr = arr.to_numpy();
        return ndarray(np_arr, true);
    }

    // Convert to serializable vector (must be done on calling thread with GIL)
    std::vector<double> local_vec = ndarray_to_vector(arr);

    std::uint32_t this_site = get_locality_id_impl();
    std::size_t gen = ++all_reduce_gen;

    // Run collective on an HPX thread (main thread after hpx::start is not HPX)
    auto f = hpx::async([local_vec = std::move(local_vec), op, num_locs, this_site, gen]() mutable {
        using namespace hpx::collectives;

        switch (op) {
        case ReduceOp::Sum:
            return hpx::collectives::all_reduce(
                "hpxpy_all_reduce_sum", std::move(local_vec), vector_sum_op{},
                num_sites_arg(num_locs), this_site_arg(this_site),
                generation_arg(gen)).get();
        case ReduceOp::Prod:
            return hpx::collectives::all_reduce(
                "hpxpy_all_reduce_prod", std::move(local_vec), vector_prod_op{},
                num_sites_arg(num_locs), this_site_arg(this_site),
                generation_arg(gen)).get();
        case ReduceOp::Min:
            return hpx::collectives::all_reduce(
                "hpxpy_all_reduce_min", std::move(local_vec), vector_min_op{},
                num_sites_arg(num_locs), this_site_arg(this_site),
                generation_arg(gen)).get();
        case ReduceOp::Max:
            return hpx::collectives::all_reduce(
                "hpxpy_all_reduce_max", std::move(local_vec), vector_max_op{},
                num_sites_arg(num_locs), this_site_arg(this_site),
                generation_arg(gen)).get();
        }
        return std::vector<double>{};  // unreachable
    });

    return vector_to_ndarray(f.get());
}

// ============================================================================
// Broadcast: root sends array to all localities
// ============================================================================

ndarray broadcast_impl(ndarray const& arr, std::uint32_t root) {
    std::uint32_t num_locs = get_num_localities_impl();
    std::uint32_t this_site = get_locality_id_impl();

    if (num_locs == 1) {
        py::array np_arr = arr.to_numpy();
        return ndarray(np_arr, true);
    }

    // Convert on calling thread (has GIL)
    std::vector<double> local_vec = ndarray_to_vector(arr);
    std::size_t gen = ++broadcast_gen;

    // Run collective on HPX thread
    auto f = hpx::async([local_vec = std::move(local_vec), root, num_locs, this_site, gen]() mutable {
        using namespace hpx::collectives;

        if (this_site == root) {
            return hpx::collectives::broadcast_to(
                "hpxpy_broadcast", std::move(local_vec),
                num_sites_arg(num_locs), this_site_arg(this_site),
                generation_arg(gen)).get();
        } else {
            return hpx::collectives::broadcast_from<std::vector<double>>(
                "hpxpy_broadcast",
                this_site_arg(this_site), generation_arg(gen),
                root_site_arg(root)).get();
        }
    });

    return vector_to_ndarray(f.get());
}

// ============================================================================
// Gather: all localities send to root
// ============================================================================

py::list gather_impl(ndarray const& arr, std::uint32_t root) {
    std::uint32_t num_locs = get_num_localities_impl();
    std::uint32_t this_site = get_locality_id_impl();

    py::list result;

    if (num_locs == 1) {
        result.append(arr.to_numpy());
        return result;
    }

    // Convert on calling thread (has GIL)
    std::vector<double> local_vec = ndarray_to_vector(arr);
    std::size_t gen = ++gather_gen;

    // Run collective on HPX thread; return gathered data (or empty for non-root)
    auto f = hpx::async([local_vec = std::move(local_vec), root, num_locs, this_site, gen]() mutable
        -> std::vector<std::vector<double>>
    {
        using namespace hpx::collectives;

        if (this_site == root) {
            return hpx::collectives::gather_here(
                "hpxpy_gather", std::move(local_vec),
                num_sites_arg(num_locs), this_site_arg(this_site),
                generation_arg(gen)).get();
        } else {
            hpx::collectives::gather_there(
                "hpxpy_gather", std::move(local_vec),
                this_site_arg(this_site), generation_arg(gen),
                root_site_arg(root)).get();
            return {};  // non-root gets nothing
        }
    });

    // Convert back to Python (needs GIL)
    auto gathered = f.get();
    for (auto const& vec : gathered) {
        py::array_t<double> np_arr(static_cast<py::ssize_t>(vec.size()));
        auto buf = np_arr.mutable_unchecked<1>();
        for (std::size_t i = 0; i < vec.size(); ++i) {
            buf(static_cast<py::ssize_t>(i)) = vec[i];
        }
        result.append(np_arr);
    }

    return result;
}

// ============================================================================
// Scatter: root distributes chunks to all localities
// ============================================================================

ndarray scatter_impl(ndarray const& arr, std::uint32_t root) {
    std::uint32_t num_locs = get_num_localities_impl();
    std::uint32_t this_site = get_locality_id_impl();

    if (num_locs == 1) {
        py::array np_arr = arr.to_numpy();
        return ndarray(np_arr, true);
    }

    // Convert on calling thread (has GIL); only root's data matters
    std::vector<double> full_vec;
    if (this_site == root) {
        full_vec = ndarray_to_vector(arr);
    }
    std::size_t gen = ++scatter_gen;

    // Run collective on HPX thread
    auto f = hpx::async([full_vec = std::move(full_vec), root, num_locs, this_site, gen]() mutable {
        using namespace hpx::collectives;

        if (this_site == root) {
            std::size_t chunk_size = full_vec.size() / num_locs;

            std::vector<std::vector<double>> chunks;
            chunks.reserve(num_locs);
            for (std::uint32_t i = 0; i < num_locs; ++i) {
                std::size_t start = i * chunk_size;
                std::size_t end = (i == num_locs - 1) ? full_vec.size() : start + chunk_size;
                chunks.emplace_back(full_vec.begin() + start, full_vec.begin() + end);
            }

            return hpx::collectives::scatter_to(
                "hpxpy_scatter", std::move(chunks),
                num_sites_arg(num_locs), this_site_arg(this_site),
                generation_arg(gen)).get();
        } else {
            return hpx::collectives::scatter_from<std::vector<double>>(
                "hpxpy_scatter",
                this_site_arg(this_site), generation_arg(gen),
                root_site_arg(root)).get();
        }
    });

    return vector_to_ndarray(f.get());
}

// ============================================================================
// Register collective bindings
// ============================================================================

void register_collective_bindings(py::module_& m) {
    // Reduction operation enum
    py::enum_<ReduceOp>(m, "ReduceOp")
        .value("sum", ReduceOp::Sum)
        .value("prod", ReduceOp::Prod)
        .value("min", ReduceOp::Min)
        .value("max", ReduceOp::Max);

    // Collective operations submodule
    auto collectives = m.def_submodule("collectives",
        "HPX collective operations for distributed computing");

    // Locality info
    collectives.def("get_num_localities", &get_num_localities_impl,
        "Get the number of HPX localities (nodes)");

    collectives.def("get_locality_id", &get_locality_id_impl,
        "Get the ID of the current locality");

    // Barrier
    collectives.def("barrier", &barrier_impl,
        py::arg("name") = "hpxpy_barrier",
        R"doc(
        Synchronize all localities.

        In single-locality mode, this is a no-op.
        In multi-locality mode, all localities wait until everyone reaches the barrier.

        Parameters
        ----------
        name : str, optional
            Name for the barrier (default: "hpxpy_barrier")
        )doc");

    // All-reduce
    collectives.def("all_reduce", &all_reduce_impl,
        py::arg("arr"),
        py::arg("op") = ReduceOp::Sum,
        R"doc(
        Combine values from all localities using a reduction operation.

        Each locality contributes its local array, and all localities receive
        the combined result (element-wise).

        Parameters
        ----------
        arr : ndarray
            Local array to contribute
        op : ReduceOp, optional
            Reduction operation (sum, prod, min, max). Default: sum

        Returns
        -------
        ndarray
            Combined result (same on all localities)
        )doc");

    // Broadcast
    collectives.def("broadcast", &broadcast_impl,
        py::arg("arr"),
        py::arg("root") = 0,
        R"doc(
        Broadcast array from root locality to all localities.

        Parameters
        ----------
        arr : ndarray
            Array to broadcast (only meaningful on root)
        root : int, optional
            Locality ID to broadcast from (default: 0)

        Returns
        -------
        ndarray
            Broadcasted array (same on all localities)
        )doc");

    // Gather
    collectives.def("gather", &gather_impl,
        py::arg("arr"),
        py::arg("root") = 0,
        R"doc(
        Gather arrays from all localities to root.

        Parameters
        ----------
        arr : ndarray
            Local array to contribute
        root : int, optional
            Locality ID to gather to (default: 0)

        Returns
        -------
        list
            List of arrays from all localities (only valid on root;
            non-root localities receive an empty list)
        )doc");

    // Scatter
    collectives.def("scatter", &scatter_impl,
        py::arg("arr"),
        py::arg("root") = 0,
        R"doc(
        Scatter array from root to all localities.

        The array on root is divided evenly among all localities.

        Parameters
        ----------
        arr : ndarray
            Array to scatter (only used on root)
        root : int, optional
            Locality ID to scatter from (default: 0)

        Returns
        -------
        ndarray
            This locality's portion of the scattered array
        )doc");

    // Convenience aliases at module level
    m.attr("all_reduce") = collectives.attr("all_reduce");
    m.attr("broadcast") = collectives.attr("broadcast");
    m.attr("gather") = collectives.attr("gather");
    m.attr("scatter") = collectives.attr("scatter");
    m.attr("barrier") = collectives.attr("barrier");
}

}  // namespace hpxpy
