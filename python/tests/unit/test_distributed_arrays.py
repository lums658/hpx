# HPXPy Distributed Array Tests
#
# SPDX-License-Identifier: BSL-1.0

"""Unit tests for HPXPy distributed arrays."""

import numpy as np
import pytest


class TestDistributedArrayCreation:
    """Test distributed array creation functions."""

    def test_distributed_zeros(self, hpx_runtime):
        """distributed_zeros should create zero-filled array."""
        arr = hpx_runtime.distributed_zeros([100])
        assert arr.shape == [100]
        assert arr.size == 100
        np.testing.assert_array_equal(arr.to_numpy(), np.zeros(100))

    def test_distributed_zeros_with_shape_tuple(self, hpx_runtime):
        """distributed_zeros should work with tuple shape."""
        arr = hpx_runtime.distributed_zeros((50, 2))
        assert arr.shape == [50, 2]
        assert arr.size == 100

    def test_distributed_ones(self, hpx_runtime):
        """distributed_ones should create one-filled array."""
        arr = hpx_runtime.distributed_ones([100])
        np.testing.assert_array_equal(arr.to_numpy(), np.ones(100))

    def test_distributed_full(self, hpx_runtime):
        """distributed_full should create array with specified value."""
        arr = hpx_runtime.distributed_full([50], 3.14)
        np.testing.assert_array_almost_equal(arr.to_numpy(), np.full(50, 3.14))

    def test_distributed_from_numpy(self, hpx_runtime):
        """distributed_from_numpy should copy numpy array data."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr = hpx_runtime.distributed_from_numpy(np_arr)
        np.testing.assert_array_equal(arr.to_numpy(), np_arr)


class TestDistributionPolicy:
    """Test distribution policy handling."""

    def test_default_policy_is_none(self, hpx_runtime):
        """Default distribution should be none."""
        arr = hpx_runtime.distributed_zeros([100])
        assert arr.policy == hpx_runtime.DistributionPolicy.none

    def test_block_distribution(self, hpx_runtime):
        """Block distribution should be recognized."""
        arr = hpx_runtime.distributed_zeros([100], distribution='block')
        assert arr.policy == hpx_runtime.DistributionPolicy.block

    def test_cyclic_distribution(self, hpx_runtime):
        """Cyclic distribution should be recognized."""
        arr = hpx_runtime.distributed_zeros([100], distribution='cyclic')
        assert arr.policy == hpx_runtime.DistributionPolicy.cyclic

    def test_none_distribution_string(self, hpx_runtime):
        """String 'none' should work as distribution."""
        arr = hpx_runtime.distributed_zeros([100], distribution='none')
        assert arr.policy == hpx_runtime.DistributionPolicy.none


class TestDistributedArrayProperties:
    """Test distributed array properties."""

    def test_shape_property(self, hpx_runtime):
        """shape property should return correct shape."""
        arr = hpx_runtime.distributed_zeros([10, 20])
        assert arr.shape == [10, 20]

    def test_size_property(self, hpx_runtime):
        """size property should return total elements."""
        arr = hpx_runtime.distributed_zeros([10, 20])
        assert arr.size == 200

    def test_ndim_property(self, hpx_runtime):
        """ndim property should return number of dimensions."""
        arr = hpx_runtime.distributed_zeros([10, 20, 30])
        assert arr.ndim == 3

    def test_num_partitions_property(self, hpx_runtime):
        """num_partitions should be at least 1."""
        arr = hpx_runtime.distributed_zeros([100])
        assert arr.num_partitions >= 1

    def test_locality_id_property(self, hpx_runtime):
        """locality_id should be valid."""
        arr = hpx_runtime.distributed_zeros([100])
        assert arr.locality_id >= 0


class TestDistributedArrayMethods:
    """Test distributed array methods."""

    def test_fill(self, hpx_runtime):
        """fill should set all elements to value."""
        arr = hpx_runtime.distributed_zeros([100])
        arr.fill(42.0)
        np.testing.assert_array_equal(arr.to_numpy(), np.full(100, 42.0))

    def test_to_numpy(self, hpx_runtime):
        """to_numpy should return correct numpy array."""
        arr = hpx_runtime.distributed_ones([50])
        np_arr = arr.to_numpy()
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.shape == (50,)
        np.testing.assert_array_equal(np_arr, np.ones(50))

    def test_is_distributed_single_locality(self, hpx_runtime):
        """is_distributed should be False on single locality."""
        arr = hpx_runtime.distributed_zeros([100], distribution='block')
        # In single-locality mode, arrays are not actually distributed
        assert arr.is_distributed() is False

    def test_get_distribution_info(self, hpx_runtime):
        """get_distribution_info should return DistributionInfo."""
        arr = hpx_runtime.distributed_zeros([100], distribution='block')
        info = arr.get_distribution_info()
        assert info.policy == hpx_runtime.DistributionPolicy.block
        assert info.num_partitions >= 1


class TestDistributedArrayRepr:
    """Test distributed array string representation."""

    def test_repr_contains_shape(self, hpx_runtime):
        """repr should contain shape information."""
        arr = hpx_runtime.distributed_zeros([100])
        repr_str = repr(arr)
        assert '100' in repr_str

    def test_repr_contains_distribution(self, hpx_runtime):
        """repr should contain distribution information."""
        arr = hpx_runtime.distributed_zeros([100], distribution='block')
        repr_str = repr(arr)
        assert 'block' in repr_str

    def test_repr_contains_partitions(self, hpx_runtime):
        """repr should contain partition information."""
        arr = hpx_runtime.distributed_zeros([100])
        repr_str = repr(arr)
        assert 'partitions' in repr_str


# ============================================================================
# Phase 11: PartitionedArray Tests (HPX partitioned_vector)
# ============================================================================

class TestPartitionedArrayCreation:
    """Test partitioned array creation using HPX partitioned_vector."""

    def test_partitioned_zeros(self, hpx_runtime):
        """partitioned_zeros should create zero-filled array."""
        arr = hpx_runtime.partitioned_zeros(1000)
        assert arr.size == 1000
        np.testing.assert_array_equal(arr.to_numpy(), np.zeros(1000))

    def test_partitioned_ones(self, hpx_runtime):
        """partitioned_ones should create one-filled array."""
        arr = hpx_runtime.partitioned_ones(1000)
        np.testing.assert_array_equal(arr.to_numpy(), np.ones(1000))

    def test_partitioned_full(self, hpx_runtime):
        """partitioned_full should create array with specified value."""
        arr = hpx_runtime.partitioned_full(500, 3.14)
        np.testing.assert_array_almost_equal(arr.to_numpy(), np.full(500, 3.14))

    def test_partitioned_arange(self, hpx_runtime):
        """partitioned_arange should create sequential values."""
        arr = hpx_runtime.partitioned_arange(100)
        np.testing.assert_array_almost_equal(arr.to_numpy(), np.arange(100, dtype=np.float64))

    def test_partitioned_from_numpy(self, hpx_runtime):
        """partitioned_from_numpy should copy numpy array data."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr = hpx_runtime.partitioned_from_numpy(np_arr)
        np.testing.assert_array_equal(arr.to_numpy(), np_arr)


class TestPartitionedArrayProperties:
    """Test partitioned array properties."""

    def test_size_property(self, hpx_runtime):
        """size property should return total elements."""
        arr = hpx_runtime.partitioned_zeros(1000)
        assert arr.size == 1000

    def test_num_localities(self, hpx_runtime):
        """num_localities should be at least 1."""
        arr = hpx_runtime.partitioned_zeros(100)
        assert arr.num_localities >= 1

    def test_locality_id(self, hpx_runtime):
        """locality_id should be valid (0 in single-locality)."""
        arr = hpx_runtime.partitioned_zeros(100)
        assert arr.locality_id >= 0

    def test_num_partitions(self, hpx_runtime):
        """num_partitions should be at least 1."""
        arr = hpx_runtime.partitioned_zeros(100)
        assert arr.num_partitions >= 1

    def test_is_initialized(self, hpx_runtime):
        """is_initialized should be True after creation."""
        arr = hpx_runtime.partitioned_zeros(100)
        assert arr.is_initialized() is True

    def test_is_distributed_single_locality(self, hpx_runtime):
        """is_distributed should be False on single locality."""
        arr = hpx_runtime.partitioned_zeros(100)
        # In single-locality mode, not actually distributed
        assert arr.is_distributed() is False


class TestPartitionedArrayReductions:
    """Test distributed reduction operations on partitioned arrays."""

    def test_sum(self, hpx_runtime):
        """sum should compute correct total."""
        arr = hpx_runtime.partitioned_ones(1000)
        assert arr.sum() == 1000.0

    def test_sum_with_values(self, hpx_runtime):
        """sum should compute correct total with arange."""
        arr = hpx_runtime.partitioned_arange(101)  # 0 + 1 + ... + 100 = 5050
        assert abs(arr.sum() - 5050.0) < 1e-10

    def test_mean(self, hpx_runtime):
        """mean should compute correct average."""
        arr = hpx_runtime.partitioned_arange(101)  # mean of 0..100 is 50
        assert abs(arr.mean() - 50.0) < 1e-10

    def test_min(self, hpx_runtime):
        """min should find minimum element."""
        arr = hpx_runtime.partitioned_arange(100)
        assert arr.min() == 0.0

    def test_max(self, hpx_runtime):
        """max should find maximum element."""
        arr = hpx_runtime.partitioned_arange(100)
        assert arr.max() == 99.0

    def test_var(self, hpx_runtime):
        """var should compute variance."""
        # Variance of [1, 1, 1, ...] is 0
        arr = hpx_runtime.partitioned_ones(100)
        assert abs(arr.var()) < 1e-10

    def test_std(self, hpx_runtime):
        """std should compute standard deviation."""
        arr = hpx_runtime.partitioned_ones(100)
        assert abs(arr.std()) < 1e-10

    def test_prod(self, hpx_runtime):
        """prod should compute product."""
        # Product of all 1s is 1
        arr = hpx_runtime.partitioned_ones(100)
        assert arr.prod() == 1.0


class TestPartitionedArrayMethods:
    """Test partitioned array methods."""

    def test_fill(self, hpx_runtime):
        """fill should set all elements to value."""
        arr = hpx_runtime.partitioned_zeros(100)
        arr.fill(42.0)
        np.testing.assert_array_equal(arr.to_numpy(), np.full(100, 42.0))

    def test_add_scalar(self, hpx_runtime):
        """add_scalar should add value to all elements."""
        arr = hpx_runtime.partitioned_arange(10)
        arr.add_scalar(10.0)
        expected = np.arange(10, dtype=np.float64) + 10.0
        np.testing.assert_array_almost_equal(arr.to_numpy(), expected)

    def test_multiply_scalar(self, hpx_runtime):
        """multiply_scalar should multiply all elements."""
        arr = hpx_runtime.partitioned_arange(10)
        arr.multiply_scalar(2.0)
        expected = np.arange(10, dtype=np.float64) * 2.0
        np.testing.assert_array_almost_equal(arr.to_numpy(), expected)

    def test_to_numpy(self, hpx_runtime):
        """to_numpy should return correct numpy array."""
        arr = hpx_runtime.partitioned_full(50, 7.5)
        np_arr = arr.to_numpy()
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.shape == (50,)
        np.testing.assert_array_almost_equal(np_arr, np.full(50, 7.5))


class TestPartitionedArrayRepr:
    """Test partitioned array string representation."""

    def test_repr_contains_size(self, hpx_runtime):
        """repr should contain size information."""
        arr = hpx_runtime.partitioned_zeros(1000)
        repr_str = repr(arr)
        assert '1000' in repr_str

    def test_repr_contains_localities(self, hpx_runtime):
        """repr should contain localities information."""
        arr = hpx_runtime.partitioned_zeros(100)
        repr_str = repr(arr)
        assert 'localities' in repr_str

    def test_repr_contains_partitions(self, hpx_runtime):
        """repr should contain partitions information."""
        arr = hpx_runtime.partitioned_zeros(100)
        repr_str = repr(arr)
        assert 'partitions' in repr_str


class TestPartitionedArrayNumpyComparison:
    """Verify partitioned array reductions match NumPy."""

    def test_sum_matches_numpy(self, hpx_runtime):
        """Sum should match NumPy."""
        np_arr = np.random.rand(1000)
        arr = hpx_runtime.partitioned_from_numpy(np_arr)
        assert abs(arr.sum() - np_arr.sum()) < 1e-10

    def test_mean_matches_numpy(self, hpx_runtime):
        """Mean should match NumPy."""
        np_arr = np.random.rand(1000)
        arr = hpx_runtime.partitioned_from_numpy(np_arr)
        assert abs(arr.mean() - np_arr.mean()) < 1e-10

    def test_min_matches_numpy(self, hpx_runtime):
        """Min should match NumPy."""
        np_arr = np.random.rand(1000)
        arr = hpx_runtime.partitioned_from_numpy(np_arr)
        assert abs(arr.min() - np_arr.min()) < 1e-10

    def test_max_matches_numpy(self, hpx_runtime):
        """Max should match NumPy."""
        np_arr = np.random.rand(1000)
        arr = hpx_runtime.partitioned_from_numpy(np_arr)
        assert abs(arr.max() - np_arr.max()) < 1e-10

    def test_var_matches_numpy(self, hpx_runtime):
        """Variance should match NumPy (population variance)."""
        np_arr = np.random.rand(1000)
        arr = hpx_runtime.partitioned_from_numpy(np_arr)
        # Note: we use population variance (ddof=0)
        expected_var = np.var(np_arr, ddof=0)
        assert abs(arr.var() - expected_var) < 1e-10

    def test_std_matches_numpy(self, hpx_runtime):
        """Standard deviation should match NumPy."""
        np_arr = np.random.rand(1000)
        arr = hpx_runtime.partitioned_from_numpy(np_arr)
        expected_std = np.std(np_arr, ddof=0)
        assert abs(arr.std() - expected_std) < 1e-10
