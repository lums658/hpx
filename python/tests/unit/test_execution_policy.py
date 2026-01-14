# HPXPy Execution Policy Tests
#
# SPDX-License-Identifier: BSL-1.0
#
# Tests for the execution policy parameter added to reduction functions.

import pytest
import numpy as np


class TestExecutionPolicySum:
    """Test sum() with different execution policies."""

    def test_sum_seq(self, hpx_runtime):
        """Test sum with sequential execution policy."""
        import hpxpy as hpx
        arr = hpx.arange(1000)
        result = hpx.sum(arr, policy="seq")
        assert result == 499500

    def test_sum_par(self, hpx_runtime):
        """Test sum with parallel execution policy."""
        import hpxpy as hpx
        arr = hpx.arange(1000)
        result = hpx.sum(arr, policy="par")
        assert result == 499500

    def test_sum_par_unseq(self, hpx_runtime):
        """Test sum with parallel unsequenced execution policy."""
        import hpxpy as hpx
        arr = hpx.arange(1000)
        result = hpx.sum(arr, policy="par_unseq")
        assert result == 499500

    def test_sum_default_is_seq(self, hpx_runtime):
        """Test that default policy is sequential."""
        import hpxpy as hpx
        arr = hpx.arange(1000)
        # Default should work the same as explicit "seq"
        result_default = hpx.sum(arr)
        result_seq = hpx.sum(arr, policy="seq")
        assert result_default == result_seq

    def test_sum_invalid_policy(self, hpx_runtime):
        """Test that invalid policy raises an error."""
        import hpxpy as hpx
        arr = hpx.arange(1000)
        with pytest.raises(Exception):  # Could be ValueError or RuntimeError
            hpx.sum(arr, policy="invalid")

    def test_sum_large_array_par(self, hpx_runtime):
        """Test parallel sum on a larger array."""
        import hpxpy as hpx
        arr = hpx.arange(100000)
        result_seq = hpx.sum(arr, policy="seq")
        result_par = hpx.sum(arr, policy="par")
        # Results should be the same (within floating point tolerance for large arrays)
        assert result_seq == result_par


class TestExecutionPolicyProd:
    """Test prod() with different execution policies."""

    def test_prod_seq(self, hpx_runtime):
        """Test product with sequential execution policy."""
        import hpxpy as hpx
        arr = hpx.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = hpx.prod(arr, policy="seq")
        assert result == 120.0

    def test_prod_par(self, hpx_runtime):
        """Test product with parallel execution policy."""
        import hpxpy as hpx
        arr = hpx.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = hpx.prod(arr, policy="par")
        assert result == 120.0


class TestExecutionPolicyMinMax:
    """Test min() and max() with different execution policies."""

    def test_min_seq(self, hpx_runtime):
        """Test min with sequential execution policy."""
        import hpxpy as hpx
        arr = hpx.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        result = hpx.min(arr, policy="seq")
        assert result == 1.0

    def test_min_par(self, hpx_runtime):
        """Test min with parallel execution policy."""
        import hpxpy as hpx
        arr = hpx.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        result = hpx.min(arr, policy="par")
        assert result == 1.0

    def test_max_seq(self, hpx_runtime):
        """Test max with sequential execution policy."""
        import hpxpy as hpx
        arr = hpx.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        result = hpx.max(arr, policy="seq")
        assert result == 5.0

    def test_max_par(self, hpx_runtime):
        """Test max with parallel execution policy."""
        import hpxpy as hpx
        arr = hpx.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
        result = hpx.max(arr, policy="par")
        assert result == 5.0


class TestExecutionPolicySort:
    """Test sort() with different execution policies."""

    def test_sort_seq(self, hpx_runtime):
        """Test sort with sequential execution policy."""
        import hpxpy as hpx
        arr = hpx.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]))
        result = hpx.sort(arr, policy="seq")
        expected = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_sort_par(self, hpx_runtime):
        """Test sort with parallel execution policy."""
        import hpxpy as hpx
        arr = hpx.from_numpy(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]))
        result = hpx.sort(arr, policy="par")
        expected = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0])
        np.testing.assert_array_equal(result.to_numpy(), expected)


class TestExecutionPolicyCount:
    """Test count() with different execution policies."""

    def test_count_seq(self, hpx_runtime):
        """Test count with sequential execution policy."""
        import hpxpy as hpx
        arr = hpx.from_numpy(np.array([1.0, 2.0, 1.0, 3.0, 1.0]))
        result = hpx.count(arr, 1.0, policy="seq")
        assert result == 3

    def test_count_par(self, hpx_runtime):
        """Test count with parallel execution policy."""
        import hpxpy as hpx
        arr = hpx.from_numpy(np.array([1.0, 2.0, 1.0, 3.0, 1.0]))
        result = hpx.count(arr, 1.0, policy="par")
        assert result == 3


class TestExecutionPolicyMean:
    """Test mean() with different execution policies."""

    def test_mean_seq(self, hpx_runtime):
        """Test mean with sequential execution policy."""
        import hpxpy as hpx
        arr = hpx.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = hpx.mean(arr, policy="seq")
        assert result == 3.0

    def test_mean_par(self, hpx_runtime):
        """Test mean with parallel execution policy."""
        import hpxpy as hpx
        arr = hpx.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = hpx.mean(arr, policy="par")
        assert result == 3.0


class TestExecutionPolicyConsistency:
    """Test that all policies produce consistent results."""

    def test_all_policies_consistent_sum(self, hpx_runtime):
        """All policies should give same result for sum (for exact integer arrays)."""
        import hpxpy as hpx
        arr = hpx.arange(10000)
        result_seq = hpx.sum(arr, policy="seq")
        result_par = hpx.sum(arr, policy="par")
        result_par_unseq = hpx.sum(arr, policy="par_unseq")

        assert result_seq == result_par
        assert result_seq == result_par_unseq

    def test_all_policies_consistent_minmax(self, hpx_runtime):
        """All policies should give same result for min/max."""
        import hpxpy as hpx
        np.random.seed(42)
        arr = hpx.from_numpy(np.random.randn(10000))

        min_seq = hpx.min(arr, policy="seq")
        min_par = hpx.min(arr, policy="par")
        max_seq = hpx.max(arr, policy="seq")
        max_par = hpx.max(arr, policy="par")

        assert min_seq == min_par
        assert max_seq == max_par
