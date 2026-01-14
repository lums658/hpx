# Phase 2: Operator tests
#
# SPDX-License-Identifier: BSL-1.0

import numpy as np
import pytest

import hpxpy as hpx


class TestArithmeticOperators:
    """Test arithmetic operator overloading."""

    def test_add_arrays(self, hpx_runtime):
        a = hpx.arange(10)
        b = hpx.ones(10)
        result = a + b
        expected = np.arange(10) + 1
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_add_scalar(self, hpx_runtime):
        a = hpx.arange(10)
        result = a + 5
        expected = np.arange(10) + 5
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_radd_scalar(self, hpx_runtime):
        a = hpx.arange(10)
        result = 5 + a
        expected = 5 + np.arange(10)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_sub_arrays(self, hpx_runtime):
        a = hpx.arange(10)
        b = hpx.ones(10) * 2
        result = a - b
        expected = np.arange(10) - 2
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_sub_scalar(self, hpx_runtime):
        a = hpx.arange(10)
        result = a - 3
        expected = np.arange(10) - 3
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_rsub_scalar(self, hpx_runtime):
        a = hpx.arange(10)
        result = 10 - a
        expected = 10 - np.arange(10)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_mul_arrays(self, hpx_runtime):
        a = hpx.arange(10)
        b = hpx.arange(10)
        result = a * b
        expected = np.arange(10) * np.arange(10)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_mul_scalar(self, hpx_runtime):
        a = hpx.arange(10)
        result = a * 3
        expected = np.arange(10) * 3
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_div_arrays(self, hpx_runtime):
        a = hpx.arange(1, 11)  # 1-10 to avoid div by 0
        b = hpx.ones(10) * 2
        result = a / b
        expected = np.arange(1, 11) / 2
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_div_scalar(self, hpx_runtime):
        a = hpx.arange(10)
        result = a / 2
        expected = np.arange(10) / 2
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_pow(self, hpx_runtime):
        a = hpx.arange(5)
        result = a ** 2
        expected = np.arange(5) ** 2
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestUnaryOperators:
    """Test unary operators."""

    def test_neg(self, hpx_runtime):
        a = hpx.arange(10)
        result = -a
        expected = -np.arange(10)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_pos(self, hpx_runtime):
        a = hpx.arange(10)
        result = +a
        expected = +np.arange(10)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_abs(self, hpx_runtime):
        a = hpx.array([-3, -2, -1, 0, 1, 2, 3])
        result = abs(a)
        expected = np.abs([-3, -2, -1, 0, 1, 2, 3])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestComparisonOperators:
    """Test comparison operators."""

    def test_eq(self, hpx_runtime):
        a = hpx.arange(5)
        result = a == 2
        expected = np.arange(5) == 2
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_ne(self, hpx_runtime):
        a = hpx.arange(5)
        result = a != 2
        expected = np.arange(5) != 2
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_lt(self, hpx_runtime):
        a = hpx.arange(5)
        result = a < 3
        expected = np.arange(5) < 3
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_le(self, hpx_runtime):
        a = hpx.arange(5)
        result = a <= 3
        expected = np.arange(5) <= 3
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_gt(self, hpx_runtime):
        a = hpx.arange(5)
        result = a > 2
        expected = np.arange(5) > 2
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_ge(self, hpx_runtime):
        a = hpx.arange(5)
        result = a >= 2
        expected = np.arange(5) >= 2
        np.testing.assert_array_equal(result.to_numpy(), expected)


class TestOperatorChaining:
    """Test chained operations."""

    def test_chain_arithmetic(self, hpx_runtime):
        a = hpx.arange(10)
        result = (a + 1) * 2 - 3
        expected = (np.arange(10) + 1) * 2 - 3
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_expression_evaluation(self, hpx_runtime):
        x = hpx.arange(1, 6)  # [1, 2, 3, 4, 5]
        result = x ** 2 + 2 * x + 1  # (x+1)^2
        expected = np.arange(1, 6) ** 2 + 2 * np.arange(1, 6) + 1
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


# Phase 9: Broadcasting Tests

class TestBroadcastingBasic:
    """Test NumPy-style broadcasting for binary operations."""

    def test_broadcast_1d_to_2d(self, hpx_runtime):
        """Broadcast (4,) to (3, 4)."""
        a_np = np.arange(12, dtype=np.float64).reshape((3, 4))
        b_np = np.array([1, 2, 3, 4], dtype=np.float64)

        a = hpx.from_numpy(a_np)
        b = hpx.from_numpy(b_np)

        result = a + b
        expected = a_np + b_np

        assert result.shape == expected.shape
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_broadcast_column_to_2d(self, hpx_runtime):
        """Broadcast (3, 1) to (3, 4)."""
        a_np = np.arange(12, dtype=np.float64).reshape((3, 4))
        b_np = np.array([[10], [20], [30]], dtype=np.float64)

        a = hpx.from_numpy(a_np)
        b = hpx.from_numpy(b_np)

        result = a + b
        expected = a_np + b_np

        assert result.shape == expected.shape
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_broadcast_row_col(self, hpx_runtime):
        """Broadcast (1, 4) and (3, 1) to (3, 4)."""
        a_np = np.array([[1, 2, 3, 4]], dtype=np.float64)
        b_np = np.array([[10], [20], [30]], dtype=np.float64)

        a = hpx.from_numpy(a_np)
        b = hpx.from_numpy(b_np)

        result = a + b
        expected = a_np + b_np

        assert result.shape == expected.shape
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_broadcast_3d_1d(self, hpx_runtime):
        """Broadcast (4,) to (2, 3, 4)."""
        a_np = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
        b_np = np.array([1, 2, 3, 4], dtype=np.float64)

        a = hpx.from_numpy(a_np)
        b = hpx.from_numpy(b_np)

        result = a * b
        expected = a_np * b_np

        assert result.shape == expected.shape
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_broadcast_same_shape_fast_path(self, hpx_runtime):
        """Same shape arrays use fast path (no broadcasting needed)."""
        a_np = np.arange(12, dtype=np.float64).reshape((3, 4))
        b_np = np.ones((3, 4), dtype=np.float64) * 2

        a = hpx.from_numpy(a_np)
        b = hpx.from_numpy(b_np)

        result = a + b
        expected = a_np + b_np

        assert result.shape == expected.shape
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestBroadcastingAllOperators:
    """Test broadcasting works for all arithmetic operators."""

    def test_broadcast_add(self, hpx_runtime):
        a = hpx.from_numpy(np.arange(12, dtype=np.float64).reshape((3, 4)))
        b = hpx.from_numpy(np.array([1, 2, 3, 4], dtype=np.float64))
        result = a + b
        expected = np.arange(12).reshape((3, 4)) + np.array([1, 2, 3, 4])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_broadcast_sub(self, hpx_runtime):
        a = hpx.from_numpy(np.arange(12, dtype=np.float64).reshape((3, 4)))
        b = hpx.from_numpy(np.array([1, 2, 3, 4], dtype=np.float64))
        result = a - b
        expected = np.arange(12).reshape((3, 4)) - np.array([1, 2, 3, 4])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_broadcast_mul(self, hpx_runtime):
        a = hpx.from_numpy(np.arange(12, dtype=np.float64).reshape((3, 4)))
        b = hpx.from_numpy(np.array([1, 2, 3, 4], dtype=np.float64))
        result = a * b
        expected = np.arange(12).reshape((3, 4)) * np.array([1, 2, 3, 4])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_broadcast_div(self, hpx_runtime):
        a = hpx.from_numpy(np.arange(1, 13, dtype=np.float64).reshape((3, 4)))
        b = hpx.from_numpy(np.array([1, 2, 3, 4], dtype=np.float64))
        result = a / b
        expected = np.arange(1, 13).reshape((3, 4)) / np.array([1, 2, 3, 4])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestBroadcastingComparison:
    """Test broadcasting for comparison operators."""

    def test_broadcast_eq(self, hpx_runtime):
        a_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float64)
        b_np = np.array([1, 2, 3], dtype=np.float64)

        a = hpx.from_numpy(a_np)
        b = hpx.from_numpy(b_np)

        result = a == b
        expected = a_np == b_np

        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_broadcast_lt(self, hpx_runtime):
        a_np = np.arange(12, dtype=np.float64).reshape((3, 4))
        b_np = np.array([5, 5, 5, 5], dtype=np.float64)

        a = hpx.from_numpy(a_np)
        b = hpx.from_numpy(b_np)

        result = a < b
        expected = a_np < b_np

        np.testing.assert_array_equal(result.to_numpy(), expected)


class TestBroadcastingErrors:
    """Test broadcasting error cases."""

    def test_incompatible_shapes_raises(self, hpx_runtime):
        """Incompatible shapes should raise an error."""
        a = hpx.from_numpy(np.arange(12, dtype=np.float64).reshape((3, 4)))
        b = hpx.from_numpy(np.arange(5, dtype=np.float64))  # (5,) not compatible with (3, 4)

        with pytest.raises(RuntimeError, match="Cannot broadcast"):
            _ = a + b

    def test_incompatible_inner_dims(self, hpx_runtime):
        """Inner dimensions must match."""
        a = hpx.from_numpy(np.arange(12, dtype=np.float64).reshape((3, 4)))
        b = hpx.from_numpy(np.arange(3, dtype=np.float64).reshape((3, 1)))

        # This should work (3,4) + (3,1) -> (3,4)
        result = a + b
        expected = np.arange(12).reshape((3, 4)) + np.arange(3).reshape((3, 1))
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestBroadcastingNumpyComparison:
    """Verify broadcasting matches NumPy exactly."""

    @pytest.mark.parametrize("a_shape,b_shape", [
        ((3, 4), (4,)),
        ((3, 4), (1, 4)),
        ((3, 4), (3, 1)),
        ((3, 1), (1, 4)),
        ((2, 3, 4), (4,)),
        ((2, 3, 4), (3, 4)),
        ((2, 3, 4), (1, 3, 4)),
        ((5,), (5,)),
        ((1, 5), (5, 1)),
    ])
    def test_broadcast_shapes_match_numpy(self, hpx_runtime, a_shape, b_shape):
        """Verify HPXPy broadcasting produces same result as NumPy."""
        a_np = np.arange(np.prod(a_shape), dtype=np.float64).reshape(a_shape)
        b_np = np.arange(np.prod(b_shape), dtype=np.float64).reshape(b_shape) + 1

        a = hpx.from_numpy(a_np)
        b = hpx.from_numpy(b_np)

        # Test add
        result = a + b
        expected = a_np + b_np
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

        # Test multiply
        result = a * b
        expected = a_np * b_np
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestZeroDimensionalArrays:
    """
    Test 0-dimensional (scalar) array handling.

    Regression tests for the bug where 0-d arrays had size=0 instead of size=1,
    causing segfaults during broadcasting operations.
    """

    def test_0d_array_creation(self, hpx_runtime):
        """0-d array from NumPy scalar should have size=1."""
        scalar_np = np.array(42.0)
        assert scalar_np.shape == ()
        assert scalar_np.size == 1

        scalar_hpx = hpx.from_numpy(scalar_np)
        assert scalar_hpx.shape == ()
        assert scalar_hpx.size == 1  # Regression test: was 0, now 1

    def test_0d_to_numpy_roundtrip(self, hpx_runtime):
        """0-d array should round-trip through to_numpy correctly."""
        scalar_np = np.array(3.14159)
        scalar_hpx = hpx.from_numpy(scalar_np)
        result = scalar_hpx.to_numpy()

        assert result.shape == ()
        np.testing.assert_almost_equal(result, 3.14159)

    def test_0d_add_1d(self, hpx_runtime):
        """Broadcast 0-d array (scalar) + 1-d array."""
        scalar_np = np.array(10.0)
        arr_np = np.arange(5, dtype=np.float64)

        scalar = hpx.from_numpy(scalar_np)
        arr = hpx.from_numpy(arr_np)

        result = scalar + arr
        expected = scalar_np + arr_np

        assert result.shape == expected.shape
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_1d_add_0d(self, hpx_runtime):
        """Broadcast 1-d array + 0-d array (scalar)."""
        scalar_np = np.array(10.0)
        arr_np = np.arange(5, dtype=np.float64)

        scalar = hpx.from_numpy(scalar_np)
        arr = hpx.from_numpy(arr_np)

        result = arr + scalar
        expected = arr_np + scalar_np

        assert result.shape == expected.shape
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_0d_mul_2d(self, hpx_runtime):
        """Broadcast 0-d array (scalar) * 2-d array."""
        scalar_np = np.array(2.0)
        arr_np = np.arange(12, dtype=np.float64).reshape((3, 4))

        scalar = hpx.from_numpy(scalar_np)
        arr = hpx.from_numpy(arr_np)

        result = scalar * arr
        expected = scalar_np * arr_np

        assert result.shape == expected.shape
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_0d_sub_1d(self, hpx_runtime):
        """Broadcast 0-d - 1-d (order matters for subtraction)."""
        scalar_np = np.array(100.0)
        arr_np = np.arange(5, dtype=np.float64)

        scalar = hpx.from_numpy(scalar_np)
        arr = hpx.from_numpy(arr_np)

        # scalar - arr
        result = scalar - arr
        expected = scalar_np - arr_np
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

        # arr - scalar
        result = arr - scalar
        expected = arr_np - scalar_np
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_0d_div_1d(self, hpx_runtime):
        """Broadcast 0-d / 1-d (order matters for division)."""
        scalar_np = np.array(100.0)
        arr_np = np.arange(1, 6, dtype=np.float64)

        scalar = hpx.from_numpy(scalar_np)
        arr = hpx.from_numpy(arr_np)

        # scalar / arr
        result = scalar / arr
        expected = scalar_np / arr_np
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

        # arr / scalar
        result = arr / scalar
        expected = arr_np / scalar_np
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_0d_comparison_broadcast(self, hpx_runtime):
        """Broadcast 0-d array in comparison operations."""
        threshold_np = np.array(5.0)
        arr_np = np.arange(10, dtype=np.float64)

        threshold = hpx.from_numpy(threshold_np)
        arr = hpx.from_numpy(arr_np)

        result = arr > threshold
        expected = arr_np > threshold_np

        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_black_scholes_pattern(self, hpx_runtime):
        """
        Regression test for Black-Scholes tutorial pattern.

        The bug manifested when math functions returned 0-d arrays
        that were then used in operations with 1-d arrays.
        """
        # This pattern caused segfaults before the fix
        sqrt_2pi_np = np.sqrt(2 * np.pi)  # Returns 0-d array
        x_np = np.linspace(-3, 3, 100)

        sqrt_2pi = hpx.from_numpy(np.array(sqrt_2pi_np))
        x = hpx.from_numpy(x_np)

        # This operation would segfault with size=0 for 0-d arrays
        result = x / sqrt_2pi
        expected = x_np / sqrt_2pi_np

        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_chained_0d_operations(self, hpx_runtime):
        """Multiple operations with 0-d arrays in chain."""
        a_np = np.array(2.0)
        b_np = np.array(3.0)
        c_np = np.arange(5, dtype=np.float64)

        a = hpx.from_numpy(a_np)
        b = hpx.from_numpy(b_np)
        c = hpx.from_numpy(c_np)

        # (a + b) * c
        result = (a + b) * c
        expected = (a_np + b_np) * c_np

        np.testing.assert_array_almost_equal(result.to_numpy(), expected)
