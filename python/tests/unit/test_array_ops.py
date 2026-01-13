# HPXPy Array Operations Tests
#
# SPDX-License-Identifier: BSL-1.0

"""Unit tests for HPXPy array operations (slicing, indexing, etc.)."""

import numpy as np
import pytest


class TestArraySlicing:
    """Test array slicing operations."""

    def test_slice_basic(self, hpx_runtime):
        """Basic slice: arr[2:5]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[2:5]

        assert sliced.shape == (3,)
        np.testing.assert_array_equal(sliced.to_numpy(), [2, 3, 4])

    def test_slice_from_start(self, hpx_runtime):
        """Slice from start: arr[:5]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[:5]

        assert sliced.shape == (5,)
        np.testing.assert_array_equal(sliced.to_numpy(), [0, 1, 2, 3, 4])

    def test_slice_to_end(self, hpx_runtime):
        """Slice to end: arr[5:]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[5:]

        assert sliced.shape == (5,)
        np.testing.assert_array_equal(sliced.to_numpy(), [5, 6, 7, 8, 9])

    def test_slice_full(self, hpx_runtime):
        """Full slice: arr[:]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[:]

        assert sliced.shape == (10,)
        np.testing.assert_array_equal(sliced.to_numpy(), np.arange(10))

    def test_slice_step(self, hpx_runtime):
        """Step slice: arr[::2]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[::2]

        assert sliced.shape == (5,)
        np.testing.assert_array_equal(sliced.to_numpy(), [0, 2, 4, 6, 8])

    def test_slice_step_with_start(self, hpx_runtime):
        """Step slice with start: arr[1::2]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[1::2]

        assert sliced.shape == (5,)
        np.testing.assert_array_equal(sliced.to_numpy(), [1, 3, 5, 7, 9])

    def test_slice_step_with_range(self, hpx_runtime):
        """Step slice with range: arr[2:8:2]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[2:8:2]

        assert sliced.shape == (3,)
        np.testing.assert_array_equal(sliced.to_numpy(), [2, 4, 6])

    def test_slice_negative_start(self, hpx_runtime):
        """Negative start: arr[-3:]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[-3:]

        assert sliced.shape == (3,)
        np.testing.assert_array_equal(sliced.to_numpy(), [7, 8, 9])

    def test_slice_negative_stop(self, hpx_runtime):
        """Negative stop: arr[:-3]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[:-3]

        assert sliced.shape == (7,)
        np.testing.assert_array_equal(sliced.to_numpy(), [0, 1, 2, 3, 4, 5, 6])

    def test_slice_negative_both(self, hpx_runtime):
        """Negative start and stop: arr[-5:-2]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[-5:-2]

        assert sliced.shape == (3,)
        np.testing.assert_array_equal(sliced.to_numpy(), [5, 6, 7])

    def test_slice_empty_result(self, hpx_runtime):
        """Slice that results in empty array: arr[5:5]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[5:5]

        assert sliced.shape == (0,)
        assert sliced.size == 0

    def test_slice_single_element(self, hpx_runtime):
        """Slice with single element: arr[5:6]."""
        arr = hpx_runtime.arange(10)
        sliced = arr[5:6]

        assert sliced.shape == (1,)
        np.testing.assert_array_equal(sliced.to_numpy(), [5])


class TestSliceDtypes:
    """Test slicing with different data types."""

    def test_slice_float32(self, hpx_runtime):
        """Slice float32 array."""
        np_arr = np.arange(10, dtype=np.float32)
        arr = hpx_runtime.from_numpy(np_arr)
        sliced = arr[2:5]

        assert sliced.dtype == np.float32
        np.testing.assert_array_equal(sliced.to_numpy(), np_arr[2:5])

    def test_slice_int64(self, hpx_runtime):
        """Slice int64 array."""
        np_arr = np.arange(10, dtype=np.int64)
        arr = hpx_runtime.from_numpy(np_arr)
        sliced = arr[::2]

        assert sliced.dtype == np.int64
        np.testing.assert_array_equal(sliced.to_numpy(), np_arr[::2])

    def test_slice_int32(self, hpx_runtime):
        """Slice int32 array."""
        np_arr = np.arange(10, dtype=np.int32)
        arr = hpx_runtime.from_numpy(np_arr)
        sliced = arr[-3:]

        assert sliced.dtype == np.int32
        np.testing.assert_array_equal(sliced.to_numpy(), np_arr[-3:])


class TestSliceChaining:
    """Test chaining of slice operations."""

    def test_double_slice(self, hpx_runtime):
        """Apply two slices: arr[2:8][1:4]."""
        arr = hpx_runtime.arange(10)
        first = arr[2:8]  # [2, 3, 4, 5, 6, 7]
        second = first[1:4]  # [3, 4, 5]

        assert second.shape == (3,)
        np.testing.assert_array_equal(second.to_numpy(), [3, 4, 5])

    def test_slice_then_step(self, hpx_runtime):
        """Apply slice then step: arr[1:9][::2]."""
        arr = hpx_runtime.arange(10)
        first = arr[1:9]  # [1, 2, 3, 4, 5, 6, 7, 8]
        second = first[::2]  # [1, 3, 5, 7]

        assert second.shape == (4,)
        np.testing.assert_array_equal(second.to_numpy(), [1, 3, 5, 7])


class TestSliceNumpyComparison:
    """Compare HPXPy slicing with NumPy behavior."""

    @pytest.mark.parametrize("slice_spec", [
        slice(2, 5),
        slice(None, 5),
        slice(5, None),
        slice(None, None),
        slice(None, None, 2),
        slice(1, None, 2),
        slice(2, 8, 2),
        slice(-3, None),
        slice(None, -3),
        slice(-5, -2),
        slice(5, 5),
        slice(5, 6),
    ])
    def test_slice_matches_numpy(self, hpx_runtime, slice_spec):
        """Verify HPXPy slice matches NumPy slice."""
        np_arr = np.arange(10, dtype=np.float64)
        arr = hpx_runtime.from_numpy(np_arr)

        expected = np_arr[slice_spec]
        result = arr[slice_spec]

        assert result.shape == expected.shape
        np.testing.assert_array_equal(result.to_numpy(), expected)


class TestReshape:
    """Test reshape operations."""

    def test_reshape_1d_to_2d(self, hpx_runtime):
        """Reshape 1D array to 2D."""
        arr = hpx_runtime.arange(12)
        reshaped = arr.reshape((3, 4))

        assert reshaped.shape == (3, 4)
        assert reshaped.size == 12
        np.testing.assert_array_equal(
            reshaped.to_numpy(),
            np.arange(12).reshape((3, 4))
        )

    def test_reshape_2d_to_1d(self, hpx_runtime):
        """Reshape 2D array back to 1D."""
        np_arr = np.arange(12, dtype=np.float64).reshape((3, 4))
        arr = hpx_runtime.from_numpy(np_arr)
        reshaped = arr.reshape((12,))

        assert reshaped.shape == (12,)
        np.testing.assert_array_equal(reshaped.to_numpy(), np.arange(12))

    def test_reshape_2d_to_2d(self, hpx_runtime):
        """Reshape 2D to different 2D shape."""
        np_arr = np.arange(12, dtype=np.float64).reshape((3, 4))
        arr = hpx_runtime.from_numpy(np_arr)
        reshaped = arr.reshape((4, 3))

        assert reshaped.shape == (4, 3)
        np.testing.assert_array_equal(
            reshaped.to_numpy(),
            np_arr.reshape((4, 3))
        )

    def test_reshape_infer_dimension(self, hpx_runtime):
        """Reshape with -1 to infer dimension."""
        arr = hpx_runtime.arange(12)
        reshaped = arr.reshape((3, -1))

        assert reshaped.shape == (3, 4)

    def test_reshape_infer_first_dim(self, hpx_runtime):
        """Reshape with -1 as first dimension."""
        arr = hpx_runtime.arange(12)
        reshaped = arr.reshape((-1, 4))

        assert reshaped.shape == (3, 4)

    def test_reshape_preserves_dtype(self, hpx_runtime):
        """Reshape preserves data type."""
        np_arr = np.arange(12, dtype=np.float32)
        arr = hpx_runtime.from_numpy(np_arr)
        reshaped = arr.reshape((3, 4))

        assert reshaped.dtype == np.float32

    def test_reshape_contiguous_returns_view(self, hpx_runtime):
        """Reshape of contiguous array shares data (view)."""
        arr = hpx_runtime.arange(12)
        reshaped = arr.reshape((3, 4))

        # View should have the same underlying data
        # We verify by checking the values match
        np.testing.assert_array_equal(
            reshaped.to_numpy(),
            arr.to_numpy().reshape((3, 4))
        )

    def test_reshape_invalid_size(self, hpx_runtime):
        """Reshape with invalid size raises error."""
        arr = hpx_runtime.arange(12)

        with pytest.raises(RuntimeError, match="Cannot reshape"):
            arr.reshape((3, 5))  # 15 != 12

    def test_reshape_multiple_infer_fails(self, hpx_runtime):
        """Multiple -1 dimensions should fail."""
        arr = hpx_runtime.arange(12)

        with pytest.raises(RuntimeError, match="one unknown dimension"):
            arr.reshape((-1, -1))

    def test_reshape_to_3d(self, hpx_runtime):
        """Reshape to 3D array."""
        arr = hpx_runtime.arange(24)
        reshaped = arr.reshape((2, 3, 4))

        assert reshaped.shape == (2, 3, 4)
        np.testing.assert_array_equal(
            reshaped.to_numpy(),
            np.arange(24).reshape((2, 3, 4))
        )


class TestFlatten:
    """Test flatten operations."""

    def test_flatten_1d(self, hpx_runtime):
        """Flatten already 1D array."""
        arr = hpx_runtime.arange(10)
        flattened = arr.flatten()

        assert flattened.shape == (10,)
        np.testing.assert_array_equal(flattened.to_numpy(), np.arange(10))

    def test_flatten_2d(self, hpx_runtime):
        """Flatten 2D array."""
        np_arr = np.arange(12, dtype=np.float64).reshape((3, 4))
        arr = hpx_runtime.from_numpy(np_arr)
        flattened = arr.flatten()

        assert flattened.shape == (12,)
        np.testing.assert_array_equal(flattened.to_numpy(), np_arr.flatten())

    def test_flatten_returns_copy(self, hpx_runtime):
        """Flatten always returns a copy."""
        arr = hpx_runtime.arange(10)
        flattened = arr.flatten()

        # Should be equal but independent data
        assert flattened.shape == arr.shape
        np.testing.assert_array_equal(flattened.to_numpy(), arr.to_numpy())

    def test_flatten_preserves_dtype(self, hpx_runtime):
        """Flatten preserves data type."""
        np_arr = np.arange(12, dtype=np.int32).reshape((3, 4))
        arr = hpx_runtime.from_numpy(np_arr)
        flattened = arr.flatten()

        assert flattened.dtype == np.int32

    def test_flatten_non_contiguous(self, hpx_runtime):
        """Flatten non-contiguous (sliced) array."""
        arr = hpx_runtime.arange(10)
        sliced = arr[::2]  # [0, 2, 4, 6, 8]
        flattened = sliced.flatten()

        assert flattened.shape == (5,)
        np.testing.assert_array_equal(flattened.to_numpy(), [0, 2, 4, 6, 8])


class TestRavel:
    """Test ravel operations."""

    def test_ravel_1d(self, hpx_runtime):
        """Ravel already 1D array."""
        arr = hpx_runtime.arange(10)
        raveled = arr.ravel()

        assert raveled.shape == (10,)
        np.testing.assert_array_equal(raveled.to_numpy(), np.arange(10))

    def test_ravel_2d(self, hpx_runtime):
        """Ravel 2D array."""
        np_arr = np.arange(12, dtype=np.float64).reshape((3, 4))
        arr = hpx_runtime.from_numpy(np_arr)
        raveled = arr.ravel()

        assert raveled.shape == (12,)
        np.testing.assert_array_equal(raveled.to_numpy(), np_arr.ravel())

    def test_ravel_contiguous_returns_view(self, hpx_runtime):
        """Ravel of contiguous array returns a view."""
        arr = hpx_runtime.arange(10)
        raveled = arr.ravel()

        # Should match original data
        np.testing.assert_array_equal(raveled.to_numpy(), arr.to_numpy())

    def test_ravel_preserves_dtype(self, hpx_runtime):
        """Ravel preserves data type."""
        np_arr = np.arange(12, dtype=np.float32).reshape((3, 4))
        arr = hpx_runtime.from_numpy(np_arr)
        raveled = arr.ravel()

        assert raveled.dtype == np.float32

    def test_ravel_non_contiguous(self, hpx_runtime):
        """Ravel non-contiguous (sliced) array returns copy."""
        arr = hpx_runtime.arange(10)
        sliced = arr[::2]  # Non-contiguous due to step
        raveled = sliced.ravel()

        assert raveled.shape == (5,)
        np.testing.assert_array_equal(raveled.to_numpy(), [0, 2, 4, 6, 8])


class TestReshapeNumpyComparison:
    """Compare HPXPy reshape with NumPy behavior."""

    @pytest.mark.parametrize("original_shape,new_shape", [
        ((12,), (3, 4)),
        ((12,), (4, 3)),
        ((12,), (2, 6)),
        ((12,), (6, 2)),
        ((12,), (2, 2, 3)),
        ((12,), (3, -1)),
        ((12,), (-1, 4)),
        ((3, 4), (12,)),
        ((3, 4), (4, 3)),
        ((3, 4), (2, 6)),
        ((2, 3, 4), (24,)),
        ((2, 3, 4), (6, 4)),
        ((2, 3, 4), (4, 6)),
    ])
    def test_reshape_matches_numpy(self, hpx_runtime, original_shape, new_shape):
        """Verify HPXPy reshape matches NumPy reshape."""
        size = 1
        for dim in original_shape:
            size *= dim

        np_arr = np.arange(size, dtype=np.float64).reshape(original_shape)
        arr = hpx_runtime.from_numpy(np_arr)

        expected = np_arr.reshape(new_shape)
        result = arr.reshape(new_shape)

        assert result.shape == expected.shape
        np.testing.assert_array_equal(result.to_numpy(), expected)


# Phase 7: Integer Indexing and Assignment Tests

class TestIntegerIndexing:
    """Test integer indexing operations."""

    def test_getitem_int_positive(self, hpx_runtime):
        """Get element by positive index: arr[5]."""
        arr = hpx_runtime.arange(10)
        value = arr[5]

        assert value == 5.0

    def test_getitem_int_first(self, hpx_runtime):
        """Get first element: arr[0]."""
        arr = hpx_runtime.arange(10)
        value = arr[0]

        assert value == 0.0

    def test_getitem_int_last(self, hpx_runtime):
        """Get last element with positive index: arr[9]."""
        arr = hpx_runtime.arange(10)
        value = arr[9]

        assert value == 9.0

    def test_getitem_int_negative(self, hpx_runtime):
        """Get element by negative index: arr[-1]."""
        arr = hpx_runtime.arange(10)
        value = arr[-1]

        assert value == 9.0

    def test_getitem_int_negative_middle(self, hpx_runtime):
        """Get element by negative index: arr[-5]."""
        arr = hpx_runtime.arange(10)
        value = arr[-5]

        assert value == 5.0

    def test_getitem_int_out_of_bounds(self, hpx_runtime):
        """Out of bounds index raises error."""
        arr = hpx_runtime.arange(10)

        with pytest.raises((IndexError, RuntimeError)):
            _ = arr[10]

    def test_getitem_int_negative_out_of_bounds(self, hpx_runtime):
        """Negative out of bounds index raises error."""
        arr = hpx_runtime.arange(10)

        with pytest.raises((IndexError, RuntimeError)):
            _ = arr[-11]

    def test_getitem_int_float32(self, hpx_runtime):
        """Get element from float32 array."""
        np_arr = np.arange(10, dtype=np.float32)
        arr = hpx_runtime.from_numpy(np_arr)
        value = arr[5]

        assert isinstance(value, float)
        assert value == 5.0

    def test_getitem_int_int64(self, hpx_runtime):
        """Get element from int64 array."""
        np_arr = np.arange(10, dtype=np.int64)
        arr = hpx_runtime.from_numpy(np_arr)
        value = arr[5]

        assert value == 5

    def test_getitem_int_int32(self, hpx_runtime):
        """Get element from int32 array."""
        np_arr = np.arange(10, dtype=np.int32)
        arr = hpx_runtime.from_numpy(np_arr)
        value = arr[5]

        assert value == 5


class TestIntegerSetitem:
    """Test integer setitem operations."""

    def test_setitem_int_basic(self, hpx_runtime):
        """Set element by index: arr[5] = 42."""
        arr = hpx_runtime.arange(10)
        arr[5] = 42.0

        np.testing.assert_array_equal(
            arr.to_numpy(),
            [0, 1, 2, 3, 4, 42, 6, 7, 8, 9]
        )

    def test_setitem_int_first(self, hpx_runtime):
        """Set first element: arr[0] = 99."""
        arr = hpx_runtime.arange(10)
        arr[0] = 99.0

        assert arr[0] == 99.0

    def test_setitem_int_last(self, hpx_runtime):
        """Set last element: arr[9] = 99."""
        arr = hpx_runtime.arange(10)
        arr[9] = 99.0

        assert arr[9] == 99.0

    def test_setitem_int_negative(self, hpx_runtime):
        """Set element by negative index: arr[-1] = 42."""
        arr = hpx_runtime.arange(10)
        arr[-1] = 42.0

        assert arr[-1] == 42.0
        assert arr[9] == 42.0

    def test_setitem_int_negative_middle(self, hpx_runtime):
        """Set element by negative index: arr[-5] = 42."""
        arr = hpx_runtime.arange(10)
        arr[-5] = 42.0

        assert arr[5] == 42.0

    def test_setitem_int_out_of_bounds(self, hpx_runtime):
        """Out of bounds setitem raises error."""
        arr = hpx_runtime.arange(10)

        with pytest.raises((IndexError, RuntimeError)):
            arr[10] = 42.0

    def test_setitem_int_type_coercion(self, hpx_runtime):
        """Set with different type (int to float array)."""
        arr = hpx_runtime.arange(10)  # float64 by default
        arr[5] = 42  # Set int value

        assert arr[5] == 42.0


class TestSliceSetitem:
    """Test slice setitem operations."""

    def test_setitem_slice_scalar(self, hpx_runtime):
        """Set slice with scalar: arr[2:5] = 42."""
        arr = hpx_runtime.arange(10)
        arr[2:5] = 42.0

        np.testing.assert_array_equal(
            arr.to_numpy(),
            [0, 1, 42, 42, 42, 5, 6, 7, 8, 9]
        )

    def test_setitem_slice_from_start(self, hpx_runtime):
        """Set slice from start: arr[:3] = 0."""
        arr = hpx_runtime.arange(10)
        arr[:3] = 0.0

        np.testing.assert_array_equal(
            arr.to_numpy(),
            [0, 0, 0, 3, 4, 5, 6, 7, 8, 9]
        )

    def test_setitem_slice_to_end(self, hpx_runtime):
        """Set slice to end: arr[7:] = 99."""
        arr = hpx_runtime.arange(10)
        arr[7:] = 99.0

        np.testing.assert_array_equal(
            arr.to_numpy(),
            [0, 1, 2, 3, 4, 5, 6, 99, 99, 99]
        )

    def test_setitem_slice_full(self, hpx_runtime):
        """Set full slice: arr[:] = 5."""
        arr = hpx_runtime.arange(10)
        arr[:] = 5.0

        np.testing.assert_array_equal(
            arr.to_numpy(),
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        )

    def test_setitem_slice_step(self, hpx_runtime):
        """Set slice with step: arr[::2] = 0."""
        arr = hpx_runtime.arange(10)
        arr[::2] = 0.0

        np.testing.assert_array_equal(
            arr.to_numpy(),
            [0, 1, 0, 3, 0, 5, 0, 7, 0, 9]
        )

    def test_setitem_slice_negative(self, hpx_runtime):
        """Set slice with negative indices: arr[-3:] = 42."""
        arr = hpx_runtime.arange(10)
        arr[-3:] = 42.0

        np.testing.assert_array_equal(
            arr.to_numpy(),
            [0, 1, 2, 3, 4, 5, 6, 42, 42, 42]
        )

    def test_setitem_slice_array(self, hpx_runtime):
        """Set slice with array: arr[2:5] = [10, 20, 30]."""
        arr = hpx_runtime.arange(10)
        arr[2:5] = np.array([10, 20, 30], dtype=np.float64)

        np.testing.assert_array_equal(
            arr.to_numpy(),
            [0, 1, 10, 20, 30, 5, 6, 7, 8, 9]
        )

    def test_setitem_slice_hpxpy_array(self, hpx_runtime):
        """Set slice with HPXPy array."""
        arr = hpx_runtime.arange(10)
        values = hpx_runtime.ones(3) * 99.0
        arr[2:5] = values

        np.testing.assert_array_equal(
            arr.to_numpy(),
            [0, 1, 99, 99, 99, 5, 6, 7, 8, 9]
        )


class TestIndexingNumpyComparison:
    """Compare HPXPy indexing with NumPy behavior."""

    @pytest.mark.parametrize("index", [0, 1, 5, 9, -1, -5, -10])
    def test_getitem_matches_numpy(self, hpx_runtime, index):
        """Verify HPXPy integer indexing matches NumPy."""
        np_arr = np.arange(10, dtype=np.float64)
        arr = hpx_runtime.from_numpy(np_arr)

        expected = np_arr[index]
        result = arr[index]

        assert result == expected

    @pytest.mark.parametrize("slice_spec,value", [
        (slice(2, 5), 42.0),
        (slice(None, 3), 0.0),
        (slice(7, None), 99.0),
        (slice(None, None), 5.0),
        (slice(None, None, 2), 0.0),
        (slice(-3, None), 42.0),
    ])
    def test_setitem_slice_matches_numpy(self, hpx_runtime, slice_spec, value):
        """Verify HPXPy slice setitem matches NumPy."""
        np_arr = np.arange(10, dtype=np.float64)
        arr = hpx_runtime.from_numpy(np_arr.copy())

        np_arr[slice_spec] = value
        arr[slice_spec] = value

        np.testing.assert_array_equal(arr.to_numpy(), np_arr)
