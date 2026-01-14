# HPXPy Runtime Tests
#
# SPDX-License-Identifier: BSL-1.0

"""Unit tests for HPX runtime management."""

import pytest


class TestRuntimeBasics:
    """Test basic runtime initialization and finalization."""

    def test_is_running_after_init(self, hpx_runtime):
        """Runtime should be running after initialization."""
        assert hpx_runtime.is_running()

    def test_num_threads(self, hpx_runtime):
        """Should report correct number of threads."""
        num = hpx_runtime.num_threads()
        assert num >= 1
        assert isinstance(num, int)

    def test_num_localities(self, hpx_runtime):
        """Should report at least 1 locality."""
        num = hpx_runtime.num_localities()
        assert num >= 1
        assert isinstance(num, int)

    def test_locality_id(self, hpx_runtime):
        """Locality ID should be valid."""
        lid = hpx_runtime.locality_id()
        assert lid >= 0
        assert lid < hpx_runtime.num_localities()


class TestRuntimeContextManager:
    """Test runtime context manager.

    Note: These tests run separately from the session fixture
    to test the context manager independently.
    """

    @pytest.mark.skip(reason="Cannot test context manager while session fixture is active")
    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        import hpxpy as hpx

        with hpx.runtime(num_threads=2):
            assert hpx.is_running()
            assert hpx.num_threads() >= 1

        assert not hpx.is_running()

    @pytest.mark.skip(reason="Cannot test context manager while session fixture is active")
    def test_context_manager_exception(self):
        """Runtime should finalize even if exception occurs."""
        import hpxpy as hpx

        try:
            with hpx.runtime():
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert not hpx.is_running()


class TestRuntimeErrors:
    """Test runtime error handling."""

    def test_double_init_raises(self, hpx_runtime):
        """Calling init twice should raise an error."""
        with pytest.raises(RuntimeError, match="already initialized"):
            hpx_runtime.init()

    @pytest.mark.skip(reason="Cannot test finalize while session fixture is active")
    def test_finalize_without_init_raises(self):
        """Calling finalize without init should raise an error."""
        import hpxpy as hpx

        with pytest.raises(RuntimeError, match="not initialized"):
            hpx.finalize()


class TestFinalizeRegression:
    """
    Regression tests for finalize() behavior.

    The finalize() function was fixed to use hpx::post() to schedule
    hpx::finalize() on an HPX thread before calling hpx::stop().
    This was needed because hpx::finalize() must be called from HPX thread context.
    """

    def test_finalize_can_be_called(self, hpx_runtime):
        """
        Verify that finalize() can be called without throwing.

        This test verifies the fix where finalize() would fail with:
        'this function can be called from an HPX thread only'

        Note: We can't actually call finalize() in this test because the
        session fixture manages the runtime. We just verify the runtime
        is in a state where finalize could be called (i.e., is_running).
        """
        # Runtime should be in a valid state
        assert hpx_runtime.is_running()
        # These functions should work (they're called from HPX context correctly)
        assert hpx_runtime.num_threads() >= 1
        assert hpx_runtime.num_localities() >= 1


class TestRuntimeRobustness:
    """Test runtime robustness and edge cases."""

    def test_operations_after_many_calls(self, hpx_runtime):
        """
        Runtime should remain stable after many operations.

        This tests that the runtime doesn't leak resources or become
        unstable after many array operations.
        """
        import numpy as np

        for i in range(100):
            arr = hpx_runtime.arange(100)
            _ = hpx_runtime.sum(arr)

        # Runtime should still be healthy
        assert hpx_runtime.is_running()
        assert hpx_runtime.num_threads() >= 1

    def test_parallel_sum_stability(self, hpx_runtime):
        """
        Parallel sum should be stable across multiple calls.

        This is a regression test to ensure HPX parallel algorithms
        work correctly after the runtime initialization fix.
        """
        import numpy as np

        for _ in range(10):
            arr = hpx_runtime.arange(1000)
            result = hpx_runtime.sum(arr)
            # Sum of 0..999 = 999*1000/2 = 499500
            # hpx.sum returns a Python float directly
            np.testing.assert_almost_equal(result, 499500.0)
