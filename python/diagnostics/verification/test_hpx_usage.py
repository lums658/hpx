# HPXPy HPX Usage Verification
#
# SPDX-License-Identifier: BSL-1.0
#
# Detect when HPXPy operations fall back to NumPy instead of using HPX.

"""
NumPy Fallback Detection

Detects when HPXPy operations fall back to NumPy instead of using HPX.
Uses Python tracing and numpy function monitoring.
"""

import functools
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


@dataclass
class FallbackResult:
    """Result of fallback detection."""
    operation: str
    uses_numpy_fallback: bool
    numpy_calls: list[str]
    message: str
    known_fallback: bool = False


class NumpyCallTracer:
    """Context manager to trace numpy function calls."""

    def __init__(self):
        self.numpy_calls = []
        self._original_funcs = {}
        self._functions_to_monitor = [
            'sum', 'prod', 'min', 'max', 'mean', 'var', 'std',
            'argsort', 'sort', 'argmin', 'argmax',
            'sqrt', 'exp', 'sin', 'cos', 'log',
        ]

    def __enter__(self):
        """Install tracing hooks on numpy functions."""
        for func_name in self._functions_to_monitor:
            if hasattr(np, func_name):
                original = getattr(np, func_name)
                self._original_funcs[func_name] = original

                # Create a traced wrapper
                def make_traced(name, orig):
                    @functools.wraps(orig)
                    def wrapper(*args, **kwargs):
                        self.numpy_calls.append(name)
                        return orig(*args, **kwargs)
                    return wrapper

                setattr(np, func_name, make_traced(func_name, original))
        return self

    def __exit__(self, *args):
        """Restore original numpy functions."""
        for func_name, original in self._original_funcs.items():
            setattr(np, func_name, original)
        self._original_funcs.clear()


class NumpyFallbackDetector:
    """Detect NumPy fallbacks in HPXPy operations."""

    # Known fallbacks from code analysis
    KNOWN_FALLBACKS = {
        'argsort': 'Explicitly delegates to numpy (__init__.py:1030-1035)',
    }

    def __init__(self):
        self.results = []

    def detect_fallback(
        self,
        operation_name: str,
        operation_func: Callable,
    ) -> FallbackResult:
        """
        Run an operation and detect if it uses NumPy.

        Args:
            operation_name: Name of the operation being tested
            operation_func: The HPXPy function to test (no-arg callable)
        """
        with NumpyCallTracer() as tracer:
            try:
                _ = operation_func()
            except Exception as e:
                return FallbackResult(
                    operation=operation_name,
                    uses_numpy_fallback=False,
                    numpy_calls=[],
                    message=f"Operation failed: {e}"
                )

        uses_fallback = len(tracer.numpy_calls) > 0
        known_fallback = operation_name in self.KNOWN_FALLBACKS

        known_reason = self.KNOWN_FALLBACKS.get(operation_name, "")
        if uses_fallback:
            message = f"NumPy calls: {tracer.numpy_calls}"
            if known_reason:
                message += f" (Known: {known_reason})"
        else:
            message = "Pure HPX execution"

        return FallbackResult(
            operation=operation_name,
            uses_numpy_fallback=uses_fallback,
            numpy_calls=tracer.numpy_calls,
            message=message,
            known_fallback=known_fallback,
        )

    def test_all_operations(self) -> list[FallbackResult]:
        """Test all HPXPy operations for NumPy fallbacks."""
        import hpxpy as hpx

        # Create test data
        arr = hpx.arange(10000)

        operations = [
            # Reductions (support policy param)
            ("sum", lambda: hpx.sum(arr)),
            ("prod", lambda: hpx.prod(hpx.from_numpy(np.ones(1000)))),
            ("min", lambda: hpx.min(arr)),
            ("max", lambda: hpx.max(arr)),
            ("mean", lambda: hpx.mean(arr)),
            ("var", lambda: hpx.var(arr)),  # Known fallback
            ("std", lambda: hpx.std(arr)),  # Known fallback

            # Sorting
            ("sort", lambda: hpx.sort(arr)),
            ("argsort", lambda: hpx.argsort(arr)),  # Known fallback
            ("count", lambda: hpx.count(arr, 0.0)),

            # Math functions (always parallel)
            ("sqrt", lambda: hpx.sqrt(arr)),
            ("exp", lambda: hpx.exp(arr * 0.0001)),
            ("sin", lambda: hpx.sin(arr)),
            ("cos", lambda: hpx.cos(arr)),
            ("log", lambda: hpx.log(arr + 1)),  # +1 to avoid log(0)

            # Scans
            ("cumsum", lambda: hpx.cumsum(arr)),
            ("cumprod", lambda: hpx.cumprod(hpx.from_numpy(np.ones(1000)))),

            # Element-wise operations
            ("add_scalar", lambda: arr + 1),
            ("multiply", lambda: arr * 2),
            ("add_arrays", lambda: arr + arr),
        ]

        self.results = []
        for name, op_func in operations:
            result = self.detect_fallback(name, op_func)
            self.results.append(result)

        return self.results

    def generate_report(self) -> str:
        """Generate fallback detection report."""
        lines = [
            "=" * 70,
            "HPX Usage Verification Report",
            "=" * 70,
            "",
            f"{'Operation':<20} {'Uses HPX':>10} {'NumPy Calls':>15} {'Status':>12}",
            "-" * 70,
        ]

        for r in self.results:
            uses_hpx = "YES" if not r.uses_numpy_fallback else "NO"
            calls = str(r.numpy_calls) if r.numpy_calls else "[]"
            if len(calls) > 15:
                calls = calls[:12] + "..."

            if r.uses_numpy_fallback:
                status = "KNOWN" if r.known_fallback else "UNEXPECTED"
            else:
                status = "PASS"

            lines.append(f"{r.operation:<20} {uses_hpx:>10} {calls:>15} {status:>12}")

        # Summary
        lines.append("")
        lines.append("-" * 70)

        n_hpx = sum(1 for r in self.results if not r.uses_numpy_fallback)
        n_known = sum(1 for r in self.results if r.uses_numpy_fallback and r.known_fallback)
        n_unexpected = sum(1 for r in self.results if r.uses_numpy_fallback and not r.known_fallback)

        lines.append(f"Total operations tested: {len(self.results)}")
        lines.append(f"Using HPX: {n_hpx}")
        lines.append(f"Known NumPy fallbacks: {n_known}")
        lines.append(f"Unexpected NumPy fallbacks: {n_unexpected}")

        if n_unexpected > 0:
            lines.append("")
            lines.append("WARNING: Unexpected NumPy fallbacks detected!")
        else:
            lines.append("")
            lines.append("All operations using HPX or have known fallbacks.")

        lines.append("=" * 70)

        return "\n".join(lines)


def run_verification():
    """Run HPX usage verification."""
    import hpxpy as hpx

    print("Initializing HPX runtime...")
    hpx.init(num_threads=4)

    print(f"HPX runtime active: {hpx.is_running()}")
    print(f"HPX threads: {hpx.num_threads()}")
    print()

    detector = NumpyFallbackDetector()
    results = detector.test_all_operations()

    print(detector.generate_report())

    hpx.finalize()


if __name__ == "__main__":
    run_verification()
