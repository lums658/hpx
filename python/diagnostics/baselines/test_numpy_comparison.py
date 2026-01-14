# HPXPy NumPy Comparison Benchmarks
#
# SPDX-License-Identifier: BSL-1.0
#
# Compare HPXPy performance against NumPy.

"""
NumPy Comparison Benchmarks

Compare HPXPy operations against NumPy to identify overhead and speedups.
"""

import time
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class ComparisonResult:
    """Result of HPXPy vs NumPy comparison."""
    operation: str
    array_size: int
    hpxpy_time_ms: float
    numpy_time_ms: float
    speedup: float  # numpy_time / hpxpy_time (>1 means HPXPy faster)
    overhead_ms: float  # hpxpy_time - numpy_time (negative means HPXPy faster)
    verdict: str  # "FASTER", "SLOWER", "COMPARABLE"


class NumpyComparison:
    """Compare HPXPy operations against NumPy."""

    SPEEDUP_THRESHOLD_FASTER = 1.1  # >10% faster
    SPEEDUP_THRESHOLD_SLOWER = 0.9  # <10% slower

    def __init__(self, warmup: int = 3, repeats: int = 10):
        self.warmup = warmup
        self.repeats = repeats
        self.results = []

    def _benchmark(self, func: Callable) -> float:
        """Benchmark a function, return minimum time in seconds."""
        # Warmup
        for _ in range(self.warmup):
            func()

        # Benchmark
        times = []
        for _ in range(self.repeats):
            start = time.perf_counter()
            func()
            times.append(time.perf_counter() - start)

        return min(times)

    def compare_operation(
        self,
        operation_name: str,
        hpxpy_func: Callable,
        numpy_func: Callable,
        array_size: int,
    ) -> ComparisonResult:
        """Compare a single operation."""

        hpxpy_time = self._benchmark(hpxpy_func)
        numpy_time = self._benchmark(numpy_func)

        hpxpy_ms = hpxpy_time * 1000
        numpy_ms = numpy_time * 1000

        speedup = numpy_time / hpxpy_time if hpxpy_time > 0 else float('inf')
        overhead = hpxpy_ms - numpy_ms

        if speedup > self.SPEEDUP_THRESHOLD_FASTER:
            verdict = "FASTER"
        elif speedup < self.SPEEDUP_THRESHOLD_SLOWER:
            verdict = "SLOWER"
        else:
            verdict = "COMPARABLE"

        return ComparisonResult(
            operation=operation_name,
            array_size=array_size,
            hpxpy_time_ms=hpxpy_ms,
            numpy_time_ms=numpy_ms,
            speedup=speedup,
            overhead_ms=overhead,
            verdict=verdict,
        )

    def run_all_comparisons(self, array_size: int = 10_000_000) -> list[ComparisonResult]:
        """Run all comparison benchmarks."""
        import hpxpy as hpx

        # Create test arrays
        np_arr = np.arange(array_size, dtype=np.float64)
        hpx_arr = hpx.from_numpy(np_arr)

        # Small arrays for prod (avoid overflow)
        np_ones = np.ones(1000, dtype=np.float64)
        hpx_ones = hpx.from_numpy(np_ones)

        comparisons = [
            # Reductions (seq policy for fair comparison with NumPy's single-threaded)
            ("sum_seq", lambda: hpx.sum(hpx_arr, policy="seq"), lambda: np.sum(np_arr)),
            ("sum_par", lambda: hpx.sum(hpx_arr, policy="par"), lambda: np.sum(np_arr)),
            ("prod_seq", lambda: hpx.prod(hpx_ones, policy="seq"), lambda: np.prod(np_ones)),
            ("min_seq", lambda: hpx.min(hpx_arr, policy="seq"), lambda: np.min(np_arr)),
            ("max_seq", lambda: hpx.max(hpx_arr, policy="seq"), lambda: np.max(np_arr)),
            ("mean_seq", lambda: hpx.mean(hpx_arr, policy="seq"), lambda: np.mean(np_arr)),

            # Element-wise (always parallel in HPXPy)
            ("sqrt", lambda: hpx.sqrt(hpx_arr), lambda: np.sqrt(np_arr)),
            ("exp_scaled", lambda: hpx.exp(hpx_arr * 1e-8), lambda: np.exp(np_arr * 1e-8)),
            ("sin", lambda: hpx.sin(hpx_arr), lambda: np.sin(np_arr)),
            ("cos", lambda: hpx.cos(hpx_arr), lambda: np.cos(np_arr)),

            # Arithmetic
            ("add_scalar", lambda: hpx_arr + 1, lambda: np_arr + 1),
            ("multiply_scalar", lambda: hpx_arr * 2, lambda: np_arr * 2),
            ("add_arrays", lambda: hpx_arr + hpx_arr, lambda: np_arr + np_arr),

            # Scans
            ("cumsum", lambda: hpx.cumsum(hpx_arr), lambda: np.cumsum(np_arr)),
        ]

        self.results = []
        for name, hpx_func, np_func in comparisons:
            try:
                result = self.compare_operation(name, hpx_func, np_func, array_size)
                self.results.append(result)
            except Exception as e:
                print(f"Warning: {name} comparison failed: {e}")

        return self.results

    def generate_report(self) -> str:
        """Generate comparison report."""
        lines = [
            "=" * 85,
            "HPXPy vs NumPy Performance Comparison",
            "=" * 85,
            "",
            f"{'Operation':<20} {'HPXPy (ms)':>12} {'NumPy (ms)':>12} {'Speedup':>10} {'Verdict':>12}",
            "-" * 85,
        ]

        for r in self.results:
            lines.append(
                f"{r.operation:<20} {r.hpxpy_time_ms:>12.3f} {r.numpy_time_ms:>12.3f} "
                f"{r.speedup:>9.2f}x {r.verdict:>12}"
            )

        # Summary statistics
        lines.append("")
        lines.append("-" * 85)

        faster_count = sum(1 for r in self.results if r.verdict == "FASTER")
        slower_count = sum(1 for r in self.results if r.verdict == "SLOWER")
        comparable_count = sum(1 for r in self.results if r.verdict == "COMPARABLE")

        avg_speedup = sum(r.speedup for r in self.results) / len(self.results) if self.results else 0
        max_speedup = max(r.speedup for r in self.results) if self.results else 0
        min_speedup = min(r.speedup for r in self.results) if self.results else 0

        lines.append(f"Summary: {faster_count} faster, {comparable_count} comparable, {slower_count} slower")
        lines.append(f"Average speedup: {avg_speedup:.2f}x")
        lines.append(f"Range: {min_speedup:.2f}x to {max_speedup:.2f}x")
        lines.append("=" * 85)

        return "\n".join(lines)


def run_numpy_comparison(array_size: int = 10_000_000):
    """Run NumPy comparison benchmarks."""
    import hpxpy as hpx

    print("Initializing HPX runtime...")
    hpx.init(num_threads=8)

    print(f"HPX threads: {hpx.num_threads()}")
    print(f"Array size: {array_size:,}")
    print()

    comparison = NumpyComparison(warmup=3, repeats=10)
    comparison.run_all_comparisons(array_size)

    print(comparison.generate_report())

    hpx.finalize()


if __name__ == "__main__":
    run_numpy_comparison()
