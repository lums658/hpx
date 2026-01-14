# HPXPy Scaling Diagnostics
#
# SPDX-License-Identifier: BSL-1.0
#
# Tests for scaling performance with different thread counts and execution policies.

"""
Scaling Tests for HPXPy

Tests operations with different thread counts and execution policies to measure
parallel speedup and efficiency.
"""

import subprocess
import sys
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScalingResult:
    """Result of a scaling test."""
    operation: str
    array_size: int
    thread_counts: list[int]
    times_seconds: list[float]
    speedups: list[float]  # Relative to 1-thread
    efficiencies: list[float]  # speedup / ideal_speedup
    policy: str  # "seq" or "par"
    expected_to_scale: bool
    actually_scales: bool  # >50% efficiency at max threads

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "array_size": self.array_size,
            "thread_counts": self.thread_counts,
            "times_seconds": self.times_seconds,
            "speedups": self.speedups,
            "efficiencies": self.efficiencies,
            "policy": self.policy,
            "expected_to_scale": self.expected_to_scale,
            "actually_scales": self.actually_scales,
        }


# Template script for subprocess execution
BENCHMARK_SCRIPT_TEMPLATE = '''
import sys
import time
import json

# Import hpxpy
import hpxpy as hpx
import numpy as np

def run_benchmark(operation, array_size, num_threads, policy, warmup=2, repeats=5):
    """Run a single benchmark."""
    hpx.init(num_threads=num_threads)

    # Create test array
    arr = hpx.arange(array_size, dtype=np.float64)

    # Operation dispatch - operations that support policy parameter
    policy_ops = {
        "sum": lambda a: hpx.sum(a, policy=policy),
        "prod": lambda a: hpx.prod(hpx.from_numpy(np.ones(min(array_size, 10000))), policy=policy),
        "min": lambda a: hpx.min(a, policy=policy),
        "max": lambda a: hpx.max(a, policy=policy),
        "sort": lambda a: hpx.sort(a, policy=policy),
        "count": lambda a: hpx.count(a, 0.0, policy=policy),
        "mean": lambda a: hpx.mean(a, policy=policy),
    }

    # Operations that always use parallel execution
    parallel_ops = {
        "sqrt": lambda a: hpx.sqrt(a),
        "exp": lambda a: hpx.exp(a * 0.00001),  # Scale to avoid overflow
        "sin": lambda a: hpx.sin(a),
        "cumsum": lambda a: hpx.cumsum(a),
        "add_scalar": lambda a: a + 1.0,
        "multiply_arrays": lambda a: a * a,
    }

    if operation in policy_ops:
        op_func = policy_ops[operation]
    elif operation in parallel_ops:
        op_func = parallel_ops[operation]
    else:
        raise ValueError(f"Unknown operation: {operation}")

    # Warmup
    for _ in range(warmup):
        _ = op_func(arr)

    # Benchmark
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = op_func(arr)
        times.append(time.perf_counter() - start)

    hpx.finalize()

    return min(times)

if __name__ == "__main__":
    operation = sys.argv[1]
    array_size = int(sys.argv[2])
    num_threads = int(sys.argv[3])
    policy = sys.argv[4]

    time_sec = run_benchmark(operation, array_size, num_threads, policy)
    print(json.dumps({"time": time_sec}))
'''


class ScalingRunner:
    """Run scaling tests across multiple thread counts."""

    def __init__(
        self,
        thread_counts: list[int] = None,
        array_size: int = 10_000_000,
    ):
        self.thread_counts = thread_counts or [1, 2, 4, 8]
        self.array_size = array_size

        # Operations that support policy parameter
        self.policy_operations = ["sum", "prod", "min", "max", "sort", "count", "mean"]

        # Operations that always use parallel execution (no policy param)
        self.parallel_operations = ["sqrt", "exp", "sin", "cumsum", "add_scalar", "multiply_arrays"]

    def _run_single_benchmark(
        self,
        operation: str,
        array_size: int,
        num_threads: int,
        policy: str = "seq"
    ) -> float:
        """Run a single benchmark in a subprocess."""

        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(BENCHMARK_SCRIPT_TEMPLATE)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path, operation, str(array_size), str(num_threads), policy],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Benchmark failed: {result.stderr}")

            data = json.loads(result.stdout.strip())
            return data["time"]

        finally:
            os.unlink(script_path)

    def run_scaling_test(
        self,
        operation: str,
        policy: str = "seq",
    ) -> ScalingResult:
        """Run scaling test for a single operation."""

        times = []
        for n_threads in self.thread_counts:
            t = self._run_single_benchmark(operation, self.array_size, n_threads, policy)
            times.append(t)

        # Calculate speedups relative to 1 thread
        base_time = times[0]
        speedups = [base_time / t if t > 0 else 0 for t in times]

        # Calculate efficiency (speedup / ideal)
        efficiencies = [
            speedup / n_threads
            for speedup, n_threads in zip(speedups, self.thread_counts)
        ]

        # Determine if operation is expected to scale
        # Operations with policy="par" should scale
        # Parallel operations (sqrt, exp, etc) always scale
        expected_to_scale = (policy == "par") or (operation in self.parallel_operations)

        # Determine if it actually scales (>50% efficiency at max threads)
        max_efficiency = efficiencies[-1]
        actually_scales = max_efficiency > 0.5

        return ScalingResult(
            operation=operation,
            array_size=self.array_size,
            thread_counts=self.thread_counts,
            times_seconds=times,
            speedups=speedups,
            efficiencies=efficiencies,
            policy=policy,
            expected_to_scale=expected_to_scale,
            actually_scales=actually_scales,
        )

    def run_all_scaling_tests(self) -> list[ScalingResult]:
        """Run scaling tests for all operations."""
        results = []

        # Test policy-controllable operations with both seq and par
        for op in self.policy_operations:
            try:
                # Test with seq policy (shouldn't scale)
                result_seq = self.run_scaling_test(op, policy="seq")
                results.append(result_seq)

                # Test with par policy (should scale)
                result_par = self.run_scaling_test(op, policy="par")
                results.append(result_par)
            except Exception as e:
                print(f"Warning: {op} failed: {e}")

        # Test always-parallel operations
        for op in self.parallel_operations:
            try:
                result = self.run_scaling_test(op, policy="par")  # Policy ignored but passed
                results.append(result)
            except Exception as e:
                print(f"Warning: {op} failed: {e}")

        return results

    def generate_scaling_report(self, results: list[ScalingResult]) -> str:
        """Generate a text report of scaling results."""
        lines = [
            "=" * 90,
            "HPXPy Scaling Diagnostic Report",
            f"Array size: {self.array_size:,}",
            f"Thread counts: {self.thread_counts}",
            "=" * 90,
            "",
        ]

        # Group by expected scaling behavior
        should_scale = [r for r in results if r.expected_to_scale]
        should_not_scale = [r for r in results if not r.expected_to_scale]

        # Section for operations expected to scale
        lines.append("OPERATIONS EXPECTED TO SCALE (policy='par' or always-parallel):")
        lines.append("-" * 90)
        header = f"{'Operation':<20} {'Policy':<8}"
        for tc in self.thread_counts:
            header += f" {tc}T (ms)"
        header += f" {'Eff@' + str(self.thread_counts[-1]) + 'T':>10} {'Status':>8}"
        lines.append(header)
        lines.append("-" * 90)

        for r in should_scale:
            times_str = "".join(f" {t*1000:>8.2f}" for t in r.times_seconds)
            eff_str = f"{r.efficiencies[-1]*100:>9.1f}%"
            status = "PASS" if r.actually_scales else "FAIL"
            lines.append(f"{r.operation:<20} {r.policy:<8}{times_str} {eff_str} {status:>8}")

        lines.append("")

        # Section for operations NOT expected to scale
        lines.append("OPERATIONS NOT EXPECTED TO SCALE (policy='seq'):")
        lines.append("-" * 90)
        lines.append(header)
        lines.append("-" * 90)

        for r in should_not_scale:
            times_str = "".join(f" {t*1000:>8.2f}" for t in r.times_seconds)
            eff_str = f"{r.efficiencies[-1]*100:>9.1f}%"
            # Sequential ops should NOT scale, so low efficiency is expected (PASS)
            status = "PASS" if not r.actually_scales else "UNEXPECTED"
            lines.append(f"{r.operation:<20} {r.policy:<8}{times_str} {eff_str} {status:>8}")

        # Summary
        lines.append("")
        lines.append("=" * 90)
        lines.append("Summary:")

        scale_pass = sum(1 for r in should_scale if r.actually_scales)
        scale_fail = sum(1 for r in should_scale if not r.actually_scales)
        noscale_pass = sum(1 for r in should_not_scale if not r.actually_scales)
        noscale_unexpected = sum(1 for r in should_not_scale if r.actually_scales)

        lines.append(f"  Operations expected to scale: {scale_pass} pass, {scale_fail} fail")
        lines.append(f"  Operations expected NOT to scale: {noscale_pass} pass, {noscale_unexpected} unexpected")
        lines.append("=" * 90)

        return "\n".join(lines)


def run_quick_scaling_test():
    """Run a quick scaling test for demonstration."""
    import hpxpy as hpx

    print("=" * 60)
    print("HPXPy Quick Scaling Test")
    print("=" * 60)
    print()

    # Test within a single process by comparing seq vs par
    hpx.init(num_threads=8)

    arr = hpx.arange(10_000_000, dtype='float64')

    print(f"Array size: {arr.size:,}")
    print(f"HPX threads: {hpx.num_threads()}")
    print()

    # Test sum with seq vs par
    warmup = 2
    repeats = 5

    for policy in ["seq", "par"]:
        # Warmup
        for _ in range(warmup):
            hpx.sum(arr, policy=policy)

        # Time
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = hpx.sum(arr, policy=policy)
            times.append(time.perf_counter() - start)

        min_time = min(times)
        print(f"sum(policy='{policy}'): {min_time*1000:.2f} ms")

    print()

    # Test element-wise operations (always parallel)
    for _ in range(warmup):
        hpx.sqrt(arr)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = hpx.sqrt(arr)
        times.append(time.perf_counter() - start)

    print(f"sqrt (always par): {min(times)*1000:.2f} ms")

    hpx.finalize()
    print()
    print("Done.")


if __name__ == "__main__":
    run_quick_scaling_test()
