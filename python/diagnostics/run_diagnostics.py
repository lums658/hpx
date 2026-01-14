#!/usr/bin/env python3
# HPXPy Diagnostic Test Suite
#
# SPDX-License-Identifier: BSL-1.0
#
# Run comprehensive diagnostics to verify HPX is being used correctly
# and measure scaling performance.

"""
HPXPy Diagnostic Test Suite

Run comprehensive diagnostics to verify HPX is being used correctly
and measure scaling performance.

Usage::

    python -m diagnostics.run_diagnostics [options]

Options::

    --full          Run all diagnostics (default)
    --quick         Run quick diagnostics only
    --scaling       Run scaling tests only
    --verification  Run HPX verification only
    --comparison    Run NumPy comparison only
    --threads N     Number of threads for tests (default: 8)
    --size N        Array size for benchmarks (default: 10000000)
"""

import argparse
import sys
from datetime import datetime


def run_quick_diagnostics(num_threads: int = 4):
    """Run quick diagnostic checks."""
    import hpxpy as hpx

    from diagnostics.verification.test_runtime import RuntimeDiagnostics

    print("=" * 60)
    print("HPXPy Quick Diagnostics")
    print("=" * 60)
    print()

    # Initialize runtime
    print(f"Initializing HPX with {num_threads} threads...")
    hpx.init(num_threads=num_threads)

    print(f"HPX runtime active: {hpx.is_running()}")
    print(f"HPX threads: {hpx.num_threads()}")
    print()

    # Runtime checks
    print("Runtime Diagnostics:")
    print("-" * 40)
    runtime_diag = RuntimeDiagnostics()
    for result in runtime_diag.run_all(expected_threads=num_threads):
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name}: {result.message}")
    print()

    hpx.finalize()


def run_verification_diagnostics(num_threads: int = 4):
    """Run HPX verification diagnostics."""
    import hpxpy as hpx

    from diagnostics.verification.test_hpx_usage import NumpyFallbackDetector

    print("=" * 60)
    print("HPX Verification Diagnostics")
    print("=" * 60)
    print()

    hpx.init(num_threads=num_threads)

    detector = NumpyFallbackDetector()
    detector.test_all_operations()
    print(detector.generate_report())

    hpx.finalize()


def run_comparison_diagnostics(num_threads: int = 8, array_size: int = 10_000_000):
    """Run NumPy comparison benchmarks."""
    import hpxpy as hpx

    from diagnostics.baselines.test_numpy_comparison import NumpyComparison

    print("=" * 85)
    print("HPXPy vs NumPy Comparison")
    print(f"Threads: {num_threads}, Array size: {array_size:,}")
    print("=" * 85)
    print()

    hpx.init(num_threads=num_threads)

    comparison = NumpyComparison(warmup=3, repeats=10)
    comparison.run_all_comparisons(array_size)
    print(comparison.generate_report())

    hpx.finalize()


def run_scaling_diagnostics(max_threads: int = 8, array_size: int = 10_000_000):
    """Run scaling diagnostics (requires subprocess execution)."""
    from diagnostics.scaling.test_scaling import ScalingRunner

    print("=" * 90)
    print("HPXPy Scaling Diagnostics")
    print(f"Max threads: {max_threads}, Array size: {array_size:,}")
    print("=" * 90)
    print()
    print("This will spawn multiple processes and may take several minutes...")
    print()

    # Determine thread counts
    thread_counts = [1]
    t = 2
    while t <= max_threads:
        thread_counts.append(t)
        t *= 2

    runner = ScalingRunner(thread_counts=thread_counts, array_size=array_size)

    # Run a subset for quick testing
    print("Running scaling tests for key operations...")
    print()

    results = []

    # Test a few representative operations
    test_ops = [
        ("sum", "seq"),
        ("sum", "par"),
        ("sqrt", "par"),  # Always parallel
    ]

    for op, policy in test_ops:
        try:
            print(f"  Testing {op} with policy={policy}...")
            result = runner.run_scaling_test(op, policy=policy)
            results.append(result)
        except Exception as e:
            print(f"    Failed: {e}")

    if results:
        print()
        print(runner.generate_scaling_report(results))


def main():
    parser = argparse.ArgumentParser(
        description="HPXPy Diagnostic Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m diagnostics.run_diagnostics --quick
  python -m diagnostics.run_diagnostics --scaling --threads 16
  python -m diagnostics.run_diagnostics --comparison --size 100000000
  python -m diagnostics.run_diagnostics --full
        """
    )
    parser.add_argument("--full", action="store_true", help="Run all diagnostics")
    parser.add_argument("--quick", action="store_true", help="Run quick diagnostics only")
    parser.add_argument("--scaling", action="store_true", help="Run scaling tests only")
    parser.add_argument("--verification", action="store_true", help="Run HPX verification only")
    parser.add_argument("--comparison", action="store_true", help="Run NumPy comparison only")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads (default: 8)")
    parser.add_argument("--size", type=int, default=10_000_000, help="Array size (default: 10000000)")

    args = parser.parse_args()

    # Default to full if no specific test selected
    if not any([args.quick, args.scaling, args.verification, args.comparison]):
        args.full = True

    print()
    print("HPXPy Diagnostic Test Suite")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    try:
        if args.full or args.quick:
            run_quick_diagnostics(args.threads)

        if args.full or args.verification:
            run_verification_diagnostics(args.threads)

        if args.full or args.comparison:
            run_comparison_diagnostics(args.threads, args.size)

        if args.full or args.scaling:
            run_scaling_diagnostics(args.threads, args.size)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Diagnostics Complete")
    print(f"Finished: {datetime.now().isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
