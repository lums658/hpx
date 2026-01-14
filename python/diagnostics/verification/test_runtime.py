# HPXPy Runtime Diagnostics
#
# SPDX-License-Identifier: BSL-1.0
#
# Verify HPX runtime is correctly configured and active.

"""
HPX Runtime Verification

Verifies that the HPX runtime is properly initialized and configured.
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class RuntimeResult:
    """Result of a runtime diagnostic check."""
    name: str
    passed: bool
    message: str
    details: Optional[dict] = None


class RuntimeDiagnostics:
    """Verify HPX runtime is correctly configured."""

    def __init__(self):
        self.results = []

    def check_runtime_active(self) -> RuntimeResult:
        """Verify HPX runtime is running."""
        import hpxpy as hpx

        if not hpx.is_running():
            return RuntimeResult(
                name="runtime_active",
                passed=False,
                message="HPX runtime is not running",
            )

        return RuntimeResult(
            name="runtime_active",
            passed=True,
            message=f"HPX runtime active with {hpx.num_threads()} threads",
            details={"num_threads": hpx.num_threads()}
        )

    def check_thread_count(self, expected: int) -> RuntimeResult:
        """Verify thread count matches expectation."""
        import hpxpy as hpx

        actual = hpx.num_threads()
        passed = actual == expected

        return RuntimeResult(
            name="thread_count",
            passed=passed,
            message=f"Expected {expected} threads, got {actual}",
            details={"expected": expected, "actual": actual}
        )

    def check_locality_info(self) -> RuntimeResult:
        """Check locality information."""
        import hpxpy as hpx

        num_localities = hpx.num_localities()
        locality_id = hpx.locality_id()

        return RuntimeResult(
            name="locality_info",
            passed=True,
            message=f"Locality {locality_id} of {num_localities} total",
            details={
                "num_localities": num_localities,
                "locality_id": locality_id
            }
        )

    def check_basic_operations(self) -> RuntimeResult:
        """Verify basic operations work correctly."""
        import hpxpy as hpx
        import numpy as np

        try:
            # Test array creation
            arr = hpx.arange(1000)
            assert arr.size == 1000

            # Test reduction
            result = hpx.sum(arr)
            expected = 499500
            assert result == expected, f"sum mismatch: {result} != {expected}"

            # Test element-wise
            arr2 = arr + 1
            assert arr2[0] == 1.0

            return RuntimeResult(
                name="basic_operations",
                passed=True,
                message="Basic operations (arange, sum, add) work correctly"
            )

        except Exception as e:
            return RuntimeResult(
                name="basic_operations",
                passed=False,
                message=f"Basic operations failed: {e}"
            )

    def check_policy_support(self) -> RuntimeResult:
        """Verify execution policy support works."""
        import hpxpy as hpx

        try:
            arr = hpx.arange(10000)

            # Test all policies
            result_seq = hpx.sum(arr, policy="seq")
            result_par = hpx.sum(arr, policy="par")
            result_par_unseq = hpx.sum(arr, policy="par_unseq")

            # All should give same result
            expected = 49995000
            assert result_seq == expected
            assert result_par == expected
            assert result_par_unseq == expected

            return RuntimeResult(
                name="policy_support",
                passed=True,
                message="All execution policies (seq, par, par_unseq) work correctly"
            )

        except Exception as e:
            return RuntimeResult(
                name="policy_support",
                passed=False,
                message=f"Execution policy support failed: {e}"
            )

    def run_all(self, expected_threads: int = 4) -> list[RuntimeResult]:
        """Run all runtime diagnostics."""
        self.results = [
            self.check_runtime_active(),
            self.check_thread_count(expected_threads),
            self.check_locality_info(),
            self.check_basic_operations(),
            self.check_policy_support(),
        ]
        return self.results

    def generate_report(self) -> str:
        """Generate runtime diagnostics report."""
        lines = [
            "=" * 60,
            "HPX Runtime Diagnostics Report",
            "=" * 60,
            "",
        ]

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"[{status}] {r.name}: {r.message}")
            if r.details:
                for k, v in r.details.items():
                    lines.append(f"       {k}: {v}")

        lines.append("")

        n_pass = sum(1 for r in self.results if r.passed)
        n_fail = sum(1 for r in self.results if not r.passed)
        lines.append(f"Summary: {n_pass} passed, {n_fail} failed")
        lines.append("=" * 60)

        return "\n".join(lines)


def run_runtime_diagnostics(num_threads: int = 4):
    """Run runtime diagnostics."""
    import hpxpy as hpx

    print("Initializing HPX runtime...")
    hpx.init(num_threads=num_threads)

    diag = RuntimeDiagnostics()
    diag.run_all(expected_threads=num_threads)

    print(diag.generate_report())

    hpx.finalize()


if __name__ == "__main__":
    run_runtime_diagnostics()
