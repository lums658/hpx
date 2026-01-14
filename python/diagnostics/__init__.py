# HPXPy Diagnostics Package
#
# SPDX-License-Identifier: BSL-1.0
#
# Diagnostic tools to verify HPX is being used correctly and measure scaling.

"""
HPXPy Diagnostics
=================

This package provides diagnostic tools to verify that HPX is actually being
used for computations (not falling back to NumPy) and to measure parallel
scaling performance.

Modules
-------
scaling
    Thread scaling tests comparing seq vs par execution policies.
verification
    Tests to verify HPX runtime and detect NumPy fallbacks.
baselines
    Performance comparison benchmarks against NumPy.

Usage
-----
Run all diagnostics::

    python -m diagnostics.run_diagnostics --full

Run specific diagnostics::

    python -m diagnostics.run_diagnostics --scaling
    python -m diagnostics.run_diagnostics --verification
    python -m diagnostics.run_diagnostics --comparison
"""

__version__ = "0.1.0"
