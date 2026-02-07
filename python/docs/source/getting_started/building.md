# Building HPXPy from Source

This guide covers building HPXPy against an **installed** HPX library.

## Prerequisites

### System Dependencies

- CMake 3.18+
- C++17 compiler (GCC 9+, Clang 10+, or Apple Clang 12+)
- Python 3.9+ with development headers

**macOS:**

```bash
brew install cmake python
```

**Ubuntu/Debian:**

```bash
sudo apt install cmake python3-dev python3-venv
```

### HPX (Installed)

HPXPy requires HPX to be built and **installed** at a known prefix. For example,
to install HPX to `~/usr/local/hpx`:

```bash
git clone https://github.com/STEllAR-GROUP/hpx.git
cd hpx && mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=~/usr/local/hpx \
    -DHPX_WITH_FETCH_ASIO=ON \
    -DHPX_WITH_EXAMPLES=OFF \
    -DHPX_WITH_TESTS=OFF

make -j$(nproc)      # Linux
make -j$(sysctl -n hw.ncpu)  # macOS

make install
```

After installation, the HPX CMake config files will be at
`<prefix>/lib/cmake/HPX/HPXConfig.cmake`.

## Build Instructions

### Step 1: Python Environment

Your active Python must have NumPy and pybind11 installed. Use whichever
environment you prefer (venv, conda, system Python):

```bash
pip install numpy pybind11
```

Or install all dependencies (including test and notebook extras):

```bash
pip install -r requirements.txt   # from hpx/python/
```

```{important}
CMake's `find_package(Python)` discovers whichever Python is active at
configure time. Make sure the Python environment you want to use is active
**before** running `cmake`.
```

### Step 2: Configure with CMake

Point CMake at the HPX install prefix using `CMAKE_PREFIX_PATH`:

```bash
mkdir build && cd build

cmake -DCMAKE_PREFIX_PATH=~/usr/local/hpx ..
```

You should see output like:

```
-- Found HPX: 2.0.0 at /Users/you/usr/local/hpx/lib/cmake/HPX
-- HPX prefix: /Users/you/usr/local/hpx
-- HPX lib dir: /Users/you/usr/local/hpx/lib
-- Found Python: 3.13.7 at /path/to/python/.venv/bin/python
-- Found NumPy: 2.4.1
-- Found pybind11: 2.13.6
```

Verify that:
- **HPX prefix** points to your HPX install
- **Python** points to your **venv** Python (not the system Python)
- **NumPy** and **pybind11** are found

### Step 3: Build

```bash
make -j$(nproc)      # Linux
make -j$(sysctl -n hw.ncpu)  # macOS
```

This compiles the `_core` extension module and copies it into the source tree's
`hpxpy/` directory for development use.

### Step 4: Set Up the Runtime Environment

CMake generates a `setup_env.sh` in the build directory. Source it to configure
library paths and `PYTHONPATH`:

```bash
source setup_env.sh
```

This sets:
- `DYLD_LIBRARY_PATH` (macOS) or `LD_LIBRARY_PATH` (Linux) to find HPX shared
  libraries
- `PYTHONPATH` to include the build and source directories so `import hpxpy`
  works

### Step 5: Verify

```bash
python -c "import hpxpy as hpx; hpx.init(); print('HPXPy works!'); hpx.finalize()"
```

## Quick Reference

Assuming HPX is installed at some prefix (e.g. `~/usr/local/hpx`):

```bash
cd hpx/python

# Ensure Python deps are installed
pip install numpy pybind11

# Build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=~/usr/local/hpx ..
make -j8

# Use
source setup_env.sh
python -c "import hpxpy as hpx; hpx.init(); print(hpx.sum(hpx.arange(100))); hpx.finalize()"
```

## Running Jupyter Notebooks

After building and sourcing the environment:

```bash
source build/setup_env.sh
cd tutorials    # or examples
jupyter lab
```

The notebooks can `import hpxpy` because `PYTHONPATH` is set by `setup_env.sh`.

```{tip}
If you launch VS Code, JupyterLab, or any other tool from the same terminal
where you sourced `setup_env.sh`, it will inherit the environment and hpxpy
will be importable.
```

## Installation (Optional)

To install hpxpy into the active Python environment's site-packages:

```bash
cd build
cmake --install .
```

This installs:
- `_core.so` → `<site-packages>/hpxpy/`
- `hpxpy/*.py` → `<site-packages>/hpxpy/`

The installed module has RPATH set to the HPX lib directory, so it finds HPX
at runtime without `LD_LIBRARY_PATH`.

You can override the install location:

```bash
cmake -DHPXPY_INSTALL_PYTHONDIR=/custom/path ..
cmake --install .
```

## GPU Support (Optional)

### CUDA

Requires HPX built with `HPX_WITH_CUDA=ON` and the CUDA toolkit:

```bash
cmake -DCMAKE_PREFIX_PATH=~/usr/local/hpx -DHPXPY_WITH_CUDA=ON ..
```

### SYCL

Requires HPX built with `HPX_WITH_SYCL=ON` and a SYCL compiler:

```bash
cmake -DCMAKE_PREFIX_PATH=~/usr/local/hpx -DHPXPY_WITH_SYCL=ON ..
```

## Directory Structure After Build

```
hpx/
├── python/
│   ├── .venv/                    # Python virtual environment
│   ├── requirements.txt          # Python dependencies
│   ├── hpxpy/                    # Python package
│   │   ├── __init__.py
│   │   ├── _core.cpython-*.so    # ← copied here by post-build step
│   │   └── ...
│   ├── build/                    # CMake build directory
│   │   ├── _core.cpython-*.so    # Compiled extension
│   │   ├── setup_env.sh          # ← generated by CMake
│   │   └── ...
│   ├── tutorials/                # Jupyter notebooks
│   ├── examples/                 # Example notebooks
│   └── tests/                    # Test suite
└── ...
```

## Troubleshooting

### CMake finds wrong Python

If CMake picks up a Python that lacks NumPy or pybind11:

```
-- Found Python: 3.14.2 at /usr/bin/python3
```

**Fix:** Make sure the correct Python environment is active *before* running cmake:

```bash
which python  # verify this is the Python you want
python -c "import numpy; import pybind11"  # verify deps
cmake -DCMAKE_PREFIX_PATH=~/usr/local/hpx ..
```

If you already configured with the wrong Python, delete `CMakeCache.txt` (or
the entire build directory) and reconfigure.

### "Module not found: hpxpy"

Ensure you sourced the environment:

```bash
source build/setup_env.sh
```

### "Library not found: libhpx"

The `setup_env.sh` script sets `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` for you.
If you're running without it, set the library path manually:

```bash
# Linux
export LD_LIBRARY_PATH=~/usr/local/hpx/lib:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=~/usr/local/hpx/lib:$DYLD_LIBRARY_PATH
```

### CMake can't find HPX

Point CMake at the HPX install prefix (the directory containing `lib/`, `include/`, etc.):

```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/hpx/install ..
```

### NumPy or pybind11 not found

Install them in your active venv:

```bash
pip install -r requirements.txt
```
