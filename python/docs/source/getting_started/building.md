# Building from Source

This guide covers building HPX and HPXPy from source.

## Prerequisites

### System Dependencies

- CMake 3.18+
- C++ compiler: GCC 9+, Clang 10+, or Apple Clang 12+
- Python 3.9+ with development headers
- Boost (HPX dependency)
- hwloc (HPX dependency, optional but recommended)

**macOS:**

```bash
brew install cmake gcc boost hwloc python
```

**Ubuntu/Debian:**

```bash
sudo apt install cmake g++ libboost-all-dev libhwloc-dev python3-dev python3-venv
```

### Python Environment

Set up a Python virtual environment and install dependencies:

```bash
python3 -m venv ~/.venv
source ~/.venv/bin/activate
pip install -r hpx/python/requirements.txt
```

```{important}
CMake discovers whichever Python is active at configure time. Make sure your
virtual environment is active **before** running `cmake` for both HPX and HPXPy.
```

---

## Step 1: Build and Install HPX

HPXPy requires HPX to be built and **installed** at a known prefix.

```bash
git clone https://github.com/STEllAR-GROUP/hpx.git
cd hpx
mkdir build && cd build

cmake .. \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15 \
    -DCMAKE_INSTALL_PREFIX=$HOME/usr/local/hpx \
    -DCMAKE_BUILD_TYPE=Release \
    -DHPX_WITH_CXX_STANDARD=20 \
    -DHPX_WITH_FETCH_ASIO=ON \
    -DHPX_WITH_FETCH_STDEXEC=ON \
    -DHPX_WITH_EXAMPLES=ON \
    -DHPX_WITH_TESTS=ON

cmake --build . -j8
cmake --install .
```

### HPX CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| `CMAKE_CXX_COMPILER` | C++ compiler (must match across HPX and HPXPy) | System default |
| `CMAKE_INSTALL_PREFIX` | Where to install HPX | `/usr/local` |
| `CMAKE_BUILD_TYPE` | `Release`, `Debug`, `RelWithDebInfo` | |
| `HPX_WITH_CXX_STANDARD` | C++ standard: `17`, `20`, `23` | `17` |
| `HPX_WITH_FETCH_ASIO` | Download Asio instead of using system version | `OFF` |
| `HPX_WITH_FETCH_STDEXEC` | Download stdexec (P2300 reference impl) | `OFF` |
| `HPX_WITH_EXAMPLES` | Build HPX examples | `ON` |
| `HPX_WITH_TESTS` | Build HPX test suite | `ON` |
| `HPX_WITH_CUDA` | Enable CUDA support | `OFF` |

After installation, HPX CMake config files will be at
`<prefix>/lib/cmake/HPX/HPXConfig.cmake`.

---

## Step 2: Build and Install HPXPy

### Configure

From the `hpx/python/` directory, point CMake at the HPX install prefix:

```bash
cd hpx/python
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/usr/local/hpx ..
```

HPXPy's CMake automatically:
- **Detects the compiler** HPX was built with (from `HPXConfig.cmake`)
- **Matches the C++ standard** HPX uses
- **Checks ABI compatibility** between the compiler and HPX

You should see output like:

```
-- Auto-detected HPX compiler: /opt/homebrew/bin/g++-15
-- Found HPX: 2.0.0 at /Users/you/usr/local/hpx/lib/cmake/HPX
-- C++ standard: 20 (from HPX)
-- Found Python: 3.12.11 at /Users/you/.venv/bin/python
-- Found NumPy: 2.2.6
-- Found pybind11: 2.13.6
```

### Build

```bash
cmake --build . -j8
```

### Install

Install hpxpy into the active Python environment's site-packages:

```bash
cmake --install .
```

This installs:
- `_core.so` (compiled C++ extension) → `<site-packages>/hpxpy/`
- `hpxpy/*.py` (Python wrapper) → `<site-packages>/hpxpy/`

The installed module has RPATH set to the HPX lib directory, so it finds HPX
shared libraries at runtime.

### Set Up Runtime Environment

CMake generates a `setup_env.sh` in the build directory. Source it so the HPX
shared libraries can be found at runtime:

```bash
source setup_env.sh
```

This sets `DYLD_LIBRARY_PATH` (macOS) or `LD_LIBRARY_PATH` (Linux) to include
the HPX lib directory. No `PYTHONPATH` manipulation is needed — after
`cmake --install`, Python finds hpxpy in site-packages automatically.

### Verify

```bash
python -c "import hpxpy as hpx; hpx.init(); print('HPXPy works!'); hpx.finalize()"
```

---

## Quick Reference

```bash
# 1. Build and install HPX
cd hpx && mkdir build && cd build
cmake .. \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15 \
    -DCMAKE_INSTALL_PREFIX=$HOME/usr/local/hpx \
    -DCMAKE_BUILD_TYPE=Release \
    -DHPX_WITH_CXX_STANDARD=20 \
    -DHPX_WITH_FETCH_ASIO=ON \
    -DHPX_WITH_FETCH_STDEXEC=ON
cmake --build . -j8
cmake --install .

# 2. Build and install HPXPy
cd hpx/python && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/usr/local/hpx ..
cmake --build . -j8
cmake --install .

# 3. Use
source setup_env.sh
python -c "import hpxpy as hpx; hpx.init(); print(hpx.sum(hpx.arange(100))); hpx.finalize()"
```

## Running Jupyter Notebooks

After building, installing, and sourcing the environment:

```bash
source build/setup_env.sh
cd tutorials    # or examples
jupyter lab
```

```{tip}
Launch Jupyter from a terminal where you've sourced `setup_env.sh` so it
inherits the HPX library path.
```

## GPU Support (Optional)

### CUDA

Requires HPX built with `HPX_WITH_CUDA=ON` and the CUDA toolkit:

```bash
cmake -DCMAKE_PREFIX_PATH=$HOME/usr/local/hpx -DHPXPY_WITH_CUDA=ON ..
```

### SYCL

Requires HPX built with `HPX_WITH_SYCL=ON` and a SYCL compiler:

```bash
cmake -DCMAKE_PREFIX_PATH=$HOME/usr/local/hpx -DHPXPY_WITH_SYCL=ON ..
```

## HPXPy CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| `CMAKE_PREFIX_PATH` | HPX install prefix | *(required)* |
| `HPXPY_DEFAULT_EXECUTION_POLICY` | Default execution policy (`seq`, `par`, `par_unseq`) | `par_unseq` |
| `HPXPY_WITH_CUDA` | Enable CUDA GPU support | `OFF` |
| `HPXPY_WITH_SYCL` | Enable SYCL GPU support | `OFF` |
| `HPXPY_WITH_TESTS` | Build test targets | `ON` |
| `HPXPY_INSTALL_PYTHONDIR` | Override Python install directory | auto-detected |

## Troubleshooting

### CMake finds wrong Python

If CMake picks up a Python that lacks NumPy or pybind11:

```
-- Found Python: 3.14.2 at /usr/bin/python3
```

**Fix:** Activate your virtual environment *before* running cmake:

```bash
source ~/.venv/bin/activate
which python    # verify
rm -rf build && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/usr/local/hpx ..
```

### "Module not found: hpxpy"

Make sure you ran `cmake --install .` and that the install went to your
active Python's site-packages. Check:

```bash
python -c "import sysconfig; print(sysconfig.get_path('platlib'))"
```

### "Library not found: libhpx"

Source the environment script:

```bash
source build/setup_env.sh
```

Or set the library path manually:

```bash
# macOS
export DYLD_LIBRARY_PATH=$HOME/usr/local/hpx/lib:$DYLD_LIBRARY_PATH

# Linux
export LD_LIBRARY_PATH=$HOME/usr/local/hpx/lib:$LD_LIBRARY_PATH
```

### CMake can't find HPX

Point CMake at the HPX install prefix:

```bash
cmake -DCMAKE_PREFIX_PATH=$HOME/usr/local/hpx ..
```

### ABI mismatch error

HPX and HPXPy must be built with the same compiler and standard library.
HPXPy auto-detects the compiler from HPX, but if you override it with
`-DCMAKE_CXX_COMPILER`, make sure it matches what HPX was built with.

### NumPy or pybind11 not found

Install them in your active environment:

```bash
pip install -r requirements.txt
```
