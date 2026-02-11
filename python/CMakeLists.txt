# Copyright (c) 2024-2025 Louisiana State University
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# ------------------------------------------------------------------------------
# Auto-detect compiler from HPX (before project() tests the compiler)
# ------------------------------------------------------------------------------
# If the user didn't specify a compiler, try to match the one HPX was built with.
# HPXConfig.cmake sets HPX_CXX_COMPILER, but it's only available after
# find_package(HPX). We do a lightweight pre-scan here.
if(NOT CMAKE_CXX_COMPILER)
  # Locate HPXConfig.cmake via CMAKE_PREFIX_PATH or HPX_ROOT
  foreach(_prefix ${CMAKE_PREFIX_PATH} $ENV{CMAKE_PREFIX_PATH} ${HPX_ROOT} $ENV{HPX_ROOT})
    set(_hpx_config "${_prefix}/lib/cmake/HPX/HPXConfig.cmake")
    if(EXISTS "${_hpx_config}")
      # HPX_CXX_COMPILER is set across multiple lines; scan for the path
      file(STRINGS "${_hpx_config}" _hpx_config_lines)
      set(_found_compiler_var FALSE)
      foreach(_line ${_hpx_config_lines})
        if(_line MATCHES "set\\(HPX_CXX_COMPILER")
          set(_found_compiler_var TRUE)
        elseif(_found_compiler_var AND _line MATCHES "\"([^\"]*)\"")
          set(_hpx_compiler "${CMAKE_MATCH_1}")
          if(EXISTS "${_hpx_compiler}")
            message(STATUS "Auto-detected HPX compiler: ${_hpx_compiler}")
            set(CMAKE_CXX_COMPILER "${_hpx_compiler}")
          endif()
          break()
        endif()
      endforeach()
      break()
    endif()
  endforeach()
endif()

project(hpxpy VERSION 0.1.0 LANGUAGES CXX)

# ------------------------------------------------------------------------------
# C++ Standard — must match HPX (queried after find_package(HPX))
# Minimum is C++17; HPX may require a higher standard.
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Build type
# ------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
option(HPXPY_WITH_TESTS "Build tests" ON)
option(HPXPY_WITH_BENCHMARKS "Build benchmarks" ON)
option(HPXPY_WITH_CUDA "Enable CUDA support" OFF)
option(HPXPY_WITH_SYCL "Enable SYCL support" OFF)
option(HPXPY_WITH_NATIVE_ARCH "Enable -march=native -mtune=native for CPU-specific optimizations" OFF)

set(HPXPY_DEFAULT_EXECUTION_POLICY "par_unseq" CACHE STRING
    "Default execution policy for HPXPy operations (seq, par, par_unseq)")
set_property(CACHE HPXPY_DEFAULT_EXECUTION_POLICY PROPERTY STRINGS seq par par_unseq)

# ------------------------------------------------------------------------------
# Native architecture optimization
# ------------------------------------------------------------------------------
if(HPXPY_WITH_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" HPXPY_HAS_MARCH_NATIVE)
  check_cxx_compiler_flag("-mtune=native" HPXPY_HAS_MTUNE_NATIVE)

  if(HPXPY_HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
    message(STATUS "Enabled -march=native for CPU-specific SIMD instructions")
  endif()
  if(HPXPY_HAS_MTUNE_NATIVE)
    add_compile_options(-mtune=native)
    message(STATUS "Enabled -mtune=native for CPU-specific tuning")
  endif()
endif()

# ------------------------------------------------------------------------------
# Standard install directories
# ------------------------------------------------------------------------------
include(GNUInstallDirs)

# ------------------------------------------------------------------------------
# Find dependencies
# ------------------------------------------------------------------------------

# Find HPX — set CMAKE_PREFIX_PATH or HPX_ROOT to the HPX install prefix
find_package(HPX QUIET)
if(NOT HPX_FOUND)
  message(FATAL_ERROR
    "HPX not found. Point CMake at the HPX install prefix:\n"
    "  cmake -DCMAKE_PREFIX_PATH=/path/to/hpx ..\n"
    "  cmake -DHPX_ROOT=/path/to/hpx ..\n"
    "See the building guide for details.")
endif()
message(STATUS "Found HPX: ${HPX_VERSION} at ${HPX_DIR}")

# Match the C++ standard HPX was built with (at least C++17)
if(DEFINED HPX_WITH_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD ${HPX_WITH_CXX_STANDARD})
else()
  set(CMAKE_CXX_STANDARD 17)
endif()
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD} (from HPX)")

# Verify compiler ABI compatibility with HPX
# HPX and HPXPy must use the same C++ standard library (libstdc++ vs libc++).
# Linking a libc++ module against libstdc++ HPX (or vice versa) will cause
# symbol resolution failures at runtime.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${HPX_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES HPX::hpx)
check_cxx_source_compiles("
  #include <hpx/version.hpp>
  int main() { return hpx::full_version_as_string().empty(); }
" HPXPY_ABI_COMPATIBLE)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_LIBRARIES)

if(NOT HPXPY_ABI_COMPATIBLE)
  message(FATAL_ERROR
    "Compiler ABI mismatch: cannot link against the installed HPX.\n"
    "HPX and HPXPy must be built with the same compiler and standard library.\n"
    "If HPX was built with GCC, use: cmake -DCMAKE_CXX_COMPILER=g++ ...\n"
    "If HPX was built with Clang, use: cmake -DCMAKE_CXX_COMPILER=clang++ ...\n")
endif()

# Derive HPX lib directory and install prefix from HPX_DIR
# HPX_DIR is set by find_package to <prefix>/lib/cmake/HPX
get_filename_component(_hpx_lib_dir "${HPX_DIR}/../.." ABSOLUTE)
get_filename_component(HPX_PREFIX "${_hpx_lib_dir}/.." ABSOLUTE)
message(STATUS "HPX prefix: ${HPX_PREFIX}")
message(STATUS "HPX lib dir: ${_hpx_lib_dir}")

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
message(STATUS "Found Python: ${Python_VERSION} at ${Python_EXECUTABLE}")
message(STATUS "Found NumPy: ${Python_NumPy_VERSION} at ${Python_NumPy_INCLUDE_DIRS}")

# Find pybind11 - try config mode first, then module mode
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  # Try to find via Python
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_result
  )
  if(pybind11_result EQUAL 0)
    find_package(pybind11 CONFIG REQUIRED HINTS ${pybind11_DIR})
  else()
    message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
  endif()
endif()
message(STATUS "Found pybind11: ${pybind11_VERSION}")

# ------------------------------------------------------------------------------
# Workaround: Apple Clang libc++ std::execution policy detection
# ------------------------------------------------------------------------------
# HPX's installed defines.hpp assumes std::execution policies exist if
# __cplusplus >= 201500, but Apple's libc++ does not provide them.
# We detect this at configure time and generate a force-included header
# that undefs the broken macros.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")
check_cxx_source_compiles("
  #include <execution>
  int main() { [[maybe_unused]] auto p = std::execution::seq; return 0; }
" HPXPY_HAVE_STD_EXECUTION_POLICIES)

if(NOT HPXPY_HAVE_STD_EXECUTION_POLICIES)
  message(STATUS "std::execution policies NOT available — applying workaround")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/hpxpy_compat.hpp"
    "// Auto-generated by CMake — workaround for missing std::execution policies\n"
    "#include <hpx/config.hpp>\n"
    "#undef HPX_HAVE_CXX17_STD_EXECUTION_POLICES\n"
    "#undef HPX_HAVE_CXX20_STD_EXECUTION_POLICES\n"
  )
  set(HPXPY_COMPAT_INCLUDE "-include${CMAKE_CURRENT_BINARY_DIR}/hpxpy_compat.hpp")
else()
  message(STATUS "std::execution policies available")
  set(HPXPY_COMPAT_INCLUDE "")
endif()

# ------------------------------------------------------------------------------
# RPATH configuration
# ------------------------------------------------------------------------------
# Build tree: use RPATH from the build (for development)
# Install tree: set RPATH to HPX lib directory so the module finds HPX at runtime
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${_hpx_lib_dir}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()

# ------------------------------------------------------------------------------
# HPXPy Core Extension Module
# ------------------------------------------------------------------------------
pybind11_add_module(_core MODULE
  src/bindings/core_module.cpp
  src/bindings/runtime_bindings.cpp
  src/bindings/array_bindings.cpp
  src/bindings/algorithm_bindings.cpp
  src/bindings/collective_bindings.cpp
  src/bindings/distributed_array_bindings.cpp
  src/bindings/gpu_bindings.cpp
  src/bindings/sycl_bindings.cpp
)

target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${Python_NumPy_INCLUDE_DIRS}
)

target_link_libraries(_core PRIVATE
  HPX::hpx
  HPX::wrap_main
  HPX::partitioned_vector_component
)

target_compile_definitions(_core PRIVATE
  HPXPY_VERSION="${PROJECT_VERSION}"
  HPXPY_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  HPXPY_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  HPXPY_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  HPXPY_DEFAULT_POLICY="${HPXPY_DEFAULT_EXECUTION_POLICY}"
)

# Apply std::execution workaround if needed
if(HPXPY_COMPAT_INCLUDE)
  target_compile_options(_core PRIVATE "${HPXPY_COMPAT_INCLUDE}")
endif()

# Enable position independent code
set_target_properties(_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# Copy built module to hpxpy package directory for development
add_custom_command(TARGET _core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:_core>
    ${CMAKE_SOURCE_DIR}/hpxpy/$<TARGET_FILE_NAME:_core>
  COMMENT "Copying _core module to hpxpy/ for development"
)

# Compiler warnings and optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic)

  # Enable SIMD vectorization for Release builds
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # GCC on ARM uses -mcpu=native; Clang/x86 uses -march=native
    # NOTE: -ffast-math is intentionally omitted — it causes hpx::reduce(par)
    # to produce wrong results due to interaction with HPX's header-only
    # algorithm implementation (ODR violation: HPX compiled without -ffast-math,
    # but template instantiations here compiled with it).
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
      target_compile_options(_core PRIVATE -mcpu=native -funroll-loops)
    else()
      target_compile_options(_core PRIVATE -march=native -mtune=native -funroll-loops)
    endif()
    message(STATUS "SIMD optimizations enabled for ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
endif()

# ------------------------------------------------------------------------------
# GPU Support (Optional)
# ------------------------------------------------------------------------------
if(HPXPY_WITH_CUDA)
  # Require HPX built with CUDA support for async_cuda module
  find_package(HPX REQUIRED COMPONENTS cuda)
  find_package(CUDAToolkit REQUIRED)

  target_compile_definitions(_core PRIVATE HPXPY_HAVE_CUDA)

  # Link CUDA runtime for memory allocation and device management
  target_link_libraries(_core PRIVATE CUDA::cudart)

  # Add cuBLAS for GPU BLAS operations (optional but recommended)
  if(TARGET CUDA::cublas)
    target_link_libraries(_core PRIVATE CUDA::cublas)
    target_compile_definitions(_core PRIVATE HPXPY_HAVE_CUBLAS)
    message(STATUS "CUDA cuBLAS support enabled")
  endif()

  message(STATUS "CUDA support enabled (HPX async_cuda integration)")
endif()

if(HPXPY_WITH_SYCL)
  # Require HPX built with SYCL support for async_sycl module
  find_package(HPX REQUIRED COMPONENTS sycl)

  target_compile_definitions(_core PRIVATE HPXPY_HAVE_SYCL)

  message(STATUS "SYCL support enabled (HPX async_sycl integration)")
endif()

# ------------------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------------------
# Detect Python site-packages directory (user can override with
# -DHPXPY_INSTALL_PYTHONDIR=...)
if(NOT DEFINED HPXPY_INSTALL_PYTHONDIR)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c
      "import sysconfig; print(sysconfig.get_path('platlib'))"
    OUTPUT_VARIABLE HPXPY_INSTALL_PYTHONDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()
message(STATUS "Python install dir: ${HPXPY_INSTALL_PYTHONDIR}")

# Install the extension module
install(TARGETS _core
  LIBRARY DESTINATION "${HPXPY_INSTALL_PYTHONDIR}/hpxpy"
  RUNTIME DESTINATION "${HPXPY_INSTALL_PYTHONDIR}/hpxpy"
)

# Install Python package files
install(DIRECTORY hpxpy/
  DESTINATION "${HPXPY_INSTALL_PYTHONDIR}/hpxpy"
  FILES_MATCHING PATTERN "*.py"
)

# ------------------------------------------------------------------------------
# Generate setup_env.sh for development builds
# ------------------------------------------------------------------------------
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/setup_env.sh.in"
  "${CMAKE_CURRENT_BINARY_DIR}/setup_env.sh"
  @ONLY
)

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------
if(HPXPY_WITH_TESTS)
  enable_testing()

  # Add a test that runs pytest
  add_test(
    NAME pytest
    COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  # Make sure the extension module can be found
  set_tests_properties(pytest PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "HPXPy ${PROJECT_VERSION} Configuration Summary")
message(STATUS "================================")
message(STATUS "  HPX version:      ${HPX_VERSION}")
message(STATUS "  HPX prefix:       ${HPX_PREFIX}")
message(STATUS "  HPX lib dir:      ${_hpx_lib_dir}")
message(STATUS "  Python version:   ${Python_VERSION}")
message(STATUS "  NumPy version:    ${Python_NumPy_VERSION}")
message(STATUS "  pybind11 version: ${pybind11_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA support:     ${HPXPY_WITH_CUDA}")
message(STATUS "  SYCL support:     ${HPXPY_WITH_SYCL}")
message(STATUS "  Build tests:      ${HPXPY_WITH_TESTS}")
message(STATUS "  Default policy:   ${HPXPY_DEFAULT_EXECUTION_POLICY}")
message(STATUS "  Install dir:      ${HPXPY_INSTALL_PYTHONDIR}/hpxpy")
message(STATUS "  Install RPATH:    ${CMAKE_INSTALL_RPATH}")
message(STATUS "")
