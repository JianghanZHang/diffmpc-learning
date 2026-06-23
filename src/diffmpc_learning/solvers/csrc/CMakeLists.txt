cmake_minimum_required(VERSION 3.20)

# ── GPU architecture detection ───────────────────────────────────
# Must happen BEFORE project() — cmake sets CMAKE_CUDA_ARCHITECTURES
# to a useless default during enable_language(CUDA) otherwise.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    # cmake < 3.24: auto-detect via nvidia-smi
    execute_process(
      COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _GPU_CC OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _GPU_CC_RESULT
    )
    if(_GPU_CC_RESULT EQUAL 0 AND _GPU_CC)
      # "8.9" → "89", "12.0" → "120"
      string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _MATCH "${_GPU_CC}")
      set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
    else()
      # Fallback: build for all common GPUs (Turing through Blackwell)
      set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90;120")
    endif()
  endif()
endif()

project(diffmpc_learning_ffi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import os, jaxlib; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
  OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "XLA FFI include: ${JAXLIB_INCLUDE_DIR}")

find_package(CUDAToolkit REQUIRED)

# ── pip nvidia cuDSS auto-detect ─────────────────────────────────
# Auto-detect cuDSS from pip nvidia-cudss package (same logic as turbompc).
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import nvidia.cu12; print(list(nvidia.cu12.__path__)[0])"
  OUTPUT_VARIABLE NVIDIA_CU12_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE _nvidia_cu12_rc
)
if(_nvidia_cu12_rc EQUAL 0)
  message(STATUS "nvidia.cu12 package: ${NVIDIA_CU12_DIR}")
endif()
# Prefer pip nvidia-cudss-cu12 (libcublas-compatible) over system .deb cuDSS.
find_library(CUDSS_LIB cudss HINTS
  ${CUDSS_DIR}/lib
  $ENV{CUDSS_DIR}/lib
  ${NVIDIA_CU12_DIR}/lib
  NO_DEFAULT_PATH
)
if(NOT CUDSS_LIB)
  # pip nvidia-cudss-cu12 ships libcudss.so.0 without an unversioned symlink
  file(GLOB _CUDSS_SO0
    "${NVIDIA_CU12_DIR}/lib/libcudss.so.*"
    "${CUDSS_DIR}/lib/libcudss.so.*"
    "$ENV{CUDSS_DIR}/lib/libcudss.so.*"
  )
  if(_CUDSS_SO0)
    list(GET _CUDSS_SO0 0 CUDSS_LIB)
    message(STATUS "cuDSS found (versioned, pip): ${CUDSS_LIB}")
  endif()
endif()
if(NOT CUDSS_LIB)
  # Last resort: system paths (may have incompatible libcublas version)
  find_library(CUDSS_LIB cudss HINTS
    ${CUDAToolkit_LIBRARY_DIR}
    /usr/local/cuda/lib64
  )
endif()
if(CUDSS_LIB)
  message(STATUS "cuDSS found: ${CUDSS_LIB}")
else()
  message(WARNING "cuDSS not found — no cuDSS targets will be built. "
    "Install cuDSS (https://developer.nvidia.com/cudss) and set -DCUDSS_DIR=<path> to enable.")
endif()

# ── TODO: add central-path CUDA targets here ─────────────────────
# Mirror turbompc's pcg_blktridi / cudss_blktridi pattern:
#
#   add_library(central_path_ffi SHARED
#     csrc/central_path.cu
#     csrc/central_path_ffi.cc
#   )
#   target_link_libraries(central_path_ffi PRIVATE CUDA::cudart CUDA::cusparse ${CUDSS_LIB})
#   ...
#
# No build targets yet — this file is a scaffold for future CUDA kernels.
