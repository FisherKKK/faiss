# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About Faiss

Faiss is a library for efficient similarity search and clustering of dense vectors developed by Meta's Fundamental AI Research group. It contains algorithms for searching in sets of vectors of any size, supporting L2 (Euclidean) distances, dot products, and cosine similarity. Faiss includes CPU implementations in C++ with Python bindings, and optional GPU implementations via CUDA, ROCm, and NVIDIA cuVS.

## Build System

Faiss uses CMake (minimum version 3.24.0) as its build system.

### Initial Build Setup

```bash
# Configure the build
cmake -B build .

# Build the C++ library
make -C build -j faiss

# Build with specific optimization level (avx2, avx512, avx512_spr on x86-64; sve on aarch64)
make -C build -j faiss_avx2
```

### Common CMake Options

- `-DFAISS_ENABLE_GPU=OFF` - Disable GPU indices (default: ON)
- `-DFAISS_ENABLE_PYTHON=OFF` - Disable Python bindings (default: ON)
- `-DFAISS_ENABLE_CUVS=ON` - Enable NVIDIA cuVS implementations (default: OFF, requires GPU enabled)
- `-DFAISS_ENABLE_ROCM=ON` - Enable AMD ROCm for GPU (default: OFF)
- `-DFAISS_ENABLE_SVS=ON` - Enable Intel SVS integration (default: OFF)
- `-DBUILD_TESTING=OFF` - Disable building C++ tests (default: ON)
- `-DBUILD_SHARED_LIBS=ON` - Build shared library instead of static (default: OFF)
- `-DFAISS_ENABLE_C_API=ON` - Enable C API (default: OFF)
- `-DCMAKE_BUILD_TYPE=Release` - Enable compiler optimizations
- `-DFAISS_OPT_LEVEL=avx2` - Set SIMD optimization level

### Python Bindings

```bash
# Build Python bindings
make -C build -j swigfaiss

# Install Python package
cd build/faiss/python && python setup.py install
```

### Testing

```bash
# Run C++ test suite
make -C build test

# Run a specific C++ test
./build/tests/faiss_test --gtest_filter=TestName.*

# Run Python tests (after building Python bindings)
cd build/faiss/python && python setup.py build
PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_*.py

# Run a specific Python test
PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_index_composite.py -v
```

### Running Demos

```bash
# Build and run basic CPU demo
make -C build demo_ivfpq_indexing
./build/demos/demo_ivfpq_indexing

# Build and run GPU demo
make -C build demo_ivfpq_indexing_gpu
./build/demos/demo_ivfpq_indexing_gpu
```

## Code Architecture

### Core Directory Structure

- `faiss/` - Main C++ library source code
  - `Index*.{h,cpp}` - Index implementations (root level)
  - `impl/` - Core data structures and algorithms
    - Quantizers: `AdditiveQuantizer`, `ProductQuantizer`, `ResidualQuantizer`, `ScalarQuantizer`, `RaBitQuantizer`
    - Graph structures: `HNSW`, `NSG`, `NNDescent`
    - Auxiliary structures: `AuxIndexStructures`, `IDSelector`, `DistanceComputer`
    - I/O operations: `index_read.cpp`, `index_write.cpp`, `io.cpp`
  - `utils/` - Utility functions
    - Distance computation: `distances.cpp`, `distances_simd.cpp`, `extra_distances.cpp`
    - SIMD operations: `simdlib*.h` (AVX2, AVX512, NEON, etc.)
    - Data structures: `Heap`, `AlignedTable`
    - Other utilities: `hamming.cpp`, `partitioning.cpp`, `random.cpp`
  - `invlists/` - Inverted list implementations (InvertedLists, BlockInvertedLists, OnDiskInvertedLists)
  - `gpu/` - GPU implementations (CUDA/ROCm)
  - `python/` - Python binding code (SWIG-based)
  - `cppcontrib/` - Contributed C++ utilities
  - `svs/` - Intel SVS integration
- `tests/` - C++ and Python test files (test_*.cpp, test_*.py)
- `benchs/` - Benchmark code
- `demos/` - Example programs
- `c_api/` - C API wrapper
- `tutorial/cpp/` - C++ tutorials

### Key Abstractions

**Index Hierarchy**: The `Index` class (faiss/Index.h) is the base class for all index types. Key index families include:
- Flat indices: `IndexFlat`, `IndexBinaryFlat` - Exact search using brute force
- IVF (Inverted File) indices: `IndexIVF*` - Partitioned indices with coarse quantizer
- PQ (Product Quantization): `IndexPQ`, `IndexIVFPQ` - Compressed vector representations
- HNSW: `IndexHNSW` - Hierarchical Navigable Small World graphs
- Scalar Quantization: `IndexScalarQuantizer` - Scalar quantization of vector components
- FastScan: `IndexFastScan`, `IndexIVFFastScan` - Optimized implementations using SIMD
- Binary indices: `IndexBinary*` - Indices for binary vectors (Hamming distance)
- Meta indices: `IndexPreTransform`, `IndexRefine`, `IndexShards`, `IndexReplicas` - Wrappers for composing indices

**Index Factory**: The `index_factory()` function (faiss/index_factory.cpp) creates indices from string descriptions using regex parsing. This is a high-level API for quickly instantiating complex index configurations.

**Quantizers**: Quantizers reduce memory footprint by encoding vectors compactly:
- `AdditiveQuantizer` (faiss/impl/AdditiveQuantizer.h) - Base for quantizers that sum codebook entries
- Product Quantizer - Splits vectors into subvectors and quantizes independently
- Residual Quantizer - Iteratively quantizes residuals

**Inverted Lists**: IVF-based indices use inverted lists to store vectors by partition (faiss/invlists/InvertedLists.h). Implementations include in-memory, block-based, and on-disk variants.

**Distance Computation**: Distance functions are in `faiss/utils/distances.{h,cpp}` with SIMD-optimized implementations in `faiss/utils/distances_simd.cpp`.

**GPU Implementation**: GPU indices in `faiss/gpu/` mirror CPU index structure. The `GpuCloner` class converts between CPU and GPU indices. GPU implementations support both NVIDIA (CUDA/cuVS) and AMD (ROCm) hardware.

### Important Types

- `idx_t` (defined in MetricType.h as `int64_t`) - Vector/index ID type used throughout the codebase
- `MetricType` - Enum for distance metrics (METRIC_L2, METRIC_INNER_PRODUCT, METRIC_L1, etc.)
- `SearchParameters` - Base class for search-time parameters (e.g., filtering with IDSelector)

### Vector Representation

Vectors are provided as `float*` pointers in row-major storage. When n vectors of size d are passed as `float* x`, component j of vector i is accessed as `x[i * d + j]` where 0 <= i < n and 0 <= j < d.

### Code Organization Patterns

- Index classes define virtual methods for `add()`, `search()`, `train()`, `reset()`
- Implementation details for complex algorithms live in `faiss/impl/` (e.g., HNSW graph implementation)
- SIMD-optimized code uses preprocessor macros and template specialization
- GPU code uses `.cu` extension for CUDA and is "hipified" to `.hip.cpp` for ROCm

## Coding Standards

- C++ language level: C++17 (C++20 when not using cuVS)
- Indentation: 4 spaces (no tabs)
- Line length: 80 characters for both C++ and Python
- Use `faiss::idx_t` instead of `long` for compatibility across platforms
- All code is in the `faiss` namespace
- Code formatting: Use `clang-format` with the `.clang-format` configuration file in the repository root
  - Format a file: `clang-format -i <file>`
  - The configuration enforces project style: 4-space indentation, 80-column limit, left pointer alignment

## Testing Conventions

- C++ tests use Google Test framework and are discovered automatically by CMake
- Python tests use pytest
- Test files follow naming convention: `test_*.cpp` or `test_*.py`
- C++ test files are listed in `tests/CMakeLists.txt`
- Some tests require GPU hardware and will be skipped if not available

## Python Integration

- Python bindings are generated using SWIG
- Python API mirrors C++ API closely with snake_case naming
- Vectors are passed as numpy arrays
- The python package installation copies necessary libraries (including libsvs_runtime.so for SVS)

## GPU Development Notes

- When GPU support is enabled, use `-DCUDAToolkit_ROOT=/path/to/cuda` to specify CUDA location
- Use `-DCMAKE_CUDA_ARCHITECTURES="75;72"` to specify target GPU architectures
- For ROCm: set `-DFAISS_ENABLE_ROCM=ON` and `-DFAISS_ENABLE_GPU=ON`
- GPU code in `faiss/gpu/` is preprocessed by `hipify.sh` script for ROCm compatibility
- GPU indices can be drop-in replacements for CPU indices (e.g., `GpuIndexFlatL2` vs `IndexFlatL2`)

## Common Development Patterns

When adding new index types:
1. Inherit from appropriate base class (Index, IndexBinary, IndexIVF, etc.)
2. Implement required virtual methods: add(), search(), reset(), train() if needed
3. Add factory string support in index_factory.cpp if appropriate
4. Add corresponding tests in tests/
5. Update GPU implementation in faiss/gpu/ if GPU support is needed

When optimizing performance:
- Consider SIMD implementations in utils/ directory
- Look at FastScan variants for scan-optimized approaches
- Profile with benchs/ tools before optimizing

## Building for Development

For iterative development, build specific targets:
```bash
# Build only the core library
make -C build -j faiss

# Build specific test executable
make -C build -j faiss_test

# Build specific demo
make -C build demo_sift1M

# Build specific benchmark
make -C build -j bench_gpu_1bn
```

The `-j` flag enables parallel compilation but may cause out-of-memory issues on resource-constrained systems. Use `-j4` to limit parallelism.

## Useful Development Commands

```bash
# Format code with clang-format
clang-format -i faiss/IndexIVFPQ.cpp

# Check if build directory is already configured
ls build/CMakeCache.txt

# Clean and rebuild
rm -rf build && cmake -B build . && make -C build -j faiss

# View available CMake targets
cmake --build build --target help
```
