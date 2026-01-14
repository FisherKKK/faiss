# 性能优化实战练习手册

## 从零开始的性能优化训练营

> 本手册提供循序渐进的实战练习，每个练习都可以独立完成，涵盖CPU、内存、IO、并发等各个方面。

---

## 目录

1. [环境准备](#1-环境准备)
2. [初级练习 - 基础性能分析](#2-初级练习)
3. [中级练习 - 深度优化](#3-中级练习)
4. [高级练习 - 系统级优化](#4-高级练习)
5. [挑战练习 - 综合项目](#5-挑战练习)
6. [性能优化竞赛题目](#6-性能优化竞赛题目)

---

## 1. 环境准备

### 1.1 系统要求

```bash
# 推荐配置
操作系统: Ubuntu 22.04 / CentOS 8 / Debian 11
CPU: 4核心以上 (支持AVX2)
内存: 8GB以上
磁盘: 50GB可用空间
```

### 1.2 工具安装

```bash
# 更新系统
sudo apt-get update

# 安装编译工具
sudo apt-get install -y build-essential cmake git

# 安装性能分析工具
sudo apt-get install -y \
    linux-tools-common \
    linux-tools-generic \
    linux-tools-`uname -r` \
    valgrind \
    sysstat \
    iotop \
    htop \
    bpftrace

# 安装开发库
sudo apt-get install -y \
    libnuma-dev \
    libjemalloc-dev \
    libgoogle-perftools-dev \
    liburing-dev

# 验证perf安装
perf --version

# 如果perf不可用，允许非root用户使用perf
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid

# 安装Python工具
pip3 install numpy scipy matplotlib psutil
```

### 1.3 创建工作目录

```bash
mkdir -p ~/performance-workshop
cd ~/performance-workshop

# 创建子目录
mkdir -p {01-basic,02-intermediate,03-advanced,04-challenge}
```

---

## 2. 初级练习

### 练习 1.1: 第一次性能剖析

**目标**: 学会使用perf找到程序热点

**步骤**:

```cpp
// 01-basic/slow_program.cpp
#include <iostream>
#include <vector>
#include <cmath>

// 故意写的低效代码
double slow_computation(int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum += std::sqrt(i * i + j * j);  // 热点!
        }
    }
    return sum;
}

double medium_computation(int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += std::sin(i) * std::cos(i);
    }
    return sum;
}

double fast_computation(int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += i;
    }
    return sum;
}

int main() {
    const int N = 5000;

    std::cout << "Computing..." << std::endl;

    double result1 = slow_computation(N);
    double result2 = medium_computation(N * 100);
    double result3 = fast_computation(N * 1000);

    std::cout << "Results: " << result1 << ", "
              << result2 << ", " << result3 << std::endl;

    return 0;
}
```

**编译和运行**:

```bash
cd 01-basic

# 使用-O0编译 (无优化)
g++ -O0 -g slow_program.cpp -o slow_program

# 运行程序
./slow_program

# 使用perf分析
perf record -g ./slow_program
perf report

# 查看火焰图
perf script | ~/FlameGraph/stackcollapse-perf.pl | \
    ~/FlameGraph/flamegraph.pl > flamegraph.svg
```

**任务**:

1. ✅ 找出哪个函数占用CPU时间最多
2. ✅ 记录热点函数的百分比
3. ✅ 识别热点代码行 (使用 perf annotate)

**预期结果**:

```
函数                    CPU占比
─────────────────────────────
slow_computation        ~75%
medium_computation      ~20%
fast_computation        ~2%
```

**优化任务**:

```cpp
// 优化slow_computation
double optimized_computation(int n) {
    double sum = 0;

    // 优化1: 提取sqrt
    for (int i = 0; i < n; i++) {
        double i_sq = i * i;
        for (int j = 0; j < n; j++) {
            sum += std::sqrt(i_sq + j * j);
        }
    }

    return sum;
}

// 进一步优化
double optimized_computation_v2(int n) {
    double sum = 0;

    // 利用对称性
    for (int i = 0; i < n; i++) {
        double i_sq = i * i;
        sum += std::sqrt(i_sq);  // j=0

        for (int j = 1; j < n; j++) {
            sum += 2 * std::sqrt(i_sq + j * j);  // 计算一次，乘2
        }
    }

    return sum;
}
```

**验证优化**:

```bash
# 编译优化版本
g++ -O0 -g optimized_program.cpp -o optimized_program

# 对比性能
time ./slow_program
time ./optimized_program

# 使用perf对比
perf stat -e cycles,instructions ./slow_program
perf stat -e cycles,instructions ./optimized_program
```

---

### 练习 1.2: 缓存友好的数据访问

**目标**: 理解缓存局部性的重要性

**代码**:

```cpp
// 01-basic/cache_test.cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>

const int SIZE = 1024;
const int ITERATIONS = 10000;

// 行优先访问 (缓存友好)
double row_major_sum(int matrix[SIZE][SIZE]) {
    double sum = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            sum += matrix[i][j];  // 连续访问
        }
    }
    return sum;
}

// 列优先访问 (缓存不友好)
double col_major_sum(int matrix[SIZE][SIZE]) {
    double sum = 0;
    for (int j = 0; j < SIZE; j++) {
        for (int i = 0; i < SIZE; i++) {
            sum += matrix[i][j];  // 跳跃访问
        }
    }
    return sum;
}

int main() {
    // 分配并初始化矩阵
    int (*matrix)[SIZE] = new int[SIZE][SIZE];
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = i + j;
        }
    }

    // 测试行优先
    auto start = std::chrono::high_resolution_clock::now();
    double sum1 = 0;
    for (int k = 0; k < ITERATIONS; k++) {
        sum1 += row_major_sum(matrix);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto row_time = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    // 测试列优先
    start = std::chrono::high_resolution_clock::now();
    double sum2 = 0;
    for (int k = 0; k < ITERATIONS; k++) {
        sum2 += col_major_sum(matrix);
    }
    end = std::chrono::high_resolution_clock::now();
    auto col_time = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    std::cout << "Row-major time: " << row_time << " ms" << std::endl;
    std::cout << "Col-major time: " << col_time << " ms" << std::endl;
    std::cout << "Slowdown: " << (double)col_time / row_time << "x" << std::endl;

    delete[] matrix;
    return 0;
}
```

**编译运行**:

```bash
g++ -O2 cache_test.cpp -o cache_test
./cache_test

# 使用perf查看缓存miss
perf stat -e cache-references,cache-misses,\
    L1-dcache-loads,L1-dcache-load-misses \
    ./cache_test
```

**任务**:

1. ✅ 对比两种访问模式的性能差异
2. ✅ 使用perf测量缓存miss率
3. ✅ 计算性能差异倍数

**预期结果**:

```
访问模式      时间      L1 Cache Miss率
────────────────────────────────────────
行优先        ~1500ms   2-3%
列优先        ~8000ms   25-30%  (5-6x slower)
```

**扩展练习**:

```cpp
// 实现分块访问优化列优先
double blocked_col_major_sum(int matrix[SIZE][SIZE]) {
    const int BLOCK = 64;  // 块大小
    double sum = 0;

    for (int jj = 0; jj < SIZE; jj += BLOCK) {
        for (int ii = 0; ii < SIZE; ii += BLOCK) {
            // 块内访问
            for (int j = jj; j < std::min(jj + BLOCK, SIZE); j++) {
                for (int i = ii; i < std::min(ii + BLOCK, SIZE); i++) {
                    sum += matrix[i][j];
                }
            }
        }
    }

    return sum;
}

// 目标: 将列优先访问优化到接近行优先的性能
```

---

### 练习 1.3: 编译器优化实验

**目标**: 理解编译器优化的威力

**代码**:

```cpp
// 01-basic/compiler_opt.cpp
#include <iostream>
#include <chrono>
#include <vector>

const int N = 10000000;

// 简单的循环
long long simple_loop() {
    long long sum = 0;
    for (int i = 0; i < N; i++) {
        sum += i;
    }
    return sum;
}

// 有依赖的循环
long long dependent_loop() {
    long long sum = 0;
    for (int i = 0; i < N; i++) {
        sum = sum + i;  // 依赖前一次的sum
    }
    return sum;
}

// 可向量化的循环
void vectorizable_loop(int* a, int* b, int* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    long long result = simple_loop();
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Result: " << result << std::endl;
    std::cout << "Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                  end - start).count()
              << " us" << std::endl;

    return 0;
}
```

**实验步骤**:

```bash
# 测试不同优化级别
for opt in O0 O1 O2 O3 Ofast; do
    echo "=== Testing -$opt ==="
    g++ -$opt compiler_opt.cpp -o compiler_opt_$opt
    time ./compiler_opt_$opt
done

# 查看生成的汇编代码
g++ -O0 -S compiler_opt.cpp -o compiler_opt_O0.s
g++ -O3 -S compiler_opt.cpp -o compiler_opt_O3.s

# 对比汇编代码
diff -u compiler_opt_O0.s compiler_opt_O3.s | less

# 查看向量化报告
g++ -O3 -march=native -fopt-info-vec-optimized \
    compiler_opt.cpp -o compiler_opt_vec
```

**任务**:

1. ✅ 记录不同优化级别的性能差异
2. ✅ 识别哪些优化被应用了
3. ✅ 查看是否启用了向量化

**预期结果表格**:

```
优化级别    时间(ms)    加速比    说明
────────────────────────────────────────
-O0         ~25         1.0x      无优化
-O1         ~12         2.1x      基础优化
-O2         ~8          3.1x      推荐级别
-O3         ~3          8.3x      激进优化
-Ofast      ~2          12.5x     快速数学
```

---

### 练习 1.4: 分支预测实验

**目标**: 理解分支预测的影响

**代码**:

```cpp
// 01-basic/branch_test.cpp
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <vector>

const int SIZE = 100000000;

// 有分支的求和
long long sum_with_branch(const std::vector<int>& data) {
    long long sum = 0;
    for (int val : data) {
        if (val >= 128) {  // 分支
            sum += val;
        }
    }
    return sum;
}

// 无分支的求和
long long sum_without_branch(const std::vector<int>& data) {
    long long sum = 0;
    for (int val : data) {
        // 使用位运算消除分支
        int mask = -(val >= 128);  // 全1或全0
        sum += val & mask;
    }
    return sum;
}

int main() {
    std::vector<int> data_random(SIZE);
    std::vector<int> data_sorted(SIZE);

    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (int i = 0; i < SIZE; i++) {
        data_random[i] = dis(gen);
        data_sorted[i] = data_random[i];
    }

    // 排序一份
    std::sort(data_sorted.begin(), data_sorted.end());

    // 测试1: 随机数据 + 有分支
    auto start = std::chrono::high_resolution_clock::now();
    long long sum1 = sum_with_branch(data_random);
    auto end = std::chrono::high_resolution_clock::now();
    auto time1 = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    // 测试2: 排序数据 + 有分支
    start = std::chrono::high_resolution_clock::now();
    long long sum2 = sum_with_branch(data_sorted);
    end = std::chrono::high_resolution_clock::now();
    auto time2 = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    // 测试3: 随机数据 + 无分支
    start = std::chrono::high_resolution_clock::now();
    long long sum3 = sum_without_branch(data_random);
    end = std::chrono::high_resolution_clock::now();
    auto time3 = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    std::cout << "Random + Branch:   " << time1 << " ms" << std::endl;
    std::cout << "Sorted + Branch:   " << time2 << " ms" << std::endl;
    std::cout << "Random + No Branch:" << time3 << " ms" << std::endl;

    return 0;
}
```

**运行实验**:

```bash
g++ -O2 branch_test.cpp -o branch_test

# 运行程序
./branch_test

# 使用perf测量分支miss
perf stat -e branches,branch-misses ./branch_test
```

**任务**:

1. ✅ 对比随机vs排序数据的分支预测性能
2. ✅ 对比有分支vs无分支的实现
3. ✅ 测量branch miss率

**预期结果**:

```
场景                    时间      Branch Miss率
────────────────────────────────────────────────
随机数据 + 有分支       ~850ms    ~50%
排序数据 + 有分支       ~180ms    ~1%   (4.7x faster)
随机数据 + 无分支       ~320ms    <1%   (2.7x faster)
```

**学习要点**:
- 不可预测的分支非常昂贵
- 排序数据可以提高分支预测成功率
- 有时消除分支比优化分支更好

---

### 练习 1.5: 内存分配器对比

**目标**: 理解内存分配器的性能差异

**代码**:

```cpp
// 01-basic/allocator_test.cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>

const int NUM_THREADS = 8;
const int ALLOCS_PER_THREAD = 1000000;
const int ALLOC_SIZE = 64;

void allocation_worker() {
    std::vector<void*> ptrs;
    ptrs.reserve(ALLOCS_PER_THREAD);

    // 分配
    for (int i = 0; i < ALLOCS_PER_THREAD; i++) {
        void* ptr = malloc(ALLOC_SIZE);
        ptrs.push_back(ptr);

        // 模拟使用
        memset(ptr, i & 0xFF, ALLOC_SIZE);
    }

    // 释放
    for (void* ptr : ptrs) {
        free(ptr);
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(allocation_worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    std::cout << "Total time: " << duration << " ms" << std::endl;
    std::cout << "Allocs/sec: "
              << (NUM_THREADS * ALLOCS_PER_THREAD * 1000.0) / duration
              << std::endl;

    return 0;
}
```

**测试不同分配器**:

```bash
# 编译
g++ -O2 -pthread allocator_test.cpp -o allocator_test

# 1. 默认glibc malloc
./allocator_test

# 2. jemalloc
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ./allocator_test

# 3. tcmalloc
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 ./allocator_test

# 使用perf对比
perf stat -e cycles,instructions,cache-misses ./allocator_test
perf stat -e cycles,instructions,cache-misses \
    env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    ./allocator_test
```

**任务**:

1. ✅ 对比三种分配器的性能
2. ✅ 测量内存使用情况
3. ✅ 记录缓存miss率

**预期结果**:

```
分配器      时间(ms)   内存峰值   Cache Miss率
────────────────────────────────────────────
glibc       ~8500      2.5GB      18%
jemalloc    ~2200      1.8GB      8%   (3.9x faster)
tcmalloc    ~2500      1.9GB      9%   (3.4x faster)
```

---

## 3. 中级练习

### 练习 2.1: 实现高性能矩阵乘法

**目标**: 综合应用缓存优化、SIMD、循环展开

**起始代码**:

```cpp
// 02-intermediate/matmul.cpp
#include <iostream>
#include <chrono>
#include <immintrin.h>  // AVX
#include <random>

const int N = 1024;

// 朴素实现
void matmul_naive(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

// TODO: 实现优化版本
void matmul_optimized(float* A, float* B, float* C, int n) {
    // 你的优化代码
}

double benchmark(void (*func)(float*, float*, float*, int),
                float* A, float* B, float* C, int n,
                const char* name) {
    auto start = std::chrono::high_resolution_clock::now();
    func(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(
        end - start).count();

    double gflops = (2.0 * n * n * n) / (time_ms / 1000.0) / 1e9;

    std::cout << name << ": " << time_ms << " ms, "
              << gflops << " GFLOPS" << std::endl;

    return gflops;
}

int main() {
    // 分配对齐内存
    float* A = (float*)aligned_alloc(32, N * N * sizeof(float));
    float* B = (float*)aligned_alloc(32, N * N * sizeof(float));
    float* C = (float*)aligned_alloc(32, N * N * sizeof(float));

    // 初始化随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < N * N; i++) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    // 基准测试
    benchmark(matmul_naive, A, B, C, N, "Naive");
    // benchmark(matmul_optimized, A, B, C, N, "Optimized");

    free(A);
    free(B);
    free(C);

    return 0;
}
```

**优化任务** (循序渐进):

```cpp
// 优化1: 循环重排序 (i-k-j)
void matmul_reorder(float* A, float* B, float* C, int n) {
    memset(C, 0, n * n * sizeof(float));

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float a = A[i*n + k];
            for (int j = 0; j < n; j++) {
                C[i*n + j] += a * B[k*n + j];
            }
        }
    }
}

// 优化2: 分块 (Blocking)
void matmul_blocked(float* A, float* B, float* C, int n) {
    const int BLOCK = 32;
    memset(C, 0, n * n * sizeof(float));

    for (int ii = 0; ii < n; ii += BLOCK) {
        for (int jj = 0; jj < n; jj += BLOCK) {
            for (int kk = 0; kk < n; kk += BLOCK) {
                // 块内计算
                for (int i = ii; i < std::min(ii + BLOCK, n); i++) {
                    for (int k = kk; k < std::min(kk + BLOCK, n); k++) {
                        float a = A[i*n + k];
                        for (int j = jj; j < std::min(jj + BLOCK, n); j++) {
                            C[i*n + j] += a * B[k*n + j];
                        }
                    }
                }
            }
        }
    }
}

// 优化3: SIMD (AVX)
void matmul_simd(float* A, float* B, float* C, int n) {
    const int BLOCK = 32;
    memset(C, 0, n * n * sizeof(float));

    for (int ii = 0; ii < n; ii += BLOCK) {
        for (int jj = 0; jj < n; jj += BLOCK) {
            for (int kk = 0; kk < n; kk += BLOCK) {
                for (int i = ii; i < std::min(ii + BLOCK, n); i++) {
                    for (int k = kk; k < std::min(kk + BLOCK, n); k++) {
                        __m256 a_vec = _mm256_set1_ps(A[i*n + k]);

                        int j;
                        for (j = jj; j + 7 < std::min(jj + BLOCK, n); j += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[k*n + j]);
                            __m256 c_vec = _mm256_loadu_ps(&C[i*n + j]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&C[i*n + j], c_vec);
                        }

                        // 处理剩余
                        for (; j < std::min(jj + BLOCK, n); j++) {
                            C[i*n + j] += A[i*n + k] * B[k*n + j];
                        }
                    }
                }
            }
        }
    }
}

// 优化4: 多线程
#include <omp.h>

void matmul_parallel(float* A, float* B, float* C, int n) {
    const int BLOCK = 32;
    memset(C, 0, n * n * sizeof(float));

    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ii += BLOCK) {
        for (int jj = 0; jj < n; jj += BLOCK) {
            for (int kk = 0; kk < n; kk += BLOCK) {
                for (int i = ii; i < std::min(ii + BLOCK, n); i++) {
                    for (int k = kk; k < std::min(kk + BLOCK, n); k++) {
                        __m256 a_vec = _mm256_set1_ps(A[i*n + k]);

                        int j;
                        for (j = jj; j + 7 < std::min(jj + BLOCK, n); j += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[k*n + j]);
                            __m256 c_vec = _mm256_loadu_ps(&C[i*n + j]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&C[i*n + j], c_vec);
                        }

                        for (; j < std::min(jj + BLOCK, n); j++) {
                            C[i*n + j] += A[i*n + k] * B[k*n + j];
                        }
                    }
                }
            }
        }
    }
}
```

**编译和测试**:

```bash
# 编译
g++ -O3 -march=native -fopenmp matmul.cpp -o matmul

# 运行
./matmul

# 使用perf分析
perf stat -e cycles,instructions,cache-misses,cache-references ./matmul
```

**目标性能** (1024x1024):

```
实现          时间(ms)   GFLOPS   vs朴素
──────────────────────────────────────────
朴素          ~15000     0.14     1.0x
循环重排      ~8000      0.27     1.9x
分块          ~2300      0.93     6.5x
SIMD          ~550       3.89     27.3x
并行(8核)     ~85        25.18    176.5x ← 目标
```

**挑战**: 能否达到或超过这个性能？

---

### 练习 2.2: 实现无锁队列

**目标**: 学习无锁编程和memory_order

**模板代码**:

```cpp
// 02-intermediate/lockfree_queue.cpp
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

template<typename T, size_t Size>
class SPSCQueue {
private:
    struct alignas(64) AlignedIndex {
        std::atomic<size_t> value;
        char padding[64 - sizeof(std::atomic<size_t>)];
    };

    AlignedIndex head;
    AlignedIndex tail;
    T buffer[Size];

public:
    SPSCQueue() {
        head.value.store(0, std::memory_order_relaxed);
        tail.value.store(0, std::memory_order_relaxed);
    }

    // TODO: 实现enqueue
    bool enqueue(const T& item) {
        // 你的代码
        return false;
    }

    // TODO: 实现dequeue
    bool dequeue(T& item) {
        // 你的代码
        return false;
    }
};

// 基准测试
void benchmark_queue() {
    const int NUM_ITEMS = 10000000;
    SPSCQueue<int, 1024> queue;

    // 生产者线程
    auto producer = [&]() {
        for (int i = 0; i < NUM_ITEMS; i++) {
            while (!queue.enqueue(i)) {
                // 自旋等待
            }
        }
    };

    // 消费者线程
    auto consumer = [&]() {
        int item;
        for (int i = 0; i < NUM_ITEMS; i++) {
            while (!queue.dequeue(item)) {
                // 自旋等待
            }
        }
    };

    auto start = std::chrono::high_resolution_clock::now();

    std::thread prod(producer);
    std::thread cons(consumer);

    prod.join();
    cons.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    std::cout << "Time: " << duration << " ms" << std::endl;
    std::cout << "Throughput: "
              << (NUM_ITEMS * 1000.0) / duration / 1e6
              << " Mops/s" << std::endl;
}

int main() {
    benchmark_queue();
    return 0;
}
```

**实现提示**:

```cpp
// enqueue实现
bool enqueue(const T& item) {
    size_t tail_idx = tail.value.load(std::memory_order_relaxed);
    size_t next_tail = (tail_idx + 1) % Size;

    size_t head_idx = head.value.load(std::memory_order_acquire);
    if (next_tail == head_idx) {
        return false;  // 满了
    }

    buffer[tail_idx] = item;
    tail.value.store(next_tail, std::memory_order_release);
    return true;
}

// dequeue实现
bool dequeue(T& item) {
    size_t head_idx = head.value.load(std::memory_order_relaxed);

    size_t tail_idx = tail.value.load(std::memory_order_acquire);
    if (head_idx == tail_idx) {
        return false;  // 空了
    }

    item = buffer[head_idx];
    size_t next_head = (head_idx + 1) % Size;
    head.value.store(next_head, std::memory_order_release);
    return true;
}
```

**任务**:

1. ✅ 实现正确的无锁队列
2. ✅ 对比有锁实现的性能
3. ✅ 测试不同memory_order的影响
4. ✅ 使用ThreadSanitizer验证正确性

**性能目标**:

```
实现                吞吐量
──────────────────────────────
std::queue + mutex  2.5 Mops
无锁队列(seq_cst)   28 Mops
无锁队列(优化)      45 Mops  ← 目标
```

---

由于篇幅限制，让我继续创建剩余部分...
