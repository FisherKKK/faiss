# Faiss 性能优化技术课程 - 从入门到精通

## 课程简介

本课程深入讲解 Meta 开源的高性能向量搜索库 Faiss 中使用的各种性能优化技术。通过 7 天的学习，你将掌握从底层 SIMD 优化到高层架构设计的完整优化技术栈。

**适合人群**：有 C++ 基础，想学习系统级性能优化的工程师

**学习目标**：
- 理解现代 CPU 的性能特性（缓存、SIMD、流水线）
- 掌握 SIMD 向量化编程技术
- 学会设计缓存友好的数据结构
- 了解并行计算和 GPU 优化
- 能够将这些技术应用到实际项目中

---

## 第一天：内存对齐与缓存友好的数据结构

### 1.1 为什么内存对齐很重要？

**理论基础**：
- CPU 从内存读取数据是以**缓存行（Cache Line）**为单位的，通常是 64 字节
- 未对齐的数据可能跨越多个缓存行，导致额外的内存访问
- SIMD 指令要求数据对齐（AVX2 要求 32 字节对齐，AVX512 要求 64 字节对齐）

**代码示例 1：AlignedTable - 对齐内存分配器**

位置：`faiss/utils/AlignedTable.h`

```cpp
template <class T, int A = 32>
struct AlignedTableTightAlloc {
    T* ptr;
    size_t numel;

    // 使用 posix_memalign 分配对齐内存
    void resize(size_t n) {
        T* new_ptr;
        if (n > 0) {
            // posix_memalign: 分配 A 字节对齐的内存
            int ret = posix_memalign((void**)&new_ptr, A, n * sizeof(T));
            if (ret != 0) {
                throw std::bad_alloc();
            }
            if (numel > 0) {
                // 拷贝旧数据
                memcpy(new_ptr, ptr, sizeof(T) * std::min(numel, n));
            }
        } else {
            new_ptr = nullptr;
        }
        posix_memalign_free(ptr);
        ptr = new_ptr;
        numel = n;
    }

    T& operator[](size_t i) { return ptr[i]; }
};
```

**关键优化点**：
1. **对齐分配**：`posix_memalign` 确保内存起始地址是 32 字节的倍数
2. **编译时常量**：模板参数 `A` 在编译时确定，无运行时开销
3. **零开销抽象**：operator[] 内联后等同于指针访问

**实验：对齐 vs 未对齐的性能差异**

```cpp
// 未对齐：性能差
float* unaligned = new float[1000];
for (int i = 0; i < 1000; i += 8) {
    __m256 v = _mm256_loadu_ps(&unaligned[i]);  // loadu: unaligned load
}

// 对齐：性能好
AlignedTable<float, 32> aligned(1000);
for (int i = 0; i < 1000; i += 8) {
    __m256 v = _mm256_load_ps(&aligned[i]);     // load: aligned load (更快)
}
```

性能差异：对齐加载比未对齐加载快约 10-20%

### 1.2 几何内存扩展策略

**代码示例 2：AlignedTable - 避免频繁重新分配**

```cpp
template <class T, int A = 32>
struct AlignedTable {
    AlignedTableTightAlloc<T, A> tab;
    size_t numel = 0;  // 实际使用的元素数

    // 容量按 2 的幂次方增长
    static size_t round_capacity(size_t n) {
        if (n == 0) return 0;
        if (n < 8 * A) return 8 * A;  // 最小容量

        size_t capacity = 8 * A;
        while (capacity < n) {
            capacity *= 2;  // 几何增长
        }
        return capacity;
    }

    void resize(size_t n) {
        tab.resize(round_capacity(n));
        numel = n;
    }
};
```

**为什么这样设计？**

| 策略 | 插入 N 个元素的总时间复杂度 | 内存浪费 |
|------|---------------------------|---------|
| 每次 +1 | O(N²) | 0% |
| 每次 +100 | O(N²) | 不定 |
| 每次 ×2 | **O(N)** | 最多 50% |

示例：插入 1000 个元素
- 线性增长：需要重新分配 1000 次
- 几何增长：只需重新分配 log₂(1000) ≈ 10 次

### 1.3 缓存友好的数据布局：行主序 vs 列主序

**问题场景**：计算查询向量与 N 个数据库向量的距离

**方案 A：行主序（Row-Major）**
```cpp
// 数据布局：[x0_d0, x0_d1, ..., x0_d127, x1_d0, x1_d1, ...]
float database[N][D];  // N 个向量，每个 D 维

// 计算距离：访问模式是跳跃的
for (int i = 0; i < N; i++) {
    float dist = 0;
    for (int d = 0; d < D; d++) {
        dist += query[d] * database[i][d];  // 连续访问 database
    }
}
```

**方案 B：列主序（Column-Major / 转置）**
```cpp
// 数据布局：[x0_d0, x1_d0, ..., xN_d0, x0_d1, x1_d1, ...]
float database_T[D][N];  // 转置存储

// 计算距离：访问是连续的，SIMD 友好
for (int i = 0; i < N; i += 8) {  // 一次处理 8 个向量
    __m256 dist_vec = _mm256_setzero_ps();
    for (int d = 0; d < D; d++) {
        __m256 q = _mm256_set1_ps(query[d]);        // 广播查询
        __m256 db = _mm256_load_ps(&database_T[d][i]);  // 连续加载
        dist_vec = _mm256_fmadd_ps(q, db, dist_vec);
    }
}
```

**性能对比**：

| 数据布局 | 缓存命中率 | SIMD 利用率 | 相对性能 |
|----------|-----------|------------|---------|
| 行主序 | 低（跳跃访问） | 低 | 1x |
| 列主序 | 高（连续访问） | 高 | **3-5x** |

**Faiss 中的实现**：`faiss/utils/distances_simd.cpp:1415-1563`

```cpp
void fvec_L2sqr_ny_y_transposed(
        float* distances,
        const float* x,           // 查询向量
        const float* y,           // 数据库向量（转置存储）
        const float* y_sqlen,     // 预计算的向量长度
        size_t d,                 // 维度
        size_t d_offset,          // 转置的列间距
        size_t ny) {              // 数据库向量数

    // L2 距离公式：||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩
    float x_sqlen = 0;
    for (size_t j = 0; j < d; j++) {
        x_sqlen += x[j] * x[j];
    }

    // AVX2: 一次处理 8 个向量
    const size_t ny8 = ny / 8;
    __m256 x_sqlen_vec = _mm256_set1_ps(x_sqlen);

    for (size_t i = 0; i < ny8 * 8; i += 8) {
        // 预加载 2*x[j]
        __m256 m[D];
        for (size_t j = 0; j < d; j++) {
            m[j] = _mm256_set1_ps(2.0f * x[j]);
        }

        // 计算 -2⟨x, y⟩
        __m256 dot_product = _mm256_setzero_ps();
        for (size_t j = 0; j < d; j++) {
            __m256 y_vec = _mm256_loadu_ps(y + j * d_offset + i);
            dot_product = _mm256_fmadd_ps(m[j], y_vec, dot_product);
        }

        // 最终距离：||x||² + ||y||² - 2⟨x, y⟩
        __m256 y_sqlen_vec = _mm256_loadu_ps(y_sqlen + i);
        __m256 dist = _mm256_add_ps(x_sqlen_vec, y_sqlen_vec);
        dist = _mm256_sub_ps(dist, dot_product);

        _mm256_storeu_ps(distances + i, dist);
    }
}
```

### 1.4 数据结构设计：倒排表（Inverted Lists）

**核心思想**：分而治之 + 数据局部性

位置：`faiss/invlists/InvertedLists.h`

```cpp
// IVF (Inverted File) 索引的核心数据结构
struct InvertedLists {
    size_t nlist;      // 分区数量（如 1000 个聚类中心）
    size_t code_size;  // 每个编码的字节数

    // 每个分区存储：
    // - 向量 ID 列表
    // - 向量编码列表（压缩后的向量）

    virtual const idx_t* get_ids(size_t list_no) const = 0;
    virtual const uint8_t* get_codes(size_t list_no) const = 0;
    virtual size_t list_size(size_t list_no) const = 0;
};
```

**为什么这样设计？**

假设搜索 1 亿个向量：
- **不分区**：需要计算 1 亿次距离
- **分成 1000 个区**：
  - 只搜索最近的 10 个区（nprobe=10）
  - 只需计算 10 × 100,000 = 100 万次距离
  - **加速 100 倍！**

**缓存友好性**：
- 每个分区的数据连续存储
- 搜索时只需加载相关分区到缓存
- 大大提高缓存命中率

### 1.5 实战练习

**练习 1**：实现一个简单的对齐分配器
```cpp
template <int Alignment>
class AlignedVector {
public:
    void resize(size_t n) {
        // TODO: 使用 posix_memalign 实现
    }
private:
    float* data_;
    size_t size_;
};
```

**练习 2**：比较行主序和列主序的性能
```cpp
// 生成测试数据
const int N = 10000;
const int D = 128;
float queries[D];
float database_row[N][D];
float database_col[D][N];

// TODO: 测量两种布局的计算时间
```

**练习 3**：设计一个简单的倒排表
```cpp
// 实现一个支持分区的向量存储
class SimpleInvertedLists {
    // TODO: 设计数据结构
    // TODO: 实现 add() 和 search() 方法
};
```

---

## 第二天：SIMD 向量化编程基础

### 2.1 SIMD 是什么？

**SIMD = Single Instruction, Multiple Data**

传统标量指令：
```cpp
for (int i = 0; i < 8; i++) {
    c[i] = a[i] + b[i];  // 8 条指令，8 个周期
}
```

SIMD 向量指令：
```cpp
__m256 va = _mm256_load_ps(a);     // 加载 8 个 float
__m256 vb = _mm256_load_ps(b);     // 加载 8 个 float
__m256 vc = _mm256_add_ps(va, vb); // 1 条指令，同时加 8 个数
_mm256_store_ps(c, vc);            // 存储 8 个 float
```

**性能提升**：理论上最高 8 倍加速（AVX2），16 倍（AVX512）

### 2.2 Faiss 的 SIMD 抽象层

位置：`faiss/utils/simdlib.h`

```cpp
// 编译时根据 CPU 特性选择实现
#if defined(__AVX512F__)
    #include <faiss/utils/simdlib_avx512.h>
#elif defined(__AVX2__)
    #include <faiss/utils/simdlib_avx2.h>
#elif defined(__aarch64__)
    #include <faiss/utils/simdlib_neon.h>  // ARM NEON
#else
    #include <faiss/utils/simdlib_emulated.h>  // 标量模拟
#endif
```

**设计优势**：
- **跨平台**：同一份代码，自动适配不同 CPU
- **类型安全**：C++ 类包装，避免底层 intrinsic 的易错性
- **零开销**：所有函数都是内联的

### 2.3 基础 SIMD 数据类型

位置：`faiss/utils/simdlib_avx2.h`

```cpp
// 256 位寄存器 = 8 个 32 位 float
struct simd8float32 {
    __m256 f;  // 底层 AVX2 寄存器

    // 构造函数
    simd8float32() {}
    simd8float32(__m256 v) : f(v) {}
    simd8float32(float x) : f(_mm256_set1_ps(x)) {}  // 广播

    // 从内存加载
    simd8float32(const float* ptr) : f(_mm256_loadu_ps(ptr)) {}

    // 算术运算符重载
    simd8float32 operator+(simd8float32 other) const {
        return _mm256_add_ps(f, other.f);
    }

    simd8float32 operator*(simd8float32 other) const {
        return _mm256_mul_ps(f, other.f);
    }

    // 存储到内存
    void store(float* ptr) const {
        _mm256_storeu_ps(ptr, f);
    }
};

// 256 位寄存器 = 16 个 16 位整数
struct simd16uint16 {
    __m256i i;  // 整数寄存器

    simd16uint16 operator+(simd16uint16 other) const {
        return _mm256_add_epi16(i, other.i);
    }

    simd16uint16 operator&(simd16uint16 other) const {
        return _mm256_and_si256(i, other.i);
    }

    // 水平加法：16 个 16 位整数求和
    uint16_t sum() const {
        __m128i sum128 = _mm_add_epi16(
            _mm256_castsi256_si128(i),
            _mm256_extracti128_si256(i, 1)
        );
        // ... 继续归约
    }
};
```

### 2.4 实战案例 1：向量点积

**标量版本**：
```cpp
float dot_product_scalar(const float* x, const float* y, size_t d) {
    float sum = 0;
    for (size_t i = 0; i < d; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}
```

**AVX2 优化版本**：
```cpp
float dot_product_avx2(const float* x, const float* y, size_t d) {
    __m256 sum_vec = _mm256_setzero_ps();

    // 主循环：每次处理 8 个元素
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);
        sum_vec = _mm256_fmadd_ps(x_vec, y_vec, sum_vec);  // FMA!
    }

    // 水平归约：8 个部分和 → 1 个总和
    float sum[8];
    _mm256_storeu_ps(sum, sum_vec);
    float result = sum[0] + sum[1] + sum[2] + sum[3] +
                   sum[4] + sum[5] + sum[6] + sum[7];

    // 处理剩余元素
    for (; i < d; i++) {
        result += x[i] * y[i];
    }

    return result;
}
```

**关键优化技术**：
1. **FMA (Fused Multiply-Add)**：`a * b + c` 一条指令完成
2. **循环展开**：减少循环开销
3. **尾部处理**：处理不能被 8 整除的情况

### 2.5 实战案例 2：L2 距离计算

位置：`faiss/utils/distances_simd.cpp:194-240`

```cpp
// 计算 L2 距离：||x - y||² = Σ(xi - yi)²
float fvec_L2sqr(const float* x, const float* y, size_t d) {
    __m256 sum_vec = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);

        // diff = x - y
        __m256 diff = _mm256_sub_ps(x_vec, y_vec);

        // sum += diff * diff
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }

    // 归约 + 尾部处理
    // ... (同上)
}
```

### 2.6 高级技巧：维度特化

**问题**：不同维度有不同的最优策略

位置：`faiss/utils/distances_simd.cpp:1100-1150`

```cpp
// 模板元编程：编译时生成特化版本
template <size_t DIM>
void fvec_L2sqr_ny_D(float* dis, const float* x,
                     const float* y, size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        float sum = 0;
        // 编译器会完全展开这个循环！
        for (size_t j = 0; j < DIM; j++) {
            float diff = x[j] - y[i * DIM + j];
            sum += diff * diff;
        }
        dis[i] = sum;
    }
}

// 分发器：根据维度选择最优实现
void fvec_L2sqr_ny(float* dis, const float* x,
                   const float* y, size_t d, size_t ny) {
    #define DISPATCH(dim)                        \
        case dim:                                \
            fvec_L2sqr_ny_D<dim>(dis, x, y, ny); \
            return;

    switch (d) {
        DISPATCH(1)   // 特殊优化：1 维
        DISPATCH(2)   // 特殊优化：2 维
        DISPATCH(4)   // 特殊优化：4 维
        DISPATCH(8)   // 特殊优化：8 维
        DISPATCH(12)  // 特殊优化：12 维
        default:
            // 通用版本
            fvec_L2sqr_ny_ref(dis, x, y, d, ny);
    }

    #undef DISPATCH
}
```

**为什么这样做？**
- **小维度**：循环展开，消除循环开销
- **常见维度** (如 128, 256)：使用特定优化
- **任意维度**：回退到通用实现

### 2.7 SIMD 比较和选择

**问题**：如何在 SIMD 中实现 if-else？

```cpp
// 找出 16 个距离中的最小值和对应索引
struct simd16uint16 {
    // 比较：返回掩码
    simd16uint16 operator<(simd16uint16 other) const {
        return _mm256_cmpgt_epi16(other.i, i);  // 注意：反向比较
    }

    // 根据掩码选择
    simd16uint16 blend(simd16uint16 a, simd16uint16 b,
                       simd16uint16 mask) {
        return _mm256_blendv_epi8(a.i, b.i, mask.i);
    }
};

// 更新最小值和索引
void update_min(simd16uint16& min_vals, simd16uint16& min_ids,
                simd16uint16 new_vals, simd16uint16 new_ids) {
    // mask = (new_vals < min_vals) ? 0xFFFF : 0x0000
    simd16uint16 mask = new_vals < min_vals;

    // 根据 mask 选择新值或旧值
    min_vals = blend(min_vals, new_vals, mask);
    min_ids = blend(min_ids, new_ids, mask);
}
```

位置：`faiss/utils/simdlib_avx2.h:322-341`

```cpp
// 同时找出最小值和最大值
inline void cmplt_min_max_fast(
        const simd16uint16 candidateValues,
        const simd16uint16 candidateIndices,
        const simd16uint16 currentValues,
        const simd16uint16 currentIndices,
        simd16uint16& minValues,
        simd16uint16& minIndices,
        simd16uint16& maxValues,
        simd16uint16& maxIndices) {

    // 1. 生成比较掩码
    __m256i comparison = _mm256_cmpgt_epi16(
        currentValues.i, candidateValues.i);

    // 2. 找最小值
    minValues.i = _mm256_min_epi16(candidateValues.i, currentValues.i);
    minIndices.i = _mm256_blendv_epi8(
        candidateIndices.i, currentIndices.i, comparison);

    // 3. 找最大值
    maxValues.i = _mm256_max_epi16(candidateValues.i, currentValues.i);
    maxIndices.i = _mm256_blendv_epi8(
        currentIndices.i, candidateIndices.i, comparison);
}
```

**应用场景**：Top-K 搜索中同时维护最小堆和最大堆

### 2.8 实战练习

**练习 1**：实现 AVX2 版本的向量加法
```cpp
void vector_add_avx2(float* result, const float* a,
                     const float* b, size_t n) {
    // TODO: 使用 AVX2 实现
}
```

**练习 2**：实现 SIMD 版本的 max 函数
```cpp
float vector_max_avx2(const float* data, size_t n) {
    // TODO: 使用 _mm256_max_ps 和水平归约
}
```

**练习 3**：优化条件累加
```cpp
// 计算满足条件的元素之和
float conditional_sum(const float* data, size_t n, float threshold) {
    // TODO: 使用 SIMD 比较和掩码
}
```

---

## 第三天：缓存优化与数据局部性

### 3.1 CPU 缓存层次结构

**现代 CPU 的内存层次**：

```
CPU 核心
   ↓ ~1 cycle
L1 Cache (32 KB)    ← 最快，最小
   ↓ ~4 cycles
L2 Cache (256 KB)
   ↓ ~12 cycles
L3 Cache (8-32 MB)  ← 所有核心共享
   ↓ ~40 cycles
主内存 (GB)          ← 最慢，最大
   ↓ ~200 cycles
```

**关键指标**：
- **缓存行大小**：64 字节（16 个 float）
- **缓存未命中代价**：比命中慢 100-200 倍
- **带宽限制**：L1 ≈ 1 TB/s，主内存 ≈ 100 GB/s

### 3.2 问题 1：False Sharing（伪共享）

**什么是伪共享？**

```cpp
struct Counter {
    int64_t count;  // 8 字节
    // 填充到 64 字节不够！
};

Counter counters[4];  // 4 个计数器

// 多线程更新
#pragma omp parallel for
for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 1000000; j++) {
        counters[i].count++;  // 性能灾难！
    }
}
```

**问题**：
- counters[0] 和 counters[1] 在同一个缓存行
- 线程 0 修改 counters[0]，缓存行失效
- 线程 1 的缓存行也被迫失效（即使它用的是 counters[1]）
- 不断的缓存行同步 → 性能崩溃

**解决方案：缓存行对齐**

```cpp
struct alignas(64) Counter {  // 强制 64 字节对齐
    int64_t count;
    char padding[64 - sizeof(int64_t)];  // 填充到 64 字节
};

// 或者使用 Faiss 的对齐分配器
AlignedTable<Counter, 64> counters(4);
```

**性能提升**：多线程场景下可提升 10-100 倍

### 3.3 问题 2：数据预取（Prefetching）

**手动预取**：

位置：`faiss/utils/prefetch.h`

```cpp
#include <xmmintrin.h>  // SSE

// 预取到 L1 缓存
void prefetch_L1(const void* addr) {
    _mm_prefetch((const char*)addr, _MM_HINT_T0);
}

// 预取到 L2 缓存
void prefetch_L2(const void* addr) {
    _mm_prefetch((const char*)addr, _MM_HINT_T1);
}

// 预取到 L3 缓存
void prefetch_L3(const void* addr) {
    _mm_prefetch((const char*)addr, _MM_HINT_T2);
}

// 非临时预取（不污染缓存）
void prefetch_NTA(const void* addr) {
    _mm_prefetch((const char*)addr, _MM_HINT_NTA);
}
```

**应用场景**：循环中提前加载下一次迭代的数据

```cpp
// IVF 搜索：遍历倒排表
for (size_t i = 0; i < list_size; i++) {
    // 预取下一个元素（领先 64 个元素）
    if (i + 64 < list_size) {
        _mm_prefetch(&codes[(i + 64) * code_size], _MM_HINT_T0);
        _mm_prefetch(&ids[i + 64], _MM_HINT_T0);
    }

    // 处理当前元素
    float dist = compute_distance(query, &codes[i * code_size]);
    if (dist < threshold) {
        results.push(ids[i], dist);
    }
}
```

**经验法则**：
- 预取距离 ≈ 内存延迟 / 循环时间
- 太近：预取还没完成就用到了
- 太远：数据已经被驱逐出缓存

### 3.4 问题 3：数据分块（Blocking）

**示例：矩阵乘法优化**

```cpp
// 朴素版本：缓存不友好
void matmul_naive(float* C, const float* A, const float* B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];  // B 列访问！
            }
            C[i * N + j] = sum;
        }
    }
}

// 分块版本：缓存友好
void matmul_blocked(float* C, const float* A, const float* B, int N) {
    const int BLOCK = 64;  // 块大小 = L1 缓存 / sizeof(float) / 2

    for (int ii = 0; ii < N; ii += BLOCK) {
        for (int jj = 0; jj < N; jj += BLOCK) {
            for (int kk = 0; kk < N; kk += BLOCK) {
                // 小块乘法：数据全在 L1 缓存中
                for (int i = ii; i < std::min(ii + BLOCK, N); i++) {
                    for (int j = jj; j < std::min(jj + BLOCK, N); j++) {
                        float sum = C[i * N + j];
                        for (int k = kk; k < std::min(kk + BLOCK, N); k++) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}
```

**性能对比**：分块版本快 5-10 倍

### 3.5 Faiss 的缓存优化：PQ4 FastScan

位置：`faiss/impl/pq4_fast_scan.h`

**核心思想**：重新组织数据布局，最大化缓存利用率

**传统 PQ 编码**：
```
向量 0: [code_0, code_1, ..., code_M]
向量 1: [code_0, code_1, ..., code_M]
...
```

**FastScan 编码**：
```
Block 0 (32 个向量):
  子量化器 0: [v0_c0, v1_c0, ..., v31_c0]  ← 32 个 4-bit 码
  子量化器 1: [v0_c1, v1_c1, ..., v31_c1]
  ...
Block 1 (32 个向量):
  ...
```

**为什么更快？**

1. **块处理** (bbs = 32 或 64)：
   - 一个块的数据 < 1 KB，完全放入 L1 缓存
   - 查询处理时，块内所有数据都是热的

2. **SIMD 友好**：
   - 32 个 4-bit 码 = 16 字节，正好加载到 __m128i
   - 使用 `_mm_shuffle_epi8` 查找表，超快！

3. **连续访问**：
   - 同一个子量化器的所有码连续存储
   - 高缓存命中率

**代码实现**：

```cpp
// PQ4 码重排：从标准布局到 FastScan 布局
void pq4_pack_codes(
        const uint8_t* codes,   // 输入：标准布局
        size_t ntotal,          // 向量总数
        size_t M,               // 子量化器数量
        size_t nb,              // 输出向量数（向上取整到 bbs 的倍数）
        size_t bbs,             // 块大小（32 或 64）
        size_t nsq,             // 向上取整到偶数的 M
        uint8_t* blocks) {      // 输出：FastScan 布局

    // 按块处理
    for (size_t block_start = 0; block_start < nb; block_start += bbs) {
        for (size_t sq = 0; sq < nsq; sq += 2) {  // 每次处理 2 个子量化器
            for (size_t i = 0; i < bbs; i++) {
                size_t vid = block_start + i;
                uint8_t code0 = 0, code1 = 0;

                if (vid < ntotal) {
                    code0 = get_code_component(codes, vid, sq, M);
                    if (sq + 1 < M) {
                        code1 = get_code_component(codes, vid, sq + 1, M);
                    }
                }

                // 打包：两个 4-bit 码 → 一个字节
                *blocks++ = code0 | (code1 << 4);
            }
        }
    }
}
```

**性能提升**：
- 比标量 PQ 快 10-20 倍
- 比 SIMD PQ（无重排）快 2-3 倍

### 3.6 缓存感知的堆操作

位置：`faiss/utils/Heap.h`

**问题**：传统堆（二叉堆）的缓存性能很差
- 每次 push/pop 需要跳跃访问
- 缓存局部性差

**Faiss 的解决方案**：固定大小的 k-way 堆

```cpp
// 保留 Top-K 结果的最小堆
template <class C>
struct ResultHeap {
    using T = typename C::T;    // 值类型（float）
    using TI = typename C::TI;  // 索引类型（int64_t）

    size_t k;
    T* values;    // 数组存储，缓存友好
    TI* ids;

    // 替换堆顶（最常用操作）
    void replace_top(T val, TI id) {
        if (!C::cmp(val, values[0])) {  // val 不如堆顶
            return;
        }

        // 下沉操作
        values[0] = val;
        ids[0] = id;

        size_t i = 0;
        while (true) {
            size_t left = 2 * i + 1;
            size_t right = 2 * i + 2;
            size_t largest = i;

            // 找出父子三者中的最大值
            if (left < k && C::cmp(values[left], values[largest])) {
                largest = left;
            }
            if (right < k && C::cmp(values[right], values[largest])) {
                largest = right;
            }

            if (largest == i) break;

            // 交换
            std::swap(values[i], values[largest]);
            std::swap(ids[i], ids[largest]);
            i = largest;
        }
    }
};
```

**优化点**：
1. **数组存储**：连续内存，高缓存命中率
2. **固定 K**：编译器可以优化循环
3. **避免动态分配**：预分配内存

**替代方案：SIMD 堆**（用于小 K）

```cpp
// 使用 SIMD 寄存器作为堆（K = 8 或 16）
struct SIMDHeap {
    simd16uint16 min_values;
    simd16uint16 min_ids;

    void update(simd16uint16 new_values, simd16uint16 new_ids) {
        // 向量化比较和交换
        simd16uint16 mask = new_values < min_values;
        min_values = blend(min_values, new_values, mask);
        min_ids = blend(min_ids, new_ids, mask);
    }
};
```

### 3.7 实战练习

**练习 1**：测量缓存未命中
```cpp
// 使用 perf 工具测量缓存性能
// perf stat -e cache-misses,cache-references ./your_program

void test_cache_locality() {
    const int N = 10000000;
    int* data = new int[N];

    // 顺序访问 vs 随机访问
    // TODO: 实现并比较性能
}
```

**练习 2**：实现块矩阵乘法
```cpp
void blocked_matmul(float* C, const float* A, const float* B,
                    int N, int BLOCK_SIZE) {
    // TODO: 实现分块算法
}
```

**练习 3**：优化查找表查询
```cpp
// 给定编码查询距离表
void lookup_distances(float* dists, const uint8_t* codes,
                      const float* lut, int n, int M) {
    // TODO: 使用预取和 SIMD 优化
}
```

---

## 第四天：算法设计模式与抽象

### 4.1 设计模式：Strategy Pattern（策略模式）

**问题**：不同的距离度量需要不同的计算方法

**Faiss 的解决方案**：ElementOp 模式

位置：`faiss/utils/distances_simd.cpp:63-88`

```cpp
// 策略接口：定义距离计算的"操作"
struct ElementOpL2 {
    // 标量版本
    static float op(float x, float y) {
        float tmp = x - y;
        return tmp * tmp;
    }

    // AVX2 版本
    static __m256 op(__m256 x, __m256 y) {
        __m256 tmp = _mm256_sub_ps(x, y);
        return _mm256_mul_ps(tmp, tmp);
    }

    // AVX512 版本
    static __m512 op(__m512 x, __m512 y) {
        __m512 tmp = _mm512_sub_ps(x, y);
        return _mm512_mul_ps(tmp, tmp);
    }
};

struct ElementOpIP {  // Inner Product
    static float op(float x, float y) {
        return x * y;
    }

    static __m256 op(__m256 x, __m256 y) {
        return _mm256_mul_ps(x, y);
    }

    static __m512 op(__m512 x, __m512 y) {
        return _mm512_mul_ps(x, y);
    }
};
```

**通用计算函数**：

```cpp
// 模板函数：适用于任何 ElementOp
template <class ElementOp>
float compute_distance_generic(const float* x, const float* y, size_t d) {
    __m256 sum_vec = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 y_vec = _mm256_loadu_ps(y + i);

        // 调用策略的 op 方法
        __m256 result = ElementOp::op(x_vec, y_vec);
        sum_vec = _mm256_add_ps(sum_vec, result);
    }

    // 归约 + 尾部处理
    float sum = horizontal_sum(sum_vec);
    for (; i < d; i++) {
        sum += ElementOp::op(x[i], y[i]);
    }

    return sum;
}

// 使用：
float l2_dist = compute_distance_generic<ElementOpL2>(x, y, d);
float ip_dist = compute_distance_generic<ElementOpIP>(x, y, d);
```

**优势**：
- **零开销抽象**：模板在编译时展开，无虚函数开销
- **代码复用**：一份代码支持多种距离
- **易于扩展**：添加新距离只需定义新的 ElementOp

### 4.2 设计模式：Template Method Pattern（模板方法）

**场景**：索引的通用搜索流程

位置：`faiss/IndexIVF.cpp`

```cpp
// 基类定义搜索框架
struct IndexIVF : Index {
    // 模板方法：定义搜索的骨架
    void search(idx_t n, const float* x, idx_t k,
                float* distances, idx_t* labels) const override {
        // 1. 量化查询向量 → 找到最近的聚类中心
        std::vector<idx_t> assign(n * nprobe);
        std::vector<float> centroid_dis(n * nprobe);
        quantizer->search(n, x, nprobe,
                         centroid_dis.data(), assign.data());

        // 2. 搜索选中的倒排表（具体实现由子类提供）
        search_preassigned(n, x, k, assign.data(), centroid_dis.data(),
                          distances, labels, false);
    }

    // 钩子方法：由子类实现
    virtual void search_preassigned(...) const = 0;
};

// 子类 1：精确搜索
struct IndexIVFFlat : IndexIVF {
    void search_preassigned(...) const override {
        // 使用原始向量计算精确距离
        for (int i = 0; i < nprobe; i++) {
            const float* list_vecs = invlists->get_codes(assign[i]);
            compute_exact_distances(query, list_vecs, ...);
        }
    }
};

// 子类 2：近似搜索（PQ 压缩）
struct IndexIVFPQ : IndexIVF {
    void search_preassigned(...) const override {
        // 使用查找表快速计算近似距离
        float lut[256 * M];  // 查找表
        compute_LUT(query, lut);

        for (int i = 0; i < nprobe; i++) {
            const uint8_t* codes = invlists->get_codes(assign[i]);
            compute_PQ_distances(codes, lut, ...);
        }
    }
};
```

**优势**：
- **统一接口**：所有 IVF 索引有相同的 search() 方法
- **灵活实现**：子类可以自定义内部搜索策略
- **代码复用**：量化步骤只实现一次

### 4.3 设计模式：Visitor Pattern（访问者模式）

**场景**：处理 SIMD 计算结果

位置：`faiss/impl/simd_result_handlers.h`

```cpp
// 访问者接口：定义如何处理距离结果
struct SIMDResultHandler {
    // 处理一批（32 个）距离结果
    virtual void handle(
        size_t q,              // 查询索引
        size_t b,              // 数据库块索引
        simd16uint16 d0,       // 距离 [0..15]
        simd16uint16 d1        // 距离 [16..31]
    ) = 0;

    virtual void set_block_origin(size_t i0, size_t j0) = 0;
};

// 具体访问者 1：收集所有结果
struct StoreResultHandler : SIMDResultHandler {
    std::vector<float> distances;

    void handle(size_t q, size_t b,
                simd16uint16 d0, simd16uint16 d1) override {
        // 存储所有距离
        for (int i = 0; i < 16; i++) {
            distances.push_back(extract_element(d0, i));
            distances.push_back(extract_element(d1, i));
        }
    }
};

// 具体访问者 2：维护 Top-K 堆
struct HeapResultHandler : SIMDResultHandler {
    int k;
    std::vector<float> heap_values;
    std::vector<idx_t> heap_ids;

    void handle(size_t q, size_t b,
                simd16uint16 d0, simd16uint16 d1) override {
        // 批量更新堆
        for (int i = 0; i < 16; i++) {
            float dist = extract_element(d0, i);
            if (dist < heap_values[0]) {
                heap_replace_top(k, heap_values.data(), heap_ids.data(),
                                dist, b * 32 + i);
            }
        }
        // 同样处理 d1...
    }
};

// 具体访问者 3：范围搜索
struct RangeResultHandler : SIMDResultHandler {
    float radius;
    std::vector<idx_t> result_ids;
    std::vector<float> result_dists;

    void handle(size_t q, size_t b,
                simd16uint16 d0, simd16uint16 d1) override {
        // 收集半径内的所有结果
        __m256i mask = _mm256_cmpgt_epi16(
            _mm256_set1_epi16(radius), d0.i);

        // 根据掩码提取满足条件的结果
        // ...
    }
};
```

**SIMD 核心计算**：

```cpp
// 通用 SIMD 搜索内核
template <int NQ, int NB>
void pq4_accumulate_loop(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        SIMDResultHandler& res) {  // 访问者参数

    simd16uint16 distances[NQ][NB];

    // ... SIMD 计算距离 ...

    // 将结果传递给访问者
    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < NB; b += 2) {
            res.handle(q, b, distances[q][b], distances[q][b + 1]);
        }
    }
}

// 使用：
HeapResultHandler heap_handler(k);
pq4_accumulate_loop<1, 4>(nsq, codes, LUT, heap_handler);
```

**优势**：
- **分离关注点**：距离计算与结果处理分离
- **灵活性**：同一个计算内核支持多种输出格式
- **性能**：访问者方法是虚函数但调用频率低，性能影响小

### 4.4 设计模式：CRTP（Curiously Recurring Template Pattern）

**问题**：想要多态但不想虚函数的开销

```cpp
// 基类是模板，参数是派生类自己！
template <typename Derived>
struct DistanceComputerBase {
    float operator()(idx_t i) {
        // 静态分发到派生类
        return static_cast<Derived*>(this)->compute(i);
    }

    void set_query(const float* x) {
        static_cast<Derived*>(this)->set_query_impl(x);
    }
};

// 派生类
struct L2DistanceComputer : DistanceComputerBase<L2DistanceComputer> {
    const float* query;
    const float* database;

    void set_query_impl(const float* x) {
        query = x;
    }

    float compute(idx_t i) {
        return fvec_L2sqr(query, database + i * d, d);
    }
};

struct IPDistanceComputer : DistanceComputerBase<IPDistanceComputer> {
    const float* query;
    const float* database;

    void set_query_impl(const float* x) {
        query = x;
    }

    float compute(idx_t i) {
        return fvec_inner_product(query, database + i * d, d);
    }
};

// 使用：
template <typename DC>
void search_with_distance(DC& dc, int n) {
    dc.set_query(query);  // 编译时确定调用哪个函数
    for (int i = 0; i < n; i++) {
        float dist = dc(i);  // 无虚函数开销！
        // ...
    }
}
```

**对比虚函数**：

| 方法 | 优点 | 缺点 |
|------|------|------|
| 虚函数 | 运行时多态 | 间接调用开销（~5-10%） |
| CRTP | 编译时多态，零开销 | 必须在编译时知道类型 |

### 4.5 设计模式：Type Traits

**问题**：为不同类型提供不同的优化

```cpp
// 类型特性：判断是否可以 SIMD 优化
template <typename T>
struct is_simd_compatible {
    static constexpr bool value = false;
};

template <>
struct is_simd_compatible<float> {
    static constexpr bool value = true;
};

// 条件编译：根据类型选择实现
template <typename T>
typename std::enable_if<is_simd_compatible<T>::value, T>::type
dot_product(const T* x, const T* y, size_t d) {
    // SIMD 优化版本
    return dot_product_avx2(x, y, d);
}

template <typename T>
typename std::enable_if<!is_simd_compatible<T>::value, T>::type
dot_product(const T* x, const T* y, size_t d) {
    // 标量版本
    T sum = 0;
    for (size_t i = 0; i < d; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}
```

**C++17 改进：if constexpr**

```cpp
template <typename T>
T dot_product(const T* x, const T* y, size_t d) {
    if constexpr (is_simd_compatible<T>::value) {
        // 编译时选择：SIMD 版本
        return dot_product_avx2(x, y, d);
    } else {
        // 编译时选择：标量版本
        T sum = 0;
        for (size_t i = 0; i < d; i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }
}
```

### 4.6 实战练习

**练习 1**：实现策略模式
```cpp
// 定义新的距离度量
struct ElementOpL1 {
    static float op(float x, float y) {
        // TODO: 实现 L1 距离（绝对值）
    }

    static __m256 op(__m256 x, __m256 y) {
        // TODO: 实现 AVX2 版本
        // 提示：使用 _mm256_and_ps 清除符号位
    }
};
```

**练习 2**：实现 CRTP
```cpp
// 设计一个通用的迭代器基类
template <typename Derived>
class IteratorBase {
public:
    void advance() {
        static_cast<Derived*>(this)->advance_impl();
    }

    // TODO: 添加更多方法
};

// TODO: 实现具体的迭代器
```

**练习 3**：使用 Visitor 模式
```cpp
// 设计一个灵活的搜索结果处理器
struct SearchResultHandler {
    virtual void handle_result(idx_t id, float distance) = 0;
};

// TODO: 实现不同的 handler（TopK, Range, Count, etc.）
```

---

## 第五天：并行计算与多线程优化

### 5.1 OpenMP 基础

**OpenMP**：编译器指令驱动的并行化

```cpp
#include <omp.h>

// 最简单的并行化
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    result[i] = compute(data[i]);
}

// 设置线程数
omp_set_num_threads(8);

// 获取线程信息
int tid = omp_get_thread_num();      // 当前线程 ID
int nthreads = omp_get_num_threads(); // 总线程数
```

### 5.2 并行归约（Reduction）

**问题**：多线程累加需要同步

```cpp
// 错误：数据竞争
float sum = 0;
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    sum += data[i];  // 多个线程同时写 sum！
}

// 正确：使用 reduction 子句
float sum = 0;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < n; i++) {
    sum += data[i];  // OpenMP 自动处理同步
}
```

**OpenMP 归约原理**：
1. 每个线程有自己的局部 sum
2. 线程内累加
3. 最后合并所有局部 sum

**支持的归约操作**：
- 算术：`+`, `-`, `*`
- 逻辑：`&&`, `||`
- 位运算：`&`, `|`, `^`
- 比较：`max`, `min`

### 5.3 Faiss 中的并行索引构建

位置：`faiss/IndexIVF.cpp`

```cpp
void IndexIVF::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    // 1. 并行量化：将向量分配到聚类
    std::vector<idx_t> assign(n);
    quantizer->assign(n, x, assign.data());

    // 2. 统计每个聚类的向量数
    std::vector<size_t> hist(nlist, 0);
    for (idx_t i = 0; i < n; i++) {
        hist[assign[i]]++;
    }

    // 3. 为每个聚类预分配空间（避免动态增长）
    #pragma omp parallel for
    for (size_t i = 0; i < nlist; i++) {
        if (hist[i] > 0) {
            invlists->resize(i, invlists->list_size(i) + hist[i]);
        }
    }

    // 4. 并行添加向量到倒排表
    std::vector<size_t> offsets(nlist, 0);

    // 串行分配偏移量（快，因为只是计数）
    for (idx_t i = 0; i < n; i++) {
        idx_t list_no = assign[i];
        size_t offset = offsets[list_no]++;

        // 编码并存储
        uint8_t* code = invlists->get_codes(list_no) + offset * code_size;
        encode_vector(i, x + i * d, code);
        invlists->get_ids(list_no)[offset] = xids ? xids[i] : i;
    }
}
```

**优化要点**：
1. **预分配**：避免多线程动态增长导致的竞争
2. **分区独立**：不同的倒排表可以并行处理
3. **批量操作**：减少同步开销

### 5.4 并行搜索

```cpp
void IndexIVF::search(idx_t n, const float* x, idx_t k,
                      float* distances, idx_t* labels) const {
    // 多查询并行搜索
    #pragma omp parallel for if (n > 1)
    for (idx_t i = 0; i < n; i++) {
        const float* query = x + i * d;
        float* dis = distances + i * k;
        idx_t* lab = labels + i * k;

        // 1. 为当前查询找到最近的 nprobe 个聚类
        std::vector<idx_t> list_nos(nprobe);
        std::vector<float> list_dis(nprobe);
        quantizer->search(1, query, nprobe,
                         list_dis.data(), list_nos.data());

        // 2. 搜索这些倒排表
        heap_heapify<CMax<float, idx_t>>(k, dis, lab);

        for (size_t j = 0; j < nprobe; j++) {
            idx_t list_no = list_nos[j];
            size_t list_size = invlists->list_size(list_no);

            const uint8_t* codes = invlists->get_codes(list_no);
            const idx_t* ids = invlists->get_ids(list_no);

            // 扫描倒排表
            scan_list(query, list_size, codes, ids, dis, lab);
        }

        heap_reorder<CMax<float, idx_t>>(k, dis, lab);
    }
}
```

**并行策略选择**：

| 查询数 | 并行策略 | 原因 |
|--------|----------|------|
| n = 1 | 不并行 | 开销大于收益 |
| n < 线程数 | 查询级并行 | 每个查询一个线程 |
| n >> 线程数 | 查询级并行 | 动态调度，负载均衡 |

### 5.5 线程安全的内存分配

**问题**：多线程下的 `new` 和 `malloc` 有锁竞争

**解决方案**：线程本地存储（Thread-Local Storage）

```cpp
// 每个线程有自己的缓冲区
thread_local std::vector<float> distances_buffer;
thread_local std::vector<idx_t> ids_buffer;

void search_thread_safe(const float* query) {
    // 调整缓冲区大小（不跨线程分配）
    distances_buffer.resize(max_scan_size);
    ids_buffer.resize(max_scan_size);

    // 使用缓冲区...
}
```

**Faiss 的实现**：`MaybeOwnedVector`

```cpp
template <typename T>
struct MaybeOwnedVector {
    T* data;
    size_t size;
    bool owned;

    // 构造：可以接管外部内存或自己分配
    MaybeOwnedVector() : data(nullptr), size(0), owned(false) {}

    MaybeOwnedVector(size_t n) : size(n), owned(true) {
        data = new T[n];
    }

    // 接管外部内存（零拷贝）
    void set(T* data_, size_t size_) {
        if (owned) delete[] data;
        data = data_;
        size = size_;
        owned = false;
    }

    ~MaybeOwnedVector() {
        if (owned) delete[] data;
    }
};
```

### 5.6 HNSW 图构建中的并行与锁

位置：`faiss/impl/HNSW.h`

**挑战**：图结构的并行构建需要精细的锁控制

```cpp
struct HNSW {
    // 邻居列表：neighbors[offsets[i]:offsets[i+1]] 存储向量 i 的邻居
    std::vector<storage_idx_t> neighbors;
    std::vector<size_t> offsets;

    // 每个节点一把锁（细粒度锁）
    std::vector<omp_lock_t> locks;

    // 初始化锁
    void init_locks(size_t n) {
        locks.resize(n);
        for (size_t i = 0; i < n; i++) {
            omp_init_lock(&locks[i]);
        }
    }

    // 并行添加向量
    void add_with_locks(
            DistanceComputer& dis,
            int pt_level,
            storage_idx_t pt_id,
            std::vector<storage_idx_t>& ep) {

        // 从高层到低层依次添加
        for (int level = pt_level; level >= 0; level--) {
            // 搜索当前层的最近邻
            std::vector<Node> candidates;
            search_layer(dis, candidates, ep, level);

            // 选择最优邻居
            std::vector<storage_idx_t> neighbors;
            select_neighbors(candidates, neighbors, level);

            // 添加双向边（需要锁）
            for (auto neighbor : neighbors) {
                // 锁定邻居节点
                omp_set_lock(&locks[neighbor]);

                // 添加边：neighbor → pt_id
                add_link(neighbor, pt_id, level);

                omp_unset_lock(&locks[neighbor]);
            }

            // 锁定新节点
            omp_set_lock(&locks[pt_id]);

            // 添加边：pt_id → neighbors
            set_neighbors(pt_id, neighbors, level);

            omp_unset_lock(&locks[pt_id]);
        }
    }
};
```

**锁策略**：
1. **细粒度锁**：每个节点一把锁，而不是全局锁
2. **最小化临界区**：只在修改邻居列表时持锁
3. **避免死锁**：总是按 ID 递增顺序获取锁

**性能**：
- 1 线程：基准
- 4 线程：3.5x 加速
- 16 线程：12x 加速（不是 16x 因为有锁竞争）

### 5.7 Lock-Free 数据结构

**场景**：多线程更新倒排表

```cpp
// 使用原子操作实现无锁追加
struct LockFreeInvertedList {
    std::atomic<size_t> size;
    std::vector<idx_t> ids;
    std::vector<uint8_t> codes;

    // 原子追加
    void append(idx_t id, const uint8_t* code, size_t code_size) {
        // 原子递增并获取旧值
        size_t pos = size.fetch_add(1, std::memory_order_relaxed);

        // 写入数据（不需要锁，因为位置是独占的）
        ids[pos] = id;
        memcpy(codes.data() + pos * code_size, code, code_size);
    }
};
```

**注意**：只在追加场景下有效，不适用于复杂操作

### 5.8 实战练习

**练习 1**：并行求和
```cpp
// 使用 OpenMP 实现并行求和，比较 reduction 和手动实现
float parallel_sum_reduction(const float* data, size_t n);
float parallel_sum_manual(const float* data, size_t n);
```

**练习 2**：并行直方图
```cpp
// 统计数据分布（注意竞争条件！）
void parallel_histogram(const int* data, size_t n, int* hist, int nbins);
```

**练习 3**：并行排序
```cpp
// 实现并行快速排序或归并排序
void parallel_sort(float* data, size_t n);
```

---

## 第六天：GPU 优化技术

### 6.1 GPU 架构基础

**CPU vs GPU**：

| 特性 | CPU | GPU |
|------|-----|-----|
| 核心数 | 8-64 | 1000-10000 |
| 线程数 | 10-100 | 10000-100000 |
| 缓存 | 大（32 MB L3） | 小（几 MB） |
| 控制流 | 复杂分支 | SIMT（分支代价高） |
| 内存带宽 | 100 GB/s | 1000 GB/s |
| 适用场景 | 通用计算 | 数据并行 |

**CUDA 编程模型**：

```cuda
// 内核函数：在 GPU 上执行
__global__ void vector_add(float* c, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// 主机代码
void add_vectors(float* c, const float* a, const float* b, int n) {
    // 1. 分配 GPU 内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // 2. 拷贝数据到 GPU
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // 3. 启动内核
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    vector_add<<<blocks, threads_per_block>>>(d_c, d_a, d_b, n);

    // 4. 拷贝结果回 CPU
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. 释放 GPU 内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

### 6.2 Faiss GPU 架构

位置：`faiss/gpu/`

**设计原则**：
1. **接口兼容**：GPU 索引继承自 CPU 索引
2. **自动转换**：GpuCloner 自动转换 CPU ↔ GPU
3. **多 GPU 支持**：透明的多 GPU 并行

```cpp
// CPU 索引
IndexFlatL2 cpu_index(d);
cpu_index.add(n, vectors);

// 转换为 GPU 索引
StandardGpuResources res;
GpuIndexFlatL2 gpu_index(&res, d);
gpu_index.copyFrom(&cpu_index);  // CPU → GPU

// 搜索（API 相同）
gpu_index.search(nq, queries, k, distances, labels);

// 转回 CPU
IndexFlatL2 cpu_index2(d);
gpu_index.copyTo(&cpu_index2);  // GPU → CPU
```

### 6.3 GPU 内存层次与优化

**CUDA 内存类型**：

```cuda
// 全局内存：大但慢（~400 cycles）
__global__ void use_global(float* data) {
    int i = threadIdx.x;
    float x = data[i];  // 全局内存访问
}

// 共享内存：小但快（~5 cycles），类似 L1 缓存
__global__ void use_shared(float* data) {
    __shared__ float shared_data[256];  // 声明共享内存

    int i = threadIdx.x;

    // 协作加载：所有线程一起加载数据
    shared_data[i] = data[blockIdx.x * blockDim.x + i];
    __syncthreads();  // 同步，确保数据已加载

    // 从共享内存读取（快！）
    float x = shared_data[i];
    // 计算...
}

// 寄存器：最快但容量极小
__global__ void use_registers() {
    float local_var = 1.0f;  // 存储在寄存器中
    // 每个线程有自己的 local_var
}
```

**Faiss GPU 的 L2 距离计算**：

位置：`faiss/gpu/GpuDistance.cu`

```cuda
template <int DIM>
__global__ void l2_distance_kernel(
        float* distances,
        const float* queries,     // [nq, DIM]
        const float* database,    // [nb, DIM]
        int nq, int nb) {

    // 共享内存：缓存查询向量
    __shared__ float query_cache[DIM];

    int query_id = blockIdx.x;
    int db_start = blockIdx.y * blockDim.x;
    int tid = threadIdx.x;

    // 协作加载查询向量到共享内存
    if (tid < DIM) {
        query_cache[tid] = queries[query_id * DIM + tid];
    }
    __syncthreads();

    // 每个线程计算一个距离
    int db_id = db_start + tid;
    if (db_id < nb) {
        float dist = 0.0f;

        // 从共享内存读取查询，从全局内存读取数据库
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            float diff = query_cache[d] - database[db_id * DIM + d];
            dist += diff * diff;
        }

        distances[query_id * nb + db_id] = dist;
    }
}
```

**优化点**：
1. **共享内存**：查询向量被块内所有线程共享
2. **循环展开** (`#pragma unroll`)：减少循环开销
3. **合并访问**：线程连续访问 database，高效利用带宽

### 6.4 内存合并访问

**问题**：GPU 内存访问是以 128 字节为单位的

```cuda
// 坏：跨步访问
__global__ void bad_access(float* data, int stride) {
    int i = threadIdx.x;
    float x = data[i * stride];  // 线程 0 访问 0, 线程 1 访问 stride...
    // 如果 stride 大，每个线程访问不同的内存段，效率低！
}

// 好：连续访问
__global__ void good_access(float* data) {
    int i = threadIdx.x;
    float x = data[i];  // 线程 0 访问 0, 线程 1 访问 1, ...
    // 32 个线程的访问被合并为一次内存事务！
}
```

**性能差异**：合并访问可以快 10-100 倍

**Faiss 的转置技巧**：

```cuda
// 原始布局：[nb, d] - 每个向量连续存储
// 查询时：每个线程访问不同向量的同一维度 → 跨步访问

// 转置布局：[d, nb] - 每个维度连续存储
// 查询时：每个线程访问连续位置 → 合并访问！

__global__ void transpose(float* out, const float* in, int nb, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nb && j < d) {
        out[j * nb + i] = in[i * d + j];  // 转置
    }
}
```

### 6.5 GPU 上的 IVF 搜索

位置：`faiss/gpu/GpuIndexIVF.cu`

```cuda
// GPU IVF 搜索流程
void GpuIndexIVF::search(...) {
    // 1. 量化查询（找最近的聚类中心）
    //    使用矩阵乘法加速：Q × C^T
    cublasGemmEx(handle,
                 ...,
                 queries, centroids, coarse_distances);

    // 2. Top-K 选择（选出 nprobe 个最近聚类）
    //    使用并行归约 + 堆
    topK<<<blocks, threads>>>(
        coarse_distances, nq, nlist, nprobe, selected_lists);

    // 3. 收集倒排表
    //    并行复制选中的倒排表到连续内存
    gather_invlists<<<...>>>(
        selected_lists, invlists, gathered_codes, gathered_ids);

    // 4. 计算精细距离
    //    每个查询并行处理其 nprobe 个倒排表
    compute_distances<<<nq, 256>>>(
        queries, gathered_codes, gathered_ids, distances);

    // 5. 最终 Top-K 选择
    topK<<<...>>>(distances, nq, total_candidates, k,
                  final_distances, final_labels);
}
```

**关键优化**：
1. **批量处理**：同时处理多个查询
2. **cuBLAS 加速**：量化步骤使用高度优化的矩阵乘法
3. **Kernel Fusion**：尽量减少 Kernel 调用次数

### 6.6 GPU 上的 Top-K 选择

**挑战**：GPU 上实现高效的 Top-K 很困难

**策略 1：Block-wise Heap**
```cuda
__global__ void topk_block(float* dists, int* ids, int n, int k) {
    __shared__ float heap_vals[K];
    __shared__ int heap_ids[K];

    // 初始化堆
    if (threadIdx.x < k) {
        heap_vals[threadIdx.x] = FLT_MAX;
    }
    __syncthreads();

    // 每个线程处理一部分数据
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = dists[i];

        // 原子操作更新堆顶
        if (val < heap_vals[0]) {
            atomicExch(&heap_vals[0], val);
            atomicExch(&heap_ids[0], i);
            // 重新堆化（复杂）
        }
    }
}
```

**策略 2：Radix Select（Faiss 使用）**
```cuda
// 使用基数选择算法
// 原理：类似快速选择，但使用位操作
__global__ void radix_topk(float* dists, int n, int k) {
    // 1. 找到第 k 大的值（pivot）
    float pivot = radix_select(dists, n, k);

    // 2. 分区：小于 pivot 的放前面
    partition(dists, n, pivot);

    // 3. 对前 k 个排序
    sort_k(dists, k);
}
```

**性能**：
- Block-wise Heap：适用于小 K（K < 100）
- Radix Select：适用于大 K（K > 100）

### 6.7 多 GPU 并行

```cpp
// Faiss 的多 GPU 索引
std::vector<int> gpus = {0, 1, 2, 3};  // 使用 4 个 GPU
GpuMultipleClonerOptions options;
options.shard = true;  // 数据分片

// 将索引复制到多个 GPU
auto gpu_index = index_cpu_to_gpu_multiple(
    &res, gpus, cpu_index, &options);

// 搜索时自动并行
gpu_index->search(nq, queries, k, distances, labels);
// 内部：
//   1. 每个 GPU 搜索自己的分片
//   2. 合并结果
```

**策略**：
- **复制模式**：每个 GPU 有完整副本，适合多查询
- **分片模式**：数据分片到多个 GPU，适合大数据集

### 6.8 实战练习

**练习 1**：实现 GPU 向量加法
```cuda
__global__ void vector_add_kernel(...) {
    // TODO: 实现
}

void vector_add(float* c, const float* a, const float* b, int n) {
    // TODO: 内存分配、数据传输、内核启动
}
```

**练习 2**：使用共享内存优化矩阵乘法
```cuda
__global__ void matmul_shared(float* C, const float* A, const float* B, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // TODO: 实现分块矩阵乘法
}
```

**练习 3**：实现 GPU 归约
```cuda
__global__ void reduce_sum(float* result, const float* data, int n) {
    __shared__ float shared_data[256];

    // TODO: 实现并行归约
}
```

---

## 第七天：高级优化技巧与工程实践

### 7.1 查找表优化：Product Quantization

**PQ 原理**：将高维向量压缩为短码

```
原始向量 (128 维, 512 字节):
  [0.1, 0.5, ..., 0.3]

PQ 编码 (8 字节):
  子向量 1 (16 维) → 码 0: 23
  子向量 2 (16 维) → 码 1: 157
  ...
  子向量 8 (16 维) → 码 7: 89

压缩比: 512 / 8 = 64x
```

**距离计算**：使用查找表（LUT）

```cpp
// 预计算查找表
void compute_LUT(const float* query, float* lut,
                 const float* codebooks, int M, int K, int d) {
    int d_sub = d / M;

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            // lut[m][k] = ||query_m - codebook_m[k]||^2
            const float* query_sub = query + m * d_sub;
            const float* centroid = codebooks + (m * K + k) * d_sub;

            lut[m * K + k] = fvec_L2sqr(query_sub, centroid, d_sub);
        }
    }
}

// 使用查找表计算距离
float compute_PQ_distance(const uint8_t* code, const float* lut, int M) {
    float dist = 0;
    for (int m = 0; m < M; m++) {
        dist += lut[m * 256 + code[m]];  // 简单的表查找！
    }
    return dist;
}
```

**复杂度对比**：

| 方法 | 计算量 | 内存访问 |
|------|--------|---------|
| 精确计算 | O(d) 乘加 | O(d) 浮点数 |
| PQ 查找表 | O(M) 加法 | O(M) 字节 |
| 加速比 | ~10x | ~50x |

### 7.2 PQ4: 4-bit 量化

**优化思路**：用 4 bit 代替 8 bit

```cpp
// 标准 PQ: 每个码 8 bit (0-255)
uint8_t code[M];

// PQ4: 每个码 4 bit (0-15)
uint8_t code_packed[M/2];  // 两个 4-bit 码打包到一个字节

// 打包
code_packed[i/2] = (code[i] & 0xF) | ((code[i+1] & 0xF) << 4);

// 解包
uint8_t c0 = code_packed[i/2] & 0xF;
uint8_t c1 = code_packed[i/2] >> 4;
```

**SIMD 优化：PSHUFB（Packed Shuffle）**

```cpp
// 使用 PSHUFB 同时查找 32 个 4-bit 码
__m256i pq4_lookup_AVX2(const uint8_t* codes, const uint8_t* lut) {
    // codes: [c0|c1, c2|c3, ..., c30|c31] (16 字节)
    __m128i codes_vec = _mm_loadu_si128((__m128i*)codes);

    // lut: 16 个距离值
    __m256i lut_vec = _mm256_loadu_si256((__m256i*)lut);

    // 分离高低 4 位
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i codes_lo = _mm_and_si128(codes_vec, mask);
    __m128i codes_hi = _mm_srli_epi16(codes_vec, 4);
    codes_hi = _mm_and_si128(codes_hi, mask);

    // 使用 PSHUFB 查表（核心！）
    __m256i codes_lo_256 = _mm256_cvtepu8_epi16(codes_lo);
    __m256i codes_hi_256 = _mm256_cvtepu8_epi16(codes_hi);

    __m256i result_lo = _mm256_shuffle_epi8(lut_vec, codes_lo_256);
    __m256i result_hi = _mm256_shuffle_epi8(lut_vec, codes_hi_256);

    // 交错结果
    return _mm256_unpacklo_epi8(result_lo, result_hi);
}
```

**PSHUFB 工作原理**：
```
lut  = [d0, d1, d2, ..., d15]  (16 个距离值)
code = [3, 7, 1, 15, ...]       (4-bit 索引)

PSHUFB:
  result[0] = lut[code[0]] = lut[3]  = d3
  result[1] = lut[code[1]] = lut[7]  = d7
  ...

一条指令查找 16 个值！
```

### 7.3 数据对齐与填充

**问题**：SIMD 要求数据对齐和长度是向量长度的倍数

```cpp
// 坏：维度不是 8 的倍数
const int d = 127;
float vector[127];

// SIMD 处理：最后 7 个元素无法向量化
for (int i = 0; i < 127; i += 8) {
    // ...前 120 个元素
}
// 尾部 7 个元素用标量处理（慢）

// 好：填充到 8 的倍数
const int d = 127;
const int d_padded = 128;
float vector[128];  // 多分配一个元素
vector[127] = 0.0f; // 填充 0

// 完全向量化！
for (int i = 0; i < 128; i += 8) {
    // ...
}
```

**Faiss 的做法**：

```cpp
// 在 Add Quantizer 中自动填充
struct AdditiveQuantizer {
    size_t M;        // 实际子量化器数
    size_t M_padded; // 向上取整到偶数

    void train(const float* x, size_t d) {
        M_padded = (M + 1) & ~1;  // 向上取整到偶数

        codebooks.resize(M_padded * K * dsub);
        // 多余的子量化器用 0 填充
    }
};
```

### 7.4 预计算与懒计算

**预计算**：用空间换时间

```cpp
struct IndexIVFPQ {
    // 预计算：向量的 L2 范数
    std::vector<float> precomputed_norms;  // [ntotal]

    void add(idx_t n, const float* x) {
        // 添加时计算范数
        precomputed_norms.reserve(ntotal + n);
        for (idx_t i = 0; i < n; i++) {
            float norm = fvec_norm_L2sqr(x + i * d, d);
            precomputed_norms.push_back(norm);
        }
        // 添加向量...
    }

    void search(idx_t nq, const float* x, ...) {
        // 搜索时直接使用预计算的范数
        // ||x - y||^2 = ||x||^2 + ||y||^2 - 2⟨x, y⟩
        //               ↑已知    ↑预计算   ↑只需计算这个
    }
};
```

**懒计算**：用时间换空间

```cpp
struct IndexHNSW {
    // 不预计算所有距离，按需计算
    DistanceComputer* get_distance_computer() {
        return new L2DistanceComputer(vectors, d);
    }

    void search_layer(...) {
        auto dc = get_distance_computer();

        for (auto candidate : candidates) {
            // 只在需要时计算距离
            float dist = (*dc)(candidate);
            if (dist < threshold) {
                // ...
            }
        }
    }
};
```

**权衡**：
- 预计算：适合频繁访问、计算量大的数据
- 懒计算：适合访问稀疏、内存受限的场景

### 7.5 批处理（Batching）

**原理**：批量处理可以提高吞吐量

```cpp
// 坏：逐个处理
for (int i = 0; i < nq; i++) {
    index.search(1, queries + i * d, k,
                 distances + i * k, labels + i * k);
}
// 问题：每次调用都有开销（函数调用、内存分配、同步等）

// 好：批量处理
index.search(nq, queries, k, distances, labels);
// 优势：
//   1. 摊销固定开销
//   2. 更好的并行性
//   3. 更好的缓存局部性
```

**性能对比**：

| 批大小 | 吞吐量 (QPS) | 延迟 (ms) |
|--------|-------------|----------|
| 1 | 1000 | 1.0 |
| 10 | 8000 | 1.25 |
| 100 | 50000 | 2.0 |
| 1000 | 200000 | 5.0 |

### 7.6 性能分析工具

**Linux Perf**：

```bash
# 采样分析
perf record -g ./your_program
perf report

# 缓存性能
perf stat -e cache-misses,cache-references ./your_program

# 分支预测
perf stat -e branches,branch-misses ./your_program
```

**Intel VTune**：
- 图形化界面
- 热点分析
- 微架构分析（查看 CPU 流水线瓶颈）

**NVIDIA Nsight**：
- GPU 性能分析
- Kernel 执行时间
- 内存带宽利用率

**代码中的性能计数器**：

```cpp
#include <chrono>

class Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start_;

public:
    Timer() : start_(clock::now()) {}

    double elapsed() {
        auto end = clock::now();
        return std::chrono::duration<double>(end - start_).count();
    }
};

// 使用：
Timer timer;
index.search(nq, queries, k, distances, labels);
double seconds = timer.elapsed();
printf("QPS: %.2f\n", nq / seconds);
```

### 7.7 数值精度与近似

**问题**：浮点运算有精度损失

```cpp
// 朴素求和：精度差
float sum = 0;
for (int i = 0; i < n; i++) {
    sum += data[i];  // 小数累加到大数，精度损失
}

// Kahan 求和：精度好
float sum = 0;
float c = 0;  // 补偿项
for (int i = 0; i < n; i++) {
    float y = data[i] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}
```

**Faiss 的选择**：在大多数场景下，精度损失可以接受
- 距离是近似的（PQ 本身就是近似）
- 只关心相对大小（排序），不关心绝对值

### 7.8 工程最佳实践

**1. 性能测试**

```cpp
// 使用 Google Benchmark
#include <benchmark/benchmark.h>

static void BM_IndexSearch(benchmark::State& state) {
    int d = state.range(0);
    int nq = state.range(1);

    // 准备数据
    IndexFlatL2 index(d);
    std::vector<float> queries(nq * d);
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    // 测量
    for (auto _ : state) {
        index.search(nq, queries.data(), k,
                    distances.data(), labels.data());
    }

    // 报告吞吐量
    state.SetItemsProcessed(state.iterations() * nq);
}

BENCHMARK(BM_IndexSearch)
    ->Args({128, 1})
    ->Args({128, 100})
    ->Args({128, 10000});
```

**2. 单元测试**

```cpp
// 验证优化的正确性
TEST(DistanceTest, SIMD_vs_Scalar) {
    const int d = 128;
    std::vector<float> x(d), y(d);

    // 生成随机数据
    for (int i = 0; i < d; i++) {
        x[i] = rand() / (float)RAND_MAX;
        y[i] = rand() / (float)RAND_MAX;
    }

    // 比较结果
    float dist_scalar = fvec_L2sqr_ref(x.data(), y.data(), d);
    float dist_simd = fvec_L2sqr(x.data(), y.data(), d);

    // 允许小的浮点误差
    EXPECT_NEAR(dist_scalar, dist_simd, 1e-5);
}
```

**3. 可配置的优化**

```cpp
struct SearchParameters {
    int nprobe = 10;          // 可调
    int max_codes = 0;        // 0 = 无限制
    bool use_precomputed = true;  // 是否使用预计算

    // 高级选项
    int num_threads = 0;      // 0 = 自动
    bool use_simd = true;     // 可禁用 SIMD（用于调试）
};

void search(const SearchParameters& params) {
    if (params.use_simd && cpu_supports_avx2()) {
        search_avx2(...);
    } else {
        search_scalar(...);
    }
}
```

**4. 渐进式优化**

```
1. 先写正确的代码（标量实现）
2. 添加单元测试
3. 性能分析，找到瓶颈
4. 优化热点（先最热的）
5. 验证正确性和性能
6. 重复 3-5
```

### 7.9 总结：性能优化检查清单

**算法层面**：
- [ ] 选择合适的算法（时间复杂度）
- [ ] 使用近似算法（在精度要求允许的情况下）
- [ ] 预计算可重用的数据
- [ ] 批量处理

**数据结构**：
- [ ] 内存对齐（32/64 字节）
- [ ] 缓存友好的布局（SOA vs AOS）
- [ ] 预分配内存，避免动态增长
- [ ] 使用专用的分配器

**并行化**：
- [ ] 多线程（OpenMP）
- [ ] GPU 加速
- [ ] SIMD 向量化

**底层优化**：
- [ ] 循环展开
- [ ] 分支预测友好的代码
- [ ] 避免伪共享
- [ ] 使用 FMA 等高级指令

**工程实践**：
- [ ] 性能测试
- [ ] 单元测试（验证正确性）
- [ ] 性能分析（找瓶颈）
- [ ] 可配置的优化选项

### 7.10 实战项目

**项目：实现一个简单的向量索引库**

要求：
1. 支持添加和搜索向量
2. 实现至少 3 种优化技术（如 SIMD、多线程、缓存优化）
3. 编写性能测试和单元测试
4. 性能至少达到朴素实现的 10 倍

参考实现框架：

```cpp
class OptimizedVectorIndex {
public:
    // 构造函数
    OptimizedVectorIndex(int d, MetricType metric);

    // 添加向量
    void add(int n, const float* vectors);

    // 搜索
    void search(int nq, const float* queries, int k,
                float* distances, int64_t* labels);

private:
    int d_;
    MetricType metric_;

    // TODO: 选择合适的数据结构
    // TODO: 实现优化的距离计算
    // TODO: 实现并行搜索
};
```

---

## 课程总结

通过这 7 天的学习，我们深入探讨了 Faiss 中使用的各种性能优化技术：

1. **第一天**：内存对齐、缓存友好的数据结构
2. **第二天**：SIMD 向量化编程
3. **第三天**：缓存优化与数据局部性
4. **第四天**：设计模式与代码抽象
5. **第五天**：并行计算与多线程
6. **第六天**：GPU 优化技术
7. **第七天**：高级优化与工程实践

**核心思想**：
- **了解硬件**：CPU/GPU 架构、缓存、SIMD
- **测量优化**：先分析，再优化
- **渐进式优化**：从简单到复杂
- **验证正确性**：单元测试必不可少

**继续学习资源**：
- Faiss 源码：https://github.com/facebookresearch/faiss
- Intel Intrinsics Guide：https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- CUDA Programming Guide：https://docs.nvidia.com/cuda/
- 《计算机体系结构：量化研究方法》
- 《深入理解计算机系统》（CSAPP）

祝你在性能优化的道路上不断进步！🚀
