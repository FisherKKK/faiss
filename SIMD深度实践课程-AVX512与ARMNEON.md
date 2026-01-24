# SIMD 深度实践课程 - AVX-512 与 ARM NEON

## 课程简介

本课程深入讲解现代 SIMD 指令集的实战应用，重点覆盖 Intel AVX-512 和 ARM NEON，结合 Faiss 向量搜索场景，提供完整的优化实践。

**前置知识**：
- 已完成《Faiss 性能优化技术课程》第 2 天（SIMD 基础）
- 熟悉 C++ 和基本的汇编语言
- 了解向量搜索的基本概念

**学习目标**：
- 掌握 AVX-512 指令集的完整使用
- 精通 ARM NEON 优化技术
- 能够为特定算法选择最优的 SIMD 策略
- 理解 SIMD 与向量搜索的结合

---

## 第一部分：AVX-512 深度解析

### 1.1 AVX-512 概览

**AVX-512 的优势**：
- 512 位寄存器 = 16 个 float 或 8 个 double
- 更多的寄存器（32 个 zmm 寄存器）
- 新增指令：掩码操作、融合指令、扩展指令
- 理论加速比：比 AVX2 快 2 倍

**支持的 CPU**：
- Intel Skylake-X 及更新（Core X 系列）
- Intel Cascade Lake（服务器）
- Intel Ice Lake（移动端）
- AMD Zen 4（部分支持）

**检测 AVX-512 支持**：

```cpp
#include <cpuid.h>
#include <iostream>

bool has_avx512f() {
    uint32_t eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);

    // 检查 AVX512F 基础指令集
    return (ebx & bit_AVX512F) != 0;
}

bool has_avx512_extensions() {
    uint32_t eax, ebx, ecx, edx;
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);

    bool has_dq = (ebx & bit_AVX512DQ) != 0;    // Double/Quadword
    bool has_vl = (ecx & bit_AVX512VL) != 0;    // Vector Length
    bool has_bf16 = (ecx & bit_AVX512BF16) != 0; // BFloat16
    bool has_vnni = (ecx & bit_AVX512_VNNI) != 0; // Vector Neural Network

    return has_dq || has_vl || has_bf16 || has_vnni;
}

int main() {
    if (has_avx512f()) {
        std::cout << "AVX-512 Foundation: YES" << std::endl;
        if (has_avx512_extensions()) {
            std::cout << "AVX-512 Extensions: YES" << std::endl;
        }
    } else {
        std::cout << "AVX-512: NOT SUPPORTED" << std::endl;
    }
    return 0;
}
```

### 1.2 AVX-512 基础指令

#### 1.2.1 数据加载和存储

```cpp
#include <immintrin.h>
#include <cstring>

// AVX-512 提供了三种加载方式

// 1. 对齐加载 (要求 64 字节对齐)
__m512 load_aligned(const float* ptr) {
    return _mm512_load_ps(ptr);  // 加载 16 个 float
}

// 2. 未对齐加载 (自动处理对齐)
__m512 load_unaligned(const float* ptr) {
    return _mm512_loadu_ps(ptr);  // 加载 16 个 float，无需对齐
}

// 3. 掩码加载 (条件加载)
__m512 load_masked(const float* ptr, __mmask16 mask) {
    return _mm512_maskz_load_ps(mask, ptr);
    // mask 的每一位控制是否加载对应的 float
    // mask=0xFFFF -> 全部加载
    // mask=0x00FF -> 只加载前 8 个
}

// 存储操作
void store_aligned(float* ptr, __m512 data) {
    _mm512_store_ps(ptr, data);  // 要求 64 字节对齐
}

void store_unaligned(float* ptr, __m512 data) {
    _mm512_storeu_ps(ptr, data);  // 无需对齐
}

void store_masked(float* ptr, __m512 data, __mmask16 mask) {
    _mm512_mask_store_ps(ptr, mask, data);
    // 只存储 mask 为 1 的位置
}

// 非临时存储（绕过缓存，直接写内存）
void stream_store(float* ptr, __m512 data) {
    _mm512_stream_ps(ptr, data);
    // 用于大块数据写入，避免污染缓存
}
```

**对齐内存分配**：

```cpp
// 分配 64 字节对齐的内存
float* allocate_aligned_avx512(size_t n) {
    void* ptr = nullptr;
    // posix_memalign 分配对齐内存
    int ret = posix_memalign(&ptr, 64, n * sizeof(float));
    if (ret != 0) {
        throw std::bad_alloc();
    }
    return static_cast<float*>(ptr);
}

// C++17 aligned_alloc
float* allocate_aligned_cpp17(size_t n) {
    return static_cast<float*>(aligned_alloc(64, n * sizeof(float)));
}

// 或使用 alignas
alignas(64) float array[1024];  // 编译时对齐
```

#### 1.2.2 算术运算

```cpp
// 基础算术运算
__m512 arithmetic_ops(__m512 a, __m512 b) {
    // 加法
    __m512 sum = _mm512_add_ps(a, b);

    // 减法
    __m512 diff = _mm512_sub_ps(a, b);

    // 乘法
    __m512 prod = _mm512_mul_ps(a, b);

    // 除法
    __m512 quot = _mm512_div_ps(a, b);

    // FMA: a * b + c (一条指令)
    __m512 fma = _mm512_fmadd_ps(a, b, sum);

    // FMA: a * b - c
    __m512 fms = _mm512_fmsub_ps(a, b, diff);

    // FMA: -(a * b) + c
    __m512 fnma = _mm512_fnmadd_ps(a, b, sum);

    // FMA: -(a * b) - c
    __m512 fnms = _mm512_fnmsub_ps(a, b, diff);

    return fma;
}

// 水平运算（归约）
float horizontal_sum(__m512 v) {
    // AVX-512 高效归约方法
    // 方法 1: 使用 vreduce
    return _mm512_reduce_add_ps(v);

    // 方法 2: 手动实现（更灵活）
    __m512 shuf = _mm512_permute_ps(v, _MM_SHUFFLE(0, 0, 0, 0));
    __m256 sum256 = _mm256_add_ps(
        _mm512_castps512_ps256(v),
        _mm512_castps512_ps256(shuf)
    );
    __m128 sum128 = _mm_add_ps(
        _mm256_castps256_ps128(sum256),
        _mm256_extractf128_ps(sum256, 1)
    );
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));

    return _mm_cvtss_f32(sum128);
}

// 最大值/最小值
__m512 min_max_ops(__m512 a, __m512 b) {
    // 每个位置的最大值
    __m512 max_vals = _mm512_max_ps(a, b);

    // 每个位置的最小值
    __m512 min_vals = _mm512_min_ps(a, b);

    // 找出整个向量中的最大值（标量）
    float max_scalar = _mm512_reduce_max_ps(a);

    // 找出整个向量中的最小值（标量）
    float min_scalar = _mm512_reduce_min_ps(a);

    return max_vals;
}
```

#### 1.2.3 逻辑和位运算

```cpp
// 逻辑运算（针对整数）
__m512i bitwise_ops(__m512i a, __m512i b) {
    // AND
    __m512i and_result = _mm512_and_epi32(a, b);

    // OR
    __m512i or_result = _mm512_or_epi32(a, b);

    // XOR
    __m512i xor_result = _mm512_xor_epi32(a, b);

    // AND NOT (NOT a AND b)
    __m512i andnot_result = _mm512_andnot_epi32(a, b);

    return and_result;
}

// 位运算用于条件选择
__m512 select_conditional(__m512 a, __m512 b, __m512 condition) {
    // 根据 condition 的符号位选择 a 或 b
    // condition < 0 ? a : b
    return _mm512_mask_blend_ps(
        _mm512_cmp_ps_mask(condition, _mm512_setzero_ps(), _CMP_LT_OQ),
        b, a
    );
}

// 更清晰的写法
__m512 if_else(__m512 a, __m512 b, __m512 condition, float threshold) {
    // condition > threshold ? a : b
    __mmask16 mask = _mm512_cmp_ps_mask(condition, _mm512_set1_ps(threshold), _CMP_GT_OQ);
    return _mm512_mask_blend_ps(mask, b, a);
}
```

### 1.3 AVX-512 掩码操作

掩码是 AVX-512 的核心特性之一，每个操作都可以带掩码。

```cpp
// 掩码基础
void mask_operations_demo() {
    // 16 位掩码（对应 16 个 float）
    __mmask16 mask1 = 0xFFFF;  // 全 1：全部有效
    __mmask16 mask2 = 0x00FF;  // 低 8 位有效
    __mmask16 mask3 = 0xFF00;  // 高 8 位有效

    // 创建掩码
    __m512 a = _mm512_set1_ps(5.0f);
    __m512 b = _mm512_set1_ps(3.0f);
    __m512 c = _mm512_set1_ps(2.0f);

    // 比较生成掩码
    __mmask16 cmp_mask = _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
    // 现在 cmp_mask 中，a > b 的位置是 1

    // 掩码加法：只在 mask=1 的位置计算
    __m512 result = _mm512_mask_add_ps(c, cmp_mask, a, b);
    // mask=1: result = a + b
    // mask=0: result = c

    // 掩码加载
    float data[16];
    __m512 loaded = _mm512_maskz_load_ps(cmp_mask, data);
    // 只加载 mask=1 的位置，其他位置为 0

    // 掩码存储
    float output[16];
    _mm512_mask_store_ps(output, cmp_mask, result);
    // 只存储 mask=1 的位置
}

// 实际应用：条件累加
float conditional_sum_avx512(const float* data, size_t n, float threshold) {
    __m512 sum_vec = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);

        // 生成掩码：v > threshold
        __mmask16 mask = _mm512_cmp_ps_mask(
            v, _mm512_set1_ps(threshold), _CMP_GT_OQ
        );

        // 掩码加法：只累加满足条件的
        sum_vec = _mm512_mask_add_ps(sum_vec, mask, sum_vec, v);
    }

    // 处理剩余元素
    float sum = _mm512_reduce_add_ps(sum_vec);
    for (; i < n; i++) {
        if (data[i] > threshold) {
            sum += data[i];
        }
    }

    return sum;
}
```

### 1.4 AVX-512 向量距离计算

结合 Faiss 场景，实现高效的向量距离计算。

```cpp
// L2 距离计算（AVX-512 版本）
float fvec_L2sqr_avx512(const float* x, const float* y, size_t d) {
    __m512 sum_vec = _mm512_setzero_ps();

    size_t i = 0;

    // 主循环：每次处理 16 个元素
    for (; i + 15 < d; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(&x[i]);
        __m512 y_vec = _mm512_loadu_ps(&y[i]);

        // diff = x - y
        __m512 diff = _mm512_sub_ps(x_vec, y_vec);

        // sum += diff * diff (FMA)
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
    }

    // 水平归约
    float sum = _mm512_reduce_add_ps(sum_vec);

    // 处理剩余元素
    for (; i < d; i++) {
        float diff = x[i] - y[i];
        sum += diff * diff;
    }

    return sum;
}

// 内积计算（AVX-512 版本）
float fvec_inner_product_avx512(const float* x, const float* y, size_t d) {
    __m512 sum_vec = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 15 < d; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(&x[i]);
        __m512 y_vec = _mm512_loadu_ps(&y[i]);

        // FMA: sum += x * y
        sum_vec = _mm512_fmadd_ps(x_vec, y_vec, sum_vec);
    }

    float sum = _mm512_reduce_add_ps(sum_vec);

    for (; i < d; i++) {
        sum += x[i] * y[i];
    }

    return sum;
}

// 批量 L2 距离计算（1 个查询 vs N 个向量）
// 这是最常用的场景
void fvec_L2sqr_ny_avx512(
        float* distances,
        const float* x,        // 查询向量 [d]
        const float* y,        // 数据库向量 [n, d]（转置存储）
        size_t d,
        size_t n) {

    // 计算查询向量的平方长度
    float x_norm = 0;
    for (size_t j = 0; j < d; j++) {
        x_norm += x[j] * x[j];
    }

    __m512 x_norm_vec = _mm512_set1_ps(x_norm);

    // 每次处理 16 个向量
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 dot_products = _mm512_setzero_ps();
        __m512 y_norms = _mm512_setzero_ps();

        // 计算点积和 y 的平方长度
        for (size_t j = 0; j < d; j++) {
            __m512 x_j = _mm512_set1_ps(x[j]);

            // 加载 16 个向量的第 j 个维度
            __m512 y_j = _mm512_loadu_ps(&y[j * n + i]);

            // 点积累加
            dot_products = _mm512_fmadd_ps(x_j, y_j, dot_products);

            // y_norm 累加
            y_norms = _mm512_fmadd_ps(y_j, y_j, y_norms);
        }

        // 最终距离：||x||² + ||y||² - 2⟨x, y⟩
        __m512 distances_vec = _mm512_add_ps(x_norm_vec, y_norms);
        __m512 two_dot = _mm512_mul_ps(dot_products, _mm512_set1_ps(2.0f));
        distances_vec = _mm512_sub_ps(distances_vec, two_dot);

        _mm512_storeu_ps(&distances[i], distances_vec);
    }

    // 处理剩余向量
    for (; i < n; i++) {
        float dot = 0;
        float y_norm = 0;
        for (size_t j = 0; j < d; j++) {
            float x_j = x[j];
            float y_j = y[j * n + i];
            dot += x_j * y_j;
            y_norm += y_j * y_j;
        }
        distances[i] = x_norm + y_norm - 2 * dot;
    }
}
```

### 1.5 AVX-512 高级技巧

#### 1.5.1 压缩和展开

```cpp
// 压缩：根据掩码选择元素
void compress_demo() {
    __m512 v = _mm512_set_ps(
        15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0
    );

    // 压缩偶数位置
    __mmask16 mask = 0x5555;  // 0101 0101 0101 0101
    __m512 compressed = _mm512_mask_compress_ps(
        _mm512_setzero_ps(),
        mask,
        v
    );
    // 结果：[0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0]

    // 展开：将输入分散到目标位置
    __m512 expanded = _mm512_mask_expand_ps(
        _mm512_setzero_ps(),
        mask,
        compressed
    );
}

// 应用：过滤无效结果
void filter_results_avx512(
        float* results,
        const float* input,
        const float* valid_mask,
        size_t n) {

    size_t i = 0;
    int write_idx = 0;

    for (; i + 15 < n; i += 16) {
        __m512 data = _mm512_loadu_ps(&input[i]);
        __m512 mask_data = _mm512_loadu_ps(&valid_mask[i]);

        // 生成掩码：valid_mask != 0
        __mmask16 mask = _mm512_cmpneq_ps_mask(
            mask_data,
            _mm512_setzero_ps()
        );

        // 计算有效元素数量
        int count = _mm_popcnt_u32(mask);

        if (count > 0) {
            // 压缩有效元素
            __m512 compressed = _mm512_maskz_compress_ps(mask, data);

            // 存储到结果
            // 注意：需要处理跨边界情况
            if (write_idx + 16 <= n) {
                _mm512_storeu_ps(&results[write_idx], compressed);
            } else {
                // 使用掩码存储
                __mmask16 store_mask = (0xFFFF >> (16 - count));
                _mm512_mask_storeu_ps(&results[write_idx], store_mask, compressed);
            }

            write_idx += count;
        }
    }
}
```

#### 1.5.2 排列和重排

```cpp
// 排列操作用于优化访问模式
void permute_demo() {
    __m512 v = _mm512_set_ps(
        15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0
    );

    // 反转元素顺序
    __m512 reversed = _mm512_permutex_ps(v, _MM_SHUFFLE(0, 1, 2, 3));

    // 交换相邻元素
    __m512 swapped = _mm512_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));

    // 用于矩阵转置或其他模式转换
}

// 应用：4x4 矩阵转置（16 个 float）
void transpose_4x4_avx512(float* m) {
    __m512 r0 = _mm512_loadu_ps(&m[0]);   // [00, 01, 02, 03, ...]
    __m512 r1 = _mm512_loadu_ps(&m[16]);  // [10, 11, 12, 13, ...]
    __m512 r2 = _mm512_loadu_ps(&m[32]);
    __m512 r3 = _mm512_loadu_ps(&m[48]);

    // 使用 shuffle 和 permute 实现转置
    __m512 t0 = _mm512_unpacklo_ps(r0, r1);
    __m512 t1 = _mm512_unpackhi_ps(r0, r1);
    __m512 t2 = _mm512_unpacklo_ps(r2, r3);
    __m512 t3 = _mm512_unpackhi_ps(r2, r3);

    __m512 m0 = _mm512_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 m1 = _mm512_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 m2 = _mm512_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 m3 = _mm512_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));

    _mm512_storeu_ps(&m[0], m0);
    _mm512_storeu_ps(&m[16], m1);
    _mm512_storeu_ps(&m[32], m2);
    _mm512_storeu_ps(&m[48], m3);
}
```

### 1.6 AVX-512 性能优化技巧

#### 1.6.1 避免延迟链

```cpp
// 反例：依赖链
__m512 bad_reduction(__m512* data, size_t n) {
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < n; i++) {
        sum = _mm512_add_ps(sum, data[i]);  // 每次依赖前一次
    }
    return sum;
}

// 优化：打破依赖链（多累加器）
__m512 good_reduction(__m512* data, size_t n) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        sum0 = _mm512_add_ps(sum0, data[i]);
        sum1 = _mm512_add_ps(sum1, data[i + 1]);
        sum2 = _mm512_add_ps(sum2, data[i + 2]);
        sum3 = _mm512_add_ps(sum3, data[i + 3]);
    }

    // 最后合并
    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);

    // 处理剩余
    for (; i < n; i++) {
        sum0 = _mm512_add_ps(sum0, data[i]);
    }

    return sum0;
}
```

#### 1.6.2 预取优化

```cpp
// 手动预取下一个缓存行
void compute_with_prefetch(const float* x, const float* y, float* result, size_t n) {
    const size_t PREFETCH_DISTANCE = 8;  // 预取距离

    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        // 预取未来的数据
        if (i + PREFETCH_DISTANCE * 16 < n) {
            _mm_prefetch((char*)&x[i + PREFETCH_DISTANCE * 16], _MM_HINT_T0);
            _mm_prefetch((char*)&y[i + PREFETCH_DISTANCE * 16], _MM_HINT_T0);
        }

        __m512 xv = _mm512_loadu_ps(&x[i]);
        __m512 yv = _mm512_loadu_ps(&y[i]);
        __m512 rv = _mm512_mul_ps(xv, yv);
        _mm512_storeu_ps(&result[i], rv);
    }
}
```

---

## 第二部分：ARM NEON 深度解析

### 2.1 ARM NEON 概览

**ARM NEON** 是 ARM 架构的 SIMD 指令集，广泛用于移动设备和嵌入式系统。

**NEON 寄存器**：
- 128 位寄存器（32 个 q0-q31）
- 可以看作 16 个 8 位、8 个 16 位、4 个 32 位或 2 个 64 位元素

**NEON 数据类型**：
```
float32_t: 4 x float
float64_t: 2 x double
int8_t, uint8_t: 16 x byte
int16_t, uint16_t: 8 x halfword
int32_t, uint32_t: 4 x word
int64_t, uint64_t: 2 x doubleword
```

**检测 NEON 支持**：

```cpp
#if defined(__aarch64__) || defined(__ARM_NEON)
    #include <arm_neon.h>
    #define HAS_NEON 1
#else
    #define HAS_NEON 0
    #warning "NEON not supported"
#endif
```

### 2.2 NEON 基础指令

#### 2.2.1 数据加载和存储

```cpp
#include <arm_neon.h>

// 加载操作
void neon_load_demo() {
    // 对齐加载（要求 16 字节对齐）
    float32_t data[4] __attribute__((aligned(16))) = {1, 2, 3, 4};
    float32x4_t v1 = vld1q_f32(data);  // 加载 4 个 float

    // 未对齐加载
    float32x4_t v2 = vld1q_f32(data);  // NEON 没有严格的 u 版本

    // 加载并设置所有元素为相同值
    float32x4_t v3 = vdupq_n_f32(5.0f);  // [5, 5, 5, 5]

    // 从内存加载到不同寄存器（跨步加载）
    float32x4_t v4 = vld1q_lane_f32(data, v1, 0);  // 加载单个元素

    // 加载并 dup
    float32x4_t v5 = vdupq_laneq_f32(v1, 1);  // 复制 lane 1 到所有位置
}

// 存储操作
void neon_store_demo() {
    float32_t result[4] __attribute__((aligned(16)));

    float32x4_t v = vdupq_n_f32(3.14f);

    // 存储
    vst1q_f32(result, v);  // 存储 4 个 float

    // 存储单个 lane
    vst1q_lane_f32(result, v, 2);  // 只存储 lane 2

    // 存储到多个不连续位置
    float32_t a[4], b[4];
    vst2q_f32(a, b, v);  // 交错存储到两个数组
}
```

#### 2.2.2 算术运算

```cpp
// 基础算术
float32x4_t neon_arithmetic(float32x4_t a, float32x4_t b) {
    // 加法
    float32x4_t sum = vaddq_f32(a, b);

    // 减法
    float32x4_t diff = vsubq_f32(a, b);

    // 乘法
    float32x4_t prod = vmulq_f32(a, b);

    // 融合乘加（如果支持）
    #ifdef __ARM_FEATURE_FMA
    float32x4_t fma = vfmaq_f32(a, b, sum);  // a + b * sum
    #endif

    // 除法（NEON 没有硬件除法，使用倒数近似）
    float32x4_t inv_b = vrecpeq_f32(b);  // 倒数近似
    float32x4_t quot = vmulq_f32(a, inv_b);  // a / b ≈ a * (1/b)

    // 平方根
    float32x4_t sqrt = vsqrtq_f32(a);

    // 倒数平方根
    float32x4_t rsqrt = vrsqrteq_f32(a);

    return fma;
}

// 最大值/最小值
float32x4_t neon_min_max(float32x4_t a, float32x4_t b) {
    float32x4_t max_vals = vmaxq_f32(a, b);
    float32x4_t min_vals = vminq_f32(a, b);

    // 水平最大值
    float max_scalar = vmaxvq_f32(a);

    // 水平最小值
    float min_scalar = vminvq_f32(a);

    return max_vals;
}

// 水平归约
float horizontal_sum_neon(float32x4_t v) {
    // 方法 1: 使用 vaddvq (如果支持)
    #ifdef __ARM_FEATURE_DOTPROD
    return vaddvq_f32(v);
    #endif

    // 方法 2: 手动实现
    float32x2_t sum = vadd_f32(
        vget_low_f32(v),
        vget_high_f32(v)
    );
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
}
```

### 2.3 NEON 向量距离计算

```cpp
// L2 距离（NEON 版本）
float fvec_L2sqr_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 3 < d; i += 4) {
        float32x4_t x_vec = vld1q_f32(&x[i]);
        float32x4_t y_vec = vld1q_f32(&y[i]);

        // diff = x - y
        float32x4_t diff = vsubq_f32(x_vec, y_vec);

        // sum += diff * diff
        sum_vec = vmlaq_f32(sum_vec, diff, diff);  // FMA: sum + diff * diff
    }

    // 水平归约
    float sum = horizontal_sum_neon(sum_vec);

    // 处理剩余
    for (; i < d; i++) {
        float diff = x[i] - y[i];
        sum += diff * diff;
    }

    return sum;
}

// 内积（NEON 版本）
float fvec_inner_product_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 3 < d; i += 4) {
        float32x4_t x_vec = vld1q_f32(&x[i]);
        float32x4_t y_vec = vld1q_f32(&y[i]);

        sum_vec = vmlaq_f32(sum_vec, x_vec, y_vec);
    }

    float sum = horizontal_sum_neon(sum_vec);

    for (; i < d; i++) {
        sum += x[i] * y[i];
    }

    return sum;
}

// 批量处理：4 个查询向量同时计算
void fvec_inner_product_4_neon(
        float* distances,
        const float* x,  // [4, d] - 4 个查询向量
        const float* y,  // [d, n] - n 个数据库向量（转置）
        size_t d,
        size_t n) {

    for (size_t j = 0; j < n; j++) {
        float32x4_t sums[4] = {
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f)
        };

        // 计算内积
        for (size_t k = 0; k < d; k++) {
            // 加载 4 个查询的第 k 维
            float32x4_t xk = vld1q_f32(&x[k * 4]);

            // 加载 n 个数据库向量的第 k 维（每次 4 个）
            float y_k = y[k * n + j];
            float32x4_t y_vec = vdupq_n_f32(y_k);

            // 累加
            sums[0] = vmlaq_f32(sums[0], xk, y_vec);
        }

        // 水平归约
        distances[j] = horizontal_sum_neon(sums[0]);
    }
}
```

### 2.4 NEON 优化技巧

#### 2.4.1 后处理指令（利用空闲周期）

```cpp
// NEON 后置指令可以不消耗周期
void neon_postcheduling_demo() {
    float32x4_t a = vdupq_n_f32(1.0f);
    float32x4_t b = vdupq_n_f32(2.0f);
    float32x4_t c = vdupq_n_f32(3.0f);

    // 主指令
    float32x4_t result = vfmaq_f32(c, a, b);

    // 后置指令（几乎不消耗周期）
    float32x4_t d = vabsq_f32(result);       // 绝对值
    float32x4_t e = vnegq_f32(d);            // 取反
    float32x4_t f = vtrn1q_f32(a, b);        // 转置
    float32x4_t g = vuzp1q_f32(a, b);        // 解交错

    // 这些指令与主指令并行执行
}
```

#### 2.4.2 多向量处理

```cpp
// 一次处理多个向量以提高吞吐量
void process_multiple_vectors(
        const float* input,
        float* output,
        size_t n) {

    size_t i = 0;

    // 一次处理 4 个向量（16 个元素）
    for (; i + 15 < n; i += 16) {
        float32x4_t v0 = vld1q_f32(&input[i]);
        float32x4_t v1 = vld1q_f32(&input[i + 4]);
        float32x4_t v2 = vld1q_f32(&input[i + 8]);
        float32x4_t v3 = vld1q_f32(&input[i + 12]);

        // 并行处理
        v0 = vmulq_n_f32(v0, 2.0f);
        v1 = vmulq_n_f32(v1, 2.0f);
        v2 = vmulq_n_f32(v2, 2.0f);
        v3 = vmulq_n_f32(v3, 2.0f);

        vst1q_f32(&output[i], v0);
        vst1q_f32(&output[i + 4], v1);
        vst1q_f32(&output[i + 8], v2);
        vst1q_f32(&output[i + 12], v3);
    }

    // 处理剩余
    for (; i < n; i++) {
        output[i] = input[i] * 2.0f;
    }
}
```

---

## 第三部分：SIMD 选择策略

### 3.1 运行时检测和分派

```cpp
// 跨平台的 SIMD 分发器
class VectorDistance {
public:
    using DistanceFunc = float(*)(const float*, const float*, size_t);

    static DistanceFunc get_optimal_func() {
        if (has_avx512f()) {
            return fvec_L2sqr_avx512;
        } else if (has_avx2()) {
            return fvec_L2sqr_avx2;
        } else if (has_neon()) {
            return fvec_L2sqr_neon;
        } else {
            return fvec_L2sqr_ref;
        }
    }

    static bool has_avx512f() {
        #ifdef __AVX512F__
        return true;
        #else
        return false;
        #endif
    }

    static bool has_avx2() {
        #ifdef __AVX2__
        return true;
        #else
        return false;
        #endif
    }

    static bool has_neon() {
        #if defined(__aarch64__) || defined(__ARM_NEON)
        return true;
        #else
        return false;
        #endif
    }

    // 参考实现
    static float fvec_L2sqr_ref(const float* x, const float* y, size_t d) {
        float sum = 0;
        for (size_t i = 0; i < d; i++) {
            float diff = x[i] - y[i];
            sum += diff * diff;
        }
        return sum;
    }
};

// 使用
void benchmark_distances() {
    auto func = VectorDistance::get_optimal_func();

    const int d = 128;
    float x[d], y[d];
    // ... 初始化 ...

    float dist = func(x, y, d);
    printf("Distance: %f\n", dist);
}
```

### 3.2 性能对比

```cpp
#include <chrono>

void benchmark_all_implementations() {
    const int d = 128;
    const int iterations = 1000000;

    float x[d], y[d];
    for (int i = 0; i < d; i++) {
        x[i] = (float)i / d;
        y[i] = (float)(d - i) / d;
    }

    // 测试参考实现
    auto start = std::chrono::high_resolution_clock::now();
    float ref_result = 0;
    for (int i = 0; i < iterations; i++) {
        ref_result = VectorDistance::fvec_L2sqr_ref(x, y, d);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ref_time = std::chrono::duration<double>(end - start).count();

    // 测试 AVX2
    #ifdef __AVX2__
    start = std::chrono::high_resolution_clock::now();
    float avx2_result = 0;
    for (int i = 0; i < iterations; i++) {
        avx2_result = fvec_L2sqr_avx2(x, y, d);
    }
    end = std::chrono::high_resolution_clock::now();
    double avx2_time = std::chrono::duration<double>(end - start).count();
    #endif

    // 测试 AVX-512
    #ifdef __AVX512F__
    start = std::chrono::high_resolution_clock::now();
    float avx512_result = 0;
    for (int i = 0; i < iterations; i++) {
        avx512_result = fvec_L2sqr_avx512(x, y, d);
    }
    end = std::chrono::high_resolution_clock::now();
    double avx512_time = std::chrono::duration<double>(end - start).count();
    #endif

    // 测试 NEON
    #if defined(__aarch64__) || defined(__ARM_NEON)
    start = std::chrono::high_resolution_clock::now();
    float neon_result = 0;
    for (int i = 0; i < iterations; i++) {
        neon_result = fvec_L2sqr_neon(x, y, d);
    }
    end = std::chrono::high_resolution_clock::now();
    double neon_time = std::chrono::duration<double>(end - start).count();
    #endif

    // 输出结果
    printf("Results: ref=%.3f", ref_result);
    #ifdef __AVX2__
    printf(" avx2=%.3f", avx2_result);
    #endif
    #ifdef __AVX512F__
    printf(" avx512=%.3f", avx512_result);
    #endif
    #if defined(__aarch64__) || defined(__ARM_NEON)
    printf(" neon=%.3f", neon_result);
    #endif
    printf("\n");

    printf("Performance (MOPS/s):\n");
    printf("  Reference: %.2f\n", iterations / ref_time / 1e6);
    #ifdef __AVX2__
    printf("  AVX2:      %.2f (%.2fx)\n", iterations / avx2_time / 1e6, ref_time / avx2_time);
    #endif
    #ifdef __AVX512F__
    printf("  AVX-512:   %.2f (%.2fx)\n", iterations / avx512_time / 1e6, ref_time / avx512_time);
    #endif
    #if defined(__aarch64__) || defined(__ARM_NEON)
    printf("  NEON:      %.2f (%.2fx)\n", iterations / neon_time / 1e6, ref_time / neon_time);
    #endif
}
```

---

## 第四部分：实战练习

### 练习 1：实现 AVX-512 版本的 Hamming 距离

```cpp
// Hamming 距离用于二进制向量
// 实现：计算两个 512 位二进制向量的 Hamming 距离
// 要求：使用 AVX-512 指令

int hamming_distance_avx512(const uint8_t* a, const uint8_t* b, size_t n) {
    // TODO: 实现
    // 提示：
    // 1. 使用 _mm512_loadu_si512 加载 64 字节
    // 2. 使用 _mm512_xor_si512 计算 XOR
    // 3. 使用 _mm512_popcnt_epi64 统计 1 的个数
    // 4. 使用 _mm512_reduce_add_epi64 归约

    return 0;
}
```

### 练习 2：实现 NEON 版本的 Top-K 选择

```cpp
// 从 n 个距离中找出最小的 k 个
// 使用 NEON 优化

void topk_neon(const float* distances, int64_t* labels, int n, int k) {
    // TODO: 实现
    // 提示：
    // 1. 使用 vminq_f32 找最小值
    // 2. 维护一个大小为 k 的堆
    // 3. 使用 NEON 批量比较和更新
}
```

### 练习 3：对比不同 SIMD 实现的性能

```cpp
// 实现完整的性能测试框架
// 对比：Scalar vs AVX2 vs AVX-512 vs NEON
// 测试内容：批量向量距离计算

void simd_benchmark_suite() {
    // TODO: 实现
    // 测试不同维度（64, 128, 256, 512）
    // 测试不同批量大小（1, 10, 100, 1000）
    // 输出性能对比表格
}
```

---

## 附录：快速参考

### AVX-512 常用指令速查

```cpp
// 数据类型
__m512    // 16 x float
__m512d   // 8 x double
__m512i   // 16 x int32, 8 x int64, etc.

// 加载/存储
_mm512_load_ps()        // 对齐加载
_mm512_loadu_ps()       // 未对齐加载
_mm512_store_ps()       // 对齐存储
_mm512_storeu_ps()      // 未对齐存储
_mm512_stream_ps()      // 非临时存储

// 算术
_mm512_add_ps()         // 加法
_mm512_sub_ps()         // 减法
_mm512_mul_ps()         // 乘法
_mm512_div_ps()         // 除法
_mm512_fmadd_ps()       // FMA: a * b + c

// 归约
_mm512_reduce_add_ps()  // 求和
_mm512_reduce_max_ps()  // 最大值
_mm512_reduce_min_ps()  // 最小值

// 比较
_mm512_cmp_ps_mask()    // 比较，返回掩码
_mm512_mask_blend_ps()  // 根据掩码选择

// 掩码操作
__mmask16               // 16 位掩码
_mm512_kmov()           // 掩码移动
```

### ARM NEON 常用指令速查

```cpp
// 数据类型
float32x4_t   // 4 x float
float32x2_t   // 2 x float
int32x4_t     // 4 x int32
int8x16_t     // 16 x int8

// 加载/存储
vld1q_f32()   // 加载 4 x float
vst1q_f32()   // 存储 4 x float
vdupq_n_f32() // 广播标量到所有位置

// 算术
vaddq_f32()   // 加法
vsubq_f32()   // 减法
vmulq_f32()   // 乘法
vmlaq_f32()   // FMA: a + b * c

// 归约
vaddvq_f32()  // 求和（需要 ARMv8.3+）
vmaxvq_f32()  // 最大值
vminvq_f32()  // 最小值

// 比较
vceqq_f32()   // 相等比较
vcgtq_f32()   // 大于比较
vcltq_f32()   // 小于比较
```

---

## 第三部分：Faiss SIMD 源码深度剖析

### 3.1 Faiss SIMD 抽象层设计

#### 3.1.1 simdlib 统一接口

```cpp
// 位置: faiss/utils/simdlib.h

namespace faiss {

// SIMD 统一抽象层
// 根据编译时宏自动选择最优实现

#if defined(__AVX512F__)
    #include <faiss/utils/simdlib_avx2.h>
    #include <faiss/utils/simdlib_avx512.h>
    // 使用 AVX-512 实现

#elif defined(__AVX2__)
    #include <faiss/utils/simdlib_avx2.h>
    // 使用 AVX2 实现

#elif defined(__aarch64__)
    #include <faiss/utils/simdlib_neon.h>
    // 使用 ARM NEON 实现

#elif defined(__PPC64__)
    #include <faiss/utils/simdlib_ppc64.h>
    // 使用 PowerPC AltiVec 实现

#else
    #include <faiss/utils/simdlib_emulated.h>
    // 使用标量模拟实现（无 SIMD）
#endif

} // namespace faiss
```

#### 3.1.2 AVX2 实现详解

```cpp
// 位置: faiss/utils/simdlib_avx2.h

namespace faiss {

// 256 位寄存器基础表示（不解释数据类型）
struct simd256bit {
    union {
        __m256i i;    // 整数寄存器
        __m256 f;     // 浮点寄存器
    };

    simd256bit() = default;

    // 从内存加载
    void loadu(const void* ptr) {
        i = _mm256_loadu_si256((__m256i*)ptr);
    }

    // 存储到内存
    void storeu(void* ptr) const {
        _mm256_storeu_si256((__m256i*)ptr, i);
    }

    // 清零
    void clear() {
        i = _mm256_setzero_si256();
    }
};

// 16 个 uint16 向量（256 位）
struct simd16uint16 : simd256bit {
    simd16uint16() = default;

    explicit simd16uint16(uint16_t x)
        : simd256bit(_mm256_set1_epi16(x)) {}

    explicit simd16uint16(const uint16_t* x)
        : simd256bit((const void*)x) {}

    // 乘法（无符号 16 位）
    simd16uint16 operator*(const simd16uint16& other) const {
        return simd16uint16(_mm256_mullo_epi16(i, other.i));
    }

    // 右移
    simd16uint16 operator>>(int shift) const {
        return simd16uint16(_mm256_srli_epi16(i, shift));
    }

    // 加法
    simd16uint16 operator+(simd16uint16 other) const {
        return simd16uint16(_mm256_add_epi16(i, other.i));
    }
};

// 8 个 float 向量（256 位）
struct simd8float32 : simd256bit {
    simd8float32() = default;

    explicit simd8float32(float x)
        : simd256bit(_mm256_set1_ps(x)) {}

    explicit simd8float32(const float* x)
        : simd256bit(_mm256_loadu_ps(x)) {}

    // 加法
    simd8float32 operator+(simd8float32 other) const {
        return simd8float32(_mm256_add_ps(f, other.f));
    }

    // 乘法
    simd8float32 operator*(simd8float32 other) const {
        return simd8float32(_mm256_mul_ps(f, other.f));
    }

    // FMA: a * b + c
    static simd8float32 fmadd(simd8float32 a, simd8float32 b, simd8float32 c) {
        #ifdef __FMA__
            return simd8float32(_mm256_fmadd_ps(a.f, b.f, c.f));
        #else
            return simd8float32(_mm256_add_ps(_mm256_mul_ps(a.f, b.f), c.f));
        #endif
    }
};

} // namespace faiss
```

### 3.2 距离计算的 SIMD 优化

#### 3.2.1 L2 距离的 AVX2 实现

```cpp
// 位置: faiss/utils/distances_simd.cpp

namespace faiss {

float fvec_L2sqr(const float* x, const float* y, size_t d) {
    float res = 0.0f;

    #ifdef __AVX2__
    __m256 sum = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 xv = _mm256_loadu_ps(x);
        __m256 yv = _mm256_loadu_ps(y);

        __m256 diff = _mm256_sub_ps(xv, yv);
        __m256 sq = _mm256_mul_ps(diff, diff);

        sum = _mm256_add_ps(sum, sq);

        x += 8;
        y += 8;
        d -= 8;
    }

    // 水平求和
    sum = _mm256_hadd_ps(sum, sum);
    sum = _mm256_hadd_ps(sum, sum);
    res += _mm256_cvtss_f32(sum);
    #endif

    while (d > 0) {
        float tmp = *x++ - *y++;
        res += tmp * tmp;
        d--;
    }

    return res;
}
```

#### 3.2.2 内积的 SIMD 实现

```cpp
float fvec_inner_product(const float* x, const float* y, size_t d) {
    float res = 0.0f;

    #ifdef __AVX2__
    __m256 sum = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 xv = _mm256_loadu_ps(x);
        __m256 yv = _mm256_loadu_ps(y);

        #ifdef __FMA__
            sum = _mm256_fmadd_ps(xv, yv, sum);  // sum += x * y
        #else
            __m256 prod = _mm256_mul_ps(xv, yv);
            sum = _mm256_add_ps(sum, prod);
        #endif

        x += 8;
        y += 8;
        d -= 8;
    }

    sum = _mm256_hadd_ps(sum, sum);
    sum = _mm256_hadd_ps(sum, sum);
    res += _mm256_cvtss_f32(sum);
    #endif

    while (d > 0) {
        res += *x++ * *y++;
        d--;
    }

    return res;
}
```

### 3.3 ARM NEON 实现详解

```cpp
#ifdef __aarch64__
#include <arm_neon.h>

float fvec_L2sqr_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 3 < d; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t yv = vld1q_f32(y + i);

        float32x4_t diff = vsubq_f32(xv, yv);
        sum = vmlaq_f32(sum, diff, diff);  // sum += diff * diff
    }

    // 水平求和
    sum = vpaddq_f32(sum, sum);
    sum = vpaddq_f32(sum, sum);

    float res = vgetq_lane_f32(sum, 0);

    for (; i < d; i++) {
        float diff = x[i] - y[i];
        res += diff * diff;
    }

    return res;
}
#endif
```

### 3.4 性能优化技巧

#### 3.4.1 对齐优化

```cpp
// 使用 posix_memalign 分配对齐内存
float* allocate_aligned(size_t n) {
    float* ptr;
    posix_memalign((void**)&ptr, 32, n * sizeof(float));
    return ptr;  // 32 字节对齐，可使用 _mm256_load_ps
}
```

#### 3.4.2 预取优化

```cpp
void compute_with_prefetch(const float* x, const float* y, float* result, size_t n) {
    const size_t PREFETCH_DISTANCE = 4;

    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        if (i + PREFETCH_DISTANCE * 8 < n) {
            _mm_prefetch((char*)(x + i + PREFETCH_DISTANCE * 8), _MM_HINT_T0);
            _mm_prefetch((char*)(y + i + PREFETCH_DISTANCE * 8), _MM_HINT_T0);
        }

        __m256 xv = _mm256_loadu_ps(&x[i]);
        __m256 yv = _mm256_loadu_ps(&y[i]);
        _mm256_storeu_ps(&result[i], _mm256_mul_ps(xv, yv));
    }
}
```

---

## 总结

本课程深入讲解了 AVX-512 和 ARM NEON 的实战应用：

1. **AVX-512 部分**：
   - 512 位寄存器和掩码操作
   - 高效的归约和条件操作
   - 压缩和展开技术

2. **ARM NEON 部分**：
   - 128 位寄存器和数据类型
   - FMA 和后置指令优化
   - 多向量并行处理

3. **Faiss SIMD 源码剖析**：
   - simdlib 统一抽象层设计
   - 距离计算的 SIMD 优化实现
   - 运行时检测和分派策略

**下一步学习**：
- 《现代硬件特性专题》：Intel AMX、持久内存
- 《向量搜索完整优化案例》：从零开始优化
- 《高级专题深入》：NUMA、异步I/O

**练习建议**：
1. 在本地实现所有练习
2. 使用 perf 验证 SIMD 加速效果
3. 测试不同数据规模下的性能
4. 尝试优化自己的向量搜索代码

祝学习顺利！🚀
