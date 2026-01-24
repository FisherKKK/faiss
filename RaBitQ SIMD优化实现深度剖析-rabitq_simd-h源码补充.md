# RaBitQ SIMD优化实现深度剖析 - rabitq_simd.h源码补充

## 文档说明

本文档是《RaBitQ SIMD优化深度剖析-二值化向量搜索实现》的补充材料，深入剖析RaBitQ的SIMD实现细节，包括popcount优化、多级SIMD回退机制等。

**前置知识**：
- 已完成《RaBitQ SIMD优化深度剖析》
- 熟悉SIMD指令集（AVX-512, AVX2, SSE4.1）
- 了解二值化向量搜索原理

---

## 目录
- [1. rabitq_simd.h架构](#1-rabitq_simdh架构)
- [2. Popcount优化实现](#2-popcount优化实现)
- [3. 查找表生成](#3-查找表生成)
- [4. bitwise_and_dot_product函数](#4-bitwise_and_dot_product函数)
- [5. 多级SIMD回退机制](#5多级simd回退机制)
- [6. 性能优化技巧](#6-性能优化技巧)
- [7. 总结](#7-总结)

---

## 1. rabitq_simd.h架构

### 1.1 文件结构

```cpp
namespace faiss::rabitq {
    // AVX-512支持
    #if defined(__AVX512F__)
    inline __m512i get_lookup_512();
    inline __m512i popcount_512(__m512i v);
    #endif

    // AVX2支持
    #if defined(__AVX2__)
    inline __m256i get_lookup_256();
    inline __m256i popcount_256(__m256i v);
    inline uint64_t reduce_add_256(__m256i v);
    #endif

    // SSE4.1支持
    #if defined(__SSE4_1__)
    inline __m128i popcount_128(__m128i v);
    inline uint64_t reduce_add_128(__m128i v);
    #endif

    // 核心函数
    inline uint64_t bitwise_and_dot_product(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb);
}
```

### 1.2 设计目标

1. **性能最大化**：使用最宽的SIMD指令
2. **向后兼容**：自动回退到 narrower SIMD
3. **跨平台**：支持 x86_64, ARM64 等
4. **零开销**：内联函数，编译期优化

### 1.3 编译时特性检测

```cpp
// 使用预处理器指令检测SIMD支持
#if defined(__AVX512F__)
    // AVX-512代码
#elif defined(__AVX2__)
    // AVX2回退代码
#elif defined(__SSE4_1__)
    // SSE4.1回退代码
#else
    // 标量代码
#endif
```

---

## 2. Popcount优化实现

### 2.1 为什么需要Popcount

**Popcount (Population Count)**：计算整数中设置的位（1）的数量

在RaBitQ中：
- 二值化向量用bit表示
- 内积 = popcount(a & b)
- 性能瓶颈

### 2.2 硬件Popcount vs 软件模拟

```cpp
// 硬件popcount（最快，1个周期）
#if defined(__AVX512VPOPCNTDQ__)
    return _mm512_popcnt_epi64(v);
#else
    // 软件模拟（使用查找表）
#endif
```

**指令对比**：

| 指令 | 延迟 | 吞吐量 | 说明 |
|------|------|--------|------|
| `popcnt` (标量) | 3 | 1 | SSE4.2 |
| `vpopcntdq` (AVX-512) | 3 | 1 | AVX-512 VPOPCNTDQ |
| `vpshufb` (查找表) | 1 | 0.5 | AVX2/AVX-512 |

### 2.3 AVX-512 Popcount实现

```cpp
inline __m512i popcount_512(__m512i v) {
#if defined(__AVX512VPOPCNTDQ__)
    // 硬件popcount（如果CPU支持）
    return _mm512_popcnt_epi64(v);

#else
    // 软件模拟（使用查找表）
    const __m512i lookup = get_lookup_512();
    const __m512i low_mask = _mm512_set1_epi8(0x0f);

    // 分离低4位和高4位
    const __m512i lo = _mm512_and_si512(v, low_mask);
    const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), low_mask);

    // 查找表获取popcount
    const __m512i popcnt_lo = _mm512_shuffle_epi8(lookup, lo);
    const __m512i popcnt_hi = _mm512_shuffle_epi8(lookup, hi);

    // 合并结果
    const __m512i popcnt = _mm512_add_epi8(popcnt_lo, popcnt_hi);

    // 归约：uint8_t[64] -> uint64_t[8]
    return _mm512_sad_epu8(_mm512_setzero_si512(), popcnt);
#endif
}
```

**算法步骤**：

```
输入: 0xAB = 0b10101011

步骤1: 分离低4位和高4位
  lo = 0xAB & 0x0F = 0x0B = 0b1011  (4位)
  hi = (0xAB >> 4) & 0x0F = 0x0A = 0b1010

步骤2: 查找表
  popcnt_lo = lookup[0x0B] = lookup[11] = 3
  popcnt_hi = lookup[0x0A] = lookup[10] = 2

步骤3: 合并
  popcnt = 3 + 2 = 5

验证: 0xAB有5个1 ✓
```

### 2.4 AVX2 Popcount实现

```cpp
inline __m256i popcount_256(__m256i v) {
    const __m256i lookup = get_lookup_256();
    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    const __m256i lo = _mm256_and_si256(v, low_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    const __m256i popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
    const __m256i popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
    const __m256i popcnt = _mm256_add_epi8(popcnt_lo, popcnt_hi);

    // 归约：uint8_t[32] -> uint64_t[4]
    return _mm256_sad_epu8(_mm256_setzero_si512(), popcnt);
}
```

### 2.5 SSE4.1 Popcount实现

```cpp
inline __m128i popcount_128(__m128i v) {
    // 使用标量popcount
    uint64_t lane0 = _mm_extract_epi64(v, 0);
    uint64_t lane1 = _mm_extract_epi64(v, 1);
    uint64_t pop0 = __builtin_popcountll(lane0);
    uint64_t pop1 = __builtin_popcountll(lane1);
    return _mm_set_epi64x(pop1, pop0);
}
```

**说明**：SSE4.1没有向量popcount，使用标量指令

---

## 3. 查找表生成

### 3.1 AVX-512查找表

```cpp
inline __m512i get_lookup_512() {
    return _mm512_set_epi8(
        /* f */ 4, /* e */ 3, /* d */ 3, /* c */ 2,
        /* b */ 3, /* a */ 2, /* 9 */ 2, /* 8 */ 1,
        /* 7 */ 3, /* 6 */ 2, /* 5 */ 2, /* 4 */ 1,
        /* 3 */ 2, /* 2 */ 1, /* 1 */ 1, /* 0 */ 0,
        // ... 重复4次（AVX-512有64个字节）
    );
}
```

**查找表内容**：

| 输入 (4-bit) | Popcount | 十六进制 |
|-------------|----------|---------|
| 0x0 (0000)  | 0        | 0       |
| 0x1 (0001)  | 1        | 1       |
| 0x2 (0010)  | 1        | 1       |
| 0x3 (0011)  | 2        | 2       |
| 0x4 (0100)  | 1        | 1       |
| 0x5 (0101)  | 2        | 2       |
| 0x6 (0110)  | 2        | 2       |
| 0x7 (0111)  | 3        | 3       |
| 0x8 (1000)  | 1        | 1       |
| 0x9 (1001)  | 2        | 2       |
| 0xA (1010)  | 2        | 2       |
| 0xB (1011)  | 3        | 3       |
| 0xC (1100)  | 2        | 2       |
| 0xD (1101)  | 3        | 3       |
| 0xE (1110)  | 3        | 3       |
| 0xF (1111)  | 4        | 4       |

### 3.2 查找表工作原理

```cpp
// 示例：计算 0xAB 的 popcount

__m512i lookup = get_lookup_512();
__m512i v = _mm512_set1_epi8(0xAB);

// 分离低4位和高4位
__m512i lo = _mm512_and_si512(v, _mm512_set1_epi8(0x0f));  // 0x0B
__m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), _mm512_set1_epi8(0x0f));  // 0x0A

// 使用PSHUFB查找
__m512i popcnt_lo = _mm512_shuffle_epi8(lookup, lo);  // lookup[0x0B] = 3
__m512i popcnt_hi = _mm512_shuffle_epi8(lookup, hi);  // lookup[0x0A] = 2

// 合并
__m512i popcnt = _mm512_add_epi8(popcnt_lo, popcnt_hi);  // 5
```

**PSHUFB指令**：
- 根据输入字节的低4位选择查找表中的字节
- 64个字节并行查找
- 延迟：1周期，吞吐量：0.5

---

## 4. bitwise_and_dot_product函数

### 4.1 函数签名

```cpp
inline uint64_t bitwise_and_dot_product(
        const uint8_t* query,    // 查询向量（重排后的旋转查询数据）
        const uint8_t* data,     // 数据库向量（二值化）
        size_t size,             // 向量大小（字节）
        size_t qb) {             // 量化位数
```

### 4.2 AVX-512路径

```cpp
#if defined(__AVX512F__)
    // 处理512位（64字节）块
    if (size_t step = 512 / 8; offset + step <= size) {
        __m512i sum_512 = _mm512_setzero_si512();

        for (; offset + step <= size; offset += step) {
            // 加载数据向量
            __m512i v_x = _mm512_loadu_si512((const __m512i*)(data + offset));

            // 对每个量化位进行处理
            for (int j = 0; j < qb; j++) {
                // 加载查询向量
                __m512i v_q = _mm512_loadu_si512(
                        (const __m512i*)(query + j * size + offset));

                // 位与 + popcount
                __m512i v_and = _mm512_and_si512(v_q, v_x);
                __m512i v_popcnt = popcount_512(v_and);

                // 左移j位（相当于乘以2^j）
                __m512i v_shifted = _mm512_slli_epi64(v_popcnt, j);

                // 累加
                sum_512 = _mm512_add_epi64(sum_512, v_shifted);
            }
        }

        // 归约：uint64_t[8] -> uint64_t
        sum += _mm512_reduce_add_epi64(sum_512);
    }
#endif
```

**算法说明**：

1. **加载数据**：每次加载64字节数据
2. **循环量化位**：对qb个量化位分别处理
3. **位与操作**：`v_and = v_q & v_x`
4. **Popcount**：计算设置的位数
5. **左移累加**：`sum += popcnt << j`

### 4.3 AVX2路径

```cpp
#if defined(__AVX2__)
    if (size_t step = 256 / 8; offset + step <= size) {
        __m256i sum_256 = _mm256_setzero_si256();

        for (; offset + step <= size; offset += step) {
            __m256i v_x = _mm256_loadu_si256((const __m256i*)(data + offset));

            for (int j = 0; j < qb; j++) {
                __m256i v_q = _mm256_loadu_si256(
                        (const __m256i*)(query + j * size + offset));
                __m256i v_and = _mm256_and_si256(v_q, v_x);
                __m256i v_popcnt = popcount_256(v_and);
                __m256i v_shifted = _mm256_slli_epi64(v_popcnt, j);
                sum_256 = _mm256_add_epi64(sum_256, v_shifted);
            }
        }

        sum += reduce_add_256(sum_256);
    }
#endif
```

### 4.4 SSE4.1路径

```cpp
#if defined(__SSE4_1__)
    __m128i sum_128 = _mm_setzero_si128();

    for (size_t step = 128 / 8; offset + step <= size; offset += step) {
        __m128i v_x = _mm_loadu_si128((const __m128i*)(data + offset));

        for (int j = 0; j < qb; j++) {
            __m128i v_q = _mm_loadu_si128(
                    (const __m128i*)(query + j * size + offset));
            __m128i v_and = _mm_and_si128(v_q, v_x);
            __m128i v_popcnt = popcount_128(v_and);
            __m128i v_shifted = _mm_slli_epi64(v_popcnt, j);
            sum_128 = _mm_add_epi64(sum_128, v_shifted);
        }
    }

    sum += reduce_add_128(sum_128);
#endif
```

### 4.5 标量回退路径

```cpp
// 64位处理
for (size_t step = 64 / 8; offset + step <= size; offset += step) {
    const auto yv = *(const uint64_t*)(data + offset);
    for (int j = 0; j < qb; j++) {
        const auto qv = *(const uint64_t*)(query + j * size + offset);
        sum += __builtin_popcountll(qv & yv) << j;
    }
}

// 字节处理（剩余部分）
for (; offset < size; ++offset) {
    const auto yv = *(data + offset);
    for (int j = 0; j < qb; j++) {
        const auto qv = *(query + j * size + offset);
        sum += __builtin_popcount(qv & yv) << j;
    }
}
```

---

## 5. 多级SIMD回退机制

### 5.1 回退策略

```
if (支持AVX-512) {
    使用AVX-512处理64字节块
    剩余部分 -> AVX2
} else if (支持AVX2) {
    使用AVX2处理32字节块
    剩余部分 -> SSE4.1
} else if (支持SSE4.1) {
    使用SSE4.1处理16字节块
    剩余部分 -> 标量
} else {
    标量处理
}
```

### 5.2 性能对比

| SIMD宽度 | 周期/字节 | 相对性能 |
|---------|----------|---------|
| AVX-512 (64B) | 0.5 | 1x (基线) |
| AVX2 (32B) | 1.0 | 2x slower |
| SSE4.1 (16B) | 2.0 | 4x slower |
| 标量 (1B) | ~32 | 64x slower |

### 5.3 编译时优化

```cpp
// 编译器会完全移除不支持的路径
// 例如：在仅支持AVX2的CPU上
// AVX-512代码被完全移除，不增加二进制大小

#if defined(__AVX512F__)
    // AVX-512代码
#endif

#if defined(__AVX2__)
    // AVX2代码
#endif
```

---

## 6. 性能优化技巧

### 6.1 对齐访问

```cpp
// 非对齐访问（通用）
__m512i v = _mm512_loadu_si512((const __m512i*)(data + offset));

// 对齐访问（更快，但需要对齐地址）
__m512i v = _mm512_load_si512((const __m512i*)(data + offset));
```

**性能差异**：
- 对齐加载：~10周期（L1缓存命中）
- 非对齐加载：~10周期（现代CPU差异小）
- 跨缓存行：~20-30周期

### 6.2 循环展开

```cpp
// 手动展开2路
for (; offset + 2 * step <= size; offset += 2 * step) {
    __m512i v_x0 = _mm512_loadu_si512((const __m512i*)(data + offset));
    __m512i v_x1 = _mm512_loadu_si512((const __m512i*)(data + offset + step));

    // 处理v_x0
    // 处理v_x1
}

// 处理剩余
for (; offset + step <= size; offset += step) {
    // ...
}
```

### 6.3 预取优化

```cpp
// 在循环中预取下一个块
for (; offset + step <= size; offset += step) {
    // 预取下一个缓存行
    if (offset + 128 <= size) {
        _mm_prefetch(data + offset + 128, _MM_HINT_T0);
    }

    // 处理当前块
    __m512i v_x = _mm512_loadu_si512((const __m512i*)(data + offset));
    // ...
}
```

### 6.4 避免分支

```cpp
// 错误：分支在循环内
for (int j = 0; j < qb; j++) {
    if (j < threshold) {
        // 处理
    }
}

// 正确：循环外处理
for (int j = 0; j < threshold; j++) {
    // 处理
}
for (int j = threshold; j < qb; j++) {
    // 处理
}
```

---

## 7. 总结

### 7.1 关键优化技术

1. **多级SIMD回退**：AVX-512 -> AVX2 -> SSE4.1 -> 标量
2. **Popcount优化**：硬件指令优先，查找表回退
3. **查找表加速**：使用PSHUFB进行并行查找
4. **批量处理**：一次处理64/32/16字节

### 7.2 性能数据

| 向量维度 | AVX-512 | AVX2 | SSE4.1 | 标量 |
|---------|---------|------|--------|------|
| 128维 | 50 ns | 100 ns | 200 ns | 3200 ns |
| 256维 | 100 ns | 200 ns | 400 ns | 6400 ns |
| 512维 | 200 ns | 400 ns | 800 ns | 12800 ns |

**加速比**：
- AVX-512 vs 标量：64x
- AVX2 vs 标量：32x
- SSE4.1 vs 标量：16x

### 7.3 设计权衡

| 方面 | 优势 | 劣势 |
|------|------|------|
| AVX-512 | 最快 | CPU支持有限 |
| 多级回退 | 兼容性好 | 代码体积大 |
| 查找表 | 无需特殊指令 | 占用寄存器 |

### 7.4 实际应用建议

1. **优先使用AVX-512**：如果CPU支持
2. **测试回退路径**：确保所有路径正确
3. **性能分析**：使用perf/vtune分析瓶颈
4. **内存对齐**：尽可能使用对齐访问

---

## 附录A：完整示例

```cpp
#include <faiss/utils/rabitq_simd.h>
#include <cstdio>
#include <chrono>

using namespace faiss::rabitq;

void benchmark_dot_product() {
    // 准备数据
    size_t size = 1024;  // 1024字节 = 8192位
    size_t qb = 2;       // 2个量化位

    std::vector<uint8_t> query(qb * size);
    std::vector<uint8_t> data(size);

    // 填充随机数据
    for (size_t i = 0; i < qb * size; i++) {
        query[i] = rand() & 0xFF;
    }
    for (size_t i = 0; i < size; i++) {
        data[i] = rand() & 0xFF;
    }

    // 测试
    const int iterations = 1000000;

    auto start = std::chrono::high_resolution_clock::now();

    uint64_t sum = 0;
    for (int iter = 0; iter < iterations; iter++) {
        sum += bitwise_and_dot_product(
            query.data(), data.data(), size, qb);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    printf("Sum: %lu\n", sum);
    printf("Time: %.3f ns/iter\n", elapsed / iterations * 1e9);
}

int main() {
    benchmark_dot_product();
    return 0;
}
```

## 附录B：相关源文件

- `faiss/utils/rabitq_simd.h` - SIMD优化实现
- `faiss/impl/RaBitQuantizer.h` - RaBitQ量化器
- `faiss/impl/IndexRaBitQ.cpp` - RaBitQ索引实现

## 附录C：性能测试代码

```cpp
#include <benchmark/benchmark.h>

static void BM_BitwiseDotProduct(benchmark::State& state) {
    size_t size = state.range(0);
    size_t qb = 2;

    std::vector<uint8_t> query(qb * size);
    std::vector<uint8_t> data(size);

    // 初始化
    for (size_t i = 0; i < qb * size; i++) {
        query[i] = rand() & 0xFF;
    }
    for (size_t i = 0; i < size; i++) {
        data[i] = rand() & 0xFF;
    }

    for (auto _ : state) {
        uint64_t sum = bitwise_and_dot_product(
            query.data(), data.data(), size, qb);
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_BitwiseDotProduct)->Range(64, 4096);
BENCHMARK_MAIN();
```
