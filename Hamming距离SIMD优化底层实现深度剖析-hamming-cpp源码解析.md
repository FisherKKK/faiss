# Hamming 距离 SIMD 优化底层实现深度剖析 - hamming.cpp 源码解析

## 1. 概述

Faiss 的汉明距离模块为二进制向量提供高效的距离计算和 k-NN 搜索。汉明距离是二进制向量检索的核心度量，广泛用于图像哈希、签名匹配等场景。

### 核心特性
- **Popcount 优化**: 使用 `__builtin_popcount` 硬件指令
- **HammingComputer 模板**: 编译时特化，避免分支开销
- **架构特定优化**: AVX2、AVX512、ARM NEON 变体
- **批处理**: 支持大规模数据库的块式搜索
- **近似 Top-K**: HeapWithBuckets 优化

## 2. 基础数据结构

### 2.1 hamdis_t 类型 (hamming_distance/common.h:16)

```cpp
using hamdis_t = int32_t;
```

汉明距离使用 32 位整数存储：
- 最大距离受限于代码长度
- 对于 256-bit 代码，最大距离为 256
- 16 位足够但 32 位提供更好的对齐和性能

### 2.2 查表优化 (hamming_distance/common.h:33-44)

```cpp
inline constexpr uint8_t hamdis_tab_ham_bytes[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, ...
};
```

**用途**: 快速查找字节中 1 的个数
- `hamdis_tab_ham_bytes[x]` = x 的 popcount
- 用于非 64 位对齐代码的尾部处理

### 2.3 Popcount 实现 (hamming_distance/common.h:21-28)

```cpp
// trust the compiler to provide efficient popcount implementations
inline int popcount32(uint32_t x) {
    return __builtin_popcount(x);
}

inline int popcount64(uint64_t x) {
    return __builtin_popcountl(x);
}
```

**硬件指令映射**:
| 架构 | 指令 | 延迟 |
|------|------|------|
| x86-64 | `POPCNT` | 1-3 cycles |
| ARM | `VCNT` (NEON) | 1 cycle |

## 3. HammingComputer 类

### 3.1 设计原理

HammingComputer 使用**模板特化**避免运行时分支：

```cpp
// 通用模板
template <int CODE_SIZE>
struct HammingComputer : HammingComputerDefault {
    HammingComputer(const uint8_t* a, int code_size)
            : HammingComputerDefault(a, code_size) {}
};

// 特化版本
#define SPECIALIZED_HC(CODE_SIZE)                                    \
    template <>                                                      \
    struct HammingComputer<CODE_SIZE> : HammingComputer##CODE_SIZE { \
        HammingComputer(const uint8_t* a)                            \
                : HammingComputer##CODE_SIZE(a, CODE_SIZE) {}        \
    }

SPECIALIZED_HC(4);
SPECIALIZED_HC(8);
SPECIALIZED_HC(16);
SPECIALIZED_HC(20);
SPECIALIZED_HC(32);
SPECIALIZED_HC(64);
```

### 3.2 HammingComputer8 (generic-inl.h:105-126)

```cpp
struct HammingComputer8 {
    uint64_t a0;

    HammingComputer8() {}

    HammingComputer8(const uint8_t* a, int code_size) {
        set(a, code_size);
    }

    void set(const uint8_t* a, int code_size) {
        assert(code_size == 8);
        a0 = *(uint64_t*)a;  // 加载 64 位到寄存器
    }

    inline int hamming(const uint8_t* b) const {
        return popcount64(*(uint64_t*)b ^ a0);  // 单指令异或 + popcount
    }

    inline static constexpr int get_code_size() {
        return 8;
    }
};
```

### 3.3 HammingComputer16 (generic-inl.h:128-152)

```cpp
struct HammingComputer16 {
    uint64_t a0, a1;  // 两个 64 位寄存器

    HammingComputer16() {}

    HammingComputer16(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 16);
        const uint64_t* a = (uint64_t*)a8;
        a0 = a[0];
        a1 = a[1];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = (uint64_t*)b8;
        return popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1);
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};
```

**优化要点**:
1. **寄存器驻留**: `a0, a1` 保持在寄存器中
2. **无分支**: 纯算术操作
3. **并行性**: 两个 popcount 可以并行执行

### 3.4 HammingComputer32 (generic-inl.h:188-215)

```cpp
struct HammingComputer32 {
    uint64_t a0, a1, a2, a3;  // 四个 64 位寄存器

    HammingComputer32() {}

    HammingComputer32(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size == 32);
        const uint64_t* a = (uint64_t*)a8;
        a0 = a[0];
        a1 = a[1];
        a2 = a[2];
        a3 = a[3];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = (uint64_t*)b8;
        return popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1) +
               popcount64(b[2] ^ a2) + popcount64(b[3] ^ a3);
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};
```

### 3.5 HammingComputerDefault (generic-inl.h:252-348)

处理任意长度的代码：

```cpp
struct HammingComputerDefault {
    const uint8_t* a8;
    int quotient8;   // 代码字节数 / 8
    int remainder8;  // 代码字节数 % 8

    HammingComputerDefault() {}

    HammingComputerDefault(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        this->a8 = a8;
        quotient8 = code_size / 8;
        remainder8 = code_size % 8;
    }

    int hamming(const uint8_t* b8) const {
        int accu = 0;

        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b8);
        int i = 0, len = quotient8;

        // Duff's device 风格的循环展开
        switch (len & 7) {
            default:
                while (len > 7) {
                    len -= 8;
                    accu += popcount64(a64[i] ^ b64[i]);
                    i++;
                    [[fallthrough]];
                    case 7:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                        [[fallthrough]];
                    case 6:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                        [[fallthrough]];
                    case 5:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                        [[fallthrough]];
                    case 4:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                        [[fallthrough]];
                    case 3:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                        [[fallthrough]];
                    case 2:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                        [[fallthrough]];
                    case 1:
                        accu += popcount64(a64[i] ^ b64[i]);
                        i++;
                }
        }

        // 处理剩余字节（使用查表）
        if (remainder8) {
            const uint8_t* a = a8 + 8 * quotient8;
            const uint8_t* b = b8 + 8 * quotient8;
            switch (remainder8) {
                case 7:
                    accu += hamdis_tab_ham_bytes[a[6] ^ b[6]];
                    [[fallthrough]];
                case 6:
                    accu += hamdis_tab_ham_bytes[a[5] ^ b[5]];
                    [[fallthrough]];
                case 5:
                    accu += hamdis_tab_ham_bytes[a[4] ^ b[4]];
                    [[fallthrough]];
                case 4:
                    accu += hamdis_tab_ham_bytes[a[3] ^ b[3]];
                    [[fallthrough]];
                case 3:
                    accu += hamdis_tab_ham_bytes[a[2] ^ b[2]];
                    [[fallthrough]];
                case 2:
                    accu += hamdis_tab_ham_bytes[a[1] ^ b[1]];
                    [[fallthrough]];
                case 1:
                    accu += hamdis_tab_ham_bytes[a[0] ^ b[0]];
                    [[fallthrough]];
                default:
                    break;
            }
        }

        return accu;
    }

    inline int get_code_size() const {
        return quotient8 * 8 + remainder8;
    }
};
```

**优化技术**:
1. **Duff's Device**: 循环展开，减少分支预测失败
2. **64 位对齐处理**: 使用 popcount64
3. **尾部查表**: 剩余字节使用查表法
4. **[[fallthrough]]**: 显式告诉编译器这是故意的 fallthrough

## 4. 模板分发机制

### 4.1 dispatch_HammingComputer (hamdis-inl.h:64-83)

```cpp
template <class Consumer, class... Types>
typename Consumer::T dispatch_HammingComputer(
        int code_size,
        Consumer& consumer,
        Types... args) {
    switch (code_size) {
#define DISPATCH_HC(CODE_SIZE) \
    case CODE_SIZE:            \
        return consumer.template f<HammingComputer##CODE_SIZE>(args...);

        DISPATCH_HC(4);
        DISPATCH_HC(8);
        DISPATCH_HC(16);
        DISPATCH_HC(20);
        DISPATCH_HC(32);
        DISPATCH_HC(64);
        default:
            return consumer.template f<HammingComputerDefault>(args...);
    }
#undef DISPATCH_HC
}
```

**使用示例**:
```cpp
struct Run_hammings_knn_hc {
    using T = void;
    template <class HammingComputer, class... Types>
    void f(Types... args) {
        hammings_knn_hc<HammingComputer>(args...);
    }
};

// 调用
Run_hammings_knn_hc r;
dispatch_HammingComputer(ncodes, r, ncodes, ha, a, b, nb, order, ...);
```

**效果**:
- 编译时生成多个特化版本
- 运行时仅需一次 switch
- 内联后无函数调用开销

## 5. KNN 搜索算法

### 5.1 堆方法 (hamming.cpp:169-239)

```cpp
template <class HammingComputer>
void hammings_knn_hc(
        int bytes_per_code,
        int_maxheap_array_t* __restrict ha,
        const uint8_t* __restrict bs1,
        const uint8_t* __restrict bs2,
        size_t n2,
        bool order = true,
        bool init_heap = true,
        ApproxTopK_mode_t approx_topk_mode = ApproxTopK_mode_t::EXACT_TOPK,
        const faiss::IDSelector* sel = nullptr) {
    size_t k = ha->k;
    if (init_heap) {
        ha->heapify();  // 初始化堆为中性值
    }

    const size_t block_size = hamming_batch_size;  // 65536

    // 分块处理数据库
    for (size_t j0 = 0; j0 < n2; j0 += block_size) {
        const size_t j1 = std::min(j0 + block_size, n2);

#pragma omp parallel for
        for (int64_t i = 0; i < ha->nh; i++) {
            HammingComputer hc(bs1 + i * bytes_per_code, bytes_per_code);

            const uint8_t* __restrict bs2_ = bs2 + j0 * bytes_per_code;
            hamdis_t dis;
            hamdis_t* __restrict bh_val_ = ha->val + i * k;
            int64_t* __restrict bh_ids_ = ha->ids + i * k;

            // 近似 top-k 或精确堆
            switch (approx_topk_mode) {
                case ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B8_D3:
                    HeapWithBucketsForHamming32<
                            CMax<hamdis_t, int64_t>, 8, 3, HammingComputer>::
                            addn(j1 - j0, hc, bs2_, k, bh_val_, bh_ids_, sel);
                    break;
                case ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B16_D2:
                    HeapWithBucketsForHamming32<
                            CMax<hamdis_t, int64_t>, 16, 2, HammingComputer>::
                            addn(j1 - j0, hc, bs2_, k, bh_val_, bh_ids_, sel);
                    break;
                default: {
                    // 精确版本
                    for (size_t j = j0; j < j1; j++, bs2_ += bytes_per_code) {
                        if (sel && !sel->is_member(j)) {
                            continue;
                        }
                        dis = hc.hamming(bs2_);
                        if (dis < bh_val_[0]) {
                            faiss::maxheap_replace_top<hamdis_t>(
                                    k, bh_val_, bh_ids_, dis, j);
                        }
                    }
                } break;
            }
        }
    }

    if (order) {
        ha->reorder();  // 堆排序
    }
}
```

**算法流程**:
```
For each query i in 0..na-1:
    Initialize HammingComputer with query i
    For each block of database vectors:
        For each database vector j in block:
            Compute distance = hc.hamming(db[j])
            If distance < heap_max:
                Replace heap top with (distance, j)
    Sort heap
```

### 5.2 计数方法 (hamming.cpp:242-298)

使用桶计数优化：

```cpp
template <class HammingComputer>
void hammings_knn_mc(
        int bytes_per_code,
        const uint8_t* __restrict a,
        const uint8_t* __restrict b,
        size_t na,
        size_t nb,
        size_t k,
        int32_t* __restrict distances,
        int64_t* __restrict labels,
        const faiss::IDSelector* sel) {
    // 距离范围: [0, bytes_per_code * 8]
    const int nBuckets = bytes_per_code * 8 + 1;

    // 为每个查询分配桶
    std::vector<int> all_counters(na * nBuckets, 0);
    std::unique_ptr<int64_t[]> all_ids_per_dis(new int64_t[na * nBuckets * k]);

    // 初始化计数器状态
    std::vector<HCounterState<HammingComputer>> cs;
    for (size_t i = 0; i < na; ++i) {
        cs.push_back(
                HCounterState<HammingComputer>(
                        all_counters.data() + i * nBuckets,
                        all_ids_per_dis.get() + i * nBuckets * k,
                        a + i * bytes_per_code,
                        8 * bytes_per_code,
                        k));
    }

    // 分块处理
    const size_t block_size = hamming_batch_size;
    for (size_t j0 = 0; j0 < nb; j0 += block_size) {
        const size_t j1 = std::min(j0 + block_size, nb);
#pragma omp parallel for
        for (int64_t i = 0; i < na; ++i) {
            for (size_t j = j0; j < j1; ++j) {
                if (!sel || sel->is_member(j)) {
                    cs[i].update_counter(b + j * bytes_per_code, j);
                }
            }
        }
    }

    // 从桶中收集结果
    for (size_t i = 0; i < na; ++i) {
        HCounterState<HammingComputer>& csi = cs[i];

        int nres = 0;
        for (int b_2 = 0; b_2 < nBuckets && nres < k; b_2++) {
            for (int l = 0; l < csi.counters[b_2] && nres < k; l++) {
                labels[i * k + nres] = csi.ids_per_dis[b_2 * k + l];
                distances[i * k + nres] = b_2;
                nres++;
            }
        }
        // 填充剩余位置
        while (nres < k) {
            labels[i * k + nres] = -1;
            distances[i * k + nres] = std::numeric_limits<int32_t>::max();
            ++nres;
        }
    }
}
```

**HCounterState 实现** (hamming-inl.h:76-119):

```cpp
template <class HammingComputer>
struct HCounterState {
    int* counters;          // 每个距离的计数 [0..max_dis]
    int64_t* ids_per_dis;   // 每个距离对应的 ID 列表
    HammingComputer hc;     // 汉明距离计算器
    int thres;              // 当前阈值
    int count_lt;           // 距离 < thres 的计数
    int count_eq;           // 距离 == thres 的计数
    int k;                  // 需要 k 个最近邻

    HCounterState(
            int* counters,
            int64_t* ids_per_dis,
            const uint8_t* x,
            int d,
            int k)
            : counters(counters),
              ids_per_dis(ids_per_dis),
              hc(x, d / 8),
              thres(d + 1),
              count_lt(0),
              count_eq(0),
              k(k) {}

    void update_counter(const uint8_t* y, size_t j) {
        int32_t dis = hc.hamming(y);

        if (dis <= thres) {
            if (dis < thres) {
                // 添加到对应距离的桶
                ids_per_dis[dis * k + counters[dis]++] = j;
                ++count_lt;

                // 如果收集够了，收紧阈值
                while (count_lt == k && thres > 0) {
                    --thres;
                    count_eq = counters[thres];
                    count_lt -= count_eq;
                }
            } else if (count_eq < k) {
                // 当前距离桶未满
                ids_per_dis[dis * k + count_eq++] = j;
                counters[dis] = count_eq;
            }
        }
    }
};
```

**算法优势**:
- **O(n) 复杂度**: 每个向量只需一次距离计算
- **阈值收紧**: 动态缩小搜索范围
- **并行友好**: 每个查询独立计数

**适用场景**:
- k 相对较小 (k < 100)
- 距离分布集中
- 大规模数据库

### 5.3 两种方法对比

| 特性 | 堆方法 (hc) | 计数方法 (mc) |
|------|-----------|--------------|
| **时间复杂度** | O(na * nb * log k) | O(na * nb) |
| **空间复杂度** | O(na * k) | O(na * max_dis * k) |
| **小 k** | 更优 | 较差 |
| **大 k** | 较差 | 更优 |
| **内存** | 低 | 高 |

## 6. 广义汉明距离

广义汉明距离计算**不同字节数**，而非不同位数：

```cpp
// 统计字节数而非位数
inline int generalized_hamming_64(uint64_t a) {
    a |= a >> 1;    // 将每字节的所有位传播到最低位
    a |= a >> 2;
    a |= a >> 4;
    a &= 0x0101010101010101UL;  // 只保留最低位
    return popcount64(a);
}
```

**示例**:
```
a = 0x1234567890ABCDEF
b = 0x1234567890ABCDE0

普通汉明距离: 4 (最后 4 位不同)
广义汉明距离: 1 (只有最后 1 字节不同)
```

### GenHammingComputer32 (avx2-inl.h:400-418)

使用 AVX2 优化：

```cpp
struct GenHammingComputer32 {
    __m256i a;

    GenHammingComputer32(const uint8_t* a8, int code_size) {
        assert(code_size == 32);
        a = _mm256_loadu_si256((const __m256i_u*)a8);
    }

    inline int hamming(const uint8_t* b8) const {
        const __m256i b = _mm256_loadu_si256((const __m256i_u*)b8);
        const __m256i cmp = _mm256_cmpeq_epi8(a, b);  // 字节相等比较
        const uint32_t movemask = _mm256_movemask_epi8(cmp);  // 提取符号位
        return 32 - popcount32(movemask);  // 统计不相等的字节数
    }

    inline static constexpr int get_code_size() {
        return 32;
    }
};
```

**指令分解**:
1. `_mm256_cmpeq_epi8`: 并行比较 32 字节
2. `_mm256_movemask_epi8`: 提取比较结果的符号位
3. `popcount32`: 统计符号位中 0 的个数

## 7. 架构特定优化

### 7.1 平台检测 (hamdis-inl.h:16-27)

```cpp
#ifdef __aarch64__
#include <faiss/utils/hamming_distance/neon-inl.h>
#elif __AVX512F__
#include <faiss/utils/hamming_distance/avx512-inl.h>
#elif __AVX2__
#include <faiss/utils/hamming_distance/avx2-inl.h>
#else
#include <faiss/utils/hamming_distance/generic-inl.h>
#endif
```

### 7.2 AVX2 优化 (avx2-inl.h:380-398)

GenHammingComputer16 使用 SIMD：

```cpp
struct GenHammingComputer16 {
    __m128i a;

    GenHammingComputer16(const uint8_t* a8, int code_size) {
        assert(code_size == 16);
        a = _mm_loadu_si128((const __m128i_u*)a8);
    }

    inline int hamming(const uint8_t* b8) const {
        const __m128i b = _mm_loadu_si128((const __m128i_u*)b8);
        const __m128i cmp = _mm_cmpeq_epi8(a, b);
        const auto movemask = _mm_movemask_epi8(cmp);
        return 16 - popcount32(movemask);
    }

    inline static constexpr int get_code_size() {
        return 16;
    }
};
```

### 7.3 GenHammingComputerM8 AVX2 变体 (avx2-inl.h:426-458)

```cpp
struct GenHammingComputerM8 {
    const uint64_t* a;
    int n;

    GenHammingComputerM8(const uint8_t* a8, int code_size) {
        assert(code_size % 8 == 0);
        a = (uint64_t*)a8;
        n = code_size / 8;
    }

    int hamming(const uint8_t* b8) const {
        const uint64_t* b = (uint64_t*)b8;
        int accu = 0;

        int i = 0;
        int n4 = (n / 4) * 4;  // 4 个 64 位 = 32 字节 = AVX2 宽度
        for (; i < n4; i += 4) {
            // 每次 256 位
            const __m256i av = _mm256_loadu_si256((const __m256i_u*)(a + i));
            const __m256i bv = _mm256_loadu_si256((const __m256i_u*)(b + i));
            const __m256i cmp = _mm256_cmpeq_epi8(av, bv);
            const uint32_t movemask = _mm256_movemask_epi8(cmp);
            accu += 32 - popcount32(movemask);
        }

        // 处理剩余部分
        for (; i < n; i++)
            accu += generalized_hamming_64(a[i] ^ b[i]);
        return accu;
    }

    inline int get_code_size() const {
        return n * 8;
    }
};
```

**优化要点**:
1. **32 字节对齐**: AVX2 最佳宽度
2. **展开 4 次**: 每次处理 256 位
3. **尾部标量**: 剩余部分回退到标量代码

## 8. 位串读写

### 8.1 BitstringWriter (hamming-inl.h:13-37)

```cpp
inline BitstringWriter::BitstringWriter(uint8_t* code, size_t code_size)
        : code(code), code_size(code_size), i(0) {
    memset(code, 0, code_size);
}

inline void BitstringWriter::write(uint64_t x, int nbit) {
    assert(code_size * 8 >= nbit + i);
    int na = 8 - (i & 7);  // 当前字节的可用位数

    if (nbit <= na) {
        // 全部写入当前字节
        code[i >> 3] |= x << (i & 7);
        i += nbit;
        return;
    } else {
        size_t j = i >> 3;
        // 写入当前字节的剩余位
        code[j++] |= x << (i & 7);
        i += nbit;
        x >>= na;

        // 写入完整字节
        while (x != 0) {
            code[j++] |= x;
            x >>= 8;
        }
    }
}
```

**示例**:
```
write(0b1101, 4)  // i = 0
-> code[0] = 0b00001101
i = 4

write(0b10110, 5)  // i = 4
-> code[0] |= 0b11010000  (4 bits in current byte)
-> code[1] = 0b00000001  (1 bit in next byte)
i = 9
```

### 8.2 BitstringReader (hamming-inl.h:39-67)

```cpp
inline BitstringReader::BitstringReader(const uint8_t* code, size_t code_size)
        : code(code), code_size(code_size), i(0) {}

inline uint64_t BitstringReader::read(int nbit) {
    assert(code_size * 8 >= nbit + i);
    int na = 8 - (i & 7);  // 当前字节的可用位数

    // 获取当前字节的可用位
    uint64_t res = code[i >> 3] >> (i & 7);
    if (nbit <= na) {
        // 全部在当前字节
        res &= (1 << nbit) - 1;
        i += nbit;
        return res;
    } else {
        // 需要跨越多个字节
        int ofs = na;
        size_t j = (i >> 3) + 1;
        i += nbit;
        nbit -= na;

        while (nbit > 8) {
            res |= ((uint64_t)code[j++]) << ofs;
            ofs += 8;
            nbit -= 8;
        }

        uint64_t last_byte = code[j];
        last_byte &= (1 << nbit) - 1;
        res |= last_byte << ofs;
        return res;
    }
}
```

## 9. 批处理优化

### 9.1 批大小配置 (hamming.cpp:39)

```cpp
size_t hamming_batch_size = 65536;
```

**权衡考虑**:
- **太小**: 函数调用开销
- **太大**: 缓存失效
- **65536**: 在大多数系统上的良好平衡

### 9.2 批处理循环 (hamming.cpp:185-235)

```cpp
const size_t block_size = hamming_batch_size;
for (size_t j0 = 0; j0 < n2; j0 += block_size) {
    const size_t j1 = std::min(j0 + block_size, n2);
#pragma omp parallel for
    for (int64_t i = 0; i < ha->nh; i++) {
        // 处理查询 i 对数据库块 [j0, j1)
        HammingComputer hc(bs1 + i * bytes_per_code, bytes_per_code);
        // ...
    }
}
```

**优势**:
1. **缓存友好**: 每个块适配 L2 缓存
2. **负载均衡**: 动态调度 (OpenMP 默认)
3. **可扩展**: 支持 nb >> 内存容量

## 10. 向量位转换

### 10.1 float 到 bit (hamming.cpp:367-381)

```cpp
void fvec2bitvec(const float* __restrict x, uint8_t* __restrict b, size_t d) {
    for (int i = 0; i < d; i += 8) {
        uint8_t w = 0;
        uint8_t mask = 1;
        int nj = i + 8 <= d ? 8 : d - i;
        for (int j = 0; j < nj; j++) {
            if (x[i + j] >= 0) {  // 使用符号位
                w |= mask;
            }
            mask <<= 1;
        }
        *b = w;
        b++;
    }
}
```

**编码方式**: 符号编码
- `x[i] >= 0` -> bit = 1
- `x[i] < 0` -> bit = 0

### 10.2 批量转换 (hamming.cpp:385-395)

```cpp
void fvecs2bitvecs(
        const float* __restrict x,
        uint8_t* __restrict b,
        size_t d,
        size_t n) {
    const int64_t ncodes = ((d + 7) / 8);
#pragma omp parallel for if (n > 100000)
    for (int64_t i = 0; i < n; i++) {
        fvec2bitvec(x + i * d, b + i * ncodes, d);
    }
}
```

### 10.3 bit 重排 (hamming.cpp:436-460)

```cpp
void bitvec_shuffle(
        size_t n,
        size_t da,
        size_t db,
        const int* __restrict order,
        const uint8_t* __restrict a,
        uint8_t* __restrict b) {
    for (size_t i = 0; i < db; i++) {
        FAISS_THROW_IF_NOT(order[i] >= 0 && order[i] < da);
    }
    size_t lda = (da + 7) / 8;
    size_t ldb = (db + 7) / 8;

#pragma omp parallel for if (n > 10000)
    for (int64_t i = 0; i < n; i++) {
        const uint8_t* ai = a + i * lda;
        uint8_t* bi = b + i * ldb;
        memset(bi, 0, ldb);
        for (size_t j = 0; j < db; j++) {
            int o = order[j];
            uint8_t the_bit = (ai[o >> 3] >> (o & 7)) & 1;
            bi[j >> 3] |= the_bit << (j & 7);
        }
    }
}
```

**用途**: PCA 后的位重排，将重要位放在一起

## 11. 打包与解包

### 11.1 pack_bitstrings (hamming.cpp:716-733)

```cpp
void pack_bitstrings(
        size_t n,
        size_t M,
        int nbit,
        const int32_t* unpacked,
        uint8_t* packed,
        size_t code_size) {
    FAISS_THROW_IF_NOT(code_size >= (M * nbit + 7) / 8);
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        const int32_t* in = unpacked + i * M;
        uint8_t* out = packed + i * code_size;
        BitstringWriter wr(out, code_size);
        for (int j = 0; j < M; j++) {
            wr.write(in[j], nbit);
        }
    }
}
```

**示例**:
```
Input:  M=3, nbit=4, unpacked=[5, 10, 7]
       5 = 0b0101
       10 = 0b1010
       7 = 0b0111

Output: 0b010110100111 = 0x5A7
```

### 11.2 unpack_bitstrings (hamming.cpp:758-775)

```cpp
void unpack_bitstrings(
        size_t n,
        size_t M,
        int nbit,
        const uint8_t* packed,
        size_t code_size,
        int32_t* unpacked) {
    FAISS_THROW_IF_NOT(code_size >= (M * nbit + 7) / 8);
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        const uint8_t* in = packed + i * code_size;
        int32_t* out = unpacked + i * M;
        BitstringReader rd(in, code_size);
        for (int j = 0; j < M; j++) {
            out[j] = rd.read(nbit);
        }
    }
}
```

## 12. 性能分析

### 12.1 时间复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| **hamming()** | O(code_size) | 线性于代码字节数 |
| **knn (堆)** | O(na * nb * log k) | 每个距离需要 log k 堆操作 |
| **knn (计数)** | O(na * nb) | 桶排序，但需要 O(max_dis * k) 空间 |

### 12.2 每周期处理位数

| 架构 | 实现 | 位/周期 (理论) |
|------|------|---------------|
| x86-64 (POPCNT) | 标量 | 64 / 1 = 64 |
| AVX2 (GenHamming) | `_mm256_cmpeq_epi8` | 256 / 1 = 256 |
| AVX512 (GenHamming) | `_mm512_cmpeq_epi8` | 512 / 1 = 512 |
| ARM NEON | `vceq_u8` + `vaddv` | 128 / 2 = 64 |

### 12.3 内存带宽需求

对于 256-bit (32 字节) 代码，100M 向量：
- **数据库大小**: 32 × 100M = 3.2 GB
- **查询时间 (单线程)**: ~3.2 GB / 20 GB/s = 160 ms (内存限制)
- **实际时间**: ~200-500 ms (含计算)

## 13. 关键源码位置

| 文件 | 函数 | 行号 |
|------|------|------|
| `hamming.cpp` | `hammings_knn_hc()` | 169-239 |
| `hamming.cpp` | `hammings_knn_mc()` | 242-298 |
| `hamming.cpp` | `fvec2bitvec()` | 367-381 |
| `generic-inl.h` | `HammingComputer8` | 105-126 |
| `generic-inl.h` | `HammingComputer16` | 128-152 |
| `generic-inl.h` | `HammingComputer32` | 188-215 |
| `generic-inl.h` | `HammingComputerDefault` | 252-348 |
| `avx2-inl.h` | `GenHammingComputer32` | 400-418 |
| `hamming-inl.h` | `HCounterState` | 76-119 |
| `common.h` | `hamdis_tab_ham_bytes` | 33-44 |
