# Hamming距离SIMD优化深度剖析：二值化向量检索实现

## 概述

本文深入剖析Faiss中Hamming距离计算的底层实现，特别关注SIMD优化技术、popcount指令、广义Hamming距离以及二值化向量的高效检索算法。Hamming距离是二值化向量相似度搜索的核心操作，广泛用于图像检索、文本匹配和生物信息学应用。

---

## 目录

1. [Hamming距离基础](#1-hamming距离基础)
2. [Popcount优化技术](#2-popcount优化技术)
3. [专用HammingComputer类](#3-专用hammingcomputer类)
4. [AVX-512硬件加速](#4-avx-512硬件加速)
5. [广义Hamming距离](#5-广义hamming距离)
6. [KNN搜索优化](#6-knn搜索优化)
7. [位串读写器](#7-位串读写器)

---

## 1. Hamming距离基础

### 1.1 定义

**Hamming距离**: 两个等长字符串中不同字符的个数。对于二进制向量：

```
H(x, y) = Σ(x_i ⊕ y_i)  i = 0, 1, ..., d-1
```

其中⊕表示异或(XOR)操作。

**示例**:

```
x = [1, 0, 1, 1, 0, 0, 1, 0]
y = [1, 1, 0, 1, 0, 1, 1, 1]
H(x, y) = 4 (位置2,4,6,8不同)
```

### 1.2 数据类型定义

**hamdis_t类型** (hamming_distance/common.h:16):

```cpp
// Hamming距离类型
using hamdis_t = int32_t;
```

**设计考虑**:
- 支持最大2^15位向量 (32768位)
- 内存使用与性能的平衡
- 32位足以覆盖大多数应用场景

### 1.3 基础实现

**查表法** (avx2-inl.h:24-33):

```cpp
template <size_t nbits, typename T>
inline T hamming(const uint8_t* bs1, const uint8_t* bs2) {
    const size_t nbytes = nbits / 8;
    size_t i;
    T h = 0;
    for (i = 0; i < nbytes; i++) {
        // 使用查找表计算每字节的popcount
        h += (T)hamdis_tab_ham_bytes[bs1[i] ^ bs2[i]];
    }
    return h;
}
```

**查找表** (hamming_distance/common.h:33-44):

```cpp
// 256字节的popcount查找表
// hamdis_tab_ham_bytes[i] = popcount(i)
inline constexpr uint8_t hamdis_tab_ham_bytes[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    // ... (完整的256个元素)
    5, 6, 6, 7, 6, 7, 7, 8
};
```

**性能分析**:
- 查表: O(n/8) 次内存访问
- XOR + 查表: 每字节1次异或 + 1次查表
- 缓存友好: 查找表仅256字节，适合L1缓存

---

## 2. Popcount优化技术

### 2.1 内置函数

**编译器内置popcount** (common.h:21-28):

```cpp
// trust the compiler to provide efficient popcount implementations
inline int popcount32(uint32_t x) {
    return __builtin_popcount(x);  // GCC/Clang内置函数
}

inline int popcount64(uint64_t x) {
    return __builtin_popcountl(x);  // 64位版本
}
```

**硬件指令映射**:

| 架构 | 指令 | 延迟 | 吞吐量 |
|------|------|------|--------|
| x86_64 | POPCNT | 3 cycles | 1/cycle |
| ARM64 | VCNT (SIMD) | 1-2 cycles | 2/cycle |
| 通用 | 查表/分治 | - | - |

### 2.2 64位优化版本

**针对64位倍数的优化** (avx2-inl.h:36-45):

```cpp
template <size_t nbits>
inline hamdis_t hamming(const uint64_t* bs1, const uint64_t* bs2) {
    const size_t nwords = nbits / 64;
    size_t i;
    hamdis_t h = 0;
    for (i = 0; i < nwords; i++) {
        h += popcount64(bs1[i] ^ bs2[i]);
    }
    return h;
}
```

**特化版本** (avx2-inl.h:48-62):

```cpp
// 64-bit代码 (8字节)
template <>
inline hamdis_t hamming<64>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]);
}

// 128-bit代码 (16字节)
template <>
inline hamdis_t hamming<128>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]);
}

// 256-bit代码 (32字节)
template <>
inline hamdis_t hamming<256>(const uint64_t* pa, const uint64_t* pb) {
    return popcount64(pa[0] ^ pb[0]) + popcount64(pa[1] ^ pb[1]) +
           popcount64(pa[2] ^ pb[2]) + popcount64(pa[3] ^ pb[3]);
}
```

**性能优化技巧**:
1. **模板特化**: 编译期展开，消除循环
2. **寄存器分配**: 将a0-a3放入寄存器，避免内存访问
3. **指令级并行**: 多个popcount独立执行

---

## 3. 专用HammingComputer类

### 3.1 设计理念

**HammingComputer**: 用于单个查询向量与多个数据库向量的比较。

**核心思想**:
1. 预加载查询向量到寄存器
2. 内联hamming()函数避免函数调用开销
3. 使用模板特化优化不同代码长度

### 3.2 HammingComputer8

**8字节代码实现** (avx2-inl.h:107-128):

```cpp
struct HammingComputer8 {
    uint64_t a0;  // 查询向量存储在寄存器中

    HammingComputer8() {}

    HammingComputer8(const uint8_t* a, int code_size) {
        set(a, code_size);
    }

    void set(const uint8_t* a, int code_size) {
        assert(code_size == 8);
        a0 = *(uint64_t*)a;  // 一次加载64位
    }

    inline int hamming(const uint8_t* b) const {
        return popcount64(*(uint64_t*)b ^ a0);
    }

    inline static constexpr int get_code_size() {
        return 8;
    }
};
```

### 3.3 HammingComputer16

**16字节代码实现** (avx2-inl.h:130-154):

```cpp
struct HammingComputer16 {
    uint64_t a0, a1;  // 两个64位寄存器

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

### 3.4 HammingComputer32

**32字节代码实现** (avx2-inl.h:187-214):

```cpp
struct HammingComputer32 {
    uint64_t a0, a1, a2, a3;  // 四个64位寄存器

    HammingComputer32(const uint8_t* a8, int code_size) {
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

### 3.5 HammingComputerDefault

**通用长度实现** (avx2-inl.h:251-346):

```cpp
struct HammingComputerDefault {
    const uint8_t* a8;
    int quotient8;   // 完整的8字节块数
    int remainder8;  // 剩余字节数

    HammingComputerDefault(const uint8_t* a8, int code_size) {
        this->a8 = a8;
        quotient8 = code_size / 8;
        remainder8 = code_size % 8;
    }

    int hamming(const uint8_t* b8) const {
        int accu = 0;

        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b8);
        int i = 0, len = quotient8;

        // Duff's Device: 展开循环，处理8的倍数
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

        // 处理剩余字节 (查表法)
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

**Duff's Device技巧**:
- 通过switch和fallthrough实现循环展开
- 减少循环计数器的比较操作
- 提高指令级并行性

---

## 4. AVX-512硬件加速

### 4.1 VPOPCNTDQ指令集

**AVX-512 VPOPCNTDQ**: 提供向量popcount操作，每个时钟周期可处理512位。

**硬件要求**:
- Intel Xeon Scalable (Sapphire Rapids) 或更新
- 需要编译选项 `-mavx512vpopcntdq`

### 4.2 HammingComputer64优化

**AVX-512版本** (avx512-inl.h:221-263):

```cpp
struct HammingComputer64 {
    uint64_t a0, a1, a2, a3, a4, a5, a6, a7;
    const uint64_t* a;

    HammingComputer64(const uint8_t* a8, int code_size) {
        assert(code_size == 64);
        a = (uint64_t*)a8;
        a0 = a[0];
        a1 = a[1];
        a2 = a[2];
        a3 = a[3];
        a4 = a[4];
        a5 = a[5];
        a6 = a[6];
        a7 = a[7];
    }

    inline int hamming(const uint8_t* b8) const {
        const uint64_t* b = (uint64_t*)b8;
#ifdef __AVX512VPOPCNTDQ__
        // AVX-512硬件加速版本
        __m512i vxor = _mm512_xor_si512(
                _mm512_loadu_si512(a),
                _mm512_loadu_si512(b));
        __m512i vpcnt = _mm512_popcnt_epi64(vxor);
        // reduce操作比分开添加上下半部分更高效
        return _mm512_reduce_add_epi32(vpcnt);
#else
        // 标量版本
        return popcount64(b[0] ^ a0) + popcount64(b[1] ^ a1) +
               popcount64(b[2] ^ a2) + popcount64(b[3] ^ a3) +
               popcount64(b[4] ^ a4) + popcount64(b[5] ^ a5) +
               popcount64(b[6] ^ a6) + popcount64(b[7] ^ a7);
#endif
    }
};
```

**性能对比**:

| 实现方式 | 指令数 | 延迟 (约) |
|----------|--------|-----------|
| 标量版本 | 8 XOR + 8 popcnt + 7 add | ~20 cycles |
| AVX-512 | 1 load + 1 load + 1 xor + 1 popcnt + 1 reduce | ~8 cycles |

### 4.3 HammingComputerDefault优化

**AVX-512通用版本** (avx512-inl.h:282-369):

```cpp
int hamming(const uint8_t* b8) const {
    int accu = 0;

    const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a8);
    const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b8);

    int i = 0;
#ifdef __AVX512VPOPCNTDQ__
    // 处理512位块 (64字节)
    int quotient64 = quotient8 / 8;
    for (; i < quotient64; ++i) {
        __m512i vxor = _mm512_xor_si512(
                _mm512_loadu_si512(&a64[i * 8]),
                _mm512_loadu_si512(&b64[i * 8]));
        __m512i vpcnt = _mm512_popcnt_epi64(vxor);
        accu += _mm512_reduce_add_epi32(vpcnt);
    }
    i *= 8;  // 转换为64位字索引
#endif

    // 处理剩余64位字
    int len = quotient8 - i;
    switch (len & 7) {
        // ... Duff's Device处理剩余部分
    }

    // 处理剩余字节
    if (remainder8) {
        // ... 查表法处理
    }

    return accu;
}
```

**优化策略**:
1. **分块处理**: 512位块用AVX-512，剩余部分用标量
2. **提前退出**: 只在支持VPOPCNTDQ时使用AVX-512路径
3. **混合模式**: 结合SIMD和标量代码，最大化性能

---

## 5. 广义Hamming距离

### 5.1 定义

**广义Hamming距离**: 计算两个代码中不同字节的个数，而非不同比特的个数。

```
GH(x, y) = Σ[x_i ≠ y_i]  (按字节比较)
```

### 5.2 GenHammingComputer8

**8字节实现** (avx2-inl.h:361-376):

```cpp
// 广义Hamming距离优化: 64位特化
inline int generalized_hamming_64(uint64_t a) {
    // 将每个字节的所有位设置为相同值
    a |= a >> 1;   // 复制高位到相邻低位
    a |= a >> 2;   // 继续扩展
    a |= a >> 4;   // 每个字节现在要么全0要么全1
    a &= 0x0101010101010101UL;  // 只保留最低位
    return popcount64(a);  // 统计1的个数 = 不同字节数
}

struct GenHammingComputer8 {
    uint64_t a0;

    GenHammingComputer8(const uint8_t* a, int code_size) {
        assert(code_size == 8);
        a0 = *(uint64_t*)a;
    }

    inline int hamming(const uint8_t* b) const {
        return generalized_hamming_64(*(uint64_t*)b ^ a0);
    }
};
```

**算法图解**:

```
示例: a = 0xFF00AA55, b = 0xFF01AA54

步骤1: XOR
x = a ^ b = 0x00000001

步骤2: a |= a >> 1
x = 0x00000001 | 0x00000000 = 0x00000001

步骤3: a |= a >> 2
x = 0x00000001 | 0x00000000 = 0x00000001

步骤4: a |= a >> 4
x = 0x00000001 | 0x00000000 = 0x00000001

步骤5: a &= 0x0101010101010101
x = 0x00000001 & 0x0101010101010101 = 0x00000001

步骤6: popcount
result = popcount64(0x00000001) = 1
```

### 5.3 GenHammingComputer16 (AVX2)

**16字节SIMD实现** (avx2-inl.h:380-398):

```cpp
struct GenHammingComputer16 {
    __m128i a;  // 128位SIMD寄存器

    GenHammingComputer16(const uint8_t* a8, int code_size) {
        assert(code_size == 16);
        a = _mm_loadu_si128((const __m128i_u*)a8);
    }

    inline int hamming(const uint8_t* b8) const {
        const __m128i b = _mm_loadu_si128((const __m128i_u*)b8);
        // 按字节比较相等
        const __m128i cmp = _mm_cmpeq_epi8(a, b);
        // movemask生成16位掩码
        const auto movemask = _mm_movemask_epi8(cmp);
        // 16 - popcount = 不相等字节数
        return 16 - popcount32(movemask);
    }
};
```

**SIMD指令分析**:

| 指令 | 功能 | 延迟 |
|------|------|------|
| `_mm_loadu_si128` | 加载16字节 | 1 |
| `_mm_cmpeq_epi8` | 并行16个字节比较 | 1 |
| `_mm_movemask_epi8` | 提取符号位为16位掩码 | 3 |
| `popcount32` | 统计1的个数 | 3 |

### 5.4 GenHammingComputer32 (AVX2)

**32字节实现** (avx2-inl.h:400-418):

```cpp
struct GenHammingComputer32 {
    __m256i a;  // 256位SIMD寄存器

    GenHammingComputer32(const uint8_t* a8, int code_size) {
        assert(code_size == 32);
        a = _mm256_loadu_si256((const __m256i_u*)a8);
    }

    inline int hamming(const uint8_t* b8) const {
        const __m256i b = _mm256_loadu_si256((const __m256i_u*)b8);
        const __m256i cmp = _mm256_cmpeq_epi8(a, b);
        const uint32_t movemask = _mm256_movemask_epi8(cmp);
        return 32 - popcount32(movemask);
    }
};
```

**优化效果**:
- AVX2并行处理32个字节
- 单次调用完成比较
- 相比标量版本约8倍加速

### 5.5 GenHammingComputerM8

**多块处理实现** (avx2-inl.h:426-458):

```cpp
struct GenHammingComputerM8 {
    const uint64_t* a;
    int n;  // 64位字数量

    GenHammingComputerM8(const uint8_t* a8, int code_size) {
        assert(code_size % 8 == 0);
        a = (uint64_t*)a8;
        n = code_size / 8;
    }

    int hamming(const uint8_t* b8) const {
        const uint64_t* b = (uint64_t*)b8;
        int accu = 0;

        int i = 0;
        int n4 = (n / 4) * 4;  // 4的倍数，对应32字节
        // 处理32字节块 (AVX2)
        for (; i < n4; i += 4) {
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
};
```

---

## 6. KNN搜索优化

### 6.1 堆方法 (hammings_knn_hc)

**实现** (hamming.cpp:170-240):

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
        ApproxTopK_mode_t approx_topk_mode = EXACT_TOPK,
        const faiss::IDSelector* sel = nullptr) {
    size_t k = ha->k;
    if (init_heap) {
        ha->heapify();
    }

    // 分批处理以适应缓存
    const size_t block_size = hamming_batch_size;  // 65536
    for (size_t j0 = 0; j0 < n2; j0 += block_size) {
        const size_t j1 = std::min(j0 + block_size, n2);

#pragma omp parallel for
        for (int64_t i = 0; i < ha->nh; i++) {
            HammingComputer hc(bs1 + i * bytes_per_code, bytes_per_code);

            const uint8_t* __restrict bs2_ = bs2 + j0 * bytes_per_code;
            hamdis_t dis;
            hamdis_t* __restrict bh_val_ = ha->val + i * k;
            int64_t* __restrict bh_ids_ = ha->ids + i * k;

            // 根据近似模式选择实现
            switch (approx_topk_mode) {
                case APPROX_TOPK_BUCKETS_B8_D3:
                    HeapWithBucketsForHamming32<
                            CMax<hamdis_t, int64_t>, 8, 3,
                            HammingComputer>::addn(...);
                    break;
                case APPROX_TOPK_BUCKETS_B8_D2:
                    HeapWithBucketsForHamming32<
                            CMax<hamdis_t, int64_t>, 8, 2,
                            HammingComputer>::addn(...);
                    break;
                default:
                    // 精确Top-K
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
                    break;
            }
        }
    }

    if (order) {
        ha->reorder();
    }
}
```

**性能优化要点**:
1. **分批处理**: 减少缓存未命中
2. **OpenMP并行**: 查询间并行
3. **预取友好**: bs1_在循环中保持不变

### 6.2 计数方法 (hammings_knn_mc)

**HCounterState** (hamming-inl.h:76-119):

```cpp
template <class HammingComputer>
struct HCounterState {
    int* counters;        // 每个距离的计数
    int64_t* ids_per_dis; // 每个距离的ID列表

    HammingComputer hc;
    int thres;    // 当前阈值
    int count_lt; // 小于阈值的数量
    int count_eq; // 等于阈值的数量
    int k;

    void update_counter(const uint8_t* y, size_t j) {
        int32_t dis = hc.hamming(y);

        if (dis <= thres) {
            if (dis < thres) {
                // 添加到新距离桶
                ids_per_dis[dis * k + counters[dis]++] = j;
                ++count_lt;

                // 调整阈值
                while (count_lt == k && thres > 0) {
                    --thres;
                    count_eq = counters[thres];
                    count_lt -= count_eq;
                }
            } else if (count_eq < k) {
                // 添加到当前阈值桶
                ids_per_dis[dis * k + count_eq++] = j;
                counters[dis] = count_eq;
            }
        }
    }
};
```

**MC搜索实现** (hamming.cpp:243-299):

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
    const int nBuckets = bytes_per_code * 8 + 1;

    // 为每个查询分配计数器数组
    std::vector<int> all_counters(na * nBuckets, 0);
    std::unique_ptr<int64_t[]> all_ids_per_dis(
            new int64_t[na * nBuckets * k]);

    // 初始化计数器状态
    std::vector<HCounterState<HammingComputer>> cs;
    for (size_t i = 0; i < na; ++i) {
        cs.push_back(HCounterState<HammingComputer>(
                all_counters.data() + i * nBuckets,
                all_ids_per_dis.get() + i * nBuckets * k,
                a + i * bytes_per_code,
                8 * bytes_per_code,
                k));
    }

    // 分批扫描数据库
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

    // 从计数器提取Top-K结果
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

        // 填充不足的结果
        while (nres < k) {
            labels[i * k + nres] = -1;
            distances[i * k + nres] = std::numeric_limits<int32_t>::max();
            ++nres;
        }
    }
}
```

**MC vs HC选择**:

| 方法 | 适用场景 | 时间复杂度 | 空间复杂度 |
|------|----------|-----------|-----------|
| HC (堆) | k << n, 任意距离 | O(na × nb × log k) | O(na × k) |
| MC (计数) | k ≈ n, 小距离范围 | O(na × nb) | O(na × nBuckets × k) |

---

## 7. 位串读写器

### 7.1 BitstringWriter

**写入器实现** (hamming-inl.h:13-37):

```cpp
struct BitstringWriter {
    uint8_t* code;
    size_t code_size;  // 字节为单位
    size_t i;          // 当前比特偏移

    BitstringWriter(uint8_t* code, size_t code_size)
            : code(code), code_size(code_size), i(0) {
        memset(code, 0, code_size);
    }

    // 写入x的低nbit位
    void write(uint64_t x, int nbit) {
        assert(code_size * 8 >= nbit + i);

        // 当前字节剩余可用比特数
        int na = 8 - (i & 7);

        if (nbit <= na) {
            // 可以放入当前字节
            code[i >> 3] |= x << (i & 7);
            i += nbit;
            return;
        } else {
            // 跨字节写入
            size_t j = i >> 3;
            code[j++] |= x << (i & 7);
            i += nbit;
            x >>= na;

            while (x != 0) {
                code[j++] |= x;
                x >>= 8;
            }
        }
    }
};
```

**写入示例**:

```
code = [0x00, 0x00, 0x00]
i = 0

write(0b1011, 4):  // 写入4位
na = 8
code[0] |= 0b1011 << 0 = 0x0B
i = 4
code = [0x0B, 0x00, 0x00]

write(0b11011011, 8):  // 写入8位
na = 4
code[0] |= 0b11011011 << 4 = 0xBB
i = 12, x >>= 4 = 0b1101
code[1] |= 0b1101 = 0x0D
i = 20
code = [0xBB, 0x0D, 0x00]
```

### 7.2 BitstringReader

**读取器实现** (hamming-inl.h:39-67):

```cpp
struct BitstringReader {
    const uint8_t* code;
    size_t code_size;
    size_t i;

    BitstringReader(const uint8_t* code, size_t code_size)
            : code(code), code_size(code_size), i(0) {}

    // 读取nbit位
    uint64_t read(int nbit) {
        assert(code_size * 8 >= nbit + i);

        // 当前字节剩余可用比特数
        int na = 8 - (i & 7);

        // 获取当前字节的可用位
        uint64_t res = code[i >> 3] >> (i & 7);

        if (nbit <= na) {
            // 全部在当前字节内
            res &= (1 << nbit) - 1;
            i += nbit;
            return res;
        } else {
            // 跨字节读取
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
};
```

### 7.3 位串打包/解包

**打包函数** (hamming.cpp:716-733):

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

**解包函数** (hamming.cpp:758-775):

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

---

## 8. 应用示例

### 8.1 浮点向量转二值化

**实现** (hamming.cpp:368-382):

```cpp
void fvec2bitvec(const float* __restrict x, uint8_t* __restrict b, size_t d) {
    for (int i = 0; i < d; i += 8) {
        uint8_t w = 0;
        uint8_t mask = 1;
        int nj = i + 8 <= d ? 8 : d - i;
        for (int j = 0; j < nj; j++) {
            if (x[i + j] >= 0) {
                w |= mask;
            }
            mask <<= 1;
        }
        *b = w;
        b++;
    }
}

// 批量转换
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

### 8.2 位洗牌

**功能**: 按指定顺序重排比特位 (hamming.cpp:436-460):

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

**应用**: 随机投影后重排比特位以优化缓存访问模式。

---

## 性能优化技巧总结

### 1. SIMD指令选择

| 操作 | SSE4.1 | AVX2 | AVX-512 |
|------|--------|------|---------|
| Popcount | scalar | scalar | `_mm512_popcnt_epi64` |
| 字节比较 | `_mm_cmpeq_epi8` | `_mm256_cmpeq_epi8` | `_mm512_cmpeq_epi8` |
| Movemask | `_mm_movemask_epi8` | `_mm256_movemask_epi8` | `_mm512_movemask_epi8` |

### 2. 算法优化

| 技术 | 效果 | 场景 |
|------|------|------|
| 模板特化 | 编译期展开 | 固定代码长度 |
| Duff's Device | 循环展开 | 长代码处理 |
| 分批处理 | 缓存友好 | 大规模搜索 |
| 计数方法 | O(1)更新 | 小距离范围 |

### 3. 内存优化

| 技术 | 说明 | 实现 |
|------|------|------|
| 预加载查询 | 寄存器缓存 | HammingComputer |
| 分块扫描 | 减少未命中 | block_size = 65536 |
| 对齐加载 | 加速访问 | `_mm_load_si128` vs `_mm_loadu_si128` |

### 4. 并行化策略

| 层级 | 方法 | 粒度 |
|------|------|------|
| 查询级 | OpenMP并行 | 每线程处理多个查询 |
| 块级 | 分批处理 | 65536个向量/批 |
| 数据级 | SIMD指令 | 8-64字节/指令 |

---

## 参考资料

1. **相关源码**:
   - `faiss/utils/hamming.{h,cpp}` - Hamming距离高层API
   - `faiss/utils/hamming_distance/common.h` - 通用定义
   - `faiss/utils/hamming_distance/avx2-inl.h` - AVX2实现
   - `faiss/utils/hamming_distance/avx512-inl.h` - AVX-512实现
   - `faiss/utils/hamming_distance/neon-inl.h` - ARM NEON实现
   - `faiss/utils/hamming-inl.h` - HCounterState等辅助类

2. **硬件文档**:
   - Intel® AVX-512: `_mm512_popcnt_epi64`
   - ARM NEON: `vcntq_u8`

3. **算法参考**:
   - Hamming distance computation optimization
   - Duff's Device loop unrolling
   - Bucket-based Top-K algorithms

---

*本文档详细剖析了Faiss中Hamming距离计算的底层实现，包括SIMD优化的popcount、专用HammingComputer类、AVX-512硬件加速、广义Hamming距离和KNN搜索优化算法。*
