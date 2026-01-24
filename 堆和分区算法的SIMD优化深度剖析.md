# 堆与分区算法的SIMD优化深度剖析

## 课程简介

本课程深入剖析Faiss中底层数据结构(堆、分区算法)的SIMD优化实现，这些是向量搜索性能的关键组件。

**前置知识**:
- 已完成《SIMD底层优化深度剖析》
- 熟悉AVX2/AVX-512指令集
- 了解堆和分区算法的基本原理

**学习目标**:
- 掌握堆操作的优化实现
- 理解分区算法的SIMD优化
- 学习直方图计算的SIMD技巧
- 理解编码打包的底层实现

---

## 第一部分:堆数据结构优化

### 1.1 堆的基本操作

Faiss中使用了高效的堆实现,这是Top-K搜索的核心:

```cpp
// faiss/utils/Heap.h

// 堆替换: 替换堆顶元素
template <class C>
inline void heap_replace_top(
        size_t k,
        typename C::T* bh_val,  // 值数组
        typename C::TI* bh_ids,  // 索引数组
        typename C::T val,
        typename C::TI id) {

    // 使用1-based索引简化子节点计算
    bh_val--; bh_ids--;

    size_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1;    // 左子节点: 2*i
        i2 = i1 + 1;   // 右子节点: 2*i + 1

        if (i1 > k) break;

        // 找到较大的子节点
        if ((i2 == k + 1) ||
            C::cmp2(bh_val[i1], bh_val[i2], bh_ids[i1], bh_ids[i2])) {
            if (C::cmp2(val, bh_val[i1], id, bh_ids[i1])) {
                break;
            }
            bh_val[i] = bh_val[i1];
            bh_ids[i] = bh_ids[i1];
            i = i1;
        } else {
            if (C::cmp2(val, bh_val[i2], id, bh_ids[i2])) {
                break;
            }
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            i = i2;
        }
    }

    bh_val[i] = val;
    bh_ids[i] = id;
}
```

### 1.2 堆初始化优化

```cpp
// 快速堆初始化 (heapify)
template <class C>
inline void heap_heapify(size_t k, typename C::T* bh_val, typename C::TI* bh_ids) {
    // 从最后一个非叶子节点开始,向上调整
    for (size_t i = k / 2; i > 0; i--) {
        // 调整以i为根的子堆
        size_t j = i;
        while (1) {
            size_t j1 = j << 1;
            size_t j2 = j1 + 1;

            if (j1 > k) break;

            size_t j_max = j1;
            if (j2 <= k &&
                C::cmp2(bh_val[j2], bh_val[j1], bh_ids[j2], bh_ids[j1])) {
                j_max = j2;
            }

            if (C::cmp2(bh_val[j_max], bh_val[j], bh_ids[j_max], bh_ids[j])) {
                break;
            }

            // 交换
            std::swap(bh_val[j], bh_val[j_max]);
            std::swap(bh_ids[j], bh_ids[j_max]);
            j = j_max;
        }
    }
}
```

### 1.3 SIMD优化的批量堆操作

```cpp
// SIMD优化的批量堆初始化
void simd_heapify_max(
        float* values,
        int64_t* ids,
        size_t n,   // 数组大小
        size_t k) { // 堆大小

    // 使用SIMD同时处理多个元素
    // 这里使用4路展开减少依赖链

    // 每次处理4个父节点
    for (size_t i = (k / 4); i >= 1; i--) {
        size_t parent = i;

        // 处理parent的4个叶子节点
        for (int level = 0; level < 2; level++) {
            size_t child1 = parent << (level + 1);

            // 批量比较4对父子节点
            float v_parent = values[parent - 1];

            // SIMD加载4个子节点
            __m128 v_children = _mm_loadu_ps(values + child1 - 1);

            // 比较并找到最大
            __m128 v_parent = _mm_set1_ps(v_parent);
            __m128 cmp = _mm_cmplt_ps(v_parent, v_children);

            // 如果子节点更大,需要交换
            if (_mm_movemask_ps(cmp) != 0) {
                // 找到最大的子节点
                int max_idx = 0;
                float max_val = values[child1 - 1];

                if (values[child1] > max_val) {
                    max_val = values[child1];
                    max_idx = child1;
                }
                if (values[child1 + 1] > max_val) {
                    max_val = values[child1 + 1];
                    max_idx = child1 + 1;
                }
                // ... (比较所有子节点)

                // 交换
                std::swap(values[parent - 1], values[max_idx - 1]);
                std::swap(ids[parent - 1], ids[max_idx - 1]);
            }

            parent = max_idx;
        }
    }
}
```

### 1.4 堆排序优化

```cpp
// SIMD优化的堆排序
void simd_heap_sort(
        float* values,
        int64_t* ids,
        size_t n) {

    // 1. 建堆
    heap_heapify<CMax<float, int64_t>>(n, values, ids);

    // 2. 逐个提取最大元素
    for (size_t i = n; i > 1; i--) {
        // 堆顶(最大值)已经就位
        // 将最后一个元素提到堆顶,然后heapify

        float max_val = values[0];
        int64_t max_id = ids[0];

        // 用最后一个元素替换堆顶
        values[0] = values[i - 1];
        ids[0] = ids[i - 1];

        // 恢复堆性质
        heap_pop<CMax<float, int64_t>>(i, values, ids);

        // 存储排序结果
        values[i - 1] = max_val;
        ids[i - 1] = max_id;
    }
}
```

---

## 第二部分:分区算法的SIMD优化

### 2.1 SIMD分区算法

分区是Quickselect的核心,用于寻找Top-K元素:

```cpp
// 标量分区 (基准版本)
template <class C>
typename C::T partition_scalar(
        typename C::T* vals,
        typename C::TI* ids,
        size_t n,
        size_t q) {

    // 选择pivot (中位数)
    typename C::T pivot = vals[q];

    size_t left = 0;
    size_t right = n - 1;

    while (true) {
        // 从左找第一个 >= pivot 的元素
        while (left <= right && C::cmp(vals[left], pivot)) {
            left++;
        }

        // 从右找第一个 <= pivot 的元素
        while (left <= right && C::cmp(pivot, vals[right])) {
            right--;
        }

        if (left > right) break;

        // 交换
        std::swap(vals[left], vals[right]);
        std::swap(ids[left], ids[right]);
        left++;
        right--;
    }

    return left;  // 返回分区点
}
```

### 2.2 SIMD优化的分区

```cpp
// AVX2优化的分区算法
#ifdef __AVX2__

template <class C>
typename C::T partition_avx2(
        typename C::T* vals,
        typename C::TI* ids,
        size_t n,
        size_t q) {

    // 1. 处理8元素对齐块
    size_t i = 0;
    size_t left = 0;

    // 生成比较掩码
    typename C::T pivot = vals[q];
    __m256 vpivot = _mm256_set1_ps(pivot);

    for (; i + 8 <= n; i += 8) {
        __m256 vvals = _mm256_loadu_ps(vals + i);

        // 比较: vals < pivot
        __m256 cmp = _mm256_cmplt_ps(vvals, vpivot);

        // 统计小于pivot的元素数
        int mask = _mm256_movemask_ps(cmp);
        int n_lt = __builtin_popcount(mask);

        // 将小于pivot的元素移到左边
        // 这里需要复杂的shuffle操作
        // ... (实现省略,非常复杂)
    }

    // 2. 处理剩余元素
    // 使用标量代码

    return partition_scalar<C>(vals, ids, n, left);
}

// AVX-512优化的分区 (使用掩码操作)
#ifdef __AVX512F__

template <class C>
typename C::T partition_avx512(
        typename C::T* vals,
        typename C::TI* ids,
        size_t n,
        size_t q) {

    typename C::T pivot = vals[q];
    __m512 vpivot = _mm512_set1_ps(pivot);

    // 使用掩码和compress操作
    size_t i = 0;
    size_t write_left = 0;
    size_t write_right = n - 1;

    for (; i + 16 <= n; i += 16) {
        __m512 vvals = _mm512_loadu_ps(vals + i);
        __m512i vids = _mm512_loadu_si512((__m512i*)(ids + i));

        // 比较: vals < pivot
        __mmask16 mask_lt = _mm512_cmplt_ps_mask(vvals, vpivot);

        // 统计
        int n_lt = _mm_popcnt_u32(mask_lt);

        // 分别提取小于和大于pivot的元素
        if (n_lt > 0) {
            // 压缩小于pivot的元素
            __m512 v_lt = _mm512_maskz_compress_ps(vvals, mask_lt);
            __m512i id_lt = _mm512_maskz_compress_epi32(vids, mask_lt);

            // 存储到左边
            _mm512_mask_storeu_ps(vals + write_left, mask_lt, v_lt);
            // (需要处理ids)
            write_left += n_lt;
        }

        if (n_lt < 16) {
            // 压缩大于pivot的元素
            __mmask16 mask_gt = ~mask_lt & 0xFFFF;
            __m512 v_gt = _mm512_maskz_compress_ps(vvals, mask_gt);

            // 存储到右边
            int n_gt = 16 - n_lt;
            _mm512_mask_storeu_ps(vals + write_right - n_gt + 1, mask_gt, v_gt);
            write_right -= n_gt;
        }
    }

    // 处理剩余元素
    for (; i < n; i++) {
        if (C::cmp(vals[i], pivot)) {
            std::swap(vals[i], vals[write_left]);
            std::swap(ids[i], ids[write_left]);
            write_left++;
        }
    }

    return write_left;
}
#endif
```

### 2.3 模糊分区 (Fuzzy Partition)

```cpp
// 模糊分区: 允许在[q_min, q_max]范围内的任意位置分区
template <class C>
typename C::T partition_fuzzy(
        typename C::T* vals,
        typename C::TI* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out) {

    // 使用median-of-3选择更好的pivot
    size_t n1 = q_min;
    size_t n2 = (q_min + q_max) / 2;
    size_t n3 = q_max;

    typename C::T v1 = vals[n1];
    typename C::T v2 = vals[n2];
    typename C::T v3 = vals[n3];

    // 中位数
    typename C::T pivot = C::median3(v1, v2, v3);

    // 执行分区
    size_t q = partition<C>(vals, ids, n, pivot);

    if (q_out) {
        *q_out = q;
    }

    // 返回分区阈值(pivot值)
    return pivot;
}
```

---

## 第三部分:SIMD直方图计算

### 3.1 8-bin直方图

```cpp
// AVX2优化的8-bin直方图计算
// 输入: uint16数组, 范围[min, min+256*8)
// 输出: 8个bin的计数

namespace {

// 2位累加器: 只能累加最多3个元素
// 输出: 2个4-bit结果
template <int N, class Preproc>
void compute_accu2(
        const uint16_t*& data,
        Preproc& pp,
        simd16uint16& a4lo,
        simd16uint16& a4hi) {

    simd16uint16 mask2(0x3333);
    simd16uint16 a2((uint16_t)0);  // 2-bit累加器

    for (int j = 0; j < N; j++) {
        simd16uint16 v(data);
        data += 16;
        v = pp(v);

        // 构造索引: 0x800用于处理越界
        simd16uint16 idx = v | (v << 8) | simd16uint16(0x800);

        // 使用查找表累加
        simd16uint16 shifts_lookup = shifts.lookup_2_lanes(
            simd32uint8(idx)
        );

        a2 += shifts_lookup & mask2;
    }

    // 分离到高低4位
    a4lo += a2 & mask2;
    a4hi += (a2 >> 2) & mask2;
}

// 4-bit到8-bit的扩展
simd32uint8 accu4to8(simd16uint16 a4) {
    simd16uint16 mask4(0x0f0f);

    simd16uint16 a8_0 = a4 & mask4;
    simd16uint16 a8_1 = (a4 >> 4) & mask4;

    return simd32uint8(hadd(a8_0, a8_1));
}

} // anonymous namespace

// 主函数: 8-bin直方图
void simd_histogram_8(
        const uint16_t* data,
        int n,
        uint16_t min,
        int shift,
        int* hist) {

    assert(n % 16 == 0);  // 需要16的倍数

    int n_vecs = n / 16;

    // 初始化8位累加器
    simd32uint8 a8lo(0);
    simd32uint8 a8hi(0);

    // 每次处理15个向量(45个元素)
    for (int i0 = 0; i0 < n_vecs; i0 += 15) {
        simd16uint16 a4lo(0);  // 4-bit累加器
        simd16uint16 a4hi(0);

        int i1 = std::min(i0 + 15, n_vecs);
        int i;

        // 每次处理3个向量(最多9个元素)
        for (i = i0; i + 2 < i1; i += 3) {
            compute_accu2<3>(data, a4lo, a4hi);
        }

        // 处理剩余向量
        switch (i1 - i) {
            case 2:
                compute_accu2<2>(data, a4lo, a4hi);
                break;
            case 1:
                compute_accu2<1>(data, a4lo, a4hi);
                break;
        }

        // 合并到8位累加器
        a8lo += accu4to8(a4lo);
        a8hi += accu4to8(a4hi);
    }

    // 将16位累加器合并为最终结果
    simd16uint16 a16lo = accu8to16(a8lo);
    simd16uint16 a16hi = accu8to16(a8hi);
    simd16uint16 a16 = hadd(a16lo, a16hi);

    // 存储结果
    ALIGNED(32) uint16_t a16_tab[16];
    a16.store(a16_tab);

    for (int i = 0; i < 8; i++) {
        hist[i] = a16_tab[i] + a16_tab[i + 8];
    }
}
```

### 3.2 16-bin直方图

```cpp
// AVX2优化的16-bin直方图
template <class Preproc>
simd16uint16 histogram_16(
        const uint16_t* data,
        Preproc pp,      // 预处理函数
        size_t n_in) {

    assert(n_in % 16 == 0);
    int n = n_in / 16;

    // 32个8位累加器
    // 实际上分为4组,每组8个
    simd32uint8 a8lo(0);
    simd32uint8 a8hi(0);

    for (int i0 = 0; i0 < n; i0 += 7) {
        // 4组4位累加器
        simd32uint8 a4_0(0);
        simd32uint8 a4_1(0);
        simd32uint8 a4_2(0);
        simd32uint8 a4_3(0);

        int i1 = std::min(i0 + 7, n);
        int i;

        // 每次处理3个向量
        for (i = i0; i + 2 < i1; i += 3) {
            compute_accu2_16<3>(data, pp, a4_0, a4_1, a4_2, a4_3);
        }

        // 处理剩余
        switch (i1 - i) {
            case 2:
                compute_accu2_16<2>(data, pp, a4_0, a4_1, a4_2, a4_3);
                break;
            case 1:
                compute_accu2_16<1>(data, pp, a4_0, a4_1, a4_2, a4_3);
                break;
        }

        // 合并到8位累加器
        a8lo += accu4to8_2(a4_0, a4_1);
        a8hi += accu4to8_2(a4_2, a4_3);
    }

    // 最终合并
    simd16uint16 a16lo = accu8to16(a8lo);
    simd16uint16 a16hi = accu8to16(a8hi);
    simd16uint16 a16 = hadd(a16lo, a16hi);

    // 解包
    a16 = simd16uint16{simd8uint32{a16}.unzip()};

    return a16;
}
```

### 3.3 预处理函数

```cpp
// 预处理: 减去最小值并移位
template <int shift, int nbin>
struct PreprocMinShift {
    simd16uint16 min16;
    simd16uint16 max16;

    explicit PreprocMinShift(uint16_t min) {
        min16.set1(min);

        // 计算最大值(包含)
        int vmax0 = std::min((nbin << shift) + min, 65536);
        uint16_t vmax = uint16_t(vmax0 - 1 - min);
        max16.set1(vmax);
    }

    simd16uint16 operator()(simd16uint16 x) {
        // 减去最小值
        x = x - min16;

        // 生成越界掩码
        simd16uint16 mask = (x == max(x, max16)) -
                            (x == max16);

        // 移位并处理越界
        return (x >> shift) | mask;
    }
};

// 无预处理
struct PreprocNOP {
    simd16uint16 operator()(simd16uint16 x) {
        return x;
    }
};
```

---

## 第四部分:4-bit PQ编码打包

### 4.1 编码打包原理

```cpp
// faiss/impl/pq4_fast_scan.cpp

// 打包4-bit PQ编码
// 输入: nb × M字节的编码
// 输出: (nb/bbs) × (M/2) × bbs 的块格式

void pq4_pack_codes(
        const uint8_t* codes,   // 原始编码: nb × M
        size_t ntotal,
        size_t M,
        size_t nb,
        size_t bbs,            // 块大小(32的倍数)
        size_t nsq,            // 子量化器数(M的对齐版本)
        uint8_t* blocks) {

    // 输入验证
    FAISS_THROW_IF_NOT(bbs % 32 == 0);
    FAISS_THROW_IF_NOT(nb % bbs == 0);
    FAISS_THROW_IF_NOT(nsq % 2 == 0);

    // 字节序处理(大端/小端)
#ifdef FAISS_BIG_ENDIAN
    const uint8_t perm0[16] = {
        8, 0, 9, 1, 10, 2, 11, 3,
        12, 4, 13, 5, 14, 6, 15, 7
    };
#else
    const uint8_t perm0[16] = {
        0, 8, 1, 9, 2, 10, 3, 11,
        4, 12, 5, 13, 6, 14, 7, 15
    };
#endif

    // 对于每个bbs大小的块
    for (size_t i0 = 0; i0 < nb; i0 += bbs) {
        for (int sq = 0; sq < nsq; sq += 2) {
            // 对于块中的每32个向量
            for (size_t i = 0; i < bbs; i += 32) {
                // 收集32个向量的第sq/2个子量化器编码
                std::array<uint8_t, 32> c;
                get_matrix_column(
                    codes, ntotal, (M + 1) / 2,
                    i0 + i, sq / 2, c);

                // 分离高低4位
                std::array<uint8_t, 32> c0, c1;
                for (int j = 0; j < 32; j++) {
                    c0[j] = c[j] & 15;      // 低4位
                    c1[j] = c[j] >> 4;       // 高4位
                }

                // 重排为交错格式
                for (int j = 0; j < 16; j++) {
                    uint8_t d0 = c0[perm0[j]] |
                                 (c0[perm0[j] + 16] << 4);
                    uint8_t d1 = c1[perm0[j]] |
                                 (c1[perm0[j] + 16] << 4);

                    blocks[j] = d0;
                    blocks[j + 16] = d1;
                }

                blocks += 32;
            }
        }
    }
}
```

### 4.2 编码访问优化

```cpp
// 从打包格式中获取特定向量的编码
uint8_t pq4_get_packed_element(
        const uint8_t* data,
        size_t bbs,
        size_t nsq,
        size_t vector_id,
        size_t sq) {

    // 移动到正确的块
    data += (vector_id / bbs) * (((nsq + 1) / 2) * bbs);

    // 确定向量在块内的位置
    vector_id = vector_id % bbs;

    // 确定是在高4位还是低4位
    bool shift = vector_id > 15;
    vector_id = vector_id & 15;

    // 计算在子量化器中的地址
    size_t address;
    if (vector_id < 8) {
        address = vector_id << 1;
    } else {
        address = ((vector_id - 8) << 1) + 1;
    }
    if (sq & 1) {
        address += 16;
    }

    // 提取编码
    address = (sq >> 1) * bbs + address;
    if (shift) {
        return data[address] >> 4;
    } else {
        return data[address] & 15;
    }
}
```

### 4.3 SIMD优化的批量编码访问

```cpp
// SIMD优化的批量编码获取
void pq4_get_packed_batch(
        const uint8_t* data,
        size_t bbs,
        size_t nsq,
        const size_t* vector_ids,
        uint8_t* codes,
        size_t batch_size) {

    // 对于每个子量化器
    for (size_t sq = 0; sq < nsq; sq++) {
        // 计算base地址
        const uint8_t* sq_base = data +
            ((vector_ids[0] / bbs) * (((nsq + 1) / 2) * bbs) +
            ((sq >> 1) * bbs);

        // SIMD优化: 一次处理16个向量
        size_t i = 0;
        for (; i + 16 <= batch_size; i += 16) {
            __m512i v_ids = _mm512_loadu_si512(
                (__m512i*)(vector_ids + i)
            );

            // 计算每个向量在块内的位置
            __m512i v_mod = _mm512_rem_epi32(
                v_ids, _mm512_set1_epi32(bbs)
            );

            // 判断高/低位
            __mmask16 shift_mask = _mm512_cmpgt_epi32_mask(
                v_mod, _mm512_setzero_si512()
            );

            // 计算地址偏移
            __m512i v_local = v_mod & _mm512_set1_epi32(15);

            // 复杂的地址计算...
            // (需要多步计算)

            // 加载编码数据
            __m512i v_codes = _mm512_loadu_si512(
                (__m512i*)sq_base
            );

            // 提取编码
            __mmask16 low_mask = ~shift_mask & 0xFFFF;
            __m512i v_low = _mm512_maskz_and_epi32(
                low_mask, v_codes, _mm512_set1_epi32(15)
            );
            __m512i v_high = _mm512_maskz_and_epi32(
                shift_mask, _mm512_srli_epi32(v_codes, 4),
                _mm512_set1_epi32(15)
            );

            // 合并
            __m512i v_result = _mm512_or_si512(v_low, v_high);

            // 存储结果
            _mm512_storeu_si512(
                (__m512i*)(codes + i * nsq + sq),
                v_result
            );
        }
    }
}
```

---

## 第五部分:查找表(LUT)优化

### 5.1 LUT打包

```cpp
// 打包查找表以优化内存访问
void pq4_pack_LUT(
        int nq,              // 查询数
        int nsq,             // 子量化器数
        const uint8_t* src, // 输入LUT: nq × nsq × 16
        uint8_t* dest) {     // 输出: 重新排列的LUT

    // 对于每个查询
    for (int q = 0; q < nq; q++) {
        // 对于每对子量化器(需要成对处理)
        for (int sq = 0; sq < nsq; sq += 2) {
            // 复制16字节(低子量化器)
            memcpy(
                dest + (sq / 2 * nq + q) * 32,
                src + (q * nsq + sq) * 16,
                16
            );

            // 复制16字节(高子量化器)
            memcpy(
                dest + (sq / 2 * nq + q) * 32 + 16,
                src + (q * nsq + sq + 1) * 16,
                16
            );
        }
    }
}
```

### 5.2 查询批处理(qbs)

```cpp
// 查询批处理: 将多个查询的LUT打包在一起
int pq4_pack_LUT_qbs(
        int qbs,             // 批大小(编码为4-bit)
        int nsq,
        const uint8_t* src, // 输入LUT
        uint8_t* dest) {    // 输出

    FAISS_THROW_IF_NOT(nsq % 2 == 0);
    size_t dim12 = 16 * nsq;

    int i0 = 0;
    int qi = qbs;

    // 解码qbs: 每个nibble表示4个查询
    while (qi) {
        int nq = qi & 15;  // 低4位
        qi >>= 4;

        // 打包nq个查询
        pq4_pack_LUT(nq, nsq, src + i0 * dim12, dest + i0 * dim12);
        i0 += nq;
    }

    return i0;  // 返回处理的查询总数
}
```

### 5.3 SIMD优化的LUT查找

```cpp
// 使用SIMD优化的LUT查找距离
// 这是FastScan搜索的核心

// AVX2版本
void pq4_accumulate_default_avx2(
        int n,                // 查询数
        size_t nb,            // 数据库向量数
        size_t bs,            // 块大小
        const uint8_t* LUT,   // 查找表: n × nsq × 16
        const uint8_t* codes, // 打包的编码
        int nsq,
        float* results) {

    for (int i = 0; i < n; i++) {
        const uint8_t* lut = LUT + i * nsq * 16;
        float sum = 0.0f;

        // 对于每个完整块
        for (size_t b = 0; b < nb; b++) {
            const uint8_t* block_codes = codes + b * nsq * bs;

            // 初始化累加器
            __m256i accu = _mm256_setzero_si256();

            // 对于每个子量化器
            for (int sq = 0; sq < nsq; sq += 2) {
                // 加载32个4-bit编码(交错存储)
                __m256i codes = _mm256_loadu_si256(
                    (__m256i*)(block_codes + sq * bs)
                );

                // 分离高低4位
                __m256i codes_low = _mm256_and_si256(
                    codes, _mm256_set1_epi8(0x0F)
                );
                __m256i codes_high = _mm256_and_si256(
                    _mm256_srli_epi16(codes, 4),
                    _mm256_set1_epi8(0x0F)
                );

                // 查表: 使用shuffle_epi8作为查找表
                __m256i lut_low = _mm256_shuffle_epi8(
                    _mm256_loadu_si256((__m256i*)(lut + sq * 16)),
                    codes_low
                );
                __m256i lut_high = _mm256_shuffle_epi8(
                    _mm256_loadu_si256((__m256i*)(lut + sq * 16)),
                    codes_high
                );

                // 扩展为16位并累加
                __m256i accu_16 = _mm256_unpacklo_epi8(
                    accu, accu
                );

                accu_16 = _mm256_adds_epi16(
                    accu_16,
                    _mm256_unpacklo_epi8(lut_low, lut_low)
                );
                accu_16 = _mm256_adds_epi16(
                    accu_16,
                    _mm256_unpacklo_epi8(lut_high, lut_high)
                );

                // 压缩回8位
                accu = _mm256_packs_epi16(accu_16, _mm256_setzero_si256());
            }

            // 水平归约
            sum += horizontal_sum_avx2(accu);
        }

        results[i] = sum;
    }
}
```

---

## 第六部分:性能优化技巧

### 6.1 内存访问模式优化

```cpp
// 优化前: 不连续访问
void bad_memory_access(
        const uint8_t* codes,
        size_t nsq,
        size_t bbs,
        float* lut,
        float* distances) {

    for (size_t i = 0; i < bbs; i++) {
        for (size_t sq = 0; sq < nsq; sq++) {
            // 跳跃访问
            uint8_t code = codes[i * nsq + sq];
            distances[i] += lut[sq * 16 + code];
        }
    }
}

// 优化后: 连续访问
void good_memory_access(
        const uint8_t* codes,
        size_t nsq,
        size_t bbs,
        float* lut,
        float* distances) {

    for (size_t i = 0; i < bbs; i += 16) {
        // 一次处理16个向量
        __m256i accu = _mm256_setzero_si256();

        for (size_t sq = 0; sq < nsq; sq += 2) {
            // 连续加载16个编码
            __m256i codes = _mm256_loadu_si256(
                (__m256i*)(codes + sq * bbs + i)
            );

            // ... (查表并累加)
        }

        // 存储结果
        distances[i] = horizontal_sum(accu);
    }
}
```

### 6.2 预取优化

```cpp
// 添加预取优化
void pq4_accumulate_with_prefetch(
        const uint8_t* LUT,
        const uint8_t* codes,
        size_t nq,
        size_t nb,
        size_t bbs,
        int nsq,
        float* results) {

    const size_t PREFETCH_DISTANCE = 4;

    for (int i = 0; i < nq; i++) {
        const uint8_t* lut = LUT + i * nsq * 16;

        for (size_t b = 0; b < nb; b++) {
            const uint8_t* block_codes = codes + b * nsq * bbs;

            // 预取未来的块
            if (b + PREFETCH_DISTANCE < nb) {
                _mm_prefetch(
                    (const char*)(codes + (b + PREFETCH_DISTANCE) * nsq * bbs),
                    _MM_HINT_T0
                );
            }

            // 计算距离
            float sum = 0.0f;
            for (int sq = 0; sq < nsq; sq += 2) {
                // ... (查表并累加)
            }

            results[b] = sum;
        }
    }
}
```

### 6.3 分块策略

```cpp
// 最优块大小选择
size_t optimal_block_size() {
    // 考虑因素:
    // 1. L1缓存: 32KB
    // 2. L2缓存: 256KB
    // 3. 数据大小: 每个编码1字节(M=16时为2字节)
    // 4. 查找表大小: nsq × 16字节

    // 目标: 使工作集适合L2缓存
    size_t lut_size = nsq * 16;
    size_t code_size = bbs * (nsq / 2);
    size_t total_size = lut_size + code_size;

    // 限制在128KB以内(适合L2缓存的一半)
    size_t max_bbs = (128 * 1024) / ((nsq / 2) * 8);

    return std::min(32, max_bbs & ~31);  // 32字节对齐
}
```

---

## 总结

本课程深入剖析了Faiss中堆、分区算法等底层数据结构的SIMD优化:

1. **堆优化**: 高效的heap_pop/heap_push实现
2. **分区算法**: SIMD优化的partition和fuzzy partition
3. **直方图**: 8-bin和16-bin的SIMD实现
4. **编码打包**: 4-bit PQ编码的高效打包和访问
5. **LUT优化**: 查找表的内存布局和访问优化

**关键要点**:
- 堆操作是Top-K搜索的核心,需要高效实现
- 分区算法是Quickselect的基础
- 直方图使用巧妙的2-bit累加器技术
- 4-bit编码需要仔细的内存布局设计
- LUT访问模式对性能至关重要

**下一步学习**:
- 《FastScan架构深度解析》
- 《IVF索引优化》
- 《实际性能调优案例》
