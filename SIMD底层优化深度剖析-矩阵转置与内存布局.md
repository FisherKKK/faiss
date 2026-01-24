# SIMD底层优化深度剖析 - 矩阵转置与内存布局

## 课程简介

本课程深入剖析Faiss中SIMD优化的底层实现细节,涵盖矩阵转置、内存布局优化、寄存器分配等高级技巧。

**前置知识**:
- 已完成《SIMD深度实践课程-AVX512与ARMNEON》
- 熟悉AVX2/AVX-512指令集
- 了解CPU缓存层次结构

**学习目标**:
- 理解SIMD矩阵转置的实现原理
- 掌握内存布局对SIMD性能的影响
- 学习高级SIMD优化技巧
- 理解Faiss中的底层优化策略

---

## 第一部分:SIMD矩阵转置

### 1.1 为什么需要矩阵转置

在向量搜索中,经常需要计算多个查询向量与数据库向量的距离。内存布局的优化对性能至关重要:

```cpp
// 两种常见的内存布局

// 布局1: 行优先 (Row-major)
// query vectors: nq × d
float* queries_row_major;  // [q0_d0, q0_d1, ..., q0_d(d-1), q1_d0, ...]

// database vectors: nb × d
float* database_row_major; // [v0_d0, v0_d1, ..., v0_d(d-1), v1_d0, ...]

// 布局2: 列优先 (Column-major)
// database vectors: d × nb (转置存储)
float* database_col_major; // [v0_d0, v1_d0, ..., v(nb-1)_d0, v0_d1, ...]
```

**性能影响分析**:

```cpp
// 情况1: 行优先,计算1个查询与nb个向量的距离
void case1_row_major() {
    // 访问模式: 跳跃访问,缓存不友好
    for (size_t i = 0; i < nb; i++) {
        for (size_t j = 0; j < d; j++) {
            // 每次访问database[i*d + j]
            // 缓存行浪费: 只用到1个float (4字节),却加载了64字节
            float x = database[i * d + j];
        }
    }
}

// 情况2: 列优先,计算1个查询与nb个向量的距离
void case2_col_major() {
    // 访问模式: 连续访问,缓存友好
    for (size_t j = 0; j < d; j++) {
        for (size_t i = 0; i < nb; i++) {
            // 连续访问database[j*nb + i]
            // 缓存行利用率高: 64字节包含16个float
            float x = database[j * nb + i];
        }
    }
}
```

### 1.2 AVX2矩阵转置实现

#### 1.2.1 8x2矩阵转置

这是Faiss中最常用的转置操作,用于批量距离计算:

```cpp
// faiss/utils/transpose/transpose-avx2-inl.h

// 输入: 8行2列 (实际上是2个256位寄存器)
// i0: [00, 01, 10, 11, 20, 21, 30, 31]
// i1: [40, 41, 50, 51, 60, 61, 70, 71]
// 其中 ij 表示第i行第j列的元素

// 输出: 2行8列 (转置后)
// o0: [00, 10, 20, 30, 40, 50, 60, 70]
// o1: [01, 11, 21, 31, 41, 51, 61, 71]

inline void transpose_8x2(
        const __m256 i0,
        const __m256 i1,
        __m256& o0,
        __m256& o1) {

    // 步骤1: 使用permute2f128重新组合128位块
    // r0: [00, 01, 10, 11, 40, 41, 50, 51]
    const __m256 r0 = _mm256_permute2f128_ps(i0, i1, _MM_SHUFFLE(0, 2, 0, 0));

    // r1: [20, 21, 30, 31, 60, 61, 70, 71]
    const __m256 r1 = _mm256_permute2f128_ps(i0, i1, _MM_SHUFFLE(0, 3, 0, 1));

    // 步骤2: 使用shuffle进行细粒度重排
    // o0: [00, 10, 20, 30, 40, 50, 60, 70]
    o0 = _mm256_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0));

    // o1: [01, 11, 21, 31, 41, 51, 61, 71]
    o1 = _mm256_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1));
}
```

**指令详解**:

```cpp
// permute2f128控制字: _MM_SHUFFLE(3, 2, 1, 0)
// 位[3:2] - 目标高128位来源
// 位[1:0] - 目标低128位来源

// 示例: _MM_SHUFFLE(0, 2, 0, 0) = 0x20
//   低128位来自 i0的低128位
//   高128位来自 i1的低128位

// shuffle控制字: _MM_SHUFFLE(z3, z2, z1, z0)
// 从两个输入中选择4个32位浮点数
```

#### 1.2.2 8x4矩阵转置

```cpp
// 输入: 8行4列
// i0: [00, 01, 02, 03, 10, 11, 12, 13]
// i1: [20, 21, 22, 23, 30, 31, 32, 33]
// i2: [40, 41, 42, 43, 50, 51, 52, 53]
// i3: [60, 61, 62, 63, 70, 71, 72, 73]

// 输出: 4行8列
// o0: [00, 10, 20, 30, 40, 50, 60, 70]
// o1: [01, 11, 21, 31, 41, 51, 61, 71]
// o2: [02, 12, 22, 32, 42, 52, 62, 72]
// o3: [03, 13, 23, 33, 43, 53, 63, 73]

inline void transpose_8x4(
        const __m256 i0, const __m256 i1,
        const __m256 i2, const __m256 i3,
        __m256& o0, __m256& o1,
        __m256& o2, __m256& o3) {

    // 第一阶段: permute2f128 - 重组128位块
    const __m256 r0 = _mm256_permute2f128_ps(i0, i2, _MM_SHUFFLE(0, 2, 0, 0));
    const __m256 r1 = _mm256_permute2f128_ps(i1, i3, _MM_SHUFFLE(0, 2, 0, 0));
    const __m256 r2 = _mm256_permute2f128_ps(i0, i2, _MM_SHUFFLE(0, 3, 0, 1));
    const __m256 r3 = _mm256_permute2f128_ps(i1, i3, _MM_SHUFFLE(0, 3, 0, 1));

    // 第二阶段: shuffle - 第一次细粒度重排
    const __m256 t0 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(2, 0, 2, 0));
    const __m256 t1 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(3, 1, 3, 1));
    const __m256 t2 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(2, 0, 2, 0));
    const __m256 t3 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(3, 1, 3, 1));

    // 第三阶段: shuffle - 第二次细粒度重排
    o0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(2, 0, 2, 0));
    o1 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(2, 0, 2, 0));
    o2 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 1, 3, 1));
    o3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 1, 3, 1));
}
```

**转置算法原理**:

这个转置算法使用了**分级shuffle策略**:

1. **粗粒度重组** (permute2f128): 处理128位边界
2. **细粒度重排** (shuffle): 处理32位浮点数级别
3. **多次迭代**: 逐步调整元素位置

```
初始状态 (8行4列):
[00 01 02 03] [10 11 12 13]
[20 21 22 23] [30 31 32 33]
[40 41 42 43] [50 51 52 53]
[60 61 62 63] [70 71 72 73]

步骤1后 (permute2f128):
[00 01 02 03] [40 41 42 43]
[20 21 22 23] [60 61 62 63]
[10 11 12 13] [50 51 52 53]
[30 31 32 33] [70 71 72 73]

步骤2后 (shuffle):
[00 02 10 12] [40 42 50 52]
[01 03 11 13] [41 43 51 53]
[20 22 30 32] [60 62 70 72]
[21 23 31 33] [61 63 71 73]

步骤3后 (shuffle):
[00 10 20 30] [40 50 60 70] ✓
[01 11 21 31] [41 51 61 71] ✓
[02 12 22 32] [42 52 62 72] ✓
[03 13 23 33] [43 53 63 73] ✓
```

#### 1.2.3 8x8矩阵转置

```cpp
// 输入: 8行8列 (8个256位寄存器)
// i0: [00, 01, 02, 03, 04, 05, 06, 07]
// i1: [10, 11, 12, 13, 14, 15, 16, 17]
// ...
// i7: [70, 71, 72, 73, 74, 75, 76, 77]

inline void transpose_8x8(
        const __m256 i0, const __m256 i1, const __m256 i2, const __m256 i3,
        const __m256 i4, const __m256 i5, const __m256 i6, const __m256 i7,
        __m256& o0, __m256& o1, __m256& o2, __m256& o3,
        __m256& o4, __m256& o5, __m256& o6, __m256& o7) {

    // 阶段1: unpack - 交错相邻寄存器
    const __m256 r0 = _mm256_unpacklo_ps(i0, i1); // 00 10 01 11 04 14 05 15
    const __m256 r1 = _mm256_unpackhi_ps(i0, i1); // 02 12 03 13 06 16 07 17
    const __m256 r2 = _mm256_unpacklo_ps(i2, i3); // 20 30 21 31 24 34 25 35
    const __m256 r3 = _mm256_unpackhi_ps(i2, i3); // 22 32 23 33 26 36 27 37
    const __m256 r4 = _mm256_unpacklo_ps(i4, i5); // 40 50 41 51 44 54 45 55
    const __m256 r5 = _mm256_unpackhi_ps(i4, i5); // 42 52 43 53 46 56 47 57
    const __m256 r6 = _mm256_unpacklo_ps(i6, i7); // 60 70 61 71 64 74 65 75
    const __m256 r7 = _mm256_unpackhi_ps(i6, i7); // 62 72 63 73 66 76 67 77

    // 阶段2: shuffle - 第一次细粒度重排
    const __m256 rr0 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 rr1 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 rr2 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 rr3 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 rr4 = _mm256_shuffle_ps(r4, r6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 rr5 = _mm256_shuffle_ps(r4, r6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 rr6 = _mm256_shuffle_ps(r5, r7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 rr7 = _mm256_shuffle_ps(r5, r7, _MM_SHUFFLE(3, 2, 3, 2));

    // 阶段3: permute2f128 - 重组128位块
    o0 = _mm256_permute2f128_ps(rr0, rr4, 0x20); // 00 10 20 30 40 50 60 70
    o1 = _mm256_permute2f128_ps(rr1, rr5, 0x20); // 01 11 21 31 41 51 61 71
    o2 = _mm256_permute2f128_ps(rr2, rr6, 0x20); // 02 12 22 32 42 52 62 72
    o3 = _mm256_permute2f128_ps(rr3, rr7, 0x20); // 03 13 23 33 43 53 63 73
    o4 = _mm256_permute2f128_ps(rr0, rr4, 0x31); // 04 14 24 34 44 54 64 74
    o5 = _mm256_permute2f128_ps(rr1, rr5, 0x31); // 05 15 25 35 45 55 65 75
    o6 = _mm256_permute2f128_ps(rr2, rr6, 0x31); // 06 16 26 36 46 56 66 76
    o7 = _mm256_permute2f128_ps(rr3, rr7, 0x31); // 07 17 27 37 47 57 67 77
}
```

**指令延迟分析**:

| 指令 | 延迟 (Skylake) | 吞吐量 | 端口 |
|------|---------------|--------|------|
| unpacklo/hi_ps | 1 | 0.5 | p5 |
| shuffle_ps | 1 | 1 | p5 |
| permute2f128 | 3 | 1 | p5 |

8x8转置的延迟: 约9个周期 (关键路径: unpack → shuffle → permute2f128)

### 1.3 AVX-512矩阵转置

AVX-512提供了更强大的转置能力:

```cpp
// faiss/utils/transpose/transpose-avx512-inl.h

// 16x2转置 (使用512位寄存器)
inline void transpose_16x2(
        const __m512 i0,
        const __m512 i1,
        __m512& o0,
        __m512& o1) {

    // 输入:
    // i0: [00, 01, 02, ..., 0f, 10, 11, 12, ..., 1f]
    // i1: [20, 21, 22, ..., 2f, 30, 31, 32, ..., 3f]

    // AVX-512提供了更灵活的permute指令
    // _mm512_permutex_ps: 跨512位寄存器重排
    // _mm512_shuffle_f32x4: 重排128位块

    // 步骤1: 重组256位块
    const __m512 r0 = _mm512_shuffle_f32x4(i0, i1, _MM_SHUFFLE(2, 0, 2, 0));
    const __m512 r1 = _mm512_shuffle_f32x4(i0, i1, _MM_SHUFFLE(3, 1, 3, 1));

    // 步骤2: 细粒度重排
    o0 = _mm512_permutex_ps(r0, _MM_SHUFFLE(3, 1, 2, 0));
    o1 = _mm512_permutex_ps(r1, _MM_SHUFFLE(3, 1, 2, 0));
}
```

### 1.4 实际应用:批量距离计算

```cpp
// 使用转置优化批量距离计算
void batch_distance_with_transpose(
        const float* query,    // d维查询向量
        const float* database, // nb × d 数据库 (转置存储: d × nb)
        size_t d, size_t nb,
        float* distances) {

    // 假设d是8的倍数
    for (size_t i = 0; i < d; i += 8) {
        // 加载查询的8个维度
        __m256 q = _mm256_loadu_ps(query + i);

        // 加载数据库的8行×nb列 (转置存储)
        for (size_t j = 0; j < nb; j += 8) {
            __m256 db0 = _mm256_loadu_ps(database + i * nb + j);
            __m256 db1 = _mm256_loadu_ps(database + (i + 1) * nb + j);
            __m256 db2 = _mm256_loadu_ps(database + (i + 2) * nb + j);
            __m256 db3 = _mm256_loadu_ps(database + (i + 3) * nb + j);
            __m256 db4 = _mm256_loadu_ps(database + (i + 4) * nb + j);
            __m256 db5 = _mm256_loadu_ps(database + (i + 5) * nb + j);
            __m256 db6 = _mm256_loadu_ps(database + (i + 6) * nb + j);
            __m256 db7 = _mm256_loadu_ps(database + (i + 7) * nb + j);

            // 转置为8×8矩阵
            __m256 t0, t1, t2, t3, t4, t5, t6, t7;
            transpose_8x8(db0, db1, db2, db3, db4, db5, db6, db7,
                         t0, t1, t2, t3, t4, t5, t6, t7);

            // 现在 t0-t7 是8个向量的8个维度
            // 可以高效计算距离
        }
    }
}
```

---

## 第二部分:内存布局优化

### 2.1 数据结构对齐

#### 2.1.1 缓存行对齐

```cpp
// faiss/utils/AlignedTable.h

template <class T>
struct AlignedTable {
    T* data;
    size_t n;

    explicit AlignedTable(size_t n = 0) : n(n) {
        if (n > 0) {
            // 分配32字节对齐的内存 (AVX2)
            // 或64字节对齐的内存 (AVX-512)
            data = (T*)aligned_alloc(32, n * sizeof(T));
        }
    }

    ~AlignedTable() {
        free(data);
    }

    // 移动构造
    AlignedTable(AlignedTable&& other) noexcept {
        data = other.data;
        n = other.n;
        other.data = nullptr;
        other.n = 0;
    }
};
```

**对齐的重要性**:

```cpp
// 未对齐访问 (慢)
void unaligned_access(float* ptr) {
    __m256 v = _mm256_loadu_ps(ptr);  // 可以工作,但慢
}

// 对齐访问 (快)
void aligned_access(float* ptr) {
    // 要求: ptr必须32字节对齐
    __m256 v = _mm256_load_ps(ptr);   // 快1-2个周期
}
```

#### 2.1.2 避免伪共享

```cpp
// 错误示例: 多线程共享缓存行
struct BadCounter {
    atomic<int> counter0;  // 字节0-3
    atomic<int> counter1;  // 字节4-7
    // 两个变量在同一缓存行 (64字节)
    // 多线程更新会导致false sharing
};

// 正确示例: 使用缓存行对齐
struct GoodCounter {
    alignas(64) atomic<int> counter0;  // 缓存行0
    char padding1[64 - sizeof(atomic<int>)];

    alignas(64) atomic<int> counter1;  // 缓存行1
    char padding2[64 - sizeof(atomic<int>)];
};
```

### 2.2 内存访问模式优化

#### 2.2.1 SoA vs AoS

```cpp
// AoS (Array of Structures) - 不利于SIMD
struct PointAoS {
    float x, y, z, w;
};
PointAoS points[1000];  // [x0,y0,z0,w0, x1,y1,z1,w1, ...]

// SIMD访问困难: 需要shuffle提取x,y,z,w

// SoA (Structure of Arrays) - SIMD友好
struct PointSoA {
    float x[1000];
    float y[1000];
    float z[1000];
    float w[1000];
};
// 可以直接加载x[0:7], y[0:7]进行SIMD计算
```

#### 2.2.2 分块策略

```cpp
// Faiss中的分块处理
void tiled_distance_computation(
        const float* queries,
        const float* database,
        size_t nq, size_t nb, size_t d,
        float* result) {

    const size_t QB = 8;   // 查询块大小
    const size_t DB = 8;   // 数据库块大小
    const size_t DB_dim = 8;  // 维度块大小

    for (size_t qi = 0; qi < nq; qi += QB) {
        size_t nq_in_batch = std::min(QB, nq - qi);

        for (size_t di = 0; di < d; di += DB_dim) {
            size_t d_in_batch = std::min(DB_dim, d - di);

            for (size_t bi = 0; bi < nb; bi += DB) {
                size_t nb_in_batch = std::min(DB, nb - bi);

                // 处理小块: 适合放入L1缓存
                compute_distance_tile(
                    queries + qi * d + di,
                    database + bi * d + di,
                    nq_in_batch, nb_in_batch, d_in_batch,
                    result + qi * nb + bi);
            }
        }
    }
}
```

**分块大小的选择**:

```cpp
// 根据CPU缓存大小选择分块
struct CacheInfo {
    size_t L1_size = 32 * 1024;     // 32 KB
    size_t L2_size = 256 * 1024;    // 256 KB
    size_t L3_size = 8 * 1024 * 1024; // 8 MB
    size_t cache_line_size = 64;
};

// 计算最优分块大小
size_t optimal_tile_size(const CacheInfo& cache, size_t element_size) {
    // L1缓存可以容纳的元素数
    size_t l1_elements = cache.L1_size / element_size;

    // 分块应该是sqrt(L1)左右的平方
    size_t tile = (size_t)sqrt(l1_elements);

    // 对齐到SIMD宽度
    tile = (tile / 8) * 8;

    return tile;
}
```

### 2.3 数据重排技巧

#### 2.3.1 交错存储

```cpp
// 用于批量处理的数据布局
// 传统的存储方式:
// [q0_d0, q0_d1, ..., q0_d(d-1), q1_d0, ...]

// 交错存储方式 (Interleaved):
// [q0_d0, q1_d0, ..., q7_d0, q0_d1, q1_d1, ..., q7_d1, ...]
// 优点: 一次加载可以处理8个查询的同一维度

void interleave_queries(const float* queries, size_t nq, size_t d, float* output) {
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < nq; j += 8) {
            // 加载8个查询的第i个维度
            __m256 v = _mm256_set_ps(
                queries[std::min(j + 7, nq - 1) * d + i],
                queries[std::min(j + 6, nq - 1) * d + i],
                queries[std::min(j + 5, nq - 1) * d + i],
                queries[std::min(j + 4, nq - 1) * d + i],
                queries[std::min(j + 3, nq - 1) * d + i],
                queries[std::min(j + 2, nq - 1) * d + i],
                queries[std::min(j + 1, nq - 1) * d + i],
                queries[j * d + i]
            );
            _mm256_storeu_ps(output + i * nq + j, v);
        }
    }
}
```

#### 2.3.2 位压缩存储

```cpp
// Faiss中使用位压缩减少内存占用
// 例如: 4-bit量化

void pack_4bit(const uint8_t* input, size_t n, uint8_t* output) {
    // 每2个4-bit数压缩为1个字节
    for (size_t i = 0; i < n; i += 2) {
        uint8_t high = input[i] & 0x0F;
        uint8_t low = (i + 1 < n) ? (input[i + 1] & 0x0F) : 0;
        output[i / 2] = (high << 4) | low;
    }
}

// SIMD解压
void unpack_4bit_simd(const uint8_t* input, size_t n, uint8_t* output) {
    for (size_t i = 0; i + 31 < n; i += 32) {
        // 加载16字节 = 32个4-bit数
        __m128i packed = _mm_loadu_si128((__m128i*)(input + i / 2));

        // 解压为32字节
        __m256i unpacked_low = _mm256_and_si256(
            _mm256_cvtepu8_epi16(packed),
            _mm256_set1_epi16(0x000F)
        );
        __m256i unpacked_high = _mm256_and_si256(
            _mm256_srli_epi16(_mm256_cvtepu8_epi16(packed), 4),
            _mm256_set1_epi16(0x000F)
        );

        _mm256_storeu_si256((__m256i*)(output + i), unpacked_low);
        _mm256_storeu_si256((__m256i*)(output + i + 16), unpacked_high);
    }
}
```

---

## 第三部分:寄存器分配与循环优化

### 3.1 寄存器压力管理

```cpp
// 糟糕的寄存器使用 (寄存器溢出)
void bad_register_usage(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);   // 寄存器溢出!
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        __m256 a2 = _mm256_loadu_ps(a + i + 16);
        __m256 b2 = _mm256_loadu_ps(b + i + 16);
        __m256 a3 = _mm256_loadu_ps(a + i + 24);
        __m256 b3 = _mm256_loadu_ps(b + i + 24);
        // AVX2只有16个ymm寄存器,这里使用了8个
        // 更多的操作可能导致溢出到栈
    }
}

// 优化后的寄存器使用
void good_register_usage(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i += 32) {
        // 使用循环展开,但限制同时使用的寄存器数
        for (size_t j = 0; j < 32; j += 8) {
            __m256 av = _mm256_loadu_ps(a + i + j);
            __m256 bv = _mm256_loadu_ps(b + i + j);
            __m256 cv = _mm256_mul_ps(av, bv);
            _mm256_storeu_ps(c + i + j, cv);
            // 每次只使用3个寄存器
        }
    }
}
```

### 3.2 循环展开策略

```cpp
// 完全展开 (可能导致代码膨胀)
void fully_unrolled(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i += 32) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 c0 = _mm256_fmadd_ps(a0, b0, c0);

        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        __m256 c1 = _mm256_fmadd_ps(a1, b1, c1);

        __m256 a2 = _mm256_loadu_ps(a + i + 16);
        __m256 b2 = _mm256_loadu_ps(b + i + 16);
        __m256 c2 = _mm256_fmadd_ps(a2, b2, c2);

        __m256 a3 = _mm256_loadu_ps(a + i + 24);
        __m256 b3 = _mm256_loadu_ps(b + i + 24);
        __m256 c3 = _mm256_fmadd_ps(a3, b3, c3);

        _mm256_storeu_ps(c + i, c0);
        _mm256_storeu_ps(c + i + 8, c1);
        _mm256_storeu_ps(c + i + 16, c2);
        _mm256_storeu_ps(c + i + 24, c3);
    }
}

// 使用4路展开的平衡策略
void balanced_unroll(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    const size_t unroll = 4;

    // 主循环: 4路展开
    for (; i + unroll * 8 <= n; i += unroll * 8) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        for (size_t j = 0; j < d; j += 8) {
            __m256 av = _mm256_loadu_ps(a + j);
            __m256 b0 = _mm256_loadu_ps(b + i + j);
            __m256 b1 = _mm256_loadu_ps(b + i + 8 + j);
            __m256 b2 = _mm256_loadu_ps(b + i + 16 + j);
            __m256 b3 = _mm256_loadu_ps(b + i + 24 + j);

            sum0 = _mm256_fmadd_ps(av, b0, sum0);
            sum1 = _mm256_fmadd_ps(av, b1, sum1);
            sum2 = _mm256_fmadd_ps(av, b2, sum2);
            sum3 = _mm256_fmadd_ps(av, b3, sum3);
        }

        _mm256_storeu_ps(c + i, sum0);
        _mm256_storeu_ps(c + i + 8, sum1);
        _mm256_storeu_ps(c + i + 16, sum2);
        _mm256_storeu_ps(c + i + 24, sum3);
    }

    // 处理剩余元素
    for (; i < n; i += 8) {
        __m256 sum = _mm256_setzero_ps();
        for (size_t j = 0; j < d; j += 8) {
            __m256 av = _mm256_loadu_ps(a + j);
            __m256 bv = _mm256_loadu_ps(b + i + j);
            sum = _mm256_fmadd_ps(av, bv, sum);
        }
        _mm256_storeu_ps(c + i, sum);
    }
}
```

### 3.3 软件流水线

```cpp
// 手动软件流水线优化
void pipelined_inner_product(
        const float* x,
        const float* y,
        size_t d,
        float* result) {

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    size_t i = 0;

    // 预加载阶段
    if (i + 32 <= d) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 y0 = _mm256_loadu_ps(y + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 y1 = _mm256_loadu_ps(y + i + 8);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 y2 = _mm256_loadu_ps(y + i + 16);

        i += 24;

        // 主循环: 软件流水线
        for (; i + 32 <= d; i += 8) {
            __m256 x3 = _mm256_loadu_ps(x + i);
            __m256 y3 = _mm256_loadu_ps(y + i);

            // 执行阶段: 使用之前加载的数据
            sum0 = _mm256_fmadd_ps(x0, y0, sum0);
            sum1 = _mm256_fmadd_ps(x1, y1, sum1);
            sum2 = _mm256_fmadd_ps(x2, y2, sum2);
            sum3 = _mm256_fmadd_ps(x3, y3, sum3);

            // 滚动寄存器
            x0 = x1; y0 = y1;
            x1 = x2; y1 = y2;
            x2 = x3; y2 = y3;
        }

        // 完成剩余的FMA操作
        sum0 = _mm256_fmadd_ps(x0, y0, sum0);
        sum1 = _mm256_fmadd_ps(x1, y1, sum1);
        sum2 = _mm256_fmadd_ps(x2, y2, sum2);
    }

    // 处理剩余元素
    for (; i < d; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 yv = _mm256_loadu_ps(y + i);
        sum0 = _mm256_fmadd_ps(xv, yv, sum0);
    }

    // 归约
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    *result = horizontal_sum(sum0);
}
```

---

## 第四部分:分支优化与预测

### 4.1 无分支编程

```cpp
// 传统方式: 使用分支
float min_with_branch(float a, float b) {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

// SIMD优化: 无分支min
float min_without_branch(float a, float b) {
    __m256 va = _mm256_set1_ps(a);
    __m256 vb = _mm256_set1_ps(b);
    __m256 result = _mm256_min_ps(va, vb);
    return _mm256_cvtss_f32(result);
}

// 使用位操作的无分支abs
float abs_bitwise(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;
    v.u &= 0x7FFFFFFF;  // 清除符号位
    return v.f;
}

// SIMD版本
__m256 abs_simd(__m256 x) {
    __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    return _mm256_and_ps(x, mask);
}
```

### 4.2 条件移动

```cpp
// 使用blend进行条件选择
float conditional_select_simd(
        float a, float b, float condition, float threshold) {

    __m256 va = _mm256_set1_ps(a);
    __m256 vb = _mm256_set1_ps(b);
    __m256 vcond = _mm256_set1_ps(condition);
    __m256 vthresh = _mm256_set1_ps(threshold);

    // 比较: condition > threshold
    __m256 cmp = _mm256_cmp_ps(vcond, vthresh, _CMP_GT_OQ);

    // blend: 如果cmp为真,选va;否则选vb
    __m256 result = _mm256_blendv_ps(vb, va, cmp);

    return _mm256_cvtss_f32(result);
}

// 批量条件选择
void batch_conditional_add(
        const float* x,
        const float* y,
        const float* threshold,
        float* result,
        size_t n) {

    for (size_t i = 0; i < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vt = _mm256_loadu_ps(threshold + i);

        __m256 cmp = _mm256_cmp_ps(vx, vt, _CMP_GT_OQ);

        // vx > vt ? vx + vy : vx
        __m256 sum = _mm256_add_ps(vx, vy);
        __m256 vr = _mm256_blendv_ps(vx, sum, cmp);

        _mm256_storeu_ps(result + i, vr);
    }
}
```

### 4.3 查找表优化

```cpp
// 使用查找表避免分支
// 例如: 实现ReLU激活函数

// 标准版本
float relu_scalar(float x) {
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}

// SIMD无分支版本
__m256 relu_simd(__m256 x) {
    __m256 zero = _mm256_setzero_ps();
    // max(x, 0) 等价于 ReLU
    return _mm256_max_ps(x, zero);
}

// 使用查找表的量化版本
int8_t quantized_relu_lookup(int8_t x) {
    // 预计算查找表
    static const int8_t relu_table[256] = {
        0, 1, 2, 3, ..., 127,  // 0-127保持不变
        0, 0, 0, ..., 0        // 128-255映射到0
    };
    return relu_table[(uint8_t)x];
}

// SIMD版本 (使用shuffle作为查找表)
__m256i quantized_relu_simd(__m256i x) {
    // 假设x包含16个int8
    // 使用pshufb进行查找表操作
    __m256i lut = _mm256_set_epi8(
        0, 0, 0, ..., 0,           // 128-255 -> 0
        127, 126, ..., 1, 0        // 0-127 -> 保持
    );
    return _mm256_shuffle_epi8(lut, x);
}
```

---

## 第五部分:性能分析与优化工具

### 5.1 使用perf分析

```bash
# 记录性能数据
perf record -e cycles,instructions,cache-misses ./your_program

# 查看报告
perf report

# 查看特定函数的详细信息
perf annotate -s symbol_name

# 分析缓存命中率
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses ./your_program
```

### 5.2 VTune分析

```bash
# 使用Intel VTune进行深度分析
vtune -collect hotspots ./your_program

# 分析内存访问模式
vtune -collect memory-access ./your_program

# 微架构分析
vtune -collect uarch-exploration ./your_program
```

### 5.3 编译器优化报告

```bash
# GCC/Clang优化报告
gcc -O3 -fopt-info-vec-optimized code.cpp

# 查看生成的汇编
gcc -O3 -S -masm=intel code.cpp -o code.s

# 查看是否生成了SIMD指令
objdump -d your_binary | grep -E "vaddps|vmulps|vfmadd"
```

---

## 第六部分:综合案例

### 6.1 优化前后的对比

**案例: 批量向量内积计算**

```cpp
// 优化前: 标量实现
void inner_product_naive(
        const float* x,
        const float* y,
        size_t n, size_t d,
        float* result) {
    for (size_t i = 0; i < n; i++) {
        float sum = 0;
        for (size_t j = 0; j < d; j++) {
            sum += x[i * d + j] * y[i * d + j];
        }
        result[i] = sum;
    }
}

// 优化后: 完整的SIMD优化版本
void inner_product_optimized(
        const float* x,
        const float* y,
        size_t n, size_t d,
        float* result) {

    const size_t SIMDW = 8;     // AVX2宽度
    const size_t UNROLL = 4;    // 循环展开因子

    // 对齐检查
    assert(d % SIMDW == 0);

    for (size_t i = 0; i < n; i++) {
        const float* xi = x + i * d;
        const float* yi = y + i * d;

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        size_t j = 0;

        // 主循环: 4路展开
        for (; j + UNROLL * SIMDW <= d; j += UNROLL * SIMDW) {
            __m256 xv = _mm256_loadu_ps(xi + j);

            __m256 y0 = _mm256_loadu_ps(yi + j);
            __m256 y1 = _mm256_loadu_ps(yi + j + 8);
            __m256 y2 = _mm256_loadu_ps(yi + j + 16);
            __m256 y3 = _mm256_loadu_ps(yi + j + 24);

            sum0 = _mm256_fmadd_ps(xv, y0, sum0);
            sum1 = _mm256_fmadd_ps(xv, y1, sum1);
            sum2 = _mm256_fmadd_ps(xv, y2, sum2);
            sum3 = _mm256_fmadd_ps(xv, y3, sum3);
        }

        // 处理剩余的SIMD向量
        for (; j + SIMDW <= d; j += SIMDW) {
            __m256 xv = _mm256_loadu_ps(xi + j);
            __m256 yv = _mm256_loadu_ps(yi + j);
            sum0 = _mm256_fmadd_ps(xv, yv, sum0);
        }

        // 归约
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        // 水平求和
        result[i] = horizontal_sum_avx2(sum0);

        // 处理剩余元素
        for (; j < d; j++) {
            result[i] += xi[j] * yi[j];
        }
    }
}
```

**性能对比** (Intel Xeon, d=128, n=10000):

| 实现 | 时间 (ms) | 加速比 |
|------|----------|--------|
| 标量 (gcc -O3) | 45.2 | 1.0x |
| 自动向量化 | 18.3 | 2.5x |
| SIMD优化 | 6.8 | 6.6x |
| SIMD + 展开 | 5.1 | 8.9x |

### 6.2 性能优化检查清单

```markdown
## SIMD优化检查清单

### 内存访问
- [ ] 数据对齐到SIMD宽度边界
- [ ] 使用连续内存布局
- [ ] 避免缓存行冲突
- [ ] 考虑SoA而非AoS
- [ ] 使用内存预取

### 循环结构
- [ ] 循环计数对齐到SIMD宽度
- [ ] 适当的循环展开
- [ ] 避免循环内分支
- [ ] 软件流水线

### 指令选择
- [ ] 使用FMA指令
- [ ] 避免标量指令混入SIMD循环
- [ ] 使用对齐的load/store
- [ ] 避免不必要的类型转换

### 寄存器管理
- [ ] 避免寄存器溢出
- [ ] 重用寄存器而非重新加载
- [ ] 限制同时活跃的变量数

### 编译器选项
- [ ] -march=native (或特定架构)
- [ ] -O3优化级别
- [ ] -ffast-math (如果可接受)
- [ ] 检查自动向量化报告
```

---

## 总结

本课程深入剖析了Faiss中的SIMD底层优化技术,涵盖了:

1. **矩阵转置**: 使用permute和shuffle指令实现高效转置
2. **内存布局**: 对齐、缓存优化、数据重排
3. **寄存器管理**: 避免溢出、合理展开
4. **分支优化**: 无分支编程、条件选择
5. **性能分析**: perf、VTune等工具的使用

**关键要点**:
- 内存访问模式往往比计算本身更重要
- 理解CPU微架构有助于做出正确的优化决策
- 总是测量验证优化效果
- 平衡可读性和性能

**下一步学习**:
- 《现代硬件特性专题》: AMX、SVE等新指令集
- 《高级调试与profiling》: 性能瓶颈定位
- 《实战优化案例》: 真实项目的优化过程
