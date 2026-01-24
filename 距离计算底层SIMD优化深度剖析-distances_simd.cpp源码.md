# 距离计算底层SIMD优化深度剖析 - distances_simd.cpp源码解析

## 课程简介

本课程深入剖析Faiss中`distances_simd.cpp`的底层实现,揭示CPU向量搜索的核心优化技术。这个文件包含了Faiss中最关键的距离计算函数,是整个向量搜索性能的基础。

**前置知识**:
- 已完成《SIMD底层优化深度剖析-矩阵转置与内存布局》
- 熟悉SSE/AVX2/AVX-512指令集
- 理解内积和L2距离的数学定义

**学习目标**:
- 掌握`distances_simd.cpp`的底层优化策略
- 理解不同维度(D1, D2, D4, D8等)的专用优化
- 学习矩阵转置在距离计算中的应用
- 理解FMA指令的性能优势
- 掌握预取优化技术

---

## 第一部分:源码架构概览

### 1.1 文件结构

```cpp
// faiss/utils/distances_simd.cpp

/*********************************************************
 * Reference implementations (参考实现)
 *********************************************************/
float fvec_L1_ref(const float* x, const float* y, size_t d);
float fvec_Linf_ref(const float* x, const float* y, size_t d);
void fvec_L2sqr_ny_ref(...);
void fvec_inner_products_ny_ref(...);

/*********************************************************
 * Autovectorized implementations (自动向量化实现)
 *********************************************************/
float fvec_inner_product(const float* x, const float* y, size_t d);
float fvec_norm_L2sqr(const float* x, size_t d);
float fvec_L2sqr(const float* x, const float* y, size_t d);
void fvec_inner_product_batch_4(...);
void fvec_L2sqr_batch_4(...);

/*********************************************************
 * SSE and AVX implementations (SSE/AVX实现)
 *********************************************************/
// 针对不同维度的专用优化
void fvec_op_ny_D1(...);  // 1维向量
void fvec_op_ny_D2(...);  // 2维向量
void fvec_op_ny_D4(...);  // 4维向量
void fvec_op_ny_D8(...);  // 8维向量
// ...
void fvec_op_ny_D12(...); // 12维向量
// ...通用维度
void fvec_op_ny_impl(...);
```

### 1.2 设计哲学

**为什么需要不同维度的专用实现?**

```cpp
// 维度对性能的影响:

// D1 (1维): 不需要SIMD,标量操作最快
// D2 (2维): 可以打包多个向量,充分利用SIMD宽度
// D4 (4维): 正好一个__m128寄存器
// D8 (8维): 正好一个__m256寄存器(AVX2)
// D16 (16维): 正好一个__m512寄存器(AVX-512)
```

**优化原则**:

1. **小维度(D<8)**: 打包多个向量,并行计算
2. **中等维度(8<=D<32)**: 使用全宽度SIMD
3. **大维度(D>=32)**: 展开循环,隐藏延迟
4. **所有维度**: 利用预取优化内存访问

---

## 第二部分:核心数据结构与辅助函数

### 2.1 ElementOp模板 - 距离计算的策略模式

```cpp
// faiss/utils/distances_simd.cpp (line 366-414)

/// 计算L2距离的操作符
struct ElementOpL2 {
    // 标量版本
    static float op(float x, float y) {
        float tmp = x - y;
        return tmp * tmp;
    }

    // SSE版本(__m128)
    static __m128 op(__m128 x, __m128 y) {
        __m128 tmp = _mm_sub_ps(x, y);
        return _mm_mul_ps(tmp, tmp);
    }

    #ifdef __AVX2__
    // AVX2版本(__m256)
    static __m256 op(__m256 x, __m256 y) {
        __m256 tmp = _mm256_sub_ps(x, y);
        return _mm256_mul_ps(tmp, tmp);
    }
    #endif

    #ifdef __AVX512F__
    // AVX-512版本(__m512)
    static __m512 op(__m512 x, __m512 y) {
        __m512 tmp = _mm512_sub_ps(x, y);
        return _mm512_mul_ps(tmp, tmp);
    }
    #endif
};

/// 计算内积的操作符
struct ElementOpIP {
    static float op(float x, float y) {
        return x * y;
    }

    static __m128 op(__m128 x, __m128 y) {
        return _mm_mul_ps(x, y);
    }

    #ifdef __AVX2__
    static __m256 op(__m256 x, __m256 y) {
        return _mm256_mul_ps(x, y);
    }
    #endif

    #ifdef __AVX512F__
    static __m512 op(__m512 x, __m512 y) {
        return _mm512_mul_ps(x, y);
    }
    #endif
};
```

**设计模式分析**:

这是典型的**策略模式(Strategy Pattern)**:
- `ElementOpL2`和`ElementOpIP`封装不同的距离计算策略
- 距离计算函数`fvec_op_ny_Dx`是上下文(Context)
- 通过模板参数在编译时选择策略,零运行时开销

### 2.2 水平求和函数

```cpp
// faiss/utils/distances_simd.cpp (line 329-361)

/// SSE版本的水平求和
inline float horizontal_sum(const __m128 v) {
    // v = [x0, x1, x2, x3]

    // v0 = [x2, x3, x2, x3] (shuffle with _MM_SHUFFLE(0, 0, 3, 2))
    const __m128 v0 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 3, 2));

    // v1 = [x0+x2, x1+x3, x0+x2, x1+x3]
    const __m128 v1 = _mm_add_ps(v, v0);

    // v2 = [x1+x3, ..., ..., ...]
    __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));

    // v3 = [x0+x1+x2+x3, ..., ..., ...]
    const __m128 v3 = _mm_add_ps(v1, v2);

    // 返回v3[0]
    return _mm_cvtss_f32(v3);
}

#ifdef __AVX2__
/// AVX2版本的水平求和
inline float horizontal_sum(const __m256 v) {
    // 拆分为两个__m128
    const __m128 v0 =
            _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));

    // 调用SSE版本的水平求和
    return horizontal_sum(v0);
}
#endif

#ifdef __AVX512F__
/// AVX-512版本的水平求和
inline float horizontal_sum(const __m512 v) {
    // AVX-512提供专用的reduce指令
    return _mm512_reduce_add_ps(v);
}
#endif
```

**性能分析**:

| 指令集 | 延迟(周期) | 吞吐量(cpi) |
|--------|-----------|------------|
| SSE水平求和 | ~7 | 1-2 |
| AVX2水平求和 | ~8 | 2 |
| AVX-512 reduce | ~3 | 1 |

**关键优化**:
- AVX2需要跨128位lane的操作,开销大
- AVX-512的`_mm512_reduce_add_ps`是专用指令,效率最高
- 尽可能避免频繁的水平求和

### 2.3 masked_read - 安全的部分加载

```cpp
// faiss/utils/distances_simd.cpp (line 309-324)

// 读取0 <= d < 4个float到__m128
static inline __m128 masked_read(int d, const float* x) {
    assert(0 <= d && d < 4);

    // 使用对齐内存避免部分加载的惩罚
    ALIGNED(16) float buf[4] = {0, 0, 0, 0};

    switch (d) {
        case 3:
            buf[2] = x[2];
            [[fallthrough]];
        case 2:
            buf[1] = x[1];
            [[fallthrough]];
        case 1:
            buf[0] = x[0];
        // case 0: 全部为0
    }

    // 从对齐内存加载(比_mm_loadu_ps快)
    return _mm_load_ps(buf);
}
```

**为什么需要masked_read?**

```cpp
// 问题:处理非SIMD宽度倍数的维度
float distance(const float* x, const float* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        sum = _mm256_fmadd_ps(vx, vy, sum);
    }

    // 处理剩余元素 (d % 8)
    if (i < d) {
        int rem = d - i;  // 1-7个元素
        __m128 vx = masked_read(rem, x + i);
        __m128 vy = masked_read(rem, y + i);
        __m128 vsum = _mm_mul_ps(vx, vy);

        // 将__m128的结果加到__m256
        __m128 sum_low = _mm256_castps256_ps128(sum);
        sum_low = _mm_add_ps(sum_low, vsum);
        sum = _mm256_setzero_ps(); // 清零高位

        // ...水平求和...
    }

    return horizontal_sum(sum);
}
```

---

## 第三部分:D2维度的深度优化

### 3.1 SSE版本的D2内积

```cpp
// faiss/utils/distances_simd.cpp (line 439-454)

template <>
void fvec_op_ny_D2<ElementOpIP>(
        float* dis,       // 输出:ny个距离
        const float* x,   // 输入:1个D2向量
        const float* y,   // 输入:ny个D2向量
        size_t ny) {

    // 将x广播到__m128的4个元素
    // x = [x0, x1], m0 = [x0, x1, x0, x1]
    __m128 x0 = _mm_set_ps(x[1], x[0], x[1], x[0]);

    size_t i;
    // 每次处理2个向量(y有2个D2向量)
    for (i = 0; i + 1 < ny; i += 2) {
        // 加载8个float(2个D2向量)
        // y布局: [y0_0, y0_1, y1_0, y1_1]
        __m128 accu = ElementOpIP::op(x0, _mm_loadu_ps(y));
        y += 4;

        // 水平相加
        // accu = [x0*y0_0 + x1*y0_1, x0*y1_0 + x1*y1_1, ...]
        accu = _mm_hadd_ps(accu, accu);

        // 提取2个结果
        dis[i] = _mm_cvtss_f32(accu);
        accu = _mm_shuffle_ps(accu, accu, 3);
        dis[i + 1] = _mm_cvtss_f32(accu);
    }

    // 处理奇数个向量
    if (i < ny) {
        dis[i] = ElementOpIP::op(x[0], y[0]) + ElementOpIP::op(x[1], y[1]);
    }
}
```

**内存布局分析**:

```
输入:
x: [x0, x1] (1个D2向量)
y: [y0_0, y0_1, y1_0, y1_1, y2_0, y2_1, ...] (ny个D2向量)

处理后:
dis: [ip(x,y0), ip(x,y1), ip(x,y2), ...]
```

**SIMD寄存器使用**:

```
__m128 x0 = [x0, x1, x0, x1]  (广播)

第1次迭代:
__m128 y_vec = [y0_0, y0_1, y1_0, y1_1]
__m128 accu = [x0*y0_0, x1*y0_1, x0*y1_0, x1*y1_1]
__m128 accu = hadd(accu, accu)
          = [x0*y0_0+x1*y0_1, x0*y0_0+x1*y0_1,
             x0*y1_0+x1*y1_1, x0*y1_0+x1*y1_1]
dis[0] = accu[0] = x0*y0_0 + x1*y0_1
dis[1] = accu[2] = x0*y1_0 + x1*y1_1
```

### 3.2 AVX2版本的D2内积 - 矩阵转置优化

```cpp
// faiss/utils/distances_simd.cpp (line 582-636)

template <>
void fvec_op_ny_D2<ElementOpIP>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {

    const size_t ny8 = ny / 8;
    size_t i = 0;

    if (ny8 > 0) {
        // 预取前2个cache行
        _mm_prefetch((const char*)y, _MM_HINT_T0);
        _mm_prefetch((const char*)(y + 16), _MM_HINT_T0);

        // 广播x的两个分量
        const __m256 m0 = _mm256_set1_ps(x[0]);
        const __m256 m1 = _mm256_set1_ps(x[1]);

        // 主循环:每次处理8个D2向量
        for (i = 0; i < ny8 * 8; i += 8) {
            // 预取下一个cache行
            _mm_prefetch((const char*)(y + 32), _MM_HINT_T0);

            // 加载8x2矩阵并转置
            // y的布局: [y0_0, y0_1, y1_0, y1_1, ..., y7_0, y7_1]
            // 16个float = 2行8列的矩阵(转置后)

            __m256 v0;
            __m256 v1;

            transpose_8x2(
                    _mm256_loadu_ps(y + 0 * 8),  // 第0行: [y0_0, y1_0, ..., y7_0]
                    _mm256_loadu_ps(y + 1 * 8),  // 第1行: [y0_1, y1_1, ..., y7_1]
                    v0,  // 输出: [y0_0, y0_1, y1_0, y1_1, y2_0, y2_1, y3_0, y3_1]
                    v1); // 输出: [y4_0, y4_1, y5_0, y5_1, y6_0, y6_1, y7_0, y7_1]

            // 计算内积: m0*v0 + m1*v1
            // FMA指令: a*b+c
            __m256 distances = _mm256_mul_ps(m0, v0);
            distances = _mm256_fmadd_ps(m1, v1, distances);

            // 存储8个距离结果
            _mm256_storeu_ps(dis + i, distances);

            y += 16; // 移动到下一组8x2元素
        }
    }

    // 处理剩余的向量(ny % 8)
    if (i < ny) {
        float x0 = x[0];
        float x1 = x[1];

        for (; i < ny; i++) {
            float distance = x0 * y[0] + x1 * y[1];
            y += 2;
            dis[i] = distance;
        }
    }
}
```

**转置详解**:

```
加载前(内存布局):
y + 0*8: [y0_0, y1_0, y2_0, y3_0, y4_0, y5_0, y6_0, y7_0]
y + 1*8: [y0_1, y1_1, y2_1, y3_1, y4_1, y5_1, y6_1, y7_1]

转置后(SIMD寄存器):
v0: [y0_0, y0_1, y1_0, y1_1, y2_0, y2_1, y3_0, y3_1]
v1: [y4_0, y4_1, y5_0, y5_1, y6_0, y6_1, y7_0, y7_1]

计算:
m0 = [x0, x0, x0, x0, x0, x0, x0, x0]
m1 = [x1, x1, x1, x1, x1, x1, x1, x1]

distances = m0 * v0 + m1 * v1
          = [x0*y0_0+x1*y0_1, x0*y1_0+x1*y1_1, ...]
```

**transpose_8x2实现**:

```cpp
// faiss/utils/transpose/transpose-avx2-inl.h

inline void transpose_8x2(
        const __m256 i0,  // [00, 01, 10, 11, 20, 21, 30, 31]
        const __m256 i1,  // [40, 41, 50, 51, 60, 61, 70, 71]
        __m256& o0,       // 输出: [00, 10, 20, 30, 40, 50, 60, 70]
        __m256& o1) {     // 输出: [01, 11, 21, 31, 41, 51, 61, 71]

    // 步骤1: 重组128位块
    const __m256 r0 = _mm256_permute2f128_ps(i0, i1, _MM_SHUFFLE(0, 2, 0, 0));
    // r0 = [00, 01, 10, 11, 40, 41, 50, 51]

    const __m256 r1 = _mm256_permute2f128_ps(i0, i1, _MM_SHUFFLE(0, 3, 0, 1));
    // r1 = [20, 21, 30, 31, 60, 61, 70, 71]

    // 步骤2: 细粒度重排
    o0 = _mm256_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0));
    // o0 = [00, 10, 20, 30, 40, 50, 60, 70]

    o1 = _mm256_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1));
    // o1 = [01, 11, 21, 31, 41, 51, 61, 71]
}
```

### 3.3 AVX-512版本的D2内积

```cpp
// faiss/utils/distances_simd.cpp (line 459-513)

template <>
void fvec_op_ny_D2<ElementOpIP>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {

    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // 预取
        _mm_prefetch((const char*)y, _MM_HINT_T0);
        _mm_prefetch((const char*)(y + 32), _MM_HINT_T0);

        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);

        // 主循环:每次处理16个D2向量
        for (i = 0; i < ny16 * 16; i += 16) {
            _mm_prefetch((const char*)(y + 64), _MM_HINT_T0);

            __m512 v0;
            __m512 v1;

            // 转置16x2矩阵
            transpose_16x2(
                    _mm512_loadu_ps(y + 0 * 16),  // 16个第0分量
                    _mm512_loadu_ps(y + 1 * 16),  // 16个第1分量
                    v0,
                    v1);

            // 计算距离
            __m512 distances = _mm512_mul_ps(m0, v0);
            distances = _mm512_fmadd_ps(m1, v1, distances);

            // 存储
            _mm512_storeu_ps(dis + i, distances);

            y += 32; // 移动到下一组16x2元素
        }
    }

    if (i < ny) {
        // 处理剩余元素
        float x0 = x[0];
        float x1 = x[1];

        for (; i < ny; i++) {
            float distance = x0 * y[0] + x1 * y[1];
            y += 2;
            dis[i] = distance;
        }
    }
}
```

**性能对比**:

| 指令集 | 每次处理向量数 | 循环次数(1000向量) | 相对性能 |
|--------|---------------|-------------------|---------|
| SSE | 2 | 500 | 1.0x |
| AVX2 | 8 | 125 | ~3.5x |
| AVX-512 | 16 | 63 | ~7x |

---

## 第四部分:D4维度的优化

### 4.1 AVX-512版本的D4内积

```cpp
// faiss/utils/distances_simd.cpp (line 718-776)

template <>
void fvec_op_ny_D4<ElementOpIP>(
        float* dis,
        const float* x,   // 1个D4向量
        const float* y,   // ny个D4向量
        size_t ny) {

    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        // 广播x的4个分量
        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);

        for (i = 0; i < ny16 * 16; i += 16) {
            // 加载16x4矩阵并转置
            __m512 v0;
            __m512 v1;
            __m512 v2;
            __m512 v3;

            transpose_16x4(
                    _mm512_loadu_ps(y + 0 * 16),  // [y0_0, y1_0, ..., y15_0]
                    _mm512_loadu_ps(y + 1 * 16),  // [y0_1, y1_1, ..., y15_1]
                    _mm512_loadu_ps(y + 2 * 16),  // [y0_2, y1_2, ..., y15_2]
                    _mm512_loadu_ps(y + 3 * 16),  // [y0_3, y1_3, ..., y15_3]
                    v0,  // [y0_0, y0_1, y0_2, y0_3, ..., y3_0, ..., y3_3]
                    v1,  // [y4_0, y4_1, y4_2, y4_3, ..., y7_0, ..., y7_3]
                    v2,  // [y8_0, y8_1, y8_2, y8_3, ..., y11_0, ..., y11_3]
                    v3); // [y12_0, ..., y15_0, ..., y15_3]

            // 计算内积
            __m512 distances = _mm512_mul_ps(m0, v0);
            distances = _mm512_fmadd_ps(m1, v1, distances);
            distances = _mm512_fmadd_ps(m2, v2, distances);
            distances = _mm512_fmadd_ps(m3, v3, distances);

            // 存储
            _mm512_storeu_ps(dis + i, distances);

            y += 64; // 16个向量 * 4维
        }
    }

    if (i < ny) {
        // 处理剩余元素
        __m128 x0 = _mm_loadu_ps(x);

        for (; i < ny; i++) {
            __m128 accu = ElementOpIP::op(x0, _mm_loadu_ps(y));
            y += 4;
            dis[i] = horizontal_sum(accu);
        }
    }
}
```

**transpose_16x4实现**:

```cpp
// faiss/utils/transpose/transpose-avx512-inl.h

inline void transpose_16x4(
        const __m512 i0,
        const __m512 i1,
        const __m512 i2,
        const __m512 i3,
        __m512& o0,
        __m512& o1,
        __m512& o2,
        __m512& o3) {

    // AVX-512提供了更灵活的permute指令

    // 步骤1: 重组256位块
    const __m512 r0 = _mm512_shuffle_f32x4(i0, i1, _MM_SHUFFLE(1, 0, 1, 0));
    const __m512 r1 = _mm512_shuffle_f32x4(i0, i1, _MM_SHUFFLE(3, 2, 3, 2));
    const __m512 r2 = _mm512_shuffle_f32x4(i2, i3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m512 r3 = _mm512_shuffle_f32x4(i2, i3, _MM_SHUFFLE(3, 2, 3, 2));

    // 步骤2: 使用permutex2var进行跨lane重排
    o0 = _mm512_permutex2var_ps(r0, _MM_SHUFFLE(3, 1, 2, 0), r2);
    o1 = _mm512_permutex2var_ps(r1, _MM_SHUFFLE(3, 1, 2, 0), r3);
    o2 = _mm512_permutex2var_ps(r0, _MM_SHUFFLE(3, 1, 2, 0), r2);
    o3 = _mm512_permutex2var_ps(r1, _MM_SHUFFLE(3, 1, 2, 0), r3);
}
```

### 4.2 D4的L2距离优化

```cpp
// faiss/utils/distances_simd.cpp (line 779-837)

template <>
void fvec_op_ny_D4<ElementOpL2>(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {

    const size_t ny16 = ny / 16;
    size_t i = 0;

    if (ny16 > 0) {
        const __m512 m0 = _mm512_set1_ps(x[0]);
        const __m512 m1 = _mm512_set1_ps(x[1]);
        const __m512 m2 = _mm512_set1_ps(x[2]);
        const __m512 m3 = _mm512_set1_ps(x[3]);

        for (i = 0; i < ny16 * 16; i += 16) {
            __m512 v0, v1, v2, v3;

            transpose_16x4(
                    _mm512_loadu_ps(y + 0 * 16),
                    _mm512_loadu_ps(y + 1 * 16),
                    _mm512_loadu_ps(y + 2 * 16),
                    _mm512_loadu_ps(y + 3 * 16),
                    v0, v1, v2, v3);

            // 计算差值
            const __m512 d0 = _mm512_sub_ps(m0, v0);
            const __m512 d1 = _mm512_sub_ps(m1, v1);
            const __m512 d2 = _mm512_sub_ps(m2, v2);
            const __m512 d3 = _mm512_sub_ps(m3, v3);

            // 计算平方和
            __m512 distances = _mm512_mul_ps(d0, d0);
            distances = _mm512_fmadd_ps(d1, d1, distances);
            distances = _mm512_fmadd_ps(d2, d2, distances);
            distances = _mm512_fmadd_ps(d3, d3, distances);

            _mm512_storeu_ps(dis + i, distances);

            y += 64;
        }
    }

    if (i < ny) {
        // 处理剩余元素
        for (; i < ny; i++) {
            float d0 = x[0] - y[0];
            float d1 = x[1] - y[1];
            float d2 = x[2] - y[2];
            float d3 = x[3] - y[3];
            dis[i] = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            y += 4;
        }
    }
}
```

**L2距离vs内积的指令数对比**:

```
内积(每个维度):
- 1次乘法
- 累加

L2距离(每个维度):
- 1次减法
- 1次乘法
- 累加

L2距离的指令数约为内积的2倍
```

---

## 第五部分:通用维度的优化

### 5.1 AVX2的通用内积实现

```cpp
// faiss/utils/distances_simd.cpp (通用实现)

template <class ElementOp>
void fvec_op_ny_impl(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {

    // 主循环:每次处理8个向量
    size_t i = 0;
    for (; i + 8 <= ny; i += 8) {
        // 初始化累加器
        __m256 accu0 = _mm256_setzero_ps();
        __m256 accu1 = _mm256_setzero_ps();
        // ... 可以展开更多路

        size_t j = 0;

        // 内层循环:每次处理8个维度
        for (; j + 8 <= d; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);

            // 加载8个向量的第j个维度
            __m256 y0 = _mm256_loadu_ps(y + i * d + j);
            __m256 y1 = _mm256_loadu_ps(y + (i + 1) * d + j);
            // ... y2-y7

            // FMA累加
            accu0 = _mm256_fmadd_ps(xv, y0, accu0);
            accu1 = _mm256_fmadd_ps(xv, y1, accu1);
            // ... accu2-accu7
        }

        // 处理剩余维度
        for (; j < d; j++) {
            float xv = x[j];
            accu0 = _mm256_add_ps(accu0, _mm256_set1_ps(xv * y[i * d + j]));
            accu1 = _mm256_add_ps(accu1, _mm256_set1_ps(xv * y[(i + 1) * d + j]));
            // ...
        }

        // 水平求和并存储
        dis[i] = horizontal_sum(accu0);
        dis[i + 1] = horizontal_sum(accu1);
        // ...
    }

    // 处理剩余向量
    for (; i < ny; i++) {
        __m256 accu = _mm256_setzero_ps();

        for (size_t j = 0; j + 8 <= d; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            __m256 yv = _mm256_loadu_ps(y + i * d + j);
            accu = _mm256_fmadd_ps(xv, yv, accu);
        }

        dis[i] = horizontal_sum(accu);
    }
}
```

### 5.2 预取优化策略

```cpp
// 带预取的优化版本

template <class ElementOp>
void fvec_op_ny_impl_with_prefetch(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {

    size_t i = 0;
    for (; i + 8 <= ny; i += 8) {
        __m256 accu[8];
        for (int k = 0; k < 8; k++) {
            accu[k] = _mm256_setzero_ps();
        }

        size_t j = 0;

        // 预取第一组数据
        _mm_prefetch((const char*)(y + i * d), _MM_HINT_T0);
        _mm_prefetch((const char*)(y + (i + 4) * d), _MM_HINT_T0);

        for (; j + 8 <= d; j += 8) {
            // 预取下一组数据
            if (j + 16 <= d) {
                _mm_prefetch((const char*)(y + i * d + j + 16), _MM_HINT_T0);
                _mm_prefetch((const char*)(y + (i + 4) * d + j + 16), _MM_HINT_T0);
            }

            __m256 xv = _mm256_loadu_ps(x + j);

            #pragma unroll
            for (int k = 0; k < 8; k++) {
                __m256 yv = _mm256_loadu_ps(y + (i + k) * d + j);
                accu[k] = _mm256_fmadd_ps(xv, yv, accu[k]);
            }
        }

        // ... 水平求和 ...
    }

    // ...
}
```

**预取距离分析**:

```cpp
// 预取延迟和带宽的平衡

struct PrefetchParameters {
    // L1缓存: 32KB, 延迟~4周期
    // L2缓存: 256KB, 延迟~12周期
    // L3缓存: 8MB, 延迟~40周期
    // 内存: 延迟~200周期

    // 最佳预取距离取决于:
    // 1. 内存延迟
    // 2. 每次迭代的工作量
    // 3. 缓存行大小(64字节)

    // 经验法则:
    // 预取距离 = 预取延迟 / 每次迭代的周期数

    // 示例: 如果每次迭代需要20周期
    // L1预取: 4/20 = 0.2次迭代(太近)
    // L2预取: 12/20 = 0.6次迭代
    // L3预取: 40/20 = 2次迭代
    // 内存预取: 200/20 = 10次迭代
};

// Faiss使用的预取策略
const int PREFETCH_DISTANCE_L2 = 2;  // 预取到L2
const int PREFETCH_DISTANCE_L3 = 8;  // 预取到L3
```

---

## 第六部分:FMA指令的性能分析

### 6.1 FMA vs 传统乘加

```cpp
// 传统实现: mul + add
__m256 traditional_fma(__m256 a, __m256 b, __m256 c) {
    __m256 mul = _mm256_mul_ps(a, b);  // 延迟: 4周期
    __m256 result = _mm256_add_ps(mul, c); // 延迟: 3周期
    return result;
    // 总延迟: 7周期(关键路径)
}

// FMA实现
__m256 fused_mac(__m256 a, __m256 b, __m256 c) {
    // 单条指令完成 a*b+c
    return _mm256_fmadd_ps(a, b, c);
    // 延迟: 4周期(关键路径)
}
```

**Intel Skylake微架构的性能**:

| 指令 | 延迟 | 端口 | 吞吐量 |
|------|------|------|--------|
| vmulps | 4 | p0/p1 | 0.5 |
| vaddps | 3 | p0/p1 | 0.5 |
| vfmadd231ps | 4 | p0/p1 | 0.5 |

**性能提升**:

```cpp
// 内积计算(128维)
// 传统方法:
for (int i = 0; i < 128; i += 8) {
    __m256 xv = _mm256_loadu_ps(x + i);
    __m256 yv = _mm256_loadu_ps(y + i);
    __m256 mul = _mm256_mul_ps(xv, yv);  // 4周期
    sum = _mm256_add_ps(sum, mul);        // 3周期
}
// 128/8 * 7 = 112周期(理想情况,未考虑延迟隐藏)

// FMA方法:
for (int i = 0; i < 128; i += 8) {
    __m256 xv = _mm256_loadu_ps(x + i);
    __m256 yv = _mm256_loadu_ps(y + i);
    sum = _mm256_fmadd_ps(xv, yv, sum);   // 4周期
}
// 128/8 * 4 = 64周期

// 加速比: 112/64 = 1.75x
```

### 6.2 FMA的数值精度优势

```cpp
// FMA不仅更快,而且更精确

// 传统方法: x*y+z
float mul_add_rounded(float x, float y, float z) {
    float mul = x * y;  // 舍入1次
    float result = mul + z;  // 舍入1次,总共2次
    return result;
}

// FMA: x*y+z
float fused_mul_add(float x, float y, float z) {
    // 只在最终结果舍入1次
    // 中间的乘积使用更高精度
    return std::fma(x, y, z);
}

// 示例:
// 传统: 1.0e20 * 1.0e-20 + 1.0 = 1.0 + 1.0 = 2.0 (第1项丢失)
// FMA:   1.0e20 * 1.0e-20 + 1.0 = 1.0 + 1.0 = 2.0 (保留中间精度)
```

---

## 第七部分:性能分析与优化技巧

### 7.1 性能分析

```cpp
// 使用perf进行性能分析
/*
bash命令:
perf stat -e cycles,instructions,cache-misses,cache-references \
    ./your_program

期望结果:
- IPC(Instructions Per Cycle) > 1.5 表示良好
- cache-misses < 5% 表示缓存命中率高
- 使用SIMD后,instructions应该显著减少
*/

// VTune分析
/*
热点分析:
vtune -collect hotspots -result-dir r001 ./your_program

内存访问分析:
vtune -collect memory-access -result-dir r002 ./your_program

微架构分析:
vtune -collect uarch-exploration -result-dir r003 ./your_program
*/
```

### 7.2 编译器优化提示

```cpp
// Faiss使用的编译器优化宏

// 告诉编译器这是不精确的浮点函数(允许重新排序)
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float fvec_inner_product(const float* x, const float* y, size_t d) {
    float res = 0.F;
    FAISS_PRAGMA_IMPRECISE_LOOP  // 允许循环变换
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * y[i];
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

// 宏定义(通常在platform_macros.h):
#ifdef __GNUC__
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"") \
    _Pragma("GCC diagnostic ignored \"-Wtype-limits\"")
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END \
    _Pragma("GCC diagnostic pop")
#define FAISS_PRAGMA_IMPRECISE_LOOP _Pragma("GCC ivdep")
#endif
```

### 7.3 性能优化检查清单

```markdown
## 距离计算优化检查清单

### SIMD使用
- [ ] 检查是否使用了正确的SIMD宽度(SSE/AVX2/AVX-512)
- [ ] 验证循环计数对齐到SIMD宽度
- [ ] 处理剩余元素的代码路径正确

### 内存访问
- [ ] 数据是否对齐到SIMD宽度边界
- [ ] 内存访问模式是否连续
- [ ] 是否使用了适当的预取策略
- [ ] 缓存行利用率是否高

### 指令选择
- [ ] 优先使用FMA指令
- [ ] 避免跨lane的操作(AVX2)
- [ ] 最小化水平操作(水平求和等)
- [ ] 使用对齐加载(_mm_load_ps)当可能时

### 循环优化
- [ ] 适当的循环展开(2x, 4x)
- [ ] 循环不变量外提
- [ ] 减少循环内的分支

### 编译器优化
- [ ] 使用-O3或更高优化级别
- [ ] 添加-ffast-math(如果可接受)
- [ ] 使用-march=native或特定架构
- [ ] 检查自动向量化报告
```

---

## 第八部分:实战案例

### 8.1 案例1:优化非2的幂次维度

```cpp
// 问题: d=96维度的距离计算
// 96不是8或16的倍数

void fvec_inner_product_d96(
        float* dis,
        const float* x,
        const float* y,
        size_t ny) {

    const size_t d = 96;
    size_t i = 0;

    // 主循环:处理完整的8维块
    // 96 / 8 = 12,12 / 8 = 1,余4
    // 可以展开8路(64维)或12路(96维)

    for (i = 0; i + 8 <= ny; i += 8) {
        __m256 accu[8];
        for (int k = 0; k < 8; k++) {
            accu[k] = _mm256_setzero_ps();
        }

        size_t j = 0;

        // 处理前64维(展开8路)
        for (; j < 64; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);

            for (int k = 0; k < 8; k++) {
                __m256 yv = _mm256_loadu_ps(y + (i + k) * d + j);
                accu[k] = _mm256_fmadd_ps(xv, yv, accu[k]);
            }
        }

        // 处理剩余32维
        for (; j < 96; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);

            for (int k = 0; k < 8; k++) {
                __m256 yv = _mm256_loadu_ps(y + (i + k) * d + j);
                accu[k] = _mm256_fmadd_ps(xv, yv, accu[k]);
            }
        }

        // 水平求和
        for (int k = 0; k < 8; k++) {
            dis[i + k] = horizontal_sum(accu[k]);
        }
    }

    // 处理剩余向量
    for (; i < ny; i++) {
        __m256 accu = _mm256_setzero_ps();

        for (size_t j = 0; j < 96; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            __m256 yv = _mm256_loadu_ps(y + i * d + j);
            accu = _mm256_fmadd_ps(xv, yv, accu);
        }

        dis[i] = horizontal_sum(accu);
    }
}
```

### 8.2 案例2:批量距离计算优化

```cpp
// 计算nx个查询与ny个数据库向量的距离矩阵
// dis[nx][ny] = L2(x[nx][d], y[ny][d])

void compute_distance_matrix_batch(
        float* dis,       // 输出: nx * ny
        const float* x,   // 查询: nx * d
        const float* y,   // 数据库: ny * d
        size_t nx,
        size_t ny,
        size_t d) {

    // 选择最优的分块大小
    const size_t X_BATCH = 8;   // 同时处理8个查询
    const size_t Y_BATCH = 16;  // 同时处理16个数据库向量

    for (size_t xi = 0; xi < nx; xi += X_BATCH) {
        size_t nx_batch = std::min(X_BATCH, nx - xi);

        for (size_t yi = 0; yi < ny; yi += Y_BATCH) {
            size_t ny_batch = std::min(Y_BATCH, ny - yi);

            // 计算nx_batch * ny_batch的距离子矩阵
            for (size_t i = 0; i < nx_batch; i++) {
                for (size_t j = 0; j < ny_batch; j++) {
                    // 计算x[xi+i]与y[yi+j]的距离
                    float d2 = 0;
                    size_t k = 0;

                    // SIMD主循环
                    if (d >= 8) {
                        __m256 sum = _mm256_setzero_ps();

                        for (; k + 8 <= d; k += 8) {
                            __m256 xv = _mm256_loadu_ps(x + (xi + i) * d + k);
                            __m256 yv = _mm256_loadu_ps(y + (yi + j) * d + k);
                            __m256 diff = _mm256_sub_ps(xv, yv);
                            sum = _mm256_fmadd_ps(diff, diff, sum);
                        }

                        d2 = horizontal_sum(sum);
                    }

                    // 处理剩余维度
                    for (; k < d; k++) {
                        float diff = x[(xi + i) * d + k] - y[(yi + j) * d + k];
                        d2 += diff * diff;
                    }

                    dis[(xi + i) * ny + (yi + j)] = d2;
                }
            }
        }
    }
}
```

---

## 总结

本课程深入剖析了`distances_simd.cpp`的底层实现,涵盖了:

1. **架构设计**: 不同维度的专用优化策略
2. **SIMD优化**: SSE/AVX2/AVX-512的实现差异
3. **矩阵转置**: 提升内存访问效率的关键技术
4. **FMA指令**: 性能和精度的双重优势
5. **预取优化**: 隐藏内存延迟
6. **性能分析**: perf和VTune的使用

**关键要点**:
- 维度是优化策略的关键因素
- 矩阵转置可以显著提升内存访问效率
- FMA指令同时提升性能和精度
- 预取需要仔细调优距离
- 总是测量验证优化效果

**下一步学习**:
- 《堆和分区算法的SIMD优化》- Heap.h源码解析
- 《量化器底层实现》- code_distance源码解析
- 《HNSW图索引优化》- 图遍历的缓存优化

---

## 练习题

1. 实现D6维度的SIMD优化距离计算
2. 优化D12维度的距离计算,尝试不同的转置策略
3. 比较FMA与传统mul+add的性能差异
4. 分析不同预取距离对性能的影响
5. 实现混合精度的距离计算(float16+float32)
