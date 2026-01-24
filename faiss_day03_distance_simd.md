# Faiss深度课程 - 第3天：距离计算 - 底层SIMD实现

## 课程目标

深入理解Faiss中距离计算的底层实现，掌握SIMD（单指令多数据）优化技术，了解不同硬件平台的优化策略。

---

## 1. SIMD基础概念

### 1.1 什么是SIMD

SIMD（Single Instruction, Multiple Data）是一种并行计算技术，一条指令可以同时处理多个数据元素。

**基本思想**：
```cpp
// 标量计算（Scalar）
for (int i = 0; i < 8; i++) {
    c[i] = a[i] + b[i];  // 8次加法指令
}

// SIMD计算
__m256 va = _mm256_loadu_ps(a);  // 一次加载8个float
__m256 vb = _mm256_loadu_ps(b);  // 一次加载8个float
__m256 vc = _mm256_add_ps(va, vb); // 一次加8个float
_mm256_storeu_ps(c, vc);          // 一次存储8个float
```

### 1.2 SIMD指令集对比

| 指令集 | 寄存器宽度 | float数 | int32数 | 平台 |
|--------|-----------|---------|---------|------|
| SSE | 128-bit | 4 | 4 | x86/x64 |
| AVX | 256-bit | 8 | 8 | x86/x64 (Intel Sandy Bridge+) |
| AVX2 | 256-bit | 8 | 8 | x86/x64 (Intel Haswell+) |
| AVX-512 | 512-bit | 16 | 16 | x86/x64 (Intel Skylake-X+) |
| NEON | 128-bit | 4 | 4 | ARM64 |
| SVE | 可变 | 可变 | 可变 | ARMv8-A+ |

### 1.3 Faiss的SIMD抽象层

```cpp
// faiss/utils/simdlib.h
// Faiss通过条件编译选择SIMD实现

#if defined(__AVX512F__)
    #include <faiss/utils/simdlib_avx2.h>
    #include <faiss/utils/simdlib_avx512.h>
#elif defined(__AVX2__)
    #include <faiss/utils/simdlib_avx2.h>
#elif defined(__aarch64__)
    #include <faiss/utils/simdlib_neon.h>
#elif defined(__PPC64__)
    #include <faiss/utils/simdlib_ppc64.h>
#else
    #include <faiss/utils/simdlib_emulated.h>
#endif
```

---

## 2. AVX2 SIMD库详解

### 2.1 基础数据结构

```cpp
// faiss/utils/simdlib_avx2.h

// 256位寄存器的无类型表示
struct simd256bit {
    union {
        __m256i i;  // 8个int32 或 16个int16 或 32个int8
        __m256 f;   // 8个float
    };

    simd256bit() {}

    explicit simd256bit(__m256i i) : i(i) {}
    explicit simd256bit(__m256 f) : f(f) {}

    // 从内存加载
    explicit simd256bit(const void* x)
        : i(_mm256_load_si256((__m256i const*)x)) {}

    // 存储到内存
    void storeu(void* ptr) const {
        _mm256_storeu_si256((__m256i*)ptr, i);
    }

    void store(void* ptr) const {
        _mm256_store_si256((__m256i*)ptr, i);
    }
};
```

### 2.2 simd8float32 - 8个float的SIMD向量

```cpp
struct simd8float32 : simd256bit {
    simd8float32() {}
    explicit simd8float32(__m256 x) : simd256bit(x) {}
    explicit simd8float32(float x) : simd256bit(_mm256_set1_ps(x)) {}

    // 从数组加载（不需要对齐）
    explicit simd8float32(const float* x)
        : simd256bit(_mm256_loadu_ps(x)) {}

    // 构造8个float
    explicit simd8float32(float f0, float f1, float f2, float f3,
                         float f4, float f5, float f6, float f7)
        : simd256bit(_mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7)) {}

    // 加法
    simd8float32 operator+(simd8float32 other) const {
        return simd8float32(_mm256_add_ps(f, other.f));
    }

    // 减法
    simd8float32 operator-(simd8float32 other) const {
        return simd8float32(_mm256_sub_ps(f, other.f));
    }

    // 乘法
    simd8float32 operator*(simd8float32 other) const {
        return simd8float32(_mm256_mul_ps(f, other.f));
    }

    // FMA: a * b + c
    friend simd8float32 fmadd(simd8float32 a, simd8float32 b, simd8float32 c) {
        return simd8float32(_mm256_fmadd_ps(a.f, b.f, c.f));
    }
};
```

### 2.3 simd32uint8 - 32个uint8的SIMD向量

```cpp
struct simd32uint8 : simd256bit {
    simd32uint8() {}
    explicit simd32uint8(__m256i i) : simd256bit(i) {}
    explicit simd32uint8(uint8_t x) : simd256bit(_mm256_set1_epi8(x)) {}

    // 从数组加载
    explicit simd32uint8(const uint8_t* x) : simd256bit((const void*)x) {}

    // 按位与
    simd32uint8 operator&(simd256bit other) const {
        return simd32uint8(_mm256_and_si256(i, other.i));
    }

    // 加法（饱和加法）
    simd32uint8 operator+(simd32uint8 other) const {
        return simd32uint8(_mm256_add_epi8(i, other.i));
    }

    // 使用查找表进行置换
    simd32uint8 lookup_2_lanes(simd32uint8 idx) const {
        return simd32uint8(_mm256_shuffle_epi8(i, idx.i));
    }
};

// 获取每个字节的最高有效位
inline uint32_t get_MSBs(simd32uint8 a) {
    return _mm256_movemask_epi8(a.i);
}
```

### 2.4 simd16uint16 - 16个uint16的SIMD向量

```cpp
struct simd16uint16 : simd256bit {
    simd16uint16() {}
    explicit simd16uint16(__m256i i) : simd256bit(i) {}
    explicit simd16uint16(uint16_t x) : simd256bit(_mm256_set1_epi16(x)) {}

    // 乘法（低16位）
    simd16uint16 operator*(const simd16uint16& other) const {
        return simd16uint16(_mm256_mullo_epi16(i, other.i));
    }

    // 右移（编译时常量）
    simd16uint16 operator>>(const int shift) const {
        return simd16uint16(_mm256_srli_epi16(i, shift));
    }

    // 加法
    simd16uint16 operator+(simd16uint16 other) const {
        return simd16uint16(_mm256_add_epi16(i, other.i));
    }

    // 元级最小值
    friend simd16uint16 min(simd16uint16 a, simd16uint16 b) {
        return simd16uint16(_mm256_min_epu16(a.i, b.i));
    }

    // 元级最大值
    friend simd16uint16 max(simd16uint16 a, simd16uint16 b) {
        return simd16uint16(_mm256_max_epu16(a.i, b.i));
    }
};
```

---

## 3. 内积计算的SIMD优化

### 3.1 标量实现

```cpp
// 标量版本：每次处理1个元素
float fvec_inner_product_scalar(const float* x, const float* y, size_t d) {
    float res = 0.0f;
    for (size_t i = 0; i < d; i++) {
        res += x[i] * y[i];
    }
    return res;
}
```

### 3.2 AVX2优化实现

```cpp
// SIMD优化版本：每次处理8个元素
float fvec_inner_product_avx2(const float* x, const float* y, size_t d) {
    float res = 0.0f;
    size_t i = 0;

    // 处理8的倍数部分
    if (d >= 8) {
        __m256 sum = _mm256_setzero_ps();

        while (i + 8 <= d) {
            __m256 vx = _mm256_loadu_ps(x + i);  // 加载8个float
            __m256 vy = _mm256_loadu_ps(y + i);  // 加载8个float
            sum = _mm256_fmadd_ps(vx, vy, sum);  // vx * vy + sum
            i += 8;
        }

        // 水平求和：将8个部分和相加
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);

        // 提取结果
        alignas(32) float tmp[8];
        _mm256_storeu_ps(tmp, sum);
        res = tmp[0] + tmp[4];
    }

    // 处理剩余元素
    for (; i < d; i++) {
        res += x[i] * y[i];
    }

    return res;
}
```

### 3.3 使用Faiss SIMD抽象层

```cpp
float fvec_inner_product_faiss(const float* x, const float* y, size_t d) {
    float res = 0.0f;
    size_t i = 0;

    // 使用simd8float32抽象
    if (d >= 8) {
        simd8float32 sum(0.0f);

        while (i + 8 <= d) {
            simd8float32 vx(x + i);
            simd8float32 vy(y + i);
            sum = sum + vx * vy;  // 或 fmadd(vx, vy, sum)
            i += 8;
        }

        // 水平求和
        alignas(32) float tmp[8];
        sum.storeu(tmp);
        for (int j = 0; j < 8; j++) {
            res += tmp[j];
        }
    }

    // 处理剩余元素
    for (; i < d; i++) {
        res += x[i] * y[i];
    }

    return res;
}
```

### 3.4 性能对比

| 实现 | 每次迭代处理元素 | 循环次数 | 相对性能 |
|------|-----------------|---------|---------|
| 标量 | 1 | d | 1.0x |
| SSE | 4 | d/4 | ~3.5x |
| AVX | 8 | d/8 | ~7x |
| AVX2 | 8 | d/8 | ~7.5x |
| AVX-512 | 16 | d/16 | ~15x |

---

## 4. L2距离计算的SIMD优化

### 4.1 优化公式

```cpp
// L2距离：||x - y||^2 = sum((x[i] - y[i])^2)
// 使用FMA指令优化

// 方法1：直接计算
float dis1 = 0.0f;
for (size_t i = 0; i < d; i++) {
    float diff = x[i] - y[i];
    dis1 += diff * diff;
}

// 方法2：展开为 ||x||^2 + ||y||^2 - 2<x|y>
// 当y的范数预计算时更快
float dis2 = norm_x + norm_y - 2 * inner_product(x, y, d);
```

### 4.2 AVX2实现 - 直接计算

```cpp
float fvec_L2sqr_avx2(const float* x, const float* y, size_t d) {
    float res = 0.0f;
    size_t i = 0;

    if (d >= 8) {
        __m256 sum = _mm256_setzero_ps();

        while (i + 8 <= d) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = _mm256_loadu_ps(y + i);
            __m256 vd = _mm256_sub_ps(vx, vy);      // x - y
            sum = _mm256_fmadd_ps(vd, vd, sum);    // (x-y)^2 + sum
            i += 8;
        }

        // 水平求和
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        alignas(32) float tmp[8];
        _mm256_storeu_ps(tmp, sum);
        res = tmp[0] + tmp[4];
    }

    for (; i < d; i++) {
        float diff = x[i] - y[i];
        res += diff * diff;
    }

    return res;
}
```

### 4.3 使用预计算范数

```cpp
// 预计算y的L2范数
void fvec_norms_L2sqr_avx2(float* norms, const float* x, size_t d, size_t nx) {
    for (size_t i = 0; i < nx; i++) {
        const float* xi = x + i * d;
        float norm = 0.0f;
        size_t j = 0;

        if (d >= 8) {
            __m256 sum = _mm256_setzero_ps();
            while (j + 8 <= d) {
                __m256 vx = _mm256_loadu_ps(xi + j);
                sum = _mm256_fmadd_ps(vx, vx, sum);
                j += 8;
            }

            sum = _mm256_hadd_ps(sum, sum);
            sum = _mm256_hadd_ps(sum, sum);
            alignas(32) float tmp[8];
            _mm256_storeu_ps(tmp, sum);
            norm = tmp[0] + tmp[4];
        }

        for (; j < d; j++) {
            norm += xi[j] * xi[j];
        }

        norms[i] = norm;
    }
}

// 使用预计算范数的L2距离
float fvec_L2sqr_with_norm_avx2(
        const float* x,
        const float* y,
        size_t d,
        float norm_y) {
    float norm_x = fvec_norm_L2sqr(x, d);
    float ip = fvec_inner_product_avx2(x, y, d);
    return norm_x + norm_y - 2 * ip;
}
```

---

## 5. 批量距离计算

### 5.1 批量内积计算

```cpp
// faiss/utils/distances.h
void fvec_inner_product_batch_4(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& ip0,
        float& ip1,
        float& ip2,
        float& ip3)
{
    // 同时计算x与4个向量的内积
    // 利用SIMD寄存器并行度

    size_t i = 0;
    simd8float32 sum0(0.0f), sum1(0.0f), sum2(0.0f), sum3(0.0f);

    // 主循环：每次处理d个维度（但展开4个向量）
    while (i + 8 <= d) {
        simd8float32 vx(x + i);
        simd8float32 vy0(y0 + i);
        simd8float32 vy1(y1 + i);
        simd8float32 vy2(y2 + i);
        simd8float32 vy3(y3 + i);

        sum0 += vx * vy0;
        sum1 += vx * vy1;
        sum2 += vx * vy2;
        sum3 += vx * vy3;

        i += 8;
    }

    // 水平求和
    ip0 = 0; ip1 = 0; ip2 = 0; ip3 = 0;
    alignas(32) float tmp0[8], tmp1[8], tmp2[8], tmp3[8];

    sum0.storeu(tmp0);
    sum1.storeu(tmp1);
    sum2.storeu(tmp2);
    sum3.storeu(tmp3);

    for (int j = 0; j < 8; j++) {
        ip0 += tmp0[j];
        ip1 += tmp1[j];
        ip2 += tmp2[j];
        ip3 += tmp3[j];
    }

    // 处理剩余元素
    for (; i < d; i++) {
        float xi = x[i];
        ip0 += xi * y0[i];
        ip1 += xi * y1[i];
        ip2 += xi * y2[i];
        ip3 += xi * y3[i];
    }
}
```

### 5.2 批量L2距离计算

```cpp
void fvec_L2sqr_batch_4(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3)
{
    size_t i = 0;
    simd8float32 sum0(0.0f), sum1(0.0f), sum2(0.0f), sum3(0.0f);

    while (i + 8 <= d) {
        simd8float32 vx(x + i);
        simd8float32 vy0(y0 + i);
        simd8float32 vy1(y1 + i);
        simd8float32 vy2(y2 + i);
        simd8float32 vy3(y3 + i);

        simd8float32 vd0 = vx - vy0;
        simd8float32 vd1 = vx - vy1;
        simd8float32 vd2 = vx - vy2;
        simd8float32 vd3 = vx - vy3;

        sum0 += vd0 * vd0;
        sum1 += vd1 * vd1;
        sum2 += vd2 * vd2;
        sum3 += vd3 * vd3;

        i += 8;
    }

    // 水平求和
    dis0 = 0; dis1 = 0; dis2 = 0; dis3 = 0;
    alignas(32) float tmp0[8], tmp1[8], tmp2[8], tmp3[8];

    sum0.storeu(tmp0);
    sum1.storeu(tmp1);
    sum2.storeu(tmp2);
    sum3.storeu(tmp3);

    for (int j = 0; j < 8; j++) {
        dis0 += tmp0[j];
        dis1 += tmp1[j];
        dis2 += tmp2[j];
        dis3 += tmp3[j];
    }

    // 剩余元素
    for (; i < d; i++) {
        float xi = x[i];
        float d0 = xi - y0[i];
        float d1 = xi - y1[i];
        float d2 = xi - y2[i];
        float d3 = xi - y3[i];
        dis0 += d0 * d0;
        dis1 += d1 * d1;
        dis2 += d2 * d2;
        dis3 += d3 * d3;
    }
}
```

---

## 6. KNN搜索的SIMD优化

### 6.1 搜索算法结构

```cpp
// KNN搜索使用堆维护top-k结果
// 每次查询需要与所有数据库向量计算距离

void knn_L2sqr_simd(
        const float* x,      // nq * d
        const float* y,      // nb * d
        size_t d,
        size_t nq,
        size_t nb,
        size_t k,
        float* distances,    // nq * k
        idx_t* labels)       // nq * k
{
#pragma omp parallel for
    for (size_t i = 0; i < nq; i++) {
        const float* xi = x + i * d;
        float* __restrict simi = distances + i * k;
        idx_t* __restrict idxi = labels + i * k;

        // 初始化堆
        heap_heapify<CMax<float, idx_t>>(k, simi, idxi);

        // 遍历所有数据库向量
        for (size_t j = 0; j < nb; j++) {
            // 使用SIMD优化的距离计算
            float dis = fvec_L2sqr_avx2(xi, y + j * d, d);

            // 如果距离小于堆顶，替换
            if (CMax<float, idx_t>::cmp(dis, simi[0])) {
                heap_replace_top<CMax<float, idx_t>>(k, simi, idxi, dis, j);
            }
        }

        // 堆排序
        heap_reorder<CMax<float, idx_t>>(k, simi, idxi);
    }
}
```

### 6.2 DistanceComputer接口

```cpp
// faiss/IndexFlat.cpp
struct FlatL2Dis : FlatCodesDistanceComputer {
    size_t d;
    const float* q;

    float distance_to_code(const uint8_t* code) final {
        return fvec_L2sqr_avx2(q, reinterpret_cast<const float*>(code), d);
    }

    // 批量计算4个距离
    void distances_batch_4(
            const idx_t idx0, const idx_t idx1,
            const idx_t idx2, const idx_t idx3,
            float& dis0, float& dis1, float& dis2, float& dis3) final {
        const float* y0 = reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* y1 = reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* y2 = reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* y3 = reinterpret_cast<const float*>(codes + idx3 * code_size);

        fvec_L2sqr_batch_4(q, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
    }
};
```

---

## 7. ARM NEON优化

### 7.1 NEON基础

```cpp
// faiss/utils/simdlib_neon.h
// ARM NEON的128位寄存器

// 4个float的SIMD向量
struct simd4float32 {
    float32x4_t v;

    simd4float32(float x) : v(vdupq_n_f32(x)) {}
    simd4float32(const float* x) : v(vld1q_f32(x)) {}

    simd4float32 operator+(simd4float32 other) const {
        return simd4float32(vaddq_f32(v, other.v));
    }

    simd4float32 operator*(simd4float32 other) const {
        return simd4float32(vmulq_f32(v, other.v));
    }
};

// 16个uint8的SIMD向量
struct simd16uint8 {
    uint8x16_t v;

    simd16uint8(uint8_t x) : v(vdupq_n_u8(x)) {}
    simd16uint8(const uint8_t* x) : v(vld1q_u8(x)) {}

    // 查找表置换
    simd16uint16 lookup_2_lanes(simd16uint8 idx) const {
        return simd16uint16(vtbl1q_u8(v, idx.v));
    }
};
```

### 7.2 NEON内积实现

```cpp
float fvec_inner_product_neon(const float* x, const float* y, size_t d) {
    float res = 0.0f;
    size_t i = 0;

    if (d >= 4) {
        float32x4_t sum = vdupq_n_f32(0.0f);

        while (i + 4 <= d) {
            float32x4_t vx = vld1q_f32(x + i);
            float32x4_t vy = vld1q_f32(y + i);
            sum = vfmaq_f32(sum, vx, vy);  // sum += vx * vy
            i += 4;
        }

        // 水平求和
        float tmp[4];
        vst1q_f32(tmp, sum);
        res = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }

    for (; i < d; i++) {
        res += x[i] * y[i];
    }

    return res;
}
```

---

## 8. AVX-512优化

### 8.1 AVX-512优势

```cpp
// AVX-512提供512位寄存器，可以处理16个float
// faiss/utils/simdlib_avx512.h

struct simd16float32 {
    __m512 v;

    simd16float32(float x) : v(_mm512_set1_ps(x)) {}
    simd16float32(const float* x) : v(_mm512_loadu_ps(x)) {}

    simd16float32 operator+(simd16float32 other) const {
        return simd16float32(_mm512_add_ps(v, other.v));
    }

    simd16float32 operator*(simd16float32 other) const {
        return simd16float32(_mm512_mul_ps(v, other.v));
    }
};
```

### 8.2 AVX-512内积实现

```cpp
float fvec_inner_product_avx512(const float* x, const float* y, size_t d) {
    float res = 0.0f;
    size_t i = 0;

    if (d >= 16) {
        __m512 sum = _mm512_setzero_ps();

        while (i + 16 <= d) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vy = _mm512_loadu_ps(y + i);
            sum = _mm512_fmadd_ps(vx, vy, sum);
            i += 16;
        }

        // 水平求和
        res = _mm512_reduce_add_ps(sum);
    }

    for (; i < d; i++) {
        res += x[i] * y[i];
    }

    return res;
}
```

---

## 9. 数据预取优化

### 9.1 预取指令

```cpp
// faiss/utils/prefetch.h
// 预取数据到缓存，减少内存延迟

// 预取到L2缓存
inline void prefetch_L2(const void* ptr) {
#if defined(__AVX2__)
    _mm_prefetch((const char*)ptr, _MM_HINT_T1);  // L2
#elif defined(__aarch64__)
    __builtin_prefetch(ptr, 0, 2);  // L2
#endif
}

// 预取到L1缓存
inline void prefetch_L1(const void* ptr) {
#if defined(__AVX2__)
    _mm_prefetch((const char*)ptr, _MM_HINT_T0);  // L1
#elif defined(__aarch64__)
    __builtin_prefetch(ptr, 0, 3);  // L1
#endif
}
```

### 9.2 软件预取

```cpp
// 在KNN搜索中使用预取
void knn_L2sqr_with_prefetch(
        const float* x,
        const float* y,
        size_t d,
        size_t nb,
        size_t k,
        float* distances,
        idx_t* labels)
{
    const size_t prefetch_distance = 8;  // 预取距离

    for (size_t j = 0; j < nb; j++) {
        // 预取未来的数据
        if (j + prefetch_distance < nb) {
            prefetch_L2(y + (j + prefetch_distance) * d);
        }

        float dis = fvec_L2sqr_avx2(x, y + j * d, d);
        // ... 更新堆
    }
}
```

---

## 10. 内存对齐优化

### 10.1 对齐的重要性

SIMD指令对内存对齐敏感：
- 对齐的内存访问更快
- 某些指令要求对齐

### 10.2 AlignedTable

```cpp
// faiss/utils/AlignedTable.h
template <class T>
struct AlignedTable {
    T* data;
    size_t n;

    explicit AlignedTable(size_t n = 0) : n(n) {
        if (n > 0) {
            // 分配32字节对齐的内存
            data = (T*)aligned_alloc(32, n * sizeof(T));
        }
    }

    ~AlignedTable() {
        free(data);
    }
};
```

### 10.3 使用对齐的加载/存储

```cpp
// 使用对齐的加载（更快，但要求地址对齐）
void aligned_example() {
    // 分配对齐的内存
    alignas(32) float x[8];
    alignas(32) float y[8];
    alignas(32) float result[8];

    // 对齐加载
    __m256 vx = _mm256_load_ps(x);   // 要求32字节对齐
    __m256 vy = _mm256_load_ps(y);

    __m256 vr = _mm256_mul_ps(vx, vy);

    // 对齐存储
    _mm256_store_ps(result, vr);

    // 对比：非对齐版本（稍慢，但不需要对齐）
    __m256 vx_u = _mm256_loadu_ps(x);  // 不需要对齐
}
```

---

## 11. 性能分析与调优

### 11.1 性能测量

```cpp
#include <chrono>

void benchmark_distance_functions() {
    int d = 128;
    int n = 10000000;

    float* x = new float[n * d];
    float* y = new float[n * d];

    // 标量版本
    auto t0 = std::chrono::high_resolution_clock::now();
    float sum_scalar = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_scalar += fvec_inner_product_scalar(x + i * d, y + i * d, d);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Scalar: %.3f s, result=%.3f\n",
           (t1 - t0).count() / 1e9, sum_scalar);

    // SIMD版本
    t0 = std::chrono::high_resolution_clock::now();
    float sum_simd = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_simd += fvec_inner_product_avx2(x + i * d, y + i * d, d);
    }
    t1 = std::chrono::high_resolution_clock::now();
    printf("AVX2:   %.3f s, result=%.3f\n",
           (t1 - t0).count() / 1e9, sum_simd);
}
```

### 11.2 优化检查清单

- [ ] 使用SIMD指令（AVX2/AVX-512/NEON）
- [ ] 数据对齐（32字节或64字节）
- [ ] 避免分支预测失败
- [ ] 使用预取减少内存延迟
- [ ] 循环展开
- [ ] 使用FMA指令
- [ ] 多线程并行化
- [ ] 缓存友好的内存访问模式

---

## 10. 源码深度实现 - 距离计算SIMD优化

### 10.1 distances.cpp核心实现

```cpp
// faiss/utils/distances.cpp
// 距离计算的核心实现，支持多线程和SIMD优化

// L2范数计算（支持多线程）
void fvec_norms_L2sqr(
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx) {
#pragma omp parallel for if (nx > 10000)
    for (int64_t i = 0; i < nx; i++) {
        nr[i] = fvec_norm_L2sqr(x + i * d, d);
    }
}

// L2向量重归一化
#define FVEC_RENORM_L2_IMPL                   \
    float* __restrict xi = x + i * d;         \
                                              \
    float nr = fvec_norm_L2sqr(xi, d);        \
                                              \
    if (nr > 0) {                             \
        size_t j;                             \
        const float inv_nr = 1.0 / sqrtf(nr); \
        for (j = 0; j < d; j++)               \
            xi[j] *= inv_nr;                  \
    }

void fvec_renorm_L2(size_t d, size_t nx, float* __restrict x) {
    if (nx <= 10000) {
        fvec_renorm_L2_noomp(d, nx, x);
    } else {
        fvec_renorm_L2_omp(d, nx, x);
    }
}
```

### 10.2 KNN搜索实现

```cpp
// 内积KNN搜索的序列化实现
template <class BlockResultHandler>
void exhaustive_inner_product_seq(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res) {
    using SingleResultHandler =
            typename BlockResultHandler::SingleResultHandler;
    [[maybe_unused]] int nt = std::min(int(nx), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
#pragma omp for
        for (int64_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            const float* y_j = y;

            resi.begin(i);

            for (size_t j = 0; j < ny; j++, y_j += d) {
                if (!res.is_in_selection(j)) {
                    continue;
                }
                float ip = fvec_inner_product(x_i, y_j, d);
                resi.add_result(ip, j);
            }
            resi.end();
        }
    }
}

// L2距离KNN搜索的序列化实现
template <class BlockResultHandler>
void exhaustive_L2sqr_seq(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res) {
    using SingleResultHandler =
            typename BlockResultHandler::SingleResultHandler;
    [[maybe_unused]] int nt = std::min(int(nx), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
#pragma omp for
        for (int64_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            const float* y_j = y;
            resi.begin(i);
            for (size_t j = 0; j < ny; j++, y_j += d) {
                if (!res.is_in_selection(j)) {
                    continue;
                }
                float disij = fvec_L2sqr(x_i, y_j, d);
                resi.add_result(disij, j);
            }
            resi.end();
        }
    }
}
```

### 10.3 BLAS加速实现

```cpp
// 使用BLAS SGEMM进行批量内积计算
template <class BlockResultHandler>
void exhaustive_inner_product_blas(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res) {
    if (nx == 0 || ny == 0) {
        return;
    }

    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx) {
            i1 = nx;
        }

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny) {
                j1 = ny;
            }
            // 计算内积：C = Y^T * X
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose",        // Y转置
                       "Not transpose",    // X不转置
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       x + i0 * d,
                       &di,
                       &zero,
                       ip_block.get(),
                       &nyi);
            }

            res.add_results(j0, j1, ip_block.get());
        }
        res.end_multiple();
        InterruptCallback::check();
    }
}
```

### 10.4 L2距离的BLAS优化

```cpp
// 使用内积公式：||x-y||^2 = ||x||^2 + ||y||^2 - 2*<x,y>
template <class BlockResultHandler>
void exhaustive_L2sqr_blas_default_impl(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res,
        const float* y_norms = nullptr) {
    if (nx == 0 || ny == 0) {
        return;
    }

    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);
    std::unique_ptr<float[]> x_norms(new float[nx]);
    std::unique_ptr<float[]> del2;

    fvec_norms_L2sqr(x_norms.get(), x, d, nx);

    if (!y_norms) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2);
        fvec_norms_L2sqr(y_norms2, y, d, ny);
        y_norms = y_norms2;
    }

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx) {
            i1 = nx;
        }

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny) {
                j1 = ny;
            }
            // 计算内积
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       x + i0 * d,
                       &di,
                       &zero,
                       ip_block.get(),
                       &nyi);
            }
            // 转换为L2距离
            for (int64_t i = i0; i < i1; i++) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);

                for (size_t j = j0; j < j1; j++) {
                    float ip = *ip_line;
                    float dis = x_norms[i] + y_norms[j] - 2 * ip;

                    if (!res.is_in_selection(j)) {
                        dis = HUGE_VALF;
                    }
                    // 负值处理（相同向量的舍入误差）
                    if (dis < 0) {
                        dis = 0;
                    }

                    *ip_line = dis;
                    ip_line++;
                }
            }
            res.add_results(j0, j1, ip_block.get());
        }
        res.end_multiple();
        InterruptCallback::check();
    }
}
```

### 10.5 AVX2专用的Top-1优化

```cpp
// AVX2专用的Top-1搜索优化（查找最近的单个向量）
#ifdef __AVX2__
void exhaustive_L2sqr_blas_cmax_avx2(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    if (nx == 0 || ny == 0) {
        return;
    }

    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);
    std::unique_ptr<float[]> x_norms(new float[nx]);
    std::unique_ptr<float[]> del2;

    fvec_norms_L2sqr(x_norms.get(), x, d, nx);

    if (!y_norms) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2);
        fvec_norms_L2sqr(y_norms2, y, d, ny);
        y_norms = y_norms2;
    }

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx) {
            i1 = nx;
        }

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny) {
                j1 = ny;
            }
            // 计算内积
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       x + i0 * d,
                       &di,
                       &zero,
                       ip_block.get(),
                       &nyi);
            }
            for (int64_t i = i0; i < i1; i++) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);

                _mm_prefetch((const char*)ip_line, _MM_HINT_NTA);
                _mm_prefetch((const char*)(ip_line + 16), _MM_HINT_NTA);

                const __m256 mul_minus2 = _mm256_set1_ps(-2);

                // 跟踪8个最小距离和索引
                __m256 min_distances =
                        _mm256_set1_ps(res.dis_tab[i] - x_norms[i]);
                __m256i min_indices = _mm256_set1_epi32(0);
                __m256i current_indices =
                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                const __m256i indices_delta = _mm256_set1_epi32(8);

                size_t idx_j = 0;
                size_t count = j1 - j0;

                // 每次处理16个元素
                for (; idx_j < (count / 16) * 16; idx_j += 16, ip_line += 16) {
                    _mm_prefetch((const char*)(ip_line + 32), _MM_HINT_NTA);
                    _mm_prefetch((const char*)(ip_line + 48), _MM_HINT_NTA);

                    const __m256 y_norm_0 =
                            _mm256_loadu_ps(y_norms + idx_j + j0 + 0);
                    const __m256 y_norm_1 =
                            _mm256_loadu_ps(y_norms + idx_j + j0 + 8);

                    const __m256 ip_0 = _mm256_loadu_ps(ip_line + 0);
                    const __m256 ip_1 = _mm256_loadu_ps(ip_line + 8);

                    // dis = y_norm - 2 * ip (x_norm已省略，稍后添加)
                    __m256 distances_0 =
                            _mm256_fmadd_ps(ip_0, mul_minus2, y_norm_0);
                    __m256 distances_1 =
                            _mm256_fmadd_ps(ip_1, mul_minus2, y_norm_1);

                    // 比较并更新最小值
                    const __m256 comparison_0 = _mm256_cmp_ps(
                            min_distances, distances_0, _CMP_LE_OS);

                    min_distances = _mm256_blendv_ps(
                            distances_0, min_distances, comparison_0);
                    min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                            _mm256_castsi256_ps(current_indices),
                            _mm256_castsi256_ps(min_indices),
                            comparison_0));
                    current_indices =
                            _mm256_add_epi32(current_indices, indices_delta);

                    const __m256 comparison_1 = _mm256_cmp_ps(
                            min_distances, distances_1, _CMP_LE_OS);

                    min_distances = _mm256_blendv_ps(
                            distances_1, min_distances, comparison_1);
                    min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                            _mm256_castsi256_ps(current_indices),
                            _mm256_castsi256_ps(min_indices),
                            comparison_1));
                    current_indices =
                            _mm256_add_epi32(current_indices, indices_delta);
                }

                // 提取并找到最终最小值
                float min_distances_scalar[8];
                uint32_t min_indices_scalar[8];
                _mm256_storeu_ps(min_distances_scalar, min_distances);
                _mm256_storeu_si256((__m256i*)(min_indices_scalar), min_indices);

                float current_min_distance = res.dis_tab[i];
                uint32_t current_min_index = res.ids_tab[i];

                for (size_t jv = 0; jv < 8; jv++) {
                    float distance_candidate =
                            min_distances_scalar[jv] + x_norms[i];

                    if (distance_candidate < 0) {
                        distance_candidate = 0;
                    }

                    int64_t index_candidate = min_indices_scalar[jv] + j0;

                    if (current_min_distance > distance_candidate) {
                        current_min_distance = distance_candidate;
                        current_min_index = index_candidate;
                    } else if (
                            current_min_distance == distance_candidate &&
                            current_min_index > index_candidate) {
                        current_min_index = index_candidate;
                    }
                }

                // 处理剩余元素
                for (; idx_j < count; idx_j++, ip_line++) {
                    float ip = *ip_line;
                    float dis = x_norms[i] + y_norms[idx_j + j0] - 2 * ip;

                    if (dis < 0) {
                        dis = 0;
                    }

                    if (current_min_distance > dis) {
                        current_min_distance = dis;
                        current_min_index = idx_j + j0;
                    }
                }

                res.add_result(i, current_min_distance, current_min_index);
            }
        }
        res.end_multiple();
        InterruptCallback::check();
    }
}
#endif
```

### 10.6 SIMD优化的批量距离计算

```cpp
// faiss/utils/distances_simd.cpp
// 批量内积计算：同时计算4个内积
void fvec_inner_product_batch_4(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& ip0,
        float& ip1,
        float& ip2,
        float& ip3) {
    size_t i = 0;
    simd8float32 sum0(0.0f), sum1(0.0f), sum2(0.0f), sum3(0.0f);

    while (i + 8 <= d) {
        simd8float32 vx(x + i);
        simd8float32 vy0(y0 + i);
        simd8float32 vy1(y1 + i);
        simd8float32 vy2(y2 + i);
        simd8float32 vy3(y3 + i);

        sum0 += vx * vy0;
        sum1 += vx * vy1;
        sum2 += vx * vy2;
        sum3 += vx * vy3;

        i += 8;
    }

    // 水平求和
    ip0 = 0; ip1 = 0; ip2 = 0; ip3 = 0;
    alignas(32) float tmp0[8], tmp1[8], tmp2[8], tmp3[8];

    sum0.storeu(tmp0);
    sum1.storeu(tmp1);
    sum2.storeu(tmp2);
    sum3.storeu(tmp3);

    for (int j = 0; j < 8; j++) {
        ip0 += tmp0[j];
        ip1 += tmp1[j];
        ip2 += tmp2[j];
        ip3 += tmp3[j];
    }

    // 处理剩余元素
    for (; i < d; i++) {
        float xi = x[i];
        ip0 += xi * y0[i];
        ip1 += xi * y1[i];
        ip2 += xi * y2[i];
        ip3 += xi * y3[i];
    }
}

// 批量L2距离计算：同时计算4个L2距离
void fvec_L2sqr_batch_4(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    size_t i = 0;
    simd8float32 sum0(0.0f), sum1(0.0f), sum2(0.0f), sum3(0.0f);

    while (i + 8 <= d) {
        simd8float32 vx(x + i);
        simd8float32 vy0(y0 + i);
        simd8float32 vy1(y1 + i);
        simd8float32 vy2(y2 + i);
        simd8float32 vy3(y3 + i);

        simd8float32 vd0 = vx - vy0;
        simd8float32 vd1 = vx - vy1;
        simd8float32 vd2 = vx - vy2;
        simd8float32 vd3 = vx - vy3;

        sum0 += vd0 * vd0;
        sum1 += vd1 * vd1;
        sum2 += vd2 * vd2;
        sum3 += vd3 * vd3;

        i += 8;
    }

    // 水平求和
    dis0 = 0; dis1 = 0; dis2 = 0; dis3 = 0;
    alignas(32) float tmp0[8], tmp1[8], tmp2[8], tmp3[8];

    sum0.storeu(tmp0);
    sum1.storeu(tmp1);
    sum2.storeu(tmp2);
    sum3.storeu(tmp3);

    for (int j = 0; j < 8; j++) {
        dis0 += tmp0[j];
        dis1 += tmp1[j];
        dis2 += tmp2[j];
        dis3 += tmp3[j];
    }

    // 剩余元素
    for (; i < d; i++) {
        float xi = x[i];
        float d0 = xi - y0[i];
        float d1 = xi - y1[i];
        float d2 = xi - y2[i];
        float d3 = xi - y3[i];
        dis0 += d0 * d0;
        dis1 += d1 * d1;
        dis2 += d2 * d2;
        dis3 += d3 * d3;
    }
}
```

### 10.7 Heap数据结构实现

```cpp
// faiss/utils/Heap.h
// 堆操作的模板实现

// 替换堆顶元素
template <class C>
inline void heap_replace_top(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T val,
        typename C::TI id) {
    bh_val--; /* 使用1-based索引便于父子节点转换 */
    bh_ids--;
    size_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1;  // 左孩子
        i2 = i1 + 1;  // 右孩子
        if (i1 > k) {
            break;
        }

        // C::cmp2() 返回 (a1 > b1) || ((a1 == b1) && (a2 > b2)) for max heap
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

// 堆初始化
template <class C>
inline void heap_heapify(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        const typename C::T* x,
        const typename C::TI* ids,
        size_t k0) {
    if (k0 > 0) {
        assert(x);
    }

    if (ids) {
        for (size_t i = 0; i < k0; i++) {
            heap_push<C>(i + 1, bh_val, bh_ids, x[i], ids[i]);
        }
    } else {
        for (size_t i = 0; i < k0; i++) {
            heap_push<C>(i + 1, bh_val, bh_ids, x[i], i);
        }
    }

    for (size_t i = k0; i < k; i++) {
        bh_val[i] = C::neutral();
        bh_ids[i] = -1;
    }
}

// 堆重新排序（转换为有序数组）
template <typename C>
inline size_t heap_reorder(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids) {
    size_t i, ii;

    for (i = 0, ii = 0; i < k; i++) {
        typename C::T val = bh_val[0];
        typename C::TI id = bh_ids[0];

        heap_pop<C>(k - i, bh_val, bh_ids);
        bh_val[k - ii - 1] = val;
        bh_ids[k - ii - 1] = id;
        if (id != -1) {
            ii++;
        }
    }

    size_t nel = ii;

    memmove(bh_val, bh_val + k - ii, ii * sizeof(*bh_val));
    memmove(bh_ids, bh_ids + k - ii, ii * sizeof(*bh_ids));

    for (; ii < k; ii++) {
        bh_val[ii] = C::neutral();
        bh_ids[ii] = -1;
    }
    return nel;
}
```

### 10.8 性能优化总结表

| 优化技术 | 加速比 | 适用场景 |
|---------|--------|----------|
| AVX2 SIMD | 6-8x | 批量距离计算 |
| AVX-512 | 12-16x | Skylake-X及更新CPU |
| ARM NEON | 3-4x | ARM64平台 |
| BLAS SGEMM | 2-4x | 大规模批量计算 |
| 数据预取 | 1.2-1.5x | 内存受限场景 |
| OpenMP并行 | 线性核数数 | 多核CPU |

---

## 11. 性能分析与调优

### 关键概念回顾

1. **SIMD基础**：单指令处理多数据，提升并行度
2. **AVX2**：256位寄存器，8个float并行处理
3. **距离计算优化**：内积和L2距离的SIMD实现
4. **批量处理**：同时计算多个距离，提高吞吐量
5. **数据预取**：减少内存延迟
6. **内存对齐**：提高访问速度
7. **跨平台支持**：AVX2、AVX-512、NEON、SVE

### 性能要点

- SIMD可以带来**4-16倍**加速（取决于指令集）
- 批量处理和预取进一步**提高性能**
- 内存对齐和缓存友好的访问模式**至关重要**

### 下一步

在第4天，我们将学习**Product Quantization（乘积量化）**，这是Faiss中最核心的压缩算法之一，大幅减少内存使用和加速搜索。

---

## 练习题

1. 实现一个SIMD优化的内积函数
2. 比较标量、SSE、AVX2、AVX-512的性能差异
3. 实现批量内积计算
4. 使用预取优化KNN搜索

## 12. SIMD进阶优化技术

### 12.1 循环展开优化

```cpp
// 手动循环展开以提高指令级并行
float fvec_inner_product_unrolled_avx2(const float* x, const float* y, size_t d) {
    float res = 0.0f;
    size_t i = 0;

    // 4路展开：每次处理32个float (4x8)
    if (d >= 32) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        // 循环展开减少分支开销
        while (i + 32 <= d) {
            __m256 vx0 = _mm256_loadu_ps(x + i + 0);
            __m256 vy0 = _mm256_loadu_ps(y + i + 0);
            __m256 vx1 = _mm256_loadu_ps(x + i + 8);
            __m256 vy1 = _mm256_loadu_ps(y + i + 8);
            __m256 vx2 = _mm256_loadu_ps(x + i + 16);
            __m256 vy2 = _mm256_loadu_ps(y + i + 16);
            __m256 vx3 = _mm256_loadu_ps(x + i + 24);
            __m256 vy3 = _mm256_loadu_ps(y + i + 24);

            // FMA链：sum = vx * vy + sum
            sum0 = _mm256_fmadd_ps(vx0, vy0, sum0);
            sum1 = _mm256_fmadd_ps(vx1, vy1, sum1);
            sum2 = _mm256_fmadd_ps(vx2, vy2, sum2);
            sum3 = _mm256_fmadd_ps(vx3, vy3, sum3);

            i += 32;
        }

        // 合并4个部分和
        __m256 sum01 = _mm256_add_ps(sum0, sum1);
        __m256 sum23 = _mm256_add_ps(sum2, sum3);
        __m256 sum = _mm256_add_ps(sum01, sum23);

        // 水平求和
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);

        alignas(32) float tmp[8];
        _mm256_storeu_ps(tmp, sum);
        res = tmp[0] + tmp[4];
    }

    // 处理8的倍数
    while (i + 8 <= d) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 prod = _mm256_mul_ps(vx, vy);
        __m256 partial_sum = _mm256_loadu_ps(&res);
        partial_sum = _mm256_add_ps(partial_sum, prod);
        _mm256_storeu_ps(&res, partial_sum);
        i += 8;
    }

    // 处理剩余元素
    for (; i < d; i++) {
        res += x[i] * y[i];
    }

    return res;
}

// 性能对比：展开vs不展开
// Unrolled版本可以更好地利用CPU的指令流水线
// 减少循环控制开销（分支、计数器更新）
// 允许CPU乱序执行更多指令
```

### 12.2 避免依赖链 - 指令级并行优化

```cpp
// 问题：连续的依赖链限制ILP
// bad: 顺序依赖
float sum_sequential = 0;
for (int i = 0; i < n; i++) {
    sum_sequential += data[i];  // 每次加法依赖前一次结果
}

// good: 4路累加器打破依赖链
float fvec_inner_product_ilp_optimized(const float* x, const float* y, size_t d) {
    // 使用4个独立的累加器
    float res0 = 0, res1 = 0, res2 = 0, res3 = 0;
    size_t i = 0;

    if (d >= 32) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        while (i + 32 <= d) {
            // 4条独立的计算链，CPU可以并行执行
            __m256 vx0 = _mm256_loadu_ps(x + i + 0);
            __m256 vy0 = _mm256_loadu_ps(y + i + 0);
            sum0 = _mm256_fmadd_ps(vx0, vy0, sum0);

            __m256 vx1 = _mm256_loadu_ps(x + i + 8);
            __m256 vy1 = _mm256_loadu_ps(y + i + 8);
            sum1 = _mm256_fmadd_ps(vx1, vy1, sum1);

            __m256 vx2 = _mm256_loadu_ps(x + i + 16);
            __m256 vy2 = _mm256_loadu_ps(y + i + 16);
            sum2 = _mm256_fmadd_ps(vx2, vy2, sum2);

            __m256 vx3 = _mm256_loadu_ps(x + i + 24);
            __m256 vy3 = _mm256_loadu_ps(y + i + 24);
            sum3 = _mm256_fmadd_ps(vx3, vy3, sum3);

            i += 32;
        }

        // 合并结果
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        __m256 sum = _mm256_add_ps(sum0, sum2);

        alignas(32) float tmp[8];
        _mm256_storeu_ps(tmp, sum);
        res0 = tmp[0] + tmp[4];
    }

    // 处理剩余...
    return res0;
}

// CPU可以同时执行这4条FMA指令（如果有4个FMA单元）
// 理论加速比：接近4x（在FMA单元充足的CPU上）
```

### 12.3 分支预测优化

```cpp
// 分支预测失败的代价：~15-20个周期
// 条件移动避免分支

// bad: 分支
float conditional_branch_bad(const float* a, const float* b, size_t d) {
    float sum = 0;
    for (size_t i = 0; i < d; i++) {
        if (a[i] > 0) {  // 不可预测的分支
            sum += a[i] * b[i];
        }
    }
    return sum;
}

// good: 无分支（使用位操作或条件移动）
float conditional_branch_good(const float* a, const float* b, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    const __m256 zero = _mm256_setzero_ps();
    const __m256 mask_sign = _mm256_set1_ps(-0.0f);  // 符号位掩码

    while (i + 8 <= d) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);

        // 创建掩码：a[i] > 0时全1，否则全0
        __m256 mask = _mm256_cmp_ps(va, zero, _CMP_GT_OQ);

        // 只保留掩码位置的乘积
        __m256 prod = _mm256_mul_ps(va, vb);
        prod = _mm256_and_ps(prod, mask);  // 按位与掩码

        sum = _mm256_add_ps(sum, prod);
        i += 8;
    }

    // 水平求和...
    return extract_sum(sum);
}

// 使用blendv避免分支
__m256 blendv_example(__m256 a, __m256 b, __m256 mask) {
    // 如果mask对应位为1，选a；否则选b
    // 相当于: result = mask ? a : b;
    return _mm256_blendv_ps(b, a, mask);  // 无分支实现
}
```

### 12.4 内存访问模式优化

```cpp
// 优化1: SoA (Structure of Arrays) vs AoS (Array of Structures)
// AoS - 缓存不友好
struct PointAoS {
    float x, y, z, w;
};

void process_aos(PointAoS* points, size_t n) {
    for (size_t i = 0; i < n; i++) {
        points[i].x *= 2;  // 每次只修改x，但加载了整个struct（浪费缓存行）
    }
}

// SoA - 缓存友好，SIMD优化
struct PointSoA {
    float* x;
    float* y;
    float* z;
    float* w;

    PointSoA(size_t n) {
        posix_memalign((void**)&x, 32, n * sizeof(float));
        posix_memalign((void**)&y, 32, n * sizeof(float));
        posix_memalign((void**)&z, 32, n * sizeof(float));
        posix_memalign((void**)&w, 32, n * sizeof(float));
    }

    ~PointSoA() {
        free(x); free(y); free(z); free(w);
    }
};

void process_soa_simd(PointSoA& points, size_t n) {
    size_t i = 0;

    // 使用SIMD并行处理8个点的x坐标
    while (i + 8 <= n) {
        __m256 vx = _mm256_loadu_ps(points.x + i);
        __m256 v2 = _mm256_set1_ps(2.0f);
        __m256 vr = _mm256_mul_ps(vx, v2);
        _mm256_storeu_ps(points.x + i, vr);
        i += 8;
    }
}

// 优化2: 缓存行对齐的批量处理
constexpr size_t CACHE_LINE_SIZE = 64;

// 填充结构体以避免false sharing
struct alignas(CACHE_LINE_SIZE) AlignedResult {
    float distances[8];
    int64_t ids[8];
    char padding[CACHE_LINE_SIZE - 8 * (sizeof(float) + sizeof(int64_t))];
};

// 优化3: 分块处理提高缓存重用
void blocked_matrix_multiply(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K) {

    // 分块大小适合L1缓存（~32KB）
    constexpr size_t BLOCK_SIZE = 64;

    for (size_t i = 0; i < M; i += BLOCK_SIZE) {
        for (size_t j = 0; j < N; j += BLOCK_SIZE) {
            for (size_t k = 0; k < K; k += BLOCK_SIZE) {
                // 处理一个块
                size_t i_end = std::min(i + BLOCK_SIZE, M);
                size_t j_end = std::min(j + BLOCK_SIZE, N);
                size_t k_end = std::min(k + BLOCK_SIZE, K);

                for (size_t ii = i; ii < i_end; ii++) {
                    for (size_t kk = k; kk < k_end; kk++) {
                        float a_ik = A[ii * K + kk];
                        for (size_t jj = j; jj < j_end; jj++) {
                            C[ii * N + jj] += a_ik * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}
```

### 12.5 高级预取策略

```cpp
// 软件预取的深度优化
void advanced_prefetch_example(const float* __restrict x,
                               const float* __restrict y,
                               float* __restrict result,
                               size_t d) {
    constexpr size_t PREFETCH_DISTANCE = 16;  // 预取距离（缓存行数）

    size_t i = 0;
    __m256 sum = _mm256_setzero_ps();

    // 主循环
    while (i + 8 <= d) {
        // 预取未来的数据
        // _MM_HINT_T0: 预取到L1（数据很快使用）
        // _MM_HINT_T1: 预取到L2（数据稍后使用）
        // _MM_HINT_T2: 预取到L3（数据很久后使用）
        // _MM_HINT_NTA: 非时临预取（不替换缓存）

        if (i + PREFETCH_DISTANCE * 8 < d) {
            // 预取x到L2缓存
            _mm_prefetch((const char*)(x + i + PREFETCH_DISTANCE * 8), _MM_HINT_T1);
            // 预取y到L2缓存
            _mm_prefetch((const char*)(y + i + PREFETCH_DISTANCE * 8), _MM_HINT_T1);
        }

        // 当前迭代
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        sum = _mm256_fmadd_ps(vx, vy, sum);

        i += 8;
    }

    // 处理剩余...
}

// 预取与预计算结合
void prefetch_with_computation(const float* data, size_t n) {
    constexpr size_t STRIPE = 4;  // 同时处理4条流

    for (size_t i = 0; i < n; i += STRIPE * 8) {
        // 预取4条流
        for (size_t s = 0; s < STRIPE; s++) {
            if (i + s * 8 + 32 < n) {
                _mm_prefetch((const char*)(data + i + s * 8 + 32), _MM_HINT_T0);
            }
        }

        // 计算4条流
        for (size_t s = 0; s < STRIPE; s++) {
            if (i + s * 8 < n) {
                __m256 v = _mm256_loadu_ps(data + i + s * 8);
                // ... 计算
            }
        }
    }
}
```

### 12.6 AVX-512高级特性

```cpp
// AVX-512提供的新特性

// 1. 掩码寄存器 - 条件计算更灵活
void avx512_masked_operations(const float* x, const float* y, float* result, size_t d) {
    size_t i = 0;
    __mmask16 mask = 0xFFFF;  // 16位掩码

    while (i + 16 <= d) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);

        // 仅对正数元素执行操作
        __mmask16 positive_mask = _mm512_cmp_ps_mask(vx, _mm512_setzero_ps(), _CMP_GT_OQ);

        // 掩码乘法：只计算positive_mask为1的位置
        __m512 vr = _mm512_mul_ps(vx, vy);

        // 掩码存储
        _mm512_mask_storeu_ps(result + i, positive_mask, vr);

        i += 16;
    }
}

// 2. gather/scatter - 非连续内存访问
void avx512_gather_example(const float* data, const int* indices, float* result, size_t n) {
    size_t i = 0;

    while (i + 16 <= n) {
        // 加载16个索引
        __m512i vindices = _mm512_loadu_si512(indices + i);

        // 根据索引收集数据
        __m512 vdata = _mm512_i32gather_ps(vindices, data, 4);

        // 处理数据
        __m512 vprocessed = _mm512_mul_ps(vdata, _mm512_set1_ps(2.0f));

        // 根据索引分散存储
        __m512i vresult_indices = vindices;
        _mm512_i32scatter_ps(result, vresult_indices, vprocessed, 4);

        i += 16;
    }
}

// 3. vpermil - 置换操作
void avx512_permute_example(const float* data, float* result, size_t d) {
    // 矩阵转置优化
    for (size_t i = 0; i + 16 <= d; i += 16) {
        __m512 r0 = _mm512_loadu_ps(data + i);
        __m512 r1 = _mm512_loadu_ps(data + i + 16);
        __m512 r2 = _mm512_loadu_ps(data + i + 32);
        __m512 r3 = _mm512_loadu_ps(data + i + 48);

        // 转置4x16矩阵
        __m512 t0 = _mm512_shuffle_f32x4(r0, r1, _MM_SHUFFLE(1, 0, 1, 0));
        __m512 t1 = _mm512_shuffle_f32x4(r2, r3, _MM_SHUFFLE(1, 0, 1, 0));
        __m512 t2 = _mm512_shuffle_f32x4(r0, r1, _MM_SHUFFLE(3, 2, 3, 2));
        __m512 t3 = _mm512_shuffle_f32x4(r2, r3, _MM_SHUFFLE(3, 2, 3, 2));

        __m512 u0 = _mm512_shuffle_f32x4(t0, t1, _MM_SHUFFLE(2, 0, 2, 0));
        __m512 u1 = _mm512_shuffle_f32x4(t0, t1, _MM_SHUFFLE(3, 1, 3, 1));
        __m512 u2 = _mm512_shuffle_f32x4(t2, t3, _MM_SHUFFLE(2, 0, 2, 0));
        __m512 u3 = _mm512_shuffle_f32x4(t2, t3, _MM_SHUFFLE(3, 1, 3, 1));

        _mm512_storeu_ps(result + i + 0, u0);
        _mm512_storeu_ps(result + i + 16, u1);
        _mm512_storeu_ps(result + i + 32, u2);
        _mm512_storeu_ps(result + i + 48, u3);
    }
}

// 4. 减少指令计数 - 链式FMA
__m512 avx512_chained_fma(const float* x, const float* y, const float* z, size_t d) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;

    while (i + 16 <= d) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);
        __m512 vz = _mm512_loadu_ps(z + i);

        // x*y+z*vz 链式FMA，减少中间结果
        sum = _mm512_fmadd_ps(vx, vy, sum);
        sum = _mm512_fmadd_ps(vz, sum, sum);  // 复杂表达式优化

        i += 16;
    }

    return sum;
}

// 5. 自动向量化 hints
#pragma omp declare simd
#pragma omp simd
float vectorizable_function(float x, float y) {
    return x * x + y * y;
}

// 编译器可以自动生成SIMD代码
```

### 12.7 ARM SVE (Scalable Vector Extension)

```cpp
// ARM SVE - 可变长度向量
#ifdef __ARM_FEATURE_SVE

#include <arm_sve.h>

// SVE的优势：代码与向量长度无关
float fvec_inner_product_sve(const float* x, const float* y, size_t d) {
    float sum = 0.0f;
    size_t i = 0;

    // SVE使用谓词寄存器处理剩余元素
    svfloat32_t sum_vec = svdup_n_f32(0.0f);

    while (i + svcntw() <= d) {
        svfloat32_t vx = svld1_f32(svptrue_b32(), x + i);
        svfloat32_t vy = svld1_f32(svptrue_b32(), y + i);

        sum_vec = svmla_f32(sum_vec, vx, vy);  // sum += x * y
        i += svcntw();  // 获取当前向量长度
    }

    // 处理剩余元素
    if (i < d) {
        // 创建谓词：有效元素位置为1
        svbool_t pg = svwhilelt_b32_s32(i, d);

        svfloat32_t vx = svld1_f32(pg, x + i);
        svfloat32_t vy = svld1_f32(pg, y + i);

        sum_vec = svmla_f32(sum_vec, vx, vy);
    }

    // 水平求和
    return svaddv_f32(svptrue_b32(), sum_vec);
}

// SVE的优势：同一套代码适应不同SVE实现
// - SVE-128: 4个float
// - SVE-256: 8个float
// - SVE-512: 16个float
// 代码无需修改

#endif
```

### 12.8 性能测量与基准测试

```cpp
// RDTSC精确计时
inline uint64_t rdtsc() {
    uint32_t lo, hi;
    __asm__ __volatile__ (
        "rdtsc" : "=a"(lo), "=d"(hi)
    );
    return ((uint64_t)hi << 32) | lo;
}

// 性能测试框架
struct BenchmarkResult {
    uint64_t cycles;
    uint64_t ns;
    double gflops;
    double bytes_per_ns;
};

BenchmarkResult benchmark_inner_product(
    float (*func)(const float*, const float*, size_t),
    const float* x,
    const float* y,
    size_t d,
    size_t iterations = 1000000) {

    BenchmarkResult result;

    // 预热
    func(x, y, d);

    // RDTSC计时
    uint64_t start_cycles = rdtsc();
    auto start_ns = std::chrono::high_resolution_clock::now();

    float sum = 0;
    for (size_t i = 0; i < iterations; i++) {
        sum += func(x, y, d);
    }

    uint64_t end_cycles = rdtsc();
    auto end_ns = std::chrono::high_resolution_clock::now();

    // 防止编译器优化掉计算
    if (sum == 0.0f) printf("%f\n", sum);

    result.cycles = end_cycles - start_cycles;
    result.ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_ns - start_ns).count();

    // 计算性能指标
    // 每次内积: 2*d个FLOP (d次乘法 + d次加法)
    uint64_t total_flops = iterations * 2 * d;
    result.gflops = (double)total_flops / result.ns;

    // 内存带宽
    uint64_t bytes_read = iterations * 2 * d * sizeof(float);
    result.bytes_per_ns = (double)bytes_read / result.ns;

    return result;
}

// 打印性能报告
void print_benchmark_report(const char* name, const BenchmarkResult& result) {
    printf("%s:\n", name);
    printf("  Cycles: %lu\n", result.cycles);
    printf("  Time: %.3f ms\n", result.ns / 1e6);
    printf("  Throughput: %.2f GFLOPS\n", result.gflops);
    printf("  Memory bandwidth: %.2f GB/s\n", result.bytes_per_ns);
    printf("  Cycles per element: %.2f\n",
           (double)result.cycles / result.ns);
}
```

### 12.9 编译器优化选项

```cpp
// GCC/Clang优化选项

// 基础优化
// -O1: 基础优化
// -O2: 标准优化（推荐）
// -O3: 激进优化（包括循环展开、向量化等）
// -Ofast: -O3 + 不严格遵守标准（可能改变浮点语义）

// 特定架构优化
// -march=native: 生成当前CPU的最优代码
// -mtune=native: 调优为当前CPU
// -mavx2: 启用AVX2
// -mavx512f: 启用AVX-512基础指令集
// -mavx512vl: 启用AVX-512向量长度
// -mfma: 启用FMA指令

// 示例编译命令
// g++ -O3 -march=native -mavx2 -mfma -fopenmp program.cpp

// Intel编译器
// icc -O3 -xHOST -qopenmp program.cpp

// 查看生成的汇编
// objdump -d program.o | grep -A30 "function_name"

// 编译器内联汇编提示
void __attribute__((always_inline)) force_inline_function() {
    // 强制内联
}

// 禁止内联
void __attribute__((noinline)) noinline_function() {
    // 禁止内联
}

// 热点函数优化
void __attribute__((hot)) hotspot_function() {
    // 告诉编译器这是热点函数
}

// 冷函数优化
void __attribute__((cold)) cold_function() {
    // 告诉编译器这是冷函数（很少执行）
}

// 纯函数优化（无副作用）
float __attribute__((const)) pure_function(float x) {
    return x * x;
}
```

---

## 13. 底层内存模型优化

### 13.1 False Sharing 问题与解决

```cpp
// false sharing 导致的性能问题
struct BadCounter {
    alignas(64) int64_t counter1;  // 虽然对齐，但counter2可能在同一缓存行
    int64_t counter2;
};

struct GoodCounter {
    alignas(64) int64_t counter1;  // 确保在不同缓存行
    alignas(64) char padding1[64 - sizeof(int64_t)];
    alignas(64) int64_t counter2;
    alignas(64) char padding2[64 - sizeof(int64_t)];
};

// 验证false sharing的影响
void test_false_sharing() {
    constexpr int ITERATIONS = 10000000;

    BadCounter bad;
    good.counter1 = 0;
    good.counter2 = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // 两个线程更新不同变量（但在同一缓存行）
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int i = 0; i < ITERATIONS; i++) {
                bad.counter1++;
            }
        }
        #pragma omp section
        {
            for (int i = 0; i < ITERATIONS; i++) {
                bad.counter2++;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto bad_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();

    // GoodCounter版本快得多（缓存行不冲突）
}
```

### 13.2 NUMA 优化

```cpp
// NUMA (Non-Uniform Memory Access) 优化
#ifdef __linux__

#include <numa.h>

class NUMAAllocator {
public:
    void* allocate(size_t size, int node = -1) {
        // node = -1: 本地节点
        // node >= 0: 指定NUMA节点

        void* ptr = numa_alloc_onnode(size, node);
        if (!ptr) {
            ptr = numa_alloc(size);  // fallback
        }
        return ptr;
    }

    void deallocate(void* ptr, size_t size) {
        numa_free(ptr, size);
    }
};

// NUMA亲和性优化
void set_numa_affinity() {
    // 绑定当前线程到特定NUMA节点
    numa_set_preferred(numa_max_node() / 2);  // 选择中间节点

    // 绑定内存分配
    struct bitmask* mask = numa_allocate_cpumask();
    numa_bitmask_setbit(mask, 0);  // CPU 0
    numa_bind(mask);
    numa_free_cpumask(mask);
}

#endif
```

### 13.3 缓存预取策略深度分析

```cpp
// 硬件预取器行为分析
class PrefetchAnalysis {
public:
    // 检测硬件预取器是否工作
    static void detect_hardware_prefetcher() {
        constexpr size_t ARRAY_SIZE = 1024 * 1024;  // 4MB
        constexpr size_t STRIDE = 64;  // 一个缓存行

        alignas(64) float array[ARRAY_SIZE];

        // 顺序访问（硬件预取器应该有效）
        auto start = std::chrono::high_resolution_clock::now();

        volatile float sum = 0;
        for (size_t i = 0; i < ARRAY_SIZE; i += STRIDE / sizeof(float)) {
            sum += array[i];
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto sequential_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count();

        // 随机访问（硬件预取器无效）
        start = std::chrono::high_resolution_clock::now();

        sum = 0;
        for (size_t i = 0; i < ARRAY_SIZE; i += STRIDE / sizeof(float)) {
            size_t idx = ((i * 11400714819323198549ULL) % ARRAY_SIZE);
            sum += array[idx];
        }

        end = std::chrono::high_resolution_clock::now();
        auto random_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count();

        printf("Sequential access: %ld us\n", sequential_time);
        printf("Random access: %ld us\n", random_time);
        printf("Speedup: %.2fx\n", (double)random_time / sequential_time);
    }

    // 优化预取距离
    template<typename Func>
    static size_t find_optimal_prefetch_distance(Func func, size_t max_distance = 64) {
        alignas(64) float data[1024 * 1024];

        size_t best_distance = 0;
        double best_time = std::numeric_limits<double>::infinity();

        for (size_t dist = 0; dist <= max_distance; dist += 8) {
            auto start = std::chrono::high_resolution_clock::now();

            volatile float sum = 0;
            for (size_t iter = 0; iter < 100; iter++) {
                for (size_t i = 0; i < 1024 * 1024; i += 256 / sizeof(float)) {
                    if (i + dist * 8 < 1024 * 1024) {
                        _mm_prefetch((const char*)(data + i + dist * 8), _MM_HINT_T0);
                    }
                    sum += data[i];
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            double time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count() / 1e9;

            if (time < best_time) {
                best_time = time;
                best_distance = dist;
            }

            printf("Prefetch distance %zu: %.6f s\n", dist, time);
        }

        printf("Optimal distance: %zu\n", best_distance);
        return best_distance;
    }
};
```

### 13.4 写合并缓冲区优化

```cpp
// 写合并（Write Combining）优化
class WriteCombiningBuffer {
    alignas(64) uint8_t buffer[64];  // 一个缓存行

public:
    WriteCombiningBuffer() {
        memset(buffer, 0, sizeof(buffer));
    }

    // 批量写入减少写事务
    void batch_write(const uint8_t* data, size_t offset, size_t length) {
        // 先写入缓冲区
        memcpy(buffer + offset, data, length);

        // 缓冲区满时一次性刷新
        if (offset + length >= 64) {
            flush();
        }
    }

    void flush() {
        // 一次性写回内存（触发一个写事务）
        // 实际实现需要mmap或特定硬件指令
        _mm_clwb(buffer);  // Cache Line Write Back
        _mm_sfence();      // Store Fence
    }
};

// 非时临写入优化
void nontemporal_write_example(float* dest, const float* src, size_t n) {
    size_t i = 0;

    // 使用非时临存储绕过缓存
    // 适用于：大数据量写入，写入后不会立即读取
    while (i + 8 <= n) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_stream_si256((__m256i*)(dest + i), _mm256_castps_si256(v));
        i += 8;
    }

    // 确保所有流存储完成
    _mm_sfence();
}
```

---

## 14. 高级SIMD技巧

### 14.1 水平操作优化

```cpp
// 水平求和的优化实现

// 方法1: 使用hadd（慢）
float slow_hsum(__m256 v) {
    __m256 sum = _mm256_hadd_ps(v, v);
    sum = _mm256_hadd_ps(sum, sum);
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    return tmp[0] + tmp[4];
}

// 方法2: 使用shuffle和add（快）
float fast_hsum(__m256 v) {
    // 提取低128位和高128位
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);

    // 相加
    __m128 sum = _mm_add_ps(vlow, vhigh);

    // 水平求和128位
    __m128 shuf = _mm_movehdup_ps(sum);  // 复制奇数位置到偶数位置
    __m128 sums = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sums);    // 交换高低64位
    sums = _mm_add_ss(sums, shuf);

    return _mm_cvtss_f32(sums);
}

// 方法3: AVX-512专用（最快）
#ifdef __AVX512F__
float fast_hsum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
#endif

// 性能对比：
// slow_hsum:    ~10 cycles
// fast_hsum:    ~5 cycles
// avx512_hsum:  ~3 cycles
```

### 14.2 查找表优化

```cpp
// 使用SIMD查找表加速操作

// 8位查找表（PQ编码常用）
class SIMDLookupTable {
    uint8_t table[256];

public:
    SIMDLookupTable(const uint8_t* init_data) {
        memcpy(table, init_data, 256);
    }

    // 批量查找：输入32个8位值，输出32个查找结果
    void lookup_batch_32(const uint8_t* indices, uint8_t* output) {
        // 每次处理32个字节（256位）
        __m256i idx = _mm256_loadu_si256((__m256i*)indices);

        // 使用vpshufb查找表
        // 需要将表复制到一个向量中...
        __m256i table_low = _mm256_loadu_si256((__m256i*)table);
        __m256i table_high = _mm256_loadu_si256((__m256i*)(table + 32));

        __m256i result = _mm256_shuffle_epi8(table_low, idx);
        _mm256_storeu_si256((__m256i*)output, result);
    }
};

// Hamming距离的SIMD查找优化
inline int popcount_avx2(__m256i v) {
    // 计算两个256位向量的汉明距离
    // 使用vpshufb和查找表

    // 查找表：每个字节的popcount
    static const uint8_t popcount_table[256] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        // ... 完整表256个元素
    };

    // 每个字节的popcount求和
    __m256i table_vec = _mm256_loadu_si256((__m256i*)popcount_table);
    __m256i low = _mm256_and_si256(v, _mm256_set1_epi8(0x0F));
    __m256i high = _mm256_and_si256(_mm256_srli_epi16(v, 4), _mm256_set1_epi8(0x0F));

    __m256i popcount_low = _mm256_shuffle_epi8(table_vec, low);
    __m256i popcount_high = _mm256_shuffle_epi8(table_vec, high);

    __m256i total = _mm256_add_epi8(popcount_low, popcount_high);

    // 水平求和...
    int result;
    _mm256_storeu_si256((__m256i*)&result,
        _mm256_hadd_epi16(_mm256_hadd_epi8(total, total),
                         _mm256_setzero_si256()));

    return result;
}
```

### 14.3 向量条件选择

```cpp
// 使用SIMD实现复杂的条件逻辑

// C风格代码（有分支）
void scalar_conditional(float* a, float* b, float* result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] > 0) {
            result[i] = a[i] * b[i];
        } else if (a[i] < 0) {
            result[i] = a[i] + b[i];
        } else {
            result[i] = b[i];
        }
    }
}

// SIMD无分支实现
void simd_conditional(float* a, float* b, float* result, size_t n) {
    size_t i = 0;
    const __m256 zero = _mm256_setzero_ps();

    while (i + 8 <= n) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);

        // 创建条件掩码
        __m256 gt_mask = _mm256_cmp_ps(va, zero, _CMP_GT_OQ);   // a > 0
        __m256 lt_mask = _mm256_cmp_ps(va, zero, _CMP_LT_OQ);   // a < 0
        __m256 eq_mask = _mm256_cmp_ps(va, zero, _CMP_EQ_OQ);   // a == 0

        // 计算三个分支的结果
        __m256 mul_result = _mm256_mul_ps(va, vb);        // a * b
        __m256 add_result = _mm256_add_ps(va, vb);        // a + b
        __m256 id_result = vb;                           // b

        // 使用掩码选择结果
        __m256 result1 = _mm256_blendv_ps(id_result, mul_result, gt_mask);
        __m256 result2 = _mm256_blendv_ps(result1, add_result, lt_mask);

        _mm256_storeu_ps(result + i, result2);
        i += 8;
    }
}
```

---

## 15. SIMD代码调试技巧

### 15.1 常见陷阱与调试

```cpp
// 陷阱1: 未对齐访问导致段错误
void alignment_trap() {
    float* data = new float[16];  // 未对齐分配

    // 这可能崩溃（需要32字节对齐）
    __m256 v = _mm256_load_ps(data);  // 错误！

    // 正确做法
    alignas(32) float data_aligned[16];
    __m256 v2 = _mm256_load_ps(data_aligned);  // OK

    // 或使用未对齐版本
    __m256 v3 = _mm256_loadu_ps(data);  // OK
}

// 陷阱2: 忘记处理剩余元素
void remainder_trap(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    while (i + 8 <= n) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
        i += 8;
    }
    // 忘记处理剩余元素！
    // 应该添加：
    // for (; i < n; i++) c[i] = a[i] + b[i];
}

// 陷阱3: NaN传播
void nan_propagation_trap() {
    float a[8] = {1, 2, 3, 4, 5, 6, 7, NAN};
    float b[8] = {0};

    __m256 va = _mm256_loadu_ps(a);
    __m256 vb = _mm256_loadu_ps(b);

    // NaN会传播到整个计算
    __m256 vc = _mm256_add_ps(va, vb);

    // 处理NaN
    __m256 is_nan = _mm256_cmp_ps(vc, vc, _CMP_NEQ_UQ);
    if (_mm256_movemask_ps(is_nan)) {
        // 处理NaN情况
    }
}
```

### 15_2. SIMD代码验证

```cpp
// 验证SIMD实现与标量实现的一致性
template<typename SimdFunc, typename ScalarFunc>
bool verify_simd_implementation(
    SimdFunc simd_func,
    ScalarFunc scalar_func,
    const float* test_data,
    size_t d,
    size_t num_tests = 1000) {

    for (size_t test = 0; test < num_tests; test++) {
        // 准备测试数据
        const float* x = test_data + test * 2 * d;
        const float* y = x + d;

        // 标量结果
        float scalar_result = scalar_func(x, y, d);

        // SIMD结果
        float simd_result = simd_func(x, y, d);

        // 验证
        float diff = std::abs(scalar_result - simd_result);
        float max_val = std::max(std::abs(scalar_result), std::abs(simd_result));

        // 相对误差检查
        if (max_val > 1e-6f && diff / max_val > 1e-5f) {
            printf("Mismatch at test %zu: scalar=%.8f, simd=%.8f\n",
                   test, scalar_result, simd_result);
            return false;
        }
    }

    printf("All %zu tests passed!\n", num_tests);
    return true;
}

// 使用示例
void test_inner_product() {
    // 生成测试数据
    constexpr size_t TEST_SIZE = 1000 * 2 * 128;  // 1000对128维向量
    float* test_data = new float[TEST_SIZE];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < TEST_SIZE; i++) {
        test_data[i] = dis(gen);
    }

    // 验证
    bool passed = verify_simd_implementation(
        fvec_inner_product_avx2,
        fvec_inner_product_scalar,
        test_data,
        128
    );

    if (passed) {
        printf("AVX2 inner product implementation is correct!\n");
    }

    delete[] test_data;
}
```

---

## 16. CPU微架构级优化

### 16.1 指令延迟与吞吐量分析

```cpp
// 理解CPU流水线特性对于编写高效SIMD代码至关重要
// 不同微架构有不同特性

// Intel Skylake/Zen2 指令特性
// FMA: 延迟4周期，吞吐量0.5 (每周期2条)
// 加载: 延迟4-5周期，吞吐量0.5 (每周期2条，L1)
// 存储: 延迟1周期，吞吐量1 (每周期1条)

// 依赖链分析
void dependency_chain_analysis() {
    // 差：连续依赖链
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < 16; i++) {
        __m256 v = _mm256_loadu_ps(data + i * 8);
        sum = _mm256_fmadd_ps(v, v, sum);  // 每次迭代依赖前次结果
    }
    // 16次FMA * 4周期 = 64周期（最少）

    // 好：打破依赖链
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    for (int i = 0; i < 16; i += 4) {
        __m256 v0 = _mm256_loadu_ps(data + (i+0) * 8);
        __m256 v1 = _mm256_loadu_ps(data + (i+1) * 8);
        __m256 v2 = _mm256_loadu_ps(data + (i+2) * 8);
        __m256 v3 = _mm256_loadu_ps(data + (i+3) * 8);

        sum0 = _mm256_fmadd_ps(v0, v0, sum0);
        sum1 = _mm256_fmadd_ps(v1, v1, sum1);
        sum2 = _mm256_fmadd_ps(v2, v2, sum2);
        sum3 = _mm256_fmadd_ps(v3, v3, sum3);
    }
    // 4条独立链，每条4次FMA，可并行执行
    // ~4周期 * 4 = 16周期（4倍加速）
}

// 端口压力分析（Skylake: 2个FMA单元，2个加载单元）
void port_pressure_optimized(const float* a, const float* b, size_t n) {
    // 目标：均衡使用各个执行端口
    // Skylake端口：
    // p0: ALU, FMA
    // p1: ALU, FMA, 向量shuffle
    // p2/3: 加载
    // p4: 存储
    // p5: ALU, 向量shuffle
    // p6: 分支

    size_t i = 0;
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    while (i + 32 <= n) {
        // 8条加载（使用p2/p3）
        __m256 a0 = _mm256_loadu_ps(a + i + 0);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 a2 = _mm256_loadu_ps(a + i + 16);
        __m256 a3 = _mm256_loadu_ps(a + i + 24);
        __m256 b0 = _mm256_loadu_ps(b + i + 0);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        __m256 b2 = _mm256_loadu_ps(b + i + 16);
        __m256 b3 = _mm256_loadu_ps(b + i + 24);

        // 8条FMA（使用p0/p1）
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);
        sum3 = _mm256_fmadd_ps(a3, b3, sum3);

        i += 32;
    }
}
```

### 16.2 微Op缓存优化

```cpp
// Intel微Op缓存（DSB）解码优化
// 微Op缓存大小：~1536-2000 uops

// 避免微Op缓存溢出
void micro_op_cache_friendly() {
    // 差：循环体太大，无法放入微Op缓存
    // 每次迭代需要重新解码
    for (int i = 0; i < n; i++) {
        // ... 100条指令的复杂计算
    }

    // 好：保持循环体紧凑
    for (int i = 0; i < n; i += 8) {
        // ... 10-20条指令
    }
}

// 循环体大小优化示例
float compact_inner_product(const float* x, const float* y, size_t d) {
    // 目标：< 50 uops，可完全放入微Op缓存
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    size_t i = 0;
    while (i + 32 <= d) {
        // 8次加载 (8 uops)
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 y0 = _mm256_loadu_ps(y + i);
        __m256 y1 = _mm256_loadu_ps(y + i + 8);

        // 4次FMA (4 uops)
        sum0 = _mm256_fmadd_ps(x0, y0, sum0);
        sum1 = _mm256_fmadd_ps(x1, y1, sum1);

        i += 16;
    }

    // 总共约20 uops，完全放入微Op缓存
    return horizontal_add(sum0, sum1);
}
```

### 16.3 寄存器压力优化

```cpp
// x86-64 AVX2: 16个YMM寄存器
// 关键：避免寄存器溢出（spill to stack）

// 寄存器溢出检测
void register_pressure_analysis() {
    // 使用Intel Architecture Code Analyzer (IACA)分析
    // 或通过LLVM-MCA分析

    // 差：超过16个YMM寄存器导致溢出
    __m256 r0, r1, r2, r3, r4, r5, r6, r7;
    __m256 r8, r9, r10, r11, r12, r13, r14, r15;
    __m256 r16, r17;  // 溢出到栈！

    // 好：重用寄存器
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    for (int i = 0; i < n; i += 16) {
        __m256 v0 = _mm256_loadu_ps(data + i);
        __m256 v1 = _mm256_loadu_ps(data + i + 8);

        sum0 = _mm256_fmadd_ps(v0, v0, sum0);
        sum1 = _mm256_fmadd_ps(v1, v1, sum1);
        // v0, v1可以重用，不会溢出
    }
}

// 寄存器分配策略
template<int UNROLL>
void register_efficient_loop(const float* a, const float* b, size_t n) {
    // UNROLL=4: 使用~12个寄存器（安全）
    // UNROLL=8: 使用~20个寄存器（可能溢出）

    __m256 sum[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
        sum[i] = _mm256_setzero_ps();
    }

    size_t i = 0;
    while (i + UNROLL * 8 <= n) {
        for (int u = 0; u < UNROLL; u++) {
            __m256 va = _mm256_loadu_ps(a + i + u * 8);
            __m256 vb = _mm256_loadu_ps(b + i + u * 8);
            sum[u] = _mm256_fmadd_ps(va, vb, sum[u]);
        }
        i += UNROLL * 8;
    }
}
```

### 16.4 缓存层次优化

```cpp
// L1缓存: 32KB, 4-8周期延迟
// L2缓存: 256KB-1MB, 12-20周期延迟
// L3缓存: 8-32MB, 40-80周期延迟
// 内存: ~100-200ns (~300-600周期)

// 缓存块优化
void cache_blocking_optimization(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K) {

    // 计算合适的块大小
    // L1: 32KB / 3个矩阵 = ~10KB per matrix
    // ~1024 floats = 32x32 单精度矩阵
    constexpr size_t L1_BLOCK = 32;

    // L2: 256KB / 3个矩阵 = ~85KB per matrix
    // ~22000 floats = 128x170 单精度矩阵
    constexpr size_t L2_BLOCK = 128;

    for (size_t i = 0; i < M; i += L2_BLOCK) {
        for (size_t j = 0; j < N; j += L2_BLOCK) {
            for (size_t k = 0; k < K; k += L2_BLOCK) {
                // L2块循环
                size_t i_end = std::min(i + L2_BLOCK, M);
                size_t j_end = std::min(j + L2_BLOCK, N);
                size_t k_end = std::min(k + L2_BLOCK, K);

                for (size_t ii = i; ii < i_end; ii += L1_BLOCK) {
                    for (size_t jj = j; jj < j_end; jj += L1_BLOCK) {
                        for (size_t kk = k; kk < k_end; kk += L1_BLOCK) {
                            // L1块循环 - 最内层，适合L1缓存
                            size_t ii_end = std::min(ii + L1_BLOCK, i_end);
                            size_t jj_end = std::min(jj + L1_BLOCK, j_end);
                            size_t kk_end = std::min(kk + L1_BLOCK, k_end);

                            for (size_t iii = ii; iii < ii_end; iii++) {
                                for (size_t kkk = kk; kkk < kk_end; kkk++) {
                                    float a_ik = A[iii * K + kkk];
                                    for (size_t jjj = jj; jjj < jj_end; jjj++) {
                                        C[iii * N + jjj] += a_ik * B[kkk * N + jjj];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// 数据预取与缓存块结合
void tiled_with_prefetch(const float* A, const float* B, float* C,
                        size_t M, size_t N, size_t K) {
    constexpr size_t TILE = 64;
    constexpr size_t PREFETCH_DIST = 4;

    for (size_t i = 0; i < M; i += TILE) {
        for (size_t k = 0; k < K; k += TILE) {
            // 预取B的下一个tile
            if (k + PREFETCH_DIST * TILE < K) {
                const float* prefetch_ptr = B + (k + PREFETCH_DIST * TILE) * N;
                for (size_t row = 0; row < TILE && i + row < M; row++) {
                    _mm_prefetch((const char*)(prefetch_ptr + row * N), _MM_HINT_T0);
                }
            }

            // 处理当前tile
            size_t i_end = std::min(i + TILE, M);
            size_t k_end = std::min(k + TILE, K);

            for (size_t ii = i; ii < i_end; ii++) {
                for (size_t kk = k; kk < k_end; kk++) {
                    float a_ik = A[ii * K + kk];
                    __m256 va = _mm256_set1_ps(a_ik);

                    size_t j = 0;
                    for (; j + 8 <= N; j += 8) {
                        __m256 vb = _mm256_loadu_ps(B + kk * N + j);
                        __m256 vc = _mm256_loadu_ps(C + ii * N + j);
                        vc = _mm256_fmadd_ps(va, vb, vc);
                        _mm256_storeu_ps(C + ii * N + j, vc);
                    }
                }
            }
        }
    }
}
```

### 16.5 分支预测器优化

```cpp
// 现代CPU分支预测器分析
// Skylake: 融合分支预测器 + BTB + RSB
// 预测失败代价：~15-20周期

// 分支友好的数据结构
struct branch_friendly_layout {
    // 数据排序：使分支可预测
    void sort_by_predicate(const float* data, bool* predicates, size_t n) {
        // 将true和false分开存储
        std::vector<float> true_values;
        std::vector<float> false_values;

        true_values.reserve(n);
        false_values.reserve(n);

        for (size_t i = 0; i < n; i++) {
            if (predicates[i]) {
                true_values.push_back(data[i]);
            } else {
                false_values.push_back(data[i]);
            }
        }

        // 处理true值（分支总是预测为taken）
        process_batch(true_values.data(), true_values.size());

        // 处理false值（分支总是预测为not taken）
        process_batch(false_values.data(), false_values.size());
    }
};

// 间接分支优化（使用VTUNE分析）
void indirect_branch_optimization() {
    // 虚函数调用是间接分支
    // 优化：使用函数指针表 + 直接跳转

    // 差：虚函数调用
    struct Interface {
        virtual float compute(const float*) = 0;
    };

    // 好：显式函数表
    typedef float (*ComputeFunc)(const float*);

    struct DispatchTable {
        ComputeFunc funcs[16];

        float compute(int type, const float* data) {
            // 编译器可能优化为直接跳转表
            return funcs[type](data);
        }
    };
}
```

### 16.6 汇编级优化分析

```cpp
// 查看编译器生成的汇编
// g++ -O3 -march=native -S -masm=intel code.cpp

// 手动汇编优化的内积函数
extern "C" float optimized_inner_product_assembly(
    const float* x,
    const float* y,
    size_t d) {

    float result = 0.0f;
    size_t i = 0;

    // AVX2汇编实现 - 完全控制指令调度
    asm volatile (
        // 初始化
        "vxorps %%ymm0, %%ymm0, %%ymm0 \n\t"  // sum0 = 0
        "vxorps %%ymm1, %%ymm1, %%ymm1 \n\t"  // sum1 = 0

        // 主循环（假设d是16的倍数）
        "1: \n\t"
        "cmp %3, %0 \n\t"
        "jae 2f \n\t"

        // 加载x (2x 256-bit)
        "vmovups (%2,%0,4), %%ymm2 \n\t"
        "vmovups 32(%2,%0,4), %%ymm3 \n\t"

        // 加载y (2x 256-bit)
        "vmovups (%1,%0,4), %%ymm4 \n\t"
        "vmovups 32(%1,%0,4), %%ymm5 \n\t"

        // FMA: sum = x * y + sum
        "vfmadd231ps %%ymm2, %%ymm4, %%ymm0 \n\t"
        "vfmadd231ps %%ymm3, %%ymm5, %%ymm1 \n\t"

        "add $16, %0 \n\t"
        "jmp 1b \n\t"

        "2: \n\t"
        // 水平求和
        "vextractf128 $1, %%ymm0, %%xmm2 \n\t"
        "vaddps %%xmm2, %%xmm0, %%xmm0 \n\t"
        "vextractf128 $1, %%ymm1, %%xmm3 \n\t"
        "vaddps %%xmm3, %%xmm1, %%xmm1 \n\t"
        "vaddps %%xmm1, %%xmm0, %%xmm0 \n\t"
        "vshufps $0x1B, %%xmm0, %%xmm0, %%xmm0 \n\t"
        "vaddps %%xmm0, %%xmm0, %%xmm0 \n\t"
        "vshufps $0x1, %%xmm0, %%xmm0, %%xmm0 \n\t"
        "vaddss %%xmm0, %%xmm0, %%xmm0 \n\t"

        "vmovss %%xmm0, %4 \n\t"

        : "+r"(i)                          // 输出/输入：循环变量
        : "r"(y), "r"(x), "r"(d), "m"(result)  // 输入
        : "ymm0", "ymm1", "ymm2", "ymm3",
          "ymm4", "ymm5", "xmm0", "xmm1",
          "xmm2", "xmm3", "memory", "cc"   // clobber
    );

    return result;
}

// 编译器内联汇编 - 特定指令序列
inline void specialized_avx2_sequence(const float* src, float* dst) {
    __m256 v0, v1, v2, v3, v4, v5, v6, v7;

    // 精确控制指令顺序
    // 目标：最大化吞吐量，减少停顿

    // 第1组：8次加载（p2/p3端口）
    v0 = _mm256_loadu_ps(src + 0);
    v1 = _mm256_loadu_ps(src + 8);
    v2 = _mm256_loadu_ps(src + 16);
    v3 = _mm256_loadu_ps(src + 24);
    v4 = _mm256_loadu_ps(src + 32);
    v5 = _mm256_loadu_ps(src + 40);
    v6 = _mm256_loadu_ps(src + 48);
    v7 = _mm256_loadu_ps(src + 56);

    // 第2组：4次shuffle（p5/p1端口）
    __m256 t0 = _mm256_permute2f128_ps(v0, v1, 0x20);
    __m256 t1 = _mm256_permute2f128_ps(v2, v3, 0x20);
    __m256 t2 = _mm256_permute2f128_ps(v4, v5, 0x20);
    __m256 t3 = _mm256_permute2f128_ps(v6, v7, 0x20);

    // 第3组：4次FMA（p0/p1端口）
    __m256 r0 = _mm256_fmadd_ps(v0, t0, _mm256_setzero_ps());
    __m256 r1 = _mm256_fmadd_ps(v2, t1, _mm256_setzero_ps());
    __m256 r2 = _mm256_fmadd_ps(v4, t2, _mm256_setzero_ps());
    __m256 r3 = _mm256_fmadd_ps(v6, t3, _mm256_setzero_ps());

    // 第4组：存储（p4端口）
    _mm256_storeu_ps(dst + 0, r0);
    _mm256_storeu_ps(dst + 8, r1);
    _mm256_storeu_ps(dst + 16, r2);
    _mm256_storeu_ps(dst + 24, r3);
}
```

### 16.7 TLB优化

```cpp
// TLB (Translation Lookaside Buffer) 优化
// L1 TLB: ~64-128 entries
// L2 TLB: ~1024-2048 entries

// TLB友好的内存访问
void tlb_optimized_access(const float* data, size_t n) {
    // 小页面访问：每次4KB可能触发TLB miss
    // TLB miss代价：~10-50周期

    // 优化1：使用大页（huge pages, 2MB/1GB）
    // 减少TLB压力

    // 优化2：减少页跨越
    size_t remaining = n;
    size_t offset = 0;

    while (remaining >= 8) {
        // 检查是否接近页边界
        size_t page_mask = 4096 - 1;
        size_t current_page = (size_t)(data + offset) & ~page_mask;
        size_t next_page = ((size_t)(data + offset + 7 * 4) & ~page_mask);

        if (current_page != next_page && remaining < 1000) {
            // 接近页边界且剩余数据少，处理剩余部分
            for (; offset < n; offset++) {
                process_single(data[offset]);
            }
            break;
        }

        // 正常处理
        __m256 v = _mm256_loadu_ps(data + offset);
        process_simd(v);
        offset += 8;
        remaining -= 8;
    }
}
```

### 16.8 性能分析工具使用

```cpp
// 使用Intel VTune Amplifier
// 1. 编译时添加调试信息
// g++ -O3 -march=native -g code.cpp

// 2. VTune命令行分析
// vtune -collect hotspots -result-dir r001hs -- ./a.out
// vtune -collect memory-access -result-dir r001ma -- ./a.out

// 3. 微架构分析
// vtune -collect uarch-exploration -result-dir r001ux -- ./a.out

// 使用perf (Linux)
// perf stat -e cycles,instructions,cache-misses,cache-references ./a.out
// perf record -e cycles ./a.out
// perf report

// 使用LLVM-MCA（机器代码分析器）
// llvm-mca -mcpu=skylake -mattr=+avx2 assembly.s

// 自定义性能计数器读取
#ifdef __x86_64__
inline uint64_t read_perf_counter(int ecx) {
    uint32_t eax, edx;
    __asm__ volatile (
        "rdpmc"
        : "=a"(eax), "=d"(edx)
        : "c"(ecx)
    );
    return ((uint64_t)edx << 32) | eax;
}

// 测量特定事件
void measure_events() {
    // 需要root权限或perf事件许可
    // 事件0xC0: INST_RETIRED.ANY (已退役指令)
    // 事件0x3C: UNHALTED_CORE_CYCLES

    uint64_t instructions_start = read_perf_counter(0xC0);
    uint64_t cycles_start = read_perf_counter(0x3C);

    // 执行测试代码
    benchmark_function();

    uint64_t instructions_end = read_perf_counter(0xC0);
    uint64_t cycles_end = read_perf_counter(0x3C);

    double ipc = (double)(instructions_end - instructions_start) /
                (cycles_end - cycles_start);
    printf("IPC: %.2f\n", ipc);
}
#endif
```

### 16.9 能耗优化

```cpp
// CPU频率与功耗管理
// C-states: CPU空闲状态（C0=运行, C1/C1E/C3/C6/C7=睡眠）
// P-states: 性能状态（频率/电压）
// Turbo Boost: 短时超频

// 能效优化策略
void energy_efficient_computation() {
    // 策略1：最大化IPC（指令/周期）
    // IPC越高，完成任务需要的周期越少，功耗越低

    // 策略2：避免上下文切换
    // 线程迁移导致L3缓存失效，增加功耗
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    // 策略3：批量处理
    // 批量处理减少唤醒/睡眠次数
    constexpr size_t BATCH_SIZE = 1000;

    for (size_t i = 0; i < n; i += BATCH_SIZE) {
        size_t batch_end = std::min(i + BATCH_SIZE, n);

        // 批量SIMD处理
        for (size_t j = i; j < batch_end; j += 32) {
            __m256 v0 = _mm256_loadu_ps(data + j);
            __m256 v1 = _mm256_loadu_ps(data + j + 8);
            __m256 v2 = _mm256_loadu_ps(data + j + 16);
            __m256 v3 = _mm256_loadu_ps(data + j + 24);

            __m256 result = compute_complex(v0, v1, v2, v3);
            _mm256_storeu_ps(output + j, result);
        }

        // 批量后短暂sleep，允许CPU进入更深C-state
        if (i + BATCH_SIZE < n) {
            struct timespec ts = {0, 100000};  // 100us
            nanosleep(&ts, nullptr);
        }
    }
}

// AVX-512频率降频管理
void avx512_frequency_management() {
    // AVX-512可能导致频率降频（AVX频率偏移）
    // 检测并适应

    // 方法1：混合AVX2和AVX-512代码
    if (data_size < threshold) {
        // 小数据：AVX2更高效（无降频）
        avx2_implementation(data, output, size);
    } else {
        // 大数据：AVX-512优势超过降频代价
        avx512_implementation(data, output, size);
    }

    // 方法2：分阶段处理
    // 阶段1：AVX2热身（CPU频率稳定）
    warmup_phase();

    // 阶段2：AVX-512核心计算
    avx512_heavy_computation();

    // 阶段3：AVX2收尾（频率恢复）
    cooldown_phase();
}
```

---

## 17. 特定距离度量的优化

### 17.1 L1距离（Manhattan）SIMD优化

```cpp
// L1距离：sum(|x[i] - y[i]|)
float fvec_L1_avx2(const float* x, const float* y, size_t d) {
    float res = 0.0f;
    size_t i = 0;

    if (d >= 8) {
        __m256 sum = _mm256_setzero_ps();

        while (i + 8 <= d) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = _mm256_loadu_ps(y + i);
            __m256 diff = _mm256_sub_ps(vx, vy);

            // 绝对值：清除符号位
            __m256 abs_diff = _mm256_andnot_ps(
                _mm256_set1_ps(-0.0f),  // 符号位掩码
                diff
            );

            sum = _mm256_add_ps(sum, abs_diff);
            i += 8;
        }

        // 水平求和
        res = horizontal_sum_avx2(sum);
    }

    for (; i < d; i++) {
        res += std::abs(x[i] - y[i]);
    }

    return res;
}

// AVX-512优化：专用绝对值指令
#ifdef __AVX512F__
float fvec_L1_avx512(const float* x, const float* y, size_t d) {
    float res = 0.0f;
    size_t i = 0;

    if (d >= 16) {
        __m512 sum = _mm512_setzero_ps();

        while (i + 16 <= d) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vy = _mm512_loadu_ps(y + i);
            __m512 diff = _mm512_sub_ps(vx, vy);

            // AVX-512专用绝对值指令
            __m512 abs_diff = _mm512_abs_ps(diff);

            sum = _mm512_add_ps(sum, abs_diff);
            i += 16;
        }

        res = _mm512_reduce_add_ps(sum);
    }

    for (; i < d; i++) {
        res += std::abs(x[i] - y[i]);
    }

    return res;
}
#endif
```

### 17.2 Linf距离（Chebyshev）SIMD优化

```cpp
// L∞距离：max(|x[i] - y[i]|)
float fvec_Linf_avx2(const float* x, const float* y, size_t d) {
    __m256 max_diff = _mm256_setzero_ps();
    size_t i = 0;

    while (i + 8 <= d) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 diff = _mm256_sub_ps(vx, vy);

        // 绝对值
        __m256 abs_diff = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff);

        // 逐元素最大值
        max_diff = _mm256_max_ps(max_diff, abs_diff);
        i += 8;
    }

    // 从SIMD寄存器提取最大值
    alignas(32) float temp[8];
    _mm256_storeu_ps(temp, max_diff);

    float result = temp[0];
    for (int j = 1; j < 8; j++) {
        result = std::max(result, temp[j]);
    }

    // 处理剩余元素
    for (; i < d; i++) {
        result = std::max(result, std::abs(x[i] - y[i]));
    }

    return result;
}
```

### 17.3 余弦相似度SIMD优化

```cpp
// 余弦相似度：<x,y> / (||x|| * ||y||)
float fvec_cosine_avx2(const float* x, const float* y, size_t d) {
    float ip = 0.0f;   // 内积
    float norm_x = 0.0f; // x范数
    float norm_y = 0.0f; // y范数

    size_t i = 0;

    if (d >= 8) {
        __m256 vip = _mm256_setzero_ps();
        __m256 vnx = _mm256_setzero_ps();
        __m256 vny = _mm256_setzero_ps();

        while (i + 8 <= d) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = _mm256_loadu_ps(y + i);

            vip = _mm256_fmadd_ps(vx, vy, vip);  // ip += x*y
            vnx = _mm256_fmadd_ps(vx, vx, vnx);  // norm_x += x*x
            vny = _mm256_fmadd_ps(vy, vy, vny);  // norm_y += y*y

            i += 8;
        }

        ip = horizontal_sum_avx2(vip);
        norm_x = horizontal_sum_avx2(vnx);
        norm_y = horizontal_sum_avx2(vny);
    }

    for (; i < d; i++) {
        ip += x[i] * y[i];
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }

    // cos = ip / (sqrt(norm_x) * sqrt(norm_y))
    return ip / (sqrtf(norm_x) * sqrtf(norm_y));
}

// 批量余弦相似度（预计算范数）
void cosine_similarity_batch(
    const float* query,
    const float* database,
    const float* db_norms,  // 预计算的范数
    float* similarities,
    size_t d,
    size_t nb) {

    float query_norm = 0.0f;
    for (size_t i = 0; i < d; i++) {
        query_norm += query[i] * query[i];
    }
    query_norm = sqrtf(query_norm);

    #pragma omp parallel for
    for (size_t j = 0; j < nb; j++) {
        const float* vec = database + j * d;
        float ip = fvec_inner_product_avx2(query, vec, d);
        similarities[j] = ip / (query_norm * db_norms[j]);
    }
}
```

---

## 18. FastScan SIMD优化深度解析

### 18.1 4-bit Packing与位操作

Faiss的FastScan索引使用4-bit编码来压缩向量，需要复杂的SIMD位操作来高效处理。

```cpp
// 4-bit PQ编码的SIMD处理
// 每个字节存储2个4-bit码
class FastScan4BitPQ {
public:
    // AVX2实现：一次处理16个4-bit码（8字节）
    static inline __m256i lookup_4bit_avx2(
            const uint8_t* codes,      // 压缩码
            const float* cent_table,   // 质心表（256 x 16）
            size_t offset) {

        // 加载8字节（16个4-bit码）
        __m128i packed = _mm_loadl_epi64((__m128i*)(codes + offset));

        // 解包为16个字节
        __m256i unpacked = _mm256_cvtepu8_epi16(packed);

        // 提取低4位和高4位
        __m256i low_nibble = _mm256_and_si256(unpacked, _mm256_set1_epi16(0x0F));
        __m256i high_nibble = _mm256_and_si256(
            _mm256_srli_epi16(unpacked, 4),
            _mm256_set1_epi16(0x0F)
        );

        // 查找质心距离表
        __m256 distances_low = lookup_table_avx2(cent_table, low_nibble);
        __m256 distances_high = lookup_table_avx2(cent_table + 128, high_nibble);

        // 合并结果
        return _mm256_hadd_ps(distances_low, distances_high);
    }

private:
    // 查找质心表（使用shuffle指令）
    static inline __m256 lookup_table_avx2(
            const float* table,  // 128 x 4float
            __m256i indices) {   // 16 x 4bit索引

        // 实现简化：实际Faiss使用更复杂的查表
        // 基本思路：将16个4-bit索引扩展为16个字节偏移
        // 然后使用gather或多次load+shuffle

        __m256 result = _mm256_setzero_ps();

        // 这里需要16次查表（或使用gather）
        // 实际Faiss的实现更优化...
        return result;
    }
};
```

### 18.2 AVX-512的4-bit优化

```cpp
// AVX-512可以一次处理32个4-bit码
#ifdef __AVX512F__
class FastScan4BitPQ_AVX512 {
public:
    // 一次处理32个4-bit码（16字节）
    static inline __m512 lookup_4bit_avx512(
            const uint8_t* codes,
            const float* cent_table,
            size_t offset) {

        // 加载16字节（32个4-bit码）
        __m128i packed = _mm_loadu_si128((__m128i*)(codes + offset));

        // 解包为32个字节
        __m256i unpacked256 = _mm256_cvtepu8_epi16(packed);
        __m512i unpacked = _mm512_cvtepu16_epi32(unpacked256);

        // 提取4-bit索引
        __m512i low_nibble = _mm512_and_si512(unpacked, _mm512_set1_epi32(0x0F));
        __m512i high_nibble = _mm512_and_si512(
            _mm512_srli_epi32(unpacked, 4),
            _mm512_set1_epi32(0x0F)
        );

        // 使用gather指令查表（AVX-512的强大功能）
        __m512i low_indices = _mm512_slli_epi32(low_nibble, 2);  // *4 (float size)
        __m512i high_indices = _mm512_slli_epi32(high_nibble, 2);

        __m512 distances_low = _mm512_i32gather_ps(
            low_indices, cent_table, 4);
        __m512 distances_high = _mm512_i32gather_ps(
            high_indices, cent_table + 128, 4);

        // 合并：交错低和高结果
        return _mm512_shuffle_ps(distances_low, distances_high, _MM_SHUFFLE(3,1,3,1));
    }
};
#endif
```

### 18.3 Hamming距离的SIMD优化

```cpp
// 优化汉明距离计算（二进制向量）
#ifdef __AVX2__
// 计算两个256位向量的汉明距离
inline int hamming_distance_avx2(const uint8_t* a, const uint8_t* b) {
    // 加载32字节
    __m256i va = _mm256_loadu_si256((__m256i*)a);
    __m256i vb = _mm256_loadu_si256((__m256i*)b);

    // XOR找不同位
    __m256i vxor = _mm256_xor_si256(va, vb);

    // Popcount：使用AVX2的加速方法
    // 方法：查表+水平求和

    // 1. 每个字节的popcount查找表
    static const uint8_t popcount_table[256] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        // ... 完整表256个元素
    };

    // 2. 使用shuffle查表
    __m256i table_vec = _mm256_loadu_si256((__m256i*)popcount_table);

    // 3. 低4位和高4位分别查表
    __m256i low = _mm256_and_si256(vxor, _mm256_set1_epi8(0x0F));
    __m256i high = _mm256_and_si256(_mm256_srli_epi16(vxor, 4), _mm256_set1_epi8(0x0F));

    __m256i pop_low = _mm256_shuffle_epi8(table_vec, low);
    __m256i pop_high = _mm256_shuffle_epi8(table_vec, high);

    __m256i pop_total = _mm256_add_epi8(pop_low, pop_high);

    // 4. 水平求和
    // 将32个8位值相加
    __m256i sum1 = _mm256_sad_epu8(pop_total, _mm256_setzero_si256());
    __m256i sum2 = _mm256_shuffle_epi32(sum1, _MM_SHUFFLE(1,0,3,2));
    __m256i sum3 = _mm256_add_epi64(sum1, sum2);

    // 提取结果
    uint64_t result[4];
    _mm256_storeu_si256((__m256i*)result, sum3);

    return result[0] + result[2];
}

// AVX-512的popcount专用指令（更快）
#ifdef __AVX512VPOPCNTDQ__
inline int hamming_distance_avx512(const uint8_t* a, const uint8_t* b) {
    __m512i va = _mm512_loadu_si512((__m512i*)a);
    __m512i vb = _mm512_loadu_si512((__m512i*)b);

    // XOR
    __m512i vxor = _mm512_xor_si512(va, vb);

    // 直接使用popcount指令！
    return _mm512_popcnt_epi64(vxor);
}
#endif
#endif
```

### 18.4 批量距离计算的缓存优化

```cpp
// 批量距离计算的内存访问优化
class BatchDistanceComputation {
public:
    // 优化批量内积计算（4个查询向量 vs 数据库）
    static void batch_inner_product_4xQ(
            const float* Q1, const float* Q2, const float* Q3, const float* Q4,
            const float* database,
            float* ip1, float* ip2, float* ip3, float* ip4,
            size_t d, size_t n) {

        size_t i = 0;

        // 主循环：每次处理16个float（AVX2 4-way展开）
        while (i + 16 <= d) {
            // 预取下一批数据
            _mm_prefetch((const char*)(database + i + 16 * n), _MM_HINT_T0);
            _mm_prefetch((const char*)(Q1 + i + 16), _MM_HINT_T0);

            __m256 sum1_0 = _mm256_setzero_ps();
            __m256 sum1_1 = _mm256_setzero_ps();
            __m256 sum2_0 = _mm256_setzero_ps();
            __m256 sum2_1 = _mm256_setzero_ps();
            __m256 sum3_0 = _mm256_setzero_ps();
            __m256 sum3_1 = _mm256_setzero_ps();
            __m256 sum4_0 = _mm256_setzero_ps();
            __m256 sum4_1 = _mm256_setzero_ps();

            // 处理database的每个向量
            for (size_t j = 0; j < n; j++) {
                const float* db_vec = database + j * d + i;

                // 加载查询向量（复用）
                __m256 q1_0 = _mm256_loadu_ps(Q1 + i);
                __m256 q1_1 = _mm256_loadu_ps(Q1 + i + 8);
                __m256 q2_0 = _mm256_loadu_ps(Q2 + i);
                __m256 q2_1 = _mm256_loadu_ps(Q2 + i + 8);
                __m256 q3_0 = _mm256_loadu_ps(Q3 + i);
                __m256 q3_1 = _mm256_loadu_ps(Q3 + i + 8);
                __m256 q4_0 = _mm256_loadu_ps(Q4 + i);
                __m256 q4_1 = _mm256_loadu_ps(Q4 + i + 8);

                // 加载数据库向量
                __m256 db_0 = _mm256_loadu_ps(db_vec);
                __m256 db_1 = _mm256_loadu_ps(db_vec + 8);

                // FMA累加
                sum1_0 = _mm256_fmadd_ps(q1_0, db_0, sum1_0);
                sum1_1 = _mm256_fmadd_ps(q1_1, db_1, sum1_1);
                sum2_0 = _mm256_fmadd_ps(q2_0, db_0, sum2_0);
                sum2_1 = _mm256_fmadd_ps(q2_1, db_1, sum2_1);
                sum3_0 = _mm256_fmadd_ps(q3_0, db_0, sum3_0);
                sum3_1 = _mm256_fmadd_ps(q3_1, db_1, sum3_1);
                sum4_0 = _mm256_fmadd_ps(q4_0, db_0, sum4_0);
                sum4_1 = _mm256_fmadd_ps(q4_1, db_1, sum4_1);
            }

            // 水平求和并存储
            ip1[0] = horizontal_sum_avx2(sum1_0) + horizontal_sum_avx2(sum1_1);
            ip2[0] = horizontal_sum_avx2(sum2_0) + horizontal_sum_avx2(sum2_1);
            ip3[0] = horizontal_sum_avx2(sum3_0) + horizontal_sum_avx2(sum3_1);
            ip4[0] = horizontal_sum_avx2(sum4_0) + horizontal_sum_avx2(sum4_1);

            i += 16;
        }

        // 处理剩余元素
        for (; i < d; i++) {
            for (size_t j = 0; j < n; j++) {
                float db_val = database[j * d + i];
                ip1[j] += Q1[i] * db_val;
                ip2[j] += Q2[i] * db_val;
                ip3[j] += Q3[i] * db_val;
                ip4[j] += Q4[i] * db_val;
            }
        }
    }
};
```

### 18.5 寄存器压力分析与优化

```cpp
// SIMD寄存器压力分析
class RegisterPressureAnalysis {
public:
    // 高寄存器压力版本（容易溢出）
    // 使用16个ymm寄存器
    static void high_register_pressure(
            const float* a, const float* b, float* c, size_t n) {

        size_t i = 0;
        while (i + 32 <= n) {
            // 同时保持太多值在寄存器中
            __m256 a0 = _mm256_loadu_ps(a + i);
            __m256 a1 = _mm256_loadu_ps(a + i + 8);
            __m256 a2 = _mm256_loadu_ps(a + i + 16);
            __m256 a3 = _mm256_loadu_ps(a + i + 24);

            __m256 b0 = _mm256_loadu_ps(b + i);
            __m256 b1 = _mm256_loadu_ps(b + i + 8);
            __m256 b2 = _mm256_loadu_ps(b + i + 16);
            __m256 b3 = _mm256_loadu_ps(b + i + 24);

            __m256 c0 = _mm256_add_ps(a0, b0);
            __m256 c1 = _mm256_add_ps(a1, b1);
            __m256 c2 = _mm256_add_ps(a2, b2);
            __m256 c3 = _mm256_add_ps(a3, b3);

            // 再加上一些中间计算...
            __m256 sum = _mm256_add_ps(c0, c1);
            sum = _mm256_add_ps(sum, c2);
            sum = _mm256_add_ps(sum, c3);

            // 寄存器溢出到栈！
            _mm256_storeu_ps(c + i, c0);
            _mm256_storeu_ps(c + i + 8, c1);
            _mm256_storeu_ps(c + i + 16, c2);
            _mm256_storeu_ps(c + i + 24, c3);

            i += 32;
        }
    }

    // 优化版本：减少寄存器使用
    static void optimized_register_usage(
            const float* a, const float* b, float* c, size_t n) {

        size_t i = 0;
        while (i + 32 <= n) {
            // 每次只保持少量寄存器
            __m256 a0 = _mm256_loadu_ps(a + i);
            __m256 b0 = _mm256_loadu_ps(b + i);
            __m256 c0 = _mm256_add_ps(a0, b0);
            _mm256_storeu_ps(c + i, c0);  // 立即存储，释放寄存器

            __m256 a1 = _mm256_loadu_ps(a + i + 8);
            __m256 b1 = _mm256_loadu_ps(b + i + 8);
            __m256 c1 = _mm256_add_ps(a1, b1);
            _mm256_storeu_ps(c + i + 8, c1);

            __m256 a2 = _mm256_loadu_ps(a + i + 16);
            __m256 b2 = _mm256_loadu_ps(b + i + 16);
            __m256 c2 = _mm256_add_ps(a2, b2);
            _mm256_storeu_ps(c + i + 16, c2);

            __m256 a3 = _mm256_loadu_ps(a + i + 24);
            __m256 b3 = _mm256_loadu_ps(b + i + 24);
            __m256 c3 = _mm256_add_ps(a3, b3);
            _mm256_storeu_ps(c + i + 24, c3);

            i += 32;
        }
    }

    // 检查寄存器溢出（使用编译器选项）
    // gcc -O3 -fopt-info-vec-all -march=native program.cpp
    // clang -O3 -Rpass=vectorize -march=native program.cpp
};
```

---

## 19. 多线程SIMD调度策略

### 19.1 NUMA-aware SIMD调度

```cpp
// NUMA节点感知的多线程SIMD计算
class NUMASIMDScheduler {
    int num_numa_nodes;
    int cores_per_numa;

public:
    NUMASIMDScheduler() {
        #ifdef __linux__
        num_numa_nodes = numa_num_configured_nodes();
        cores_per_numa = std::thread::hardware_concurrency() / num_numa_nodes;
        #else
        num_numa_nodes = 1;
        cores_per_numa = std::thread::hardware_concurrency();
        #endif
    }

    // 跨NUMA节点的并行距离计算
    void parallel_distance_compute(
            const float* x,
            const float* y,
            size_t d,
            size_t n,
            float* distances) {

        std::vector<std::thread> threads;

        for (int node = 0; node < num_numa_nodes; node++) {
            threads.emplace_back([this, node, x, y, d, n, distances]() {
                // 绑定到NUMA节点
                #ifdef __linux__
                numa_set_preferred(node);
                numa_run_on_node(node);
                #endif

                size_t start = (n / num_numa_nodes) * node;
                size_t end = (node == num_numa_nodes - 1) ? n :
                            (n / num_numa_nodes) * (node + 1);

                // 使用OpenMP在NUMA节点内部并行
                #pragma omp parallel for num_threads(cores_per_numa)
                for (size_t i = start; i < end; i++) {
                    distances[i] = fvec_L2sqr_avx2(x, y + i * d, d);
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }
    }
};
```

### 19.2 动态负载均衡

```cpp
// SIMD任务的动态负载均衡
class DynamicSIMDScheduler {
    std::atomic<size_t> next_task{0};
    size_t total_tasks;
    size_t chunk_size;

public:
    DynamicSIMDScheduler(size_t n, size_t chunk = 64)
        : total_tasks(n), chunk_size(chunk) {}

    // 工作窃取调度
    void process_tasks(std::function<void(size_t)> simd_func) {
        std::vector<std::thread> workers;
        int num_threads = std::thread::hardware_concurrency();

        for (int t = 0; t < num_threads; t++) {
            workers.emplace_back([this, simd_func]() {
                while (true) {
                    // 获取一批任务
                    size_t start = next_task.fetch_add(chunk_size);
                    if (start >= total_tasks) break;

                    size_t end = std::min(start + chunk_size, total_tasks);

                    // SIMD处理这一批任务
                    for (size_t i = start; i < end; i++) {
                        simd_func(i);
                    }
                }
            });
        }

        for (auto& w : workers) {
            w.join();
        }
    }
};
```

---

## 20. 底层编译器优化技巧

### 20.1 编译器内在函数扩展

```cpp
// 编译器特定优化

// GCC/Clang的属性扩展
#if defined(__GNUC__) || defined(__clang__)

// 热函数优化
__attribute__((hot)) inline
float hot_distance_function(const float* x, const float* y, size_t d) {
    // 编译器会对此函数进行更激进的优化
    return fvec_L2sqr_avx2(x, y, d);
}

// 纯函数（无副作用，返回值只依赖参数）
__attribute__((const)) inline
float pure_exp(float x) {
    // 编译器可以进行CSE（公共子表达式消除）
    return std::exp(x);
}

// 目标特定优化
__attribute__((target("avx2,fma"))) inline
__m256 avx2_fma_mul_add(__m256 a, __m256 b, __m256 c) {
    // 只在支持AVX2+FMA的CPU上编译此函数
    return _mm256_fmadd_ps(a, b, c);
}

// 总是内联
__attribute__((always_inline)) inline
float always_inline_sum(const float* x, size_t n) {
    // 强制内联，避免函数调用开销
    float sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum;
}

// 版本化函数（根据CPU特性选择）
__attribute__((target("default"))) float
dispatch_distance(const float* x, const float* y, size_t d) {
    // 默认实现（标量）
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        float diff = x[i] - y[i];
        res += diff * diff;
    }
    return res;
}

__attribute__((target("avx2"))) float
dispatch_distance(const float* x, const float* y, size_t d) {
    return fvec_L2sqr_avx2(x, y, d);
}

__attribute__((target("avx512f"))) float
dispatch_distance(const float* x, const float* y, size_t d) {
    return fvec_L2sqr_avx512(x, y, d);
}

#endif

// MSVC属性
#ifdef _MSC_VER

__declspec(hot) inline
float msvc_hot_distance(const float* x, const float* y, size_t d) {
    return fvec_L2sqr_avx2(x, y, d);
}

__declspec(forceinline) inline
float msvc_force_inline(const float* x, size_t d) {
    float sum = 0;
    for (size_t i = 0; i < d; i++) {
        sum += x[i];
    }
    return sum;
}

#endif
```

### 20.2 链接时优化(LTO)配置

```cmake
# CMakeLists.txt中的LTO配置

# 启用LTO（链接时优化）
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# 特定优化级别
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_LEVEL 3)

# 使用Fat LTO对象（支持增量LTO）
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffat-lto-objects")
endif()

# ThinLTO（Clang的增量LTO）
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=thin")
    # 并行LTO
    set(CMAKE_JOB_POOLS "link=4")
endif()
```

---

## 21. 性能剖析与硬件计数器

### 21.1 perf事件分析

```bash
# CPU性能计数器分析

# 测量缓存未命中率
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
    ./faiss_benchmark

# 测量分支预测
perf stat -e branches,branch-misses ./faiss_benchmark

# 测量SIMD指令使用
perf stat -e instructions,cycles,simd_comp_inst ./faiss_benchmark

# 火焰图生成
perf record -F 99 -g ./faiss_benchmark
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg

# 热点函数分析
perf report --stdio --sort=dso,symbol
```

### 21.2 硬件性能计数器API

```cpp
// 使用Linux perf_event API
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

class PerfCounter {
    int fd;

public:
    PerfCounter(perf_type_id type, perf_hw_id event) {
        struct perf_event_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.type = type;
        attr.config = event;
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;

        fd = syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
        if (fd == -1) {
            perror("perf_event_open");
        }
    }

    void start() {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    }

    void stop() {
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    }

    uint64_t read() {
        uint64_t value;
        ::read(fd, &value, sizeof(value));
        return value;
    }

    ~PerfCounter() {
        close(fd);
    }
};

// 使用示例
void profile_distance_function() {
    PerfCounter cache_refs(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_REFERENCES);
    PerfCounter cache_misses(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_MISSES);

    cache_refs.start();
    cache_misses.start();

    // 运行代码
    benchmark_distance();

    cache_refs.stop();
    cache_misses.stop();

    printf("Cache references: %lu\n", cache_refs.read());
    printf("Cache misses: %lu\n", cache_misses.read());
    printf("Miss rate: %.2f%%\n",
           100.0 * cache_misses.read() / cache_refs.read());
}
#endif
```

---

## 扩展阅读

- faiss/utils/simdlib_avx2.h - AVX2 SIMD库
- faiss/utils/simdlib_avx512.h - AVX-512 SIMD库
- faiss/utils/simdlib_neon.h - ARM NEON库
- faiss/utils/distances_simd.cpp - SIMD优化的距离计算
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/)
- [Intel 64 and IA-32 Architectures Optimization Reference Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
