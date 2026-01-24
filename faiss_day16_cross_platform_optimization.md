# Faiss深度课程 - 第16天：跨平台优化详解

## 课程目标

深入理解Faiss在不同CPU架构（x86 AVX2/AVX-512、ARM NEON/SVE）上的SIMD优化实现，学习如何编写高性能的跨平台代码。

---

## 1. SIMD指令集概览

### 1.1 主流SIMD指令集

```cpp
// SIMD指令集对比
struct SIMDInfo {
    const char* name;
    int bits;           // 寄存器宽度
    int float_count;    // 可处理的float数
    int int8_count;     // 可处理的int8数
    const char* cpu_flag;
};

std::vector<SIMDInfo> simd_sets = {
    {"SSE",        128, 4,  16,  "__SSE__"},
    {"SSE2",       128, 2,  16,  "__SSE2__"},
    {"SSE4.1",     128, 4,  16,  "__SSE4_1__"},
    {"AVX",        256, 8,  32,  "__AVX__"},
    {"AVX2",       256, 8,  32,  "__AVX2__"},
    {"AVX-512",    512, 16, 64,  "__AVX512F__"},
    {"ARM NEON",   128, 4,  16,  "__aarch64__"},
    {"ARM SVE",    可变, 可变, 可变, "__ARM_FEATURE_SVE"},
    {"ARM SVE2",   可变, 可变, 可变, "__ARM_FEATURE_SVE2"}
};
```

### 1.2 检测CPU特性

```cpp
// faiss/utils/utils.h
// 运行时CPU特性检测
struct CpuFeatures {
    bool avx2;
    bool avx512f;
    bool avx512cd;
    bool avx512bw;
    bool avx512dq;
    bool avx512_vnni;
    bool neon;
    bool sve;
    bool sve2;
};

CpuFeatures get_cpu_features() {
    CpuFeatures f = {};

    // x86特性
#ifdef __x86_64__
    // 使用cpuid指令检测
    unsigned int eax, ebx, ecx, edx;

    // 检测AVX2
    __cpuid(7, eax, ebx, ecx, edx);
    f.avx2 = (ebx & (1 << 5)) != 0;

    // 检测AVX-512F
    f.avx512f = (ebx & (1 << 16)) != 0;
    f.avx512cd = (ebx & (1 << 28)) != 0;
    f.avx512bw = (ebx & (1 << 30)) != 0;
    f.avx512dq = (ebx & (1 << 17)) != 0;
    f.avx512_vnni = (ecx & (1 << 4)) != 0;

#elif defined(__aarch64__)
    // ARM特性
    f.neon = true;

    // 检测SVE/SVE2
    // 通过HWCAP或系统调用检测
    f.sve = getauxval(AT_HWCAP) & HWCAP_SVE;
    f.sve2 = getauxval(AT_HWCAP2) & HWCAP2_SVE2;
#endif

    return f;
}
```

---

## 2. x86 AVX2优化详解

### 2.1 AVX2基础操作

```cpp
#ifdef __AVX2__
#include <immintrin.h>

// AVX2数据类型
using __m256 = __m256;        // 8 x float
using __m256i = __m256i;      // 32 x int8 / 16 x int16 / 8 x int32

// AVX2加载/存储
void avx2_load_store_example() {
    // 对齐加载（32字节对齐）
    alignas(32) float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    __m256 a = _mm256_load_ps(data);

    // 非对齐加载
    float* unaligned = new float[8];
    __m256 b = _mm256_loadu_ps(unaligned);

    // 存储
    alignas(32) float result[8];
    _mm256_store_ps(result, a);

    // 非对齐存储
    _mm256_storeu_ps(unaligned, b);

    delete[] unaligned;
}
#endif

// AVX2算术操作
void avx2_arithmetic_example() {
#ifdef __AVX2__
    alignas(32) float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(32) float b[8] = {8, 7, 6, 5, 4, 3, 2, 1};

    __m256 va = _mm256_load_ps(a);
    __m256 vb = _mm256_load_ps(b);

    // 加法
    __m256 vadd = _mm256_add_ps(va, vb);

    // 减法
    __m256 vsub = _mm256_sub_ps(va, vb);

    // 乘法
    __m256 vmul = _mm256_mul_ps(va, vb);

    // FMA: a*b+c
    __m256 vfma = _mm256_fmadd_ps(va, vb, vadd);

    // 存储结果
    alignas(32) float result[8];
    _mm256_store_ps(result, vfma);
#endif
}
```

### 2.2 AVX2 L2距离计算

```cpp
// faiss/utils/distances_simd.cpp
// AVX2优化的L2距离平方计算
#ifdef __AVX2__
float fvec_L2sqr_avx2(const float* x, const float* y, size_t d) {
    float res = 0;
    size_t i = 0;

    // 处理8的倍数（AVX2一次处理8个float）
    for (; i + 8 <= d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);

        __m256 diff = _mm256_sub_ps(vx, vy);
        __m256 sq = _mm256_mul_ps(diff, diff);

        // 水平求和
        res += _mm256_reduce_add_ps(sq);
    }

    // 处理剩余元素
    for (; i < d; i++) {
        float tmp = x[i] - y[i];
        res += tmp * tmp;
    }

    return res;
}

// AVX2水平求和实现
inline float _mm256_reduce_add_ps(__m256 v) {
    // v = [x0, x1, x2, x3, x4, x5, x6, x7]

    // 提取高128位和低128位
    const __m128 v0 = _mm256_castps256_ps128(v);   // [x0, x1, x2, x3]
    const __m128 v1 = _mm256_extractf128_ps(v, 1); // [x4, x5, x6, x7]

    // 相加
    const __m128 v2 = _mm_add_ps(v0, v1);  // [x0+x4, x1+x5, x2+x6, x3+x7]

    // 再次相加
    __m128 v3 = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 3, 2));
    // v3 = [x2+x6, x3+x7, x2+x6, x3+x7]

    const __m128 v4 = _mm_add_ps(v2, v3);
    // v4 = [x0+x4+x2+x6, x1+x5+x3+x7, ...]

    // 最后相加
    __m128 v5 = _mm_shuffle_ps(v4, v4, _MM_SHUFFLE(0, 0, 0, 1));
    const __m128 v6 = _mm_add_ss(v4, v5);

    return _mm_cvtss_f32(v6);
}
#endif
```

### 2.3 AVX2内积计算

```cpp
#ifdef __AVX2__
// AVX2优化的内积计算
float fvec_inner_product_avx2(const float* x, const float* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    // 批量处理
    for (; i + 8 <= d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);

        sum = _mm256_add_ps(sum, _mm256_mul_ps(vx, vy));
    }

    // 水平求和
    float result = _mm256_reduce_add_ps(sum);

    // 处理剩余元素
    for (; i < d; i++) {
        result += x[i] * y[i];
    }

    return result;
}
#endif
```

### 2.4 AVX2比较和选择

```cpp
#ifdef __AVX2__
// AVX2比较操作
void avx2_compare_example() {
    alignas(32) float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(32) float b[8] = {4, 3, 2, 1, 8, 7, 6, 5};

    __m256 va = _mm256_load_ps(a);
    __m256 vb = _mm256_load_ps(b);

    // 比较（返回掩码）
    __m256 cmp_lt = _mm256_cmp_ps(va, vb, _CMP_LT_OQ);
    __m256 cmp_gt = _mm256_cmp_ps(va, vb, _CMP_GT_OQ);

    // 掩码操作
    __m256 masked = _mm256_and_ps(cmp_lt, va);

    // 混合选择
    // 如果mask中对应位为1，选a；否则选b
    __m256 blended = _mm256_blendv_ps(vb, va, cmp_lt);

    // 存储结果
    alignas(32) float result[8];
    _mm256_store_ps(result, blended);
}
#endif
```

---

## 3. x86 AVX-512优化详解

### 3.1 AVX-512基础操作

```cpp
#ifdef __AVX512F__
#include <immintrin.h>

// AVX-512数据类型
using __m512 = __m512;        // 16 x float
using __m512i = __m512i;      // 64 x int8 / 16 x int32

// AVX-512加载/存储
void avx512_load_store_example() {
    // 对齐加载（64字节对齐）
    alignas(64) float data[16] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16
    };
    __m512 a = _mm512_load_ps(data);

    // 非对齐加载
    float* unaligned = new float[16];
    __m512 b = _mm512_loadu_ps(unaligned);

    // 掩码加载（条件加载）
    __mmask16 mask = 0xFF00;  // 高8位有效
    __m512 c = _mm512_maskz_load_ps(mask, data);

    // 存储
    alignas(64) float result[16];
    _mm512_store_ps(result, a);

    // 掩码存储
    _mm512_mask_store_ps(result, mask, b);

    delete[] unaligned;
}
#endif
```

### 3.2 AVX-512 L2距离计算

```cpp
#ifdef __AVX512F__
// AVX-512优化的L2距离平方计算
float fvec_L2sqr_avx512(const float* x, const float* y, size_t d) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;

    // 批量处理16个float
    for (; i + 16 <= d; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);

        __m512 diff = _mm512_sub_ps(vx, vy);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    // 水平求和
    float result = _mm512_reduce_add_ps(sum);

    // 处理剩余元素
    for (; i < d; i++) {
        float tmp = x[i] - y[i];
        result += tmp * tmp;
    }

    return result;
}

// AVX-512水平求和
inline float _mm512_reduce_add_ps(__m512 v) {
    // 使用专用指令
    return _mm512_reduce_ps(v, _MM_FADD_JS);
}

// 或者手动实现
inline float _mm512_reduce_add_ps_manual(__m512 v) {
    // v = [x0, x1, ..., x15]

    // 256位shuffle并相加
    __m256 vlow = _mm512_castps512_ps256(v);
    __m256 vhigh = _mm512_extractf32x8_ps(v, 1);
    __m256 vsum = _mm256_add_ps(vlow, vhigh);

    // 继续求和
    return _mm256_reduce_add_ps(vsum);
}
#endif
```

### 3.3 AVX-512掩码操作

```cpp
#ifdef __AVX512F__
// AVX-512掩码操作示例
void avx512_mask_example() {
    alignas(64) float a[16];
    alignas(64) float b[16];

    // 初始化a和b...

    __m512 va = _mm512_load_ps(a);
    __m512 vb = _mm512_load_ps(b);

    // 比较生成掩码
    __mmask16 mask = _mm512_cmp_ps_mask(va, vb, _CMP_LT_OQ);

    // 掩码加载
    __m512 vc = _mm512_maskz_load_ps(mask, a);

    // 掩码算术操作
    __m512 vd = _mm512_mask_add_ps(vb, mask, va, vb);

    // 掩码统计
    int count = _mm_popcnt_u32(mask);  // mask中1的个数

    // 掩码压缩（提取有效元素）
    __m512 ve = _mm512_mask_compress_ps(vd, mask, va);
}
#endif
```

### 3.4 AVX-512 gather/scatter

```cpp
#ifdef __AVX512F__
// AVX-512 gather操作（非连续内存加载）
void avx512_gather_example() {
    // 索引数组
    alignas(64) int indices[16] = {0, 4, 8, 12, 16, 20, 24, 28,
                                     1, 5, 9, 13, 17, 21, 25, 29};

    // 大数组
    float* large_array = new float[1000];
    // 初始化...

    // Gather: 根据索引非连续加载
    __m512i vindices = _mm512_load_epi32(indices);
    __m512 gathered = _mm512_i32gather_ps(
        vindices,          // 索引
        large_array,       // 基地址
        4);                // scale（每个元素4字节）

    // Scatter: 根据索引非连续存储
    __m512 data = _mm512_set1_ps(42.0f);
    _mm512_i32scatter_ps(
        large_array,       // 基地址
        vindices,          // 索引
        data,              // 数据
        4);                // scale

    delete[] large_array;
}
#endif
```

---

## 4. ARM NEON优化详解

### 4.1 NEON基础操作

```cpp
#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>

// NEON数据类型
using float32x4_t = float32x4_t;    // 4 x float
using int32x4_t = int32x4_t;        // 4 x int32
using int8x16_t = int8x16_t;        // 16 x int8

// NEON加载/存储
void neon_load_store_example() {
    // 对齐加载（16字节对齐）
    alignas(16) float data[4] = {1, 2, 3, 4};
    float32x4_t a = vld1q_f32(data);

    // 非对齐加载
    float* unaligned = new float[4];
    float32x4_t b = vld1q_f32(unaligned);

    // 存储
    alignas(16) float result[4];
    vst1q_f32(result, a);

    // 加载多个向量
    alignas(16) float data2[8];
    float32x4_t c0, c1;
    vld1q_f32_x2(data2, &c0, &c1);

    delete[] unaligned;
}
#endif
```

### 4.2 NEON L2距离计算

```cpp
#if defined(__aarch64__) || defined(__ARM_NEON)
// NEON优化的L2距离平方计算
float fvec_L2sqr_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum = vdupq_n_f32(0);
    size_t i = 0;

    // 批量处理4个float
    for (; i + 4 <= d; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);

        float32x4_t diff = vsubq_f32(vx, vy);
        sum = vmlaq_f32(sum, diff, diff);  // sum += diff * diff
    }

    // 水平求和
    float result[4];
    vst1q_f32(result, sum);
    float res = result[0] + result[1] + result[2] + result[3];

    // 处理剩余元素
    for (; i < d; i++) {
        float tmp = x[i] - y[i];
        res += tmp * tmp;
    }

    return res;
}
#endif
```

### 4.3 NEON内积计算

```cpp
#if defined(__aarch64__) || defined(__ARM_NEON)
// NEON优化的内积计算
float fvec_inner_product_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum = vdupq_n_f32(0);
    size_t i = 0;

    // 批量处理
    for (; i + 4 <= d; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);

        sum = vfmaq_f32(sum, vx, vy);  // sum += x * y
    }

    // 水平求和
    float result[4];
    vst1q_f32(result, sum);
    float res = result[0] + result[1] + result[2] + result[3];

    // 处理剩余元素
    for (; i < d; i++) {
        res += x[i] * y[i];
    }

    return res;
}
#endif
```

### 4.4 NEON查表操作

```cpp
#if defined(__aarch64__) || defined(__ARM_NEON)
// NEON优化的查表（用于FastScan）
void neon_lookup_example() {
    // 查找表（16个float）
    alignas(16) float table[16] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15
    };

    // 索引（4个uint8）
    alignas(16) uint8_t indices[16] = {
        0, 5, 10, 15, 3, 7, 11, 13,
        1, 4, 8, 12, 2, 6, 9, 14
    };

    // 加载索引
    uint8x16_t vidx = vld1q_u8(indices);

    // 扩展为两个int8x16_t向量（高4位和低4位）
    int8x16_t idx_low = vreinterpretq_s8_u8(vandq_u8(
        vidx, vdupq_n_u8(0x0F)));
    int8x16_t idx_high = vreinterpretq_s8_u8(vshrq_n_u8(
        vidx, 4));

    // 查表（使用tbl指令）
    float32x4_t result_low = vld1q_f32(table);  // 简化，需要实际查表实现

    // ARM NEON的查表比较复杂，通常需要多次指令
    // 或者使用特殊指令如vtbl（32-bit）和vtbx（64-bit）
}
#endif
```

---

## 5. ARM SVE优化详解

### 5.1 SVE基础操作

```cpp
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>

// SVE特性：可变向量长度
void sve_hello_world() {
    // 获取向量长度（以bit为单位）
    uint64_t vl = svcntw();  // word count

    printf("SVE vector length: %lu bits (%lu words)\n", vl, vl / 32);

    // 创建向量（长度在运行时确定）
    svfloat32_t va = svdup_f32(1.0f);      // 所有元素为1.0
    svfloat32_t vb = svdup_f32(2.0f);      // 所有元素为2.0

    // 算术操作
    svfloat32_t vc = svadd_f32_x(svptrue_b32(), va, vb);

    // 存储结果
    // 需要运行时确定向量长度
    int count = svcntw();  // float32元素数
    float* result = new float[count];
    svst1_f32(svptrue_b32(), result, vc);

    delete[] result;
}
#endif
```

### 5.2 SVE L2距离计算

```cpp
#ifdef __ARM_FEATURE_SVE
// SVE优化的L2距离平方计算
float fvec_L2sqr_sve(const float* x, const float* y, size_t d) {
    svfloat32_t sum = svdup_f32(0.0f);
    size_t i = 0;

    // 获取向量长度
    uint64_t vl = svcntw();

    // 批量处理（向量长度由硬件决定）
    for (; i + vl <= d; i += vl) {
        svbool_t pg = svwhilelt_b32(i, d);  // 谓词

        svfloat32_t vx = svld1_f32(pg, x + i);
        svfloat32_t vy = svld1_f32(pg, y + i);

        svfloat32_t diff = svsub_f32_x(pg, vx, vy);
        sum = svmla_f32_x(pg, sum, diff, diff);
    }

    // 水平求和
    float res = svaddv_f32(svptrue_b32(), sum);

    // 处理剩余元素
    for (; i < d; i++) {
        float tmp = x[i] - y[i];
        res += tmp * tmp;
    }

    return res;
}
#endif
```

---

## 6. 跨平台抽象层

### 6.1 Faiss的SIMD抽象

```cpp
// faiss/utils/simdlib.h
// Faiss的跨平台SIMD抽象

#if defined(__AVX512F__)
    #include <faiss/utils/simdlib_avx512.h>
    using simd_uint16 = simd16uint16_avx512;
#elif defined(__AVX2__)
    #include <faiss/utils/simdlib_avx2.h>
    using simd_uint16 = simd16uint16_avx2;
#elif defined(__aarch64__) || defined(__ARM_NEON)
    #include <faiss/utils/simdlib_neon.h>
    using simd_uint16 = simd16uint16_neon;
#else
    #include <faiss/utils/simdlib_emulated.h>
    using simd_uint16 = simd16uint16_emulated;
#endif

// 统一接口
namespace simd {

// 16个uint16的SIMD向量
struct simd16uint16 {
    // 平台特定的实现
    #ifdef __AVX2__
        __m256i vi;
    #elif defined(__aarch64__)
        uint8x16_t val[2];  // 16 x uint16
    #endif

    // 加载
    static simd16uint16 load(const uint16_t* ptr) {
        #ifdef __AVX2__
            return _mm256_loadu_si256((__m256i*)ptr);
        #elif defined(__aarch64__)
            // NEON实现
            simd16uint16 result;
            // ...
            return result;
        #endif
    }

    // 加法
    simd16uint16 operator+(const simd16uint16& other) const {
        #ifdef __AVX2__
            return _mm256_adds_epu16(vi, other.vi);
        #elif defined(__aarch64__)
            // NEON实现
            return *this;
        #endif
    }

    // 按位与
    simd16uint16 operator&(const simd16uint16& other) const {
        #ifdef __AVX2__
            return _mm256_and_si256(vi, other.vi);
        #elif defined(__aarch64__)
            // NEON实现
            return *this;
        #endif
    }
};

} // namespace simd
```

### 6.2 距离计算的跨平台实现

```cpp
// faiss/utils/distances.cpp
// 跨平台的L2距离计算

float fvec_L2sqr(const float* x, const float* y, size_t d) {
    // 根据编译选项选择最优实现

    #if defined(__AVX512F__)
        return fvec_L2sqr_avx512(x, y, d);
    #elif defined(__AVX2__)
        return fvec_L2sqr_avx2(x, y, d);
    #elif defined(__aarch64__) || defined(__ARM_NEON)
        return fvec_L2sqr_neon(x, y, d);
    #elif defined(__ARM_FEATURE_SVE)
        return fvec_L2sqr_sve(x, y, d);
    #else
        // 标量实现
        float res = 0;
        for (size_t i = 0; i < d; i++) {
            float tmp = x[i] - y[i];
            res += tmp * tmp;
        }
        return res;
    #endif
}

// 运行时dispatch
using fvec_L2sqr_fn_t = float(*)(const float*, const float*, size_t);

fvec_L2sqr_fn_t get_fvec_L2sqr_fn() {
    static fvec_L2sqr_fn_t fn = nullptr;

    if (fn == nullptr) {
        // 运行时检测CPU特性
        auto features = get_cpu_features();

        if (features.avx512f) {
            fn = fvec_L2sqr_avx512;
        } else if (features.avx2) {
            fn = fvec_L2sqr_avx2;
        } else if (features.sve) {
            fn = fvec_L2sqr_sve;
        } else if (features.neon) {
            fn = fvec_L2sqr_neon;
        } else {
            fn = fvec_L2sqr_ref;
        }
    }

    return fn;
}
```

### 6.3 通用宏定义

```cpp
// faiss/utils/platform_macros.h
// 平台相关的通用宏

// 字节序
#if defined(__BYTE_ORDER__)
    #define FAISS_LITTLE_ENDIAN (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#elif defined(_WIN32)
    #define FAISS_LITTLE_ENDIAN true
#else
    #define FAISS_LITTLE_ENDIAN (__LITTLE_ENDIAN__ == 1)
#endif

// 内存对齐
#if defined(_WIN32)
    #define FAISS_ALIGN(x) __declspec(align(x))
#else
    #define FAISS_ALIGN(x) __attribute__((aligned(x)))
#endif

// 强内联
#if defined(_MSC_VER)
    #define FAISS_FORCE_INLINE __forceinline
#else
    #define FAISS_FORCE_INLINE __attribute__((always_inline)) inline
#endif

// 分支预测提示
#if defined(_MSC_VER)
    #define FAISS_LIKELY(x) (x)
    #define FAISS_UNLIKELY(x) (x)
#else
    #define FAISS_LIKELY(x) __builtin_expect(!!(x), 1)
    #define FAISS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
```

---

## 7. 性能测试与基准

### 7.1 跨平台性能测试框架

```cpp
// 跨平台性能测试
class SIMDPerformanceBenchmark {
public:
    void benchmark_distance_functions() {
        int d = 128;
        size_t n = 1000000;  // 100万向量

        // 生成随机数据
        std::vector<float> x(n * d);
        std::vector<float> y(n * d);
        generate_random_data(x.data(), n * d);
        generate_random_data(y.data(), n * d);

        printf("=== Distance Computation Benchmark ===\n");
        printf("Vector dimension: %d\n", d);
        printf("Number of vectors: %zu\n\n", n);

        // 测试不同实现
        benchmark_impl("Scalar", fvec_L2sqr_ref, x.data(), y.data(), n, d);

    #ifdef __AVX2__
        benchmark_impl("AVX2", fvec_L2sqr_avx2, x.data(), y.data(), n, d);
    #endif

    #ifdef __AVX512F__
        benchmark_impl("AVX-512", fvec_L2sqr_avx512, x.data(), y.data(), n, d);
    #endif

    #if defined(__aarch64__) || defined(__ARM_NEON)
        benchmark_impl("NEON", fvec_L2sqr_neon, x.data(), y.data(), n, d);
    #endif

    #ifdef __ARM_FEATURE_SVE
        benchmark_impl("SVE", fvec_L2sqr_sve, x.data(), y.data(), n, d);
    #endif
    }

private:
    void benchmark_impl(
            const char* name,
            DistanceFn fn,
            const float* x,
            const float* y,
            size_t n,
            int d) {

        // 预热
        for (size_t i = 0; i < 1000; i++) {
            fn(x + i * d, y + i * d, d);
        }

        // 计时
        auto start = std::chrono::high_resolution_clock::now();

        float total = 0;
        for (size_t i = 0; i < n; i++) {
            total += fn(x + i * d, y + i * d, d);
        }

        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double>(
            end - start).count() * 1000;

        double mops = n / time_ms / 1000;  // Millions of operations per second

        printf("%-10s: %.2f ms, %.2f Mops, result=%.2f\n",
               name, time_ms, mops, total / n);
    }
};
```

---

## 8. 编译优化选项

### 8.1 CMake配置

```cmake
# faiss CMakeLists.txt
# 根据CPU特性设置优化选项

# 检测AVX2
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(faiss_avx2 PRIVATE "-mavx2")
    endif()
endif()

# 检测AVX-512
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512)
    target_compile_options(faiss_avx512 PRIVATE
        "-mavx512f" "-mavx512cd" "-mavx512bw" "-mavx512dq")
endif()

# ARM NEON
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(faiss_neon PRIVATE "-mfpu=neon")
endif()

# ARM SVE
check_cxx_compiler_flag("-msve" COMPILER_SUPPORTS_SVE)
if(COMPILER_SUPPORTS_SVE)
    target_compile_options(faiss_sve PRIVATE "-msve")
endif()
```

### 8.2 多版本库构建

```cmake
# 构建多个优化级别的库
add_library(faiss ${FAISS_SRCS})
add_library(faiss_avx2 ${FAISS_SRCS})
add_library(faiss_avx512 ${FAISS_SRCS})

# 为不同库设置编译选项
target_compile_definitions(faiss_avx2 PRIVATE FAISS_USE_AVX2)
target_compile_options(faiss_avx2 PRIVATE "-mavx2")

target_compile_definitions(faiss_avx512 PRIVATE FAISS_USE_AVX512)
target_compile_options(faiss_avx512 PRIVATE
    "-mavx512f" "-mavx512cd" "-mavx512bw" "-mavx512dq")
```

---

## 9. 高级SIMD优化技巧

### 9.1 AVX-512掩码操作与条件执行

```cpp
#ifdef __AVX512F__
// AVX-512掩码操作：零开销条件执行
void avx512_conditional_distance(
        const float* x,
        const float* y,
        const uint8_t* mask,  // 控制哪些元素参与计算
        float* output,
        size_t d) {

    size_t i = 0;
    __m512 sum = _mm512_setzero_ps();

    // 处理16的倍数
    for (; i + 16 <= d; i += 16) {
        // 加载掩码（每16位对应1个float）
        __mmask16 m = _mm512_cmpneq_epu16_mask(
            _mm512_loadu_si512((__m512i*)(mask + i)),
            _mm512_setzero_si512());

        // 条件加载（只加载mask为1的元素）
        __m512 vx = _mm512_maskz_loadu_ps(m, x + i);
        __m512 vy = _mm512_maskz_loadu_ps(m, y + i);

        // 条件FMA操作
        __m512 diff = _mm512_sub_ps(vx, vy);
        sum = _mm512_mask3_fmadd_ps(diff, diff, sum, m);
    }

    float result = _mm512_reduce_add_ps(sum);
    output[0] = result;
}

// AVX-512 gather/scatter：非连续内存访问
void avx512_gather_example(
        const float* base,
        const int* indices,
        float* output,
        size_t n) {

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        // 加载16个索引
        __m512i vidx = _mm512_loadu_si512((__m512i*)(indices + i));

        // gather: 从非连续位置加载数据
        __m512 gathered = _mm512_i32gather_ps(vidx, base, 4);

        // 处理数据
        __m512 result = _mm512_mul_ps(gathered, _mm512_set1_ps(2.0f));

        // scatter: 写入非连续位置
        __m512i vout_idx = _mm512_loadu_si512((__m512i*)(indices + i));
        _mm512_i32scatter_ps(output, vout_idx, result, 4);
    }
}

// AVX-512压缩与扩展
void avx512_compress_expand(
        const float* input,
        float* compressed,
        float* expanded,
        const __mmask16 mask) {

    __m512 data = _mm512_loadu_ps(input);

    // 压缩：保留mask为1的元素，紧凑排列
    __m512 comp = _mm512_mask_compress_ps(data, mask, data);
    _mm512_storeu_ps(compressed, comp);

    // 扩展：从压缩数据恢复
    __m512 exp = _mm512_maskz_expand_ps(mask, comp);
    _mm512_storeu_ps(expanded, exp);
}
#endif
```

### 9.2 NEON高级查表操作

```cpp
#if defined(__aarch64__) && defined(__ARM_NEON)

// NEON查表：TBL/TBX指令
void neon_lookup_table(
        const uint8_t* data,
        const uint8_t* table,
        uint8_t* output,
        size_t n) {

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        // 加载16个索引
        uint8x16_t indices = vld1q_u8(data + i);

        // 单表查表（每个表128字节）
        uint8x16_t result = vqtbl1q_u8(vld1q_u8(table), indices);
        vst1q_u8(output + i, result);
    }
}

// NEON双表查表（更大范围）
void neon_dual_lookup(
        const uint8_t* indices,
        const uint8_t* table1,
        const uint8_t* table2,
        uint8_t* output,
        size_t n) {

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t idx = vld1q_u8(indices + i);

        // 从两个表中查表
        uint8x16_t result = vqtbl2q_u8(
            vcombine_u8(vld1q_u8(table1), vld1q_u8(table2)),
            idx);

        vst1q_u8(output + i, result);
    }
}

// NEON多项式查表（PQ距离计算）
void neon_poly_lookup(
        const uint8_t* codes,
        const float* tables,  // [M][256]
        float* output,
        size_t n) {

    const int M = 16;  // 子量化器数量

    for (size_t i = 0; i < n; i++) {
        float32x4_t sum = vdupq_n_f32(0.0f);

        for (int m = 0; m < M; m += 4) {
            // 加载4个code
            uint8x16_t c = vld1q_u8(codes + i * M + m);

            // 扩展为32位索引
            uint32x4_t idx0 = vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(c))));
            uint32x4_t idx1 = vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(c))));

            // gather加载表值
            float32x4_t tab0 = vld1q_gather_f32(idx0, tables + m * 256);
            float32x4_t tab1 = vld1q_gather_f32(idx1, tables + (m + 2) * 256);

            sum = vaddq_f32(sum, tab0);
            sum = vaddq_f32(sum, tab1);
        }

        // 水平求和
        output[i] = vaddvq_f32(sum);
    }
}

// NEON SIMD友好的gather模拟
inline float32x4_t neon_gather_f32(
        const uint32x4_t indices,
        const float* base) {

    // 手动gather（ARMv8没有直接的gather指令）
    float arr[4];
    vst1q_u32((uint32_t*)arr, indices);

    return vld1q_f32(base + arr[0]);  // 简化版
}
#endif
```

### 9.3 Cache预取优化

```cpp
// 跨平台预取策略
namespace prefetch_optimization {

// 软件预取
#ifdef __GNUC__
    #define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
    #define PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1, 3)
#elif defined(_MSC_VER)
    #define PREFETCH(addr) _mm_prefetch((char*)(addr), _MM_HINT_T0)
    #define PREFETCH_WRITE(addr) _mm_prefetch((char*)(addr), _MM_HINT_T0)
#else
    #define PREFETCH(addr)
    #define PREFETCH_WRITE(addr)
#endif

// 预取优化的批量距离计算
void prefetch_batch_distance(
        const float* queries,
        const float* database,
        size_t nq,
        size_t nb,
        int d,
        float* distances) {

    const size_t prefetch_distance = 8;  // 预取前8个向量

    for (size_t q = 0; q < nq; q++) {
        const float* query = queries + q * d;

        for (size_t b = 0; b < nb; b++) {
            // 预取未来的数据库向量
            if (b + prefetch_distance < nb) {
                PREFETCH(database + (b + prefetch_distance) * d);
            }

            // 计算距离
            float dist = 0;
            size_t i = 0;

#if defined(__AVX2__)
            __m256 sum = _mm256_setzero_ps();
            for (; i + 8 <= d; i += 8) {
                __m256 vx = _mm256_loadu_ps(query + i);
                __m256 vy = _mm256_loadu_ps(database + b * d + i);
                __m256 diff = _mm256_sub_ps(vx, vy);
                sum = _mm256_fmadd_ps(diff, diff, sum);
            }

            alignas(32) float tmp[8];
            _mm256_storeu_ps(tmp, sum);
            dist = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                   tmp[4] + tmp[5] + tmp[6] + tmp[7];
#endif

            for (; i < d; i++) {
                float tmp = query[i] - database[b * d + i];
                dist += tmp * tmp;
            }

            distances[q * nb + b] = dist;
        }
    }
}

// NT_store：绕过缓存（大数据量写入）
void non_temporal_store(
        const float* __restrict__ src,
        float* __restrict__ dst,
        size_t n) {

    size_t i = 0;

#if defined(__AVX2__)
    for (; i + 32 <= n; i += 32) {
        __m256 v0 = _mm256_loadu_ps(src + i);
        __m256 v1 = _mm256_loadu_ps(src + i + 8);
        __m256 v2 = _mm256_loadu_ps(src + i + 16);
        __m256 v3 = _mm256_loadu_ps(src + i + 24);

        // 非时序存储：不污染缓存
        _mm256_stream_ps(dst + i, v0);
        _mm256_stream_ps(dst + i + 8, v1);
        _mm256_stream_ps(dst + i + 16, v2);
        _mm256_stream_ps(dst + i + 24, v3);
    }

    // sfence：确保写完成
    _mm_sfence();
#endif

    // 处理剩余
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}
}
```

### 9.4 汇编级优化

```cpp
// 当编译器优化不足时，使用内联汇编
#ifdef __x86_64__
    #ifdef __AVX2__

// 汇编优化的L2距离（避免编译器生成次优代码）
inline float asm_l2_distance_avx2(
        const float* x,
        const float* y,
        size_t d) {

    float result;
    size_t i = 0;

    // AVX2主循环
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0 \n\t"  // sum = 0
        "1: \n\t"
        "vmovups (%[x]), %%ymm1 \n\t"         // 加载x
        "vmovups (%[y]), %%ymm2 \n\t"         // 加载y
        "vsubps %%ymm2, %%ymm1, %%ymm3 \n\t"  // diff = x - y
        "vmulps %%ymm3, %%ymm3, %%ymm3 \n\t"  // diff * diff
        "vaddps %%ymm3, %%ymm0, %%ymm0 \n\t"  // sum += sq
        "add $32, %[x] \n\t"
        "add $32, %[y] \n\t"
        "sub $8, %[cnt] \n\t"
        "jnz 1b \n\t"

        // 水平求和
        "vextractf128 $1, %%ymm0, %%xmm1 \n\t"
        "vaddps %%xmm1, %%xmm0, %%xmm0 \n\t"
        "vshufps $0xB1, %%xmm0, %%xmm0, %%xmm1 \n\t"
        "vaddps %%xmm1, %%xmm0, %%xmm0 \n\t"
        "vshufps $0x4E, %%xmm0, %%xmm0, %%xmm1 \n\t"
        "vaddps %%xmm1, %%xmm0, %%xmm0 \n\t"
        "vmovss %%xmm0, %[out] \n\t"

        : [out] "=m" (result),
          [x] "+r" (x),
          [y] "+r" (y),
          [cnt] "+r" (d)
        :
        : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "memory", "cc"
    );

    return result;
}

    #endif // __AVX2__
#endif

// ARM NEON汇编优化
#if defined(__aarch64__) && defined(__ARM_NEON)

inline float asm_dot_product_neon(
        const float* x,
        const float* y,
        size_t d) {

    float result;
    size_t i = 0;

    __asm__ __volatile__(
        "movi v0.4s, #0 \n\t"           // sum = 0

        "1: \n\t"
        "ld1 {v1.4s}, [%[x]], #16 \n\t" // 加载4个float
        "ld1 {v2.4s}, [%[y]], #16 \n\t"
        "fmla v0.4s, v1.4s, v2.4s \n\t"  // sum += x * y (FMA)
        "subs %[cnt], %[cnt], #4 \n\t"
        "b.ne 1b \n\t"

        // 水平求和
        "faddp v0.4s, v0.4s, v0.4s \n\t"  // 两两相加
        "faddp v0.2s, v0.2s, v0.2s \n\t"  // 最后相加
        "fmov %w[out], v0.s[0] \n\t"

        : [out] "=r" (result),
          [x] "+r" (x),
          [y] "+r" (y),
          [cnt] "+r" (d)
        :
        : "v0", "v1", "v2", "memory", "cc"
    );

    return result;
}

#endif
```

### 9.5 分支预测优化

```cpp
// 分支预测提示加速
namespace branch_optimization {

// 使用likely/unlikely提示编译器
float optimized_distance_with_check(
        const float* x,
        const float* y,
        size_t d) {

    if (FAISS_LIKELY(d >= 8)) {
        // 主路径：热循环
#if defined(__AVX2__)
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;

        for (; i + 32 <= d; i += 32) {
            __m256 x0 = _mm256_loadu_ps(x + i);
            __m256 y0 = _mm256_loadu_ps(y + i);
            __m256 d0 = _mm256_sub_ps(x0, y0);
            sum = _mm256_fmadd_ps(d0, d0, sum);

            __m256 x1 = _mm256_loadu_ps(x + i + 8);
            __m256 y1 = _mm256_loadu_ps(y + i + 8);
            __m256 d1 = _mm256_sub_ps(x1, y1);
            sum = _mm256_fmadd_ps(d1, d1, sum);

            __m256 x2 = _mm256_loadu_ps(x + i + 16);
            __m256 y2 = _mm256_loadu_ps(y + i + 16);
            __m256 d2 = _mm256_sub_ps(x2, y2);
            sum = _mm256_fmadd_ps(d2, d2, sum);

            __m256 x3 = _mm256_loadu_ps(x + i + 24);
            __m256 y3 = _mm256_loadu_ps(y + i + 24);
            __m256 d3 = _mm256_sub_ps(x3, y3);
            sum = _mm256_fmadd_ps(d3, d3, sum);
        }

        for (; i + 8 <= d; i += 8) {
            __m256 xv = _mm256_loadu_ps(x + i);
            __m256 yv = _mm256_loadu_ps(y + i);
            __m256 diff = _mm256_sub_ps(xv, yv);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        float result = _mm256_reduce_add_ps(sum);

        for (; i < d; i++) {
            float tmp = x[i] - y[i];
            result += tmp * tmp;
        }

        return result;
#endif
    }

    // 冷路径：小向量
    float result = 0;
    for (size_t i = 0; i < d; i++) {
        float tmp = x[i] - y[i];
        result += tmp * tmp;
    }
    return result;
}

// 无分支计算：使用条件选择代替if
inline float branchless_min(float a, float b) {
#if defined(__AVX2__)
    __m256 va = _mm256_set1_ps(a);
    __m256 vb = _mm256_set1_ps(b);
    __m256 cmp = _mm256_min_ps(va, vb);
    return _mm256_cvtss_f32(cmp);
#else
    return std::min(a, b);
#endif
}

// 无分支clamp
inline float branchless_clamp(float x, float min_val, float max_val) {
#if defined(__AVX2__)
    __m256 vx = _mm256_set1_ps(x);
    __m256 vmin = _mm256_set1_ps(min_val);
    __m256 vmax = _mm256_set1_ps(max_val);

    __m256 clamped = _mm256_min_ps(
        _mm256_max_ps(vx, vmin),
        vmax);

    return _mm256_cvtss_f32(clamped);
#else
    return std::min(std::max(x, min_val), max_val);
#endif
}
}
```

### 9.6 Loop unrolling优化

```cpp
// 手动展开提升ILP
namespace loop_unrolling {

// 4路展开的L2距离
float unrolled_l2_distance_4x(
        const float* x,
        const float* y,
        size_t d) {

#if defined(__AVX2__)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    size_t i = 0;

    // 4路展开：每次处理32个float
    for (; i + 32 <= d; i += 32) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 y0 = _mm256_loadu_ps(y + i);
        __m256 d0 = _mm256_sub_ps(x0, y0);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);

        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 y1 = _mm256_loadu_ps(y + i + 8);
        __m256 d1 = _mm256_sub_ps(x1, y1);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);

        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 y2 = _mm256_loadu_ps(y + i + 16);
        __m256 d2 = _mm256_sub_ps(x2, y2);
        sum2 = _mm256_fmadd_ps(d2, d2, sum2);

        __m256 x3 = _mm256_loadu_ps(x + i + 24);
        __m256 y3 = _mm256_loadu_ps(y + i + 24);
        __m256 d3 = _mm256_sub_ps(x3, y3);
        sum3 = _mm256_fmadd_ps(d3, d3, sum3);
    }

    // 合并累加器
    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1),
                               _mm256_add_ps(sum2, sum3));

    float result = _mm256_reduce_add_ps(sum);

    for (; i < d; i++) {
        float tmp = x[i] - y[i];
        result += tmp * tmp;
    }

    return result;
#else
    float result = 0;
    for (size_t i = 0; i < d; i++) {
        float tmp = x[i] - y[i];
        result += tmp * tmp;
    }
    return result;
#endif
}

// 8路展开（适合超宽流水线）
float unrolled_l2_distance_8x(
        const float* x,
        const float* y,
        size_t d) {

#if defined(__AVX2__)
    __m256 sum[8];
    for (int j = 0; j < 8; j++) {
        sum[j] = _mm256_setzero_ps();
    }

    size_t i = 0;

    for (; i + 64 <= d; i += 64) {
        for (int j = 0; j < 8; j++) {
            __m256 xv = _mm256_loadu_ps(x + i + j * 8);
            __m256 yv = _mm256_loadu_ps(y + i + j * 8);
            __m256 diff = _mm256_sub_ps(xv, yv);
            sum[j] = _mm256_fmadd_ps(diff, diff, sum[j]);
        }
    }

    // 树形归约求和
    __m256 total = _mm256_setzero_ps();
    for (int j = 0; j < 8; j++) {
        total = _mm256_add_ps(total, sum[j]);
    }

    float result = _mm256_reduce_add_ps(total);

    for (; i < d; i++) {
        float tmp = x[i] - y[i];
        result += tmp * tmp;
    }

    return result;
#else
    float result = 0;
    for (size_t i = 0; i < d; i++) {
        float tmp = x[i] - y[i];
        result += tmp * tmp;
    }
    return result;
#endif
}
}
```

---

## 10. AVX-512高级优化技术

### 10.1 VPCLMUL指令 - 向量点积加速

```cpp
#ifdef __AVX512VNNI__
// AVX-512 VNNI: Vector Neural Network Instructions
// 包含vpdpbusd: 8-bit点积累加到32-bit

// 混合精度点积计算（用于RaBitQ等量化方法）
void avx512_mixed_precision_dot_product(
        const int8_t* x,        // int8向量
        const int8_t* y,        // int8向量
        int32_t* output,       // int32结果
        size_t d) {

    size_t i = 0;

    // AVX-512 VNNI：一次处理64个int8，累加到16个int32
    for (; i + 64 <= d; i += 64) {
        // 加载64个int8
        __m512i vx = _mm512_loadu_si512(x + i);
        __m512i vy = _mm512_loadu_si512(y + i);

        // 零点累加器
        __m512i vsum = _mm512_setzero_si512();

        // vpdpbusd: 8-bit点积，每4对int8产生1个int32
        // 一次计算16个int32点积（64个int8 / 4）
        vsum = _mm512_dpbusd_epi32(vsum, vx, vy);

        // 存储结果
        _mm512_storeu_si512(output + (i / 4), vsum);
    }

    // 处理剩余元素
    for (i = (d / 64) * 64; i < d; i++) {
        int32_t prod = (int32_t)x[i] * (int32_t)y[i];
        output[i / 4] += prod;
    }
}

// RaBitQ专用的4-bit点积（使用AVX-512）
void avx512_4bit_dot_product(
        const uint8_t* x,       // 4-bit编码（每字节2个4-bit码）
        const uint8_t* y,       // 4-bit编码
        float* cent_table_lut,  // 查找表
        float* output,
        size_t d) {

    size_t i = 0;
    __m512 sum = _mm512_setzero_ps();

    // 一次处理32个4-bit码（16字节）
    for (; i + 16 <= d; i += 16) {
        // 加载16字节（32个4-bit码）
        __m128i packed = _mm_loadu_si128((__m128i*)(x + i / 2));

        // 解包4-bit码
        __m512i unpacked = unpack_4bit_to_8bit(packed);

        // 使用掩码从质心表gather
        // （实际实现会使用更复杂的查表）
        sum = lookup_and_accumulate_4bit(unpacked, cent_table_lut, sum);
    }

    float result = _mm512_reduce_add_ps(sum);
    output[0] = result;
}
#endif
```

### 10.2 AVX-512冲突检测与压缩存储

```cpp
#ifdef __AVX512F__
// 冲突检测：用于去重和集合操作
class AVX512ConflictDetection {
public:
    // 检测16个ID中是否有重复
    static inline bool has_duplicate_ids(const idx_t* ids) {
        __m512i vids = _mm512_loadu_si512(ids);

        // 创建所有对组合的比较
        // 简化版本：只检查相邻元素
        for (int i = 0; i < 15; i++) {
            __m512i shifted = _mm512_alignr_epi32(
                vids, _mm512_set1_epi32(0), 15 - i);
            __mmask16 eq = _mm512_cmpeq_epi32_mask(
                vids, shifted);

            if (eq != 0) {
                return true;  // 有重复
            }
        }
        return false;
    }

    // 去重（保持顺序）
    static void unique_ids(idx_t* ids, size_t* n) {
        if (*n <= 1) return;

        size_t write_idx = 1;
        __m512i prev_ids = _mm512_loadu_si512(ids);

        for (size_t i = 1; i < *n; i++) {
            __m512i curr_ids = _mm512_loadu_si512(ids + i);
            __mmask16 not_equal = _mm512_cmpneq_epi32_mask(
                prev_ids, curr_ids);

            if (not_equal) {
                // 唯一：写入
                if (write_idx < i) {
                    ids[write_idx++] = ids[i];
                }
                prev_ids = _mm512_permutevar_ps32(
                    curr_ids, _mm512_set1_epi32(15));
            }
        }

        *n = write_idx;
    }
};

// 压缩存储：只写入有效元素
class AVX512CompressedStore {
public:
    // 条件存储：只写入非负距离
    static inline void store_valid_distances(
            const __m512 distances,
            const __mmask16 valid_mask,
            float* output) {

        // 只写入有效元素
        _mm512_mask_storeu_ps(output, valid_mask, distances);
    }

    // 用于搜索结果的后处理
    static void postprocess_search_results(
            float* distances,
            idx_t* labels,
            size_t k) {

        // 标记无效结果
        __mmask16 valid_mask = 0xFFFF;
        for (size_t i = 0; i < k; i += 16) {
            __m512i vlabels = _mm512_loadu_si512(labels + i);
            __mmask16 is_valid = _mm512_cmpneq_epi32_mask(
                vlabels, _mm512_set1_epi32(-1));

            // 压缩存储有效距离
            __m512 vdist = _mm512_loadu_ps(distances + i);
            _mm512_mask_storeu_ps(distances + i, is_valid, vdist);

            valid_mask &= is_valid;
        }
    }
};
```

### 10.3 AVX-512字符串处理（二进制索引）

```cpp
#ifdef __XHOSTSETUP__
#include <immintrin.h>
#include <avx512fintrin.h>

// Hamming距离的AVX-512优化
class AVX512HammingDistance {
public:
    // 计算两个512位向量的汉明距离
    static inline int hamming_512bit(const uint8_t* a, const uint8_t* b) {
        // 加载512位（64字节）
        __m512i va = _mm512_loadu_si512(a);
        __m512i vb = _mm512_loadu_si512(b);

        // XOR找不同位
        __m512i vxor = _mm512_xor_si512(va, vb);

        // 使用VPOPCNTDQ指令直接计算popcount！
        #ifdef __AVX512VPOPCNTDQ__
            return _mm512_popcnt_epi64(vxor);
        #else
            // 后备方案：查表法
            return popcount_fallback(vxor);
        #endif
    }

    // 批量汉明距离计算（用于二进制索引）
    static void batch_hamming_distance(
            const uint8_t* database,
            const uint8_t* query,
            size_t n,
            uint8_t* distances) {

        size_t i = 0;

        // 一次处理512位（64字节）
        for (; i + 64 <= n; i += 64) {
            __m512i vdb = _mm512_loadu_si512(database + i);
            __m512i vq = _mm512_loadu_si512(query);

            __m512i vxor = _mm512_xor_si512(vdb, vq);

            // popcount
            int dist;
            #ifdef __AVX512VPOPCNTDQ__
                dist = _mm512_popcnt_epi64(vxor);
            #else
                dist = popcount_fallback(vxor);
            #endif

            distances[i / 64] = static_cast<uint8_t>(dist);
        }

        // 处理剩余字节
        for (i = (n / 64) * 64; i < n; i++) {
            distances[i] = __builtin_popcount(database[i] ^ query[i]);
        }
    }

private:
    // 查表法popcount
    static int popcount_fallback(__m512i v) {
        alignas(64) uint8_t bytes[64];
        _mm512_storeu_si512(bytes, v);

        int count = 0;
        for (int i = 0; i < 64; i++) {
            count += __builtin_popcount(bytes[i]);
        }
        return count;
    }
};

// 二进制索引的批量搜索
class BinaryIndexSearch {
public:
    static void search_binary_index(
            const uint8_t* query,
            const uint8_t* database,
            size_t n,
            size_t code_size,  // 每个向量字节数
            size_t k,
            uint32_t* results) {

        // 暂时存储所有距离
        std::vector<uint8_t> all_distances(n);

        for (size_t i = 0; i < n; i++) {
            // 计算汉明距离
            int dist = 0;
            for (size_t j = 0; j < code_size; j += 64) {
                size_t chunk_size = std::min(size_t(64), code_size - j);
                dist += AVX512HammingDistance::hamming_512bit(
                    query + j, database + i * code_size + j);
            }
            all_distances[i] = static_cast<uint8_t>(dist);
        }

        // 找top-K（使用partial_sort）
        std::partial_sort(
            all_distances.begin(),
            all_distances.begin() + k,
            all_distances.end());

        // 复制结果
        for (size_t i = 0; i < k; i++) {
            results[i] = all_distances[i];
        }
    }
};
#endif
```

### 10.4 AVX-512BF16 - BF16计算优化

```cpp
#ifdef __AVX512BF16__
// BF16（Brain Float 16）- 深度学习常用格式
class AVX512BF16Operations {
public:
    // FP32转BF16
    static inline void fp32_to_bf16(
            const float* fp32_data,
            uint16_t* bf16_data,
            size_t n) {

        size_t i = 0;

        // 一次转换32个float（16个BF16）
        for (; i + 32 <= n; i += 32) {
            __m512 vfp32 = _mm512_loadu_ps(fp32_data + i);

            // 转换为BF16
            __m512h vbf16 = _mm512_cvtne2ps_pbh(vfp32);

            // 存储BF16
            _mm512_storeu_si512((__m512i*)(bf16_data + i), vbf16);
        }

        // 处理剩余元素
        for (; i < n; i++) {
            bf16_data[i] = float_to_bf16(fp32_data[i]);
        }
    }

    // BF16点积（累加到FP32）
    static inline float bf16_dot_product_accumulate(
            const uint16_t* x_bf16,
            const uint16_t* y_bf16,
            size_t n) {

        __m512 sum = _mm512_setzero_ps();
        size_t i = 0;

        // 一次处理32个BF16（16个乘积）
        for (; i + 32 <= n; i += 32) {
            // 加载BF16
            __m512h vx = _mm512_loadu_epi16(x_bf16 + i);
            __m512h vy = _mm512_loadu_epi16(y_bf16 + i);

            // BF16乘法+累加到FP32
            sum = _mm512_dpbf16_ps(sum, vx, vy);
        }

        // 处理剩余元素
        for (; i < n; i++) {
            float fx = bf16_to_float(x_bf16[i]);
            float fy = bf16_to_float(y_bf16[i]);
            sum = _mm512_add_ps(sum, _mm512_set1_ps(fx * fy));
        }

        return _mm512_reduce_add_ps(sum);
    }

    // 批量BF16点积（用于混合精度向量搜索）
    static void batch_bf16_dot_products(
            const uint16_t* database_bf16,
            const uint16_t* query_bf16,
            float* similarities,
            size_t nb,
            size_t d) {

        for (size_t i = 0; i < nb; i++) {
            similarities[i] = bf16_dot_product_accumulate(
                query_bf16,
                database_bf16 + i * d,
                d);
        }
    }

private:
    static inline uint16_t float_to_bf16(float f) {
        uint16_t bits;
        std::memcpy(&bits, &f, sizeof(float));
        // 提取符号位+指数+尾数的高16位
        return bits >> 16;
    }

    static inline float bf16_to_float(uint16_t bf) {
        uint32_t bits = bf << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        return f;
    }
};
#endif
```

### 10.5 AVX-512CD - 加速向量距离计算

```cpp
#ifdef __AVX512CD__
// AVX-512 CD: Conflict Detection Instructions
class AVX512VectorDistance {
public:
    // 并行计算多个L2距离（单指令多数据）
    static void vectorized_l2_distances_avx512cd(
            const float* query,       // [d]
            const float* database,    // [n x d] 列优先存储
            float* distances,       // [n]
            size_t n,
            size_t d) {

        // 假设d是16的倍数（便于展开）
        constexpr size_t UNROLL = 4;

        for (size_t i = 0; i < n; i += 16) {
            // 同时处理16个向量
            __m512 sum[UNROLL];
            for (int j = 0; j < UNROLL; j++) {
                sum[j] = _mm512_setzero_ps();
            }

            const float* db_vec = database + i * d;
            size_t j = 0;

            // 主循环：4路展开
            for (; j + 64 <= d; j += 64) {
                for (int k = 0; k < UNROLL; k++) {
                    size_t offset = j + k * 16;

                    // 加载查询向量（广播）
                    __m512 vq0 = _mm512_loadu_ps(query + offset);

                    // 加载16个数据库向量
                    __m512 vd0 = _mm512_loadu_ps(db_vec + offset);
                    __m512 vd1 = _mm512_loadu_ps(db_vec + offset + d);
                    __m512 vd2 = _mm512_loadu_ps(db_vec + offset + d * 2);
                    // ... 加载更多向量

                    __m512 diff0 = _mm512_sub_ps(vq0, vd0);
                    __m512 diff1 = _mm512_sub_ps(vq0, vd1);
                    __m512 diff2 = _mm512_sub_ps(vq0, vd2);

                    sum[0] = _mm512_fmadd_ps(diff0, diff0, sum[0]);
                    sum[1] = _mm512_fmadd_ps(diff1, diff1, sum[1]);
                    sum[2] = _mm512_fmadd_ps(diff2, diff2, sum[2]);
                    // ... 更多累加器
                }
            }

            // 合并累加器
            __m512 total = _mm512_add_ps(
                _mm512_add_ps(sum[0], sum[1]),
                _mm512_add_ps(sum[2], sum[3]));

            // 水平求和
            distances[i] = _mm512_reduce_add_ps(total);

            // 处理剩余元素
            for (; j < d; j++) {
                float diff = query[j] - db_vec[j];
                distances[i] += diff * diff;
            }
        }
    }
};
#endif
```

### 10.6 AVX-512融合乘加优化

```cpp
#ifdef __AVX512F__
// FMA（Fused Multiply-Add）的性能优势
class AVX512FMAOptimization {
public:
    // FMA链优化：减少延迟
    static inline __m512 fma_chain_optimized(
            __m512 x,
            __m512 y,
            __m512 z,
            __m512 w) {

        // 标准FMA：a*b+c
        // result = x*y + z*w

        // 方法1：链式FMA
        // temp = x*y
        // result = temp + z*w
        // 延迟：3+3=6周期

        // 方法2：并行FMA（依赖分离）
        // result = fma(x, y, fma(z, w, 0))
        // 延迟：~3周期（如果有两个FMA单元）

        __m512 temp = _mm512_mul_ps(x, y);           // 5周期
        __m512 temp2 = _mm512_mul_ps(z, w);          // 5周期
        __m512 result = _mm512_add_ps(temp, temp2);  // 3周期
        // 总延迟：13周期

        // 优化版本：使用两个FMA单元
        // result = fma(x, y, fma(z, w, 0))
        __m512 result_opt = _mm512_fmadd_ps(
            x, y,
            _mm512_fmadd_ps(z, w, _mm512_setzero_ps())
        );
        // 总延迟：~6-7周期

        return result_opt;
    }

    // L2距离的FMA优化实现
    static float l2_distance_fma_optimized(
            const float* x,
            const float* y,
            size_t d) {

        __m512 sum = _mm512_setzero_ps();
        size_t i = 0;

        // 预取下一轮数据
        for (; i + 32 <= d; i += 32) {
            _mm_prefetch((const char*)(x + i + 32), _MM_HINT_T0);
            _mm_prefetch((const char*)(y + i + 32), _MM_HINT_T0);

            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vy = _mm512_loadu_ps(y + i);

            __m512 diff = _mm512_sub_ps(vx, vy);

            // FMA计算平方和
            sum = _mm512_fmadd_ps(diff, diff, sum);
        }

        float result = _mm512_reduce_add_ps(sum);

        // 处理剩余元素
        for (; i < d; i++) {
            float diff = x[i] - y[i];
            result += diff * diff;
        }

        return result;
    }

    // 内积的FMA优化
    static float inner_product_fma(
            const float* x,
            const float* y,
            size_t d) {

        __m512 sum = _mm512_setzero_ps();
        size_t i = 0;

        for (; i + 16 <= d; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vy = _mm512_loadu_ps(y + i);

            // FMA: x*y + 0 = x*y
            sum = _mm512_fmadd_ps(vx, vy, sum);
        }

        float result = _mm512_reduce_add_ps(sum);

        for (; i < d; i++) {
            result += x[i] * y[i];
        }

        return result;
    }
};
#endif
```

---

## 11. AVX-512性能调优工具

### 11.1 IACA（Intel Architecture Code Analyzer）

```bash
# IACA可以分析汇编代码的微架构性能
# 下载：https://software.intel.com/en-us/articles/intel-sde

# 分析AVX-512代码的微操作
# 编译时添加-g保留调试信息
g++ -O3 -mavx512f -g -S avx_code.cpp -o avx_code.s

# 使用IACA分析
iaca -64 avx_code.s

# IACA输出关键指标：
# - Port pressure：端口压力（0-1表示良好）
# - Latency：指令延迟
# - Throughput：吞吐量
# - Dependency chains：依赖链
```

### 11.2 性能监控工具

```cpp
// Intel VTune性能分析器集成
#ifdef __INTEL_COMPILER
    #include <ittnotify.h>
#endif

class VTuneProfiler {
public:
    static void initialize() {
        #ifdef __INTEL_COMPILER
            // 初始化ITT通知
            __ittnest_begin(itt_domain);
        #endif
    }

    static void profile_search(const char* name) {
        #ifdef __INTEL_COMPILER
            __itt_task_begin(itt_handle);
            __itt_string_handle_create(name, __itt_null);
        #endif
    }

    static void end_profile() {
        #ifdef __INTEL_COMPILER
            __itt_task_end(itt_handle);
        #endif
    }

    // 使用示例
    static void profiled_search() {
        profile_search("AVX512_L2_Search");

        // 执行搜索
        // ...

        end_profile();
    }
};
```

### 11.3 性能对比基准

```cpp
// AVX-512 vs AVX2 vs 标量性能对比
class AVX512Benchmark {
public:
    struct BenchmarkResult {
        double scalar_time_us;
        double avx2_time_us;
        double avx512_time_us;
        double avx512_speedup;
        double avx2_speedup;
    };

    static BenchmarkResult benchmark_all_implementations(
            const float* x,
            const float* y,
            size_t d,
            size_t n) {

        BenchmarkResult result;

        // 标量版本
        result.scalar_time_us = measure_time([&]() {
            volatile float sum = 0;
            for (size_t i = 0; i < n; i++) {
                sum += scalar_l2(x + i * d, y + i * d, d);
            }
        });

        #ifdef __AVX2__
        result.avx2_time_us = measure_time([&]() {
            volatile float sum = 0;
            for (size_t i = 0; i < n; i++) {
                sum += avx2_l2(x + i * d, y + i * d, d);
            }
        });
        #endif

        #ifdef __AVX512F__
        result.avx512_time_us = measure_time([&]() {
            volatile float sum = 0;
            for (size_t i = 0; i < n; i++) {
                sum += avx512_l2(x + i * d, y + i * d, d);
            }
        });
        #endif

        result.avx2_speedup = result.scalar_time_us / result.avx2_time_us;
        result.avx512_speedup = result.scalar_time_us / result.avx512_time_us;

        return result;
    }

    static void print_results(const BenchmarkResult& r) {
        printf("\n=== SIMD Performance Comparison ===\n");
        printf("Scalar:    %.2f us (1.0x)\n", r.scalar_time_us);
        printf("AVX2:      %.2f us (%.2fx)\n", r.avx2_time_us, r.avx2_speedup);
        printf("AVX-512:   %.2f us (%.2fx)\n", r.avx512_time_us, r.avx512_speedup);
    }
};
```

---

## 12. 第16天总结

### 关键概念

1. **AVX2**: 256位SIMD，8个float，FMA支持
2. **AVX-512**: 512位SIMD，掩码操作，gather/scatter
3. **ARM NEON**: 128位SIMD，查表指令
4. **ARM SVE**: 可变长度SIMD，谓词执行
5. **跨平台抽象**: 统一接口，多实现
6. **高级技巧**: 预取、分支预测、循环展开

### SIMD优化技术层级

| 层级 | 技术 | 示例 | 性能提升 |
|------|------|------|----------|
| **基础** | 向量化 | SIMD算术操作 | 4-8x |
| **中级** | 缓存优化 | 预取、对齐 | 1.5-3x |
| **高级** | ILP优化 | 循环展开、多累加器 | 1.2-2x |
| **专家** | 汇编优化 | 内联汇编、微调 | 1.1-1.5x |

### 性能对比

| 指令集 | 寄存器位宽 | Float数 | 理论加速 | 实际加速 |
|--------|----------|---------|----------|----------|
| Scalar | - | 1 | 1x | 1x |
| SSE | 128 | 4 | 4x | 3-4x |
| AVX2 | 256 | 8 | 8x | 6-8x |
| AVX-512 | 512 | 16 | 16x | 12-16x |
| NEON | 128 | 4 | 4x | 3-4x |
| SVE-256 | 256 | 8 | 8x | 6-8x |
| SVE-512 | 512 | 16 | 16x | 12-16x |

### 最佳实践

1. **优先使用编译器intrinsics**而非汇编
2. **确保内存对齐**以获得最佳性能
3. **使用软件预取**减少缓存未命中
4. **循环展开**提升指令级并行
5. **分支预测提示**指导编译器优化
6. **多累加器**减少数据依赖
7. **非时序存储**避免缓存污染

### 下一步

第17天将学习**高级调试与性能分析**，学习如何使用profiler工具分析和优化Faiss性能。

---

## 练习题

1. 实现一个跨平台的SIMD距离计算函数
2. 比较不同SIMD指令集的性能差异
3. 编写ARM NEON版本的查表操作
4. 实现运行时CPU特性检测和dispatch
5. **实现AVX-512掩码优化的距离计算**
6. **编写带预取优化的批量距离计算**
7. **实现4路展开的SIMD循环**

## 扩展阅读

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Faiss SIMD优化源码](https://github.com/facebookresearch/faiss/tree/main/faiss/utils)
- [SIMD性能优化指南](https://www.agner.org/optimize/optimizing_cpp.pdf)
