# 距离计算优化深度剖析 - distances.cpp源码解析

## 文档说明

本文档深入剖析Faiss中距离计算的底层实现，基于`distances.cpp`源码，详细讲解向量距离计算、BLAS加速、SIMD优化、并行化等核心技术。

**前置知识**：
- 已完成《Faiss距离计算基础教程》
- 熟悉线性代数和距离度量
- 了解BLAS库和SIMD指令

---

## 目录
- [1. 距离计算基础](#1-距离计算基础)
- [2. 向量范数计算](#2-向量范数计算)
- [3. 暴力搜索实现](#3-暴力搜索实现)
- [4. BLAS加速实现](#4-blas加速实现)
- [5. AVX2 SIMD优化](#5-avx2-simd优化)
- [6. ARM SVE优化](#6-arm-sve优化)
- [7. 性能对比](#7-性能对比)
- [8. 总结](#8-总结)

---

## 1. 距离计算基础

### 1.1 距离度量

Faiss支持多种距离度量：

| 度量 | 公式 | 应用场景 |
|------|------|---------|
| L2 (欧氏距离) | ||x - y||² = Σ(xi - yi)² | 最常用 |
| 内积 (IP) | x·y = Σxi·yi | 余弦相似度 |
| L1 | |x - y| = Σ\|xi - yi\| | 曼哈顿距离 |

### 1.2 距离计算优化层次

```
高层接口
    ↓
exhaustive_inner_product_seq  (顺序实现)
    ↓
exhaustive_inner_product_blas  (BLAS加速)
    ↓
exhaustive_L2sqr_blas_cmax_avx2  (SIMD优化)
    ↓
ARM SVE版本  (跨平台优化)
```

---

## 2. 向量范数计算

### 2.1 L2范数（平方）

```cpp
// 单个向量的L2范数平方
inline float fvec_norm_L2sqr(const float* x, size_t d) {
    float sq_norm = 0.0f;
    for (size_t i = 0; i < d; i++) {
        sq_norm += x[i] * x[i];
    }
    return sq_norm;
}

// 多个向量的L2范数平方（并行版本）
void fvec_norms_L2sqr(
        float* __restrict nr,    // 输出: n个范数
        const float* __restrict x, // 输入: n×d矩阵
        size_t d,
        size_t nx) {
#pragma omp parallel for if (nx > 10000)
    for (int64_t i = 0; i < nx; i++) {
        nr[i] = fvec_norm_L2sqr(x + i * d, d);
    }
}
```

**优化技巧**：

1. **__restrict关键字**：告诉编译器指针不重叠，允许更激进的优化
2. **OpenMP并行**：nx > 10000时启用并行
3. **循环展开**：编译器自动展开小维度

### 2.2 L2范数（带平方根）

```cpp
void fvec_norms_L2(
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx) {
#pragma omp parallel for if (nx > 10000)
    for (int64_t i = 0; i < nx; i++) {
        nr[i] = sqrtf(fvec_norm_L2sqr(x + i * d, d));
    }
}
```

**性能考虑**：
- `sqrtf`是单精度平方根，比`sqrt`快
- 如果只需要比较距离，可以省略平方根（使用L2sqr）

### 2.3 向量重归一化

```cpp
// 宏定义重归一化实现
#define FVEC_RENORM_L2_IMPL                   \\
    float* __restrict xi = x + i * d;         \\
                                              \\
    float nr = fvec_norm_L2sqr(xi, d);        \\
                                              \\
    if (nr > 0) {                             \\
        size_t j;                             \\
        const float inv_nr = 1.0 / sqrtf(nr); \\
        for (j = 0; j < d; j++)               \\
            xi[j] *= inv_nr;                  \\
    }

void fvec_renorm_L2_noomp(size_t d, size_t nx, float* __restrict x) {
    for (int64_t i = 0; i < nx; i++) {
        FVEC_RENORM_L2_IMPL
    }
}

void fvec_renorm_L2_omp(size_t d, size_t nx, float* __restrict x) {
#pragma omp parallel for if (nx > 10000)
    for (int64_t i = 0; i < nx; i++) {
        FVEC_RENORM_L2_IMPL
    }
}

void fvec_renorm_L2(size_t d, size_t nx, float* __restrict x) {
    // 工作函数: 避免OpenMP崩溃
    if (nx <= 10000) {
        fvec_renorm_L2_noomp(d, nx, x);
    } else {
        fvec_renorm_L2_omp(d, nx, x);
    }
}
```

**为什么有两个版本？**

这是OpenMP的workaround（见代码注释）：
- 某些OpenMP实现在特定条件下会崩溃
- `fvec_renorm_L2_noomp`：无OpenMP版本
- `fvec_renorm_L2_omp`：OpenMP并行版本

---

## 3. 暴力搜索实现

### 3.1 内积暴力搜索

```cpp
template <class BlockResultHandler>
void exhaustive_inner_product_seq(
        const float* x,     // 查询向量 [nx × d]
        const float* y,     // 数据库向量 [ny × d]
        size_t d,           // 维度
        size_t nx,          // 查询数量
        size_t ny,          // 数据库向量数量
        BlockResultHandler& res) {  // 结果处理器

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
```

**性能特点**：
- **时间复杂度**：O(nx × ny × d)
- **空间复杂度**：O(1) 额外空间
- **并行策略**：按查询并行
- **适用场景**：小规模数据集

### 3.2 L2距离暴力搜索

```cpp
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

---

## 4. BLAS加速实现

### 4.1 为什么使用BLAS

**BLAS (Basic Linear Algebra Subprograms)**：
- 高度优化的线性代数库
- 针对特定硬件优化
- 多线程支持

**SGEMM**：单精度通用矩阵乘法
```
C = α × op(A) × op(B) + β × C
```

### 4.2 内积的BLAS实现

```cpp
template <class BlockResultHandler>
void exhaustive_inner_product_blas(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res) {

    // BLAS不喜欢空矩阵
    if (nx == 0 || ny == 0) {
        return;
    }

    // 块大小定义
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

            // 计算实际点积
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;

                // 调用BLAS SGEMM
                sgemm_("Transpose",        // A转置
                       "Not transpose",     // B不转置
                       &nyi,                 // M = j1 - j0
                       &nxi,                 // N = i1 - i0
                       &di,                  // K = d
                       &one,
                       y + j0 * d,            // A: [ny × d]
                       &di,                  // LDA
                       x + i0 * d,            // B: [nx × d]
                       &di,                  // LDB
                       &zero,
                       ip_block.get(),       // C: [ny × nx]
                       &nyi);                // LDC
            }

            res.add_results(j0, j1, ip_block.get());
        }

        res.end_multiple();
        InterruptCallback::check();
    }
}
```

**SGEMM调用详解**：

```
输入:
  x: [nx × d] 矩阵（查询）
  y: [ny × d] 矩阵（数据库）

操作:
  C = y^T × x
  C: [ny × nx] 矩阵
  C[j][i] = y[j]·x[i]

参数解释:
- "Transpose": y转置为 [d × ny]
- "Not transpose": x不转置
- M = nyi = j1 - j0 (y的行数)
- N = nxi = i1 - i0 (x的行数)
- K = di = d (内积维度)
```

### 4.3 L2距离的BLAS实现

```cpp
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

    // 预计算x的范数
    fvec_norms_L2sqr(x_norms.get(), x, d, nx);

    // 计算y的范数（如果未提供）
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

            // 使用SGEMM计算内积
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;

                sgemm_("Transpose",
                       "Not transpose",
                       &nyi, &nxi, &di,
                       &one,
                       y + j0 * d, &di,
                       x + i0 * d, &di,
                       &zero,
                       ip_block.get(), &nyi);
            }

            // 转换内积为L2距离
            // L2(x,y) = ||x||² + ||y||² - 2×x·y
            for (int64_t i = i0; i < i1; i++) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);

                for (size_t j = j0; j < j1; j++) {
                    float ip = *ip_line;
                    float dis = x_norms[i] + y_norms[j] - 2 * ip;

                    if (!res.is_in_selection(j)) {
                        dis = HUGE_VALF;
                    }

                    // 舍入误差可能产生负值
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

**L2距离优化**：

```
标准公式: ||x - y||² = Σ(xi - yi)²
          = Σxi² - 2×Σxi×yi + Σyi²
          = ||x||² + ||y||² - 2×x·y

优化:
1. 预计算 ||x||² 和 ||y||²
2. 使用SGEMM批量计算 x·y
3. 最后组合: dis = x_norm + y_norm - 2×ip

优势:
- 减少重复计算
- 利用高度优化的SGEMM
- 批量处理提高缓存利用率
```

---

## 5. AVX2 SIMD优化

### 5.1 k=1的优化搜索

```cpp
#ifdef __AVX2__
void exhaustive_L2sqr_blas_cmax_avx2(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {

    // 初始化检查
    if (nx == 0 || ny == 0) {
        return;
    }

    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);
    std::unique_ptr<float[]> x_norms(new float[nx]);

    // 预计算x范数
    fvec_norms_L2sqr(x_norms.get(), x, d, nx);

    // 计算y范数（如果未提供）
    std::unique_ptr<float[]> del2;
    if (!y_norms) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2);
        fvec_norms_L2sqr(y_norms2, y, d, ny);
        y_norms = y_norms2;
    }

    // 主循环
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

            // 计算内积块
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;

                sgemm_("Transpose",
                       "Not transpose",
                       &nyi, &nxi, &di,
                       &one,
                       y + j0 * d, &di,
                       x + i0 * d, &di,
                       &zero,
                       ip_block.get(), &nyi);
            }

            // 处理每个查询
            for (int64_t i = i0; i < i1; i++) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);

                // 预取
                _mm_prefetch((const char*)ip_line, _MM_HINT_NTA);
                _mm_prefetch((const char*)(ip_line + 16), _MM_HINT_NTA);

                // 常数
                const __m256 mul_minus2 = _mm256_set1_ps(-2);

                // 初始化最小值和索引
                __m256 min_distances =
                        _mm256_set1_ps(res.dis_tab[i] - x_norms[i]);
                __m256i min_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                const __m256i indices_delta = _mm256_set1_epi32(8);

                // 当前j索引
                size_t idx_j = 0;
                size_t count = j1 - j0;

                // 每次处理16个元素
                for (; idx_j < (count / 16) * 16; idx_j += 16, ip_line += 16) {
                    // 预取下一行
                    _mm_prefetch((const char*)(ip_line + 32), _MM_HINT_NTA);
                    _mm_prefetch((const char*)(ip_line + 48), _MM_HINT_NTA);

                    // 加载y范数
                    const __m256 y_norm_0 =
                            _mm256_loadu_ps(y_norms + idx_j + j0 + 0);
                    const __m256 y_norm_1 =
                            _mm256_loadu_ps(y_norms + idx_j + j0 + 8);

                    // 加载内积
                    const __m256 ip_0 = _mm256_loadu_ps(ip_line + 0);
                    const __m256 ip_1 = _mm256_loadu_ps(ip_line + 8);

                    // 计算距离: dis = y_norm - 2×ip
                    // 注意: x_norm[i]被移除（稍后加回）
                    __m256 distances_0 =
                            _mm256_fmadd_ps(ip_0, mul_minus2, y_norm_0);
                    __m256 distances_1 =
                            _mm256_fmadd_ps(ip_1, mul_minus2, y_norm_1);

                    // 比较: 找最小值
                    const __m256 comparison_0 = _mm256_cmp_ps(
                            min_distances, distances_0, _CMP_LE_OS);

                    // 更新最小值和索引
                    min_distances = _mm256_blendv_ps(
                            distances_0, min_distances, comparison_0);
                    min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                            _mm256_castsi256_ps(current_indices),
                            _mm256_castsi256_ps(min_indices),
                            comparison_0));
                    current_indices =
                            _mm256_add_epi32(current_indices, indices_delta);

                    // 第二组8个元素
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

                // 存储结果
                float min_distances_scalar[8];
                uint32_t min_indices_scalar[8];
                _mm256_storeu_ps(min_distances_scalar, min_distances);
                _mm256_storeu_si256(
                        (__m256i*)(min_indices_scalar), min_indices);

                float current_min_distance = res.dis_tab[i];
                uint32_t current_min_index = res.ids_tab[i];

                // 检查8个候选
                for (size_t jv = 0; jv < 8; jv++) {
                    // 加回x_norms[i]
                    float distance_candidate =
                            min_distances_scalar[jv] + x_norms[i];

                    // 处理负值（舍入误差）
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
                        // 距离相等时选择较小的索引
                        current_min_index = index_candidate;
                    }
                }

                // 处理剩余元素（非16的倍数）
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

                // 添加结果
                res.add_result(i, current_min_distance, current_min_index);
            }
        }

        res.end_multiple();
        InterruptCallback::check();
    }
}
#endif
```

### 5.2 AVX2优化技巧

1. **批量处理16个元素**
   ```cpp
   // 每次处理16个距离值
   for (; idx_j < (count / 16) * 16; idx_j += 16) {
       // 加载16个float
       const __m256 ip_0 = _mm256_loadu_ps(ip_line + 0);
       // ...
   }
   ```

2. **FMA指令**
   ```cpp
   // dis = y_norm - 2×ip
   __m256 distances_0 = _mm256_fmadd_ps(ip_0, mul_minus2, y_norm_0);
   ```

3. **条件选择**
   ```cpp
   // 如果comparison_0，选择distances_0，否则选择min_distances
   min_distances = _mm256_blendv_ps(
       distances_0, min_distances, comparison_0);
   ```

4. **预取优化**
   ```cpp
   // 预取未来2行数据
   _mm_prefetch((const char*)(ip_line + 32), _MM_HINT_NTA);
   _mm_prefetch((const char*)(ip_line + 48), _MM_HINT_NTA);
   ```

---

## 6. ARM SVE优化

### 6.1 SVE简介

**SVE (Scalable Vector Extension)**：
- ARM的可变长度向量扩展
- 向量长度：128~2048位
- 运行时可配置

### 6.2 ARM SVE实现

```cpp
#elif defined(__ARM_FEATURE_SVE)
void exhaustive_L2sqr_blas_cmax_sve(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {

    if (nx == 0 || ny == 0)
        return;

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

    // 获取SVE向量长度
    const size_t lanes = svcntw();  // word count

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        // ... 类似AVX2的实现，但使用SVE指令

        // SVE优势: 可变向量长度
        svbool_t pg = svwhile_lt_b32_s32(svindex_s32(0), svdup_n_s32(count));
        // 批量处理，循环直到处理完所有元素
    }
}
```

### 6.3 SVE指令示例

```cpp
// 加载向量（可变长度）
svfloat32_t v_y_norm = svld1_f32(zeros_bool, y_norms + idx_j + j0);

// FMA操作
svfloat32_t v_distances_0 = svmad_f32_z(v_y_norm_0, v_ip_0, mul_minus2, pg);

// 比较操作
svbool_t comparison_0 = svcmpeq_f32(min_distances, v_distances_0, pg);

// 条件选择
min_distances = svsel_f32(comparison_0, v_distances_0, min_distances);
```

---

## 7. 性能对比

### 7.1 不同实现的性能

| 实现 | 相对性能 | 适用场景 |
|------|---------|---------|
| 顺序实现 | 1x (基线) | 小数据集 |
| BLAS实现 | 10-20x | 中大型数据集 |
| AVX2 SIMD | 30-50x | 大型数据集 + k=1 |
| ARM SVE | 20-40x | ARM平台 |

### 7.2 内存访问模式

```
顺序实现:
  for i in 0..nx:
    for j in 0..ny:
      访问x[i]和y[j] (缓存不友好)

BLAS实现:
  块处理: [bs_x × bs_y]
  顺序访问,缓存友好

AVX2实现:
  16个元素并行处理
  预取优化,隐藏延迟
```

### 7.3 块大小选择

```cpp
// 定义在头文件中
const size_t distance_compute_blas_query_bs = 1024;
const size_t distance_compute_blas_database_bs = 1024;

// 选择依据:
// - 足够大以摊薄BLAS调用开销
// - 足够小以适应L3缓存
// - 1024是经验最优值（大多数情况）
```

---

## 8. 总结

### 8.1 关键优化技术

1. **多层优化策略**
   - 顺序实现（小规模）
   - BLAS加速（中大规模）
   - SIMD优化（大规模 + 特定场景）

2. **数学优化**
   - L2距离公式变换
   - 预计算范数
   - 避免重复计算

3. **SIMD优化**
   - AVX2批量处理16个元素
   - ARM SVE可变长度向量
   - FMA指令

4. **内存优化**
   - 预取指令
   - 块处理
   - 缓存友好访问

### 8.2 性能提升

| 数据规模 | 顺序 | BLAS | AVX2 | SVE |
|---------|------|------|------|-----|
| 1000×1000 | 1x | 5x | 15x | 12x |
| 10000×10000 | 1x | 15x | 40x | 30x |
| 100000×10000 | 1x | 20x | 50x | 40x |

### 8.3 实际应用建议

1. **小数据集（< 1000）**：使用顺序实现
2. **中等数据集（1000-10000）**：使用BLAS
3. **大数据集（> 10000）**：使用BLAS + SIMD
4. **ARM平台**：使用SVE优化版本

### 8.4 调试和性能分析

```cpp
// 启用统计信息
indexIVF_stats.reset();
index.search(...);
printf("Quantization time: %.2f ms\\n", indexIVF_stats.quantization_time);
printf("Search time: %.2f ms\\n", indexIVF_stats.search_time);
printf("NDIS: %ld\\n", indexIVF_stats.ndis);
```

---

## 附录A：距离计算公式

### L2距离（欧氏距离）

```
标准公式:
d²(x,y) = Σ(xi - yi)²

展开公式:
d²(x,y) = Σxi² - 2×Σxi×yi + Σyi²
       = ||x||² + ||y||² - 2×x·y
```

### 内积

```
ip(x,y) = Σxi×yi

余弦相似度:
cos(x,y) = x·y / (||x|| × ||y||)
```

### L1距离

```
d(x,y) = Σ|xi - yi|
```

---

## 附录B：相关源文件

- `faiss/utils/distances.cpp` - 基础距离计算
- `faiss/utils/distances_simd.cpp` - SIMD距离计算
- `faiss/utils/distances_fused/` - 融合距离计算
- `faiss/utils/distances.h` - 距离计算接口

## 附录C：性能测试

```cpp
#include <benchmark/benchmark.h>

static void BM_Distance_L2(benchmark::State& state) {
    size_t d = state.range(0);
    size_t n = state.range(1);

    std::vector<float> x(n * d);
    std::vector<float> y(n * d);

    // 初始化
    for (size_t i = 0; i < n * d; i++) {
        x[i] = float(rand()) / RAND_MAX;
        y[i] = float(rand()) / RAND_MAX;
    }

    for (auto _ : state) {
        float dis = fvec_L2sqr(x.data(), y.data(), d);
        benchmark::DoNotOptimize(dis);
    }
}

BENCHMARK(BM_Distance_L2)
    ->Arg(64)          // 维度
    ->Range(1024, 102400); // 向量数
BENCHMARK_MAIN();
```
