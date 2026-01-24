# ResidualQuantizer 残差量化器底层实现深度剖析 - ResidualQuantizer.cpp 源码解析

## 1. 概述

ResidualQuantizer (RQ) 是 Faiss 中一种先进的加性量化器，通过**渐进式残差量化**（Progressive Residual Quantization）实现高精度的向量压缩。与 Product Quantizer (PQ) 不同，RQ 通过**叠加多个量化层的残差**来重建向量，而不是拼接子向量。

### 核心特性
- **渐进式编码**: 每层量化前一层的残差
- **Beam Search**: 使用束搜索寻找最优编码路径
- **动态码本**: 每层独立训练自己的码本
- **LUT 优化**: 支持查找表加速距离计算

## 2. 数据结构设计

### 2.1 核心数据结构 (ResidualQuantizer.h)

```cpp
struct ResidualQuantizer : AdditiveQuantizer {
    // 训练模式标志位
    train_type_t train_type = Train_progressive_dim;

    static const int Train_default = 0;           // 普通 k-means
    static const int Train_progressive_dim = 1;   // 渐进式维度聚类
    static const int Train_refine_codebook = 2;   // 码本精化迭代
    static const int Train_top_beam = 1024;       // 仅在 beam 顶部训练

    // Beam Search 参数
    int max_beam_size = 5;                        // 最大 beam 大小
    int use_beam_LUT = 0;                         // LUT 模式选择

    // 聚类参数
    ProgressiveDimClusteringParameters cp;        // 渐进式维度聚类参数
    ProgressiveDimIndexFactory* assign_index_factory;  // 分配索引工厂

    // 近似 topk 模式
    ApproxTopK_mode_t approx_topk_mode = ApproxTopK_mode_t::EXACT_TOPK;
};
```

### 2.2 AdditiveQuantizer 基类 (AdditiveQuantizer.h)

```cpp
struct AdditiveQuantizer : Quantizer {
    size_t M;                     // 码本数量（层数）
    std::vector<size_t> nbits;    // 每层的 bit 数
    std::vector<float> codebooks; // 所有码本数据

    // 码本偏移量
    std::vector<uint64_t> codebook_offsets;

    // 辅助数据结构
    std::vector<float> centroid_norms;            // 质心范数
    std::vector<float> codebook_cross_products;   // 码本叉积
    std::vector<float> norm_tabs;                 // 范数表

    // 搜索模式
    enum Search_type_t {
        ST_decompress,      // 解压缩数据库向量
        ST_LUT_nonorm,      // LUT 无范数
        ST_norm_float,      // LUT + float32 范数
        ST_norm_qint8,      // LUT + 8-bit 范数量化
        ST_norm_rq2x4,      // LUT + 2x4 bits RQ 范数
    };
};
```

### 2.3 内存池设计

#### LUT0 模式内存池 (residual_quantizer_encode_steps.h:139-145)

```cpp
struct ComputeCodesAddCentroidsLUT0MemoryPool {
    std::vector<int32_t> codes;          // 编码缓冲区
    std::vector<float> norms;            // 范数缓冲区
    std::vector<float> distances;        // 距离缓冲区
    std::vector<float> residuals;        // 残差缓冲区
    RefineBeamMemoryPool refine_beam_pool; // Beam 搜索池
};
```

#### LUT1 模式内存池 (residual_quantizer_encode_steps.h:156-163)

```cpp
struct ComputeCodesAddCentroidsLUT1MemoryPool {
    std::vector<int32_t> codes;
    std::vector<float> distances;
    std::vector<float> query_norms;      // 查询向量范数
    std::vector<float> query_cp;         // 查询-码本内积
    std::vector<float> residuals;
    RefineBeamLUTMemoryPool refine_beam_lut_pool;
};
```

## 3. Beam Search 核心算法

### 3.1 单步编码实现 (residual_quantizer_encode_steps.cpp:228-379)

Beam search 在每一步将 beam_size 个候选扩展到 new_beam_size 个候选：

```cpp
void beam_search_encode_step(
        size_t d,
        size_t K,                      // 当前层的码本大小 (2^nbits)
        const float* cent,             // 当前层的码本 (K x d)
        size_t n,
        size_t beam_size,              // 输入 beam 大小
        const float* residuals,        // 输入残差 (n x beam_size x d)
        size_t m,                      // 之前的层数
        const int32_t* codes,          // 之前的编码 (n x beam_size x m)
        size_t new_beam_size,          // 输出 beam 大小 (<= beam_size * K)
        int32_t* new_codes,            // 新编码 (n x new_beam_size x (m+1))
        float* new_residuals,          // 新残差 (n x new_beam_size x d)
        float* new_distances,          // 新距离 (n x new_beam_size)
        Index* assign_index,
        ApproxTopK_mode_t approx_topk_mode)
{
    // 扩展限制
    FAISS_THROW_IF_NOT(new_beam_size <= beam_size * K);

    std::vector<float> cent_distances;
    std::vector<idx_t> cent_ids;

    // 分支 1: 使用索引进行分配
    if (assign_index) {
        cent_distances.resize(n * beam_size * new_beam_size);
        cent_ids.resize(n * beam_size * new_beam_size);

        if (assign_index->ntotal == 0) {
            assign_index->add(K, cent);
        }

        // 搜索每个 beam 候选的最近 new_beam_size 个质心
        assign_index->search(
                n * beam_size,
                residuals,
                new_beam_size,
                cent_distances.data(),
                cent_ids.data());
    }
    // 分支 2: 直接计算所有距离
    else {
        cent_distances.resize(n * beam_size * K);
        pairwise_L2sqr(
                d, n * beam_size, residuals, K, cent,
                cent_distances.data());
    }

    // 并行处理每个向量
#pragma omp parallel for if (n > 100)
    for (int64_t i = 0; i < n; i++) {
        const int32_t* codes_i = codes + i * m * beam_size;
        int32_t* new_codes_i = new_codes + i * (m + 1) * new_beam_size;
        const float* residuals_i = residuals + i * d * beam_size;
        float* new_residuals_i = new_residuals + i * d * new_beam_size;
        float* new_distances_i = new_distances + i * new_beam_size;

        using C = CMax<float, int>;

        // 初始化堆
        for (int j = 0; j < new_beam_size; j++) {
            new_distances_i[j] = C::neutral();
        }
        std::vector<int> perm(new_beam_size, -1);

        // 使用近似 topk 或精确堆
        if (assign_index) {
            // 合并已排序的数组
            heap_addn<C>(
                    new_beam_size,
                    new_beam_size,
                    new_distances_i,
                    perm.data(),
                    cent_distances_i,
                    nullptr,
                    beam_size * new_beam_size);
        } else {
            // 从 beam_size * K 个候选中选择 best new_beam_size
            switch (approx_topk_mode) {
                case ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B8_D3:
                    HeapWithBuckets<C, 8, 3>::bs_addn(...);
                    break;
                case ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B16_D2:
                    HeapWithBuckets<C, 16, 2>::bs_addn(...);
                    break;
                default:
                    heap_addn<C>(
                            new_beam_size,
                            new_distances_i,
                            perm.data(),
                            cent_distances_i,
                            nullptr,
                            beam_size * K);
            }
        }

        heap_reorder<C>(new_beam_size, new_distances_i, perm.data());

        // 根据 permutation 构造新的 beam
        for (int j = 0; j < new_beam_size; j++) {
            int js = perm[j] / K;    // 源 beam 索引
            int ls = perm[j] % K;    // 选中的质心索引

            // 复制之前的编码
            if (m > 0) {
                memcpy(new_codes_i, codes_i + js * m, sizeof(*codes) * m);
            }
            new_codes_i[m] = ls;     // 添加当前层的编码
            new_codes_i += m + 1;

            // 计算新残差: residual - centroid
            fvec_sub(
                    d,
                    residuals_i + js * d,
                    cent + ls * d,
                    new_residuals_i);
            new_residuals_i += d;
        }
    }
}
```

### 3.2 Beam Search 流程图

```
Input:  x (原始向量)
        beam_size = 1
        M 层量化, nbits = [4, 4, 4] (每层 16 个质心)

Step 0:
    beam: [x]
    距离码本0的所有质心，选择最近的 new_beam_size 个
    beam expands to: [x + c_0_0, x + c_0_1, ..., x + c_0_beam]
    residuals: [x - c_0_0, x - c_0_1, ..., x - c_0_beam]

Step 1:
    beam: [r_0, r_1, ..., r_{beam-1}] (前一层的残差)
    对每个 beam 候选，计算其与码本1所有质心的距离
    总候选数: beam_size * K
    从中选择 best new_beam_size 个
    更新残差: residual - centroid

Step 2:
    ... (继续扩展)

Output: 优化后的编码路径 + 最小量化误差
```

## 4. SIMD 优化实现

### 4.1 模板化的累加函数 (residual_quantizer_encode_steps.cpp:46-220)

使用模板元编程实现编译时优化的 SIMD 累加：

#### accum_and_store_tab - 存储累加结果

```cpp
template <size_t M, size_t NK>
void accum_and_store_tab(
        const size_t m_offset,
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        float* const __restrict output)
{
    // 加载指针到寄存器
    const float* cbs[M];
    for (size_t ij = 0; ij < M; ij++) {
        const size_t code = static_cast<size_t>(codes_i[b * m_offset + ij]);
        cbs[ij] = &codebook_cross_norms[(codebook_offsets[ij] + code) * ldc];
    }

#if defined(__AVX2__) || defined(__aarch64__)
    const size_t K8 = (K / (8 * NK)) * (8 * NK);

    // 按 (8 * NK) 大小的块处理
    for (size_t kk = 0; kk < K8; kk += 8 * NK) {
        simd8float32 regs[NK];
        for (size_t ik = 0; ik < NK; ik++) {
            regs[ik].loadu(cbs[0] + kk + ik * 8);
        }

        // 累加其他码本的贡献
        for (size_t ij = 1; ij < M; ij++) {
            for (size_t ik = 0; ik < NK; ik++) {
                regs[ik] += simd8float32(cbs[ij] + kk + ik * 8);
            }
        }

        // 写入结果
        for (size_t ik = 0; ik < NK; ik++) {
            regs[ik].storeu(output + kk + ik * 8);
        }
    }
#else
    const size_t K8 = 0;
#endif

    // 处理剩余元素
    for (size_t kk = K8; kk < K; kk++) {
        float reg = cbs[0][kk];
        for (size_t ij = 1; ij < M; ij++) {
            reg += cbs[ij][kk];
        }
        output[kk] = reg;
    }
}
```

#### accum_and_finalize_tab - 完成距离计算

```cpp
template <size_t M, size_t NK>
void accum_and_finalize_tab(
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        const float* const __restrict distances_i,
        const float* const __restrict cd_common,
        float* const __restrict output)
{
    const float* cbs[M];
    for (size_t ij = 0; ij < M; ij++) {
        const size_t code = static_cast<size_t>(codes_i[b * M + ij]);
        cbs[ij] = &codebook_cross_norms[(codebook_offsets[ij] + code) * ldc];
    }

#if defined(__AVX2__) || defined(__aarch64__)
    const size_t K8 = (K / (8 * NK)) * (8 * NK);

    for (size_t kk = 0; kk < K8; kk += 8 * NK) {
        simd8float32 regs[NK];
        for (size_t ik = 0; ik < NK; ik++) {
            regs[ik].loadu(cbs[0] + kk + ik * 8);
        }

        for (size_t ij = 1; ij < M; ij++) {
            for (size_t ik = 0; ik < NK; ik++) {
                regs[ik] += simd8float32(cbs[ij] + kk + ik * 8);
            }
        }

        simd8float32 two(2.0f);
        for (size_t ik = 0; ik < NK; ik++) {
            // cent_distances[b * K + k] = distances_i[b] + cd_common[k] + 2 * dp[k];
            simd8float32 common_v(cd_common + kk + ik * 8);
            common_v = fmadd(two, regs[ik], common_v);  // FMA 指令
            common_v += simd8float32(distances_i[b]);
            common_v.storeu(output + b * K + kk + ik * 8);
        }
    }
#endif

    // 处理剩余元素
    for (size_t kk = K8; kk < K; kk++) {
        float reg = cbs[0][kk];
        for (size_t ij = 1; ij < M; ij++) {
            reg += cbs[ij][kk];
        }
        output[b * K + kk] = distances_i[b] + cd_common[kk] + 2 * reg;
    }
}
```

### 4.2 SIMD 优化技术

| 技术 | 实现 | 性能提升 |
|------|------|---------|
| **寄存器阻塞** | 一次加载 M 个码本指针 | 减少内存访问 |
| **向量化加载** | `simd8float32::loadu()` | 8x 并行处理 |
| **FMA 融合** | `fmadd(two, regs[ik], common_v)` | 减少指令数 |
| **循环展开** | Template NK 参数 | 减少分支开销 |
| **剩余处理** | 标量循环处理尾部 | 完全覆盖所有 K |

### 4.3 基于层数的优化路径 (residual_quantizer_encode_steps.cpp:465-550)

```cpp
switch (m) {
    case 0:
        // 平凡情况：只需计算基础距离
        for (size_t b = 0; b < beam_size; b++) {
            for (size_t k = 0; k < K; k++) {
                cent_distances[b * K + k] = distances_i[b] + cd_common[k];
            }
        }
        break;

    ACCUM_AND_FINALIZE_TAB(1)  // m=1: 单层累加
    ACCUM_AND_FINALIZE_TAB(2)  // m=2: 两层累加
    ACCUM_AND_FINALIZE_TAB(3)  // m=3: 三层累加
    ACCUM_AND_FINALIZE_TAB(4)  // m=4: 四层累加
    ACCUM_AND_FINALIZE_TAB(5)
    ACCUM_AND_FINALIZE_TAB(6)
    ACCUM_AND_FINALIZE_TAB(7)

    default: {
        // m >= 8: 使用临时缓冲区，但仍然 8x 批量累加
        std::vector<float> dp(K);

        for (size_t b = 0; b < beam_size; b++) {
            // 先累加前 8 层
            accum_and_store_tab<8, 4>(
                    m, codebook_cross_norms, codebook_offsets,
                    codes_i, b, ldc, K, dp.data());

            // 继续累加剩余的层 (每次 8 层)
            for (size_t im = 8; im < ((m + 7) / 8) * 8; im += 8) {
                size_t m_left = std::min(m - im, size_t(8));
                switch (m_left) {
                    ACCUM_AND_ADD_TAB(1)
                    ACCUM_AND_ADD_TAB(2)
                    // ... up to 8
                }
            }

            // 最终化距离计算
            for (size_t k = 0; k < K; k++) {
                cent_distances[b * K + k] =
                        distances_i[b] + cd_common[k] + 2 * dp[k];
            }
        }
    }
}
```

## 5. 训练过程

### 5.1 渐进式训练 (ResidualQuantizer.cpp:131-288)

```cpp
void ResidualQuantizer::train(size_t n, const float* x) {
    codebooks.resize(d * codebook_offsets.back());

    int cur_beam_size = 1;
    std::vector<float> residuals(x, x + n * d);
    std::vector<int32_t> codes;

    double t0 = getmillisecs();

    // 逐层训练
    for (int m = 0; m < M; m++) {
        int K = 1 << nbits[m];  // 2^nbits 个质心

        // Beam 扩展: cur_beam_size * K -> new_beam_size
        int new_beam_size = std::min(cur_beam_size * K, max_beam_size);
        std::vector<int32_t> new_codes(n * new_beam_size * (m + 1));
        std::vector<float> new_residuals(n * new_beam_size * d);
        std::vector<float> new_distances(n * new_beam_size);

        // 内存管理: 根据可用内存动态调整 batch size
        size_t bs;
        {
            size_t mem = memory_per_point();
            if (n > 1 && mem * n > max_mem_distances) {
                bs = std::max(max_mem_distances / mem, size_t(1));
            } else {
                bs = n;
            }
        }

        // 批处理训练
        for (size_t i0 = 0; i0 < n; i0 += bs) {
            size_t i1 = std::min(i0 + bs, n);

            // 核心训练步骤
            if (!(train_type & Train_progressive_dim)) {
                // 普通模式: k-means 聚类
                Clustering clus(d, K, cp);
                clus.train(train_residuals.size() / d,
                         train_residuals.data(),
                         *assign_index.get());
            } else {
                // 渐进式维度聚类模式
                ProgressiveDimClustering clus(d, K, cp);
                clus.train(...);
            }

            // Beam search 编码步骤
            beam_search_encode_step(
                    d, K, codebooks.data(),
                    i1 - i0, cur_beam_size,
                    residuals.data() + i0 * cur_beam_size * d,
                    m, codes.data() + i0 * cur_beam_size * m,
                    new_beam_size,
                    new_codes.data() + i0 * new_beam_size * (m + 1),
                    new_residuals.data() + i0 * new_beam_size * d,
                    new_distances.data() + i0 * new_beam_size,
                    assign_index.get(),
                    approx_topk_mode);
        }

        // 交换缓冲区
        codes.swap(new_codes);
        residuals.swap(new_residuals);
        distances.swap(new_distances);

        cur_beam_size = new_beam_size;
    }

    is_trained = true;

    // 可选: 码本精化
    if (train_type & Train_refine_codebook) {
        for (int iter = 0; iter < niter_codebook_refine; iter++) {
            retrain_AQ_codebook(n, x);
        }
    }

    // 计算并训练范数量化器
    std::vector<float> norms(n);
    for (size_t i = 0; i < n; i++) {
        norms[i] = fvec_L2sqr(
                x + i * d,
                residuals.data() + i * cur_beam_size * d, d);
    }
    train_norm(n, norms.data());

    // 预计算码本表
    if (!(train_type & Skip_codebook_tables)) {
        compute_codebook_tables();
    }
}
```

### 5.2 码本精化 (ResidualQuantizer.cpp:290-405)

使用最小二乘法优化码本：

```cpp
float ResidualQuantizer::retrain_AQ_codebook(size_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n >= total_codebook_size, "too few training points");

    // 1. 编码所有训练向量
    std::vector<uint8_t> codes(n * code_size);
    compute_codes(x, codes.data(), n);

    // 2. 计算初始重建误差
    float input_recons_error;
    {
        std::vector<float> x_recons(n * d);
        decode(codes.data(), x_recons.data(), n);
        input_recons_error = fvec_L2sqr(x, x_recons.data(), n * d);
    }

    // 3. 构建线性系统矩阵 C
    // C[i + (codebook_offsets[m] + idx) * n] = 1 表示向量 i 选择了该码本条目
    std::vector<float> C(n * total_codebook_size);
    for (size_t i = 0; i < n; i++) {
        BitstringReader bsr(codes.data() + i * code_size, code_size);
        for (int m = 0; m < M; m++) {
            int idx = bsr.read(nbits[m]);
            C[i + (codebook_offsets[m] + idx) * n] = 1;
        }
    }

    // 4. 转置训练向量矩阵
    std::vector<float> xt(n * d);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            xt[j * n + i] = x[i * d + j];
        }
    }

    // 5. 使用 LAPACK 的 SGELSD 求解最小二乘问题
    // min ||C * X - xt||^2
    FINTEGER lwork = -1;
    FINTEGER di = d, ni = n, tcsi = total_codebook_size;
    float rcond = 1e-4;  // 秩判断阈值

    float worksize;
    std::vector<float> sing_vals(total_codebook_size);

    // 查询工作空间大小
    sgelsd_(&ni, &tcsi, &di, C.data(), &ni, xt.data(), &ni,
            sing_vals.data(), &rcond, &rank, &worksize, &lwork, ...);

    lwork = worksize;
    std::vector<float> work(lwork);

    // 实际求解
    sgelsd_(&ni, &tcsi, &di, C.data(), &ni, xt.data(), &ni,
            sing_vals.data(), &rcond, &rank, work.data(), &lwork, ...);

    // 6. 转置结果到码本
    for (size_t i = 0; i < total_codebook_size; i++) {
        for (size_t j = 0; j < d; j++) {
            codebooks[i * d + j] = xt[j * n + i];
        }
    }

    // 7. 计算输出重建误差
    float output_recons_error = 0;
    for (size_t j = 0; j < d; j++) {
        output_recons_error += fvec_norm_L2sqr(
                xt.data() + total_codebook_size + n * j,
                n - total_codebook_size);
    }

    return output_recons_error;
}
```

### 5.3 训练流程图

```
Input: 训练向量 X (n x d)
       M 层, nbits = [b1, b2, ..., bM]
       max_beam_size = B

For m = 0 to M-1:
    K = 2^bm

    // Step 1: 聚类
    if train_type == Train_default:
        使用标准 k-means 对残差聚类
    else:  // Train_progressive_dim
        使用渐进式维度聚类

    得到码本 cent_m (K x d)

    // Step 2: Beam Search 编码
    beam_search_encode_step(
        residuals: (n x cur_beam_size x d)
        -> new_residuals: (n x new_beam_size x d)
        codes: (n x cur_beam_size x m)
        -> new_codes: (n x new_beam_size x (m+1))
    )

    // Step 3: 更新 beam
    cur_beam_size = min(cur_beam_size * K, B)

    // Step 4: 更新残差
    residuals = new_residuals
    codes = new_codes

// Step 5: 码本精化 (可选)
For iter = 0 to niter_codebook_refine-1:
    编码所有训练向量
    求解最小二乘: min ||C * codebook - X||^2
    更新码本

// Step 6: 训练范数量化器
Compute norms = ||X - reconstructed_X||^2
Train norm quantizer on norms

// Step 7: 预计算 LUT 表
compute_codebook_tables()
```

## 6. LUT 优化编码模式

### 6.1 LUT0 vs LUT1 模式

| 特性 | LUT0 (use_beam_LUT=0) | LUT1 (use_beam_LUT=1) |
|------|----------------------|----------------------|
| **距离计算** | 直接 L2 距离 | 预计算内积 + 范数 |
| **内存需求** | 较低 | 较高 |
| **适用场景** | 小规模码本 | 大规模码本 |
| **预处理** | 无 | 需要叉积表 |

### 6.2 LUT1 编码实现 (residual_quantizer_encode_steps.cpp:900-956)

```cpp
void compute_codes_add_centroids_mp_lut1(
        const ResidualQuantizer& rq,
        const float* x,
        uint8_t* codes_out,
        size_t n,
        const float* centroids,
        ComputeCodesAddCentroidsLUT1MemoryPool& pool)
{
    // 分配缓冲区
    pool.codes.resize(rq.max_beam_size * rq.M * n);
    pool.distances.resize(rq.max_beam_size * n);

    FAISS_THROW_IF_NOT_MSG(
            rq.M == 1 || rq.codebook_cross_products.size() > 0,
            "call compute_codebook_tables first");

    // 1. 计算查询向量范数
    pool.query_norms.resize(n);
    fvec_norms_L2sqr(pool.query_norms.data(), x, rq.d, n);

    // 2. 计算查询-码本内积 (使用 BLAS SGEMM)
    pool.query_cp.resize(n * rq.total_codebook_size);
    {
        FINTEGER ti = rq.total_codebook_size, di = rq.d, ni = n;
        float zero = 0, one = 1;
        sgemm_("Transposed", "Not transposed",
               &ti, &ni, &di,
               &one, rq.codebooks.data(), &di,
               x, &di,
               &zero, pool.query_cp.data(), &ti);
    }

    // 3. 使用 LUT 进行 beam search
    refine_beam_LUT_mp(
            rq, n,
            pool.query_norms.data(),
            pool.query_cp.data(),
            rq.max_beam_size,
            pool.codes.data(),
            pool.distances.data(),
            pool.refine_beam_lut_pool);

    // 4. 打码
    rq.pack_codes(
            n, pool.codes.data(), codes_out,
            rq.M * rq.max_beam_size,
            nullptr, centroids);
}
```

### 6.3 LUT Beam Search (residual_quantizer_encode_steps.cpp:382-607)

```cpp
void beam_search_encode_step_tab(
        size_t K,
        size_t n,
        size_t beam_size,
        const float* codebook_cross_norms,  // 码本叉积
        size_t ldc,
        const uint64_t* codebook_offsets,
        const float* query_cp,               // 查询-码本内积
        size_t ldqc,
        const float* cent_norms_i,           // 质心范数
        size_t m,
        const int32_t* codes,
        const float* distances,
        size_t new_beam_size,
        int32_t* new_codes,
        float* new_distances,
        ApproxTopK_mode_t approx_topk_mode)
{
#pragma omp parallel for if (n > 100) schedule(dynamic)
    for (int64_t i = 0; i < n; i++) {
        std::vector<float> cent_distances(beam_size * K);
        std::vector<float> cd_common(K);

        const int32_t* codes_i = codes + i * m * beam_size;
        const float* query_cp_i = query_cp + i * ldqc;
        const float* distances_i = distances + i * beam_size;

        // 预计算公共部分: cent_norms_i[k] - 2 * query_cp_i[k]
        for (size_t k = 0; k < K; k++) {
            cd_common[k] = cent_norms_i[k] - 2 * query_cp_i[k];
        }

        // 优化的距离计算实现
        switch (m) {
            case 0:
                // 直接使用预计算的值
                for (size_t b = 0; b < beam_size; b++) {
                    for (size_t k = 0; k < K; k++) {
                        cent_distances[b * K + k] =
                                distances_i[b] + cd_common[k];
                    }
                }
                break;

            ACCUM_AND_FINALIZE_TAB(1)  // m=1
            ACCUM_AND_FINALIZE_TAB(2)  // m=2
            // ...
            ACCUM_AND_FINALIZE_TAB(7)  // m=7

            default:
                // m >= 8: 使用临时缓冲区
                std::vector<float> dp(K);
                for (size_t b = 0; b < beam_size; b++) {
                    accum_and_store_tab<8, 4>(...);
                    for (size_t im = 8; im < m; im += 8) {
                        accum_and_add_tab<...>(...);
                    }
                    for (size_t k = 0; k < K; k++) {
                        cent_distances[b * K + k] =
                                distances_i[b] + cd_common[k] + 2 * dp[k];
                    }
                }
        }

        // 使用堆选择最佳候选
        using C = CMax<float, int>;
        for (int j = 0; j < new_beam_size; j++) {
            new_distances_i[j] = C::neutral();
        }
        std::vector<int> perm(new_beam_size, -1);

        heap_addn<C>(new_beam_size, new_distances_i, perm.data(),
                     cent_distances_i, nullptr, beam_size * K);
        heap_reorder<C>(new_beam_size, new_distances_i, perm.data());

        // 构造新编码
        for (int j = 0; j < new_beam_size; j++) {
            int js = perm[j] / K;
            int ls = perm[j] % K;
            if (m > 0) {
                memcpy(new_codes_i, codes_i + js * m, sizeof(*codes) * m);
            }
            new_codes_i[m] = ls;
            new_codes_i += m + 1;
        }
    }
}
```

### 6.4 距离计算优化

LUT 模式下的距离公式：

```
L2(x, c_0 + c_1 + ... + c_m)
 = ||x||^2 - 2*<x, sum(c_i)> + ||sum(c_i)||^2
 = ||x||^2 - 2*sum(<x, c_i>) + sum(||c_i||^2) + 2*sum(<c_i, c_j>)

预计算:
- query_cp[k] = <x, cent_m[k]>
- cent_norms_i[k] = ||cent_m[k]||^2
- codebook_cross_norms[...] = <cent_i, cent_j>

运行时:
- cd_common[k] = cent_norms_i[k] - 2 * query_cp_i[k]
- 累加 codebook_cross_norms 到 dp[k]
- final_distance = distances_i[b] + cd_common[k] + 2 * dp[k]
```

## 7. 内存优化

### 7.1 内存预算 (ResidualQuantizer.cpp:407-416)

```cpp
size_t ResidualQuantizer::memory_per_point(int beam_size) const {
    if (beam_size < 0) {
        beam_size = max_beam_size;
    }
    size_t mem;
    // 两个 beam 的残差缓冲区
    mem = beam_size * d * 2 * sizeof(float);

    // beam search 结果: 距离 + 索引
    mem += beam_size * beam_size *
            (sizeof(float) + sizeof(idx_t));
    return mem;
}
```

### 7.2 批处理策略 (ResidualQuantizer.cpp:203-235)

```cpp
// 根据内存预算动态调整 batch size
size_t bs;
{
    size_t mem = memory_per_point();
    if (n > 1 && mem * n > max_mem_distances) {
        // 分批处理以减少临时内存
        bs = std::max(max_mem_distances / mem, size_t(1));
    } else {
        bs = n;  // 一次性处理所有向量
    }
}

for (size_t i0 = 0; i0 < n; i0 += bs) {
    size_t i1 = std::min(i0 + bs, n);

    // 处理批次 [i0, i1)
    beam_search_encode_step(...);

    // InterruptCallback 允许中断长时间运行的操作
    InterruptCallback::check();
}
```

### 7.3 内存池复用 (residual_quantizer_encode_steps.cpp:615-740)

```cpp
void refine_beam_mp(
        const ResidualQuantizer& rq,
        size_t n,
        size_t beam_size,
        const float* x,
        int out_beam_size,
        int32_t* out_codes,
        float* out_residuals,
        float* out_distances,
        RefineBeamMemoryPool& pool)
{
    // 1. 预分配最大所需内存
    int max_beam_size = 0;
    {
        int tmp_beam_size = cur_beam_size;
        for (int m = 0; m < rq.M; m++) {
            int K = 1 << rq.nbits[m];
            int new_beam_size = std::min(tmp_beam_size * K, out_beam_size);
            tmp_beam_size = new_beam_size;
            max_beam_size = std::max(max_beam_size, new_beam_size);
        }
    }

    // 2. 一次性分配所有缓冲区
    pool.new_codes.resize(n * max_beam_size * (rq.M + 1));
    pool.new_residuals.resize(n * max_beam_size * rq.d);
    pool.codes.resize(n * max_beam_size * (rq.M + 1));
    pool.distances.resize(n * max_beam_size);
    pool.residuals.resize(n * rq.d * max_beam_size);

    // 3. 使用指针交换避免拷贝
    int32_t* __restrict codes_ptr = pool.codes.data();
    float* __restrict residuals_ptr = pool.residuals.data();
    int32_t* __restrict new_codes_ptr = pool.new_codes.data();
    float* __restrict new_residuals_ptr = pool.new_residuals.data();

    // 4. 主循环: 只交换指针，不重新分配内存
    for (int m = 0; m < rq.M; m++) {
        int K = 1 << rq.nbits[m];
        int new_beam_size = std::min(cur_beam_size * K, out_beam_size);

        beam_search_encode_step(
                rq.d, K, codebooks_m, n, cur_beam_size,
                residuals_ptr, m, codes_ptr,
                new_beam_size,
                new_codes_ptr, new_residuals_ptr,
                pool.distances.data(), ...);

        // 交换指针 (O(1) 操作)
        std::swap(codes_ptr, new_codes_ptr);
        std::swap(residuals_ptr, new_residuals_ptr);

        cur_beam_size = new_beam_size;
    }

    // 5. 最终拷贝到输出
    if (out_codes) {
        memcpy(out_codes, codes_ptr, codes_size * sizeof(*codes_ptr));
    }
    if (out_residuals) {
        memcpy(out_residuals, residuals_ptr,
               residuals_size * sizeof(*residuals_ptr));
    }
}
```

## 8. 编码流程总结

### 8.1 完整编码流程

```
Input: 向量 x (d 维)

LUT0 模式 (use_beam_LUT = 0):
    1. 初始化 beam_size = 1, residuals = [x]
    2. For m = 0 to M-1:
        a. 计算所有 beam 候选与码本 m 的距离
        b. 使用 heap 选择 best new_beam_size 候选
        c. 更新 residuals = residuals - selected_centroids
        d. 更新 beam_size = min(beam_size * K, max_beam_size)
    3. 从 beam 中选择最佳编码
    4. 计算范数: norm = ||x - reconstructed||^2
    5. 打包编码 + 范数

LUT1 模式 (use_beam_LUT = 1):
    1. 预计算:
        - query_norms = ||x||^2
        - query_cp = <x, codebooks>  (SGEMM)
    2. 初始化 beam_size = 1, distances = [||x||^2]
    3. For m = 0 to M-1:
        a. 使用 LUT 计算距离:
           dist = distances[b] + cent_norms[k] - 2*query_cp[k]
                  + 2*accumulated_codebook_dot_products
        b. 使用 heap 选择 best new_beam_size
        c. 更新 beam_size
    4. 打包编码
```

### 8.2 优化技术总结

| 层级 | 优化技术 | 收益 |
|------|---------|------|
| **算法** | Beam Search | 平衡精度与计算量 |
| **内存** | 池复用 + 指针交换 | 减少分配/拷贝开销 |
| **SIMD** | 向量化累加 | 8x 并行处理 |
| **FMA** | `fmadd()` 指令 | 减少延迟 |
| **BLAS** | SGEMM 内积计算 | 利用高度优化的 BLAS |
| **并行** | OpenMP 并行处理 | 多核加速 |
| **批处理** | 动态 batch size | 内存受控 |
| **近似** | HeapWithBuckets | 加速 topk 选择 |

## 9. 性能特征

### 9.1 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| **训练 (单层)** | O(T * n * K * d) | O(n * beam_size * d) |
| **编码 (单步)** | O(beam_size * K * d) | O(new_beam_size * (m+1) * d) |
| **编码 (全部)** | O(M * B^2 * d) | O(B * M * d) |
| **解码** | O(M * d) | O(d) |
| **LUT 编码** | O(beam_size * K * M) | O(new_beam_size * (m+1)) |

其中:
- `T`: k-means 迭代次数
- `K`: 码本大小 (2^nbits)
- `B`: max_beam_size
- `M`: 量化层数
- `d`: 向量维度

### 9.2 参数调优建议

| 参数 | 推荐值 | 影响 |
|------|-------|------|
| `max_beam_size` | 5-10 | 更大 = 更高精度，但更高计算量 |
| `nbits` | 4-8 | 更大 = 更高精度，但更大内存 |
| `M` | 2-8 | 更多层 = 更高精度 |
| `use_beam_LUT` | 0 (小规模), 1 (大规模) | LUT1 需要预计算 |

## 10. 与其他量化器的比较

| 特性 | RQ | PQ | SQ |
|------|-----|----|----|
| **编码方式** | 叠加残差 | 拼接子向量 | 标量量化 |
| **精度** | 高 | 中 | 低 |
| **内存** | 中 | 低 | 最低 |
| **编码速度** | 慢 (beam search) | 快 | 最快 |
| **解码速度** | 中 | 快 | 快 |
| **灵活性** | 高 (每层不同 nbits) | 低 (固定 M) | 中 |

## 11. 关键源码位置

| 文件 | 关键函数 | 行号 |
|------|---------|------|
| `ResidualQuantizer.cpp` | `train()` | 131-288 |
| `ResidualQuantizer.cpp` | `retrain_AQ_codebook()` | 290-405 |
| `residual_quantizer_encode_steps.cpp` | `beam_search_encode_step()` | 228-379 |
| `residual_quantizer_encode_steps.cpp` | `beam_search_encode_step_tab()` | 382-607 |
| `residual_quantizer_encode_steps.cpp` | `refine_beam_mp()` | 615-740 |
| `residual_quantizer_encode_steps.cpp` | `accum_and_finalize_tab()` | 158-220 |
| `residual_quantizer_encode_steps.cpp` | `compute_codes_add_centroids_mp_lut1()` | 900-956 |
