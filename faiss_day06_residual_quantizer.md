# Faiss深度课程 - 第6天：加性量化器 - ResidualQuantizer与AdditiveQuantizer

## 课程目标

深入理解加性量化器（Additive Quantizer）和残差量化器（ResidualQuantizer），这些是比Product Quantization更强大的量化方法。

---

## 1. 加性量化器概述

### 1.1 与PQ的区别

```cpp
// Product Quantizer (PQ)
// x ≈ [c_m0] || [c_m1] || ... || [c_mM]  // 拼接
// 维度：d = dsub * M

// Additive Quantizer (AQ)
// x ≈ c_0 + c_1 + ... + c_M  // 求和
// 每个c_m来自独立的码本
```

**关键区别**：

| 特性 | PQ | AQ |
|------|----|----|
| 组合方式 | 拼接 | 求和 |
| 码本维度 | d/M | d |
| 灵活性 | 受限 | 高 |
| 精度 | 较好 | 更好 |

### 1.2 AdditiveQuantizer结构（底层实现）

```cpp
// faiss/impl/AdditiveQuantizer.h (完整版本)
struct AdditiveQuantizer : Quantizer {
    // 基本参数
    size_t d;       // 向量维度
    size_t M;       // 码本数量（层数）

    // 每个码本的位数（可变）
    std::vector<size_t> nbits;

    // 所有码本存储在一个大表中
    // 内存布局：codebooks[total_codebook_size * d]
    // codebook_offsets[m] = 码本m的起始索引
    std::vector<float> codebooks;

    // 码本偏移量
    std::vector<uint64_t> codebook_offsets;

    // 派生参数
    size_t tot_bits;             // 总位数（索引 + 范数）
    size_t norm_bits;            // 范数编码位数
    size_t total_codebook_size;  // 总质心数（sum(2^nbits[m])）
    bool only_8bit;              // 所有nbits都是8（优化解码）

    // 搜索类型（关键）
    enum Search_type_t {
        ST_decompress,         // 直接解压数据库向量
        ST_LUT_nonorm,         // 使用查找表，不包含范数（用于归一化向量）
        ST_norm_float,         // LUT + float32范数
        ST_norm_qint8,         // LUT + 8位量化范数
        ST_norm_qint4,         // LUT + 4位量化范数
        ST_norm_cqint8,        // LUT + 非均匀8位量化范数
        ST_norm_cqint4,        // LUT + 非均匀4位量化范数
        ST_norm_lsq2x4,        // 特殊：2x4位LSQ范数量化器
        ST_norm_rq2x4,         // 特殊：2x4位RQ范数量化器
    };
    Search_type_t search_type;

    // 范数范围（用于范数量化）
    float norm_min = NAN;
    float norm_max = NAN;

    // 范数量化的辅助数据
    std::vector<float> norm_tabs;       // 范数查找表
    IndexFlat1D qnorm;                  // 范数索引（用于搜索）

    // 质心范数（所有质心的预计算范数）
    std::vector<float> centroid_norms;

    // 码本叉积（用于快速LUT计算）
    // sum(codebook[m][k] * codebook[m'][k'])
    std::vector<float> codebook_cross_products;

    // 内存限制
    size_t max_mem_distances = 5 * (size_t(1) << 30);  // 5GB默认限制

    // 构造函数
    AdditiveQuantizer(
            size_t d,
            const std::vector<size_t>& nbits,
            Search_type_t search_type = ST_decompress);

    AdditiveQuantizer();
    virtual ~AdditiveQuantizer() {}

    // 计算派生值
    void set_derived_values();

    // 训练范数量化器
    void train_norm(size_t n, const float* norms);

    // 编码
    void compute_codes(const float* x, uint8_t* codes, size_t n) const override {
        compute_codes_add_centroids(x, codes, n);
    }

    virtual void compute_codes_add_centroids(
            const float* x,
            uint8_t* codes,
            size_t n,
            const float* centroids = nullptr) const = 0;

    // 解码
    void decode(const uint8_t* codes, float* x, size_t n) const override;

    // 打包编码（位压缩）
    void pack_codes(
            size_t n,
            const int32_t* codes,
            uint8_t* packed_codes,
            int64_t ld_codes = -1,
            const float* norms = nullptr,
            const float* centroids = nullptr) const;

    // 非打包解码
    virtual void decode_unpacked(
            const int32_t* codes,
            float* x,
            size_t n,
            int64_t ld_codes = -1) const;

    // 编码单个范数
    uint64_t encode_norm(float norm) const;

    // 非均匀标量量化（用于范数）
    uint32_t encode_qcint(float x) const;
    float decode_qcint(uint32_t c) const;

    // LUT计算
    virtual void compute_LUT(
            size_t n,
            const float* xq,
            float* LUT,
            float alpha = 1.0f,
            long ld_lut = -1) const;

    // 使用LUT的距离计算
    template <bool is_IP, Search_type_t effective_search_type>
    float compute_1_distance_LUT(const uint8_t* codes, const float* LUT) const;

    // 精确搜索
    void knn_centroids_inner_product(
            idx_t n,
            const float* xq,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    void knn_centroids_L2(
            idx_t n,
            const float* xq,
            idx_t k,
            float* distances,
            idx_t* labels,
            const float* centroid_norms) const;

    // 计算质心范数
    void compute_centroid_norms(float* norms) const;
    void compute_codebook_tables();
};
```

**Search_type详解**：

```
ST_decompress（最慢，最精确）：
  - 完全解码向量：x_decoded = decode(codes)
  - 计算距离：dis = distance(xq, x_decoded)
  - 适用于：任意度量，需要精确结果

ST_LUT_nonorm（快，适合归一化向量）：
  - 预计算查找表：LUT[m][k] = <xq[m], centroid[m][k]>
  - 快速计算：dis = sum(LUT[m][codes[m]])
  - 适用于：内积，归一化向量
  - 内存：M * K * sizeof(float)

ST_norm_float（平衡）：
  - 预计算查找表：LUT[m][k] = <xq[m], centroid[m][k]>
  - 解码范数：norm_q = decode_norm(codes)
  - 计算：dis = norm_q² + sum(LUT[m][codes[m]]) - 2 * sum(...)
  - 适用于：内积，未归一化向量

ST_norm_qint8/4（更快，有损）：
  - 范数量化为8/4位
  - 减少范数存储空间
  - 牺牲少量精度换取速度
```

### 1.3 编码布局

```cpp
// 打包编码（位压缩）
// 例如：M=4, nbits=[8,8,8,8] -> code_size=4字节
//      nbits=[6,6,6,6] -> code_size=3字节

void pack_codes_example() {
    int M = 4;
    std::vector<size_t> nbits = {8, 8, 8, 8};  // 每层8位
    size_t code_size = 4;  // 总共4字节

    // 未打包：codes[i][m] = 第i个向量的第m层编码
    int32_t* codes = new int32_t[n * M];

    // 打包后：packed_codes[i] = 第i个向量的打包编码
    uint8_t* packed_codes = new uint8_t[n * code_size];

    pq.pack_codes(n, codes, packed_codes);

    // 位布局示例（3字节，6+6+6+6=24位）：
    // [m0:6位][m1:6位][m2:6位][m3:6位]
    //  实际存储可能跨越字节边界
}

// 非打包布局（便于访问）：
// unpacked_codes[n * M] = 每个向量M个编码
```

---

## 2. ResidualQuantizer

### 2.1 核心思想

残差量化器逐步量化残差：
```cpp
x_0 = x              // 原始向量
c_0 = quantize(x_0)  // 第0层量化
x_1 = x_0 - c_0      // 残差
c_1 = quantize(x_1)  // 第1层量化
x_2 = x_1 - c_1      // 新残差
...
x_M = x_{M-1} - c_{M-1}

// 最终近似
x ≈ c_0 + c_1 + ... + c_M
```

### 2.2 ResidualQuantizer结构

```cpp
// faiss/impl/ResidualQuantizer.h
struct ResidualQuantizer : AdditiveQuantizer {
    int max_beam_size;  // beam search的beam大小

    enum train_type_t {
        Train_default = 0,           // 默认k-means
        Train_progressive_dim = 1,   // 渐进维度聚类
        Train_refine_codebook = 2,   // 码本细化
        Train_top_beam = 1024,       // 只训练beam顶部
    };
    train_type_t train_type;

    // 渐进维度聚类参数
    ProgressiveDimClusteringParameters cp;
};
```

### 2.3 训练流程

```cpp
void ResidualQuantizer::train(size_t n, const float* x) {
    // 初始化
    float* residuals = new float[n * d];
    memcpy(residuals, x, n * d * sizeof(float));

    // 逐层训练
    for (size_t m = 0; m < M; m++) {
        size_t K = 1 << nbits[m];  // 码本大小

        // 在残差上进行聚类
        Clustering clus(d, K);
        clus.train(n, residuals, cp);

        // 存储质心
        size_t offset = codebook_offsets[m];
        for (size_t k = 0; k < K; k++) {
            memcpy(codebooks.data() + (offset + k) * d,
                   clus.centroids + k * d,
                   d * sizeof(float));
        }

        // 更新残差
        if (m < M - 1) {
            for (size_t i = 0; i < n; i++) {
                // 找到最近的质心
                idx_t idx = assign_to_nearest(
                    residuals + i * d, clus.centroids, K, d);

                const float* centroid = clus.centroids + idx * d;
                float* res = residuals + i * d;

                for (int j = 0; j < d; j++) {
                    res[j] -= centroid[j];
                }
            }
        }
    }

    delete[] residuals;
    compute_codebook_tables();
}
```

---

## 3. Beam Search编码

### 3.1 基本思想

Beam search维护多个候选编码路径，选择最优的。

```cpp
// 3层量化的beam search示例
// beam_size = 2

Layer 0:     x
             ├─ c_00 (dis=1.0) ─┬─ c_10 (dis=0.8) ─┬─ c_20 (dis=0.6) → total=2.4
             │                  └─ c_11 (dis=1.2)   → total=3.2
             │
             └─ c_01 (dis=1.5) ─┬─ c_10 (dis=0.7) → total=2.2 ✓
                                └─ c_11 (dis=1.3) → total=2.8

保留beam_size=2个最优路径
```

### 3.2 Beam Search实现

```cpp
void ResidualQuantizer::refine_beam(
        size_t n,
        size_t beam_size,
        const float* residuals,  // (n, beam_size, d)
        int new_beam_size,
        int32_t* new_codes,     // (n, new_beam_size, m+1)
        float* new_residuals,   // (n, new_beam_size, d)
        float* new_distances)   // (n, new_beam_size)
        const {

    size_t K = 1 << nbits[m];  // 当前层的质心数

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        std::vector<std::pair<float, int>> heap;

        // 遍历所有beam和质心的组合
        for (size_t b = 0; b < beam_size; b++) {
            const float* res = residuals + (i * beam_size + b) * d;
            float prev_dis = new_distances[i * beam_size + b];

            for (size_t k = 0; k < K; k++) {
                const float* centroid =
                    codebooks.data() + (codebook_offsets[m] + k) * d;

                // 计算距离
                float dis = prev_dis + fvec_L2sqr(res, centroid, d);

                // 维护heap
                heap.push_back({dis, b * K + k});
                std::push_heap(heap.begin(), heap.end());

                if (heap.size() > new_beam_size) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.pop_back();
                }
            }
        }

        // 提取new_beam_size个最优候选
        std::sort(heap.begin(), heap.end());

        for (int b = 0; b < new_beam_size; b++) {
            int idx = heap[b].second;
            int prev_beam = idx / K;
            int centroid_id = idx % K;

            // 复制之前的编码
            if (m > 0) {
                memcpy(new_codes + (i * new_beam_size + b) * (m + 1),
                       new_codes + (i * beam_size + prev_beam) * m,
                       m * sizeof(int32_t));
            }
            // 添加当前编码
            new_codes[(i * new_beam_size + b) * (m + 1) + m] = centroid_id;

            // 更新残差和距离
            if (new_residuals) {
                const float* prev_res =
                    residuals + (i * beam_size + prev_beam) * d;
                float* res = new_residuals + (i * new_beam_size + b) * d;
                const float* centroid =
                    codebooks.data() + (codebook_offsets[m] + centroid_id) * d;

                for (int j = 0; j < d; j++) {
                    res[j] = prev_res[j] - centroid[j];
                }
            }

            new_distances[i * new_beam_size + b] = heap[b].first;
        }
    }
}
```

### 3.3 编码完整流程

```cpp
void ResidualQuantizer::compute_codes_add_centroids(
        const float* x,
        uint8_t* codes,
        size_t n,
        const float* centroids) const {

    // 初始化beam
    int beam_size = 1;
    float* residuals = new float[n * d];
    memcpy(residuals, x, n * d * sizeof(float));

    int32_t* beam_codes = new int32_t[n * M];
    float* beam_distances = new float[n];

    // 逐层beam search
    for (size_t m = 0; m < M; m++) {
        int new_beam_size = std::min(max_beam_size,
                                     int(1 << nbits[m]));

        int32_t* new_codes = new int32_t[n * new_beam_size * (m + 1)];
        float* new_residuals = new float[n * new_beam_size * d];
        float* new_distances = new float[n * new_beam_size];

        refine_beam(n, beam_size, residuals,
                   new_beam_size, new_codes, new_residuals, new_distances);

        delete[] residuals;
        residuals = new_residuals;
        beam_size = new_beam_size;

        // 只保留最优的候选
        for (size_t i = 0; i < n; i++) {
            memcpy(beam_codes + i * M,
                   new_codes + i * new_beam_size * (m + 1),
                   (m + 1) * sizeof(int32_t));
            beam_distances[i] = new_distances[i * new_beam_size];
        }

        delete[] new_codes;
        delete[] new_distances;
    }

    // 打包编码
    pack_codes(n, beam_codes, codes);

    delete[] residuals;
    delete[] beam_codes;
    delete[] beam_distances;
}
```

---

## 4. 查找表（LUT）优化

### 4.1 LUT计算

```cpp
// 计算查询向量与所有质心的内积表
void AdditiveQuantizer::compute_LUT(
        size_t n,
        const float* xq,
        float* LUT,
        float alpha,
        long ld_lut) const {

    if (ld_lut == -1) {
        ld_lut = total_codebook_size;
    }

    for (size_t i = 0; i < n; i++) {
        const float* x = xq + i * d;
        float* lut = LUT + i * ld_lut;

        // 遍历所有质心
        for (size_t m = 0; m < M; m++) {
            size_t offset = codebook_offsets[m];
            size_t K = codebook_offsets[m + 1] - offset;

            for (size_t k = 0; k < K; k++) {
                const float* centroid =
                    codebooks.data() + (offset + k) * d;

                lut[offset + k] = alpha * fvec_inner_product(x, centroid, d);
            }
        }
    }
}
```

### 4.2 使用LUT搜索

```cpp
// 使用LUT快速计算距离
template <bool is_IP, Search_type_t effective_search_type>
float AdditiveQuantizer::compute_1_distance_LUT(
        const uint8_t* codes,
        const float* LUT) const {

    float dis = 0.0f;
    const uint8_t* code = codes;

    for (size_t m = 0; m < M; m++) {
        size_t K = codebook_offsets[m + 1] - codebook_offsets[m];

        // 解码索引
        uint64_t idx = decode_uint64(code, nbits[m]);
        code += (nbits[m] + 7) / 8;

        // 从LUT查找
        size_t offset = codebook_offsets[m];
        dis += LUT[offset + idx];

        if (effective_search_type == ST_norm_float) {
            // 加上范数
            uint64_t norm_idx = decode_uint64(code, norm_bits);
            float norm = decode_qcint(norm_idx);
            code += (norm_bits + 7) / 8;
            dis += norm;
        }
    }

    return is_IP ? -dis : dis;  // 内积转距离
}
```

---

## 5. LocalSearchQuantizer

### 5.1 基本思想

局部搜索量化器通过迭代优化编码：

```cpp
1. 初始化：随机编码或用PQ初始化
2. 迭代优化：
   a. 随机选择一些位置
   b. 尝试改变这些位置的编码
   c. 如果误差减小，接受改变
   d. 重复直到收敛
```

### 5.2 局部搜索算法

```cpp
struct LocalSearchQuantizer : AdditiveQuantizer {
    int niter;          // 迭代次数
    int n_redo;         // 重启次数
    int K;              // 每次尝试的改变数

    void train(size_t n, const float* x) override {
        // 1. 用RQ初始化
        ResidualQuantizer rq(d, nbits);
        rq.train(n, x);

        // 2. 局部搜索优化
        int32_t* codes = new int32_t[n * M];

        for (int redo = 0; redo < n_redo; redo++) {
            // 编码
            rq.compute_codes_add_centroids(x, (uint8_t*)codes, n);

            // 局部搜索
            for (int iter = 0; iter < niter; iter++) {
                float improvement = local_search_iteration(n, x, codes);

                if (improvement < 1e-6) {
                    break;  // 收敛
                }
            }
        }

        delete[] codes;
    }

    float local_search_iteration(size_t n, const float* x, int32_t* codes) {
        float total_improvement = 0.0f;

#pragma omp parallel for reduction(+:total_improvement)
        for (size_t i = 0; i < n; i++) {
            const float* xi = x + i * d;
            int32_t* code_i = codes + i * M;

            // 计算当前误差
            float* current = new float[d];
            decode_unpacked(code_i, current, 1, M);
            float current_error = fvec_L2sqr(xi, current, d);

            // 尝试K次改变
            for (int k = 0; k < K; k++) {
                // 随机选择一个位置
                int m = rand() % M;
                int old_val = code_i[m];

                // 随机选择新值
                size_t K_m = 1 << nbits[m];
                int new_val = rand() % K_m;

                if (new_val == old_val) continue;

                // 计算新误差
                code_i[m] = new_val;
                decode_unpacked(code_i, current, 1, M);
                float new_error = fvec_L2sqr(xi, current, d);

                if (new_error < current_error) {
                    current_error = new_error;
                    total_improvement += (current_error - new_error);
                } else {
                    code_i[m] = old_val;  // 恢复
                }
            }

            delete[] current;
        }

        return total_improvement;
    }
};
```

---

## 6. 性能比较

### 6.1 量化精度

```cpp
void compare_quantizers() {
    int d = 128;
    int n = 10000;
    int M = 8;
    int nbits = 8;

    // Product Quantizer
    ProductQuantizer pq(d, M, nbits);
    pq.train(n, xb);

    uint8_t* pq_codes = new uint8_t[n * pq.code_size];
    pq.compute_codes(xb, pq_codes, n);

    float* pq_decoded = new float[n * d];
    pq.decode(pq_codes, pq_decoded, n);

    float pq_error = 0.0f;
    for (int i = 0; i < n * d; i++) {
        pq_error += (xb[i] - pq_decoded[i]) * (xb[i] - pq_decoded[i]);
    }

    // Residual Quantizer
    ResidualQuantizer rq(d, M, nbits);
    rq.train(n, xb);

    uint8_t* rq_codes = new uint8_t[n * rq.code_size];
    rq.compute_codes(xb, rq_codes, n);

    float* rq_decoded = new float[n * d];
    rq.decode(rq_codes, rq_decoded, n);

    float rq_error = 0.0f;
    for (int i = 0; i < n * d; i++) {
        rq_error += (xb[i] - rq_decoded[i]) * (xb[i] - rq_decoded[i]);
    }

    printf("PQ error:  %.6f\n", pq_error / (n * d));
    printf("RQ error:  %.6f\n", rq_error / (n * d));
    printf("RQ/PQ:    %.2fx\n", pq_error / rq_error);
}
```

### 6.2 内存使用

| 量化器 | 码本大小 | 编码大小 | 总内存 (n=1M) |
|--------|----------|----------|--------------|
| PQ | M × 2^nbits × d/M | M × nbits | 8MB + 2MB |
| RQ | Σ 2^nbits[m] × d | Σ nbits[m] | ~8MB + 4MB |

---

## 7. 实践建议

### 7.1 参数选择

```cpp
// 推荐：RQ使用递减位数
std::vector<size_t> nbits = {8, 8, 8, 7, 6, 5};  // 42位
// 而非固定位数
std::vector<size_t> nbits = {7, 7, 7, 7, 7, 7};  // 42位
```

### 7.2 Beam Size

```cpp
// beam_size vs 性能
// beam_size = 1: 贪心（快，精度一般）
// beam_size = 5: 平衡
// beam_size = 16: 高精度（慢）

rq.max_beam_size = 5;  // 推荐值
```

### 7.3 与IVF结合

```cpp
// IVF + RQ组合
Index* quantizer = new IndexFlatL2(d);
IndexIVFAdditiveQuantizer index(quantizer, d, nlist, rq, false);

// 训练
index.train(n, xb);
index.add(n, xb);

// 搜索
IVFSearchParameters params;
params.nprobe = 16;
index.search(nq, xq, k, distances, labels, &params);
```

---

## 8. 第6天总结

### 关键概念

1. **AdditiveQuantizer**：码本求和而非拼接
2. **ResidualQuantizer**：逐步量化残差
3. **Beam Search**：维护多个候选路径
4. **LUT优化**：预计算质心距离
5. **LocalSearchQuantizer**：迭代优化编码

### 性能对比

| 方法 | 精度 | 速度 | 内存 |
|------|------|------|------|
| PQ | 好 | 快 | 低 |
| RQ | 更好 | 中 | 中 |
| LSQ | 最佳 | 慢 | 中 |

### 下一步

第7天将学习**HNSW图索引**，这是目前最先进的近似最近邻算法之一。

---

---

## 9. ResidualQuantizer源码深度实现

本节深入分析Faiss中ResidualQuantizer和AdditiveQuantizer的核心实现细节，包括beam search编码、SIMD优化、查找表(LUT)计算等关键算法。

### 9.1 AdditiveQuantizer::compute_LUT - 查找表计算

查找表(Look-Up Table)是AQ性能优化的核心，预计算查询向量与所有质心的内积。

```cpp
// faiss/impl/AdditiveQuantizer.cpp
void AdditiveQuantizer::compute_LUT(
        size_t n,
        const float* xq,
        float* LUT,
        float alpha,
        long ld_lut) const {
    // 使用BLAS的SGEMM进行批量矩阵乘法
    // LUT = codebooks^T * xq * alpha
    // codebooks: (total_codebook_size, d)
    // xq: (d, n)
    // LUT: (total_codebook_size, n)

    FINTEGER ncenti = total_codebook_size;
    FINTEGER di = d;
    FINTEGER nqi = n;
    FINTEGER ldc = ld_lut > 0 ? ld_lut : ncenti;
    float zero = 0;

    sgemm_("Transposed",           // A转置: codebooks^T
           "Not transposed",        // B不转置: xq
           &ncenti,                 // M = total_codebook_size
           &nqi,                    // N = n
           &di,                     // K = d
           &alpha,                  // alpha
           codebooks.data(),        // A: codebooks (d, total_codebook_size)
           &di,                     // LDA = d
           xq,                      // B: xq (d, n)
           &di,                     // LDB = d
           &zero,                   // beta = 0
           LUT,                     // C: LUT (total_codebook_size, n)
           &ldc);                   // LDC
}
```

**关键优化点**：
- 使用高度优化的BLAS SGEMM实现
- 转置codebooks矩阵以利用缓存局部性
- 批量处理多个查询向量

### 9.2 ResidualQuantizer::train - 渐进式训练

训练流程包含多层码本学习和beam search编码。

```cpp
// faiss/impl/ResidualQuantizer.cpp
void ResidualQuantizer::train(size_t n, const float* x) {
    codebooks.resize(d * codebook_offsets.back());

    int cur_beam_size = 1;
    std::vector<float> residuals(x, x + n * d);
    std::vector<int32_t> codes;
    std::vector<float> distances;

    for (int m = 0; m < M; m++) {
        int K = 1 << nbits[m];  // 当前层质心数

        // 选择训练残差
        std::vector<float>& train_residuals = residuals;
        std::vector<float> residuals1;
        if (train_type & Train_top_beam) {
            // 只使用beam的第一个元素训练
            residuals1.resize(n * d);
            for (size_t j = 0; j < n; j++) {
                memcpy(residuals1.data() + j * d,
                       residuals.data() + j * d * cur_beam_size,
                       sizeof(residuals[0]) * d);
            }
            train_residuals = residuals1;
        }

        std::vector<float> codebooks_m;

        // 聚类训练
        if (!(train_type & Train_progressive_dim)) {
            // 标准k-means
            Clustering clus(d, K, cp);
            clus.train(
                    train_residuals.size() / d,
                    train_residuals.data(),
                    *assign_index.get());
            codebooks_m.swap(clus.centroids);
        } else {
            // 渐进维度聚类（更快）
            ProgressiveDimClustering clus(d, K, cp);
            clus.train(
                    train_residuals.size() / d,
                    train_residuals.data(),
                    assign_index_factory ? *assign_index_factory : default_fac);
            codebooks_m.swap(clus.centroids);
        }

        // 存储码本
        memcpy(this->codebooks.data() + codebook_offsets[m] * d,
               codebooks_m.data(),
               codebooks_m.size() * sizeof(codebooks_m[0]));

        // 使用新码本编码
        int new_beam_size = std::min(cur_beam_size * K, max_beam_size);
        std::vector<int32_t> new_codes(n * new_beam_size * (m + 1));
        std::vector<float> new_residuals(n * new_beam_size * d);
        std::vector<float> new_distances(n * new_beam_size);

        // 批量处理以控制内存
        size_t bs = n;
        if (n > 1 && memory_per_point() * n > max_mem_distances) {
            bs = std::max(max_mem_distances / memory_per_point(), size_t(1));
        }

        for (size_t i0 = 0; i0 < n; i0 += bs) {
            size_t i1 = std::min(i0 + bs, n);

            beam_search_encode_step(
                    d, K, codebooks_m.data(),
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

        codes.swap(new_codes);
        residuals.swap(new_residuals);
        distances.swap(new_distances);
        cur_beam_size = new_beam_size;
    }

    is_trained = true;

    // 可选：码本细化
    if (train_type & Train_refine_codebook) {
        for (int iter = 0; iter < niter_codebook_refine; iter++) {
            retrain_AQ_codebook(n, x);
        }
    }

    // 训练范数量化器
    std::vector<float> norms(n);
    for (size_t i = 0; i < n; i++) {
        norms[i] = fvec_L2sqr(
                x + i * d, residuals.data() + i * cur_beam_size * d, d);
    }
    train_norm(n, norms.data());

    compute_codebook_tables();
}
```

### 9.3 beam_search_encode_step - 单步Beam Search编码

这是RQ编码的核心函数，实现单层的beam search。

```cpp
// faiss/impl/residual_quantizer_encode_steps.cpp
void beam_search_encode_step(
        size_t d,
        size_t K,
        const float* cent,        // size (K, d)
        size_t n,
        size_t beam_size,
        const float* residuals,   // size (n, beam_size, d)
        size_t m,
        const int32_t* codes,     // size (n, beam_size, m)
        size_t new_beam_size,
        int32_t* new_codes,       // size (n, new_beam_size, m + 1)
        float* new_residuals,     // size (n, new_beam_size, d)
        float* new_distances,     // size (n, new_beam_size)
        Index* assign_index,
        ApproxTopK_mode_t approx_topk_mode) {

    FAISS_THROW_IF_NOT(new_beam_size <= beam_size * K);

    std::vector<float> cent_distances;
    std::vector<idx_t> cent_ids;

    // 计算距离
    if (assign_index) {
        // 使用索引搜索（更快）
        cent_distances.resize(n * beam_size * new_beam_size);
        cent_ids.resize(n * beam_size * new_beam_size);

        if (assign_index->ntotal == 0) {
            assign_index->add(K, cent);
        }

        assign_index->search(
                n * beam_size, residuals,
                new_beam_size,
                cent_distances.data(),
                cent_ids.data());
    } else {
        // 直接计算所有距离
        cent_distances.resize(n * beam_size * K);
        pairwise_L2sqr(
                d, n * beam_size, residuals,
                K, cent,
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

        using C = CMax<float, int>;  // 最大堆（取最小距离）

        if (assign_index) {
            // 使用索引返回的结果
            const float* cent_distances_i =
                    cent_distances.data() + i * beam_size * new_beam_size;
            const idx_t* cent_ids_i =
                    cent_ids.data() + i * beam_size * new_beam_size;

            // 初始化堆
            for (int j = 0; j < new_beam_size; j++) {
                new_distances_i[j] = C::neutral();
            }
            std::vector<int> perm(new_beam_size, -1);

            // 选择最优的new_beam_size个候选
            heap_addn<C>(
                    new_beam_size,
                    new_distances_i,
                    perm.data(),
                    cent_distances_i,
                    nullptr,
                    beam_size * new_beam_size);
            heap_reorder<C>(new_beam_size, new_distances_i, perm.data());

            // 构造新的beam
            for (int j = 0; j < new_beam_size; j++) {
                int js = perm[j] / new_beam_size;  // 源beam索引
                int ls = cent_ids_i[perm[j]];      // 质心索引

                // 复制之前的编码
                if (m > 0) {
                    memcpy(new_codes_i,
                           codes_i + js * m,
                           sizeof(*codes) * m);
                }
                new_codes_i[m] = ls;
                new_codes_i += m + 1;

                // 更新残差
                fvec_sub(
                        d,
                        residuals_i + js * d,
                        cent + ls * d,
                        new_residuals_i);
                new_residuals_i += d;
            }

        } else {
            // 从完整距离表中选择
            const float* cent_distances_i =
                    cent_distances.data() + i * beam_size * K;

            for (int j = 0; j < new_beam_size; j++) {
                new_distances_i[j] = C::neutral();
            }
            std::vector<int> perm(new_beam_size, -1);

            // 使用近似top-k优化
            switch (approx_topk_mode) {
                case ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B8_D3:
                    HeapWithBuckets<C, 8, 3>::bs_addn(
                            beam_size, K, cent_distances_i,
                            new_beam_size,
                            new_distances_i, perm.data());
                    break;
                case ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B16_D2:
                    HeapWithBuckets<C, 16, 2>::bs_addn(
                            beam_size, K, cent_distances_i,
                            new_beam_size,
                            new_distances_i, perm.data());
                    break;
                default:
                    // 精确选择
                    heap_addn<C>(
                            new_beam_size,
                            new_distances_i,
                            perm.data(),
                            cent_distances_i,
                            nullptr,
                            beam_size * K);
            }
            heap_reorder<C>(new_beam_size, new_distances_i, perm.data());

            // 构造新的beam
            for (int j = 0; j < new_beam_size; j++) {
                int js = perm[j] / K;    // 源beam索引
                int ls = perm[j] % K;    // 质心索引

                if (m > 0) {
                    memcpy(new_codes_i,
                           codes_i + js * m,
                           sizeof(*codes) * m);
                }
                new_codes_i[m] = ls;
                new_codes_i += m + 1;

                fvec_sub(
                        d,
                        residuals_i + js * d,
                        cent + ls * d,
                        new_residuals_i);
                new_residuals_i += d;
            }
        }
    }
}
```

### 9.4 AdditiveQuantizer::decode - SIMD优化解码

解码操作将编码转换为向量，使用SIMD指令加速。

```cpp
// faiss/impl/AdditiveQuantizer.cpp
void AdditiveQuantizer::decode(
        const uint8_t* code,
        float* x,
        size_t n) const {

    FAISS_THROW_IF_NOT_MSG(is_trained, "The additive quantizer is not trained yet.");

#pragma omp parallel for if (n > 100)
    for (int64_t i = 0; i < n; i++) {
        BitstringReader bsr(code + i * code_size, code_size);
        float* xi = x + i * d;

        for (int m = 0; m < M; m++) {
            int idx = bsr.read(nbits[m]);
            const float* c =
                codebooks.data() + d * (codebook_offsets[m] + idx);

            if (m == 0) {
                // 第一层：直接复制
                memcpy(xi, c, sizeof(*x) * d);
            } else {
                // 后续层：向量加法（SIMD优化）
                fvec_add(d, xi, c, xi);
            }
        }
    }
}

// SIMD向量加法（faiss/utils/distances.cpp）
void fvec_add(size_t d, const float* a, const float* b, float* c) {
#if defined(__AVX2__)
    // AVX2: 8个float并行处理
    size_t i = 0;
    for (; i + 7 < d; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < d; i++) {
        c[i] = a[i] + b[i];
    }
#elif defined(__aarch64__)
    // ARM NEON: 4个float并行处理
    size_t i = 0;
    for (; i + 3 < d; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(c + i, vc);
    }
    for (; i < d; i++) {
        c[i] = a[i] + b[i];
    }
#else
    // 标量版本
    for (size_t i = 0; i < d; i++) {
        c[i] = a[i] + b[i];
    }
#endif
}
```

### 9.5 AdditiveQuantizer::pack_codes - 位压缩编码

将未压缩的编码打包成位级紧凑格式。

```cpp
// faiss/impl/AdditiveQuantizer.cpp
void AdditiveQuantizer::pack_codes(
        size_t n,
        const int32_t* codes,
        uint8_t* packed_codes,
        int64_t ld_codes,
        const float* norms,
        const float* centroids) const {

    if (ld_codes == -1) {
        ld_codes = M;
    }

    // 计算范数（如果需要）
    std::vector<float> norm_buf;
    if (search_type == ST_norm_float ||
        search_type == ST_norm_qint8 ||
        search_type == ST_norm_qint4 ||
        search_type == ST_norm_cqint8 ||
        search_type == ST_norm_cqint4) {

        if (centroids != nullptr || !norms) {
            norm_buf.resize(n);
            std::vector<float> x_recons(n * d);
            decode_unpacked(codes, x_recons.data(), n, ld_codes);

            if (centroids != nullptr) {
                fvec_add(n * d, x_recons.data(), centroids, x_recons.data());
            }
            fvec_norms_L2sqr(norm_buf.data(), x_recons.data(), d, n);
            norms = norm_buf.data();
        }
    }

    // 并行打包
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        const int32_t* codes1 = codes + i * ld_codes;
        BitstringWriter bsw(packed_codes + i * code_size, code_size);

        // 写入每层编码
        for (int m = 0; m < M; m++) {
            bsw.write(codes1[m], nbits[m]);
        }

        // 写入范数（如果需要）
        if (norm_bits != 0) {
            bsw.write(encode_norm(norms[i]), norm_bits);
        }
    }
}
```

**位布局示例**：
```
nbits = [6, 6, 6, 6], norm_bits = 8
tot_bits = 32, code_size = 4字节

位布局 (小端序):
字节0: [m0:6位][m1:2位]
字节1: [m1:4位][m2:4位]
字节2: [m2:2位][m3:6位]
字节3: [norm:8位]
```

### 9.6 beam_search_encode_step_tab - 使用LUT的编码

当预计算了码本叉积时，可以使用更快的LUT编码。

```cpp
// faiss/impl/residual_quantizer_encode_steps.cpp
void beam_search_encode_step_tab(
        size_t K,
        size_t n,
        size_t beam_size,
        const float* codebook_cross_norms,  // size K * ldc
        size_t ldc,
        const uint64_t* codebook_offsets,   // m
        const float* query_cp,              // size n * ldqc
        size_t ldqc,
        const float* cent_norms_i,          // size K
        size_t m,
        const int32_t* codes,               // n * beam_size * m
        const float* distances,             // n * beam_size
        size_t new_beam_size,
        int32_t* new_codes,                 // n * new_beam_size * (m + 1)
        float* new_distances,               // n * new_beam_size
        ApproxTopK_mode_t approx_topk_mode) {

    FAISS_THROW_IF_NOT(ldc >= K);
    FAISS_THROW_IF_NOT(ldqc >= K);
    FAISS_THROW_IF_NOT(new_beam_size <= beam_size * K);

    // 并行处理每个向量
#pragma omp parallel for if (n > 100)
    for (int64_t i = 0; i < n; i++) {
        const int32_t* codes_i = codes + i * beam_size * m;
        int32_t* new_codes_i = new_codes + i * new_beam_size * (m + 1);
        const float* distances_i = distances + i * beam_size;
        float* new_distances_i = new_distances + i * new_beam_size;
        const float* query_cp_i = query_cp + i * ldqc;

        using C = CMax<float, int>;

        // 初始化输出
        for (size_t j = 0; j < new_beam_size; j++) {
            new_distances_i[j] = C::neutral();
        }
        std::vector<int> perm(new_beam_size, -1);
        std::vector<float> cent_distances(beam_size * K);

        // 计算所有beam-质心组合的距离
        // dist(b, k) = distances[b] + cent_norms[k]
        //              - 2 * (query_cp[k] + cross_prod[codes[b], k])
        for (size_t b = 0; b < beam_size; b++) {
            float* cd_b = cent_distances.data() + b * K;

            // 累加之前编码的叉积
            if (m == 0) {
                memcpy(cd_b, query_cp_i, sizeof(float) * K);
            } else {
                accum_and_store_tab<1, 1>(
                        m, codebook_cross_norms, codebook_offsets,
                        codes_i, b, ldc, K, cd_b);
                for (size_t k = 0; k < K; k++) {
                    cd_b[k] += query_cp_i[k];
                }
            }

            // 计算最终距离
            for (size_t k = 0; k < K; k++) {
                cent_distances[b * K + k] =
                    distances_i[b] + cent_norms_i[k] - 2 * cd_b[k];
            }
        }

        // 选择最优的new_beam_size个
        switch (approx_topk_mode) {
            case ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B8_D3:
                HeapWithBuckets<C, 8, 3>::bs_addn(
                        beam_size, K, cent_distances.data(),
                        new_beam_size,
                        new_distances_i, perm.data());
                break;
            default:
                heap_addn<C>(
                        new_beam_size,
                        new_distances_i,
                        perm.data(),
                        cent_distances.data(),
                        nullptr,
                        beam_size * K);
        }
        heap_reorder<C>(new_beam_size, new_distances_i, perm.data());

        // 构造新的编码
        for (size_t j = 0; j < new_beam_size; j++) {
            int js = perm[j] / K;
            int ls = perm[j] % K;

            if (m > 0) {
                memcpy(new_codes_i,
                       codes_i + js * m,
                       sizeof(*codes) * m);
            }
            new_codes_i[m] = ls;
            new_codes_i += m + 1;
        }
    }
}
```

### 9.7 生产级使用示例

```cpp
// 生产环境RQ使用示例
#include <faiss/ResidualQuantizer.h>
#include <faiss/IndexIVFAdditiveQuantizer.h>

void example_residual_quantizer_production() {
    // 参数配置
    int d = 128;
    size_t n = 1000000;
    size_t nlist = 4096;

    // RQ配置：递减位数分配
    std::vector<size_t> nbits = {8, 8, 8, 7, 6, 5};  // 42位

    // 创建RQ
    faiss::ResidualQuantizer rq(d, nbits);
    rq.train_type = faiss::ResidualQuantizer::Train_progressive_dim;
    rq.max_beam_size = 5;
    rq.search_type = faiss::AdditiveQuantizer::ST_norm_float;
    rq.verbose = true;

    // 训练RQ
    rq.train(n, xb);

    // 编码向量
    std::vector<uint8_t> codes(n * rq.code_size);
    rq.compute_codes(xb, codes.data(), n);

    // 解码验证
    std::vector<float> decoded(n * d);
    rq.decode(codes.data(), decoded.data(), n);

    // 计算量化误差
    float quantization_error = 0.0f;
    for (size_t i = 0; i < n * d; i++) {
        float diff = xb[i] - decoded[i];
        quantization_error += diff * diff;
    }
    printf("Quantization error: %.6f per dimension\n",
           quantization_error / (n * d));
}

// IVF + RQ组合索引
void example_ivf_rq_index() {
    int d = 128;
    size_t n = 1000000;
    size_t nlist = 4096;

    // 创建量化器
    faiss::IndexFlatL2 quantizer(d);

    // 创建RQ
    std::vector<size_t> nbits = {8, 8, 8, 7, 6};
    faiss::ResidualQuantizer rq(d, nbits);
    rq.train_type = faiss::ResidualQuantizer::Train_progressive_dim;
    rq.max_beam_size = 5;

    // 创建IVF+RQ索引
    faiss::IndexIVFAdditiveQuantizer index(
            &quantizer, d, nlist, rq, false);

    // 训练和添加向量
    index.train(n, xb);
    index.add(n, xb);

    // 搜索
    size_t nq = 10;
    size_t k = 100;
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    faiss::IVFSearchParameters params;
    params.nprobe = 16;  // 搜索16个倒排列表

    index.search(nq, xq, k,
                 distances.data(), labels.data(), &params);

    // 评估召回率
    size_t nq_correct = 0;
    for (size_t i = 0; i < nq; i++) {
        for (size_t j = 0; j < k; j++) {
            if (labels[i * k + j] == ground_truth[i]) {
                nq_correct++;
                break;
            }
        }
    }
    printf("Recall@%d: %.2f%%\n", int(k),
           100.0 * nq_correct / nq);
}

// 使用LUT模式搜索
void example_rq_lut_search() {
    int d = 128;
    std::vector<size_t> nbits = {8, 8, 8, 8};

    // 创建RQ（LUT模式）
    faiss::ResidualQuantizer rq(d, nbits,
        faiss::AdditiveQuantizer::ST_norm_float);
    rq.train(n, xb);

    // 编码数据库
    std::vector<uint8_t> codes(n * rq.code_size);
    rq.compute_codes(xb, codes.data(), n);

    // 计算查询的LUT
    size_t nq = 10;
    std::vector<float> LUT(nq * rq.total_codebook_size);
    rq.compute_LUT(nq, xq, LUT.data());

    // 使用LUT计算距离
    size_t k = 10;
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

#pragma omp parallel for
    for (size_t i = 0; i < nq; i++) {
        const uint8_t* codes_i = codes.data();
        const float* LUT_i = LUT.data() + i * rq.total_codebook_size;

        // 使用模板特化计算距离
        for (size_t j = 0; j < n; j++) {
            float dis = rq.compute_1_distance_LUT<false,
                faiss::AdditiveQuantizer::ST_norm_float>(
                    codes_i + j * rq.code_size, LUT_i);

            // 维护堆
            if (j < k) {
                distances[i * k + j] = dis;
                labels[i * k + j] = j;
            } else if (dis < distances[i * k]) {
                faiss::heap_replace_top<faiss::CMax<float, idx_t>>(
                    k, distances.data() + i * k,
                    labels.data() + i * k, dis, j);
            }
        }

        faiss::heap_reorder<faiss::CMax<float, idx_t>>(
            k, distances.data() + i * k, labels.data() + i * k);
    }
}
```

### 9.8 性能优化总结

| 优化技术 | 描述 | 性能提升 |
|----------|------|----------|
| BLAS SGEMM | LUT计算使用优化的矩阵乘法 | 5-10x |
| SIMD向量加法 | AVX2/NEON并行向量操作 | 4-8x |
| OpenMP并行 | 多线程编码/解码 | 线性加速 |
| 近似Top-K | 桶排序加速选择 | 2-3x |
| 内存池 | 重用临时缓冲区 | 减少分配开销 |
| 位压缩 | 减少编码存储 | 节省内存 |
| 渐进式聚类 | 加速训练 | 2-4x |

---

## 练习题

1. 实现简化的RQ编码器
2. 比较不同beam_size的效果
3. 实现LUT优化的搜索
4. 比较PQ、RQ、LSQ的性能

## 扩展阅读

- faiss/impl/AdditiveQuantizer.h - AQ基类
- faiss/impl/ResidualQuantizer.h - RQ实现
- [RQ论文](https://arxiv.org/abs/1507.04952)
- [LSQ论文](https://arxiv.org/abs/1908.10396)
