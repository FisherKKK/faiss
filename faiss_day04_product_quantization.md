# Faiss深度课程 - 第4天：量化基础 - Product Quantization

## 课程目标

深入理解Product Quantization（乘积量化）的核心算法，这是Faiss中最关键的压缩技术，能够大幅减少内存使用并加速搜索。

---

## 1. Product Quantization概述

### 1.1 基本思想

Product Quantization（PQ）将高维向量分解为多个子向量，分别量化后再组合。

**核心概念**：
```cpp
// 原始向量：128维
float x[128] = {...};

// PQ参数：M=8个子量化器，每个8维，nbits=8（256个质心）
// 分解为8个子向量
x = [x_0, x_1, ..., x_7]  // 每个8维

// 每个子向量量化为8位索引
code = [idx_0, idx_1, ..., idx_7]  // 每个0-255

// 原始：128 * 4 = 512字节
// PQ编码：8字节
// 压缩比：64倍
```

### 1.2 PQ的优势

| 特性 | 原始向量 | PQ编码 |
|------|---------|--------|
| 128维存储 | 512字节 | 8字节 |
| 距离计算 | 128次乘加 | 8次查表 |
| 内存带宽 | 高 | 极低 |
| 精度 | 精确 | 有损近似 |

### 1.3 ProductQuantizer类结构（底层实现）

```cpp
// faiss/impl/ProductQuantizer.h (完整版本)
struct ProductQuantizer : Quantizer {
    // 基本参数
    size_t d;       // 原始向量维度
    size_t M;       // 子量化器数量
    size_t nbits;   // 每个子量化器的位数

    // 派生参数
    size_t dsub;        // 每个子向量的维度 = d / M
    size_t ksub;        // 每个子量化器的质心数 = 2^nbits
    size_t code_size;   // PQ码的字节数

    // 质心表: M * ksub * dsub
    // 内存布局: centroids[m * ksub * dsub + k * dsub + j]
    //          = 子量化器m的第k个质心的第j维
    std::vector<float> centroids;

    // 转置质心表（优化内存访问模式）
    // 布局: transposed_centroids[dsub * M * ksub + m * ksub + k]
    //       = 子量化器m的第k个质心（连续存储dsub维）
    std::vector<float> transposed_centroids;

    // 质心的L2范数平方（用于内积距离优化）
    std::vector<float> centroids_sq_lengths;

    // 训练类型
    enum train_type_t {
        Train_default,       // 标准k-means
        Train_hot_start,     // 从现有质心热启动
        Train_shared,        // 共享字典（所有子量化器共享质心）
        Train_hypercube,     // 超立方体初始化
        Train_hypercube_pca  // PCA + 超立方体
    };
    train_type_t train_type = Train_default;

    // 对称距离计算（SDC）预计算表
    std::vector<float> sdc_table;

    // 训练参数
    ClusteringParameters cp;  // k-means参数

    // 构造函数
    ProductQuantizer(size_t d, const std::vector<size_t>& nbits);
    ProductQuantizer(size_t d, size_t M, size_t nbits);
    ProductQuantizer();

    // 获取质心指针（内联函数，性能关键）
    inline float* get_centroids(size_t m, size_t k) {
        return centroids.data() + (m * ksub + k) * dsub;
    }

    inline const float* get_centroids(size_t m, size_t k) const {
        return centroids.data() + (m * ksub + k) * dsub;
    }

    // 从转置表获取质心（更好的缓存局部性）
    inline float* get_centroids_transposed(size_t m, size_t k) {
        return transposed_centroids.data() + (m * ksub + k) * dsub;
    }

    // 计算派生值
    void set_derived_values();

    // 计算质心范数
    void compute_centroid_norms();

    // 编码/解码向量
    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;
    void decode(const uint8_t* codes, float* x, size_t n) const override;

    // 计算单个PQ码
    void compute_code(const float* x, uint8_t* code) const;

    // 距离表计算（用于ADC）
    void compute_inner_prod_table(const float* x, float* dis_tables) const;
    void compute_L2_distance_table(const float* x, const float* y_norms,
                                   float* dis_tables) const;

    // 对称距离计算
    void compute_sdc_table();

    // 训练PQ
    void train(size_t n, const float* x) override;
    void train_default(size_t n, const float* x);
    void train_shared(size_t n, const float* x);
    void train_hypercube(size_t n, const float* x, bool pca);
};
```

**内存布局详解**：

```
centroids数组布局 (M=4, ksub=256, dsub=32):
[子量化器0]
  [质心0: 32维float]
  [质心1: 32维float]
  ...
  [质心255: 32维float]
[子量化器1]
  [质心0: 32维float]
  ...

总大小: M * ksub * dsub * sizeof(float)
       = 4 * 256 * 32 * 4 = 131,072 字节 = 128 KB
```

**transposed_centroids布局优化**：
```
// 标准布局（centroids）：不连续访问
for m in 0..M:
    for k in 0..ksub:
        centroid = centroids[m * ksub * dsub + k * dsub]  // 跳跃访问

// 转置布局（transposed_centroids）：连续访问
for m in 0..M:
    for k in 0..ksub:
        centroid = transposed_centroids[(m * ksub + k) * dsub]  // 连续
```

---

## 2. PQ编码与解码

### 2.1 编码过程

```cpp
// 编码：将浮点向量转换为PQ码
void ProductQuantizer::compute_code(const float* x, uint8_t* code) const {
    // 对于每个子量化器
    for (size_t m = 0; m < M; m++) {
        const float* xm = x + m * dsub;  // 子向量m

        // 找到最近的质心
        float min_dis = HUGE_VAL;
        idx_t idx = 0;

        for (size_t k = 0; k < ksub; k++) {
            const float* ck = get_centroids(m, k);
            float dis = fvec_L2sqr(xm, ck, dsub);

            if (dis < min_dis) {
                min_dis = dis;
                idx = k;
            }
        }

        // 存储索引（使用编码器处理位数）
        encode_uint(code, m, idx, nbits);
    }
}

// 通用编码器：处理任意位数
struct PQEncoderGeneric {
    uint8_t* code;
    uint8_t offset;
    const int nbits;
    uint8_t reg;

    PQEncoderGeneric(uint8_t* code, int nbits, uint8_t offset = 0)
        : code(code), offset(offset), nbits(nbits), reg(0) {}

    void encode(uint64_t x) {
        // 将x的nbits位打包到code中
        reg |= (x << offset);
        offset += nbits;

        if (offset >= 8) {
            *code++ = reg & 0xFF;
            reg >>= 8;
            offset -= 8;
        }
    }
};
```

### 2.2 解码过程

```cpp
// 解码：从PQ码重建近似向量
void ProductQuantizer::decode(const uint8_t* code, float* x) const {
    for (size_t m = 0; m < M; m++) {
        // 提取子量化器m的索引
        uint64_t idx = decode_uint(code, m, nbits);

        // 获取对应质心
        const float* ck = get_centroids(m, idx);

        // 复制到输出向量
        float* xm = x + m * dsub;
        memcpy(xm, ck, dsub * sizeof(float));
    }
}

// 通用解码器
struct PQDecoderGeneric {
    const uint8_t* code;
    uint8_t offset;
    const int nbits;
    const uint64_t mask;
    uint8_t reg;

    PQDecoderGeneric(const uint8_t* code, int nbits)
        : code(code), offset(0), nbits(nbits),
          mask((1ULL << nbits) - 1), reg(0) {}

    uint64_t decode() {
        uint64_t x;
        if (offset == 0) {
            reg = *code++;
        }

        x = reg & mask;
        reg >>= nbits;
        offset += nbits;

        if (offset >= 8) {
            offset -= 8;
            code++;
            if (offset > 0) {
                reg = *code;
            }
        }

        return x;
    }
};
```

### 2.3 特殊编码器

```cpp
// 8位编码器（每个子量化器1字节）
struct PQEncoder8 {
    uint8_t* code;
    PQEncoder8(uint8_t* code, int nbits) : code(code) {}
    void encode(uint64_t x) {
        *code++ = (uint8_t)x;
    }
};

// 16位编码器（每个子量化器2字节）
struct PQEncoder16 {
    uint16_t* code;
    PQEncoder16(uint8_t* code, int nbits)
        : code((uint16_t*)code) {}
    void encode(uint64_t x) {
        *code++ = (uint16_t)x;
    }
};
```

---

## 3. PQ训练 - K-Means聚类

### 3.1 训练流程

```cpp
void ProductQuantizer::train(size_t n, const float* x) {
    // 为每个子量化器独立训练

    for (size_t m = 0; m < M; m++) {
        // 提取子量化器m的训练数据
        float* sub_data = new float[n * dsub];
        for (size_t i = 0; i < n; i++) {
            memcpy(sub_data + i * dsub,
                   x + i * d + m * dsub,
                   dsub * sizeof(float));
        }

        // K-means聚类
        Clustering clus(dsub, ksub);
        clus.train(n, sub_data, cp);

        // 保存质心
        for (size_t k = 0; k < ksub; k++) {
            memcpy(get_centroids(m, k),
                   clus.centroids + k * dsub,
                   dsub * sizeof(float));
        }

        delete[] sub_data;
    }

    // 计算质心平方长度（用于内积）
    sync_centroid_sq_lengths();

    // 生成转置质心表
    sync_transposed_centroids();
}
```

### 3.2 K-Means算法

```cpp
// 简化的K-means实现
struct Clustering {
    int d;        // 维度
    size_t k;     // 质心数
    float* centroids;  // 输出质心

    void train(size_t n, const float* x, ClusteringParameters& cp) {
        // 初始化质心
        initialize_centroids(n, x);

        // 迭代优化
        for (int iter = 0; iter < cp.niter; iter++) {
            // 分配：找到每个点最近的质心
            idx_t* assign = new idx_t[n];
            assign_to_nearest(n, x, assign);

            // 更新：重新计算质心
            update_centroids(n, x, assign);

            delete[] assign;
        }
    }

    void assign_to_nearest(size_t n, const float* x, idx_t* assign) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            float min_dis = HUGE_VAL;
            idx_t best = 0;

            for (size_t j = 0; j < k; j++) {
                float dis = fvec_L2sqr(x + i * d, centroids + j * d, d);
                if (dis < min_dis) {
                    min_dis = dis;
                    best = j;
                }
            }

            assign[i] = best;
        }
    }

    void update_centroids(size_t n, const float* x, const idx_t* assign) {
        std::vector<float> sums(k * d, 0.0f);
        std::vector<int> counts(k, 0);

        // 累加每个质心的点
        for (size_t i = 0; i < n; i++) {
            idx_t j = assign[i];
            for (int dim = 0; dim < d; dim++) {
                sums[j * d + dim] += x[i * d + dim];
            }
            counts[j]++;
        }

        // 平均
        for (size_t j = 0; j < k; j++) {
            if (counts[j] > 0) {
                for (int dim = 0; dim < d; dim++) {
                    centroids[j * d + dim] = sums[j * d + dim] / counts[j];
                }
            }
        }
    }
};
```

---

## 4. 非对称距离计算（ADC）

### 4.1 距离表计算

```cpp
// 为查询向量计算距离表
// dis_table[m * ksub + k] = ||x_m - c_(m,k)||^2
void ProductQuantizer::compute_distance_table(
        const float* x,
        float* dis_table) const {

    for (size_t m = 0; m < M; m++) {
        const float* xm = x + m * dsub;
        float* dt = dis_table + m * ksub;

        for (size_t k = 0; k < ksub; k++) {
            const float* ck = get_centroids(m, k);
            dt[k] = fvec_L2sqr(xm, ck, dsub);
        }
    }
}
```

### 4.2 使用距离表搜索

```cpp
// ADC搜索：查询使用原始向量，数据库使用PQ码
void ProductQuantizer::search(
        const float* x,
        size_t nx,
        const uint8_t* codes,
        const size_t ncodes,
        float_maxheap_array_t* res) const {

    for (size_t i = 0; i < nx; i++) {
        // 1. 计算查询的距离表
        float* dis_table = new float[M * ksub];
        compute_distance_table(x + i * d, dis_table);

        // 2. 初始化堆
        float* simi = res->val + i * res->k;
        idx_t* idxi = res->ids + i * res->k;
        heap_heapify<CMax<float, idx_t>>(res->k, simi, idxi);

        // 3. 遍历所有PQ码
        for (size_t j = 0; j < ncodes; j++) {
            float dis = 0;

            // 累加所有子量化器的距离
            for (size_t m = 0; m < M; m++) {
                uint64_t idx = decode_uint(codes + j * code_size, m, nbits);
                dis += dis_table[m * ksub + idx];
            }

            // 更新堆
            if (CMax<float, idx_t>::cmp(dis, simi[0])) {
                heap_replace_top<CMax<float, idx_t>>(
                    res->k, simi, idxi, dis, j);
            }
        }

        heap_reorder<CMax<float, idx_t>>(res->k, simi, idxi);
        delete[] dis_table;
    }
}
```

### 4.3 SIMD优化的搜索

```cpp
// SIMD优化的距离累加
inline float pq_distance_accumulator(
        const float* dis_table,
        const uint8_t* code,
        size_t M,
        size_t ksub) {

    float dis = 0.0f;
    size_t m = 0;

    // 展开循环以提高性能
    for (; m + 4 <= M; m += 4) {
        uint64_t idx0 = decode_uint(code, m + 0, nbits);
        uint64_t idx1 = decode_uint(code, m + 1, nbits);
        uint64_t idx2 = decode_uint(code, m + 2, nbits);
        uint64_t idx3 = decode_uint(code, m + 3, nbits);

        dis += dis_table[(m + 0) * ksub + idx0];
        dis += dis_table[(m + 1) * ksub + idx1];
        dis += dis_table[(m + 2) * ksub + idx2];
        dis += dis_table[(m + 3) * ksub + idx3];
    }

    for (; m < M; m++) {
        uint64_t idx = decode_uint(code, m, nbits);
        dis += dis_table[m * ksub + idx];
    }

    return dis;
}
```

---

## 5. 对称距离计算（SDC）

### 5.1 SDC表计算

```cpp
// 计算对称距离表
// sdc_table[m1*ksub + m2*ksub + i*ksub + j] = ||c_(m1,i) - c_(m2,j)||^2
void ProductQuantizer::compute_sdc_table() {
    sdc_table.resize(M * M * ksub * ksub);

    for (size_t m1 = 0; m1 < M; m1++) {
        for (size_t m2 = 0; m2 < M; m2++) {
            for (size_t i = 0; i < ksub; i++) {
                for (size_t j = 0; j < ksub; j++) {
                    const float* c1 = get_centroids(m1, i);
                    const float* c2 = get_centroids(m2, j);

                    float dis = fvec_L2sqr(c1, c2, dsub);

                    size_t idx = ((m1 * M + m2) * ksub + i) * ksub + j;
                    sdc_table[idx] = dis;
                }
            }
        }
    }
}
```

### 5.2 SDC搜索

```cpp
// 查询和数据库都使用PQ码
void ProductQuantizer::search_sdc(
        const uint8_t* qcodes,
        size_t nq,
        const uint8_t* bcodes,
        const size_t ncodes,
        float_maxheap_array_t* res) const {

    for (size_t q = 0; q < nq; q++) {
        const uint8_t* qcode = qcodes + q * code_size;

        float* simi = res->val + q * res->k;
        idx_t* idxi = res->ids + q * res->k;
        heap_heapify<CMax<float, idx_t>>(res->k, simi, idxi);

        for (size_t b = 0; b < ncodes; b++) {
            const uint8_t* bcode = bcodes + b * code_size;

            float dis = 0;
            for (size_t m = 0; m < M; m++) {
                uint64_t qidx = decode_uint(qcode, m, nbits);
                uint64_t bidx = decode_uint(bcode, m, nbits);

                size_t idx = ((m * M + m) * ksub + qidx) * ksub + bidx;
                dis += sdc_table[idx];
            }

            if (CMax<float, idx_t>::cmp(dis, simi[0])) {
                heap_replace_top<CMax<float, idx_t>>(
                    res->k, simi, idxi, dis, b);
            }
        }

        heap_reorder<CMax<float, idx_t>>(res->k, simi, idxi);
    }
}
```

---

## 6. IndexPQ - PQ索引

### 6.1 IndexPQ结构

```cpp
// faiss/IndexPQ.h
struct IndexPQ : IndexFlatCodes {
    ProductQuantizer pq;  // PQ编码器

    bool do_polysemous_training;  // 是否使用多义训练
    Search_type_t search_type;    // 搜索类型

    enum Search_type_t {
        ST_PQ,                   // ADC（默认）
        ST_HE,                   // Hamming距离
        ST_generalized_HE,       // 广义Hamming
        ST_SDC,                  // SDC
        ST_polysemous,           // 多义
    };
};
```

### 6.2 IndexPQ搜索

```cpp
void IndexPQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    switch (search_type) {
        case ST_PQ:
            // ADC搜索
            pq.search(x, n, codes.data(), ntotal,
                      float_maxheap_array_t{n, k, labels, distances});
            break;

        case ST_SDC:
            // 首先编码查询
            uint8_t* qcodes = new uint8_t[n * pq.code_size];
            pq.compute_codes(x, qcodes, n);

            // SDC搜索
            pq.search_sdc(qcodes, n, codes.data(), ntotal,
                          float_maxheap_array_t{n, k, labels, distances});

            delete[] qcodes;
            break;

        case ST_polysemous:
            // 多义搜索
            search_core_polysemous(n, x, k, distances, labels,
                                  polysemous_ht, false);
            break;
    }
}
```

---

## 7. 内存与性能分析

### 7.1 内存使用

```cpp
// 原始向量 vs PQ编码
void memory_comparison() {
    int d = 128;
    int n = 1000000;  // 100万向量

    // 原始存储
    size_t original_memory = n * d * sizeof(float);  // 512 MB

    // PQ编码：M=16, nbits=8
    int M = 16;
    int nbits = 8;
    size_t code_size = (M * nbits + 7) / 8;  // 16字节
    size_t pq_memory = n * code_size;        // 16 MB

    // 质心表
    size_t centroids_memory = M * (1 << nbits) * (d / M) * sizeof(float);  // 2 MB

    // 总内存
    size_t total_pq_memory = pq_memory + centroids_memory;  // 18 MB

    // 压缩比
    float compression_ratio = (float)original_memory / total_pq_memory;

    printf("原始: %zu MB\n", original_memory / (1024 * 1024));
    printf("PQ:   %zu MB\n", total_pq_memory / (1024 * 1024));
    printf("压缩比: %.1fx\n", compression_ratio);
}
```

### 7.2 性能优化技巧

```cpp
// 1. 使用转置质心表（更好的缓存局部性）
void ProductQuantizer::sync_transposed_centroids() {
    transposed_centroids.resize(M * ksub * dsub);

    for (size_t m = 0; m < M; m++) {
        for (size_t k = 0; k < ksub; k++) {
            for (size_t dim = 0; dim < dsub; dim++) {
                size_t src_idx = (m * ksub + k) * dsub + dim;
                size_t dst_idx = (dim * M + m) * ksub + k;
                transposed_centroids[dst_idx] = centroids[src_idx];
            }
        }
    }
}

// 2. 批量计算距离表
void ProductQuantizer::compute_distance_tables(
        size_t nx,
        const float* x,
        float* dis_tables) const {

    // dis_tables布局: (nx, M, ksub)
    for (size_t i = 0; i < nx; i++) {
        compute_distance_table(x + i * d,
                              dis_tables + i * M * ksub);
    }
}
```

---

## 8. 实践示例

### 8.1 完整PQ流程

```cpp
void complete_pq_example() {
    // 参数
    int d = 128;
    int n = 100000;
    int M = 16;
    int nbits = 8;

    // 1. 创建PQ
    ProductQuantizer pq(d, M, nbits);

    // 2. 训练
    pq.train(n, xb);

    // 3. 编码
    uint8_t* codes = new uint8_t[n * pq.code_size];
    pq.compute_codes(xb, codes, n);

    // 4. 搜索
    int nq = 100;
    int k = 10;

    float* distances = new float[nq * k];
    idx_t* labels = new idx_t[nq * k];

    float_maxheap_array_t res = {nq, k, labels, distances};
    pq.search(xq, nq, codes, n, &res);

    // 5. 清理
    delete[] codes;
    delete[] distances;
    delete[] labels;
}
```

---

## 10. 源码深度实现 - ProductQuantizer完整架构

### 10.1 ProductQuantizer类结构定义

```cpp
// faiss/impl/ProductQuantizer.h
// ProductQuantizer的完整类定义

struct ProductQuantizer : Quantizer {
    // 基本参数
    size_t d;       // 原始向量维度
    size_t M;       // 子量化器数量
    size_t nbits;   // 每个子量化器的位数

    // 派生参数
    size_t dsub;        // 每个子向量的维度 = d / M
    size_t ksub;        // 每个子量化器的质心数 = 2^nbits
    size_t code_size;   // PQ码的字节数

    // 质心表: M * ksub * dsub
    // 内存布局: centroids[m * ksub * dsub + k * dsub + j]
    std::vector<float> centroids;

    // 转置质心表（优化内存访问模式）
    std::vector<float> transposed_centroids;

    // 质心的L2范数平方（用于内积距离优化）
    std::vector<float> centroids_sq_lengths;

    // 训练类型
    enum train_type_t {
        Train_default,       // 标准k-means
        Train_hot_start,     // 从现有质心热启动
        Train_shared,        // 共享字典
        Train_hypercube,     // 超立方体初始化
        Train_hypercube_pca  // PCA + 超立方体
    };
    train_type_t train_type = Train_default;

    // 对称距离计算（SDC）预计算表
    std::vector<float> sdc_table;

    // 训练参数
    ClusteringParameters cp;

    // 构造函数
    ProductQuantizer(size_t d, const std::vector<size_t>& nbits);
    ProductQuantizer(size_t d, size_t M, size_t nbits);
    ProductQuantizer();

    // 获取质心指针（内联函数，性能关键）
    inline float* get_centroids(size_t m, size_t k) {
        return centroids.data() + (m * ksub + k) * dsub;
    }

    inline const float* get_centroids(size_t m, size_t k) const {
        return centroids.data() + (m * ksub + k) * dsub;
    }

    // 从转置表获取质心（更好的缓存局部性）
    inline float* get_centroids_transposed(size_t m, size_t k) {
        return transposed_centroids.data() + (m * ksub + k) * dsub;
    }

    // 计算派生值
    void set_derived_values();

    // 计算质心范数
    void compute_centroid_norms();

    // 编码/解码向量
    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;
    void decode(const uint8_t* codes, float* x, size_t n) const override;

    // 计算单个PQ码
    void compute_code(const float* x, uint8_t* code) const;

    // 距离表计算（用于ADC）
    void compute_inner_prod_table(const float* x, float* dis_tables) const;
    void compute_L2_distance_table(const float* x, const float* y_norms,
                                   float* dis_tables) const;

    // 对称距离计算
    void compute_sdc_table();

    // 训练PQ
    void train(size_t n, const float* x) override;
};
```

### 10.2 派生值计算与初始化

```cpp
// faiss/impl/ProductQuantizer.cpp
// 计算派生参数
void ProductQuantizer::set_derived_values() {
    FAISS_THROW_IF_NOT_MSG(
            d % M == 0,
            "The dimension of the vector (d) should be a multiple of M");
    dsub = d / M;
    code_size = (nbits * M + 7) / 8;
    FAISS_THROW_IF_MSG(nbits > 24, "nbits larger than 24 is not practical.");
    ksub = 1 << nbits;
    centroids.resize(d * ksub);
    verbose = false;
    train_type = Train_default;
}

// 设置指定子量化器的质心
void ProductQuantizer::set_params(const float* centroids_, int m) {
    memcpy(get_centroids(m, 0),
           centroids_,
           ksub * dsub * sizeof(centroids_[0]));
}
```

### 10.3 PQ编码器实现

```cpp
// 通用编码器：处理任意位数
struct PQEncoderGeneric {
    uint8_t* code;
    uint8_t offset;
    const int nbits;
    uint8_t reg;

    PQEncoderGeneric(uint8_t* code, int nbits, uint8_t offset = 0)
        : code(code), offset(offset), nbits(nbits), reg(0) {}

    void encode(uint64_t x) {
        // 将x的nbits位打包到code中
        reg |= (x << offset);
        offset += nbits;

        if (offset >= 8) {
            *code++ = reg & 0xFF;
            reg >>= 8;
            offset -= 8;
        }
    }
};

// 8位编码器（每个子量化器1字节）
struct PQEncoder8 {
    uint8_t* code;
    PQEncoder8(uint8_t* code, int nbits) : code(code) {}
    void encode(uint64_t x) {
        *code++ = (uint8_t)x;
    }
};

// 16位编码器（每个子量化器2字节）
struct PQEncoder16 {
    uint16_t* code;
    PQEncoder16(uint8_t* code, int nbits)
        : code((uint16_t*)code) {}
    void encode(uint64_t x) {
        *code++ = (uint16_t)x;
    }
};
```

### 10.4 PQ解码器实现

```cpp
// 通用解码器
struct PQDecoderGeneric {
    const uint8_t* code;
    uint8_t offset;
    const int nbits;
    const uint64_t mask;
    uint8_t reg;

    PQDecoderGeneric(const uint8_t* code, int nbits)
        : code(code), offset(0), nbits(nbits),
          mask((1ULL << nbits) - 1), reg(0) {}

    uint64_t decode() {
        uint64_t x;
        if (offset == 0) {
            reg = *code++;
        }

        x = reg & mask;
        reg >>= nbits;
        offset += nbits;

        if (offset >= 8) {
            offset -= 8;
            code++;
            if (offset > 0) {
                reg = *code;
            }
        }

        return x;
    }
};

// 8位解码器
struct PQDecoder8 {
    const uint8_t* code;
    PQDecoder8(const uint8_t* code, int nbits) : code(code) {}
    uint64_t decode() {
        return *code++;
    }
};

// 16位解码器
struct PQDecoder16 {
    const uint16_t* code;
    PQDecoder16(const uint8_t* code, int nbits)
        : code((const uint16_t*)code) {}
    uint64_t decode() {
        return *code++;
    }
};
```

### 10.5 编码计算实现

```cpp
// faiss/impl/ProductQuantizer.cpp
// 计算单个向量的PQ码
template <class PQEncoder>
void compute_code(const ProductQuantizer& pq, const float* x, uint8_t* code) {
    std::vector<float> distances(pq.ksub);

    // 使用距离缓存以优化编译器生成的代码
    PQEncoder encoder(code, pq.nbits);
    for (size_t m = 0; m < pq.M; m++) {
        const float* xsub = x + m * pq.dsub;

        uint64_t idxm = 0;
        if (pq.transposed_centroids.empty()) {
            // 常规版本
            idxm = fvec_L2sqr_ny_nearest(
                    distances.data(),
                    xsub,
                    pq.get_centroids(m, 0),
                    pq.dsub,
                    pq.ksub);
        } else {
            // 使用转置质心（更好的缓存局部性）
            idxm = fvec_L2sqr_ny_nearest_y_transposed(
                    distances.data(),
                    xsub,
                    pq.transposed_centroids.data() + m * pq.ksub,
                    pq.centroids_sq_lengths.data() + m * pq.ksub,
                    pq.dsub,
                    pq.M * pq.ksub,
                    pq.ksub);
        }

        encoder.encode(idxm);
    }
}

void ProductQuantizer::compute_code(const float* x, uint8_t* code) const {
    switch (nbits) {
        case 8:
            faiss::compute_code<PQEncoder8>(*this, x, code);
            break;

        case 16:
            faiss::compute_code<PQEncoder16>(*this, x, code);
            break;

        default:
            faiss::compute_code<PQEncoderGeneric>(*this, x, code);
            break;
    }
}
```

### 10.6 解码实现

```cpp
// 解码单个PQ码
template <class PQDecoder>
void decode(const ProductQuantizer& pq, const uint8_t* code, float* x) {
    PQDecoder decoder(code, pq.nbits);
    for (size_t m = 0; m < pq.M; m++) {
        uint64_t c = decoder.decode();
        memcpy(x + m * pq.dsub,
               pq.get_centroids(m, c),
               sizeof(float) * pq.dsub);
    }
}

void ProductQuantizer::decode(const uint8_t* code, float* x) const {
    switch (nbits) {
        case 8:
            faiss::decode<PQDecoder8>(*this, code, x);
            break;

        case 16:
            faiss::decode<PQDecoder16>(*this, code, x);
            break;

        default:
            faiss::decode<PQDecoderGeneric>(*this, code, x);
            break;
    }
}

// 批量解码（支持多线程）
void ProductQuantizer::decode(const uint8_t* code, float* x, size_t n) const {
#pragma omp parallel for if (n > 100)
    for (int64_t i = 0; i < n; i++) {
        this->decode(code + code_size * i, x + d * i);
    }
}
```

### 10.7 批量编码实现

```cpp
// block size used in ProductQuantizer::compute_codes
int product_quantizer_compute_codes_bs = 256 * 1024;

void ProductQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    // 分块处理避免使用过多内存
    size_t bs = product_quantizer_compute_codes_bs;
    if (n > bs) {
        for (size_t i0 = 0; i0 < n; i0 += bs) {
            size_t i1 = std::min(i0 + bs, n);
            compute_codes(x + d * i0, codes + code_size * i0, i1 - i0);
        }
        return;
    }

    if (dsub < 16) { // 简单直接计算

#pragma omp parallel for
        for (int64_t i = 0; i < n; i++)
            compute_code(x + i * d, codes + i * code_size);

    } else { // 使用BLAS（ worthwhile to use BLAS）
        std::unique_ptr<float[]> dis_tables(new float[n * ksub * M]);
        compute_distance_tables(n, x, dis_tables.get());

#pragma omp parallel for
        for (int64_t i = 0; i < n; i++) {
            uint8_t* code = codes + i * code_size;
            const float* tab = dis_tables.get() + i * ksub * M;
            compute_code_from_distance_table(tab, code);
        }
    }
}
```

### 10.8 距离表计算

```cpp
// 计算查询向量的距离表（用于ADC）
void ProductQuantizer::compute_distance_table(const float* x, float* dis_table)
        const {
    if (transposed_centroids.empty()) {
        // 使用常规版本
        for (size_t m = 0; m < M; m++) {
            fvec_L2sqr_ny(
                    dis_table + m * ksub,
                    x + m * dsub,
                    get_centroids(m, 0),
                    dsub,
                    ksub);
        }
    } else {
        // 使用转置质心（更好的缓存局部性）
        for (size_t m = 0; m < M; m++) {
            fvec_L2sqr_ny_transposed(
                    dis_table + m * ksub,
                    x + m * dsub,
                    transposed_centroids.data() + m * ksub,
                    centroids_sq_lengths.data() + m * ksub,
                    dsub,
                    M * ksub,
                    ksub);
        }
    }
}

// 计算内积表（用于内积度量）
void ProductQuantizer::compute_inner_prod_table(
        const float* x,
        float* dis_table) const {
    size_t m;

    for (m = 0; m < M; m++) {
        fvec_inner_products_ny(
                dis_table + m * ksub,
                x + m * dsub,
                get_centroids(m, 0),
                dsub,
                ksub);
    }
}
```

### 10.9 从距离表编码

```cpp
// 从预计算的距离表生成PQ码
void ProductQuantizer::compute_code_from_distance_table(
        const float* tab,
        uint8_t* code) const {
    PQEncoderGeneric encoder(code, nbits);
    for (size_t m = 0; m < M; m++) {
        float mindis = 1e20;
        uint64_t idxm = 0;

        /* 找到最佳质心 */
        for (size_t j = 0; j < ksub; j++) {
            float dis = *tab++;
            if (dis < mindis) {
                mindis = dis;
                idxm = j;
            }
        }

        encoder.encode(idxm);
    }
}
```

### 10.10 训练实现

```cpp
// faiss/impl/ProductQuantizer.cpp
// PQ训练函数
void ProductQuantizer::train(size_t n, const float* x) {
    if (train_type != Train_shared) {
        train_type_t final_train_type;
        final_train_type = train_type;

        // 检查超立方体训练的可行性
        if (train_type == Train_hypercube ||
            train_type == Train_hypercube_pca) {
            if (dsub < nbits) {
                final_train_type = Train_default;
                printf("cannot train hypercube: nbits=%zd > log2(d=%zd)\n",
                       nbits, dsub);
            }
        }

        std::unique_ptr<float[]> xslice(new float[n * dsub]);
        for (int m = 0; m < M; m++) {
            // 提取子量化器m的训练数据
            for (int j = 0; j < n; j++)
                memcpy(xslice.get() + j * dsub,
                       x + j * d + m * dsub,
                       dsub * sizeof(float));

            Clustering clus(dsub, ksub, cp);

            // 质心初始化
            if (final_train_type != Train_default) {
                clus.centroids.resize(dsub * ksub);
            }

            switch (final_train_type) {
                case Train_hypercube:
                    init_hypercube(
                            dsub, nbits, n,
                            xslice.get(),
                            clus.centroids.data());
                    break;
                case Train_hypercube_pca:
                    init_hypercube_pca(
                            dsub, nbits, n,
                            xslice.get(),
                            clus.centroids.data());
                    break;
                case Train_hot_start:
                    memcpy(clus.centroids.data(),
                           get_centroids(m, 0),
                           dsub * ksub * sizeof(float));
                    break;
                default:;
            }

            if (verbose) {
                clus.verbose = true;
                printf("Training PQ slice %d/%zd\n", m, M);
            }

            IndexFlatL2 index(dsub);
            clus.train(n, xslice.get(), assign_index ? *assign_index : index);
            set_params(clus.centroids.data(), m);
        }

    } else {
        // 共享字典训练
        Clustering clus(dsub, ksub, cp);

        if (verbose) {
            clus.verbose = true;
            printf("Training all PQ slices at once\n");
        }

        IndexFlatL2 index(dsub);
        clus.train(n * M, x, assign_index ? *assign_index : index);
        for (int m = 0; m < M; m++) {
            set_params(clus.centroids.data(), m);
        }
    }
}

// 超立方体初始化
static void init_hypercube(
        int d,
        int nbits,
        int n,
        const float* x,
        float* centroids) {
    std::vector<float> mean(d);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            mean[j] += x[i * d + j];

    float maxm = 0;
    for (int j = 0; j < d; j++) {
        mean[j] /= n;
        if (fabs(mean[j]) > maxm)
            maxm = fabs(mean[j]);
    }

    for (int i = 0; i < (1 << nbits); i++) {
        float* cent = centroids + i * d;
        for (int j = 0; j < nbits; j++)
            cent[j] = mean[j] + (((i >> j) & 1) ? 1 : -1) * maxm;
        for (int j = nbits; j < d; j++)
            cent[j] = mean[j];
    }
}
```

### 10.11 性能优化技巧

```cpp
// 使用BLAS进行批量距离表计算
void ProductQuantizer::compute_distance_tables(
        size_t nx,
        const float* x,
        float* dis_tables) const {
#if defined(__AVX2__) || defined(__aarch64__)
    if (dsub == 2 && nbits < 8) {
        // 特殊优化：dsub=2的窄范围
        compute_PQ_dis_tables_dsub2(
                d, ksub, centroids.data(), nx, x, false, dis_tables);
    } else
#endif
            if (dsub < 16) {
        // 小维度：直接计算
#pragma omp parallel for if (nx > 1)
        for (int64_t i = 0; i < nx; i++) {
            compute_distance_table(x + i * d, dis_tables + i * ksub * M);
        }

    } else {
        // 大维度：使用BLAS
        for (int m = 0; m < M; m++) {
            pairwise_L2sqr(
                    dsub,
                    nx,
                    x + dsub * m,
                    ksub,
                    centroids.data() + m * dsub * ksub,
                    dis_tables + ksub * m,
                    d,
                    dsub,
                    ksub * M);
        }
    }
}
```

### 10.12 PQ搜索实现（ADC）

```cpp
// ADC搜索：查询使用原始向量，数据库使用PQ码
void ProductQuantizer::search(
        const float* x,
        size_t nx,
        const uint8_t* codes,
        const size_t ncodes,
        float_maxheap_array_t* res) const {

    for (size_t i = 0; i < nx; i++) {
        // 1. 计算查询的距离表
        float* dis_table = new float[M * ksub];
        compute_distance_table(x + i * d, dis_table);

        // 2. 初始化堆
        float* simi = res->val + i * res->k;
        idx_t* idxi = res->ids + i * res->k;
        heap_heapify<CMax<float, idx_t>>(res->k, simi, idxi);

        // 3. 遍历所有PQ码
        for (size_t j = 0; j < ncodes; j++) {
            float dis = 0;

            // 累加所有子量化器的距离
            for (size_t m = 0; m < M; m++) {
                uint64_t idx = decode_uint(codes + j * code_size, m, nbits);
                dis += dis_table[m * ksub + idx];
            }

            // 更新堆
            if (CMax<float, idx_t>::cmp(dis, simi[0])) {
                heap_replace_top<CMax<float, idx_t>>(
                    res->k, simi, idxi, dis, j);
            }
        }

        heap_reorder<CMax<float, idx_t>>(res->k, simi, idxi);
        delete[] dis_table;
    }
}
```

### 10.13 内存与性能总结表

| 参数 | 典型值 | 内存占用 | 压缩比 | 备注 |
|------|--------|----------|--------|------|
| d=128, M=16, nbits=8 | - | 18 MB | 28x | 通用配置 |
| d=128, M=32, nbits=8 | - | 34 MB | 15x | 更高精度 |
| d=128, M=16, nbits=6 | - | 10 MB | 51x | 更高压缩 |
| d=512, M=64, nbits=8 | - | 72 MB | 28x | 高维向量 |

**编码性能**：
- 单线程编码速度：~10M向量/秒（d=128, M=16）
- 多线程（8核）编码速度：~60M向量/秒
- 使用BLAS加速（dsub>=16）：~2x加速

**搜索性能**：
- ADC搜索：~100M向量/秒（单查询，8核）
- SDC搜索：~500M向量/秒（单查询，8核）

---

## 9. 第4天总结

### 关键概念

1. **Product Quantization**：向量分解 + 子量化
2. **编码/解码**：浮点向量 ↔ 索引序列
3. **训练**：每个子量化器独立K-means
4. **ADC搜索**：查表法快速计算距离
5. **SDC搜索**：查询和数据库都用PQ码
6. **内存效率**：可达64倍压缩

### 下一步

第5天将学习**IVF（Inverted File）索引**，结合PQ实现高效的大规模向量搜索。

---

## 练习题

1. 实现简化的PQ编码器
2. 比较ADC和SDC的搜索性能
3. 分析不同M和nbits对精度的影响
4. 实现SIMD优化的距离表查找

---

## 11. SIMD优化的PQ编码

### 11.1 8位PQ编码器优化

```cpp
// faiss/impl/ProductQuantizer.cpp
// SIMD优化的8位PQ编码器（每个子量化器8位）
namespace {

// SIMD优化的PQ编码器（AVX2）
struct SimdPQEncoder {
    const float* x;          // 输入向量 [d]
    const float* centroids;  // 质心表 [M * 256 * dsub]
    const size_t dsub;
    const size_t M;

    // 编码单向量
    inline void encode(uint8_t* code) const {
        for (size_t m = 0; m < M; m++) {
            code[m] = encode_subquantizer(m);
        }
    }

private:
    // 编码单个子量化器
    inline uint8_t encode_subquantizer(size_t m) const {
        const float* xm = x + m * dsub;
        const float* centroids_m = centroids + m * 256 * dsub;

        // 256个质心，每个dsub维
        // 寻找L2距离最小的质心
        uint8_t best_code = 0;
        float min_dist = HUGE_VALF;

#ifdef __AVX2__
        // AVX2优化：每次比较8个质心的距离
        __m256 vmin_dist = _mm256_set1_ps(HUGE_VALF);
        __m256i vbest_code = _mm256_set1_epi32(0);

        size_t k = 0;
        // 假设dsub=8，每次加载和处理8个质心
        for (; k + 8 <= 256; k += 8) {
            // 加载8个质心（需要优化内存布局）
            // 实际Faiss中使用更复杂的展开
            __m256 vdist0 = _mm256_setzero_ps();
            // ... 距离计算展开

            // 比较最小值
            // vmin_dist = _mm256_min_ps(vmin_dist, vdist0);
            // vbest_code = 更新索引...
        }

        // 提取最佳索引
        // ...

#else
        // 标量版本
        for (size_t k = 0; k < 256; k++) {
            float dist = fvec_L2sqr(xm, centroids_m + k * dsub, dsub);
            if (dist < min_dist) {
                min_dist = dist;
                best_code = static_cast<uint8_t>(k);
            }
        }
#endif

        return best_code;
    }
};

} // namespace
```

### 11.2 SIMD优化的距离表查找

```cpp
// 预计算距离表的SIMD实现
void ProductQuantizer::compute_inner_prod_table(
        const float* x,
        float* dis_tables) const {

    // dis_tables布局: [M * ksub]
    // dis_tables[m * ksub + k] = <x_m, c_{m,k}>

    for (size_t m = 0; m < M; m++) {
        const float* xm = x + m * dsub;
        float* table_m = dis_tables + m * ksub;

        const float* centroids_m = get_centroids_transposed(m, 0);

#ifdef __AVX2__
        // AVX2优化：每次计算8个质心的内积
        size_t k = 0;

        if (dsub == 8) {
            // 特化：dsub=8的快速路径
            for (; k + 8 <= ksub; k += 8) {
                // 加载查询子向量
                __m256 vx = _mm256_loadu_ps(xm);

                // 加载8个质心
                __m256 vc0 = _mm256_loadu_ps(centroids_m + k * 8 + 0);
                __m256 vc1 = _mm256_loadu_ps(centroids_m + k * 8 + 1);
                __m256 vc2 = _mm256_loadu_ps(centroids_m + k * 8 + 2);
                __m256 vc3 = _mm256_loadu_ps(centroids_m + k * 8 + 3);
                __m256 vc4 = _mm256_loadu_ps(centroids_m + k * 8 + 4);
                __m256 vc5 = _mm256_loadu_ps(centroids_m + k * 8 + 5);
                __m256 vc6 = _mm256_loadu_ps(centroids_m + k * 8 + 6);
                __m256 vc7 = _mm256_loadu_ps(centroids_m + k * 8 + 7);

                // 计算8个内积
                __m256 vip0 = _mm256_mul_ps(vx, vc0);
                __m256 vip1 = _mm256_mul_ps(vx, vc1);
                __m256 vip2 = _mm256_mul_ps(vx, vc2);
                __m256 vip3 = _mm256_mul_ps(vx, vc3);
                __m256 vip4 = _mm256_mul_ps(vx, vc4);
                __m256 vip5 = _mm256_mul_ps(vx, vc5);
                __m256 vip6 = _mm256_mul_ps(vx, vc6);
                __m256 vip7 = _mm256_mul_ps(vx, vc7);

                // 水平求和（每个向量8维，求和为标量）
                alignas(32) float ips[8];
                _mm256_storeu_ps(ips, vip0); ips[0] = ips[1] + ips[2] + ips[3] + ips[4] + ips[5] + ips[6] + ips[7];
                _mm256_storeu_ps(ips + 1, _mm256_castps256_ps128(vip1));
                // ... 其他求和

                table_m[k + 0] = ips[0];
                table_m[k + 1] = ips[1];
                // ...
            }
        } else {
            // 通用dsub版本
            for (; k < ksub; k++) {
                const float* ck = centroids_m + k * dsub;
                float ip = 0;
                for (size_t j = 0; j < dsub; j++) {
                    ip += xm[j] * ck[j];
                }
                table_m[k] = ip;
            }
        }

#else
        // 标量版本
        for (size_t k = 0; k < ksub; k++) {
            const float* ck = centroids_m + k * dsub;
            float ip = 0;
            for (size_t j = 0; j < dsub; j++) {
                ip += xm[j] * ck[j];
            }
            table_m[k] = ip;
        }
#endif
    }
}
```

### 11.3 ADC距离计算的优化

```cpp
// 非对称距离计算（ADC）的SIMD优化
// 查表法：PQ编码的y，预计算距离表
inline float pq_distance_to_code_adc(
        const float* dis_tables,  // [M * ksub]
        const uint8_t* code,
        size_t M,
        size_t ksub) {

    float sum = 0.0f;

#ifdef __AVX2__
    // AVX2优化：累加8个表项
    size_t m = 0;

    while (m + 8 <= M) {
        // 加载8个code值
        __m128i vcode = _mm_loadu_si128((__m128i*)(code + m));

        // 提取8个索引
        uint8_t codes[8];
        _mm_storeu_si128((__m128i*)codes, vcode);

        // 从距离表查找8个距离
        const float* table_m0 = dis_tables + m * ksub;
        const float* table_m1 = dis_tables + (m+1) * ksub;
        // ... 等等

        __m256 vdist = _mm256_setr_ps(
            table_m0[codes[0]],
            table_m1[codes[1]],
            table_m2[codes[2]],
            table_m3[codes[3]],
            table_m4[codes[4]],
            table_m5[codes[5]],
            table_m6[codes[6]],
            table_m7[codes[7]]
        );

        // 水平求和
        sum += hsum_ps(vdist);

        m += 8;
    }

    // 处理剩余
    for (; m < M; m++) {
        sum += dis_tables[m * ksub + code[m]];
    }

#else
    // 标量版本
    for (size_t m = 0; m < M; m++) {
        sum += dis_tables[m * ksub + code[m]];
    }
#endif

    return sum;
}
```

### 11.4 水平求和优化

```cpp
// AVX2水平求和优化
inline float hsum_ps_avx2(__m256 v) {
    // 方法1: 使用hadd（简单但较慢）
    __m256 sum = _mm256_hadd_ps(v, v);
    sum = _mm256_hadd_ps(sum, sum);

    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    return tmp[0] + tmp[4];
}

// 方法2: 使用extract和add（更快）
inline float hsum_ps_fast(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);

    __m128 sum = _mm_add_ps(vlow, vhigh);

    __m128 shuf = _mm_movehdup_ps(sum);
    sum = _mm_add_ps(sum, shuf);

    return _mm_cvtss_f32(sum);
}

// 方法3: AVX-512专用（最快）
#ifdef __AVX512F__
inline float hsum_ps_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
#endif
```

---

## 12. PQ编码的位操作优化

### 12.1 通用位打包器

```cpp
// 处理任意位数的PQ编码
class BitPacker {
    uint8_t* buffer;
    size_t bit_offset;

public:
    BitPacker(uint8_t* buf) : buffer(buf), bit_offset(0) {}

    // 打包nbits位到buffer
    inline void pack(uint32_t value, int nbits) {
        uint64_t bit_mask = (1ULL << nbits) - 1;
        value &= bit_mask;

        size_t byte_offset = bit_offset / 8;
        size_t shift = bit_offset % 8;

        uint64_t* ptr64 = reinterpret_cast<uint64_t*>(buffer + byte_offset);

        // 写入位
        *ptr64 |= (static_cast<uint64_t>(value) << shift);
        bit_offset += nbits;
    }

    // 解包nbits位
    inline uint32_t unpack(int nbits) {
        size_t byte_offset = bit_offset / 8;
        size_t shift = bit_offset % 8;

        uint64_t* ptr64 = reinterpret_cast<uint64_t*>(buffer + byte_offset);
        uint64_t masked = (*ptr64 >> shift) & ((1ULL << nbits) - 1);

        bit_offset += nbits;
        return static_cast<uint32_t>(masked);
    }
};

// 8位打包优化（最常见情况）
inline void pack_8bit(uint8_t* code, size_t m, uint8_t value) {
    code[m] = value;  // 直接存储，无需位操作
}

// 非8位打包（例如6位）
inline void pack_6bit(uint8_t* code, size_t m, uint8_t value) {
    // 每4个6位值占用3个字节
    size_t byte_idx = (m * 6) / 8;
    size_t bit_shift = (m * 6) % 8;

    code[byte_idx] |= (value << bit_shift);
    if (bit_shift + 6 > 8) {
        code[byte_idx + 1] |= (value >> (8 - bit_shift));
    }
}
```

### 12.2 SIMD优化的批量解码

```cpp
// SIMD优化的PQ解码
void decode_pq_batch(
        const uint8_t* codes,
        const float* centroids_transposed,  // 转置布局
        float* x,
        size_t n, size_t d, size_t M) {

    size_t dsub = d / M;

    for (size_t m = 0; m < M; m++) {
        const float* centroids_m = centroids_transposed + m * 256 * dsub;
        size_t offset = m * dsub;

        for (size_t i = 0; i < n; i++) {
            uint8_t c = codes[i * M + m];
            const float* centroid = centroids_m + c * dsub;
            float* xi = x + i * d + offset;

            // SIMD优化的向量复制
            size_t j = 0;
#ifdef __AVX2__
            for (; j + 8 <= dsub; j += 8) {
                __m256 vc = _mm256_loadu_ps(centroid + j);
                _mm256_storeu_ps(xi + j, vc);
            }
#endif
            for (; j < dsub; j++) {
                xi[j] = centroid[j];
            }
        }
    }
}
```

---

## 13. PQ内存布局深度优化

### 13.1 转置质心表的性能优势

```cpp
// 原布局 vs 转置布局的缓存性能对比

// 原布局（centroids）：每个质心跳跃访问
struct CentroidLayout_Original {
    // centroids[m * ksub * dsub]
    // 访问模式：centroids[m * ksub * dsub + j]  - 跨步访问dsub个float

    float load_centroid_dim(size_t m, size_t k, size_t j) {
        return centroids[(m * ksub + k) * dsub + j];  // 跳跃访问
    }
};

// 转置布局：连续存储，缓存友好
struct CentroidLayout_Transposed {
    // transposed_centroids[(m * ksub + k) * dsub]
    // 访问模式：transposed_centroids[(m * ksub + k) * dsub + j]

    float load_centroid_dim(size_t m, size_t k, size_t j) {
        return transposed_centroids[(m * ksub + k) * dsub + j];  // 连续访问
    }
};

// 性能影响分析：
// 假设dsub=8, ksub=256
// 原布局：访问256个质心的第0维 -> 256个cache miss（假设不命中）
// 转置布局：访问256个质心的第0维 -> 4个cache miss（256/8=32字节/cache line）
// 缓存命中率提升：~64x
```

### 13.2 预取优化的编码

```cpp
// 使用预取优化编码性能
void encode_with_prefetch_optimized(
        const float* x,
        uint8_t* code,
        const float* centroids_transposed,
        size_t M, size_t dsub, size_t ksub) {

    constexpr size_t PREFETCH_AHEAD = 4;  // 预取未来4个子量化器

    for (size_t m = 0; m < M; m++) {
        // 预取未来的质心数据
        if (m + PREFETCH_AHEAD < M) {
            const float* future_centroids = centroids_transposed +
                                       (m + PREFETCH_AHEAD) * ksub * dsub;

            // 预取第一个质心（触发缓存行加载）
            _mm_prefetch((const char*)future_centroids, _MM_HINT_T0);
            // 预取第8个质心
            _mm_prefetch((const char*)(future_centroids + 8 * dsub), _MM_HINT_T0);
        }

        // 当前编码
        const float* xm = x + m * dsub;
        const float* centroids_m = centroids_transposed + m * ksub * dsub;

        float min_dist = HUGE_VALF;
        uint8_t best_k = 0;

        for (size_t k = 0; k < ksub; k++) {
            const float* ck = centroids_m + k * dsub;
            float dist = fvec_L2sqr(xm, ck, dsub);

            if (dist < min_dist) {
                min_dist = dist;
                best_k = static_cast<uint8_t>(k);
            }
        }

        code[m] = best_k;
    }
}
```

---

## 14. PQ性能优化总结

### 14.1 优化技术对比

| 优化技术 | 加速比 | 适用场景 | 实现难度 |
|---------|--------|----------|---------|
| 转置质心表 | 2-3x | dsub较小，ksub较大 | 低 |
| SIMD距离表计算 | 4-6x | 查表操作 | 中 |
| 预取优化 | 1.2-1.5x | 编码阶段 | 低 |
| 批量编码 | 1.5-2x | 多向量编码 | 中 |
| 8路展开 | 1.3x | dsub=8 | 中 |

### 14.2 编译选项建议

```bash
# 推荐的编译选项
g++ -O3 \
    -march=native \       # 启用当前CPU的所有指令集
    -mavx2 \             # 启用AVX2
    -mavx512f \           # 启用AVX-512（如果支持）
    -mfma \              # 启用FMA
    -funroll-loops \      # 循环展开
    -ffast-math \         # 激进的浮点优化
    -fopenmp \            # OpenMP并行
    -I/path/to/faiss \
    pq_optimized.cpp -o pq_optimized

# Intel编译器
icc -O3 -xHOST -qopenmp -ipo pq_optimized.cpp
```

---

## 15. PQ底层SIMD优化深入

### 15.1 AVX2优化的距离表计算

```cpp
// AVX2优化的距离表计算
// 计算查询向量与所有质心的距离
void compute_distance_table_avx2(
        const float* x,
        const float* centroids,
        float* dis_table,
        size_t M,
        size_t dsub,
        size_t ksub) {

    for (size_t m = 0; m < M; m++) {
        const float* xm = x + m * dsub;
        float* dt = dis_table + m * ksub;
        const float* centroids_m = centroids + m * ksub * dsub;

        // 展开循环：每次处理8个质心
        size_t k = 0;

        // 8路展开（AVX2处理8个float）
        for (; k + 8 <= ksub; k += 8) {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();

            // 对8个质心同时计算距离
            for (size_t j = 0; j < dsub; j++) {
                __m256 xj = _mm256_set1_ps(xm[j]);

                // 加载8个质心的第j维
                __m256 c0 = _mm256_loadu_ps(centroids_m + (k + 0) * dsub + j);
                __m256 c4 = _mm256_loadu_ps(centroids_m + (k + 4) * dsub + j);

                __m256 diff0 = _mm256_sub_ps(xj, c0);
                __m256 diff1 = _mm256_sub_ps(xj, c4);

                sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
                sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
            }

            // 水平求和并存储
            alignas(32) float tmp[8];
            __m256 sum = _mm256_add_ps(sum0, sum1);
            _mm256_storeu_ps(tmp, sum);

            dt[k + 0] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            dt[k + 1] = tmp[4] + tmp[5] + tmp[6] + tmp[7];
            // 实际实现需要完整的水平求和
        }

        // 处理剩余质心
        for (; k < ksub; k++) {
            dt[k] = fvec_L2sqr(xm, centroids_m + k * dsub, dsub);
        }
    }
}

// 更高效的实现：使用转置质心布局
void compute_distance_table_avx2_transposed(
        const float* x,
        const float* centroids_transposed,
        float* dis_table,
        size_t M,
        size_t dsub,
        size_t ksub) {

    for (size_t m = 0; m < M; m++) {
        const float* xm = x + m * dsub;
        float* dt = dis_table + m * ksub;
        const float* centroids_m = centroids_transposed + m * ksub * dsub;

        // 16个质心为一组（每个质心8维，共128维=2个缓存行）
        size_t k = 0;

        for (; k + 16 <= ksub; k += 16) {
            // 初始化16个累加器
            __m256 acc0 = _mm256_setzero_ps();  // 质心0-3
            __m256 acc1 = _mm256_setzero_ps();  // 质心4-7
            __m256 acc2 = _mm256_setzero_ps();  // 质心8-11
            __m256 acc3 = _mm256_setzero_ps();  // 质心12-15

            for (size_t j = 0; j < dsub; j++) {
                __m256 xj = _mm256_set1_ps(xm[j]);

                // 加载16个质心的第j维（连续内存）
                __m256 c0 = _mm256_loadu_ps(centroids_m + (k + 0) * dsub + j);
                __m256 c1 = _mm256_loadu_ps(centroids_m + (k + 4) * dsub + j);
                __m256 c2 = _mm256_loadu_ps(centroids_m + (k + 8) * dsub + j);
                __m256 c3 = _mm256_loadu_ps(centroids_m + (k + 12) * dsub + j);

                __m256 diff0 = _mm256_sub_ps(xj, c0);
                __m256 diff1 = _mm256_sub_ps(xj, c1);
                __m256 diff2 = _mm256_sub_ps(xj, c2);
                __m256 diff3 = _mm256_sub_ps(xj, c3);

                acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
                acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
                acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);
                acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
            }

            // 水平求和
            dt[k + 0] = hsum256_ps(acc0);
            dt[k + 1] = hsum256_ps(acc1);
            dt[k + 2] = hsum256_ps(acc2);
            dt[k + 3] = hsum256_ps(acc3);
            // ... 继续处理其他4个
        }

        // 处理剩余质心
        for (; k < ksub; k++) {
            dt[k] = fvec_L2sqr(xm, centroids_m + k * dsub, dsub);
        }
    }
}

// 水平求和辅助函数
inline float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(vlow, vhigh);

    __m128 shuf = _mm_movehdup_ps(sum);
    __m128 sums = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);

    return _mm_cvtss_f32(sums);
}
```

### 15.2 AVX-512优化的PQ编码

```cpp
#ifdef __AVX512F__
// AVX-512优化的编码：同时查找16个质心的最近邻
uint8_t find_nearest_centroid_avx512(
        const float* x,
        const float* centroids,
        size_t dsub,
        size_t ksub) {

    // 16路并行：同时与16个质心比较
    __m512 min_dist = _mm512_set1_ps(HUGE_VALF);
    __m512i min_indices = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8,
                                            7, 6, 5, 4, 3, 2, 1, 0);

    size_t k = 0;

    // 主循环：每次处理16个质心
    for (; k + 16 <= ksub; k += 16) {
        __m512 sum = _mm512_setzero_ps();

        // 计算与16个质心的L2距离
        for (size_t j = 0; j < dsub; j++) {
            __m512 xj = _mm512_set1_ps(x[j]);

            // 加载16个质心的第j维
            __m512 c = _mm512_loadu_ps(centroids + k * dsub + j);

            __m512 diff = _mm512_sub_ps(xj, c);
            sum = _mm512_fmadd_ps(diff, diff, sum);
        }

        // 比较并更新最小值
        __mmask16 lt_mask = _mm512_cmp_ps_mask(sum, min_dist, _CMP_LT_OQ);

        min_dist = _mm512_mask_loadu_ps(min_dist, lt_mask,
                                        (float*)&sum);  // 使用掩码更新

        // 更新索引
        __m512i current_indices = _mm512_set1_epi32(k);
        __m512i indices = _mm512_add_epi32(
            current_indices,
            _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8,
                            7, 6, 5, 4, 3, 2, 1, 0));

        min_indices = _mm512_mask_loadu_epi32(min_indices, lt_mask,
                                             (int*)&indices);
    }

    // 提取最小距离的索引
    alignas(64) float dist_array[16];
    alignas(64) int idx_array[16];
    _mm512_storeu_ps(dist_array, min_dist);
    _mm512_storeu_si512((__m512i*)idx_array, min_indices);

    float min_val = dist_array[0];
    uint8_t best_idx = idx_array[0];

    for (int i = 1; i < 16; i++) {
        if (dist_array[i] < min_val) {
            min_val = dist_array[i];
            best_idx = idx_array[i];
        }
    }

    // 处理剩余质心
    for (; k < ksub; k++) {
        float dist = fvec_L2sqr(x, centroids + k * dsub, dsub);
        if (dist < min_val) {
            min_val = dist;
            best_idx = k;
        }
    }

    return best_idx;
}

// AVX-512优化的批量编码
void compute_codes_avx512(
        const float* x,
        uint8_t* codes,
        size_t n,
        const float* centroids,
        size_t M,
        size_t dsub,
        size_t ksub) {

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        const float* xi = x + i * M * dsub;
        uint8_t* codei = codes + i * M;

        for (size_t m = 0; m < M; m++) {
            const float* centroids_m = centroids + m * ksub * dsub;
            codei[m] = find_nearest_centroid_avx512(
                xi + m * dsub, centroids_m, dsub, ksub);
        }
    }
}
#endif
```

### 15.3 SIMD优化的ADC距离查找表

```cpp
// SIMD优化的距离表查找（ADC核心操作）
// 输入：距离表(M x ksub)，PQ码(M个uint8)
// 输出：累加距离
inline float pq_distance_lookup_avx2(
        const float* dis_table,
        const uint8_t* code,
        size_t M,
        size_t ksub) {

    // 假设code是8位PQ码（每个子量化器1字节）
    // 一次处理8个子量化器

    size_t m = 0;
    __m256 sum = _mm256_setzero_ps();

    // 8路展开：每次处理8个子量化器
    for (; m + 8 <= M; m += 8) {
        // 加载8个索引
        __m128i idx8 = _mm_loadl_epi64((__m128i*)(code + m));

        // 扩展为32位整数
        __m256i idx = _mm256_cvtepu8_epi32(idx8);  // 前4个
        __m256i idx_hi = _mm256_cvtepu8_epi32(
            _mm_srli_si128(idx8, 4));  // 后4个

        // 计算表偏移量
        __m256i offset0 = _mm256_add_epi32(
            _mm256_set1_epi32((m + 0) * ksub), idx);
        __m256i offset1 = _mm256_add_epi32(
            _mm256_set1_epi32((m + 4) * ksub), idx_hi);

        // gather：从dis_table收集数据
        __m256 dist0 = _mm256_i32gather_ps(dis_table, offset0, 4);
        __m256 dist1 = _mm256_i32gather_ps(dis_table, offset1, 4);

        // 合并两个128位向量
        __m256 dist = _mm256_permute2f128_ps(dist0, dist1, 0x20);

        sum = _mm256_add_ps(sum, dist);
    }

    // 水平求和
    float result = hsum256_ps(sum);

    // 处理剩余子量化器
    for (; m < M; m++) {
        uint8_t idx = code[m];
        result += dis_table[m * ksub + idx];
    }

    return result;
}

// AVX2 gather优化（更高效）
inline float pq_distance_lookup_gather_avx2(
        const float* dis_table,
        const uint8_t* code,
        size_t M,
        size_t ksub) {

    // 预计算索引
    alignas(32) int indices[8];

    size_t m = 0;
    __m256 sum = _mm256_setzero_ps();

    for (; m + 8 <= M; m += 8) {
        // 构造gather索引
        for (int i = 0; i < 8; i++) {
            indices[i] = (m + i) * ksub + code[m + i];
        }

        // 一次gather 8个距离值
        __m256i idx_vec = _mm256_loadu_si256((__m256i*)indices);
        __m256 distances = _mm256_i32gather_ps(dis_table, idx_vec, 4);

        sum = _mm256_add_ps(sum, distances);
    }

    float result = hsum256_ps(sum);

    for (; m < M; m++) {
        result += dis_table[m * ksub + code[m]];
    }

    return result;
}
```

### 15.4 批量ADC搜索优化

```cpp
// 批量搜索：一次处理多个查询
void batch_pq_search_avx2(
        const float* queries,
        size_t nq,
        const uint8_t* codes,
        size_t ncodes,
        const float* centroids,
        size_t M, size_t dsub, size_t ksub,
        size_t k,
        float* distances,
        idx_t* labels) {

    // 为每个查询预计算距离表
    std::vector<float> dis_tables(nq * M * ksub);

    #pragma omp parallel for
    for (size_t q = 0; q < nq; q++) {
        compute_distance_table_avx2_transposed(
            queries + q * M * dsub,
            centroids,
            dis_tables.data() + q * M * ksub,
            M, dsub, ksub
        );
    }

    // 搜索
    #pragma omp parallel for
    for (size_t q = 0; q < nq; q++) {
        const float* dis_table = dis_tables.data() + q * M * ksub;
        float* simi = distances + q * k;
        idx_t* idxi = labels + q * k;

        // 初始化堆
        heap_heapify<CMax<float, idx_t>>(k, simi, idxi);

        // 遍历所有数据库向量
        for (size_t j = 0; j < ncodes; j++) {
            const uint8_t* code = codes + j * M;

            // SIMD优化的距离查找
            float dis = pq_distance_lookup_gather_avx2(
                dis_table, code, M, ksub);

            // 更新堆
            if (CMax<float, idx_t>::cmp(dis, simi[0])) {
                heap_replace_top<CMax<float, idx_t>>(
                    k, simi, idxi, dis, j);
            }
        }

        heap_reorder<CMax<float, idx_t>>(k, simi, idxi);
    }
}
```

### 15.5 查找表SIMD优化

```cpp
// PQ查找表优化：使用SIMD查找表加速
// 用于8位PQ码的快速距离累加

class PQLookupTable {
    alignas(64) float table[256];  // 64字节对齐，避免缓存行分裂

public:
    // 为子量化器m构造查找表
    // table[k] = ||x_m - c_(m,k)||^2
    void build(const float* x_sub, const float* centroids_m,
               size_t dsub, size_t ksub) {

        // 使用SIMD并行计算所有质心的距离
        size_t k = 0;

        // 8路并行（AVX2）
        for (; k + 8 <= ksub; k += 8) {
            __m256 sum = _mm256_setzero_ps();

            for (size_t j = 0; j < dsub; j++) {
                __m256 xj = _mm256_set1_ps(x_sub[j]);

                // 加载8个质心的第j维
                __m256 c = _mm256_loadu_ps(centroids_m + k * dsub + j);

                __m256 diff = _mm256_sub_ps(xj, c);
                sum = _mm256_fmadd_ps(diff, diff, sum);
            }

            // 存储到查找表
            _mm256_storeu_ps(table + k, sum);

            // 水平求和得到8个距离值
            for (int i = 0; i < 8; i++) {
                // 实际实现需要完整的水平求和
                table[k + i] = fvec_L2sqr(
                    x_sub, centroids_m + (k + i) * dsub, dsub);
            }
        }

        // 处理剩余质心
        for (; k < ksub; k++) {
            table[k] = fvec_L2sqr(
                x_sub, centroids_m + k * dsub, dsub);
        }
    }

    // 快速查找：一次查表8个值
    __m256 lookup_8(const uint8_t* indices) const {
        // 加载8个索引
        __m128i idx8 = _mm_loadl_epi64((__m128i*)indices);

        // 使用gather从表查找
        __m256i idx = _mm256_cvtepu8_epi32(idx8);
        __m256 result = _mm256_i32gather_ps(table, idx, 4);

        return result;
    }

    // 标量查找
    inline float lookup(uint8_t index) const {
        return table[index];
    }
};

// 使用查找表的批量搜索
void pq_search_with_lookup_tables(
        const float* query,
        const uint8_t* codes,
        size_t ncodes,
        const ProductQuantizer& pq,
        size_t k,
        float* distances,
        idx_t* labels) {

    // 为每个子量化器构造查找表
    std::vector<PQLookupTable> tables(pq.M);

    for (size_t m = 0; m < pq.M; m++) {
        tables[m].build(query + m * pq.dsub,
                       pq.get_centroids(m, 0),
                       pq.dsub, pq.ksub);
    }

    // 搜索
    for (size_t j = 0; j < ncodes; j++) {
        const uint8_t* code = codes + j * pq.M;

        float dis = 0.0f;
        size_t m = 0;

        // 8路展开：每次处理8个子量化器
        for (; m + 8 <= pq.M; m += 8) {
            __m256 sum = _mm256_setzero_ps();

            for (int i = 0; i < 8; i++) {
                __m256 dist = tables[m + i].lookup_8(code + m + i);
                sum = _mm256_add_ps(sum, dist);
            }

            dis += hsum256_ps(sum);
        }

        // 处理剩余
        for (; m < pq.M; m++) {
            dis += tables[m].lookup(code[m]);
        }

        // 更新堆...
    }
}
```

### 15.6 内存对齐优化

```cpp
// 内存对齐的PQ编码器
class AlignedPQEncoder {
    alignas(64) uint8_t buffer[256];  // 64字节对齐

public:
    // 对齐编码：确保PQ码从缓存行边界开始
    void encode_aligned(
            const float* x,
            uint8_t* codes,
            const ProductQuantizer& pq,
            size_t n) {

        // 确保codes对齐到64字节
        size_t aligned_n = (n + 63) / 64 * 64;

        for (size_t i = 0; i < n; i++) {
            const float* xi = x + i * pq.d;
            uint8_t* codei = codes + i * pq.code_size;

            pq.compute_code(xi, codei);
        }

        // 填充剩余空间（保持对齐）
        for (size_t i = n; i < aligned_n; i++) {
            memset(codes + i * pq.code_size, 0, pq.code_size);
        }
    }
};

// 预取优化的批量解码
void decode_batch_with_prefetch(
        const uint8_t* codes,
        float* x,
        const ProductQuantizer& pq,
        size_t n) {

    constexpr size_t PREFETCH_DISTANCE = 8;

    for (size_t i = 0; i < n; i++) {
        // 预取未来的PQ码
        if (i + PREFETCH_DISTANCE < n) {
            _mm_prefetch((const char*)(codes + (i + PREFETCH_DISTANCE) * pq.code_size),
                        _MM_HINT_T0);
        }

        const uint8_t* codei = codes + i * pq.code_size;
        float* xi = x + i * pq.d;

        pq.decode(codei, xi);
    }
}
```

### 15.7 多线程PQ训练优化

```cpp
// 并行化的PQ训练
void train_pq_parallel(
        ProductQuantizer& pq,
        size_t n,
        const float* x) {

    // 为每个子量化器独立训练（完全并行）
    #pragma omp parallel for
    for (size_t m = 0; m < pq.M; m++) {
        // 提取子量化器m的训练数据
        std::vector<float> sub_data(n * pq.dsub);

        for (size_t i = 0; i < n; i++) {
            memcpy(sub_data.data() + i * pq.dsub,
                   x + i * pq.d + m * pq.dsub,
                   pq.dsub * sizeof(float));
        }

        // K-means聚类
        Clustering clus(pq.dsub, pq.ksub);
        clus.train(n, sub_data.data(), pq.cp);

        // 保存质心（需要互斥锁）
        #pragma omp critical
        {
            for (size_t k = 0; k < pq.ksub; k++) {
                memcpy(pq.get_centroids(m, k),
                       clus.centroids + k * pq.dsub,
                       pq.dsub * sizeof(float));
            }
        }
    }

    // 计算质心范数和转置表
    pq.sync_centroid_sq_lengths();
    pq.sync_transposed_centroids();
}

// SIMD优化的K-means分配
void assign_to_nearest_simd(
        const float* x,
        size_t n,
        const float* centroids,
        size_t k,
        size_t d,
        idx_t* assign) {

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        const float* xi = x + i * d;

        float min_dis = HUGE_VALF;
        idx_t best = 0;

        // 8路并行
        size_t j = 0;
        for (; j + 8 <= k; j += 8) {
            __m256 sum = _mm256_setzero_ps();

            for (size_t dim = 0; dim < d; dim++) {
                __m256 xi_dim = _mm256_set1_ps(xi[dim]);
                __m256 c_dim = _mm256_loadu_ps(
                    centroids + j * d + dim);

                __m256 diff = _mm256_sub_ps(xi_dim, c_dim);
                sum = _mm256_fmadd_ps(diff, diff, sum);
            }

            // 查找最小值
            alignas(32) float tmp[8];
            _mm256_storeu_ps(tmp, sum);

            for (int ii = 0; ii < 8; ii++) {
                if (tmp[ii] < min_dis) {
                    min_dis = tmp[ii];
                    best = j + ii;
                }
            }
        }

        // 处理剩余质心
        for (; j < k; j++) {
            float dis = fvec_L2sqr(xi, centroids + j * d, d);
            if (dis < min_dis) {
                min_dis = dis;
                best = j;
            }
        }

        assign[i] = best;
    }
}
```

### 15.8 NEON优化的PQ编码（ARM64）

```cpp
#ifdef __aarch64__
// ARM NEON优化的PQ编码
uint8_t find_nearest_centroid_neon(
        const float* x,
        const float* centroids,
        size_t dsub,
        size_t ksub) {

    float32x4_t min_dist = vdupq_n_f32(HUGE_VALF);
    uint32_t min_idx = 0;

    size_t k = 0;

    // 4路并行（NEON处理4个float）
    for (; k + 4 <= ksub; k += 4) {
        float32x4_t sum = vdupq_n_f32(0.0f);

        for (size_t j = 0; j < dsub; j++) {
            float32x4_t xj = vdupq_n_f32(x[j]);
            float32x4_t c = vld1q_f32(centroids + k * dsub + j);

            float32x4_t diff = vsubq_f32(xj, c);
            sum = vfmaq_f32(sum, diff, diff);
        }

        // 查找最小值
        float32x4_t cmp = vcltq_f32(sum, min_dist);

        if (vaddvq_u32(vshrn_n_u32(vreinterpretq_u32_u32(cmp), 1))) {
            // 有更小的距离，更新
            alignas(16) float tmp[4];
            vst1q_f32(tmp, sum);

            for (int i = 0; i < 4; i++) {
                if (tmp[i] < vgetq_lane_f32(min_dist, 0)) {
                    min_dist = vsetq_lane_f32(tmp[i], min_dist, 0);
                    min_idx = k + i;
                }
            }
        }
    }

    // 处理剩余质心
    for (; k < ksub; k++) {
        float dist = fvec_L2sqr(x, centroids + k * dsub, dsub);
        if (dist < vgetq_lane_f32(min_dist, 0)) {
            min_dist = vsetq_lane_f32(dist, min_dist, 0);
            min_idx = k;
        }
    }

    return min_idx;
}
#endif
```

### 15.9 PQ编码性能基准测试

```cpp
// 性能测试框架
struct PQBenchmarkResult {
    double encode_time_ms;
    double decode_time_ms;
    double search_time_ms;
    double throughput_mbps;
};

PQBenchmarkResult benchmark_pq(
        const float* data,
        size_t n,
        const ProductQuantizer& pq) {

    PQBenchmarkResult result;

    // 编码测试
    std::vector<uint8_t> codes(n * pq.code_size);

    auto start = std::chrono::high_resolution_clock::now();
    pq.compute_codes(data, codes.data(), n);
    auto end = std::chrono::high_resolution_clock::now();

    result.encode_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    // 解码测试
    std::vector<float> decoded(n * pq.d);

    start = std::chrono::high_resolution_clock::now();
    pq.decode(codes.data(), decoded.data(), n);
    end = std::chrono::high_resolution_clock::now();

    result.decode_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    // 搜索测试
    size_t nq = 100;
    size_t k = 10;

    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    start = std::chrono::high_resolution_clock::now();
    pq.search(data, nq, codes.data(), n,
              float_maxheap_array_t{nq, k, labels.data(), distances.data()});
    end = std::chrono::high_resolution_clock::now();

    result.search_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    // 计算吞吐量
    size_t total_bytes = n * pq.d * sizeof(float);
    result.throughput_mbps = total_bytes / (result.encode_time_ms / 1000.0) / 1e6;

    return result;
}

void print_benchmark_results(const PQBenchmarkResult& result) {
    printf("PQ Performance:\n");
    printf("  Encode: %.3f ms\n", result.encode_time_ms);
    printf("  Decode: %.3f ms\n", result.decode_time_ms);
    printf("  Search: %.3f ms\n", result.search_time_ms);
    printf("  Throughput: %.2f MB/s\n", result.throughput_mbps);
}
```

---

## 16. OPQ (Optimized Product Quantization)

### 16.1 OPQ概述

OPQ通过学习一个旋转矩阵来优化PQ的性能，使旋转后的数据更适合PQ分解。

```cpp
// OPQ通过旋转矩阵优化PQ
struct OPQMatrix {
    size_t d;       // 原始维度
    size_t M;       // 子量化器数
    size_t dsub;    // dsub = d / M

    // 旋转矩阵：d x d（正交矩阵）
    std::vector<float> R;

    // OPQ训练类型
    enum OPQTrainType {
        OPQ_TRAIN_DEFAULT,        // 默认训练
        OPQ_TRAIN_PER_INDEX,      // 针对特定索引优化
        OPQ_TRAIN_SVD,            // 基于SVD的训练
        OPQ_TRAIN_NORM            // 基于范数的训练
    };

    OPQTrainType train_type = OPQ_TRAIN_DEFAULT;
};
```

### 16.2 SVD-based OPQ训练

```cpp
// 基于SVD的OPQ旋转矩阵训练
struct SVDOPQTrainer {
    // 使用PCA/SVD方法训练OPQ旋转矩阵
    static void train_svd_opq(
            const float* x,          // 训练数据：n x d
            size_t n,
            size_t d,
            size_t M,                // 子量化器数
            float* R_out) {          // 输出：d x d 旋转矩阵

        // 1. 计算协方差矩阵
        std::vector<float> cov(d * d, 0.0f);

        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            const float* xi = x + i * d;

            for (size_t r = 0; r < d; r++) {
                for (size_t c = r; c < d; c++) {
                    float val = xi[r] * xi[c];
                    #pragma omp atomic
                    cov[r * d + c] += val;
                }
            }
        }

        // 对称矩阵
        for (size_t r = 0; r < d; r++) {
            for (size_t c = 0; c < r; c++) {
                cov[r * d + c] = cov[c * d + r];
            }
        }

        // 2. 特征值分解（协方差矩阵）
        // 这里简化为使用SVD
        // 实际实现可以使用Eigen或LAPACK

        // 3. 使用前M个主成分作为旋转矩阵
        // R由前M个特征向量组成

        // 简化版本：对角矩阵（独立处理每个维度）
        memset(R_out, 0, d * d * sizeof(float));
        for (size_t i = 0; i < d; i++) {
            R_out[i * d + i] = 1.0f;
        }

        // 实际实现应该计算协方差矩阵的主成分
    }

    // 应用旋转
    static void apply_rotation(
            const float* x,
            size_t n,
            size_t d,
            const float* R,
            float* x_rotated) {

        // x_rotated = x * R^T
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                float sum = 0;
                for (size_t k = 0; k < d; k++) {
                    sum += x[i * d + k] * R[k * d + j];
                }
                x_rotated[i * d + j] = sum;
            }
        }
    }
};
```

### 16.3 SIMD优化的OPQ旋转

```cpp
// SIMD优化的OPQ旋转应用
#ifdef __AVX2__

struct AVX2OPQRotation {
    // 应用旋转矩阵：8维一组
    static inline void apply_rotation_avx2(
            const float* x,      // n x d
            const float* R,      // d x d 旋转矩阵
            size_t n,
            size_t d,
            float* x_out) {

        // 对于8x8块的旋转
        if (d == 8) {
            for (size_t i = 0; i < n; i++) {
                __m256 xi = _mm256_loadu_ps(x + i * d);

                // 计算 xi * R^T（每8个元素）
                __m256 result = _mm256_setzero_ps();

                for (size_t j = 0; j < 8; j++) {
                    __m256 R_col = _mm256_loadu_ps(R + j * d);

                    // xi[j] * R_col
                    __m256 broadcast = _mm256_set1_ps(xi[0]);  // 简化
                    __m256 contrib = _mm256_mul_ps(broadcast, R_col);

                    result = _mm256_add_ps(result, contrib);
                }

                _mm256_storeu_ps(x_out + i * d, result);
            }
        } else {
            // 通用维度处理
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < d; j++) {
                    float sum = 0;
                    for (size_t k = 0; k < d; k++) {
                        sum += x[i * d + k] * R[k * d + j];
                    }
                    x_out[i * d + j] = sum;
                }
            }
        }
    }

    // 批量旋转（处理多个查询）
    static inline void batch_rotation_queries(
            const float* queries,  // nq x d
            const float* R,
            size_t nq,
            size_t d,
            float* queries_rotated) {

        // 每次处理8个查询
        size_t q = 0;
        for (; q + 8 <= nq; q += 8) {
            // 对于8个查询，对每一维应用旋转
            for (size_t j = 0; j < d; j++) {
                // 收集8个查询的第j维
                __m256 q_j = {
                    queries[(q + 0) * d + j],
                    queries[(q + 1) * d + j],
                    queries[(q + 2) * d + j],
                    queries[(q + 3) * d + j],
                    queries[(q + 4) * d + j],
                    queries[(q + 5) * d + j],
                    queries[(q + 6) * d + j],
                    queries[(q + 7) * d + j]
                };

                // 应用旋转矩阵的第j行
                __m256 rotated_j = _mm256_setzero_ps();

                for (size_t k = 0; k < d; k++) {
                    __m256 R_jk = _mm256_set1_ps(R[j * d + k]);
                    rotated_j = _mm256_fmadd_ps(R_jk, q_j, rotated_j);
                }

                // 存储旋转后的结果
                queries_rotated[(q + 0) * d + j] = rotated_j[0];
                queries_rotated[(q + 1) * d + j] = rotated_j[1];
                queries_rotated[(q + 2) * d + j] = rotated_j[2];
                queries_rotated[(q + 3) * d + j] = rotated_j[3];
                queries_rotated[(q + 4) * d + j] = rotated_j[4];
                queries_rotated[(q + 5) * d + j] = rotated_j[5];
                queries_rotated[(q + 6) * d + j] = rotated_j[6];
                queries_rotated[(q + 7) * d + j] = rotated_j[7];
            }
        }

        // 处理剩余查询
        for (; q < nq; q++) {
            for (size_t j = 0; j < d; j++) {
                float sum = 0;
                for (size_t k = 0; k < d; k++) {
                    sum += queries[q * d + k] * R[k * d + j];
                }
                queries_rotated[q * d + j] = sum;
            }
        }
    }
};
#endif
```

### 16.4 Cache-Friendly的PQ实现

```cpp
// 缓存友好的PQ实现
struct CacheFriendlyPQ {
    // 将PQ码重新组织为缓存行友好的布局
    struct CacheLineLayout {
        size_t cache_line_size = 64;
        size_t vectors_per_line;

        void reorganize_pq_codes(
                const uint8_t* codes,  // n x M
                size_t n,
                size_t M,
                uint8_t* reorganized) {

            // 原始布局：按向量存储
            // codes[0] = [m0, m1, m2, ..., mM-1]
            // codes[1] = [m0, m1, m2, ..., mM-1]

            // 重组布局：按子量化器和缓存行存储
            // 每个缓存行包含同一子量化器的一组编码
            vectors_per_line = cache_line_size;

            size_t nlines = (n + vectors_per_line - 1) / vectors_per_line;

            for (size_t m = 0; m < M; m++) {
                for (size_t line = 0; line < nlines; line++) {
                    size_t start = line * vectors_per_line;
                    size_t end = std::min(start + vectors_per_line, n);

                    // 复制这组向量的第m个编码
                    for (size_t i = start; i < end; i++) {
                        reorganized[line * cache_line_size + (i - start)] =
                            codes[i * M + m];
                    }

                    // 填充剩余位置
                    for (size_t i = end - start; i < vectors_per_line; i++) {
                        reorganized[line * cache_line_size + i] = 0;
                    }
                }
            }
        }
    };

    // 使用缓存友好布局的搜索
    static void search_cache_friendly(
            const float* dis_tables,  // M x ksub
            const uint8_t* reorganized_codes,
            size_t M,
            size_t ksub,
            size_t n,
            float* distances) {

        // 每次处理一个缓存行的向量
        size_t vectors_per_line = 64;

        // 初始化距离
        memset(distances, 0, n * sizeof(float));

        // 对于每个子量化器
        for (size_t m = 0; m < M; m++) {
            const float* dt_m = dis_tables + m * ksub;
            const uint8_t* codes_m = reorganized_codes + m * n;

            // 处理每个缓存行
            size_t nlines = (n + vectors_per_line - 1) / vectors_per_line;

            for (size_t line = 0; line < nlines; line++) {
                // 加载整个缓存行（64字节）
                __m512i codes = _mm512_loadu_si512(
                    (__m512i*)(codes_m + line * 64));

                // 查找表累加距离
                for (size_t i = 0; i < vectors_per_line; i++) {
                    size_t idx = line * vectors_per_line + i;
                    if (idx >= n) break;

                    uint8_t code = ((uint8_t*)&codes)[i];
                    distances[idx] += dt_m[code];
                }
            }
        }
    }
};
```

### 16.5 SIMD优化的对称距离计算（SDC）

```cpp
// SIMD优化的对称距离计算
// SDC预计算：sd_table[k1 * ksub + k2] = ||c1 - c2||^2
struct SIMDSDC {
    static void compute_sdc_table_simd(
            const float* centroids,  // M x ksub x dsub
            size_t M,
            size_t ksub,
            size_t dsub,
            float* sdc_table) {     // M x ksub x ksub

        for (size_t m = 0; m < M; m++) {
            const float* cent_m = centroids + m * ksub * dsub;
            float* sdc_m = sdc_table + m * ksub * ksub;

            // 计算该子量化器所有质心对的距离
            for (size_t k1 = 0; k1 < ksub; k1++) {
                const float* c1 = cent_m + k1 * dsub;

#ifdef __AVX2__
                // SIMD处理8个质心对
                size_t k2 = 0;
                for (; k2 + 8 <= ksub; k2 += 8) {
                    __m256 sum = _mm256_setzero_ps();

                    for (size_t j = 0; j < dsub; j++) {
                        __m256 c1_val = _mm256_set1_ps(c1[j]);

                        // 加载8个质心的第j维
                        __m256 c2_0 = _mm256_set1_ps(cent_m[(k2 + 0) * dsub + j]);
                        __m256 c2_1 = _mm256_set1_ps(cent_m[(k2 + 1) * dsub + j]);
                        __m256 c2_2 = _mm256_set1_ps(cent_m[(k2 + 2) * dsub + j]);
                        __m256 c2_3 = _mm256_set1_ps(cent_m[(k2 + 3) * dsub + j]);
                        __m256 c2_4 = _mm256_set1_ps(cent_m[(k2 + 4) * dsub + j]);
                        __m256 c2_5 = _mm256_set1_ps(cent_m[(k2 + 5) * dsub + j]);
                        __m256 c2_6 = _mm256_set1_ps(cent_m[(k2 + 6) * dsub + j]);
                        __m256 c2_7 = _mm256_set1_ps(cent_m[(k2 + 7) * dsub + j]);

                        __m256 diff_0 = _mm256_sub_ps(c1_val, c2_0);
                        __m256 diff_1 = _mm256_sub_ps(c1_val, c2_1);
                        __m256 diff_2 = _mm256_sub_ps(c1_val, c2_2);
                        __m256 diff_3 = _mm256_sub_ps(c1_val, c2_3);
                        __m256 diff_4 = _mm256_sub_ps(c1_val, c2_4);
                        __m256 diff_5 = _mm256_sub_ps(c1_val, c2_5);
                        __m256 diff_6 = _mm256_sub_ps(c1_val, c2_6);
                        __m256 diff_7 = _mm256_sub_ps(c1_val, c2_7);

                        sum = _mm256_fmadd_ps(diff_0, diff_0, sum);
                        sum = _mm256_fmadd_ps(diff_1, diff_1, sum);
                        sum = _mm256_fmadd_ps(diff_2, diff_2, sum);
                        sum = _mm256_fmadd_ps(diff_3, diff_3, sum);
                        sum = _mm256_fmadd_ps(diff_4, diff_4, sum);
                        sum = _mm256_fmadd_ps(diff_5, diff_5, sum);
                        sum = _mm256_fmadd_ps(diff_6, diff_6, sum);
                        sum = _mm256_fmadd_ps(diff_7, diff_7, sum);
                    }

                    _mm256_storeu_ps(sdc_m + k1 * ksub + k2, sum);
                }

                // 处理剩余质心
                for (; k2 < ksub; k2++) {
                    float dis = fvec_L2sqr(c1, cent_m + k2 * dsub, dsub);
                    sdc_m[k1 * ksub + k2] = dis;
                }
#else
                // 标量版本
                for (size_t k2 = 0; k2 < ksub; k2++) {
                    float dis = fvec_L2sqr(c1, cent_m + k2 * dsub, dsub);
                    sdc_m[k1 * ksub + k2] = dis;
                }
#endif
            }
        }
    }

    // 使用SDC表进行对称距离计算
    static inline float sdc_distance(
            const uint8_t* code1,
            const uint8_t* code2,
            const float* sdc_table,  // M x ksub x ksub
            size_t M,
            size_t ksub) {

        float dis = 0;
        for (size_t m = 0; m < M; m++) {
            uint8_t c1 = code1[m];
            uint8_t c2 = code2[m];
            dis += sdc_table[m * ksub * ksub + c1 * ksub + c2];
        }
        return dis;
    }
};
```

---

## 扩展阅读
- faiss/impl/ProductQuantizer.cpp - PQ实现
- faiss/IndexPQ.h - PQ索引
- faiss/impl/pq_convolut.c - SIMD优化的PQ计算
- [PQ论文](https://lear.inrialpes.fr/pubs/2011/Jegou11/Jegou11.pdf) - Product Quantization for Nearest Neighbor Search
- [OPQ论文](https://arxiv.org/abs/1509.04014) - Optimized Product Quantization
- [Additive Quantization论文](https://arxiv.org/abs/1706.00984) - Additive Quantization for High-Dimensional Vectors
