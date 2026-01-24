# Faiss深度课程 - 第20天：距离度量与学习

## 课程目标

深入理解各种距离度量的数学原理、实现细节，以及如何学习最优的距离度量来提升检索性能。

---

## 1. 距离度量基础

### 1.1 度量空间公理

```cpp
// 距离度量必须满足的公理
struct MetricSpaceAxioms {
    // 对于度量d: X × X → R，必须满足：

    // 1. 非负性：d(x, y) ≥ 0
    // 2. 同一性：d(x, y) = 0 ⟺ x = y
    // 3. 对称性：d(x, y) = d(y, x)
    // 4. 三角不等式：d(x, z) ≤ d(x, y) + d(y, z)

    bool verify_metric_properties(
            const float* x,
            const float* y,
            const float* z,
            size_t d,
            float tolerance = 1e-6f) {

        float d_xy = compute_L2(x, y, d);
        float d_yz = compute_L2(y, z, d);
        float d_xz = compute_L2(x, z, d);

        // 验证三角不等式
        if (d_xz > d_xy + d_yz + tolerance) {
            printf("Triangle inequality violated!\n");
            return false;
        }

        return true;
    }

    float compute_L2(const float* x, const float* y, size_t d) {
        float sum = 0;
        for (size_t i = 0; i < d; i++) {
            float diff = x[i] - y[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
};
```

### 1.2 Faiss支持的距离度量

```cpp
// Faiss中的度量类型
// faiss/MetricType.h

enum MetricType {
    METRIC_L2 = 0,              // 欧几里得距离平方（默认）
    METRIC_INNER_PRODUCT = 1,   // 内积
    METRIC_L1 = 2,               // 曼哈顿距离
    METRIC_Linf = 3,             // 切比雪夫距离
    METRIC_Lp = 4,               // Lp距离
    METRIC_CANBERRA = 5,         // Canberra距离
    METRIC_BRAYCURTIS = 6,       // Bray-Curtis距离
    METRIC_JENSEN_SHANNON = 7,   // Jensen-Shannon散度
};

// 创建不同度量的索引
void create_index_with_metric(MetricType metric) {
    int d = 128;

    switch (metric) {
        case METRIC_L2:
            // 默认L2距离
            return new faiss::IndexFlatL2(d);

        case METRIC_INNER_PRODUCT:
            // 内积（用于余弦相似度）
            return new faiss::IndexFlatIP(d);

        case METRIC_L1:
            // L1距离
            return new faiss::IndexFlat(d, metric);

        default:
            printf("Unsupported metric type\n");
            return nullptr;
    }
}
```

---

## 2. L2距离家族

### 2.1 欧几里得距离

```cpp
// 标准L2距离
struct L2Distance {
    static float compute(const float* x, const float* y, size_t d) {
        float sum = 0;
        for (size_t i = 0; i < d; i++) {
            float diff = x[i] - y[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // Faiss使用L2平方（避免开方）
    static float compute_squared(const float* x, const float* y, size_t d) {
        float sum = 0;
        for (size_t i = 0; i < d; i++) {
            float diff = x[i] - y[i];
            sum += diff * diff;
        }
        return sum;
    }

    // SIMD优化版本
    #ifdef __AVX2__
    static float compute_simd(const float* x, const float* y, size_t d) {
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;

        for (; i + 8 <= d; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = _mm256_loadu_ps(y + i);

            __m256 diff = _mm256_sub_ps(vx, vy);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        // 处理剩余
        for (; i < d; i++) {
            float diff = x[i] - y[i];
            sum += diff * diff;
        }

        return _mm256_reduce_add_ps(sum);
    }
    #endif
};
```

### 2.2 加权L2距离

```cpp
// 加权L2距离（某些维度更重要）
struct WeightedL2 {
    static float compute(
            const float* x,
            const float* y,
            const float* weights,
            size_t d) {

        float sum = 0;
        for (size_t i = 0; i < d; i++) {
            float diff = x[i] - y[i];
            sum += weights[i] * diff * diff;
        }
        return std::sqrt(sum);
    }

    // 马氏距离（加权L1的变体）
    static float compute_mahalanobis(
            const float* x,
            const float* y,
            const float* covariance_inverse,
            size_t d) {

        // 马氏距离考虑特征相关性
        // d² = (x-y)^T × Σ^(-1) × (x-y)

        std::vector<float> diff(d);
        for (size_t i = 0; i < d; i++) {
            diff[i] = x[i] - y[i];
        }

        float sum = 0;
        for (size_t i = 0; i < d; i++) {
            for (size_t j = 0; j < d; j++) {
                sum += diff[i] * covariance_inverse[i * d + j] * diff[j];
            }
        }

        return std::sqrt(sum);
    }
};
```

### 2.3 Lp距离

```cpp
// Lp距离：Lp距离的推广
struct LpDistance {
    static float compute(
            const float* x,
            const float* y,
            size_t d,
            float p = 2.0f) {

        float sum = 0;
        for (size_t i = 0; i < d; i++) {
            float diff = std::abs(x[i] - y[i]);
            sum += std::pow(diff, p);
        }
        return std::pow(sum, 1.0f / p);
    }

    // L1距离（曼哈顿距离）
    static float compute_L1(const float* x, const float* y, size_t d) {
        float sum = 0;
        for (size_t i = 0; i < d; i++) {
            sum += std::abs(x[i] - y[i]);
        }
        return sum;
    }

    // L∞距离（切比雪夫距离）
    static float compute_Linf(const float* x, const float* y, size_t d) {
        float max_diff = 0;
        for (size_t i = 0; i < d; i++) {
            max_diff = std::max(max_diff, std::abs(x[i] - y[i]));
        }
        return max_diff;
    }
};
```

---

## 3. 内积与余弦相似度

### 3.1 内积距离

```cpp
// 内积（点积）
struct InnerProductDistance {
    // 标准内积
    static float compute(
            const float* x,
            const float* y,
            size_t d) {

        float sum = 0;
        for (size_t i = 0; i < d; i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    // 负内积（用于最大化转为最小化）
    static float compute_negative(const float* x, const float* y, size_t d) {
        return -compute(x, y, d);
    }

    // SIMD优化版本
    #ifdef __AVX2__
    static float compute_simd(const float* x, const float* y, size_t d) {
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;

        for (; i + 8 <= d; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = _mm256_loadu_ps(y + i);

            sum = _mm256_add_ps(sum, _mm256_mul_ps(vx, vy));
        }

        // 处理剩余
        for (; i < d; i++) {
            sum += x[i] * y[i];
        }

        return _mm256_reduce_add_ps(sum);
    }
    #endif
};
```

### 3.2 余弦相似度

```cpp
// 余弦相似度
struct CosineSimilarity {
    // 归一化后的向量的内积
    static float compute(
            const float* x,
            const float* y,
            size_t d) {

        float norm_x = 0, norm_y = 0, dot = 0;

        for (size_t i = 0; i < d; i++) {
            norm_x += x[i] * x[i];
            norm_y += y[i] * y[i];
            dot += x[i] * y[i];
        }

        return dot / (std::sqrt(norm_x) * std::sqrt(norm_y));
    }

    // L2距离转余弦距离
    static float from_L2(
            const float* x,
            const float* y,
            float x_norm,
            float y_norm,
            const float* L2_dis_table,
            size_t d) {

        // cos(θ) = 1 - d²(x,y) / (2 × ||x|| × ||y||)
        //        = 1 - L2_dis² / (2 × x_norm × y_norm)

        float L2_dis_squared = 0;
        for (size_t i = 0; i < d; i++) {
            float diff = x[i] - y[i];
            L2_dis_squared += diff * diff;
        }

        float cosine_sim = 1.0f - L2_dis_squared / (2.0f * x_norm * y_norm);
        return cosine_sim;
    }
};
```

### 3.3 归一化向量

```cpp
// 向量归一化（用于余弦相似度）
struct VectorNormalization {
    // L2归一化
    static void normalize_L2(const float* x, float* norm, size_t d) {
        float norm_sq = 0;
        for (size_t i = 0; i < d; i++) {
            norm_sq += x[i] * x[i];
        }

        *norm = std::sqrt(norm_sq);
    }

    // 批量归一化
    static void normalize_batch(
            const float* X,
            float* X_normalized,
            float* norms,
            size_t n,
            size_t d) {

        for (size_t i = 0; i < n; i++) {
            const float* x = X + i * d;
            float* x_norm = X_normalized + i * d;

            float norm = 0;
            for (size_t j = 0; j < d; j++) {
                norm += x[j] * x[j];
            }
            norm = std::sqrt(norm);

            // 归一化
            float inv_norm = 1.0f / norm;
            for (size_t j = 0; j < d; j++) {
                x_norm[j] = x[j] * inv_norm;
            }

            norms[i] = norm;
        }
    }
};
```

---

## 4. 距离度量学习

### 4.1 度量学习概述

```cpp
// 距离度量学习：学习最优的距离度量
struct MetricLearning {

    // 目标：学习一个距离度量D(x,y)
    // 使得相似样本对距离小，不相似样本对距离大

    struct SamplePair {
        const float* x;
        const float* y;
        int label;  // 1=相似, 0=不相似
    };

    // Contrastive Loss
    float contrastive_loss(
            const SamplePair& pair,
            const float* x,
            const float* y,
            size_t d,
            float margin) {

        float d_pos = fvec_L2sqr(x, pair.x, d);      // 正样本对距离
        float d_neg = fvec_L2sqr(x, pair.y, d);      // 负样本对距离

        // Contrastive loss
        // L = max(0, d_pos - d_neg + margin)

        float loss = std::max(0.0f, d_pos - d_neg + margin);

        return loss;
    }

    // Triplet Loss
    struct Triplet {
        const float* anchor;
        const float* positive;
        const float* negative;
    };

    float triplet_loss(
            const Triplet& triplet,
            size_t d,
            float margin) {

        float d_ap = fvec_L2sqr(triplet.anchor, triplet.positive, d);
        float d_an = fvec_L2sqr(triplet.anchor, triplet.negative, d);

        // Triplet loss
        // L = max(0, d_ap - d_an + margin)

        return std::max(0.0f, d_ap - d_an + margin);
    }
};
```

### 4.2 马氏距离学习

```cpp
// 学习马氏距离的协方差矩阵
struct MahalanobisLearner {
    int d;

    // 计算协方差矩阵
    void compute_covariance(
            const float* X,
            size_t n,
            float* covariance) {

        // 1. 中心化数据
        std::vector<float> mean(d, 0.0f);

        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                mean[j] += X[i * d + j];
            }
        }

        for (size_t j = 0; j < d; j++) {
            mean[j] /= n;
        }

        // 2. 计算协方差
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                for (size_t k = 0; k < d; k++) {
                    covariance[j * d + k] +=
                        (X[i * d + j] - mean[j]) *
                        (X[i * d + k] - mean[k]);
                }
            }
        }

        // 3. 归一化
        for (size_t i = 0; i < d * d; i++) {
            covariance[i] /= n;
        }
    }

    // 计算协方差逆（加入正则化）
    bool invert_covariance(
            const float* covariance,
            float* inv_covariance,
            float reg = 1e-6f) {

        // 1. 添加正则化项
        std::vector<float> cov_reg(d * d);

        for (size_t i = 0; i < d; i++) {
            for (size_t j = 0; j < d; j++) {
                cov_reg[i * d + j] = covariance[i * d + j];
                cov_reg[i * d + i] += reg;  // 对角正则化
            }
        }

        // 2. Cholesky分解求逆
        // Σ^(-1) = (LL^T)^(-1) = L^(-T) × L^(-1)

        std::vector<float> L(d * d);

        // 使用Eigen或手动实现Cholesky
        if (!cholesky_decomposition(cov_reg.data(), L.data(), d)) {
            printf("Cholesky decomposition failed\n");
            return false;
        }

        // 3. 求逆
        // Σ^(-1) = (L^(-1))^T × L^(-1)

        // 先求L的逆（下三角）
        std::vector<float> L_inv(d * d, 0);
        for (int i = 0; i < d; i++) {
            L_inv[i * d + i] = 1.0f / L[i * d + i];
            for (int j = i + 1; j < d; j++) {
                // L_inv[i][j] = 0 (下三角)
            }
        }

        // L^(-T)
        std::vector<float> inv_covariance(d * d);
        for (size_t i = 0; i < d; i++) {
            for (size_t j = 0; j <= i; j++) {
                float sum = 0;
                for (size_t k = j; k <= i; k++) {
                    sum += L_inv[k * d + i] * L_inv[k * d + j];
                }
                inv_covariance[i * d + j] = sum;
            }
        }

        return true;
    }
};
```

### 4.3 深度度量学习

```cpp
// 深度神经网络学习距离度量
struct DeepMetricLearning {

    // Siamese网络架构
    struct SiameseNetwork {
        int input_dim;
        int embedding_dim;

        // 特征提取器
        std::shared_ptr<NeuralNetwork> feature_extractor;

        SiameseNetwork(int in_dim, int emb_dim)
            : input_dim(in_dim), embedding_dim(emb_dim) {

            // 构建网络
            feature_extractor = build_feature_extractor();
        }

        // 前向传播
        std::vector<float> forward(const float* x) {
            return feature_extractor->forward(x);
        }

        // 计算对比损失
        float contrastive_loss(
                const float* x1,
                const float* x2,
                int label) {

            auto emb1 = forward(x1);
            auto emb2 = forward(x2);

            // 欧几里得距离
            float dis = euclidean_distance(emb1, emb2);

            // Contrastive loss
            float margin = 1.0f;
            float loss = label * dis + (1 - label) * std::max(0.0f, margin - dis);

            return loss;
        }

    private:
        std::shared_ptr<NeuralNetwork> build_feature_extractor() {
            // 示例：3层MLP
            // 实际应用中可能使用ResNet等
            return nullptr;
        }

        float euclidean_distance(
                const std::vector<float>& v1,
                const std::vector<float>& v2) {

            float sum = 0;
            for (size_t i = 0; i < v1.size(); i++) {
                float diff = v1[i] - v2[i];
                sum += diff * diff;
            }
            return std::sqrt(sum);
        }
    };

    // 训练
    void train(
            const std::vector<Triplet>& triplets,
            int n_epochs,
            float learning_rate) {

        SiameseNetwork network(128, 64);

        for (int epoch = 0; epoch < n_epochs; epoch++) {
            float total_loss = 0;

            for (const auto& triplet : triplets) {
                // 前向传播
                auto anchor_emb = network.forward(triplet.anchor);
                auto positive_emb = network.forward(triplet.positive);
                auto negative_emb = network.forward(triplet.negative);

                // 计算triplet loss
                float d_ap = euclidean_distance(anchor_emb, positive_emb);
                float d_an = euclidean_distance(anchor_emb, negative_emb);

                float margin = 1.0f;
                float loss = std::max(0.0f, d_ap - d_an + margin);

                total_loss += loss;

                // 反向传播（省略）
                // ...
            }

            printf("Epoch %d: Loss = %.4f\n", epoch, total_loss / triplets.size());
        }
    }
};
```

---

## 5. 自适应距离度量

### 5.1 数据依赖的权重

```cpp
// 学习每个维度的重要性
struct AdaptiveWeights {

    // 基于方差的特征权重
    void compute_variance_weights(
            const float* X,
            size_t n,
            int d,
            float* weights) {

        // 1. 计算每个维度的方差
        std::vector<float> means(d, 0.0f);
        std::vector<float> variances(d, 0.0f);

        for (size_t i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                means[j] += X[i * d + j];
            }
        }

        for (int j = 0; j < d; j++) {
            means[j] /= n;
        }

        for (size_t i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                float diff = X[i * d + j] - means[j];
                variances[j] += diff * diff;
            }
        }

        for (int j = 0; j < d; j++) {
            variances[j] /= n;
        }

        // 2. 权重与方差成正比
        float sum_variance = 0;
        for (int j = 0; j < d; j++) {
            sum_variance += variances[j];
        }

        for (int j = 0; j < d; j++) {
            weights[j] = variances[j] / sum_variance;
        }

        // 3. 归一化
        float sum_weights = 0;
        for (int j = 0; j < d; j++) {
            sum_weights += weights[j];
        }

        for (int j = 0; j < d; j++) {
            weights[j] /= sum_weights;
        }
    }

    // 使用自适应权重的距离
    float weighted_distance(
            const float* x,
            const float* y,
            const float* weights,
            int d) {

        float sum = 0;
        for (int i = 0; i < d; i++) {
            float diff = x[i] - y[i];
            sum += weights[i] * diff * diff;
        }

        return std::sqrt(sum);
    }
};
```

### 5.2 Query-Dependent距离

```cpp
// 查询依赖的距离度量
struct QueryDependentDistance {

    // QD-ML（Quadruplet-based Deep Metric Learning）
    float qdml_distance(
            const float* query,
            const float* x,
            const float* y,
            size_t d) {

        // QD-ML使用4个投影矩阵：
        // 一个用于query，一个用于正样本，一个用于负样本，一个用于负样本对

        // 这里简化为使用query特定的权重
        std::vector<float> weights = learn_query_weights(query, d);

        return weighted_distance(query, x, weights, d);
    }

    // 学习查询权重
    std::vector<float> learn_query_weights(
            const float* query,
            size_t d,
            const float* relevant_items,
            size_t n_relevant) {

        // 简化：基于query的特征计算权重
        std::vector<float> weights(d);

        for (size_t i = 0; i < d; i++) {
            // 权重与query维度的重要性成正比
            // 这里使用query[i]作为示例
            weights[i] = std::abs(query[i]);
        }

        // 归一化
        float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
        for (size_t i = 0; i < d; i++) {
            weights[i] /= sum;
        }

        return weights;
    }
};
```

---

## 6. 角度距离

### 6.1 角度相似度

```cpp
// 角度距离
struct AngularDistance {

    // 余弦距离 = 1 - 余弦相似度
    static float cosine_distance(
            const float* x,
            const float* y,
            size_t d) {

        float norm_x = 0, norm_y = 0, dot = 0;

        for (size_t i = 0; i < d; i++) {
            norm_x += x[i] * x[i];
            norm_y += y[i] * y[i];
            dot += x[i] * y[i];
        }

        float cosine_sim = dot / (std::sqrt(norm_x) * std::sqrt(norm_y));
        return 1.0f - cosine_sim;
    }

    // 使用内积索引实现余弦距离
    static void cosine_search_with_IP_index(
            const float* queries,
            size_t nq,
            const float* database,
            size_t nb,
            int d,
            int k,
            float* distances,
            idx_t* labels) {

        // 1. 归一化所有向量
        std::vector<float> queries_norm(nq * d);
        std::vector<float> database_norm(nb * d);
        std::vector<float> query_norms(nq);
        std::vector<float> database_norms(nb);

        VectorNormalization::normalize_batch(
            queries, queries_norm.data(), query_norms.data(), nq, d);
        VectorNormalization::normalize_batch(
            database, database_norm.data(), database_norms.data(), nb, d);

        // 2. 使用内积索引搜索
        faiss::IndexFlatIP index(d);
        index.add(nb, database_norm.data());

        index.search(nq, queries_norm.data(), k, distances, labels);

        // 3. 转换为余弦距离
        for (size_t i = 0; i < nq * k; i++) {
            // cosine_distance = 1 - inner_product
            distances[i] = 1.0f - distances[i];
        }
    }
};
```

---

## 7. SIMD优化的距离计算深度实现

### 7.1 AVX-512批量L2距离矩阵计算

```cpp
// 批量L2距离矩阵计算（查询 x 数据库）
namespace simd_distance_matrix {

#if defined(__AVX512F__)

// 计算nq个查询与nb个向量的L2距离平方
// 结果矩阵: distances[nq x nb]
void compute_L2sqr_matrix_avx512(
        const float* queries,   // [nq x d]
        const float* database,  // [nb x d]
        float* distances,       // [nq x nb]
        size_t nq,
        size_t nb,
        size_t d) {

    // 处理每个查询
#pragma omp parallel for schedule(dynamic)
    for (size_t q = 0; q < nq; q++) {
        const float* query = queries + q * d;
        float* dist_row = distances + q * nb;

        size_t b = 0;

        // 处理16的倍数个数据库向量
        for (; b + 16 <= nb; b += 16) {
            // 一次计算16个距离
            __m512 sum[4];  // 4个累加器用于循环展开

            for (int acc = 0; acc < 4; acc++) {
                sum[acc] = _mm512_setzero_ps();
            }

            size_t i = 0;

            // 主循环：每次处理16维
            for (; i + 16 <= d; i += 16) {
                // 广播查询的16个分量到4个ZMM寄存器
                __m512 q0 = _mm512_loadu_ps(query + i);
                __m512 q1 = _mm512_loadu_ps(query + i + 4);
                __m512 q2 = _mm512_loadu_ps(query + i + 8);
                __m512 q3 = _mm512_loadu_ps(query + i + 12);

                // 对16个数据库向量
                for (int vec = 0; vec < 16; vec++) {
                    const float* db_vec = database + (b + vec) * d + i;

                    __m512 db = _mm512_loadu_ps(db_vec);

                    // 计算差值平方
                    __m512 diff = _mm512_sub_ps(q0, db);
                    sum[0] = _mm512_fmadd_ps(diff, diff, sum[0]);
                }
            }

            // 处理剩余维度
            for (; i < d; i++) {
                __m512 qv = _mm512_set1_ps(query[i]);

                for (int vec = 0; vec < 16; vec++) {
                    float db_val = database[(b + vec) * d + i];
                    __m512 db = _mm512_set1_ps(db_val);

                    __m512 diff = _mm512_sub_ps(qv, db);
                    sum[vec % 4] = _mm512_fmadd_ps(diff, diff, sum[vec % 4]);
                }
            }

            // 水平求和并存储
            for (int vec = 0; vec < 16; vec++) {
                __m512 total = _mm512_add_ps(sum[0], sum[2]);
                total = _mm512_add_ps(total, sum[1]);
                total = _mm512_add_ps(total, sum[3]);

                dist_row[b + vec] = _mm512_reduce_add_ps(total);
            }
        }

        // 处理剩余的数据库向量
        for (; b < nb; b++) {
            float dist = 0;
            for (size_t i = 0; i < d; i++) {
                float diff = query[i] - database[b * d + i];
                dist += diff * diff;
            }
            dist_row[b] = dist;
        }
    }
}

#endif // __AVX512F__
}
```

### 7.2 内积矩阵SIMD优化

```cpp
// 批量内积矩阵计算
namespace simd_inner_product {

#if defined(__AVX2__)

// 计算内积矩阵：IP[nq x nb]
void compute_inner_product_matrix_avx2(
        const float* queries,   // [nq x d], 归一化
        const float* database,  // [nb x d], 归一化
        float* ip_matrix,      // [nq x nb]
        size_t nq,
        size_t nb,
        size_t d) {

#pragma omp parallel for schedule(dynamic)
    for (size_t q = 0; q < nq; q++) {
        const float* query = queries + q * d;
        float* ip_row = ip_matrix + q * nb;

        size_t b = 0;

        // 每次处理8个数据库向量
        for (; b + 8 <= nb; b += 8) {
            __m256 sum[8];  // 8个累加器

            // 初始化
            for (int j = 0; j < 8; j++) {
                sum[j] = _mm256_setzero_ps();
            }

            size_t i = 0;

            // 主循环
            for (; i + 8 <= d; i += 8) {
                __m256 qv = _mm256_loadu_ps(query + i);

                // 8个数据库向量
                for (int j = 0; j < 8; j++) {
                    __m256 db = _mm256_loadu_ps(database + (b + j) * d + i);
                    sum[j] = _mm256_fmadd_ps(qv, db, sum[j]);
                }
            }

            // 处理剩余维度
            for (; i < d; i++) {
                __m256 qv = _mm256_set1_ps(query[i]);

                for (int j = 0; j < 8; j++) {
                    __m256 db = _mm256_set1_ps(database[(b + j) * d + i]);
                    sum[j] = _mm256_fmadd_ps(qv, db, sum[j]);
                }
            }

            // 水平求和并存储
            for (int j = 0; j < 8; j++) {
                ip_row[b + j] = _mm256_reduce_add_ps(sum[j]);
            }
        }

        // 剩余向量
        for (; b < nb; b++) {
            float ip = 0;
            for (size_t i = 0; i < d; i++) {
                ip += query[i] * database[b * d + i];
            }
            ip_row[b] = ip;
        }
    }
}

#endif // __AVX2__

#if defined(__aarch64__) && defined(__ARM_NEON)

// ARM NEON批量内积计算
void compute_inner_product_matrix_neon(
        const float* queries,
        const float* database,
        float* ip_matrix,
        size_t nq,
        size_t nb,
        size_t d) {

    for (size_t q = 0; q < nq; q++) {
        const float* query = queries + q * d;
        float* ip_row = ip_matrix + q * nb;

        size_t b = 0;

        // 每次处理4个数据库向量
        for (; b + 4 <= nb; b += 4) {
            float32x4_t sum[4] = {
                vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f)
            };

            size_t i = 0;

            // 4个一组处理
            for (; i + 4 <= d; i += 4) {
                float32x4_t qv = vld1q_f32(query + i);

                for (int j = 0; j < 4; j++) {
                    float32x4_t db = vld1q_f32(database + (b + j) * d + i);
                    sum[j] = vfmaq_f32(sum[j], qv, db);
                }
            }

            // 处理剩余维度
            for (; i < d; i++) {
                float32x4_t qv = vdupq_n_f32(query[i]);

                for (int j = 0; j < 4; j++) {
                    float32x4_t db = vdupq_n_f32(database[(b + j) * d + i]);
                    sum[j] = vfmaq_f32(sum[j], qv, db);
                }
            }

            // 水平求和
            for (int j = 0; j < 4; j++) {
                ip_row[b + j] = vaddvq_f32(sum[j]);
            }
        }

        // 剩余向量
        for (; b < nb; b++) {
            float ip = 0;
            for (size_t i = 0; i < d; i++) {
                ip += query[i] * database[b * d + i];
            }
            ip_row[b] = ip;
        }
    }
}

#endif // __ARM_NEON__
}
```

### 7.3 L1距离SIMD优化

```cpp
// L1距离（曼哈顿距离）SIMD优化
#if defined(__AVX2__)

inline float l1_distance_avx2(const float* x, const float* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= d; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 yv = _mm256_loadu_ps(y + i);

        // |x - y|
        __m256 diff = _mm256_sub_ps(xv, yv);

        // 绝对值：清除符号位
        __m256i diff_i = _mm256_castps_si256(diff);
        __m256i abs_mask = _mm256_set1_epi32(0x7FFFFFFF);
        __m256i abs_i = _mm256_and_si256(diff_i, abs_mask);
        __m256 abs = _mm256_castsi256_ps(abs_i);

        sum = _mm256_add_ps(sum, abs);
    }

    float result = _mm256_reduce_add_ps(sum);

    // 处理剩余
    for (; i < d; i++) {
        result += std::abs(x[i] - y[i]);
    }

    return result;
}

// AVX-512优化的L1距离
#if defined(__AVX512F__)

inline float l1_distance_avx512(const float* x, const float* y, size_t d) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= d; i += 16) {
        __m512 xv = _mm512_loadu_ps(x + i);
        __m512 yv = _mm512_loadu_ps(y + i);

        __m512 diff = _mm512_sub_ps(xv, yv);

        // 绝对值
        __m512 abs = _mm512_abs_ps(diff);

        sum = _mm512_add_ps(sum, abs);
    }

    float result = _mm512_reduce_add_ps(sum);

    for (; i < d; i++) {
        result += std::abs(x[i] - y[i]);
    }

    return result;
}

#endif // __AVX512F__

#endif // __AVX2__
```

---

## 8. 损失函数优化实现

### 8.1 Triplet Loss批量计算与梯度

```cpp
// 批量Triplet Loss计算和梯度
namespace loss_optimization {

// Triplet Loss结构
struct TripletBatch {
    std::vector<float> anchors;   // [batch_size x d]
    std::vector<float> positives; // [batch_size x d]
    std::vector<float> negatives; // [batch_size x d]
    size_t batch_size;
    size_t d;
};

struct TripletLossOutput {
    float loss;                   // 总损失
    std::vector<float> grad_anchors;   // 锚点梯度 [batch_size x d]
    std::vector<float> grad_positives; // 正样本梯度
    std::vector<float> grad_negatives; // 负样本梯度
};

// SIMD优化的Triplet Loss计算
TripletLossOutput triplet_loss_forward(
        const TripletBatch& batch,
        float margin = 1.0f) {

    const size_t n = batch.batch_size;
    const size_t d = batch.d;

    TripletLossOutput output;
    output.grad_anchors.resize(n * d, 0);
    output.grad_positives.resize(n * d, 0);
    output.grad_negatives.resize(n * d, 0);

    float total_loss = 0;
    size_t active_triplets = 0;

    // 计算每个triplet的损失和梯度
    for (size_t i = 0; i < n; i++) {
        const float* anchor = batch.anchors.data() + i * d;
        const float* positive = batch.positives.data() + i * d;
        const float* negative = batch.negatives.data() + i * d;

        // 计算距离平方
#if defined(__AVX2__)
        float d_ap = fvec_L2sqr_simd(anchor, positive, d);
        float d_an = fvec_L2sqr_simd(anchor, negative, d);
#else
        float d_ap = fvec_L2sqr_ref(anchor, positive, d);
        float d_an = fvec_L2sqr_ref(anchor, negative, d);
#endif

        // Triplet loss: L = max(0, d_ap - d_an + margin)
        float loss = d_ap - d_an + margin;

        if (loss > 0) {
            total_loss += loss;
            active_triplets++;

            // 梯度计算
            // ∂L/∂anchor = 2 * (anchor - positive) - 2 * (anchor - negative)
            // ∂L/∂positive = -2 * (anchor - positive)
            // ∂L/∂negative = 2 * (anchor - negative)

            float* grad_a = output.grad_anchors.data() + i * d;
            float* grad_p = output.grad_positives.data() + i * d;
            float* grad_n = output.grad_negatives.data() + i * d;

            for (size_t j = 0; j < d; j++) {
                float da = 2.0f * (anchor[j] - positive[j]);
                float dn = 2.0f * (anchor[j] - negative[j]);

                grad_a[j] += da - dn;
                grad_p[j] -= da;
                grad_n[j] += dn;
            }
        }
    }

    output.loss = total_loss;
    return output;
}
}
```

### 8.2 Contrastive Loss优化

```cpp
// Contrastive Loss批量实现
struct ContrastiveBatch {
    std::vector<float> x1;  // [batch_size x d]
    std::vector<float> x2;  // [batch_size x d]
    std::vector<int> labels;  // [batch_size], 0或1
    size_t batch_size;
    size_t d;
};

struct ContrastiveLossOutput {
    float loss;
    std::vector<float> grad_x1;
    std::vector<float> grad_x2;
};

ContrastiveLossOutput contrastive_loss_forward(
        const ContrastiveBatch& batch,
        float margin = 1.0f) {

    const size_t n = batch.batch_size;
    const size_t d = batch.d;

    ContrastiveLossOutput output;
    output.grad_x1.resize(n * d, 0);
    output.grad_x2.resize(n * d, 0);

    float total_loss = 0;

    for (size_t i = 0; i < n; i++) {
        const float* v1 = batch.x1.data() + i * d;
        const float* v2 = batch.x2.data() + i * d;
        int label = batch.labels[i];

        // 计算欧几里得距离
        float dist_sq = 0;
        for (size_t j = 0; j < d; j++) {
            float diff = v1[j] - v2[j];
            dist_sq += diff * diff;
        }
        float dist = std::sqrt(dist_sq);

        // Contrastive loss
        float loss;
        if (label == 1) {
            // 相同类：最小化距离
            loss = dist_sq;
        } else {
            // 不同类：最大化距离，但至少要大于margin
            loss = std::max(0.0f, margin - dist);
        }

        total_loss += loss;

        // 梯度
        float* grad1 = output.grad_x1.data() + i * d;
        float* grad2 = output.grad_x2.data() + i * d;

        float scale = (label == 1) ? 2.0f : -2.0f * loss;

        for (size_t j = 0; j < d; j++) {
            float diff = v1[j] - v2[j];
            grad1[j] = scale * diff;
            grad2[j] = -scale * diff;
        }
    }

    output.loss = total_loss / n;
    return output;
}
```

### 8.3 InfoNCE损失（对比学习）

```cpp
// InfoNCE损失（用于SimCLR等对比学习）
struct InfoNCELoss {

    // InfoNCE = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ)
    static float compute(
            const float* embeddings,  // [batch_size * 2 * embedding_dim]
            size_t batch_size,
            size_t embedding_dim,
            float temperature = 0.07f) {

        // embeddings包含正负对：
        // [z_1^+, z_1^-, z_2^+, z_2^-, ...]
        // 其中z_i^+和z_i^-是正对

        float total_loss = 0;

        for (size_t i = 0; i < batch_size; i++) {
            const float* z_i = embeddings + i * 2 * embedding_dim;
            const float* z_j = embeddings + i * 2 * embedding_dim + embedding_dim;

            // 正对相似度
            float sim_pos = cosine_similarity(z_i, z_j, embedding_dim);

            // 计算所有负对的相似度
            std::vector<float> neg_simils;
            neg_simils.reserve(2 * batch_size - 1);

            for (size_t k = 0; k < batch_size; k++) {
                if (k != i) {
                    const float* z_k = embeddings + k * 2 * embedding_dim;
                    neg_simils.push_back(cosine_similarity(z_i, z_k, embedding_dim));
                    neg_simils.push_back(cosine_similarity(z_j, z_k, embedding_dim));
                }
            }

            // 归一化
            sim_pos /= temperature;
            for (auto& sim : neg_simils) {
                sim /= temperature;
            }

            // 计算log-sum-exp
            float max_sim = std::max(sim_pos, *std::max_element(neg_simils.begin(), neg_simils.end()));
            float log_sum_exp = std::exp(sim_pos - max_sim);

            for (float sim : neg_simils) {
                log_sum_exp += std::exp(sim - max_sim);
            }

            log_sum_exp = std::log(log_sum_exp) + max_sim;

            total_loss += -sim_pos + log_sum_exp;
        }

        return total_loss / batch_size;
    }

    static float cosine_similarity(const float* x, const float* y, size_t d) {
        float dot = 0, norm_x = 0, norm_y = 0;
        for (size_t i = 0; i < d; i++) {
            dot += x[i] * y[i];
            norm_x += x[i] * x[i];
            norm_y += y[i] * y[i];
        }
        return dot / (std::sqrt(norm_x) * std::sqrt(norm_y));
    }
};
```

---

## 9. 矩阵操作优化

### 9.1 矩阵乘法SIMD优化

```cpp
// 小型矩阵乘法优化（用于度量学习中的变换）
namespace small_matrix {

// 4x4矩阵乘法：C = A * B
void matmul_4x4(const float* A, const float* B, float* C) {
#if defined(__AVX2__)
    // 转置B以提高缓存局部性
    alignas(32) float B_T[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            B_T[i * 4 + j] = B[j * 4 + i];
        }
    }

    for (int i = 0; i < 4; i++) {
        __m256 crow = _mm256_setzero_ps();

        for (int k = 0; k < 4; k++) {
            __m256 a_val = _mm256_set1_ps(A[i * 4 + k]);
            __m256 b_row = _mm256_loadu_ps(B_T + k * 4);

            crow = _mm256_fmadd_ps(a_val, b_row, crow);
        }

        _mm256_storeu_ps(C + i * 4, crow);
    }
#else
    // 标量实现
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            C[i * 4 + j] = 0;
            for (int k = 0; k < 4; k++) {
                C[i * 4 + j] += A[i * 4 + k] * B[k * 4 + j];
            }
        }
    }
#endif
}

// 向量-矩阵乘法：y = M * x
// 用于马氏距离中的变换
void vec_mat_mul(const float* M, const float* x, float* y, int d) {
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= d; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 mv = _mm256_loadu_ps(M + i);  // 假设M按列存储

        sum = _mm256_fmadd_ps(xv, mv, sum);
    }

    float result = _mm256_reduce_add_ps(sum);

    for (; i < d; i++) {
        result += M[i] * x[i];
    }

    *y = result;
#else
    float sum = 0;
    for (int i = 0; i < d; i++) {
        sum += M[i] * x[i];
    }
    *y = sum;
#endif
}
}
```

### 9.2 协方差计算优化

```cpp
// SIMD优化的协方差计算
namespace covariance_optimized {

// 计算协方差矩阵：Cov[X] = E[(X-μ)(X-μ)^T]
void compute_covariance_simd(
        const float* X,      // [n x d]
        size_t n,
        size_t d,
        float* covariance,  // [d x d]
        float* mean) {       // [d]

    // 1. 计算均值
    std::vector<float> mean_accum(d, 0.0f);

#if defined(__AVX2__)
    for (size_t i = 0; i < n; i++) {
        const float* row = X + i * d;
        size_t j = 0;

        __m256 sum = _mm256_setzero_ps();
        for (; j + 8 <= d; j += 8) {
            __m256 xv = _mm256_loadu_ps(row + j);
            sum = _mm256_add_ps(sum, xv);
        }

        // 水平求和到mean_accum
        alignas(32) float tmp[8];
        _mm256_storeu_ps(tmp, sum);
        for (size_t k = 0; k < 8 && (j + k) < d; k++) {
            mean_accum[j + k] += tmp[k];
        }

        for (; j < d; j++) {
            mean_accum[j] += row[j];
        }
    }

    for (size_t j = 0; j < d; j++) {
        mean[j] = mean_accum[j] / n;
    }
#else
    // 标量版本
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            mean_accum[j] += X[i * d + j];
        }
    }
    for (size_t j = 0; j < d; j++) {
        mean[j] = mean_accum[j] / n;
    }
#endif

    // 2. 计算协方差（上三角）
    std::memset(covariance, 0, d * d * sizeof(float));

    for (size_t k = 0; k < n; k++) {
        const float* row = X + k * d;

        // 中心化数据
        std::vector<float> centered(d);
        for (size_t j = 0; j < d; j++) {
            centered[j] = row[j] - mean[j];
        }

        // 计算外积：centered^T * centered
        for (size_t i = 0; i < d; i++) {
            for (size_t j = i; j < d; j++) {
                covariance[i * d + j] += centered[i] * centered[j];
            }
        }
    }

    // 复制到下三角
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < i; j++) {
            covariance[i * d + j] = covariance[j * d + i];
        }
    }

    // 归一化
    float inv_n = 1.0f / n;
    for (size_t i = 0; i < d * d; i++) {
        covariance[i] *= inv_n;
    }
}
}
```

### 9.3 Cholesky分解优化

```cpp
// Cholesky分解：A = LL^T
namespace cholesky_optimized {

// 使用block-wised算法的Cholesky分解
bool cholesky_decomposition(
        const float* A,
        float* L,
        int d,
        int block_size = 32) {

    // 初始化L为单位矩阵
    std::memset(L, 0, d * d * sizeof(float));
    for (int i = 0; i < d; i++) {
        L[i * d + i] = 1.0f;
    }

    // 拷贝A到L（将原地修改）
    std::memcpy(L, A, d * d * sizeof(float));

    for (int j = 0; j < d; j++) {
        // 对角线元素
        float sum = 0;
        for (int k = 0; k < j; k++) {
            float val = L[j * d + k];
            sum += val * val;
        }

        float diag = L[j * d + j] - sum;

        if (diag <= 0) {
            return false;  // 不是正定矩阵
        }

        L[j * d + j] = std::sqrt(diag);

        // 第j列的其余元素
        for (int i = j + 1; i < d; i++) {
            sum = 0;
            for (int k = 0; k < j; k++) {
                sum += L[i * d + k] * L[j * d + k];
            }

            L[i * d + j] = (L[i * d + j] - sum) / L[j * d + j];
        }
    }

    return true;
}

// SIMD加速的内积计算（用于Cholesky）
float simd_dot_product(const float* x, const float* y, size_t d) {
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= d; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 yv = _mm256_loadu_ps(y + i);
        sum = _mm256_fmadd_ps(xv, yv, sum);
    }

    float result = _mm256_reduce_add_ps(sum);

    for (; i < d; i++) {
        result += x[i] * y[i];
    }

    return result;
#else
    float result = 0;
    for (size_t i = 0; i < d; i++) {
        result += x[i] * y[i];
    }
    return result;
#endif
}
}
```

---

## 10. 训练优化

### 10.1 Mini-batch处理

```cpp
// Mini-batch Triplet Loss训练优化
namespace training_optimization {

class TripletTrainer {
public:
    struct TrainingConfig {
        int batch_size = 256;
        float margin = 1.0f;
        float learning_rate = 0.001f;
        int n_epochs = 100;
    };

    void train(
            const float* X,           // [n x d] 训练数据
            size_t n,
            int d,
            const TrainingConfig& config) {

        // 1. 生成triplet
        std::vector<Triplet> triplets = generate_triplets(X, n, d, config.batch_size);

        // 2. 训练循环
        for (int epoch = 0; epoch < config.n_epochs; epoch++) {
            float epoch_loss = 0;

            // 打乱triplets
            std::shuffle(triplets.begin(), triplets.end(), rng);

            // Mini-batch处理
            for (size_t batch_start = 0; batch_start < triplets.size();
                    batch_start += config.batch_size) {

                size_t batch_end = std::min(batch_start + config.batch_size, triplets.size());

                // 前向传播
                TripletBatch batch;
                batch.anchors.resize(config.batch_size * d);
                batch.positives.resize(config.batch_size * d);
                batch.negatives.resize(config.batch_size * d);

                // 填充batch数据...
                for (size_t i = batch_start; i < batch_end; i++) {
                    size_t local_idx = i - batch_start;
                    const auto& trip = triplets[i];

                    std::memcpy(batch.anchors.data() + local_idx * d, trip.anchor, d * sizeof(float));
                    std::memcpy(batch.positives.data() + local_idx * d, trip.positive, d * sizeof(float));
                    std::memcpy(batch.negatives.data() + local_idx * d, trip.negative, d * sizeof(float));
                }

                batch.batch_size = batch_end - batch_start;
                batch.d = d;

                // 计算损失和梯度
                auto output = triplet_loss_forward(batch, config.margin);
                epoch_loss += output.loss;

                // 反向传播和参数更新（省略具体实现）
                // update_parameters(output.grad_anchors, ...);
            }

            printf("Epoch %d: Loss = %.4f\n", epoch, epoch_loss / triplets.size());
        }
    }

private:
    std::mt19937 rng;

    std::vector<Triplet> generate_triplets(
            const float* X, size_t n, int d,
            int batch_size) {
        // 硬负样本挖掘或随机采样
        std::vector<Triplet> triplets;
        // 实现省略...
        return triplets;
    }
};
}
```

### 10.2 梯度累积优化

```cpp
// 梯度累积（支持大batch size）
namespace gradient_accumulation {

class GradientAccumulator {
public:
    void accumulate_gradients(
            const float* grad,
            size_t size,
            size_t accumulation_steps) {

        // 检查是否需要更新
        if (accumulated_gradients.size() != size) {
            accumulated_gradients.resize(size, 0.0f);
        }

        // 累积梯度
        if (accumulated_count < accumulation_steps) {
            for (size_t i = 0; i < size; i++) {
                accumulated_gradients[i] += grad[i];
            }
            accumulated_count++;
        } else {
            // 重置并开始新的累积
            std::fill(accumulated_gradients.begin(), accumulated_gradients.end(), 0.0f);
            for (size_t i = 0; i < size; i++) {
                accumulated_gradients[i] += grad[i];
            }
            accumulated_count = 1;
        }
    }

    const float* get_accumulated() const {
        return accumulated_gradients.data();
    }

    bool is_ready() const {
        return accumulated_count >= accumulation_steps;
    }

    void reset() {
        std::fill(accumulated_gradients.begin(), accumulated_gradients.end(), 0.0f);
        accumulated_count = 0;
    }

private:
    std::vector<float> accumulated_gradients;
    size_t accumulated_count = 0;
    size_t accumulation_steps = 1;
};
}
```

---

## 11. 第20天总结

### 关键概念

1. **L2距离家族**：欧几里得、L1、L∞、Lp
2. **内积与余弦**：通过归一化实现转换
3. **度量学习**：Contrastive Loss、Triplet Loss、InfoNCE
4. **马氏距离**：考虑特征相关性
5. **自适应权重**：数据依赖和查询依赖
6. **SIMD优化**：批量距离矩阵计算、内积矩阵
7. **训练优化**：Mini-batch、梯度累积

### 底层优化技术

| 技术 | 实现方式 | 性能提升 |
|------|----------|----------|
| **批量距离计算** | AVX-512并行处理16个向量 | 8-16x |
| **内积矩阵** | 8路展开 + 多累加器 | 6-8x |
| **损失计算** | SIMD距离 + 向量化梯度 | 4-6x |
| **矩阵乘法** | 4x4专用kernel | 2-3x |
| **协方差计算** | SIMD均值 + 外积 | 3-5x |

### 实践技巧

1. **余弦搜索**：使用IndexFlatIP + 向量归一化
2. **度量学习**：提升特定任务的检索精度
3. **权重学习**：自动发现重要维度
4. **SIMD批处理**：一次计算多个距离
5. **梯度累积**：支持大batch size训练

### 下一步

第21天将学习**向量压缩与编码技术**，深入了解PQ、OPQ、标量量化等的压缩原理。

---

## 练习题

1. 实现Lp距离函数
2. 对比内积和余弦相似度
3. 实现简单的Contrastive Loss
4. 计算马氏距离
5. **实现AVX-512批量距离矩阵计算**
6. **优化Triplet Loss的梯度计算**
7. **实现SIMD优化的协方差计算**

## 12. Faiss距离计算底层实现详解

### 12.1 距离计算函数分发

```cpp
// faiss/utils/distances.cpp
// 距离计算的分发层

namespace faiss {

// 距离计算函数指针类型
typedef float (*fvec_func_t)(const float*, const float*, size_t);

// 根据度量类型选择对应的计算函数
inline fvec_func_t get_distance_func(MetricType metric) {
    switch (metric) {
        case METRIC_L2:
            return fvec_L2sqr;
        case METRIC_INNER_PRODUCT:
            return fvec_inner_product;
        case METRIC_L1:
            return fvec_L1;
        case METRIC_Linf:
            return fvec_Linf;
        case METRIC_Lp:
            return fvec_Lp;  // p需要额外参数
        default:
            return nullptr;
    }
}

// 批量距离计算(使用SIMD)
inline void compute_distance_matrix(
        const float* x,      // [n1 x d]
        const float* y,      // [n2 x d]
        size_t n1,
        size_t n2,
        size_t d,
        float* dis,         // [n1 x n2]
        MetricType metric) {

    // 根据metric类型调用不同的实现
    if (metric == METRIC_L2) {
        fvec_L2sqr_ny(x, y, d, n2, dis);
    } else if (metric == METRIC_INNER_PRODUCT) {
        fvec_inner_products_ny(x, y, d, n2, dis);
    } else {
        // 回退到标量实现
        for (size_t i = 0; i < n1; i++) {
            for (size_t j = 0; j < n2; j++) {
                dis[i * n2 + j] = get_distance_func(metric)(
                    x + i * d, y + j * d, d);
            }
        }
    }
}
}
```

### 12.2 L2距离底层实现

```cpp
// faiss/utils/distances.cpp
// 参考实现

// 标量版本
float fvec_L2sqr_ref(const float* x, const float* y, size_t d) {
    float result = 0;
    for (size_t i = 0; i < d; i++) {
        float tmp = x[i] - y[i];
        result += tmp * tmp;
    }
    return result;
}

// 使用SIMD的实现(编译时选择)
float fvec_L2sqr(const float* x, const float* y, size_t d) {
#ifdef __AVX2__
    // AVX2版本(一次处理8个float)
    return fvec_L2sqr_avx2(x, y, d);
#elif defined(__aarch64__)
    // ARM NEON版本
    return fvec_L2sqr_neon(x, y, d);
#else
    // 标量版本
    return fvec_L2sqr_ref(x, y, d);
#endif
}
```

### 12.3 内积计算底层实现

```cpp
// faiss/utils/distances.cpp

// 标量版本
float fvec_inner_product_ref(const float* x, const float* y, size_t d) {
    float result = 0;
    for (size_t i = 0; i < d; i++) {
        result += x[i] * y[i];
    }
    return result;
}

// SIMD版本(在distances_simd.cpp中实现)
#ifdef __AVX2__
float fvec_inner_product_avx2(const float* x, const float* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    // 主循环: 处理8的倍数
    for (; i + 8 <= d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(vx, vy));
    }

    // 处理剩余元素
    float result = _mm256_reduce_add_ps(sum);
    for (; i < d; i++) {
        result += x[i] * y[i];
    }

    return result;
}
#endif
```

### 12.4 批量距离计算

```cpp
// faiss/utils/distances.cpp

// 计算x与n个y向量的L2距离
// dis[i] = ||x - y[i*d]||^2
void fvec_L2sqr_ny(
        const float* x,
        const float* y,
        size_t d,
        size_t ny,
        float* dis) {

#ifdef __AVX2__
    // AVX2优化版本
    fvec_L2sqr_ny_avx2(x, y, d, ny, dis);
#elif defined(__aarch64__)
    // ARM NEON版本
    fvec_L2sqr_ny_neon(x, y, d, ny, dis);
#else
    // 标量版本
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr_ref(x, y + i * d, d);
    }
#endif
}

// 批量内积计算
void fvec_inner_products_ny(
        const float* x,
        const float* y,
        size_t d,
        size_t ny,
        float* ip) {

#ifdef __AVX2__
    fvec_inner_products_ny_avx2(x, y, d, ny, ip);
#else
    for (size_t i = 0; i < ny; i++) {
        ip[i] = fvec_inner_product_ref(x, y + i * d, d);
    }
#endif
}
```

### 12.5 L1距离实现

```cpp
// faiss/utils/extra_distances.cpp

// L1距离(曼哈顿距离)
float fvec_L1(const float* x, const float* y, size_t d) {
    float result = 0;
    for (size_t i = 0; i < d; i++) {
        result += std::abs(x[i] - y[i]);
    }
    return result;
}

// SIMD优化的L1距离
#ifdef __AVX2__
float fvec_L1_avx2(const float* x, const float* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);

        // |x - y|需要绝对值
        __m256 diff = _mm256_sub_ps(vx, vy);

        // AVX2没有直接的浮点绝对值指令
        // 使用mask清除符号位
        __m256 mask = _mm256_set1_ps(-0.0f);  // 符号位为1, 其他为0
        __m256 abs_diff = _mm256_andnot_ps(mask, diff);

        sum = _mm256_add_ps(sum, abs_diff);
    }

    float result = _mm256_reduce_add_ps(sum);
    for (; i < d; i++) {
        result += std::abs(x[i] - y[i]);
    }

    return result;
}
#endif
```

### 12.6 L∞距离实现

```cpp
// faiss/utils/extra_distances.cpp

// L∞距离(切比雪夫距离)
float fvec_Linf(const float* x, const float* y, size_t d) {
    float result = 0;
    for (size_t i = 0; i < d; i++) {
        float diff = std::abs(x[i] - y[i]);
        if (diff > result) {
            result = diff;
        }
    }
    return result;
}

// SIMD优化的L∞距离
#ifdef __AVX2__
float fvec_Linf_avx2(const float* x, const float* y, size_t d) {
    __m256 max_val = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);

        __m256 diff = _mm256_sub_ps(vx, vy);
        __m256 abs_diff = _mm256_andnot_ps(
            _mm256_set1_ps(-0.0f), diff);

        // 更新最大值
        max_val = _mm256_max_ps(max_val, abs_diff);
    }

    // 从SIMD寄存器中提取最大值
    alignas(32) float temp[8];
    _mm256_store_ps(temp, max_val);

    float result = 0;
    for (int j = 0; j < 8; j++) {
        if (temp[j] > result) {
            result = temp[j];
        }
    }

    // 处理剩余元素
    for (; i < d; i++) {
        float diff = std::abs(x[i] - y[i]);
        if (diff > result) {
            result = diff;
        }
    }

    return result;
}
#endif
```

### 12.7 针对特定维度的优化

```cpp
// faiss/utils/distances_simd.cpp
// 维度特化的距离计算

// D1: 单维向量
template <class C>
void fvec_op_ny_D1(float* dis, const float* x, const float* y, size_t ny) {
    float x0 = x[0];

#ifdef __AVX2__
    __m256 vx0 = _mm256_set1_ps(x0);
    size_t i = 0;

    for (; i + 8 <= ny; i += 8) {
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vdis = C::op(vx0, vy);
        _mm256_storeu_ps(dis + i, vdis);
    }

    for (; i < ny; i++) {
        dis[i] = C::op(x0, y[i]);
    }
#else
    for (size_t i = 0; i < ny; i++) {
        dis[i] = C::op(x0, y[i]);
    }
#endif
}

// D2: 二维向量
template <class C>
void fvec_op_ny_D2(float* dis, const float* x, const float* y, size_t ny) {
    // 特化实现,避免循环开销
    // ...

    // 使用矩阵转置优化
    // [y0_0, y1_0, y2_0, ...]  [y0_1, y1_1, y2_1, ...]
    // 转换为:
    // [y0_0, y0_1, y1_0, y1_1, ...]
}
```

### 12.8 距离度量与索引选择

```cpp
// 选择合适的索引类型

template <class C, class ResultHandler>
void exhaustive_search(
        const float* x,
        const float* xb,
        size_t nx,
        size_t d,
        ResultHandler& handler) {

    // 1. 归一化向量(用于内积)
    std::vector<float> xb_norm(nx * d);
    for (size_t i = 0; i < nx; i++) {
        float norm = std::sqrt(
            fvec_inner_product_ref(
                xb + i * d, xb + i * d, d));
        float inv_norm = 1.0f / (norm + 1e-10f);
        for (size_t j = 0; j < d; j++) {
            xb_norm[i * d + j] = xb[i * d + j] * inv_norm;
        }
    }

    // 2. 归一化查询
    std::vector<float> x_norm(d);
    float x_norm_val = std::sqrt(
        fvec_inner_product_ref(x, x, d));
    float x_inv_norm = 1.0f / (x_norm_val + 1e-10f);
    for (size_t j = 0; j < d; j++) {
        x_norm[j] = x[j] * x_inv_norm;
    }

    // 3. 计算内积
    std::vector<float> ip(nx);
    fvec_inner_products_ny(x_norm.data(), xb_norm.data(), d, nx, ip.data());

    // 4. 转换为距离(余弦距离 = 1 - 内积)
    for (size_t i = 0; i < nx; i++) {
        ip[i] = 1.0f - ip[i];
    }

    // 5. 交给结果处理器
    handler.add_results(nx, ip.data(), ids.data());
}
```

---

## 扩展阅读

- [Metric Learning Wikipedia](https://en.wikipedia.org/wiki/Metric_learning)
- [FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.00836)
- [Triplet Loss论文](https://arxiv.org/abs/1404.5667)
- [Mahalanobis距离](https://en.wikipedia.org/wiki/Mahalanobis_distance)
- [SimCLR论文](https://arxiv.org/abs/2002.05709)
- [SIMD优化技术](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
