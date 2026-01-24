# Faiss深度学习课程 - 第21天：向量压缩与编码技术

## 课程概述

第21天深入探讨向量数据库中的压缩与编码技术，这是实现大规模向量检索的关键。本课程将从理论基础到工程实践，全面解析各种压缩算法的原理、实现和权衡。

## 学习目标

- 理解向量压缩的信息论基础
- 掌握乘积量化的数学原理与优化
- 学习优化的乘积量化（OPQ）技术
- 深入了解标量量化变种
- 掌握混合编码策略
- 实践压缩比与精度的权衡

---

## 第一部分：压缩理论基础

### 1.1 信息论基础

向量压缩的核心是在保持检索精度的同时减少存储和计算开销。

#### 率失真理论

```cpp
// 率失真权衡分析
struct RateDistortionTradeoff {
    size_t bit_rate;        // 每向量比特数
    float distortion;       // 重建误差（L2）
    float recall;           // 检索召回率

    void print_analysis() const {
        printf("Rate: %zu bits/vector, Distortion: %.4f, Recall: %.2f%%\n",
               bit_rate, distortion, recall * 100);
    }
};

// 计算编码率（bits per dimension）
float compute_bpd(size_t n_bits, size_t d) {
    return static_cast<float>(n_bits) / d;
}
```

#### 压缩界

**Johnson-Lindenstrauss引理推论**：
- 对于n个d维向量，可以用O(log n)位保持相对距离
- 实际中需要考虑查询质量和延迟约束

```cpp
// 估计理论压缩下界
size_t estimate_compression_lower_bound(
    size_t n, float epsilon, size_t d) {

    // JL引理：目标维度 k >= O(epsilon^{-2} * log n)
    size_t k = static_cast<size_t>(
        std::ceil(4 * std::log(n) / (epsilon * epsilon))
    );

    // 假设float精度（32位）
    return k * 32;
}

// 对比不同压缩方法
void compare_compression_bounds(size_t n, size_t d) {
    printf("Dataset: %zu vectors of %zu dimensions\n", n, d);
    printf("Original size: %.2f MB\n", n * d * 4.0 / (1024*1024));

    size_t jl_bound = estimate_compression_lower_bound(n, 0.1f, d);
    printf("JL theoretical lower bound: %.2f MB\n",
           jl_bound / 8.0 / (1024*1024));

    // PQ (8 bits * d/8 subquantizers)
    size_t pq_size = n * d * 1;  // 1 byte per dimension
    printf("PQ compressed size: %.2f MB\n", pq_size / 8.0 / (1024*1024));
}
```

### 1.2 量化误差分析

#### 均匀量化误差

```cpp
// 均匀标量量化的MSE分析
float uniform_quantization_mse(
    float min_val, float max_val, size_t n_levels) {

    // 量化步长
    float delta = (max_val - min_val) / n_levels;

    // 均匀分布下的理论MSE = delta^2 / 12
    return (delta * delta) / 12.0f;
}

// 计算最优比特数（给定目标MSE）
size_t compute_optimal_bits_for_mse(
    float target_mse, float range, size_t d) {

    // 每维的MSE预算
    float mse_per_dim = target_mse / d;

    // 反推所需级数：mse = (range/2^n)^2 / 12
    // => 2^n = range / sqrt(12*mse)
    float n_levels = range / std::sqrt(12.0f * mse_per_dim);
    size_t n_bits = static_cast<size_t>(std::ceil(std::log2(n_levels)));

    return std::min(n_bits, 32UL);  // 上限32位
}
```

#### 高斯分布下的最优量化

```cpp
// Lloyd-Max量化器（针对高斯分布）
class GaussianQuantizer {
    float mean;      // 均值
    float std;       // 标准差
    size_t n_bits;   // 量化比特数
    std::vector<float> centroids;  // 质心
    std::vector<float> boundaries; // 决策边界

public:
    GaussianQuantizer(float mu, float sigma, size_t bits)
        : mean(mu), std(sigma), n_bits(bits) {

        size_t n_levels = 1 << bits;
        centroids.resize(n_levels);
        boundaries.resize(n_levels + 1);

        // 使用高斯分位数初始化
        init_gaussian_levels();
    }

private:
    void init_gaussian_levels() {
        // 高斯分布的最优量化器使用分位数
        size_t n_levels = 1 << n_bits;

        // 边界：使用高斯分位数
        for (size_t i = 0; i <= n_levels; i++) {
            float p = static_cast<float>(i) / n_levels;
            boundaries[i] = mean + std * inverse_gaussian_cdf(p);
        }

        // 质心：使用条件期望
        for (size_t i = 0; i < n_levels; i++) {
            centroids[i] = conditional_mean(
                boundaries[i], boundaries[i+1], mean, std);
        }
    }

    // 高斯分布的逆CDF（近似）
    float inverse_gaussian_cdf(float p) {
        // Beasley-Springer-Moro近似
        static const float a[4] = {
            -3.969683028665376e+01, 2.209460984245205e+02,
            -2.759285104469687e+02, 1.383577518672690e+02
        };
        static const float b[4] = {
            -5.447609879822406e+01, 1.615858368580409e+02,
            -1.556989798598866e+02, 6.680131188771972e+01
        };
        static const float c[9] = {
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00,
            -3.798642867009970e-02, -6.292612284327761e-02
        };

        float q = std::min(p, 1.0f - p);
        float t, u;

        if (q > 0.02425f) {
            // Rational approximation for central region
            u = q - 0.5f;
            t = u * u;
            u = u * (((((c[0]*t + c[1])*t + c[2])*t + c[3])*t + c[4])*t + c[5])
                    / ((((c[6]*t + c[7])*t + c[8])*t + 1.0f);
        } else {
            // Rational approximation for tail region
            t = std::sqrt(-2.0f * std::log(q));
            u = t + (((((c[0]*t + c[1])*t + c[2])*t + c[3])*t + c[4])*t + c[5])
                    / ((((c[6]*t + c[7])*t + c[8])*t + 1.0f);
        }

        return p > 0.5f ? -u : u;
    }

    // 区间[a,b]上截断高斯的条件期望
    float conditional_mean(float a, float b, float mu, float sigma) {
        float phi_a = std::exp(-0.5f * std::pow((a-mu)/sigma, 2));
        float phi_b = std::exp(-0.5f * std::pow((b-mu)/sigma, 2));
        float Phi_a = 0.5f * (1.0f + std::erf((a-mu)/(sigma*std::sqrt(2.0f))));
        float Phi_b = 0.5f * (1.0f + std::erf((b-mu)/(sigma*std::sqrt(2.0f))));

        return mu + sigma * (phi_a - phi_b) / (Phi_b - Phi_a);
    }
};
```

---

## 第二部分：乘积量化（PQ）深度解析

### 2.1 PQ的数学原理

#### 空间分解

```cpp
// PQ将d维空间分解为M个子空间
struct PQDecomposition {
    size_t d;           // 原始维度
    size_t M;           // 子空间数量
    size_t d_sub;       // 每个子空间维度 (d_sub = d / M)
    size_t nbits;       // 每子空间编码位数

    // 子空间索引映射
    std::vector<std::pair<size_t, size_t>> subspaces;

    PQDecomposition(size_t dim, size_t n_subquantizers, size_t bits)
        : d(dim), M(n_subquantizers), nbits(bits) {

        d_sub = d / M;
        assert(d % M == 0 && "Dimension must be divisible by M");

        // 创建子空间映射
        for (size_t m = 0; m < M; m++) {
            size_t start = m * d_sub;
            size_t end = start + d_sub;
            subspaces.push_back({start, end});
        }
    }

    // 获取向量在子空间m中的切片
    float* get_subvector(float* x, size_t m) {
        return x + subspaces[m].first;
    }

    const float* get_subvector(const float* x, size_t m) const {
        return x + subspaces[m].first;
    }
};
```

#### 量化码本学习

```cpp
// 使用k-means学习子量化器码本
class SubquantizerTrainer {
    size_t d_sub;       // 子空间维度
    size_t k;           // 码本大小 (k = 2^nbits)
    size_t max_iter;    // 最大迭代次数

public:
    SubquantizerTrainer(size_t dim, size_t nbits, size_t iter = 25)
        : d_sub(dim), k(1 << nbits), max_iter(iter) {}

    // 训练子量化器（简化版k-means）
    void train(
        const float* training_set,  // 训练集 [n x d_sub]
        size_t n,                    // 训练样本数
        float* centroids) const {    // 输出码本 [k x d_sub]

        // 初始化质心（随机选择）
        initialize_centroids(training_set, n, centroids);

        std::vector<size_t> assignments(n);
        std::vector<float> centroid_sums(k * d_sub);
        std::vector<size_t> centroid_counts(k);

        for (size_t iter = 0; iter < max_iter; iter++) {
            // E步：分配到最近的质心
            assign_to_nearest(training_set, n, centroids, assignments.data());

            // M步：更新质心
            std::fill(centroid_sums.begin(), centroid_sums.end(), 0.0f);
            std::fill(centroid_counts.begin(), centroid_counts.end(), 0);

            for (size_t i = 0; i < n; i++) {
                size_t c = assignments[i];
                const float* x = training_set + i * d_sub;

                for (size_t j = 0; j < d_sub; j++) {
                    centroid_sums[c * d_sub + j] += x[j];
                }
                centroid_counts[c]++;
            }

            // 计算新质心
            for (size_t c = 0; c < k; c++) {
                if (centroid_counts[c] > 0) {
                    for (size_t j = 0; j < d_sub; j++) {
                        centroids[c * d_sub + j] =
                            centroid_sums[c * d_sub + j] / centroid_counts[c];
                    }
                }
            }
        }
    }

private:
    void initialize_centroids(
        const float* training_set, size_t n, float* centroids) const {

        // k-means++初始化
        std::vector<float> min_distances(n * d_sub);
        std::vector<float> squared_distances(n);

        // 随机选择第一个质心
        size_t first_idx = rand() % n;
        std::memcpy(centroids, training_set + first_idx * d_sub,
                   d_sub * sizeof(float));

        for (size_t c = 1; c < k; c++) {
            // 计算到最近质心的距离
            for (size_t i = 0; i < n; i++) {
                float min_dist = INFINITY;
                for (size_t cc = 0; cc < c; cc++) {
                    float dist = fvec_L2sqr(
                        training_set + i * d_sub,
                        centroids + cc * d_sub,
                        d_sub
                    );
                    if (dist < min_dist) min_dist = dist;
                }
                squared_distances[i] = min_dist;
            }

            // 按距离平方概率选择下一个质心
            float sum = std::accumulate(squared_distances.begin(),
                                       squared_distances.end(), 0.0f);
            float r = rand() / (float)RAND_MAX * sum;
            float partial_sum = 0.0f;
            size_t idx = 0;

            for (size_t i = 0; i < n; i++) {
                partial_sum += squared_distances[i];
                if (partial_sum >= r) {
                    idx = i;
                    break;
                }
            }

            std::memcpy(centroids + c * d_sub,
                       training_set + idx * d_sub,
                       d_sub * sizeof(float));
        }
    }

    void assign_to_nearest(
        const float* training_set, size_t n,
        const float* centroids, size_t* assignments) const {

        for (size_t i = 0; i < n; i++) {
            float min_dist = INFINITY;
            size_t nearest = 0;

            for (size_t c = 0; c < k; c++) {
                float dist = fvec_L2sqr(
                    training_set + i * d_sub,
                    centroids + c * d_sub,
                    d_sub
                );
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest = c;
                }
            }
            assignments[i] = nearest;
        }
    }

    float fvec_L2sqr(const float* x, const float* y, size_t d) const {
        float sum = 0.0f;
        for (size_t i = 0; i < d; i++) {
            float diff = x[i] - y[i];
            sum += diff * diff;
        }
        return sum;
    }
};
```

### 2.2 SIMD优化的PQ编码

```cpp
// SIMD优化的PQ编码器
class SIMDEncoderPQ {
    PQDecomposition pq;
    std::vector<float> centroids;  // [M * k * d_sub]
    std::vector<uint8_t> codes;    // [n * M]

public:
    SIMDEncoderPQ(const PQDecomposition& decomp,
                 const float* cent)
        : pq(decomp), centroids(cent, cent + decomp.M * (1 << decomp.nbits) * decomp.d_sub) {}

    // 编码单个向量
    void encode(const float* x, uint8_t* code) const {
        for (size_t m = 0; m < pq.M; m++) {
            const float* x_sub = pq.get_subvector(x, m);
            code[m] = encode_subquantizer(x_sub, m);
        }
    }

    // 批量编码（SIMD优化）
    void encode_batch(
        const float* vectors,  // [n x d]
        size_t n,
        uint8_t* codes) const {  // [n x M]

#ifdef __AVX2__
        encode_batch_avx2(vectors, n, codes);
#else
        encode_batch_scalar(vectors, n, codes);
#endif
    }

private:
#ifdef __AVX2__
    void encode_batch_avx2(
        const float* vectors, size_t n, uint8_t* codes) const {

        const size_t k = 1 << pq.nbits;
        const size_t d_sub = pq.d_sub;

        for (size_t m = 0; m < pq.M; m++) {
            const float* centroids_m = centroids.data() + m * k * d_sub;
            size_t offset = m * d_sub;

            for (size_t i = 0; i < n; i++) {
                const float* x = vectors + i * pq.d + offset;

                // 计算到所有质心的距离
                std::vector<float, aligned_allocator<float, 32>> distances(k);

                for (size_t c = 0; c < k; c++) {
                    const float* cent = centroids_m + c * d_sub;

                    __m256 sum = _mm256_setzero_ps();
                    size_t j = 0;

                    // 8路展开
                    for (; j + 8 <= d_sub; j += 8) {
                        __m256 vx = _mm256_loadu_ps(x + j);
                        __m256 vc = _mm256_loadu_ps(cent + j);
                        __m256 diff = _mm256_sub_ps(vx, vc);
                        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
                    }

                    // 水平求和
                    float dist_array[8];
                    _mm256_storeu_ps(dist_array, sum);
                    float dist = dist_array[0] + dist_array[1] +
                                dist_array[2] + dist_array[3] +
                                dist_array[4] + dist_array[5] +
                                dist_array[6] + dist_array[7];

                    // 处理剩余元素
                    for (; j < d_sub; j++) {
                        float diff = x[j] - cent[j];
                        dist += diff * diff;
                    }

                    distances[c] = dist;
                }

                // 找最小距离
                size_t best_code = 0;
                float min_dist = distances[0];
                for (size_t c = 1; c < k; c++) {
                    if (distances[c] < min_dist) {
                        min_dist = distances[c];
                        best_code = c;
                    }
                }

                codes[i * pq.M + m] = static_cast<uint8_t>(best_code);
            }
        }
    }
#endif

    void encode_subquantizer_scalar(
        const float* x, size_t m, uint8_t* code) const {

        const size_t k = 1 << pq.nbits;
        const size_t d_sub = pq.d_sub;
        const float* centroids_m = centroids.data() + m * k * d_sub;

        float min_dist = INFINITY;
        size_t best_code = 0;

        for (size_t c = 0; c < k; c++) {
            const float* cent = centroids_m + c * d_sub;
            float dist = fvec_L2sqr(x, cent, d_sub);

            if (dist < min_dist) {
                min_dist = dist;
                best_code = c;
            }
        }

        code[0] = static_cast<uint8_t>(best_code);
    }
};
```

### 2.3 非对称距离计算（ADC）

```cpp
// 非对称距离计算（Asymmetric Distance Computation）
class ADCTable {
    size_t M;           // 子空间数
    size_t k;           // 每子空间码本大小
    size_t d_sub;       // 子空间维度

    // 预计算的查询-质心距离表 [M x k]
    std::vector<float> distance_tables;

public:
    ADCTable(size_t n_subquantizers, size_t nbits, size_t sub_dim)
        : M(n_subquantizers), k(1 << nbits), d_sub(sub_dim),
          distance_tables(M * k) {}

    // 为查询向量预计算距离表
    void precompute_distances(
        const float* query,              // [d]
        const PQDecomposition& pq,
        const float* centroids) {        // [M x k x d_sub]

        for (size_t m = 0; m < M; m++) {
            const float* q_sub = pq.get_subvector(query, m);
            const float* centroids_m = centroids + m * k * d_sub;

            for (size_t c = 0; c < k; c++) {
                const float* cent = centroids_m + c * d_sub;
                distance_tables[m * k + c] =
                    fvec_L2sqr(q_sub, cent, d_sub);
            }
        }
    }

    // 查表计算距离（SIMD优化）
    float lookup_distance(const uint8_t* code) const {
#ifdef __AVX2__
        return lookup_distance_avx2(code);
#else
        return lookup_distance_scalar(code);
#endif
    }

private:
#ifdef __AVX2__
    float lookup_distance_avx2(const uint8_t* code) const {
        __m256 sum = _mm256_setzero_ps();
        size_t m = 0;

        // 8个一组处理
        for (; m + 8 <= M; m += 8) {
            __m256 dists = _mm256_setr_ps(
                distance_tables[m * k + code[m]],
                distance_tables[(m+1) * k + code[m+1]],
                distance_tables[(m+2) * k + code[m+2]],
                distance_tables[(m+3) * k + code[m+3]],
                distance_tables[(m+4) * k + code[m+4]],
                distance_tables[(m+5) * k + code[m+5]],
                distance_tables[(m+6) * k + code[m+6]],
                distance_tables[(m+7) * k + code[m+7]]
            );
            sum = _mm256_add_ps(sum, dists);
        }

        // 水平求和
        float dist_array[8];
        _mm256_storeu_ps(dist_array, sum);
        float result = dist_array[0] + dist_array[1] + dist_array[2] +
                      dist_array[3] + dist_array[4] + dist_array[5] +
                      dist_array[6] + dist_array[7];

        // 处理剩余
        for (; m < M; m++) {
            result += distance_tables[m * k + code[m]];
        }

        return result;
    }
#endif

    float lookup_distance_scalar(const uint8_t* code) const {
        float sum = 0.0f;
        for (size_t m = 0; m < M; m++) {
            sum += distance_tables[m * k + code[m]];
        }
        return sum;
    }

    float fvec_L2sqr(const float* x, const float* y, size_t d) const {
        float sum = 0.0f;
        for (size_t i = 0; i < d; i++) {
            float diff = x[i] - y[i];
            sum += diff * diff;
        }
        return sum;
    }
};
```

---

## 第三部分：优化的乘积量化（OPQ）

### 3.1 OPQ理论

OPQ通过学习线性变换使得数据在变换后的空间更适合PQ。

#### 数学目标

```
min_{R,Q} Σ||x_i - R^T * Q(R * x_i)||^2
```

其中：
- R 是d×d正交旋转矩阵
- Q是PQ量化器

### 3.2 OPQ实现

```cpp
// OPQ训练器
class OPQTrainer {
    size_t d;           // 维度
    size_t M;           // 子空间数
    size_t nbits;       // 每子空间位数
    size_t max_iter;    // 最大迭代次数

    std::vector<float> rotation;   // 旋转矩阵 [d x d]
    std::vector<float> centroids;  // PQ码本 [M x k x d_sub]

public:
    OPQTrainer(size_t dim, size_t n_subquantizers, size_t bits, size_t iter = 50)
        : d(dim), M(n_subquantizers), nbits(bits), max_iter(iter),
          rotation(d * d), centroids(M * (1 << bits) * (dim / M)) {

        // 初始化为单位矩阵
        std::fill(rotation.begin(), rotation.end(), 0.0f);
        for (size_t i = 0; i < d; i++) {
            rotation[i * d + i] = 1.0f;
        }
    }

    // 训练OPQ（交替优化）
    void train(const float* training_set, size_t n) {
        std::vector<float> rotated_vectors(n * d);

        for (size_t iter = 0; iter < max_iter; iter++) {
            // 步骤1：应用当前旋转
            rotate_vectors(training_set, n, rotated_vectors.data());

            // 步骤2：在旋转空间训练PQ
            train_pq_on_rotated(rotated_vectors.data(), n);

            // 步骤3：更新旋转矩阵（参数化PQ）
            update_rotation_matrix(training_set, n);

            if (iter % 10 == 0) {
                float error = compute_reconstruction_error(
                    training_set, n);
                printf("Iteration %zu, reconstruction error: %.4f\n",
                       iter, error);
            }
        }
    }

    // 使用OPQ编码向量
    void encode(const float* x, uint8_t* code) const {
        // 应用旋转
        std::vector<float> rotated_x(d);
        rotate_vector(x, rotated_x.data());

        // PQ编码
        PQDecomposition pq(d, M, nbits);
        for (size_t m = 0; m < M; m++) {
            const float* x_sub = pq.get_subvector(rotated_x.data(), m);
            code[m] = encode_subquantizer(x_sub, m);
        }
    }

private:
    void rotate_vectors(
        const float* input, size_t n, float* output) const {

        // output = input * R^T
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < d; k++) {
                    sum += input[i * d + k] * rotation[j * d + k];
                }
                output[i * d + j] = sum;
            }
        }
    }

    void rotate_vector(const float* input, float* output) const {
        for (size_t j = 0; j < d; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < d; k++) {
                sum += input[k] * rotation[j * d + k];
            }
            output[j] = sum;
        }
    }

    void train_pq_on_rotated(const float* rotated_vectors, size_t n) {
        PQDecomposition pq(d, M, nbits);
        SubquantizerTrainer trainer(pq.d_sub, nbits);

        for (size_t m = 0; m < M; m++) {
            // 提取子空间数据
            std::vector<float> sub_data(n * pq.d_sub);
            for (size_t i = 0; i < n; i++) {
                const float* x = rotated_vectors + i * d;
                const float* x_sub = pq.get_subvector(x, m);
                std::memcpy(sub_data.data() + i * pq.d_sub, x_sub,
                           pq.d_sub * sizeof(float));
            }

            // 训练子量化器
            float* centroids_m = centroids.data() + m * (1 << nbits) * pq.d_sub;
            trainer.train(sub_data.data(), n, centroids_m);
        }
    }

    void update_rotation_matrix(const float* training_set, size_t n) {
        // 使用参数化方法（Polychaud训练）
        // 简化实现：使用PCA初始化

        // 计算协方差矩阵
        std::vector<float> mean(d, 0.0f);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                mean[j] += training_set[i * d + j];
            }
        }
        for (size_t j = 0; j < d; j++) mean[j] /= n;

        std::vector<float> cov(d * d, 0.0f);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                for (size_t k = 0; k < d; k++) {
                    float xj = training_set[i * d + j] - mean[j];
                    float xk = training_set[i * d + k] - mean[k];
                    cov[j * d + k] += xj * xk;
                }
            }
        }
        for (size_t j = 0; j < d * d; j++) cov[j] /= n;

        // 特征分解（简化：使用幂迭代）
        // 实际中应使用LAPACK/ARPACK等库
        // 这里仅作示意
    }

    float compute_reconstruction_error(const float* training_set, size_t n) {
        float total_error = 0.0f;

        for (size_t i = 0; i < std::min(n, (size_t)1000); i++) {
            // 编码
            std::vector<uint8_t> code(M);
            encode(training_set + i * d, code.data());

            // 解码（略）
            // 计算误差...
        }

        return total_error / std::min(n, (size_t)1000);
    }
};
```

---

## 第四部分：标量量化变种

### 4.1 各种标量量化方法

```cpp
// 标量量化器枚举
enum class ScalarQuantizerType {
    QT_8bit,        // 8位均匀量化
    QT_4bit,        // 4位均匀量化
    QT_8bit_uniform, // 均匀分布优化的8位
    QT_4bit_uniform,
    QT_8bit_fp16,   // 半精度浮点
    QT_bf16,        // bfloat16
    QT_8bit_direct, // 直接量化（无训练）
    QT_6bit         // 6位量化
};

// 通用标量量化器
class AdvancedScalarQuantizer {
    ScalarQuantizerType qtype;
    size_t d;           // 维度
    size_t n_bits;      // 位数
    bool trained;       // 是否已训练

    // 量化参数
    std::vector<float> min_vals;
    std::vector<float> max_vals;
    std::vector<float> scales;
    std::vector<float> biases;

public:
    AdvancedScalarQuantizer(size_t dim, ScalarQuantizerType type)
        : d(dim), qtype(type), trained(false) {

        switch (type) {
            case ScalarQuantizerType::QT_8bit:
                n_bits = 8;
                break;
            case ScalarQuantizerType::QT_4bit:
                n_bits = 4;
                break;
            case ScalarQuantizerType::QT_6bit:
                n_bits = 6;
                break;
            default:
                n_bits = 8;
        }

        min_vals.resize(d);
        max_vals.resize(d);
        scales.resize(d);
        biases.resize(d);
    }

    // 训练量化器
    void train(const float* training_set, size_t n) {
        switch (qtype) {
            case ScalarQuantizerType::QT_8bit_uniform:
                train_uniform(training_set, n);
                break;
            case ScalarQuantizerType::QT_4bit_uniform:
                train_uniform(training_set, n);
                break;
            default:
                train_range_based(training_set, n);
        }

        trained = true;
    }

    // 编码向量
    void encode_vector(const float* x, uint8_t* code) const {
        assert(trained && "Quantizer not trained");

        switch (n_bits) {
            case 8:
                encode_8bit(x, code);
                break;
            case 4:
                encode_4bit(x, code);
                break;
            case 6:
                encode_6bit(x, code);
                break;
        }
    }

    // 解码向量
    void decode_vector(const uint8_t* code, float* x) const {
        switch (n_bits) {
            case 8:
                decode_8bit(code, x);
                break;
            case 4:
                decode_4bit(code, x);
                break;
            case 6:
                decode_6bit(code, x);
                break;
        }
    }

private:
    void train_range_based(const float* training_set, size_t n) {
        // 找每维的最小最大值
        for (size_t j = 0; j < d; j++) {
            min_vals[j] = INFINITY;
            max_vals[j] = -INFINITY;

            for (size_t i = 0; i < n; i++) {
                float val = training_set[i * d + j];
                if (val < min_vals[j]) min_vals[j] = val;
                if (val > max_vals[j]) max_vals[j] = val;
            }

            // 避免零范围
            if (max_vals[j] - min_vals[j] < 1e-6f) {
                max_vals[j] = min_vals[j] + 1.0f;
            }

            // 计算缩放和偏置
            float range = max_vals[j] - min_vals[j];
            float qmax = static_cast<float>((1 << n_bits) - 1);
            scales[j] = qmax / range;
            biases[j] = -min_vals[j] * scales[j];
        }
    }

    void train_uniform(const float* training_set, size_t n) {
        // 针对均匀分布优化（使用3-sigma规则）
        for (size_t j = 0; j < d; j++) {
            // 计算均值和标准差
            float mean = 0.0f;
            for (size_t i = 0; i < n; i++) {
                mean += training_set[i * d + j];
            }
            mean /= n;

            float var = 0.0f;
            for (size_t i = 0; i < n; i++) {
                float diff = training_set[i * d + j] - mean;
                var += diff * diff;
            }
            var /= n;
            float std = std::sqrt(var);

            // 使用3-sigma范围
            min_vals[j] = mean - 3.0f * std;
            max_vals[j] = mean + 3.0f * std;

            float range = max_vals[j] - min_vals[j];
            float qmax = static_cast<float>((1 << n_bits) - 1);
            scales[j] = qmax / range;
            biases[j] = -min_vals[j] * scales[j];
        }
    }

    void encode_8bit(const float* x, uint8_t* code) const {
        for (size_t j = 0; j < d; j++) {
            float quantized = x[j] * scales[j] + biases[j];
            quantized = std::max(0.0f, std::min(255.0f, quantized));
            code[j] = static_cast<uint8_t>(std::round(quantized));
        }
    }

    void decode_8bit(const uint8_t* code, float* x) const {
        for (size_t j = 0; j < d; j++) {
            x[j] = (code[j] - biases[j]) / scales[j];
        }
    }

    void encode_4bit(const float* x, uint8_t* code) const {
        for (size_t j = 0; j < d; j++) {
            float quantized = x[j] * scales[j] + biases[j];
            quantized = std::max(0.0f, std::min(15.0f, quantized));
            uint8_t val = static_cast<uint8_t>(std::round(quantized));

            // 打包：两个4位值到一个字节
            if (j % 2 == 0) {
                code[j / 2] = val;
            } else {
                code[j / 2] |= (val << 4);
            }
        }
    }

    void decode_4bit(const uint8_t* code, float* x) const {
        for (size_t j = 0; j < d; j++) {
            uint8_t val = (j % 2 == 0) ?
                         (code[j / 2] & 0x0F) :
                         ((code[j / 2] >> 4) & 0x0F);
            x[j] = (val - biases[j]) / scales[j];
        }
    }

    void encode_6bit(const float* x, uint8_t* code) const {
        // 6位打包：4个6位值 = 3字节
        for (size_t j = 0; j < d; j++) {
            float quantized = x[j] * scales[j] + biases[j];
            quantized = std::max(0.0f, std::min(63.0f, quantized));
            uint8_t val = static_cast<uint8_t>(std::round(quantized));

            size_t byte_idx = (j * 6) / 8;
            size_t bit_offset = (j * 6) % 8;

            if (bit_offset <= 2) {
                code[byte_idx] |= (val << bit_offset);
                if (bit_offset + 6 > 8) {
                    code[byte_idx + 1] |= (val >> (8 - bit_offset));
                }
            }
        }
    }

    void decode_6bit(const uint8_t* code, float* x) const {
        for (size_t j = 0; j < d; j++) {
            size_t byte_idx = (j * 6) / 8;
            size_t bit_offset = (j * 6) % 8;

            uint8_t val;
            if (bit_offset <= 2) {
                val = (code[byte_idx] >> bit_offset) & 0x3F;
                if (bit_offset + 6 > 8 && byte_idx + 1 < d) {
                    val |= ((code[byte_idx + 1] & 0x3F) << (8 - bit_offset));
                }
            } else {
                val = 0; // 简化
            }

            x[j] = (val - biases[j]) / scales[j];
        }
    }
};
```

### 4.2 半精度浮点（FP16）

```cpp
// FP16/BF16量化器
class HalfPrecisionQuantizer {
    bool use_bfloat16;  // true=bfloat16, false=float16

public:
    HalfPrecisionQuantizer(bool bf16 = false) : use_bfloat16(bf16) {}

    void encode(const float* x, uint16_t* code, size_t d) const {
        for (size_t i = 0; i < d; i++) {
            code[i] = float_to_half(x[i]);
        }
    }

    void decode(const uint16_t* code, float* x, size_t d) const {
        for (size_t i = 0; i < d; i++) {
            x[i] = half_to_float(code[i]);
        }
    }

private:
    // IEEE 754 half precision (float16)
    uint16_t float_to_half(float f) const {
        // FP32: [SEEE EEEE EMMM MMMM MMMM MMMM MMMM MMMM]
        // FP16: [SEEE EMMM MMMM MMMM]

        uint32_t x = *((uint32_t*)&f);
        uint32_t sign = (x >> 16) & 0x8000;
        uint32_t exponent = (x >> 23) & 0xFF;
        uint32_t mantissa = x & 0x7FFFFF;

        if (use_bfloat16) {
            // BF16: 1-8-7 格式
            return (x >> 16) & 0xFFFF;
        }

        // FP16: 1-5-10 格式
        if (exponent == 255) {
            // Inf或NaN
            return sign | 0x7C00 | (mantissa >> 13);
        }

        // 处理指数偏置调整 (127 -> 15)
        int32_t new_exp = (int32_t)exponent - 127 + 15;

        if (new_exp <= 0) {
            // 下溢，返回零（或次正规数，这里简化）
            if (new_exp < -10) return sign;
            // 次正规数处理（略）
        }

        if (new_exp >= 31) {
            // 上溢，返回无穷
            return sign | 0x7C00;
        }

        return sign | (new_exp << 10) | (mantissa >> 13);
    }

    float half_to_float(uint16_t h) const {
        if (use_bfloat16) {
            // BF16转FP32：零扩展
            uint32_t x = h << 16;
            return *((float*)&x);
        }

        // FP16转FP32
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exponent = (h & 0x7C00) >> 10;
        uint32_t mantissa = (h & 0x03FF);

        if (exponent == 31) {
            // Inf或NaN
            uint32_t result = sign | 0x7F800000 | (mantissa << 13);
            return *((float*)&result);
        }

        if (exponent == 0) {
            if (mantissa == 0) {
                // 零
                uint32_t result = sign;
                return *((float*)&result);
            }
            // 次正规数（简化处理）
        }

        // 指数偏置调整 (15 -> 127)
        uint32_t new_exp = exponent + 127 - 15;
        uint32_t result = sign | (new_exp << 23) | (mantissa << 13);
        return *((float*)&result);
    }
};
```

---

## 第五部分：混合编码策略

### 5.1 多层量化

```cpp
// 多层量化器（粗量化+细量化）
class MultiLevelQuantizer {
    // 粗量化：标量量化
    std::unique_ptr<AdvancedScalarQuantizer> coarse_quantizer;

    // 细量化：PQ
    PQDecomposition pq_decomp;
    std::vector<float> pq_centroids;

    size_t d;  // 维度

public:
    MultiLevelQuantizer(size_t dim)
        : d(dim),
          coarse_quantizer(std::make_unique<AdvancedScalarQuantizer>(
              dim, ScalarQuantizerType::QT_8bit)),
          pq_decomp(dim, 8, 8),  // 8个子量化器，每8位
          pq_centroids(8 * 256 * (dim / 8)) {}

    void train(const float* training_set, size_t n) {
        // 步骤1：训练粗量化器
        coarse_quantizer->train(training_set, n);

        // 步骤2：计算残差
        std::vector<float> residuals(n * d);
        for (size_t i = 0; i < n; i++) {
            std::vector<uint8_t> coarse_code(d);
            coarse_quantizer->encode_vector(
                training_set + i * d, coarse_code.data());

            std::vector<float> coarse_recon(d);
            coarse_quantizer->decode_vector(
                coarse_code.data(), coarse_recon.data());

            for (size_t j = 0; j < d; j++) {
                residuals[i * d + j] =
                    training_set[i * d + j] - coarse_recon[j];
            }
        }

        // 步骤3：在残差上训练PQ
        SubquantizerTrainer trainer(pq_decomp.d_sub, pq_decomp.nbits);
        for (size_t m = 0; m < pq_decomp.M; m++) {
            // 提取子空间残差
            std::vector<float> sub_residuals(n * pq_decomp.d_sub);
            for (size_t i = 0; i < n; i++) {
                const float* r = residuals.data() + i * d;
                const float* r_sub = pq_decomp.get_subvector(r, m);
                std::memcpy(sub_residuals.data() + i * pq_decomp.d_sub,
                           r_sub, pq_decomp.d_sub * sizeof(float));
            }

            float* centroids_m = pq_centroids.data() +
                               m * (1 << pq_decomp.nbits) * pq_decomp.d_sub;
            trainer.train(sub_residuals.data(), n, centroids_m);
        }
    }

    // 编码：返回粗编码+细编码
    void encode(const float* x, std::vector<uint8_t>& coarse_code,
                std::vector<uint8_t>& fine_code) const {
        coarse_code.resize(d);
        fine_code.resize(pq_decomp.M);

        // 粗编码
        coarse_quantizer->encode_vector(x, coarse_code.data());

        // 解码粗编码
        std::vector<float> coarse_recon(d);
        coarse_quantizer->decode_vector(coarse_code.data(), coarse_recon.data());

        // 计算残差
        std::vector<float> residual(d);
        for (size_t j = 0; j < d; j++) {
            residual[j] = x[j] - coarse_recon[j];
        }

        // 细编码（PQ）
        for (size_t m = 0; m < pq_decomp.M; m++) {
            const float* r_sub = pq_decomp.get_subvector(residual.data(), m);
            fine_code[m] = encode_subquantizer(r_sub, m);
        }
    }

    // 解码
    void decode(const std::vector<uint8_t>& coarse_code,
                const std::vector<uint8_t>& fine_code,
                float* x) const {
        // 解码粗编码
        std::vector<float> coarse_recon(d);
        coarse_quantizer->decode_vector(coarse_code.data(), coarse_recon.data());

        // 解码细编码
        std::vector<float> fine_recon(d);
        for (size_t m = 0; m < pq_decomp.M; m++) {
            const float* centroids_m = pq_centroids.data() +
                                      m * 256 * pq_decomp.d_sub;
            const float* cent = centroids_m + fine_code[m] * pq_decomp.d_sub;
            std::memcpy(fine_recon.data() + m * pq_decomp.d_sub,
                       cent, pq_decomp.d_sub * sizeof(float));
        }

        // 合并
        for (size_t j = 0; j < d; j++) {
            x[j] = coarse_recon[j] + fine_recon[j];
        }
    }

private:
    uint8_t encode_subquantizer(const float* x, size_t m) const {
        const size_t k = 1 << pq_decomp.nbits;
        const float* centroids_m = pq_centroids.data() +
                                  m * k * pq_decomp.d_sub;

        float min_dist = INFINITY;
        uint8_t best_code = 0;

        for (size_t c = 0; c < k; c++) {
            const float* cent = centroids_m + c * pq_decomp.d_sub;
            float dist = 0.0f;
            for (size_t j = 0; j < pq_decomp.d_sub; j++) {
                float diff = x[j] - cent[j];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_code = static_cast<uint8_t>(c);
            }
        }

        return best_code;
    }
};
```

### 5.2 压缩比与精度权衡

```cpp
// 压缩分析工具
class CompressionAnalyzer {
    size_t d;  // 原始维度

public:
    CompressionAnalyzer(size_t dim) : d(dim) {}

    struct CompressionConfig {
        std::string name;
        size_t original_bits;     // 原始比特数 (通常 d * 32)
        size_t compressed_bits;   // 压缩后比特数
        float compression_ratio;  // 压缩比
        float recall;             // 检索召回率@10
        float encoding_time_ms;   // 编码时间
        float decoding_time_ms;   // 解码时间
        float mse;                // 均方误差
    };

    // 分析不同配置
    void analyze_configurations(
        const float* test_set,  // [n x d]
        size_t n) {

        std::vector<CompressionConfig> configs;

        // 配置1：原始FP32
        configs.push_back({
            "FP32 (baseline)",
            d * 32,
            d * 32,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f
        });

        // 配置2：8位标量量化
        {
            AdvancedScalarQuantizer sq(d, ScalarQuantizerType::QT_8bit);
            // sq.train(...);
            auto metrics = evaluate_quantizer(sq, test_set, n);
            configs.push_back({
                "SQ-8bit",
                d * 32,
                d * 8,
                4.0f,
                metrics.recall,
                metrics.encoding_time,
                metrics.decoding_time,
                metrics.mse
            });
        }

        // 配置3：4位标量量化
        {
            AdvancedScalarQuantizer sq(d, ScalarQuantizerType::QT_4bit);
            // sq.train(...);
            auto metrics = evaluate_quantizer(sq, test_set, n);
            configs.push_back({
                "SQ-4bit",
                d * 32,
                d * 4,
                8.0f,
                metrics.recall,
                metrics.encoding_time,
                metrics.decoding_time,
                metrics.mse
            });
        }

        // 配置4：PQ (8x8bit)
        configs.push_back({
            "PQ-8x8",
            d * 32,
            8 * 8,  // 8个子量化器，每8位
            (d * 32.0f) / (8 * 8),
            0.85f,  // 示例值
            1.2f,
            0.8f,
            0.023f
        });

        // 配置5：OPQ (8x8bit)
        configs.push_back({
            "OPQ-8x8",
            d * 32,
            8 * 8,
            (d * 32.0f) / (8 * 8),
            0.89f,  // 比PQ略好
            2.5f,   // 编码更慢
            0.8f,
            0.018f
        });

        // 打印对比表
        print_comparison_table(configs);
    }

private:
    struct Metrics {
        float recall;
        float encoding_time;
        float decoding_time;
        float mse;
    };

    Metrics evaluate_quantizer(
        const AdvancedScalarQuantizer& quantizer,
        const float* test_set,
        size_t n) {

        Metrics m{0};
        // 实际评估逻辑...
        return m;
    }

    void print_comparison_table(
        const std::vector<CompressionConfig>& configs) {

        printf("\n=== Compression Comparison ===\n");
        printf("%-20s %10s %10s %8s %8s %10s %10s %8s\n",
               "Method", "Orig(b)", "Comp(b)", "Ratio", "Recall",
               "Enc(ms)", "Dec(ms)", "MSE");
        printf("%s\n", std::string(90, '-').c_str());

        for (const auto& cfg : configs) {
            printf("%-20s %10zu %10zu %8.2fx %8.3f %10.2f %10.2f %8.4f\n",
                   cfg.name.c_str(),
                   cfg.original_bits,
                   cfg.compressed_bits,
                   cfg.compression_ratio,
                   cfg.recall,
                   cfg.encoding_time_ms,
                   cfg.decoding_time_ms,
                   cfg.mse);
        }
        printf("\n");
    }
};
```

---

## 第六部分：实践案例

### 6.1 完整的压缩索引示例

```cpp
// 使用OPQ + IVF的完整压缩索引
class CompressedIVFIndex {
    size_t d;                   // 维度
    size_t nlist;               // 倒排表数量
    size_t nbits;               // PQ位数

    // 粗量化器（用于IVF分区）
    std::unique_ptr<faiss::IndexFlatL2> coarse_quantizer;

    // OPQ编码器
    std::unique_ptr<OPQTrainer> opq;

    // 编码数据
    std::vector<std::vector<uint8_t>> codes;  // codes[list_id]
    std::vector<std::vector<faiss::idx_t>> ids; // ids[list_id]

public:
    CompressedIVFIndex(size_t dim, size_t n_lists, size_t bits)
        : d(dim), nlist(n_lists), nbits(bits),
          coarse_quantizer(std::make_unique<faiss::IndexFlatL2>(dim)),
          opq(std::make_unique<OPQTrainer>(dim, dim / 4, bits)) {

        codes.resize(nlist);
        ids.resize(nlist);
    }

    // 训练索引
    void train(const float* training_set, size_t n) {
        printf("Training coarse quantizer...\n");
        // 训练粗量化器（k-means）
        // ...

        printf("Training OPQ encoder...\n");
        // 训练OPQ
        // ...
    }

    // 添加向量
    void add(const float* vectors, size_t n) {
        for (size_t i = 0; i < n; i++) {
            const float* x = vectors + i * d;

            // 找到最近的倒排表
            faiss::idx_t list_id = find_nearest_list(x);

            // 编码向量
            std::vector<uint8_t> code(d / 4);  // 假设M = d/4
            opq->encode(x, code.data());

            // 存储到倒排表
            codes[list_id].push_back(...);
            ids[list_id].push_back(i);
        }
    }

    // 搜索
    void search(
        const float* queries,  // [nq x d]
        size_t nq,
        size_t k,
        float* distances,
        faiss::idx_t* labels) {

        // 为每个查询搜索
        for (size_t q = 0; q < nq; q++) {
            const float* query = queries + q * d;

            // 找到探测的倒排表（nprobe）
            std::vector<faiss::idx_t> probe_lists = find_probe_lists(query);

            // 使用ADC计算距离
            ADCTable adc_table(d / 4, nbits, 4);
            adc_table.precompute_distances(query, /*...*/);

            // 在探测的表中搜索
            std::priority_queue<
                std::pair<float, faiss::idx_t>
            > top_k;

            for (faiss::idx_t list_id : probe_lists) {
                for (size_t i = 0; i < codes[list_id].size(); i++) {
                    const uint8_t* code = codes[list_id][i].data();
                    float dist = adc_table.lookup_distance(code);

                    if (top_k.size() < k || dist < top_k.top().first) {
                        top_k.push({dist, ids[list_id][i]});
                        if (top_k.size() > k) top_k.pop();
                    }
                }
            }

            // 提取结果
            for (size_t i = 0; i < k && !top_k.empty(); i++) {
                labels[q * k + k - 1 - i] = top_k.top().second;
                distances[q * k + k - 1 - i] = top_k.top().first;
                top_k.pop();
            }
        }
    }

private:
    faiss::idx_t find_nearest_list(const float* x) {
        // 搜索到最近质心
        // ...
        return 0;
    }

    std::vector<faiss::idx_t> find_probe_lists(const float* query) {
        // 返回nprobe个最近的倒排表ID
        // ...
        return {0, 1, 2};
    }
};
```

### 6.2 压缩索引性能测试

```cpp
// 压缩索引基准测试
void benchmark_compression() {
    const size_t d = 128;
    const size_t n = 1000000;  // 100万向量
    const size_t nq = 1000;

    // 生成测试数据
    std::vector<float> database(n * d);
    std::vector<float> queries(nq * d);

    // 生成随机向量（标准正态分布）
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& val : database) val = dist(gen);
    for (auto& val : queries) val = dist(gen);

    printf("=== Compression Benchmark ===\n");
    printf("Database: %zu vectors of %zu dimensions\n", n, d);
    printf("Queries: %zu\n\n", nq);

    // 测试不同压缩配置
    std::vector<std::pair<std::string, size_t>> configs = {
        {"PQ-8x8", 64},
        {"PQ-16x8", 128},
        {"OPQ-8x8", 64},
        {"SQ-8bit", d * 8},
        {"SQ-4bit", d * 4}
    };

    for (const auto& [name, bits_per_vector] : configs) {
        printf("--- %s ---\n", name.c_str());
        printf("Bits per vector: %zu\n", bits_per_vector);
        printf("Compression ratio: %.2fx\n", (d * 32.0f) / bits_per_vector);
        printf("Memory: %.2f MB -> %.2f MB\n",
               n * d * 4.0 / (1024*1024),
               n * bits_per_vector / 8.0 / (1024*1024));

        // 创建并训练索引
        CompressedIVFIndex index(d, 100, 8);
        // index.train(database.data(), n);
        // index.add(database.data(), n);

        // 搜索测试
        // std::vector<float> distances(nq * 10);
        // std::vector<faiss::idx_t> labels(nq * 10);
        // index.search(queries.data(), nq, 10,
        //              distances.data(), labels.data());

        // 计算召回率（与精确搜索对比）
        // ...

        printf("\n");
    }
}
```

---

## 第七部分：高级优化

### 7.1 自适应量化

```cpp
// 自适应量化器：根据向量密度动态调整
class AdaptiveQuantizer {
    size_t d;
    std::vector<float> densities;  // 每维的密度估计

public:
    AdaptiveQuantizer(size_t dim) : d(dim), densities(dim) {}

    // 估计每维的分布密度
    void estimate_densities(const float* training_set, size_t n) {
        for (size_t j = 0; j < d; j++) {
            // 计算该维的直方图
            const int nbins = 256;
            std::vector<int> hist(nbins, 0);

            // 找范围
            float min_val = INFINITY, max_val = -INFINITY;
            for (size_t i = 0; i < n; i++) {
                float val = training_set[i * d + j];
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }

            // 构建直方图
            float bin_width = (max_val - min_val) / nbins;
            for (size_t i = 0; i < n; i++) {
                float val = training_set[i * d + j];
                int bin = static_cast<int>((val - min_val) / bin_width);
                bin = std::max(0, std::min(nbins - 1, bin));
                hist[bin]++;
            }

            // 密度 = 信息熵
            float entropy = 0.0f;
            for (int count : hist) {
                if (count > 0) {
                    float p = static_cast<float>(count) / n;
                    entropy -= p * std::log2(p);
                }
            }

            densities[j] = entropy;
        }
    }

    // 根据密度分配量化位数
    std::vector<size_t> allocate_bits(size_t total_bits) const {
        std::vector<size_t> bits_per_dim(d);

        // 按密度比例分配
        float total_density = std::accumulate(densities.begin(),
                                             densities.end(), 0.0f);

        size_t allocated = 0;
        for (size_t j = 0; j < d; j++) {
            // 密度高的维度分配更多位
            float ratio = densities[j] / total_density;
            bits_per_dim[j] = std::max(1UL,
                static_cast<size_t>(std::round(ratio * total_bits / d)));
            allocated += bits_per_dim[j];
        }

        // 调整以确保总和正确
        while (allocated < total_bits) {
            size_t best_j = 0;
            float min_mse = INFINITY;

            for (size_t j = 0; j < d; j++) {
                float mse_reduction = estimate_mse_reduction(
                    bits_per_dim[j] + 1, bits_per_dim[j], densities[j]);
                if (mse_reduction < min_mse) {
                    min_mse = mse_reduction;
                    best_j = j;
                }
            }
            bits_per_dim[best_j]++;
            allocated++;
        }

        return bits_per_dim;
    }

private:
    float estimate_mse_reduction(
        size_t new_bits, size_t old_bits, float density) const {

        // 量化MSE ~ delta^2 / 12，delta ~ range / 2^nbits
        float old_mse = 1.0f / (12.0f * (1 << (2 * old_bits)));
        float new_mse = 1.0f / (12.0f * (1 << (2 * new_bits)));
        return (old_mse - new_mse) * density;
    }
};
```

### 7.2 稀疏向量编码

```cpp
// 稀疏向量编码器
class SparseVectorEncoder {
    size_t d;  // 维度
    float sparsity_threshold;

public:
    SparseVectorEncoder(size_t dim, float threshold = 0.01f)
        : d(dim), sparsity_threshold(threshold) {}

    struct SparseCode {
        std::vector<uint32_t> indices;  // 非零索引
        std::vector<float> values;      // 非零值
    };

    // 编码稀疏向量
    SparseCode encode(const float* x) const {
        SparseCode code;

        for (size_t i = 0; i < d; i++) {
            if (std::abs(x[i]) > sparsity_threshold) {
                code.indices.push_back(static_cast<uint32_t>(i));
                code.values.push_back(x[i]);
            }
        }

        return code;
    }

    // 解码稀疏向量
    void decode(const SparseCode& code, float* x) const {
        std::fill(x, x + d, 0.0f);
        for (size_t i = 0; i < code.indices.size(); i++) {
            x[code.indices[i]] = code.values[i];
        }
    }

    // 稀疏距离计算
    float sparse_l2sqr(const SparseCode& a, const SparseCode& b) const {
        // 仅计算重叠的维度
        float sum = 0.0f;

        size_t i = 0, j = 0;
        while (i < a.indices.size() && j < b.indices.size()) {
            if (a.indices[i] == b.indices[j]) {
                float diff = a.values[i] - b.values[j];
                sum += diff * diff;
                i++;
                j++;
            } else if (a.indices[i] < b.indices[j]) {
                float diff = a.values[i];
                sum += diff * diff;
                i++;
            } else {
                float diff = b.values[j];
                sum += diff * diff;
                j++;
            }
        }

        // 剩余元素
        while (i < a.indices.size()) {
            float diff = a.values[i];
            sum += diff * diff;
            i++;
        }
        while (j < b.indices.size()) {
            float diff = b.values[j];
            sum += diff * diff;
            j++;
        }

        return sum;
    }

    // 计算压缩比
    float compression_ratio(const SparseCode& code) const {
        size_t original_bytes = d * sizeof(float);
        size_t compressed_bytes =
            code.indices.size() * sizeof(uint32_t) +
            code.values.size() * sizeof(float);

        return static_cast<float>(original_bytes) / compressed_bytes;
    }
};
```

---

## 实验练习

### 练习1：实现并比较不同量化方法

```cpp
void exercise_1_quantization_comparison() {
    // 1. 实现FP16、BF16、8位SQ、4位SQ
    // 2. 在SIFT数据集上测试
    // 3. 绘制：
    //    - 压缩比 vs 召回率曲线
    //    - 编码时间 vs 压缩比
    //    - MSE vs 压缩比
}
```

### 练习2：优化PQ的子空间划分

```cpp
void exercise_2_pq_subspace_optimization() {
    // 1. 实现动态子空间划分（非均匀）
    // 2. 使用PCA找到最优分解
    // 3. 比较不同M值（子空间数）的性能
    // 4. 分析维度间的相关性对PQ的影响
}
```

### 练习3：实现自适应混合编码

```cpp
void exercise_3_adaptive_hybrid_encoding() {
    // 1. 设计一个混合方案：
    //    - 重要维度：8位SQ
    //    - 其他维度：PQ
    // 2. 使用特征重要性（方差）选择
    // 3. 在不同数据集上验证
    // 4. 分析内存-精度权衡
}
```

### 练习4：稀疏编码优化

```cpp
void exercise_4_sparse_encoding() {
    // 1. 分析文本嵌入的稀疏性
    // 2. 实现稀疏+密集混合编码
    // 3. 设计高效的稀疏距离计算
    // 4. 评估在大规模文本检索中的性能
}
```

---

## 总结

第21天深入探讨了向量压缩与编码技术，涵盖：

1. **理论基础**：
   - 率失真理论
   - 量化误差分析
   - 压缩下界

2. **乘积量化**：
   - PQ数学原理
   - SIMD优化实现
   - 非对称距离计算（ADC）

3. **优化乘积量化**：
   - OPQ训练算法
   - 旋转矩阵学习

4. **标量量化变种**：
   - 均匀/非均匀量化
   - FP16/BF16
   - 自适应量化

5. **混合策略**：
   - 多层量化
   - 压缩比权衡

6. **高级技术**：
   - 自适应比特分配
   - 稀疏向量编码

**关键要点**：
- 压缩是在精度、速度、内存之间的三维权衡
- PQ适合低维子空间独立的情况
- OPQ通过旋转显著提升PQ性能
- 实际中常混合多种编码方法
- SIMD优化对编码/解码性能至关重要

## 下一步

第22天将深入分析ANN算法的复杂度，包括：
- 理论复杂度界
- 渐近分析
- 实际性能建模
- 可扩展性分析
