# Clustering 聚类算法底层实现深度剖析 - Clustering.cpp 源码解析

## 1. 概述

Faiss 的 `Clustering` 类实现了用于向量量化的 k-means 聚类算法，支持多种初始化方法、并行优化和渐进式维度聚类。这是 Product Quantizer、Residual Quantizer 等量化器训练的核心组件。

### 核心特性
- **多种初始化方法**: Random、k-means++、AFK-MC²
- **并行质心计算**: OpenMP 多线程优化
- **空簇处理**: 自动分裂大簇填充空簇
- **渐进式维度聚类**: ProgressiveDimClustering 逐步增加维度
- **编码向量支持**: 支持量化编码的训练数据

## 2. 数据结构

### 2.1 ClusteringParameters (Clustering.h:21-71)

```cpp
struct ClusteringParameters {
    int niter = 25;                      // 聚类迭代次数
    int nredo = 1;                       // 重复次数，保留最佳结果

    bool verbose = false;                // 是否输出日志
    bool spherical = false;              // 是否对质心进行 L2 归一化
    bool int_centroids = false;          // 是否将质心四舍五入为整数
    bool update_index = false;           // 是否每次迭代重新训练索引

    bool frozen_centroids = false;       // 是否冻结输入的质心
    int min_points_per_centroid = 39;    // 每个质心的最小点数（警告阈值）
    int max_points_per_centroid = 256;   // 每个质心的最大点数（采样阈值）
    int seed = 1234;                     // 随机种子

    size_t decode_block_size = 32768;    // 编码向量的解码批次大小
    bool check_input_data_for_NaNs = true;

    bool use_faster_subsampling = false; // 是否使用 splitmix64 快速采样

    // 初始化方法
    ClusteringInitMethod init_method = ClusteringInitMethod::RANDOM;
    uint16_t afkmc2_chain_length = 50;   // AFK-MC² 马尔可夫链长度
};
```

### 2.2 Clustering 类 (Clustering.h:93-141)

```cpp
struct Clustering : ClusteringParameters {
    size_t d;                    // 向量维度
    size_t k;                    // 质心数量

    // 质心 (k * d)，如果输入时 centroids 非空，则用作初始化
    std::vector<float> centroids;

    // 每次迭代的统计信息
    std::vector<ClusteringIterationStats> iteration_stats;

    Clustering(int d, int k);
    Clustering(int d, int k, const ClusteringParameters& cp);

    virtual void train(idx_t n, const float* x, Index& index,
                      const float* x_weights = nullptr);
    void post_process_centroids();
};
```

### 2.3 ClusteringIterationStats (Clustering.h:73-79)

```cpp
struct ClusteringIterationStats {
    float obj;               // 目标函数值（距离总和）
    double time;             // 本次迭代耗时（秒）
    double time_search;      // 仅搜索阶段耗时
    double imbalance_factor; // 不平衡因子
    int nsplit;              // 分裂操作次数
};
```

## 3. 初始化方法

### 3.1 三种初始化方法对比 (ClusteringInitialization.h)

| 方法 | 时间复杂度 | 质量保证 | 参考论文 |
|------|-----------|---------|---------|
| **RANDOM** | O(k) | 无保证 | - |
| **KMEANS_PLUS_PLUS** | O(nkd) | O(log k) 近似 | Arthur & Vassilvitskii, 2006 |
| **AFK_MC2** | O(nd) + O(mk²d) | 理论保证 | Bachem et al., 2016 |

### 3.2 Random 初始化 (ClusteringInitialization.cpp:177-190)

```cpp
void ClusteringInitialization::init_random(
        size_t n,
        const float* x,
        float* centroids) const {
    // 使用 rand_perm 生成随机排列
    std::vector<int> perm(n);
    rand_perm(perm.data(), n, seed);

    // 复制前 k 个点作为初始质心
    for (size_t i = 0; i < k; i++) {
        std::memcpy(centroids + i * d, x + perm[i] * d, d * sizeof(float));
    }
}
```

**特点**: 最快的初始化，但质量不稳定，容易陷入局部最优。

### 3.3 k-means++ 初始化 (ClusteringInitialization.cpp:192-238)

k-means++ 使用 D² 采样（距离平方采样）来选择初始质心：

```cpp
void ClusteringInitialization::init_kmeans_plus_plus(
        size_t n,
        const float* x,
        float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids) const {
    std::mt19937_64 rng(get_seed(seed));
    std::vector<double> min_distances(n);

    // 初始化距离数组
    auto result = init_distances_for_d2_sampling(
            d, n, x, centroids,
            n_existing_centroids, existing_centroids,
            min_distances, rng);

    // 如果已有质心，直接跳过第一个选择
    if (result.first_new_centroid_idx == 1 && k == 1) {
        return;
    }

    std::vector<double> cumsum(n);

    // D² 采样选择剩余的 k-1 个质心
    for (size_t c = result.first_new_centroid_idx; c < k; c++) {
        // 计算累积和
        cumsum[0] = min_distances[0];
        for (size_t i = 1; i < n; i++) {
            cumsum[i] = cumsum[i - 1] + min_distances[i];
        }

        // 按概率 proportional to D(x)² 采样
        size_t next_idx = sample_from_cumsum(cumsum, rng);

        // 复制选中的点作为新质心
        float* new_centroid = centroids + c * d;
        std::memcpy(new_centroid, x + next_idx * d, d * sizeof(float));

        // 增量更新最小距离
        for (size_t i = 0; i < n; i++) {
            double dist = fvec_L2sqr(x + i * d, new_centroid, d);
            min_distances[i] = std::min(min_distances[i], dist);
        }
    }
}
```

**核心思想**:
```
1. 随机选择第一个质心
2. 对于每个点 x，计算 D(x) = 到最近质心的距离
3. 按 probability proportional to D(x)² 选择下一个质心
4. 重复直到选择 k 个质心
```

**算法伪代码**:
```
Select first centroid uniformly at random
for c = 2 to k:
    for each point x:
        D(x) = min distance to any selected centroid
    Select next centroid with probability D(x)² / Σ D(x)²
```

### 3.4 AFK-MC² 初始化 (ClusteringInitialization.cpp:240-353)

AFK-MC² 使用马尔可夫链蒙特卡洛 (MCMC) 采样来近似 k-means++：

```cpp
void ClusteringInitialization::init_afkmc2(
        size_t n,
        const float* x,
        float* centroids,
        size_t n_existing_centroids,
        const float* existing_centroids) const {
    std::mt19937_64 rng(get_seed(seed));
    std::uniform_real_distribution<double> uniform_01(0.0, 1.0);

    // 跟踪已选择的质心，防止重复
    std::unordered_set<size_t> selected_centroids;

    // 计算提案分布 q(x)
    // q(x) = 0.5 * D(x)² / ΣD(x)² + 0.5 * 1/n
    // 混合 D² 采样和均匀采样
    std::vector<double> dist_to_nearest(n);
    auto result = init_distances_for_d2_sampling(
            d, n, x, centroids,
            n_existing_centroids, existing_centroids,
            dist_to_nearest, rng);

    // 计算 q(x) 和累积和
    std::vector<double> q(n);
    std::vector<double> q_cumsum(n);
    double uniform_term = 0.5 / static_cast<double>(n);

    for (size_t i = 0; i < n; i++) {
        double d2_term = (result.sum_d2 > 0)
                ? 0.5 * dist_to_nearest[i] / result.sum_d2
                : 0.0;
        q[i] = d2_term + uniform_term;
        q_cumsum[i] = (i > 0 ? q_cumsum[i - 1] : 0.0) + q[i];
    }

    // 主循环：使用 MCMC 选择剩余质心
    for (size_t c = result.first_new_centroid_idx; c < k; c++) {
        // 从 q 采样初始候选
        size_t current_idx;
        do {
            current_idx = sample_from_cumsum(q_cumsum, rng);
        } while (selected_centroids.count(current_idx) > 0);

        // 计算到最近质心的距离
        double current_dist = distance_to_nearest_centroid(
                d, c, x, current_idx, centroids,
                n_existing_centroids, existing_centroids);
        double current_q = q[current_idx];

        // 运行马尔可夫链
        for (size_t m = 0; m < afkmc2_chain_length; m++) {
            // 从 q 采样提案
            size_t proposed_idx = sample_from_cumsum(q_cumsum, rng);

            if (selected_centroids.count(proposed_idx) > 0) {
                continue;
            }

            double proposed_dist = distance_to_nearest_centroid(
                    d, c, x, proposed_idx, centroids,
                    n_existing_centroids, existing_centroids);
            double proposed_q = q[proposed_idx];

            // Metropolis-Hastings 接受率
            double acceptance_prob = 0.0;
            if (current_dist <= 0) {
                acceptance_prob = 0.0;
            } else if (proposed_q > 0) {
                double numerator = proposed_dist * current_q;
                double denominator = current_dist * proposed_q;
                acceptance_prob = std::min(1.0, numerator / denominator);
            }

            if (uniform_01(rng) < acceptance_prob) {
                current_idx = proposed_idx;
                current_dist = proposed_dist;
                current_q = proposed_q;
            }
        }

        // 使用链的最终状态作为新质心
        selected_centroids.insert(current_idx);
        std::memcpy(centroids + c * d, x + current_idx * d, d * sizeof(float));
    }
}
```

**提案分布**:
```
q(x) = α * q_D²(x) + (1-α) * q_uniform(x)
     = 0.5 * D(x)² / ΣD(x)² + 0.5 * 1/n
```

**Metropolis-Hastings 接受率**:
```
α = min(1, D(y)² * q(x) / (D(x)² * q(y)))
```

**优势**:
- 比纯 k-means++ 快，尤其是在 k 较大时
- 有理论质量保证
- 均匀采样部分确保所有点都有非零概率被选中

## 4. 主训练循环

### 4.1 train_encoded 实现 (Clustering.cpp:267-605)

```cpp
void Clustering::train_encoded(
        idx_t nx,
        const uint8_t* x_in,
        const Index* codec,
        Index& index,
        const float* weights) {
    // 参数验证
    FAISS_THROW_IF_NOT(nx >= k);
    FAISS_THROW_IF_NOT(!codec || codec->d == d);
    FAISS_THROW_IF_NOT(index.d == d);

    double t0 = getmillisecs();

    // 检查 NaN
    if (!codec && check_input_data_for_NaNs) {
        const float* x = reinterpret_cast<const float*>(x_in);
        for (size_t i = 0; i < nx * d; i++) {
            FAISS_THROW_IF_NOT(std::isfinite(x[i]));
        }
    }

    const uint8_t* x = x_in;
    size_t line_size = codec ? codec->sa_code_size() : sizeof(float) * d;

    // 采样：如果数据点太多，进行子采样
    if (nx > k * max_points_per_centroid) {
        uint8_t* x_new;
        float* weights_new;
        nx = subsample_training_set(
                *this, nx, x, line_size, weights,
                &x_new, &weights_new);
        del1.reset(x_new);
        x = x_new;
        del3.reset(weights_new);
        weights = weights_new;
    }

    // 特殊情况：nx == k
    if (nx == k) {
        centroids.resize(d * k);
        if (!codec) {
            memcpy(centroids.data(), x_in, sizeof(float) * d * k);
        } else {
            codec->sa_decode(nx, x_in, centroids.data());
        }
        index.reset();
        index.add(k, centroids.data());
        return;
    }

    // 分配缓冲区
    std::unique_ptr<idx_t[]> assign(new idx_t[nx]);
    std::unique_ptr<float[]> dis(new float[nx]);

    // 跟踪最佳迭代（用于 nredo）
    bool lower_is_better = !is_similarity_metric(index.metric_type);
    float best_obj = lower_is_better ? HUGE_VALF : -HUGE_VALF;
    std::vector<float> best_centroids;

    // 支持输入质心
    size_t n_input_centroids = centroids.size() / d;

    // 主循环：多次重试
    for (int redo = 0; redo < nredo; redo++) {
        // 初始化质心
        centroids.resize(d * k);
        size_t k_to_init = k - n_input_centroids;

        if (k_to_init > 0) {
            if (init_method == ClusteringInitMethod::RANDOM) {
                // 随机初始化
                std::vector<int> perm(nx);
                rand_perm(perm.data(), nx, actual_seed + 1 + redo * 15486557L);
                for (size_t i = 0; i < k_to_init; i++) {
                    if (!codec) {
                        memcpy(centroids.data() + (n_input_centroids + i) * d,
                               x + perm[n_input_centroids + i] * line_size,
                               line_size);
                    } else {
                        codec->sa_decode(
                                1,
                                x + perm[n_input_centroids + i] * line_size,
                                centroids.data() + (n_input_centroids + i) * d);
                    }
                }
            } else {
                // k-means++ 或 AFK-MC²
                const float* x_float = nullptr;
                std::vector<float> x_decoded;

                if (!codec) {
                    x_float = reinterpret_cast<const float*>(x);
                } else {
                    x_decoded.resize(nx * d);
                    codec->sa_decode(nx, x, x_decoded.data());
                    x_float = x_decoded.data();
                }

                ClusteringInitialization initializer(d, k_to_init);
                initializer.method = init_method;
                initializer.seed = actual_seed + 1 + redo * 15486557L;
                initializer.afkmc2_chain_length = afkmc2_chain_length;
                initializer.init_centroids(
                        nx,
                        x_float,
                        centroids.data() + n_input_centroids * d,
                        n_input_centroids,
                        n_input_centroids > 0 ? centroids.data() : nullptr);
            }
        }

        post_process_centroids();

        // 准备索引
        if (index.ntotal != 0) {
            index.reset();
        }

        if (!index.is_trained) {
            index.train(k, centroids.data());
        }

        index.add(k, centroids.data());

        // k-means 迭代
        float obj = 0;
        for (int i = 0; i < niter; i++) {
            double t0s = getmillisecs();

            // Step 1: 分配点到最近的质心
            if (!codec) {
                index.search(
                        nx,
                        reinterpret_cast<const float*>(x),
                        1,
                        dis.get(),
                        assign.get());
            } else {
                // 分批解码并搜索
                for (size_t i0 = 0; i0 < nx; i0 += decode_block_size) {
                    size_t i1 = std::min(i0 + decode_block_size, nx);
                    codec->sa_decode(
                            i1 - i0, x + codec->sa_code_size() * i0,
                            decode_buffer.data());
                    index.search(
                            i1 - i0,
                            decode_buffer.data(),
                            1,
                            dis.get() + i0,
                            assign.get() + i0);
                }
            }

            // 累积目标函数
            obj = 0;
            for (int j = 0; j < nx; j++) {
                obj += dis[j];
            }

            // Step 2: 更新质心
            std::vector<float> hassign(k);

            size_t k_frozen = frozen_centroids ? n_input_centroids : 0;
            compute_centroids(
                    d, k, nx, k_frozen,
                    x, codec, assign.get(), weights,
                    hassign.data(),
                    centroids.data());

            // Step 3: 处理空簇
            int nsplit = split_clusters(
                    d, k, nx, k_frozen,
                    hassign.data(), centroids.data());

            // 收集统计信息
            ClusteringIterationStats stats = {
                    obj,
                    (getmillisecs() - t0) / 1000.0,
                    t_search_tot / 1000,
                    imbalance_factor(nx, k, assign.get()),
                    nsplit};
            iteration_stats.push_back(stats);

            if (verbose) {
                printf("  Iteration %d (%.2f s, search %.2f s): "
                       "objective=%g imbalance=%.3f nsplit=%d\n",
                       i, stats.time, stats.time_search,
                       stats.obj, stats.imbalance_factor, nsplit);
            }

            post_process_centroids();

            // 准备下一次迭代
            index.reset();
            if (update_index) {
                index.train(k, centroids.data());
            }
            index.add(k, centroids.data());

            // 早停：如果目标函数不再变化
            if (i > 0) {
                float prev_obj = iteration_stats[iteration_stats.size() - 2].obj;
                if (obj == prev_obj) {
                    if (verbose) {
                        printf("\n  Converged at iteration %d\n", i);
                    }
                    break;
                }
            }
        }

        // 跟踪最佳结果
        if (nredo > 1) {
            if ((lower_is_better && obj < best_obj) ||
                (!lower_is_better && obj > best_obj)) {
                best_centroids = centroids;
                best_iteration_stats = iteration_stats;
                best_obj = obj;
            }
            index.reset();
        }
    }

    // 恢复最佳结果
    if (nredo > 1) {
        centroids = best_centroids;
        iteration_stats = best_iteration_stats;
        index.reset();
        index.add(k, best_centroids.data());
    }
}
```

### 4.2 训练流程图

```
Input: 训练集 X (n x d), 索引 index, 参数
Output: 质心 centroids (k x d)

For redo = 1 to nredo:
    1. 初始化质心:
       -RANDOM: 随机选择 k 个点
       -KMEANS_PLUS_PLUS: D² 采样
       -AFK_MC2: MCMC 采样

    2. 将质心添加到索引

    3. For iter = 1 to niter:
        a. 分配: index.search(X) -> assign, dis
        b. 累积: obj = sum(dis)
        c. 更新: compute_centroids() -> new_centroids
        d. 分裂: split_clusters() 处理空簇
        e. 后处理: spherical / int_centroids
        f. 更新索引: index.add(centroids)
        g. 早停检查: if obj == prev_obj: break

    4. 如果 nredo > 1 且 obj 更优，保存 centroids

选择最佳 centroids 作为输出
```

## 5. 质心计算优化

### 5.1 并行质心计算 (Clustering.cpp:135-204)

```cpp
void compute_centroids(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        const uint8_t* x,
        const Index* codec,
        const int64_t* assign,
        const float* weights,
        float* hassign,
        float* centroids) {
    k -= k_frozen;
    centroids += k_frozen * d;

    // 清零质心
    memset(centroids, 0, sizeof(*centroids) * d * k);

    size_t line_size = codec ? codec->sa_code_size() : d * sizeof(float);

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // 每个线程负责一部分质心
        size_t c0 = (k * rank) / nt;
        size_t c1 = (k * (rank + 1)) / nt;
        std::vector<float> decode_buffer(d);

        for (size_t i = 0; i < n; i++) {
            int64_t ci = assign[i];
            assert(ci >= 0 && ci < k + k_frozen);
            ci -= k_frozen;

            // 只处理属于当前线程的质心
            if (ci >= c0 && ci < c1) {
                float* c = centroids + ci * d;
                const float* xi;

                // 解码向量（如果需要）
                if (!codec) {
                    xi = reinterpret_cast<const float*>(x + i * line_size);
                } else {
                    float* xif = decode_buffer.data();
                    codec->sa_decode(1, x + i * line_size, xif);
                    xi = xif;
                }

                // 累加到质心
                if (weights) {
                    float w = weights[i];
                    hassign[ci] += w;
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j] * w;
                    }
                } else {
                    hassign[ci] += 1.0;
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j];
                    }
                }
            }
        }
    }

    // 归一化质心
#pragma omp parallel for
    for (idx_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) {
            continue;
        }
        float norm = 1 / hassign[ci];
        float* c = centroids + ci * d;
        for (size_t j = 0; j < d; j++) {
            c[j] *= norm;
        }
    }
}
```

**优化策略**:
1. **线程局部质心**: 每个线程处理独立的质心范围，避免竞争
2. **线程局部解码缓冲区**: 避免共享缓冲区的竞争
3. **SIMD 友好**: 连续内存访问模式有利于向量化
4. **条件归一化**: 只归一化非空质心

### 5.2 空簇处理 (Clustering.cpp:209-263)

```cpp
int split_clusters(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        float* hassign,
        float* centroids) {
    k -= k_frozen;
    centroids += k_frozen * d;

    size_t nsplit = 0;
    RandomGenerator rng(1234);

    for (size_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) { // 空簇
            // 按概率 proportional to (|Cj| - 1) 选择要分裂的簇
            size_t cj;
            for (cj = 0; true; cj = (cj + 1) % k) {
                float p = (hassign[cj] - 1.0) / (float)(n - k);
                float r = rng.rand_float();
                if (r < p) {
                    break;
                }
            }

            // 复制簇 cj 的质心到 ci
            memcpy(centroids + ci * d,
                   centroids + cj * d,
                   sizeof(*centroids) * d);

            // 添加微小对称扰动
            for (size_t j = 0; j < d; j++) {
                if (j % 2 == 0) {
                    centroids[ci * d + j] *= 1 + EPS;
                    centroids[cj * d + j] *= 1 - EPS;
                } else {
                    centroids[ci * d + j] *= 1 - EPS;
                    centroids[cj * d + j] *= 1 + EPS;
                }
            }

            // 假设均匀分裂
            hassign[ci] = hassign[cj] / 2;
            hassign[cj] -= hassign[ci];
            nsplit++;
        }
    }

    return nsplit;
}
```

**EPS 值** (Clustering.cpp:207):
```cpp
#define EPS (1 / 1024.)  // 略大于 float16 的机器精度
```

**分裂策略**:
```
For each empty cluster ci:
    1. Find cluster cj with probability proportional to (|Cj| - 1)
    2. Copy centroid from cj to ci
    3. Apply small perturbation:
       - Even dimensions: ci *= 1+ε, cj *= 1-ε
       - Odd dimensions:  ci *= 1-ε, cj *= 1+ε
    4. Assume even split: |ci| = |cj| / 2
```

## 6. 子采样优化

### 6.1 子采样实现 (Clustering.cpp:70-120)

```cpp
idx_t subsample_training_set(
        const Clustering& clus,
        idx_t nx,
        const uint8_t* x,
        size_t line_size,
        const float* weights,
        uint8_t** x_out,
        float** weights_out) {
    if (clus.verbose) {
        printf("Sampling a subset of %zd / %" PRId64 " for training\n",
               clus.k * clus.max_points_per_centroid,
               nx);
    }

    const uint64_t actual_seed = get_actual_rng_seed(clus.seed);

    std::vector<int> perm;
    if (clus.use_faster_subsampling) {
        // 使用 splitmix64 快速随机数生成器
        SplitMix64RandomGenerator rng(actual_seed);

        const idx_t new_nx = clus.k * clus.max_points_per_centroid;
        perm.resize(new_nx);
        for (idx_t i = 0; i < new_nx; i++) {
            perm[i] = rng.rand_int(nx);  // 允许重复
        }
    } else {
        // 使用默认 RNG 生成排列（无重复）
        perm.resize(nx);
        rand_perm(perm.data(), nx, actual_seed);
    }

    nx = clus.k * clus.max_points_per_centroid;
    uint8_t* x_new = new uint8_t[nx * line_size];
    *x_out = x_new;

    // 复制选中的向量
    for (idx_t i = 0; i < nx; i++) {
        memcpy(x_new + i * line_size, x + perm[i] * line_size, line_size);
    }

    // 处理权重
    if (weights) {
        float* weights_new = new float[nx];
        for (idx_t i = 0; i < nx; i++) {
            weights_new[i] = weights[perm[i]];
        }
        *weights_out = weights_new;
    } else {
        *weights_out = nullptr;
    }

    return nx;
}
```

**两种采样模式对比**:

| 模式 | 速度 | 重复 | 内存 |
|------|------|------|------|
| `use_faster_subsampling=false` | 较慢 | 无重复 | O(n) |
| `use_faster_subsampling=true` | 快 | 可能有重复 | O(k*max_ppc) |

## 7. 渐进式维度聚类

### 7.1 ProgressiveDimClustering (Clustering.cpp:688-748)

渐进式维度聚类从低维开始逐步增加维度，有助于避免局部最优：

```cpp
void ProgressiveDimClustering::train(
        idx_t n,
        const float* x,
        ProgressiveDimIndexFactory& factory) {
    int d_prev = 0;

    PCAMatrix pca(d, d);
    std::vector<float> xbuf;

    // 可选: 应用 PCA
    if (apply_pca) {
        if (verbose) {
            printf("Training PCA transform\n");
        }
        pca.train(n, x);
        if (verbose) {
            printf("Apply PCA\n");
        }
        xbuf.resize(n * d);
        pca.apply_noalloc(n, x, xbuf.data());
        x = xbuf.data();
    }

    // 渐进式增加维度
    for (int iter = 0; iter < progressive_dim_steps; iter++) {
        // 维度按指数增长: d^(1 + iter/steps)
        int di = int(pow(d, (1. + iter) / progressive_dim_steps));

        if (verbose) {
            printf("Progressive dim step %d: cluster in dimension %d\n",
                   iter, di);
        }

        std::unique_ptr<Index> clustering_index(factory(di));

        Clustering clus(di, k, *this);

        // 从前一步的质心热启动
        if (d_prev > 0) {
            clus.centroids.resize(k * di);
            copy_columns(
                    k, d_prev, centroids.data(),
                    di, clus.centroids.data());
        }

        // 提取前 di 维
        std::vector<float> xsub(n * di);
        copy_columns(n, d, x, di, xsub.data());

        // 在 di 维空间聚类
        clus.train(n, xsub.data(), *clustering_index.get());

        centroids = clus.centroids;
        iteration_stats.insert(
                iteration_stats.end(),
                clus.iteration_stats.begin(),
                clus.iteration_stats.end());

        d_prev = di;
    }

    // 反向 PCA 变换
    if (apply_pca) {
        if (verbose) {
            printf("Revert PCA transform on centroids\n");
        }
        std::vector<float> cent_transformed(d * k);
        pca.reverse_transform(k, centroids.data(), cent_transformed.data());
        cent_transformed.swap(centroids);
    }
}
```

### 7.2 维度增长曲线

```
d = 128, progressive_dim_steps = 10

Step | Dimension | Growth Rate
-----|-----------|-------------
  0  |     2     | 128^(1/10) ≈ 1.74
  1  |     3     | ~1.74x
  2  |     6     | ~1.74x
  3  |    11     |
  4  |    20     |
  5  |    36     |
  6  |    65     |
  7  |   116     |
  8  |   128     | (capped at d)
  9  |   128     |
```

**公式**:
```
dim(iter) = min(d, round(d^((1 + iter) / steps)))
```

### 7.3 copy_columns 实现 (Clustering.cpp:677-684)

```cpp
void copy_columns(idx_t n, idx_t d1, const float* src, idx_t d2, float* dest) {
    idx_t d = std::min(d1, d2);
    for (idx_t i = 0; i < n; i++) {
        memcpy(dest, src, sizeof(float) * d);
        src += d1;
        dest += d2;
    }
}
```

**用途**: 从高维向量中提取前 di 维，或将低维质心扩展到高维（用零填充）。

## 8. 后处理

### 8.1 post_process_centroids (Clustering.cpp:35-45)

```cpp
void Clustering::post_process_centroids() {
    // 球形化: L2 归一化
    if (spherical) {
        fvec_renorm_L2(d, k, centroids.data());
    }

    // 整数化: 四舍五入
    if (int_centroids) {
        for (size_t i = 0; i < centroids.size(); i++) {
            centroids[i] = roundf(centroids[i]);
        }
    }
}
```

**球形化**:
```
For each centroid c:
    c = c / ||c||_2
```

用于内积搜索，确保所有质心在单位超球面上。

## 9. 不平衡因子

### 9.1 imbalance_factor 计算

```cpp
// imbal.cpp
double imbalance_factor(int n, int k, const int64_t* assign) {
    std::vector<int> hist(k, 0);
    for (int i = 0; i < n; i++) {
        hist[assign[i]]++;
    }

    double total = 0;
    double ideal = n / (double)k;

    for (int i = 0; i < k; i++) {
        double diff = hist[i] - ideal;
        total += diff * diff;
    }

    return sqrt(total / k) / ideal;
}
```

**含义**:
- `imbalance_factor = 0`: 完美平衡
- `imbalance_factor = 1`: 平均偏差等于理想值
- `imbalance_factor > 1`: 严重不平衡

## 10. 性能分析

### 10.1 时间复杂度

| 阶段 | 复杂度 | 说明 |
|------|--------|------|
| **初始化** | O(k) ~ O(nkd) | Random < AFK-MC² < k-means++ |
| **每次迭代** | O(nkd) | 分配 + 质心更新 |
| **总复杂度** | O(niter * nkd) | 通常是 O(25 * nkd) |

### 10.2 空间复杂度

| 数据结构 | 大小 |
|----------|------|
| **质心** | k * d * 4 bytes |
| **分配** | n * 8 bytes |
| **距离** | n * 4 bytes |
| **临时缓冲** | decode_block_size * d * 4 bytes |

### 10.3 优化技术总结

| 技术 | 实现 | 收益 |
|------|------|------|
| **OpenMP 并行** | `#pragma omp parallel` | 多核加速 |
| **线程局部性** | 分离质心范围 | 无锁竞争 |
| **批处理解码** | decode_block_size | 内存友好 |
| **子采样** | max_points_per_centroid | 减少大 n 的开销 |
| **早停** | obj 检查 | 避免无效迭代 |
| **热启动** | ProgressiveDim | 更好的初始化 |

## 11. 使用示例

### 11.1 基本用法

```cpp
// 创建聚类器
Clustering clus(128, 256);  // d=128, k=256
clus.niter = 25;
clus.verbose = true;

// 使用初始化方法
clus.init_method = ClusteringInitMethod::KMEANS_PLUS_PLUS;

// 创建索引
IndexFlatL2 index(128);

// 训练
clus.train(n, x, index);

// 获取质心
const float* centroids = clus.centroids.data();
```

### 11.2 渐进式维度聚类

```cpp
ProgressiveDimClustering clus(128, 256);
clus.progressive_dim_steps = 10;
clus.apply_pca = true;

IndexFactory factory;
clus.train(n, x, factory);
```

### 11.3 编码向量训练

```cpp
// 训练数据已经量化编码
Clustering clus(128, 256);

ScalarQuantizer codec(128, QuantizerType::QT_8bit);
codec.train(n, x);
codec.compute_codes(x, codes, n);

// 使用编码数据训练
clus.train_encoded(n, codes, &codec, index);
```

## 12. 关键源码位置

| 文件 | 函数 | 行号 |
|------|------|------|
| `Clustering.cpp` | `train_encoded()` | 267-605 |
| `Clustering.cpp` | `compute_centroids()` | 135-204 |
| `Clustering.cpp` | `split_clusters()` | 216-263 |
| `Clustering.cpp` | `subsample_training_set()` | 70-120 |
| `Clustering.cpp` | `ProgressiveDimClustering::train()` | 688-748 |
| `ClusteringInitialization.cpp` | `init_kmeans_plus_plus()` | 192-238 |
| `ClusteringInitialization.cpp` | `init_afkmc2()` | 240-353 |
