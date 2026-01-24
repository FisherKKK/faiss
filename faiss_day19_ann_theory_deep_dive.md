# Faiss深度课程 - 第19天：近似最近邻(ANN)理论深入

## 课程目标

深入理解近似最近邻搜索的理论基础，包括概率保证、空间划分理论和距离界限分析。

---

## 1. ANN理论基础

### 1.1 问题定义

```cpp
// 近似最近邻搜索问题定义
struct ANNSProblem {
    // 数据集
    const float* X;      // n × d 数据集
    size_t n;            // 向量数量
    int d;               // 向量维度

    // 查询集
    const float* Q;      // nq × d 查询集
    size_t nq;           // 查询数量

    // 目标
    // 对每个查询q，找到数据集中(1+ε)-近似最近邻
    // 即：d(q, ANN(q)) ≤ (1+ε) × d(q, NN(q))

    float epsilon;        // 近似因子
    int k;                // k-NN
};
```

### 1.2 严格(c-ANN)定义

```cpp
// 严格近似最近邻定义
struct StrictANN {
    // 给定查询q，返回点p是c-ANN当且仅当：
    // d(q, p) ≤ c × d(q, NN(q))
    // 其中NN(q)是真正的最近邻

    bool is_c_approximate_neighbor(
            const float* q,
            const float* candidate,
            const float* true_nearest,
            float c,
            size_t d) {

        float d_candidate = fvec_L2sqr(q, candidate, d);
        float d_true = fvec_L2sqr(q, true_nearest, d);

        return d_candidate <= c * d_true;
    }
};
```

### 1.3 概率保证

```cpp
// 概率保证：以概率p返回(1+ε)-近似结果
struct ProbabilisticGuarantee {
    // 主定理：对于随机投影等方法
    // Pr[返回(1+ε)-ANN] ≥ 1 - δ

    // Johnson-Lindenstrauss引理
    // 高维向量可以被投影到低维空间，同时保持距离

    float jl_transform_bound(int d_original, int d_projected, float epsilon) {
        // 目标维度
        // d_projected = O(ε^-2 × log(n) × log(1/δ))

        return 4.0f * log(1.0f / epsilon) * log(1.0f / delta);
    }

    // LSH（局部敏感哈希）理论
    float lsh_collision_probability(float distance1, float distance2, float R) {
        // p(d) = Pr[h(v1) = h(v2)] ∝ R^c / d^c

        // 对于E2LSH
        // p(d) = 1 / (d^c × R^c)

        // 碰撞概率随距离单调递减
        return std::pow(R / distance2, 2.0f);  // 简化
    }
};
```

---

## 2. 空间划分理论

### 2.1 KD树分解

```cpp
// KD树空间划分分析
struct KDSpacePartition {
    // KD树递归划分空间
    // 每次选择一个维度，按中位数分割

    struct Cell {
        // 定义超矩形cell
        std::vector<float> low;   // 下界
        std::vector<float> high;  // 上界

        int d;
        Cell(int d) : low(d), high(d), d(d) {}
    };

    // 计算cell的体积
    float volume(const Cell& cell) {
        float v = 1.0f;
        for (int i = 0; i < cell.d; i++) {
            v *= (cell.high[i] - cell.low[i]);
        }
        return v;
    }

    // 高维空间中的"维度灾难"
    void analyze_curse_of_dimensionality() {
        // 在高维空间中：
        // 1. 数据变得稀疏
        // 2. 距离失去区分度
        // 3. KD树退化为线性扫描

        // 测试：边长为R的超立方体
        for (int d = 2; d <= 128; d *= 2) {
            float edge_length = 1.0f;

            // 中心到顶点的距离
            float center_to_corner = std::sqrt(d) * edge_length / 2;

            // 内切球半径
            float inscribed_sphere_radius = edge_length / 2;

            // 球体积 / 立方体体积
            float ratio = pow(M_PI / 6, d / 2.0f);

            printf("d=%3d: center_to_corner=%.3f, ratio=%.6f\n",
                   d, center_to_corner, ratio);
        }
    }
};
```

### 2.2 覆盖半径分析

```cpp
// 覆盖半径：搜索空间的大小
struct CoveringRadiusAnalysis {
    // 对于空间划分方法，每个cell的覆盖半径
    // 影响搜索效率和精度

    // IVF的覆盖半径
    float ivf_covering_radius(
            const std::vector<std::vector<float>>& centroids,
            size_t nlist,
            const std::vector<float>& data_points,
            size_t n) {

        float max_radius = 0;
        int d = centroids[0].size();

        // 对于每个cell
        for (size_t i = 0; i < nlist; i++) {
            float max_dist_in_cell = 0;

            // 找到该cell中离质心最远的点
            for (size_t j = 0; j < n; j++) {
                // 假设我们已经知道每个点属于哪个cell
                if (assign(j) == i) {
                    float dist = fvec_L2sqr(
                        centroids[i].data(),
                        data_points.data() + j * d, d);

                    max_dist_in_cell = std::max(max_dist_in_cell, dist);
                }
            }

            max_radius = std::max(max_radius, max_dist_in_cell);
        }

        return std::sqrt(max_radius);
    }
};
```

### 2.3 VC维分析

```cpp
// Vapnik-Chervonenkis维：衡量函数集复杂度
struct VCAnalysis {
    // 对于ANN算法，VC维影响样本复杂度

    // 定理：要学习一个好的哈希函数
    // 需要的样本数量与VC维成比例

    // LSH的VC维
    int compute_lsh_vc_dimension(int d, int L) {
        // 对于E2LSH
        // VC-dim = O(d × L)

        // 其中：
        // d: 输入维度
        // L: 哈希表数量

        return d * L;
    }

    // PAC学习界
    float pac_bound(int vc_dim, int n, float delta) {
        // 样本复杂度界
        // m(ε, δ) = O((VC_dim + log(1/δ)) / ε^2)

        float epsilon = 0.1f;
        float delta = 0.05f;

        return (vc_dim + std::log(1.0f / delta)) / (epsilon * epsilon);
    }
};
```

---

## 3. 距离分布理论

### 3.1 距离分布模型

```cpp
// 距离分布对ANN性能的影响
struct DistanceDistribution {
    // 分析数据集的距离分布特性

    struct DistributionStats {
        float mean;       // 平均距离
        float variance;   // 方差
        float min;
        float max;
        std::vector<float> percentiles;  // 百分位数
    };

    DistributionStats analyze(
            const float* X,
            size_t n,
            int d,
            size_t sample_size = 10000) {

        DistributionStats stats;
        stats.percentiles.resize(100);

        // 随机采样计算距离
        std::vector<float> distances;

        for (size_t i = 0; i < sample_size; i++) {
            size_t idx1 = rand() % n;
            size_t idx2 = rand() % n;

            if (idx1 != idx2) {
                float dist = std::sqrt(fvec_L2sqr(
                    X + idx1 * d,
                    X + idx2 * d, d));

                distances.push_back(dist);
            }
        }

        // 计算统计量
        std::sort(distances.begin(), distances.end());

        stats.min = distances.front();
        stats.max = distances.back();
        stats.mean = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();

        // 方差
        float variance_sum = 0;
        for (float dist : distances) {
            variance_sum += (dist - stats.mean) * (dist - stats.mean);
        }
        stats.variance = variance_sum / distances.size();

        // 百分位数
        for (int p = 0; p < 100; p++) {
            size_t idx = distances.size() * p / 100;
            stats.percentiles[p] = distances[idx];
        }

        return stats;
    }

    // 距离分布对ANN的影响
    void impact_on_ann_performance(const DistributionStats& stats) {
        printf("=== Distance Distribution Impact ===\n");
        printf("Mean distance: %.2f\n", stats.mean);
        printf("Std dev: %.2f\n", std::sqrt(stats.variance));
        printf("Range: [%.2f, %.2f]\n", stats.min, stats.max);

        // 距离集中度影响：
        // 1. 高方差 → 难以找到好的划分
        // 2. 低方差 → 可能只需要少数几个质心

        float cv = std::sqrt(stats.variance) / stats.mean;  // 变异系数
        printf("Coefficient of variation: %.2f\n", cv);

        if (cv < 0.3) {
            printf("→ Low variance: Consider using fewer centroids\n");
        } else if (cv > 0.7) {
            printf("→ High variance: Consider adaptive nprobe\n");
        }
    }
};
```

### 3.2 距离界限不等式

```cpp
// 距离三角不等式优化
struct DistanceBounds {
    // 利用三角不等式避免距离计算

    // 下界过滤
    float lower_bound(
            const float* q,
            const float* centroid,
            const float* min_dist_to_centroid,
            int d) {

        // d(q, p) ≥ |d(q, c) - d(c, p)|

        float d_qc = std::sqrt(fvec_L2sqr(q, centroid, d));
        float d_cp_min = min_dist_to_centroid;

        return std::abs(d_qc - d_cp_min);
    }

    // 上界过滤
    float upper_bound(
            const float* q,
            const float* centroid,
            const float* max_dist_to_centroid,
            int d) {

        // d(q, p) ≤ d(q, c) + d(c, p)

        float d_qc = std::sqrt(fvec_L2sqr(q, centroid, d));
        float d_cp_max = max_dist_to_centroid;

        return d_qc + d_cp_max;
    }

    // 在IVF搜索中应用
    void ivf_search_with_bounds(
            IndexIVF* index,
            const float* q,
            float threshold) {

        size_t nlist = index->nlist;
        float* centroid_dis = new float[nlist];
        idx_t* list_ids = new idx_t[nlist];

        // 1. 计算到所有质心的距离
        index->quantizer->search(1, q, nlist, centroid_dis, list_ids);

        // 2. 对每个list，计算距离下界
        for (size_t i = 0; i < nlist; i++) {
            idx_t list_id = list_ids[i];
            float d_qc = centroid_dis[i];

            // 该list中点的最小距离估计
            float min_dist_in_list = get_min_distance_to_list(index, list_id);

            // 下界
            float lower = std::abs(d_qc - min_dist_in_list);

            if (lower > threshold) {
                // 这个list中所有点都太远，跳过
                continue;
            }

            // 搜索这个list
            search_list(index, list_id, q, threshold);
        }

        delete[] centroid_dis;
        delete[] list_ids;
    }
};
```

---

## 4. 概率数据结构

### 4.1 SkipList分析

```cpp
// 跳表（用于NSG等图索引）
struct SkipListAnalysis {
    // 跳表的概率性能保证

    // 查找复杂度分析
    float expected_search_steps(int n, int max_level) {
        // 对于n个元素，max_level层跳表
        // 期望查找步数：O(log n)

        // 每层跳跃的概率
        float p = 0.5;  // 提升到上一层的概率

        // 期望层数
        float expected_levels = 1.0f / (1.0f - p);

        // 每层期望步数
        float steps_per_level = 1.0f / p;

        return expected_levels * steps_per_level;
    }

    // 实际跳跃距离
    int expected_skip_distance(int level, int max_level) {
        // 第level层的跳跃间隔

        // 几何分布：P(distance = k) = p^(k-1) × (1-p)

        float p = 0.5f;  // 默认概率

        // 期望距离
        return 1.0f / (1.0f - p);
    }
};
```

### 4.2 Locality Sensitive Hashing理论

```cpp
// LSH理论基础
struct LSHTheory {
    // E2LSH: Euclidean Distance Squared LSH

    // 家族：L_k(d)族哈希函数
    // h(v) = ⌊(a·v + b) / w⌋

    struct E2LSHFamily {
        float a;  // 随机投影向量
        float b;  // 随机偏移 [0, w)
        float w;  // 宽度参数

        // 投影
        float project(const float* v, int d) {
            // 计算a·v（点积）
            float dot_product = 0;
            for (int i = 0; i < d; i++) {
                dot_product += a * v[i];
            }

            // 加上偏移并量化
            return std::floor((dot_product + b) / w);
        }

        // 碰撞概率
        float collision_probability(float d1, float d2, float w) {
            // p(d) = Pr[h(p1) = h(p2)]

            // 对于E2LSH：
            // p(d) = 1 - 2 × Z(d/w) - (w/πd) × (1 - Z(d/w))

            // 其中Z(x) = CDF[标准正态](x)

            float normalized_d = d1 / w;

            // 近似计算
            return 1.0f - 2.0f * standard_normal_cdf(normalized_d);
        }

    private:
        float standard_normal_cdf(float x) {
            // 标准正态CDF的近似
            static const float a1 =  0.254829592;
            static const float a2 = -0.284496736;
            static const float a3 =  1.421413741;
            static const float a4 = -1.453152027;
            static const float a5 =  1.061405429;
            static const float p  =  0.3275911;

            int sign = (x < 0) ? -1 : 1;
            x = std::abs(x) / std::sqrt(2);

            float t = 1.0f / (1.0f + p * x);
            float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
            y = sign * std::sqrt(std::max(0.0f, 1.0f - y));

            return 0.5f * (1.0f + y);
        }
    };
```

### 4.3 LSH参数优化

```cpp
// LSH参数理论分析
struct LSHParameterOptimization {
    // 给定n, d, ε, δ，确定最优参数

    struct OptimalParameters {
        int L;          // 哈希表数量
        int K;          // 每个表的哈希函数数量
        float w;        // 宽度参数
    };

    OptimalParameters compute_optimal_parameters(
            size_t n,           // 数据集大小
            int d,               // 维度
            float epsilon,       // 近似因子
            float delta,         // 失败概率
            float R) {           // 搜索半径

        OptimalParameters params;

        // 根据LSH理论
        // L = O(n^ρ × log n)
        // K = O(log n)
        // ρ = 1/(1+ε)

        float rho = 1.0f / (1.0f + epsilon);

        // 表数量
        params.L = (int)std::pow(n, rho) * std::log(n);

        // 每表哈希函数数
        params.K = (int)(std::log(n) / std::log(1.0f / delta));

        // 宽度参数
        // w 应该与数据分布相关
        params.w = estimate_optimal_width(R, epsilon);

        return params;
    }

    float estimate_optimal_width(float R, float epsilon) {
        // w ≈ R × (1 + ε)

        return R * (1.0f + epsilon) / 2.0f;
    }
};
```

---

## 5. 理论性能界

### 5.1 查询复杂度界

```cpp
// 不同ANN算法的理论复杂度
struct ComplexityBounds {

    void print_complexity_bounds() {
        printf("=== ANN Algorithm Complexity Bounds ===\n\n");

        // 1. 空间划分方法
        printf("Space Partitioning Methods:\n");
        printf("  KD-tree:\n");
        printf("    Build: O(n log n)\n");
        printf("    Query: O(n^(1-1/d)) [low-d], O(n) [high-d]\n");
        printf("    Memory: O(n)\n\n");

        printf("  Ball-tree:\n");
        printf("    Build: O(n log n)\n");
        printf("    Query: O(log n)\n\n");

        // 2. LSH方法
        printf("LSH Methods:\n");
        printf("    Build: O(n^ρ L)\n");
        printf("    Query: O(L × n^ρ)\n");
        printf("    where ρ = 1/(1+ε)\n\n");

        // 3. 图方法
        printf("Graph Methods:\n");
        printf("  NSG:\n");
        printf("    Build: O(n × k × log n)\n");
        printf("    Query: O(log^2 n)\n\n");

        printf("  HNSW:\n");
        printf("    Build: O(n log n × M)\n");
        printf("    Query: O(log n × M)\n\n");

        // 4. 量化方法
        printf("Quantization Methods:\n");
        printf("  PQ/OPQ:\n");
        printf("    Build: O(n × d × M × K)\n");
        printf("    Query: O(d × M + n × K)\n\n");

        printf("  IVF+PQ:\n");
        printf("    Build: O(n × d × M × K + nlist × d × K)\n");
        printf("    Query: O(nprobe × d × M + nprobe × K)\n\n");
    }
};
```

### 5.2 内存-速度权衡

```cpp
// 内存与查询速度的理论权衡
struct MemorySpeedTradeoff {
    // 对于固定的数据集和精度

    struct TradeoffPoint {
        size_t memory_mb;
        double qps;
        float recall;
    };

    std::vector<TradeoffPoint> analyze_pareto_frontier(
            const float* xb,
            size_t n,
            int d,
            const float* xq,
            size_t nq) {

        std::vector<TradeoffPoint> frontier;

        // 测试不同配置
        std::vector<std::pair<std::string, std::function<void()>>> configs = {
            {"Flat", [&]() {
                // 精确搜索，高内存
            }},
            {"IVF,Flat,nlist=100", [&]() {
                // 中等内存，中等速度
            }},
            {"IVF,PQ,nlist=100,M=32", [&]() {
                // 低内存，快速
            }},
            {"HNSW,M=16", [&]() {
                // 高精度，中等内存
            }},
        };

        for (auto& [name, config_fn] : configs) {
            // 创建索引并测试
            TradeoffPoint point;
            test_configuration(xb, n, d, xq, nq, name, point);
            frontier.push_back(point);
        }

        return frontier;
    }

    void print_pareto_frontier(const std::vector<TradeoffPoint>& frontier) {
        printf("\n=== Pareto Frontier ===\n");
        printf("%-20s %15s %10s %10s\n",
               "Method", "Memory(MB)", "QPS", "Recall");
        printf("%-20s %15s %10s %10s\n",
               "--------------------", "---------------",
               "----------", "----------");

        for (const auto& p : frontier) {
            printf("%-20s %15.2f %10.0f %10.3f%%\n",
                   "", p.memory_mb, p.qps, p.recall * 100);
        }
    }
};
```

### 5.3 渐进优化界

```cpp
// 渐进优化算法的理论保证
struct ProgressiveOptimization {
    // 从粗到细的搜索策略

    // 渐进式Refinement
    float progressive_refinement(
            IndexIVF* index,
            const float* q,
            float epsilon) {

        // 从小的nprobe开始，逐步增加
        std::vector<float> results_history;

        for (int nprobe = 1; nprobe <= index->nlist; nprobe *= 2) {
            index->nprobe = nprobe;

            // 搜索
            float distances[k];
            idx_t labels[k];
            index->search(1, q, k, distances, labels);

            // 记录结果
            float score = distances[0];
            results_history.push_back(score);

            // 检查收敛
            if (results_history.size() >= 2) {
                float improvement = std::abs(
                    results_history[results_history.size() - 1] -
                    results_history[results_history.size() - 2]);

                if (improvement < epsilon) {
                    break;  // 收敛
                }
            }
        }

        return results_history.back();
    }
};
```

---

## 6. 理论保证验证

### 6.1 验证(1+ε)-ANN性质

```cpp
// 验证算法是否满足(1+ε)-ANN保证
struct ANNValidator {

    bool validate_c_approx(
            Index* index,
            const float* queries,
            size_t nq,
            int k,
            float c,
            const float* ground_truth_neighbors) {

        int correct = 0;
        int total = nq * k;

        // 对每个查询
        for (size_t q = 0; q < nq; q++) {
            // 搜索
            float distances[k];
            idx_t labels[k];
            index->search(1, queries + q * index->d, k,
                           distances, labels);

            // 检查是否满足c-近似
            for (int i = 0; i < k; i++) {
                float true_dist = ground_truth_distances[q * k + i];
                float approx_dist = distances[i];

                if (approx_dist <= c * true_dist) {
                    correct++;
                }
            }
        }

        float accuracy = (float)correct / total;
        printf("c-Approximation Accuracy: %.3f%% (c=%.2f)\n",
               accuracy * 100, c);

        return accuracy >= 0.95;  // 95%以上满足
    }

    // 验证概率保证
    bool validate_probabilistic_guarantee(
            Index* index,
            const float* queries,
            size_t nq,
            float epsilon,
            float delta) {

        // 多次运行，统计失败率
        int failures = 0;
        int trials = 100;

        for (int t = 0; t < trials; t++) {
            // 随机选择查询
            size_t q = rand() % nq;

            // 搜索
            float distances[k];
            idx_t labels[k];
            index->search(1, queries + q * index->d, k,
                           distances, labels);

            // 验证是否为(1+ε)-ANN
            if (!is_epsilon_approx(queries + q * index->d,
                                  labels[0], epsilon)) {
                failures++;
            }
        }

        float failure_rate = (float)failures / trials;
        printf("Failure rate: %.3f%% (target δ=%.3f)\n",
               failure_rate * 100, delta);

        return failure_rate <= delta;
    }

private:
    bool is_epsilon_approx(
            const float* query,
            idx_t candidate_id,
            float epsilon) {

        // 获取真正的最近邻距离
        float true_dist = compute_true_nearest_distance(query);
        float approx_dist = get_distance(query, candidate_id);

        return approx_dist <= (1 + epsilon) * true_dist;
    }
};
```

### 6.2 召回率与精度权衡曲线

```cpp
// 绘制召回率-精度权衡曲线
struct RecallPrecisionCurve {

    void analyze_curve(
            Index* index,
            const float* queries,
            size_t nq,
            const idx_t* ground_truth,
            int k_gt) {

        printf("\n=== Recall-Precision Trade-off Curve ===\n\n");

        // 测试不同参数配置
        struct Config {
            std::string name;
            std::function<void()> set_params;
        };

        std::vector<Config> configs;

        if (auto* ivf = dynamic_cast<IndexIVF*>(index)) {
            for (int nprobe : {1, 5, 10, 20, 50, 100}) {
                configs.push_back({
                    "nprobe=" + std::to_string(nprobe),
                    [ivf, nprobe]() {
                        ivf->nprobe = nprobe;
                    }
                });
            }
        }

        if (auto* hnsw = dynamic_cast<IndexHNSW*>(index)) {
            for (int efSearch : {10, 20, 40, 80, 160}) {
                configs.push_back({
                    "efSearch=" + std::to_string(efSearch),
                    [hnsw, efSearch]() {
                        hnsw->hnsw.efSearch = efSearch;
                    }
                });
            }
        }

        // 测试每个配置
        for (const auto& config : configs) {
            config.set_params();

            // 搜索
            float distances[nq * k];
            idx_t labels[nq * k];
            index->search(nq, queries, k, distances, labels);

            // 计算召回率
            float recall = compute_recall_at_k(
                nq, k, labels, ground_truth, k_gt);

            // 计算精度（假设ground truth是top-k_gt）
            float precision = compute_precision_at_k(
                nq, k, labels, ground_truth, k_gt);

            printf("%-20s: Recall@10=%.3f%%, Precision@10=%.3f%%\n",
                   config.name.c_str(), recall * 100, precision * 100);
        }
    }

    float compute_recall_at_k(
            size_t nq, int k,
            const idx_t* labels,
            const idx_t* ground_truth,
            int k_gt) {

        int correct = 0;
        for (size_t q = 0; q < nq; q++) {
            std::set<idx_t> gt_set(
                ground_truth + q * k_gt,
                ground_truth + q * k_gt + k_gt);

            for (int i = 0; i < k; i++) {
                if (gt_set.count(labels[q * k + i])) {
                    correct++;
                }
            }
        }

        return (float)correct / (nq * k);
    }
};
```

---

## 7. 第19天总结

### 核心理论概念

1. **(1+ε)-ANN定义**：近似精度的数学定义
2. **空间划分理论**：高维空间的维度灾难
3. **概率保证**：PAC学习框架
4. **LSH理论**：局部敏感哈希的碰撞概率
5. **复杂度界**：不同算法的查询/构建复杂度

### 理论工具

1. **Johnson-Lindenstrauss引理**：降维理论
2. **VC维**：函数集复杂度
3. **距离界限**：三角不等式优化
4. **Pareto最优**：内存-速度权衡

### 下一步

第20天将学习**距离度量与学习**，探索如何学习最优的距离度量。

---

## 练习题

1. 实现LSH哈希函数族
2. 分析高维空间中的维度灾难现象
3. 验证HNSW的理论保证
4. 绘制不同算法的Pareto前沿

## 扩展阅读

- [LSH论文](https://www.cs.princeton.edu/courses/archive/fall06/cos598/papers/lsh.pdf)
- [Approximate Nearest Neighbor Survey](https://www.cs.umd.edu/~mount/Papers/heracl2009_a.pdf)
- [Johnson-Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)
