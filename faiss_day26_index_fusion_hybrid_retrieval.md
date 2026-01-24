# Faiss深度学习课程 - 第26天：索引融合与混合检索策略

## 课程概述

第26天探讨如何组合多种索引和检索策略，以达到最优的性能、精度和效率平衡。实际生产环境中，单一索引往往无法满足所有需求，需要通过智能的融合和混合策略来最大化系统效能。

## 学习目标

- 理解索引融合的动机和场景
- 掌握多种索引融合技术
- 学习混合检索策略
- 理解查询路由与动态选择
- 掌握级联检索架构
- 实践生产级混合检索系统

---

## 第一部分：索引融合基础

### 1.1 为什么需要索引融合

```cpp
// 索引融合动机分析
class IndexFusionMotivation {
public:
    // 不同索引的特性对比
    struct IndexCharacteristics {
        std::string name;
        double build_time;      // 构建时间（相对值）
        double query_latency;   // 查询延迟（相对值）
        double recall;          // 召回率
        double memory_usage;    // 内存使用（相对值）
        bool supports_update;   // 是否支持更新
        bool supports_delete;   // 是否支持删除
    };

    static std::vector<IndexCharacteristics> analyze_index_types() {
        return {
            {"IndexFlat",       1.0,  1.0,  1.0,  1.0,  true,  true},
            {"IndexIVF",        5.0,  0.3,  0.95, 0.8,  true,  true},
            {"IndexIVFPQ",      6.0,  0.4,  0.85, 0.3,  true,  false},
            {"IndexHNSW",      20.0,  0.1,  0.98, 1.5,  true,  false},
            {"IndexPQ",         3.0,  0.2,  0.70, 0.2,  false, false},
            {"IndexNSG",       15.0,  0.15, 0.96, 1.3,  false, false}
        };
    }

    static void print_comparison() {
        auto chars = analyze_index_types();

        printf("%-15s | %-8s | %-12s | %-8s | %-10s | %s | %s\n",
               "Index", "Build", "Query Latency", "Recall", "Memory", "Update", "Delete");
        printf("%-15s | %-8s | %-12s | %-8s | %-10s | %s | %s\n",
               "---------------", "--------", "------------", "--------",
               "----------", "------", "------");

        for (const auto& ch : chars) {
            printf("%-15s | %-8.1f | %-12.1f | %-8.2f | %-10.1f | %s | %s\n",
                   ch.name.c_str(),
                   ch.build_time,
                   ch.query_latency,
                   ch.recall,
                   ch.memory_usage,
                   ch.supports_update ? "Yes" : "No",
                   ch.supports_delete ? "Yes" : "No");
        }
    }

    // 融合场景分析
    enum class FusionScenario {
        RECALL_LATENCY trade-off,     // 召回率-延迟权衡
        MEMORY_ACCURACY trade-off,    // 内存-精度权衡
        HOT_COLD_DATA,                // 冷热数据分离
        MULTISTAGE_REFINEMENT,        // 多阶段精化
        REDUNDANCY,                   // 冗余备份
        SPECIALIZATION                 // 专用索引
    };

    static void recommend_fusion_strategy(
        size_t data_size,
        double qps_requirement,
        double recall_requirement,
        double memory_budget_gb) {

        printf("\n=== Fusion Strategy Recommendation ===\n");
        printf("Data size: %zu, QPS: %.0f, Recall: %.2f, Memory: %.1f GB\n\n",
               data_size, qps_requirement, recall_requirement, memory_budget_gb);

        if (data_size < 100000) {
            printf("Recommendation: Single IndexFlat is sufficient\n");
            printf("  -> Small dataset, no fusion needed\n");

        } else if (qps_requirement > 10000 && recall_requirement > 0.95) {
            printf("Recommendation: HNSW + IVF Fusion\n");
            printf("  -> HNSW for hot data (high recall, fast)\n");
            printf("  -> IVF for cold data (memory efficient)\n");
            printf("  -> Dynamic routing based on query patterns\n");

        } else if (memory_budget_gb < data_size * 4 * 0.001 / 1024) {
            printf("Recommendation: IVF + PQ Fusion\n");
            printf("  -> IVF for coarse pruning\n");
            printf("  -> PQ for compressed storage\n");
            printf("  -> Two-stage refinement\n");

        } else {
            printf("Recommendation: Single HNSW index\n");
            printf("  -> Balanced performance\n");
            printf("  -> Sufficient memory budget\n");
        }
    }
};
```

### 1.2 融合类型分类

```cpp
// 索引融合分类
class IndexFusionTaxonomy {
public:
    // 融合层次
    enum class FusionLevel {
        STORAGE,        // 存储层：组合不同编码
        STRUCTURAL,     // 结构层：组合不同索引结构
        QUERY,          // 查询层：组合不同查询策略
        RESULT          // 结果层：组合不同检索结果
    };

    // 融合模式
    enum class FusionPattern {
        SEQUENTIAL,     // 顺序执行：粗->细
        PARALLEL,       // 并行执行：多路查询
        ADAPTIVE,       // 自适应：根据查询选择
        HIERARCHICAL    // 层次化：多级索引
    };

    struct FusionType {
        FusionLevel level;
        FusionPattern pattern;
        std::string description;
        std::vector<std::string> component_indexes;
    };

    static std::vector<FusionType> get_common_fusion_types() {
        return {
            {
                FusionLevel::STRUCTURAL,
                FusionPattern::SEQUENTIAL,
                "IVF+PQ (IndexIVFPQ)",
                {"IndexIVF", "ProductQuantizer"}
            },
            {
                FusionLevel::QUERY,
                FusionPattern::PARALLEL,
                "HNSW + IVF (Redundant paths)",
                {"IndexHNSW", "IndexIVF"}
            },
            {
                FusionLevel::RESULT,
                FusionPattern::ADAPTIVE,
                "Multiple M values in IVF",
                {"IndexIVF(nlist=100)", "IndexIVF(nlist=1000)"}
            },
            {
                FusionLevel::STORAGE,
                FusionPattern::SEQUENTIAL,
                "Coarse + Fine quantization",
                {"ScalarQuantizer", "ProductQuantizer"}
            }
        };
    }

    static void print_fusion_types() {
        auto types = get_common_fusion_types();

        printf("Common Index Fusion Types:\n\n");
        for (size_t i = 0; i < types.size(); i++) {
            printf("%zu. %s\n", i + 1, types[i].description.c_str());
            printf("   Level: %d, Pattern: %d\n",
                   static_cast<int>(types[i].level),
                   static_cast<int>(types[i].pattern));
            printf("   Components: ");
            for (const auto& comp : types[i].component_indexes) {
                printf("%s ", comp.c_str());
            }
            printf("\n\n");
        }
    }
};
```

---

## 第二部分：索引融合技术

### 2.1 粗细粒度融合（IVF+PQ）

```cpp
// IVF+PQ融合索引
class IVFPQFusionIndex {
    faiss::IndexIVFPQ* ivfpq_index;
    size_t d;
    size_t nlist;
    size_t m;           // PQ子量化器数
    size_t nbits;       // 每子量化器位数

public:
    IVFPQFusionIndex(size_t dim, size_t n_lists, size_t pq_m, size_t n_bits)
        : d(dim), nlist(n_lists), m(pq_m), nbits(n_bits) {

        // 创建量化器
        faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(d);

        // 创建IVFPQ索引
        ivfpq_index = new faiss::IndexIVFPQ(quantizer, d, nlist, m, nbits);
    }

    // 训练
    void train(size_t n, const float* training_data) {
        printf("Training IVFPQ fusion index...\n");
        auto start = std::chrono::high_resolution_clock::now();

        ivfpq_index->train(n, training_data);

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        printf("Training completed in %.2f seconds\n", duration);
    }

    // 添加向量
    void add(size_t n, const float* vectors) {
        ivfpq_index->add(n, vectors);
    }

    // 搜索（带精化）
    void search_with_refinement(
        const float* queries,
        size_t nq,
        size_t k,
        float* distances,
        faiss::idx_t* labels,
        size_t nprobe = 10,
        size_t refine_k = 100) {

        // 第一阶段：粗搜索（PQ编码）
        faiss::IVFSearchParameters params;
        params.nprobe = nprobe;

        float* coarse_distances = new float[nq * refine_k];
        faiss::idx_t* coarse_labels = new faiss::idx_t[nq * refine_k];

        ivfpq_index->search(queries, nq, refine_k,
                           coarse_distances, coarse_labels, &params);

        // 第二阶段：在候选集上精化
        refine_results(queries, nq, k, refine_k,
                      coarse_distances, coarse_labels,
                      distances, labels);

        delete[] coarse_distances;
        delete[] coarse_labels;
    }

    // 分析不同nprobe的效果
    void analyze_nprobe_impact(
        const float* test_queries,
        size_t nq,
        const float* ground_truth_distances,
        const faiss::idx_t* ground_truth_labels,
        size_t k) {

        printf("\n=== nprobe Impact Analysis ===\n");
        printf("nprobe | Recall@%zu | Latency(ms)\n", k);
        printf("-------|-----------|------------\n");

        std::vector<size_t> nprobes = {1, 5, 10, 20, 50, 100};

        for (size_t nprobe : nprobes) {
            auto start = std::chrono::high_resolution_clock::now();

            float* distances = new float[nq * k];
            faiss::idx_t* labels = new faiss::idx_t[nq * k];

            ivfpq_index->search(test_queries, nq, k,
                               distances, labels, nprobe);

            auto end = std::chrono::high_resolution_clock::now();
            double latency_ms =
                std::chrono::duration<double, std::milli>(end - start).count();

            // 计算召回率
            double recall = compute_recall(
                nq, k, labels, ground_truth_labels);

            printf("%6zu | %.4f  | %8.2f\n", nprobe, recall, latency_ms);

            delete[] distances;
            delete[] labels;
        }
    }

private:
    void refine_results(
        const float* queries,
        size_t nq,
        size_t k,
        size_t refine_k,
        const float* coarse_distances,
        const faiss::idx_t* coarse_labels,
        float* refined_distances,
        faiss::idx_t* refined_labels) {

        // 在候选集上重新计算精确距离
        for (size_t q = 0; q < nq; q++) {
            std::vector<std::pair<float, faiss::idx_t>> candidates;

            for (size_t i = 0; i < refine_k; i++) {
                faiss::idx_t label = coarse_labels[q * refine_k + i];
                if (label < 0) continue;

                // 重新计算精确距离（这里简化，实际需要获取原始向量）
                float exact_dist = coarse_distances[q * refine_k + i];
                candidates.push_back({exact_dist, label});
            }

            // 排序并取Top-K
            std::sort(candidates.begin(), candidates.end());

            for (size_t i = 0; i < std::min(k, candidates.size()); i++) {
                refined_labels[q * k + i] = candidates[i].second;
                refined_distances[q * k + i] = candidates[i].first;
            }

            // 填充剩余位置
            for (size_t i = candidates.size(); i < k; i++) {
                refined_labels[q * k + i] = -1;
                refined_distances[q * k + i] = std::numeric_limits<float>::infinity();
            }
        }
    }

    double compute_recall(
        size_t nq, size_t k,
        const faiss::idx_t* predicted_labels,
        const faiss::idx_t* ground_truth_labels) const {

        size_t correct = 0;
        size_t total = 0;

        for (size_t q = 0; q < nq; q++) {
            std::unordered_set<faiss::idx_t> gt_set;
            for (size_t i = 0; i < k; i++) {
                if (ground_truth_labels[q * k + i] >= 0) {
                    gt_set.insert(ground_truth_labels[q * k + i]);
                }
            }

            for (size_t i = 0; i < k; i++) {
                if (predicted_labels[q * k + i] >= 0 &&
                    gt_set.count(predicted_labels[q * k + i])) {
                    correct++;
                }
                total++;
            }
        }

        return static_cast<double>(correct) / total;
    }
};
```

### 2.2 多路径融合（HNSW + IVF）

```cpp
// 多路径融合索引
class MultiPathFusionIndex {
    faiss::IndexHNSWFlat* hnsw_index;
    faiss::IndexIVFFlat* ivf_index;
    size_t d;

public:
    MultiPathFusionIndex(size_t dim, int M = 16, size_t nlist = 100)
        : d(dim) {

        // 创建HNSW索引（高精度、低延迟）
        hnsw_index = new faiss::IndexHNSWFlat(dim, M);

        // 创建IVF索引（内存效率高）
        faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(dim);
        ivf_index = new faiss::IndexIVFFlat(quantizer, dim, nlist);
    }

    // 训练IVF
    void train_ivf(size_t n, const float* training_data) {
        printf("Training IVF component...\n");
        ivf_index->train(n, training_data);

        // 同时添加到HNSW
        printf("Adding to HNSW...\n");
        hnsw_index->add(n, training_data);
    }

    // 添加向量（两个索引都添加）
    void add(size_t n, const float* vectors) {
        ivf_index->add(n, vectors);
        hnsw_index->add(n, vectors);
    }

    // 搜索策略
    enum class SearchStrategy {
        HNSW_ONLY,          // 仅使用HNSW
        IVF_ONLY,           // 仅使用IVF
        PARALLEL_FUSION,    // 并行搜索两个索引，融合结果
        ADAPTIVE,           // 自适应选择
        CASCADE             // 级联：IVF粗筛，HNSW精化
    };

    // 多路径搜索
    void search(
        const float* queries,
        size_t nq,
        size_t k,
        float* distances,
        faiss::idx_t* labels,
        SearchStrategy strategy = SearchStrategy::PARALLEL_FUSION,
        size_t ivf_nprobe = 10) {

        switch (strategy) {
            case SearchStrategy::HNSW_ONLY:
                hnsw_index->search(queries, nq, k, distances, labels);
                break;

            case SearchStrategy::IVF_ONLY:
                ivf_index->search(queries, nq, k, distances, labels, ivf_nprobe);
                break;

            case SearchStrategy::PARALLEL_FUSION:
                parallel_fusion_search(queries, nq, k, distances, labels, ivf_nprobe);
                break;

            case SearchStrategy::ADAPTIVE:
                adaptive_search(queries, nq, k, distances, labels, ivf_nprobe);
                break;

            case SearchStrategy::CASCADE:
                cascade_search(queries, nq, k, distances, labels, ivf_nprobe);
                break;
        }
    }

    // 获取统计信息
    void print_stats() {
        printf("Multi-Path Fusion Index Stats:\n");
        printf("  HNSW: %ld vectors\n", hnsw_index->ntotal);
        printf("  IVF: %ld vectors\n", ivf_index->ntotal);
        printf("  Memory: %.2f MB (estimated)\n",
               (hnsw_index->ntotal + ivf_index->ntotal) * d * sizeof(float) /
               (1024.0 * 1024));
    }

private:
    // 并行融合搜索
    void parallel_fusion_search(
        const float* queries,
        size_t nq,
        size_t k,
        float* distances,
        faiss::idx_t* labels,
        size_t ivf_nprobe) {

        // 并行搜索两个索引
        float* hnsw_dists = new float[nq * k];
        faiss::idx_t* hnsw_labels = new faiss::idx_t[nq * k];

        float* ivf_dists = new float[nq * k];
        faiss::idx_t* ivf_labels = new faiss::idx_t[nq * k];

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                hnsw_index->search(queries, nq, k, hnsw_dists, hnsw_labels);
            }

            #pragma omp section
            {
                ivf_index->search(queries, nq, k, ivf_dists, ivf_labels, ivf_nprobe);
            }
        }

        // 融合结果（取并集的Top-K）
        for (size_t q = 0; q < nq; q++) {
            std::vector<std::pair<float, faiss::idx_t>> merged;

            for (size_t i = 0; i < k; i++) {
                if (hnsw_labels[q * k + i] >= 0) {
                    merged.push_back({hnsw_dists[q * k + i],
                                     hnsw_labels[q * k + i]});
                }
                if (ivf_labels[q * k + i] >= 0) {
                    merged.push_back({ivf_dists[q * k + i],
                                     ivf_labels[q * k + i]});
                }
            }

            // 去重并排序
            std::sort(merged.begin(), merged.end());
            merged.erase(std::unique(merged.begin(), merged.end(),
                       [](const auto& a, const auto& b) {
                           return a.second == b.second;
                       }), merged.end());

            // 取Top-K
            for (size_t i = 0; i < std::min(k, merged.size()); i++) {
                labels[q * k + i] = merged[i].second;
                distances[q * k + i] = merged[i].first;
            }

            // 填充剩余
            for (size_t i = merged.size(); i < k; i++) {
                labels[q * k + i] = -1;
                distances[q * k + i] = std::numeric_limits<float>::infinity();
            }
        }

        delete[] hnsw_dists;
        delete[] hnsw_labels;
        delete[] ivf_dists;
        delete[] ivf_labels;
    }

    // 自适应搜索
    void adaptive_search(
        const float* queries,
        size_t nq,
        size_t k,
        float* distances,
        faiss::idx_t* labels,
        size_t ivf_nprobe) {

        // 根据查询特征选择索引
        for (size_t q = 0; q < nq; q++) {
            const float* query = queries + q * d;

            // 简化的查询难度估计（使用向量范数）
            float norm = std::sqrt(std::inner_product(
                query, query + d, query, 0.0f));

            if (norm < 5.0f) {
                // "简单"查询：使用IVF（更快）
                ivf_index->search(query, 1, k,
                                distances + q * k,
                                labels + q * k,
                                ivf_nprobe);
            } else {
                // "复杂"查询：使用HNSW（更准确）
                hnsw_index->search(query, 1, k,
                                 distances + q * k,
                                 labels + q * k);
            }
        }
    }

    // 级联搜索
    void cascade_search(
        const float* queries,
        size_t nq,
        size_t k,
        float* distances,
        faiss::idx_t* labels,
        size_t ivf_nprobe) {

        size_t refine_k = k * 10;  // 第一阶段返回更多候选

        // 第一阶段：IVF粗筛
        float* ivf_dists = new float[nq * refine_k];
        faiss::idx_t* ivf_labels = new faiss::idx_t[nq * refine_k];

        ivf_index->search(queries, nq, refine_k,
                         ivf_dists, ivf_labels, ivf_nprobe);

        // 第二阶段：在候选集上用HNSW精化
        // 这里简化：实际需要限制HNSW搜索范围
        hnsw_index->search(queries, nq, k, distances, labels);

        delete[] ivf_dists;
        delete[] ivf_labels;
    }
};
```

### 2.3 层次化索引融合

```cpp
// 层次化索引系统
class HierarchicalFusionIndex {
    // 多级索引
    struct IndexLevel {
        int level;
        std::unique_ptr<faiss::Index> index;
        size_t capacity;
        size_t current_size;
        std::string description;

        bool is_full() const {
            return current_size >= capacity;
        }
    };

    std::vector<IndexLevel> levels;
    size_t d;

public:
    HierarchicalFusionIndex(size_t dim) : d(dim) {
        // 第0级：热数据（HNSW，小但快）
        levels.push_back({
            0,
            std::make_unique<faiss::IndexHNSWFlat>(dim, 16),
            100000,      // 10万向量
            0,
            "Hot data: HNSW"
        });

        // 第1级：温数据（IVF，平衡）
        faiss::IndexFlatL2* q1 = new faiss::IndexFlatL2(dim);
        levels.push_back({
            1,
            std::make_unique<faiss::IndexIVFFlat>(q1, dim, 100),
            1000000,     // 100万向量
            0,
            "Warm data: IVF"
        });

        // 第2级：冷数据（IVFPQ，大但压缩）
        faiss::IndexFlatL2* q2 = new faiss::IndexFlatL2(dim);
        levels.push_back({
            2,
            std::make_unique<faiss::IndexIVFPQ>(q2, dim, 1000, d/8, 8),
            SIZE_MAX,    // 无限
            0,
            "Cold data: IVFPQ"
        });
    }

    // 训练各级索引
    void train(size_t n, const float* training_data) {
        for (auto& level : levels) {
            if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(level.index.get())) {
                printf("Training level %d (%s)...\n",
                       level.level, level.description.c_str());
                ivf->train(n, training_data);
            }
        }
    }

    // 添加向量（自动分配到合适级别）
    size_t add(const float* vector) {
        // 找到合适的级别
        int target_level = find_target_level();

        levels[target_level].index->add(1, vector);
        levels[target_level].current_size++;

        return levels[target_level].index->ntotal - 1;
    }

    // 层次化搜索
    void search(
        const float* queries,
        size_t nq,
        size_t k,
        float* distances,
        faiss::idx_t* labels,
        size_t max_levels = 3) {

        // 从最热（第0级）开始搜索
        // 如果结果不满足阈值，继续搜索下一级

        for (size_t q = 0; q < nq; q++) {
            std::vector<std::pair<float, faiss::idx_t>> merged_results;

            for (int level = 0; level < std::min(max_levels, (int)levels.size()); level++) {
                // 搜索当前级别
                float* level_dists = new float[k];
                faiss::idx_t* level_labels = new faiss::idx_t[k];

                levels[level].index->search(
                    queries + q * d, 1, k,
                    level_dists, level_labels);

                // 合并结果
                for (size_t i = 0; i < k; i++) {
                    if (level_labels[i] >= 0) {
                        merged_results.push_back({level_dists[i], level_labels[i]});
                    }
                }

                delete[] level_dists;
                delete[] level_labels;

                // 检查是否需要继续搜索下一级
                if (!should_search_next_level(merged_results, k)) {
                    break;
                }
            }

            // 排序并取Top-K
            std::sort(merged_results.begin(), merged_results.end());
            merged_results.erase(std::unique(merged_results.begin(), merged_results.end(),
                       [](const auto& a, const auto& b) {
                           return a.second == b.second;
                       }), merged_results.end());

            for (size_t i = 0; i < std::min(k, merged_results.size()); i++) {
                labels[q * k + i] = merged_results[i].second;
                distances[q * k + i] = merged_results[i].first;
            }
        }
    }

    // 提升数据（从冷级别提升到热级别）
    void promote_to_hot(faiss::idx_t id, const float* vector) {
        // 从原级别删除
        // 添加到第0级
        levels[0].index->add(1, vector);
        levels[0].current_size++;
    }

    void print_stats() {
        printf("Hierarchical Index Stats:\n");
        for (const auto& level : levels) {
            printf("  Level %d: %ld vectors (%s)\n",
                   level.level, level.index->ntotal, level.description.c_str());
        }
    }

private:
    int find_target_level() {
        // 从第0级开始找第一个未满的级别
        for (size_t i = 0; i < levels.size(); i++) {
            if (!levels[i].is_full()) {
                return i;
            }
        }
        return levels.size() - 1;  // 最后一级总是不设限
    }

    bool should_search_next_level(
        const std::vector<std::pair<float, faiss::idx_t>>& results,
        size_t k) const {

        if (results.size() < k) {
            return true;  // 结果不够，继续
        }

        // 检查第k个结果的质量
        float threshold = 100.0f;  // 距离阈值
        if (results.size() >= k && results[k-1].first > threshold) {
            return true;  // 第k个结果太远，继续搜索
        }

        return false;
    }
};
```

---

## 第三部分：混合检索策略

### 3.1 文本+向量混合检索

```cpp
// 文本+向量混合检索（常见于RAG、推荐等场景）
class HybridTextVectorRetriever {
    // 向量索引
    faiss::Index* vector_index;

    // 文本索引（简化，实际应使用倒排索引）
    std::unordered_map<uint64_t, std::string> text_store;
    std::unordered_map<std::string, std::vector<uint64_t>> inverted_index;

    size_t d;

public:
    HybridTextVectorRetriever(size_t dim) : d(dim) {
        vector_index = faiss::index_factory(dim, "HNSW32");
    }

    // 添加文档
    void add_document(uint64_t doc_id, const std::string& text,
                     const std::vector<float>& embedding) {

        // 添加到向量索引
        vector_index->add_with_ids(1, embedding.data(), &doc_id);

        // 添加到文本索引
        text_store[doc_id] = text;

        // 简单分词（实际应使用专业分词器）
        std::vector<std::string> tokens = tokenize(text);
        for (const auto& token : tokens) {
            inverted_index[token].push_back(doc_id);
        }
    }

    // 混合检索结果
    struct HybridResult {
        uint64_t doc_id;
        float vector_score;
        float text_score;
        float combined_score;
        std::string text;
    };

    // 混合检索
    enum class FusionMethod {
        WEIGHTED_SUM,     // 加权求和
        RRF,              // Reciprocal Rank Fusion
        LEARNING_TO_RANK  // 学习排序
    };

    std::vector<HybridResult> hybrid_search(
        const std::string& text_query,
        const std::vector<float>& vector_query,
        size_t k = 10,
        float alpha = 0.5f,  // 向量权重
        FusionMethod method = FusionMethod::WEIGHTED_SUM) {

        // 1. 向量检索
        float* vector_dists = new float[k * 2];
        faiss::idx_t* vector_labels = new faiss::idx_t[k * 2];

        vector_index->search(1, vector_query.data(), k * 2,
                            vector_dists, vector_labels);

        std::unordered_map<faiss::idx_t, float> vector_scores;
        for (size_t i = 0; i < k * 2; i++) {
            if (vector_labels[i] >= 0) {
                // 转换为相似度（0-1）
                vector_scores[vector_labels[i]] = 1.0f / (1.0f + vector_dists[i]);
            }
        }

        delete[] vector_dists;
        delete[] vector_labels;

        // 2. 文本检索（简化BM25）
        auto tokens = tokenize(text_query);
        std::unordered_map<uint64_t, float> text_scores = bm25_search(tokens);

        // 3. 融合
        std::vector<HybridResult> results;
        std::unordered_set<uint64_t> all_ids;

        for (const auto& [id, _] : vector_scores) all_ids.insert(id);
        for (const auto& [id, _] : text_scores) all_ids.insert(id);

        for (uint64_t id : all_ids) {
            HybridResult result;
            result.doc_id = id;
            result.text = text_store[id];

            // 获取分数（默认0）
            result.vector_score = vector_scores.count(id) ? vector_scores[id] : 0.0f;
            result.text_score = text_scores.count(id) ? text_scores[id] : 0.0f;

            // 归一化
            result.vector_score = normalize_score(result.vector_score);
            result.text_score = normalize_score(result.text_score);

            // 融合
            switch (method) {
                case FusionMethod::WEIGHTED_SUM:
                    result.combined_score = alpha * result.vector_score +
                                          (1 - alpha) * result.text_score;
                    break;

                case FusionMethod::RRF:
                    // RRF融合
                    result.combined_score = compute_rrf(
                        result.vector_score, result.text_score);
                    break;

                default:
                    result.combined_score = result.vector_score;
            }

            results.push_back(result);
        }

        // 排序并返回Top-K
        std::sort(results.begin(), results.end(),
                 [](const auto& a, const auto& b) {
                     return a.combined_score > b.combined_score;
                 });

        if (results.size() > k) {
            results.resize(k);
        }

        return results;
    }

private:
    std::vector<std::string> tokenize(const std::string& text) {
        // 简化分词：按空格和标点分割
        std::vector<std::string> tokens;
        std::string token;
        for (char c : text) {
            if (std::isalnum(c)) {
                token += c;
            } else if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        }
        if (!token.empty()) {
            tokens.push_back(token);
        }
        return tokens;
    }

    std::unordered_map<uint64_t, float> bm25_search(
        const std::vector<std::string>& tokens) {

        std::unordered_map<uint64_t, float> scores;

        for (const auto& token : tokens) {
            auto it = inverted_index.find(token);
            if (it == inverted_index.end()) continue;

            // BM25打分（简化）
            float idf = std::log(
                (text_store.size() - it->second.size() + 0.5f) /
                (it->second.size() + 0.5f)
            ) + 1.0f;

            for (uint64_t doc_id : it->second) {
                scores[doc_id] += idf;
            }
        }

        return scores;
    }

    float normalize_score(float score) {
        // Sigmoid归一化到0-1
        return 1.0f / (1.0f + std::exp(-score));
    }

    float compute_rrf(float score1, float score2, float k = 60.0f) {
        // Reciprocal Rank Fusion
        return 1.0f / (k + 1.0f / (score1 + 1e-6f)) +
               1.0f / (k + 1.0f / (score2 + 1e-6f));
    }
};
```

### 3.2 稀疏+密集混合

```cpp
// 稀疏（如BM25）+ 密集（向量）混合
class SparseDenseFusionRetriever {
    // 稠密向量索引
    faiss::Index* dense_index;
    size_t d;

    // 稀疏索引（简化）
    struct SparseVector {
        std::vector<uint32_t> indices;
        std::vector<float> values;
    };

    std::unordered_map<uint64_t, SparseVector> sparse_vectors;
    std::unordered_map<uint64_t, std::string> metadata;

public:
    SparseDenseFusionRetriever(size_t dim) : d(dim) {
        dense_index = faiss::index_factory(dim, "IVF1024,PQ64");
    }

    // 添加文档
    void add_document(
        uint64_t doc_id,
        const std::string& text,
        const std::vector<float>& dense_embedding,
        const SparseVector& sparse_features) {

        // 添加稠密向量
        dense_index->add_with_ids(1, dense_embedding.data(), &doc_id);

        // 添加稀疏特征
        sparse_vectors[doc_id] = sparse_features;
        metadata[doc_id] = text;
    }

    // 融合检索
    struct FusionResult {
        uint64_t doc_id;
        float dense_score;
        float sparse_score;
        float fused_score;
        std::string metadata;
    };

    std::vector<FusionResult> fused_search(
        const std::vector<float>& dense_query,
        const SparseVector& sparse_query,
        size_t k = 10,
        float dense_weight = 0.7f,
        float sparse_weight = 0.3f) {

        // 1. 稠密检索
        float* dense_dists = new float[k * 2];
        faiss::idx_t* dense_labels = new faiss::idx_t[k * 2];

        dense_index->search(1, dense_query.data(), k * 2,
                           dense_dists, dense_labels);

        std::unordered_map<uint64_t, float> dense_scores;
        for (size_t i = 0; i < k * 2; i++) {
            if (dense_labels[i] >= 0) {
                dense_scores[dense_labels[i]] =
                    1.0f / (1.0f + dense_dists[i]);  // 转为相似度
            }
        }

        delete[] dense_dists;
        delete[] dense_labels;

        // 2. 稀疏检索
        std::unordered_map<uint64_t, float> sparse_scores =
            sparse_search(sparse_query);

        // 3. 融合
        std::unordered_map<uint64_t, FusionResult> fused;

        for (const auto& [id, score] : dense_scores) {
            fused[id].doc_id = id;
            fused[id].dense_score = score;
            fused[id].sparse_score = sparse_scores.count(id) ?
                                     sparse_scores[id] : 0.0f;
            fused[id].metadata = metadata[id];
        }

        for (const auto& [id, score] : sparse_scores) {
            if (!fused.count(id)) {
                fused[id].doc_id = id;
                fused[id].dense_score = 0.0f;
                fused[id].sparse_score = score;
                fused[id].metadata = metadata[id];
            }
        }

        // 计算融合分数
        for (auto& [id, result] : fused) {
            // 归一化
            float norm_dense = normalize_score(result.dense_score);
            float norm_sparse = normalize_score(result.sparse_score);

            // 加权融合
            result.fused_score = dense_weight * norm_dense +
                                sparse_weight * norm_sparse;
        }

        // 转换为vector并排序
        std::vector<FusionResult> results;
        for (auto& [id, result] : fused) {
            results.push_back(result);
        }

        std::sort(results.begin(), results.end(),
                 [](const auto& a, const auto& b) {
                     return a.fused_score > b.fused_score;
                 });

        if (results.size() > k) {
            results.resize(k);
        }

        return results;
    }

private:
    std::unordered_map<uint64_t, float> sparse_search(
        const SparseVector& query) {

        std::unordered_map<uint64_t, float> scores;

        for (const auto& [doc_id, sparse_vec] : sparse_vectors) {
            float score = sparse_dot_product(query, sparse_vec);
            if (score > 0) {
                scores[doc_id] = score;
            }
        }

        return scores;
    }

    float sparse_dot_product(const SparseVector& a, const SparseVector& b) {
        size_t i = 0, j = 0;
        float sum = 0.0f;

        while (i < a.indices.size() && j < b.indices.size()) {
            if (a.indices[i] == b.indices[j]) {
                sum += a.values[i] * b.values[j];
                i++;
                j++;
            } else if (a.indices[i] < b.indices[j]) {
                i++;
            } else {
                j++;
            }
        }

        return sum;
    }

    float normalize_score(float score) {
        return 1.0f / (1.0f + std::exp(-score));
    }
};
```

---

## 第四部分：动态查询路由

### 4.1 查询复杂度估计

```cpp
// 查询复杂度估计器
class QueryComplexityEstimator {
public:
    // 查询特征
    struct QueryFeatures {
        float norm;           // 向量范数
        float sparsity;       // 稀疏度（接近0的维度占比）
        float entropy;        // 熵（信息量）
        float concentration;  // 集中度（最大几个维度的占比）

        void print() const {
            printf("Query Features: norm=%.4f, sparsity=%.4f, "
                   "entropy=%.4f, concentration=%.4f\n",
                   norm, sparsity, entropy, concentration);
        }
    };

    // 提取查询特征
    static QueryFeatures extract_features(const float* query, size_t d) {
        QueryFeatures features;

        // 计算范数
        features.norm = std::sqrt(std::inner_product(
            query, query + d, query, 0.0f));

        // 计算稀疏度
        size_t near_zero = 0;
        for (size_t i = 0; i < d; i++) {
            if (std::abs(query[i]) < 0.01f) {
                near_zero++;
            }
        }
        features.sparsity = static_cast<float>(near_zero) / d;

        // 计算熵
        std::vector<float> abs_values(d);
        for (size_t i = 0; i < d; i++) {
            abs_values[i] = std::abs(query[i]);
        }
        float sum = std::accumulate(abs_values.begin(), abs_values.end(), 0.0f);

        features.entropy = 0.0f;
        for (size_t i = 0; i < d; i++) {
            float p = abs_values[i] / (sum + 1e-10f);
            if (p > 1e-10f) {
                features.entropy -= p * std::log2(p + 1e-10f);
            }
        }

        // 计算集中度（Top-10维度的占比）
        std::partial_sort(abs_values.begin(), abs_values.begin() + 10,
                          abs_values.end(), std::greater<float>());
        float top10_sum = std::accumulate(abs_values.begin(),
                                          abs_values.begin() + 10, 0.0f);
        features.concentration = top10_sum / (sum + 1e-10f);

        return features;
    }

    // 估计查询难度（0-1，1最难）
    static float estimate_difficulty(const QueryFeatures& features) {
        // 多个特征的加权组合
        float w1 = 0.3f;  // 范数权重
        float w2 = 0.2f;  // 稀疏度权重
        float w3 = 0.3f;  // 熵权重
        float w4 = 0.2f;  // 集中度权重

        float difficulty =
            w1 * std::min(1.0f, features.norm / 20.0f) +
            w2 * features.sparsity +
            w3 * (1.0f - features.entropy / std::log2(1024.0f)) +
            w4 * features.concentration;

        return std::min(1.0f, std::max(0.0f, difficulty));
    }
};
```

### 4.2 自适应路由器

```cpp
// 自适应查询路由器
class AdaptiveQueryRouter {
public:
    // 路由目标
    enum class RouteTarget {
        EXACT_SEARCH,       // 精确搜索（Flat）
        APPROXIMATE_FAST,   // 快速近似（HNSW低ef）
        APPROXIMATE_ACCURATE, // 准确近似（HNSW高ef）
        HYBRID,            // 混合检索
        REJECT             // 拒绝（缓存查询）
    };

    // 路由决策
    struct RoutingDecision {
        RouteTarget target;
        std::string reason;
        std::unordered_map<std::string, float> parameters;

        void print() const {
            printf("Route: %d (%s)\n", static_cast<int>(target), reason.c_str());
            printf("Parameters: ");
            for (const auto& [k, v] : parameters) {
                printf("%s=%.2f ", k.c_str(), v);
            }
            printf("\n");
        }
    };

    // 路由配置
    struct RouterConfig {
        float difficulty_threshold_low;    // 低难度阈值
        float difficulty_threshold_high;   // 高难度阈值
        float cache_hit_ratio_threshold;   // 缓存命中率阈值
        size_t recent_query_window;       // 最近查询窗口

        static RouterConfig default_config() {
            return {0.3f, 0.7f, 0.8f, 100};
        }
    };

    AdaptiveQueryRouter(const RouterConfig& config = RouterConfig::default_config())
        : config(config) {}

    // 路由查询
    RoutingDecision route_query(
        const float* query,
        size_t d,
        const std::vector<float>& recent_query_difficulties = {}) {

        // 1. 提取特征
        auto features = QueryComplexityEstimator::extract_features(query, d);
        float difficulty = QueryComplexityEstimator::estimate_difficulty(features);

        RoutingDecision decision;

        // 2. 检查缓存
        if (check_cache(query, d)) {
            decision.target = RouteTarget::REJECT;
            decision.reason = "Cache hit";
            return decision;
        }

        // 3. 根据难度路由
        if (difficulty < config.difficulty_threshold_low) {
            // 简单查询：快速近似
            decision.target = RouteTarget::APPROXIMATE_FAST;
            decision.reason = "Low difficulty query";
            decision.parameters["ef"] = 20.0f;

        } else if (difficulty < config.difficulty_threshold_high) {
            // 中等查询：准确近似
            decision.target = RouteTarget::APPROXIMATE_ACCURATE;
            decision.reason = "Medium difficulty query";
            decision.parameters["ef"] = 100.0f;

        } else {
            // 复杂查询：精确或混合
            if (recent_query_difficulties.empty() ||
                std::accumulate(recent_query_difficulties.begin(),
                              recent_query_difficulties.end(), 0.0f) /
                recent_query_difficulties.size() < 0.5f) {

                decision.target = RouteTarget::EXACT_SEARCH;
                decision.reason = "High difficulty, low system load";
            } else {
                decision.target = RouteTarget::HYBRID;
                decision.reason = "High difficulty, high system load";
                decision.parameters["alpha"] = 0.6f;
            }
        }

        return decision;
    }

    // 更新路由策略（基于反馈）
    void update_policy(
        const std::vector<RoutingDecision>& decisions,
        const std::vector<float>& latencies,
        const std::vector<float>& recalls) {

        // 分析各路由策略的效果
        std::map<RouteTarget, std::vector<float>> target_latencies;
        std::map<RouteTarget, std::vector<float>> target_recalls;

        for (size_t i = 0; i < decisions.size(); i++) {
            target_latencies[decisions[i].target].push_back(latencies[i]);
            target_recalls[decisions[i].target].push_back(recalls[i]);
        }

        // 打印统计
        printf("\n=== Routing Policy Analysis ===\n");
        for (const auto& [target, lats] : target_latencies) {
            float avg_lat = std::accumulate(lats.begin(), lats.end(), 0.0f) / lats.size();
            float avg_rec = std::accumulate(target_recalls[target].begin(),
                                           target_recalls[target].end(), 0.0f) /
                           target_recalls[target].size();

            printf("Target %d: Lat=%.2fms, Recall=%.3f\n",
                   static_cast<int>(target), avg_lat, avg_rec);
        }

        // 根据效果调整阈值（简化）
        // 实际实现需要更复杂的强化学习
    }

private:
    RouterConfig config;

    // LRU缓存
    struct CacheEntry {
        std::vector<float> query;
        std::chrono::system_clock::time_point timestamp;
    };

    std::list<CacheEntry> query_cache;
    size_t max_cache_size = 1000;

    bool check_cache(const float* query, size_t d) {
        // 简化：检查是否有相同查询
        std::vector<float> q(query, query + d);

        for (const auto& entry : query_cache) {
            if (entry.query == q) {
                return true;
            }
        }

        // 添加到缓存
        query_cache.push_front({q, std::chrono::system_clock::now()});
        if (query_cache.size() > max_cache_size) {
            query_cache.pop_back();
        }

        return false;
    }
};
```

---

## 实验练习

### 练习1: 实现IVF+PQ融合索引

```cpp
void exercise_1_ivf_pq_fusion() {
    // 1. 实现IVF+PQ融合索引
    // 2. 测试不同nprobe和M值的影响
    // 3. 对比单独IVF和单独PQ
    // 4. 找到最优参数组合
}
```

### 练习2: 实现多路径融合

```cpp
void exercise_2_multi_path_fusion() {
    // 1. 实现HNSW+IVF多路径索引
    // 2. 测试并行融合效果
    // 3. 实现自适应路由
    // 4. 对比不同融合策略
}
```

### 练习3: 实现混合检索

```cpp
void exercise_3_hybrid_retrieval() {
    // 1. 实现文本+向量混合检索
    // 2. 实现稀疏+密集混合
    // 3. 测试不同融合方法
    // 4. 评估检索质量
}
```

### 练习4: 实现自适应路由

```cpp
void exercise_4_adaptive_routing() {
    // 1. 实现查询复杂度估计
    // 2. 实现自适应路由器
    // 3. 基于反馈优化路由策略
    // 4. 测试系统性能提升
}
```

---

## 总结

第26天深入探讨了索引融合与混合检索策略，涵盖：

1. **融合基础**：
   - 索引融合动机
   - 不同索引的特性对比
   - 融合层次与模式分类

2. **融合技术**：
   - IVF+PQ粗细粒度融合
   - 多路径融合（HNSW+IVF）
   - 层次化索引系统

3. **混合检索**：
   - 文本+向量混合
   - 稀疏+密集混合
   - RRF等融合算法

4. **动态路由**：
   - 查询复杂度估计
   - 自适应路由策略
   - 基于反馈的优化

**关键要点**：
- 索引融合可以在精度、速度、内存间取得更好平衡
- 多路径融合提供冗余和鲁棒性
- 层次化索引适合冷热数据分离
- 混合检索能利用多种信号的优势
- 自适应路由是提升系统效率的关键
- 需要根据实际场景选择合适的融合策略

## 后续学习

- 研究更复杂的融合策略（学习排序）
- 探索在线学习的路由优化
- 实践大规模生产环境的混合检索
- 研究多目标优化（延迟、召回、成本）
