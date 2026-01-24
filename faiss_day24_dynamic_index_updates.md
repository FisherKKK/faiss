# Faiss深度学习课程 - 第24天：动态索引更新与维护

## 课程概述

第24天探讨向量索引的动态更新与维护策略。实际应用中，数据集往往是动态变化的 - 新向量不断加入，旧向量可能删除或更新。本课程全面解析如何高效地处理这些动态操作。

## 学习目标

- 理解动态索引的挑战
- 掌握增量索引构建策略
- 学习删除操作的各种实现
- 理解索引重组与优化时机
- 掌握内存管理和资源释放
- 实践生产环境的索引维护

---

## 第一部分：动态索引的挑战

### 1.1 问题分析

```cpp
// 动态索引挑战分析
class DynamicIndexChallenges {
public:
    // 动态操作类型
    enum class OperationType {
        INSERT,
        DELETE,
        UPDATE,
        BATCH_INSERT,
        BATCH_DELETE
    };

    // 性能指标
    struct PerformanceMetrics {
        double insert_latency_ms;
        double delete_latency_ms;
        double search_latency_ms;
        double index_size_mb;
        double fragmentation_ratio;
        double recall;

        void print() const {
            printf("Insert latency: %.2f ms\n", insert_latency_ms);
            printf("Delete latency: %.2f ms\n", delete_latency_ms);
            printf("Search latency: %.2f ms\n", search_latency_ms);
            printf("Index size: %.2f MB\n", index_size_mb);
            printf("Fragmentation: %.2f%%\n", fragmentation_ratio * 100);
            printf("Recall: %.3f\n", recall);
        }
    };

    // 分析不同索引对动态操作的支持
    static void analyze_index_support() {
        printf("\n=== Index Type Support for Dynamic Operations ===\n");
        printf("%-20s | %-8s | %-8s | %-8s | %-10s\n",
               "Index Type", "Insert", "Delete", "Update", "Performance");
        printf("%-20s | %-8s | %-8s | %-8s | %-10s\n",
               "--------------------", "--------", "--------", "--------", "----------");

        printf("%-20s | %-8s | %-8s | %-8s | %-10s\n",
               "IndexFlat", "Fast", "Fast", "Fast", "High");
        printf("%-20s | %-8s | %-8s | %-8s | %-10s\n",
               "IndexIVF", "Medium", "Medium", "Medium", "High");
        printf("%-20s | %-8s | %-8s | %-8s | %-10s\n",
               "IndexIVFPQ", "Medium", "Slow", "Slow", "Medium");
        printf("%-20s | %-8s | %-8s | %-8s | %-10s\n",
               "IndexHNSW", "Fast", "Slow", "Slow", "High");
        printf("%-20s | %-8s | %-8s | %-8s | %-10s\n",
               "IndexNSG", "Fast", "Slow", "Slow", "Medium");
        printf("%-20s | %-8s | %-8s | %-8s | %-10s\n",
               "IndexPQ", "Slow", "N/A", "N/A", "Low");
    }

    // 分析性能衰减
    static void analyze_performance_degradation() {
        printf("\n=== Performance Degradation Analysis ===\n");

        std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};
        std::vector<double> fragmentation_rates = {0.0, 0.05, 0.15, 0.30};

        printf("Size   | Frag | Search Latency | Recall | Memory Overhead\n");
        printf("-------|------|----------------|--------|----------------\n");

        for (size_t i = 0; i < sizes.size(); i++) {
            size_t n = sizes[i];
            double frag = fragmentation_rates[i];

            // 模拟性能衰减
            double base_latency = std::log2(static_cast<double>(n)) * 0.1;
            double actual_latency = base_latency * (1.0 + frag * 2.0);
            double recall = 0.98 - frag * 0.3;
            double memory_overhead = 1.0 + frag;

            printf("%7zu | %.2f | %14.2f | %6.3f | %.2f\n",
                   n, frag, actual_latency, recall, memory_overhead);
        }
    }
};
```

### 1.2 设计考量

```cpp
// 动态索引设计原则
class DynamicIndexDesign {
public:
    // 设计权衡
    struct DesignTradeoffs {
        bool use_lazy_deletion;        // 惰性删除
        bool background_compaction;    // 后台压缩
        double compaction_trigger;     // 压缩触发阈值
        size_t max_segment_size;       // 最大段大小
        int merge_policy;              // 合并策略

        void print() const {
            printf("Design Configuration:\n");
            printf("  Lazy deletion: %s\n", use_lazy_deletion ? "Yes" : "No");
            printf("  Background compaction: %s\n",
                   background_compaction ? "Yes" : "No");
            printf("  Compaction trigger: %.2f%%\n",
                   compaction_trigger * 100);
            printf("  Max segment size: %zu\n", max_segment_size);
            printf("  Merge policy: %d\n", merge_policy);
        }
    };

    // 推荐配置
    static DesignTradeoffs recommend_config(
        double insert_rate_per_sec,
        double delete_rate_per_sec,
        double query_rate_per_sec) {

        DesignTradeoffs config;

        double total_rate = insert_rate_per_sec +
                           delete_rate_per_sec +
                           query_rate_per_sec;

        if (total_rate < 100) {
            // 低负载：简单配置
            config.use_lazy_deletion = false;
            config.background_compaction = false;
            config.compaction_trigger = 0.5;
            config.max_segment_size = 100000;
            config.merge_policy = 0;
        } else if (total_rate < 1000) {
            // 中负载
            config.use_lazy_deletion = true;
            config.background_compaction = true;
            config.compaction_trigger = 0.3;
            config.max_segment_size = 500000;
            config.merge_policy = 1;
        } else {
            // 高负载：激进优化
            config.use_lazy_deletion = true;
            config.background_compaction = true;
            config.compaction_trigger = 0.2;
            config.max_segment_size = 1000000;
            config.merge_policy = 2;
        }

        return config;
    }
};
```

---

## 第二部分：增量索引构建

### 2.1 分段索引策略

```cpp
// 分段索引（Segment-based Index）
class SegmentedIndex {
    struct Segment {
        std::unique_ptr<faiss::Index> index;
        size_t start_id;
        size_t size;
        bool read_only;

        // 统计信息
        size_t delete_count;
        std::chrono::system_clock::time_point created_at;
    };

    std::vector<Segment> segments;
    size_t d;
    size_t max_segment_size;
    size_t next_id;

public:
    SegmentedIndex(size_t dim, size_t max_seg_size = 100000)
        : d(dim), max_segment_size(max_seg_size), next_id(0) {

        // 创建初始段
        create_new_segment();
    }

    // 添加向量
    size_t add_vectors(const float* vectors, size_t n) {
        Segment& active_seg = segments.back();

        // 检查是否需要新段
        if (active_seg.size + n > max_segment_size) {
            active_seg.read_only = true;
            create_new_segment();

            // 触发合并检查
            check_and_merge();
        }

        // 添加到当前活跃段
        Segment& current_seg = segments.back();
        faiss::idx_t* ids = new faiss::idx_t[n];

        for (size_t i = 0; i < n; i++) {
            ids[i] = next_id++;
        }

        current_seg.index->add_with_ids(n, vectors, ids);
        current_seg.size += n;

        delete[] ids;
        return next_id - n;
    }

    // 搜索向量
    void search(const float* queries, size_t nq, size_t k,
               float* distances, faiss::idx_t* labels) const {

        // 收集所有段的结果
        using HeapElement = std::tuple<float, faiss::idx_t>;
        std::vector<std::priority_queue<HeapElement,
            std::vector<HeapElement>, std::greater<HeapElement>>> heaps(nq);

        // 并行搜索所有段
        #pragma omp parallel for
        for (size_t seg_idx = 0; seg_idx < segments.size(); seg_idx++) {
            const Segment& seg = segments[seg_idx];

            // 临时存储该段的结果
            float* seg_distances = new float[nq * k];
            faiss::idx_t* seg_labels = new faiss::idx_t[nq * k];

            seg.index->search(queries, nq, k,
                            seg_distances, seg_labels);

            // 合并到堆中
            for (size_t q = 0; q < nq; q++) {
                #pragma omp critical
                {
                    for (size_t i = 0; i < k; i++) {
                        faiss::idx_t label = seg_labels[q * k + i];
                        if (label >= 0) {  // 有效的结果
                            heaps[q].push({
                                seg_distances[q * k + i],
                                label
                            });
                        }
                    }
                }
            }

            delete[] seg_distances;
            delete[] seg_labels;
        }

        // 提取Top-K
        for (size_t q = 0; q < nq; q++) {
            for (size_t i = 0; i < k && !heaps[q].empty(); i++) {
                auto [dist, label] = heaps[q].top();
                heaps[q].pop();
                labels[q * k + k - 1 - i] = label;
                distances[q * k + k - 1 - i] = dist;
            }

            // 填充剩余位置
            for (size_t i = heaps[q].size(); i < k; i++) {
                labels[q * k + i] = -1;
                distances[q * k + i] = std::numeric_limits<float>::infinity();
            }
        }
    }

    // 获取统计信息
    void print_stats() const {
        printf("Segmented Index Stats:\n");
        printf("  Total segments: %zu\n", segments.size());
        printf("  Total vectors: %zu\n", next_id);

        size_t total_deletes = 0;
        for (const auto& seg : segments) {
            total_deletes += seg.delete_count;
        }

        printf("  Total deletions: %zu\n", total_deletes);
        printf("  Fragmentation: %.2f%%\n",
               static_cast<double>(total_deletes) / next_id * 100);

        printf("\nSegment Details:\n");
        for (size_t i = 0; i < segments.size(); i++) {
            const auto& seg = segments[i];
            printf("  [%zu] Size: %zu, Deleted: %zu, RO: %s\n",
                   i, seg.size, seg.delete_count,
                   seg.read_only ? "Yes" : "No");
        }
    }

private:
    void create_new_segment() {
        Segment seg;
        seg.index = std::make_unique<faiss::IndexFlatL2>(d);
        seg.start_id = next_id;
        seg.size = 0;
        seg.read_only = false;
        seg.delete_count = 0;
        seg.created_at = std::chrono::system_clock::now();

        segments.push_back(std::move(seg));
    }

    void check_and_merge() {
        // 简化的合并策略：合并小的只读段
        if (segments.size() < 3) return;

        // 检查最后两个只读段是否可以合并
        size_t n = segments.size();
        if (segments[n-2].read_only &&
            segments[n-2].size < max_segment_size / 2) {

            printf("Merging segments %zu and %zu\n", n-3, n-2);

            // 合并逻辑（简化）
            merge_segments(n-3, n-2);
        }
    }

    void merge_segments(size_t idx1, size_t idx2) {
        // 实际实现需要更复杂的逻辑
        // 这里仅作示意
    }
};
```

### 2.2 增量IVF索引

```cpp
// 增量IVF索引
class IncrementalIVFIndex {
    faiss::IndexIVFFlat* base_index;
    size_t d;
    size_t nlist;
    bool trained;

    // 追踪倒排表大小
    std::vector<size_t> list_sizes;

public:
    IncrementalIVFIndex(size_t dim, size_t n_lists)
        : d(dim), nlist(n_lists), trained(false) {

        auto quantizer = new faiss::IndexFlatL2(d);
        base_index = new faiss::IndexIVFFlat(quantizer, d, n_lists);
        list_sizes.resize(nlist, 0);
    }

    // 训练（如果需要）
    void train(size_t n, const float* vectors) {
        if (!trained) {
            base_index->train(n, vectors);
            trained = true;
        }
    }

    // 添加向量
    void add_vectors(size_t n, const float* vectors) {
        if (!trained) {
            throw std::runtime_error("Index not trained");
        }

        // 记录添加前的列表大小
        auto old_sizes = list_sizes;

        // 添加向量
        base_index->add(n, vectors);

        // 更新列表大小（简化）
        // 实际需要从索引中提取
        for (size_t i = 0; i < nlist; i++) {
            list_sizes[i] = old_sizes[i];  // 简化
        }
    }

    // 添加到特定倒排表（优化）
    void add_to_list(size_t list_id, size_t n, const float* vectors) {
        // 直接添加到指定列表，避免重新计算
        // 需要访问IVF内部结构
    }

    // 搜索
    void search(const float* queries, size_t nq, size_t k,
               float* distances, faiss::idx_t* labels,
               size_t nprobe = 10) const {

        faiss::IVFSearchParameters params;
        params.nprobe = nprobe;

        base_index->search(nq, queries, k,
                          distances, labels, &params);
    }

    // 获取统计信息
    void print_stats() const {
        printf("Incremental IVF Index Stats:\n");
        printf("  Trained: %s\n", trained ? "Yes" : "No");
        printf("  Lists: %zu\n", nlist);
        printf("  Total vectors: %zu\n", base_index->ntotal);

        size_t min_size = SIZE_MAX;
        size_t max_size = 0;
        double avg_size = 0.0;

        for (size_t sz : list_sizes) {
            min_size = std::min(min_size, sz);
            max_size = std::max(max_size, sz);
            avg_size += sz;
        }
        avg_size /= nlist;

        printf("\nList Size Distribution:\n");
        printf("  Min: %zu\n", min_size);
        printf("  Max: %zu\n", max_size);
        printf("  Avg: %.1f\n", avg_size);
        printf("  Imbalance ratio: %.2f\n",
               static_cast<double>(max_size) / (min_size + 1));
    }

    // 重新平衡倒排表
    void rebalance_lists() {
        printf("Rebalancing inverted lists...\n");

        // 1. 分析当前分布
        std::vector<std::pair<size_t, size_t>> list_sizes_with_id;
        for (size_t i = 0; i < nlist; i++) {
            list_sizes_with_id.push_back({list_sizes[i], i});
        }

        // 2. 排序
        std::sort(list_sizes_with_id.begin(), list_sizes_with_id.end(),
                 std::greater<>());

        // 3. 重新分配（简化）
        // 实际实现需要移动数据
    }
};
```

---

## 第三部分：删除操作

### 3.1 惰性删除

```cpp
// 惰性删除索引包装器
class LazyDeletionIndex {
    faiss::Index* base_index;
    std::unordered_set<faiss::idx_t> deleted_ids;
    size_t d;
    std::mutex mutex;

public:
    LazyDeletionIndex(faiss::Index* index) : base_index(index) {
        d = index->d;
    }

    // 添加向量
    void add(size_t n, const float* vectors) {
        std::lock_guard<std::mutex> lock(mutex);

        faiss::idx_t* ids = new faiss::idx_t[n];
        for (size_t i = 0; i < n; i++) {
            ids[i] = base_index->ntotal + i;
        }

        base_index->add_with_ids(n, vectors, ids);
        delete[] ids;
    }

    // 标记删除
    void remove_ids(size_t n, const faiss::idx_t* ids) {
        std::lock_guard<std::mutex> lock(mutex);

        for (size_t i = 0; i < n; i++) {
            deleted_ids.insert(ids[i]);
        }

        // 触发压缩检查
        maybe_compact();
    }

    // 搜索（过滤已删除的ID）
    void search(const float* queries, size_t nq, size_t k,
               float* distances, faiss::idx_t* labels) const {

        // 搜索更多结果（考虑删除）
        size_t search_k = k + deleted_ids.size() * 2;

        float* all_distances = new float[nq * search_k];
        faiss::idx_t* all_labels = new faiss::idx_t[nq * search_k];

        base_index->search(queries, nq, search_k,
                          all_distances, all_labels);

        // 过滤已删除的ID
        for (size_t q = 0; q < nq; q++) {
            size_t found = 0;
            for (size_t i = 0; i < search_k && found < k; i++) {
                faiss::idx_t label = all_labels[q * search_k + i];

                if (label < 0) continue;  // 无效结果

                // 检查是否已删除
                std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex));
                if (deleted_ids.find(label) == deleted_ids.end()) {
                    labels[q * k + found] = label;
                    distances[q * k + found] = all_distances[q * search_k + i];
                    found++;
                }
            }

            // 填充剩余位置
            for (size_t i = found; i < k; i++) {
                labels[q * k + i] = -1;
                distances[q * k + i] = std::numeric_limits<float>::infinity();
            }
        }

        delete[] all_distances;
        delete[] all_labels;
    }

    // 获取统计信息
    void print_stats() const {
        printf("Lazy Deletion Stats:\n");
        printf("  Total vectors: %zu\n", base_index->ntotal);
        printf("  Deleted vectors: %zu\n", deleted_ids.size());
        printf("  Active vectors: %zu\n",
               base_index->ntotal - deleted_ids.size());
        printf("  Deletion ratio: %.2f%%\n",
               static_cast<double>(deleted_ids.size()) / base_index->ntotal * 100);
    }

    // 压缩索引
    void compact() {
        printf("Compacting index...\n");

        // 创建新索引
        faiss::Index* new_index = faiss::index_factory(d, "Flat");

        // 复制未删除的向量
        // 实际实现需要遍历原索引
        // 这里简化

        // 替换索引
        delete base_index;
        base_index = new_index;
        deleted_ids.clear();

        printf("Compaction complete\n");
    }

private:
    void maybe_compact() {
        // 当删除比例超过阈值时压缩
        double deletion_ratio =
            static_cast<double>(deleted_ids.size()) /
            (base_index->ntotal + 1);

        if (deletion_ratio > 0.3) {
            printf("Deletion ratio %.2f%% exceeds threshold, triggering compaction\n",
                   deletion_ratio * 100);
            compact();
        }
    }
};
```

### 3.2 实时删除（HNSW）

```cpp
// HNSW删除支持（需要特殊处理）
class HNSWWithDeletion {
    faiss::IndexHNSWFlat* hnsw_index;
    std::unordered_set<faiss::idx_t> deleted_ids;
    size_t d;

public:
    HNSWWithDeletion(size_t dim, int M = 16) : d(dim) {
        hnsw_index = new faiss::IndexHNSWFlat(dim, M);
    }

    // 添加向量
    void add(size_t n, const float* vectors) {
        faiss::idx_t* ids = new faiss::idx_t[n];
        for (size_t i = 0; i < n; i++) {
            ids[i] = hnsw_index->ntotal + i;
        }

        hnsw_index->add_with_ids(n, vectors, ids);
        delete[] ids;
    }

    // 删除向量（惰性）
    void remove(faiss::idx_t id) {
        deleted_ids.insert(id);

        // HNSW的优化：标记节点的邻居链接为无效
        // 这需要访问HNSW内部图结构
        invalidate_hnsw_node(id);
    }

    // 搜索
    void search(const float* queries, size_t nq, size_t k,
               float* distances, faiss::idx_t* labels) const {

        // 搜索Top-(K + deleted_count)
        size_t search_k = k + deleted_ids.size();

        float* all_distances = new float[nq * search_k];
        faiss::idx_t* all_labels = new faiss::idx_t[nq * search_k];

        hnsw_index->search(queries, nq, search_k,
                          all_distances, all_labels);

        // 过滤已删除的
        for (size_t q = 0; q < nq; q++) {
            size_t found = 0;
            for (size_t i = 0; i < search_k && found < k; i++) {
                faiss::idx_t label = all_labels[q * search_k + i];

                if (label >= 0 && deleted_ids.find(label) == deleted_ids.end()) {
                    labels[q * k + found] = label;
                    distances[q * k + found] = all_distances[q * search_k + i];
                    found++;
                }
            }

            for (size_t i = found; i < k; i++) {
                labels[q * k + i] = -1;
                distances[q * k + i] = std::numeric_limits<float>::infinity();
            }
        }

        delete[] all_distances;
        delete[] all_labels;
    }

    // 重建索引（清理删除）
    void rebuild() {
        printf("Rebuilding HNSW index...\n");

        // 创建新索引
        faiss::IndexHNSWFlat* new_index = new faiss::IndexHNSWFlat(d, 16);

        // 复制未删除的向量（需要访问原始数据）
        // 简化：假设可以遍历

        delete hnsw_index;
        hnsw_index = new_index;
        deleted_ids.clear();

        printf("Rebuild complete\n");
    }

private:
    void invalidate_hnsw_node(faiss::idx_t id) {
        // 实际需要访问HNSW内部结构
        // 标记节点的所有边为无效
    }
};
```

---

## 第四部分：索引重组与优化

### 4.1 压缩策略

```cpp
// 索引压缩器
class IndexCompactor {
public:
    // 压缩策略
    enum class Strategy {
        IMMEDIATE,       // 立即压缩
        THRESHOLD_BASED, // 基于阈值
        SCHEDULED,       // 定期压缩
        ADAPTIVE         // 自适应
    };

    IndexCompactor(Strategy s) : strategy(s) {
        last_compaction = std::chrono::system_clock::now();
    }

    // 压缩配置
    struct CompactionConfig {
        double fragmentation_threshold;  // 碎片阈值
        double time_interval_hours;      // 时间间隔
        size_t min_size_to_compact;      // 最小压缩大小

        static CompactionConfig default_config() {
            return {0.3, 24.0, 10000};
        }
    };

    // 执行压缩
    void compact(faiss::Index* index,
                const std::unordered_set<faiss::idx_t>& deleted_ids,
                const CompactionConfig& config) {

        if (!should_compact(index, deleted_ids, config)) {
            return;
        }

        printf("Starting index compaction...\n");

        auto start = std::chrono::high_resolution_clock::now();

        // 创建新索引
        faiss::Index* new_index = create_clean_index(index, deleted_ids);

        // 替换旧索引
        // 实际应用中需要原子替换

        auto end = std::chrono::high_resolution_clock::now();
        double duration_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        printf("Compaction completed in %.2f ms\n", duration_ms);
        printf("  Old size: %zu\n", index->ntotal);
        printf("  New size: %zu\n", new_index->ntotal);
        printf("  Space saved: %.2f%%\n",
               (1.0 - static_cast<double>(new_index->ntotal) / index->ntotal) * 100);
    }

private:
    Strategy strategy;
    std::chrono::system_clock::time_point last_compaction;

    bool should_compact(faiss::Index* index,
                       const std::unordered_set<faiss::idx_t>& deleted_ids,
                       const CompactionConfig& config) {

        // 检查大小
        if (index->ntotal < config.min_size_to_compact) {
            return false;
        }

        switch (strategy) {
            case Strategy::IMMEDIATE:
                return !deleted_ids.empty();

            case Strategy::THRESHOLD_BASED: {
                double frag_ratio =
                    static_cast<double>(deleted_ids.size()) / index->ntotal;
                return frag_ratio >= config.fragmentation_threshold;
            }

            case Strategy::SCHEDULED: {
                auto now = std::chrono::system_clock::now();
                auto hours_since_last =
                    std::chrono::duration_cast<std::chrono::hours>(
                        now - last_compaction).count();

                if (hours_since_last >= config.time_interval_hours) {
                    last_compaction = now;
                    return !deleted_ids.empty();
                }
                return false;
            }

            case Strategy::ADAPTIVE: {
                // 自适应：根据系统负载决定
                double frag_ratio =
                    static_cast<double>(deleted_ids.size()) / index->ntotal;
                double load_factor = get_system_load_factor();

                // 高负载时更激进，低负载时保守
                double adjusted_threshold = config.fragmentation_threshold *
                                           (1.0 - load_factor * 0.5);

                return frag_ratio >= adjusted_threshold;
            }

            default:
                return false;
        }
    }

    faiss::Index* create_clean_index(
        faiss::Index* old_index,
        const std::unordered_set<faiss::idx_t>& deleted_ids) {

        // 创建新索引（同类型）
        faiss::Index* new_index = faiss::index_factory(
            old_index->d, "Flat", faiss::METRIC_L2);

        // 复制未删除的向量
        // 实际实现需要遍历原索引
        // 这里简化

        return new_index;
    }

    double get_system_load_factor() const {
        // 获取系统负载（0-1）
        // 实际实现需要监控系统指标
        return 0.5;
    }
};
```

### 4.2 倒排表重平衡

```cpp
// IVF倒排表重平衡器
class IVFRebalancer {
public:
    // 重平衡配置
    struct RebalanceConfig {
        double imbalance_threshold;  // 不平衡阈值
        bool use_data_movement;      // 是否移动数据
        bool resize_centroids;       // 是否调整质心

        static RebalanceConfig default_config() {
            return {2.0, true, true};
        }
    };

    // 分析不平衡
    struct ImbalanceAnalysis {
        double imbalance_ratio;
        std::vector<size_t> overloaded_lists;
        std::vector<size_t> underloaded_lists;
        double std_dev;

        void print() const {
            printf("Imbalance Analysis:\n");
            printf("  Ratio: %.2f\n", imbalance_ratio);
            printf("  Std dev: %.2f\n", std_dev);
            printf("  Overloaded: %zu lists\n", overloaded_lists.size());
            printf("  Underloaded: %zu lists\n", underloaded_lists.size());
        }
    };

    // 分析IVF索引的不平衡
    static ImbalanceAnalysis analyze_imbalance(
        const std::vector<size_t>& list_sizes) {

        ImbalanceAnalysis analysis;

        if (list_sizes.empty()) return analysis;

        // 计算统计量
        size_t sum = 0;
        size_t min_size = SIZE_MAX;
        size_t max_size = 0;

        for (size_t sz : list_sizes) {
            sum += sz;
            min_size = std::min(min_size, sz);
            max_size = std::max(max_size, sz);
        }

        double mean = static_cast<double>(sum) / list_sizes.size();

        // 计算标准差
        double variance = 0.0;
        for (size_t sz : list_sizes) {
            double diff = sz - mean;
            variance += diff * diff;
        }
        analysis.std_dev = std::sqrt(variance / list_sizes.size());

        // 不平衡比率
        analysis.imbalance_ratio =
            static_cast<double>(max_size) / (min_size + 1);

        // 识别过载和欠载列表
        for (size_t i = 0; i < list_sizes.size(); i++) {
            if (list_sizes[i] > mean + 2 * analysis.std_dev) {
                analysis.overloaded_lists.push_back(i);
            } else if (list_sizes[i] < mean - 2 * analysis.std_dev) {
                analysis.underloaded_lists.push_back(i);
            }
        }

        return analysis;
    }

    // 重平衡倒排表
    static void rebalance(
        faiss::IndexIVF* ivf_index,
        const RebalanceConfig& config) {

        printf("Rebalancing IVF index...\n");

        // 1. 获取当前列表大小
        size_t nlist = ivf_index->nlist;
        std::vector<size_t> list_sizes(nlist);

        for (size_t i = 0; i < nlist; i++) {
            list_sizes[i] = ivf_index->invlists->list_size(i);
        }

        // 2. 分析不平衡
        auto analysis = analyze_imbalance(list_sizes);
        analysis.print();

        // 3. 检查是否需要重平衡
        if (analysis.imbalance_ratio < config.imbalance_threshold) {
            printf("Imbalance ratio %.2f below threshold %.2f, skipping\n",
                   analysis.imbalance_ratio, config.imbalance_threshold);
            return;
        }

        // 4. 执行重平衡
        if (config.resize_centroids) {
            // 选项A: 重新训练质心
            retrain_centroids(ivf_index);
        } else if (config.use_data_movement) {
            // 选项B: 移动数据
            move_data_between_lists(ivf_index,
                                   analysis.overloaded_lists,
                                   analysis.underloaded_lists);
        }

        printf("Rebalancing complete\n");
    }

private:
    static void retrain_centroids(faiss::IndexIVF* ivf_index) {
        printf("Retraining centroids...\n");

        // 收集所有向量
        // 重新运行k-means
        // 重新分配向量
        // 实际实现较为复杂
    }

    static void move_data_between_lists(
        faiss::IndexIVF* ivf_index,
        const std::vector<size_t>& from_lists,
        const std::vector<size_t>& to_lists) {

        printf("Moving data between lists...\n");

        // 从过载列表移动部分向量到欠载列表
        // 实际实现需要修改InvertedLists结构
    }
};
```

---

## 第五部分：内存管理

### 5.1 内存池

```cpp
// 向量内存池
class VectorMemoryPool {
    struct MemoryBlock {
        float* data;
        size_t capacity;
        size_t used;
        bool in_use;
    };

    std::vector<MemoryBlock> blocks;
    size_t d;
    size_t default_block_size;

public:
    VectorMemoryPool(size_t dim, size_t block_size = 10000)
        : d(dim), default_block_size(block_size) {

        // 预分配一个块
        allocate_block();
    }

    ~VectorMemoryPool() {
        for (auto& block : blocks) {
            delete[] block.data;
        }
    }

    // 分配向量空间
    float* allocate_vectors(size_t n) {
        for (auto& block : blocks) {
            if (!block.in_use &&
                block.capacity - block.used >= n * d) {

                // 在当前块中分配
                float* ptr = block.data + block.used * d;
                block.used += n;
                return ptr;
            }
        }

        // 需要新块
        allocate_block(std::max(default_block_size, n));
        return blocks.back().data;
    }

    // 释放块
    void deallocate_block(float* ptr) {
        for (auto& block : blocks) {
            if (block.data == ptr) {
                block.in_use = false;
                block.used = 0;
                return;
            }
        }
    }

    // 获取统计信息
    void print_stats() const {
        size_t total_capacity = 0;
        size_t total_used = 0;

        for (const auto& block : blocks) {
            total_capacity += block.capacity;
            total_used += block.used;
        }

        printf("Memory Pool Stats:\n");
        printf("  Blocks: %zu\n", blocks.size());
        printf("  Total capacity: %zu vectors\n", total_capacity);
        printf("  Total used: %zu vectors\n", total_used);
        printf("  Utilization: %.2f%%\n",
               static_cast<double>(total_used) / total_capacity * 100);
        printf("  Memory: %.2f MB\n",
               total_capacity * d * sizeof(float) / (1024.0 * 1024));
    }

private:
    void allocate_block(size_t min_size = 0) {
        size_t size = std::max(default_block_size, min_size);

        MemoryBlock block;
        block.data = new float[size * d];
        block.capacity = size;
        block.used = 0;
        block.in_use = false;

        blocks.push_back(block);
    }
};
```

### 5.2 资源释放

```cpp
// 索引资源管理器
class IndexResourceManager {
    struct IndexEntry {
        std::string name;
        faiss::Index* index;
        size_t last_used;
        size_t memory_usage;
        bool pinned;  // 是否常驻内存
    };

    std::unordered_map<std::string, IndexEntry> indexes;
    size_t max_memory_mb;
    size_t current_memory_mb;
    std::mutex mutex;

public:
    IndexResourceManager(size_t max_mem) : max_memory_mb(max_mem),
                                           current_memory_mb(0) {}

    // 注册索引
    void register_index(const std::string& name, faiss::Index* index,
                       bool pinned = false) {

        std::lock_guard<std::mutex> lock(mutex);

        size_t mem_usage = estimate_memory_usage(index);

        // 检查内存限制
        if (current_memory_mb + mem_usage > max_memory_mb) {
            // 尝试释放未pinned的索引
            if (!evict_lru(mem_usage)) {
                printf("Warning: Cannot accommodate index %s (%zu MB)\n",
                       name.c_str(), mem_usage);
                return;
            }
        }

        indexes[name] = {name, index, 0, mem_usage, pinned};
        current_memory_mb += mem_usage;

        printf("Registered index %s (%zu MB)\n", name.c_str(), mem_usage);
    }

    // 获取索引
    faiss::Index* get_index(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = indexes.find(name);
        if (it != indexes.end()) {
            it->second.last_used = get_current_time();
            return it->second.index;
        }

        // 尝试从磁盘加载
        return load_index_from_disk(name);
    }

    // 卸载索引
    void unload_index(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = indexes.find(name);
        if (it != indexes.end() && !it->second.pinned) {
            // 保存到磁盘
            save_index_to_disk(it->second.index, name);

            // 释放内存
            current_memory_mb -= it->second.memory_usage;
            delete it->second.index;
            indexes.erase(it);

            printf("Unloaded index %s\n", name.c_str());
        }
    }

    // 打印状态
    void print_status() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex));

        printf("Index Resource Manager:\n");
        printf("  Memory: %zu / %zu MB\n",
               current_memory_mb, max_memory_mb);
        printf("  Indexes: %zu\n", indexes.size());

        for (const auto& [name, entry] : indexes) {
            printf("    %s: %zu MB, %s\n",
                   name.c_str(),
                   entry.memory_usage,
                   entry.pinned ? "pinned" : "swappable");
        }
    }

private:
    faiss::Index* load_index_from_disk(const std::string& name) {
        // 从磁盘加载索引
        printf("Loading index %s from disk...\n", name.c_str());
        return nullptr;
    }

    void save_index_to_disk(faiss::Index* index, const std::string& name) {
        // 保存索引到磁盘
        printf("Saving index %s to disk...\n", name.c_str());
    }

    size_t estimate_memory_usage(faiss::Index* index) const {
        // 估计索引内存使用
        // 简化实现
        return index->ntotal * index->d * sizeof(float) / (1024 * 1024);
    }

    bool evict_lru(size_t required_memory) {
        // 找到最久未使用的非pinned索引
        std::string lru_name;
        size_t lru_time = SIZE_MAX;

        for (const auto& [name, entry] : indexes) {
            if (!entry.pinned && entry.last_used < lru_time) {
                lru_time = entry.last_used;
                lru_name = name;
            }
        }

        if (!lru_name.empty()) {
            unload_index(lru_name);
            return true;
        }

        return false;
    }

    size_t get_current_time() const {
        return std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};
```

---

## 实验练习

### 练习1: 实现动态IVF索引

```cpp
void exercise_1_dynamic_ivf() {
    // 1. 实现支持增量添加的IVF索引
    // 2. 测试添加性能
    // 3. 实现倒排表重平衡
    // 4. 测试重平衡效果
}
```

### 练习2: 比较删除策略

```cpp
void exercise_2_deletion_strategies() {
    // 1. 实现惰性删除
    // 2. 实现立即删除（重建）
    // 3. 比较性能和召回率
    // 4. 找到最佳压缩触发点
}
```

### 练习3: 内存管理优化

```cpp
void exercise_3_memory_management() {
    // 1. 实现内存池
    // 2. 实现LRU缓存
    // 3. 测试多索引场景
    // 4. 优化内存使用
}
```

### 练习4: 分段索引合并

```cpp
void exercise_4_segment_merge() {
    // 1. 实现分段索引
    // 2. 实现合并策略
    // 3. 测试合并性能
    // 4. 优化合并算法
}
```

---

## 总结

第24天深入探讨了动态索引的更新与维护，涵盖：

1. **动态索引挑战**：
   - 性能衰减问题
   - 碎片化
   - 设计权衡

2. **增量构建**：
   - 分段索引策略
   - 增量IVF实现
   - 在线训练

3. **删除操作**：
   - 惰性删除
   - 实时删除（HNSW）
   - 过滤策略

4. **索引重组**：
   - 压缩策略
   - 倒排表重平衡
   - 优化触发机制

5. **内存管理**：
   - 内存池
   - 资源释放
   - LRU缓存

**关键要点**：
- 动态操作需要在性能和一致性之间权衡
- 惰性删除简单但会导致碎片化
- 分段索引适合高插入速率场景
- 定期压缩对于长期运行的系统至关重要
- 内存管理需要考虑多索引场景
- 倒排表不平衡会显著影响性能

## 后续学习

- 研究LSM树等数据结构在向量检索中的应用
- 学习增量学习算法（在线k-means）
- 实践生产环境的索引维护策略
- 研究向量检索的事务处理

---

## 16. Faiss动态索引底层实现详解

### 16.1 IndexIVF的动态添加

```cpp
// faiss/IndexIVF.cpp
// IndexIVF的add_with_ids实现

void IndexIVF::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids) {

    // 1. 确保量化器已训练
    if (!is_trained) {
        train(n, x);
    }

    // 2. 找到每个向量所属的倒排列表
    std::vector<idx_t> idx(n);
    quantizer->assign(n, x, idx.data());

    // 3. 为每个向量分配ID
    std::vector<idx_t> ids(n);
    if (xids) {
        // 使用用户提供的ID
        for (idx_t i = 0; i < n; i++) {
            ids[i] = xids[i];
        }
    } else {
        // 自动生成连续ID
        for (idx_t i = 0; i < n; i++) {
            ids[i] = ntotal + i;
        }
    }

    // 4. 编码向量
    std::vector<uint8_t> codes(n * code_size);
    encode_vectors(n, x, idx.data(), codes.data());

    // 5. 添加到倒排列表
    add_core(n, x, ids.data(), idx.data(), codes.data());

    // 6. 更新ntotal
    ntotal += n;
}

// add_core: 将向量添加到倒排列表
void IndexIVF::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* precomputed_idx,
        const uint8_t* codes) {

    // 按照list_no分组
    std::vector<size_t> offsets(nlist + 1, 0);
    for (idx_t i = 0; i < n; i++) {
        idx_t list_no = precomputed_idx[i];
        if (list_no >= 0 && list_no < nlist) {
            offsets[list_no + 1]++;
        }
    }

    // 计算累积偏移
    for (size_t i = 0; i < nlist; i++) {
        offsets[i + 1] += offsets[i];
    }

    // 添加到invlists
    invlists->add_entries_with_ids(
        n,
        xids,
        codes,
        precomputed_idx,
        offsets.data());
}
```

### 16.2 InvertedLists的动态添加

```cpp
// faiss/invlists/InvertedLists.cpp

// ArrayInvertedLists的add_entries实现
size_t ArrayInvertedLists::add_entries(
        size_t list_no,
        size_t n,
        const idx_t* xids,
        const uint8_t* xcode) {

    // 1. 确保有足够空间
    size_t o = ids[list_no].size();
    ids[list_no].resize(o + n);
    codes[list_no].resize(o + n * code_size);

    // 2. 复制ID
    memcpy(ids[list_no].data() + o, xids, n * sizeof(idx_t));

    // 3. 复制编码
    memcpy(codes[list_no].data() + o * code_size,
           xcode, n * code_size);

    return o;
}
```

### 16.3 删除操作实现

```cpp
// faiss/IndexIVF.cpp
// remove_ids实现

size_t IndexIVF::remove_ids(const IDSelector& sel) {

    // 1. 遍历所有倒排列表
    size_t n_removed = 0;

    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);

        // 2. 获取列表内容和IDs
        const idx_t* ids = invlists->get_ids(list_no);
        const uint8_t* codes = invlists->get_codes(list_no);

        // 3. 找出需要保留的向量
        std::vector<idx_t> ids_to_keep;
        std::vector<const uint8_t*> codes_to_keep;

        for (size_t i = 0; i < list_size; i++) {
            idx_t id = ids[i];
            if (!sel.is_member(id)) {
                // 保留这个向量
                ids_to_keep.push_back(id);
                codes_to_keep.push_back(codes + i * code_size);
            } else {
                n_removed++;
            }
        }

        // 4. 重构倒排列表(只保留未删除的)
        invlists->resize(list_no, ids_to_keep.size());

        if (!ids_to_keep.empty()) {
            invlists->update_entries(
                list_no, 0, ids_to_keep.size(),
                ids_to_keep.data(),
                (const uint8_t*)codes_to_keep.data());
        }
    }

    // 5. 更新ntotal
    ntotal -= n_removed;

    return n_removed;
}
```

### 16.4 增量训练

```cpp
// 增量训练IVF量化器

class IncrementalIVFTrainer {
    Index* quantizer;
    size_t nlist;

    bool trained;
    size_t num_vectors_trained;

public:
    IncrementalIVFTrainer(size_t d, size_t nlist)
        : nlist(nlist), trained(false), num_vectors_trained(0) {
        quantizer = new IndexFlatL2(d);
    }

    // 增量训练
    void incremental_train(size_t n, const float* x) {
        if (!trained) {
            // 首次训练: 运行k-means
            Clustering clus(quantizer->d, nlist);
            clus.train(n, x);
            quantizer->add(nlist, clus.centroids);
            trained = true;
        } else {
            // 增量更新质心
            update_centroids(n, x);
        }

        num_vectors_trained += n;
    }

private:
    void update_centroids(size_t n, const float* x) {
        // 获取当前质心
        float* centroids = new float[nlist * quantizer->d];
        // ... 从quantizer中获取质心 ...

        // 找到每个向量最近的质心
        std::vector<idx_t> assign(n);
        quantizer->assign(n, x, assign.data());

        // 计算新的质心(移动平均)
        std::vector<float> new_centroids(nlist * quantizer->d, 0.0f);
        std::vector<size_t> counts(nlist, 0);

        for (size_t i = 0; i < n; i++) {
            idx_t list_no = assign[i];
            const float* xi = x + i * quantizer->d;
            float* cent = new_centroids.data() + list_no * quantizer->d;

            for (size_t j = 0; j < quantizer->d; j++) {
                cent[j] += xi[j];
            }
            counts[list_no]++;
        }

        // 归一化并与旧质心混合
        float alpha = 0.1f;  // 学习率

        for (size_t i = 0; i < nlist; i++) {
            float* old_cent = centroids + i * quantizer->d;
            float* new_cent = new_centroids.data() + i * quantizer->d;

            if (counts[i] > 0) {
                for (size_t j = 0; j < quantizer->d; j++) {
                    new_cent[j] /= counts[i];
                    // 指数移动平均
                    old_cent[j] = (1 - alpha) * old_cent[j] + alpha * new_cent[j];
                }
            }
        }

        // 更新量化器
        delete[] centroids;
    }
};
```

### 16.5 倒排表重平衡

```cpp
// 倒排表重平衡操作

class InvertedListBalancer {
    IndexIVF* index;

public:
    InvertedListBalancer(IndexIVF* idx) : index(idx) {}

    // 重平衡倒排列表
    void rebalance() {
        size_t nlist = index->nlist;
        InvertedLists* invlists = index->invlists;

        // 1. 收集所有向量
        std::vector<std::vector<float>> all_vectors(nlist);
        std::vector<std::vector<idx_t>> all_ids(nlist);

        for (size_t i = 0; i < nlist; i++) {
            size_t list_size = invlists->list_size(i);
            const uint8_t* codes = invlists->get_codes(i);
            const idx_t* ids = invlists->get_ids(i);

            // 解码向量
            all_vectors[i].reserve(list_size * index->d);
            all_ids[i].reserve(list_size);

            for (size_t j = 0; j < list_size; j++) {
                // 解码并存储
                // ...
            }
        }

        // 2. 重新训练量化器
        std::vector<float> all_vectors_flat;
        std::vector<idx_t> all_ids_flat;

        for (size_t i = 0; i < nlist; i++) {
            all_vectors_flat.insert(all_vectors_flat.end(),
                all_vectors[i].begin(), all_vectors[i].end());
            all_ids_flat.insert(all_ids_flat.end(),
                all_ids[i].begin(), all_ids[i].end());
        }

        // 重新训练
        index->quantizer->train(all_vectors_flat.size(), all_vectors_flat.data());

        // 3. 重新分配向量
        // 清空原有invlists
        for (size_t i = 0; i < nlist; i++) {
            invlists->resize(i, 0);
        }

        // 重新添加
        index->add(all_ids_flat.size(), all_vectors_flat.data(), all_ids_flat.data());
    }
};
```

### 16.6 内存池实现

```cpp
// 内存池实现,减少频繁分配

template <typename T>
class MemoryPool {
    struct Block {
        T* data;
        size_t capacity;
        size_t used;
    };

    std::vector<Block> blocks;
    size_t block_size;

public:
    MemoryPool(size_t block_size = 1024)
        : block_size(block_size) {}

    ~MemoryPool() {
        for (auto& block : blocks) {
            free(block.data);
        }
    }

    T* allocate(size_t n) {
        // 在当前block中寻找空间
        if (!blocks.empty()) {
            Block& last = blocks.back();
            if (last.used + n <= last.capacity) {
                T* ptr = last.data + last.used;
                last.used += n;
                return ptr;
            }
        }

        // 需要新的block
        size_t new_capacity = std::max(block_size, n);
        Block new_block;
        new_block.data = (T*)malloc(new_capacity * sizeof(T));
        new_block.capacity = new_capacity;
        new_block.used = n;

        blocks.push_back(new_block);
        return new_block.data;
    }

    void reset() {
        for (auto& block : blocks) {
            block.used = 0;
        }
    }
};

// 在IndexIVF中使用内存池
class IndexIVFWithMemoryPool : public IndexIVF {
    MemoryPool<uint8_t> code_pool;
    MemoryPool<idx_t> id_pool;

public:
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override {
        // 使用内存池而不是直接分配
        // ...
    }
};
```

### 16.7 直接映射管理

```cpp
// faiss/IndexIVF.cpp
// DirectMap用于快速定位向量位置

struct DirectMap {
    enum Type {
        None,      // 无直接映射
        Array,     // 数组映射
        Hashtable  // 哈希表映射
    };

    Type type;

    // Array实现(适用于连续ID)
    std::vector<std::pair<idx_t, idx_t>> array;  // (list_no, offset)

    // Hashtable实现(适用于任意ID)
    std::unordered_map<idx_t, std::pair<idx_t, idx_t>> map;

    // 查找向量位置
    bool get(idx_t id, idx_t& list_no, idx_t& offset) const {
        if (type == Array) {
            if (id >= 0 && id < array.size()) {
                list_no = array[id].first;
                offset = array[id].second;
                return true;
            }
            return false;
        } else if (type == Hashtable) {
            auto it = map.find(id);
            if (it != map.end()) {
                list_no = it->second.first;
                offset = it->second.second;
                return true;
            }
            return false;
        }
        return false;
    }

    // 添加向量映射
    void add(idx_t id, idx_t list_no, idx_t offset) {
        if (type == Array) {
            if (id >= array.size()) {
                array.resize(id + 1);
            }
            array[id] = {list_no, offset};
        } else if (type == Hashtable) {
            map[id] = {list_no, offset};
        }
    }

    // 删除向量映射
    bool remove(idx_t id) {
        if (type == Array) {
            if (id >= 0 && id < array.size()) {
                array[id] = {-1, -1};
                return true;
            }
            return false;
        } else if (type == Hashtable) {
            return map.erase(id) > 0;
        }
        return false;
    }
};
```

### 16.8 并发安全的添加

```cpp
// 并发安全的索引添加

class ConcurrentIndexIVF : public IndexIVF {
    std::vector<std::mutex> list_mutexes;  // 每个list一个锁

public:
    ConcurrentIndexIVF(size_t nlist, size_t d)
        : IndexIVF(nlist, d) {
        list_mutexes.resize(nlist);
    }

    void add_with_ids(
            idx_t n,
            const float* x,
            const idx_t* xids) override {

        // 1. 分配向量到list
        std::vector<idx_t> idx(n);
        quantizer->assign(n, x, idx.data());

        // 2. 编码向量
        std::vector<uint8_t> codes(n * code_size);
        encode_vectors(n, x, idx.data(), codes.data());

        // 3. 按list分组
        std::vector<std::vector<idx_t>> ids_by_list(nlist);
        std::vector<std::vector<const uint8_t*>> codes_by_list(nlist);
        std::vector<size_t> offsets_by_list(nlist + 1, 0);

        for (idx_t i = 0; i < n; i++) {
            idx_t list_no = idx[i];
            if (list_no >= 0 && list_no < nlist) {
                ids_by_list[list_no].push_back(xids ? xids[i] : ntotal + i);
                codes_by_list[list_no].push_back(codes.data() + i * code_size);
            }
        }

        // 4. 并发添加到每个list
        #pragma omp parallel for schedule(dynamic)
        for (size_t list_no = 0; list_no < nlist; list_no++) {
            if (!ids_by_list[list_no].empty()) {
                std::lock_guard<std::mutex> lock(list_mutexes[list_no]);

                invlists->add_entries(
                    list_no,
                    ids_by_list[list_no].size(),
                    ids_by_list[list_no].data(),
                    (const uint8_t*)codes_by_list[list_no].data());
            }
        }

        ntotal += n;
    }
};
```

### 16.9 动态索引最佳实践

```cpp
// 动态索引的最佳实践配置

struct DynamicIndexConfig {
    // 添加配置
    size_t batch_size = 1000;           // 批量添加大小
    bool use_concurrent_add = true;     // 使用并发添加

    // 删除配置
    bool use_lazy_deletion = true;      // 使用惰性删除
    size_t compaction_threshold = 10000; // 删除多少个后触发压缩

    // 重平衡配置
    size_t rebalance_interval = 100000; // 每添加多少个向量后重平衡
    double imbalance_threshold = 2.0;   // 不平衡阈值

    // 内存配置
    size_t max_memory_mb = 8192;        // 最大内存限制
    bool enable_memory_pool = true;     // 启用内存池
};

DynamicIndexConfig get_optimal_config(
        double insert_rate_per_sec,
        double delete_rate_per_sec,
        double query_rate_per_sec) {

    DynamicIndexConfig config;

    double total_rate = insert_rate_per_sec + delete_rate_per_sec + query_rate_per_sec;

    if (total_rate < 100) {
        // 低负载: 简单配置
        config.use_concurrent_add = false;
        config.use_lazy_deletion = false;
    } else if (total_rate < 10000) {
        // 中等负载
        config.batch_size = 5000;
        config.use_concurrent_add = true;
        config.use_lazy_deletion = true;
    } else {
        // 高负载: 激进优化
        config.batch_size = 10000;
        config.use_concurrent_add = true;
        config.use_lazy_deletion = true;
        config.compaction_threshold = 1000;
        config.rebalance_interval = 50000;
    }

    return config;
}
```

这些底层实现细节展示了Faiss如何高效地处理动态索引更新,包括向量添加、删除、重平衡和并发控制等关键操作。
