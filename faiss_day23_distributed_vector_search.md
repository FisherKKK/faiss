# Faiss深度学习课程 - 第23天：分布式向量检索

## 课程概述

第23天探讨当数据规模超过单机容量时的分布式向量检索方案。本课程从理论到实践，全面解析分布式系统的架构、一致性、负载均衡和容错机制。

## 学习目标

- 理解分布式向量检索的架构设计
- 掌握数据分片策略
- 学习分布式查询处理与结果合并
- 理解一致性与可用性权衡
- 掌握负载均衡与容错机制
- 实践大规模分布式检索系统

---

## 第一部分：分布式架构设计

### 1.1 系统架构概览

```cpp
// 分布式向量检索系统架构
class DistributedVectorSearchArchitecture {
public:
    // 节点角色定义
    enum class NodeRole {
        COORDINATOR,    // 协调节点（路由、聚合）
        DATA_NODE,      // 数据节点（存储向量）
        MASTER          // 主节点（元数据管理）
    };

    // 集群配置
    struct ClusterConfig {
        int num_data_nodes;
        int num_coordinators;
        int replication_factor;
        std::string consensus_protocol;  // Raft, Paxos

        void print() const {
            printf("Cluster Configuration:\n");
            printf("  Data nodes: %d\n", num_data_nodes);
            printf("  Coordinators: %d\n", num_coordinators);
            printf("  Replication factor: %d\n", replication_factor);
            printf("  Consensus: %s\n", consensus_protocol.c_str());
        }
    };

    // 系统组件
    struct ClusterComponents {
        // 数据分片管理
        std::unique_ptr<ShardManager> shard_manager;

        // 路由表
        std::unique_ptr<RoutingTable> routing_table;

        // 负载均衡器
        std::unique_ptr<LoadBalancer> load_balancer;

        // 一致性管理器
        std::unique_ptr<ConsistencyManager> consistency_mgr;

        // 容错管理器
        std::unique_ptr<FaultToleranceManager> fault_tolerance_mgr;
    };
};
```

### 1.2 数据分片策略

```cpp
// 分片策略基类
class ShardStrategy {
public:
    virtual ~ShardStrategy() = default;

    // 计算向量所属的分片
    virtual int get_shard_id(const float* vector, size_t d) = 0;

    // 获取查询需要访问的分片列表
    virtual std::vector<int> get_shards_for_query(
        const float* query, size_t d, int nprobe) = 0;

    // 分片元数据
    struct ShardInfo {
        int shard_id;
        std::vector<std::string> replica_addresses;
        size_t num_vectors;
        size_t capacity;
        double load_factor;

        void print() const {
            printf("Shard %d: %zu vectors (%.1f%% full)\n",
                   shard_id, num_vectors, load_factor * 100);
        }
    };

    virtual std::vector<ShardInfo> get_shard_info() const = 0;
};

// 策略1: 基于哈希的分片
class HashBasedSharding : public ShardStrategy {
    int num_shards;
    std::hash<std::string> hasher;

public:
    HashBasedSharding(int n) : num_shards(n) {}

    int get_shard_id(const float* vector, size_t d) override {
        // 将向量序列化为字符串并哈希
        std::string vec_str(reinterpret_cast<const char*>(vector),
                           d * sizeof(float));
        return static_cast<int>(hasher(vec_str) % num_shards);
    }

    std::vector<int> get_shards_for_query(
        const float* query, size_t d, int nprobe) override {

        // 哈希分片需要查询所有分片
        std::vector<int> shards(num_shards);
        std::iota(shards.begin(), shards.end(), 0);
        return shards;
    }

    std::vector<ShardInfo> get_shard_info() const override {
        std::vector<ShardInfo> info;
        for (int i = 0; i < num_shards; i++) {
            info.push_back({i, {"node-" + std::to_string(i)}, 0, 1000000, 0.0});
        }
        return info;
    }
};

// 策略2: 基于空间分片的IVF
class SpaceBasedSharding : public ShardStrategy {
    int num_shards;
    size_t d;
    std::vector<std::vector<float>> centroids;  // 每个分片的质心
    std::vector<std::vector<int>> voronoi_cells; // Voronoi单元

public:
    SpaceBasedSharding(int n, size_t dim,
                      const float* centroid_data)
        : num_shards(n), d(dim) {

        centroids.resize(n);
        for (int i = 0; i < n; i++) {
            centroids[i].assign(centroid_data + i * d,
                               centroid_data + (i + 1) * d);
        }
    }

    int get_shard_id(const float* vector, size_t d) override {
        // 找到最近的质心
        int nearest = 0;
        float min_dist = INFINITY;

        for (int i = 0; i < num_shards; i++) {
            float dist = compute_l2_distance(vector,
                                            centroids[i].data(), d);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = i;
            }
        }

        return nearest;
    }

    std::vector<int> get_shards_for_query(
        const float* query, size_t d, int nprobe) override {

        // 找到nprobe个最近的分片
        std::vector<std::pair<float, int>> distances;

        for (int i = 0; i < num_shards; i++) {
            float dist = compute_l2_distance(query,
                                            centroids[i].data(), d);
            distances.push_back({dist, i});
        }

        std::sort(distances.begin(), distances.end());

        std::vector<int> shards;
        for (int i = 0; i < std::min(nprobe, num_shards); i++) {
            shards.push_back(distances[i].second);
        }

        return shards;
    }

    std::vector<ShardInfo> get_shard_info() const override {
        std::vector<ShardInfo> info;
        for (int i = 0; i < num_shards; i++) {
            info.push_back({i, {"node-" + std::to_string(i)}, 0, 1000000, 0.0});
        }
        return info;
    }

private:
    float compute_l2_distance(const float* x, const float* y, size_t d) const {
        float sum = 0.0f;
        for (size_t i = 0; i < d; i++) {
            float diff = x[i] - y[i];
            sum += diff * diff;
        }
        return sum;
    }
};

// 策略3: 混合分片（空间+哈希）
class HybridSharding : public ShardStrategy {
    int num_space_shards;   // 空间分片数
    int num_hash_shards;    // 每个空间分片的哈希子分片数
    size_t d;
    std::unique_ptr<SpaceBasedSharding> space_sharding;
    std::vector<std::unique_ptr<HashBasedSharding>> hash_shardings;

public:
    HybridSharding(int space_shards, int hash_per_space,
                  size_t dim, const float* centroids)
        : num_space_shards(space_shards),
          num_hash_shards(hash_per_space),
          d(dim) {

        space_sharding = std::make_unique<SpaceBasedSharding>(
            space_shards, dim, centroids);

        for (int i = 0; i < space_shards; i++) {
            hash_shardings.push_back(
                std::make_unique<HashBasedSharding>(hash_per_space));
        }
    }

    int get_shard_id(const float* vector, size_t d) override {
        // 先找到空间分片
        int space_shard = space_sharding->get_shard_id(vector, d);

        // 在空间分片内进行哈希
        int hash_shard = hash_shardings[space_shard]->get_shard_id(vector, d);

        return space_shard * num_hash_shards + hash_shard;
    }

    std::vector<int> get_shards_for_query(
        const float* query, size_t d, int nprobe) override {

        auto space_shards = space_sharding->get_shards_for_query(
            query, d, nprobe);

        std::vector<int> shards;
        for (int space_shard : space_shards) {
            // 每个空间分片内的所有哈希分片都需要访问
            for (int h = 0; h < num_hash_shards; h++) {
                shards.push_back(space_shard * num_hash_shards + h);
            }
        }

        return shards;
    }

    std::vector<ShardInfo> get_shard_info() const override {
        std::vector<ShardInfo> info;
        int total_shards = num_space_shards * num_hash_shards;

        for (int i = 0; i < total_shards; i++) {
            info.push_back({i, {"node-" + std::to_string(i)}, 0, 1000000, 0.0});
        }

        return info;
    }
};
```

---

## 第二部分：分布式查询处理

### 2.1 查询路由与分发

```cpp
// 分布式查询处理器
class DistributedQueryProcessor {
    std::unique_ptr<ShardStrategy> shard_strategy;
    std::unique_ptr<RoutingTable> routing_table;
    int nprobe;  // 探测的分片数

public:
    DistributedQueryProcessor(std::unique_ptr<ShardStrategy> strategy,
                             int nprobe_val)
        : shard_strategy(std::move(strategy)),
          nprobe(nprobe_val) {

        routing_table = std::make_unique<RoutingTable>();
    }

    // 查询请求
    struct SearchRequest {
        std::vector<float> query;   // 查询向量
        size_t k;                    // Top-K
        int timeout_ms;
        uint64_t request_id;
    };

    // 单分片查询结果
    struct ShardResult {
        int shard_id;
        std::vector<float> distances;
        std::vector<int64_t> labels;
        double latency_ms;
        bool success;

        bool is_valid() const {
            return success &&
                   distances.size() == labels.size() &&
                   !distances.empty();
        }
    };

    // 最终聚合结果
    struct AggregateResult {
        std::vector<float> distances;
        std::vector<int64_t> labels;
        double total_latency_ms;
        std::vector<int> shards_queried;
        int successful_shards;
        int failed_shards;

        void print() const {
            printf("Results from %d/%d shards (%.2f ms)\n",
                   successful_shards,
                   successful_shards + failed_shards,
                   total_latency_ms);
            printf("Top results:\n");
            size_t n = std::min(size_t(5), distances.size());
            for (size_t i = 0; i < n; i++) {
                printf("  [%zu] ID=%ld, dist=%.4f\n",
                       i, labels[i], distances[i]);
            }
        }
    };

    // 执行分布式查询
    AggregateResult search(const SearchRequest& req) {
        auto start = std::chrono::high_resolution_clock::now();

        // 1. 确定要查询的分片
        std::vector<int> shard_ids = shard_strategy->get_shards_for_query(
            req.query.data(), req.query.size(), nprobe);

        // 2. 并行查询各个分片
        std::vector<ShardResult> shard_results =
            query_shards_parallel(req, shard_ids);

        // 3. 聚合结果
        AggregateResult result = aggregate_results(req.k, shard_results);

        auto end = std::chrono::high_resolution_clock::now();
        result.total_latency_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        result.shards_queried = shard_ids;

        return result;
    }

private:
    // 并行查询多个分片
    std::vector<ShardResult> query_shards_parallel(
        const SearchRequest& req,
        const std::vector<int>& shard_ids) {

        std::vector<ShardResult> results(shard_ids.size());

        // 使用线程池并行查询
        #pragma omp parallel for
        for (size_t i = 0; i < shard_ids.size(); i++) {
            results[i] = query_single_shard(req, shard_ids[i]);
        }

        return results;
    }

    // 查询单个分片
    ShardResult query_single_shard(const SearchRequest& req, int shard_id) {
        ShardResult result;
        result.shard_id = shard_id;

        auto start = std::chrono::high_resolution_clock::now();

        try {
            // 获取分片地址
            std::string shard_address = routing_table->get_shard_address(shard_id);

            // 发送RPC请求
            // 实际实现中使用gRPC/Thrift等
            auto response = send_rpc_request(shard_address, req);

            result.distances = response.distances;
            result.labels = response.labels;
            result.success = true;

        } catch (const std::exception& e) {
            result.success = false;
            printf("Error querying shard %d: %s\n", shard_id, e.what());
        }

        auto end = std::chrono::high_resolution_clock::now();
        result.latency_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }

    // 聚合多个分片的结果
    AggregateResult aggregate_results(
        size_t k,
        const std::vector<ShardResult>& shard_results) {

        AggregateResult result;
        result.successful_shards = 0;
        result.failed_shards = 0;

        // 使用最大堆合并结果
        using HeapElement = std::tuple<float, int64_t, int>; // distance, label, shard_id
        std::priority_queue<HeapElement, std::vector<HeapElement>,
                           std::greater<HeapElement>> min_heap;

        for (const auto& shard_res : shard_results) {
            if (!shard_res.is_valid()) {
                result.failed_shards++;
                continue;
            }

            result.successful_shards++;

            // 将分片结果加入堆
            for (size_t i = 0; i < shard_res.distances.size(); i++) {
                min_heap.push({
                    shard_res.distances[i],
                    shard_res.labels[i],
                    shard_res.shard_id
                });
            }
        }

        // 提取Top-K
        while (!min_heap.empty() && result.distances.size() < k) {
            auto [dist, label, shard_id] = min_heap.top();
            min_heap.pop();

            result.distances.push_back(dist);
            result.labels.push_back(label);
        }

        return result;
    }

    // 模拟RPC请求
    struct RPCResponse {
        std::vector<float> distances;
        std::vector<int64_t> labels;
    };

    RPCResponse send_rpc_request(const std::string& address,
                                 const SearchRequest& req) {
        // 实际实现中：
        // 1. 序列化请求
        // 2. 发送RPC（gRPC/Thrift）
        // 3. 反序列化响应

        // 这里返回模拟数据
        RPCResponse response;
        response.distances = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        response.labels = {100, 200, 300, 400, 500};
        return response;
    }
};
```

### 2.2 结果聚合与去重

```cpp
// 高级结果聚合器
class ResultAggregator {
public:
    // 聚合策略
    enum class AggregateStrategy {
        MERGE_HEAP,      // 使用堆合并
        SORT_MERGE,      // 排序后合并
        TOP_K_PER_SHARD  // 每个分片返回Top-K再合并
    };

    // 去重策略
    enum class DedupStrategy {
        NONE,           // 不去重
        BY_LABEL,       // 按标签去重（保留最近的）
        BY_DISTANCE,    // 按距离阈值去重
        HYBRID          // 混合策略
    };

    // 聚合配置
    struct AggregateConfig {
        size_t top_k;
        AggregateStrategy strategy;
        DedupStrategy dedup;
        float distance_threshold;  // 用于BY_DISTANCE策略

        static AggregateConfig default_config() {
            return {10, AggregateStrategy::MERGE_HEAP,
                   DedupStrategy::BY_LABEL, 0.01f};
        }
    };

    // 执行聚合
    static DistributedQueryProcessor::AggregateResult aggregate(
        const std::vector<DistributedQueryProcessor::ShardResult>& shard_results,
        const AggregateConfig& config) {

        DistributedQueryProcessor::AggregateResult result;

        // 根据策略选择聚合方法
        switch (config.strategy) {
            case AggregateStrategy::MERGE_HEAP:
                result = aggregate_by_heap(shard_results, config);
                break;
            case AggregateStrategy::SORT_MERGE:
                result = aggregate_by_sort_merge(shard_results, config);
                break;
            case AggregateStrategy::TOP_K_PER_SHARD:
                result = aggregate_top_k_per_shard(shard_results, config);
                break;
        }

        // 应用去重
        if (config.dedup != DedupStrategy::NONE) {
            apply_dedup(result, config.dedup, config.distance_threshold);
        }

        return result;
    }

private:
    // 使用堆聚合
    static DistributedQueryProcessor::AggregateResult aggregate_by_heap(
        const std::vector<DistributedQueryProcessor::ShardResult>& shard_results,
        const AggregateConfig& config) {

        using HeapElement = std::tuple<float, int64_t, int>;
        std::priority_queue<HeapElement, std::vector<HeapElement>,
                           std::greater<HeapElement>> min_heap;

        // 收集所有结果
        for (const auto& shard_res : shard_results) {
            if (!shard_res.is_valid()) continue;

            for (size_t i = 0; i < shard_res.distances.size(); i++) {
                min_heap.push({
                    shard_res.distances[i],
                    shard_res.labels[i],
                    shard_res.shard_id
                });
            }
        }

        // 提取Top-K
        DistributedQueryProcessor::AggregateResult result;
        while (!min_heap.empty() && result.distances.size() < config.top_k) {
            auto [dist, label, shard_id] = min_heap.top();
            min_heap.pop();

            result.distances.push_back(dist);
            result.labels.push_back(label);
        }

        return result;
    }

    // 排序合并
    static DistributedQueryProcessor::AggregateResult aggregate_by_sort_merge(
        const std::vector<DistributedQueryProcessor::ShardResult>& shard_results,
        const AggregateConfig& config) {

        // 收集所有结果到vector
        std::vector<std::tuple<float, int64_t, int>> all_results;

        for (const auto& shard_res : shard_results) {
            if (!shard_res.is_valid()) continue;

            for (size_t i = 0; i < shard_res.distances.size(); i++) {
                all_results.push_back({
                    shard_res.distances[i],
                    shard_res.labels[i],
                    shard_res.shard_id
                });
            }
        }

        // 排序
        std::sort(all_results.begin(), all_results.end());

        // 提取Top-K
        DistributedQueryProcessor::AggregateResult result;
        size_t n = std::min(config.top_k, all_results.size());

        for (size_t i = 0; i < n; i++) {
            result.distances.push_back(std::get<0>(all_results[i]));
            result.labels.push_back(std::get<1>(all_results[i]));
        }

        return result;
    }

    // 每个分片Top-K再聚合
    static DistributedQueryProcessor::AggregateResult aggregate_top_k_per_shard(
        const std::vector<DistributedQueryProcessor::ShardResult>& shard_results,
        const AggregateConfig& config) {

        // 每个分片已经返回了Top-K，直接聚合
        return aggregate_by_heap(shard_results, config);
    }

    // 应用去重
    static void apply_dedup(
        DistributedQueryProcessor::AggregateResult& result,
        DedupStrategy strategy,
        float distance_threshold) {

        switch (strategy) {
            case DedupStrategy::BY_LABEL:
                dedup_by_label(result);
                break;
            case DedupStrategy::BY_DISTANCE:
                dedup_by_distance(result, distance_threshold);
                break;
            case DedupStrategy::HYBRID:
                dedup_by_label(result);
                dedup_by_distance(result, distance_threshold);
                break;
            default:
                break;
        }
    }

    // 按标签去重
    static void dedup_by_label(
        DistributedQueryProcessor::AggregateResult& result) {

        std::unordered_map<int64_t, size_t> label_to_index;
        std::vector<float> unique_distances;
        std::vector<int64_t> unique_labels;

        for (size_t i = 0; i < result.labels.size(); i++) {
            int64_t label = result.labels[i];

            auto it = label_to_index.find(label);
            if (it == label_to_index.end()) {
                // 第一次出现
                label_to_index[label] = unique_labels.size();
                unique_distances.push_back(result.distances[i]);
                unique_labels.push_back(label);
            } else {
                // 已存在，保留距离更近的
                if (result.distances[i] < unique_distances[it->second]) {
                    unique_distances[it->second] = result.distances[i];
                }
            }
        }

        result.distances = std::move(unique_distances);
        result.labels = std::move(unique_labels);
    }

    // 按距离阈值去重
    static void dedup_by_distance(
        DistributedQueryProcessor::AggregateResult& result,
        float threshold) {

        std::vector<bool> keep(result.distances.size(), true);

        for (size_t i = 0; i < result.distances.size(); i++) {
            if (!keep[i]) continue;

            for (size_t j = i + 1; j < result.distances.size(); j++) {
                if (!keep[j]) continue;

                // 如果距离很近，认为重复
                if (std::abs(result.distances[i] - result.distances[j]) < threshold) {
                    keep[j] = false;  // 保留i，去除j
                }
            }
        }

        std::vector<float> filtered_distances;
        std::vector<int64_t> filtered_labels;

        for (size_t i = 0; i < keep.size(); i++) {
            if (keep[i]) {
                filtered_distances.push_back(result.distances[i]);
                filtered_labels.push_back(result.labels[i]);
            }
        }

        result.distances = std::move(filtered_distances);
        result.labels = std::move(filtered_labels);
    }
};
```

---

## 第三部分：一致性管理

### 3.1 数据一致性模型

```cpp
// 一致性管理器
class ConsistencyManager {
public:
    // 一致性级别
    enum class ConsistencyLevel {
        STRONG,         // 强一致性
        EVENTUAL,       // 最终一致性
        QUORUM,         // 法定人数一致
        SESSION,        // 会话一致性
        MONOTONIC       // 单调读
    };

    // 写操作配置
    struct WriteConfig {
        ConsistencyLevel level;
        int replication_factor;
        int min_acks;  // 最少确认数（用于QUORUM）

        static WriteConfig for_strong_consistency(int rf = 3) {
            return {ConsistencyLevel::STRONG, rf, rf};
        }

        static WriteConfig for_quorum(int rf = 3) {
            return {ConsistencyLevel::QUORUM, rf, rf / 2 + 1};
        }

        static WriteConfig for_eventual(int rf = 3) {
            return {ConsistencyLevel::EVENTUAL, rf, 1};
        }
    };

    // 读操作配置
    struct ReadConfig {
        ConsistencyLevel level;
        int min_read_replicas;  // 最少读取副本数

        static ReadConfig for_strong_consistency() {
            return {ConsistencyLevel::STRONG, 1};  // 从主节点读
        }

        static ReadConfig for_quorum(int rf = 3) {
            return {ConsistencyLevel::QUORUM, rf / 2 + 1};
        }

        static ReadConfig for_eventual() {
            return {ConsistencyLevel::EVENTUAL, 1};  // 从任意副本读
        }
    };

    // 添加向量（带一致性保证）
    bool add_vectors(
        const float* vectors,
        size_t n,
        size_t d,
        const WriteConfig& config) {

        switch (config.level) {
            case ConsistencyLevel::STRONG:
                return add_with_strong_consistency(vectors, n, d, config);

            case ConsistencyLevel::QUORUM:
                return add_with_quorum(vectors, n, d, config);

            case ConsistencyLevel::EVENTUAL:
                return add_with_eventual_consistency(vectors, n, d, config);

            default:
                return false;
        }
    }

    // 搜索向量（带一致性保证）
    std::vector<std::pair<float, int64_t>> search_vectors(
        const float* query,
        size_t d,
        size_t k,
        const ReadConfig& config) {

        switch (config.level) {
            case ConsistencyLevel::STRONG:
                return search_with_strong_consistency(query, d, k, config);

            case ConsistencyLevel::QUORUM:
                return search_with_quorum(query, d, k, config);

            case ConsistencyLevel::EVENTUAL:
                return search_with_eventual_consistency(query, d, k, config);

            default:
                return {};
        }
    }

private:
    // 强一致性写入
    bool add_with_strong_consistency(
        const float* vectors,
        size_t n,
        size_t d,
        const WriteConfig& config) {

        // 使用两阶段提交（2PC）或分布式锁
        // 这里简化实现

        // 阶段1: 准备
        std::vector<std::string> participants = get_replicas(config.replication_factor);
        bool all_prepared = true;

        for (const auto& replica : participants) {
            if (!send_prepare(replica, vectors, n, d)) {
                all_prepared = false;
                break;
            }
        }

        // 阶段2: 提交或回滚
        if (all_prepared) {
            for (const auto& replica : participants) {
                send_commit(replica);
            }
            return true;
        } else {
            for (const auto& replica : participants) {
                send_rollback(replica);
            }
            return false;
        }
    }

    // Quorum写入
    bool add_with_quorum(
        const float* vectors,
        size_t n,
        size_t d,
        const WriteConfig& config) {

        std::vector<std::string> replicas = get_replicas(config.replication_factor);
        int acks = 0;

        // 并行写入所有副本
        #pragma omp parallel for
        for (size_t i = 0; i < replicas.size(); i++) {
            if (write_to_replica(replicas[i], vectors, n, d)) {
                #pragma omp atomic
                acks++;
            }
        }

        return acks >= config.min_acks;
    }

    // 最终一致性写入
    bool add_with_eventual_consistency(
        const float* vectors,
        size_t n,
        size_t d,
        const WriteConfig& config) {

        // 异步写入，不等待确认
        async_write_to_replicas(get_replicas(config.replication_factor),
                               vectors, n, d);
        return true;
    }

    // 强一致性读
    std::vector<std::pair<float, int64_t>> search_with_strong_consistency(
        const float* query,
        size_t d,
        size_t k,
        const ReadConfig& config) {

        // 从主节点读取
        std::string primary = get_primary_replica();
        return search_on_replica(primary, query, d, k);
    }

    // Quorum读
    std::vector<std::pair<float, int64_t>> search_with_quorum(
        const float* query,
        size_t d,
        size_t k,
        const ReadConfig& config) {

        auto replicas = get_replicas(config.min_read_replicas);
        std::vector<std::vector<std::pair<float, int64_t>>> partial_results;

        // 并行读取多个副本
        for (const auto& replica : replicas) {
            partial_results.push_back(
                search_on_replica(replica, query, d, k * 2)  // 读更多结果
            );
        }

        // 合并并去重
        return merge_and_dedup(partial_results, k);
    }

    // 最终一致性读
    std::vector<std::pair<float, int64_t>> search_with_eventual_consistency(
        const float* query,
        size_t d,
        size_t k,
        const ReadConfig& config) {

        // 从任意副本读取（可能是最近的）
        std::string replica = get_nearest_replica();
        return search_on_replica(replica, query, d, k);
    }

    // 辅助函数
    std::vector<std::string> get_replicas(int count) {
        // 返回副本地址列表
        std::vector<std::string> replicas;
        for (int i = 0; i < count; i++) {
            replicas.push_back("replica-" + std::to_string(i));
        }
        return replicas;
    }

    std::string get_primary_replica() {
        return "replica-0";  // 主节点
    }

    std::string get_nearest_replica() {
        return "replica-1";  // 简化实现
    }

    bool send_prepare(const std::string& replica,
                     const float* vectors, size_t n, size_t d) {
        // 发送准备请求
        return true;
    }

    void send_commit(const std::string& replica) {
        // 发送提交
    }

    void send_rollback(const std::string& replica) {
        // 发送回滚
    }

    bool write_to_replica(const std::string& replica,
                         const float* vectors, size_t n, size_t d) {
        // 写入副本
        return true;
    }

    void async_write_to_replicas(const std::vector<std::string>& replicas,
                                const float* vectors, size_t n, size_t d) {
        // 异步写入
    }

    std::vector<std::pair<float, int64_t>> search_on_replica(
        const std::string& replica,
        const float* query, size_t d, size_t k) {
        // 在副本上搜索
        return {{0.1f, 100}, {0.2f, 200}};
    }

    std::vector<std::pair<float, int64_t>> merge_and_dedup(
        const std::vector<std::vector<std::pair<float, int64_t>>>& partial_results,
        size_t k) {
        // 合并并去重
        return {{0.1f, 100}, {0.2f, 200}};
    }
};
```

### 3.2 向量时钟与冲突解决

```cpp
// 向量时钟实现
class VectorClock {
    std::unordered_map<std::string, uint64_t> clock;

public:
    // 事件发生（在本地节点）
    void increment(const std::string& node_id) {
        clock[node_id]++;
    }

    // 发送事件（携带向量时钟）
    VectorClock send(const std::string& node_id) {
        increment(node_id);
        return *this;
    }

    // 接收事件（合并向量时钟）
    void receive(const VectorClock& other, const std::string& node_id) {
        increment(node_id);

        // 合并：取每个节点的最大值
        for (const auto& [node, version] : other.clock) {
            clock[node] = std::max(clock[node], version);
        }
    }

    // 比较两个向量时钟的关系
    enum class Order {
        EQUAL,          // 相等
        BEFORE,         // this < other
        AFTER,          // this > other
        CONCURRENT      // 并发
    };

    Order compare(const VectorClock& other) const {
        bool this_before_other = false;
        bool other_before_this = false;

        // 检查所有节点
        std::set<std::string> all_nodes;
        for (const auto& [node, _] : clock) all_nodes.insert(node);
        for (const auto& [node, _] : other.clock) all_nodes.insert(node);

        for (const auto& node : all_nodes) {
            uint64_t this_ver = get_version(node);
            uint64_t other_ver = other.get_version(node);

            if (this_ver < other_ver) {
                this_before_other = true;
            } else if (this_ver > other_ver) {
                other_before_this = true;
            }
        }

        if (this_before_other && !other_before_this) {
            return Order::BEFORE;
        } else if (!this_before_other && other_before_this) {
            return Order::AFTER;
        } else if (!this_before_other && !other_before_this) {
            return Order::EQUAL;
        } else {
            return Order::CONCURRENT;
        }
    }

    uint64_t get_version(const std::string& node_id) const {
        auto it = clock.find(node_id);
        return (it != clock.end()) ? it->second : 0;
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "{";
        bool first = true;
        for (const auto& [node, version] : clock) {
            if (!first) ss << ", ";
            ss << node << ":" << version;
            first = false;
        }
        ss << "}";
        return ss.str()
    }
};

// 带版本控制的向量存储
class VersionedVectorStore {
    struct VectorVersion {
        std::vector<float> vector;
        VectorClock clock;
        uint64_t timestamp;
        bool deleted;  // 标记删除
    };

    std::unordered_map<int64_t, VectorVersion> data;
    std::string node_id;

public:
    VersionedVectorStore(const std::string& nid) : node_id(nid) {}

    // 添加向量（带版本）
    void add(int64_t id, const float* vector, size_t d,
             const VectorClock& incoming_clock) {

        auto it = data.find(id);

        if (it == data.end()) {
            // 新向量
            VectorClock new_clock = incoming_clock;
            new_clock.increment(node_id);

            data[id] = {std::vector<float>(vector, vector + d),
                       new_clock,
                       get_timestamp(),
                       false};
        } else {
            // 已存在，检查冲突
            VectorClock& existing_clock = it->second.clock;

            auto order = existing_clock.compare(incoming_clock);

            if (order == VectorClock::Order::BEFORE) {
                // 本地版本更旧，更新
                it->second.clock = incoming_clock;
                it->second.clock.increment(node_id);
                it->second.vector.assign(vector, vector + d);
                it->second.timestamp = get_timestamp();
            } else if (order == VectorClock::Order::CONCURRENT) {
                // 并发修改，需要解决冲突
                resolve_conflict(it->second, vector, d, incoming_clock);
            }
            // else: AFTER或EQUAL，忽略
        }
    }

    // 删除向量（带版本）
    void remove(int64_t id, const VectorClock& incoming_clock) {
        auto it = data.find(id);
        if (it != data.end()) {
            it->second.deleted = true;
            it->second.clock = incoming_clock;
            it->second.clock.increment(node_id);
        }
    }

private:
    // 冲突解决策略
    void resolve_conflict(VectorVersion& existing,
                         const float* new_vector, size_t d,
                         const VectorClock& new_clock) {
        // 策略1: Last-Write-Wins（基于时间戳）
        uint64_t new_timestamp = get_timestamp();
        if (new_timestamp > existing.timestamp) {
            existing.clock = new_clock;
            existing.clock.increment(node_id);
            existing.vector.assign(new_vector, new_vector + d);
            existing.timestamp = new_timestamp;
        }

        // 策略2: 应用特定（可根据业务逻辑）
        // 例如：保留距离更接近质心的向量
    }

    uint64_t get_timestamp() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};
```

---

## 第四部分：负载均衡

### 4.1 负载均衡策略

```cpp
// 负载均衡器
class LoadBalancer {
public:
    // 负载均衡策略
    enum class Strategy {
        ROUND_ROBIN,        // 轮询
        LEAST_CONNECTIONS,  // 最少连接
        WEIGHTED,           // 加权
        CONSISTENT_HASH,    // 一致性哈希
        ADAPTIVE            // 自适应
    };

    LoadBalancer(Strategy s) : strategy(s) {
        // 初始化节点
        for (int i = 0; i < 10; i++) {
            nodes.push_back({"node-" + std::to_string(i), 0, 1.0});
        }
    }

    struct Node {
        std::string address;
        int active_connections;
        double weight;  // 节点权重

        double load() const {
            return active_connections / weight;
        }
    };

    // 选择节点处理请求
    std::string select_node(const float* query = nullptr, size_t d = 0) {
        switch (strategy) {
            case Strategy::ROUND_ROBIN:
                return select_round_robin();

            case Strategy::LEAST_CONNECTIONS:
                return select_least_connections();

            case Strategy::WEIGHTED:
                return select_weighted();

            case Strategy::CONSISTENT_HASH:
                return select_consistent_hash(query, d);

            case Strategy::ADAPTIVE:
                return select_adaptive(query, d);

            default:
                return nodes[0].address;
        }
    }

    // 更新节点状态
    void update_node_status(const std::string& address, int delta_connections) {
        std::lock_guard<std::mutex> lock(mutex);

        for (auto& node : nodes) {
            if (node.address == address) {
                node.active_connections += delta_connections;
                break;
            }
        }
    }

    // 设置节点权重
    void set_node_weight(const std::string& address, double weight) {
        std::lock_guard<std::mutex> lock(mutex);

        for (auto& node : nodes) {
            if (node.address == address) {
                node.weight = weight;
                break;
            }
        }
    }

    void print_status() const {
        printf("Load Balancer Status (%s)\n", strategy_name().c_str());
        for (const auto& node : nodes) {
            printf("  %s: %d connections, weight=%.2f, load=%.2f\n",
                   node.address.c_str(),
                   node.active_connections,
                   node.weight,
                   node.load());
        }
    }

private:
    Strategy strategy;
    std::vector<Node> nodes;
    size_t round_robin_index = 0;
    std::mutex mutex;

    std::string strategy_name() const {
        switch (strategy) {
            case Strategy::ROUND_ROBIN: return "Round Robin";
            case Strategy::LEAST_CONNECTIONS: return "Least Connections";
            case Strategy::WEIGHTED: return "Weighted";
            case Strategy::CONSISTENT_HASH: return "Consistent Hash";
            case Strategy::ADAPTIVE: return "Adaptive";
            default: return "Unknown";
        }
    }

    std::string select_round_robin() {
        std::lock_guard<std::mutex> lock(mutex);

        size_t index = round_robin_index % nodes.size();
        round_robin_index++;

        return nodes[index].address;
    }

    std::string select_least_connections() {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = std::min_element(nodes.begin(), nodes.end(),
            [](const Node& a, const Node& b) {
                return a.load() < b.load();
            });

        return it->address;
    }

    std::string select_weighted() {
        std::lock_guard<std::mutex> lock(mutex);

        // 计算总权重
        double total_weight = 0.0;
        for (const auto& node : nodes) {
            total_weight += node.weight;
        }

        // 加权随机选择
        double r = (double)rand() / RAND_MAX * total_weight;
        double sum = 0.0;

        for (const auto& node : nodes) {
            sum += node.weight;
            if (sum >= r) {
                return node.address;
            }
        }

        return nodes.back().address;
    }

    std::string select_consistent_hash(const float* query, size_t d) {
        // 一致性哈希（简化）
        std::hash<std::string> hasher;
        std::string vec_str(reinterpret_cast<const char*>(query),
                           d * sizeof(float));
        size_t hash = hasher(vec_str);

        return nodes[hash % nodes.size()].address;
    }

    std::string select_adaptive(const float* query, size_t d) {
        // 自适应：根据历史延迟和负载选择
        // 简化实现：使用最少连接
        return select_least_connections();
    }
};
```

### 4.2 动态负载均衡

```cpp
// 自适应负载均衡器
class AdaptiveLoadBalancer {
public:
    // 节点性能指标
    struct NodeMetrics {
        std::string address;
        int active_connections;
        double avg_latency_ms;
        double cpu_usage;
        double memory_usage;
        uint64_t last_update;

        // 综合得分（越高越好）
        double score() const {
            double w1 = 0.3;  // 延迟权重
            double w2 = 0.3;  // CPU权重
            double w3 = 0.2;  // 内存权重
            double w4 = 0.2;  // 连接数权重

            double latency_score = 1.0 / (1.0 + avg_latency_ms);
            double cpu_score = 1.0 - cpu_usage;
            double memory_score = 1.0 - memory_usage;
            double connection_score = 1.0 / (1.0 + active_connections);

            return w1 * latency_score + w2 * cpu_score +
                   w3 * memory_score + w4 * connection_score;
        }
    };

    AdaptiveLoadBalancer() {
        // 初始化节点
        for (int i = 0; i < 10; i++) {
            node_metrics["node-" + std::to_string(i)] = {
                "node-" + std::to_string(i),
                0, 10.0, 0.3, 0.4,
                get_current_time()
            };
        }
    }

    // 选择最佳节点
    std::string select_best_node() {
        std::lock_guard<std::mutex> lock(mutex);

        std::string best_node;
        double best_score = -1.0;

        for (auto& [addr, metrics] : node_metrics) {
            double score = metrics.score();
            if (score > best_score) {
                best_score = score;
                best_node = addr;
            }
        }

        // 更新连接数
        node_metrics[best_node].active_connections++;

        return best_node;
    }

    // 报告查询完成（用于更新指标）
    void report_query_complete(const std::string& address,
                              double latency_ms) {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = node_metrics.find(address);
        if (it != node_metrics.end()) {
            it->second.active_connections--;

            // 指数移动平均（EMA）更新延迟
            double alpha = 0.2;
            it->second.avg_latency_ms =
                alpha * latency_ms +
                (1 - alpha) * it->second.avg_latency_ms;

            it->second.last_update = get_current_time();
        }
    }

    // 更新资源使用率
    void update_resource_usage(const std::string& address,
                              double cpu, double memory) {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = node_metrics.find(address);
        if (it != node_metrics.end()) {
            it->second.cpu_usage = cpu;
            it->second.memory_usage = memory;
            it->second.last_update = get_current_time();
        }
    }

    // 获取不健康节点
    std::vector<std::string> get_unhealthy_nodes(
        double max_latency = 100.0,
        double max_cpu = 0.9,
        double max_memory = 0.9) const {

        std::vector<std::string> unhealthy;

        for (const auto& [addr, metrics] : node_metrics) {
            if (metrics.avg_latency_ms > max_latency ||
                metrics.cpu_usage > max_cpu ||
                metrics.memory_usage > max_memory) {
                unhealthy.push_back(addr);
            }
        }

        return unhealthy;
    }

    void print_status() const {
        printf("Adaptive Load Balancer Status:\n");
        printf("Node        | Conn | Latency | CPU  | Mem  | Score\n");
        printf("------------|------|---------|------|------|-------\n");

        for (const auto& [addr, metrics] : node_metrics) {
            printf("%-11s | %4d | %7.2f | %4.1f | %4.1f | %5.2f\n",
                   addr.c_str(),
                   metrics.active_connections,
                   metrics.avg_latency_ms,
                   metrics.cpu_usage * 100,
                   metrics.memory_usage * 100,
                   metrics.score());
        }
    }

private:
    std::unordered_map<std::string, NodeMetrics> node_metrics;
    std::mutex mutex;

    uint64_t get_current_time() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};
```

---

## 第五部分：容错与恢复

### 5.1 副本管理

```cpp
// 副本管理器
class ReplicaManager {
public:
    // 副本状态
    enum class ReplicaStatus {
        HEALTHY,
        DEGRADED,
        FAILED
    };

    struct ReplicaInfo {
        std::string address;
        int shard_id;
        ReplicaStatus status;
        std::chrono::system_clock::time_point last_heartbeat;

        bool is_healthy() const {
            if (status != ReplicaStatus::HEALTHY) return false;

            auto now = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                now - last_heartbeat).count();

            return duration < 30;  // 30秒内有心跳
        }
    };

    ReplicaManager() {
        // 初始化副本
        for (int shard = 0; shard < 10; shard++) {
            for (int replica = 0; replica < 3; replica++) {
                std::string addr = "shard-" + std::to_string(shard) +
                                  "-replica-" + std::to_string(replica);
                replicas.push_back({
                    addr,
                    shard,
                    ReplicaStatus::HEALTHY,
                    std::chrono::system_clock::now()
                });
            }
        }
    }

    // 获取分片的所有副本
    std::vector<std::string> get_replicas_for_shard(int shard_id) {
        std::vector<std::string> addrs;

        for (const auto& replica : replicas) {
            if (replica.shard_id == shard_id && replica.is_healthy()) {
                addrs.push_back(replica.address);
            }
        }

        return addrs;
    }

    // 获取主副本
    std::string get_primary_replica(int shard_id) {
        for (const auto& replica : replicas) {
            if (replica.shard_id == shard_id &&
                replica.status == ReplicaStatus::HEALTHY) {
                // 第一个健康的副本作为主副本
                return replica.address;
            }
        }
        return "";  // 无可用副本
    }

    // 标记副本失败
    void mark_replica_failed(const std::string& address) {
        std::lock_guard<std::mutex> lock(mutex);

        for (auto& replica : replicas) {
            if (replica.address == address) {
                replica.status = ReplicaStatus::FAILED;
                printf("Marked replica %s as FAILED\n", address.c_str());

                // 触发重新复制
                trigger_replication(replica.shard_id);
                break;
            }
        }
    }

    // 更新心跳
    void update_heartbeat(const std::string& address) {
        std::lock_guard<std::mutex> lock(mutex);

        for (auto& replica : replicas) {
            if (replica.address == address) {
                replica.last_heartbeat = std::chrono::system_clock::now();

                if (replica.status == ReplicaStatus::FAILED) {
                    replica.status = ReplicaStatus::HEALTHY;
                    printf("Replica %s recovered\n", address.c_str());
                }
                break;
            }
        }
    }

    // 健康检查
    void health_check() {
        auto now = std::chrono::system_clock::now();

        for (auto& replica : replicas) {
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                now - replica.last_heartbeat).count();

            if (duration > 30 && replica.status == ReplicaStatus::HEALTHY) {
                printf("Replica %s timeout, marking as DEGRADED\n",
                       replica.address.c_str());
                replica.status = ReplicaStatus::DEGRADED;
            }
        }
    }

    void print_status() const {
        printf("Replica Status:\n");
        for (const auto& replica : replicas) {
            const char* status_str =
                (replica.status == ReplicaStatus::HEALTHY) ? "HEALTHY" :
                (replica.status == ReplicaStatus::DEGRADED) ? "DEGRADED" : "FAILED";

            printf("  %s: %s\n", replica.address.c_str(), status_str);
        }
    }

private:
    std::vector<ReplicaInfo> replicas;
    std::mutex mutex;

    void trigger_replication(int shard_id) {
        printf("Triggering replication for shard %d\n", shard_id);
        // 实现复制逻辑
    }
};
```

### 5.2 故障检测与恢复

```cpp
// 故障检测器
class FailureDetector {
public:
    // 检测策略
    enum class DetectionStrategy {
        HEARTBEAT,        // 心跳检测
        PHI_ACCRUAL,      // φ累积故障检测器
        GOSSIP_PROTOCOL   // Gossip协议
    };

    FailureDetector(DetectionStrategy s) : strategy(s) {}

    // 节点状态
    struct NodeState {
        std::string address;
        bool is_alive;
        std::chrono::system_clock::time_point last_seen;
        double phi;  // φ值（用于PHI_ACCRUAL）

        void print() const {
            printf("Node %s: %s (phi=%.2f)\n",
                   address.c_str(),
                   is_alive ? "ALIVE" : "SUSPECTED",
                   phi);
        }
    };

    // 注册节点
    void register_node(const std::string& address) {
        std::lock_guard<std::mutex> lock(mutex);

        nodes[address] = {
            address,
            true,
            std::chrono::system_clock::now(),
            0.0
        };
    }

    // 报告心跳
    void report_heartbeat(const std::string& address) {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = nodes.find(address);
        if (it != nodes.end()) {
            it->second.last_seen = std::chrono::system_clock::now();

            if (!it->second.is_alive) {
                printf("Node %s is back\n", address.c_str());
                it->second.is_alive = true;
            }
        }
    }

    // 检测失败节点
    std::vector<std::string> detect_failures() {
        std::lock_guard<std::mutex> lock(mutex);

        std::vector<std::string> failed_nodes;
        auto now = std::chrono::system_clock::now();

        for (auto& [addr, state] : nodes) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - state.last_seen).count();

            switch (strategy) {
                case DetectionStrategy::HEARTBEAT:
                    // 简单超时检测
                    if (duration > heartbeat_timeout_ms && state.is_alive) {
                        state.is_alive = false;
                        failed_nodes.push_back(addr);
                        printf("Node %s suspected failure (timeout)\n", addr.c_str());
                    }
                    break;

                case DetectionStrategy::PHI_ACCRUAL:
                    // φ值计算（简化）
                    state.phi = compute_phi(duration);

                    if (state.phi > phi_threshold && state.is_alive) {
                        state.is_alive = false;
                        failed_nodes.push_back(addr);
                        printf("Node %s suspected failure (phi=%.2f)\n",
                               addr.c_str(), state.phi);
                    }
                    break;

                default:
                    break;
            }
        }

        return failed_nodes;
    }

    void print_status() const {
        printf("Failure Detector Status (%s):\n", strategy_name().c_str());
        for (const auto& [addr, state] : nodes) {
            state.print();
        }
    }

private:
    DetectionStrategy strategy;
    std::unordered_map<std::string, NodeState> nodes;
    std::mutex mutex;

    // 心跳超时（毫秒）
    const long heartbeat_timeout_ms = 10000;
    // φ阈值
    const double phi_threshold = 8.0;

    std::string strategy_name() const {
        switch (strategy) {
            case DetectionStrategy::HEARTBEAT: return "Heartbeat";
            case DetectionStrategy::PHI_ACCRUAL: return "Phi Accrual";
            case DetectionStrategy::GOSSIP_PROTOCOL: return "Gossip";
            default: return "Unknown";
        }
    }

    // 计算φ值（简化实现）
    double compute_phi(long duration_ms) const {
        // 实际实现需要维护到达时间的分布
        // 这里使用简化公式
        double expected_interval_ms = 5000.0;  // 期望心跳间隔
        double deviation = duration_ms / expected_interval_ms;
        return deviation * deviation;
    }
};
```

---

## 第六部分：性能优化

### 6.1 查询优化

```cpp
// 分布式查询优化器
class DistributedQueryOptimizer {
public:
    // 查询计划
    struct QueryPlan {
        std::vector<int> shards_to_query;
        bool use_parallel;
        int max_parallelism;
        bool use_cache;
        bool use_prefetch;

        void print() const {
            printf("Query Plan:\n");
            printf("  Shards: ");
            for (int shard : shards_to_query) {
                printf("%d ", shard);
            }
            printf("\n");
            printf("  Parallel: %s (max=%d)\n",
                   use_parallel ? "Yes" : "No", max_parallelism);
            printf("  Cache: %s\n", use_cache ? "Yes" : "No");
            printf("  Prefetch: %s\n", use_prefetch ? "Yes" : "No");
        }
    };

    // 优化查询
    QueryPlan optimize_query(
        const float* query,
        size_t d,
        const std::vector<int>& candidate_shards,
        const QueryHistory& history) {

        QueryPlan plan;

        // 1. 选择最优分片子集
        plan.shards_to_query = select_optimal_shards(
            query, d, candidate_shards, history);

        // 2. 决定并行度
        plan.use_parallel = plan.shards_to_query.size() > 1;
        plan.max_parallelism = std::min(
            static_cast<int>(plan.shards_to_query.size()),
            get_available_threads());

        // 3. 决定是否使用缓存
        plan.use_cache = should_use_cache(query, d, history);

        // 4. 决定是否预取
        plan.use_prefetch = should_prefetch(plan.shards_to_query.size());

        return plan;
    }

private:
    struct QueryHistory {
        // 查询历史记录
        std::vector<std::vector<float>> recent_queries;

        bool has_similar_query(const float* query, size_t d) const {
            // 检查是否有相似的最近查询
            for (const auto& q : recent_queries) {
                if (q.size() == d) {
                    float dist = compute_distance(query, q.data(), d);
                    if (dist < 0.01f) {  // 非常相似
                        return true;
                    }
                }
            }
            return false;
        }

        void add_query(const float* query, size_t d) {
            recent_queries.emplace_back(query, query + d);
            if (recent_queries.size() > 100) {
                recent_queries.erase(recent_queries.begin());
            }
        }

        float compute_distance(const float* x, const float* y, size_t d) const {
            float sum = 0.0f;
            for (size_t i = 0; i < d; i++) {
                float diff = x[i] - y[i];
                sum += diff * diff;
            }
            return std::sqrt(sum);
        }
    };

    std::vector<int> select_optimal_shards(
        const float* query,
        size_t d,
        const std::vector<int>& candidates,
        const QueryHistory& history) {

        // 基于查询历史优化分片选择
        // 简化实现：返回所有候选分片
        return candidates;
    }

    bool should_use_cache(const float* query, size_t d,
                         const QueryHistory& history) {
        // 如果有相似的最近查询，使用缓存
        return history.has_similar_query(query, d);
    }

    bool should_prefetch(size_t num_shards) {
        // 如果分片数较多，启用预取
        return num_shards > 3;
    }

    int get_available_threads() {
        return std::thread::hardware_concurrency();
    }
};
```

### 6.2 批处理优化

```cpp
// 批处理查询优化
class BatchQueryOptimizer {
public:
    // 批处理查询
    struct BatchRequest {
        std::vector<std::vector<float>> queries;
        std::vector<size_t> top_ks;

        size_t size() const { return queries.size(); }
    };

    // 批处理结果
    struct BatchResult {
        std::vector<std::vector<float>> distances;
        std::vector<std::vector<int64_t>> labels;
        double total_time_ms;
    };

    // 优化批处理
    BatchResult process_batch(const BatchRequest& batch) {
        // 1. 按分片分组查询
        auto shards_to_queries = group_by_shard(batch);

        // 2. 并行处理每个分片
        std::map<int, std::vector<std::pair<float, int64_t>>> shard_results;

        #pragma omp parallel for
        for (size_t i = 0; i < shards_to_queries.size(); i++) {
            const auto& [shard_id, query_indices] = shards_to_queries[i];

            // 收集该分片的所有查询
            std::vector<const float*> shard_queries;
            for (size_t idx : query_indices) {
                shard_queries.push_back(batch.queries[idx].data());
            }

            // 批处理查询该分片
            auto results = batch_query_shard(shard_id, shard_queries);

            // 合并结果
            #pragma omp critical
            {
                shard_results[shard_id] = results;
            }
        }

        // 3. 聚合所有分片的结果
        return aggregate_batch_results(batch, shard_results);
    }

private:
    // 按分片分组查询
    std::vector<std::pair<int, std::vector<size_t>>>
    group_by_shard(const BatchRequest& batch) {

        std::map<int, std::vector<size_t>> groups;

        for (size_t i = 0; i < batch.queries.size(); i++) {
            int shard_id = determine_shard(batch.queries[i].data(),
                                          batch.queries[i].size());
            groups[shard_id].push_back(i);
        }

        return std::vector<std::pair<int, std::vector<size_t>>>(
            groups.begin(), groups.end());
    }

    int determine_shard(const float* query, size_t d) {
        // 简化实现
        return static_cast<int>(std::hash<std::string>{}(
            std::string(reinterpret_cast<const char*>(query), d * sizeof(float))
        ) % 10);
    }

    // 批处理查询单个分片
    std::vector<std::pair<float, int64_t>>
    batch_query_shard(int shard_id,
                     const std::vector<const float*>& queries) {

        // 实际实现中发送批处理RPC
        std::vector<std::pair<float, int64_t>> results;
        for (const auto* query : queries) {
            results.push_back({0.1f, 100});  // 模拟结果
        }
        return results;
    }

    // 聚合批处理结果
    BatchResult aggregate_batch_results(
        const BatchRequest& batch,
        const std::map<int, std::vector<std::pair<float, int64_t>>>& shard_results) {

        BatchResult aggregated;

        for (size_t i = 0; i < batch.queries.size(); i++) {
            // 收集该查询的所有分片结果
            std::vector<std::pair<float, int64_t>> all_results;

            for (const auto& [shard_id, results] : shard_results) {
                // 假设results[i]对应queries[i]
                if (i < results.size()) {
                    all_results.push_back(results[i]);
                }
            }

            // 排序并取Top-K
            std::sort(all_results.begin(), all_results.end());
            size_t k = batch.top_ks[i];

            std::vector<float> distances;
            std::vector<int64_t> labels;

            for (size_t j = 0; j < std::min(k, all_results.size()); j++) {
                distances.push_back(all_results[j].first);
                labels.push_back(all_results[j].second);
            }

            aggregated.distances.push_back(distances);
            aggregated.labels.push_back(labels);
        }

        return aggregated;
    }
};
```

---

## 实验练习

### 练习1: 实现简单的分布式向量检索

```cpp
void exercise_1_distributed_search() {
    // 1. 实现基于哈希的分片策略
    // 2. 实现2-3个节点的分布式搜索
    // 3. 测试查询延迟和吞吐量
    // 4. 对比单节点性能
}
```

### 练习2: 实现一致性管理

```cpp
void exercise_2_consistency() {
    // 1. 实现向量时钟
    // 2. 实现冲突检测与解决
    // 3. 测试不同一致性级别的性能差异
}
```

### 练习3: 实现故障恢复

```cpp
void exercise_3_fault_tolerance() {
    // 1. 实现心跳检测
    // 2. 模拟节点故障
    // 3. 测试自动恢复机制
}
```

### 练习4: 性能优化实验

```cpp
void exercise_4_optimization() {
    // 1. 实现查询缓存
    // 2. 实现批处理优化
    // 3. 测试优化效果
}
```

---

## 总结

第23天深入探讨了分布式向量检索，涵盖：

1. **架构设计**：
   - 系统组件与角色
   - 数据分片策略（哈希、空间、混合）

2. **查询处理**：
   - 查询路由与分发
   - 结果聚合与去重
   - 批处理优化

3. **一致性管理**：
   - 一致性级别（强一致、最终一致、Quorum）
   - 向量时钟与冲突解决

4. **负载均衡**：
   - 多种负载均衡策略
   - 自适应负载均衡
   - 动态权重调整

5. **容错机制**：
   - 副本管理
   - 故障检测
   - 自动恢复

6. **性能优化**：
   - 查询优化
   - 批处理
   - 缓存策略

**关键要点**：
- 分布式检索需要在一致性和可用性之间权衡
- 空间分片可以减少查询的分片数
- 向量时钟是处理并发更新的重要工具
- 自适应负载均衡能更好地处理不均衡负载
- 批处理可以显著提高吞吐量

## 后续学习

- 研究CAP定理在向量检索中的应用
- 学习分布式一致性算法（Raft、Paxos）
- 实践真实分布式系统部署
- 研究向量检索的serverless架构
