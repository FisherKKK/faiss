# Faiss深度学习课程 - 第22天：ANN算法复杂度分析

## 课程概述

第22天深入分析近似最近邻（ANN）算法的计算复杂度、空间复杂度和查询性能。本课程从理论分析到实际建模，帮助理解不同算法的权衡边界。

## 学习目标

- 掌握ANN算法的渐近复杂度分析
- 理解索引的内存占用模型
- 学习查询延迟的理论建模
- 掌握吞吐量-延迟权衡分析
- 理解可扩展性理论
- 实践性能预测与优化

---

## 第一部分：基础复杂度分析

### 1.1 精确搜索的复杂度下界

#### 线性扫描的复杂度

```cpp
// 线性扫描（暴力搜索）的复杂度分析
class LinearScanComplexity {
    size_t n;  // 数据库大小
    size_t d;  // 维度
    size_t nq; // 查询数量

public:
    LinearScanComplexity(size_t db_size, size_t dim, size_t n_queries)
        : n(db_size), d(dim), nq(n_queries) {}

    // 时间复杂度分析
    struct TimeComplexity {
        size_t flops_per_distance;  // 每次距离计算的FLOPs
        size_t distance_calculations;
        size_t total_flops;
        double theoretical_time_ms; // 基于峰值FLOPS的理论时间

        void print() const {
            printf("FLOPs per distance: %zu\n", flops_per_distance);
            printf("Distance calculations: %zu\n", distance_calculations);
            printf("Total FLOPs: %zu\n", total_flops);
            printf("Theoretical time: %.2f ms\n", theoretical_time_ms);
        }
    };

    TimeComplexity analyze_time_complexity(double peak_flops = 100.0) const {
        TimeComplexity tc;

        // L2距离: d次减法 + d次乘法 + (d-1)次加法
        tc.flops_per_distance = 3 * d - 1;

        // 总距离计算: nq * n
        tc.distance_calculations = nq * n;
        tc.total_flops = tc.flops_per_distance * tc.distance_calculations;

        // 理论时间（假设100 GFLOPS峰值性能）
        tc.theoretical_time_ms = (tc.total_flops / 1e9) / peak_flops * 1000.0;

        return tc;
    }

    // 内存访问分析
    struct MemoryComplexity {
        size_t database_bytes;
        size_t query_bytes;
        size_t total_bytes_read;
        double memory_bandwidth_gb;
        double bandwidth_bound_time_ms;

        void print() const {
            printf("Database size: %.2f MB\n", database_bytes / (1024.0*1024));
            printf("Total bytes read: %.2f GB\n", total_bytes_read / (1e9));
            printf("Memory bandwidth: %.2f GB/s\n", memory_bandwidth_gb);
            printf("Bandwidth-bound time: %.2f ms\n", bandwidth_bound_time_ms);
        }
    };

    MemoryComplexity analyze_memory_complexity(
        double bandwidth_gb_per_s = 50.0) const {

        MemoryComplexity mc;
        mc.database_bytes = n * d * sizeof(float);
        mc.query_bytes = nq * d * sizeof(float);
        mc.total_bytes_read = mc.database_bytes * nq + mc.query_bytes;

        mc.memory_bandwidth_gb = bandwidth_gb_per_s;
        mc.bandwidth_bound_time_ms =
            (mc.total_bytes_read / 1e9) / bandwidth_gb_per_s * 1000.0;

        return mc;
    }

    // 屋顶线模型分析
    void print_roofline_analysis() const {
        auto tc = analyze_time_complexity();
        auto mc = analyze_memory_complexity();

        printf("\n=== Roofline Model Analysis ===\n");
        printf("Computation-bound time: %.2f ms\n", tc.theoretical_time_ms);
        printf("Memory-bound time: %.2f ms\n", mc.bandwidth_bound_time_ms);

        // 算术强度 (Arithmetic Intensity)
        double arithmetic_intensity =
            static_cast<double>(tc.total_flops) / mc.total_bytes_read;

        printf("\nArithmetic Intensity: %.2f FLOPs/byte\n",
               arithmetic_intensity);

        // 判断是否受计算或内存限制
        if (tc.theoretical_time_ms > mc.bandwidth_bound_time_ms) {
            printf("=> COMPUTE-BOUND (needs more FLOPs)\n");
        } else {
            printf("=> MEMORY-BOUND (needs better cache reuse)\n");
        }
    }
};
```

#### 精确搜索的空间复杂度

```cpp
// 空间复杂度分析
class SpaceComplexityAnalyzer {
    size_t n;  // 数据库大小
    size_t d;  // 维度

public:
    SpaceComplexityAnalyzer(size_t db_size, size_t dim)
        : n(db_size), d(dim) {}

    struct IndexMemoryFootprint {
        size_t vectors_bytes;
        size_t metadata_bytes;
        size_t total_bytes;
        double total_mb;

        void print() const {
            printf("Vectors: %.2f MB\n", vectors_bytes / (1024.0*1024));
            printf("Metadata: %.2f MB\n", metadata_bytes / (1024.0*1024));
            printf("Total: %.2f MB\n", total_mb);
        }
    };

    IndexMemoryFootprint analyze_flat_index() const {
        IndexMemoryFootprint imf;

        // 原始向量
        imf.vectors_bytes = n * d * sizeof(float);

        // 元数据（ID、可选标签等）
        imf.metadata_bytes = n * sizeof(uint64_t);

        imf.total_bytes = imf.vectors_bytes + imf.metadata_bytes;
        imf.total_mb = imf.total_bytes / (1024.0 * 1024);

        return imf;
    }

    IndexMemoryFootprint analyze_ivf_index(
        size_t nlist, size_t nbits_per_code) const {

        IndexMemoryFootprint imf;

        // 1. 粗量化器质心
        imf.vectors_bytes += nlist * d * sizeof(float);

        // 2. 编码数据
        size_t code_size = nbits_per_code / 8;
        imf.vectors_bytes += n * code_size;

        // 3. 倒排表结构开销
        imf.metadata_bytes = nlist * sizeof(void*); // 指针数组
        imf.metadata_bytes += n * sizeof(uint32_t);  // 列表内ID

        // 4. 可选的向量ID
        imf.metadata_bytes += n * sizeof(uint64_t);

        imf.total_bytes = imf.vectors_bytes + imf.metadata_bytes;
        imf.total_mb = imf.total_bytes / (1024.0 * 1024);

        return imf;
    }

    IndexMemoryFootprint analyze_hnsw_index(
        size_t M,             // 每层连接数
        size_t ef_construction) const {

        IndexMemoryFootprint imf;

        // HNSW使用层次图
        // 第0层：每个节点约M个连接
        // 第l层：约n / (2^l)个节点，每个节点约M个连接

        // 估计总边数
        double ml = std::log2(static_cast<double>(n));
        size_t total_edges = 0;

        for (int l = 0; l < ml; l++) {
            size_t n_l = static_cast<size_t>(n / std::pow(2, l));
            size_t edges_per_node = (l == 0) ? M * 2 : M;
            total_edges += n_l * edges_per_node;
        }

        // 每条边存储：邻居ID + 浮点数距离（可选）
        imf.vectors_bytes = n * d * sizeof(float);  // 原始向量
        imf.metadata_bytes = total_edges * (sizeof(uint32_t) + sizeof(float));

        // 层信息
        imf.metadata_bytes += n * sizeof(uint8_t);  // 每节点的层分配

        imf.total_bytes = imf.vectors_bytes + imf.metadata_bytes;
        imf.total_mb = imf.total_bytes / (1024.0 * 1024);

        return imf;
    }

    // 打印对比表
    void print_comparison() const {
        printf("\n=== Memory Footprint Comparison (n=%zu, d=%zu) ===\n",
               n, d);

        auto flat = analyze_flat_index();
        printf("\nIndexFlat:\n");
        flat.print();

        auto ivf = analyze_ivf_index(100, 64);  // 100 lists, 64-bit PQ
        printf("\nIndexIVFPQ (100 lists, 64-bit):\n");
        ivf.print();

        auto hnsw = analyze_hnsw_index(16, 200);  // M=16
        printf("\nIndexHNSW (M=16):\n");
        hnsw.print();

        printf("\nCompression ratios:\n");
        printf("  IVFPQ: %.2fx\n", flat.total_mb / ivf.total_mb);
        printf("  HNSW:  %.2fx (vs vectors only)\n",
               flat.vectors_bytes / (double)hnsw.total_bytes);
    }
};
```

### 1.2 IVF系列的复杂度分析

```cpp
// IVF索引复杂度分析
class IVFComplexityAnalyzer {
    size_t n;      // 数据库大小
    size_t d;      // 维度
    size_t nlist;  // 倒排表数量
    size_t nprobe; // 探测的倒排表数

public:
    IVFComplexityAnalyzer(size_t db_size, size_t dim,
                         size_t n_lists, size_t n_probe)
        : n(db_size), d(dim), nlist(n_lists), nprobe(n_probe) {}

    struct QueryComplexity {
        size_t coarse_flops;      // 粗量化器FLOPs
        size_t fine_flops;        // 细搜索FLOPs
        size_t total_flops;
        double expected_probed_vectors;
        double speedup_vs_bruteforce;

        void print() const {
            printf("Coarse quantizer: %zu FLOPs\n", coarse_flops);
            printf("Fine search: %zu FLOPs\n", fine_flops);
            printf("Total: %zu FLOPs\n", total_flops);
            printf("Expected probed: %.0f vectors\n", expected_probed_vectors);
            printf("Speedup: %.2fx\n", speedup_vs_bruteforce);
        }
    };

    QueryComplexity analyze_single_query() const {
        QueryComplexity qc;

        // 粗量化器：计算到nlist个质心的距离
        // 假设使用IndexFlatL2作为粗量化器
        qc.coarse_flops = nlist * (3 * d - 1);

        // 期望探测的向量数
        double avg_list_size = static_cast<double>(n) / nlist;
        qc.expected_probed_vectors = avg_list_size * nprobe;

        // 细搜索：在探测的向量上计算距离
        size_t code_size = d;  // 假设PQ编码大小 = d (8x8bit PQ)
        qc.fine_flops = static_cast<size_t>(
            qc.expected_probed_vectors * code_size);

        qc.total_flops = qc.coarse_flops + qc.fine_flops;

        // 相比暴力搜索的加速比
        size_t brute_flops = n * (3 * d - 1);
        qc.speedup_vs_bruteforce =
            static_cast<double>(brute_flops) / qc.total_flops;

        return qc;
    }

    // 构建复杂度
    struct BuildComplexity {
        size_t training_flops;
        size_t assignment_flops;
        size_t encoding_flops;
        size_t total_flops;

        void print() const {
            printf("Training: %zu FLOPs\n", training_flops);
            printf("Assignment: %zu FLOPs\n", assignment_flops);
            printf("Encoding: %zu FLOPs\n", encoding_flops);
            printf("Total: %.2e GFLOPs\n", total_flops / 1e9);
        }
    };

    BuildComplexity analyze_build(
        size_t ntrain,          // 训练集大小
        size_t kmeans_iters) const {

        BuildComplexity bc;

        // 1. 训练粗量化器 (k-means)
        // 每次迭代: ntrain * nlist 次距离计算
        bc.training_flops = kmeans_iters * ntrain * nlist * (3 * d - 1);

        // 2. 分配向量到倒排表
        bc.assignment_flops = n * nlist * (3 * d - 1);

        // 3. 编码向量 (PQ)
        // 每个向量: M个子量化器，每个k次距离计算
        size_t M = d / 8;  // 假设
        size_t k = 256;    // 8-bit
        bc.encoding_flops = n * M * k * (3 * (d / M) - 1);

        bc.total_flops = bc.training_flops + bc.assignment_flops +
                        bc.encoding_flops;

        return bc;
    }
};
```

---

## 第二部分：HNSW复杂度分析

### 2.1 图结构的理论分析

```cpp
// HNSW复杂度模型
class HNSWComplexityModel {
    size_t n;      // 数据库大小
    size_t d;      // 维度
    size_t M;      // 每层最大连接数
    size_t ef_construction;
    size_t ef_search;

public:
    HNSWComplexityModel(size_t db_size, size_t dim,
                       size_t max_conn, size_t ef_c, size_t ef_s)
        : n(db_size), d(dim), M(max_conn),
          ef_construction(ef_c), ef_search(ef_s) {}

    // 搜索复杂度（期望）
    double expected_search_complexity() const {
        // HNSW搜索复杂度: O(log n) * (M * d)
        // 考虑efSearch参数

        double ml = std::log2(static_cast<double>(n));

        // 每层访问的节点数
        double visited_per_layer = ef_search;

        // 总距离计算
        double distance_computations = ml * visited_per_layer;

        // 每次距离计算的代价: O(d)
        double flops = distance_computations * (3 * d - 1);

        return flops;
    }

    // 构建复杂度
    double expected_build_complexity() const {
        // HNSW构建: O(n * log n * M * d)

        double ml = std::log2(static_cast<double>(n));

        // 每个插入需要:
        // 1. 找到插入位置: O(log n * ef_construction * d)
        double search_per_insert = ml * ef_construction * (3 * d - 1);

        // 2. 更新连接: O(M * d)
        double update_per_insert = M * (3 * d - 1);

        double total = n * (search_per_insert + update_per_insert);

        return total;
    }

    // 内存复杂度
    struct HNSWMemoryModel {
        size_t vector_storage;
        size_t graph_edges;
        size_t layer_data;
        size_t total;

        void print() const {
            printf("Vector storage: %.2f MB\n",
                   vector_storage / (1024.0*1024));
            printf("Graph edges: %.2f MB\n",
                   graph_edges / (1024.0*1024));
            printf("Layer data: %.2f MB\n",
                   layer_data / (1024.0*1024));
            printf("Total: %.2f MB\n",
                   total / (1024.0*1024));
        }
    };

    HNSWMemoryModel analyze_memory() const {
        HNSWMemoryModel hm;

        // 1. 向量存储
        hm.vector_storage = n * d * sizeof(float);

        // 2. 图边
        double ml = std::log2(static_cast<double>(n));
        size_t total_edges = 0;

        for (int l = 0; l < ml; l++) {
            size_t n_l = static_cast<size_t>(n / std::pow(2, l));
            size_t edges_per_node = (l == 0) ? M * 2 : M;
            total_edges += n_l * edges_per_node;
        }

        // 每条边: neighbor_id + optional distance
        hm.graph_edges = total_edges * (sizeof(uint32_t) + sizeof(float));

        // 3. 层数信息
        hm.layer_data = n * sizeof(uint8_t);  // 每节点所属层

        hm.total = hm.vector_storage + hm.graph_edges + hm.layer_data;

        return hm;
    }

    // 打印完整分析
    void print_analysis() const {
        printf("\n=== HNSW Complexity Analysis ===\n");
        printf("Configuration: n=%zu, d=%zu, M=%zu\n",
               n, d, M);
        printf("  ef_construction=%zu, ef_search=%zu\n\n",
               ef_construction, ef_search);

        printf("Query complexity: %.2e FLOPs\n",
               expected_search_complexity());
        printf("Build complexity: %.2e FLOPs\n\n",
               expected_build_complexity());

        auto mem = analyze_memory();
        mem.print();

        // 与暴力搜索对比
        double brute_flops = n * (3 * d - 1);
        double hnsw_flops = expected_search_complexity();
        printf("\nSpeedup vs brute force: %.2fx\n",
               brute_flops / hnsw_flops);
    }
};
```

### 2.2 参数对复杂度的影响

```cpp
// HNSW参数影响分析
class HNSWParameterSensitivity {
    size_t n;  // 数据库大小
    size_t d;  // 维度

public:
    HNSWParameterSensitivity(size_t db_size, size_t dim)
        : n(db_size), d(dim) {}

    // 分析M参数对性能的影响
    void analyze_M_parameter() const {
        printf("\n=== M Parameter Impact ===\n");
        printf("M | Speedup | Memory(MB) | Recall@10\n");
        printf("--|---------|------------|-----------\n");

        std::vector<size_t> M_values = {8, 16, 32, 64};

        for (size_t M : M_values) {
            HNSWComplexityModel model(n, d, M, 200, 50);

            // 估计召回率（简化模型）
            double recall = estimate_recall_for_M(M);

            // 计算加速比
            double search_flops = model.expected_search_complexity();
            double brute_flops = n * (3 * d - 1);
            double speedup = brute_flops / search_flops;

            // 内存占用
            auto mem = model.analyze_memory();
            double memory_mb = mem.total / (1024.0 * 1024);

            printf("%2zu | %6.2fx | %10.2f | %.4f\n",
                   M, speedup, memory_mb, recall);
        }
    }

    // 分析efSearch参数
    void analyze_ef_search() const {
        printf("\n=== efSearch Parameter Impact ===\n");
        printf("efSearch | Latency(ms) | Recall@10 | QPS\n");
        printf("---------|-------------|-----------|-----\n");

        size_t M = 16;
        std::vector<size_t> ef_values = {10, 20, 50, 100, 200, 500};

        for (size_t ef : ef_values) {
            HNSWComplexityModel model(n, d, M, 200, ef);

            // 估计延迟（假设50 GFLOPS峰值）
            double flops = model.expected_search_complexity();
            double latency_ms = (flops / 1e9) / 50.0 * 1000.0;

            // 估计召回率
            double recall = estimate_recall_for_ef(ef);

            // QPS（单线程）
            double qps = 1000.0 / latency_ms;

            printf("%8zu | %11.2f | %9.4f | %.0f\n",
                   ef, latency_ms, recall, qps);
        }
    }

    // M-召回率曲线建模
    double estimate_recall_for_M(size_t M) const {
        // 简化模型：recall ~ 1 - exp(-alpha * M)
        double alpha = 0.1;
        return 1.0 - std::exp(-alpha * M);
    }

    // ef-召回率曲线建模
    double estimate_recall_for_ef(size_t ef) const {
        // 简化模型：recall ~ ef / (ef + beta)
        double beta = 20.0;
        return ef / (ef + beta);
    }
};
```

---

## 第三部分：查询性能建模

### 3.1 延迟模型

```cpp
// 查询延迟分解模型
class QueryLatencyModel {
public:
    struct LatencyComponents {
        double cpu_time_ms;       // CPU计算时间
        double memory_time_ms;    // 内存访问时间
        double locking_time_ms;   // 并发控制开销
        double overhead_time_ms;  // 其他开销
        double total_ms;

        void print() const {
            printf("  CPU: %.2f ms (%.1f%%)\n",
                   cpu_time_ms, cpu_time_ms/total_ms*100);
            printf("  Memory: %.2f ms (%.1f%%)\n",
                   memory_time_ms, memory_time_ms/total_ms*100);
            printf("  Locking: %.2f ms (%.1f%%)\n",
                   locking_time_ms, locking_time_ms/total_ms*100);
            printf("  Overhead: %.2f ms (%.1f%%)\n",
                   overhead_time_ms, overhead_time_ms/total_ms*100);
            printf("  Total: %.2f ms\n", total_ms);
        }
    };

    // IVF搜索延迟模型
    static LatencyComponents model_ivf_latency(
        size_t d,             // 维度
        size_t nlist,         // 倒排表数
        size_t nprobe,        // 探测表数
        size_t vectors_probed,// 实际探测向量数
        size_t pq_m,          // PQ子空间数
        bool use_precomputed_table) {

        LatencyComponents lc;

        // 1. 粗量化器延迟
        double coarse_flops = nlist * (3 * d - 1);
        lc.cpu_time_ms = coarse_flops / 50e9 * 1000.0;  // 假设50 GFLOPS

        // 2. 细搜索延迟
        if (use_precomputed_table) {
            // 使用预计算表（ADC）
            lc.cpu_time_ms += vectors_probed * pq_m / 50e9 * 1000.0;
        } else {
            // 直接计算距离
            lc.cpu_time_ms += vectors_probed * (3 * d - 1) / 50e9 * 1000.0;
        }

        // 3. 内存延迟
        // 读取编码数据
        size_t bytes_read = vectors_probed * (pq_m);  // 假设8-bit PQ
        double memory_bandwidth = 50.0;  // GB/s
        lc.memory_time_ms = (bytes_read / 1e9) / memory_bandwidth * 1000.0;

        // 4. 锁开销（并发搜索）
        lc.locking_time_ms = nprobe * 0.001;  // 每个倒排表1微秒

        // 5. 其他开销
        lc.overhead_time_ms = 0.01;  // 10微秒

        lc.total_ms = lc.cpu_time_ms + lc.memory_time_ms +
                     lc.locking_time_ms + lc.overhead_time_ms;

        return lc;
    }

    // HNSW搜索延迟模型
    static LatencyComponents model_hnsw_latency(
        size_t d,
        size_t n,
        size_t M,
        size_t ef_search) {

        LatencyComponents lc;

        // 1. 图遍历计算
        double ml = std::log2(static_cast<double>(n));

        // 期望访问节点数
        double expected_visits = ml * ef_search;

        // 距离计算FLOPs
        double distance_flops = expected_visits * (3 * d - 1);
        lc.cpu_time_ms = distance_flops / 50e9 * 1000.0;

        // 2. 内存访问
        // 每个节点: 读取邻居列表 (M个ID)
        size_t bytes_read = static_cast<size_t>(expected_visits) * M *
                           sizeof(uint32_t);
        double memory_bandwidth = 50.0;
        lc.memory_time_ms = (bytes_read / 1e9) / memory_bandwidth * 1000.0;

        // 3. 锁开销（假设无锁实现）
        lc.locking_time_ms = 0.0;

        // 4. 其他开销
        lc.overhead_time_ms = 0.005;  // 5微秒

        lc.total_ms = lc.cpu_time_ms + lc.memory_time_ms +
                     lc.locking_time_ms + lc.overhead_time_ms;

        return lc;
    }

    // 打印对比分析
    static void print_comparison(
        size_t d, size_t n, size_t nlist, size_t nprobe) {

        printf("\n=== Query Latency Model Comparison ===\n");
        printf("Configuration: d=%zu, n=%zu, nlist=%zu, nprobe=%zu\n\n",
               d, n, nlist, nprobe);

        printf("IndexIVF (PQ, ADC):\n");
        auto ivf_lat = model_ivf_latency(d, nlist, nprobe,
                                        n/nlist*nprobe, d/8, true);
        ivf_lat.print();

        printf("\nIndexHNSW (M=16, ef=50):\n");
        auto hnsw_lat = model_hnsw_latency(d, n, 16, 50);
        hnsw_lat.print();

        printf("\nSpeedup: %.2fx\n",
               ivf_lat.total_ms / hnsw_lat.total_ms);
    }
};
```

### 3.2 吞吐量模型

```cpp
// 吞吐量分析
class ThroughputModel {
public:
    struct ThroughputMetrics {
        double single_thread_qps;    // 单线程QPS
        double multi_thread_qps;     // 多线程QPS
        double scalability_efficiency;  // 并行效率

        void print(size_t num_threads) const {
            printf("Single-thread QPS: %.0f\n", single_thread_qps);
            printf("%zu-thread QPS: %.0f\n", num_threads, multi_thread_qps);
            printf("Parallel efficiency: %.1f%%\n",
                   scalability_efficiency * 100);
        }
    };

    // 并行度分析
    static ThroughputMetrics analyze_scalability(
        double single_query_latency_ms,
        size_t num_threads,
        bool memory_bound) {

        ThroughputMetrics tm;

        // 单线程QPS
        tm.single_thread_qps = 1000.0 / single_query_latency_ms;

        // 多线程QPS（Amdahl定律）
        // 如果是内存受限，并行效率会较低
        double parallel_fraction = memory_bound ? 0.7 : 0.95;

        // 理论加速比
        double theoretical_speedup =
            1.0 / ((1.0 - parallel_fraction) + parallel_fraction / num_threads);

        // 实际加速比（考虑开销）
        double actual_speedup = theoretical_speedup * 0.9;

        tm.multi_thread_qps = tm.single_thread_qps * actual_speedup;
        tm.scalability_efficiency = actual_speedup / num_threads;

        return tm;
    }

    // 批处理优化
    static double batch_throughup_gain(
        double single_latency_ms,
        size_t batch_size,
        double kernel_launch_overhead_us = 10.0) {

        // 批处理降低固定开销摊销
        double batch_latency_ms =
            single_latency_ms * batch_size +
            kernel_launch_overhead_us / 1000.0;

        double single_total = single_latency_ms * batch_size;
        double speedup = single_total / batch_latency_ms;

        return speedup;
    }

    // 打印吞吐量分析
    static void print_analysis(
        double ivf_latency_ms,
        double hnsw_latency_ms) {

        printf("\n=== Throughput Analysis ===\n");

        std::vector<size_t> thread_counts = {1, 4, 8, 16, 32};

        printf("\nIndexIVF:\n");
        printf("Threads | QPS    | Efficiency\n");
        printf("--------|--------|------------\n");
        for (size_t t : thread_counts) {
            auto metrics = analyze_scalability(ivf_latency_ms, t, true);
            printf("%7zu | %6.0f | %9.1f%%\n", t,
                   metrics.multi_thread_qps,
                   metrics.scalability_efficiency * 100);
        }

        printf("\nIndexHNSW:\n");
        printf("Threads | QPS    | Efficiency\n");
        printf("--------|--------|------------\n");
        for (size_t t : thread_counts) {
            auto metrics = analyze_scalability(hnsw_latency_ms, t, false);
            printf("%7zu | %6.0f | %9.1f%%\n", t,
                   metrics.multi_thread_qps,
                   metrics.scalability_efficiency * 100);
        }

        // 批处理优化
        printf("\n=== Batch Processing Gain ===\n");
        printf("Batch Size | IVF Gain | HNSW Gain\n");
        printf("-----------|-----------|------------\n");
        for (size_t bs : {1, 4, 16, 32, 64, 128}) {
            double ivf_gain = batch_throughup_gain(ivf_latency_ms, bs);
            double hnsw_gain = batch_throughup_gain(hnsw_latency_ms, bs);
            printf("%10zu | %9.2fx | %10.2fx\n", bs, ivf_gain, hnsw_gain);
        }
    }
};
```

---

## 第四部分：可扩展性分析

### 4.1 数据规模可扩展性

```cpp
// 数据规模可扩展性模型
class ScalabilityModel {
public:
    // 算法复杂度类别
    enum class ComplexityClass {
        O_LOG_N,      // O(log n)
        O_N,          // O(n)
        O_N_LOG_N,    // O(n log n)
        O_N_SQRT,     // O(n^0.5)
        CUSTOM        // 自定义
    };

    struct ScalabilityPrediction {
        size_t current_n;
        size_t target_n;
        double current_latency_ms;
        double predicted_latency_ms;
        double acceptable_latency_ms;
        bool is_scalable;

        void print() const {
            printf("Current: n=%zu, latency=%.2f ms\n",
                   current_n, current_latency_ms);
            printf("Target: n=%zu, predicted latency=%.2f ms\n",
                   target_n, predicted_latency_ms);
            printf("Acceptable: %.2f ms\n", acceptable_latency_ms);
            printf("Scalable: %s\n", is_scalable ? "Yes" : "No");
        }
    };

    // 预测扩展性
    static ScalabilityPrediction predict_scalability(
        size_t current_n,
        double current_latency_ms,
        size_t target_n,
        ComplexityClass complexity,
        double acceptable_latency_ms = 100.0) {

        ScalabilityPrediction sp;
        sp.current_n = current_n;
        sp.target_n = target_n;
        sp.current_latency_ms = current_latency_ms;
        sp.acceptable_latency_ms = acceptable_latency_ms;

        // 计算预测延迟
        double ratio = static_cast<double>(target_n) / current_n;

        switch (complexity) {
            case ComplexityClass::O_LOG_N:
                sp.predicted_latency_ms = current_latency_ms *
                    (std::log(target_n) / std::log(current_n));
                break;

            case ComplexityClass::O_N:
                sp.predicted_latency_ms = current_latency_ms * ratio;
                break;

            case ComplexityClass::O_N_LOG_N:
                sp.predicted_latency_ms = current_latency_ms * ratio *
                    (std::log(target_n) / std::log(current_n));
                break;

            case ComplexityClass::O_N_SQRT:
                sp.predicted_latency_ms = current_latency_ms * std::sqrt(ratio);
                break;

            default:
                sp.predicted_latency_ms = current_latency_ms;
        }

        sp.is_scalable = sp.predicted_latency_ms <= sp.acceptable_latency_ms;

        return sp;
    }

    // 打印扩展性分析
    static void print_scalability_analysis(
        size_t current_n,
        double current_latency_ivf_ms,
        double current_latency_hnsw_ms) {

        printf("\n=== Scalability Analysis ===\n");
        printf("Current: n=%zu\n", current_n);
        printf("IVF latency: %.2f ms\n", current_latency_ivf_ms);
        printf("HNSW latency: %.2f ms\n\n", current_latency_hnsw_ms);

        std::vector<size_t> targets = {
            current_n * 10,
            current_n * 100,
            current_n * 1000
        };

        printf("IndexIVF (O(n^0.5) approx):\n");
        printf("Target size | Predicted Latency | Scalable\n");
        printf("------------|-------------------|----------\n");
        for (size_t target : targets) {
            auto pred = predict_scalability(
                current_n, current_latency_ivf_ms, target,
                ComplexityClass::O_N_SQRT, 100.0);
            printf("%11zu | %17.2f ms | %s\n",
                   target, pred.predicted_latency_ms,
                   pred.is_scalable ? "Yes" : "No");
        }

        printf("\nIndexHNSW (O(log n)):\n");
        printf("Target size | Predicted Latency | Scalable\n");
        printf("------------|-------------------|----------\n");
        for (size_t target : targets) {
            auto pred = predict_scalability(
                current_n, current_latency_hnsw_ms, target,
                ComplexityClass::O_LOG_N, 100.0);
            printf("%11zu | %17.2f ms | %s\n",
                   target, pred.predicted_latency_ms,
                   pred.is_scalable ? "Yes" : "No");
        }
    }
};
```

### 4.2 维度灾难分析

```cpp
// 维度灾难分析
class CurseOfDimensionality {
public:
    // 距离集中度分析
    static void analyze_distance_concentration() {
        printf("\n=== Distance Concentration in High Dimensions ===\n");

        std::vector<size_t> dimensions = {2, 8, 16, 32, 64, 128, 256, 512};

        printf("Dim | Ratio(max/min) | Contrast | Discriminative Power\n");
        printf("----|----------------|-----------|---------------------\n");

        for (size_t d : dimensions) {
            // 模拟高维分布
            size_t n = 10000;
            std::vector<float> distances(n);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, 1.0f);

            // 生成随机点并计算到原点的距离
            for (size_t i = 0; i < n; i++) {
                float sum_sq = 0.0f;
                for (size_t j = 0; j < d; j++) {
                    float val = dist(gen);
                    sum_sq += val * val;
                }
                distances[i] = std::sqrt(sum_sq);
            }

            // 分析距离分布
            float min_dist = *std::min_element(distances.begin(), distances.end());
            float max_dist = *std::max_element(distances.begin(), distances.end());

            // 计算对比度 (max/min) - 趋向于1
            float ratio = max_dist / min_dist;

            // 判别力估计
            float discriminative_power = 1.0f / (1.0f + ratio * 0.01f);

            printf("%3zu | %14.2f | %9.2f | %19.4f\n",
                   d, ratio, ratio, discriminative_power);
        }

        printf("\nConclusion: As d increases, distances become\n");
        printf("more concentrated, reducing discriminative power.\n");
    }

    // 所需样本数分析（Johnson-Lindenstrauss）
    static void analyze_jl_requirements() {
        printf("\n=== Johnson-Lindenstrauss Lemma Requirements ===\n");
        printf("To preserve distances with epsilon accuracy:\n\n");

        std::vector<size_t> n_values = {1000, 10000, 100000, 1000000};
        std::vector<float> epsilon_values = {0.1f, 0.05f, 0.01f};

        printf("n      | eps=0.1 | eps=0.05 | eps=0.01\n");
        printf("-------|---------|----------|----------\n");

        for (size_t n : n_values) {
            size_t k_01 = static_cast<size_t>(std::ceil(
                4 * std::log(n) / (0.1 * 0.1)));
            size_t k_005 = static_cast<size_t>(std::ceil(
                4 * std::log(n) / (0.05 * 0.05)));
            size_t k_001 = static_cast<size_t>(std::ceil(
                4 * std::log(n) / (0.01 * 0.01)));

            printf("%7zu | %7zu | %8zu | %8zu\n", n, k_01, k_005, k_001);
        }

        printf("\nKey insight: Target dimension k = O(log n / epsilon^2)\n");
    }

    // 量化误差随维度变化
    static void analyze_quantization_error_vs_dimension() {
        printf("\n=== Quantization Error vs Dimension ===\n");

        std::vector<size_t> dimensions = {16, 32, 64, 128, 256, 512};
        std::vector<size_t> bits_per_dim = {4, 8, 16};

        printf("Dim | 4-bit PQ | 8-bit PQ | 16-bit PQ\n");
        printf("----|----------|----------|-----------\n");

        for (size_t d : dimensions) {
            for (size_t bits : bits_per_dim) {
                // 简化模型: MSE ~ O(d / 2^(bits/M))
                // 其中M是子空间数
                size_t M = std::max(1UL, d / 16);
                float mse = static_cast<float>(d) / (1 << (bits / M));

                if (bits == 4) {
                    printf("%3zu | %8.4f", d, mse);
                } else if (bits == 8) {
                    printf(" | %8.4f", mse);
                } else {
                    printf(" | %9.4f\n", mse);
                }
            }
        }

        printf("\n");
    }
};
```

---

## 第五部分：缓存与内存层次优化

### 5.1 缓存效率分析

```cpp
// 缓存效率分析
class CacheEfficiencyAnalyzer {
public:
    struct CacheStatistics {
        size_t l1_hits;
        size_t l1_misses;
        size_t l2_hits;
        size_t l2_misses;
        size_t l3_hits;
        size_t l3_misses;

        double l1_hit_rate() const {
            return static_cast<double>(l1_hits) / (l1_hits + l1_misses);
        }

        double l2_hit_rate() const {
            return static_cast<double>(l2_hits) / (l2_hits + l2_misses);
        }

        double l3_hit_rate() const {
            return static_cast<double>(l3_hits) / (l3_hits + l3_misses);
        }

        void print() const {
            printf("L1: %.2f%% hits\n", l1_hit_rate() * 100);
            printf("L2: %.2f%% hits\n", l2_hit_rate() * 100);
            printf("L3: %.2f%% hits\n", l3_hit_rate() * 100);
        }
    };

    // 分析数据访问模式的缓存友好性
    static void analyze_access_pattern(
        size_t d,              // 维度
        size_t n,              // 数据库大小
        size_t cache_line_size = 64) {

        printf("\n=== Cache Efficiency Analysis ===\n");
        printf("Configuration: d=%zu, n=%zu\n\n", d, n);

        // L1缓存: 32 KB, 8路
        constexpr size_t L1_SIZE = 32 * 1024;
        constexpr size_t L1_LINE = 64;

        // L2缓存: 256 KB
        constexpr size_t L2_SIZE = 256 * 1024;

        // L3缓存: 8 MB
        constexpr size_t L3_SIZE = 8 * 1024 * 1024;

        // 分析向量大小
        size_t vector_bytes = d * sizeof(float);
        printf("Vector size: %zu bytes\n", vector_bytes);

        // 数据库大小
        size_t database_bytes = n * vector_bytes;
        printf("Database: %.2f MB\n", database_bytes / (1024.0*1024));

        // 可以放入L1缓存的向量数
        size_t vectors_in_l1 = L1_SIZE / vector_bytes;
        size_t vectors_in_l2 = L2_SIZE / vector_bytes;
        size_t vectors_in_l3 = L3_SIZE / vector_bytes;

        printf("\nCache capacity (in vectors):\n");
        printf("  L1: %zu vectors\n", vectors_in_l1);
        printf("  L2: %zu vectors\n", vectors_in_l2);
        printf("  L3: %zu vectors\n", vectors_in_l3);

        // 预取距离分析
        size_t prefetch_distance = L2_SIZE / vector_bytes;
        printf("\nSuggested prefetch distance: %zu vectors\n",
               prefetch_distance);

        // 数据布局建议
        printf("\nData layout recommendations:\n");

        if (vector_bytes > L1_LINE) {
            printf("  - Vector spans multiple cache lines\n");
            printf("  - Consider dimension shuffling for better cache use\n");
        }

        if (vectors_in_l1 < 10) {
            printf("  - Each query will cache-thrash L1\n");
            printf("  - Use blocking/software prefetching\n");
        }

        if (vectors_in_l2 < n) {
            double working_set_ratio = static_cast<double>(vectors_in_l2) / n;
            printf("  - L2 can hold %.1f%% of database\n",
                   working_set_ratio * 100);
            printf("  - Consider partitioning or sharding\n");
        }
    }

    // 内存带宽利用率
    static void analyze_bandwidth_utilization(
        size_t d,
        size_t n,
        double peak_bandwidth_gb_s = 50.0) {

        printf("\n=== Memory Bandwidth Analysis ===\n");

        // 顺序访问 vs 随机访问
        size_t vector_bytes = d * sizeof(float);

        // 顺序读取理论带宽
        double sequential_read_gb_s = peak_bandwidth_gb_s;

        // 随机读取实际带宽（考虑页表walk等）
        double random_read_gb_s = peak_bandwidth_gb_s * 0.3;

        printf("Sequential read bandwidth: %.1f GB/s\n", sequential_read_gb_s);
        printf("Random read bandwidth: %.1f GB/s\n\n", random_read_gb_s);

        // 扫描整个数据库的时间
        double database_gb = (n * vector_bytes) / 1e9;
        double sequential_time_ms = (database_gb / sequential_read_gb_s) * 1000;
        double random_time_ms = (database_gb / random_read_gb_s) * 1000;

        printf("Full database scan:\n");
        printf("  Sequential: %.2f ms\n", sequential_time_ms);
        printf("  Random: %.2f ms\n", random_time_ms);
        printf("  Ratio: %.2fx\n", random_time_ms / sequential_time_ms);
    }
};
```

### 5.2 NUMA感知优化

```cpp
// NUMA架构下的性能分析
class NUMAAnalyzer {
public:
    struct NUMAConfiguration {
        int num_nodes;
        int cpus_per_node;
        size_t memory_per_node;

        void print() const {
            printf("NUMA nodes: %d\n", num_nodes);
            printf("CPUs per node: %d\n", cpus_per_node);
            printf("Memory per node: %.2f GB\n",
                   memory_per_node / (1024.0*1024*1024));
        }
    };

    // 分析NUMA远程访问开销
    static void analyze_remote_access_penalty() {
        printf("\n=== NUMA Remote Access Penalty ===\n");

        // 典型延迟（纳秒）
        double local_latency_ns = 70.0;    // 本地内存访问
        double remote_latency_ns = 150.0;  // 远程内存访问

        double penalty_ratio = remote_latency_ns / local_latency_ns;

        printf("Local memory latency: %.0f ns\n", local_latency_ns);
        printf("Remote memory latency: %.0f ns\n", remote_latency_ns);
        printf("Penalty ratio: %.2fx\n\n", penalty_ratio);

        // 对查询性能的影响
        size_t d = 128;
        size_t vectors_accessed = 10000;

        // 假设每次查询访问vectors_accessed个向量
        // 每个向量d个float
        size_t bytes_per_vector = d * sizeof(float);
        size_t total_bytes = vectors_accessed * bytes_per_vector;

        // 计算延迟（简化）
        double local_time_ms =
            (vectors_accessed * local_latency_ns) / 1e6;
        double remote_time_ms =
            (vectors_accessed * remote_latency_ns) / 1e6;

        printf("Query accessing %zu vectors of %zu dimensions:\n",
               vectors_accessed, d);
        printf("  Local-only: %.2f ms\n", local_time_ms);
        printf("  50%% remote: %.2f ms\n",
               local_time_ms * 0.5 + remote_time_ms * 0.5);
        printf("  100%% remote: %.2f ms\n", remote_time_ms);
    }

    // NUMA感知的数据分布策略
    static void print_numa_distribution_strategies(
        size_t total_vectors,
        size_t d,
        int num_nodes) {

        printf("\n=== NUMA-Aware Distribution Strategies ===\n");
        printf("Total: %zu vectors of %zu dimensions\n", total_vectors, d);
        printf("NUMA nodes: %d\n\n", num_nodes);

        // 策略1: 均匀分片
        size_t vectors_per_node = total_vectors / num_nodes;
        printf("1. Uniform sharding:\n");
        for (int i = 0; i < num_nodes; i++) {
            printf("   Node %d: %zu vectors\n", i, vectors_per_node);
        }

        // 策略2: 按查询亲和性分组
        printf("\n2. Query affinity groups:\n");
        printf("   - Partition by expected query distribution\n");
        printf("   - Hot data on all nodes (replicated)\n");

        // 策略3: 混合
        printf("\n3. Hybrid approach:\n");
        printf("   - Hot subset: replicated\n");
        printf("   - Cold subset: sharded\n");
        printf("   - Example: 20%% hot, 80%% cold\n");
        size_t hot_vectors = total_vectors / 5;
        size_t cold_vectors = total_vectors - hot_vectors;
        printf("     Hot: %zu vectors (on all nodes)\n", hot_vectors);
        printf("     Cold: %zu vectors per node\n",
               cold_vectors / num_nodes);
    }
};
```

---

## 第六部分：综合性能模型

### 6.1 端到端性能预测

```cpp
// 端到端性能预测模型
class EndToEndPerformanceModel {
public:
    struct SystemConfig {
        double cpu_peak_gflops;
        double memory_bandwidth_gb_s;
        int num_cpu_cores;
        bool has_gpu;
        double gpu_peak_tflops;
    };

    struct Workload {
        size_t n;           // 数据库大小
        size_t d;           // 维度
        size_t nq;          // 查询数
        size_t k;           // Top-K
        double read_ratio;  // 读操作比例
    };

    struct PerformancePrediction {
        double latency_p50_ms;
        double latency_p95_ms;
        double latency_p99_ms;
        double qps;
        double recall;

        void print() const {
            printf("P50 latency: %.2f ms\n", latency_p50_ms);
            printf("P95 latency: %.2f ms\n", latency_p95_ms);
            printf("P99 latency: %.2f ms\n", latency_p99_ms);
            printf("QPS: %.0f\n", qps);
            printf("Recall@10: %.3f\n", recall);
        }
    };

    // 预测IVF性能
    static PerformancePrediction predict_ivf_performance(
        const SystemConfig& sys,
        const Workload& wl,
        size_t nlist,
        size_t nprobe,
        size_t pq_bits) {

        PerformancePrediction pred;

        // 计算单查询延迟
        double avg_list_size = static_cast<double>(wl.n) / nlist;
        double vectors_probed = avg_list_size * nprobe;

        // CPU时间（距离计算）
        double flops_per_query =
            nlist * (3 * wl.d - 1) +           // 粗量化器
            vectors_probed * (pq_bits / 8);     // PQ查找表

        double cpu_time_ms =
            (flops_per_query / 1e9) / sys.cpu_peak_gflops * 1000.0;

        // 内存时间
        size_t bytes_read = static_cast<size_t>(vectors_probed) * (pq_bits / 8);
        double memory_time_ms =
            (bytes_read / 1e9) / sys.memory_bandwidth_gb_s * 1000.0;

        // 基础延迟
        pred.latency_p50_ms = cpu_time_ms + memory_time_ms;

        // 考虑尾部延迟（队列效应、cache miss等）
        pred.latency_p95_ms = pred.latency_p50_ms * 1.5;
        pred.latency_p99_ms = pred.latency_p50_ms * 2.0;

        // QPS（多线程）
        double qps_single_thread = 1000.0 / pred.latency_p50_ms;
        pred.qps = qps_single_thread * sys.num_cpu_cores * 0.7;  // 并行效率70%

        // 召回率估计（简化）
        double coverage = static_cast<double>(nprobe) / nlist;
        pred.recall = std::min(0.98, coverage * 0.9);

        return pred;
    }

    // 预测HNSW性能
    static PerformancePrediction predict_hnsw_performance(
        const SystemConfig& sys,
        const Workload& wl,
        size_t M,
        size_t ef_search) {

        PerformancePrediction pred;

        // 期望访问节点数
        double ml = std::log2(static_cast<double>(wl.n));
        double nodes_visited = ml * ef_search;

        // CPU时间
        double flops_per_query = nodes_visited * (3 * wl.d - 1);
        double cpu_time_ms =
            (flops_per_query / 1e9) / sys.cpu_peak_gflops * 1000.0;

        // 内存时间（随机访问图结构）
        size_t edges_read = static_cast<size_t>(nodes_visited) * M;
        size_t bytes_read = edges_read * sizeof(uint32_t);
        // 随机访问效率较低
        double memory_time_ms =
            (bytes_read / 1e9) / (sys.memory_bandwidth_gb_s * 0.4) * 1000.0;

        pred.latency_p50_ms = cpu_time_ms + memory_time_ms;
        pred.latency_p95_ms = pred.latency_p50_ms * 1.3;  // HNSW延迟更稳定
        pred.latency_p99_ms = pred.latency_p50_ms * 1.6;

        double qps_single_thread = 1000.0 / pred.latency_p50_ms;
        pred.qps = qps_single_thread * sys.num_cpu_cores * 0.8;  // 并行效率80%

        // 召回率
        pred.recall = std::min(0.99, ef_search / (ef_search + 20.0));

        return pred;
    }

    // 打印对比分析
    static void print_comparison(
        const SystemConfig& sys,
        const Workload& wl) {

        printf("\n=== End-to-End Performance Prediction ===\n");
        printf("System: %.0f GFLOPS, %.0f GB/s, %d cores\n",
               sys.cpu_peak_gflops, sys.memory_bandwidth_gb_s, sys.num_cpu_cores);
        printf("Workload: n=%zu, d=%zu, nq=%zu, k=%zu\n\n",
               wl.n, wl.d, wl.nq, wl.k);

        printf("--- IndexIVF (nlist=100, nprobe=10, PQ=64bit) ---\n");
        auto ivf_pred = predict_ivf_performance(sys, wl, 100, 10, 64);
        ivf_pred.print();

        printf("\n--- IndexHNSW (M=16, ef=50) ---\n");
        auto hnsw_pred = predict_hnsw_performance(sys, wl, 16, 50);
        hnsw_pred.print();

        printf("\nComparison:\n");
        printf("  Latency ratio (IVF/HNSW): %.2fx\n",
               ivf_pred.latency_p50_ms / hnsw_pred.latency_p50_ms);
        printf("  QPS ratio (HNSW/IVF): %.2fx\n",
               hnsw_pred.qps / ivf_pred.qps);
        printf("  Recall difference: %.3f\n",
               hnsw_pred.recall - ivf_pred.recall);
    }
};
```

---

## 实验练习

### 练习1：复杂度验证实验

```cpp
void exercise_1_complexity_validation() {
    // 1. 在不同规模数据集上测量实际查询时间
    // 2. 绘制 n vs latency 曲线
    // 3. 拟合复杂度曲线，验证理论分析
    // 4. 对比IVF、HNSW、Flat的实际复杂度
}
```

### 练习2：延迟分解profiling

```cpp
void exercise_2_latency_breakdown() {
    // 1. 使用perf/VTune分解查询延迟
    // 2. 识别瓶颈（CPU、内存、锁）
    // 3. 验证模型预测的准确性
    // 4. 提出并实施优化措施
}
```

### 练习3：扩展性压力测试

```cpp
void exercise_3_scalability_stress_test() {
    // 1. 测试1M -> 10M -> 100M的扩展性
    // 2. 分析延迟增长曲线
    // 3. 识别扩展性瓶颈
    // 4. 设计分片方案以突破单机限制
}
```

### 练习4：NUMA优化实验

```cpp
void exercise_4_numa_optimization() {
    // 1. 在NUMA机器上测试不同数据分布
    // 2. 测量远程访问的实际开销
    // 3. 实现NUMA-aware的内存分配
    // 4. 评估优化效果
}
```

---

## 总结

第22天深入分析了ANN算法的复杂度，涵盖：

1. **基础复杂度**：
   - 线性扫描的复杂度下界
   - 空间复杂度分析
   - IVF系列的时间/空间复杂度

2. **HNSW复杂度**：
   - 图结构的理论分析
   - 参数影响建模
   - 构建与搜索的复杂度

3. **查询性能建模**：
   - 延迟分解模型
   - 吞吐量分析
   - 并行扩展性

4. **可扩展性**：
   - 数据规模扩展预测
   - 维度灾难分析
   - Johnson-Lindenstrauss应用

5. **内存层次优化**：
   - 缓存效率分析
   - NUMA感知优化
   - 带宽利用率

6. **综合模型**：
   - 端到端性能预测
   - 多维度权衡分析

**关键要点**：
- 理论复杂度提供上限，实际性能受缓存、NUMA等影响
- HNSW提供O(log n)查询，适合高吞吐场景
- IVF提供更好的内存局部性，适合大规模数据
- 精确的延迟建模需要考虑CPU、内存、锁等多个维度
- 维度灾难是高维检索的根本挑战
- NUMA优化对多路CPU系统至关重要

## 后续学习建议

- 实际profiling验证理论模型
- 针对特定工作负载调优参数
- 考虑分布式系统架构
- 研究最新的ANN算法改进
