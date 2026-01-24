# Faiss深度课程 - 第14天：性能优化与最佳实践

## 课程目标

总结Faiss的性能优化技巧、最佳实践和生产环境部署建议。

---

## 1. 索引选择指南

### 1.1 决策树

```
开始
  │
  ├─ 数据规模？
  │   ├─ < 10K       → Flat
  │   ├─ 10K-1M      → IVF + Flat/PQ
  │   ├─ 1M-10M      → IVF + PQ (nlist=√N)
  │   └─ > 10M       → HNSW 或 IVF + PQ + nprobe优化
  │
  ├─ 内存限制？
  │   ├─ 严格限制    → PQ (M=16-32) 或 ScalarQuantizer
  │   └─ 宽松        → Flat 或 IVFFlat
  │
  ├─ 延迟要求？
  │   ├─ 极低 (< 1ms) → GPU 或 HNSW
  │   ├─ 低 (< 10ms)   → IVF (nprobe=10-50)
  │   └─ 中等          → IVF + PQ
  │
  └─ 精度要求？
      ├─ 100%        → Flat 或 IVFFlat
      ├─ 95%+        → IVF + PQ (nprobe足够大)
      ├─ 90%+        → IVF + PQ 或 HNSW
      └─ 85%+        → IVF + PQ (小nprobe) + OPQ
```

### 1.2 推荐配置

```cpp
// 场景1：小规模，高精度
IndexFlatL2 index(d);

// 场景2：中等规模，平衡
Index* quantizer = new IndexFlatL2(d);
IndexIVFFlat index(quantizer, d, nlist = sqrt(n));
index.nprobe = 20;

// 场景3：大规模，内存受限
Index* quantizer = new IndexFlatL2(d);
IndexIVFPQ index(quantizer, d, nlist = sqrt(n), M = 32, nbits = 8);
index.nprobe = 10;

// 场景4：超大规模，低延迟
IndexHNSWFlat index(d, M = 32);
index.hnsw.efSearch = 32;
index.hnsw.efConstruction = 64;

// 场景5：GPU加速
auto res = GpuResources::getResources();
GpuIndexIVFFlat gpu_index(res, 0, d, nlist);
```

---

## 2. 参数调优

### 2.1 IVF参数

```cpp
// nlist选择
size_t optimize_nlist(size_t n) {
    // 经验公式
    return std::max(size_t(1), (size_t)sqrt(n));
}

// nprobe选择（精度vs速度权衡）
void optimize_nprobe(IndexIVF& index, const float* xq) {
    std::vector<int> nprobes = {1, 5, 10, 20, 50, 100};
    float* ground_truth = new float[nq * k];

    // 计算ground truth（使用flat）
    IndexFlat ground_truth_index(d);
    ground_truth_index.add(nb, xb);
    ground_truth_index.search(nq, xq, k, ground_truth, labels_gt);

    for (int nprobe : nprobes) {
        index.nprobe = nprobe;
        index.search(nq, xq, k, distances, labels);

        float recall = compute_recall(nq, k, labels, labels_gt);
        printf("nprobe=%d, recall=%.3f\n", nprobe, recall);

        if (recall > 0.95) break;  // 达到目标精度
    }

    delete[] ground_truth;
}
```

### 2.2 PQ参数

```cpp
// M和nbits选择
struct PQConfig {
    int M;       // 子量化器数
    int nbits;   // 每个的位数
    size_t code_size;  // 编码大小（字节）

    float compression_ratio;  // 压缩比
    float estimated_quality;   // 估计质量
};

std::vector<PQConfig> get_pq_configs(int d, size_t original_memory) {
    return {
        // M, nbits, code_size, compression_ratio, quality
        {8,  8,  8,   64.0f,  0.90f},
        {16, 8,  16,  32.0f,  0.93f},
        {32, 8,  32,  16.0f,  0.95f},
        {16, 6,  12,  21.0f,  0.91f},
        {64, 8,  64,  8.0f,   0.97f},
    };
};
```

### 2.3 HNSW参数

```cpp
void hnsw_param_guide() {
    printf("=== HNSW参数指南 ===\n");
    printf("M (连接数):\n");
    printf("  - d=128:  M=16-32\n");
    printf("  - d=256:  M=32-64\n");
    printf("  - d>256:  M=64-128\n");
    printf("  ↗ 精度↑ 速度↓ 内存↑\n\n");

    printf("efConstruction (构建质量):\n");
    printf("  - 默认: 40\n");
    printf("  - 高精度: 64-128\n");
    printf("  - 快速构建: 20-32\n");
    printf("  ↗ 构建时间↑ 精度↑\n\n");

    printf("efSearch (搜索质量):\n");
    printf("  - 默认: 16\n");
    printf("  - 高精度: 32-64\n");
    printf("  - 快速: 8-10\n");
    printf("  ↗ 搜索时间↑ 精度↑\n");
}
```

---

## 3. 性能分析

### 3.1 测量工具

```cpp
#include <faiss/utils/utils.h>
#include <faiss/MetricType.h>

// 性能测量
void benchmark_index(Index& index, const char* name) {
    auto t0 = gettime();

    // 添加
    index.add(nb, xb);

    auto t1 = gettime();
    printf("%s add: %.3f s (%.2f MB/s)\n", name,
           t1 - t0, nb * d * sizeof(float) / (t1 - t0) / 1e6);

    // 搜索
    int nq = 100;
    t0 = gettime();
    index.search(nq, xq, k, distances, labels);
    t1 = gettime();

    double qps = nq / (t1 - t0);
    printf("%s search: %.3f s (%.0f QPS)\n", name, t1 - t0, qps);
}

// 详细统计
void print_ivf_stats(IndexIVF& index) {
    IndexIVFStats stats = index.ivf_stats;

    printf("=== IVF Statistics ===\n");
    printf("Queries: %zu\n", stats.nq);
    printf("Lists scanned: %zu\n", stats.nlist);
    printf("Distances computed: %zu\n", stats.ndis);
    printf("Heap updates: %zu\n", stats.nheap_updates);
    printf("Quantization time: %.2f ms\n", stats.quantization_time);
    printf("Search time: %.2f ms\n", stats.search_time);
}

// HNSW统计
void print_hnsw_stats(HNSW& hnsw) {
    HNSWStats stats = hnsw_stats;

    printf("=== HNSW Statistics ===\n");
    printf("Queries: %zu\n", stats.n1);
    printf("Exhausted: %zu\n", stats.n2);
    printf("Distances: %zu\n", stats.ndis);
    printf("Hops: %zu\n", stats.nhops);
}
```

### 3.2 内存分析

```cpp
// 计算索引内存使用
size_t estimate_memory(Index* index) {
    if (auto* ivf = dynamic_cast<IndexIVFFlat*>(index)) {
        // 粗量化器
        size_t quantizer_mem = ivf->quantizer->ntotal * d * sizeof(float);

        // 向量数据
        size_t vectors_mem = ivf->ntotal * d * sizeof(float);

        // 倒排列表元数据
        size_t overhead = ivf->invlists->get_overhead();

        return quantizer_mem + vectors_mem + overhead;
    }

    if (auto* pq = dynamic_cast<IndexPQ*>(index)) {
        // 质心表
        size_t centroids_mem = pq->pq.M * (1 << pq->pq.nbits) * d / pq->pq.M *
                              sizeof(float);

        // 编码
        size_t codes_mem = pq->ntotal * pq->code_size;

        return centroids_mem + codes_mem;
    }

    if (auto* hnsw = dynamic_cast<IndexHNSWFlat*>(index)) {
        // 向量数据
        size_t vectors_mem = hnsw->ntotal * d * sizeof(float);

        // 图结构
        size_t edges = 0;
        for (int i = 0; i < hnsw->ntotal; i++) {
            edges += hnsw->hnsw.levels[i];
        }

        size_t graph_mem = edges * sizeof(int);

        return vectors_mem + graph_mem;
    }

    return 0;
}
```

---

## 4. 生产环境最佳实践

### 4.1 索引序列化

```cpp
void save_load_index() {
    // 1. 创建并训练索引
    IndexFlatL2 index(d);
    index.add(nb, xb);

    // 2. 保存到文件
    {
        FILE* f = fopen("index.faiss", "wb");
        IOWriter* writer = new FileIOWriter(f);
        write_index(&index, writer);
        delete writer;
    }

    // 3. 从文件加载
    {
        FILE* f = fopen("index.faiss", "rb");
        IOReader* reader = new FileIOReader(f);
        Index* loaded_index = read_index(reader);
        delete reader;

        // 使用
        loaded_index->search(nq, xq, k, distances, labels);
    }
}
```

### 4.2 多线程配置

```cpp
// 设置OpenMP线程数
void set_threads(int nthreads) {
    omp_set_num_threads(nthreads);

    // 环境变量
    // export OMP_NUM_THREADS=8
}

// 线程数建议
int optimal_threads() {
    int n_cores = std::thread::hardware_concurrency();

    // 对于搜索：与核心数相同
    // 对于构建：核心数的1-2倍
    return n_cores;
}
```

### 4.3 批处理

```cpp
// 批量搜索以提高吞吐量
void batch_search(Index& index, size_t batch_size = 10000) {
    size_t n_processed = 0;

    while (n_processed < nq) {
        size_t n_current = std::min(batch_size, nq - n_processed);

        index.search(n_current,
                   xq + n_processed * d,
                   k,
                   distances + n_processed * k,
                   labels + n_processed * k);

        n_processed += n_current;
    }
}
```

---

## 5. 常见问题排查

### 5.1 精度问题

```cpp
// 问题：搜索结果不准确
void diagnose_accuracy() {
    // 1. 检查nprobe
    if (auto* ivf = dynamic_cast<IndexIVF*>(index)) {
        if (ivf->nprobe < ivf->nlist) {
            printf("Warning: nprobe=%d < nlist=%d\n",
                   ivf->nprobe, ivf->nlist);
        }
    }

    // 2. 检查是否训练
    if (!index->is_trained) {
        printf("Warning: Index not trained\n");
    }

    // 3. 比较与ground truth
    IndexFlat ground_truth(d);
    ground_truth.add(nb, xb);
    ground_truth.search(nq, xq, k, gt_distances, gt_labels);

    float recall = compute_recall(nq, k, labels, gt_labels);
    printf("Current recall: %.3f\n", recall);
}
```

### 5.2 内存问题

```cpp
// 问题：内存不足
void solve_memory_issue() {
    // 1. 检查内存使用
    size_t current_mem = estimate_memory(index);
    size_t available_mem = get_available_memory();

    printf("Current: %zu MB, Available: %zu MB\n",
           current_mem / (1024 * 1024),
           available_mem / (1024 * 1024));

    // 2. 优化策略
    if (current_mem > available_mem) {
        printf("Optimization strategies:\n");
        printf("1. Use PQ instead of Flat\n");
        printf("2. Reduce nlist\n");
        printf("3. Use ScalarQuantizer\n");
        printf("4. Use on-disk inverted lists\n");
    }

    // 3. 实施优化
    Index* compressed_index = compress_index(index);
}
```

### 5.3 速度问题

```cpp
// 问题：搜索太慢
void diagnose_speed() {
    // 1. 检查索引类型
    printf("Index type: %s\n", index->get_type());

    // 2. 检查SIMD
    #ifdef __AVX2__
        printf("AVX2: enabled\n");
    #else
        printf("AVX2: disabled (rebuild with -DFAISS_OPT_LEVEL=avx2)\n");
    #endif

    // 3. 检查GPU
    #ifdef FAISS_ENABLE_GPU
        printf("GPU: enabled\n");
    #else
        printf("GPU: disabled (rebuild with -DFAISS_ENABLE_GPU=ON)\n");
    #endif

    // 4. 优化建议
    if (dynamic_cast<IndexFlat*>(index)) {
        printf("Consider using IVF or HNSW for large datasets\n");
    } else if (auto* ivf = dynamic_cast<IndexIVF*>(index)) {
        if (ivf->nprobe == 1) {
            printf("Try increasing nprobe for better recall\n");
        }
    }
}
```

---

## 6. 高级技巧

### 6.1 混合索引

```cpp
// 根据数据分布选择索引
Index* adaptive_index(const float* xb, size_t n, int d) {
    // 分析数据分布
    float density = estimate_data_density(xb, n, d);

    if (density > 0.8) {
        // 密集数据：HNSW效果好
        return new IndexHNSWFlat(d, M=32);
    } else if (n < 100000) {
        // 小规模：Flat足够
        return new IndexFlatL2(d);
    } else {
        // 大规模稀疏：IVF+PQ
        Index* q = new IndexFlatL2(d);
        return new IndexIVFPQ(q, d, sqrt(n), 32, 8);
    }
}

float estimate_data_density(const float* xb, size_t n, int d) {
    // 计算最近邻平均距离
    // 距离越小，数据越密集
    return 0.0f;  // 简化
}
```

### 6.2 自适应nprobe

```cpp
// 根据查询动态调整nprobe
void adaptive_nprobe_search(IndexIVF& index, const float* xq) {
    for (idx_t q = 0; q < nq; q++) {
        // 尝试小nprobe
        index.nprobe = 10;
        float distances_small[k];
        idx_t labels_small[k];

        index.search(1, xq + q * d, k, distances_small, labels_small);

        // 检查结果质量
        float quality = assess_result_quality(distances_small, k);

        if (quality < 0.9) {
            // 质量不够，增加nprobe
            index.nprobe = 50;
            index.search(1, xq + q * d, k, distances + q * k, labels + q * k);
        } else {
            memcpy(distances + q * k, distances_small, k * sizeof(float));
            memcpy(labels + q * k, labels_small, k * sizeof(idx_t));
        }
    }
}
```

### 6.3 分布式部署

```cpp
// 简单的分布式搜索架构
class DistributedFaiss {
    std::vector<std::string> workers;  // 工作节点地址

    void search_distributed(
            const float* xq, size_t nq, size_t k,
            float* distances, idx_t* labels) {

        // 广播查询到所有节点
        for (const auto& worker : workers) {
            send_query(worker, xq, nq, d);
        }

        // 收集并合并结果
        for (const auto& worker : workers) {
            float* dist_worker = new float[nq * k];
            idx_t* labels_worker = new idx_t[nq * k];

            receive_results(worker, dist_worker, labels_worker);

            // 合并到全局结果
            merge_results(dist_worker, labels_worker,
                        distances, labels, nq, k);

            delete[] dist_worker;
            delete[] labels_worker;
        }
    }
};
```

---

## 7. 完整示例

### 7.1 端到端流程

```cpp
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>

int main() {
    // 1. 加载数据
    float* xb = load_vectors("database.fvecs", &n, &d);
    float* xq = load_vectors("queries.fvecs", &nq, &d);

    // 2. 选择索引
    Index* index = nullptr;

    if (n < 10000) {
        // 小规模：Flat
        index = new IndexFlatL2(d);
    } else if (n < 1000000) {
        // 中规模：IVF+PQ
        Index* quantizer = new IndexFlatL2(d);
        index = new IndexIVFPQ(quantizer, d, sqrt(n), 32, 8);
        dynamic_cast<IndexIVFPQ*>(index)->nprobe = 20;
    } else {
        // 大规模：HNSW
        index = new IndexHNSWFlat(d, 32);
        dynamic_cast<IndexHNSWFlat*>(index)->hnsw.efSearch = 32;
    }

    // 3. 训练
    if (!index->is_trained) {
        index->train(n, xb);
    }

    // 4. 添加
    index->add(n, xb);

    // 5. 搜索
    float* distances = new float[nq * k];
    idx_t* labels = new idx_t[nq * k];

    index->search(nq, xq, k, distances, labels);

    // 6. 评估
    float recall = compute_recall(...);

    printf("Recall@%d: %.3f\n", k, recall);

    // 7. 清理
    delete index;
    delete[] xb;
    delete[] xq;
    delete[] distances;
    delete[] labels;

    return 0;
}
```

---

## 8. 第14天总结

### 课程回顾

**14天课程体系**：
1. Faiss基础架构
2. Flat索引详解
3. SIMD距离计算
4. Product Quantization
5. IVF索引架构
6. 加性量化器
7. HNSW图索引
8. NSG与NNDescent
9. FastScan架构
10. 标量量化与RaBitQ
11. 二进制索引
12. GPU实现
13. 复合索引与高级特性
14. 性能优化与最佳实践

### 关键要点

1. **选择合适的索引**：根据数据规模、内存、延迟要求
2. **调优参数**：nlist、nprobe、M、efSearch等
3. **利用SIMD**：确保使用正确的优化级别
4. **考虑GPU**：大规模数据和低延迟场景
5. **组合索引**：Refine、Shards、PreTransform等
6. **监控性能**：定期检查recall和QPS

### 推荐资源

- [Faiss GitHub](https://github.com/facebookresearch/faiss)
- [Faiss文档](https://faiss.ai/)
- [Faiss论文集](https://github.com/facebookresearch/faiss/wiki/Papers)
- [Faiss教程](https://github.com/facebookresearch/faiss/wiki)

---

## 9. 性能优化底层实现详解

### 9.1 内存估算工具

#### IndexMemoryEstimator 完整结构

```cpp
namespace faiss {

// 内存估算器 - 用于预测索引内存使用
struct IndexMemoryEstimator {
    // 估算不同索引类型的内存
    struct MemoryInfo {
        size_t vector_data;      // 向量数据存储
        size_t quantizer;        // 量化器（质心表）
        size_t codes;            // 压缩编码
        size_t graph_structure;  // 图结构（边、链接）
        size_t metadata;         // 元数据（ID、偏移量）
        size_t overhead;         // 对齐、填充开销

        size_t total() const {
            return vector_data + quantizer + codes +
                   graph_structure + metadata + overhead;
        }

        void print() const {
            printf("  Vector data: %.2f MB\n", vector_data / 1e6);
            printf("  Quantizer:   %.2f MB\n", quantizer / 1e6);
            printf("  Codes:       %.2f MB\n", codes / 1e6);
            printf("  Graph:       %.2f MB\n", graph_structure / 1e6);
            printf("  Metadata:    %.2f MB\n", metadata / 1e6);
            printf("  Overhead:    %.2f MB\n", overhead / 1e6);
            printf("  Total:       %.2f MB\n", total() / 1e6);
        }
    };

    // 估算Flat索引
    static MemoryInfo estimate_IndexFlat(
            idx_t n, int d) {
        MemoryInfo info;
        info.vector_data = n * d * sizeof(float);
        info.metadata = 0;  // 无额外元数据
        info.overhead = 0;
        return info;
    }

    // 估算IVFFlat索引
    static MemoryInfo estimate_IndexIVFFlat(
            idx_t n, int d, int nlist) {
        MemoryInfo info;

        // 粗量化器（质心表）
        info.quantizer = nlist * d * sizeof(float);

        // 向量数据（存储在倒排列表中）
        info.vector_data = n * d * sizeof(float);

        // 倒排列表元数据（ID列表、偏移量）
        idx_t avg_list_size = n / nlist;
        info.metadata = nlist * sizeof(std::vector<idx_t>) +  // list id
                        nlist * sizeof(std::vector<float>) +  // list codes
                        n * sizeof(idx_t) +                    // ids
                        64 * nlist;                            // overhead

        return info;
    }

    // 估算IVFPQ索引
    static MemoryInfo estimate_IndexIVFPQ(
            idx_t n, int d, int nlist, int M, int nbits) {
        MemoryInfo info;

        // 粗量化器
        info.quantizer = nlist * d * sizeof(float);

        // PQ质心表
        idx_t ksub = 1 << nbits;  // 2^nbits
        idx_t d_sub = d / M;
        info.codes = M * ksub * d_sub * sizeof(float);

        // 编码存储
        size_t code_size = (M * nbits + 7) / 8;  // 字节
        info.vector_data = n * code_size;

        // 倒排列表元数据
        idx_t avg_list_size = n / nlist;
        info.metadata = nlist * sizeof(std::vector<uint8_t>) +
                        nlist * sizeof(std::vector<idx_t>) +
                        n * sizeof(idx_t) +
                        64 * nlist;

        return info;
    }

    // 估算HNSW索引
    static MemoryInfo estimate_IndexHNSW(
            idx_t n, int d, int M, int M_max) {
        MemoryInfo info;

        // 向量数据
        info.vector_data = n * d * sizeof(float);

        // 图结构（多层）
        // 第0层：每个节点约M个邻居
        // 第1层：约1/M的节点，每个约M个邻居
        // ...
        size_t total_edges = 0;
        size_t n_current = n;
        int level = 0;

        while (n_current > 1) {
            size_t edges_at_level = n_current * M;
            total_edges += edges_at_level;
            n_current /= M;
            level++;
        }

        info.graph_structure = total_edges * (sizeof(idx_t) + sizeof(float));
        info.metadata = n * sizeof(int);  // level info

        return info;
    }
};

} // namespace faiss
```

### 9.2 量化误差分析工具

#### QuantizationErrorAnalyzer 实现

```cpp
namespace faiss {

// 量化误差分析器
struct QuantizationErrorAnalyzer {
    // 分析PQ量化误差
    static void analyze_PQ_error(
            const ProductQuantizer& pq,
            const float* codes,
            idx_t n,
            int d) {

        printf("=== PQ Quantization Error Analysis ===\n");

        // 1. 重构误差
        float* reconstructed = new float[n * d];
        pq.decode(codes, reconstructed, n);

        float* original = ...;  // 原始向量

        float total_error = 0;
        for (idx_t i = 0; i < n * d; i++) {
            float diff = original[i] - reconstructed[i];
            total_error += diff * diff;
        }

        float avg_error = total_error / (n * d);
        printf("Average reconstruction error: %.6f\n", avg_error);

        // 2. 每个子量化器的误差
        printf("\nPer-subquantizer error:\n");
        int M = pq.M;
        int d_sub = d / M;

        for (int m = 0; m < M; m++) {
            // 计算该子空间的误差
            float sub_error = 0;
            for (idx_t i = 0; i < n; i++) {
                int code = get_code(codes, i, m, M);
                const float* centroid = pq.centroids[m] + code * d_sub;
                const float* original_vec = original + i * d + m * d_sub;

                for (int j = 0; j < d_sub; j++) {
                    float diff = original_vec[j] - centroid[j];
                    sub_error += diff * diff;
                }
            }
            printf("  SubQ %d: %.6f\n", m, sub_error / n / d_sub);
        }

        // 3. 距离误差（对搜索的影响）
        printf("\nDistance estimation error:\n");
        analyze_distance_error(pq, original, reconstructed, n, d);

        delete[] reconstructed;
    }

    // 分析距离估计误差
    static void analyze_distance_error(
            const ProductQuantizer& pq,
            const float* original,
            const float* reconstructed,
            idx_t n, int d) {

        // 采样一些查询向量
        int nq = 100;
        float* queries = sample_queries(original, n, d, nq);

        // 计算真实距离和估计距离
        float* true_distances = new float[nq * n];
        float* approx_distances = new float[nq * n];

        pairwise_L2sref(queries, original, nq, n, d, true_distances);
        pq.compute_cross_distances(queries, nq, reconstructed, n, approx_distances);

        // 分析误差分布
        float max_error = 0;
        float avg_error = 0;

        for (int i = 0; i < nq * n; i++) {
            float error = fabs(true_distances[i] - approx_distances[i]);
            avg_error += error;
            max_error = std::max(max_error, error);
        }

        avg_error /= (nq * n);

        printf("  Average distance error: %.4f\n", avg_error);
        printf("  Max distance error: %.4f\n", max_error);

        delete[] queries;
        delete[] true_distances;
        delete[] approx_distances;
    }
};

} // namespace faiss
```

### 9.3 Index Factory 字符串解析

#### 字符串解析器实现

```cpp
namespace faiss {

// Index factory字符串解析
// 格式: "IVF1024,PQ32x8" 或 "HNSW32,Flat"
struct IndexFactoryParser {
    struct IndexSpec {
        std::string type;          // 索引类型: IVF, PQ, HNSW, Flat等
        std::map<std::string, std::string> params;  // 参数键值对
    };

    // 解析字符串为索引规范列表
    static std::vector<IndexSpec> parse(const std::string& description) {
        std::vector<IndexSpec> result;

        // 按逗号分割
        size_t start = 0;
        while (start < description.size()) {
            size_t end = description.find(',', start);
            if (end == std::string::npos) end = description.size();

            std::string part = description.substr(start, end - start);
            result.push_back(parse_part(part));

            start = end + 1;
        }

        return result;
    }

    // 解析单个部分
    static IndexSpec parse_part(const std::string& part) {
        IndexSpec spec;

        // 提取类型（字母部分）
        size_t type_end = 0;
        while (type_end < part.size() && isalpha(part[type_end])) {
            type_end++;
        }

        spec.type = part.substr(0, type_end);

        // 提取参数
        if (type_end < part.size()) {
            std::string params_str = part.substr(type_end);

            // 解析参数: "1024" 或 "32x8" 或 "efSearch=32"
            parse_params(params_str, spec.params);
        }

        return spec;
    }

    // 解析参数字符串
    static void parse_params(
            const std::string& params_str,
            std::map<std::string, std::string>& params) {

        // 检查是否是key=value格式
        size_t eq_pos = params_str.find('=');

        if (eq_pos != std::string::npos) {
            // key=value格式
            std::string key = params_str.substr(0, eq_pos);
            std::string value = params_str.substr(eq_pos + 1);
            params[key] = value;
        } else {
            // 简写格式: "1024" -> nlist=1024
            //           "32x8" -> M=32,nbits=8

            size_t x_pos = params_str.find('x');
            if (x_pos != std::string::npos) {
                // "Mxnbits"格式
                params["M"] = params_str.substr(0, x_pos);
                params["nbits"] = params_str.substr(x_pos + 1);
            } else {
                // 单个数字
                // 根据索引类型推断参数名
                params["nlist"] = params_str;  // 默认是nlist
            }
        }
    }

    // 根据规范创建索引
    static Index* create_index(
            const std::vector<IndexSpec>& specs,
            int d,
            MetricType metric) {

        Index* index = nullptr;

        for (const auto& spec : specs) {
            if (spec.type == "Flat") {
                index = new IndexFlat(d, metric);
            }
            else if (spec.type == "IVF") {
                int nlist = std::stoi(spec.params.at("nlist"));
                Index* quantizer = new IndexFlat(d, metric);
                index = new IndexIVFFlat(quantizer, d, nlist, metric);
            }
            else if (spec.type == "PQ") {
                int M = std::stoi(spec.params["M"]);
                int nbits = std::stoi(spec.params["nbits"]);
                index = new IndexPQ(d, M, nbits, metric);
            }
            else if (spec.type == "IVFPQ") {
                int nlist = std::stoi(spec.params["nlist"]);
                int M = std::stoi(spec.params["M"]);
                int nbits = std::stoi(spec.params["nbits"]);
                Index* quantizer = new IndexFlat(d, metric);
                index = new IndexIVFPQ(quantizer, d, nlist, M, nbits, metric);
            }
            else if (spec.type == "HNSW") {
                int M = 16;  // 默认值
                if (spec.params.count("M")) {
                    M = std::stoi(spec.params["M"]);
                }
                index = new IndexHNSWFlat(d, M);
            }
            // ... 更多索引类型
        }

        return index;
    }
};

} // namespace faiss
```

### 9.4 并行优化工具

#### ParallelSearchScheduler 实现

```cpp
namespace faiss {

// 并行搜索调度器
struct ParallelSearchScheduler {
    int nthread;              // 线程数
    ThreadPool* pool;         // 线程池

    ParallelSearchScheduler(int nt = 0) {
        nthread = nt > 0 ? nt : omp_get_max_threads();
        pool = new ThreadPool(nthread);
    }

    ~ParallelSearchScheduler() {
        delete pool;
    }

    // 并行搜索（按查询并行）
    void search_parallel(
            Index* index,
            const float* xq,
            idx_t nq,
            idx_t k,
            float* distances,
            idx_t* labels) {

        // 将查询分配给不同线程
        idx_t q_per_thread = (nq + nthread - 1) / nthread;

        std::vector<std::future<void>> futures;

        for (int t = 0; t < nthread; t++) {
            idx_t q_start = t * q_per_thread;
            idx_t q_end = std::min(q_start + q_per_thread, nq);

            if (q_start >= q_end) break;

            futures.push_back(pool->submit([=]() {
                index->search(
                    q_end - q_start,
                    xq + q_start * index->d,
                    k,
                    distances + q_start * k,
                    labels + q_start * k);
            }));
        }

        // 等待所有线程完成
        for (auto& f : futures) {
            f.wait();
        }
    }

    // IVF并行搜索（按inverted list并行）
    void search_ivf_parallel(
            IndexIVF* index,
            const float* xq,
            idx_t nq,
            idx_t k,
            float* distances,
            idx_t* labels) {

        // 1. 粗量化（确定查询的nprobe个list）
        // 这一步是串行的（因为每个查询需要不同的list）

        // 2. 细粒度搜索（可以并行）
        for (idx_t q = 0; q < nq; q++) {
            const float* query = xq + q * index->d;

            // 获取该查询要搜索的list
            std::vector<idx_t> lists_to_search;
            get_search_lists(index, query, lists_to_search);

            // 并行搜索这些list
            size_t n_lists = lists_to_search.size();
            size_t lists_per_thread = (n_lists + nthread - 1) / nthread;

            // 每个线程的局部堆
            std::vector<float*> local_distances(nthread);
            std::vector<idx_t*> local_labels(nthread);

            for (int t = 0; t < nthread; t++) {
                local_distances[t] = new float[k];
                local_labels[t] = new idx_t[k];
                heap_heap_array<idx_t>(local_labels[t], local_distances[t], k);
            }

            std::vector<std::future<void>> futures;

            for (int t = 0; t < nthread; t++) {
                size_t list_start = t * lists_per_thread;
                size_t list_end = std::min(list_start + lists_per_thread, n_lists);

                futures.push_back(pool->submit([=]() {
                    for (size_t i = list_start; i < list_end; i++) {
                        idx_t list_no = lists_to_search[i];
                        search_single_list(
                            index, query, list_no,
                            k, local_distances[t], local_labels[t]);
                    }
                }));
            }

            // 等待并合并结果
            for (auto& f : futures) {
                f.wait();
            }

            // 合并所有线程的堆
            heap_heap_array<idx_t>(labels + q * k, distances + q * k, k);
            for (int t = 0; t < nthread; t++) {
                heap_addn<idx_t>(
                    k, labels + q * k, distances + q * k,
                    local_labels[t], local_distances[t], k);
                delete[] local_distances[t];
                delete[] local_labels[t];
            }
        }
    }

private:
    void get_search_lists(
            IndexIVF* index,
            const float* query,
            std::vector<idx_t>& lists) {

        float distances[index->nlist];
        idx_t labels[index->nlist];

        index->quantizer->search(1, query, index->nprobe, distances, labels);

        for (int i = 0; i < index->nprobe; i++) {
            lists.push_back(labels[i]);
        }
    }

    void search_single_list(
            IndexIVF* index,
            const float* query,
            idx_t list_no,
            idx_t k,
            float* distances,
            idx_t* labels) {

        // 获取list内容
        size_t list_size = index->invlists->list_size(list_no);
        const idx_t* ids = index->invlists->get_ids(list_no);

        // 计算距离并更新堆
        for (size_t i = 0; i < list_size; i++) {
            // ... 距离计算和堆更新
        }
    }
};

} // namespace faiss
```

### 9.5 NUMA优化工具

#### NUMAAwareIndex 实现

```cpp
#ifdef FAISS_ENABLE_NUMA

#include <numa.h>

namespace faiss {

// NUMA感知的索引包装器
struct NUMAAwareIndex : Index {
    Index* base_index;          // 底层索引
    int numa_node;              // NUMA节点ID

    NUMAAwareIndex(Index* base, int node)
        : Index(base->d, base->metric_type),
          base_index(base), numa_node(node) {}

    // 在指定NUMA节点上分配内存
    void* allocate_numa(size_t size) const {
        void* ptr = numa_alloc_onnode(size, numa_node);
        return ptr;
    }

    void free_numa(void* ptr, size_t size) const {
        numa_free(ptr, size);
    }

    // 在指定节点上添加向量
    void add(idx_t n, const float* x) override {
        // 将数据迁移到本地NUMA节点
        float* x_local = (float*)allocate_numa(n * d * sizeof(float));
        memcpy(x_local, x, n * d * sizeof(float));

        base_index->add(n, x_local);

        // 注意：这里不释放，因为索引需要持有数据
    }

    void search(
            idx_t n, const float* x, idx_t k,
            float* distances, idx_t* labels) const override {

        // 将查询迁移到本地节点
        float* x_local = (float*)allocate_numa(n * d * sizeof(float));
        memcpy(x_local, x, n * d * sizeof(float));

        base_index->search(n, x_local, k, distances, labels);

        free_numa(x_local, n * d * sizeof(float));
    }
};

// NUMA感知的分片索引
struct NUMAAwareShards : Index {
    std::vector<Index*> shards;     // 每个NUMA节点一个分片

    NUMAAwareShards(int d, MetricType metric) : Index(d, metric) {
        int num_nodes = numa_num_configured_nodes();
        shards.resize(num_nodes);

        // 为每个NUMA节点创建一个索引分片
        for (int node = 0; node < num_nodes; node++) {
            shards[node] = new NUMAAwareIndex(
                new IndexFlatL2(d), node);
        }
    }

    // 添加数据到所有分片
    void add(idx_t n, const float* x) override {
        idx_t n_per_shard = (n + shards.size() - 1) / shards.size();

        for (size_t i = 0; i < shards.size(); i++) {
            idx_t start = i * n_per_shard;
            idx_t end = std::min(start + n_per_shard, n);

            if (start < end) {
                shards[i]->add(end - start, x + start * d);
            }
        }
    }

    // 并行搜索所有分片
    void search(
            idx_t n, const float* x, idx_t k,
            float* distances, idx_t* labels) const override {

        // 初始化结果
        for (idx_t i = 0; i < n * k; i++) {
            distances[i] = std::numeric_limits<float>::infinity();
            labels[i] = -1;
        }

        // 并行搜索各分片
        #pragma omp parallel for
        for (size_t i = 0; i < shards.size(); i++) {
            float* local_dist = new float[n * k];
            idx_t* local_labels = new idx_t[n * k];

            shards[i]->search(n, x, k, local_dist, local_labels);

            // 合并结果
            #pragma omp critical
            {
                for (idx_t q = 0; q < n; q++) {
                    for (idx_t j = 0; j < k; j++) {
                        idx_t idx = q * k + j;
                        if (local_labels[idx] >= 0 &&
                            local_dist[idx] < distances[idx]) {
                            distances[idx] = local_dist[idx];
                            labels[idx] = local_labels[idx];
                        }
                    }
                }
            }

            delete[] local_dist;
            delete[] local_labels;
        }
    }
};

} // namespace faiss

#endif // FAISS_ENABLE_NUMA
```

### 9.6 预取优化工具

#### PrefetchOptimizer 实现

```cpp
namespace faiss {

// 预取优化器
struct PrefetchOptimizer {

    // 为IVF搜索启用预取
    static void ivf_search_with_prefetch(
            IndexIVFFlat& index,
            const float* xq,
            idx_t nq,
            idx_t k,
            float* distances,
            idx_t* labels) {

        for (idx_t q = 0; q < nq; q++) {
            const float* query = xq + q * index.d;

            // 找到要搜索的list
            float list_distances[index.nlist];
            idx_t list_labels[index.nlist];

            index.quantizer->search(1, query, index.nprobe,
                                   list_distances, list_labels);

            // 搜索每个list（带预取）
            heap_heap_array<idx_t>(labels + q * k, distances + q * k, k);

            for (int p = 0; p < index.nprobe; p++) {
                idx_t list_no = list_labels[p];

                // 预取下一个list的数据
                if (p + 1 < index.nprobe) {
                    idx_t next_list = list_labels[p + 1];
                    prefetch_list_data(index, next_list);
                }

                // 搜索当前list
                search_single_list_prefetch(
                    index, query, list_no, k,
                    distances + q * k, labels + q * k);
            }
        }
    }

    // 预取list数据
    static void prefetch_list_data(
            IndexIVFFlat& index,
            idx_t list_no) {

        // 预取inverted list的内容
        size_t list_size = index.invlists->list_size(list_no);
        const float* codes = index.invlists->get_codes(list_no);

        // 预取多行（硬件预取通常一次预取多行）
        const size_t prefetch_distance = 8;  // 预取距离

        for (size_t i = 0; i < list_size; i += prefetch_distance) {
            if (i + prefetch_distance < list_size) {
                _mm_prefetch((char*)(codes + (i + prefetch_distance) * index.d),
                            _MM_HINT_T0);
            }
        }
    }

    // 搜索单个list（带预取）
    static void search_single_list_prefetch(
            IndexIVFFlat& index,
            const float* query,
            idx_t list_no,
            idx_t k,
            float* distances,
            idx_t* labels) {

        size_t list_size = index.invlists->list_size(list_no);
        const float* codes = index.invlists->get_codes(list_no);
        const idx_t* ids = index.invlists->get_ids(list_no);

        const size_t prefetch_distance = 4;

        for (size_t i = 0; i < list_size; i++) {
            // 预取下一个向量
            if (i + prefetch_distance < list_size) {
                _mm_prefetch((char*)(codes + (i + prefetch_distance) * index.d),
                            _MM_HINT_T0);
            }

            // 计算距离
            const float* vec = codes + i * index.d;
            float dist = fvec_L2sqr(query, vec, index.d);

            // 更新堆
            if (dist < distances[0]) {
                heap_replace_top<idx_t>(k, labels, distances);
                heap_push<idx_t>(k, labels, distances, ids[i], dist);
            }
        }
    }

    // 图搜索预取优化（HNSW/NSG）
    static void graph_search_with_prefetch(
            const float* query,
            const float* all_vectors,
            const idx_t* graph_neighbors,  // 邻接表
            int degree,
            idx_t start_node,
            int ef,
            idx_t* result_ids,
            float* result_distances) {

        // 访问集合（visited set）
        std::unordered_set<idx_t> visited;

        // 候选集（最小堆）
        using Candidate = std::pair<float, idx_t>;
        std::priority_queue<Candidate, std::vector<Candidate>,
                           std::greater<Candidate>> candidates;

        // 结果集
        heap_heap_array<idx_t>(result_ids, result_distances, ef);

        // 从起点开始
        candidates.push({0, start_node});
        visited.insert(start_node);

        while (!candidates.empty()) {
            auto [dist, node] = candidates.top();
            candidates.pop();

            // 如果不能改进结果，停止
            if (dist > result_distances[0]) {
                break;
            }

            // 获取邻居
            const idx_t* neighbors = graph_neighbors + node * degree;

            for (int i = 0; i < degree; i++) {
                idx_t neighbor = neighbors[i];
                if (neighbor == -1) break;

                // 预取邻居向量
                if (i + 4 < degree) {
                    idx_t prefetch_neighbor = neighbors[i + 4];
                    if (prefetch_neighbor != -1) {
                        _mm_prefetch(
                            (char*)(all_vectors + prefetch_neighbor * degree),
                            _MM_HINT_T0);
                    }
                }

                if (visited.count(neighbor)) continue;
                visited.insert(neighbor);

                // 计算距离
                const float* neighbor_vec = all_vectors + neighbor * degree;
                float neighbor_dist = fvec_L2sqr(query, neighbor_vec, degree);

                // 添加到候选集
                candidates.push({neighbor_dist, neighbor});

                // 更新结果集
                if (neighbor_dist < result_distances[0]) {
                    heap_replace_top<idx_t>(ef, result_ids, result_distances);
                    heap_push<idx_t>(ef, result_ids, result_distances,
                                     neighbor, neighbor_dist);
                }
            }
        }
    }
};

} // namespace faiss
```

### 9.7 缓存友好数据布局

#### CacheFriendlyLayout 实现

```cpp
namespace faiss {

// 缓存友好的数据布局
struct CacheFriendlyLayout {

    // 转换为SoA（Structure of Arrays）布局
    // 更好的缓存利用和SIMD效率
    static void convert_to_SoA(
            const float* AOS_data,  // Array of Structures
            float* SoA_data,
            idx_t n, int d) {

        // AOS: [v0_d0, v0_d1, ..., v0_d{d-1},
        //       v1_d0, v1_d1, ..., v1_d{d-1},
        //       ...]
        //
        // SoA: [v0_d0, v1_d0, ..., v{n-1}_d0,
        //       v0_d1, v1_d1, ..., v{n-1}_d1,
        //       ...]

        for (int dim = 0; dim < d; dim++) {
            for (idx_t i = 0; i < n; i++) {
                SoA_data[dim * n + i] = AOS_data[i * d + dim];
            }
        }
    }

    // SoA到AOS转换
    static void convert_to_AOS(
            const float* SoA_data,
            float* AOS_data,
            idx_t n, int d) {

        for (idx_t i = 0; i < n; i++) {
            for (int dim = 0; dim < d; dim++) {
                AOS_data[i * d + dim] = SoA_data[dim * n + i];
            }
        }
    }

    // 使用SoA布局的批量距离计算
    static void compute_distances_SoA(
            const float* SoA_database,  // SoA布局
            const float* query,
            idx_t n, int d,
            float* distances) {

        // 对每个维度，处理所有向量
        // 这样可以充分利用缓存和SIMD

        // 初始化距离
        for (idx_t i = 0; i < n; i++) {
            distances[i] = 0;
        }

        // 按维度循环
        for (int dim = 0; dim < d; dim++) {
            const float* dim_data = SoA_database + dim * n;
            float q_val = query[dim];

            // SIMD向量化计算
            idx_t i = 0;
            #ifdef __AVX2__
            __m256 q_vec = _mm256_set1_ps(q_val);

            for (; i + 8 <= n; i += 8) {
                __m256 db_vec = _mm256_loadu_ps(dim_data + i);
                __m256 diff = _mm256_sub_ps(db_vec, q_vec);
                __m256 sq = _mm256_mul_ps(diff, diff);
                __m256 dist = _mm256_loadu_ps(distances + i);
                _mm256_storeu_ps(distances + i, _mm256_add_ps(dist, sq));
            }
            #endif

            // 处理剩余元素
            for (; i < n; i++) {
                float diff = dim_data[i] - q_val;
                distances[i] += diff * diff;
            }
        }
    }
};

// 缓存友好的IVF索引
struct CacheFriendlyIVF {
    int d;
    int nlist;

    // SoA布局的inverted lists
    struct List {
        float* codes_soa;    // SoA布局: [d0][d1][...][d{d-1}]
        idx_t* ids;
        size_t size;
        size_t capacity;
    };

    std::vector<List> lists;

    // 添加向量到list（自动转换为SoA）
    void add_to_list(
            idx_t list_no,
            idx_t id,
            const float* vector) {

        List& list = lists[list_no];

        // 扩容
        if (list.size >= list.capacity) {
            size_t new_capacity = list.capacity * 2;
            if (new_capacity == 0) new_capacity = 16;

            // 重新分配SoA数据
            float* new_codes = new float[new_capacity * d];
            for (int dim = 0; dim < d; dim++) {
                memcpy(new_codes + dim * new_capacity,
                       list.codes_soa + dim * list.capacity,
                       list.size * sizeof(float));
            }

            delete[] list.codes_soa;
            list.codes_soa = new_codes;

            idx_t* new_ids = new idx_t[new_capacity];
            memcpy(new_ids, list.ids, list.size * sizeof(idx_t));
            delete[] list.ids;
            list.ids = new_ids;

            list.capacity = new_capacity;
        }

        // 添加向量（逐维存储）
        for (int dim = 0; dim < d; dim++) {
            list.codes_soa[dim * list.capacity + list.size] = vector[dim];
        }
        list.ids[list.size] = id;
        list.size++;
    }

    // 搜索单个list（使用SoA优化）
    void search_list(
            idx_t list_no,
            const float* query,
            idx_t k,
            float* distances,
            idx_t* labels) {

        const List& list = lists[list_no];

        // 使用SoA优化的距离计算
        float* list_distances = new float[list.size];

        CacheFriendlyLayout::compute_distances_SoA(
            list.codes_soa, query, list.size, d, list_distances);

        // 堆排序
        heap_heap_array<idx_t>(labels, distances, k);
        for (size_t i = 0; i < list.size; i++) {
            if (list_distances[i] < distances[0]) {
                heap_replace_top<idx_t>(k, labels, distances);
                heap_push<idx_t>(k, labels, distances,
                                list.ids[i], list_distances[i]);
            }
        }

        delete[] list_distances;
    }
};

} // namespace faiss
```

### 9.8 自动调优工具

#### AutoTuner 实现

```cpp
namespace faiss {

// 自动调优器
struct AutoTuner {
    struct TuningResult {
        std::string config;
        float qps;           // Queries per second
        float recall;
        size_t memory_mb;
        float score;         // 综合评分
    };

    // 自动调优IVF参数
    static TuningResult tune_ivf_params(
            const float* xb, idx_t n, int d,
            const float* xq, idx_t nq,
            const float* gt_distances, const idx_t* gt_labels,
            idx_t k) {

        std::vector<TuningResult> results;

        // 尝试不同的nlist
        std::vector<int> nlist_values = {
            32, 64, 128, 256, 512, 1024, 2048
        };

        // 尝试不同的nprobe
        std::vector<int> nprobe_values = {
            1, 5, 10, 20, 50, 100
        };

        for (int nlist : nlist_values) {
            // 训练索引
            IndexFlatL2 quantizer(d);
            IndexIVFFlat index(&quantizer, d, nlist);
            index.train(n, xb);
            index.add(n, xb);

            for (int nprobe : nprobe_values) {
                index.nprobe = nprobe;

                // 测量性能
                auto t0 = gettime();
                float* distances = new float[nq * k];
                idx_t* labels = new idx_t[nq * k];

                index.search(nq, xq, k, distances, labels);

                auto t1 = gettime();
                double qps = nq / (t1 - t0);

                // 计算recall
                float recall = compute_recall(
                    nq, k, labels, gt_labels);

                // 估算内存
                size_t mem = n * d * sizeof(float) +  // vectors
                             nlist * d * sizeof(float);  // quantizer

                // 综合评分
                float score = qps * recall / (mem / 1e6);

                TuningResult result;
                result.config = "IVF" + std::to_string(nlist) +
                               ",nprobe=" + std::to_string(nprobe);
                result.qps = qps;
                result.recall = recall;
                result.memory_mb = mem / (1024 * 1024);
                result.score = score;

                results.push_back(result);

                delete[] distances;
                delete[] labels;
            }
        }

        // 选择最佳配置
        auto best = std::max_element(
            results.begin(), results.end(),
            [](const TuningResult& a, const TuningResult& b) {
                return a.score < b.score;
            });

        return *best;
    }

    // 自动调优HNSW参数
    static TuningResult tune_hnsw_params(
            const float* xb, idx_t n, int d,
            const float* xq, idx_t nq,
            const float* gt_distances, const idx_t* gt_labels,
            idx_t k) {

        std::vector<TuningResult> results;

        // 尝试不同的M
        std::vector<int> M_values = {16, 32, 64};

        // 尝试不同的efSearch
        std::vector<int> ef_values = {16, 32, 64, 128, 256};

        for (int M : M_values) {
            // 构建索引
            IndexHNSWFlat index(d, M);
            index.hnsw.efConstruction = 64;
            index.add(n, xb);

            for (int efSearch : ef_values) {
                index.hnsw.efSearch = efSearch;

                // 测量性能
                auto t0 = gettime();
                float* distances = new float[nq * k];
                idx_t* labels = new idx_t[nq * k];

                index.search(nq, xq, k, distances, labels);

                auto t1 = gettime();
                double qps = nq / (t1 - t0);

                // 计算recall
                float recall = compute_recall(
                    nq, k, labels, gt_labels);

                // 估算内存（粗略）
                size_t n_edges = n * M * (1 + log(n) / log(M));
                size_t mem = n * d * sizeof(float) +  // vectors
                             n_edges * (sizeof(idx_t) + sizeof(float));  // graph

                float score = qps * recall / (mem / 1e6);

                TuningResult result;
                result.config = "HNSW,M=" + std::to_string(M) +
                               ",efSearch=" + std::to_string(efSearch);
                result.qps = qps;
                result.recall = recall;
                result.memory_mb = mem / (1024 * 1024);
                result.score = score;

                results.push_back(result);

                delete[] distances;
                delete[] labels;
            }
        }

        auto best = std::max_element(
            results.begin(), results.end(),
            [](const TuningResult& a, const TuningResult& b) {
                return a.score < b.score;
            });

        return *best;
    }

    static float compute_recall(
            idx_t nq, idx_t k,
            const idx_t* labels,
            const idx_t* gt_labels) {

        idx_t correct = 0;
        for (idx_t i = 0; i < nq * k; i++) {
            // 检查gt_labels[0..k-1]中是否包含labels[i]
            idx_t q = i / k;
            idx_t start = q * k;
            idx_t end = start + k;

            for (idx_t j = start; j < end; j++) {
                if (gt_labels[j] == labels[i]) {
                    correct++;
                    break;
                }
            }
        }

        return float(correct) / (nq * k);
    }
};

} // namespace faiss
```

---

## 10. 编译器优化深度解析

### 10.1 GCC/Clang优化选项详解

```cpp
// 基础优化级别
// -O0: 无优化（默认）
// -O1: 基础优化（减少代码大小和执行时间）
// -O2: 推荐级别（几乎所有优化）
// -O3: 最高优化（包括循环展开、向量化等）
// -Os: 优化代码大小
// -Og: 优化调试体验
// -Ofast: -O3 + 不严格遵守标准（可能改变浮点语义）

// SIMD相关选项
// -mavx: 启用AVX指令集
// -mavx2: 启用AVX2（整数操作）
// -mavx512f: 启用AVX-512基础指令集
// -mavx512bw: 启用AVX-512字节和字指令集
// -mavx512vl: 启用AVX-512向量长度扩展
// -mfma: 启用FMA（融合乘加）指令
// -mfma4: 启用AMD的FMA4指令

// 特定架构优化
// -march=native: 生成当前CPU的最优代码
// -march=haswell: Intel Haswell (AVX2)
// -march=skylake-avx512: Intel Skylake-X (AVX-512)
// -march=zen2: AMD Zen2
// -mtune=native: 调优为当前CPU

// 示例编译命令
// g++ -O3 -march=native -mavx2 -mfma -fopenmp program.cpp

// Intel编译器
// icc -O3 -xHOST -qopenmp program.cpp
// -xHOST: 为当前CPU生成代码
// -ipo: 过程间优化
// -no-prec-div: 不保证浮点除法精度（更快）
// -fast: 等价于 -O3 -ipo -no-prec-div -fma
```

### 10.2 编译器内联与提示

```cpp
// 强制内联（关键性能函数）
inline void __attribute__((always_inline)) hot_function(float* x, size_t n) {
    // 编译器必须内联此函数
    for (size_t i = 0; i < n; i++) {
        x[i] *= x[i];
    }
}

// 禁止内联（大函数或调试函数）
inline void __attribute__((noinline)) cold_function(float* x, size_t n) {
    // 编译器不应内联此函数
    for (size_t i = 0; i < n; i++) {
        x[i] = sqrt(x[i]);
    }
}

// 热点函数提示（告诉编译器这是热点）
void __attribute__((hot)) search_loop(float* x, size_t n) {
    // 编译器会激进优化此函数
}

// 冷函数提示（很少执行）
void __attribute__((cold)) error_handling(const char* msg) {
    // 编译器会优化代码大小而非速度
    printf("Error: %s\n", msg);
}

// 纯函数（无副作用，可优化）
float __attribute__((const)) pure_computation(float x) {
    return x * x + 1.0f;
}

// constexpr函数（编译时求值）
constexpr int log2_ceiling(int n) {
    int r = 0;
    while ((1 << r) < n) {
        r++;
    }
    return r;
}
```

### 10.3 循环优化提示

```cpp
// 循环展开提示
#pragma GCC unroll 4
void unrolled_loop_example(float* a, float* b, float* c, size_t n) {
    // 建议编译器展开4次
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// 禁止展开
#pragma GCC unroll 0
void no_unroll(float* a, float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// 循环向量化提示
#pragma omp simd
void vectorized_loop(float* __restrict a,
                    float* __restrict b,
                    float* __restrict c,
                    size_t n) {
    // 建议使用SIMD向量化
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

// 对齐数组提示
void __attribute__((assume_aligned(32)))
    aligned_array_operation(float* a, size_t n) {
    // 告诉编译器a是32字节对齐的
    for (size_t i = 0; i < n; i++) {
        a[i] *= 2.0f;
    }
}

// 限制指针范围（有助于编译器优化）
void __attribute__((access(read_only, 1, 4)))
    process_4_elements(const float* x) {
    // x至少有4个元素可读
    __m128 v = _mm_loadu_ps(x);
    v = _mm_mul_ps(v, _mm_set1_ps(2.0f));
    _mm_storeu_ps((float*)x, v);
}
```

### 10.4 链接时优化（LTO）

```bash
# 链接时优化允许编译器在链接阶段进行跨文件优化

# GCC/Clang
# 步骤1: 编译为中间代码
g++ -O3 -flto -c file1.cpp -o file1.o
g++ -O3 -flto -c file2.cpp -o file2.o

# 步骤2: 链接（链接时优化）
g++ -O3 -flto -o program file1.o file2.o

# 或一次性编译链接
g++ -O3 -flto -o program file1.cpp file2.cpp

# Intel编译器
icc -O3 -ipo -o program file1.cpp file2.cpp

# LTO的好处：
# 1. 跨文件内联
# 2. 更好的寄存器分配
# 3. 更激进的死代码消除
# 4. 过程间分析
```

### 10.5 Profile-Guided Optimization (PGO)

```bash
# PGO：基于运行时剖析数据优化代码

# 步骤1: 生成剖析数据
g++ -O3 -fprofile-generate -o program program.cpp
./program  # 运行典型工作负载

# 步骤2: 使用剖析数据重新编译
g++ -O3 -fprofile-use -o program program.cpp

# GCC/Clang PGO选项
# -fprofile-generate: 生成gcov剖析数据
# -fprofile-use: 使用gcov剖析数据优化
# -fprofile-dir=<dir>: 指定剖析数据目录

# Intel编译器PGO
# 步骤1: 生成
icc -O3 -prof-gen -o program program.cpp
./program
# 步骤2: 使用
icc -O3 -prof-use -o program program.cpp

# PGO的优化：
# 1. 更好的分支预测
# 2. 热点函数的内联决策
# 3. 基本块布局优化
# 4. 代码大小/速度权衡
```

---

## 11. 内存对齐深度优化

### 11.1 数据结构对齐

```cpp
#include <stdint.h>
#include <stdlib.h>

// 错误的对齐：缓存行浪费
struct BadAligned {
    bool flag1;      // 1 byte
    // 7 bytes padding
    float value1;     // 4 bytes
    bool flag2;       // 1 byte
    // 3 bytes padding
    double value2;    // 8 bytes
}; // 总计: 24 bytes，但flags和value交错

// 正确的对齐：减少padding
struct GoodAligned {
    double value2;    // 8 bytes (最大)
    float value1;     // 4 bytes
    bool flag1;       // 1 byte
    bool flag2;       // 1 byte
    // 2 bytes padding
}; // 总计: 16 bytes

// 缓存行对齐（64字节）
struct alignas(64) CacheLineAligned {
    float data[16];   // 正好64字节
};

// 使用关键字对齐
struct alignas(32) AVX2Aligned {
    float values[8];  // 32字节，适合AVX2
};

struct alignas(64) AVX512Aligned {
    float values[16]; // 64字节，适合AVX-512
};

// 动态分配对齐内存
void* allocate_aligned(size_t size, size_t alignment) {
    // C++17标准方法
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;

    // 或使用aligned_alloc
    // return aligned_alloc(alignment, size);
}

// 释放对齐内存
void deallocate_aligned(void* ptr) {
    free(ptr);
}
```

### 11.2 False Sharing 问题

```cpp
#include <atomic>
#include <thread>
#include <vector>

// 错误：两个计数器在同一缓存行
struct BadCounters {
    std::atomic<int> counter1;
    std::atomic<int> counter2;  // 可能在同一缓存行
};

// 正确：确保独立缓存行
struct alignas(64) GoodCounters {
    std::atomic<int> counter1;
    char padding1[64 - sizeof(std::atomic<int>)];

    alignas(64) std::atomic<int> counter2;
    char padding2[64 - sizeof(std::atomic<int>)];
};

// 测试false sharing的影响
void test_false_sharing() {
    constexpr int iterations = 10000000;

    // Bad版本
    BadCounters bad;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::thread t1([&]() {
        for (int i = 0; i < iterations; i++) {
            bad.counter1++;
        }
    });

    std::thread t2([&]() {
        for (int i = 0; i < iterations; i++) {
            bad.counter2++;
        }
    });

    t1.join();
    t2.join();

    auto t1_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // Good版本
    GoodCounters good;
    auto t0_good = std::chrono::high_resolution_clock::now();

    std::thread t3([&]() {
        for (int i = 0; i < iterations; i++) {
            good.counter1++;
        }
    });

    std::thread t4([&]() {
        for (int i = 0; i < iterations; i++) {
            good.counter2++;
        }
    });

    t3.join();
    t4.join();

    auto t2_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0_good).count();

    printf("Bad (false sharing): %ld ms\n", t1_time);
    printf("Good (cache line aligned): %ld ms\n", t2_time);
    printf("Speedup: %.2fx\n", (double)t1_time / t2_time);
}
```

### 11.3 NUMA感知分配

```cpp
#ifdef __linux__
#include <numa.h>

class NUMAAllocator {
public:
    void* allocate(size_t size, int node = -1) {
        // node = -1: 本地节点
        // node >= 0: 指定NUMA节点

        void* ptr = nullptr;
        if (node >= 0) {
            ptr = numa_alloc_onnode(size, node);
        } else {
            ptr = numa_alloc(size);  // 本地分配
        }

        return ptr;
    }

    void deallocate(void* ptr, size_t size) {
        numa_free(ptr, size);
    }

    // 绑定当前线程到NUMA节点
    static void bind_thread(int node) {
        numa_set_preferred(node);

        // 绑定到特定CPU
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        int num_cpus = numa_num_task_cpus();
        for (int i = 0; i < num_cpus; i++) {
            if (numa_node_of_cpu(i) == node) {
                CPU_SET(i, &cpuset);
            }
        }

        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }

    static int get_current_node() {
        return numa_node_of_cpu(sched_getcpu());
    }
};

// NUMA感知的向量存储
class NUMAAwareVectorStorage {
    struct NodeStorage {
        void* data;
        size_t size;
        int node;
    };

    std::vector<NodeStorage> nodes;
    NUMAAllocator allocator;

public:
    void init(int num_nodes, size_t size_per_node) {
        nodes.resize(num_nodes);

        for (int i = 0; i < num_nodes; i++) {
            nodes[i].data = allocator.allocate(size_per_node, i);
            nodes[i].size = size_per_node;
            nodes[i].node = i;
        }
    }

    void add_vector(const float* vec, int node) {
        // 添加到指定NUMA节点
        memcpy(nodes[node].data, vec, nodes[node].size);
    }

    // 搜索时访问本地节点数据
    void search_local(const float* query, float* distances) {
        int node = NUMAAllocator::get_current_node();
        const float* local_data = static_cast<const float*>(nodes[node].data);

        // 执行搜索...
    }

    ~NUMAAwareVectorStorage() {
        for (auto& node : nodes) {
            allocator.deallocate(node.data, node.size);
        }
    }
};

#endif // __linux__
```

### 11.4 预取策略

```cpp
#include <xmmintrin.h>

// 预取提示级别
// _MM_HINT_T0: 预取到所有级缓存（L1, L2, L3）
// _MM_HINT_T1: 预取到L2及以下
// _MM_HINT_T2: 预取到L3及以下
// _MM_HINT_NTA: 非时临预取（不替换缓存中数据）

// 软件预取
void prefetch_example(const float* data, size_t n) {
    constexpr size_t prefetch_distance = 16;

    for (size_t i = 0; i < n; i++) {
        // 预取未来的数据
        if (i + prefetch_distance < n) {
            _mm_prefetch((const char*)(data + i + prefetch_distance),
                        _MM_HINT_T0);
        }

        // 处理当前数据
        float val = data[i];
        // ... 处理 ...
    }
}

// 自适应预取距离
template <typename T>
class AdaptivePrefetch {
    size_t prefetch_distance = 8;
    size_t miss_count = 0;
    size_t hit_count = 0;

public:
    void process(const T* data, size_t n) {
        for (size_t i = 0; i < n; i++) {
            if (i + prefetch_distance < n) {
                _mm_prefetch((const char*)(data + i + prefetch_distance),
                           _MM_HINT_T0);
            }

            // 处理data[i]
            // 根据缓存未命中率调整prefetch_distance
        }
    }

    void adjust_distance(bool cache_miss) {
        if (cache_miss) {
            miss_count++;
            if (miss_count > 10) {
                prefetch_distance = std::min(64, prefetch_distance + 8);
                miss_count = 0;
            }
        } else {
            hit_count++;
            if (hit_count > 100) {
                prefetch_distance = std::max(4, prefetch_distance - 4);
                hit_count = 0;
            }
        }
    }
};

// 链表预取（HNSW等图结构）
void graph_search_with_prefetch(
    const float* query,
    const idx_t* neighbors,  // 邻接表
    size_t degree,
    idx_t start_node) {

    std::unordered_set<idx_t> visited;
    std::vector<std::pair<float, idx_t>> candidates;

    candidates.push_back({0.0f, start_node});

    while (!candidates.empty()) {
        // 预取未来要访问的节点
        for (size_t i = 0; i < std::min(size_t(4), candidates.size()); i++) {
            idx_t node = candidates[i].second;
            if (!visited.count(node)) {
                const float* node_data = get_vector(node);
                _mm_prefetch((const char*)node_data, _MM_HINT_T0);
            }
        }

        // 处理当前候选
        auto current = candidates.back();
        candidates.pop_back();

        if (visited.count(current.second)) continue;
        visited.insert(current.second);

        // 计算距离
        float dist = compute_distance(query, current.second);

        // 扩展邻居
        const idx_t* node_neighbors = neighbors + current.second * degree;
        for (size_t i = 0; i < degree; i++) {
            if (node_neighbors[i] >= 0 && !visited.count(node_neighbors[i])) {
                float d = compute_distance(query, node_neighbors[i]);
                candidates.push_back({d, node_neighbors[i]});
            }
        }

        // 保持候选集大小
        if (candidates.size() > 100) {
            std::partial_sort(
                candidates.begin(),
                candidates.begin() + 100,
                candidates.end());
            candidates.resize(100);
        }
    }
}
```

---

## 12. 高级缓存优化技术

### 12.1 缓存行感知算法设计

```cpp
// Cache-oblivious算法避免特定缓存大小的依赖
// Cache-oblivious矩阵乘法（用于批量内积计算）
namespace cache_optimized {

// Cache-oblivious矩阵分块
template<typename T>
void cache_oblivious_gemm(
        const T* A, const T* B, T* C,
        size_t m, size_t n, size_t p,
        size_t threshold = 64) {  // 阈值：小矩阵直接计算

    if (m * n * p <= threshold) {
        // 小矩阵：直接计算
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < p; j++) {
                T sum = 0;
                for (size_t k = 0; k < n; k++) {
                    sum += A[i * n + k] * B[k * p + j];
                }
                C[i * p + j] += sum;
            }
        }
    } else {
        // 大矩阵：递归分块
        size_t m2 = m / 2;
        size_t n2 = n / 2;
        size_t p2 = p / 2;

        // 分块计算8个子矩阵
        cache_oblivious_gemm(A, B, C, m2, n2, p2, threshold);
        cache_oblivious_gemm(A, B + p2, C + p2, m2, n2, p - p2, threshold);
        cache_oblivious_gemm(A + n2 * m2, B, C + m2 * p, m - m2, n2, p2, threshold);
        cache_oblivious_gemm(A + n2 * m2, B + p2, C + m2 * p + p2,
                           m - m2, n2, p - p2, threshold);
        // ... 其余4个子块
    }
}

// 批量内积的缓存优化版本
void batch_inner_product_cache_optimized(
        const float* queries,      // [nq x d]
        const float* database,     // [nb x d]
        float* ips,               // [nq x nb]
        size_t nq, size_t nb, size_t d) {

    // 分块大小：L1缓存约32KB，可容纳256个float
    constexpr size_t BLOCK_M = 64;   // 查询块
    constexpr size_t BLOCK_N = 64;   // 数据库块
    constexpr size_t BLOCK_K = 64;   // 维度块

    for (size_t q0 = 0; q0 < nq; q0 += BLOCK_M) {
        size_t q1 = std::min(q0 + BLOCK_M, nq);

        for (size_t b0 = 0; b0 < nb; b0 += BLOCK_N) {
            size_t b1 = std::min(b0 + BLOCK_N, nb);

            // 初始化结果块
            for (size_t q = q0; q < q1; q++) {
                for (size_t b = b0; b < b1; b++) {
                    ips[q * nb + b] = 0;
                }
            }

            // 分块计算
            for (size_t d0 = 0; d0 < d; d0 += BLOCK_K) {
                size_t d1 = std::min(d0 + BLOCK_K, d);

                // 预取下一块数据
                if (d1 + BLOCK_K <= d) {
                    for (size_t q = q0; q < q1; q += 8) {
                        _mm_prefetch((const char*)(queries + q * d + d1),
                                    _MM_HINT_T0);
                    }
                    for (size_t b = b0; b < b1; b += 8) {
                        _mm_prefetch((const char*)(database + b * d + d1),
                                    _MM_HINT_T0);
                    }
                }

                // 计算当前块
                for (size_t q = q0; q < q1; q++) {
                    for (size_t b = b0; b < b1; b++) {
                        for (size_t d = d0; d < d1; d++) {
                            ips[q * nb + b] += queries[q * d + d] *
                                             database[b * d + d];
                        }
                    }
                }
            }
        }
    }
}
}
```

### 12.2 缓存阻塞（Cache Blocking）优化

```cpp
// 循环阻塞优化：提高空间局部性
namespace cache_blocking {

// L2距离计算的缓存优化版本
void l2_distance_cache_blocked(
        const float* x,        // [d]
        const float* y,        // [nb x d]
        float* distances,      // [nb]
        size_t d, size_t nb) {

    // 缓存块大小：根据L2缓存大小调整
    // 假设L2缓存256KB，每个向量128维x4字节=512字节
    // 256KB / 512B ≈ 512个向量
    constexpr size_t BLOCK_NB = 128;
    constexpr size_t BLOCK_D = 32;

    // 对y向量进行分块处理
    for (size_t b0 = 0; b0 < nb; b0 += BLOCK_NB) {
        size_t b1 = std::min(b0 + BLOCK_NB, nb);

        for (size_t d0 = 0; d0 < d; d0 += BLOCK_D) {
            size_t d1 = std::min(d0 + BLOCK_D, d);

            // 预取下一块y
            if (d1 + BLOCK_D <= d) {
                for (size_t b = b0; b < b1; b += 8) {
                    _mm_prefetch((const char*)(y + b * d + d1),
                                _MM_HINT_T0);
                }
            }

            // 计算当前块的距离
            for (size_t b = b0; b < b1; b++) {
                for (size_t i = d0; i < d1; i++) {
                    float diff = x[i] - y[b * d + i];
                    distances[b] += diff * diff;
                }
            }
        }
    }

    // 处理剩余元素
    for (size_t b = 0; b < nb; b++) {
        for (size_t i = (d / BLOCK_D) * BLOCK_D; i < d; i++) {
            float diff = x[i] - y[b * d + i];
            distances[b] += diff * diff;
        }
    }
}

// IVF搜索的缓存优化：按列表分组处理
class CachedIVFSearcher {
public:
    void search_cached(
            faiss::IndexIVFFlat* ivf_index,
            const float* query,
            idx_t k,
            float* distances,
            idx_t* labels) {

        int nprobe = ivf_index->nprobe;
        int nlist = ivf_index->nlist;

        // 找到最近的nprobe个倒排表
        std::vector<idx_t> nearby_lists(nprobe);
        ivf_index->quantizer->search(1, query, nprobe,
                                     distances, nearby_lists.data());

        // 按列表大小排序：优先处理小列表（缓存友好）
        std::vector<std::pair<size_t, idx_t>> list_sizes;
        for (int i = 0; i < nprobe; i++) {
            size_t list_size = ivf_index->invlists->list_size(nearby_lists[i]);
            list_sizes.push_back({list_size, nearby_lists[i]});
        }
        std::sort(list_sizes.begin(), list_sizes.end());

        // 按大小批次处理：缓存效率更高
        constexpr size_t LIST_BATCH = 16;

        for (size_t batch_start = 0; batch_start < list_sizes.size();
             batch_start += LIST_BATCH) {

            size_t batch_end = std::min(batch_start + LIST_BATCH,
                                       list_sizes.size());

            // 预取下一批列表的码本
            if (batch_end + LIST_BATCH <= list_sizes.size()) {
                for (size_t i = batch_end; i < batch_end + LIST_BATCH; i++) {
                    idx_t list_id = list_sizes[i].second;
                    ivf_index->invlists->prefetch_list(list_id);
                }
            }

            // 处理当前批次
            for (size_t i = batch_start; i < batch_end; i++) {
                idx_t list_id = list_sizes[i].second;
                search_single_list(ivf_index, query, list_id, k,
                                 distances, labels);
            }
        }
    }

private:
    void search_single_list(
            faiss::IndexIVFFlat* ivf_index,
            const float* query,
            idx_t list_id,
            idx_t k,
            float* distances,
            idx_t* labels);
};
}
```

### 12.3 数据布局优化

```cpp
// 数据布局对缓存性能的影响
namespace layout_optimization {

// 结构体数组（Array of Structures） vs 结构体数组（Structure of Arrays）
struct VectorSoA {  // Structure of Arrays
    float* x;      // 所有向量的第0维
    float* y;      // 所有向量的第1维
    float* z;      // 所有向量的第2维
    // ...
    size_t n;      // 向量数

    // SoA布局：SIMD友好，缓存友好（处理维度时）
    // 适合：逐维计算（如PQ编码）
};

struct VectorAoS {  // Array of Structures
    float* data;    // [n x d]，连续存储
    size_t n;
    size_t d;

    // AoS布局：处理向量时友好
    // 适合：随机访问单个向量
};

// 转换AoS到SoA
void aos_to_soa(const float* aos_data, size_t n, size_t d, VectorSoA& soa) {
    soa.n = n;
    soa.x = new float[n];
    soa.y = new float[n];
    soa.z = new float[n];

    for (size_t i = 0; i < n; i++) {
        soa.x[i] = aos_data[i * d + 0];
        soa.y[i] = aos_data[i * d + 1];
        soa.z[i] = aos_data[i * d + 2];
    }
}

// SoA布局的SIMD优势
void compute_norms_soa(const VectorSoA& soa, float* norms) {
    size_t i = 0;

    #ifdef __AVX2__
    // 一次处理8个向量的x分量
    for (; i + 8 <= soa.n; i += 8) {
        __m256 vx = _mm256_loadu_ps(soa.x + i);
        __m256 vy = _mm256_loadu_ps(soa.y + i);
        __m256 vz = _mm256_loadu_ps(soa.z + i);

        __m256 vresult = _mm256_mul_ps(vx, vx);
        vresult = _mm256_fmadd_ps(vy, vy, vresult);
        vresult = _mm256_fmadd_ps(vz, vz, vresult);

        _mm256_storeu_ps(norms + i, vresult);
    }
    #endif

    // 处理剩余向量
    for (; i < soa.n; i++) {
        norms[i] = soa.x[i] * soa.x[i] + soa.y[i] * soa.y[i] +
                   soa.z[i] * soa.z[i];
    }
}
}
```

### 12.4 缓存预取调优工具

```cpp
// 自动寻找最优预取距离
class PrefetchTuner {
public:
    // 测试不同预取距离的性能
    static size_t find_optimal_distance(
            std::function<void(size_t)> compute_func,
            size_t max_distance = 256) {

        std::map<size_t, double> timings;

        for (size_t dist = 0; dist <= max_distance; dist += 8) {
            double time = measure_with_prefetch(compute_func, dist);
            timings[dist] = time;
        }

        // 找最优
        auto best = std::min_element(timings.begin(), timings.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });

        printf("Optimal prefetch distance: %zu (time: %.2f us)\n",
               best->first, best->second);

        return best->first;
    }

private:
    static double measure_with_prefetch(
            std::function<void(size_t)> func,
            size_t prefetch_dist) {

        // 预热
        for (int i = 0; i < 10; i++) {
            func(prefetch_dist);
        }

        // 计时
        constexpr int ITER = 100;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < ITER; i++) {
            func(prefetch_dist);
        }

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start).count() / ITER;
    }
};

// 使用示例：调优IVF搜索的预取距离
void tune_ivf_prefetch_distance() {
    auto search_func = [](size_t prefetch_dist) {
        // 执行带指定预取距离的搜索
        // ...
    };

    size_t optimal_dist = PrefetchTuner::find_optimal_distance(search_func);
    printf("Recommended IVF prefetch distance: %zu cache lines\n",
           optimal_dist);
}
```

---

## 练习题

1. 实现完整的端到端搜索流程
2. 比较不同索引在真实数据集上的性能
3. 实现自适应nprobe策略
4. 构建分布式搜索系统
5. 实现NUMA感知的索引分配
6. 测试不同编译选项的性能影响
7. 实现False Sharing的检测和修复
8. 编写PGO优化的完整示例

## 结语

恭喜完成14天Faiss深度课程！你现在应该对Faiss的核心算法和实现细节有了深入理解。继续实践和探索，将这些知识应用到实际项目中。

### 性能优化清单

- [ ] 使用SIMD指令（AVX2/AVX-512/NEON）
- [ ] 数据对齐（32字节或64字节）
- [ ] 避免False Sharing
- [ ] 使用预取减少内存延迟
- [ ] 编译器优化选项（-O3 -march=native）
- [ ] 链接时优化（LTO）
- [ ] Profile-Guided Optimization
- [ ] NUMA感知分配
- [ ] 缓存友好的内存访问模式
- [ ] 多线程并行化

### 进一步学习

- 深入研究特定索引的源码
- 阅读CPU微架构手册
- 学习性能分析工具（perf, VTune）
- 实践异步I/O和CUDA优化
- 研究分布式向量搜索算法
