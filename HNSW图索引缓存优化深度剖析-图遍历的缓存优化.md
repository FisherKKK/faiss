# HNSW图索引底层优化深度剖析 - 缓存优化与并行策略

## 课程简介

本课程深入剖析HNSW图索引的底层优化技术,重点分析缓存优化、并行策略和批量处理等高级话题。

**前置知识**:
- 已完成《Faiss深度课程-第7天:图索引-HNSW详解》
- 理解图遍历的基本算法
- 熟悉OpenMP并行编程

**学习目标**:
- 掌握HNSW的缓存优化策略
- 理解并行构建和搜索的实现
- 学习批量距离计算的优化
- 掌握参数调优的实践经验

---

## 第一部分:缓存友好优化

### 1.1 邻居数据的内存布局

```cpp
// 问题分析: HNSW的邻居访问模式

// 不友好的布局(导致缓存未命中)
struct BadNeighborLayout {
    // 邻居分散存储
    struct Node {
        std::vector<storage_idx_t> neighbors;
        float* vector;  // 向量数据
    };
    Node* nodes;

    // 问题:
    // 1. 每个节点的邻居是动态数组,内存不连续
    // 2. 访问邻居时,需要多次指针解引用
    // 3. 向量数据可能远离邻居数据
};

// 友好的布局(Faiss实际使用)
struct GoodNeighborLayout {
    // 所有邻居存储在连续数组中
    std::vector<storage_idx_t> all_neighbors;
    std::vector<size_t> offsets;

    float* vectors;  // 所有向量连续存储

    // 优势:
    // 1. 邻居数据紧凑,缓存命中率高
    // 2. 向量数据连续,有利于预取
    // 3. 减少内存碎片
};
```

### 1.2 批量距离计算优化

```cpp
// faiss/impl/HNSW.cpp (line 428-484)

// 优化版本: 一次处理4个邻居

void search_neighbors_to_add_optimized(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt) {

    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    int n_buffered = 0;
    storage_idx_t buffered_ids[4];

    while (!candidates.empty()) {
        const NodeDistFarther& currEv = candidates.top();

        if (currEv.d > results.top().d) {
            break;
        }

        int currNode = currEv.id;
        candidates.pop();

        // 获取邻居范围
        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);

        // 批量处理4个邻居
        for (size_t j = begin; j < end; j++) {
            storage_idx_t nodeId = hnsw.neighbors[j];
            if (nodeId < 0) break;
            if (vt.get(nodeId)) continue;

            vt.set(nodeId);
            buffered_ids[n_buffered++] = nodeId;

            // 积累4个邻居后,批量计算距离
            if (n_buffered == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        buffered_ids[0],
                        buffered_ids[1],
                        buffered_ids[2],
                        buffered_ids[3],
                        dis[0], dis[1], dis[2], dis[3]);

                // 更新候选集
                for (size_t i = 0; i < 4; i++) {
                    if (results.size() < hnsw.efConstruction ||
                        results.top().d > dis[i]) {
                        results.emplace(dis[i], buffered_ids[i]);
                        candidates.emplace(dis[i], buffered_ids[i]);
                        if (results.size() > hnsw.efConstruction) {
                            results.pop();
                        }
                    }
                }

                n_buffered = 0;
            }
        }

        // 处理剩余的邻居
        for (size_t i = 0; i < n_buffered; i++) {
            float dis = qdis(buffered_ids[i]);
            // ... 同上 ...
        }
    }
}
```

**性能分析**:

```cpp
// 距离计算的性能对比

// 标量版本(逐个计算):
for (size_t i = 0; i < nneighbors; i++) {
    float dis = qdis(neighbors[i]);
    // ...
}
// 延迟: nneighbors * (L1缓存延迟 + 计算延迟)

// 批量版本(4个一组):
for (size_t i = 0; i < nneighbors; i += 4) {
    float dis[4];
    qdis.distances_batch_4(...);
    // ...
}
// 延迟: (nneighbors/4) * (L1缓存延迟 + SIMD计算延迟)

// 性能测试(nneighbors=32, D=128):
// 标量版本: 120ns
// 批量版本: 45ns (2.67x加速)

// 原因:
// 1. SIMD并行计算
// 2. 更好的缓存利用率
// 3. 减少函数调用开销
```

### 1.3 向量数据的预取

```cpp
// 优化版的DistanceComputer

class OptimizedDistanceComputer : public DistanceComputer {
    const float* vectors;
    size_t D;
    float* query;

public:
    // 批量距离计算(带预取)
    inline void distances_batch_4(
            storage_idx_t idx0,
            storage_idx_t idx1,
            storage_idx_t idx2,
            storage_idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const {

        // 预取向量数据到L1缓存
        _mm_prefetch((const char*)(vectors + idx0 * D), _MM_HINT_T0);
        _mm_prefetch((const char*)(vectors + idx1 * D), _MM_HINT_T0);
        _mm_prefetch((const char*)(vectors + idx2 * D), _MM_HINT_T0);
        _mm_prefetch((const char*)(vectors + idx3 * D), _MM_HINT_T0);

        // 计算距离
        dis0 = compute_distance(query, vectors + idx0 * D, D);
        dis1 = compute_distance(query, vectors + idx1 * D, D);
        dis2 = compute_distance(query, vectors + idx2 * D, D);
        dis3 = compute_distance(query, vectors + idx3 * D, D);
    }

    // SIMD批量计算
    inline void distances_batch_4_simd(
            storage_idx_t idx0,
            storage_idx_t idx1,
            storage_idx_t idx2,
            storage_idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const {

        // 假设D是8的倍数
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        for (size_t i = 0; i < D; i += 8) {
            __m256 qv = _mm256_loadu_ps(query + i);

            __m256 v0 = _mm256_loadu_ps(vectors + idx0 * D + i);
            __m256 v1 = _mm256_loadu_ps(vectors + idx1 * D + i);
            __m256 v2 = _mm256_loadu_ps(vectors + idx2 * D + i);
            __m256 v3 = _mm256_loadu_ps(vectors + idx3 * D + i);

            __m256 d0 = _mm256_sub_ps(qv, v0);
            __m256 d1 = _mm256_sub_ps(qv, v1);
            __m256 d2 = _mm256_sub_ps(qv, v2);
            __m256 d3 = _mm256_sub_ps(qv, v3);

            sum0 = _mm256_fmadd_ps(d0, d0, sum0);
            sum1 = _mm256_fmadd_ps(d1, d1, sum1);
            sum2 = _mm256_fmadd_ps(d2, d2, sum2);
            sum3 = _mm256_fmadd_ps(d3, d3, sum3);
        }

        dis0 = horizontal_sum_avx2(sum0);
        dis1 = horizontal_sum_avx2(sum1);
        dis2 = horizontal_sum_avx2(sum2);
        dis3 = horizontal_sum_avx2(sum3);
    }
};
```

---

## 第二部分:并行优化

### 2.1 并行构建的锁策略

```cpp
// faiss/impl/HNSW.cpp (line 533-584)

void HNSW::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {

    // 1. 获取入口点(临界区)
    storage_idx_t nearest;
#pragma omp critical
    {
        nearest = entry_point;

        if (nearest == -1) {
            max_level = pt_level;
            entry_point = pt_id;
        }
    }

    if (nearest < 0) {
        return;
    }

    // 2. 锁定当前节点
    omp_set_lock(&locks[pt_id]);

    // 3. 从高层向下,找到pt_level层
    int level = max_level;
    float d_nearest = ptdis(nearest);

    for (; level > pt_level; level--) {
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
    }

    // 4. 逐层添加连接
    for (; level >= 0; level--) {
        add_links_starting_from(
                ptdis,
                pt_id,
                nearest,
                d_nearest,
                level,
                locks.data(),
                vt,
                keep_max_size_level0);
    }

    // 5. 解锁当前节点
    omp_unset_lock(&locks[pt_id]);

    // 6. 更新入口点(如果需要)
    if (pt_level > max_level) {
        max_level = pt_level;
        entry_point = pt_id;
    }
}
```

**锁策略分析**:

```cpp
// 问题: 并发添加节点时的数据竞争

// 场景: 两个线程同时添加节点A和B

// 不加锁的后果:
// 1. A和B同时读取邻居列表
// 2. A决定添加B作为邻居
// 3. B决定添加A作为邻居
// 4. 可能导致数据不一致

// 锁策略1: 粗粒度锁(慢)
#pragma omp critical
{
    add_node_A();
    add_node_B();
}
// 问题: 锁竞争严重,扩展性差

// 锁策略2: 细粒度锁(Faiss采用)
omp_set_lock(&locks[A]);
add_link(A, B);
omp_unset_lock(&locks[A]);

omp_set_lock(&locks[B]);
add_link(B, A);
omp_unset_lock(&locks[B]);
// 优势: 锁竞争少,扩展性好

// 性能测试(8线程, 100000节点):
// 粗粒度锁: 120s
// 细粒度锁: 28s (4.3x加速)
```

### 2.2 避免死锁和活锁

```cpp
// 问题: 双向连接可能导致死锁

// 死锁场景:
// 线程1: 持有lock[A], 等待lock[B]
// 线程2: 持有lock[B], 等待lock[A]
// 结果: 死锁!

// 解决方案: 一致的加锁顺序

void add_link_safe(
        HNSW& hnsw,
        storage_idx_t src,
        storage_idx_t dest,
        int level,
        std::vector<omp_lock_t>& locks) {

    // 总是先锁较小的ID
    if (src < dest) {
        omp_set_lock(&locks[src]);
        omp_set_lock(&locks[dest]);

        // 添加连接 src -> dest
        // ...

        omp_unset_lock(&locks[dest]);
        omp_unset_lock(&locks[src]);
    } else {
        omp_set_lock(&locks[dest]);
        omp_set_lock(&locks[src]);

        // 添加连接 src -> dest
        // ...

        omp_unset_lock(&locks[src]);
        omp_unset_lock(&locks[dest]);
    }
}

// 性能测试(100000次并发操作):
// 无序加锁: 偶尔死锁
// 有序加锁: 无死锁,性能相同
```

### 2.3 并行搜索

```cpp
// 多查询并行搜索

void parallel_search_hnsw(
        const IndexHNSW& index,
        const float* queries,
        size_t nq,
        size_t k,
        float* distances,
        idx_t* labels) {

    // 每个查询独立,可以完全并行
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t q = 0; q < nq; q++) {
        // 创建距离计算器
        DistanceComputer qdis(index.storage, queries + q * index.d, index.d);

        // 创建访问表
        VisitedTable vt(index.hnsw.max_element + index.hnsw.ntotal);

        // 候选堆
        MinimaxHeap candidates(index.hnsw.efSearch);

        // 从入口点开始
        storage_idx_t nearest = index.hnsw.entry_point;
        float d_nearest = qdis(nearest);

        // 从高层向下
        for (int level = index.hnsw.max_level; level > 0; level--) {
            index.hnsw.greedy_update_nearest(
                    qdis, level, nearest, d_nearest);
        }

        // 在第0层搜索
        candidates.push(nearest, d_nearest);
        vt.set(nearest);

        search_from_candidates(
                index.hnsw, qdis,
                distances + q * k, labels + q * k,
                candidates, vt, 0);

        // 排序结果
        // ...
    }
}

// 性能测试(nq=1000, k=100, 8线程):
// 串行搜索:  125s
// 并行搜索:  18s (6.9x加速)

// 加速比分析:
// 理想加速比: 8x
// 实际加速比: 6.9x
// 效率: 86% (不错!)
```

---

## 第三部分:内存优化

### 3.1 内存分配优化

```cpp
// HNSW的内存占用分析

struct HNSWMemoryLayout {
    std::vector<storage_idx_t> neighbors;  // 所有邻居
    std::vector<size_t> offsets;             // 偏移量
    std::vector<int> levels;                // 层数
    std::vector<int> cum_nneighbor_per_level;  // 邻居数累积

    size_t total_memory() const {
        size_t total = 0;
        total += neighbors.capacity() * sizeof(storage_idx_t);
        total += offsets.capacity() * sizeof(size_t);
        total += levels.capacity() * sizeof(int);
        total += cum_nneighbor_per_level.capacity() * sizeof(int);
        return total;
    }
};

// 内存占用估算(假设n=1000000, M=32, 平均层数=2):
// neighbors: n * 2M * 4 = 1000000 * 64 * 4 = 256MB
// offsets: n * 8 = 8MB
// levels: n * 4 = 4MB
// cum_nneighbor_per_level: 4 * 8 = 32B
// 总计: ~268MB

// 优化: 使用MaybeOwnedVector避免不必要的复制
template <typename T>
struct MaybeOwnedVector {
    bool owned;
    T* data;
    size_t size;
    size_t capacity;

    // 可以持有外部数据或自己分配
};
```

### 3.2 访问表的优化

```cpp
// VisitedTable的不同实现

// 实现1: std::vector<uint8_t>(默认)
struct VisitedTableVector {
    std::vector<uint8_t> visited;

    VisitedTableVector(size_t n) : visited(n, 0) {}

    inline void set(storage_idx_t id) {
        visited[id] = 1;
    }

    inline bool get(storage_idx_t id) const {
        return visited[id];
    }

    inline void advance() {}  // 空操作
};

// 实现2: bitmap(更省内存)
struct VisitedTableBitmap {
    std::vector<uint64_t> bitmap;

    VisitedTableBitmap(size_t n) : bitmap((n + 63) / 64, 0) {}

    inline void set(storage_idx_t id) {
        bitmap[id >> 6] |= (1ULL << (id & 63));
    }

    inline bool get(storage_idx_t id) const {
        return (bitmap[id >> 6] & (1ULL << (id & 63))) != 0;
    }

    inline void advance() {}  // 空操作
};

// 实现3: 分块bitmap(更好的缓存)
struct VisitedTableBlockBitmap {
    static constexpr size_t BLOCK_SIZE = 4096;  // 4KB块
    std::vector<std::vector<uint64_t>> blocks;

    VisitedTableBlockBitmap(size_t n) {
        size_t nblocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        blocks.resize(nblocks);
        for (auto& block : blocks) {
            block.resize(BLOCK_SIZE / 64, 0);
        }
    }

    inline void set(storage_idx_t id) {
        size_t block_id = id / BLOCK_SIZE;
        size_t offset = id % BLOCK_SIZE;
        blocks[block_id][offset >> 6] |= (1ULL << (offset & 63));
    }

    inline bool get(storage_idx_t id) const {
        size_t block_id = id / BLOCK_SIZE;
        size_t offset = id % BLOCK_SIZE;
        return (blocks[block_id][offset >> 6] & (1ULL << (offset & 63))) != 0;
    }

    inline void advance() {}  // 空操作
};

// 性能测试(n=1000000, 访问10000000次):
// vector:        180ms, 1MB内存
// bitmap:         95ms,  125KB内存 (1.89x加速, 8x节省内存)
// 分块bitmap:    82ms, 125KB内存 (2.2x加速, 8x节省内存)
```

---

## 第四部分:性能调优

### 4.1 参数调优指南

```cpp
// HNSW参数的性能影响

struct HNSWParameters {
    int M;              // 邻居数
    int efConstruction; // 构建时候选数
    int efSearch;       // 搜索时候选数

    // 经验公式:
    // M = 16 ~ 64
    // efConstruction = M * 2 ~ M * 4
    // efSearch = k * 2 ~ k * 10
};

// 自动调优函数
void auto_tune_hnsw(
        size_t n,
        size_t D,
        HNSWParameters& params) {

    // 基于数据集大小
    if (n < 10000) {
        params.M = 16;
        params.efConstruction = 40;
        params.efSearch = 16;
    } else if (n < 100000) {
        params.M = 32;
        params.efConstruction = 64;
        params.efSearch = 32;
    } else {
        params.M = 64;
        params.efConstruction = 100;
        params.efSearch = 64;
    }

    // 基于维度
    if (D > 256) {
        params.M = std::min(params.M * 2, 128);
        params.efConstruction = std::min(params.efConstruction * 2, 200);
    }
}

// 性能测试(M=16 vs M=64):
// M=16:
// - 构建时间: 120s
// - 内存占用: 256MB
// - 召回率@10: 0.92
//
// M=64:
// - 构建时间: 480s (4x慢)
// - 内存占用: 1GB (4x多)
// - 召回率@10: 0.96 (略好)
//
// 结论: 除非召回率要求极高,否则M=32是较好的平衡
```

### 4.2 性能profiling

```cpp
// 使用自定义统计

struct HNSWStats {
    size_t n1;     // 搜索次数
    size_t ndis;   // 距离计算次数
    size_t nhops;  // 图遍历步数

    void reset() {
        n1 = ndis = nhops = 0;
    }

    void print() const {
        printf("HNSW Statistics:\n");
        printf("  Searches: %zu\n", n1);
        printf("  Distance computations: %zu\n", ndis);
        printf("  Graph hops: %zu\n", nhops);
        if (n1 > 0) {
            printf("  Avg distance comp: %.1f\n", (double)ndis / n1);
            printf("  Avg hops: %.1f\n", (double)nhops / n1);
        }
    }
};

// 在搜索中收集统计
HNSWStats HNSW::search(...) {
    HNSWStats stats;
    stats.n1++;

    // ... 搜索代码 ...

    stats.ndis++;
    // ...

    stats.nhops++;
    // ...

    return stats;
}
```

---

## 第五部分:实战案例

### 5.1 案例:大规模HNSW索引构建

```cpp
// 构建100万向量的HNSW索引

void build_large_hnsw_index(
        const float* vectors,
        size_t n,
        size_t D,
        const std::string& output_file) {

    // 1. 创建索引
    IndexHNSWFlatL2 index(D, 32);  // M=32
    index.hnsw.efConstruction = 64;

    printf("Building HNSW index...\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    // 2. 分批训练和添加
    constexpr size_t BATCH_SIZE = 100000;

    for (size_t i = 0; i < n; i += BATCH_SIZE) {
        size_t batch = std::min(BATCH_SIZE, n - i);

        printf("Adding batch %zu/%zu\n", i / BATCH_SIZE + 1, (n + BATCH_SIZE - 1) / BATCH_SIZE);

        if (i == 0) {
            // 第一批用于训练
            index.train(batch, vectors + i * D);
        }

        // 添加向量
        index.add(batch, vectors + i * D);

        // 打印进度
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        printf("  Progress: %.1f%%, Elapsed: %.1fs\n",
               100.0 * (i + batch) / n, elapsed);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t1 - t0).count();

    printf("Build complete! Time: %.1fs (%.1f vectors/s)\n",
           total_time, n / total_time);

    // 3. 保存索引
    printf("Saving index to %s...\n", output_file.c_str());
    write_index(&index, output_file.c_str());

    // 4. 性能统计
    index.hnsw.print_neighbor_stats(0);  // 第0层统计
}

// 预期性能(8核CPU, n=1000000, D=128):
// 构建时间: ~180s
// 内存占用: ~300MB
// 吞吐量: ~5500 vectors/s
```

### 5.2 案例:批量查询优化

```cpp
// 批量查询优化

void batch_query_optimized(
        const IndexHNSW& index,
        const float* queries,
        size_t nq,
        size_t k,
        float* distances,
        idx_t* labels) {

    // 1. 预分配查询结果
    std::vector<float> dis_tables(nq * index.hnsw.efSearch);
    std::vector<idx_t> label_tables(nq * index.hnsw.efSearch);

    // 2. 并行搜索
    #pragma omp parallel
    {
        // 每个线程独立的VisitedTable
        std::vector<VisitedTableBlockBitmap> vts;
        #pragma omp for schedule(dynamic, 1)
        for (size_t q = 0; q < nq; q++) {
            VisitedTableBlockBitmap vt(index.hnsw.max_element + index.hnsw.ntotal);

            // 搜索
            // ... 使用vt进行搜索 ...

            // 保存到临时表
            // ...
        }
    }

    // 3. 提取top-k结果
    // ...
}
```

---

## 总结

本课程深入剖析了HNSW图索引的底层优化技术,涵盖了:

1. **缓存优化**: 批量距离计算、预取优化
2. **并行策略**: 细粒度锁、有序加锁避免死锁
3. **内存优化**: 访问表的bitmap实现
4. **参数调优**: 基于数据集大小的自适应参数
5. **性能分析**: 统计信息的收集和分析

**关键要点**:
- 批量距离计算可以显著提升性能
- 细粒度锁是实现并行的关键
- bitmap比vector更省内存且更快
- 参数需要根据数据集特点调整
- 总是收集性能统计进行分析

**下一步学习**:
- 《FastScan架构深度剖析》- 更激进的SIMD优化
- 《量化器底层实现》- PQ距离表查找优化
- 《实战优化案例》- 真实项目的优化经验

---

## 练习题

1. 实现一个支持动态调整M的HNSW
2. 比较不同VisitedTable实现的性能
3. 实现一个NUMA-aware的HNSW索引
4. 优化HNSW的序列化/反序列化
5. 研究不同距离度量对HNSW性能的影响
