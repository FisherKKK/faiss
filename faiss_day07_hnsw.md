# Faiss深度课程 - 第7天：图索引 - HNSW详解

## 课程目标

深入理解Hierarchical Navigable Small World（HNSW）图索引，这是目前最先进的近似最近邻算法之一。

---

## 1. HNSW概述

### 1.1 核心思想

HNSW结合了：
1. **Small World图**：短路径特征
2. **层次结构**：加速收敛
3. **贪婪搜索**：高效遍历

```cpp
// HNSW的多层图结构
Level 2:  ●────●        (稀疏，长连接)
           ╱      ╲
Level 1:  ●───────●────●   (中等密度)
         ╱   ╲   ╱   ╲    ╱
Level 0: ●─●──●─●──●─●──●─●  (稠密，短连接)
```

### 1.2 与NSW的区别

| 特性 | NSW | HNSW |
|------|-----|------|
| 层次 | 单层 | 多层 |
| 起点 | 随机 | 固定入口点 |
| 收敛速度 | 慢 | 快 |
| 构建复杂度 | O(n log n) | O(n log n) |
| 查询复杂度 | O(log n) | O(log n) |

---

## 2. HNSW数据结构

### 2.1 核心成员

```cpp
// faiss/impl/HNSW.h
struct HNSW {
    using storage_idx_t = int32_t;

    // 每层的邻居数（累积）
    std::vector<int> cum_nneighbor_per_level;

    // 每个向量的层数
    std::vector<int> levels;

    // 邻居表的偏移量
    std::vector<size_t> offsets;

    // 所有邻居的存储
    MaybeOwnedVector<storage_idx_t> neighbors;

    // 入口点（最高层的节点）
    storage_idx_t entry_point = -1;

    // 参数
    int efConstruction = 40;  // 构建时的候选数
    int efSearch = 16;        // 搜索时的候选数

    int max_level = -1;       // 最大层数
    RandomGenerator rng;      // 随机数生成器

    // 层分配概率
    std::vector<double> assign_probas;
};
```

### 2.2 邻居存储

```cpp
// 邻居布局
// neighbors[offsets[i] : offsets[i+1]] 包含向量i的所有层邻居

// 示例：向量i在第2层
// neighbors = [
//   level_2_neighbors,  // M个邻居
//   level_1_neighbors,  // M个邻居
//   level_0_neighbors,  // 2M个邻居
// ]

void HNSW::neighbor_range(
        idx_t no,
        int layer_no,
        size_t* begin,
        size_t* end) const {

    size_t o = offsets[no];
    int n = cum_nb_neighbors(layer_no);
    *begin = o + n;
    *end = o + cum_nb_neighbors(layer_no + 1);
}
```

### 2.3 MinimaxHeap

```cpp
// 搜索时维护候选节点的堆
struct MinimaxHeap {
    int n;      // 容量
    int k;      // 当前大小
    int nvalid; // 有效元素数

    std::vector<storage_idx_t> ids;
    std::vector<float> dis;

    explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0),
                                  ids(n), dis(n) {}

    void push(storage_idx_t i, float v) {
        if (k < n) {
            // 堆未满，直接插入
            dis[k] = v;
            ids[k] = i;
            k++;
            // 向上调整
            std::push_heap(ids.begin(), ids.begin() + k,
                          [this](int a, int b) {
                              return C::cmp(dis[a], dis[b]);
                          });
        } else if (C::cmp(v, dis[0])) {
            // 比堆顶更好，替换
            C::set_nearer(dis[0], v);
            ids[0] = i;
            // 向下调整
            heapify<C>(ids.data(), dis.data(), k);
        }
    }

    float max() const {
        return k > 0 ? dis[0] : 0.0f;
    }

    int size() const {
        return k;
    }

    void clear() {
        k = nvalid = 0;
    }
};
```

---

## 3. HNSW构建

### 3.1 层级分配

```cpp
// 随机分配新节点的层数
int HNSW::random_level() {
    double rand_val = rng.rand_float();
    int level = 0;

    // 几何分布：P(level = l) ∝ levelMult^(-l)
    while (rand_val < assign_probas[level] && level < max_level) {
        level++;
        rand_val = rng.rand_float();
    }

    return level;
}

// 初始化层级概率
void HNSW::set_default_probas(int M, float levelMult) {
    assign_probas.clear();
    cum_nneighbor_per_level.clear();

    int level = 0;
    double prob = 1.0;

    while (prob > 1e-9 && level < 32) {  // 最多32层
        assign_probas.push_back(prob);
        prob /= levelMult;
        level++;
    }

    // 设置邻居数
    // level 0: 2M邻居
    // level > 0: M邻居
    for (int l = 0; l < level; l++) {
        int n = (l == 0) ? 2 * M : M;
        cum_nneighbor_per_level.push_back(
            (cum_nneighbor_per_level.empty() ? 0 :
             cum_nneighbor_per_level.back()) + n);
    }
}
```

### 3.2 添加节点

```cpp
void HNSW::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {

    // 1. 在高层找到最近邻
    int max_level = entry_point >= 0 ? levels[entry_point] : -1;
    storage_idx_t nearest = entry_point;
    float d_nearest = std::numeric_limits<float>::infinity();

    for (int level = max_level; level > pt_level; level--) {
        // 在该层贪婪搜索
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
    }

    // 2. 从pt_level层向下，逐层添加连接
    for (int level = std::min(pt_level, max_level);
         level >= 0;
         level--) {
        add_links_starting_from(
            ptdis, pt_id, nearest, d_nearest,
            level, locks.data(), vt, keep_max_size_level0);

        // 如果不是最底层，更新nearest
        if (level > 0) {
            greedy_update_nearest(*this, ptdis, level - 1,
                                 nearest, d_nearest);
        }
    }

    // 3. 更新入口点
    if (pt_level > max_level) {
        entry_point = pt_id;
        max_level = pt_level;
    }

    levels[pt_id] = pt_level + 1;  // level 0 = 1
}
```

### 3.3 添加连接

```cpp
void HNSW::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {

    // 1. 搜索候选邻居
    std::priority_queue<NodeDistCloser> cand;
    std::priority_queue<NodeDistFarther> result;

    // 初始化
    cand.emplace(d_nearest, nearest);
    result.emplace(d_nearest, nearest);
    vt.set(nearest);

    int M = nb_neighbors(level);

    // 2. 贪婪扩展候选集
    while (!cand.empty()) {
        NodeDistCloser curr = cand.top();
        cand.pop();

        float lower_bound = result.top().d;

        if (curr.d > lower_bound) {
            break;  // 找到足够好的邻居
        }

        // 遍历邻居
        size_t begin, end;
        neighbor_range(curr.id, level, &begin, &end);

        for (size_t i = begin; i < end; i++) {
            storage_idx_t nid = neighbors[i];

            if (vt.get(nid)) continue;
            vt.set(nid);

            float dis = ptdis(nid);

            cand.emplace(dis, nid);
            result.emplace(dis, nid);

            // 限制候选集大小
            if (result.size() > efConstruction) {
                result.pop();
            }
        }
    }

    // 3. 选择最优的M个邻居
    std::vector<NodeDistFarther> selected;
    shrink_neighbor_list(ptdis, result, selected, M,
                         keep_max_size_level0);

    // 4. 添加双向连接
    for (const auto& other : selected) {
        size_t begin, end;
        neighbor_range(other.id, level, &begin, &end);
        int nb = end - begin;

        if (nb < M) {
            // 有空位，直接添加
            neighbors.push_back(pt_id);
        } else {
            // 没有空位，尝试替换最远邻居
            // (需要锁保护)
            #pragma omp critical
            {
                // 重新检查（可能已被修改）
                neighbor_range(other.id, level, &begin, &end);

                // 找到最远的邻居
                float max_d = -1;
                size_t max_idx = begin;
                for (size_t i = begin; i < end; i++) {
                    float d = ptdis(neighbors[i]);
                    if (d > max_d) {
                        max_d = d;
                        max_idx = i;
                    }
                }

                // 如果新节点更近，替换
                if (d_nearest < max_d) {
                    neighbors[max_idx] = pt_id;
                }
            }
        }
    }
}
```

---

## 4. HNSW搜索

### 4.1 搜索流程

```cpp
HNSWStats HNSW::search(
        DistanceComputer& qdis,
        const IndexHNSW* index,
        ResultHandler<C>& res,
        VisitedTable& vt,
        const SearchParameters* params) const {

    HNSWStats stats;
    stats.n1++;

    // 1. 从顶层向下，找到第1层的入口点
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for (int level = max_level; level > 0; level--) {
        greedy_update_nearest(*this, qdis, level,
                             nearest, d_nearest);
    }

    // 2. 在第0层进行精确搜索
    MinimaxHeap candidates(efSearch);
    candidates.push(nearest, d_nearest);
    vt.set(nearest);

    search_from_candidates(*this, qdis, res, candidates,
                           vt, stats, 0, 0, params);

    return stats;
}
```

### 4.2 从候选集搜索

```cpp
int search_from_candidates(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler<HNSW::C>& res,
        HNSW::MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params) {

    // candidates初始化包含一些起点
    // nres_in是结果堆中已有的结果数

    int nstep = 0;
    int nres = nres_in;
    float radius = res.heap_dis[0];  // 当前最远距离

    while (candidates.size() > 0) {
        // 取出最近的候选
        float d_cur = candidates.max();
        storage_idx_t cur = candidates.ids[0];
        candidates.pop();

        // 检查是否需要继续
        if (d_cur > radius) {
            break;  // 剩余候选都太远
        }

        // 遍历邻居
        size_t begin, end;
        hnsw.neighbor_range(cur, level, &begin, &end);

        for (size_t i = begin; i < end; i++) {
            storage_idx_t nid = hnsw.neighbors[i];
            stats.ndis++;

            if (vt.get(nid)) continue;
            vt.set(nid);

            float dis = qdis(nid);

            // 检查是否进入结果集
            if (HNSW::C::cmp(dis, radius)) {
                // 更新结果集
                res.add_result(dis, nid);
                radius = res.heap_dis[0];
                nres++;
            }

            // 添加到候选集
            if (candidates.size() < hnsw.efSearch) {
                candidates.push(nid, dis);
            } else if (HNSW::C::cmp(dis, candidates.max())) {
                candidates.push(nid, dis);  // 会替换堆顶
            }
        }

        nstep++;
        stats.nhops++;
    }

    return nres;
}
```

### 4.3 贪婪更新最近点

```cpp
HNSWStats greedy_update_nearest(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        HNSW::storage_idx_t& nearest,
        float& d_nearest) {

    HNSWStats stats;
    VisitedTable vt(hnsw.max_element + hnsw.ntotal);  // 假设
    vt.set(nearest);

    storage_idx_t cur = nearest;

    while (true) {
        size_t begin, end;
        hnsw.neighbor_range(cur, level, &begin, &end);

        bool changed = false;

        for (size_t i = begin; i < end; i++) {
            storage_idx_t nid = hnsw.neighbors[i];
            stats.ndis++;

            if (vt.get(nid)) continue;
            vt.set(nid);

            float dis = qdis(nid);

            if (dis < d_nearest) {
                d_nearest = dis;
                nearest = nid;
                cur = nid;
                changed = true;
            }
        }

        if (!changed) {
            break;  // 局部最优
        }

        stats.nhops++;
    }

    return stats;
}
```

---

## 5. IndexHNSW实现

### 5.1 IndexHNSW结构

```cpp
// faiss/IndexHNSW.h
struct IndexHNSW : Index {
    HNSW hnsw;        // 图结构
    Index* storage;   // 向量存储
    bool own_fields;  // 是否拥有storage

    IndexHNSW(Index* storage, int M = 32)
        : d(storage->d), metric_type(storage->metric_type),
          storage(storage), own_fields(false) {
        hnsw.set_default_probas(M, 1.0 / log2(M));
    }

    void add(idx_t n, const float* x) override {
        storage->add(n, x);

        std::vector<omp_lock_t> locks(ntotal + n);
        for (auto& lock : locks) {
            omp_init_lock(&lock);
        }

        VisitedTable vt(ntotal + n);

        for (idx_t i = 0; i < n; i++) {
            DistanceComputer* dis =
                storage->get_distance_computer();
            dis->set_query(x + i * d);

            int pt_level = hnsw.random_level();
            storage_idx_t pt_id = ntotal + i;

            hnsw.add_with_locks(*dis, pt_level, pt_id,
                               locks, vt);

            delete dis;
        }

        for (auto& lock : locks) {
            omp_destroy_lock(&lock);
        }
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 初始化结果处理器
        HeapHandler<CMax<float, idx_t>, false> handler(
            n, distances, labels, k);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);
            std::vector<DistanceComputer*> dis_per_thread;

            #pragma omp for
            for (idx_t i = 0; i < n; i++) {
                DistanceComputer* qdis =
                    storage->get_distance_computer();
                qdis->set_query(x + i * d);

                handler.begin(i);

                hnsw.search(*qdis, this, handler, vt,
                           static_cast<const SearchParametersHNSW*>(params));

                handler.end();

                delete qdis;
            }
        }
    }
};
```

---

## 6. 性能优化

### 6.1 efSearch参数

```cpp
// efSearch vs 性能/精度权衡
void benchmark_efSearch() {
    IndexHNSWFlat index(d, M=32);

    index.hnsw.efConstruction = 40;  // 构建参数
    index.add(n, xb);

    std::vector<int> efSearches = {10, 20, 40, 80, 160};

    for (int ef : efSearches) {
        index.hnsw.efSearch = ef;

        auto t0 = std::chrono::high_resolution_clock::now();
        index.search(nq, xq, k, distances, labels);
        auto t1 = std::chrono::high_resolution_clock::now();

        double time_ms =
            std::chrono::duration<double>(t1 - t0).count() * 1000;

        float recall = compute_recall(nq, k, labels, ground_truth);

        printf("efSearch=%3d: recall=%.3f, time=%.2f ms\n",
               ef, recall, time_ms);
    }
}
```

### 6.2 M参数

```cpp
// M（每层邻居数）的影响
// M越大：
// - 精度越高
// - 内存越大
// - 构建越慢
// - 搜索稍慢

// 推荐值：
// d <= 100:  M = 16
// d <= 256:  M = 32
// d > 256:   M = 64
```

### 6.3 并行构建

```cpp
// 使用OpenMP并行添加节点
void parallel_add(IndexHNSW& index, idx_t n, const float* x) {
    std::vector<omp_lock_t> locks(index.ntotal + n);

#pragma omp parallel
    {
        #pragma omp for
        for (idx_t i = 0; i < n; i++) {
            DistanceComputer* dis =
                index.storage->get_distance_computer();
            dis->set_query(x + i * index.d);

            int pt_level = index.hnsw.random_level();
            storage_idx_t pt_id = index.ntotal + i;

            VisitedTable vt(index.ntotal + n);

            index.hnsw.add_with_locks(
                *dis, pt_level, pt_id, locks, vt);

            delete dis;
        }
    }
}
```

---

## 7. HNSW变体

### 7.1 IndexHNSWFlat

```cpp
// Flat存储 + HNSW图
struct IndexHNSWFlat : IndexHNSW {
    IndexHNSWFlat(int d, int M, MetricType metric)
        : IndexHNSW(new IndexFlat(d, metric), M) {
        own_fields = true;
    }
};
```

### 7.2 IndexHNSWPQ

```cpp
// PQ压缩 + HNSW图
struct IndexHNSWPQ : IndexHNSW {
    IndexHNSWPQ(int d, int pq_m, int M, int pq_nbits = 8)
        : IndexHNSW(new IndexPQ(d, pq_m, pq_nbits), M) {
        own_fields = true;
    }

    void train(idx_t n, const float* x) override {
        IndexPQ* ipq = static_cast<IndexPQ*>(storage);
        ipq->train(n, x);
    }
};
```

### 7.3 IndexHNSW2Level

```cpp
// IVF + HNSW
struct IndexHNSW2Level : IndexHNSW {
    Index* quantizer;  // 粗量化器
    size_t nlist;

    void flip_to_ivf() {
        // 将HNSW转换为IVF结构
        // 使用HNSW的level 0作为倒排列表
    }
};
```

---

## 8. HNSW底层实现详解

### 8.1 SearchParametersHNSW

```cpp
// faiss/impl/HNSW.h
// HNSW的搜索参数结构
struct SearchParametersHNSW : SearchParameters {
    int efSearch = 16;                      // 搜索时的候选集大小
    bool check_relative_distance = true;    // 是否使用相对距离检查
    bool bounded_queue = true;              // 是否使用有界队列

    ~SearchParametersHNSW() {}
};

// 使用示例
void search_with_params(IndexHNSW& index) {
    SearchParametersHNSW params;
    params.efSearch = 64;
    params.check_relative_distance = true;
    params.bounded_queue = true;

    index.search(nq, xq, k, distances, labels, &params);
}
```

### 8.2 HNSWStats - 统计信息

```cpp
// HNSW搜索统计
struct HNSWStats {
    size_t n1 = 0;      // 搜索的向量数
    size_t n2 = 0;      // 候选列表耗尽的查询数
    size_t ndis = 0;    // 计算的距离数
    size_t nhops = 0;   // 遍历的边数（跳数）

    void reset() {
        n1 = n2 = 0;
        ndis = 0;
        nhops = 0;
    }

    void combine(const HNSWStats& other) {
        n1 += other.n1;
        n2 += other.n2;
        ndis += other.ndis;
        nhops += other.nhops;
    }
};

// 全局统计变量
extern HNSWStats hnsw_stats;
```

### 8.3 neighbor_range详细实现

```cpp
// faiss/impl/HNSW.cpp
// 获取节点no在layer_no层的邻居范围
void HNSW::neighbor_range(
        idx_t no,
        int layer_no,
        size_t* begin,
        size_t* end) const {
    size_t o = offsets[no];
    *begin = o + cum_nb_neighbors(layer_no);
    *end = o + cum_nb_neighbors(layer_no + 1);
}

// cum_nb_neighbors返回第layer层及以下的累积邻居数
int HNSW::cum_nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no];
}

// 示例：
// cum_nneighbor_per_level = [0, 64, 96, 112]
// 表示：
// level 0: 64个邻居
// level 1: 32个邻居（累积96）
// level 2: 16个邻居（累积112）
```

### 8.4 shrink_neighbor_list - 邻居剪枝

```cpp
// faiss/impl/HNSW.cpp
// 枚举从最近到最远的顶点，仅当没有先前邻居比查询更近于该顶点时才保留邻居
void HNSW::shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFarther>& input,
        std::vector<NodeDistFarther>& output,
        int max_size,
        bool keep_max_size_level0) {

    std::vector<NodeDistFarther> outsiders;

    while (input.size() > 0) {
        NodeDistFarther v1 = input.top();
        input.pop();
        float dist_v1_q = v1.d;

        bool good = true;
        // 检查是否与已有邻居太近
        for (NodeDistFarther v2 : output) {
            float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

            // 如果v1和v2的距离小于v1到查询的距离
            // 则说明v2已经是v1的一个好的近似，不需要v1
            if (dist_v1_v2 < dist_v1_q) {
                good = false;
                break;
            }
        }

        if (good) {
            output.push_back(v1);
            if (output.size() >= max_size) {
                return;
            }
        } else if (keep_max_size_level0) {
            outsiders.push_back(v1);
        }
    }

    // 如果需要保持最大大小，添加 outsiders
    size_t idx = 0;
    while (keep_max_size_level0 && (output.size() < max_size) &&
           (idx < outsiders.size())) {
        output.push_back(outsiders[idx++]);
    }
}
```

### 8.5 search_neighbors_to_add - 批量距离计算

```cpp
// faiss/impl/HNSW.cpp
// 搜索要添加的邻居，使用批量距离计算优化
void search_neighbors_to_add(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        bool reference_version) {

    std::priority_queue<NodeDistFarther> candidates;
    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    while (!candidates.empty()) {
        const NodeDistFarther& currEv = candidates.top();
        if (currEv.d > results.top().d) {
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);

        if (reference_version) {
            // 参考版本（简单但慢）
            for (size_t i = begin; i < end; i++) {
                storage_idx_t nodeId = hnsw.neighbors[i];
                if (nodeId < 0) break;
                if (vt.get(nodeId)) continue;
                vt.set(nodeId);

                float dis = qdis(nodeId);
                if (results.size() < hnsw.efConstruction ||
                    results.top().d > dis) {
                    results.emplace(dis, nodeId);
                    candidates.emplace(dis, nodeId);
                    if (results.size() > hnsw.efConstruction) {
                        results.pop();
                    }
                }
            }
        } else {
            // 优化版本：批量处理4个邻居
            auto update_with_candidate = [&](const storage_idx_t idx,
                                             const float dis) {
                if (results.size() < hnsw.efConstruction ||
                    results.top().d > dis) {
                    results.emplace(dis, idx);
                    candidates.emplace(dis, idx);
                    if (results.size() > hnsw.efConstruction) {
                        results.pop();
                    }
                }
            };

            int n_buffered = 0;
            storage_idx_t buffered_ids[4];

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nodeId = hnsw.neighbors[j];
                if (nodeId < 0) break;
                if (vt.get(nodeId)) continue;
                vt.set(nodeId);

                buffered_ids[n_buffered] = nodeId;
                n_buffered += 1;

                if (n_buffered == 4) {
                    // 批量计算4个距离
                    float dis[4];
                    qdis.distances_batch_4(
                            buffered_ids[0],
                            buffered_ids[1],
                            buffered_ids[2],
                            buffered_ids[3],
                            dis[0], dis[1], dis[2], dis[3]);

                    for (size_t id4 = 0; id4 < 4; id4++) {
                        update_with_candidate(buffered_ids[id4], dis[id4]);
                    }

                    n_buffered = 0;
                }
            }

            // 处理剩余的
            for (size_t icnt = 0; icnt < n_buffered; icnt++) {
                float dis = qdis(buffered_ids[icnt]);
                update_with_candidate(buffered_ids[icnt], dis);
            }
        }
    }

    vt.advance();
}
```

### 8.6 MinimaxHeap::pop_min - SIMD优化版本

```cpp
// faiss/impl/HNSW.cpp
// AVX-512优化的pop_min实现
#ifdef __AVX512F__
int HNSW::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);

    int32_t min_idx = -1;
    float min_dis = std::numeric_limits<float>::infinity();

    __m512i min_indices = _mm512_set1_epi32(-1);
    __m512 min_distances = _mm512_set1_ps(std::numeric_limits<float>::infinity());
    __m512i current_indices = _mm512_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i offset = _mm512_set1_epi32(16);

    // 每次处理16个元素
    const int k16 = (k / 16) * 16;
    for (size_t iii = 0; iii < k16; iii += 16) {
        __m512i indices = _mm512_loadu_si512(
                (const __m512i*)(ids.data() + iii));
        __m512 distances = _mm512_loadu_ps(dis.data() + iii);

        // 过滤-1值
        __mmask16 m1mask = _mm512_cmpgt_epi32_mask(
                _mm512_setzero_si512(), indices);

        __mmask16 dmask = _mm512_cmp_ps_mask(
                min_distances, distances, _CMP_LT_OS);
        __mmask16 finalmask = m1mask | dmask;

        min_indices = _mm512_mask_blend_epi32(
                finalmask, current_indices, min_indices);
        min_distances = _mm512_mask_blend_ps(
                finalmask, distances, min_distances);

        current_indices = _mm512_add_epi32(current_indices, offset);
    }

    // 处理剩余元素
    if (k16 != k) {
        const __mmask16 kmask = (1 << (k - k16)) - 1;
        __m512i indices = _mm512_mask_loadu_epi32(
                _mm512_set1_epi32(-1), kmask, ids.data() + k16);
        __m512 distances = _mm512_maskz_loadu_ps(kmask, dis.data() + k16);

        __mmask16 m1mask = _mm512_cmpgt_epi32_mask(
                _mm512_setzero_si512(), indices);
        __mmask16 dmask = _mm512_cmp_ps_mask(
                min_distances, distances, _CMP_LT_OS);
        __mmask16 finalmask = m1mask | dmask;

        min_indices = _mm512_mask_blend_epi32(
                finalmask, current_indices, min_indices);
        min_distances = _mm512_mask_blend_ps(
                finalmask, distances, min_distances);
    }

    // 获取最小距离
    min_dis = _mm512_reduce_min_ps(min_distances);
    __mmask16 mindmask = _mm512_cmpeq_ps_mask(
            min_distances, _mm512_set1_ps(min_dis));
    min_idx = _mm512_mask_reduce_max_epi32(mindmask, min_indices);

    if (min_idx == -1) return -1;

    if (vmin_out) *vmin_out = min_dis;
    int ret = ids[min_idx];
    ids[min_idx] = -1;
    --nvalid;
    return ret;
}

#elif __AVX2__
// AVX2优化版本（每次处理8个元素）
int HNSW::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);

    int32_t min_idx = -1;
    float min_dis = std::numeric_limits<float>::infinity();

    __m256i min_indices = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);
    __m256 min_distances = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i offset = _mm256_set1_epi32(8);

    const int k8 = (k / 8) * 8;
    for (; iii < k8; iii += 8) {
        __m256i indices = _mm256_loadu_si256(
                (const __m256i*)(ids.data() + iii));
        __m256 distances = _mm256_loadu_ps(dis.data() + iii);

        __m256i m1mask = _mm256_cmpgt_epi32(
                _mm256_setzero_si256(), indices);
        __m256i dmask = _mm256_castps_si256(
                _mm256_cmp_ps(min_distances, distances, _CMP_LT_OS));
        __m256 finalmask = _mm256_castsi256_ps(
                _mm256_or_si256(m1mask, dmask));

        min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(current_indices),
                _mm256_castsi256_ps(min_indices),
                finalmask));
        min_distances = _mm256_blendv_ps(
                distances, min_distances, finalmask);

        current_indices = _mm256_add_epi32(current_indices, offset);
    }

    // ... 处理剩余元素

    return min_idx;
}

#else
// 基线非向量化版本
int HNSW::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    // O(n)操作
    int i = k - 1;
    while (i >= 0) {
        if (ids[i] != -1) break;
        i--;
    }
    if (i == -1) return -1;

    int imin = i;
    float vmin = dis[i];
    i--;
    while (i >= 0) {
        if (ids[i] != -1 && dis[i] < vmin) {
            vmin = dis[i];
            imin = i;
        }
        i--;
    }

    if (vmin_out) *vmin_out = vmin;
    int ret = ids[imin];
    ids[imin] = -1;
    --nvalid;
    return ret;
}
#endif
```

### 8.7 greedy_update_nearest

```cpp
// faiss/impl/HNSW.cpp
// 在给定层级上贪婪更新最近邻
HNSWStats greedy_update_nearest(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {

    HNSWStats stats;

    for (;;) {
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);

        auto update_with_candidate = [&](const storage_idx_t idx,
                                         const float dis) {
            if (dis < d_nearest) {
                nearest = idx;
                d_nearest = dis;
            }
        };

        // 批量处理4个邻居
        int n_buffered = 0;
        storage_idx_t buffered_ids[4];

        for (size_t j = begin; j < end; j++) {
            storage_idx_t v = hnsw.neighbors[j];
            if (v < 0) break;
            stats.ndis += 1;

            buffered_ids[n_buffered] = v;
            n_buffered += 1;

            if (n_buffered == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        buffered_ids[0], buffered_ids[1],
                        buffered_ids[2], buffered_ids[3],
                        dis[0], dis[1], dis[2], dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    update_with_candidate(buffered_ids[id4], dis[id4]);
                }

                n_buffered = 0;
            }
        }

        // 处理剩余的
        for (size_t icnt = 0; icnt < n_buffered; icnt++) {
            float dis = qdis(buffered_ids[icnt]);
            update_with_candidate(buffered_ids[icnt], dis);
        }

        stats.nhops += 1;

        // 如果没有更近的邻居，停止
        if (nearest == prev_nearest) {
            return stats;
        }
    }
}
```

---

## 9. 第7天总结

### 关键概念

1. **HNSW结构**：多层Small World图
2. **层级分配**：几何分布随机分配
3. **贪婪搜索**：从顶层向下搜索
4. **efSearch**：搜索时的候选集大小
5. **efConstruction**：构建时的候选集大小

### 参数推荐

| 参数 | 推荐值 | 影响 |
|------|--------|------|
| M | 16-64 | 邻居数，影响精度和内存 |
| efConstruction | 40 | 构建质量 |
| efSearch | 16-64 | 搜索精度vs速度 |

### 性能特点

- **构建**：O(n log n)
- **搜索**：O(log n)
- **内存**：O(n × M)
- **精度**：接近100%

### 下一步

第8天将学习**NSG和NNDescent**，其他高效的图索引算法。

---

## 10. HNSW源码深度实现

### 10.1 VisitedTable - 访问标记表

```cpp
// faiss/impl/AuxIndexStructures.h
// 使用访问计数而非布尔标记，避免每次重置时O(n)的memset操作

struct VisitedTable {
    std::vector<uint8_t> visited;  // 访问标记数组
    uint8_t visno;                 // 当前访问计数

    explicit VisitedTable(int size) : visited(size), visno(1) {}

    // 标记节点为已访问
    void set(int no) {
        visited[no] = visno;
    }

    // 检查节点是否已访问
    bool get(int no) const {
        return visited[no] == visno;
    }

    // 重置所有标记（O(1)操作，只需递增计数器）
    void advance() {
        visno++;
        if (visno == 250) {  // 避免溢出到255
            memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
            visno = 1;
        }
    }
};

// 使用示例
VisitedTable vt(ntotal);
vt.set(node_id);              // 标记已访问
if (vt.get(candidate_id)) {   // 检查是否已访问
    continue;                 // 跳过已访问节点
}
vt.advance();                 // 每次搜索后重置
```

### 10.2 DistanceComputer接口 - 距离计算抽象

```cpp
// faiss/impl/DistanceComputer.h
// 距离计算器接口，支持批量SIMD优化

struct DistanceComputer {
    // 设置查询向量（指针需在调用operator()期间保持有效）
    virtual void set_query(const float* x) = 0;

    // 计算当前查询到向量i的距离
    virtual float operator()(idx_t i) = 0;

    // 批量计算4个距离（可被重写以利用SIMD）
    virtual void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) {
        // 默认实现：逐个计算
        dis0 = this->operator()(idx0);
        dis1 = this->operator()(idx1);
        dis2 = this->operator()(idx2);
        dis3 = this->operator()(idx3);
    }

    // 计算两个存储向量之间的距离
    virtual float symmetric_dis(idx_t i, idx_t j) = 0;

    virtual ~DistanceComputer() {}
};

// FlatCodesDistanceComputer - 用于Panorama优化
struct FlatCodesDistanceComputer : DistanceComputer {
    const uint8_t* codes;
    size_t code_size;
    const float* q;  // 当前查询向量

    // 计算部分点积（用于Panorama渐进式剪枝）
    virtual float partial_dot_product(
            const idx_t i,
            const uint32_t offset,
            const uint32_t num_components) {
        FAISS_THROW_MSG("partial_dot_product not implemented");
    }

    // 批量计算4个部分点积
    virtual void partial_dot_product_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dp0,
            float& dp1,
            float& dp2,
            float& dp3,
            const uint32_t offset,
            const uint32_t num_components) {
        dp0 = this->partial_dot_product(idx0, offset, num_components);
        dp1 = this->partial_dot_product(idx1, offset, num_components);
        dp2 = this->partial_dot_product(idx2, offset, num_components);
        dp3 = this->partial_dot_product(idx3, offset, num_components);
    }
};
```

### 10.3 prefetch优化 - 内存预取

```cpp
// faiss/utils/prefetch.h
// 跨平台的内存预取宏

#ifdef __AVX__
// x86 AVX平台
#include <xmmintrin.h>

inline void prefetch_L1(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T0);  // L1缓存
}
inline void prefetch_L2(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T1);  // L2缓存
}
inline void prefetch_L3(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T2);  // L3缓存
}

#elif defined(__aarch64__)
// ARM64平台
inline void prefetch_L2(const void* address) {
    __builtin_prefetch(address, 0, 2);  // locality=2, L2缓存
}
#endif

// HNSW中的预取使用
for (size_t j = begin; j < end; j++) {
    int v1 = hnsw.neighbors[j];
    if (v1 < 0) break;
    // 预取visited标记，减少缓存未命中
    prefetch_L2(vt.visited.data() + v1);
    jmax += 1;
}
```

### 10.4 add_link - 添加双向连接

```cpp
// faiss/impl/HNSW.cpp
// 在两个节点之间添加连接，可能需要收缩邻居列表

void add_link(
        HNSW& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level,
        bool keep_max_size_level0) {

    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);

    // 检查是否有空位
    if (hnsw.neighbors[end - 1] == -1) {
        // 有空位，从后往前找第一个非空位置
        size_t i = end;
        while (i > begin) {
            if (hnsw.neighbors[i - 1] != -1) {
                break;
            }
            i--;
        }
        hnsw.neighbors[i] = dest;
        return;
    }

    // 没有空位，需要选择保留哪些邻居
    // 使用优先队列按距离排序
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) {
        storage_idx_t neigh = hnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    // 收缩邻居列表到max_size
    shrink_neighbor_list(qdis, resultSet, end - begin, keep_max_size_level0);

    // 写回邻居
    size_t i = begin;
    while (resultSet.size()) {
        hnsw.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }

    // 填充-1表示空位
    while (i < end) {
        hnsw.neighbors[i++] = -1;
    }
}
```

### 10.5 add_links_starting_from - 完整实现

```cpp
// faiss/impl/HNSW.cpp
void HNSW::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {

    // 1. 搜索候选邻居
    std::priority_queue<NodeDistCloser> link_targets;
    search_neighbors_to_add(
        *this, ptdis, link_targets, nearest, d_nearest, level, vt);

    // 2. 限制邻居数量
    int M = nb_neighbors(level);
    ::faiss::shrink_neighbor_list(ptdis, link_targets, M, keep_max_size_level0);

    // 3. 收集要连接的节点
    std::vector<storage_idx_t> neighbors_to_add;
    neighbors_to_add.reserve(link_targets.size());
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        add_link(*this, ptdis, pt_id, other_id, level, keep_max_size_level0);
        neighbors_to_add.push_back(other_id);
        link_targets.pop();
    }

    // 4. 添加反向连接（需要锁定邻居节点）
    omp_unset_lock(&locks[pt_id]);  // 临时释放自己的锁
    for (storage_idx_t other_id : neighbors_to_add) {
        omp_set_lock(&locks[other_id]);
        add_link(*this, ptdis, other_id, pt_id, level, keep_max_size_level0);
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);  // 重新获取自己的锁
}
```

### 10.6 search_from_candidates - 完整搜索实现

```cpp
// faiss/impl/HNSW.cpp
int search_from_candidates(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler<HNSW::C>& res,
        HNSW::MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params) {

    int nres = nres_in;
    int ndis = 0;

    // 提取搜索参数
    bool do_dis_check;
    int efSearch;
    const IDSelector* sel;
    extract_search_params(hnsw, params, do_dis_check, efSearch, sel);

    // 初始化：将候选集加入结果
    HNSW::C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (d < threshold) {
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    // BFS遍历候选集
    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        // 停止条件检查
        if (do_dis_check) {
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;  // 已有足够的更近距离
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        // 预取visited标记
        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) break;
            prefetch_L2(vt.visited.data() + v1);
            jmax += 1;
        }

        int counter = 0;
        size_t saved_j[4];
        threshold = res.threshold;

        // 处理邻居（批量4个）
        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (!sel || sel->is_member(idx)) {
                if (dis < threshold) {
                    if (res.add_result(dis, idx)) {
                        threshold = res.threshold;
                        nres += 1;
                    }
                }
            }
            candidates.push(idx, dis);
        };

        for (size_t j = begin; j < jmax; j++) {
            int v1 = hnsw.neighbors[j];

            bool vget = vt.get(v1);
            vt.set(v1);
            saved_j[counter] = v1;
            counter += vget ? 0 : 1;

            // 批量计算4个距离
            if (counter == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        saved_j[0], saved_j[1], saved_j[2], saved_j[3],
                        dis[0], dis[1], dis[2], dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    add_to_heap(saved_j[id4], dis[id4]);
                }
                ndis += 4;
                counter = 0;
            }
        }

        // 处理剩余的
        for (size_t icnt = 0; icnt < counter; icnt++) {
            float dis = qdis(saved_j[icnt]);
            add_to_heap(saved_j[icnt], dis);
            ndis += 1;
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nhops += nstep;
    }

    return nres;
}
```

### 10.7 HNSW::search - 完整搜索流程

```cpp
// faiss/impl/HNSW.cpp
HNSWStats HNSW::search(
        DistanceComputer& qdis,
        const IndexHNSW* index,
        ResultHandler<C>& res,
        VisitedTable& vt,
        const SearchParameters* params) const {

    HNSWStats stats;
    if (entry_point == -1) {
        return stats;
    }
    int k = extract_k_from_ResultHandler(res);

    // 提取搜索参数
    bool bounded_queue = this->search_bounded_queue;
    int efSearch = this->efSearch;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            bounded_queue = hnsw_params->bounded_queue;
            efSearch = hnsw_params->efSearch;
        }
    }

    // 第一阶段：从顶层向下贪婪搜索
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for (int level = max_level; level >= 1; level--) {
        HNSWStats local_stats =
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        stats.combine(local_stats);
    }

    // 第二阶段：在第0层精确搜索
    int ef = std::max(efSearch, k);

    if (bounded_queue) {
        // 使用有界队列（标准HNSW）
        MinimaxHeap candidates(ef);
        candidates.push(nearest, d_nearest);

        if (!is_panorama) {
            search_from_candidates(
                    *this, qdis, res, candidates, vt, stats, 0, 0, params);
        } else {
            // Panorama渐进式剪枝
            search_from_candidates_panorama(
                    *this, index, qdis, res, candidates, vt,
                    stats, 0, 0, params);
        }
    } else {
        // 使用无界队列
        std::priority_queue<Node> top_candidates =
                search_from_candidate_unbounded(
                        *this, Node(d_nearest, nearest), qdis, ef, &vt, stats);

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }

        while (!top_candidates.empty()) {
            float d;
            storage_idx_t label;
            std::tie(d, label) = top_candidates.top();
            res.add_result(d, label);
            top_candidates.pop();
        }
    }

    vt.advance();
    return stats;
}
```

### 10.8 MinimaxHeap::pop_min - AVX-512优化实现

```cpp
// faiss/impl/HNSW.cpp
#ifdef __AVX512F__
// 每次处理16个元素，使用AVX-512指令
int HNSW::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);

    int32_t min_idx = -1;
    float min_dis = std::numeric_limits<float>::infinity();

    __m512i min_indices = _mm512_set1_epi32(-1);
    __m512 min_distances = _mm512_set1_ps(std::numeric_limits<float>::infinity());
    __m512i current_indices = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                                 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i offset = _mm512_set1_epi32(16);

    // 处理16的倍数个元素
    const int k16 = (k / 16) * 16;
    for (size_t iii = 0; iii < k16; iii += 16) {
        __m512i indices = _mm512_loadu_si512(
                (const __m512i*)(ids.data() + iii));
        __m512 distances = _mm512_loadu_ps(dis.data() + iii);

        // 过滤-1值（已删除的元素）
        __mmask16 m1mask = _mm512_cmpgt_epi32_mask(
                _mm512_setzero_si512(), indices);

        // 比较距离
        __mmask16 dmask = _mm512_cmp_ps_mask(
                min_distances, distances, _CMP_LT_OS);
        __mmask16 finalmask = m1mask | dmask;

        // 选择更小的距离
        min_indices = _mm512_mask_blend_epi32(
                finalmask, current_indices, min_indices);
        min_distances = _mm512_mask_blend_ps(
                finalmask, distances, min_distances);

        current_indices = _mm512_add_epi32(current_indices, offset);
    }

    // 处理剩余元素
    if (k16 != k) {
        const __mmask16 kmask = (1 << (k - k16)) - 1;
        __m512i indices = _mm512_mask_loadu_epi32(
                _mm512_set1_epi32(-1), kmask, ids.data() + k16);
        __m512 distances = _mm512_maskz_loadu_ps(kmask, dis.data() + k16);

        __mmask16 m1mask = _mm512_cmpgt_epi32_mask(
                _mm512_setzero_si512(), indices);
        __mmask16 dmask = _mm512_cmp_ps_mask(
                min_distances, distances, _CMP_LT_OS);
        __mmask16 finalmask = m1mask | dmask;

        min_indices = _mm512_mask_blend_epi32(
                finalmask, current_indices, min_indices);
        min_distances = _mm512_mask_blend_ps(
                finalmask, distances, min_distances);
    }

    // 提取最小距离
    min_dis = _mm512_reduce_min_ps(min_distances);
    __mmask16 mindmask = _mm512_cmpeq_ps_mask(
            min_distances, _mm512_set1_ps(min_dis));
    min_idx = _mm512_mask_reduce_max_epi32(mindmask, min_indices);

    if (min_idx == -1) return -1;

    if (vmin_out) *vmin_out = min_dis;
    int ret = ids[min_idx];
    ids[min_idx] = -1;
    --nvalid;
    return ret;
}
#endif
```

### 10.9 search_from_candidates_panorama - Panorama优化

```cpp
// faiss/impl/HNSW.cpp
// Panorama渐进式剪枝：对高维向量更高效
int search_from_candidates_panorama(
        const HNSW& hnsw,
        const IndexHNSW* index,
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params) {

    // 获取Panorama索引
    const auto* panorama_index =
            dynamic_cast<const IndexHNSWFlatPanorama*>(index);
    auto* flat_codes_qdis = dynamic_cast<FlatCodesDistanceComputer*>(&qdis);

    // 分配临时数组
    size_t M = hnsw.nb_neighbors(0);
    std::vector<idx_t> index_array(M);
    std::vector<float> exact_distances(M);

    // 计算查询的累积和
    const float* query = flat_codes_qdis->q;
    std::vector<float> query_cum_sums(panorama_index->pano.n_levels + 1);
    panorama_index->pano.compute_query_cum_sums(query, query_cum_sums.data());
    float query_norm_sq = query_cum_sums[0] * query_cum_sums[0];

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        // 收集候选并初始化距离
        size_t initial_size = 0;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) break;

            const float* cum_sums_v1 = panorama_index->get_cum_sum(v1);
            index_array[initial_size] = v1;
            // 初始化精确距离上界
            exact_distances[initial_size] =
                    query_norm_sq + cum_sums_v1[0] * cum_sums_v1[0];

            bool is_selected = !sel || sel->is_member(v1);
            initial_size += is_selected && !vt.get(v1) ? 1 : 0;
            vt.set(v1);
        }

        // 渐进式细化
        size_t batch_size = initial_size;
        size_t curr_panorama_level = 0;
        const size_t num_panorama_levels = panorama_index->pano.n_levels;

        while (curr_panorama_level < num_panorama_levels && batch_size > 0) {
            float query_cum_norm = query_cum_sums[curr_panorama_level + 1];

            size_t start_dim = curr_panorama_level *
                    panorama_index->pano.level_width_floats;
            size_t end_dim = (curr_panorama_level + 1) *
                    panorama_index->pano.level_width_floats;
            end_dim = std::min(end_dim, static_cast<size_t>(panorama_index->d));

            // 批量处理4个候选
            size_t i = 0;
            size_t next_batch_size = 0;
            for (; i + 3 < batch_size; i += 4) {
                idx_t idx_0 = index_array[i];
                idx_t idx_1 = index_array[i + 1];
                idx_t idx_2 = index_array[i + 2];
                idx_t idx_3 = index_array[i + 3];

                float dp[4];
                flat_codes_qdis->partial_dot_product_batch_4(
                        idx_0, idx_1, idx_2, idx_3,
                        dp[0], dp[1], dp[2], dp[3],
                        start_dim, end_dim - start_dim);
                ndis += 4;

                // 更新精确距离
                float new_exact_0 = exact_distances[i + 0] - 2 * dp[0];
                float new_exact_1 = exact_distances[i + 1] - 2 * dp[1];
                float new_exact_2 = exact_distances[i + 2] - 2 * dp[2];
                float new_exact_3 = exact_distances[i + 3] - 2 * dp[3];

                // 获取累积和用于计算下界
                float cum_sum_0 = panorama_index->get_cum_sum(
                        idx_0)[curr_panorama_level + 1];
                float cum_sum_1 = panorama_index->get_cum_sum(
                        idx_1)[curr_panorama_level + 1];
                float cum_sum_2 = panorama_index->get_cum_sum(
                        idx_2)[curr_panorama_level + 1];
                float cum_sum_3 = panorama_index->get_cum_sum(
                        idx_3)[curr_panorama_level + 1];

                // 计算Cauchy-Schwarz下界
                float cs_bound_0 = 2.0f * cum_sum_0 * query_cum_norm;
                float cs_bound_1 = 2.0f * cum_sum_1 * query_cum_norm;
                float cs_bound_2 = 2.0f * cum_sum_2 * query_cum_norm;
                float cs_bound_3 = 2.0f * cum_sum_3 * query_cum_norm;

                float lower_bound_0 = new_exact_0 - cs_bound_0;
                float lower_bound_1 = new_exact_1 - cs_bound_1;
                float lower_bound_2 = new_exact_2 - cs_bound_2;
                float lower_bound_3 = new_exact_3 - cs_bound_3;

                // 基于下界剪枝
                if (lower_bound_0 <= threshold) {
                    exact_distances[next_batch_size] = new_exact_0;
                    index_array[next_batch_size] = idx_0;
                    next_batch_size += 1;
                } else {
                    candidates.push(idx_0, new_exact_0);
                }
                // ... 处理其他3个候选
            }

            batch_size = next_batch_size;
            curr_panorama_level++;
        }

        // 添加幸存候选到结果
        for (size_t i = 0; i < batch_size; i++) {
            idx_t idx = index_array[i];
            if (res.add_result(exact_distances[i], idx)) {
                nres += 1;
            }
            candidates.push(idx, exact_distances[i]);
        }
    }

    return nres;
}
```

### 10.10 生产级使用示例

```cpp
// 生产环境HNSW索引配置
#include <faiss/IndexHNSW.h>

void production_hnsw_example() {
    int d = 128;          // 维度
    int M = 32;           // 每层邻居数
    idx_t ntotal = 1000000;

    // 1. 创建索引
    faiss::IndexHNSWFlat index(d, M);

    // 2. 配置构建参数
    index.hnsw.efConstruction = 40;   // 构建质量
    index.hnsw.efSearch = 16;          // 搜索速度

    // 3. 添加向量
    std::vector<float> xb(d * ntotal);
    // ... 填充xb ...
    index.add(ntotal, xb.data());

    // 4. 搜索
    idx_t nq = 100;
    idx_t k = 100;
    std::vector<float> xq(d * nq);
    std::vector<float> distances(k * nq);
    std::vector<idx_t> labels(k * nq);

    // 使用动态efSearch
    faiss::SearchParametersHNSW params;
    params.efSearch = 64;  // 提高精度

    index.search(nq, xq.data(), k,
                 distances.data(), labels.data(), &params);

    // 5. 性能分析
    faiss::HNSWStats stats = faiss::hnsw_stats;
    printf("Distance computations: %zu\n", stats.ndis);
    printf("Hops: %zu\n", stats.nhops);
    printf("Queries processed: %zu\n", stats.n1);

    // 6. 保存索引
    {
        faiss::IOWriter* writer = new faiss::IOFile("hnsw.index", "wb");
        faiss::write_index(&index, writer);
        delete writer;
    }
}

// 并行构建优化
void parallel_build_hnsw() {
    int d = 128;
    int M = 32;
    idx_t ntotal = 1000000;

    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = 40;

    // 批量添加，利用OpenMP并行
    idx_t batch_size = 10000;
    std::vector<float> xb(d * batch_size);

    for (idx_t i = 0; i < ntotal; i += batch_size) {
        idx_t current_batch = std::min(batch_size, ntotal - i);
        // ... 填充xb ...
        index.add(current_batch, xb.data());
    }
}
```

---

## 11. HNSW底层SIMD优化

### 11.1 SIMD优化的邻居遍历

```cpp
// faiss/impl/HNSW.cpp
// 使用SIMD批量计算邻居距离，减少函数调用开销

#ifdef __AVX2__
// 批量处理8个邻居的距离计算
inline void process_neighbors_batch_8(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        const storage_idx_t* neighbor_ids,
        float* distances_out,
        const VisitedTable& vt,
        int count) {

    // 预取visited标记到L1缓存
    for (int i = 0; i < count; i++) {
        _mm_prefetch((const char*)(vt.visited.data() + neighbor_ids[i]),
                     _MM_HINT_T0);
    }

    // 批量计算距离（假设DistanceComputer支持批量）
    if (count >= 8) {
        qdis.distances_batch_8(
                neighbor_ids[0], neighbor_ids[1],
                neighbor_ids[2], neighbor_ids[3],
                neighbor_ids[4], neighbor_ids[5],
                neighbor_ids[6], neighbor_ids[7],
                distances_out[0], distances_out[1],
                distances_out[2], distances_out[3],
                distances_out[4], distances_out[5],
                distances_out[6], distances_out[7]);
    } else {
        // 处理剩余元素
        for (int i = 0; i < count; i++) {
            distances_out[i] = qdis(neighbor_ids[i]);
        }
    }
}
#endif

// 优化的BFS遍历模板
template<int BATCH_SIZE>
void bfs_traverse_optimized(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler<HNSW::C>& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level) {

    alignas(64) storage_idx_t neighbor_buffer[BATCH_SIZE];
    alignas(64) float distance_buffer[BATCH_SIZE];

    while (candidates.size() > 0) {
        float d_cur = candidates.max();
        storage_idx_t cur = candidates.ids[0];
        candidates.pop();

        if (d_cur > res.heap_dis[0]) {
            break;
        }

        size_t begin, end;
        hnsw.neighbor_range(cur, level, &begin, &end);

        // 批量处理邻居
        int buffer_count = 0;
        for (size_t i = begin; i < end; i++) {
            storage_idx_t nid = hnsw.neighbors[i];
            if (nid < 0) break;

            if (!vt.get(nid)) {
                vt.set(nid);
                neighbor_buffer[buffer_count++] = nid;
            }

            // 达到批量大小时处理
            if (buffer_count == BATCH_SIZE) {
                process_neighbors_batch_8(
                        hnsw, qdis, neighbor_buffer,
                        distance_buffer, vt, buffer_count);

                for (int i = 0; i < buffer_count; i++) {
                    stats.ndis++;
                    if (HNSW::C::cmp(distance_buffer[i], res.heap_dis[0])) {
                        res.add_result(distance_buffer[i], neighbor_buffer[i]);
                    }
                    candidates.push(neighbor_buffer[i], distance_buffer[i]);
                }
                buffer_count = 0;
            }
        }

        // 处理剩余的邻居
        if (buffer_count > 0) {
            for (int i = 0; i < buffer_count; i++) {
                float dis = qdis(neighbor_buffer[i]);
                stats.ndis++;

                if (HNSW::C::cmp(dis, res.heap_dis[0])) {
                    res.add_result(dis, neighbor_buffer[i]);
                }
                candidates.push(neighbor_buffer[i], dis);
            }
        }

        stats.nhops++;
    }
}
```

### 11.2 SIMD优化的距离表预计算

```cpp
// 对于PQ压缩的HNSW，可以预计算部分距离表
struct HNSWPQDistanceTable {
    alignas(64) float distance_table[256];  // 256个质心距离

    // SIMD初始化距离表
#ifdef __AVX2__
    void init_table_avx2(
            const float* query,
            const float* centroids,
            int dsub) {

        __m256i zero = _mm256_setzero_si256();
        __m256 query_vec;

        for (int k = 0; k < 256; k += 8) {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();

            const float* ck = centroids + k * dsub;
            for (int j = 0; j + 8 <= dsub; j += 8) {
                query_vec = _mm256_loadu_ps(query + j);
                __m256 ck0 = _mm256_loadu_ps(ck + j + 0);
                __m256 ck1 = _mm256_loadu_ps(ck + j + 8);

                __m256 diff0 = _mm256_sub_ps(query_vec, ck0);
                __m256 diff1 = _mm256_sub_ps(query_vec, ck1);

                sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
                sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
            }

            // 水平求和
            __m256 sum = _mm256_add_ps(sum0, sum1);
            sum = _mm256_hadd_ps(sum, sum);
            sum = _mm256_hadd_ps(sum, sum);

            // 存储到距离表
            _mm256_storeu_ps(distance_table + k, sum);
        }
    }
#endif
};
```

### 11.3 向量化的候选选择

```cpp
// SIMD优化的候选节点选择
#ifdef __AVX512F__
inline void filter_candidates_avx512(
        const float* distances,
        const float threshold,
        uint8_t* mask,
        int count) {

    const int stride = 16;  // AVX-512处理16个float
    int i = 0;

    for (; i + stride <= count; i += stride) {
        __m512 dist_vec = _mm512_loadu_ps(distances + i);
        __m512 thresh_vec = _mm512_set1_ps(threshold);

        // 比较：dist < threshold
        __mmask16 cmp_mask = _mm512_cmp_ps_mask(
                dist_vec, thresh_vec, _CMP_LT_OS);

        // 存储掩码
        mask[i / 16] = static_cast<uint8_t>(cmp_mask);
    }

    // 处理剩余元素
    for (; i < count; i++) {
        mask[i / 8] |= (distances[i] < threshold) << (i % 8);
    }
}

// 使用SIMD掩码快速过滤候选
void add_filtered_candidates_simd(
        MinimaxHeap& heap,
        const storage_idx_t* ids,
        const float* distances,
        int count,
        float threshold) {

    alignas(64) uint8_t mask[16] = {0};
    filter_candidates_avx512(distances, threshold, mask, count);

    // 批量添加通过过滤的候选
    for (int i = 0; i < count; i += 16) {
        uint16_t m = reinterpret_cast<uint16_t*>(mask)[i / 16];

        // 遍历掩码中的位
        while (m) {
            int bit = __builtin_ctz(m);  // 计数尾随零
            int idx = i + bit;

            heap.push(ids[idx], distances[idx]);
            m &= ~(1 << bit);  // 清除已处理的位
        }
    }
}
#endif
```

---

## 12. HNSW内存布局深度优化

### 12.1 缓存行对齐的邻居存储

```cpp
// faiss/impl/HNSW.h
// 优化邻居存储布局，减少false sharing

struct HNSW::NeighborStorage {
    // 每个节点的邻居数据单独缓存行对齐
    struct alignas(64) AlignedNeighbors {
        storage_idx_t neighbors[MAX_NEIGHBORS];
        float distances[MAX_NEIGHBORS];  // 可选：缓存距离
        uint8_t padding[64 - sizeof(storage_idx_t) * MAX_NEIGHBORS
                        - (sizeof(float) * MAX_NEIGHBORS) % 64];
    };

    std::vector<AlignedNeighbors> layer_neighbors;

    // 避免false sharing：每个线程访问不同的缓存行
    void allocate_thread_local(int nthreads) {
        // 为每个线程预留独立的邻居区域
        size_t per_thread_capacity = 1024;
        layer_neighbors.resize(nthreads * per_thread_capacity);
    }
};

// 紧凑的邻居存储（节省内存）
struct HNSW::CompactNeighborStorage {
    // 位打包的邻居ID和层信息
    union NeighborPacked {
        struct {
            uint32_t node_id : 24;  // 支持最多16M节点
            uint32_t layer : 5;     // 最多32层
            uint32_t reserved : 3;
        };
        uint32_t value;
    };

    std::vector<NeighborPacked> packed_neighbors;

    // SIMD友好的访问
#ifdef __AVX2__
    void unpack_neighbors_avx2(
            size_t offset,
            storage_idx_t* output,
            int count) {

        int i = 0;
        for (; i + 8 <= count; i += 8) {
            __m256i packed = _mm256_loadu_si256(
                    (const __m256i*)(packed_neighbors.data() + offset + i));

            // 提取node_id字段（低24位）
            __m256i mask = _mm256_set1_epi32(0xFFFFFF);
            __m256i unpacked = _mm256_and_si256(packed, mask);

            _mm256_storeu_si256((__m256i*)(output + i), unpacked);
        }

        // 处理剩余元素
        for (; i < count; i++) {
            output[i] = packed_neighbors[offset + i].node_id;
        }
    }
#endif
};
```

### 12.2 NUMA感知的HNSW分配

```cpp
// faiss/impl/HNSW.cpp
// NUMA（Non-Uniform Memory Access）优化
#ifdef __linux__
#include <numa.h>

struct NUMAAwareHNSW {
    int numa_nodes;

    // 为每个NUMA节点分配独立的邻居存储
    std::vector<void*> numa_neighbors;

    void init_numa(int nodes) {
        numa_nodes = nodes;
        numa_neighbors.resize(nodes);

        for (int node = 0; node < nodes; node++) {
            // 在指定NUMA节点上分配内存
            numa_neighbors[node] = numa_alloc_onnode(
                    sizeof(AlignedNeighbors) * max_elements_per_node,
                    node);
        }
    }

    // 获取当前线程所在NUMA节点
    int get_current_numa_node() {
        return numa_node_of_cpu(sched_getcpu());
    }

    // 在本地NUMA节点上分配邻居
    storage_idx_t allocate_neighbor_local(storage_idx_t node_id) {
        int numa_node = get_current_numa_node();
        size_t offset = numa_node * max_elements_per_node;

        // 分配在本地NUMA节点的内存上
        AlignedNeighbors* neighbors =
                static_cast<AlignedNeighbors*>(numa_neighbors[numa_node]);

        return offset + (node_id % max_elements_per_node);
    }

    ~NUMAAwareHNSW() {
        for (int node = 0; node < numa_nodes; node++) {
            numa_free(numa_neighbors[node],
                      sizeof(AlignedNeighbors) * max_elements_per_node);
        }
    }
};
#endif
```

### 12.3 预取优化的图遍历

```cpp
// 预取优化的HNSW遍历
template<int PREFETCH_DISTANCE = 4>
struct PrefetchingHNSWIterator {
    const HNSW& hnsw;
    DistanceComputer& qdis;

    // 预取队列
    struct alignas(64) PrefetchEntry {
        storage_idx_t node_id;
        float distance;
    };

    PrefetchEntry prefetch_queue[PREFETCH_DISTANCE];
    int queue_head = 0;
    int queue_tail = 0;

    void prefetch_neighbors(storage_idx_t node, int level) {
        size_t begin, end;
        hnsw.neighbor_range(node, level, &begin, &end);

        // 预取邻居数据到L2缓存
        for (size_t i = begin; i < end; i++) {
            storage_idx_t nid = hnsw.neighbors[i];
            if (nid < 0) break;

            // 预取向量数据
            const float* vec = get_vector_data(nid);
            _mm_prefetch((const char*)vec, _MM_HINT_T1);

            // 预取邻居列表
            _mm_prefetch((const char*)(hnsw.neighbors.data() + nid),
                         _MM_HINT_T1);
        }
    }

    // 软件预取与计算重叠
    void process_with_overlap(ResultHandler<HNSW::C>& res) {
        while (queue_head != queue_tail) {
            PrefetchEntry entry = prefetch_queue[queue_head];
            queue_head = (queue_head + 1) % PREFETCH_DISTANCE;

            // 处理当前节点
            if (HNSW::C::cmp(entry.distance, res.heap_dis[0])) {
                res.add_result(entry.distance, entry.node_id);
            }

            // 异步预取下一批邻居
            if ((queue_tail + 1) % PREFETCH_DISTANCE != queue_head) {
                prefetch_neighbors(entry.node_id, 0);
            }
        }
    }
};

// 使用prefetch优化的搜索
HNSWStats search_with_prefetch(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler<HNSW::C>& res,
        VisitedTable& vt,
        int level) {

    HNSWStats stats;
    PrefetchingHNSWIterator<8> iterator{hnsw, qdis};

    storage_idx_t nearest = hnsw.entry_point;
    float d_nearest = qdis(nearest);

    // 预取入口点的邻居
    iterator.prefetch_neighbors(nearest, level);

    while (true) {
        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);

        bool improved = false;

        // 批量处理邻居
        for (size_t i = begin; i < end; i++) {
            storage_idx_t nid = hnsw.neighbors[i];
            if (nid < 0) break;

            stats.ndis++;

            if (!vt.get(nid)) {
                vt.set(nid);
                float dis = qdis(nid);

                if (dis < d_nearest) {
                    d_nearest = dis;
                    nearest = nid;
                    improved = true;

                    // 提前预取新最近点的邻居
                    iterator.prefetch_neighbors(nearest, level);
                }
            }
        }

        stats.nhops++;

        if (!improved) {
            break;
        }
    }

    return stats;
}
```

### 12.4 内存池优化的邻居分配

```cpp
// 对象池模式：避免频繁的小对象分配
struct HNSWNeighborPool {
    struct Block {
        alignas(64) storage_idx_t neighbors[BLOCK_SIZE];
        Block* next;

        Block() : next(nullptr) {
            // 初始化为无效ID
            std::fill(neighbors, neighbors + BLOCK_SIZE, -1);
        }
    };

    std::vector<Block*> blocks;
    Block* free_list = nullptr;
    std::mutex pool_mutex;

    static constexpr size_t BLOCK_SIZE = 64;  // 缓存行大小

    HNSWNeighborPool(size_t initial_blocks = 1024) {
        for (size_t i = 0; i < initial_blocks; i++) {
            Block* block = new Block();
            blocks.push_back(block);
            block->next = free_list;
            free_list = block;
        }
    }

    // 快速分配（线程安全）
    storage_idx_t* allocate() {
        Block* block;

        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            if (!free_list) {
                // 扩容
                Block* new_block = new Block();
                blocks.push_back(new_block);
                new_block->next = free_list;
                free_list = new_block;
            }

            block = free_list;
            free_list = free_list->next;
        }

        return block->neighbors;
    }

    // 快速释放
    void deallocate(storage_idx_t* ptr) {
        Block* block = reinterpret_cast<Block*>(
                reinterpret_cast<char*>(ptr) - offsetof(Block, neighbors));

        std::lock_guard<std::mutex> lock(pool_mutex);
        block->next = free_list;
        free_list = block;
    }

    ~HNSWNeighborPool() {
        for (Block* block : blocks) {
            delete block;
        }
    }
};
```

---

## 13. HNSW并发优化

### 13.1 无锁邻居更新

```cpp
// 使用CAS操作实现无锁邻居更新
#include <atomic>

struct LockFreeHNSW {
    struct NeighborArray {
        std::atomic<storage_idx_t> neighbors[MAX_NEIGHBORS];

        void init() {
            for (int i = 0; i < MAX_NEIGHBORS; i++) {
                neighbors[i].store(-1, std::memory_order_relaxed);
            }
        }

        // CAS添加邻居
        bool try_add(storage_idx_t new_neighbor, int max_size) {
            // 首先尝试找空位
            for (int i = 0; i < max_size; i++) {
                storage_idx_t expected = -1;

                if (neighbors[i].compare_exchange_strong(
                        expected, new_neighbor,
                        std::memory_order_release,
                        std::memory_order_relaxed)) {
                    return true;  // 成功添加
                }
            }

            // 没有空位，尝试替换最远的邻居
            // （需要额外维护距离信息）
            return false;
        }
    };

    std::vector<NeighborArray> layer_neighbors;

    // 无锁的邻居添加
    bool add_neighbor_lockfree(
            storage_idx_t node_id,
            storage_idx_t new_neighbor,
            int level,
            DistanceComputer& dis) {

        NeighborArray& arr = layer_neighbors[node_id];
        int max_size = nb_neighbors(level);

        // 尝试添加到空位
        for (int i = 0; i < max_size; i++) {
            storage_idx_t expected = -1;

            if (arr.neighbors[i].compare_exchange_strong(
                    expected, new_neighbor,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }

        // 空位已满，需要竞争替换
        // 找到最远的邻居并尝试替换
        float max_dis = -1;
        int max_idx = -1;

        for (int i = 0; i < max_size; i++) {
            storage_idx_t nid = arr.neighbors[i].load(
                    std::memory_order_relaxed);
            float d = dis.symmetric_dis(node_id, nid);

            if (d > max_dis) {
                max_dis = d;
                max_idx = i;
            }
        }

        // 尝试替换
        storage_idx_t old_neighbor = arr.neighbors[max_idx].load(
                std::memory_order_relaxed);
        float new_dis = dis.symmetric_dis(node_id, new_neighbor);

        if (new_dis < max_dis) {
            // CAS替换
            if (arr.neighbors[max_idx].compare_exchange_strong(
                    old_neighbor, new_neighbor,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }

        return false;
    }
};
```

### 13.2 读写锁优化的并发搜索

```cpp
// 使用读写锁允许多读者并发
#include <shared_mutex>

struct RWLockHNSW {
    mutable std::shared_mutex graph_mutex;

    // 读操作（搜索）
    HNSWStats search_concurrent(
            DistanceComputer& qdis,
            ResultHandler<HNSW::C>& res,
            VisitedTable& vt) const {

        // 共享锁：多个读者可以并发
        std::shared_lock<std::shared_mutex> lock(graph_mutex);

        return search_impl(qdis, res, vt);
    }

    // 写操作（添加节点）
    void add_concurrent(
            DistanceComputer& ptdis,
            int pt_level,
            storage_idx_t pt_id) {

        // 独占锁：写者独占
        std::unique_lock<std::shared_mutex> lock(graph_mutex);

        add_impl(ptdis, pt_level, pt_id);
    }

    // 细粒度锁：每个节点一把锁
    std::vector<std::shared_mutex> node_locks;

    void add_with_fine_grained_locks(
            DistanceComputer& ptdis,
            int pt_level,
            storage_idx_t pt_id) {

        // 只锁定涉及到的节点
        for (int level = pt_level; level >= 0; level--) {
            // 找到候选邻居
            std::vector<storage_idx_t> neighbors =
                    find_candidates(ptdis, level);

            // 按ID排序避免死锁
            std::sort(neighbors.begin(), neighbors.end());

            // 逐个加锁并添加连接
            for (storage_idx_t neighbor_id : neighbors) {
                std::unique_lock<std::shared_mutex> lock(
                        node_locks[neighbor_id]);

                // 添加pt_id到neighbor_id的邻居列表
                add_link_one_way(neighbor_id, pt_id, level);
            }
        }
    }
};
```

### 13.3 批量添加优化

```cpp
// 批量添加节点，减少锁竞争
struct BatchHNSWBuilder {
    HNSW& hnsw;
    size_t batch_size;

    struct BatchEntry {
        storage_idx_t id;
        int level;
        std::vector<float> vector;
    };

    std::vector<BatchEntry> batch;

    void add_to_batch(
            storage_idx_t id,
            int level,
            const float* vec,
            int d) {

        batch.push_back({id, level, std::vector<float>(vec, vec + d)});

        if (batch.size() >= batch_size) {
            flush_batch();
        }
    }

    void flush_batch() {
        if (batch.empty()) return;

        // 1. 添加所有节点到图结构
        {
            std::unique_lock<std::shared_mutex> lock(hnsw.graph_mutex);

            for (const auto& entry : batch) {
                hnsw.allocate_node(entry.id, entry.level);
            }
        }

        // 2. 并行构建连接
#pragma omp parallel
        {
            VisitedTable vt(hnsw.ntotal);

#pragma omp for schedule(dynamic)
            for (size_t i = 0; i < batch.size(); i++) {
                const auto& entry = batch[i];

                // 创建临时distance computer
                DistanceComputer* dis =
                        hnsw.storage->get_distance_computer();
                dis->set_query(entry.vector.data());

                // 添加连接（使用细粒度锁）
                hnsw.add_links_with_locks(
                        *dis, entry.level, entry.id,
                        hnsw.node_locks.data(), vt);

                delete dis;
            }
        }

        batch.clear();
    }
};
```

### 13.4 乐观并发控制（OCC）

```cpp
// 乐观并发控制：先尝试不加锁，冲突时重试
struct OCCHNSW {
    struct VersionedNeighbor {
        storage_idx_t neighbor_id;
        uint64_t version;  // 版本号

        VersionedNeighbor()
            : neighbor_id(-1), version(0) {}
    };

    std::vector<std::vector<VersionedNeighbor>> neighbors;
    std::vector<std::atomic<uint64_t>> node_versions;

    bool try_add_neighbor_optimistic(
            storage_idx_t node_id,
            storage_idx_t new_neighbor,
            int level) {

        auto& level_neighbors = neighbors[node_id];

        while (true) {
            // 读取当前版本
            uint64_t start_version = node_versions[node_id].load(
                    std::memory_order_acquire);

            // 尝试添加邻居（不加锁）
            bool added = try_add_to_array(
                    level_neighbors, new_neighbor, level);

            if (!added) {
                return false;  // 数组已满
            }

            // 验证版本未改变
            uint64_t end_version = node_versions[node_id].load(
                    std::memory_order_acquire);

            if (start_version == end_version) {
                // 成功：版本未变，更新版本号
                node_versions[node_id].fetch_add(1,
                        std::memory_order_release);
                return true;
            }

            // 冲突：版本已变，回滚并重试
            rollback_add(level_neighbors, new_neighbor, level);
            // 继续循环重试
        }
    }

    bool try_add_to_array(
            std::vector<VersionedNeighbor>& arr,
            storage_idx_t new_neighbor,
            int level) {

        // 查找空位或重复
        for (size_t i = 0; i < arr.size(); i++) {
            if (arr[i].neighbor_id == -1) {
                arr[i].neighbor_id = new_neighbor;
                return true;
            }
            if (arr[i].neighbor_id == new_neighbor) {
                return true;  // 已存在
            }
        }

        return false;  // 数组已满
    }

    void rollback_add(
            std::vector<VersionedNeighbor>& arr,
            storage_idx_t neighbor,
            int level) {

        // 移除之前添加的邻居
        for (size_t i = 0; i < arr.size(); i++) {
            if (arr[i].neighbor_id == neighbor) {
                arr[i].neighbor_id = -1;
                break;
            }
        }
    }
};
```

---

## 14. HNSW性能分析与调优

### 14.1 性能计数器

```cpp
// 详细的HNSW性能分析
struct HNSWPerformanceCounters {
    // 距离计算
    std::atomic<uint64_t> distance_computations{0};
    std::atomic<uint64_t> distance_simd_batches{0};

    // 图遍历
    std::atomic<uint64_t> hops{0};
    std::atomic<uint64_t> neighbors_visited{0};
    std::atomic<uint64_t> prefetches_issued{0};

    // 并发
    std::atomic<uint64_t> lock_contentions{0};
    std::atomic<uint64_t> lock_retries{0};
    std::atomic<uint64_t> cas_failures{0};

    // 内存
    std::atomic<uint64_t> cache_misses{0};  // 需要PMU支持
    std::atomic<uint64_t> allocations{0};

    void print_report() const {
        printf("=== HNSW Performance Report ===\n");
        printf("Distance computations: %lu\n", distance_computations.load());
        printf("SIMD batches: %lu (%.1f%% vectorized)\n",
               distance_simd_batches.load(),
               100.0 * distance_simd_batches.load() /
                   std::max<uint64_t>(1, distance_computations.load()));
        printf("Hops: %lu\n", hops.load());
        printf("Neighbors visited per hop: %.2f\n",
               (double)neighbors_visited.load() /
                   std::max<uint64_t>(1, hops.load()));
        printf("Lock contentions: %lu\n", lock_contentions.load());
        printf("CAS failures: %lu\n", cas_failures.load());
    }
};

// 在关键路径上插入计数
void instrumented_search(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler<HNSW::C>& res,
        HNSWPerformanceCounters& counters) {

    storage_idx_t nearest = hnsw.entry_point;
    float d_nearest = qdis(nearest);
    counters.distance_computations++;

    for (int level = hnsw.max_level; level >= 0; level--) {
        while (true) {
            counters.hops++;

            size_t begin, end;
            hnsw.neighbor_range(nearest, level, &begin, &end);

            for (size_t i = begin; i < end; i++) {
                storage_idx_t nid = hnsw.neighbors[i];
                counters.neighbors_visited++;

                float dis = qdis(nid);
                counters.distance_computations++;

                if (dis < d_nearest) {
                    d_nearest = dis;
                    nearest = nid;
                }
            }

            break;
        }
    }
}
```

### 14.2 缓存性能分析

```cpp
#ifdef __linux__
#include <perfmon/pfmlib.h>
#include <perfmon/perf_event.h>

struct CacheProfiler {
    int fd_cache_miss;
    int fd_cache_ref;

    void init() {
        // 配置PMU计数器
        struct perf_event_attr attr;

        memset(&attr, 0, sizeof(attr));
        attr.type = PERF_TYPE_HARDWARE;
        attr.config = PERF_COUNT_HW_CACHE_MISSES;
        attr.size = sizeof(attr);

        fd_cache_miss = syscall(__NR_perf_event_open, &attr,
                                -1, 0, -1, 0);

        attr.config = PERF_COUNT_HW_CACHE_REFERENCES;
        fd_cache_ref = syscall(__NR_perf_event_open, &attr,
                              -1, 0, -1, 0);

        // 启用计数器
        ioctl(fd_cache_miss, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd_cache_ref, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd_cache_miss, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(fd_cache_ref, PERF_EVENT_IOC_ENABLE, 0);
    }

    double get_miss_rate() {
        long long misses, refs;

        read(fd_cache_miss, &misses, sizeof(misses));
        read(fd_cache_ref, &refs, sizeof(refs));

        return (double)misses / std::max(1LL, refs);
    }

    ~CacheProfiler() {
        close(fd_cache_miss);
        close(fd_cache_ref);
    }
};

// 使用分析器
void profile_hnsw_cache_behavior() {
    CacheProfiler profiler;
    profiler.init();

    HNSW hnsw;
    // ... 构建hnsw ...

    DistanceComputer* qdis = ...;
    ResultHandler<> res = ...;

    hnsw.search(*qdis, &res, ...);

    double miss_rate = profiler.get_miss_rate();
    printf("Cache miss rate: %.2f%%\n", miss_rate * 100);
}
#endif
```

### 14.3 自动参数调优

```cpp
// HNSW参数自动调优
struct HNSWTuner {
    struct ParameterConfig {
        int M;
        int efConstruction;
        int efSearch;

        float recall;
        double qps;  // queries per second
        double memory_mb;
    };

    std::vector<ParameterConfig> results;

    void grid_search(
            const float* train_vectors,
            idx_t ntrain,
            const float* query_vectors,
            idx_t nquery,
            const idx_t* ground_truth,
            int d) {

        // M的搜索范围
        std::vector<int> M_values = {16, 24, 32, 48, 64};

        // efConstruction的搜索范围
        std::vector<int> efC_values = {20, 40, 80, 120};

        // efSearch的搜索范围
        std::vector<int> efS_values = {10, 20, 40, 80, 160};

        for (int M : M_values) {
            for (int efC : efC_values) {
                // 构建索引
                IndexHNSWFlat index(d, M);
                index.hnsw.efConstruction = efC;
                index.add(ntrain, train_vectors);

                for (int efS : efS_values) {
                    index.hnsw.efSearch = efS;

                    // 测量性能
                    auto t0 = std::chrono::high_resolution_clock::now();

                    std::vector<float> distances(nquery * k);
                    std::vector<idx_t> labels(nquery * k);
                    index.search(nquery, query_vectors, k,
                                distances.data(), labels.data());

                    auto t1 = std::chrono::high_resolution_clock::now();
                    double time_ms =
                        std::chrono::duration<double>(t1 - t0).count() * 1000;
                    double qps = nquery * 1000.0 / time_ms;

                    // 计算recall
                    float recall = compute_recall(
                            nquery, k, labels.data(), ground_truth);

                    // 记录结果
                    results.push_back({M, efC, efS, recall, qps, 0.0});
                }
            }
        }
    }

    // 找到最优配置
    ParameterConfig find_optimal(float target_recall) {
        auto best = std::min_element(
                results.begin(), results.end(),
                [target_recall](const ParameterConfig& a,
                                const ParameterConfig& b) {
                    // 只考虑满足recall要求的
                    if (a.recall < target_recall) return false;
                    if (b.recall < target_recall) return true;

                    // 最大化QPS
                    return a.qps > b.qps;
                });

        return *best;
    }
};
```

### 14.4 热点分析工具

```cpp
// 基于perf的HNSW热点分析
#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>

struct HNSWProfiler {
    static constexpr int PROFILE_DURATION_SEC = 30;

    static void profile_build(const char* output_file) {
        pid_t pid = getpid();

        // 启动perf record
        char cmd[256];
        snprintf(cmd, sizeof(cmd),
                "perf record -g -p %d -o %s sleep %d",
                pid, output_file, PROFILE_DURATION_SEC);

        int child_pid = fork();
        if (child_pid == 0) {
            // 子进程：运行perf
            system(cmd);
            exit(0);
        }

        // 父进程：继续执行HNSW构建
        // ... 构建代码 ...

        // 等待perf完成
        waitpid(child_pid, nullptr, 0);

        printf("Profile data saved to %s\n", output_file);
        printf("View with: perf report -i %s\n", output_file);
    }

    static void profile_searches(const char* output_file, int nqueries) {
        pid_t pid = getpid();

        // 使用perf stat统计
        char cmd[256];
        snprintf(cmd, sizeof(cmd),
                "perf stat -p %d -e cycles,instructions,cache-misses,"
                "cache-references,L1-dcache-load-misses,LLC-load-misses "
                "-o %s -- sleep %d",
                pid, output_file, PROFILE_DURATION_SEC);

        system(cmd);

        printf("Perf statistics saved to %s\n", output_file);
    }
};

// 使用示例
void profile_hnsw_build() {
    HNSWProfiler::profile_build("hnsw_build.data");

    IndexHNSWFlat index(128, 32);
    index.hnsw.efConstruction = 40;
    // ... 执行构建 ...
}
#endif
```

---

## 15. HNSW底层SIMD与并发优化深入

### 15.1 SIMD优化的批量距离计算

```cpp
// AVX2优化的批量邻居距离计算
// 一次计算4个邻居的距离
void compute_neighbor_distances_avx2(
        const float* query,
        const float* vectors,
        const idx_t* neighbor_ids,
        size_t M,
        size_t d,
        float* distances) {

    size_t m = 0;

    // 4路展开：每次处理4个邻居
    for (; m + 4 <= M; m += 4) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        for (size_t dim = 0; dim < d; dim++) {
            __m256 qdim = _mm256_set1_ps(query[dim]);

            // 加载4个向量的第dim维
            __m256 v0dim = _mm256_set1_ps(
                vectors[neighbor_ids[m + 0] * d + dim]);
            __m256 v1dim = _mm256_set1_ps(
                vectors[neighbor_ids[m + 1] * d + dim]);
            __m256 v2dim = _mm256_set1_ps(
                vectors[neighbor_ids[m + 2] * d + dim]);
            __m256 v3dim = _mm256_set1_ps(
                vectors[neighbor_ids[m + 3] * d + dim]);

            __m256 diff0 = _mm256_sub_ps(qdim, v0dim);
            __m256 diff1 = _mm256_sub_ps(qdim, v1dim);
            __m256 diff2 = _mm256_sub_ps(qdim, v2dim);
            __m256 diff3 = _mm256_sub_ps(qdim, v3dim);

            sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
            sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
            sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
            sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
        }

        // 水平求和
        distances[m + 0] = hsum256_ps(sum0);
        distances[m + 1] = hsum256_ps(sum1);
        distances[m + 2] = hsum256_ps(sum2);
        distances[m + 3] = hsum256_ps(sum3);
    }

    // 处理剩余邻居
    for (; m < M; m++) {
        const float* vec = vectors + neighbor_ids[m] * d;
        distances[m] = fvec_L2sqr(query, vec, d);
    }
}

// AVX-512版本：16路并行
#ifdef __AVX512F__
void compute_neighbor_distances_avx512(
        const float* query,
        const float* vectors,
        const idx_t* neighbor_ids,
        size_t M,
        size_t d,
        float* distances) {

    size_t m = 0;

    for (; m + 16 <= M; m += 16) {
        __m512 sum[4];
        sum[0] = _mm512_setzero_ps();
        sum[1] = _mm512_setzero_ps();
        sum[2] = _mm512_setzero_ps();
        sum[3] = _mm512_setzero_ps();

        for (size_t dim = 0; dim < d; dim++) {
            __m512 qdim = _mm512_set1_ps(query[dim]);

            // 加载16个向量的第dim维（使用gather）
            __m512i indices = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(neighbor_ids + m));

            // 需要重新计算偏移量
            // 实际实现中应该预先准备连续的向量布局
            for (int i = 0; i < 4; i++) {
                __m512 vdim = _mm512_loadu_ps(
                    vectors + (m + i * 4) * d + dim);
                __m512 diff = _mm512_sub_ps(qdim, vdim);
                sum[i] = _mm512_fmadd_ps(diff, diff, sum[i]);
            }
        }

        // 水平求和
        for (int i = 0; i < 4; i++) {
            distances[m + i] = _mm512_reduce_add_ps(sum[i]);
        }
    }

    // 处理剩余...
}
#endif
```

### 15.2 缓存优化的图遍历

```cpp
// 缓存友好的邻居访问模式
class CacheOptimizedHNSW {
    struct alignas(64) NeighborBlock {
        idx_t ids[16];          // 16个邻居ID
        float distances[16];    // 预计算的距离
        uint8_t valid_mask[2];  // 16位有效位
    };

    std::vector<std::vector<NeighborBlock>> blocked_neighbors;

    // 将邻居组织到缓存行对齐的块中
    void organize_neighbors_by_cache_line(
            const HNSW& hnsw,
            size_t max_neighbors_per_node) {

        size_t n = hnsw.ntotal;
        blocked_neighbors.resize(n);

        for (size_t node_id = 0; node_id < n; node_id++) {
            int max_level = hnsw.levels[node_id] - 1;
            size_t offset = 0;

            for (int level = 0; level <= max_level; level++) {
                size_t begin, end;
                hnsw.neighbor_range(node_id, level, &begin, &end);
                size_t n_neighbors = end - begin;

                // 分配块
                size_t n_blocks = (n_neighbors + 15) / 16;
                blocked_neighbors[node_id].resize(
                    offset + n_blocks * sizeof(NeighborBlock) / sizeof(uint8_t));

                for (size_t b = 0; b < n_blocks; b++) {
                    NeighborBlock& block = reinterpret_cast<NeighborBlock*>(
                        blocked_neighbors[node_id].data())[offset + b];

                    size_t block_start = b * 16;
                    size_t block_end = std::min(block_start + 16, n_neighbors);

                    // 填充块
                    for (size_t i = 0; i < 16; i++) {
                        if (i < block_end - block_start) {
                            block.ids[i] = hnsw.neighbors[begin + block_start + i];
                            block.valid_mask[i / 8] |= (1 << (i % 8));
                        } else {
                            block.ids[i] = -1;
                            block.valid_mask[i / 8] = 0;
                        }
                    }
                }

                offset += n_blocks;
            }
        }
    }

    // 缓存友好的邻居遍历
    void traverse_cache_optimized(
            const CacheOptimizedHNSW& graph,
            idx_t start_node,
            const float* query,
            const DistanceComputer& dis_comp) {

        // 使用预取优化
        constexpr size_t PREFETCH_AHEAD = 2;

        for (const NeighborBlock& block : graph.blocked_neighbors[start_node]) {
            // 预取下一个块
            // ...
        }
    }
};
```

### 15.3 内存池优化

```cpp
// 专用的HNSW内存池
class HNSWMemoryPool {
    struct Pool {
        void* base;
        size_t capacity;
        size_t used;
        std::mutex mutex;

        Pool(size_t cap) : capacity(cap), used(0) {
            // 使用huge pages减少TLB miss
            #ifdef __linux__
            posix_memalign(&base, 2 * 1024 * 1024, capacity);
            madvise(base, capacity, MADV_HUGEPAGE);
            #else
            base = aligned_alloc(64, capacity);
            #endif
        }

        ~Pool() {
            #ifdef __linux__
                free(base);
            #else
                free(base);
            #endif
        }

        template<typename T>
        T* allocate(size_t n, size_t alignment = 64) {
            std::lock_guard<std::mutex> lock(mutex);

            size_t size = n * sizeof(T);
            size_t aligned_used = (used + alignment - 1) / alignment * alignment;

            if (aligned_used + size > capacity) {
                return nullptr;  // 池已满
            }

            void* ptr = static_cast<char*>(base) + aligned_used;
            used = aligned_used + size;

            return reinterpret_cast<T*>(ptr);
        }
    };

    std::vector<std::unique_ptr<Pool>> pools;
    std::atomic<size_t> next_pool{0};

public:
    HNSWMemoryPool(size_t pool_size = 16 * 1024 * 1024) {
        // 为每个NUMA节点创建一个池
        int num_pools = 1;
        #ifdef __linux__
        num_pools = numa_num_configured_nodes();
        #endif

        for (int i = 0; i < num_pools; i++) {
            pools.push_back(std::make_unique<Pool>(pool_size));
        }
    }

    // 分配邻居数组
    idx_t* allocate_neighbor_array(size_t n) {
        constexpr size_t alignment = 64;
        size_t pool_id = next_pool.fetch_add(1) % pools.size();

        idx_t* ptr = pools[pool_id]->allocate<idx_t>(n, alignment);

        if (!ptr) {
            // 回退到malloc
            posix_memalign((void**)&ptr, alignment, n * sizeof(idx_t));
        }

        return ptr;
    }

    void deallocate_neighbor_array(idx_t* ptr, size_t n) {
        // 简化实现：直接释放
        free(ptr);
    }
};
```

### 15.4 无锁并发HNSW

```cpp
// 无锁的HNSW实现（使用原子操作）
class LockFreeHNSW {
    struct Node {
        std::atomic<storage_idx_t> neighbors[64];  // 最多64个邻居
        std::atomic<int> n_neighbors;
        int level;
        // ...
    };

    std::vector<Node> nodes;
    std::atomic<storage_idx_t> entry_point{-1};

    // 无锁添加连接
    bool add_connection_lock_free(
            storage_idx_t src,
            storage_idx_t dst,
            int level) {

        Node& src_node = nodes[src];
        int expected_n = src_node.n_neighbors.load(std::memory_order_relaxed);
        int max_n = get_max_neighbors(level);

        while (expected_n < max_n) {
            // CAS添加邻居
            bool success = src_node.n_neighbors.compare_exchange_weak(
                expected_n,
                expected_n + 1,
                std::memory_order_release,
                std::memory_order_relaxed);

            if (success) {
                // 成功获取槽位
                int slot = expected_n;
                src_node.neighbors[slot].store(dst, std::memory_order_release);
                return true;
            }
        }

        // 没有空位，需要替换最远邻居
        // 这里需要更复杂的逻辑...
        return false;
    }

    // 无锁搜索
    void search_lock_free(
            const float* query,
            DistanceComputer& dis_comp,
            size_t k,
            float* distances,
            idx_t* labels) {

        storage_idx_t ep = entry_point.load(std::memory_order_acquire);

        // 自顶向下贪婪搜索
        for (int level = max_level; level >= 0; level--) {
            storage_idx_t nearest = ep;
            float d_nearest = dis_comp(nearest);

            // 无锁贪婪更新
            while (true) {
                bool changed = false;

                Node& node = nodes[nearest];

                int n_neighbors = node.n_neighbors.load(std::memory_order_acquire);
                for (int i = 0; i < n_neighbors; i++) {
                    storage_idx_t neighbor =
                        node.neighbors[i].load(std::memory_order_acquire);

                    float dis = dis_comp(neighbor);
                    if (dis < d_nearest) {
                        d_nearest = dis;
                        nearest = neighbor;
                        changed = true;
                    }
                }

                if (!changed) break;
            }

            if (level > 0) {
                ep = nearest;
            }
        }

        // 在第0层搜索...
    }
};
```

### 15.5 SIMD优化的VisitedTable

```cpp
// SIMD优化的Visited表：使用位向量加速查找
class SIMDOptimizedVisitedTable {
    static constexpr size_t BLOCK_SIZE = 256;  // 256个节点一个块
    static constexpr size_t BITS_PER_BLOCK = BLOCK_SIZE * 8;

    std::vector<uint64_t> visited_blocks;  // 每个块256位 = 32个uint64
    size_t max_elements;

public:
    SIMDOptimizedVisitedTable(size_t n)
        : max_elements(n) {
        size_t n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        visited_blocks.assign(n_blocks, 0);
    }

    // AVX2优化的批量检查
    inline bool check_batch(const idx_t* ids, size_t n) const {
        size_t checked = 0;

        for (size_t i = 0; i < n; i += 8) {
            size_t block_ids[8];
            uint64_t block_masks[8];

            // 收集块的ID和mask
            for (size_t j = 0; j < 8 && i + j < n; j++) {
                size_t id = ids[i + j];
                size_t block = id / BLOCK_SIZE;
                size_t bit = id % BLOCK_SIZE;

                block_ids[j] = block;
                block_masks[j] = 1ULL << bit;
            }

            // 加载并检查8个块
            __m512i blocks = _mm512_loadu_si512(block_ids);
            __m512i masks = _mm512_loadu_si512(block_masks);

            // OR masks
            __m512i combined = _mm512_or_si512(masks, _mm512_setzero_si512());

            size_t block = _mm512_cvtsi512(_mm512_extract_epi32(blocks, 0));

            // 检查是否已访问
            if ((visited_blocks[block] & combined) != combined) {
                // 有未访问的节点
                checked += 8;
            }
        }

        return checked < n;
    }

    // 批量设置
    inline void set_batch(const idx_t* ids, size_t n) {
        for (size_t i = 0; i < n; i++) {
            idx_t id = ids[i];
            size_t block = id / BLOCK_SIZE;
            size_t bit = id % BLOCK_SIZE;

            visited_blocks[block] |= (1ULL << bit);
        }
    }

    // SIMD-优化的清空
    void clear() {
        visited_blocks.assign(visited_blocks.size(), 0);
    }
};
```

### 15.6 预取优化的图遍历

```cpp
// 预取优化的HNSW搜索
void hnsw_search_with_prefetch(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        int level) {

    constexpr size_t PREFETCH_DISTANCE = 4;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        // 获取邻居
        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);
        size_t n_neighbors = end - begin;

        // 预取未来的邻居节点数据
        for (size_t i = 0; i < n_neighbors; i += PREFETCH_DISTANCE) {
            if (i + PREFETCH_DISTANCE < n_neighbors) {
                idx_t prefetch_id = hnsw.neighbors[begin + i + PREFETCH_DISTANCE];

                // 预取向量数据
                const float* vec = get_vector(prefetch_id);
                _mm_prefetch((const char*)(vec + 0), _MM_HINT_T0);
                _mm_prefetch((const char*)(vec + 64), _MM_HINT_T0);

                // 预取邻居表
                const idx_t* neighbors = get_neighbors(prefetch_id, level);
                _mm_prefetch((const char*)(neighbors + 0), _MM_HINT_T0);
                _mm_prefetch((const char*)(neighbors + 64), _MM_HINT_T0);
            }
        }

        // 处理当前节点
        for (size_t i = begin; i < end; i++) {
            idx_t neighbor_id = hnsw.neighbors[i];

            if (vt.get(neighbor_id)) continue;
            vt.set(neighbor_id);

            float dis = qdis(neighbor_id);
            candidates.push(neighbor_id, dis);
        }

        // 检查停止条件
        if (d0 > candidates.max()) {
            break;
        }
    }
}

// 硬件预取计数器优化
class PrefetchCounter {
    std::array<uint64_t, 256> counters;
    size_t threshold;

public:
    PrefetchCounter() : threshold(16) {}

    bool should_prefetch(size_t node_id) {
        return counters[node_id % 256]++ < threshold;
    }

    void reset() {
        counters.fill(0);
    }
};
```

### 15.7 NUMA感知的HNSW构建

```cpp
// NUMA感知的HNSW构建
class NUMAAwareHNSW {
    struct NUMANodeData {
        std::vector<idx_t> local_nodes;      // 本NUMA节点的向量
        std::unique_ptr<HNSWMemoryPool> pool;
        int numa_node;
    };

    std::vector<NUMANodeData> numa_nodes;

public:
    NUMAAwareHNSW() {
        #ifdef __linux__
        int max_node = numa_max_node() + 1;
        #else
        int max_node = 1;
        #endif

        for (int i = 0; i < max_node; i++) {
            NUMANodeData data;
            data.numa_node = i;
            data.pool = std::make_unique<HNSWMemoryPool>();
            numa_nodes.push_back(std::move(data));
        }
    }

    // 添加向量到NUMA感知的HNSW
    idx_t add_with_numa_awareness(
            const float* vector,
            size_t d,
            DistanceComputer& dis_comp) {

        // 决定应该添加到哪个NUMA节点
        int target_node = 0;
        #ifdef __linux__
        target_node = numa_preferred();
        #endif

        NUMANodeData& node_data = numa_nodes[target_node];

        // 在本地内存池中分配
        idx_t new_id = node_data.local_nodes.size();
        float* local_vec = node_data.pool->allocate<float>(d);
        memcpy(local_vec, vector, d * sizeof(float));

        node_data.local_nodes.push_back(new_id);

        // 绑定当前线程到NUMA节点
        #ifdef __linux__
        numa_run_on_node(target_node);
        numa_set_preferred(target_node);
        #endif

        // 执行HNSW添加逻辑
        // ...

        return new_id;
    }

    // NUMA感知的搜索
    void search_numa_aware(
            const float* query,
            DistanceComputer& dis_comp,
            size_t k,
            float* distances,
            idx_t* labels) {

        // 收集所有NUMA节点的候选
        std::vector<std::pair<float, idx_t>> all_candidates;

        #pragma omp parallel
        {
            std::vector<std::pair<float, idx_t>> local_candidates;

            #pragma omp for
            for (int node_id = 0; node_id < numa_nodes.size(); node_id++) {
                // 绑定到NUMA节点
                #ifdef __linux__
                numa_run_on_node(node_id);
                numa_set_preferred(node_id);
                #endif

                // 在本地节点上搜索
                // 搜索结果存入local_candidates
            }

            #pragma omp critical
            {
                all_candidates.insert(all_candidates.end(),
                                     local_candidates.begin(),
                                     local_candidates.end());
            }
        }

        // 合并并排序
        std::partial_sort(
            all_candidates.begin(),
            all_candidates.begin() + std::min(k, all_candidates.size()),
            all_candidates.end());

        // 返回top-k结果
        for (size_t i = 0; i < k && i < all_candidates.size(); i++) {
            distances[i] = all_candidates[i].first;
            labels[i] = all_candidates[i].second;
        }
    }
};
```

### 15.8 延迟优化构建

```cpp
// 延迟构建：先添加向量，稍后构建图结构
class LazyHNSW {
    struct PendingNode {
        std::vector<float> vector;
        idx_t id;
        int assigned_level;
    };

    std::vector<PendingNode> pending_nodes;
    bool is_built;

public:
    LazyHNSW() : is_built(false) {}

    // 添加阶段：只存储向量
    idx_t add_deferred(const float* x, size_t d) {
        idx_t id = pending_nodes.size();
        pending_nodes.push_back({std::vector<float>(x, x + d), id, 0});
        return id;
    }

    // 构建阶段：批量构建HNSW
    void build(int M, float levelMult) {
        // 1. 为每个节点分配层级
        #pragma omp parallel for
        for (size_t i = 0; i < pending_nodes.size(); i++) {
            pending_nodes[i].assigned_level = random_level(M, levelMult);
        }

        // 2. 按层级分组
        std::vector<std::vector<size_t>> levels;
        for (size_t i = 0; i < pending_nodes.size(); i++) {
            levels[pending_nodes[i].assigned_level].push_back(i);
        }

        // 3. 从高层到低层构建连接
        for (int level = max_level; level >= 0; level--) {
            // 并行构建该层的连接
            #pragma omp parallel for schedule(dynamic)
            for (size_t idx = 0; idx < levels[level].size(); idx++) {
                size_t i = levels[level][idx];
                build_connections_for_node(i, level, M);
            }
        }

        is_built = true;
    }

    void build_connections_for_node(size_t node_id, int level, int M) {
        PendingNode& node = pending_nodes[node_id];

        // 搜索候选邻居
        std::priority_queue<NodeDistCloser> candidates;
        std::priority_queue<NodeDistFarther> selected;

        // 从高层继承邻居
        if (level < max_level) {
            for (idx_t neighbor_id : get_neighbors(node_id, level + 1)) {
                candidates.emplace(distance(node_id, neighbor_id), neighbor_id);
                selected.emplace(distance(node_id, neighbor_id), neighbor_id);
            }
        }

        // 在当前层搜索更多邻居
        // ... 使用标准HNSW连接逻辑

        // 限制到M个邻居
        while (selected.size() > M) {
            selected.pop();
        }
    }

    // 搜索（检查是否已构建）
    void search(const float* query, size_t k, float* distances, idx_t* labels) {
        if (!is_built) {
            throw std::runtime_error("HNSW not built yet");
        }
        // ... 标准HNSW搜索逻辑
    }
};
```

---

## 16. SIMD优化的邻居遍历

### 16.1 批量距离计算

```cpp
// SIMD优化的批量邻居距离计算
#ifdef __AVX2__
void batch_distance_computation_avx2(
        const float* query,
        const idx_t* neighbor_ids,
        size_t n_neighbors,
        const float* vectors,
        size_t d,
        float* distances_out) {

    // 一次处理8个邻居（假设d是8的倍数）
    size_t i = 0;
    for (; i + 8 <= n_neighbors; i += 8) {
        __m256 sum = _mm256_setzero_ps();

        // 计算查询向量与8个邻居的距离
        for (size_t j = 0; j < d; j += 8) {
            // 加载查询向量
            __m256 q = _mm256_loadu_ps(query + j);

            // 加载8个邻居的第j维
            __m256 n0 = _mm256_set1_ps(vectors[neighbor_ids[i + 0] * d + j]);
            __m256 n1 = _mm256_set1_ps(vectors[neighbor_ids[i + 1] * d + j]);
            __m256 n2 = _mm256_set1_ps(vectors[neighbor_ids[i + 2] * d + j]);
            __m256 n3 = _mm256_set1_ps(vectors[neighbor_ids[i + 3] * d + j]);
            __m256 n4 = _mm256_set1_ps(vectors[neighbor_ids[i + 4] * d + j]);
            __m256 n5 = _mm256_set1_ps(vectors[neighbor_ids[i + 5] * d + j]);
            __m256 n6 = _mm256_set1_ps(vectors[neighbor_ids[i + 6] * d + j]);
            __m256 n7 = _mm256_set1_ps(vectors[neighbor_ids[i + 7] * d + j]);

            // 组装邻居向量
            __m256 n_01 = _mm256_unpacklo_ps(n0, n1);
            __m256 n_23 = _mm256_unpacklo_ps(n2, n3);
            __m256 n_45 = _mm256_unpacklo_ps(n4, n5);
            __m256 n_67 = _mm256_unpacklo_ps(n6, n7);

            __m256 n_0123 = _mm256_permute2f128_ps(n_01, n_23, 0x20);
            __m256 n_4567 = _mm256_permute2f128_ps(n_45, n_67, 0x20);

            // 计算差值平方并累加
            __m256 diff = _mm256_sub_ps(q, n_0123);
            sum = _mm256_fmadd_ps(diff, diff, sum);

            diff = _mm256_sub_ps(q, n_4567);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        // 水平求和
        alignas(32) float tmp[8];
        _mm256_storeu_ps(tmp, sum);

        for (size_t k = 0; k < 8; k++) {
            distances_out[i + k] = tmp[k];
        }
    }

    // 处理剩余的邻居
    for (; i < n_neighbors; i++) {
        float dis = 0;
        for (size_t j = 0; j < d; j++) {
            float diff = query[j] - vectors[neighbor_ids[i] * d + j];
            dis += diff * diff;
        }
        distances_out[i] = dis;
    }
}
#endif

// AVX-512优化的批量距离计算（一次16个邻居）
#ifdef __AVX512F__
void batch_distance_computation_avx512(
        const float* query,
        const idx_t* neighbor_ids,
        size_t n_neighbors,
        const float* vectors,
        size_t d,
        float* distances_out) {

    size_t i = 0;
    for (; i + 16 <= n_neighbors; i += 16) {
        __m512 sum = _mm512_setzero_ps();

        for (size_t j = 0; j < d; j++) {
            // 加载查询向量
            __m512 q = _mm512_set1_ps(query[j]);

            // 收集16个邻居的第j维
            __m512i ids = _mm512_loadu_si512((__m512i*)(neighbor_ids + i));
            __m512 neighbor_vals = _mm512_i32gather_ps(
                    vectors, ids, d * sizeof(float), _MM_SCALE_4);

            // 计算差值平方并累加
            __m512 diff = _mm512_sub_ps(q, neighbor_vals);
            sum = _mm512_fmadd_ps(diff, diff, sum);
        }

        _mm512_storeu_ps(distances_out + i, sum);
    }

    // 处理剩余邻居
    for (; i < n_neighbors; i++) {
        float dis = 0;
        for (size_t j = 0; j < d; j++) {
            float diff = query[j] - vectors[neighbor_ids[i] * d + j];
            dis += diff * diff;
        }
        distances_out[i] = dis;
    }
}
#endif
```

### 16.2 预取优化的图遍历

```cpp
// 预取优化的HNSW搜索
template <int PREFETCH_DISTANCE = 4>
struct PrefetchingHNSWSearch {
    static void search_with_prefetch(
            const HNSW& hnsw,
            DistanceComputer& qdis,
            MinimaxHeap& candidates,
            VisitedTable& vt,
            int level) {

        while (candidates.size() > 0) {
            float d_cur = candidates.max();
            storage_idx_t cur = candidates.ids[0];
            candidates.pop();

            // 获取邻居范围
            size_t begin, end;
            hnsw.neighbor_range(cur, level, &begin, &end);

            size_t n_neighbors = end - begin;

            // 预取未来的邻居节点
            for (size_t i = 0; i < n_neighbors; i++) {
                size_t prefetch_idx = i + PREFETCH_DISTANCE;
                if (prefetch_idx < n_neighbors) {
                    storage_idx_t prefetch_id = hnsw.neighbors[begin + prefetch_idx];

                    // 预取向量数据
                    const float* vec = hnsw.storage->get_vector(prefetch_id);
                    _mm_prefetch((const char*)vec, _MM_HINT_T0);

                    // 预取邻居表
                    size_t p_begin, p_end;
                    hnsw.neighbor_range(prefetch_id, level, &p_begin, &p_end);
                    if (p_end - p_begin > 0) {
                        _mm_prefetch(
                            (const char*)&hnsw.neighbors[p_begin],
                            _MM_HINT_T1);
                    }
                }
            }

            // 处理当前节点的邻居
            for (size_t i = 0; i < n_neighbors; i++) {
                storage_idx_t nid = hnsw.neighbors[begin + i];

                if (vt.get(nid)) continue;
                vt.set(nid);

                float dis = qdis(nid);

                if (candidates.size() < hnsw.efSearch) {
                    candidates.push(nid, dis);
                } else if (C::cmp(dis, candidates.max())) {
                    candidates.push(nid, dis);
                }
            }
        }
    }
};
```

### 16.3 ARM NEON优化的邻居遍历

```cpp
#ifdef __ARM_NEON
#include <arm_neon.h>

void batch_distance_computation_neon(
        const float* query,
        const idx_t* neighbor_ids,
        size_t n_neighbors,
        const float* vectors,
        size_t d,
        float* distances_out) {

    // 一次处理4个邻居
    size_t i = 0;
    for (; i + 4 <= n_neighbors; i += 4) {
        float32x4_t sum = vdupq_n_f32(0.0f);

        for (size_t j = 0; j < d; j += 4) {
            // 加载查询向量
            float32x4_t q = vld1q_f32(query + j);

            // 加载4个邻居的向量
            float32x4_t n0 = vld1q_f32(vectors + neighbor_ids[i + 0] * d + j);
            float32x4_t n1 = vld1q_f32(vectors + neighbor_ids[i + 1] * d + j);
            float32x4_t n2 = vld1q_f32(vectors + neighbor_ids[i + 2] * d + j);
            float32x4_t n3 = vld1q_f32(vectors + neighbor_ids[i + 3] * d + j);

            // 转置矩阵（4x4块）
            float32x4x2_t t01 = vtrnq_f32(n0, n1);
            float32x4x2_t t23 = vtrnq_f32(n2, n3);

            float32x4_t n_0 = vcombine_f32(
                vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
            float32x4_t n_1 = vcombine_f32(
                vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
            float32x4_t n_2 = vcombine_f32(
                vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
            float32x4_t n_3 = vcombine_f32(
                vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));

            // 计算距离
            float32x4_t diff0 = vsubq_f32(q, n_0);
            sum = vmlaq_f32(sum, diff0, diff0);

            float32x4_t diff1 = vsubq_f32(q, n_1);
            sum = vmlaq_f32(sum, diff1, diff1);

            float32x4_t diff2 = vsubq_f32(q, n_2);
            sum = vmlaq_f32(sum, diff2, diff2);

            float32x4_t diff3 = vsubq_f32(q, n_3);
            sum = vmlaq_f32(sum, diff3, diff3);
        }

        // 水平求和
        float32x2_t sum_low = vget_low_f32(sum);
        float32x2_t sum_high = vget_high_f32(sum);
        sum_low = vadd_f32(sum_low, sum_high);

        float dis_array[4];
        vst1q_f32(dis_array, sum);
        distances_out[i + 0] = dis_array[0];
        distances_out[i + 1] = dis_array[1];
        distances_out[i + 2] = dis_array[2];
        distances_out[i + 3] = dis_array[3];
    }

    // 处理剩余邻居
    for (; i < n_neighbors; i++) {
        float dis = 0;
        for (size_t j = 0; j < d; j++) {
            float diff = query[j] - vectors[neighbor_ids[i] * d + j];
            dis += diff * diff;
        }
        distances_out[i] = dis;
    }
}
#endif
```

---

## 17. 无锁并发HNSW构建

### 17.1 Lock-Free邻居更新

```cpp
// 无锁的邻居表更新
template <typename T>
class LockFreeNeighborList {
    struct Node {
        std::atomic<idx_t> id;
        std::atomic<float> distance;
        std::atomic<Node*> next;

        Node(idx_t i, float d) : id(i), distance(d), next(nullptr) {}
    };

    std::atomic<Node*> head;
    std::atomic<size_t> size;
    int max_neighbors;

public:
    LockFreeNeighborList(int M) : head(nullptr), size(0), max_neighbors(M) {}

    // 尝试插入新邻居
    bool try_insert(idx_t id, float distance) {
        Node* new_node = new Node(id, distance);

        // 无锁链表插入
        Node* old_head = head.load(std::memory_order_acquire);
        new_node->next.store(old_head, std::memory_order_release);

        if (head.compare_exchange_strong(old_head, new_node,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
            size_t new_size = size.fetch_add(1, std::memory_order_acq_rel) + 1;

            // 如果超过最大邻居数，需要裁剪
            if (new_size > (size_t)max_neighbors) {
                prune_to_max();
            }

            return true;
        }

        delete new_node;
        return false;
    }

    // 裁剪到max_neighbors个最近邻居
    void prune_to_max() {
        // 收集所有邻居
        std::vector<std::pair<float, idx_t>> neighbors;
        Node* current = head.load(std::memory_order_acquire);

        while (current != nullptr) {
            neighbors.push_back({
                current->distance.load(std::memory_order_acquire),
                current->id.load(std::memory_order_acquire)
            });
            current = current->next.load(std::memory_order_acquire);
        }

        // 排序并保留最近的max_neighbors个
        std::sort(neighbors.begin(), neighbors.end());
        if (neighbors.size() > (size_t)max_neighbors) {
            neighbors.resize(max_neighbors);
        }

        // 重建链表
        Node* new_head = nullptr;
        for (auto& [dis, id] : neighbors) {
            Node* node = new Node(id, dis);
            node->next.store(new_head, std::memory_order_release);
            new_head = node;
        }

        // 原子替换
        Node* old_head = head.exchange(new_head, std::memory_order_acq_rel);

        // 释放旧链表
        while (old_head != nullptr) {
            Node* next = old_head->next.load();
            delete old_head;
            old_head = next;
        }

        size.store(neighbors.size(), std::memory_order_release);
    }

    size_t get_size() const {
        return size.load(std::memory_order_acquire);
    }

    std::vector<idx_t> get_neighbors() const {
        std::vector<idx_t> result;
        Node* current = head.load(std::memory_order_acquire);

        while (current != nullptr) {
            result.push_back(current->id.load(std::memory_order_acquire));
            current = current->next.load(std::memory_order_acquire);
        }

        return result;
    }
};
```

### 17.2 图剪枝与压缩

```cpp
// HNSW图剪枝：移除冗余边
class HNSWGraphPruner {
public:
    // 基于三角不等式剪枝
    static void prune_with_triangle_inequality(
            HNSW& hnsw,
            float prune_threshold) {

        for (int level = 0; level <= hnsw.max_level; level++) {
            for (idx_t node_id = 0; node_id < hnsw.ntotal; node_id++) {
                prune_node_neighbors(hnsw, node_id, level,
                                    prune_threshold);
            }
        }
    }

    static void prune_node_neighbors(
            HNSW& hnsw,
            idx_t node_id,
            int level,
            float threshold) {

        size_t begin, end;
        hnsw.neighbor_range(node_id, level, &begin, &end);

        std::vector<idx_t> to_remove;

        // 检查每条边的必要性
        for (size_t i = begin; i < end; i++) {
            idx_t neighbor_a = hnsw.neighbors[i];

            // 检查是否存在其他邻居使这条边冗余
            bool redundant = false;

            for (size_t j = begin; j < end; j++) {
                if (i == j) continue;

                idx_t neighbor_b = hnsw.neighbors[j];

                // 计算三角不等式
                float dis_a_b = get_distance(hnsw, neighbor_a, neighbor_b);
                float dis_node_b = get_distance(hnsw, node_id, neighbor_b);
                float dis_node_a = get_distance(hnsw, node_id, neighbor_a);

                // 如果通过b到a的距离不大于直接到a的距离
                // 则边(node, a)是冗余的
                if (dis_node_b + dis_a_b <= dis_node_a * (1.0f + threshold)) {
                    redundant = true;
                    break;
                }
            }

            if (redundant) {
                to_remove.push_back(i);
            }
        }

        // 移除冗余边
        if (!to_remove.empty()) {
            std::vector<idx_t> new_neighbors;
            for (size_t i = begin; i < end; i++) {
                if (std::find(to_remove.begin(), to_remove.end(), i) ==
                    to_remove.end()) {
                    new_neighbors.push_back(hnsw.neighbors[i]);
                }
            }

            // 更新邻居列表
            for (size_t i = 0; i < new_neighbors.size(); i++) {
                hnsw.neighbors[begin + i] = new_neighbors[i];
            }
        }
    }

private:
    static float get_distance(const HNSW& hnsw, idx_t a, idx_t b) {
        // 获取两个向量之间的距离
        return 0.0f;
    }
};
```

---

## 练习题

1. 实现简化的HNSW构建
2. 研究efSearch对性能的影响
3. 实现贪婪搜索算法
4. 比较HNSW与IVF的性能
5. 实现SIMD优化的邻居遍历
6. 分析HNSW的缓存命中率
7. 实现无锁的邻居更新
8. 实现图剪枝算法
9. 比较AVX2和AVX-512的批量距离计算性能
10. 实现压缩的HNSW图存储

## 扩展阅读

- faiss/impl/HNSW.h - HNSW实现
- faiss/IndexHNSW.h - HNSW索引
- [HNSW论文](https://arxiv.org/abs/1603.09320)
- [NSMlib](https://github.com/searchivarius/nmslib)
- faiss/utils/distances_simd.cpp - SIMD距离计算优化
