# HNSW图索引底层实现深度剖析 - HNSW.cpp源码解析

## 概述

`faiss/impl/HNSW.cpp` 是Faiss中HNSW（Hierarchical Navigable Small World）图索引的核心实现。HNSW是一种高效的近似最近邻搜索算法，通过构建多层图结构实现O(log n)的搜索复杂度。本文档深入剖析其图结构设计、构建算法、搜索优化和并行化策略。

---

## 1. HNSW数据结构

### 1.1 核心数据成员

```cpp
struct HNSW {
    // 每个节点的层号
    std::vector<int> levels;  // size = ntotal

    // 邻居表的偏移量（压缩存储）
    std::vector<size_t> offsets;  // size = ntotal + 1

    // 所有邻居（扁平化存储）
    std::vector<storage_idx_t> neighbors;  // 可变长度

    // 每层的累积邻居数
    std::vector<int> cum_nneighbor_per_level;

    // 节点分配到各层的概率
    std::vector<float> assign_probas;

    // 最大层号和入口点
    int max_level = -1;
    storage_idx_t entry_point = -1;

    RandomGenerator rng;
};
```

**内存布局详解：**

```
假设有3个节点，M=16：

levels: [2, 1, 0]  // 节点0在层2，节点1在层1，节点2在层0

cum_nneighbor_per_level: [0, 32, 48, 64]
// 层0: 32个邻居 (2*M)
// 层1: 16个邻居 (M)
// 层2: 16个邻居 (M)
// 累积: [0, 32, 48, 64]

offsets: [0, 80, 96, 96]
// 节点0的邻居偏移: [0, 32)
// 节点1的邻居偏移: [80, 96)  (层0: 80-96, 共16个)
// 节点2的邻居偏移: [96, ...)  (层0: 96-112, 共16个)

neighbors (扁平化):
[节点0的64个邻居]
[节点1的16个邻居]
[节点2的16个邻居]
...
```

### 1.2 邻居范围计算

```cpp
void HNSW::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
        const {
    size_t o = offsets[no];
    *begin = o + cum_nb_neighbors(layer_no);
    *end = o + cum_nb_neighbors(layer_no + 1);
}
```

**算法详解：**

```cpp
// 示例：获取节点1在层0的邻居范围
idx_t no = 1;
int layer_no = 0;
size_t o = offsets[1] = 80;
size_t begin = 80 + cum_nb_neighbors(0) = 80 + 0 = 80;
size_t end = 80 + cum_nb_neighbors(1) = 80 + 32 = 32;

// 邻居存储在neighbors[80:112)
```

**计算公式：**
```
邻居开始位置 = offsets[node_id] + cum_nneighbor_per_level[layer]
邻居结束位置 = offsets[node_id] + cum_nneighbor_per_level[layer + 1]
邻居数量 = cum_nneighbor_per_level[layer + 1] - cum_nneighbor_per_level[layer]
```

### 1.3 层数分配概率

```cpp
void HNSW::set_default_probas(int M, float levelMult) {
    int nn = 0;
    cum_nneighbor_per_level.push_back(0);
    for (int level = 0;; level++) {
        // 指数分布概率
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));

        if (proba < 1e-9) {
            break;
        }

        assign_probas.push_back(proba);

        // 层0: 2*M个邻居，其他层: M个邻居
        nn += level == 0 ? M * 2 : M;
        cum_nneighbor_per_level.push_back(nn);
    }
}
```

**概率分布公式：**

```
P(level = l) = exp(-l / λ) × (1 - exp(-1 / λ))

其中 λ = levelMult = ln(M)

示例 (M=16, λ=ln(16)≈2.77):
P(0) = exp(0) × (1 - exp(-1/2.77)) ≈ 0.30  (30%)
P(1) = exp(-1/2.77) × (1 - exp(-1/2.77)) ≈ 0.22  (22%)
P(2) = exp(-2/2.77) × (1 - exp(-1/2.77)) ≈ 0.16  (16%)
P(3) ≈ 0.12
P(4) ≈ 0.09
...
```

### 1.4 随机层号生成

```cpp
int HNSW::random_level() {
    double f = rng.rand_float();  // [0, 1)

    // 线性搜索（可以用二分优化）
    for (int level = 0; level < assign_probas.size(); level++) {
        if (f < assign_probas[level]) {
            return level;
        }
        f -= assign_probas[level];
    }

    // 极低概率事件
    return assign_probas.size() - 1;
}
```

**算法示例：**

```
assign_probas = [0.30, 0.22, 0.16, 0.12, ...]

随机数f = 0.25:
level=0: f < 0.30? 是 → return 0

随机数f = 0.55:
level=0: f < 0.30? 否, f = 0.55 - 0.30 = 0.25
level=1: f < 0.22? 否, f = 0.25 - 0.22 = 0.03
level=2: f < 0.16? 是 → return 2
```

---

## 2. 图构建算法

### 2.1 添加节点主流程

```cpp
void HNSW::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,      // 新节点的层号
        int pt_id,         // 新节点的ID
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {

    // ==================== Step 1: 确定入口点 ====================
    storage_idx_t nearest;
    #pragma omp critical
    {
        nearest = entry_point;

        if (nearest == -1) {
            // 第一个节点
            max_level = pt_level;
            entry_point = pt_id;
        }
    }

    if (nearest < 0) {
        return;  // 空图
    }

    omp_set_lock(&locks[pt_id]);

    // ==================== Step 2: 从高层到pt_level贪心搜索 ====================
    int level = max_level;
    float d_nearest = ptdis(nearest);

    for (; level > pt_level; level--) {
        // 在每一层贪心更新最近邻
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
    }

    // ==================== Step 3: 在level=pt_level到0添加连接 ====================
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

    omp_unset_lock(&locks[pt_id]);

    // ==================== Step 4: 更新入口点 ====================
    if (pt_level > max_level) {
        max_level = pt_level;
        entry_point = pt_id;
    }
}
```

**构建流程图：**

```
新节点 (pt_id, pt_level)
        │
        ▼
┌───────────────────┐
│ 获取入口点entry_point │
│ 如果是第一个节点     │
│   max_level = pt_level│
│   entry_point = pt_id │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 从max_level向下    │
│ 贪心搜索到pt_level │
│ (greedy_update_nearest)│
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 对每层level       │
│   [pt_level, 0]   │
│   搜索候选邻居     │
│ (search_neighbors) │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 精简邻居列表      │
│ (shrink_neighbor) │
│ 添加双向连接      │
└─────────┬─────────┘
          │
          ▼
    更新入口点（如果需要）
```

### 2.2 搜索候选邻居

```cpp
void search_neighbors_to_add(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,  // 最近邻居
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        bool reference_version) {

    // candidates: 按距离从远到近排序（最大堆）
    // results: 按距离从近到远排序（最小堆）
    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    while (!candidates.empty()) {
        // 取最近的候选
        const NodeDistFarther& currEv = candidates.top();

        // 停止条件：所有剩余候选都比results中最远的还远
        if (currEv.d > results.top().d) {
            break;
        }

        int currNode = currEv.id;
        candidates.pop();

        // 遍历邻居
        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);

        if (reference_version) {
            // ==================== 参考版本 ====================
            for (size_t i = begin; i < end; i++) {
                storage_idx_t nodeId = hnsw.neighbors[i];
                if (nodeId < 0) {
                    break;
                }
                if (vt.get(nodeId)) {
                    continue;  // 已访问
                }
                vt.set(nodeId);

                float dis = qdis(nodeId);

                // 如果results未满或距离更近
                if (results.size() < hnsw.efConstruction ||
                    results.top().d > dis) {
                    results.emplace(dis, nodeId);
                    candidates.emplace(dis, nodeId);
                    if (results.size() > hnsw.efConstruction) {
                        results.pop();  // 保持大小为efConstruction
                    }
                }
            }
        } else {
            // ==================== 优化版本：批处理 ====================

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
                if (nodeId < 0) {
                    break;
                }
                if (vt.get(nodeId)) {
                    continue;
                }
                vt.set(nodeId);

                buffered_ids[n_buffered] = nodeId;
                n_buffered += 1;

                // 批量处理4个邻居
                if (n_buffered == 4) {
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

**优化技术详解：**

1. **双堆结构**：
   ```cpp
   // candidates: 最大堆（最远的在顶）
   std::priority_queue<NodeDistFarther> candidates;

   // results: 最小堆（最远的在顶）
   std::priority_queue<NodeDistCloser> results;

   // 停止条件：candidates最近 > results最远
   if (currEv.d > results.top().d) {
       break;
   }
   ```

2. **批量距离计算**：
   ```cpp
   // 单次计算 vs 批量计算
   // 单次: 4次函数调用 + 4次距离计算
   float dis0 = qdis(id0);
   float dis1 = qdis(id1);
   float dis2 = qdis(id2);
   float dis3 = qdis(id3);

   // 批量: 1次函数调用 + SIMD优化
   float dis[4];
   qdis.distances_batch_4(id0, id1, id2, id3, dis[0], dis[1], dis[2], dis[3]);
   ```

3. **预取优化**：
   - L1预取：访问访问模式
   - L2预取：提前访问候选节点的邻居表

### 2.3 邻居列表精简（Shrink）

```cpp
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
        float dist_v1_q = v1.d;  // v1到查询点的距离

        bool good = true;

        // 检查是否应该保留v1
        for (NodeDistFarther v2 : output) {
            float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

            // 条件：如果v2到v1的距离 < v1到查询点的距离
            // 则v1是多余的（可以通过v2更近地到达）
            if (dist_v1_v2 < dist_v1_q) {
                good = false;
                break;
            }
        }

        if (good) {
            output.push_back(v1);
            if (output.size() >= max_size) {
                return;  // 已达到最大容量
            }
        } else if (keep_max_size_level0) {
            outsiders.push_back(v1);  // 暂存被拒绝的候选
        }
    }

    // 如果还不够，从outsiders中补充
    size_t idx = 0;
    while (keep_max_size_level0 && (output.size() < max_size) &&
           (idx < outsiders.size())) {
        output.push_back(outsiders[idx++]);
    }
}
```

**精简算法原理：**

```
目标：从候选邻居中选择最能代表各个方向的子集

输入候选：
Q (查询点)
A (候选1): dist(Q, A) = 10
B (候选2): dist(Q, B) = 12
C (候选3): dist(Q, C) = 15

检查过程：
1. output = [], 尝试添加A
   output = [A]

2. 尝试添加B，检查dist(A, B) < dist(Q, B)?
   如果dist(A, B) = 5 < 12，则B被拒绝（可以通过A更近地到达B）
   如果dist(A, B) = 20 > 12，则B被接受（B提供了新的方向）

3. 尝试添加C，检查dist(A, C) < dist(Q, C)或dist(B, C) < dist(Q, C)?
   ...

最终output = [A, B, ...] (max_size个)
```

### 2.4 添加连接

```cpp
void add_link(
        HNSW& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level,
        bool keep_max_size_level0) {

    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);

    // ==================== 情况1：有空位 ====================
    if (hnsw.neighbors[end - 1] == -1) {
        size_t i = end;
        while (i > begin) {
            if (hnsw.neighbors[i - 1] != -1) {
                break;
            }
            i--;
        }
        hnsw.neighbors[i] = dest;  // 填入空位
        return;
    }

    // ==================== 情况2：已满，需要精简 ====================

    // 收集所有现有邻居+新候选的距离
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);

    for (size_t i = begin; i < end; i++) {
        storage_idx_t neigh = hnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    // 精简到max_size
    shrink_neighbor_list(qdis, resultSet, end - begin, keep_max_size_level0);

    // 写回邻居列表
    size_t i = begin;
    while (resultSet.size()) {
        hnsw.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }

    // 清空剩余位置
    while (i < end) {
        hnsw.neighbors[i++] = -1;
    }
}
```

**双向连接建立：**

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
    std::priority_queue<NodeDistCloser> link_targets;
    search_neighbors_to_add(*this, ptdis, link_targets, nearest, d_nearest, level, vt);

    // 2. 精简到M个
    int M = nb_neighbors(level);
    ::faiss::shrink_neighbor_list(ptdis, link_targets, M, keep_max_size_level0);

    // 3. 添加连接（双向）
    std::vector<storage_idx_t> neighbors_to_add;
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        add_link(*this, ptdis, pt_id, other_id, level, keep_max_size_level0);
        neighbors_to_add.push_back(other_id);
        link_targets.pop();
    }

    // 4. 释放自己的锁，添加反向连接
    omp_unset_lock(&locks[pt_id]);

    for (storage_idx_t other_id : neighbors_to_add) {
        omp_set_lock(&locks[other_id]);
        add_link(*this, ptdis, other_id, pt_id, level, keep_max_size_level0);
        omp_unset_lock(&locks[other_id]);
    }

    omp_set_lock(&locks[pt_id]);
}
```

---

## 3. 搜索算法

### 3.1 主搜索流程

```cpp
int search_from_candidates(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params) {

    int nres = nres_in;
    int ndis = 0;

    bool do_dis_check;
    int efSearch;
    const IDSelector* sel;
    extract_search_params(hnsw, params, do_dis_check, efSearch, sel);

    C::T threshold = res.threshold;

    // ==================== Step 1: 处理初始候选 ====================
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];

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

    // ==================== Step 2: BFS搜索 ====================
    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);  // 取最近的候选

        // 停止条件（相对距离检查）
        if (do_dis_check) {
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;  // 已有足够的更近距离
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        // ==================== 预取邻居 ====================
        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }
            prefetch_L2(vt.visited.data() + v1);  // 预取访问标记
            jmax += 1;
        }

        // ==================== 批量处理邻居 ====================
        int counter = 0;
        size_t saved_j[4];

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
            counter += vget ? 0 : 1;  // 只统计未访问的

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

        // 另一个停止条件
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    // ==================== Step 3: 更新统计 ====================
    if (level == 0) {
        stats.n1++;  // 搜索次数
        if (candidates.size() == 0) {
            stats.n2++;  // 耗尽候选次数
        }
        stats.ndis += ndis;   // 距离计算次数
        stats.nhops += nstep;  // 跳数
    }

    return nres;
}
```

**搜索优化技术：**

1. **L2预取**：
   ```cpp
   prefetch_L2(vt.visited.data() + v1);
   // 提前访问visited标记，减少cache miss
   ```

2. **批量距离计算**：
   ```cpp
   qdis.distances_batch_4(id0, id1, id2, id3, dis[0], dis[1], dis[2], dis[3]);
   // SIMD优化的批量计算
   ```

3. **提前终止**：
   ```cpp
   if (do_dis_check) {
       int n_dis_below = candidates.count_below(d0);
       if (n_dis_below >= efSearch) {
           break;  // 已有足够的更近候选
       }
   }
   ```

4. **动态阈值**：
   ```cpp
   C::T threshold = res.threshold;
   if (dis < threshold) {
       if (res.add_result(dis, idx)) {
           threshold = res.threshold;  // 更新阈值
       }
   }
   ```

### 3.2 多层搜索

```cpp
void HNSW::search(
        DistanceComputer& qdis,
        idx_t k,
        idx_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    // ==================== Step 1: 从入口点开始 ====================
    if (entry_point == -1) {
        return;  // 空图
    }

    // ==================== Step 2: 从max_level到1层贪心搜索 ====================
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for (int level = max_level; level > 0; level--) {
        greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
    }

    // ==================== Step 3: 在层0执行完整搜索 ====================
    int ef = std::max(effSearch, k);
    MinimaxHeap candidates(ef);
    candidates.push(nearest, d_nearest);

    VisitedTable vt(ntotal);
    search_from_candidates(
            *this, qdis, *res, candidates, vt, stats, 0, 0, params);
}
```

**搜索层次结构：**

```
查询点Q
    │
    ▼
┌────────────────────────┐
│  max_level层 (高层)     │
│  节点少，距离远         │
│  贪心搜索最近邻         │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  中间层                │
│  节点适中，距离适中     │
│  贪心搜索最近邻         │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  层0 (底层)            │
│  节点多，距离近         │
│  完整BFS搜索            │
│  返回top-k结果          │
└────────────────────────┘
```

---

## 4. 性能优化技术

### 4.1 批量距离计算

```cpp
// DistanceComputer接口
class DistanceComputer {
public:
    // 单次距离计算
    virtual float operator()(idx_t i) = 0;

    // 批量距离计算（4个）
    virtual void distances_batch_4(
            idx_t idx0, idx_t idx1, idx_t idx2, idx_t idx3,
            float& dis0, float& dis1, float& dis2, float& dis3) {
        dis0 = operator()(idx0);
        dis1 = operator()(idx1);
        dis2 = operator()(idx2);
        dis3 = operator()(idx3);
    }
};
```

**SIMD优化版本：**

```cpp
// L2距离的SIMD实现
void distances_batch_4(
        idx_t idx0, idx_t idx1, idx_t idx2, idx_t idx3,
        float& dis0, float& dis1, float& dis2, float& dis3) final {
    // 1. 加载4个向量（部分）
    const float* y0 = get_vector(idx0);
    const float* y1 = get_vector(idx1);
    const float* y2 = get_vector(idx2);
    const float* y3 = get_vector(idx3);

    __m128 x0 = _mm_loadu_ps(x);
    __m128 x1 = _mm_loadu_ps(x + 4);
    __m128 x2 = _mm_loadu_ps(x + 8);
    __m128 x3 = _mm_loadu_ps(x + 12);

    // 2. 计算距离
    __m128 accu = ElementOpL2::op(x0, _mm_loadu_ps(y0));
    accu = _mm_add_ps(accu, ElementOpL2::op(x1, _mm_loadu_ps(y1)));
    accu = _mm_add_ps(accu, ElementOpL2::op(x2, _mm_loadu_ps(y2)));
    accu = _mm_add_ps(accu, ElementOpL2::op(x3, _mm_loadu_ps(y3)));

    accu = _mm_hadd_ps(accu, accu);
    accu = _mm_hadd_ps(accu, accu);

    // 3. 存储结果
    float results[4];
    _mm_storeu_ps(results, accu);
    dis0 = results[0];
    dis1 = results[1];
    dis2 = results[2];
    dis3 = results[3];
}
```

### 4.2 并行构建优化

```cpp
void HNSW::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {

    // ... 搜索逻辑 ...

    for (; level >= 0; level--) {
        // ==================== 关键区保护 ====================
        // 1. 释放自己的锁
        omp_unset_lock(&locks[pt_id]);

        // 2. 添加双向连接
        for (storage_idx_t other_id : neighbors_to_add) {
            omp_set_lock(&locks[other_id]);
            add_link(*this, ptdis, other_id, pt_id, level, ...);
            omp_unset_lock(&locks[other_id]);
        }

        // 3. 重新获取自己的锁
        omp_set_lock(&locks[pt_id]);
    }
}
```

**锁策略：**

1. **细粒度锁**：每个节点一个锁
2. **最小化临界区**：只在修改邻居表时持锁
3. **避免死锁**：按固定顺序获取锁

### 4.3 内存预取

```cpp
// 预取visited标记
for (size_t j = begin; j < end; j++) {
    int v1 = hnsw.neighbors[j];
    if (v1 < 0) {
        break;
    }
    prefetch_L2(vt.visited.data() + v1);  // L2缓存预取
    jmax += 1;
}

// 稍后使用
for (size_t j = begin; j < jmax; j++) {
    int v1 = hnsw.neighbors[j];
    bool vget = vt.get(v1);  // 可能已命中cache
    ...
}
```

**预取距离分析：**

```
假设：
- L2缓存延迟：~20周期
- L2缓存行大小：64字节
- vt.visited[]是uint8_t数组

预取效果：
- 无预取：每次vt.get()可能cache miss，~20周期
- 有预取：vt.get()很可能cache hit，~4周期

加速比：~5x
```

### 4.4 VisitedTable优化

```cpp
class VisitedTable {
    std::vector<uint8_t> visited;  // 访问标记
    size_t ntotal;

public:
    bool get(storage_idx_t idx) const {
        return visited[idx] != 0;
    }

    void set(storage_idx_t idx) {
        visited[idx] = 1;
    }

    void advance() {
        // 批量清理（如果需要）
    }
};
```

**优化技巧：**

1. **紧凑存储**：使用uint8_t而非bool
2. **批量操作**：可以批量清理
3. **缓存友好**：顺序访问，cache命中率高

---

## 5. 高级搜索模式

### 5.1 Panorama搜索

```cpp
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

    // ==================== Panorama优化技术 ====================
    // 1. 计算查询的累积和
    std::vector<float> query_cum_sums(panorama_index->pano.n_levels + 1);
    panorama_index->pano.compute_query_cum_sums(query, query_cum_sums.data());

    // 2. 层级距离计算
    size_t batch_size = initial_size;
    size_t curr_panorama_level = 0;

    while (curr_panorama_level < num_panorama_levels && batch_size > 0) {
        // 2.1 计算部分点积
        float query_cum_norm = query_cum_sums[curr_panorama_level + 1];
        size_t start_dim = curr_panorama_level * level_width_floats;
        size_t end_dim = (curr_panorama_level + 1) * level_width_floats;

        // 2.2 批量计算距离
        for (size_t i = 0; i + 3 < batch_size; i += 4) {
            float dp[4];
            flat_codes_qdis->partial_dot_product_batch_4(
                    idx_0, idx_1, idx_2, idx_3,
                    dp[0], dp[1], dp[2], dp[3],
                    start_dim, end_dim - start_dim);

            // 2.3 更新精确距离界
            float new_exact_0 = exact_distances[i + 0] - 2 * dp[0];
            float new_exact_1 = exact_distances[i + 1] - 2 * dp[1];
            float new_exact_2 = exact_distances[i + 2] - 2 * dp[2];
            float new_exact_3 = exact_distances[i + 3] - 2 * dp[3];

            // 2.4 基于累积和的界
            float cum_sum_0 = get_cum_sum(idx_0)[curr_panorama_level + 1];
            float cs_bound_0 = 2.0f * cum_sum_0 * query_cum_norm;
            float lower_bound_0 = new_exact_0 - cs_bound_0;

            // 2.5 提前终止
            if (lower_bound_0 < threshold) {
                add_to_heap(idx_0, lower_bound_0);
            }
        }

        curr_panorama_level++;
    }

    return nres;
}
```

**Panorama优化原理：**

```
传统L2距离计算：
dist²(q, x) = ||q||² + ||x||² - 2·q·x
           需要计算完整的点积

Panorama层级计算：
将向量分成多个层级：
  level 0: x[0:64]
  level 1: x[64:128]
  level 2: x[128:192]
  ...

每一层的累积和：
  cum_sum[0] = sum(x[0:64])
  cum_sum[1] = sum(x[0:128])
  cum_sum[2] = sum(x[0:192])

距离界：
  lower_bound = ||q||² + ||x||² - 2·q·x[0:level_i]
              - 2·cum_sum[i]·query_cum_norm

如果lower_bound > threshold：
  该节点不可能进入top-k，提前终止
```

---

## 6. 性能统计与调优

### 6.1 统计信息收集

```cpp
struct HNSWStats {
    size_t n1;      // 搜索次数
    size_t n2;      // 耗尽候选次数
    size_t ndis;    // 距离计算次数
    size_t nhops;   // 跳数

    HNSWStats() : n1(0), n2(0), ndis(0), nhops(0) {}
};

// 收集统计
if (level == 0) {
    stats.n1++;
    if (candidates.size() == 0) {
        stats.n2++;
    }
    stats.ndis += ndis;
    stats.nhops += nstep;
}
```

### 6.2 性能指标

```cpp
void HNSW::print_neighbor_stats(int level) const {
    printf("stats on level %d, max %d neighbors per vertex:\n",
           level, nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;

    #pragma omp parallel for reduction(+ : tot_neigh) reduction(+ : tot_common) \
        reduction(+ : tot_reciprocal) reduction(+ : n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);

            // 统计邻居数量
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0) {
                    break;
                }
                neighset.insert(neighbors[j]);
            }

            // 统计双向连接
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                if (i2 < 0) {
                    break;
                }
                // 检查i2的邻居中是否包含i
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    if (neighbors[j2] == i) {
                        n_reciprocal++;
                        break;
                    }
                }
            }

            // 统计公共邻居
            int n_common = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                // ... 统计i2的邻居中在neighset中的数量
            }

            tot_neigh += n_neigh;
            tot_reciprocal += n_reciprocal;
            tot_common += n_common;
        }
    }

    float normalizer = n_node;
    printf("   neighbors per node: %.2f (%zd)\n", tot_neigh / normalizer, tot_neigh);
    printf("   nb of reciprocal neighbors: %.2f\n", tot_reciprocal / normalizer);
    printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f\n",
           tot_common / normalizer);
}
```

**性能调优建议：**

1. **efSearch参数**：
   ```
   efSearch越大，召回率越高，但速度越慢
   建议：
   - 高召回率：efSearch = 64-128
   - 平衡模式：efSearch = 32-64
   - 高速模式：efSearch = 16-32
   ```

2. **efConstruction参数**：
   ```
   efConstruction影响构建质量
   建议：efConstruction >= M * 4
   ```

3. **M参数**：
   ```
   M控制每层邻居数
   建议：
   - 高质量：M = 32-64
   - 平衡：M = 16-32
   - 高速：M = 8-16
   ```

---

## 7. 实际应用示例

### 7.1 构建HNSW索引

```cpp
#include <faiss/IndexHNSW.h>

// 1. 创建索引
int d = 128;  // 向量维度
int M = 16;  // 邻居数
faiss::IndexHNSWFlat index(d, M);

// 2. 添加向量
size_t n = 100000;
float* xb = new float[n * d];
index.add(n, xb);

// 3. 搜索
int k = 10;
float* distances = new float[k];
int64_t* labels = new int64_t[k];

index.search(1, xb, k, distances, labels);
```

### 7.2 性能优化配置

```cpp
// 1. 调整efSearch
faiss::IndexHNSWFlat index(d, M);
index.hnsw.efSearch = 64;  // 默认16

// 2. 调整efConstruction
index.hnsw.efConstruction = 64;  // 默认40

// 3. 并行构建
index.hnsw.search_bounded_queue = false;  // 禁用有界队列

// 4. 启用相对距离检查
index.hnsw.check_relative_distance = true;
```

---

## 8. 与其他图算法对比

### 8.1 HNSW vs NSG

| 特性 | HNSW | NSG |
|------|------|-----|
| 图结构 | 分层图 | 单层图 |
| 构建复杂度 | O(n log n × M) | O(n × M) |
| 搜索复杂度 | O(log n × M) | O(M × √n) |
| 内存占用 | 高（多层） | 低（单层） |
| 召回率 | 高 | 中高 |
| 查询速度 | 快 | 中快 |

### 8.2 HNSW vs IVF

| 特性 | HNSW | IVF |
|------|------|-----|
| 索引类型 | 图索引 | 倒排文件索引 |
| 构建时间 | 长 | 短 |
| 内存占用 | 高 | 低 |
| 召回率 | 极高 | 高 |
| 查询速度 | 快（小数据集） | 快（大数据集） |
| 可扩展性 | 中 | 高 |

---

## 9. 调试与性能分析

### 9.1 调试工具

```cpp
// 1. 打印邻居统计
index.hnsw.print_neighbor_stats(0);

// 2. 检查图结构
for (int i = 0; i < 100; i++) {
    size_t begin, end;
    index.hnsw.neighbor_range(i, 0, &begin, &end);
    printf("Node %d: ", i);
    for (size_t j = begin; j < end; j++) {
        printf("%d ", index.hnsw.neighbors[j]);
    }
    printf("\n");
}

// 3. 验证搜索一致性
// 同一查询多次搜索，结果应该一致
```

### 9.2 性能分析

```bash
# 使用perf分析
perf record -e cycles,instructions,cache-misses ./your_program
perf report

# 查看HNSW统计信息
index.hnsw.stats.ndis   // 距离计算次数
index.hnsw.stats.nhops  // 平均跳数
index.hnsw.stats.n2     // 耗尽候选比率
```

---

## 总结

HNSW.cpp展示了现代图索引的多个核心优化技术：

1. **分层图结构**：对数级搜索复杂度
2. **双向连接**：提升图连通性和召回率
3. **邻居精简**：动态优化邻居列表
4. **批量距离计算**：SIMD优化
5. **内存预取**：L2缓存预取
6. **并行构建**：细粒度锁
7. **Panorama优化**：层级距离界
8. **提前终止**：多种停止条件

这些优化使得HNSW成为高效的近似最近邻搜索算法，广泛应用于向量检索系统。

---

## 参考资料

- HNSW论文: "Efficient and robust approximate nearest neighbor search by Hierarchical Navigable Small World graphs"
- Faiss源码: https://github.com/facebookresearch/faiss
- 图算法优化: "Graph-based search in Faiss"
