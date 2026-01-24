# NSG 导航传播图索引底层实现深度剖析 - NSG.cpp 源码解析

## 1. 概述

NSG (Navigating Spreading-out Graph) 是另一种高效的图索引结构，与 HNSW 不同，它采用单层图结构并基于 k-NN 图进行构建。NSG 通过精心设计的邻居选择和遮挡关系检测，在保持搜索精度的同时降低了图的构建复杂度。

**论文**: Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph, VLDB 2019

### 核心特性
- **单层图结构**: 与 HNSW 的多层不同，NSG 只有一层
- **基于 k-NN 构建**: 从精确 k-NN 图开始修剪
- **遮挡关系检测**: 避免冗余邻居
- **反向链接**: 动态添加反向边
- **连通性保证**: tree_grow 确保图完全连通

## 2. 数据结构

### 2.1 Neighbor 结构 (NSG.cpp:37-49)

```cpp
struct Neighbor {
    int32_t id;        // 邻居节点 ID
    float distance;   // 到查询点的距离
    bool flag;        // 标记位：是否已访问/扩展

    Neighbor() = default;
    Neighbor(int id, float distance, bool f)
            : id(id), distance(distance), flag(f) {}

    inline bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};
```

**flag 的作用**:
- `flag = true`: 该候选尚未扩展，需要探索其邻居
- `flag = false`: 该候选已扩展过

### 2.2 Node 结构 (NSG.cpp:51-66)

```cpp
struct Node {
    int32_t id;
    float distance;

    Node() = default;
    Node(int id, float distance) : id(id), distance(distance) {}

    inline bool operator<(const Node& other) const {
        return distance < other.distance;
    }

    // 为了兼容性
    inline bool operator<(int other) const {
        return id < other;
    }
};
```

### 2.3 Graph 结构 (NSG.h:53-102)

```cpp
template <class node_t>
struct Graph {
    node_t* data;    ///< 扁平化的邻接矩阵，大小 N×K
    int K;           ///< 每个节点的邻居数
    int N;           ///< 节点总数
    bool own_fields; ///< 是否拥有底层内存

    // 访问节点 i 的第 j 个邻居
    inline node_t at(int i, int j) const {
        return data[i * K + j];
    }

    inline node_t& at(int i, int j) {
        return data[i * K + j];
    }

    // 获取节点 i 的所有邻居（用于搜索）
    virtual size_t get_neighbors(int i, node_t* neighbors) const {
        for (int j = 0; j < K; j++) {
            if (data[i * K + j] < 0) {
                return j;  // 返回实际邻居数
            }
            neighbors[j] = data[i * K + j];
        }
        return K;
    }
};
```

**内存布局**:
```
data[0] = [neighbor[0], neighbor[1], ..., neighbor[K-1]]
data[1] = [neighbor[0], neighbor[1], ..., neighbor[K-1]]
...
data[N-1] = [neighbor[0], neighbor[1], ..., neighbor[K-1]]
```

### 2.4 NSG 主结构 (NSG.h:108-199)

```cpp
struct NSG {
    using storage_idx_t = int32_t;
    using Node = nsg::Node;
    using Neighbor = nsg::Neighbor;

    int ntotal = 0;     ///< 节点数量

    // 构建时参数
    int R;  ///< 每个节点的邻居数
    int L;  ///< 构建时搜索路径长度
    int C;  ///< 构建时候选池大小

    // 搜索时参数
    int search_L = 16; ///< 搜索时路径长度

    int enterpoint;     ///< 入口点

    std::shared_ptr<nsg::Graph<int32_t>> final_graph; ///< NSG 图结构
    bool is_built = false;
    RandomGenerator rng;

    explicit NSG(int R = 32);
};
```

**参数设置** (NSG.cpp:111-115):
```cpp
NSG::NSG(int R) : R(R), rng(0x0903) {
    L = R + 32;    // 搜索路径长度
    C = R + 100;   // 候选池大小
    srand(0x1998);
}
```

## 3. 搜索算法

### 3.1 search_on_graph 核心实现 (NSG.cpp:244-330)

NSG 使用基于**标记-扩展**的贪心搜索：

```cpp
template <bool collect_fullset, class index_t>
void NSG::search_on_graph(
        const nsg::Graph<index_t>& graph,
        DistanceComputer& dis,
        VisitedTable& vt,
        int ep,              // 入口点
        int pool_size,       // 候选池大小
        std::vector<Neighbor>& retset,
        std::vector<Node>& fullset) const {
    RandomGenerator gen(0x1234);
    retset.resize(pool_size + 1);  // +1 用于边界检查
    std::vector<int> init_ids(pool_size);

    // 步骤 1: 初始化候选池
    int num_ids = 0;
    std::vector<index_t> neighbors(graph.K);

    // 从入口点的邻居开始
    size_t nneigh = graph.get_neighbors(ep, neighbors.data());
    for (int i = 0; i < init_ids.size() && i < nneigh; i++) {
        int id = (int)neighbors[i];
        if (id >= ntotal) {
            continue;
        }
        init_ids[i] = id;
        vt.set(id);        // 标记为已访问
        num_ids += 1;
    }

    // 随机采样填充剩余位置
    while (num_ids < pool_size) {
        int id = gen.rand_int(ntotal);
        if (vt.get(id)) {
            continue;
        }
        init_ids[num_ids] = id;
        num_ids++;
        vt.set(id);
    }

    // 步骤 2: 计算初始距离并排序
    for (int i = 0; i < init_ids.size(); i++) {
        int id = init_ids[i];
        float dist = dis(id);
        retset[i] = Neighbor(id, dist, true);  // flag = true: 待扩展

        if (collect_fullset) {
            fullset.emplace_back(id, dist);
        }
    }

    std::sort(retset.begin(), retset.begin() + pool_size);

    // 步骤 3: 迭代扩展候选
    int k = 0;
    while (k < pool_size) {
        int updated_pos = pool_size;

        if (retset[k].flag) {
            retset[k].flag = false;  // 标记为已扩展
            int n = retset[k].id;

            // 扩展 n 的邻居
            size_t nneigh_for_n = graph.get_neighbors(n, neighbors.data());
            for (int m = 0; m < nneigh_for_n; m++) {
                int id = neighbors[m];
                if (id > ntotal || vt.get(id)) {
                    continue;
                }
                vt.set(id);

                float dist = dis(id);
                Neighbor nn(id, dist, true);

                if (collect_fullset) {
                    fullset.emplace_back(id, dist);
                }

                // 剪枝：如果距离大于当前最大距离，跳过
                if (dist >= retset[pool_size - 1].distance) {
                    continue;
                }

                // 有序插入到候选池
                int r = insert_into_pool(retset.data(), pool_size, nn);
                updated_pos = std::min(updated_pos, r);
            }
        }

        // 如果有更好的候选被插入，从那个位置重新检查
        k = (updated_pos <= k) ? updated_pos : (k + 1);
    }
}
```

### 3.2 insert_into_pool 有序插入 (NSG.cpp:68-105)

```cpp
inline int insert_into_pool(Neighbor* addr, int K, Neighbor nn) {
    int left = 0, right = K - 1;

    // 比当前最小距离还小
    if (addr[left].distance > nn.distance) {
        memmove(&addr[left + 1], &addr[left], K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }

    // 比当前最大距离还大
    if (addr[right].distance < nn.distance) {
        addr[K] = nn;
        return K;
    }

    // 二分查找插入位置
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance) {
            right = mid;
        } else {
            left = mid;
        }
    }

    // 检查是否已存在（去重）
    while (left > 0) {
        if (addr[left].distance < nn.distance) {
            break;
        }
        if (addr[left].id == nn.id) {
            return K + 1;  // 已存在，返回越界值
        }
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id) {
        return K + 1;
    }

    // 插入到正确位置
    memmove(&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
}
```

**优化要点**:
1. **二分查找**: O(log K) 定位插入位置
2. **去重检查**: 避免同一节点多次进入候选池
3. **memmove**: 高效的批量内存移动

### 3.3 搜索流程图

```
Input: 查询向量 q, 入口点 ep, 候选池大小 pool_size

1. 初始化:
   - 从 ep 的邻居开始填充候选池
   - 随机采样填充剩余位置
   - 计算所有候选的距离
   - 排序候选池

2. 迭代扩展 (k = 0..pool_size-1):
   If retset[k].flag == true:
       - 标记 retset[k].flag = false
       - 扩展 retset[k].id 的所有邻居
       - 对于每个未访问的邻居 n:
           * 计算距离 dis(q, n)
           * 如果 dis < retset[pool_size-1].distance:
               有序插入到候选池
           * 标记为已访问
       - 如果有更好的候选插入，k 回退到插入位置
   Else:
       - k++

3. 返回: retset[0..k-1] 作为 k-NN 结果
```

## 4. 构建算法

### 4.1 build 主流程 (NSG.cpp:138-200)

```cpp
void NSG::build(
        Index* storage,
        idx_t n,
        const nsg::Graph<idx_t>& knn_graph,
        bool verbose) {
    FAISS_THROW_IF_NOT(!is_built && ntotal == 0);

    ntotal = n;

    // 步骤 1: 初始化入口点
    init_graph(storage, knn_graph);

    std::vector<int> degrees(n, 0);
    {
        nsg::Graph<Node> tmp_graph(n, R);

        // 步骤 2: 链接（构建主图）
        link(storage, knn_graph, tmp_graph, verbose);

        // 步骤 3: 转换为最终图格式
        final_graph = std::make_shared<nsg::Graph<int>>(n, R);
        std::fill_n(final_graph->data, n * R, EMPTY_ID);

#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            int cnt = 0;
            for (int j = 0; j < R; j++) {
                int id = tmp_graph.at(i, j).id;
                if (id != EMPTY_ID) {
                    final_graph->at(i, cnt) = id;
                    cnt += 1;
                }
                degrees[i] = cnt;
            }
        }
    }

    // 步骤 4: 确保图连通
    int num_attached = tree_grow(storage, degrees);
    check_graph();
    is_built = true;
}
```

### 4.2 init_graph - 寻找入口点 (NSG.cpp:208-242)

```cpp
void NSG::init_graph(Index* storage, const nsg::Graph<idx_t>& knn_graph) {
    int d = storage->d;
    int n = storage->ntotal;

    // 计算数据中心点
    std::unique_ptr<float[]> center(new float[d]);
    std::unique_ptr<float[]> tmp(new float[d]);
    std::fill_n(center.get(), d, 0.0f);

    for (int i = 0; i < n; i++) {
        storage->reconstruct(i, tmp.get());
        for (int j = 0; j < d; j++) {
            center[j] += tmp[j];
        }
    }
    for (int i = 0; i < d; i++) {
        center[i] /= n;
    }

    // 随机选择初始入口点
    int ep = rng.rand_int(n);
    std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
    dis->set_query(center.get());
    VisitedTable vt(ntotal);

    // 从中心搜索最近的点作为入口点
    std::vector<Neighbor> retset;
    std::vector<Node> tmpset;
    search_on_graph<false>(knn_graph, *dis, vt, ep, L, retset, tmpset);

    // 设置入口点
    enterpoint = retset[0].id;
}
```

**策略**: 选择数据中心点最近的节点作为入口点，确保从图的"中心"开始搜索。

### 4.3 link - 图构建主循环 (NSG.cpp:332-376)

```cpp
void NSG::link(
        Index* storage,
        const nsg::Graph<idx_t>& knn_graph,
        nsg::Graph<Node>& graph,
        bool /* verbose */) {
#pragma omp parallel
    {
        std::unique_ptr<float[]> vec(new float[storage->d]);
        std::vector<Node> pool;
        std::vector<Neighbor> tmp;
        VisitedTable vt(ntotal);
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for schedule(dynamic, 100)
        for (int i = 0; i < ntotal; i++) {
            storage->reconstruct(i, vec.get());
            dis->set_query(vec.get());

            // 搜索收集候选节点（collect_fullset = true）
            search_on_graph<true>(
                    knn_graph, *dis, vt, enterpoint, L, tmp, pool);

            // 同步修剪：选择最优邻居
            sync_prune(i, pool, *dis, vt, knn_graph, graph);

            pool.clear();
            tmp.clear();
            vt.advance();
        }
    } // omp parallel

    // 添加反向链接
    std::vector<std::mutex> locks(ntotal);
#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for schedule(dynamic, 100)
        for (int i = 0; i < ntotal; ++i) {
            add_reverse_links(i, locks, *dis, graph);
        }
    }
}
```

**并行策略**:
1. **第一阶段** (OpenMP parallel): 每个节点独立选择其邻居
2. **第二阶段** (OpenMP parallel + mutex): 添加反向链接，需要锁保护

### 4.4 sync_prune - 遮挡关系修剪 (NSG.cpp:378-432)

```cpp
void NSG::sync_prune(
        int q,
        std::vector<Node>& pool,
        DistanceComputer& dis,
        VisitedTable& vt,
        const nsg::Graph<idx_t>& knn_graph,
        nsg::Graph<Node>& graph) {
    // 步骤 1: 添加 k-NN 图中的邻居
    for (int i = 0; i < knn_graph.K; i++) {
        int id = knn_graph.at(q, i);
        if (id < 0 || id >= ntotal || vt.get(id)) {
            continue;
        }
        float dist = dis.symmetric_dis(q, id);
        pool.emplace_back(id, dist);
    }

    // 步骤 2: 按距离排序
    std::sort(pool.begin(), pool.end());

    // 步骤 3: 遮挡关系修剪
    std::vector<Node> result;

    int start = 0;
    if (pool[start].id == q) {
        start++;  // 跳过自身
    }
    result.push_back(pool[start]);

    while (result.size() < R && (++start) < pool.size() && start < C) {
        auto& p = pool[start];
        bool occlude = false;

        // 检查是否被现有邻居遮挡
        for (int t = 0; t < result.size(); t++) {
            if (p.id == result[t].id) {
                occlude = true;
                break;
            }

            // 关键：遮挡检测
            float djk = dis.symmetric_dis(result[t].id, p.id);
            if (djk < p.distance /* dik */) {
                // result[t] 遮挡了 p
                // 因为 d(result[t], p) < d(q, p)
                // 说明从 result[t] 到 p 比从 q 到 p 更近
                occlude = true;
                break;
            }
        }

        if (!occlude) {
            result.push_back(p);
        }
    }

    // 步骤 4: 写入结果
    for (size_t i = 0; i < R; i++) {
        if (i < result.size()) {
            graph.at(q, i).id = result[i].id;
            graph.at(q, i).distance = result[i].distance;
        } else {
            graph.at(q, i).id = EMPTY_ID;
        }
    }
}
```

**遮挡关系原理**:
```
对于候选邻居 p 和已选邻居 r:
如果 d(r, p) < d(q, p)，则 r 遮挡 p

几何解释:
- 从 r 到 p 的距离小于从 q 到 p
- 意味着经过 r 访问 p 比直接从 q 访问 p 更优
- 因此 p 是冗余的，可以去掉
```

**示例**:
```
q 是查询点
r 和 p 都是候选邻居

如果:
  d(q, r) = 3
  d(q, p) = 5
  d(r, p) = 2

则 r 遮挡 p，因为 d(r, p) < d(q, p)
保留 r，去掉 p
```

### 4.5 add_reverse_links (NSG.cpp:434-512)

```cpp
void NSG::add_reverse_links(
        int q,
        std::vector<std::mutex>& locks,
        DistanceComputer& dis,
        nsg::Graph<Node>& graph) {
    for (size_t i = 0; i < R; i++) {
        if (graph.at(q, i).id == EMPTY_ID) {
            break;
        }

        Node sn(q, graph.at(q, i).distance);
        int des = graph.at(q, i).id;  // 目标节点

        std::vector<Node> tmp_pool;
        int dup = 0;

        // 检查是否已经存在反向链接
        {
            LockGuard guard(locks[des]);
            for (int j = 0; j < R; j++) {
                if (graph.at(des, j).id == EMPTY_ID) {
                    break;
                }
                if (q == graph.at(des, j).id) {
                    dup = 1;
                    break;
                }
                tmp_pool.push_back(graph.at(des, j));
            }
        }

        if (dup) {
            continue;  // 已存在反向链接
        }

        // 添加 q 到 des 的邻居列表
        tmp_pool.push_back(sn);

        if (tmp_pool.size() > R) {
            // 需要修剪，重新选择最优 R 个邻居
            std::vector<Node> result;
            int start = 0;
            std::sort(tmp_pool.begin(), tmp_pool.end());
            result.push_back(tmp_pool[start]);

            while (result.size() < R && (++start) < tmp_pool.size()) {
                auto& p = tmp_pool[start];
                bool occlude = false;

                for (int t = 0; t < result.size(); t++) {
                    if (p.id == result[t].id) {
                        occlude = true;
                        break;
                    }
                    float djk = dis.symmetric_dis(result[t].id, p.id);
                    if (djk < p.distance) {
                        occlude = true;
                        break;
                    }
                }

                if (!occlude) {
                    result.push_back(p);
                }
            }

            {
                LockGuard guard(locks[des]);
                for (int t = 0; t < result.size(); t++) {
                    graph.at(des, t) = result[t];
                }
            }
        } else {
            // 直接添加
            LockGuard guard(locks[des]);
            for (int t = 0; t < R; t++) {
                if (graph.at(des, t).id == EMPTY_ID) {
                    graph.at(des, t) = sn;
                    break;
                }
            }
        }
    }
}
```

**关键点**:
1. **细粒度锁**: 每个节点一把锁，减少竞争
2. **双向修剪**: 反向链接也需要应用遮挡关系
3. **去重**: 检查反向链接是否已存在

### 4.6 tree_grow - 确保连通性 (NSG.cpp:514-533)

```cpp
int NSG::tree_grow(Index* storage, std::vector<int>& degrees) {
    int root = enterpoint;
    VisitedTable vt(ntotal);   // 用于 DFS
    VisitedTable vt2(ntotal);  // 用于搜索最近的连通节点

    int num_attached = 0;
    int cnt = 0;

    while (true) {
        // 计算当前连通分量大小
        cnt = dfs(vt, root, cnt);
        if (cnt >= ntotal) {
            break;  // 全部连通
        }

        // 附加未连通的节点
        root = attach_unlinked(storage, vt, vt2, degrees);
        vt2.advance();
        num_attached += 1;
    }

    return num_attached;
}
```

### 4.7 dfs - 深度优先搜索 (NSG.cpp:535-570)

```cpp
int NSG::dfs(VisitedTable& vt, int root, int cnt) const {
    int node = root;
    std::stack<int> stack;
    stack.push(root);

    if (!vt.get(root)) {
        cnt++;
    }
    vt.set(root);

    while (!stack.empty()) {
        int next = EMPTY_ID;

        // 找第一个未访问的邻居
        for (int i = 0; i < R; i++) {
            int id = final_graph->at(node, i);
            if (id != EMPTY_ID && !vt.get(id)) {
                next = id;
                break;
            }
        }

        if (next == EMPTY_ID) {
            // 没有未访问的邻居，回溯
            stack.pop();
            if (stack.empty()) {
                break;
            }
            node = stack.top();
            continue;
        }

        // 访问 next
        node = next;
        vt.set(node);
        stack.push(node);
        cnt++;
    }

    return cnt;
}
```

### 4.8 attach_unlinked (NSG.cpp:572-639)

```cpp
int NSG::attach_unlinked(
        Index* storage,
        VisitedTable& vt,
        VisitedTable& vt2,
        std::vector<int>& degrees) {
    // 找一个未连通的节点
    int id = EMPTY_ID;
    for (int i = 0; i < ntotal; i++) {
        if (!vt.get(i)) {
            id = i;
            break;
        }
    }

    if (id == EMPTY_ID) {
        return EMPTY_ID;  // 没有未连通节点
    }

    std::vector<Neighbor> tmp;
    std::vector<Node> pool;
    std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
    std::unique_ptr<float[]> vec(new float[storage->d]);

    storage->reconstruct(id, vec.get());
    dis->set_query(vec.get());

    // 搜索最近的节点
    search_on_graph<true>(
            *final_graph, *dis, vt2, enterpoint, search_L, tmp, pool);

    std::sort(pool.begin(), pool.end());

    // 找度数 < R 的最近节点
    int node;
    bool found = false;
    for (int i = 0; i < pool.size(); i++) {
        node = pool[i].id;
        if (degrees[node] < R && node != id) {
            found = true;
            break;
        }
    }

    // 如果没找到，随机选择
    if (!found) {
        do {
            node = rng.rand_int(ntotal);
            if (vt.get(node) && degrees[node] < R && node != id) {
                found = true;
            }
        } while (!found);
    }

    // 连接到选定节点
    int pos = degrees[node];
    final_graph->at(node, pos) = id;
    degrees[node] += 1;

    return node;
}
```

**优化要点**:
1. **度数约束**: 只连接到度数 < R 的节点，避免度数爆炸
2. **就近原则**: 优先选择最近的节点连接
3. **随机后备**: 如果没有合适的节点，随机选择

## 5. NSG vs HNSW 对比

| 特性 | NSG | HNSW |
|------|-----|------|
| **图结构** | 单层 | 多层 |
| **构建基础** | k-NN 图 | 从头构建 |
| **邻居选择** | 遮挡关系修剪 | 概率 + 修剪 |
| **搜索策略** | flag 标记扩展 | 双堆（候选 + 结果） |
| **复杂度** | O(n × C × R) | O(n × log(n) × M × R) |
| **精度** | 接近 HNSW | 略高于 NSG |
| **构建速度** | 更快 | 较慢 |

## 6. 并行优化

### 6.1 动态调度 (NSG.cpp:348, 371)

```cpp
#pragma omp for schedule(dynamic, 100)
for (int i = 0; i < ntotal; i++) {
    // ...
}
```

**动态调度优势**:
1. **负载均衡**: 不同节点的处理时间差异大
2. **避免空闲**: 快线程不会空闲等待
3. **chunk size = 100**: 平衡调度开销和负载均衡

### 6.2 细粒度锁 (NSG.cpp:365, 449, 496)

```cpp
std::vector<std::mutex> locks(ntotal);

{
    LockGuard guard(locks[des]);
    // 临界区：修改节点 des 的邻居
}
```

**锁粒度设计**:
- **每个节点一把锁**: 最小化竞争
- **RAII**: 自动释放锁
- **临界区最小化**: 只在必要时持有锁

## 7. 内存优化

### 7.1 图存储格式

```cpp
// 扁平化的邻接矩阵
int32_t* data;  // 大小 N × R

// 访问节点 i 的第 j 个邻居
node_t neighbor = data[i * R + j];

// 支持稀疏存储（-1 表示空）
if (data[i * R + j] == EMPTY_ID) {
    break;  // 后续邻居都是空的
}
```

**优势**:
1. **缓存友好**: 连续内存访问
2. **无指针开销**: 无额外指针存储
3. **SIMD 友好**: 可向量化加载

### 7.2 VisitedTable 复用

```cpp
VisitedTable vt(ntotal);
// ... 使用 vt ...
vt.advance();  // 重置但保留内存
```

**避免重复分配**: 在循环中复用 VisitedTable

## 8. 性能特征

### 8.1 时间复杂度

| 阶段 | 复杂度 | 说明 |
|------|--------|------|
| **init_graph** | O(n × L × d) | 中心点搜索 |
| **link - 搜索** | O(n × L × d) | 每个节点搜索 |
| **sync_prune** | O(n × C × R) | 遮挡检测 |
| **add_reverse_links** | O(n × R²) | 反向链接 |
| **tree_grow** | O(n × R) | 连通性检查 |
| **总构建** | O(n × (L + R) × d + n × R²) | |

### 8.2 空间复杂度

| 数据结构 | 大小 |
|----------|------|
| **Graph** | n × R × sizeof(int32_t) |
| **VisitedTable** | n / 8 bytes |
| **临时池** | C × sizeof(Node) |

### 8.3 参数调优建议

| 参数 | 推荐值 | 影响 |
|------|-------|------|
| R | 32-64 | 更大 = 更高精度，更高内存 |
| L | R + 32 | 搜索路径长度 |
| C | R + 100 | 候选池大小 |
| search_L | 16-32 | 搜索路径长度 |

## 9. 优化技术总结

### 9.1 算法级优化

| 技术 | 实现 | 收益 |
|------|------|------|
| **遮挡关系** | sync_prune | 减少冗余边 |
| **双向链接** | add_reverse_links | 提高召回率 |
| **就近入口点** | init_graph | 更好的搜索起点 |
| **连通性保证** | tree_grow | 100% 召回率 |

### 9.2 实现级优化

| 技术 | 实现 | 收益 |
|------|------|------|
| **有序插入** | insert_into_pool | O(log K) |
| **细粒度锁** | std::mutex per node | 低竞争 |
| **动态调度** | schedule(dynamic) | 负载均衡 |
| **内存复用** | vt.advance() | 减少分配 |

### 9.3 内存访问优化

| 技术 | 实现 | 收益 |
|------|------|------|
| **扁平化存储** | N×R 数组 | 缓存友好 |
| **局部性原理** | 邻居连续存储 | 预取友好 |
| **标志位** | bool flag | 分支预测 |

## 10. 关键源码位置

| 文件 | 函数 | 行号 |
|------|------|------|
| `NSG.cpp` | `search_on_graph()` | 244-330 |
| `NSG.cpp` | `insert_into_pool()` | 68-105 |
| `NSG.cpp` | `sync_prune()` | 378-432 |
| `NSG.cpp` | `add_reverse_links()` | 434-512 |
| `NSG.cpp` | `tree_grow()` | 514-533 |
| `NSG.cpp` | `dfs()` | 535-570 |
| `NSG.cpp` | `attach_unlinked()` | 572-639 |
| `NSG.h` | `Graph` 结构 | 53-102 |
| `NSG.h` | `NSG` 主类 | 108-199 |
