# Faiss深度课程 - 第8天：图索引 - NSG与NNDescent

## 课程目标

理解NSG（Navigating Spreading-out Graph）和NN-Descent算法，这些是HNSW之外的高效图索引方法。

---

## 1. NSG概述

### 1.1 核心思想

NSG通过构建单调性属性的有向图实现高效搜索：

```cpp
// NSG特点：
// 1. 单层图（非层次结构）
// 2. 有向边
// 3. 满足单调性：从导航点出发，距离单调递减
// 4. 稀疏：每个节点度数受限

导航点 (Navigator)
   │
   ├─→ [节点1] ─→ [节点5] ─→ ...
   │    ↘
   ├─→ [节点2] ─→ [节点6]
   │    ↘ ↘
   └─→ [节点3] ─→ [节点7]
        ↘
         [节点4]
```

### 1.2 与HNSW对比

| 特性 | HNSW | NSG |
|------|------|-----|
| 层次 | 多层 | 单层 |
| 边类型 | 无向 | 有向 |
| 构建复杂度 | O(n log n) | O(n log n) |
| 内存 | 较高 | 较低 |
| 查询速度 | 快 | 稍慢 |

---

## 2. NSG数据结构

```cpp
// NSG核心结构（简化）
struct NSG {
    int N;              // 节点数
    int d;              // 维度
    int R;              // 最大度数
    int L;              // 构建时的候选数

    const float* data;  // 原始数据

    // 邻居表
    std::vector<std::vector<int>> neighbors;

    // 导航点
    int navigator;

    // 完整图标记（用于构建）
    std::vector<bool> is_full_graph;

    NSG(int N, int d, int R, int L)
        : N(N), d(d), R(R), L(L),
          neighbors(N), is_full_graph(N, false) {}
};
```

---

## 3. NSG构建

### 3.1 NN-Descent初始化

```cpp
// 使用NN-Descent构建初始kNN图
std::vector<std::vector<int>> nndescent_knn(
        const float* data,
        int N, int d,
        int K) {

    // 1. 随机初始化
    std::vector<std::vector<int>> graph(N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            int rand_nb = rand() % N;
            if (rand_nb != i) {
                graph[i].push_back(rand_nb);
            }
        }
    }

    // 2. 迭代优化
    for (int iter = 0; iter < 10; iter++) {
        // 对于每个节点
        for (int i = 0; i < N; i++) {
            // 采样邻居的邻居
            std::vector<int> candidates;

            for (int nb : graph[i]) {
                for (int nn : graph[nb]) {
                    if (nn != i) {
                        candidates.push_back(nn);
                    }
                }
            }

            // 计算距离，更新KNN
            std::vector<std::pair<float, int>> dists;

            for (int c : candidates) {
                float dis = fvec_L2sqr(
                    data + i * d,
                    data + c * d,
                    d);
                dists.push_back({dis, c});
            }

            std::sort(dists.begin(), dists.end());

            graph[i].clear();
            for (int j = 0; j < std::min(K, (int)dists.size()); j++) {
                graph[i].push_back(dists[j].second);
            }
        }
    }

    return graph;
}
```

### 3.2 NSG构建主流程

```cpp
void build_nsg(NSG& nsg) {
    // 1. 使用NN-Descent构建初始图
    auto knn_graph = nndescent_knn(
        nsg.data, nsg.N, nsg.d, nsg.R);

    // 2. 寻找导航点（质心）
    nsg.navigator = find_navigator(nsg.data, nsg.N, nsg.d);

    // 3. 构建完整图（以导航点为根）
    build_full_graph(nsg, knn_graph);

    // 4. 裁剪边，满足单调性
    prune_edges(nsg);
}

int find_navigator(const float* data, int N, int d) {
    // 找到离质心最近的点作为导航点
    std::vector<float> centroid(d, 0.0f);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            centroid[j] += data[i * d + j];
        }
    }

    for (int j = 0; j < d; j++) {
        centroid[j] /= N;
    }

    int navigator = 0;
    float min_dis = fvec_L2sqr(data, centroid.data(), d);

    for (int i = 1; i < N; i++) {
        float dis = fvec_L2sqr(data + i * d, centroid.data(), d);
        if (dis < min_dis) {
            min_dis = dis;
            navigator = i;
        }
    }

    return navigator;
}
```

### 3.3 构建完整图

```cpp
void build_full_graph(NSG& nsg,
                       const std::vector<std::vector<int>>& knn_graph) {

    // 从导航点开始BFS
    std::queue<int> queue;
    std::vector<bool> visited(nsg.N, false);

    queue.push(nsg.navigator);
    visited[nsg.navigator] = true;

    while (!queue.empty()) {
        int u = queue.front();
        queue.pop();

        // 标记为完整图节点
        nsg.is_full_graph[u] = true;

        // 添加邻居
        for (int v : knn_graph[u]) {
            nsg.neighbors[u].push_back(v);

            if (!visited[v]) {
                visited[v] = true;
                queue.push(v);
            }
        }
    }

    // 确保所有节点都被访问
    for (int i = 0; i < nsg.N; i++) {
        if (!visited[i]) {
            // 孤立节点，连接到最近节点
            int nearest = find_nearest(nsg.data, nsg.d, i,
                                       visited.data(), nsg.N);
            nsg.neighbors[i].push_back(nearest);
            nsg.neighbors[nearest].push_back(i);
        }
    }
}
```

### 3.4 边裁剪

```cpp
void prune_edges(NSG& nsg) {
    for (int i = 0; i < nsg.N; i++) {
        if (nsg.is_full_graph[i]) continue;

        // 非完整图节点，裁剪边
        auto& nb = nsg.neighbors[i];

        // 按距离排序
        std::vector<std::pair<float, int>> dists;
        for (int j : nb) {
            float dis = fvec_L2sqr(
                nsg.data + i * nsg.d,
                nsg.data + j * nsg.d,
                nsg.d);
            dists.push_back({dis, j});
        }

        std::sort(dists.begin(), dists.end());

        // 只保留最近的R个
        nb.clear();
        for (int j = 0; j < std::min(nsg.R, (int)dists.size()); j++) {
            nb.push_back(dists[j].second);
        }
    }
}
```

---

## 4. NSG搜索

### 4.1 贪婪搜索

```cpp
void search_nsg(
        const NSG& nsg,
        const float* query,
        int K,
        float* distances,
        int* labels) {

    // 初始化结果堆
    std::priority_queue<
        std::pair<float, int>,
        std::vector<std::pair<float, int>>,
        std::greater<>> result;  // 最小堆

    std::vector<bool> visited(nsg.N, false);

    // 1. 从导航点开始
    int current = nsg.navigator;
    float current_dis = fvec_L2sqr(
        query, nsg.data + current * nsg.d, nsg.d);

    // 2. 贪婪遍历
    while (true) {
        visited[current] = true;

        // 更新结果
        if (result.size() < K) {
            result.push({current_dis, current});
        } else if (current_dis < result.top().first) {
            result.pop();
            result.push({current_dis, current});
        }

        // 找到未访问的最优邻居
        float best_dis = std::numeric_limits<float>::infinity();
        int best_nb = -1;

        for (int nb : nsg.neighbors[current]) {
            if (visited[nb]) continue;

            float dis = fvec_L2sqr(
                query, nsg.data + nb * nsg.d, nsg.d);

            if (dis < best_dis) {
                best_dis = dis;
                best_nb = nb;
            }
        }

        if (best_nb == -1 || best_dis > current_dis) {
            break;  // 局部最优
        }

        current = best_nb;
        current_dis = best_dis;
    }

    // 3. 提取结果
    for (int i = K - 1; i >= 0; i--) {
        labels[i] = result.top().second;
        distances[i] = result.top().first;
        result.pop();
    }
}
```

---

## 5. NN-Descent算法

### 5.1 算法思想

NN-Descent通过迭代优化近似KNN图：

```cpp
// 核心思想：
// 对于每个节点，其最近邻很可能也在其邻居的邻居中
//
// 迭代：
// repeat:
//   for each node i:
//     sample = 邻居的邻居
//     new_knn = K个最近的sample
//     if new_knn比当前knn更好:
//       更新knn
// until 收敛
```

### 5.2 详细实现

```cpp
struct NNDescent {
    int N;              // 节点数
    int d;              // 维度
    int K;              // K值
    int S;              // 采样率
    const float* data;

    // 当前图
    std::vector<std::vector<int>> graph;

    // 反向图（用于优化）
    std::vector<std::vector<int>> reverse_graph;

    NNDescent(int N, int d, int K, int S, const float* data)
        : N(N), d(d), K(K), S(S), data(data),
          graph(N), reverse_graph(N) {}

    void build() {
        // 1. 随机初始化
        random_init();

        // 2. 迭代更新
        for (int iter = 0; iter < 10; iter++) {
            // 更新反向图
            update_reverse_graph();

            // 采样邻居的邻居
            std::vector<std::vector<int>> candidates(N);

            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                // 当前邻居
                for (int nb : graph[i]) {
                    // 邻居的邻居
                    for (int nn : graph[nb]) {
                        if (nn != i) {
                            candidates[i].push_back(nn);
                        }
                    }
                }

                // 反向邻居
                for (int nb : reverse_graph[i]) {
                    candidates[i].push_back(nb);
                }
            }

            // 更新KNN
            update_knn(candidates);

            // 检查收敛
            if (check_convergence()) {
                break;
            }
        }
    }

    void random_init() {
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            // 随机选择K个邻居
            std::vector<int> indices;
            for (int j = 0; j < N; j++) {
                if (j != i) {
                    indices.push_back(j);
                }
            }

            std::shuffle(indices.begin(), indices.end(), rng);

            for (int j = 0; j < std::min(K, (int)indices.size()); j++) {
                graph[i].push_back(indices[j]);
            }
        }
    }

    void update_reverse_graph() {
        for (int i = 0; i < N; i++) {
            reverse_graph[i].clear();
        }

        for (int i = 0; i < N; i++) {
            for (int nb : graph[i]) {
                reverse_graph[nb].push_back(i);
            }
        }
    }

    void update_knn(const std::vector<std::vector<int>>& candidates) {
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            // 采样
            std::vector<int> sample;
            for (int c : candidates[i]) {
                if ((rand() % 100) < S) {
                    sample.push_back(c);
                }
            }

            // 计算距离
            std::vector<std::pair<float, int>> dists;

            for (int j : sample) {
                float dis = fvec_L2sqr(
                    data + i * d,
                    data + j * d,
                    d);
                dists.push_back({dis, j});
            }

            // 排序并更新
            std::sort(dists.begin(), dists.end());

            graph[i].clear();
            for (int j = 0; j < std::min(K, (int)dists.size()); j++) {
                graph[i].push_back(dists[j].second);
            }
        }
    }

    bool check_convergence() {
        // 检查更新率
        int changes = 0;

        // 需要维护旧图进行比较
        // 简化版本：假设固定迭代次数
        return false;
    }
};
```

---

## 6. 性能优化

### 6.1 批量距离计算

```cpp
// SIMD优化的批量距离计算
void batch_compute_distances(
        const float* query,
        const std::vector<int>& candidates,
        const float* data,
        int d,
        std::vector<float>& distances) {

    size_t n = candidates.size();
    distances.resize(n);

    size_t i = 0;

    // 处理4的倍数
    for (; i + 4 <= n; i += 4) {
        const float* v0 = data + candidates[i + 0] * d;
        const float* v1 = data + candidates[i + 1] * d;
        const float* v2 = data + candidates[i + 2] * d;
        const float* v3 = data + candidates[i + 3] * d;

        fvec_L2sqr_batch_4(query, v0, v1, v2, v3, d,
                          distances[i + 0],
                          distances[i + 1],
                          distances[i + 2],
                          distances[i + 3]);
    }

    // 处理剩余
    for (; i < n; i++) {
        distances[i] = fvec_L2sqr(
            query,
            data + candidates[i] * d,
            d);
    }
}
```

### 6.2 并行构建

```cpp
void parallel_build_nsg(NSG& nsg) {
    // 1. 并行计算KNN
    std::vector<std::vector<std::pair<float, int>>> all_knns(nsg.N);

    #pragma omp parallel for
    for (int i = 0; i < nsg.N; i++) {
        std::vector<std::pair<float, int>> knns;

        for (int j = 0; j < nsg.N; j++) {
            if (i == j) continue;

            float dis = fvec_L2sqr(
                nsg.data + i * nsg.d,
                nsg.data + j * nsg.d,
                nsg.d);

            knns.push_back({dis, j});
        }

        std::sort(knns.begin(), knns.end());
        all_knns[i] = knns;
    }

    // 2. 构建图
    for (int i = 0; i < nsg.N; i++) {
        for (int j = 0; j < std::min(nsg.R, nsg.N); j++) {
            nsg.neighbors[i].push_back(all_knns[i][j].second);
        }
    }

    // 3. 后处理
    build_full_graph(nsg, all_knns);
    prune_edges(nsg);
}
```

---

## 7. 实践建议

### 7.1 参数选择

```cpp
// NSG参数
NSG nsg(N, d,
        R = 32,    // 最大度数
        L = 50);   // 候选数

// R越大：
// - 精度越高
// - 内存越大
// - 搜索越慢

// L越大：
// - 构建质量越高
// - 构建时间越长
```

### 7.2 与IVF结合

```cpp
// IVF + NSG组合
// 1. IVF粗量化
// 2. 每个倒排列表内使用NSG

Index* quantizer = new IndexFlatL2(d);
IndexIVFFlat index(quantizer, d, nlist);
index.train(n, xb);
index.add(n, xb);

// 为每个列表构建NSG
for (size_t list_no = 0; list_no < nlist; list_no++) {
    size_t list_size = index.invlists->list_size(list_no);

    if (list_size > 1000) {  // 只对大列表构建NSG
        // 提取向量
        // 构建NSG
        // 存储
    }
}
```

---

## 8. NSG与NNDescent底层实现详解

### 8.1 NSG完整结构

```cpp
// faiss/impl/NSG.h
struct NSG {
    using storage_idx_t = int32_t;  // 内部存储索引类型

    int ntotal = 0;      // 节点总数

    // 构建时参数
    int R;               // 每个节点的邻居数
    int L;               // 构建时搜索路径长度
    int C;               // 构建时候选池大小

    // 搜索时参数
    int search_L = 16;   // 搜索时路径长度

    int enterpoint;      // 入口点（导航点）

    std::shared_ptr<nsg::Graph<int32_t>> final_graph;  // NSG图结构

    bool is_built = false;  // NSG是否已构建

    RandomGenerator rng;    // 随机数生成器

    explicit NSG(int R = 32);

    // 从KNN图构建NSG
    void build(
            Index* storage,
            idx_t n,
            const nsg::Graph<idx_t>& knn_graph,
            bool verbose);

    void reset();

    // 搜索接口
    void search(
            DistanceComputer& dis,
            int k,
            idx_t* I,
            float* D,
            VisitedTable& vt) const;
};
```

### 8.2 Graph结构

```cpp
// faiss/impl/NSG.h
namespace nsg {

// 图结构：用邻接矩阵表示
template <class node_t>
struct Graph {
    node_t* data;        // 扁平化的邻接矩阵，大小为 N×K
    int K;               // 每个节点的邻居数
    int N;               // 总节点数
    bool own_fields;     // 是否拥有底层数据

    // 从已知图构造
    Graph(node_t* data, int N, int K)
        : data(data), K(K), N(N), own_fields(false) {}

    // 构造空图
    Graph(int N, int K) : K(K), N(N), own_fields(true) {
        data = new node_t[N * K];
    }

    // 释放内存
    virtual ~Graph() {
        if (own_fields) {
            delete[] data;
        }
    }

    // 访问节点i的第j个邻居
    inline node_t at(int i, int j) const {
        return data[i * K + j];
    }

    // 获取节点i的所有邻居
    virtual size_t get_neighbors(int i, node_t* neighbors) const {
        for (int j = 0; j < K; j++) {
            if (data[i * K + j] < 0) {
                return j;  // 遇到-1表示邻居结束
            }
            neighbors[j] = data[i * K + j];
        }
        return K;
    }
};

} // namespace nsg
```

### 8.3 NNDescent核心结构

```cpp
// faiss/impl/NNDescent.h
struct NNDescent {
    int d;              // 向量维度
    int K;              // 邻居数
    int L;              // 候选池大小 (K + 50)
    int S;              // 采样大小
    int iter;           // 迭代次数
    int search_L;       // 搜索时路径长度

    idx_t ntotal;       // 总向量数

    std::vector<nndescent::Nhood> graph;  // 图结构

    NNDescent(const int d, const int K);

    // 构建NNDescent图
    void build(DistanceComputer& qdis, const Graph<idx_t>& knn_graph);

    // 搜索接口
    void search(
            DistanceComputer& qdis,
            idx_t k,
            idx_t* I,
            float* D) const;
};
```

### 8.4 Nhood（邻域）结构

```cpp
// faiss/impl/NNDescent.h
namespace nndescent {

// 邻居节点
struct Neighbor {
    int id;             // 节点ID
    float distance;     // 距离
    bool flag;          // 是否为新的邻居（用于迭代优化）

    Neighbor(int id, float distance, bool flag)
        : id(id), distance(distance), flag(flag) {}

    // 用于堆排序
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

// 节点的邻域
struct Nhood {
    int M;                              // 最大邻居数
    std::vector<int> nn_new;            // 新邻居列表
    std::vector<int> nn_old;            // 旧邻居列表
    std::vector<Neighbor> pool;         // 候选池
    std::mutex lock;                    // 用于并行构建的锁

    Nhood(int l, int s, std::mt19937& rng, int N) {
        M = s;
        nn_new.resize(s * 2);
        // 随机初始化
        gen_random(rng, nn_new.data(), (int)nn_new.size(), N);
    }

    // 将候选插入池中
    void insert(int id, float dist) {
        std::lock_guard<std::mutex> guard(lock);
        if (dist > pool.front().distance) {
            return;  // 比最远的候选还远
        }
        // 检查是否已存在
        for (int i = 0; i < pool.size(); i++) {
            if (id == pool[i].id) {
                return;
            }
        }
        // 插入到池中
        if (pool.size() < pool.capacity()) {
            pool.push_back(Neighbor(id, dist, true));
            std::push_heap(pool.begin(), pool.end());
        } else {
            std::pop_heap(pool.begin(), pool.end());
            pool[pool.size() - 1] = Neighbor(id, dist, true);
            std::push_heap(pool.begin(), pool.end());
        }
    }

    // 局部连接：在本地join中，只有至少一个对象是新的时候才比较两个对象
    template <typename C>
    void join(C callback) const {
        for (int const i : nn_new) {
            for (int const j : nn_new) {
                if (i < j) {
                    callback(i, j);
                }
            }
            for (int j : nn_old) {
                callback(i, j);
            }
        }
    }
};

} // namespace nndescent
```

### 8.5 NNDescent::join实现

```cpp
// faiss/impl/NNDescent.cpp
// 局部join操作
void NNDescent::join(DistanceComputer& qdis) {
    idx_t check_period = InterruptCallback::get_period_hint(d * search_L);

    for (idx_t i0 = 0; i0 < (idx_t)ntotal; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, (idx_t)ntotal);

#pragma omp parallel for default(shared) schedule(dynamic, 100)
        for (idx_t n = i0; n < i1; n++) {
            // 对节点n的邻域执行join
            graph[n].join([&](int i, int j) {
                if (i != j) {
                    // 计算对称距离
                    float dist = qdis.symmetric_dis(i, j);
                    // 将候选添加到双方的池中
                    graph[i].insert(j, dist);
                    graph[j].insert(i, dist);
                }
            });
        }

        InterruptCallback::check();
    }
}
```

### 8.6 NNDescent::update实现

```cpp
// faiss/impl/NNDescent.cpp
// 更新邻域：为每个节点采样新/旧邻居
void NNDescent::update() {
    // Step 1: 清空所有nn_new和nn_old
#pragma omp parallel for
    for (int i = 0; i < ntotal; i++) {
        std::vector<int>().swap(graph[i].nn_new);
        std::vector<int>().swap(graph[i].nn_old);
    }

    // Step 2: 计算新邻居的数量
#pragma omp parallel for
    for (int n = 0; n < ntotal; ++n) {
        auto& nn = graph[n];
        std::sort(nn.pool.begin(), nn.pool.end());

        if (nn.pool.size() > L) {
            nn.pool.resize(L);
        }

        // 统计flag为true的数量
        int new_nd = std::count_if(
                nn.pool.begin(), nn.pool.end(),
                [](const Neighbor& n) { return n.flag; });

        int max_new_nd = std::min(S, (int)nn.pool.size());
        new_nd = std::min(new_nd, max_new_nd);

        // 随机选择新邻居
        std::sample(
                nn.pool.begin(), nn.pool.end(),
                std::back_inserter(nn.nn_new),
                new_nd,
                std::mt19937(std::random_device()()));

        // 选择旧邻居
        for (const auto& neighbor : nn.pool) {
            if (!neighbor.flag &&
                nn.nn_new.size() + nn.nn_old.size() < (size_t)S) {
                nn.nn_old.push_back(neighbor.id);
            }
        }

        // 清空池
        nn.pool.clear();
    }
}
```

### 8.7 NSG::search实现

```cpp
// faiss/impl/NSG.cpp (简化版)
void NSG::search(
        DistanceComputer& dis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt) const {

    if (enterpoint == -1) {
        return;  // 图未构建
    }

    // 初始化结果堆
    std::priority_queue<std::pair<float, idx_t>> results;
    std::priority_queue<std::pair<float, idx_t>, ...> candidates;

    // 从入口点开始
    float d_entry = dis(enterpoint);
    candidates.push({d_entry, enterpoint});
    results.push({d_entry, enterpoint});
    vt.set(enterpoint);

    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        // 如果当前候选比结果中最远的还远，停止
        if (current.first > results.top().first) {
            break;
        }

        idx_t node = current.second;

        // 获取邻居
        std::vector<int32_t> neighbors(R);
        size_t n_neighbors = final_graph->get_neighbors(node, neighbors.data());

        // 访问所有邻居
        for (size_t i = 0; i < n_neighbors; i++) {
            idx_t neighbor = neighbors[i];

            if (vt.get(neighbor)) {
                continue;  // 已访问
            }
            vt.set(neighbor);

            float d = dis(neighbor);

            // 更新候选集和结果集
            if (results.size() < (size_t)k || d < results.top().first) {
                candidates.push({d, neighbor});
                results.push({d, neighbor});

                if (results.size() > (size_t)k) {
                    results.pop();
                }
            }
        }
    }

    // 提取结果
    for (int i = k - 1; i >= 0; i--) {
        I[i] = results.top().second;
        D[i] = results.top().first;
        results.pop();
    }
}
```

### 8.8 NSG::tree_grow - 计算中心点

```cpp
// faiss/impl/NSG.cpp (简化版)
// 使NSG完全连通：计算中心点作为导航点
int NSG::tree_grow(Index* storage, std::vector<int>& degrees) {
    // 1. 找到度数最小的节点作为候选
    int min_degree = ntotal;
    int root = -1;

    for (int i = 0; i < ntotal; i++) {
        if (degrees[i] < min_degree) {
            min_degree = degrees[i];
            root = i;
        }
    }

    // 2. 检查连通性
    VisitedTable vt(ntotal);
    int n_connected = dfs(vt, root, 0);

    // 3. 如果不连通，添加边连接孤立节点
    while (n_connected < ntotal) {
        for (int i = 0; i < ntotal; i++) {
            if (!vt.get(i)) {
                // 找到孤立节点，连接到root
                final_graph->at(root, degrees[root]++) = i;
                n_connected = dfs(vt, root, 0);
                break;
            }
        }
    }

    enterpoint = root;
    return root;
}

// 深度优先搜索：计算连通分量大小
int NSG::dfs(VisitedTable& vt, int root, int cnt) const {
    vt.set(root);
    cnt++;

    std::vector<int32_t> neighbors(R);
    size_t n_neighbors = final_graph->get_neighbors(root, neighbors.data());

    for (size_t i = 0; i < n_neighbors; i++) {
        int neighbor = neighbors[i];
        if (!vt.get(neighbor)) {
            cnt = dfs(vt, neighbor, cnt);
        }
    }

    return cnt;
}
```

### 8.9 sync_prune - 同步剪枝

```cpp
// faiss/impl/NSG.cpp (简化版)
// 同步剪枝：裁剪邻居列表以保持单调性和度数限制
void NSG::sync_prune(
        int q,
        std::vector<Node>& pool,
        DistanceComputer& dis,
        VisitedTable& vt,
        const nsg::Graph<idx_t>& knn_graph,
        nsg::Graph<Node>& graph) {

    // 1. 按距离排序
    std::sort(pool.begin(), pool.end());

    // 2. 裁剪以保持单调性
    std::vector<Node> selected;
    for (const auto& node : pool) {
        bool good = true;

        // 检查是否与已选节点太近
        for (const auto& s : selected) {
            float d = dis.symmetric_dis(node.id, s.id);
            if (d < node.distance) {
                good = false;
                break;
            }
        }

        if (good) {
            selected.push_back(node);
            if (selected.size() >= (size_t)R) {
                break;
            }
        }
    }

    // 3. 更新邻居表
    for (size_t i = 0; i < selected.size(); i++) {
        graph.at(q, i) = selected[i];
    }
}
```

---

## 10. NSG与NNDescent源码深度实现

### 10.1 NSG::search_on_graph - 核心搜索实现

```cpp
// faiss/impl/NSG.cpp
// 在NSG图上执行贪婪搜索
template <bool collect_fullset, class index_t>
void NSG::search_on_graph(
        const nsg::Graph<index_t>& graph,
        DistanceComputer& dis,
        VisitedTable& vt,
        int ep,
        int pool_size,
        std::vector<Neighbor>& retset,
        std::vector<Node>& fullset) const {

    RandomGenerator gen(0x1234);
    retset.resize(pool_size + 1);
    std::vector<int> init_ids(pool_size);

    // 1. 初始化候选集：从入口点的邻居开始
    int num_ids = 0;
    std::vector<index_t> neighbors(graph.K);
    size_t nneigh = graph.get_neighbors(ep, neighbors.data());

    for (int i = 0; i < init_ids.size() && i < nneigh; i++) {
        int id = (int)neighbors[i];
        if (id >= ntotal) {
            continue;
        }
        init_ids[i] = id;
        vt.set(id);
        num_ids += 1;
    }

    // 2. 填充候选集到pool_size（随机选择未访问节点）
    while (num_ids < pool_size) {
        int id = gen.rand_int(ntotal);
        if (vt.get(id)) {
            continue;
        }
        init_ids[num_ids] = id;
        num_ids++;
        vt.set(id);
    }

    // 3. 计算初始候选集的距离
    for (int i = 0; i < init_ids.size(); i++) {
        int id = init_ids[i];
        float dist = dis(id);
        retset[i] = Neighbor(id, dist, true);

        if (collect_fullset) {
            fullset.emplace_back(retset[i].id, retset[i].distance);
        }
    }

    // 4. 按距离排序
    std::sort(retset.begin(), retset.begin() + pool_size);

    // 5. 贪婪扩展
    int k = 0;
    while (k < pool_size) {
        int updated_pos = pool_size;

        if (retset[k].flag) {  // flag=true表示是新节点
            retset[k].flag = false;
            int n = retset[k].id;

            // 遍历节点n的邻居
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

                // 如果距离太远，跳过
                if (dist >= retset[pool_size - 1].distance) {
                    continue;
                }

                // 插入到候选池（保持有序）
                int r = insert_into_pool(retset.data(), pool_size, nn);
                updated_pos = std::min(updated_pos, r);
            }
        }

        // 如果没有更新或已到达末尾，继续下一个
        k = (updated_pos <= k) ? updated_pos : (k + 1);
    }
}
```

### 10.2 insert_into_pool - 二分插入

```cpp
// faiss/impl/NSG.cpp
// 使用二分查找将邻居插入有序池中
inline int insert_into_pool(Neighbor* addr, int K, Neighbor nn) {
    // 查找插入位置（二分查找）
    int left = 0, right = K - 1;

    // 比所有元素都更近
    if (addr[left].distance > nn.distance) {
        memmove(&addr[left + 1], &addr[left], K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }

    // 比所有元素都更远
    if (addr[right].distance < nn.distance) {
        addr[K] = nn;
        return K;
    }

    // 二分查找
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance) {
            right = mid;
        } else {
            left = mid;
        }
    }

    // 检查是否已存在相同ID
    while (left > 0) {
        if (addr[left].distance < nn.distance) {
            break;
        }
        if (addr[left].id == nn.id) {
            return K + 1;  // 返回K+1表示已存在
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

### 10.3 NSG::init_graph - 计算导航点

```cpp
// faiss/impl/NSG.cpp
// 计算中心点并设置导航点（入口点）
void NSG::init_graph(Index* storage, const nsg::Graph<idx_t>& knn_graph) {
    int d = storage->d;
    int n = storage->ntotal;

    std::unique_ptr<float[]> center(new float[d]);
    std::unique_ptr<float[]> tmp(new float[d]);
    std::fill_n(center.get(), d, 0.0f);

    // 1. 计算所有向量的质心
    for (int i = 0; i < n; i++) {
        storage->reconstruct(i, tmp.get());
        for (int j = 0; j < d; j++) {
            center[j] += tmp[j];
        }
    }

    for (int i = 0; i < d; i++) {
        center[i] /= n;
    }

    // 2. 在KNN图上搜索离质心最近的点作为导航点
    std::vector<Neighbor> retset;
    std::vector<Node> tmpset;

    // 随机初始化导航点
    int ep = rng.rand_int(n);
    std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));

    dis->set_query(center.get());
    VisitedTable vt(ntotal);

    // 在KNN图上搜索
    search_on_graph<false>(knn_graph, *dis, vt, ep, L, retset, tmpset);

    // 设置入口点为最近点
    enterpoint = retset[0].id;
}
```

### 10.4 NSG::sync_prune - 边裁剪实现

```cpp
// faiss/impl/NSG.cpp
// 裁剪邻居列表，保持单调性和度数限制
void NSG::sync_prune(
        int q,
        std::vector<Node>& pool,
        DistanceComputer& dis,
        VisitedTable& vt,
        const nsg::Graph<idx_t>& knn_graph,
        nsg::Graph<Node>& graph) {

    // 1. 添加KNN图的邻居到候选池
    for (int i = 0; i < knn_graph.K; i++) {
        int id = knn_graph.at(q, i);
        if (id < 0 || id >= ntotal || vt.get(id)) {
            continue;
        }

        float dist = dis.symmetric_dis(q, id);
        pool.emplace_back(id, dist);
    }

    // 2. 按距离排序
    std::sort(pool.begin(), pool.end());

    // 3. 裁剪以保持单调性
    std::vector<Node> result;

    int start = 0;
    if (pool[start].id == q) {
        start++;
    }
    result.push_back(pool[start]);

    // 遍历候选池
    while (result.size() < R && (++start) < pool.size() && start < C) {
        auto& p = pool[start];
        bool occlude = false;

        // 检查是否被已有邻居遮挡
        for (int t = 0; t < result.size(); t++) {
            if (p.id == result[t].id) {
                occlude = true;
                break;
            }
            // 遮挡条件：如果d(j,k) < d(i,k)，则j遮挡k
            float djk = dis.symmetric_dis(result[t].id, p.id);
            if (djk < p.distance /* dik */) {
                occlude = true;
                break;
            }
        }

        if (!occlude) {
            result.push_back(p);
        }
    }

    // 4. 更新邻居表
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

### 10.5 NSG::add_reverse_links - 添加反向链接

```cpp
// faiss/impl/NSG.cpp
// 添加反向链接使NSG图更连通
void NSG::add_reverse_links(
        int q,
        std::vector<std::mutex>& locks,
        DistanceComputer& dis,
        nsg::Graph<Node>& graph) {

    for (size_t i = 0; i < R; i++) {
        if (graph.at(q, i).id == EMPTY_ID) {
            break;
        }

        // 尝试添加反向链接：q -> neighbors[i] 变为 neighbors[i] -> q
        Node sn(q, graph.at(q, i).distance);
        int des = graph.at(q, i).id;

        std::vector<Node> tmp_pool;
        int dup = 0;

        {
            LockGuard guard(locks[des]);
            // 收集des的现有邻居
            for (int j = 0; j < R; j++) {
                if (graph.at(des, j).id == EMPTY_ID) {
                    break;
                }
                if (q == graph.at(des, j).id) {
                    dup = 1;  // 反向链接已存在
                    break;
                }
                tmp_pool.push_back(graph.at(des, j));
            }
        }

        if (dup) {
            continue;
        }

        tmp_pool.push_back(sn);

        // 如果邻居数超过R，需要裁剪
        if (tmp_pool.size() > R) {
            std::vector<Node> result;
            int start = 0;
            std::sort(tmp_pool.begin(), tmp_pool.end());
            result.push_back(tmp_pool[start]);

            // 裁剪（遮挡检查）
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
            for (int j = 0; j < tmp_pool.size(); j++) {
                graph.at(des, j) = tmp_pool[j];
            }
        }
    }
}
```

### 10.6 NSG::tree_grow - 确保连通性

```cpp
// faiss/impl/NSG.cpp
// 使NSG完全连通：连接孤立节点到主连通分量
int NSG::tree_grow(Index* storage, std::vector<int>& degrees) {
    VisitedTable vt(ntotal);
    VisitedTable vt2(ntotal);

    int root = enterpoint;
    int num_attached = 0;

    // 1. 计算初始连通分量大小
    int n_connected = dfs(vt, root, 0);

    // 2. 连接孤立节点
    while (n_connected < ntotal) {
        num_attached++;

        // 找到第一个孤立节点
        bool found_unlinked = false;
        for (int i = 0; i < ntotal; i++) {
            if (!vt.get(i)) {
                // 连接孤立节点i到root
                if (degrees[root] < R) {
                    final_graph->at(root, degrees[root]++) = i;
                } else {
                    // root的邻居已满，找到其他可连接点
                    int attached = attach_unlinked(storage, vt, vt2, degrees);
                    if (attached == -1) {
                        // 无法连接，添加新边
                        final_graph->at(root, R - 1) = i;
                    }
                }
                n_connected = dfs(vt, root, 0);
                found_unlinked = true;
                break;
            }
        }

        if (!found_unlinked) {
            break;
        }
    }

    return num_attached;
}

// 深度优先搜索：计算连通分量大小
int NSG::dfs(VisitedTable& vt, int root, int cnt) const {
    vt.set(root);
    cnt++;

    std::vector<int32_t> neighbors(R);
    size_t n_neighbors = final_graph->get_neighbors(root, neighbors.data());

    for (size_t i = 0; i < n_neighbors; i++) {
        int neighbor = neighbors[i];
        if (!vt.get(neighbor)) {
            cnt = dfs(vt, neighbor, cnt);
        }
    }

    return cnt;
}
```

### 10.7 NNDescent::Nhood - 邻域结构

```cpp
// faiss/impl/NNDescent.cpp
// 节点的邻域：管理候选池和新/旧邻居
namespace nndescent {

struct Nhood {
    std::mutex lock;               // 并行构建时的锁
    std::vector<Neighbor> pool;   // 候选池（最大堆）
    int M;                         // 要操作的新邻居数量

    std::vector<int> nn_old;       // 旧邻居列表
    std::vector<int> nn_new;       // 新邻居列表
    std::vector<int> rnn_old;      // 反向旧邻居
    std::vector<int> rnn_new;      // 反向新邻居

    Nhood(int l, int s, std::mt19937& rng, int N) {
        M = s;
        pool.reserve(l);
        nn_new.reserve(s * 2);
        nn_old.reserve(s * 2);
        rnn_new.reserve(s * 2);
        rnn_old.reserve(s * 2);

        // 随机初始化邻居
        gen_random(rng, nn_new.data(), (int)nn_new.capacity(), N);
    }

    // 将候选插入池中（线程安全）
    void insert(int id, float dist) {
        std::lock_guard<std::mutex> guard(lock);

        // 检查是否已在池中
        for (int i = 0; i < pool.size(); i++) {
            if (id == pool[i].id) {
                return;
            }
        }

        // 插入到池中
        if (pool.size() < pool.capacity()) {
            pool.push_back(Neighbor(id, dist, true));
            std::push_heap(pool.begin(), pool.end());
        } else if (dist < pool.front().distance) {
            // 比最远候选更近，替换
            std::pop_heap(pool.begin(), pool.end());
            pool.back() = Neighbor(id, dist, true);
            std::push_heap(pool.begin(), pool.end());
        }
    }

    // 局部join：只在新邻居参与时才比较
    template <typename C>
    void join(C callback) const {
        // 新-新
        for (int const i : nn_new) {
            for (int const j : nn_new) {
                if (i < j) {
                    callback(i, j);
                }
            }
        }
        // 新-旧
        for (int const i : nn_new) {
            for (int const j : nn_old) {
                callback(i, j);
            }
        }
        // 反向新-旧
        for (int const i : rnn_new) {
            for (int const j : nn_old) {
                callback(i, j);
            }
        }
    }
};

} // namespace nndescent
```

### 10.8 NNDescent::nndescent - 主迭代循环

```cpp
// faiss/impl/NNDescent.cpp
// NNDescent主算法：迭代优化KNN图
void NNDescent::nndescent(DistanceComputer& qdis, bool verbose) {
    std::mt19937 rng(random_seed);
    const int L = this->L;  // 候选池大小

    // 1. 随机初始化图
    init_graph(qdis);

    // 2. 迭代优化
    for (int it = 0; it < iter; it++) {
        if (verbose) {
            printf("NNDescent iteration %d\n", it);
        }

        // 局部join：邻居的邻居
        join(qdis);

        // 更新：采样新/旧邻居
        update();

        // 评估收敛（可选）
        if (it % 5 == 0) {
            std::vector<int> ctrl_points;
            std::vector<std::vector<int>> acc_eval_set;
            generate_eval_set(qdis, ctrl_points, acc_eval_set, 100);
            float recall = eval_recall(ctrl_points, acc_eval_set);
            if (verbose) {
                printf("  Recall: %.3f\n", recall);
            }
        }
    }

    // 3. 构建最终图
    final_graph.resize(ntotal * K);
    for (int i = 0; i < ntotal; i++) {
        std::vector<Neighbor> neighbors;
        // 从pool提取K个最近的
        std::sort(graph[i].pool.begin(), graph[i].pool.end());
        for (int j = 0; j < std::min(K, (int)graph[i].pool.size()); j++) {
            final_graph[i * K + j] = graph[i].pool[j].id;
        }
        // 填充-1
        for (int j = (int)graph[i].pool.size(); j < K; j++) {
            final_graph[i * K + j] = -1;
        }
    }

    has_built = true;
}
```

### 10.9 NNDescent::join - 局部join操作

```cpp
// faiss/impl/NNDescent.cpp
// 局部join：对每个节点，在其邻居的邻居中寻找更好的KNN
void NNDescent::join(DistanceComputer& qdis) {
    idx_t check_period = InterruptCallback::get_period_hint(d * search_L);

    // 分块处理以支持中断检查
    for (idx_t i0 = 0; i0 < (idx_t)ntotal; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, (idx_t)ntotal);

#pragma omp parallel for default(shared) schedule(dynamic, 100)
        for (idx_t n = i0; n < i1; n++) {
            // 对节点n的邻域执行join
            graph[n].join([&](int i, int j) {
                if (i != j) {
                    // 计算对称距离
                    float dist = qdis.symmetric_dis(i, j);
                    // 将候选添加到双方的池中
                    graph[i].insert(j, dist);
                    graph[j].insert(i, dist);
                }
            });
        }

        InterruptCallback::check();
    }
}
```

### 10.10 NNDescent::update - 采样新/旧邻居

```cpp
// faiss/impl/NNDescent.cpp
// 为每个节点采样新/旧邻居用于下一轮join
void NNDescent::update() {
    // 1. 清空所有nn_new和nn_old
#pragma omp parallel for
    for (int i = 0; i < ntotal; i++) {
        std::vector<int>().swap(graph[i].nn_new);
        std::vector<int>().swap(graph[i].nn_old);
        std::vector<int>().swap(graph[i].rnn_new);
        std::vector<int>().swap(graph[i].rnn_old);
    }

    // 2. 计算新邻居的数量并采样
#pragma omp parallel for
    for (int n = 0; n < ntotal; ++n) {
        auto& nn = graph[n];
        std::sort(nn.pool.begin(), nn.pool.end());

        // 限制池大小
        if (nn.pool.size() > L) {
            nn.pool.resize(L);
        }

        // 统计flag为true（新邻居）的数量
        int new_nd = std::count_if(
                nn.pool.begin(), nn.pool.end(),
                [](const Neighbor& n) { return n.flag; });

        int max_new_nd = std::min(S, (int)nn.pool.size());
        new_nd = std::min(new_nd, max_new_nd);

        // 随机选择新邻居
        std::mt19937 rng(random_seed + n);
        std::vector<int> selected_new;
        for (const auto& neighbor : nn.pool) {
            if (neighbor.flag && selected_new.size() < (size_t)new_nd) {
                selected_new.push_back(neighbor.id);
            }
        }
        std::shuffle(selected_new.begin(), selected_new.end(), rng);
        nn.nn_new.assign(selected_new.begin(),
                        selected_new.begin() + std::min(new_nd, (int)selected_new.size()));

        // 选择旧邻居
        std::vector<int> selected_old;
        for (const auto& neighbor : nn.pool) {
            if (!neighbor.flag &&
                nn.nn_new.size() + nn.nn_old.size() < (size_t)S) {
                nn.nn_old.push_back(neighbor.id);
            }
        }

        // 清空池并重置flag
        nn.pool.clear();
    }
}
```

### 10.11 生产级使用示例

```cpp
// 生产环境NSG索引配置
#include <faiss/IndexNSG.h>

void production_nsg_example() {
    int d = 128;          // 维度
    int R = 32;           // 邻居数
    idx_t ntotal = 1000000;

    // 1. 创建索引
    faiss::IndexNSGFlat index(d, R);

    // 2. 添加向量
    std::vector<float> xb(d * ntotal);
    // ... 填充xb ...
    index.add(ntotal, xb.data());

    // 3. 搜索
    idx_t nq = 100;
    idx_t k = 100;
    std::vector<float> xq(d * nq);
    std::vector<float> distances(k * nq);
    std::vector<idx_t> labels(k * nq);

    index.search(nq, xq.data(), k,
                 distances.data(), labels.data());

    // 4. 性能分析
    printf("NSG navigation point: %d\n", index.nsg.enterpoint);

    // 5. 保存索引
    {
        faiss::IOWriter* writer = new faiss::IOFile("nsg.index", "wb");
        faiss::write_index(&index, writer);
        delete writer;
    }
}

// NNDescent构建KNN图
void build_knn_graph_example() {
    int d = 128;
    int K = 32;
    idx_t ntotal = 100000;

    std::vector<float> xb(d * ntotal);
    // ... 填充xb ...

    // 创建NNDescent
    faiss::NNDescent nnd(d, K);
    nnd.S = 10;      // 采样邻居数
    nnd.iter = 10;   // 迭代次数

    // 构建KNN图
    faiss::IndexFlat storage(d, faiss::METRIC_L2);
    storage.add(ntotal, xb.data());

    std::unique_ptr<faiss::DistanceComputer> dis(
        storage.get_distance_computer());
    nnd.build(*dis, ntotal, true);

    // 获取KNN图
    const int* graph = nnd.final_graph.data();
}
```

---

## 9. 第8天总结

### 关键概念

1. **NSG**：有向图，单调性搜索
2. **NN-Descent**：迭代优化KNN图
3. **导航点**：搜索起点（通常为质心）
4. **边裁剪**：控制度数和单调性
5. **Nhood结构**：管理新旧邻居

### 算法对比

| 算法 | 构建复杂度 | 查询复杂度 | 内存 | 精度 |
|------|-----------|-----------|------|------|
| HNSW | O(n log n) | O(log n) | 高 | 最高 |
| NSG | O(n log n) | O(log n) | 中 | 高 |
| NN-Descent | O(n log n) | O(√n) | 中 | 中 |

### 下一步

第9天将学习**FastScan架构**，这是Faiss中极致SIMD优化的索引。

---

## 练习题

1. 实现简化的NN-Descent算法
2. 实现NSG的搜索算法
3. 比较HNSW和NSG的性能
4. 实现并行构建优化

## 10. NSG底层实现细节补充

### 10.1 occlusion判断底层实现

```cpp
// faiss/impl/NSG.cpp:sync_prune
// occlusion(遮挡)判断的核心实现

void sync_prune(
        int q,
        std::vector<Node>& pool,
        DistanceComputer& dis,
        VisitedTable& vt,
        const nsg::Graph<idx_t>& knn_graph,
        nsg::Graph<Node>& graph) {

    // 1. 按距离排序候选池
    std::sort(pool.begin(), pool.end());

    // 2. 获取q的当前邻居
    auto& neighbors = graph[vp];

    // 3. 裁剪候选
    size_t start = neighbors.size();
    if (start == 0) start++;

    int cnt = 0;
    for (size_t i = start; i < pool.size() && cnt < R; i++) {
        auto& p = pool[i];

        // occlusion判断: 如果p被某个现有邻居遮挡,则跳过
        bool occlude = false;
        for (int t = 0; t < neighbors.size(); t++) {
            idx_t id = neighbors[t].id;
            if (id == p.id) {
                continue;  // 自己
            }

            // 计算p与现有邻居的距离
            float djk = dis.symmetric_dis(id, p.id);

            // 关键判断: djk <= dik 则认为p被neighbor[t]遮挡
            if (djk < p.distance * 1.001f) {  // 1.001f是容差系数
                occlude = true;
                break;
            }
        }

        if (!occlude) {
            graph.add(q, p.id);
            cnt++;
        }
    }
}
```

**关键参数**:
- `1.001f`: 容差系数,处理浮点误差
- 理论上应该用`<=`,但由于浮点精度,使用略大于1的系数

### 10.2 反向链接添加

```cpp
// faiss/impl/NSG.cpp:链接反向边的实现

void add_reverse_links(
        const nsg::Graph<Node>& graph,
        nsg::Graph<Node>& graph_reverse,
        std::mutex& link_mutex) {

    for (size_t i = 0; i < graph.size(); i++) {
        auto& neighbors = graph[i];

        for (size_t j = 0; j < neighbors.size(); j++) {
            idx_t to = neighbors[j].id;

            // 使用互斥锁保证线程安全
            {
                std::lock_guard<std::mutex> lock(link_mutex);

                // 添加反向链接: to -> i
                graph_reverse[to].push_back(
                    Node(i, 0));  // 距离在后处理时更新
            }
        }
    }
}
```

**设计要点**:
- 反向链接用于支持反向搜索(从结果回溯到起点)
- 每个list一个细粒度锁,减少锁竞争

### 10.3 DFS连通性检查

```cpp
// faiss/impl/NSG.cpp:check_connectivity实现

int check_connectivity(
        const nsg::Graph<idx_t>& graph,
        idx_t root) {

    VisitedTable vt(graph.size());
    std::stack<idx_t> stack;

    // 从root开始DFS
    stack.push(root);
    vt.set(root);

    int visited_count = 0;

    while (!stack.empty()) {
        idx_t node = stack.top();
        stack.pop();

        visited_count++;

        // 遍历邻居
        for (idx_t neighbor : graph[node]) {
            if (!vt.get(neighbor)) {
                vt.set(neighbor);
                stack.push(neighbor);
            }
        }
    }

    return visited_count;
}
```

**用途**:
- 确保图连通(从导航点可达所有节点)
- 检测孤立节点并处理

### 10.4 NNDescent采样优化

```cpp
// faiss/impl/NNDescent.cpp:采样优化

void NNDescent::build(
        DistanceComputer& dis,
        idx_t n,
        const Graph<idx_t>& init_graph,
        bool search_knn_graph) {

    // 1. 初始化随机种子
    std::mt19937 rng(12345);

    for (int iter = 0; iter < this->iter; iter++) {
        // 2. 对每个节点
        for (idx_t i = 0; i < n; i++) {
            auto& neighbors = final_graph[i];

            // 3. 采样邻居的邻居
            std::vector<idx_t> sampled;
            sampled.reserve(S * R);

            // 从邻居中随机采样S个
            for (int s = 0; s < S && s < neighbors.size(); s++) {
                idx_t nb = neighbors[s].id;

                // 从nb的邻居中随机采样R个
                for (int r = 0; r < R && r < final_graph[nb].size(); r++) {
                    idx_t nn = final_graph[nb][r].id;
                    if (nn != i) {
                        sampled.push_back(nn);
                    }
                }
            }

            // 4. 去重并计算距离
            std::sort(sampled.begin(), sampled.end());
            sampled.erase(
                std::unique(sampled.begin(), sampled.end()),
                sampled.end());

            // 5. 计算距离并更新KNN
            std::vector<Node> new_pool;

            for (idx_t candidate : sampled) {
                float d = dis(i, candidate);
                new_pool.push_back(Node(candidate, d));
            }

            // 6. 更新邻居(只保留前R个)
            update_knn(final_graph[i], new_pool, R);
        }
    }
}
```

### 10.5 NSG与HNSW的性能对比

| 指标 | HNSW | NSG | 说明 |
|------|------|-----|------|
| **构建时间** | O(n log n) | O(n log n) | 相当 |
| **内存** | 2-3x基础 | 1-1.5x基础 | NSG更省 |
| **搜索延迟** | 100% | 120-150% | HNSW更快 |
| **精度** | 100% | 95-98% | HNSW略高 |
| **度数控制** | M参数 | R参数 | 机制不同 |

### 10.6 NSG构建动态负载均衡

```cpp
// 动态负载均衡的NSG构建

class LoadBalancedNSGBuilder {
    std::vector<std::queue<idx_t>> node_queues;  // 每个线程一个队列

    void build_parallel(NSG& nsg, int num_threads) {
        node_queues.resize(num_threads);

        // 1. 初始分配
        for (idx_t i = 0; i < nsg.ntotal; i++) {
            node_queues[i % num_threads].push(i);
        }

        // 2. 并发构建邻居
        #pragma omp parallel for num_threads(num_threads)
        for (int t = 0; t < num_threads; t++) {
            while (!node_queues[t].empty()) {
                idx_t node = node_queues[t].front();
                node_queues[t].pop();

                // 构建该节点的邻居
                build_neighbors(nsg, node, t);

                // 动态负载均衡:偷取其他线程的任务
                if (node_queues[t].empty()) {
                    for (int ot = 0; ot < num_threads; ot++) {
                        if (!node_queues[ot].empty() &&
                            node_queues[ot].size() > 1) {
                            // 偷取一半任务
                            idx_t stolen = node_queues[ot].front();
                            node_queues[ot].pop();
                            node_queues[t].push(stolen);
                        }
                    }
                }
            }
        }
    }

    void build_neighbors(NSG& nsg, idx_t node, int thread_id) {
        // 标准的NSG邻居构建逻辑
        // ...
    }
};
```

---

## 扩展阅读

- faiss/impl/NSG.h - NSG实现
- faiss/impl/NNDescent.h - NNDescent实现
- [NSG论文](https://arxiv.org/abs/1711.08530)
- [NN-Descent论文](https://dl.acm.org/doi/10.1145/1835804.1835877)
