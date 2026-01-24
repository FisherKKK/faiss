# Faiss深度课程 - 第13天：复合索引与高级特性

## 课程目标

学习Faiss的复合索引和高级特性，包括索引组合、包装器、预变换等。

---

## 1. 复合索引概述

### 1.1 IndexPreTransform

```cpp
// faiss/IndexPreTransform.h
struct IndexPreTransform : Index {
    Index* sub_index;     // 底层索引
    bool own_index;       // 是否拥有索引
    VectorTransform* transform;  // 预变换

    IndexPreTransform(Index* sub_index)
        : sub_index(sub_index), own_index(false) {
        d = sub_index->d;
        metric_type = sub_index->metric_type;
    }

    ~IndexPreTransform() {
        if (own_index && sub_index) {
            delete sub_index;
        }
    }

    void add(idx_t n, const float* x) override {
        // 应用变换
        float* xt = transform->apply(n, x);

        // 添加到子索引
        sub_index->add(n, xt);

        // 清理
        transform->reverse_transform(n, xt);
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 变换查询
        float* xt = transform->apply(n, x);

        // 在变换后的空间搜索
        sub_index->search(n, xt, k, distances, labels, params);

        // 距离需要调整吗？某些变换需要
        transform->reverse_transform(1, distances);  // 如果需要
    }
};
```

### 1.2 PCA降维

```cpp
// PCA降维索引
void pca_index_example() {
    int d_original = 512;
    int d_reduced = 128;
    int n = 100000;

    // 1. 训练PCA
    float* pcamat = new float[d_original * d_reduced];
    float* mean = new float[d_original];

    compute_pca(xb, n, d_original, d_reduced,
                pcamat, mean);

    // 2. 创建PCA变换
    PCAMatrix* pca = new PCAMatrix(d_original, d_reduced);
    pca->set_A(pcamat);
    pca->set_b(mean);

    // 3. 创建索引
    IndexFlatL2 sub_index(d_reduced);
    IndexPreTransform index(&sub_index, pca, true);

    // 4. 训练和添加
    index.train(n, xb);
    index.add(n, xb);

    // 5. 搜索（自动应用PCA）
    index.search(nq, xq, k, distances, labels);
}
```

### 1.3 OPQ旋转

```cpp
// OPQ（Product Quantization with rotation）
struct OPQMatrix {
    int d;      // 原始维度
    int M;      // PQ子量化器数
    float* A;   // 旋转矩阵 (d × d)
    float* b;   // 偏移向量

    void train(const float* x, size_t n) {
        // 随机初始化旋转矩阵（正交矩阵）
        random_rotation(d, A);

        // 迭代优化旋转
        for (int iter = 0; iter < 100; iter++) {
            // 应用旋转
            float* x_rot = new float[n * d];
            apply_rotation(x, n, A, x_rot);

            // 训练PQ
            ProductQuantizer pq(d, M, 8);
            pq.train(n, x_rot);

            // 优化旋转
            optimize_rotation(x, n, pq, A);

            delete[] x_rot;
        }
    }
};
```

---

## 2. IndexRefine

### 2.1 两阶段搜索

```cpp
// faiss/IndexRefine.h
struct IndexRefine : Index {
    Index* base_index;    // 基础索引（粗糙但快）
    Index* refine_index;  // 精炼索引（精确但慢）
    float k_factor;       // 精炼倍数

    IndexRefine(Index* base_index, Index* refine_index, float k_factor = 8)
        : base_index(base_index), refine_index(refine_index),
          k_factor(k_factor) {
        d = base_index->d;
        ntotal = base_index->ntotal;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 阶段1：粗糙搜索
        idx_t k_base = (idx_t)(k * k_factor);
        idx_t* base_labels = new idx_t[n * k_base];
        float* base_distances = new float[n * k_base];

        base_index->search(n, x, k_base,
                          base_distances, base_labels);

        // 阶段2：精炼
        for (idx_t i = 0; i < n; i++) {
            // 从精炼索引获取精确距离
            idx_t* lbl_i = base_labels + i * k_base;

            refine_index->search(1, x + i * d, k,
                               distances + i * k,
                               labels + i * k);
        }

        delete[] base_labels;
        delete[] base_distances;
    }
};
```

### 2.2 使用示例

```cpp
void refine_example() {
    // 粗糙索引：IVF+PQ（快）
    Index* quantizer = new IndexFlatL2(d);
    IndexIVFPQ* base = new IndexIVFPQ(quantizer, d, nlist, M, nbits);

    // 精炼索引：Flat（精确）
    IndexFlat* refine = new IndexFlatL2(d);

    // 合并
    IndexRefine index(base, refine, k_factor = 10);

    index.train(n, xb);
    base->add(n, xb);
    refine->add(n, xb);

    index.search(nq, xq, k, distances, labels);
}
```

---

## 3. IndexShards

### 3.1 分片索引

```cpp
// faiss/IndexShards.h
struct IndexShards : Index {
    std::vector<Index*> shards;   // 分片索引
    bool own_fields;              // 是否拥有索引

    IndexShards(idx_t d, MetricType metric = METRIC_L2)
        : Index(d, metric), own_fields(true) {}

    void add_shard(Index* shard) {
        shards.push_back(shard);
        ntotal += shard->ntotal;
    }

    void add(idx_t n, const float* x) override {
        // 分配到各分片
        idx_t n_per_shard = n / shards.size();

        for (size_t i = 0; i < shards.size(); i++) {
            idx_t start = i * n_per_shard;
            idx_t count = (i == shards.size() - 1) ?
                n - start : n_per_shard;

            shards[i]->add(count, x + start * d);
        }

        ntotal += n;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 并行搜索所有分片
        #pragma omp parallel for
        for (size_t i = 0; i < shards.size(); i++) {
            float* distances_i = new float[n * k];
            idx_t* labels_i = new idx_t[n * k];

            shards[i]->search(n, x, k,
                            distances_i, labels_i);

            // 合并结果
            // ...（需要实现n个结果的k近邻合并）

            delete[] distances_i;
            delete[] labels_i;
        }
    }
};
```

### 3.2 数据并行

```cpp
// 创建分片索引
void create_sharded_index() {
    int n_shards = 4;

    IndexShards index(d, METRIC_L2);

    // 创建并添加分片
    for (int i = 0; i < n_shards; i++) {
        IndexFlatL2* shard = new IndexFlatL2(d);
        index.add_shard(shard);
    }

    // 训练和添加
    index.train(n, xb);
    index.add(n, xb);

    // 搜索
    index.search(nq, xq, k, distances, labels);
}
```

---

## 4. IndexReplicas

### 4.1 副本索引

```cpp
// faiss/IndexReplicas.h
struct IndexReplicas : Index {
    std::vector<Index*> replicas;  // 副本（相同数据）
    bool own_fields;

    IndexReplicas(idx_t d, MetricType metric = METRIC_L2)
        : Index(d, metric), own_fields(true) {}

    void add_replica(Index* replica) {
        replicas.push_back(replica);
        sync_ntotal();
    }

    void sync_ntotal() {
        ntotal = replicas.empty() ? 0 : replicas[0]->ntotal;
    }

    void add(idx_t n, const float* x) override {
        // 添加到所有副本
        for (auto* replica : replicas) {
            replica->add(n, x);
        }

        ntotal += n;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 只搜索一个副本（或动态选择）
        replicas[0]->search(n, x, k, distances, labels);
    }
};
```

### 4.2 查询并行

```cpp
// 查询并行版本
void IndexReplicas::search_parallel(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {

    // 将查询分配到不同副本
    #pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        int replica_id = omp_get_thread_num() % replicas.size();

        replicas[replica_id]->search(
            1, x + i * d, k,
            distances + i * k,
            labels + i * k);
    }
}
```

---

## 5. IndexIDMap

### 5.1 自定义ID映射

```cpp
// faiss/IndexIDMap.h
struct IndexIDMap : Index {
    Index* index;      // 底层索引
    std::vector<idx_t> id_map;  // 外部ID → 内部ID

    IndexIDMap(Index* index)
        : Index(index->d, index->metric_type),
          index(index) {
        own_fields = true;
    }

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override {
        // 生成内部ID
        idx_t base_id = ntotal;

        // 更新映射
        id_map.resize(ntotal + n);
        for (idx_t i = 0; i < n; i++) {
            id_map[ntotal + i] = xids[i];
        }

        // 使用内部ID添加
        index->add(n, x);
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 搜索返回内部ID
        index->search(n, x, k, distances, labels);

        // 转换为外部ID
        for (idx_t i = 0; i < n * k; i++) {
            if (labels[i] >= 0 && labels[i] < ntotal) {
                labels[i] = id_map[labels[i]];
            }
        }
    }
};
```

---

## 6. Index2Layer

### 6.1 两层结构

```cpp
// faiss/Index2Layer.h
struct Index2Layer : Index {
    Index* quantizer;   // 第一层：粗量化器
    Index* q1;         // 第二层：量化索引
    Index* q2;         // 第二层：存储索引

    int nlist;
    idx_t* list_ids;   // 列表的ID映射

    Index2Layer(Index* quantizer, size_t nlist, Index* q1, Index* q2)
        : Index(quantizer->d),
          quantizer(quantizer), q1(q1), q2(q2),
          nlist(nlist) {}

    void train(idx_t n, const float* x) override {
        // 训练粗量化器
        quantizer->train(n, x);

        // 量化向量
        idx_t* assign = new idx_t[n];
        quantizer->assign(n, x, assign);

        // 训练第二层
        q1->train(n, x);
        q2->train(n, x);

        delete[] assign;
    }

    void add(idx_t n, const float* x) override {
        // 分配到倒排列表
        idx_t* list_nos = new idx_t[n];
        quantizer->assign(n, x, list_nos);

        // 添加到第二层
        q1->add_with_ids(n, x, list_nos);
        q2->add_with_ids(n, x, list_nos);

        delete[] list_nos;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 第一层：粗量化
        idx_t* list_nos = new idx_t[n * nprobe];
        quantizer->search(n, x, nprobe, distances, list_nos);

        // 第二层：在候选列表中搜索
        for (idx_t i = 0; i < n; i++) {
            q1->search(
                1, x + i * d, k,
                distances + i * k,
                labels + i * k);

            q2->search(
                1, x + i * d, k,
                distances + i * k,
                labels + i * k);
        }

        delete[] list_nos;
    }
};
```

---

## 7. IndexFlatPanorama

### 7.1 渐进式剪枝

```cpp
// Panorama：渐进式距离计算
struct IndexFlatPanorama : IndexFlat {
    int n_levels;
    Panorama pano;

    IndexFlatPanorama(int d, int n_levels)
        : IndexFlat(d), n_levels(n_levels),
          pano(d, n_levels, batch_size) {}

    void add(idx_t n, const float* x) override {
        // 计算累积和
        for (idx_t i = 0; i < n; i++) {
            pano.compute_cumulative_sums(
                x + i * d, cum_sums.data() + i * (n_levels + 1));
        }

        // 添加基础向量
        IndexFlat::add(n, x);
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        for (idx_t i = 0; i < n; i++) {
            const float* xi = x + i * d;

            // 计算查询的累积和
            std::vector<float> query_cum_sums(n_levels + 1);
            pano.compute_query_cum_sums(xi, query_cum_sums.data());

            // 渐进式搜索
            heap_heapify<CMax<float, idx_t>>(k, distances + i * k,
                                             labels + i * k);

            for (idx_t j = 0; j < ntotal; j++) {
                float upper_bound = distances[i * k];

                // 渐进式计算距离，提前剪枝
                float dis = progressive_distance(
                    xi, j, query_cum_sums, upper_bound);

                if (dis < upper_bound) {
                    heap_replace_top<CMax<float, idx_t>>(
                        k, distances + i * k, labels + i * k,
                        dis, j);
                }
            }

            heap_reorder<CMax<float, idx_t>>(k, distances + i * k,
                                             labels + i * k);
        }
    }
};
```

---

## 9. 复合索引底层实现详解

### 9.1 IndexPreTransform完整结构

```cpp
// faiss/IndexPreTransform.h
struct IndexPreTransform : Index {
    std::vector<VectorTransform*> chain; ///! 变换链
    Index* index;                          ///! 子索引
    bool own_fields;                       ///! 是否拥有字段

    explicit IndexPreTransform(Index* index);
    IndexPreTransform();

    /// ltrans是索引前的最后一个变换
    IndexPreTransform(VectorTransform* ltrans, Index* index);

    ~IndexPreTransform() {
        if (own_fields) {
            for (auto t : chain) {
                delete t;
            }
            delete index;
        }
    }

    void prepend_transform(VectorTransform* ltrans);

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;
    void reset() override;
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

    size_t remove_ids(const IDSelector& sel) override;
};
```

### 9.2 VectorTransform完整结构

```cpp
// faiss/VectorTransform.h
struct VectorTransform {
    int d_in;      // 输入维度
    int d_out;     // 输出维度
    bool is_trained;

    VectorTransform(int d_in = 0, int d_out = 0)
        : d_in(d_in), d_out(d_out), is_trained(false) {}

    virtual ~VectorTransform() {}

    // 训练变换
    virtual void train(idx_t n, const float* x) = 0;

    // 应用变换
    virtual void apply(idx_t n, const float* x, float* xt) const = 0;

    // 反向变换（如果支持）
    virtual void reverse_transform(idx_t, float* xt) const {
        FAISS_THROW_MSG("reverse_transform not implemented");
    }
};

// PCA变换
struct PCAMatrix : VectorTransform {
    std::vector<float> A;      // d_out × d_in 旋转矩阵
    std::vector<float> b;      // d_in      偏移向量
    std::vector<float> mean;   // d_in      均值

    // have_mean：是否在变换中减去均值
    bool have_mean;
    // random_rotation：是否使用随机旋转
    bool random_rotation;
    // normalize：是否归一化
    bool normalize;

    PCAMatrix(
            int d_in = 0,
            int d_out = 0,
            bool have_mean = true,
            bool random_rotation = false,
            bool normalize = false);

    void train(idx_t n, const float* x) override;

    void apply(idx_t n, const float* x, float* xt) const override {
        // xt = A * (x - b)  或  xt = A * x
        // 其中b是mean（如果have_mean）

        for (idx_t i = 0; i < n; i++) {
            const float* xi = x + i * d_in;
            float* xto = xt + i * d_out;

            // 乘以A
            for (int j = 0; j < d_out; j++) {
                float accu = 0;
                for (int k = 0; k < d_in; k++) {
                    float val = xi[k];
                    if (have_mean) {
                        val -= mean[k];
                    }
                    accu += A[j * d_in + k] * val;
                }
                xto[j] = accu;
            }

            // 归一化
            if (normalize) {
                float norm = 0;
                for (int j = 0; j < d_out; j++) {
                    norm += xto[j] * xto[j];
                }
                norm = 1.0f / (sqrtf(norm) + 1e-10f);
                for (int j = 0; j < d_out; j++) {
                    xto[j] *= norm;
                }
            }
        }
    }

    void reverse_transform(idx_t n, float* xt) const override;
};
```

### 9.3 IndexRefine完整实现

```cpp
// faiss/IndexRefine.h
struct IndexRefine : Index {
    Index* base_index;      // 基础索引（快速但粗糙）
    Index* refine_index;    // 精炼索引（精确但慢）
    float k_factor;          // 精炼倍数（k_factor * k）

    bool own_fields;        // 是否拥有索引

    IndexRefine(
            Index* base_index,
            Index* refine_index,
            float k_factor = 8.0f)
        : Index(base_index->d, base_index->metric_type),
          base_index(base_index),
          refine_index(refine_index),
          k_factor(k_factor),
          own_fields(false) {

        ntotal = base_index->ntotal;
        is_trained = base_index->is_trained;
    }

    ~IndexRefine() {
        if (own_fields) {
            delete base_index;
            delete refine_index;
        }
    }

    void add(idx_t n, const float* x) override {
        // 添加到两个索引
        base_index->add(n, x);
        refine_index->add(n, x);
        ntotal += n;
    }

    void reset() override {
        base_index->reset();
        refine_index->reset();
        ntotal = 0;
    }

    void train(idx_t n, const float* x) override {
        base_index->train(n, x);
        refine_index->train(n, x);
        is_trained = true;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        idx_t k_base = (idx_t)(k * k_factor);

        // 阶段1：粗糙搜索（基础索引）
        idx_t* base_labels = new idx_t[n * k_base];
        float* base_distances = new float[n * k_base];

        base_index->search(n, x, k_base, base_distances, base_labels);

        // 阶段2：精炼（使用精炼索引）
        // 对于每个查询，使用候选ID进行精确搜索
        idx_t* refine_labels = new idx_t[k_base];
        float* refine_distances = new float[k_base];

        for (idx_t i = 0; i < n; i++) {
            idx_t* lbl_i = base_labels + i * k_base;
            float* dis_i = base_distances + i * k_base;
            const float* xi = x + i * d;

            // 在精炼索引中搜索
            refine_index->search(1, xi, k_base,
                                   refine_distances,
                                   refine_labels);

            // 合并结果
            // 这里简化处理：选择最好的k个结果
            heap_heapify<CMin<float, idx_t>>(k, distances + i * k, labels + i * k);

            for (idx_t j = 0; j < k_base; j++) {
                if (CMin<float, idx_t>::cmp(refine_distances[j], distances[i * k])) {
                    heap_replace_top<CMin<float, idx_t>>(
                        k, distances + i * k, labels + i * k,
                        refine_distances[j], refine_labels[j]);
                }
            }

            heap_reorder<CMin<float, idx_t>>(k, distances + i * k, labels + i * k);
        }

        delete[] base_labels;
        delete[] base_distances;
        delete[] refine_labels;
        delete[] refine_distances;
    }
};
```

### 9.4 IndexShards（分片索引）

```cpp
// faiss/IndexShards.h
// 将索引分成多个shard，并行搜索
struct IndexShards : Index {
    std::vector<Index*> shards;     // 索引分片
    bool own_fields;                // 是否拥有索引
    bool verbose;                   // 是否输出详细信息

    IndexShards();
    ~IndexShards();

    void add_shard(Index* shard);
    void shrink_shards();

    void add(idx_t n, const float* x) override {
        // 将数据分配到各个shard
        idx_t shard_size = n / shards.size();

        for (size_t i = 0; i < shards.size(); i++) {
            idx_t start = i * shard_size;
            idx_t count = (i == shards.size() - 1) ?
                           (n - start) : shard_size;

            shards[i]->add(count, x + start * d);
        }

        ntotal += n;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 初始化堆（用于合并结果）
        std::vector<idx_t> all_labels(n * k);
        std::vector<float> all_distances(n * k);

#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            idx_t* lbl_i = all_labels.data() + i * k;
            float* dis_i = all_distances.data() + i * k;

            heap_heapify<CMax<float, idx_t>>(k, dis_i, lbl_i);

            // 搜索所有shard
            for (const auto& shard : shards) {
                std::vector<float> shard_dists(k);
                std::vector<idx_t> shard_labels(k);

                shard->search(1, x + i * d, k,
                              shard_dists.data(),
                              shard_labels.data());

                // 合并到全局结果
                for (idx_t j = 0; j < k; j++) {
                    if (CMax<float, idx_t>::cmp(shard_dists[j], dis_i[0])) {
                        heap_replace_top<CMax<float, idx_t>>(
                            k, dis_i, lbl_i, shard_dists[j], shard_labels[j]);
                    }
                }
            }

            heap_reorder<CMax<float, idx_t>>(k, dis_i, lbl_i);
        }

        // 复制到输出
        memcpy(distances, all_distances.data(), n * k * sizeof(float));
        memcpy(labels, all_labels.data(), n * k * sizeof(idx_t));
    }

    void sync_shards() {
        // 同步所有shard的ntotal
        for (auto& shard : shards) {
            shard->ntotal = ntotal;
        }
    }
};
```

### 9.5 IndexReplicas（副本索引）

```cpp
// faiss/IndexReplicas.h
// 将索引复制到多个线程/CPU核心，并行搜索
struct IndexReplicas : Index {
    std::vector<Index*> replicas;   // 索引副本
    bool own_fields;                 // 是否拥有索引
    int parallel_mode;               // 并行模式

    IndexReplicas();
    ~IndexReplicas();

    void add_replica(Index* replica);

    void add(idx_t n, const float* x) override {
        // 添加到所有副本
        for (auto& replica : replicas) {
            replica->add(n, x);
        }
        ntotal += n;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t *labels,
            const SearchParameters* params) const override {

#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            // 每个线程使用一个副本进行搜索
            int tid = omp_get_thread_num();
            Index* replica = replicas[tid % replicas.size()];

            replica->search(1, x + i * d, k,
                            distances + i * k,
                            labels + i * k);
        }
    }

    void sync_replicas() {
        // 同步所有副本的ntotal
        for (auto& replica : replicas) {
            replica->ntotal = ntotal;
        }
    }
};
```

### 9. IndexIDMap（ID映射）

```cpp
// faiss/IndexIDMap.h
// 将外部ID映射到内部连续ID
struct IndexIDMap : Index {
    Index* index;           // 底层索引
    std::vector<idx_t> id_map;  // 外部ID -> 内部ID 映射
    bool own_fields;        // 是否拥有索引

    IndexIDMap(Index* index);
    ~IndexIDMap();

    void add_with_ids(
            idx_t n,
            const float* x,
            const idx_t* xids) override {

        // 检查ID是否已存在
        for (idx_t i = 0; i < n; i++) {
            // 这里简化处理，实际需要检查重复
        }

        // 生成内部ID
        idx_t base_id = ntotal;

        // 更新映射
        id_map.resize(ntotal + n);
        for (idx_t i = 0; i < n; i++) {
            id_map[ntotal + i] = xids[i];
        }

        // 使用内部ID添加
        index->add(n, x);
        ntotal += n;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 搜索返回内部ID
        index->search(n, x, k, distances, labels);

        // 转换为外部ID
        for (idx_t i = 0; i < n * k; i++) {
            if (labels[i] >= 0 && labels[i] < ntotal) {
                labels[i] = id_map[labels[i]];
            }
        }
    }

    void reconstruct(idx_t key, float* recons) const override {
        // 先查找内部ID
        idx_t internal_id = -1;
        for (size_t i = 0; i < id_map.size(); i++) {
            if (id_map[i] == key) {
                internal_id = i;
                break;
            }
        }

        if (internal_id >= 0) {
            index->reconstruct(internal_id, recons);
        }
    }
};
```

### 9. Index2Layer完整实现

```cpp
// faiss/Index2Layer.h
// 两层索引：第一层粗量化，第二层细量化
struct Index2Layer : Index {
    Index* quantizer;    // 第一层：粗量化器
    Index* q1;           // 第二层：量化索引
    Index* q2;           // 第二层：存储索引
    Index* q3;           // 第三层（可选）

    size_t nlist;
    std::vector<idx_t> list_ids;   // 列表ID映射

    bool own_fields;

    Index2Layer(
            Index* quantizer,
            size_t nlist,
            Index* q1,
            Index* q2);

    ~Index2Layer() {
        if (own_fields) {
            delete quantizer;
            delete q1;
            delete q2;
        }
    }

    void train(idx_t n, const float* x) override {
        // 训练第一层粗量化器
        quantizer->train(n, x);

        // 分配向量到列表
        idx_t* assign = new idx_t[n];
        quantizer->assign(n, x, assign);

        // 训练第二层索引
        q1->train(n, x);
        q2->train(n, x);

        delete[] assign;
    }

    void add(idx_t n, const float* x) override {
        // 分配到倒排列表
        idx_t* list_nos = new idx_t[n];
        quantizer->assign(n, x, list_nos);

        // 添加到第二层
        q1->add_with_ids(n, x, list_nos);
        q2->add_with_ids(n, x, list_nos);

        delete[] list_nos;
        ntotal += n;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 第一层：粗量化找到候选列表
        idx_t* list_nos = new idx_t[n * nprobe];
        float* coarse_dis = new float[n * nprobe];

        quantizer->search(n, x, nprobe, coarse_dis, list_nos);

        // 第二层：在候选列表中搜索
        for (idx_t i = 0; i < n; i++) {
            heap_heapify<CMax<float, idx_t>>(k, distances + i * k, labels + i * k);

            for (size_t ij = 0; ij < nprobe; ij++) {
                idx_t list_no = list_nos[i * nprobe + ij];

                if (list_no >= 0) {
                    // 使用q1或q2搜索该列表
                    size_t list_size = q1->invlists->list_size(list_no);

                    if (list_size > 0) {
                        // 获取列表向量
                        const float* list_data = get_list_data(q1, list_no);
                        const idx_t* list_ids = get_list_ids(q1, list_no);

                        // 在列表中搜索
                        for (size_t j = 0; j < list_size; j++) {
                            const float* vec = list_data + j * d;
                            float dis = fvec_L2sqr(x + i * d, vec, d);

                            if (CMax<float, idx_t>::cmp(dis, distances[i * k])) {
                                heap_replace_top<CMax<float, idx_t>>(
                                    k, distances + i * k, labels + i * k,
                                    dis, list_ids[j]);
                            }
                        }
                    }
                }
            }

            heap_reorder<CMax<float, idx_t>>(k, distances + i * k, labels + i * k);
        }

        delete[] list_nos;
        delete[] coarse_dis;
    }

   private:
    int nprobe = 16;

    // 辅助函数：获取列表数据
    const float* get_list_data(Index* index, idx_t list_no) const;
    const idx_t* get_list_ids(Index* index, idx_t list_no) const;
};
```

### 9. IndexSplitParts（分片索引）

```cpp
// faiss/IndexSplitParts.h
// 按维度分片，每个索引处理部分维度
struct IndexSplitParts : Index {
    std::vector<Index*> sub_indices;   // 子索引
    std::vector<int> sub_dims;           // 每个子索引的维度
    bool own_fields;

    IndexSplitParts(int d);
    ~IndexSplitParts();

    void add_index(Index* index, int dim);

    void add(idx_t n, const float* x) override {
        // 按维度分割数据
        for (size_t i = 0; i < sub_indices.size(); i++) {
            int dim_offset = 0;
            for (size_t j = 0; j < i; j++) {
                dim_offset += sub_dims[j];
            }

            const float* xi = x + dim_offset;
            sub_indices[i]->add(n, xi);
        }

        ntotal += n;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 对每个子索引搜索
        std::vector<std::vector<float>> all_distances(sub_indices.size());
        std::vector<std::vector<idx_t>> all_labels(sub_indices.size());

        for (size_t i = 0; i < sub_indices.size(); i++) {
            int dim_offset = 0;
            for (size_t j = 0; j < i; j++) {
                dim_offset += sub_dims[j];
            }

            const float* xi = x + dim_offset;

            all_distances[i].resize(n * k);
            all_labels[i].resize(n * k);

            sub_indices[i]->search(n, xi, k,
                                  all_distances[i].data(),
                                  all_labels[i].data());
        }

        // 合并距离（假设独立）
        for (idx_t i = 0; i < n; i++) {
            for (idx_t j = 0; j < k; j++) {
                float total_dis = 0;
                for (size_t m = 0; m < sub_indices.size(); m++) {
                    total_dis += all_distances[m][i * k + j];
                }

                distances[i * k + j] = total_dis;
                labels[i * k + j] = all_labels[0][i * k + j];  // 使用第一个子索引的标签
            }
        }
    }
};
```

---

## 10. 复合索引底层实现详解

### 10.1 IndexPreTransform完整实现分析

```cpp
// faiss/IndexPreTransform.cpp:27-56
// IndexPreTransform的构造函数和析构函数实现

IndexPreTransform::IndexPreTransform(Index* index)
        : Index(index->d, index->metric_type), index(index), own_fields(false) {
    is_trained = index->is_trained;
    ntotal = index->ntotal;
}

// 关键：prepend_transform在变换链前面插入变换
void IndexPreTransform::prepend_transform(VectorTransform* ltrans) {
    FAISS_THROW_IF_NOT(ltrans->d_out == d);  // 输出维度必须等于当前维度
    is_trained = is_trained && ltrans->is_trained;
    chain.insert(chain.begin(), ltrans);  // 在链表头部插入
    d = ltrans->d_in;  // 更新输入维度
}

// 析构函数：释放所有变换链和索引
IndexPreTransform::~IndexPreTransform() {
    if (own_fields) {
        for (int i = 0; i < chain.size(); i++) {
            delete chain[i];
        }
        delete index;
    }
}
```

**关键实现细节**：

1. **变换链管理**：chain是一个std::vector<VectorTransform*>，支持多个变换的串联
2. **维度传播**：d = ltrans->d_in 表示输入维度是第一个变换的输入维度
3. **训练状态管理**：is_trained &&= ltrans->is_trained，所有变换都必须训练

```cpp
// faiss/IndexPreTransform.cpp:58-116
// train实现：按顺序训练未训练的变换
void IndexPreTransform::train(idx_t n, const float* x) {
    // 找到最后一个未训练的变换
    int last_untrained = 0;
    if (!index->is_trained) {
        last_untrained = chain.size();
    } else {
        for (int i = chain.size() - 1; i >= 0; i--) {
            if (!chain[i]->is_trained) {
                last_untrained = i;
                break;
            }
        }
    }

    const float* prev_x = x;
    std::unique_ptr<const float[]> del;

    // 逐个训练变换
    for (int i = 0; i <= last_untrained; i++) {
        if (i < chain.size()) {
            VectorTransform* ltrans = chain[i];
            if (!ltrans->is_trained) {
                ltrans->train(n, prev_x);
            }
        } else {
            index->train(n, prev_x);
        }

        if (i == last_untrained) {
            break;
        }

        // 应用变换到下一阶段
        float* xt = chain[i]->apply(n, prev_x);
        if (prev_x != x) {
            del.reset();
        }
        prev_x = xt;
        del.reset(xt);  // 使用unique_ptr管理临时内存
    }

    is_trained = true;
}
```

**训练流程优化**：
- 只训练未训练的变换（跳过已训练的）
- 使用unique_ptr管理临时内存，避免内存泄漏
- 数据在变换链中逐级传递

```cpp
// faiss/IndexPreTransform.cpp:118-130
// apply_chain：应用所有变换
const float* IndexPreTransform::apply_chain(idx_t n, const float* x) const {
    const float* prev_x = x;
    std::unique_ptr<const float[]> del;

    for (int i = 0; i < chain.size(); i++) {
        float* xt = chain[i]->apply(n, prev_x);
        std::unique_ptr<const float[]> del2(xt);
        del2.swap(del);  // 使用swap转移所有权，避免内存泄漏
        prev_x = xt;
    }
    del.release();  // 释放最后一个变换的结果，返回给调用者
    return prev_x;
}

// reverse_chain：反向应用所有变换（用于reconstruct等操作）
void IndexPreTransform::reverse_chain(idx_t n, const float* xt, float* x) const {
    const float* next_x = xt;
    std::unique_ptr<const float[]> del;

    for (int i = chain.size() - 1; i >= 0; i--) {
        float* prev_x = (i == 0) ? x : new float[n * chain[i]->d_in];
        std::unique_ptr<const float[]> del2((prev_x == x) ? nullptr : prev_x);
        chain[i]->reverse_transform(n, next_x, prev_x);
        del2.swap(del);
        next_x = prev_x;
    }
}
```

**内存管理优化**：
- 使用swap转移unique_ptr所有权
- 最后一个结果不释放，返回给调用者
- 反向变换时，第一个变换输出到x（避免分配）

```cpp
// faiss/IndexPreTransform.cpp:146-161
// TransformedVectors RAII包装器
void IndexPreTransform::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    TransformedVectors tv(x, apply_chain(n, x));  // RAII自动管理内存
    index->add(n, tv.x);
    ntotal = index->ntotal;
}

// TransformedVectors的实现（在faiss/IndexPreTransform.h）
struct TransformedVectors {
    const float* x;
    std::unique_ptr<const float[]> deleter;

    TransformedVectors(const float* x_in, const float* x_out)
        : x(x_out), deleter(x_in == x_out ? nullptr : x_out) {}

    // 析构时自动释放内存
    ~TransformedVectors() {
        // deleter自动释放
    }
};
```

### 10.2 IndexRefine完整实现分析

```cpp
// faiss/IndexRefine.cpp:21-40
// IndexRefine构造函数
IndexRefine::IndexRefine(Index* base_index, Index* refine_index)
        : Index(base_index->d, base_index->metric_type),
          base_index(base_index),
          refine_index(refine_index) {
    own_fields = own_refine_index = false;
    if (refine_index != nullptr) {
        FAISS_THROW_IF_NOT(base_index->d == refine_index->d);
        FAISS_THROW_IF_NOT(base_index->metric_type == refine_index->metric_type);
        is_trained = base_index->is_trained && refine_index->is_trained;
        FAISS_THROW_IF_NOT(base_index->ntotal == refine_index->ntotal);
    }
    ntotal = base_index->ntotal;
}
```

**关键验证**：
- 两个索引必须具有相同的维度和度量类型
- 两个索引的训练状态必须一致
- 两个索引的向量数量必须相等

```cpp
// faiss/IndexRefine.cpp:61-139
// search实现：两阶段搜索
void IndexRefine::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {

    // 提取k_factor参数
    const IndexRefineSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IndexRefineSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexRefine params have incorrect type");
    }

    idx_t k_base = (params != nullptr) ? idx_t(k * params->k_factor)
                                       : idx_t(k * k_factor);

    FAISS_THROW_IF_NOT(k_base >= k);

    // 阶段1：使用base_index搜索k_base个候选
    idx_t* base_labels = labels;
    float* base_distances = distances;
    std::unique_ptr<idx_t[]> del1;
    std::unique_ptr<float[]> del2;

    if (k != k_base) {
        // 如果k_base != k，分配临时内存
        base_labels = new idx_t[n * k_base];
        del1.reset(base_labels);
        base_distances = new float[n * k_base];
        del2.reset(base_distances);
    }

    base_index->search(n, x, k_base, base_distances, base_labels, base_index_params);

    // 验证标签有效性
    for (int i = 0; i < n * k_base; i++) {
        assert(base_labels[i] >= -1 && base_labels[i] < ntotal);
    }

    // 阶段2：使用refine_index重新计算距离
#pragma omp parallel if (n > 1)
    {
        // 每个线程创建独立的DistanceComputer
        std::unique_ptr<DistanceComputer> dc(refine_index->get_distance_computer());
#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            dc->set_query(x + i * d);
            idx_t ij = i * k_base;
            for (idx_t j = 0; j < k_base; j++) {
                idx_t idx = base_labels[ij];
                if (idx < 0) {
                    break;
                }
                base_distances[ij] = (*dc)(idx);  // 重新计算精确距离
                ij++;
            }
        }
    }

    // 阶段3：排序并选择top-k
    if (metric_type == METRIC_L2) {
        typedef CMax<float, idx_t> C;
        reorder_2_heaps<C>(n, k, labels, distances, k_base, base_labels, base_distances);
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        typedef CMin<float, idx_t> C;
        reorder_2_heaps<C>(n, k, labels, distances, k_base, base_labels, base_distances);
    }
}
```

**性能优化细节**：

1. **内存重用**：如果k == k_base，直接使用输出缓冲区
2. **并行精炼**：使用OpenMP并行计算精确距离
3. **线程局部DistanceComputer**：每个线程创建独立的DistanceComputer避免竞争
4. **批量重排序**：使用reorder_2_heaps高效选择top-k

```cpp
// reorder_2_heaps的实现（faiss/utils/Heap.h）
template <class C, typename T, typename TI>
void reorder_2_heaps(
        size_t n,
        size_t k,
        TI* labels,
        T* distances,
        size_t k_base,
        const TI* base_labels,
        const T* base_distances) {

    // 对每个查询的结果进行堆排序
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        const TI* labels_i = base_labels + i * k_base;
        const T* dis_i = base_distances + i * k_base;

        // 建立堆
        heap_heapify<C>(k_base, const_cast<T*>(dis_i), const_cast<TI*>(labels_i));

        // 提取top-k
        T* output_dis = distances + i * k;
        TI* output_labels = labels + i * k;

        for (size_t j = 0; j < k; j++) {
            output_dis[j] = dis_i[j];
            output_labels[j] = labels_i[j];
        }

        heap_reorder<C>(k, output_dis, output_labels);
    }
}
```

### 10.3 IndexShards完整实现分析

```cpp
// faiss/IndexShards.cpp:47-63
// IndexShards模板实现
template <typename IndexT>
IndexShardsTemplate<IndexT>::IndexShardsTemplate(
        idx_t d,
        bool threaded,
        bool successive_ids)
        : ThreadedIndex<IndexT>(d, threaded), successive_ids(successive_ids) {
    sync_d(this);  // 对于IndexBinary，设置code_size = d / 8
}
```

**successive_ids模式**：当为true时，ID连续分配到各个shard
- shard 0: ID [0, n0)
- shard 1: ID [n0, n0 + n1)
- shard 2: ID [n0 + n1, n0 + n1 + n2)

```cpp
// faiss/IndexShards.cpp:87-110
// syncWithSubIndexes：同步子索引状态
template <typename IndexT>
void IndexShardsTemplate<IndexT>::syncWithSubIndexes() {
    if (!this->count()) {
        this->is_trained = false;
        this->ntotal = 0;
        return;
    }

    auto firstIndex = this->at(0);
    this->d = firstIndex->d;
    sync_d(this);
    this->metric_type = firstIndex->metric_type;
    this->is_trained = firstIndex->is_trained;
    this->ntotal = firstIndex->ntotal;

    // 累加所有shard的ntotal
    for (int i = 1; i < this->count(); ++i) {
        auto index = this->at(i);
        FAISS_THROW_IF_NOT(this->metric_type == index->metric_type);
        FAISS_THROW_IF_NOT(this->d == index->d);
        FAISS_THROW_IF_NOT(this->is_trained == index->is_trained);
        this->ntotal += index->ntotal;
    }
}
```

**同步验证**：确保所有shard具有相同的维度、度量类型和训练状态

```cpp
// faiss/IndexShards.cpp:136-194
// add_with_ids实现
template <typename IndexT>
void IndexShardsTemplate<IndexT>::add_with_ids(
        idx_t n,
        const component_t* x,
        const idx_t* xids) {

    FAISS_THROW_IF_NOT_MSG(
            !(successive_ids && xids),
            "It makes no sense to pass in ids and request them to be shifted");

    if (successive_ids) {
        FAISS_THROW_IF_NOT_MSG(
                this->ntotal == 0,
                "when adding to IndexShards with successive_ids, "
                "only add() in a single pass is supported");
    }

    idx_t nshard = this->count();
    const idx_t* ids = xids;

    std::vector<idx_t> aids;

    if (!ids && !successive_ids) {
        // 自动生成ID
        aids.resize(n);
        for (idx_t i = 0; i < n; i++) {
            aids[i] = this->ntotal + i;
        }
        ids = aids.data();
    }

    size_t components_per_vec =
            sizeof(component_t) == 1 ? (this->d + 7) / 8 : this->d;

    // Lambda函数：为每个shard添加数据
    auto fn = [n, ids, x, nshard, components_per_vec](int no, IndexT* index) {
        idx_t i0 = (idx_t)no * n / nshard;
        idx_t i1 = ((idx_t)no + 1) * n / nshard;
        auto x0 = x + i0 * components_per_vec;

        if (ids) {
            index->add_with_ids(i1 - i0, x0, ids + i0);
        } else {
            index->add(i1 - i0, x0);
        }
    };

    this->runOnIndex(fn);  // ThreadedIndex的并行执行
    syncWithSubIndexes();
}
```

**数据分配策略**：
- 将n个向量均匀分配到nshard个分片
- shard i接收向量范围[i*n/nshard, (i+1)*n/nshard)
- 使用runOnIndex并行执行（threaded模式）

```cpp
// faiss/IndexShards.cpp:197-265
// search实现
template <typename IndexT>
void IndexShardsTemplate<IndexT>::search(
        idx_t n,
        const component_t* x,
        idx_t k,
        distance_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    int64_t nshard = this->count();

    // 分配所有shard的结果缓冲区
    std::vector<distance_t> all_distances(nshard * k * n);
    std::vector<idx_t> all_labels(nshard * k * n);
    std::vector<int64_t> translations(nshard, 0);

    // 计算successive_ids的ID偏移
    if (successive_ids) {
        translations[0] = 0;
        for (int s = 0; s + 1 < nshard; s++) {
            translations[s + 1] = translations[s] + this->at(s)->ntotal;
        }
    }

    // Lambda函数：搜索每个shard
    auto fn = [n, k, x, params, &all_distances, &all_labels, &translations](
                      int no, const IndexT* index) {
        index->search(
                n, x, k,
                all_distances.data() + no * k * n,
                all_labels.data() + no * k * n,
                params);

        // 转换标签（successive_ids模式）
        translate_labels(n * k, all_labels.data() + no * k * n, translations[no]);
    };

    this->runOnIndex(fn);

    // 合并所有shard的结果
    if (this->metric_type == METRIC_L2) {
        merge_knn_results<idx_t, CMin<distance_t, int>>(
                n, k, nshard,
                all_distances.data(), all_labels.data(),
                distances, labels);
    } else {
        merge_knn_results<idx_t, CMax<distance_t, int>>(
                n, k, nshard,
                all_distances.data(), all_labels.data(),
                distances, labels);
    }
}
```

**结果合并策略**：
1. **并行搜索**：所有shard并行搜索相同的查询
2. **ID转换**：successive_ids模式下，转换每个shard的ID偏移
3. **堆合并**：使用merge_knn_results从nshard * k个候选中选择top-k

```cpp
// merge_knn_results的实现（faiss/utils/Heap.h）
template <typename TI, typename C>
void merge_knn_results(
        size_t n,
        size_t k,
        size_t nshard,
        const float* all_distances,
        const TI* all_labels,
        float* distances,
        TI* labels) {

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        // 初始化堆
        heap_heapify<C>(k, distances + i * k, labels + i * k);

        // 合并所有shard的结果
        for (size_t s = 0; s < nshard; s++) {
            const float* dis_s = all_distances + s * k * n + i * k;
            const TI* lbl_s = all_labels + s * k * n + i * k;

            for (size_t j = 0; j < k; j++) {
                if (C::cmp(dis_s[j], distances[i * k])) {
                    heap_replace_top<C>(k, distances + i * k, labels + i * k,
                                       dis_s[j], lbl_s[j]);
                }
            }
        }

        heap_reorder<C>(k, distances + i * k, labels + i * k);
    }
}
```

### 10.4 IndexReplicas完整实现分析

```cpp
// faiss/IndexReplicas.cpp:97-112
// add实现：向所有副本添加相同数据
template <typename IndexT>
void IndexReplicasTemplate<IndexT>::add(idx_t n, const component_t* x) {
    auto fn = [n, x](int i, IndexT* index) {
        index->add(n, x);  // 每个副本添加相同的数据
    };

    this->runOnIndex(fn);
    syncWithSubIndexes();
}
```

**数据复制**：所有副本存储相同的数据，提供查询并行能力

```cpp
// faiss/IndexReplicas.cpp:123-174
// search实现：查询并行
template <typename IndexT>
void IndexReplicasTemplate<IndexT>::search(
        idx_t n,
        const component_t* x,
        idx_t k,
        distance_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    FAISS_THROW_IF_NOT_MSG(!params, "search params not supported");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(this->count() > 0, "no replicas in index");

    if (n == 0) {
        return;
    }

    auto dim = this->d;
    size_t componentsPerVec = sizeof(component_t) == 1 ? (dim + 7) / 8 : dim;

    // 将查询分配到不同副本
    faiss::idx_t queriesPerIndex =
            (faiss::idx_t)(n + this->count() - 1) / (faiss::idx_t)this->count();

    auto fn = [queriesPerIndex, componentsPerVec, n, x, k, distances, labels](
                      int i, const IndexT* index) {
        faiss::idx_t base = (faiss::idx_t)i * queriesPerIndex;

        if (base < n) {
            auto numForIndex = std::min(queriesPerIndex, n - base);
            index->search(
                    numForIndex,
                    x + base * componentsPerVec,
                    k,
                    distances + base * k,
                    labels + base * k);
        }
    };

    this->runOnIndex(fn);
}
```

**查询并行策略**：
- 每个副本处理不同的查询子集
- replica 0: 查询 [0, queriesPerIndex)
- replica 1: 查询 [queriesPerIndex, 2 * queriesPerIndex)
- 无需结果合并（每个查询只在一个副本上执行）

### 10.5 IndexIDMap完整实现分析

```cpp
// faiss/IndexIDMap.cpp:51-59
// IndexIDMap构造函数
template <typename IndexT>
IndexIDMapTemplate<IndexT>::IndexIDMapTemplate(IndexT* index) : index(index) {
    FAISS_THROW_IF_NOT_MSG(index->ntotal == 0, "index must be empty on input");
    this->is_trained = index->is_trained;
    this->metric_type = index->metric_type;
    this->verbose = index->verbose;
    this->d = index->d;
    sync_d(this);
}
```

**初始化约束**：底层索引必须为空，确保ID映射的一致性

```cpp
// faiss/IndexIDMap.cpp:62-78
// add操作被禁用，必须使用add_with_ids
template <typename IndexT>
void IndexIDMapTemplate<IndexT>::add(
        idx_t,
        const typename IndexT::component_t*) {
    FAISS_THROW_MSG(
            "add does not make sense with IndexIDMap, "
            "use add_with_ids");
}
```

**设计约束**：强制使用add_with_ids，确保ID映射的完整性

```cpp
// faiss/IndexIDMap.cpp:107-129
// add_with_ids实现
template <typename IndexT>
void IndexIDMapTemplate<IndexT>::add_with_ids(
        idx_t n,
        const typename IndexT::component_t* x,
        const idx_t* xids) {
    // 向底层索引添加（内部ID = 0, 1, 2, ..., n-1）
    index->add(n, x);

    // 更新ID映射
    for (idx_t i = 0; i < n; i++) {
        id_map.push_back(xids[i]);
    }

    this->ntotal = index->ntotal;
}
```

**ID映射实现**：
- id_map是std::vector<idx_t>，存储外部ID
- id_map[internal_id] = external_id
- 内部ID连续：0, 1, 2, ..., ntotal-1

```cpp
// faiss/IndexIDMap.cpp:170-248
// search实现
template <typename IndexT>
void IndexIDMapTemplate<IndexT>::search(
        idx_t n,
        const component_t* x,
        idx_t k,
        distance_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    // 在底层索引搜索（返回内部ID）
    index->search(n, x, k, distances, labels, params);

    // 转换为外部ID
    for (idx_t i = 0; i < n * k; i++) {
        idx_t internal_id = labels[i];
        if (internal_id < 0) {
            continue;  // 保留-1（无效ID）
        }
        labels[i] = id_map[internal_id];
    }
}
```

**ID转换**：搜索结果从内部ID转换为外部ID

### 10.6 性能对比表

| 复合索引类型 | 并行模式 | 内存占用 | 搜索延迟 | 吞吐量 | 适用场景 |
|-------------|---------|---------|---------|--------|----------|
| **IndexPreTransform** | 无（顺序） | 1x | 中等 | 低 | 降维、预变换 |
| **IndexRefine** | 无（顺序） | 2x | 高 | 低 | 高精度要求 |
| **IndexShards** | 数据并行 | 1x | 低 | 高（nshard倍） | 大数据集、高吞吐量 |
| **IndexReplicas** | 查询并行 | nreplicas倍 | 低 | 高（nreplicas倍） | 高并发查询 |
| **IndexIDMap** | 无 | 1x + 映射表 | 低 | 低 | 自定义ID需求 |

### 10.7 并行执行模型

```cpp
// ThreadedIndex::runOnIndex的实现（faiss/utils/WorkerThread.h）
template <typename IndexT>
template <typename Fn>
void ThreadedIndex<IndexT>::runOnIndex(Fn fn) const {
    if (!this->threaded) {
        // 单线程模式
        for (int i = 0; i < this->count(); i++) {
            fn(i, this->at(i));
        }
    } else {
        // 多线程模式：使用WorkerThread池
        std::vector<std::future<void>> futures;
        for (int i = 0; i < this->count(); i++) {
            auto f = this->workerThreadPool->enqueue([fn, i, this]() {
                fn(i, this->at(i));
            });
            futures.push_back(std::move(f));
        }

        // 等待所有任务完成
        for (auto& f : futures) {
            f.get();
        }
    }
}
```

**线程池实现**：
- 使用C++11 std::future和std::async
- 每个索引在一个线程中执行
- 等待所有任务完成（同步屏障）

---

## 11. WorkerThread完整底层实现

### 11.1 WorkerThread设计目的

WorkerThread是Faiss中用于异步执行任务的线程类，主要用于：
- **IndexShards/IndexReplicas**：并行执行多个子索引的操作
- **异步任务队列**：支持Lambda函数的异步执行
- **线程安全关闭**：优雅地停止线程，刷新所有待处理任务

### 11.2 WorkerThread完整结构

```cpp
// faiss/utils/WorkerThread.h:18-59
class WorkerThread {
   public:
    WorkerThread();

    // 析构函数：停止线程并等待退出
    ~WorkerThread();

    // 请求线程停止
    void stop();

    // 阻塞等待线程退出
    void waitForThreadExit();

    // 添加Lambda函数到工作线程，返回future用于阻塞等待完成
    std::future<bool> add(std::function<void()> f);

   private:
    void startThread();
    void threadMain();
    void threadLoop();

    // 执行Lambda的工作线程
    std::thread thread_;

    // 队列和退出状态的互斥锁
    std::mutex mutex_;

    // 监视退出状态和队列的条件变量
    std::condition_variable monitor_;

    // 是否希望线程退出
    bool wantStop_;

    // 待调用的Lambda队列（包含promise）
    std::deque<std::pair<std::function<void()>, std::promise<bool>>> queue_;
};
```

### 11.3 构造函数与线程启动

```cpp
// faiss/utils/WorkerThread.cpp:28-33
WorkerThread::WorkerThread() : wantStop_(false) {
    startThread();

    // 确保线程已启动后才继续
    add([]() {}).get();
}
```

**启动线程**：

```cpp
// faiss/utils/WorkerThread.cpp:40-42
void WorkerThread::startThread() {
    thread_ = std::thread([this]() { threadMain(); });
}
```

**线程主函数**：

```cpp
// faiss/utils/WorkerThread.cpp:75-85
void WorkerThread::threadMain() {
    threadLoop();

    // 调用所有待处理任务
    FAISS_ASSERT(wantStop_);

    // 刷新所有待处理操作
    for (auto& f : queue_) {
        runCallback(f.first, f.second);
    }
}
```

### 11.4 任务执行循环

```cpp
// faiss/utils/WorkerThread.cpp:87-108
void WorkerThread::threadLoop() {
    while (true) {
        std::pair<std::function<void()>, std::promise<bool>> data;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            // 等待直到有任务或需要退出
            while (!wantStop_ && queue_.empty()) {
                monitor_.wait(lock);
            }

            // 检查是否需要退出
            if (wantStop_) {
                return;
            }

            // 取出队列头部任务
            data = std::move(queue_.front());
            queue_.pop_front();
        }

        // 执行任务（解锁状态下）
        runCallback(data.first, data.second);
    }
}
```

**关键设计点**：
1. **条件变量等待**：`monitor_.wait(lock)` 在队列为空时阻塞
2. **双重检查**：取出任务前再次检查 `wantStop_`
3. **缩小锁范围**：任务执行时释放锁，提高并发性

### 11.5 添加任务（add）

```cpp
// faiss/utils/WorkerThread.cpp:51-73
std::future<bool> WorkerThread::add(std::function<void()> f) {
    std::lock_guard<std::mutex> guard(mutex_);

    // 如果线程已停止，返回已完成的future（值为false）
    if (wantStop_) {
        std::promise<bool> p;
        auto fut = p.get_future();
        p.set_value(false);  // 标记为未执行
        return fut;
    }

    // 创建promise和future
    auto pr = std::promise<bool>();
    auto fut = pr.get_future();

    // 将任务和promise加入队列
    queue_.emplace_back(std::make_pair(std::move(f), std::move(pr)));

    // 唤醒工作线程
    monitor_.notify_one();
    return fut;
}
```

**使用流程**：
1. 调用者调用 `add(lambda)` 获取 `std::future<bool>`
2. Lambda被加入队列
3. 工作线程被唤醒并执行Lambda
4. Promise被设置为 `true`（已执行）或异常
5. 调用者可以通过 `future.get()` 等待完成

### 11.6 异常处理

```cpp
// faiss/utils/WorkerThread.cpp:17-24
namespace {

// 捕获Lambda抛出的任何异常并通过promise返回
void runCallback(std::function<void()>& fn, std::promise<bool>& promise) {
    try {
        fn();
        promise.set_value(true);  // 标记为成功执行
    } catch (...) {
        promise.set_exception(std::current_exception());  // 传递异常
    }
}

} // namespace
```

**异常处理机制**：
- Lambda中的异常被 `catch(...)` 捕获
- 异常通过 `set_exception` 传递给 future
- 调用者可以通过 `future.get()` 重新抛出异常

### 11.7 停止线程（stop）

```cpp
// faiss/utils/WorkerThread.cpp:44-49
void WorkerThread::stop() {
    std::lock_guard<std::mutex> guard(mutex_);

    wantStop_ = true;      // 设置退出标志
    monitor_.notify_one();  // 唤醒工作线程
}
```

**停止流程**：
1. 设置 `wantStop_ = true`
2. 唤醒线程（如果正在等待）
3. 线程检查到 `wantStop_` 后退出循环
4. 执行剩余的队列任务（在 `threadMain` 中）

### 11.8 等待线程退出（waitForThreadExit）

```cpp
// faiss/utils/WorkerThread.cpp:110-115
void WorkerThread::waitForThreadExit() {
    try {
        thread_.join();  // 等待线程结束
    } catch (...) {
        // 捕获join可能抛出的异常
    }
}
```

**析构函数调用**：

```cpp
// faiss/utils/WorkerThread.cpp:35-38
WorkerThread::~WorkerThread() {
    stop();               // 请求停止
    waitForThreadExit();  // 等待退出
}
```

### 11.9 使用示例

```cpp
// 示例1：基本使用
void example_worker_thread() {
    WorkerThread worker;

    // 添加任务
    auto future1 = worker.add([]() {
        printf("Task 1\n");
    });

    auto future2 = worker.add([]() {
        printf("Task 2\n");
    });

    // 等待任务完成
    bool result1 = future1.get();  // true
    bool result2 = future2.get();  // true

    printf("Both tasks completed\n");
}

// 示例2：IndexShards中的使用
template <typename IndexT>
void IndexShardsTemplate<IndexT>::search(...) {
    auto fn = [n, k, x, params, &all_distances, &all_labels](
                      int no, const IndexT* index) {
        index->search(n, x, k,
                     all_distances.data() + no * k * n,
                     all_labels.data() + no * k * n,
                     params);
    };

    // 在每个shard的WorkerThread上执行
    this->runOnIndex(fn);
}

// 示例3：异常处理
void example_exception_handling() {
    WorkerThread worker;

    auto future = worker.add([]() {
        throw std::runtime_error("Task failed!");
    });

    try {
        bool result = future.get();  // 重新抛出异常
    } catch (const std::runtime_error& e) {
        printf("Caught exception: %s\n", e.what());
    }
}

// 示例4：优雅停止
void example_graceful_shutdown() {
    WorkerThread worker;

    // 添加任务
    for (int i = 0; i < 10; i++) {
        worker.add([i]() {
            printf("Task %d\n", i);
        });
    }

    // 析构函数会自动停止线程并执行所有任务
    // ~WorkerThread() 被调用
}
```

### 11.10 ThreadedIndex基类

```cpp
// faiss/IndexShards.h (简化版)
template <typename IndexT>
class ThreadedIndex : public std::vector<IndexT*> {
   public:
    bool threaded;  // 是否使用多线程
    std::vector<std::unique_ptr<WorkerThread>> workerThreadPool;

    ThreadedIndex(idx_t d, bool threaded)
        : std::vector<IndexT*>(), threaded(threaded) {
        if (threaded) {
            // 为每个shard/replica创建WorkerThread
            workerThreadPool.resize(10);  // 最多10个工作线程
        }
    }

    // 在每个索引上执行函数
    template <typename Fn>
    void runOnIndex(Fn fn) const {
        if (!this->threaded) {
            // 单线程：顺序执行
            for (int i = 0; i < this->count(); i++) {
                fn(i, this->at(i));
            }
        } else {
            // 多线程：并行执行
            std::vector<std::future<void>> futures;

            for (int i = 0; i < this->count(); i++) {
                // 在WorkerThread上执行
                auto future = workerThreadPool[i % workerThreadPool.size()]->add(
                    [fn, i, this]() {
                        fn(i, this->at(i));
                    });
                futures.push_back(std::move(future));
            }

            // 等待所有任务完成
            for (auto& f : futures) {
                f.get();
            }
        }
    }

    virtual ~ThreadedIndex() {
        // 清理所有索引
        for (int i = 0; i < this->size(); i++) {
            delete (*this)[i];
        }
    }
};
```

### 11.11 性能特性表

| 特性 | 实现方式 | 性能特点 |
|------|---------|---------|
| 任务队列 | std::deque | O(1)插入和删除 |
| 线程同步 | std::mutex + condition_variable | 惊惊效应最小化 |
| 异常传递 | std::promise.set_exception | 完整异常信息保留 |
| 优雅停止 | 刷新队列 | 无任务丢失 |
| 线程复用 | 单工作线程 | 避免线程创建开销 |

### 11.12 设计技巧总结

1. **RAII管理**：析构函数自动停止线程
2. **异常安全**：异常通过promise传递给调用者
3. **条件变量优化**：只在必要时唤醒线程
4. **缩小锁范围**：任务执行时不持有锁
5. **状态检查**：添加任务前检查线程状态

---

## 12. 第13天总结

### 关键概念

1. **IndexPreTransform**：预变换（PCA、OPQ等）
2. **IndexRefine**：两阶段搜索（粗糙+精炼）
3. **IndexShards**：索引分片并行
4. **IndexReplicas**：索引副本并行
5. **IndexIDMap**：外部ID到内部ID映射
6. **Index2Layer**：两层索引结构
3. **IndexShards**：数据并行
4. **IndexReplicas**：查询并行
5. **IndexIDMap**：自定义ID映射
6. **Index2Layer**：两层索引结构
7. **Panorama**：渐进式剪枝

### 组合策略

| 需求 | 推荐组合 |
|------|-----------|
| 内存受限 | IVF + PQ + Refine |
| 高吞吐量 | Shards 或 Replicas |
| 高精度 | IVF + Flat 或 HNSW |
| 自定义ID | IndexIDMap + 任何索引 |

### 下一步

第14天将总结**性能优化与最佳实践**。

---

## 练习题

1. 实现简单的PCA降维索引
2. 实现IndexRefine
3. 创建并行的分片索引
4. 实现IndexIDMap

## 扩展阅读

- faiss/IndexPreTransform.h - 预变换索引
- faiss/IndexRefine.h - 精炼索引
- faiss/IndexShards.h - 分片索引
- faiss/IndexReplicas.h - 副本索引
- faiss/IndexIDMap.h - ID映射
