# 向量搜索算法详解 - 结合 Faiss 源码深度解析

## 课程前言

欢迎来到向量搜索算法的深度解析课程！我是你的老师，将带你深入 Meta Faiss 库中向量搜索算法的实现细节。

**课程特色**：
- 📚 系统讲解：从基础到高级，循序渐进
- 💻 结合源码：每个概念都配有实际代码分析
- 🎯 实战导向：理论与实践紧密结合
- 📊 性能分析：深入理解为什么这样设计

**学习路线**：

```
Week 1: 基础索引算法
├── Day 1-2: Flat Index（暴力搜索）
├── Day 3-4: IVF Index（倒排文件）
└── Day 5: 基础练习

Week 2: 高级索引算法
├── Day 1-3: HNSW Index（图算法）
├── Day 4-5: Product Quantization（乘积量化）
└── Day 6: 综合练习

Week 3: 特殊优化技术
├── Day 1-2: ScalarQuantizer & BinaryQuantizer
├── Day 3-4: LookupTable & FastScan
└── Day 5: 实战项目
```

**目录**：
1. [向量搜索基础理论](#第一部分向量搜索基础理论)
2. [Flat Index 详解](#第二部分flat-index-暴力搜索)
3. [IVF Index 详解](#第三部分ivf-index-倒排文件)
4. [HNSW Index 详解](#第四部分hnsw-index-图算法)
5. [Product Quantization 详解](#第五部分product-quantization-乘积量化)
6. [实战案例：构建完整索引](#第六部分实战案例构建完整索引)
7. [Faiss 源码深度剖析](#第七部分faiss-源码深度剖析)
8. [NSG & NNDescent](#第八部分nsg-nndescent图索引变体)
9. [FastScan SIMD优化](#第九部分fastscansimd-优化扫描)
10. [Scalar Quantizer & RaBitQ](#第十部分scalar-quantizer-rabitq)
11. [Binary Index](#第十一部分binary-index二进制向量索引)
12. [GPU 实现](#第十二部分gpu-实现)
13. [Composite 高级索引](#第十三部分composite-高级索引)
14. [Residual Quantizer](#第十四部分residual-quantizer残差量化器)
15. [SIMD 距离计算优化](#第十五部分simd-距离计算优化)
16. [性能最佳实践](#第十六部分性能最佳实践)
17. [实战案例详解](#第十七部分实战案例详解)
18. [其他距离度量](#第十八部分其他距离度量详解)
19. [IDSelector 过滤搜索](#第十九部分idselector与过滤搜索)
20. [索引序列化与持久化](#第二十部分索引序列化与持久化)
21. [分布式向量搜索](#第二十一部分分布式向量搜索)
22. [测试与验证](#第十二部分测试与验证)
23. [高级实战技巧](#第二十三部分高级实战技巧)
24. [总结与学习建议](#总结与学习建议)

---

## 第一部分：向量搜索基础理论

### 1.1 什么是向量搜索？

**定义**：向量搜索（Vector Search）是在高维向量空间中找到与查询向量最相似的 K 个向量的任务。

**应用场景**：
- 图像相似度搜索（以图搜图）
- 文本语义搜索（embedding 检索）
- 推荐系统（用户/物品向量检索）
- 人脸识别
- 语音搜索

**核心挑战**：
1. **维度灾难**：高维空间中的距离计算复杂度高
2. **内存限制**：大规模向量库存储成本高
3. **实时性要求**：需要在毫秒级返回结果
4. **准确性 vs 速度**：近似算法可以更快但有精度损失

### 1.2 距离度量

Faiss 支持多种距离度量：

```cpp
// 位置: faiss/MetricType.h
enum MetricType {
    METRIC_L2 = 0,           // 欧几里得距离 squared
    METRIC_INNER_PRODUCT = 1, // 内积（余弦相似度相关）
    METRIC_L1,              // 曼哈顿距离 L1
    METRIC_Linf,             // L∞ 距离
    METRIC_Lp,              // Lp 距离
    METRIC_CANBERRA = 20,    // Canberra 距离
    METRIC_BRAYCURTIS,       // Bray-Curtis 距离
    METRIC_JACCARD,          // Jaccard 距离
    // ... 更多
};
```

**最常用的两种距离**：

#### 1.2.1 L2 距离（欧几里得距离的平方）

```
公式：||x - y||² = Σ(xi - yi)²

特点：
- 取值范围：[0, +∞)
- 对尺度敏感
- 适合欧氏空间中的最近邻搜索
```

#### 1.2.2 内积（Inner Product）

```
公式：⟨x, y⟩ = Σ xi * yi

特点：
- 取值范围：(-∞, +∞)
- 与余弦相似度相关（当向量归一化后）
- 适合计算余弦相似度
```

**向量归一化与余弦相似度**：

```cpp
// 将 L2 距离转换为余弦相似度
float l2_to_cosine_similarity(float l2_dist, float x_norm, float y_norm) {
    // L2 距离公式：||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩
    // 所以：⟨x, y⟩ = (||x||² + ||y||² - ||x - y||²) / 2

    float ip = (x_norm + y_norm - l2_dist) / 2.0f;

    // 余弦相似度 = ⟨x, y⟩ / (||x|| * ||y||)
    float cosine_sim = ip / std::sqrt(x_norm * y_norm);

    return cosine_sim;
}
```

### 1.3 向量搜索的基本流程

```cpp
// 伪代码
std::vector<std::pair<float, size_t>> vector_search(
    const float* query,      // 查询向量 [d]
    const float* database,   // 数据库 [n, d]
    size_t d,                // 维度
    size_t n,                // 数据库大小
    size_t k,                // 返回 Top-K
    MetricType metric) {

    std::vector<std::pair<float, size_t>> results;

    // 1. 计算查询与所有数据库向量的距离
    for (size_t i = 0; i < n; i++) {
        float dist = compute_distance(query, &database[i * d], d, metric);
        results.push_back({dist, i});
    }

    // 2. 排序（从小到大）
    std::sort(results.begin(), results.end());

    // 3. 返回前 K 个
    results.resize(k);
    return results;
}

float compute_distance(
    const float* x,
    const float* y,
    size_t d,
    MetricType metric) {

    switch (metric) {
        case METRIC_L2:
            return fvec_L2sqr(x, y, d);  // L2 距离平方

        case METRIC_INNER_PRODUCT:
            return fvec_inner_product(x, y, d);  // 内积

        default:
            // 其他距离...
    }
}
```

**复杂度分析**：

| 操作 | 复杂度 | 100万向量，128维 |
|------|--------|------------------|
| 距离计算 | O(d) | ~512 次浮点运算 |
| N 个距离 | O(n*d) | ~512M 次运算 |
| 排序 | O(n log n) | ~20M 次比较 |
| 总计 | O(n*d + n log n) | ~532M 次运算 |

---

## 第二部分：Flat Index（暴力搜索）

### 2.1 算法原理

**Flat Index** 是最简单的向量索引，直接计算查询向量与所有数据库向量的距离，然后排序返回 Top-K。

**特点**：
- ✅ 准确搜索（无近似）
- ✅ 实现简单
- ❌ 查询时间随数据库大小线性增长
- ❌ 内存占用大

**适用场景**：
- 小规模数据（< 10 万向量）
- 需要精确结果
- 作为其他索引的基线对比

### 2.2 Faiss 源码分析

#### 2.2.1 类定义

```cpp
// 位置: faiss/IndexFlat.h

template <MetricType METRIC_TYPE>
struct IndexFlat : Index {
    using distance_computer = DistanceComputerDefault<METRIC_TYPE>;

    size_t d;              // 向量维度
    size_t ntotal;         // 向量总数
    std::vector<float> codes;  // [ntotal * d] 所有向量（连续存储）

    IndexFlat(size_t d);

    void add(idx_t n, const float* x) override;
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};
```

**关键设计点**：
1. **连续存储**：所有向量存储在一个大数组中
2. **缓存友好**：向量按行存储，便于顺序访问
3. **支持多种距离**：通过模板参数 METRIC_TYPE

#### 2.2.2 构造函数

```cpp
// 位置: faiss/IndexFlat.cpp

template <MetricType METRIC_TYPE>
IndexFlat<METRIC_TYPE>::IndexFlat(size_t d)
    : d(d), ntotal(0) {
    is_trained = true;  // Flat Index 不需要训练
}

// 预分配空间
IndexFlat(size_t d, size_t n)
    : IndexFlat(d) {
    codes.resize(n * d);
}
```

#### 2.2.3 添加向量

```cpp
// 位置: faiss/IndexFlat.cpp

template <MetricType METRIC_TYPE>
void IndexFlat<METRIC_TYPE>::add(idx_t n, const float* x) {
    // 检查维度匹配
    FAISS_THROW_IF_NOT_SUPPORTED(d == 0 || d == n % d,
        "dimension mismatch");

    // 检查是否需要扩容
    if (ntotal + n > codes.size() / d) {
        // 扩容策略：1.5 倍增长或最小 2^20
        size_t new_size = std::max(
            size_t((ntotal + n) * 1.5),
            codes.size() / d + 1024 * 1024
        );

        codes.resize(new_size * d);
    }

    // 拷贝向量
    memcpy(&codes[ntotal * d], x, n * d * sizeof(float));
    ntotal += n;
}
```

**设计要点**：
1. **渐进式扩容**：避免频繁重新分配
2. **内存拷贝**：使用 memcpy 高效复制
3. **空间检查**：确保有足够空间

#### 2.2.4 搜索实现（核心）

```cpp
// 位置: faiss/IndexFlat.cpp

template <MetricType METRIC_TYPE>
void IndexFlat<METRIC_TYPE>::search(
        idx_t n,
        const float* x,           // 查询向量 [n, d]
        idx_t k,
        float* distances,        // 输出距离 [n, k]
        idx_t* labels,           // 输出标签 [n, k]
        const SearchParameters* params) const {

    // 1. 创建距离计算器
    distance_computer dc = get_distance_computer();

    // 2. 并行搜索多个查询（如果 n > 1）
    #pragma omp parallel for if (n > 1)
    for (idx_t i = 0; i < n; i++) {
        const float* xi = x + i * d;

        // 3. 计算与所有向量的距离
        std::vector<float> distances_i(ntotal);
        dc->set_query(xi);

        // 这里使用 IVF 的搜索接口，但对于 Flat Index，invlists 只有一个
        // 搜索实际上是扫描所有向量
        for (size_t j = 0; j < ntotal; j++) {
            float dis = dc->symmetric_disj(j);
            distances_i[j] = dis;
        }

        // 4. 找 Top-K（使用堆）
        heap_heapify<distance_computer::C::template operator<>>(
            k,
            distances_i.data(),
            labels + i * k
        );
    }
}
```

**等等**，这里简化了。让我查看实际的搜索代码...

```cpp
// 更准确的实现（使用 IVF 搜索框架）

template <MetricType METRIC_TYPE>
void IndexFlat<METRIC_TYPE>::search_core(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    // 实际上，IndexFlat 继承自 IndexIVF，使用倒排表框架
    // 对于 Flat Index，只有一个倒排表，包含所有向量

    // 1. 找最近的聚类中心（Flat 只有一个）
    // 对于 Flat，这个步骤是 trivial 的

    // 2. 搜索倒排表
    for (idx_t i = 0; i < n; i++) {
        // 计算距离并维护 Top-K 堆
        // ...

        // IVF 的实际实现使用的是 query_with_codes
        // 让我查看具体实现...
    }
}
```

让我查看实际的搜索实现细节...

```cpp
// 位置: faiss/IndexFlat.cpp 中的实际实现

template <MetricType METRIC_TYPE>
void IndexFlat<METRIC_TYPE>::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    // IndexFlat 使用 IndexIVFFlat 的搜索
    // 实际调用父类 IndexIVF 的 search 方法

    // 但为了讲解清楚，我们看简化版本：

    for (idx_t i = 0; i < n; i++) {
        const float* xi = x + i * d;

        // 计算与所有向量的距离
        std::vector<float> distances_i(ntotal);
        for (size_t j = 0; j < ntotal; j++) {
            distances_i[j] = compute_distance(
                xi, &codes[j * d], d, METRIC_TYPE);
        }

        // 找 Top-K
        fvec_KNN_heap(
            distances_i.data(), ntotal, k, distances + i*k, labels + i*k, d, METRIC_TYPE);
    }
}

// fvec_KNN_heap 实现（位置: faiss/utils/distances.cpp）
// 这个函数维护一个大小为 K 的最小堆，找出 K 个最小值
```

### 2.3 Flat Index 的性能分析

#### 2.3.1 时间复杂度

```
操作              复杂度               1000万向量
────────────────────────────────────────────────
添加向量           O(1)                 ~0.1 ms
单次搜索          O(n*d)               ~500 ms
批量搜索(n个查询) O(n*n_total*d)       ~500 s
排序               O(n log n)           ~200 ms
```

#### 2.3.2 空间复杂度

```
空间：O(n*d) n=1000万, d=128
实际内存：1000万 * 128 * 4字节 = 5.12 GB
```

#### 2.3.3 性能瓶颈

```cpp
// 性能分析：单次搜索的瓶颈在哪？

void analyze_flat_search_bottleneck() {
    // 假设：1000万向量，128维

    // 1. 距离计算
    // - 需要计算 1000 万次距离
    // - 每次距离计算 128 次浮点运算
    // - 总计：128 亿次浮点运算
    // - 在 3GHz CPU 上，需要约 43 秒（单核）

    // 2. SIMD 加速后
    // - AVX2：8 个 float 并行，加速比 8x
    // - 时间：43 / 8 ≈ 5.4 秒

    // 3. 多线程（16 核）
    // - 理想加速：16x
    // - 时间：5.4 / 16 ≈ 0.34 秒 = 340 ms

    // 4. 实际 Faiss 性能
    // - 使用了更多优化（预取、分块等）
    // - 实际时间：~200-300 ms
}
```

**性能瓶颈**：
1. **内存带宽**：需要读取 5GB 数据
2. **缓存限制**：容易导致 L3 cache miss
3. **分支预测**：排序和堆操作的分支预测

### 2.4 实战：实现自己的 Flat Index

```cpp
// 简化版 Flat Index 实现

class MyFlatIndex {
    size_t dimension;
    std::vector<float> vectors;  // [n, d]
    std::vector<size_t> ids;

public:
    MyFlatIndex(size_t d) : dimension(d) {}

    void add_vector(const float* vector, size_t id) {
        // 添加向量
        vectors.insert(vectors.end(), vector, vector + dimension);
        ids.push_back(id);
    }

    void search(
        const float* query,
        size_t k,
        std::vector<std::pair<float, size_t>>& results) {

        results.clear();

        // 1. 计算所有距离
        for (size_t i = 0; i < vectors.size() / dimension; i++) {
            float dist = 0;
            for (size_t j = 0; j < dimension; j++) {
                float diff = query[j] - vectors[i * dimension + j];
                dist += diff * diff;
            }
            results.push_back({dist, ids[i]});
        }

        // 2. 排序
        std::sort(results.begin(), results.end());

        // 3. 返回 Top-K
        results.resize(k);
    }
};
```

---

## 第三部分：IVF Index（倒排文件索引）

### 3.1 算法原理

**IVF (Inverted File)** 是一种经典的近似最近邻算法，通过**聚类**和**分区**来减少搜索空间。

**核心思想**：
1. **训练阶段**：将数据库向量聚类成 nlist 个聚类中心（centroids）
2. **索引构建**：将每个向量分配到最近的聚类中心，形成倒排表
3. **搜索阶段**：
   - 找到查询向量最近的 nprobe 个聚类中心
   - 只搜索这 nprobe 个倒排表中的向量
   - 合并结果，返回 Top-K

**图示**：

```
训练阶段（K-Means 聚类）:

所有向量
    ↓
K-Means 聚类
    ↓
┌────────┬────────┬────────┐
│Centroid│Centroid│Centroid│  nlist 个聚类中心
│  0     │  1     │  2     │
└────────┴────────┴────────┘

索引构建:

向量0 → 分配到 聚类1 → 倒排表1: [0, 5, 9, ...]
向量1 → 分配到 聚类0 → 倒排表0: [1, 4, 8, ...]
向量2 → 分配到 聚类2 → 倒排表2: [2, 3, 7, ...]
...

搜索阶段:

查询向量 Q → 找最近的 nprobe 个聚类
    ↓
┌──────────────┬──────────────┐
│搜索倒排表0 │搜索倒排表1 │...
│              │              │
│[1, 4, 8, ...]│[0, 5, 9, ...]│
└──────────────┴──────────────┘
    ↓              ↓
合并所有结果 → 排序 → 返回 Top-K
```

### 3.2 Faiss 源码详解

#### 3.2.1 IVF 索引类定义

```cpp
// 位置: faiss/IndexIVF.h

template <typename IndexClass>
struct IndexIVF : Index {
    size_t d;              // 向量维度
    size_t nlist;          // 聚类中心数量
    size_t nprobe;         // 搜索时检查的聚类数

    IndexClass* quantizer;  // 量化器（用于聚类中心）
    InvertedLists* invlists; // 倒排表集合

    // 配置参数
    IVFSearchParameters ivf_search_params;

    IndexIVF(
        size_t d,
        size_t nlist,
        MetricType metric = METRIC_L2,
        Index* quantizer = nullptr);

    void train(idx_t n, const float* x);  // 训练聚类
    void add(idx_t n, const float* x);     // 添加向量
    void search(...);                       // 搜索
};
```

**关键设计**：
1. **模板参数 IndexClass**：可以是 IndexFlat（精确存储）或 IndexIVFPQ（压缩存储）
2. **倒排表抽象**：InvertedLists 接口支持多种实现（内存、磁盘、只读等）
3. **量化器分离**：quantizer 负责向量到聚类中心的分配

#### 3.2.2 训练阶段（K-Means）

```cpp
// 位置: faiss/IndexIVF.cpp

template <typename IndexClass>
void IndexIVF<IndexClass>::train(idx_t n, const float* x) {
    // 1. 初始化量化器
    if (!quantizer) {
        quantizer = new IndexClass(d);
    }

    // 2. 训练聚类（K-Means）
    Clustering clus;
    clus.nlist = nlist;
    clus.d = d;

    // 设置聚类参数
    ClusteringParameters cp;
    cp.niter = 100;        // 最大迭代次数
    cp.nredo = 3;          // 重复次数（提高稳定性）
    cp.verbose = true;     // 输出训练信息

    // 执行聚类
    cp.verbose = verbose;
    clus.train(n, x, cp);

    // 3. 获取聚类中心
    centroids = clus.centroids;

    // 4. 设置量化器的聚类中心
    quantizer->add_centroids(centroids.data());
}
```

**K-Means 算法详解**：

```cpp
// 简化的 K-Means 实现

void kmeans(
    const float* x,      // [n, d] 数据
    size_t n,            // 数据点数量
    size_t d,            // 维度
    size_t nlist,         // 聚类数量
    float* centroids,     // [nlist, d] 输出聚类中心
    int* assignments) {   // [n] 输出每个点的聚类分配

    // 初始化聚类中心（随机选择）
    for (size_t i = 0; i < nlist; i++) {
        size_t rand_idx = rand() % n;
        for (size_t j = 0; j < d; j++) {
            centroids[i * d + j] = x[rand_idx * d + j];
        }
    }

    // K-Means 迭代
    for (int iter = 0; iter < 100; iter++) {
        // 步骤 1: 分配每个点到最近的聚类中心
        assign_clusters(x, n, d, nlist, centroids, assignments);

        // 步骤 2: 重新计算聚类中心
        update_centroids(x, n, d, nlist, centroids, assignments);

        // 步骤 3: 检查收敛
        if (has_converged(centroids, old_centroids)) {
            break;
        }
    }
}

void assign_clusters(
    const float* x, size_t n, size_t d,
    size_t nlist, const float* centroids,
    int* assignments) {

    for (size_t i = 0; i < n; i++) {
        float min_dist = std::numeric_limits<float>::max();
        int best_cluster = 0;

        // 找最近的聚类中心
        for (size_t c = 0; c < nlist; c++) {
            float dist = 0;
            for (size_t j = 0; j < d; j++) {
                float diff = x[i * d + j] - centroids[c * d + j];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }

        assignments[i] = best_cluster;
    }
}

void update_centroids(
    const float* x, size_t n, size_t d,
    size_t nlist, float* centroids,
    const int* assignments) {

    // 重新计算聚类中心（均值）
    for (size_t c = 0; c < nlist; c++) {
        // 初始化为 0
        for (size_t j = 0; j < d; j++) {
            centroids[c * d + j] = 0;
        }

        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            if (assignments[i] == c) {
                for (size_t j = 0; j < d; j++) {
                    centroids[c * d + j] += x[i * d + j];
                }
                count++;
            }
        }

        // 取平均
        if (count > 0) {
            for (size_t j = 0; j < d; j++) {
                centroids[c * d + j] /= count;
            }
        }
    }
}
```

#### 3.2.3 添加向量到倒排表

```cpp
// 位置: faiss/IndexIVF.cpp

template <typename IndexClass>
void IndexIVF<IndexClass>::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids) {

    // 1. 量化：找到每个向量属于哪个聚类
    std::vector<idx_t> list_nos(n);

    quantizer->assign(n, x, list_nos.data());

    // 2. 编码向量
    std::vector<uint8_t> codes(n * code_size);

    quantizer->encode_vectors(n, x, codes.data());

    // 3. 添加到倒排表
    for (idx_t i = 0; i < n; i++) {
        idx_t list_no = list_nos[i];
        size_t offset = invlists->add_entry(
            list_no,
            xids ? xids[i] : i,
            codes.data() + i * code_size
        );
    }

    ntotal += n;
}
```

### 3.3 IVF 搜索详解

```cpp
// 位置: faiss/IndexIVF.cpp

template <typename IndexClass>
void IndexIVF<IndexClass>::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const IVFSearchParameters* params) const {

    // 1. 量化查询：找到最近的 nprobe 个聚类中心
    std::vector<idx_t> nearest_centers(nprobe * n);

    quantizer->search(n, x, nearest_centers.data(),
                    nprobe * n);

    // 2. 搜索倒排表
    // 这是最关键的部分

    // 对于每个查询
    for (idx_t i = 0; i < n; i++) {
        const float* xi = x + i * d;

        // 初始化结果堆
        heap_heapify<IndexClass::C::template operator<>>(
            k,
            distances + i * k,
            labels + i * k,
            d
        );

        // 对于每个选中的聚类
        for (idx_t list_no : nearest_centers) {
            // 获取倒排表
            const size_t* list_ids = invlists->get_ids(list_no);
            const uint8_t* list_codes = invlists->get_codes(list_no);
            size_t list_size = invlists->list_size(list_no);

            // 扫描倒排表
            search_one_list(
                xi,
                list_ids, list_codes, list_size,
                k, distances + i * k, labels + i * k
            );
        }

        // 最终排序
        heap_reorder<IndexClass::C>(k, distances + i * k, labels + i * k);
    }
}
```

**search_one_list 的实现**：

```cpp
// 扫描单个倒排表并维护 Top-K 堆

template <typename IndexClass>
void IndexIVF<IndexClass>::search_one_list(
        const float* xi,
        const size_t* list_ids,
        const uint8_t* list_codes,
        size_t list_size,
        size_t k,
        float* distances,
        idx_t* labels) const {

    // 对于每个倒排表中的向量
    for (size_t i = 0; i < list_size; i++) {
        // 1. 解码向量（如果是 PQ，需要查表）
        float decoded[128];  // 假设维度为 128
        decode_vector(list_codes + i * code_size, decoded);

        // 2. 计算距离
        float dist = compute_distance(xi, decoded, d);

        // 3. 更新堆
        if (dist < distances[0]) {
            heap_replace_top(
                k, distances, labels, dist, list_ids[i]
            );
        }
    }
}

void heap_replace_top(
        size_t k,
        float* distances,
        idx_t* labels,
        float new_dist,
        idx_t new_id) {

    // distances[0] 是当前最大值（最小堆的堆顶）
    if (new_dist < distances[0]) {
        // 替换堆顶
        distances[0] = new_dist;
        labels[0] = new_id;

        // 下沉操作：维护堆性质
        heap_sink_down(k, distances, labels);
    }
}
```

### 3.4 IVF 性能分析

#### 3.4.1 加速比分析

```
配置                  nlist    nprobe    搜索比例    加速比
───────────────────────────────────────────────────────
Flat (baseline)      -       -        100%          1x
IVF-100-10            100      10        10%           9x
IVF-1000-100          1000     100       1%            90x
IVF-10000-1000        10000    1000      0.1%          900x
```

**权衡**：
- **nlist 越大**：聚类中心训练时间长，内存占用大
- **nprobe 越大**：搜索越慢，但精度越高

#### 3.4.2 实际性能测试

```cpp
// 性能测试代码
void benchmark_ivf() {
    const size_t d = 128;
    const size_t n = 10000000;  // 1000 万向量
    const size_t nlist = 1000;
    const size_t nprobe = 100;

    IndexIVFFlat index(d, nlist, METRIC_L2);

    // 训练
    auto start_train = std::chrono::high_resolution_clock::now();
    index.train(n, train_vectors);
    auto end_train = std::chrono::high_resolution_clock::now();
    printf("Training time: %.2f s\n",
           std::chrono::duration<double>(end_train - start_train).count());

    // 添加
    index.add(n, vectors);

    // 搜索
    auto start_search = std::chrono::high_resolution_clock::now();
    index.search(1000, queries, 10, distances, labels);
    auto end_search = std::chrono::high_resolution_clock::now();
    printf("Search time (1000 queries): %.2f ms\n",
           std::chrono::duration<double, std::milli>(end_search - start_search).count());
}
```

---

## 第四部分：HNSW Index（图算法）

### 4.1 算法原理

**HNSW (Hierarchical Navigable Small World)** 是基于图的高效近似最近邻算法。

**核心思想**：
1. **图构建**：构建多层图，每层是一个邻近图（NSG）
2. **搜索**：从顶层开始，贪心向下搜索
3. **层次化**：上层粗略定位，下层精确搜索

**图结构示意**：

```
Layer 2 (最上层，最稀疏):
    ┌─────────────┐
    │      3      │  ← 只有少量长连接
    │     /|\      │
    │   1─┼─5     │
    └─────────────┘

Layer 1 (中间层):
    ┌───────────────────────────┐
    │    1    2    3    4    5   │  ← 更多连接
    │    /|\   /|\   /|\   |\    |\
    │   2─┼─6─┼─7─┼─8─┼─9    │
    └───────────────────────────┘

Layer 0 (最底层，最稠密):
    ┌──────────────────────────────────────┐
    │ 1   2   3   4   5   6   7   8   9   ...   │  ← 几乎全连接图
    │    \   |   /   |   /   |   /    \   |    \    │
    └──────────────────────────────────────┘
```

**搜索流程**：

```
查询 Q:
1. 从 Layer 2 的入口点开始
2. 在当前层贪心搜索：在每个节点，只访问比当前节点更近的节点
3. 进入下一层：记录已访问的节点，进入其邻居层
4. 重复 2-3 直到到达 Layer 0
5. 返回找到的所有候选
```

### 4.2 Faiss 源码详解

#### 4.2.1 HNSW 类定义

```cpp
// 位置: faiss/HNSW.h

struct HNSW {
    struct HNSWNode {
        std::vector<size_t> neighbors;  // 邻居列表
    };

    int M;                    // 图的层数
    int ef_construction;       // 构建时的连接数（每个节点的最大邻居数）
    int ef_search;           // 搜索时的连接数（贪心搜索时的邻居数）

    size_t max_level;         // 当前图的最大层数
    std::vector<HNSWNode> nodes;  // 所有节点
    std::vector<size_t> entry_point;  // 入口点列表（Layer M）

    void add_node(size_t id);     // 添加节点
    void search(const float* x, size_t k);  // 搜索
};
```

#### 4.2.2 图构建算法

```cpp
// 位置: faiss/HNSW.cpp

void HNSW::add_node(size_t id) {
    // 获取新节点的层数
    int level = get_random_level();
    max_level = std::max(max_level, level);

    nodes.resize(id + 1);
    auto& node = nodes[id];

    // 从顶层到该节点的层数，逐层插入
    for (int lc = level; lc >= 0; lc--) {
        // 在当前层搜索 ef_construction 个最近邻
        auto neighbors = search_layer(
            node, lc, ef_construction
        );

        // 添加双向连接
        for (size_t neighbor_id : neighbors) {
            node.neighbors[lc].push_back(neighbor_id);
            nodes[neighbor_id].neighbors[lc].push_back(id);
        }

        // 连接数限制为 ef_construction
        node.neighbors[lc].resize(ef_construction);
    }

    // 设置入口点
    if (level > max_level) {
        entry_point.push_back(id);
    }
}

std::vector<size_t> HNSW::search_layer(
        HNSWNode& node,
        int level,
        int ef) {

    // 贪心：在当前层找到 ef 个最近邻
    std::priority_queue<std::pair<float, size_t>> top_candidates;

    // 1. 将入口点加入候选集
    for (size_t ep : entry_point) {
        if (ep >= nodes.size()) continue;
        float dist = compute_distance(node, nodes[ep]);
        top_candidates.push({dist, ep});
    }

    // 2. 贪心扩展（ef_constraint 次数）
    while (top_candidates.size() < ef_construction) {
        auto [dist, node_id] = top_candidates.top();

        // 添加该节点的邻居
        for (size_t neighbor : nodes[node_id].neighbors[level]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                float new_dist = compute_distance(node, nodes[neighbor]);
                top_candidates.push({new_dist, neighbor});
            }
        }
    }

    // 3. 排序并返回前 ef_constraint 个
    std::vector<size_t> result;
    while (!top_candidates.empty() && result.size() < ef_constraint) {
        result.push_back(top_candidates.top().second);
        top_candidates.pop();
    }

    return result;
}
```

#### 4.2.3 搜索算法

```cpp
// 位置: faiss/HNSW.cpp

void HNSW::search(const float* x, size_t k) {
    // 1. 从顶层入口点开始
    std::vector<size_t> entry_points = entry_point;

    // 2. 从顶层到底层搜索
    for (int level = max_level - 1; level >= 0; level--) {
        // 在当前层搜索 ef_search 个最近邻
        auto neighbors = search_layer(
            nodes, level, ef_search, entry_points
        );

        // 3. 更新入口点（下一层从这个节点开始）
        entry_points = neighbors;
    }

    // 4. 底层结果处理
    // - 进一步精确计算距离
    // - 过滤重复结果
    // - 排序返回 Top-K
}
```

### 4.3 HNSW 性能特点

**构建复杂度**：
```
操作                复杂度
─────────────────────────────
添加节点              O(log n) * M
搜索                  O(log n) * M
内存占用              O(n * M * ef)
```

**性能特点**：
- 对**动态数据集**友好：增量式添加
- **召回率高**：通常 > 95%
- **搜索速度快**：对高维数据特别有效

---

## 第五部分：Product Quantization（乘积量化）

### 5.1 算法原理

**Product Quantization (PQ)** 是一种向量压缩技术，通过将高维向量分解为多个低维子向量，分别量化。

**核心思想**：
```
原始 128 维向量：
[x1, x2, ..., x128]

分解为 8 个 16 维子向量：
子向量 0: [x1, x2, ..., x16]
子向量 1: [x17, x18, ..., x32]
...
子向量 7: [x113, x114, ..., x128]

每个子向量用 256 个码本（8 bit）表示
```

**量化过程**：

```
1. 训练阶段：
   - 对每个子维度训练 256 个码本（ centroids）
   - 每个码本是一个 16 维向量

2. 编码阶段：
   - 计算每个子向量到所有码本的距离
   - 选择距离最小的码本索引
   - 8 个子向量 → 8 个码本索引 → 8 字节编码

3. 解码阶段：
   - 根据 8 个码本索引查找对应的码本向量
   - 拼接 8 个码本向量 → 128 维向量
```

### 5.2 Faiss 源码详解

#### 5.2.1 ProductQuantizer 类

```cpp
// 位置: faiss/impl/ProductQuantizer.h

struct ProductQuantizer {
    size_t M;          // 子量化器数量（子向量数）
    size_t nbits;       // 每个子量化器的位数
    size_t dsub;        // 子向量维度 = d / M

    // 码本：[M, 256, dsub]
    std::vector<float> centroids;

    // 编码/解码表
    std::vector<float> decode_tables;  // [M, 256, dsub]

    void train(size_t n, const float* x);      // 训练
    void encode(const float* x, uint8_t* codes) const;  // 编码
    void decode(const uint8_t* codes, float* x) const;  // 解码
};
```

#### 5.2.2 训练算法

```cpp
// 位置: faiss/impl/ProductQuantizer.cpp

void ProductQuantizer::train(size_t n, const float* x) {
    // 1. 初始化码本（随机选择）
    initialize_centroids(n, x);

    // 2. 迭代优化码本
    for (int iter = 0; iter < 25; iter++) {
        // 2.1 分配向量到最近的码本
        std::vector<uint8_t> codes(n * M);
        assign_to_centroids(n, x, codes.data());

        // 2.2 更新码本
        update_centroids(x, codes.data());
    }
}

void ProductQuantizer::assign_to_centroids(
        size_t n,
        const float* x,
        uint8_t* codes) {

    for (size_t i = 0; i < n; i++) {
        for (size_t m = 0; m < M; m++) {
            // 找到最近的码本
            size_t best_code = 0;
            float min_dist = std::numeric_limits<float>::max();

            for (size_t k = 0; k < 256; k++) {
                // 计算距离
                float dist = 0;
                for (size_t j = 0; j < dsub; j++) {
                    float diff = x[i * d + m * dsub + j] -
                                centroids[m * 256 * dsub + k * dsub + j];
                    dist += diff * diff;
                }

                if (dist < min_dist) {
                    min_dist = dist;
                    best_code = k;
                }
            }

            codes[i * M + m] = best_code;
        }
    }
}
```

#### 5.2.3 编码过程

```cpp
// 编码：将向量压缩为 8 字节（假设 M=8, nbits=8）

void ProductQuantizer::encode(
        const float* x,
        uint8_t* codes) const {

    for (size_t m = 0; m < M; m++) {
        // 获取子向量
        const float* sub_vector = x + m * dsub;

        // 计算到所有码本的距离
        std::vector<float> distances(256);
        for (size_t k = 0; k < 256; k++) {
            float dist = 0;
            for (size_t j = 0; j < dsub; j++) {
                float diff = sub_vector[j] -
                           centroids[m * 256 * dsub + k * dsub + j];
                dist += diff * diff;
            }
            distances[k] = dist;
        }

        // 找最小距离的码本
        size_t best_code = std::min_element(
            distances.begin(), distances.end()
        ) - distances.begin();

        // 存储码本索引
        codes[m] = (uint8_t)best_code;
    }
}
```

#### 5.2.4 解码与距离计算

```cpp
// 解码：从编码恢复向量

void ProductQuantizer::decode(
        const uint8_t* codes,
        float* x) const {

    for (size_t m = 0; m < M; m++) {
        uint8_t code = codes[m];

        // 复制码本向量
        const float* centroid = &centroids[m * 256 * dsub];
        memcpy(x + m * dsub, centroid, dsub * sizeof(float));
    }
}

// 使用查找表的快速距离计算
void ProductQuantizer::compute_L2_distance_table(
        const float* query,
        float* lut) const {

    // 为每个子量化器计算查找表
    for (size_t m = 0; m < M; m++) {
        // 获取子查询向量
        const float* sub_query = query + m * dsub;

        for (size_t k = 0; k < 256; k++) {
            // 计算子查询与码本的 L2 距离
            float dist = 0;
            for (size_t j = 0; j < dsub; j++) {
                float diff = sub_query[j] -
                           centroids[m * 256 * dsub + k * dsub + j];
                dist += diff * diff;
            }
            lut[m * 256 + k] = dist;
        }
    }
}

// 快速距离计算：给定编码向量，计算与查询的距离
float ProductQuantizer::compute_distance(
        const uint8_t* codes,
        const float* lut) const {

    float dist = 0;
    for (size_t m = 0; m < M; m++) {
        uint8_t code = codes[m];
        dist += lut[m * 256 + code];
    }
    return dist;
}
```

### 5.3 IVFPQ：IVF + PQ

**IVFPQ** 结合了 IVF 的分区思想和 PQ 的压缩技术。

**存储对比**：

```
IndexIVFFlat（未压缩）:
- 1000 万 × 128 维 × 4 字节 = 512 MB

IndexIVFPQ（PQ8x8）:
- 1000 万 × 8 字节 = 8 MB
- 压缩比：64x！
```

**IVFPQ 搜索流程**：

```cpp
// 位置: faiss/IndexIVFPQ.h

void IndexIVFPQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const IVFSearchParameters* params) {

    // 1. 计算查找表
    float lut[256 * M];  // [M, 256]
    compute_L2_distance_table(x, lut);

    // 2. 量化查询向量
    std::vector<uint8_t> q_codes(M);
    quantizer->encode(x, q_codes.data());

    // 3. 搜索每个倒排表
    for (size_t list_id : selected_lists) {
        const uint8_t* codes = invlists->get_codes(list_id);
        size_t list_size = invlists->list_size(list_id);

        // 4. 使用查找表快速计算距离
        for (size_t i = 0; i < list_size; i++) {
            float dist = 0;
            for (size_t m = 0; m < M; m++) {
                dist += lut[m * 256 + codes[i * M + m]];
            }

            // 5. 更新堆
            if (dist < distances[0]) {
                heap_replace_top(k, distances, labels, dist,
                               invlists->get_ids(list_id)[i]);
            }
        }
    }
}
```

---

## 第六部分：实战案例 - 构建完整索引

### 6.1 综合案例：图像搜索引擎

让我们实现一个完整的图像搜索引擎，综合应用所学知识。

```cpp
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <memory>

class ImageSearchEngine {
public:
    struct Config {
        size_t dimension = 512;       // ResNet-50 特征维度
        size_t num_vectors = 10000000;  // 1000 万图像
        size_t nlist = 1000;          // IVF 聚类数
        size_t nprobe = 100;          // IVF 搜索聚类数
        size_t M = 32;                // PQ 子量化器数
        size_t nbits = 8;             // PQ 每个子量化器位数
    };

    ImageSearchEngine(const Config& cfg)
        : config(cfg) {

        if (cfg.dimension % cfg.M != 0) {
            throw std::invalid_argument(
                "dimension must be divisible by M"
            );
        }
    }

    // 添加图像特征
    void add_image(size_t image_id, const float* feature) {
        // TODO: 实现
    }

    // 搜索相似图像
    void search(
        const float* query_feature,
        size_t k,
        std::vector<size_t>& image_ids,
        std::vector<float>& distances) {
        // TODO: 实现
    }

private:
    Config config;
    std::unique_ptr<class Index> index;
};
```

### 6.2 完整实现

```cpp
// 完整的图像搜索引擎实现

#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexHNSW.h"

class ImageSearchEngineImpl {
public:
    struct Config {
        size_t dimension = 512;
        size_t num_vectors = 0;
        size_t nlist = 1000;
        size_t nprobe = 100;
        MetricType metric = METRIC_L2;
        bool use_pq = false;
        bool use_hnsw = false;
    };

    ImageSearchEngineImpl(const Config& cfg)
        : config(cfg) {

        if (cfg.use_pq) {
            // IVF + PQ
            auto ivfpq = std::make_unique<IndexIVFPQ>(
                cfg.dimension, cfg.nlist, cfg.metric
            );

            // 配置 PQ 参数
            ivfpq->nprobe = cfg.nprobe;
            ivfpq->quantizer->nbits = 8;  // 8 bit
            ivfpq->quantizer->M = cfg.dimension / cfg.nbits;

            index = std::move(ivfpq);
        } else if (cfg.use_hnsw) {
            // HNSW
            auto hnsw = std::make_unique<IndexHNSW>(
                cfg.dimension, cfg.metric
            );

            // 配置 HNSW 参数
            hnsw->hnsw.M = 32;           // 层数
            hnsw->hnsw.ef_construction = 64;  // 构建连接数
            hnsw->hnsw.ef_search = 32;       // 搜索连接数

            index = std::move(hnsw);
        } else {
            // IVFFlat
            auto ivf = std::make_unique<IndexIVFFlat>(
                cfg.dimension, cfg.nlist, cfg.metric
            );

            ivf->nprobe = cfg.nprobe;

            index = std::move(ivf);
        }
    }

    void add_vectors(const float* vectors, size_t count) {
        index->add(count, vectors);
    }

    void search(
        const float* queries,
        size_t num_queries,
        size_t k,
        float* distances,
        size_t* labels) {

        index->search(num_queries, queries, k, distances, labels);
    }

    void save(const char* filename) {
        index->write(filename);
    }

    void load(const char* filename) {
        index = read_index(filename);
    }

private:
    Config config;
    std::unique_ptr<Index> index;
};

// 使用示例
int main() {
    // 配置
    ImageSearchEngineImpl::Config cfg;
    cfg.dimension = 512;
    cfg.num_vectors = 1000000;
    cfg.nlist = 1000;
    cfg.nprobe = 100;
    cfg.metric = METRIC_L2;
    cfg.use_pq = true;  // 启用 PQ 压缩

    // 创建引擎
    ImageSearchEngineImpl engine(cfg);

    // 生成模拟数据
    std::vector<float> features(cfg.num_vectors * cfg.dimension);
    // ... 填充随机特征 ...

    // 添加特征
    engine.add_vectors(features.data(), cfg.num_vectors);

    // 搜索
    const size_t num_queries = 100;
    std::vector<float> queries(num_queries * cfg.dimension);
    // ... 填充查询 ...

    std::vector<float> distances(num_queries * 10);
    std::vector<size_t> labels(num_queries * 10);

    auto start = std::chrono::high_resolution_clock::now();
    engine.search(queries.data(), num_queries, 10,
                 distances.data(), labels.data());
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<
        double, std::milli>(end - start).count();

    printf("Search %zu queries in %.2f ms\n", num_queries, ms);
    printf("Average latency: %.4f ms\n", ms / num_queries);
    printf("QPS: %.2f\n", num_queries / (ms / 1000.0));

    return 0;
}
```

---

## 第七部分：Faiss 源码深度剖析

### 7.1 Index 类继承体系

#### 7.1.1 完整的类层次结构

```cpp
// 位置: faiss/Index.h

namespace faiss {

// 所有索引的基类
struct Index {
    idx_t d;                    // 向量维度
    idx_t ntotal;               // 向量总数
    MetricType metric_type;     // 距离度量类型
    bool is_trained;            // 是否已训练

    Index(idx_t d = 0, MetricType metric = METRIC_L2);

    // 虚析构函数（支持多态）
    virtual ~Index();

    // 训练（某些索引需要训练，如 PQ、IVF）
    virtual void train(idx_t n, const float* x);

    // 添加向量
    virtual void add(idx_t n, const float* x);
    virtual void add_with_ids(idx_t n, const float* x, const idx_t* xids);

    // 搜索
    virtual void search(
            idx_t n, const float* x, idx_t k,
            float* distances, idx_t* labels,
            const SearchParameters* params = nullptr) const = 0;

    // 范围搜索
    virtual void range_search(
            idx_t n, const float* x, float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const;

    // 重构向量（从索引中恢复原始向量）
    virtual void reconstruct(idx_t key, float* recons) const;

    // 重置索引
    virtual void reset();

    // 序列化
    virtual void serialize(FILE* fp) const;
};

// 编码索引基类（存储编码后的向量）
struct IndexFlatCodes : Index {
    size_t code_size;                           // 每个向量的编码大小（字节）
    MaybeOwnedVector<uint8_t> codes;            // 编码数据 [ntotal * code_size]

    IndexFlatCodes(size_t code_size, idx_t d, MetricType metric);

    void add(idx_t n, const float* x) override;
    void reconstruct(idx_t key, float* recons) const override;

    // 编码/解码接口
    virtual void sa_encode(idx_t n, const float* x, uint8_t* bytes) const = 0;
    virtual void sa_decode(idx_t n, const uint8_t* bytes, float* x) const = 0;
};

// Flat 索引（存储原始浮点向量）
// IndexFlat : IndexFlatCodes : Index
struct IndexFlat : IndexFlatCodes {
    // code_size = d * sizeof(float)
    // codes 直接存储 float 数组

    IndexFlat(idx_t d, MetricType metric = METRIC_L2);

    // 编码 = 直接复制浮点数据
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override {
        memcpy(bytes, x, n * code_size);
    }

    // 解码 = 直接复制回来
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override {
        memcpy(x, bytes, n * code_size);
    }
};

// L2 距离的 Flat 索引（带 L2 norm 缓存）
struct IndexFlatL2 : IndexFlat {
    std::vector<float> cached_l2norms;    // 缓存的 L2 范数

    IndexFlatL2(idx_t d);

    // 预计算所有向量的 L2 范数
    void sync_l2norms();

    // 使用缓存的 L2 范数进行距离计算
    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;
};

// 内积的 Flat 索引
struct IndexFlatIP : IndexFlat {
    IndexFlatIP(idx_t d) : IndexFlat(d, METRIC_INNER_PRODUCT) {}
};

} // namespace faiss
```

#### 7.1.2 距离计算器层次

```cpp
// 位置: faiss/impl/DistanceComputer.h

namespace faiss {

// 距离计算器接口
struct DistanceComputer {
    idx_t d;  // 维度

    virtual ~DistanceComputer() = default;

    // 设置查询向量
    virtual void set_query(const float* x) = 0;

    // 计算与向量 j 的对称距离
    virtual float symmetric_dis(idx_t j) = 0;

    // 计算与单个向量的距离
    virtual float distance_to_code(const uint8_t* code) = 0;
};

// Flat 索引的距离计算器
struct FlatCodesDistanceComputer : DistanceComputer {
    const uint8_t* codes;      // 编码数据
    size_t code_size;          // 编码大小

    const float* query;        // 当前查询向量

    void set_query(const float* x) override {
        query = x;
    }

    // 需要子类实现具体的距离计算
};

// L2 距离计算器
struct L2DistanceComputer : FlatCodesDistanceComputer {
    float symmetric_dis(idx_t j) override {
        // 计算查询与第 j 个向量的 L2 距离
        const float* vec = (float*)(codes + j * code_size);
        return fvec_L2sqr(query, vec, d);
    }

    float distance_to_code(const uint8_t* code) override {
        const float* vec = (float*)code;
        return fvec_L2sqr(query, vec, d);
    }
};

// 内积距离计算器
struct IPDistanceComputer : FlatCodesDistanceComputer {
    float symmetric_dis(idx_t j) override {
        const float* vec = (float*)(codes + j * code_size);
        return fvec_inner_product(query, vec, d);
    }

    float distance_to_code(const uint8_t* code) override {
        const float* vec = (float*)code;
        return fvec_inner_product(query, vec, d);
    }
};

// 使用 L2 norm 缓存的 L2 距离计算器
// 利用公式：||q - v||² = ||q||² + ||v||² - 2⟨q, v⟩
struct L2NormCachedDistanceComputer : FlatCodesDistanceComputer {
    const float* norms;        // 预计算的向量 L2 范数
    float query_norm;          // 查询向量的 L2 范数

    void set_query(const float* x) override {
        query = x;
        query_norm = fvec_norm_L2sqr(x, d);
    }

    float symmetric_dis(idx_t j) override {
        const float* vec = (float*)(codes + j * code_size);
        float ip = fvec_inner_product(query, vec, d);
        float vec_norm = norms[j];

        // ||q - v||² = ||q||² + ||v||² - 2⟨q, v⟩
        return query_norm + vec_norm - 2 * ip;
    }
};

} // namespace faiss
```

### 7.2 InvertedLists 实现详解

#### 7.2.1 InvertedLists 基类完整实现

```cpp
// 位置: faiss/invlists/InvertedLists.h

namespace faiss {

// 倒排表迭代器
struct InvertedListsIterator {
    virtual ~InvertedListsIterator() = default;

    // 检查迭代器是否有效
    virtual bool is_available() const = 0;

    // 移动到下一个元素
    virtual void next() = 0;

    // 获取当前元素的 id 和 code
    virtual std::pair<idx_t, const uint8_t*> get_id_and_codes() = 0;
};

// 倒排表基类
struct InvertedLists {
    size_t nlist;              // 倒排表数量（聚类中心数）
    size_t code_size;          // 每个向量的编码大小（字节）
    bool use_iterator;         // 是否使用迭代器接口

    InvertedLists(size_t nlist, size_t code_size)
        : nlist(nlist), code_size(code_size), use_iterator(false) {}

    virtual ~InvertedLists() = default;

    /***** 只读操作 *****/

    // 获取某个倒排表的大小
    virtual size_t list_size(size_t list_no) const = 0;

    // 获取某个倒排表的所有编码
    // 返回: list_size(list_no) * code_size 字节
    // 注意: 使用后需要调用 release_codes
    virtual const uint8_t* get_codes(size_t list_no) const = 0;

    // 获取某个倒排表的所有 ID
    // 返回: list_size(list_no) 个 idx_t
    // 注意: 使用后需要调用 release_ids
    virtual const idx_t* get_ids(size_t list_no) const = 0;

    // 释放 get_codes 返回的内存
    virtual void release_codes(size_t list_no, const uint8_t* codes) const {
        // 默认实现：无需操作（内存由倒排表管理）
    }

    // 释放 get_ids 返回的内存
    virtual void release_ids(size_t list_no, const idx_t* ids) const {
        // 默认实现：无需操作
    }

    // 获取单个 ID
    virtual idx_t get_single_id(size_t list_no, size_t offset) const {
        const idx_t* ids = get_ids(list_no);
        idx_t id = ids[offset];
        release_ids(list_no, ids);
        return id;
    }

    // 获取单个编码
    virtual const uint8_t* get_single_code(size_t list_no, size_t offset) const {
        const uint8_t* codes = get_codes(list_no);
        const uint8_t* code = codes + offset * code_size;
        // 注意：这里不能直接 release，因为返回的是指针的一部分
        // 子类需要重写这个方法
        return code;
    }

    // 预取指定的倒排表（性能优化）
    virtual void prefetch_lists(const idx_t* list_nos, int nlist) const {
        // 默认实现：无操作
        // 子类可以重写以实现硬件预取
    }

    /***** 迭代器接口 *****/

    // 检查倒排表是否为空
    virtual bool is_empty(size_t list_no, void* context) const {
        return list_size(list_no) == 0;
    }

    // 获取迭代器
    virtual InvertedListsIterator* get_iterator(
            size_t list_no, void* context) const {
        return nullptr;  // 默认不支持迭代器
    }

    /***** 写操作 *****/

    // 添加单个条目到倒排表
    // 返回: 新条目的 offset
    virtual size_t add_entry(
            size_t list_no,
            idx_t id,
            const uint8_t* code,
            void* context) {

        return add_entries(list_no, 1, &id, code);
    }

    // 批量添加条目
    virtual size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) = 0;

    // 更新单个条目
    virtual void update_entry(
            size_t list_no,
            size_t offset,
            idx_t id,
            const uint8_t* code) {

        update_entries(list_no, offset, 1, &id, code);
    }

    // 批量更新条目
    virtual void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) = 0;

    // 调整倒排表大小
    virtual void resize(size_t list_no, size_t new_size) = 0;

    // 重置所有倒排表
    virtual void reset() {
        for (size_t i = 0; i < nlist; i++) {
            resize(i, 0);
        }
    }

    /***** 高级操作 *****/

    // 合并另一个倒排表
    void merge_from(InvertedLists* oivf, size_t add_id) {
        for (size_t i = 0; i < nlist; i++) {
            size_t n = oivf->list_size(i);
            if (n == 0) continue;

            const idx_t* ids = oivf->get_ids(i);
            const uint8_t* codes = oivf->get_codes(i);

            // 添加 add_id 到每个 ID
            std::vector<idx_t> new_ids(n);
            for (size_t j = 0; j < n; j++) {
                new_ids[j] = ids[j] + add_id;
            }

            add_entries(i, n, new_ids.data(), codes);

            oivf->release_ids(i, ids);
            oivf->release_codes(i, codes);
        }
        oivf->reset();
    }

    /***** 统计 *****/

    // 计算不平衡因子
    // 1 = 完美平衡，> 1 = 不平衡
    double imbalance_factor() const {
        size_t max_size = 0;
        size_t total_size = 0;

        for (size_t i = 0; i < nlist; i++) {
            size_t size = list_size(i);
            max_size = std::max(max_size, size);
            total_size += size;
        }

        if (total_size == 0) return 1.0;
        double avg = (double)total_size / nlist;
        return max_size / avg;
    }

    // 计算总向量数
    size_t compute_ntotal() const {
        size_t total = 0;
        for (size_t i = 0; i < nlist; i++) {
            total += list_size(i);
        }
        return total;
    }

    // 打印统计信息
    void print_stats() const {
        printf("InvertedLists stats:\n");
        printf("  nlist: %zu\n", nlist);
        printf("  code_size: %zu\n", code_size);
        printf("  ntotal: %zu\n", compute_ntotal());
        printf("  imbalance_factor: %.3f\n", imbalance_factor());

        // 每个 list 的大小分布
        std::vector<size_t> sizes(nlist);
        for (size_t i = 0; i < nlist; i++) {
            sizes[i] = list_size(i);
        }
        std::sort(sizes.begin(), sizes.end());

        printf("  size distribution:\n");
        printf("    min: %zu\n", sizes[0]);
        printf("    25%%: %zu\n", sizes[nlist / 4]);
        printf("    50%%: %zu\n", sizes[nlist / 2]);
        printf("    75%%: %zu\n", sizes[nlist * 3 / 4]);
        printf("    max: %zu\n", sizes[nlist - 1]);
    }
};

// RAII 风格的 codes 管理器
struct ScopedCodes {
    const InvertedLists* il;
    size_t list_no;
    const uint8_t* codes;

    ScopedCodes(const InvertedLists* il, size_t list_no)
        : il(il), list_no(list_no), codes(il->get_codes(list_no)) {}

    ~ScopedCodes() {
        il->release_codes(list_no, codes);
    }

    const uint8_t* get() const {
        return codes;
    }

    // 禁止拷贝
    ScopedCodes(const ScopedCodes&) = delete;
    ScopedCodes& operator=(const ScopedCodes&) = delete;
};

// RAII 风格的 ids 管理器
struct ScopedIds {
    const InvertedLists* il;
    size_t list_no;
    const idx_t* ids;

    ScopedIds(const InvertedLists* il, size_t list_no)
        : il(il), list_no(list_no), ids(il->get_ids(list_no)) {}

    ~ScopedIds() {
        il->release_ids(list_no, ids);
    }

    const idx_t* get() const {
        return ids;
    }

    ScopedIds(const ScopedIds&) = delete;
    ScopedIds& operator=(const ScopedIds&) = delete;
};

} // namespace faiss
```

#### 7.2.2 ArrayInvertedLists 实现

```cpp
// 位置: faiss/invlists/InvertedLists.cpp

namespace faiss {

// 内存中的倒排表实现（使用数组）
struct ArrayInvertedLists : InvertedLists {
    // 每个倒排表的内容
    std::vector<std::vector<uint8_t>> codes;      // [nlist][list_i_size * code_size]
    std::vector<std::vector<idx_t>> ids;          // [nlist][list_i_size]

    ArrayInvertedLists(size_t nlist, size_t code_size)
        : InvertedLists(nlist, code_size),
          codes(nlist),
          ids(nlist) {}

    size_t list_size(size_t list_no) const override {
        return ids[list_no].size();
    }

    const uint8_t* get_codes(size_t list_no) const override {
        return codes[list_no].data();
    }

    const idx_t* get_ids(size_t list_no) const override {
        return ids[list_no].data();
    }

    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids_in,
            const uint8_t* code) override {

        size_t o = ids[list_no].size();
        ids[list_no].resize(o + n_entry);
        memcpy(ids[list_no].data() + o, ids_in, n_entry * sizeof(idx_t));

        codes[list_no].resize((o + n_entry) * code_size);
        memcpy(codes[list_no].data() + o * code_size, code, n_entry * code_size);

        return o;
    }

    void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t* ids_in,
            const uint8_t* code) override {

        memcpy(ids[list_no].data() + offset, ids_in, n_entry * sizeof(idx_t));
        memcpy(codes[list_no].data() + offset * code_size, code, n_entry * code_size);
    }

    void resize(size_t list_no, size_t new_size) override {
        ids[list_no].resize(new_size);
        codes[list_no].resize(new_size * code_size);
    }
};

} // namespace faiss
```

### 7.3 IndexIVF 完整实现

#### 7.3.1 Level1Quantizer（粗量化器）

```cpp
// 位置: faiss/IndexIVF.h

namespace faiss {

// Level1Quantizer: 封装量化器（coarse quantizer）
struct Level1Quantizer {
    Index* quantizer;         // 量化器索引（通常是 IndexFlat）
    size_t nlist;             // 聚类中心数量
    bool own_fields;          // 是否拥有 quantizer 的所有权

    // 量化器训练模式
    char quantizer_trains_alone;  // 0, 1, 2

    // 聚类参数
    ClusteringParameters cp;       // K-Means 参数
    Index* clustering_index;       // 聚类时使用的索引

    Level1Quantizer(Index* quantizer, size_t nlist)
        : quantizer(quantizer),
          nlist(nlist),
          own_fields(false),
          quantizer_trains_alone(0),
          clustering_index(nullptr) {}

    ~Level1Quantizer() {
        if (own_fields) {
            delete quantizer;
            delete clustering_index;
        }
    }

    // 计算粗编码（coarse code）的大小
    size_t coarse_code_size() const {
        return nlist <= 256 ? 1 : nlist <= 65536 ? 2 : 4;
    }

    // 将 list_no 编码为字节数组
    void encode_listno(idx_t list_no, uint8_t* code) const {
        if (nlist <= 256) {
            code[0] = (uint8_t)list_no;
        } else if (nlist <= 65536) {
            uint16_t* c = (uint16_t*)code;
            c[0] = (uint16_t)list_no;
        } else {
            idx_t* c = (idx_t*)code;
            c[0] = list_no;
        }
    }

    // 从字节数组解码 list_no
    idx_t decode_listno(const uint8_t* code) const {
        if (nlist <= 256) {
            return code[0];
        } else if (nlist <= 65536) {
            uint16_t* c = (uint16_t*)code;
            return c[0];
        } else {
            idx_t* c = (idx_t*)code;
            return c[0];
        }
    }

    // 训练量化器
    void train_q1(
            size_t n,
            const float* x,
            bool verbose,
            MetricType metric_type) {

        if (quantizer_trains_alone == 1) {
            // 模式1: 直接调用 quantizer 的训练
            quantizer->train(n, x);
            return;
        }

        // 模式0或2: 使用 K-Means 训练
        Clustering clus(d, nlist);
        clus.cp = cp;

        if (clustering_index) {
            // 使用指定的索引进行聚类
            clus.train(n, x, *clustering_index);
        } else {
            // 使用默认的 flat 索引
            clus.train(n, x);
        }

        // 将聚类中心添加到 quantizer
        quantizer->reset();
        quantizer->add(nlist, clus.centroids.data());

        if (quantizer_trains_alone == 2) {
            // 模式2: 将聚类中心添加到 quantizer
            quantizer->add(nlist, clus.centroids.data());
        }
    }
};

} // namespace faiss
```

#### 7.3.2 IndexIVF 搜索参数

```cpp
// IVF 搜索参数
struct SearchParametersIVF : SearchParameters {
    size_t nprobe;                 // 搜索的聚类数
    size_t max_codes;              // 最大访问的编码数（限制搜索范围）
    SearchParameters* quantizer_params;  // 传递给 quantizer 的参数
    void* inverted_list_context;   // 传递给 InvertedLists 的上下文

    SearchParametersIVF()
        : nprobe(1),
          max_codes(0),
          quantizer_params(nullptr),
          inverted_list_context(nullptr) {}

    virtual ~SearchParametersIVF() {
        delete quantizer_params;
    }
};
```

#### 7.3.3 IndexIVF 搜索实现（核心）

```cpp
// 位置: faiss/IndexIVF.cpp

namespace faiss {

struct IndexIVFInterface : Level1Quantizer {
    size_t nprobe;           // 搜索时检查的聚类数
    size_t max_codes;        // 最大访问编码数

    IndexIVFInterface(Index* quantizer, size_t nlist)
        : Level1Quantizer(quantizer, nlist),
          nprobe(1),
          max_codes(0) {}

    // 已预分配的搜索（核心搜索函数）
    virtual void search_preassigned(
            idx_t n,                 // 查询数量
            const float* x,          // 查询向量 [n, d]
            idx_t k,                 // 返回 Top-K
            const idx_t* assign,     // 每个查询分配的聚类 [n, nprobe]
            const float* centroid_dis,// 到聚类中心的距离 [n, nprobe]
            float* distances,        // 输出距离 [n, k]
            idx_t* labels,           // 输出标签 [n, k]
            bool store_pairs,        // 存储对而不是ID
            const IVFSearchParameters* params,
            IndexIVFStats* stats) const = 0;

    // 普通搜索（内部调用 search_preassigned）
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 1. 量化查询：找到最近的 nprobe 个聚类
        idx_t* assign = new idx_t[n * nprobe];
        float* centroid_dis = new float[n * nprobe];

        quantizer->search(n, x, nprobe, centroid_dis, assign);

        // 2. 调用预分配的搜索
        search_preassigned(
            n, x, k, assign, centroid_dis,
            distances, labels, false,
            dynamic_cast<const IVFSearchParameters*>(params),
            nullptr
        );

        delete[] assign;
        delete[] centroid_dis;
    }
};

} // namespace faiss
```

### 7.4 堆操作详解

#### 7.4.1 Heap 基础结构

```cpp
// 位置: faiss/utils/Heap.h

namespace faiss {

// 堆操作：C++ 模板实现
// 用于维护 Top-K 结果

// 初始化堆（最小堆）
// 堆性质：heap[0] 是最大值（用于 Top-K 最小值）
template <typename C>
inline void heap_heapify(size_t k, float* distances, idx_t* labels) {
    // 使用 std::make_heap 建立最大堆
    // C 是比较器（通常为 CMax <float, idx_t>）
    for (size_t i = 0; i < k; i++) {
        labels[i] = -1;
        distances[i] = C::neutral();
    }
}

// 堆的上浮操作
template <typename C>
inline void heap_push(size_t k, float* distances, idx_t* labels, idx_t label, float distance) {
    // 如果新距离比堆顶（最大值）小，则替换
    if (C::cmp(distances[0], distance)) {
        heap_replace_top<C>(k, distances, labels, label, distance);
    }
}

// 替换堆顶并重新堆化
template <typename C>
inline void heap_replace_top(
        size_t k, float* distances, idx_t* labels,
        idx_t label, float distance) {

    distances[0] = distance;
    labels[0] = label;
    heap_sink<C>(k, distances, labels, 0);
}

// 堆的下沉操作
template <typename C>
inline void heap_sink(size_t k, float* distances, idx_t* labels, size_t o) {
    size_t j = o;
    while (1) {
        size_t i = j;
        size_t left = 2 * i + 1;
        size_t right = left + 1;

        if (left < k && C::cmp(distances[left], distances[j])) {
            j = left;
        }
        if (right < k && C::cmp(distances[right], distances[j])) {
            j = right;
        }

        if (i == j) break;

        std::swap(distances[i], distances[j]);
        std::swap(labels[i], labels[j]);
        j = i;
    }
}

// 堆排序（结果从大到小排序）
template <typename C>
inline void heap_reorder(size_t k, float* distances, idx_t* labels) {
    // 使用 std::sort_heap
    // 堆 -> 排序数组（从小到大）
    for (size_t i = k - 1; i > 0; i--) {
        std::swap(distances[0], distances[i]);
        std::swap(labels[0], labels[i]);
        heap_sink<C>(i, distances, labels, 0);
    }
}

// 比较器：最大堆（用于最小距离）
template <typename T, typename TI>
struct CMax {
    static inline bool cmp(T a, T b) {
        return a < b;  // a < b 时交换（维护最大堆）
    }

    static inline T neutral() {
        return -std::numeric_limits<T>::infinity();
    }
};

// 比较器：最小堆（用于最大距离）
template <typename T, typename TI>
struct CMin {
    static inline bool cmp(T a, T b) {
        return a > b;  // a > b 时交换（维护最小堆）
    }

    static inline T neutral() {
        return std::numeric_limits<T>::infinity();
    }
};

} // namespace faiss
```

#### 7.4.2 堆数组操作

```cpp
// 批量堆操作（用于并行搜索）

// 初始化多个堆
template <typename C>
inline void heap_heap_array(size_t n, size_t k, float* distances, idx_t* labels) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        heap_heapify<C>(k, distances + i * k, labels + i * k);
    }
}

// 合并多个堆
template <typename C>
inline void heap_addn(
        size_t k, float* distances_base, idx_t* labels_base,
        const idx_t* labels, const float* distances, size_t n) {

    for (size_t i = 0; i < n; i++) {
        heap_push<C>(
            k,
            distances_base + i * k,
            labels_base + i * k,
            labels[i * k],
            distances[i * k]
        );
    }
}
```

---

## 第八部分：NSG & NNDescent（图索引变体）

### 8.1 算法背景

**NSG (Navigating Spreading-out Graph)** 是另一种高效的近似最近邻图索引算法，与 HNSW 相比具有不同的构建策略。

**核心差异**：

| 特性 | HNSW | NSG |
|------|------|-----|
| 图结构 | 多层分层图 | 单层图 |
| 构建方式 | 逐层贪心 | 基于近似近邻构建 |
| 内存占用 | 较高（多层） | 较低（单层） |
| 查询速度 | 更快 | 稍慢 |
| 召回率 | 相当 | 相当 |

**NNDescent (Nearest Neighbor Descent)** 是用于高效构建近似 K-NN 图的算法，常作为 NSG 的前置步骤。

### 8.2 NNDescent 算法原理

**核心思想**：通过迭代更新近邻列表来收敛到高质量的 K-NN 图。

```cpp
// 位置: faiss/impl/NNDescent.cpp

namespace faiss {

struct NNDescent {
    size_t n;              // 数据点数量
    size_t d;              // 维度
    size_t K;              // 近邻数

    const float* data;     // [n, d] 数据
    std::vector<std::vector<size_t>> graph;  // 输出图 [n][K]

    // 迭代参数
    int iterations;        // 迭代次数（默认 10-20）
    float rho;             // 采样率
    bool use_rng;          // 是否使用随机采样

    void compute() {
        // 1. 初始化：随机或使用简单方法初始化近邻
        initialize_graph();

        // 2. 迭代改进
        for (int iter = 0; iter < iterations; iter++) {
            // 2.1 对于每个点的近邻
            for (size_t i = 0; i < n; i++) {
                // 获取当前近邻的近邻（近邻的近邻）
                auto new_candidates = explore_neighbors(i);

                // 2.2 评估新候选
                for (size_t candidate : new_candidates) {
                    float dist = compute_distance(data + i * d,
                                                  data + candidate * d, d);

                    // 2.3 如果更近，则更新近邻列表
                    update_neighbor_list(i, candidate, dist);
                }
            }

            // 2.4 清理和优化图
            prune_graph();
        }
    }

private:
    void initialize_graph() {
        // 初始化策略：随机选择 K 个近邻
        std::mt19937 rng(1234);
        std::uniform_int_distribution<size_t> dist(0, n - 1);

        graph.resize(n);
        for (size_t i = 0; i < n; i++) {
            graph[i].resize(K);
            for (size_t j = 0; j < K; j++) {
                graph[i][j] = dist(rng);
            }
        }
    }

    std::vector<size_t> explore_neighbors(size_t i) {
        std::set<size_t> candidates;

        // 收集所有近邻的近邻
        for (size_t neighbor : graph[i]) {
            if (neighbor < graph.size()) {
                for (size_t nn : graph[neighbor]) {
                    candidates.insert(nn);
                }
            }
        }

        return std::vector<size_t>(candidates.begin(), candidates.end());
    }

    void update_neighbor_list(size_t i, size_t candidate, float dist) {
        // 维护大小为 K 的最小堆
        auto& neighbors = graph[i];

        // 找到最远的近邻
        size_t worst_idx = 0;
        float worst_dist = compute_distance(
            data + i * d, data + neighbors[0] * d, d);

        for (size_t j = 1; j < K; j++) {
            float d = compute_distance(
                data + i * d, data + neighbors[j] * d, d);
            if (d > worst_dist) {
                worst_dist = d;
                worst_idx = j;
            }
        }

        // 如果新候选更近，则替换
        if (dist < worst_dist) {
            neighbors[worst_idx] = candidate;
        }
    }
};

} // namespace faiss
```

### 8.3 NSG 图结构

**NSG 构建流程**：

```cpp
// 位置: faiss/impl/NSG.h

namespace faiss {

struct NSG {
    size_t n;              // 节点数量
    size_t d;              // 维度
    size_t K;              // 每个节点的近邻数

    const float* data;     // [n, d] 数据
    std::vector<std::vector<size_t>> graph;  // [n][K] 邻接表
    size_t enter_point;    // 入口点（导航起点）

    // 构建 NSG
    void build() {
        // 1. 使用 NNDescent 构建 K-NN 图
        NNDescent nnd(n, d, K, data);
        nnd.iterations = 15;
        nnd.compute();
        graph = std::move(nnd.graph);

        // 2. 构建导航结构（Navigator）
        build_navigator();

        // 3. 剪枝优化
        prune();
    }

private:
    void build_navigator() {
        // 选择一个中心点作为导航起点
        // 通常选择数据点中接近质心的点

        // 计算质心
        std::vector<float> centroid(d, 0.0f);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                centroid[j] += data[i * d + j];
            }
        }
        for (size_t j = 0; j < d; j++) {
            centroid[j] /= n;
        }

        // 找最近的质心点
        float min_dist = std::numeric_limits<float>::max();
        enter_point = 0;

        for (size_t i = 0; i < n; i++) {
            float dist = 0;
            for (size_t j = 0; j < d; j++) {
                float diff = data[i * d + j] - centroid[j];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                enter_point = i;
            }
        }
    }

    void prune() {
        // 剪枝策略：移除低质量边
        for (size_t i = 0; i < n; i++) {
            // 确保图的连通性
            // 移除冗余边
            // 控制出度
        }
    }

public:
    // NSG 搜索
    void search(const float* query, size_t k,
                std::vector<float>& distances,
                std::vector<size_t>& labels) {

        // 1. 从导航起点开始
        size_t current = enter_point;
        float best_dist = compute_distance(query, data + current * d, d);

        // 2. 贪心搜索
        std::vector<bool> visited(n, false);
        visited[current] = true;

        bool improved = true;
        while (improved) {
            improved = false;

            // 检查当前点的所有邻居
            for (size_t neighbor : graph[current]) {
                if (neighbor >= n || visited[neighbor]) continue;

                float dist = compute_distance(query, data + neighbor * d, d);

                if (dist < best_dist) {
                    best_dist = dist;
                    current = neighbor;
                    improved = true;
                }

                visited[neighbor] = true;
            }
        }

        // 3. 从最佳点开始 BFS 搜索 Top-K
        bfs_search(query, k, distances, labels, current);
    }

private:
    void bfs_search(const float* query, size_t k,
                    std::vector<float>& distances,
                    std::vector<size_t>& labels,
                    size_t start) {

        // 使用优先队列进行 BFS
        std::priority_queue<std::pair<float, size_t>> pq;
        std::vector<bool> visited(n, false);

        pq.push({0.0f, start});
        visited[start] = true;

        while (!pq.empty() && labels.size() < k) {
            auto [_, node] = pq.top();
            pq.pop();

            float dist = compute_distance(query, data + node * d, d);
            distances.push_back(dist);
            labels.push_back(node);

            // 添加邻居
            for (size_t neighbor : graph[node]) {
                if (!visited[neighbor]) {
                    float d = compute_distance(query, data + neighbor * d, d);
                    pq.push({d, neighbor});
                    visited[neighbor] = true;
                }
            }
        }
    }

    float compute_distance(const float* x, const float* y, size_t d) {
        float dist = 0;
        for (size_t i = 0; i < d; i++) {
            float diff = x[i] - y[i];
            dist += diff * diff;
        }
        return dist;
    }
};

} // namespace faiss
```

### 8.4 NSG vs HNSW 实战对比

```cpp
// 性能对比测试

void compare_graph_indexes() {
    const size_t n = 1000000;
    const size_t d = 128;
    const size_t k = 10;

    // 生成数据
    std::vector<float> data(n * d);
    // ... 填充随机数据 ...

    // HNSW
    IndexHNSW hnsw(d, METRIC_L2);
    hnsw.hnsw.M = 32;
    hnsw.hnsw.ef_construction = 64;
    hnsw.add(n, data.data());

    // NSG (需要使用 IndexNSG)
    IndexNSG nsg(d, METRIC_L2);
    nsg.nsg.K = 32;
    nsg.add(n, data.data());

    // 测试查询性能
    std::vector<float> queries(100 * d);
    // ... 填充查询 ...

    // HNSW 搜索
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> h_dist(100 * k);
    std::vector<size_t> h_labels(100 * k);
    hnsw.search(100, queries.data(), k, h_dist.data(), h_labels.data());
    auto t2 = std::chrono::high_resolution_clock::now();

    // NSG 搜索
    auto t3 = std::chrono::high_resolution_clock::now();
    std::vector<float> n_dist(100 * k);
    std::vector<size_t> n_labels(100 * k);
    nsg.search(100, queries.data(), k, n_dist.data(), n_labels.data());
    auto t4 = std::chrono::high_resolution_clock::now();

    double h_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double n_time = std::chrono::duration<double, std::milli>(t4 - t3).count();

    printf("HNSW: %.2f ms, NSG: %.2f ms\n", h_time, n_time);
    printf("NSG speedup: %.2fx\n", h_time / n_time);
}
```

---

## 第九部分：FastScan（SIMD 优化扫描）

### 9.1 FastScan 概述

**FastScan** 是 Faiss 中专门优化的索引类型，通过 SIMD 指令实现超快速扫描。

**核心优势**：
- 使用 SIMD 打包多个距离计算
- 优化的内存访问模式
- 适用于高吞吐量场景

**支持的操作**：
- `IndexFastScan`：基于 FastScan 的基础索引
- `IndexIVFFastScan`：IVF + FastScan 结合

### 9.2 FastScan 原理

**SIMD 打包技术**：

```cpp
// 位置: faiss/IndexFastScan.h

namespace faiss {

template <MetricType METRIC_TYPE, class C>
struct IndexFastScan : Index {
    size_t M;                     // 子量化器数量
    size_t nbits;                 // 每个量化器的位数
    size_t nlist;                 // 聚类中心数量

    // 编码数据 [ntotal][M]
    AlignedTable<uint8_t> codes;

    // 查找表 [M][2^nbits]
    AlignedTable<float> quantizer_norms;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 1. 计算查找表
        std::vector<float> lut(M * (1 << nbits));

        #pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            const float* xi = x + i * d;

            // 为每个子量化器计算 LUT
            for (size_t m = 0; m < M; m++) {
                for (size_t j = 0; j < (1 << nbits); j++) {
                    lut[m * (1 << nbits) + j] =
                        compute_table_entry(xi, m, j);
                }
            }

            // 2. 使用 SIMD 快速扫描
            simd_scan(n, i, lut.data(), k, distances, labels);
        }
    }

protected:
    void simd_scan(
            idx_t n,
            idx_t query_idx,
            const float* lut,
            idx_t k,
            float* distances,
            idx_t* labels) const {

        // 使用 SIMD 指令并行处理多个向量
        // 这里展示 AVX2 版本

        constexpr size_t SIMDW = 8;  // AVX2 处理 8 个 float

        // 初始化堆
        heap_heapify<C>(k, distances + query_idx * k,
                       labels + query_idx * k);

        // 批量处理
        for (size_t i = 0; i < n; i += SIMDW) {
            // 加载 8 个编码
            __m256i acc_lo = _mm256_setzero_si256();
            __m256i acc_hi = _mm256_setzero_si256();

            // 对于每个子量化器
            for (size_t m = 0; m < M; m++) {
                // 加载 8 个编码
                __m256i codes = load_8_codes(i, m);

                // 查表并累加
                accumulate_8_distances(codes, lut + m * (1 << nbits),
                                      acc_lo, acc_hi);
            }

            // 转换为 float 并更新堆
            update_heap_8(acc_lo, acc_hi, i, k,
                         distances + query_idx * k,
                         labels + query_idx * k);
        }
    }

    __m256i load_8_codes(size_t base_idx, size_t m) const {
        // 从编码表中加载 8 个向量
        // 假设编码按行优先存储
        __m256i result;
        // ... 实现细节 ...
        return result;
    }

    void accumulate_8_distances(
            __m256i codes,
            const float* table,
            __m256i& acc_lo,
            __m256i& acc_hi) {
        // 使用 SIMD 查表并累加距离
        // ... 实现细节 ...
    }
};

} // namespace faiss
```

### 9.3 IVFFastScan 实现

```cpp
// 位置: faiss/IndexIVFFastScan.h

namespace faiss {

template <MetricType METRIC_TYPE, class C>
struct IndexIVFFastScan : IndexFastScan<METRIC_TYPE, C> {

    using IndexFastScan<METRIC_TYPE, C>::codes;
    using IndexFastScan<METRIC_TYPE, C>::M;
    using IndexFastScan<METRIC_TYPE, C>::nbits;

    size_t nprobe;         // 搜索的聚类数
    InvertedLists* invlists;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 1. 找最近的 nprobe 个聚类
        std::vector<idx_t> nearest(n * nprobe);
        quantizer->search(n, x, nprobe, nullptr, nearest.data());

        // 2. 对每个查询
        #pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            // 2.1 计算查找表
            std::vector<float> lut(M * (1 << nbits));
            compute_lut(x + i * d, lut.data());

            // 2.2 搜索选中的倒排表
            search_ivf(
                i,
                nearest.data() + i * nprobe,
                lut.data(),
                k,
                distances + i * k,
                labels + i * k
            );
        }
    }

    void search_ivf(
            idx_t query_idx,
            const idx_t* list_nos,
            const float* lut,
            idx_t k,
            float* distances,
            idx_t* labels) const {

        // 初始化堆
        heap_heapify<C>(k, distances, labels);

        // 扫描所有选中的倒排表
        for (size_t i = 0; i < nprobe; i++) {
            idx_t list_no = list_nos[i];
            if (list_no < 0 || list_no >= nlist) continue;

            // 获取倒排表
            const uint8_t* list_codes = invlists->get_codes(list_no);
            const idx_t* list_ids = invlists->get_ids(list_no);
            size_t list_size = invlists->list_size(list_no);

            // SIMD 扫描
            simd_scan_list(
                list_codes, list_ids, list_size,
                lut, k, distances, labels
            );

            invlists->release_codes(list_no, list_codes);
            invlists->release_ids(list_no, list_ids);
        }
    }

    void simd_scan_list(
            const uint8_t* codes,
            const idx_t* ids,
            size_t size,
            const float* lut,
            idx_t k,
            float* distances,
            idx_t* labels) const {

        // SIMD 优化的倒排表扫描
        constexpr size_t SIMDW = 8;

        for (size_t i = 0; i + SIMDW <= size; i += SIMDW) {
            // 加载 8 个编码
            __m256i acc = _mm256_setzero_si256();

            for (size_t m = 0; m < M; m++) {
                // 从 8 个向量中提取第 m 个子量化器的编码
                __m256i m_codes = gather_8_codes(codes, i, m, size);

                // 查表并累加
                acc = accumulate_8_lut(m_codes, lut + m * (1 << nbits), acc);
            }

            // 转换为 float 并更新堆
            float dists[SIMDW];
            _mm256_storeu_ps(dists, _mm256_cvtepi32_ps(acc));

            for (size_t j = 0; j < SIMDW; j++) {
                if (i + j < size) {
                    heap_push<C>(k, distances, labels,
                                ids[i + j], dists[j]);
                }
            }
        }

        // 处理剩余元素
        for (size_t i = (size / SIMDW) * SIMDW; i < size; i++) {
            float dist = 0;
            for (size_t m = 0; m < M; m++) {
                uint8_t code = codes[i * M + m];
                dist += lut[m * (1 << nbits) + code];
            }
            heap_push<C>(k, distances, labels, ids[i], dist);
        }
    }
};

} // namespace faiss
```

### 9.4 FastScan 性能优化技巧

```cpp
// FastScan 优化策略

struct FastScanOptimizations {

    // 1. 内存对齐优化
    void* allocate_aligned(size_t size) {
        // 使用 64 字节对齐（AVX-512 缓存行）
        void* ptr = nullptr;
        posix_memalign(&ptr, 64, size);
        return ptr;
    }

    // 2. 预取优化
    void prefetch_codes(const uint8_t* codes, size_t size) {
        for (size_t i = 0; i < size; i += 64 / sizeof(uint8_t)) {
            _mm_prefetch((char*)(codes + i), _MM_HINT_T0);
        }
    }

    // 3. 批处理优化
    void batch_search(
            IndexIVFFastScan<>* index,
            const float* queries,
            size_t n_queries,
            size_t k) {

        // 批处理查询以提高缓存利用率
        constexpr size_t batch_size = 32;

        for (size_t i = 0; i < n_queries; i += batch_size) {
            size_t batch = std::min(batch_size, n_queries - i);

            std::vector<float> distances(batch * k);
            std::vector<idx_t> labels(batch * k);

            index->search(
                batch,
                queries + i * index->d,
                k,
                distances.data(),
                labels.data()
            );

            // 处理结果...
        }
    }

    // 4. 查找表缓存优化
    struct CachedLUT {
        std::vector<float> lut;
        size_t last_query_hash;

        bool is_valid(const float* query, size_t d) const {
            // 检查查询是否相似（可以重用 LUT）
            size_t hash = hash_query(query, d);
            return hash == last_query_hash;
        }

        void update(const float* query, size_t d,
                   const std::vector<float>& new_lut) {
            lut = new_lut;
            last_query_hash = hash_query(query, d);
        }

    private:
        size_t hash_query(const float* query, size_t d) const {
            // 简单哈希：取前几个维度
            size_t h = 0;
            for (size_t i = 0; i < std::min(d, size_t(4)); i++) {
                h ^= *(const uint32_t*)(query + i);
            }
            return h;
        }
    };
};
```

---

## 第十部分：Scalar Quantizer & RaBitQ

### 10.1 Scalar Quantizer（标量量化器）

**Scalar Quantizer** 对向量的每个维度独立量化，是 PQ 的简化版本。

```cpp
// 位置: faiss/impl/ScalarQuantizer.h

namespace faiss {

struct ScalarQuantizer {
    enum QuantizerType {
        QT_8bit,           // 8-bit uniform quantization
        QT_4bit,           // 4-bit uniform quantization
        QT_8bit_uniform,   // 8-bit uniform, signed
        QT_4bit_uniform,   // 4-bit uniform, signed
        QT_fp16,           // half-precision floating point
        QT_bf16            // bfloat16
    };

    size_t d;              // 维度
    QuantizerType qtype;   // 量化类型

    // 量化参数
    std::vector<float> vmin;    // 每个维度的最小值
    std::vector<float> vmax;    // 每个维度的最大值
    std::vector<float> scale;   // 缩放因子
    std::vector<float> bias;    // 偏移

    void train(size_t n, const float* x) {
        vmin.resize(d);
        vmax.resize(d);
        scale.resize(d);
        bias.resize(d);

        // 计算每个维度的范围
        #pragma omp parallel for
        for (size_t i = 0; i < d; i++) {
            float min_val = std::numeric_limits<float>::max();
            float max_val = -std::numeric_limits<float>::max();

            for (size_t j = 0; j < n; j++) {
                float val = x[j * d + i];
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }

            vmin[i] = min_val;
            vmax[i] = max_val;

            // 计算量化参数
            switch (qtype) {
                case QT_8bit:
                    // 范围 [0, 255]
                    scale[i] = (max_val - min_val) / 255.0f;
                    bias[i] = min_val;
                    break;

                case QT_8bit_uniform:
                    // 范围 [-128, 127]
                    float range = std::max(fabs(max_val), fabs(min_val));
                    scale[i] = range / 128.0f;
                    bias[i] = 0;
                    break;
            }
        }
    }

    void encode(const float* x, uint8_t* codes) const {
        for (size_t i = 0; i < d; i++) {
            float val = x[i];
            float quantized = (val - bias[i]) / scale[i];

            // 截断到有效范围
            quantized = std::max(0.0f, std::min(255.0f, quantized));
            codes[i] = (uint8_t)std::lround(quantized);
        }
    }

    void decode(const uint8_t* codes, float* x) const {
        for (size_t i = 0; i < d; i++) {
            x[i] = bias[i] + scale[i] * codes[i];
        }
    }
};

} // namespace faiss
```

### 10.2 RaBitQ（Random Bit Quantization）

**RaBitQ** 是 Faiss 中引入的新型量化技术，使用随机投影进行比特级量化。

```cpp
// 位置: faiss/impl/RaBitQuantizer.h

namespace faiss {

struct RaBitQuantizer {
    size_t d;              // 原始维度
    size_t nbits;          // 量化位数
    size_t M;              // 子量化器数量

    // 随机投影矩阵 [d][M]
    std::vector<std::vector<float>> random_matrix;

    // 量化码本 [M][2^nbits]
    std::vector<std::vector<float>> codebooks;

    void train(size_t n, const float* x) {
        // 1. 生成随机投影矩阵
        initialize_random_projection();

        // 2. 投影数据到低维空间
        std::vector<float> projected(n * M);
        project_data(n, x, projected.data());

        // 3. 训练每个子量化器的码本
        for (size_t m = 0; m < M; m++) {
            train_subquantizer(m, n, projected.data());
        }
    }

private:
    void initialize_random_projection() {
        std::mt19937 rng(12345);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        random_matrix.resize(d);
        for (size_t i = 0; i < d; i++) {
            random_matrix[i].resize(M);
            for (size_t j = 0; j < M; j++) {
                random_matrix[i][j] = dist(rng);
            }
        }
    }

    void project_data(size_t n, const float* x, float* projected) {
        // projected[i][m] = sum_j x[i][j] * random_matrix[j][m]
        for (size_t i = 0; i < n; i++) {
            for (size_t m = 0; m < M; m++) {
                float sum = 0;
                for (size_t j = 0; j < d; j++) {
                    sum += x[i * d + j] * random_matrix[j][m];
                }
                projected[i * M + m] = sum;
            }
        }
    }

    void train_subquantizer(size_t m, size_t n, const float* projected) {
        // 对第 m 个维度进行标量量化
        float min_val = std::numeric_limits<float>::max();
        float max_val = -std::numeric_limits<float>::max();

        for (size_t i = 0; i < n; i++) {
            float val = projected[i * M + m];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        // 创建均匀码本
        codebooks[m].resize(1 << nbits);
        for (size_t k = 0; k < (1 << nbits); k++) {
            float t = (float)k / ((1 << nbits) - 1);
            codebooks[m][k] = min_val + t * (max_val - min_val);
        }
    }

public:
    void encode(const float* x, uint8_t* codes) const {
        // 1. 投影
        std::vector<float> projected(M);
        for (size_t m = 0; m < M; m++) {
            float sum = 0;
            for (size_t j = 0; j < d; j++) {
                sum += x[j] * random_matrix[j][m];
            }
            projected[m] = sum;
        }

        // 2. 量化
        for (size_t m = 0; m < M; m++) {
            // 找最近的码本
            float val = projected[m];
            size_t best_code = 0;
            float min_dist = fabs(val - codebooks[m][0]);

            for (size_t k = 1; k < (1 << nbits); k++) {
                float dist = fabs(val - codebooks[m][k]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_code = k;
                }
            }

            codes[m] = (uint8_t)best_code;
        }
    }

    float compute_distance(const uint8_t* codes, const float* query) const {
        // 使用查找表计算距离
        float dist = 0;

        // 投影查询
        std::vector<float> projected_query(M);
        for (size_t m = 0; m < M; m++) {
            float sum = 0;
            for (size_t j = 0; j < d; j++) {
                sum += query[j] * random_matrix[j][m];
            }
            projected_query[m] = sum;
        }

        // 计算距离
        for (size_t m = 0; m < M; m++) {
            float code_val = codebooks[m][codes[m]];
            float diff = projected_query[m] - code_val;
            dist += diff * diff;
        }

        return dist;
    }
};

} // namespace faiss
```

### 10.3 IndexScalarQuantizer 实现

```cpp
// 位置: faiss/IndexScalarQuantizer.h

namespace faiss {

struct IndexScalarQuantizer : Index {
    ScalarQuantizer sq;
    std::vector<uint8_t> codes;  // [ntotal * d]

    IndexScalarQuantizer(size_t d, ScalarQuantizer::QuantizerType qtype)
        : Index(d, METRIC_L2) {
        sq.d = d;
        sq.qtype = qtype;
    }

    void train(idx_t n, const float* x) override {
        sq.train(n, x);
        is_trained = true;
    }

    void add(idx_t n, const float* x) override {
        FAISS_THROW_IF_NOT(is_trained);

        size_t old_size = codes.size();
        codes.resize(old_size + n * d);

        sq.encode_batch(x, codes.data() + old_size, n);
        ntotal += n;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        #pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            // 初始化堆
            heap_heapify<CMin<float, idx_t>>(k,
                                             distances + i * k,
                                             labels + i * k);

            const float* query = x + i * d;

            // 扫描所有编码
            for (idx_t j = 0; j < ntotal; j++) {
                float dist = compute_sq_distance(query, codes.data() + j * d);

                // 更新堆
                heap_push<CMin<float, idx_t>>(
                    k, distances + i * k, labels + i * k, j, dist
                );
            }
        }
    }

private:
    float compute_sq_distance(const float* query, const uint8_t* code) const {
        float dist = 0;
        for (size_t j = 0; j < d; j++) {
            float decoded = sq.bias[j] + sq.scale[j] * code[j];
            float diff = query[j] - decoded;
            dist += diff * diff;
        }
        return dist;
    }
};

} // namespace faiss
```

---

## 第十一部分：Binary Index（二进制向量索引）

### 11.1 二进制向量基础

**二进制向量**（Binary Vectors）是元素为 0 或 1 的向量，常用 Hamming 距离度量相似度。

**应用场景**：
- 图像哈希（感知哈希、pHash）
- 文本二值化特征
- 深度学习二值化嵌入

**Hamming 距离**：两个二进制向量中不同位的数量。

```cpp
// 位置: faiss/IndexBinary.h

namespace faiss {

// 二进制索引基类
struct IndexBinary {
    idx_t d;              // 位数（bit）
    idx_t ntotal;         // 向量总数
    idx_t code_size;      // 每个向量的字节数 = (d + 7) / 8

    IndexBinary(idx_t d = 0) : d(d), ntotal(0), code_size(0) {
        code_size = (d + 7) / 8;
    }

    virtual ~IndexBinary() = default;

    virtual void add(idx_t n, const uint8_t* x) = 0;
    virtual void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels) const = 0;
};

// 二进制 Flat 索引
struct IndexBinaryFlat : IndexBinary {
    std::vector<uint8_t> codes;  // [ntotal * code_size]

    IndexBinaryFlat(idx_t d) : IndexBinary(d) {}

    void add(idx_t n, const uint8_t* x) override {
        codes.insert(codes.end(), x, x + n * code_size);
        ntotal += n;
    }

    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels) const override {

        #pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* query = x + i * code_size;

            // 初始化堆（最大堆，找最小距离）
            std::priority_queue<std::pair<int32_t, idx_t>> heap;

            // 计算与所有向量的 Hamming 距离
            for (idx_t j = 0; j < ntotal; j++) {
                int32_t dist = hamming_distance(
                    query,
                    codes.data() + j * code_size,
                    code_size
                );

                // 维护大小为 k 的堆
                if (heap.size() < (size_t)k) {
                    heap.push({dist, j});
                } else if (dist < heap.top().first) {
                    heap.pop();
                    heap.push({dist, j});
                }
            }

            // 输出结果（从小到大）
            for (idx_t j = 0; j < k && !heap.empty(); j++) {
                distances[(k - 1 - j) + i * k] = heap.top().first;
                labels[(k - 1 - j) + i * k] = heap.top().second;
                heap.pop();
            }
        }
    }

private:
    // Hamming 距离计算（优化版本）
    static int32_t hamming_distance(
            const uint8_t* a,
            const uint8_t* b,
            size_t size) {

        int32_t dist = 0;
        for (size_t i = 0; i < size; i++) {
            // POPCNT：计算 1 的个数
            dist += __builtin_popcount(a[i] ^ b[i]);
        }
        return dist;
    }
};

} // namespace faiss
```

### 11.2 Hamming 距离 SIMD 优化

```cpp
// 位置: faiss/utils/hamming.cpp

namespace faiss {

// 使用 POPCNT 指令的 Hamming 距离计算
inline int32_t hamming_distance_popcnt(
        const uint8_t* a,
        const uint8_t* b,
        size_t size) {

    int32_t dist = 0;
    size_t i = 0;

    // 64 位对齐的快速路径
    const uint64_t* a64 = (const uint64_t*)a;
    const uint64_t* b64 = (const uint64_t*)b;
    size_t n64 = size / sizeof(uint64_t);

    for (i = 0; i < n64; i++) {
        uint64_t x = a64[i] ^ b64[i];
        dist += _mm_popcnt_u64(x);  // POPCNT 指令
    }

    // 处理剩余字节
    i = n64 * sizeof(uint64_t);
    for (; i < size; i++) {
        dist += __builtin_popcount(a[i] ^ b[i]);
    }

    return dist;
}

// AVX2 批量 Hamming 距离计算
void batch_hamming_distance_avx2(
        const uint8_t* query,
        const uint8_t* codes,
        size_t n,
        size_t code_size,
        int32_t* distances) {

    for (size_t i = 0; i < n; i++) {
        distances[i] = hamming_distance_popcnt(
            query,
            codes + i * code_size,
            code_size
        );
    }
}

} // namespace faiss
```

### 11.3 IndexBinaryIVF

```cpp
// 位置: faiss/IndexBinaryIVF.h

namespace faiss {

struct IndexBinaryIVF : IndexBinary {
    size_t nlist;              // 聚类数
    size_t nprobe;             // 搜索聚类数

    IndexBinary* quantizer;    // 粗量化器
    InvertedLists* invlists;   // 倒排表

    void train(idx_t n, const uint8_t* x) {
        // 使用 K-Means 训练聚类中心
        // 注意：二进制向量需要特殊的距离度量
    }

    void add(idx_t n, const uint8_t* x) override {
        // 1. 分配到聚类
        std::vector<idx_t> list_nos(n);
        quantizer->assign(n, x, list_nos.data());

        // 2. 添加到倒排表
        for (idx_t i = 0; i < n; i++) {
            idx_t list_no = list_nos[i];
            invlists->add_entry(
                list_no,
                ntotal + i,
                x + i * code_size
            );
        }
        ntotal += n;
    }

    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels) const override {

        for (idx_t i = 0; i < n; i++) {
            const uint8_t* query = x + i * code_size;

            // 1. 找最近的聚类
            std::vector<idx_t> nearby(nprobe);
            quantizer->search(1, query, nprobe, nullptr, nearby.data());

            // 2. 搜索倒排表
            std::priority_queue<std::pair<int32_t, idx_t>> heap;

            for (idx_t list_no : nearby) {
                const uint8_t* list_codes = invlists->get_codes(list_no);
                const idx_t* list_ids = invlists->get_ids(list_no);
                size_t list_size = invlists->list_size(list_no);

                for (size_t j = 0; j < list_size; j++) {
                    int32_t dist = hamming_distance_popcnt(
                        query,
                        list_codes + j * code_size,
                        code_size
                    );

                    if (heap.size() < (size_t)k || dist < heap.top().first) {
                        if (heap.size() >= (size_t)k) {
                            heap.pop();
                        }
                        heap.push({dist, list_ids[j]});
                    }
                }

                invlists->release_codes(list_no, list_codes);
                invlists->release_ids(list_no, list_ids);
            }

            // 3. 输出结果
            for (idx_t j = 0; j < k && !heap.empty(); j++) {
                labels[(k - 1 - j) + i * k] = heap.top().second;
                distances[(k - 1 - j) + i * k] = heap.top().first;
                heap.pop();
            }
        }
    }
};

} // namespace faiss
```

---

## 第十二部分：GPU 实现

### 12.1 GPU 架构概述

**Faiss GPU** 使用 CUDA/CUDA 实现高性能向量搜索。

**核心组件**：
- `GpuResources`：GPU 资源管理（内存、流）
- `GpuIndex`：GPU 索引基类
- `GpuCloner`：CPU-GPU 索引转换

### 12.2 GPU 资源管理

```cpp
// 位置: faiss/gpu/GpuResources.h

namespace faiss { namespace gpu {

struct GpuResources {
    virtual ~GpuResources() = default;

    // 获取默认流
    virtual cudaStream_t getDefaultStream(int device) = 0;

    // 分配内存
    virtual void* allocMemory(int device, size_t size) = 0;

    // 释放内存
    virtual void freeMemory(int device, void* p) = 0;

    // 内存拷贝
    virtual void copyToGPU(
            int device,
            void* dst,
            const void* src,
            size_t size) = 0;

    virtual void copyFromGPU(
            int device,
            void* dst,
            const void* src,
            size_t size) = 0;
};

// 默认 GPU 资源实现
struct DefaultGpuResources : GpuResources {
    std::vector<std::unique_ptr<CudaMemoryWorkspace>> workspaces;
    std::vector<cudaStream_t> streams;

    DefaultGpuResources() {
        // 初始化所有可见 GPU
        int num_devices = getNumDevices();
        for (int i = 0; i < num_devices; i++) {
            workspaces.emplace_back(
                std::make_unique<CudaMemoryWorkspace>(i)
            );

            cudaStream_t stream;
            CUDA_VERIFY(cudaStreamCreate(&stream));
            streams.push_back(stream);
        }
    }

    ~DefaultGpuResources() {
        for (auto stream : streams) {
            cudaStreamDestroy(stream);
        }
    }

    cudaStream_t getDefaultStream(int device) override {
        return streams[device];
    }

    void* allocMemory(int device, size_t size) override {
        return workspaces[device]->alloc(size);
    }

    void freeMemory(int device, void* p) override {
        workspaces[device]->free(p);
    }

    void copyToGPU(int device, void* dst, const void* src, size_t size) override {
        CUDA_VERIFY(cudaMemcpyAsync(
            dst, src, size,
            cudaMemcpyHostToDevice,
            streams[device]
        ));
    }

    void copyFromGPU(int device, void* dst, const void* src, size_t size) override {
        CUDA_VERIFY(cudaMemcpyAsync(
            dst, src, size,
            cudaMemcpyDeviceToHost,
            streams[device]
        ));
    }

private:
    int getNumDevices() {
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        return num_devices;
    }
};

} // namespace gpu
} // namespace faiss
```

### 12.3 GPU Flat 索引

```cpp
// 位置: faiss/gpu/GpuIndexFlat.h

namespace faiss { namespace gpu {

struct GpuIndexFlat : Index {
    // 基础数据
    size_t d;
    std::shared_ptr<GpuResources> resources;
    int device;

    // GPU 数据
    DeviceVector<float> vectors;  // [ntotal, d]

    GpuIndexFlat(
            std::shared_ptr<GpuResources> res,
            int device,
            size_t d)
        : Index(d, METRIC_L2),
          resources(res),
          device(device),
          vectors_(res, device, 0) {

    }

    void add(idx_t n, const float* x) override {
        // 拷贝数据到 GPU
        size_t old_size = vectors_.size();
        vectors_.resize(old_size + n * d, resources->getDefaultStream(device));

        // 拷贝新数据
        resources->copyToGPU(
            device,
            vectors_.data() + old_size,
            x,
            n * d * sizeof(float)
        );

        ntotal += n;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        auto stream = resources->getDefaultStream(device);

        // 1. 拷贝查询到 GPU
        DeviceVector<float> queries(resources, device, n * d);
        resources->copyToGPU(device, queries.data(), x, n * d * sizeof(float));

        // 2. 在 GPU 上计算距离
        DeviceVector<float> gpu_distances(resources, device, n * k);
        DeviceVector<idx_t> gpu_labels(resources, device, n * k);

        // 调用 CUDA kernel
        runL2DistanceKernel(
            stream,
            vectors_.data(), ntotal, d,
            queries.data(), n,
            gpu_distances.data(),
            gpu_labels.data(),
            k
        );

        // 3. 拷贝结果回 CPU
        resources->copyFromGPU(
            device,
            distances,
            gpu_distances.data(),
            n * k * sizeof(float)
        );

        resources->copyFromGPU(
            device,
            labels,
            gpu_labels.data(),
            n * k * sizeof(idx_t)
        );

        // 同步
        CUDA_VERIFY(cudaStreamSynchronize(stream));
    }
};

// CUDA kernel 实现（简化版）
__global__ void l2DistanceKernel(
        const float* vectors,
        size_t ntotal,
        size_t d,
        const float* queries,
        size_t n_queries,
        float* distances,
        idx_t* labels,
        size_t k) {

    // 每个线程块处理一个查询
    size_t query_idx = blockIdx.x;
    if (query_idx >= n_queries) return;

    const float* query = queries + query_idx * d;

    // 使用共享内存计算距离
    extern __shared__ float shared_dist[];

    // 每个线程计算一个向量的距离
    size_t vec_idx = threadIdx.x;
    float min_dist = 1e20;
    idx_t min_idx = -1;

    if (vec_idx < ntotal) {
        const float* vec = vectors + vec_idx * d;
        float dist = 0.0f;

        for (size_t i = 0; i < d; i++) {
            float diff = query[i] - vec[i];
            dist += diff * diff;
        }

        min_dist = dist;
        min_idx = vec_idx;
    }

    shared_dist[threadIdx.x] = min_dist;
    __syncthreads();

    // 找 Top-K（简化版）
    if (threadIdx.x == 0) {
        for (size_t i = 1; i < min((size_t)blockDim.x, ntotal); i++) {
            if (shared_dist[i] < min_dist) {
                min_dist = shared_dist[i];
                min_idx = i;
            }
        }

        distances[query_idx * k] = min_dist;
        labels[query_idx * k] = min_idx;
    }
}

void runL2DistanceKernel(
        cudaStream_t stream,
        const float* vectors,
        size_t ntotal,
        size_t d,
        const float* queries,
        size_t n_queries,
        float* distances,
        idx_t* labels,
        size_t k) {

    int threads = 256;
    int blocks = n_queries;

    size_t shared_mem = threads * sizeof(float);

    l2DistanceKernel<<<blocks, threads, shared_mem, stream>>>(
        vectors, ntotal, d,
        queries, n_queries,
        distances, labels,
        k
    );
}

} // namespace gpu
} // namespace faiss
```

### 12.4 GPU IVF 索引

```cpp
// 位置: faiss/gpu/GpuIndexIVFFlat.h

namespace faiss { namespace gpu {

struct GpuIndexIVFFlat : IndexIVFInterface {
    std::shared_ptr<GpuResources> resources;
    int device;

    GpuIndexIVFFlat(
            std::shared_ptr<GpuResources> res,
            int device,
            Index* quantizer,
            size_t nlist)
        : IndexIVFInterface(quantizer, nlist),
          resources(res),
          device(device) {

        // 在 GPU 上分配倒排表内存
        // ...
    }

    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params,
            IndexIVFStats* stats) const override {

        auto stream = resources->getDefaultStream(device);

        // 拷贝数据到 GPU
        // ...（省略拷贝代码）

        // 调用优化的 IVF 搜索 kernel
        runIVFSearchKernel(
            stream,
            x, n, d, k,
            assign, nprobe,
            gpu_invlists,
            distances, labels
        );

        // 拷贝结果回 CPU
        // ...
    }
};

} // namespace gpu
} // namespace faiss
```

### 12.5 CPU-GPU 互操作

```cpp
// 位置: faiss/gpu/GpuCloner.h

namespace faiss { namespace gpu {

// 将 CPU 索引复制到 GPU
Index* index_cpu_to_gpu(
        std::shared_ptr<GpuResources> resources,
        int device,
        const Index* index) {

    if (const auto* flat = dynamic_cast<const IndexFlat*>(index)) {
        return new GpuIndexFlat(resources, device, flat->d);
    } else if (const auto* ivf = dynamic_cast<const IndexIVFFlat*>(index)) {
        // 复制 quantizer
        Index* gpu_quantizer = index_cpu_to_gpu(
            resources, device, ivf->quantizer
        );

        auto* gpu_ivf = new GpuIndexIVFFlat(
            resources, device, gpu_quantizer, ivf->nlist
        );

        // 复制倒排表数据
        // ...

        return gpu_ivf;
    }
    // ... 其他索引类型

    return nullptr;
}

// 将 GPU 索引复制回 CPU
Index* index_gpu_to_cpu(const Index* gpu_index) {
    // 反向转换
    // ...
}

} // namespace gpu
} // namespace faiss
```

---

## 第十三部分：Composite 高级索引

### 13.1 IndexRefine（精炼索引）

**IndexRefine** 结合快速粗索引和精确重算，平衡速度和精度。

```cpp
// 位置: faiss/IndexRefine.h

namespace faiss {

struct IndexRefine : Index {
    Index* base_index;      // 粗索引（快速）
    Index* refine_index;    // 精索引（慢速但精确）
    float k_factor;         // 精索返回的候选数 = k * k_factor

    IndexRefine(Index* base, Index* refine, float k_factor = 10.0f)
        : Index(base->d, base->metric_type),
          base_index(base),
          refine_index(refine),
          k_factor(k_factor) {
        ntotal = base->ntotal;
        is_trained = base->is_trained;
    }

    void add(idx_t n, const float* x) override {
        base_index->add(n, x);
        refine_index->add(n, x);
        ntotal += n;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 1. 粗搜索：返回更多候选
        idx_t k_refine = (idx_t)(k * k_factor);
        std::vector<float> base_distances(n * k_refine);
        std::vector<idx_t> base_labels(n * k_refine);

        base_index->search(
            n, x, k_refine,
            base_distances.data(),
            base_labels.data()
        );

        // 2. 对每个查询的候选进行精确计算
        #pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            // 初始化堆
            heap_heapify<CMin<float, idx_t>>(
                k, distances + i * k, labels + i * k
            );

            // 对每个候选进行精确距离计算
            const float* query = x + i * d;

            for (idx_t j = 0; j < k_refine; j++) {
                idx_t candidate = base_labels[i * k_refine + j];
                if (candidate < 0) continue;

                // 使用精索引重新计算距离
                float refined_dist = compute_refined_distance(
                    query, candidate
                );

                // 更新堆
                heap_push<CMin<float, idx_t>>(
                    k,
                    distances + i * k,
                    labels + i * k,
                    candidate,
                    refined_dist
                );
            }
        }
    }

private:
    float compute_refined_distance(const float* query, idx_t id) const {
        // 从精索引重构向量并计算精确距离
        std::vector<float> vec(d);
        refine_index->reconstruct(id, vec.data());

        float dist = 0;
        for (size_t i = 0; i < d; i++) {
            float diff = query[i] - vec[i];
            dist += diff * diff;
        }
        return dist;
    }
};

} // namespace faiss
```

### 13.2 IndexPreTransform（预处理索引）

**IndexPreTransform** 在索引前应用向量变换（如 PCA、归一化）。

```cpp
// 位置: faiss/IndexPreTransform.h

namespace faiss {

struct IndexPreTransform : Index {
    Index* sub_index;       // 底层索引
    VectorTransform* transform;  // 向量变换
    bool own_transform;     // 是否拥有 transform 所有权

    IndexPreTransform(Index* sub_index, VectorTransform* transform)
        : Index(sub_index->d, sub_index->metric_type),
          sub_index(sub_index),
          transform(transform),
          own_transform(false) {
    }

    ~IndexPreTransform() {
        delete sub_index;
        if (own_transform) {
            delete transform;
        }
    }

    void add(idx_t n, const float* x) override {
        // 1. 应用变换
        std::vector<float> transformed(n * transform->d_out);
        transform->apply(n, x, transformed.data());

        // 2. 添加到子索引
        sub_index->add(n, transformed.data());
        ntotal += n;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 1. 变换查询
        std::vector<float> transformed(n * transform->d_out);
        transform->apply(n, x, transformed.data());

        // 2. 在变换后的空间搜索
        sub_index->search(
            n, transformed.data(), k, distances, labels, params
        );
    }
};

// 使用示例：PCA + IVF
Index* create_pca_ivf_index(size_t d, size_t d_out, size_t nlist) {
    // 1. 创建 PCA 变换
    auto* pca = new PCAMatrix(d, d_out);
    // pca->train(...);  // 需要先训练

    // 2. 创建 IVF 索引
    auto* ivf = new IndexIVFFlat(d_out, nlist, METRIC_L2);

    // 3. 组合
    auto* index = new IndexPreTransform(ivf, pca);
    index->own_transform = true;

    return index;
}

} // namespace faiss
```

### 13.3 IndexShards（分片索引）

**IndexShards** 将数据分片到多个索引，支持并行搜索。

```cpp
// 位置: faiss/IndexShards.h

namespace faiss {

struct IndexShards : Index {
    std::vector<std::unique_ptr<Index>> shard_indexes;  // 分片索引
    bool threaded;                                      // 是否使用多线程

    IndexShards(size_t d, bool threaded = true)
        : Index(d, METRIC_L2), threaded(threaded) {
    }

    void add_shard(Index* shard) {
        shard_indexes.emplace_back(shard);
        ntotal += shard->ntotal;
    }

    void add(idx_t n, const float* x) override {
        // 将数据均匀分配到各个分片
        size_t nshards = shard_indexes.size();
        size_t shard_n = (n + nshards - 1) / nshards;

        for (size_t i = 0; i < nshards; i++) {
            size_t start = i * shard_n;
            size_t end = std::min(start + shard_n, n);

            if (start < n) {
                shard_indexes[i]->add(end - start, x + start * d);
            }
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

        const size_t nshards = shard_indexes.size();

        // 为每个分片分配结果缓冲区
        std::vector<std::vector<float>> shard_distances(nshards);
        std::vector<std::vector<idx_t>> shard_labels(nshards);

        for (size_t i = 0; i < nshards; i++) {
            shard_distances[i].resize(n * k);
            shard_labels[i].resize(n * k);
        }

        if (threaded) {
            // 并行搜索所有分片
            std::vector<std::thread> threads;
            for (size_t i = 0; i < nshards; i++) {
                threads.emplace_back([this, i, n, x, k, &shard_distances, &shard_labels]() {
                    shard_indexes[i]->search(
                        n, x, k,
                        shard_distances[i].data(),
                        shard_labels[i].data()
                    );
                });
            }

            for (auto& t : threads) {
                t.join();
            }
        } else {
            // 串行搜索
            for (size_t i = 0; i < nshards; i++) {
                shard_indexes[i]->search(
                    n, x, k,
                    shard_distances[i].data(),
                    shard_labels[i].data()
                );
            }
        }

        // 合并结果
        #pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            // 初始化堆
            heap_heapify<CMin<float, idx_t>>(
                k, distances + i * k, labels + i * k
            );

            // 合并所有分片的结果
            for (size_t s = 0; s < nshards; s++) {
                for (idx_t j = 0; j < k; j++) {
                    heap_push<CMin<float, idx_t>>(
                        k,
                        distances + i * k,
                        labels + i * k,
                        shard_labels[s][i * k + j],
                        shard_distances[s][i * k + j]
                    );
                }
            }
        }
    }
};

} // namespace faiss
```

### 13.4 IndexReplicas（副本索引）

**IndexReplicas** 在多个设备上维护索引副本，支持并行查询。

```cpp
// 位置: faiss/IndexReplicas.h

namespace faiss {

struct IndexReplicas : Index {
    std::vector<std::unique_ptr<Index>> replicas;  // 副本索引

    IndexReplicas(size_t d) : Index(d, METRIC_L2) {
    }

    void add_replica(Index* replica) {
        replicas.emplace_back(replica);
    }

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
            idx_t* labels,
            const SearchParameters* params) const override {

        // 所有副本的结果相同，只需在一个副本上搜索
        replicas[0]->search(n, x, k, distances, labels, params);
    }

    // 并行批量搜索
    void batch_search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const {

        size_t nreplicas = replicas.size();

        // 将查询分配到不同副本
        #pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            size_t replica_id = i % nreplicas;
            size_t local_idx = i / nreplicas;

            // 这里简化了，实际需要更复杂的分配策略
            replicas[replica_id]->search(
                1, x + i * d, k,
                distances + i * k,
                labels + i * k
            );
        }
    }
};

} // namespace faiss
```

---

## 第十四部分：Residual Quantizer（残差量化器）

### 14.1 算法原理

**Residual Quantizer (RQ)** 是一种迭代量化技术，通过逐级量化残差来提高精度。

**核心思想**：
```
原始向量: x
第1级量化: q1 = quantize(x)
第1级残差: r1 = x - q1
第2级量化: q2 = quantize(r1)
第2级残差: r2 = r1 - q2
...
最终编码: [code1, code2, ..., codeM]
重构: x_approx = q1 + q2 + ... + qM
```

**与 PQ 的对比**：

| 特性 | Product Quantizer | Residual Quantizer |
|------|------------------|-------------------|
| 量化方式 | 并行（子向量独立） | 串行（逐级残差） |
| 内存占用 | 固定 | 可变（级数越多越大） |
| 精度 | 中等 | 高（残差逐级减小） |
| 计算复杂度 | 低 | 中等 |

### 14.2 Faiss 源码实现

```cpp
// 位置: faiss/impl/ResidualQuantizer.h

namespace faiss {

struct ResidualQuantizer : AdditiveQuantizer {
    size_t M;              // 量化级数
    size_t nbits;          // 每级的位数

    // 每级的量化器
    std::vector<std::vector<float>> codebooks;  // [M][2^nbits][d]
    std::vector<float> norms;                   // [M][2^nbits] 码本范数

    // 搜索参数
    size_t max_beam_size;  // Beam search 的最大宽度

    ResidualQuantizer(size_t d, size_t M, size_t nbits = 8)
        : AdditiveQuantizer(d, M, nbits),
          M(M), nbits(nbits), max_beam_size(32) {
    }

    void train(size_t n, const float* x) override {
        // 迭代训练每级量化器
        std::vector<float> residuals(x, x + n * d);

        for (size_t m = 0; m < M; m++) {
            // 在当前残差上训练码本
            train_stage(m, n, residuals.data());

            // 更新残差
            update_residuals(m, n, residuals.data());
        }
    }

    void encode(const float* x, uint8_t* codes) const override {
        // 逐级编码
        std::vector<float> residual(x, x + d);

        for (size_t m = 0; m < M; m++) {
            // 在当前残差上找最近的码本
            uint8_t code = find_nearest_codebook(m, residual.data());
            codes[m] = code;

            // 更新残差
            const float* centroid = &codebooks[m][code * d];
            for (size_t i = 0; i < d; i++) {
                residual[i] -= centroid[i];
            }
        }
    }

    void decode(const uint8_t* codes, float* x) const override {
        // 叠加所有码本
        std::fill(x, x + d, 0.0f);

        for (size_t m = 0; m < M; m++) {
            uint8_t code = codes[m];
            const float* centroid = &codebooks[m][code * d];

            for (size_t i = 0; i < d; i++) {
                x[i] += centroid[i];
            }
        }
    }

protected:
    void train_stage(size_t m, size_t n, const float* residuals) {
        size_t K = 1 << nbits;

        // 使用 K-Means 训练当前级的码本
        std::vector<float> centroids(K * d, 0.0f);
        std::vector<size_t> counts(K, 0);
        std::vector<uint8_t> assignments(n);

        // 初始化：随机选择 K 个残差作为初始中心
        std::mt19937 rng(12345 + m);
        std::uniform_int_distribution<size_t> dist(0, n - 1);

        for (size_t k = 0; k < K; k++) {
            size_t idx = dist(rng);
            for (size_t i = 0; i < d; i++) {
                centroids[k * d + i] = residuals[idx * d + i];
            }
        }

        // K-Means 迭代
        for (int iter = 0; iter < 25; iter++) {
            // 分配到最近的中心
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                float min_dist = std::numeric_limits<float>::max();
                size_t best_k = 0;

                for (size_t k = 0; k < K; k++) {
                    float dist = 0;
                    for (size_t j = 0; j < d; j++) {
                        float diff = residuals[i * d + j] - centroids[k * d + j];
                        dist += diff * diff;
                    }

                    if (dist < min_dist) {
                        min_dist = dist;
                        best_k = k;
                    }
                }

                assignments[i] = best_k;
            }

            // 更新中心
            std::fill(centroids.begin(), centroids.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t i = 0; i < n; i++) {
                size_t k = assignments[i];
                for (size_t j = 0; j < d; j++) {
                    centroids[k * d + j] += residuals[i * d + j];
                }
                counts[k]++;
            }

            for (size_t k = 0; k < K; k++) {
                if (counts[k] > 0) {
                    for (size_t j = 0; j < d; j++) {
                        centroids[k * d + j] /= counts[k];
                    }
                }
            }
        }

        // 保存码本
        codebooks[m] = centroids;

        // 计算码本范数（用于快速距离计算）
        norms[m].resize(K);
        for (size_t k = 0; k < K; k++) {
            float norm = 0;
            for (size_t i = 0; i < d; i++) {
                norm += centroids[k * d + i] * centroids[k * d + i];
            }
            norms[m][k] = norm;
        }
    }

    void update_residuals(size_t m, size_t n, float* residuals) {
        // 减去当前级的量化值
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            uint8_t code = find_nearest_codebook(m, residuals + i * d);
            const float* centroid = &codebooks[m][code * d];

            for (size_t j = 0; j < d; j++) {
                residuals[i * d + j] -= centroid[j];
            }
        }
    }

    uint8_t find_nearest_codebook(size_t m, const float* vec) const {
        size_t K = 1 << nbits;
        float min_dist = std::numeric_limits<float>::max();
        uint8_t best_code = 0;

        for (size_t k = 0; k < K; k++) {
            float dist = 0;
            for (size_t i = 0; i < d; i++) {
                float diff = vec[i] - codebooks[m][k * d + i];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_code = k;
            }
        }

        return best_code;
    }
};

} // namespace faiss
```

### 14.3 使用查找表的快速距离计算

```cpp
// ResidualQuantizer 的 LUT 距离计算

void ResidualQuantizer::compute_distance_LUT(
        const float* query,
        float* lut) const {

    // LUT 布局: [M][2^nbits]
    // lut[m][k] = ||query - centroid_mk||²

    for (size_t m = 0; m < M; m++) {
        for (size_t k = 0; k < (size_t)(1 << nbits); k++) {
            float dist = 0;
            for (size_t i = 0; i < d; i++) {
                float diff = query[i] - codebooks[m][k * d + i];
                dist += diff * diff;
            }
            lut[m * (1 << nbits) + k] = dist;
        }
    }
}

float ResidualQuantizer::compute_distance_from_codes(
        const uint8_t* codes,
        const float* lut) const {

    // RQ 的距离计算需要考虑累加效应
    // 使用 beam search 寻找最优组合

    float min_total_dist = std::numeric_limits<float>::max();

    // 简化版本：直接叠加所有级的 LUT 值
    float total_dist = 0;
    for (size_t m = 0; m < M; m++) {
        total_dist += lut[m * (1 << nbits) + codes[m]];
    }

    return total_dist;
}
```

### 14.4 Beam Search 优化

```cpp
// 使用 Beam Search 优化 RQ 距离计算

struct BeamSearchState {
    float cost;               // 当前累积代价
    std::vector<uint8_t> codes;  // 当前编码路径

    bool operator>(const BeamSearchState& other) const {
        return cost > other.cost;
    }
};

float ResidualQuantizer::compute_distance_beam_search(
        const float* query,
        const uint8_t* codes) const {

    // 计算查询的查找表
    std::vector<float> lut(M * (1 << nbits));
    compute_distance_LUT(query, lut.data());

    // Beam search
    std::priority_queue<BeamSearchState,
                       std::vector<BeamSearchState>,
                       std::greater<BeamSearchState>> beam;

    // 初始化：第0级的所有可能
    size_t K = 1 << nbits;
    for (size_t k = 0; k < K; k++) {
        BeamSearchState state;
        state.cost = lut[k];
        state.codes.push_back(k);
        beam.push(state);
    }

    // 逐级扩展
    for (size_t m = 1; m < M; m++) {
        std::priority_queue<BeamSearchState,
                           std::vector<BeamSearchState>,
                           std::greater<BeamSearchState>> next_beam;

        while (!beam.empty() && next_beam.size() < max_beam_size) {
            auto state = beam.top();
            beam.pop();

            // 扩展到下一级
            for (size_t k = 0; k < K; k++) {
                BeamSearchState new_state = state;
                new_state.cost += lut[m * K + k];
                new_state.codes.push_back(k);
                next_beam.push(new_state);
            }
        }

        beam = std::move(next_beam);
    }

    // 返回最小代价
    return beam.top().cost;
}
```

### 14.5 IndexIVFResidualQuantizer

```cpp
// IVF + RQ 组合索引

struct IndexIVFResidualQuantizer : IndexIVFInterface {
    ResidualQuantizer rq;     // 残差量化器
    InvertedLists* invlists;  // 倒排表

    void add(idx_t n, const float* x) override {
        // 量化
        std::vector<uint8_t> codes(n * rq.M);
        rq.encode_batch(x, codes.data(), n);

        // 分配到聚类
        std::vector<idx_t> list_nos(n);
        quantizer->assign(n, x, list_nos.data());

        // 添加到倒排表
        for (idx_t i = 0; i < n; i++) {
            invlists->add_entry(
                list_nos[i],
                ntotal + i,
                codes.data() + i * rq.M
            );
        }

        ntotal += n;
    }

    void search_preassigned(
            idx_t n, const float* x, idx_t k,
            const idx_t* assign, const float* centroid_dis,
            float* distances, idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params,
            IndexIVFStats* stats) const override {

        // 对每个查询
        #pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            const float* query = x + i * d;

            // 计算查找表
            std::vector<float> lut(rq.M * (1 << rq.nbits));
            rq.compute_distance_LUT(query, lut.data());

            // 初始化堆
            heap_heapify<CMin<float, idx_t>>(
                k, distances + i * k, labels + i * k
            );

            // 搜索选中的倒排表
            for (size_t j = 0; j < nprobe; j++) {
                idx_t list_no = assign[i * nprobe + j];
                if (list_no < 0) continue;

                const uint8_t* list_codes = invlists->get_codes(list_no);
                const idx_t* list_ids = invlists->get_ids(list_no);
                size_t list_size = invlists->list_size(list_no);

                // 扫描倒排表
                for (size_t l = 0; l < list_size; l++) {
                    const uint8_t* code = list_codes + l * rq.M;
                    float dist = rq.compute_distance_from_codes(code, lut.data());

                    heap_push<CMin<float, idx_t>>(
                        k, distances + i * k, labels + i * k,
                        list_ids[l], dist
                    );
                }

                invlists->release_codes(list_no, list_codes);
                invlists->release_ids(list_no, list_ids);
            }
        }
    }
};
```

---

## 第十五部分：SIMD 距离计算优化

### 15.1 SIMD 基础

**SIMD (Single Instruction Multiple Data)** 允许一条指令同时处理多个数据。

**常见 SIMD 指令集**：

| 指令集 | 位宽 | 并行 float 数 | 并发 int 数 |
|--------|------|--------------|-------------|
| SSE | 128-bit | 4 | 4 (32-bit) |
| AVX | 256-bit | 8 | 8 (32-bit) |
| AVX2 | 256-bit | 8 | 8 (32-bit) |
| AVX-512 | 512-bit | 16 | 16 (32-bit) |
| ARM NEON | 128-bit | 4 | 4 (32-bit) |

### 15.2 L2 距离的 SIMD 实现

```cpp
// 位置: faiss/utils/distances_simd.cpp

namespace faiss {

// AVX2 实现的 L2 距离计算
inline float fvec_L2sqr_avx2(const float* x, const float* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;
    // 处理 8 个一组（AVX2 可以同时处理 8 个 float）
    for (; i + 8 <= d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);     // 加载 8 个 x
        __m256 vy = _mm256_loadu_ps(y + i);     // 加载 8 个 y
        __m256 diff = _mm256_sub_ps(vx, vy);    // diff = x - y
        __m256 sq = _mm256_mul_ps(diff, diff);  // sq = diff * diff
        sum = _mm256_add_ps(sum, sq);           // sum += sq
    }

    // 水平求和
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(sum_low, sum_high);

    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);

    float result = _mm_cvtss_f32(sums);

    // 处理剩余元素
    for (; i < d; i++) {
        float diff = x[i] - y[i];
        result += diff * diff;
    }

    return result;
}

// AVX-512 实现（更快的版本）
inline float fvec_L2sqr_avx512(const float* x, const float* y, size_t d) {
    __m512 sum = _mm512_setzero_ps();

    size_t i = 0;
    // 处理 16 个一组
    for (; i + 16 <= d; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);
        __m512 diff = _mm512_sub_ps(vx, vy);
        __m512 sq = _mm512_mul_ps(diff, diff);
        sum = _mm512_add_ps(sum, sq);
    }

    // 水平求和
    float result = _mm512_reduce_add_ps(sum);

    // 处理剩余元素
    for (; i < d; i++) {
        float diff = x[i] - y[i];
        result += diff * diff;
    }

    return result;
}

// 内积的 SIMD 实现
inline float fvec_inner_product_avx2(const float* x, const float* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 prod = _mm256_mul_ps(vx, vy);   // prod = x * y
        sum = _mm256_add_ps(sum, prod);        // sum += prod
    }

    // 水平求和
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(sum_low, sum_high);

    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);

    float result = _mm_cvtss_f32(sums);

    // 处理剩余元素
    for (; i < d; i++) {
        result += x[i] * y[i];
    }

    return result;
}

} // namespace faiss
```

### 15.3 批量距离计算

```cpp
// 批量计算查询与多个向量的距离

void fvec_L2sqr_ny_avx2(
        float* distances,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {

    // x: 查询向量 [d]
    // y: 数据库 [ny, d]
    // distances: 输出 [ny]

    for (size_t i = 0; i < ny; i++) {
        distances[i] = fvec_L2sqr_avx2(x, y + i * d, d);
    }
}

// 更高效的版本：展开循环
void fvec_L2sqr_ny_avx2_unrolled(
        float* distances,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {

    // 预加载查询向量到寄存器
    // 假设 d <= 8，可以全部放入一个 YMM 寄存器
    if (d <= 8) {
        __m256 vx = _mm256_loadu_ps(x);

        for (size_t i = 0; i + 4 <= ny; i += 4) {
            // 同时处理 4 个目标向量
            __m256 vy0 = _mm256_loadu_ps(y + (i + 0) * d);
            __m256 vy1 = _mm256_loadu_ps(y + (i + 1) * d);
            __m256 vy2 = _mm256_loadu_ps(y + (i + 2) * d);
            __m256 vy3 = _mm256_loadu_ps(y + (i + 3) * d);

            __m256 diff0 = _mm256_sub_ps(vx, vy0);
            __m256 diff1 = _mm256_sub_ps(vx, vy1);
            __m256 diff2 = _mm256_sub_ps(vx, vy2);
            __m256 diff3 = _mm256_sub_ps(vx, vy3);

            __m256 sq0 = _mm256_mul_ps(diff0, diff0);
            __m256 sq1 = _mm256_mul_ps(diff1, diff1);
            __m256 sq2 = _mm256_mul_ps(diff2, diff2);
            __m256 sq3 = _mm256_mul_ps(diff3, diff3);

            // 水平求和
            distances[i + 0] = horizontal_sum_ps(sq0);
            distances[i + 1] = horizontal_sum_ps(sq1);
            distances[i + 2] = horizontal_sum_ps(sq2);
            distances[i + 3] = horizontal_sum_ps(sq3);
        }

        // 处理剩余
        for (size_t i = (ny / 4) * 4; i < ny; i++) {
            distances[i] = fvec_L2sqr_avx2(x, y + i * d, d);
        }
    } else {
        // d > 8 的情况
        for (size_t i = 0; i < ny; i++) {
            distances[i] = fvec_L2sqr_avx2(x, y + i * d, d);
        }
    }
}
```

### 15.4 ARM NEON 实现

```cpp
// ARM 平台的 NEON 实现

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>

namespace faiss {

inline float fvec_L2sqr_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        float32x4_t diff = vsubq_f32(vx, vy);
        float32x4_t sq = vmulq_f32(diff, diff);
        sum = vaddq_f32(sum, sq);
    }

    // 水平求和
    float32x2_t sum_high = vget_high_f32(vreinterpretq_f32_f64(sum));
    float32x2_t sum_low = vget_low_f32(vreinterpretq_f32_f64(sum));
    float32x2_t sum2 = vadd_f32(sum_low, sum_high);

    float result = vaddvq_f32(sum);

    // 处理剩余元素
    for (; i < d; i++) {
        float diff = x[i] - y[i];
        result += diff * diff;
    }

    return result;
}

inline float fvec_inner_product_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        float32x4_t prod = vmulq_f32(vx, vy);
        sum = vaddq_f32(sum, prod);
    }

    float result = vaddvq_f32(sum);

    for (; i < d; i++) {
        result += x[i] * y[i];
    }

    return result;
}

} // namespace faiss

#endif // ARM_NEON
```

---

## 第十六部分：性能最佳实践

### 16.1 索引选择指南

**决策树**：

```
开始
  │
  ├─ 数据规模 < 10万？
  │    └─ 是 → IndexFlat（精确搜索）
  │
  ├─ 需要实时添加？
  │    └─ 是 → IndexHNSW 或 IndexIVF
  │
  ├─ 内存受限？
  │    └─ 是 → IndexIVFPQ 或 IndexScalarQuantizer
  │
  ├─ 需要高精度？
  │    └─ 是 → IndexIVFFlat + IndexRefine
  │
  └─ 其他 → IndexIVFFlat（默认选择）
```

### 16.2 参数调优策略

#### IVF 参数调优

```cpp
// IVF 参数优化框架

struct IVFParameters {
    size_t nlist;      // 聚类数
    size_t nprobe;     // 搜索聚类数

    static IVFParameters optimize(
            const float* train_data,
            size_t n_train,
            size_t d,
            const float* queries,
            size_t n_queries,
            const float* ground_truth) {

        IVFParameters best_params;
        float best_score = 0;

        // 网格搜索 nlist
        std::vector<size_t> nlist_values = {100, 500, 1000, 5000, 10000};

        for (size_t nlist : nlist_values) {
            // 训练索引
            IndexIVFFlat index(d, nlist, METRIC_L2);
            index.train(n_train, train_data);
            index.add(n_train, train_data);

            // 网格搜索 nprobe
            for (size_t nprobe = 1; nprobe <= std::min(nlist, (size_t)100); nprobe += 10) {
                index.nprobe = nprobe;

                // 测试性能
                auto score = evaluate_index(&index, queries, n_queries, ground_truth);

                if (score > best_score) {
                    best_score = score;
                    best_params.nlist = nlist;
                    best_params.nprobe = nprobe;
                }
            }
        }

        return best_params;
    }

    static float evaluate_index(
            Index* index,
            const float* queries,
            size_t n_queries,
            const float* ground_truth) {

        // 计算 recall@100
        const size_t k = 100;
        std::vector<float> distances(n_queries * k);
        std::vector<idx_t> labels(n_queries * k);

        index->search(n_queries, queries, k, distances.data(), labels.data());

        // 计算 recall
        size_t correct = 0;
        for (size_t i = 0; i < n_queries; i++) {
            for (size_t j = 0; j < k; j++) {
                if (labels[i * k + j] == ground_truth[i]) {
                    correct++;
                    break;
                }
            }
        }

        return (float)correct / n_queries;
    }
};
```

#### HNSW 参数调优

```cpp
// HNSW 参数优化

struct HNSWParameters {
    int M;                   // 连接数
    int ef_construction;     // 构建时的搜索宽度
    int ef_search;           // 搜索时的搜索宽度

    static HNSWParameters optimize(
            const float* train_data,
            size_t n_train,
            size_t d,
            const float* queries,
            size_t n_queries) {

        HNSWParameters best_params;
        float best_score = 0;

        // 测试不同的 M 值
        std::vector<int> M_values = {16, 32, 64, 128};

        for (int M : M_values) {
            IndexHNSW index(d, METRIC_L2);
            index.hnsw.M = M;
            index.hnsw.ef_construction = M * 2;

            // 添加向量（HNSW 不需要训练）
            index.add(n_train, train_data);

            // 测试不同的 ef_search
            for (int ef_search = M; ef_search <= M * 4; ef_search += 10) {
                index.hnsw.ef_search = ef_search;

                // 测试查询延迟
                auto start = std::chrono::high_resolution_clock::now();

                const size_t k = 10;
                std::vector<float> distances(n_queries * k);
                std::vector<idx_t> labels(n_queries * k);

                index.search(n_queries, queries, k, distances.data(), labels.data());

                auto end = std::chrono::high_resolution_clock::now();
                double latency = std::chrono::duration<double, std::milli>(end - start).count();

                // 计算得分（权衡延迟和精度）
                float recall = compute_recall(labels.data(), n_queries, k);
                float score = recall / (1 + latency / 1000.0f);  // 归一化

                if (score > best_score) {
                    best_score = score;
                    best_params.M = M;
                    best_params.ef_construction = M * 2;
                    best_params.ef_search = ef_search;
                }
            }
        }

        return best_params;
    }
};
```

### 16.3 内存优化技巧

```cpp
// 内存优化策略

struct MemoryOptimizer {

    // 1. 使用更紧凑的数据类型
    static void optimize_data_types(Index*& index) {
        // 原始: float (4 bytes)
        // 优化: int8 (1 byte) 或 float16 (2 bytes)

        if (auto* ivf_flat = dynamic_cast<IndexIVFFlat*>(index)) {
            // 转换为 IVFPQ
            auto* ivf_pq = new IndexIVFPQ(ivf_flat->d, ivf_flat->nlist,
                                         ivf_flat->metric_type);

            // 配置 PQ 参数
            ivf_pq->pq.nbits = 8;        // 8 bits per sub-quantizer
            ivf_pq->pq.M = ivf_flat->d / ivf_pq->pq.dsub;

            // 训练并转换
            ivf_pq->train(ivf_flat->ntotal, /* training data */);
            ivf_pq->add(ivf_flat->ntotal, /* data */);

            delete index;
            index = ivf_pq;
        }
    }

    // 2. 使用磁盘倒排表
    static void use_disk_inverted_lists(IndexIVF* ivf_index) {
        size_t nlist = ivf_index->nlist;
        size_t code_size = ivf_index->code_size;

        // 创建磁盘倒排表
        auto* disk_invlists = new OnDiskInvertedLists(
            "/path/to/storage",  // 存储路径
            nlist,
            code_size,
            /* memory_limit */ 1024 * 1024 * 1024  // 1GB 内存缓存
        );

        // 替换现有的倒排表
        delete ivf_index->invlists;
        ivf_index->invlists = disk_invlists;
        ivf_index->own_invlists = true;
    }

    // 3. 分片索引
    static Index* shard_index(Index* index, size_t num_shards) {
        auto* shards = new IndexShards(index->d, true);

        for (size_t i = 0; i < num_shards; i++) {
            // 创建分片索引
            Index* shard = /* 复制索引配置 */;
            shards->add_shard(shard);
        }

        return shards;
    }
};
```

### 16.4 多线程优化

```cpp
// 多线程搜索优化

struct MultiThreadOptimizer {

    // 1. 批量查询优化
    static void batch_search_optimized(
            Index* index,
            const float* queries,
            size_t n_queries,
            size_t k,
            float* distances,
            idx_t* labels) {

        // 设置 OpenMP 线程数
        omp_set_num_threads(omp_get_max_threads());

        // 批量搜索
        index->search(n_queries, queries, k, distances, labels);
    }

    // 2. 并行构建索引
    static void parallel_build(
            IndexIVFFlat* index,
            const float* vectors,
            size_t n) {

        // 训练阶段（并行）
        #pragma omp parallel
        {
            // Faiss 内部会自动并行化训练过程
        }

        index->train(/* 训练数据 */, /* 训练大小 */);

        // 添加阶段（并行）
        size_t batch_size = 100000;
        for (size_t i = 0; i < n; i += batch_size) {
            size_t end = std::min(i + batch_size, n);
            index->add(end - i, vectors + i * index->d);
        }
    }
};
```

### 16.5 NUMA 优化

```cpp
// NUMA (Non-Uniform Memory Access) 优化

#if defined(__linux__) && defined(NUMA_AWARE)

#include <numa.h>

struct NUMAOptimizer {

    static void bind_to_numa_node(int node) {
        // 绑定当前线程到指定的 NUMA 节点
        numa_run_on_node(node);
        numa_set_preferred(node);

        // 在该节点分配内存
        numa_set_localalloc();
    }

    static Index* create_numa_aware_index(
            size_t d,
            const float* vectors,
            size_t n,
            int num_numa_nodes) {

        auto* shards = new IndexShards(d, true);

        // 每个 NUMA 节点一个分片
        for (int node = 0; node < num_numa_nodes; node++) {
            // 创建线程并绑定到 NUMA 节点
            std::thread thread([node, shards, d, vectors, n, num_numa_nodes]() {
                bind_to_numa_node(node);

                // 创建该节点的索引分片
                auto* shard = new IndexFlat(d);

                // 添加该节点的数据
                size_t shard_size = n / num_numa_nodes;
                size_t offset = node * shard_size;

                shard->add(shard_size, vectors + offset * d);

                shards->add_shard(shard);
            });

            thread.join();
        }

        return shards;
    }
};

#endif // NUMA_AWARE
```

### 16.6 缓存优化

```cpp
// 缓存友好优化

struct CacheOptimizer {

    // 1. 预取优化
    static void prefetch_data(
            const float* data,
            size_t size,
            size_t prefetch_distance = 64) {

        // 预取下一块数据到缓存
        for (size_t i = 0; i < size; i += prefetch_distance / sizeof(float)) {
            _mm_prefetch((const char*)(data + i + prefetch_distance / sizeof(float)),
                         _MM_HINT_T0);  // 预取到 L1 缓存
        }
    }

    // 2. 数据重排以提高缓存命中率
    static void reorder_data_cache_friendly(
            const float* data,
            size_t n,
            size_t d,
            float* reordered) {

        // 按照缓存行大小重排数据
        constexpr size_t cache_line = 64;  // bytes
        constexpr size_t floats_per_line = cache_line / sizeof(float);

        // 块大小（应该是缓存行大小的倍数）
        constexpr size_t block_size = floats_per_line;

        // 重排：block-major order
        for (size_t i = 0; i < n; i += block_size) {
            for (size_t j = 0; j < d; j++) {
                for (size_t k = 0; k < block_size && i + k < n; k++) {
                    reordered[(j * n + i + k)] = data[(i + k) * d + j];
                }
            }
        }
    }
};
```

### 16.7 性能监控

```cpp
// 性能监控工具

struct PerformanceMonitor {

    struct Metrics {
        double avg_latency_ms;
        double p95_latency_ms;
        double p99_latency_ms;
        double qps;
        size_t total_queries;
    };

    static Metrics measure_search_performance(
            Index* index,
            const float* queries,
            size_t n_queries,
            size_t k,
            size_t warmup_rounds = 10) {

        std::vector<double> latencies;
        std::vector<float> distances(n_queries * k);
        std::vector<idx_t> labels(n_queries * k);

        // Warmup
        for (size_t i = 0; i < warmup_rounds; i++) {
            index->search(n_queries, queries, k, distances.data(), labels.data());
        }

        // 实际测量
        for (size_t i = 0; i < n_queries; i++) {
            auto start = std::chrono::high_resolution_clock::now();

            index->search(1, queries + i * index->d, k,
                         distances.data(), labels.data());

            auto end = std::chrono::high_resolution_clock::now();

            double latency = std::chrono::duration<double, std::milli>(end - start).count();
            latencies.push_back(latency);
        }

        // 计算统计数据
        Metrics metrics;
        metrics.total_queries = n_queries;

        std::sort(latencies.begin(), latencies.end());

        metrics.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / n_queries;
        metrics.p95_latency_ms = latencies[size_t(n_queries * 0.95)];
        metrics.p99_latency_ms = latencies[size_t(n_queries * 0.99)];
        metrics.qps = 1000.0 / metrics.avg_latency_ms;

        return metrics;
    }

    static void print_metrics(const Metrics& metrics) {
        printf("Performance Metrics:\n");
        printf("  Total queries: %zu\n", metrics.total_queries);
        printf("  Avg latency: %.3f ms\n", metrics.avg_latency_ms);
        printf("  P95 latency: %.3f ms\n", metrics.p95_latency_ms);
        printf("  P99 latency: %.3f ms\n", metrics.p99_latency_ms);
        printf("  QPS: %.2f\n", metrics.qps);
    }
};
```

---

## 第十七部分：实战案例详解

### 17.1 图像相似度搜索引擎

```cpp
// 完整的图像相似度搜索引擎

class ImageSimilaritySearchEngine {
public:
    struct Config {
        size_t feature_dim = 512;        // ResNet-50 特征维度
        size_t nlist = 1000;             // IVF 聚类数
        size_t nprobe = 100;             // 搜索聚类数
        size_t M = 32;                   // PQ 子量化器数
        size_t nbits = 8;                // PQ 位数
        bool use_gpu = true;             // 使用 GPU
        int gpu_device = 0;              // GPU 设备 ID
    };

    ImageSimilaritySearchEngine(const Config& cfg) : config(cfg) {
        initialize_index();
    }

    void add_image(size_t image_id, const std::vector<float>& feature) {
        FAISS_THROW_IF_NOT_MSG(feature.size() == config.feature_dim,
                               "Feature dimension mismatch");

        index->add(1, feature.data());
        id_map.push_back(image_id);
    }

    std::vector<std::pair<size_t, float>> search(
            const std::vector<float>& query_feature,
            size_t k = 10) {

        std::vector<float> distances(k);
        std::vector<idx_t> labels(k);

        index->search(1, query_feature.data(), k,
                     distances.data(), labels.data());

        // 转换结果
        std::vector<std::pair<size_t, float>> results;
        for (size_t i = 0; i < k; i++) {
            if (labels[i] >= 0 && labels[i] < (idx_t)id_map.size()) {
                results.emplace_back(id_map[labels[i]], distances[i]);
            }
        }

        return results;
    }

    void save(const std::string& path) {
        index->write(path.c_str());

        // 保存 ID 映射
        std::string id_map_path = path + ".idmap";
        FILE* f = fopen(id_map_path.c_str(), "wb");
        fwrite(id_map.data(), sizeof(size_t), id_map.size(), f);
        fclose(f);
    }

    void load(const std::string& path) {
        index.reset(read_index(path.c_str()));

        // 加载 ID 映射
        std::string id_map_path = path + ".idmap";
        FILE* f = fopen(id_map_path.c_str(), "rb");
        fseek(f, 0, SEEK_END);
        size_t size = ftell(f) / sizeof(size_t);
        fseek(f, 0, SEEK_SET);

        id_map.resize(size);
        fread(id_map.data(), sizeof(size_t), size, f);
        fclose(f);
    }

private:
    Config config;
    std::unique_ptr<Index> index;
    std::vector<size_t> id_map;  // 内部 ID -> 外部 image_id 映射

    void initialize_index() {
        if (config.use_gpu) {
            #if defined(FAISS_ENABLE_GPU)
            // GPU 实现
            auto res = std::make_shared<gpu::StandardGpuResources>();
            gpu::GpuIndexIVFPQConfig gpu_cfg;
            gpu_cfg.device = config.gpu_device;

            index.reset(new gpu::GpuIndexIVFPQ(
                res, config.gpu_device,
                config.feature_dim,
                config.nlist,
                config.M,
                config.nbits
            ));
            #else
            // 回退到 CPU
            initialize_cpu_index();
            #endif
        } else {
            initialize_cpu_index();
        }
    }

    void initialize_cpu_index() {
        auto* ivf_pq = new IndexIVFPQ(
            config.feature_dim,
            config.nlist,
            config.M,
            config.nbits
        );

        ivf_pq->nprobe = config.nprobe;
        index.reset(ivf_pq);
    }
};

// 使用示例
int main() {
    // 配置
    ImageSimilaritySearchEngine::Config config;
    config.feature_dim = 512;
    config.nlist = 1000;
    config.nprobe = 100;
    config.use_gpu = true;

    // 创建引擎
    ImageSimilaritySearchEngine engine(config);

    // 训练索引（需要先有训练数据）
    // engine.train(training_features);

    // 添加图像
    for (size_t i = 0; i < 1000000; i++) {
        std::vector<float> feature = extract_feature_from_image(i);
        engine.add_image(i, feature);
    }

    // 搜索
    std::vector<float> query = extract_feature_from_query();
    auto results = engine.search(query, 10);

    for (auto [image_id, distance] : results) {
        printf("Image %zu: distance = %.4f\n", image_id, distance);
    }

    // 保存索引
    engine.save("image_index.faiss");

    return 0;
}
```

### 17.2 推荐系统中的向量检索

```cpp
// 推荐系统的向量检索引擎

class RecommendationSearchEngine {
public:
    struct Config {
        size_t user_dim = 128;           // 用户嵌入维度
        size_t item_dim = 128;           // 物品嵌入维度
        size_t nlist = 500;              // 聚类数
        size_t nprobe = 50;              // 搜索聚类数
        bool use_hnsw = true;            // 使用 HNSW
    };

    RecommendationSearchEngine(const Config& cfg) : config(cfg) {
        // 用户索引用快速检索
        if (config.use_hnsw) {
            user_index = std::make_unique<IndexHNSW>(config.user_dim, METRIC_INNER_PRODUCT);
            static_cast<IndexHNSW*>(user_index.get())->hnsw.ef_search = 64;
        } else {
            user_index = std::make_unique<IndexIVFFlat>(config.user_dim, config.nlist,
                                                         METRIC_INNER_PRODUCT);
            static_cast<IndexIVFFlat*>(user_index.get())->nprobe = config.nprobe;
        }

        // 物品索引用批量检索
        item_index = std::make_unique<IndexIVFFlat>(config.item_dim, config.nlist,
                                                     METRIC_INNER_PRODUCT);
        static_cast<IndexIVFFlat*>(item_index.get())->nprobe = config.nprobe;
    }

    // 添加用户嵌入
    void add_user(size_t user_id, const std::vector<float>& embedding) {
        user_index->add(1, embedding.data());
        user_id_map.push_back(user_id);
    }

    // 添加物品嵌入
    void add_item(size_t item_id, const std::vector<float>& embedding) {
        item_index->add(1, embedding.data());
        item_id_map.push_back(item_id);
    }

    // 为用户推荐物品
    std::vector<size_t> recommend_for_user(
            size_t user_id,
            const std::vector<float>& user_embedding,
            size_t k = 100) {

        std::vector<float> distances(k);
        std::vector<idx_t> labels(k);

        // 搜索与用户最相似的物品
        item_index->search(1, user_embedding.data(), k,
                          distances.data(), labels.data());

        // 转换为物品 ID
        std::vector<size_t> recommended_items;
        for (size_t i = 0; i < k; i++) {
            if (labels[i] >= 0 && labels[i] < (idx_t)item_id_map.size()) {
                recommended_items.push_back(item_id_map[labels[i]]);
            }
        }

        return recommended_items;
    }

    // 找相似用户
    std::vector<size_t> find_similar_users(
            const std::vector<float>& user_embedding,
            size_t k = 50) {

        std::vector<float> distances(k);
        std::vector<idx_t> labels(k);

        user_index->search(1, user_embedding.data(), k,
                          distances.data(), labels.data());

        std::vector<size_t> similar_users;
        for (size_t i = 0; i < k; i++) {
            if (labels[i] >= 0 && labels[i] < (idx_t)user_id_map.size()) {
                similar_users.push_back(user_id_map[labels[i]]);
            }
        }

        return similar_users;
    }

private:
    Config config;
    std::unique_ptr<Index> user_index;
    std::unique_ptr<Index> item_index;
    std::vector<size_t> user_id_map;
    std::vector<size_t> item_id_map;
};
```

### 17.3 实时更新索引

```cpp
// 支持实时更新的索引

class DynamicVectorIndex {
public:
    DynamicVectorIndex(size_t d, size_t nlist = 1000)
        : dimension(d), nlist(nlist) {

        // 使用 IVF + HNSW 支持动态添加
        quantizer = std::make_unique<IndexFlatL2>(d);
        invlists = new ArrayInvertedLists(nlist, code_size);

        index = std::make_unique<IndexIVFFlat>(d, nlist, METRIC_L2);
        index->quantizer = quantizer.get();
        index->invlists = invlists;
        index->own_invlists = false;

        // 训练
        train_index();
    }

    void add_vector(size_t id, const float* vector) {
        // 直接添加（IVF 支持增量添加）
        index->add(1, vector);

        // 维护 ID 映射
        id_map[index->ntotal - 1] = id;
    }

    void remove_vector(size_t id) {
        // 找到内部 ID
        idx_t internal_id = find_internal_id(id);
        if (internal_id < 0) return;

        // 标记为已删除
        removed_ids.insert(internal_id);
    }

    std::vector<std::pair<size_t, float>> search(
            const float* query,
            size_t k = 10) {

        std::vector<float> distances(k);
        std::vector<idx_t> labels(k);

        index->search(1, query, k, distances.data(), labels.data());

        // 过滤已删除的 ID
        std::vector<std::pair<size_t, float>> results;
        for (size_t i = 0; i < k; i++) {
            if (labels[i] >= 0 && removed_ids.find(labels[i]) == removed_ids.end()) {
                auto it = id_map.find(labels[i]);
                if (it != id_map.end()) {
                    results.emplace_back(it->second, distances[i]);
                }
            }
        }

        return results;
    }

private:
    size_t dimension;
    size_t nlist;
    size_t code_size = sizeof(float) * 128;  // 假设 128 维

    std::unique_ptr<IndexFlatL2> quantizer;
    InvertedLists* invlists;
    std::unique_ptr<IndexIVFFlat> index;

    std::map<idx_t, size_t> id_map;      // 内部 ID -> 外部 ID
    std::set<idx_t> removed_ids;         // 已删除的 ID

    void train_index() {
        // 生成训练数据（实际应用中应该使用真实数据）
        size_t n_train = nlist * 256;
        std::vector<float> train_data(n_train * dimension);
        // ... 填充训练数据 ...

        index->train(n_train, train_data.data());
    }

    idx_t find_internal_id(size_t external_id) {
        for (const auto& [internal, external] : id_map) {
            if (external == external_id) {
                return internal;
            }
        }
        return -1;
    }
};
```

---

## 第十八部分：其他距离度量详解

### 18.1 L1 距离（曼哈顿距离）

**L1 距离**是向量各维度绝对差之和。

```cpp
// 位置: faiss/utils/distances.cpp

namespace faiss {

// L1 距离计算
float fvec_L1(const float* x, const float* y, size_t d) {
    float dist = 0;
    for (size_t i = 0; i < d; i++) {
        dist += fabsf(x[i] - y[i]);
    }
    return dist;
}

// SIMD 优化的 L1 距离 (AVX2)
float fvec_L1_avx2(const float* x, const float* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 diff = _mm256_sub_ps(vx, vy);

        // 绝对值：清除符号位
        __m256 abs_diff = _mm256_andnot_ps(
            _mm256_set1_ps(-0.0f),  // 符号掩码
            diff
        );

        sum = _mm256_add_ps(sum, abs_diff);
    }

    // 水平求和
    float result = horizontal_sum_ps(sum);

    // 处理剩余元素
    for (; i < d; i++) {
        result += fabsf(x[i] - y[i]);
    }

    return result;
}

// L1 距离的 SIMD 实现（使用位操作优化绝对值）
inline __m256 abs_ps(__m256 x) {
    // 清除符号位（第31位）
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
}

} // namespace faiss
```

### 18.2 Canberra 距离

**Canberra 距离**是加权 L1 距离，对尺度差异敏感。

```
公式：D(x, y) = Σ |xi - yi| / (|xi| + |yi|)
```

```cpp
// Canberra 距离实现

float fvec_canberra(const float* x, const float* y, size_t d) {
    float dist = 0;
    for (size_t i = 0; i < d; i++) {
        float abs_x = fabsf(x[i]);
        float abs_y = fabsf(y[i]);
        float abs_diff = fabsf(x[i] - y[i]);

        float denom = abs_x + abs_y;
        if (denom > 0) {
            dist += abs_diff / denom;
        }
    }
    return dist;
}
```

### 18.3 Bray-Curtis 距离

**Bray-Curtis 距离**常用于生态学数据分析。

```
公式：D(x, y) = Σ |xi - yi| / Σ (xi + yi)
```

```cpp
// Bray-Curtis 距离实现

float fvec_braycurtis(const float* x, const float* y, size_t d) {
    float num = 0;  // 分子
    float den = 0;  // 分母

    for (size_t i = 0; i < d; i++) {
        num += fabsf(x[i] - y[i]);
        den += x[i] + y[i];
    }

    return (den > 0) ? (num / den) : 0;
}
```

### 18.4 Jaccard 距离

**Jaccard 距离**用于集合相似度，也可用于稀疏向量。

```
公式：D(x, y) = 1 - (min(xi, yi)之和 / max(xi, yi)之和)
对于二进制向量：D(x, y) = 1 - (交集大小 / 并集大小)
```

```cpp
// Jaccard 距离实现（稀疏向量）

struct SparseVector {
    size_t size;
    const idx_t* indices;    // 非零元素索引
    const float* values;     // 非零元素值
};

float sparse_jaccard_distance(
        const SparseVector& x,
        const SparseVector& y) {

    float intersection = 0;
    float union_val = 0;

    size_t i = 0, j = 0;
    while (i < x.size && j < y.size) {
        if (x.indices[i] == y.indices[j]) {
            // 相同索引
            intersection += fmin(x.values[i], y.values[j]);
            union_val += fmax(x.values[i], y.values[j]);
            i++;
            j++;
        } else if (x.indices[i] < y.indices[j]) {
            union_val += x.values[i];
            i++;
        } else {
            union_val += y.values[j];
            j++;
        }
    }

    // 处理剩余元素
    for (; i < x.size; i++) union_val += x.values[i];
    for (; j < y.size; j++) union_val += y.values[j];

    return (union_val > 0) ? (1.0f - intersection / union_val) : 0;
}
```

### 18.5 余弦相似度

**余弦相似度**衡量向量方向差异，常用于文本检索。

```
公式：cos(x, y) = (x · y) / (||x|| * ||y||)
余弦距离：1 - cos(x, y)
```

```cpp
// 余弦相似度计算

float fvec_cosine_similarity(const float* x, const float* y, size_t d) {
    // 计算内积
    float ip = 0;
    float norm_x = 0;
    float norm_y = 0;

    for (size_t i = 0; i < d; i++) {
        ip += x[i] * y[i];
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }

    float denom = sqrtf(norm_x) * sqrtf(norm_y);
    return (denom > 0) ? (ip / denom) : 0;
}

// 预计算范数的余弦相似度（更快）
float fvec_cosine_similarity_with_norms(
        const float* x,
        const float* y,
        size_t d,
        float norm_x,
        float norm_y) {

    float ip = fvec_inner_product(x, y, d);
    float denom = norm_x * norm_y;
    return (denom > 0) ? (ip / denom) : 0;
}

// SIMD 优化的内积+范数计算
void fvec_inner_product_and_norms(
        const float* x,
        const float* y,
        size_t d,
        float* ip,
        float* norm_x,
        float* norm_y) {

    __m256 sum_ip = _mm256_setzero_ps();
    __m256 sum_x2 = _mm256_setzero_ps();
    __m256 sum_y2 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);

        __m256 prod = _mm256_mul_ps(vx, vy);
        sum_ip = _mm256_add_ps(sum_ip, prod);

        __m256 x2 = _mm256_mul_ps(vx, vx);
        sum_x2 = _mm256_add_ps(sum_x2, x2);

        __m256 y2 = _mm256_mul_ps(vy, vy);
        sum_y2 = _mm256_add_ps(sum_y2, y2);
    }

    *ip = horizontal_sum_ps(sum_ip);
    *norm_x = horizontal_sum_ps(sum_x2);
    *norm_y = horizontal_sum_ps(sum_y2);

    // 处理剩余元素
    for (; i < d; i++) {
        *ip += x[i] * y[i];
        *norm_x += x[i] * x[i];
        *norm_y += y[i] * y[i];
    }
}
```

---

## 第十九部分：IDSelector 与过滤搜索

### 19.1 IDSelector 基础

**IDSelector** 允许在搜索时过滤掉特定的向量 ID。

```cpp
// 位置: faiss/IDSelector.h

namespace faiss {

struct IDSelector {
    size_t n;                  // 总向量数
    const idx_t* indices;      // 选择的 ID 列表

    IDSelector(size_t n, const idx_t* indices) : n(n), indices(indices) {}

    virtual ~IDSelector() = default;

    // 检查 ID 是否被选中
    virtual bool is_member(idx_t id) const {
        // 默认实现：线性搜索
        for (size_t i = 0; i < n; i++) {
            if (indices[i] == id) return true;
        }
        return false;
    }
};

// 位图实现（更快的查找）
struct IDSelectorBitmap : IDSelector {
    std::vector<uint8_t> bitmap;  // 位图：1表示选中

    IDSelectorBitmap(size_t n, const idx_t* indices)
        : IDSelector(n, indices) {

        size_t bitmap_size = (n + 7) / 8;
        bitmap.resize(bitmap_size, 0);

        for (size_t i = 0; i < n; i++) {
            idx_t id = indices[i];
            bitmap[id / 8] |= (1 << (id % 8));
        }
    }

    bool is_member(idx_t id) const override {
        if (id >= 0 && id < (idx_t)n * 8) {
            return (bitmap[id / 8] >> (id % 8)) & 1;
        }
        return false;
    }
};

// 取反选择器
struct IDSelectorNot : IDSelector {
    const IDSelector* selector;

    IDSelectorNot(const IDSelector* sel) : IDSelector(0, nullptr), selector(sel) {}

    bool is_member(idx_t id) const override {
        return !selector->is_member(id);
    }
};

} // namespace faiss
```

### 19.2 使用 IDSelector 搜索

```cpp
// 带过滤的搜索示例

void search_with_filter(
        Index* index,
        const float* query,
        size_t k,
        float* distances,
        idx_t* labels,
        const IDSelector* selector) {

    // 方法1：搜索后过滤
    size_t k_search = k * 10;  // 搜索更多候选
    std::vector<float> all_distances(k_search);
    std::vector<idx_t> all_labels(k_search);

    index->search(1, query, k_search,
                 all_distances.data(), all_labels.data());

    // 过滤结果
    size_t found = 0;
    for (size_t i = 0; i < k_search && found < k; i++) {
        if (!selector || selector->is_member(all_labels[i])) {
            distances[found] = all_distances[i];
            labels[found] = all_labels[i];
            found++;
        }
    }

    // 如果没有找到足够的有效结果
    for (size_t i = found; i < k; i++) {
        labels[i] = -1;
        distances[i] = std::numeric_limits<float>::infinity();
    }
}
```

### 19.3 批量 ID 优化

```cpp
// 批量 ID 选择的优化

struct BatchIDSelector {
    std::vector<idx_t> valid_ids;
    std::unordered_set<idx_t> valid_set;

    BatchIDSelector(const std::vector<idx_t>& ids)
        : valid_ids(ids), valid_set(ids.begin(), ids.end()) {}

    bool is_member(idx_t id) const {
        return valid_set.find(id) != valid_set.end();
    }

    // 获取批量查询的掩码
    std::vector<uint8_t> get_batch_mask(
            const idx_t* labels,
            size_t n) const {

        std::vector<uint8_t> mask(n);
        for (size_t i = 0; i < n; i++) {
            mask[i] = is_member(labels[i]) ? 1 : 0;
        }
        return mask;
    }
};
```

---

## 第二十部分：索引序列化与持久化

### 20.1 基本序列化

```cpp
// 索引的保存和加载

void save_index(Index* index, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    // 使用 Faiss 的 I/O
    IOWriter* writer = new FileIOWriter(f);
    index->serialize(writer);
    delete writer;

    fclose(f);
}

Index* load_index(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for reading\n", filename);
        return nullptr;
    }

    IOReader* reader = new FileIOReader(f);
    Index* index = nullptr;

    // 读取索引
    index = Index::deserialize(reader);

    delete reader;
    fclose(f);

    return index;
}
```

### 20.2 增量保存

```cpp
// 增量保存策略

class IncrementalIndexSaver {
public:
    std::string base_path;
    size_t save_interval = 10000;  // 每10000个向量保存一次

    void add_and_maybe_save(IndexIVF* index, const float* x, idx_t n) {
        idx_t old_ntotal = index->ntotal;
        index->add(n, x);

        // 检查是否需要保存
        if ((index->ntotal / save_interval) > (old_ntotal / save_interval)) {
            save_incremental(index);
        }
    }

private:
    void save_incremental(IndexIVF* index) {
        // 保存增量数据
        std::string filename = base_path + "_incremental_" +
                              std::to_string(index->ntotal);
        save_index(index, filename.c_str());
    }
};
```

### 20.3 内存映射文件

```cpp
// 使用内存映射加速加载

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

class MappedIndexLoader {
public:
    static Index* load_mmap(const char* filename) {
        int fd = open(filename, O_RDONLY);
        if (fd < 0) {
            perror("open");
            return nullptr;
        }

        // 获取文件大小
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            perror("fstat");
            close(fd);
            return nullptr;
        }

        size_t size = sb.st_size;

        // 内存映射
        void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            perror("mmap");
            close(fd);
            return nullptr;
        }

        // 从内存映射反序列化
        BufferIOReader reader((uint8_t*)mapped, size);
        Index* index = Index::deserialize(&reader);

        // 注意：不要立即 munmap，索引在使用期间需要保持映射
        // 可以将 mapped 和 fd 保存起来，在索引析构时释放

        return index;
    }
};
```

---

## 第二十一部分：分布式向量搜索

### 21.1 数据分片策略

```cpp
// 分布式向量搜索架构

class DistributedVectorSearch {
public:
    struct Shard {
        std::string address;
        std::unique_ptr<Index> index;
    };

    std::vector<Shard> shards;

    // 添加分片
    void add_shard(const std::string& address) {
        Shard shard;
        shard.address = address;
        // 连接到远程分片...
        shards.push_back(std::move(shard));
    }

    // 添加向量（路由到分片）
    void add_vector(size_t id, const float* vector, size_t d) {
        // 根据ID哈希选择分片
        size_t shard_id = hash_id(id) % shards.size();
        shards[shard_id].index->add(1, vector);
    }

    // 搜索（查询所有分片）
    std::vector<std::pair<size_t, float>> search(
            const float* query,
            size_t d,
            size_t k) {

        std::vector<std::pair<float, size_t>> all_results;

        // 并行查询所有分片
        #pragma omp parallel for
        for (size_t i = 0; i < shards.size(); i++) {
            std::vector<float> distances(k);
            std::vector<idx_t> labels(k);

            shards[i].index->search(1, query, k,
                                   distances.data(), labels.data());

            // 合并结果
            #pragma omp critical
            {
                for (size_t j = 0; j < k; j++) {
                    all_results.emplace_back(distances[j], labels[j]);
                }
            }
        }

        // 全局排序并返回 Top-K
        std::sort(all_results.begin(), all_results.end());

        std::vector<std::pair<size_t, float>> results;
        for (size_t i = 0; i < k && i < all_results.size(); i++) {
            results.emplace_back(all_results[i].second, all_results[i].first);
        }

        return results;
    }

private:
    size_t hash_id(size_t id) {
        // 简单的哈希函数
        return std::hash<size_t>{}(id);
    }
};
```

### 21.2 负载均衡

```cpp
// 分片负载均衡

class LoadBalancedShards {
public:
    struct ShardInfo {
        std::string address;
        size_t vector_count;
        double avg_query_time;
    };

    std::vector<ShardInfo> shards;

    // 动态选择分片
    std::vector<size_t> select_shards_for_search(size_t nshards) {
        // 根据负载选择最快的前 nshards 个分片
        std::vector<size_t> indices(shards.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(),
            [this](size_t a, size_t b) {
                return shards[a].avg_query_time < shards[b].avg_query_time;
            });

        return std::vector<size_t>(indices.begin(), indices.begin() + nshards);
    }

    // 监控分片性能
    void update_shard_stats(size_t shard_id, double query_time) {
        // 指数移动平均
        double alpha = 0.1;
        shards[shard_id].avg_query_time =
            alpha * query_time + (1 - alpha) * shards[shard_id].avg_query_time;
    }
};
```

### 21.3 一致性哈希

```cpp
// 一致性哈希用于数据分布

class ConsistentHashRing {
public:
    struct VirtualNode {
        size_t shard_id;
        size_t hash;
    };

    std::vector<VirtualNode> ring;
    size_t virtual_nodes_per_shard = 100;

    void add_shard(size_t shard_id) {
        // 为每个分片创建多个虚拟节点
        for (size_t i = 0; i < virtual_nodes_per_shard; i++) {
            std::string key = std::to_string(shard_id) + "-" + std::to_string(i);
            size_t hash = std::hash<std::string>{}(key);

            ring.push_back({shard_id, hash});
        }

        // 按哈希值排序
        std::sort(ring.begin(), ring.end(),
            [](const VirtualNode& a, const VirtualNode& b) {
                return a.hash < b.hash;
            });
    }

    size_t get_shard_for_id(size_t id) {
        size_t hash = std::hash<size_t>{}(id);

        // 二分查找
        auto it = std::lower_bound(ring.begin(), ring.end(), hash,
            [](const VirtualNode& node, size_t h) {
                return node.hash < h;
            });

        // 环绕
        if (it == ring.end()) {
            it = ring.begin();
        }

        return it->shard_id;
    }
};
```

---

## 第二十二部分：测试与验证

### 22.1 单元测试框架

```cpp
// 索引测试框架

class IndexTester {
public:
    // 测试索引的正确性
    static bool test_index_correctness(
            Index* index,
            const float* data,
            size_t n,
            size_t d,
            size_t k) {

        printf("Testing index correctness...\n");

        // 测试添加
        index->reset();
        index->add(n, data);

        if (index->ntotal != (idx_t)n) {
            fprintf(stderr, "Add failed: expected %zu, got %zu\n",
                   n, (size_t)index->ntotal);
            return false;
        }

        // 测试搜索
        std::vector<float> distances(k);
        std::vector<idx_t> labels(k);

        index->search(1, data, k, distances.data(), labels.data());

        // 第一个结果应该是自己（距离为0）
        if (labels[0] != 0) {
            fprintf(stderr, "Search failed: expected label 0, got %ld\n",
                   (long)labels[0]);
            return false;
        }

        if (distances[0] > 1e-6) {
            fprintf(stderr, "Distance too large: expected ~0, got %f\n",
                   distances[0]);
            return false;
        }

        printf("  ✓ Add test passed\n");
        printf("  ✓ Search test passed\n");
        return true;
    }

    // 计算 Recall
    static float compute_recall(
            const idx_t* result_labels,
            const idx_t* ground_truth,
            size_t n_queries,
            size_t k) {

        size_t correct = 0;
        for (size_t i = 0; i < n_queries; i++) {
            for (size_t j = 0; j < k; j++) {
                if (result_labels[i * k + j] == ground_truth[i * k + j]) {
                    correct++;
                    break;
                }
            }
        }

        return (float)correct / n_queries;
    }

    // 性能基准测试
    static void benchmark_search(
            Index* index,
            const float* queries,
            size_t n_queries,
            size_t k) {

        printf("Benchmarking search (%zu queries, k=%zu)...\n",
               n_queries, k);

        std::vector<float> distances(n_queries * k);
        std::vector<idx_t> labels(n_queries * k);

        // Warmup
        index->search(n_queries, queries, k, distances.data(), labels.data());

        // 实际测试
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < 10; iter++) {
            index->search(n_queries, queries, k,
                         distances.data(), labels.data());
        }

        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        double avg_time = elapsed / 10.0;

        printf("  Total time: %.3f s (10 iterations)\n", elapsed);
        printf("  Avg time per query: %.6f s\n", avg_time / n_queries);
        printf("  QPS: %.2f\n", n_queries / avg_time);
    }
};
```

### 22.2 压力测试

```cpp
// 并发压力测试

class ConcurrencyTest {
public:
    static void concurrent_search_test(
            Index* index,
            const float* queries,
            size_t n_queries,
            size_t k,
            int n_threads) {

        printf("Testing concurrent search with %d threads...\n", n_threads);

        std::vector<std::thread> threads;
        std::vector<std::vector<float>> distances(n_threads);
        std::vector<std::vector<idx_t>> labels(n_threads);

        auto start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < n_threads; t++) {
            distances[t].resize(n_queries * k);
            labels[t].resize(n_queries * k);

            threads.emplace_back([&, t]() {
                for (size_t i = 0; i < n_queries; i++) {
                    index->search(1, queries + i * index->d, k,
                                 distances[t].data() + i * k,
                                 labels[t].data() + i * k);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        size_t total_queries = n_queries * n_threads;
        printf("  Total queries: %zu\n", total_queries);
        printf("  Total time: %.3f s\n", elapsed);
        printf("  QPS: %.2f\n", total_queries / elapsed);
    }

    static void memory_leak_test(
            size_t d,
            size_t n_vectors,
            int iterations) {

        printf("Testing for memory leaks (%d iterations)...\n", iterations);

        size_t initial_memory = get_memory_usage();

        for (int iter = 0; iter < iterations; iter++) {
            Index* index = new IndexFlatL2(d);

            std::vector<float> vectors(n_vectors * d);
            // ... 填充随机数据 ...

            index->add(n_vectors, vectors.data());

            std::vector<float> query(d);
            // ... 填充查询 ...

            std::vector<float> distances(10);
            std::vector<idx_t> labels(10);

            index->search(1, query.data(), 10,
                         distances.data(), labels.data());

            delete index;
        }

        size_t final_memory = get_memory_usage();
        size_t leaked = final_memory > initial_memory ?
                       final_memory - initial_memory : 0;

        if (leaked > 1024 * 1024) {  // > 1MB
            printf("  ✗ Memory leak detected: %zu MB\n", leaked / (1024 * 1024));
        } else {
            printf("  ✓ No significant memory leak\n");
        }
    }

private:
    static size_t get_memory_usage() {
        // Linux specific
        std::ifstream file("/proc/self/status");
        std::string line;

        while (std::getline(file, line)) {
            if (line.find("VmRSS:") == 0) {
                size_t kb;
                sscanf(line.c_str(), "VmRSS: %zu", &kb);
                return kb * 1024;
            }
        }

        return 0;
    }
};
```

### 22.3 精度测试

```cpp
// 精度和召回率测试

class AccuracyTester {
public:
    // 与暴力搜索对比精度
    static float compare_with_bruteforce(
            Index* index,
            Index* ground_truth,
            const float* queries,
            size_t n_queries,
            size_t k) {

        printf("Comparing with brute force...\n");

        std::vector<float> distances(n_queries * k);
        std::vector<idx_t> labels(n_queries * k);

        std::vector<float> gt_distances(n_queries * k);
        std::vector<idx_t> gt_labels(n_queries * k);

        // 测试索引搜索
        index->search(n_queries, queries, k,
                     distances.data(), labels.data());

        // 暴力搜索
        ground_truth->search(n_queries, queries, k,
                            gt_distances.data(), gt_labels.data());

        // 计算召回率
        size_t correct = 0;
        for (size_t i = 0; i < n_queries; i++) {
            std::unordered_set<idx_t> gt_set(
                gt_labels + i * k,
                gt_labels + i * k + k
            );

            for (size_t j = 0; j < k; j++) {
                if (gt_set.count(labels[i * k + j]) > 0) {
                    correct++;
                    break;
                }
            }
        }

        float recall = (float)correct / n_queries;
        printf("  Recall@%zu: %.4f (%zu/%zu)\n",
               k, recall, correct, n_queries);

        return recall;
    }

    // 计算误差分布
    static void analyze_error_distribution(
            const float* approx_distances,
            const float* exact_distances,
            size_t n) {

        std::vector<float> errors(n);
        float max_error = 0;
        float sum_error = 0;

        for (size_t i = 0; i < n; i++) {
            float error = fabs(approx_distances[i] - exact_distances[i]);
            errors[i] = error;
            max_error = std::max(max_error, error);
            sum_error += error;
        }

        std::sort(errors.begin(), errors.end());

        printf("Error distribution:\n");
        printf("  Min:  %.6f\n", errors[0]);
        printf("  50%%: %.6f\n", errors[n / 2]);
        printf("  90%%: %.6f\n", errors[size_t(n * 0.9)]);
        printf("  99%%: %.6f\n", errors[size_t(n * 0.99)]);
        printf("  Max:  %.6f\n", max_error);
        printf("  Mean: %.6f\n", sum_error / n);
    }
};
```

---

## 第二十三部分：高级实战技巧

### 23.1 混合索引策略

```cpp
// 根据数据规模自动选择索引

class AdaptiveIndexSelector {
public:
    static Index* create_optimal_index(
            const float* train_data,
            size_t n_train,
            size_t d,
            size_t expected_queries_per_second) {

        printf("Selecting optimal index...\n");
        printf("  Data size: %zu vectors\n", n_train);
        printf("  Dimension: %zu\n", d);
        printf("  Expected QPS: %zu\n", expected_queries_per_second);

        // 小规模：使用 Flat
        if (n_train < 100000) {
            printf("  → Using IndexFlat (small data)\n");
            return new IndexFlatL2(d);
        }

        // 中规模：使用 IVFFlat
        if (n_train < 10000000) {
            printf("  → Using IndexIVFFlat (medium data)\n");
            size_t nlist = std::min((size_t)1000, n_train / 1000);
            auto* index = new IndexIVFFlat(d, nlist, METRIC_L2);
            index->nprobe = std::max((size_t)10, nlist / 10);
            return index;
        }

        // 大规模，高QPS：使用 HNSW
        if (expected_queries_per_second > 1000) {
            printf("  → Using IndexHNSW (high QPS requirement)\n");
            auto* index = new IndexHNSW(d, METRIC_L2);
            index->hnsw.M = 32;
            index->hnsw.ef_construction = 64;
            index->hnsw.ef_search = 32;
            return index;
        }

        // 大规模，内存受限：使用 IVFPQ
        printf("  → Using IndexIVFPQ (large data, memory constrained)\n");
        size_t nlist = 1000;
        size_t M = 32;
        size_t nbits = 8;

        auto* index = new IndexIVFPQ(d, nlist, M, nbits);
        index->nprobe = 100;
        return index;
    }
};
```

### 23.2 多阶段搜索优化

```cpp
// 多阶段搜索：粗筛选 + 精确排序

class TwoStageSearchEngine {
public:
    Index* coarse_index;     // 粗索引（快速）
    Index* fine_index;       // 精索引（精确）

    TwoStageSearchEngine(size_t d, size_t n) {
        // 粗索引：IVFPQ（快速）
        coarse_index = new IndexIVFPQ(d, 1000, 32, 8);
        coarse_index->nprobe = 50;

        // 精索引：IVFFlat（精确）
        fine_index = new IndexIVFFlat(d, 1000, METRIC_L2);
        fine_index->nprobe = 10;
    }

    void add_vectors(const float* vectors, size_t n) {
        coarse_index->add(n, vectors);
        fine_index->add(n, vectors);
    }

    std::vector<std::pair<size_t, float>> search(
            const float* query,
            size_t k_coarse,
            size_t k_fine) {

        // 第一阶段：粗筛选，返回更多候选
        std::vector<float> coarse_distances(k_coarse);
        std::vector<idx_t> coarse_labels(k_coarse);

        coarse_index->search(1, query, k_coarse,
                            coarse_distances.data(),
                            coarse_labels.data());

        // 第二阶段：从候选中精确排序
        // 重新计算精确距离
        std::vector<std::pair<float, size_t>> candidates;
        for (size_t i = 0; i < k_coarse; i++) {
            if (coarse_labels[i] < 0) continue;

            // 从精索引重构向量
            std::vector<float> vector(fine_index->d);
            fine_index->reconstruct(coarse_labels[i], vector.data());

            // 计算精确距离
            float dist = 0;
            for (size_t j = 0; j < fine_index->d; j++) {
                float diff = query[j] - vector[j];
                dist += diff * diff;
            }

            candidates.emplace_back(dist, coarse_labels[i]);
        }

        // 排序并返回 Top-K
        std::sort(candidates.begin(), candidates.end());

        std::vector<std::pair<size_t, float>> results;
        for (size_t i = 0; i < k_fine && i < candidates.size(); i++) {
            results.emplace_back(candidates[i].second, candidates[i].first);
        }

        return results;
    }
};
```

### 23.3 动态参数调优

```cpp
// 根据查询负载动态调整参数

class AdaptiveParameterTuner {
public:
    struct PerformanceMetrics {
        double avg_latency;
        double p95_latency;
        float recall;
    };

    static void tune_ivf_parameters(
            IndexIVFFlat* index,
            const float* validation_queries,
            size_t n_validation,
            const idx_t* ground_truth,
            size_t k) {

        printf("Tuning IVF parameters...\n");

        size_t nprobe_start = 10;
        size_t nprobe_end = std::min(index->nlist, (size_t)200);
        size_t nprobe_step = 10;

        float best_score = 0;
        size_t best_nprobe = nprobe_start;

        for (size_t nprobe = nprobe_start; nprobe <= nprobe_end; nprobe += nprobe_step) {
            index->nprobe = nprobe;

            // 测试性能
            auto metrics = measure_performance(
                index, validation_queries, n_validation, k
            );

            // 计算得分（平衡延迟和精度）
            float score = metrics.recall / (1 + metrics.avg_latency / 0.1);

            printf("  nprobe=%zu: recall=%.4f, latency=%.3fms, score=%.4f\n",
                   nprobe, metrics.recall, metrics.avg_latency, score);

            if (score > best_score) {
                best_score = score;
                best_nprobe = nprobe;
            }
        }

        printf("  → Optimal nprobe: %zu (score=%.4f)\n",
               best_nprobe, best_score);
        index->nprobe = best_nprobe;
    }

private:
    static PerformanceMetrics measure_performance(
            Index* index,
            const float* queries,
            size_t n,
            size_t k) {

        std::vector<float> distances(n * k);
        std::vector<idx_t> labels(n * k);

        // 测量延迟
        auto start = std::chrono::high_resolution_clock::now();
        index->search(n, queries, k, distances.data(), labels.data());
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        PerformanceMetrics metrics;
        metrics.avg_latency = elapsed / n;

        // 计算 P95
        std::vector<double> latencies;
        for (size_t i = 0; i < n; i++) {
            // ... 测量单个查询延迟 ...
        }
        std::sort(latencies.begin(), latencies.end());
        metrics.p95_latency = latencies[size_t(n * 0.95)];

        // 计算召回率（如果有ground truth）
        // metrics.recall = ...

        return metrics;
    }
};
```

### 23.4 故障恢复

```cpp
// 索引故障恢复机制

class ResilientIndex {
public:
    Index* primary_index;
    Index* replica_index;
    std::string backup_path;

    void add_vectors(const float* vectors, size_t n) {
        try {
            // 写入主索引
            primary_index->add(n, vectors);

            // 异步备份
            std::thread([this, vectors, n]() {
                try {
                    replica_index->add(n, vectors);
                    save_backup();
                } catch (...) {
                    fprintf(stderr, "Backup failed\n");
                }
            }).detach();

        } catch (...) {
            // 主索引失败，使用副本
            fprintf(stderr, "Primary index failed, using replica\n");
            replica_index->add(n, vectors);
            save_backup();
        }
    }

    void search_with_fallback(
            const float* query,
            size_t k,
            float* distances,
            idx_t* labels) {

        try {
            primary_index->search(1, query, k, distances, labels);
        } catch (...) {
            fprintf(stderr, "Primary search failed, trying replica\n");
            replica_index->search(1, query, k, distances, labels);
        }
    }

private:
    void save_backup() {
        std::string filename = backup_path + "_" +
                              std::to_string(std::time(nullptr));
        save_index(replica_index, filename.c_str());
    }
};
```

---

## 第二十四部分：Index Factory 与索引创建

### 24.1 Index Factory 概述

**Index Factory** 是 Faiss 提供的强大功能，可以通过字符串描述快速创建复杂的索引组合。

```cpp
// 位置: faiss/index_factory.cpp

namespace faiss {

// 索引工厂函数
Index* index_factory(size_t d, const char* index_desc, const float* trained) {
    Index* index = nullptr;

    // 解析索引描述字符串
    std::string desc = index_desc;

    // 解析ID
    size_t id;
    char* id_ptr;
    id = strtol(desc.c_str(), &id_ptr, 10);
    desc = id_ptr;

    // 跳过空格
    while (desc[0] == ' ') desc++;

    // 根据索引类型创建
    if (desc.find("Flat") == 0) {
        index = new IndexFlatL2(d);
    }
    else if (desc.find("IVF") == 0) {
        // 解析 IVF 参数
        size_t nlist = 100;
        sscanf(desc.c_str(), "IVF%zu", &nlist);

        Index* quantizer = new IndexFlatL2(d);
        index = new IndexIVFFlat(d, nlist, METRIC_L2);
        ((IndexIVFFlat*)index)->quantizer = quantizer;
        ((IndexIVFFlat*)index)->own_fields = true;
    }
    else if (desc.find("IVFPQ") == 0) {
        size_t nlist, M, nbits;
        sscanf(desc.c_str(), "IVFPQ%zu,%zu,%zu", &nlist, &M, &nbits);

        Index* quantizer = new IndexFlatL2(d);
        index = new IndexIVFPQ(d, nlist, M, nbits);
        ((IndexIVFPQ*)index)->quantizer = quantizer;
        ((IndexIVFPQ*)index)->own_fields = true;
    }
    else if (desc.find("HNSW") == 0) {
        index = new IndexHNSW(d);
        ((IndexHNSW*)index)->hnsw.M = 32;
    }
    // ... 更多索引类型

    // 如果需要训练
    if (trained && index->is_trained) {
        index->train(..., trained);
    }

    return index;
}

} // namespace faiss
```

### 24.2 常用索引描述字符串

```cpp
// 索引工厂字符串示例

// 基础索引
"Flat"                          → IndexFlatL2
"FlatIP"                        → IndexFlatIP (内积)

// IVF 索引
"IVF4096"                       → IVF with 4096 clusters
"IVF4096,Flat"                  → IVFFlat with 4096 clusters
"IVF2048,PQ32"                   → IVFPQ with 2048 clusters, PQ32

// PQ 索引
"PQ32"                          → Product Quantizer, M=32
"PQ64x8"                        → PQ64 with 8 bits

// HNSW 索引
"HNSW"                          → Hierarchical Navigable Small World
"HNSW32,M=48"                     → HNSW with M=48 connections

// 组合索引
"PCAR8,IVF4096,PQ8"            → PCA(8) + IVF(4096) + PQ(8)
"OPQ16_64,IVF4096,PQ8"           → OPQ16_64 + IVF4096 + PQ8

// 预处理 + 索引
"L2norm,IVF4096,PQ8"             → 先 L2 归一化，然后 IVFPQ8
"Normalize,IVF4096,Flat"          → 先向量归一化，然后 IVFFlat

// GPU 索引
"IVF4096(PQ32x8)"                → IVF4096 with GPU PQ32x8
```

### 24.3 IndexFactory 实战示例

```cpp
// 使用索引工厂创建索引

void demo_index_factory() {
    size_t d = 128;

    // 示例1：快速创建 IVFPQ 索引
    Index* index1 = index_factory(d, "IVF4096,PQ32");
    printf("Created: IVF4096,PQ32\n");

    // 示例2：创建带预处理的索引
    Index* index2 = index_factory(d, "L2norm,IVF2048,PQ16");
    printf("Created: L2norm,IVF2048,PQ16\n");

    // 示例3：创建 HNSW 索引
    Index* index3 = index_factory(d, "HNSW");
    printf("Created: HNSW\n");

    // 示例4：创建复杂组合索引
    Index* index4 = index_factory(d, "OPQ64x64,IVF8192,PQ8");
    printf("Created: OPQ64x64,IVF8192,PQ8\n");

    // 训练并使用索引
    std::vector<float> train_data(1000000 * d);
    // ... 填充训练数据 ...

    index1->train(1000000, train_data.data());
    index1->add(1000000, train_data.data());

    // 搜索
    std::vector<float> query(d);
    std::vector<float> distances(10);
    std::vector<idx_t> labels(10);

    index1->search(1, query.data(), 10,
                 distances.data(), labels.data());

    // 清理
    delete index1;
    delete index2;
    delete index3;
    delete index4;
}
```

---

## 第二十五部分：向量预处理技术

### 25.1 L2 归一化

**L2 归一化**将向量缩放到单位球面，提高余弦相似度计算效率。

```cpp
// L2 彸一化实现

class L2Normalizer {
public:
    static void normalize(size_t n, const float* x, float* norm) {
        // 计算每个向量的 L2 范数
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            const float* xi = x + i * d;
            float norm = 0;

            for (size_t j = 0; j < d; j++) {
                norm += xi[j] * xi[j];
            }

            norm = sqrtf(norm);
            norm[i] = (norm > 0) ? norm : 1.0f;
        }
    }

    static void normalize_inplace(size_t n, float* x) {
        std::vector<float> norms(n);
        normalize(n, x, norms.data());

        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            float* xi = x + i * d;
            float norm = norms[i];

            for (size_t j = 0; j < d; j++) {
                xi[j] /= norm;
            }
        }
    }

    static void normalize_with_norms(
            size_t n,
            float* x,
            const float* norms) {

        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            float* xi = x + i * d;
            float norm = norms[i];

            for (size_t j = 0; j < d; j++) {
                xi[j] /= norm;
            }
        }
    }
};
```

### 25.2 PCA 降维

**PCA (Principal Component Analysis)** 降维可以减少向量维度，提升性能。

```cpp
// PCA 降维

class PCAReducer {
public:
    size_t d_in;    // 输入维度
    size_t d_out;   // 输出维度
    std::vector<float> mean;    // [d_in] 均值
    std::vector<float> pca_matrix;  // [d_out, d_in] PCA 矩阵

    PCAReducer(size_t d_in, size_t d_out)
        : d_in(d_in), d_out(d_out) {
        mean.resize(d_in, 0);
        pca_matrix.resize(d_out * d_in, 0);
    }

    void train(size_t n, const float* x) {
        // 1. 计算均值
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d_in; j++) {
                mean[j] += x[i * d_in + j];
            }
        }
        for (size_t j = 0; j < d_in; j++) {
            mean[j] /= n;
        }

        // 2. 中心化数据
        std::vector<float> centered(n * d_in);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d_in; j++) {
                centered[i * d_in + j] = x[i * d_in + j] - mean[j];
            }
        }

        // 3. 计算协方差矩阵
        std::vector<float> cov(d_in * d_in, 0);
        // cov = centered^T * centered / n
        for (size_t i = 0; i < d_in; i++) {
            for (size_t j = 0; j < d_in; j++) {
                for (size_t k = 0; k < n; k++) {
                    cov[i * d_in + j] +=
                        centered[k * d_in + i] * centered[k * d_in + j];
                }
                cov[i * d_in + j] /= n;
            }
        }

        // 4. 计算特征向量（前 d_out 个最大特征值对应的向量）
        // 这里简化为随机初始化
        // 实际应该使用 SVD 或特征值分解

        // 5. 选择前 d_out 个主成分
        // ...（实现省略）
    }

    void transform(size_t n, const float* x, float* output) {
        // y = (x - mean) * pca_matrix^T
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d_out; j++) {
                float sum = 0;
                for (size_t k = 0; k < d_in; k++) {
                    float centered = x[i * d_in + k] - mean[k];
                    sum += centered * pca_matrix[j * d_in + k];
                }
                output[i * d_out + j] = sum;
            }
        }
    }

    void inverse_transform(size_t n, const float* compressed, float* output) {
        // x = compressed * pca_matrix + mean
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d_in; j++) {
                float sum = mean[j];
                for (size_t k = 0; k < d_out; k++) {
                    sum += compressed[i * d_out + k] * pca_matrix[k * d_in + j];
                }
                output[i * d_in + j] = sum;
            }
        }
    }
};
```

### 25.3 随机投影

**随机投影**（Random Projection）是一种简单的降维方法，基于 Johnson-Lindenstrauss 引理。

```cpp
// 随机投影降维

class RandomProjector {
public:
    size_t d_in;
    size_t d_out;
    std::vector<float> projection_matrix;  // [d_out, d_in]

    RandomProjector(size_t d_in, size_t d_out)
        : d_in(d_in), d_out(d_out) {

        projection_matrix.resize(d_out * d_in);

        // 生成随机高斯矩阵
        std::mt19937 rng(12345);
        std::normal_distribution<float> dist(0.0f, 1.0f / sqrtf(d_out));

        for (size_t i = 0; i < d_out * d_in; i++) {
            projection_matrix[i] = dist(rng);
        }
    }

    void project(size_t n, const float* x, float* output) {
        // output = x * projection_matrix^T
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d_out; j++) {
                float sum = 0;
                for (size_t k = 0; k < d_in; k++) {
                    sum += x[i * d_in + k] * projection_matrix[j * d_in + k];
                }
                output[i * d_out + j] = sum;
            }
        }
    }
};
```

### 25.4 OPQ 乘积量化器

**OPQ (Optimized Product Quantization)** 是 PQ 的一种变体，使用旋转优化。

```cpp
// 位置: faiss/impl/OpqProductQuantizer.h

namespace faiss {

struct OpqProductQuantizer : ProductQuantizer {
    // 旋转矩阵 [d][d]
    std::vector<float> rotation;

    // 训练 OPQ
    void train(size_t n, const float* x) override {
        // 1. 初始化 PQ
        ProductQuantizer::train(n, x);

        // 2. 学习最优旋转矩阵
        train_rotation(n, x);
    }

private:
    void train_rotation(size_t n, const float* x) {
        // 使用非负矩阵分解(NMF)或交替优化
        // 目标：最小化量化误差

        // 初始化旋转为单位矩阵
        rotation.resize(d * d, 0);
        for (size_t i = 0; i < d; i++) {
            rotation[i * d + i] = 1.0f;
        }

        // 迭代优化
        for (int iter = 0; iter < 100; iter++) {
            // 计算当前旋转下的量化误差
            float error = compute_quantization_error(n, x, rotation);

            // 计算梯度
            std::vector<float> grad = compute_rotation_gradient(n, x, rotation);

            // 更新旋转矩阵
            update_rotation(grad, 0.01f);

            printf("Iteration %d: error = %.6f\n", iter, error);
        }
    }

    float compute_quantization_error(size_t n, const float* x,
                                     const std::vector<float>& rot) {
        // 计算旋转和量化后的误差
        float total_error = 0;

        for (size_t i = 0; i < n; i++) {
            // 旋转
            std::vector<float> rotated(d_out);
            apply_rotation(x + i * d_in, rot, rotated.data());

            // 量化
            std::vector<uint8_t> code(M);
            encode(rotated.data(), code.data());

            // 反量化
            std::vector<float> reconstructed(d_in);
            decode(code.data(), reconstructed.data());

            // 计算误差
            for (size_t j = 0; j < d_in; j++) {
                float diff = x[i * d_in + j] - reconstructed[j];
                total_error += diff * diff;
            }
        }

        return total_error / n;
    }
};

} // namespace faiss
```

---

## 第二十六部分：高级索引类型

### 26.1 LSH 索引（局部敏感哈希）

**LSH (Locality-Sensitive Hashing)** 使用哈希函数实现近似最近邻搜索。

```cpp
// 位置: faiss/IndexLSH.h

namespace faiss {

template <typename SG_t>
struct IndexLSH : Index {
    size_t nbits;          // 哈希位数
    size_t nrotate;       // 旋转次数

    std::vector<std::vector<uint8_t>> hash_tables;  // [n_tables][n][nbits/8]

    IndexLSH(size_t d, size_t nbits, size_t nrotate = 0)
        : Index(d, METRIC_L2), nbits(nbits), nrotate(nrotate) {

        // 初始化哈希表
        size_t n_tables = (nrotate + 1) * 32;
        hash_tables.resize(n_tables);
        for (auto& table : hash_tables) {
            table.resize((nbits + 7) / 8, 0);
        }
    }

    void add(idx_t n, const float* x) override {
        for (idx_t i = 0; i < n; i++) {
            // 为每个向量计算哈希
            for (size_t t = 0; t < hash_tables.size(); t++) {
                uint8_t hash = compute_hash(x + i * d, t);
                size_t offset = t * n + i;

                // 将哈希值存储到哈希表中
                size_t byte_offset = offset / 8;
                size_t bit_offset = offset % 8;

                hash_tables[t][byte_offset] |= (hash << bit_offset);
            }
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

        for (idx_t q = 0; q < n; q++) {
            const float* xq = x + q * d;

            // 计算查询的哈希值
            std::vector<std::pair<size_t, uint8_t>> candidates;

            for (size_t t = 0; t < hash_tables.size(); t++) {
                uint8_t query_hash = compute_hash(xq, t);

                // 找到与查询哈希匹配的向量
                size_t match_count = count_matches(t, query_hash);
                // ... 收集候选 ...
            }

            // 从候选中返回 Top-K
            // ... 实现细节 ...
        }
    }

private:
    uint8_t compute_hash(const float* x, size_t table_id) const {
        // 简化的哈希函数
        uint8_t hash = 0;
        for (size_t i = 0; i < d; i++) {
            hash ^= (uint8_t)(x[i] * 255);
        }
        return hash;
    }

    size_t count_matches(size_t table_id, uint8_t query_hash) const {
        // 计算有多少向量的哈希与查询哈希匹配
        // ... 实现 ...
        return 0;
    }
};

} // namespace faiss
```

### 26.2 平面索引

**平面索引（2D Index）专门用于二维向量的可视化应用。

```cpp
// 2D 向量索引

class Index2D : public Index {
public:
    struct Point2D {
        float x, y;
        idx_t id;
    };

    std::vector<Point2D> points;

    Index2D() : Index(2, METRIC_L2) {
        is_trained = true;
    }

    void add(idx_t n, const float* x) override {
        for (idx_t i = 0; i < n; i++) {
            points.push_back({x[i * 2 + 0], x[i * 2 + 1], ntotal + i});
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

        for (idx_t q = 0; q < n; q++) {
            float qx = x[q * 2 + 0];
            float qy = x[q * 2 + 1];

            // 使用 KD-Tree 进行快速最近邻搜索
            std::vector<std::pair<float, size_t>> results;

            for (size_t i = 0; i < points.size(); i++) {
                float dx = qx - points[i].x;
                float dy = qy - points[i].y;
                float dist = dx * dx + dy * dy;
                results.emplace_back(dist, points[i].id);
            }

            // 排序并返回 Top-K
            std::sort(results.begin(), results.end());

            for (idx_t j = 0; j < k && j < results.size(); j++) {
                labels[q * k + j] = results[j].second;
                distances[q * k + j] = results[j].first;
            }
        }
    }

    // 获取所有点用于可视化
    const std::vector<Point2D>& get_points() const {
        return points;
    }
};
```

### 26.3 标量量化器索引组合

```cpp
// 多种标量量化器的组合

class MultiScalarQuantizerIndex : public Index {
public:
    std::vector<ScalarQuantizer> quantizers;  // 多个 SQ 实例
    size_t nquantizers;

    MultiScalarQuantizerIndex(size_t d, size_t nq)
        : Index(d, METRIC_L2), nquantizers(nq) {

        size_t d_per_q = d / nq;
        for (size_t i = 0; i < nq; i++) {
            ScalarQuantizer sq;
            sq.d = d_per_q;
            sq.qtype = ScalarQuantizer::QT_8bit;
            quantizers.push_back(sq);
        }
    }

    void train(idx_t n, const float* x) override {
        // 训练每个标量量化器
        size_t d_per_q = d / nquantizers;

        for (size_t i = 0; i < nquantizers; i++) {
            quantizers[i].train(n, x + i * d_per_q);
        }
        is_trained = true;
    }

    void add(idx_t n, const float* x) override {
        FAISS_THROW_IF_NOT(is_trained);
        size_t d_per_q = d / nquantizers;

        std::vector<uint8_t> all_codes(n * nquantizers);

        for (size_t i = 0; i < n; i++) {
            for (size_t q = 0; q < nquantizers; q++) {
                uint8_t code;
                quantizers[q].encode(x + i * d + q * d_per_q, &code);
                all_codes[i * nquantizers + q] = code;
            }
        }

        // 存储编码
        // ...
        ntotal += n;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 对每个查询，解量化所有量化器并计算总距离
        // ... 实现细节 ...
    }
};
```

---

## 第二十七部分：索引合并与分割

### 27.1 索引合并

```cpp
// 合并多个索引

class IndexMerger {
public:
    // 合并多个同类型索引
    static Index* merge_indexes(
            const std::vector<Index*>& indexes,
            MergeStrategy strategy = MERGE Concat) {

        if (indexes.empty()) {
            return nullptr;
        }

        size_t d = indexes[0]->d;

        switch (strategy) {
            case MERGE_Concat:
                return merge_concat(indexes);

            case MERGE_Add:
                return merge_add(indexes);

            case MERGE_Vstack:
                return merge_vstack(indexes);

            default:
                return nullptr;
        }
    }

private:
    enum MergeStrategy {
        MERGE_Concat,   // 拼接索引
        MERGE_Add,       // 向量加法索引
        MERGE_Vstack     // 垂直堆叠索引
    };

    static Index* merge_concat(const std::vector<Index*>& indexes) {
        // 拼接所有索引的向量
        IndexFlat* merged = new IndexFlatL2(indexes[0]->d);

        for (auto* idx : indexes) {
            if (auto* flat = dynamic_cast<IndexFlat*>(idx)) {
                merged->add(flat->ntotal, flat->codes.data());
            }
        }

        return merged;
    }

    static Index* merge_add(const std::vector<Index*>& indexes) {
        // 创建一个 MetaIndex，在添加时累加向量
        // ... 实现 ...
        return nullptr;
    }
};
```

### 27.2 索引分割

```cpp
// 索引分割用于并行处理

class IndexSplitter {
public:
    static std::vector<Index*> split_index(
            Index* source,
            size_t num_shards) {

        std::vector<Index*> shards(num_shards);
        size_t d = source->d;
        idx_t ntotal = source->ntotal;

        // 重构所有向量
        std::vector<float> all_vectors(ntotal * d);
        source->reconstruct_n(ntotal, all_vectors.data());

        size_t shard_size = (ntotal + num_shards - 1) / num_shards;

        // 创建分片
        for (size_t i = 0; i < num_shards; i++) {
            IndexFlat* shard = new IndexFlatL2(d);

            size_t start = i * shard_size;
            size_t end = std::min(start + shard_size, (size_t)ntotal);

            shard->add(end - start, all_vectors.data() + start * d);
            shards[i] = shard;
        }

        return shards;
    }

    static Index* split_index_by_range(
            Index* source,
            const std::vector<std::pair<idx_t, idx_t>>& ranges) {

        IndexFlat* shard = new IndexFlatL2(source->d);

        for (auto [start, end] : ranges) {
            size_t count = end - start + 1;

            // 重构指定范围的向量
            std::vector<float> vectors(count * source->d);
            for (idx_t i = start; i <= end; i++) {
                source->reconstruct(1, vectors.data() + (i - start) * source->d);
            }

            shard->add(count, vectors.data());
        }

        return shard;
    }
};
```

---

## 第二十八部分：性能分析工具

### 28.1 性能分析器

```cpp
// Faiss 内置性能分析

class FaissProfiler {
public:
    struct ProfilingStats {
        size_t distance_computations;
        size_t memory_accesses;
        double avg_latency;
        size_t cache_misses;
    };

    static void enable_profiling() {
        // 设置 Faiss 内部的性能统计
        faiss::enable_profiling = true;
    }

    static ProfilingStats get_stats(Index* index) {
        ProfilingStats stats;

        // 获取距离计算次数
        stats.distance_computations = index->get_distance_computation_count();

        // 获取内存访问统计
        stats.memory_accesses = index->get_memory_access_count();

        return stats;
    }

    static void print_report(Index* index) {
        printf("=== Faiss Performance Report ===\n");

        // 打印索引统计信息
        printf("Index type: %s\n", typeid(*index).name());
        printf("Dimension: %zu\n", index->d);
        printf("Total vectors: %zu\n", index->ntotal);

        if (auto* ivf = dynamic_cast<IndexIVF*>(index)) {
            printf("IVF nlist: %zu\n", ivf->nlist);
            printf("IVF nprobe: %zu\n", ivf->nprobe);

            // 打印倒排表统计
            auto* invlists = ivf->invlists;
            printf("Inverted list imbalance: %.3f\n",
                   invlists->imbalance_factor());
        }

        if (auto* pq = dynamic_cast<IndexIVFPQ*>(index)) {
            printf("PQ M: %zu\n", pq->pq.M);
            printf("PQ nbits: %zu\n", pq->pq.nbits);
        }

        printf("=============================\n");
    }
};
```

### 28.2 内存分析器

```cpp
// 内存使用分析

class MemoryAnalyzer {
public:
    struct MemoryStats {
        size_t index_memory;
        size_t vector_memory;
        size_t overhead_memory;
        size_t total_memory;
    };

    static MemoryStats analyze(Index* index) {
        MemoryStats stats;

        // 基础索引结构内存
        stats.index_memory = sizeof(*index);

        // 向量数据内存
        if (auto* flat = dynamic_cast<IndexFlat*>(index)) {
            stats.vector_memory = flat->codes.size() * sizeof(float);
        }

        // 倒排表内存
        if (auto* ivf = dynamic_cast<IndexIVF*>(index)) {
            auto* invlists = ivf->invlists;
            size_t list_size = invlists->compute_ntotal();
            size_t code_size = ivf->code_size;

            stats.vector_memory = list_size * code_size;

            // 倒排表开销
            stats.overhead_memory =
                sizeof(*invlists) +
                invlists->nlist * sizeof(size_t);
        }

        stats.total_memory = stats.index_memory +
                             stats.vector_memory +
                             stats.overhead_memory;

        return stats;
    }

    static void print_memory_report(const MemoryStats& stats) {
        printf("=== Memory Usage Report ===\n");
        printf("Index structure: %zu MB\n",
               stats.index_memory / (1024 * 1024));
        printf("Vector data: %zu MB\n",
               stats.vector_memory / (1024 * 1024));
        printf("Overhead: %zu MB\n",
               stats.overhead_memory / (1024 * 1024));
        printf("Total: %zu MB\n",
               stats.total_memory / (1024 * 1024));
        printf("=========================\n");
    }
};
```

### 28.3 热点图分析

```cpp
// 热点图分析工具

class HotspotAnalyzer {
public:
    struct HotspotInfo {
        std::vector<size_t> frequent_access_indices;
        std::vector<size_t> slow_queries;
    };

    static HotspotAnalyzer analyze(
            Index* index,
            const float* queries,
            size_t n_queries) {

        HotspotInfo info;

        // 收集被频繁访问的向量ID
        std::map<size_t, size_t> access_count;

        // 模拟搜索并记录访问
        size_t k = 10;
        std::vector<float> distances(k);
        std::vector<idx_t> labels(k);

        for (size_t i = 0; i < n_queries; i++) {
            index->search(1, queries + i * index->d, k,
                         distances.data(), labels.data());

            for (size_t j = 0; j < k; j++) {
                access_count[labels[j]]++;
            }
        }

        // 找出最常访问的向量
        std::vector<std::pair<size_t, size_t>> sorted_counts;
        for (const auto& [id, count] : access_count) {
            sorted_counts.emplace_back(count, id);
        }
        std::sort(sorted_counts.rbegin(), sorted_counts.rend());

        // 取前 100 个最常访问的
        for (size_t i = 0; i < std::min(size_t)100, sorted_counts.size()); i++) {
            info.frequent_access_indices.push_back(sorted_counts[i].second);
        }

        return info;
    }
};
```

---

## 第二十九部分：生产环境最佳实践

### 29.1 部署架构

```cpp
// 生产环境部署架构

class ProductionIndexService {
public:
    struct Config {
        std::string index_path;
        size_t dim;
        std::string index_type;
        size_t replication_factor = 2;
        bool enable_cache = true;
        size_t cache_size = 1000000;
    };

    ProductionIndexService(const Config& cfg) : config(cfg) {
        // 初始化
        load_or_create_index();

        // 创建副本（用于高可用）
        if (cfg.replication_factor > 1) {
            create_replicas();
        }

        // 初始化查询缓存
        if (cfg.enable_cache) {
            cache = std::make_unique<LRUCache>(cfg.cache_size);
        }
    }

    std::vector<std::pair<size_t, float>> search(
            const float* query,
            size_t k) {

        // 1. 检查缓存
        std::string cache_key = compute_cache_key(query, config.dim);
        if (config.enable_cache && cache->exists(cache_key)) {
            return cache->get(cache_key);
        }

        // 2. 执行搜索
        std::vector<float> distances(k);
        std::vector<idx_t> labels(k);

        auto start = std::chrono::high_resolution_clock::now();
        primary_index->search(1, query, k,
                             distances.data(), labels.data());
        auto end = std::chrono::high_resolution_clock::now();

        // 3. 更新缓存
        if (config.enable_cache) {
            cache->put(cache_key, distances, labels);
        }

        // 4. 记录延迟
        record_latency(start, end);

        // 5. 转换结果
        std::vector<std::pair<size_t, float>> results;
        for (size_t i = 0; i < k; i++) {
            results.emplace_back(labels[i], distances[i]);
        }

        return results;
    }

private:
    Config config;
    Index* primary_index;
    std::vector<Index*> replicas;
    std::unique_ptr<LRUCache> cache;

    void load_or_create_index() {
        // 尝试加载现有索引
        if (file_exists(config.index_path.c_str())) {
            printf("Loading existing index from %s\n",
                   config.index_path.c_str());
            primary_index = read_index(config.index_path.c_str());
        } else {
            printf("Creating new index: %s\n",
                   config.index_type.c_str());
            primary_index = index_factory(
                config.dim,
                config.index_type.c_str(),
                nullptr
            );
            // ... 训练和添加数据 ...

            save_index();
        }
    }

    void create_replicas() {
        for (size_t i = 0; i < config.replication_factor - 1; i++) {
            Index* replica = read_index(config.index_path.c_str());
            replicas.push_back(replica);
        }
    }

    void save_index() {
        // 定期保存
        write_index(primary_index, config.index_path.c_str());
    }

    std::string compute_cache_key(const float* query, size_t d) {
        // 简化的缓存键：查询向量的哈希
        size_t hash = 0;
        for (size_t i = 0; i < d; i++) {
            hash ^= std::hash<float>{}(query[i]) + 0x9e3779b9;
            hash *= 31;
        }
        return std::to_string(hash);
    }

    bool file_exists(const char* path) {
        struct stat buffer;
        return (stat(path, &buffer) == 0);
    }
};
```

### 29.2 监控和告警

```cpp
// 生产环境监控系统

class IndexMonitor {
public:
    struct Alert {
        std::string message;
        double value;
        double threshold;
        AlertLevel level;
    };

    enum AlertLevel {
        INFO,
        WARNING,
        CRITICAL
    };

    static void monitor_index(Index* index) {
        printf("=== Monitoring Index ===\n");

        // 检查内存使用
        auto stats = FaissProfiler::get_stats(index);
        printf("Distance computations: %zu\n", stats.distance_computations);
        printf("Memory accesses: %zu\n", stats.memory_accesses);

        // 检查索引健康状态
        check_index_health(index);

        // 检查查询延迟
        check_query_latency(index);

        printf("=====================\n");
    }

    static void check_index_health(Index* index) {
        if (auto* ivf = dynamic_cast<IndexIVF*>(index)) {
            auto* invlists = ivf->invlists;

            // 检查倒排表平衡性
            double imbalance = invlists->imbalance_factor();
            if (imbalance > 3.0) {
                Alert alert;
                alert.message = "Inverted list imbalance too high";
                alert.value = imbalance;
                alert.threshold = 3.0;
                alert.level = WARNING;
                alert.trigger();
            }

            // 检查空倒排表
            size_t empty_count = 0;
            for (size_t i = 0; i < invlists->nlist; i++) {
                if (invlists->list_size(i) == 0) {
                    empty_count++;
                }
            }

            double empty_ratio = (double)empty_count / invlists->nlist;
            if (empty_ratio > 0.5) {
                Alert alert;
                alert.message = "Too many empty inverted lists";
                alert.value = empty_ratio;
                alert.threshold = 0.5;
                alert.level = WARNING;
                alert.trigger();
            }
        }
    }

private:
    struct Alert {
        std::string message;
        double value;
        double threshold;
        AlertLevel level;

        void trigger() const {
            const char* level_str[] = {"INFO", "WARNING", "CRITICAL"};
            printf("[%s] %s: %.2f (threshold: %.2f)\n",
                   level_str[level], message.c_str(), value, threshold);
        }
    };
};
```

### 29.3 A/B 测试框架

```cpp
// A/B 测试索引性能

class IndexABTester {
public:
    struct TestResult {
        std::string index_name;
        double avg_latency_ms;
        double qps;
        float recall;
        double memory_mb;
    };

    static std::vector<TestResult> ab_test(
            const std::vector<std::string>& index_descriptions,
            const float* train_data,
            size_t n_train,
            const float* test_queries,
            size_t n_test,
            const idx_t* ground_truth) {

        std::vector<TestResult> results;

        for (const auto& desc : index_descriptions) {
            printf("Testing: %s\n", desc.c_str());

            TestResult result;
            result.index_name = desc;

            // 1. 创建索引
            auto* index = index_factory(d, desc.c_str(), train_data);

            // 2. 测试添加性能
            auto start_add = std::chrono::high_resolution_clock::now();
            index->add(n_train, train_data);
            auto end_add = std::chrono::high_resolution_clock::now();

            double add_time = std::chrono::duration<double, std::milli>(
                end_add - start_add).count();

            // 3. 测试搜索性能
            const size_t k = 10;
            const size_t warmup = 100;
            std::vector<float> distances(n_test * k);
            std::vector<idx_t> labels(n_test * k);

            // Warmup
            index->search(warmup, test_queries, k,
                         distances.data(), labels.data());

            // 实际测试
            auto start_search = std::chrono::high_resolution_clock::now();
            index->search(n_test, test_queries, k,
                         distances.data(), labels.data());
            auto end_search = std::chrono::high_resolution_clock::now();

            double search_time = std::chrono::duration<double, std::milli>(
                end_search - start_search).count();

            // 4. 计算召回率
            float recall = compute_recall(
                labels.data(), ground_truth, n_test, k);

            // 5. 计算内存使用
            double memory_mb = estimate_memory_size(index);

            // 6. 记录结果
            result.avg_latency_ms = search_time / n_test;
            result.qps = 1000.0 / result.avg_latency_ms;
            result.recall = recall;
            result.memory_mb = memory_mb;

            results.push_back(result);

            // 打印结果
            printf("  Add time: %.2f s\n", add_time);
            printf("  Search latency: %.4f ms\n", result.avg_latency_ms);
            printf("  QPS: %.2f\n", result.qps);
            printf("  Recall@10: %.4f\n", recall);
            printf("  Memory: %.2f MB\n\n", memory_mb);

            delete index;
        }

        // 比较不同索引的性能
        printf("=== Comparison ===\n");
        for (const auto& r : results) {
            printf("%s: QPS=%.2f, Recall=%.4f, Memory=%.2f MB\n",
                   r.index_name.c_str(),
                   r.qps,
                   r.recall,
                   r.memory_mb);
        }

        return results;
    }

private:
    static float compute_recall(
            const idx_t* labels,
            const idx_t* ground_truth,
            size_t n,
            size_t k) {

        size_t correct = 0;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < k; j++) {
                if (labels[i * k + j] == ground_truth[i * k + j]) {
                    correct++;
                    break;
                }
            }
        }
        return (float)correct / n;
    }

    static double estimate_memory_size(Index* index) {
        // 估算内存使用
        double size = 0;

        if (auto* flat = dynamic_cast<IndexFlat*>(index)) {
            size = flat->codes.size() * sizeof(float);
        }
        else if (auto* ivf = dynamic_cast<IndexIVF*>(index)) {
            size = ivf->invlists->compute_ntotal() * ivf->code_size;
            size += ivf->nlist * sizeof(size_t);
        }

        return size / (1024.0 * 1024.0);  // 转换为 MB
    }
};
```

---

## 第三十部分：故障排查指南

### 30.1 常见问题与解决方案

```cpp
// 故障排查工具箱

class IndexTroubleshooter {
public:
    // 问题1：搜索结果不准确
    static void diagnose_low_recall(Index* index) {
        printf("=== Diagnosing Low Recall ===\n");

        // 检查1：IVF nprobe 是否太小
        if (auto* ivf = dynamic_cast<IndexIVF*>(index)) {
            printf("IVF nprobe: %zu\n", ivf->nprobe);

            if (ivf->nprobe < 10) {
                printf("  → Suggestion: Increase nprobe\n");
                printf("     Try: nprobe = min(100, nlist/10)\n");
            }
        }

        // 检查2：HNSW ef_search 是否太小
        if (auto* hnsw = dynamic_cast<IndexHNSW*>(index)) {
            printf("HNSW ef_search: %d\n", hnsw->hnsw.ef_search);

            if (hnsw->hnsw.ef_search < 16) {
                printf("  → Suggestion: Increase ef_search\n");
                printf("     Try: ef_search = 32 or 64\n");
            }
        }

        // 检查3：索引是否经过训练
        if (!index->is_trained) {
            printf("  → Error: Index not trained!\n");
        }

        // 检查4：训练数据是否足够
        if (auto* ivf = dynamic_cast<IndexIVF*>(index)) {
            if (ivf->ntotal < ivf->nlist * 256) {
                printf("  → Warning: Insufficient training data\n");
                printf("     Recommend: at least nlist * 256 vectors\n");
            }
        }

        printf("=============================\n");
    }

    // 问题2：搜索速度太慢
    static void diagnose_slow_search(Index* index) {
        printf("=== Diagnosing Slow Search ===\n");

        // 检查1：是否使用了 SIMD
        #ifndef FAISS_USE_AVX2
        printf("  → Warning: AVX2 not enabled\n");
        #endif

        // 检查2：线程数设置
        #ifdef _OPENMP
        int max_threads = omp_get_max_threads();
        printf("OpenMP threads: %d\n", max_threads);
        #else
        printf("  → Warning: Multi-threading not enabled\n");
        #endif

        // 检查3：是否使用了 GPU
        #if defined(FAISS_ENABLE_GPU)
        printf("GPU support: Enabled\n");
        #else
        printf("  → Consider enabling GPU for large datasets\n");
        #endif

        // 检查4：索引类型是否合适
        if (auto* flat = dynamic_cast<IndexFlat*>(index)) {
            if (flat->ntotal > 100000) {
                printf("  → Suggestion: Use IVF or HNSW for large datasets\n");
            }
        }

        // 检查5：内存是否充足
        size_t available_memory = get_available_memory_mb();
        printf("Available memory: %zu MB\n", available_memory);

        size_t index_memory = estimate_memory_size(index);
        if (index_memory > available_memory * 0.8) {
            printf("  → Warning: Index may not fit in memory\n");
        }

        printf("=============================\n");
    }

    // 问题3：内存占用过大
    static void diagnose_high_memory(Index* index) {
        printf("=== Diagnosing High Memory ===\n");

        size_t index_memory = estimate_memory_size(index);
        printf("Estimated memory: %.2f MB\n", index_memory);

        // 检查向量存储格式
        if (auto* flat = dynamic_cast<IndexFlat*>(index)) {
            printf("Storage: Float32 (4 bytes per dimension)\n");
        }
        else if (auto* ivf_pq = dynamic_cast<IndexIVFPQ*>(index)) {
            printf("Storage: PQ%zu (bit), %zu bytes per vector\n",
                   ivf_pq->pq.nbits,
                   ivf_pq->pq.M);
        }

        // 优化建议
        printf("\nOptimization suggestions:\n");

        if (dynamic_cast<IndexFlat*>(index)) {
            printf("  → Switch to IndexIVFPQ for ~64x compression\n");
        }
        else if (dynamic_cast<IndexIVFFlat*>(index)) {
            printf("  → Switch to IndexIVFPQ for ~8x compression\n");
        }
        else if (dynamic_cast<IndexIVFPQ*>(index)) {
            printf("  → Increase PQ compression (reduce M or nbits)\n");
        }

        // 检查是否可以使用 ScalarQuantizer
        printf("  → Consider IndexScalarQuantizer for 4x compression\n");

        // 检查倒排表实现
        if (auto* ivf = dynamic_cast<IndexIVF*>(index)) {
            printf("  → Use OnDiskInvertedLists to offload memory\n");
        }

        printf("=============================\n");
    }

    // 问题4：添加向量失败
    static void diagnose_add_failure(Index* index, const float* x, idx_t n) {
        printf("=== Diagnosing Add Failure ===\n");

        // 检查1：维度匹配
        if (x && index->d > 0) {
            printf("Expected dimension: %zu\n", index->d);
            printf("Input dimension: %zu\n", n > 0 ? index->d : n);

            // 检查输入数据有效性
            bool has_nan = false;
            for (size_t i = 0; i < n * index->d; i++) {
                if (std::isnan(x[i]) || std::isinf(x[i])) {
                    has_nan = true;
                    break;
                }
            }

            if (has_nan) {
                printf("  → Error: Input contains NaN or Inf\n");
                printf("     Solution: Filter or normalize your data\n");
            }
        }

        // 检查2：是否需要训练
        if (!index->is_trained) {
            printf("  → Error: Index not trained\n");
            printf("     Solution: Call train() before add()\n");
        }

        // 检查3：倒排表是否已满
        if (auto* ivf = dynamic_cast<IndexIVF*>(index)) {
            if (ivf->invlists->size() == ivf->invlists->nlist) {
                printf("  → Warning: All inverted lists may be full\n");
            }
        }

        // 检查4：内存是否足够
        size_t required_memory = n * index->code_size;
        size_t available_memory = get_available_memory_mb() * 1024 * 1024;

        if (required_memory > available_memory) {
            printf("  → Error: Insufficient memory\n");
            printf("     Required: %zu MB\n", required_memory / (1024 * 1024));
            printf("     Available: %zu MB\n", available_memory / (1024 * 1024));
        }

        printf("=============================\n");
    }

private:
    static size_t get_available_memory_mb() {
        // Linux 实现
        std::ifstream file("/proc/meminfo");
        std::string line;

        while (std::getline(file, line)) {
            if (line.find("MemAvailable:") == 0) {
                size_t kb;
                sscanf(line.c_str(), "MemAvailable: %zu", &kb);
                return kb;
            }
        }

        return 0;
    }

    static size_t estimate_memory_size(Index* index) {
        // 估算索引内存占用
        size_t size = 0;

        if (auto* flat = dynamic_cast<IndexFlat*>(index)) {
            size = flat->codes.size() * sizeof(float);
        }
        else if (auto* ivf = dynamic_cast<IndexIVF*>(index)) {
            auto* invlists = ivf->invlists;
            size = invlists->compute_ntotal() * ivf->code_size;
            size += invlists->nlist * sizeof(size_t);
        }

        return size;
    }
};
```

### 30.2 性能调优清单

```cpp
// 性能调优检查清单

class PerformanceOptimizationChecklist {
public:
    struct ChecklistItem {
        std::string item;
        bool checked;
        std::string notes;
    };

    static std::vector<ChecklistItem> generate_checklist(Index* index) {
        std::vector<ChecklistItem> checklist = {
            {"数据预处理", false, "向量是否已归一化？"},
            {"索引类型", false, "索引类型是否适合数据规模？"},
            {"训练充分性", false, "训练数据是否足够（> nlist * 256）？"},
            {"nprobe设置", false, "IVF nprobe 是否足够大？"},
            {"ef_search", false, "HNSW ef_search 是否足够大？"},
            {"SIMD启用", false, "AVX2/AVX-512 是否启用？"},
            {"多线程", false, "OpenMP 是否启用？线程数设置？"},
            {"GPU利用", false, "是否可以利用GPU加速？"},
            {"内存对齐", false, "数据是否内存对齐？"},
            {"批量查询", false, "是否使用批量查询？"},
            {"缓存策略", false, "是否有查询缓存？"},
            {"负载均衡", false, "是否使用了负载均衡？"}
        };

        // 检查每一项
        for (auto& item : checklist) {
            item.checked = check_item(item, index);
        }

        return checklist;
    }

    static void print_optimization_suggestions(
            const std::vector<ChecklistItem>& checklist) {

        printf("=== Optimization Suggestions ===\n");
        for (const auto& item : checklist) {
            if (!item.checked) {
                printf("[-] %s: %s\n",
                       item.item.c_str(), item.notes.c_str());
            } else {
                printf("[✓] %s: OK\n", item.item.c_str());
            }
        }
        printf("=============================\n");
    }

private:
    static bool check_item(const ChecklistItem& item, Index* index) {
        if (item.item == "数据预处理") {
            // 检查向量是否归一化
            // ...
            return true;
        }
        // ... 其他检查项的实现 ...
        return false;
    }
};
```

---

## 总结与学习建议

### 课程回顾

我们深入讲解了 Faiss 中核心向量搜索算法：

1. **Flat Index**：基础暴力搜索
2. **IVF Index**：聚类分区索引
3. **HNSW Index**：图结构高级索引
4. **Product Quantization**：向量压缩技术
5. **NSG & NNDescent**：图索引变体
6. **FastScan**：SIMD 优化扫描
7. **Scalar Quantizer & RaBitQ**：新型量化器
8. **Binary Index**：二进制向量索引
9. **GPU 实现**：GPU 加速
10. **Composite 索引**：组合高级索引
11. **Residual Quantizer**：残差量化器
12. **SIMD 距离计算优化**：硬件加速
13. **性能最佳实践**：调优与监控
14. **实战案例**：完整应用场景
15. **其他距离度量**：L1、Canberra、Bray-Curtis、Jaccard、余弦相似度
16. **IDSelector**：过滤搜索
17. **索引序列化**：持久化与内存映射
18. **分布式向量搜索**：分片与负载均衡
19. **测试验证**：单元测试、压力测试、精度测试
20. **高级技巧**：自适应选择、多阶段搜索、动态调优、故障恢复
21. **Index Factory**：索引工厂
22. **向量预处理**：归一化、PCA、随机投影、OPQ
23. **高级索引类型**：LSH、2D索引、多标量量化器
24. **索引合并与分割**：合并、分片策略
25. **性能分析工具**：分析器、内存分析、热点分析
26. **生产环境实践**：部署、监控、A/B测试、故障排查
27. **优化清单**：性能调优检查清单

### 学习建议

**完整课程路线（3周计划）**：

```
Week 1: 基础索引算法
├── Day 1-2: 向量搜索基础 + Flat Index
├── Day 3-4: IVF Index（倒排文件）
├── Day 5: 实战练习
└── 目标：理解基本原理，能使用基础索引

Week 2: 高级索引与优化
├── Day 1-2: HNSW Index（图算法）
├── Day 3-4: Product Quantization（乘积量化）
├── Day 5: NSG & NNDescent（图索引变体）
└── 目标：掌握高级索引，理解优化技术

Week 3: 特殊技术与综合应用
├── Day 1-2: FastScan + SIMD 优化
├── Day 3: Scalar Quantizer & RaBitQ
├── Day 4: Binary Index + GPU
├── Day 5: Composite 索引 + 综合项目
└── 目标：了解前沿技术，能构建完整系统
```

**检查清单**：

**基础部分（Week 1）**：
- [ ] 理解向量搜索的基本原理
- [ ] 掌握 L2 和 Inner Product 距离计算
- [ ] 理解 Flat Index 的实现
- [ ] 理解 IVF 的聚类思想
- [ ] 完成 Flat + IVF 练习

**进阶部分（Week 2）**：
- [ ] 理解 HNSW 多层图结构
- [ ] 掌握 PQ 编码/解码流程
- [ ] 理解 NSG 与 HNSW 的区别
- [ ] 完成图索引练习

**高级部分（Week 3）**：
- [ ] 理解 FastScan SIMD 优化
- [ ] 掌握 Scalar Quantizer 实现
- [ ] 了解 Binary Index 和 Hamming 距离
- [ ] 理解 GPU 加速原理
- [ ] 掌握 Composite 索引组合
- [ ] 完成综合项目

**下一步**：
1. 阅读 Faiss 源码中的关键文件
2. 实现一个简单的向量索引
3. 运行性能测试
4. 尝试优化现有代码
5. 构建自己的向量搜索系统

**推荐阅读顺序**：
1. `faiss/Index.h` - 索引基类
2. `faiss/IndexFlat.h` 和 `IndexFlat.cpp`
3. `faiss/IndexIVF.h` 和 `IndexIVF.cpp`
4. `faiss/utils/Heap.h` - 堆操作
5. `faiss/invlists/InvertedLists.h` - 倒排表
6. `faiss/impl/ProductQuantizer.h` - PQ
7. `faiss/impl/HNSW.h` - HNSW
8. `faiss/IndexFastScan.h` - FastScan
9. `faiss/impl/ScalarQuantizer.h` - SQ
10. `faiss/gpu/GpuIndex.h` - GPU

**实战项目建议**：

**初级项目**：
1. 实现一个简化版的 Flat 索引
2. 实现 K-Means 聚类算法
3. 实现基本的 PQ 编码/解码

**中级项目**：
1. 实现一个简化版的 IVF 索引
2. 实现 NNDescent 图构建
3. 对比不同索引的性能

**高级项目**：
1. 构建完整的图像搜索引擎
2. 实现分布式向量搜索
3. 优化索引的内存占用
4. 在自己的数据集上测试和调优

**性能测试建议**：

```cpp
// 性能测试框架
void benchmark_index(
        Index* index,
        const char* name,
        const float* queries,
        size_t n_queries,
        size_t k) {

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> distances(n_queries * k);
    std::vector<idx_t> labels(n_queries * k);

    index->search(n_queries, queries, k, distances.data(), labels.data());

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    printf("%s:\n", name);
    printf("  Total: %.2f ms\n", ms);
    printf("  QPS: %.2f\n", n_queries / (ms / 1000.0));
    printf("  Latency: %.4f ms\n", ms / n_queries);
}
```

**深入研究方向**：
1. 新型量化算法（如 RaBitQ）
2. 动态索引更新
3. 分布式向量搜索
4. 硬件加速（FPGA、ASIC）
5. 混合精度搜索
6. 自适应参数调优

**常见问题解答**：

**Q1: 如何选择合适的索引？**
```
小规模（<10万）：IndexFlat
中规模（10万-1000万）：IndexIVFFlat
大规模（>1000万）：IndexIVFPQ 或 IndexHNSW
超大规模：IndexIVFPQ + 分片
```

**Q2: 如何平衡速度和精度？**
- IVF：调整 nprobe 参数
- HNSW：调整 ef_search 参数
- PQ：调整 M 和 nbits 参数
- 使用 IndexRefine 进行粗精两阶段搜索

**Q3: 如何减少内存占用？**
- 使用 PQ 压缩（IndexIVFPQ）
- 使用 Scalar Quantizer（IndexScalarQuantizer）
- 考虑 Binary Index（如果适用）
- 使用磁盘倒排表（OnDiskInvertedLists）

**相关资源**：
- Faiss 官方文档：https://github.com/facebookresearch/faiss/wiki
- Faiss 论文：Johnson et al., "Billion-scale similarity search with GPUs"
- HNSW 论文：Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search"
- PQ 论文：Jegou et al., "Product quantization for nearest neighbor search"

祝学习顺利！下一课见！🚀
