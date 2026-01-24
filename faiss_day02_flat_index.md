# Faiss深度课程 - 第2天：基础索引 - Flat索引详解

## 课程目标

深入理解Faiss中最基础的索引类型 - **Flat索引**，掌握精确搜索的实现细节、优化技巧和特殊变体。

---

## 1. Flat索引概述

### 1.1 什么是Flat索引

Flat索引是最简单、最基础的索引类型，它：
- **完整存储**所有原始向量
- 执行**穷举搜索**（Exhaustive Search）
- 返回**精确结果**（无近似）
- 作为其他复杂索引的**构建块**

```cpp
// faiss/IndexFlat.h
struct IndexFlat : IndexFlatCodes {
    explicit IndexFlat(idx_t d, MetricType metric = METRIC_L2);

    void search(
        idx_t n,      // 查询向量数
        const float* x,
        idx_t k,      // 返回top-k结果
        float* distances,
        idx_t* labels,
        const SearchParameters* params = nullptr) const override;

    void range_search(
        idx_t n,
        const float* x,
        float radius,  // 半径
        RangeSearchResult* result,
        const SearchParameters* params = nullptr) const override;
};
```

### 1.2 Flat索引的继承层次

```
IndexFlatCodes (存储编码的向量)
    └── IndexFlat (通用Flat索引)
        ├── IndexFlatIP (内积)
        ├── IndexFlatL2 (L2距离，带norm缓存)
        ├── IndexFlatPanorama (渐进式剪枝)
        └── IndexFlat1D (1D特化版本)
```

---

## 2. IndexFlat核心实现

### 2.1 数据存储

```cpp
// faiss/IndexFlatCodes.h - IndexFlat的基类
struct IndexFlatCodes : Index {
    std::vector<uint8_t> codes;  // 存储所有向量
    size_t code_size;             // 每个向量的字节数

    IndexFlatCodes(size_t code_size, idx_t d, MetricType metric)
        : code_size(code_size) {
        this->d = d;
        this->metric_type = metric;
    }

    // 添加向量：直接追加到codes
    void add(idx_t n, const float* x) override {
        codes.resize((ntotal + n) * code_size);
        memcpy(codes.data() + ntotal * code_size, x, n * code_size);
        ntotal += n;
    }

    // 重建向量：从codes复制
    void reconstruct(idx_t key, float* recons) const override {
        memcpy(recons, codes.data() + key * code_size, code_size);
    }
};

// IndexFlat的特殊化：code_size = sizeof(float) * d
IndexFlat::IndexFlat(idx_t d, MetricType metric)
    : IndexFlatCodes(sizeof(float) * d, d, metric) {}
```

**内存布局**：
```
codes数组:
[向量0: d个float] [向量1: d个float] [向量2: d个float] ...
 [0:4] [4:8] ... [4*d-4:4*d] [4*d:4*(d+1)] ...
```

### 2.2 精确搜索实现

```cpp
// faiss/IndexFlat.cpp
void IndexFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    IDSelector* sel = params ? params->sel : nullptr;

    // 根据度量类型选择搜索算法
    if (metric_type == METRIC_INNER_PRODUCT) {
        // 内积：使用最小堆（因为我们要最大的内积）
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_inner_product(x, get_xb(), d, n, ntotal, &res, sel);

    } else if (metric_type == METRIC_L2) {
        // L2距离：使用最大堆（因为我们要最小的距离）
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_L2sqr(x, get_xb(), d, n, ntotal, &res, nullptr, sel);

    } else {
        // 其他度量：通用处理
        knn_extra_metrics(x, get_xb(), d, n, ntotal,
                         metric_type, metric_arg, k, distances, labels);
    }
}
```

### 2.3 KNN搜索 - L2距离

```cpp
// faiss/utils/distances.cpp
void knn_L2sqr(
        const float* x,      // 查询向量 (n * d)
        const float* y,      // 数据库向量 (nb * d)
        size_t d,            // 维度
        size_t n,            // 查询数
        size_t nb,           // 数据库向量数
        float_maxheap_array_t* res,  // 结果堆
        const float* y_norms,        // 预计算的y的L2范数
        const IDSelector* sel)       // ID选择器
{
    // res->val 存储距离 (n * k)
    // res->ids 存储标签 (n * k)

#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        const float* xi = x + i * d;       // 第i个查询向量
        float* __restrict simi = res->val + i * res->k;  // 第i个查询的距离数组
        idx_t* __restrict idxi = res->ids + i * res->k;  // 第i个查询的ID数组

        // 初始化堆为无穷大
        heap_heapify<CMax<float, idx_t>>(res->k, simi, idxi);

        // 遍历所有数据库向量
        for (size_t j = 0; j < nb; j++) {
            if (sel && !sel->is_member(j)) {
                continue;  // 跳过不在选择器中的ID
            }

            // 计算距离
            float dis = 0;
            if (y_norms) {
                // 使用预计算的norm优化
                float ip = fvec_inner_product(xi, y + j * d, d);
                dis = y_norms[j] + fvec_norm_L2sqr(xi, d) - 2 * ip;
            } else {
                // 直接计算L2距离
                dis = fvec_L2sqr(xi, y + j * d, d);
            }

            // 如果距离小于堆顶，替换堆顶
            if (CMax<float, idx_t>::cmp(dis, simi[0])) {
                heap_replace_top<CMax<float, idx_t>>(res->k, simi, idxi, dis, j);
            }
        }

        // 堆排序：从小到大
        heap_reorder<CMax<float, idx_t>>(res->k, simi, idxi);
    }
}
```

### 2.4 KNN搜索 - 内积

```cpp
// faiss/utils/distances.cpp
void knn_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t n,
        size_t nb,
        float_minheap_array_t* res,
        const IDSelector* sel)
{
    // 内积使用最小堆，因为我们要最大的内积（最相似）
    // 最小堆的根是最小的，我们要保持最大的k个

#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        const float* xi = x + i * d;
        float* __restrict simi = res->val + i * res->k;
        idx_t* __restrict idxi = res->ids + i * res->k;

        // 初始化堆为负无穷大（最小堆）
        heap_heapify<CMin<float, idx_t>>(res->k, simi, idxi);

        for (size_t j = 0; j < nb; j++) {
            if (sel && !sel->is_member(j)) {
                continue;
            }

            float ip = fvec_inner_product(xi, y + j * d, d);

            // 如果内积大于堆顶（最小值），替换堆顶
            if (CMin<float, idx_t>::cmp(ip, simi[0])) {
                heap_replace_top<CMin<float, idx_t>>(res->k, simi, idxi, ip, j);
            }
        }

        // 堆排序：从大到小（因为是最小堆，需要反转）
        heap_reorder<CMin<float, idx_t>>(res->k, simi, idxi);
    }
}
```

---

## 3. IndexFlatL2 - L2范数缓存优化

### 3.1 L2距离的优化公式

```
||x - y||^2 = ||x||^2 + ||y||^2 - 2<x|y>
```

关键观察：
- `||y||^2` 对于数据库向量是**常数**
- 可以**预计算**并缓存
- 内积计算比直接L2距离**更快**

### 3.2 L2Norm缓存实现

```cpp
// faiss/IndexFlat.h
struct IndexFlatL2 : IndexFlat {
    std::vector<float> cached_l2norms;  // 缓存的L2范数

    explicit IndexFlatL2(idx_t d) : IndexFlat(d, METRIC_L2) {}

    // 计算并缓存所有向量的L2范数
    void sync_l2norms();

    // 清除缓存
    void clear_l2norms();
};

// faiss/IndexFlat.cpp
void IndexFlatL2::sync_l2norms() {
    cached_l2norms.resize(ntotal);
    fvec_norms_L2sqr(
        cached_l2norms.data(),
        reinterpret_cast<const float*>(codes.data()),
        d,
        ntotal);
}

// faiss/utils/distances.cpp
void fvec_norms_L2sqr(
        float* norms,
        const float* x,
        size_t d,
        size_t n) {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        const float* xi = x + i * d;
        float norm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            norm += xi[j] * xi[j];
        }
        norms[i] = norm;
    }
}
```

### 3.3 使用缓存优化的距离计算器

```cpp
// faiss/IndexFlat.cpp
struct FlatL2WithNormsDis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    const float* l2norms;    // 缓存的L2范数
    float query_l2norm;      // 查询向量的L2范数

    float operator()(const idx_t i) final override {
        const float* y = reinterpret_cast<const float*>(codes + i * code_size);

        // 预取L2范数到缓存
        prefetch_L2(l2norms + i);

        // 计算内积
        const float dp = fvec_inner_product(q, y, d);

        // L2距离 = ||x||^2 + ||y||^2 - 2<x|y>
        return query_l2norm + l2norms[i] - 2 * dp;
    }

    void set_query(const float* x) override {
        q = x;
        // 预计算查询向量的L2范数
        query_l2norm = fvec_norm_L2sqr(q, d);
    }

    // 批量计算4个距离（SIMD优化）
    void distances_batch_4(
            const idx_t idx0, const idx_t idx1,
            const idx_t idx2, const idx_t idx3,
            float& dis0, float& dis1, float& dis2, float& dis3) final override {
        ndis += 4;

        const float* y0 = reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* y1 = reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* y2 = reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* y3 = reinterpret_cast<const float*>(codes + idx3 * code_size);

        // 预取L2范数
        prefetch_L2(l2norms + idx0);
        prefetch_L2(l2norms + idx1);
        prefetch_L2(l2norms + idx2);
        prefetch_L2(l2norms + idx3);

        // 批量计算内积（SIMD优化）
        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        fvec_inner_product_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);

        // 转换为L2距离
        dis0 = query_l2norm + l2norms[idx0] - 2 * dp0;
        dis1 = query_l2norm + l2norms[idx1] - 2 * dp1;
        dis2 = query_l2norm + l2norms[idx2] - 2 * dp2;
        dis3 = query_l2norm + l2norms[idx3] - 2 * dp3;
    }
};
```

### 3.4 性能对比

| 方法 | 每次距离计算 | 适合场景 |
|------|-------------|---------|
| 直接L2 | d次减法 + d次乘法 + d次加法 | 小数据集，不常搜索 |
| 缓存优化 | d次乘法 + d次加法 + 3次加法 | 大数据集，频繁搜索 |
| 内存开销 | 无 | nb * 4字节 |

**何时使用缓存**：
- 数据库向量数 > 100,000
- 需要多次搜索
- 内存充足

---

## 4. IndexFlatIP - 内积索引

### 4.1 内积索引的特点

```cpp
// faiss/IndexFlat.h
struct IndexFlatIP : IndexFlat {
    explicit IndexFlatIP(idx_t d)
        : IndexFlat(d, METRIC_INNER_PRODUCT) {}
};
```

**内积 vs 余弦相似度**：

```cpp
// 内积：<x|y> = sum(x[i] * y[i])
// 余弦相似度：cos(x, y) = <x|y> / (||x|| * ||y||)

// 如果向量已归一化（L2 norm = 1），则：
// <x|y> = cos(x, y)

// 归一化函数
void normalize_vectors(float* x, size_t n, size_t d) {
    for (size_t i = 0; i < n; i++) {
        float* xi = x + i * d;
        float norm = sqrt(fvec_norm_L2sqr(xi, d));
        for (size_t j = 0; j < d; j++) {
            xi[j] /= norm;
        }
    }
}

// 使用示例
void inner_product_example() {
    int d = 128;
    int n = 1000000;

    // 创建索引
    IndexFlatIP index(d);

    // 添加归一化向量
    float* vectors = new float[n * d];
    normalize_vectors(vectors, n, d);
    index.add(n, vectors);

    // 查询向量也需要归一化
    float* query = new float[d];
    normalize_vectors(query, 1, d);

    // 搜索：返回最大的内积（最相似的向量）
    float distances[10];
    idx_t labels[10];
    index.search(1, query, 10, distances, labels);

    // distances现在包含的是内积值（越大越相似）
}
```

### 4.2 内积搜索实现

```cpp
// faiss/IndexFlat.cpp
struct FlatIPDis : FlatCodesDistanceComputer {
    size_t d;
    const float* q;
    const float* b;

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_inner_product(q, reinterpret_cast<const float*>(code), d);
    }

    void distances_batch_4(
            const idx_t idx0, const idx_t idx1,
            const idx_t idx2, const idx_t idx3,
            float& dis0, float& dis1, float& dis2, float& dis3) final override {
        ndis += 4;

        const float* y0 = reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* y1 = reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* y2 = reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* y3 = reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        fvec_inner_product_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);

        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }
};
```

---

## 5. IndexFlat1D - 一维特化索引

### 5.1 设计思路

对于一维向量，可以利用**排序**和**二分查找**大幅加速搜索。

```cpp
// faiss/IndexFlat.h
struct IndexFlat1D : IndexFlatL2 {
    bool continuous_update;  // 是否连续更新排列
    std::vector<idx_t> perm; // 排序后的索引

    explicit IndexFlat1D(bool continuous_update = true);

    // 更新排列（如果不连续更新）
    void update_permutation();

    void add(idx_t n, const float* x) override;

    void search(
        idx_t n, const float* x,
        idx_t k, float* distances, idx_t* labels) const override;
};
```

### 5.2 排列更新

```cpp
// faiss/IndexFlat.cpp
void IndexFlat1D::update_permutation() {
    perm.resize(ntotal);

    if (ntotal < 1000000) {
        // 小数据集：单线程排序
        fvec_argsort(ntotal, get_xb(), reinterpret_cast<size_t*>(perm.data()));
    } else {
        // 大数据集：多线程排序
        fvec_argsort_parallel(ntotal, get_xb(), reinterpret_cast<size_t*>(perm.data()));
    }
}

void IndexFlat1D::add(idx_t n, const float* x) {
    IndexFlatL2::add(n, x);

    if (continuous_update) {
        update_permutation();  // 每次添加后更新排序
    }
}
```

### 5.3 搜索实现 - 双向扩展

```cpp
// faiss/IndexFlat.cpp
void IndexFlat1D::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    const float* xb = get_xb();  // 原始数据

#pragma omp parallel for if (n > 10000)
    for (idx_t i = 0; i < n; i++) {
        float q = x[i];  // 查询值（1D）
        float* D = distances + i * k;
        idx_t* I = labels + i * k;

        // 二分查找找到查询值的位置
        idx_t i0 = 0, i1 = ntotal;
        idx_t wp = 0;  // 已找到的邻居数

        // 特殊情况处理
        if (ntotal == 0) {
            for (idx_t j = 0; j < k; j++) {
                I[j] = -1;
                D[j] = HUGE_VAL;
            }
            continue;
        }

        if (xb[perm[i0]] > q) {
            // 查询值小于所有值
            i1 = 0;
            goto finish_right;
        }

        if (xb[perm[i1 - 1]] <= q) {
            // 查询值大于等于所有值
            i0 = i1 - 1;
            goto finish_left;
        }

        // 二分查找
        while (i0 + 1 < i1) {
            idx_t imed = (i0 + i1) / 2;
            if (xb[perm[imed]] <= q) {
                i0 = imed;
            } else {
                i1 = imed;
            }
        }

        // 现在：xb[perm[i0]] <= q < xb[perm[i1]]
        // 双向扩展查找最近的k个值
        while (wp < k) {
            float xleft = xb[perm[i0]];
            float xright = xb[perm[i1]];

            if (q - xleft < xright - q) {
                // 左边更近
                D[wp] = q - xleft;
                I[wp] = perm[i0];
                i0--;
                wp++;
                if (i0 < 0) {
                    goto finish_right;
                }
            } else {
                // 右边更近
                D[wp] = xright - q;
                I[wp] = perm[i1];
                i1++;
                wp++;
                if (i1 >= ntotal) {
                    goto finish_left;
                }
            }
        }
        goto done;

    finish_right:
        // 只向右扩展
        while (wp < k) {
            if (i1 < ntotal) {
                D[wp] = xb[perm[i1]] - q;
                I[wp] = perm[i1];
                i1++;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
        goto done;

    finish_left:
        // 只向左扩展
        while (wp < k) {
            if (i0 >= 0) {
                D[wp] = q - xb[perm[i0]];
                I[wp] = perm[i0];
                i0--;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
    done:;
    }
}
```

**时间复杂度**：
- 预处理：O(n log n) 排序
- 每次查询：O(k + log n)

**适用场景**：
- 时间戳查询
- 年龄、分数等标量值搜索
- 作为复合索引的一部分

---

## 6. 范围搜索 (Range Search)

### 6.1 基本概念

范围搜索返回所有距离小于给定半径的向量，而不是固定top-k。

```cpp
// faiss/impl/AuxIndexStructures.h
struct RangeSearchResult {
    idx_t nq;              // 查询数
    idx_t* lims;           // 每个查询的结果数前缀和
    idx_t* labels;         // 所有结果的标签
    float* distances;      // 所有结果的距离

    // lims[i+1] - lims[i] = 第i个查询的结果数
    // labels[lims[i]:lims[i+1]] = 第i个查询的标签
    // distances[lims[i]:lims[i+1]] = 第i个查询的距离
};
```

### 6.2 范围搜索实现

```cpp
// faiss/IndexFlat.cpp
void IndexFlat::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {

    IDSelector* sel = params ? params->sel : nullptr;

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product(
                x, get_xb(), d, n, ntotal, radius, result, sel);
            break;

        case METRIC_L2:
            range_search_L2sqr(
                x, get_xb(), d, n, ntotal, radius, result, sel);
            break;

        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

// faiss/utils/distances.cpp
void range_search_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t n,
        size_t nb,
        float radius,
        RangeSearchResult* result,
        const IDSelector* sel)
{
    // 第一遍：计算每个查询的结果数
    std::vector<size_t> nresult(n, 0);

#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        const float* xi = x + i * d;
        size_t count = 0;

        for (size_t j = 0; j < nb; j++) {
            if (sel && !sel->is_member(j)) {
                continue;
            }

            float dis = fvec_L2sqr(xi, y + j * d, d);
            if (dis < radius) {
                count++;
            }
        }

        nresult[i] = count;
    }

    // 计算前缀和
    result->lims[0] = 0;
    for (idx_t i = 0; i < n; i++) {
        result->lims[i + 1] = result->lims[i] + nresult[i];
    }

    // 分配内存
    size_t total = result->lims[n];
    result->labels = new idx_t[total];
    result->distances = new float[total];

    // 第二遍：填充结果
#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        const float* xi = x + i * d;
        idx_t* labels_i = result->labels + result->lims[i];
        float* distances_i = result->distances + result->lims[i];
        size_t count = 0;

        for (size_t j = 0; j < nb; j++) {
            if (sel && !sel->is_member(j)) {
                continue;
            }

            float dis = fvec_L2sqr(xi, y + j * d, d);
            if (dis < radius) {
                labels_i[count] = j;
                distances_i[count] = dis;
                count++;
            }
        }
    }
}
```

### 6.3 使用示例

```cpp
void range_search_example() {
    int d = 128;
    int nb = 100000;  // 数据库向量数
    int nq = 1000;    // 查询向量数

    // 创建索引
    IndexFlatL2 index(d);
    index.add(nb, xb);

    // 范围搜索
    float radius = 100.0f;  // 半径
    RangeSearchResult result(nq);

    index.range_search(nq, xq, radius, &result);

    // 处理结果
    for (idx_t i = 0; i < nq; i++) {
        idx_t start = result.lims[i];
        idx_t end = result.lims[i + 1];
        idx_t count = end - start;

        printf("Query %zd: found %zd results\n", i, count);

        for (idx_t j = start; j < end; j++) {
            printf("  id=%lld, distance=%f\n",
                   result.labels[j], result.distances[j]);
        }
    }
}
```

---

## 7. DistanceComputer接口（底层实现）

### 7.1 FlatCodesDistanceComputer基类

```cpp
// faiss/impl/AuxIndexStructures.h
// Flat编码向量的距离计算器基类

struct FlatCodesDistanceComputer : DistanceComputer {
    const uint8_t* codes;     // 编码的向量数据
    size_t code_size;          // 每个向量的字节数
    size_t d;                  // 维度
    uint64_t ndis = 0;         // 距离计算计数器（性能统计）

    FlatCodesDistanceComputer(
            const uint8_t* codes,
            size_t code_size,
            size_t d)
        : codes(codes), code_size(code_size), d(d) {}

    // 默认的批量距离计算：逐个调用operator()
    void distances_batch_4(
            const idx_t idx0, const idx_t idx1,
            const idx_t idx2, const idx_t idx3,
            float& dis0, float& dis1, float& dis2, float& dis3) override {
        dis0 = this->operator()(idx0);
        dis1 = this->operator()(idx1);
        dis2 = this->operator()(idx2);
        dis3 = this->operator()(idx3);
    }

    // 获取向量指针
    inline const float* get_vector(idx_t i) const {
        return reinterpret_cast<const float*>(codes + i * code_size);
    }
};
```

### 7.2 L2距离计算器实现

```cpp
// faiss/IndexFlat.cpp
// 标准L2距离计算器

struct FlatL2Dis : FlatCodesDistanceComputer {
    const float* q;  // 查询向量

    FlatL2Dis(const IndexFlat& index)
        : FlatCodesDistanceComputer(
            index.codes.data(),
            index.code_size,
            index.d) {}

    void set_query(const float* x) override {
        q = x;
    }

    float operator()(idx_t i) override {
        ndis++;
        const float* y = get_vector(i);
        return fvec_L2sqr(q, y, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const float* x = get_vector(i);
        const float* y = get_vector(j);
        return fvec_L2sqr(x, y, d);
    }

    // 优化的批量计算：展开循环
    void distances_batch_4(
            const idx_t idx0, const idx_t idx1,
            const idx_t idx2, const idx_t idx3,
            float& dis0, float& dis1,
            float& dis2, float& dis3) override {
        ndis += 4;

        const float* y0 = get_vector(idx0);
        const float* y1 = get_vector(idx1);
        const float* y2 = get_vector(idx2);
        const float* y3 = get_vector(idx3);

        // 使用SIMD优化的批量计算
        fvec_L2sqr_batch_4(q, y0, y1, y2, y3, d,
                          dis0, dis1, dis2, dis3);
    }
};
```

### 7.3 内积距离计算器实现

```cpp
// faiss/IndexFlat.cpp
// 内积距离计算器

struct FlatIPDis : FlatCodesDistanceComputer {
    const float* q;

    FlatIPDis(const IndexFlat& index)
        : FlatCodesDistanceComputer(
            index.codes.data(),
            index.code_size,
            index.d) {}

    void set_query(const float* x) override {
        q = x;
    }

    float operator()(idx_t i) override {
        ndis++;
        const float* y = get_vector(i);
        return fvec_inner_product(q, y, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const float* x = get_vector(i);
        const float* y = get_vector(j);
        return fvec_inner_product(x, y, d);
    }

    void distances_batch_4(
            const idx_t idx0, const idx_t idx1,
            const idx_t idx2, const idx_t idx3,
            float& dis0, float& dis1,
            float& dis2, float& dis3) override {
        ndis += 4;

        const float* y0 = get_vector(idx0);
        const float* y1 = get_vector(idx1);
        const float* y2 = get_vector(idx2);
        const float* y3 = get_vector(idx3);

        // 使用SIMD优化的批量内积计算
        fvec_inner_product_batch_4(q, y0, y1, y2, y3, d,
                                   dis0, dis1, dis2, dis3);
    }
};
```

### 7.4 DistanceComputer工厂模式

```cpp
// faiss/IndexFlat.cpp
// 根据度量类型和缓存状态选择最优的距离计算器

FlatCodesDistanceComputer* IndexFlat::get_FlatCodesDistanceComputer() const {
    if (metric_type == METRIC_L2) {
        // L2距离：检查是否可用缓存
        if (const auto* indexL2 = dynamic_cast<const IndexFlatL2*>(this)) {
            if (!indexL2->cached_l2norms.empty()) {
                // 使用L2范数缓存优化版本
                return new FlatL2WithNormsDis(*indexL2);
            }
        }
        // 标准L2距离计算
        return new FlatL2Dis(*this);

    } else if (metric_type == METRIC_INNER_PRODUCT) {
        // 内积距离
        return new FlatIPDis(*this);

    } else {
        // 其他度量类型：使用通用处理器
        return get_extra_distance_computer(
            d, metric_type, metric_arg, ntotal, get_xb());
    }
}
```

### 7.5 使用DistanceComputer进行搜索

```cpp
// 通用的基于DistanceComputer的搜索实现
template <class C, class DistanceComputer>
void search_with_distance_computer(
        const DistanceComputer& dc,
        idx_t nb,           // 数据库向量数
        idx_t k,            // 返回结果数
        float* distances,   // 输出距离
        idx_t* labels)      // 输出标签
{
    // 初始化堆
    heap_heapify<C>(k, distances, labels);

    // 遍历所有向量
    for (idx_t i = 0; i < nb; i++) {
        float dis = dc(i);

        // 如果距离更好，替换堆顶
        if (C::cmp(dis, distances[0])) {
            heap_replace_top<C>(k, distances, labels, dis, i);
        }
    }

    // 排序堆
    heap_reorder<C>(k, distances, labels);
}

// 使用示例
void distance_computer_example() {
    IndexFlatL2 index(128);
    index.add(nb, xb);
    index.sync_l2norms();  // 启用L2范数缓存

    // 获取优化的距离计算器
    std::unique_ptr<FlatCodesDistanceComputer> dc(
        index.get_FlatCodesDistanceComputer());

    // 设置查询向量
    dc->set_query(xq);

    // 初始化结果堆
    const int k = 10;
    float distances[k];
    idx_t labels[k];
    heap_heapify<CMax<float, idx_t>>(k, distances, labels);

    // 搜索：手动调用distance computer
    for (idx_t i = 0; i < nb; i++) {
        float dis = (*dc)(i);

        if (CMax<float, idx_t>::cmp(dis, distances[0])) {
            heap_replace_top<CMax<float, idx_t>>(
                k, distances, labels, dis, i);
        }
    }

    heap_reorder<CMax<float, idx_t>>(k, distances, labels);

    // distances[0..k-1] 是最近的k个距离
    // labels[0..k-1] 是对应的ID

    printf("Distance computations: %lu\n", dc->ndis);
}
```

---

## 8. 批量处理优化

```cpp
void distance_computer_example() {
    IndexFlatL2 index(128);
    index.add(nb, xb);
    index.sync_l2norms();  // 启用缓存

    // 获取距离计算器
    std::unique_ptr<DistanceComputer> dc(
        index.get_FlatCodesDistanceComputer());

    // 设置查询
    dc->set_query(xq);

    // 计算单个距离
    float dis0 = (*dc)(0);  // 第0个向量的距离

    // 批量计算4个距离（SIMD优化）
    float dis0, dis1, dis2, dis3;
    dc->distances_batch_4(0, 1, 2, 3, dis0, dis1, dis2, dis3);
}
```

---

## 8. 批量处理优化

### 8.1 批量内积计算

```cpp
// faiss/utils/distances.h
void fvec_inner_product_batch_4(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        size_t d,
        float& ip0,
        float& ip1,
        float& ip2,
        float& ip3)
{
    // SIMD优化：同时计算4个内积
    // 利用AVX2或AVX512指令集
    float ip0 = 0, ip1 = 0, ip2 = 0, ip3 = 0;

    for (size_t i = 0; i < d; i++) {
        float xi = x[i];
        ip0 += xi * y0[i];
        ip1 += xi * y1[i];
        ip2 += xi * y2[i];
        ip3 += xi * y3[i];
    }
}
```

### 8.2 批量L2距离计算

```cpp
// faiss/utils/distances.h
void fvec_L2sqr_batch_4(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3)
{
    float dis0 = 0, dis1 = 0, dis2 = 0, dis3 = 0;

    for (size_t i = 0; i < d; i++) {
        float xi = x[i];
        float d0 = xi - y0[i];
        float d1 = xi - y1[i];
        float d2 = xi - y2[i];
        float d3 = xi - y3[i];

        dis0 += d0 * d0;
        dis1 += d1 * d1;
        dis2 += d2 * d2;
        dis3 += d3 * d3;
    }
}
```

### 9.1 性能瓶颈

| 操作 | 复杂度 | 瓶颈 |
|------|--------|------|
| 添加向量 | O(n * d) | 内存分配 |
| KNN搜索 | O(nq * nb * d) | 距离计算 |
| 范围搜索 | O(nq * nb * d) | 距离计算 |

### 9.2 优化技巧

1. **L2Norm缓存**：预计算数据库向量的L2范数
2. **SIMD优化**：使用AVX2/AVX512批量计算
3. **多线程**：OpenMP并行化查询
4. **批量处理**：一次处理多个查询
5. **内存对齐**：使用AlignedTable确保SIMD友好的内存布局

### 9.3 性能测试

```cpp
void benchmark_flat_index() {
    int d = 128;
    int nb = 1000000;
    int nq = 1000;

    IndexFlatL2 index(d);

    // 添加
    auto t0 = gettime();
    index.add(nb, xb);
    printf("Add time: %.3f s\n", gettime() - t0);

    // 无缓存搜索
    t0 = gettime();
    index.search(nq, xq, 100, distances, labels);
    printf("Search (no cache): %.3f s\n", gettime() - t0);

    // 启用缓存搜索
    index.sync_l2norms();
    t0 = gettime();
    index.search(nq, xq, 100, distances, labels);
    printf("Search (with cache): %.3f s\n", gettime() - t0);
}
```

---

## 10. 源码深度实现 - IndexFlat完整架构

### 10.1 IndexFlat完整结构

```cpp
// faiss/IndexFlat.h - 完整定义
namespace faiss {

// IndexFlat: 存储完整向量，执行精确搜索
struct IndexFlat : IndexFlatCodes {
    explicit IndexFlat(idx_t d, MetricType metric = METRIC_L2);

    // 核心搜索接口
    void search(
        idx_t n,              // 查询向量数
        const float* x,       // 查询向量 (n * d)
        idx_t k,              // 返回top-k
        float* distances,     // 输出距离 (n * k)
        idx_t* labels,        // 输出标签 (n * k)
        const SearchParameters* params = nullptr) const override;

    // 范围搜索
    void range_search(
        idx_t n, const float* x, float radius,
        RangeSearchResult* result,
        const SearchParameters* params = nullptr) const override;

    // 重建向量
    void reconstruct(idx_t key, float* recons) const override;

    // 计算子集距离
    void compute_distance_subset(
        idx_t n, const float* x, idx_t k,
        float* distances, const idx_t* labels) const;

    // 获取原始向量指针
    float* get_xb() { return (float*)codes.data(); }
    const float* get_xb() const { return (const float*)codes.data(); }

    // 获取DistanceComputer
    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    // 独立编解码（Flat索引只是memcpy）
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

// 内积索引
struct IndexFlatIP : IndexFlat {
    explicit IndexFlatIP(idx_t d) : IndexFlat(d, METRIC_INNER_PRODUCT) {}
};

// L2距离索引（带范数缓存）
struct IndexFlatL2 : IndexFlat {
    // L2范数缓存
    std::vector<float> cached_l2norms;

    explicit IndexFlatL2(idx_t d) : IndexFlat(d, METRIC_L2) {}

    // 计算并缓存L2范数
    void sync_l2norms();
    void clear_l2norms();

    // 获取优化的DistanceComputer
    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;
};

} // namespace faiss
```

### 10.2 IndexFlatCodes基类

```cpp
// faiss/IndexFlatCodes.h
// 存储编码向量的基类

struct IndexFlatCodes : Index {
    std::vector<uint8_t> codes;  // 存储所有编码向量
    size_t code_size;             // 每个向量的字节数

    IndexFlatCodes(size_t code_size, idx_t d, MetricType metric)
        : code_size(code_size) {
        this->d = d;
        this->metric_type = metric;
    }

    // 添加向量：直接追加到codes
    void add(idx_t n, const float* x) override {
        codes.resize((ntotal + n) * code_size);
        memcpy(codes.data() + ntotal * code_size, x, n * code_size);
        ntotal += n;
    }

    // 重建向量：从codes复制
    void reconstruct(idx_t key, float* recons) const override {
        memcpy(recons, codes.data() + key * code_size, code_size);
    }

    // 获取向量指针
    const uint8_t* get_codes() const { return codes.data(); }
};
```

### 10.3 KNN搜索实现

```cpp
// faiss/IndexFlat.cpp
// IndexFlat的search实现

void IndexFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    IDSelector* sel = params ? params->sel : nullptr;
    FAISS_THROW_IF_NOT(k > 0);

    // 根据度量类型选择搜索算法
    if (metric_type == METRIC_INNER_PRODUCT) {
        // 内积：使用最小堆（因为我们要最大的内积）
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_inner_product(x, get_xb(), d, n, ntotal, &res, sel);

    } else if (metric_type == METRIC_L2) {
        // L2距离：使用最大堆（因为我们要最小的距离）
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_L2sqr(x, get_xb(), d, n, ntotal, &res, nullptr, sel);

    } else {
        // 其他度量：通用处理
        FAISS_THROW_IF_NOT(!sel); // TODO: 实现selector支持
        knn_extra_metrics(
            x, get_xb(), d, n, ntotal,
            metric_type, metric_arg, k,
            distances, labels);
    }
}

// faiss/utils/distances.cpp
// KNN L2距离搜索实现

void knn_L2sqr(
        const float* x,      // 查询向量 (n * d)
        const float* y,      // 数据库向量 (nb * d)
        size_t d,            // 维度
        size_t n,            // 查询数
        size_t nb,           // 数据库向量数
        float_maxheap_array_t* res,  // 结果堆
        const float* y_norms,        // 预计算的y的L2范数
        const IDSelector* sel)       // ID选择器
{
    // res->val 存储距离 (n * k)
    // res->ids 存储标签 (n * k)

#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        const float* xi = x + i * d;       // 第i个查询向量
        float* __restrict simi = res->val + i * res->k;  // 距离数组
        idx_t* __restrict idxi = res->ids + i * res->k;  // ID数组

        // 初始化堆（填充为无穷大）
        heap_heapify<CMax<float, idx_t>>(res->k, simi, idxi);

        // 遍历所有数据库向量
        for (size_t j = 0; j < nb; j++) {
            // 检查selector
            if (sel && !sel->is_member(j)) {
                continue;
            }

            // 计算距离
            float dis = 0;
            if (y_norms) {
                // 使用预计算的norm优化
                // ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x|y>
                float ip = fvec_inner_product(xi, y + j * d, d);
                float norm_x = fvec_norm_L2sqr(xi, d);
                dis = norm_x + y_norms[j] - 2 * ip;
            } else {
                // 直接计算L2距离
                dis = fvec_L2sqr(xi, y + j * d, d);
            }

            // 如果距离小于堆顶，替换堆顶
            if (CMax<float, idx_t>::cmp(dis, simi[0])) {
                heap_replace_top<CMax<float, idx_t>>(
                    res->k, simi, idxi, dis, j);
            }
        }

        // 堆排序：从小到大
        heap_reorder<CMax<float, idx_t>>(res->k, simi, idxi);
    }
}
```

### 10.4 DistanceComputer完整实现

```cpp
// faiss/IndexFlat.cpp
// FlatL2距离计算器

struct FlatL2Dis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* b;
    size_t ndis;           // 距离计算计数器

    float distance_to_code(const uint8_t* code) final {
        ndis++;
        return fvec_L2sqr(q, (float*)code, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_L2sqr(b + j * d, b + i * d, d);
    }

    explicit FlatL2Dis(const IndexFlat& storage, const float* q = nullptr)
        : FlatCodesDistanceComputer(
            storage.codes.data(),
            storage.code_size,
            q),
          d(storage.d),
          nb(storage.ntotal),
          b(storage.get_xb()),
          ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }

    // SIMD优化的批量计算（关键优化）
    void distances_batch_4(
        const idx_t idx0,
        const idx_t idx1,
        const idx_t idx2,
        const idx_t idx3,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) final override {
        ndis += 4;

        const float* __restrict y0 =
            reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
            reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
            reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
            reinterpret_cast<const float*>(codes + idx3 * code_size);

        // 使用SIMD优化的批量计算
        fvec_L2sqr_batch_4(q, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
    }
};

// FlatIP距离计算器
struct FlatIPDis : FlatCodesDistanceComputer {
    size_t d;
    const float* q;
    const float* b;
    size_t ndis;

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_inner_product(q, (const float*)code, d);
    }

    float symmetric_dis(idx_t i, idx_t j) final override {
        return fvec_inner_product(b + j * d, b + i * d, d);
    }

    explicit FlatIPDis(const IndexFlat& storage, const float* q = nullptr)
        : FlatCodesDistanceComputer(
            storage.codes.data(),
            storage.code_size),
          d(storage.d),
          q(q),
          b(storage.get_xb()),
          ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }

    void distances_batch_4(
        const idx_t idx0, const idx_t idx1,
        const idx_t idx2, const idx_t idx3,
        float& dis0, float& dis1, float& dis2, float& dis3) final override {
        ndis += 4;

        const float* __restrict y0 =
            reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
            reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
            reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
            reinterpret_cast<const float*>(codes + idx3 * code_size);

        fvec_inner_product_batch_4(q, y0, y1, y2, y3, d,
                                 dis0, dis1, dis2, dis3);
    }
};
```

### 10.5 L2范数缓存优化

```cpp
// faiss/IndexFlat.cpp
// 带L2范数缓存的距离计算器

struct FlatL2WithNormsDis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    const float* l2norms;    // 缓存的L2范数
    float query_l2norm;      // 查询向量的L2范数

    float operator()(const idx_t i) final override {
        const float* __restrict y =
            reinterpret_cast<const float*>(codes + i * code_size);

        // 预取L2范数到L2缓存
        prefetch_L2(l2norms + i);

        // 使用内积计算L2距离
        // ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x|y>
        const float dp0 = fvec_inner_product(q, y, d);
        return query_l2norm + l2norms[i] - 2 * dp0;
    }

    float symmetric_dis(idx_t i, idx_t j) final override {
        const float* __restrict yi =
            reinterpret_cast<const float*>(codes + i * code_size);
        const float* __restrict yj =
            reinterpret_cast<const float*>(codes + j * code_size);

        prefetch_L2(l2norms + i);
        prefetch_L2(l2norms + j);

        const float dp0 = fvec_inner_product(yi, yj, d);
        return l2norms[i] + l2norms[j] - 2 * dp0;
    }

    explicit FlatL2WithNormsDis(
        const IndexFlatL2& storage,
        const float* q = nullptr)
        : FlatCodesDistanceComputer(
            storage.codes.data(),
            storage.code_size),
          d(storage.d),
          nb(storage.ntotal),
          q(q),
          b(storage.get_xb()),
          ndis(0),
          l2norms(storage.cached_l2norms.data()),
          query_l2norm(0) {}

    void set_query(const float* x) override {
        q = x;
        query_l2norm = fvec_norm_L2sqr(q, d);
    }

    // 带缓存的批量距离计算
    void distances_batch_4(
        const idx_t idx0,
        const idx_t idx1,
        const idx_t idx2,
        const idx_t idx3,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) final override {
        ndis += 4;

        const float* __restrict y0 =
            reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
            reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
            reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
            reinterpret_cast<const float*>(codes + idx3 * code_size);

        // 预取L2范数
        prefetch_L2(l2norms + idx0);
        prefetch_L2(l2norms + idx1);
        prefetch_L2(l2norms + idx2);
        prefetch_L2(l2norms + idx3);

        // 使用内积计算
        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        fvec_inner_product_batch_4(q, y0, y1, y2, y3, d,
                                 dp0, dp1, dp2, dp3);

        // 转换为L2距离
        dis0 = query_l2norm + l2norms[idx0] - 2 * dp0;
        dis1 = query_l2norm + l2norms[idx1] - 2 * dp1;
        dis2 = query_l2norm + l2norms[idx2] - 2 * dp2;
        dis3 = query_l2norm + l2norms[idx3] - 2 * dp3;
    }
};

// 同步L2范数
void IndexFlatL2::sync_l2norms() {
    cached_l2norms.resize(ntotal);
    fvec_norms_L2sqr(
        cached_l2norms.data(),
        reinterpret_cast<const float*>(codes.data()),
        d,
        ntotal);
}

// 清除L2范数缓存
void IndexFlatL2::clear_l2norms() {
    cached_l2norms.clear();
    cached_l2norms.shrink_to_fit();
}
```

### 10.6 IndexFlat1D实现

```cpp
// faiss/IndexFlat.cpp
// 一维向量的优化实现（利用排序）

void IndexFlat1D::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    const float* xb = get_xb();

#pragma omp parallel for if (n > 10000)
    for (idx_t i = 0; i < n; i++) {
        float q = x[i];  // 查询值（1D）
        float* D = distances + i * k;
        idx_t* I = labels + i * k;

        // 二分查找找到查询值的位置
        idx_t i0 = 0, i1 = ntotal;
        idx_t wp = 0;  // 已找到的邻居数

        // 特殊情况处理
        if (ntotal == 0) {
            for (idx_t j = 0; j < k; j++) {
                I[j] = -1;
                D[j] = HUGE_VAL;
            }
            continue;
        }

        if (xb[perm[i0]] > q) {
            // 查询值小于所有值
            i1 = 0;
            goto finish_right;
        }

        if (xb[perm[i1 - 1]] <= q) {
            // 查询值大于等于所有值
            i0 = i1 - 1;
            goto finish_left;
        }

        // 二分查找
        while (i0 + 1 < i1) {
            idx_t imed = (i0 + i1) / 2;
            if (xb[perm[imed]] <= q) {
                i0 = imed;
            } else {
                i1 = imed;
            }
        }

        // 双向扩展查找最近的k个值
        while (wp < k) {
            float xleft = xb[perm[i0]];
            float xright = xb[perm[i1]];

            if (q - xleft < xright - q) {
                // 左边更近
                D[wp] = q - xleft;
                I[wp] = perm[i0];
                i0--;
                wp++;
                if (i0 < 0) {
                    goto finish_right;
                }
            } else {
                // 右边更近
                D[wp] = xright - q;
                I[wp] = perm[i1];
                i1++;
                wp++;
                if (i1 >= ntotal) {
                    goto finish_left;
                }
            }
        }
        goto done;

    finish_right:
        // 只向右扩展
        while (wp < k) {
            if (i1 < ntotal) {
                D[wp] = xb[perm[i1]] - q;
                I[wp] = perm[i1];
                i1++;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
        goto done;

    finish_left:
        // 只向左扩展
        while (wp < k) {
            if (i0 >= 0) {
                D[wp] = q - xb[perm[i0]];
                I[wp] = perm[i0];
                i0--;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
    done:;
    }
}
```

### 10.7 生产级使用示例

```cpp
// 完整的生产环境使用示例

class FlatIndexService {
    IndexFlatL2* index;
    std::mutex index_mutex;

public:
    FlatIndexService(int d) {
        index = new IndexFlatL2(d);
        index->verbose = true;
    }

    // 添加向量
    void add_vectors(const std::vector<std::vector<float>>& vectors) {
        std::lock_guard<std::mutex> lock(index_mutex);

        size_t n = vectors.size();
        float* xb = new float[n * index->d];

        for (size_t i = 0; i < n; i++) {
            memcpy(xb + i * index->d, vectors[i].data(),
                   index->d * sizeof(float));
        }

        index->add(n, xb);
        delete[] xb;

        // 启用L2范数缓存以加速搜索
        index->sync_l2norms();
    }

    // 搜索（返回top-k最近邻）
    std::vector<std::pair<idx_t, float>> search(
        const std::vector<float>& query, size_t k) {

        std::lock_guard<std::mutex> lock(index_mutex);

        float* distances = new float[k];
        idx_t* labels = new idx_t[k];

        index->search(1, query.data(), k, distances, labels);

        std::vector<std::pair<idx_t, float>> results;
        for (size_t i = 0; i < k; i++) {
            results.push_back({labels[i], distances[i]});
        }

        delete[] distances;
        delete[] labels;

        return results;
    }

    // 保存索引
    void save(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(index_mutex);

        FILE* f = fopen(filepath.c_str(), "wb");
        IOWriter* writer = new FileIOWriter(f);
        write_index(index, writer);
        delete writer;
    }

    // 加载索引
    void load(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(index_mutex);

        FILE* f = fopen(filepath.c_str(), "rb");
        IOReader* reader = new FileIOReader(f);
        Index* loaded = read_index(reader);
        delete reader;

        delete index;
        index = dynamic_cast<IndexFlatL2*>(loaded);
    }

    ~FlatIndexService() {
        delete index;
    }
};
```

### 10.8 性能优化总结表

| 优化技术 | 实现位置 | 加速比 | 内存开销 |
|---------|---------|--------|---------|
| L2Norm缓存 | IndexFlatL2::cached_l2norms | 1.5-3x | nb × 4字节 |
| SIMD批量计算 | fvec_L2sqr_batch_4 | 2-4x | 无 |
| 数据预取 | prefetch_L2 | 1.2-1.5x | 无 |
| 多线程 | OpenMP | 线性扩展 | 无 |
| 内存对齐 | AlignedTable | 1.1-1.3x | 对齐填充 |
| 1D排序优化 | IndexFlat1D::perm | O(log n) | nb × 8字节 |

---

## 11. Heap数据结构完整底层实现

### 11.1 Heap基础与比较器

```cpp
// faiss/utils/ordered_key_value.h
// C对象：统一处理min和max堆

template <typename T_, typename TI_>
struct CMax;

// min-heap特性：最小值在堆顶，用于查找数组的最大值
template <typename T_, typename TI_>
struct CMin {
    typedef T_ T;
    typedef TI_ TI;
    typedef CMax<T_, TI_> Crev; // 反向比较引用

    // 比较函数：a是否应该排在b前面
    inline static bool cmp(T a, T b) {
        return a < b;
    }

    // 复合比较：先比较值，相等时比较键（用于打破平局）
    inline static bool cmp2(T a1, T b1, TI a2, TI b2) {
        return (a1 < b1) || ((a1 == b1) && (a2 < b2));
    }

    // 中性元素：初始化堆的默认值
    inline static T neutral() {
        return std::numeric_limits<T>::lowest();
    }

    static const bool is_max = false;
};

// max-heap特性：最大值在堆顶，用于查找数组的最小值
template <typename T_, typename TI_>
struct CMax {
    typedef T_ T;
    typedef TI_ TI;
    typedef CMin<T_, TI_> Crev;

    inline static bool cmp(T a, T b) {
        return a > b;
    }

    inline static bool cmp2(T a1, T b1, TI a2, TI b2) {
        return (a1 > b1) || ((a1 == b1) && (a2 > b2));
    }

    inline static T neutral() {
        return std::numeric_limits<T>::max();
    }

    static const bool is_max = true;
};
```

**关键设计点**：
- **CMin用于L2距离**：查找最小的距离（最近的向量）
- **CMax用于内积**：查找最大的内积（最相似的向量）
- **cmp2处理平局**：当距离相等时，比较ID确保稳定排序
- **neutral初始化**：min-heap用最小值初始化，max-heap用最大值初始化

### 11.2 heap_push - 向堆中插入元素

```cpp
// faiss/utils/Heap.h:84-105
// 向堆中插入元素(val, id)
template <class C>
inline void heap_push(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T val,
        typename C::TI id) {
    bh_val--; /* 使用1-based索引，方便计算父子关系 */
    bh_ids--;
    size_t i = k, i_father;

    // 从底部向上调整堆
    while (i > 1) {
        i_father = i >> 1;  // i / 2

        // 如果父节点不小于新值，堆结构已满足
        if (!C::cmp2(val, bh_val[i_father], id, bh_ids[i_father])) {
            break;
        }

        // 将父节点下移
        bh_val[i] = bh_val[i_father];
        bh_ids[i] = bh_ids[i_father];
        i = i_father;
    }

    // 在正确位置插入新元素
    bh_val[i] = val;
    bh_ids[i] = id;
}
```

**算法详解**：
1. **1-based索引**：使用1-based索引简化父子关系计算
   - 父节点：i >> 1 (i / 2)
   - 左子节点：i << 1 (i * 2)
   - 右子节点：(i << 1) + 1 (i * 2 + 1)

2. **向上筛选**：从底部开始，如果新值大于父节点，则交换
3. **时间复杂度**：O(log k)

### 11.3 heap_pop - 弹出堆顶元素

```cpp
// faiss/utils/Heap.h:46-78
// 弹出堆顶元素，并保持堆性质
template <class C>
inline void heap_pop(size_t k, typename C::T* bh_val, typename C::TI* bh_ids) {
    bh_val--; /* 使用1-based索引 */
    bh_ids--;

    // 保存堆顶和最后一个元素
    typename C::T val = bh_val[k];
    typename C::TI id = bh_ids[k];
    size_t i = 1, i1, i2;

    while (1) {
        i1 = i << 1;      // 左子节点
        i2 = i1 + 1;      // 右子节点

        if (i1 > k) {
            break;  // 没有子节点
        }

        // 选择较大的子节点
        if ((i2 == k + 1) ||  // 右子节点不存在
            C::cmp2(bh_val[i1], bh_val[i2], bh_ids[i1], bh_ids[i2])) {
            // 比较左子节点
            if (C::cmp2(val, bh_val[i1], id, bh_ids[i1])) {
                break;
            }
            bh_val[i] = bh_val[i1];
            bh_ids[i] = bh_ids[i1];
            i = i1;
        } else {
            // 比较右子节点
            if (C::cmp2(val, bh_val[i2], id, bh_ids[i2])) {
                break;
            }
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            i = i2;
        }
    }

    // 将最后一个元素放到最终位置
    bh_val[i] = bh_val[k];
    bh_ids[i] = bh_ids[k];
}
```

**算法详解**：
1. **保存最后一个元素**：用堆底元素替换堆顶
2. **向下筛选**：将新堆顶向下移动，直到满足堆性质
3. **选择较大的子节点**：确保父节点始终大于/小于子节点

### 11.4 heap_replace_top - 替换堆顶（最常用）

```cpp
// faiss/utils/Heap.h:113-151
// 替换堆顶元素，并重新调整堆
template <class C>
inline void heap_replace_top(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T val,
        typename C::TI id) {
    bh_val--; /* 使用1-based索引 */
    bh_ids--;

    size_t i = 1, i1, i2;

    while (1) {
        i1 = i << 1;
        i2 = i1 + 1;

        if (i1 > k) {
            break;
        }

        // 选择较大的子节点
        if ((i2 == k + 1) ||
            C::cmp2(bh_val[i1], bh_val[i2], bh_ids[i1], bh_ids[i2])) {
            if (C::cmp2(val, bh_val[i1], id, bh_ids[i1])) {
                break;
            }
            bh_val[i] = bh_val[i1];
            bh_ids[i] = bh_ids[i1];
            i = i1;
        } else {
            if (C::cmp2(val, bh_val[i2], id, bh_ids[i2])) {
                break;
            }
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            i = i2;
        }
    }

    bh_val[i] = val;
    bh_ids[i] = id;
}
```

**使用场景**：
- **KNN搜索**：维护top-k结果，当找到更好的候选时替换堆顶
- **比heap_pop + heap_push更高效**：只需要一次向下筛选

### 11.5 heap_heapify - 初始化堆

```cpp
// faiss/utils/Heap.h:318-343
// 初始化堆（从数组构建堆）
template <class C>
inline void heap_heapify(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        const typename C::T* x = nullptr,
        const typename C::TI* ids = nullptr,
        size_t k0 = 0) {
    if (k0 > 0) {
        assert(x);
    }

    // 逐个插入前k0个元素
    if (ids) {
        for (size_t i = 0; i < k0; i++) {
            heap_push<C>(i + 1, bh_val, bh_ids, x[i], ids[i]);
        }
    } else {
        for (size_t i = 0; i < k0; i++) {
            heap_push<C>(i + 1, bh_val, bh_ids, x[i], i);
        }
    }

    // 剩余位置填充中性元素
    for (size_t i = k0; i < k; i++) {
        bh_val[i] = C::neutral();
        bh_ids[i] = -1;
    }
}
```

**使用场景**：
- KNN搜索开始时初始化结果堆
- k0 = 0：创建空堆（全部填充neutral值）
- k0 > 0：从已有数据初始化堆

### 11.6 heap_reorder - 堆排序

```cpp
// faiss/utils/Heap.h:427-457
// 将堆转换为有序数组
template <typename C>
inline size_t heap_reorder(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids) {
    size_t i, ii;

    // 逐个弹出堆顶元素，从后往前填充
    for (i = 0, ii = 0; i < k; i++) {
        typename C::T val = bh_val[0];
        typename C::TI id = bh_ids[0];

        heap_pop<C>(k - i, bh_val, bh_ids);
        bh_val[k - ii - 1] = val;
        bh_ids[k - ii - 1] = id;

        if (id != -1) {
            ii++;
        }
    }

    // 统计有效元素数量
    size_t nel = ii;

    // 将有效元素移到数组开头
    memmove(bh_val, bh_val + k - ii, ii * sizeof(*bh_val));
    memmove(bh_ids, bh_ids + k - ii, ii * sizeof(*bh_ids));

    // 填充剩余位置
    for (; ii < k; ii++) {
        bh_val[ii] = C::neutral();
        bh_ids[ii] = -1;
    }

    return nel;
}
```

**算法详解**：
1. **逐个弹出堆顶**：每次pop得到当前最大/最小元素
2. **从后往前填充**：确保最终结果有序
3. **处理无效元素**：ID为-1的元素不计入结果
4. **返回有效元素数**：实际找到的邻居数可能小于k

### 11.7 heap_addn - 批量添加元素

```cpp
// faiss/utils/Heap.h:373-394
// 向堆中添加n个元素
template <class C>
inline void heap_addn(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        const typename C::T* x,
        const typename C::TI* ids,
        size_t n) {
    size_t i;
    if (ids) {
        for (i = 0; i < n; i++) {
            // 只有当新值优于堆顶时才替换
            if (C::cmp(bh_val[0], x[i])) {
                heap_replace_top<C>(k, bh_val, bh_ids, x[i], ids[i]);
            }
        }
    } else {
        for (i = 0; i < n; i++) {
            if (C::cmp(bh_val[0], x[i])) {
                heap_replace_top<C>(k, bh_val, bh_ids, x[i], i);
            }
        }
    }
}
```

**使用场景**：
- 批量距离计算后，将结果添加到堆中
- **提前剪枝**：只有优于堆顶的值才会被考虑

### 11.8 HeapArray - 多堆管理

```cpp
// faiss/utils/Heap.h:478-499
// 管理多个堆的数组结构
template <typename C>
struct HeapArray {
    typedef typename C::TI TI;
    typedef typename C::T T;

    size_t nh;    // 堆的数量（通常等于查询数nq）
    size_t k;     // 每个堆的容量
    TI* ids;      // ID数组 (nh * k)
    T* val;       // 值数组 (nh * k)

    // 获取第key个堆的值数组
    T* get_val(size_t key) {
        return val + key * k;
    }

    // 获取第key个堆的ID数组
    TI* get_ids(size_t key) {
        return ids + key * k;
    }

    // 初始化所有堆
    void heapify();
};
```

**使用场景**：
- **批量KNN搜索**：同时处理nq个查询，每个查询维护一个堆
- **内存布局**：连续内存存储，缓存友好

### 11.9 使用示例：KNN搜索

```cpp
// 完整的KNN搜索示例
void knn_search_example(
        const float* query,  // 查询向量 (d维)
        const float* database,  // 数据库 (nb * d)
        size_t d,
        size_t nb,
        size_t k,
        float* distances,
        idx_t* labels) {

    // 使用max-heap查找最小的k个距离
    std::vector<float> heap_dis(k);
    std::vector<idx_t> heap_ids(k);

    // 初始化堆（L2距离用max-heap）
    heap_heapify<CMax<float, idx_t>>(k, heap_dis.data(), heap_ids.data());

    // 遍历数据库
    for (size_t i = 0; i < nb; i++) {
        const float* vec = database + i * d;

        // 计算L2距离
        float dis = fvec_L2sqr(query, vec, d);

        // 如果距离小于堆顶，替换
        if (dis < heap_dis[0]) {
            heap_replace_top<CMax<float, idx_t>>(
                k, heap_dis.data(), heap_ids.data(), dis, i);
        }
    }

    // 堆排序，得到有序结果
    heap_reorder<CMax<float, idx_t>>(k, heap_dis.data(), heap_ids.data());

    // 复制结果
    memcpy(distances, heap_dis.data(), k * sizeof(float));
    memcpy(labels, heap_ids.data(), k * sizeof(idx_t));
}
```

### 11.10 性能特性表

| 操作 | 时间复杂度 | 是否原地 | 使用场景 |
|------|-----------|---------|----------|
| heap_push | O(log k) | 是 | 初始构建 |
| heap_pop | O(log k) | 是 | 堆排序 |
| heap_replace_top | O(log k) | 是 | KNN搜索（最常用） |
| heap_heapify | O(k log k) | 是 | 初始化 |
| heap_reorder | O(k log k) | 是 | 最终排序 |
| heap_addn | O(n log k) | 是 | 批量添加 |

### 11.11 优化技巧

1. **1-based索引**：简化父子关系计算，避免减法
2. **内联函数**：所有操作都是inline，减少函数调用开销
3. **模板特化**：编译期生成优化的min/max-heap代码
4. **缓存友好**：值和ID分开存储，提高缓存命中率
5. **提前剪枝**：heap_addn只处理优于堆顶的元素

---

## 12. RangeSearchResult与BufferList底层实现

### 12.1 RangeSearchResult结构

```cpp
// faiss/impl/AuxIndexStructures.h:30-47
// 范围搜索结果：存储所有距离小于半径的向量

struct RangeSearchResult {
    size_t nq;          // 查询向量数量
    size_t* lims;       // 每个查询的结果数量 (nq + 1)

    idx_t* labels;      // 结果标签：labels[lims[i]:lims[i+1]]对应第i个查询
    float* distances;  // 对应距离（未排序）

    size_t buffer_size; // 结果缓冲区大小

    // 构造函数：alloc_lims控制是否预分配lims
    explicit RangeSearchResult(size_t nq, bool alloc_lims = true);

    // 当lims包含每个查询的结果数量时调用
    virtual void do_allocation();

    virtual ~RangeSearchResult();
};
```

**内存布局**：
```
查询0: lims[0]=0, lims[1]=n0
       labels[0:n0-1], distances[0:n0-1]

查询1: lims[1]=n0, lims[2]=n0+n1
       labels[n0:n0+n1-1], distances[n0:n0+n1-1]

...

查询i: lims[i]=sum(n0...ni-1), lims[i+1]=sum(n0...ni)
       labels[sum...sum+ni-1], distances[sum...sum+ni-1]
```

### 12.2 RangeSearchResult实现

```cpp
// faiss/impl/AuxIndexStructures.cpp:23-56

// 构造函数
RangeSearchResult::RangeSearchResult(size_t nq, bool alloc_lims) : nq(nq) {
    if (alloc_lims) {
        lims = new size_t[nq + 1];
        memset(lims, 0, sizeof(*lims) * (nq + 1));
    } else {
        lims = nullptr;
    }
    labels = nullptr;
    distances = nullptr;
    buffer_size = 1024 * 256;  // 默认缓冲区大小
}

// 分配内存
void RangeSearchResult::do_allocation() {
    // 仅在所有部分结果聚合后调用
    FAISS_THROW_IF_NOT(labels == nullptr && distances == nullptr);

    size_t ofs = 0;
    for (int i = 0; i < nq; i++) {
        size_t n = lims[i];
        lims[i] = ofs;      // 设置起始偏移
        ofs += n;
    }
    lims[nq] = ofs;  // 总数量

    labels = new idx_t[ofs];
    distances = new float[ofs];
}

// 析构函数
RangeSearchResult::~RangeSearchResult() {
    delete[] labels;
    delete[] distances;
    delete[] lims;
}
```

### 12.3 BufferList - 缓冲区列表

```cpp
// faiss/impl/AuxIndexStructures.h:61-86
// 固定大小缓冲区的列表，避免频繁的内存分配

struct BufferList {
    size_t buffer_size;  // 每个缓冲区的条目数

    struct Buffer {
        idx_t* ids;    // ID数组
        float* dis;   // 距离数组
    };

    std::vector<Buffer> buffers;  // 缓冲区列表
    size_t wp;                   // 写指针（当前缓冲区位置）

    explicit BufferList(size_t buffer_size);
    ~BufferList();

    // 创建新缓冲区
    void append_buffer();

    // 添加一个结果（必要时追加新缓冲区）
    void add(idx_t id, float dis);

    // 从缓冲区复制元素到目标数组
    void copy_range(size_t ofs, size_t n, idx_t* dest_ids, float* dest_dis);
};
```

**设计目的**：
- **避免频繁分配**：使用固定大小的缓冲区
- **并行支持**：支持多线程并发添加结果
- **批量复制**：最后统一复制到结果数组

### 12.4 BufferList实现

```cpp
// faiss/impl/AuxIndexStructures.cpp:62-109

// 构造函数
BufferList::BufferList(size_t buffer_size) : buffer_size(buffer_size) {
    wp = buffer_size;  // 初始状态：第一个缓冲区满
}

// 析构函数：释放所有缓冲区
BufferList::~BufferList() {
    for (int i = 0; i < buffers.size(); i++) {
        delete[] buffers[i].ids;
        delete[] buffers[i].dis;
    }
}

// 添加结果
void BufferList::add(idx_t id, float dis) {
    if (wp == buffer_size) {  // 当前缓冲区已满
        append_buffer();
    }

    Buffer& buf = buffers.back();
    buf.ids[wp] = id;
    buf.dis[wp] = dis;
    wp++;
}

// 追加新缓冲区
void BufferList::append_buffer() {
    Buffer buf = {new idx_t[buffer_size], new float[buffer_size]};
    buffers.push_back(buf);
    wp = 0;  // 重置写指针
}

// 从缓冲区复制到线性数组
void BufferList::copy_range(
        size_t ofs, size_t n,
        idx_t* dest_ids,
        float* dest_dis) {

    size_t bno = ofs / buffer_size;  // 起始缓冲区索引
    ofs -= bno * buffer_size;        // 缓冲区内偏移

    while (n > 0) {
        // 计算本次复制的数量
        size_t ncopy = ofs + n < buffer_size ? n : buffer_size - ofs;

        Buffer buf = buffers[bno];

        // 复制ID和距离
        memcpy(dest_ids, buf.ids + ofs, ncopy * sizeof(*dest_ids));
        memcpy(dest_dis, buf.dis + ofs, ncopy * sizeof(*dest_dis));

        dest_ids += ncopy;
        dest_dis += ncopy;
        ofs = 0;
        bno++;
        n -= ncopy;
    }
}
```

**内存布局图**：
```
Buffer 0: [0..buffer_size-1]
Buffer 1: [buffer_size..2*buffer_size-1]
Buffer 2: [2*buffer_size..3*buffer_size-1]
...
```

### 12.5 使用示例

```cpp
// 示例1：基本范围搜索
void example_range_search() {
    int d = 128;
    IndexFlatL2 index(d);
    index.add(10000, xb);

    // 创建范围搜索结果对象
    RangeSearchResult result(1);  // 1个查询

    // 执行范围搜索
    float radius = 100.0f;
    index.range_search(1, xq, radius, &result);

    // 访问结果
    for (size_t i = 0; i < result.lims[1]; i++) {
        idx_t id = result.labels[i];
        float dis = result.distances[i];
        printf("ID=%ld, distance=%g\n", id, dis);
    }
}

// 示例2：多线程范围搜索
void example_parallel_range_search() {
    int d = 128;
    IndexFlatL2 index(d);
    index.add(10000, xb);

    int nq = 10;
    float radius = 100.0f;

    // 多个部分结果（每个线程一个）
    std::vector<RangeSearchPartialResult*> partial_results;

#pragma omp parallel
    {
        // 每个线程创建自己的部分结果
        RangeSearchPartialResult* pres =
            new RangeSearchPartialResult(&result);

        // ... 执行范围搜索，添加结果到pres ...

#pragma omp critical
        partial_results.push_back(pres);
    }

    // 合并所有部分结果
    RangeSearchPartialResult::merge(partial_results, true);

    // 现在result包含所有线程的结果
}
```

### 12.6 性能特性表

| 组件 | 内存开销 | 时间复杂度 | 用途 |
|------|---------|-----------|------|
| RangeSearchResult | O(nresults) | O(1)访问 | 最终结果存储 |
| BufferList | nbuffer × buffer_size | O(1)添加 | 临时结果存储 |
| BufferList::append_buffer | buffer_size × 8 | O(1) | 扩展容量 |
| BufferList::copy_range | O(nresults) | O(nresults) | 批量复制 |
| RangeSearchPartialResult::merge | O(nqueries) | O(nresults) | 结果合并 |

### 12.7 优化技巧总结

1. **固定大小缓冲区**：避免频繁内存分配
2. **延迟分配**：先收集结果，统计数量后再分配
3. **并行支持**：每个线程维护独立的部分结果
4. **批量复制**：使用memcpy高效复制
5. **灵活buffer_size**：根据预期结果数调整大小

---

## 13. 第2天总结

### 关键概念回顾

1. **IndexFlat**：完整的向量存储 + 穷举搜索
2. **距离计算**：L2和内积的优化实现
3. **L2Norm缓存**：通过预计算优化L2距离
4. **IndexFlat1D**：利用排序的一维特化实现
5. **DistanceComputer**：避免虚函数开销的距离计算
6. **批量处理**：SIMD优化的批量计算
7. **生产级包装**：线程安全的索引服务

### 下一步

在第3天，我们将深入Faiss的**距离计算底层实现**，理解SIMD优化是如何工作的。

---

## 练习题

1. 实现一个简单的Flat索引，支持L2距离和内积搜索
2. 比较有/无L2Norm缓存的性能差异
3. 实现IndexFlat1D的双向扩展搜索
4. 使用DistanceComputer接口实现自定义距离度量

## 扩展阅读

- faiss/IndexFlat.h - Flat索引定义
- faiss/IndexFlat.cpp - Flat索引实现
- faiss/utils/distances.h - 距离计算函数
- faiss/utils/Heap.h - 堆数据结构
