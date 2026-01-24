# Faiss深度课程 - 第5天：IVF索引 - Inverted File架构

## 课程目标

深入理解Inverted File（IVF）索引架构，掌握其核心思想、实现细节和优化策略，这是Faiss中实现大规模向量搜索的关键技术。

---

## 1. IVF索引概述

### 1.1 核心思想

IVF（Inverted File）借鉴了信息检索中的倒排索引思想：
1. **粗量化器**：将向量空间划分为多个Voronoi单元（cell）
2. **倒排列表**：每个cell维护一个向量列表
3. **搜索时**：只搜索查询最近的nprobe个cell

```cpp
// IVF搜索流程示意
查询向量 q
    ↓
粗量化器 → 找到最近的nprobe个质心 [c1, c2, ..., cnprobe]
    ↓
访问对应的倒排列表 → 只搜索这些列表中的向量
    ↓
返回top-k结果
```

### 1.2 IVF的优势

| 特性 | Flat索引 | IVF索引 |
|------|---------|---------|
| 搜索复杂度 | O(nb) | O(nb/nlist * nprobe) |
| 精度 | 100% | 接近100%（取决于nprobe） |
| 内存 | 高 | 高（Flat子索引）|
| 速度 | 慢 | 快（10-100倍） |

### 1.3 IndexIVF类层次

```
IndexIVF (基类)
├── IndexIVFFlat (Flat子索引)
├── IndexIVFPQ (PQ子索引)
├── IndexIVFFastScan (优化版)
├── IndexIVFScalarQuantizer (标量子索引)
├── IndexIVFPQR (乘积残差量化)
└── IndexIVFAdditiveQuantizer (加性量化)
```

---

## 2. IndexIVF核心结构

### 2.1 主要成员变量

```cpp
// faiss/IndexIVF.h
struct IndexIVF : Index, IndexIVFInterface {
    // 1. 粗量化器
    Index* quantizer;      // 用于分配向量到倒排列表的索引
    size_t nlist;          // 倒排列表数量（Voronoi cell数）

    // 2. 倒排列表
    InvertedLists* invlists;  // 存储实际的向量编码
    bool own_invlists;        // 是否拥有invlists的所有权

    // 3. 编码参数
    size_t code_size;      // 每个向量的编码大小（字节）
    bool by_residual;      // 是否存储残差向量

    // 4. 搜索参数
    size_t nprobe;         // 每次查询探测的列表数
    size_t max_codes;      // 最大访问编码数

    // 5. 直接映射
    DirectMap direct_map;  // ID → (list_no, offset) 映射

    // 6. 并行模式
    int parallel_mode;     // 0: 按查询并行, 1: 按列表并行
};
```

### 2.2 Level1Quantizer

```cpp
// faiss/IndexIVF.h
struct Level1Quantizer {
    Index* quantizer;      // 粗量化器索引
    size_t nlist;          // 列表数
    bool own_fields;       // 是否拥有quantizer

    ClusteringParameters cp;  // 训练参数

    // 训练粗量化器
    void train_q1(
        size_t n,
        const float* x,
        bool verbose,
        MetricType metric_type) {

        // 在数据上运行k-means
        Clustering clus(quantizer->d, nlist);
        clus.train(n, x, cp);

        // 将质心添加到quantizer
        quantizer->add(nlist, clus.centroids);
    }

    // 计算粗量化编码大小
    size_t coarse_code_size() const {
        return (nlist <= 256) ? 1 : (nlist <= 65536) ? 2 : 4;
    }
};
```

---

## 3. InvertedLists接口

### 3.1 抽象接口

```cpp
// faiss/invlists/InvertedLists.h
struct InvertedLists {
    size_t nlist;      // 倒排列表数
    size_t code_size;  // 每个编码的字节数

    // 获取列表大小
    virtual size_t list_size(size_t list_no) const = 0;

    // 获取列表指针
    virtual const uint8_t* get_codes(size_t list_no) const = 0;
    virtual const idx_t* get_ids(size_t list_no) const = 0;

    // 添加到列表
    virtual size_t add_entries(
        size_t list_no,
        size_t n,
        const idx_t* ids,
        const uint8_t* code) = 0;

    // 更新列表
    virtual void update_entries(
        size_t list_no,
        size_t offset,
        size_t n,
        const idx_t* ids,
        const uint8_t* code) = 0;

    // 删除列表
    virtual void resize(size_t list_no, size_t new_size) = 0;
};
```

### 3.2 ArrayInvertedLists

```cpp
// 内存中的简单实现
struct ArrayInvertedLists : InvertedLists {
    // codes[nlist][n]  // 每个列表的编码数组
    std::vector<std::vector<uint8_t>> codes;
    // ids[nlist][n]     // 每个列表的ID数组
    std::vector<std::vector<idx_t>> ids;

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
            size_t n,
            const idx_t* xids,
            const uint8_t* xcode) override {
        size_t o = ids[list_no].size();
        ids[list_no].resize(o + n);
        memcpy(ids[list_no].data() + o, xids, n * sizeof(idx_t));

        codes[list_no].resize(o + n * code_size);
        memcpy(codes[list_no].data() + o * code_size,
               xcode, n * code_size);
        return o;
    }
};
```

### 3.3 BlockInvertedLists

```cpp
// 块存储实现（更好的缓存局部性）
struct BlockInvertedLists : InvertedLists {
    struct Block {
        size_t list_no;      // 所属列表
        size_t capacity;     // 容量
        size_t size;         // 当前大小
        std::vector<uint8_t> codes;
        std::vector<idx_t> ids;
    };

    std::vector<Block> blocks;
    std::vector<size_t> list_sizes;  // 每个列表的总大小

    size_t add_entries(
            size_t list_no,
            size_t n,
            const idx_t* xids,
            const uint8_t* xcode) override {

        // 找到最后一个块或创建新块
        if (blocks.empty() || blocks.back().list_no != list_no ||
            blocks.back().size + n > blocks.back().capacity) {
            blocks.push_back({list_no, 65536, 0, {}, {}});
        }

        Block& b = blocks.back();
        size_t offset = b.size;
        b.size += n;

        b.codes.resize(b.size * code_size);
        b.ids.resize(b.size);

        memcpy(b.codes.data() + offset * code_size, xcode, n * code_size);
        memcpy(b.ids.data() + offset, xids, n * sizeof(idx_t));

        list_sizes[list_no] += n;
        return offset;
    }
};
```

---

## 4. IVF训练与添加

### 4.1 训练流程

```cpp
void IndexIVF::train(idx_t n, const float* x) {
    // 1. 训练粗量化器
    train_q1(n, x, true, metric_type);

    // 2. 训练编码器（子索引）
    if (by_residual) {
        // 计算残差：r = x - centroid(x)
        float* residuals = new float[n * d];
        idx_t* assign = new idx_t[n];

        quantizer->assign(n, x, assign);

        for (idx_t i = 0; i < n; i++) {
            idx_t list_no = assign[i];
            const float* centroid = get_centroid(list_no);
            float* res = residuals + i * d;

            for (int j = 0; j < d; j++) {
                res[j] = x[i * d + j] - centroid[j];
            }
        }

        // 在残差上训练编码器
        train_encoder(n, residuals, assign);

        delete[] residuals;
        delete[] assign;
    } else {
        train_encoder(n, x, nullptr);
    }

    // 3. 初始化倒排列表
    if (!invlists) {
        invlists = new ArrayInvertedLists(nlist, code_size);
    }
}
```

### 4.2 添加向量

```cpp
void IndexIVF::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids) {

    // 1. 分配到倒排列表
    idx_t* list_nos = new idx_t[n];
    float* distances = new float[n];

    quantizer->search(n, x, 1, distances, list_nos);

    // 2. 编码向量
    uint8_t* codes = new uint8_t[n * code_size];
    encode_vectors(n, x, list_nos, codes, false);

    // 3. 添加到倒排列表
    add_core(n, x, xids, list_nos);

    delete[] list_nos;
    delete[] distances;
    delete[] codes;
}

void IndexIVF::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* precomputed_idx,
        void* inverted_list_context) {

    idx_t* list_nos = precomputed_idx;
    if (!list_nos) {
        // 如果没有预先分配，现在分配
        list_nos = new idx_t[n];
        quantizer->assign(n, x, list_nos);
    }

    uint8_t* codes = new uint8_t[n * code_size];
    encode_vectors(n, x, list_nos, codes, false);

    // 按列表分组
    std::vector<std::vector<idx_t>> to_add(nlist);
    for (idx_t i = 0; i < n; i++) {
        idx_t list_no = list_nos[i];
        if (list_no >= 0 && list_no < nlist) {
            to_add[list_no].push_back(i);
        }
    }

    // 批量添加
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t n = to_add[list_no].size();
        if (n == 0) continue;

        std::vector<idx_t> ids(n);
        std::vector<uint8_t> list_codes(n * code_size);

        for (size_t i = 0; i < n; i++) {
            idx_t idx = to_add[list_no][i];
            ids[i] = xids ? xids[idx] : ntotal + idx;
            memcpy(list_codes.data() + i * code_size,
                   codes + idx * code_size,
                   code_size);
        }

        invlists->add_entries(
            list_no, n, ids.data(), list_codes.data());
    }

    ntotal += n;
    delete[] codes;
    if (!precomputed_idx) {
        delete[] list_nos;
    }
}
```

---

## 5. IVF搜索实现

### 5.1 搜索流程

```cpp
void IndexIVF::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    // 1. 为查询分配到倒排列表
    idx_t* assign = new idx_t[n * nprobe];
    float* centroid_dis = new float[n * nprobe];

    quantizer->search(n, x, nprobe, centroid_dis, assign);

    // 2. 调用预分配版本的搜索
    search_preassigned(
        n, x, k, assign, centroid_dis,
        distances, labels, false,
        static_cast<const IVFSearchParameters*>(params));

    delete[] assign;
    delete[] centroid_dis;
}
```

### 5.2 预分配搜索

```cpp
void IndexIVF::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {

    // 初始化堆
    for (idx_t i = 0; i < n; i++) {
        float* simi = distances + i * k;
        idx_t* idxi = labels + i * k;
        heap_heapify<CMax<float, idx_t>>(k, simi, idxi);
    }

    // 并行处理查询
#pragma omp parallel for if (parallel_mode == 0)
    for (idx_t i = 0; i < n; i++) {
        float* simi = distances + i * k;
        idx_t* idxi = labels + i * k;

        // 获取扫描器
        InvertedListScanner* scanner =
            get_InvertedListScanner(store_pairs, nullptr, params);

        scanner->set_query(x + i * d);

        // 遍历nprobe个列表
        for (size_t ij = 0; ij < nprobe; ij++) {
            idx_t list_no = assign[i * nprobe + ij];
            if (list_no < 0 || list_no >= nlist) continue;

            size_t list_size = invlists->list_size(list_no);
            if (list_size == 0) continue;

            const uint8_t* codes = invlists->get_codes(list_no);
            const idx_t* ids = invlists->get_ids(list_no);

            scanner->set_list(list_no, centroid_dis[i * nprobe + ij]);

            // 扫描倒排列表
            scanner->scan_codes(
                list_size, codes, ids, simi, idxi, k);
        }

        delete scanner;

        // 排序堆
        heap_reorder<CMax<float, idx_t>>(k, simi, idxi);
    }
}
```

---

## 6. InvertedListScanner接口（底层实现）

### 6.1 InvertedListScanner完整接口

```cpp
// faiss/IndexIVF.h (完整定义)
struct InvertedListScanner {
    // 当前状态
    idx_t list_no = -1;      // 当前列表编号
    bool keep_max = false;   // 保持最大而非最小（用于内积）
    bool store_pairs;        // 存储(list_no, offset)对而非ID
    const IDSelector* sel;   // ID选择器（过滤）
    size_t code_size = 0;     // 编码大小（字节）

    // 构造函数
    InvertedListScanner(
            bool store_pairs = false,
            const IDSelector* sel = nullptr)
        : store_pairs(store_pairs), sel(sel) {}

    // 虚析构函数
    virtual ~InvertedListScanner() {}

    /** 设置查询向量
     * @param query_vector 查询向量（d维）
     */
    virtual void set_query(const float* query_vector) = 0;

    /** 设置当前处理的倒排列表
     * @param list_no     倒排列表编号
     * @param coarse_dis 查询到该列表质心的粗距离
     */
    virtual void set_list(idx_t list_no, float coarse_dis) = 0;

    /** 计算查询到单个编码的距离
     * @param code 向量编码（code_size字节）
     * @return 距离值
     */
    virtual float distance_to_code(const uint8_t* code) const = 0;

    /** 扫描一组编码，更新堆结果
     * @param n         编码数量
     * @param codes     编码数组（n * code_size）
     * @param ids       ID数组（n），如果store_pairs则忽略
     * @param distances 堆距离（k个元素）
     * @param labels    堆标签（k个元素）
     * @param k         堆大小
     * @return 堆更新次数
     */
    virtual size_t scan_codes(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            float* distances,
            idx_t* labels,
            size_t k) const;

    /** 使用迭代器扫描编码（灵活版本）
     * @param iterator  倒排列表迭代器
     * @param distances 堆距离
     * @param labels    堆标签
     * @param k         堆大小
     * @param list_size 当前列表大小（引用，可被更新）
     * @return 堆更新次数
     */
    virtual size_t iterate_codes(
            InvertedListsIterator* iterator,
            float* distances,
            idx_t* labels,
            size_t k,
            size_t& list_size) const;

    /** 范围搜索：扫描编码，收集半径内的结果
     * @param n         编码数量
     * @param codes     编码数组
     * @param ids       ID数组
     * @param radius    搜索半径
     * @param result    结果收集器
     */
    virtual void scan_codes_range(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& result) const;

    virtual void iterate_codes_range(
            InvertedListsIterator* iterator,
            float radius,
            RangeQueryResult& result,
            size_t& list_size) const;
};
```

### 6.2 scan_codes默认实现

```cpp
// faiss/IndexIVF.cpp
size_t InvertedListScanner::scan_codes(
        size_t n,
        const uint8_t* codes,
        const idx_t* ids,
        float* distances,
        idx_t* labels,
        size_t k) const {

    size_t n_updated = 0;  // 堆更新计数

    for (size_t i = 0; i < n; i++) {
        // 1. ID选择器过滤
        if (sel && !sel->is_member(ids[i])) {
            continue;
        }

        // 2. 计算距离
        float dis = distance_to_code(codes + i * code_size);

        // 3. 检查是否应该更新堆
        bool better;
        if (keep_max) {
            // 内积：找最大值，使用最小堆
            better = CMin<float, idx_t>::cmp(dis, distances[0]);
        } else {
            // L2距离：找最小值，使用最大堆
            better = CMax<float, idx_t>::cmp(dis, distances[0]);
        }

        if (better) {
            // 4. 更新堆
            if (keep_max) {
                heap_replace_top<CMin<float, idx_t>>(
                    k, distances, labels, dis,
                    store_pairs ? (list_no << 32 | i) : ids[i]);
            } else {
                heap_replace_top<CMax<float, idx_t>>(
                    k, distances, labels, dis,
                    store_pairs ? (list_no << 32 | i) : ids[i]);
            }
            n_updated++;
        }
    }

    return n_updated;
}
```

### 6.3 IVFFlatScanner实现

```cpp
// faiss/IndexIVFFlat.cpp
// IndexIVFFlat的扫描器：直接存储原始float向量

struct IVFFlatScanner : InvertedListScanner {
    const float* q;    // 查询向量
    size_t d;          // 维度

    IVFFlatScanner(size_t d, bool store_pairs, const IDSelector* sel)
        : InvertedListScanner(store_pairs, sel), d(d) {}

    void set_query(const float* query_vector) override {
        q = query_vector;
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        // 对于IVFFlat，coarse_dis可用于by_residual优化
        // 当by_residual=true时，最终距离 = coarse_dis + residual_dis
    }

    float distance_to_code(const uint8_t* code) const override {
        // code直接指向float向量
        const float* db = reinterpret_cast<const float*>(code);
        return fvec_L2sqr(q, db, d);
    }

    // 批量距离计算优化（可重载）
    size_t scan_codes(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            float* distances,
            idx_t* labels,
            size_t k) const override {

        size_t n_updated = 0;

        for (size_t i = 0; i < n; i++) {
            if (sel && !sel->is_member(ids[i])) {
                continue;
            }

            const float* db = reinterpret_cast<const float*>(
                codes + i * code_size);
            float dis = fvec_L2sqr(q, db, d);

            if (CMax<float, idx_t>::cmp(dis, distances[0])) {
                heap_replace_top<CMax<float, idx_t>>(
                    k, distances, labels, dis, ids[i]);
                n_updated++;
            }
        }

        return n_updated;
    }
};
```

### 6.4 IVFPQScanner实现

```cpp
// faiss/IndexIVFPQ.cpp
// IndexIVFPQ的扫描器：使用查找表优化

struct IVFPQScanner : InvertedListScanner {
    const ProductQuantizer& pq;   // PQ量化器
    size_t d;                      // 维度
    size_t M;                      // 子量化器数
    size_t ksub;                   // 每个子量化器的质心数

    // 查找表：dis_tables[m * ksub + k] = 查询到子量化器m的第k个质心的距离
    AlignedTable<float> dis_tables;

    IVFPQScanner(
            const ProductQuantizer& pq,
            size_t d,
            bool store_pairs,
            const IDSelector* sel)
        : InvertedListScanner(store_pairs, sel),
          pq(pq), d(d), M(pq.M), ksub(pq.ksub),
          dis_tables(pq.M * pq.ksub) {}

    void set_query(const float* query_vector) override {
        // 预计算查找表
        pq.compute_L2_distance_table(
            query_vector,
            nullptr,  // y_norms（不需要）
            dis_tables.get());
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        // coarse_dis可用于累积残差距离
    }

    float distance_to_code(const uint8_t* code) const override {
        // 使用查找表快速计算距离
        float dis = 0;

        for (size_t m = 0; m < M; m++) {
            // 提取子量化器m的编码
            uint64_t code_m = decode_uint64(code, m * pq.nbits, pq.nbits);
            // 查找表距离
            dis += dis_tables[m * ksub + code_m];
        }

        return dis;
    }

    // SIMD优化的扫描实现
    size_t scan_codes(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            float* distances,
            idx_t* labels,
            size_t k) const override {

        size_t n_updated = 0;

        // 批量处理：一次处理4个编码
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            if (sel) {
                // 批量检查ID选择器
                if (!sel->is_member(ids[i + 0]) ||
                    !sel->is_member(ids[i + 1]) ||
                    !sel->is_member(ids[i + 2]) ||
                    !sel->is_member(ids[i + 3])) {
                    // 回退到逐个处理
                    break;
                }
            }

            // SIMD优化的距离计算
            float dis[4];
            pq.compute_codes_dis_tables(
                codes + i * code_size,
                dis_tables.get(),
                dis);

            // 更新堆（4次）
            for (int j = 0; j < 4; j++) {
                if (CMax<float, idx_t>::cmp(dis[j], distances[0])) {
                    heap_replace_top<CMax<float, idx_t>>(
                        k, distances, labels, dis[j], ids[i + j]);
                    n_updated++;
                }
            }
        }

        // 处理剩余编码
        for (; i < n; i++) {
            if (sel && !sel->is_member(ids[i])) {
                continue;
            }

            float dis = distance_to_code(codes + i * code_size);

            if (CMax<float, idx_t>::cmp(dis, distances[0])) {
                heap_replace_top<CMax<float, idx_t>>(
                    k, distances, labels, dis, ids[i]);
                n_updated++;
            }
        }

        return n_updated;
    }
};
```

---

## 7. IndexIVFFlat详解

### 7.1 结构与编码

```cpp
// faiss/IndexIVFFlat.h
struct IndexIVFFlat : IndexIVF {
    IndexIVFFlat(
        Index* quantizer,  // 通常为IndexFlatL2
        size_t d,
        size_t nlist,
        MetricType metric = METRIC_L2)
        : IndexIVF(quantizer, d, nlist, sizeof(float) * d, metric) {}

    // 编码：直接存储原始向量
    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos) const override {

        if (!by_residual) {
            // 直接存储原始向量
            memcpy(codes, x, n * d * sizeof(float));
        } else {
            // 存储残差向量
            for (idx_t i = 0; i < n; i++) {
                const float* xi = x + i * d;
                float* code_i = reinterpret_cast<float*>(
                    codes + i * code_size);

                if (list_nos[i] >= 0) {
                    const float* centroid =
                        quantizer->get_xb() + list_nos[i] * d;

                    for (int j = 0; j < d; j++) {
                        code_i[j] = xi[j] - centroid[j];
                    }
                } else {
                    memcpy(code_i, xi, d * sizeof(float));
                }
            }
        }
    }

    // 解码
    void decode_vectors(
            idx_t n,
            const uint8_t* codes,
            const idx_t* list_nos,
            float* x) const override {

        for (idx_t i = 0; i < n; i++) {
            const float* code_i =
                reinterpret_cast<const float*>(codes + i * code_size);
            float* xi = x + i * d;

            if (by_residual && list_nos[i] >= 0) {
                const float* centroid =
                    quantizer->get_xb() + list_nos[i] * d;

                for (int j = 0; j < d; j++) {
                    xi[j] = centroid[j] + code_i[j];
                }
            } else {
                memcpy(xi, code_i, d * sizeof(float));
            }
        }
    }
};
```

### 7.2 使用示例

```cpp
void ivfflat_example() {
    int d = 128;
    int nlist = 100;     // 倒排列表数
    int n = 100000;      // 数据库向量数

    // 1. 创建粗量化器
    IndexFlatL2 quantizer(d);

    // 2. 创建IVFFlat索引
    IndexIVFFlat index(&quantizer, d, nlist, METRIC_L2);

    // 3. 训练（学习质心）
    index.train(n, xb);

    // 4. 添加向量
    index.add(n, xb);

    // 5. 搜索
    int nq = 100;
    int k = 10;
    float* distances = new float[nq * k];
    idx_t* labels = new idx_t[nq * k];

    // 设置nprobe
    IVFSearchParameters params;
    params.nprobe = 10;  // 探测10个最近列表

    index.search(nq, xq, k, distances, labels, &params);
}
```

---

## 8. 性能优化

### 8.1 nprobe参数调优

```cpp
// nprobe vs 性能/精度权衡
void benchmark_nprobe() {
    std::vector<int> nprobes = {1, 5, 10, 20, 50, 100};
    std::vector<float> recalls;
    std::vector<double> times;

    for (int nprobe : nprobes) {
        index.nprobe = nprobe;

        auto t0 = std::chrono::high_resolution_clock::now();
        index.search(nq, xq, k, distances, labels);
        auto t1 = std::chrono::high_resolution_clock::now();

        double time_ms =
            std::chrono::duration<double>(t1 - t0).count() * 1000;

        float recall = compute_recall(nq, k, labels, ground_truth);

        recalls.push_back(recall);
        times.push_back(time_ms);
    }

    // 绘制曲线：nprobe vs recall, time
}
```

### 8.2 并行模式

```cpp
// parallel_mode设置
void set_parallel_mode(IndexIVF& index, int mode) {
    /*
    0 (默认): 按查询并行
       - 每个线程处理一个查询
       - 适合查询数多的情况

    1: 按倒排列表并行
       - 每个线程处理一个列表
       - 适合查询数少但列表大的情况

    2: 混合并行
       - 同时按查询和列表并行
       - 最大化资源利用

    3: 细粒度查询并行
       - 查询内部也并行
    */
    index.parallel_mode = mode;
}
```

### 8.3 DirectMap加速重建

```cpp
// 启用direct_map
void enable_direct_map(IndexIVF& index) {
    /*
    DirectMap维护 ID → (list_no, offset) 的映射

    Type 0: 无映射（最快，不能reconstruct）
    Type 1: 弱映射（数组）
    Type 2: 强映射（哈希表）
    */
    index.make_direct_map(true);
    index.set_direct_map_type(DirectMap::Type::WeakMap);

    // 现在可以快速重建向量
    float recons[128];
    index.reconstruct(vector_id, recons);
}
```

---

## 9. 第5天总结

### 关键概念

1. **IVF架构**：粗量化 + 倒排列表
2. **粗量化器**：将空间划分为Voronoi cell
3. **倒排列表**：每个cell维护向量列表
4. **nprobe**：搜索时探测的cell数
5. **by_residual**：存储残差而非原始向量
6. **InvertedListScanner**：灵活的距离计算接口

### 性能权衡

| 参数 | 增加影响 | 推荐值 |
|------|---------|--------|
| nlist | 内存↑，精度↓，速度↑ | sqrt(nb) |
| nprobe | 精度↑，速度↓ | 10-100 |
| by_residual | 精度↑，速度↓ | 通常启用 |

### 下一步

第6天将学习**加性量化器**（ResidualQuantizer、AdditiveQuantizer），这是比PQ更强大的量化方法。

---

## 10. IVF源码深度实现

本部分深入剖析Faiss中IVF索引的核心实现细节，展示实际生产代码中的优化技巧。

### 10.1 IndexIVF完整搜索实现

```cpp
// faiss/IndexIVF.cpp
// 完整的search_preassigned实现

void IndexIVF::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {

    // 1. 解析搜索参数
    size_t nprobe = params ? params->nprobe : this->nprobe;
    size_t max_codes = params ? params->max_codes : this->max_codes;
    const IDSelector* sel = params ? params->sel : nullptr;
    void* inverted_list_context = params ? params->inverted_list_context : nullptr;

    // 2. 初始化统计信息
    IndexIVFStats local_stats;
    if (!stats) {
        stats = &local_stats;
    }
    stats->nq += n;

    // 3. 初始化结果堆（每个查询一个k大小堆）
    float_minheap_array_t res = {
        size_t(n),              // nh: 查询数
        size_t(k),               // k: 堆大小
        labels,                 // ids: 结果标签
        distances              // val: 结果距离
    };

    // 初始化所有堆为空
    if (parallel_mode & 1024) { // PARALLEL_MODE_NO_HEAP_INIT
        // 不初始化堆（调用者负责）
    } else {
        res.heapify();
    }

    // 4. 并行模式选择
    if (parallel_mode == 0) {
        // 模式0：按查询并行（默认，适合查询数多的情况）
        #pragma omp parallel for if (n > 100)
        for (idx_t i = 0; i < n; i++) {
            std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, sel, params));

            scanner->set_query(x + i * d);

            size_t n_scan = 0;  // 已扫描的编码数

            // 遍历nprobe个倒排列表
            for (size_t ij = 0; ij < nprobe; ij++) {
                idx_t list_no = assign[i * nprobe + ij];
                float coarse_dis = centroid_dis[i * nprobe + ij];

                if (list_no < 0 || list_no >= nlist) {
                    continue;  // 无效列表
                }

                size_t list_size = invlists->list_size(list_no);
                if (list_size == 0) {
                    continue;  // 空列表
                }

                // 获取倒排列表数据
                InvertedLists::ScopedCodes codes(invlists, list_no);
                InvertedLists::ScopedIds ids(invlists, list_no);

                scanner->set_list(list_no, coarse_dis);

                // 扫描列表
                size_t n_updated = scanner->scan_codes(
                    list_size, codes.get(), ids.get(),
                    res.get_val(i), res.get_ids(i), k);

                n_scan += list_size;
                stats->ndis += list_size;

                // max_codes优化：限制扫描数量
                if (max_codes > 0 && n_scan >= max_codes) {
                    break;
                }
            }

            stats->nlist += 1;
        }

    } else if (parallel_mode == 1) {
        // 模式1：按倒排列表并行（适合查询少但列表大的情况）
        for (idx_t i = 0; i < n; i++) {
            std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, sel, params));

            scanner->set_query(x + i * d);

            size_t n_scan = 0;

            // 对nprobe个列表并行处理
            #pragma omp parallel for reduction(+ : n_scan) schedule(dynamic)
            for (size_t ij = 0; ij < nprobe; ij++) {
                idx_t list_no = assign[i * nprobe + ij];

                if (list_no < 0 || list_no >= nlist) {
                    continue;
                }

                size_t list_size = invlists->list_size(list_no);
                if (list_size == 0) {
                    continue;
                }

                InvertedLists::ScopedCodes codes(invlists, list_no);
                InvertedLists::ScopedIds ids(invlists, list_no);

                // 注意：scanner->set_list需要在线程安全的方式调用
                // 由于每个线程处理不同的list_no，这是线程安全的

                // 临界区保护堆更新
                #pragma omp critical
                {
                    scanner->set_list(list_no, centroid_dis[i * nprobe + ij]);
                    size_t n_updated = scanner->scan_codes(
                        list_size, codes.get(), ids.get(),
                        res.get_val(i), res.get_ids(i), k);
                    n_scan += list_size;
                }
            }

            stats->nlist += 1;
            stats->ndis += n_scan;
        }

    } else if (parallel_mode == 3) {
        // 模式3：细粒度查询并行（查询内部也并行）
        #pragma omp parallel
        {
            // 每个线程获取自己的scanner
            std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, sel, params));

            #pragma omp for schedule(dynamic)
            for (idx_t i = 0; i < n; i++) {
                scanner->set_query(x + i * d);

                size_t n_scan = 0;
                for (size_t ij = 0; ij < nprobe; ij++) {
                    idx_t list_no = assign[i * nprobe + ij];

                    if (list_no < 0 || list_no >= nlist) {
                        continue;
                    }

                    size_t list_size = invlists->list_size(list_no);
                    if (list_size == 0) {
                        continue;
                    }

                    InvertedLists::ScopedCodes codes(invlists, list_no);
                    InvertedLists::ScopedIds ids(invlists, list_no);

                    scanner->set_list(list_no, centroid_dis[i * nprobe + ij]);
                    scanner->scan_codes(
                        list_size, codes.get(), ids.get(),
                        res.get_val(i), res.get_ids(i), k);

                    n_scan += list_size;
                    if (max_codes > 0 && n_scan >= max_codes) {
                        break;
                    }
                }

                #pragma omp critical
                {
                    stats->nlist += 1;
                    stats->ndis += n_scan;
                }
            }
        }
    }

    // 5. 整理堆结果
    if (!(parallel_mode & 1024)) {
        res.reorder();
    }

    // 6. 合并统计信息
    if (stats == &local_stats) {
        indexIVF_stats.add(local_stats);
    }
}
```

### 10.2 IVFFlatScanner完整实现

```cpp
// faiss/IndexIVFFlat.cpp
// 使用VectorDistance模板的IVFFlat扫描器

namespace {

// VectorDistance模板：抽象距离计算
template <typename Distance>
struct VectorDistance {
    size_t d;                       // 维度
    bool is_similarity;            // 是否为相似度度量
    using C = typename Distance::C;  // 比较器类型

    float operator()(const float* x, const float* y) const {
        return Distance::compute(x, y, d);
    }

    VectorDistance(size_t d) : d(d), is_similarity(Distance::is_similarity) {}
};

// L2距离特化
struct L2Distance {
    using C = CMax<float, idx_t>;  // L2找最小值，用最大堆
    static constexpr bool is_similarity = false;
    size_t d;

    static float compute(const float* x, const float* y, size_t d) {
        return fvec_L2sqr(x, y, d);
    }

    L2Distance(size_t d) : d(d) {}
};

// 内积距离特化
struct InnerProductDistance {
    using C = CMin<float, idx_t>;  // 内积找最大值，用最小堆
    static constexpr bool is_similarity = true;
    size_t d;

    static float compute(const float* x, const float* y, size_t d) {
        return fvec_inner_product(x, y, d);
    }

    InnerProductDistance(size_t d) : d(d) {}
};

// IVFFlat扫描器实现
template <typename VectorDistance, bool use_sel>
struct IVFFlatScanner : InvertedListScanner {
    VectorDistance vd;
    using C = typename VectorDistance::C;

    IVFFlatScanner(
            const VectorDistance& vd,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), vd(vd) {
        keep_max = vd.is_similarity;
        code_size = vd.d * sizeof(float);
    }

    const float* xi;  // 查询向量

    void set_query(const float* query) override {
        xi = query;
    }

    void set_list(idx_t list_no, float /* coarse_dis */) override {
        this->list_no = list_no;
    }

    float distance_to_code(const uint8_t* code) const override {
        const float* yj = reinterpret_cast<const float*>(code);
        return vd(xi, yj);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {

        const float* list_vecs = reinterpret_cast<const float*>(codes);
        size_t nup = 0;

        for (size_t j = 0; j < list_size; j++) {
            const float* yj = list_vecs + vd.d * j;

            // ID选择器过滤（编译时分支）
            if constexpr (use_sel) {
                if (!sel->is_member(ids[j])) {
                    continue;
                }
            }

            // 计算距离
            float dis = vd(xi, yj);

            // 检查是否应该更新堆
            if (C::cmp(simi[0], dis)) {
                int64_t id;
                if (store_pairs) {
                    // 存储(list_no, offset)对
                    id = lo_build(list_no, j);
                } else {
                    id = ids[j];
                }

                heap_replace_top<C>(k, simi, idxi, dis, id);
                nup++;
            }
        }

        return nup;
    }
};

} // anonymous namespace

// IndexIVFFlat::get_InvertedListScanner实现
InvertedListScanner* IndexIVFFlat::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters* params) const {

    if (metric_type == METRIC_L2) {
        // L2距离
        if (sel) {
            return new IVFFlatScanner<L2Distance, true>(
                L2Distance(d), store_pairs, sel);
        } else {
            return new IVFFlatScanner<L2Distance, false>(
                L2Distance(d), store_pairs, sel);
        }
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        // 内积
        if (sel) {
            return new IVFFlatScanner<InnerProductDistance, true>(
                InnerProductDistance(d), store_pairs, sel);
        } else {
            return new IVFFlatScanner<InnerProductDistance, false>(
                InnerProductDistance(d), store_pairs, sel);
        }
    } else {
        // 其他度量类型
        return get_generic_InvertedListScanner(
            d, metric_type, store_pairs, sel);
    }
}
```

**关键优化点**：
1. **编译时多态**：`use_sel` 模板参数在编译时确定
2. **零拷贝**：直接reinterpret_cast，避免数据复制
3. **内联优化**：距离计算完全内联，消除函数调用
4. **提前终止**：`max_codes`参数可以提前终止扫描

### 10.3 DirectMap实现

```cpp
// faiss/invlists/DirectMap.cpp
// DirectMap: ID → (list_no, offset) 的快速映射

// LO编码：将list_no和offset编码到一个64位整数
inline uint64_t lo_build(uint64_t list_id, uint64_t offset) {
    return list_id << 32 | offset;
}

inline uint64_t lo_listno(uint64_t lo) {
    return lo >> 32;
}

inline uint64_t lo_offset(uint64_t lo) {
    return lo & 0xffffffff;
}

void DirectMap::set_type(
        Type new_type,
        const InvertedLists* invlists,
        size_t ntotal) {

    type = new_type;

    if (new_type == NoMap) {
        array.clear();
        hashtable.clear();
    } else if (new_type == Array) {
        // 数组模式：ID必须连续
        array.resize(ntotal);
        for (size_t i = 0; i < ntotal; i++) {
            array[i] = i;  // 初始化为自身
        }
        hashtable.clear();
    } else if (new_type == Hashtable) {
        // 哈希表模式：支持任意ID
        hashtable.clear();
        hashtable.reserve(ntotal * 2);
        array.clear();
    }
}

idx_t DirectMap::get(idx_t id) const {
    if (type == NoMap) {
        return -1;
    } else if (type == Array) {
        if (id >= 0 && id < array.size()) {
            return array[id];
        }
        return -1;
    } else { // Hashtable
        auto it = hashtable.find(id);
        if (it != hashtable.end()) {
            return it->second;
        }
        return -1;
    }
}

void DirectMap::add_single_id(idx_t id, idx_t list_no, size_t offset) {
    if (type == NoMap) {
        return;
    } else if (type == Array) {
        if (id >= 0 && id < array.size()) {
            array[id] = lo_build(list_no, offset);
        }
    } else { // Hashtable
        hashtable[id] = lo_build(list_no, offset);
    }
}

// DirectMapAdd：线程安全的批量添加
DirectMapAdd::DirectMapAdd(
        DirectMap& direct_map,
        size_t n,
        const idx_t* xids)
        : direct_map(direct_map),
          type(direct_map.type),
          ntotal(direct_map.no() ? 0 : invlists->ntotal),
          n(n),
          xids(xids) {

    if (type == DirectMap::Hashtable) {
        all_ofs.resize(n);
    }
}

void DirectMapAdd::add(size_t i, idx_t list_no, size_t offset) {
    idx_t id = xids ? xids[i] : ntotal + i;

    if (type == DirectMap::Array) {
        if (id >= 0 && id < direct_map.array.size()) {
            direct_map.array[id] = lo_build(list_no, offset);
        }
    } else if (type == DirectMap::Hashtable) {
        // 收集所有offset，批量更新
        all_ofs[i] = lo_build(list_no, offset);
    }
}

DirectMapAdd::~DirectMapAdd() {
    if (type == DirectMap::Hashtable && !all_ofs.empty()) {
        // 批量插入哈希表
        for (size_t i = 0; i < n; i++) {
            idx_t id = xids ? xids[i] : ntotal + i;
            direct_map.hashtable[id] = all_ofs[i];
        }
    }
}

// 使用DirectMap快速重建向量
void IndexIVF::reconstruct(idx_t key, float* recons) const {
    if (direct_map.no()) {
        FAISS_THROW_FMT(
            "Cannot reconstruct vector because direct_map is not initialized\n");
    }

    // 1. 获取(list_no, offset)
    idx_t lo = direct_map.get(key);
    if (lo < 0) {
        FAISS_THROW_FMT("Could not reconstruct vector %" PRId64, key);
    }

    idx_t list_no = lo_listno(lo);
    size_t offset = lo_offset(lo);

    // 2. 从倒排列表获取向量
    //    decode_vector需要知道list_no，因为by_residual时需要质心
    const uint8_t* code = invlists->get_single_code(list_no, offset);
    decode_vectors(1, &code, &list_no, recons);

    if (by_residual) {
        // 加回质心
        const float* centroid = quantizer->get_xb() + list_no * d;
        for (size_t j = 0; j < d; j++) {
            recons[j] += centroid[j];
        }
    }
}
```

### 10.4 多线程优化技巧

```cpp
// faiss/IndexIVFFlat.cpp
// add_core的多线程实现

void IndexIVFFlat::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {

    // 1. 检查前置条件
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(!by_residual);  // IVFFlat不支持by_residual
    assert(invlists);

    // 2. 初始化DirectMap更新器
    DirectMapAdd dm_adder(direct_map, n, xids);

    int64_t n_add = 0;

    // 3. OpenMP并行添加
    #pragma omp parallel reduction(+ : n_add)
    {
        int nt = omp_get_num_threads();   // 线程数
        int rank = omp_get_thread_num();   // 当前线程ID

        // 每个线程处理特定的列表（通过取模分配）
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];

            // 只处理分配给当前线程的列表
            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                const float* xi = x + i * d;

                // 添加到倒排列表
                size_t offset = invlists->add_entry(
                    list_no,
                    id,
                    reinterpret_cast<const uint8_t*>(xi),
                    inverted_list_context
                );

                // 更新direct_map
                dm_adder.add(i, list_no, offset);
                n_add++;
            } else if (rank == 0 && list_no == -1) {
                // 无效向量，只更新direct_map
                dm_adder.add(i, -1, 0);
            }
        }
    }

    // 4. 更新计数
    if (verbose) {
        printf("IndexIVFFlat::add_core: added %" PRId64 " / %" PRId64 " vectors\n",
               n_add, n);
    }

    ntotal += n;
}
```

**线程分配策略**：
```
列表 0: Thread 0
列表 1: Thread 1
列表 2: Thread 2
列表 3: Thread 3
列表 4: Thread 0  // 循环分配
列表 5: Thread 1
...
```

这种分配策略：
- 负载均衡：假设列表大小相近
- 无锁：每个线程写不同的列表
- 缓存友好：连续的向量可能到同一列表

### 10.5 IVFPQ扫描器优化

```cpp
// faiss/IndexIVFPQ.cpp
// IVFPQ扫描器：使用查找表优化

struct IVFPQScannerCommon : InvertedListScanner {
    const ProductQuantizer& pq;
    const float* query_norms;      // 查询范数（用于L2）
    AlignedTable<float> dis_tables;  // [M * ksub]
    AlignedTable<float> coarse_dis;  // 粗距离

    IVFPQScannerCommon(
            const ProductQuantizer& pq,
            bool store_pairs,
            const IDSelector* sel)
        : InvertedListScanner(store_pairs, sel),
          pq(pq),
          dis_tables(pq.M * pq.ksub),
          coarse_dis(1) {}

    void set_query(const float* query) override {
        // 1. 预计算查询范数
        if (query_norms) {
            // query_norms已经在外部计算
        } else {
            // 计算L2范数：||q||²
            float norm = 0;
            for (size_t i = 0; i < pq.d; i++) {
                norm += query[i] * query[i];
            }
            query_norms = &norm;
        }

        // 2. 计算查找表
        pq.compute_L2_distance_table(
            query,
            query_norms,
            dis_tables.get());
    }

    void set_list(idx_t list_no, float coarse_dis_in) override {
        this->list_no = list_no;
        coarse_dis[0] = coarse_dis_in;
    }
};

// SIMD优化的scan_codes
template <MetricType metric, bool use_sel>
struct IVFPQScanner : IVFPQScannerCommon {

    IVFPQScanner(
            const ProductQuantizer& pq,
            bool store_pairs,
            const IDSelector* sel)
        : IVFPQScannerCommon(pq, store_pairs, sel) {
        code_size = pq.code_size;
    }

    size_t scan_codes(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {

        size_t nup = 0;

        // 批量处理：一次处理4个向量
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            // 批量检查ID选择器
            if constexpr (use_sel) {
                if (!sel->is_member(ids[i + 0]) ||
                    !sel->is_member(ids[i + 1]) ||
                    !sel->is_member(ids[i + 2]) ||
                    !sel->is_member(ids[i + 3])) {
                    break;  // 有一个不满足，回退到逐个处理
                }
            }

            // SIMD批量计算4个距离
            float dis[4];
            pq.compute_codes_dis_tables(
                codes + i * code_size,
                dis_tables.get(),
                dis);

            // 更新堆（4次）
            for (int j = 0; j < 4; j++) {
                // L2距离：dis + coarse_dis
                float final_dis = coarse_dis[0] + dis[j];

                if (CMax<float, idx_t>::cmp(final_dis, simi[0])) {
                    idx_t id = store_pairs ? lo_build(list_no, i + j) : ids[i + j];
                    heap_replace_top<CMax<float, idx_t>>(
                        k, simi, idxi, final_dis, id);
                    nup++;
                }
            }
        }

        // 处理剩余向量
        for (; i < n; i++) {
            if constexpr (use_sel) {
                if (!sel->is_member(ids[i])) {
                    continue;
                }
            }

            // 单个距离计算
            float dis = coarse_dis[0];
            const uint8_t* code = codes + i * code_size;

            for (size_t m = 0; m < pq.M; m++) {
                // 提取子量化器m的编码
                uint64_t code_m = decode_uint64(
                    code, m * pq.nbits, pq.nbits);
                // 查找表距离
                dis += dis_tables[m * pq.ksub + code_m];
            }

            if (CMax<float, idx_t>::cmp(dis, simi[0])) {
                idx_t id = store_pairs ? lo_build(list_no, i) : ids[i];
                heap_replace_top<CMax<float, idx_t>>(
                    k, simi, idxi, dis, id);
                nup++;
            }
        }

        return nup;
    }
};
```

### 10.6 性能分析工具

```cpp
// faiss/IndexIVF.h
// IndexIVFStats：搜索统计信息

struct IndexIVFStats {
    size_t nq;                // 查询数量
    size_t nlist;             // 扫描的倒排列表数
    size_t ndis;              // 计算的距离数
    size_t nheap_updates;     // 堆更新次数
    double quantization_time; // 量化时间（毫秒）
    double search_time;       // 搜索时间（毫秒）

    IndexIVFStats() {
        reset();
    }

    void reset() {
        memset(this, 0, sizeof(*this));
    }

    void add(const IndexIVFStats& other) {
        nq += other.nq;
        nlist += other.nlist;
        ndis += other.ndis;
        nheap_updates += other.nheap_updates;
        quantization_time += other.quantization_time;
        search_time += other.search_time;
    }
};

// 全局统计变量
FAISS_API extern IndexIVFStats indexIVF_stats;

// 使用示例
void search_with_stats(IndexIVF& index, const float* query, size_t k) {
    indexIVF_stats.reset();

    auto t0 = std::chrono::high_resolution_clock::now();
    index.search(1, query, k, distances, labels);
    auto t1 = std::chrono::high_resolution_clock::now();

    indexIVF_stats.search_time =
        std::chrono::duration<double>(t1 - t0).count() * 1000;

    printf("Search stats:\n");
    printf("  nlist: %zu\n", indexIVF_stats.nlist);
    printf("  ndis: %zu\n", indexIVF_stats.ndis);
    printf("  heap updates: %zu\n", indexIVF_stats.nheap_updates);
    printf("  time: %.3f ms\n", indexIVF_stats.search_time);
}
```

---

## 11. IVF SIMD底层优化

### 11.1 批量粗量化优化

```cpp
// faiss/IndexIVF.cpp
// SIMD优化的批量粗量化

#ifdef __AVX2__
// AVX2优化的质心距离计算
void compute_quantizer_distances_avx2(
        const float* x,             // 查询向量
        idx_t nq,                   // 查询数
        const float* centroids,     // 质心 [nlist * d]
        size_t nlist,
        size_t d,
        float* distances) {         // 输出 [nq * nlist]

    for (idx_t q = 0; q < nq; q++) {
        const float* xq = x + q * d;
        float* dist_q = distances + q * nlist;

        for (size_t list_no = 0; list_no < nlist; list_no++) {
            const float* centroid = centroids + list_no * d;

            // AVX2优化的L2距离计算
            __m256 sum = _mm256_setzero_ps();

            size_t i = 0;
            for (; i + 8 <= d; i += 8) {
                __m256 xv = _mm256_loadu_ps(xq + i);
                __m256 cv = _mm256_loadu_ps(centroid + i);

                __m256 diff = _mm256_sub_ps(xv, cv);
                sum = _mm256_fmadd_ps(diff, diff, sum);
            }

            // 水平求和
            sum = _mm256_hadd_ps(sum, sum);
            sum = _mm256_hadd_ps(sum, sum);

            float result;
            _mm256_store_ss(&result, sum);

            // 处理剩余元素
            for (; i < d; i++) {
                float diff = xq[i] - centroid[i];
                result += diff * diff;
            }

            dist_q[list_no] = result;
        }
    }
}

// AVX-512优化的版本（一次处理16个float）
#ifdef __AVX512F__
void compute_quantizer_distances_avx512(
        const float* x,
        idx_t nq,
        const float* centroids,
        size_t nlist,
        size_t d,
        float* distances) {

    for (idx_t q = 0; q < nq; q++) {
        const float* xq = x + q * d;

        for (size_t list_no = 0; list_no < nlist; list_no++) {
            const float* centroid = centroids + list_no * d;

            __m512 sum = _mm512_setzero_ps();

            size_t i = 0;
            for (; i + 16 <= d; i += 16) {
                __m512 xv = _mm512_loadu_ps(xq + i);
                __m512 cv = _mm512_loadu_ps(centroid + i);

                __m512 diff = _mm512_sub_ps(xv, cv);
                sum = _mm512_fmadd_ps(diff, diff, sum);
            }

            float result = _mm512_reduce_add_ps(sum);

            // 处理剩余元素
            for (; i < d; i++) {
                float diff = xq[i] - centroid[i];
                result += diff * diff;
            }

            distances[q * nlist + list_no] = result;
        }
    }
}
#endif
#endif

// 内积度量的SIMD优化
#ifdef __AVX2__
void compute_quantizer_inner_products_avx2(
        const float* x,
        idx_t nq,
        const float* centroids,
        size_t nlist,
        size_t d,
        float* products) {  // 输出 [nq * nlist]

    for (idx_t q = 0; q < nq; q++) {
        const float* xq = x + q * d;
        float* prod_q = products + q * nlist;

        for (size_t list_no = 0; list_no < nlist; list_no++) {
            const float* centroid = centroids + list_no * d;

            __m256 sum = _mm256_setzero_ps();

            size_t i = 0;
            for (; i + 8 <= d; i += 8) {
                __m256 xv = _mm256_loadu_ps(xq + i);
                __m256 cv = _mm256_loadu_ps(centroid + i);

                sum = _mm256_fmadd_ps(xv, cv, sum);
            }

            // 水平求和
            sum = _mm256_hadd_ps(sum, sum);
            sum = _mm256_hadd_ps(sum, sum);

            float result;
            _mm256_store_ss(&result, sum);

            for (; i < d; i++) {
                result += xq[i] * centroid[i];
            }

            prod_q[list_no] = result;
        }
    }
}
#endif
```

### 11.2 SIMD优化的倒排列表扫描

```cpp
// faiss/IndexIVFFlat.cpp
// SIMD优化的批量向量距离计算

#ifdef __AVX2__
// 批量处理8个向量的L2距离
void batch_L2_distances_avx2(
        const float* query,
        const float* vectors,  // [n * d]
        size_t n,
        size_t d,
        float* distances) {

    for (size_t i = 0; i + 8 <= n; i += 8) {
        // 8个向量，每个d维
        const float* vec0 = vectors + (i + 0) * d;
        const float* vec1 = vectors + (i + 1) * d;
        const float* vec2 = vectors + (i + 2) * d;
        const float* vec3 = vectors + (i + 3) * d;
        const float* vec4 = vectors + (i + 4) * d;
        const float* vec5 = vectors + (i + 5) * d;
        const float* vec6 = vectors + (i + 6) * d;
        const float* vec7 = vectors + (i + 7) * d;

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        __m256 sum4 = _mm256_setzero_ps();
        __m256 sum5 = _mm256_setzero_ps();
        __m256 sum6 = _mm256_setzero_ps();
        __m256 sum7 = _mm256_setzero_ps();

        size_t j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 qv = _mm256_loadu_ps(query + j);

            __m256 v0 = _mm256_loadu_ps(vec0 + j);
            __m256 v1 = _mm256_loadu_ps(vec1 + j);
            __m256 v2 = _mm256_loadu_ps(vec2 + j);
            __m256 v3 = _mm256_loadu_ps(vec3 + j);
            __m256 v4 = _mm256_loadu_ps(vec4 + j);
            __m256 v5 = _mm256_loadu_ps(vec5 + j);
            __m256 v6 = _mm256_loadu_ps(vec6 + j);
            __m256 v7 = _mm256_loadu_ps(vec7 + j);

            __m256 diff0 = _mm256_sub_ps(qv, v0);
            __m256 diff1 = _mm256_sub_ps(qv, v1);
            __m256 diff2 = _mm256_sub_ps(qv, v2);
            __m256 diff3 = _mm256_sub_ps(qv, v3);
            __m256 diff4 = _mm256_sub_ps(qv, v4);
            __m256 diff5 = _mm256_sub_ps(qv, v5);
            __m256 diff6 = _mm256_sub_ps(qv, v6);
            __m256 diff7 = _mm256_sub_ps(qv, v7);

            sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
            sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
            sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
            sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
            sum4 = _mm256_fmadd_ps(diff4, diff4, sum4);
            sum5 = _mm256_fmadd_ps(diff5, diff5, sum5);
            sum6 = _mm256_fmadd_ps(diff6, diff6, sum6);
            sum7 = _mm256_fmadd_ps(diff7, diff7, sum7);
        }

        // 水平求和
        __m256 sums0 = _mm256_hadd_ps(sum0, sum1);
        __m256 sums1 = _mm256_hadd_ps(sum2, sum3);
        __m256 sums2 = _mm256_hadd_ps(sum4, sum5);
        __m256 sums3 = _mm256_hadd_ps(sum6, sum7);

        sums0 = _mm256_hadd_ps(sums0, sums1);
        sums2 = _mm256_hadd_ps(sums2, sums3);

        // 提取结果
        alignas(32) float results[8];
        _mm256_storeu_ps(results, sums0);
        _mm256_storeu_ps(results + 4, sums2);

        // 处理剩余维度
        for (; j < d; j++) {
            float qv_j = query[j];
            for (int k = 0; k < 8; k++) {
                float v = vectors[(i + k) * d + j];
                float diff = qv_j - v;
                results[k] += diff * diff;
            }
        }

        // 存储结果
        for (int k = 0; k < 8; k++) {
            distances[i + k] = results[k];
        }
    }

    // 处理剩余向量（不足8个）
    for (size_t i = (n / 8) * 8; i < n; i++) {
        const float* vec = vectors + i * d;
        float dis = 0;
        for (size_t j = 0; j < d; j++) {
            float diff = query[j] - vec[j];
            dis += diff * diff;
        }
        distances[i] = dis;
    }
}
#endif

// AVX-512优化的批量距离计算（一次16个向量）
#ifdef __AVX512F__
void batch_L2_distances_avx512(
        const float* query,
        const float* vectors,
        size_t n,
        size_t d,
        float* distances) {

    for (size_t i = 0; i + 16 <= n; i += 16) {
        __m512 sums[16];

        // 初始化累加器
        for (int k = 0; k < 16; k++) {
            sums[k] = _mm512_setzero_ps();
        }

        size_t j = 0;
        for (; j + 16 <= d; j += 16) {
            __m512 qv = _mm512_loadu_ps(query + j);

            // 展开16个向量
            for (int k = 0; k < 16; k++) {
                __m512 vk = _mm512_loadu_ps(vectors + (i + k) * d + j);
                __m512 diff = _mm512_sub_ps(qv, vk);
                sums[k] = _mm512_fmadd_ps(diff, diff, sums[k]);
            }
        }

        // 求和
        for (int k = 0; k < 16; k++) {
            distances[i + k] = _mm512_reduce_add_ps(sums[k]);
        }
    }

    // 处理剩余向量
    for (size_t i = (n / 16) * 16; i < n; i++) {
        const float* vec = vectors + i * d;
        float dis = 0;
        for (size_t j = 0; j < d; j++) {
            float diff = query[j] - vec[j];
            dis += diff * diff;
        }
        distances[i] = dis;
    }
}
#endif
```

### 11.3 SIMD优化的PQ查找表计算

```cpp
// faiss/impl/ProductQuantizer.cpp
// SIMD优化的PQ距离表计算

#ifdef __AVX2__
void compute_L2_distance_table_avx2(
        const float* query,          // [d]
        const float* centroids,      // [M * ksub * dsub]
        size_t M,
        size_t ksub,
        size_t dsub,
        float* dis_tables) {         // 输出 [M * ksub]

    for (size_t m = 0; m < M; m++) {
        const float* query_m = query + m * dsub;
        const float* centroids_m = centroids + m * ksub * dsub;
        float* table_m = dis_tables + m * ksub;

        // 每次处理8个质心
        for (size_t k = 0; k + 8 <= ksub; k += 8) {
            // 初始化8个累加器
            __m256 sums[8];
            for (int i = 0; i < 8; i++) {
                sums[i] = _mm256_setzero_ps();
            }

            // 遍历维度
            size_t j = 0;
            for (; j + 8 <= dsub; j += 8) {
                __m256 qv = _mm256_loadu_ps(query_m + j);

                // 加载8个质心的第j个维度
                for (int i = 0; i < 8; i++) {
                    __m256 cv = _mm256_loadu_ps(
                            centroids_m + (k + i) * dsub + j);
                    __m256 diff = _mm256_sub_ps(qv, cv);
                    sums[i] = _mm256_fmadd_ps(diff, diff, sums[i]);
                }
            }

            // 处理剩余维度
            for (; j < dsub; j++) {
                float qv_j = query_m[j];
                for (int i = 0; i < 8; i++) {
                    float cv = centroids_m[(k + i) * dsub + j];
                    float diff = qv_j - cv;
                    float current;
                    _mm256_store_ss(&current, sums[i]);
                    sums[i] = _mm256_set_ps(
                        0, 0, 0, 0, 0, 0, 0,
                        current + diff * diff);
                }
            }

            // 水平求和并存储
            for (int i = 0; i < 8; i++) {
                sums[i] = _mm256_hadd_ps(sums[i], sums[i]);
                sums[i] = _mm256_hadd_ps(sums[i], sums[i]);
                _mm256_store_ss(table_m + k + i, sums[i]);
            }
        }

        // 处理剩余质心
        for (size_t k = (ksub / 8) * 8; k < ksub; k++) {
            float dis = 0;
            for (size_t j = 0; j < dsub; j++) {
                float diff = query_m[j] - centroids_m[k * dsub + j];
                dis += diff * diff;
            }
            table_m[k] = dis;
        }
    }
}
#endif

// ARM NEON优化的版本
#ifdef __ARM_NEON
void compute_L2_distance_table_neon(
        const float* query,
        const float* centroids,
        size_t M,
        size_t ksub,
        size_t dsub,
        float* dis_tables) {

    for (size_t m = 0; m < M; m++) {
        const float* query_m = query + m * dsub;
        const float* centroids_m = centroids + m * ksub * dsub;
        float* table_m = dis_tables + m * ksub;

        // 每次处理4个质心
        for (size_t k = 0; k + 4 <= ksub; k += 4) {
            float32x4_t sums[4];
            for (int i = 0; i < 4; i++) {
                sums[i] = vdupq_n_f32(0);
            }

            size_t j = 0;
            for (; j + 4 <= dsub; j += 4) {
                float32x4_t qv = vld1q_f32(query_m + j);

                for (int i = 0; i < 4; i++) {
                    float32x4_t cv = vld1q_f32(
                            centroids_m + (k + i) * dsub + j);
                    float32x4_t diff = vsubq_f32(qv, cv);
                    sums[i] = vmlaq_f32(sums[i], diff, diff);
                }
            }

            // 处理剩余维度
            for (; j < dsub; j++) {
                float qv_j = query_m[j];
                for (int i = 0; i < 4; i++) {
                    float cv = centroids_m[(k + i) * dsub + j];
                    float diff = qv_j - cv;
                    sums[i] = vsetq_lane_f32(
                            vgetq_lane_f32(sums[i], 0) + diff * diff,
                            sums[i], 0);
                }
            }

            // 提取结果
            for (int i = 0; i < 4; i++) {
                table_m[k + i] = vaddvq_f32(sums[i]);
            }
        }

        // 处理剩余质心
        for (size_t k = (ksub / 4) * 4; k < ksub; k++) {
            float dis = 0;
            for (size_t j = 0; j < dsub; j++) {
                float diff = query_m[j] - centroids_m[k * dsub + j];
                dis += diff * diff;
            }
            table_m[k] = dis;
        }
    }
}
#endif
```

---

## 12. IVF内存布局深度优化

### 12.1 缓存友好的倒排列表布局

```cpp
// faiss/invlists/InvertedLists.h
// 优化内存布局的倒排列表实现

struct alignas(64) CacheLineAwareInvertedList {
    // 每个缓存行存储固定数量的编码
    static constexpr size_t CODES_PER_CACHE_LINE = 64 / sizeof(uint8_t);
    static constexpr size_t IDS_PER_CACHE_LINE = 64 / sizeof(idx_t);

    // 编码数组（缓存行对齐）
    std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> codes;

    // ID数组（缓存行对齐）
    std::vector<idx_t, AlignedAllocator<idx_t, 64>> ids;

    size_t capacity;
    size_t size;

    void resize(size_t new_size) {
        if (new_size > capacity) {
            // 按2的幂扩展，对齐到缓存行
            size_t new_capacity = std::max(capacity, size_t(1));
            while (new_capacity < new_size) {
                new_capacity *= 2;
            }

            // 对齐到缓存行
            codes.resize(new_capacity * code_size);
            ids.resize(new_capacity);
            capacity = new_capacity;
        }
        size = new_size;
    }

    size_t add(const idx_t* xids, const uint8_t* xcodes, size_t n) {
        size_t o = size;
        resize(size + n);

        memcpy(ids.data() + o, xids, n * sizeof(idx_t));
        memcpy(codes.data() + o * code_size, xcodes, n * code_size);

        return o;
    }
};

// 交错布局：codes和ids交错存储，提高缓存命中率
struct InterleavedInvertedLists : InvertedLists {
    struct alignas(64) InterleavedBlock {
        // 交错存储：[id0, code0, id1, code1, ..., idN, codeN]
        std::vector<uint8_t> data;

        size_t n;           // 当前列表中的向量数
        size_t capacity;    // 容量

        InterleavedBlock() : n(0), capacity(1024) {
            data.reserve(capacity * (sizeof(idx_t) + code_size));
        }

        size_t add(idx_t id, const uint8_t* code) {
            if (n >= capacity) {
                resize(capacity * 2);
            }

            size_t offset = n * (sizeof(idx_t) + code_size);
            memcpy(data.data() + offset, &id, sizeof(idx_t));
            memcpy(data.data() + offset + sizeof(idx_t), code, code_size);

            return n++;
        }

        void resize(size_t new_capacity) {
            data.resize(new_capacity * (sizeof(idx_t) + code_size));
            capacity = new_capacity;
        }
    };

    std::vector<InterleavedBlock> lists;

    const uint8_t* get_codes(size_t list_no) const override {
        // 跳过ID，只返回codes指针
        return lists[list_no].data.data() + sizeof(idx_t);
    }

    const idx_t* get_ids(size_t list_no) const override {
        return reinterpret_cast<const idx_t*>(lists[list_no].data.data());
    }
};
```

### 12.2 NUMA感知的IVF索引

```cpp
#ifdef __linux__
#include <numa.h>

// NUMA优化的IVF索引
struct NUMAAwareIndexIVF : IndexIVF {
    int numa_nodes;
    std::vector<void*> numa_invlists;

    void init_numa(int nodes) {
        numa_nodes = nodes;
        numa_invlists.resize(nodes);

        for (int node = 0; node < nodes; node++) {
            // 为每个NUMA节点分配倒排列表
            size_t size = (nlist / nodes) * sizeof(InvertedLists*);
            numa_invlists[node] = numa_alloc_onnode(size, node);
        }
    }

    void add_with_numa(idx_t n, const float* x, const idx_t* xids) {
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int numa_node = thread_id % numa_nodes;

            // 绑定到NUMA节点
            numa_set_preferred(numa_node);

            // 为当前线程分配临时存储
            std::vector<std::vector<idx_t>> local_list_nos(nlist);
            std::vector<std::vector<float>> local_residuals(nlist);

            #pragma omp for schedule(dynamic)
            for (idx_t i = 0; i < n; i++) {
                // 找到最近的质心
                idx_t list_no = quantizer->assign(x + i * d);

                if (list_no >= 0) {
                    local_list_nos[list_no].push_back(i);
                    if (by_residual) {
                        // 存储残差
                        const float* centroid = get_centroid(list_no);
                        std::vector<float> res(d);
                        for (int j = 0; j < d; j++) {
                            res[j] = x[i * d + j] - centroid[j];
                        }
                        local_residuals[list_no].insert(
                            local_residuals[list_no].end(),
                            res.begin(), res.end());
                    }
                }
            }

            // 批量添加到倒排列表
            // （这部分需要加锁保护）
        }
    }

    ~NUMAAwareIndexIVF() {
        for (int node = 0; node < numa_nodes; node++) {
            numa_free(numa_invlists[node], /* size */);
        }
    }
};
#endif
```

### 12.3 预取优化的搜索

```cpp
// 软件预取优化的IVF搜索
struct PrefetchingIVFScanner : InvertedListScanner {
    static constexpr size_t PREFETCH_DISTANCE = 4;

    template<typename DistanceComputer>
    void scan_with_prefetch(
            const DistanceComputer& dis_computer,
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* distances,
            idx_t* labels,
            size_t k) const {

        size_t nup = 0;

        for (size_t i = 0; i < list_size; i++) {
            // 预取未来的编码
            if (i + PREFETCH_DISTANCE < list_size) {
                const uint8_t* future_code = codes +
                    (i + PREFETCH_DISTANCE) * code_size;
                _mm_prefetch((const char*)future_code, _MM_HINT_T0);

                // 预取未来的ID
                const idx_t* future_id = ids + (i + PREFETCH_DISTANCE);
                _mm_prefetch((const char*)future_id, _MM_HINT_T0);
            }

            // 处理当前编码
            if (sel && !sel->is_member(ids[i])) {
                continue;
            }

            float dis = dis_computer(codes + i * code_size);

            if (C::cmp(dis, distances[0])) {
                heap_replace_top<C>(k, distances, labels, dis, ids[i]);
                nup++;
            }
        }

        // 返回更新次数
        return nup;
    }
};

// 双缓冲预取：隐藏内存延迟
template<int BUFFER_SIZE = 16>
struct DoubleBufferedIVFScanner {
    struct BufferEntry {
        const uint8_t* code;
        idx_t id;
        bool valid;
    };

    BufferEntry buffers[2][BUFFER_SIZE];
    int current_buffer = 0;
    int buffer_pos = 0;

    void fill_buffer(
            const uint8_t* codes,
            const idx_t* ids,
            size_t start,
            size_t list_size) {

        BufferEntry* buf = buffers[current_buffer];
        int count = 0;

        for (int i = 0; i < BUFFER_SIZE && start + i < list_size; i++) {
            buf[i].code = codes + (start + i) * code_size;
            buf[i].id = ids[start + i];
            buf[i].valid = !(sel && !sel->is_member(buf[i].id));
            count++;
        }

        // 清空剩余槽位
        for (int i = count; i < BUFFER_SIZE; i++) {
            buf[i].valid = false;
        }
    }

    size_t scan(
            const DistanceComputer& dis_computer,
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* distances,
            idx_t* labels,
            size_t k) {

        size_t nup = 0;
        size_t pos = 0;

        // 填充第一个缓冲区
        fill_buffer(codes, ids, pos, list_size);
        pos += BUFFER_SIZE;

        while (buffers[current_buffer][0].valid) {
            // 交换缓冲区
            current_buffer ^= 1;

            // 异步填充下一个缓冲区
            if (pos < list_size) {
                fill_buffer(codes, ids, pos, list_size);
                pos += BUFFER_SIZE;
            }

            // 处理当前缓冲区
            BufferEntry* buf = buffers[current_buffer];

            for (int i = 0; i < BUFFER_SIZE && buf[i].valid; i++) {
                if (!buf[i].valid) continue;

                float dis = dis_computer(buf[i].code);

                if (C::cmp(dis, distances[0])) {
                    heap_replace_top<C>(k, distances, labels, dis, buf[i].id);
                    nup++;
                }
            }
        }

        return nup;
    }
};
```

### 12.4 压缩倒排列表

```cpp
// 使用位压缩减少内存占用
struct CompressedInvertedLists : InvertedLists {
    // 如果nlist <= 65536，ID可以用uint16_t存储
    // 如果code_size很小，多个编码可以打包到一个64位整数

    std::vector<uint64_t> packed_data;  // 打包的数据

    size_t add_entries(
            size_t list_no,
            size_t n,
            const idx_t* xids,
            const uint8_t* xcode) override {

        size_t offset = packed_data.size();

        // 假设nlist <= 65536，ID压缩为uint16_t
        // code_size = 4字节，可以与ID一起打包
        for (size_t i = 0; i < n; i++) {
            uint64_t packed = 0;

            // 打包ID（低32位）
            packed |= static_cast<uint64_t>(xids[i] & 0xFFFF);

            // 打码code（高32位）
            uint32_t code_val = 0;
            memcpy(&code_val, xcode + i * code_size, code_size);
            packed |= static_cast<uint64_t>(code_val) << 32;

            packed_data.push_back(packed);
        }

        return offset;
    }

    const uint8_t* get_codes(size_t list_no) const override {
        // 需要解压
        // 返回临时缓冲区的指针（不推荐）
        return nullptr;
    }

    const idx_t* get_ids(size_t list_no) const override {
        // 需要解压
        return nullptr;
    }

    // 提供压缩版本的扫描
    size_t scan_compressed(
            size_t list_no,
            const uint64_t* packed,
            size_t n,
            const DistanceComputer& dis_computer,
            float* distances,
            idx_t* labels,
            size_t k) const {

        size_t nup = 0;

        for (size_t i = 0; i < n; i++) {
            uint64_t packed = packed[i];

            // 解包ID
            idx_t id = packed & 0xFFFF;

            // 解包code
            uint32_t code_val = packed >> 32;
            uint8_t code[4];
            memcpy(code, &code_val, code_size);

            float dis = dis_computer(code);

            if (C::cmp(dis, distances[0])) {
                heap_replace_top<C>(k, distances, labels, dis, id);
                nup++;
            }
        }

        return nup;
    }
};
```

---

## 13. IVF并发优化

### 13.1 无锁倒排列表添加

```cpp
#include <atomic>

// 无锁的倒排列表实现
struct LockFreeInvertedList {
    struct Node {
        idx_t id;
        uint8_t code[64];  // 假设最大code_size
        std::atomic<Node*> next;

        Node(idx_t id, const uint8_t* code, size_t code_size)
            : id(id), next(nullptr) {
            memcpy(this->code, code, code_size);
        }
    };

    std::atomic<Node*> head;

    LockFreeInvertedList() : head(nullptr) {}

    void add(idx_t id, const uint8_t* code, size_t code_size) {
        Node* new_node = new Node(id, code, code_size);

        while (true) {
            Node* old_head = head.load(std::memory_order_acquire);
            new_node->next.store(old_head, std::memory_order_relaxed);

            if (head.compare_exchange_weak(old_head, new_node,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                break;  // 成功插入
            }
            // CAS失败，重试
        }
    }

    // 遍历所有节点（需要外部同步）
    template<typename Func>
    void iterate(Func&& func) {
        Node* current = head.load(std::memory_order_acquire);

        while (current) {
            func(current->id, current->code);
            current = current->next.load(std::memory_order_acquire);
        }
    }
};

// 使用TLS（线程本地存储）的批量添加
struct TLSInvertedLists : InvertedLists {
    struct TLSData {
        std::vector<std::vector<idx_t>> ids;
        std::vector<std::vector<uint8_t>> codes;

        TLSData(size_t nlist, size_t code_size) {
            ids.resize(nlist);
            codes.resize(nlist);
        }

        void add(size_t list_no, idx_t id, const uint8_t* code) {
            ids[list_no].push_back(id);
            codes[list_no].insert(codes[list_no].end(),
                                 code, code + code_size);
        }
    };

    std::vector<TLSData*> thread_local_data;
    std::mutex tls_mutex;

    TLSData* get_tls_data() {
        int thread_id = omp_get_thread_num();

        if (thread_id >= thread_local_data.size()) {
            std::lock_guard<std::mutex> lock(tls_mutex);
            // 双重检查
            if (thread_id >= thread_local_data.size()) {
                thread_local_data.resize(thread_id + 1);
                thread_local_data[thread_id] =
                    new TLSData(nlist, code_size);
            }
        }

        return thread_local_data[thread_id];
    }

    size_t add_entries(
            size_t list_no,
            size_t n,
            const idx_t* xids,
            const uint8_t* xcode) override {

        TLSData* tls = get_tls_data();

        for (size_t i = 0; i < n; i++) {
            tls->add(list_no, xids[i], xcode + i * code_size);
        }

        return 0;  // 返回值在TLS模式下无意义
    }

    // 合并所有线程的数据
    void merge_all() {
        // 清空主存储
        for (size_t list_no = 0; list_no < nlist; list_no++) {
            ids[list_no].clear();
            codes[list_no].clear();
        }

        // 合并TLS数据
        for (TLSData* tls : thread_local_data) {
            for (size_t list_no = 0; list_no < nlist; list_no++) {
                size_t offset = ids[list_no].size();

                ids[list_no].insert(ids[list_no].end(),
                                   tls->ids[list_no].begin(),
                                   tls->ids[list_no].end());

                codes[list_no].insert(codes[list_no].end(),
                                     tls->codes[list_no].begin(),
                                     tls->codes[list_no].end());
            }
        }

        // 清空TLS数据
        for (TLSData* tls : thread_local_data) {
            for (size_t list_no = 0; list_no < nlist; list_no++) {
                tls->ids[list_no].clear();
                tls->codes[list_no].clear();
            }
        }
    }
};
```

### 13.2 并行搜索优化

```cpp
// 细粒度并行搜索：按列表+查询二维并行
struct ParallelIVFSearcher {
    static void search_2d_parallel(
            const IndexIVF& index,
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const IVFSearchParameters* params) {

        // 1. 粗量化
        idx_t* assign = new idx_t[n * index.nprobe];
        float* centroid_dis = new float[n * index.nprobe];

        index.quantizer->search(n, x, index.nprobe,
                               centroid_dis, assign);

        // 2. 二维并行：查询 × 列表
        #pragma omp parallel collapse(2)
        for (idx_t q = 0; q < n; q++) {
            for (size_t ij = 0; ij < index.nprobe; ij++) {
                idx_t list_no = assign[q * index.nprobe + ij];

                if (list_no < 0 || list_no >= index.nlist) {
                    continue;
                }

                size_t list_size = index.invlists->list_size(list_no);
                if (list_size == 0) {
                    continue;
                }

                // 每个线程独立的scanner
                std::unique_ptr<InvertedListScanner> scanner(
                    index.get_InvertedListScanner(false, nullptr, params));

                scanner->set_query(x + q * index.d);
                scanner->set_list(list_no, centroid_dis[q * index.nprobe + ij]);

                InvertedLists::ScopedCodes codes(index.invlists, list_no);
                InvertedLists::ScopedIds ids(index.invlists, list_no);

                // 临界区保护堆更新
                #pragma omp critical
                {
                    scanner->scan_codes(
                        list_size, codes.get(), ids.get(),
                        distances + q * k, labels + q * k, k);
                }
            }
        }

        // 3. 整理结果
        for (idx_t q = 0; q < n; q++) {
            heap_reorder<CMax<float, idx_t>>(
                k, distances + q * k, labels + q * k);
        }

        delete[] assign;
        delete[] centroid_dis;
    }
};

// 工作窃取调度：负载均衡
struct WorkStealingIVFSearcher {
    struct WorkItem {
        idx_t query_id;
        size_t probe_idx;
    };

    std::vector<WorkItem> work_queue;
    std::atomic<size_t> work_ptr{0};
    std::mutex queue_mutex;

    void distribute_work(
            const IndexIVF& index,
            idx_t n,
            const idx_t* assign) {

        work_queue.clear();
        work_queue.reserve(n * index.nprobe);

        for (idx_t q = 0; q < n; q++) {
            for (size_t ij = 0; ij < index.nprobe; ij++) {
                idx_t list_no = assign[q * index.nprobe + ij];

                if (list_no >= 0 && list_no < index.nlist) {
                    work_queue.push_back({q, ij});
                }
            }
        }
    }

    WorkItem steal_work() {
        size_t old_ptr = work_ptr.fetch_add(1, std::memory_order_relaxed);

        if (old_ptr >= work_queue.size()) {
            return {-1, 0};  // 无工作
        }

        return work_queue[old_ptr];
    }

    void search_parallel(
            const IndexIVF& index,
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const idx_t* assign,
            const float* centroid_dis,
            const IVFSearchParameters* params) {

        distribute_work(index, n, assign);

        #pragma omp parallel
        {
            std::unique_ptr<InvertedListScanner> scanner(
                index.get_InvertedListScanner(false, nullptr, params));

            while (true) {
                WorkItem work = steal_work();

                if (work.query_id < 0) {
                    break;  // 无工作
                }

                idx_t list_no = assign[work.query_id * index.nprobe +
                                      work.probe_idx];

                scanner->set_query(x + work.query_id * index.d);
                scanner->set_list(list_no,
                    centroid_dis[work.query_id * index.nprobe + work.probe_idx]);

                size_t list_size = index.invlists->list_size(list_no);
                if (list_size == 0) {
                    continue;
                }

                InvertedLists::ScopedCodes codes(index.invlists, list_no);
                InvertedLists::ScopedIds ids(index.invlists, list_no);

                #pragma omp critical
                {
                    scanner->scan_codes(
                        list_size, codes.get(), ids.get(),
                        distances + work.query_id * k,
                        labels + work.query_id * k, k);
                }
            }
        }

        // 整理结果
        #pragma omp parallel for
        for (idx_t q = 0; q < n; q++) {
            heap_reorder<CMax<float, idx_t>>(
                k, distances + q * k, labels + q * k);
        }
    }
};
```

### 13.3 读-写锁优化

```cpp
#include <shared_mutex>

// 使用读写锁的IVF索引
struct RWLockIndexIVF : IndexIVF {
    mutable std::shared_mutex mutex;

    // 读操作（搜索）：共享锁
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        std::shared_lock<std::shared_mutex> lock(mutex);

        // 执行搜索...
        IndexIVF::search(n, x, k, distances, labels, params);
    }

    // 写操作（添加）：独占锁
    void add(idx_t n, const float* x) override {
        std::unique_lock<std::shared_mutex> lock(mutex);

        // 执行添加...
        IndexIVF::add(n, x);
    }

    // 允许读-写并行的接口
    void add_with_read_parallel(
            idx_t n,
            const float* x,
            const idx_t* xids) {

        // 分批添加，每批之间释放锁
        constexpr idx_t batch_size = 10000;

        for (idx_t i = 0; i < n; i += batch_size) {
            idx_t batch = std::min(batch_size, n - i);

            {
                std::unique_lock<std::shared_mutex> lock(mutex);
                IndexIVF::add_core(batch, x + i * d, xids ? xids + i : nullptr,
                                  nullptr, nullptr);
            }

            // 允许搜索在批次之间执行
        }
    }
};
```

---

## 14. IVF性能分析与调优

### 14.1 自动参数调优工具

```cpp
// IVF参数自动调优器
struct IVFTuner {
    struct Config {
        size_t nlist;
        size_t nprobe;
        bool by_residual;
        double qps;
        float recall;
    };

    std::vector<Config> results;

    void grid_search(
            const float* train_vectors,
            idx_t ntrain,
            const float* query_vectors,
            idx_t nquery,
            const idx_t* ground_truth,
            int d) {

        std::vector<size_t> nlist_values = {
            100, 256, 512, 1024, 2048, 4096
        };

        std::vector<size_t> nprobe_values = {
            1, 5, 10, 20, 50, 100
        };

        for (size_t nlist : nlist_values) {
            // 创建IVF索引
            IndexFlatL2 quantizer(d);
            IndexIVFFlat index(&quantizer, d, nlist);

            // 训练
            index.train(ntrain, train_vectors);
            index.add(ntrain, train_vectors);

            for (size_t nprobe : nprobe_values) {
                // 测试by_residual两种模式
                for (int by_res : {false, true}) {
                    index.by_residual = by_res;
                    index.nprobe = nprobe;

                    // 测量性能
                    auto t0 = std::chrono::high_resolution_clock::now();

                    std::vector<float> distances(nquery * 100);
                    std::vector<idx_t> labels(nquery * 100);
                    index.search(nquery, query_vectors, 100,
                                distances.data(), labels.data());

                    auto t1 = std::chrono::high_resolution_clock::now();
                    double time_ms =
                        std::chrono::duration<double>(t1 - t0).count() * 1000;
                    double qps = nquery * 1000.0 / time_ms;

                    // 计算recall
                    float recall = compute_recall(
                        nquery, 100, labels.data(), ground_truth);

                    results.push_back({nlist, nprobe, by_res, qps, recall});
                }
            }
        }
    }

    Config find_optimal(float target_recall, double min_qps) {
        auto best = std::min_element(
                results.begin(), results.end(),
                [&](const Config& a, const Config& b) {
                    // 过滤不满足条件的配置
                    if (a.recall < target_recall || a.qps < min_qps) {
                        return false;
                    }
                    if (b.recall < target_recall || b.qps < min_qps) {
                        return true;
                    }
                    // 最大化QPS
                    return a.qps > b.qps;
                });

        return *best;
    }

    void print_results() {
        printf("%-10s %-10s %-10s %-10s %-10s\n",
               "nlist", "nprobe", "by_res", "QPS", "Recall");
        printf("--------------------------------------------\n");

        for (const auto& r : results) {
            printf("%-10zu %-10zu %-10d %-10.0f %-10.3f\n",
                   r.nlist, r.nprobe, r.by_residual,
                   r.qps, r.recall);
        }
    }
};
```

### 14.2 实时性能监控

```cpp
// 实时监控IVF搜索性能
struct IVFPerformanceMonitor {
    std::atomic<uint64_t> total_queries{0};
    std::atomic<uint64_t> total_scanned{0};
    std::atomic<double> total_time_ms{0};

    std::array<double, 100> recent_latencies;
    std::atomic<size_t> latency_ptr{0};

    void record_query(double latency_ms, size_t n_scanned) {
        total_queries++;
        total_scanned += n_scanned;
        total_time_ms += latency_ms;

        // 记录延迟（循环缓冲）
        size_t pos = latency_ptr.fetch_add(1) % recent_latencies.size();
        recent_latencies[pos] = latency_ms;
    }

    void print_stats() const {
        uint64_t nq = total_queries.load();
        double time = total_time_ms.load();

        if (nq == 0) {
            printf("No queries recorded\n");
            return;
        }

        printf("=== IVF Performance Stats ===\n");
        printf("Total queries: %lu\n", nq);
        printf("Total scanned: %lu\n", total_scanned.load());
        printf("Avg scanned per query: %.1f\n",
               (double)total_scanned.load() / nq);
        printf("Avg latency: %.3f ms\n", time / nq);
        printf("QPS: %.1f\n", nq * 1000.0 / time);

        // 计算延迟百分位数
        std::vector<double> sorted_latencies(
            recent_latencies.begin(),
            recent_latencies.begin() +
                std::min(latency_ptr.load(), recent_latencies.size()));

        std::sort(sorted_latencies.begin(), sorted_latencies.end());

        if (!sorted_latencies.empty()) {
            printf("P50 latency: %.3f ms\n",
                   sorted_latencies[sorted_latencies.size() / 2]);
            printf("P99 latency: %.3f ms\n",
                   sorted_latencies[sorted_latencies.size() * 99 / 100]);
            printf("P99.9 latency: %.3f ms\n",
                   sorted_latencies[sorted_latencies.size() * 999 / 1000]);
        }
    }
};
```

### 14.3 内存使用分析

```cpp
// IVF索引内存占用分析
struct IVFMemoryAnalyzer {
    static size_t calculate_memory(const IndexIVF& index) {
        size_t total = 0;

        // 1. 粗量化器内存
        total += index.quantizer->calculate_memory();

        // 2. 倒排列表内存
        for (size_t list_no = 0; list_no < index.nlist; list_no++) {
            size_t list_size = index.invlists->list_size(list_no);

            // IDs内存
            total += list_size * sizeof(idx_t);

            // Codes内存
            total += list_size * index.code_size;
        }

        // 3. DirectMap内存
        if (!index.direct_map.no()) {
            if (index.direct_map.type == DirectMap::Array) {
                total += index.ntotal * sizeof(uint64_t);
            } else if (index.direct_map.type == DirectMap::Hashtable) {
                // 哈希表大小估计
                total += index.ntotal * 2 * (sizeof(uint64_t) + sizeof(idx_t));
            }
        }

        return total;
    }

    static void print_memory_breakdown(const IndexIVF& index) {
        size_t total = calculate_memory(index);

        printf("=== IVF Memory Breakdown ===\n");
        printf("Total memory: %.2f MB\n", total / (1024.0 * 1024.0));
        printf("  Quantizer: %.2f MB\n",
               index.quantizer->calculate_memory() / (1024.0 * 1024.0));

        size_t invlists_memory = 0;
        size_t total_vectors = 0;

        for (size_t list_no = 0; list_no < index.nlist; list_no++) {
            size_t list_size = index.invlists->list_size(list_no);
            invlists_memory += list_size * (sizeof(idx_t) + index.code_size);
            total_vectors += list_size;
        }

        printf("  InvertedLists: %.2f MB\n",
               invlists_memory / (1024.0 * 1024.0));
        printf("    Total vectors: %zu\n", total_vectors);
        printf("    Avg per vector: %.2f bytes\n",
               (double)invlists_memory / total_vectors);

        // 各个倒排列表的大小分布
        std::vector<size_t> list_sizes;
        for (size_t list_no = 0; list_no < index.nlist; list_no++) {
            list_sizes.push_back(index.invlists->list_size(list_no));
        }

        std::sort(list_sizes.begin(), list_sizes.end(), std::greater<size_t>());

        printf("\nTop 10 largest lists:\n");
        for (int i = 0; i < 10 && i < list_sizes.size(); i++) {
            printf("  %2d: %zu vectors\n", i + 1, list_sizes[i]);
        }

        // 填充率统计
        size_t empty_lists = 0;
        for (size_t size : list_sizes) {
            if (size == 0) {
                empty_lists++;
            }
        }

        printf("\nFill statistics:\n");
        printf("  Empty lists: %zu / %zu (%.1f%%)\n",
               empty_lists, index.nlist,
               100.0 * empty_lists / index.nlist);
        printf("  Avg list size: %.1f\n",
               (double)index.ntotal / index.nlist);
    }
};
```

### 14.4 热点识别工具

```cpp
// 基于perf的IVF热点分析
#ifdef __linux__
struct IVFProfiler {
    static void profile_search(IndexIVF& index,
                              const float* queries,
                              idx_t nq,
                              idx_t k) {

        pid_t pid = getpid();

        // 启动perf记录
        char perf_cmd[256];
        snprintf(perf_cmd, sizeof(perf_cmd),
                "perf record -g -p %d --call-graph dwarf sleep 30",
                pid);

        system(perf_cmd);

        // 执行搜索
        std::vector<float> distances(nq * k);
        std::vector<idx_t> labels(nq * k);

        index.search(nq, queries, k,
                    distances.data(), labels.data());

        printf("Profile data saved. View with:\n");
        printf("  perf report -i perf.data\n");
    }

    static void profile_cache_misses(IndexIVF& index,
                                      const float* queries,
                                      idx_t nq,
                                      idx_t k) {

        pid_t pid = getpid();

        // 使用perf stat统计缓存未命中
        char perf_cmd[512];
        snprintf(perf_cmd, sizeof(perf_cmd),
                "perf stat -p %d -e cache-misses,cache-references,"
                "L1-dcache-load-misses,L1-dcache-loads,LLC-load-misses,LLC-loads "
                "-- sleep 10",
                pid);

        // 在后台启动perf
        int perf_pid = fork();
        if (perf_pid == 0) {
            system(perf_cmd);
            exit(0);
        }

        // 执行搜索
        std::vector<float> distances(nq * k);
        std::vector<idx_t> labels(nq * k);

        for (int i = 0; i < 10; i++) {
            index.search(nq, queries, k,
                        distances.data(), labels.data());
        }

        // 等待perf完成
        waitpid(perf_pid, nullptr, 0);
    }
};
#endif
```

---

## 15. IVF底层SIMD与并行优化深入

### 15.1 SIMD优化的倒排列表扫描

```cpp
// SIMD优化的批量向量比较
// 一次处理多个查询或多个数据库向量

// AVX2优化的L2距离扫描
void scan_inverted_list_avx2_l2(
        const float* query,
        const float* list_vectors,
        size_t list_size,
        size_t d,
        float* heap_dist,
        idx_t* heap_ids,
        size_t k) {

    // 4路展开：每次处理4个向量
    size_t j = 0;

    for (; j + 4 <= list_size; j += 4) {
        const float* v0 = list_vectors + (j + 0) * d;
        const float* v1 = list_vectors + (j + 1) * d;
        const float* v2 = list_vectors + (j + 2) * d;
        const float* v3 = list_vectors + (j + 3) * d;

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        // 计算与4个向量的L2距离
        for (size_t dim = 0; dim < d; dim++) {
            __m256 qdim = _mm256_set1_ps(query[dim]);
            __m256 v0dim = _mm256_set1_ps(v0[dim]);
            __m256 v1dim = _mm256_set1_ps(v1[dim]);
            __m256 v2dim = _mm256_set1_ps(v2[dim]);
            __m256 v3dim = _mm256_set1_ps(v3[dim]);

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
        float dist0 = hsum256_ps(sum0);
        float dist1 = hsum256_ps(sum1);
        float dist2 = hsum256_ps(sum2);
        float dist3 = hsum256_ps(sum3);

        // 更新堆
        if (CMax<float, idx_t>::cmp(dist0, heap_dist[0])) {
            heap_replace_top<CMax<float, idx_t>>(k, heap_dist, heap_ids, dist0, j + 0);
        }
        if (CMax<float, idx_t>::cmp(dist1, heap_dist[0])) {
            heap_replace_top<CMax<float, idx_t>>(k, heap_dist, heap_ids, dist1, j + 1);
        }
        if (CMax<float, idx_t>::cmp(dist2, heap_dist[0])) {
            heap_replace_top<CMax<float, idx_t>>(k, heap_dist, heap_ids, dist2, j + 2);
        }
        if (CMax<float, idx_t>::cmp(dist3, heap_dist[0])) {
            heap_replace_top<CMax<float, idx_t>>(k, heap_dist, heap_ids, dist3, j + 3);
        }
    }

    // 处理剩余向量
    for (; j < list_size; j++) {
        float dis = fvec_L2sqr(query, list_vectors + j * d, d);
        if (CMax<float, idx_t>::cmp(dis, heap_dist[0])) {
            heap_replace_top<CMax<float, idx_t>>(k, heap_dist, heap_ids, dis, j);
        }
    }
}

// 内积的SIMD扫描
void scan_inverted_list_avx2_ip(
        const float* query,
        const float* list_vectors,
        size_t list_size,
        size_t d,
        float* heap_simi,
        idx_t* heap_idxi,
        size_t k) {

    size_t j = 0;

    // 4路展开
    for (; j + 4 <= list_size; j += 4) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        for (size_t dim = 0; dim < d; dim++) {
            __m256 qdim = _mm256_set1_ps(query[dim]);

            __m256 v0dim = _mm256_set1_ps(list_vectors[(j + 0) * d + dim]);
            __m256 v1dim = _mm256_set1_ps(list_vectors[(j + 1) * d + dim]);
            __m256 v2dim = _mm256_set1_ps(list_vectors[(j + 2) * d + dim]);
            __m256 v3dim = _mm256_set1_ps(list_vectors[(j + 3) * d + dim]);

            sum0 = _mm256_fmadd_ps(qdim, v0dim, sum0);
            sum1 = _mm256_fmadd_ps(qdim, v1dim, sum1);
            sum2 = _mm256_fmadd_ps(qdim, v2dim, sum2);
            sum3 = _mm256_fmadd_ps(qdim, v3dim, sum3);
        }

        float ip0 = hsum256_ps(sum0);
        float ip1 = hsum256_ps(sum1);
        float ip2 = hsum256_ps(sum2);
        float ip3 = hsum256_ps(sum3);

        // 内积使用CMin（找最大值）
        if (CMin<float, idx_t>::cmp(ip0, heap_simi[0])) {
            heap_replace_top<CMin<float, idx_t>>(k, heap_simi, heap_idxi, ip0, j + 0);
        }
        if (CMin<float, idx_t>::cmp(ip1, heap_simi[0])) {
            heap_replace_top<CMin<float, idx_t>>(k, heap_simi, heap_idxi, ip1, j + 1);
        }
        if (CMin<float, idx_t>::cmp(ip2, heap_simi[0])) {
            heap_replace_top<CMin<float, idx_t>>(k, heap_simi, heap_idxi, ip2, j + 2);
        }
        if (CMin<float, idx_t>::cmp(ip3, heap_simi[0])) {
            heap_replace_top<CMin<float, idx_t>>(k, heap_simi, heap_idxi, ip3, j + 3);
        }
    }

    for (; j < list_size; j++) {
        float ip = fvec_inner_product(query, list_vectors + j * d, d);
        if (CMin<float, idx_t>::cmp(ip, heap_simi[0])) {
            heap_replace_top<CMin<float, idx_t>>(k, heap_simi, heap_idxi, ip, j);
        }
    }
}
```

### 15.2 AVX-512优化的粗量化器搜索

```cpp
#ifdef __AVX512F__
// AVX-512优化的粗量化器搜索
// 快速找到查询最近的nprobe个质心
void search_coarse_quantizer_avx512(
        const float* query,
        const float* centroids,
        size_t nlist,
        size_t d,
        idx_t* list_nos,
        float* coarse_dis,
        size_t nprobe) {

    // 初始化堆
    heap_heapify<CMax<float, idx_t>>(nprobe, coarse_dis, list_nos);

    // 16路并行：每次处理16个质心
    size_t c = 0;

    for (; c + 16 <= nlist; c += 16) {
        __m512 sum = _mm512_setzero_ps();

        // 计算与16个质心的距离
        for (size_t dim = 0; dim < d; dim++) {
            __m512 qdim = _mm512_set1_ps(query[dim]);

            // 加载16个质心的第dim维
            __m512 cdim = _mm512_loadu_ps(centroids + c * d + dim);

            __m512 diff = _mm512_sub_ps(qdim, cdim);
            sum = _mm512_fmadd_ps(diff, diff, sum);
        }

        // 提取16个距离值
        alignas(64) float dists[16];
        _mm512_storeu_ps(dists, sum);

        // 更新堆
        for (int i = 0; i < 16; i++) {
            if (CMax<float, idx_t>::cmp(dists[i], coarse_dis[0])) {
                heap_replace_top<CMax<float, idx_t>>(
                    nprobe, coarse_dis, list_nos, dists[i], c + i);
            }
        }
    }

    // 处理剩余质心
    for (; c < nlist; c++) {
        float dis = fvec_L2sqr(query, centroids + c * d, d);
        if (CMax<float, idx_t>::cmp(dis, coarse_dis[0])) {
            heap_replace_top<CMax<float, idx_t>>(
                nprobe, coarse_dis, list_nos, dis, c);
        }
    }

    // 堆排序
    heap_reorder<CMax<float, idx_t>>(nprobe, coarse_dis, list_nos);
}

// 批量粗量化器搜索（多查询）
void batch_search_coarse_quantizer_avx512(
        const float* queries,
        size_t nq,
        const float* centroids,
        size_t nlist,
        size_t d,
        idx_t* all_list_nos,
        float* all_coarse_dis,
        size_t nprobe) {

    #pragma omp parallel for
    for (size_t q = 0; q < nq; q++) {
        idx_t* list_nos = all_list_nos + q * nprobe;
        float* coarse_dis = all_coarse_dis + q * nprobe;

        search_coarse_quantizer_avx512(
            queries + q * d, centroids, nlist, d,
            list_nos, coarse_dis, nprobe);
    }
}
#endif
```

### 15.3 NUMA感知的IVF索引

```cpp
// NUMA（Non-Uniform Memory Access）优化的IVF实现
#ifdef __linux__

#include <numa.h>

class NUMAAwareInvertedLists : public InvertedLists {
    struct NUMANode {
        int node_id;
        std::unique_ptr<InvertedLists> invlists;
        std::vector<size_t> list_mapping;  // list_no -> node assignment
    };

    std::vector<NUMANode> nodes;
    size_t nlist;
    size_t code_size;

public:
    NUMAAwareInvertedLists(size_t nlist, size_t code_size)
        : nlist(nlist), code_size(code_size) {

        int max_node = numa_max_node() + 1;

        for (int node = 0; node < max_node; node++) {
            NUMANode n;
            n.node_id = node;
            n.invlists = std::make_unique<ArrayInvertedLists>(nlist, code_size);

            // 轮询分配list到NUMA节点
            for (size_t list_no = 0; list_no < nlist; list_no++) {
                if (list_no % max_node == node) {
                    n.list_mapping.push_back(list_no);
                }
            }

            nodes.push_back(std::move(n));
        }
    }

    size_t add_entries(
            size_t list_no,
            size_t n,
            const idx_t* xids,
            const uint8_t* xcode) override {

        // 找到负责该list的NUMA节点
        int node_id = list_no % nodes.size();
        NUMANode& node = nodes[node_id];

        return node.invlists->add_entries(list_no, n, xids, xcode);
    }

    // 获取list数据（可能需要跨NUMA节点访问）
    size_t list_size(size_t list_no) const override {
        int node_id = list_no % nodes.size();
        return nodes[node_id].invlists->list_size(list_no);
    }

    const uint8_t* get_codes(size_t list_no) const override {
        int node_id = list_no % nodes.size();
        return nodes[node_id].invlists->get_codes(list_no);
    }

    const idx_t* get_ids(size_t list_no) const override {
        int node_id = list_no % nodes.size();
        return nodes[node_id].invlists->get_ids(list_no);
    }

    // NUMA感知的批量添加
    void add_entries_numa_aware(
            size_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* list_nos) {

        // 按NUMA节点分组
        std::vector<std::vector<std::pair<size_t, idx_t>>> node_groups(
            nodes.size());

        for (size_t i = 0; i < n; i++) {
            int node_id = list_nos[i] % nodes.size();
            node_groups[node_id].push_back({i, list_nos[i]});
        }

        // 并行添加到各NUMA节点
        #pragma omp parallel
        {
            // 绑定当前线程到NUMA节点
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();

            for (int node_id = 0; node_id < nodes.size(); node_id++) {
                if (thread_id % num_threads == node_id % num_threads) {
                    // 处理该NUMA节点的数据
                    numa_set_preferred(node_id);

                    for (auto [i, list_no] : node_groups[node_id]) {
                        nodes[node_id].invlists->add_entries(
                            list_no, 1, xids + i, nullptr);
                    }
                }
            }
        }
    }
};

// NUMA感知的搜索
void search_ivf_numa_aware(
        const IndexIVF& index,
        const float* queries,
        size_t nq,
        size_t k,
        float* distances,
        idx_t* labels) {

    // 为每个查询找到最近的nprobe个list
    std::vector<idx_t> all_list_nos(nq * index.nprobe);
    std::vector<float> all_coarse_dis(nq * index.nprobe);

    #pragma omp parallel for
    for (size_t q = 0; q < nq; q++) {
        index.quantizer->search(
            1, queries + q * index.d, index.nprobe,
            all_coarse_dis.data() + q * index.nprobe,
            all_list_nos.data() + q * index.nprobe);
    }

    // 按NUMA节点分组处理倒排列表
    for (size_t q = 0; q < nq; q++) {
        float* simi = distances + q * k;
        idx_t* idxi = labels + q * k;

        heap_heapify<CMax<float, idx_t>>(k, simi, idxi);

        idx_t* list_nos = all_list_nos.data() + q * index.nprobe;

        for (size_t p = 0; p < index.nprobe; p++) {
            idx_t list_no = list_nos[p];

            // 绑定到负责该list的NUMA节点
            int node_id = list_no % numa_num_configured_nodes();
            numa_run_on_node(node_id);

            // 扫描倒排列表
            size_t list_size = index.invlists->list_size(list_no);
            const uint8_t* codes = index.invlists->get_codes(list_no);
            const idx_t* ids = index.invlists->get_ids(list_no);

            // 使用SIMD优化的扫描
            scan_inverted_list_avx2_l2(
                queries + q * index.d,
                reinterpret_cast<const float*>(codes),
                list_size, index.d, simi, idxi, k);
        }

        heap_reorder<CMax<float, idx_t>>(k, simi, idxi);
    }
}
#endif
```

### 15.4 缓存优化的倒排列表布局

```cpp
// 缓存行优化的倒排列表存储
struct CacheLineAwareInvertedLists : public InvertedLists {
    static constexpr size_t CACHE_LINE_SIZE = 64;

    struct alignas(CACHE_LINE_SIZE) CacheLine {
        uint8_t codes[16];  // 假设code_size=16
        idx_t ids[16 / sizeof(idx_t)];
    };

    std::vector<std::vector<CacheLine>> lists;
    size_t nlist;
    size_t code_size;

    CacheLineAwareInvertedLists(size_t nlist, size_t code_size)
        : nlist(nlist), code_size(code_size) {
        lists.resize(nlist);
    }

    size_t add_entries(
            size_t list_no,
            size_t n,
            const idx_t* xids,
            const uint8_t* xcode) override {

        size_t offset = lists[list_no].size();
        size_t n_lines = (n + 15) / 16;

        for (size_t line = 0; line < n_lines; line++) {
            CacheLine cl;
            memset(&cl, 0, sizeof(CacheLine));

            size_t start = line * 16;
            size_t end = std::min(start + 16, n);

            for (size_t i = start; i < end; i++) {
                size_t idx_in_line = i - start;

                // 复制code
                memcpy(cl.codes + idx_in_line * code_size,
                       xcode + i * code_size, code_size);

                // 复制ID
                cl.ids[idx_in_line] = xids ? xids[i] : offset + i;
            }

            lists[list_no].push_back(cl);
        }

        return offset;
    }

    // 缓存友好的扫描
    size_t scan_cached(
            size_t list_no,
            const float* query,
            float* simi,
            idx_t* idxi,
            size_t k,
            const std::function<float(const float*, const uint8_t*)>& dist_func) const {

        size_t nup = 0;

        for (const CacheLine& line : lists[list_no]) {
            // 一次处理16个向量（正好一个缓存行）

            for (size_t i = 0; i < 16; i++) {
                const uint8_t* code = line.codes + i * code_size;
                idx_t id = line.ids[i];

                float dis = dist_func(query, code);

                if (CMax<float, idx_t>::cmp(dis, simi[0])) {
                    heap_replace_top<CMax<float, idx_t>>(k, simi, idxi, dis, id);
                    nup++;
                }
            }
        }

        return nup;
    }
};
```

### 15.5 多粒度并行IVF搜索

```cpp
// 三级并行IVF搜索
// 级别1: 查询级并行
// 级别2: 倒排列表级并行
// 级别3: 向量级并行（SIMD）

void ivf_search_three_level_parallel(
        const IndexIVF& index,
        const float* queries,
        size_t nq,
        size_t k,
        float* distances,
        idx_t* labels) {

    // 级别1: 查询级并行（最外层）
    #pragma omp parallel for schedule(dynamic)
    for (size_t q = 0; q < nq; q++) {
        const float* query = queries + q * index.d;
        float* simi = distances + q * k;
        idx_t* idxi = labels + q * k;

        heap_heapify<CMax<float, idx_t>>(k, simi, idxi);

        // 找到最近的nprobe个列表
        idx_t list_nos[256];
        float coarse_dis[256];

        index.quantizer->search(
            1, query, index.nprobe, coarse_dis, list_nos);

        // 级别2: 倒排列表级并行
        // 使用critical section保护堆更新
        #pragma omp parallel for
        for (size_t p = 0; p < index.nprobe; p++) {
            idx_t list_no = list_nos[p];
            size_t list_size = index.invlists->list_size(list_no);

            if (list_size == 0) continue;

            const uint8_t* codes = index.invlists->get_codes(list_no);
            const idx_t* ids = index.invlists->get_ids(list_no);

            // 局部堆（避免锁竞争）
            alignas(64) float local_simi[128];
            alignas(64) idx_t local_idxi[128];
            size_t local_k = std::min(k, size_t(128));

            heap_heapify<CMax<float, idx_t>>(local_k, local_simi, local_idxi);

            // 级别3: SIMD向量级并行
            scan_inverted_list_avx2_l2(
                query, reinterpret_cast<const float*>(codes),
                list_size, index.d, local_simi, local_idxi, local_k);

            // 合并到全局堆
            #pragma omp critical
            {
                for (size_t i = 0; i < local_k; i++) {
                    if (CMax<float, idx_t>::cmp(local_simi[i], simi[0])) {
                        heap_replace_top<CMax<float, idx_t>>(
                            k, simi, idxi, local_simi[i], local_idxi[i]);
                    }
                }
            }
        }

        heap_reorder<CMax<float, idx_t>>(k, simi, idxi);
    }
}

// 无锁并行搜索（使用原子操作）
void ivf_search_lock_free(
        const IndexIVF& index,
        const float* queries,
        size_t nq,
        size_t k,
        float* distances,
        idx_t* labels) {

    // 使用计数器实现工作窃取
    std::atomic<size_t> next_query(0);
    std::atomic<size_t> next_list[256];

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        while (true) {
            // 获取下一个查询
            size_t q = next_query.fetch_add(1);

            if (q >= nq) break;

            const float* query = queries + q * index.d;
            float* simi = distances + q * k;
            idx_t* idxi = labels + q * k;

            heap_heapify<CMax<float, idx_t>>(k, simi, idxi);

            // 找到最近的nprobe个列表
            idx_t list_nos[256];
            float coarse_dis[256];

            index.quantizer->search(
                1, query, index.nprobe, coarse_dis, list_nos);

            // 初始化列表计数器
            if (tid == 0) {
                for (size_t p = 0; p < index.nprobe; p++) {
                    next_list[p].store(0);
                }
            }
            #pragma omp barrier

            // 每个线程处理一部分倒排列表
            while (true) {
                bool found_work = false;

                for (size_t p = 0; p < index.nprobe; p++) {
                    idx_t list_no = list_nos[p];
                    size_t list_size = index.invlists->list_size(list_no);

                    if (list_size == 0) continue;

                    // 原子获取下一批向量
                    size_t batch_size = 1024;
                    size_t start = next_list[p].fetch_add(batch_size);

                    if (start >= list_size) continue;
                    found_work = true;

                    size_t end = std::min(start + batch_size, list_size);

                    const uint8_t* codes = index.invlists->get_codes(list_no);
                    const idx_t* ids = index.invlists->get_ids(list_no);

                    const float* list_vecs =
                        reinterpret_cast<const float*>(codes) + start * index.d;

                    // 扫描这批向量
                    for (size_t j = start; j < end; j++) {
                        const float* vec = reinterpret_cast<const float*>(
                            codes) + j * index.d;
                        idx_t id = ids[j];

                        float dis = fvec_L2sqr(query, vec, index.d);

                        if (CMax<float, idx_t>::cmp(dis, simi[0])) {
                            heap_replace_top<CMax<float, idx_t>>(
                                k, simi, idxi, dis, id);
                        }
                    }
                }

                if (!found_work) break;
            }

            heap_reorder<CMax<float, idx_t>>(k, simi, idxi);
        }
    }
}
```

### 15.6 SIMD优化的IVFPQ搜索

```cpp
// IVFPQ: 倒排列表+PQ编码的组合索引
// SIMD优化的PQ距离计算

// SIMD优化的ADC距离计算
inline float ivfpq_compute_distance_avx2(
        const float* query_dis_table,  // M x ksub
        const uint8_t* pq_code,        // M bytes
        size_t M,
        size_t ksub) {

    __m256 sum = _mm256_setzero_ps();
    size_t m = 0;

    // 8路展开：每次处理8个子量化器
    for (; m + 8 <= M; m += 8) {
        // 加载8个PQ索引
        __m128i idx8 = _mm_loadl_epi64((__m128i*)(pq_code + m));

        // 扩展为32位
        __m256i idx0 = _mm256_cvtepu8_epi32(idx8);
        __m256i idx1 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx8, 4));

        // 计算表偏移
        __m256i offset0 = _mm256_add_epi32(
            _mm256_set1_epi32((m + 0) * ksub), idx0);
        __m256i offset1 = _mm256_add_epi32(
            _mm256_set1_epi32((m + 4) * ksub), idx1);

        // Gather距离值
        __m256 dist0 = _mm256_i32gather_ps(query_dis_table, offset0, 4);
        __m256 dist1 = _mm256_i32gather_ps(query_dis_table, offset1, 4);

        // 合并
        __m256 dist = _mm256_permute2f128_ps(dist0, dist1, 0x20);
        sum = _mm256_add_ps(sum, dist);
    }

    float result = hsum256_ps(sum);

    // 处理剩余子量化器
    for (; m < M; m++) {
        uint8_t idx = pq_code[m];
        result += query_dis_table[m * ksub + idx];
    }

    return result;
}

// IVFPQ扫描器
struct IVFPQScannerAVX2 : InvertedListScanner {
    const float* query_dis_table;
    const ProductQuantizer* pq;
    size_t M;
    size_t ksub;

    void set_query(const float* query) override {
        // 预计算距离表
        query_dis_table = new float[M * ksub];

        for (size_t m = 0; m < M; m++) {
            const float* qm = query + m * pq->dsub;
            float* dt = query_dis_table + m * ksub;

            for (size_t k = 0; k < ksub; k++) {
                const float* ck = pq->get_centroids(m, k);
                dt[k] = fvec_L2sqr(qm, ck, pq->dsub);
            }
        }
    }

    float distance_to_code(const uint8_t* code) const override {
        return ivfpq_compute_distance_avx2(
            query_dis_table, code, M, ksub);
    }

    ~IVFPQScannerAVX2() {
        delete[] query_dis_table;
    }
};
```

### 15.7 预取优化的IVF搜索

```cpp
// 软件预取优化的IVF搜索
void ivf_search_with_prefetch(
        const IndexIVF& index,
        const float* query,
        float* simi,
        idx_t* idxi,
        size_t k) {

    heap_heapify<CMax<float, idx_t>>(k, simi, idxi);

    // 找到最近的nprobe个列表
    idx_t list_nos[256];
    float coarse_dis[256];

    index.quantizer->search(1, query, index.nprobe, coarse_dis, list_nos);

    constexpr size_t PREFETCH_AHEAD = 2;

    for (size_t p = 0; p < index.nprobe; p++) {
        idx_t list_no = list_nos[p];

        // 预取下一个列表的数据
        if (p + PREFETCH_AHEAD < index.nprobe) {
            idx_t next_list_no = list_nos[p + PREFETCH_AHEAD];
            size_t next_list_size = index.invlists->list_size(next_list_no);

            if (next_list_size > 0) {
                const uint8_t* next_codes = index.invlists->get_codes(next_list_no);
                const idx_t* next_ids = index.invlists->get_ids(next_list_no);

                // 预取前几个缓存行
                for (size_t i = 0; i < 4 && i * 64 < next_list_size * index.code_size; i++) {
                    _mm_prefetch((const char*)(next_codes + i * 64), _MM_HINT_T0);
                }

                for (size_t i = 0; i < 4 && i * 64 < next_list_size * sizeof(idx_t); i++) {
                    _mm_prefetch((const char*)(next_ids + i * 64), _MM_HINT_T0);
                }
            }
        }

        // 处理当前列表
        size_t list_size = index.invlists->list_size(list_no);
        const uint8_t* codes = index.invlists->get_codes(list_no);
        const idx_t* ids = index.invlists->get_ids(list_no);

        // 批量扫描
        for (size_t j = 0; j < list_size; j++) {
            // 预取未来的向量
            if (j + 8 < list_size) {
                _mm_prefetch((const char*)(codes + (j + 8) * index.code_size), _MM_HINT_T0);
                _mm_prefetch((const char*)(ids + (j + 8)), _MM_HINT_T0);
            }

            const uint8_t* code = codes + j * index.code_size;
            idx_t id = ids[j];

            float dis = compute_single_distance(query, code, index.code_size);

            if (CMax<float, idx_t>::cmp(dis, simi[0])) {
                heap_replace_top<CMax<float, idx_t>>(k, simi, idxi, dis, id);
            }
        }
    }

    heap_reorder<CMax<float, idx_t>>(k, simi, idxi);
}
```

### 15.8 动态负载均衡的IVF搜索

```cpp
// 动态负载均衡：根据倒排列表大小动态分配工作
void ivf_search_dynamic_load_balance(
        const IndexIVF& index,
        const float* queries,
        size_t nq,
        size_t k,
        float* distances,
        idx_t* labels) {

    // 统计所有倒排列表的大小
    std::vector<size_t> list_sizes(index.nlist);
    size_t total_size = 0;

    for (size_t list_no = 0; list_no < index.nlist; list_no++) {
        list_sizes[list_no] = index.invlists->list_size(list_no);
        total_size += list_sizes[list_no];
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // 动态调度：每个线程获取下一块工作
        size_t chunk_start = 0;
        size_t chunk_size = 10000;  // 每块处理的向量数

        while (chunk_start < total_size) {
            // 每个线程获取一个chunk
            size_t my_start = chunk_start + tid * chunk_size;

            if (my_start >= total_size) {
                #pragma omp barrier
                chunk_start += nthreads * chunk_size;
                continue;
            }

            size_t my_end = std::min(my_start + chunk_size, total_size);

            // 找到my_start对应的(list_no, offset)
            size_t acc = 0;
            size_t list_no = 0;
            size_t offset = 0;

            for (list_no = 0; list_no < index.nlist; list_no++) {
                if (acc + list_sizes[list_no] > my_start) {
                    offset = my_start - acc;
                    break;
                }
                acc += list_sizes[list_no];
            }

            // 处理这个chunk
            for (size_t pos = my_start; pos < my_end; ) {
                while (offset >= list_sizes[list_no]) {
                    offset = 0;
                    list_no++;
                }

                size_t batch_end = std::min(
                    my_end,
                    acc + list_sizes[list_no] - offset);

                // 处理当前列表的一部分
                for (size_t j = offset; j < batch_end; j++) {
                    // ... 执行距离计算和堆更新
                }

                pos = batch_end;
                offset = batch_end;
            }

            #pragma omp barrier
            chunk_start += nthreads * chunk_size;
        }
    }
}
```

---

## 练习题

1. 实现简单的IVF索引
2. 研究nlist和nprobe对性能的影响
3. 实现自定义InvertedListScanner
4. 比较IndexIVFFlat和IndexIVFPQ的性能
5. 实现SIMD优化的倒排列表扫描
6. 分析IVF的内存布局优化效果
7. 实现NUMA感知的IVF索引

## 扩展阅读

- faiss/IndexIVF.h - IVF基类
- faiss/IndexIVFFlat.h - IVFFlat实现
- faiss/invlists/InvertedLists.h - 倒排列表接口
- faiss/utils/distances_simd.cpp - SIMD距离计算
- [IVF论文](https://arxiv.org/abs/1603.09320)
- [倒排索引优化技术](https://en.wikipedia.org/wiki/Inverted_index)
