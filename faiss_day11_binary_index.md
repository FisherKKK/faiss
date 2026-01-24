# Faiss深度课程 - 第11天：二进制索引 - Hamming距离与二进制搜索

## 课程目标

深入理解二进制索引（Binary Index），学习Hamming距离计算和针对二进制向量优化的索引结构。

---

## 1. 二进制索引概述

### 1.1 应用场景

二进制索引适用于以下场景：
- 哈希函数输出
- 神经网络二进制特征
- 压缩的浮点向量
- 图像特征（局部二进制模式）

### 1.2 IndexBinary结构

```cpp
// faiss/IndexBinary.h
struct IndexBinary {
    using component_t = uint8_t;   // 二进制分量
    using distance_t = int32_t;    // 距离类型（汉明距离）

    int d;              // 位数
    int code_size;      // 字节数 = d / 8
    idx_t ntotal;       // 总向量数

    MetricType metric_type;

    IndexBinary(idx_t d = 0, MetricType metric = METRIC_L2)
        : d(d), code_size((d + 7) / 8), metric_type(metric) {}

    // 核心接口
    virtual void add(idx_t n, const uint8_t* x) = 0;

    virtual void search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,  // 汉明距离
        idx_t* labels,
        const SearchParameters* params = nullptr) const = 0;
};
```

---

## 2. Hamming距离

### 2.1 基本定义

```cpp
// Hamming距离：两个等长字符串对应位置不同字符的个数
// 对于二进制向量：异或后统计1的个数

int32_t hamming_distance(const uint8_t* a, const uint8_t* b, size_t code_size) {
    int32_t dis = 0;

    for (size_t i = 0; i < code_size; i++) {
        uint8_t x = a[i] ^ b[i];  // 异或

        // 统计1的位数
        while (x) {
            dis += x & 1;
            x >>= 1;
        }
    }

    return dis;
}
```

### 2.2 POPCNT优化

```cpp
// 使用CPU指令加速统计位数

// 方法1：使用__builtin_popcount
int32_t hamming_distance_popcnt(const uint8_t* a, const uint8_t* b,
                                  size_t code_size) {
    int32_t dis = 0;

    for (size_t i = 0; i < code_size; i++) {
        uint8_t x = a[i] ^ b[i];
        dis += __builtin_popcount(x);  // 单指令
    }

    return dis;
}

// 方法2：使用SSE/AVX的POPCNT指令
int32_t hamming_distance_avx2(const uint8_t* a, const uint8_t* b,
                               size_t code_size) {
    int32_t dis = 0;
    size_t i = 0;

    // 处理32字节的块（AVX2）
    for (; i + 32 <= code_size; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));

        __m256i vxor = _mm256_xor_si256(va, vb);

        // 统计位数（需要拆分调用popcount）
        alignas(32) uint8_t tmp[32];
        _mm256_storeu_si256((__m256i*)tmp, vxor);

        for (int j = 0; j < 32; j++) {
            dis += __builtin_popcount(tmp[j]);
        }
    }

    // 处理剩余字节
    for (; i < code_size; i++) {
        dis += __builtin_popcount(a[i] ^ b[i]);
    }

    return dis;
}
```

### 2.3 批量Hamming距离计算

```cpp
// 计算一个查询与多个数据库向量的距离
void batch_hamming_distance(
        const uint8_t* query,
        const uint8_t* database,
        size_t n,
        size_t code_size,
        int32_t* distances) {

    for (size_t i = 0; i < n; i++) {
        distances[i] = hamming_distance_popcnt(
            query,
            database + i * code_size,
            code_size);
    }
}

// SIMD优化的批量计算
void batch_hamming_distance_simd(
        const uint8_t* query,
        const uint8_t* database,
        size_t n,
        size_t code_size,
        int32_t* distances) {

    // 展开循环以提高ILP
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        distances[i + 0] = hamming_distance_popcnt(
            query, database + (i + 0) * code_size, code_size);
        distances[i + 1] = hamming_distance_popcnt(
            query, database + (i + 1) * code_size, code_size);
        distances[i + 2] = hamming_distance_popcnt(
            query, database + (i + 2) * code_size, code_size);
        distances[i + 3] = hamming_distance_popcnt(
            query, database + (i + 3) * code_size, code_size);
    }

    for (; i < n; i++) {
        distances[i] = hamming_distance_popcnt(
            query, database + i * code_size, code_size);
    }
}
```

---

## 3. IndexBinaryFlat

### 3.1 结构与实现

```cpp
// faiss/IndexBinaryFlat.h
struct IndexBinaryFlat : IndexBinary {
    std::vector<uint8_t> codes;  // 所有向量

    IndexBinaryFlat(idx_t d) : IndexBinary(d) {
        code_size = (d + 7) / 8;
    }

    void add(idx_t n, const uint8_t* x) override {
        codes.resize((ntotal + n) * code_size);
        memcpy(codes.data() + ntotal * code_size,
               x, n * code_size);
        ntotal += n;
    }

    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* xq = x + i * code_size;
            int32_t* dis_i = distances + i * k;
            idx_t* lbl_i = labels + i * k;

            // 初始化堆
            heap_heapify<CMax<int32_t, idx_t>>(k, dis_i, lbl_i);

            // 穷举搜索
            for (idx_t j = 0; j < ntotal; j++) {
                int32_t dis = hamming_distance_popcnt(
                    xq, codes.data() + j * code_size, code_size);

                if (CMax<int32_t, idx_t>::cmp(dis, dis_i[0])) {
                    heap_replace_top<CMax<int32_t, idx_t>>(
                        k, dis_i, lbl_i, dis, j);
                }
            }

            heap_reorder<CMax<int32_t, idx_t>>(k, dis_i, lbl_i);
        }
    }
};
```

### 3.2 使用示例

```cpp
void binary_index_example() {
    // 1. 创建浮点向量
    int d_float = 128;
    int n = 10000;
    float* xb_float = new float[n * d_float];

    // 2. 转换为二进制（例如：大于0为1，否则为0）
    int d = d_float;
    uint8_t* xb_binary = new uint8_t[n * ((d + 7) / 8)];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            if (xb_float[i * d_float + j] > 0) {
                int byte_idx = (i * d + j) / 8;
                int bit_idx = 7 - (j % 8);
                xb_binary[byte_idx] |= (1 << bit_idx);
            }
        }
    }

    // 3. 创建索引
    IndexBinaryFlat index(d);
    index.add(n, xb_binary);

    // 4. 搜索
    int nq = 100;
    int k = 10;

    uint8_t* xq_binary = new uint8_t[nq * ((d + 7) / 8)];
    // ... 转换查询向量

    int32_t* distances = new int32_t[nq * k];
    idx_t* labels = new idx_t[nq * k];

    index.search(nq, xq_binary, k, distances, labels);

    delete[] xb_float;
    delete[] xb_binary;
    delete[] xq_binary;
    delete[] distances;
    delete[] labels;
}
```

---

## 4. IndexBinaryIVF

### 4.1 结构

```cpp
struct IndexBinaryIVF : IndexBinary {
    Index* quantizer;    // 粗量化器
    size_t nlist;        // 倒排列表数
    InvertedLists* invlists;

    IndexBinaryIVF(Index* quantizer, idx_t d, size_t nlist)
        : IndexBinary(d),
          quantizer(quantizer),
          nlist(nlist) {}

    void add(idx_t n, const uint8_t* x) override {
        // 1. 分配到倒排列表
        idx_t* list_nos = new idx_t[n];
        quantizer->assign(n, x, list_nos);

        // 2. 添加到倒排列表
        for (idx_t i = 0; i < n; i++) {
            idx_t list_no = list_nos[i];
            const uint8_t* xi = x + i * code_size;
            idx_t id = ntotal + i;

            invlists->add_entry(list_no, id, xi);
        }

        delete[] list_nos;
        ntotal += n;
    }

    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 1. 找到最近的nprobe个列表
        idx_t* assign = new idx_t[n * nprobe];
        quantizer->assign(n, x, assign);

        // 2. 搜索每个列表
#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            int32_t* dis_i = distances + i * k;
            idx_t* lbl_i = labels + i * k;

            heap_heapify<CMax<int32_t, idx_t>>(k, dis_i, lbl_i);

            for (size_t ij = 0; ij < nprobe; ij++) {
                idx_t list_no = assign[i * nprobe + ij];
                size_t list_size = invlists->list_size(list_no);

                const uint8_t* codes = invlists->get_codes(list_no);
                const idx_t* ids = invlists->get_ids(list_no);

                for (size_t j = 0; j < list_size; j++) {
                    int32_t dis = hamming_distance_popcnt(
                        x + i * code_size,
                        codes + j * code_size,
                        code_size);

                    if (CMax<int32_t, idx_t>::cmp(dis, dis_i[0])) {
                        heap_replace_top<CMax<int32_t, idx_t>>(
                            k, dis_i, lbl_i, dis, ids[j]);
                    }
                }
            }

            heap_reorder<CMax<int32_t, idx_t>>(k, dis_i, lbl_i);
        }

        delete[] assign;
    }
};
```

---

## 5. IndexBinaryHash

### 5.1 多索引哈希

```cpp
struct IndexBinaryHash : IndexBinary {
    // LSH（局部敏感哈希）
    int nhash;           // 哈希表数
    int nbits_per_hash;  // 每个哈希的位数

    std::vector<std::unordered_map<uint64_t, std::vector<idx_t>>> hashtables;

    IndexBinaryHash(idx_t d, int nhash, int nbits_per_hash)
        : IndexBinary(d),
          nhash(nhash),
          nbits_per_hash(nbits_per_hash),
          hashtables(nhash) {}

    void add(idx_t n, const uint8_t* x) override {
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* xi = x + i * code_size;

            // 为每个哈希表计算哈希
            for (int h = 0; h < nhash; h++) {
                uint64_t hash = compute_hash(xi, h);

                // 添加到对应的桶
                hashtables[h][hash].push_back(ntotal + i);
            }
        }

        ntotal += n;
    }

    uint64_t compute_hash(const uint8_t* code, int h) {
        // 简单实现：随机选择一些位作为哈希
        uint64_t hash = 0;
        int offset = h * nbits_per_hash;

        for (int i = 0; i < nbits_per_hash; i++) {
            int bit_idx = (offset + i) % d;
            int byte_idx = bit_idx / 8;
            int bit = 7 - (bit_idx % 8);

            if (code[byte_idx] & (1 << bit)) {
                hash |= (1ULL << i);
            }
        }

        return hash;
    }

    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* xq = x + i * code_size;
            int32_t* dis_i = distances + i * k;
            idx_t* lbl_i = labels + i * k;

            heap_heapify<CMax<int32_t, idx_t>>(k, dis_i, lbl_i);

            // 收集候选
            std::unordered_set<idx_t> candidates;

            for (int h = 0; h < nhash; h++) {
                uint64_t hash = compute_hash(xq, h);

                if (hashtables[h].find(hash) != hashtables[h].end()) {
                    for (idx_t id : hashtables[h].at(hash)) {
                        candidates.insert(id);
                    }
                }
            }

            // 计算精确距离
            for (idx_t id : candidates) {
                const uint8_t* code = get_code(id);
                int32_t dis = hamming_distance_popcnt(xq, code, code_size);

                if (CMax<int32_t, idx_t>::cmp(dis, dis_i[0])) {
                    heap_replace_top<CMax<int32_t, idx_t>>(
                        k, dis_i, lbl_i, dis, id);
                }
            }

            heap_reorder<CMax<int32_t, idx_t>>(k, dis_i, lbl_i);
        }
    }
};
```

---

## 6. 浮点向量二值化

### 6.1 简单阈值

```cpp
// 浮点向量转二进制：大于0为1
void float_to_binary_simple(
        const float* x,
        size_t n, size_t d,
        uint8_t* binary) {

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            int byte_idx = (i * d + j) / 8;
            int bit_idx = 7 - (j % 8);

            if (x[i * d + j] > 0) {
                binary[byte_idx] |= (1 << bit_idx);
            }
        }
    }
}
```

### 6.2 均值阈值

```cpp
// 使用均值作为阈值
void float_to_binary_mean(
        const float* x,
        size_t n, size_t d,
        uint8_t* binary) {

    // 计算每维均值
    std::vector<float> means(d, 0.0f);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            means[j] += x[i * d + j];
        }
    }

    for (size_t j = 0; j < d; j++) {
        means[j] /= n;
    }

    // 二值化
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            int byte_idx = (i * d + j) / 8;
            int bit_idx = 7 - (j % 8);

            if (x[i * d + j] > means[j]) {
                binary[byte_idx] |= (1 << bit_idx);
            }
        }
    }
}
```

### 6.3 局部敏感哈希

```cpp
// 使用随机投影的LSH
struct LSHBinary {
    std::vector<std::vector<float>> random_vectors;

    LSHBinary(size_t d, size_t k) {
        // 生成k个随机向量
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0, 1.0);

        for (size_t i = 0; i < k; i++) {
            std::vector<float> rv(d);
            for (size_t j = 0; j < d; j++) {
                rv[j] = distribution(generator);
            }
            random_vectors.push_back(rv);
        }
    }

    void encode(
            const float* x,
            size_t n, size_t d,
            uint8_t* binary,
            size_t k) {

        for (size_t i = 0; i < n; i++) {
            for (size_t h = 0; h < k; h++) {
                // 计算内积
                float ip = 0.0f;
                for (size_t j = 0; j < d; j++) {
                    ip += x[i * d + j] * random_vectors[h][j];
                }

                // 编码
                int byte_idx = (i * k + h) / 8;
                int bit_idx = 7 - ((i * k + h) % 8);

                if (ip > 0) {
                    binary[byte_idx] |= (1 << bit_idx);
                }
            }
        }
    }
};
```

---

## 8. 二进制索引底层实现详解

### 8.1 IndexBinary完整结构

```cpp
// faiss/IndexBinary.h
struct IndexBinary {
    using component_t = uint8_t;   // 二进制分量类型
    using distance_t = int32_t;    // 距离类型（汉明距离）

    int d;              // 位数
    int code_size;      // 字节数 = d / 8
    idx_t ntotal;       // 总向量数
    MetricType metric_type;
    bool is_trained;

    IndexBinary(idx_t d = 0, MetricType metric = METRIC_L2)
        : d(d), code_size((d + 7) / 8), ntotal(0),
          metric_type(metric), is_trained(true) {}

    virtual ~IndexBinary() {}

    // 核心接口
    virtual void add(idx_t n, const uint8_t* x) = 0;

    virtual void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,  // 汉明距离
            idx_t* labels,
            const SearchParameters* params = nullptr) const = 0;

    virtual void train(idx_t n, const uint8_t* x) {
        // 默认不需要训练
    }

    virtual void reset() {
        ntotal = 0;
    }

    // 其他接口
    virtual void reconstruct(idx_t key, uint8_t* recons) const = ;
    virtual void sa_encode(idx_t n, const uint8_t* x, uint8_t* bytes) const = 0;
    virtual void sa_decode(idx_t n, const uint8_t* bytes, uint8_t* x) const = 0;
};
```

### 8.2 IndexBinaryFlat完整实现

```cpp
// faiss/IndexBinaryFlat.h
struct IndexBinaryFlat : IndexBinary {
    std::vector<uint8_t> codes;  // 所有向量，存储为 ntotal × code_size

    IndexBinaryFlat(idx_t d)
        : IndexBinary(d, METRIC_L2) {}

    void add(idx_t n, const uint8_t* x) override {
        codes.resize((ntotal + n) * code_size);
        memcpy(codes.data() + ntotal * code_size, x, n * code_size);
        ntotal += n;
    }

    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* xq = x + i * code_size;
            int32_t* dis_i = distances + i * k;
            idx_t* lbl_i = labels + i * k;

            // 初始化堆（用于找最近的k个）
            heap_heapify<CMax<int32_t, idx_t>>(k, dis_i, lbl_i);

            // 穷举搜索所有向量
            for (idx_t j = 0; j < ntotal; j++) {
                int32_t dis = hamming::popcount(
                    xq,
                    codes.data() + j * code_size,
                    code_size);

                if (CMax<int32_t, idx_t>::cmp(dis, dis_i[0])) {
                    heap_replace_top<CMax<int32_t, idx_t>>(
                        k, dis_i, lbl_i, dis, j);
                }
            }

            heap_reorder<CMax<int32_t, idx_t>>(k, dis_i, lbl_i);
        }
    }

    void reconstruct(idx_t key, uint8_t* recons) const override {
        memcpy(recons, codes.data() + key * code_size, code_size);
    }

    void sa_encode(idx_t n, const uint8_t* x, uint8_t* bytes) const override {
        memcpy(bytes, x, n * code_size);
    }

    void sa_decode(idx_t n, const uint8_t* bytes, uint8_t* x) const override {
        memcpy(x, bytes, n * code_size);
    }

    void reset() override {
        codes.clear();
        ntotal = 0;
    }
};
```

### 8.3 Hamming距离优化实现

```cpp
// faiss/utils/hamming.h
namespace hamming {

// 使用popcount指令的汉明距离计算
inline int32_t popcount(const uint8_t* a, const uint8_t* b, size_t n) {
    int32_t dis = 0;
    for (size_t i = 0; i < n; i++) {
        dis += __builtin_popcount(a[i] ^ b[i]);
    }
    return dis;
}

// 批量汉明距离计算（SIMD优化）
inline void batch_popcount(
        const uint8_t* query,
        const uint8_t* database,
        size_t n,
        size_t code_size,
        int32_t* distances) {

    for (size_t i = 0; i < n; i++) {
        distances[i] = popcount(
            query,
            database + i * code_size,
            code_size);
    }
}

// SIMD优化的批量汉明距离（AVX2）
inline void batch_popcount_avx2(
        const uint8_t* query,
        const uint8_t* database,
        size_t n,
        size_t code_size,
        int32_t* distances) {

    size_t i = 0;

    // 处理4的倍数（展开以提高ILP）
    for (; i + 4 <= n; i += 4) {
        distances[i + 0] = popcount(
            query, database + (i + 0) * code_size, code_size);
        distances[i + 1] = popcount(
            query, database + (i + 1) * code_size, code_size);
        distances[i + 2] = popcount(
            query, database + (i + 2) * code_size, code_size);
        distances[i + 3] = popcount(
            query, database + (i + 3) * code_size, code_size);
    }

    // 处理剩余向量
    for (; i < n; i++) {
        distances[i] = popcount(
            query, database + i * code_size, code_size);
    }
}

// 比较汉明距离并返回最小距离的索引
inline size_t min_hamming(
        const uint8_t* query,
        const uint8_t* database,
        size_t n,
        size_t code_size,
        int32_t* min_dis_out = nullptr) {

    int32_t min_dis = std::numeric_limits<int32_t>::max();
    size_t min_idx = 0;

    for (size_t i = 0; i < n; i++) {
        int32_t dis = popcount(
            query, database + i * code_size, code_size);

        if (dis < min_dis) {
            min_dis = dis;
            min_idx = i;
        }
    }

    if (min_dis_out) {
        *min_dis_out = min_dis;
    }

    return min_idx;
}

} // namespace hamming
```

### 8.4 IndexBinaryIVF完整结构

```cpp
// faiss/IndexBinaryIVF.h
struct IndexBinaryIVF : IndexBinary {
    Index* quantizer;    // 粗量化器（用于分配倒排列表）
    size_t nlist;        // 倒排列表数
    size_t nprobe;       // 搜索时访问的列表数
    InvertedLists* invlists;  // 倒排列表

    IndexBinaryIVF(Index* quantizer, idx_t d, size_t nlist)
        : IndexBinary(d, METRIC_L2),
          quantizer(quantizer),
          nlist(nlist),
          nprobe(1) {}

    ~IndexBinaryIVF() {
        if (own_fields) {
            delete quantizer;
            delete invlists;
        }
    }

    void add(idx_t n, const uint8_t* x) override {
        // 1. 分配到倒排列表
        idx_t* list_nos = new idx_t[n];
        quantizer->assign(n, x, list_nos);

        // 2. 添加到倒排列表
        for (idx_t i = 0; i < n; i++) {
            idx_t list_no = list_nos[i];
            const uint8_t* xi = x + i * code_size;
            idx_t id = ntotal + i;

            invlists->add_entry(list_no, id, xi);
        }

        delete[] list_nos;
        ntotal += n;
    }

    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 1. 找到最近的nprobe个列表
        idx_t* assign = new idx_t[n * nprobe];
        quantizer->assign(n, x, assign);

        // 2. 搜索每个查询的候选列表
#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            int32_t* dis_i = distances + i * k;
            idx_t* lbl_i = labels + i * k;

            heap_heapify<CMax<int32_t, idx_t>>(k, dis_i, lbl_i);

            // 访问nprobe个列表
            for (size_t ij = 0; ij < nprobe; ij++) {
                idx_t list_no = assign[i * nprobe + ij];
                size_t list_size = invlists->list_size(list_no);

                const uint8_t* codes = invlists->get_codes(list_no);
                const idx_t* ids = invlists->get_ids(list_no);

                // 扫描列表中的所有向量
                for (size_t j = 0; j < list_size; j++) {
                    int32_t dis = hamming::popcount(
                        x + i * code_size,
                        codes + j * code_size,
                        code_size);

                    if (CMax<int32_t, idx_t>::cmp(dis, dis_i[0])) {
                        heap_replace_top<CMax<int32_t, idx_t>>(
                            k, dis_i, lbl_i, dis, ids[j]);
                    }
                }
            }

            heap_reorder<CMax<int32_t, idx_t>>(k, dis_i, lbl_i);
        }

        delete[] assign;
    }

    void train(idx_t n, const uint8_t* x) override {
        // 训练量化器
        quantizer->train(n, x);
        is_trained = true;
    }
};
```

### 8.5 IndexBinaryHash完整实现

```cpp
// faiss/IndexBinaryHash.h (相关实现)
struct IndexBinaryHash : IndexBinary {
    int nhash;               // 哈希表数
    int nbits_per_hash;      // 每个哈希的位数
    int nflip;               // 搜索时允许的翻转数

    std::vector<std::unordered_map<uint64_t, std::vector<idx_t>>> hashtables;
    RandomGenerator rng;

    IndexBinaryHash(idx_t d, int nhash, int nbits_per_hash)
        : IndexBinary(d, METRIC_L2),
          nhash(nhash),
          nbits_per_hash(nbits_per_hash),
          nflip(0),
          rng(12345),
          hashtables(nhash) {}

    void add(idx_t n, const uint8_t* x) override {
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* xi = x + i * code_size;

            // 为每个哈希表计算哈希
            for (int h = 0; h < nhash; h++) {
                uint64_t hash = compute_hash(xi, h);
                hashtables[h][hash].push_back(ntotal + i);
            }
        }

        ntotal += n;
    }

    uint64_t compute_hash(const uint8_t* code, int h) {
        // 从code中选择nbits_per_hash位作为哈希
        uint64_t hash = 0;
        int offset = h * nbits_per_hash;

        for (int i = 0; i < nbits_per_hash; i++) {
            int bit_idx = (offset + i) % d;
            int byte_idx = bit_idx / 8;
            int bit = 7 - (bit_idx % 8);

            if (code[byte_idx] & (1 << bit)) {
                hash |= (1ULL << i);
            }
        }

        return hash;
    }

    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* xq = x + i * code_size;
            int32_t* dis_i = distances + i * k;
            idx_t* lbl_i = labels + i * k;

            heap_heapify<CMax<int32_t, idx_t>>(k, dis_i, lbl_i);

            // 收集候选
            std::unordered_set<idx_t> candidates;

            for (int h = 0; h < nhash; h++) {
                uint64_t hash = compute_hash(xq, h);

                if (hashtables[h].find(hash) != hashtables[h].end()) {
                    for (idx_t id : hashtables[h].at(hash)) {
                        candidates.insert(id);
                    }
                }
            }

            // 计算精确距离
            for (idx_t id : candidates) {
                const uint8_t* code = get_code(id);
                int32_t dis = hamming::popcount(xq, code, code_size);

                if (CMax<int32_t, idx_t>::cmp(dis, dis_i[0])) {
                    heap_replace_top<CMax<int32_t, idx_t>>(
                        k, dis_i, lbl_i, dis, id);
                }
            }

            heap_reorder<CMax<int32_t, idx_t>>(k, dis_i, lbl_i);
        }
    }
};
```

### 8.6 多索引哈希（Multi-Index Hashing）

```cpp
// 多索引哈希：将长二进制向量分成多个段
// 每个段独立构建哈希表，减少哈希冲突
struct MultiIndexHash {
    int nseg;              // 段数
    int bits_per_seg;      // 每段位数
    std::vector<IndexBinaryHash> segments;

    MultiIndexHash(int d, int nseg, int bits_per_seg)
        : nseg(nseg), bits_per_seg(bits_per_seg) {

        int bits_per_hash = d / nseg;

        for (int i = 0; i < nseg; i++) {
            segments.emplace_back(bits_per_hash, 1, bits_per_hash);
        }
    }

    void add(idx_t n, const uint8_t* x) {
        for (int seg = 0; seg < nseg; seg++) {
            // 提取该段
            int offset = seg * bits_per_seg / 8;
            int size = bits_per_seg / 8;

            std::vector<uint8_t> segment_codes(n * size);
            for (idx_t i = 0; i < n; i++) {
                memcpy(segment_codes.data() + i * size,
                       x + i * code_size + offset, size);
            }

            segments[seg].add(n, segment_codes.data());
        }
    }

    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels) {

        for (idx_t i = 0; i < n; i++) {
            std::unordered_map<idx_t, int> vote_count;

            // 对每个段进行搜索
            for (int seg = 0; seg < nseg; seg++) {
                int offset = seg * bits_per_seg / 8;
                int size = bits_per_seg / 8;

                const uint8_t* segment_query = x + i * code_size + offset;

                // 搜索该段
                idx_t seg_labels[100];
                int32_t seg_dists[100];
                int seg_k = std::min(100, k);

                segments[seg].search(1, segment_query, seg_k,
                                    seg_dists, seg_labels);

                // 投票
                for (int j = 0; j < seg_k; j++) {
                    vote_count[seg_labels[j]]++;
                }
            }

            // 选择得票最多的候选
            std::vector<std::pair<int, idx_t>> sorted;
            for (const auto& entry : vote_count) {
                sorted.push_back({entry.second, entry.first});
            }
            std::sort(sorted.rbegin(), sorted.rend());

            // 在top候选上计算精确距离
            heap_heapify<CMax<int32_t, idx_t>>(k, distances, labels);

            for (const auto& entry : sorted) {
                idx_t id = entry.second;
                int32_t dis = hamming::popcount(
                    x + i * code_size, get_code(id), code_size);

                if (CMax<int32_t, idx_t>::cmp(dis, distances[0])) {
                    heap_replace_top<CMax<int32_t, idx_t>>(
                        k, distances, labels, dis, id);
                }
            }

            heap_reorder<CMax<int32_t, idx_t>>(k, distances, labels);
        }
    }
};
```

### 8.7 位操作优化

```cpp
// faiss/utils/hamming.h (相关实现)
// 位操作优化的汉明距离计算

// 快速比较：判断两个汉明距离是否小于阈值
inline bool hamming_less_than(
        const uint8_t* a, const uint8_t* b, size_t n, int threshold) {

    int accu = 0;
    for (size_t i = 0; i < n; i++) {
        accu += __builtin_popcount(a[i] ^ b[i]);
        if (accu > threshold) {
            return false;
        }
    }
    return true;
}

// 批量比较：找出所有距离小于阈值的数据点
inline size_t hamming_range_search(
        const uint8_t* query,
        const uint8_t* database,
        size_t n,
        size_t code_size,
        int threshold,
        idx_t* result_ids,
        int32_t* result_distances) {

    size_t n_found = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t dis = hamming::popcount(
            query, database + i * code_size, code_size);

        if (dis < threshold) {
            result_ids[n_found] = i;
            result_distances[n_found] = dis;
            n_found++;
        }
    }

    return n_found;
}

// 计数汉明距离为某个值的向量数
inline size_t count_hamming_at(
        const uint8_t* query,
        const uint8_t* database,
        size_t n,
        size_t code_size,
        int target_distance) {

    size_t count = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t dis = hamming::popcount(
            query, database + i * code_size, code_size);

        if (dis == target_distance) {
            count++;
        }
    }

    return count;
}

} // namespace hamming
```

---

## 9. 第11天总结

### 关键概念

1. **二进制索引**：处理二进制向量
2. **Hamming距离**：异或后统计1的个数
3. **POPCNT**：CPU指令加速统计位数
4. **IndexBinaryIVF**：二向量的IVF索引
5. **二值化方法**：阈值、均值、LSH

### 性能特点

| 特性 | 浮点索引 | 二进制索引 |
|------|---------|-----------|
| 存储 | 4d字节 | d/8字节 |
| 距离计算 | 浮点运算 | 位运算+POPCNT |
| 精度 | 高 | 中-低 |
| 速度 | 慢 | 极快 |

### 下一步

第12天将学习**GPU实现**，利用CUDA和ROCm加速向量搜索。

---

---

## 10. Binary Index源码深度实现

本节深入分析Faiss中二进制索引的核心实现细节，包括Hamming距离计算优化、批量搜索算法、位操作优化等关键算法。

### 10.1 IndexBinaryFlat::search - 堆优化搜索

```cpp
// faiss/IndexBinaryFlat.cpp
void IndexBinaryFlat::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    // 提取ID选择器（用于过滤特定ID）
    const IDSelector* sel = params ? params->sel : nullptr;
    FAISS_THROW_IF_NOT(k > 0);

    // 批量处理查询向量
    const idx_t block_size = query_batch_size;

    for (idx_t s = 0; s < n; s += block_size) {
        idx_t nn = block_size;
        if (s + block_size > n) {
            nn = n - s;
        }

        if (use_heap) {
            // 使用堆方法：适用于k相对较小的情况
            int_maxheap_array_t res = {
                    size_t(nn),    // 查询数
                    size_t(k),     // top-k
                    labels + s * k,
                    distances + s * k};

            hammings_knn_hc(
                    &res,
                    x + s * code_size,
                    xb.data(),
                    ntotal,
                    code_size,
                    /* ordered = */ true,
                    approx_topk_mode,
                    sel);
        } else {
            // 使用计数方法：适用于k相对较大的情况
            hammings_knn_mc(
                    x + s * code_size,
                    xb.data(),
                    nn,
                    ntotal,
                    k,
                    code_size,
                    distances + s * k,
                    labels + s * k,
                    sel);
        }
    }
}
```

### 10.2 hammings_knn_hc - 堆方法KNN搜索

```cpp
// faiss/utils/hamming.cpp (简化版)
void hammings_knn_hc(
        int_maxheap_array_t* ha,
        const uint8_t* a,
        const uint8_t* b,
        size_t nb,
        size_t ncodes,
        int ordered,
        ApproxTopK_mode_t approx_topk_mode,
        const IDSelector* sel) {

    // 初始化堆
    int_maxheap_array_t heap = ha[0];
    for (size_t i = 0; i < heap.nh; i++) {
        maxheap_heapify(heap.k, heap.dis + i * heap.k, heap.ids + i * heap.k);
    }

    // 批量处理数据库向量
    size_t i0 = 0;
    const size_t batch_size = hamming_batch_size;

    while (i0 < nb) {
        size_t i1 = std::min(i0 + batch_size, nb);

        // 对每个查询向量
        for (size_t j = 0; j < heap.nh; j++) {
            const uint8_t* aq = a + j * ncodes;
            int32_t* dis_j = heap.dis + j * heap.k;
            idx_t* ids_j = heap.ids + j * heap.k;

            // 扫描数据库向量的一个批次
            for (size_t i = i0; i < i1; i++) {
                // 应用ID选择器
                if (sel && !sel->is_member(i)) {
                    continue;
                }

                // 计算Hamming距离
                hamdis_t dis = hamming(
                        (const uint64_t*)(aq),
                        (const uint64_t*)(b + i * ncodes),
                        ncodes / sizeof(uint64_t));

                // 更新堆
                if (dis < dis_j[0]) {
                    maxheap_replace_top(
                            heap.k, dis_j, ids_j, dis, i);
                }
            }
        }

        i0 = i1;
    }

    // 排序结果
    if (ordered) {
        for (size_t i = 0; i < heap.nh; i++) {
            maxheap_reorder(heap.k, heap.dis + i * heap.k, heap.ids + i * heap.k);
        }
    }
}
```

### 10.3 hammings_knn_mc - 计数方法KNN搜索

```cpp
// faiss/utils/hamming.cpp (简化版)
// 计数方法：当k较大时比堆方法更高效
void hammings_knn_mc(
        const uint8_t* a,
        const uint8_t* b,
        size_t na,
        size_t nb,
        size_t k,
        size_t ncodes,
        int32_t* distances,
        idx_t* labels,
        const IDSelector* sel) {

    // Hamming距离范围
    const int max_ham = ncodes * 8;

    // 为每个查询向量计数距离分布
    for (size_t i = 0; i < na; i++) {
        const uint8_t* aq = a + i * ncodes;
        int32_t* dis_i = distances + i * k;
        idx_t* lbl_i = labels + i * k;

        // 初始化
        for (size_t j = 0; j < k; j++) {
            dis_i[j] = max_ham + 1;
            lbl_i[j] = -1;
        }

        // 计数直方图：counts[d] = 距离为d的向量数
        std::vector<int> count(max_ham + 1, 0);
        std::vector<std::vector<idx_t>> ids_by_dis(max_ham + 1);

        for (size_t j = 0; j < nb; j++) {
            if (sel && !sel->is_member(j)) {
                continue;
            }

            hamdis_t dis = hamming(
                    (const uint64_t*)(aq),
                    (const uint64_t*)(b + j * ncodes),
                    ncodes / sizeof(uint64_t));

            count[dis]++;
            ids_by_dis[dis].push_back(j);
        }

        // 从小到大收集向量直到达到k个
        size_t collected = 0;
        for (int d = 0; d <= max_ham && collected < k; d++) {
            for (idx_t id : ids_by_dis[d]) {
                if (collected >= k) break;
                dis_i[collected] = d;
                lbl_i[collected] = id;
                collected++;
            }
        }
    }
}
```

### 10.4 hamming - POPCNT优化的距离计算

```cpp
// faiss/utils/hamming_distance/hamdis-inl.h
// 使用CPU POPCNT指令优化的Hamming距离计算

namespace faiss {

// 基础Hamming距离计算（使用POPCNT指令）
inline hamdis_t hamming(
        const uint64_t* bs1,
        const uint64_t* bs2,
        size_t nwords) {

    hamdis_t accu = 0;
    for (size_t i = 0; i < nwords; i++) {
        uint64_t x = bs1[i] ^ bs2[i];  // 异或
        accu += __builtin_popcountll(x);  // 统计1的个数
    }
    return accu;
}

// SIMD优化的批量Hamming距离计算
inline void hammings(
        const uint8_t* a,
        const uint8_t* b,
        size_t na,
        size_t nb,
        size_t nbytespercode,
        hamdis_t* dis) {

    // nbytespercode应该是8的倍数
    size_t nwords = nbytespercode / sizeof(uint64_t);

    for (size_t i = 0; i < na; i++) {
        const uint64_t* aptr = (const uint64_t*)(a + i * nbytespercode);

        for (size_t j = 0; j < nb; j++) {
            const uint64_t* bptr = (const uint64_t*)(b + j * nbytespercode);
            dis[i * nb + j] = hamming(aptr, bptr, nwords);
        }
    }
}

} // namespace faiss
```

### 10.5 BitstringWriter/Reader - 位串读写

```cpp
// faiss/utils/hamming.h (相关实现)
// 位串写入器：将位打包到字节数组

struct BitstringWriter {
    uint8_t* code;
    size_t code_size;  // 字节大小
    size_t i;          // 当前位偏移

    BitstringWriter(uint8_t* code, size_t code_size)
        : code(code), code_size(code_size), i(0) {
        memset(code, 0, code_size);
    }

    // 写入x的低nbit位
    void write(uint64_t x, int nbit) {
        FAISS_THROW_IF_NOT(nbit <= 64);
        FAISS_THROW_IF_NOT(i + nbit <= code_size * 8);

        size_t i0 = i / 8;
        size_t i1 = (i + nbit - 1) / 8;

        int n_inside = (i0 + 1) * 8 - i;
        x = x & ((1ULL << nbit) - 1);

        if (i0 == i1) {
            // 所有位在同一个字节内
            code[i0] |= (x << (8 - n_inside - nbit));
        } else {
            // 跨越多个字节
            code[i0] |= (x >> (nbit - n_inside));
            code[i1] |= (x << (8 - ((nbit - n_inside) % 8))) & ((1 << 8) - 1);
        }

        i += nbit;
    }
};

// 位串读取器：从字节数组读取位
struct BitstringReader {
    const uint8_t* code;
    size_t code_size;
    size_t i;

    BitstringReader(const uint8_t* code, size_t code_size)
        : code(code), code_size(code_size), i(0) {}

    // 读取nbit位
    uint64_t read(int nbit) {
        FAISS_THROW_IF_NOT(nbit <= 64);
        FAISS_THROW_IF_NOT(i + nbit <= code_size * 8);

        size_t i0 = i / 8;
        size_t i1 = (i + nbit - 1) / 8;

        int n_inside = (i0 + 1) * 8 - i;

        uint64_t x = 0;

        if (i0 == i1) {
            // 所有位在同一个字节内
            uint8_t mask = (1 << nbit) - 1;
            x = (code[i0] >> (8 - n_inside - nbit)) & mask;
        } else {
            // 跨越多个字节
            x = code[i0];  // 第一部分的低位
            x <<= (nbit - n_inside);
            x |= code[i1] >> (8 - (nbit - n_inside));
        }

        i += nbit;
        return x;
    }
};
```

### 10.6 fvecs2bitvecs - 浮点向量转二进制

```cpp
// faiss/utils/hamming.cpp
// 将浮点向量转换为二进制向量（基于符号位）

void fvecs2bitvecs(const float* x, uint8_t* b, size_t d, size_t n) {
    // d应该是8的倍数
    for (size_t i = 0; i < n; i++) {
        const float* xi = x + i * d;
        uint8_t* bi = b + i * ((d + 7) / 8);

        for (size_t j = 0; j < (d + 7) / 8; j++) {
            uint8_t byte = 0;
            for (size_t k = 0; k < 8 && j * 8 + k < d; k++) {
                if (xi[j * 8 + k] > 0) {
                    byte |= (1 << (7 - k));
                }
            }
            bi[j] = byte;
        }
    }
}

// 单个浮点向量转二进制
void fvec2bitvec(const float* x, uint8_t* b, size_t d) {
    fvecs2bitvecs(x, b, d, 1);
}

// 二进制向量转浮点向量
void bitvecs2fvecs(const uint8_t* b, float* x, size_t d, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint8_t* bi = (uint8_t*)b + i * ((d + 7) / 8);
        float* xi = x + i * d;

        for (size_t j = 0; j < d; j++) {
            int byte_idx = j / 8;
            int bit_idx = 7 - (j % 8);
            xi[j] = (bi[byte_idx] & (1 << bit_idx)) ? 1.0f : -1.0f;
        }
    }
}
```

### 10.7 hamming_range_search - 范围搜索

```cpp
// faiss/utils/hamming.cpp (简化版)
// Hamming距离范围搜索：返回所有距离小于等于radius的向量

void hamming_range_search(
        const uint8_t* a,
        const uint8_t* b,
        size_t na,
        size_t nb,
        int radius,
        size_t ncodes,
        RangeSearchResult* result,
        const IDSelector* sel) {

    // 为每个查询向量收集结果
    result->lims.resize(na + 1);
    result->labels.resize(nb * na);  // 预分配最大空间
    result->distances.resize(nb * na);

    size_t n_found = 0;

    for (size_t i = 0; i < na; i++) {
        result->lims[i] = n_found;

        const uint8_t* aq = a + i * ncodes;

        for (size_t j = 0; j < nb; j++) {
            if (sel && !sel->is_member(j)) {
                continue;
            }

            hamdis_t dis = hamming(
                    (const uint64_t*)(aq),
                    (const uint64_t*)(b + j * ncodes),
                    ncodes / sizeof(uint64_t));

            if (dis <= radius) {
                result->labels[n_found] = j;
                result->distances[n_found] = dis;
                n_found++;
            }
        }
    }

    result->lims[na] = n_found;

    // 压缩结果
    result->labels.resize(n_found);
    result->distances.resize(n_found);
}
```

### 10.8 生产级使用示例

```cpp
// 二进制索引生产环境使用
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>

// 浮点向量转二进制并创建索引
void binary_index_production_example() {
    int d_float = 128;
    int n = 1000000;
    int nq = 100;
    int k = 10;

    // 将浮点向量转换为二进制
    int d = d_float;
    size_t code_size = (d + 7) / 8;

    // 方法1：符号位二值化
    std::vector<uint8_t> xb_binary(n * code_size);
    faiss::fvecs2bitvecs(xb, xb_binary.data(), d, n);

    // 方法2：阈值二值化
    std::vector<uint8_t> xb_binary2(n * code_size);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            int byte_idx = (i * d + j) / 8;
            int bit_idx = 7 - (j % 8);
            if (xb[i * d + j] > 0.5f) {  // 阈值
                xb_binary2[byte_idx] |= (1 << bit_idx);
            }
        }
    }

    // 创建IndexBinaryFlat
    faiss::IndexBinaryFlat index(d);
    index.add(n, xb_binary.data());

    // 搜索
    std::vector<uint8_t> xq_binary(nq * code_size);
    faiss::fvecs2bitvecs(xq, xq_binary.data(), d, nq);

    std::vector<int32_t> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    // 设置批量大小
    index.query_batch_size = 32;

    // 选择搜索方法
    index.use_heap = true;  // 堆方法（适合小k）
    // index.use_heap = false;  // 计数方法（适合大k）

    index.search(nq, xq_binary.data(), k,
                 distances.data(), labels.data());

    // 评估
    evaluate_binary_search(xb, xq, distances, labels, nq, k);
}

// IVF二进制索引
void binary_ivf_example() {
    int d = 128;
    size_t nlist = 256;
    int n = 1000000;

    // 转换为二进制
    size_t code_size = (d + 7) / 8;
    std::vector<uint8_t> xb_binary(n * code_size);
    faiss::fvecs2bitvecs(xb, xb_binary.data(), d, n);

    // 创建粗量化器（用于分配倒排列表）
    faiss::IndexFlatL2 quantizer(d);

    // 创建IndexBinaryIVF
    faiss::IndexBinaryIVF index(&quantizer, d, nlist);

    // 训练和添加
    index.train(n, xb_binary.data());
    index.add(n, xb_binary.data());

    // 搜索
    int nq = 100;
    int k = 10;

    std::vector<uint8_t> xq_binary(nq * code_size);
    faiss::fvecs2bitvecs(xq, xq_binary.data(), d, nq);

    std::vector<int32_t> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    faiss::IVFSearchParameters params;
    params.nprobe = 16;  // 搜索16个倒排列表

    index.search(nq, xq_binary.data(), k,
                 distances.data(), labels.data(), &params);
}

// 批量距离计算优化
void batch_hamming_example() {
    int d = 128;
    size_t code_size = (d + 7) / 8;
    size_t n = 10000;
    size_t nq = 100;

    std::vector<uint8_t> xb(n * code_size);
    std::vector<uint8_t> xq(nq * code_size);

    // 批量计算距离矩阵
    std::vector<hamdis_t> dis(nq * n);

    faiss::hammings(
            xq.data(),
            xb.data(),
            nq,
            n,
            code_size,
            dis.data());

    // 找出每个查询的top-k
    int k = 10;
    std::vector<int32_t> min_dis(nq * k);
    std::vector<idx_t> min_labels(nq * k);

    for (size_t i = 0; i < nq; i++) {
        faiss::int_maxheap_array_t heap = {1, (size_t)k,
            min_labels.data() + i * k,
            min_dis.data() + i * k};
        faiss::maxheap_heapify(heap.k, heap.dis, heap.ids);

        for (size_t j = 0; j < n; j++) {
            hamdis_t dis_ij = dis[i * n + j];
            if (dis_ij < heap.dis[0]) {
                faiss::maxheap_replace_top(
                        heap.k, heap.dis, heap.ids, dis_ij, j);
            }
        }

        faiss::maxheap_reorder(heap.k, heap.dis, heap.ids);
    }
}

// 位打包示例
void pack_bitstrings_example() {
    int n = 1000;
    int M = 4;    // 每个编码的元素数
    int nbit = 6;  // 每个元素的位数

    std::vector<int32_t> unpacked(n * M);
    // ... 填充unpacked

    // 计算code_size
    size_t code_size = (M * nbit + 7) / 8;
    std::vector<uint8_t> packed(n * code_size);

    // 打包
    faiss::pack_bitstrings(
            n, M, nbit,
            unpacked.data(),
            packed.data(),
            code_size);

    // 解包验证
    std::vector<int32_t> unpacked2(n * M);
    faiss::unpack_bitstrings(
            n, M, nbit,
            packed.data(),
            code_size,
            unpacked2.data());

    // 验证
    for (size_t i = 0; i < n * M; i++) {
        assert(unpacked[i] == unpacked2[i]);
    }
}
```

### 10.9 性能优化总结

| 优化技术 | 描述 | 适用场景 | 性能提升 |
|----------|------|----------|----------|
| POPCNT指令 | 单指令统计位数 | 所有平台 | 10-20x |
| 堆方法(hc) | 维护小顶堆 | k较小(k/n<0.01) | 高效 |
| 计数方法(mc) | 直方图计数 | k较大(k/n>0.01) | 2-5x |
| 批量处理 | 一次处理多个查询 | 大量查询 | 减少开销 |
| SIMD位操作 | AVX2处理多个字节 | 对齐数据 | 4-8x |
| 位压缩 | 减少内存占用 | 大规模数据 | 节省32x内存 |

---

## 练习题

1. 实现Hamming距离计算
2. 实现POPCNT优化的批量距离计算
3. 比较不同二值化方法的精度
4. 实现IndexBinaryIVF

## 扩展阅读

- faiss/IndexBinary.h - 二进制索引基类
- faiss/IndexBinaryFlat.h - 二进制Flat索引
- faiss/IndexBinaryIVF.h - 二进制IVF索引
- [Hamming距离](https://en.wikipedia.org/wiki/Hamming_distance)
- [LSH教程](https://www.cs.princeton.edu/courses/archive/fall06/cos529F/lectures/lsk.pdf)
