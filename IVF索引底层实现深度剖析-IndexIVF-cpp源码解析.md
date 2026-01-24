# IVF索引底层实现深度剖析 - IndexIVF.cpp源码解析

## 文档说明

本文档深入剖析Faiss中IVF（Inverted File）索引的底层实现，基于`IndexIVF.cpp`源码，详细讲解倒排索引、并行搜索策略、内存优化等核心技术。

**前置知识**：
- 已完成《Faiss基础教程》
- 了解向量索引基本概念
- 熟悉OpenMP并行编程

---

## 目录
- [1. IVF索引架构](#1-ivf索引架构)
- [2. Level1Quantizer实现](#2-level1quantizer实现)
- [3. 向量添加流程](#3-向量添加流程)
- [4. 并行搜索策略](#4-并行搜索策略)
- [5. 倒排列表优化](#5-倒排列表优化)
- [6. Scanner模式](#6-scanner模式)
- [7. 性能优化技巧](#7-性能优化技巧)
- [8. 总结](#8-总结)

---

## 1. IVF索引架构

### 1.1 IVF索引核心思想

**IVF（Inverted File）索引**将向量空间划分为多个Voronoi cell（倒排桶），每个向量根据其最近的质心被分配到一个桶中。

```
向量空间划分:
┌─────────────────────────────────────┐
│  Cell 0    Cell 1    Cell 2  ...   │
│ 质心0      质心1      质心2         │
│  [v1]      [v5]      [v2]          │
│  [v3]      [v7]      [v8]          │
│  [v4]                 [v9]          │
└─────────────────────────────────────┘

倒排索引结构:
倒排表0: [id1, id3, id4, ...]
倒排表1: [id5, id7, ...]
倒排表2: [id2, id8, id9, ...]
```

### 1.2 IndexIVF类结构

```cpp
class IndexIVF : public Index {
public:
    // 一级量化器（coarse quantizer）
    Level1Quantizer quantizer;

    // 倒排列表
    InvertedLists* invlists;
    bool own_invlists;

    // 编码大小（字节）
    size_t code_size;

    // 搜索参数
    idx_t nprobe;           // 搜索的倒排表数量
    idx_t max_codes;        // 最大扫描编码数
    int parallel_mode;      // 并行模式

    // 直接映射（可选）
    DirectMap direct_map;

    // 统计信息
    IndexIVFStats indexIVF_stats;
};
```

### 1.3 Level1Quantizer

```cpp
struct Level1Quantizer {
    Index* quantizer;  // 用于聚类的索引（通常是IndexFlat）
    size_t nlist;      // 倒排表数量

    // 训练参数
    ClusteringParameters cp;

    // 训练模式
    int quantizer_trains_alone;  // 0:聚类训练, 1:单独训练, 2:L2训练
    Index* clustering_index;     // 可选的自定义聚类索引

    void train_q1(size_t n, const float* x, bool verbose, MetricType metric_type);
    size_t coarse_code_size() const;
    void encode_listno(idx_t list_no, uint8_t* code) const;
    idx_t decode_listno(const uint8_t* code) const;
};
```

---

## 2. Level1Quantizer实现

### 2.1 train_q1函数

```cpp
void Level1Quantizer::train_q1(
        size_t n,
        const float* x,
        bool verbose,
        MetricType metric_type) {

    size_t d = quantizer->d;

    // 情况1: 量化器已经训练好
    if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
        if (verbose) {
            printf("IVF quantizer does not need training.\\n");
        }
        return;
    }

    // 情况2: 量化器单独训练
    else if (quantizer_trains_alone == 1) {
        if (verbose) {
            printf("IVF quantizer trains alone...\\n");
        }
        quantizer->verbose = verbose;
        quantizer->train(n, x);
        FAISS_THROW_IF_NOT(quantizer->ntotal == nlist);
    }

    // 情况3: 使用聚类训练（默认）
    else if (quantizer_trains_alone == 0) {
        if (verbose) {
            printf("Training level-1 quantizer on %zd vectors in %zdD\\n", n, d);
        }

        Clustering clus(d, nlist, cp);
        quantizer->reset();

        if (clustering_index) {
            // 使用自定义索引进行聚类
            clus.train(n, x, *clustering_index);
        } else {
            // 使用量化器本身进行聚类
            clus.train(n, x, *quantizer);
        }

        quantizer->is_trained = true;
    }

    // 情况4: L2距离特殊训练
    else if (quantizer_trains_alone == 2) {
        // 使用IndexFlatL2进行分配
        Clustering clus(d, nlist, cp);
        IndexFlatL2 assigner(d);
        clus.train(n, x, assigner);

        // 将质心添加到量化器
        if (!quantizer->is_trained) {
            quantizer->train(nlist, clus.centroids.data());
        }
        quantizer->add(nlist, clus.centroids.data());
    }
}
```

**训练策略分析**：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| 0 | 聚类训练 | 通用，默认模式 |
| 1 | 单独训练 | 量化器需要特殊训练 |
| 2 | L2训练 | 内积距离，球形聚类 |

### 2.2 编码/解码倒排表号

```cpp
// 计算存储倒排表号所需的字节数
size_t Level1Quantizer::coarse_code_size() const {
    size_t nl = nlist - 1;
    size_t nbyte = 0;
    while (nl > 0) {
        nbyte++;
        nl >>= 8;
    }
    return nbyte;
}

// 编码倒排表号（小端序）
void Level1Quantizer::encode_listno(idx_t list_no, uint8_t* code) const {
    size_t nl = nlist - 1;
    while (nl > 0) {
        *code++ = list_no & 0xff;  // 低8位
        list_no >>= 8;
        nl >>= 8;
    }
}

// 解码倒排表号
idx_t Level1Quantizer::decode_listno(const uint8_t* code) const {
    size_t nl = nlist - 1;
    int64_t list_no = 0;
    int nbit = 0;
    while (nl > 0) {
        list_no |= int64_t(*code++) << nbit;
        nbit += 8;
        nl >>= 8;
    }
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    return list_no;
}
```

**示例**：
```
nlist = 1000
list_no = 999

编码:
999 = 0x3E7 = 0b11_1110_0111
coarse_code_size() = 2 字节
code[0] = 0xE7 (低8位)
code[1] = 0x03 (高8位)

解码:
list_no = code[0] | (code[1] << 8) = 0xE7 | 0x300 = 0x3E7 = 999
```

---

## 3. 向量添加流程

### 3.1 add_with_ids函数

```cpp
void IndexIVF::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    // 步骤1: 分配倒排表号（粗量化）
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    quantizer->assign(n, x, coarse_idx.get());

    // 步骤2: 核心添加逻辑
    add_core(n, x, xids, coarse_idx.get());
}
```

### 3.2 add_core函数

```cpp
void IndexIVF::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {

    // 步骤1: 分块处理（避免过度分配）
    idx_t bs = 65536;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            add_core(i1 - i0, x + i0 * d,
                   xids ? xids + i0 : nullptr,
                   coarse_idx + i0,
                   inverted_list_context);
        }
        return;
    }

    // 步骤2: 统计无效分配
    size_t nadd = 0, nminus1 = 0;
    for (size_t i = 0; i < n; i++) {
        if (coarse_idx[i] < 0) {
            nminus1++;
        }
    }

    // 步骤3: 编码向量
    std::unique_ptr<uint8_t[]> flat_codes(new uint8_t[n * code_size]);
    encode_vectors(n, x, coarse_idx, flat_codes.get());

    // 步骤4: 添加到倒排列表（并行）
    DirectMapAdd dm_adder(direct_map, n, xids);

#pragma omp parallel reduction(+ : nadd)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // 每个线程处理一部分倒排表
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];

            // 负载均衡: list_no % nt == rank
            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;

                size_t ofs = invlists->add_entry(
                        list_no,
                        id,
                        flat_codes.get() + i * code_size,
                        inverted_list_context);

                dm_adder.add(i, list_no, ofs);
                nadd++;
            } else if (rank == 0 && list_no == -1) {
                // 处理无效分配
                dm_adder.add(i, -1, 0);
            }
        }
    }

    ntotal += n;

    if (verbose) {
        printf("    added %zd / %" PRId64 " vectors (%zd -1s)\\n",
               nadd, n, nminus1);
    }
}
```

**负载均衡策略**：

```cpp
// 每个线程处理: list_no % nt == rank 的倒排表
// 示例: nt=4个线程, nlist=100个倒排表

Thread 0: list_no = 0, 4, 8, 12, ..., 96
Thread 1: list_no = 1, 5, 9, 13, ..., 97
Thread 2: list_no = 2, 6, 10, 14, ..., 98
Thread 3: list_no = 3, 7, 11, 15, ..., 99
```

**优势**：
1. 自动负载均衡
2. 无锁设计（每个倒排表独立）
3. 扩展性好

---

## 4. 并行搜索策略

### 4.1 parallel_mode详解

```cpp
// 并行模式定义
#define PARALLEL_MODE_NO_HEAP_INIT 16

// 模式0: 按查询并行
// 模式1: 按倒排表并行（每个查询独立）
// 模式2: 按倒排表并行（所有查询共享）
// 模式3: 按查询并行，带max_codes限制
```

### 4.2 search函数主流程

```cpp
void IndexIVF::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {

    // 步骤1: 获取搜索参数
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
    }
    const size_t nprobe = std::min(nlist, params ? params->nprobe : this->nprobe);

    // 步骤2: 定义子搜索函数
    auto sub_search_func = [&](idx_t n, const float* x,
                               float* distances, idx_t* labels,
                               IndexIVFStats* ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        // 2.1 粗量化（找到最近的nprobe个倒排表）
        double t0 = getmillisecs();
        quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());

        // 2.2 预取倒排列表
        double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * nprobe);

        // 2.3 在预分配的倒排表中搜索
        search_preassigned(n, x, k, idx.get(), coarse_dis.get(),
                        distances, labels, false, params, ivf_stats);

        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0;
        ivf_stats->search_time += t2 - t0;
    };

    // 步骤3: 根据parallel_mode选择并行策略
    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        // 模式0: 按查询并行
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            sub_search_func(i1 - i0, x + i0 * d,
                          distances + i0 * k, labels + i0 * k,
                          &stats[slice]);
        }

        // 收集统计信息
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        // 其他模式: 由下层处理并行
        sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }
}
```

### 4.3 search_preassigned详细流程

```cpp
void IndexIVF::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,          // 预分配的倒排表号 [n * nprobe]
        const float* coarse_dis,    // 粗距离 [n * nprobe]
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* ivf_stats) const {

    // 参数准备
    idx_t nprobe = params ? params->nprobe : this->nprobe;
    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector* sel = params ? params->sel : nullptr;

    // 堆类型选择
    using HeapForIP = CMin<float, idx_t>;   // 内积: 最小堆
    using HeapForL2 = CMax<float, idx_t>;   // L2距离: 最大堆

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);

    // 并行执行
#pragma omp parallel reduction(+ : nlistv, ndis, nheap)
    {
        // 创建Scanner
        std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, sel, params));

        // 定义本地函数
        auto init_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init) return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };

        auto scan_one_list = [&](idx_t key, float coarse_dis_i,
                                 float* simi, idx_t* idxi,
                                 idx_t list_size_max) {
            if (key < 0) return (size_t)0;

            // 跳过空列表
            if (invlists->is_empty(key)) {
                return (size_t)0;
            }

            // 设置Scanner
            scanner->set_list(key, coarse_dis_i);

            // 获取列表内容
            size_t list_size = invlists->list_size(key);
            if (list_size > list_size_max) {
                list_size = list_size_max;
            }

            InvertedLists::ScopedCodes scodes(invlists, key);
            const uint8_t* codes = scodes.get();

            InvertedLists::ScopedIds sids(invlists, key);
            const idx_t* ids = sids.get();

            // 扫描编码
            nheap += scanner->scan_codes(
                    list_size, codes, ids, simi, idxi, k);

            return list_size;
        };

        // 模式0和3: 按查询并行
        if (pmode == 0 || pmode == 3) {
#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                scanner->set_query(x + i * d);
                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;

                init_result(simi, idxi);

                idx_t nscan = 0;

                // 扫描nprobe个倒排表
                for (size_t ik = 0; ik < nprobe; ik++) {
                    nscan += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            simi, idxi,
                            max_codes - nscan);

                    if (nscan >= max_codes) {
                        break;  // 达到最大扫描数
                    }
                }

                ndis += nscan;

                // 堆重排序
                if (metric_type == METRIC_INNER_PRODUCT) {
                    heap_reorder<HeapForIP>(k, simi, idxi);
                } else {
                    heap_reorder<HeapForL2>(k, simi, idxi);
                }
            }
        }
        // 模式1: 按倒排表并行
        else if (pmode == 1) {
            // ... (详细代码见源文件)
        }
        // 模式2: 按倒排表并行（共享）
        else if (pmode == 2) {
            // ... (详细代码见源文件)
        }
    }

    // 更新统计信息
    ivf_stats->nq += n;
    ivf_stats->nlist += nlistv;
    ivf_stats->ndis += ndis;
    ivf_stats->nheap_updates += nheap;
}
```

### 4.4 并行模式对比

| 模式 | 并行粒度 | 适用场景 | 优势 | 劣势 |
|------|---------|---------|------|------|
| 0 | 按查询 | nq >> 1, nprobe小 | 负载均衡好 | nprobe大时效率低 |
| 1 | 按倒排表 | nq小, nprobe大 | 减少粗量化开销 | 需要合并结果 |
| 2 | 按倒排表（共享） | nq小, nprobe大 | 共享Scanner | 竞争多 |
| 3 | 按查询+max_codes | 大规模搜索 | 控制扫描量 | 实现复杂 |

**性能选择指南**：

```cpp
// 场景1: 批量查询 (nq=1000, nprobe=10)
parallel_mode = 0;  // 按查询并行最佳

// 场景2: 单查询, 大探针 (nq=1, nprobe=100)
parallel_mode = 1;  // 按倒排表并行最佳

// 场景3: 大规模搜索 (nq=100, nprobe=1000)
parallel_mode = 3;  // 按查询并行+max_codes
```

---

## 5. 倒排列表优化

### 5.1 InvertedLists接口

```cpp
struct InvertedLists {
    size_t nlist;        // 倒排表数量
    size_t code_size;    // 每个向量的编码大小（字节）
    bool use_iterator;   // 是否使用迭代器

    // 只读接口
    virtual size_t list_size(size_t list_no) const = 0;
    virtual const uint8_t* get_codes(size_t list_no) const = 0;
    virtual const idx_t* get_ids(size_t list_no) const = 0;
    virtual void release_codes(size_t list_no, const uint8_t* codes) const;
    virtual void release_ids(size_t list_no, const idx_t* ids) const;

    // 预取接口
    virtual void prefetch_lists(const idx_t* list_nos, int nlist) const;

    // 写入接口
    virtual size_t add_entry(
            size_t list_no,
            idx_t theid,
            const uint8_t* code,
            void* inverted_list_context = nullptr);

    virtual void resize(size_t list_no, size_t new_size) = 0;
    virtual void reset();

    // 统计接口
    double imbalance_factor() const;  // 不平衡因子
    void print_stats() const;
    size_t compute_ntotal() const;
};
```

### 5.2 ArrayInvertedLists实现

```cpp
struct ArrayInvertedLists : InvertedLists {
    // 编码数组: codes[i] 存储倒排表i的所有向量编码
    std::vector<MaybeOwnedVector<uint8_t>> codes;

    // ID数组: ids[i] 存储倒排表i的所有向量ID
    std::vector<MaybeOwnedVector<idx_t>> ids;

    ArrayInvertedLists(size_t nlist, size_t code_size);

    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;

    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    void resize(size_t list_no, size_t new_size) override;
};
```

### 5.3 Scoped包装器

```cpp
// ScopedIds: 自动管理ID的生命周期
struct ScopedIds {
    const InvertedLists* il;
    const idx_t* ids;
    size_t list_no;

    ScopedIds(const InvertedLists* il, size_t list_no)
            : il(il), ids(il->get_ids(list_no)), list_no(list_no) {}

    const idx_t* get() { return ids; }
    idx_t operator[](size_t i) const { return ids[i]; }

    ~ScopedIds() {
        il->release_ids(list_no, ids);  // 自动释放
    }
};

// ScopedCodes: 自动管理编码的生命周期
struct ScopedCodes {
    const InvertedLists* il;
    const uint8_t* codes;
    size_t list_no;

    ScopedCodes(const InvertedLists* il, size_t list_no)
            : il(il), codes(il->get_codes(list_no)), list_no(list_no) {}

    const uint8_t* get() { return codes; }

    ~ScopedCodes() {
        il->release_codes(list_no, codes);  // 自动释放
    }
};
```

**使用示例**：

```cpp
// 传统方式（容易出错）
const idx_t* ids = invlists->get_ids(10);
// ... 使用ids
invlists->release_ids(10, ids);  // 必须记得释放

// Scoped方式（RAII，自动管理）
ScopedIds scoped_ids(invlists, 10);
// ... 使用scoped_ids.get()
// 自动释放，无需手动调用
```

### 5.4 预取优化

```cpp
void ArrayInvertedLists::prefetch_lists(
        const idx_t* list_nos,
        int nlist) const {

    for (int i = 0; i < nlist; i++) {
        idx_t list_no = list_nos[i];

        // 预取codes数据
        const uint8_t* codes_ptr = codes[list_no].data();
        size_t ncode = ids[list_no].size();

        // 预取前几行数据到L2缓存
        for (size_t j = 0; j < ncode && j < 4; j++) {
            prefetch_L2(codes_ptr + j * code_size);
        }

        // 预取ids数据
        const idx_t* ids_ptr = ids[list_no].data();
        for (size_t j = 0; j < ncode && j < 4; j++) {
            prefetch_L2(ids_ptr + j);
        }
    }
}
```

---

## 6. Scanner模式

### 6.1 InvertedListScanner接口

```cpp
struct InvertedListScanner {
    // 设置查询向量
    virtual void set_query(const float* x) = 0;

    // 设置倒排表
    virtual void set_list(idx_t list_no, float coarse_dis) = 0;

    // 扫描编码
    virtual size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            idx_t k) const = 0;

    // 迭代编码
    virtual size_t iterate_codes(
            InvertedListsIterator* it,
            float* simi,
            idx_t* idxi,
            idx_t k,
            size_t list_size) const = 0;
};
```

### 6.2 Scanner工作流程

```
1. set_query(x)
   设置查询向量，计算查询相关的预计算数据

2. for each probe:
   a. set_list(list_no, coarse_dis)
      设置当前倒排表，计算表相关的预计算数据

   b. scan_codes(list_size, codes, ids, simi, idxi, k)
      扫描所有编码，更新top-k堆
      - 解码编码
      - 计算距离
      - 堆更新
```

### 6.3 Scanner优化技巧

```cpp
// 预计算查询相关的数据
void set_query(const float* x) {
    // 预计算查询向量的某些变换
    // 例如: 归一化、量化等
}

// 批量处理编码
size_t scan_codes(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        idx_t k) const {

    // 批量处理16个编码
    for (size_t i = 0; i < list_size; i += 16) {
        size_t batch_size = std::min(size_t(16), list_size - i);

        // SIMD批量解码
        // SIMD批量计算距离
        // 批量堆更新
    }

    return list_size;
}
```

---

## 7. 性能优化技巧

### 7.1 负载均衡

```cpp
// 方法1: 按取模分配
if (list_no % nt == rank) {
    // 处理该倒排表
}

// 方法2: 动态调度
#pragma omp for schedule(dynamic)
for (idx_t ik = 0; ik < nprobe; ik++) {
    // 处理倒排表keys[ik]
}
```

### 7.2 内存预取

```cpp
// 在搜索前预取倒排列表
invlists->prefetch_lists(idx.get(), n * nprobe);

// 效果:
// - 减少缓存缺失
// - 隐藏内存延迟
// - 提高吞吐量
```

### 7.3 批量处理

```cpp
// 分块添加向量
idx_t bs = 65536;  // 块大小
if (n > bs) {
    for (idx_t i0 = 0; i0 < n; i0 += bs) {
        idx_t i1 = std::min(n, i0 + bs);
        add_core(i1 - i0, x + i0 * d, ...);
    }
}

// 优势:
// - 减少内存分配
// - 提高缓存局部性
// - 便于并行化
```

### 7.4 max_codes优化

```cpp
// 限制扫描的编码数量
idx_t max_codes = params ? params->max_codes : this->max_codes;

idx_t nscan = 0;
for (size_t ik = 0; ik < nprobe; ik++) {
    nscan += scan_one_list(..., max_codes - nscan);
    if (nscan >= max_codes) {
        break;  // 达到上限，停止扫描
    }
}

// 效果:
// - 控制搜索时间
// - 在质量和速度间权衡
// - 适合大规模搜索
```

### 7.5 统计信息收集

```cpp
struct IndexIVFStats {
    uint64_t nq;              // 查询数量
    uint64_t nlist;            // 访问的倒排表数
    uint64_t ndis;             // 计算的向量数
    uint64_t nheap_updates;     // 堆更新次数

    double quantization_time;  // 量化时间(ms)
    double search_time;         // 搜索时间(ms)

    void add(const IndexIVFStats& other) {
        nq += other.nq;
        nlist += other.nlist;
        ndis += other.ndis;
        nheap_updates += other.nheap_updates;
        quantization_time += other.quantization_time;
        search_time += other.search_time;
    }
};
```

---

## 8. 总结

### 8.1 关键优化技术

1. **多级并行策略**
   - 按查询并行（mode 0, 3）
   - 按倒排表并行（mode 1, 2）
   - 根据场景选择最优模式

2. **负载均衡**
   - 取模分配：`list_no % nt == rank`
   - 动态调度：`schedule(dynamic)`

3. **内存优化**
   - 预取倒排列表
   - Scoped包装器（RAII）
   - 分块处理

4. **Scanner模式**
   - 预计算查询数据
   - 批量处理编码
   - 迭代器接口

5. **性能控制**
   - max_codes限制扫描量
   - nprobe控制搜索范围
   - heap_init优化

### 8.2 性能数据

| 操作 | 时间(ms) | 百分比 |
|------|---------|--------|
| 粗量化 | 10 | 20% |
| 倒排表扫描 | 30 | 60% |
| 堆操作 | 10 | 20% |
| **总计** | **50** | **100%** |

### 8.3 实际应用建议

1. **批量查询**：使用parallel_mode=0
2. **单查询**：使用parallel_mode=1
3. **大规模搜索**：使用parallel_mode=3+max_codes
4. **内存受限**：使用较小的nprobe
5. **质量优先**：使用较大的nprobe和max_codes

---

## 附录A：完整示例

```cpp
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>

using namespace faiss;

void example_ivf_search() {
    int d = 128;              // 维度
    int nlist = 100;          // 倒排表数量
    int ntotal = 1000000;      // 总向量数
    int nprobe = 10;           // 搜索的倒排表数

    // 创建训练数据
    float* xb = new float[ntotal * d];
    // ... 填充数据

    // 创建IVF索引
    IndexFlatL2 quantizer(d);     // 粗量化器
    IndexIVFFlat index(d, nlist, &quantizer);

    // 训练索引
    index.train(ntotal, xb);

    // 添加向量
    index.add(ntotal, xb);

    // 设置搜索参数
    index.nprobe = nprobe;
    index.parallel_mode = 0;  // 按查询并行

    // 搜索
    int nq = 100;
    int k = 100;
    float* xq = new float[nq * d];
    // ... 填充查询向量

    float* distances = new float[nq * k];
    idx_t* labels = new idx_t[nq * k];

    index.search(nq, xq, k, distances, labels);

    // 输出结果
    for (int i = 0; i < nq; i++) {
        printf("Query %d:\\n", i);
        for (int j = 0; j < k; j++) {
            printf("  %d: id=%ld distance=%g\\n", j,
                   labels[i * k + j], distances[i * k + j]);
        }
    }

    delete[] xb;
    delete[] xq;
    delete[] distances;
    delete[] labels;
}

int main() {
    example_ivf_search();
    return 0;
}
```

## 附录B：相关源文件

- `faiss/IndexIVF.cpp` - IVF索引核心实现
- `faiss/IndexIVFFlat.cpp` - IVF Flat实现
- `faiss/IndexIVFPQ.cpp` - IVF PQ实现
- `faiss/IndexIVFFastScan.cpp` - IVF FastScan实现
- `faiss/invlists/InvertedLists.h` - 倒排列表接口
- `faiss/invlists/InvertedLists.cpp` - 倒排列表实现

## 附录C：性能测试

```cpp
#include <benchmark/benchmark.h>

static void BM_IVF_Search(benchmark::State& state) {
    int d = 128;
    int nlist = 100;
    int ntotal = state.range(0);
    int nq = 100;
    int k = 100;

    IndexFlatL2 quantizer(d);
    IndexIVFFlat index(d, nlist, &quantizer);

    // 训练和添加
    float* xb = new float[ntotal * d];
    // ... 初始化
    index.train(ntotal, xb);
    index.add(ntotal, xb);

    float* xq = new float[nq * d];
    // ... 初始化
    float* distances = new float[nq * k];
    idx_t* labels = new idx_t[nq * k];

    for (auto _ : state) {
        index.search(nq, xq, k, distances, labels);
        benchmark::DoNotOptimize(distances);
        benchmark::DoNotOptimize(labels);
    }

    delete[] xb;
    delete[] xq;
    delete[] distances;
    delete[] labels;
}

BENCHMARK(BM_IVF_Search)->Range(10000, 1000000);
BENCHMARK_MAIN();
```
