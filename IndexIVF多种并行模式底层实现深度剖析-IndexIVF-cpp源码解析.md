# IndexIVF 多种并行模式底层实现深度剖析 - IndexIVF.cpp 源码解析

## 1. 概述

IndexIVF 是 Faiss 中倒排文件索引的核心实现，支持**多种并行模式**以适应不同的工作负载。其设计充分考虑了负载均衡、缓存亲和性和 NUMA 优化。

### 核心特性
- **4种并行模式**: 针对不同查询和数据规模的优化策略
- **动态调度**: 负载均衡的精细化控制
- **Prefetch 优化**: 异步预取倒排列表数据
- **Lambda 函数**: 代码复用和模板化设计
- **异常安全**: 并行区域的异常捕获和传播

## 2. 并行模式详解

### 2.1 并行模式定义 (IndexIVF.h)

```cpp
enum ParallelMode {
    PARALLEL_MODE_NO_HEAP_INIT = 1,    // 不初始化堆
    PARALLEL_MODE_CPU_TO_SHARE_CACHE = 0, // CPU 级并行
    PARALLEL_MODE_CPU_TO_SHARE_CACHE_PARALLEL_PROBE = 1, // CPU 并行 + probe 并行
    PARALLEL_MODE_CPU_TO_SHARE_CACHE_QUERY_PARALLEL_PROBE = 2, // CPU + query 并行 + probe 并行
    PARALLEL_MODE_CPU_TO_SHARE_CACHE_QUERY_PARALLEL_PROBE_SPLIT_CACHE = 3 // CPU + query/probe 并行
};
```

### 2.2 模式选择逻辑 (IndexIVF.cpp:453-457)

```cpp
[[maybe_unused]] bool do_parallel = omp_get_max_threads() >= 2 &&
        (pmode == 0           ? false
                 : pmode == 3 ? n > 1
                 : pmode == 1 ? nprobe > 1
                              : nprobe * n > 1);
```

**决策矩阵**:
| 模式 | 条件 | 使用场景 |
|------|------|---------|
| 0 | `n > 1` | 多查询，每个查询独立处理 |
| 1 | `nprobe > 1` | 单查询，多 probe 并行 |
| 2 | `nprobe * n > 1` | 多查询多 probe，细粒度并行 |
| 3 | `n > 1` | 多查询，split cache 优化 |

## 3. 模式 0: 基础并行模式

### 3.1 实现代码 (IndexIVF.cpp:594-631)

```cpp
if (pmode == 0 || pmode == 3) {
#pragma omp for
    for (idx_t i = 0; i < n; i++) {
        if (interrupt) {
            continue;
        }

        // 设置查询
        scanner->set_query(x + i * d);
        float* simi = distances + i * k;
        idx_t* idxi = labels + i * k;

        // 初始化结果堆
        init_result(simi, idxi);

        idx_t nscan = 0;

        // 遍历 probe 个倒排列表
        for (size_t ik = 0; ik < nprobe; ik++) {
            nscan += scan_one_list(
                    keys[i * nprobe + ik],      // 倒排列表键
                    coarse_dis[i * nprobe + ik], // 粗距离
                    simi,
                    idxi,
                    max_codes - nscan);        // 剩余扫描预算

            // 达到扫描预算后停止
            if (nscan >= max_codes) {
                break;
            }
        }

        ndis += nscan;

        // 排序结果堆
        reorder_result(simi, idxi);

        // 检查中断
        if (InterruptCallback::is_interrupted()) {
            interrupt = true;
        }
    } // parallel for
}
```

**特点**:
1. **查询级并行**: 每个查询独立分配给不同线程
2. **串行 probe**: 每个查询内部串行扫描 probe
3. **独立堆**: 每个查询有自己的结果堆，无需同步
4. **适用场景**: 多查询小查询 (n > 1, nprobe 较小)

**伪代码**:
```
For each query i in parallel:
    Initialize heap[i]
    For each probe ik:
        Scan list keys[i, ik]
        Update heap[i]
    Sort heap[i]
Return heap[i][0..k-1]
```

## 4. 模式 1: Probe 并行模式

### 4.1 实现代码 (IndexIVF.cpp:631-666)

```cpp
else if (pmode == 1) {
    std::vector<idx_t> local_idx(k);
    std::vector<float> local_dis(k);

    for (size_t i = 0; i < n; i++) {
        scanner->set_query(x + i * d);
        init_result(local_dis.data(), local_idx.data());

        // 并行扫描所有 probe
#pragma omp for schedule(dynamic)
        for (idx_t ik = 0; ik < nprobe; ik++) {
            ndis += scan_one_list(
                    keys[i * nprobe + ik],
                    coarse_dis[i * nprobe + ik],
                    local_dis.data(),
                    local_idx.data(),
                    unlimited_list_size);  // 不限制扫描数量
        }

        // 合并线程局部结果到全局堆
        float* simi = distances + i * k;
        idx_t* idxi = labels + i * k;

#pragma omp single
        init_result(simi, idxi);

#pragma omp barrier
#pragma omp critical
        {
            add_local_results(
                    local_dis.data(), local_idx.data(), simi, idxi);
        }
#pragma omp barrier
#pragma omp single
        reorder_result(simi, idxi);
    }
}
```

**特点**:
1. **Probe 级并行**: 不同 probe 分配给不同线程
2. **查询串行**: 每个查询串行处理所有 probe
3. **局部堆**: 线程局部结果，需要合并
4. **动态调度**: `schedule(dynamic)` 负载均衡

**数据流**:
```
Thread 0:               Thread 1:
  Probe 0 → local_heap    Probe 1 → local_heap
  ↓                      ↓
  critical section        critical section
  ↓                      ↓
  merge → global_heap ← merge
```

**适用场景**: 单查询，多 probe (nprobe > 1)

## 5. 模式 2: Query+Probe 并行模式

### 5.1 实现代码 (IndexIVF.cpp:667-701)

```cpp
else if (pmode == 2) {
    std::vector<idx_t> local_idx(k);
    std::vector<float> local_dis(k);

#pragma omp single
    for (int64_t i = 0; i < n; i++) {
        init_result(distances + i * k, labels + i * k);
    }

    // 最细粒度并行：(查询, probe) 对
#pragma omp for schedule(dynamic)
    for (int64_t ij = 0; ij < n * nprobe; ij++) {
        size_t i = ij / nprobe;  // 查询索引
        size_t ik = ij % nprobe;  // probe 索引

        if (qres == nullptr || qres->qno != i) {
            qres = &pres.new_result(i);
            scanner->set_query(x + i * d);
        }

        ndis += scan_one_list(
                keys[ij],
                coarse_dis[ij],
                local_dis.data(),
                local_idx.data(),
                unlimited_list_size);

#pragma omp critical
        {
            // 直接添加到全局结果
            add_local_results(
                    local_dis.data(),
                    local_idx.data(),
                    distances + i * k,
                    labels + i * k);
        }
    }

#pragma omp single
    for (int64_t i = 0; i < n; i++) {
        reorder_result(distances + i * k, labels + i * k);
    }
}
```

**特点**:
1. **最细粒度并行**: (查询, probe) 二维并行
2. **频繁临界区**: 每次扫描都需要同步
3. **全局堆**: 直接更新全局结果堆
4. **动态调度**: 负载均衡

**数据流**:
```
Thread 0:               Thread 1:
  (Q0, P0) → critical  (Q1, P0) → critical
  (Q0, P1) → critical  (Q0, P2) → critical
  ↓                      ↓
  global_heap[0]     global_heap[1]
```

**适用场景**: 多查询多 probe，需要最大并行度

## 6. 优化技术

### 6.1 Prefetch 优化 (IndexIVF.cpp:337)

```cpp
double t1 = getmillisecs();
invlists->prefetch_lists(idx.get(), n * nprobe);
```

**prefetch_lists 实现** (推测):
```cpp
void InvertedLists::prefetch_lists(const idx_t* keys, size_t nkeys) {
    for (size_t i = 0; i < nkeys; i++) {
        idx_t key = keys[i];
        size_t list_size = list_size(key);
        const uint8_t* codes = get_codes(key);

        // 预取列表数据到 L2/L3 缓存
        for (size_t j = 0; j < list_size; j += CACHE_LINE_SIZE / code_size) {
            _mm_prefetch(codes + j * code_size, _MM_HINT_T0);
        }
    }
}
```

**优势**:
1. **隐藏延迟**: 预取与距离计算重叠
2. **缓存友好**: 扫描时数据已在缓存
3. **自动管理**: 实现细节对用户透明

### 6.2 Lambda 函数设计 (IndexIVF.cpp:473-506)

```cpp
// 初始化 + 排序结果堆
auto init_result = [&](float* simi, idx_t* idxi) {
    if (!do_heap_init) {
        return;
    }
    if (metric_type == METRIC_INNER_PRODUCT) {
        heap_heapify<HeapForIP>(k, simi, idxi);
    } else {
        heap_heapify<HeapForL2>(k, simi, idxi);
    }
};

// 添加局部结果到全局堆
auto add_local_results = [&](const float* local_dis,
                             const idx_t* local_idx,
                             float* simi,
                             idx_t* idxi) {
    if (metric_type == METRIC_INNER_PRODUCT) {
        heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
    } else {
        heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
    }
};

// 排序结果堆
auto reorder_result = [&](float* simi, idx_t* idxi) {
    if (!do_heap_init) {
        return;
    }
    if (metric_type == METRIC_INNER_PRODUCT) {
        heap_reorder<HeapForIP>(k, simi, idxi);
    } else {
        heap_reorder<HeapForL2>(k, simi, idxi);
    }
};
```

**设计优势**:
1. **代码复用**: 不同并行模式共享相同逻辑
2. **类型安全**: 编译期类型检查
3. **零开销**: Lambda 内联优化
4. **可读性**: 意图清晰，易于维护

### 6.3 scan_one_list 函数 (IndexIVF.cpp:510-588)

```cpp
auto scan_one_list = [&](idx_t key,
                         float coarse_dis_i,
                         float* simi,
                         idx_t* idxi,
                         idx_t list_size_max) {
    if (key < 0) {
        return (size_t)0;  // 无效 probe
    }

    // 检查空列表
    if (invlists->is_empty(key, inverted_list_context)) {
        return (size_t)0;
    }

    scanner->set_list(key, coarse_dis_i);
    nlistv++;

    try {
        if (invlists->use_iterator) {
            // 迭代器模式
            std::unique_ptr<InvertedListsIterator> it(
                    invlists->get_iterator(key, inverted_list_context));

            nheap += scanner->iterate_codes(
                    it.get(), simi, idxi, k, list_size);

            return list_size;
        } else {
            // 直接扫描模式
            size_t list_size = invlists->list_size(key);
            if (list_size > list_size_max) {
                list_size = list_size_max;
            }

            InvertedLists::ScopedCodes scodes(invlists, key);
            const uint8_t* codes = scodes.get();

            std::unique_ptr<InvertedLists::ScopedIds> sids;
            const idx_t* ids = nullptr;

            if (!store_pairs) {
                sids = std::make_unique<InvertedLists::ScopedIds>(
                        invlists, key);
                ids = sids->get();
            }

            // IDSelectorRange 优化
            if (selr) { // IDSelectorRange
                size_t jmin, jmax;
                selr->find_sorted_ids_bounds(
                        list_size, ids, &jmin, &jmax);
                list_size = jmax - jmin;
                if (list_size == 0) {
                    return (size_t)0;
                }
                codes += jmin * code_size;
                ids += jmin;
            }

            nheap += scanner->scan_codes(
                    list_size, codes, ids, simi, idxi, k);

            return list_size;
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(exception_mutex);
        exception_string =
                demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
        interrupt = true;
        return size_t(0);
    }
};
```

**关键点**:
1. **两种扫描模式**: Iterator vs Direct
2. **IDSelectorRange 优化**: 利用排序特性加速范围查询
3. **异常处理**: 并行区域的异常安全
4. **list_size_max**: 扫描预算控制

### 6.4 负载均衡 (IndexIVF.cpp:256-270)

```cpp
// add_core 中的负载均衡
#pragma omp parallel reduction(+ : nadd)
{
    int nt = omp_get_num_threads();
    int rank = omp_get_thread_num();

    // 每个线程负责特定的倒排列表
    for (size_t i = 0; i < n; i++) {
        idx_t list_no = coarse_idx[i];
        // 负载均衡：list_no % nt == rank
        if (list_no >= 0 && list_no % nt == rank) {
            idx_t id = xids ? xids[i] : ntotal + i;
            size_t ofs = invlists->add_entry(
                    list_no, id, flat_codes.get() + i * code_size,
                    inverted_list_context);

            dm_adder.add(i, list_no, ofs);
            nadd++;
        } else if (rank == 0 && list_no == -1) {
            dm_adder.add(i, -1, 0);
        }
    }
}
```

**负载均衡策略**:
```
Thread 0: 处理 list_no = 0, nt, 2*nt, 3*nt, ...
Thread 1: 处理 list_no = 1, nt+1, 2*nt+1, ...
Thread 2: �理 list_no = 2, nt+2, 2*nt+2, ...
...
```

**效果**:
- 均匀分配工作到各线程
- 减少 False Sharing
- 提高 L2 缓存局部性

### 6.5 异常处理 (IndexIVF.cpp:356-383)

```cpp
if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
    int nt = std::min(omp_get_max_threads(), int(n));
    std::vector<IndexIVFStats> stats(nt);
    std::mutex exception_mutex;
    std::string exception_string;

#pragma omp parallel for if (nt > 1)
    for (idx_t slice = 0; slice < nt; slice++) {
        IndexIVFStats local_stats;
        idx_t i0 = n * slice / nt;
        idx_t i1 = n * (slice + 1) / nt;
        if (i1 > i0) {
            try {
                sub_search_func(
                        i1 - i0,
                        x + i0 * d,
                        distances + i0 * k,
                        labels + i0 * k,
                        &stats[slice]);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string = e.what();
            }
        }
    }

    if (!exception_string.empty()) {
        FAISS_THROW_MSG(exception_string.c_str());
    }

    // 收集统计信息
    for (idx_t slice = 0; slice < nt; slice++) {
        indexIVF_stats.add(stats[slice]);
    }
}
```

**异常处理流程**:
```
Try:
    Thread 0: sub_search_func() → success or exception
    Thread 1: sub_search_func() → success or exception
    Thread 2: sub_search_func() → success or exception
Catch:
    - 捕获异常到 exception_string
    - 中断其他线程
    - 重新抛出异常
```

## 7. 范围搜索实现

### 7.1 range_search_preassigned (IndexIVF.cpp:764-921)

```cpp
void IndexIVF::range_search_preassigned(
        idx_t nx,
        const float* x,
        float radius,
        const idx_t* keys,
        const float* coarse_dis,
        RangeSearchResult* result,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    // ... 参数检查

    std::vector<RangeSearchPartialResult*> all_pres(omp_get_max_threads());

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;

    [[maybe_unused]] bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 3           ? false
                     : pmode == 0 ? nx > 1
                     : pmode == 1 ? nprobe > 1
                                  : nprobe * nx > 1);

    void* inverted_list_context =
            params ? params->inverted_list_context : nullptr;

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis)
    {
        RangeSearchPartialResult pres(result);
        std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, sel, params));
        all_pres[omp_get_thread_num()] = &pres;

        // 扫描列表函数
        auto scan_list_func = [&](size_t i, size_t ik, RangeQueryResult& qres) {
            idx_t key = keys[i * nprobe + ik];
            if (key < 0) {
                return;
            }

            if (invlists->is_empty(key, inverted_list_context)) {
                return;
            }

            try {
                scanner->set_list(key, coarse_dis[i * nprobe + ik]);
                if (invlists->use_iterator) {
                    std::unique_ptr<InvertedListsIterator> it(
                            invlists->get_iterator(key, inverted_list_context));
                    scanner->iterate_codes_range(
                            it.get(), radius, qres, list_size);
                } else {
                    InvertedLists::ScopedCodes scodes(invlists, key);
                    InvertedLists::ScopedIds ids(invlists, key);
                    list_size = invlists->list_size(key);

                    scanner->scan_codes_range(
                            list_size, scodes.get(), ids.get(), radius, qres);
                }
                nlistv++;
                ndis += list_size;
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
            }
        };

        // 根据并行模式执行
        if (parallel_mode == 0) {
#pragma omp for
            for (idx_t i = 0; i < nx; i++) {
                scanner->set_query(x + i * d);
                RangeQueryResult& qres = pres.new_result(i);

                for (size_t ik = 0; ik < nprobe; ik++) {
                    scan_list_func(i, ik, qres);
                }
            }

        } else if (parallel_mode == 1) {
            for (size_t i = 0; i < nx; i++) {
                scanner->set_query(x + i * d);
                RangeQueryResult& qres = pres.new_result(i);

#pragma omp for schedule(dynamic)
                for (idx_t ik = 0; ik < nprobe; ik++) {
                    scan_list_func(i, ik, qres);
                }
            }

        } else if (parallel_mode == 2) {
            RangeQueryResult* qres = nullptr;

#pragma omp for schedule(dynamic)
            for (idx_t iik = 0; iik < nx * (idx_t)nprobe; iik++) {
                idx_t i = iik / (idx_t)nprobe;
                idx_t ik = iik % (idx_t)nprobe;
                if (qres == nullptr || qres->qno != i) {
                    qres = &pres.new_result(i);
                    scanner->set_query(x + i * d);
                }
                scan_list_func(i, ik, *qres);
            }
        }

        // 合并部分结果
#pragma omp single
        RangeSearchPartialResult::merge(all_pres, false);
#pragma omp barrier
    }

    if (interrupt) {
        FAISS_THROW_MSG("computation interrupted");
    }
}
```

**RangeQueryResult 优化**:
1. **增量分配**: 动态扩展结果缓冲区
2. **延迟排序: 高效合并策略
3. **批量添加**: 批量插入结果

## 8. 性能优化总结

### 8.1 并行模式选择指南

| 场景 | 推荐模式 | 理由 |
|------|---------|------|
| 多查询，小 nprobe | 0 | 查询级并行，低开销 |
| 单查询，大 nprobe | 1 | Probe 级并行，良好负载均衡 |
| 多查询，大 nprobe | 2 | 最大并行度，同步开销可接受 |
| 多查询，split cache | 3 | 缓存分裂优化 |

### 8.2 缓存优化策略

```cpp
// 模式 0: 每个查询独立的缓存行
Thread 0 → Cache Set 0 → Query 0
Thread 1 → Cache Set 1 → Query 1
Thread 2 → Cache Set 2 → Query 2
// 无冲突，高局部性

// 模式 1: 共享缓存但不同 probe
Thread 0 → List 0 → Cache Line 0
Thread 1 → List 1 → Cache Line 0 (after remap)
// 可能有 False Sharing，但动态调度缓解

// 模式 2: 细粒度任务
Thread 0 → (Q0, P0), (Q0, P1), (Q0, P2)...
Thread 1 → (Q1, P0), (Q1, P1), (Q1, P2)...
// 缓存竞争最大，但并行度最高
```

### 8.3 内存访问模式

```cpp
// 模式 0 内存访问 (最优)
Thread 0: x[0:d], keys[0:nprobe], distances[0:k], labels[0:k]
Thread 1: x[1*d], keys[nprobe:2*nprobe], ...
// 完全独立，无共享写

// 模式 1 内存访问 (中等)
Thread 0: x[0:d], local_dis[0:k], local_idx[0:k]
Thread 1: x[0:d], local_dis[1:k], local_idx[1:k]
// 读共享（x），写独立

// 模式 2 内存访问 (高竞争)
All threads: distances[], labels[]
// 频繁 critical section
```

### 8.4 同步开销分析

| 模式 | 同步原语 | 开销估计 |
|------|----------|---------|
| 0 | 无 | 0 |
| 1 | barrier + critical | 中等 |
| 2 | critical (频繁) | 高 |
| 3 | 无（split cache） | 0 |

## 9. InvertedListScanner 统一接口

### 9.1 scan_codes 实现 (IndexIVF.cpp:1301-1348)

```cpp
size_t InvertedListScanner::scan_codes(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        size_t k) const {
    size_t nup = 0;

    if (!keep_max) {
        // Min-heap: 寻找最小 k 个
        for (size_t j = 0; j < list_size; j++) {
            if (sel != nullptr) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                if (!sel->is_member(id)) {
                    codes += code_size;
                    continue;
                }
            }

            float dis = distance_to_code(codes);
            if (dis < simi[0]) {  // 比当前最大距离小
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                maxheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
            codes += code_size;
        }
    } else {
        // Max-heap: 寻找最大 k 个
        for (size_t j = 0; j < list_size; j++) {
            if (sel != nullptr) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                if (!sel->is_member(id)) {
                    codes += code_size;
                    continue;
                }
            }

            float dis = distance_to_code(codes);
            if (dis > simi[0]) {  // 比当前最小距离大
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                minheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
            codes += code_size;
        }
    }
    return nup;
}
```

**关键设计**:
1. **模板化**: keep_max 控制最小堆/最大堆
2. **IDSelector 集成**: 支持过滤
3. **store_pairs**: 存储(list_no, offset)用于重建
4. **distance_to_code**: 虚函数，支持不同度量

### 9.2 iterate_codes 实现 (IndexIVF.cpp:1350-1381)

```cpp
size_t InvertedListScanner::iterate_codes(
        InvertedListsIterator* it,
        float* simi,
        idx_t* idxi,
        size_t k,
        size_t& list_size) const {
    size_t nup = 0;
    list_size = 0;

    if (!keep_max) {
        for (; it->is_available(); it->next()) {
            auto id_and_codes = it->get_id_and_codes();
            float dis = distance_to_code(id_and_codes.second);
            if (dis < simi[0]) {
                maxheap_replace_top(k, simi, idxi, dis, id_and_codes.first);
                nup++;
            }
            list_size++;
        }
    } else {
        for (; it->is_available(); it->next()) {
            auto id_and_codes = it->get_id_and_codes();
            float dis = distance_to_code(id_and_codes.second);
            if (dis > simi[0]) {
                minheap_replace_top(k, simi, idxi, dis, id_and_codes.first);
                nup++;
            }
            list_size++;
        }
    }
    return nup;
}
```

**迭代器优势**:
1. **懒加载**: 按需加载代码
2. **流式处理**: 不需要一次性加载整个列表
3. **内存友好**: 只保留当前处理的代码

## 10. 性能特征

### 10.1 时间复杂度

| 操作 | 模式 0 | 模式 1 | 模式 2 |
|------|--------|--------|--------|
| **量化** | O(n × nprobe × d) | O(n × nprobe × d) | O(n × nprobe × d) |
| **扫描** | O(n × nprobe × avg_list_size) | O(n × nprobe × avg_list_size / nt) | O(n × nprobe × avg_list_size / nt) |
| **合并** | 0 | O(n × k / nt) | O(n × nprobe × k / nt) |
| **排序** | O(n × k × log k) | O(n × k × log k) | O(n × k × log k) |

### 10.2 空间复杂度

| 数据结构 | 大小 |
|----------|------|
| **keys** | n × nprobe × sizeof(idx_t) |
| **coarse_dis** | n × nprobe × sizeof(float) |
| **local_dis (模式 1)** | k × sizeof(float) |
| **local_idx (模式 1)** | k × sizeof(idx_t) |
| **parallel_mode 0** | 无额外空间 |
| **parallel_mode 1** | n × k (临时) |
| **parallel_mode 2** | k (临时) |

### 10.3 参数调优建议

| 参数 | 推荐值 | 影响 |
|------|-------|------|
| nprobe | 1-2048 | 更大 = 更高召回，更慢 |
| parallel_mode | 0 (多查询), 1 (单查询大 nprobe), 2 (多查询大 nprobe) | 根据负载选择 |
| max_codes | 0 (无限制) 或 限制值 | 控制扫描量 |

## 11. 关键源码位置

| 文件 | 函数 | 行号 |
|------|------|------|
| `IndexIVF.cpp` | `search()` | 299-394 |
| `IndexIVF.cpp` | `search_preassigned()` | 396-722 |
| `IndexIVF.cpp` | `range_search_preassigned()` | 764-921 |
| `IndexIVF.cpp` | `add_core()` | 206-281 |
| `IndexIVF.cpp` | scan_one_list (lambda) | 510-588 |
| `IndexIVF.cpp` | InvertedListScanner::scan_codes() | 1301-1348 |
| `IndexIVF.cpp` | InvertedListScanner::iterate_codes() | 1350-1381 |

## 12. 总结

IndexIVF 的多种并行模式展示了 Faiss 对**不同工作负载的精细优化**：

1. **模式 0**: 查询级并行，适合多查询小搜索
2. **模式 1**: Probe 级并行，适合单查询多 probe
3. **模式 2**: Query+Probe 并行，最大化并行度

通过**Lambda 函数**实现代码复用，通过**Prefetch**隐藏内存延迟，通过**细粒度锁**实现负载均衡。这些优化技术使得 IndexIVF 能够高效处理从单查询到大规模批量查询的各类工作负载。
