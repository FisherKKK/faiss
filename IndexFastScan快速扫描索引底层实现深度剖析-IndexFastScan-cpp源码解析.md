# IndexFastScan快速扫描索引底层实现深度剖析 - IndexFastScan.cpp源码解析

## 概述

`faiss/IndexFastScan.cpp` 是Faiss中FastScan索引的核心实现，通过4-bit Product Quantization和SIMD优化的批量扫描实现高效向量检索。FastScan是Faiss中最快的索引之一，专门针对大规模数据集的近似最近邻搜索进行了极致优化。本文档深入剖析其底层实现细节、批量处理策略和多实现变体。

---

## 1. FastScan索引架构

### 1.1 核心数据结构

```cpp
class IndexFastScan : public Index {
public:
    // 基础参数
    size_t d;          // 向量维度
    size_t M;          // PQ子量化器数量
    size_t nbits;      // 每个量化器的位数（固定为4）
    size_t ksub;       // 每个子量化器的质心数 (2^nbits = 16)
    size_t code_size;  // 每个向量的编码大小（字节）
    size_t bbs;        // 块大小（必须是32的倍数）
    size_t M2;         // 向上取整到2的M

    // 数据存储
    AlignedTable<uint8_t> codes;  // 打包的4-bit PQ编码

    // 元数据
    size_t ntotal;     // 向量总数
    size_t ntotal2;    // 向上取整到bbs的向量总数
};
```

**参数关系：**

```cpp
// 示例：d=128, M=32, nbits=4, bbs=32
ksub = 2^4 = 16;                  // 每个子量化器16个质心
code_size = (32 * 4 + 7) / 8 = 16;  // 每个向量16字节
M2 = roundup(32, 2) = 32;         // 向上取整到2

// 编码布局：
// 每个向量32个4-bit值 = 16字节
// 32个子量化器 × 4-bit = 128位
```

### 1.2 初始化流程

```cpp
void IndexFastScan::init_fastscan(
        int d,
        size_t M_init,
        size_t nbits_init,
        MetricType metric,
        int bbs) {
    FAISS_THROW_IF_NOT(nbits_init == 4);  // 固定4-bit
    FAISS_THROW_IF_NOT(bbs % 32 == 0);   // 必须是32的倍数

    this->d = d;
    this->M = M_init;
    this->nbits = nbits_init;
    this->metric_type = metric;
    this->bbs = bbs;

    ksub = (1 << nbits_init);  // 16

    // 计算编码大小
    code_size = (M_init * nbits_init + 7) / 8;

    // 重置状态
    ntotal = ntotal2 = 0;
    M2 = roundup(M_init, 2);  // 向上取整到2的倍数
    is_trained = false;
}
```

**块大小约束：**

```
为什么bbs必须是32的倍数？

原因1: SIMD对齐要求
- AVX2: 256-bit = 32字节
- AVX512: 512-bit = 64字节
- 32字节是最小对齐单位

原因2: 批处理效率
- 每次处理32个向量可以充分利用SIMD
- 避免边界检查和剩余元素处理

原因3: 内存访问模式
- 32个向量 × 16字节 = 512字节
- 正好覆盖多个cache line
```

---

## 2. 向量添加与编码

### 2.1 add函数实现

```cpp
void IndexFastScan::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);

    // ==================== 大批量分块处理 ====================
    constexpr idx_t bs = 65536;  // 65536个向量/块
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            add(i1 - i0, x + i0 * d);  // 递归处理
        }
        return;
    }

    // ==================== 编码向量 ====================
    AlignedTable<uint8_t> tmp_codes(n * code_size);
    compute_codes(tmp_codes.get(), n, x);

    // ==================== 扩展存储空间 ====================
    ntotal2 = roundup(ntotal + n, bbs);  // 向上取整到bbs的倍数
    size_t new_size = ntotal2 * M2 / 2;     // 计算需要的空间
    size_t old_size = codes.size();

    if (new_size > old_size) {
        codes.resize(new_size);
        memset(codes.get() + old_size, 0, new_size - old_size);  // 清零
    }

    // ==================== 打包编码 ====================
    pq4_pack_codes_range(
            tmp_codes.get(),
            M,
            ntotal,         // 起始位置
            ntotal + n,     // 结束位置
            bbs,
            M2,
            codes.get());

    ntotal += n;
}
```

**内存分配策略：**

```
初始状态 (ntotal=0, n=1000, bbs=32):

ntotal2 = roundup(0 + 1000, 32) = 992
new_size = 992 * 32 / 2 = 15872

codes分配: 0 -> 15872字节

为什么要向上取整？
- 确保每个块的边界对齐
- 简化SIMD循环的逻辑
- 避免处理不完整的块
```

### 2.2 compute_codes实现

```cpp
// compute_codes由子类实现，例如IndexFastScanIVF
void IndexFastScanIVF::compute_codes(
        uint8_t* codes,
        size_t n,
        const float* x) const {

    // 1. 计算粗量化器的残差
    std::vector<float> residual(n * d);
    for (size_t i = 0; i < n; i++) {
        idx_t key = coarse_quantizer->search(x + i * d);  // 找到最近的coarse centroid
        for (size_t j = 0; j < d; j++) {
            residual[i * d + j] = x[i * d + j] - coarse_quantizer->codebook[key * d + j];
        }
    }

    // 2. 使用PQ编码器编码残差
    pq->compute_codes(codes, n, residual.data());
}
```

**编码流程图：**

```
原始向量x (d=128)
    │
    ▼
┌──────────────────────┐
│ Coarse Quantization   │
│ (IVF coarse quantizer) │
│ key = search(x)       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Compute Residual      │
│ r = x - centroid[key] │
└──────────┬───────────┘
           │
           ▼
┌�──────────────────────┐
│ PQ Encoding (4-bit)  │
│ codes = pq.encode(r)  │
└──────────┬───────────┘
           │
           ▼
    打包存储到codes[]
```

### 2.3 代码打包优化

```cpp
// pq4_pack_codes_range在pq4_fast_scan.cpp中实现
void pq4_pack_codes_range(
        const uint8_t* codes,
        size_t M,
        size_t i0,
        size_t i1,
        size_t bbs,
        size_t M2,
        uint8_t* __restrict packed_codes) {

    // ==================== 块对齐处理 ====================
    size_t nblocks = (i1 - i0 + bbs - 1) / bbs;

    for (size_t b = 0; b < nblocks; b++) {
        size_t base = i0 + b * bbs;
        size_t nblock = std::min(bbs, i1 - base);

        // ==================== 向量化打包 ====================
        for (size_t m = 0; m < M; m++) {
            // 提取第m个子量化器的所有4-bit编码
            // 打包成连续的字节流
            for (size_t i = 0; i < nblock; i++) {
                uint8_t byte0 = codes[base + i] [2 * m];
                uint8_t byte1 = codes[base + i] [2 * m + 1];

                // 重新排列以优化SIMD访问
                packed_codes[...] = (byte0 & 0x0F) | (byte1 << 4);
            }
        }
    }
}
```

**打包优化详解：**

```
原始编码布局（未优化）:
向量0: [c0_0, c0_1, c0_2, c0_3, ..., c0_31]  (32个4-bit值)
向量1: [c1_0, c1_1, c1_2, c1_3, ..., c1_31]

打包后（SIMD友好）:
块0: [
  子量化0的32个4-bit值,
  子量化1的32个4-bit值,
  ...
  子量化31的32个4-bit值
]

访问优势：
1. 顺序访问，cache命中率高
2. SIMD向量化加载
3. 减少内存跳转
```

---

## 3. 搜索实现变体

### 3.1 实现变体概览

```cpp
// implem参数控制不同的实现策略
// implem = 0: 自动选择
// implem = 1: 未实现
// implem = 2,3,4: 浮点实现（带归一化）
// implem = 12,13: bbs=32的优化实现
// implem = 14,15: 通用bbs实现

template <bool is_max>
void IndexFastScan::search_dispatch_implem(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const FastScanDistancePostProcessing& context) const {

    // 自动选择实现
    int impl = implem;
    if (impl == 0) {
        if (bbs == 32) {
            impl = 12;  // bbs=32时使用implem=12
        } else {
            impl = 14;  // 其他bbs使用implem=14
        }
        if (k > 20) {
            impl++;  // k>20时使用更优化的版本
        }
    }

    // 路由到具体实现
    if (implem == 2 || implem == 3 || implem == 4) {
        search_implem_234<Cfloat>(n, x, k, distances, labels, context);
    } else if (implem >= 12 && implem <= 15) {
        // 并行搜索
        int nt = std::min(omp_get_max_threads(), int(n));

        if (nt < 2) {
            // 单线程
            if (impl == 12 || impl == 13) {
                search_implem_12<C>(n, x, k, distances, labels, impl, context);
            } else {
                search_implem_14<C>(n, x, k, distances, labels, impl, context);
            }
        } else {
            // 多线程（显式分片）
#pragma omp parallel for num_threads(nt)
            for (int slice = 0; slice < nt; slice++) {
                idx_t i0 = n * slice / nt;
                idx_t i1 = n * (slice + 1) / nt;

                // 创建线程特定上下文
                FastScanDistancePostProcessing thread_context = context;
                if (thread_context.query_factors != nullptr) {
                    thread_context.query_factors += i0;  // 调整指针
                }

                float* dis_i = distances + i0 * k;
                idx_t* lab_i = labels + i0 * k;

                // 递归调用
                if (impl == 12 || impl == 13) {
                    search_implem_12<C>(i1 - i0, x + i0 * d, k, dis_i, lab_i, impl, thread_context);
                } else {
                    search_implem_14<C>(i1 - i0, x + i0 * d, k, dis_i, lab_i, impl, thread_context);
                }
            }
        }
    }
}
```

**实现变体对比：**

| implem | bbs限制 | 特点 | 适用场景 |
|--------|---------|------|---------|
| 12 | 32 | 最大优化，固定块大小 | 32的倍数，大批量 |
| 13 | 32 | 同12，支持skip | 同12 + skip模式 |
| 14 | 任意 | 通用实现 | 任意bbs |
| 15 | 任意 | 同14，支持skip | 同14 + skip模式 |
| 234 | - | 浮点实现 | 调试、验证 |

### 3.2 search_implem_12实现（bbs=32优化版）

```cpp
template <class C>
void IndexFastScan::search_implem_12(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl,
        const FastScanDistancePostProcessing& context) const {

    using RH = ResultHandlerCompare<C, false>;
    FAISS_THROW_IF_NOT(bbs == 32);

    // ==================== 块大小递归处理 ====================
    int64_t qbs2 = this->qbs == 0 ? 11 : pq4_qbs_to_nq(this->qbs);

    if (n > qbs2) {
        // 递归处理大查询
        for (int64_t i0 = 0; i0 < n; i0 += qbs2) {
            int64_t i1 = std::min(i0 + qbs2, n);

            FastScanDistancePostProcessing sub_context = context;
            if (sub_context.query_factors != nullptr) {
                sub_context.query_factors += i0;
            }

            search_implem_12<C>(
                    i1 - i0,
                    x + d * i0,  // 调整指针
                    k,
                    distances + i0 * k,
                    labels + i0 * k,
                    impl,
                    sub_context);
        }
        return;
    }

    // ==================== 计算量化LUT ====================
    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> quantized_dis_tables(n * dim12);
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    if (!(skip & 1)) {
        compute_quantized_LUT(
                n, x, quantized_dis_tables.get(), normalizers.get(), context);
    }

    // ==================== 打包LUT ====================
    AlignedTable<uint8_t> LUT(n * dim12);

    int qbs = this->qbs;
    if (n != pq4_qbs_to_nq(qbs)) {
        qbs = pq4_preferred_qbs(n);  // 选择最优块大小
    }

    int LUT_nq = pq4_pack_LUT_qbs(
            qbs, M2, quantized_dis_tables.get(), LUT.get());
    FAISS_THROW_IF_NOT(LUT_nq == n);

    // ==================== 创建结果处理器 ====================
    std::unique_ptr<RH> handler(
            static_cast<RH*>(make_knn_handler(
                    C::is_max,
                    impl,
                    n,
                    k,
                    ntotal,
                    distances,
                    labels,
                    nullptr,
                    context));

    handler->disable = bool(skip & 2);
    handler->normalizers = normalizers.get();

    // ==================== 主搜索循环 ====================
    if (!(skip & 4)) {
        pq4_accumulate_loop_qbs(
                qbs,
                ntotal2,      // 向上取整的向量总数
                M2,
                codes.get(),
                LUT.get(),
                *handler.get(),
                context.norm_scaler);
    }

    if (!(skip & 8)) {
        handler->end();  // 完成处理（排序等）
    }
}
```

**search_implem_12流程图：**

```
输入：n个查询，k个近邻
    │
    ▼
┌─────────────────────────┐
│ 计算量化LUT              │
│ - 每个查询维度的距离表   │
│ - 量化为uint8_t          │
│ - 计算归一化因子          │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ 打包LUT                   │
│ - 重排数据以优化SIMD     │
│ - 对齐到32字节边界        │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ pq4_accumulate_loop_qbs  │
│ - SIMD批量扫描编码       │
│ - 查表累加距离           │
│ - 处理skip标志          │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ handler->end()           │
│ - 堆排序                  │
│ - top-k结果               │
└──────────┬──────────────┘
           │
           ▼
    输出：distances[], labels[]
```

### 3.3 search_implem_14实现（通用bbs版本）

```cpp
template <class C>
void IndexFastScan::search_implem_14(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl,
        const FastScanDistancePostProcessing& context) const {

    using RH = ResultHandlerCompare<C, false>;
    FAISS_THROW_IF_NOT(bbs % 32 == 0);

    // ==================== 递归处理 ====================
    int qbs2 = qbs == 0 ? 4 : qbs;

    if (n > qbs2) {
        for (int64_t i0 = 0; i0 < n; i0 += qbs2) {
            int64_t i1 = std::min(i0 + qbs2, n);

            FastScanDistancePostProcessing sub_context = context;
            if (sub_context.query_factors != nullptr) {
                sub_context.query_factors += i0;
            }

            search_implem_14<C>(
                    i1 - i0,
                    x + d * i0,
                    k,
                    distances + i0 * k,
                    labels + i0 * k,
                    impl,
                    sub_context);
        }
        return;
    }

    // ==================== 计算LUT ====================
    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> quantized_dis_tables(n * dim12);
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    if (!(skip & 1)) {
        compute_quantized_LUT(
                n, x, quantized_dis_tables.get(), normalizers.get(), context);
    }

    // ==================== 打包LUT（implem_14专用） ====================
    AlignedTable<uint8_t> LUT(n * dim12);
    pq4_pack_LUT(n, M2, quantized_dis_tables.get(), LUT.get());

    // ==================== 创建handler ====================
    std::unique_ptr<RH> handler(
            static_cast<RH*>(make_knn_handler(
                    C::is_max,
                    impl,
                    n,
                    k,
                    ntotal,
                    distances,
                    labels,
                    nullptr,
                    context));

    handler->disable = bool(skip & 2);
    handler->normalizers = normalizers.get();

    // ==================== 搜索循环（implem_14专用） ====================
    if (!(skip & 4)) {
        pq4_accumulate_loop_nq(
                n,
                ntotal2,
                M2,
                codes.get(),
                LUT.get(),
                *handler.get(),
                context.norm_scaler);
    }

    if (!(skip & 8)) {
        handler->end();
    }
}
```

**implem_12 vs implem_14对比：**

| 特性 | implem_12 | implem_14 |
|------|-----------|-----------|
| bbs支持 | 固定32 | 任意32的倍数 |
| LUT打包 | pq4_pack_LUT_qbs | pq4_pack_LUT |
| 搜索循环 | pq4_accumulate_loop_qbs | pq4_accumulate_loop_nq |
| 优化程度 | 极致优化 | 通用优化 |
| 适用场景 | bbs=32固定 | 灵活配置 |

---

## 4. 量化查找表（LUT）计算

### 4.1 compute_quantized_LUT实现

```cpp
void IndexFastScan::compute_quantized_LUT(
        idx_t n,
        const float* x,
        uint8_t* lut,
        float* normalizers,
        const FastScanDistancePostProcessing& context) const {

    size_t dim12 = ksub * M;  // 16 * M
    std::unique_ptr<float[]> dis_tables(new float[n * dim12]);

    // ==================== 计算浮点距离表 ====================
    compute_float_LUT(dis_tables.get(), n, x, context);

    // ==================== 量化为uint8 ====================
    for (uint64_t i = 0; i < n; i++) {
        round_uint8_per_column(
                dis_tables.get() + i * dim12,  // 每个查询的M×16距离表
                M,
                ksub,
                &normalizers[2 * i],      // 预留缩放因子空间
                &normalizers[2 * i + 1]);  // 预留偏移空间
    }

    // ==================== 转换为uint8 ====================
    for (uint64_t i = 0; i < n; i++) {
        const float* t_in = dis_tables.get() + i * dim12;
        uint8_t* t_out = lut + i * M2 * ksub;

        for (int j = 0; j < dim12; j++) {
            t_out[j] = int(t_in[j]);  // 截断为uint8
        }

        // 清零剩余部分（M2 - M）
        memset(t_out + dim12, 0, (M2 - M) * ksub);
    }
}
```

**LUT数据布局：**

```
假设M=32, ksub=16, n=4:

dim12 = 16 × 32 = 512

dis_tables布局 (n × dim12):
查询0: [512个浮点距离]
       [d(0,0), d(0,1), ..., d(0,15),   // 子量化器0的16个距离
        d(1,0), d(1,1), ..., d(1,15),   // 子量化器1的16个距离
        ...
        d(31,0), ..., d(31,15)]          // 子量化器31的16个距离
查询1: [512个浮点距离]
...

lut布局 (n × M2 × ksub):
查询0: [512个uint8值]
       [量化后距离...]

normalizers布局 (n × 2):
[alpha_0, beta_0]   // 查询0的缩放因子和偏移
 [alpha_1, beta_1]   // 查询1的缩放因子和偏移
 ...
```

### 4.2 round_uint8_per_column实现

```cpp
void round_uint8_per_column(
        const float* dis_table,
        size_t M,
        size_t ksub,
        float* alpha,
        float* beta) {

    // 每一列（子量化器）独立处理
    for (size_t m = 0; m < M; m++) {
        const float* __restrict distances = dis_table + m * ksub;

        // 找出最小和最大距离
        float vmin = HUGE_VALF;
        float vmax = -HUGE_VALF;

        for (size_t i = 0; i < ksub; i++) {
            if (distances[i] < vmin) {
                vmin = distances[i];
            }
            if (distances[i] > vmax) {
                vmax = distances[i];
            }
        }

        // 计算量化参数
        float range = vmax - vmin;

        if (range > 0) {
            *alpha = 255.0f / range;  // 缩放因子
            *beta = vmin;               // 偏移
        } else {
            *alpha = 0.0f;
            *beta = vmin;
        }

        // 量化
        float* __restrict q = const_cast<float*>(dis_table) + m * ksub;
        for (size_t i = 0; i < ksub; i++) {
            q[i] = (q[i] - *beta) * *alpha;
        }
    }
}
```

**量化公式：**

```
原始距离表：d[q][m][k] (查询q × 子量化器m × 质心k)

对每个子量化器m独立量化：
  vmin = min_k d[q][m][k]
  vmax = max_k d[q][m][k]

  alpha = 255 / (vmax - vmin)
  beta  = vmin

  d_quantized[q][m][k] = (d[q][m][k] - beta) × alpha

反量化时恢复：
  d_restored = d_quantized / alpha + beta
```

---

## 5. SIMD优化的搜索循环

### 5.1 pq4_accumulate_loop_qbs实现

```cpp
void pq4_accumulate_loop_qbs(
        int nq,                 // 查询数量
        size_t n_total2,        // 向量总数（向上取整）
        size_t M2,              // 向上取整的子量化器数
        const uint8_t* codes,   // 打包的PQ编码
        const uint8_t* lut,      // 查找表
        ResultHandler<C>& handler,  // 结果处理器
        const NormTableScaler* nt_scaler) {

    // ==================== 外层循环：遍历块 ====================
    for (size_t b = 0; b < (n_total2 + 31) / 32; b++) {
        // ==================== 加载当前块的编码 ====================
        const uint8_t* code_p = codes + b * 32 * M2 / 2;

        // ==================== SIMD优化距离计算 ====================
        #ifdef __AVX512F__
        // AVX512版本：每次处理16个查询
        for (size_t q = 0; q < nq; q += 16) {
            const uint8_t* lut_p = lut + q * M2 * 16;

            __m512i accu = _mm512_setzero_si512();

            // 批量处理32个向量
            for (size_t i = 0; i < 32; i++) {
                // 加载编码
                __m512i code = _mm512_loadu_si512(code_p + i * (M2 / 2));

                // 查表累加距离
                for (size_t m = 0; m < M2; m++) {
                    __m512i lut_entry = _mm512_loadu_si512(lut_p + m * 16);

                    // 提取4-bit索引并查表
                    __m512i indices = _mm512_and_si512(
                            _mm512_srli_epi16(code, m * 4, 0),
                            _mm512_set1_epi8(0x0F));

                    __m512i distances = _mm512_shuffle_epi8(
                            lut_entry, indices, _MM_SHUFFLE(EPI8(0,0,0,0)));

                    accu = _mm512_add_epi8(accu, distances);
                }

                // 处理结果
                uint16_t distances[32];
                _mm512_storeu_si512(distances, accu);

                // 提交到结果处理器
                for (size_t i = 0; i < 32; i++) {
                    handler.add_result(distances[i], b * 32 + i);
                }
            }
        }

        #elif defined(__AVX2__)
        // AVX2版本：每次处理8个查询
        // ... 类似的SIMD代码 ...

        #endif
    }
}
```

**SIMD优化技巧：**

1. **4-bit索引提取**：
   ```cpp
   // 从打包的字节中提取4-bit索引
   __m512i code = _mm512_loadu_si512(code_p + i * (M2 / 2));

   // 掩码提取（每字节2个4-bit值）
   __m512i indices = _mm512_and_si512(
           _mm512_srli_epi16(code, m * 4, 0),  // 右移4*m位
           _mm512_set1_epi8(0x0F));              // 掩码低4位

   // 使用索引查表
   __m512i distances = _mm512_shuffle_epi8(lut_entry, indices, ...);
   ```

2. **查表累加**：
   ```cpp
   // 使用shuffle_epi8实现查表
   // indices: [idx0, idx1, idx2, ..., idx15]
   // lut_entry: [val0, val1, val2, ..., val15]
   // result: [lut_entry[idx0], lut_entry[idx1], ...]

   __m512i distances = _mm512_shuffle_epi8(
           lut_entry, indices, _MM_SHUFFLE(EPI8(0,0,0,0)));

   // 累加距离
   accu = _mm512_add_epi8(accu, distances);
   ```

3. **批量处理**：
   ```
   每次处理：
   - 32个向量（固定块大小）
   - 16个查询（AVX512）
   - 32个子量化器

   总距离计算：32 × 16 = 512次查表/迭代
   ```

---

## 6. 结果处理器优化

### 6.1 结果处理器选择

```cpp
SIMDResultHandlerToFloat* IndexFastScan::make_knn_handler(
        bool is_max,
        int impl,
        idx_t n,
        idx_t k,
        size_t ntotal,
        float* distances,
        idx_t* labels,
        const IDSelector* sel,
        const FastScanDistancePostProcessing&) const {

    if (is_max) {
        // 最大堆（用于内积搜索）
        using HeapHC = HeapHandler<CMax<uint16_t, int>, false>;
        using ReservoirHC = ReservoirHandler<CMax<uint16_t, int>, false>;
        using SingleResultHC = SingleResultHandler<CMax<uint16_t, int>, false>;

        if (k == 1) {
            return new SingleResultHC(n, ntotal, distances, labels, sel);
        } else if (impl % 2 == 0) {
            return new HeapHC(n, ntotal, k, distances, labels, sel);
        } else {
            // impl % 2 == 1: 使用Reservoir（更大容量）
            return new ReservoirHC(n, ntotal, k, 2 * k, distances, labels, sel);
        }
    } else {
        // 最小堆（用于L2距离搜索）
        using HeapHC = HeapHandler<CMin<uint16_t, int>, false>;
        using ReservoirHC = ReservoirHandler<CMin<uint16_t, int>, false>;
        using SingleResultHC = SingleResultHandler<CMin<uint16_t, int>, false>;

        if (k == 1) {
            return new SingleResultHC(n, ntotal, distances, labels, sel);
        } else if (impl % 2 == 0) {
            return new HeapHC(n, ntotal, k, distances, labels, sel);
        } else {
            return new ReservoirHC(n, ntotal, k, 2 * k, distances, labels, sel);
        }
    }
}
```

**处理器选择策略：**

```
k=1:
  → SingleResultHandler
  - 单结果优化
  - 无堆维护开销

k<=20, impl%2==0:
  → HeapHandler (capacity=k)
  - 标准堆实现
  - 内存占用小

k>20 or impl%2==1:
  → ReservoirHandler (capacity=2k)
  - 更大容量
  - 更好的统计特性
```

### 6.2 skip标志优化

```cpp
// skip标志控制搜索的不同阶段
// skip & 1: 跳过LUT计算（使用预计算的）
// skip & 2: 禁用handler（不收集结果）
// skip & 4: 跳过搜索循环
// skip & 8: 跳过end()（不排序）

if (skip & 1) {
    quantized_dis_tables.clear();  // 不计算LUT
} else {
    compute_quantized_LUT(n, x, quantized_dis_tables.get(), normalizers.get(), context);
}

handler->disable = bool(skip & 2);  // 禁用结果收集

if (skip & 4) {
    // 跳过搜索
} else {
    pq4_accumulate_loop_qbs(...);
}

if (!(skip & 8)) {
    handler->end();  // 完成并排序
}
```

**skip的应用场景：**

1. **性能测试**：
   ```cpp
   // 只测量搜索速度，不关心结果
   search(..., skip = 0b00001010);  // 跳过LUT和end()
   ```

2. **增量更新**：
   ```cpp
   // 重用之前的LUT
   search(..., skip = 0b00000001);  // 跳过LUT计算
   ```

3. **调试模式**：
   ```cpp
   // 只收集LUT，不搜索
   search(..., skip = 0b00010110);  // 只计算LUT
   ```

---

## 7. 性能优化总结

### 7.1 内存访问优化

```cpp
// 1. 块对齐
bbs = 32  // 32字节对齐（AVX2缓存行大小）

// 2. 顺序访问
for (size_t b = 0; b < nblocks; b++) {
    // 顺序处理每个块
    // 提高cache命中率
}

// 3. 预取
// 在pq4_accumulate_loop_qbs中隐式预取
// 编译器自动优化
```

### 7.2 计算优化

```cpp
// 1. 量化距离计算
// 用uint8代替float32：
// - 减少内存带宽需求（4x压缩）
// - 提高SIMD效率（处理2x元素）

// 2. 批量距离计算
distances_batch_4(idx0, idx1, idx2, idx3, ...);
// SIMD并行计算4个距离

// 3. 查表累加
shuffle_epi8 + add_epi8
// 单指令完成查表和累加
```

### 7.3 并行化策略

```cpp
// 1. 查询级并行
#pragma omp parallel for
for (int slice = 0; slice < nt; slice++) {
    // 每个线程处理一批查询
    search_implem_12(..., slice);
}

// 2. 块级递归
if (n > qbs2) {
    for (int64_t i0 = 0; i0 < n; i0 += qbs2) {
        // 递归处理每个块
        search_implem_12(i1 - i0, ...);
    }
}

// 3. 无锁设计
// 每个线程独立处理不同数据
// 避免同步开销
```

---

## 8. 性能调优建议

### 8.1 参数选择

```cpp
// 1. M（子量化器数量）
M = 16:   // 平衡精度和速度
M = 32:   // 更高精度，稍慢
M = 64:   // 最高精度，更慢

// 2. bbs（块大小）
bbs = 32:   // 默认，适合大多数场景
bbs = 64:   // 大数据集优化
bbs = 128:  // 超大数据集

// 3. qbs（查询块大小）
qbs = 0:    // 自动选择
qbs = 11:   // 2048查询/块
qbs = 12:   // 4096查询/块
```

### 8.2 实现选择

```cpp
// 自动选择
implem = 0;

// 手动选择
implem = 12:  // bbs=32，最高性能
implem = 14:  // 通用bbs，灵活性
implem = 234: // 浮点实现，调试
```

### 8.3 性能分析

```cpp
// 收集统计信息
FastScanStats stats;
stats.n = ...;
stats.ncode = ...;
stats.nscan = ...;

// 分析瓶颈
// - ncode: 编码数量
// - nscan: 实际扫描数量
// - 比率 = nscan / ncode (扫描率)
```

---

## 9. 与其他索引对比

### 9.1 FastScan vs IVFPQ

| 特性 | IndexFastScan | IndexIVFPQ |
|------|---------------|------------|
| 精度 | 略高 | 稍低 |
| 速度 | 很快 | 快 |
| 内存占用 | 低 | 低 |
| 构建时间 | 中等 | 短 |
| 支持度量 | L2, IP | L2, IP |
| 灵活性 | 中 | 高 |

### 9.2 FastScan vs HNSW

| 特性 | IndexFastScan | IndexHNSW |
|------|---------------|-----------|
| 精度 | 高 | 极高 |
| 速度 | 极快 | 快 |
| 内存占用 | 低 | 高 |
| 构建时间 | 短 | 长 |
| 动态添加 | 慢 | 快 |
| 适用场景 | 静态数据集 | 动态数据集 |

---

## 总结

IndexFastScan.cpp展示了多个极致的优化技术：

1. **4-bit PQ压缩**：4倍内存节省
2. **SIMD查表累加**：单指令完成查表和累加
3. **块对齐处理**：优化cache访问
4. **多实现变体**：针对不同场景优化
5. **并行搜索**：查询级和块级并行
6. **量化LUT**：进一步减少计算
7. **skip模式**：灵活的功能裁剪
8. **结果处理器**：Heap vs Reservoir策略

这些优化使得FastScan成为Faiss中最快的索引之一，特别适合大规模静态数据集的近似最近邻搜索。

---

## 参考资料

- Faiss FastScan文档: https://github.com/facebookresearch/faiss/wiki/Fast-Scan
- Product Quantization论文: "Product quantization for nearest neighbor search"
- SIMD优化技术: AVX2/AVX512 Intrinsics Guide
- Faiss源码: https://github.com/facebookresearch/faiss
