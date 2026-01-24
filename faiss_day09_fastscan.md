# Faiss深度课程 - 第9天：FastScan架构 - SIMD优化深度解析

## 课程目标

深入理解FastScan架构，这是Faiss中最极致的SIMD优化实现，专门针对PQ和量化索引进行了底层优化。

---

## 1. FastScan概述

### 1.1 核心思想

FastScan通过将PQ码重新组织为SIMD友好的布局，实现极致的搜索性能：

```cpp
// 传统PQ存储（按向量）
vector0: [m0, m1, m2, ..., mM]
vector1: [m0, m1, m2, ..., mM]
vector2: [m0, m1, m2, ..., mM]
...

// FastScan存储（按子量化器）
batch0: [v0_m0, v1_m0, v2_m0, ..., vN_m0]
batch1: [v0_m1, v1_m1, v2_m1, ..., vN_m1]
...
batchM: [v0_mM, v1_mM, v2_mM, ..., vN_mM]
```

### 1.2 性能提升

| 特性 | 常规PQ | FastScan |
|------|--------|----------|
| SIMD利用率 | ~30% | ~90% |
| 缓存命中率 | 中 | 高 |
| 搜索速度 | 1x | 4-8x |

---

## 2. IndexFastScan结构（底层实现）

### 2.1 IndexFastScan完整类定义

```cpp
// faiss/IndexFastScan.h (完整版本)
struct IndexFastScan : Index {
    // 实现选择
    int implem = 0;        // 实现版本号（12-15）
    int skip = 0;          // 跳过某些部分（用于计时）

    // 块处理参数
    int bbs;              // 每批处理的向量数（构建时设置）
    int qbs = 0;           // 查询块大小（0=使用默认）

    // 向量量化器参数
    size_t M;              // 子量化器数量
    size_t nbits;          // 每个子量化器的位数
    size_t ksub;           // 每个子量化器的质心数 = 2^nbits
    size_t code_size;      // 每个向量的编码字节

    // 打包后的编码存储
    size_t ntotal2;         // 向量总数（可能填充）
    size_t M2;             // M的对齐版本

    AlignedTable<uint8_t> codes;  // 打包的编码表

    // 测试用（当从IndexPQ/IndexAQ初始化时设置）
    const uint8_t* orig_codes = nullptr;

    // 初始化FastScan索引
    void init_fastscan(
            int d,
            size_t M,
            size_t nbits,
            MetricType metric,
            int bbs);  // 块大小

    IndexFastScan();
    void reset() override;

    // 核心接口
    void add(idx_t n, const float* x) override;
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    // 子类必须实现
    virtual void compute_codes(
            uint8_t* codes,
            idx_t n,
            const float* x) const = 0;

    // 计算浮点数查找表
    virtual void compute_float_LUT(
            float* lut,
            idx_t n,
            const float* x,
            const FastScanDistancePostProcessing& context) const = 0;

    // 创建KNN处理器
    virtual SIMDResultHandlerToFloat* make_knn_handler(
            bool is_max,
            int impl,
            idx_t n,
            idx_t k,
            size_t ntotal,
            float* distances,
            idx_t* labels,
            const IDSelector* sel,
            const FastScanDistancePostProcessing& context) const;

    // 重建向量
    void reconstruct(idx_t key, float* recons) const override;
    size_t remove_ids(const IDSelector& sel) override;
    void merge_from(Index& otherIndex, idx_t add_id = 0) override;

    // 标准接口
    size_t sa_code_size() const override {
        return code_size;
    }

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override {
        compute_codes(bytes, n, x);
    }

protected:
    // 计算量化查找表
    void compute_quantized_LUT(
            idx_t n,
            const float* x,
            uint8_t* lut,
            float* normalizers,
            const FastScanDistancePostProcessing& context) const;

    // 搜索实现模板
    template <class Cfloat>
    void search_implem_12(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl,
            const FastScanDistancePostProcessing& context) const;

    template <class C>
    void search_implem_14(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl,
            const FastScanDistancePostProcessing& context) const;

    template <class C>
    void search_dispatch_implem(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const FastScanDistancePostProcessing& context) const;
};
```

### 2.2 实现版本说明

```cpp
// IndexFastScan实现版本说明

// implem = 12: 带内部qbs循环的blocked版本
//   - 使用SIMD优化的批量处理
//   - 支持查询批处理（qbs）
//   - 适用于：大批量查询

// implem = 13: 带 reservoir累加器的版本12
//   - 使用reservoir代替heap
//   - 适用于：大k值（减少堆操作开销）

// implem = 14: 无qbs的heap版本
//   - 每个查询独立处理
//   - 适用于：小批量查询

// implem = 15: 无qbs的reservoir版本
//   - 最简洁的实现
//   - 适用于：小查询，大k值

// 选择策略：
// - 批量大：implem=12/13
// - 批量小：implem=14/15
// - k值大：prefer reservoir（13/15）
// - k值小：prefer heap（12/14）
```

### 2.1 传统的PQ布局

```cpp
// 传统PQ编码布局
struct TraditionalPQLayout {
    uint8_t* codes;  // nb × M
    size_t nb;       // 向量数
    size_t M;        // 子量化器数

    // 访问向量i的编码
    uint8_t* get_vector(size_t i) {
        return codes + i * M;
    }

    // 访问向量i的第m个编码
    uint8_t get_code(size_t i, size_t m) {
        return codes[i * M + m];
    }
};
```

### 2.2 FastScan布局

```cpp
// FastScan布局：按batches重新组织
struct FastScanLayout {
    // M个子量化器分成batches
    // 每个batch包含多个子量化器的编码

    size_t M;            // 子量化器数
    size_t nbits;        // 每个编码的位数
    size_t batch_size;   // 每批处理的向量数

    // 重新组织后的编码
    // layout: (nbatch, M_per_batch, batch_size)
    //        其中 nbatch = ceil(nb / batch_size)
    //              M_per_batch = bits per vector (对齐)
    std::vector<uint8_t> codes;

    // 示例：M=16, nbits=8, batch_size=32
    // codes[0:32]    = 32个向量的第0个编码
    // codes[32:64]   = 32个向量的第1个编码
    // ...
    // codes[496:512] = 32个向量的第15个编码
    // codes[512:544] = 下一个batch的32个向量的第0个编码
};
```

### 2.3 布局转换

```cpp
// 从传统布局转换为FastScan布局
void convert_to_fastscan(
        const uint8_t* traditional_codes,  // nb × M
        size_t nb, size_t M,
        uint8_t* fastscan_codes,
        size_t batch_size) {

    size_t nbatch = (nb + batch_size - 1) / batch_size;

    for (size_t batch = 0; batch < nbatch; batch++) {
        size_t start = batch * batch_size;
        size_t end = std::min(start + batch_size, nb);
        size_t n_in_batch = end - start;

        for (size_t i = 0; i < n_in_batch; i++) {
            for (size_t m = 0; m < M; m++) {
                size_t src_idx = (start + i) * M + m;
                size_t dst_idx = batch * M * batch_size + m * batch_size + i;

                fastscan_codes[dst_idx] = traditional_codes[src_idx];
            }
        }
    }
}
```

---

## 3. FastScan距离计算

### 3.1 SIMD优化的查表

```cpp
// 使用SIMD优化的查表计算距离
void fast_scan_distance(
        const float* dis_table,  // M × ksub
        const uint8_t* codes,     // batch_size × M
        size_t batch_size, size_t M,
        float* distances) {       // batch_size

    // dis_table[m * ksub + k] = 查询到质心k在子量化器m的距离

    for (size_t m = 0; m < M; m++) {
        const float* dt = dis_table + m * 256;  // 假设nbits=8

        // 加载batch_size个编码
        const uint8_t* code_m = codes + m * batch_size;

        // SIMD优化：一次处理多个编码
        for (size_t i = 0; i < batch_size; i += 16) {  // AVX2
            // 加载16个编码
            __m256i codes = _mm256_loadu_si256(
                (__m256i*)(code_m + i));

            // 查找对应距离（使用shuffle作为查找表）
            __m256 dis = lookup256(codes, dt);

            // 累加到结果
            __m256* result = (__m256*)(distances + i);
            *result = _mm256_add_ps(*result, dis);
        }
    }
}
```

### 3.2 AVX2查找表

```cpp
// 使用AVX2的shuffle作为查找表
__m256 lookup256_avx2(__m256i codes, const float* table) {
    // codes: 16个uint8编码
    // table: 256个float距离值

    // 由于AVX2没有直接的查找表指令，
    // 我们使用shuffle来模拟

    // 将编码分为高4位和低4位
    __m256i zero = _mm256_setzero_si256();
    __m256i low = _mm256_and_si256(codes, _mm256_set1_epi8(0x0F));
    __m256i high = _mm256_and_si256(
        _mm256_srli_epi16(codes, 4), _mm256_set1_epi8(0x0F));

    // 从table查找（需要展开为16个shuffle）
    // 这里简化为标量代码
    alignas(32) float result[16];
    uint8_t code[16];
    _mm256_storeu_si256((__m256i*)code, codes);

    for (int i = 0; i < 16; i++) {
        result[i] = table[code[i]];
    }

    return _mm256_load_ps(result);
}
```

### 3.3 AVX-512查找表

```cpp
// AVX-512有专门的查找表指令
__m512 lookup512_avx512(__m512i codes, const float* table) {
    // codes: 64个uint8编码
    // table: 256个float距离值

    // 使用vpexpandd指令扩展
    // 或者使用__m512i_gather

    __m512 result;
    for (int i = 0; i < 64; i++) {
        uint8_t c = _mm512_extract_epi8(codes, i);
        result[i] = table[c];
    }

    return result;
}

// 更高效的版本：使用gather
__m512 lookup512_gather(__m512i indices, const float* base) {
    // indices: 64个索引（0-255）
    // base: 表基址

    // 使用vgatherdps
    return _mm512_i32gather_ps(base, indices, 4);
}
```

---

## 4. IndexIVFFastScan

### 4.1 结构

```cpp
// faiss/IndexIVFFastScan.h
struct IndexIVFFastScan : IndexIVF {
    size_t M;             // 子量化器数
    size_t nbits;         // 每个子量化器的位数
    bool is_cosine;       // 是否使用余弦距离

    // FastScan特定的布局
    size_t bbs;           // bits per block
    size_t M2;            // M per block

    IndexIVFFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits)
        : IndexIVF(quantizer, d, nlist,
                    (M * nbits + 7) / 8,  // code_size
                    METRIC_L2) {}
};
```

### 4.2 编码

```cpp
void IndexIVFFastScan::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {

    // 1. 计算残差
    float* residuals = new float[n * d];
    compute_residuals(n, x, list_nos, residuals);

    // 2. PQ编码
    uint8_t* pq_codes = new uint8_t[n * M];
    pq.compute_codes(residuals, pq_codes, n);

    // 3. 转换为FastScan布局
    // 这里需要将每个倒排列表的编码重新组织
    for (idx_t i = 0; i < n; i++) {
        idx_t list_no = list_nos[i];
        size_t offset = invlists->list_size(list_no);

        // 将编码添加到列表
        uint8_t* batch_codes = codes + i * code_size;
        convert_to_fastscan_layout(
            pq_codes + i * M, 1, M,
            batch_codes,
            batch_size);
    }

    delete[] residuals;
    delete[] pq_codes;
}
```

### 4.3 搜索实现

```cpp
void IndexIVFFastScan::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    // 1. 粗量化
    idx_t* assign = new idx_t[n * nprobe];
    float* centroid_dis = new float[n * nprobe];
    quantizer->search(n, x, nprobe, centroid_dis, assign);

    // 2. 为每个查询计算距离表
    size_t M2 = 2;  // 每个block的M数
    float* dis_tables = new float[n * M2 * 256];

    for (size_t m = 0; m < M; m += M2) {
        for (idx_t i = 0; i < n; i++) {
            // 计算查询到质心的距离
            compute_partial_distance_table(
                x + i * d, m, M2,
                dis_tables + (i * M + m) * 256);
        }
    }

    // 3. 扫描倒排列表
    for (size_t q = 0; q < n; q++) {
        float* simi = distances + q * k;
        idx_t* idxi = labels + q * k;
        heap_heapify<CMax<float, idx_t>>(k, simi, idxi);

        for (size_t ij = 0; ij < nprobe; ij++) {
            idx_t list_no = assign[q * nprobe + ij];

            // 获取列表的FastScan编码
            size_t list_size = invlists->list_size(list_no);
            const uint8_t* batch_codes =
                invlists->get_codes(list_no);

            // 扫描整个batch
            size_t nbatch = (list_size + batch_size - 1) / batch_size;

            for (size_t batch = 0; batch < nbatch; batch++) {
                size_t start = batch * batch_size;
                size_t end = std::min(start + batch_size, list_size);
                size_t n_in_batch = end - start;

                // SIMD优化的扫描
                const uint8_t* codes = batch_codes + batch * M * batch_size;
                float* batch_dists = new float[n_in_batch];

                fast_scan_distance(
                    dis_tables + q * M * 256,
                    codes, n_in_batch, M,
                    batch_dists);

                // 更新堆
                const idx_t* ids = invlists->get_ids(list_no);
                for (size_t i = 0; i < n_in_batch; i++) {
                    if (CMax<float, idx_t>::cmp(batch_dists[i], simi[0])) {
                        heap_replace_top<CMax<float, idx_t>>(
                            k, simi, idxi,
                            batch_dists[i],
                            ids[start + i]);
                    }
                }

                delete[] batch_dists;
            }
        }

        heap_reorder<CMax<float, idx_t>>(k, simi, idxi);
    }

    delete[] assign;
    delete[] centroid_dis;
    delete[] dis_tables;
}
```

---

## 5. FastScan变种

### 5.1 IndexIVFPQFastScan

```cpp
struct IndexIVFPQFastScan : IndexIVFFastScan {
    ProductQuantizer pq;

    IndexIVFPQFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits)
        : IndexIVFFastScan(quantizer, d, nlist, M, nbits) {}
};
```

### 5.2 IndexIVFFastScanIO

```cpp
// 使用更优化的内存布局
struct IndexIVFFastScanIO {
    // 交错编码：进一步减少缓存未命中
    // layout: (nbatch, M_per_batch, batch_size, interleave)

    void encode_with_interleaving(
            const uint8_t* fastscan_codes,
            size_t batch_size, size_t M,
            uint8_t* interleaved_codes,
            size_t interleave) {
        // 将编码按interleave模式重新组织
        // 例如：interleave=4表示每4个向量为一组
    }
};
```

---

## 6. 性能优化技巧

### 6.1 批量大小选择

```cpp
// 最优batch_size取决于CPU缓存
size_t optimal_batch_size() {
    // L1缓存：32KB
    // L2缓存：256KB
    // L3缓存：8-32MB

    // 对于PQ编码：
    // 每个编码：1字节（nbits=8）
    // M个编码：M字节
    // 距离表：256×M×4字节 = 1KB×M

    // 最佳batch_size使工作集适合L2缓存
    size_t batch_size = 32;  // 默认
    return batch_size;
}
```

### 6.2 预取优化

```cpp
void fast_scan_with_prefetch(
        const float* dis_table,
        const uint8_t* codes,
        size_t batch_size, size_t M,
        float* distances) {

    const size_t prefetch_distance = 4;

    for (size_t i = 0; i < batch_size; i++) {
        // 预取未来的编码
        if (i + prefetch_distance < batch_size) {
            _mm_prefetch(
                codes + (i + prefetch_distance) * M,
                _MM_HINT_T0);
        }

        // 计算距离
        float dis = 0;
        for (size_t m = 0; m < M; m++) {
            uint8_t code = codes[i * M + m];
            dis += dis_table[m * 256 + code];
        }

        distances[i] = dis;
    }
}
```

---

## 10. FastScan源码深度实现

### 10.1 IndexFastScan::init_fastscan - 初始化

```cpp
// faiss/IndexFastScan.cpp
// 初始化FastScan索引，设置编码布局
void IndexFastScan::init_fastscan(
        int d,
        size_t M_init,
        size_t nbits_init,
        MetricType metric,
        int bbs) {

    // 目前FastScan只支持4-bit PQ
    FAISS_THROW_IF_NOT(nbits_init == 4);
    // bbs必须是32的倍数（SIMD友好）
    FAISS_THROW_IF_NOT(bbs % 32 == 0);

    this->d = d;
    this->M = M_init;            // 子量化器数量
    this->nbits = nbits_init;    // 每个子量化器位数
    this->metric_type = metric;
    this->bbs = bbs;             // 块大小（32的倍数）

    ksub = (1 << nbits_init);    // 每个子量化器质心数 = 16

    // 计算编码大小（每个向量）
    code_size = (M_init * nbits_init + 7) / 8;  // (M*4+7)/8字节

    ntotal = ntotal2 = 0;
    // M的对齐版本（必须对齐到2）
    M2 = roundup(M_init, 2);
    is_trained = false;
}

// 辅助函数：向上取整
inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}
```

### 10.2 IndexFastScan::add - 添加向量

```cpp
// faiss/IndexFastScan.cpp
// 添加向量到FastScan索引（自动打包编码）
void IndexFastScan::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);

    // 分块处理避免过大分配
    constexpr idx_t bs = 65536;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            add(i1 - i0, x + i0 * d);
        }
        return;
    }

    // 1. 计算PQ编码
    AlignedTable<uint8_t> tmp_codes(n * code_size);
    compute_codes(tmp_codes.get(), n, x);

    // 2. 调整存储大小（填充到bbs倍数）
    ntotal2 = roundup(ntotal + n, bbs);
    size_t new_size = ntotal2 * M2 / 2;  // nbits=4时的存储大小
    size_t old_size = codes.size();

    if (new_size > old_size) {
        codes.resize(new_size);
        // 清零新分配的区域
        memset(codes.get() + old_size, 0, new_size - old_size);
    }

    // 3. 打包编码到SIMD友好的布局
    pq4_pack_codes_range(
            tmp_codes.get(),      // 输入：原始PQ编码
            M,                    // 子量化器数
            ntotal,               // 起始偏移
            ntotal + n,           // 结束偏移
            bbs,                  // 块大小
            M2,                   // 对齐的M
            codes.get());         // 输出：打包后的编码

    ntotal += n;
}
```

### 10.3 pq4_pack_codes_range - 编码打包

```cpp
// faiss/impl/pq4_fast_scan.cpp
// 将PQ编码打包为SIMD友好的布局
void pq4_pack_codes_range(
        const uint8_t* codes,    // 输入：nb × M字节
        int M,
        size_t i0,               // 起始索引
        size_t i1,               // 结束索引
        int bbs,                 // 块大小（32的倍数）
        int M2,                  // 对齐的M
        uint8_t* packed_codes) { // 输出：打包后的编码

    // 打包布局说明：
    // 原始：[vec0_M0, vec0_M1, ..., vec0_MM-1, vec1_M0, ...]
    // 打包后：对于每个bbs大小的块
    //         [block0_vec0_M0, block0_vec1_M0, ..., block0_vec31_M0,
    //          block0_vec0_M1, block0_vec1_M1, ..., block0_vec31_M1,
    //          ...,
    //          block0_vec0_MM-1, ..., block0_vec31_MM-1]

    for (size_t i = i0; i < i1; i += bbs) {
        size_t i_end = std::min(i + bbs, i1);
        size_t n_in_block = i_end - i;

        for (int m = 0; m < M; m++) {
            // 收集bbs个向量的第m个编码
            for (size_t j = 0; j < n_in_block; j++) {
                packed_codes[j] = codes[(i + j) * M + m];
            }

            // 填充剩余位置
            for (size_t j = n_in_block; j < (size_t)bbs; j++) {
                packed_codes[j] = 0;
            }

            packed_codes += bbs;
        }

        // M2-M的填充空间
        memset(packed_codes, 0, (M2 - M) * bbs);
        packed_codes += (M2 - M) * bbs;
    }
}
```

### 10.4 compute_quantized_LUT - 量化查找表

```cpp
// faiss/IndexFastScan.cpp
// 计算量化的查找表（用于快速扫描）
void IndexFastScan::compute_quantized_LUT(
        idx_t n,
        const float* x,
        uint8_t* lut,              // 输出：量化后的LUT
        float* normalizers,
        const FastScanDistancePostProcessing& context) const {

    // 1. 计算浮点数查找表
    size_t dim12 = ksub * M;       // 16 × M
    std::unique_ptr<float[]> dis_tables(new float[n * dim12]);
    compute_float_LUT(dis_tables.get(), n, x, context);

    // 2. 每列（子量化器）独立量化
    for (uint64_t i = 0; i < n; i++) {
        round_uint8_per_column(
                dis_tables.get() + i * dim12,
                M,
                ksub,
                &normalizers[2 * i],      // 最小值
                &normalizers[2 * i + 1]); // 缩放因子
    }

    // 3. 转换为uint8类型
    for (uint64_t i = 0; i < n; i++) {
        const float* t_in = dis_tables.get() + i * dim12;
        uint8_t* t_out = lut + i * M2 * ksub;

        for (int j = 0; j < dim12; j++) {
            t_out[j] = (uint8_t)t_in[j];
        }
        // 填充
        memset(t_out + dim12, 0, (M2 - M) * ksub);
    }
}
```

### 10.5 search_implem_12 - 核心搜索实现

```cpp
// faiss/IndexFastScan.cpp
// 实现12：带内部qbs循环的blocked版本（最常用）
template <class C>
void IndexFastScan::search_implem_12(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl,
        const FastScanDistancePostProcessing& context) const {

    FAISS_THROW_IF_NOT(bbs == 32);

    // 1. 处理qbs2分块（递归处理大查询批次）
    int64_t qbs2 = this->qbs == 0 ? 11 : pq4_qbs_to_nq(this->qbs);
    if (n > qbs2) {
        for (int64_t i0 = 0; i0 < n; i0 += qbs2) {
            int64_t i1 = std::min(i0 + qbs2, n);
            FastScanDistancePostProcessing sub_context = context;
            if (sub_context.query_factors != nullptr) {
                sub_context.query_factors += i0;
            }
            search_implem_12<C>(
                    i1 - i0, x + d * i0, k,
                    distances + i0 * k, labels + i0 * k,
                    impl, sub_context);
        }
        return;
    }

    // 2. 分配量化查找表
    size_t dim12 = ksub * M2;  // 16 × M2
    AlignedTable<uint8_t> quantized_dis_tables(n * dim12);
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    if (skip & 1) {
        quantized_dis_tables.clear();
    } else {
        compute_quantized_LUT(
                n, x,
                quantized_dis_tables.get(),
                normalizers.get(),
                context);
    }

    // 3. 创建KNN处理器
    std::unique_ptr<SIMDResultHandlerToFloat> handler(
            make_knn_handler(
                    std::is_same<C, CMax<uint16_t, int>>::value,
                    impl, n, k, ntotal,
                    distances, labels, nullptr, context));

    // 4. 执行搜索
    if (skip & 2) {
        // 跳过搜索（用于计时）
    } else {
        size_t nb = ntotal / bbs;  // 完整块数
        size_t bs = bbs;

        // 处理完整的bbs块
        if (nb > 0) {
            // 使用SIMD优化的扫描函数
            pq4_accumulate_default(
                    n, nb, bs,
                    quantized_dis_tables.get(),
                    codes.get(),
                    M2,
                    *handler);
        }

        // 处理剩余向量（不足一个完整块）
        size_t nrem = ntotal % bbs;
        if (nrem > 0) {
            pq4_accumulate_default(
                    n, 1, nrem,
                    quantized_dis_tables.get(),
                    codes.get() + nb * M2 * bs,
                    M2,
                    *handler);
        }
    }
}
```

### 10.6 pq4_accumulate_default - SIMD扫描

```cpp
// faiss/impl/pq4_fast_scan.cpp
// SIMD优化的距离累积（4-bit PQ的FastScan核心）
void pq4_accumulate_default(
        int n,                    // 查询数
        size_t nb,                // 完整块数
        size_t bs,                // 块大小
        const uint8_t* LUT,       // 查找表：n × M2 × 16
        const uint8_t* codes,     // 打包的编码
        int M2,                   // 对齐的子量化器数
        SIMDResultHandlerToFloat& handler) {

    // 对于每个查询
    for (int i = 0; i < n; i++) {
        handler.begin(i);

        const uint8_t* lut = LUT + i * M2 * 16;

        // 对于每个块
        for (size_t b = 0; b < nb; b++) {
            // 初始化累加器（bs个距离）
            uint16_t accu[32];

            // 对于每个子量化器
            for (int m = 0; m < M2; m++) {
                const uint8_t* lut_m = lut + m * 16;
                const uint8_t* codes_m = codes + (b * M2 + m) * bs;

                // SIMD优化：一次处理32个编码
                // 使用AVX2或AVX-512加速
                #ifdef __AVX2__
                __m256i zero = _mm256_setzero_si256();
                __m256i result = _mm256_setzero_si256();

                // 加载32个4-bit编码
                // 每个字节包含2个4-bit编码
                for (int j = 0; j < bs / 8; j++) {
                    __m256i packed = _mm256_loadu_si256(
                            (__m256i*)(codes_m + j * 8));

                    // 解包为两个16字节向量
                    __m256i code_low = _mm256_and_si256(
                            packed, _mm256_set1_epi8(0x0F));
                    __m256i code_high = _mm256_and_si256(
                            _mm256_srli_epi16(packed, 4), _mm256_set1_epi8(0x0F));

                    // 查找表：使用shuffle或permutex2var
                    __m256i dis_low = _mm256_shuffle_epi8(
                            _mm256_loadu_si256((__m256i*)lut_m),
                            code_low);

                    __m256i dis_high = _mm256_shuffle_epi8(
                            _mm256_loadu_si256((__m256i*)lut_m),
                            code_high);

                    // 累加
                    result = _mm256_add_epi16(
                            result, _mm256_add_epi16(dis_low, dis_high));
                }

                // 存储到accu
                _mm256_storeu_si256((__m256i*)accu, result);
                #else
                // 标量版本（无SIMD）
                for (size_t j = 0; j < bs; j++) {
                    uint8_t code = codes_m[j];
                    accu[j] += lut_m[code];
                }
                #endif

                // 处理后续子量化器（累加到accu）
                if (m > 0) {
                    // 第一次已经初始化accu，这里需要累加
                    // ...
                }
            }

            // 将结果添加到handler
            for (size_t j = 0; j < bs; j++) {
                float dis = (float)accu[j];
                handler.add_result(dis, b * bs + j);
            }
        }

        handler.end();
    }
}
```

### 10.7 FastScanDistancePostProcessing - 后处理上下文

```cpp
// faiss/impl/FastScanDistancePostProcessing.h
// FastScan搜索的后处理上下文对象
struct FastScanDistancePostProcessing {
    /// 范数表缩放器（用于加性量化器）
    const NormTableScaler* norm_scaler = nullptr;

    /// RaBitQ的查询因子数据指针
    rabitq_utils::QueryFactorsData* query_factors = nullptr;

    /// 分配query_factors时使用的nprobe值
    size_t nprobe = 0;

    /// 检查是否启用范数缩放
    bool has_norm_scaling() const {
        return norm_scaler != nullptr;
    }

    /// 检查是否启用查询处理
    bool has_query_processing() const {
        return query_factors != nullptr;
    }
};
```

### 10.8 SIMDResultHandlerToFloat - 结果处理器层次

```cpp
// faiss/impl/simd_result_handlers.h
// SIMD结果处理器的基类接口
struct SIMDResultHandlerToFloat {
    // 开始处理第i个查询
    virtual void begin(idx_t i) = 0;

    // 添加一个结果
    virtual void add_result(float dis, idx_t id) = 0;

    // 结束处理第i个查询
    virtual void end(idx_t i) = 0;

    virtual ~SIMDResultHandlerToFloat() {}
};

// Heap处理器：使用堆维护top-K结果
template <class C, bool is_max_ID>
struct HeapHandler : SIMDResultHandlerToFloat {
    idx_t n;                    // 总向量数
    idx_t k;                    // K值
    float* distances;          // 输出距离
    idx_t* labels;              // 输出标签
    const IDSelector* sel;      // ID选择器

    // 堆维护的临时数组
    std::vector<typename C::TI> heap_ids;
    std::vector<typename C::TT> heap_dis;

    HeapHandler(
            idx_t n, idx_t k,
            float* distances, idx_t* labels,
            const IDSelector* sel)
        : n(n), k(k), distances(distances), labels(labels), sel(sel) {
        heap_dis.resize(k);
        heap_ids.resize(k);
    }

    void begin(idx_t i) override {
        // 初始化堆为空或无限值
        for (idx_t j = 0; j < k; j++) {
            heap_dis[j] = C::neutral();
            heap_ids[j] = is_max_ID ? -1 : 0;
        }
        if (k > 0) {
            heap_heapify<C>(k, heap_dis.data(), heap_ids.data());
        }
    }

    void add_result(float dis, idx_t id) override {
        // 检查ID选择器
        if (sel && !sel->is_member(id)) {
            return;
        }

        // 尝试插入到堆中
        if (C::cmp(heap_dis[0], dis)) {
            heap_replace_top<C>(k, heap_dis.data(), heap_ids.data(),
                                dis, id);
        }
    }

    void end(idx_t i) override {
        // 重新排序堆
        heap_reorder<C>(k, heap_dis.data(), heap_ids.data());

        // 复制到输出
        memcpy(distances + i * k, heap_dis.data(), k * sizeof(float));
        memcpy(labels + i * k, heap_ids.data(), k * sizeof(idx_t));
    }
};

// Reservoir处理器：使用蓄水池采样
template <class C, bool is_max_ID>
struct ReservoirHandler : SIMDResultHandlerToFloat {
    idx_t n;
    idx_t k;
    size_t reservoir_size;    // 蓄水池大小（通常>k）
    float* distances;
    idx_t* labels;
    const IDSelector* sel;

    std::vector<typename C::TI> reservoir_ids;
    std::vector<typename C::TT> reservoir_dis;

    ReservoirHandler(
            idx_t n, idx_t k, size_t reservoir_size,
            float* distances, idx_t* labels,
            const IDSelector* sel)
        : n(n), k(k), reservoir_size(reservoir_size),
          distances(distances), labels(labels), sel(sel) {

        reservoir_dis.resize(reservoir_size);
        reservoir_ids.resize(reservoir_size);
    }

    void begin(idx_t i) override {
        // 初始化蓄水池
        for (size_t j = 0; j < reservoir_size; j++) {
            reservoir_dis[j] = C::neutral();
            reservoir_ids[j] = is_max_ID ? -1 : 0;
        }
    }

    void add_result(float dis, idx_t id) override {
        if (sel && !sel->is_member(id)) {
            return;
        }

        // 蓄水池采样逻辑
        for (size_t j = 0; j < reservoir_size; j++) {
            if (C::cmp(reservoir_dis[j], dis)) {
                // 插入并移动其他元素
                for (size_t l = reservoir_size - 1; l > j; l--) {
                    reservoir_dis[l] = reservoir_dis[l - 1];
                    reservoir_ids[l] = reservoir_ids[l - 1];
                }
                reservoir_dis[j] = dis;
                reservoir_ids[j] = id;
                break;
            }
        }
    }

    void end(idx_t i) override {
        // 从蓄水池提取top-K
        // ...堆排序并提取K个最小的...

        memcpy(distances + i * k, reservoir_dis.data(), k * sizeof(float));
        memcpy(labels + i * k, reservoir_ids.data(), k * sizeof(idx_t));
    }
};
```

### 10.9 search_implem_14 - 无qbs版本

```cpp
// faiss/IndexFastScan.cpp
// 实现14：无qbs的heap版本（适用于小查询批次）
template <class C>
void IndexFastScan::search_implem_14(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int impl,
        const FastScanDistancePostProcessing& context) const {

    // 创建结果处理器
    std::unique_ptr<SIMDResultHandlerToFloat> handler(
            make_knn_handler(
                    std::is_same<C, CMax<uint16_t, int>>::value,
                    impl, n, k, ntotal,
                    distances, labels, nullptr, context));

    // 计算量化查找表
    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> quantized_dis_tables(n * dim12);
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    if (!(skip & 1)) {
        compute_quantized_LUT(
                n, x,
                quantized_dis_tables.get(),
                normalizers.get(),
                context);
    }

    // 处理每个查询（独立处理）
    for (idx_t i = 0; i < n; i++) {
        handler->begin(i);

        const uint8_t* lut = quantized_dis_tables.get() + i * dim12;

        // 扫描所有向量
        size_t nb = ntotal / bbs;
        size_t nrem = ntotal % bbs;

        for (size_t b = 0; b < nb; b++) {
            const uint8_t* codes_b = codes.get() + b * M2 * bbs;

            // SIMD优化的扫描
            pq4_scan_database(
                    1, bbs,
                    lut,
                    codes_b,
                    M2,
                    *handler);
        }

        // 处理剩余向量
        if (nrem > 0) {
            const uint8_t* codes_b = codes.get() + nb * M2 * bbs;
            pq4_scan_database(
                    1, nrem,
                    lut,
                    codes_b,
                    M2,
                    *handler);
        }

        handler->end(i);
    }
}
```

### 10.10 生产级使用示例

```cpp
// 生产环境FastScan索引配置
#include <faiss/IndexFastScan.h>
#include <faiss/IndexPQFastScan.h>

void production_fastscan_example() {
    int d = 128;          // 维度
    int M = 16;           // 子量化器数（必须是偶数）
    int nbits = 4;        // 每个子量化器位数（目前只支持4）
    idx_t ntotal = 1000000;

    // 1. 创建IndexPQFastScan（4-bit PQ + FastScan）
    faiss::IndexPQFastScan index(d, M, nbits);

    // 2. 训练量化器
    std::vector<float> xb(d * ntotal);
    // ... 填充xb ...
    index.train(ntotal, xb.data());

    // 3. 添加向量
    index.add(ntotal, xb.data());

    // 4. 搜索
    idx_t nq = 100;
    idx_t k = 100;
    std::vector<float> xq(d * nq);
    std::vector<float> distances(k * nq);
    std::vector<idx_t> labels(k * nq);

    // 选择实现版本
    index.implem = 12;  // blocked with qbs (推荐)

    index.search(nq, xq.data(), k,
                 distances.data(), labels.data());

    // 5. 性能分析
    printf("FastScan index size: %zu bytes\n", index.codes.size());

    // 6. 保存索引
    {
        faiss::IOWriter* writer = new faiss::IOFile("fastscan.index", "wb");
        faiss::write_index(&index, writer);
        delete writer;
    }
}

// IVF + FastScan组合
void ivf_fastscan_example() {
    int d = 128;
    int nlist = 100;       // 倒排列表数
    int M = 16;            // 子量化器数
    idx_t ntotal = 1000000;

    // 1. 创建粗量化器
    faiss::IndexFlatL2 quantizer(d);

    // 2. 创建IndexIVFPQFastScan
    faiss::IndexIVFPQFastScan index(&quantizer, d, nlist, M, 4);

    // 3. 训练并添加
    std::vector<float> xb(d * ntotal);
    // ... 填充xb ...
    index.train(ntotal, xb.data());
    index.add(ntotal, xb.data());

    // 4. 搜索
    idx_t nq = 100;
    idx_t k = 100;
    idx_t nprobe = 10;     // 搜索的倒排列表数

    std::vector<float> xq(d * nq);
    std::vector<float> distances(k * nq);
    std::vector<idx_t> labels(k * nq);

    faiss::SearchParametersIVF params;
    params.nprobe = nprobe;

    index.search(nq, xq.data(), k,
                 distances.data(), labels.data(), &params);
}
```

### 10.11 SIMD优化细节

```cpp
// AVX2优化的4-bit PQ查找表实现
#ifdef __AVX2__
inline __m256i lookup_4bit_avx2(__m256i codes_4bit, const float* table) {
    // codes_4bit: 32个4-bit编码（打包成16字节）
    // table: 16个float距离值

    // 由于每个字节包含2个4-bit编码
    // 我们需要分别处理低4位和高4位

    __m256i mask_low = _mm256_set1_epi8(0x0F);
    __m256i mask_high = _mm256_set1_epi8(0xF0);

    // 提取低4位
    __m256i low = _mm256_and_si256(codes_4bit, mask_low);

    // 提取高4位并右移
    __m256i high = _mm256_and_si256(codes_4bit, mask_high);
    high = _mm256_srli_epi16(high, 4);

    // 使用shuffle作为查找表
    // table需要预先安排为特定格式
    __m256i table_vec = _mm256_loadu_si256((__m256i*)table);

    __m256i result_low = _mm256_shuffle_epi8(table_vec, low);
    __m256i result_high = _mm256_shuffle_epi8(table_vec, high);

    // 组合结果（这里简化为返回，实际需要int16累加）
    return _mm256_add_epi16(result_low, result_high);
}
#endif

// AVX-512优化的4-bit PQ查找表
#ifdef __AVX512F__
inline __m512i lookup_4bit_avx512(__m512i codes_4bit, const float* table) {
    // codes_4bit: 64个4-bit编码（32字节）
    // table: 16个float距离值

    __m512i mask_low = _mm512_set1_epi8(0x0F);
    __m512i mask_high = _mm512_set1_epi8(0xF0);

    __m512i low = _mm512_and_si512(codes_4bit, mask_low);
    __m512i high = _mm512_and_si512(codes_4bit, mask_high);
    high = _mm512_srli_epi16(high, 4);

    // AVX-512的permutevar更灵活
    __m512i table_vec = _mm512_loadu_si512((__m512i*)table);

    __m512i result_low = _mm512_permutexvar_epi8(table_vec, low);
    __m512i result_high = _mm512_permutexvar_epi8(table_vec, high);

    return _mm512_add_epi16(result_low, result_high);
}
#endif
```

---

## 7. 第9天总结

### 关键概念

1. **FastScan布局**：按子量化器重组编码
2. **SIMD查找表**：使用shuffle/gather加速
3. **批量处理**：一次处理batch_size个向量
4. **内存对齐**：确保最佳SIMD性能

### 性能提升

| 优化 | 加速比 |
|------|--------|
| 重新布局 | 2x |
| SIMD查找表 | 2x |
| 预取 | 1.2x |
| 批量处理 | 1.5x |
| **总计** | **~6x** |

### 下一步

第10天将学习**标量量化和RaBitQ**，其他重要的量化方法。

---

## 11. FastScan底层SIMD深度优化

### 11.1 AVX-512BW优化的4-bit查找

```cpp
// faiss/impl/pq4_fast_scan.cpp
// AVX-512BW（Byte Words）优化的查找表实现
#ifdef __AVX512BW__

// 单次处理64个4-bit编码
void pq4_accumulate_512bw(
        int nquery,
        size_t nb,
        const uint8_t* LUT,
        const uint8_t* codes,
        int M2,
        SIMDResultHandlerToFloat& handler) {

    for (int q = 0; q < nquery; q++) {
        handler.begin(q);

        const uint8_t* lut = LUT + q * M2 * 16;
        alignas(64) uint16_t accu[64];

        for (size_t b = 0; b < nb; b++) {
            // 初始化累加器
            __m512i sum = _mm512_setzero_si512();

            for (int m = 0; m < M2; m++) {
                const uint8_t* lut_m = lut + m * 16;
                const uint8_t* codes_m = codes + (b * M2 + m) * 32;

                // 加载32字节编码（包含64个4-bit值）
                __m512i packed = _mm512_loadu_si512(codes_m);

                // 掩码提取
                __m512i mask_0f = _mm512_set1_epi8(0x0F);

                // 提取低4位
                __m512i low = _mm512_and_si512(packed, mask_0f);

                // 提取高4位
                __m512i high = _mm512_and_si512(
                        _mm512_srli_epi16(packed, 4), mask_0f);

                // 加载LUT（需要预先安排）
                __m512i lut_vec = _mm512_loadu_si512(
                        (const __m512i*)lut_m);

                // 使用vpshufb进行查找表（需要分两次处理）
                __m512i dis_low = _mm512_shuffle_epi8(lut_vec, low);
                __m512i dis_high = _mm512_shuffle_epi8(lut_vec, high);

                // 零扩展到16位
                __m512i dis_low_16 = _mm512_cvtepu8_epi16(dis_low);
                __m512i dis_high_16 = _mm512_cvtepu8_epi16(dis_high);

                // 累加
                sum = _mm512_add_epi16(sum,
                        _mm512_add_epi16(dis_low_16, dis_high_16));
            }

            // 存储结果
            _mm512_storeu_si512((__m512i*)accu, sum);

            // 添加到handler
            for (int i = 0; i < 64; i++) {
                handler.add_result((float)accu[i], b * 64 + i);
            }
        }

        handler.end(q);
    }
}

// 使用AVX-512BW的掩码操作过滤无效向量
void pq4_accumulate_512bw_with_mask(
        int nquery,
        size_t nb,
        const uint8_t* LUT,
        const uint8_t* codes,
        const uint64_t* valid_mask,  // 每个bit表示一个向量是否有效
        int M2,
        SIMDResultHandlerToFloat& handler) {

    for (int q = 0; q < nquery; q++) {
        handler.begin(q);

        const uint8_t* lut = LUT + q * M2 * 16;
        alignas(64) uint16_t accu[64];

        for (size_t b = 0; b < nb; b++) {
            __m512i sum = _mm512_setzero_si512();

            // 加载有效性掩码
            __mmask64 valid = valid_mask[b];

            for (int m = 0; m < M2; m++) {
                const uint8_t* lut_m = lut + m * 16;
                const uint8_t* codes_m = codes + (b * M2 + m) * 32;

                __m512i packed = _mm512_loadu_si512(codes_m);
                __m512i mask_0f = _mm512_set1_epi8(0x0F);

                __m512i low = _mm512_and_si512(packed, mask_0f);
                __m512i high = _mm512_and_si512(
                        _mm512_srli_epi16(packed, 4), mask_0f);

                __m512i lut_vec = _mm512_loadu_si512(
                        (const __m512i*)lut_m);

                __m512i dis_low = _mm512_shuffle_epi8(lut_vec, low);
                __m512i dis_high = _mm512_shuffle_epi8(lut_vec, high);

                __m512i dis_low_16 = _mm512_cvtepu8_epi16(dis_low);
                __m512i dis_high_16 = _mm512_cvtepu8_epi16(dis_high);

                sum = _mm512_add_epi16(sum,
                        _mm512_add_epi16(dis_low_16, dis_high_16));
            }

            _mm512_storeu_si512((__m512i*)accu, sum);

            // 只处理有效向量
            for (int i = 0; i < 64; i++) {
                if (valid & (1ULL << i)) {
                    handler.add_result((float)accu[i], b * 64 + i);
                }
            }
        }

        handler.end(q);
    }
}
#endif
```

### 11.2 ARM NEON优化的FastScan

```cpp
// faiss/impl/pq4_fast_scan.cpp
// ARM NEON优化的4-bit PQ扫描
#ifdef __ARM_NEON

#include <arm_neon.h>

void pq4_accumulate_neon(
        int nquery,
        size_t nb,
        const uint8_t* LUT,
        const uint8_t* codes,
        int M2,
        SIMDResultHandlerToFloat& handler) {

    for (int q = 0; q < nquery; q++) {
        handler.begin(q);

        const uint8_t* lut = LUT + q * M2 * 16;
        alignas(16) uint16_t accu[32];

        for (size_t b = 0; b < nb; b++) {
            // 初始化累加器（处理32个向量）
            uint16x8_t sum0 = vdupq_n_u16(0);
            uint16x8_t sum1 = vdupq_n_u16(0);
            uint16x8_t sum2 = vdupq_n_u16(0);
            uint16x8_t sum3 = vdupq_n_u16(0);

            for (int m = 0; m < M2; m++) {
                const uint8_t* lut_m = lut + m * 16;
                const uint8_t* codes_m = codes + (b * M2 + m) * 16;

                // 加载16字节编码（32个4-bit值）
                uint8x16_t packed = vld1q_u8(codes_m);

                // 掩码
                uint8x16_t mask_0f = vdupq_n_u8(0x0F);

                // 提取低4位
                uint8x16_t low = vandq_u8(packed, mask_0f);

                // 提取高4位
                uint8x16_t high = vshlq_n_u8(packed, -4);
                high = vandq_u8(high, mask_0f);

                // NEON的vtbl查表（需要分两次）
                // 准备查找表（每16字节为一组）
                uint8x8_t lut_low = vld1_u8(lut_m);
                uint8x8_t lut_high = vld1_u8(lut_m + 8);

                // 查表（8元素一组）
                uint8x8_t low_0 = vget_low_u8(low);
                uint8x8_t low_1 = vget_high_u8(low);
                uint8x8_t high_0 = vget_low_u8(high);
                uint8x8_t high_1 = vget_high_u8(high);

                // 使用vtbl查表
                uint8x8_t dis_low_0 = vtbl1_u8(lut_low, low_0);
                uint8x8_t dis_low_1 = vtbl1_u8(lut_low, low_1);
                uint8x8_t dis_high_0 = vtbl1_u8(lut_high, high_0);
                uint8x8_t dis_high_1 = vtbl1_u8(lut_high, high_1);

                // 合并
                uint8x16_t dis_low = vcombine_u8(dis_low_0, dis_low_1);
                uint8x16_t dis_high = vcombine_u8(dis_high_0, dis_high_1);

                // 扩展到16位
                uint16x8_t dis_low_0_16 = vmovl_u8(vget_low_u8(dis_low));
                uint16x8_t dis_low_1_16 = vmovl_u8(vget_high_u8(dis_low));
                uint16x8_t dis_high_0_16 = vmovl_u8(vget_low_u8(dis_high));
                uint16x8_t dis_high_1_16 = vmovl_u8(vget_high_u8(dis_high));

                // 累加
                sum0 = vaddq_u16(sum0, vaddq_u16(dis_low_0_16, dis_high_0_16));
                sum1 = vaddq_u16(sum1, vaddq_u16(dis_low_1_16, dis_high_1_16));
            }

            // 存储结果
            vst1q_u16(accu, sum0);
            vst1q_u16(accu + 8, sum1);
            vst1q_u16(accu + 16, sum2);
            vst1q_u16(accu + 24, sum3);

            // 添加到handler
            for (int i = 0; i < 32; i++) {
                handler.add_result((float)accu[i], b * 32 + i);
            }
        }

        handler.end(q);
    }
}
#endif
```

### 11.3 向量长度无关（VLA）优化

```cpp
// ARM SVE（Scalable Vector Extension）优化
// 向量长度在运行时确定，编译时未知
#ifdef __ARM_SVE__

#include <arm_sve.h>

void pq4_accumulate_sve(
        int nquery,
        size_t nb,
        const uint8_t* LUT,
        const uint8_t* codes,
        int M2,
        SIMDResultHandlerToFloat& handler) {

    for (int q = 0; q < nquery; q++) {
        handler.begin(q);

        const uint8_t* lut = LUT + q * M2 * 16;

        // 获取向量长度（运行时）
        const uint32_t vl = svcnth();  // number of uint16_t in vector

        std::vector<uint16_t> accu_buffer(vl * 4);

        for (size_t b = 0; b < nb; b++) {
            svuint16_t sum0 = svdup_n_u16(0);
            svuint16_t sum1 = svdup_n_u16(0);
            svuint16_t sum2 = svdup_n_u16(0);
            svuint16_t sum3 = svdup_n_u16(0);

            for (int m = 0; m < M2; m++) {
                const uint8_t* lut_m = lut + m * 16;
                const uint8_t* codes_m = codes + (b * M2 + m) * vl;

                // 加载可变长度的编码
                svuint8_t packed = svld1_u8(svptrue_b8(), codes_m);

                // 掩码
                svuint8_t mask_0f = svdup_n_u8(0x0F);

                // 提取低4位和高4位
                svuint8_t low = svand_u8_m(svptrue_b8(), packed, mask_0f);
                svuint8_t high = svand_u8_m(
                        svptrue_b8(),
                        svlsr_n_u8_m(svptrue_b8(), packed, 4),
                        mask_0f);

                // SVE查表使用svtbl
                // 需要分两次处理（每16个元素一组）
                svuint8_t lut_vec = svld1_u8(svptrue_b8(), lut_m);

                svuint8_t dis_low = svtbl_u8(lut_vec, low);
                svuint8_t dis_high = svtbl_u8(lut_vec, high);

                // 扩展到16位
                svuint16_t dis_low_16 = svunpklo_u16(dis_low);  // 低半部分
                svuint16_t dis_high_16 = svunpklo_u16(dis_high);

                // 累加
                sum0 = svadd_u16_m(svptrue_b16(), sum0,
                                   svadd_u16_m(svptrue_b16(),
                                              dis_low_16, dis_high_16));
            }

            // 存储结果
            svst1_u16(svptrue_b16(), accu_buffer.data(), sum0);
            svst1_u16(svptrue_b16(), accu_buffer.data() + vl, sum1);

            // 添加到handler
            for (uint32_t i = 0; i < vl * 2; i++) {
                handler.add_result((float)accu_buffer[i], b * vl * 2 + i);
            }
        }

        handler.end(q);
    }
}
#endif
```

---

## 12. FastScan内存布局深度优化

### 12.1 缓存行对齐的数据结构

```cpp
// faiss/impl/pq4_fast_scan.cpp
// 确保所有数据结构缓存行对齐

struct alignas(64) FastScanBlock {
    // 每个块包含bbs个向量的编码
    // 确保块起始地址对齐到64字节（缓存行）

    uint8_t codes[M2 * 32];  // M2个子量化器，每个32字节

    // 预取优化：访问模式预测
    void prefetch_next() const {
        // 预取下一个块到L2缓存
        const FastScanBlock* next = this + 1;
        _mm_prefetch((const char*)next, _MM_HINT_T1);
    }
};

// 交错存储：进一步优化缓存利用率
struct InterleavedFastScanLayout {
    // 将多个子量化器的编码交错存储
    // 提高空间局部性

    size_t M;           // 子量化器数
    size_t bbs;         // 块大小
    size_t interleave;  // 交错因子（例如4）

    std::vector<uint8_t> codes;

    void encode_interleaved(
            const uint8_t* flat_codes,
            size_t nb) {

        // 原始布局：[M0, M1, M2, ..., M15] × bbs
        // 交错布局：[M0_M4_M8_M12, M1_M5_M9_M13, ...] × (bbs/4)

        for (size_t b = 0; b < nb; b += bbs) {
            for (size_t inter = 0; inter < interleave; inter++) {
                for (size_t m = 0; m < M; m++) {
                    if (m % interleave == inter) {
                        // 复制编码
                        size_t src = b * M + m;
                        size_t dst = (b / interleave) * M +
                                    (m / interleave) * bbs +
                                    (m % interleave);
                        codes[dst] = flat_codes[src];
                    }
                }
            }
        }
    }
};
```

### 12.2 NUMA感知的FastScan

```cpp
// NUMA优化的FastScan索引
#ifdef __linux__
#include <numa.h>

struct NUMAFastScanIndex {
    int numa_nodes;
    size_t vectors_per_node;

    // 每个NUMA节点独立的编码存储
    std::vector<void*> node_codes;
    std::vector<size_t> node_sizes;

    void init_numa(int nodes, size_t total_vectors) {
        numa_nodes = nodes;
        vectors_per_node = (total_vectors + nodes - 1) / nodes;

        node_codes.resize(nodes);
        node_sizes.resize(nodes);

        for (int node = 0; node < nodes; node++) {
            size_t size = vectors_per_node * M2 / 2;
            node_codes[node] = numa_alloc_onnode(size, node);
            node_sizes[node] = size;
        }
    }

    void add_vector_to_local_node(
            const float* vec,
            idx_t id) {

        int node = numa_node_of_cpu(sched_getcpu());
        size_t local_id = id % vectors_per_node;
        size_t offset = local_id * M2 / 2;

        // 编码并存储到本地NUMA节点
        uint8_t* local_codes = static_cast<uint8_t*>(node_codes[node]);
        encode_and_store(vec, local_codes + offset);
    }

    // 搜索时访问本地NUMA节点的数据
    void search_numa_aware(
            const float* query,
            float* distances,
            idx_t* labels) {

        // 绑定到当前NUMA节点
        numa_set_preferred(numa_node_of_cpu(sched_getcpu()));

        // 搜索本地节点数据
        int node = numa_node_of_cpu(sched_getcpu());
        const uint8_t* local_codes =
                static_cast<const uint8_t*>(node_codes[node]);

        pq4_accumulate_default(
                1, vectors_per_node, bbs,
                get_lut(query),
                local_codes,
                M2,
                *handler);
    }

    ~NUMAFastScanIndex() {
        for (int node = 0; node < numa_nodes; node++) {
            numa_free(node_codes[node], node_sizes[node]);
        }
    }
};
#endif
```

### 12.3 预取策略深度优化

```cpp
// 软件预取与硬件预取协同
template<int PREFETCH_DISTANCE = 8>
struct PrefetchingFastScanIterator {
    const uint8_t* codes;
    const uint8_t* lut;
    size_t nb;
    int M2;

    // 预取队列（流水线）
    struct alignas(64) PrefetchEntry {
        const uint8_t* code_ptr;
        size_t block_id;
    };

    PrefetchEntry prefetch_queue[PREFETCH_DISTANCE];
    int queue_head = 0;
    int queue_tail = 0;

    void prefetch_ahead(size_t current_block) {
        // 预取未来的块
        for (int i = 1; i <= PREFETCH_DISTANCE; i++) {
            size_t future_block = current_block + i;
            if (future_block < nb) {
                const uint8_t* future_codes =
                        codes + future_block * M2 * bbs;

                // 预取到L2缓存（T1）
                _mm_prefetch((const char*)future_codes, _MM_HINT_T1);

                // 预取LUT到L1缓存
                _mm_prefetch((const char*)lut, _MM_HINT_T0);
            }
        }
    }

    void prefetch_with_nta(size_t current_block) {
        // 使用非临时提示（NTA）避免污染缓存
        // 适用于数据只访问一次的场景

        size_t future_block = current_block + PREFETCH_DISTANCE;
        if (future_block < nb) {
            const uint8_t* future_codes =
                    codes + future_block * M2 * bbs;

            // _MM_HINT_NTA: Non-Temporal Address
            _mm_prefetch((const char*)future_codes, _MM_HINT_NTA);
        }
    }
};

// 自适应预取：根据缓存未命中率调整
struct AdaptivePrefetchFastScan {
    float cache_miss_rate;
    int prefetch_distance;

    void adjust_prefetch_distance(float new_miss_rate) {
        // 如果缓存未命中率高，增加预取距离
        if (new_miss_rate > 0.1) {  // 10%阈值
            prefetch_distance = std::min(16, prefetch_distance + 2);
        } else if (new_miss_rate < 0.05) {
            // 未命中率低，减少预取避免过度预取
            prefetch_distance = std::max(4, prefetch_distance - 1);
        }
    }
};
```

### 12.4 Huge Pages优化

```cpp
// 使用大页减少TLB未命中
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>

struct HugePageFastScanIndex {
    size_t code_size;

    void* allocate_huge_page(size_t size) {
        // 使用透明大页（THP）
        void* ptr = mmap(nullptr, size,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS,
                         -1, 0);

        if (ptr == MAP_FAILED) {
            perror("mmap");
            return nullptr;
        }

        // 建议使用大页（MADV_HUGEPAGE）
        madvise(ptr, size, MADV_HUGEPAGE);

        return ptr;
    }

    void init_with_huge_pages(size_t ntotal, int M, int nbits) {
        code_size = ntotal * M * nbits / 8;

        // 对齐到大页大小（通常2MB）
        size_t huge_page_size = 2 * 1024 * 1024;
        code_size = (code_size + huge_page_size - 1) / huge_page_size *
                    huge_page_size;

        codes = static_cast<uint8_t*>(
                allocate_huge_page(code_size));
    }

    ~HugePageFastScanIndex() {
        if (codes) {
            munmap(codes, code_size);
        }
    }

private:
    uint8_t* codes = nullptr;
};
#endif
```

---

## 13. FastScan并发优化

### 13.1 并行块级处理

```cpp
// OpenMP并行的FastScan搜索
void parallel_fast_scan_search(
        const IndexFastScan& index,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) {

    // 计算查找表（可并行）
    size_t dim12 = index.ksub * index.M2;
    AlignedTable<uint8_t> quantized_dis_tables(n * dim12);
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    #pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        index.compute_quantized_LUT(
                1, x + i * index.d,
                quantized_dis_tables.get() + i * dim12,
                normalizers.get() + 2 * i,
                FastScanDistancePostProcessing());
    }

    // 并行搜索
    #pragma omp parallel
    {
        // 每个线程独立的handler
        std::unique_ptr<SIMDResultHandlerToFloat> handler(
                index.make_knn_handler(
                        true, 12, 1, k, index.ntotal,
                        distances, labels, nullptr,
                        FastScanDistancePostProcessing()));

        #pragma omp for schedule(dynamic, 1)
        for (idx_t i = 0; i < n; i++) {
            handler->begin(i);

            const uint8_t* lut = quantized_dis_tables.get() + i * dim12;
            size_t nb = index.ntotal / index.bbs;

            for (size_t b = 0; b < nb; b++) {
                const uint8_t* codes_b =
                        index.codes.get() + b * index.M2 * index.bbs;

                pq4_accumulate_default(
                        1, 1, index.bbs,
                        lut,
                        codes_b,
                        index.M2,
                        *handler);
            }

            handler->end(i);
        }
    }
}
```

### 13.2 NUMA亲和性优化

```cpp
// 绑定线程到NUMA节点
#ifdef __linux__
#include <numa.h>
#include <sched.h>

struct NUMAAwareThreadPool {
    int numa_nodes;
    int threads_per_node;

    void bind_threads_to_numa_nodes() {
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int numa_node = thread_id % numa_nodes;

            // 获取NUMA节点的CPU列表
            struct bitmask* cpumask = numa_allocate_cpumask();
            numa_node_to_cpus(numa_node, cpumask);

            // 绑定到第一个CPU
            for (unsigned int cpu = 0; cpu < numa_num_possible_cpus(); cpu++) {
                if (numa_bitmask_isbitset(cpumask, cpu)) {
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(cpu, &cpuset);

                    pthread_setaffinity_np(
                            pthread_self(),
                            sizeof(cpuset),
                            &cpuset);
                    break;
                }
            }

            numa_free_cpumask(cpumask);
        }
    }

    void set_numa_memory_policy() {
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int numa_node = thread_id % numa_nodes;

            // 设置内存分配策略为本地
            numa_set_preferred(numa_node);
        }
    }
};
#endif
```

### 13.3 无锁结果聚合

```cpp
// 使用原子操作的无锁结果聚合
#include <atomic>

struct LockFreeResultAggregator {
    struct alignas(64) AtomicResult {
        std::atomic<float> distance;
        std::atomic<idx_t> label;

        AtomicResult()
            : distance(INFINITY), label(-1) {}
    };

    std::vector<AtomicResult> results;
    idx_t k;

    LockFreeResultAggregator(idx_t nqueries, idx_t k)
        : k(k) {
        // 每个查询k个结果
        results.resize(nqueries * k);
    }

    bool try_add_result(
            idx_t query_id,
            float distance,
            idx_t label) {

        // 尝试插入到结果集中
        AtomicResult* query_results = &results[query_id * k];

        for (idx_t i = 0; i < k; i++) {
            float current_dist = query_results[i].distance.load(
                    std::memory_order_relaxed);

            // 如果新距离更小，尝试替换
            if (distance < current_dist) {
                float expected = current_dist;

                if (query_results[i].distance.compare_exchange_strong(
                        expected, distance,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire)) {

                    // 成功替换距离，更新标签
                    query_results[i].label.store(label,
                            std::memory_order_release);
                    return true;
                }
                // CAS失败，重试或继续
            }
        }

        return false;  // 未进入top-K
    }

    void copy_to_output(float* distances, idx_t* labels) {
        // 复制到输出数组（需要排序）
        for (size_t q = 0; q < results.size() / k; q++) {
            std::vector<std::pair<float, idx_t>> tmp;

            for (idx_t i = 0; i < k; i++) {
                float dis = results[q * k + i].distance.load(
                        std::memory_order_relaxed);
                idx_t lbl = results[q * k + i].label.load(
                        std::memory_order_relaxed);

                if (lbl != -1) {
                    tmp.push_back({dis, lbl});
                }
            }

            // 排序
            std::sort(tmp.begin(), tmp.end());

            // 复制
            for (idx_t i = 0; i < k && i < tmp.size(); i++) {
                distances[q * k + i] = tmp[i].first;
                labels[q * k + i] = tmp[i].second;
            }
        }
    }
};
```

### 13.4 SIMD并行的结果处理

```cpp
// SIMD优化的top-K维护
template<int K>
struct SIMDTopKHeap {
    alignas(64) float distances[K];
    alignas(64) idx_t labels[K];
    int size = 0;

    void init() {
        for (int i = 0; i < K; i++) {
            distances[i] = std::numeric_limits<float>::infinity();
            labels[i] = -1;
        }
        size = 0;
    }

#ifdef __AVX2__
    // SIMD优化的批量插入
    void add_batch_simd(
            const float* batch_dis,
            const idx_t* batch_labels,
            int batch_size) {

        for (int i = 0; i < batch_size; i++) {
            float dis = batch_dis[i];
            idx_t lbl = batch_labels[i];

            // SIMD比较：检查是否比最大值小
            __m256 dis_vec = _mm256_set1_ps(dis);
            __m256 heap_vec = _mm256_loadu_ps(distances);

            __m256 cmp = _mm256_cmp_ps(dis_vec, heap_vec, _CMP_LT_OS);

            // 如果有任何一个满足
            if (_mm256_movemask_ps(cmp) != 0) {
                // 找到最大值的位置并替换
                int max_idx = 0;
                float max_val = distances[0];

                for (int j = 1; j < K; j++) {
                    if (distances[j] > max_val) {
                        max_val = distances[j];
                        max_idx = j;
                    }
                }

                if (dis < distances[max_idx]) {
                    distances[max_idx] = dis;
                    labels[max_idx] = lbl;
                }
            }
        }
    }
#endif
};
```

---

## 14. FastScan性能分析工具

### 14.1 细粒度性能计数器

```cpp
// FastScan专用的性能分析器
struct FastScanProfiler {
    std::atomic<uint64_t> lut_computations{0};
    std::atomic<uint64_t> blocks_processed{0};
    std::atomic<uint64_t> vectors_scanned{0};
    std::atomic<uint64_t> simd_operations{0};

    // 缓存性能
    std::atomic<uint64_t> l1_misses{0};
    std::atomic<uint64_t> l2_misses{0};
    std::atomic<uint64_t> l3_misses{0};

    // 时间统计
    std::atomic<double> time_lut{0};
    std::atomic<double> time_scan{0};
    std::atomic<double> time_aggregate{0};

    void print_report() const {
        printf("=== FastScan Performance Report ===\n");
        printf("LUT computations: %lu\n", lut_computations.load());
        printf("Blocks processed: %lu\n", blocks_processed.load());
        printf("Vectors scanned: %lu\n", vectors_scanned.load());
        printf("SIMD operations: %lu\n", simd_operations.load());
        printf("Vectorization ratio: %.1f%%\n",
               100.0 * simd_operations.load() /
                   std::max<uint64_t>(1, vectors_scanned.load()));
        printf("\nTiming breakdown:\n");
        printf("  LUT computation: %.2f ms\n", time_lut.load());
        printf("  Scanning: %.2f ms\n", time_scan.load());
        printf("  Aggregation: %.2f ms\n", time_aggregate.load());
    }

    void reset() {
        lut_computations = 0;
        blocks_processed = 0;
        vectors_scanned = 0;
        simd_operations = 0;
        l1_misses = 0;
        l2_misses = 0;
        l3_misses = 0;
        time_lut = 0;
        time_scan = 0;
        time_aggregate = 0;
    }
};

// 在关键路径插入计数器
void profiled_fast_scan_search(
        const IndexFastScan& index,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        FastScanProfiler& profiler) {

    auto t0 = std::chrono::high_resolution_clock::now();

    // 计算LUT
    size_t dim12 = index.ksub * index.M2;
    AlignedTable<uint8_t> quantized_dis_tables(n * dim12);

    for (idx_t i = 0; i < n; i++) {
        index.compute_quantized_LUT(
                1, x + i * index.d,
                quantized_dis_tables.get() + i * dim12,
                nullptr,
                FastScanDistancePostProcessing());
        profiler.lut_computations++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    profiler.time_lut += std::chrono::duration<double>(t1 - t0).count() * 1000;

    // 扫描
    for (idx_t i = 0; i < n; i++) {
        const uint8_t* lut = quantized_dis_tables.get() + i * dim12;
        size_t nb = index.ntotal / index.bbs;

        for (size_t b = 0; b < nb; b++) {
            profiler.blocks_processed++;
            profiler.vectors_scanned += index.bbs;
            profiler.simd_operations += index.bbs / 16;  // AVX2处理16个

            // ... 实际扫描代码 ...
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    profiler.time_scan += std::chrono::duration<double>(t2 - t1).count() * 1000;
}
```

### 14.2 自动性能调优

```cpp
// FastScan参数自动调优
struct FastScanTuner {
    struct Config {
        int bbs;           // 块大小
        int qbs;           // 查询块大小
        int implem;        // 实现版本
        double qps;        // queries per second
        double memory_mb;
    };

    std::vector<Config> results;

    void grid_search(
            const float* train_vectors,
            idx_t ntrain,
            const float* query_vectors,
            idx_t nquery,
            int d) {

        std::vector<int> bbs_values = {32, 64, 128};
        std::vector<int> qbs_values = {0, 11, 12, 13};
        std::vector<int> implem_values = {12, 14, 15};

        for (int bbs : bbs_values) {
            IndexPQFastScan index(d, 16, 4);
            index.bbs = bbs;
            index.train(ntrain, train_vectors);
            index.add(ntrain, train_vectors);

            for (int qbs : qbs_values) {
                index.qbs = qbs;

                for (int implem : implem_values) {
                    index.implem = implem;

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

                    results.push_back({bbs, qbs, implem, qps, 0.0});
                }
            }
        }
    }

    Config find_optimal() {
        return *std::max_element(
                results.begin(), results.end(),
                [](const Config& a, const Config& b) {
                    return a.qps < b.qps;
                });
    }
};
```

---

## 练习题

1. 实现FastScan布局转换
2. 编写AVX2优化的查表函数
3. 比较FastScan与常规PQ的性能
4. 研究不同batch_size的影响
5. 实现ARM NEON优化的FastScan
6. 分析FastScan的缓存命中率
7. 实现NUMA感知的FastScan索引

## 扩展阅读

- faiss/IndexIVFFastScan.h - FastScan索引
- faiss/utils/distances_simd.cpp - SIMD距离计算
- faiss/impl/pq4_fast_scan.cpp - 4-bit PQ FastScan实现
- [SIMD查找表技术](https://github.com/facebookresearch/faiss/wiki/Fast-Scan-(IVF-PQ)-mode)
- [AVX-512编程指南](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
