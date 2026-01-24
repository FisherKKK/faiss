# ProductQuantizer底层实现深度剖析 - 源码解析

## 文档说明

本文档深入剖析Faiss中ProductQuantizer的底层实现，基于`ProductQuantizer.h`和`ProductQuantizer.cpp`源码，详细讲解乘积量化的训练、编码、解码等核心技术。

**前置知识**：
- 已完成《Faiss基础教程》
- 了解k-means聚类算法
- 熟悉向量量化原理

---

## 目录
- [1. Product Quantizer概述](#1-product-quantizer概述)
- [2. 数据结构设计](#2-数据结构设计)
- [3. 编码器实现](#3-编码器实现)
- [4. 解码器实现](#4-解码器实现)
- [5. 训练流程](#5-训练流程)
- [6. 距离计算优化](#6-距离计算优化)
- [7. 性能优化技巧](#7-性能优化技巧)
- [8. 总结](#8-总结)

---

## 1. Product Quantizer概述

### 1.1 什么是Product Quantization

**Product Quantization (PQ)**是一种向量量化技术，将高维向量分解为多个低维子向量，分别量化：

```
原始向量: x ∈ R^d (128维)
分解为: x = [x_0, x_1, ..., x_7], 每个子向量16维

量化: 对每个子向量独立量化
x_i ∈ R^16 → code_i ∈ {0, 1, ..., 255} (8-bit)
x ≈ [c_{code_0}, c_{code_1}, ..., c_{code_7}]

压缩比: 128×4字节 = 512字节 → 8字节 (64倍压缩)
```

### 1.2 PQ的数学原理

**量化目标**：
```
min Σ ||x - c_{code(x)}||²
code
```

**解的结构**：
```
原始向量: d维
子量化器: M个
子向量维度: dsub = d / M
每个子量化器: ksub = 2^nbits个质心
编码大小: code_size = (nbits × M + 7) / 8 字节
```

### 1.3 Faiss PQ参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| d | 原始向量维度 | 128, 256, 384 |
| M | 子量化器数量 | 8, 16, 32, 64 |
| nbits | 每个子向量编码位数 | 8, 16 |
| dsub | 子向量维度 | d/M |
| ksub | 质心数量 | 2^nbits |
| code_size | 编码大小(字节) | (nbits×M+7)/8 |

---

## 2. 数据结构设计

### 2.1 ProductQuantizer类结构

```cpp
struct ProductQuantizer : Quantizer {
    size_t M;         ///< 子量化器数量
    size_t nbits;     ///< 每个子量化器的编码位数
    size_t dsub;      ///< 每个子向量的维度
    size_t ksub;      ///< 每个子量化器的质心数量 (2^nbits)
    bool verbose;     ///< 训练时是否输出详细信息

    // 训练类型
    enum train_type_t {
        Train_default,       // 默认k-means
        Train_hot_start,     // 热启动（已有质心）
        Train_shared,        // 共享字典
        Train_hypercube,     // 超立方初始化
        Train_hypercube_pca,  // 超立方+PCA初始化
    };
    train_type_t train_type;

    // 聚类参数
    ClusteringParameters cp;

    // 分配索引（可选）
    Index* assign_index;

    // 质心表
    std::vector<float> centroids;         // (M × ksub × dsub)
    std::vector<float> transposed_centroids;  // (dsub × M × ksub)
    std::vector<float> centroids_sq_lengths;  // (M × ksub)
};
```

### 2.2 派生值计算

```cpp
void ProductQuantizer::set_derived_values() {
    // 验证维度可整除
    FAISS_THROW_IF_NOT_MSG(
            d % M == 0,
            "The dimension of the vector (d) should be a multiple of "
            "the number of subquantizers (M)");

    // 计算子向量维度
    dsub = d / M;

    // 计算编码大小
    code_size = (nbits * M + 7) / 8;

    // 限制nbits
    FAISS_THROW_IF_MSG(nbits > 24,
            "nbits larger than 24 is not practical.");

    // 计算质心数量
    ksub = 1 << nbits;  // 2^nbits

    // 分配质心表
    centroids.resize(d * ksub);

    // 设置默认值
    verbose = false;
    train_type = Train_default;
}
```

**示例计算**：
```
输入: d = 128, M = 8, nbits = 8

dsub = 128 / 8 = 16
code_size = (8 × 8 + 7) / 8 = 8 字节
ksub = 2^8 = 256

内存布局:
centroids[0]: 256个质心 × 16维
centroids[1]: 256个质心 × 16维
...
centroids[7]: 256个质心 × 16维

总计: 8 × 256 × 16 = 32768 个float
```

---

## 3. 编码器实现

### 3.1 PQEncoderGeneric

```cpp
struct PQEncoderGeneric {
    uint8_t* code;      // 输出编码
    uint8_t offset;     // 当前位偏移
    const int nbits;    // 每个子量化器的位数
    uint8_t reg;       // 当前寄存器

    PQEncoderGeneric(uint8_t* code, int nbits, uint8_t offset = 0)
            : code(code), offset(offset), nbits(nbits), reg(0) {}

    void encode(uint64_t x) {
        // 编码x到code[offset]
        // 每次编码nbits位
        // 支持任意nbits (不限于8位)
    }
};
```

### 3.2 PQEncoder8

```cpp
struct PQEncoder8 {
    uint8_t* code;
    PQEncoder8(uint8_t* code, int nbits);

    void encode(uint64_t x) {
        // 专门优化8位编码
        // 每次编码8位到1个字节
        *code++ = (uint8_t)x;
    }
};
```

### 3.3 compute_code函数

```cpp
template <class PQEncoder>
void compute_code(const ProductQuantizer& pq, const float* x, uint8_t* code) {
    std::vector<float> distances(pq.ksub);

    PQEncoder encoder(code, pq.nbits);

    for (size_t m = 0; m < pq.M; m++) {
        const float* xsub = x + m * pq.dsub;

        // 找到最近的质心索引
        uint64_t idxm = 0;
        if (pq.transposed_centroids.empty()) {
            // 常规版本
            idxm = fvec_L2sqr_ny_nearest(
                    distances.data(),
                    xsub,
                    pq.get_centroids(m, 0),
                    pq.dsub,
                    pq.ksub);
        } else {
            // 转置优化版本（更快）
            idxm = fvec_L2sqr_ny_nearest_y_transposed(
                    distances.data(),
                    xsub,
                    pq.transposed_centroids.data() + m * pq.ksub,
                    pq.centroids_sq_lengths.data() + m * pq.ksub,
                    pq.dsub,
                    pq.M * pq.ksub,
                    pq.ksub);
        }

        // 编码索引
        encoder.encode(idxm);
    }
}

void ProductQuantizer::compute_code(const float* x, uint8_t* code) const {
    switch (nbits) {
        case 8:
            faiss::compute_code<PQEncoder8>(*this, x, code);
            break;

        case 16:
            faiss::compute_code<PQEncoder16>(*this, x, code);
            break;

        default:
            faiss::compute_code<PQEncoderGeneric>(*this, x, code);
            break;
    }
}
```

**编码流程**：

```
输入: x ∈ R^128

步骤1: 分解
  x_0 = x[0:16], x_1 = x[16:32], ..., x_7 = x[112:128]

步骤2: 对每个子向量编码
  for m = 0 to 7:
    idxm = argmin_j ||x_m - c_{m,j}||²
    code[m] = idxm

输出: code[8字节] = [idx_0, idx_1, ..., idx_7]
```

### 3.4 编码效率优化

```cpp
// 注释中的性能讨论
// 编译器生成的代码 vs 手动优化

// 方案1: 使用std::vector + 条件判断（慢）
std::vector<float> distances_cached(N);
for (size_t i = 0; i < N; i++) {
    distances_cached[i] = compute_distance(x, y + i * d, d);
}
size_t min_distance = HUGE_VALF;
size_t idxm = 0;
for (size_t i = 0; i < N; i++) {
    const float distance = distances_cached[i];
    if (distance < min_distance) {
        min_distance = distance;
        idxm = i;
    }
}

// 方案2: 实时计算 + 向量化（快）
size_t idxm = fvec_L2sqr_ny_nearest(
    distances.data(),
    xsub,
    pq.get_centroids(m, 0),
    pq.dsub,
    pq.ksub);

// 为什么方案2更快？
// 1. 向量化距离计算
// 2. 编译器优化向量化循环
// 3. 减少内存访问（不用缓存整个数组）
```

---

## 4. 解码器实现

### 4.1 PQDecoderGeneric

```cpp
struct PQDecoderGeneric {
    const uint8_t* code;
    uint8_t offset;
    const int nbits;
    const uint64_t mask;  //掩码: (1 << nbits) - 1

    PQDecoderGeneric(const uint8_t* code, int nbits)
            : code(code), offset(offset), nbits(nbits),
              mask((uint64_t(1) << nbits) - 1) {
        reg = 0;
    }

    uint64_t decode() {
        // 从code中解码nbits位
        uint64_t c = (code[offset >> 3]) & mask;
        offset += nbits;
        return c;
    }
};
```

**解码流程**：

```
输入: code[8字节] = [idx_0, idx_1, ..., idx_7]

步骤1: 解码每个idx_m
  for m = 0 to 7:
    idx_m = decoder.decode()  // 解码8位

步骤2: 重构向量
  x_m = centroids[m][idx_m]  // 从质心表获取

步骤3: 拼接
  x = [x_0, x_1, ..., x_7]

输出: x ∈ R^128
```

### 4.2 decode函数

```cpp
template <class PQDecoder>
void decode(const ProductQuantizer& pq, const uint8_t* code, float* x) {
    PQDecoder decoder(code, pq.nbits);

    for (size_t m = 0; m < pq.M; m++) {
        uint64_t c = decoder.decode();
        memcpy(x + m * pq.dsub,
               pq.get_centroids(m, c),
               sizeof(float) * pq.dsub);
    }
}

void ProductQuantizer::decode(const uint8_t* code, float* x) const {
    switch (nbits) {
        case 8:
            faiss::decode<PQDecoder8>(*this, code, x);
            break;

        case 16:
            faiss::decode<PQDecoder16>(*this, code, x);
            break;

        default:
            faiss::decode<PQDecoderGeneric>(*this, code, x);
            break;
    }
}
```

---

## 5. 训练流程

### 5.1 train函数主流程

```cpp
void ProductQuantizer::train(size_t n, const float* x) {
    // 步骤1: 确定训练类型
    train_type_t final_train_type = train_type;

    // 检查超立方初始化条件
    if (train_type == Train_hypercube ||
        train_type == Train_hypercube_pca) {
        if (dsub < nbits) {
            final_train_type = Train_default;
        }
    }

    // 步骤2: 分配数据块
    std::unique_ptr<float[]> xslice(new float[n * dsub]);

    // 步骤3: 训练每个子量化器
    for (int m = 0; m < M; m++) {
        // 提取子向量
        for (int j = 0; j < n; j++) {
            memcpy(xslice.get() + j * dsub,
                   x + j * d + m * dsub,
                   dsub * sizeof(float));
        }

        // 训练子量化器m
        Clustering clus(dsub, ksub, cp);

        // 初始化质心
        switch (final_train_type) {
            case Train_hypercube:
                init_hypercube(...);
                break;
            case Train_hypercube_pca:
                init_hypercube_pca(...);
                break;
            case Train_hot_start:
                // 使用已有质心
                break;
            default:;
        }

        // k-means聚类
        IndexFlatL2 index(dsub);
        clus.train(n, xslice.get(), index);

        // 存储质心
        set_params(clus.centroids.data(), m);
    }
}
```

### 5.2 超立方初始化

```cpp
static void init_hypercube(
        int d,
        int nbits,
        int n,
        const float* x,
        float* centroids) {

    // 步骤1: 计算均值
    std::vector<float> mean(d);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            mean[j] += x[i * d + j];

    for (int j = 0; j < d; j++)
        mean[j] /= n;

    // 步骤2: 找最大均值分量
    float maxm = 0;
    for (int j = 0; j < d; j++) {
        if (fabs(mean[j]) > maxm)
            maxm = fabs(mean[j]);
    }

    // 步骤3: 初始化超立方体质心
    for (int i = 0; i < (1 << nbits); i++) {
        float* cent = centroids + i * d;

        // 为每个nbits位维度赋值
        for (int j = 0; j < nbits; j++)
            cent[j] = mean[j] + (((i >> j) & 1) ? 1 : -1) * maxm;

        // 剩余维度补0
        for (int j = nbits; j < d; j++)
            cent[j] = mean[j];
    }
}
```

**超立方体结构**：

```
d = 4, nbits = 2
ksub = 4个质心

质心分布:
(0) [μ0-α, μ1-α, μ2-α, μ3+α]
(1) [μ0+α, μ1-α, μ2+α, μ3-α]
(2) [μ0-α, μ1+α, μ2-α, μ3+α]
(3) [μ0+α, μ1+α, μ2+α, μ3+α]

其中: α = max(μ0, μ1, μ2, μ3)
```

### 5.3 超立方体+PCA初始化

```cpp
static void init_hypercube_pca(
        int d,
        int nbits,
        int n,
        const float* x,
        float* centroids) {

    // 步骤1: PCA降维
    PCAMatrix pca(d, nbits);
    pca.train(n, x);

    // 步骤2: 沿主成分方向初始化
    for (int i = 0; i < (1 << nbits); i++) {
        float* cent = centroids + i * d;

        // 从均值开始
        for (int j = 0; j < d; j++) {
            cent[j] = pca.mean[j];
            float f = 1.0;

            // 沿主成分方向分布
            for (int k = 0; k < nbits; k++)
                cent[j] += f * sqrt(pca.eigenvalues[k]) *
                        (((i >> k) & 1) ? 1 : -1) * pca.PCAMat[j + k * d];
        }
    }
}
```

---

## 6. 距离计算优化

### 6.1 compute_distance_table

```cpp
void ProductQuantizer::compute_distance_table(
        const float* x,
        float* dis_table) const {

    // dis_table: [M × ksub] 矩阵
    // dis_table[m][j] = ||x_m - c_{m,j}||²

    for (size_t m = 0; m < M; m++) {
        const float* x_sub = x + m * dsub;
        float* dt = dis_table + m * ksub;

        // 计算到所有质心的距离
        for (size_t j = 0; j < ksub; j++) {
            const float* c = get_centroids(m, j);

            // L2距离平方
            float dis = 0;
            for (size_t k = 0; k < dsub; k++) {
                float diff = x_sub[k] - c[k];
                dis += diff * diff;
            }

            dt[j] = dis;
        }
    }
}
```

### 6.2 search函数

```cpp
void ProductQuantizer::search(
        const float* x,
        size_t nx,
        const uint8_t* codes,
        const size_t ncodes,
        float_maxheap_array_t* res,
        bool init_finalize_heap) const {

    for (size_t i = 0; i < nx; i++) {
        res[i].heap = std::numeric_limits<float>::max();
        res[i].ids[0] = -1;

        // 计算距离表
        std::vector<float> dis_tables(M * ksub);
        compute_distance_tables(1, x + i * d, dis_tables.data());

        // 遍历所有编码
        for (size_t j = 0; j < ncodes; j++) {
            float dis = 0.0f;

            // 对每个子量化器累加距离
            for (size_t m = 0; m < M; m++) {
                uint64_t code = get_idx(codes, j, m);
                dis += dis_tables[m * ksub + code];
            }

            // 更新堆
            if (dis < res[i].heap) {
                heap_push_top<CMax<float, idx_t>>(k, res[i].heap, res[i].ids, dis, j);
            }
        }

        if (init_finalize_heap) {
            heap_reorder<CMax<float, idx_t>>(k, res[i].heap, res[i].ids);
        }
    }
}
```

---

## 7. 性能优化技巧

### 7.1 转置质心优化

```cpp
// 常规版本: centroids [M × ksub × dsub]
inline float* get_centroids(size_t m, size_t i) {
    return &centroids[(m * ksub + i) * dsub];
}

// 转置优化版本: transposed_centroids [dsub × M × ksub]
// 优势: 更好的缓存局部性

float distance_with_transposed(
        const float* x,
        const float* transposed_centroids,
        size_t m,
        size_t ksub,
        size_t dsub) {

    // 加载转置质心: 连续访问
    const float* centroids_m = transposed_centroids + m * ksub;

    for (size_t j = 0; j < ksub; j++) {
        float dis = 0;
        for (size_t k = 0; k < dsub; k++) {
            float diff = x[k] - centroids_m[j + k * dsub];
            dis += diff * diff;
        }
    }
    return dis;
}
```

**缓存性能对比**：

```
常规版本: centroids[m][j][k]
访问模式: 跨步长 -> 缓存未命中

转置版本: transposed_centroids[j][k][m]
访问模式: 连续 -> 缓存命中
```

### 7.2 距离表预计算

```cpp
// 查询时: 预计算距离表
void compute_distance_tables(
        size_t nx,
        const float* x,
        float* dis_tables) const {

    for (size_t i = 0; i < nx; i++) {
        compute_distance_table(x + i * d, dis_tables + i * M * ksub);
    }
}

// 搜索时: 直接查表
void search_with_tables(...) {
    // 避免重复计算距离
    // 只需要查表和累加
}
```

### 7.3 SDC表优化

```cpp
// Symmetric Distance Computation table
std::vector<float> sdc_table;

void compute_sdc_table() {
    // SDC: 用于快速计算量化码之间的距离
    // sdc_table[code1][code2] = ||c_{code1} - c_{code2}||²

    for (size_t i = 0; i < nlist; i++) {
        for (size_t j = 0; j < nlist; j++) {
            float dis = 0;
            for (size_t m = 0; m < M; m++) {
                float* c1 = get_centroids(m, i);
                float* c2 = get_centroids(m, j);
                for (size_t k = 0; k < dsub; k++) {
                    float diff = c1[k] - c2[k];
                    dis += diff * diff;
                }
            }
            sdc_table[i * nlist + j] = dis;
        }
    }
}
```

### 7.4 批量编码优化

```cpp
// 批量编码多个向量
void compute_codes(
        const float* x,
        uint8_t* codes,
        size_t n) const {

    // 块大小
    const size_t block_size = product_quantizer_compute_codes_bs;

    for (size_t i0 = 0; i0 < n; i0 += block_size) {
        size_t i1 = std::min(i0 + block_size, n);

        // 批量处理i1-i0个向量
        for (size_t i = i0; i < i1; i++) {
            compute_code(x + i * d, codes + i * code_size);
        }
    }
}
```

---

## 8. 总结

### 8.1 关键优化技术

1. **转置质心表**
   - 提高缓存局部性
   - SIMD友好访问模式

2. **距离表预计算**
   - 避免重复计算
   - 查表操作O(1)

3. **多种编码器**
   - PQEncoder8: 8位专用（最快）
   - PQEncoder16: 16位专用
   - PQEncoderGeneric: 通用（任意位数）

4. **训练优化**
   - 超立方体初始化
   - PCA增强初始化
   - 热启动模式

5. **批量处理**
   - 块编码
   - 块解码
   - 向量化距离计算

### 8.2 内存和计算复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 编码 | O(d) | O(d) |
| 解码 | O(d) | O(d) |
| 距离表计算 | O(M×ksub×d) | O(M×ksub) |
| 搜索(无表) | O(M×d) | O(1) |
| 搜索(有表) | O(M) | O(M×ksub) |

### 8.3 压缩比分析

| 参数 | 原始大小 | PQ后大小 | 压缩比 |
|------|---------|----------|--------|
| d=128, M=8, nbits=8 | 512字节 | 8字节 | 64x |
| d=256, M=16, nbits=8 | 1024字节 | 16字节 | 64x |
| d=384, M=32, nbits=8 | 1536字节 | 32字节 | 48x |

### 8.4 实际应用建议

1. **选择M和nbits**
   - M: 根据维度d，通常8-32
   - nbits: 8位（256质心）最常用

2. **使用转置质心**
   - 启用`transposed_centroids`
   - 提高搜索性能10-20%

3. **预计算距离表**
   - 适用于批量查询
   - 减少重复计算

---

## 附录A:完整示例

```cpp
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/MetricType.h>

using namespace faiss;

void example_pq() {
    int d = 128;          // 维度
    int ntotal = 1000000; // 向量数量
    int M = 8;            // 子量化器数量
    int nbits = 8;         // 每个子量化器8位
    int nlist = 100;       // IVF倒排表数量
    int nprobe = 10;       // 搜索的倒排表数

    // 训练集
    float* xb = new float[ntotal * d];
    // ... 填充数据

    // 创建IVFPQ索引
    IndexIVFPQ index(d, M, nbits);
    index.quantizer_trains_alone = true;
    index.nlist = nlist;
    index.nprobe = nprobe;

    // 训练
    index.train(ntotal, xb);

    // 添加向量
    index.add(ntotal, xb);

    // 搜索
    int nq = 100;
    int k = 100;
    float* xq = new float[nq * d];
    // ... 填充查询向量

    float* distances = new float[nq * k];
    int64_t* labels = new int64_t[nq * k];

    index.search(nq, xq, k, distances, labels);

    // 输出结果
    for (int i = 0; i < nq; i++) {
        printf("Query %d:\\n", i);
        for (int j = 0; j < k; j++) {
            printf("  %d: id=%ld distance=%g\\n",
                   j, labels[i * k + j], distances[i * k + j]);
        }
    }

    delete[] xb;
    delete[] xq;
    delete[] distances;
    delete[] labels;
}

int main() {
    example_pq();
    return 0;
}
```

## 附录B:相关源文件

- `faiss/impl/ProductQuantizer.h` - PQ接口定义
- `faiss/impl/ProductQuantizer.cpp` - PQ实现
- `faiss/impl/ScalarQuantizer.h` - 标量量化器
- `faiss/impl/Quantizer.h` - 量化器基类
- `faiss/utils/distances.h` - 距离计算

## 附录C:编码器实现细节

```cpp
// PQEncoderGeneric的encode实现
void PQEncoderGeneric::encode(uint64_t x) {
    code += offset >> 3;
    code[offset & 7] = x & 0xff;
    offset += nbits;
}

// PQEncoder8的encode实现
void PQEncoder8::encode(uint64_t x) {
    *code++ = (uint8_t)x;
}

// PQEncoder16的encode实现
void PQEncoder16::encode(uint64_t x) {
    *code++ = (uint16_t)x;
}
```

## 附录D:性能基准测试

```cpp
#include <benchmark/benchmark.h>

static void BM_PQ_Train(benchmark::State& state) {
    int d = 128;
    int M = 8;
    int nbits = 8;
    size_t n = state.range(0);

    std::vector<float> x(n * d);
    // ... 初始化

    ProductQuantizer pq(d, M, nbits);
    pq.verbose = true;

    for (auto _ : state) {
        pq.train(n, x.data());
        benchmark::DoNotOptimize(pq.is_trained);
    }
}

static void BM_PQ_Encode(benchmark::State& state) {
    int d = 128;
    int M = 8;
    int nbits = 8;
    size_t n = state.range(0);

    std::vector<float> x(n * d);
    uint8_t* codes = new uint8_t[n * ((nbits * M + 7) / 8)];

    ProductQuantizer pq(d, M, nbits);
    pq.train(n, x.data());

    for (auto _ : state) {
        pq.compute_codes(x.data(), codes, n);
        benchmark::DoNotOptimize(codes);
    }

    delete[] codes;
}

BENCHMARK(BM_PQ_Train)->Range(10000, 1000000);
BENCHMARK(BM_PQ_Encode)->Range(1000, 1000000);
BENCHMARK_MAIN();
```
