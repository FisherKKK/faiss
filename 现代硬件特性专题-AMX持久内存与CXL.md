# 现代硬件特性专题 - AMX、持久内存与 CXL

## 课程简介

本课程深入讲解现代硬件的最新特性，包括 Intel AMX（Advanced Matrix Extensions）、持久内存（Persistent Memory）和 CXL（Compute Express Link），并探讨如何将这些技术应用于向量搜索和 AI 推理加速。

**适合人群**：
- 需要极致性能的系统架构师
- AI/ML 推理优化工程师
- 向量数据库开发者
- 性能敏感应用的开发者

**前置知识**：
- 已完成《SIMD 深度实践课程》
- 了解计算机体系结构基础
- 熟悉 C++ 和内存管理

---

## 第一部分：Intel AMX (Advanced Matrix Extensions)

### 1.1 AMX 概述

**什么是 AMX？**

Intel AMX 是在 Sapphire Rapids（第四代 Xeon）中引入的新指令集架构，专门用于加速矩阵运算，特别是 AI 和机器学习工作负载。

**核心特性**：
- **Tile 寄存器**：8 个 1KB 的 Tile 寄存器（TMM0-TMM7）
- **二维寄存器**：每个 Tile 可以存储矩阵（如 16x16 bf16）
- **矩阵乘法加速**：TMUL 指令可以在单周期内执行 16x16 矩阵乘法
- **灵活配置**：支持多种数据类型（BF16, INT8, FP16, FP32）

**性能提升**：
- 矩阵乘法：比 AVX-512 快 4-8 倍
- AI 推理：INT8 推理性能提升 2-3 倍
- 向量搜索：批量距离计算提升 3-5 倍

### 1.2 AMX 架构

```
传统 SIMD 架构:
┌─────────┐  ┌─────────┐  ┌─────────┐
│ ZMM0    │  │ ZMM1    │  │ ZMM2    │  512-bit
│ 16x fp32│  │ 16x fp32│  │ 16x fp32│
└─────────┘  └─────────┘  └─────────┘

AMX Tile 架构:
┌──────────────────────────────────────┐
│ TMM0 (1KB)                           │
│   ┌────┬────┬────┬────┬────┬────┐   │
│   │ 0  │ 1  │ 2  │... │15  │16  │   │  16x16 bf16
│   ├────┼────┼────┼────┼────┼────┤   │
│   │ 1  │    │    │    │    │    │   │
│   ├────┼────┼────┼────┼────┼────┤   │
│   │... │    │    │    │    │    │   │
│   └────┴────┴────┴────┴────┴────┘   │
└──────────────────────────────────────┘

可配置的 Tile 形状:
- 16x16 BF16 (256 元素)
- 32x32 INT8 (1024 元素)
- 8x32 FP32 (256 元素)
```

### 1.3 AMX 编程基础

#### 1.3.1 检测 AMX 支持

```cpp
#include <cpuid.h>
#include <stdio.h>

bool has_amx() {
    uint32_t eax, ebx, ecx, edx;

    // 检查 AMX-BF16 (BFloat16)
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    bool amx_bf16 = (edx & (1 << 22)) != 0;

    // 检查 AMX-INT8
    bool amx_int8 = (edx & (1 << 23)) != 0;

    // 检查 AMX-FP16 (需要额外支持)
    bool amx_fp16 = (ecx & (1 << 23)) != 0;

    printf("AMX Support:\n");
    printf("  AMX-BF16: %s\n", amx_bf16 ? "YES" : "NO");
    printf("  AMX-INT8: %s\n", amx_int8 ? "YES" : "NO");
    printf("  AMX-FP16: %s\n", amx_fp16 ? "YES" : "NO");

    return amx_bf16 || amx_int8 || amx_fp16;
}

int main() {
    if (!has_amx()) {
        printf("AMX not supported on this CPU\n");
        return 1;
    }
    return 0;
}
```

#### 1.3.2 配置 Tile 寄存器

```cpp
#include <immintrin.h>
#include <stdalign.h>

// 配置 Tile 形状
void configure_tiles() {
    // 配置 TMM0 为 16x16 BF16 矩阵
    // 每个 Tile 由行和列配置
    _tile_loadconfig(
        (16 << 16) | 16  // rows=16, cols=16
    );
}

// 分配对齐的 Tile 内存
alignas(64) float tile_buffer[16 * 16];
```

#### 1.3.3 基础矩阵运算

```cpp
// 矩阵乘法：C = A * B
// TMM1 = TMM0 * TMM2
void matrix_multiply_amx(
        const float* A,  // [M, K]
        const float* B,  // [K, N]
        float* C,        // [M, N]
        int M, int K, int N) {

    // 配置 Tile 形状
    // TMM0: M x K (假设 16x16)
    // TMM1: K x N (假设 16x16)
    // TMM2: M x N (结果)
    _tile_loadconfig((16 << 16) | 16);

    // 加载矩阵 A 到 TMM0
    _tile_loadd(
        0,              // TMM0
        A,              // 源地址
        16 * 4          // stride (字节)
    );

    // 加载矩阵 B 到 TMM1
    _tile_loadd(
        1,              // TMM1
        B,
        16 * 4
    );

    // 初始化结果矩阵 C (TMM3) 为零
    _tile_zero(3);

    // 矩阵乘法：TMM3 += TMM0 * TMM1
    _tile_dpbus_16x16(
        3,  // 目标 Tile (累加器)
        0,  // 源 Tile 1
        1   // 源 Tile 2
    );

    // 存储结果
    _tile_stored(
        3,              // TMM3
        C,
        16 * 4
    );

    // 释放 Tile 寄存器
    _tile_release();
}
```

### 1.4 AMX 向量搜索优化

使用 AMX 加速批量向量距离计算。

#### 1.4.1 批量点积计算

```cpp
// 使用 AMX 计算批量点积
// 查询向量 Q [d] vs 数据库矩阵 D [d, n]
void batch_dot_product_amx(
        const float* Q,      // [d] - 查询向量
        const float* D,      // [d, n] - 数据库向量（转置）
        float* distances,    // [n] - 输出距离
        int d,               // 向量维度
        int n) {             // 向量数量

    // 假设 d 和 n 都是 16 的倍数
    const int TILE_SIZE = 16;

    // 配置 Tile
    _tile_loadconfig((TILE_SIZE << 16) | TILE_SIZE);

    // 1. 将查询向量加载到 Tile 行
    __attribute__((aligned(64))) float Q_tile[TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < TILE_SIZE; i++) {
        for (int j = 0; j < TILE_SIZE; j++) {
            Q_tile[i * TILE_SIZE + j] = (j < d) ? Q[j] : 0;
        }
    }

    _tile_loadd(0, Q_tile, TILE_SIZE * 4);

    // 初始化结果 Tile
    _tile_zero(3);

    // 2. 对每个块进行处理
    for (int col_start = 0; col_start < n; col_start += TILE_SIZE) {
        // 加载数据库向量块
        __attribute__((aligned(64))) float D_tile[TILE_SIZE * TILE_SIZE];

        for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                int row = i;
                int col = col_start + j;
                if (row < d && col < n) {
                    D_tile[i * TILE_SIZE + j] = D[row * n + col];
                } else {
                    D_tile[i * TILE_SIZE + j] = 0;
                }
            }
        }

        _tile_loadd(1, D_tile, TILE_SIZE * 4);

        // 矩阵乘法
        _tile_dpbus_16x16(3, 0, 1);

        // 存储结果
        __attribute__((aligned(64))) float result_tile[TILE_SIZE * TILE_SIZE];
        _tile_stored(3, result_tile, TILE_SIZE * 4);

        // 提取需要的部分
        for (int j = 0; j < TILE_SIZE && col_start + j < n; j++) {
            distances[col_start + j] = result_tile[0 * TILE_SIZE + j];
        }

        // 重置累加器（如果需要多块累加）
        _tile_zero(3);
    }

    _tile_release();
}
```

#### 1.4.2 向量距离矩阵计算

```cpp
// 计算距离矩阵：N 个查询 vs M 个数据库向量
// 输出：[N, M] 距离矩阵
void compute_distance_matrix_amx(
        const float* queries,      // [N, d]
        const float* database,     // [M, d]
        float* dist_matrix,        // [N, M]
        int N, int M, int d) {

    const int TILE_SIZE = 16;

    // 配置 Tile
    _tile_loadconfig((TILE_SIZE << 16) | TILE_SIZE);

    // 分块处理
    for (int q_block = 0; q_block < N; q_block += TILE_SIZE) {
        for (int d_block = 0; d_block < M; d_block += TILE_SIZE) {
            // 加载查询块
            __attribute__((aligned(64))) float Q_tile[TILE_SIZE * TILE_SIZE];
            for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
                    int q_id = q_block + i;
                    int dim = j;
                    if (q_id < N && dim < d) {
                        Q_tile[i * TILE_SIZE + j] = queries[q_id * d + dim];
                    } else {
                        Q_tile[i * TILE_SIZE + j] = 0;
                    }
                }
            }
            _tile_loadd(0, Q_tile, TILE_SIZE * 4);

            // 加载数据库块（转置）
            __attribute__((aligned(64))) float D_tile[TILE_SIZE * TILE_SIZE];
            for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
                    int dim = i;
                    int db_id = d_block + j;
                    if (dim < d && db_id < M) {
                        D_tile[i * TILE_SIZE + j] = database[db_id * d + dim];
                    } else {
                        D_tile[i * TILE_SIZE + j] = 0;
                    }
                }
            }
            _tile_loadd(1, D_tile, TILE_SIZE * 4);

            // 计算点积
            _tile_zero(3);
            _tile_dpbus_16x16(3, 0, 1);

            // 获取结果并计算 L2 距离
            __attribute__((aligned(64))) float dot_result[TILE_SIZE * TILE_SIZE];
            _tile_stored(3, dot_result, TILE_SIZE * 4);

            // 计算完整距离：||Q||² + ||D||² - 2*<Q,D>
            for (int i = 0; i < TILE_SIZE && q_block + i < N; i++) {
                for (int j = 0; j < TILE_SIZE && d_block + j < M; j++) {
                    float q_norm = 0;  // 预计算
                    float d_norm = 0;  // 预计算
                    float dot = dot_result[i * TILE_SIZE + j];
                    dist_matrix[(q_block + i) * M + (d_block + j)] =
                        q_norm + d_norm - 2 * dot;
                }
            }
        }
    }

    _tile_release();
}
```

### 1.5 AMX 性能优化技巧

#### 1.5.1 数据预取和流水线

```cpp
// 流水线化处理多个块
void pipelined_amx_compute(
        const float* input_buffer,
        float* output_buffer,
        int num_blocks) {

    const int TILE_SIZE = 16;
    _tile_loadconfig((TILE_SIZE << 16) | TILE_SIZE);

    // 使用多个 Tile 寄存器实现流水线
    for (int i = 0; i < num_blocks; i += 4) {
        // Stage 1: 加载块 0
        if (i < num_blocks) {
            _tile_loadd(0, &input_buffer[i * 256], TILE_SIZE * 4);
        }

        // Stage 2: 加载块 1，处理块 0
        if (i + 1 < num_blocks) {
            _tile_loadd(1, &input_buffer[(i + 1) * 256], TILE_SIZE * 4);
        }
        _tile_dpbus_16x16(3, 0, 2);

        // Stage 3: 存储块 0，处理块 1
        if (i + 2 < num_blocks) {
            _tile_loadd(2, &input_buffer[(i + 2) * 256], TILE_SIZE * 4);
        }
        _tile_stored(3, &output_buffer[i * 256], TILE_SIZE * 4);
        _tile_dpbus_16x16(4, 1, 2);

        // Stage 4: 存储块 1，处理块 2
        if (i + 3 < num_blocks) {
            _tile_loadd(0, &input_buffer[(i + 3) * 256], TILE_SIZE * 4);
        }
        _tile_stored(4, &output_buffer[(i + 1) * 256], TILE_SIZE * 4);
        _tile_dpbus_16x16(3, 2, 0);
    }

    _tile_release();
}
```

#### 1.5.2 混合精度计算

```cpp
// 使用 BF16 加速，FP32 累加
void mixed_precision_amx(
        const float* A,
        const float* B,
        float* C,
        int M, int K, int N) {

    // 转换为 BF16
    __attribute__((aligned(64))) uint16_t A_bf16[16 * 16];
    __attribute__((aligned(64))) uint16_t B_bf16[16 * 16];

    for (int i = 0; i < 16 * 16; i++) {
        A_bf16[i] = float_to_bf16(A[i]);
        B_bf16[i] = float_to_bf16(B[i]);
    }

    _tile_loadconfig((16 << 16) | 16);

    // 加载 BF16 数据
    _tile_loaddt1(0, A_bf16, 16 * 2);
    _tile_loaddt1(1, B_bf16, 16 * 2);

    // 初始化 FP32 累加器
    _tile_zero(3);

    // BF16 矩阵乘法，FP32 累加
    _tile_dpbf16ps(3, 0, 1);

    // 存储结果
    _tile_stored(3, C, 16 * 4);

    _tile_release();
}

// FP32 到 BF16 转换
inline uint16_t float_to_bf16(float f) {
    uint32_t u = *((uint32_t*)&f);
    // BF16: [sign(1) | exponent(8) | mantissa(7)]
    // FP32: [sign(1) | exponent(8) | mantissa(23)]
    // 舍入：保留 FP32 的高 16 位
    return (u + (1 << 15) & 0xFFFF0000) >> 16;
}
```

---

## 第二部分：持久内存 (Persistent Memory)

### 2.1 持久内存概述

**什么是持久内存？**

Intel Optane DC Persistent Memory（DCPM）是一种新型的内存技术，结合了 DRAM 的速度和存储的持久性。

**关键特性**：
- **容量大**：单条 512GB，系统可达 6TB
- **非易失**：断电后数据不丢失
- **可字节寻址**：支持 load/store 指令
- **延迟高于 DRAM**：约 350ns（DRAM ~100ns）
- **带宽接近 DRAM**：约 6-10 GB/s

**应用场景**：
- 大规模向量数据库
- 内存数据库
- AI 模型缓存
- 检查点/恢复

### 2.2 持久内存编程模式

#### 2.2.1 内存模式 (Memory Mode)

```cpp
// 内存模式：PMem 用作大容量内存（对应用透明）
// 无需修改代码，操作系统自动处理

#include <stdlib.h>
#include <stdio.h>

void memory_mode_demo() {
    // 大规模向量数据库
    size_t num_vectors = 1000000000;  // 10 亿向量
    size_t dim = 128;

    // 直接分配，操作系统会自动使用 PMem
    float* database = (float*)aligned_alloc(
        64,
        num_vectors * dim * sizeof(float)
    );

    if (!database) {
        perror("aligned_alloc");
        return;
    }

    printf("Allocated %.2f GB for vector database\n",
           num_vectors * dim * sizeof(float) / 1e9);

    // 使用与普通内存相同
    for (size_t i = 0; i < 100; i++) {
        database[i * dim] = 1.0f;
    }

    free(database);
}
```

#### 2.2.2 App Direct 模式

```cpp
// App Direct 模式：显式使用持久内存
// 需要 libpmem 库

#include <libpmem.h>
#include <cstring>

#define PMEM_FILE "/mnt/pmem/vector_db"
#define PMEM_SIZE (1024UL * 1024UL * 1024UL * 100UL)  // 100GB

struct VectorDatabase {
    size_t num_vectors;
    size_t dimension;
    float data[];  // 柔性数组
};

void app_direct_demo() {
    // 1. 创建或打开持久内存文件
    int is_pmem;
    size_t mapped_len;
    void* pmemaddr = pmem_map_file(
        PMEM_FILE,     // 文件路径
        PMEM_SIZE,      // 长度
        PMEM_FILE_CREATE,  // 创建标志
        0666,           // 权限
        &mapped_len,
        &is_pmem
    );

    if (!pmemaddr) {
        perror("pmem_map_file");
        return;
    }

    if (!is_pmem) {
        printf("Warning: %s is not on persistent memory\n", PMEM_FILE);
    }

    // 2. 使用持久内存
    VectorDatabase* db = (VectorDatabase*)pmemaddr;
    db->num_vectors = 10000000;
    db->dimension = 128;

    // 3. 持久化数据
    // 方法 1: pmem_persist (确保数据写入持久内存)
    pmem_persist(db, sizeof(VectorDatabase));

    // 4. 写入向量数据
    for (size_t i = 0; i < 1000; i++) {
        for (size_t j = 0; j < db->dimension; j++) {
            db->data[i * db->dimension + j] = (float)(i + j);
        }
    }

    // 5. 持久化更新
    pmem_persist(db->data, 1000 * db->dimension * sizeof(float));

    printf("Stored %zu vectors in persistent memory\n",
           db->num_vectors);

    // 6. 使用数据（与普通内存相同）
    float query[128];
    for (size_t j = 0; j < 128; j++) {
        query[j] = (float)j / 128.0f;
    }

    // 计算距离
    for (size_t i = 0; i < 100; i++) {
        float dist = 0;
        for (size_t j = 0; j < 128; j++) {
            float diff = query[j] - db->data[i * 128 + j];
            dist += diff * diff;
        }
        printf("Distance to vector %zu: %f\n", i, dist);
    }

    // 7. 清理
    pmem_unmap(pmemaddr, mapped_len);
}
```

### 2.3 持久内存优化向量搜索

#### 2.3.1 大规模向量索引

```cpp
#include <libpmem.h>
#include <immintrin.h>

// 持久内存上的向量索引
struct PMemVectorIndex {
    size_t num_vectors;
    size_t dimension;
    size_t capacity;

    // 元数据
    size_t* ids;        // 向量 ID
    float* norms;       // 预计算的范数

    // 向量数据（列主序存储，优化搜索）
    float* vectors;     // [dimension, num_vectors]

    // 持久化方法
    void persist() {
        pmem_persist(this, sizeof(PMemVectorIndex));
        pmem_persist(ids, num_vectors * sizeof(size_t));
        pmem_persist(norms, num_vectors * sizeof(float));
        pmem_persist(vectors, num_vectors * dimension * sizeof(float));
    }

    // 添加向量
    void add_vector(size_t id, const float* vector) {
        if (num_vectors >= capacity) {
            // 扩容（需要重新分配持久内存）
            return;
        }

        // 存储 ID
        ids[num_vectors] = id;

        // 计算并存储范数
        float norm = 0;
        for (size_t i = 0; i < dimension; i++) {
            norm += vector[i] * vector[i];
            // 列主序存储
            vectors[i * capacity + num_vectors] = vector[i];
        }
        norms[num_vectors] = norm;

        num_vectors++;
    }

    // 搜索（SIMD 优化）
    void search(
            const float* query,
            size_t k,
            float* distances,
            size_t* labels) {

        // 预计算查询范数
        float query_norm = 0;
        for (size_t i = 0; i < dimension; i++) {
            query_norm += query[i] * query[i];
        }

        __m256 qnorm_vec = _mm256_set1_ps(query_norm);

        // 批量处理（每次 8 个向量）
        size_t i = 0;
        for (; i + 7 < num_vectors; i += 8) {
            __m256 dot_products = _mm256_setzero_ps();
            __m256 db_norms = _mm256_loadu_ps(&norms[i]);

            // 计算点积
            for (size_t d = 0; d < dimension; d++) {
                __m256 q_d = _mm256_set1_ps(query[d]);
                __m256 v_d = _mm256_loadu_ps(&vectors[d * capacity + i]);

                dot_products = _mm256_fmadd_ps(q_d, v_d, dot_products);
            }

            // L2 距离：||q||² + ||v||² - 2*<q,v>
            __m256 distances_vec = _mm256_add_ps(qnorm_vec, db_norms);
            __m256 two_dot = _mm256_mul_ps(dot_products, _mm256_set1_ps(2.0f));
            distances_vec = _mm256_sub_ps(distances_vec, two_dot);

            _mm256_storeu_ps(&distances[i], distances_vec);
        }

        // 处理剩余向量
        for (; i < num_vectors; i++) {
            float dot = 0;
            for (size_t d = 0; d < dimension; d++) {
                dot += query[d] * vectors[d * capacity + i];
            }
            distances[i] = query_norm + norms[i] - 2 * dot;
            labels[i] = ids[i];
        }

        // Top-K 选择...
    }
};

// 创建持久内存索引
PMemVectorIndex* create_pmem_index(
        const char* path,
        size_t dimension,
        size_t capacity) {

    size_t file_size = sizeof(PMemVectorIndex) +
                       capacity * (sizeof(size_t) + sizeof(float)) +
                       capacity * dimension * sizeof(float);

    int is_pmem;
    size_t mapped_len;
    void* pmemaddr = pmem_map_file(
        path,
        file_size,
        PMEM_FILE_CREATE,
        0666,
        &mapped_len,
        &is_pmem
    );

    if (!pmemaddr) {
        perror("pmem_map_file");
        return nullptr;
    }

    PMemVectorIndex* index = (PMemVectorIndex*)pmemaddr;
    index->dimension = dimension;
    index->capacity = capacity;
    index->num_vectors = 0;

    // 设置指针
    char* base = (char*)pmemaddr + sizeof(PMemVectorIndex);
    index->ids = (size_t*)base;
    index->norms = (float*)(base + capacity * sizeof(size_t));
    index->vectors = (float*)(base + capacity * (sizeof(size_t) + sizeof(float)));

    // 持久化结构
    index->persist();

    return index;
}
```

### 2.4 持久内存性能优化

#### 2.4.1 减少持久化次数

```cpp
// 优化：批量更新后统一持久化
void batch_add_vectors_optimized(
        PMemVectorIndex* index,
        const size_t* ids,
        const float* vectors,
        size_t count) {

    // 1. 使用临时缓冲区批量更新
    float* temp_buffer = (float*)aligned_alloc(
        64,
        count * index->dimension * sizeof(float)
    );

    // 2. 写入临时缓冲区
    for (size_t i = 0; i < count; i++) {
        index->ids[index->num_vectors + i] = ids[i];

        float norm = 0;
        for (size_t d = 0; d < index->dimension; d++) {
            float val = vectors[i * index->dimension + d];
            norm += val * val;
            temp_buffer[i * index->dimension + d] = val;
        }
        index->norms[index->num_vectors + i] = norm;
    }

    // 3. 一次性拷贝到持久内存
    memcpy(
        &index->vectors[index->num_vectors * index->dimension],
        temp_buffer,
        count * index->dimension * sizeof(float)
    );

    // 4. 只持久化一次
    index->num_vectors += count;
    index->persist();

    free(temp_buffer);
}
```

#### 2.4.2 使用非临时存储

```cpp
// 对于大规模写入，使用非临时存储绕过缓存
void write_large_dataset_pmem(
        PMemVectorIndex* index,
        const float* data,
        size_t count) {

    // 使用 movnti 指令（非临时存储）
    for (size_t i = 0; i < count; i++) {
        _mm256_stream_ps(
            &index->vectors[i * index->dimension],
            _mm256_loadu_ps(&data[i * index->dimension])
        );
    }

    // 内存屏障 + 持久化
    _mm_sfence();
    pmem_persist(index->vectors, count * index->dimension * sizeof(float));
}
```

---

## 第三部分：CXL (Compute Express Link)

### 3.1 CXL 概述

**什么是 CXL？**

CXL (Compute Express Link) 是新一代高速互连标准，实现 CPU 和加速器/内存之间的高带宽、低延迟通信。

**关键特性**：
- **高带宽**：高达 64 GT/s
- **低延迟**：与内存访问相当
- **内存一致性**：支持缓存一致性协议
- **内存池化**：多个 CPU 共享内存池

**三种协议**：
1. **CXL.io**：基于 PCIe，用于 I/O
2. **CXL.cache**：设备缓存 CPU 内存
3. **CXL.mem**：CPU 访问设备内存

### 3.2 CXL 应用场景

#### 3.2.1 内存池化

```cpp
// CXL 内存池：多个节点共享大内存
// 虚拟代码（需要特定硬件支持）

struct CXLMemoryPool {
    void* base_addr;
    size_t size;
    int node_id;

    // 分配 CXL 内存
    void* allocate(size_t size, size_t align) {
        // CXL 内存分配（通过特殊驱动）
        return cxl_alloc(size, align, node_id);
    }

    // 释放 CXL 内存
    void deallocate(void* ptr, size_t size) {
        cxl_free(ptr, size);
    }

    // 获取本地副本（NUMA 优化）
    void* get_local_copy(void* cxl_ptr, size_t size) {
        void* local = aligned_alloc(64, size);
        memcpy(local, cxl_ptr, size);  // 拉取到本地
        return local;
    }
};

// 使用 CXL 内存池扩展向量数据库
class CXLVectorDatabase {
    CXLMemoryPool pool;
    size_t local_capacity;
    size_t cxl_capacity;

public:
    CXLVectorDatabase() {
        // 本地内存：热数据
        local_capacity = 10000000;  // 1000 万向量

        // CXL 内存：冷数据
        cxl_capacity = 1000000000;  // 10 亿向量
    }

    void add_vector(const float* vector, bool is_hot) {
        if (is_hot) {
            // 存储到本地内存
            store_local(vector);
        } else {
            // 存储到 CXL 内存池
            void* cxl_ptr = pool.allocate(d * sizeof(float), 64);
            memcpy(cxl_ptr, vector, d * sizeof(float));
        }
    }

    float* get_vector(size_t id) {
        // 先检查本地缓存
        if (is_in_local_cache(id)) {
            return get_from_local(id);
        }

        // 从 CXL 内存拉取
        void* cxl_ptr = get_cxl_pointer(id);
        float* local_copy = pool.get_local_copy(cxl_ptr, d * sizeof(float));
        cache_local(id, local_copy);

        return local_copy;
    }
};
```

### 3.3 CXL 优化策略

#### 3.3.1 数据分层

```cpp
// 三层数据分层：本地 DRAM -> CXL 内存 -> SSD

enum class DataTier {
    HOT,      // 本地 DRAM（最快，最小）
    WARM,     // CXL 内存（中等速度，中等容量）
    COLD      // SSD（最慢，最大容量）
};

struct TieredVectorIndex {
    // 热数据层：本地内存
    float* hot_vectors;
    size_t* hot_ids;
    size_t hot_capacity;
    size_t hot_size;

    // 温数据层：CXL 内存
    float* warm_vectors;
    size_t* warm_ids;
    size_t warm_capacity;
    size_t warm_size;

    // 冷数据层：SSD
    int cold_fd;
    size_t cold_size;

    void search(const float* query, int k, float* distances, size_t* ids) {
        // 1. 搜索热数据（最快）
        int found = 0;
        search_hot(query, k - found, distances + found, ids + found, &found);

        if (found >= k) return;  // 已找到足够多的结果

        // 2. 搜索温数据（CXL，稍慢）
        search_warm(query, k - found, distances + found, ids + found, &found);

        if (found >= k) return;

        // 3. 搜索冷数据（SSD，最慢）
        search_cold(query, k - found, distances + found, ids + found, &found);
    }

    void promote_tier(size_t id) {
        // 将数据从冷层提升到热层
        if (is_cold(id)) {
            float* vec = load_from_cold(id);
            add_to_hot(id, vec);
            remove_from_cold(id);
        }
    }

    void demote_tier(size_t id) {
        // 将数据从热层降级到温层
        if (is_hot(id)) {
            float* vec = hot_vectors[id];
            add_to_warm(id, vec);
            remove_from_hot(id);
        }
    }
};
```

---

## 第四部分：综合案例

### 4.1 使用 AMX + PMem 的大规模向量搜索

```cpp
#include <immintrin.h>
#include <libpmem.h>

class UltraFastVectorIndex {
    // PMem 上的持久化索引
    struct {
        size_t num_vectors;
        size_t dimension;
        float* vectors;       // PMem: [d, n] 列主序
        float* norms;         // PMem: [n]
        size_t* ids;          // PMem: [n]
    } pmem_index;

    // 本地缓存（热查询）
    float* query_cache;
    size_t cache_size;

public:
    // 初始化
    void init(const char* pmem_path, size_t d, size_t capacity) {
        // 映射持久内存
        size_t file_size = capacity * d * sizeof(float) +
                          capacity * (sizeof(float) + sizeof(size_t));

        int is_pmem;
        size_t mapped_len;
        void* pmem = pmem_map_file(
            pmem_path, file_size,
            PMEM_FILE_CREATE, 0666,
            &mapped_len, &is_pmem
        );

        pmem_index.num_vectors = 0;
        pmem_index.dimension = d;
        pmem_index.vectors = (float*)pmem;
        pmem_index.norms = (float*)(pmem + capacity * d * sizeof(float));
        pmem_index.ids = (size_t*)(pmem + capacity * d * sizeof(float) +
                                    capacity * sizeof(float));

        // 分配本地缓存
        query_cache = (float*)aligned_alloc(64, d * sizeof(float));
    }

    // 添加向量
    void add_vector(size_t id, const float* vector) {
        // 写入持久内存（列主序）
        size_t n = pmem_index.num_vectors;
        for (size_t i = 0; i < pmem_index.dimension; i++) {
            pmem_index.vectors[i * pmem_index.capacity + n] = vector[i];
        }

        // 计算范数
        float norm = 0;
        for (size_t i = 0; i < pmem_index.dimension; i++) {
            norm += vector[i] * vector[i];
        }
        pmem_index.norms[n] = norm;
        pmem_index.ids[n] = id;

        pmem_index.num_vectors++;

        // 定期持久化
        if (n % 10000 == 0) {
            pmem_persist(&pmem_index, sizeof(pmem_index));
        }
    }

    // 使用 AMX 加速的批量搜索
    void search_amx(
            const float* query,
            size_t k,
            float* distances_out,
            size_t* ids_out) {

        size_t n = pmem_index.num_vectors;
        size_t d = pmem_index.dimension;

        // 配置 AMX Tile (16x16)
        _tile_loadconfig((16 << 16) | 16);

        // 计算查询范数
        float query_norm = 0;
        for (size_t i = 0; i < d; i++) {
            query_norm += query[i] * query[i];
        }
        __m512 qnorm_vec = _mm512_set1_ps(query_norm);

        // 批量处理（每次 16 个向量）
        for (size_t base = 0; base < n; base += 16) {
            size_t batch_size = (base + 16 <= n) ? 16 : (n - base);

            // 加载 16 个向量到 Tile
            __attribute__((aligned(64))) float batch_vectors[16 * 16];
            __attribute__((aligned(64))) float batch_norms[16];

            for (size_t i = 0; i < batch_size; i++) {
                batch_norms[i] = pmem_index.norms[base + i];
                for (size_t j = 0; j < d; j++) {
                    batch_vectors[i * 16 + j] =
                        pmem_index.vectors[j * pmem_index.capacity + base + i];
                }
            }

            // AMX 矩阵乘法
            _tile_loadd(0, batch_vectors, 16 * 4);
            _tile_zero(3);

            // 查询向量为 Tile
            __attribute__((aligned(64))) float query_tile[16 * 16];
            for (size_t i = 0; i < 16; i++) {
                for (size_t j = 0; j < 16; j++) {
                    query_tile[i * 16 + j] = (j < d) ? query[j] : 0;
                }
            }
            _tile_loadd(1, query_tile, 16 * 4);

            // 计算点积
            _tile_dpbus_16x16(3, 0, 1);

            // 获取结果
            __attribute__((aligned(64))) float results[16 * 16];
            _tile_stored(3, results, 16 * 4);

            // 计算距离
            for (size_t i = 0; i < batch_size; i++) {
                float dot = results[0];  // Tile 的第一行
                float norm = batch_norms[i];
                float dist = query_norm + norm - 2 * dot;

                distances_out[base + i] = dist;
                ids_out[base + i] = pmem_index.ids[base + i];
            }
        }

        _tile_release();

        // Top-K 选择（使用堆）
        topk_select(distances_out, ids_out, n, k);
    }

private:
    void topk_select(float* dists, size_t* ids, size_t n, size_t k) {
        // 简单的 Top-K 实现
        std::partial_sort(
            ids, ids + k, ids + n,
            [dists](size_t a, size_t b) {
                return dists[a] < dists[b];
            }
        );
    }
};
```

### 4.2 性能对比

```cpp
// 不同配置的性能对比

void benchmark_configurations() {
    const size_t d = 128;
    const size_t n = 100000000;  // 1 亿向量

    // 配置 1: 纯 DRAM
    printf("Configuration 1: Pure DRAM\n");
    printf("  Capacity: 100M vectors * 128 * 4 bytes = ~50 GB\n");
    printf("  Latency: 50-100 ns\n");
    printf("  Bandwidth: 100 GB/s\n");
    printf("  Cost: $$$$$\n");

    // 配置 2: DRAM + PMem
    printf("Configuration 2: DRAM + PMem (1TB)\n");
    printf("  DRAM: 10M vectors (hot data)\n");
    printf("  PMem: 90M vectors (cold data)\n");
    printf("  Avg Latency: 150-200 ns\n");
    printf("  Bandwidth: 80 GB/s (DRAM) + 10 GB/s (PMem)\n");
    printf("  Cost: $$\n");

    // 配置 3: DRAM + CXL Memory Pool
    printf("Configuration 3: DRAM + CXL Pool (1TB)\n");
    printf("  DRAM: 10M vectors (local)\n");
    printf("  CXL: 90M vectors (pool)\n");
    printf("  Latency: 100 ns (local) + 300 ns (pool)\n");
    printf("  Bandwidth: 100 GB/s (local) + 64 GT/s (CXL)\n");
    printf("  Cost: $$$\n");

    // 配置 4: 全栈优化 (AMX + PMem + CXL)
    printf("Configuration 4: Full Stack Optimized\n");
    printf("  AMX Acceleration: 4-8x faster\n");
    printf("  PMem Storage: 1TB persistent index\n");
    printf("  CXL Pooling: 10TB shared across nodes\n");
    printf("  Throughput: 1M queries/sec\n");
    printf("  Cost: $ (best performance per dollar)\n");
}
```

---

## 第五部分：实战练习

### 练习 1: AMX 矩阵乘法优化

```cpp
// 实现一个完整的 AMX 矩阵乘法
// 要求：支持任意大小的矩阵（自动分块）

void amx_matrix_multiply(
    const float* A, int M, int K,
    const float* B, int K, int N,
    float* C) {
    // TODO: 实现
    // 提示：
    // 1. 将矩阵分块为 16x16 的子块
    // 2. 使用 _tile_loadd 加载
    // 3. 使用 _tile_dpbus_16x16 计算
    // 4. 使用 _tile_stored 存储
    // 5. 边界处理
}
```

### 练习 2: PMem 向量数据库

```cpp
// 使用持久内存实现一个可持久化的向量索引
// 要求：支持添加、搜索、保存/加载

class PersistentVectorIndex {
public:
    void add(const float* vector, size_t id);
    void search(const float* query, int k, float* distances, size_t* ids);
    void save(const char* filename);
    void load(const char* filename);
};
```

### 练习 3: CXL 数据分层

```cpp
// 实现一个三层缓存的数据结构
// 要求：自动在本地 DRAM、CXL 内存和 SSD 之间迁移数据

class TieredDataStore {
public:
    enum Tier { LOCAL_DRAM, CXL_MEM, SSD };

    void put(const void* key, const void* value, size_t size, Tier tier);
    void* get(const void* key);
    void promote(const void* key);
    void demote(const void* key);
};
```

---

## 附录：硬件检测工具

### 检测脚本

```bash
#!/bin/bash
# detect_hardware.sh - 检测现代硬件支持

echo "=== Hardware Detection ==="

# 检测 AVX-512
echo -n "AVX-512: "
if grep -q avx512f /proc/cpuinfo; then
    echo "YES"
    grep -o 'avx512[^ ]*' /proc/cpuinfo | sort -u
else
    echo "NO"
fi

# 检测 AMX
echo -n "AMX: "
if grep -q 'avx512_bf16\|avx512_int8' /proc/cpuinfo; then
    echo "YES"
else
    echo "NO"
fi

# 检测持久内存
echo -n "Persistent Memory: "
if ls /dev/pmem* 2>/dev/null | grep -q .; then
    echo "YES"
    ls -l /dev/pmem*
else
    echo "NO"
fi

# 检测 NUMA
echo -n "NUMA Nodes: "
if command -v numactl &> /dev/null; then
    numactl --hardware | grep "node:" | wc -l
else
    echo "N/A (numactl not installed)"
fi

# 检测 CXL (需要特定硬件)
echo -n "CXL: "
if lspci | grep -qi "compute express"; then
    echo "YES"
else
    echo "NO (or not detectable)"
fi
```

---

## 第六部分：Faiss 与现代硬件集成

### 6.1 Faiss 索引的 AMX 加速

#### 6.1.1 量化距离计算的 AMX 优化

```cpp
// 使用 AMX 加速 PQ 查找表计算
// Product Quantization 的查找表计算是矩阵乘法，非常适合 AMX

namespace faiss {

struct AMXAcceleratedPQ {
    const ProductQuantizer& pq;
    int M;              // 子量化器数
    int nbits;          // 每个子量化器位数
    int ksub;           // 2^nbits

    AMXAcceleratedPQ(const ProductQuantizer& pq)
        : pq(pq), M(pq.M), nbits(pq.nbits), ksub(1 << nbits) {}

    // 使用 AMX 批量计算查找表
    void compute_lookup_table_amx(
            const float* query,
            float* lut) const {  // 输出: [M, ksub]

        // lut[m][k] = ||query[m] - centroid[m][k]||²
        // 这是一个 M × ksub 的矩阵计算

        // 为每个子量化器
        for (int m = 0; m < M; m++) {
            int d_sub = pq.dsub;  // 子向量维度
            const float* sub_query = query + m * d_sub;
            const float* centroids = pq.centroids.data() + m * ksub * d_sub;

            // 配置 AMX Tile
            _tile_loadconfig((d_sub << 16) | ksub);

            // 查询向量作为行向量 (1 × d_sub)
            __attribute__((aligned(64))) float query_tile[16 * 16];
            for (int i = 0; i < d_sub; i++) {
                query_tile[i] = sub_query[i];
            }
            _tile_loadd(0, query_tile, 16 * 4);

            // 质心表作为 (d_sub × ksub) 矩阵
            _tile_loadd(1, centroids, 16 * 4);

            // 计算查询与所有质心的距离
            // 结果 Tile 为 (1 × ksub)
            _tile_zero(2);
            for (int i = 0; i < d_sub; i += 16) {
                // AMX 矩阵乘法
                _tile_dpbus_16x16(2, 0, 1);
            }

            // 存储结果到 lut
            __attribute__((aligned(64))) float result[16 * 16];
            _tile_stored(2, result, 16 * 4);

            memcpy(lut + m * ksub, result, ksub * sizeof(float));
        }

        _tile_release();
    }

    // 批量距离计算（一次处理多个查询）
    void batch_compute_distances(
            const float* queries,   // [nq, d]
            int nq,
            const uint8_t* codes,   // [n, M]
            int n,
            float* distances) {     // [nq, n]

        // 为每个查询计算查找表
        std::vector<float> luts(nq * M * ksub);
        for (int q = 0; q < nq; q++) {
            compute_lookup_table_amx(
                queries + q * pq.d,
                luts.data() + q * M * ksub);
        }

        // 使用查找表快速计算距离
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int q = 0; q < nq; q++) {
                float dist = 0;
                for (int m = 0; m < M; m++) {
                    uint8_t code = codes[i * M + m];
                    dist += luts[q * M * ksub + m * ksub + code];
                }
                distances[q * n + i] = dist;
            }
        }
    }
};

} // namespace faiss
```

#### 6.1.2 HNSW 搜索的 AMX 优化

```cpp
// 使用 AMX 加速 HNSW 的批量距离计算

namespace faiss {

struct AMXAcceleratedHNSW {
    const HNSW& hnsw;
    const float* vectors;  // [ntotal, d]
    int d;

    // 批量计算查询与候选集的距离
    void compute_distances_batch(
            const float* queries,      // [nq, d]
            int nq,
            const idx_t* candidates,   // [nq, n_candidates]
            int n_candidates,
            float* distances) {        // [nq, n_candidates]

        // 使用 AMX 批量处理
        int batch_size = 16;  // AMX Tile 大小

        for (int q = 0; q < nq; q++) {
            const float* query = queries + q * d;

            // 配置 AMX
            _tile_loadconfig((d << 16) | batch_size);

            // 查询向量广播
            __attribute__((aligned(64))) float query_tile[16 * 16];
            for (int i = 0; i < d; i++) {
                query_tile[i] = query[i];
            }
            for (int row = 1; row < batch_size; row++) {
                memcpy(&query_tile[row * 16], query_tile, 16 * sizeof(float));
            }
            _tile_loadd(0, query_tile, 16 * 4);

            // 批量处理候选向量
            for (int base = 0; base < n_candidates; base += batch_size) {
                int current_batch = std::min(batch_size, n_candidates - base);

                // 加载候选向量
                __attribute__((aligned(64))) float candidates_tile[16 * 16];
                for (int i = 0; i < current_batch; i++) {
                    idx_t cand_id = candidates[q * n_candidates + base + i];
                    const float* vec = vectors + cand_id * d;
                    memcpy(&candidates_tile[i * 16], vec, d * sizeof(float));
                }
                _tile_loadd(1, candidates_tile, 16 * 4);

                // 计算距离
                _tile_zero(2);
                _tile_dpbus_16x16(2, 0, 1);

                // 获取结果
                __attribute__((aligned(64))) float dists[16 * 16];
                _tile_stored(2, dists, 16 * 4);

                // 存储距离（对角线元素）
                for (int i = 0; i < current_batch; i++) {
                    distances[q * n_candidates + base + i] = dists[i * 16 + i];
                }
            }
        }

        _tile_release();
    }
};

} // namespace faiss
```

### 6.2 Faiss 的持久内存集成

#### 6.2.1 PMem 倒排索引

```cpp
// 使用持久内存实现大规模倒排索引

namespace faiss {

struct PMemInvertedLists : InvertedLists {
    size_t nlist;
    size_t code_size;

    // PMem 映射区域
    void* pmem_base;
    size_t pmem_size;

    // PMem 上的数据结构
    struct {
        size_t nlist;
        size_t code_size;
        size_t total_size;      // 总向量数
        size_t list_offsets[0]; // 每个列表的偏移量
    } * pmem_header;

    // 每个 list 的数据: [ids..., codes...]
    // list_offsets[i] 指向第 i 个 list 的起始位置

    PMemInvertedLists(size_t nlist, size_t code_size, const char* pmem_path)
        : InvertedLists(nlist, code_size) {

        // 计算 PMem 大小
        size_t header_size = sizeof(size_t) * 3 + nlist * sizeof(size_t);
        size_t max_vectors = 100000000;  // 1 亿向量
        size_t data_size = max_vectors * (sizeof(idx_t) + code_size);
        pmem_size = header_size + data_size;

        // 创建或映射 PMem 文件
        int is_pmem;
        pmem_base = pmem_map_file(
            pmem_path, pmem_size,
            PMEM_FILE_CREATE,
            0666,
            &pmem_size,
            &is_pmem
        );

        pmem_header = (decltype(pmem_header))pmem_base;
        pmem_header->nlist = nlist;
        pmem_header->code_size = code_size;
        pmem_header->total_size = 0;

        // 初始化偏移量
        size_t offset = header_size;
        for (size_t i = 0; i < nlist; i++) {
            pmem_header->list_offsets[i] = offset;
            offset += 1024 * sizeof(idx_t);  // 初始空间
        }
    }

    ~PMemInvertedLists() {
        pmem_unmap(pmem_base, pmem_size);
    }

    size_t list_size(size_t list_no) const override {
        // PMem 上的 list 格式: [count, id1, id2, ..., code1, code2, ...]
        uint8_t* list_ptr = (uint8_t*)pmem_base + pmem_header->list_offsets[list_no];
        return *(size_t*)list_ptr;
    }

    const uint8_t* get_codes(size_t list_no) const override {
        uint8_t* list_ptr = (uint8_t*)pmem_base + pmem_header->list_offsets[list_no];
        size_t count = *(size_t*)list_ptr;

        // 跳过 count 和 ids
        size_t ids_size = count * sizeof(idx_t);
        return list_ptr + sizeof(size_t) + ids_size;
    }

    const idx_t* get_ids(size_t list_no) const override {
        uint8_t* list_ptr = (uint8_t*)pmem_base + pmem_header->list_offsets[list_no];
        return (idx_t*)(list_ptr + sizeof(size_t));
    }

    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override {

        uint8_t* list_ptr = (uint8_t*)pmem_base + pmem_header->list_offsets[list_no];
        size_t* count_ptr = (size_t*)list_ptr;
        size_t old_count = *count_ptr;

        // 检查空间是否足够
        size_t required_size = sizeof(size_t) +
                              (old_count + n_entry) * (sizeof(idx_t) + code_size);
        size_t max_size = (list_no == nlist - 1) ?
                         pmem_size - pmem_header->list_offsets[list_no] :
                         pmem_header->list_offsets[list_no + 1] - pmem_header->list_offsets[list_no];

        if (required_size > max_size) {
            // 扩容逻辑（简化：抛出异常）
            throw std::runtime_error("PMem list full, need expansion");
        }

        // 添加 ids
        idx_t* ids_ptr = (idx_t*)(list_ptr + sizeof(size_t) + old_count * sizeof(idx_t));
        memcpy(ids_ptr, ids, n_entry * sizeof(idx_t));

        // 添加 codes
        uint8_t* codes_ptr = list_ptr + sizeof(size_t) +
                             (old_count + n_entry) * sizeof(idx_t) +
                             old_count * code_size;
        memcpy(codes_ptr, code, n_entry * code_size);

        // 更新 count
        *count_ptr = old_count + n_entry;
        pmem_header->total_size += n_entry;

        // 持久化
        pmem_persist(list_ptr, required_size);

        return old_count;
    }

    void resize(size_t list_no, size_t new_size) override {
        // 简化实现：只允许缩小
        uint8_t* list_ptr = (uint8_t*)pmem_base + pmem_header->list_offsets[list_no];
        size_t* count_ptr = (size_t*)list_ptr;

        if (new_size < *count_ptr) {
            *count_ptr = new_size;
            pmem_persist(count_ptr, sizeof(size_t));
        }
    }
};

} // namespace faiss
```

#### 6.2.2 PMem IndexIVF 实现

```cpp
// 使用 PMem 存储大规模 IVF 索引

namespace faiss {

struct IndexIVFPMem : IndexIVFFlat {
    PMemInvertedLists* pmem_invlists;

    IndexIVFPMem(
            size_t d,
            size_t nlist,
            const char* pmem_path,
            MetricType metric = METRIC_L2)
        : IndexIVFFlat(d, nlist, metric) {

        pmem_invlists = new PMemInvertedLists(nlist, d * sizeof(float), pmem_path);
        invlists = pmem_invlists;
    }

    // 从 PMem 加载索引
    void load_from_pmem(const char* pmem_path) {
        // 映射 PMem 文件
        int is_pmem;
        size_t mapped_len;
        void* pmem = pmem_map_file(
            pmem_path, 0,
            PMEM_FILE_CREATE,
            0666,
            &mapped_len,
            &is_pmem
        );

        // 重建索引
        // 1. 加载聚类中心
        // 2. 加载倒排列表
        // 3. 设置 ntotal

        pmem_unmap(pmem, mapped_len);
    }

    // 保存到 PMem（已经是持久化的，无需额外操作）
    void flush_to_pmem() {
        // PMem 是持久化的，但可以确保数据已写入
        pmem_invlists->print_stats();
    }
};

} // namespace faiss
```

### 6.3 CXL 内存池化集成

#### 6.3.1 CXL-aware 内存分配器

```cpp
// CXL 内存感知的分配器

namespace faiss {

struct CXLMemoryAllocator {
    struct CXLMemoryRegion {
        void* base;           // CXL 内存基址
        size_t size;          // 区域大小
        int node_id;          // NUMA 节点 ID
        double latency_ns;    // 访问延迟（纳秒）
        double bandwidth_gbps; // 带宽（GB/s）
    };

    std::vector<CXLMemoryRegion> regions;

    CXLMemoryAllocator() {
        // 发现 CXL 内存区域
        discover_cxl_regions();
    }

    void discover_cxl_regions() {
        // 通过 sysfs 发现 CXL 内存
        // /sys/bus/cxl/devices/

        for (int node = 0; ; node++) {
            char path[256];
            snprintf(path, sizeof(path),
                    "/sys/bus/cxl/devices/cxl%d/size", node);

            FILE* f = fopen(path, "r");
            if (!f) break;

            size_t size;
            fscanf(f, "%zu", &size);
            fclose(f);

            // 映射 CXL 内存
            void* base = numa_alloc_onnode(size, node);

            CXLMemoryRegion region;
            region.base = base;
            region.size = size;
            region.node_id = node;
            region.latency_ns = 200 + node * 50;  // 估算
            region.bandwidth_gbps = 64.0;

            regions.push_back(region);
        }
    }

    // 分配内存（自动选择最优区域）
    void* allocate(size_t size, int access_pattern) {
        if (regions.empty()) {
            // 回退到普通分配
            return aligned_alloc(64, size);
        }

        // 根据访问模式选择区域
        // 0 = 随机访问，选择低延迟
        // 1 = 顺序访问，选择高带宽
        // 2 = 大块数据，选择大容量

        if (access_pattern == 0) {
            // 选择延迟最低的区域（通常是本地 DRAM）
            return regions[0].base;
        } else if (access_pattern == 1) {
            // 选择带宽最高的区域
            int best = 0;
            for (size_t i = 1; i < regions.size(); i++) {
                if (regions[i].bandwidth_gbps > regions[best].bandwidth_gbps) {
                    best = i;
                }
            }
            return regions[best].base;
        } else {
            // 选择容量最大的区域
            int best = 0;
            for (size_t i = 1; i < regions.size(); i++) {
                if (regions[i].size > regions[best].size) {
                    best = i;
                }
            }
            return regions[best].base;
        }
    }
};

} // namespace faiss
```

#### 6.3.2 跨节点索引分片

```cpp
// 使用 CXL 实现跨节点索引

namespace faiss {

struct CXLShardedIndex : Index {
    int nshards;                     // 分片数
    std::vector<Index*> shards;     // 每个分片在一个 CXL 节点上

    CXLShardedIndex(int d, MetricType metric, int nshards)
        : Index(d, metric), nshards(nshards) {

        // 为每个 CXL 节点创建索引
        for (int i = 0; i < nshards; i++) {
            // 索引数据存储在对应的 CXL 内存区域
            shards[i] = new IndexFlatL2(d);
        }
    }

    void add(idx_t n, const float* x) override {
        // 轮询分配到不同分片
        for (idx_t i = 0; i < n; i++) {
            int shard_id = i % nshards;
            shards[shard_id]->add(1, x + i * d);
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
        for (int s = 0; s < nshards; s++) {
            for (idx_t q = 0; q < n; q++) {
                // 每个分片返回 k 个结果
                float* local_dist = new float[k];
                idx_t* local_labels = new idx_t[k];

                shards[s]->search(
                    1, x + q * d, k,
                    local_dist, local_labels
                );

                // 合并到全局结果（需要同步）
                #pragma omp critical
                {
                    merge_results(
                        local_dist, local_labels, k,
                        distances + q * k, labels + q * k, k
                    );
                }

                delete[] local_dist;
                delete[] local_labels;
            }
        }
    }

private:
    void merge_results(
            const float* local_dist,
            const idx_t* local_labels,
            int k,
            float* global_dist,
            idx_t* global_labels,
            int k_global) {

        // 将 local 结果合并到 global 堆
        for (int i = 0; i < k; i++) {
            heap_push<CMax<idx_t>>(
                k_global, global_dist, global_labels,
                local_labels[i], local_dist[i]
            );
        }
    }
};

} // namespace faiss
```

---

## 总结

本课程深入讲解了现代硬件特性的实战应用：

### 关键要点

1. **Intel AMX**：
   - Tile 寄存器和矩阵乘法加速
   - 混合精度计算
   - 向量搜索批量优化

2. **持久内存**：
   - 大容量向量存储
   - App Direct 模式编程
   - 性能优化技巧

3. **CXL**：
   - 内存池化
   - 数据分层
   - 分布式扩展

### 性能对比

| 配置 | 容量 | 延迟 | 带宽 | 成本 | 适用场景 |
|------|------|------|------|------|----------|
| 纯 DRAM | ~500GB | 50-100ns | 100GB/s | $$$$$ | 小规模热数据 |
| DRAM+PMem | ~1.5TB | 100-200ns | 混合 | $$$ | 中规模混合 |
| DRAM+CXL | ~10TB | 100-300ns | 混合+64GT/s | $$$ | 大规模共享 |
| 全栈优化 | ~10TB+ | 100-200ns | 混合+AMX | $$ | 生产环境 |

### 下一步学习

- 《向量搜索完整优化案例》：从零开始构建
- 《高级专题深入》：NUMA、异步I/O
- 继续关注硬件发展：新指令集、新硬件

**建议练习**：
1. 在支持 AMX 的机器上测试代码
2. 使用 PMem 构建持久化向量索引
3. 设计数据分层策略
4. 性能测试和对比

祝学习顺利！🚀
