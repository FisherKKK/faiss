# FastScan优化深度剖析 - pq4_fast_scan.cpp源码解析

## 文档说明

本文档深入剖析Faiss中FastScan优化的核心实现`pq4_fast_scan.cpp`，详细讲解4-bit Product Quantization的打包、查找表处理等底层优化技术。

**前置知识**：
- 已完成《faiss_day09_fastscan》基础课程
- 熟悉Product Quantization (PQ)原理
- 了解SIMD优化基础

---

## 目录
- [1. FastScan概述](#1-fastscan概述)
- [2. 代码打包函数](#2-代码打包函数)
- [3. 查找表(LUT)打包](#3-查找表lut打包)
- [4. CodePackerPQ4类](#4-codepackerpq4类)
- [5. 内存布局优化](#5-内存布局优化)
- [6. 性能分析](#6-性能分析)
- [7. 总结](#7-总结)

---

## 1. FastScan概述

### 1.1 什么是FastScan

**FastScan**是Faiss中针对Product Quantization的一种优化实现，主要特点：

1. **4-bit量化**：每个子量化器使用4-bit编码（16个质心）
2. **SIMD优化**：批量处理32个向量
3. **查找表加速**：预计算距离查找表
4. **内存打包**：特殊的数据布局提高缓存效率

### 1.2 核心数据结构

```cpp
// 原始PQ编码（未打包）
// 每个向量的编码: uint8_t codes[M]  // M个子量化器
// 每个codes[i]是4-bit值（0-15），存储在uint8_t的低4位或高4位

// 打包后的编码（FastScan格式）
// 特殊的内存布局，优化SIMD访问
uint8_t* blocks;  // 打包后的编码块
```

### 1.3 参数定义

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `M` | 子量化器数量 | 64-256 |
| `nsq` | 处理的子量化器数量 | M（全部）或部分 |
| `bbs` | 块大小（block size） | 32, 64 |
| `nb` | 向量总数 | - |
| `code_stride` | 编码步长 | 0（自动）或指定 |

---

## 2. 代码打包函数

### 2.1 pq4_pack_codes函数

```cpp
void pq4_pack_codes(
        const uint8_t* codes,      // 原始编码
        size_t ntotal,             // 向量总数
        size_t M,                  // 子量化器数量
        size_t nb,                 // 要打包的向量数
        size_t bbs,                // 块大小
        size_t nsq,                // 子量化器数量
        uint8_t* blocks,           // 输出：打包后的块
        size_t code_stride) {      // 编码步长（0=自动）
```

**功能**：将原始的PQ编码重新打包为SIMD友好的格式

**处理流程**：

```cpp
// 步骤1: 确定步长
size_t actual_stride = (code_stride == 0) ? (M + 1) / 2 : code_stride;

// 步骤2: 输入验证
FAISS_THROW_IF_NOT(bbs % 32 == 0);   // 块大小必须是32的倍数
FAISS_THROW_IF_NOT(nb % bbs == 0);   // 向量数是块大小的倍数
FAISS_THROW_IF_NOT(nsq % 2 == 0);    // 子量化器数必须是偶数

// 步骤3: 初始化输出
memset(blocks, 0, nb * nsq / 2);

// 步骤4: 定义重排表（字节序处理）
#ifdef FAISS_BIG_ENDIAN
const uint8_t perm0[16] = {8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7};
#else
const uint8_t perm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
#endif
```

### 2.2 打包算法详解

```cpp
uint8_t* codes2 = blocks;
for (size_t i0 = 0; i0 < nb; i0 += bbs) {          // 遍历块
    for (int sq = 0; sq < nsq; sq += 2) {            // 遍历子量化器对
        for (size_t i = 0; i < bbs; i += 32) {       // 32个向量一组
            std::array<uint8_t, 32> c, c0, c1;

            // 从源矩阵提取一列
            get_matrix_column(
                codes, ntotal, actual_stride,
                i0 + i, sq / 2, c);

            // 分离低4位和高4位
            for (int j = 0; j < 32; j++) {
                c0[j] = c[j] & 15;       // 低4位 (0-15)
                c1[j] = c[j] >> 4;       // 高4位 (0-15)
            }

            // 重排并打包
            for (int j = 0; j < 16; j++) {
                uint8_t d0, d1;
                d0 = c0[perm0[j]] | (c0[perm0[j] + 16] << 4);
                d1 = c1[perm0[j]] | (c1[perm0[j] + 16] << 4);
                codes2[j] = d0;
                codes2[j + 16] = d1;
            }
            codes2 += 32;
        }
    }
}
```

**打包示意图**：

```
原始编码（32个向量，每个1字节）:
[c0] [c1] [c2] ... [c31]

分离为低4位和高4位:
c0: [00, 01, 02, ..., 31]  // 低4位
c1: [00, 01, 02, ..., 31]  // 高4位

重排后（16字节存储32个4-bit值）:
[d0_0, d0_1, ..., d0_15]  // 前16个值的低4位
[d1_0, d1_1, ..., d1_15]  // 后16个值的低4位
...

内存布局:
[codes2 + 0]  = [c0_0, c0_8, ..., c0_7, c0_15]  (重排后的低4位)
[codes2 + 16] = [c1_0, c1_8, ..., c1_7, c1_15]  (重排后的高4位)
```

### 2.3 重排表(perm0)的作用

```cpp
// perm0将线性索引重排为SIMD友好的模式
// 原始索引: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// perm0重排: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]

// 效果：将连续的16个值分成两组，交替排列
// 原始: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// 重排: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]

// 优势：SIMD操作时提高缓存局部性
```

### 2.4 pq4_pack_codes_range函数

```cpp
void pq4_pack_codes_range(
        const uint8_t* codes,
        size_t M,
        size_t i0,              // 范围起始
        size_t i1,              // 范围结束
        size_t bbs,
        size_t nsq,
        uint8_t* blocks,
        size_t code_stride) {
    // ...
    size_t block0 = i0 / bbs;
    size_t block1 = ((i1 - 1) / bbs) + 1;

    for (size_t b = block0; b < block1; b++) {
        // 只处理受影响的块
        // ...
    }
}
```

**用途**：增量更新打包编码，而不是重新打包所有数据

---

## 3. 查找表(LUT)打包

### 3.1 pq4_pack_LUT函数

```cpp
void pq4_pack_LUT(
        int nq,                 // 查询数量
        int nsq,                // 子量化器数量
        const uint8_t* src,     // 原始LUT: [nq * nsq * 16]
        uint8_t* dest) {        // 打包后的LUT
    for (int q = 0; q < nq; q++) {
        for (int sq = 0; sq < nsq; sq += 2) {
            memcpy(dest + (sq / 2 * nq + q) * 32,
                   src + (q * nsq + sq) * 16,
                   16);
            memcpy(dest + (sq / 2 * nq + q) * 32 + 16,
                   src + (q * nsq + sq + 1) * 16,
                   16);
        }
    }
}
```

**LUT内存布局转换**：

```
原始布局（未优化）:
Query 0:
  SQ0: [d00, d01, ..., d015]  // 16个距离
  SQ1: [d10, d11, ..., d115]
  SQ2: [d20, d21, ..., d215]
  ...
Query 1:
  SQ0: [d00, d01, ..., d015]
  ...

打包后布局:
SQ对0, Query 0:
  [d00_Q0_SQ0, ..., d015_Q0_SQ0, d00_Q0_SQ1, ..., d015_Q0_SQ1]
SQ对0, Query 1:
  [d00_Q1_SQ0, ..., d015_Q1_SQ0, d00_Q1_SQ1, ..., d015_Q1_SQ1]
...
```

### 3.2 pq4_pack_LUT_qbs函数

```cpp
int pq4_pack_LUT_qbs(
        int qbs,                // 查询块大小（编码）
        int nsq,
        const uint8_t* src,
        uint8_t* dest) {
    FAISS_THROW_IF_NOT(nsq % 2 == 0);
    size_t dim12 = 16 * nsq;
    int i0 = 0;
    int qi = qbs;

    // 解码qbs（位编码）
    while (qi) {
        int nq = qi & 15;  // 低4位
        qi >>= 4;          // 右移4位
        pq4_pack_LUT(nq, nsq, src + i0 * dim12, dest + i0 * dim12);
        i0 += nq;
    }
    return i0;
}
```

**qbs编码**：将查询数量编码在4位一组中

```
qbs = 0x1234
  解码为: nq0=4, nq1=3, nq2=2, nq3=1

处理:
  i0=0:  处理4个查询
  i0=4:  处理3个查询
  i0=7:  处理2个查询
  i0=9:  处理1个查询
```

---

## 4. CodePackerPQ4类

### 4.1 类定义

```cpp
class CodePackerPQ4 {
public:
    size_t nsq;          // 子量化器数量
    size_t nvec;         // 每块的向量数
    size_t code_size;    // 编码大小（字节）
    size_t block_size;   // 块大小（字节）

    CodePackerPQ4(size_t nsq, size_t bbs);

    // 打包单个向量
    void pack_1(
            const uint8_t* flat_code,
            size_t offset,
            uint8_t* block) const;

    // 解包单个向量
    void unpack_1(
            const uint8_t* block,
            size_t offset,
            uint8_t* flat_code) const;
};
```

### 4.2 构造函数

```cpp
CodePackerPQ4::CodePackerPQ4(size_t nsq, size_t bbs) {
    this->nsq = nsq;
    nvec = bbs;
    code_size = (nsq * 4 + 7) / 8;      // 4-bit -> 字节
    block_size = ((nsq + 1) / 2) * bbs; // 每块的字节数
}
```

**计算示例**：
```cpp
nsq = 64, bbs = 32
code_size = (64 * 4 + 7) / 8 = 32 字节
block_size = ((64 + 1) / 2) * 32 = 32 * 32 = 1024 字节
```

### 4.3 pack_1函数

```cpp
void CodePackerPQ4::pack_1(
        const uint8_t* flat_code,
        size_t offset,
        uint8_t* block) const {
    size_t bbs = nvec;
    if (offset >= nvec) {
        block += (offset / nvec) * block_size;
        offset = offset % nvec;
    }
    for (size_t i = 0; i < code_size; i++) {
        uint8_t code = flat_code[i];
        // 每字节包含2个4-bit码
        pq4_set_packed_element(block, code & 15, bbs, nsq, offset, 2 * i);
        pq4_set_packed_element(block, code >> 4, bbs, nsq, offset, 2 * i + 1);
    }
}
```

### 4.4 unpack_1函数

```cpp
void CodePackerPQ4::unpack_1(
        const uint8_t* block,
        size_t offset,
        uint8_t* flat_code) const {
    size_t bbs = nvec;
    if (offset >= nvec) {
        block += (offset / nvec) * block_size;
        offset = offset % nvec;
    }
    for (size_t i = 0; i < code_size; i++) {
        uint8_t code0, code1;
        code0 = pq4_get_packed_element(block, bbs, nsq, offset, 2 * i);
        code1 = pq4_get_packed_element(block, bbs, nsq, offset, 2 * i + 1);
        flat_code[i] = code0 | (code1 << 4);
    }
}
```

---

## 5. 内存布局优化

### 5.1 原始布局 vs 打包布局

```
原始布局（每行一个向量的编码）:
Vector 0: [c0_0, c1_0, c2_0, ..., cM_0]
Vector 1: [c0_1, c1_1, c2_1, ..., cM_1]
...
Vector 31: [c0_31, c1_31, c2_31, ..., cM_31]

打包布局（SIMD友好）:
Block 0:
  SQ对0: [c0_0_低4位, c0_8_低4位, ..., c0_7_低4位, c0_15_低4位,
         c0_0_高4位, c0_8_高4位, ..., c0_7_高4位, c0_15_高4位]
  SQ对1: [c2_0_低4位, c2_8_低4位, ..., ...]
  ...
```

### 5.2 优势分析

1. **缓存友好**：连续访问32个向量的同一子量化器
2. **SIMD对齐**：32个4-bit值正好16字节
3. **批量处理**：一次处理32个向量

### 5.3 地址计算

```cpp
// 获取向量在块中的特定地址
size_t get_vector_specific_address(
        size_t bbs,
        size_t vector_id,
        size_t sq,
        bool& shift) {
    // 向量在块内的位置
    vector_id = vector_id % bbs;
    shift = vector_id > 15;       // 高4位还是低4位？
    vector_id = vector_id & 15;   // 低4位索引

    // 计算地址
    size_t address;
    if (vector_id < 8) {
        address = vector_id << 1;  // 0, 2, 4, ..., 14
    } else {
        address = ((vector_id - 8) << 1) + 1;  // 1, 3, 5, ..., 15
    }
    if (sq & 1) {
        address += 16;
    }
    return (sq >> 1) * bbs + address;
}
```

**地址计算示意**：

```
bbs = 32, nsq = 64

vector_id = 0, sq = 0:
  shift = false (低4位)
  address = 0 * 2 = 0
  最终地址 = (0 >> 1) * 32 + 0 = 0

vector_id = 8, sq = 0:
  shift = false (低4位)
  address = (8 - 8) * 2 + 1 = 1
  最终地址 = (0 >> 1) * 32 + 1 = 1

vector_id = 16, sq = 0:
  shift = true (高4位)
  address = 0 * 2 = 0
  最终地址 = (0 >> 1) * 32 + 0 = 0 (但要>>4)
```

---

## 6. 性能分析

### 6.1 时间复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| pq4_pack_codes | O(nb * nsq) | 线性遍历所有向量 |
| pq4_pack_LUT | O(nq * nsq) | 线性遍历所有查询 |
| pack_1/unpack_1 | O(code_size) | 单个向量 |

### 6.2 空间复杂度

| 数据类型 | 空间 | 说明 |
|----------|------|------|
| 原始编码 | nb * (M+1)/2 | 未打包 |
| 打包编码 | nb * nsq/2 | 打包后 |
| LUT | nq * nsq * 16 | 查找表 |

### 6.3 SIMD加速比

```
标量版本: 每次1个向量
FastScan: 每次处理32个向量
理论加速: 32x

实际加速: 20-25x (考虑内存延迟等)
```

---

## 7. 总结

### 7.1 关键优化技术

1. **4-bit编码**：每个子量化器16个质心，压缩率高
2. **内存打包**：SIMD友好的数据布局
3. **查找表**：预计算距离，避免重复计算
4. **批量处理**：一次处理32个向量

### 7.2 设计权衡

| 方面 | 优势 | 劣势 |
|------|------|------|
| 4-bit量化 | 内存占用小 | 精度降低 |
| 打包布局 | SIMD快速 | 随机访问慢 |
| 批量处理 | 吞吐量高 | 延迟增加 |

### 7.3 适用场景

1. **大规模搜索**：nb > 1M
2. **内存受限**：需要压缩编码
3. **批量查询**：nq > 10
4. **SIMD可用**：AVX2或AVX-512

---

## 附录A：完整示例

```cpp
#include <faiss/impl/pq4_fast_scan.h>

using namespace faiss;

void example_pack_codes() {
    // 参数设置
    size_t M = 64;           // 子量化器数量
    size_t nb = 10000;       // 向量数量
    size_t bbs = 32;         // 块大小
    size_t nsq = 64;         // 处理的子量化器数量

    // 原始编码
    std::vector<uint8_t> codes(nb * (M + 1) / 2);
    // ... 填充编码

    // 打包
    std::vector<uint8_t> blocks(nb * nsq / 2);
    pq4_pack_codes(
        codes.data(), nb, M, nb, bbs, nsq,
        blocks.data(), 0);

    // 使用打包后的编码进行搜索
    // ...
}

void example_pack_lut() {
    // 参数设置
    int nq = 10;             // 查询数量
    int nsq = 64;            // 子量化器数量

    // 原始LUT
    std::vector<uint8_t> lut(nq * nsq * 16);
    // ... 填充LUT

    // 打包LUT
    std::vector<uint8_t> packed_lut(nq * nsq * 16);
    pq4_pack_LUT(nq, nsq, lut.data(), packed_lut.data());

    // 使用打包后的LUT进行搜索
    // ...
}

void example_code_packer() {
    // 参数设置
    size_t nsq = 64;
    size_t bbs = 32;

    // 创建CodePacker
    CodePackerPQ4 packer(nsq, bbs);

    // 单个向量编码
    std::vector<uint8_t> flat_code(32);
    std::vector<uint8_t> block(packer.block_size);

    // 打包
    packer.pack_1(flat_code.data(), 0, block.data());

    // 解包
    std::vector<uint8_t> unpacked(32);
    packer.unpack_1(block.data(), 0, unpacked.data());
}

int main() {
    example_pack_codes();
    example_pack_lut();
    example_code_packer();
    return 0;
}
```

## 附录B：相关源文件

- `faiss/impl/pq4_fast_scan.cpp` - 打包函数实现
- `faiss/impl/pq4_fast_scan.h` - 接口定义
- `faiss/impl/pq4_fast_scan_search_1.cpp` - k=1搜索实现
- `faiss/impl/pq4_fast_scan_search_qbs.cpp` - 批量搜索实现
