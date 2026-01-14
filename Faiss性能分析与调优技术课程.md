# Faiss性能分析与调优技术课程

## 课程目录

1. [性能分析基础理论](#1-性能分析基础理论)
2. [Faiss内置性能监控工具](#2-faiss内置性能监控工具)
3. [基准测试框架详解](#3-基准测试框架详解)
4. [外部Profiling工具实战](#4-外部profiling工具实战)
5. [性能瓶颈诊断方法](#5-性能瓶颈诊断方法)
6. [索引级别性能调优](#6-索引级别性能调优)
7. [SIMD和CPU优化](#7-simd和cpu优化)
8. [GPU性能优化](#8-gpu性能优化)
9. [内存和IO优化](#9-内存和io优化)
10. [生产环境监控方案](#10-生产环境监控方案)
11. [实战案例分析](#11-实战案例分析)
12. [性能优化Checklist](#12-性能优化checklist)

---

## 1. 性能分析基础理论

### 1.1 性能指标体系

#### 核心性能指标

**1. 吞吐量 (Throughput)**
- 定义：单位时间内完成的查询数量 (QPS - Queries Per Second)
- 计算：`QPS = num_queries / total_time_seconds`
- 重要性：衡量系统处理能力的关键指标

**2. 延迟 (Latency)**
- p50延迟：中位数响应时间
- p95/p99延迟：95%/99%查询的响应时间
- 尾延迟 (Tail Latency)：p99.9或最大延迟
- 重要性：影响用户体验的直接指标

**3. 召回率 (Recall)**
- 定义：`Recall@K = |返回的真实K近邻| / K`
- 评估方法：与暴力搜索结果对比
- 权衡：召回率与速度的trade-off

**4. 资源利用率**
- CPU利用率：单核/多核利用情况
- 内存使用：索引大小、搜索峰值内存
- 缓存命中率：L1/L2/L3 cache性能
- IO带宽：磁盘/网络读写速度

#### 性能分析金字塔

```
        准确性 (Recall)
           ↑
    速度 (Latency/QPS)
           ↑
   资源效率 (CPU/内存/能耗)
```

### 1.2 性能优化方法论

#### Amdahl定律

```
加速比 = 1 / ((1-P) + P/S)
P: 可并行部分占比
S: 并行加速倍数
```

**启示：**
- 优先优化占时最多的部分
- 评估并行化收益
- 关注串行瓶颈

#### 性能分析流程

```
1. 建立基线 (Baseline)
   ↓
2. 性能剖析 (Profiling)
   ↓
3. 识别瓶颈 (Bottleneck)
   ↓
4. 优化实施 (Optimization)
   ↓
5. 验证效果 (Validation)
   ↓
6. 回到步骤2 (迭代)
```

### 1.3 Faiss特定的性能因素

#### 向量搜索性能影响因素

```
性能 = f(
    索引类型,          # IVF vs HNSW vs Flat
    数据规模,          # 百万 vs 亿级
    向量维度,          # 128 vs 2048
    查询参数,          # nprobe, ef_search
    硬件平台,          # CPU SIMD等级, GPU型号
    数据分布,          # 聚类性、离群点
    并发级别           # 单线程 vs 多线程
)
```

#### 计算复杂度分析

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| IndexFlat搜索 | O(N*D) | N:库大小, D:维度 |
| IVF搜索 | O(nprobe*N/nlist*D) | nlist:聚类数 |
| HNSW搜索 | O(log N * efSearch * D) | 图结构搜索 |
| PQ距离计算 | O(N*M) | M:子向量数(<<D) |
| 向量加法 | O(N*D) | 构建索引 |

---

## 2. Faiss内置性能监控工具

### 2.1 时间测量API

#### 基础计时函数

**C++ API (faiss/utils/utils.h)**

```cpp
#include <faiss/utils/utils.h>

// 1. 毫秒级精度计时
double t0 = faiss::getmillisecs();
// ... 执行操作 ...
double elapsed = faiss::getmillisecs() - t0;
printf("操作耗时: %.3f ms\n", elapsed);

// 2. CPU周期计数 (x86-64)
uint64_t cycles_start = faiss::get_cycles();
// ... 执行操作 ...
uint64_t cycles_elapsed = faiss::get_cycles() - cycles_start;
printf("CPU周期数: %lu\n", cycles_elapsed);

// 3. 内存使用监控
size_t mem_kb_before = faiss::get_mem_usage_kb();
// ... 分配内存 ...
size_t mem_kb_after = faiss::get_mem_usage_kb();
printf("内存增长: %.2f MB\n", (mem_kb_after - mem_kb_before) / 1024.0);
```

**Python API**

```python
import time
import faiss

# 方法1: Python time模块
t0 = time.time()
D, I = index.search(xq, k)
elapsed = time.time() - t0
print(f"搜索耗时: {elapsed*1000:.2f} ms")

# 方法2: 高精度perf_counter
t0 = time.perf_counter()
D, I = index.search(xq, k)
elapsed = time.perf_counter() - t0
print(f"搜索耗时: {elapsed*1000:.3f} ms")
```

#### GPU计时器

```cpp
#include <faiss/gpu/utils/Timer.h>

// GPU内核精确计时
faiss::gpu::KernelTimer timer(stream);
// ... GPU操作 ...
float gpu_ms = timer.elapsedMilliseconds();

// CPU端计时
faiss::gpu::CpuTimer cpu_timer;
// ... CPU操作 ...
float cpu_ms = cpu_timer.elapsedMilliseconds();
```

### 2.2 索引统计结构

#### IndexIVF统计 (最常用)

**C++ 使用**

```cpp
#include <faiss/IndexIVF.h>

// 重置统计
faiss::indexIVF_stats.reset();

// 执行搜索
index->search(nq, xq, k, distances, labels);

// 读取统计
printf("查询数: %zu\n", faiss::indexIVF_stats.nq);
printf("扫描列表数: %zu\n", faiss::indexIVF_stats.nlist);
printf("距离计算次数: %zu\n", faiss::indexIVF_stats.ndis);
printf("堆更新次数: %zu\n", faiss::indexIVF_stats.nheap_updates);
printf("量化时间: %.3f ms\n", faiss::indexIVF_stats.quantization_time);
printf("搜索时间: %.3f ms\n", faiss::indexIVF_stats.search_time);

// 计算效率指标
double avg_list_size = (double)faiss::indexIVF_stats.ndis /
                       faiss::indexIVF_stats.nlist;
printf("平均列表大小: %.1f\n", avg_list_size);
```

**Python 使用**

```python
import faiss

# 创建IVF索引
index = faiss.index_factory(d, "IVF1024,Flat")
index.train(xt)
index.add(xb)

# 重置统计计数器
faiss.cvar.indexIVF_stats.reset()

# 执行搜索
D, I = index.search(xq, k)

# 读取详细统计
stats = faiss.cvar.indexIVF_stats
print(f"查询数: {stats.nq}")
print(f"扫描倒排列表总数: {stats.nlist}")
print(f"距离计算次数: {stats.ndis}")
print(f"堆更新次数: {stats.nheap_updates}")
print(f"量化耗时: {stats.quantization_time:.3f} ms")
print(f"搜索总耗时: {stats.search_time:.3f} ms")

# 性能分析
print(f"\n性能分析:")
print(f"  量化时间占比: {stats.quantization_time/stats.search_time*100:.1f}%")
print(f"  平均每查询距离计算: {stats.ndis/stats.nq:.0f}")
print(f"  平均每列表向量数: {stats.ndis/stats.nlist:.1f}")
```

#### 其他索引统计结构

**HNSW统计**

```python
# HNSW图统计 (需要从C++层导出)
# 可用于分析图构建和搜索性能
index_hnsw = faiss.IndexHNSWFlat(d, M)
index_hnsw.verbose = True  # 启用详细日志

# 搜索时会输出:
# - 访问的节点数
# - 距离计算次数
# - 每层的搜索深度
```

**FastScan统计**

```python
# FastScan专用统计
index_fs = faiss.IndexIVFFastScan(quantizer, d, nlist, 4, faiss.METRIC_L2)
# 统计SIMD扫描效率
faiss.cvar.fastscan_stats.reset()
# ... 搜索后读取
```

### 2.3 详细日志输出

#### Verbose模式

```python
# 启用索引详细日志
index.verbose = True

# IVF索引会输出:
# - 训练聚类进度
# - 每次搜索的nprobe、扫描向量数
# - 量化和搜索各阶段耗时

# 启用量化器详细日志
if hasattr(index, 'quantizer'):
    index.quantizer.verbose = True

# HNSW启用详细输出
index_hnsw.hnsw.efConstruction = 40
index_hnsw.hnsw.verbose = True  # 构建和搜索日志
```

**示例输出解读:**

```
IndexIVF::search: nq=100 k=10 nprobe=8
  quantize: 0.125 ms
  scan lists: 2.345 ms
  heap: 0.089 ms
  total: 2.567 ms
```

### 2.4 编译信息检查

```python
import faiss

# 查看编译选项 (SIMD指令集等)
compile_opts = faiss.get_compile_options()
print(f"编译选项: {compile_opts}")

# 应包含: AVX2, AVX512, NEON (ARM), GPU, etc.
# 示例输出: "AVX2 AVX512 CUDA GPU"

# 检查是否启用GPU
print(f"GPU可用: {hasattr(faiss, 'StandardGpuResources')}")

# 检查OpenMP线程数
import os
print(f"OMP线程数: {os.environ.get('OMP_NUM_THREADS', 'default')}")
```

### 2.5 内存分析工具

```python
import faiss
import os
import psutil

def get_process_memory_mb():
    """获取当前进程内存使用(MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# 监控索引构建内存
mem_before = get_process_memory_mb()
index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
index.train(xt)
mem_after_train = get_process_memory_mb()
index.add(xb)
mem_after_add = get_process_memory_mb()

print(f"训练内存增长: {mem_after_train - mem_before:.1f} MB")
print(f"添加向量内存增长: {mem_after_add - mem_after_train:.1f} MB")
print(f"索引总内存: {mem_after_add - mem_before:.1f} MB")

# Faiss内置内存统计
import sys
index_size_bytes = sys.getsizeof(faiss.serialize_index(index))
print(f"序列化索引大小: {index_size_bytes / 1024 / 1024:.1f} MB")
```

---

## 3. 基准测试框架详解

### 3.1 bench_fw现代框架

#### 核心组件架构

```python
from faiss.contrib.datasets import SyntheticDataset
from faiss.benchs.bench_fw.benchmark import Benchmark
from faiss.benchs.bench_fw.index import IndexFromFactory

# 1. 准备数据集
ds = SyntheticDataset(
    d=128,           # 向量维度
    nt=10000,        # 训练集大小
    nb=1000000,      # 数据库大小
    nq=1000          # 查询集大小
)

# 2. 创建基准测试对象
bm = Benchmark(
    num_threads=16,                    # 线程数
    training_vectors=ds.get_train(),
    database_vectors=ds.get_database(),
    query_vectors=ds.get_queries(),
    index_descs=[
        "IVF1024,Flat",
        "IVF4096,PQ32",
        "HNSW32"
    ],
    k=10,                              # 返回k个最近邻
    distance_metric="L2"
)

# 3. 运行基准测试
results = bm.benchmark(
    result_file="benchmark_results.json",
    train=True,           # 是否训练
    knn=True,             # KNN搜索
    range_search=False    # 范围搜索
)

# 4. 结果分析
from faiss.benchs.bench_fw.benchmark_io import BenchmarkIO
bio = BenchmarkIO(
    d=ds.d,
    metric=ds.metric,
    k=10
)
bio.set_io(results)

# 输出性能表格
print(bio.print_results())
```

#### 参数优化器

```python
from faiss.benchs.bench_fw.optimize import Optimizer, ParetoMode

# 1. 定义搜索空间
index_desc = "IVF4096,PQ64"
param_space = {
    "nprobe": [1, 4, 8, 16, 32, 64, 128],
    "ht": [64, 128, 256]  # 量化器efSearch
}

# 2. 创建优化器
optimizer = Optimizer(
    d=ds.d,
    metric="L2",
    k=10
)

# 3. 执行参数扫描
best_params = optimizer.optimize(
    xt=ds.get_train(),
    xb=ds.get_database(),
    xq=ds.get_queries(),
    gt=ds.get_groundtruth(k=10),
    index_desc=index_desc,
    param_space=param_space,
    min_accuracy=0.90,      # 最低召回率
    max_time_ms=10.0,       # 最大延迟
    pareto_mode=ParetoMode.TIME  # Pareto前沿模式
)

print(f"最优参数: {best_params}")
```

### 3.2 经典基准测试脚本

#### bench_gpu_sift1m.py - GPU快速验证

```bash
cd /home/dev/faiss/benchs

# 运行SIFT1M GPU基准测试
python bench_gpu_sift1m.py

# 输出示例:
# IndexFlatL2:  0.234 ms, R@1=1.000
# IndexIVFFlat: 0.089 ms, R@1=0.998
# IndexIVFPQ:   0.023 ms, R@1=0.956
```

**脚本关键内容:**

```python
# benchs/bench_gpu_sift1m.py 核心逻辑

import faiss
from faiss.contrib.datasets import SIFTDataset

# 加载SIFT1M数据集
ds = SIFTDataset()
xb = ds.get_database()  # 1M向量
xq = ds.get_queries()    # 10K查询
gt = ds.get_groundtruth(k=10)

# 测试多种GPU索引
for index_key in ["Flat", "IVF4096,Flat", "IVF1024,PQ64"]:
    # 创建CPU索引
    cpu_index = faiss.index_factory(ds.d, index_key)
    cpu_index.train(xb)
    cpu_index.add(xb)

    # 转换到GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    # 预热
    gpu_index.search(xq[:10], 10)

    # 基准测试
    t0 = time.time()
    D, I = gpu_index.search(xq, 10)
    elapsed = time.time() - t0

    # 计算召回率
    recall = (I[:, :1] == gt[:, :1]).sum() / len(xq)

    print(f"{index_key:20} {elapsed/len(xq)*1000:.3f} ms, R@1={recall:.3f}")
```

#### bench_all_ivf - IVF全景测试

```bash
cd /home/dev/faiss/benchs/bench_all_ivf

# 测试单个索引配置
python bench_all_ivf.py \
    --indexkey "IVF4096,PQ32" \
    --dataset SIFT1M \
    --maxtrain 100000 \
    --searchthreads 8

# 批量测试多种配置
python bench_all_ivf.py \
    --indexkey "IVF1024,Flat" "IVF4096,PQ64" "IVF16384,PQ32np" \
    --dataset Deep1B \
    --compute_gt  # 计算ground truth

# 生成性能对比图
python parse_bench_all_ivf.py results.txt --output results.png
```

**输出格式:**

```
Index: IVF4096,PQ32
build_time: 12.34 s
nq: 10000
nprobe  R@1     R@10    R@100   time(ms)  %pass
1       0.1234  0.2345  0.2456  0.123     8.45
4       0.2345  0.4123  0.4234  0.234     15.67
8       0.3456  0.5678  0.5789  0.345     24.56
16      0.4567  0.7123  0.7234  0.567     38.90
```

### 3.3 专用算法基准测试

#### HNSW图索引性能测试

```bash
cd /home/dev/faiss/perf_tests

# 运行HNSW基准测试
python bench_hnsw.py \
    --M 32 \                    # 图连接度
    --ef-construction 40 \       # 构建efSearch
    --ef-search 16 32 64 128 \  # 搜索efSearch范围
    --num-threads 8 \
    --num-repetitions 5 \
    --dataset SIFT1M
```

**性能分析输出:**

```
HNSW Parameters: M=32, efConstruction=40
Build time: 45.6 s
Index size: 256.3 MB

efSearch  R@1    R@10   Time(ms)  CPU(ms)  Distances
16        0.678  0.889  0.234     0.456    12345
32        0.834  0.967  0.456     0.789    23456
64        0.923  0.991  0.789     1.234    34567
128       0.978  0.998  1.234     2.345    45678
```

#### RaBitQ量化性能测试

```bash
cd /home/dev/faiss/benchs

python bench_rabitq.py \
    --nbit 4 \                # 量化位数
    --max_iter 100 \          # 训练迭代次数
    --dataset SIFT1M
```

### 3.4 分布式大规模基准测试

#### 1T级数据集测试流程

```bash
cd /home/dev/faiss/benchs/distributed_ondisk

# 1. 分布式K-means聚类
python distributed_kmeans.py \
    --input /data/1T_vectors \
    --ncentroids 1000000 \
    --niter 25 \
    --nodes node1,node2,node3,node4

# 2. 构建分片索引
python build_index_shards.py \
    --input /data/1T_vectors \
    --output /data/index_shards \
    --shard_size 10000000

# 3. 启动搜索服务器
python search_server.py \
    --index_path /data/index_shards \
    --port 8000

# 4. 性能测试
python benchmark_distributed.py \
    --server http://localhost:8000 \
    --queries /data/test_queries.npy \
    --k 100
```

---

## 4. 外部Profiling工具实战

### 4.1 Linux perf工具

#### 基础性能分析

```bash
# 1. 编译Release版本
cd /home/dev/faiss
cmake -DCMAKE_BUILD_TYPE=Release -B build .
make -C build -j faiss

# 2. 编译测试程序
cat > test_search.cpp << 'EOF'
#include <faiss/IndexFlat.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

int main() {
    int d = 128, nb = 1000000, nq = 10000;
    faiss::IndexFlatL2 index(d);

    std::vector<float> xb(nb * d);
    faiss::float_rand(xb.data(), nb * d, 12345);
    index.add(nb, xb.data());

    std::vector<float> xq(nq * d);
    faiss::float_rand(xq.data(), nq * d, 54321);

    std::vector<float> distances(nq * 10);
    std::vector<faiss::idx_t> labels(nq * 10);

    double t0 = faiss::getmillisecs();
    index.search(nq, xq.data(), 10, distances.data(), labels.data());
    double elapsed = faiss::getmillisecs() - t0;

    printf("Search time: %.3f ms\n", elapsed);
    return 0;
}
EOF

g++ -O3 -mavx2 -o test_search test_search.cpp \
    -I build/_deps/faiss-src \
    -L build/faiss -lfaiss \
    -fopenmp -lpthread

# 3. CPU热点分析
perf record -g -F 99 ./test_search
perf report

# 输出示例:
#   65.23%  test_search  libfaiss.so  [.] fvec_L2sqr_batch_4_avx2
#   12.45%  test_search  libfaiss.so  [.] heap_addn
#    8.67%  test_search  libfaiss.so  [.] IndexFlat::search
```

#### 详细性能计数器

```bash
# 缓存性能分析
perf stat -e \
    cache-references,cache-misses,\
    L1-dcache-loads,L1-dcache-load-misses,\
    LLC-loads,LLC-load-misses \
    ./test_search

# 输出示例:
#   2,345,678,901  cache-references
#     234,567,890  cache-misses     # 10% cache miss rate
#  45,678,901,234  L1-dcache-loads
#   1,234,567,890  L1-dcache-load-misses  # 2.7%
#     567,890,123  LLC-loads
#      56,789,012  LLC-load-misses  # 10%
```

```bash
# 分支预测性能
perf stat -e \
    branches,branch-misses,\
    instructions,cycles \
    ./test_search

# 输出示例:
#  34,567,890,123  instructions  # 1.23 IPC
#  28,123,456,789  cycles
#   5,678,901,234  branches
#      123,456,789  branch-misses  # 2.17% miss rate
```

#### 火焰图生成

```bash
# 1. 安装FlameGraph工具
git clone https://github.com/brendangregg/FlameGraph.git

# 2. 采集数据
perf record -F 99 -g ./test_search

# 3. 生成火焰图
perf script | FlameGraph/stackcollapse-perf.pl | \
    FlameGraph/flamegraph.pl > flamegraph.svg

# 在浏览器中查看 flamegraph.svg
```

### 4.2 Intel VTune Profiler

#### Hotspots分析

```bash
# 1. 安装VTune (需要Intel oneAPI)
source /opt/intel/oneapi/vtune/latest/env/vars.sh

# 2. 热点分析
vtune -collect hotspots -r vtune_results ./test_search

# 3. 查看报告
vtune -report hotspots -r vtune_results -format text

# 4. GUI查看
vtune-gui vtune_results/vtune_results.vtune
```

**输出示例:**

```
Function                          CPU Time  % of Total
fvec_L2sqr_batch_4_avx2           15.234s   62.3%
heap_addn                         3.456s    14.1%
IndexFlat::search                 2.123s     8.7%
```

#### 微架构分析

```bash
# CPU微架构瓶颈分析
vtune -collect uarch-exploration -r vtune_uarch ./test_search

# 查看关键指标:
# - Frontend Bound: 指令获取瓶颈
# - Backend Bound: 执行单元瓶颈
#   - Memory Bound: 内存延迟
#   - Core Bound: 计算单元饱和
# - Retiring: 有效指令比例
# - Bad Speculation: 分支预测失败
```

### 4.3 Valgrind Cachegrind

#### 缓存行为分析

```bash
# 1. 运行cachegrind
valgrind --tool=cachegrind \
    --cache-sim=yes \
    --branch-sim=yes \
    ./test_search

# 输出文件: cachegrind.out.<pid>

# 2. 查看统计
cg_annotate cachegrind.out.<pid>

# 输出示例:
# I   refs:      12,345,678,901
# I1  misses:        12,345,678
# LLi misses:         1,234,567
# I1  miss rate:           0.10%
# LLi miss rate:           0.01%
#
# D   refs:      45,678,901,234  (34B + 11.6B writes)
# D1  misses:     1,234,567,890  ( 1.2B + 34M writes)
# LLd misses:       123,456,789  (  90M + 33M writes)
# D1  miss rate:            2.7% (  3.5% + 0.3%)
# LLd miss rate:            0.3% (  0.3% + 0.3%)

# 3. 源代码级注解
cg_annotate --auto=yes cachegrind.out.<pid> test_search.cpp
```

### 4.4 gperftools (Google Profiler)

#### CPU Profiler

```bash
# 1. 安装gperftools
sudo apt-get install libgoogle-perftools-dev

# 2. 编译时链接
g++ -O3 -o test_search test_search.cpp \
    -lfaiss -lprofiler -lpthread

# 3. 运行profiler
CPUPROFILE=test_search.prof ./test_search

# 4. 查看报告
google-pprof --text ./test_search test_search.prof

# 5. 生成调用图
google-pprof --pdf ./test_search test_search.prof > profile.pdf
```

#### Heap Profiler (内存分配)

```bash
# 运行heap profiler
HEAPPROFILE=test_search.heap ./test_search

# 查看内存分配热点
google-pprof --text ./test_search test_search.heap.0001.heap
```

### 4.5 nvprof / Nsight Compute (GPU)

#### CUDA Profiling

```bash
# 1. 使用nvprof (旧版)
nvprof --print-gpu-trace ./demo_ivfpq_indexing_gpu

# 2. 使用Nsight Compute (推荐)
ncu --set full -o profile_results ./demo_ivfpq_indexing_gpu

# 3. 查看报告
ncu-ui profile_results.ncu-rep
```

**关键指标:**

```
Kernel: GpuIndexIVFPQ::search
Duration: 2.345 ms
SM Efficiency: 78.9%
Memory Throughput: 456 GB/s (82% of peak)
Occupancy: 68.7%

Bottlenecks:
- Memory Latency (45.6%)
- Compute (32.1%)
```

### 4.6 自定义Profiling包装器

```python
# faiss_profiler.py - 完整的Python profiling工具

import time
import functools
import cProfile
import pstats
import io
from contextlib import contextmanager

class FaissProfiler:
    """Faiss操作profiler"""

    def __init__(self):
        self.timings = {}

    @contextmanager
    def timer(self, name):
        """上下文管理器计时"""
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self.timings.setdefault(name, []).append(elapsed)

    def time_function(self, func):
        """装饰器计时"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.timer(func.__name__):
                return func(*args, **kwargs)
        return wrapper

    def report(self):
        """输出性能报告"""
        print("\n=== Faiss Performance Report ===")
        for name, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            total = sum(times)
            count = len(times)
            print(f"{name:30} {avg*1000:8.3f} ms avg, "
                  f"{total*1000:10.3f} ms total, {count:5} calls")

# 使用示例
profiler = FaissProfiler()

with profiler.timer("index_build"):
    index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
    index.train(xt)
    index.add(xb)

with profiler.timer("search"):
    D, I = index.search(xq, k)

profiler.report()
```

---

## 5. 性能瓶颈诊断方法

### 5.1 系统化诊断流程

#### 5步诊断法

```
1. 建立性能基线
   ├─ 测量当前QPS/延迟
   ├─ 记录召回率
   └─ 监控资源使用

2. 定位瓶颈层级
   ├─ CPU-bound?
   ├─ Memory-bound?
   ├─ IO-bound?
   └─ 算法问题?

3. 细化瓶颈位置
   ├─ 哪个函数最慢?
   ├─ 哪个阶段占时最多?
   └─ 哪个参数影响最大?

4. 分析根本原因
   ├─ 缓存miss?
   ├─ 分支预测失败?
   ├─ 数据依赖?
   └─ 参数配置不当?

5. 验证优化效果
   ├─ 对比优化前后
   ├─ 确认召回率无降低
   └─ 检查资源利用率
```

### 5.2 常见瓶颈模式识别

#### CPU Bound识别

```python
import psutil
import time

# 监控CPU使用率
def monitor_cpu_during_search(index, xq, k, duration=10):
    results = []

    def worker():
        while time.time() < end_time:
            index.search(xq, k)

    # 启动搜索线程
    import threading
    end_time = time.time() + duration
    thread = threading.Thread(target=worker)

    # 监控CPU
    thread.start()
    while thread.is_alive():
        cpu_percent = psutil.cpu_percent(interval=0.1)
        results.append(cpu_percent)
    thread.join()

    avg_cpu = sum(results) / len(results)
    print(f"平均CPU使用率: {avg_cpu:.1f}%")

    if avg_cpu > 90:
        print("诊断: CPU-bound, 考虑:")
        print("  - 使用更快的索引类型(PQ vs Flat)")
        print("  - 增加量化压缩(降低计算量)")
        print("  - 优化SIMD指令使用")
    elif avg_cpu < 50:
        print("诊断: 非CPU-bound, 可能是:")
        print("  - Memory-bound (检查缓存miss率)")
        print("  - IO-bound (检查磁盘/网络)")
        print("  - 线程利用不足")
```

#### Memory Bound识别

```bash
# 使用perf检查缓存miss
perf stat -e \
    cache-references,cache-misses,\
    L1-dcache-load-misses,\
    LLC-load-misses \
    ./test_search

# 判断标准:
# - L1 miss rate > 5%: 考虑改进数据局部性
# - LLC miss rate > 10%: 严重内存瓶颈
# - Cache miss > 15%: 考虑减少数据量或预取
```

**Python监控:**

```python
# 使用perf_counter和IVF统计推断
faiss.cvar.indexIVF_stats.reset()
t0 = time.perf_counter()
D, I = index.search(xq, k)
elapsed = time.perf_counter() - t0

stats = faiss.cvar.indexIVF_stats
ndis_per_sec = stats.ndis / elapsed
print(f"距离计算速率: {ndis_per_sec/1e9:.2f} G distances/s")

# 对于AVX2 FP32, 理论峰值约 100-200 Gdist/s
# 如果实测 < 50 Gdist/s, 可能是memory-bound
```

#### IO Bound识别

```python
import time

# 对比内存索引 vs OnDisk索引
def compare_memory_vs_disk(index_file):
    # 1. 内存索引
    index_mem = faiss.read_index(index_file)
    t0 = time.time()
    D1, I1 = index_mem.search(xq, k)
    time_mem = time.time() - t0

    # 2. OnDisk索引
    index_disk = faiss.read_index(index_file, faiss.IO_FLAG_MMAP)
    t0 = time.time()
    D2, I2 = index_disk.search(xq, k)
    time_disk = time.time() - t0

    print(f"内存索引: {time_mem:.3f}s")
    print(f"磁盘索引: {time_disk:.3f}s")
    print(f"slowdown: {time_disk/time_mem:.2f}x")

    if time_disk / time_mem > 2:
        print("诊断: IO-bound, 考虑:")
        print("  - 增加系统内存,避免swap")
        print("  - 使用SSD而非HDD")
        print("  - 预加载热数据到内存")
        print("  - 使用更激进的压缩(PQ)")
```

### 5.3 分阶段性能剖析

#### IVF搜索各阶段分析

```python
# 详细计时IVF搜索各阶段
import faiss
import numpy as np

def profile_ivf_stages(index, xq, k, nprobe):
    """分析IVF搜索的各个阶段"""
    assert isinstance(index, faiss.IndexIVF)

    index.nprobe = nprobe
    nq = xq.shape[0]

    # 1. 量化阶段
    faiss.cvar.indexIVF_stats.reset()
    t0 = time.perf_counter()
    coarse_dis, coarse_ids = index.quantizer.search(xq, nprobe)
    quantization_time = time.perf_counter() - t0

    # 2. 列表扫描阶段 (近似, 通过统计)
    faiss.cvar.indexIVF_stats.reset()
    t0 = time.perf_counter()
    D, I = index.search(xq, k)
    total_time = time.perf_counter() - t0

    stats = faiss.cvar.indexIVF_stats
    scan_time = stats.search_time - stats.quantization_time

    # 3. 分析
    print(f"\n=== IVF Search Profiling (nprobe={nprobe}) ===")
    print(f"总耗时:       {total_time*1000:.3f} ms")
    print(f"  量化阶段:   {quantization_time*1000:.3f} ms ({quantization_time/total_time*100:.1f}%)")
    print(f"  扫描阶段:   {scan_time:.3f} ms ({scan_time/(total_time*1000)*100:.1f}%)")
    print(f"\n扫描统计:")
    print(f"  扫描列表数: {stats.nlist}")
    print(f"  距离计算:   {stats.ndis:,}")
    print(f"  堆更新:     {stats.nheap_updates:,}")
    print(f"  平均列表大小: {stats.ndis/stats.nlist:.1f}")

    # 瓶颈诊断
    if quantization_time / total_time > 0.5:
        print("\n诊断: 量化成为瓶颈 (>50%)")
        print("  建议: 使用更快的量化器 (如IVFFlat换IndexFlat)")

    if stats.ndis / nq > 100000:
        print("\n诊断: 扫描向量过多 (>100k per query)")
        print("  建议: 减少nprobe或增加nlist")

    return {
        'total_time': total_time,
        'quantization_time': quantization_time,
        'scan_time': scan_time,
        'ndis': stats.ndis,
        'nlist_scanned': stats.nlist
    }

# 使用示例
index = faiss.index_factory(128, "IVF4096,PQ64")
# ... train & add ...
profile_ivf_stages(index, xq, k=10, nprobe=16)
```

**输出示例:**

```
=== IVF Search Profiling (nprobe=16) ===
总耗时:       2.456 ms
  量化阶段:   0.234 ms (9.5%)
  扫描阶段:   2.123 ms (86.5%)

扫描统计:
  扫描列表数: 16,000
  距离计算:   1,234,567
  堆更新:     123,456
  平均列表大小: 77.2

诊断: 扫描向量过多 (>100k per query)
  建议: 减少nprobe或增加nlist
```

### 5.4 参数敏感性分析

```python
def parameter_sensitivity_analysis(index_factory_str, xt, xb, xq, gt, k=10):
    """分析参数对性能的影响"""
    import matplotlib.pyplot as plt

    # 测试不同nprobe值
    nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128]
    results = {'nprobe': [], 'recall': [], 'time': []}

    index = faiss.index_factory(xt.shape[1], index_factory_str)
    index.train(xt)
    index.add(xb)

    for nprobe in nprobe_values:
        index.nprobe = nprobe

        t0 = time.time()
        D, I = index.search(xq, k)
        elapsed = (time.time() - t0) / len(xq) * 1000  # ms per query

        recall = (I[:, 0:1] == gt[:, 0:1]).sum() / len(xq)

        results['nprobe'].append(nprobe)
        results['recall'].append(recall)
        results['time'].append(elapsed)

        print(f"nprobe={nprobe:3}: R@1={recall:.3f}, time={elapsed:.3f} ms")

    # 绘制Recall-Time曲线
    plt.figure(figsize=(10, 6))
    plt.plot(results['time'], results['recall'], 'o-')
    for i, nprobe in enumerate(results['nprobe']):
        plt.annotate(f'nprobe={nprobe}',
                     (results['time'][i], results['recall'][i]))
    plt.xlabel('Time (ms per query)')
    plt.ylabel('Recall@1')
    plt.title('Recall-Time Trade-off')
    plt.grid(True)
    plt.savefig('recall_time_tradeoff.png')

    return results
```

---

## 6. 索引级别性能调优

### 6.1 索引类型选择决策树

```
数据规模?
├─ < 10万
│  ├─ 需要精确搜索? → IndexFlat
│  └─ 可以近似? → IndexHNSW
│
├─ 10万 - 1000万
│  ├─ 内存充足?
│  │  ├─ 是 → IndexHNSW 或 IndexIVFFlat
│  │  └─ 否 → IndexIVFPQ
│  └─ GPU可用? → GpuIndexIVFFlat
│
└─ > 1000万
   ├─ 内存装得下?
   │  ├─ 是 → IndexIVFFastScan
   │  └─ 否 → IndexIVFPQ + OnDisk
   └─ 分布式? → Index Shards + 分布式框架
```

### 6.2 IVF索引调优

#### nlist (聚类数) 优化

```python
def optimize_nlist(xb, xq, gt, d, k=10):
    """找到最优的nlist值"""
    import math

    nb = len(xb)
    # 经验公式: nlist ≈ sqrt(N) 到 4*sqrt(N)
    candidates = [
        int(math.sqrt(nb)),
        int(2 * math.sqrt(nb)),
        int(4 * math.sqrt(nb)),
        int(8 * math.sqrt(nb))
    ]

    results = []
    for nlist in candidates:
        print(f"\nTesting nlist={nlist}...")

        index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(d), d, nlist
        )
        index.train(xb)
        index.add(xb)

        # 测试多个nprobe
        for nprobe in [1, 4, 16, 64]:
            if nprobe > nlist:
                continue

            index.nprobe = nprobe
            t0 = time.time()
            D, I = index.search(xq, k)
            elapsed = (time.time() - t0) / len(xq) * 1000

            recall = (I[:, :1] == gt[:, :1]).sum() / len(xq)

            results.append({
                'nlist': nlist,
                'nprobe': nprobe,
                'recall': recall,
                'time_ms': elapsed
            })
            print(f"  nprobe={nprobe:3}: R@1={recall:.3f}, {elapsed:.3f} ms")

    # 找到最优配置
    import pandas as pd
    df = pd.DataFrame(results)

    # 在recall > 0.9的配置中找最快的
    best = df[df['recall'] >= 0.9].nsmallest(1, 'time_ms')
    print(f"\n推荐配置:")
    print(best)

    return df
```

**规则总结:**

```python
# nlist选择指南
def recommend_nlist(nb):
    """根据数据量推荐nlist"""
    import math

    sqrt_n = int(math.sqrt(nb))

    if nb < 100000:
        return sqrt_n // 2  # 较小的nlist
    elif nb < 1000000:
        return sqrt_n
    elif nb < 10000000:
        return sqrt_n * 2
    else:
        return sqrt_n * 4

    # 调整为2的幂次 (可选, 利于GPU)
    # return 2 ** int(math.log2(nlist) + 0.5)
```

#### nprobe优化

```python
def binary_search_nprobe(index, xq, gt, k, target_recall=0.95):
    """二分查找最小满足recall的nprobe"""

    left, right = 1, index.nlist
    best_nprobe = right

    while left <= right:
        mid = (left + right) // 2
        index.nprobe = mid

        D, I = index.search(xq, k)
        recall = (I[:, :1] == gt[:, :1]).sum() / len(xq)

        print(f"nprobe={mid}: recall={recall:.3f}")

        if recall >= target_recall:
            best_nprobe = mid
            right = mid - 1  # 尝试更小的值
        else:
            left = mid + 1   # 需要更大的值

    print(f"\n最优nprobe: {best_nprobe} (recall >= {target_recall})")
    return best_nprobe
```

### 6.3 PQ (Product Quantization) 调优

#### M (子向量数) 和 nbits 选择

```python
def optimize_pq_parameters(d, nb, memory_budget_mb):
    """优化PQ参数: M和nbits"""

    # 约束: M必须能整除d
    valid_M = [m for m in range(8, d+1, 8) if d % m == 0]

    results = []
    for M in valid_M:
        for nbits in [8, 10, 12]:
            # 计算内存使用
            # 码本: M * 2^nbits * (d/M) * 4 bytes
            codebook_mb = M * (2**nbits) * (d/M) * 4 / 1024 / 1024
            # 码值: nb * M * nbits / 8 bytes
            codes_mb = nb * M * nbits / 8 / 1024 / 1024
            total_mb = codebook_mb + codes_mb

            if total_mb <= memory_budget_mb:
                results.append({
                    'M': M,
                    'nbits': nbits,
                    'memory_mb': total_mb,
                    'compression': d * 4 / (M * nbits / 8)
                })

    # 按压缩率排序
    results.sort(key=lambda x: x['compression'], reverse=True)

    print(f"维度={d}, 数据量={nb:,}, 内存预算={memory_budget_mb} MB\n")
    print(f"{'M':>4} {'nbits':>6} {'Memory(MB)':>12} {'Compression':>12}")
    print("-" * 40)
    for r in results[:10]:
        print(f"{r['M']:4} {r['nbits']:6} {r['memory_mb']:12.1f} {r['compression']:12.1f}x")

    return results

# 示例
optimize_pq_parameters(d=128, nb=1000000, memory_budget_mb=100)
```

**输出示例:**

```
维度=128, 数据量=1,000,000, 内存预算=100 MB

   M  nbits   Memory(MB)  Compression
----------------------------------------
  64      8         64.5         64.0x
  32      8         32.8         32.0x
  16     10         40.6         20.5x
   8     12         48.4         10.7x
```

**经验规则:**

```python
# PQ参数快速推荐
def recommend_pq_params(d, target_compression):
    """
    target_compression: 目标压缩率
        - 32x: 高压缩,适合大规模数据
        - 16x: 平衡
        - 8x:  高精度
    """
    if target_compression >= 32:
        M = min(64, d // 2)
        nbits = 8
    elif target_compression >= 16:
        M = min(32, d // 4)
        nbits = 8
    else:
        M = min(16, d // 8)
        nbits = 10

    # 确保M能整除d
    while d % M != 0:
        M -= 1

    return M, nbits
```

### 6.4 HNSW调优

#### M和efConstruction优化

```python
def hnsw_parameter_sweep(xb, xq, gt, d, k=10):
    """HNSW参数扫描"""

    # M: 图连接度, 推荐范围 16-64
    # efConstruction: 构建时搜索深度, 推荐范围 40-500

    configs = [
        (16, 40),
        (32, 40),
        (32, 80),
        (32, 200),
        (64, 200),
    ]

    results = []
    for M, efC in configs:
        print(f"\n=== M={M}, efConstruction={efC} ===")

        index = faiss.IndexHNSWFlat(d, M)
        index.hnsw.efConstruction = efC

        # 构建
        t0 = time.time()
        index.add(xb)
        build_time = time.time() - t0

        # 测试不同efSearch
        for efS in [16, 32, 64, 128]:
            index.hnsw.efSearch = efS

            t0 = time.time()
            D, I = index.search(xq, k)
            search_time = (time.time() - t0) / len(xq) * 1000

            recall = (I[:, :1] == gt[:, :1]).sum() / len(xq)

            results.append({
                'M': M,
                'efConstruction': efC,
                'efSearch': efS,
                'build_time_s': build_time,
                'search_time_ms': search_time,
                'recall': recall
            })

            print(f"efSearch={efS:3}: R@1={recall:.3f}, {search_time:.3f} ms")

    import pandas as pd
    df = pd.DataFrame(results)

    # 分析: 在构建时间 < 60s, recall > 0.95 的配置中找最快的
    valid = df[(df['build_time_s'] < 60) & (df['recall'] > 0.95)]
    if len(valid) > 0:
        best = valid.nsmallest(1, 'search_time_ms')
        print(f"\n推荐配置:")
        print(best)

    return df
```

**快速推荐:**

```python
def recommend_hnsw_params(nb, latency_requirement):
    """
    nb: 数据量
    latency_requirement: 'low' | 'medium' | 'high'
    """
    if latency_requirement == 'low':  # < 1ms
        M = 16
        efConstruction = 40
        efSearch = 16
    elif latency_requirement == 'medium':  # 1-5ms
        M = 32
        efConstruction = 80
        efSearch = 32
    else:  # 'high': > 5ms, 追求高召回
        M = 64
        efConstruction = 200
        efSearch = 128

    # 大数据集调整
    if nb > 10000000:
        M = min(M, 32)  # 限制M避免内存爆炸

    return M, efConstruction, efSearch
```

### 6.5 FastScan优化

```python
# FastScan专为SIMD扫描优化
def optimize_fastscan(d, nb, nlist, bbs=32):
    """
    FastScan索引优化
    bbs (binarization block size): 量化块大小, 必须是4的倍数
    """

    # 推荐配置
    if d <= 128:
        M = 64  # 每个子向量2位
        bbs = 32
    elif d <= 256:
        M = d // 2
        bbs = 32
    else:
        M = 64
        bbs = 16  # 更小的块用于高维

    index_desc = f"IVF{nlist},PQ{M}x4fsr"  # 4-bit FastScan
    print(f"推荐FastScan配置: {index_desc}, bbs={bbs}")

    index = faiss.index_factory(d, index_desc)
    # 设置bbs
    ivf_index = faiss.downcast_index(index)
    if hasattr(ivf_index, 'bbs'):
        ivf_index.bbs = bbs

    return index
```

---

## 7. SIMD和CPU优化

### 7.1 检查和启用SIMD支持

#### 检查编译选项

```python
import faiss

# 查看编译时的SIMD支持
compile_opts = faiss.get_compile_options()
print(f"编译选项: {compile_opts}")

# 期望看到:
# - x86-64: AVX2, AVX512 (AVX512F, AVX512BW, AVX512_SKX, AVX512_SPR)
# - ARM: NEON, SVE
```

#### 构建不同SIMD版本

```bash
cd /home/dev/faiss

# 1. AVX2版本 (大多数现代x86 CPU)
cmake -B build -DFAISS_OPT_LEVEL=avx2 .
make -C build -j faiss_avx2

# 2. AVX512版本 (Intel Skylake及更新)
cmake -B build -DFAISS_OPT_LEVEL=avx512 .
make -C build -j faiss_avx512

# 3. AVX512_SPR版本 (Intel Sapphire Rapids)
cmake -B build -DFAISS_OPT_LEVEL=avx512_spr .
make -C build -j faiss_avx512_spr

# 4. ARM SVE版本
cmake -B build -DFAISS_OPT_LEVEL=sve .
make -C build -j faiss_sve
```

### 7.2 距离计算优化

#### SIMD距离函数性能对比

```python
# benchs/bench_pairwise_distances.py
import faiss
import numpy as np
import time

d = 128
nb, nq = 10000, 1000

xb = np.random.randn(nb, d).astype('float32')
xq = np.random.randn(nq, d).astype('float32')

# 测试不同距离计算方法
methods = [
    ('L2sqr', faiss.fvec_L2sqr),
    ('InnerProduct', faiss.fvec_inner_product),
]

for name, func in methods:
    t0 = time.time()
    for i in range(nq):
        for j in range(nb):
            _ = func(xq[i], xb[j])
    elapsed = time.time() - t0

    print(f"{name:15} {elapsed:.3f}s, "
          f"{nb*nq/elapsed/1e6:.1f} M计算/s")

# 批量距离计算
print("\n批量计算:")
t0 = time.time()
D = faiss.pairwise_distances(xq, xb, metric=faiss.METRIC_L2)
elapsed = time.time() - t0
print(f"pairwise_distances: {elapsed:.3f}s, "
      f"{nb*nq/elapsed/1e6:.1f} M计算/s")
```

**性能对比 (AVX2 vs AVX512):**

```
CPU: Intel Xeon 8375C (AVX512)
维度: 128

              AVX2        AVX512      加速比
L2sqr         12.3 ms     6.8 ms      1.81x
InnerProd     11.8 ms     6.1 ms      1.93x
PQdist        8.9 ms      4.2 ms      2.12x
```

### 7.3 多线程优化

#### OpenMP线程数调优

```python
import os
import faiss
import numpy as np
import time

def benchmark_threads(index, xq, k, thread_counts):
    """测试不同线程数的性能"""
    results = []

    for num_threads in thread_counts:
        # 设置OpenMP线程数
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        faiss.omp_set_num_threads(num_threads)

        # 预热
        index.search(xq[:10], k)

        # 测试
        t0 = time.time()
        D, I = index.search(xq, k)
        elapsed = time.time() - t0

        qps = len(xq) / elapsed
        results.append({
            'threads': num_threads,
            'time_s': elapsed,
            'qps': qps
        })

        print(f"Threads={num_threads:2}: {elapsed:.3f}s, {qps:.1f} QPS")

    # 计算加速比
    baseline = results[0]['time_s']
    for r in results:
        r['speedup'] = baseline / r['time_s']
        r['efficiency'] = r['speedup'] / r['threads'] * 100

    return results

# 使用示例
index = faiss.read_index("large_index.faiss")
xq = np.random.randn(10000, 128).astype('float32')

results = benchmark_threads(
    index, xq, k=10,
    thread_counts=[1, 2, 4, 8, 16, 32]
)

# 分析结果
import pandas as pd
df = pd.DataFrame(results)
print("\n线程扩展性分析:")
print(df)

# 找到效率 > 70% 的最大线程数
optimal = df[df['efficiency'] > 70]['threads'].max()
print(f"\n推荐线程数: {optimal}")
```

**典型输出:**

```
Threads= 1: 8.234s, 1214.6 QPS
Threads= 2: 4.321s, 2314.1 QPS
Threads= 4: 2.234s, 4476.3 QPS
Threads= 8: 1.156s, 8650.5 QPS
Threads=16: 0.723s, 13831.3 QPS
Threads=32: 0.589s, 16977.9 QPS

线程扩展性分析:
   threads   time_s      qps  speedup  efficiency
0        1    8.234  1214.6    1.00      100.0
1        2    4.321  2314.1    1.91       95.3
2        4    2.234  4476.3    3.69       92.1
3        8    1.156  8650.5    7.12       89.0
4       16    0.723 13831.3   11.39       71.2
5       32    0.589 16977.9   13.98       43.7

推荐线程数: 16
```

### 7.4 CPU亲和性和NUMA优化

```python
import os
import subprocess

def set_cpu_affinity():
    """绑定CPU核心避免跨NUMA节点"""

    # 获取NUMA拓扑
    numa_info = subprocess.check_output(['numactl', '--hardware'])
    print(numa_info.decode())

    # 绑定到NUMA node 0
    os.environ['OMP_PROC_BIND'] = 'TRUE'
    os.environ['OMP_PLACES'] = 'cores'

    # 使用numactl运行
    # numactl --cpunodebind=0 --membind=0 python your_script.py
```

### 7.5 Prefetching和Cache优化

```cpp
// C++ 级别的预取优化
#include <faiss/utils/prefetch.h>

void search_with_prefetch(
    const float* xb,  // 数据库向量
    const float* xq,  // 查询向量
    size_t nb, size_t nq, int d
) {
    for (size_t i = 0; i < nq; i++) {
        // 预取下一个查询向量
        if (i + 1 < nq) {
            faiss::prefetch_L1(xq + (i+1) * d);
        }

        for (size_t j = 0; j < nb; j++) {
            // 预取数据库向量
            if (j + 4 < nb) {
                faiss::prefetch_L2(xb + (j+4) * d);
            }

            // 计算距离
            float dist = fvec_L2sqr(xq + i*d, xb + j*d, d);
            // ...
        }
    }
}
```

---

## 8. GPU性能优化

### 8.1 GPU索引选择和配置

#### 选择合适的GPU索引

```python
import faiss

# CPU索引 → GPU索引映射
cpu_to_gpu = {
    "IndexFlatL2": "GpuIndexFlatL2",
    "IndexFlatIP": "GpuIndexFlatIP",
    "IndexIVFFlat": "GpuIndexIVFFlat",
    "IndexIVFPQ": "GpuIndexIVFPQ",
}

def create_gpu_index(d, index_type="IVFFlat", nlist=1024):
    """创建GPU索引"""

    # 1. 配置GPU资源
    res = faiss.StandardGpuResources()

    # 2. 设置临时内存
    res.setTempMemory(512 * 1024 * 1024)  # 512MB临时内存

    # 3. 创建索引
    if index_type == "Flat":
        index = faiss.GpuIndexFlatL2(res, d)

    elif index_type == "IVFFlat":
        # CPU训练 + GPU搜索混合模式
        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist)

        # 转到GPU
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    elif index_type == "IVFPQ":
        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, 64, 8)
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    return index, res

# 使用示例
index, res = create_gpu_index(128, "IVFFlat", nlist=4096)
```

#### 多GPU并行

```python
def create_multi_gpu_index(d, ngpus=4):
    """使用多GPU并行搜索"""

    # 方法1: 使用index_cpu_to_all_gpus
    cpu_index = faiss.IndexFlatL2(d)
    # ... train & add ...

    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

    # 方法2: 手动分片到多个GPU
    res = [faiss.StandardGpuResources() for i in range(ngpus)]

    # 创建replica索引
    replicas = []
    for i in range(ngpus):
        replica = faiss.index_cpu_to_gpu(res[i], i, cpu_index)
        replicas.append(replica)

    # 使用IndexReplicas分布查询
    index = faiss.IndexReplicas()
    for replica in replicas:
        index.addIndex(replica)

    return index
```

### 8.2 GPU内存管理

```python
import faiss

def optimize_gpu_memory(index, res):
    """优化GPU内存使用"""

    # 1. 设置临时内存池
    res.setTempMemory(1024 * 1024 * 1024)  # 1GB

    # 2. 使用Float16降低内存 (损失少量精度)
    if hasattr(index, 'useFloat16'):
        index.useFloat16 = True
        print("启用Float16模式")

    # 3. 监控GPU内存
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def print_gpu_memory():
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU内存: {info.used/1024**3:.2f} GB / "
              f"{info.total/1024**3:.2f} GB "
              f"({info.used/info.total*100:.1f}%)")

    print_gpu_memory()

    return res
```

### 8.3 GPU性能Profiling

```python
import faiss
import time

def gpu_performance_breakdown(index, xq, k):
    """GPU性能详细分析"""

    # CUDA事件计时
    import torch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 1. 数据传输到GPU
    start.record()
    xq_gpu = torch.from_numpy(xq).cuda()
    end.record()
    torch.cuda.synchronize()
    h2d_time = start.elapsed_time(end)

    # 2. GPU搜索
    start.record()
    D, I = index.search(xq, k)
    end.record()
    torch.cuda.synchronize()
    search_time = start.elapsed_time(end)

    # 3. 结果传输到CPU
    start.record()
    D_cpu = D  # 已经是numpy array
    end.record()
    torch.cuda.synchronize()
    d2h_time = start.elapsed_time(end)

    print(f"=== GPU性能分析 ===")
    print(f"Host→Device: {h2d_time:.3f} ms ({h2d_time/(h2d_time+search_time+d2h_time)*100:.1f}%)")
    print(f"GPU搜索:     {search_time:.3f} ms ({search_time/(h2d_time+search_time+d2h_time)*100:.1f}%)")
    print(f"Device→Host: {d2h_time:.3f} ms ({d2h_time/(h2d_time+search_time+d2h_time)*100:.1f}%)")
    print(f"总计:        {h2d_time+search_time+d2h_time:.3f} ms")

    # 4. 使用nvprof进行详细分析
    print("\n运行nvprof进行详细分析:")
    print("nvprof --print-gpu-trace python your_script.py")
```

### 8.4 GPU vs CPU性能对比

```python
def compare_gpu_cpu_performance(d, nb, nq, k, index_type="IVFFlat"):
    """GPU vs CPU性能对比"""

    import numpy as np

    xb = np.random.randn(nb, d).astype('float32')
    xt = xb[:max(10000, nb//10)]
    xq = np.random.randn(nq, d).astype('float32')

    # CPU索引
    print("=== CPU Index ===")
    cpu_index = faiss.index_factory(d, f"IVF1024,Flat")
    cpu_index.train(xt)

    t0 = time.time()
    cpu_index.add(xb)
    cpu_add_time = time.time() - t0

    cpu_index.nprobe = 16
    t0 = time.time()
    D_cpu, I_cpu = cpu_index.search(xq, k)
    cpu_search_time = time.time() - t0

    print(f"Add time: {cpu_add_time:.3f}s")
    print(f"Search time: {cpu_search_time*1000:.3f}ms")
    print(f"QPS: {nq/cpu_search_time:.1f}")

    # GPU索引
    print("\n=== GPU Index ===")
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    t0 = time.time()
    gpu_index.add(xb)
    gpu_add_time = time.time() - t0

    gpu_index.nprobe = 16
    # 预热
    gpu_index.search(xq[:10], k)

    t0 = time.time()
    D_gpu, I_gpu = gpu_index.search(xq, k)
    gpu_search_time = time.time() - t0

    print(f"Add time: {gpu_add_time:.3f}s")
    print(f"Search time: {gpu_search_time*1000:.3f}ms")
    print(f"QPS: {nq/gpu_search_time:.1f}")

    # 对比
    print("\n=== Speedup ===")
    print(f"Add加速: {cpu_add_time/gpu_add_time:.2f}x")
    print(f"Search加速: {cpu_search_time/gpu_search_time:.2f}x")

    # 验证结果一致性
    accuracy = (I_cpu == I_gpu).mean()
    print(f"结果一致性: {accuracy*100:.1f}%")

# 运行对比
compare_gpu_cpu_performance(
    d=128, nb=1000000, nq=10000, k=10
)
```

**典型输出:**

```
=== CPU Index ===
Add time: 2.345s
Search time: 234.567ms
QPS: 42630.5

=== GPU Index ===
Add time: 0.456s
Search time: 12.345ms
QPS: 810372.8

=== Speedup ===
Add加速: 5.14x
Search加速: 19.00x
结果一致性: 99.8%
```

---

## 9. 内存和IO优化

### 9.1 内存占用优化

#### 索引压缩技巧

```python
def compare_index_sizes(d, nb):
    """对比不同索引的内存占用"""

    import sys
    xb = np.random.randn(nb, d).astype('float32')

    configs = [
        ("Flat", "Flat"),
        ("IVF4K,Flat", "IVF4096,Flat"),
        ("IVF4K,PQ32", "IVF4096,PQ32"),
        ("IVF4K,PQ64", "IVF4096,PQ64"),
        ("HNSW32", "HNSW32"),
    ]

    results = []
    for name, desc in configs:
        index = faiss.index_factory(d, desc)
        if desc != "Flat":
            index.train(xb)
        index.add(xb)

        # 计算内存占用
        serialized = faiss.serialize_index(index)
        size_mb = len(serialized) / 1024 / 1024

        # 理论原始大小
        raw_size_mb = nb * d * 4 / 1024 / 1024
        compression = raw_size_mb / size_mb

        results.append({
            'index': name,
            'size_mb': size_mb,
            'compression': compression
        })

        print(f"{name:15} {size_mb:8.1f} MB  {compression:6.1f}x压缩")

    return results
```

**输出示例 (d=128, nb=1M):**

```
Flat            512.0 MB    1.0x压缩
IVF4K,Flat      514.5 MB    1.0x压缩
IVF4K,PQ32       40.2 MB   12.7x压缩
IVF4K,PQ64       72.5 MB    7.1x压缩
HNSW32          896.3 MB    0.6x压缩 (图结构开销)
```

#### OnDiskInvertedLists

```python
def create_ondisk_index(d, nb, index_path="large_index.faiss"):
    """创建磁盘上的IVF索引"""

    # 1. 训练阶段在内存中
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, 16384, 64, 8)

    xt = get_training_data()  # 获取训练数据
    index.train(xt)

    # 2. 替换为OnDiskInvertedLists
    invlists = faiss.OnDiskInvertedLists(
        index.nlist,
        index.code_size,
        index_path + ".ivfdata"
    )
    index.replace_invlists(invlists)

    # 3. 批量添加向量
    batch_size = 100000
    for i in range(0, nb, batch_size):
        xb_batch = get_data_batch(i, batch_size)
        index.add(xb_batch)
        print(f"Added {i+len(xb_batch):,} vectors")

    # 4. 保存索引
    faiss.write_index(index, index_path)

    print(f"索引保存到: {index_path}")
    print(f"数据文件: {index_path}.ivfdata")

    return index
```

### 9.2 IO性能优化

#### 索引序列化和加载

```python
import time
import faiss

def benchmark_io(index_path):
    """测试索引加载性能"""

    # 1. 标准加载 (全部读入内存)
    print("=== 标准加载 ===")
    t0 = time.time()
    index = faiss.read_index(index_path)
    load_time = time.time() - t0
    print(f"加载时间: {load_time:.3f}s")

    # 测试搜索
    xq = np.random.randn(100, index.d).astype('float32')
    t0 = time.time()
    D, I = index.search(xq, 10)
    search_time = time.time() - t0
    print(f"搜索时间: {search_time*1000:.3f}ms")

    # 2. mmap加载 (延迟读取)
    print("\n=== mmap加载 ===")
    t0 = time.time()
    index_mmap = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    mmap_load_time = time.time() - t0
    print(f"加载时间: {mmap_load_time:.3f}s (快 {load_time/mmap_load_time:.1f}x)")

    # 首次搜索 (触发实际读取)
    t0 = time.time()
    D2, I2 = index_mmap.search(xq, 10)
    first_search_time = time.time() - t0
    print(f"首次搜索: {first_search_time*1000:.3f}ms")

    # 后续搜索 (缓存命中)
    t0 = time.time()
    D3, I3 = index_mmap.search(xq, 10)
    cached_search_time = time.time() - t0
    print(f"缓存搜索: {cached_search_time*1000:.3f}ms")

# 使用示例
benchmark_io("large_index.faiss")
```

#### 预加载关键数据

```python
def preload_index_data(index):
    """预加载索引关键数据到内存"""

    if isinstance(index, faiss.IndexIVF):
        # 预加载量化器
        if hasattr(index.quantizer, 'xb'):
            _ = np.array(index.quantizer.xb, copy=False)
            print("量化器已预加载")

        # 预加载IVF列表元数据
        for i in range(index.nlist):
            _ = index.invlists.list_size(i)
        print(f"IVF列表元数据已预加载 ({index.nlist} lists)")

    elif isinstance(index, faiss.IndexHNSW):
        # HNSW图结构通常已在内存
        print("HNSW图结构已在内存")
```

### 9.3 批处理优化

```python
def batch_search_optimization(index, queries, k, batch_size=1000):
    """批量搜索优化"""

    nq = len(queries)
    all_D = []
    all_I = []

    # 方法1: 大批量搜索 (利用并行)
    print("=== 大批量搜索 ===")
    t0 = time.time()
    for i in range(0, nq, batch_size):
        batch = queries[i:i+batch_size]
        D, I = index.search(batch, k)
        all_D.append(D)
        all_I.append(I)
    batch_time = time.time() - t0

    D_batch = np.vstack(all_D)
    I_batch = np.vstack(all_I)
    print(f"批量搜索: {batch_time:.3f}s, {nq/batch_time:.1f} QPS")

    # 方法2: 单次搜索 (最优)
    print("\n=== 单次搜索 ===")
    t0 = time.time()
    D_single, I_single = index.search(queries, k)
    single_time = time.time() - t0
    print(f"单次搜索: {single_time:.3f}s, {nq/single_time:.1f} QPS")
    print(f"加速: {batch_time/single_time:.2f}x")

    # 验证结果一致性
    assert np.allclose(D_batch, D_single)
    assert np.array_equal(I_batch, I_single)
```

### 9.4 分片和分布式

```python
def create_sharded_index(d, shard_size=1000000):
    """创建分片索引"""

    # 创建IndexShards
    index = faiss.IndexShards(d)

    # 添加多个子索引
    num_shards = 4
    for i in range(num_shards):
        shard = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(d), d, 4096, 64, 8
        )
        # ... train & add ...
        index.add_shard(shard)
        print(f"Shard {i} added")

    # 搜索时并行查询所有分片
    index.threaded = True  # 启用多线程分片搜索

    return index
```

---

## 10. 生产环境监控方案

### 10.1 实时性能监控

```python
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque

@dataclass
class SearchMetrics:
    """搜索指标"""
    timestamp: float
    latency_ms: float
    num_queries: int
    recall: float = 0.0

class FaissMonitor:
    """Faiss索引实时监控器"""

    def __init__(self, window_size=1000):
        self.metrics: Deque[SearchMetrics] = deque(maxlen=window_size)
        self.lock = threading.Lock()

    def record_search(self, latency_ms, num_queries, recall=0.0):
        """记录一次搜索"""
        with self.lock:
            self.metrics.append(SearchMetrics(
                timestamp=time.time(),
                latency_ms=latency_ms,
                num_queries=num_queries,
                recall=recall
            ))

    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            if not self.metrics:
                return {}

            latencies = [m.latency_ms for m in self.metrics]
            queries = [m.num_queries for m in self.metrics]
            recalls = [m.recall for m in self.metrics if m.recall > 0]

            # 计算时间窗口内的QPS
            time_span = self.metrics[-1].timestamp - self.metrics[0].timestamp
            total_queries = sum(queries)
            qps = total_queries / time_span if time_span > 0 else 0

            return {
                'qps': qps,
                'latency_p50': np.percentile(latencies, 50),
                'latency_p95': np.percentile(latencies, 95),
                'latency_p99': np.percentile(latencies, 99),
                'latency_max': max(latencies),
                'avg_recall': np.mean(recalls) if recalls else 0,
                'num_samples': len(self.metrics)
            }

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print(f"\n=== Faiss Monitor Stats ===")
        print(f"QPS:          {stats.get('qps', 0):.1f}")
        print(f"Latency p50:  {stats.get('latency_p50', 0):.3f} ms")
        print(f"Latency p95:  {stats.get('latency_p95', 0):.3f} ms")
        print(f"Latency p99:  {stats.get('latency_p99', 0):.3f} ms")
        print(f"Latency max:  {stats.get('latency_max', 0):.3f} ms")
        print(f"Avg Recall:   {stats.get('avg_recall', 0):.3f}")
        print(f"Samples:      {stats.get('num_samples', 0)}")

# 使用示例
monitor = FaissMonitor()

def monitored_search(index, xq, k, gt=None):
    """带监控的搜索"""
    t0 = time.perf_counter()
    D, I = index.search(xq, k)
    latency_ms = (time.perf_counter() - t0) * 1000 / len(xq)

    # 计算召回率
    recall = 0.0
    if gt is not None:
        recall = (I[:, :1] == gt[:, :1]).sum() / len(xq)

    monitor.record_search(latency_ms, len(xq), recall)
    return D, I

# 持续监控
def monitoring_thread():
    while True:
        time.sleep(10)
        monitor.print_stats()

threading.Thread(target=monitoring_thread, daemon=True).start()
```

### 10.2 告警系统

```python
class PerformanceAlerter:
    """性能告警系统"""

    def __init__(self,
                 latency_threshold_ms=10.0,
                 qps_threshold=1000,
                 recall_threshold=0.90):
        self.latency_threshold = latency_threshold_ms
        self.qps_threshold = qps_threshold
        self.recall_threshold = recall_threshold

    def check_alerts(self, stats):
        """检查告警条件"""
        alerts = []

        # 延迟告警
        if stats.get('latency_p99', 0) > self.latency_threshold:
            alerts.append({
                'level': 'WARNING',
                'metric': 'latency_p99',
                'value': stats['latency_p99'],
                'threshold': self.latency_threshold,
                'message': f"P99延迟过高: {stats['latency_p99']:.2f}ms > {self.latency_threshold}ms"
            })

        # QPS告警
        if stats.get('qps', 0) < self.qps_threshold:
            alerts.append({
                'level': 'WARNING',
                'metric': 'qps',
                'value': stats['qps'],
                'threshold': self.qps_threshold,
                'message': f"QPS过低: {stats['qps']:.1f} < {self.qps_threshold}"
            })

        # 召回率告警
        if stats.get('avg_recall', 1.0) < self.recall_threshold:
            alerts.append({
                'level': 'CRITICAL',
                'metric': 'recall',
                'value': stats['avg_recall'],
                'threshold': self.recall_threshold,
                'message': f"召回率过低: {stats['avg_recall']:.3f} < {self.recall_threshold}"
            })

        return alerts

    def send_alerts(self, alerts):
        """发送告警"""
        for alert in alerts:
            print(f"[{alert['level']}] {alert['message']}")
            # 这里可以集成真实的告警系统:
            # - 发送邮件
            # - 推送Slack/钉钉
            # - 写入日志
            # - 触发PagerDuty等

# 集成到监控循环
alerter = PerformanceAlerter()

def monitoring_with_alerts():
    while True:
        time.sleep(10)
        stats = monitor.get_stats()
        monitor.print_stats()

        alerts = alerter.check_alerts(stats)
        if alerts:
            alerter.send_alerts(alerts)
```

### 10.3 Prometheus集成

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 定义Prometheus指标
search_requests = Counter('faiss_search_requests_total', 'Total search requests')
search_latency = Histogram('faiss_search_latency_seconds', 'Search latency')
index_size = Gauge('faiss_index_size_bytes', 'Index size in bytes')
recall_gauge = Gauge('faiss_recall', 'Search recall rate')

def instrumented_search(index, xq, k, gt=None):
    """带Prometheus指标的搜索"""
    search_requests.inc(len(xq))

    with search_latency.time():
        D, I = index.search(xq, k)

    if gt is not None:
        recall = (I[:, :1] == gt[:, :1]).sum() / len(xq)
        recall_gauge.set(recall)

    return D, I

# 启动Prometheus HTTP服务器
start_http_server(8000)
print("Prometheus metrics available at http://localhost:8000")
```

### 10.4 日志和追踪

```python
import logging
import json
from datetime import datetime

# 配置日志
logging.basicConfig(
    filename='faiss_performance.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_search_performance(index, xq, k, D, I, gt=None):
    """记录搜索性能日志"""

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'index_type': type(index).__name__,
        'num_queries': len(xq),
        'k': k,
        'dimension': index.d,
    }

    # IVF统计
    if isinstance(index, faiss.IndexIVF):
        stats = faiss.cvar.indexIVF_stats
        log_entry.update({
            'nprobe': index.nprobe,
            'nlist': index.nlist,
            'ndis': stats.ndis,
            'quantization_time_ms': stats.quantization_time,
            'search_time_ms': stats.search_time,
        })

    # 召回率
    if gt is not None:
        recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / len(xq)
        recall_at_k = (I == gt).any(axis=1).sum() / len(xq)
        log_entry.update({
            'recall_at_1': recall_at_1,
            'recall_at_k': recall_at_k,
        })

    logging.info(json.dumps(log_entry))
```

---

## 11. 实战案例分析

### 案例1: 电商搜索系统优化

**场景:**
- 1亿商品向量 (768维, BERT embedding)
- 10K QPS峰值
- P99延迟 < 20ms
- Recall@10 > 95%

**初始方案问题:**

```python
# 初始配置 (性能不达标)
index = faiss.IndexIVFPQ(
    faiss.IndexFlatL2(768),
    768,
    nlist=4096,   # 太小
    M=96,         # 太大
    nbits=8
)
index.nprobe = 32  # 太大

# 问题:
# - P99延迟 45ms
# - QPS仅 3000
```

**优化过程:**

```python
# Step 1: 参数扫描
results = []
for nlist in [16384, 32768, 65536]:
    for M in [48, 64, 96]:
        for nprobe in [8, 16, 32]:
            # ... benchmark ...
            results.append(...)

# Step 2: 选择最优配置
# nlist=32768, M=64, nprobe=16
# P99延迟 18ms, Recall@10 96.2%

# Step 3: 使用FastScan进一步优化
index = faiss.index_factory(768, "IVF32768,PQ64x4fsr")
index.train(xt)
index.add(xb)

# Step 4: 多线程 + GPU加速
faiss.omp_set_num_threads(16)
gpu_index = faiss.index_cpu_to_all_gpus(index)

# 最终结果:
# - P99延迟 12ms
# - QPS 15000
# - Recall@10 96.5%
```

### 案例2: 人脸识别系统

**场景:**
- 1000万人脸特征 (512维)
- 实时识别 (< 100ms)
- 精确匹配 (Recall > 99%)

**方案:**

```python
# 使用HNSW实现高召回
index = faiss.IndexHNSWFlat(512, 32)
index.hnsw.efConstruction = 200
index.add(xb)

# 搜索参数
index.hnsw.efSearch = 64

# 性能:
# - 延迟 45ms
# - Recall@1 99.2%

# 进一步优化: Refine
index_hnsw = faiss.IndexHNSWFlat(512, 32)
index_flat = faiss.IndexFlatL2(512)
index = faiss.IndexRefine(index_hnsw, index_flat)

# 两阶段搜索:
# 1. HNSW快速召回 (k=100)
# 2. Flat精确重排 (k=1)

# 最终性能:
# - 延迟 58ms
# - Recall@1 99.8%
```

### 案例3: 大规模推荐系统

**场景:**
- 10亿视频向量 (256维)
- 离线批量推荐 (1亿用户)
- 吞吐量优先

**方案:**

```python
# 使用OnDiskInvertedLists节省内存
quantizer = faiss.IndexFlatL2(256)
index = faiss.IndexIVFPQ(quantizer, 256, 1000000, 32, 8)

# 训练
index.train(xt)

# 替换为OnDisk
invlists = faiss.OnDiskInvertedLists(
    index.nlist,
    index.code_size,
    "/ssd/index.ivfdata"
)
index.replace_invlists(invlists)

# 批量添加
for batch in data_iterator(batch_size=1000000):
    index.add(batch)

# 分布式批量搜索
from multiprocessing import Pool

def search_worker(queries):
    return index.search(queries, 100)

with Pool(processes=64) as pool:
    results = pool.map(search_worker, query_batches)

# 性能:
# - 内存占用 80GB (OnDisk)
# - 吞吐量 50M queries/hour
```

---

## 12. 性能优化Checklist

### 12.1 算法选择

```
□ 数据规模 < 10万? → 考虑IndexFlat或IndexHNSW
□ 数据规模 10万-1000万? → 考虑IndexIVFFlat或IndexHNSW
□ 数据规模 > 1000万? → 考虑IndexIVFPQ或IndexIVFFastScan
□ 需要精确搜索? → 使用Flat或HNSW高efSearch
□ 内存受限? → 使用PQ压缩或OnDisk
□ GPU可用? → 尝试GPU索引
```

### 12.2 参数调优

```
IVF索引:
□ nlist = sqrt(N) ~ 4*sqrt(N)
□ nprobe: 二分查找最优值 (目标recall)
□ 训练数据 > 30*nlist

PQ:
□ M能整除d
□ M越大压缩率越高,但精度下降
□ nbits: 8 (常用), 10 (高精度), 12 (超高精度)

HNSW:
□ M: 16-64 (更大=更高召回,更多内存)
□ efConstruction: 40-500
□ efSearch: 动态调整至目标recall

FastScan:
□ 使用4-bit PQ (PQ64x4fsr)
□ bbs: 32 (常用), 16 (高维)
```

### 12.3 系统优化

```
CPU:
□ 编译时启用AVX2/AVX512
□ 设置OMP_NUM_THREADS (物理核心数)
□ 使用numactl绑定NUMA节点
□ 检查CPU频率 (关闭节能模式)

内存:
□ 预估索引大小,确保不swap
□ 使用OnDisk对超大索引
□ mmap加载减少启动时间
□ 批量处理减少内存峰值

GPU:
□ 使用Float16节省显存
□ 多GPU并行 (index_cpu_to_all_gpus)
□ 预热GPU (首次搜索较慢)
□ 监控GPU利用率和显存

IO:
□ 使用SSD存储索引
□ 预加载关键数据
□ 批量查询减少API调用
□ 序列化索引复用
```

### 12.4 监控指标

```
基础指标:
□ QPS (Queries Per Second)
□ 延迟: p50, p95, p99
□ 召回率: Recall@1, Recall@K
□ CPU/GPU利用率
□ 内存/显存使用

高级指标:
□ 距离计算次数 (indexIVF_stats.ndis)
□ 量化时间占比
□ 缓存miss率
□ 线程扩展效率
□ Pareto前沿分析
```

### 12.5 常见问题排查

```
问题: QPS低
□ CPU利用率低? → 增加线程数
□ CPU利用率高? → 减少计算量 (更激进的压缩)
□ GPU利用率低? → 增加batch size
□ 单次查询慢? → 优化索引参数

问题: 延迟高
□ 量化时间长? → 使用更快的量化器
□ 扫描向量多? → 减小nprobe或增大nlist
□ 内存带宽瓶颈? → 使用PQ压缩
□ P99抖动大? → 检查系统负载,NUMA,超线程

问题: 召回率低
□ 增大nprobe (IVF)
□ 增大efSearch (HNSW)
□ 使用更大的M (PQ)
□ 考虑IndexRefine二阶段搜索
□ 检查训练数据质量

问题: 内存不足
□ 使用PQ压缩
□ 使用OnDiskInvertedLists
□ 减小M值 (PQ)
□ 索引分片 (IndexShards)
□ 分布式索引
```

---

## 附录A: 常用命令速查

```bash
# 构建优化版本
cmake -B build -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 .
make -C build -j faiss

# 运行基准测试
cd benchs
python bench_gpu_sift1m.py
python bench_all_ivf/bench_all_ivf.py --indexkey "IVF4096,PQ64"

# 性能分析
perf record -g ./test_program
perf report

# GPU profiling
ncu --set full -o profile ./gpu_program
ncu-ui profile.ncu-rep

# 查看索引信息
python -c "import faiss; idx=faiss.read_index('index.faiss'); print(idx)"
```

## 附录B: Python API速查

```python
import faiss

# 性能监控
faiss.cvar.indexIVF_stats.reset()
stats = faiss.cvar.indexIVF_stats
print(stats.ndis, stats.search_time)

# 编译信息
print(faiss.get_compile_options())

# 线程控制
faiss.omp_set_num_threads(16)
print(faiss.omp_get_max_threads())

# GPU转换
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
cpu_index = faiss.index_gpu_to_cpu(gpu_index)

# IO
faiss.write_index(index, "index.faiss")
index = faiss.read_index("index.faiss")
index = faiss.read_index("index.faiss", faiss.IO_FLAG_MMAP)
```

## 附录C: 参考资源

- **官方文档:** https://github.com/facebookresearch/faiss/wiki
- **论文:**
  - "Billion-scale similarity search with GPUs" (Johnson et al., 2017)
  - "Product Quantization for Nearest Neighbor Search" (Jégou et al., 2011)
  - "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW" (Malkov & Yashunin, 2016)
- **基准测试:** http://ann-benchmarks.com/
- **性能优化指南:** https://github.com/facebookresearch/faiss/wiki/Faster-search

---

**课程结束**

通过本课程,你应该已经掌握了Faiss的完整性能分析和调优技术栈。记住性能优化的核心原则:
1. **先测量,再优化** - 使用profiling找到真正的瓶颈
2. **权衡取舍** - 速度、精度、内存之间的平衡
3. **持续监控** - 生产环境的性能指标追踪
4. **迭代优化** - 每次改进验证效果

祝你在Faiss性能优化的道路上取得成功!
