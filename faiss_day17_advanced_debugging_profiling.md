# Faiss深度课程 - 第17天：高级调试与性能分析

## 课程目标

学习使用profiler、debugger等工具分析Faiss性能瓶颈，掌握高级调试技巧和性能优化方法论。

---

## 1. 性能分析工具概览

### 1.1 工具对比

| 工具 | 类型 | 平台 | 用途 |
|------|------|------|------|
| perf | CPU profiling | Linux | 函数级性能分析 |
| Valgrind/Callgrind | CPU/Mem profiling | Linux | 详细的函数调用图 |
| VTune | CPU profiling | Linux/Windows | Intel CPU深度分析 |
| flamegraph | 可视化 | Linux | 火焰图生成 |
| gprof | CPU profiling | 跨平台 | GCC内置profiler |
| Instruments | CPU/Mem profiling | macOS | 苹果官方工具 |
| Tracy Profiler | 实时profiling | 跨平台 | 游戏引擎常用 |

---

## 2. perf使用详解

### 2.1 perf基础命令

```bash
# 1. 记录性能数据
perf record -F 99 -g --call-graph dwarf ./my_faiss_app

# -F 99: 采样频率99Hz
# -g: 记录调用栈
# --call-graph dwarf: 使用DWARF调试信息

# 2. 查看报告
perf report

# 3. 生成火焰图
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg

# 4. 分析特定函数
perf report --stdio --sort overhead | grep "fvec_L2sqr"

# 5. 查看最耗时的函数
perf report --stdio --sort overhead -n 20
```

### 2.2 perf高级用法

```bash
# 记录cache miss
perf stat -e cache-references,cache-misses,L1-dcache-load-misses ./my_faiss_app

# 记录CPU周期
perf stat -e cycles,instructions,branches ./my_faiss_app

# 分析特定进程
perf top -p $(pgrep my_faiss_app)

# 系统-wide分析
perf record -a -g
perf report

# 热点函数分析
perf record -e cycles ./my_faiss_app
perf annotate --stdio
```

### 2.3 perf输出解读

```bash
# perf report输出示例
# Samples: 123K of event 'cycles'
# Event count (approx.): 123456789012
#
# Overhead  Command  Shared Object      Symbol
# ########  #######  #############  #######
#    15.23%  my_app   my_app           [.] fvec_L2sqr_avx2
#    10.45%  my_app   my_app           [.] IndexIVF::search
#     8.32%  my_app   libfaiss.so       [.] heap_replace_top
#     6.18%  my_app   my_app           [.] fvec_L2sqr_ref
#     5.01%  my_app   libfaiss.so       [.] ProductQuantizer::compute_code

# 解读：
# Overhead: 该函数占用CPU时间百分比
# Command: 进程名
# Shared Object: 所在库
# Symbol: 函数符号
```

---

## 3. Valgrind/Callgrind详解

### 3.1 Callgrind基础使用

```bash
# 1. 运行Callgrind
valgrind --tool=callgrind ./my_faiss_app

# 2. 查看结果（交互式）
kcachegrind callgrind.out.<pid>

# 3. 命令行分析
callgrind_annotate --auto=yes callgrind.out.<pid>

# 4. 可视化调用图
gprof2dot --format=callgrind --output=callgraph.dot callgrind.out.<pid>
dot -Tpng -o callgraph.png callgraph.dot
```

### 3.2 Callgrind高级选项

```bash
# 详细采样
valgrind --tool=callgrind \
    --callgrind-out-file=faiss.prof \
    --dump-instr=yes \
    --collect-jumps=yes \
    --simulate-cache=yes \
    --simulate-hwpref=yes \
    ./my_faiss_app

# 仅分析特定函数
valgrind --tool=callgrind \
    --fn-match[3]=fvec_* \
    ./my_faiss_app

# 排除库函数
valgrind --tool=callgrind \
    --skip-allocs=yes \
    --collect-bus=yes \
    ./my_faiss_app
```

### 3.3 cachegrind缓存分析

```bash
# 分析cache行为
valgrind --tool=cachegrind ./my_faiss_app

# 查看详细报告
cg_annotate cachegrind.out.<pid>

# 输出解读：
# ==12345==
# I   refs:      1,234,567,890  (指令读取次数)
# I1  misses:           123,456  (L1指令cache未命中)
# L2i misses:            12,345  (L2指令cache未命中)
# D   refs:        987,654,321  (数据读取次数)
# D1  misses:           234,567  (L1数据cache未命中)
# L2d misses:            23,456  (L2数据cache未命中)
```

---

## 4. VTune Profiler使用

### 4.1 VTune基础命令

```bash
# 1. 热点分析
vtune -collect hotspots -result-dir vtune_hotspots -- ./my_faiss_app

# 2. 微架构分析
vtune -collect uarch-exploration -result-dir vtune_uarch -- ./my_faiss_app

# 3. 内存访问分析
vtune -collect memory-access -result-dir vtune_memory -- ./my_faiss_app

# 4. 线程分析
vtune -collect threading -result-dir vtune_threads -- ./my_faiss_app

# 5. 查看结果（GUI或命令行）
vtune -report hotspots -result-dir vtune_hotspots -format csv > report.csv
```

### 4.2 VTune输出分析

```python
# 解析VTune CSV报告
import pandas as pd

def analyze_vtune_report(csv_file):
    df = pd.read_csv(csv_file)

    # 按CPU时间排序
    df_sorted = df.sort_values('CPU Time:Self', ascending=False)

    print("Top 20 Hotspots:")
    print(df_sorted[['Function', 'CPU Time:Self', 'CPU Time:Total']].head(20))

    # 计算缓存未命中率
    if 'L1 Hit' in df.columns and 'L1 Miss' in df.columns:
        df['L1 Miss Rate'] = df['L1 Miss'] / (df['L1 Hit'] + df['L1 Miss'])
        print("\nL1 Cache Miss Rate:")
        print(df[['Function', 'L1 Miss Rate']].head(10))
```

### 4.3 VTune高级分析

```bash
# 向量代码效率分析
vtune -collect code-analytics -result-dir vtune_vector -- ./my_faiss_app

# 查看SIMD利用率
vtune -report summary -result-dir vtune_vector | grep "Vector "

# 内存带宽分析
vtune -collect memory-bandwidth -result-dir vtune_bw -- ./my_faiss_app
```

---

## 5. 火焰图分析

### 5.1 生成火焰图

```bash
# 1. 使用perf生成火焰图
perf record -F 99 -g ./my_faiss_app
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg

# 2. 使用FlameGraph工具栈
git clone https://github.com/brendangregg/FlameGraph.git
cd FlameGraph

# 3. 生成不同类型的火焰图
perf script | ./stackcollapse-perf.pl | ./flamegraph.pl > flamegraph.svg
perf script | ./stackcollapse-perf.pl | ./flamegraph.pl --invert > iciclegraph.svg
perf script | ./stackcollapse-perf.pl | ./flamegraph.pl --colors=mem > mem_flamegraph.svg
```

### 5.2 解读火焰图

```
火焰图解读：
- x轴: 样本数量（宽度表示耗时）
- y轴: 调用栈深度
- 颜色: 随机或按类型着色

典型火焰图：

fvec_L2sqr_avx2  ████████████████████  (热点)
  ├── avx2_add    ████                   (子函数)
  ├── avx2_mul    ████████
  └── avx2_sub    ████

IndexIVF::search  ████████████
  ├── quantizer->search  ██████
  ├── invlists->get_codes  ████████████  (瓶颈)
  └── heap_update  ████

优化建议：
1. 宽度最宽的函数是首要优化目标
2. 可以展开火焰图查看详细的调用关系
```

### 5.3 差异火焰图

```bash
# 比较两次运行的差异
perf diff --baseline-only old_perf_data new_perf_data

# 生成差异火焰图
# 需要两个perf.data文件
perf diff old_perf_data new_perf_data | \
    stackcollapse-perf.pl | \
    flamegraph.pl --colors=red,green > diff_flamegraph.svg
```

---

## 6. Faiss内置分析工具

### 6.1 启用统计信息

```cpp
// Faiss内置的统计宏
#define FINTEGER_FAISS_STATS 1

#include <faiss/Index.h>
#include <faiss/utils/utils.h>

// 运行后打印统计信息
int main() {
    faiss::IndexFlatL2 index(128);

    // 添加向量
    index.add(n, xb);

    // 搜索
    index.search(nq, xq, k, distances, labels);

    // 打印统计信息
    printf("=== Faiss Statistics ===\n");
    printf("Distance computations: %zu\n", faiss::stats.ndis);
    printf("Heap updates: %zu\n", faiss::stats.nheap_updates);

    return 0;
}
```

### 6.2 IVF搜索统计

```cpp
// IndexIVF的详细统计
struct IndexIVFStats {
    size_t nq;        // 查询数
    size_t nlist;     // 访问的列表数
    size_t ndis;      // 距离计算数
    size_t nheap_updates;  // 堆更新数

    void reset() {
        nq = nlist = ndis = nheap_updates = 0;
    }

    void print() const {
        printf("=== IVF Search Statistics ===\n");
        printf("Queries: %zu\n", nq);
        printf("Lists accessed: %zu\n", nlist);
        printf("Distance computations: %zu\n", ndis);
        printf("Heap updates: %zu\n", nheap_updates);

        if (nq > 0) {
            printf("\nPer-query averages:\n");
            printf("  Lists: %.2f\n", (double)nlist / nq);
            printf("  Distances: %.2f\n", (double)ndis / nq);
            printf("  Heap updates: %.2f\n", (double)nheap_updates / nq);
        }
    }
};

// 使用
IndexIVFFlat index;
// ... 搜索 ...

IndexIVFStats stats = index.ivf_stats;
stats.print();
```

### 6.3 HNSW统计信息

```cpp
// HNSW搜索统计
struct HNSWStats {
    size_t n1;       // 搜索次数
    size_t n2;       // 候选耗尽次数
    size_t ndis;     // 距离计算数
    size_t nhops;    // 跳数（遍历的边数）

    void print() const {
        printf("=== HNSW Statistics ===\n");
        printf("Searches: %zu\n", n1);
        printf("Exhausted: %zu\n", n2);
        printf("Distances: %zu\n", ndis);
        printf("Hops: %zu\n", nhops);

        if (n1 > 0) {
            printf("\nPer-search averages:\n");
            printf("  Distances: %.2f\n", (double)ndis / n1);
            printf("  Hops: %.2f\n", (double)nhops / n1);
        }
    }
};

// 使用
IndexHNSWFlat index;
// ... 搜索 ...

HNSWStats stats = index.hnsw_stats;
stats.print();
```

---

## 7. 高级调试技巧

### 7.1 GDB调试技巧

```bash
# 1. 编译带调试符号的版本
cmake -DCMAKE_BUILD_TYPE=Debug ..
make faiss

# 2. 启动GDB
gdb ./my_faiss_app

# 3. 常用GDB命令
(gdb) break fvec_L2sqr_avx2    # 设置断点
(gdb) run                      # 运行程序
(gdb) print d                  # 打印变量d
(gdb) print x[0]@8             # 打印数组前8个元素
(gdb) info registers            # 查看寄存器
(gdb) x/16gx $rsp              # 查看栈内存
(gdb) disassemble              # 反汇编当前函数
(gdb) stepi                    # 单步执行（指令级）
(gdb) finish                   # 完成当前函数
```

### 7.2 SIMD调试

```cpp
// 打印SIMD寄存器内容
#ifdef __AVX2__
void print_m256(__m256 v) {
    alignas(32) float data[8];
    _mm256_store_ps(data, v);

    printf("[");
    for (int i = 0; i < 8; i++) {
        printf("%.2f", data[i]);
        if (i < 7) printf(", ");
    }
    printf("]\n");
}

void debug_avx2_code() {
    alignas(32) float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(32) float b[8] = {8, 7, 6, 5, 4, 3, 2, 1};

    __m256 va = _mm256_load_ps(a);
    __m256 vb = _mm256_load_ps(b);

    printf("va = ");
    print_m256(va);

    printf("vb = ");
    print_m256(vb);

    __m256 vc = _mm256_add_ps(va, vb);
    printf("va + vb = ");
    print_m256(vc);
}
#endif
```

### 7.3 内存调试

```bash
# 使用Valgrind检测内存错误
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         ./my_faiss_app

# 使用AddressSanitizer
cmake -DCMAKE_CXX_FLAGS="-fsanitize=address -g" \
      -DCMAKE_C_FLAGS="-fsanitize=address -g" ..
make

# 运行
ASAN_OPTIONS=detect_leaks=1:symbolize=1 ./my_faiss_app
```

---

## 8. 性能优化方法论

### 8.1 性能分析流程

```cpp
// 性能优化框架
class PerformanceOptimizer {
public:
    void optimize_search(Index* index) {
        // 1. 基准测试
        auto baseline = benchmark_search(index);
        printf("Baseline: %.2f ms\n", baseline.time_ms);

        // 2. 分析热点
        auto hotspots = analyze_hotspots();
        printf("Hotspots:\n");
        for (auto& spot : hotspots) {
            printf("  %s: %.1f%%\n", spot.function, spot.overhead);
        }

        // 3. 针对性优化
        for (auto& spot : hotspots) {
            if (spot.overhead > 10) {
                optimize_function(index, spot.function);
            }
        }

        // 4. 验证改进
        auto optimized = benchmark_search(index);
        printf("Optimized: %.2f ms\n", optimized.time_ms);
        printf("Speedup: %.2fx\n", baseline.time_ms / optimized.time_ms);
    }

private:
    struct Hotspot {
        std::string function;
        float overhead;  // 百分比
    };

    std::vector<Hotspot> analyze_hotspots() {
        // 使用perf或VTune分析
        std::vector<Hotspot> hotspots;

        // 示例数据
        hotspots.push_back({"fvec_L2sqr", 35.0});
        hotspots.push_back({"heap_replace_top", 15.0});
        hotspots.push_back({"pq::decode", 10.0});

        return hotspots;
    }

    void optimize_function(Index* index, const std::string& function) {
        if (function == "fvec_L2sqr") {
            // 确保使用最优SIMD实现
            printf("Optimizing %s...\n", function.c_str());
            // 实施优化...
        }
    }
};
```

### 8.2 微基准测试

```cpp
// Google Benchmark微基准
#include <benchmark/benchmark.h>

// 基准测试：L2距离计算
static void BM_L2sqr_Scalar(benchmark::State& state) {
    int d = 128;
    std::vector<float> x(d), y(d);

    for (auto _ : state) {
        float result = fvec_L2sqr_ref(x.data(), y.data(), d);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * d);
}
BENCHMARK(BM_L2sqr_Scalar);

#ifdef __AVX2__
static void BM_L2sqr_AVX2(benchmark::State& state) {
    int d = 128;
    std::vector<float> x(d), y(d);

    for (auto _ : state) {
        float result = fvec_L2sqr_avx2(x.data(), y.data(), d);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * d);
}
BENCHMARK(BM_L2sqr_AVX2);
#endif

// 基准测试：IVF搜索
static void BM_IVF_Search(benchmark::State& state) {
    int d = 128;
    int nlist = 100;

    IndexFlatL2 quantizer(d);
    IndexIVFFlat index(&quantizer, d, nlist);

    // 训练和添加数据
    size_t n = 100000;
    std::vector<float> xb(n * d);
    index.train(n, xb.data());
    index.add(n, xb.data());

    // 测试搜索
    int nq = 100;
    int k = 10;
    std::vector<float> xq(nq * d);
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances);
        benchmark::DoNotOptimize(labels);
    }

    state.SetItemsProcessed(state.iterations() * nq);
}
BENCHMARK(BM_IVF_Search);

BENCHMARK_MAIN();
```

### 8.3 A/B测试框架

```cpp
// A/B测试：比较不同索引配置
class ABTestFramework {
public:
    struct Config {
        std::string name;
        std::function<Index*()> create_index;
    };

    void run_ab_test(
            const std::vector<Config>& configs,
            const float* xb,
            size_t n,
            const float* xq,
            size_t nq,
            int k) {

        printf("=== A/B Test Results ===\n\n");

        for (const auto& config : configs) {
            // 创建索引
            std::unique_ptr<Index> index(config.create_index());

            // 训练
            index->train(n, xb);
            index->add(n, xb);

            // 测试搜索
            std::vector<float> distances(nq * k);
            std::vector<idx_t> labels(nq * k);

            auto start = std::chrono::high_resolution_clock::now();
            index->search(nq, xq, k, distances.data(), labels.data());
            auto end = std::chrono::high_resolution_clock::now();

            double time_ms = std::chrono::duration<double>(end - start).count() * 1000;
            double qps = nq / time_ms * 1000;

            printf("Config: %s\n", config.name.c_str());
            printf("  Time: %.2f ms\n", time_ms);
            printf("  QPS: %.0f\n", qps);
            printf("\n");
        }
    }
};

// 使用示例
int main() {
    int d = 128;
    size_t n = 1000000;
    size_t nq = 100;
    int k = 100;

    std::vector<float> xb(n * d), xq(nq * d);
    // 生成数据...

    ABTestFramework ab_test;

    ab_test.run_ab_test({
        {"Flat", [d]() { return new IndexFlatL2(d); }},
        {"IVF,Flat,nlist=100", [d]() {
            Index* q = new IndexFlatL2(d);
            return new IndexIVFFlat(q, d, 100);
        }},
        {"IVF,Flat,nlist=1000", [d]() {
            Index* q = new IndexFlatL2(d);
            return new IndexIVFFlat(q, d, 1000);
        }},
        {"HNSW,M=16", [d]() { return new IndexHNSWFlat(d, 16); }},
        {"HNSW,M=32", [d]() { return new IndexHNSWFlat(d, 32); }}
    }, xb.data(), n, xq.data(), nq, k);
}
```

---

## 9. 常见性能问题诊断

### 9.1 内存带宽瓶颈

```cpp
// 诊断内存带宽问题
void diagnose_memory_bandwidth() {
    // 1. 检查cache miss
    // 使用perf或VTune

    // 2. 优化内存访问模式
    // - 预取
    // - 提高数据局部性
    // - 使用SoA布局

    // 3. 示例：使用预取
    for (size_t i = 0; i < n; i++) {
        // 预取未来数据
        if (i + 16 < n) {
            _mm_prefetch((char*)(data + (i + 16) * d), _MM_HINT_T0);
        }

        // 处理当前数据
        process(data + i * d);
    }
}
```

### 9.2 CPU利用率瓶颈

```cpp
// 诊断CPU利用率
void diagnose_cpu_utilization() {
    // 1. 检查是否充分利用SIMD
    // - 向量化效率
    // - SIMD利用率

    // 2. 检查并行效率
    // - OpenMP线程数
    // - 负载均衡

    // 3. 优化示例：调整线程数
    int max_threads = omp_get_max_threads();

    for (int nt = 1; nt <= max_threads; nt++) {
        omp_set_num_threads(nt);

        auto start = std::chrono::high_resolution_clock::now();
        // 执行搜索...
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double>(end - start).count() * 1000;
        double speedup = baseline_time_ms / time_ms;
        double efficiency = speedup / nt;

        printf("Threads: %d, Time: %.2f ms, Speedup: %.2fx, Efficiency: %.2f%%\n",
               nt, time_ms, speedup, efficiency * 100);
    }
}
```

### 9.3 精度问题诊断

```cpp
// 诊断精度损失
void diagnose_accuracy(Index* index, Index* ground_truth,
                       const float* xq, size_t nq, int k) {
    // 1. 搜索ground truth
    std::vector<float> gt_distances(nq * k);
    std::vector<idx_t> gt_labels(nq * k);
    ground_truth->search(nq, xq, k, gt_distances.data(), gt_labels.data());

    // 2. 搜索测试索引
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);
    index->search(nq, xq, k, distances.data(), labels.data());

    // 3. 计算recall
    int correct = 0;
    for (size_t q = 0; q < nq; q++) {
        std::set<idx_t> gt_set(
            gt_labels.begin() + q * k,
            gt_labels.begin() + (q + 1) * k);

        for (int i = 0; i < k; i++) {
            if (gt_set.count(labels[q * k + i])) {
                correct++;
            }
        }
    }

    float recall = (float)correct / (nq * k);
    printf("Recall@%d: %.3f%%\n", k, recall * 100);
}
```

---

## 10. 高级硬件事件分析

### 10.1 perf硬件计数器详解

```bash
# CPU性能计数器事件
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses ./my_faiss_app

# 输出解读：
# 1,234,567,890 cycles              # CPU周期数
#   987,654,321 instructions       # 指令数
#     1.25 insn per cycle          # IPC (理想是4-8)
#     123,456 cache-references     # 缓存访问次数
#      12,345 cache-misses         # 缓存未命中 (10% miss rate)
#       1,234 branches             # 分支指令
#         567 branch-misses        # 分支预测失败

# 微架构事件（Intel specific）
perf stat -e r1801,r1001,r801 -a sleep 1  # 每10秒的CPU周期

# TLB未命中
perf stat -e dTLB-loads,dTLB-load-misses,iTLB-loads,iTLB-load-misses ./my_faiss_app

# 分支预测分析
perf stat -e branches,branch-misses,branch-instructions ./my_faiss_app

# SIMD指令统计
perf stat -e simd.fp_arity.packed,sse.avx512.ops,sse.avx2.ops ./my_faiss_app

# 内存带宽
perf stat -e cycles,instructions,cache-misses,cache-references,L1-dcache-load-misses,L1-dcache-loads ./my_faiss_app
```

### 10.2 eBPF动态追踪

```bash
# 使用bpftrace进行动态分析
# 安装bpftrace
sudo apt-get install bpftrace bcc-tools

# 追踪函数调用延迟
sudo bpftrace -e '
kprobe:faiss::IndexIVF_search {
    @start[tid] = nsecs;
}
kretprobe:faiss::IndexIVF_search /@start[tid]/ {
    @duration_ns = hist(nsecs - @start[tid]);
    delete(@start[tid]);
}
'

# 追踪内存分配
sudo bpftrace -e '
uprobe:libc.so:malloc {
    @malloc_sizes[arg0] = count();
}
'

# 追踪cache miss
sudo bpftrace -e '
tracepoint:cache:cache_hit_result {
    @cache_hits[args->type] = count();
}
'

# 实时火焰图
sudo bpftrace -e '
profile:hz:99 {
    @[ustack] = count();
}
'
```

### 10.3 Intel PT (Processor Trace)

```bash
# Intel PT: 精确的分支追踪
# 1. 启用PT追踪
perf record -e intel_pt//u -e intel_pt//k ./my_faiss_app

# 2. 解码PT数据
perf script --itrace=i1000 --ns --pid=1234

# 3. 查看详细执行流
perf inject -i perf.data --itrace=i1000 --strip -o perf.data.inj

# 4. 生成PT火焰图
perf script -s script_pt.py > pt_flamegraph.svg
```

---

## 11. 内存深度分析

### 11.1 堆内存分析

```bash
# 使用jemalloc进行堆profiling
export MALLOC_CONF="prof:true,prof_final:true"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
./my_faiss_app

# 生成heap profile
jeprof --pdf ./my_faiss_app jeprof.<pid>.<script>.heap > heap_profile.pdf

# 使用tcmalloc (Google Performance Tools)
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
HEAPPROFILE=/tmp/heap_profile ./my_faiss_app

# 分析heap profile
pprof --pdf ./my_faiss_app /tmp/heap_profile.*.heap > tcmalloc_profile.pdf
```

### 11.2 内存泄漏检测

```cpp
// 使用Valgrind的Memcheck
// 编译选项: -g -O0 (保留调试信息，无优化)

#include <valgrind/memcheck.h>

// 标记内存块
void* tracked_malloc(size_t size) {
    void* ptr = malloc(size);
    VALGRIND_MALLOCLIKE_BLOCK(ptr, size, 0, 1);
    return ptr;
}

// 检查内存泄漏
void check_leak_example() {
    char* leak = new char[100];
    VALGRIND_CREATE_MEMPOOL(leak, 0, 0);  // 创建内存池
    VALGRIND_MEMPOOL_ALLOC(leak, leak, 100);

    // 使用内存...

    // 忘记delete
    VALGRIND_MEMPOOL_FREE(leak, leak);
    VALGRIND_DESTROY_MEMPOOL(leak);
}

// 运行检测
// valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./my_app
```

### 11.3 内存访问模式分析

```cpp
// 分析内存访问效率
namespace memory_access_analysis {

// 缓存行访问分析
struct CacheLineAnalyzer {
    static constexpr size_t CACHE_LINE_SIZE = 64;

    static void analyze_access_pattern(
            const float* data,
            size_t n,
            size_t stride) {

        printf("=== Cache Line Analysis ===\n");
        printf("Data: %p, Elements: %zu, Stride: %zu\n\n", data, n, stride);

        std::map<size_t, size_t> cache_line_accesses;

        for (size_t i = 0; i < n; i++) {
            uintptr_t addr = (uintptr_t)(data + i * stride);
            size_t cache_line = addr / CACHE_LINE_SIZE;
            cache_line_accesses[cache_line]++;
        }

        printf("Cache line accesses:\n");
        size_t total_lines = cache_line_accesses.size();
        size_t total_accesses = 0;
        for (auto& [line, count] : cache_line_accesses) {
            printf("  Line %05zu: %zu times\n", line, count);
            total_accesses += count;
        }

        printf("\nSummary:\n");
        printf("  Total cache lines: %zu\n", total_lines);
        printf("  Total accesses: %zu\n", total_accesses);
        printf("  Accesses per line: %.2f\n", (double)total_accesses / total_lines);
    }
};

// False Sharing检测
struct FalseSharingDetector {
    struct alignas(64) Counter {
        std::atomic<int> value;
        char padding[64 - sizeof(std::atomic<int>)];
    };

    static void detect_false_sharing() {
        // 使用perf检测
        // perf stat -e cache-references,cache-misses ./my_app

        // 高cache-misses可能表示False Sharing
    }
};

// NUMA感知分析
#ifdef __linux__
#include <numa.h>

struct NUMAAccessPattern {
    static void analyze_numa_locality() {
        // 检查NUMA拓扑
        int numa_nodes = numa_num_configured_nodes();
        printf("NUMA nodes: %d\n", numa_nodes);

        for (int i = 0; i < numa_nodes; i++) {
            nodemask_t nodemask;
            numa_bitmask_clearall(&nodemask);
            numa_bitmask_setbit(&nodemask, i);

            long long size = numa_node_size64(i, nullptr);
            printf("  Node %d: %lld MB\n", i, size / (1024 * 1024));
        }
    }

    static void* numa_alloc_local(size_t size) {
        return numa_alloc_local(size);
    }

    static void* numa_alloc_onnode(size_t size, int node) {
        void* ptr = numa_alloc_onnode(size, node);
        if (ptr) {
            printf("Allocated %zu bytes on node %d: %p\n", size, node, ptr);
        }
        return ptr;
    }
};
#endif
}
```

---

## 12. 锁竞争与同步分析

### 12.1 pthread锁分析

```bash
# 使用perf分析锁竞争
perf record -e locks --call-graph dwarf ./my_faiss_app
perf report

# 使用pthread竞争检测
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4
export CPUPROFILE=/tmp/profile
export CPUPROFILE_FREQUENCY=250
./my_faiss_app

# 使用perf lock子命令
perf lock record ./my_faiss_app
perf lock report

# 输出示例:
# Name   acquired  contended  total wait (ns)  avg wait (ns)
# ----   --------  ---------  ---------------  -------------
# mutex  12345     678         12345678         18200
# rwlock 5678      123         2345678          19000
```

### 12.2 OpenMP性能分析

```cpp
// OpenMP tracing
void profile_openmp_code() {
    // 设置OpenMP环境变量
    // export OMP_DISPLAY_ENV=TRUE
    // export OMP_PROC_BIND=close
    // export OMP_PLACES=cores

    // 运行时显示线程绑定
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        #pragma omp critical
        {
            printf("Thread %d of %d on CPU %d\n",
                   tid, nthreads, sched_getcpu());
        }
    }

    // 查看负载均衡
    int work_per_thread = 1000;
    std::vector<int> thread_work(omp_get_max_threads(), 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(dynamic, 10)
        for (int i = 0; i < work_per_thread * omp_get_num_threads(); i++) {
            thread_work[tid]++;
            // 执行工作...
        }
    }

    printf("Work distribution:\n");
    for (size_t i = 0; i < thread_work.size(); i++) {
        printf("  Thread %zu: %d iterations\n", i, thread_work[i]);
    }
}
```

### 12.3 死锁检测

```cpp
// 使用gdb检测死锁
// 1. 程序挂起时，attach gdb
// gdb -p <pid>

// 2. 查看所有线程
// (gdb) thread apply all bt

// 3. 查看锁状态
// (gdb) print pthread_mutex

// 4. 查看死锁的线程
// (gdb) info threads

// 使用Valgrind的Helgrind
// valgrind --tool=helgrind ./my_faiss_app

// 输出解读:
// ==12345== Possible data race during write of size 8
// ==12345==    at 0x123456: IndexIVF::search (index.cpp:456)
// ==12345==  by thread #1
// ==12345==  This conflicts with a previous read of size 8
// ==12345==    at 0x234567: fvec_L2sqr (distances.cpp:123)
// ==12345==  by thread #2
```

---

## 13. 编译器优化分析

### 13.1 检查汇编输出

```bash
# 生成汇编代码
g++ -S -O3 -mavx2 -o faiss_asm.s faiss.cpp

# 使用objdump分析
objdump -d -M intel my_faiss_app | grep -A 20 "fvec_L2sqr"

# 使用Godbolt在线查看
# https://godbolt.org/

# GCC优化报告
g++ -O3 -fopt-info-vec -mavx2 faiss.cpp

# 输出示例:
# faiss.cpp:123:20: optimized: loop vectorized using 32 byte vectors
# faiss.cpp:234:15: missed: vectorization possible but not profitable
```

### 13.2 内联分析

```cpp
// 检查函数是否被内联
__attribute__((noinline)) void trace_function() {
    printf("This function will NOT be inlined\n");
}

// 强制内联
__attribute__((always_inline)) inline void hot_function() {
    printf("This function MUST be inlined\n");
}

// 查看实际内联情况
// 使用objdump或readelf查看符号表
// nm -C my_faiss_app | grep "function_name"

// 使用GCC的dump功能
g++ -fdump-ipa-inline -O3 faiss.cpp
cat faiss.cpp.???i.inline  # 查看内联决策
```

### 13.3 优化标志测试

```bash
# 测试不同优化级别
for opt in O0 O1 O2 O3 Os; do
    echo "Testing -$opt"
    g++ -$opt -mavx2 my_faiss_app.cpp -o my_faiss_app_$opt
    ./my_faiss_app_$opt
done

# 测试特定优化
g++ -O3 -ftree-vectorize -funroll-loops my_faiss_app.cpp -o my_faiss_app
g++ -O3 -fno-vectorize -fno-unroll-loops my_faiss_app.cpp -o my_faiss_app_novec

# 比较性能
hyperfine ./my_faiss_app ./my_faiss_app_novec

# 查看链接时优化(LTO)
g++ -O3 -flto -fuse-linker-plugin -ffat-lto-objects my_faiss_app.cpp -o my_faiss_app_lto
```

---

## 14. 生产环境profiling

### 14.1 低开销profiling

```cpp
// 生产环境采样profiler
namespace production_profiling {

// 基于信号的安全采样
class SignalProfiler {
public:
    struct Sample {
        void* ip;      // 指令指针
        void* bp;      // 基址指针
        int tid;       // 线程ID
        uint64_t ts;   // 时间戳
    };

    static constexpr size_t MAX_SAMPLES = 1000000;

    std::vector<Sample> samples;
    std::atomic<bool> running{false};

    void start(int frequency_hz = 99) {
        samples.clear();
        running = true;

        std::thread([this, frequency_hz]() {
            struct sigaction sa;
            sa.sa_sigaction = signal_handler;
            sa.sa_flags = SA_SIGINFO | SA_RESTART;
            sigemptyset(&sa.sa_mask);

            // 设置定时器
            struct itimerval timer;
            timer.it_interval.tv_sec = 0;
            timer.it_interval.tv_usec = 1000000 / frequency_hz;
            timer.it_value = timer.it_interval;

            setitimer(ITIMER_PROF, &timer, nullptr);

            while (running) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }).detach();
    }

    void stop() {
        running = false;
    }

    void report() {
        printf("=== Profiling Report ===\n");
        printf("Samples: %zu\n\n", samples.size());

        // 聚合符号
        std::map<void*, size_t> symbol_counts;
        for (auto& s : samples) {
            symbol_counts[s.ip]++;
        }

        // 排序并输出top 10
        std::vector<std::pair<void*, size_t>> sorted(
            symbol_counts.begin(), symbol_counts.end());
        std::sort(sorted.begin(), sorted.end(),
                 [](auto& a, auto& b) { return a.second > b.second; });

        printf("Top 10 hotspots:\n");
        for (size_t i = 0; i < std::min(size_t(10), sorted.size()); i++) {
            char sym[256];
            char* demangled = abi::__cxa_demangle(
                sym, nullptr, nullptr, nullptr);
            printf("  %2zu. %p: %.1f%% (%s)\n",
                   i + 1, sorted[i].first,
                   100.0 * sorted[i].second / samples.size(),
                   demangled ? demangled : "??");
            free(demangled);
        }
    }

private:
    static void signal_handler(int sig, siginfo_t* info, void* ctx) {
        // 捕获调用栈
        ucontext_t* ucontext = (ucontext_t*)ctx;

        // 只记录一部分样本以减少开销
        static thread_local size_t counter = 0;
        if (++counter % 16 == 0) {  // 每16次采样记录1次
            Sample s;
            s.ip = (void*)ucontext->uc_mcontext.gregs[REG_RIP];
            s.bp = (void*)ucontext->uc_mcontext.gregs[REG_RBP];
            s.tid = syscall(SYS_gettid);
            s.ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();

            // 存储到线程本地缓冲
            get_local_samples().push_back(s);
        }
    }

    static std::vector<Sample>& get_local_samples() {
        static thread_local std::vector<Sample> samples;
        return samples;
    }
};

// 使用示例
void production_profile_search(Index* index) {
    SignalProfiler profiler;

    profiler.start(99);  // 99Hz采样

    // 执行搜索
    for (int i = 0; i < 1000; i++) {
        index->search(...);
    }

    profiler.stop();
    profiler.report();
}
}
```

### 14.2 热点函数重载

```cpp
// 动态dispatch到最优实现
namespace runtime_dispatch {

typedef float (*DistanceFn)(const float*, const float*, size_t);

// 性能测试并选择最快实现
DistanceFn select_fastest_distance_fn() {
    static DistanceFn best = nullptr;

    if (best == nullptr) {
        const size_t test_size = 100000;
        std::vector<float> x(test_size), y(test_size);

        std::map<std::string, DistanceFn> candidates = {
            {"scalar", fvec_L2sqr_ref},
#ifdef __AVX2__
            {"avx2", fvec_L2sqr_avx2},
#endif
#ifdef __AVX512F__
            {"avx512", fvec_L2sqr_avx512},
#endif
#ifdef __aarch64__
            {"neon", fvec_L2sqr_neon},
#endif
        };

        std::string fastest;
        double best_time = std::numeric_limits<double>::max();

        for (auto& [name, fn] : candidates) {
            auto start = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < test_size; i += 128) {
                fn(x.data() + i, y.data() + i, 128);
            }

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();

            printf("%s: %.6f seconds\n", name.c_str(), elapsed);

            if (elapsed < best_time) {
                best_time = elapsed;
                fastest = name;
                best = fn;
            }
        }

        printf("Selected: %s\n", fastest.c_str());
    }

    return best;
}
}
```

---

## 15. 第17天总结

### 关键工具

1. **perf**: Linux系统级profiler
2. **Valgrind**: 内存和CPU分析
3. **VTune**: Intel深度分析工具
4. **eBPF/bpftrace**: 动态追踪
5. **Intel PT**: 精确分支追踪
6. **jemalloc/tcmalloc**: 堆内存profiling
7. **Helgrind**: 线程竞争检测

### 分析方法

1. **热点分析**: 找到耗时最多的函数
2. **内存分析**: 诊断cache miss、内存泄漏
3. **并发分析**: 检测锁竞争、死锁
4. **编译分析**: 检查SIMD、内联、LTO
5. **生产profiling**: 低开销采样profiling

### 诊断流程

```
发现问题 → 性能测试 → 热点定位 → 根因分析 → 实施优化 → 验证效果
    ↑                                                    ↓
    └────────────────── 监控告警 ←───────────────────────┘
```

### 下一步

第18天将学习**与其他向量库对比分析**，了解Faiss相对于其他向量检索库的优势和劣势。

---

## 练习题

1. 使用perf分析Faiss搜索性能
2. 生成并解读火焰图
3. 使用VTune分析SIMD效率
4. 编写A/B测试框架比较不同索引
5. **使用bpftrace追踪函数调用延迟**
6. **实现生产环境的采样profiler**
7. **分析并优化False Sharing问题**

## 16. Faiss底层性能分析函数实现详解

### 16.1 时间测量函数

#### getmillisecs() - 毫秒级时间测量

```cpp
// faiss/utils/utils.cpp:144-158

#ifdef _MSC_VER
// Windows平台实现
double getmillisecs() {
    LARGE_INTEGER ts;      // 时间戳计数器值
    LARGE_INTEGER freq;    // 频率(每秒计数)

    // 获取性能计数器频率
    QueryPerformanceFrequency(&freq);

    // 获取当前时间戳计数
    QueryPerformanceCounter(&ts);

    // 转换为毫秒: (count / freq) * 1000
    return (ts.QuadPart * 1e3) / freq.QuadPart;
}

#else
// Linux/Unix平台实现
double getmillisecs() {
    struct timeval tv;
    // gettimeofday提供微秒级精度
    gettimeofday(&tv, nullptr);

    // 转换为毫秒
    // tv.tv_sec: 秒, tv.tv_usec: 微秒
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}
#endif

// 使用示例
void benchmark_function() {
    double t0 = faiss::getmillisecs();

    // 执行操作
    index->search(nq, xq, k, distances, labels);

    double t1 = faiss::getmillisecs();
    printf("Search time: %.2f ms\n", t1 - t0);
}
```

**实现要点**:
- **Windows**: 使用`QueryPerformanceCounter`,精度可达纳秒级
- **Linux**: 使用`gettimeofday`,精度为微秒级
- **跨平台**: 通过预编译宏实现平台隔离

#### get_cycles() - CPU周期计数

```cpp
// faiss/utils/utils.cpp:160-168

uint64_t get_cycles() {
#ifdef __x86_64__
    // x86-64架构: 使用RDTSC指令
    uint32_t high, low;

    // 内联汇编执行RDTSC(Read Time-Stamp Counter)
    // RDTSC返回自CPU重启以来的时钟周期数
    asm volatile(
        "rdtsc \n\t"           // 执行RDTSC指令
        : "=a"(low),           // 输出: EAX寄存器 -> low
          "=d"(high)           // 输出: EDX寄存器 -> high
    );

    // 组合高32位和低32位
    return ((uint64_t)high << 32) | (low);

#else
    // 其他架构: 返回0(不支持)
    return 0;
#endif
}

// 使用示例
void measure_cycles() {
    uint64_t c0 = faiss::get_cycles();

    // 执行操作
    float dis = fvec_L2sqr(x, y, d);

    uint64_t c1 = faiss::get_cycles();

    printf("Cycles: %lu\n", c1 - c0);

    // 如果CPU频率为3GHz,则:
    // 时间(秒) = cycles / 3e9
    // 时间(纳秒) = cycles / 3
}
```

**RDTSC指令详解**:
- **功能**: 读取64位时间戳计数器(Time-Stamp Counter)
- **精度**: CPU时钟周期级(纳秒级)
- **用途**:
  - 测量短时间间隔(函数执行时间)
  - 计算CPU周期数
  - 性能计数

**注意事项**:
1. CPU动态调频会导致周期数不稳定
2. 多核CPU上需要固定CPU核心
3. 乱序执行可能影响测量准确性

```cpp
// 更精确的测量方法(避免乱序执行)
inline uint64_t get_cycles_serialized() {
#ifdef __x86_64__
    unsigned int aux;
    // 使用__rdtscp序列化指令
    return __rdtscp(&aux);  // GCC/Clang内置函数
#else
    return 0;
#endif
}
```

### 16.2 内存使用测量

#### get_mem_usage_kb() - 获取RSS内存

```cpp
// faiss/utils/utils.cpp:172-188

#ifdef __linux__

size_t get_mem_usage_kb() {
    // 1. 获取当前进程ID
    int pid = getpid();

    // 2. 构造/proc/[pid]/status路径
    char fname[256];
    snprintf(fname, 256, "/proc/%d/status", pid);

    // 3. 打开文件
    FILE* f = fopen(fname, "r");
    FAISS_THROW_IF_NOT_MSG(f, "cannot open proc status file");

    // 4. 解析VmRSS字段
    // /proc/[pid]/status格式:
    // VmRSS:     123456 kB  <-- 我们需要这个值
    size_t sz = 0;
    char buf[256];
    while (fgets(buf, 256, f)) {
        if (sscanf(buf, "VmRSS: %zu kB", &sz) == 1) {
            break;
        }
    }

    fclose(f);
    return sz;
}

#else
size_t get_mem_usage_kb() {
    // 非Linux平台: 返回0
    return 0;
}
#endif

// 使用示例
void monitor_memory() {
    size_t mem_before = faiss::get_mem_usage_kb();

    // 执行操作(例如添加向量)
    index->add(n, xb);

    size_t mem_after = faiss::get_mem_usage_kb();
    size_t delta = mem_after - mem_before;

    printf("Memory before: %zu KB\n", mem_before);
    printf("Memory after:  %zu KB\n", mem_after);
    printf("Memory delta:  %zu KB\n", delta);
}
```

**/proc/[pid]/status字段说明**:

```
VmPeak:     123456 kB    # 峰值虚拟内存
VmSize:     100000 kB    # 当前虚拟内存
VmRSS:       80000 kB    # 驻留集大小(物理内存使用)
VmData:      50000 kB    # 数据段大小
VmStk:        1024 kB    # 栈段大小
VmExe:         500 kB    # 代码段大小
```

**RSS(Resident Set Size)**: 进程实际占用的物理内存,不包括swap out的部分。

### 16.3 性能统计宏

#### FAISS stats实现

```cpp
// faiss/utils/utils.h (可能定义,具体位置可能不同)

#ifdef FINTEGER_FAISS_STATS
namespace faiss {
    // 全局统计结构
    struct FaissStats {
        size_t ndis;              // 距离计算次数
        size_t nheap_updates;     // 堆更新次数

        void reset() {
            ndis = 0;
            nheap_updates = 0;
        }

        void print() const {
            printf("=== Faiss Statistics ===\n");
            printf("Distance computations: %zu\n", ndis);
            printf("Heap updates: %zu\n", nheap_updates);
        }
    };

    // 全局统计实例
    static FaissStats stats;
}
#endif

// 在代码中使用
void IndexIVF::search_impl(
        size_t n,
        const float* x,
        size_t k,
        float* distances,
        idx_t* labels) const {

    // 重置统计
#ifdef FINTEGER_FAISS_STATS
    stats.reset();
#endif

    // 搜索过程
    for (size_t i = 0; i < n; i++) {
        // ...

        // 统计距离计算
#ifdef FINTEGER_FAISS_STATS
        stats.ndis += computed_distances;
#endif

        // 统计堆更新
#ifdef FINTEGER_FAISS_STATS
        stats.nheap_updates += heap_updates;
#endif
    }

    // 打印统计
#ifdef FINTEGER_FAISS_STATS
    stats.print();
#endif
}
```

### 16.4 自定义性能分析器

```cpp
// 实现一个轻量级的性能分析器

class FaissProfiler {
    struct FunctionStats {
        std::string name;
        uint64_t total_cycles;
        size_t call_count;

        FunctionStats() : total_cycles(0), call_count(0) {}
    };

    std::unordered_map<std::string, FunctionStats> stats;

public:
    // 开始计时
    uint64_t start() {
        return faiss::get_cycles();
    }

    // 结束计时并记录
    void end(const std::string& function, uint64_t start_cycles) {
        uint64_t end_cycles = faiss::get_cycles();
        stats[function].total_cycles += (end_cycles - start_cycles);
        stats[function].call_count++;
        stats[function].name = function;
    }

    // 打印报告
    void print_report() {
        printf("\n=== Faiss Profiler Report ===\n");
        printf("%-30s | %-10s | %-15s | %-15s\n",
               "Function", "Calls", "Total Cycles", "Avg Cycles");
        printf("----------------------------------------------"
               "--------------------------------\n");

        // 按总周期排序
        std::vector<std::pair<std::string, FunctionStats>> sorted(
            stats.begin(), stats.end());
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) {
                return a.second.total_cycles > b.second.total_cycles;
            });

        for (const auto& entry : sorted) {
            const auto& s = entry.second;
            uint64_t avg = s.call_count > 0
                ? s.total_cycles / s.call_count
                : 0;

            printf("%-30s | %-10zu | %-15lu | %-15lu\n",
                   s.name.c_str(),
                   s.call_count,
                   s.total_cycles,
                   avg);
        }
        printf("\n");
    }
};

// 使用示例(带RAII的自动计时)
class ScopedTimer {
    FaissProfiler& profiler;
    std::string function;
    uint64_t start_cycles;

public:
    ScopedTimer(FaissProfiler& p, const std::string& func)
        : profiler(p), function(func), start_cycles(p.start()) {}

    ~ScopedTimer() {
        profiler.end(function, start_cycles);
    }
};

// 在代码中使用
void IndexIVF::search(...) {
    static FaissProfiler profiler;

    {
        ScopedTimer timer(profiler, "coarse_quantizer");
        coarse_quantizer->search(...);
    }

    {
        ScopedTimer timer(profiler, "scan_lists");
        scan_lists(...);
    }

    {
        ScopedTimer timer(profiler, "heap_update");
        update_heap(...);
    }

    profiler.print_report();
}
```

### 16.5 编译时性能分析选项

```cpp
// Faiss编译时的性能分析选项

// 1. 启用统计信息收集
#define FINTEGER_FAISS_STATS 1

// 2. 启用详细的性能日志
#define FAISS_ENABLE_STATS 1

// 3. 启用搜索时的统计
// 在IndexIVF等索引中会收集详细的搜索统计信息

// 使用CMake配置
cmake -DFAISS_ENABLE_STATS=ON \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      ..
make

// 编译后运行
./my_faiss_app

// 查看统计信息
// 程序会自动打印:
// - nq: 查询数量
// - nlist: 访问的倒排列表数量
// - ndis: 距离计算次数
// - nheap_updates: 堆更新次数
```

### 16.6 性能分析最佳实践

```cpp
// 综合性能分析示例

class ComprehensiveBenchmark {
public:
    void run_benchmark(Index* index) {
        // 1. 预热
        printf("=== Warming up ===\n");
        for (int i = 0; i < 10; i++) {
            index->search(nq, xq, k, distances, labels);
        }

        // 2. 基准测试
        printf("\n=== Benchmark ===\n");

        // 时间测量
        double t0 = faiss::getmillisecs();
        index->search(nq, xq, k, distances, labels);
        double t1 = faiss::getmillisecs();

        printf("Time: %.2f ms\n", t1 - t0);
        printf("QPS: %.0f\n", nq / ((t1 - t0) / 1000));

        // 内存测量
        size_t mem = faiss::get_mem_usage_kb();
        printf("Memory: %zu KB\n", mem);

        // 统计信息
#ifdef FINTEGER_FAISS_STATS
        faiss::stats.print();
#endif

        // 3. 详细分析
        printf("\n=== Detailed Analysis ===\n");

        // 使用perf(外部)
        printf("Run: perf stat -p %d\n", getpid());

        // 等待用户分析
        sleep(10);

        // 继续执行...
        index->search(nq, xq, k, distances, labels);
    }
};
```

---

## 扩展阅读

- [Linux perf wiki](https://perf.wiki.kernel.org/)
- [FlameGraph](https://github.com/brendangregg/FlameGraph)
- [VTune Profiler Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)
- [bpftrace Reference Guide](https://www.bpftrace.org/reference.html)
- [eBPF Overview](https://ebpf.io/)
- [RDTSC指令](https://www.felixcloutier.com/x86/rdtsc.html)
- [Linux /proc filesystem](https://man7.org/linux/man-pages/man5/proc.5.html)
