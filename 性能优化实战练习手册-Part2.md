# 性能优化实战练习手册 - Part 2

## 续前文...

### 练习 2.3: IO性能优化

**目标**: 学习异步IO和零拷贝技术

**场景**: 文件拷贝程序优化

**起始代码**:

```cpp
// 02-intermediate/file_copy.cpp
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <chrono>
#include <cstring>

// 方法1: 传统read/write
double copy_traditional(const char* src, const char* dst) {
    int src_fd = open(src, O_RDONLY);
    int dst_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);

    const size_t BUFFER_SIZE = 4096;
    char buffer[BUFFER_SIZE];

    auto start = std::chrono::high_resolution_clock::now();

    ssize_t n;
    while ((n = read(src_fd, buffer, BUFFER_SIZE)) > 0) {
        write(dst_fd, buffer, n);
    }

    auto end = std::chrono::high_resolution_clock::now();

    close(src_fd);
    close(dst_fd);

    return std::chrono::duration<double>(end - start).count();
}

// TODO: 实现优化版本
double copy_optimized(const char* src, const char* dst) {
    // 你的代码: 使用sendfile或io_uring
    return 0.0;
}

void create_test_file(const char* filename, size_t size_mb) {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    const size_t CHUNK = 1024 * 1024;
    char* buffer = new char[CHUNK];
    memset(buffer, 'A', CHUNK);

    for (size_t i = 0; i < size_mb; i++) {
        write(fd, buffer, CHUNK);
    }

    delete[] buffer;
    close(fd);
}

int main() {
    const char* src = "/tmp/test_src.dat";
    const char* dst = "/tmp/test_dst.dat";
    const size_t SIZE_MB = 1024;  // 1GB

    std::cout << "Creating test file..." << std::endl;
    create_test_file(src, SIZE_MB);

    std::cout << "Testing traditional copy..." << std::endl;
    double time1 = copy_traditional(src, dst);
    std::cout << "Time: " << time1 << " s" << std::endl;
    std::cout << "Speed: " << SIZE_MB / time1 << " MB/s" << std::endl;

    unlink(dst);

    // std::cout << "Testing optimized copy..." << std::endl;
    // double time2 = copy_optimized(src, dst);
    // std::cout << "Time: " << time2 << " s" << std::endl;
    // std::cout << "Speed: " << SIZE_MB / time2 << " MB/s" << std::endl;
    // std::cout << "Speedup: " << time1 / time2 << "x" << std::endl;

    unlink(src);
    unlink(dst);

    return 0;
}
```

**优化实现**:

```cpp
// 优化1: 增大缓冲区
double copy_large_buffer(const char* src, const char* dst) {
    int src_fd = open(src, O_RDONLY);
    int dst_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);

    const size_t BUFFER_SIZE = 1024 * 1024;  // 1MB
    char* buffer = new char[BUFFER_SIZE];

    auto start = std::chrono::high_resolution_clock::now();

    ssize_t n;
    while ((n = read(src_fd, buffer, BUFFER_SIZE)) > 0) {
        write(dst_fd, buffer, n);
    }

    auto end = std::chrono::high_resolution_clock::now();

    delete[] buffer;
    close(src_fd);
    close(dst_fd);

    return std::chrono::duration<double>(end - start).count();
}

// 优化2: sendfile (零拷贝)
#include <sys/sendfile.h>

double copy_sendfile(const char* src, const char* dst) {
    int src_fd = open(src, O_RDONLY);
    int dst_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);

    struct stat stat_buf;
    fstat(src_fd, &stat_buf);

    auto start = std::chrono::high_resolution_clock::now();

    off_t offset = 0;
    sendfile(dst_fd, src_fd, &offset, stat_buf.st_size);

    auto end = std::chrono::high_resolution_clock::now();

    close(src_fd);
    close(dst_fd);

    return std::chrono::duration<double>(end - start).count();
}

// 优化3: io_uring (异步IO)
#include <liburing.h>

double copy_iouring(const char* src, const char* dst) {
    int src_fd = open(src, O_RDONLY);
    int dst_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);

    struct io_uring ring;
    io_uring_queue_init(32, &ring, 0);

    const size_t BUFFER_SIZE = 64 * 1024;  // 64KB
    const int NUM_BUFFERS = 8;
    char* buffers[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; i++) {
        buffers[i] = new char[BUFFER_SIZE];
    }

    auto start = std::chrono::high_resolution_clock::now();

    off_t offset = 0;
    int buffer_idx = 0;
    bool eof = false;

    // 预提交读请求
    for (int i = 0; i < NUM_BUFFERS && !eof; i++) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, src_fd, buffers[i],
                          BUFFER_SIZE, offset);
        io_uring_sqe_set_data(sqe, (void*)(long)i);
        offset += BUFFER_SIZE;
    }

    io_uring_submit(&ring);

    // 处理完成
    int pending = NUM_BUFFERS;
    while (pending > 0) {
        struct io_uring_cqe* cqe;
        io_uring_wait_cqe(&ring, &cqe);

        int buf_id = (int)(long)io_uring_cqe_get_data(cqe);
        ssize_t n = cqe->res;

        if (n > 0) {
            // 写入
            write(dst_fd, buffers[buf_id], n);

            // 提交下一个读请求
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            io_uring_prep_read(sqe, src_fd, buffers[buf_id],
                              BUFFER_SIZE, offset);
            io_uring_sqe_set_data(sqe, (void*)(long)buf_id);
            offset += BUFFER_SIZE;
            io_uring_submit(&ring);
        } else {
            pending--;
        }

        io_uring_cqe_seen(&ring, cqe);
    }

    auto end = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_BUFFERS; i++) {
        delete[] buffers[i];
    }

    io_uring_queue_exit(&ring);
    close(src_fd);
    close(dst_fd);

    return std::chrono::duration<double>(end - start).count();
}

// 优化4: Direct IO + 对齐
double copy_direct(const char* src, const char* dst) {
    int src_fd = open(src, O_RDONLY | O_DIRECT);
    int dst_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);

    const size_t BUFFER_SIZE = 1024 * 1024;
    void* buffer;
    posix_memalign(&buffer, 4096, BUFFER_SIZE);

    auto start = std::chrono::high_resolution_clock::now();

    ssize_t n;
    while ((n = read(src_fd, buffer, BUFFER_SIZE)) > 0) {
        // 对齐写入大小
        size_t aligned = (n + 4095) & ~4095;
        write(dst_fd, buffer, aligned);
    }

    auto end = std::chrono::high_resolution_clock::now();

    free(buffer);
    close(src_fd);
    close(dst_fd);

    return std::chrono::duration<double>(end - start).count();
}
```

**编译运行**:

```bash
# 需要liburing
sudo apt-get install liburing-dev

# 编译
g++ -O3 file_copy.cpp -o file_copy -luring

# 运行
./file_copy

# 使用iostat监控IO
iostat -x 1
```

**性能目标** (1GB文件, SSD):

```
方法              时间(s)   速度(MB/s)   vs基线
────────────────────────────────────────────────
read/write(4KB)   ~12.0     85           1.0x
read/write(1MB)   ~2.8      365          4.3x
sendfile          ~2.1      490          5.7x
io_uring          ~1.8      570          6.7x
Direct IO         ~1.6      640          7.5x  ← 目标
```

**学习要点**:
- 缓冲区大小的影响
- 零拷贝技术的威力
- 异步IO的优势
- Direct IO的使用场景

---

### 练习 2.4: NUMA感知编程

**目标**: 学习NUMA系统的性能优化

**代码**:

```cpp
// 02-intermediate/numa_test.cpp
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <numa.h>
#include <numaif.h>
#include <sched.h>

const size_t ARRAY_SIZE = 100 * 1024 * 1024;  // 100M元素
const int NUM_THREADS = 8;

// 测试函数: 累加数组
long long sum_array(int* data, size_t size) {
    long long sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

// 方法1: 默认分配 (通常在node 0)
double test_default_allocation() {
    int* data = new int[ARRAY_SIZE];

    // 初始化
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        data[i] = i % 100;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // 多线程访问
    std::vector<std::thread> threads;
    std::vector<long long> results(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&, t]() {
            size_t chunk = ARRAY_SIZE / NUM_THREADS;
            results[t] = sum_array(data + t * chunk, chunk);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();

    delete[] data;

    return std::chrono::duration<double>(end - start).count();
}

// 方法2: 交错分配
double test_interleaved_allocation() {
    int* data = (int*)numa_alloc_interleaved(
        ARRAY_SIZE * sizeof(int)
    );

    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        data[i] = i % 100;
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    std::vector<long long> results(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&, t]() {
            size_t chunk = ARRAY_SIZE / NUM_THREADS;
            results[t] = sum_array(data + t * chunk, chunk);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();

    numa_free(data, ARRAY_SIZE * sizeof(int));

    return std::chrono::duration<double>(end - start).count();
}

// 方法3: First-touch + 线程绑定
double test_first_touch() {
    int* data = new int[ARRAY_SIZE];
    int num_nodes = numa_num_configured_nodes();

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    std::vector<long long> results(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&, t]() {
            // 绑定线程到NUMA节点
            int node = t % num_nodes;
            numa_run_on_node(node);

            size_t chunk = ARRAY_SIZE / NUM_THREADS;
            size_t start_idx = t * chunk;

            // First touch: 初始化本线程的数据
            for (size_t i = 0; i < chunk; i++) {
                data[start_idx + i] = (start_idx + i) % 100;
            }

            // 计算
            results[t] = sum_array(data + start_idx, chunk);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();

    delete[] data;

    return std::chrono::duration<double>(end - start).count();
}

// 方法4: 显式NUMA分配
double test_explicit_numa() {
    int num_nodes = numa_num_configured_nodes();

    // 为每个节点分配内存
    std::vector<int*> node_data(NUM_THREADS);
    size_t chunk = ARRAY_SIZE / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        int node = t % num_nodes;
        node_data[t] = (int*)numa_alloc_onnode(
            chunk * sizeof(int), node
        );

        // 初始化
        for (size_t i = 0; i < chunk; i++) {
            node_data[t][i] = (t * chunk + i) % 100;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    std::vector<long long> results(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&, t]() {
            // 绑定到对应节点
            int node = t % num_nodes;
            numa_run_on_node(node);

            // 访问本地数据
            results[t] = sum_array(node_data[t], chunk);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();

    // 释放内存
    for (int t = 0; t < NUM_THREADS; t++) {
        numa_free(node_data[t], chunk * sizeof(int));
    }

    return std::chrono::duration<double>(end - start).count();
}

int main() {
    if (numa_available() < 0) {
        std::cerr << "NUMA not available" << std::endl;
        return 1;
    }

    std::cout << "NUMA nodes: " << numa_num_configured_nodes() << std::endl;
    std::cout << "Array size: " << ARRAY_SIZE * sizeof(int) / 1024 / 1024
              << " MB" << std::endl;
    std::cout << "Threads: " << NUM_THREADS << std::endl;
    std::cout << std::endl;

    // 测试各种方法
    double time1 = test_default_allocation();
    std::cout << "Default allocation:    " << time1 << " s" << std::endl;

    double time2 = test_interleaved_allocation();
    std::cout << "Interleaved:           " << time2 << " s ("
              << time1 / time2 << "x)" << std::endl;

    double time3 = test_first_touch();
    std::cout << "First-touch:           " << time3 << " s ("
              << time1 / time3 << "x)" << std::endl;

    double time4 = test_explicit_numa();
    std::cout << "Explicit NUMA:         " << time4 << " s ("
              << time1 / time4 << "x)" << std::endl;

    return 0;
}
```

**编译运行**:

```bash
# 编译
g++ -O3 -pthread numa_test.cpp -o numa_test -lnuma

# 查看NUMA拓扑
numactl --hardware

# 运行测试
./numa_test

# 使用numastat监控
watch -n 1 numastat
```

**性能目标** (2 NUMA节点):

```
方法              时间(s)   vs基线
────────────────────────────────
默认分配          0.450     1.0x
交错分配          0.380     1.2x
First-touch       0.285     1.6x
显式NUMA          0.270     1.7x  ← 目标
```

**扩展任务**:

1. ✅ 使用`numastat -p <pid>`观察内存分布
2. ✅ 使用`perf stat -e node-loads,node-load-misses`测量跨节点访问
3. ✅ 实验不同的线程-节点绑定策略

---

## 4. 高级练习

### 练习 3.1: 实现高性能哈希表

**目标**: 综合应用无锁编程、缓存优化、SIMD

**要求**:

设计并实现一个高性能哈希表，支持：
- 并发插入、查找
- 高缓存命中率
- 低锁竞争或无锁

**起始模板**:

```cpp
// 03-advanced/fast_hashtable.cpp
#include <atomic>
#include <functional>
#include <vector>
#include <iostream>

template<typename K, typename V>
class FastHashTable {
private:
    static const size_t NUM_SEGMENTS = 64;

    struct alignas(64) Entry {
        std::atomic<K> key;
        std::atomic<V> value;
        std::atomic<bool> occupied;

        Entry() : occupied(false) {}
    };

    struct alignas(64) Segment {
        Entry* entries;
        size_t capacity;
        std::atomic<size_t> size;

        Segment(size_t cap) : capacity(cap), size(0) {
            entries = new Entry[capacity];
        }

        ~Segment() {
            delete[] entries;
        }
    };

    std::vector<Segment*> segments;
    std::hash<K> hasher;

    size_t hash1(const K& key) const {
        return hasher(key);
    }

    size_t hash2(const K& key) const {
        return hasher(key) * 2654435761u;
    }

public:
    FastHashTable(size_t capacity_per_segment = 1024) {
        for (size_t i = 0; i < NUM_SEGMENTS; i++) {
            segments.push_back(new Segment(capacity_per_segment));
        }
    }

    ~FastHashTable() {
        for (auto seg : segments) {
            delete seg;
        }
    }

    // TODO: 实现高性能插入
    bool insert(const K& key, const V& value) {
        size_t h = hash1(key);
        size_t seg_idx = h % NUM_SEGMENTS;
        Segment* seg = segments[seg_idx];

        size_t idx = h % seg->capacity;

        // 线性探测
        for (size_t i = 0; i < seg->capacity; i++) {
            size_t pos = (idx + i) % seg->capacity;
            Entry& entry = seg->entries[pos];

            bool expected = false;
            if (entry.occupied.compare_exchange_strong(
                expected, true,
                std::memory_order_acquire,
                std::memory_order_relaxed)) {

                entry.key.store(key, std::memory_order_relaxed);
                entry.value.store(value, std::memory_order_release);
                seg->size.fetch_add(1, std::memory_order_relaxed);
                return true;
            }

            if (entry.key.load(std::memory_order_relaxed) == key) {
                entry.value.store(value, std::memory_order_release);
                return true;
            }
        }

        return false;  // 满了
    }

    // TODO: 实现高性能查找
    bool find(const K& key, V& value) const {
        size_t h = hash1(key);
        size_t seg_idx = h % NUM_SEGMENTS;
        const Segment* seg = segments[seg_idx];

        size_t idx = h % seg->capacity;

        for (size_t i = 0; i < seg->capacity; i++) {
            size_t pos = (idx + i) % seg->capacity;
            const Entry& entry = seg->entries[pos];

            if (!entry.occupied.load(std::memory_order_acquire)) {
                return false;
            }

            if (entry.key.load(std::memory_order_relaxed) == key) {
                value = entry.value.load(std::memory_order_acquire);
                return true;
            }
        }

        return false;
    }

    size_t size() const {
        size_t total = 0;
        for (const auto* seg : segments) {
            total += seg->size.load(std::memory_order_relaxed);
        }
        return total;
    }
};
```

**基准测试**:

```cpp
#include <thread>
#include <chrono>
#include <random>

void benchmark_hashtable() {
    const int NUM_THREADS = 8;
    const int OPS_PER_THREAD = 1000000;

    FastHashTable<uint64_t, uint64_t> table;

    // 插入基准测试
    auto insert_worker = [&](int tid) {
        std::mt19937_64 gen(tid);
        for (int i = 0; i < OPS_PER_THREAD; i++) {
            uint64_t key = gen();
            table.insert(key, key * 2);
        }
    };

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(insert_worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();

    std::cout << "Insert time: " << duration << " ms" << std::endl;
    std::cout << "Insert throughput: "
              << (NUM_THREADS * OPS_PER_THREAD * 1000.0) / duration / 1e6
              << " Mops/s" << std::endl;

    // 查找基准测试
    // ... (类似实现)
}
```

**性能目标**:

```
操作          吞吐量(Mops/s)
────────────────────────────
Insert        15-25
Lookup        40-60
Mixed(50/50)  25-35
```

**优化方向**:

1. ✅ 使用Hopscotch hashing减少探测距离
2. ✅ 使用SIMD加速批量查找
3. ✅ 实现RCU用于读多写少场景
4. ✅ 使用预取优化链表遍历

---

### 练习 3.2: eBPF性能监控

**目标**: 学习使用eBPF进行低开销性能监控

**任务**: 编写eBPF程序监控系统调用延迟

**代码**:

```c
// 03-advanced/syscall_latency.bpf.c
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, u32);      // tid
    __type(value, u64);    // start time
} start_times SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 256);
    __type(key, u32);      // syscall number
    __type(value, u64);    // total latency
} syscall_latency SEC(".maps");

// 系统调用入口
SEC("tracepoint/raw_syscalls/sys_enter")
int trace_sys_enter(struct trace_event_raw_sys_enter *ctx) {
    u32 tid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();

    bpf_map_update_elem(&start_times, &tid, &ts, BPF_ANY);
    return 0;
}

// 系统调用退出
SEC("tracepoint/raw_syscalls/sys_exit")
int trace_sys_exit(struct trace_event_raw_sys_exit *ctx) {
    u32 tid = bpf_get_current_pid_tgid();
    u64 *start_ts = bpf_map_lookup_elem(&start_times, &tid);

    if (!start_ts)
        return 0;

    u64 delta = bpf_ktime_get_ns() - *start_ts;

    // 记录延迟
    u32 syscall_nr = ctx->id;
    u64 *latency = bpf_map_lookup_elem(&syscall_latency, &syscall_nr);

    if (latency) {
        __sync_fetch_and_add(latency, delta);
    } else {
        bpf_map_update_elem(&syscall_latency, &syscall_nr,
                           &delta, BPF_ANY);
    }

    bpf_map_delete_elem(&start_times, &tid);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
```

**用户态程序**:

```cpp
// 03-advanced/syscall_monitor.cpp
#include <iostream>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <signal.h>

static bool running = true;

void signal_handler(int sig) {
    running = false;
}

int main() {
    struct bpf_object *obj;
    struct bpf_link *link_enter, *link_exit;

    // 加载BPF程序
    obj = bpf_object__open_file("syscall_latency.bpf.o", NULL);
    if (!obj) {
        fprintf(stderr, "Failed to open BPF object\n");
        return 1;
    }

    if (bpf_object__load(obj)) {
        fprintf(stderr, "Failed to load BPF object\n");
        return 1;
    }

    // 附加到tracepoint
    struct bpf_program *prog_enter, *prog_exit;
    prog_enter = bpf_object__find_program_by_name(
        obj, "trace_sys_enter"
    );
    prog_exit = bpf_object__find_program_by_name(
        obj, "trace_sys_exit"
    );

    link_enter = bpf_program__attach(prog_enter);
    link_exit = bpf_program__attach(prog_exit);

    // 获取map fd
    int latency_fd = bpf_object__find_map_fd_by_name(
        obj, "syscall_latency"
    );

    signal(SIGINT, signal_handler);

    std::cout << "Monitoring syscall latency... Press Ctrl-C to stop"
              << std::endl;

    // 定期输出统计
    while (running) {
        sleep(5);

        std::cout << "\n=== Syscall Latency (top 10) ===" << std::endl;

        uint32_t key, next_key;
        uint64_t value;

        // 遍历map
        key = 0;
        while (bpf_map_get_next_key(latency_fd, &key, &next_key) == 0) {
            bpf_map_lookup_elem(latency_fd, &next_key, &value);

            std::cout << "Syscall " << next_key << ": "
                      << value / 1000 << " us total" << std::endl;

            key = next_key;
        }
    }

    // 清理
    bpf_link__destroy(link_enter);
    bpf_link__destroy(link_exit);
    bpf_object__close(obj);

    return 0;
}
```

**编译运行**:

```bash
# 安装依赖
sudo apt-get install clang llvm libbpf-dev

# 编译BPF程序
clang -O2 -target bpf -c syscall_latency.bpf.c -o syscall_latency.bpf.o

# 编译用户态程序
g++ -O2 syscall_monitor.cpp -o syscall_monitor -lbpf

# 运行 (需要root)
sudo ./syscall_monitor
```

**学习要点**:
- eBPF的安全沙箱机制
- Map的使用和数据传递
- 低开销监控的实现
- tracepoint vs kprobe

---

### 练习 3.3: DPDK网络加速

**目标**: 学习内核旁路技术

**场景**: 实现高性能包转发器

**代码框架**:

```c
// 03-advanced/dpdk_forwarder.c
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_cycles.h>

#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32

static const struct rte_eth_conf port_conf_default = {
    .rxmode = {
        .max_rx_pkt_len = RTE_ETHER_MAX_LEN,
    },
};

// 初始化端口
static inline int
port_init(uint16_t port, struct rte_mempool *mbuf_pool) {
    struct rte_eth_conf port_conf = port_conf_default;
    const uint16_t rx_rings = 1, tx_rings = 1;
    uint16_t nb_rxd = RX_RING_SIZE;
    uint16_t nb_txd = TX_RING_SIZE;

    // 配置网卡
    int retval = rte_eth_dev_configure(port, rx_rings, tx_rings,
                                       &port_conf);
    if (retval != 0)
        return retval;

    // 设置RX队列
    retval = rte_eth_rx_queue_setup(port, 0, nb_rxd,
                                     rte_eth_dev_socket_id(port),
                                     NULL, mbuf_pool);
    if (retval < 0)
        return retval;

    // 设置TX队列
    retval = rte_eth_tx_queue_setup(port, 0, nb_txd,
                                     rte_eth_dev_socket_id(port),
                                     NULL);
    if (retval < 0)
        return retval;

    // 启动网卡
    retval = rte_eth_dev_start(port);
    if (retval < 0)
        return retval;

    // 启用混杂模式
    rte_eth_promiscuous_enable(port);

    return 0;
}

// 转发逻辑
static void
lcore_main(void) {
    uint16_t port_in = 0, port_out = 1;
    struct rte_mbuf *bufs[BURST_SIZE];

    printf("Core %u forwarding packets from port %u to %u\n",
           rte_lcore_id(), port_in, port_out);

    uint64_t total_packets = 0;
    uint64_t last_tsc = rte_rdtsc();

    while (1) {
        // 接收包
        const uint16_t nb_rx = rte_eth_rx_burst(
            port_in, 0, bufs, BURST_SIZE
        );

        if (unlikely(nb_rx == 0))
            continue;

        // 处理包 (这里只是简单转发)
        // TODO: 添加你的包处理逻辑

        // 发送包
        const uint16_t nb_tx = rte_eth_tx_burst(
            port_out, 0, bufs, nb_rx
        );

        // 释放未发送的包
        if (unlikely(nb_tx < nb_rx)) {
            for (uint16_t i = nb_tx; i < nb_rx; i++) {
                rte_pktmbuf_free(bufs[i]);
            }
        }

        total_packets += nb_rx;

        // 每秒输出统计
        uint64_t cur_tsc = rte_rdtsc();
        if (cur_tsc - last_tsc > rte_get_tsc_hz()) {
            printf("Packets/sec: %lu\n", total_packets);
            total_packets = 0;
            last_tsc = cur_tsc;
        }
    }
}

int
main(int argc, char *argv[]) {
    struct rte_mempool *mbuf_pool;
    unsigned nb_ports;

    // 初始化EAL
    int ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");

    argc -= ret;
    argv += ret;

    nb_ports = rte_eth_dev_count_avail();
    if (nb_ports < 2)
        rte_exit(EXIT_FAILURE, "Need at least 2 ports\n");

    // 创建mbuf池
    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL",
                                        NUM_MBUFS * nb_ports,
                                        MBUF_CACHE_SIZE, 0,
                                        RTE_MBUF_DEFAULT_BUF_SIZE,
                                        rte_socket_id());
    if (mbuf_pool == NULL)
        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

    // 初始化端口
    if (port_init(0, mbuf_pool) != 0)
        rte_exit(EXIT_FAILURE, "Cannot init port 0\n");

    if (port_init(1, mbuf_pool) != 0)
        rte_exit(EXIT_FAILURE, "Cannot init port 1\n");

    // 启动转发
    lcore_main();

    return 0;
}
```

**编译运行**:

```bash
# 安装DPDK
sudo apt-get install dpdk dpdk-dev

# 编译
gcc -O3 dpdk_forwarder.c -o dpdk_forwarder \
    $(pkg-config --cflags --libs libdpdk)

# 配置大页
echo 1024 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# 绑定网卡到DPDK
sudo dpdk-devbind.py --bind=uio_pci_generic 0000:03:00.0

# 运行
sudo ./dpdk_forwarder
```

**性能目标**:

```
包大小(B)    PPS(Mpps)    吞吐量(Gbps)
────────────────────────────────────────
64           14.88        10.0
128          14.88        20.0
256          14.88        40.0
1500         8.33         100.0  (线速)
```

---

## 5. 挑战练习

### 挑战 1: 优化真实应用

**任务**: 选择一个开源项目，进行性能优化

**建议项目**:
1. **Redis** - 内存数据库
2. **Nginx** - Web服务器
3. **SQLite** - 嵌入式数据库
4. **FFmpeg** - 视频处理

**目标**: 实现至少2x的性能提升

**步骤**:

1. ✅ 下载和编译项目
2. ✅ 建立性能基线
3. ✅ 使用perf/VTune找到热点
4. ✅ 应用学到的优化技术
5. ✅ 验证优化效果
6. ✅ 提交patch到上游项目

---

### 挑战 2: 性能优化竞赛

**场景**: 给定一个慢程序，优化到最快

**示例题目**:

```cpp
// challenge/sort_competition.cpp
// 任务: 优化这个排序程序

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

const int N = 100000000;  // 1亿个元素

// 朴素实现
void baseline_sort(std::vector<int>& data) {
    std::sort(data.begin(), data.end());
}

int main() {
    // 生成随机数据
    std::vector<int> data(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1000000);

    for (int& val : data) {
        val = dis(gen);
    }

    // 基准测试
    auto start = std::chrono::high_resolution_clock::now();
    baseline_sort(data);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<
        std::chrono::seconds>(end - start).count();

    std::cout << "Time: " << duration << " seconds" << std::endl;

    return 0;
}
```

**优化方向**:

1. ✅ 使用并行排序 (std::sort vs __gnu_parallel::sort)
2. ✅ 使用Radix sort (特定数据范围)
3. ✅ SIMD优化比较操作
4. ✅ NUMA感知分区
5. ✅ 使用GPU排序 (Thrust)

**目标**: 从~35秒优化到< 3秒 (>10x speedup)

---

## 6. 性能优化竞赛题目

### 题目 1: JSON解析器优化

**给定**: 一个慢速JSON解析器
**任务**: 优化到接近simdjson的速度
**难度**: ⭐⭐⭐⭐

### 题目 2: 图像模糊滤波

**给定**: 朴素的高斯模糊实现
**任务**: 使用SIMD + 多线程优化
**目标**: > 20x加速
**难度**: ⭐⭐⭐

### 题目 3: 数据库查询引擎

**给定**: 简单的SELECT查询引擎
**任务**: 实现向量化执行
**目标**: > 10x加速
**难度**: ⭐⭐⭐⭐⭐

---

## 附录: 学习路径

### 初学者 (1-2个月)

```
Week 1-2: 练习1.1-1.5 (基础性能分析)
Week 3-4: 练习2.1-2.2 (矩阵乘法、无锁队列)
Week 5-6: 练习2.3-2.4 (IO、NUMA)
Week 7-8: 总结和复习
```

### 进阶 (2-3个月)

```
Week 1-4: 练习3.1-3.3 (哈希表、eBPF、DPDK)
Week 5-8: 挑战1 (优化真实项目)
Week 9-12: 挑战2 + 竞赛题目
```

### 资源推荐

- **书籍**: Systems Performance (Brendan Gregg)
- **课程**: CMU 15-418 Parallel Computing
- **论坛**: Stack Overflow, Reddit r/programming
- **代码**: Linux内核, DPDK, Folly

---

**练习手册结束**

通过这些练习，你将掌握：
- 使用perf/VTune进行性能分析
- 缓存优化和数据布局
- SIMD向量化编程
- 无锁并发编程
- NUMA系统优化
- IO和网络优化
- eBPF和内核编程

祝你在性能优化的道路上不断进步！🚀
