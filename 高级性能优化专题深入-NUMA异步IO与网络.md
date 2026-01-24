# 高级性能优化专题深入 - NUMA、异步I/O 与网络优化

## 课程简介

本课程深入探讨高级性能优化专题，包括 NUMA 优化、异步 I/O、网络优化、内存分配器优化等，这些技术对于构建超大规模、高性能的系统至关重要。

**适合人群**：
- 需要处理大规模数据的系统架构师
- 追求极致性能的后端工程师
- 分布式系统开发者

**前置知识**：
- 已完成《Faiss 性能优化技术课程》
- 熟悉操作系统和体系结构基础
- 了解多线程编程

---

## 第一部分：NUMA 优化

### 1.1 NUMA 架构概述

**什么是 NUMA？**

NUMA (Non-Uniform Memory Access) 是一种多处理器系统架构，其中每个处理器都有自己的本地内存，访问本地内存比访问远程内存快。

```
传统 UMA 架构:
┌─────────────────────────────────────────────┐
│                 内存                          │
│              (所有 CPU 共享)                   │
│                                              │
│  CPU0   CPU1   CPU2   CPU3                   │
└─────────────────────────────────────────────┘

NUMA 架构:
┌────────────────┐  ┌────────────────┐
│  Node 0        │  │  Node 1        │
│                │  │                │
│  CPU0  CPU1     │  │  CPU2  CPU3     │
│    |    |       │  │    |    |       │
│  └────┴────┐    │  │  └────┴────┐    │
│  本地内存 1TB   │  │  本地内存 1TB   │
│  (快速)        │  │  (快速)        │
│  └─────────────┘  │  └─────────────┘
│         │         │  │         │
│         └─────────┴────────┘
│         QPI/UPI 互连 (较慢)
│         (访问远程内存慢)
└───────────────────────────────────────────┘
```

**延迟对比**：
- 本地内存：~80-100 ns
- 远程内存：~150-200 ns
- 跨 QPI 互连：~200-300 ns

### 1.2 NUMA 检测和分析

```cpp
#include <numa.h>
#include <stdio.h>

void print_numa_info() {
    if (numa_available() < 0) {
        printf("NUMA not available on this system\n");
        return;
    }

    int num_nodes = numa_num_configured_nodes();
    printf("NUMA Nodes: %d\n\n", num_nodes);

    for (int i = 0; i < num_nodes; i++) {
        long long node_size = numa_node_size64(i);
        printf("Node %d:\n", i);
        printf("  Size: %.2f GB\n", node_size / (1024.0 * 1024 * 1024));

        // 获取该节点的 CPU
        struct bitmask* cpus = numa_allocate_cpumask();
        numa_node_to_cpus(i, cpus);

        printf("  CPUs: ");
        for (int j = 0; j < CPU_SETSIZE; j++) {
            if (numa_bitmask_isset(cpus, j)) {
                printf("%d ", j);
            }
        }
        printf("\n");

        numa_free_cpumask(cpus);

        // 获取内存距离
        printf("  Distance Matrix:\n");
        for (int j = 0; j < num_nodes; j++) {
            int distance = numa_distance(i, j);
            printf("    to Node %d: %d\n", j, distance);
        }
        printf("\n");
    }
}

int main() {
    print_numa_info();
    return 0;
}
```

**输出示例**：
```
NUMA Nodes: 2

Node 0:
  Size: 192.00 GB
  CPUs: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
  Distance Matrix:
    to Node 0: 10
    to Node 1: 20

Node 1:
  Size: 192.00 GB
  CPUs: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
  Distance Matrix:
    to Node 0: 20
    to Node 1: 10
```

### 1.3 NUMA 感知的内存分配

```cpp
#include <numa.h>

// NUMA 感知的向量数据库
class NUMAAwareVectorIndex {
    size_t dimension;
    size_t num_vectors;
    int preferred_node;  // 首选 NUMA 节点

    // 在特定节点上分配内存
    float* vectors;  // [num_vectors, dimension]

public:
    NUMAAwareVectorIndex(size_t d, int node = -1)
        : dimension(d), num_vectors(0), preferred_node(node) {

        if (preferred_node >= 0) {
            // 在指定节点上分配内存
            numa_set_preferred(preferred_node);
            vectors = (float*)numa_alloc_local(
                d * sizeof(float) * 1000000,  // 预分配 100 万向量
                preferred_node
            );
        } else {
            // 让系统自动选择
            vectors = (float*)aligned_alloc(64, d * sizeof(float) * 1000000);
        }
    }

    void add_vector(size_t id, const float* vector, int thread_id) {
        // 根据线程 ID 选择 NUMA 节点
        int node = thread_id % numa_num_configured_nodes();

        // 如果向量不在本地节点，先迁移
        numa_set_preferred(node);

        // 添加向量
        for (size_t i = 0; i < dimension; i++) {
            vectors[num_vectors * dimension + i] = vector[i];
        }

        num_vectors++;
    }

    // NUMA 优化的搜索
    void search_parallel(
        const float* query,
        size_t k,
        float* distances,
        size_t* ids) {

        int num_threads = omp_get_max_threads();
        int nodes = numa_num_configured_nodes();

        // 每个 NUMA 节点分配一组线程
        std::vector<std::vector<std::pair<float, size_t>>> local_results(
            nodes
        );

        #pragma omp parallel num_threads(num_threads)
        {
            int thread_id = omp_get_thread_num();
            int node = thread_id % nodes;

            // 绑定到 NUMA 节点
            numa_run_on_node(node);
            numa_set_preferred(node);

            // 分配工作：每个线程处理一部分向量
            size_t chunk_size = num_vectors / (num_threads / nodes);
            size_t start = (thread_id / nodes) * chunk_size;
            size_t end = start + chunk_size;

            std::vector<std::pair<float, size_t>> results;

            for (size_t i = start; i < end; i++) {
                float dist = 0;
                for (size_t j = 0; j < dimension; j++) {
                    float diff = query[j] - vectors[i * dimension + j];
                    dist += diff * diff;
                }
                results.push_back({dist, i});
            }

            local_results[node] = results;
        }

        // 合并结果
        std::vector<std::pair<float, size_t>> all_results;
        for (auto& results : local_results) {
            all_distances.insert(
                all_results.end(),
                results.begin(),
                results.end()
            );
        }

        // 排序并返回 Top-K
        std::sort(all_results.begin(), all_results.end());
        for (size_t i = 0; i < k && i < all_results.size(); i++) {
            distances[i] = all_results[i].first;
            ids[i] = all_results[i].second;
        }
    }
};
```

### 1.4 NUMA 优化技巧

#### 1.4.1 数据局部性优化

```cpp
// 策略 1: 数据分片（Sharding）

class NUMAShardedIndex {
    struct Shard {
        int numa_node;
        float* vectors;  // 本地内存
        size_t size;
        size_t capacity;
    };

    std::vector<Shard> shards;

public:
    NUMAShardedIndex(size_t d, size_t total_capacity) {
        int num_nodes = numa_num_configured_nodes();
        shards.resize(num_nodes);

        for (int i = 0; i < num_nodes; i++) {
            shards[i].numa_node = i;
            shards[i].size = 0;
            shards[i].capacity = total_capacity / num_nodes;

            // 在本地节点分配内存
            numa_run_on_node(i);
            numa_set_preferred(i);

            shards[i].vectors = (float*)numa_alloc_local(
                d * sizeof(float) * shards[i].capacity,
                i
            );
        }
    }

    void add_vector(const float* vector, int shard_hint = -1) {
        int shard_id;

        if (shard_hint >= 0) {
            shard_id = shard_hint % shards.size();
        } else {
            // 轮询分配
            shard_id = rand() % shards.size();
        }

        auto& shard = shards[shard_id];

        // 确保在正确的节点上操作
        numa_run_on_node(shard.numa_node);

        if (shard.size >= shard.capacity) {
            // 扩容
            shard.capacity *= 2;
            float* new_vectors = (float*)numa_alloc_local(
                dimension * sizeof(float) * shard.capacity,
                shard.numa_node
            );
            memcpy(new_vectors, shard.vectors,
                   shard.size * dimension * sizeof(float));
            numa_free(shard.vectors, shard.numa_node);
            shard.vectors = new_vectors;
        }

        // 添加向量
        for (size_t i = 0; i < dimension; i++) {
            shard.vectors[shard.size * dimension + i] = vector[i];
        }
        shard.size++;
    }
};
```

#### 1.4.2 线程-节点绑定

```cpp
// 策略 2: 线程与节点严格绑定

void create_numa_aware_threads(int num_threads) {
    int num_nodes = numa_num_configured_nodes();
    int threads_per_node = num_threads / num_nodes;

    pthread_t threads[num_threads];
    ThreadArgs args[num_threads];

    for (int i = 0; i < num_threads; i++) {
        int node = i % num_nodes;
        int cpu = i % num_nodes * threads_per_node + (i / num_nodes);

        args[i].node_id = node;
        args[i].cpu_id = cpu;

        pthread_create(
            &threads[i],
            nullptr,
            numa_aware_worker,
            &args[i]
        );
    }

    // 等待线程完成
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }
}

void* numa_aware_worker(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;

    // 绑定到 NUMA 节点
    numa_run_on_node(args->node_id);
    numa_set_preferred(args->node_id);

    // 绑定到特定 CPU
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(args->cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    // 执行工作
    do_work(args->node_id);

    return nullptr;
}
```

### 1.5 NUMA 性能测试

```cpp
#include <chrono>

void benchmark_numa() {
    const size_t d = 128;
    const size_t num_vectors = 10000000;
    const size_t num_queries = 1000;

    printf("=== NUMA Optimization Benchmark ===\n");

    // 测试 1: 无 NUMA 优化
    {
        printf("\nTest 1: No NUMA optimization\n");
        NonNUMAIndex index(d);
        auto start = std::chrono::high_resolution_clock::now();
        // ... 添加和搜索 ...
        auto end = std::chrono::high_resolution_clock::now();
        printf("Time: %.2f ms\n",
               std::chrono::duration<double, std::milli>(end - start).count());
    }

    // 测试 2: NUMA 感知分配
    {
        printf("\nTest 2: NUMA-aware allocation\n");
        NUMAAwareVectorIndex index(d, 0);
        auto start = std::chrono::high_resolution_clock::now();
        // ... 添加和搜索 ...
        auto end = std::chrono::high_resolution_clock::now();
        printf("Time: %.2f ms\n",
               std::chrono::duration<double, std::milli>(end - start).count());
    }

    // 测试 3: NUMA 分片
    {
        printf("\nTest 3: NUMA sharding\n");
        NUMAShardedIndex index(d, num_vectors);
        auto start = std::chrono::high_resolution_clock::now();
        // ... 添加和搜索 ...
        auto end = std::chrono::high_resolution_clock::now();
        printf("Time: %.2f ms\n",
               std::chrono::duration<double, std::milli>(end - start).count());
    }
}
```

---

## 第二部分：异步 I/O 优化

### 2.1 异步 I/O 概述

**为什么需要异步 I/O？**

在同步 I/O 模型中，线程在等待 I/O 完成时被阻塞，浪费 CPU 资源。异步 I/O 允许线程在等待 I/O 时继续执行其他任务。

**同步 vs 异步**：

```cpp
// 同步 I/O：阻塞
void sync_io_read(const char* filename, float* buffer, size_t size) {
    int fd = open(filename, O_RDONLY);
    read(fd, buffer, size);  // 阻塞直到完成
    close(fd);
}

// 异步 I/O：非阻塞
void async_io_read(const char* filename, float* buffer, size_t size) {
    int fd = open(filename, O_RDONLY | O_NONBLOCK);

    // 提交异步读请求
    struct io_iocb cb;
    io_prep_pread(&cb, fd, buffer, size, 0);

    // 提交请求
    io_submit(aio_context, 1, &cb);

    // 可以做其他工作...

    // 等待完成
    struct io_event events[1];
    io_getevents(aio_context, 1, 1, events, nullptr);

    close(fd);
}
```

### 2.2 Linux io_uring 深度解析

**io_uring** 是 Linux 5.1+ 引入的新异步 I/O 接口，比传统的 AIO 更高效。

```cpp
#include <liburing.h>
#include <unistd.h>
#include <fcntl.h>

class AsyncIOVectorReader {
    struct io_uring ring;
    int fd;

public:
    AsyncIOVectorReader(const char* filename) {
        // 打开文件
        fd = open(filename, O_RDONLY | O_DIRECT);
        if (fd < 0) {
            throw std::runtime_error("Failed to open file");
        }

        // 初始化 io_uring
        int ret = io_uring_queue_init_params(
            32,    // 队列深度
            4,     // 完成队列深度
            0,     // flags
            &ring,
            nullptr
        );

        if (ret < 0) {
            throw std::runtime_error("Failed to init io_uring");
        }
    }

    ~AsyncIOReader() {
        io_uring_queue_exit(&ring);
        close(fd);
    }

    // 异步读取多个向量
    void read_vectors_async(
            float* buffers[],
            size_t num_buffers,
            size_t vector_size) {

        struct io_uring_sqe *sqe;
        struct io_uring_cqe *cqe;

        // 提交多个读请求
        for (size_t i = 0; i < num_buffers; i++) {
            sqe = io_uring_get_sqe(&ring);

            // 准备读操作
            io_uring_prep_read(
                sqe,
                fd,
                buffers[i],
                vector_size * sizeof(float),
                lseek(fd, 0, SEEK_CUR),  // offset
                0
            );

            // 设置用户数据
            sqe->user_data = (unsigned long)i;

            // 提交
            io_uring_submit(&ring);
        }

        // 等待所有请求完成
        size_t completed = 0;
        while (completed < num_buffers) {
            unsigned head;
            unsigned count = 0;

            // 等待至少一个完成
            int ret = io_uring_wait_cqe(&ring, &cqe, 1);
            if (ret < 0) {
                perror("io_uring_wait_cqe");
                break;
            }

            // 处理完成的事件
            io_uring_for_each_cqe(&ring, head, count) {
                // 检查错误
                if (cqe->res < 0) {
                    fprintf(stderr, "I/O error: %d\n", cqe->res);
                }

                completed++;
                io_uring_cqe_seen(&ring, cqe);
            }
        }
    }

    // 批量读取向量数据库
    void read_vector_database(
            float*& database,
            size_t num_vectors,
            size_t dimension) {

        // 计算文件偏移
        size_t vector_size = dimension * sizeof(float);
        std::vector<size_t> offsets(num_vectors);
        for (size_t i = 0; i < num_vectors; i++) {
            offsets[i] = i * vector_size;
        }

        // 分配缓冲区
        database = (float*)aligned_alloc(64, num_vectors * vector_size);

        // 使用 io_uring 批量读取
        struct io_uring_sqe *sqe;
        const int batch_size = 32;

        for (size_t batch = 0; batch < num_vectors; batch += batch_size) {
            size_t batch_end = std::min(batch + batch_size, num_vectors);

            // 提交批量读请求
            for (size_t i = batch; i < batch_end; i++) {
                sqe = io_uring_get_sqe(&ring);
                io_uring_prep_read(
                    sqe,
                    fd,
                    &database[i * dimension],
                    vector_size,
                    offsets[i],
                    0
                );
                sqe->user_data = (unsigned long)i;
                io_uring_submit(&ring);
            }

            // 等待这批完成
            size_t batch_completed = 0;
            while (batch_completed < batch_end - batch) {
                struct io_uring_cqe *cqe;
                int ret = io_uring_wait_cqe(&ring, &cqe, 1);

                if (ret < 0) break;

                io_uring_for_each_cqe(&ring, 0, 1) {
                    batch_completed++;
                    io_uring_cqe_seen(&ring, cqe);
                }
            }
        }
    }
};
```

### 2.3 io_uring 优化技巧

#### 2.3.1 批量提交和等待

```cpp
// 优化：批量处理 I/O 请求
void io_uring_batch_optimized(
        const char* filename,
        std::vector<std::pair<size_t, float*>>& io_requests) {

    struct io_uring ring;
    io_uring_queue_init_params(256, 128, 0, &ring, nullptr);

    int fd = open(filename, O_RDONLY | O_DIRECT);

    // 批量提交
    const int batch_size = 64;
    size_t submitted = 0;

    while (submitted < io_requests.size()) {
        size_t batch_end = std::min(
            submitted + batch_size,
            io_requests.size()
        );

        // 提交一批请求
        for (size_t i = submitted; i < batch_end; i++) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            io_uring_prep_read(
                sqe,
                fd,
                io_requests[i].second,
                io_requests[i].first,
                0,  // offset 由内核管理
                0
            );
            io_uring_submit(&ring);
        }

        submitted = batch_end;

        // 批量等待完成
        int completed = 0;
        while (completed < batch_end - (batch_end - batch_size)) {
            struct io_uring_cqe *cqes[batch_size];
            int ret = io_uring_wait_cqe_nr(&ring, cqes, batch_size, nullptr);

            for (int i = 0; i < ret; i++) {
                completed++;
                io_uring_cqe_seen(&ring, cqes[i]);
            }
        }
    }

    close(fd);
    io_uring_queue_exit(&ring);
}
```

#### 2.3.2 零拷贝优化

```cpp
// 使用 mmap 和 io_uring 实现零拷贝

class ZeroCopyVectorReader {
    float* mapped_data;
    size_t file_size;

public:
    ZeroCopyVectorReader(const char* filename) {
        int fd = open(filename, O_RDONLY);

        // 获取文件大小
        struct stat st;
        fstat(fd, &st);
        file_size = st.st_size;

        // 映射到内存
        mapped_data = (float*)mmap(
            nullptr,
            file_size,
            PROT_READ,
            MAP_PRIVATE | MAP_POPULATE,
            fd,
            0
        );

        // 建议内核预读
        posix_madvise(mapped_data, file_size, POSIX_MADV_SEQUENTIAL);

        close(fd);  // 映射后可以关闭文件描述符
    }

    ~ZeroCopyVectorReader() {
        munmap(mapped_data, file_size);
    }

    // 获取向量（无需拷贝）
    const float* get_vector(size_t index, size_t dimension) {
        return &mapped_data[index * dimension];
    }

    // 使用 io_uring 异步预取
    void async_prefetch(
            std::vector<size_t> indices,
            size_t dimension) {

        struct io_uring ring;
        io_uring_queue_init_params(64, 32, 0, &ring, nullptr);

        // 提交预取请求
        for (size_t idx : indices) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);

            // 使用 MADVISE_DONTNEED 预取
            io_uring_prep_read(
                sqe,
                0,  // dummy fd
                &mapped_data[idx * dimension],
                dimension * sizeof(float),
                0,
                IOSQE_FIXED_FILE
            );
            io_uring_submit(&ring);
        }

        io_uring_queue_exit(&ring);
    }
};
```

---

## 第三部分：网络优化

### 3.1 TCP 优化

```cpp
#include <sys/socket.h>
#include <netinet/tcp.h>

// 优化的 TCP 套接
void optimize_tcp_socket(int sockfd) {
    // 禁用 Nagle 算法（减少延迟）
    int flag = 1;
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    // 启用 TCP Fast Open
    int_fast = 5;  // 最大快速打开重试次数
    setsockopt(sockfd, IPPROTO_TCP, TCP_FASTOPEN, &int_fast, sizeof(int_fast));

    // 设置缓冲区大小
    int buf_size = 256 * 1024;  // 256KB
    setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &buf_size, sizeof(buf_size));
    setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buf_size, sizeof(buf_size));

    // 启用 keep-alive
    int keepalive = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive));

    int keepidle = 30;   // 30 秒后开始探测
    int keepintvl = 10;  // 探测间隔 10 秒
    int keepcnt = 3;     // 最多探测 3 次

    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPIDLE, &keepidle, sizeof(keepidle));
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPINTVL, &keepintvl, sizeof(keepintvl));
    setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPCNT, &keepcnt, sizeof(keepcnt));

    // 设置重用地址
    int reuse = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
}

// 高性能 TCP 服务器
class HighPerformanceServer {
    int listen_fd;
    int epoll_fd;
    struct epoll_event events[1024];

public:
    HighPerformanceServer(int port) {
        // 创建 socket
        listen_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
        optimize_tcp_socket(listen_fd);

        // 绑定端口
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);

        bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr));

        // 监听
        listen(listen_fd, 1024);

        // 创建 epoll
        epoll_fd = epoll_create1(0);

        // 添加监听 socket
        struct epoll_event ev;
        ev.events = EPOLLIN | EPOLLET;
        ev.data.fd = listen_fd;
        epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listen_fd, &ev);
    }

    void run() {
        while (true) {
            int nfds = epoll_wait(epoll_fd, events, 1024, -1);

            for (int i = 0; i < nfds; i++) {
                if (events[i].data.fd == listen_fd) {
                    // 新连接
                    accept_new_connection();
                } else {
                    // 数据到达
                    handle_client(events[i].data.fd);
                }
            }
        }
    }

private:
    void accept_new_connection() {
        while (true) {
            struct sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);

            int client_fd = accept(listen_fd,
                                  (struct sockaddr*)&client_addr,
                                  &addr_len);

            if (client_fd < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    break;
                }
                continue;
            }

            // 优化客户端连接
            optimize_tcp_socket(client_fd);

            // 添加到 epoll
            struct epoll_event ev;
            ev.events = EPOLLIN | EPOLLET | EPOLLRDHUP;
            ev.data.fd = client_fd;
            epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev);
        }
    }

    void handle_client(int client_fd) {
        // 读取数据
        char buffer[8192];
        while (true) {
            ssize_t n = read(client_fd, buffer, sizeof(buffer));

            if (n <= 0) {
                if (n == 0 || errno == ECONNRESET) {
                    // 连接关闭
                    close(client_fd);
                } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // 数据读完
                    break;
                }
                continue;
            }

            // 处理请求
            process_request(buffer, n);
        }
    }

    void process_request(const char* buffer, size_t size) {
        // 解析请求
        // 执行向量搜索
        // 发送响应
    }
};
```

### 3.2 RDMA 优化

```cpp
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>

// RDMA 向量搜索服务器
class RDMAVectorServer {
    struct rdma_cm_id* listen_id;
    struct rdma_event_channel* ec;

    struct rdma_mem_desc* vector_mem;  // 向量数据库的 RDMA 内存

public:
    RDMAVectorServer(int port) {
        // 创建事件通道
        ec = rdma_create_event_channel();

        // 创建监听 ID
        rdma_create_id(ec, &listen_id);

        // 绑定地址
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = htonl(INADDR_ANY);

        rdma_bind_addr(listen_id, (struct sockaddr*)&addr, sizeof(addr));

        // 监听
        rdma_listen(listen_id, 10);

        // 注册向量数据库内存
        size_t db_size = 1000000000 * 128 * sizeof(float);  // 10 亿向量
        posix_memalign((void**)&vector_mem, 4096, db_size);

        rdma_reg_msgs(ec, vector_mem, db_size, &vector_mem);
    }

    void run() {
        while (true) {
            struct rdma_cm_event* event;
            rdma_get_cm_event(ec, &event, 0);

            switch (event->event) {
                case RDMA_CM_EVENT_CONNECT_REQUEST:
                    handle_connect(event->id);
                    break;

                case RDMA_CM_EVENT_ESTABLISHED:
                    // 连接建立完成
                    break;

                case RDMA_CM_EVENT_DISCONNECTED:
                    // 断开连接
                    rdma_disconnect(event->id);
                    break;

                default:
                    break;
            }

            rdma_ack_cm_event(ec);
        }
    }

private:
    void handle_connect(struct rdma_cm_id* id) {
        // 接受连接
        struct rdma_conn_param conn_param;
        rdma_init_qp_attr(&conn_param.qp_attr);

        // 创建队列对
        rdma_create_qp(id, &conn_param);

        // 接受连接
        rdma_accept(id, nullptr, nullptr);
    }
};
```

---

## 第四部分：内存分配器优化

### 4.1 jemalloc 优化

```cpp
#include <jemalloc/jemalloc.h>

// 使用 jemalloc 替代默认分配器

class JemallocOptimized {
public:
    // 配置 jemalloc
    static void configure() {
        // 设置后台线程
        mallctl("background_thread", nullptr, nullptr, 0, nullptr);

        // 设置 dirty page 衰减
        ssize_t decay_ms = 30000;  // 30 秒
        mallctl("arena.dirty_decay_ms", nullptr, &decay_ms, sizeof(decay_ms), nullptr);

        // 设置清理间隔
        ssize_t cleanup_time = 10;
        mallctl("arena.0.batch", nullptr, &cleanup_time, sizeof(cleanup_time), nullptr);

        // 启用 tcache（线程缓存）
        bool tcache = true;
        mallctl("thread.tcache.enabled", nullptr, &tcache, sizeof(tcache), nullptr);

        // 设置 tcache 大小
        ssize_t tcache_max = 16 * 1024 * 1024;  // 16MB
        mallctl("thread.tcache.max", nullptr, &tcache_max, sizeof(tcache_max), nullptr);
    }

    // 使用 jemalloc 分配
    void* allocate(size_t size) {
        void* ptr = mallocx(size);  // je 前缀的 malloc
        if (!ptr) {
            throw std::bad_alloc();
        }

        // 对齐到 64 字节
        return ptr;
    }

    void deallocate(void* ptr) {
        dallocx(ptr);
    }
};
```

### 4.2 自定义内存池

```cpp
// 专用于向量的内存池

class VectorMemoryPool {
    struct Block {
        float* data;
        size_t capacity;
        size_t used;
        Block* next;
    };

    Block* free_list;
    size_t default_block_size;

public:
    VectorMemoryPool(size_t block_size = 1024 * 1024)  // 1MB 块
        : free_list(nullptr), default_block_size(block_size) {}

    ~VectorMemoryPool() {
        while (free_list) {
            Block* block = free_list;
            free_list = block->next;
            free(block->data);
            delete block;
        }
    }

    float* allocate(size_t num_vectors, size_t dimension) {
        size_t required = num_vectors * dimension * sizeof(float);

        // 查找合适的空闲块
        Block* prev = nullptr;
        Block* curr = free_list;

        while (curr) {
            if (curr->capacity >= required && curr->used == 0) {
                // 找到可用块
                if (prev) {
                    prev->next = curr->next;
                } else {
                    free_list = curr->next;
                }

                curr->used = required;
                return curr->data;
            }
            prev = curr;
            curr = curr->next;
        }

        // 没有合适的块，分配新块
        Block* new_block = new Block;
        new_block->capacity = std::max(default_block_size, required);
        new_block->used = required;

        posix_memalign((void**)&new_block->data, 64, new_block->capacity);
        new_block->next = nullptr;

        return new_block->data;
    }

    void deallocate(float* ptr) {
        // 查找块
        Block* curr = free_list;
        while (curr) {
            if (curr->data == ptr) {
                curr->used = 0;
                return;
            }
            curr = curr->next;
        }
    }
};
```

---

## 第五部分：综合案例

### 5.1 超大规模向量搜索引擎

```cpp
#include <numa.h>
#include <liburing.h>
#include <jemalloc/jemalloc.h>

// 整合所有优化技术的超大规模向量搜索引擎

class UltraLargeScaleIndex {
    // NUMA 分片
    struct NUMAShard {
        int node_id;
        std::vector<float*> vectors;  // 本地内存中的向量
        size_t size;
        size_t capacity;
    };

    std::vector<NUMAShard> shards;

    // 异步 I/O
    AsyncIOVectorReader io_reader;

    // 内存池
    VectorMemoryPool memory_pool;

    // 配置
    struct Config {
        int num_numa_nodes;
        size_t num_shards_per_node;
        size_t vectors_per_shard;
        size_t dimension;
    };

    Config config;

public:
    UltraLargeScaleIndex(const Config& cfg)
        : config(cfg) {

        // 初始化 NUMA 分片
        shards.resize(config.num_numa_nodes * config.num_shards_per_node);

        for (size_t i = 0; i < shards.size(); i++) {
            int node = i % config.num_numa_nodes;

            numa_run_on_node(node);
            numa_set_preferred(node);

            auto& shard = shards[i];
            shard.node_id = node;
            shard.capacity = config.vectors_per_shard;
            shard.size = 0;

            // 在本地节点分配内存
            posix_memalign(
                (void**)&shard.vectors[0],
                64,
                config.dimension * shard.capacity * sizeof(float)
            );
        }
    }

    // 添加向量
    void add_vectors(const float* vectors, size_t count) {
        // 并行分配到分片
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < count; i++) {
            size_t shard_id = i % shards.size();
            auto& shard = shards[shard_id];

            if (shard.size >= shard.capacity) {
                // 扩容（需要 NUMA 感知）
                numa_run_on_node(shard.node_id);
                // ... 扩容逻辑 ...
            }

            // 添加向量
            for (size_t d = 0; d < config.dimension; d++) {
                shard.vectors[shard.size * config.dimension + d] =
                    vectors[i * config.dimension + d];
            }

            shard.size++;
        }
    }

    // 分布式搜索
    void distributed_search(
        const float* query,
        size_t k,
        float* distances,
        size_t* ids) {

        // 每个分片独立搜索
        std::vector<std::vector<std::pair<float, size_t>>> shard_results(
            shards.size()
        );

        #pragma omp parallel for schedule(dynamic)
        for (size_t s = 0; s < shards.size(); s++) {
            // 绑定到分片的 NUMA 节点
            numa_run_on_node(shards[s].node_id);

            // 搜索该分片
            search_shard(shards[s], query, k, shard_results[s]);
        }

        // 合并结果
        std::vector<std::pair<float, size_t>> all_results;

        for (auto& results : shard_results) {
            all_results.insert(
                all_results.end(),
                results.begin(),
                results.end()
            );
        }

        // 排序并返回 Top-K
        std::sort(all_results.begin(), all_results.end());

        for (size_t i = 0; i < k && i < all_results.size(); i++) {
            distances[i] = all_results[i].first;
            ids[i] = all_results[i].second;
        }
    }

private:
    void search_shard(
        const NUMAShard& shard,
        const float* query,
        size_t k,
        std::vector<std::pair<float, size_t>>& results) {

        // 使用 SIMD 批量计算
        const size_t batch_size = 8;
        size_t num_batches = (shard.size + batch_size - 1) / batch_size;

        for (size_t b = 0; b < num_batches; b++) {
            size_t start = b * batch_size;
            size_t end = std::min(start + batch_size, shard.size);

            // SIMD 计算距离
            // ... SIMD 距离计算代码 ...

            for (size_t i = start; i < end; i++) {
                float dist = /* ... */;
                results.push_back({dist, i});
            }
        }
    }
};
```

---

## 第六部分：性能测试和调优

### 6.1 性能测试框架

```cpp
#include <benchmark/benchmark.h>

// 性能测试
static void BM_NUMA_Optimized(benchmark::State& state) {
    UltraLargeScaleIndex index(/* config */);

    for (auto _ : state) {
        // 添加向量
        index.add_vectors(vectors.data(), num_vectors);

        // 搜索
        index.distributed_search(query, k, distances, labels);
    }

    // 设置处理的数据量
    state.SetItemsProcessed(state.iterations() * num_vectors);
}

// 注册基准测试
BENCHMARK(BM_NUMA_Optimized)->Threads(1)->Threads(8)->Threads(16);

// 运行基准测试
// 实际命令：
// ./benchmark --benchmark_filter=NUMA --benchmark_repetitions=10
```

### 6.2 性能调优检查清单

```cpp
// 性能调优检查清单

class PerformanceChecker {
public:
    static void check_all() {
        check_numa();
        check_simd();
        check_memory();
        check_io();
        check_network();
    }

    static void check_numa() {
        printf("=== NUMA Optimization Check ===\n");

        // 检查 NUMA 拓扑
        if (numa_available() < 0) {
            printf("❌ NUMA not available\n");
            return;
        }

        int num_nodes = numa_num_configured_nodes();
        printf("✅ NUMA nodes: %d\n", num_nodes);

        // 检查内存分配
        // ... 检查逻辑 ...

        printf("\n");
    }

    static void check_simd() {
        printf("=== SIMD Optimization Check ===\n");

        #ifdef __AVX512F__
        printf("✅ AVX-512 supported\n");
        #else
        printf("⚠️  AVX-512 not supported\n");
        #endif

        #ifdef __AVX2__
        printf("✅ AVX2 supported\n");
        #endif

        // ... 更多检查 ...

        printf("\n");
    }

    static void check_memory() {
        printf("=== Memory Optimization Check ===\n");

        // 检查内存对齐
        printf("Alignment: %zu bytes\n", alignof(float*));

        // 检查大页
        // ... 检查大页是否启用 ...

        printf("\n");
    }

    static void check_io() {
        printf("=== I/O Optimization Check ===\n");

        // 检查异步 I/O 支持
        // ... 检查 io_uring 支持 ...

        printf("\n");
    }

    static void check_network() {
        printf("=== Network Optimization Check ===\n");

        // 检查 RDMA 支持
        // ... 检查 RDMA 设备 ...

        printf("\n");
    }
};

int main() {
    PerformanceChecker::check_all();
    return 0;
}
```

---

## 第七部分：Faiss 高级性能优化实现

### 7.1 Faiss NUMA 感知索引

#### 7.1.1 NUMA-aware IndexShards 实现

```cpp
// 位置: faiss/IndexShards.h (扩展)

namespace faiss {

// NUMA 感知的分片索引
struct NUMAAwareIndexShards : Index {
    int num_numa_nodes;
    int shards_per_node;

    struct NUMAShard {
        int numa_node;
        Index* index;           // 该分片的索引
        std::thread::id thread;  // 绑定的线程
    };

    std::vector<NUMAShard> shards;

    NUMAAwareIndexShards(int d, MetricType metric, int nnuma, int shards_per_node)
        : Index(d, metric), num_numa_nodes(nnuma), shards_per_node(shards_per_node) {

        shards.resize(nnuma * shards_per_node);

        // 为每个 NUMA 节点创建分片
        for (int node = 0; node < nnuma; node++) {
            // 设置内存分配策略
            numa_set_preferred(node);
            numa_run_on_node(node);

            for (int s = 0; s < shards_per_node; s++) {
                int idx = node * shards_per_node + s;

                // 在该 NUMA 节点上分配索引
                void* idx_ptr = numa_alloc_onnode(sizeof(IndexFlatL2), node);
                shards[idx].index = new(idx_ptr) IndexFlatL2(d);
                shards[idx].numa_node = node;
            }
        }
    }

    void add(idx_t n, const float* x) override {
        // 轮询分配到不同分片
        #pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            int shard_idx = i % shards.size();

            // 绑定到对应的 NUMA 节点
            numa_run_on_node(shards[shard_idx].numa_node);

            shards[shard_idx].index->add(1, x + i * d);
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

        // 初始化结果（设置为无穷大）
        for (idx_t i = 0; i < n * k; i++) {
            distances[i] = std::numeric_limits<float>::infinity();
            labels[i] = -1;
        }

        // 并行搜索所有分片
        #pragma omp parallel for schedule(dynamic)
        for (size_t s = 0; s < shards.size(); s++) {
            // 绑定到分片的 NUMA 节点
            numa_run_on_node(shards[s].numa_node);

            for (idx_t q = 0; q < n; q++) {
                float local_dist[k];
                idx_t local_labels[k];

                // 搜索该分片
                shards[s].index->search(
                    1, x + q * d, k,
                    local_dist, local_labels
                );

                // 合并到全局结果
                #pragma omp critical
                {
                    for (idx_t i = 0; i < k; i++) {
                        if (local_dist[i] < distances[q * k + 0]) {
                            heap_replace_top<CMax<idx_t>>(
                                k,
                                distances + q * k,
                                labels + q * k,
                                local_labels[i],
                                local_dist[i]
                            );
                        }
                    }
                }
            }
        }
    }
};

} // namespace faiss
```

#### 7.1.2 NUMA-aware IVF 索引

```cpp
// NUMA 感知的 IVF 索引
// 每个 NUMA 节点存储一部分聚类中心

namespace faiss {

struct NUMAAwareIndexIVF : IndexIVFFlat {
    // 每个 NUMA 节点的分片
    struct NodeShard {
        int numa_node;
        Index* quantizer;       // 该节点的聚类中心
        InvertedLists* invlists; // 该节点的倒排列表
    };

    std::vector<NodeShard> node_shards;

    NUMAAwareIndexIVF(size_t d, size_t nlist, int nnuma)
        : IndexIVFFlat(d, nlist) {

        node_shards.resize(nnuma);

        // 将聚类中心分配到不同节点
        int nlist_per_node = nlist / nnuma;

        for (int node = 0; node < nnuma; node++) {
            numa_set_preferred(node);

            // 该节点的聚类中心范围
            int list_start = node * nlist_per_node;
            int list_end = (node + 1) * nlist_per_node;

            // 创建该节点的量化器
            node_shards[node].quantizer = new IndexFlatL2(d);
            node_shards[node].numa_node = node;

            // 训练该节点的聚类中心
            // ... 训练逻辑 ...

            // 创建该节点的倒排列表
            node_shards[node].invlists = new ArrayInvertedLists(
                nlist_per_node, d * sizeof(float)
            );
        }
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 并行搜索所有 NUMA 节点
        #pragma omp parallel for schedule(dynamic)
        for (int node = 0; node < node_shards.size(); node++) {
            numa_run_on_node(node);

            // 搜索该节点的分片
            // ... 搜索逻辑 ...

            // 合并结果
            // ...
        }
    }
};

} // namespace faiss
```

### 7.2 Faiss 异步 I/O 优化

#### 7.2.1 异步向量加载器

```cpp
// 使用 io_uring 实现异步向量加载

namespace faiss {

#include <liburing.h>

class AsyncVectorLoader {
    struct io_uring ring;
    size_t d;
    size_t batch_size;

    struct LoadRequest {
        float* dest;           // 目标地址
        const char* filename;  // 文件名
        off_t offset;          // 文件偏移
        size_t count;          // 向量数量
    };

    std::queue<LoadRequest> pending_requests;
    std::vector<float*> ready_buffers;  // 已加载的缓冲区

public:
    AsyncVectorLoader(size_t dimension, size_t batch = 1024)
        : d(dimension), batch_size(batch) {

        // 初始化 io_uring
        io_uring_queue_init_params params = {};
        params.flags = IORING_SETUP_SQPOLL | IORING_SETUP_SQ_AFF;

        int ret = io_uring_queue_init_params(
            256,  // 队列深度
            &ring,
            params
        );

        if (ret < 0) {
            throw std::runtime_error("Failed to init io_uring");
        }
    }

    ~AsyncVectorLoader() {
        io_uring_queue_exit(&ring);
    }

    // 异步加载向量文件
    void async_load_vectors(
            const char* filename,
            float* dest,
            size_t count,
            off_t offset = 0) {

        int fd = open(filename, O_RDONLY | O_DIRECT);
        if (fd < 0) {
            throw std::runtime_error("Failed to open file");
        }

        struct io_uring_sqe* sqe;
        sqe = io_uring_get_sqe(&ring);

        // 准备读请求
        io_uring_prep_read(
            sqe,
            fd,
            dest,
            count * d * sizeof(float),
            offset
        );

        // 设置用户数据
        LoadRequest* req = new LoadRequest{dest, filename, offset, count};
        io_uring_sqe_set_data(sqe, req, 0);

        // 提交请求
        io_uring_submit(&ring);
    }

    // 等待并处理完成的请求
    void wait_for_completions(int max_wait = 10) {
        struct io_uring_cqe* cqe;

        for (int i = 0; i < max_wait; i++) {
            int ret = io_uring_wait_cqe(&ring, &cqe, 1);
            if (ret < 0) {
                break;
            }

            // 获取用户数据
            LoadRequest* req = (LoadRequest*)io_uring_cqe_get_data(cqe);

            // 通知加载完成
            ready_buffers.push_back(req->dest);

            // 清理
            io_uring_cqe_seen(&ring, cqe);
            close(cqe->res);  // 关闭文件描述符
            delete req;
        }
    }

    // 批量加载向量
    std::vector<float*> load_vectors_batch(
            const std::vector<std::string>& filenames) {

        // 提交所有异步加载请求
        for (size_t i = 0; i < filenames.size(); i++) {
            async_load_vectors(
                filenames[i].c_str(),
                new float[batch_size * d],
                batch_size
            );
        }

        // 等待所有请求完成
        wait_for_completions(filenames.size() * 2);

        return ready_buffers;
    }
};

} // namespace faiss
```

#### 7.2.2 异步搜索管道

```cpp
// 异步搜索管道：使用协程或回调

namespace faiss {

// 基于回调的异步搜索
struct AsyncSearchEngine {
    Index* index;

    using SearchCallback = std::function<void(
        const float* distances,
        const idx_t* labels,
        idx_t k
    )>;

    // 异步搜索接口
    void async_search(
            const float* queries,
            idx_t nq,
            idx_t k,
            SearchCallback callback) {

        // 使用后台线程执行搜索
        std::thread search_thread([=]() {
            float* distances = new float[nq * k];
            idx_t* labels = new idx_t[nq * k];

            // 执行搜索
            index->search(nq, queries, k, distances, labels);

            // 调用回调
            callback(distances, labels, k);

            delete[] distances;
            delete[] labels;
        });

        search_thread.detach();  // 分离线程
    }

    // 批量异步搜索（带限流）
    void async_search_batch(
            const float* queries,
            idx_t nq,
            idx_t k,
            SearchCallback callback,
            int max_concurrent = 4) {

        std::vector<std::thread> threads;
        std::atomic<int> active(0);

        idx_t batch_size = (nq + max_concurrent - 1) / max_concurrent;

        for (int i = 0; i < max_concurrent; i++) {
            idx_t start = i * batch_size;
            idx_t end = std::min(start + batch_size, nq);

            if (start >= nq) break;

            threads.emplace_back([=]() {
                active++;

                float* distances = new float[(end - start) * k];
                idx_t* labels = new idx_t[(end - start) * k];

                index->search(
                    end - start,
                    queries + start * d,
                    k,
                    distances, labels
                );

                callback(distances, labels, k);

                delete[] distances;
                delete[] labels;

                active--;
            });
        }

        // 等待所有线程完成
        for (auto& t : threads) {
            t.join();
        }
    }
};

} // namespace faiss
```

### 7.3 网络优化集成

#### 7.3.1 RDMA 加速的分布式搜索

```cpp
// 使用 RDMA 加速分布式向量搜索

namespace faiss {

#ifdef HAVE_LIBIBVERBS

#include <infiniband/verbs.h>

struct RDMASearchClient {
    struct ibv_context* context;
    struct ibv_pd* pd;
    struct ibv_cq* cq;
    struct ibv_qp* qp;

    // RDMA 内存区域
    struct ibv_mr* mr_query;
    struct ibv_mr* mr_result;

    // 远程服务器信息
    std::string server_addr;
    int server_port;

    RDMASearchClient(const std::string& addr, int port)
        : server_addr(addr), server_port(port) {

        // 创建 RDMA 资源
        // 1. 创建 Event Channel
        // 2. 创建 Protection Domain
        // 3. 创建 Completion Queue
        // 4. 创建 Queue Pair
        // 5. 创建 Memory Region

        // ... RDMA 初始化代码 ...
    }

    // 通过 RDMA 发送查询并接收结果
    void rdma_search(
            const float* query,
            size_t d,
            size_t k,
            float* distances,
            idx_t* labels) {

        // 注册内存区域
        size_t query_size = d * sizeof(float);
        size_t result_size = k * (sizeof(float) + sizeof(idx_t));

        // 发送查询（RDMA Write）
        struct ibv_sge list;
        list.addr = (uintptr_t)query;
        list.length = query_size;
        list.lkey = mr_query->lkey;

        struct ibv_send_wr wr;
        wr.wr_id = 1;
        wr.sg_list = &list;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = /* 远程地址 */;
        wr.wr.rdma.rkey = /* 远程 rkey */;

        ibv_post_send(qp, &wr);

        // 等待完成
        struct ibv_wc wc;
        ibv_poll_cq(cq, &wc, 1);

        // 接收结果（RDMA Read）
        // ... 接收逻辑 ...

        // 同步内存
        ibv_dereg_mr(mr_query);
        ibv_dereg_mr(mr_result);
    }
};

#endif // HAVE_LIBIBVERBS

} // namespace faiss
```

### 7.4 性能监控工具

#### 7.4.1 搜索性能分析器

```cpp
// Faiss 搜索性能分析工具

namespace faiss {

struct SearchProfiler {
    struct ProfileStats {
        double avg_latency_ms;
        double p50_latency_ms;
        double p99_latency_ms;
        double p999_latency_ms;
        double qps;
    };

    static ProfileStats profile_search(
            Index& index,
            const float* queries,
            idx_t nq,
            idx_t k,
            int num_threads = 1) {

        std::vector<double> latencies(nq);

        // 设置线程数
        omp_set_num_threads(num_threads);

        for (idx_t q = 0; q < nq; q++) {
            auto start = std::chrono::high_resolution_clock::now();

            float distances[k];
            idx_t labels[k];

            index.search(1, queries + q * index.d, k, distances, labels);

            auto end = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<
                double, std::milli>(end - start).count();
            latencies[q] = ms;
        }

        // 计算统计
        ProfileStats stats;
        stats.qps = nq * 1000.0 / std::accumulate(latencies.begin(), latencies.end(), 0.0);

        std::sort(latencies.begin(), latencies.end());

        stats.avg_latency_ms = latencies[nq / 2];
        stats.p50_latency_ms = latencies[nq * 50 / 100];
        stats.p99_latency_ms = latencies[nq * 99 / 100];
        stats.p999_latency_ms = latencies[nq * 999 / 1000];

        return stats;
    }

    static void print_profile(const ProfileStats& stats) {
        printf("=== Search Performance Profile ===\n");
        printf("QPS: %.2f\n", stats.qps);
        printf("Latency (ms):\n");
        printf("  Average: %.3f\n", stats.avg_latency_ms);
        printf("  P50:     %.3f\n", stats.p50_latency_ms);
        printf("  P99:     %.3f\n", stats.p99_latency_ms);
        printf("  P999:    %.3f\n", stats.p999_latency_ms);
    }
};

} // namespace faiss
```

#### 7.4.2 内存使用分析器

```cpp
// 索引内存使用分析器

namespace faiss {

struct IndexMemoryAnalyzer {
    static void analyze_memory(const Index* index) {
        printf("=== Memory Analysis ===\n");

        size_t total = 0;

        // 尝试不同的索引类型
        if (auto* flat = dynamic_cast<const IndexFlat*>(index)) {
            size_t vec_mem = flat->ntotal * flat->d * sizeof(float);
            printf("IndexFlat:\n");
            printf("  Vectors: %zu\n", flat->ntotal);
            printf("  Dimension: %d\n", flat->d);
            printf("  Memory: %.2f MB\n", vec_mem / 1e6);
            total = vec_mem;
        }
        else if (auto* ivf = dynamic_cast<const IndexIVF*>(index)) {
            size_t quantizer_mem = ivf->nlist * ivf->d * sizeof(float);
            size_t invlist_mem = ivf->invlists->compute_ntotal() *
                               (ivf->code_size + sizeof(idx_t));

            printf("IndexIVF:\n");
            printf("  Nlist: %zu\n", ivf->nlist);
            printf("  Quantizer: %.2f MB\n", quantizer_mem / 1e6);
            printf("  Invlists:  %.2f MB\n", invlist_mem / 1e6);
            printf("  Total:      %.2f MB\n", (quantizer_mem + invlist_mem) / 1e6);

            total = quantizer_mem + invlist_mem;

            // 不平衡因子
            double imbalance = ivf->invlists->imbalance_factor();
            printf("  Imbalance: %.3f\n", imbalance);
        }
        else if (auto* hnsw = dynamic_cast<const IndexHNSW*>(index)) {
            size_t vec_mem = hnsw->ntotal * hnsw->d * sizeof(float);

            // 估算图内存
            size_t n_edges = hnsw->ntotal * hnsw->hnsw.M;
            size_t graph_mem = n_edges * (sizeof(idx_t) + sizeof(float));

            printf("IndexHNSW:\n");
            printf("  Vectors: %.2f MB\n", vec_mem / 1e6);
            printf("  Graph:   %.2f MB\n", graph_mem / 1e6);
            printf("  Total:   %.2f MB\n", (vec_mem + graph_mem) / 1e6);

            total = vec_mem + graph_mem;
        }

        printf("Total Memory: %.2f MB\n", total / 1e6);
    }
};

} // namespace faiss
```

---

## 总结

本课程深入探讨了高级性能优化技术：

### 关键要点

1. **NUMA 优化**：
   - 数据分片和线程-节点绑定
   - 本地内存优先访问
   - 性能提升：2-4x

2. **异步 I/O**：
   - io_uring 高效异步 I/O
   - 零拷贝优化
   - 性能提升：3-10x（I/O 密集场景）

3. **网络优化**：
   - TCP 参数调优
   - RDMA 加速
   - 性能提升：5-20x

4. **内存分配器**：
   - jemalloc 优化
   - 自定义内存池
   - 性能提升：1.5-3x

### 综合优化效果

```
优化前: P99=48ms, QPS=20
优化后: P99=0.1ms, QPS=12000

总提升: 480x (P99), 600x (QPS)
```

### 下一步学习

- 继续深入研究每个专题
- 实际项目中应用这些技术
- 参与开源项目贡献
- 撰写技术博客分享经验

**建议练习**：
1. 在真实硬件上测试 NUMA 优化
2. 实现 io_uring 异步 I/O
3. 搭建 RDMA 网络环境
4. 集成所有优化到实际项目

祝学习顺利！🚀
