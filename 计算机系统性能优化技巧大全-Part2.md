# 计算机系统性能优化技巧大全 - Part 2

## 续前文...

## 6. 磁盘和存储优化

### 6.1 SSD优化技巧

#### 6.1.1 对齐和擦除块优化

```cpp
// 案例来源: RocksDB, LevelDB

// SSD特性:
// • 页面大小: 4KB-16KB
// • 擦除块大小: 256KB-4MB
// • 读快写慢擦除更慢

// 优化1: 对齐写入
class AlignedWriter {
private:
    int fd;
    size_t alignment;  // 通常4KB
    char* buffer;
    size_t buffer_size;
    size_t buffer_pos;

public:
    AlignedWriter(const char* filename, size_t align = 4096)
        : alignment(align), buffer_pos(0) {

        fd = open(filename, O_WRONLY | O_CREAT | O_DIRECT, 0644);

        buffer_size = 1024 * 1024;  // 1MB缓冲
        posix_memalign((void**)&buffer, alignment, buffer_size);
    }

    void write(const char* data, size_t size) {
        size_t remaining = size;
        const char* ptr = data;

        while (remaining > 0) {
            size_t space = buffer_size - buffer_pos;
            size_t to_copy = std::min(remaining, space);

            memcpy(buffer + buffer_pos, ptr, to_copy);
            buffer_pos += to_copy;
            ptr += to_copy;
            remaining -= to_copy;

            if (buffer_pos == buffer_size) {
                flush();
            }
        }
    }

    void flush() {
        if (buffer_pos == 0) return;

        // 对齐到alignment
        size_t aligned_size = (buffer_pos + alignment - 1)
                              & ~(alignment - 1);

        // 填充
        if (aligned_size > buffer_pos) {
            memset(buffer + buffer_pos, 0, aligned_size - buffer_pos);
        }

        ::write(fd, buffer, aligned_size);
        buffer_pos = 0;
    }

    ~AlignedWriter() {
        flush();
        close(fd);
        free(buffer);
    }
};

// 优化2: 批量写入
// SSD内部并行 (多通道、多die)
void parallel_write_ssd(const char* filename, char** data_blocks,
                        int num_blocks, size_t block_size) {
    // 使用异步IO批量提交
    struct io_uring ring;
    io_uring_queue_init(256, &ring, 0);

    int fd = open(filename, O_WRONLY | O_CREAT | O_DIRECT, 0644);

    for (int i = 0; i < num_blocks; i++) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        io_uring_prep_write(sqe, fd, data_blocks[i],
                           block_size, i * block_size);
    }

    io_uring_submit(&ring);

    // 等待完成
    for (int i = 0; i < num_blocks; i++) {
        struct io_uring_cqe* cqe;
        io_uring_wait_cqe(&ring, &cqe);
        io_uring_cqe_seen(&ring, cqe);
    }

    close(fd);
    io_uring_queue_exit(&ring);
}

性能对比 (写入10GB数据):
方法                写入速度    写放大
────────────────────────────────────
小随机写(4KB)       120 MB/s    4.5x
对齐写(4KB)         450 MB/s    1.2x
大批量写(1MB)       1.8 GB/s    1.05x
```

#### 6.1.2 TRIM和垃圾回收优化

```cpp
// 案例来源: 文件系统、数据库

#include <linux/fs.h>
#include <sys/ioctl.h>

// 主动TRIM (释放不用的块)
void trim_file_range(int fd, off_t offset, off_t len) {
    uint64_t range[2] = {(uint64_t)offset, (uint64_t)len};

    // BLKDISCARD: 通知SSD这些块不再使用
    if (ioctl(fd, BLKDISCARD, &range) < 0) {
        perror("BLKDISCARD failed");
    }
}

// 文件系统级TRIM
void fstrim_filesystem(const char* mount_point) {
    // fstrim命令的实现
    struct fstrim_range range;
    range.start = 0;
    range.len = ULLONG_MAX;
    range.minlen = 0;

    int fd = open(mount_point, O_RDONLY);
    ioctl(fd, FITRIM, &range);
    close(fd);
}

// Copy-on-Write + TRIM优化 (类似btrfs)
class COWFile {
private:
    int fd;
    std::map<off_t, off_t> block_map;  // 逻辑→物理映射

public:
    void write_block(off_t logical_offset, const char* data,
                     size_t size) {
        // 分配新物理块
        off_t physical_offset = allocate_new_block(size);

        // 写入数据
        pwrite(fd, data, size, physical_offset);

        // 查找旧物理块
        auto it = block_map.find(logical_offset);
        if (it != block_map.end()) {
            // TRIM旧块
            trim_file_range(fd, it->second, size);
        }

        // 更新映射
        block_map[logical_offset] = physical_offset;
    }
};

最佳实践:
• 使用文件系统的discard选项 (ext4, xfs)
• 定期运行fstrim
• 应用层及时释放不用的数据
• 避免频繁小写入
```

### 6.2 RAID优化

#### 6.2.1 RAID条带化优化

```cpp
// 案例来源: Ceph, GlusterFS

// RAID 0: 条带化
// 优化: chunk size选择

void benchmark_raid_chunk_size() {
    // 测试不同chunk size
    int chunk_sizes[] = {4, 8, 16, 32, 64, 128, 256, 512};  // KB

    for (int chunk_kb : chunk_sizes) {
        // 配置RAID
        set_raid_chunk_size(chunk_kb * 1024);

        // 测试顺序写
        auto seq_write = benchmark_sequential_write();

        // 测试随机写
        auto rand_write = benchmark_random_write();

        printf("Chunk %dKB: SeqWrite=%d MB/s, RandWrite=%d MB/s\n",
               chunk_kb, seq_write, rand_write);
    }
}

/*
典型结果 (RAID0, 4个SSD):

Chunk Size   Sequential Write   Random Write
────────────────────────────────────────────
4KB          850 MB/s           180 MB/s
16KB         1.2 GB/s           220 MB/s
64KB         2.8 GB/s           280 MB/s  ← 最佳
256KB        2.9 GB/s           260 MB/s
512KB        2.7 GB/s           210 MB/s

选择原则:
• 顺序访问: 大chunk (128-512KB)
• 随机访问: 小chunk (16-64KB)
• 混合负载: 64KB (平衡点)
*/

// RAID 5/6: 部分条带写优化
class RAID5Optimizer {
public:
    // 问题: 部分条带写需要Read-Modify-Write
    // 解决: 批量写入完整条带

    void write_full_stripe(int raid_fd, const char* data,
                          size_t size, int num_disks) {
        size_t stripe_size = CHUNK_SIZE * (num_disks - 1);

        // 确保写入对齐到完整条带
        size_t aligned_size = (size + stripe_size - 1)
                              / stripe_size * stripe_size;

        char* aligned_buffer = aligned_alloc(4096, aligned_size);
        memcpy(aligned_buffer, data, size);

        // 填充到完整条带
        if (aligned_size > size) {
            memset(aligned_buffer + size, 0, aligned_size - size);
        }

        // 写入
        write(raid_fd, aligned_buffer, aligned_size);
        free(aligned_buffer);
    }
};

性能对比 (RAID5, 4盘):
• 随机小写 (4KB):    45 MB/s   (RMW开销)
• 对齐条带写:         320 MB/s  (7x faster)
```

### 6.3 文件系统调优

#### 6.3.1 ext4优化

```bash
# 案例来源: 数据库服务器、高性能存储

# 挂载选项优化
mount -o noatime,nodiratime,data=writeback,barrier=0 \
      /dev/sda1 /mnt/data

# 参数说明:
# noatime: 不更新访问时间 (减少写入)
# nodiratime: 目录不更新访问时间
# data=writeback: 数据不保证在元数据之前写入 (快但不安全)
# data=ordered: 默认，安全但慢
# data=journal: 最安全但最慢
# barrier=0: 禁用写屏障 (需要电池保护的RAID卡)

# 文件系统创建优化
mkfs.ext4 -b 4096 \              # 块大小4KB
          -E stride=16,stripe-width=64 \  # RAID参数
          -T largefile4 \        # 大文件优化
          -O ^has_journal \      # 禁用日志 (极端性能)
          /dev/sda1

# 运行时调优
# 增加commit间隔 (减少元数据更新)
echo 30 > /proc/sys/vm/dirty_expire_centisecs
echo 60 > /proc/sys/vm/dirty_writeback_centisecs

# 调整预读
blockdev --setra 8192 /dev/sda1  # 4MB预读

性能提升案例 (MySQL InnoDB):
配置                    IOPS     延迟
───────────────────────────────────────
默认挂载                8.5k     12ms
noatime + ordered       12k      8ms
writeback + nobarrier   18k      5ms  (需要UPS!)
```

#### 6.3.2 XFS高性能配置

```bash
# 案例来源: Red Hat推荐、大文件服务器

# XFS挂载选项
mount -o noatime,nodiratime,logbufs=8,logbsize=256k,\
      largeio,inode64,swalloc \
      /dev/sda1 /mnt/data

# logbufs=8: 增加日志缓冲数
# logbsize=256k: 日志缓冲大小
# largeio: 报告大IO能力
# inode64: 允许64位inode (大文件系统)
# swalloc: 条带化感知分配器

# 创建XFS
mkfs.xfs -f -d agcount=64 \      # 分配组数量
         -l size=128m,lazy-count=1 \  # 日志大小
         -i size=512 \           # inode大小
         /dev/sda1

# XFS实时子卷 (real-time subvolume)
# 用于流媒体、视频等顺序IO
mkfs.xfs -f -d agcount=4 \
         -r extsize=1m \         # 实时extent大小
         /dev/sda1

性能特点:
• 大文件性能优异
• 并行IO性能强
• 延迟分配 (allocate-on-flush)
• 在线碎片整理
```

---

## 7. NUMA系统优化

### 7.1 NUMA拓扑感知

#### 7.1.1 查看和分析NUMA拓扑

```bash
# 案例来源: HPC系统、多路服务器

# 查看NUMA拓扑
numactl --hardware

# 输出示例 (2路服务器):
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23
node 0 size: 64GB
node 0 free: 48GB
node 1 cpus: 8 9 10 11 12 13 14 15 24 25 26 27 28 29 30 31
node 1 size: 64GB
node 1 free: 52GB
node distances:
node   0   1
  0:  10  21    # 本地访问: 10, 跨节点: 21
  1:  21  10

# 查看进程NUMA状态
numastat -p <PID>

# 监控NUMA统计
numastat -c <PID>

# 查看CPU缓存拓扑
lstopo  # 需要hwloc包

# 测量NUMA延迟
numactl --hardware | grep distance
```

```cpp
// C++ NUMA拓扑API
#include <numa.h>
#include <numaif.h>

class NUMATopology {
public:
    static void print_topology() {
        if (numa_available() < 0) {
            printf("NUMA not available\n");
            return;
        }

        int num_nodes = numa_num_configured_nodes();
        int num_cpus = numa_num_configured_cpus();

        printf("NUMA Nodes: %d\n", num_nodes);
        printf("CPUs: %d\n", num_cpus);

        // 每个节点的信息
        for (int node = 0; node < num_nodes; node++) {
            long long size = numa_node_size64(node, nullptr);
            struct bitmask* cpus = numa_allocate_cpumask();

            numa_node_to_cpus(node, cpus);

            printf("\nNode %d:\n", node);
            printf("  Memory: %lld MB\n", size / (1024*1024));
            printf("  CPUs: ");

            for (int cpu = 0; cpu < num_cpus; cpu++) {
                if (numa_bitmask_isbitset(cpus, cpu)) {
                    printf("%d ", cpu);
                }
            }
            printf("\n");

            numa_free_cpumask(cpus);
        }

        // 距离矩阵
        printf("\nDistance Matrix:\n");
        printf("     ");
        for (int i = 0; i < num_nodes; i++) {
            printf(" %3d", i);
        }
        printf("\n");

        for (int i = 0; i < num_nodes; i++) {
            printf("%3d: ", i);
            for (int j = 0; j < num_nodes; j++) {
                printf(" %3d", numa_distance(i, j));
            }
            printf("\n");
        }
    }

    // 查找最近的NUMA节点
    static int nearest_node(int cpu) {
        return numa_node_of_cpu(cpu);
    }

    // 获取当前线程的NUMA节点
    static int current_node() {
        return numa_preferred();
    }
};
```

### 7.2 NUMA内存分配策略

#### 7.2.1 本地优先分配

```cpp
// 案例来源: Redis, MongoDB

#include <numa.h>

class NUMAAllocator {
public:
    // 策略1: 本地节点分配
    static void* alloc_local(size_t size) {
        int node = numa_node_of_cpu(sched_getcpu());
        return numa_alloc_onnode(size, node);
    }

    // 策略2: 交错分配 (interleave)
    static void* alloc_interleaved(size_t size) {
        // 在所有节点间轮转分配
        return numa_alloc_interleaved(size);
    }

    // 策略3: 优选节点 (preferred)
    static void* alloc_preferred(size_t size, int node) {
        // 优先在指定节点，满了自动溢出
        numa_set_preferred(node);
        void* ptr = numa_alloc(size);
        numa_set_preferred(-1);  // 重置
        return ptr;
    }

    // 策略4: 绑定节点 (bind)
    static void* alloc_bind(size_t size, int node) {
        // 强制在指定节点，失败返回NULL
        return numa_alloc_onnode(size, node);
    }
};

// 应用场景

// 场景1: 线程本地数据
void worker_thread() {
    // 分配本线程的工作内存
    size_t buffer_size = 100 * 1024 * 1024;
    void* buffer = NUMAAllocator::alloc_local(buffer_size);

    // 处理数据...

    numa_free(buffer, buffer_size);
}

// 场景2: 共享只读数据
void* global_readonly_data;

void init_shared_data(size_t size) {
    // 交错分配，所有节点平均访问
    global_readonly_data =
        NUMAAllocator::alloc_interleaved(size);
}

// 场景3: 分区数据结构
struct PartitionedHashTable {
    int num_partitions;
    HashTable** partitions;  // 每个分区在不同节点

    PartitionedHashTable(int n) : num_partitions(n) {
        partitions = new HashTable*[n];

        int num_nodes = numa_num_configured_nodes();

        for (int i = 0; i < n; i++) {
            int node = i % num_nodes;

            // 分区i分配在节点node
            void* mem = NUMAAllocator::alloc_bind(
                sizeof(HashTable), node
            );

            partitions[i] = new (mem) HashTable();
        }
    }

    void insert(Key key, Value val) {
        int partition = hash(key) % num_partitions;
        partitions[partition]->insert(key, val);
    }
};

性能对比 (2 NUMA节点):
分配策略        本地访问延迟   跨节点访问延迟   总体吞吐
──────────────────────────────────────────────────
默认(first-touch)  60ns         140ns         100%
本地分配           60ns         140ns         120%
交错分配           100ns        100ns         85%
绑定分配           60ns         N/A           130%
```

#### 7.2.2 First-Touch策略

```cpp
// 案例来源: OpenMP, Linux内核

// Linux默认: 页面在首次访问时分配到访问线程的节点

// 错误示例: 主线程初始化，所有内存在node 0
void bad_initialization() {
    const size_t SIZE = 1024 * 1024 * 1024;  // 1GB
    int* data = new int[SIZE / sizeof(int)];

    // 主线程初始化 (假设在node 0)
    for (size_t i = 0; i < SIZE / sizeof(int); i++) {
        data[i] = 0;  // 所有页面分配到node 0
    }

    // 工作线程访问 (分布在多个节点)
    #pragma omp parallel for
    for (size_t i = 0; i < SIZE / sizeof(int); i++) {
        data[i] += compute(i);  // 跨节点访问!
    }
}

// 正确示例: 并行初始化
void good_initialization() {
    const size_t SIZE = 1024 * 1024 * 1024;
    int* data = new int[SIZE / sizeof(int)];

    // 并行初始化，每个线程触及自己的数据
    #pragma omp parallel for
    for (size_t i = 0; i < SIZE / sizeof(int); i++) {
        data[i] = 0;  // 页面分配到访问线程的节点
    }

    // 工作线程访问本地数据
    #pragma omp parallel for
    for (size_t i = 0; i < SIZE / sizeof(int); i++) {
        data[i] += compute(i);  // 本地访问!
    }
}

性能对比 (4 NUMA节点, 32线程):
初始化方式       带宽        延迟
────────────────────────────────
串行初始化       45 GB/s    185ns
并行first-touch  156 GB/s   62ns  (3.5x faster)
```

### 7.3 NUMA线程亲和性

#### 7.3.1 绑定线程和内存

```cpp
// 案例来源: DPDK, SPDK, VPP

#include <sched.h>
#include <pthread.h>
#include <numa.h>

class NUMAThreadAffinity {
public:
    // 绑定线程到指定CPU
    static void bind_to_cpu(int cpu) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu, &cpuset);

        pthread_t thread = pthread_self();
        pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
    }

    // 绑定线程到NUMA节点
    static void bind_to_node(int node) {
        struct bitmask* cpumask = numa_allocate_cpumask();
        numa_node_to_cpus(node, cpumask);

        numa_run_on_node_mask(cpumask);
        numa_set_preferred(node);
        numa_set_membind(cpumask);

        numa_free_cpumask(cpumask);
    }

    // 绑定到node，并排除超线程
    static void bind_to_node_physical(int node) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        // 获取node的CPU列表
        struct bitmask* cpumask = numa_allocate_cpumask();
        numa_node_to_cpus(node, cpumask);

        int num_cpus = numa_num_configured_cpus();

        // 只选择物理核心 (假设偶数core ID)
        for (int cpu = 0; cpu < num_cpus; cpu++) {
            if (numa_bitmask_isbitset(cpumask, cpu) &&
                cpu % 2 == 0) {  // 物理核心
                CPU_SET(cpu, &cpuset);
            }
        }

        pthread_setaffinity_np(pthread_self(),
                              sizeof(cpuset), &cpuset);

        numa_free_cpumask(cpumask);
    }
};

// 工作线程模式
void* worker_thread(void* arg) {
    int thread_id = *(int*)arg;
    int num_nodes = numa_num_configured_nodes();
    int node = thread_id % num_nodes;

    // 绑定线程和内存到同一节点
    NUMAThreadAffinity::bind_to_node(node);

    // 分配本地内存
    size_t work_size = 100 * 1024 * 1024;
    void* work_buffer = numa_alloc_onnode(work_size, node);

    // 处理数据 (全部本地访问)
    process_data(work_buffer, work_size);

    numa_free(work_buffer, work_size);
    return nullptr;
}

// 启动优化
void optimized_startup(int num_threads) {
    pthread_t* threads = new pthread_t[num_threads];
    int* thread_ids = new int[num_threads];

    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], nullptr,
                      worker_thread, &thread_ids[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    delete[] threads;
    delete[] thread_ids;
}

性能提升 (2节点, 16核每节点):
配置                     吞吐量      延迟
──────────────────────────────────────────
无绑定 (OS调度)          850 Mops   120ns
绑定到node               1320 Mops   76ns  (1.55x)
绑定+禁用超线程          1450 Mops   69ns  (1.71x)
```

---

## 8. 网络性能优化

### 8.1 零拷贝技术

#### 8.1.1 sendfile系统调用

```cpp
// 案例来源: Nginx, Apache

#include <sys/sendfile.h>

// 传统方式: 4次数据拷贝
ssize_t send_file_traditional(int sock_fd, const char* filename) {
    int file_fd = open(filename, O_RDONLY);
    struct stat stat_buf;
    fstat(file_fd, &stat_buf);
    size_t file_size = stat_buf.st_size;

    char* buffer = new char[file_size];

    // 拷贝1: 磁盘 -> 内核缓冲
    // 拷贝2: 内核缓冲 -> 用户空间
    read(file_fd, buffer, file_size);

    // 拷贝3: 用户空间 -> 内核socket缓冲
    // 拷贝4: 内核socket缓冲 -> 网卡
    ssize_t sent = send(sock_fd, buffer, file_size, 0);

    delete[] buffer;
    close(file_fd);
    return sent;
}

// 零拷贝: sendfile (2次拷贝，DMA完成)
ssize_t send_file_zerocopy(int sock_fd, const char* filename) {
    int file_fd = open(filename, O_RDONLY);
    struct stat stat_buf;
    fstat(file_fd, &stat_buf);
    off_t offset = 0;

    // 数据直接从文件 -> socket
    // 拷贝1: 磁盘 -> 内核缓冲 (DMA)
    // 拷贝2: 内核缓冲 -> 网卡 (DMA)
    // 无用户空间参与!
    ssize_t sent = sendfile(sock_fd, file_fd,
                           &offset, stat_buf.st_size);

    close(file_fd);
    return sent;
}

性能对比 (发送100MB文件):
方法              吞吐量      CPU使用率
──────────────────────────────────────
read/send         680 MB/s    85%
sendfile          920 MB/s    12%  (CPU节省73%)
```

#### 8.1.2 splice和vmsplice

```cpp
// 案例来源: 高性能代理服务器

#include <fcntl.h>

// splice: 在两个文件描述符间移动数据 (通过管道)
void proxy_connection_splice(int client_fd, int server_fd) {
    int pipe_fds[2];
    pipe(pipe_fds);

    const size_t SPLICE_SIZE = 64 * 1024;  // 64KB

    while (true) {
        // 客户端 -> 管道
        ssize_t n = splice(client_fd, nullptr,
                          pipe_fds[1], nullptr,
                          SPLICE_SIZE,
                          SPLICE_F_MOVE | SPLICE_F_MORE);

        if (n <= 0) break;

        // 管道 -> 服务器
        splice(pipe_fds[0], nullptr,
              server_fd, nullptr,
              n,
              SPLICE_F_MOVE | SPLICE_F_MORE);
    }

    close(pipe_fds[0]);
    close(pipe_fds[1]);
}

// vmsplice: 用户缓冲区 -> 管道 (零拷贝)
void send_with_vmsplice(int sock_fd, void* buffer, size_t size) {
    int pipe_fds[2];
    pipe(pipe_fds);

    struct iovec iov;
    iov.iov_base = buffer;
    iov.iov_len = size;

    // 用户空间 -> 管道 (映射，无拷贝)
    vmsplice(pipe_fds[1], &iov, 1, SPLICE_F_GIFT);

    // 管道 -> socket
    splice(pipe_fds[0], nullptr, sock_fd, nullptr,
          size, SPLICE_F_MOVE);

    close(pipe_fds[0]);
    close(pipe_fds[1]);
}

性能 (代理10Gbps网络):
方法           吞吐量       CPU使用率
────────────────────────────────────
recv/send      4.2 Gbps    95%
splice         9.5 Gbps    25%  (接近线速)
```

### 8.2 网络缓冲区调优

#### 8.2.1 Socket缓冲区优化

```cpp
// 案例来源: 高频交易、流媒体服务器

#include <sys/socket.h>

void optimize_socket_buffers(int sock_fd, bool is_server) {
    // 增大socket缓冲区
    int sndbuf_size = 4 * 1024 * 1024;  // 4MB发送缓冲
    int rcvbuf_size = 4 * 1024 * 1024;  // 4MB接收缓冲

    setsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF,
              &sndbuf_size, sizeof(sndbuf_size));
    setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF,
              &rcvbuf_size, sizeof(rcvbuf_size));

    // 禁用Nagle算法 (降低延迟)
    int flag = 1;
    setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY,
              &flag, sizeof(flag));

    // TCP快速打开 (TFO)
    if (is_server) {
        int qlen = 128;
        setsockopt(sock_fd, SOL_TCP, TCP_FASTOPEN,
                  &qlen, sizeof(qlen));
    }

    // TCP_CORK: 批量发送 (与NODELAY相反)
    // 用于发送大量小包
    #ifdef USE_CORK
    int cork = 1;
    setsockopt(sock_fd, IPPROTO_TCP, TCP_CORK,
              &cork, sizeof(cork));
    #endif

    // TCP窗口缩放
    int window_scale = 7;  // 窗口 * 2^7
    setsockopt(sock_fd, IPPROTO_TCP, TCP_WINDOW_CLAMP,
              &window_scale, sizeof(window_scale));

    // 启用TCP时间戳 (精确RTT)
    int timestamps = 1;
    setsockopt(sock_fd, IPPROTO_TCP, TCP_TIMESTAMP,
              &timestamps, sizeof(timestamps));

    // SO_BUSY_POLL: 低延迟轮询
    int busy_poll = 50;  // 微秒
    setsockopt(sock_fd, SOL_SOCKET, SO_BUSY_POLL,
              &busy_poll, sizeof(busy_poll));
}

// 系统级TCP调优
void system_tcp_tuning() {
    // /etc/sysctl.conf

    /*
    # 增大TCP缓冲区
    net.core.rmem_max = 134217728          # 128MB
    net.core.wmem_max = 134217728
    net.ipv4.tcp_rmem = 4096 87380 67108864  # min default max
    net.ipv4.tcp_wmem = 4096 65536 67108864

    # 拥塞控制算法
    net.ipv4.tcp_congestion_control = bbr   # Google BBR

    # 快速回收TIME_WAIT
    net.ipv4.tcp_tw_reuse = 1
    net.ipv4.tcp_fin_timeout = 30

    # 增大连接队列
    net.core.somaxconn = 4096
    net.ipv4.tcp_max_syn_backlog = 8192

    # TCP Fast Open
    net.ipv4.tcp_fastopen = 3  # client + server

    # 启用窗口缩放
    net.ipv4.tcp_window_scaling = 1

    # 选择性确认
    net.ipv4.tcp_sack = 1

    # 时间戳
    net.ipv4.tcp_timestamps = 1
    */
}

性能提升 (高带宽延迟积网络):
配置                吞吐量      延迟
────────────────────────────────────
默认                450 Mbps    250ms
优化缓冲区          2.8 Gbps    180ms
+ BBR拥塞控制       4.5 Gbps    120ms
+ TFO               4.6 Gbps    85ms  (首次连接)
```

---

## 9. 数据库性能优化

### 9.1 索引优化

#### 9.1.1 B+树 vs LSM树

```cpp
// 案例来源: MySQL InnoDB (B+树), RocksDB (LSM树)

// B+树索引 - 读优化
class BPlusTreeIndex {
    /*
    特点:
    • 读性能优异 O(log N)
    • 写需要随机IO
    • 范围查询高效
    • 需要定期碎片整理

    适用场景:
    • 读多写少
    • 需要范围查询
    • 传统OLTP
    */

public:
    void insert(Key key, Value val) {
        // 查找叶子节点 (log N次IO)
        LeafNode* leaf = find_leaf(key);

        // 插入 (可能导致分裂)
        if (leaf->is_full()) {
            split_leaf(leaf);  // 额外IO
        }

        leaf->insert(key, val);  // 随机写
    }

    Value lookup(Key key) {
        LeafNode* leaf = find_leaf(key);  // log N次IO
        return leaf->get(key);
    }

    // 优化: 缓存内部节点
    std::unordered_map<PageID, InternalNode*> node_cache;
};

// LSM树索引 - 写优化
class LSMTreeIndex {
    /*
    特点:
    • 写性能优异 (顺序写)
    • 读需要查多层 (读放大)
    • Compaction开销
    • 空间放大

    适用场景:
    • 写多读少
    • 日志、时序数据
    • NoSQL数据库
    */

private:
    MemTable* memtable;  // 内存表 (跳表/红黑树)
    std::vector<SSTable*> sstables;  // 磁盘有序表

public:
    void insert(Key key, Value val) {
        // 写入内存表 (无IO!)
        memtable->put(key, val);

        // 满了flush到磁盘
        if (memtable->size() > threshold) {
            flush_memtable();  // 顺序写
            trigger_compaction();  // 后台合并
        }
    }

    Value lookup(Key key) {
        // 1. 查内存表
        if (memtable->contains(key)) {
            return memtable->get(key);
        }

        // 2. 查各层SSTable (从新到旧)
        for (auto& sst : sstables) {
            if (sst->might_contain(key)) {  // Bloom filter
                if (auto val = sst->get(key)) {
                    return *val;
                }
            }
        }

        return Value();  // Not found
    }

    void compact() {
        // 合并多层SSTable
        // 减少读放大，但占用IO
    }
};

性能对比 (100GB数据):
                   B+树         LSM树
────────────────────────────────────
随机写             12k ops/s    85k ops/s
随机读             45k ops/s    28k ops/s
范围扫描           250 MB/s     180 MB/s
空间占用           100 GB       120 GB (写放大)
```

### 9.2 查询优化

#### 9.2.1 向量化执行引擎

```cpp
// 案例来源: ClickHouse, DuckDB, Apache Arrow

// 传统火山模型 (Volcano Model) - 行式处理
class TraditionalExecutor {
    // 每次处理一行，虚函数调用开销大

    virtual Tuple next() = 0;
};

class FilterOperator : public TraditionalExecutor {
    TraditionalExecutor* child;
    Predicate pred;

public:
    Tuple next() override {
        while (true) {
            Tuple tuple = child->next();  // 虚函数调用
            if (tuple.is_null()) return Tuple();
            if (pred.evaluate(tuple)) return tuple;
        }
    }
};

// 向量化执行 - 批量处理
class VectorizedExecutor {
    // 每次处理一批行 (1000-10000行)
    // 利用SIMD, 减少虚函数调用

    virtual size_t next_batch(Batch& batch) = 0;
};

class VectorizedFilter : public VectorizedExecutor {
    VectorizedExecutor* child;
    Predicate pred;

public:
    size_t next_batch(Batch& batch) override {
        size_t count = child->next_batch(batch);

        // 向量化谓词评估
        uint8_t* selected = pred.evaluate_batch(batch);

        // SIMD压缩: 只保留满足条件的行
        size_t output_count = compress_batch(
            batch, selected, count
        );

        return output_count;
    }
};

// SIMD过滤实现
size_t filter_batch_simd(
    int32_t* values,
    uint8_t* selected,
    size_t count,
    int32_t threshold
) {
    size_t output_idx = 0;

    #ifdef __AVX2__
    __m256i threshold_vec = _mm256_set1_epi32(threshold);

    for (size_t i = 0; i + 7 < count; i += 8) {
        // 加载8个值
        __m256i vals = _mm256_loadu_si256(
            (__m256i*)&values[i]
        );

        // 比较: vals > threshold
        __m256i cmp = _mm256_cmpgt_epi32(vals, threshold_vec);

        // 生成掩码
        int mask = _mm256_movemask_ps(
            _mm256_castsi256_ps(cmp)
        );

        // 压缩存储满足条件的值
        for (int j = 0; j < 8; j++) {
            if (mask & (1 << j)) {
                values[output_idx++] = values[i + j];
                selected[output_idx - 1] = 1;
            }
        }
    }
    #endif

    // 处理剩余
    for (size_t i = (count / 8) * 8; i < count; i++) {
        if (values[i] > threshold) {
            values[output_idx++] = values[i];
            selected[output_idx - 1] = 1;
        }
    }

    return output_idx;
}

性能对比 (扫描10亿行):
执行模式       时间      CPU效率
──────────────────────────────
行式处理       125s     IPC: 0.6
向量化         18s      IPC: 2.1  (7x faster)
```

---

由于篇幅限制，我将继续创建最后一部分。

