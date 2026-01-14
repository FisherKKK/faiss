# 计算机系统性能优化技巧大全 - Part 3

## 续前文...

## 10. 并发和同步优化

### 10.1 无锁数据结构

#### 10.1.1 无锁队列 (Lock-Free Queue)

```cpp
// 案例来源: Disruptor (LMAX), Folly (Facebook)

#include <atomic>

// 单生产者单消费者无锁队列 (SPSC)
template<typename T, size_t Size>
class SPSCQueue {
private:
    // 避免false sharing
    struct alignas(64) AlignedIndex {
        std::atomic<size_t> value;
    };

    AlignedIndex head;  // 消费者索引
    AlignedIndex tail;  // 生产者索引
    T buffer[Size];

public:
    SPSCQueue() {
        head.value.store(0, std::memory_order_relaxed);
        tail.value.store(0, std::memory_order_relaxed);
    }

    // 生产者: 入队
    bool enqueue(const T& item) {
        size_t tail_idx = tail.value.load(std::memory_order_relaxed);
        size_t next_tail = (tail_idx + 1) % Size;

        // 检查队列是否满
        size_t head_idx = head.value.load(std::memory_order_acquire);
        if (next_tail == head_idx) {
            return false;  // 满了
        }

        // 写入数据
        buffer[tail_idx] = item;

        // 更新tail (release语义，确保写入可见)
        tail.value.store(next_tail, std::memory_order_release);
        return true;
    }

    // 消费者: 出队
    bool dequeue(T& item) {
        size_t head_idx = head.value.load(std::memory_order_relaxed);

        // 检查队列是否空
        size_t tail_idx = tail.value.load(std::memory_order_acquire);
        if (head_idx == tail_idx) {
            return false;  // 空了
        }

        // 读取数据
        item = buffer[head_idx];

        // 更新head (release语义)
        size_t next_head = (head_idx + 1) % Size;
        head.value.store(next_head, std::memory_order_release);
        return true;
    }
};

// 多生产者多消费者无锁队列 (MPMC) - 更复杂
template<typename T, size_t Size>
class MPMCQueue {
private:
    struct Node {
        std::atomic<size_t> sequence;
        T data;
    };

    alignas(64) std::atomic<size_t> enqueue_pos;
    alignas(64) std::atomic<size_t> dequeue_pos;
    Node buffer[Size];

public:
    MPMCQueue() {
        enqueue_pos.store(0, std::memory_order_relaxed);
        dequeue_pos.store(0, std::memory_order_relaxed);

        for (size_t i = 0; i < Size; i++) {
            buffer[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    bool enqueue(const T& item) {
        Node* node;
        size_t pos = enqueue_pos.load(std::memory_order_relaxed);

        while (true) {
            node = &buffer[pos % Size];
            size_t seq = node->sequence.load(std::memory_order_acquire);
            intptr_t diff = (intptr_t)seq - (intptr_t)pos;

            if (diff == 0) {
                // 尝试占用这个位置
                if (enqueue_pos.compare_exchange_weak(
                    pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;  // 满了
            } else {
                pos = enqueue_pos.load(std::memory_order_relaxed);
            }
        }

        node->data = item;
        node->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool dequeue(T& item) {
        Node* node;
        size_t pos = dequeue_pos.load(std::memory_order_relaxed);

        while (true) {
            node = &buffer[pos % Size];
            size_t seq = node->sequence.load(std::memory_order_acquire);
            intptr_t diff = (intptr_t)seq - (intptr_t)(pos + 1);

            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(
                    pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;  // 空了
            } else {
                pos = dequeue_pos.load(std::memory_order_relaxed);
            }
        }

        item = node->data;
        node->sequence.store(
            pos + Size, std::memory_order_release
        );
        return true;
    }
};

性能对比 (1生产者1消费者):
实现              吞吐量         延迟
─────────────────────────────────────
std::queue + mutex  2.5 Mops/s   400ns
SPSC无锁队列        45 Mops/s    22ns  (18x faster)

(4生产者4消费者):
std::queue + mutex  1.2 Mops/s   850ns
MPMC无锁队列        28 Mops/s    36ns  (23x faster)
```

#### 10.1.2 无锁哈希表

```cpp
// 案例来源: folly::AtomicHashMap, Intel TBB

template<typename K, typename V, size_t Size>
class LockFreeHashMap {
private:
    struct Entry {
        std::atomic<K> key;
        std::atomic<V> value;
        std::atomic<bool> occupied;

        Entry() : occupied(false) {}
    };

    Entry table[Size];

    size_t hash(const K& key) const {
        return std::hash<K>{}(key) % Size;
    }

public:
    // 插入 (CAS)
    bool insert(const K& key, const V& value) {
        size_t idx = hash(key);

        // 线性探测
        for (size_t i = 0; i < Size; i++) {
            size_t pos = (idx + i) % Size;
            Entry& entry = table[pos];

            // 尝试占用空位
            bool expected = false;
            if (entry.occupied.compare_exchange_strong(
                expected, true, std::memory_order_acquire)) {

                // 成功占用，写入数据
                entry.key.store(key, std::memory_order_relaxed);
                entry.value.store(value, std::memory_order_release);
                return true;
            }

            // 位置已被占用，检查是否是相同key
            if (entry.key.load(std::memory_order_relaxed) == key) {
                // 更新值
                entry.value.store(value, std::memory_order_release);
                return true;
            }
        }

        return false;  // 表满
    }

    // 查找
    bool find(const K& key, V& value) const {
        size_t idx = hash(key);

        for (size_t i = 0; i < Size; i++) {
            size_t pos = (idx + i) % Size;
            const Entry& entry = table[pos];

            if (!entry.occupied.load(std::memory_order_acquire)) {
                return false;  // 找到空位，不存在
            }

            if (entry.key.load(std::memory_order_relaxed) == key) {
                value = entry.value.load(std::memory_order_acquire);
                return true;
            }
        }

        return false;
    }
};

// 更高级: Hopscotch Hashing
// • 限制探测距离
// • 使用位图标记邻近bucket
// • 更好的缓存局部性
```

### 10.2 读写锁优化

#### 10.2.1 序列锁 (Seqlock)

```cpp
// 案例来源: Linux内核 (时钟、路由表)

#include <atomic>

// Seqlock: 写优先，读乐观无锁
template<typename T>
class SeqLock {
private:
    std::atomic<uint64_t> sequence;
    T data;
    mutable std::atomic_flag write_lock = ATOMIC_FLAG_INIT;

public:
    SeqLock() {
        sequence.store(0, std::memory_order_relaxed);
    }

    // 写操作 (获取锁)
    void write(const T& new_data) {
        // 获取写锁
        while (write_lock.test_and_set(std::memory_order_acquire)) {
            // 自旋等待
        }

        // 增加序列号 (标记开始写)
        uint64_t seq = sequence.load(std::memory_order_relaxed);
        sequence.store(seq + 1, std::memory_order_release);

        // 写入数据
        data = new_data;

        // 再次增加序列号 (标记完成写)
        sequence.store(seq + 2, std::memory_order_release);

        // 释放写锁
        write_lock.clear(std::memory_order_release);
    }

    // 读操作 (无锁)
    T read() const {
        T result;

        while (true) {
            // 读取序列号
            uint64_t seq1 = sequence.load(std::memory_order_acquire);

            // 检查是否有写者 (奇数表示正在写)
            if (seq1 & 1) {
                continue;  // 有写者，重试
            }

            // 读取数据
            result = data;

            // 内存屏障
            std::atomic_thread_fence(std::memory_order_acquire);

            // 再次读取序列号
            uint64_t seq2 = sequence.load(std::memory_order_acquire);

            // 检查数据是否一致
            if (seq1 == seq2) {
                return result;  // 成功
            }

            // 数据不一致，重试
        }
    }
};

// 使用示例
struct Point3D {
    double x, y, z;
};

SeqLock<Point3D> position;

// 写线程 (低频)
void update_position() {
    Point3D new_pos = {1.0, 2.0, 3.0};
    position.write(new_pos);
}

// 读线程 (高频)
void read_position() {
    Point3D pos = position.read();  // 无锁!
    use_position(pos);
}

适用场景:
✓ 读多写少 (99:1或更高)
✓ 写操作简短
✓ 数据结构小 (< 缓存行)
✗ 写频繁 (读者饥饿)
✗ 数据结构大 (拷贝开销)

性能 (99%读, 1%写):
实现                读操作延迟    写操作延迟
────────────────────────────────────────
std::shared_mutex   180ns        650ns
SeqLock             12ns         680ns  (15x faster read)
```

#### 10.2.2 RCU (Read-Copy-Update)

```cpp
// 案例来源: Linux内核

// RCU原理:
// • 读者无同步开销
// • 写者创建新版本
// • 等待所有读者完成后删除旧版本

#include <atomic>
#include <vector>

template<typename T>
class RCUList {
private:
    struct Node {
        T data;
        std::atomic<Node*> next;

        Node(const T& d) : data(d), next(nullptr) {}
    };

    std::atomic<Node*> head;
    std::vector<Node*> retired_nodes;  // 待删除节点
    std::atomic<int> grace_period;

public:
    RCUList() : head(nullptr), grace_period(0) {}

    // 读操作 (无锁)
    void read(std::function<void(const T&)> callback) {
        // 进入读临界区 (记录grace period)
        int gp = grace_period.load(std::memory_order_acquire);

        // 遍历链表
        Node* curr = head.load(std::memory_order_acquire);
        while (curr != nullptr) {
            callback(curr->data);
            curr = curr->next.load(std::memory_order_acquire);
        }

        // 离开读临界区
    }

    // 写操作: 添加节点
    void insert(const T& data) {
        Node* new_node = new Node(data);

        Node* old_head = head.load(std::memory_order_relaxed);
        do {
            new_node->next.store(old_head, std::memory_order_relaxed);
        } while (!head.compare_exchange_weak(
            old_head, new_node,
            std::memory_order_release,
            std::memory_order_relaxed
        ));
    }

    // 写操作: 删除节点
    void remove(const T& data) {
        Node* prev = nullptr;
        Node* curr = head.load(std::memory_order_acquire);

        while (curr != nullptr) {
            if (curr->data == data) {
                // 从链表中移除
                Node* next = curr->next.load(std::memory_order_acquire);

                if (prev == nullptr) {
                    head.store(next, std::memory_order_release);
                } else {
                    prev->next.store(next, std::memory_order_release);
                }

                // 加入retire列表，等待grace period后删除
                retired_nodes.push_back(curr);

                // 增加grace period
                grace_period.fetch_add(1, std::memory_order_release);

                return;
            }

            prev = curr;
            curr = curr->next.load(std::memory_order_acquire);
        }
    }

    // 清理旧版本 (后台线程)
    void reclaim() {
        // 等待所有读者完成当前grace period
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // 删除retire的节点
        for (Node* node : retired_nodes) {
            delete node;
        }
        retired_nodes.clear();
    }
};

性能 (读写比例 95:5):
实现                  读延迟    写延迟    吞吐量
────────────────────────────────────────────
std::mutex            250ns    280ns     8 Mops
std::shared_mutex     120ns    450ns     15 Mops
RCU                   8ns      900ns     42 Mops  (读接近无开销)
```

### 10.3 内存顺序优化

#### 10.3.1 正确使用memory_order

```cpp
// 案例来源: Folly, Boost.Lockfree

#include <atomic>

// 示例1: 生产者-消费者
struct Message {
    int data;
    bool ready;
};

std::atomic<Message*> message_ptr{nullptr};

// 生产者
void producer() {
    Message* msg = new Message;
    msg->data = 42;

    // 确保data写入对消费者可见
    message_ptr.store(msg, std::memory_order_release);
}

// 消费者
void consumer() {
    Message* msg;

    // 等待消息
    while ((msg = message_ptr.load(
        std::memory_order_acquire)) == nullptr) {
        std::this_thread::yield();
    }

    // 此时可以安全访问msg->data
    process(msg->data);
}

// 示例2: 双重检查锁 (DCLP)
class Singleton {
private:
    static std::atomic<Singleton*> instance;
    static std::mutex mutex;

    Singleton() {}

public:
    static Singleton* get_instance() {
        // 第一次检查 (无锁)
        Singleton* tmp = instance.load(std::memory_order_acquire);

        if (tmp == nullptr) {
            std::lock_guard<std::mutex> lock(mutex);

            // 第二次检查
            tmp = instance.load(std::memory_order_relaxed);
            if (tmp == nullptr) {
                tmp = new Singleton;

                // 确保对象完全构造后再发布
                instance.store(tmp, std::memory_order_release);
            }
        }

        return tmp;
    }
};

// 示例3: 无锁计数器
class Counter {
    std::atomic<int> count{0};

public:
    // 如果不关心顺序，使用relaxed最快
    void increment() {
        count.fetch_add(1, std::memory_order_relaxed);
    }

    int get() const {
        return count.load(std::memory_order_relaxed);
    }

    // 如果需要与其他操作同步，使用acquire/release
    void increment_sync() {
        count.fetch_add(1, std::memory_order_release);
    }

    int get_sync() const {
        return count.load(std::memory_order_acquire);
    }
};

// 性能对比 (1000万次操作):
/*
memory_order          操作延迟
──────────────────────────────
relaxed               2.5ns
acquire/release       3.8ns
seq_cst (默认)        6.2ns
*/

选择指南:
• relaxed: 独立原子操作，无需同步
• acquire/release: 生产者-消费者，发布-订阅
• seq_cst: 需要全局一致性顺序 (最安全但最慢)
```

---

## 11. 内核态优化技巧

### 11.1 eBPF高性能追踪

#### 11.1.1 使用BPF优化网络

```c
// 案例来源: Cilium, Cloudflare

// BPF程序: XDP (eXpress Data Path) - 最早的包处理点
// 在网卡驱动层丢弃DDoS流量

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>

// XDP程序: 过滤SYN flood
SEC("xdp")
int xdp_filter_syn_flood(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    // 解析以太网头
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    // 只处理IPv4
    if (eth->h_proto != htons(ETH_P_IP))
        return XDP_PASS;

    // 解析IP头
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;

    // 只处理TCP
    if (ip->protocol != IPPROTO_TCP)
        return XDP_PASS;

    // 解析TCP头
    struct tcphdr *tcp = (void *)ip + (ip->ihl * 4);
    if ((void *)(tcp + 1) > data_end)
        return XDP_PASS;

    // 检查是否是SYN包
    if (tcp->syn && !tcp->ack) {
        // 查询速率限制map
        __u32 key = ip->saddr;
        __u64 *count = bpf_map_lookup_elem(&syn_count, &key);

        if (count) {
            *count += 1;

            // 超过阈值，丢弃
            if (*count > SYN_THRESHOLD) {
                return XDP_DROP;  // 在网卡层丢弃!
            }
        } else {
            __u64 init_count = 1;
            bpf_map_update_elem(&syn_count, &key,
                               &init_count, BPF_ANY);
        }
    }

    return XDP_PASS;
}

性能对比 (10Gbps网络, DDoS攻击):
方法                处理能力    CPU使用率
────────────────────────────────────────
iptables            2.5 Gbps   100%
XDP + BPF           9.8 Gbps   15%  (接近线速)

优势:
• 最早的包处理点 (网卡驱动)
• 零拷贝
• JIT编译
• 可以直接转发到其他网卡
```

#### 11.1.2 BPF性能追踪

```bash
# 案例来源: Netflix性能分析

# 1. 追踪慢系统调用
bpftrace -e '
tracepoint:syscalls:sys_enter_* {
    @start[tid] = nsecs;
}

tracepoint:syscalls:sys_exit_* {
    $duration = nsecs - @start[tid];
    if ($duration > 1000000) {  # > 1ms
        printf("%s took %d ms\n", probe, $duration / 1000000);
    }
    delete(@start[tid]);
}'

# 2. 追踪磁盘IO延迟
biolatency.bt  # 内置工具

# 输出直方图:
#    usecs          : count    distribution
#        0 -> 1     : 0        |                    |
#        2 -> 3     : 0        |                    |
#        4 -> 7     : 45       |*                   |
#        8 -> 15    : 234      |******              |
#       16 -> 31    : 812      |********************|
#       32 -> 63    : 456      |***********         |
#       64 -> 127   : 123      |***                 |

# 3. 追踪TCP重传
tcpretrans.bt

# 4. CPU采样 (类似perf)
profile.bt -F 99  # 99Hz采样

# 5. 自定义追踪: MySQL查询延迟
bpftrace -e '
usdt:/usr/bin/mysqld:mysql:query__start {
    @start[tid] = nsecs;
}

usdt:/usr/bin/mysqld:mysql:query__done {
    $duration = (nsecs - @start[tid]) / 1000000;
    @query_latency = hist($duration);
    delete(@start[tid]);
}'

优势:
• 生产环境安全 (沙箱执行)
• 低开销 (< 1%)
• 动态插桩 (无需重编译)
• 内核和用户态统一追踪
```

### 11.2 内核旁路技术

#### 11.2.1 DPDK (Data Plane Development Kit)

```c
// 案例来源: 高性能路由器、防火墙、NFV

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>

// DPDK核心: 用户态网络栈
// 特点:
// • 轮询模式 (PMD)
// • 零拷贝
// • 大页内存
// • CPU亲和性

// 初始化DPDK
int init_dpdk(int argc, char **argv) {
    // 初始化EAL (Environment Abstraction Layer)
    int ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "EAL init failed\n");

    // 检查网卡
    uint16_t nb_ports = rte_eth_dev_count_avail();
    printf("Found %u ports\n", nb_ports);

    return ret;
}

// 配置网卡
void setup_port(uint16_t port_id) {
    struct rte_eth_conf port_conf = {};

    // RSS (Receive Side Scaling)
    port_conf.rxmode.mq_mode = ETH_MQ_RX_RSS;
    port_conf.rx_adv_conf.rss_conf.rss_hf =
        ETH_RSS_IP | ETH_RSS_TCP | ETH_RSS_UDP;

    // 配置队列
    uint16_t nb_rx_queues = 4;
    uint16_t nb_tx_queues = 4;

    rte_eth_dev_configure(port_id, nb_rx_queues,
                         nb_tx_queues, &port_conf);

    // 分配mbuf池
    struct rte_mempool *mbuf_pool = rte_pktmbuf_pool_create(
        "MBUF_POOL",
        8192,           // 池大小
        250,            // cache大小
        0,
        RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id()
    );

    // 设置RX队列
    for (uint16_t q = 0; q < nb_rx_queues; q++) {
        rte_eth_rx_queue_setup(port_id, q, 1024,
                              rte_eth_dev_socket_id(port_id),
                              NULL, mbuf_pool);
    }

    // 设置TX队列
    for (uint16_t q = 0; q < nb_tx_queues; q++) {
        rte_eth_tx_queue_setup(port_id, q, 1024,
                              rte_eth_dev_socket_id(port_id),
                              NULL);
    }

    // 启动网卡
    rte_eth_dev_start(port_id);
}

// 收发包主循环
int packet_processing_loop(void *arg) {
    uint16_t port_id = *(uint16_t *)arg;
    uint16_t queue_id = rte_lcore_id();

    struct rte_mbuf *pkts[BURST_SIZE];

    while (1) {
        // 批量接收包 (零拷贝)
        uint16_t nb_rx = rte_eth_rx_burst(
            port_id, queue_id,
            pkts, BURST_SIZE
        );

        if (nb_rx == 0)
            continue;

        // 处理包
        for (uint16_t i = 0; i < nb_rx; i++) {
            process_packet(pkts[i]);
        }

        // 批量发送包
        uint16_t nb_tx = rte_eth_tx_burst(
            port_id, queue_id,
            pkts, nb_rx
        );

        // 释放未发送的包
        if (unlikely(nb_tx < nb_rx)) {
            for (uint16_t i = nb_tx; i < nb_rx; i++) {
                rte_pktmbuf_free(pkts[i]);
            }
        }
    }

    return 0;
}

性能对比 (10Gbps网络, 64字节包):
方法                  PPS           CPU核心
─────────────────────────────────────────────
Linux网络栈           1.5 Mpps     4核
DPDK (轮询)          14.88 Mpps    1核  (线速)

吞吐量 (1500字节包):
Linux: 9.2 Gbps
DPDK:  10.0 Gbps (线速)
```

---

## 12. 实战案例集锦

### 12.1 案例1: Redis性能优化全流程

```
场景: Redis单实例，QPS从10万提升到100万

问题诊断:
1. 使用redis-benchmark测试
   $ redis-benchmark -t get,set -n 1000000 -q
   SET: 98,234 requests per second
   GET: 105,678 requests per second

2. 使用perf分析
   $ perf record -g -p <redis-pid>
   $ perf report

   发现热点:
   • 35% zmalloc/zfree (内存分配)
   • 28% dictFind (哈希查找)
   • 15% addReply (网络发送)
   • 12% readQueryFromClient (网络接收)

优化步骤:

Step 1: 内存分配器优化
━━━━━━━━━━━━━━━━━━━━━━━
问题: glibc malloc在高并发下性能差

解决:
# 使用jemalloc重新编译Redis
$ make MALLOC=jemalloc

效果: QPS +25% (105k → 131k)

Step 2: 网络优化
━━━━━━━━━━━━━━━━━
配置TCP参数:
# redis.conf
tcp-backlog 511
tcp-keepalive 300

# /etc/sysctl.conf
net.core.somaxconn = 4096
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.tcp_tw_reuse = 1

客户端使用连接池，启用pipeline

效果: QPS +40% (131k → 183k)

Step 3: CPU绑定
━━━━━━━━━━━━━━━━
# 绑定Redis到专用CPU核心
taskset -c 0-3 redis-server redis.conf

# 禁用THP (透明大页可能导致延迟抖动)
echo never > /sys/kernel/mm/transparent_hugepage/enabled

效果: QPS +15%, P99延迟 -40%

Step 4: 数据结构优化
━━━━━━━━━━━━━━━━━━━
# redis.conf
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
set-max-intset-entries 512

小对象使用压缩数据结构

效果: 内存 -30%, QPS +8%

Step 5: 持久化优化
━━━━━━━━━━━━━━━━━━
# 禁用AOF (如果可接受)
appendonly no

# 或使用AOF + everysec
appendonly yes
appendfsync everysec
no-appendfsync-on-rewrite yes

效果: QPS +35% (写密集场景)

Step 6: IO优化
━━━━━━━━━━━━━━
# 使用io_uring (Redis 7.0+)
# redis.conf
io-threads 4
io-threads-do-reads yes

效果: QPS +45%

最终结果:
━━━━━━━━━━━━━━
优化前: 105k QPS, p99: 850μs
优化后: 1.05M QPS, p99: 320μs

总提升: 10x QPS, 2.6x lower latency
```

### 12.2 案例2: 视频转码服务优化

```
场景: FFmpeg视频转码，从30fps优化到实时120fps

初始状态:
━━━━━━━━━━━━━━
输入: 1080p@60fps H.264
输出: 720p@60fps H.264
性能: 30fps (2x slower than real-time)
CPU: Intel Xeon Gold 6248R (48核)

问题分析:
━━━━━━━━━━━━━━
$ perf record -g ffmpeg -i input.mp4 output.mp4
$ perf report

热点:
• 62% x264_pixel_sad_16x16 (运动估计)
• 18% x264_dct_quant (DCT变换)
• 12% x264_deblock_h (去块滤波)

优化过程:

Step 1: 启用SIMD优化
━━━━━━━━━━━━━━━━━━━━
# 编译时启用AVX2
./configure --enable-nonfree --enable-libx264 \
    --cpu=native --enable-avx2

# x264 preset调整
ffmpeg -i input.mp4 -c:v libx264 \
    -preset ultrafast \    # 快速preset
    -tune zerolatency \    # 低延迟调优
    output.mp4

效果: 30fps → 85fps (2.8x)

Step 2: 多线程优化
━━━━━━━━━━━━━━━━━━
# 启用slice-based多线程
ffmpeg -i input.mp4 -c:v libx264 \
    -threads 16 \          # 线程数
    -slices 8 \            # slice数
    output.mp4

效果: 85fps → 145fps (4.8x)

Step 3: 硬件加速
━━━━━━━━━━━━━━━━━
# 使用Intel Quick Sync Video (QSV)
ffmpeg -hwaccel qsv -c:v h264_qsv -i input.mp4 \
    -c:v h264_qsv -preset fast \
    -global_quality 23 \
    output.mp4

# 或NVIDIA NVENC
ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
    -i input.mp4 \
    -c:v h264_nvenc -preset p4 \
    output.mp4

效果: 145fps → 480fps (16x, GPU加速)

Step 4: 批处理优化
━━━━━━━━━━━━━━━━━━
# 使用GNU Parallel批量处理
ls *.mp4 | parallel -j 8 \
    ffmpeg -i {} -c:v libx264 {.}_out.mp4

# 8个视频并行转码
效果: 吞吐量 8x

Step 5: 参数调优
━━━━━━━━━━━━━━━━
# x264高级参数
ffmpeg -i input.mp4 -c:v libx264 \
    -preset ultrafast \
    -crf 23 \
    -refs 2 \              # 参考帧数
    -bf 0 \                # 禁用B帧
    -me_method dia \       # 快速运动估计
    -subq 2 \              # 子像素运动估计质量
    -trellis 0 \           # 禁用trellis量化
    output.mp4

最终结果:
━━━━━━━━━━━━━━
CPU软编码: 145fps (多线程优化)
GPU硬编码: 480fps (QSV/NVENC)
批量处理: 1160fps (8并发 × 145fps)

相比初始: 38x faster!
```

### 12.3 案例3: 机器学习推理优化

```
场景: ResNet-50图像分类，延迟从100ms降到5ms

环境:
━━━━━━━━━━━━━━
模型: ResNet-50 (25M参数)
输入: 224x224x3 图像
硬件: Intel Xeon + NVIDIA T4 GPU

初始实现 (PyTorch):
━━━━━━━━━━━━━━━━━━━━━
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()

# 推理
with torch.no_grad():
    output = model(input_tensor)

性能: 100ms/image (CPU)

优化路径:

Step 1: GPU加速
━━━━━━━━━━━━━━━━
model = model.cuda()
input_tensor = input_tensor.cuda()

性能: 100ms → 15ms (6.7x)

Step 2: FP16混合精度
━━━━━━━━━━━━━━━━━━━
model = model.half()  # FP16
input_tensor = input_tensor.half()

with torch.cuda.amp.autocast():
    output = model(input_tensor)

性能: 15ms → 8ms (12.5x)

Step 3: TensorRT优化
━━━━━━━━━━━━━━━━━━━━
import torch_tensorrt

# 编译模型为TensorRT引擎
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        shape=[1, 3, 224, 224],
        dtype=torch.half
    )],
    enabled_precisions={torch.half},
    workspace_size=1 << 30
)

性能: 8ms → 3.5ms (28.6x)

Step 4: 动态batching
━━━━━━━━━━━━━━━━━━━━
# Triton Inference Server配置
max_batch_size: 32
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 500
}

# 批量推理
batch_output = trt_model(batch_input)  # 32张图

单图延迟: 3.5ms
批量吞吐: 5000 images/sec (vs 286 images/sec)

Step 5: 模型量化 (INT8)
━━━━━━━━━━━━━━━━━━━━━
# 使用TensorRT INT8量化
trt_model_int8 = torch_tensorrt.compile(
    model,
    inputs=[...],
    enabled_precisions={torch.int8},
    calibrator=calibrator  # 校准数据
)

性能: 3.5ms → 2.2ms (45.5x)
精度损失: < 1%

Step 6: ONNX Runtime优化
━━━━━━━━━━━━━━━━━━━━━━━
import onnxruntime as ort

# 导出ONNX
torch.onnx.export(model, input_tensor, "resnet50.onnx")

# 使用ONNX Runtime (CPU)
session = ort.InferenceSession(
    "resnet50.onnx",
    providers=['CPUExecutionProvider']
)

# Intel优化: OpenVINO
# 性能: 12ms (CPU, 优于原始PyTorch)

最终对比:
━━━━━━━━━━━━━━
PyTorch CPU:           100ms
PyTorch GPU (FP32):     15ms
PyTorch GPU (FP16):      8ms
TensorRT (FP16):        3.5ms
TensorRT (INT8):        2.2ms  ← 最佳单图延迟
TensorRT Batching:    0.2ms/image (32 batch)

总提升: 45x latency, 500x throughput!
```

---

## 总结：性能优化最佳实践

### 通用原则

```
1. 测量先行
   ✓ 使用profiler找瓶颈
   ✓ 关注热点代码 (80/20法则)
   ✗ 不要猜测，不要过早优化

2. 由粗到细
   ✓ 先优化算法 O(n²) → O(n log n)
   ✓ 再优化数据结构
   ✓ 最后优化微架构

3. 理解硬件
   ✓ 缓存层次结构
   ✓ SIMD并行
   ✓ 分支预测
   ✓ NUMA拓扑

4. 权衡取舍
   ✓ 速度 vs 内存
   ✓ 延迟 vs 吞吐
   ✓ 可读性 vs 性能

5. 持续迭代
   ✓ 优化-测量-验证
   ✓ A/B测试
   ✓ 回归测试
```

### 优化决策树

```
性能问题?
│
├─ CPU Bound?
│  ├─ IPC < 1.0? → 优化缓存/分支
│  ├─ IPC > 2.0? → 优化算法/并行
│  └─ 单热点? → 优化该函数
│
├─ Memory Bound?
│  ├─ Cache Miss > 10%? → 改进数据布局
│  ├─ TLB Miss > 1%? → 使用大页
│  └─ 跨NUMA? → 绑定线程和内存
│
├─ IO Bound?
│  ├─ 磁盘IO? → 异步IO, Direct IO
│  ├─ 网络IO? → 零拷贝, 批处理
│  └─ 系统调用多? → 批量操作
│
└─ 锁竞争?
   ├─ 读多写少? → RCU, SeqLock
   ├─ 短临界区? → 自旋锁
   └─ 可分片? → 分段锁, 无锁结构
```

### 工具箱速查

```
通用分析:
perf stat/record/report    - Linux性能分析
VTune Profiler             - Intel微架构分析
AMD uProf                  - AMD处理器分析
gprof                      - 函数级profiling

缓存分析:
valgrind --tool=cachegrind - 缓存模拟
perf c2c                   - 缓存一致性
Intel CAT                  - 缓存分区

内存分析:
valgrind --tool=massif     - 堆分析
jemalloc/tcmalloc          - 高性能分配器
numactl/numastat           - NUMA分析

网络分析:
iperf3                     - 带宽测试
netstat/ss                 - 连接状态
tcpdump/wireshark          - 包分析
eBPF/bpftrace              - 内核追踪

编译优化:
-O3 -march=native          - 激进优化
-flto                      - 链接时优化
-fprofile-generate/use     - PGO
```

### 推荐阅读资源

```
书籍:
• "Systems Performance" - Brendan Gregg
• "Computer Architecture" - Hennessy & Patterson
• "The Art of Multiprocessor Programming"

文档:
• Intel Optimization Reference Manual
• AMD Software Optimization Guide
• ARM Cortex-A Series Programmer's Guide

网站:
• Agner Fog's Optimization Manuals
• Brendan Gregg's Blog
• LWN.net (Linux内核)

工具:
• https://www.brendangregg.com/perf.html
• https://github.com/KDAB/hotspot
• https://github.com/iovisor/bpftrace
```

---

**课程结束**

这份指南涵盖了从CPU微架构到网络协议栈的全方位性能优化技巧。记住：
- 性能优化是科学，不是艺术
- 测量是关键
- 理解原理比记住技巧更重要
- 不要为了优化而优化

祝你在性能优化的道路上取得成功！🚀
