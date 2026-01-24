# Faiss深度课程 - 第12天：GPU实现 - CUDA与ROCm架构

## 课程目标

理解Faiss的GPU实现，学习如何利用CUDA和ROCm加速向量搜索，掌握CPU-GPU数据传输和内存管理。

---

## 1. GPU架构概述

### 1.1 支持的平台

| 平台 | 厂商 | 架构 |
|------|------|------|
| CUDA | NVIDIA | Tesla, RTX系列 |
| ROCm | AMD | Radeon, Instinct系列 |

### 1.2 Faiss GPU支持

```cpp
// 编译选项
-DCMAKE_CUDA_ARCHITECTURES="75;72"  # NVIDIA GPU
-DFAISS_ENABLE_GPU=ON                 # 启用GPU
-DFAISS_ENABLE_ROCM=ON                 # 启用ROCm
-DFAISS_ENABLE_CUVS=ON                 # 启用cuVS
```

---

## 2. GPU资源管理

### 2.1 GpuResources

```cpp
// faiss/gpu/GpuResources.h
struct GpuResources {
    int numDevices;           // GPU设备数

    // 为每个设备分配流和内存
    std::vector<std::shared_ptr<Stream>> streams_;
    std::vector<std::shared_ptr>> MemorySpace>> memSpaces_;

    // 获取GPU资源
    static std::shared_ptr<GpuResources> getResources();

    // 获取指定GPU的流
    Stream* getDefaultStream(int device);

    // 获取内存空间
    MemorySpace* getMemorySpace(int device);
};

// 使用示例
void gpu_resources_example() {
    auto res = GpuResources::getResources();
    int device = 0;

    Stream* stream = res->getDefaultStream(device);
    MemorySpace* mem = res->getMemorySpace(device);
}
```

### 2.2 StandardGpuResources

```cpp
// faiss/gpu/StandardGpuResources.h
struct StandardGpuResources : GpuResources {
    StandardGpuResources();

    // 配置选项
    int numDevices;           // 使用的GPU数（-1表示全部）
    std::vector<int> device;  // 指定GPU ID

    // 内存管理
    size_t tempMemSize;       // 临时内存大小
    bool pinnedMemory;        // 是否使用固定内存
};
```

---

## 3. GPU索引基础

### 3.1 GpuIndex基类

```cpp
// faiss/gpu/GpuIndex.h
struct GpuIndex {
    std::shared_ptr<GpuResources> resources_;
    int device_;                    // GPU设备ID
    GpuIndexConfig config_;

    GpuIndex(std::shared_ptr<GpuResources> resources,
            int device,
            GpuIndexConfig config)
        : resources_(resources),
          device_(device),
          config_(config) {}

    virtual ~GpuIndex() {}

    // 与CPU索引相同的接口
    virtual void add(idx_t n, const float* x) = 0;
    virtual void search(
        idx_t n, const float* x, idx_t k,
        float* distances, idx_t* labels) = 0;
};
```

### 3.2 GpuIndexFlat

```cpp
// faiss/gpu/GpuIndexFlat.h
struct GpuIndexFlat : GpuIndex {
    FlatIndex* data_;  // GPU上的向量数据

    GpuIndexFlat(
            std::shared_ptr<GpuResources> resources,
            int d,
            MetricType metric,
            GpuIndexFlatConfig config);

    void add(idx_t n, const float* x) override {
        // 1. 分配GPU内存
        data_->add(n, x);

        // 2. 复制数据到GPU
        cudaMemcpy(data_->get() + data_->getNumVecs() * d,
                   x, n * d * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override {

        // 1. 复制查询到GPU
        float* x_device;
        cudaMalloc(&x_device, n * d * sizeof(float));
        cudaMemcpy(x_device, x, n * d * sizeof(float),
                   cudaMemcpyHostToDevice);

        // 2. 分配结果内存
        float* distances_device;
        idx_t* labels_device;
        cudaMalloc(&distances_device, n * k * sizeof(float));
        cudaMalloc(&labels_device, n * k * sizeof(idx_t));

        // 3. GPU搜索
        search_kernel<<<grid, block>>>(
            data_->get(), data_->getNumVecs(), d,
            x_device, n, k,
            distances_device, labels_device);

        // 4. 复制结果回CPU
        cudaMemcpy(distances, distances_device,
                   n * k * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(labels, labels_device,
                   n * k * sizeof(idx_t),
                   cudaMemcpyDeviceToHost);

        // 5. 清理
        cudaFree(x_device);
        cudaFree(distances_device);
        cudaFree(labels_device);
    }
};
```

---

## 4. CUDA内核实现

### 4.1 L2距离内核

```cpp
// CUDA内核：计算L2距离
__global__ void l2_distance_kernel(
        const float* database,  // nb × d
        const float* queries,   // nq × d
        int nb, int nq, int d,
        float* distances) {     // nq × nb

    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int db_idx = blockIdx.y;

    if (query_idx >= nq || db_idx >= nb) {
        return;
    }

    // 计算距离
    float dis = 0.0f;
    for (int i = 0; i < d; i++) {
        float diff = queries[query_idx * d + i] -
                     database[db_idx * d + i];
        dis += diff * diff;
    }

    distances[query_idx * nb + db_idx] = dis;
}

// 调用
void call_l2_kernel(
        const float* database, const float* queries,
        int nb, int nq, int d,
        float* distances) {

    dim3 block(256);
    dim3 grid((nq + block.x - 1) / block.x, nb);

    l2_distance_kernel<<<grid, block>>>(
        database, queries, nb, nq, d, distances);

    cudaDeviceSynchronize();
}
```

### 4.2 内积内核

```cpp
__global__ void inner_product_kernel(
        const float* database,
        const float* queries,
        int nb, int nq, int d,
        float* ips) {

    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int db_idx = blockIdx.y;

    if (query_idx >= nq || db_idx >= nb) {
        return;
    }

    float ip = 0.0f;
    for (int i = 0; i < d; i++) {
        ip += queries[query_idx * d + i] *
              database[db_idx * d + i];
    }

    ips[query_idx * nb + db_idx] = ip;
}
```

### 4.3 Top-K选择

```cpp
// Top-K选择（简化版本）
__global__ void topk_kernel(
        const float* distances,  // nq × nb
        int nq, int nb, int k,
        float* topk_distances,  // nq × k
        idx_t* topk_labels) {    // nq × k

    int query_idx = blockIdx.x;
    if (query_idx >= nq) return;

    const float* dis = distances + query_idx * nb;

    // 简单的冒泡排序（实际应使用更高效的算法）
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < nb; j++) {
            if (dis[j] < dis[i]) {
                // 交换
                float temp_dis = dis[j];
                dis[j] = dis[i];
                dis[i] = temp_dis;
            }
        }
        topk_distances[query_idx * k + i] = dis[i];
        topk_labels[query_idx * k + i] = i;
    }
}
```

---

## 5. GPU IVF索引

### 5.1 GpuIndexIVFFlat

```cpp
// faiss/gpu/GpuIndexIVFFlat.h
struct GpuIndexIVFFlat : GpuIndexIVF {
    // IVF列表的GPU表示
    std::vector<IVFList> ivf_lists_;

    void add(idx_t n, const float* x) override {
        // 1. 粗量化（CPU或GPU）
        idx_t* list_nos = new idx_t[n];
        quantizer_->assign(n, x, list_nos);

        // 2. 复制到GPU
        for (idx_t i = 0; i < n; i++) {
            idx_t list_no = list_nos[i];
            ivf_lists_[list_no].add(x + i * d);
        }

        delete[] list_nos;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

        // 1. 分配查询到列表（GPU）
        idx_t* list_nos_device;
        float* centroid_dis_device;
        // ... cudaMalloc和cudaMemcpy

        // 2. 并行搜索每个列表
        for (size_t list = 0; list < nlist; list++) {
            search_ivf_list<<<grid, block>>>(
                ivf_lists_[list],
                x, list, nprobe, k,
                distances, labels);
        }
    }
};
```

---

## 6. CPU-GPU互操作

### 6.1 GpuCloner

```cpp
// faiss/gpu/GpuCloner.h
struct GpuCloner {
    // 将CPU索引复制到GPU
    static Index* copyCpuToGpu(
            Index* cpu_index,
            std::shared_ptr<GpuResources> resources,
            int device = 0) {

        Index* gpu_index = nullptr;

        switch (cpu_index->index_type) {
            case Index::IndexFlat:
                gpu_index = new GpuIndexFlat(
                    resources, device,
                    static_cast<IndexFlat*>(cpu_index));
                break;

            case Index::IndexIVFFlat:
                gpu_index = new GpuIndexIVFFlat(
                    resources, device,
                    static_cast<IndexIVFFlat*>(cpu_index));
                break;

            // ... 其他索引类型
        }

        // 复制数据
        gpu_index->copyFrom(cpu_index);

        return gpu_index;
    }

    // 将GPU索引复制回CPU
    static void copyGpuToCpu(
            Index* gpu_index,
            Index* cpu_index) {

        gpu_index->copyTo(cpu_index);
    }
};
```

### 6.2 使用示例

```cpp
void gpu_cloner_example() {
    // 1. 创建CPU索引
    IndexFlatL2 cpu_index(d);
    cpu_index.add(n, xb);

    // 2. 获取GPU资源
    auto res = GpuResources::getResources();

    // 3. 复制到GPU
    Index* gpu_index = GpuCloner::copyCpuToGpu(&cpu_index, res);

    // 4. GPU搜索
    gpu_index->search(nq, xq, k, distances, labels);

    // 5. 可选：复制回CPU
    // GpuCloner::copyGpuToCpu(gpu_index, &cpu_index);

    delete gpu_index;
}
```

---

## 7. ROCm支持

### 7.1 HIPify转换

```cpp
// Faiss使用hipify工具将CUDA代码转换为ROCm
// faiss/gpu/hipify.sh

# CUDA代码
__global__ void kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] *= 2.0f;
}

// 转换后的HIP代码（概念）
__global__ void kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * hipBlockDim_x;
    data[idx] *= 2.0f;
}
```

### 7.2 通用代码

```cpp
// 使用条件编译同时支持CUDA和ROCm
#ifdef __CUDACC__
    #define DEVICE_CALLABLE __host__ __device__
#else
    #define DEVICE_CALLABLE __host__ __device__
#endif

DEVICE_CALLABLE float distance(float a, float b) {
    return fabs(a - b);
}
```

---

## 8. 性能优化

### 8.1 异步执行

```cpp
void async_search() {
    Stream* stream = res->getDefaultStream(device);

    // 异步复制
    float* x_device;
    cudaMallocAsync(&x_device, n * d * sizeof(float), *stream);
    cudaMemcpyAsync(x_device, x, n * d * sizeof(float),
                    cudaMemcpyHostToDevice, *stream);

    // 异步搜索
    search_kernel<<<grid, block, 0, stream->getStream()>>>(
        data_->get(), data_->getNumVecs(), d,
        x_device, n, k,
        distances_device, labels_device);

    // 异步复制回
    cudaMemcpyAsync(distances, distances_device,
                    n * k * sizeof(float),
                    cudaMemcpyDeviceToHost, *stream);
    cudaMemcpyAsync(labels, labels_device,
                    n * k * sizeof(idx_t),
                    cudaMemcpyDeviceToHost, *stream);

    // 同步（如果需要）
    stream->wait();
}
```

### 8.2 批处理

```cpp
// 批量处理查询以提高GPU利用率
void batch_search(
        GpuIndexFlat* gpu_index,
        size_t batch_size,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) {

    size_t n_processed = 0;

    while (n_processed < batch_size) {
        size_t n_current = std::min(batch_size - n_processed,
                                     size_t(10000));

        gpu_index->search(n_current, x + n_processed * d,
                         k, distances + n_processed * k,
                         labels + n_processed * k);

        n_processed += n_current;
    }
}
```

---

## 10. GPU底层实现详解

### 10.1 GpuResources完整结构

```cpp
// faiss/gpu/GpuResources.h
namespace faiss {
namespace gpu {

// 分配类型：用于跟踪不同类型的GPU内存分配
enum AllocType {
    Other = 0,                              // 其他/未分类
    FlatData = 1,                           // GpuIndexFlat的主数据存储
    IVFLists = 2,                           // GpuIndexIVF*的IVF列表
    Quantizer = 3,                          // 量化器(PQ, SQ)字典
    QuantizerPrecomputedCodes = 4,          // IVFPQ的预计算码
    TemporaryMemoryBuffer = 10,             // 临时内存缓冲区
    TemporaryMemoryOverflow = 11,           // 溢出的临时内存
};

// 内存空间类型
enum MemorySpace {
    Temporary = 0,  // 临时设备内存（退出调用后释放）
    Device = 1,     // 标准GPU设备内存（cudaMalloc/cudaFree）
    Unified = 2,    // 统一CPU/GPU内存（cudaMallocManaged）
};

// GPU资源基类
class GpuResources {
   public:
    virtual ~GpuResources();

    // 获取默认流
    virtual cudaStream_t getDefaultStream(int device) = 0;

    // 获取内存空间
    virtual MemorySpace getMemorySpace(int device) = 0;

    // 分配临时内存
    virtual void* allocMemory(
            int device,
            AllocType allocType,
            size_t size,
            Stream* stream) = 0;

    // 释放内存
    virtual void freeMemory(
            int device,
            AllocType allocType,
            void* p,
            Stream* stream) = 0;

    // 获取cublas句柄
    virtual cublasHandle_t getCublasHandle(int device) = 0;

    // 异步复制CPU到GPU
    virtual void memcpyAsync(
            int device,
            void* dest,
            const void* src,
            size_t size,
            cudaMemcpyKind kind,
            cudaStream_t stream) = 0;

    // 同步流
    virtual void syncStream(int device, cudaStream_t stream) = 0;
};

} // namespace gpu
} // namespace faiss
```

### 10.2 StandardGpuResources实现

```cpp
// faiss/gpu/StandardGpuResources.h
namespace faiss {
namespace gpu {

// 标准GPU资源实现
class StandardGpuResources : public GpuResources {
   public:
    struct Config {
        // 临时内存大小（默认1.5GB）
        size_t tempMemSize = 0;

        // 是否使用固定内存（pinned memory）
        bool pinnedMemory = true;

        // 使用的GPU设备
        std::vector<int> devices;

        // 是否为每个GPU创建默认流
        bool getDefaultStream = true;
    };

    StandardGpuResources();
    explicit StandardGpuResources(const Config& config);

    // 实现GpuResources接口
    cudaStream_t getDefaultStream(int device) override;

    MemorySpace getMemorySpace(int device) override;

    void* allocMemory(
            int device,
            AllocType allocType,
            size_t size,
            Stream* stream) override;

    void freeMemory(
            int device,
            AllocType allocType,
            void* p,
            Stream* stream) override;

    cublasHandle_t getCublasHandle(int device) override;

    void memcpyAsync(
            int device,
            void* dest,
            const void* src,
            size_t size,
            cudaMemcpyKind kind,
            cudaStream_t stream) override;

    void syncStream(int device, cudaStream_t stream) override;

    // 工厂方法
    static std::shared_ptr<GpuResources> getResources(
            const Config& config = Config());

   private:
    struct DeviceProps {
        int device;
        cudaStream_t defaultStream;
        cublasHandle_t cublasHandle;
        std::shared_ptr<void> memory;

        DeviceProps(int device, const Config& config);
        ~DeviceProps();
    };

    std::vector<DeviceProps> devices_;
    Config config_;
};

} // namespace gpu
} // namespace faiss
```

### 10.3 GpuIndex完整结构

```cpp
// faiss/gpu/GpuIndex.h
namespace faiss {
namespace gpu {

struct GpuIndexConfig {
    int device = 0;                        // GPU设备ID
    MemorySpace memorySpace = MemorySpace::Device;

    // 是否使用cuVS（NVIDIA优化的向量搜索库）
#if defined USE_NVIDIA_CUVS
    bool use_cuvs = true;
#else
    bool use_cuvs = false;
#endif
};

class GpuIndex : public faiss::Index {
   public:
    GpuIndex(
            std::shared_ptr<GpuResources> resources,
            int dims,
            faiss::MetricType metric,
            float metricArg,
            GpuIndexConfig config);

    virtual ~GpuIndex();

    // 获取GPU设备ID
    int getDevice() const;

    // 获取GpuResources
    std::shared_ptr<GpuResources> getResources();

    // 设置分页最小大小（MiB）
    void setMinPagingSize(size_t size);
    size_t getMinPagingSize() const;

    // 添加向量（支持CPU/GPU数据源）
    void add(idx_t n, const float* x) override;
    void add_with_ids(idx_t n, const float* x, const idx_t* ids) override;

    // 分配（支持CPU/GPU数据源）
    void assign(idx_t n, const float* x, idx_t* labels, idx_t k = 1)
            const override;

    // 搜索（支持CPU/GPU数据源）
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

    // 恢复训练
    void reverse_train(idx_t n, const float* x) override;

   protected:
    std::shared_ptr<GpuResources> resources_;
    int device_;
    GpuIndexConfig config_;
    size_t minPagingSize_ = 0;

    // 子类实现
    virtual void addInternal_(idx_t n, const float* x) = 0;
    virtual void searchInternal_(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) = 0;
};

} // namespace gpu
} // namespace faiss
```

### 10.4 GpuIndexFlat完整实现

```cpp
// faiss/gpu/GpuIndexFlat.h
namespace faiss {
namespace gpu {

struct GpuIndexFlatConfig : public GpuIndexConfig {
    // 是否预先计算向量范数（内积搜索用）
    bool useFloat16 = false;

    // 是否使用半精度浮点
    bool useFloat16Cosine = false;

    // 是否存储转置数据（提高缓存效率）
    bool storeTransposed = true;
};

class GpuIndexFlat : public GpuIndex {
   public:
    GpuIndexFlat(
            std::shared_ptr<GpuResources> resources,
            int dims,
            faiss::MetricType metric,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    ~GpuIndexFlat() override;

    // 获取向量数量
    idx_t getVectorCount() const {
        return numVecs_;
    }

    // 获取原始数据指针
    const float* getGpuData() const {
        return vecs_.get();
    }

   protected:
    void addInternal_(idx_t n, const float* x) override;
    void searchInternal_(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) override;

   private:
    // GPU上的向量数据
    DeviceVector<float> vecs_;        // nb × d
    DeviceVector<float> vecsNorms_;   // nb（范数，IP搜索用）

    idx_t numVecs_ = 0;
    GpuIndexFlatConfig config_;
};

// CUDA内核：L2距离搜索
template <typename T>
__global__ void l2DistanceKernel(
        const T* database,    // nb × d
        const T* queries,     // nq × d
        int nb,
        int nq,
        int d,
        T* distances) {       // nq × nb

    int queryIdx = blockIdx.x;
    int dbIdx = threadIdx.x;

    if (queryIdx >= nq || dbIdx >= nb) {
        return;
    }

    const T* query = &queries[queryIdx * d];
    const T* vec = &database[dbIdx * d];

    // 计算L2距离
    T dis = 0;
    for (int i = 0; i < d; ++i) {
        T diff = query[i] - vec[i];
        dis += diff * diff;
    }

    distances[queryIdx * nb + dbIdx] = dis;
}

// CUDA内核：内积搜索
template <typename T>
__global__ void innerProductKernel(
        const T* database,    // nb × d
        const T* queries,     // nq × d
        int nb,
        int nq,
        int d,
        const T* norms,      // nb
        T* distances) {       // nq × nb

    int queryIdx = blockIdx.x;
    int dbIdx = threadIdx.x;

    if (queryIdx >= nq || dbIdx >= nb) {
        return;
    }

    const T* query = &queries[queryIdx * d];
    const T* vec = &database[dbIdx * d];

    // 计算内积
    T ip = 0;
    for (int i = 0; i < d; ++i) {
        ip += query[i] * vec[i];
    }

    // 距离 = ||q||^2 + ||v||^2 - 2*<q,v>
    T queryNorm = 0;  // 应该预先计算
    for (int i = 0; i < d; ++i) {
        queryNorm += query[i] * query[i];
    }

    distances[queryIdx * nb + dbIdx] = queryNorm + norms[dbIdx] - 2 * ip;
}

} // namespace gpu
} // namespace faiss
```

### 10.5 GpuIndexIVFFlat完整实现

```cpp
// faiss/gpu/GpuIndexIVFFlat.h
namespace faiss {
namespace gpu {

struct GpuIndexIVFFlatConfig : public GpuIndexConfig {
    // IVF列表
    size_t nlist;

    // 虚拟列表大小（避免频繁的GPU内存分配）
    size_t maxListLength = 0;

    // 热点IVF列表缓存
    bool cacheListData = true;
    size_t numCacheLists = 4;
};

class GpuIndexIVFFlat : public GpuIndex {
   public:
    GpuIndexIVFFlat(
            std::shared_ptr<GpuResources> resources,
            GpuIndexIVFFlatConfig config);

    // 设置粗量化器
    void setQuantizer(Index* quantizer);

    // 添加向量
    void addImpl_(idx_t n, const float* x) override;

    // 搜索
    void searchImpl_(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

   private:
    // 粗量化器（可以是CPU或GPU）
    std::shared_ptr<Index> quantizer_;

    // IVF列表
    struct IVFList {
        DeviceVector<int> ids;          // 向量ID
        DeviceVector<float> vecs;       // 向量数据
        int numVecs = 0;
    };

    std::vector<IVFList> lists_;
    size_t nlist_;
    size_t maxListLength_;

    // 粗量化器搜索
    void searchCoarse_(
            idx_t n,
            const float* x,
            idx_t nprobe,
            idx_t* coarseLabels) const;
};

// CUDA内核：IVF搜索
template <typename T>
__global__ void ivfSearchKernel(
        const float* queries,         // nq × d
        idx_t* coarseLabels,          // nq × nprobe
        int nq,
        int nprobe,
        int k,
        int d,
        const float* __restrict__ listData,  // 预处理的IVF列表
        const idx_t* __restrict__ listIds,    // 预处理的IVF ID
        const int* __restrict__ listLengths,   // 每个列表长度
        float* outDistances,
        idx_t* outLabels) {

    int queryIdx = blockIdx.x;
    int probeIdx = blockIdx.y;

    if (queryIdx >= nq || probeIdx >= nprobe) {
        return;
    }

    int listNo = coarseLabels[queryIdx * nprobe + probeIdx];
    if (listNo < 0) {
        return;
    }

    int listLength = listLengths[listNo];
    const float* listVecs = &listData[listNo * maxListLength * d];
    const idx_t* listIds = &listIds[listNo * maxListLength];

    // 在列表中搜索
    for (int i = threadIdx.x; i < listLength; i += blockDim.x) {
        const float* vec = &listVecs[i * d];
        idx_t id = listIds[i];

        // 计算距离
        float dis = 0;
        for (int j = 0; j < d; ++j) {
            float diff = queries[queryIdx * d + j] - vec[j];
            dis += diff * diff;
        }

        // 更新top-k（使用共享内存）
        // ...（省略详细实现）
    }
}

} // namespace gpu
} // namespace faiss
```

### 10.6 GpuCloner实现

```cpp
// faiss/gpu/GpuCloner.h
namespace faiss {
namespace gpu {

// 将CPU索引复制到GPU
class GpuCloner {
   public:
    // 复制CPU索引到GPU
    static Index* copyCpuToGpu(
            const Index* cpuIndex,
            std::shared_ptr<GpuResources> resources,
            int device = 0);

    // 复制GPU索引回CPU
    static Index* copyGpuToCpu(
            const Index* gpuIndex,
            const Index* cpuIndex);

    // 支持的索引类型检查
    static bool isSupportedGpuIndex(const std::string& index_type);
};

// 使用示例
void gpu_cloner_example() {
    // 1. 创建CPU索引
    IndexFlatL2 cpu_index(d);
    cpu_index.add(nb, xb);

    // 2. 获取GPU资源
    auto res = StandardGpuResources::getResources();

    // 3. 复制到GPU
    Index* gpu_index = GpuCloner::copyCpuToGpu(&cpu_index, res, 0);

    // 4. GPU搜索
    gpu_index->search(nq, xq, k, distances, labels);

    // 5. 可选：复制回CPU
    // Index* cpu_copy = GpuCloner::copyGpuToCpu(gpu_index, &cpu_index);

    delete gpu_index;
}

} // namespace gpu
} // namespace faiss
```

### 10.7 Stream和异步执行

```cpp
// faiss/gpu/Stream.h
namespace faiss {
namespace gpu {

// CUDA流包装器
class Stream {
   public:
    explicit Stream(cudaStream_t stream = nullptr);
    ~Stream();

    // 获取原生CUDA流
    cudaStream_t getStream() const {
        return stream_;
    }

    // 异步复制
    void copyAsync(
            void* dest,
            const void* src,
            size_t size,
            cudaMemcpyKind kind);

    // 同步
    void wait();

    // 是否为空流
    bool isNull() const {
        return stream_ == nullptr;
    }

   private:
    cudaStream_t stream_;
    bool ownStream_;  // 是否拥有流（需要销毁）
};

// 异步搜索示例
void async_search_example(GpuIndexFlat* index) {
    auto res = index->getResources();
    int device = index->getDevice();
    Stream* stream = new Stream(res->getDefaultStream(device));

    // 分配GPU内存
    float* x_device;
    float* d_device;
    idx_t* l_device;

    cudaMallocAsync(&x_device, nq * d * sizeof(float), stream->getStream());
    cudaMallocAsync(&d_device, nq * k * sizeof(float), stream->getStream());
    cudaMallocAsync(&l_device, nq * k * sizeof(idx_t), stream->getStream());

    // 异步复制查询
    cudaMemcpyAsync(x_device, xq, nq * d * sizeof(float),
                    cudaMemcpyHostToDevice, stream->getStream());

    // 异步搜索
    index->search(nq, x_device, k, d_device, l_device);

    // 异步复制结果
    cudaMemcpyAsync(distances, d_device, nq * k * sizeof(float),
                    cudaMemcpyDeviceToHost, stream->getStream());
    cudaMemcpyAsync(labels, l_device, nq * k * sizeof(idx_t),
                    cudaMemcpyDeviceToHost, stream->getStream());

    // 同步等待
    stream->wait();

    // 清理
    cudaFreeAsync(x_device, stream->getStream());
    cudaFreeAsync(d_device, stream->getStream());
    cudaFreeAsync(l_device, stream->getStream());

    delete stream;
}

} // namespace gpu
} // namespace faiss
```

### 10.8 cuVS集成

```cpp
// faiss/gpu/impl/GpuIndexIVF_cuvs.h (概念实现)
namespace faiss {
namespace gpu {

// NVIDIA cuVS（CUDA Vector Search）集成
class GpuIndexIVFCuVS : public GpuIndexIVFFlat {
   public:
    GpuIndexIVFCuVS(
            std::shared_ptr<GpuResources> resources,
            GpuIndexIVFFlatConfig config)
        : GpuIndexIVFFlat(resources, config) {

        // 检查cuVS可用性
#if defined USE_NVIDIA_CUVS
        // 初始化cuVS资源
        raftResources_ = std::make_shared<raft::resources::Resource>();
        deviceMemoryResource_ = std::make_shared<rmm::mr::DeviceMemoryResource>();
#else
        FAISS_THROW_MSG("cuVS support not enabled");
#endif
    }

    void searchImpl_(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {

#if defined USE_NVIDIA_CUVS
        // 使用cuVS的IVF搜索
        if (config_.use_cuvs) {
            search_with_cuvs_(n, x, k, distances, labels, params);
        } else {
            GpuIndexIVFFlat::searchImpl_(n, x, k, distances, labels, params);
        }
#endif
    }

   private:
#if defined USE_NVIDIA_CUVS
    std::shared_ptr<raft::resources::Resource> raftResources_;
    std::shared_ptr<rmm::mr::DeviceMemoryResource> deviceMemoryResource_;

    void search_with_cuvs_(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const;

    // cuVS IVF索引（RAFT管理的GPU数据结构）
    cuvs::IVFIndex cuvsIndex_;
#endif
};

} // namespace gpu
} // namespace faiss
```

---

## 11. 第12天总结

### 关键概念

1. **GPU资源管理**：GpuResources、Stream、MemorySpace
2. **GPU索引**：GpuIndexFlat、GpuIndexIVFFlat等
3. **CUDA内核**：并行距离计算和top-k选择
4. **CPU-GPU交互**：GpuCloner实现双向复制
5. **ROCm支持**：hipify转换和通用代码

### 性能对比

| 操作 | CPU | GPU | 加速比 |
|------|-----|-----|--------|
| 距离计算 (1M) | ~100ms | ~10ms | 10x |
| IVF搜索 | ~200ms | ~20ms | 10x |
| 大规模 (100M) | ~10s | ~1s | 10x |

### 下一步

第13天将学习**复合索引与高级特性**。

---

## 练习题

1. 实现简单的CUDA L2距离内核
2. 使用GpuCloner在CPU和GPU间传输索引
3. 实现异步GPU搜索
4. 比较CPU和GPU的性能

## 扩展阅读

- faiss/gpu/GpuIndex.h - GPU索引基类
- faiss/gpu/GpuIndexFlat.h - GPU Flat索引
- faiss/gpu/GpuResources.h - GPU资源管理
- [CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [ROCm文档](https://rocm.docs.amd.com/)
