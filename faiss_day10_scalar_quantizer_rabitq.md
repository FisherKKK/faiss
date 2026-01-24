# Faiss深度课程 - 第10天：标量量化与RaBitQ

## 课程目标

深入理解标量量化（Scalar Quantization）和RaBitQ（Random-Bounded Quantization），这是Faiss中重要的有损压缩技术。

---

## 1. 标量量化概述

### 1.1 基本思想

标量量化独立地量化向量的每个分量：

```cpp
// 原始向量
float x[128] = {x0, x1, x2, ..., x127};

// 8位标量量化
uint8_t q[128];
for (int i = 0; i < 128; i++) {
    q[i] = round(x[i] / scale + offset);
}

// 压缩比：4字节 → 1字节
```

### 1.2 ScalarQuantizer类型

```cpp
// faiss/impl/ScalarQuantizer.h
struct ScalarQuantizer : Quantizer {
    enum QuantizerType {
        QT_8bit,             // 8位，每维独立范围
        QT_4bit,             // 4位，每维独立范围
        QT_8bit_uniform,     // 8位，共享范围
        QT_4bit_uniform,     // 4位，共享范围
        QT_fp16,             // 半精度浮点
        QT_8bit_direct,      // 8位，快速索引
        QT_6bit,             // 6位
        QT_bf16,             // bfloat16
        QT_8bit_direct_signed, // 8位有符号
    };

    QuantizerType qtype;

    // 范围统计方式
    enum RangeStat {
        RS_minmax,    // [min - rs*(max-min), max + rs*(max-min)]
        RS_meanstd,   // [mean - std * rs, mean + std * rs]
        RS_quantiles, // [Q(rs), Q(1-rs)]
        RS_optim,     // 优化重构误差
    };

    RangeStat rangestat;
    float rangestat_arg;

    size_t bits;              // 每个编码的位数
    std::vector<float> trained; // 训练参数
};
```

---

## 2. 标量量化训练

### 2.1 计算量化范围

```cpp
void ScalarQuantizer::train(size_t n, const float* x) {
    std::vector<float> vmin(d), vmax(d);

    switch (rangestat) {
        case RS_minmax:
            // 找到每维的最小最大值
            for (int dim = 0; dim < d; dim++) {
                vmin[dim] = HUGE_VAL;
                vmax[dim] = -HUGE_VAL;

                for (size_t i = 0; i < n; i++) {
                    float val = x[i * d + dim];
                    vmin[dim] = std::min(vmin[dim], val);
                    vmax[dim] = std::max(vmax[dim], val);
                }

                // 扩展范围
                float delta = vmax[dim] - vmin[dim];
                vmin[dim] -= rangestat_arg * delta;
                vmax[dim] += rangestat_arg * delta;
            }
            break;

        case RS_meanstd:
            // 使用均值和标准差
            for (int dim = 0; dim < d; dim++) {
                float mean = 0.0f;
                for (size_t i = 0; i < n; i++) {
                    mean += x[i * d + dim];
                }
                mean /= n;

                float var = 0.0f;
                for (size_t i = 0; i < n; i++) {
                    float diff = x[i * d + dim] - mean;
                    var += diff * diff;
                }
                float std = sqrt(var / n);

                vmin[dim] = mean - std * rangestat_arg;
                vmax[dim] = mean + std * rangestat_arg;
            }
            break;

        case RS_quantiles:
            // 计算分位数
            for (int dim = 0; dim < d; dim++) {
                std::vector<float> values;
                for (size_t i = 0; i < n; i++) {
                    values.push_back(x[i * d + dim]);
                }
                std::sort(values.begin(), values.end());

                size_t q1 = (size_t)(rangestat_arg * n);
                size_t q2 = (size_t)((1 - rangestat_arg) * n);

                vmin[dim] = values[q1];
                vmax[dim] = values[q2];
            }
            break;
    }

    // 存储训练参数
    trained.resize(2 * d);
    for (int dim = 0; dim < d; dim++) {
        trained[dim * 2 + 0] = vmin[dim];
        trained[dim * 2 + 1] = vmax[dim];
    }
}
```

### 2.2 量化编码

```cpp
void ScalarQuantizer::compute_codes(
        const float* x,
        uint8_t* codes,
        size_t n) const {

    if (qtype == QT_8bit) {
        for (size_t i = 0; i < n; i++) {
            for (int dim = 0; dim < d; dim++) {
                float vmin = trained[dim * 2 + 0];
                float vmax = trained[dim * 2 + 1];

                float val = x[i * d + dim];
                val = std::max(vmin, std::min(vmax, val));

                // 线性量化到[0, 255]
                uint8_t q = (uint8_t)((val - vmin) / (vmax - vmin) * 255);
                codes[i * d + dim] = q;
            }
        }
    } else if (qtype == QT_4bit) {
        for (size_t i = 0; i < n; i++) {
            for (int dim = 0; dim < d; dim++) {
                float vmin = trained[dim * 2 + 0];
                float vmax = trained[dim * 2 + 1];

                float val = x[i * d + dim];
                val = std::max(vmin, std::min(vmax, val));

                // 线性量化到[0, 15]
                uint8_t q = (uint8_t)((val - vmin) / (vmax - vmin) * 15);

                // 打包：两个4位编码在一个字节
                size_t idx = (i * d + dim) / 2;
                int shift = ((i * d + dim) % 2) * 4;
                codes[idx] = (codes[idx] & ~(0xF << shift)) | (q << shift);
            }
        }
    }
}
```

### 2.3 解码

```cpp
void ScalarQuantizer::decode(
        const uint8_t* codes,
        float* x,
        size_t n) const {

    if (qtype == QT_8bit) {
        for (size_t i = 0; i < n; i++) {
            for (int dim = 0; dim < d; dim++) {
                float vmin = trained[dim * 2 + 0];
                float vmax = trained[dim * 2 + 1];

                uint8_t q = codes[i * d + dim];
                x[i * d + dim] = vmin + (vmax - vmin) * (q / 255.0f);
            }
        }
    } else if (qtype == QT_4bit) {
        for (size_t i = 0; i < n; i++) {
            for (int dim = 0; dim < d; dim++) {
                float vmin = trained[dim * 2 + 0];
                float vmax = trained[dim * 2 + 1];

                size_t idx = (i * d + dim) / 2;
                int shift = ((i * d + dim) % 2) * 4;
                uint8_t q = (codes[idx] >> shift) & 0xF;

                x[i * d + dim] = vmin + (vmax - vmin) * (q / 15.0f);
            }
        }
    }
}
```

---

## 3. 距离计算

### 3.1 SIMD优化的8位距离

```cpp
// 8位标量量化的SIMD距离计算
void SQ8_distance_simd(
        const float* x,         // 查询向量
        const uint8_t* codes,   // 量化编码
        const float* vmin,      // 最小值
        const float* vmax,      // 最大值
        size_t d,
        float& dis) {

    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    // 处理8的倍数
    for (; i + 8 <= d; i += 8) {
        // 加载8个编码
        __m256i codes = _mm256_loadu_si256((__m256i*)(codes + i));

        // 解码：[0, 255] → float
        __m256 vmin_v = _mm256_loadu_ps(vmin + i);
        __m256 vmax_v = _mm256_loadu_ps(vmax + i);

        // 扩展为float
        __m256i zero = _mm256_setzero_si256();
        __m256i codes_lo = _mm256_unpacklo_epi8(codes, zero);
        __m256i codes_hi = _mm256_unpackhi_epi8(codes, zero);
        __m256 f_lo = _mm256_cvtepi32_ps(codes_lo);
        __m256 f_hi = _mm256_cvtepi32_ps(codes_hi);

        // 归一化到[vmin, vmax]
        __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
        __m256 range = _mm256_sub_ps(vmax_v, vmin_v);

        f_lo = _mm256_fmadd_ps(f_lo, scale, vmin_v);
        f_hi = _mm256_fmadd_ps(f_hi, scale, vmin_v);

        // 合并低128位和高128位
        __m256 decoded = _mm256_permute2f128_ps(f_lo, f_hi, 0x20);

        // 加载查询
        __m256 xv = _mm256_loadu_ps(x + i);

        // 计算距离
        __m256 diff = _mm256_sub_ps(xv, decoded);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // 水平求和
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    dis = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
          tmp[4] + tmp[5] + tmp[6] + tmp[7];

    // 处理剩余元素
    for (; i < d; i++) {
        float decoded = vmin[i] + (vmax[i] - vmin[i]) * codes[i] / 255.0f;
        float diff = x[i] - decoded;
        dis += diff * diff;
    }
}
```

---

## 4. RaBitQ

### 4.1 概述

RaBitQ（Random-Bounded Quantization）提供了理论误差界的随机量化：

```cpp
// RaBitQ特点：
// 1. 随机投影
// 2. 理论误差界
// 3. 两阶段搜索（1位过滤 + 多位精化）

// 第1阶段：1位过滤（超快速）
uint8_t bit1 = compute_1bit(x);

// 第2阶段：k位精化（仅在候选上）
uint8_t bitk = compute_kbit(x, candidates);
```

### 4.2 1位量化

```cpp
// 1位量化：随机投影
uint8_t rabitq_1bit(const float* x, size_t d, const float* r) {
    // r是随机向量，每维±1

    float ip = 0.0f;
    for (size_t i = 0; i < d; i++) {
        ip += x[i] * r[i];
    }

    // 1位编码：符号位
    return ip >= 0 ? 1 : 0;
}

// 批量1位编码
void rabitq_encode_1bit(
        const float* x,
        size_t n, size_t d,
        uint8_t* codes) {  // 压缩：n × d / 8字节

    // 生成随机投影矩阵
    float* R = new float[n * d];
    generate_random_matrix(R, n, d);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            uint8_t bit = rabitq_1bit(x + i * d, d, R + j * d);

            // 打包到字节
            size_t byte_idx = i * d / 8;
            int bit_idx = 7 - (j % 8);
            if (bit) {
                codes[byte_idx] |= (1 << bit_idx);
            }
        }
    }

    delete[] R;
}
```

### 4.3 多位量化

```cpp
// k位量化
uint8_t rabitq_kbit(const float* x, size_t d, size_t k) {
    // 1. 随机投影到k维
    float* rp = new float[k];
    random_projection(x, d, rp, k);

    // 2. 标量量化
    for (size_t i = 0; i < k; i++) {
        rp[i] = std::max(-1.0f, std::min(1.0f, rp[i]));
        rp[i] = (rp[i] + 1.0f) / 2.0f;  // [0, 1]
    }

    // 3. 编码为k位
    uint8_t code = 0;
    for (size_t i = 0; i < k; i++) {
        uint8_t bit = rp[i] > 0.5f ? 1 : 0;
        code |= (bit << i);
    }

    delete[] rp;
    return code;
}
```

### 4.4 两阶段搜索

```cpp
void IndexIVFRaBitQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    // 阶段1：1位过滤
    uint8_t* query_1bit = new uint8_t[n * d];
    rabitq_encode_1bit(x, n, d, query_1bit);

    // 使用1位汉明距离快速过滤
    for (idx_t q = 0; q < n; q++) {
        std::vector<idx_t> candidates;

        for (idx_t list_no = 0; list_no < nlist; list_no++) {
            const uint8_t* list_codes = invlists->get_codes_1bit(list_no);
            const idx_t* list_ids = invlists->get_ids(list_no);
            size_t list_size = invlists->list_size(list_no);

            // 计算汉明距离
            for (size_t i = 0; i < list_size; i++) {
                int hamming = popcount(
                    query_1bit[q * d / 8],
                    list_codes[i * d / 8]);

                // 阈值过滤
                if (hamming < threshold) {
                    candidates.push_back(list_ids[i]);
                }
            }
        }

        // 阶段2：k位精化
        uint8_t query_kbit = rabitq_kbit(x + q * d, d, kbits);

        for (idx_t id : candidates) {
            const uint8_t* code = get_code_kbit(id);

            // 精确距离计算
            float dis = compute_kbit_distance(query_kbit, code);

            // 更新堆
            if (CMax<float, idx_t>::cmp(dis, distances[q * k])) {
                heap_replace_top<CMax<float, idx_t>>(
                    k, distances + q * k, labels + q * k,
                    dis, id);
            }
        }
    }

    delete[] query_1bit;
}
```

---

## 5. IndexScalarQuantizer

### 5.1 结构

```cpp
struct IndexScalarQuantizer : IndexIVF {
    ScalarQuantizer sq;

    IndexScalarQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            ScalarQuantizer::QuantizerType qtype)
        : IndexIVF(quantizer, d, nlist,
                  sq.code_size, METRIC_L2),
          sq(d, qtype) {}
};
```

### 5.2 使用示例

```cpp
void scalar_quantizer_example() {
    int d = 128;
    int nlist = 100;

    // 粗量化器
    IndexFlatL2 quantizer(d);

    // 8位标量量化
    IndexScalarQuantizer index(
        &quantizer, d, nlist,
        ScalarQuantizer::QT_8bit);

    // 训练
    index.train(n, xb);

    // 添加
    index.add(n, xb);

    // 搜索
    index.search(nq, xq, k, distances, labels);
}
```

---

## 7. RaBitQ底层实现详解

### 7.1 RaBitQuantizer完整结构

```cpp
// faiss/impl/RaBitQuantizer.h
// RaBitQ的参考实现，基于论文：
// "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
//  for Approximate Nearest Neighbor Search"
struct RaBitQuantizer : Quantizer {
    // 所有RaBitQ操作都相对于一个质心
    // 质心需要外部提供（nullptr表示全零质心）
    float* centroid = nullptr;

    // 度量类型（用于存储额外的fp32常量）
    MetricType metric_type = MetricType::METRIC_L2;

    // 每维位数（1-9）
    // nb_bits = 1: 标准1位RaBitQ（仅符号位）
    // nb_bits = 2-9: 多位RaBitQ（1符号位 + ex_bits个额外位）
    size_t nb_bits = 1;

    RaBitQuantizer(
            size_t d = 0,
            MetricType metric = MetricType::METRIC_L2,
            size_t nb_bits = 1);

    // 计算编码大小
    // nb_bits=1:  (d+7)/8 + 8 字节 (1位编码 + 基础因子)
    // nb_bits>1:  (d+7)/8 + 8 + d*ex_bits/8 + 8 字节
    //            (1位编码 + 基础因子 + 额外位编码 + 额外因子)
    size_t compute_code_size(size_t d, size_t num_bits) const;

    void train(size_t n, const float* x) override;
    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;
    void decode(const uint8_t* codes, float* x, size_t n) const override;

    // 核心编码/解码函数
    void compute_codes_core(
            const float* x,
            uint8_t* codes,
            size_t n,
            const float* centroid_in) const;

    void decode_core(
            const uint8_t* codes,
            float* x,
            size_t n,
            const float* centroid_in) const;

    // 获取距离计算机
    // qb = 0: 不量化查询
    // qb > 0: 使用qb位标量量化查询
    FlatCodesDistanceComputer* get_distance_computer(
            uint8_t qb = 0,
            const float* centroid = nullptr,
            bool centered = false) const;
};
```

### 7.2 RaBitQDistanceComputer

```cpp
// faiss/impl/RaBitQuantizer.h
// RaBitQ距离计算机基类
// 这个中间类提供了两阶段多位搜索的统一接口
struct RaBitQDistanceComputer : FlatCodesDistanceComputer {
    size_t d = 0;
    const float* centroid = nullptr;
    MetricType metric_type = MetricType::METRIC_L2;
    size_t nb_bits = 1;

    // 查询误差因子（用于边界计算）
    float g_error = 0.0f;

    float symmetric_dis(idx_t /*i*/, idx_t /*j*/) override {
        FAISS_THROW_MSG("Not implemented");
    }

    // 计算1位距离估计（快速）
    virtual float distance_to_code_1bit(const uint8_t* code) = 0;

    // 计算完整多位距离（精确）
    virtual float distance_to_code_full(const uint8_t* code) = 0;

    // 从FlatCodesDistanceComputer继承
    // 委托给distance_to_code_full()
    float distance_to_code(const uint8_t* code) final {
        return distance_to_code_full(code);
    }
};
```

### 7.3 SignBitFactors结构

```cpp
// faiss/impl/RaBitQuantizer.h (相关定义)
// RaBitQ编码中存储的基向量因子
// 用于1位RaBitQ的快速距离计算
struct SignBitFactors {
    // L2距离:   ||q - c||^2 = ||q||^2 - 2*<q,c> + ||c||^2
    // IP距离:   -<q,c> = -<q,c>
    // 对于1位编码：<q,c> ≈ sum(signs) * base_factor

    float base_factor;     // 基础因子
    float norm_q_squared;   // ||q||^2（L2距离用）

    // 对于内积度量，解码输出旨在保持IP而非L2
    // 重构的向量可能比预期的L2距离误差更大
    // 但查询与原始向量的内积值可能与查询与重构编码的内积非常接近
};
```

### 7.4 RaBitQ编码过程

```cpp
// faiss/impl/RaBitQuantizer.cpp (简化版)
void RaBitQuantizer::compute_codes_core(
        const float* x,
        uint8_t* codes,
        size_t n,
        const float* centroid_in) const {

    // 1. 计算相对于质心的残差
    std::vector<float> residuals(n * d);
    if (centroid_in) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                residuals[i * d + j] = x[i * d + j] - centroid_in[j];
            }
        }
    } else {
        // 质心为零，直接使用原始向量
        memcpy(residuals.data(), x, n * d * sizeof(float));
    }

    // 2. 1位编码：符号位
    std::vector<uint8_t> sign_bits((n * d + 7) / 8, 0);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            int bit_idx = i * d + j;
            int byte_idx = bit_idx / 8;
            int bit_pos = 7 - (bit_idx % 8);

            // 符号位：正数为1，负数为0
            if (residuals[i * d + j] >= 0) {
                sign_bits[byte_idx] |= (1 << bit_pos);
            }
        }
    }

    // 3. 计算base_factor（所有维度的均值）
    std::vector<float> base_factors(n);
    for (size_t i = 0; i < n; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < d; j++) {
            sum += std::abs(residuals[i * d + j]);
        }
        base_factors[i] = sum / d;
    }

    // 4. 对于多位RaBitQ，计算额外位的编码
    if (nb_bits > 1) {
        size_t ex_bits = nb_bits - 1;  // 额外的位数

        // 量化每个维度到ex_bits
        // ... (省略详细实现)
    }

    // 5. 打包编码
    size_t code_offset = 0;

    // 符号位
    memcpy(codes + code_offset, sign_bits.data(), (n * d + 7) / 8);
    code_offset += (n * d + 7) / 8;

    // base_factors
    memcpy(codes + code_offset, base_factors.data(), n * sizeof(float));
    code_offset += n * sizeof(float);

    // 如果是多位，添加额外编码和因子
    if (nb_bits > 1) {
        // ...
    }
}
```

### 7.5 RaBitQ解码过程

```cpp
// faiss/impl/RaBitQuantizer.cpp (简化版)
void RaBitQuantizer::decode_core(
        const uint8_t* codes,
        float* x,
        size_t n,
        const float* centroid_in) const {

    size_t code_offset = 0;

    // 1. 提取符号位
    size_t sign_bits_size = (n * d + 7) / 8;
    const uint8_t* sign_bits = codes + code_offset;
    code_offset += sign_bits_size;

    // 2. 提取base_factors
    const float* base_factors = (const float*)(codes + code_offset);
    code_offset += n * sizeof(float);

    // 3. 重构向量
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            int bit_idx = i * d + j;
            int byte_idx = bit_idx / 8;
            int bit_pos = 7 - (bit_idx % 8);

            // 获取符号
            int sign = (sign_bits[byte_idx] & (1 << bit_pos)) ? 1 : -1;

            // 重构：sign * base_factor
            x[i * d + j] = sign * base_factors[i];
        }

        // 4. 加上质心
        if (centroid_in) {
            for (size_t j = 0; j < d; j++) {
                x[i * d + j] += centroid_in[j];
            }
        }
    }

    // 注意：解码输出旨在保持内积而非L2距离
    // 重构向量可能与原始向量的L2距离较大
    // 但内积值会非常接近
}
```

### 7.6 两阶段搜索

```cpp
// faiss/impl/RaBitQUtils.h (相关实现)
// 两阶段搜索：1位过滤 + 多位精化
template <typename DC>
void rabitq_two_stage_search(
        const DC& dc,
        const uint8_t* codes,
        size_t n,
        float* distances,
        idx_t* labels,
        size_t k,
        float filter_threshold) {

    // 阶段1：1位快速过滤
    std::vector<size_t> candidates;
    candidates.reserve(n);

    for (size_t i = 0; i < n; i++) {
        // 使用1位编码快速计算距离下界
        float dis_1bit = dc.distance_to_code_1bit(codes + i * dc.code_size);

        // 如果距离下界小于阈值，加入候选集
        if (dis_1bit < filter_threshold) {
            candidates.push_back(i);
        }
    }

    // 阶段2：多位精确计算
    // 在候选集上使用完整的距离计算
    std::priority_queue<std::pair<float, idx_t>> heap;

    for (size_t idx : candidates) {
        float dis_full = dc.distance_to_code_full(codes + idx * dc.code_size);

        if (heap.size() < k || dis_full < heap.top().first) {
            heap.push({dis_full, idx});
            if (heap.size() > k) {
                heap.pop();
            }
        }
    }

    // 提取结果
    for (size_t i = 0; i < k && !heap.empty(); i++) {
        distances[k - 1 - i] = heap.top().first;
        labels[k - 1 - i] = heap.top().second;
        heap.pop();
    }
}
```

### 7.7 1位距离计算

```cpp
// faiss/impl/RaBitQUtils.h (相关实现)
// 快速的1位距离计算（L2距离）
float distance_to_code_1bit_l2(
        const uint8_t* code,
        const float* query_residuals,  // 查询相对于质心的残差
        size_t d,
        float norm_q_squared) {

    const uint8_t* sign_bits = code;
    const float* base_factors = (const float*)(code + (d + 7) / 8);
    float base_factor = base_factors[0];

    // 计算：||q||^2 - 2 * <signs, |q|> * base_factor + d * base_factor^2
    // 其中 <signs, |q|> = sum(sign_i * |q_i|)

    float ip_sign_abs_q = 0.0f;
    for (size_t j = 0; j < d; j++) {
        int byte_idx = j / 8;
        int bit_pos = 7 - (j % 8);
        int sign = (sign_bits[byte_idx] & (1 << bit_pos)) ? 1 : -1;

        ip_sign_abs_q += sign * std::abs(query_residuals[j]);
    }

    float dis = norm_q_squared
                - 2.0f * ip_sign_abs_q * base_factor
                + d * base_factor * base_factor;

    return dis;
}

// 内积距离的1位计算
float distance_to_code_1bit_ip(
        const uint8_t* code,
        const float* query_residuals,
        size_t d) {

    const uint8_t* sign_bits = code;
    const float* base_factors = (const float*)(code + (d + 7) / 8);
    float base_factor = base_factors[0];

    // 计算：-<signs, |q|> * base_factor
    float ip_sign_abs_q = 0.0f;
    for (size_t j = 0; j < d; j++) {
        int byte_idx = j / 8;
        int bit_pos = 7 - (j % 8);
        int sign = (sign_bits[byte_idx] & (1 << bit_pos)) ? 1 : -1;

        ip_sign_abs_q += sign * std::abs(query_residuals[j]);
    }

    return -ip_sign_abs_q * base_factor;
}
```

### 7.8 查询量化

```cpp
// faiss/impl/RaBitQuantizer.cpp (相关实现)
// 对查询向量进行标量量化以加速距离计算
struct QuantizedQueryRaBitQ {
    std::vector<uint8_t> q_signs;     // 查询的符号位
    std::vector<float> q_abs;         // 查询的绝对值
    float norm_q_squared;             // ||q||^2

    QuantizedQueryRaBitQ(const float* query, size_t d) {
        q_signs.resize((d + 7) / 8, 0);
        q_abs.resize(d);
        norm_q_squared = 0.0f;

        for (size_t j = 0; j < d; j++) {
            int byte_idx = j / 8;
            int bit_pos = 7 - (j % 8);

            q_abs[j] = std::abs(query[j]);

            if (query[j] >= 0) {
                q_signs[byte_idx] |= (1 << bit_pos);
            }

            norm_q_squared += query[j] * query[j];
        }
    }
};
```

---

## 8. 第10天总结

### 关键概念

1. **标量量化**：独立量化每个分量
2. **量化范围**：minmax、meanstd、quantiles
3. **打包存储**：4位编码打包为字节
4. **RaBitQ**：随机有界量化
5. **两阶段搜索**：1位过滤 + 多位精化

### 量化方法对比

| 方法 | 压缩比 | 精度 | 速度 |
|------|--------|------|------|
| 8位SQ | 4x | 高 | 快 |
| 4位SQ | 8x | 中 | 快 |
| PQ(M=16) | 64x | 中 | 中 |
| RaBitQ(1+8) | ~32x | 中-高 | 很快 |

### 下一步

第11天将学习**二进制索引**，针对汉明距离优化的索引结构。

---

---

## 9. ScalarQuantizer与RaBitQ源码深度实现

本节深入分析Faiss中标量量化器(ScalarQuantizer)和RaBitQ的核心实现细节，包括SIMD优化的编解码、距离计算、两阶段搜索等关键算法。

### 9.1 ScalarQuantizer编解码器结构

Faiss使用模板特化实现高效的编解码，针对不同位数和SIMD指令集进行优化。

```cpp
// faiss/impl/ScalarQuantizer.cpp
// 8位编解码器（AVX2优化）
struct Codec8bit {
    // 编码单个分量：float ∈ [0,1] → uint8_t ∈ [0,255]
    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            uint8_t* code,
            int i) {
        code[i] = (int)(255 * x);
    }

    // 解码单个分量
    static FAISS_ALWAYS_INLINE float decode_component(
            const uint8_t* code,
            int i) {
        return (code[i] + 0.5f) / 255.0f;  // +0.5用于减少量化误差
    }

#if defined(__AVX2__)
    // SIMD解码：一次解码8个分量
    static FAISS_ALWAYS_INLINE __m256
    decode_8_components(const uint8_t* code, int i) {
        // 加载8个字节
        const uint64_t c8 = *(uint64_t*)(code + i);

        // 扩展为8个32位整数
        const __m128i i8 = _mm_set1_epi64x(c8);
        const __m256i i32 = _mm256_cvtepu8_epi32(i8);

        // 转换为float并归一化
        const __m256 f8 = _mm256_cvtepi32_ps(i32);
        const __m256 half_one_255 = _mm256_set1_ps(0.5f / 255.f);
        const __m256 one_255 = _mm256_set1_ps(1.f / 255.f);

        // fma: f8 * one_255 + half_one_255
        return _mm256_fmadd_ps(f8, one_255, half_one_255);
    }
#endif
};

// 4位编解码器（两个4位编码打包在一个字节）
struct Codec4bit {
    // 编码：float ∈ [0,1] → 4位 ∈ [0,15]
    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            uint8_t* code,
            int i) {
        // 打包：低4位或高4位
        code[i / 2] |= (int)(x * 15.0) << ((i & 1) << 2);
    }

    // 解码
    static FAISS_ALWAYS_INLINE float decode_component(
            const uint8_t* code,
            int i) {
        return (((code[i / 2] >> ((i & 1) << 2)) & 0xf) + 0.5f) / 15.0f;
    }

#if defined(__AVX2__)
    // SIMD解码：一次解码8个4位编码（4个字节）
    static FAISS_ALWAYS_INLINE __m256
    decode_8_components(const uint8_t* code, int i) {
        uint32_t c4 = *(uint32_t*)(code + (i >> 1));
        uint32_t mask = 0x0f0f0f0f;
        uint32_t c4ev = c4 & mask;      // 偶数位置的4位
        uint32_t c4od = (c4 >> 4) & mask; // 奇数位置的4位

        // 交叉组合成8个字节
        __m128i c8 = _mm_unpacklo_epi8(
            _mm_set1_epi32(c4ev), _mm_set1_epi32(c4od));

        // 扩展为32位整数
        __m128i c4lo = _mm_cvtepu8_epi32(c8);
        __m128i c4hi = _mm_cvtepu8_epi32(_mm_srli_si128(c8, 4));

        __m256i i8 = _mm256_castsi128_si256(c4lo);
        i8 = _mm256_insertf128_si256(i8, c4hi, 1);

        // 转换为float
        __m256 f8 = _mm256_cvtepi32_ps(i8);
        __m256 half = _mm256_set1_ps(0.5f);
        f8 = _mm256_add_ps(f8, half);
        __m256 one_15 = _mm256_set1_ps(1.f / 15.f);

        return _mm256_mul_ps(f8, one_15);
    }
#endif
};

// 6位编解码器（特殊打包：4个6位=3字节）
struct Codec6bit {
    // 编码：float ∈ [0,1] → 6位 ∈ [0,63]
    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            uint8_t* code,
            int i) {
        int bits = (int)(x * 63.0);
        code += (i >> 2) * 3;  // 每4个分量3字节

        switch (i & 3) {
            case 0:  // [xxxxxx00 xxxxxxxx xxxxxxxx]
                code[0] |= bits;
                break;
            case 1:  // [xx111111 222222xx xxxxxxxx]
                code[0] |= bits << 6;
                code[1] |= bits >> 2;
                break;
            case 2:  // [xxxxxxxx 22223333 3333xxxx]
                code[1] |= bits << 4;
                code[2] |= bits >> 4;
                break;
            case 3:  // [xxxxxxxx xxxxxxxx xx333333]
                code[2] |= bits << 2;
                break;
        }
    }

    // 解码
    static FAISS_ALWAYS_INLINE float decode_component(
            const uint8_t* code,
            int i) {
        uint8_t bits;
        code += (i >> 2) * 3;

        switch (i & 3) {
            case 0:
                bits = code[0] & 0x3f;
                break;
            case 1:
                bits = code[0] >> 6;
                bits |= (code[1] & 0xf) << 2;
                break;
            case 2:
                bits = code[1] >> 4;
                bits |= (code[2] & 3) << 4;
                break;
            case 3:
                bits = code[2] >> 2;
                break;
        }
        return (bits + 0.5f) / 63.0f;
    }
};
```

### 9.2 ScalarQuantizer训练 - 量化范围计算

```cpp
// faiss/impl/ScalarQuantizer.cpp (简化版)
void ScalarQuantizer::train(size_t n, const float* x) {
    if (qtype == QT_8bit_direct || qtype == QT_8bit_direct_signed) {
        // 直接索引类型，不需要训练
        return;
    }

    std::vector<float> vmin(d), vmax(d);

    // 根据统计类型计算范围
    switch (rangestat) {
        case RS_minmax: {
            // 找到每维的最小最大值
            for (int dim = 0; dim < d; dim++) {
                vmin[dim] = HUGE_VAL;
                vmax[dim] = -HUGE_VAL;

                for (size_t i = 0; i < n; i++) {
                    float val = x[i * d + dim];
                    if (std::isnan(val)) continue;
                    vmin[dim] = std::min(vmin[dim], val);
                    vmax[dim] = std::max(vmax[dim], val);
                }

                // 扩展范围（避免边界溢出）
                float delta = vmax[dim] - vmin[dim];
                vmin[dim] -= rangestat_arg * delta;
                vmax[dim] += rangestat_arg * delta;

                // 处理空范围
                if (vmax[dim] == vmin[dim]) {
                    vmax[dim] += 1;
                    vmin[dim] -= 1;
                }
            }
            break;
        }

        case RS_meanstd: {
            // 使用均值和标准差
            for (int dim = 0; dim < d; dim++) {
                double sum = 0;
                for (size_t i = 0; i < n; i++) {
                    sum += x[i * d + dim];
                }
                float mean = sum / n;

                double var = 0;
                for (size_t i = 0; i < n; i++) {
                    float diff = x[i * d + dim] - mean;
                    var += diff * diff;
                }
                float std = sqrt(var / n);

                vmin[dim] = mean - std * rangestat_arg;
                vmax[dim] = mean + std * rangestat_arg;

                if (vmax[dim] == vmin[dim]) {
                    vmax[dim] += 1;
                    vmin[dim] -= 1;
                }
            }
            break;
        }

        case RS_quantiles: {
            // 计算分位数
            for (int dim = 0; dim < d; dim++) {
                std::vector<float> values;
                values.reserve(n);
                for (size_t i = 0; i < n; i++) {
                    float val = x[i * d + dim];
                    if (!std::isnan(val)) {
                        values.push_back(val);
                    }
                }

                if (values.empty()) {
                    vmin[dim] = -1;
                    vmax[dim] = 1;
                    continue;
                }

                std::sort(values.begin(), values.end());

                size_t q1 = std::min(
                    (size_t)(rangestat_arg * values.size()),
                    values.size() - 1);
                size_t q2 = std::min(
                    (size_t)((1 - rangestat_arg) * values.size()),
                    values.size() - 1);

                vmin[dim] = values[q1];
                vmax[dim] = values[q2];

                if (vmax[dim] == vmin[dim]) {
                    vmax[dim] += 1;
                    vmin[dim] -= 1;
                }
            }
            break;
        }

        case RS_optim: {
            // 优化重构误差（迭代优化）
            // 初始化使用minmax
            for (int dim = 0; dim < d; dim++) {
                vmin[dim] = HUGE_VAL;
                vmax[dim] = -HUGE_VAL;
                for (size_t i = 0; i < n; i++) {
                    float val = x[i * d + dim];
                    vmin[dim] = std::min(vmin[dim], val);
                    vmax[dim] = std::max(vmax[dim], val);
                }
            }

            // 迭代优化（简化版）
            for (int iter = 0; iter < 10; iter++) {
                // ... 优化代码
            }
            break;
        }
    }

    // 存储训练参数
    trained.resize(2 * d);
    for (int dim = 0; dim < d; dim++) {
        trained[dim * 2 + 0] = vmin[dim];
        trained[dim * 2 + 1] = vmax[dim];
    }

    is_trained = true;
}
```

### 9.3 ScalarQuantizer::compute_codes - 编码实现

```cpp
// faiss/impl/ScalarQuantizer.cpp (简化版)
void ScalarQuantizer::compute_codes(
        const float* x,
        uint8_t* codes,
        size_t n) const {

    // 选择合适的量化器
    std::unique_ptr<SQuantizer> quantizer(select_quantizer());

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        // 清零编码
        memset(codes + i * code_size, 0, code_size);
        // 编码向量
        quantizer->encode_vector(x + i * d, codes + i * code_size);
    }
}

// 统一量化器模板
template <class Codec, QuantizerTemplateScaling SCALING>
struct QuantizerTemplate : ScalarQuantizer::SQuantizer {
    const size_t d;
    const float* vmin;  // 每维最小值
    const float* vmax;  // 每维最大值

    QuantizerTemplate(size_t d, const std::vector<float>& trained)
        : d(d), vmin(trained.data()), vmax(trained.data() + d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        if constexpr (SCALING == QuantizerTemplateScaling::UNIFORM) {
            // 统一范围：所有维度共享vmin和vmax
            for (size_t i = 0; i < d; i++) {
                float xi = (x[i] - vmin[0]) / (vmax[0] - vmin[0]);
                xi = std::max(0.0f, std::min(1.0f, xi));
                Codec::encode_component(xi, code, i);
            }
        } else {
            // 非统一：每维独立范围
            for (size_t i = 0; i < d; i++) {
                float vdiff = vmax[i] - vmin[i];
                float xi = 0;
                if (vdiff != 0) {
                    xi = (x[i] - vmin[i]) / vdiff;
                    xi = std::max(0.0f, std::min(1.0f, xi));
                }
                Codec::encode_component(xi, code, i);
            }
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        if constexpr (SCALING == QuantizerTemplateScaling::UNIFORM) {
            for (size_t i = 0; i < d; i++) {
                float xi = Codec::decode_component(code, i);
                x[i] = vmin[0] + xi * (vmax[0] - vmin[0]);
            }
        } else {
            for (size_t i = 0; i < d; i++) {
                float xi = Codec::decode_component(code, i);
                x[i] = vmin[i] + xi * (vmax[i] - vmin[i]);
            }
        }
    }
};
```

### 9.4 RaBitQ编码实现

```cpp
// faiss/impl/RaBitQuantizer.cpp
void RaBitQuantizer::compute_codes_core(
        const float* x,
        uint8_t* codes,
        size_t n,
        const float* centroid_in) const {

    const size_t ex_bits = nb_bits - 1;

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        uint8_t* code = codes + i * code_size;
        memset(code, 0, code_size);

        const float* x_row = x + i * d;

        // 步骤1：计算1位符号编码和基础因子
        std::vector<float> residual(d);

        // 计算向量因子（内积乘数和L2距离）
        SignBitFactorsWithError factors_data =
                rabitq_utils::compute_vector_factors(
                        x_row, d, centroid_in, metric_type, ex_bits > 0);

        // 写入因子
        if (ex_bits == 0) {
            // 1位：只写SignBitFactors (8字节)
            SignBitFactors* base_factors =
                    reinterpret_cast<SignBitFactors*>(
                            code + (d + 7) / 8);
            base_factors->or_minus_c_l2sqr = factors_data.or_minus_c_l2sqr;
            base_factors->dp_multiplier = factors_data.dp_multiplier;
        } else {
            // 多位：写SignBitFactorsWithError (12字节)
            SignBitFactorsWithError* full_factors =
                    reinterpret_cast<SignBitFactorsWithError*>(
                            code + (d + 7) / 8);
            *full_factors = factors_data;
        }

        // 打包符号位
        uint8_t* binary_code = code;
        for (size_t j = 0; j < d; j++) {
            const float centroid_val =
                    (centroid_in == nullptr) ? 0.0f : centroid_in[j];
            const float or_minus_c = x_row[j] - centroid_val;
            residual[j] = or_minus_c;

            // 符号位：正数为1，负数为0
            if (or_minus_c > 0.0f) {
                rabitq_utils::set_bit_standard(binary_code, j);
            }
        }

        // 步骤2：计算额外位（如果nb_bits > 1）
        if (ex_bits > 0) {
            uint8_t* ex_code =
                    code + (d + 7) / 8 + sizeof(SignBitFactorsWithError);
            ExtraBitsFactors* ex_factors =
                    reinterpret_cast<ExtraBitsFactors*>(
                            ex_code + (d * ex_bits + 7) / 8);

            // 量化残差到ex_bits
            rabitq_multibit::quantize_ex_bits(
                    residual.data(), d, nb_bits,
                    ex_code, *ex_factors,
                    metric_type, centroid_in);
        }
    }
}
```

### 9.5 RaBitQ距离计算（1位快速过滤）

```cpp
// faiss/impl/RaBitQuantizer.cpp
float RaBitQDistanceComputerNotQ::distance_to_code_1bit(
        const uint8_t* code) {

    // 分离编码结构
    const uint8_t* binary_data = code;  // 符号位
    size_t ex_bits = nb_bits - 1;

    // 提取基础因子
    const SignBitFactors* base_fac = (ex_bits == 0)
            ? reinterpret_cast<const SignBitFactors*>(code + (d + 7) / 8)
            : reinterpret_cast<const SignBitFactorsWithError*>(
                      code + (d + 7) / 8);

    // 计算内积：dot_qo = <q, o> 其中o是符号向量
    float dot_qo = 0;
    uint64_t sum_q = 0;  // 符号位中1的个数

    for (size_t i = 0; i < d; i++) {
        bool bit = rabitq_utils::extract_bit_standard(binary_data, i);
        if (bit) {
            dot_qo += rotated_q[i];  // 只累加正号对应的值
            sum_q++;
        }
    }

    // 应用查询因子
    // final_dot = <qr - c, or - c>的近似值
    float final_dot =
            query_fac.c1 * dot_qo + query_fac.c2 * sum_q - query_fac.c34;

    // 计算L2距离
    // ||or - qr||^2 = ||or - c||^2 + ||qr - c||^2 - 2*<or - c, qr - c>
    float pre_dist = base_fac->or_minus_c_l2sqr +
                     query_fac.qr_to_c_L2sqr -
                     2 * base_fac->dp_multiplier * final_dot;

    if (metric_type == MetricType::METRIC_L2) {
        return pre_dist;
    } else {
        // 内积度量：2*(or, qr) = ||or - qr||^2 - ||qr||^2 - ||or||^2
        return -0.5f * (pre_dist - query_fac.qr_norm_L2sqr);
    }
}

// 设置查询（预处理查询数据）
void RaBitQDistanceComputerNotQ::set_query(const float* x) {
    q = x;

    // 计算查询到质心的距离
    if (centroid != nullptr) {
        query_fac.qr_to_c_L2sqr = fvec_L2sqr(x, centroid, d);
    } else {
        query_fac.qr_to_c_L2sqr = fvec_norm_L2sqr(x, d);
    }

    // 计算旋转后的查询：qr - c
    rotated_q.resize(d);
    for (size_t i = 0; i < d; i++) {
        rotated_q[i] = x[i] - ((centroid == nullptr) ? 0 : centroid[i]);
    }

    // 计算查询误差因子（用于两阶段搜索的下界计算）
    g_error = std::sqrt(query_fac.qr_to_c_L2sqr);

    // 计算查询因子
    const float inv_d = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));

    float sum_q = 0;
    for (size_t i = 0; i < d; i++) {
        sum_q += rotated_q[i];
    }

    query_fac.c1 = 2 * inv_d;
    query_fac.c2 = 0;
    query_fac.c34 = sum_q * inv_d;

    if (metric_type == MetricType::METRIC_INNER_PRODUCT) {
        query_fac.qr_norm_L2sqr = fvec_norm_L2sqr(x, d);
    }
}
```

### 9.6 RaBitQ查询量化（Q变体）

```cpp
// faiss/impl/RaBitQuantizer.cpp
void RaBitQDistanceComputerQ::set_query(const float* x) {
    q = x;

    // 与NotQ版本类似，但额外量化查询
    if (centroid != nullptr) {
        query_fac.qr_to_c_L2sqr = fvec_L2sqr(x, centroid, d);
    } else {
        query_fac.qr_to_c_L2sqr = fvec_norm_L2sqr(x, d);
    }

    rotated_q.resize(d);

    // 量化查询
    rotated_qq.resize((d + 7) / 8, 0);
    for (size_t i = 0; i < d; i++) {
        float val = x[i] - ((centroid == nullptr) ? 0 : centroid[i]);
        rotated_q[i] = val;

        // 标量量化到qb位
        if (centered) {
            // 有符号量化
        } else {
            // 无符号量化
            uint8_t qval = (uint8_t)(val * scale + offset);
            if (val > 0) {
                rotated_qq[i / 8] |= (1 << (i % 8));
            }
        }
    }

    // 重排查询编码以加速popcount
    rearranged_rotated_qq.resize((d + 7) / 8, 0);
    for (size_t i = 0; i < d; i++) {
        // 重排逻辑
        // ...
    }

    popcount_aligned_dim = ((d + 7) / 8 + 7) / 8 * 8;

    // 计算查询因子
    const float inv_d = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));
    query_fac.c1 = 2 * inv_d;
    query_fac.c2 = 0;
    query_fac.c34 = 0;

    if (metric_type == MetricType::METRIC_INNER_PRODUCT) {
        query_fac.qr_norm_L2sqr = fvec_norm_L2sqr(x, d);
    }
}
```

### 9.7 两阶段搜索实现

```cpp
// 两阶段搜索：1位过滤 + 多位精化
template <typename DistanceComputer>
void rabitq_two_stage_search(
        const DistanceComputer& dc,
        const uint8_t* codes,
        size_t n,
        float* distances,
        idx_t* labels,
        size_t k,
        float filter_ratio) {

    // 阶段1：1位快速过滤
    std::vector<float> distances_1bit(n);
    std::vector<float> min_distances_1bit(k);
    std::vector<idx_t> min_labels_1bit(k);

    // 使用1位距离快速计算所有向量的距离
#pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        distances_1bit[i] = dc.distance_to_code_1bit(
                codes + i * dc.code_size);
    }

    // 找到1位距离最小的k个
    float heap_threshold = std::numeric_limits<float>::max();
    using C = CMax<float, idx_t>;

    for (size_t i = 0; i < k; i++) {
        min_distances_1bit[i] = C::neutral();
        min_labels_1bit[i] = -1;
    }

    for (size_t i = 0; i < n; i++) {
        if (distances_1bit[i] < heap_threshold) {
            heap_replace_top<C>(
                    k, min_distances_1bit.data(), min_labels_1bit.data(),
                    distances_1bit[i], i);
            heap_threshold = min_distances_1bit[0];
        }
    }

    // 基于阈值确定候选集
    float threshold = heap_threshold * filter_ratio;
    std::vector<size_t> candidates;
    candidates.reserve(n);

    for (size_t i = 0; i < n; i++) {
        if (distances_1bit[i] < threshold) {
            candidates.push_back(i);
        }
    }

    // 阶段2：在候选集上进行多位精确计算
    for (size_t i = 0; i < k; i++) {
        distances[i] = C::neutral();
        labels[i] = -1;
    }

    float current_threshold = C::neutral();

    for (size_t idx : candidates) {
        float dis_full = dc.distance_to_code_full(
                codes + idx * dc.code_size);

        if (dis_full < current_threshold) {
            heap_replace_top<C>(
                    k, distances, labels, dis_full, idx);
            current_threshold = distances[0];
        }
    }

    heap_reorder<C>(k, distances, labels);
}
```

### 9.8 生产级使用示例

```cpp
// ScalarQuantizer生产环境使用
#include <faiss/ScalarQuantizer.h>
#include <faiss/IndexIVFSQ.h>

void scalar_quantizer_production_example() {
    int d = 128;
    size_t n = 1000000;
    size_t nlist = 4096;

    // 创建标量量化器
    faiss::ScalarQuantizer sq(
            d,
            faiss::ScalarQuantizer::QT_8bit);

    // 设置范围统计方式
    sq.rangestat = faiss::ScalarQuantizer::RS_minmax;
    sq.rangestat_arg = 0.01f;  // 扩展1%

    // 训练
    sq.train(n, xb);

    // 编码向量
    std::vector<uint8_t> codes(n * sq.code_size);
    sq.compute_codes(xb, codes.data(), n);

    // 解码验证
    std::vector<float> decoded(n * d);
    sq.decode(codes.data(), decoded.data(), n);

    // 计算量化误差
    float error = 0;
    for (size_t i = 0; i < n * d; i++) {
        float diff = xb[i] - decoded[i];
        error += diff * diff;
    }
    printf("SQ8 quantization error: %.6f per dimension\n",
           error / (n * d));
}

// IVF + SQ组合
void ivf_sq_example() {
    int d = 128;
    size_t nlist = 4096;

    // 粗量化器
    faiss::IndexFlatL2 quantizer(d);

    // IVF+SQ索引
    faiss::IndexIVFSQ index(
            &quantizer,
            d,
            nlist,
            faiss::ScalarQuantizer::QT_8bit,
            faiss::METRIC_L2);

    // 训练和添加
    index.train(n, xb);
    index.add(n, xb);

    // 搜索
    size_t nq = 10;
    size_t k = 100;
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    faiss::IVFSearchParameters params;
    params.nprobe = 16;

    index.search(nq, xq, k,
                 distances.data(), labels.data(), &params);
}

// RaBitQ使用
void rabitq_example() {
    int d = 128;

    // 创建RaBitQ（9位：1符号位 + 8额外位）
    faiss::RaBitQuantizer rq(
            d,
            faiss::MetricType::METRIC_L2,
            9);  // nb_bits

    // 计算编码大小
    size_t code_size = rq.compute_code_size(d, 9);
    printf("RaBitQ code_size: %zu bytes (d=%d, bits=9)\n",
           code_size, d);

    // 编码向量
    std::vector<uint8_t> codes(n * rq.code_size);
    rq.compute_codes(xb, codes.data(), n);

    // 获取距离计算机
    std::unique_ptr<faiss::FlatCodesDistanceComputer> dc(
            rq.get_distance_computer(0, nullptr, false));

    // 搜索
    size_t nq = 10;
    size_t k = 100;
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    for (size_t i = 0; i < nq; i++) {
        dc->set_query(xq + i * d);

        // 两阶段搜索
        rabitq_two_stage_search(*dc, codes.data(), n,
                                distances.data() + i * k,
                                labels.data() + i * k, k, 1.5f);
    }
}
```

### 9.9 性能优化总结

| 优化技术 | 描述 | 适用场景 | 性能提升 |
|----------|------|----------|----------|
| AVX2解码 | 8个float并行解码 | 8位SQ | 4-8x |
| AVX512解码 | 16个float并行解码 | AVX512硬件 | 8-16x |
| 位打包 | 4/6位编码压缩存储 | 4位SQ, 6位SQ | 节省50-75%内存 |
| 查询量化 | 标量量化查询向量 | RaBitQ-Q | 2-4x |
| 两阶段搜索 | 1位过滤+多位精化 | RaBitQ多位 | 3-10x |
| Popcount位运算 | 使用硬件popcount指令 | RaBitQ | 2-3x |

---

## 10. SIMD深度优化实现

### 10.1 AVX-512优化的8位解码器

AVX-512提供512位寄存器，一次可处理16个float，相比AVX2的8个float吞吐量翻倍。

```cpp
// faiss/utils/simd_simdlib.h (相关实现)
// AVX-512优化的8位标量量化解码
#if defined(__AVX512F__)

struct Codec8bit_AVX512 {
    // SIMD解码：一次解码16个分量
    static FAISS_ALWAYS_INLINE __m512
    decode_16_components(const uint8_t* code, int i) {
        // 加载16个字节到ZMM寄存器
        __m128i bytes = _mm_loadu_si128((__m128i*)(code + i));

        // 扩展为16个32位整数
        // 第一步：扩展为16个16位整数
        __m256i u16_lo = _mm256_cvtepu8_epi16(bytes);
        __m256i u16_hi = _mm256_cvtepu8_epi16(_mm_srli_si128(bytes, 8));

        // 第二步：扩展为16个32位整数
        __m512i u32 = _mm512_castsi256_si512(_mm256_cvtepu16_epi32(u16_lo));
        u32 = _mm512_inserti32x8(u32, _mm256_cvtepu16_epi32(u16_hi), 1);

        // 转换为float
        __m512 f32 = _mm512_cvtepi32_ps(u32);

        // 归一化：f32 / 255.0f + 0.5f/255.0f
        __m512 scale = _mm512_set1_ps(1.0f / 255.0f);
        __m512 bias = _mm512_set1_ps(0.5f / 255.0f);

        return _mm512_fmadd_ps(f32, scale, bias);
    }

    // 批量解码（16的倍数）
    static FAISS_ALWAYS_INLINE void
    decode_batch(const uint8_t* code, float* x, size_t d) {
        size_t i = 0;
        for (; i + 16 <= d; i += 16) {
            __m512 decoded = decode_16_components(code, i);
            _mm512_storeu_ps(x + i, decoded);
        }

        // 处理剩余元素（使用AVX2）
        for (; i + 8 <= d; i += 8) {
            __m256 decoded = Codec8bit::decode_8_components(code, i);
            _mm256_storeu_ps(x + i, decoded);
        }

        // 标量处理尾部
        for (; i < d; i++) {
            x[i] = Codec8bit::decode_component(code, i);
        }
    }
};

#endif // __AVX512F__
```

### 10.2 AVX-512距离计算优化

```cpp
// AVX-512优化的L2距离计算
#if defined(__AVX512F__)

void SQ8_distance_avx512(
        const float* x,
        const uint8_t* codes,
        const float* vmin,
        const float* vmax,
        size_t d,
        float& dis) {

    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;

    // 处理16的倍数
    for (; i + 16 <= d; i += 16) {
        // 加载16个查询
        __m512 xv = _mm512_loadu_ps(x + i);

        // 解码16个编码
        __m512 decoded = Codec8bit_AVX512::decode_16_components(codes, i);

        // 归一化到[vmin, vmax]
        __m512 vmin_v = _mm512_loadu_ps(vmin + i);
        __m512 vmax_v = _mm512_loadu_ps(vmax + i);
        __m512 range = _mm512_sub_ps(vmax_v, vmin_v);
        decoded = _mm512_fmadd_ps(decoded, range, vmin_v);

        // 计算L2距离
        __m512 diff = _mm512_sub_ps(xv, decoded);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    // 水平求和（AVX-512专用指令）
    dis = _mm512_reduce_add_ps(sum);

    // 处理剩余元素
    for (; i < d; i++) {
        float decoded = vmin[i] + (vmax[i] - vmin[i]) * codes[i] / 255.0f;
        float diff = x[i] - decoded;
        dis += diff * diff;
    }
}

// AVX-512内积距离计算
inline float SQ8_inner_product_avx512(
        const float* x,
        const uint8_t* codes,
        const float* vmin,
        const float* vmax,
        size_t d) {

    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= d; i += 16) {
        __m512 xv = _mm512_loadu_ps(x + i);
        __m512 decoded = Codec8bit_AVX512::decode_16_components(codes, i);

        __m512 vmin_v = _mm512_loadu_ps(vmin + i);
        __m512 vmax_v = _mm512_loadu_ps(vmax + i);
        __m512 range = _mm512_sub_ps(vmax_v, vmin_v);
        decoded = _mm512_fmadd_ps(decoded, range, vmin_v);

        // 内积：sum(x[i] * decoded[i])
        sum = _mm512_fmadd_ps(xv, decoded, sum);
    }

    return _mm512_reduce_add_ps(sum);
}

#endif // __AVX512F__
```

### 10.3 ARM NEON优化实现

```cpp
// faiss/utils/simd_neon.h (相关实现)
#if defined(__aarch64__) && defined(__ARM_NEON)

struct Codec8bit_NEON {
    // NEON解码：一次解码8个float（128位寄存器）
    static FAISS_ALWAYS_INLINE float32x4_t
    decode_4_components(const uint8_t* code, int i) {
        // 加载4个字节
        uint8x8_t bytes = vld1_dup_u8(code + i);

        // 扩展为16位整数
        uint16x8_t u16 = vmovl_u8(bytes);

        // 转换为float
        float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16)));
        float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16)));

        // 归一化
        float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);
        float32x4_t bias = vdupq_n_f32(0.5f / 255.0f);

        f0 = vfmaq_f32(bias, f0, scale);
        f1 = vfmaq_f32(bias, f1, scale);

        // 返回低128位（前4个）
        return f0;
    }

    // 批量解码（8个一组）
    static void decode_batch_neon(
            const uint8_t* code,
            float* x,
            size_t d) {

        size_t i = 0;

        // 处理8的倍数（两个128位寄存器）
        for (; i + 8 <= d; i += 8) {
            uint8x8_t bytes = vld1_u8(code + i);

            // 扩展
            uint16x8_t u16 = vmovl_u8(bytes);
            uint32x4_t u0 = vmovl_u16(vget_low_u16(u16));
            uint32x4_t u1 = vmovl_u16(vget_high_u16(u16));

            // 转换为float
            float32x4_t f0 = vcvtq_f32_u32(u0);
            float32x4_t f1 = vcvtq_f32_u32(u1);

            // 归一化
            float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);
            float32x4_t bias = vdupq_n_f32(0.5f / 255.0f);

            f0 = vfmaq_f32(bias, f0, scale);
            f1 = vfmaq_f32(bias, f1, scale);

            // 存储
            vst1q_f32(x + i, f0);
            vst1q_f32(x + i + 4, f1);
        }

        // 处理4的倍数
        for (; i + 4 <= d; i += 4) {
            float32x4_t decoded = decode_4_components(code, i);
            vst1q_f32(x + i, decoded);
        }

        // 标量处理尾部
        for (; i < d; i++) {
            x[i] = (code[i] + 0.5f) / 255.0f;
        }
    }
};

// NEON优化的L2距离计算
inline float SQ8_distance_neon(
        const float* x,
        const uint8_t* codes,
        const float* vmin,
        const float* vmax,
        size_t d) {

    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t i = 0;

    for (; i + 4 <= d; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);

        // 解码
        uint8x8_t bytes = vld1_dup_u8(codes + i);
        uint16x8_t u16 = vmovl_u8(bytes);
        float32x4_t decoded = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16)));

        // 归一化
        float32x4_t vmin_v = vld1q_f32(vmin + i);
        float32x4_t vmax_v = vld1q_f32(vmax + i);
        float32x4_t range = vsubq_f32(vmax_v, vmin_v);
        float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);

        decoded = vfmaq_f32(vmin_v, decoded, scale);

        // 距离
        float32x4_t diff = vsubq_f32(xv, decoded);
        sum = vfmaq_f32(sum, diff, diff);
    }

    // 水平求和
    float dis = vaddvq_f32(sum);

    for (; i < d; i++) {
        float decoded = vmin[i] + (vmax[i] - vmin[i]) * codes[i] / 255.0f;
        float diff = x[i] - decoded;
        dis += diff * diff;
    }

    return dis;
}

#endif // __aarch64__ && __ARM_NEON
```

### 10.4 ARM SVE (可变长度向量) 优化

```cpp
// faiss/utils/simd_sve.h (相关实现)
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)

struct Codec8bit_SVE {
    // SVE解码：可变长度向量，根据硬件决定
    static void decode_batch_sve(
            const uint8_t* code,
            float* x,
            size_t d) {

        size_t i = 0;

        while (i < d) {
            // 获取当前向量长度（运行时决定）
            svuint32_t pg = svwhilelt_b32_u32(i, d);
            svbool_t pg_b = svwhilelt_b32((unsigned int)i, (unsigned int)d);

            // 加载字节
            svuint8_t bytes = svld1_u8(pg_b, code + i);

            // 扩展为32位整数
            svuint16_t u16 = svunpklo_u16(bytes);
            svuint32_t u32 = svunpklo_u32(u16);

            // 转换为float
            svfloat32_t f = svsvcvt_f32_u32_z(pg_b, u32);

            // 归一化
            svfloat32_t scale = svdup_f32(1.0f / 255.0f);
            svfloat32_t bias = svdup_f32(0.5f / 255.0f);

            f = svmla_f32_z(pg_b, bias, f, scale);

            // 存储
            svst1_f32(pg_b, x + i, f);

            i += svcntb();  // 增加向量长度（字节数）
        }
    }
};

// SVE优化的距离计算
inline float SQ8_distance_sve(
        const float* x,
        const uint8_t* codes,
        const float* vmin,
        const float* vmax,
        size_t d) {

    svfloat32_t sum = svdup_f32(0.0f);
    size_t i = 0;

    while (i < d) {
        svbool_t pg = svwhilelt_b32((unsigned int)i, (unsigned int)d);

        svfloat32_t xv = svld1_f32(pg, x + i);

        // 解码
        svuint8_t bytes = svld1_u8(pg, codes + i);
        svuint16_t u16 = svunpklo_u16(bytes);
        svuint32_t u32 = svunpklo_u32(u16);
        svfloat32_t decoded = svsvcvt_f32_u32_z(pg, u32);

        // 归一化
        svfloat32_t vmin_v = svld1_f32(pg, vmin + i);
        svfloat32_t vmax_v = svld1_f32(pg, vmax + i);
        svfloat32_t range = svsub_f32_z(pg, vmax_v, vmin_v);
        svfloat32_t scale = svdup_f32(1.0f / 255.0f);
        decoded = svmla_f32_z(pg, vmin_v, decoded, scale);

        // 累加距离
        svfloat32_t diff = svsub_f32_z(pg, xv, decoded);
        sum = svmla_f32_z(pg, sum, diff, diff);

        i += svcntw();
    }

    // 水平求和
    return svaddv_f32(svptrue_b32(), sum);
}

#endif // __aarch64__ && __ARM_FEATURE_SVE
```

### 10.5 SIMD优化的编码实现

```cpp
// SIMD优化的8位编码（批量处理）
#if defined(__AVX2__)

struct Codec8bit_Encode_SIMD {
    // 批量编码：float → uint8_t
    static void encode_batch_simd(
            const float* x,
            const float* vmin,
            const float* vmax,
            uint8_t* codes,
            size_t n,
            size_t d) {

        // 预计算缩放因子
        std::vector<float> inv_range(d);
        for (size_t j = 0; j < d; j++) {
            inv_range[j] = 255.0f / (vmax[j] - vmin[j]);
        }

#pragma omp parallel for
        for (int64_t i = 0; i < n; i++) {
            const float* x_row = x + i * d;
            uint8_t* code_row = codes + i * d;

            size_t j = 0;

            // AVX2处理8的倍数
            for (; j + 8 <= d; j += 8) {
                // 加载8个float
                __m256 xv = _mm256_loadu_ps(x_row + j);
                __m256 vmin_v = _mm256_loadu_ps(vmin + j);
                __m256 inv_v = _mm256_loadu_ps(inv_range.data() + j);

                // 归一化到[0, 255]
                __m256 norm = _mm256_mul_ps(_mm256_sub_ps(xv, vmin_v), inv_v);
                norm = _mm256_max_ps(norm, _mm256_setzero_ps());
                norm = _mm256_min_ps(norm, _mm256_set1_ps(255.0f));

                // 转换为整数（带截断）
                __m256i u8 = _mm256_cvtps_epi32(norm);

                // 打包为8个字节
                __m128i u8_lo = _mm256_castsi256_si128(u8);
                __m128i u8_hi = _mm256_extracti128_si256(u8, 1);

                // 打包：32位→16位→8位
                __m128i u16 = _mm_packus_epi32(u8_lo, u8_hi);
                __m128i bytes = _mm_packus_epi16(u16, u16);

                // 存储8个字节
                _mm_storel_epi64((__m128i*)(code_row + j), bytes);
            }

            // 标量处理尾部
            for (; j < d; j++) {
                float xi = (x_row[j] - vmin[j]) * inv_range[j];
                xi = std::max(0.0f, std::min(255.0f, xi));
                code_row[j] = (uint8_t)(int)xi;
            }
        }
    }
};

#endif // __AVX2__
```

### 10.6 硬件加速的Popcount实现

```cpp
// RaBitQ 1位距离计算中的popcount优化
namespace popcount_optimized {

// x86_64硬件popcount（一条指令）
#if defined(__POPCNT__)

inline int popcnt64(uint64_t x) {
    return _mm_popcnt_u64(x);
}

inline int hamming_distance_avx2(
        const uint8_t* a,
        const uint8_t* b,
        size_t n) {

    int sum = 0;
    size_t i = 0;

    // 64位一组处理
    const uint64_t* a64 = (const uint64_t*)a;
    const uint64_t* b64 = (const uint64_t*)b;

    for (; i + 8 <= n; i += 8) {
        uint64_t xor_result = a64[i / 8] ^ b64[i / 8];
        sum += popcnt64(xor_result);
    }

    // 处理剩余字节
    for (; i < n; i++) {
        sum += _mm_popcnt_u32(a[i] ^ b[i]);
    }

    return sum;
}

// AVX2优化的批量popcount
inline void batch_hamming_distance_avx2(
        const uint8_t* query,
        const uint8_t* codes,
        size_t n,
        size_t code_size,
        int* distances) {

#pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        int dist = 0;
        const uint8_t* code = codes + i * code_size;

        // 64位一组
        const uint64_t* q64 = (const uint64_t*)query;
        const uint64_t* c64 = (const uint64_t*)code;
        size_t n64 = code_size / 8;

        for (size_t j = 0; j < n64; j++) {
            dist += popcnt64(q64[j] ^ c64[j]);
        }

        // 剩余字节
        for (size_t j = n64 * 8; j < code_size; j++) {
            dist += _mm_popcnt_u32(query[j] ^ code[j]);
        }

        distances[i] = dist;
    }
}

#endif // __POPCNT__

// ARM NEON popcount
#if defined(__aarch64__)

inline int popcnt64_neon(uint64_t x) {
    // 使用NEON指令计算popcount
    uint8x16_t v = vreinterpretq_u8_u64(vdupq_n_u64(x));
    uint8x16_t cnt = vcntq_u8(v);
    return vaddvq_u8(cnt);
}

inline int hamming_distance_neon(
        const uint8_t* a,
        const uint8_t* b,
        size_t n) {

    int sum = 0;
    size_t i = 0;

    // 128位一组（16字节）
    for (; i + 16 <= n; i += 16) {
        uint8x16_t av = vld1q_u8(a + i);
        uint8x16_t bv = vld1q_u8(b + i);

        // XOR然后popcount
        uint8x16_t xor_v = veorq_u8(av, bv);
        uint8x16_t cnt = vcntq_u8(xor_v);

        // 水平求和
        sum += vaddvq_u8(cnt);
    }

    // 处理剩余字节
    for (; i < n; i++) {
        sum += vaddv_u8(vcnt_u8(a[i] ^ b[i]));
    }

    return sum;
}

#endif // __aarch64__

} // namespace popcount_optimized
```

### 10.7 Cache友好的内存布局和预取

```cpp
// 量化编码的cache优化布局
namespace cache_optimized {

// Cache-line对齐的编码存储
struct alignas(64) AlignedCodeBlock {
    uint8_t codes[64];  // 一个cacheline (64字节)
};

// 预取优化的批量距离计算
inline void batch_distance_with_prefetch(
        const float* queries,
        const uint8_t* codes,
        const float* vmin,
        const float* vmax,
        size_t nq,
        size_t nb,
        size_t d,
        float* distances) {

    const size_t code_size = d;  // 8位SQ

    // 每个查询的处理
#pragma omp parallel for
    for (int64_t q = 0; q < nq; q++) {
        const float* query = queries + q * d;
        float* dist_row = distances + q * nb;

        // 软件预取距离（提前几个cacheline）
        const int prefetch_distance = 4;

        for (size_t i = 0; i < nb; i++) {
            // 预取未来的编码
            if (i + prefetch_distance < nb) {
                const uint8_t* future_code =
                        codes + (i + prefetch_distance) * code_size;

                // 内置预取指令
#if defined(__GNUC__)
                __builtin_prefetch(future_code, 0, 3);  // 读，高局部性
#endif
            }

            // 计算距离
            const uint8_t* code = codes + i * code_size;
            float dis = 0;

            // SIMD距离计算
            size_t j = 0;
#if defined(__AVX2__)
            __m256 sum = _mm256_setzero_ps();
            for (; j + 8 <= d; j += 8) {
                __m256 xv = _mm256_loadu_ps(query + j);
                __m256 vmin_v = _mm256_loadu_ps(vmin + j);
                __m256 vmax_v = _mm256_loadu_ps(vmax + j);

                // 解码
                __m256i codes_v = _mm256_loadu_si256((__m256i*)(code + j));
                __m256i zero = _mm256_setzero_si256();
                __m256i c_lo = _mm256_unpacklo_epi8(codes_v, zero);
                __m256i c_hi = _mm256_unpackhi_epi8(codes_v, zero);

                __m256 f_lo = _mm256_cvtepi32_ps(c_lo);
                __m256 f_hi = _mm256_cvtepi32_ps(c_hi);

                __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
                __m256 decoded_lo = _mm256_fmadd_ps(f_lo, scale, vmin_v);
                __m256 decoded_hi = _mm256_fmadd_ps(f_hi, scale, vmin_v);

                __m256 diff_lo = _mm256_sub_ps(xv, decoded_lo);
                __m256 diff_hi = _mm256_sub_ps(
                        _mm256_loadu_ps(query + j + 4), decoded_hi);

                sum = _mm256_add_ps(sum, _mm256_mul_ps(diff_lo, diff_lo));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(diff_hi, diff_hi));
            }

            alignas(32) float tmp[8];
            _mm256_storeu_ps(tmp, sum);
            dis = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                  tmp[4] + tmp[5] + tmp[6] + tmp[7];
#endif

            // 标量尾部
            for (; j < d; j++) {
                float decoded = vmin[j] +
                        (vmax[j] - vmin[j]) * code[j] / 255.0f;
                float diff = query[j] - decoded;
                dis += diff * diff;
            }

            dist_row[i] = dis;
        }
    }
}

// 交错存储模式：提高向量化效率
struct InterleavedCodes {
    // 存储格式：codes[n][d] → 按维度交错
    // 维度0的所有向量编码连续存储，然后维度1，等等
    // 这样可以最大化SIMD利用率

    std::vector<uint8_t> storage;

    InterleavedCodes(size_t n, size_t d) {
        storage.resize(n * d);
    }

    // 访问(i, j)元素
    inline uint8_t& operator()(size_t i, size_t j) {
        return storage[j * n + i];  // 交错存储
    }

    // SIMD友好的批量解码
    void decode_batch_simd(
            const std::vector<size_t>& indices,
            size_t d,
            float* output) const {

        // 一次解码一个维度的所有向量
        for (size_t j = 0; j < d; j++) {
            const uint8_t* dim_codes = storage.data() + j * indices.size();

            size_t i = 0;
#if defined(__AVX2__)
            // 批量解码8个向量的同一维度
            for (; i + 8 <= indices.size(); i += 8) {
                __m128i bytes = _mm_loadl_epi64((__m128i*)(dim_codes + i));
                // ... SIMD解码逻辑
            }
#endif

            for (; i < indices.size(); i++) {
                output[indices[i] * d + j] = dim_codes[i] / 255.0f;
            }
        }
    }
};

} // namespace cache_optimized
```

### 10.8 线程级并行优化

```cpp
// OpenMP优化的量化编码
namespace parallel_optimized {

// 动态调度的并行编码
inline void parallel_encode(
        const float* x,
        uint8_t* codes,
        size_t n,
        size_t d,
        const float* vmin,
        const float* vmax) {

    // 设置线程数
    int nthreads = omp_get_max_threads();
    omp_set_num_threads(nthreads);

    // 每个线程的私有数据
    std::vector<std::vector<float>> thread_inv_range(nthreads);

    for (int t = 0; t < nthreads; t++) {
        thread_inv_range[t].resize(d);
        for (size_t j = 0; j < d; j++) {
            thread_inv_range[t][j] = 255.0f / (vmax[j] - vmin[j]);
        }
    }

    // 动态调度以平衡负载
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        const float* inv_range = thread_inv_range[tid].data();

#pragma omp for schedule(dynamic, 1000)
        for (int64_t i = 0; i < n; i++) {
            uint8_t* code = codes + i * d;
            const float* x_row = x + i * d;

            for (size_t j = 0; j < d; j++) {
                float xi = (x_row[j] - vmin[j]) * inv_range[j];
                xi = std::max(0.0f, std::min(255.0f, xi));
                code[j] = (uint8_t)(int)xi;
            }
        }
    }
}

// NUMA感知的并行搜索
inline void numa_aware_search(
        const float* queries,
        const uint8_t* codes,
        size_t nq,
        size_t nb,
        size_t d,
        float* distances,
        idx_t* labels,
        size_t k) {

    // 获取NUMA节点数
    int numa_nodes = numa_num_configured_nodes();
    if (numa_nodes <= 1) {
        // 非NUMA系统，使用普通并行
        // ...
        return;
    }

    // 将数据分布到NUMA节点
    std::vector<std::vector<size_t>> node_splits(numa_nodes);
    for (size_t i = 0; i < nb; i++) {
        node_splits[i % numa_nodes].push_back(i);
    }

    // 在每个NUMA节点上执行搜索
#pragma omp parallel num_threads(numa_nodes)
    {
        int node_id = omp_get_thread_num();
        numa_run_on_node(node_id);

        // 处理本节点的数据
        for (size_t idx : node_splits[node_id]) {
            // 距离计算
            // ...
        }
    }
}

} // namespace parallel_optimized
```

### 10.9 SIMD优化性能对比

| 优化技术 | 数据类型 | 每周期处理 | 延迟 | 吞吐量 |
|----------|----------|-----------|------|--------|
| **x86_64** |
| SSE 4.2 | 4×float | 4 | 3-4 | 1/cycle |
| AVX2 | 8×float | 8 | 4-5 | 2/cycle |
| AVX-512 | 16×float | 16 | 5-6 | 2/cycle |
| **ARM** |
| NEON | 4×float | 4 | 3-4 | 2/cycle |
| SVE-128 | 4×float | 4 | 3-4 | 2/cycle |
| SVE-256 | 8×float | 8 | 4-5 | 2/cycle |
| SVE-512 | 16×float | 16 | 5-6 | 2/cycle |

### 10.10 编译器内联优化

```cpp
// 强制内联的关键路径函数
namespace FAISS_FAKE_ICC { // prevent ICC from interpreting this attribute
// 强制内联宏
#if defined(__GNUC__)
#define FAISS_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define FAISS_ALWAYS_INLINE __forceinline
#else
#define FAISS_ALWAYS_INLINE inline
#endif

// 标记为热点的函数
#if defined(__GNUC__) && (__GNUC__ >= 5)
#define FAISS_HOT_FUNCTION __attribute__((hot))
#else
#define FAISS_HOT_FUNCTION
#endif

// 示例：热点编码函数
FAISS_ALWAYS_INLINE FAISS_HOT_FUNCTION
void encode_component_hot(
        float x,
        const float* vmin,
        const float* vmax,
        const float* inv_range,
        uint8_t* code,
        int i) {

    float xi = (x - vmin[i]) * inv_range[i];
    xi = std::max(0.0f, std::min(255.0f, xi));
    code[i] = (uint8_t)(int)xi;
}
}
```

---

## 11. RaBitQ SIMD深度优化

### 11.1 AVX-512优化的1位距离计算

```cpp
// AVX-512优化的1位RaBitQ距离计算
// 一次处理64个向量的1位编码
#ifdef __AVX512F__

struct RaBitQ1BitAVX512 {
    // 计算1位编码的内积：IP = sum(signs * abs_query) * base_factor
    static inline void compute_1bit_ip_avx512(
            const uint8_t* query_signs,  // 查询的符号位 (d/8 字节)
            const float* query_abs,      // 查询的绝对值 (d 维)
            const uint8_t* db_signs,     // 数据库的符号位 (n * d/8)
            const float* db_factors,     // 数据库的base_factor (n)
            size_t d,
            size_t n,
            float* ips_out) {

        size_t d_bytes = (d + 7) / 8;

        // 每次处理64个向量
        size_t i = 0;
        for (; i + 64 <= n; i += 64) {
            __m512 ip_sum = _mm512_setzero_ps();

            // 计算每个维度的贡献
            for (size_t j = 0; j < d; j++) {
                size_t byte_idx = j / 8;
                size_t bit_idx = 7 - (j % 8);
                uint8_t query_bit = (query_signs[byte_idx] >> bit_idx) & 1;

                // 加载64个向量的第j位符号
                const uint8_t* signs_ptr = db_signs + i * d_bytes + byte_idx;
                __m512i signs_bytes = _mm512_loadu_si512(
                    (const __m512i*)signs_ptr);

                // 提取第bit_idx位
                __mmask64 signs_mask = _mm512_test_epi64_mask(
                    signs_bytes, _mm512_set1_epi64(1ULL << bit_idx));

                // 将mask转换为向量：1表示正，-1表示负
                __m512i ones = _mm512_maskz_set1_epi32(signs_mask, 1);
                __m512i neg_ones = _mm512_maskz_set1_epi32(~signs_mask, -1);
                __m512i signs = _mm512_add_epi32(ones, neg_ones);

                // 如果查询位为1，signs不变；否则反转
                if (query_bit) {
                    // 查询位为1：直接使用signs
                } else {
                    // 查询位为0：反转signs
                    signs = _mm512_xor_epi32(signs, _mm512_set1_epi32(-1));
                }

                // 累加：signs[j] * abs_query[j]
                __m512 abs_val = _mm512_set1_ps(query_abs[j]);
                __m512 contrib = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(signs), abs_val);
                ip_sum = _mm512_add_ps(ip_sum, contrib);
            }

            // 乘以base_factor并存储
            __m512 factors = _mm512_loadu_ps(db_factors + i);
            ip_sum = _mm512_mul_ps(ip_sum, factors);
            _mm512_storeu_ps(ips_out + i, ip_sum);
        }

        // 处理剩余向量
        for (; i < n; i++) {
            float ip = 0;
            for (size_t j = 0; j < d; j++) {
                size_t byte_idx = j / 8;
                size_t bit_idx = 7 - (j % 8);
                uint8_t query_bit = (query_signs[byte_idx] >> bit_idx) & 1;
                uint8_t db_bit = (db_signs[i * d_bytes + byte_idx] >> bit_idx) & 1;

                int sign = (query_bit == db_bit) ? 1 : -1;
                ip += sign * query_abs[j];
            }
            ips_out[i] = ip * db_factors[i];
        }
    }

    // 批量L2距离计算：||q||^2 - 2*IP + d*factor^2
    static inline void compute_1bit_l2_avx512(
            const uint8_t* query_signs,
            const float* query_abs,
            float query_norm_sq,
            const uint8_t* db_signs,
            const float* db_factors,
            size_t d,
            size_t n,
            float* distances_out) {

        std::vector<float> ips(n);
        compute_1bit_ip_avx512(query_signs, query_abs,
                              db_signs, db_factors,
                              d, n, ips.data());

        // 计算L2距离
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 ip = _mm512_loadu_ps(ips.data() + i);
            __m512 factors = _mm512_loadu_ps(db_factors + i);

            // ||q||^2 - 2*IP + d*factor^2
            __m512 term1 = _mm512_set1_ps(query_norm_sq);
            __m512 term2 = _mm512_mul_ps(ip, _mm512_set1_ps(2.0f));
            __m512 term3 = _mm512_mul_ps(factors, factors);
            term3 = _mm512_mul_ps(term3, _mm512_set1_ps((float)d));

            __m512 dis = _mm512_sub_ps(term1, term2);
            dis = _mm512_add_ps(dis, term3);

            _mm512_storeu_ps(distances_out + i, dis);
        }

        for (; i < n; i++) {
            distances_out[i] = query_norm_sq - 2.0f * ips[i] +
                               d * db_factors[i] * db_factors[i];
        }
    }
};
#endif
```

### 11.2 AVX2优化的多位RaBitQ

```cpp
// AVX2优化的多位RaBitQ编码/解码
#ifdef __AVX2__

struct RaBitQMultiBitAVX2 {
    // 批量编码：float → k位
    static inline void encode_kbit_avx2(
            const float* x,
            size_t n,
            size_t d,
            size_t kbits,
            uint8_t* codes_out) {

        size_t codes_per_byte = 8 / kbits;
        size_t code_bytes = (d * kbits + 7) / 8;

        for (size_t i = 0; i < n; i++) {
            const float* vec = x + i * d;
            uint8_t* codes = codes_out + i * code_bytes;

            // 每次处理8个分量
            size_t j = 0;
            for (; j + 8 <= d; j += 8) {
                // 加载8个浮点数
                __m256 vals = _mm256_loadu_ps(vec + j);

                // 量化到[0, 2^kbits - 1]
                __m256 scaled = _mm256_mul_ps(vals, _mm256_set1_ps((float)(1 << kbits) - 1));
                scaled = _mm256_max_ps(scaled, _mm256_setzero_ps());
                scaled = _mm256_min_ps(scaled, _mm256_set1_ps((float)(1 << kbits) - 1));

                // 转换为整数
                __m256i quantized = _mm256_cvtps_epi32(scaled);

                // 打包到字节
                alignas(32) uint32_t q[8];
                _mm256_storeu_si256((__m256i*)q, quantized);

                // 打包k位编码
                pack_kbits(q, 8, kbits, codes + (j * kbits / 8));
            }

            // 处理剩余元素
            for (; j < d; j++) {
                float val = vec[j];
                val = std::max(0.0f, std::min(1.0f, val));
                uint8_t q = (uint8_t)(val * ((1 << kbits) - 1));
                pack_kbit(q, j, kbits, codes);
            }
        }
    }

    // 批量解码：k位 → float
    static inline void decode_kbit_avx2(
            const uint8_t* codes,
            size_t n,
            size_t d,
            size_t kbits,
            float* x_out) {

        size_t code_bytes = (d * kbits + 7) / 8;
        float scale = 1.0f / ((1 << kbits) - 1);

        for (size_t i = 0; i < n; i++) {
            const uint8_t* code = codes + i * code_bytes;
            float* vec = x_out + i * d;

            // 每次解码8个分量
            size_t j = 0;
            for (; j + 8 <= d; j += 8) {
                alignas(32) uint8_t q[8];
                unpack_kbits(code, j, 8, kbits, q);

                // 转换为浮点数
                __m128i q_low = _mm_loadl_epi64((__m128i*)q);
                __m256i q_i32 = _mm256_cvtepu8_epi32(q_low);

                // 归一化
                __m256 vals = _mm256_cvtepi32_ps(q_i32);
                vals = _mm256_mul_ps(vals, _mm256_set1_ps(scale));

                _mm256_storeu_ps(vec + j, vals);
            }

            // 处理剩余元素
            for (; j < d; j++) {
                uint8_t q = unpack_kbit(code, j, kbits);
                vec[j] = q * scale;
            }
        }
    }

private:
    static inline void pack_kbits(
            const uint32_t* vals,
            size_t n,
            size_t kbits,
            uint8_t* out) {

        uint8_t current = 0;
        int bit_offset = 0;

        for (size_t i = 0; i < n; i++) {
            current |= (vals[i] << bit_offset);
            bit_offset += kbits;

            while (bit_offset >= 8) {
                *out++ = current & 0xFF;
                current >>= 8;
                bit_offset -= 8;
            }
        }

        if (bit_offset > 0) {
            *out = current;
        }
    }

    static inline void unpack_kbits(
            const uint8_t* in,
            size_t offset,
            size_t n,
            size_t kbits,
            uint8_t* out) {

        const uint8_t* ptr = in + (offset * kbits / 8);
        int bit_offset = (offset * kbits) % 8;

        for (size_t i = 0; i < n; i++) {
            uint16_t val = *ptr++;
            val |= (*ptr) << 8;

            out[i] = (val >> bit_offset) & ((1 << kbits) - 1);

            bit_offset += kbits;
            if (bit_offset >= 8) {
                ptr += bit_offset / 8;
                bit_offset %= 8;
            }
        }
    }

    static inline uint8_t unpack_kbit(
            const uint8_t* in,
            size_t offset,
            size_t kbits) {

        size_t byte_idx = (offset * kbits) / 8;
        int bit_offset = (offset * kbits) % 8;

        uint16_t val = in[byte_idx] | (in[byte_idx + 1] << 8);
        return (val >> bit_offset) & ((1 << kbits) - 1);
    }

    static inline void pack_kbit(
            uint8_t val,
            size_t offset,
            size_t kbits,
            uint8_t* out) {

        size_t byte_idx = (offset * kbits) / 8;
        int bit_offset = (offset * kbits) % 8;

        out[byte_idx] |= (val << bit_offset);

        int remaining = bit_offset + kbits - 8;
        if (remaining > 0) {
            out[byte_idx + 1] = val >> (kbits - remaining);
        }
    }
};
#endif
```

### 11.3 ARM NEON优化的RaBitQ

```cpp
#ifdef __ARM_NEON

struct RaBitQNEON {
    // NEON优化的1位内积计算
    static inline void compute_1bit_ip_neon(
            const uint8_t* query_signs,
            const float* query_abs,
            const uint8_t* db_signs,
            const float* db_factors,
            size_t d,
            size_t n,
            float* ips_out) {

        size_t d_bytes = (d + 7) / 8;

        // 每次处理4个向量
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t ip_sum = vdupq_n_f32(0.0f);

            for (size_t j = 0; j < d; j++) {
                size_t byte_idx = j / 8;
                size_t bit_idx = 7 - (j % 8);
                uint8_t query_bit = (query_signs[byte_idx] >> bit_idx) & 1;

                // 加载4个向量的第j位符号
                uint8x8_t signs_bytes = vld1_u8(db_signs + (i + 0) * d_bytes + byte_idx);
                uint8x8_t signs_bytes_1 = vld1_u8(db_signs + (i + 1) * d_bytes + byte_idx);
                uint8x8_t signs_bytes_2 = vld1_u8(db_signs + (i + 2) * d_bytes + byte_idx);
                uint8x8_t signs_bytes_3 = vld1_u8(db_signs + (i + 3) * d_bytes + byte_idx);

                // 提取特定位
                uint8x8_t mask = vdup_n_u8(1 << bit_idx);
                uint8_t bits = vtst_u8(signs_bytes, mask);
                uint8_t bits_1 = vtst_u8(signs_bytes_1, mask);
                uint8_t bits_2 = vtst_u8(signs_bytes_2, mask);
                uint8_t bits_3 = vtst_u8(signs_bytes_3, mask);

                // 转换为float：1或-1
                float32x4_t signs_f = {
                    bits[0] ? 1.0f : -1.0f,
                    bits_1[0] ? 1.0f : -1.0f,
                    bits_2[0] ? 1.0f : -1.0f,
                    bits_3[0] ? 1.0f : -1.0f
                };

                // 如果查询位为0，反转signs
                if (!query_bit) {
                    signs_f = vnegq_f32(signs_f);
                }

                // 累加
                float32x4_t abs_val = vdupq_n_f32(query_abs[j]);
                ip_sum = vmlaq_f32(ip_sum, signs_f, abs_val);
            }

            // 乘以base_factor
            float32x4_t factors = vld1q_f32(db_factors + i);
            ip_sum = vmulq_f32(ip_sum, factors);
            vst1q_f32(ips_out + i, ip_sum);
        }

        // 处理剩余向量
        for (; i < n; i++) {
            float ip = 0;
            for (size_t j = 0; j < d; j++) {
                size_t byte_idx = j / 8;
                size_t bit_idx = 7 - (j % 8);
                uint8_t query_bit = (query_signs[byte_idx] >> bit_idx) & 1;
                uint8_t db_bit = (db_signs[i * d_bytes + byte_idx] >> bit_idx) & 1;

                int sign = (query_bit == db_bit) ? 1 : -1;
                ip += sign * query_abs[j];
            }
            ips_out[i] = ip * db_factors[i];
        }
    }

    // NEON优化的popcount汉明距离
    static inline uint32x4_t popcount_neon(uint8x16_t bytes) {
        // 计算每个字节的popcount
        const uint8x16_t mask1 = vdupq_n_u8(0x55);
        const uint8x16_t mask2 = vdupq_n_u8(0x33);
        const uint8x16_t mask4 = vdupq_n_u8(0x0F);

        uint8x16_t n = bytes;
        n = vshlq_n_u8(n, 4);     // 每字节的高4位
        uint8x16_t m = vandq_u8(bytes, mask4);  // 低4位

        // 查表法计算popcount（0-15）
        static const uint8_t popcount_table[16] = {
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
        };

        // 对于每个半字节查表
        uint8x16_t table = vld1q_u8(popcount_table);
        uint8x16_t low_count = vqtbl1q_u8(table, m);
        uint8x16_t high_count = vqtbl1q_u8(table, n);

        // 合并
        return vaddl_u16(vget_low_u8(low_count), vget_low_u8(high_count));
    }
};
#endif
```

### 11.4 混合精度RaBitQ搜索

```cpp
// 混合精度：使用1位快速过滤，多位精化
template<typename SIMD_1BIT, typename SIMD_MULTIBIT>
struct HybridRaBitQSearch {
    // 两阶段搜索
    static void search_hybrid(
            const float* queries,
            const uint8_t* db_codes_1bit,
            const uint8_t* db_codes_multibit,
            const float* db_factors,
            size_t nq,
            size_t nb,
            size_t d,
            size_t k,
            float* distances,
            idx_t* labels) {

        // 阶段1：1位快速过滤，保留top-4k候选
        size_t filter_k = k * 4;

        for (size_t q = 0; q < nq; q++) {
            std::vector<float> distances_1bit(nb);
            std::vector<idx_t> labels_1bit(nb);

            // 使用SIMD优化的1位距离计算
            SIMD_1BIT::compute_1bit_l2(
                queries + q * d,
                db_codes_1bit,
                db_factors,
                d, nb,
                distances_1bit.data());

            // 部分排序，保留top-4k
            std::partial_sort(
                distances_1bit.begin(),
                distances_1bit.begin() + filter_k,
                distances_1bit.end());

            // 提取候选
            std::vector<idx_t> candidates(filter_k);
            for (size_t i = 0; i < filter_k; i++) {
                candidates[i] = labels_1bit[i];
            }

            // 阶段2：在候选上使用多位精确计算
            for (size_t idx : candidates) {
                // 多位精确距离计算
                float dis = SIMD_MULTIBIT::compute_multibit_l2(
                    queries + q * d,
                    db_codes_multibit + idx * get_code_size(d),
                    d);

                // 更新top-k
                // ...
            }
        }
    }
};
```

---

## 练习题

1. 实现8位标量量化编码/解码
2. 实现4位打包存储
3. 实现SIMD优化的距离计算
4. 比较不同SQ类型的性能
5. **实现AVX-512优化的批量编码**
6. **实现ARM NEON的4位解码器**
7. **优化popcount的批量汉明距离计算**
8. **实现混合精度RaBitQ搜索**
9. **实现1位RaBitQ的内积计算**
10. **分析SIMD优化的加速比**

## 扩展阅读

- faiss/impl/ScalarQuantizer.h - SQ实现
- faiss/IndexIVFRaBitQ.h - RaBitQ索引
- faiss/IndexScalarQuantizer.h - SQ索引
- faiss/utils/simd_simdlib.h - SIMD库实现
- Intel Intrinsics Guide - SIMD指令参考
- [量化理论](https://en.wikipedia.org/wiki/Quantization)
