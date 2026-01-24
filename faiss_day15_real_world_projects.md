# Faiss深度课程 - 第15天：实战案例与项目实战

## 课程目标

通过完整的实战项目，将前面14天学到的知识应用到实际场景中，学习如何构建生产级的向量检索系统。

---

## 1. 实战项目一：图像相似搜索引擎

### 1.1 项目概述

构建一个基于深度学习特征的图像相似搜索引擎，支持百万级图片的实时检索。

### 1.2 系统架构

```cpp
// 图像搜索引擎架构
struct ImageSearchEngine {
    // 特征提取模型
    std::unique_ptr<FeatureExtractor> extractor;

    // Faiss索引
    std::unique_ptr<faiss::Index> index;

    // 图像元数据
    std::vector<ImageMetadata> metadata;

    // 线程池
    ThreadPool pool;

    // 统计信息
    SearchStats stats;
};
```

### 1.3 特征提取模块

```cpp
// 深度学习特征提取器
class FeatureExtractor {
public:
    FeatureExtractor(const std::string& model_path) {
        // 加载预训练模型（ResNet-50等）
        model = load_model(model_path);
    }

    // 批量提取特征
    std::vector<float> extract(const std::vector<Image>& images) {
        std::vector<float> features;
        features.reserve(images.size() * d);

        for (const auto& img : images) {
            // 预处理
            auto processed = preprocess(img);

            // 模型推理
            auto feature = model->forward(processed);

            // L2归一化
            normalize_l2(feature);

            features.insert(features.end(), feature.begin(), feature.end());
        }

        return features;
    }

    int d = 2048;  // 特征维度（ResNet-50）

private:
    std::unique_ptr<NeuralNetwork> model;

    Image preprocess(const Image& img) {
        // Resize to 224x224
        // Normalize with ImageNet stats
        // ...
    }

    void normalize_l2(std::vector<float>& feature) {
        float norm = 0;
        for (float v : feature) norm += v * v;
        norm = std::sqrt(norm);

        for (float& v : feature) v /= norm;
    }
};
```

### 1.4 索引构建

```cpp
// 构建IVF+PQ索引（适合大规模）
void build_image_index(
        const std::vector<float>& features,
        int nlist,
        int M,
        int nbits) {

    int d = 2048;
    size_t n = features.size() / d;

    // 1. 创建粗量化器
    faiss::IndexFlatL2 quantizer(d);

    // 2. 创建IVFPQ索引
    faiss::IndexIVFPQ index(
        &quantizer, d, nlist, M, nbits);

    // 3. 训练
    size_t ntrain = std::min(n, (size_t)100000);
    index.train(ntrain, features.data());

    // 4. 添加向量
    index.add(n, features.data());

    // 5. 配置nprobe（精度vs速度权衡）
    index.nprobe = 16;

    return index;
}
```

### 1.5 完整搜索引擎实现

```cpp
class ImageSearchEngine {
public:
    ImageSearchEngine(
            const std::string& model_path,
            const std::string& index_path = "") {

        // 1. 初始化特征提取器
        extractor = std::make_unique<FeatureExtractor>(model_path);

        // 2. 加载或创建索引
        if (!index_path.empty()) {
            load_index(index_path);
        }
    }

    // 添加图像
    void add_images(
            const std::vector<Image>& images,
            const std::vector<ImageMetadata>& meta) {

        // 提取特征
        auto features = extractor->extract(images);

        // 添加到索引
        idx_t n = images.size();
        index->add(n, features.data());

        // 保存元数据
        metadata.insert(
            metadata.end(),
            meta.begin(),
            meta.end());
    }

    // 搜索相似图像
    std::vector<SearchResult> search(
            const Image& query,
            int k = 10) {

        // 提取查询特征
        auto query_features = extractor->extract({query});

        // 搜索
        std::vector<float> distances(k);
        std::vector<idx_t> labels(k);

        index->search(1, query_features.data(), k,
                     distances.data(), labels.data());

        // 组装结果
        std::vector<SearchResult> results;
        for (int i = 0; i < k; i++) {
            if (labels[i] >= 0 && labels[i] < (idx_t)metadata.size()) {
                results.push_back({
                    .image_id = labels[i],
                    .distance = distances[i],
                    .metadata = metadata[labels[i]]
                });
            }
        }

        return results;
    }

    // 保存索引
    void save(const std::string& path) {
        faiss::IOWriter* writer = new faiss::IOFile(path.c_str(), "wb");
        faiss::write_index(index.get(), writer);
        delete writer;

        // 保存元数据
        save_metadata(path + ".meta");
    }

private:
    void load_index(const std::string& path) {
        faiss::IOReader* reader = new faiss::IOFile(path.c_str(), "rb");
        index.reset(faiss::read_index(reader));
        delete reader;

        // 加载元数据
        load_metadata(path + ".meta");
    }

    std::unique_ptr<FeatureExtractor> extractor;
    std::unique_ptr<faiss::Index> index;
    std::vector<ImageMetadata> metadata;
};
```

### 1.6 性能优化

```cpp
// 批量搜索优化
std::vector<std::vector<SearchResult>> batch_search(
        const std::vector<Image>& queries,
        int k = 10) {

    // 批量提取特征（更高效）
    auto query_features = extractor->extract(queries);
    idx_t nq = queries.size();

    // 批量搜索
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    index->search(nq, query_features.data(), k,
                 distances.data(), labels.data());

    // 组装结果
    std::vector<std::vector<SearchResult>> results(nq);
    for (idx_t q = 0; q < nq; q++) {
        for (int i = 0; i < k; i++) {
            idx_t idx = q * k + i;
            if (labels[idx] >= 0) {
                results[q].push_back({
                    .image_id = labels[idx],
                    .distance = distances[idx],
                    .metadata = metadata[labels[idx]]
                });
            }
        }
    }

    return results;
}
```

---

## 2. 实战项目二：文本语义搜索引擎

### 2.1 项目概述

使用预训练的Transformer模型（如BERT）提取文本embeddings，实现语义相似度搜索。

### 2.2 BERT特征提取

```cpp
class BERTFeatureExtractor {
public:
    BERTFeatureExtractor() {
        // 加载预训练BERT模型
        model = load_bert_model("bert-base-uncased");
        tokenizer = load_tokenizer("bert-base-uncased");
    }

    std::vector<float> encode(const std::string& text) {
        // 1. Tokenize
        auto tokens = tokenizer->encode(text);

        // 2. 模型推理
        auto outputs = model->forward(tokens);

        // 3. 获取[CLS] token的embedding
        auto cls_embedding = outputs.last_hidden_state[0];  // [CLS]

        // 4. 投影到固定维度
        return project(cls_embedding);
    }

    std::vector<float> encode_batch(
            const std::vector<std::string>& texts) {

        // 批量编码（更高效）
        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(texts.size());

        // 按批次处理
        const int batch_size = 32;
        for (size_t i = 0; i < texts.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, texts.size());
            std::vector<std::string> batch(
                texts.begin() + i,
                texts.begin() + end);

            auto batch_embeddings = model->encode_forward(batch);
            embeddings.insert(
                embeddings.end(),
                batch_embeddings.begin(),
                batch_embeddings.end());
        }

        // Flatten
        std::vector<float> result;
        for (const auto& emb : embeddings) {
            result.insert(result.end(), emb.begin(), emb.end());
        }

        return result;
    }

    int d = 768;  // BERT-base hidden size

private:
    std::unique_ptr<BERTModel> model;
    std::unique_ptr<Tokenizer> tokenizer;
};
```

### 2.3 内积索引配置

```cpp
// 文本搜索通常使用内积（余弦相似度）
faiss::Index* build_text_index(
        const std::vector<float>& embeddings,
        size_t nlist) {

    int d = 768;
    size_t n = embeddings.size() / d;

    // 1. 先归一化所有向量
    std::vector<float> normalized = embeddings;
    for (size_t i = 0; i < n; i++) {
        float* vec = normalized.data() + i * d;
        normalize_l2(vec, d);
    }

    // 2. 使用内积索引（等价于余弦相似度）
    faiss::IndexFlatIP quantizer(d);

    // 3. IVFFlat for exact search
    faiss::IndexIVFFlat index(
        &quantizer, d, nlist, faiss::METRIC_INNER_PRODUCT);

    // 4. 训练和添加
    size_t ntrain = std::min(n, (size_t)50000);
    index.train(ntrain, normalized.data());
    index.add(n, normalized.data());

    return new faiss::IndexIDMap2(index);
}
```

### 2.4 混合检索（关键词+语义）

```cpp
class HybridSearchEngine {
public:
    HybridSearchEngine() {
        // 语义索引
        semantic_index = build_semantic_index();

        // 关键词索引（BM25）
        keyword_index = build_keyword_index();
    }

    // 混合搜索
    std::vector<Document> search(
            const std::string& query,
            int k = 10,
            float alpha = 0.5) {

        // 1. 语义搜索
        auto semantic_results = semantic_search(query, k * 2);

        // 2. 关键词搜索
        auto keyword_results = keyword_search(query, k * 2);

        // 3. 融合分数
        std::map<idx_t, float> combined_scores;

        // 归一化并合并
        float max_semantic = 0;
        for (auto& r : semantic_results) {
            max_semantic = std::max(max_semantic, r.score);
        }

        float max_keyword = 0;
        for (auto& r : keyword_results) {
            max_keyword = std::max(max_keyword, r.score);
        }

        for (auto& r : semantic_results) {
            float normalized_score = r.score / max_semantic;
            combined_scores[r.doc_id] += alpha * normalized_score;
        }

        for (auto& r : keyword_results) {
            float normalized_score = r.score / max_keyword;
            combined_scores[r.doc_id] += (1 - alpha) * normalized_score;
        }

        // 4. 排序并返回top-K
        std::vector<std::pair<float, idx_t>> sorted;
        for (auto& [id, score] : combined_scores) {
            sorted.push_back({score, id});
        }
        std::sort(sorted.rbegin(), sorted.rend());

        std::vector<Document> results;
        for (int i = 0; i < k && i < sorted.size(); i++) {
            results.push_back(get_document(sorted[i].second));
        }

        return results;
    }

private:
    std::unique_ptr<faiss::Index> semantic_index;
    std::unique_ptr<KeywordIndex> keyword_index;
};
```

---

## 3. 实战项目三：推荐系统中的向量检索

### 3.1 项目概述

构建一个基于用户-物品协同过滤的推荐系统，使用Faiss进行高效相似度计算。

### 3.2 矩阵分解特征提取

```cpp
// 矩阵分解模型
class MatrixFactorization {
public:
    MatrixFactorization(int n_users, int n_items, int k) :
        n_users(n_users), n_items(n_items), k(k) {

        // 随机初始化用户和物品矩阵
        user_factors = random_matrix(n_users, k);
        item_factors = random_matrix(n_items, k);
    }

    // 训练（使用ALS）
    void train(
            const std::vector<Rating>& ratings,
            int n_iter = 10,
            float lambda = 0.01) {

        for (int iter = 0; iter < n_iter; iter++) {
            // 固定物品因子，更新用户因子
            update_user_factors(ratings, lambda);

            // 固定用户因子，更新物品因子
            update_item_factors(ratings, lambda);

            // 计算训练误差
            float error = compute_rmse(ratings);
            printf("Iteration %d: RMSE = %.4f\n", iter, error);
        }
    }

    // 获取用户embedding
    std::vector<float> get_user_embedding(int user_id) {
        return std::vector<float>(
            user_factors[user_id].begin(),
            user_factors[user_id].end());
    }

    // 获取物品embedding
    std::vector<float> get_item_embedding(int item_id) {
        return std::vector<float>(
            item_factors[item_id].begin(),
            item_factors[item_id].end());
    }

private:
    int n_users, n_items, k;
    std::vector<std::vector<float>> user_factors;
    std::vector<std::vector<float>> item_factors;

    void update_user_factors(
            const std::vector<Rating>& ratings,
            float lambda) {
        // ALS交替最小二乘
        // ...
    }
};
```

### 3.3 推荐索引构建

```cpp
class RecommenderSystem {
public:
    RecommenderSystem(int n_users, int n_items, int k = 128) :
        mf(n_users, n_items, k) {

        // 构建物品索引
        build_item_index();
    }

    // 训练模型
    void train(const std::vector<Rating>& ratings) {
        mf.train(ratings);

        // 更新物品索引
        update_item_index();
    }

    // 为用户推荐物品
    std::vector<int> recommend(
            int user_id,
            int k = 10,
            const std::vector<int>& exclude = {}) {

        // 1. 获取用户embedding
        auto user_emb = mf.get_user_embedding(user_id);

        // 2. 搜索相似的物品
        std::vector<float> distances(k);
        std::vector<idx_t> labels(k);

        // 使用IDSelector排除已交互物品
        faiss::IDSelectorArray selector(
            exclude.size(),
            exclude.data());

        faiss::SearchParametersIVF params;
        params.sel = &selector;

        item_index->search(
            1, user_emb.data(), k,
            distances.data(), labels.data(), &params);

        // 3. 返回推荐物品ID
        return std::vector<int>(
            labels.begin(),
            labels.end());
    }

    // 查找相似物品
    std::vector<int> similar_items(int item_id, int k = 10) {
        auto item_emb = mf.get_item_embedding(item_id);

        std::vector<float> distances(k);
        std::vector<idx_t> labels(k);

        item_index->search(
            1, item_emb.data(), k,
            distances.data(), labels.data());

        return std::vector<int>(
            labels.begin(),
            labels.end());
    }

private:
    MatrixFactorization mf;
    std::unique_ptr<faiss::Index> item_index;

    void build_item_index() {
        int k = mf.k;
        int n_items = mf.n_items;

        // 收集所有物品embedding
        std::vector<float> item_embeddings;
        item_embeddings.reserve(n_items * k);

        for (int i = 0; i < n_items; i++) {
            auto emb = mf.get_item_embedding(i);
            item_embeddings.insert(
                item_embeddings.end(),
                emb.begin(),
                emb.end());
        }

        // 构建IVF索引
        int nlist = std::min(1000, (int)std::sqrt(n_items));
        faiss::IndexFlatL2 quantizer(k);
        item_index.reset(
            new faiss::IndexIVFFlat(
                &quantizer, k, nlist));

        item_index->train(n_items, item_embeddings.data());
        item_index->add(n_items, item_embeddings.data());
    }
};
```

### 3.4 在线学习更新

```cpp
// 支持增量更新的推荐系统
class IncrementalRecommender {
public:
    // 在线更新用户embedding
    void update_user(
            int user_id,
            const std::vector<Rating>& new_ratings) {

        // 1. 获取当前用户embedding
        auto user_emb = mf.get_user_embedding(user_id);

        // 2. 使用新交互更新embedding
        auto updated_emb = compute_updated_embedding(
            user_emb, new_ratings);

        // 3. 更新用户embedding
        mf.set_user_embedding(user_id, updated_emb);
    }

    // 批量添加新物品
    void add_new_items(
            const std::vector<int>& item_ids,
            const std::vector<std::vector<float>>& embeddings) {

        // 1. 添加到MF模型
        for (size_t i = 0; i < item_ids.size(); i++) {
            mf.add_item(item_ids[i], embeddings[i]);
        }

        // 2. 重建索引（或使用增量索引）
        rebuild_item_index();
    }

private:
    std::vector<float> compute_updated_embedding(
            const std::vector<float>& old_emb,
            const std::vector<Rating>& ratings) {

        // 使用SGD更新
        std::vector<float> new_emb = old_emb;
        float lr = 0.01;

        for (const auto& r : ratings) {
            auto item_emb = mf.get_item_embedding(r.item_id);

            // 计算梯度
            float pred = dot_product(new_emb, item_emb);
            float error = r.rating - pred;

            for (size_t i = 0; i < new_emb.size(); i++) {
                new_emb[i] += lr * (error * item_emb[i] -
                                    0.01 * new_emb[i]);
            }
        }

        return new_emb;
    }
};
```

---

## 4. 实战项目四：实时向量检索服务

### 4.1 RESTful API服务

```cpp
// 使用Crow框架构建REST API
#include <crow.h>

class VectorSearchService {
public:
    VectorSearchService(int port = 8080) : port(port) {}

    void run() {
        crow::SimpleApp app;

        // 健康检查
        CROW_ROUTE(app, "/health").methods("GET"_method)
        ([this](const crow::request& req) {
            return crow::response(200, "OK");
        });

        // 添加向量
        CROW_ROUTE(app, "/vectors").methods("POST"_method)
        ([this](const crow::request& req) {
            auto body = crow::json::load(req.body);

            std::vector<float> vectors;
            // Parse vectors from JSON...

            idx_t n = parse_count(body);
            index->add(n, vectors.data());

            return crow::response(201, "Added");
        });

        // 搜索
        CROW_ROUTE(app, "/search").methods("POST"_method)
        ([this](const crow::request& req) {
            auto body = crow::json::load(req.body);

            std::vector<float> query = parse_query(body);
            int k = body["k"].i();

            std::vector<float> distances(k);
            std::vector<idx_t> labels(k);

            index->search(1, query.data(), k,
                         distances.data(), labels.data());

            // 构建JSON响应
            crow::json::wvalue result;
            for (int i = 0; i < k; i++) {
                result["results"][i]["id"] = labels[i];
                result["results"][i]["distance"] = distances[i];
            }

            return crow::response(result);
        });

        // 启动服务
        app.port(port).multithreaded().run();
    }

private:
    int port;
    std::unique_ptr<faiss::Index> index;
};
```

### 4.2 gRPC服务

```cpp
// 使用gRPC构建高性能服务
// vector_search.proto
syntax = "proto3";

service VectorSearch {
    rpc Search(SearchRequest) returns (SearchResponse);
    rpc Add(AddRequest) returns (AddResponse);
}

message SearchRequest {
    repeated float query = 1;
    int32 k = 2;
}

message SearchResponse {
    repeated float distances = 1;
    repeated int64 labels = 2;
}

message AddRequest {
    repeated float vectors = 1;
}

message AddResponse {
    int64 count = 1;
}

// C++实现
class VectorSearchImpl final : public VectorSearch::Service {
public:
    grpc::Status Search(
            grpc::ServerContext* context,
            const SearchRequest* request,
            SearchResponse* response) override {

        const auto& query = request->query();
        int k = request->k();

        std::vector<float> distances(k);
        std::vector<idx_t> labels(k);

        index->search(1, query.data(), k,
                     distances.data(), labels.data());

        // 填充响应
        for (int i = 0; i < k; i++) {
            response->add_distances(distances[i]);
            response->add_labels(labels[i]);
        }

        return grpc::Status::OK;
    }

private:
    std::unique_ptr<faiss::Index> index;
};
```

### 4.3 分片索引

```cpp
// 分布式分片索引
class ShardedIndex {
public:
    ShardedIndex(int n_shards, int d) : n_shards(n_shards), d(d) {
        for (int i = 0; i < n_shards; i++) {
            shards.push_back(create_shard(d));
        }
    }

    void add(const std::vector<float>& vectors) {
        size_t n = vectors.size() / d;

        for (size_t i = 0; i < n; i++) {
            const float* vec = vectors.data() + i * d;

            // 计算分片ID（基于向量hash）
            int shard_id = compute_shard(vec);
            shards[shard_id]->add(1, vec);
        }
    }

    void search(
            const std::vector<float>& queries,
            int k,
            std::vector<float>& distances,
            std::vector<idx_t>& labels) {

        size_t nq = queries.size() / d;

        // 并行搜索所有分片
        std::vector<std::vector<float>> shard_distances(n_shards);
        std::vector<std::vector<idx_t>> shard_labels(n_shards);

#pragma omp parallel for
        for (int s = 0; s < n_shards; s++) {
            shard_distances[s].resize(nq * k);
            shard_labels[s].resize(nq * k);

            shards[s]->search(
                nq, queries.data(), k,
                shard_distances[s].data(),
                shard_labels[s].data());
        }

        // 合并结果
        for (size_t q = 0; q < nq; q++) {
            std::vector<std::pair<float, idx_t>> candidates;

            for (int s = 0; s < n_shards; s++) {
                for (int i = 0; i < k; i++) {
                    idx_t idx = q * k + i;
                    candidates.push_back({
                        shard_distances[s][idx],
                        shard_labels[s][idx]
                    });
                }
            }

            // 排序并取top-K
            std::sort(candidates.begin(), candidates.end());

            for (int i = 0; i < k; i++) {
                distances[q * k + i] = candidates[i].first;
                labels[q * k + i] = candidates[i].second;
            }
        }
    }

private:
    int n_shards;
    int d;
    std::vector<std::unique_ptr<faiss::Index>> shards;

    int compute_shard(const float* vec) {
        // 简单的hash分片
        uint32_t hash = 0;
        for (int i = 0; i < d; i++) {
            hash ^= *(uint32_t*)(vec + i);
        }
        return hash % n_shards;
    }

    std::unique_ptr<faiss::Index> create_shard(int d) {
        // 每个分片使用HNSW
        return std::make_unique<faiss::IndexHNSWFlat>(d, 32);
    }
};
```

---

## 5. 实战项目五：时序向量检索

### 5.1 时间衰减的相似度搜索

```cpp
// 带时间衰减的向量检索
class TemporalVectorIndex {
public:
    TemporalVectorIndex(int d, float decay_rate = 0.1) :
        d(d), decay_rate(decay_rate) {

        // 使用IVF索引
        int nlist = 100;
        quantizer = std::make_unique<faiss::IndexFlatL2>(d);
        index = std::make_unique<faiss::IndexIVFFlat>(
            quantizer.get(), d, nlist);
    }

    void add_with_timestamp(
            const float* vectors,
            const idx_t* ids,
            const int64_t* timestamps,
            idx_t n) {

        // 添加向量到索引
        index->add(n, vectors);

        // 记录时间戳
        for (idx_t i = 0; i < n; i++) {
            vector_timestamps[ids[i]] = timestamps[i];
        }
    }

    void search_with_temporal_decay(
            const float* query,
            int64_t current_time,
            idx_t k,
            float* distances,
            idx_t* labels) {

        // 1. 标准搜索
        index->search(1, query, k, distances, labels);

        // 2. 应用时间衰减
        for (idx_t i = 0; i < k; i++) {
            idx_t id = labels[i];
            auto it = vector_timestamps.find(id);

            if (it != vector_timestamps.end()) {
                int64_t age = current_time - it->second;
                float decay = std::exp(-decay_rate * age / 86400.0);  // 天为单位
                distances[i] /= decay;  // 距离越小越好
            }
        }

        // 3. 重新排序
        std::vector<std::pair<float, idx_t>> pairs(k);
        for (idx_t i = 0; i < k; i++) {
            pairs[i] = {distances[i], labels[i]};
        }
        std::sort(pairs.begin(), pairs.end());

        for (idx_t i = 0; i < k; i++) {
            distances[i] = pairs[i].first;
            labels[i] = pairs[i].second;
        }
    }

private:
    int d;
    float decay_rate;
    std::unique_ptr<faiss::IndexFlatL2> quantizer;
    std::unique_ptr<faiss::IndexIVFFlat> index;
    std::unordered_map<idx_t, int64_t> vector_timestamps;
};
```

---

## 6. 性能监控与调优

### 6.1 搜索性能监控

```cpp
class SearchMonitor {
public:
    struct Metrics {
        std::atomic<uint64_t> total_searches{0};
        std::atomic<uint64_t> total_time_us{0};
        std::atomic<uint64_t> error_count{0};

        std::array<uint64_t, 10> latency_buckets{};  // 延迟分布
        std::mutex mutex;
    };

    void record_search(uint64_t duration_us) {
        metrics.total_searches++;
        metrics.total_time_us += duration_us;

        // 记录延迟分布
        int bucket = std::min(9, (int)std::log2(duration_us));
        metrics.latency_buckets[bucket]++;
    }

    void report() {
        uint64_t total = metrics.total_searches;
        uint64_t total_time = metrics.total_time_us;

        printf("=== Search Metrics ===\n");
        printf("Total searches: %lu\n", total);
        printf("Average latency: %.2f us\n", (double)total_time / total);
        printf("Error rate: %.2f%%\n",
               100.0 * metrics.error_count / total);

        printf("\nLatency distribution:\n");
        for (int i = 0; i < 10; i++) {
            printf("  %d us: %lu\n", 1 << i, metrics.latency_buckets[i]);
        }
    }

private:
    Metrics metrics;
};
```

### 6.2 动态参数调优

```cpp
// 自适应nprobe调整
class AdaptiveIndex {
public:
    AdaptiveIndex(faiss::IndexIVF* index) : index(index) {
        // 初始nprobe
        index->nprobe = 10;
        target_recall = 0.95;
    }

    void auto_tune_nprobe(
            const float* test_queries,
            const idx_t* ground_truth,
            idx_t nq,
            int k) {

        std::vector<int> nprobes = {1, 5, 10, 20, 50, 100};

        for (int nprobe : nprobes) {
            index->nprobe = nprobe;

            // 测试搜索
            std::vector<float> distances(nq * k);
            std::vector<idx_t> labels(nq * k);

            auto start = std::chrono::high_resolution_clock::now();
            index->search(nq, test_queries, k,
                         distances.data(), labels.data());
            auto end = std::chrono::high_resolution_clock::now();

            double time_ms = std::chrono::duration<double>(
                end - start).count() * 1000;

            // 计算recall
            float recall = compute_recall(
                nq, k, labels.data(), ground_truth);

            printf("nprobe=%d: recall=%.3f, time=%.2f ms\n",
                   nprobe, recall, time_ms);

            // 选择满足目标recall的最小nprobe
            if (recall >= target_recall) {
                optimal_nprobe = nprobe;
                break;
            }
        }

        // 恢复最优nprobe
        index->nprobe = optimal_nprobe;
    }

private:
    faiss::IndexIVF* index;
    float target_recall;
    int optimal_nprobe;
};
```

---

## 7. 生产级部署优化

### 7.1 内存池与分配器优化

```cpp
// 生产环境内存管理：专用的内存池
namespace production_memory {

// 对齐分配器（SIMD友好）
template<typename T, size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;

    T* allocate(size_t n) {
        // 使用aligned_alloc或posix_memalign
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, size_t) {
#ifdef _WIN32
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

// 大块内存池（减少碎片）
class VectorMemoryPool {
public:
    VectorMemoryPool(size_t pool_size_gb = 8) {
        pool_size = pool_size_gb * 1024 * 1024 * 1024;
        base_ptr = aligned_alloc(64, pool_size);

        if (!base_ptr) {
            throw std::runtime_error("Failed to allocate memory pool");
        }

        offset = 0;
    }

    ~VectorMemoryPool() {
        free(base_ptr);
    }

    // 分配对齐的内存块
    void* allocate(size_t size, size_t alignment = 64) {
        size_t aligned_offset = (offset + alignment - 1) & ~(alignment - 1);

        if (aligned_offset + size > pool_size) {
            // 池已满，回退到系统分配
            return aligned_alloc(alignment, size);
        }

        void* ptr = static_cast<uint8_t*>(base_ptr) + aligned_offset;
        offset = aligned_offset + size;
        return ptr;
    }

    // 重置池（清空所有分配）
    void reset() {
        offset = 0;
    }

    // 获取使用情况
    double utilization() const {
        return 100.0 * offset / pool_size;
    }

private:
    void* base_ptr;
    size_t pool_size;
    size_t offset;
};

// 索引专用的内存管理器
class IndexMemoryManager {
public:
    IndexMemoryManager(size_t max_memory_mb = 4096) {
        max_memory = max_memory_mb * 1024 * 1024;
        used_memory = 0;
    }

    // 分配索引内存（带追踪）
    void* allocate(size_t size, const char* label = "") {
        std::lock_guard<std::mutex> lock(mutex);

        if (used_memory + size > max_memory) {
            // 触发内存整理或拒绝分配
            cleanup();
            if (used_memory + size > max_memory) {
                throw std::runtime_error("Memory limit exceeded");
            }
        }

        void* ptr = aligned_alloc(64, size);
        allocations[ptr] = {size, label};
        used_memory += size;

        return ptr;
    }

    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = allocations.find(ptr);
        if (it != allocations.end()) {
            used_memory -= it->second.size;
            allocations.erase(it);
            free(ptr);
        }
    }

    // 打印内存使用报告
    void report() {
        printf("=== Memory Usage ===\n");
        printf("Used: %.2f MB / %.2f MB\n",
               used_memory / 1024.0 / 1024.0,
               max_memory / 1024.0 / 1024.0);

        // 按类别汇总
        std::map<std::string, size_t> by_label;
        for (auto& [ptr, info] : allocations) {
            by_label[info.label] += info.size;
        }

        for (auto& [label, size] : by_label) {
            printf("  %s: %.2f MB\n", label.c_str(),
                   size / 1024.0 / 1024.0);
        }
    }

private:
    void cleanup() {
        // 实现内存整理逻辑
        // 例如：释放缓存、压缩索引等
    }

    struct AllocationInfo {
        size_t size;
        std::string label;
    };

    std::mutex mutex;
    size_t max_memory;
    size_t used_memory;
    std::unordered_map<void*, AllocationInfo> allocations;
};
}
```

### 7.2 多线程并发优化

```cpp
// 生产级并发搜索架构
namespace production_concurrency {

// 线程池（任务窃取）
class WorkStealingThreadPool {
public:
    WorkStealingThreadPool(int n_threads = 0) {
        if (n_threads <= 0) {
            n_threads = std::thread::hardware_concurrency();
        }

        for (int i = 0; i < n_threads; i++) {
            workers.emplace_back([this, i] { worker_loop(i); });
        }
    }

    ~WorkStealingThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            shutdown = true;
        }
        cv.notify_all();

        for (auto& w : workers) {
            w.join();
        }
    }

    // 提交任务到本地队列
    template<typename F>
    auto submit(F&& f) -> std::future<decltype(f())> {
        using ReturnType = decltype(f());

        int tid = get_thread_id();
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::forward<F>(f));

        std::future<ReturnType> result = task->get_future();

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            local_queues[tid].push([task]() { (*task)(); });
        }

        cv.notify_one();
        return result;
    }

private:
    void worker_loop(int tid) {
        set_thread_id(tid);

        while (true) {
            std::function<void()> task;

            // 1. 从本地队列获取
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (shutdown) return;

                if (!local_queues[tid].empty()) {
                    task = local_queues[tid].front();
                    local_queues[tid].pop();
                }
            }

            // 2. 尝试从其他线程窃取
            if (!task) {
                for (int i = 0; i < local_queues.size(); i++) {
                    if (i == tid) continue;

                    std::lock_guard<std::mutex> lock(queue_mutex);
                    if (!local_queues[i].empty()) {
                        task = local_queues[i].front();
                        local_queues[i].pop();
                        break;
                    }
                }
            }

            if (task) {
                task();
            } else {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [this, tid] {
                    return shutdown || !local_queues[tid].empty();
                });
            }
        }
    }

    static thread_local int thread_id;

    int get_thread_id() const {
        return thread_id;
    }

    void set_thread_id(int id) {
        thread_id = id;
    }

    std::vector<std::thread> workers;
    std::vector<std::queue<std::function<void()>>> local_queues;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::mutex mutex;
    bool shutdown = false;
};

// 并行批量搜索（优化版）
class ParallelSearchEngine {
public:
    ParallelSearchEngine(faiss::Index* index, int n_threads = 0)
        : index(index), pool(n_threads) {

        if (n_threads <= 0) {
            n_threads = std::thread::hardware_concurrency();
        }

        // 为每个线程准备查询缓冲区
        thread_query_buffers.resize(n_threads);
    }

    void parallel_search(
            const float* queries,
            idx_t nq,
            idx_t k,
            float* distances,
            idx_t* labels) {

        // 按批次并行处理
        const idx_t batch_size = 128;

        std::vector<std::future<void>> futures;

        for (idx_t q_start = 0; q_start < nq; q_start += batch_size) {
            idx_t q_end = std::min(q_start + batch_size, nq);
            idx_t batch_nq = q_end - q_start;

            futures.push_back(pool.submit([this, queries, nq,
                                           q_start, q_end, batch_nq, k,
                                           distances, labels]() {
                // 获取线程本地缓冲区
                int tid = get_thread_id();

                std::vector<float>& local_distances =
                    thread_query_buffers[tid].distances;
                std::vector<idx_t>& local_labels =
                    thread_query_buffers[tid].labels;

                local_distances.resize(batch_nq * k);
                local_labels.resize(batch_nq * k);

                // 执行搜索
                index->search(
                    batch_nq,
                    queries + q_start * index->d,
                    k,
                    local_distances.data(),
                    local_labels.data());

                // 复制结果
                std::copy(local_distances.begin(),
                         local_distances.end(),
                         distances + q_start * k);
                std::copy(local_labels.begin(),
                         local_labels.end(),
                         labels + q_start * k);
            }));
        }

        // 等待所有批次完成
        for (auto& f : futures) {
            f.get();
        }
    }

private:
    faiss::Index* index;
    WorkStealingThreadPool pool;

    struct ThreadBuffers {
        std::vector<float> distances;
        std::vector<idx_t> labels;
    };

    std::vector<ThreadBuffers> thread_query_buffers;
};
}
```

### 7.3 索引预热策略

```cpp
// 生产环境索引预热
namespace production_warmup {

class IndexWarmup {
public:
    // 冷启动预热：填充缓存
    static void warmup_index(
            faiss::Index* index,
            const float* sample_queries,
            idx_t nq,
            int iterations = 100) {

        printf("Warming up index with %ld queries...\n", nq);

        int k = 10;
        std::vector<float> distances(nq * k);
        std::vector<idx_t> labels(nq * k);

        // 执行多次搜索以预热CPU缓存
        for (int iter = 0; iter < iterations; iter++) {
            index->search(nq, sample_queries, k,
                         distances.data(), labels.data());
        }

        printf("Warmup complete.\n");
    }

    // 针对IVF索引的预热
    static void warmup_ivf_index(
            faiss::IndexIVF* ivf_index,
            const float* sample_queries,
            idx_t nq) {

        // 预热倒排列表
        printf("Warming up IVF inverted lists...\n");

        int nlist = ivf_index->nlist;
        int nprobe = ivf_index->nprobe;

        // 生成访问所有列表的查询
        for (int list_id = 0; list_id < nlist; list_id++) {
            // 触发列表访问
            ivf_index->invlists->get_codes(list_id);
            ivf_index->invlists->get_ids(list_id);
        }

        // 预热量化器
        if (auto* ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(ivf_index)) {
            warmup_pq_quantizer(ivfpq, sample_queries, nq);
        }
    }

    // PQ量化器预热
    static void warmup_pq_quantizer(
            faiss::IndexIVFPQ* ivfpq,
            const float* queries,
            idx_t nq) {

        printf("Warming up PQ quantizer...\n");

        auto& pq = ivfpq->pq;
        int M = pq.M;
        int d = pq.d;

        // 预计算所有子量化器的查找表
        std::vector<float> tables(M * pq.ksub * d);

        for (idx_t q = 0; q < std::min(nq, (idx_t)1000); q++) {
            const float* query = queries + q * d;

            // 计算查找表（这会预热PQ计算）
            ivfpq->compute_quantized_distances(query, tables.data());
        }
    }

    // HNSW索引预热
    static void warmup_hnsw_index(
            faiss::IndexHNSW* hnsw_index,
            const float* sample_queries,
            idx_t nq) {

        printf("Warming up HNSW index...\n");

        auto& hnsw = hnsw_index->hnsw;
        int k = 10;

        std::vector<float> distances(nq * k);
        std::vector<idx_t> labels(nq * k);

        // 触发图遍历预热
        for (int iter = 0; iter < 10; iter++) {
            hnsw_index->search(nq, sample_queries, k,
                             distances.data(), labels.data());
        }

        // 预热entry point
        hnsw.prefetch_entry_points();
    }

    // 综合预热策略
    static void comprehensive_warmup(
            faiss::Index* index,
            const float* sample_queries,
            idx_t nq) {

        printf("=== Comprehensive Index Warmup ===\n");

        auto start = std::chrono::high_resolution_clock::now();

        // 通用预热
        warmup_index(index, sample_queries, nq, 50);

        // 特定类型预热
        if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index)) {
            warmup_ivf_index(ivf, sample_queries, nq);
        }

        if (auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index)) {
            warmup_hnsw_index(hnsw, sample_queries, nq);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(
            end - start).count();

        printf("Warmup completed in %.2f seconds\n", elapsed);
    }
};
}
```

### 7.4 GPU生产优化

```cpp
// 生产环境GPU加速
namespace production_gpu {

// GPU内存池
class GPUMemoryPool {
public:
    GPUMemoryPool(int device_id = 0, size_t pool_size_gb = 8) {
        CUDA_CHECK(cudaSetDevice(device_id));

        pool_size = pool_size_gb * 1024 * 1024 * 1024;
        CUDA_CHECK(cudaMalloc(&base_ptr, pool_size));

        offset = 0;
    }

    ~GPUMemoryPool() {
        CUDA_CHECK(cudaFree(base_ptr));
    }

    void* allocate(size_t size) {
        size_t aligned_offset = (offset + 511) & ~511;  // 512字节对齐

        if (aligned_offset + size > pool_size) {
            // 回退到正常分配
            void* ptr;
            CUDA_CHECK(cudaMalloc(&ptr, size));
            return ptr;
        }

        void* ptr = static_cast<uint8_t*>(base_ptr) + aligned_offset;
        offset = aligned_offset + size;
        return ptr;
    }

private:
    void* base_ptr;
    size_t pool_size;
    size_t offset;
};

// 多GPU并行搜索
class MultiGPUSearchEngine {
public:
    MultiGPUSearchEngine(
            const std::vector<int>& device_ids,
            faiss::Index* cpu_index) {

        for (int device_id : device_ids) {
            // 将索引克隆到每个GPU
            faiss::GpuClonerOptions options;
            options.allInGpu = true;

            cudaSetDevice(device_id);
            auto gpu_index =
                dynamic_cast<faiss::GpuIndex*>(
                    faiss::gpu_clone_index(cpu_index, device_id, options));

            gpu_indices.push_back(gpu_index);
            gpu_streams.emplace_back(device_id);
        }
    }

    ~MultiGPUSearchEngine() {
        for (auto* idx : gpu_indices) {
            delete idx;
        }
    }

    void parallel_search(
            const float* queries,
            idx_t nq,
            idx_t k,
            float* distances,
            idx_t* labels) {

        int n_gpus = gpu_indices.size();
        idx_t q_per_gpu = (nq + n_gpus - 1) / n_gpus;

        std::vector<std::future<void>> futures;

        for (int gpu_id = 0; gpu_id < n_gpus; gpu_id++) {
            idx_t q_start = gpu_id * q_per_gpu;
            idx_t q_end = std::min(q_start + q_per_gpu, nq);

            if (q_start >= nq) break;

            futures.push_back(std::async(std::launch::async, [this,
                                                             gpu_id,
                                                             queries,
                                                             q_start,
                                                             q_end,
                                                             k,
                                                             distances,
                                                             labels]() {
                cudaSetDevice(gpu_id);
                cudaStream_t stream = gpu_streams[gpu_id];

                idx_t batch_nq = q_end - q_start;
                idx_t offset = q_start * k;

                gpu_indices[gpu_id]->search(
                    batch_nq,
                    queries + q_start * gpu_indices[0]->d,
                    k,
                    distances + offset,
                    labels + offset);
            }));
        }

        for (auto& f : futures) {
            f.get();
        }
    }

private:
    std::vector<faiss::GpuIndex*> gpu_indices;
    std::vector<cudaStream_t> gpu_streams;
};
}
```

### 7.5 故障容错与高可用

```cpp
// 生产环境容错机制
namespace production_fault_tolerance {

// 索引快照与恢复
class IndexSnapshotManager {
public:
    // 创建快照
    void create_snapshot(
            faiss::Index* index,
            const std::string& snapshot_path) {

        printf("Creating snapshot: %s\n", snapshot_path.c_str());

        // 1. 保存索引
        std::string index_path = snapshot_path + ".index";
        {
            faiss::IOWriter* writer =
                new faiss::IOFile(index_path.c_str(), "wb");
            faiss::write_index(index, writer);
            delete writer;
        }

        // 2. 计算checksum
        std::string checksum_path = snapshot_path + ".checksum";
        std::string checksum = compute_file_checksum(index_path);
        write_file(checksum_path, checksum);

        // 3. 保存元数据
        std::string meta_path = snapshot_path + ".meta";
        save_metadata(index, meta_path);

        printf("Snapshot created successfully.\n");
    }

    // 恢复快照
    std::unique_ptr<faiss::Index> restore_snapshot(
            const std::string& snapshot_path) {

        printf("Restoring snapshot: %s\n", snapshot_path.c_str());

        std::string index_path = snapshot_path + ".index";
        std::string checksum_path = snapshot_path + ".checksum";

        // 1. 验证checksum
        std::string expected_checksum = read_file(checksum_path);
        std::string actual_checksum = compute_file_checksum(index_path);

        if (expected_checksum != actual_checksum) {
            throw std::runtime_error("Checksum mismatch! Snapshot corrupted.");
        }

        // 2. 加载索引
        faiss::IOReader* reader =
            new faiss::IOFile(index_path.c_str(), "rb");
        faiss::Index* index = faiss::read_index(reader);
        delete reader;

        printf("Snapshot restored successfully.\n");
        return std::unique_ptr<faiss::Index>(index);
    }

private:
    std::string compute_file_checksum(const std::string& path) {
        // 计算文件SHA256
        // ...
        return "sha256_checksum";
    }
};

// 主备切换机制
class PrimaryStandbyIndex {
public:
    PrimaryStandbyIndex(
            faiss::Index* primary,
            faiss::Index* standby)
        : primary_index(primary),
          standby_index(standby),
          is_primary_healthy(true) {

        // 启动健康检查线程
        health_check_thread =
            std::thread(&PrimaryStandbyIndex::health_check_loop, this);
    }

    ~PrimaryStandbyIndex() {
        shutdown = true;
        if (health_check_thread.joinable()) {
            health_check_thread.join();
        }
    }

    void search(
            idx_t nq,
            const float* queries,
            idx_t k,
            float* distances,
            idx_t* labels) {

        if (is_primary_healthy) {
            try {
                primary_index->search(nq, queries, k,
                                    distances, labels);
                return;
            } catch (const std::exception& e) {
                fprintf(stderr, "Primary search failed: %s\n", e.what());
                is_primary_healthy = false;
                // 触发告警
                alert_primary_failure();
            }
        }

        // 降级到备用
        standby_index->search(nq, queries, k, distances, labels);
    }

private:
    void health_check_loop() {
        while (!shutdown) {
            std::this_thread::sleep_for(std::chrono::seconds(5));

            if (is_primary_healthy) {
                // 执行健康检查
                if (!check_primary_health()) {
                    is_primary_healthy = false;
                    alert_primary_failure();
                }
            } else {
                // 尝试恢复主索引
                if (attempt_primary_recovery()) {
                    is_primary_healthy = true;
                    alert_primary_recovery();
                }
            }
        }
    }

    bool check_primary_health() {
        try {
            // 执行测试查询
            std::vector<float> test_query(primary_index->d, 0);
            std::vector<float> distances(1);
            std::vector<idx_t> labels(1);

            primary_index->search(1, test_query.data(), 1,
                                distances.data(), labels.data());
            return true;
        } catch (...) {
            return false;
        }
    }

    void alert_primary_failure() {
        // 发送告警（Prometheus、日志等）
        printf("ALERT: Primary index failure detected!\n");
    }

    void alert_primary_recovery() {
        printf("INFO: Primary index recovered.\n");
    }

    faiss::Index* primary_index;
    faiss::Index* standby_index;
    std::atomic<bool> is_primary_healthy;
    std::thread health_check_thread;
    std::atomic<bool> shutdown{false};
};
}
```

### 7.6 性能基准测试框架

```cpp
// 生产环境性能测试
namespace production_benchmark {

struct BenchmarkConfig {
    int num_queries;
    int k;
    int num_threads;
    int warmup_iterations;
    int benchmark_iterations;
};

struct BenchmarkResults {
    double avg_latency_us;
    double p50_latency_us;
    double p95_latency_us;
    double p99_latency_us;
    double qps;
    double recall;
};

class SearchBenchmark {
public:
    static BenchmarkResults run(
            faiss::Index* index,
            const float* test_queries,
            const idx_t* ground_truth,
            const BenchmarkConfig& config) {

        printf("=== Running Search Benchmark ===\n");
        printf("Queries: %d, K: %d, Threads: %d\n",
               config.num_queries, config.k, config.num_threads);

        int d = index->d;
        int nq = config.num_queries;
        int k = config.k;

        // 准备结果缓冲区
        std::vector<float> distances(nq * k);
        std::vector<idx_t> labels(nq * k);

        // 预热
        printf("Warming up...\n");
        for (int i = 0; i < config.warmup_iterations; i++) {
            index->search(nq, test_queries, k,
                         distances.data(), labels.data());
        }

        // 基准测试
        printf("Benchmarking...\n");
        std::vector<double> latencies_us;

        latencies_us.reserve(config.benchmark_iterations);

        for (int iter = 0; iter < config.benchmark_iterations; iter++) {
            auto start = std::chrono::high_resolution_clock::now();

            index->search(nq, test_queries, k,
                         distances.data(), labels.data());

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_us = std::chrono::duration<double,
                std::micro>(end - start).count();

            latencies_us.push_back(elapsed_us);
        }

        // 计算统计
        BenchmarkResults results;
        results.avg_latency_us =
            std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) /
            latencies_us.size();

        std::sort(latencies_us.begin(), latencies_us.end());
        results.p50_latency_us = latencies_us[latencies_us.size() * 0.5];
        results.p95_latency_us = latencies_us[latencies_us.size() * 0.95];
        results.p99_latency_us = latencies_us[latencies_us.size() * 0.99];

        results.qps = 1000000.0 * nq / results.avg_latency_us;

        // 计算recall
        if (ground_truth) {
            results.recall = compute_recall(
                nq, k, labels.data(), ground_truth);
        }

        // 打印结果
        printf("\n=== Benchmark Results ===\n");
        printf("Avg Latency: %.2f us\n", results.avg_latency_us);
        printf("P50 Latency: %.2f us\n", results.p50_latency_us);
        printf("P95 Latency: %.2f us\n", results.p95_latency_us);
        printf("P99 Latency: %.2f us\n", results.p99_latency_us);
        printf("QPS: %.2f\n", results.qps);
        printf("Recall@%d: %.4f\n", k, results.recall);

        return results;
    }

private:
    static float compute_recall(
            int nq,
            int k,
            const idx_t* labels,
            const idx_t* ground_truth) {

        int correct = 0;
        int total = nq * k;

        for (int i = 0; i < nq * k; i++) {
            for (int j = 0; j < k; j++) {
                if (labels[i] == ground_truth[i * k + j]) {
                    correct++;
                    break;
                }
            }
        }

        return (float)correct / total;
    }
};
}
```

---

## 8. Kubernetes部署模式

### 8.1 容器化Faiss服务

```dockerfile
# Dockerfile for Faiss vector search service
FROM ubuntu:22.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    python3-dev \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制Faiss源码
COPY . /faiss
WORKDIR /faiss

# 编译Faiss（AVX2优化）
RUN mkdir build && cd build && \
    cmake -DFAISS_ENABLE_GPU=OFF \
          -DFAISS_ENABLE_PYTHON=ON \
          -DBUILD_TESTING=OFF \
          -DCMAKE_BUILD_TYPE=Release \
          -DFAISS_OPT_LEVEL=avx2 \
    .. && \
    make -j$(nproc) faiss && \
    make -j$(nproc) swigfaiss

# 安装Python依赖
RUN pip3 install flask gunicorn grpcio grpcio-tools prometheus_client

# 复制应用代码
COPY app/ /app
WORKDIR /app

# 暴露端口
EXPOSE 8080 9090

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 启动命令
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", \
     "--threads", "4", "--timeout", "120", "app:app"]
```

### 8.2 Kubernetes部署配置

```yaml
# faiss-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: faiss-vector-search
  labels:
    app: faiss-search
spec:
  replicas: 3
  selector:
    matchLabels:
      app: faiss-search
  template:
    metadata:
      labels:
        app: faiss-search
    spec:
      # CPU资源请求
      containers:
      - name: faiss-search
        image: faiss-search:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        resources:
          requests:
            cpu: "2000m"      # 2核CPU
            memory: "4Gi"     # 4GB内存
          limits:
            cpu: "4000m"      # 最大4核
            memory: "8Gi"     # 最大8GB
        env:
        - name: FAISS_INDEX_PATH
          value: "/data/index.faiss"
        - name: OMP_NUM_THREADS
          value: "4"  # OpenMP线程数
        - name: NUMA_AWARE
          value: "true"
        volumeMounts:
        - name: index-data
          mountPath: /data
        - name: shm
          mountPath: /dev/shm  # 共享内存（多线程）
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: index-data
        persistentVolumeClaim:
          claimName: faiss-index-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 1Gi
      # NUMA绑定（可选）
      nodeSelector:
        node.kubernetes.io/instance-type: "c5.4xlarge"
---
apiVersion: v1
kind: Service
metadata:
  name: faiss-search-service
spec:
  selector:
    app: faiss-search
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: faiss-index-pvc
spec:
  accessModes:
    - ReadOnlyMany  # 多个Pod只读共享
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
---
# 水平自动扩展
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: faiss-search-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: faiss-vector-search
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 8.3 分布式Faiss部署

```python
# distributed_faiss.py
# 分布式Faiss部署架构

import grpc
from concurrent import futures
import faiss
import numpy as np
import threading

# gRPC服务定义
class FaissShard:
    """单个分片服务"""
    def __init__(self, shard_id, index_path, port=50051):
        self.shard_id = shard_id
        self.port = port
        self.index = faiss.read_index(index_path)

        # 设置nprobe（可调整）
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 16

    def search(self, query_vectors, k):
        """本地搜索"""
        distances = np.zeros((len(query_vectors), k), dtype=np.float32)
        labels = np.zeros((len(query_vectors), k), dtype=np.int64)

        self.index.search(query_vectors, k, distances, labels)
        return distances, labels

class DistributedFaissCluster:
    """分布式Faiss集群"""
    def __init__(self, shard_addresses):
        """
        shard_addresses: list of (host, port) tuples
        """
        self.shards = []
        for addr in shard_addresses:
            channel = grpc.insecure_channel(f'{addr[0]}:{addr[1]}')
            stub = faiss_pb2_grpc.FaissServiceStub(channel)
            self.shards.append(stub)

    def distributed_search(self, queries, k):
        """分布式搜索：并行查询所有分片"""
        nq = queries.shape[0]
        num_shards = len(self.shards)

        # 每个分片返回k个结果
        shard_k = k * 2  # 获取更多候选

        # 并行查询
        with ThreadPoolExecutor(max_workers=num_shards) as executor:
            futures = []
            for shard in self.shards:
                future = executor.submit(
                    self._search_shard, shard, queries, shard_k)
                futures.append(future)

            # 收集结果
            all_results = []
            for future in futures:
                distances, labels = future.result()
                all_results.append((distances, labels))

        # 合并结果：取全局top-K
        return self._merge_results(all_results, nq, k)

    def _search_shard(self, shard, queries, k):
        """查询单个分片"""
        request = faiss_pb2.SearchRequest()
        request.vectors.extend(queries.flatten().tolist())
        request.nq = queries.shape[0]
        request.d = queries.shape[1]
        request.k = k

        response = shard.Search(request)

        distances = np.array(response.distances).reshape(queries.shape[0], k)
        labels = np.array(response.labels).reshape(queries.shape[0], k)

        return distances, labels

    def _merge_results(self, all_results, nq, k):
        """合并多个分片的结果"""
        final_distances = np.zeros((nq, k), dtype=np.float32)
        final_labels = np.zeros((nq, k), dtype=np.int64)

        for q in range(nq):
            # 收集所有候选
            candidates = []
            for shard_distances, shard_labels in all_results:
                for i in range(shard_distances.shape[1]):
                    candidates.append((
                        shard_distances[q, i],
                        shard_labels[q, i]
                    ))

            # 排序并取top-K
            candidates.sort(key=lambda x: x[0])
            for i in range(min(k, len(candidates))):
                final_distances[q, i] = candidates[i][0]
                final_labels[q, i] = candidates[i][1]

        return final_distances, final_labels

# 使用示例
def deploy_distributed_cluster():
    """部署分布式集群"""

    # 分片配置
    num_shards = 8
    shard_size = 1000000  # 每个分片100万向量

    # 创建分片
    for shard_id in range(num_shards):
        # 生成数据
        shard_data = np.random.rand(shard_size, 128).astype('float32')

        # 构建索引
        quantizer = faiss.IndexFlatL2(128)
        index = faiss.IndexIVFPQ(quantizer, 128, 100, 32, 8)
        index.train(shard_data[:100000])  # 训练
        index.add(shard_data)

        # 保存
        faiss.write_index(index, f'/data/shard_{shard_id}.faiss')

    # 启动分片服务（实际会在不同机器）
    shards = []
    for shard_id in range(num_shards):
        shard = FaissShard(
            shard_id,
            f'/data/shard_{shard_id}.faiss',
            port=50051 + shard_id
        )
        shards.append(shard)

    # 创建集群
    shard_addresses = [(f'shard-{i}', 50051 + i) for i in range(num_shards)]
    cluster = DistributedFaissCluster(shard_addresses)

    return cluster
```

---

## 9. 高级监控与可观测性

### 9.1 Prometheus指标导出

```cpp
// Prometheus指标收集
#include <prometheus/registry.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/counter.h>

class FaissMetrics {
public:
    FaissMetrics()
        : registry(std::make_shared<prometheus::Registry>()),

          // 搜索延迟直方图
          search_latency(prometheus::BuildHistogram()
                         .Name("faiss_search_latency_seconds")
                         .Help("Search latency in seconds")
                         .Buckets({0.001, 0.005, 0.01, 0.025, 0.05,
                                   0.1, 0.25, 0.5, 1.0, 2.5, 5.0})
                         .Register(*registry)),

          // QPS计数器
          search_qps(prometheus::BuildCounter()
                     .Name("faiss_search_queries_total")
                     .Help("Total number of searches")
                     .Register(*registry)),

          // 向量数量
          vector_count(prometheus::BuildGauge()
                       .Name("faiss_index_vector_count")
                       .Help("Number of vectors in index")
                       .Register(*registry)),

          // 内存使用
          memory_usage(prometheus::BuildGauge()
                       .Name("faiss_memory_bytes")
                       .Help("Memory usage in bytes")
                       .Register(*registry)),

          // 召回率
          recall_rate(prometheus::BuildGauge()
                      .Name("faiss_recall_rate")
                      .Help("Search recall rate")
                      .Register(*registry)) {}

    // 记录搜索指标
    void record_search(double latency_sec, int k, float recall) {
        search_latency.Observe(latency_sec);
        search_qps.Increment();

        // 定期更新向量数和内存
        if (search_qps.Value() % 1000 == 0) {
            update_index_stats();
        }
    }

    // 更新索引统计
    void update_index_stats(faiss::Index* index = nullptr) {
        if (index) {
            vector_count.Set(index->ntotal);

            // 估算内存使用
            size_t mem = estimate_index_memory(index);
            memory_usage.Set(mem);
        }
    }

    // 设置召回率
    void set_recall(float recall) {
        recall_rate.Set(recall);
    }

    // 生成指标文本
    std::string collect_metrics() {
        return registry->Collect();
    }

private:
    std::shared_ptr<prometheus::Registry> registry;
    prometheus::Histogram& search_latency;
    prometheus::Counter& search_qps;
    prometheus::Gauge& vector_count;
    prometheus::Gauge& memory_usage;
    prometheus::Gauge& recall_rate;

    size_t estimate_index_memory(faiss::Index* index) {
        // 简化的内存估算
        size_t base_size = index->ntotal * index->d * sizeof(float);

        if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index)) {
            // IVF索引额外开销
            base_size += ivf->nlist * index->d * sizeof(float);
            base_size += index->ntotal * sizeof(idx_t);
        }

        return base_size;
    }
};

// 集成到搜索服务
class MonitoredSearchEngine {
public:
    MonitoredSearchEngine(faiss::Index* index)
        : index(index), metrics() {}

    void search(const float* queries, idx_t nq, idx_t k,
                float* distances, idx_t* labels) {

        auto start = std::chrono::high_resolution_clock::now();

        // 执行搜索
        index->search(nq, queries, k, distances, labels);

        auto end = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double>(end - start).count();

        // 记录指标
        for (idx_t i = 0; i < nq; i++) {
            metrics.record_search(latency / nq, k, 0.95f);
        }

        metrics.update_index_stats(index);
    }

    std::string get_metrics() {
        return metrics.collect_metrics();
    }

private:
    faiss::Index* index;
    FaissMetrics metrics;
};
```

### 9.2 分布式追踪（OpenTelemetry）

```cpp
// OpenTelemetry集成
#include <opentelemetry/trace/provider.h>
#include <opentelemetry/trace/tracer.h>
#include <opentelemetry/exporters/otlp/otlp_http_exporter.h>

class TracedSearchEngine {
public:
    TracedSearchEngine(faiss::Index* index) : index(index) {
        // 初始化OpenTelemetry
        auto exporter = std::make_unique<
            opentelemetry::exporter::otlp::OtlpHttpExporter>();

        auto provider = opentelemetry::trace::Provider::GetTracerProvider();
        tracer = provider->GetTracer("faiss-search", "1.0.0");
    }

    void traced_search(const float* queries, idx_t nq, idx_t k,
                      float* distances, idx_t* labels) {

        auto span = tracer->StartSpan("faiss_search");
        auto scope = tracer->WithActiveSpan(span);

        // 添加属性
        span->SetAttribute("num_queries", nq);
        span->SetAttribute("k", k);
        span->SetAttribute("dimension", index->d);
        span->SetAttribute("index_type", index_type_name());

        // 如果是IVF索引，记录nprobe
        if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index)) {
            span->SetAttribute("nprobe", ivf->nprobe);
            span->SetAttribute("nlist", ivf->nlist);
        }

        // 执行搜索
        auto search_start = std::chrono::high_resolution_clock::now();

        index->search(nq, queries, k, distances, labels);

        auto search_end = std::chrono::high_resolution_clock::now();
        double search_ms = std::chrono::duration<double,
            std::milli>(search_end - search_start).count();

        span->SetAttribute("search_duration_ms", search_ms);

        span->End();
    }

private:
    faiss::Index* index;
    std::shared_ptr<opentelemetry::trace::Tracer> tracer;

    std::string index_type_name() {
        if (dynamic_cast<faiss::IndexFlat*>(index)) {
            return "IndexFlat";
        } else if (dynamic_cast<faiss::IndexIVF*>(index)) {
            return "IndexIVF";
        } else if (dynamic_cast<faiss::IndexHNSW*>(index)) {
            return "IndexHNSW";
        }
        return "Unknown";
    }
};
```

---

## 10. CI/CD流水线

### 10.1 自动化测试

```yaml
# .github/workflows/faiss-test.yml
name: Faiss CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    runs-on: [self-hosted, x64, linux]

    strategy:
      matrix:
        simd_level: [avx2, avx512, neon]
        compiler: [gcc-11, clang-14]

    steps:
    - uses: actions/checkout@v3

    - name: Configure CMake
      run: |
        cmake -B build -S . \
          -DFAISS_ENABLE_GPU=OFF \
          -DFAISS_OPT_LEVEL=${{ matrix.simd_level }} \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_COMPILER=${{ matrix.compiler }}

    - name: Build
      run: |
        cmake --build build --target faiss -j$(nproc)
        cmake --build build --target swigfaiss -j$(nproc)

    - name: Run Unit Tests
      run: |
        cd build
        ctest --output-on-failure -j$(nproc)

    - name: Benchmark Tests
      run: |
        python3 benchmarks/run_benchmarks.py \
          --simd-level=${{ matrix.simd_level }} \
          --output=results_${{ matrix.simd_level }}_${{ matrix.compiler }}.json

    - name: Upload Results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results-${{ matrix.simd_level }}-${{ matrix.compiler }}
        path: results_*.json

  memory-sanity-test:
    runs-on: ubuntu-latest
    container: ubuntu:22.04

    steps:
    - uses: actions/checkout@v3

    - name: Install Valgrind
      run: |
        apt-get update
        apt-get install -y valgrind cmake g++

    - name: Build with Debug Symbols
      run: |
        cmake -B build -S . \
          -DCMAKE_BUILD_TYPE=Debug \
          -DFAISS_ENABLE_GPU=OFF
        cmake --build build --target faiss_test

    - name: Run Valgrind Memory Check
      run: |
        valgrind --leak-check=full \
                 --show-leak-kinds=all \
                 --track-origins=yes \
                 --error-exitcode=1 \
                 ./build/tests/faiss_test

  performance-regression-test:
    runs-on: [self-hosted, high-perf]

    steps:
    - uses: actions/checkout@v3

    - name: Build Faiss
      run: |
        cmake -B build -S . \
          -DFAISS_OPT_LEVEL=avx512 \
          -DCMAKE_BUILD_TYPE=Release
        cmake --build build --target faiss -j$(nproc)

    - name: Run Performance Baseline
      run: |
        python3 tests/benchmark_baseline.py \
          --baseline=baselines/nightly.json

    - name: Check for Regression
      run: |
        python3 scripts/check_regression.py \
          --current=results/current.json \
          --baseline=baselines/nightly.json \
          --threshold=0.05  # 5% regression threshold

  deploy-staging:
    needs: [build-and-test, memory-sanity-test, performance-regression-test]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'

    steps:
    - name: Deploy to Staging
      run: |
        kubectl set image deployment/faiss-staging \
          faiss-search=ghcr.io/${{ github.repository }}:${{ github.sha }} \
          --namespace=staging

    - name: Smoke Tests
      run: |
        python3 tests/smoke_tests.py --endpoint=https://staging.api.com

  deploy-production:
    needs: [deploy-staging]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Deploy to Production
      run: |
        kubectl set image deployment/faiss-production \
          faiss-search=ghcr.io/${{ github.repository }}:${{ github.sha }} \
          --namespace=production

    - name: Monitor Rollout
      run: |
        kubectl rollout status deployment/faiss-production \
          --namespace=production --timeout=5m
```

### 10.2 蓝绿部署

```bash
#!/bin/bash
# blue_green_deploy.sh

set -e

NEW_VERSION=$1
CURRENT_VERSION=$(kubectl get service faiss-service -o jsonpath='{.spec.selector.version}')

echo "Current version: $CURRENT_VERSION"
echo "New version: $NEW_VERSION"

# 1. 部署新版本（green环境）
kubectl apply -f deployment-green.yaml --namespace=production
sed "s/{{VERSION}}/$NEW_VERSION/g" deployment-green.yaml | kubectl apply -f -

# 2. 等待green环境就绪
echo "Waiting for green deployment to be ready..."
kubectl wait --for=condition=available deployment/faiss-green \
  --namespace=production --timeout=5m

# 3. 运行金丝雀测试
echo "Running canary tests..."
python3 canary_test.py --endpoint=https://green.api.com

if [ $? -eq 0 ]; then
  # 4. 测试通过，切换流量
  echo "Canary tests passed. Switching traffic..."

  # 逐步切换流量（10% -> 50% -> 100%）
  for percentage in 10 50 100; do
    kubectl patch service faiss-service -p '{"spec":{"selector":{"version":"green"}}}' \
      --namespace=production
    sleep 60  # 观察期

    # 检查错误率
    ERROR_RATE=$(kubectl get service faiss-service --namespace=production \
      -o jsonpath='{.status.error_rate}')

    if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
      echo "High error rate: $ERROR_RATE. Rolling back..."
      kubectl patch service faiss-service -p '{"spec":{"selector":{"version":"blue"}}}'
      exit 1
    fi
  done

  echo "Deployment successful!"

  # 5. 清理blue环境
  kubectl delete deployment faiss-blue --namespace=production

else
  echo "Canary tests failed. Rolling back..."
  kubectl delete deployment faiss-green --namespace=production
  exit 1
fi
```

---

## 11. 第15天总结

### 实战要点

1. **图像搜索**：深度特征 + IVFPQ索引
2. **文本搜索**：BERT embeddings + 内积索引
3. **推荐系统**：矩阵分解 + 相似物品检索
4. **实时服务**：REST/gRPC API + 分片索引
5. **时序检索**：时间衰减 + 重新排序

### 生产级优化

1. **内存管理**：对齐分配器、内存池、追踪管理
2. **并发优化**：工作窃取线程池、并行批量搜索
3. **索引预热**：冷启动优化、缓存预热策略
4. **GPU加速**：GPU内存池、多GPU并行
5. **故障容错**：快照恢复、主备切换
6. **性能测试**：基准测试框架、延迟分析

### 最佳实践

1. **特征提取**：批量处理，多线程优化
2. **索引选择**：根据数据规模选择合适索引
3. **参数调优**：自动调优工具
4. **监控告警**：完善的性能监控
5. **扩展性**：分片、分布式设计
6. **高可用**：主备切换、快照恢复
7. **资源管理**：内存池、连接池
8. **性能基准**：定期基准测试

### 下一步

第16天将学习**跨平台优化详解**，包括ARM NEON、AVX-512等不同平台的SIMD优化技巧。

---

## 练习题

1. 实现一个完整的图像搜索引擎
2. 构建文本语义搜索系统
3. 开发实时向量检索API服务
4. 实现动态参数调优机制
5. **实现生产级内存管理器**
6. **构建主备高可用索引系统**
7. **编写完整的性能基准测试**

## 扩展阅读

- [Milvus: 开源向量数据库](https://milvus.io/)
- [Weaviate: 开源向量搜索引擎](https://weaviate.io/)
- [Faiss生产部署最佳实践](https://github.com/facebookresearch/faiss/wiki)
- [C++内存管理最佳实践](https://en.cppreference.com/w/cpp/memory)
