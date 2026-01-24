# Faiss深度学习课程 - 第25天：多模态向量检索

## 课程概述

第25天探讨多模态（Multimodal）向量检索技术。现代应用常常需要同时处理文本、图像、音频、视频等多种模态数据，并支持跨模态检索。本课程全面解析多模态嵌入、融合策略和检索技术。

## 学习目标

- 理解多模态检索的基本概念
- 掌握不同模态的嵌入方法
- 学习模态融合策略
- 理解跨模态检索技术
- 掌握多模态索引设计
- 实践CLIP等前沿模型的应用

---

## 第一部分：多模态检索基础

### 1.1 什么是多模态检索

```cpp
// 多模态检索系统架构
class MultimodalSearchSystem {
public:
    // 模态类型
    enum class ModalityType {
        TEXT,
        IMAGE,
        AUDIO,
        VIDEO,
        GRAPH_3D,       // 3D模型
        TABULAR,        // 表格数据
        BEHAVIORAL      // 用户行为
    };

    // 多模态数据
    struct MultimodalData {
        uint64_t id;
        std::unordered_map<ModalityType, std::vector<float>> embeddings;

        // 原始数据引用
        std::string text;
        std::string image_path;
        std::string audio_path;

        void print() const {
            printf("Data ID: %lu\n", id);
            printf("Modalities: %zu\n", embeddings.size());
            for (const auto& [modality, emb] : embeddings) {
                printf("  %d: [%zu dims]\n",
                       static_cast<int>(modality), emb.size());
            }
        }
    };

    // 跨模态查询
    struct CrossModalQuery {
        ModalityType query_type;
        std::vector<float> query_embedding;
        ModalityType target_type;  // 要检索的目标模态

        // 例如：用文本检索图像
        // query_type = TEXT, target_type = IMAGE
    };

    // 系统配置
    struct SystemConfig {
        std::unordered_map<ModalityType, size_t> embedding_dims;
        bool use_joint_embedding;      // 是否使用联合嵌入空间
        bool use_modality_specific;    // 是否使用模态特定索引
        double fusion_weight;          // 融合权重

        void print() const {
            printf("Multimodal System Config:\n");
            printf("  Joint embedding: %s\n",
                   use_joint_embedding ? "Yes" : "No");
            printf("  Modality-specific: %s\n",
                   use_modality_specific ? "Yes" : "No");
            printf("  Fusion weight: %.2f\n", fusion_weight);
        }
    };
};
```

### 1.2 挑战与问题

```cpp
// 多模态检索挑战分析
class MultimodalChallenges {
public:
    // 模态鸿沟（Modality Gap）
    static double analyze_modality_gap(
        const std::vector<std::vector<float>>& text_embeds,
        const std::vector<std::vector<float>>& image_embeds) {

        // 计算文本和图像嵌入在空间中的分布差异
        double text_mean = compute_mean_norm(text_embeds);
        double image_mean = compute_mean_norm(image_embeds);

        double gap = std::abs(text_mean - image_mean);

        printf("Modality Gap Analysis:\n");
        printf("  Text mean norm: %.4f\n", text_mean);
        printf("  Image mean norm: %.4f\n", image_mean);
        printf("  Gap: %.4f\n", gap);

        return gap;
    }

    // 对齐问题
    static void analyze_alignment_quality(
        const std::vector<std::pair<std::string, std::string>>& pairs,
        // text-image pairs
        const std::vector<std::vector<float>>& text_embeds,
        const std::vector<std::vector<float>>& image_embeds) {

        // 计算配对数据的相似度
        std::vector<float> similarities;
        for (size_t i = 0; i < pairs.size(); i++) {
            float sim = cosine_similarity(
                text_embeds[i].data(),
                image_embeds[i].data(),
                text_embeds[i].size()
            );
            similarities.push_back(sim);
        }

        // 统计
        double mean_sim = std::accumulate(
            similarities.begin(), similarities.end(), 0.0) / similarities.size();

        double aligned_ratio = std::count_if(
            similarities.begin(), similarities.end(),
            [](float s) { return s > 0.5; }) / (double)similarities.size();

        printf("Alignment Quality:\n");
        printf("  Mean similarity: %.4f\n", mean_sim);
        printf("  Aligned ratio (>0.5): %.2f%%\n", aligned_ratio * 100);
    }

    // 维度不一致问题
    static void analyze_dimension_mismatch(
        const std::unordered_map<MultimodalSearchSystem::ModalityType,
                                 size_t>& dims) {

        printf("\nDimension Mismatch Analysis:\n");
        for (const auto& [modality, dim] : dims) {
            printf("  %d: %zu dims\n",
                   static_cast<int>(modality), dim);
        }

        // 检查是否需要投影到公共空间
        bool all_same = std::all_of(dims.begin(), dims.end(),
            [&](const auto& p) {
                return p.second == dims.begin()->second;
            });

        if (!all_same) {
            printf("  Warning: Dimensions differ, need projection!\n");
        }
    }

private:
    static double compute_mean_norm(
        const std::vector<std::vector<float>>& embeddings) {

        double sum = 0.0;
        for (const auto& emb : embeddings) {
            sum += std::sqrt(std::inner_product(
                emb.begin(), emb.end(), emb.begin(), 0.0f));
        }
        return sum / embeddings.size();
    }

    static float cosine_similarity(const float* a, const float* b, size_t d) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < d; i++) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
};
```

---

## 第二部分：模态嵌入

### 2.1 文本嵌入

```cpp
// 文本嵌入生成器
class TextEmbeddingGenerator {
public:
    // 嵌入模型类型
    enum class ModelType {
        BERT_BASE,
        BERT_LARGE,
        ROBERTA,
        SENTENCE_TRANSFORMER,
        OPENAI_EMBEDDING,
        COHERE_EMBEDDING
    };

    TextEmbeddingGenerator(ModelType model) : model_type(model) {
        // 初始化模型
    }

    // 生成嵌入
    std::vector<float> generate(const std::string& text) {
        switch (model_type) {
            case ModelType::BERT_BASE:
                return encode_bert(text, 768);

            case ModelType::SENTENCE_TRANSFORMER:
                return encode_sentence_transformer(text, 384);

            case ModelType::OPENAI_EMBEDDING:
                return encode_openai(text, 1536);

            default:
                return encode_bert(text, 768);
        }
    }

    // 批量生成
    std::vector<std::vector<float>> generate_batch(
        const std::vector<std::string>& texts) {

        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(texts.size());

        for (const auto& text : texts) {
            embeddings.push_back(generate(text));
        }

        return embeddings;
    }

private:
    ModelType model_type;

    std::vector<float> encode_bert(const std::string& text, size_t dim) {
        // 简化：实际应调用BERT模型
        std::vector<float> embedding(dim);
        for (size_t i = 0; i < dim; i++) {
            embedding[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        return embedding;
    }

    std::vector<float> encode_sentence_transformer(
        const std::string& text, size_t dim) {
        // Sentence-BERT等
        std::vector<float> embedding(dim);
        // ...
        return embedding;
    }

    std::vector<float> encode_openai(const std::string& text, size_t dim) {
        // OpenAI API调用
        std::vector<float> embedding(dim);
        // ...
        return embedding;
    }
};
```

### 2.2 图像嵌入

```cpp
// 图像嵌入生成器
class ImageEmbeddingGenerator {
public:
    // 模型类型
    enum class ModelType {
        RESNET,
        VGG,
        EFFICIENTNET,
        VIT,               // Vision Transformer
        CLIP_VISION,       // CLIP视觉编码器
        DINO               // DINO自监督
    };

    ImageEmbeddingGenerator(ModelType model) : model_type(model) {}

    // 从图像生成嵌入
    std::vector<float> generate(const std::string& image_path) {
        // 加载图像
        cv::Mat image = cv::imread(image_path);

        switch (model_type) {
            case ModelType::RESNET:
                return encode_resnet(image);

            case ModelType::CLIP_VISION:
                return encode_clip(image);

            case ModelType::VIT:
                return encode_vit(image);

            default:
                return encode_resnet(image);
        }
    }

    // 从原始像素生成嵌入
    std::vector<float> generate_from_pixels(
        const uint8_t* pixels,
        int width,
        int height,
        int channels) {

        cv::Mat image(height, width,
                     channels == 3 ? CV_8UC3 : CV_8UC1,
                     const_cast<uint8_t*>(pixels));

        return generate_resnet_features(image);
    }

private:
    ModelType model_type;

    std::vector<float> encode_resnet(const cv::Mat& image) {
        // 预处理
        cv::Mat processed;
        cv::resize(image, processed, cv::Size(224, 224));

        // 提取特征（简化）
        size_t dim = 2048;  // ResNet-50
        std::vector<float> features(dim);

        // 实际应调用深度学习模型
        for (size_t i = 0; i < dim; i++) {
            features[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        return features;
    }

    std::vector<float> encode_clip(const cv::Mat& image) {
        // CLIP视觉编码器
        size_t dim = 768;  // ViT-B/32
        std::vector<float> features(dim);
        // ...
        return features;
    }

    std::vector<float> encode_vit(const cv::Mat& image) {
        // Vision Transformer
        size_t dim = 768;
        std::vector<float> features(dim);
        // ...
        return features;
    }

    std::vector<float> generate_resnet_features(const cv::Mat& image) {
        // ResNet特征提取
        return encode_resnet(image);
    }
};
```

### 2.3 联合嵌入（CLIP风格）

```cpp
// CLIP风格的联合嵌入生成器
class CLIPStyleJointEmbedding {
public:
    // CLIP模型变体
    enum class CLIPModel {
        ViT_B_32,      // Visual Transformer B/32
        ViT_B_16,
        ViT_L_14,
        RN50,          // ResNet-50
        RN101
    };

    CLIPStyleJointEmbedding(CLIPModel model) : clip_model(model) {
        // 初始化CLIP模型
    }

    // 文本编码器
    std::vector<float> encode_text(const std::string& text) {
        // CLIP文本编码器（基于Transformer）
        size_t dim = get_text_dim();

        // 1. Tokenization
        auto tokens = tokenize(text);

        // 2. Transformer编码
        std::vector<float> embedding = run_text_transformer(tokens);

        // 3. L2归一化
        normalize(embedding);

        return embedding;
    }

    // 图像编码器
    std::vector<float> encode_image(const std::string& image_path) {
        // CLIP图像编码器（ViT或ResNet）
        size_t dim = get_image_dim();

        // 1. 加载和预处理图像
        cv::Mat image = cv::imread(image_path);
        cv::Mat processed = preprocess_image(image);

        // 2. 视觉编码器
        std::vector<float> embedding = run_vision_encoder(processed);

        // 3. L2归一化
        normalize(embedding);

        return embedding;
    }

    // 计算跨模态相似度
    float cross_modal_similarity(const std::string& text,
                                const std::string& image_path) {
        auto text_emb = encode_text(text);
        auto image_emb = encode_image(image_path);

        return cosine_similarity(text_emb.data(), image_emb.data(),
                               text_emb.size());
    }

    // 批量检索：用文本检索图像
    std::vector<std::pair<float, size_t>> retrieve_images(
        const std::string& text_query,
        const std::vector<std::string>& image_database,
        size_t top_k = 10) {

        // 1. 编码查询
        auto query_emb = encode_text(text);

        // 2. 批量编码图像（可缓存）
        std::vector<std::vector<float>> image_embs;
        for (const auto& img_path : image_database) {
            image_embs.push_back(encode_image(img_path));
        }

        // 3. 计算相似度
        std::vector<std::pair<float, size_t>> similarities;
        for (size_t i = 0; i < image_embs.size(); i++) {
            float sim = cosine_similarity(
                query_emb.data(),
                image_embs[i].data(),
                query_emb.size()
            );
            similarities.push_back({sim, i});
        }

        // 4. 排序并返回Top-K
        std::sort(similarities.begin(), similarities.end(),
                 std::greater<>());

        if (similarities.size() > top_k) {
            similarities.resize(top_k);
        }

        return similarities;
    }

private:
    CLIPModel clip_model;

    size_t get_text_dim() const {
        switch (clip_model) {
            case CLIPModel::ViT_B_32:
            case CLIPModel::ViT_B_16:
            case CLIPModel::RN50:
                return 512;
            case CLIPModel::ViT_L_14:
            case CLIPModel::RN101:
                return 768;
            default:
                return 512;
        }
    }

    size_t get_image_dim() const {
        return get_text_dim();  // CLIP文本和图像维度相同
    }

    std::vector<int> tokenize(const std::string& text) {
        // 简化的tokenization
        return {1, 2, 3};  // 实际应使用CLIP tokenizer
    }

    std::vector<float> run_text_transformer(const std::vector<int>& tokens) {
        // 简化的Transformer编码
        std::vector<float> embedding(get_text_dim());
        for (size_t i = 0; i < embedding.size(); i++) {
            embedding[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        return embedding;
    }

    cv::Mat preprocess_image(const cv::Mat& image) {
        cv::Mat processed;
        cv::resize(image, processed, cv::Size(224, 224));
        // 其他预处理：归一化、中心裁剪等
        return processed;
    }

    std::vector<float> run_vision_encoder(const cv::Mat& image) {
        // 简化的视觉编码器
        std::vector<float> embedding(get_image_dim());
        for (size_t i = 0; i < embedding.size(); i++) {
            embedding[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        return embedding;
    }

    void normalize(std::vector<float>& embedding) {
        float norm = std::sqrt(std::inner_product(
            embedding.begin(), embedding.end(),
            embedding.begin(), 0.0f));

        for (auto& val : embedding) {
            val /= norm;
        }
    }

    float cosine_similarity(const float* a, const float* b, size_t d) {
        float dot = 0.0f;
        for (size_t i = 0; i < d; i++) {
            dot += a[i] * b[i];
        }
        // 假设已归一化
        return dot;
    }
};
```

---

## 第三部分：模态融合策略

### 3.1 早期融合（Early Fusion）

```cpp
// 早期融合：在特征层面融合
class EarlyFusionStrategy {
public:
    struct FusedEmbedding {
        std::vector<float> vector;
        size_t dim;

        void normalize() {
            float norm = std::sqrt(std::inner_product(
                vector.begin(), vector.end(),
                vector.begin(), 0.0f));

            for (auto& val : vector) {
                val /= norm;
            }
        }
    };

    // 拼接融合（Concatenation）
    static FusedEmbedding concat_fusion(
        const std::vector<std::vector<float>>& embeddings) {

        size_t total_dim = 0;
        for (const auto& emb : embeddings) {
            total_dim += emb.size();
        }

        FusedEmbedding fused;
        fused.dim = total_dim;
        fused.vector.reserve(total_dim);

        for (const auto& emb : embeddings) {
            fused.vector.insert(fused.vector.end(),
                              emb.begin(), emb.end());
        }

        fused.normalize();
        return fused;
    }

    // 加权融合
    static FusedEmbedding weighted_fusion(
        const std::vector<std::vector<float>>& embeddings,
        const std::vector<float>& weights) {

        if (embeddings.size() != weights.size()) {
            throw std::invalid_argument("Embeddings and weights size mismatch");
        }

        // 假设所有嵌入维度相同
        size_t dim = embeddings[0].size();

        FusedEmbedding fused;
        fused.dim = dim;
        fused.vector.resize(dim, 0.0f);

        // 加权求和
        for (size_t i = 0; i < embeddings.size(); i++) {
            float w = weights[i];
            for (size_t j = 0; j < dim; j++) {
                fused.vector[j] += w * embeddings[i][j];
            }
        }

        fused.normalize();
        return fused;
    }

    // 注意力融合
    static FusedEmbedding attention_fusion(
        const std::vector<std::vector<float>>& embeddings) {

        size_t dim = embeddings[0].size();
        size_t num_modalities = embeddings.size();

        // 计算注意力权重（基于内容的自注意力）
        std::vector<float> attention_weights(num_modalities);

        // 计算每个模态的"重要性"
        std::vector<float> importance(num_modalities);
        for (size_t i = 0; i < num_modalities; i++) {
            float norm = std::sqrt(std::inner_product(
                embeddings[i].begin(), embeddings[i].end(),
                embeddings[i].begin(), 0.0f));
            importance[i] = norm;
        }

        // Softmax归一化
        float sum = std::accumulate(importance.begin(),
                                   importance.end(), 0.0f);
        for (size_t i = 0; i < num_modalities; i++) {
            attention_weights[i] = importance[i] / sum;
        }

        // 加权融合
        return weighted_fusion(embeddings, attention_weights);
    }
};
```

### 3.2 晚期融合（Late Fusion）

```cpp
// 晚期融合：在分数层面融合
class LateFusionStrategy {
public:
    // 检索结果（分数）
    struct Score {
        float value;
        size_t item_id;
    };

    // 融合结果
    struct FusedResult {
        std::vector<float> final_scores;
        std::vector<size_t> item_ids;
    };

    // 最大融合（Max Fusion）
    static FusedResult max_fusion(
        const std::vector<std::vector<Score>>& modality_scores) {

        // 收集所有item_id
        std::unordered_set<size_t> all_items;
        for (const auto& scores : modality_scores) {
            for (const auto& score : scores) {
                all_items.insert(score.item_id);
            }
        }

        FusedResult fused;
        fused.item_ids.assign(all_items.begin(), all_items.end());

        // 对每个item，取各模态的最大分数
        std::map<size_t, float> max_scores;

        for (const auto& scores : modality_scores) {
            for (const auto& score : scores) {
                auto it = max_scores.find(score.item_id);
                if (it == max_scores.end() || score.value > it->second) {
                    max_scores[score.item_id] = score.value;
                }
            }
        }

        // 转换为vector
        for (size_t id : fused.item_ids) {
            fused.final_scores.push_back(max_scores[id]);
        }

        return fused;
    }

    // 加权融合
    static FusedResult weighted_fusion(
        const std::vector<std::vector<Score>>& modality_scores,
        const std::vector<float>& weights) {

        if (modality_scores.size() != weights.size()) {
            throw std::invalid_argument("Scores and weights size mismatch");
        }

        // 收集所有item
        std::unordered_map<size_t, float> combined_scores;

        for (size_t m = 0; m < modality_scores.size(); m++) {
            float w = weights[m];
            for (const auto& score : modality_scores[m]) {
                combined_scores[score.item_id] += w * score.value;
            }
        }

        // 转换为结果
        FusedResult fused;
        for (const auto& [id, score] : combined_scores) {
            fused.item_ids.push_back(id);
            fused.final_scores.push_back(score);
        }

        return fused;
    }

    // 排序投票（Rank Voting）
    static FusedResult rank_voting(
        const std::vector<std::vector<Score>>& modality_scores,
        size_t top_k = 100) {

        // 为每个模态的分数排序
        std::vector<std::vector<size_t>> rankings;

        for (const auto& scores : modality_scores) {
            std::vector<std::pair<float, size_t>> sorted;
            for (const auto& score : scores) {
                sorted.push_back({score.value, score.item_id});
            }
            std::sort(sorted.begin(), sorted.end(), std::greater<>());

            std::vector<size_t> ranking;
            for (size_t i = 0; i < std::min(top_k, sorted.size()); i++) {
                ranking.push_back(sorted[i].second);
            }
            rankings.push_back(ranking);
        }

        // 累积排名分数（排名越高分数越高）
        std::unordered_map<size_t, float> rank_scores;

        for (const auto& ranking : rankings) {
            for (size_t i = 0; i < ranking.size(); i++) {
                // Borda count: 第i名得ranking.size() - i分
                rank_scores[ranking[i]] += ranking.size() - i;
            }
        }

        // 转换为结果
        FusedResult fused;
        for (const auto& [id, score] : rank_scores) {
            fused.item_ids.push_back(id);
            fused.final_scores.push_back(score);
        }

        return fused;
    }

    // 递归归并融合（Reciprocal Rank Fusion）
    static FusedResult rrf_fusion(
        const std::vector<std::vector<Score>>& modality_scores,
        float k = 60.0f) {

        std::unordered_map<size_t, float> rrf_scores;

        for (const auto& scores : modality_scores) {
            // 按分数排序
            std::vector<std::pair<float, size_t>> sorted;
            for (const auto& score : scores) {
                sorted.push_back({score.value, score.item_id});
            }
            std::sort(sorted.begin(), sorted.end(), std::greater<>());

            // 累积RRF分数：1 / (k + rank)
            for (size_t rank = 0; rank < sorted.size(); rank++) {
                size_t item_id = sorted[rank].second;
                rrf_scores[item_id] += 1.0f / (k + rank + 1);
            }
        }

        // 转换为结果
        FusedResult fused;
        for (const auto& [id, score] : rrf_scores) {
            fused.item_ids.push_back(id);
            fused.final_scores.push_back(score);
        }

        return fused;
    }
};
```

---

## 第四部分：多模态索引设计

### 4.1 联合嵌入索引

```cpp
// 联合嵌入空间索引（CLIP风格）
class JointEmbeddingIndex {
    faiss::Index* index;
    size_t d;

    // 元数据
    struct IndexMetadata {
        uint64_t id;
        MultimodalSearchSystem::ModalityType modality;
        std::string source_path;  // 原始数据路径

        // 其他模态特定数据
        std::string text;
        std::string image_path;
    };

    std::unordered_map<uint64_t, IndexMetadata> metadata;
    std::mutex mutex;

public:
    JointEmbeddingIndex(size_t dim, const std::string& index_type = "Flat")
        : d(dim) {

        // 创建Faiss索引
        std::string desc = index_type + ",IDMap,Flat";
        index = faiss::index_factory(dim, desc.c_str());
    }

    // 添加项目
    void add_item(
        uint64_t id,
        const std::vector<float>& embedding,
        MultimodalSearchSystem::ModalityType modality,
        const std::string& source) {

        std::lock_guard<std::mutex> lock(mutex);

        // 添加到索引
        index->add_with_ids(1, embedding.data(), &id);

        // 保存元数据
        metadata[id] = {id, modality, source, "", ""};
    }

    // 添加文本项目
    void add_text_item(uint64_t id, const std::string& text,
                      const std::vector<float>& embedding) {
        add_item(id, embedding,
                MultimodalSearchSystem::ModalityType::TEXT, "");
        metadata[id].text = text;
    }

    // 添加图像项目
    void add_image_item(uint64_t id, const std::string& image_path,
                       const std::vector<float>& embedding) {
        add_item(id, embedding,
                MultimodalSearchSystem::ModalityType::IMAGE, image_path);
        metadata[id].image_path = image_path;
    }

    // 跨模态搜索
    std::vector<std::pair<float, IndexMetadata>> search(
        const std::vector<float>& query_embedding,
        size_t k,
        const std::vector<MultimodalSearchSystem::ModalityType>& filter_types = {}) {

        // 搜索
        float* distances = new float[k];
        faiss::idx_t* labels = new faiss::idx_t[k];

        index->search(1, query_embedding.data(), k,
                     distances, labels);

        // 收集结果（应用过滤）
        std::vector<std::pair<float, IndexMetadata>> results;

        for (size_t i = 0; i < k; i++) {
            faiss::idx_t label = labels[i];
            if (label < 0) continue;

            auto it = metadata.find(label);
            if (it == metadata.end()) continue;

            // 应用模态过滤
            if (!filter_types.empty()) {
                bool matches = std::find(filter_types.begin(),
                                        filter_types.end(),
                                        it->second.modality) != filter_types.end();
                if (!matches) continue;
            }

            results.push_back({distances[i], it->second});
        }

        delete[] distances;
        delete[] labels;

        return results;
    }

    // 文本检索图像
    std::vector<std::pair<float, std::string>> text_to_image(
        const std::vector<float>& text_embedding, size_t k) {

        auto results = search(
            text_embedding, k * 2,  // 搜索更多，过滤后取k
            {MultimodalSearchSystem::ModalityType::IMAGE});

        std::vector<std::pair<float, std::string>> image_results;
        for (size_t i = 0; i < std::min(k, results.size()); i++) {
            image_results.push_back({
                results[i].first,
                results[i].second.image_path
            });
        }

        return image_results;
    }

    // 图像检索文本
    std::vector<std::pair<float, std::string>> image_to_text(
        const std::vector<float>& image_embedding, size_t k) {

        auto results = search(
            image_embedding, k * 2,
            {MultimodalSearchSystem::ModalityType::TEXT});

        std::vector<std::pair<float, std::string>> text_results;
        for (size_t i = 0; i < std::min(k, results.size()); i++) {
            text_results.push_back({
                results[i].first,
                results[i].second.text
            });
        }

        return text_results;
    }

    void print_stats() const {
        printf("Joint Embedding Index Stats:\n");
        printf("  Total items: %ld\n", index->ntotal);
        printf("  Dimension: %zu\n", d);

        // 统计各模态数量
        std::unordered_map<MultimodalSearchSystem::ModalityType,
                          size_t> modality_counts;

        for (const auto& [id, meta] : metadata) {
            modality_counts[meta.modality]++;
        }

        printf("  Modality distribution:\n");
        for (const auto& [modality, count] : modality_counts) {
            printf("    %d: %zu\n",
                   static_cast<int>(modality), count);
        }
    }
};
```

### 4.2 模态特定索引

```cpp
// 模态特定索引集合
class ModalitySpecificIndexes {
public:
    // 每个模态的索引
    struct ModalityIndex {
        MultimodalSearchSystem::ModalityType modality;
        std::unique_ptr<faiss::Index> index;
        size_t dim;
        std::unordered_map<faiss::idx_t, uint64_t> id_map;  // Faiss ID -> 项目ID
    };

    std::unordered_map<MultimodalSearchSystem::ModalityType,
                      ModalityIndex> indexes;

    // 添加模态索引
    void add_modality_index(
        MultimodalSearchSystem::ModalityType modality,
        size_t dim,
        const std::string& index_type = "IVF100,PQ64") {

        ModalityIndex mod_idx;
        mod_idx.modality = modality;
        mod_idx.dim = dim;

        // 创建索引
        std::string desc = index_type + ",IDMap,Flat";
        mod_idx.index.reset(
            faiss::index_factory(dim, desc.c_str())
        );

        indexes[modality] = std::move(mod_idx);
    }

    // 添加项目到对应模态索引
    void add_item(
        MultimodalSearchSystem::ModalityType modality,
        uint64_t id,
        const std::vector<float>& embedding) {

        auto it = indexes.find(modality);
        if (it == indexes.end()) {
            throw std::runtime_error("No index for this modality");
        }

        // 添加到索引
        faiss::idx_t internal_id = it->second.index->ntotal;
        it->second.index->add_with_ids(1, embedding.data(), &internal_id);

        // 记录ID映射
        it->second.id_map[internal_id] = id;
    }

    // 在特定模态中搜索
    std::vector<std::pair<float, uint64_t>> search_modality(
        MultimodalSearchSystem::ModalityType modality,
        const std::vector<float>& query,
        size_t k) {

        auto it = indexes.find(modality);
        if (it == indexes.end()) {
            return {};
        }

        float* distances = new float[k];
        faiss::idx_t* labels = new faiss::idx_t[k];

        it->second.index->search(1, query.data(), k,
                                distances, labels);

        std::vector<std::pair<float, uint64_t>> results;

        for (size_t i = 0; i < k; i++) {
            faiss::idx_t internal_id = labels[i];
            if (internal_id < 0) continue;

            auto id_it = it->second.id_map.find(internal_id);
            if (id_it != it->second.id_map.end()) {
                results.push_back({distances[i], id_it->second});
            }
        }

        delete[] distances;
        delete[] labels;

        return results;
    }

    // 跨模态搜索（先搜各模态，再融合）
    std::vector<std::pair<float, uint64_t>> cross_modal_search(
        const std::vector<float>& query,
        size_t k,
        const std::vector<MultimodalSearchSystem::ModalityType>& modalities_to_search,
        LateFusionStrategy::FusionMethod fusion =
            LateFusionStrategy::FusionMethod::RRF) {

        // 在各模态中搜索
        std::vector<std::vector<LateFusionStrategy::Score>> all_scores;

        for (auto modality : modalities_to_search) {
            auto modality_results = search_modality(modality, query, k * 2);

            std::vector<LateFusionStrategy::Score> scores;
            for (const auto& [dist, id] : modality_results) {
                scores.push_back({1.0f - dist, id});  // 转换为相似度
            }
            all_scores.push_back(scores);
        }

        // 融合结果
        LateFusionStrategy::FusedResult fused;

        switch (fusion) {
            case LateFusionStrategy::FusionMethod::MAX:
                fused = LateFusionStrategy::max_fusion(all_scores);
                break;
            case LateFusionStrategy::FusionMethod::RRF:
                fused = LateFusionStrategy::rrf_fusion(all_scores);
                break;
            // 其他融合方法...
        }

        // 排序并返回Top-K
        std::vector<std::pair<float, uint64_t>> results;
        for (size_t i = 0; i < fused.item_ids.size(); i++) {
            results.push_back({fused.final_scores[i], fused.item_ids[i]});
        }

        std::sort(results.begin(), results.end(), std::greater<>());
        if (results.size() > k) {
            results.resize(k);
        }

        return results;
    }
};
```

---

## 第五部分：实践案例

### 5.1 图文检索系统

```cpp
// 完整的图文检索系统
class ImageTextRetrievalSystem {
    CLIPStyleJointEmbedding clip_model;
    JointEmbeddingIndex index;

public:
    ImageTextRetrievalSystem(CLIPStyleJointEmbedding::CLIPModel model)
        : clip_model(model), index(512) {}  // CLIP维度

    // 索引图像
    void index_images(const std::vector<std::string>& image_paths) {
        printf("Indexing %zu images...\n", image_paths.size());

        for (size_t i = 0; i < image_paths.size(); i++) {
            auto embedding = clip_model.encode_image(image_paths[i]);
            index.add_image_item(i, image_paths[i], embedding);

            if ((i + 1) % 100 == 0) {
                printf("  Indexed %zu/%zu images\n", i + 1, image_paths.size());
            }
        }

        printf("Image indexing complete\n");
    }

    // 索引文本
    void index_texts(const std::vector<std::string>& texts) {
        printf("Indexing %zu texts...\n", texts.size());

        for (size_t i = 0; i < texts.size(); i++) {
            auto embedding = clip_model.encode_text(texts[i]);
            index.add_text_item(i, texts[i], embedding);

            if ((i + 1) % 100 == 0) {
                printf("  Indexed %zu/%zu texts\n", i + 1, texts.size());
            }
        }

        printf("Text indexing complete\n");
    }

    // 用文本检索图像
    std::vector<std::string> search_images_by_text(
        const std::string& query, size_t k = 10) {

        auto query_emb = clip_model.encode_text(query);
        auto results = index.text_to_image(query_emb, k);

        std::vector<std::string> image_paths;
        for (const auto& [score, path] : results) {
            image_paths.push_back(path);
            printf("  [%.4f] %s\n", score, path.c_str());
        }

        return image_paths;
    }

    // 用图像检索文本
    std::vector<std::string> search_texts_by_image(
        const std::string& image_path, size_t k = 10) {

        auto query_emb = clip_model.encode_image(image_path);
        auto results = index.image_to_text(query_emb, k);

        std::vector<std::string> texts;
        for (const auto& [score, text] : results) {
            texts.push_back(text);
            printf("  [%.4f] %s\n", score, text.c_str());
        }

        return texts;
    }

    void print_stats() {
        index.print_stats();
    }
};
```

### 5.2 电商多模态搜索

```cpp
// 电商平台多模态搜索
class EcommerceMultimodalSearch {
    // 商品数据
    struct Product {
        uint64_t id;
        std::string title;
        std::string description;
        std::string category;
        std::vector<std::string> image_urls;
        float price;
    };

    CLIPStyleJointEmbedding clip_encoder;
    TextEmbeddingGenerator text_encoder;
    ImageEmbeddingGenerator image_encoder;

    JointEmbeddingIndex clip_index;
    faiss::Index* text_index;
    faiss::Index* image_index;

public:
    EcommerceMultimodalSearch()
        : clip_encoder(CLIPStyleJointEmbedding::CLIPModel::ViT_B_32),
          clip_index(512),
          text_encoder(TextEmbeddingGenerator::ModelType::OPENAI_EMBEDDING),
          image_encoder(ImageEmbeddingGenerator::ModelType::RESNET) {

        // 初始化文本索引
        text_index = faiss::index_factory(1536, "IVF1000,PQ64");

        // 初始化图像索引
        image_index = faiss::index_factory(2048, "HNSW32");
    }

    // 索引商品
    void index_product(const Product& product) {
        // 1. CLIP联合嵌入（用于图文跨模态）
        auto clip_emb = clip_encoder.encode_text(product.title);
        // 简化：仅用标题

        // 2. 文本专用嵌入（更精确的文本搜索）
        auto text_emb = text_encoder.generate(product.title + " " +
                                            product.description);

        // 3. 图像嵌入
        std::vector<std::vector<float>> image_embs;
        for (const auto& img_url : product.image_urls) {
            image_embs.push_back(image_encoder.generate(img_url));
        }

        // 添加到索引
        // ...（具体实现）
    }

    // 多模态商品搜索
    struct SearchResult {
        Product product;
        float clip_score;
        float text_score;
        float image_score;
        float combined_score;
    };

    std::vector<SearchResult> search_products(
        const std::string& query,
        const std::string& image_query = "",
        size_t k = 20) {

        // 1. CLIP跨模态搜索
        auto clip_emb = clip_encoder.encode_text(query);
        auto clip_results = clip_index.search(clip_emb, k * 3);

        // 2. 文本搜索（如果提供了文本查询）
        std::vector<SearchResult> text_results;
        if (!query.empty()) {
            auto text_emb = text_encoder.generate(query);
            // 搜索文本索引...
        }

        // 3. 图像搜索（如果提供了图像查询）
        std::vector<SearchResult> image_results;
        if (!image_query.empty()) {
            auto img_emb = image_encoder.generate(image_query);
            // 搜索图像索引...
        }

        // 4. 融合结果
        std::vector<SearchResult> fused_results;
        // ...（融合逻辑）

        // 5. 排序并返回Top-K
        std::sort(fused_results.begin(), fused_results.end(),
                 [](const auto& a, const auto& b) {
                     return a.combined_score > b.combined_score;
                 });

        if (fused_results.size() > k) {
            fused_results.resize(k);
        }

        return fused_results;
    }
};
```

---

## 实验练习

### 练习1: 实现CLIP风格图文检索

```cpp
void exercise_1_clip_retrieval() {
    // 1. 集成CLIP模型（使用OpenAI CLIP或开源实现）
    // 2. 索引图像-文本对
    // 3. 实现文本检索图像
    // 4. 实现图像检索文本
    // 5. 评估检索质量
}
```

### 练习2: 多模态融合对比

```cpp
void exercise_2_fusion_comparison() {
    // 1. 实现早期融合（拼接、注意力）
    // 2. 实现晚期融合（最大、加权、RRF）
    // 3. 在多模态数据集上比较
    // 4. 分析各融合策略的优劣
}
```

### 练习3: 音视频检索

```cpp
void exercise_3_audio_video_retrieval() {
    // 1. 使用预训练音频模型（VGGish, PANNs）
    // 2. 使用预训练视频模型（I3D, X3D）
    // 3. 实现音频-视频跨模态检索
    // 4. 测试检索性能
}
```

### 练习4: 多模态推荐

```cpp
void exercise_4_multimodal_recommendation() {
    // 1. 结合用户行为、文本、图像
    // 2. 实现多模态商品推荐
    // 3. A/B测试不同融合策略
    // 4. 分析推荐效果
}
```

---

## 总结

第25天深入探讨了多模态向量检索，涵盖：

1. **基础概念**：
   - 多模态检索定义
   - 模态鸿沟问题
   - 对齐质量分析

2. **模态嵌入**：
   - 文本嵌入（BERT、Sentence-Transformer）
   - 图像嵌入（ResNet、ViT）
   - 联合嵌入（CLIP）

3. **融合策略**：
   - 早期融合（拼接、加权、注意力）
   - 晚期融合（最大、加权、RRF）

4. **索引设计**：
   - 联合嵌入索引
   - 模态特定索引

5. **实践案例**：
   - 图文检索系统
   - 电商多模态搜索

**关键要点**：
- CLIP等联合嵌入模型是实现跨模态检索的关键
- 融合策略需要根据应用场景选择
- 早期融合保持更多信息，晚期融合更灵活
- 联合嵌入空间最适合跨模态检索
- 多模态检索在电商、内容推荐等场景价值巨大

## 后续学习

- 研究更多多模态模型（ALIGN、ALBEF、BLIP）
- 学习多模态大语言模型（GPT-4V、Gemini）
- 探索多模态检索的强化学习优化
- 实践端到端的多模态检索系统
