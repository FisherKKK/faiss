//
// Created by yk120 on 2024/5/21.
//

#ifndef FAISS_INDEXGRAPHCLUSTER_H
#define FAISS_INDEXGRAPHCLUSTER_H

#include <unordered_map>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/HNSW.h>

namespace faiss {

struct IndexGraphCluster : Index {

    size_t nlist;

    size_t nprobe;

    int seed = 1234;

    IndexFlatL2* storage = nullptr;

    std::vector<std::vector<idx_t>> ivf;

    HNSW hnsw;

    std::unordered_map<idx_t, idx_t> graph2id;
    std::unordered_set<idx_t> vertexes;

    explicit IndexGraphCluster(
            int d,
            size_t nlist,
            int M,
            MetricType metric = METRIC_L2);

    explicit IndexGraphCluster(
            Index *storage,
            size_t nlist,
            int M = 32,
            MetricType metric = METRIC_L2);

    IndexGraphCluster();

    void train(idx_t n, const float *x) override;

    void add(idx_t n, const float *x) override;

    void create_graph(const float *x);

    void select_vertex(idx_t n);

    void create_storage(idx_t n, const float *x);

    std::pair<float, idx_t> search_top1(const float *x) const;

    std::unordered_set<idx_t> prune_neighbor(std::pair<float, idx_t>& top1, DistanceComputer& dc);

    void search(faiss::idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override



};
}
#endif // FAISS_INDEXGRAPHCLUSTER_H
