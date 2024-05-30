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

    size_t duplicate;

    int seed = 1234;

    HNSW hnsw;

    IndexFlatL2* storage = nullptr;

    std::vector<std::vector<idx_t>> ivf;

    std::unordered_map<idx_t, idx_t> graph2id;
    std::unordered_set<idx_t> vertexes;

    explicit IndexGraphCluster(
            IndexFlatL2 *storage,
            size_t nlist,
            size_t duplicate,
            int M = 32);

    IndexGraphCluster();

    void train(idx_t n, const float *x) override;

    void add(idx_t n, const float *x) override;

    void create_graph(const float *x);

    void select_vertex(idx_t n);

    void create_storage(idx_t n, const float *x);

    std::pair<float, idx_t> search_top1(const float *x, DistanceComputer& dc) const;

    std::vector<idx_t> prune_neighbor(std::pair<float, idx_t>& top1, DistanceComputer& dc);

    void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const SearchParameters *params = nullptr) const override;

    void reset() override {

    }

};
}
#endif // FAISS_INDEXGRAPHCLUSTER_H
