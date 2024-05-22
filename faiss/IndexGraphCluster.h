//
// Created by yk120 on 2024/5/21.
//

#ifndef FAISS_INDEXGRAPHCLUSTER_H
#define FAISS_INDEXGRAPHCLUSTER_H

#include <unordered_map>

#include <faiss/Index.h>
#include <faiss/impl/HNSW.h>

namespace faiss {

struct IndexGraphCluster :  Index {
    //

    size_t nlist;

    int seed = 1234;

    Index* storage = nullptr;


    HNSW hnsw;

    std::unordered_map<idx_t, idx_t> graph2id;

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


};
}
#endif // FAISS_INDEXGRAPHCLUSTER_H
