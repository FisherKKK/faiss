/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <algorithm>
#include <mutex>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>

namespace faiss {

/** Implementation of NNDescent which is one of the most popular
 *  KNN graph building algorithms
 *
 * Efficient K-Nearest Neighbor Graph Construction for Generic
 * Similarity Measures
 *
 *  Dong, Wei, Charikar Moses, and Kai Li, WWW 2011
 *
 * This implmentation is heavily influenced by the efanna
 * implementation by Cong Fu and the KGraph library by Wei Dong
 * (https://github.com/ZJULearning/efanna_graph)
 * (https://github.com/aaalgo/kgraph)
 *
 * The NNDescent object stores only the neighbor link structure,
 * see IndexNNDescent.h for the full index object.
 */

struct VisitedTable;
struct DistanceComputer;

namespace nndescent {

struct Neighbor {
    int id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(int id, float distance, bool f)
            : id(id), distance(distance), flag(f) {}

    inline bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

struct Nhood {
    std::mutex lock; /// 当前这个节点的lock, 管理这个节点对应的竞争问题
    std::vector<Neighbor> pool; // candidate pool (a max heap), 候选池子, 保存了当前节点对其它节点的距离
    int M;                      // number of new neighbors to be operated

    std::vector<int> nn_old;  // old neighbors
    std::vector<int> nn_new;  // new neighbors
    std::vector<int> rnn_old; // reverse old neighbors
    std::vector<int> rnn_new; // reverse new neighbors

    Nhood() = default;

    Nhood(int l, int s, std::mt19937& rng, int N);

    Nhood& operator=(const Nhood& other);

    Nhood(const Nhood& other);

    void insert(int id, float dist);

    /// 这里相当于是一个回调函数, 向join中添加函数
    template <typename C>
    void join(C callback) const;
};

} // namespace nndescent

/** TODO KNNG构建过程的算法, 但是目前不知道什么意思
 *
 */
struct NNDescent {
    using storage_idx_t = int;

    using KNNGraph = std::vector<nndescent::Nhood>;

    explicit NNDescent(const int d, const int K);

    ~NNDescent();

    void build(DistanceComputer& qdis, const int n, bool verbose); // 采用NNDescent构建KNNG

    void search(
            DistanceComputer& qdis,
            const int topk,
            idx_t* indices,
            float* dists,
            VisitedTable& vt) const;

    void reset();

    /// Initialize the KNN graph randomly, 随机初始化KNNG
    void init_graph(DistanceComputer& qdis);

    /// Perform NNDescent algorithm, 构建KNNG的具体算法
    void nndescent(DistanceComputer& qdis, bool verbose);

    /// Perform local join on each node, 每个节点进行局部的连接
    void join(DistanceComputer& qdis);

    /// Sample new neighbors for each node to peform local join later
    void update();

    /// Sample a small number of points to evaluate the quality of KNNG built
    void generate_eval_set(
            DistanceComputer& qdis,
            std::vector<int>& c,
            std::vector<std::vector<int>>& v,
            int N);

    /// Evaluate the quality of KNNG built
    float eval_recall(
            std::vector<int>& ctrl_points,
            std::vector<std::vector<int>>& acc_eval_set);

    bool has_built = false;

    int S = 10;  // number of sample neighbors to be updated for each node, 每个节点采用一定数目的邻居进行更新
    int R = 100; // size of reverse links, 0 means the reverse links will not be, 逆向边的数目
                 // used
    int iter = 10;          // number of iterations to iterate over, 迭代次数
    int search_L = 0;       // size of candidate pool in searching
    int random_seed = 2021; // random seed for generators

    int K; // K in KNN graph
    int d; // dimensions
    int L; // size of the candidate pool in building, 构建候选池的大小

    int ntotal = 0;

    KNNGraph graph; /// KNNGraph中实际上就是一个邻接表表示的图, 但是里面包含了新旧的neighbor
    std::vector<int> final_graph;
};

} // namespace faiss
