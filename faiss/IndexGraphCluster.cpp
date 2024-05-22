//
// Created by yk120 on 2024/5/21.
//

#include <faiss/IndexGraphCluster.h>

namespace faiss {

void IndexGraphCluster::train(idx_t n, const float* x) {
    // 选择nlist个vector构建hnsw, 但是为了防止重复计算, 我们需要一个mapper
    graph2id.reserve(nlist);
    std::vector<int> perm(n);
    rand_perm(perm.data(), n, seed); // 实际上perm已经涵盖了映射关系

    for (int i = 0; i < nlist; i++)
        graph2id[i] = perm[i];

    int max_level = hnsw.prepare_level_tab(nlist, false);
    for (int i = 0; i < nlist; i++) {

    }

}

}