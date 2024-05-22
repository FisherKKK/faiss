//
// Created by yk120 on 2024/5/21.
//

#include <faiss/IndexGraphCluster.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/impl/DistanceComputer.h>

namespace faiss {

using storage_idx_t = HNSW::storage_idx_t;

namespace {

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */


struct GraphMapDistanceComputer : DistanceComputer {
    /// owned by this
    DistanceComputer* basedis;
    std::unordered_map<idx_t, idx_t> &graph2id;

    explicit GraphMapDistanceComputer(DistanceComputer* basedis, std::unordered_map<idx_t, idx_t> &graph2id)
            : basedis(basedis), graph2id(graph2id) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }


    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        return (*basedis)(graph2id[i]);
    }

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        basedis->distances_batch_4(
                graph2id[idx0], graph2id[idx1], graph2id[idx2],graph2id[idx3],
                dis0, dis1, dis2, dis3);
    }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return basedis->symmetric_dis(graph2id[i], graph2id[j]);
    }

    virtual ~GraphMapDistanceComputer() {
        delete basedis;
    }
};

} // namespace


void IndexGraphCluster::train(idx_t n, const float* x) {
    // 选择nlist个vector构建hnsw, 但是为了防止重复计算, 我们需要一个mapper
    graph2id.reserve(nlist);
    std::vector<int> perm(n);
    rand_perm(perm.data(), n, seed); // 实际上perm已经涵盖了映射关系

    for (int i = 0; i < nlist; i++)
        graph2id[i] = perm[i];

    int max_level = hnsw.prepare_level_tab(nlist, false);

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    // 构建HNSW insert order, 基于bucket sort
    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < nlist; i++) {
            storage_idx_t pt_id = i;
            int pt_level = hnsw.levels[pt_id] - 1;
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < nlist; i++) {
            storage_idx_t pt_id = i;
            int pt_level = hnsw.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }


    { // perform add
        RandomGenerator rng2(789);

        int i1 = nlist;

        for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
            int i0 = i1 - hist[pt_level];


            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(nlist);

                std::unique_ptr<DistanceComputer> dis(
                        new GraphMapDistanceComputer(storage->get_distance_computer(), graph2id));
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                size_t counter = 0;

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    storage_idx_t real_id = graph2id[pt_id];
                    dis->set_query(x + real_id * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    hnsw.add_with_locks(*dis, pt_level, pt_id, locks, vt);

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }

                    counter++;
                }
            }

            i1 = i0;
        }
        FAISS_ASSERT(i1 == 0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }

}

}