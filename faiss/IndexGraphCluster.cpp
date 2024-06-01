//
// Created by yk120 on 2024/5/21.
//

#include <algorithm>

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
    const std::unordered_map<idx_t, idx_t> &graph2id;

    explicit GraphMapDistanceComputer(DistanceComputer* basedis, const std::unordered_map<idx_t, idx_t> &graph2id)
            : basedis(basedis), graph2id(graph2id) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }


    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        return (*basedis)(graph2id.at(i));
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
                graph2id.at(idx0), graph2id.at(idx1),
                graph2id.at(idx2),graph2id.at(idx3),
                dis0, dis1, dis2, dis3);
    }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return basedis->symmetric_dis(graph2id.at(i), graph2id.at(j));
    }

    virtual ~GraphMapDistanceComputer() {
        delete basedis;
    }
};

} // namespace

IndexGraphCluster::IndexGraphCluster(IndexFlatL2* storage, size_t nlist, size_t duplicate, int M)
                : Index(storage->d, storage->metric_type),
                    nlist(nlist), duplicate(duplicate), hnsw(M), storage(storage) {

}

void IndexGraphCluster::select_vertex(idx_t n) {
    // 选择nlist个vector构建hnsw, 但是为了防止重复计算, 我们需要一个mapper
    graph2id.reserve(nlist);
    std::vector<int> perm(n);
    rand_perm(perm.data(), n, seed); // 实际上perm已经涵盖了映射关系
    for (int i = 0; i < nlist; i++) {
        graph2id[i] = perm[i];
        vertexes.insert(perm[i]);
    }

}


void IndexGraphCluster::create_graph(const float *x) {
    hnsw.prepare_level_tab(nlist);
    std::vector<omp_lock_t> locks(nlist);
    for (int i = 0; i < nlist; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(nlist);

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


#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(nlist);

                std::unique_ptr<DistanceComputer> dis(
                        new GraphMapDistanceComputer(storage->get_distance_computer(), graph2id));

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    storage_idx_t real_id = graph2id[pt_id];
                    dis->set_query(x + real_id * d);
                    hnsw.add_with_locks(*dis, pt_level, pt_id, locks, vt);
                }
            }
            i1 = i0;
        }
        FAISS_ASSERT(i1 == 0);
    }

    for (int i = 0; i < nlist; i++) {
        omp_destroy_lock(&locks[i]);
    }
}

void IndexGraphCluster::create_storage(idx_t n, const float* x) {
    storage->add(n, x);
}

void IndexGraphCluster::train(idx_t n, const float* x) {
    select_vertex(n);
    create_storage(n, x);
    create_graph(x);
    is_trained = true;
}

std::pair<float, idx_t> IndexGraphCluster::search_top1(const float *x, DistanceComputer& dc) const {
    std::pair<float, idx_t> result;
    using RH = Top1BlockResultHandler<HNSW::C>;

    RH bres(1, &result.first, &result.second);
    RH::SingleResultHandler res(bres);
    VisitedTable vt(nlist);

    res.begin(0);
    hnsw.search(dc, res, vt);
    res.end();

    return result;

}

std::vector<idx_t> IndexGraphCluster::prune_neighbor(std::pair<float, idx_t>& top1, DistanceComputer& dc) {
    size_t begin, end;
    hnsw.neighbor_range(top1.second, 0, &begin, &end);

    std::vector<std::pair<float, idx_t>> candidate_list;
    std::vector<idx_t> prune_list;
    candidate_list.push_back(top1);
    for (size_t nn = begin; nn < end; nn++) {
        int v = hnsw.neighbors[nn];
        if (v < 0)
            break;
        candidate_list.emplace_back(dc(v), v);
    }
    std::sort(candidate_list.begin(), candidate_list.end());
    std::unique(candidate_list.begin(), candidate_list.end(), [&](const std::pair<float, idx_t>& f,  const std::pair<float, idx_t>& n) {
        return f.second == n.second;
    });

    for (auto &nn: candidate_list) {
        bool occlude = false;
        for (auto &prune: prune_list) {
            if (dc.symmetric_dis(prune, nn.second) < nn.first) {
                occlude = true;
                break;
            }
        }

//        if (!occlude) prune_list.push_back(nn.second);
        prune_list.push_back(nn.second);
    }
    return prune_list;
}


void IndexGraphCluster::add(idx_t n, const float* x) {
    std::unique_ptr<DistanceComputer> dis(new GraphMapDistanceComputer(storage->get_distance_computer(), graph2id));
    ivf.resize(nlist);
    for (idx_t i = 0; i < n; i++) {
        if (vertexes.count(i) != 0)
            continue;
        const float *xi = x + i * d;
        dis->set_query(xi);

        auto top1 = search_top1(xi, *dis);
        auto nn = prune_neighbor(top1, *dis);

        int nn_size = std::min(nn.size(), duplicate);

        int counter = 0;
        for (auto id: nn) {
            counter++;
            ivf[id].push_back(i);
            if (counter >= nn_size)
                break;
        }
    }
}


void IndexGraphCluster::search_use_topk(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels) const {
    using RH = HeapBlockResultHandler<HNSW::C>;
    std::vector<float> bucket_dis(nprobe * n);
    std::vector<idx_t> bucket_lable(nprobe * n);
    RH bnn(n, distances, labels, k);

    auto find_nearest_partition = [&]() {

        std::unique_ptr<DistanceComputer> dc(new GraphMapDistanceComputer(storage->get_distance_computer(), graph2id));
        RH bnc(n, bucket_dis.data(), bucket_lable.data(), nprobe);
        for (idx_t i = 0; i < n; i++) {
            const float *xi = x + i * d;
            VisitedTable vt(nlist);
            RH::SingleResultHandler res(bnc);
            res.begin(i);
            dc->set_query(xi);
            hnsw.search(*dc, res, vt); // 计算最近的centroid
            res.end();

            RH::SingleResultHandler centroid_handler(bnn);
            centroid_handler.begin(i);
            for (int j = 0; j < nprobe; j++) {
                if (bucket_lable[nprobe * i + j] < 0)
                    continue;
                centroid_handler.add_result(bucket_dis[nprobe * i + j],
                                            graph2id.at(bucket_lable[nprobe * i + j]));

            } // 同步更新到最终结果
            centroid_handler.end();
        }

    };

    auto search_subroutine = [](DistanceComputer& dc , RH::SingleResultHandler &res, const std::vector<idx_t>& subset, VisitedTable &vt) {
        for (auto id: subset) {
            if (vt.get(id))
                continue;
            vt.set(id);
            res.add_result(dc(id), id);
        }
    };

    auto find_nearest_neighbor = [&]() {
        std::unique_ptr<DistanceComputer> dc(storage->get_distance_computer());
        for (idx_t i = 0; i < n; i++) {
            const float *xi = x + i * d;
            VisitedTable vt(storage->ntotal);
            dc->set_query(xi);
            RH::SingleResultHandler res(bnn);
            res.begin(i);
            for (int j = 0; j < nprobe; j++) {
                int centroid = bucket_lable[i * nprobe + j];
                if (centroid >= 0)
                    search_subroutine(*dc, res, ivf[centroid], vt);
            }
            res.end();
        }
    };


    find_nearest_partition();
    find_nearest_neighbor();
}

void IndexGraphCluster::search_use_neighbor(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels) const {
    using RH = HeapBlockResultHandler<HNSW::C>;
    std::vector<float> bucket_dis(nprobe * n);
    std::vector<idx_t> bucket_lable(nprobe * n);
    RH bnn(n, distances, labels, k);

    auto explore_centroid_neighbor = [&](idx_t iq) -> std::vector<idx_t> {
        std::vector<idx_t> neighbors;
        std::queue<idx_t> q;
        VisitedTable vt(nlist);

        for (int i = 0; i < nprobe; i++) {
            idx_t id = bucket_lable[iq * nprobe + i];
            if (id < 0)
                continue;
            vt.set(id);
            q.push(id);
        }


        int num_visited = 1;
        while (num_visited < nprobe * 2 && !q.empty()) {
            idx_t vn = q.front(); q.pop();
            size_t begin, end;
            hnsw.neighbor_range(vn, 0, &begin, &end);
            for (size_t nn = begin; nn < end; nn++) {
                idx_t nxt = hnsw.neighbors[nn];
                if (nxt < 0)
                    break;
                if (vt.get(nxt))
                    continue;
                vt.set(nxt);
                q.push(nxt);
                neighbors.push_back(nxt);
                num_visited += 1;
            }
        }
        return neighbors;
    };

    auto find_nearest_point_neighbor = [&]() {
        std::unique_ptr<DistanceComputer> dc(new GraphMapDistanceComputer(storage->get_distance_computer(), graph2id));

        RH bnc(n, bucket_dis.data(), bucket_lable.data(), nprobe);
        for (idx_t i = 0; i < n; i++) {
            const float *xi = x + i * d;
            VisitedTable vt(nlist);
            RH::SingleResultHandler res(bnc);
            res.begin(i);
            dc->set_query(xi);
            hnsw.search(*dc, res, vt); // 计算最近的centroid
//            res.end();

            // find the neighbor of the centroid
            RH::SingleResultHandler centroid_handler(bnn);
            centroid_handler.begin(i);

            for (int j = 0; j < nprobe; j++) {
                if (bucket_lable[nprobe * i + j] < 0)
                    continue;
                centroid_handler.add_result(bucket_dis[nprobe * i + j],
                                            graph2id.at(bucket_lable[nprobe * i + j]));

            } // 同步更新到最终结果



            auto nns = explore_centroid_neighbor(i);
//            res.begin(i);


            for (auto id: nns) {
                float _d = (*dc)(id);
                res.add_result(_d, id); // 为中心做准备
                centroid_handler.add_result(_d,graph2id.at(id));
            }

            res.end();
            centroid_handler.end();
        }
    };

    auto search_subroutine = [](DistanceComputer& dc , RH::SingleResultHandler &res, const std::vector<idx_t>& subset, VisitedTable &vt) {
        for (auto id: subset) {
            if (vt.get(id))
                continue;
            vt.set(id);
            res.add_result(dc(id), id);
        }
    };

    auto find_nearest_neighbor = [&]() {
        std::unique_ptr<DistanceComputer> dc(storage->get_distance_computer());
        for (idx_t i = 0; i < n; i++) {
            const float *xi = x + i * d;
            VisitedTable vt(storage->ntotal);
            dc->set_query(xi);
            RH::SingleResultHandler res(bnn);
            res.begin(i);
            for (int j = 0; j < nprobe; j++) {
                int centroid = bucket_lable[i * nprobe + j];
                if (centroid >= 0)
                    search_subroutine(*dc, res, ivf[centroid], vt);
            }
            res.end();
        }
    };

    find_nearest_point_neighbor();
    find_nearest_neighbor();

}



void IndexGraphCluster::search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels, const SearchParameters* params) const {
    search_use_neighbor(n, x, k, distances, labels);
//    search_use_topk(n, x, k, distances, labels);
}





}