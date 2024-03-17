/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/HNSW.h>

#include <string>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/prefetch.h>

#include <faiss/impl/platform_macros.h>

#ifdef __AVX2__
#include <immintrin.h>

#include <limits>
#include <type_traits>
#endif

namespace faiss {

/**************************************************************
 * HNSW structure implementation
 **************************************************************/

int HNSW::nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no + 1] -
            cum_nneighbor_per_level[layer_no];
}

void HNSW::set_nb_neighbors(int level_no, int n) {
    FAISS_THROW_IF_NOT(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);
    for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
        cum_nneighbor_per_level[i] += n - cur_n;
    }
}

int HNSW::cum_nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no];
}

void HNSW::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
        const {
    size_t o = offsets[no]; // 获取neighbor偏置
    *begin = o + cum_nb_neighbors(layer_no); // 添加layer之后的neighbor偏置
    *end = o + cum_nb_neighbors(layer_no + 1);
}

HNSW::HNSW(int M) : rng(12345) {
    // mL = 1.0 / log(M)
    set_default_probas(M, 1.0 / log(M));
    offsets.push_back(0);
}

int HNSW::random_level() {
    double f = rng.rand_float();
    // could be a bit faster with bissection
    for (int level = 0; level < assign_probas.size(); level++) {
        if (f < assign_probas[level]) {
            return level;
        }
        f -= assign_probas[level];
    }
    // happens with exponentially low probability
    return assign_probas.size() - 1;
}
// 1. HNSW初始化的第一个函数
void HNSW::set_default_probas(int M, float levelMult) {
    int nn = 0;
    cum_nneighbor_per_level.push_back(0);
    for (int level = 0;; level++) { // 这里的概率相当于不断的递减,
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9)
            break;
        assign_probas.push_back(proba); // 每一层的概率
        nn += level == 0 ? M * 2 : M; // 累计每一层的边数
        cum_nneighbor_per_level.push_back(nn);
    }
}

void HNSW::clear_neighbor_tables(int level) {
    for (int i = 0; i < levels.size(); i++) {
        size_t begin, end;
        neighbor_range(i, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            neighbors[j] = -1;
        }
    }
}

void HNSW::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
}

void HNSW::print_neighbor_stats(int level) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));
    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0)
                    break;
                neighset.insert(neighbors[j]);
            }
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                if (i2 < 0)
                    break;
                FAISS_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];
                    if (i3 < 0)
                        break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   nb of nodes at that level %zd\n", n_node);
    printf("   neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}

void HNSW::fill_with_random_links(size_t n) {
    int max_level = prepare_level_tab(n);
    RandomGenerator rng2(456);

    for (int level = max_level - 1; level >= 0; --level) {
        std::vector<int> elts;
        for (int i = 0; i < n; i++) {
            if (levels[i] > level) {
                elts.push_back(i);
            }
        }
        printf("linking %zd elements in level %d\n", elts.size(), level);

        if (elts.size() == 1)
            continue;

        for (int ii = 0; ii < elts.size(); ii++) {
            int i = elts[ii];
            size_t begin, end;
            neighbor_range(i, 0, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                int other = 0;
                do {
                    other = elts[rng2.rand_int(elts.size())];
                } while (other == i);

                neighbors[j] = other;
            }
        }
    }
}
// 为每个点计算所在的level, 然后它的neighbor所处的offset, 因为之前已经累计过每一个level的邻居数目
int HNSW::prepare_level_tab(size_t n, bool preset_levels) {
    size_t n0 = offsets.size() - 1;

    if (preset_levels) {
        FAISS_ASSERT(n0 + n == levels.size());
    } else {
        FAISS_ASSERT(n0 == levels.size());
        for (int i = 0; i < n; i++) { // 这里相当于为每个点预先挑选level
            int pt_level = random_level();
            levels.push_back(pt_level + 1); // levels中就是存储每个点的level
        }
    }

    int max_level = 0; // 如下计算最大的level
    for (int i = 0; i < n; i++) {
        int pt_level = levels[i + n0] - 1;
        if (pt_level > max_level)
            max_level = pt_level;
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1)); // 每一个点拥有的neighbors数目, 以及邻居所在的offset
        neighbors.resize(offsets.back(), -1); // 调整neighbor的大小
    }

    return max_level;
}

/** Enumerate vertices from nearest to farthest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
void HNSW::shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFarther>& input,
        std::vector<NodeDistFarther>& output,
        int max_size) {
    while (input.size() > 0) {
        NodeDistFarther v1 = input.top();
        input.pop();
        float dist_v1_q = v1.d;

        bool good = true;
        for (NodeDistFarther v2 : output) { // 对已经连接邻居进行遍历, for all nh: dist(current, q) < dist(current, nh)
            float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id); // 计算current, nh的距离

            if (dist_v1_v2 < dist_v1_q) { // 判断距离
                good = false;
                break;
            }
        }
        // 如果足够好就push
        if (good) {
            output.push_back(v1);
            if (output.size() >= max_size) {
                return;
            }
        }
    }
}

namespace {

using storage_idx_t = HNSW::storage_idx_t;
using NodeDistCloser = HNSW::NodeDistCloser;
using NodeDistFarther = HNSW::NodeDistFarther;

/**************************************************************
 * Addition subroutines
 **************************************************************/

/// remove neighbors from the list to make it smaller than max_size, 这里就相当于裁边策略
void shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& resultSet1,
        int max_size) {
    if (resultSet1.size() < max_size) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet; // 结果集
    std::vector<NodeDistFarther> returnlist; // 返回列表
    // 将resultSet1的内容放到resultSet
    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }
    // 裁边策略, 将裁边后的结果放到returnlist中
    HNSW::shrink_neighbor_list(qdis, resultSet, returnlist, max_size);
    // 裁边后的结果集, 加入到resultSet1, 返回出去
    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}

/// add a link between two elements, possibly shrinking the list
/// of links to make room for it. 连边操作, src是源点, dest是目标点, 如果邻居的数目超出限制就会进行剪枝
void add_link(
        HNSW& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level) {
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end); // 获取当前节点的邻居offset
    if (hnsw.neighbors[end - 1] == -1) { // 存在room连边
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (hnsw.neighbors[i - 1] != -1)
                break;
            i--; // 找到一个可以添加的i
        }
        hnsw.neighbors[i] = dest; // 连边
        return;
    }

    // otherwise we let them fight out which to keep, 这里相当于不存在空间连边

    // copy to resultSet... 这里相当于将所有的邻居都计算一遍
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
        storage_idx_t neigh = hnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }
    // 这里相当于裁边
    shrink_neighbor_list(qdis, resultSet, end - begin);

    // ...and back
    size_t i = begin; // 重置起始点
    while (resultSet.size()) {
        hnsw.neighbors[i++] = resultSet.top().id; // 这里相当于将resultSet中的结果加入到neighbor中
        resultSet.pop(); // 弹出
    }
    // they may have shrunk more than just by 1 element. 对于不够的点, 直接-1填充
    while (i < end) {
        hnsw.neighbors[i++] = -1;
    }
}

/// search neighbors on a single level, starting from an entry point, 从入口点搜索, 进行单层最近邻搜索
void search_neighbors_to_add(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt) {
    // top is nearest candidate
    std::priority_queue<NodeDistFarther> candidates; // 候选点的集合, 这里是一个最小堆吧

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point); // 加入结果集
    vt.set(entry_point); // 当前已经访问过的点
    // 如果候选集不为空
    while (!candidates.empty()) {
        // get nearest
        const NodeDistFarther& currEv = candidates.top(); // 获取当前candidate中的最近邻

        if (currEv.d > results.top().d) { // 如果candidate的最近邻大于result, 那么终止查找
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors, 遍历候选点的邻居
        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end); // 获取邻居的偏置
        for (size_t i = begin; i < end; i++) { // 遍历所有的邻居加入到结果集
            storage_idx_t nodeId = hnsw.neighbors[i];
            if (nodeId < 0)
                break;
            if (vt.get(nodeId)) // 如果当前节点已经被访问过
                continue;
            vt.set(nodeId); // 设置已经访问过

            float dis = qdis(nodeId);
            NodeDistFarther evE1(dis, nodeId); // 计算距离
            // 相当于这里只处理距离更小的邻居, 否者就不管
            if (results.size() < hnsw.efConstruction || results.top().d > dis) { // 结果集还可以加入(小于ef或者距离更小)
                results.emplace(dis, nodeId);
                candidates.emplace(dis, nodeId); // 加入候选集
                if (results.size() > hnsw.efConstruction) {
                    results.pop();
                }
            }
        }
    }
    vt.advance();
}

/**************************************************************
 * Searching subroutines
 **************************************************************/

/// greedily update a nearest vector at a given level, 在给定的level更新最近邻(搜索当前层的最近邻)
void greedy_update_nearest(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    for (;;) {
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end); // 计算得到neighbor的begin和end
        for (size_t i = begin; i < end; i++) { // 遍历当前nearest,在当前level的所有neighbor
            storage_idx_t v = hnsw.neighbors[i];
            if (v < 0)
                break;
            float dis = qdis(v); // 计算distance
            if (dis < d_nearest) { // 更新nearest信息
                nearest = v;
                d_nearest = dis;
            }
        }
        if (nearest == prev_nearest) { // 如果nearest没有改变, 直接return
            return;
        }
    }
}

} // namespace

/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void HNSW::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt) {
    std::priority_queue<NodeDistCloser> link_targets;
    // 搜索当前点的最近邻
    search_neighbors_to_add(
            *this, ptdis, link_targets, nearest, d_nearest, level, vt);

    // but we can afford only this many neighbors, 获取当前level允许的neighbor数目
    int M = nb_neighbors(level);

    ::faiss::shrink_neighbor_list(ptdis, link_targets, M);

    std::vector<storage_idx_t> neighbors;
    neighbors.reserve(link_targets.size());
    while (!link_targets.empty()) { // 应该是将link_target进行连边
        storage_idx_t other_id = link_targets.top().id;
        add_link(*this, ptdis, pt_id, other_id, level); // 这里是连正向边
        neighbors.push_back(other_id);
        link_targets.pop();
    }

    omp_unset_lock(&locks[pt_id]); // 释放当前节点的锁
    for (storage_idx_t other_id : neighbors) {
        omp_set_lock(&locks[other_id]); // 连边操作的时候锁住这个点(这是邻居点啊), 也就是对谁的邻居操作, 就锁上谁
        add_link(*this, ptdis, other_id, pt_id, level); // 反向连边操作
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]); // 设置当前节点的锁
}

/**************************************************************
 * Building, parallel
 **************************************************************/

void HNSW::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt) {
    //  greedy search on upper levels

    storage_idx_t nearest; // 在upper level搜索最近邻
#pragma omp critical
    {
        nearest = entry_point;

        if (nearest == -1) { // 没有入口点就设置这个点作为入口点, 同时设置max_level
            max_level = pt_level;
            entry_point = pt_id;
        }
    }

    if (nearest < 0) {
        return;
    }

    omp_set_lock(&locks[pt_id]); // 为当前这个节点设置锁

    int level = max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest); // 计算当前点到nearest的距离
    // upper的话我们直接搜最近邻即可, 采用贪心的策略, 在每一层搜索最近邻, 逐层向下
    for (; level > pt_level; level--) {
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
    }
    // 到达了这个节点应该放置的层(以及更低的层), 这时候就需要连边了, 这里相当于搜点裁边都在里面
    for (; level >= 0; level--) {
        add_links_starting_from(
                ptdis, pt_id, nearest, d_nearest, level, locks.data(), vt);
    }

    omp_unset_lock(&locks[pt_id]); // 释放锁

    if (pt_level > max_level) { // 如果大于当前的最大level, 将这个点作为入口点
        max_level = pt_level;
        entry_point = pt_id;
    }
}

/**************************************************************
 * Searching
 **************************************************************/

namespace {
using MinimaxHeap = HNSW::MinimaxHeap;
using Node = HNSW::Node;
using C = HNSW::C;
/** Do a BFS on the candidates list */
// 从目前的candidate中进行BFS搜索
int search_from_candidates(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersHNSW* params = nullptr) {
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    const IDSelector* sel = params ? params->sel : nullptr;

    C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) { // 对每一个candidate进行搜索, 加入到result中
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (d < threshold) { // 小于阈值加入到结果集中
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1); // 设置已经访问过
    }

    int nstep = 0;

    while (candidates.size() > 0) { // 首先拿出最小值, 然后依次进行扩展邻居
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0); // 计算比当前距离还要小的点的数目
            if (n_dis_below >= efSearch) { // 这就说明了当前点不符合规则, 直接break掉
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end); // 获取这个节点的neighbor offset

        // // baseline version
        // for (size_t j = begin; j < end; j++) {
        //     int v1 = hnsw.neighbors[j];
        //     if (v1 < 0)
        //         break;
        //     if (vt.get(v1)) {
        //         continue;
        //     }
        //     vt.set(v1);
        //     ndis++;
        //     float d = qdis(v1);
        //     if (!sel || sel->is_member(v1)) {
        //         if (nres < k) {
        //             faiss::maxheap_push(++nres, D, I, d, v1);
        //         } else if (d < D[0]) {
        //             faiss::maxheap_replace_top(nres, D, I, d, v1);
        //         }
        //     }
        //     candidates.push(v1, d);
        // }

        // the following version processes 4 neighbors at a time, 如下版本是一次性处理4个neighbor, 采用预取函数
        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;

            prefetch_L2(vt.visited.data() + v1); // 将这些数据预取
            jmax += 1;
        }

        int counter = 0;
        size_t saved_j[4];

        ndis += jmax - begin;
        threshold = res.threshold;

        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (!sel || sel->is_member(idx)) {
                if (dis < threshold) {
                    if (res.add_result(dis, idx)) {
                        threshold = res.threshold;
                    }
                }
            }
            candidates.push(idx, dis);
        };

        for (size_t j = begin; j < jmax; j++) { // 实际上的存在的neigbor
            int v1 = hnsw.neighbors[j];

            bool vget = vt.get(v1);
            vt.set(v1);
            saved_j[counter] = v1;
            counter += vget ? 0 : 1;

            if (counter == 4) { // 4批量的计算
                float dis[4];
                qdis.distances_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    add_to_heap(saved_j[id4], dis[id4]);
                }

                counter = 0;
            }
        }

        for (size_t icnt = 0; icnt < counter; icnt++) {
            float dis = qdis(saved_j[icnt]);
            add_to_heap(saved_j[icnt], dis);
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.n3 += ndis;
    }

    return nres;
}

std::priority_queue<HNSW::Node> search_from_candidate_unbounded(
        const HNSW& hnsw,
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        HNSWStats& stats) {
    int ndis = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }

        candidates.pop();

        size_t begin, end;
        hnsw.neighbor_range(v0, 0, &begin, &end);

        // // baseline version
        // for (size_t j = begin; j < end; ++j) {
        //     int v1 = hnsw.neighbors[j];
        //
        //     if (v1 < 0) {
        //         break;
        //     }
        //     if (vt->get(v1)) {
        //         continue;
        //     }
        //
        //     vt->set(v1);
        //
        //     float d1 = qdis(v1);
        //     ++ndis;
        //
        //     if (top_candidates.top().first > d1 ||
        //         top_candidates.size() < ef) {
        //         candidates.emplace(d1, v1);
        //         top_candidates.emplace(d1, v1);
        //
        //         if (top_candidates.size() > ef) {
        //             top_candidates.pop();
        //         }
        //     }
        // }

        // the following version processes 4 neighbors at a time
        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;

            prefetch_L2(vt->visited.data() + v1);
            jmax += 1;
        }

        int counter = 0;
        size_t saved_j[4];

        ndis += jmax - begin;

        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (top_candidates.top().first > dis ||
                top_candidates.size() < ef) {
                candidates.emplace(dis, idx);
                top_candidates.emplace(dis, idx);

                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        };

        for (size_t j = begin; j < jmax; j++) {
            int v1 = hnsw.neighbors[j];

            bool vget = vt->get(v1);
            vt->set(v1);
            saved_j[counter] = v1;
            counter += vget ? 0 : 1;

            if (counter == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    add_to_heap(saved_j[id4], dis[id4]);
                }

                counter = 0;
            }
        }

        for (size_t icnt = 0; icnt < counter; icnt++) {
            float dis = qdis(saved_j[icnt]);
            add_to_heap(saved_j[icnt], dis);
        }
    }

    ++stats.n1;
    if (candidates.size() == 0) {
        ++stats.n2;
    }
    stats.n3 += ndis;

    return top_candidates;
}

// just used as a lower bound for the minmaxheap, but it is set for heap search
int extract_k_from_ResultHandler(ResultHandler<C>& res) {
    using RH = HeapBlockResultHandler<C>;
    if (auto hres = dynamic_cast<RH::SingleResultHandler*>(&res)) {
        return hres->k;
    }
    return 1;
}

} // anonymous namespace

HNSWStats HNSW::search(
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        VisitedTable& vt,
        const SearchParametersHNSW* params) const {
    HNSWStats stats;
    if (entry_point == -1) {
        return stats;
    }
    int k = extract_k_from_ResultHandler(res); // 多添一个参数得了

    if (upper_beam == 1) { // beam search = 1的时候, 每一层只有一个入口
        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);
        // 在0 above搜
        for (int level = max_level; level >= 1; level--) { // 在每一层更新最近邻(0 above)
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }
        // 在0 搜
        int ef = std::max(efSearch, k);
        if (search_bounded_queue) { // this is the most common branch, 维护k-efsearch堆
            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);
            // 从candidate中进行搜索
            search_from_candidates(
                    *this, qdis, res, candidates, vt, stats, 0, 0, params);
        } else { // 非bound queue
            std::priority_queue<Node> top_candidates =
                    search_from_candidate_unbounded(
                            *this,
                            Node(d_nearest, nearest),
                            qdis,
                            ef,
                            &vt,
                            stats);

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }

            while (!top_candidates.empty()) {
                float d;
                storage_idx_t label;
                std::tie(d, label) = top_candidates.top();
                res.add_result(d, label);
                top_candidates.pop();
            }
        }

        vt.advance();

    } else { // 对于beam search
        int candidates_size = upper_beam;
        MinimaxHeap candidates(candidates_size);

        std::vector<idx_t> I_to_next(candidates_size);
        std::vector<float> D_to_next(candidates_size);

        HeapBlockResultHandler<C> block_resh(
                1, D_to_next.data(), I_to_next.data(), candidates_size);
        HeapBlockResultHandler<C>::SingleResultHandler resh(block_resh);

        int nres = 1;
        I_to_next[0] = entry_point;
        D_to_next[0] = qdis(entry_point);

        for (int level = max_level; level >= 0; level--) {
            // copy I, D -> candidates

            candidates.clear();

            for (int i = 0; i < nres; i++) {
                candidates.push(I_to_next[i], D_to_next[i]);
            }

            if (level == 0) {
                nres = search_from_candidates(
                        *this, qdis, res, candidates, vt, stats, 0);
            } else {
                resh.begin(0);
                nres = search_from_candidates(
                        *this, qdis, resh, candidates, vt, stats, level);
                resh.end();
            }
            vt.advance();
        }
    }

    return stats;
}

void HNSW::search_level_0(
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        idx_t nprobe,
        const storage_idx_t* nearest_i,
        const float* nearest_d,
        int search_type,
        HNSWStats& search_stats,
        VisitedTable& vt) const {
    const HNSW& hnsw = *this;
    int k = extract_k_from_ResultHandler(res);
    if (search_type == 1) {
        int nres = 0;

        for (int j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0)
                break;

            if (vt.get(cj))
                continue;

            int candidates_size = std::max(hnsw.efSearch, k);
            MinimaxHeap candidates(candidates_size);

            candidates.push(cj, nearest_d[j]);

            nres = search_from_candidates(
                    hnsw, qdis, res, candidates, vt, search_stats, 0, nres);
        }
    } else if (search_type == 2) {
        int candidates_size = std::max(hnsw.efSearch, int(k));
        candidates_size = std::max(candidates_size, int(nprobe));

        MinimaxHeap candidates(candidates_size);
        for (int j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0)
                break;
            candidates.push(cj, nearest_d[j]);
        }

        search_from_candidates(
                hnsw, qdis, res, candidates, vt, search_stats, 0);
    }
}

void HNSW::permute_entries(const idx_t* map) {
    // remap levels
    storage_idx_t ntotal = levels.size();
    std::vector<storage_idx_t> imap(ntotal); // inverse mapping
    // map: new index -> old index
    // imap: old index -> new index
    for (int i = 0; i < ntotal; i++) {
        assert(map[i] >= 0 && map[i] < ntotal);
        imap[map[i]] = i;
    }
    if (entry_point != -1) {
        entry_point = imap[entry_point];
    }
    std::vector<int> new_levels(ntotal);
    std::vector<size_t> new_offsets(ntotal + 1);
    std::vector<storage_idx_t> new_neighbors(neighbors.size());
    size_t no = 0;
    for (int i = 0; i < ntotal; i++) {
        storage_idx_t o = map[i]; // corresponding "old" index
        new_levels[i] = levels[o];
        for (size_t j = offsets[o]; j < offsets[o + 1]; j++) {
            storage_idx_t neigh = neighbors[j];
            new_neighbors[no++] = neigh >= 0 ? imap[neigh] : neigh;
        }
        new_offsets[i + 1] = no;
    }
    assert(new_offsets[ntotal] == offsets[ntotal]);
    // swap everyone
    std::swap(levels, new_levels);
    std::swap(offsets, new_offsets);
    std::swap(neighbors, new_neighbors);
}

/**************************************************************
 * MinimaxHeap
 **************************************************************/
/// 堆的push操作, 这里如果用在搜索过程, 应该会使用大根堆
void HNSW::MinimaxHeap::push(storage_idx_t i, float v) {
    if (k == n) {
        if (v >= dis[0])
            return;
        if (ids[0] != -1) {
            --nvalid;
        }
        faiss::heap_pop<HC>(k--, dis.data(), ids.data());
    }
    faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
    ++nvalid;
}

float HNSW::MinimaxHeap::max() const {
    return dis[0];
}

int HNSW::MinimaxHeap::size() const {
    return nvalid;
}

void HNSW::MinimaxHeap::clear() {
    nvalid = k = 0;
}

#ifdef __AVX2__
int HNSW::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    static_assert(
            std::is_same<storage_idx_t, int32_t>::value,
            "This code expects storage_idx_t to be int32_t");

    int32_t min_idx = -1;
    float min_dis = std::numeric_limits<float>::infinity();

    size_t iii = 0;

    __m256i min_indices = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);
    __m256 min_distances =
            _mm256_set1_ps(std::numeric_limits<float>::infinity());
    __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i offset = _mm256_set1_epi32(8);

    // The baseline version is available in non-AVX2 branch.

    // The following loop tracks the rightmost index with the min distance.
    // -1 index values are ignored.
    const int k8 = (k / 8) * 8;
    for (; iii < k8; iii += 8) {
        __m256i indices =
                _mm256_loadu_si256((const __m256i*)(ids.data() + iii));
        __m256 distances = _mm256_loadu_ps(dis.data() + iii);

        // This mask filters out -1 values among indices.
        __m256i m1mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), indices);

        __m256i dmask = _mm256_castps_si256(
                _mm256_cmp_ps(min_distances, distances, _CMP_LT_OS));
        __m256 finalmask = _mm256_castsi256_ps(_mm256_or_si256(m1mask, dmask));

        const __m256i min_indices_new = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(current_indices),
                _mm256_castsi256_ps(min_indices),
                finalmask));

        const __m256 min_distances_new =
                _mm256_blendv_ps(distances, min_distances, finalmask);

        min_indices = min_indices_new;
        min_distances = min_distances_new;

        current_indices = _mm256_add_epi32(current_indices, offset);
    }

    // Vectorizing is doable, but is not practical
    int32_t vidx8[8];
    float vdis8[8];
    _mm256_storeu_ps(vdis8, min_distances);
    _mm256_storeu_si256((__m256i*)vidx8, min_indices);

    for (size_t j = 0; j < 8; j++) {
        if (min_dis > vdis8[j] || (min_dis == vdis8[j] && min_idx < vidx8[j])) {
            min_idx = vidx8[j];
            min_dis = vdis8[j];
        }
    }

    // process last values. Vectorizing is doable, but is not practical
    for (; iii < k; iii++) {
        if (ids[iii] != -1 && dis[iii] <= min_dis) {
            min_dis = dis[iii];
            min_idx = iii;
        }
    }

    if (min_idx == -1) {
        return -1;
    }

    if (vmin_out) {
        *vmin_out = min_dis;
    }
    int ret = ids[min_idx];
    ids[min_idx] = -1;
    --nvalid;
    return ret;
}

#else

// baseline non-vectorized version
int HNSW::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    // returns min. This is an O(n) operation
    int i = k - 1;
    while (i >= 0) {
        if (ids[i] != -1) {
            break;
        }
        i--;
    }
    if (i == -1) {
        return -1;
    }
    int imin = i;
    float vmin = dis[i];
    i--;
    while (i >= 0) {
        if (ids[i] != -1 && dis[i] < vmin) {
            vmin = dis[i];
            imin = i;
        }
        i--;
    }
    if (vmin_out) {
        *vmin_out = vmin;
    }
    int ret = ids[imin];
    ids[imin] = -1;
    --nvalid;

    return ret;
}
#endif

int HNSW::MinimaxHeap::count_below(float thresh) {
    int n_below = 0;
    for (int i = 0; i < k; i++) {
        if (dis[i] < thresh) {
            n_below++;
        }
    }

    return n_below;
}

} // namespace faiss
