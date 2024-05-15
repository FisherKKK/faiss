//
// Created by ECNU on 2024/3/20.
//
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/utils.h>
#include <fstream>
#include <iostream>

void fvecs_read(const char *path, float vecs[], int n, int d);

using idx_t = faiss::idx_t;

float calculate_recall_k(idx_t *x, idx_t *y, int k) {
    float number = 0.f;
    std::unordered_set<idx_t> set;
    for (size_t i = 0; i != k; i++) set.insert(x[i]);
    for (size_t i = 0; i != k; i++)
        if (set.count(y[i])) number += 1;
    return number / k;
}

int main() {
    int d = 128;      // dimension
    int nb = 1000000; // database size
    int nq = 10000;  // nb of queries



    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    /*
    fvecs_read("/data/sift/sift_base.fvecs", xb, nb, d);
    fvecs_read("/data/sift/sift_query.fvecs", xq, nq, d);
     */

    // random data
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
//        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
//        xq[d * i] += i / 1000.;
    }
     //

    double sample_rate = 0.1;
    int seed = 12345;
    int k = 5000;

    idx_t *I_G = new idx_t[nq * k];
    float *D_G = new float[nq * k];

    {
        printf("**************Flat Index****************\n");
        faiss::IndexFlatL2 index(d);
        index.add(nb, xb);
        index.search(nq, xq, k, D_G, I_G);
    }



    /******************************IVF**************************/
//    {
//        printf("**************IVF Index****************\n");
//        idx_t* I = new idx_t[k * nq];
//        float* D = new float[k * nq];
//        faiss::IndexFlatL2 quantizer(d);
//        faiss::IndexIVFFlat index(&quantizer, d, nb * sample_rate);
//        double t1 = faiss::getmillisecs();
//        index.train(nb, xb);
//        index.add(nb, xb); // 这里是添加每一个点
//        double t2 = faiss::getmillisecs();
//        printf("Time cost for k-means: %lf\n", t2 - t1);
//
//        for (size_t i = 1; i < 200; i += 10) {
//            index.nprobe = i; // 设置探测数
//            t1 = faiss::getmillisecs();
//            index.search(nq, xq, k, D, I);
//            t2 = faiss::getmillisecs();
//            float avg_recall = 0.f;
//            for (size_t j = 0; j != nq; j++) {
//                idx_t *qi = I + k * j;
//                idx_t *gi = I_G + k * j;
//                avg_recall += calculate_recall_k(qi, gi, k);
//            }
//
//           avg_recall /= nq;
//
//            printf("Recall@%d-probe@%ld: (Time, Recall)/(%lf, %f)\n", k, i, t2 - t1, avg_recall);
//        }
//
//        delete[] I;
//        delete[] D;
//    }



    /******************************HNSW**************************/
    /*
        std::fstream fout("/tmp/hnsw_m32_uniform.txt", std::ios_base::out);

        {
            printf("**************HNSW Index****************\n");
            idx_t* I = new idx_t[k * nq];
            float* D = new float[k * nq];
            faiss::IndexHNSWFlat index(d, 32);
            double t1 = faiss::getmillisecs();
//            index.hnsw.efConstruction = 50;
            index.train(nb, xb);
            index.add(nb, xb); // 这里是添加每一个点
            double t2 = faiss::getmillisecs();
            printf("Time cost for build HNSW: %lf\n", t2 - t1);
            fout << t2 - t1 << "\n";

            for (size_t i = 10; i < 5000; i += 100) {
                index.hnsw.efSearch = i; // 设置探测数
                t1 = faiss::getmillisecs();
                index.search(nq, xq, k, D, I);
                t2 = faiss::getmillisecs();
                float avg_recall = 0.f;
                for (size_t j = 0; j != nq; j++) {
                    idx_t *qi = I + k * j;
                    idx_t *gi = I_G + k * j;
                    avg_recall += calculate_recall_k(qi, gi, k);
                }

               avg_recall /= nq;

               printf("Recall@%d-efSearch@%ld: (Time, Recall)/(%lf, %f)\n", k, i, t2 - t1, avg_recall);
               fout << "(" << t2 - t1 << ", " << avg_recall << ")" << ", ";
               fout << std::flush;
            }

            delete[] I;
            delete[] D;
        }
        */


    {
        std::fstream fout("/tmp/hnsw_m32_ivf0_ef1024_uniform.txt", std::ios_base::out);
        printf("**************HNSW-IVF Index****************\n");
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];
        int M = 32; // Index的连边数目
        faiss::IndexHNSWFlat quantizer(d, M);
        quantizer.hnsw.efSearch = 32;
//        faiss::IndexFlatL2 scan(d);
        faiss::IndexIVFFlat index(&quantizer, d, nb * sample_rate);
//        index.clustering_index = &scan;

        double t1 = faiss::getmillisecs();
//        index.quantizer_trains_alone = 0;
        index.cp.niter = 0;
        index.quantizer_trains_alone = 0;
        index.train(nb, xb);
        index.add(nb, xb); // 这里是添加每一个点
        double t2 = faiss::getmillisecs();
        fout << t2 - t1 << "\n";
        printf("Time cost for HNSW-Graph: %lf\n", t2 - t1);
        quantizer.hnsw.efSearch = 2048;

//        index.clustering_index = nullptr;
        for (size_t i = 1; i <= 30000; i *= 2) {
            index.nprobe = i; // 设置探测数
            t1 = faiss::getmillisecs();
            index.search(nq, xq, k, D, I);
            t2 = faiss::getmillisecs();


            float avg_recall = 0.f;
            for (size_t j = 0; j != nq; j++) {
                idx_t *qi = I + k * j;
                idx_t *gi = I_G + k * j;
                avg_recall += calculate_recall_k(qi, gi, k);
            }
            avg_recall /= nq;
            fout << "(" << t2 - t1 << ", " << avg_recall << ")" << ", " << std::flush;
            printf("Recall@%d-probe@%ld: (Time, Recall)/(%lf, %f)\n", k, i, t2 - t1, avg_recall);
        }

        delete[] I;
        delete[] D;
        fout.close();
















        // 首先我们需要对数据进行采样
//        idx_t *topk = new idx_t[nb * r];
//        float *dis = new float[nb * r];
//
//        std::vector<int> perm(nb);
//        faiss::rand_perm(perm.data(), nb, seed); // 这里也相当于是一个映射表
//
//        int point_number = static_cast<int>(sample_rate * nb);
//
//        float *sample_data = new float[point_number * d];
//        for (size_t i = 0; i != point_number; i++) {
//            memcpy(sample_data + i * d, xb + perm[i] * d, sizeof(float) * d);
//        }
//
//
//        double t1 = faiss::getmillisecs();
//        faiss::IndexHNSWFlat index(d, M);
//        index.add(point_number, sample_data); // 这里就相当于已经生成了索引
//
//        std::vector<std::vector<idx_t>> ivf(point_number);
//        index.search(nb, xb, k, dis, topk);
//
//        for (size_t i = 0; i < nb; i++) {
//            assert(topk[i] >= 0);
//            ivf[topk[i]].push_back(i);
//        }
//        double t2 = faiss::getmillisecs();
//        printf("Time cost for HNSW-Graph: %lf\n", t2 - t1);



//        delete[] sample_data;
//        delete[] topk;
//        delete[] dis;

    }


    delete[] xb;
    delete[] xq;
    delete[] D_G;
    delete[] I_G;

    return 0;
}

void fvecs_read(const char *path, float vecs[], int n, int d) {
    std::fstream fin(path, std::ios_base::binary | std::ios_base::in);
    if (!fin) {
            std::cerr << "Can't open file: " << path << std::endl;
            exit(1);
    }

    int df; // read dimension in the file
    fin.read((char*)&df, sizeof(df));
    assert(df == d);

    int dummy; // dummy for one
    for (int i = 0; i < n; i++) {
            float *vec_i = vecs + i * d;
            fin.read((char*)vec_i, sizeof(*vecs) * d);
            fin.read((char*)&dummy, sizeof(dummy));
    }
}


