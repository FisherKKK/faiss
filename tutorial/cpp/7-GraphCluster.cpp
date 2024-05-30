/**
* Copyright (c) Facebook, Inc. and its affiliates.
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/

#include <faiss/IndexGraphCluster.h>
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

using idx_t = faiss::idx_t;

int main() {
    int d = 128;      // dimension
    int nb = 1000000; // database size
    int nq = 10000;  // nb of queries



    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    fvecs_read("/home/dataset/sift/sift_base.fvecs", xb, nb, d);
    fvecs_read("/home/dataset/sift/sift_query.fvecs", xq, nq, d);

    double sample_rate = 0.1;
    int duplicate = 4;
    int k = 1;


    idx_t *I_G = new idx_t[nq * k];
    float *D_G = new float[nq * k];

    {
        printf("**************Flat Index****************\n");
        faiss::IndexFlatL2 index(d);
        index.add(nb, xb);
        index.search(nq, xq, k, D_G, I_G);
    }


    {
        std::fstream fout("/tmp/1.txt", std::ios_base::out);
        printf("**************HNSW-IVF Index****************\n");
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];
        faiss::IndexFlatL2 storage(d);
        faiss::IndexGraphCluster index(&storage, sample_rate * nb, duplicate, 32);

        double t1 = faiss::getmillisecs();
        index.train(nb, xb);
        index.add(nb, xb);
        double t2 = faiss::getmillisecs();
        fout << t2 - t1 << "\n";

        printf("Time cost for HNSW-Graph: %lf\n", t2 - t1);
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

    }


   delete[] xb;
   delete[] xq;

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
