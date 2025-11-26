//
// Created by salehm32 on 2025-11-17.
//

#pragma once
#include <vector>
#include <set>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include "cuda_dag_builder_viz.h"

template<typename T>
T* cp_vec(std::vector<T> vec){
    auto *data = new T[vec.size()];
    for (int i = 0; i < vec.size(); i++){
        data[i] = vec[i];
    }
    return data;
}

// -----------------------
// CSC Matrix definition
// -----------------------
template<typename T>
struct CSCMatrix {
    int n;  // square matrix n x n
    int* indptr;   // size n+1
    int* indices;  // size nnz
    T* data;       // size nnz
    CSCMatrix(int n, std::vector<int> indptr, std::vector<int> indices, std::vector<T> data){
        this->n = n;
        this->indptr = cp_vec(indptr);
        this->indices = cp_vec(indices);
        this->data = cp_vec(data);
    }
    CSCMatrix(){
        this->indptr = nullptr;
        this->indices = nullptr;
        this->data = nullptr;
    }
    ~CSCMatrix(){
        delete[] this->indptr;
        delete[] this->indices;
        delete[] this->data;
    }
};

// -----------------------
// Supernodal output
// -----------------------
template<typename T>
struct SupernodalMatrix {
    std::vector<T> x;              // data
    std::vector<int> i;            // Li: row indices
    std::vector<int> ip;           // Lip: row pointer per supernode
    std::vector<int> p;            // Lp: column pointer per column
    std::vector<int> supernodes;   // start index of each supernode
    std::vector<int> col_to_super; // mapping column -> supernode
    int rows, cols;
    int nSuper;
    int max_width;
};



template<typename T>
std::vector<std::vector<int>>
detect_supernodes(const CSCMatrix<T>& L, int relaxation)
{
    int n = L.n;

    std::vector<std::vector<int>> supernodes;
    std::vector<int> cur_sn;

    // Loop over columns i = 0 .. n-2
    for (int i = 0; i < n - 1; i++) {

        // col_i pattern = L.indices[L.indptr[i] : L.indptr[i+1]]
        std::set<int> col_i_p;
        for (int k = L.indptr[i]; k < L.indptr[i + 1]; k++)
            col_i_p.insert(L.indices[k]);

        // col_{i+1} pattern = L.indices[L.indptr[i+1] : L.indptr[i+2]]
        std::set<int> col_ipone_p;
        for (int k = L.indptr[i + 1]; k < L.indptr[i + 2]; k++)
            col_ipone_p.insert(L.indices[k]);

        // union and intersection
        std::set<int> union_set;
        std::set<int> intersect_set;

        // union(col_i_p, col_{i+1}_p)
        std::set_union(col_i_p.begin(), col_i_p.end(),
                       col_ipone_p.begin(), col_ipone_p.end(),
                       std::inserter(union_set, union_set.begin()));

        // intersection(col_i_p, col_{i+1}_p)
        std::set_intersection(col_i_p.begin(), col_i_p.end(),
                              col_ipone_p.begin(), col_ipone_p.end(),
                              std::inserter(intersect_set, intersect_set.begin()));

        // Python condition:
        // if len(union) - len(intersect) <= relaxation + 1:
        if ((int)union_set.size() - (int)intersect_set.size() <= relaxation + 1) {
            cur_sn.push_back(i);
        } else {
            if (!cur_sn.empty()) {
                cur_sn.push_back(i);
                supernodes.push_back(cur_sn);
                cur_sn.clear();
            } else {
                supernodes.push_back({i});
            }
        }
    }

    // handle last column (n-1)
    if (!cur_sn.empty()) {
        cur_sn.push_back(n - 1);
        supernodes.push_back(cur_sn);
    } else {
        supernodes.push_back({n - 1});
    }

    // Verify that all columns 0..n-1 appear exactly once
    std::set<int> all_cols;
    for (const auto& sn : supernodes)
        for (int c : sn)
            all_cols.insert(c);

    if ((int)all_cols.size() != n)
        throw std::runtime_error("detect_supernodes: sanity check failed");

    return supernodes;
}

//// You must implement this based on your Python version
//extern std::vector<std::vector<int>>
//detect_supernodes(const CSCMatrix<double>& L, int relaxation);

// ---------------------------------------------------------------
//  convert_csc_to_supernodal: 1:1 faithful translation from Python
// ---------------------------------------------------------------
template<typename T>
SupernodalMatrix<T> convert_csc_to_supernodal(
        const CSCMatrix<T>& L,
        int relaxation = 0)
{
    SupernodalMatrix<T> Ls;

    int n = L.n;

    // -------- step 1: detect supernodes ----------
    auto supernodes = detect_supernodes(L, relaxation);

    // Variables identical to Python
    std::vector<std::set<int>> row_patterns;
    std::vector<int> Li;  // actual row indices
    std::vector<int> Lip; // row pointer per supernode
    std::vector<int> Lp;  // column pointer per column

    Lip.push_back(0);
    Lp.push_back(0);

    int max_width = 0;

    // ------------- step 2 ------------------
    for (const auto& sn : supernodes) {
        max_width = std::max(max_width, (int)sn.size());

        std::set<int> rp_union;

        // union of row patterns of each column in this supernode
        for (int i : sn) {
            for (int k = L.indptr[i]; k < L.indptr[i+1]; k++)
                rp_union.insert(L.indices[k]);
        }

        // update Lp for each column j in the supernode
        for (int i : sn) {
            int last = Lp.back();
            Lp.push_back(last + rp_union.size());
        }

        row_patterns.push_back(rp_union);

        // append sorted union into Li
        for (int r : rp_union)
            Li.push_back(r);

        Lip.push_back((int)Li.size());
    }

    // ------------- step 3 ------------------
    std::vector<int> super_begin(supernodes.size() + 1, 0);

    for (size_t i = 0; i < supernodes.size(); i++)
        super_begin[i] = supernodes[i][0];

    super_begin.back() = n;

    // col -> supernode mapping
    std::vector<int> col_to_super(n, 0);
    for (size_t i = 0; i < supernodes.size(); i++)
        for (int j : supernodes[i])
            col_to_super[j] = i;

    int lnnz = Lp.back();

    std::vector<T> data(lnnz, T(0));
    std::vector<int> lookup_table(n, 0);

    // fill "data" array exactly like Python version
    for (size_t i = 0; i < supernodes.size(); i++) {

        // Build lookup table: row_index -> offset inside block
        for (int m = Lip[i]; m < Lip[i+1]; m++)
            lookup_table[ Li[m] ] = m - Lip[i];

        // iterate over columns j of this supernode
        for (int j = super_begin[i]; j < super_begin[i+1]; j++) {

            // iterate CSC column j
            for (int k = L.indptr[j]; k < L.indptr[j+1]; k++) {
                int row_idx = L.indices[k];

                int offset = lookup_table[row_idx];
                data[Lp[j] + offset] = L.data[k];
            }
        }
    }

    // ---------- populate supernodal structure -------------
    Ls.x = std::move(data);
    Ls.i = std::move(Li);
    Ls.ip = std::move(Lip);
    Ls.p = std::move(Lp);
    Ls.supernodes = std::move(super_begin);
    Ls.col_to_super = std::move(col_to_super);
    Ls.rows = n;
    Ls.cols = n;
    Ls.nSuper = supernodes.size();
    Ls.max_width = max_width;
//    std::cout << Ls.supernodes[0] << std::endl;
    return Ls;
}




template<typename T>
CSCMatrix<int>* build_supernodal_dag_csc(const SupernodalMatrix<T>& Ls)
{
    const std::vector<int>& Lip = Ls.ip;        // row pointer per supernode
    const std::vector<int>& Li  = Ls.i;         // row indices concatenated per supernode
    const std::vector<int>& Lp  = Ls.p;         // column pointer per column
    const std::vector<int>& col_to_super = Ls.col_to_super;
    const std::vector<int>& super_begin = Ls.supernodes;

    int m = Ls.nSuper;
    if ((int)Lip.size() < m+1) throw std::runtime_error("build_supernodal_dag_csc: Lip size inconsistent");
    if ((int)col_to_super.size() < Ls.cols) throw std::runtime_error("build_supernodal_dag_csc: col_to_super size inconsistent");

    std::vector<std::vector<int>> DAG(m);

    int nnz_scan = 0;
    for (int i = 0; i < m; ++i) {
        // For every k in Lip[i] .. Lip[i+1]-1, get row_idx = Li[k], map to supernode
        std::set<int> dests;
        for (int k = Lip[i]; k < Lip[i+1]; ++k) {
            int row_idx = Li[k];
            int dest_sup = col_to_super[row_idx];
            dests.insert(dest_sup);
            ++nnz_scan; // mirrors Python's nnz += 1 per k (scanned entries)
        }
        // copy set into vector (sorted by construction)
        DAG[i].assign(dests.begin(), dests.end());
    }

    // compute total number of edges (sum of per-node adjacency sizes)
    int total_edges = 0;
    for (int i = 0; i < m; ++i) total_edges += static_cast<int>(DAG[i].size());

    // build CSC representation Dp (size m+1), Di (size total_edges), data (ones)
    std::vector<int> Dp(m + 1, 0);
    for (int i = 0; i < m; ++i) {
        Dp[i+1] = Dp[i] + static_cast<int>(DAG[i].size());
    }
    std::vector<int> Di;
    Di.reserve(total_edges);
    for (int i = 0; i < m; ++i) {
        // in python: Di[Dp[i]:Dp[i+1]] = DAG[i]
        for (int v : DAG[i]) Di.push_back(v);
    }

    // values: ones (int)
    std::vector<int> data(total_edges, 1.);

    auto *D = new CSCMatrix<int>(m, Dp, Di, data);
    return D;
}

template<typename T>
using Wavefronts = std::vector<std::vector<T>>;

Wavefronts<int> compute_wavefront(const CSCMatrix<int>& L) {
    int n = L.n;
//    if (L.n + 1 != (size_t)(n + 1))
//        throw std::runtime_error("compute_wavefront: CSC indptr size mismatch");

    std::vector<int> incoming_edges(n, 0);

    // -------- 1. Count incoming edges (identical to Python) --------
    for (int i = 0; i < n; ++i) {
        for (int j = L.indptr[i]; j < L.indptr[i + 1]; ++j) {
            int dest = L.indices[j];
            incoming_edges[dest] += 1;
        }
    }

    // -------- 2. Initial wavefront: nodes with incoming_edges == 1 --------
    std::vector<int> wavefront;
    wavefront.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (incoming_edges[i] == 1)
            wavefront.push_back(i);
    }

    Wavefronts<int> wavefront_list;

    // -------- 3. Iteratively peel layers --------
    while (!wavefront.empty()) {

        wavefront_list.push_back(wavefront);

        std::vector<int> new_wavefront;
        new_wavefront.reserve(n);

        for (int node : wavefront) {

            int begin = L.indptr[node] + 1;
            int end   = L.indptr[node + 1];

            for (int j = begin; j < end; ++j) {
                int child = L.indices[j];
                incoming_edges[child] -= 1;

                if (incoming_edges[child] == 1) {
                    new_wavefront.push_back(child);
                }
            }
        }

        wavefront = std::move(new_wavefront);
    }
    return wavefront_list;
}

struct Compact2DList {
    std::vector<int> wave_start_loc; // size = num_wavefronts + 1
    std::vector<int> waves;          // flattened wavefront entries
};

Compact2DList compact_2d_list(const std::vector<std::vector<int>>& wavefront_list)
{
    Compact2DList out;

    const size_t num_waves = wavefront_list.size();
    // ------------------------------
    // (1) Flatten wavefront_list
    // waves = [element for sublist in wavefront_list for element in sublist]
    // ------------------------------
    size_t total_entries = 0;
    for (const auto& wf : wavefront_list)
        total_entries += wf.size();

    out.waves.reserve(total_entries);

    for (const auto& wf : wavefront_list)
        for (int v : wf)
            out.waves.push_back(v);

    // ------------------------------
    // (2) Compute wave_start_loc
    // Same as Python:
    //
    // wave_start_loc = np.zeros(len(wavefront_list)+1)
    // wave_start_loc[i] = wave_start_loc[i-1] + len(wavefront_list[i-1])
    // ------------------------------
    out.wave_start_loc.resize(num_waves + 1);
    out.wave_start_loc[0] = 0;

    for (size_t i = 1; i <= num_waves; ++i) {
        out.wave_start_loc[i] =
                out.wave_start_loc[i - 1] + wavefront_list[i - 1].size();
    }

    return out;
}

template <typename T>
std::vector<std::vector<int>>
build_dependency_graph(const SupernodalMatrix<T>& Ls)
{
    const std::vector<int>& Lip = Ls.ip;            // supernode row ptr
    const std::vector<int>& Li  = Ls.i;             // row indices
    const std::vector<int>& col_to_super = Ls.col_to_super;
    int m = Ls.nSuper;

    if ((int)Lip.size() < m+1)
        throw std::runtime_error("build_dependency_graph: Lip size inconsistent");
    if ((int)col_to_super.size() < Ls.cols)
        throw std::runtime_error("build_dependency_graph: col_to_super size inconsistent");

    // --------------------------------------------------------
    // BUILD ORIGINAL DAG (adjacency list)
    // --------------------------------------------------------
    std::vector<std::vector<int>> DAG(m);

    for (int i = 0; i < m; ++i)
    {
        std::set<int> dests;   // avoid duplicates
        for (int k = Lip[i]; k < Lip[i+1]; ++k)
        {
            int row_idx = Li[k];
            int dest_sup = col_to_super[row_idx];
            dests.insert(dest_sup);
        }
        DAG[i].assign(dests.begin(), dests.end()); // sorted by set
    }

    // --------------------------------------------------------
    // INVERT THE EDGES
    // If DAG has u → v, then dependency graph has v → u
    // Meaning: "v depends on u"
    // --------------------------------------------------------
    std::vector<std::vector<int>> deps(m);

    for (int u = 0; u < m; ++u) {
        for (int v : DAG[u]) {
            deps[v].push_back(u);   // v depends on u
        }
    }
    return deps;
}
