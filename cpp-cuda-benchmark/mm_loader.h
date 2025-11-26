//
// Created by salehm32 on 2025-11-17.
//
#pragma once
#include <Eigen/Sparse>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

inline Eigen::SparseMatrix<float>
load_matrix_market(const std::string &filename)
{
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("Cannot open file: " + filename);

    std::string line;

    // Skip comments
    do {
        std::getline(in, line);
    } while (line[0] == '%');

    // Read header: rows, cols, nnz
    std::istringstream header(line);
    int rows, cols, nnz;
    header >> rows >> cols >> nnz;

    std::vector<Eigen::Triplet<float>> triplets;

    triplets.reserve(nnz);

    int r, c;
    float v;

    // Read COO entries (1-based indexing)
    while (in >> r >> c >> v) {
        triplets.emplace_back(r - 1, c - 1, v);
    }
    Eigen::SparseMatrix<float, Eigen::ColMajor> A(rows, cols);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}