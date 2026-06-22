// Copyright © 2026 Alexander Taffe

#pragma once
#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <algorithm>

// Minimal dense linear algebra for the tracker. Everything here is <= 8x8, so a
// plain row-major matrix with naive O(n^3) ops is more than fast enough — and it
// compiles in milliseconds (unlike Eigen, which thrashes swap on the Pi Zero 2 W).
namespace byte_track {

using Vec4 = std::array<double, 4>;

struct Mat {
    int r = 0, c = 0;
    std::vector<double> d;

    Mat() = default;
    Mat(int rows, int cols, double val = 0.0) : r(rows), c(cols), d(static_cast<size_t>(rows) * cols, val) {}

    double& operator()(int i, int j) { return d[static_cast<size_t>(i) * c + j]; }
    double operator()(int i, int j) const { return d[static_cast<size_t>(i) * c + j]; }

    static Mat eye(int n) {
        Mat m(n, n);
        for (int i = 0; i < n; ++i) m(i, i) = 1.0;
        return m;
    }
};

inline Mat matmul(const Mat& A, const Mat& B) {
    Mat C(A.r, B.c);
    for (int i = 0; i < A.r; ++i)
        for (int k = 0; k < A.c; ++k) {
            const double a = A(i, k);
            if (a == 0.0) continue;
            for (int j = 0; j < B.c; ++j) C(i, j) += a * B(k, j);
        }
    return C;
}

inline Mat transpose(const Mat& A) {
    Mat T(A.c, A.r);
    for (int i = 0; i < A.r; ++i)
        for (int j = 0; j < A.c; ++j) T(j, i) = A(i, j);
    return T;
}

inline Mat operator+(const Mat& A, const Mat& B) {
    Mat C(A.r, A.c);
    for (size_t i = 0; i < A.d.size(); ++i) C.d[i] = A.d[i] + B.d[i];
    return C;
}

inline Mat operator-(const Mat& A, const Mat& B) {
    Mat C(A.r, A.c);
    for (size_t i = 0; i < A.d.size(); ++i) C.d[i] = A.d[i] - B.d[i];
    return C;
}

// Solve A X = B for X, with A (n x n) and B (n x m). Gauss-Jordan with partial pivot.
inline Mat solve(Mat A, Mat B) {
    const int n = A.r;
    const int m = B.c;
    for (int col = 0; col < n; ++col) {
        int piv = col;
        double best = std::fabs(A(col, col));
        for (int i = col + 1; i < n; ++i) {
            const double v = std::fabs(A(i, col));
            if (v > best) { best = v; piv = i; }
        }
        if (piv != col) {
            for (int j = 0; j < n; ++j) std::swap(A(col, j), A(piv, j));
            for (int j = 0; j < m; ++j) std::swap(B(col, j), B(piv, j));
        }
        const double diag = A(col, col);
        for (int i = 0; i < n; ++i) {
            if (i == col) continue;
            const double f = A(i, col) / diag;
            if (f == 0.0) continue;
            for (int j = col; j < n; ++j) A(i, j) -= f * A(col, j);
            for (int j = 0; j < m; ++j) B(i, j) -= f * B(col, j);
        }
    }
    Mat X(n, m);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) X(i, j) = B(i, j) / A(i, i);
    return X;
}

}  // namespace byte_track