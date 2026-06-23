// Copyright © 2026 Alexander Taffe

#pragma once
#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <algorithm>

/**
 * @file linalg.h
 * @brief Minimal dense linear algebra for the tracker.
 *
 * Everything here operates on matrices that are at most 8x8, so a plain
 * row-major matrix with naive O(n^3) operations is more than fast enough — and
 * it compiles in milliseconds (unlike Eigen, which thrashes swap on the
 * Raspberry Pi Zero 2 W).
 */
namespace byte_track {

/// Fixed 4-element vector, used for bounding boxes (tlbr / tlwh / xyah) and
/// 4-dimensional measurements.
using Vec4 = std::array<double, 4>;

/**
 * @brief Dense row-major matrix of doubles.
 *
 * Storage is a flat @ref d vector of size @c r*c indexed in row-major order.
 * All linear-algebra helpers in this header operate on this type.
 */
struct Mat {
    int r = 0;               ///< Number of rows.
    int c = 0;               ///< Number of columns.
    std::vector<double> d;   ///< Row-major element storage (size @c r*c).

    /// Construct an empty 0x0 matrix.
    Mat() = default;

    /**
     * @brief Construct a @p rows x @p cols matrix filled with @p val.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param val  Value written to every element (defaults to 0).
     */
    Mat(int rows, int cols, double val = 0.0) : r(rows), c(cols), d(static_cast<size_t>(rows) * cols, val) {}

    /**
     * @brief Mutable element access.
     * @param i Row index.
     * @param j Column index.
     * @return Reference to the element at (@p i, @p j).
     */
    double& operator()(int i, int j) { return d[static_cast<size_t>(i) * c + j]; }

    /**
     * @brief Const element access.
     * @param i Row index.
     * @param j Column index.
     * @return Value of the element at (@p i, @p j).
     */
    double operator()(int i, int j) const { return d[static_cast<size_t>(i) * c + j]; }

    /**
     * @brief Build an @p n x @p n identity matrix.
     * @param n Dimension.
     * @return The identity matrix of size @p n.
     */
    static Mat eye(int n) {
        Mat m(n, n);
        for (int i = 0; i < n; ++i) m(i, i) = 1.0;
        return m;
    }
};

/**
 * @brief Matrix product @p A * @p B.
 * @param A Left operand (@c A.r x @c A.c).
 * @param B Right operand (@c A.c x @c B.c).
 * @return The (@c A.r x @c B.c) product. Inner dimensions are assumed to match.
 */
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

/**
 * @brief Transpose of @p A.
 * @param A Input matrix (@c A.r x @c A.c).
 * @return The (@c A.c x @c A.r) transpose.
 */
inline Mat transpose(const Mat& A) {
    Mat T(A.c, A.r);
    for (int i = 0; i < A.r; ++i)
        for (int j = 0; j < A.c; ++j) T(j, i) = A(i, j);
    return T;
}

/**
 * @brief Element-wise sum @p A + @p B.
 * @param A First operand.
 * @param B Second operand (same shape as @p A).
 * @return The element-wise sum.
 */
inline Mat operator+(const Mat& A, const Mat& B) {
    Mat C(A.r, A.c);
    for (size_t i = 0; i < A.d.size(); ++i) C.d[i] = A.d[i] + B.d[i];
    return C;
}

/**
 * @brief Element-wise difference @p A - @p B.
 * @param A First operand.
 * @param B Second operand (same shape as @p A).
 * @return The element-wise difference.
 */
inline Mat operator-(const Mat& A, const Mat& B) {
    Mat C(A.r, A.c);
    for (size_t i = 0; i < A.d.size(); ++i) C.d[i] = A.d[i] - B.d[i];
    return C;
}

/**
 * @brief Solve the linear system @c A*X = B for @c X.
 *
 * Uses Gauss-Jordan elimination with partial pivoting. @p A and @p B are taken
 * by value because they are mutated in place during elimination.
 *
 * @param A Square coefficient matrix (n x n).
 * @param B Right-hand side (n x m).
 * @return The solution @c X (n x m).
 */
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
