// Copyright © 2026 Alexander Taffe

#include "tracking/matching.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace byte_track {
    
double iou(const Vec4& A, const Vec4& B) {
    const double iw = std::min(A[2], B[2]) - std::max(A[0], B[0]) + 1.0;
    if (iw <= 0) return 0.0;
    const double ih = std::min(A[3], B[3]) - std::max(A[1], B[1]) + 1.0;
    if (ih <= 0) return 0.0;
    const double inter = iw * ih;
    const double area_a = (A[2] - A[0] + 1.0) * (A[3] - A[1] + 1.0);
    const double area_b = (B[2] - B[0] + 1.0) * (B[3] - B[1] + 1.0);
    return inter / (area_a + area_b - inter);
}

Mat iou_distance(const std::vector<STrackPtr>& a, const std::vector<STrackPtr>& b) {
    Mat cost(static_cast<int>(a.size()), static_cast<int>(b.size()));
    for (size_t i = 0; i < a.size(); ++i) {
        const Vec4 at = a[i]->tlbr();
        for (size_t j = 0; j < b.size(); ++j)
            cost(static_cast<int>(i), static_cast<int>(j)) = 1.0 - iou(at, b[j]->tlbr());
    }
    return cost;
}

// Jonker-Volgenant is what lap.lapjv uses; for the tiny matrices here (<=~20) a dense
// O(n^3) Hungarian (Kuhn-Munkres, e-maxx variant) finds the same optimal assignment.
// Padded to a square matrix so rectangular inputs are handled.
static std::vector<int> hungarian(const Mat& cost, int rows, int cols) {
    std::vector<int> row_assign(rows, -1);
    if (rows == 0 || cols == 0) return row_assign;

    const double BIG = 1e9;
    const int N = std::max(rows, cols);
    std::vector<std::vector<double>> a(N + 1, std::vector<double>(N + 1, 0.0));
    for (int i = 1; i <= N; ++i)
        for (int j = 1; j <= N; ++j) {
            double v = BIG;
            if (i <= rows && j <= cols) {
                v = cost(i - 1, j - 1);
                if (!std::isfinite(v)) v = BIG;
            }
            a[i][j] = v;
        }

    const double INF = std::numeric_limits<double>::max() / 4;
    std::vector<double> u(N + 1, 0.0), v(N + 1, 0.0);
    std::vector<int> p(N + 1, 0), way(N + 1, 0);
    for (int i = 1; i <= N; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(N + 1, INF);
        std::vector<char> used(N + 1, 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            int j1 = -1;
            double delta = INF;
            for (int j = 1; j <= N; ++j)
                if (!used[j]) {
                    const double cur = a[i0][j] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            for (int j = 0; j <= N; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    // p[j] = i  means column j is assigned to row i (1-indexed)
    for (int j = 1; j <= cols; ++j) {
        const int i = p[j];
        if (i >= 1 && i <= rows) row_assign[i - 1] = j - 1;
    }
    return row_assign;
}

void linear_assignment(const Mat& cost, double thresh,
                       std::vector<std::pair<int, int>>& matches,
                       std::vector<int>& ua, std::vector<int>& ub) {
    matches.clear();
    ua.clear();
    ub.clear();
    const int rows = cost.r;
    const int cols = cost.c;
    if (rows == 0 || cols == 0) {
        for (int i = 0; i < rows; ++i) ua.push_back(i);
        for (int j = 0; j < cols; ++j) ub.push_back(j);
        return;
    }

    const std::vector<int> ra = hungarian(cost, rows, cols);
    std::vector<char> col_matched(cols, 0);
    for (int i = 0; i < rows; ++i) {
        const int j = ra[i];
        if (j >= 0 && cost(i, j) <= thresh) {
            matches.emplace_back(i, j);
            col_matched[j] = 1;
        } else {
            ua.push_back(i);
        }
    }
    for (int j = 0; j < cols; ++j)
        if (!col_matched[j]) ub.push_back(j);
}

}  // namespace byte_track
