// Copyright © 2026 Alexander Taffe

#pragma once
#include <vector>
#include <utility>
#include "linalg.h"
#include "strack.h"

/**
 * @file matching.h
 * @brief IoU computation and detection-to-track assignment helpers.
 */
namespace byte_track {

/**
 * @brief Intersection-over-union of two boxes.
 *
 * Replicates `cython_bbox.bbox_overlaps`, including its `+1` width/height
 * convention (so a 0..10 box is treated as 11 pixels wide).
 *
 * @param a_tlbr First box as `(x1, y1, x2, y2)`.
 * @param b_tlbr Second box as `(x1, y1, x2, y2)`.
 * @return IoU in `[0, 1]`; 0 when the boxes do not overlap.
 */
double iou(const Vec4& a_tlbr, const Vec4& b_tlbr);

/**
 * @brief Pairwise IoU-distance cost matrix between two track lists.
 *
 * entry (i, j) is `1 - iou(a[i], b[j])`.
 *
 * @param a Row tracks.
 * @param b Column tracks.
 * @return A (@c a.size() x @c b.size()) cost matrix.
 */
Mat iou_distance(const std::vector<STrackPtr>& a, const std::vector<STrackPtr>& b);

/**
 * @brief Optimal assignment between rows and columns of a cost matrix.
 *
 * Computes the optimal assignment, then drops any matched pair whose cost exceeds @p thresh (both
 * its row and column become unmatched).
 *
 * @param[in]  cost        Cost matrix (rows vs. columns).
 * @param[in]  thresh      Maximum allowed cost for a kept match.
 * @param[out] matches     Accepted (row, column) index pairs.
 * @param[out] unmatched_a Row indices left unmatched.
 * @param[out] unmatched_b Column indices left unmatched.
 */
void linear_assignment(const Mat& cost, double thresh,
                       std::vector<std::pair<int, int>>& matches,
                       std::vector<int>& unmatched_a,
                       std::vector<int>& unmatched_b);

}  // namespace byte_track
