// Copyright © 2026 Alexander Taffe

#pragma once
#include <vector>
#include <utility>
#include "linalg.h"
#include "strack.h"

namespace byte_track {

// IoU between two tlbr boxes, replicating cython_bbox bbox_overlaps (the +1 convention).
double iou(const Vec4& a_tlbr, const Vec4& b_tlbr);

// Cost matrix (rows = a, cols = b) of 1 - IoU. Mirrors matching.iou_distance.
Mat iou_distance(const std::vector<STrackPtr>& a, const std::vector<STrackPtr>& b);

// Mirrors matching.linear_assignment(cost, thresh): optimal assignment, then any
// matched pair with cost > thresh is dropped (both row & col become unmatched).
void linear_assignment(const Mat& cost, double thresh,
                       std::vector<std::pair<int, int>>& matches,
                       std::vector<int>& unmatched_a,
                       std::vector<int>& unmatched_b);

}  // namespace byte_track
