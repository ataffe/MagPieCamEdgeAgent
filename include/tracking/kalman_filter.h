// Copyright © 2026 Alexander Taffe

#pragma once
#include <vector>
#include "linalg.h"

namespace byte_track {

// 8-dim constant-velocity Kalman filter for bounding boxes in image space.
// State: x, y, a, h, vx, vy, va, vh  (center x/y, aspect ratio, height + velocities)
// Port of bytetrack/kalman_filter.py (only the methods exercised by BYTETracker:
// initiate, predict, project, update). mean is an 8-vector, cov an 8x8 Mat.
class KalmanFilter {
public:
    KalmanFilter();

    void initiate(std::vector<double>& mean, Mat& cov, const Vec4& measurement) const;  // measurement = (x,y,a,h)
    void predict(std::vector<double>& mean, Mat& cov) const;
    void update(std::vector<double>& mean, Mat& cov, const Vec4& measurement) const;

private:
    void project(Vec4& pmean, Mat& pcov, const std::vector<double>& mean, const Mat& cov) const;

    Mat motion_mat_;  // 8x8
    Mat update_mat_;  // 4x8
    double std_weight_position_ = 1.0 / 20.0;
    double std_weight_velocity_ = 1.0 / 160.0;
};

}  // namespace byte_track
