// Copyright © 2026 Alexander Taffe

#pragma once
#include <vector>
#include "linalg.h"

/**
 * @file kalman_filter.h
 * @brief Constant-velocity Kalman filter for bounding boxes in image space.
 */
namespace byte_track {

/**
 * @brief 8-dimensional constant-velocity Kalman filter for bounding boxes.
 *
 * The filter tracks an 8-element state
 * `(x, y, a, h, vx, vy, va, vh)` — center x/y, aspect ratio, height, and their
 * velocities — in image space. It is a port of `bytetrack/kalman_filter.py`,
 * implementing only the methods exercised by @ref BYTETracker:
 * @ref initiate, @ref predict, @ref project, and @ref update.
 *
 * Throughout, @c mean is an 8-vector and @c cov is an 8x8 @ref Mat.
 */
class KalmanFilter {
public:
    /// Construct the filter, building the constant-velocity motion model
    /// (dt = 1) and the measurement projection matrices.
    KalmanFilter();

    /**
     * @brief Create a track state from an unassociated measurement.
     * @param[out] mean        Initialized 8-vector state; the velocity
     *                         components are set to zero.
     * @param[out] cov         Initialized 8x8 covariance.
     * @param[in]  measurement Measurement `(x, y, a, h)` — center, aspect
     *                         ratio, height.
     */
    void initiate(std::vector<double>& mean, Mat& cov, const Vec4& measurement) const;

    /**
     * @brief Run the prediction step, advancing the state one frame.
     * @param[in,out] mean 8-vector state, propagated by the motion model.
     * @param[in,out] cov  8x8 covariance, inflated by process noise.
     */
    void predict(std::vector<double>& mean, Mat& cov) const;

    /**
     * @brief Run the correction step, folding a new measurement into the state.
     * @param[in,out] mean        8-vector state, corrected toward the
     *                            measurement.
     * @param[in,out] cov         8x8 covariance, reduced by the correction.
     * @param[in]     measurement Measurement `(x, y, a, h)`.
     */
    void update(std::vector<double>& mean, Mat& cov, const Vec4& measurement) const;

private:
    /**
     * @brief Project a state distribution into measurement space.
     * @param[out] pmean Projected 4-vector mean `(x, y, a, h)`.
     * @param[out] pcov  Projected 4x4 covariance (with measurement noise added).
     * @param[in]  mean  8-vector state.
     * @param[in]  cov   8x8 covariance.
     */
    void project(Vec4& pmean, Mat& pcov, const std::vector<double>& mean, const Mat& cov) const;

    Mat motion_mat_;                         ///< 8x8 state-transition matrix.
    Mat update_mat_;                         ///< 4x8 measurement matrix.
    double std_weight_position_ = 1.0 / 20.0;  ///< Position uncertainty weight (× height).
    double std_weight_velocity_ = 1.0 / 160.0; ///< Velocity uncertainty weight (× height).
};

}  // namespace byte_track
