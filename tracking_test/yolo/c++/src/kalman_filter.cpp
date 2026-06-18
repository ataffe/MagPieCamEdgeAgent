#include "kalman_filter.h"

namespace byte_track {

KalmanFilter::KalmanFilter() : motion_mat_(Mat::eye(8)), update_mat_(4, 8) {
    for (int i = 0; i < 4; ++i) motion_mat_(i, 4 + i) = 1.0;  // dt = 1
    for (int i = 0; i < 4; ++i) update_mat_(i, i) = 1.0;
}

void KalmanFilter::initiate(std::vector<double>& mean, Mat& cov, const Vec4& m) const {
    mean.assign(8, 0.0);
    for (int i = 0; i < 4; ++i) mean[i] = m[i];
    const double h = m[3];
    const double s[8] = {2 * std_weight_position_ * h, 2 * std_weight_position_ * h, 1e-2, 2 * std_weight_position_ * h,
                         10 * std_weight_velocity_ * h, 10 * std_weight_velocity_ * h, 1e-5, 10 * std_weight_velocity_ * h};
    cov = Mat(8, 8);
    for (int i = 0; i < 8; ++i) cov(i, i) = s[i] * s[i];
}

void KalmanFilter::predict(std::vector<double>& mean, Mat& cov) const {
    const double h = mean[3];
    const double s[8] = {std_weight_position_ * h, std_weight_position_ * h, 1e-2, std_weight_position_ * h,
                         std_weight_velocity_ * h, std_weight_velocity_ * h, 1e-5, std_weight_velocity_ * h};
    Mat motion_cov(8, 8);
    for (int i = 0; i < 8; ++i) motion_cov(i, i) = s[i] * s[i];

    // mean = motion_mat * mean
    std::vector<double> nm(8, 0.0);
    for (int i = 0; i < 8; ++i) {
        double acc = 0.0;
        for (int j = 0; j < 8; ++j) acc += motion_mat_(i, j) * mean[j];
        nm[i] = acc;
    }
    mean = nm;
    cov = matmul(matmul(motion_mat_, cov), transpose(motion_mat_)) + motion_cov;
}

void KalmanFilter::project(Vec4& pmean, Mat& pcov, const std::vector<double>& mean, const Mat& cov) const {
    const double h = mean[3];
    const double s[4] = {std_weight_position_ * h, std_weight_position_ * h, 1e-1, std_weight_position_ * h};
    for (int i = 0; i < 4; ++i) pmean[i] = mean[i];  // update_mat selects the first 4 components
    pcov = matmul(matmul(update_mat_, cov), transpose(update_mat_));  // == top-left 4x4 of cov
    for (int i = 0; i < 4; ++i) pcov(i, i) += s[i] * s[i];
}

void KalmanFilter::update(std::vector<double>& mean, Mat& cov, const Vec4& meas) const {
    Vec4 pmean;
    Mat pcov;
    project(pmean, pcov, mean, cov);

    // Python: b = (cov @ update_mat.T).T (4x8);  kalman_gain = solve(pcov, b).T (8x4)
    const Mat B = matmul(cov, transpose(update_mat_));  // 8x4
    const Mat b = transpose(B);                         // 4x8
    const Mat X = solve(pcov, b);                       // 4x8
    const Mat K = transpose(X);                         // 8x4 (kalman gain)

    Vec4 innov;
    for (int i = 0; i < 4; ++i) innov[i] = meas[i] - pmean[i];

    // mean += innovation * K^T  =>  mean[k] += sum_i innov[i] * K(k,i)
    for (int k = 0; k < 8; ++k) {
        double acc = 0.0;
        for (int i = 0; i < 4; ++i) acc += innov[i] * K(k, i);
        mean[k] += acc;
    }
    // cov -= K * pcov * K^T
    cov = cov - matmul(matmul(K, pcov), transpose(K));
}

}  // namespace byte_track
