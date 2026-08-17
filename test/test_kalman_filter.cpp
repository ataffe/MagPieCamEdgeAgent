
//
// Unit tests for the 8-dim constant-velocity KalmanFilter.

#include <gtest/gtest.h>

#include <vector>

#include "tracking/kalman_filter.h"
#include "tracking/linalg.h"

using namespace byte_track;

namespace {
// Mirrors the weights baked into kalman_filter.cpp.
constexpr double kPosWeight = 1.0 / 20.0;
constexpr double kVelWeight = 1.0 / 160.0;
}  // namespace

TEST(KalmanFilter, InitiateSetsMeanAndZeroVelocity) {
    KalmanFilter kf;
    std::vector<double> mean;
    Mat cov;
    const Vec4 measurement = {20, 35, 0.5, 30};  // x, y, a, h
    kf.initiate(mean, cov, measurement);

    ASSERT_EQ(mean.size(), 8u);
    EXPECT_DOUBLE_EQ(mean[0], 20);
    EXPECT_DOUBLE_EQ(mean[1], 35);
    EXPECT_DOUBLE_EQ(mean[2], 0.5);
    EXPECT_DOUBLE_EQ(mean[3], 30);
    for (int i = 4; i < 8; ++i) EXPECT_DOUBLE_EQ(mean[i], 0.0) << "velocity " << i;
}

TEST(KalmanFilter, InitiateCovarianceDiagonalMatchesFormula) {
    KalmanFilter kf;
    std::vector<double> mean;
    Mat cov;
    const double h = 30.0;
    kf.initiate(mean, cov, {20, 35, 0.5, h});

    ASSERT_EQ(cov.r, 8);
    ASSERT_EQ(cov.c, 8);
    const double sp = 2 * kPosWeight * h;
    const double sv = 10 * kVelWeight * h;
    EXPECT_DOUBLE_EQ(cov(0, 0), sp * sp);
    EXPECT_DOUBLE_EQ(cov(1, 1), sp * sp);
    EXPECT_DOUBLE_EQ(cov(2, 2), 1e-2 * 1e-2);
    EXPECT_DOUBLE_EQ(cov(3, 3), sp * sp);
    EXPECT_DOUBLE_EQ(cov(4, 4), sv * sv);
    EXPECT_DOUBLE_EQ(cov(6, 6), 1e-5 * 1e-5);
}

TEST(KalmanFilter, PredictWithZeroVelocityLeavesMeanButGrowsCovariance) {
    KalmanFilter kf;
    std::vector<double> mean;
    Mat cov;
    kf.initiate(mean, cov, {20, 35, 0.5, 30});
    const std::vector<double> before = mean;
    const double cov00_before = cov(0, 0);

    kf.predict(mean, cov);

    // Constant-velocity model with zero velocity: position is unchanged...
    for (int i = 0; i < 8; ++i) EXPECT_DOUBLE_EQ(mean[i], before[i]) << "component " << i;
    // ...but process noise inflates the covariance.
    EXPECT_GT(cov(0, 0), cov00_before);
}

TEST(KalmanFilter, UpdateWithIdenticalMeasurementKeepsMeanAndShrinksCovariance) {
    KalmanFilter kf;
    std::vector<double> mean;
    Mat cov;
    const Vec4 measurement = {20, 35, 0.5, 30};
    kf.initiate(mean, cov, measurement);
    kf.predict(mean, cov);

    const std::vector<double> predicted = mean;
    const double cov00_before = cov(0, 0);

    // Correcting with a measurement equal to the predicted position leaves the
    // mean put (zero innovation) while reducing positional uncertainty.
    kf.update(mean, cov, measurement);

    for (int i = 0; i < 4; ++i) EXPECT_NEAR(mean[i], predicted[i], 1e-9) << "component " << i;
    EXPECT_LT(cov(0, 0), cov00_before);
}

TEST(KalmanFilter, UpdateMovesMeanTowardMeasurement) {
    KalmanFilter kf;
    std::vector<double> mean;
    Mat cov;
    kf.initiate(mean, cov, {20, 35, 0.5, 30});
    kf.predict(mean, cov);

    // Shift the measured center; the corrected estimate should move part-way
    // toward it (a convex blend, so strictly between old and measured).
    kf.update(mean, cov, {30, 35, 0.5, 30});
    EXPECT_GT(mean[0], 20.0);
    EXPECT_LT(mean[0], 30.0);
}
