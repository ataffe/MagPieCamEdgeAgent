// Copyright © 2026 Alexander Taffe
//
// Unit tests for STrack: box conversions, the shared id counter, and the
// activation lifecycle.

#include <gtest/gtest.h>

#include "tracking/kalman_filter.h"
#include "tracking/strack.h"

using namespace byte_track;

namespace {
constexpr double kEps = 1e-9;
void expect_vec4_near(const Vec4& got, const Vec4& want, double eps = kEps) {
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(got[i], want[i], eps) << "component " << i;
}
}  // namespace

// STrack keeps a static id counter, so reset it before every test for isolation.
class STrackTest : public ::testing::Test {
protected:
    void SetUp() override { STrack::reset_count(); }
};

// --- static conversions --------------------------------------------------

TEST_F(STrackTest, TlbrToTlwh) {
    expect_vec4_near(STrack::tlbr_to_tlwh({10, 20, 30, 50}), {10, 20, 20, 30});
}

TEST_F(STrackTest, TlwhToXyah) {
    // center = (10+10, 20+15), aspect = w/h = 20/30, height = 30.
    expect_vec4_near(STrack::tlwh_to_xyah({10, 20, 20, 30}), {20, 35, 20.0 / 30.0, 30});
}

// --- id counter ----------------------------------------------------------

TEST_F(STrackTest, NextIdIncrementsFromOne) {
    EXPECT_EQ(STrack::next_id(), 1);
    EXPECT_EQ(STrack::next_id(), 2);
    EXPECT_EQ(STrack::next_id(), 3);
}

TEST_F(STrackTest, ResetCountRestartsIds) {
    STrack::next_id();
    STrack::next_id();
    STrack::reset_count();
    EXPECT_EQ(STrack::next_id(), 1);
}

// --- construction --------------------------------------------------------

TEST_F(STrackTest, ConstructorStoresMeasurementAndDefaults) {
    STrack t({10, 20, 20, 30}, 0.9, 2);
    EXPECT_DOUBLE_EQ(t.score, 0.9);
    EXPECT_EQ(t.label, 2);
    EXPECT_FALSE(t.is_activated);
    EXPECT_EQ(t.track_id, 0);
    EXPECT_EQ(t.state, static_cast<int>(TrackState::New));
    // Before activation tlwh() returns the cached measurement unchanged.
    expect_vec4_near(t.tlwh(), {10, 20, 20, 30});
}

// --- activation lifecycle ------------------------------------------------

TEST_F(STrackTest, ActivateOnFirstFrameMarksActivated) {
    KalmanFilter kf;
    STrack t({10, 20, 20, 30}, 0.9, 0);
    t.activate(kf, /*fid=*/1);

    EXPECT_EQ(t.track_id, 1);
    EXPECT_EQ(t.state, static_cast<int>(TrackState::Tracked));
    EXPECT_TRUE(t.is_activated);  // is_activated == (frame_id == 1)
    EXPECT_EQ(t.frame_id, 1);
    EXPECT_EQ(t.start_frame, 1);
    EXPECT_EQ(t.tracklet_len, 0);
    // Round-tripping tlwh -> xyah -> Kalman state -> tlwh is exact here.
    expect_vec4_near(t.tlwh(), {10, 20, 20, 30}, 1e-6);
}

TEST_F(STrackTest, ActivateOnLaterFrameIsNotYetActivated) {
    KalmanFilter kf;
    STrack t({10, 20, 20, 30}, 0.9, 0);
    t.activate(kf, /*fid=*/5);

    EXPECT_EQ(t.state, static_cast<int>(TrackState::Tracked));
    EXPECT_FALSE(t.is_activated);  // only frame 1 auto-activates
    EXPECT_EQ(t.start_frame, 5);
}

TEST_F(STrackTest, MarkLostAndRemovedChangeState) {
    STrack t({10, 20, 20, 30}, 0.9, 0);
    t.mark_lost();
    EXPECT_EQ(t.state, static_cast<int>(TrackState::Lost));
    t.mark_removed();
    EXPECT_EQ(t.state, static_cast<int>(TrackState::Removed));
}

TEST_F(STrackTest, UpdateAdvancesFrameAndTrackletLength) {
    KalmanFilter kf;
    STrack t({10, 20, 20, 30}, 0.9, 0);
    t.activate(kf, 1);

    STrack det({10, 20, 20, 30}, 0.8, 0);
    t.update(det, kf, /*fid=*/2);

    EXPECT_EQ(t.frame_id, 2);
    EXPECT_EQ(t.tracklet_len, 1);
    EXPECT_TRUE(t.is_activated);
    EXPECT_DOUBLE_EQ(t.score, 0.8);  // score copied from the new detection
}

TEST_F(STrackTest, ReactivateCanAssignNewId) {
    KalmanFilter kf;
    STrack t({10, 20, 20, 30}, 0.9, 0);
    t.activate(kf, 1);  // track_id == 1
    t.mark_lost();

    STrack det({10, 20, 20, 30}, 0.7, 0);
    t.re_activate(det, kf, /*fid=*/4, /*new_id=*/true);

    EXPECT_EQ(t.state, static_cast<int>(TrackState::Tracked));
    EXPECT_TRUE(t.is_activated);
    EXPECT_EQ(t.frame_id, 4);
    EXPECT_EQ(t.track_id, 2);  // new id pulled from the counter
}
