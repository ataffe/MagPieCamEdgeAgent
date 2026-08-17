
//
// End-to-end tests for BYTETracker.update over short detection sequences.

#include <gtest/gtest.h>

#include <vector>

#include "tracking/byte_tracker.h"
#include "tracking/strack.h"

using namespace byte_track;

namespace {
DetectedObject det(const Vec4& tlbr, double score, int label = 0) {
    return DetectedObject{tlbr, score, label};
}
}  // namespace

// BYTETracker leans on STrack's static id counter; reset it for isolation.
class ByteTrackerTest : public ::testing::Test {
protected:
    void SetUp() override { STrack::reset_count(); }
};

TEST_F(ByteTrackerTest, HighScoreDetectionStartsTrackOnFirstFrame) {
    BYTETracker tracker;
    auto out = tracker.update({det({100, 100, 150, 200}, 0.9)});

    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0]->track_id, 1);
    EXPECT_TRUE(out[0]->is_activated);
}

TEST_F(ByteTrackerTest, LowScoreDetectionDoesNotStartTrack) {
    BYTETracker tracker;  // track_thresh 0.5, det_thresh 0.6
    // A 0.3 detection is below track_thresh, so it never seeds a new track.
    auto out = tracker.update({det({100, 100, 150, 200}, 0.3)});
    EXPECT_TRUE(out.empty());
}

TEST_F(ByteTrackerTest, StableTrackKeepsSameIdAcrossFrames) {
    BYTETracker tracker;
    const Vec4 box = {100, 100, 150, 200};

    auto out = tracker.update({det(box, 0.9)});
    ASSERT_EQ(out.size(), 1u);
    const int id = out[0]->track_id;

    // The same object, frame after frame, must keep its id and not spawn duplicates.
    for (int f = 0; f < 5; ++f) {
        out = tracker.update({det(box, 0.9)});
        ASSERT_EQ(out.size(), 1u) << "frame " << f;
        EXPECT_EQ(out[0]->track_id, id) << "frame " << f;
    }
}

TEST_F(ByteTrackerTest, SlowlyMovingObjectIsTrackedAsOne) {
    BYTETracker tracker;
    auto out = tracker.update({det({100, 100, 150, 200}, 0.9)});
    ASSERT_EQ(out.size(), 1u);
    const int id = out[0]->track_id;

    // Shift the box a few pixels each frame; IoU stays high enough to associate.
    for (int f = 1; f <= 5; ++f) {
        const double x = 100 + f * 5;
        out = tracker.update({det({x, 100, x + 50, 200}, 0.9)});
        ASSERT_EQ(out.size(), 1u) << "frame " << f;
        EXPECT_EQ(out[0]->track_id, id) << "frame " << f;
    }
}

TEST_F(ByteTrackerTest, TwoSeparateObjectsGetDistinctIds) {
    BYTETracker tracker;
    auto out = tracker.update({
        det({100, 100, 150, 200}, 0.9),
        det({400, 100, 450, 200}, 0.9),
    });

    ASSERT_EQ(out.size(), 2u);
    EXPECT_NE(out[0]->track_id, out[1]->track_id);
}

TEST_F(ByteTrackerTest, TrackDropsFromOutputWhenObjectDisappears) {
    BYTETracker tracker;
    ASSERT_EQ(tracker.update({det({100, 100, 150, 200}, 0.9)}).size(), 1u);

    // With no detections the track is marked lost and leaves the output set.
    EXPECT_TRUE(tracker.update({}).empty());
}

TEST_F(ByteTrackerTest, ResetClearsStateAndIdCounter) {
    BYTETracker tracker;
    auto out = tracker.update({det({100, 100, 150, 200}, 0.9)});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0]->track_id, 1);

    tracker.reset();

    // After reset the id counter restarts from 1 and no tracks linger.
    out = tracker.update({det({300, 300, 350, 400}, 0.9)});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0]->track_id, 1);
}
