#pragma once
#include <vector>
#include "linalg.h"
#include "kalman_filter.h"
#include "strack.h"

namespace byte_track {

// A single detection fed into the tracker. Box is tlbr (x1, y1, x2, y2),
// matching what bytetrack/byte_tracker.py receives (boxes are tlbr there too).
struct Object {
    Vec4 tlbr;
    double score;
    int label;
};

// Port of bytetrack/byte_tracker.py BYTETracker.
class BYTETracker {
public:
    BYTETracker(double track_thresh = 0.5, double match_thresh = 0.8,
                int track_buffer = 30, double frame_rate = 30.0);

    std::vector<STrackPtr> update(const std::vector<Object>& objects);
    void reset();

private:
    std::vector<STrackPtr> tracked_stracks_;
    std::vector<STrackPtr> lost_stracks_;
    std::vector<STrackPtr> removed_stracks_;

    int frame_id_ = 0;
    double track_thresh_;
    double match_thresh_;
    double det_thresh_;
    int max_time_lost_;
    KalmanFilter kf_;
};

}  // namespace byte_track
