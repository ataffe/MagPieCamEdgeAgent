// Copyright © 2026 Alexander Taffe

#pragma once
#include <memory>
#include <vector>
#include "linalg.h"
#include "kalman_filter.h"

namespace byte_track {

enum class TrackState { New = 0, Tracked = 1, Lost = 2, Removed = 3 };

// Port of bytetrack/byte_tracker.py STrack (+ relevant BaseTrack bits).
class STrack {
public:
    // tlwh = (top-left x, top-left y, width, height)
    STrack(const Vec4& tlwh, double score, int label);

    static int next_id();
    static void reset_count();

    void predict(const KalmanFilter& kf);
    void activate(const KalmanFilter& kf, int frame_id);
    void re_activate(const STrack& new_track, const KalmanFilter& kf, int frame_id, bool new_id = false);
    void update(const STrack& new_track, const KalmanFilter& kf, int frame_id);

    void mark_lost() { state = static_cast<int>(TrackState::Lost); }
    void mark_removed() { state = static_cast<int>(TrackState::Removed); }

    Vec4 tlwh() const;  // (x, y, w, h)
    Vec4 tlbr() const;  // (x1, y1, x2, y2)

    static Vec4 tlwh_to_xyah(const Vec4& tlwh);
    static Vec4 tlbr_to_tlwh(const Vec4& tlbr);

    bool is_activated = false;
    int track_id = 0;
    int state = static_cast<int>(TrackState::New);
    int frame_id = 0;
    int start_frame = 0;
    int tracklet_len = 0;
    double score = 0.0;
    int label = 0;

private:
    Vec4 tlwh_{};                 // cached initial measurement (pre-activation)
    std::vector<double> mean_;    // 8
    Mat cov_;                     // 8x8
    bool has_state_ = false;
    static int count_;
};

using STrackPtr = std::shared_ptr<STrack>;

}  // namespace byte_track
