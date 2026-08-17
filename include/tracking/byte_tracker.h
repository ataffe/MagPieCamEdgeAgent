

#pragma once
#include <vector>
#include <random>

#include "linalg.h"
#include "kalman_filter.h"
#include "strack.h"

/**
 * @file byte_tracker.h
 * @brief Multi-object tracker (ByteTrack) driving detections into stable tracks.
 */
namespace byte_track {

    /**
     * @brief A single detection fed into the tracker for one frame.
     *
     * The box is stored in tlbr form `(x1, y1, x2, y2)`, matching what
     * `bytetrack/byte_tracker.py` receives (boxes are tlbr there too).
     */
    struct DetectedObject {
        Vec4 tlbr;     ///< Bounding box `(x1, y1, x2, y2)`.
        double score;  ///< Detection confidence.
        int label;     ///< Class label.
    };

    /// @brief Snapshot of a confirmed track ready to be drawn or passed to the upload pipeline.
    struct Drawable {
        Vec4 tlwh;  ///< Bounding box `(x, y, width, height)`.
        int id;     ///< Track id.
        int label;  ///< Class label.
    };

/**
 * @brief ByteTrack multi-object tracker.
 *
 * Port of the `BYTETracker` class from `bytetrack/byte_tracker.py`. Maintains a
 * pool of active, lost, and removed tracks across frames and associates new
 * detections to them in two stages (high-score then low-score), using IoU
 * matching backed by a per-track @ref KalmanFilter.
 */
class BYTETracker {
public:
    /**
     * @brief Construct a tracker.
     * @param track_thresh Score above which a detection is a "high" detection;
     *                     the new-track threshold is this plus 0.1.
     * @param match_thresh Maximum IoU distance for the first association stage.
     * @param track_buffer Number of frames a lost track is retained before
     *                     removal (scaled by @p frame_rate).
     * @param frame_rate   Source frame rate, used to scale @p track_buffer.
     */
    BYTETracker(double track_thresh = 0.5, double match_thresh = 0.8,
                int track_buffer = 30, double frame_rate = 30.0);

    /**
     * @brief Advance the tracker by one frame.
     * @param objects Detections for the current frame.
     * @return The currently activated tracks (the live tracking output).
     */
    std::vector<STrackPtr> update(const std::vector<DetectedObject>& objects);

    /// Clear all track state and reset the frame counter and track-id counter.
    void reset();

private:
    std::vector<STrackPtr> tracked_stracks_;  ///< Currently tracked targets.
    std::vector<STrackPtr> lost_stracks_;     ///< Recently lost, re-associable targets.
    std::vector<STrackPtr> removed_stracks_;  ///< Targets removed this frame.

    int frame_id_ = 0;        ///< Number of frames processed so far.
    double track_thresh_;     ///< High-detection score threshold.
    double match_thresh_;     ///< First-stage IoU-distance match threshold.
    double det_thresh_;       ///< Score required to start a new track.
    int max_time_lost_;       ///< Max frames a track may stay lost before removal.
    KalmanFilter kf_;         ///< Shared motion model for all tracks.

    // Noise generator for exponential backoff with noise
    std::random_device rd_;
    std::mt19937 noise_generator_;
    std::uniform_int_distribution<int> uniform_dist_;
};

}  // namespace byte_track
