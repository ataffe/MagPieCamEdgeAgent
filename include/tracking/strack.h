// Copyright © 2026 Alexander Taffe

#pragma once
#include <memory>
#include <vector>
#include <chrono>

#include "linalg.h"
#include "kalman_filter.h"

/**
 * @file strack.h
 * @brief Single-target track state for the ByteTrack tracker.
 */
namespace byte_track {

/// Lifecycle state of an @ref STrack.
enum class TrackState {
    New = 0,      ///< Just created, not yet activated.
    Tracked = 1,  ///< Currently associated with detections.
    Lost = 2,     ///< Temporarily unmatched, eligible for re-association.
    Removed = 3   ///< Aged out; will be dropped.
};

/**
 * @brief A single tracked target with its Kalman state and lifecycle.
 *
 * Port of `bytetrack/byte_tracker.py` `STrack` (plus the relevant `BaseTrack`
 * bits). Each instance owns an 8-dimensional @ref KalmanFilter state and is
 * advanced via @ref predict / @ref update / @ref re_activate as detections
 * arrive.
 */
class STrack {
public:
    /**
     * @brief Construct a track from an initial detection box.
     * @param tlwh  Bounding box as `(top-left x, top-left y, width, height)`.
     * @param score Detection confidence.
     * @param label Class label.
     * @param delay_noise_ms Noise for track send delay.
     */
    STrack(const Vec4& tlwh, double score, int label, int delay_noise_ms = 0);

    /// @brief Return the next monotonically increasing track id.
    /// @return A fresh, unique track id.
    static int next_id();

    /// Reset the shared track-id counter back to zero (e.g. between sequences).
    static void reset_count();

    /**
     * @brief Run the Kalman prediction step for this track.
     * @param kf The filter providing the motion model.
     */
    void predict(const KalmanFilter& kf);

    /**
     * @brief Activate a new track, assigning it an id and initial Kalman state.
     * @param kf       The filter used to initiate the state.
     * @param frame_id Current frame number; a track activated on frame 1 is
     *                 immediately marked activated.
     */
    void activate(const KalmanFilter& kf, int frame_id);

    /**
     * @brief Re-associate a previously lost track with a new detection.
     * @param new_track The matched detection providing the new measurement.
     * @param kf        The filter used to update the state.
     * @param frame_id  Current frame number.
     * @param new_id    If true, assign a brand-new track id.
     */
    void re_activate(const STrack& new_track, const KalmanFilter& kf, int frame_id, bool new_id = false);

    /**
     * @brief Update a matched track with a new detection for this frame.
     * @param new_track The matched detection providing the new measurement.
     * @param kf        The filter used to update the state.
     * @param frame_id  Current frame number.
     */
    void update(const STrack& new_track, const KalmanFilter& kf, int frame_id);

    /// Mark this track as @ref TrackState::Lost.
    void mark_lost() { state = static_cast<int>(TrackState::Lost); }

    /// Mark this track as @ref TrackState::Removed.
    void mark_removed() { state = static_cast<int>(TrackState::Removed); }

    /// @brief Current box as `(x, y, w, h)`.
    /// @return The top-left/width-height box from the current Kalman state.
    Vec4 tlwh() const;

    /// @brief Current box as `(x1, y1, x2, y2)`.
    /// @return The top-left/bottom-right box from the current Kalman state.
    Vec4 tlbr() const;

    /// @brief Convert a `(x, y, w, h)` box to `(center x, center y, aspect, height)`.
    /// @param tlwh Box as top-left/width-height.
    /// @return The box in xyah form.
    static Vec4 tlwh_to_xyah(const Vec4& tlwh);

    /// @brief Convert a `(x1, y1, x2, y2)` box to `(x, y, w, h)`.
    /// @param tlbr Box as top-left/bottom-right.
    /// @return The box in tlwh form.
    static Vec4 tlbr_to_tlwh(const Vec4& tlbr);

    /// @brief Determines if this track should trigger a frame to be sent to the backend.
    /// @return True if the track should trigger a frame to be sent, False otherwise.
    bool is_ready_to_send() const;

    /// @brief Increments the number of times this track has triggered an image to be sent to the backend.
    /// Also resets the last send time for the track.
    void increment_send_count();

    /// @brief Returns the current required wait between uploads for this track in milliseconds.
    /// Grows exponentially with each send and is capped at @ref kMaxDelayMs.
    int get_current_delay() const;

    /// Maximum upload interval, in milliseconds, regardless of how many sends have occurred.
    static constexpr int kMaxDelayMs = 300'000;

    bool is_activated = false;                       ///< True once the track is confirmed.
    int track_id = 0;                                ///< Unique id (0 until activated).
    int state = static_cast<int>(TrackState::New);   ///< Current @ref TrackState.
    int frame_id = 0;                                ///< Frame of the last update.
    int start_frame = 0;                             ///< Frame the track was activated on.
    int tracklet_len = 0;                            ///< Consecutive frames matched.
    double score = 0.0;                              ///< Latest detection confidence.
    int label = 0;                                   ///< Class label.

private:
    Vec4 tlwh_{};                 ///< Cached initial measurement (pre-activation).
    std::vector<double> mean_;    ///< 8-vector Kalman state.
    Mat cov_;                     ///< 8x8 Kalman covariance.
    bool has_state_ = false;      ///< True once @ref activate has set up the state.
    static int count_;            ///< Shared track-id counter.


    // Time tracking for delayed notifications
    int send_count_ = 0;
    // TODO(project): Make this configurable.
    int base_wait_time_multiplier_ = 2;
    int delay_noise_ms_ = 0;
    std::chrono::time_point<std::chrono::steady_clock> last_frame_send_time_;
};

/// Shared-ownership handle to an @ref STrack.
using STrackPtr = std::shared_ptr<STrack>;

}  // namespace byte_track
