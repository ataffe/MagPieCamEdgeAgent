// Copyright © 2026 Alexander Taffe

#include "../../include/tracking/byte_tracker.h"
#include "tracking/byte_tracker.h"
#include "tracking/matching.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace byte_track {

static std::vector<STrackPtr> joint_stracks(const std::vector<STrackPtr>& a,
                                            const std::vector<STrackPtr>& b) {
    std::unordered_map<int, int> exists;
    std::vector<STrackPtr> res;
    for (const auto& t : a) { exists[t->track_id] = 1; res.push_back(t); }
    for (const auto& t : b) {
        if (!exists.count(t->track_id)) { exists[t->track_id] = 1; res.push_back(t); }
    }
    return res;
}

static std::vector<STrackPtr> sub_stracks(const std::vector<STrackPtr>& a,
                                          const std::vector<STrackPtr>& b) {
    std::unordered_set<int> b_ids;
    for (const auto& t : b) b_ids.insert(t->track_id);
    std::vector<STrackPtr> res;
    std::unordered_map<int, size_t> pos;  // preserve insertion order, dict semantics
    for (const auto& t : a) {
        if (b_ids.count(t->track_id)) continue;
        auto it = pos.find(t->track_id);
        if (it == pos.end()) { pos[t->track_id] = res.size(); res.push_back(t); }
        else res[it->second] = t;
    }
    return res;
}

static void remove_duplicate_stracks(std::vector<STrackPtr>& a, std::vector<STrackPtr>& b) {
    const Mat pdist = iou_distance(a, b);
    std::unordered_set<int> dup_a, dup_b;
    for (int i = 0; i < pdist.r; ++i)
        for (int j = 0; j < pdist.c; ++j) {
            if (pdist(i, j) < 0.15) {
                const int tp = a[i]->frame_id - a[i]->start_frame;
                const int tq = b[j]->frame_id - b[j]->start_frame;
                if (tp > tq) dup_b.insert(j);
                else dup_a.insert(i);
            }
        }
    std::vector<STrackPtr> ra, rb;
    for (int i = 0; i < static_cast<int>(a.size()); ++i) if (!dup_a.count(i)) ra.push_back(a[i]);
    for (int j = 0; j < static_cast<int>(b.size()); ++j) if (!dup_b.count(j)) rb.push_back(b[j]);
    a = ra;
    b = rb;
}

BYTETracker::BYTETracker(double track_thresh, double match_thresh, int track_buffer, double frame_rate)
    : track_thresh_(track_thresh), match_thresh_(match_thresh), noise_generator_(rd_()), uniform_dist_(0, 500) {
    det_thresh_ = track_thresh + 0.1;
    max_time_lost_ = static_cast<int>(frame_rate / 30.0 * track_buffer);
}

void BYTETracker::reset() {
    tracked_stracks_.clear();
    lost_stracks_.clear();
    removed_stracks_.clear();
    frame_id_ = 0;
    STrack::reset_count();
}

std::vector<STrackPtr> BYTETracker::update(const std::vector<DetectedObject>& objects) {
    frame_id_++;
    std::vector<STrackPtr> activated, refind, lost, removed;

    // Split detections by score into high / low.
    std::vector<STrackPtr> dets, dets_second;
    for (const auto& o : objects) {
        int delay_noise = uniform_dist_(noise_generator_);
        if (o.score > track_thresh_)
            dets.push_back(std::make_shared<STrack>(STrack::tlbr_to_tlwh(o.tlbr), o.score, o.label, delay_noise));
        else if (o.score > 0.1 && o.score < track_thresh_)
            dets_second.push_back(std::make_shared<STrack>(STrack::tlbr_to_tlwh(o.tlbr), o.score, o.label, delay_noise));
    }

    // Split current tracks into confirmed / unconfirmed.
    std::vector<STrackPtr> unconfirmed, tracked;
    for (const auto& t : tracked_stracks_) {
        if (!t->is_activated) unconfirmed.push_back(t);
        else tracked.push_back(t);
    }

    // Step 2: first association with high-score detections.
    std::vector<STrackPtr> strack_pool = joint_stracks(tracked, lost_stracks_);
    for (auto& t : strack_pool) t->predict(kf_);

    std::vector<std::pair<int, int>> matches;
    std::vector<int> u_track, u_detection;
    {
        const Mat d = iou_distance(strack_pool, dets);
        linear_assignment(d, match_thresh_, matches, u_track, u_detection);
    }
    for (const auto& m : matches) {
        auto& track = strack_pool[m.first];
        auto& det = dets[m.second];
        if (track->state == static_cast<int>(TrackState::Tracked)) {
            track->update(*det, kf_, frame_id_);
            activated.push_back(track);
        } else {
            track->re_activate(*det, kf_, frame_id_, false);
            refind.push_back(track);
        }
    }

    // Step 3: second association with low-score detections.
    std::vector<STrackPtr> r_tracked;
    for (int idx : u_track)
        if (strack_pool[idx]->state == static_cast<int>(TrackState::Tracked))
            r_tracked.push_back(strack_pool[idx]);

    std::vector<std::pair<int, int>> matches2;
    std::vector<int> u_track2, u_detection2;
    {
        const Mat d = iou_distance(r_tracked, dets_second);
        linear_assignment(d, 0.5, matches2, u_track2, u_detection2);
    }
    for (const auto& m : matches2) {
        auto& track = r_tracked[m.first];
        auto& det = dets_second[m.second];
        if (track->state == static_cast<int>(TrackState::Tracked)) {
            track->update(*det, kf_, frame_id_);
            activated.push_back(track);
        } else {
            track->re_activate(*det, kf_, frame_id_, false);
            refind.push_back(track);
        }
    }
    for (int it : u_track2) {
        auto& track = r_tracked[it];
        if (track->state != static_cast<int>(TrackState::Lost)) {
            track->mark_lost();
            lost.push_back(track);
        }
    }

    // Deal with unconfirmed tracks (usually new tracks with a single beginning frame).
    std::vector<STrackPtr> det_remain;
    for (int idx : u_detection) det_remain.push_back(dets[idx]);
    std::vector<std::pair<int, int>> matches3;
    std::vector<int> u_unconfirmed, u_detection3;
    {
        const Mat d = iou_distance(unconfirmed, det_remain);
        linear_assignment(d, 0.7, matches3, u_unconfirmed, u_detection3);
    }
    for (const auto& m : matches3) {
        unconfirmed[m.first]->update(*det_remain[m.second], kf_, frame_id_);
        activated.push_back(unconfirmed[m.first]);
    }
    for (int it : u_unconfirmed) {
        auto& track = unconfirmed[it];
        track->mark_removed();
        removed.push_back(track);
    }

    // Step 4: init new tracks from unmatched high-score detections.
    for (int inew : u_detection3) {
        auto& track = det_remain[inew];
        if (track->score < det_thresh_) continue;
        track->activate(kf_, frame_id_);
        activated.push_back(track);
    }

    // Step 5: age out lost tracks.
    for (auto& track : lost_stracks_) {
        if (frame_id_ - track->frame_id > max_time_lost_) {
            track->mark_removed();
            removed.push_back(track);
        }
    }

    // Merge / bookkeeping (mirrors the Python tail exactly).
    std::vector<STrackPtr> new_tracked;
    for (auto& t : tracked_stracks_)
        if (t->state == static_cast<int>(TrackState::Tracked)) new_tracked.push_back(t);
    new_tracked = joint_stracks(new_tracked, activated);
    new_tracked = joint_stracks(new_tracked, refind);
    tracked_stracks_ = new_tracked;
    lost_stracks_ = sub_stracks(lost_stracks_, tracked_stracks_);
    for (auto& t : lost) lost_stracks_.push_back(t);
    // Only this frame's removals can still be in lost_stracks_; tracks removed in earlier
    // frames were already subtracted then. Accumulating removed_stracks_ across all frames
    // (as the original Python did) made this O(frames) per update. Use the local list and
    // don't accumulate.
    lost_stracks_ = sub_stracks(lost_stracks_, removed);
    removed_stracks_ = removed;
    remove_duplicate_stracks(tracked_stracks_, lost_stracks_);

    std::vector<STrackPtr> output;
    for (auto& t : tracked_stracks_)
        if (t->is_activated) output.push_back(t);
    return output;
}

}  // namespace byte_track