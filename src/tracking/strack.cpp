// Copyright © 2026 Alexander Taffe

#include "tracking/strack.h"

namespace byte_track {

int STrack::count_ = 0;
int STrack::next_id() { return ++count_; }
void STrack::reset_count() { count_ = 0; }

STrack::STrack(const Vec4& tlwh, double score, int label)
    : score(score), label(label), tlwh_(tlwh) {}

Vec4 STrack::tlbr_to_tlwh(const Vec4& tlbr) {
    return {tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]};
}

Vec4 STrack::tlwh_to_xyah(const Vec4& t) {
    return {t[0] + t[2] / 2.0, t[1] + t[3] / 2.0, t[2] / t[3], t[3]};
}

Vec4 STrack::tlwh() const {
    if (!has_state_) return tlwh_;
    const double cx = mean_[0], cy = mean_[1], a = mean_[2], h = mean_[3];
    const double w = a * h;
    return {cx - w / 2.0, cy - h / 2.0, w, h};
}

Vec4 STrack::tlbr() const {
    const Vec4 t = tlwh();
    return {t[0], t[1], t[0] + t[2], t[1] + t[3]};
}

void STrack::predict(const KalmanFilter& kf) {
    if (state != static_cast<int>(TrackState::Tracked)) mean_[7] = 0.0;
    kf.predict(mean_, cov_);
}

void STrack::activate(const KalmanFilter& kf, int fid) {
    track_id = next_id();
    kf.initiate(mean_, cov_, tlwh_to_xyah(tlwh_));
    has_state_ = true;
    tracklet_len = 0;
    state = static_cast<int>(TrackState::Tracked);
    is_activated = (fid == 1);
    frame_id = fid;
    start_frame = fid;
}

void STrack::re_activate(const STrack& nt, const KalmanFilter& kf, int fid, bool new_id) {
    kf.update(mean_, cov_, tlwh_to_xyah(nt.tlwh()));
    tracklet_len = 0;
    state = static_cast<int>(TrackState::Tracked);
    is_activated = true;
    frame_id = fid;
    if (new_id) track_id = next_id();
    score = nt.score;
}

void STrack::update(const STrack& nt, const KalmanFilter& kf, int fid) {
    frame_id = fid;
    tracklet_len += 1;
    kf.update(mean_, cov_, tlwh_to_xyah(nt.tlwh()));
    state = static_cast<int>(TrackState::Tracked);
    is_activated = true;
    score = nt.score;
}

}  // namespace byte_track