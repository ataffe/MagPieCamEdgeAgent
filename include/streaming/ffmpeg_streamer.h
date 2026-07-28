// Copyright © 2026 Alexander Taffe

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/types.h>

namespace byte_track {

// Streams raw video frames to an RTSP server (MediaMTX) by piping them into a
// child ffmpeg process.
//
// This is the in-process equivalent of the shell pipeline
//
//   rpicam-vid -t 0 -n --codec h264 --inline --width 1280 --height 720 \
//       --framerate 30 --bitrate 2000000 --intra 30 -o - \
//     | ffmpeg -f h264 -framerate 30 -i - -c:v copy \
//         -f rtsp -rtsp_transport tcp rtsp://10.0.0.126:8554/cam
//
// libcamera only allows one owner of the camera and the tracker already holds
// it, so rpicam-vid cannot run alongside us. Instead we feed ffmpeg the frames
// the tracker is already receiving and let it do the H.264 encode rpicam-vid
// would have done — by default on the Pi's hardware encoder (h264_v4l2m2m) with
// the same bitrate/GOP. So `-c:v copy` becomes a real encode and the input is
// rawvideo instead of h264; everything downstream of ffmpeg is unchanged.
//
// write_frame() never blocks the caller: it copies the frame into a single-slot
// buffer that a writer thread drains into ffmpeg's stdin. If the encoder falls
// behind, the pending frame is overwritten (dropped) rather than stalling the
// camera callback — on a live stream the newest frame is the only one worth
// having. If ffmpeg dies (e.g. MediaMTX restarts) the writer thread respawns it.
class FfmpegStreamer {
public:
    struct Config {
        std::string rtsp_url = "rtsp://10.0.0.126:8554/cam";
        int width = 1280;
        int height = 720;
        int framerate = 30;
        int bitrate = 2000000;   // bits/sec, matches rpicam-vid --bitrate
        int gop = 30;            // keyframe interval, matches rpicam-vid --intra
        // Pi 4 has a hardware H.264 encoder behind bcm2835-codec; libx264 is the
        // portable (CPU) fallback.
        std::string encoder = "h264_v4l2m2m";
    };

    explicit FfmpegStreamer(Config config);
    ~FfmpegStreamer();

    FfmpegStreamer(const FfmpegStreamer &) = delete;
    FfmpegStreamer &operator=(const FfmpegStreamer &) = delete;

    // Spawns ffmpeg and starts the writer thread. Returns false if the process
    // could not be spawned at all (e.g. ffmpeg not installed); later failures
    // are handled by the writer thread's restart logic.
    bool start();

    // Closes ffmpeg's stdin and waits for it to flush and exit. Idempotent.
    void stop();

    // Hands over one tightly packed I420 (yuv420p) frame of frame_bytes() bytes.
    // Returns false if the frame replaced an unwritten one (i.e. it was dropped)
    // or the streamer isn't running.
    bool write_frame(const uint8_t *data, size_t len);

    // Bytes of a single packed I420 frame: the exact size write_frame() expects.
    size_t frame_bytes() const;

    uint64_t dropped_frames() const { return dropped_frames_.load(); }
    const Config &config() const { return config_; }

private:
    // Builds the ffmpeg command line for config_ (also used to log it).
    std::vector<std::string> build_args() const;

    // fork/exec ffmpeg with a pipe wired to its stdin. Sets pipe_fd_/child_pid_.
    bool spawn();
    // Closes the pipe (ffmpeg sees EOF) and reaps the child.
    void reap();
    // write(2) loop; returns false once the pipe is broken.
    bool write_all(const uint8_t *data, size_t len);
    void writer_loop();

    Config config_;

    int pipe_fd_ = -1;
    pid_t child_pid_ = -1;
    std::chrono::steady_clock::time_point next_spawn_{};

    std::thread writer_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> dropped_frames_{0};

    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<uint8_t> pending_;
    bool has_frame_ = false;
};

}  // namespace byte_track
