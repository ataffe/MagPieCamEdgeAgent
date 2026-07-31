// Copyright © 2026 Alexander Taffe

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
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
//         -f rtsp -rtsp_transport tcp rtsp://10.0.0.126:8554/<public_camera_id>
//
//
// write_frame() never blocks the caller: it copies the frame into a single-slot
// buffer that a writer thread drains into ffmpeg's stdin. If the encoder falls
// behind, the pending frame is overwritten (dropped) rather than stalling the
// camera callback — on a live stream the newest frame is the only one worth
// having. If ffmpeg dies (e.g. MediaMTX restarts) the writer thread respawns it.
//
// If MediaMTX requires auth, set_jwt_provider() supplies a fresh JWT before
// each (re)spawn, appended to rtsp_url as MediaMTX's documented `?token=`
// query parameter -- the same JWT BackendClient::get_jwt_token() hands out
// for the backend's own APIs, since MediaMTX validates it the same way.
class FfmpegStreamer {
public:
    struct Config {
        // Host and port of the MediaMTX server. The per-camera path segment
        // (the public_camera_id) is appended by set_public_camera_id(), not
        // baked into this default, since one MediaMTX instance now serves
        // multiple cameras rather than a single fixed "/cam" path.
        std::string rtsp_url = "rtsp://10.0.0.126:8554";
        int width = 1280;
        int height = 720;
        int framerate = 30;
        int bitrate = 2000000;   // bits/sec, matches rpicam-vid --bitrate
        // Keyframe interval (rpicam-vid --intra). Unset means one keyframe per
        // second, i.e. it tracks `framerate` however that ends up being set.
        std::optional<int> gop;
        // Pi 4 has a hardware H.264 encoder behind bcm2835-codec; libx264 is the
        // portable (CPU) fallback.
        std::string encoder = "h264_v4l2m2m/ex";

        // Reads the "streaming" section of the client config JSON
        // Every key is optional and falls back to the default above. "gop" is
        // the one key not in the committed config; left out, it tracks
        // `framerate` (one keyframe per second, what --intra 30 gave at 30fps).
        //
        // Throws std::runtime_error if the file is missing or malformed, if
        // there is no "streaming" section, or if a value is the wrong type or
        // out of range — callers decide whether that is fatal.
        static Config from_file(const std::string &path);
    };

    explicit FfmpegStreamer(Config config);
    ~FfmpegStreamer();

    FfmpegStreamer(const FfmpegStreamer &) = delete;
    FfmpegStreamer &operator=(const FfmpegStreamer &) = delete;

    // Spawns ffmpeg and starts the writer thread. Returns false if the process
    // could not be spawned at all (e.g. ffmpeg not installed); later failures
    // are handled by the writer thread's restart logic. Idempotent, and may be
    // called again after stop() to bring the stream back -- StreamCommandPoller
    // cycles the two as viewers come and go. Calling start()/stop() from more
    // than one thread at a time is not supported; the poller is the only caller.
    bool start();

    // Closes ffmpeg's stdin and waits for it to flush and exit. Idempotent.
    // Safe to call while another thread is in write_frame(): that call simply
    // sees the streamer stopped and drops its frame.
    void stop();

    // Hands over one tightly packed I420 (yuv420p) frame of frame_bytes() bytes.
    // Returns false if the frame replaced an unwritten one (i.e. it was dropped)
    // or the streamer isn't running.
    bool write_frame(const uint8_t *data, size_t len);

    // Bytes of a single packed I420 frame: the exact size write_frame() expects.
    size_t frame_bytes() const;

    // Supplies a fresh JWT before every (re)spawn of ffmpeg. Call this before
    // start(); the provider is invoked from the writer thread, so it must be
    // safe to call on its own (BackendClient::get_jwt_token() is). If never
    // set, or if it returns std::nullopt, ffmpeg publishes to rtsp_url
    // unauthenticated.
    void set_jwt_provider(std::function<std::optional<std::string>()> provider);

    // Sets the per-camera path segment appended to rtsp_url, e.g.
    // set_public_camera_id("019fabb9-...") turns "rtsp://host:8554" into
    // "rtsp://host:8554/019fabb9-...". Call before start(). If never set,
    // ffmpeg publishes directly to rtsp_url unchanged (e.g. for tests or a
    // MediaMTX path already fully specified in rtsp_url).
    void set_public_camera_id(std::string camera_id);

    // rtsp_url, with the camera id path segment and a fresh `?token=` from
    // the JWT provider (if set) -- the same URL the next (re)spawn would use.
    // Exposed so the URL-building logic is testable without spawning ffmpeg.
    std::string resolved_rtsp_url() const;

    // True between a successful start() and a stop(). Callers use this to skip
    // the per-frame packing work while the stream is idle, since write_frame()
    // would only drop those frames anyway.
    bool is_running() const { return running_.load(); }

    uint64_t dropped_frames() const { return dropped_frames_.load(); }
    const Config &config() const { return config_; }

private:
    // Builds the ffmpeg command line for config_ (also used to log it).
    std::vector<std::string> build_args() const;

    // rtsp_url with the public_camera_id path segment appended, before the
    // `?token=` query is added. Factored out so both resolved_rtsp_url() and
    // spawn()'s log line build the same base URL.
    std::string rtsp_url_with_camera_id() const;

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

    std::function<std::optional<std::string>()> jwt_provider_;
    std::string public_camera_id_;

    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<uint8_t> pending_;
    bool has_frame_ = false;
};

}  // namespace byte_track
