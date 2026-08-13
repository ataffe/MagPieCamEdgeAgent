// Copyright © 2026 Alexander Taffe

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <sys/types.h>

namespace byte_track {

// Publishes an already-encoded H.264 stream to an RTSP server (MediaMTX) by
// piping it into a child ffmpeg process that only muxes -- it does not encode.
//
// This is the in-process equivalent of the shell pipeline
//
//   rpicam-vid -t 0 -n --codec h264 --inline --width 1280 --height 720 \
//       --framerate 30 --bitrate 2000000 --intra 30 -o - \
//     | ffmpeg -f h264 -framerate 30 -i - -c:v copy \
//         -f rtsp -rtsp_transport tcp rtsp://10.0.0.126:8554/<public_camera_id>
//
// with H264Tee playing the part of rpicam-vid. Encoding happens once, upstream,
// and is shared with the event clip recorder; this class is the `-c:v copy`
// half.
//
// write_access_unit() never blocks the caller: it copies the access unit onto a
// queue that a writer thread drains into ffmpeg's stdin. If ffmpeg falls behind,
// individual frames cannot simply be dropped the way raw ones could -- a P-frame
// whose reference is missing decodes to garbage -- so the whole queue is
// discarded and the stream resynchronises at the next keyframe. With inline
// headers and a one-second GOP that costs at most a second of video. If ffmpeg
// dies (e.g. MediaMTX restarts) the writer thread respawns it and resynchronises
// the same way.
//
// If MediaMTX requires auth, set_jwt_provider() supplies a fresh JWT before
// each (re)spawn, appended to rtsp_url as MediaMTX's documented `?token=`
// query parameter -- the same JWT BackendClient::get_jwt_token() hands out
// for the backend's own APIs, since MediaMTX validates it the same way.
class FfmpegStreamer {
public:
    // The "streaming" section of the client config, in full. Only rtsp_url and
    // framerate are used by the streamer itself; width, height, bitrate and gop
    // describe the camera and the shared H.264 encoder (see H264Tee), and are
    // parsed here so there is one place that reads the section.
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
        // It also bounds how much video is lost to a resync, and how far back
        // beyond pre_event_length_seconds a clip's pre-roll can reach.
        std::optional<int> gop;

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
    // Safe to call while another thread is in write_access_unit(): that call
    // simply sees the streamer stopped and drops its access unit.
    void stop();

    // Hands over one encoded H.264 access unit in Annex-B form. Returns false
    // if the unit was not queued -- because the streamer isn't running, or
    // because the stream is waiting for a keyframe to (re)synchronise on, or
    // because the queue overflowed and was discarded.
    bool write_access_unit(const uint8_t *data, size_t len, bool keyframe);

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

    // True between a successful start() and a stop().
    bool is_running() const { return running_.load(); }

    // Access units discarded: those arriving while waiting for a keyframe, plus
    // everything thrown away when the queue overflowed.
    uint64_t dropped_frames() const { return dropped_frames_.load(); }
    const Config &config() const { return config_; }

private:
    // One encoded access unit awaiting the writer thread. The keyframe flag
    // travels with it because a freshly spawned ffmpeg must be fed a keyframe
    // first.
    struct AccessUnit {
        std::vector<uint8_t> data;
        bool keyframe;
    };

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
    // Throws away anything queued and waits for the next keyframe before
    // feeding ffmpeg again -- what a fresh process, or a broken pipe, needs.
    void request_resync();
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
    std::deque<AccessUnit> queue_;
    size_t queued_bytes_ = 0;
    bool need_keyframe_ = true;
};

}  // namespace byte_track
