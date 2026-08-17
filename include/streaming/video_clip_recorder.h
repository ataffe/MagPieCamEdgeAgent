

#pragma once

#include <atomic>
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

namespace byte_track {

// Keeps a rolling window of recently encoded H.264 frames so that, when a
// detection fires, the clip that gets uploaded covers the seconds *before* the
// event as well as the seconds after it.
//
// The buffer holds encoded access units, not camera frames. Raw I420 is not an
// option: a 10s pre-roll at 1280x720 is ~415 MB, whereas the same 10s of H.264
// at 2 Mbit/s is ~2.5 MB. The cost of that is that a clip can only begin on a
// keyframe, so the pre-roll is trimmed back to the last keyframe at or before
// the requested window -- i.e. it is always at least pre_event_length_seconds
// long, and at most one GOP longer.
//
// Threading: on_encoded_frame() is called from the encoder's output thread and
// on_event() from the detection thread; both only touch the ring buffer and
// return immediately. Muxing and uploading -- seconds of work -- happen on this
// class's own worker thread, so neither the camera pipeline nor the encoder
// ever blocks on the network.
//
// Timing is measured entirely in encoder timestamps (the sensor timestamps
// handed to on_encoded_frame), never in wall-clock time. The pre-roll window,
// the post-event window and the cooldown are therefore all in the same clock as
// the frames themselves, which also makes the whole thing deterministic to test.
class VideoClipRecorder {
public:
    struct Config {
        // Seconds of already-encoded footage kept ahead of an event, and
        // seconds to keep recording after one.
        int pre_event_length_seconds = 10;
        int post_event_length_seconds = 10;
        // Minimum spacing between clips, in seconds. This is what stops
        // overlapping detections producing near-duplicate uploads: an event
        // that lands inside the cooldown of the previous one is already covered
        // by the clip that event produced, so it is dropped. Should be at least
        // pre + post, or consecutive clips would cover overlapping footage.
        // Detection stills are held off for the same stretch -- see
        // in_cooldown() -- so a repeat sighting produces neither a clip nor an
        // image.
        int send_video_clip_cooldown = 60;
        // Frame rate declared to the muxer. The encoded stream carries no
        // container timestamps, so this is what gives the MP4 its timebase;
        // it should match the rate the camera is actually running at.
        int framerate = 30;
        // Where the intermediate .h264 / .mp4 files are written. Both are
        // deleted as soon as the clip has been uploaded.
        std::string work_dir = "/tmp";
        // Passed straight through to the backend's presign endpoint and to the
        // storage PUT respectively.
        std::string upload_type = "VIDEO_CLIP";
        std::string content_type = "video/mp4";
        // Hard ceiling on the ring buffer. Only reachable if the bitrate is far
        // higher than configured or the encoder emits no keyframes; hitting it
        // abandons the clip rather than growing until the Pi runs out of memory.
        int max_buffer_bytes = 64 * 1024 * 1024;

        // Reads the "video_clips" section of the client config JSON, plus
        // "framerate" from the "streaming" section (where it already lives, and
        // where it has to match the camera). Every key is optional and falls
        // back to the default above.
        //
        // Throws std::runtime_error if the file is missing or malformed, if
        // there is no "video_clips" section, or if a value is the wrong type or
        // out of range -- callers decide whether that is fatal.
        static Config from_file(const std::string &path);
    };

    // Uploads a finished MP4, tagged with the detection key the clip was armed
    // with so the backend can pair the two. Returns true on success. Invoked
    // from the worker thread, so it must be safe to call there --
    // BackendClient::upload_object() is, since it only reads state fixed at
    // construction.
    using Uploader = std::function<bool(const std::vector<uint8_t> &clip,
                                        const std::string &detection_key)>;

    // Turns a concatenated Annex-B H.264 elementary stream into a finished
    // container. Returns std::nullopt if the clip could not be muxed. Defaults
    // to a child ffmpeg doing a stream copy; replaceable so the buffering and
    // cooldown logic can be tested without spawning a process.
    using Muxer = std::function<std::optional<std::vector<uint8_t>>(const std::vector<uint8_t> &annexb)>;

    explicit VideoClipRecorder(Config config);
    ~VideoClipRecorder();

    VideoClipRecorder(const VideoClipRecorder &) = delete;
    VideoClipRecorder &operator=(const VideoClipRecorder &) = delete;

    // Call before start(). Without an uploader a finished clip is muxed and
    // then discarded, which is only useful in tests.
    void set_uploader(Uploader uploader);
    void set_muxer(Muxer muxer);

    // Starts the worker thread and begins accepting frames. Idempotent.
    void start();
    // Stops accepting frames and joins the worker. A clip already being muxed
    // or uploaded is allowed to finish; clips still queued behind it are
    // dropped, so shutdown isn't held up by a backlog. Idempotent.
    void stop();

    // Hands over one encoded access unit. Called from the encoder's output
    // thread; copies the bytes and returns. Frames arriving before the first
    // keyframe are discarded, since a clip cannot start mid-GOP.
    void on_encoded_frame(const uint8_t *data, size_t len, int64_t timestamp_us, bool keyframe);

    // Reports that a detection fired. Returns true if this armed a new capture,
    // false if it was absorbed -- either a capture is already running or the
    // cooldown since the last one has not elapsed. Called from the detection
    // thread; does no I/O.
    //
    // detection_key identifies the detection image this clip belongs to, and is
    // handed back to the Uploader when the clip is ready. Absorbed events keep
    // the key of the event that armed the capture: they are already covered by
    // that clip, which is the whole point of suppressing them.
    bool on_event(const std::string &detection_key);

    // True between arming a capture and the post-event window closing.
    bool is_capturing() const;

    // True when an event right now would be absorbed rather than start a clip:
    // a capture is already running, or the cooldown since the last one has not
    // elapsed. Lets a caller skip work that a suppressed event would make
    // pointless -- the client gates the detection image upload on it, so a
    // detection inside the cooldown costs neither a clip nor a still.
    //
    // Measured in encoder timestamps like the rest of the class, so it only
    // advances while frames are arriving; a stopped recorder is never in
    // cooldown, since it can't produce clips to be duplicates of.
    bool in_cooldown() const;

    size_t buffered_frames() const;
    size_t buffered_bytes() const;
    uint64_t clips_captured() const { return clips_captured_.load(); }
    uint64_t clips_uploaded() const { return clips_uploaded_.load(); }
    uint64_t clips_failed() const { return clips_failed_.load(); }
    // Events absorbed by the cooldown or by an in-progress capture.
    uint64_t events_suppressed() const { return events_suppressed_.load(); }

    const Config &config() const { return config_; }

private:
    // One encoded access unit, with the metadata needed to trim the ring back
    // to a keyframe and to know when the post-event window has closed.
    struct AccessUnit {
        std::vector<uint8_t> data;
        int64_t timestamp_us;
        bool keyframe;
    };

    // A captured clip on its way to the worker thread: the frames, plus the
    // detection image it is keyed to.
    struct Job {
        std::deque<AccessUnit> frames;
        std::string detection_key;
    };

    // Shared by on_event() and in_cooldown(): whether an event now would be
    // absorbed. Caller holds ring_mtx_.
    bool in_cooldown_locked() const;
    // Drops leading access units that are older than the pre-event window,
    // stopping at the last keyframe at or before the window's start so what is
    // left is still independently decodable. Caller holds ring_mtx_.
    void trim_locked();
    // Moves the ring into the worker's queue. Caller holds ring_mtx_.
    void dispatch_locked();

    void worker_loop();
    // Concatenates a clip's access units, muxes, and uploads it.
    void process_clip(const Job &job);
    // The default Muxer: writes the elementary stream to work_dir, runs ffmpeg
    // to wrap it in MP4, reads the result back, and removes both files.
    std::optional<std::vector<uint8_t>> mux_with_ffmpeg(const std::vector<uint8_t> &annexb) const;

    int64_t pre_event_us() const { return static_cast<int64_t>(config_.pre_event_length_seconds) * 1000000; }
    int64_t post_event_us() const { return static_cast<int64_t>(config_.post_event_length_seconds) * 1000000; }
    int64_t cooldown_us() const { return static_cast<int64_t>(config_.send_video_clip_cooldown) * 1000000; }

    Config config_;
    Uploader uploader_;
    Muxer muxer_;

    mutable std::mutex ring_mtx_;
    std::deque<AccessUnit> ring_;
    size_t ring_bytes_ = 0;
    int64_t latest_timestamp_us_ = 0;
    bool capturing_ = false;
    int64_t capture_until_us_ = 0;
    // Detection image the in-progress capture is keyed to.
    std::string capture_detection_key_;
    // Timestamp of the last event that armed a capture, and whether there has
    // been one at all (so the very first event isn't held off by the cooldown).
    int64_t last_trigger_us_ = 0;
    bool has_triggered_ = false;

    std::mutex job_mtx_;
    std::condition_variable job_cv_;
    std::deque<Job> jobs_;

    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> clips_captured_{0};
    std::atomic<uint64_t> clips_uploaded_{0};
    std::atomic<uint64_t> clips_failed_{0};
    std::atomic<uint64_t> events_suppressed_{0};
    // Only ever touched by the worker; makes the temp file names unique.
    uint64_t clip_sequence_ = 0;
};

}  // namespace byte_track
