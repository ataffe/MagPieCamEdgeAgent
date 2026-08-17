

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "core/completed_request.hpp"
#include "core/stream_info.hpp"
#include "core/video_options.hpp"

class Encoder;
class RPiCamApp;

namespace libcamera {
class Stream;
}

namespace byte_track {

// Encodes the camera's main stream to H.264 exactly once and fans the resulting
// access units out to every registered sink.
//
// Both consumers of encoded video -- the RTSP stream and the event clip
// recorder -- need the same bitstream, and the Pi has one hardware H.264
// encoder. Encoding per consumer would mean two concurrent V4L2 sessions and
// twice the work, so instead this owns the single encoder and hands each
// finished access unit to all of them.
//
// It uses rpicam-apps' H264Encoder directly, which takes the camera's DMABUF by
// fd -- no packing of I420 into an intermediate buffer, and no copy of the raw
// frame at all. Note that H264Encoder is the V4L2 (bcm2835-codec) encoder: it
// exists on Pi 4 and earlier, and the constructor throws where it does not.
//
// Threading: encode() is called from the camera/post-processor thread, while
// sinks are invoked on the encoder's own output thread. Sinks must not block --
// both FfmpegStreamer::write_access_unit() and
// VideoClipRecorder::on_encoded_frame() copy and return. The buffer passed to a
// sink is only valid for the duration of the call.
class H264Tee {
public:
    struct Config {
        int bitrate = 2000000;   // bits/sec
        int gop = 30;            // keyframe interval in frames (rpicam-vid --intra)
        float framerate = 30.0f;
    };

    // data is one access unit in Annex-B form; because inline headers are
    // enabled, every keyframe is preceded by its SPS/PPS, so any keyframe is a
    // valid place to start decoding.
    using Sink = std::function<void(const uint8_t *data, size_t len, int64_t timestamp_us, bool keyframe)>;

    // Throws std::runtime_error if the encoder cannot be opened (no V4L2 H.264
    // codec, or it is already in use).
    H264Tee(RPiCamApp *app, libcamera::Stream *stream, const Config &config);
    ~H264Tee();

    H264Tee(const H264Tee &) = delete;
    H264Tee &operator=(const H264Tee &) = delete;

    // Register a consumer of encoded frames. Call before the camera starts:
    // sinks are read without locking from the encoder's output thread.
    void add_sink(Sink sink);

    // Submits one camera frame to the encoder. Returns immediately; the encoded
    // result reaches the sinks later, on the encoder's output thread. Holds a
    // reference to the request until the encoder is finished with the buffer,
    // so the camera cannot recycle it mid-encode.
    void encode(CompletedRequestPtr &request);

    uint64_t encoded_frames() const { return encoded_frames_.load(); }
    uint64_t encoded_bytes() const { return encoded_bytes_.load(); }
    uint64_t keyframes() const { return keyframes_.load(); }

private:
    void on_output_ready(void *mem, size_t size, int64_t timestamp_us, bool keyframe);

    RPiCamApp *app_;
    libcamera::Stream *stream_;
    StreamInfo info_;
    // Must outlive encoder_, which holds a pointer to it.
    VideoOptions options_;
    std::unique_ptr<Encoder> encoder_;

    std::vector<Sink> sinks_;

    // Frames handed to the encoder but not yet released. Holding the
    // CompletedRequestPtr is what keeps the underlying DMABUF alive; the
    // encoder's input-done callback pops one as each buffer comes back.
    std::mutex queue_mtx_;
    std::queue<CompletedRequestPtr> in_flight_;

    std::atomic<uint64_t> encoded_frames_{0};
    std::atomic<uint64_t> encoded_bytes_{0};
    std::atomic<uint64_t> keyframes_{0};
    // Guards against logging a size mismatch once per frame.
    bool warned_short_buffer_ = false;
};

}  // namespace byte_track
