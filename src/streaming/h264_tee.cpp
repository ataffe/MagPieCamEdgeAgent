// Copyright © 2026 Alexander Taffe

#include "streaming/h264_tee.h"

#include <stdexcept>
#include <utility>

#include <libcamera/control_ids.h>
#include <libcamera/framebuffer.h>
#include <libcamera/stream.h>

#include <spdlog/spdlog.h>

#include "core/buffer_sync.hpp"
#include "core/rpicam_app.hpp"
#include "encoder/h264_encoder.hpp"

namespace byte_track {

H264Tee::H264Tee(RPiCamApp *app, libcamera::Stream *stream, const Config &config)
    : app_(app), stream_(stream)
{
    if (!app_ || !stream_)
        throw std::runtime_error("H264Tee: no camera stream to encode");

    info_ = app_->GetStreamInfo(stream_);

    // H264Encoder reads its settings straight out of VideoOptions, the same way
    // rpicam-vid's do. Every field it looks at has to be set explicitly: a
    // default-constructed VideoOptions is not default-valued, because boost
    // applies the per-option defaults only when a command line is parsed, and
    // until then the plain members hold whatever was on the heap. (Parsing an
    // empty command line is not an alternative -- Options::Parse dereferences
    // an app pointer that only RPiCamApp's constructor ever sets.) The list
    // below is exactly what the encoder consumes: width/height become the
    // capture format, and the other three become V4L2 controls. profile and
    // level are std::strings, so they are properly empty, and the encoder skips
    // them.
    //
    // Getting this wrong is not benign: uninitialised width/height go straight
    // into VIDIOC_S_FMT, and the codec then refuses to start streaming.
    options_.Set().width  = info_.width;
    options_.Set().height = info_.height;
    // Inline headers matter to both consumers: they put SPS/PPS in front of
    // every keyframe, so a viewer joining the RTSP stream mid-GOP and a clip
    // that begins at a buffered keyframe are both decodable.
    options_.Set().inline_headers = true;
    options_.Set().bitrate.set(std::to_string(config.bitrate) + "bps");
    options_.Set().intra = static_cast<unsigned int>(config.gop);
    options_.Set().framerate = config.framerate;

    encoder_ = std::make_unique<H264Encoder>(&options_, info_);

    encoder_->SetInputDoneCallback([this](void *) {
        // The encoder is done with a camera buffer; dropping our reference to
        // the request hands it back to libcamera. Buffers come back in the
        // order they were submitted, so the front is always the right one.
        std::lock_guard<std::mutex> lock(queue_mtx_);
        if (!in_flight_.empty())
            in_flight_.pop();
    });

    encoder_->SetOutputReadyCallback(
        [this](void *mem, size_t size, int64_t timestamp_us, bool keyframe) {
            on_output_ready(mem, size, timestamp_us, keyframe);
        });

    spdlog::info("[Encode] H.264 encoder up: {}x{} @ {:.0f}fps, {} bps, keyframe every {} frames",
                 info_.width, info_.height, config.framerate, config.bitrate, config.gop);
}

H264Tee::~H264Tee()
{
    // Joins the encoder's poll/output threads first, so no sink or input-done
    // callback can fire while the queue below is being emptied.
    encoder_.reset();

    std::lock_guard<std::mutex> lock(queue_mtx_);
    while (!in_flight_.empty())
        in_flight_.pop();
}

void H264Tee::add_sink(Sink sink) { sinks_.push_back(std::move(sink)); }

void H264Tee::on_output_ready(void *mem, size_t size, int64_t timestamp_us, bool keyframe)
{
    ++encoded_frames_;
    encoded_bytes_ += size;
    if (keyframe)
        ++keyframes_;

    const auto *data = static_cast<const uint8_t *>(mem);
    for (const auto &sink : sinks_)
        sink(data, size, timestamp_us, keyframe);
}

void H264Tee::encode(CompletedRequestPtr &request)
{
    const auto it = request->buffers.find(stream_);
    if (it == request->buffers.end())
        return;

    libcamera::FrameBuffer *buffer = it->second;
    if (buffer->planes().empty())
        return;

    BufferReadSync sync(app_, buffer);
    const auto &planes = sync.Get();
    if (planes.empty() || planes[0].data() == nullptr)
        return;

    // The encoder takes the whole frame as one DMABUF. libcamera may describe it
    // as a single plane or as three contiguous ones within the same allocation,
    // so the length is the sum rather than just the luma plane's.
    size_t size = 0;
    for (const auto &plane : planes)
        size += plane.size();

    const size_t expected = static_cast<size_t>(info_.stride) * info_.height * 3 / 2;
    if (size < expected && !warned_short_buffer_) {
        spdlog::warn("[Encode] Camera buffer is {} bytes, expected {} for {}x{} YUV420",
                     size, expected, info_.width, info_.height);
        warned_short_buffer_ = true;
    }

    const auto sensor_timestamp = request->metadata.get(libcamera::controls::SensorTimestamp);
    const int64_t timestamp_ns =
        sensor_timestamp ? *sensor_timestamp : static_cast<int64_t>(buffer->metadata().timestamp);

    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        in_flight_.push(request);
    }
    encoder_->EncodeBuffer(buffer->planes()[0].fd.get(), size, planes[0].data(), info_,
                           timestamp_ns / 1000);
}

}  // namespace byte_track
