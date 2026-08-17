

#include "streaming/video_clip_recorder.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace byte_track {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {
// Sections of the client config file the settings are read from. The clip
// windows have their own section; the frame rate is shared with the RTSP
// stream, because both describe the one camera.
constexpr char kConfigSection[] = "video_clips";
constexpr char kStreamingSection[] = "streaming";

// Overwrites `out` only if `key` is present, so absent keys keep their default.
// Throws json::type_error if the value is present but of the wrong type.
template <typename T>
void read_field(const json &obj, const char *key, T &out)
{
    if (const auto it = obj.find(key); it != obj.end())
        out = it->get<T>();
}

// fork/exec a command and wait for it, with stdin closed off. Returns true only
// if it ran and exited 0. stdout/stderr are inherited so ffmpeg's own warnings
// land in the client's log alongside everything else.
bool run_process(const std::vector<std::string> &args)
{
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto &arg : args)
        argv.push_back(const_cast<char *>(arg.c_str()));
    argv.push_back(nullptr);

    const pid_t pid = ::fork();
    if (pid < 0) {
        spdlog::error("[Clip] fork() failed: {}", std::strerror(errno));
        return false;
    }
    if (pid == 0) {
        // Child: only async-signal-safe calls here -- argv was built before the
        // fork. ffmpeg must not inherit a stdin it might try to read from.
        if (const int devnull = ::open("/dev/null", O_RDONLY); devnull >= 0) {
            ::dup2(devnull, STDIN_FILENO);
            ::close(devnull);
        }
        ::execvp(argv[0], argv.data());
        ::_exit(127);  // execvp only returns on failure
    }

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR)
            return false;
    }
    if (!WIFEXITED(status)) {
        spdlog::error("[Clip] ffmpeg terminated abnormally");
        return false;
    }
    if (WEXITSTATUS(status) != 0) {
        spdlog::error("[Clip] ffmpeg exited with status {}", WEXITSTATUS(status));
        return false;
    }
    return true;
}

bool write_file(const fs::path &path, const std::vector<uint8_t> &bytes)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        return false;
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return out.good();
}

std::optional<std::vector<uint8_t>> read_file(const fs::path &path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        return std::nullopt;
    const auto size = in.tellg();
    if (size < 0)
        return std::nullopt;
    in.seekg(0);

    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!bytes.empty())
        in.read(reinterpret_cast<char *>(bytes.data()), size);
    if (!in)
        return std::nullopt;
    return bytes;
}
}  // namespace

VideoClipRecorder::Config VideoClipRecorder::Config::from_file(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("VideoClipRecorder: could not open config file: " + path);

    json doc;
    try {
        doc = json::parse(in);
    } catch (const json::parse_error &e) {
        throw std::runtime_error("VideoClipRecorder: failed to parse " + path + ": " + e.what());
    }

    const auto section = doc.find(kConfigSection);
    if (section == doc.end())
        throw std::runtime_error("VideoClipRecorder: no \"" + std::string(kConfigSection) +
                                 "\" section in " + path);

    Config config;
    try {
        read_field(*section, "pre_event_length_seconds", config.pre_event_length_seconds);
        read_field(*section, "post_event_length_seconds", config.post_event_length_seconds);
        read_field(*section, "send_video_clip_cooldown", config.send_video_clip_cooldown);
        read_field(*section, "work_dir", config.work_dir);
        read_field(*section, "upload_type", config.upload_type);
        read_field(*section, "content_type", config.content_type);
        read_field(*section, "max_buffer_bytes", config.max_buffer_bytes);
        // The frame rate is a property of the camera, so it is read from where
        // the stream already declares it rather than duplicated per section.
        if (const auto streaming = doc.find(kStreamingSection); streaming != doc.end())
            read_field(*streaming, "framerate", config.framerate);
    } catch (const json::exception &e) {
        throw std::runtime_error("VideoClipRecorder: invalid \"" + std::string(kConfigSection) +
                                 "\" section in " + path + ": " + e.what());
    }

    const auto require = [&path](bool ok, const char *what) {
        if (!ok)
            throw std::runtime_error("VideoClipRecorder: invalid \"" + std::string(kConfigSection) +
                                     "\" section in " + path + ": " + what);
    };
    require(config.pre_event_length_seconds > 0, "pre_event_length_seconds must be positive");
    require(config.post_event_length_seconds > 0, "post_event_length_seconds must be positive");
    require(config.send_video_clip_cooldown >= 0, "send_video_clip_cooldown must not be negative");
    require(config.framerate > 0, "framerate must be positive");
    require(config.max_buffer_bytes > 0, "max_buffer_bytes must be positive");
    require(!config.work_dir.empty(), "work_dir must not be empty");
    require(!config.upload_type.empty(), "upload_type must not be empty");
    require(!config.content_type.empty(), "content_type must not be empty");

    // Not fatal -- it still produces valid clips -- but consecutive clips would
    // then cover overlapping footage, which is the duplication the cooldown is
    // there to prevent in the first place.
    if (config.send_video_clip_cooldown <
        config.pre_event_length_seconds + config.post_event_length_seconds) {
        spdlog::warn("[Clip] send_video_clip_cooldown ({}) is below pre + post ({}); "
                     "consecutive clips will overlap",
                     config.send_video_clip_cooldown,
                     config.pre_event_length_seconds + config.post_event_length_seconds);
    }

    return config;
}

VideoClipRecorder::VideoClipRecorder(Config config) : config_(std::move(config))
{
    muxer_ = [this](const std::vector<uint8_t> &annexb) { return mux_with_ffmpeg(annexb); };
}

VideoClipRecorder::~VideoClipRecorder() { stop(); }

void VideoClipRecorder::set_uploader(Uploader uploader) { uploader_ = std::move(uploader); }

void VideoClipRecorder::set_muxer(Muxer muxer) { muxer_ = std::move(muxer); }

void VideoClipRecorder::start()
{
    if (running_.exchange(true))
        return;
    worker_ = std::thread(&VideoClipRecorder::worker_loop, this);
    spdlog::info("[Clip] Recording {}s before / {}s after each event, at most one clip every {}s",
                 config_.pre_event_length_seconds, config_.post_event_length_seconds,
                 config_.send_video_clip_cooldown);
}

void VideoClipRecorder::stop()
{
    if (!running_.exchange(false))
        return;

    job_cv_.notify_all();
    if (worker_.joinable())
        worker_.join();

    std::lock_guard<std::mutex> lock(ring_mtx_);
    ring_.clear();
    ring_bytes_ = 0;
    capturing_ = false;
    capture_detection_key_.clear();
}

bool VideoClipRecorder::is_capturing() const
{
    std::lock_guard<std::mutex> lock(ring_mtx_);
    return capturing_;
}

bool VideoClipRecorder::in_cooldown() const
{
    std::lock_guard<std::mutex> lock(ring_mtx_);
    // A stopped recorder holds a frozen timestamp, which would otherwise leave
    // it permanently "in cooldown" after shutdown.
    return running_ && in_cooldown_locked();
}

bool VideoClipRecorder::in_cooldown_locked() const
{
    // Mid-capture, the clip being recorded already covers whatever just
    // happened; afterwards, the clip it produced does, until the cooldown from
    // the event that armed it has elapsed.
    if (capturing_)
        return true;
    return has_triggered_ && latest_timestamp_us_ - last_trigger_us_ < cooldown_us();
}

size_t VideoClipRecorder::buffered_frames() const
{
    std::lock_guard<std::mutex> lock(ring_mtx_);
    return ring_.size();
}

size_t VideoClipRecorder::buffered_bytes() const
{
    std::lock_guard<std::mutex> lock(ring_mtx_);
    return ring_bytes_;
}

void VideoClipRecorder::on_encoded_frame(const uint8_t *data, size_t len, int64_t timestamp_us,
                                         bool keyframe)
{
    if (!running_ || data == nullptr || len == 0)
        return;

    std::lock_guard<std::mutex> lock(ring_mtx_);

    // A clip has to start on a keyframe, so anything before the first one is of
    // no use -- it can neither be decoded nor be trimmed back to.
    if (ring_.empty() && !keyframe)
        return;

    ring_.push_back(AccessUnit{std::vector<uint8_t>(data, data + len), timestamp_us, keyframe});
    ring_bytes_ += len;
    latest_timestamp_us_ = timestamp_us;

    if (!capturing_) {
        trim_locked();
        return;
    }

    if (timestamp_us >= capture_until_us_) {
        dispatch_locked();
        return;
    }

    // Only reachable if the encoder is producing far more than the configured
    // bitrate; drop the clip rather than let the buffer grow without bound.
    if (ring_bytes_ > static_cast<size_t>(config_.max_buffer_bytes)) {
        spdlog::error("[Clip] Buffer exceeded {} bytes mid-capture, abandoning clip",
                      config_.max_buffer_bytes);
        ring_.clear();
        ring_bytes_ = 0;
        capturing_ = false;
        capture_detection_key_.clear();
        ++clips_failed_;
    }
}

bool VideoClipRecorder::on_event(const std::string &detection_key)
{
    std::lock_guard<std::mutex> lock(ring_mtx_);

    if (!running_ || ring_.empty())
        return false;  // nothing buffered yet, so there is no clip to build

    // Already covered -- either by the clip being recorded right now or by the
    // one the last event produced -- so a second upload would be a near
    // duplicate. Callers that check in_cooldown() first rarely reach this;
    // it stays as the authority on whether a clip gets armed.
    if (in_cooldown_locked()) {
        ++events_suppressed_;
        return false;
    }

    has_triggered_ = true;
    last_trigger_us_ = latest_timestamp_us_;
    capturing_ = true;
    capture_until_us_ = latest_timestamp_us_ + post_event_us();
    capture_detection_key_ = detection_key;
    ++clips_captured_;
    return true;
}

void VideoClipRecorder::trim_locked()
{
    const int64_t cutoff = latest_timestamp_us_ - pre_event_us();

    // Everything before the last keyframe at or before the cutoff is outside
    // the window and not needed to decode what is inside it.
    size_t drop_before = 0;
    for (size_t i = 0; i < ring_.size(); ++i) {
        if (ring_[i].timestamp_us > cutoff)
            break;
        if (ring_[i].keyframe)
            drop_before = i;
    }

    for (size_t i = 0; i < drop_before; ++i) {
        ring_bytes_ -= ring_.front().data.size();
        ring_.pop_front();
    }
}

void VideoClipRecorder::dispatch_locked()
{
    Job job;
    job.frames.swap(ring_);
    job.detection_key = std::move(capture_detection_key_);
    capture_detection_key_.clear();
    ring_bytes_ = 0;
    capturing_ = false;

    // The ring restarts empty: the next event cannot come until the cooldown
    // has elapsed anyway, which leaves ample time to rebuild the pre-roll.
    {
        std::lock_guard<std::mutex> lock(job_mtx_);
        jobs_.push_back(std::move(job));
    }
    job_cv_.notify_one();
}

void VideoClipRecorder::worker_loop()
{
    while (true) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(job_mtx_);
            job_cv_.wait(lock, [this] { return !running_ || !jobs_.empty(); });
            // On shutdown, queued clips are dropped rather than holding up the
            // exit; a clip already being processed below still finishes.
            if (!running_)
                break;
            job = std::move(jobs_.front());
            jobs_.pop_front();
        }
        process_clip(job);
    }
}

void VideoClipRecorder::process_clip(const Job &job)
{
    const auto &clip = job.frames;
    if (clip.empty())
        return;

    size_t total = 0;
    for (const auto &au : clip)
        total += au.data.size();

    std::vector<uint8_t> annexb;
    annexb.reserve(total);
    for (const auto &au : clip)
        annexb.insert(annexb.end(), au.data.begin(), au.data.end());

    ++clip_sequence_;
    const double duration_s =
        static_cast<double>(clip.back().timestamp_us - clip.front().timestamp_us) / 1e6;
    spdlog::info("[Clip] Building clip for detection {}: {} frames, {:.1f}s, {} bytes of H.264",
                 job.detection_key, clip.size(), duration_s, annexb.size());

    const auto mp4 = muxer_ ? muxer_(annexb) : std::nullopt;
    if (!mp4) {
        ++clips_failed_;
        return;
    }

    if (!uploader_) {
        spdlog::warn("[Clip] No uploader configured, discarding {} byte clip", mp4->size());
        return;
    }

    if (uploader_(*mp4, job.detection_key)) {
        ++clips_uploaded_;
        spdlog::info("[Clip] Uploaded {:.1f}s clip ({} bytes) for detection {}",
                     duration_s, mp4->size(), job.detection_key);
    } else {
        ++clips_failed_;
        spdlog::error("[Clip] Failed to upload {:.1f}s clip for detection {}",
                      duration_s, job.detection_key);
    }
}

std::optional<std::vector<uint8_t>> VideoClipRecorder::mux_with_ffmpeg(
    const std::vector<uint8_t> &annexb) const
{
    // The elementary stream carries no container timestamps, so ffmpeg is told
    // the frame rate on the input side and then just copies the bitstream --
    // no re-encode, which is the whole point of buffering it already encoded.
    //
    // ffmpeg logs "Timestamps are unset in a packet for stream 0" once per
    // clip. It is a deprecation notice from the raw H.264 demuxer, not a
    // failure: the muxed MP4 still gets the right duration and frame count.
    const std::string stem = "scout-clip-" + std::to_string(::getpid()) + "-" +
                             std::to_string(clip_sequence_);
    const fs::path in_path = fs::path(config_.work_dir) / (stem + ".h264");
    const fs::path out_path = fs::path(config_.work_dir) / (stem + ".mp4");

    std::error_code ec;
    fs::create_directories(config_.work_dir, ec);

    const auto cleanup = [&]() {
        std::error_code remove_ec;
        fs::remove(in_path, remove_ec);
        fs::remove(out_path, remove_ec);
    };

    if (!write_file(in_path, annexb)) {
        spdlog::error("[Clip] Could not write {}: {}", in_path.string(), std::strerror(errno));
        cleanup();
        return std::nullopt;
    }

    const std::vector<std::string> args = {
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-nostdin",
        "-y",
        "-f", "h264",
        "-framerate", std::to_string(config_.framerate),
        "-i", in_path.string(),
        "-c:v", "copy",
        // Puts the index at the front so the clip can be played back without
        // downloading all of it first.
        "-movflags", "+faststart",
        out_path.string(),
    };

    if (!run_process(args)) {
        cleanup();
        return std::nullopt;
    }

    auto mp4 = read_file(out_path);
    if (!mp4)
        spdlog::error("[Clip] ffmpeg produced no readable output at {}", out_path.string());

    cleanup();
    return mp4;
}

}  // namespace byte_track
