// Copyright © 2026 Alexander Taffe

#include "streaming/ffmpeg_streamer.h"

#include <csignal>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <sys/wait.h>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace byte_track {

using namespace std::chrono;
using json = nlohmann::json;

namespace {
// How long to wait before respawning ffmpeg after it dies. Long enough that a
// permanently broken configuration doesn't spin, short enough that a MediaMTX
// restart is picked up quickly.
constexpr auto kRestartDelay = seconds(2);
// Grace period for ffmpeg to flush and exit after its stdin is closed.
constexpr auto kShutdownTimeout = seconds(3);
// Section of the client config file holding the streaming settings.
constexpr char kConfigSection[] = "streaming";

// Overwrites `out` only if `key` is present, so absent keys keep their default.
// Throws json::type_error if the value is present but of the wrong type.
template <typename T>
void read_field(const json &obj, const char *key, T &out)
{
    if (const auto it = obj.find(key); it != obj.end())
        out = it->get<T>();
}
}  // namespace

FfmpegStreamer::Config FfmpegStreamer::Config::from_file(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("FfmpegStreamer: could not open config file: " + path);

    json doc;
    try {
        doc = json::parse(in);
    } catch (const json::parse_error &e) {
        throw std::runtime_error("FfmpegStreamer: failed to parse " + path + ": " + e.what());
    }

    const auto section = doc.find(kConfigSection);
    if (section == doc.end())
        throw std::runtime_error("FfmpegStreamer: no \"" + std::string(kConfigSection) +
                                 "\" section in " + path);

    Config config;
    try {
        read_field(*section, "rtsp_url", config.rtsp_url);
        read_field(*section, "width", config.width);
        read_field(*section, "height", config.height);
        read_field(*section, "framerate", config.framerate);
        read_field(*section, "bitrate", config.bitrate);
        read_field(*section, "encoder", config.encoder);
        // Left unset, gop tracks framerate — see the Config declaration.
        if (const auto it = section->find("gop"); it != section->end())
            config.gop = it->get<int>();
    } catch (const json::exception &e) {
        throw std::runtime_error("FfmpegStreamer: invalid \"" + std::string(kConfigSection) +
                                 "\" section in " + path + ": " + e.what());
    }

    // Catch nonsense here rather than letting it surface as an opaque ffmpeg
    // failure several seconds into startup.
    const auto require = [&path](bool ok, const char *what) {
        if (!ok)
            throw std::runtime_error("FfmpegStreamer: invalid \"" + std::string(kConfigSection) +
                                     "\" section in " + path + ": " + what);
    };
    require(!config.rtsp_url.empty(), "rtsp_url must not be empty");
    require(!config.encoder.empty(), "encoder must not be empty");
    // H.264 needs even dimensions, and so does the I420 packing.
    require(config.width > 0 && config.width % 2 == 0, "width must be positive and even");
    require(config.height > 0 && config.height % 2 == 0, "height must be positive and even");
    require(config.framerate > 0, "framerate must be positive");
    require(config.bitrate > 0, "bitrate must be positive");
    require(!config.gop || *config.gop > 0, "gop must be positive");

    return config;
}

FfmpegStreamer::FfmpegStreamer(Config config) : config_(std::move(config)) {}

FfmpegStreamer::~FfmpegStreamer() { stop(); }

size_t FfmpegStreamer::frame_bytes() const
{
    // Computes size of YUV420 frame
    // I420: full-size Y plane + two quarter-size chroma planes.
    return static_cast<size_t>(config_.width) * config_.height * 3 / 2;
}

std::vector<std::string> FfmpegStreamer::build_args() const
{
    std::vector<std::string> args = {
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-nostdin",
        // Input: packed I420 frames on stdin, at the camera's frame rate.
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-s", std::to_string(config_.width) + "x" + std::to_string(config_.height),
        "-framerate", std::to_string(config_.framerate),
        "-i", "-",
        // Encode: the job rpicam-vid used to do before `-c:v copy`.
        "-c:v", config_.encoder,
        "-b:v", std::to_string(config_.bitrate),
        "-g", std::to_string(config_.gop.value_or(config_.framerate)),
        "-bf", "0",  // no B-frames: they only add latency on a live stream
    };

    // Software encoding needs to be told not to buffer ahead, or it adds
    // seconds of latency. The hardware encoder is already low-latency.
    if (config_.encoder.rfind("libx264", 0) == 0) {
        args.insert(args.end(), {"-preset", "ultrafast", "-tune", "zerolatency"});
    }

    args.insert(args.end(), {"-f", "rtsp", "-rtsp_transport", "tcp", config_.rtsp_url});
    return args;
}

bool FfmpegStreamer::spawn()
{
    const std::vector<std::string> args = build_args();
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto &arg : args)
        argv.push_back(const_cast<char *>(arg.c_str()));
    argv.push_back(nullptr);

    int fds[2];
    if (::pipe(fds) != 0) {
        spdlog::error("[Stream] pipe() failed: {}", std::strerror(errno));
        return false;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        spdlog::error("[Stream] fork() failed: {}", std::strerror(errno));
        ::close(fds[0]);
        ::close(fds[1]);
        return false;
    }

    if (pid == 0) {
        // Child: read end becomes stdin, then hand over to ffmpeg. Only
        // async-signal-safe calls here — argv was built before the fork.
        ::dup2(fds[0], STDIN_FILENO);
        ::close(fds[0]);
        ::close(fds[1]);
        ::signal(SIGPIPE, SIG_DFL);
        ::execvp("ffmpeg", argv.data());
        ::_exit(127);  // execvp only returns on failure
    }

    ::close(fds[0]);
    pipe_fd_ = fds[1];
    child_pid_ = pid;
    spdlog::info("[Stream] ffmpeg started (pid {}): {}x{}@{} -> {}",
                 pid, config_.width, config_.height, config_.framerate, config_.rtsp_url);
    std::string cmdline;
    for (const auto &arg : args)
        cmdline += arg + " ";
    spdlog::debug("[Stream] {}", cmdline);
    return true;
}

void FfmpegStreamer::reap()
{
    if (pipe_fd_ >= 0) {
        ::close(pipe_fd_);  // ffmpeg sees EOF on stdin and exits cleanly
        pipe_fd_ = -1;
    }
    if (child_pid_ < 0)
        return;

    // Give ffmpeg a moment to flush, then insist.
    const auto deadline = steady_clock::now() + kShutdownTimeout;
    int status = 0;
    while (true) {
        const pid_t r = ::waitpid(child_pid_, &status, WNOHANG);
        if (r == child_pid_)
            break;
        if (r < 0) {
            if (errno == EINTR)
                continue;
            break;  // already reaped or never ours
        }
        if (steady_clock::now() >= deadline) {
            spdlog::warn("[Stream] ffmpeg (pid {}) did not exit, killing it", child_pid_);
            ::kill(child_pid_, SIGKILL);
            ::waitpid(child_pid_, &status, 0);
            break;
        }
        ::usleep(20 * 1000);
    }

    // During shutdown a non-zero status is expected: a Ctrl-C reaches ffmpeg
    // too (same process group) and it exits 255 on signal.
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        if (running_)
            spdlog::warn("[Stream] ffmpeg exited with status {}", WEXITSTATUS(status));
        else
            spdlog::debug("[Stream] ffmpeg exited with status {}", WEXITSTATUS(status));
    }
    child_pid_ = -1;
}

bool FfmpegStreamer::write_all(const uint8_t *data, size_t len)
{
    size_t written = 0;
    while (written < len) {
        const ssize_t n = ::write(pipe_fd_, data + written, len - written);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            spdlog::warn("[Stream] write to ffmpeg failed: {}", std::strerror(errno));
            return false;
        }
        written += static_cast<size_t>(n);
    }
    return true;
}

bool FfmpegStreamer::start()
{
    if (running_)
        return true;

    // A dead ffmpeg must surface as a write() error, not a fatal SIGPIPE.
    std::signal(SIGPIPE, SIG_IGN);

    if (!spawn())
        return false;

    running_ = true;
    writer_ = std::thread(&FfmpegStreamer::writer_loop, this);
    return true;
}

void FfmpegStreamer::stop()
{
    if (!running_.exchange(false)) {
        reap();  // spawned but never started, or already stopped
        return;
    }

    cv_.notify_all();
    if (writer_.joinable())
        writer_.join();
    reap();
}

bool FfmpegStreamer::write_frame(const uint8_t *data, size_t len)
{
    if (!running_)
        return false;

    if (len != frame_bytes()) {
        spdlog::warn("[Stream] dropping frame of {} bytes, expected {}", len, frame_bytes());
        return false;
    }

    bool dropped;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        dropped = has_frame_;  // the writer never picked up the previous frame
        pending_.assign(data, data + len);
        has_frame_ = true;
    }
    cv_.notify_one();

    if (dropped)
        ++dropped_frames_;
    return !dropped;
}

void FfmpegStreamer::writer_loop()
{
    // Swapped with pending_ so neither buffer has to reallocate per frame.
    std::vector<uint8_t> frame;

    while (running_) {
        {
            // Blocks while copying the pending frame
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this] { return !running_ || has_frame_; });
            if (!running_)
                break;
            frame.swap(pending_);
            has_frame_ = false;
        }

        if (pipe_fd_ < 0) {
            // ffmpeg died; back off, then bring it back and drop this frame.
            if (steady_clock::now() < next_spawn_)
                continue;
            if (!spawn()) {
                next_spawn_ = steady_clock::now() + kRestartDelay;
                continue;
            }
        }

        if (!write_all(frame.data(), frame.size())) {
            spdlog::warn("[Stream] ffmpeg pipe broken, restarting in {}s", kRestartDelay.count());
            reap();
            next_spawn_ = steady_clock::now() + kRestartDelay;
        }
    }
}

}  // namespace byte_track
