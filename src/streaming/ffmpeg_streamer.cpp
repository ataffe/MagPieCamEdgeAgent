// Copyright © 2026 Alexander Taffe

#include "streaming/ffmpeg_streamer.h"

#include <csignal>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <poll.h>
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
// If ffmpeg's RTSP muxer dies (e.g. MediaMTX rejects an expired/invalid
// token) the process can stay alive without draining stdin. A blocking
// write() would then hang forever, and stop() would too -- it joins the
// writer thread before closing the pipe. Bounding each write lets a stalled
// ffmpeg be detected and treated as a broken pipe (see write_all()) instead.
constexpr int kWriteTimeoutMs = 2000;
// How much encoded video may sit waiting for ffmpeg before the queue is
// declared hopeless and dropped. At the configured 2 Mbit/s this is roughly two
// seconds -- past that, a live stream is better off resynchronising at the next
// keyframe than delivering stale frames.
constexpr size_t kMaxQueuedBytes = 512 * 1024;
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

// Percent-encodes everything but RFC 3986 "unreserved" characters, so a JWT
// (or anything else) is safe to embed as a single query parameter value.
std::string url_encode(const std::string &value)
{
    std::ostringstream out;
    for (const unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out << static_cast<char>(c);
        } else {
            out << '%' << std::hex << std::uppercase << std::setw(2) << std::setfill('0')
                << static_cast<int>(c) << std::dec;
        }
    }
    return out.str();
}

// Appends MediaMTX's documented auth query parameter to a base RTSP URL, e.g.
// append_token("rtsp://h/cam", "abc") -> "rtsp://h/cam?token=abc", or with
// `&` if the URL already carries a query string. Returns base_url unchanged
// if there is no token.
std::string append_token(const std::string &base_url, const std::optional<std::string> &token)
{
    if (!token)
        return base_url;
    std::string url = base_url;
    url += (url.find('?') == std::string::npos) ? '?' : '&';
    url += "token=" + url_encode(*token);
    return url;
}

// Replaces a `token=...` query value with a placeholder so the ffmpeg command
// line logged at debug level doesn't leak the JWT.
std::string redact_token(const std::string &arg)
{
    const auto pos = arg.find("token=");
    if (pos == std::string::npos)
        return arg;
    const auto amp = arg.find('&', pos);
    return arg.substr(0, pos + 6) + "<redacted>" + (amp == std::string::npos ? "" : arg.substr(amp));
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
    // H.264 needs even dimensions, and so does the camera's YUV420 output.
    require(config.width > 0 && config.width % 2 == 0, "width must be positive and even");
    require(config.height > 0 && config.height % 2 == 0, "height must be positive and even");
    require(config.framerate > 0, "framerate must be positive");
    require(config.bitrate > 0, "bitrate must be positive");
    require(!config.gop || *config.gop > 0, "gop must be positive");

    return config;
}

FfmpegStreamer::FfmpegStreamer(Config config) : config_(std::move(config)) {}

FfmpegStreamer::~FfmpegStreamer() { stop(); }

void FfmpegStreamer::set_jwt_provider(std::function<std::optional<std::string>()> provider)
{
    jwt_provider_ = std::move(provider);
}

void FfmpegStreamer::set_public_camera_id(std::string camera_id)
{
    public_camera_id_ = std::move(camera_id);
}

std::string FfmpegStreamer::rtsp_url_with_camera_id() const
{
    if (public_camera_id_.empty())
        return config_.rtsp_url;

    std::string url = config_.rtsp_url;
    if (!url.empty() && url.back() != '/')
        url += '/';
    url += public_camera_id_;
    return url;
}

std::string FfmpegStreamer::resolved_rtsp_url() const
{
    const std::string base = rtsp_url_with_camera_id();

    if (!jwt_provider_)
        return base;

    const auto token = jwt_provider_();
    if (!token)
        spdlog::warn("[Stream] No JWT available, publishing to {} unauthenticated", base);

    return append_token(base, token);
}

std::vector<std::string> FfmpegStreamer::build_args() const
{
    std::vector<std::string> args = {
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-nostdin",
        // Input: an Annex-B H.264 elementary stream on stdin. It carries no
        // timestamps of its own, so ffmpeg is told the rate it was encoded at.
        "-f", "h264",
        "-framerate", std::to_string(config_.framerate),
        "-i", "-",
        // No encoding here: H264Tee already did it, once, for every consumer.
        "-c:v", "copy",
    };

    // Resolved once per spawn (not cached) so a respawn -- e.g. after MediaMTX
    // restarts -- reconnects with a fresh, unexpired token.
    args.insert(args.end(), {"-f", "rtsp", "-rtsp_transport", "tcp", resolved_rtsp_url()});
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
    // Non-blocking so write_all() can bound how long it waits for ffmpeg to
    // drain the pipe instead of blocking forever -- see kWriteTimeoutMs.
    if (::fcntl(pipe_fd_, F_SETFL, O_NONBLOCK) != 0)
        spdlog::warn("[Stream] fcntl(O_NONBLOCK) on ffmpeg pipe failed: {}", std::strerror(errno));
    child_pid_ = pid;
    spdlog::info("[Stream] ffmpeg started (pid {}): {}x{}@{} -> {}",
                 pid, config_.width, config_.height, config_.framerate, rtsp_url_with_camera_id());
    std::string cmdline;
    for (const auto &arg : args)
        cmdline += redact_token(arg) + " ";
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
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Pipe is full and ffmpeg isn't draining it -- normally brief
                // (it's a live encoder), but if ffmpeg's RTSP muxer died (e.g.
                // an auth failure) the process can sit there not reading
                // stdin at all. Wait up to kWriteTimeoutMs for it to become
                // writable again before giving up and treating this as a
                // broken pipe, same as any other write failure.
                pollfd pfd{pipe_fd_, POLLOUT, 0};
                const int r = ::poll(&pfd, 1, kWriteTimeoutMs);
                if (r > 0 && (pfd.revents & POLLOUT))
                    continue;
                if (r < 0 && errno == EINTR)
                    continue;
                if (r == 0)
                    spdlog::warn("[Stream] ffmpeg stopped draining stdin, treating pipe as broken");
                else
                    spdlog::warn("[Stream] poll() on ffmpeg pipe failed: {}", std::strerror(errno));
                return false;
            }
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

    // A fresh ffmpeg has to be handed a keyframe first, or it spends the rest
    // of the GOP complaining about references it never saw.
    request_resync();

    if (!spawn())
        return false;

    running_ = true;
    writer_ = std::thread(&FfmpegStreamer::writer_loop, this);
    return true;
}

void FfmpegStreamer::request_resync()
{
    std::lock_guard<std::mutex> lock(mtx_);
    dropped_frames_ += queue_.size();
    queue_.clear();
    queued_bytes_ = 0;
    need_keyframe_ = true;
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

bool FfmpegStreamer::write_access_unit(const uint8_t *data, size_t len, bool keyframe)
{
    if (!running_ || data == nullptr || len == 0)
        return false;

    {
        std::lock_guard<std::mutex> lock(mtx_);

        // Mid-GOP frames are useless to a decoder that has just (re)started, so
        // they are discarded until the next keyframe rather than queued.
        if (need_keyframe_) {
            if (!keyframe) {
                ++dropped_frames_;
                return false;
            }
            need_keyframe_ = false;
        }

        queue_.push_back(AccessUnit{std::vector<uint8_t>(data, data + len), keyframe});
        queued_bytes_ += len;

        // ffmpeg is not keeping up. Unlike raw frames, encoded ones can't be
        // dropped individually -- the survivors would reference frames the
        // decoder never got -- so the backlog goes and the stream picks up
        // again at the next keyframe.
        if (queued_bytes_ > kMaxQueuedBytes) {
            spdlog::warn("[Stream] {} bytes queued for ffmpeg, resyncing at the next keyframe",
                         queued_bytes_);
            dropped_frames_ += queue_.size();
            queue_.clear();
            queued_bytes_ = 0;
            need_keyframe_ = true;
            return false;
        }
    }
    cv_.notify_one();
    return true;
}

void FfmpegStreamer::writer_loop()
{
    while (running_) {
        AccessUnit unit;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this] { return !running_ || !queue_.empty(); });
            if (!running_)
                break;
            unit = std::move(queue_.front());
            queue_.pop_front();
            queued_bytes_ -= unit.data.size();
        }

        if (pipe_fd_ < 0) {
            // ffmpeg died; back off, then bring it back. The new process needs
            // to start on a keyframe, so whatever is queued (including this
            // unit) is thrown away.
            if (steady_clock::now() < next_spawn_)
                continue;
            if (!spawn()) {
                next_spawn_ = steady_clock::now() + kRestartDelay;
                continue;
            }
            request_resync();
            continue;
        }

        if (!write_all(unit.data.data(), unit.data.size())) {
            spdlog::warn("[Stream] ffmpeg pipe broken, restarting in {}s", kRestartDelay.count());
            reap();
            next_spawn_ = steady_clock::now() + kRestartDelay;
            request_resync();
        }
    }
}

}  // namespace byte_track
