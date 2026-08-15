// Copyright © 2026 Alexander Taffe

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace byte_track {

// What the backend wants this camera's RTSP stream to be doing.
enum class StreamCommand {
    // The long poll timed out with nobody asking to watch. Not an error --
    // the expected steady state -- so it just means "poll again".
    None,
    Start,
    Stop,
    // Debug aids, not stream control: turn the bounding-box overlay's WebSocket
    // server on and off while a stream is running. See BboxWsServer.
    BboxOn,
    BboxOff,
};

// Parses a long-poll response body, e.g. {"command": "start"}. The comparison
// is case-insensitive since the backend spells other enums in caps.
//
// Returns std::nullopt if the body isn't a JSON object, has no string
// "command" key, or names a command this client doesn't know -- all of which
// the poller treats as a failed poll rather than guessing.
std::optional<StreamCommand> parse_stream_command(const std::string &body);

// Name of a command, for logs.
const char *to_string(StreamCommand command);

// Long-polls the backend asking whether the RTSP stream should be running, and
// invokes the on_start / on_stop handlers when the answer changes what the
// camera should be doing.
//
// The backend holds each GET open until either a user asks to start or stop
// watching, or ~25s elapse and it answers {"command": "none"}. So a poll
// returning is the normal case, not a signal in itself: the loop simply polls
// again. Failures (no JWT, network down, a 5xx, an unparseable body) back off
// for Config::retry_delay before retrying, so a backend outage costs one
// request every few seconds rather than a hot loop.
//
// A standing command does NOT block: while a viewer is connected the backend
// answers "start" the instant it is asked, so the long poll only actually
// blocks in the idle "none" case. Config::min_poll_interval is the floor
// between requests that keeps that from becoming a back-to-back request loop
// for as long as someone is watching. It costs nothing when the poll really
// did block, and it bounds how late a "stop" is noticed -- a few seconds of
// extra streaming after the last viewer leaves.
//
// The handlers run on the poller's own thread, so they must be safe to call
// there -- FfmpegStreamer::start()/stop() are, as long as this poller is the
// only thing calling them. They are invoked on every matching command, not
// only on transitions; both are idempotent, and re-sending "start" is how the
// backend can ask a camera that silently dropped its stream to bring it back.
class StreamCommandPoller {
public:
    struct Config {
        // Same backend host the rest of the client talks to.
        std::string base_url;
        std::string command_endpoint;
        // Must comfortably exceed the backend's own long-poll timeout (~25s),
        // or every poll would abort client-side and look like an outage.
        std::chrono::seconds request_timeout{35};
        // Backoff after a failed poll.
        std::chrono::seconds retry_delay{5};
        // Floor between the start of one poll and the next. Only bites when a
        // poll returns early -- i.e. while a command is standing -- so it is
        // both the anti-hammer guard and the upper bound on how long a stale
        // "start" keeps streaming after the viewer has gone.
        std::chrono::seconds min_poll_interval{5};

        // Reads base_url from the "magpiecam-core" section of the client
        // config JSON and stream_command_endpoint from the "streaming" one --
        // the two sections those keys actually live in. The three durations may
        // be overridden with the optional "streaming" keys
        // "command_poll_timeout_seconds", "command_retry_delay_seconds" and
        // "command_min_poll_interval_seconds"; absent (the committed config
        // leaves all three out) they keep the defaults above.
        //
        // Throws std::runtime_error if the file is missing or malformed, if
        // either section or required key is absent, or if a value is the wrong
        // type or out of range -- callers decide whether that is fatal.
        static Config from_file(const std::string &path);
    };

    explicit StreamCommandPoller(Config config);
    ~StreamCommandPoller();

    StreamCommandPoller(const StreamCommandPoller &) = delete;
    StreamCommandPoller &operator=(const StreamCommandPoller &) = delete;

    // Supplies a fresh JWT before every poll. Called from the poller thread, so
    // it must be safe there (BackendClient::get_jwt_token() is). Without one,
    // or when it returns std::nullopt, the poll is skipped and retried after
    // retry_delay -- the endpoint is authenticated, so there is nothing useful
    // to send.
    void set_jwt_provider(std::function<std::optional<std::string>()> provider);

    // Invoked when the backend answers "start" / "stop". Set both before
    // start(); neither is called after stop() returns.
    void set_on_start(std::function<void()> handler);
    void set_on_stop(std::function<void()> handler);

    // Invoked when the backend answers "bbox_on" / "bbox_off". Optional: left
    // unset, those commands are parsed and logged but do nothing. Like the two
    // above they run on the poller thread and must be idempotent, since a
    // standing command is re-delivered on every poll.
    void set_on_bbox_on(std::function<void()> handler);
    void set_on_bbox_off(std::function<void()> handler);

    // Starts the polling thread. Idempotent.
    void start();

    // Signals the thread to finish and joins it. The in-flight long poll is
    // aborted rather than waited out, so this returns in about a second
    // instead of up to request_timeout. Idempotent.
    void stop();

    bool is_running() const { return running_.load(); }
    const Config &config() const { return config_; }

private:
    void poll_loop();

    // Performs one long poll. Returns the command on success, or std::nullopt
    // if the poll failed and the caller should back off.
    std::optional<StreamCommand> poll_once();

    // Sleeps up to `duration`, waking early if stop() is called. Returns false
    // if the poller should exit.
    bool wait_for(std::chrono::milliseconds duration);

    Config config_;
    std::function<std::optional<std::string>()> jwt_provider_;
    std::function<void()> on_start_;
    std::function<void()> on_stop_;
    std::function<void()> on_bbox_on_;
    std::function<void()> on_bbox_off_;

    std::thread poller_;
    std::atomic<bool> running_{false};

    std::mutex mtx_;
    std::condition_variable cv_;
};

}  // namespace byte_track
