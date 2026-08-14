// Copyright © 2026 Alexander Taffe

#include "streaming/stream_command_poller.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace byte_track {

using namespace std::chrono;
using json = nlohmann::json;

namespace {
// Sections of the client config file the poller's settings are split across.
constexpr char kWebserviceSection[] = "magpiecam-core";
constexpr char kStreamingSection[] = "streaming";

std::string to_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}
}  // namespace

std::optional<StreamCommand> parse_stream_command(const std::string &body)
{
    json doc;
    try {
        doc = json::parse(body);
    } catch (const json::parse_error &) {
        return std::nullopt;
    }

    if (!doc.is_object())
        return std::nullopt;

    const auto it = doc.find("command");
    if (it == doc.end() || !it->is_string())
        return std::nullopt;

    const std::string command = to_lower(it->get<std::string>());
    if (command == "start")
        return StreamCommand::Start;
    if (command == "stop")
        return StreamCommand::Stop;
    if (command == "none")
        return StreamCommand::None;
    return std::nullopt;
}

const char *to_string(StreamCommand command)
{
    switch (command) {
        case StreamCommand::Start: return "start";
        case StreamCommand::Stop:  return "stop";
        case StreamCommand::None:  return "none";
    }
    return "unknown";
}

StreamCommandPoller::Config StreamCommandPoller::Config::from_file(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("StreamCommandPoller: could not open config file: " + path);

    json doc;
    try {
        doc = json::parse(in);
    } catch (const json::parse_error &e) {
        throw std::runtime_error("StreamCommandPoller: failed to parse " + path + ": " + e.what());
    }

    const auto require_section = [&doc, &path](const char *name) {
        const auto it = doc.find(name);
        if (it == doc.end())
            throw std::runtime_error("StreamCommandPoller: no \"" + std::string(name) +
                                     "\" section in " + path);
        return it;
    };
    const auto webservice = require_section(kWebserviceSection);
    const auto streaming = require_section(kStreamingSection);

    Config config;
    try {
        config.base_url = webservice->at("base_url").get<std::string>();
        config.command_endpoint = streaming->at("stream_command_endpoint").get<std::string>();
        // Absent from the committed config: both keep the defaults above.
        if (const auto it = streaming->find("command_poll_timeout_seconds"); it != streaming->end())
            config.request_timeout = seconds(it->get<int>());
        if (const auto it = streaming->find("command_retry_delay_seconds"); it != streaming->end())
            config.retry_delay = seconds(it->get<int>());
        if (const auto it = streaming->find("command_min_poll_interval_seconds"); it != streaming->end())
            config.min_poll_interval = seconds(it->get<int>());
    } catch (const json::exception &e) {
        throw std::runtime_error("StreamCommandPoller: invalid stream command settings in " + path +
                                 ": " + e.what());
    }

    const auto require = [&path](bool ok, const char *what) {
        if (!ok)
            throw std::runtime_error("StreamCommandPoller: invalid stream command settings in " +
                                     path + ": " + what);
    };
    require(!config.base_url.empty(), "base_url must not be empty");
    require(!config.command_endpoint.empty(), "stream_command_endpoint must not be empty");
    require(config.request_timeout.count() > 0, "command_poll_timeout_seconds must be positive");
    require(config.retry_delay.count() > 0, "command_retry_delay_seconds must be positive");
    require(config.min_poll_interval.count() > 0, "command_min_poll_interval_seconds must be positive");

    return config;
}

StreamCommandPoller::StreamCommandPoller(Config config) : config_(std::move(config)) {}

StreamCommandPoller::~StreamCommandPoller() { stop(); }

void StreamCommandPoller::set_jwt_provider(std::function<std::optional<std::string>()> provider)
{
    jwt_provider_ = std::move(provider);
}

void StreamCommandPoller::set_on_start(std::function<void()> handler)
{
    on_start_ = std::move(handler);
}

void StreamCommandPoller::set_on_stop(std::function<void()> handler)
{
    on_stop_ = std::move(handler);
}

void StreamCommandPoller::start()
{
    if (running_.exchange(true))
        return;

    spdlog::info("[StreamCmd] Polling {}{} for stream commands",
                 config_.base_url, config_.command_endpoint);
    poller_ = std::thread(&StreamCommandPoller::poll_loop, this);
}

void StreamCommandPoller::stop()
{
    if (!running_.exchange(false))
        return;

    // Wakes the backoff wait; the in-flight request is aborted by the progress
    // callback in poll_once(), which polls running_ about once a second.
    cv_.notify_all();
    if (poller_.joinable())
        poller_.join();
    spdlog::debug("[StreamCmd] Poller stopped");
}

bool StreamCommandPoller::wait_for(milliseconds duration)
{
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait_for(lock, duration, [this] { return !running_; });
    return running_;
}

std::optional<StreamCommand> StreamCommandPoller::poll_once()
{
    if (!jwt_provider_) {
        spdlog::error("[StreamCmd] No JWT provider set; cannot poll for stream commands");
        return std::nullopt;
    }

    const auto token = jwt_provider_();
    if (!token) {
        spdlog::warn("[StreamCmd] No JWT available, retrying in {}s", config_.retry_delay.count());
        return std::nullopt;
    }

    const cpr::Response response = cpr::Get(
        cpr::Url{config_.base_url + config_.command_endpoint},
        cpr::Header{{"Authorization", "Bearer " + *token}},
        cpr::Timeout{config_.request_timeout},
        // Returning false aborts the transfer. libcurl calls this roughly once
        // a second even while the long poll is idle, so stop() doesn't have to
        // wait out the remaining timeout.
        cpr::ProgressCallback{[this](cpr::cpr_pf_arg_t, cpr::cpr_pf_arg_t, cpr::cpr_pf_arg_t,
                                     cpr::cpr_pf_arg_t, intptr_t) { return running_.load(); }});

    // A shutdown-aborted request isn't a failure worth logging.
    if (!running_)
        return std::nullopt;

    if (response.error.code != cpr::ErrorCode::OK) {
        spdlog::warn("[StreamCmd] Poll failed: {}, retrying in {}s",
                     response.error.message, config_.retry_delay.count());
        return std::nullopt;
    }

    if (response.status_code != 200) {
        spdlog::warn("[StreamCmd] Poll returned HTTP {}, retrying in {}s",
                     response.status_code, config_.retry_delay.count());
        return std::nullopt;
    }

    const auto command = parse_stream_command(response.text);
    if (!command) {
        spdlog::warn("[StreamCmd] Unrecognized poll response \"{}\", retrying in {}s",
                     response.text, config_.retry_delay.count());
        return std::nullopt;
    }

    return command;
}

void StreamCommandPoller::poll_loop()
{
    // What the last successful poll asked for, so a command that is merely
    // still in force can be told apart from a newly issued one.
    std::optional<StreamCommand> last_command;

    while (running_) {
        const auto polled_at = steady_clock::now();
        const auto command = poll_once();
        if (!running_)
            break;

        if (!command) {
            if (!wait_for(config_.retry_delay))
                break;
            continue;
        }

        if (*command == StreamCommand::None) {
            // The long poll timed out: nobody is watching.
            spdlog::debug("[StreamCmd] No stream requested");
        } else {
            // A standing command comes back on every poll, so only the first
            // sighting is news. The handler still runs each time -- both are
            // idempotent, and re-running start() is what brings the stream
            // back if ffmpeg failed to launch or died for good.
            if (last_command != command)
                spdlog::info("[StreamCmd] Backend requested \"{}\"", to_string(*command));
            else
                spdlog::debug("[StreamCmd] Still \"{}\"", to_string(*command));

            if (*command == StreamCommand::Start) {
                if (on_start_)
                    on_start_();
            } else if (on_stop_) {
                on_stop_();
            }
        }
        last_command = command;

        // Only bites when the backend answered early, which it does for as
        // long as a command stands -- without this the loop would spin on
        // back-to-back requests the whole time someone is watching. After a
        // real long-poll timeout the interval has already elapsed and this
        // adds no delay.
        const auto elapsed = steady_clock::now() - polled_at;
        if (elapsed < config_.min_poll_interval &&
            !wait_for(duration_cast<milliseconds>(config_.min_poll_interval - elapsed)))
            break;
    }
}

}  // namespace byte_track
