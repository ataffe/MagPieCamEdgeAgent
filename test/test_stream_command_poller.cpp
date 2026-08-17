
//
// Unit tests for StreamCommandPoller. The poll loop itself needs a live
// backend holding a request open, so these tests cover the deterministic,
// network-free surface:
//   - parsing of the long-poll response body, including the unhappy paths the
//     poller deliberately treats as a failed poll rather than a guess,
//   - parsing of the stream command settings, which are split across two
//     sections of the client config JSON,
//   - defaults for the optional timeout keys the committed config leaves out,
//   - error handling for missing / malformed / mistyped / out-of-range config.

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "streaming/stream_command_poller.h"

namespace fs = std::filesystem;

using byte_track::StreamCommand;
using byte_track::StreamCommandPoller;
using byte_track::parse_stream_command;

namespace {

// Writes config files into a per-test temp directory and cleans them up after.
class StreamCommandPollerConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("stream_command_poller_test_" + std::to_string(::getpid()) + "_" +
                ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::create_directories(dir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    // Writes `contents` to <dir>/config.json and returns the full path.
    std::string write_config(const std::string &contents) {
        const fs::path path = dir_ / "config.json";
        std::ofstream out(path);
        out << contents;
        out.close();
        return path.string();
    }

    fs::path dir_;
};

// Mirrors config/backend/backend_config.json: base_url lives in the webservice
// section, the command endpoint in the streaming one.
constexpr char kValidConfig[] = R"({
    "magpiecam-core": {
        "base_url": "http://10.0.0.126:8000"
    },
    "streaming": {
        "rtsp_url": "rtsp://10.0.0.126:8554",
        "stream_command_endpoint": "/v1/cameras/streaming/command"
    }
})";

} // namespace

// --- parse_stream_command ----------------------------------------------------

TEST(ParseStreamCommandTest, ParsesTheKnownCommands) {
    EXPECT_EQ(parse_stream_command(R"({"command": "start"})"), StreamCommand::Start);
    EXPECT_EQ(parse_stream_command(R"({"command": "stop"})"), StreamCommand::Stop);
    EXPECT_EQ(parse_stream_command(R"({"command": "none"})"), StreamCommand::None);
}

TEST(ParseStreamCommandTest, ParsesTheBboxDebugCommands) {
    EXPECT_EQ(parse_stream_command(R"({"command": "bbox_on"})"), StreamCommand::BboxOn);
    EXPECT_EQ(parse_stream_command(R"({"command": "bbox_off"})"), StreamCommand::BboxOff);
}

TEST(ParseStreamCommandTest, IsCaseInsensitive) {
    // The backend spells its other enums in caps (DETECTION, CAMERA_PREVIEW),
    // so don't let a change of case silently stop the stream from starting.
    EXPECT_EQ(parse_stream_command(R"({"command": "START"})"), StreamCommand::Start);
    EXPECT_EQ(parse_stream_command(R"({"command": "Stop"})"), StreamCommand::Stop);
    EXPECT_EQ(parse_stream_command(R"({"command": "BBOX_ON"})"), StreamCommand::BboxOn);
    EXPECT_EQ(parse_stream_command(R"({"command": "Bbox_Off"})"), StreamCommand::BboxOff);
}

TEST(ParseStreamCommandTest, CommandNamesRoundTripThroughToString) {
    for (const auto command : {StreamCommand::Start, StreamCommand::Stop, StreamCommand::None,
                               StreamCommand::BboxOn, StreamCommand::BboxOff}) {
        const std::string body = std::string(R"({"command": ")") +
                                 byte_track::to_string(command) + R"("})";
        EXPECT_EQ(parse_stream_command(body), command) << body;
    }
}

TEST(ParseStreamCommandTest, NearMissesOfTheBboxCommandsAreRejected) {
    // A hyphen or a space instead of the underscore is the likely typo, and it
    // must not be mistaken for a command this build understands.
    EXPECT_FALSE(parse_stream_command(R"({"command": "bbox-on"})").has_value());
    EXPECT_FALSE(parse_stream_command(R"({"command": "bbox on"})").has_value());
    EXPECT_FALSE(parse_stream_command(R"({"command": "bbox"})").has_value());
}

TEST(ParseStreamCommandTest, IgnoresOtherKeys) {
    EXPECT_EQ(parse_stream_command(R"({"command": "start", "requested_by": "user-1"})"),
              StreamCommand::Start);
}

TEST(ParseStreamCommandTest, UnknownCommandIsRejected) {
    // Not silently treated as "none": a command this build doesn't understand
    // is a mismatch worth retrying and logging, not a reason to sit idle.
    EXPECT_FALSE(parse_stream_command(R"({"command": "pause"})").has_value());
    EXPECT_FALSE(parse_stream_command(R"({"command": ""})").has_value());
}

TEST(ParseStreamCommandTest, MalformedOrUnexpectedBodyIsRejected) {
    EXPECT_FALSE(parse_stream_command("").has_value());
    EXPECT_FALSE(parse_stream_command("{ not valid json").has_value());
    EXPECT_FALSE(parse_stream_command("<html>502 Bad Gateway</html>").has_value());
    // Valid JSON, but not the object shape the endpoint documents.
    EXPECT_FALSE(parse_stream_command(R"(["start"])").has_value());
    EXPECT_FALSE(parse_stream_command(R"("start")").has_value());
}

TEST(ParseStreamCommandTest, MissingOrMistypedCommandKeyIsRejected) {
    EXPECT_FALSE(parse_stream_command(R"({})").has_value());
    EXPECT_FALSE(parse_stream_command(R"({"cmd": "start"})").has_value());
    EXPECT_FALSE(parse_stream_command(R"({"command": 1})").has_value());
    EXPECT_FALSE(parse_stream_command(R"({"command": null})").has_value());
}

// --- Config::from_file -------------------------------------------------------

TEST_F(StreamCommandPollerConfigTest, ParsesFieldsFromBothSections) {
    const auto config = StreamCommandPoller::Config::from_file(write_config(kValidConfig));
    EXPECT_EQ(config.base_url, "http://10.0.0.126:8000");
    EXPECT_EQ(config.command_endpoint, "/v1/cameras/streaming/command");
}

TEST_F(StreamCommandPollerConfigTest, TimeoutsDefaultWhenAbsent) {
    // The committed config leaves all three keys out; the client timeout has to
    // stay comfortably above the backend's ~25s long-poll timeout, and the poll
    // floor well below it so an idle "none" poll is never the thing delayed.
    const auto config = StreamCommandPoller::Config::from_file(write_config(kValidConfig));
    EXPECT_GT(config.request_timeout, std::chrono::seconds(25));
    EXPECT_EQ(config.retry_delay, std::chrono::seconds(5));
    EXPECT_EQ(config.min_poll_interval, std::chrono::seconds(5));
    EXPECT_LT(config.min_poll_interval, std::chrono::seconds(25));
}

TEST_F(StreamCommandPollerConfigTest, ExplicitTimeoutsAreParsed) {
    const auto config = StreamCommandPoller::Config::from_file(write_config(R"({
        "magpiecam-core": {"base_url": "http://host"},
        "streaming": {
            "stream_command_endpoint": "/cmd",
            "command_poll_timeout_seconds": 60,
            "command_retry_delay_seconds": 15,
            "command_min_poll_interval_seconds": 20
        }
    })"));
    EXPECT_EQ(config.request_timeout, std::chrono::seconds(60));
    EXPECT_EQ(config.retry_delay, std::chrono::seconds(15));
    EXPECT_EQ(config.min_poll_interval, std::chrono::seconds(20));
}

TEST_F(StreamCommandPollerConfigTest, MissingFileThrows) {
    EXPECT_THROW(StreamCommandPoller::Config::from_file((dir_ / "does_not_exist.json").string()),
                 std::runtime_error);
}

TEST_F(StreamCommandPollerConfigTest, MalformedJsonThrows) {
    EXPECT_THROW(StreamCommandPoller::Config::from_file(write_config("{ not valid json")),
                 std::runtime_error);
}

TEST_F(StreamCommandPollerConfigTest, MissingSectionThrows) {
    EXPECT_THROW(StreamCommandPoller::Config::from_file(
                     write_config(R"({"streaming": {"stream_command_endpoint": "/cmd"}})")),
                 std::runtime_error);
    EXPECT_THROW(StreamCommandPoller::Config::from_file(
                     write_config(R"({"magpiecam-core": {"base_url": "http://host"}})")),
                 std::runtime_error);
}

TEST_F(StreamCommandPollerConfigTest, MissingRequiredKeyThrows) {
    EXPECT_THROW(StreamCommandPoller::Config::from_file(write_config(R"({
                     "magpiecam-core": {},
                     "streaming": {"stream_command_endpoint": "/cmd"}
                 })")),
                 std::runtime_error);
    EXPECT_THROW(StreamCommandPoller::Config::from_file(write_config(R"({
                     "magpiecam-core": {"base_url": "http://host"},
                     "streaming": {}
                 })")),
                 std::runtime_error);
}

TEST_F(StreamCommandPollerConfigTest, WrongTypeThrows) {
    EXPECT_THROW(StreamCommandPoller::Config::from_file(write_config(R"({
                     "magpiecam-core": {"base_url": 8000},
                     "streaming": {"stream_command_endpoint": "/cmd"}
                 })")),
                 std::runtime_error);
    EXPECT_THROW(StreamCommandPoller::Config::from_file(write_config(R"({
                     "magpiecam-core": {"base_url": "http://host"},
                     "streaming": {"stream_command_endpoint": "/cmd",
                                   "command_poll_timeout_seconds": "soon"}
                 })")),
                 std::runtime_error);
}

TEST_F(StreamCommandPollerConfigTest, OutOfRangeValuesThrow) {
    const auto with_streaming = [](const std::string &streaming) {
        return R"({"magpiecam-core": {"base_url": "http://host"}, "streaming": )" +
               streaming + "}";
    };
    EXPECT_THROW(StreamCommandPoller::Config::from_file(
                     write_config(with_streaming(R"({"stream_command_endpoint": ""})"))),
                 std::runtime_error);
    EXPECT_THROW(StreamCommandPoller::Config::from_file(write_config(with_streaming(
                     R"({"stream_command_endpoint": "/cmd", "command_poll_timeout_seconds": 0})"))),
                 std::runtime_error);
    EXPECT_THROW(StreamCommandPoller::Config::from_file(write_config(with_streaming(
                     R"({"stream_command_endpoint": "/cmd", "command_retry_delay_seconds": -1})"))),
                 std::runtime_error);
    // Zero would turn a standing "start" back into a back-to-back request loop.
    EXPECT_THROW(StreamCommandPoller::Config::from_file(write_config(with_streaming(
                     R"({"stream_command_endpoint": "/cmd", "command_min_poll_interval_seconds": 0})"))),
                 std::runtime_error);
    EXPECT_THROW(StreamCommandPoller::Config::from_file(write_config(
                     R"({"magpiecam-core": {"base_url": ""},
                         "streaming": {"stream_command_endpoint": "/cmd"}})")),
                 std::runtime_error);
}

// --- lifecycle ---------------------------------------------------------------

TEST(StreamCommandPollerTest, StopWithoutStartIsANoop) {
    StreamCommandPoller::Config config;
    config.base_url = "http://127.0.0.1:1";
    config.command_endpoint = "/cmd";

    StreamCommandPoller poller(config);
    EXPECT_FALSE(poller.is_running());
    poller.stop();
    EXPECT_FALSE(poller.is_running());
}

TEST(StreamCommandPollerTest, StartThenStopJoinsPromptly) {
    // Points at a closed port so each poll fails fast and the loop spends its
    // time in the interruptible backoff wait -- stop() has to cut that short
    // rather than sit through the full retry delay.
    StreamCommandPoller::Config config;
    config.base_url = "http://127.0.0.1:1";
    config.command_endpoint = "/cmd";
    config.retry_delay = std::chrono::seconds(30);

    StreamCommandPoller poller(config);
    poller.set_jwt_provider([]() -> std::optional<std::string> { return "test-token"; });

    const auto begin = std::chrono::steady_clock::now();
    poller.start();
    EXPECT_TRUE(poller.is_running());
    poller.stop();
    const auto elapsed = std::chrono::steady_clock::now() - begin;

    EXPECT_FALSE(poller.is_running());
    EXPECT_LT(elapsed, config.retry_delay);
}
