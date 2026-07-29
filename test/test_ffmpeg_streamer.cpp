// Copyright © 2026 Alexander Taffe
//
// Unit tests for FfmpegStreamer. Spawning ffmpeg and publishing to an RTSP
// server needs a live MediaMTX (and a camera), so these tests cover the
// deterministic, process-free surface:
//   - parsing of the "streaming" section of the client config JSON,
//   - defaults for absent keys, including gop tracking framerate,
//   - error handling for missing / malformed / mistyped / out-of-range config,
//   - the frame size the streamer expects for a given resolution.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "streaming/ffmpeg_streamer.h"

namespace fs = std::filesystem;

using byte_track::FfmpegStreamer;

namespace {

// Writes config files into a per-test temp directory and cleans them up after.
class FfmpegStreamerConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("ffmpeg_streamer_test_" + std::to_string(::getpid()) + "_" +
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

// Mirrors config/backend/backend_config.json: the streaming section sits
// alongside the webservice section in the same file.
constexpr char kValidConfig[] = R"({
    "scout_cam_webservice": {
        "base_url": "http://10.0.0.125:8000"
    },
    "streaming": {
        "rtsp_url": "rtsp://10.0.0.125:8554/cam",
        "width": 1920,
        "height": 1080,
        "framerate": 25,
        "bitrate": 4000000,
        "encoder": "libx264"
    }
})";

} // namespace

// --- Config::from_file -------------------------------------------------------

TEST_F(FfmpegStreamerConfigTest, ParsesStreamingFields) {
    const auto config = FfmpegStreamer::Config::from_file(write_config(kValidConfig));
    EXPECT_EQ(config.rtsp_url, "rtsp://10.0.0.125:8554/cam");
    EXPECT_EQ(config.width, 1920);
    EXPECT_EQ(config.height, 1080);
    EXPECT_EQ(config.framerate, 25);
    EXPECT_EQ(config.bitrate, 4000000);
    EXPECT_EQ(config.encoder, "libx264");
}

TEST_F(FfmpegStreamerConfigTest, GopIsUnsetWhenAbsentSoItTracksFramerate) {
    // Left unset it resolves to one keyframe per second at whatever the frame
    // rate finally is -- which is what --intra 30 gave at 30fps.
    const auto config = FfmpegStreamer::Config::from_file(write_config(kValidConfig));
    EXPECT_FALSE(config.gop.has_value());
}

TEST_F(FfmpegStreamerConfigTest, ExplicitGopIsParsed) {
    const auto config = FfmpegStreamer::Config::from_file(
        write_config(R"({"streaming": {"framerate": 30, "gop": 60}})"));
    ASSERT_TRUE(config.gop.has_value());
    EXPECT_EQ(*config.gop, 60);
}

TEST_F(FfmpegStreamerConfigTest, AbsentKeysKeepDefaults) {
    const FfmpegStreamer::Config defaults;
    const auto config = FfmpegStreamer::Config::from_file(
        write_config(R"({"streaming": {"width": 640, "height": 480}})"));

    EXPECT_EQ(config.width, 640);
    EXPECT_EQ(config.height, 480);
    EXPECT_EQ(config.rtsp_url, defaults.rtsp_url);
    EXPECT_EQ(config.framerate, defaults.framerate);
    EXPECT_EQ(config.bitrate, defaults.bitrate);
    EXPECT_EQ(config.encoder, defaults.encoder);
}

TEST_F(FfmpegStreamerConfigTest, EmptyStreamingSectionYieldsDefaults) {
    const FfmpegStreamer::Config defaults;
    const auto config = FfmpegStreamer::Config::from_file(write_config(R"({"streaming": {}})"));

    EXPECT_EQ(config.rtsp_url, defaults.rtsp_url);
    EXPECT_EQ(config.width, defaults.width);
    EXPECT_EQ(config.height, defaults.height);
    EXPECT_EQ(config.framerate, defaults.framerate);
    EXPECT_EQ(config.bitrate, defaults.bitrate);
    EXPECT_EQ(config.encoder, defaults.encoder);
}

TEST_F(FfmpegStreamerConfigTest, MissingFileThrows) {
    EXPECT_THROW(FfmpegStreamer::Config::from_file((dir_ / "does_not_exist.json").string()),
                 std::runtime_error);
}

TEST_F(FfmpegStreamerConfigTest, MalformedJsonThrows) {
    EXPECT_THROW(FfmpegStreamer::Config::from_file(write_config("{ not valid json")),
                 std::runtime_error);
}

TEST_F(FfmpegStreamerConfigTest, MissingStreamingSectionThrows) {
    EXPECT_THROW(
        FfmpegStreamer::Config::from_file(write_config(R"({"scout_cam_webservice": {}})")),
        std::runtime_error);
}

TEST_F(FfmpegStreamerConfigTest, WrongTypeThrows) {
    EXPECT_THROW(FfmpegStreamer::Config::from_file(write_config(R"({"streaming": {"width": "wide"}})")),
                 std::runtime_error);
    EXPECT_THROW(FfmpegStreamer::Config::from_file(write_config(R"({"streaming": {"rtsp_url": 8554}})")),
                 std::runtime_error);
}

TEST_F(FfmpegStreamerConfigTest, OutOfRangeValuesThrow) {
    // H.264 and the I420 packing both need even, positive dimensions.
    EXPECT_THROW(FfmpegStreamer::Config::from_file(write_config(R"({"streaming": {"width": 1281}})")),
                 std::runtime_error);
    EXPECT_THROW(FfmpegStreamer::Config::from_file(write_config(R"({"streaming": {"height": 0}})")),
                 std::runtime_error);
    EXPECT_THROW(FfmpegStreamer::Config::from_file(write_config(R"({"streaming": {"framerate": 0}})")),
                 std::runtime_error);
    EXPECT_THROW(FfmpegStreamer::Config::from_file(write_config(R"({"streaming": {"bitrate": -1}})")),
                 std::runtime_error);
    EXPECT_THROW(FfmpegStreamer::Config::from_file(write_config(R"({"streaming": {"rtsp_url": ""}})")),
                 std::runtime_error);
    EXPECT_THROW(FfmpegStreamer::Config::from_file(write_config(R"({"streaming": {"encoder": ""}})")),
                 std::runtime_error);
    EXPECT_THROW(FfmpegStreamer::Config::from_file(write_config(R"({"streaming": {"gop": 0}})")),
                 std::runtime_error);
}

// --- frame_bytes -------------------------------------------------------------

TEST(FfmpegStreamerTest, FrameBytesMatchesI420Size) {
    FfmpegStreamer::Config config;
    config.width = 1280;
    config.height = 720;

    // Full-size luma plane plus two quarter-size chroma planes.
    const FfmpegStreamer streamer(config);
    EXPECT_EQ(streamer.frame_bytes(), 1280u * 720u * 3 / 2);
}

// --- resolved_rtsp_url --------------------------------------------------------

TEST(FfmpegStreamerTest, ResolvedUrlIsUnchangedWithoutAJwtProvider) {
    FfmpegStreamer::Config config;
    config.rtsp_url = "rtsp://10.0.0.126:8554/cam";
    const FfmpegStreamer streamer(config);
    EXPECT_EQ(streamer.resolved_rtsp_url(), "rtsp://10.0.0.126:8554/cam");
}

TEST(FfmpegStreamerTest, ResolvedUrlAppendsTokenFromProvider) {
    FfmpegStreamer::Config config;
    config.rtsp_url = "rtsp://10.0.0.126:8554/cam";
    FfmpegStreamer streamer(config);
    streamer.set_jwt_provider([]() -> std::optional<std::string> { return "abc123"; });
    EXPECT_EQ(streamer.resolved_rtsp_url(), "rtsp://10.0.0.126:8554/cam?token=abc123");
}

TEST(FfmpegStreamerTest, ResolvedUrlUsesAmpersandWhenQueryStringAlreadyPresent) {
    FfmpegStreamer::Config config;
    config.rtsp_url = "rtsp://10.0.0.126:8554/cam?foo=bar";
    FfmpegStreamer streamer(config);
    streamer.set_jwt_provider([]() -> std::optional<std::string> { return "abc123"; });
    EXPECT_EQ(streamer.resolved_rtsp_url(), "rtsp://10.0.0.126:8554/cam?foo=bar&token=abc123");
}

TEST(FfmpegStreamerTest, ResolvedUrlPercentEncodesToken) {
    FfmpegStreamer::Config config;
    config.rtsp_url = "rtsp://10.0.0.126:8554/cam";
    FfmpegStreamer streamer(config);
    streamer.set_jwt_provider([]() -> std::optional<std::string> { return "a.b+c/d=="; });
    EXPECT_EQ(streamer.resolved_rtsp_url(), "rtsp://10.0.0.126:8554/cam?token=a.b%2Bc%2Fd%3D%3D");
}

TEST(FfmpegStreamerTest, ResolvedUrlIsUnauthenticatedWhenProviderReturnsNullopt) {
    FfmpegStreamer::Config config;
    config.rtsp_url = "rtsp://10.0.0.126:8554/cam";
    FfmpegStreamer streamer(config);
    streamer.set_jwt_provider([]() -> std::optional<std::string> { return std::nullopt; });
    EXPECT_EQ(streamer.resolved_rtsp_url(), "rtsp://10.0.0.126:8554/cam");
}

TEST(FfmpegStreamerTest, ResolvedUrlCallsProviderEveryTime) {
    // Each (re)spawn of ffmpeg should get a fresh token rather than a cached
    // one, since a respawn is a new RTSP connection MediaMTX re-validates.
    FfmpegStreamer::Config config;
    config.rtsp_url = "rtsp://10.0.0.126:8554/cam";
    FfmpegStreamer streamer(config);
    int calls = 0;
    streamer.set_jwt_provider([&calls]() -> std::optional<std::string> {
        return "token-" + std::to_string(++calls);
    });
    EXPECT_EQ(streamer.resolved_rtsp_url(), "rtsp://10.0.0.126:8554/cam?token=token-1");
    EXPECT_EQ(streamer.resolved_rtsp_url(), "rtsp://10.0.0.126:8554/cam?token=token-2");
}
