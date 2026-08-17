
//
// Unit tests for VideoClipRecorder. A real clip needs a camera, the Pi's H.264
// encoder and a backend, so these tests cover the deterministic surface:
//   - parsing of the "video_clips" section of the client config JSON,
//   - the ring buffer: keyframe-aligned starts and trimming to the pre-event
//     window,
//   - the cooldown, which is what stops overlapping detections producing
//     near-duplicate clips.
//
// The recorder measures everything in encoder timestamps rather than wall-clock
// time, so a test can feed it a synthetic stream and step through minutes of
// footage instantly. The muxer is replaced with an identity function, so an
// "uploaded clip" is just the concatenated access units -- with one byte per
// frame carrying that frame's index, a clip is literally the list of frames it
// contains.

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "streaming/video_clip_recorder.h"

namespace fs = std::filesystem;

using byte_track::VideoClipRecorder;

namespace {

// Writes config files into a per-test temp directory and cleans them up after.
class VideoClipRecorderConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("video_clip_recorder_test_" + std::to_string(::getpid()) + "_" +
                ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::create_directories(dir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    std::string write_config(const std::string &contents) {
        const fs::path path = dir_ / "config.json";
        std::ofstream out(path);
        out << contents;
        out.close();
        return path.string();
    }

    fs::path dir_;
};

// Mirrors config/backend/backend_config.json: the clip settings sit alongside
// the streaming section, and borrow the frame rate from it.
constexpr char kValidConfig[] = R"({
    "streaming": {
        "framerate": 25
    },
    "video_clips": {
        "pre_event_length_seconds": 8,
        "post_event_length_seconds": 6,
        "send_video_clip_cooldown": 45,
        "work_dir": "/var/tmp",
        "upload_type": "VIDEO_CLIP",
        "content_type": "video/mp4"
    }
})";

} // namespace

// --- Config::from_file -------------------------------------------------------

TEST_F(VideoClipRecorderConfigTest, ParsesVideoClipFields) {
    const auto config = VideoClipRecorder::Config::from_file(write_config(kValidConfig));
    EXPECT_EQ(config.pre_event_length_seconds, 8);
    EXPECT_EQ(config.post_event_length_seconds, 6);
    EXPECT_EQ(config.send_video_clip_cooldown, 45);
    EXPECT_EQ(config.work_dir, "/var/tmp");
    EXPECT_EQ(config.upload_type, "VIDEO_CLIP");
    EXPECT_EQ(config.content_type, "video/mp4");
}

TEST_F(VideoClipRecorderConfigTest, FramerateComesFromTheStreamingSection) {
    // It describes the camera, not the clips, so it is not duplicated into the
    // clip section -- the muxed MP4 has to use the rate the frames arrived at.
    const auto config = VideoClipRecorder::Config::from_file(write_config(kValidConfig));
    EXPECT_EQ(config.framerate, 25);
}

TEST_F(VideoClipRecorderConfigTest, FramerateFallsBackToTheDefaultWithoutAStreamingSection) {
    const VideoClipRecorder::Config defaults;
    const auto config = VideoClipRecorder::Config::from_file(
        write_config(R"({"video_clips": {"pre_event_length_seconds": 5}})"));
    EXPECT_EQ(config.framerate, defaults.framerate);
}

TEST_F(VideoClipRecorderConfigTest, AbsentKeysKeepDefaults) {
    const VideoClipRecorder::Config defaults;
    const auto config = VideoClipRecorder::Config::from_file(
        write_config(R"({"video_clips": {"send_video_clip_cooldown": 90}})"));

    EXPECT_EQ(config.send_video_clip_cooldown, 90);
    EXPECT_EQ(config.pre_event_length_seconds, defaults.pre_event_length_seconds);
    EXPECT_EQ(config.post_event_length_seconds, defaults.post_event_length_seconds);
    EXPECT_EQ(config.work_dir, defaults.work_dir);
    EXPECT_EQ(config.upload_type, defaults.upload_type);
    EXPECT_EQ(config.content_type, defaults.content_type);
}

TEST_F(VideoClipRecorderConfigTest, EmptyVideoClipsSectionYieldsDefaults) {
    const VideoClipRecorder::Config defaults;
    const auto config = VideoClipRecorder::Config::from_file(write_config(R"({"video_clips": {}})"));

    EXPECT_EQ(config.pre_event_length_seconds, defaults.pre_event_length_seconds);
    EXPECT_EQ(config.post_event_length_seconds, defaults.post_event_length_seconds);
    EXPECT_EQ(config.send_video_clip_cooldown, defaults.send_video_clip_cooldown);
}

TEST_F(VideoClipRecorderConfigTest, MissingFileThrows) {
    EXPECT_THROW(VideoClipRecorder::Config::from_file((dir_ / "does_not_exist.json").string()),
                 std::runtime_error);
}

TEST_F(VideoClipRecorderConfigTest, MalformedJsonThrows) {
    EXPECT_THROW(VideoClipRecorder::Config::from_file(write_config("{ not valid json")),
                 std::runtime_error);
}

TEST_F(VideoClipRecorderConfigTest, MissingVideoClipsSectionThrows) {
    EXPECT_THROW(VideoClipRecorder::Config::from_file(write_config(R"({"streaming": {}})")),
                 std::runtime_error);
}

TEST_F(VideoClipRecorderConfigTest, WrongTypeThrows) {
    EXPECT_THROW(VideoClipRecorder::Config::from_file(
                     write_config(R"({"video_clips": {"pre_event_length_seconds": "ten"}})")),
                 std::runtime_error);
    EXPECT_THROW(VideoClipRecorder::Config::from_file(
                     write_config(R"({"video_clips": {"upload_type": 7}})")),
                 std::runtime_error);
}

TEST_F(VideoClipRecorderConfigTest, OutOfRangeValuesThrow) {
    EXPECT_THROW(VideoClipRecorder::Config::from_file(
                     write_config(R"({"video_clips": {"pre_event_length_seconds": 0}})")),
                 std::runtime_error);
    EXPECT_THROW(VideoClipRecorder::Config::from_file(
                     write_config(R"({"video_clips": {"post_event_length_seconds": -1}})")),
                 std::runtime_error);
    EXPECT_THROW(VideoClipRecorder::Config::from_file(
                     write_config(R"({"video_clips": {"send_video_clip_cooldown": -1}})")),
                 std::runtime_error);
    EXPECT_THROW(VideoClipRecorder::Config::from_file(
                     write_config(R"({"video_clips": {"max_buffer_bytes": 0}})")),
                 std::runtime_error);
    EXPECT_THROW(VideoClipRecorder::Config::from_file(
                     write_config(R"({"video_clips": {"work_dir": ""}})")),
                 std::runtime_error);
    EXPECT_THROW(VideoClipRecorder::Config::from_file(
                     write_config(R"({"video_clips": {"upload_type": ""}})")),
                 std::runtime_error);
    EXPECT_THROW(VideoClipRecorder::Config::from_file(
                     write_config(R"({"streaming": {"framerate": 0}, "video_clips": {}})")),
                 std::runtime_error);
}

// A cooldown shorter than pre + post is legal (it only means consecutive clips
// overlap), so it warns rather than throwing.
TEST_F(VideoClipRecorderConfigTest, ShortCooldownIsAcceptedNotRejected) {
    const auto config = VideoClipRecorder::Config::from_file(write_config(R"({"video_clips": {
        "pre_event_length_seconds": 10,
        "post_event_length_seconds": 10,
        "send_video_clip_cooldown": 5
    }})"));
    EXPECT_EQ(config.send_video_clip_cooldown, 5);
}

// --- Buffering, triggering and the cooldown ----------------------------------

namespace {

// Drives a recorder with a synthetic encoded stream: one byte per frame holding
// that frame's index, a keyframe every kGop frames, at kFramerate fps.
class VideoClipRecorderTest : public ::testing::Test {
protected:
    static constexpr int kFramerate = 10;
    static constexpr int kGop = 10;  // one keyframe per second
    static constexpr int64_t kFrameIntervalUs = 1000000 / kFramerate;
    // Stands in for the uuid parsed out of a detection image's storage key.
    static constexpr char kDetectionKey[] = "detection-key-1";

    // A clip as the uploader saw it: the muxed bytes and the detection image it
    // is keyed to.
    struct Upload {
        std::vector<uint8_t> data;
        std::string detection_key;
    };

    void SetUp() override {
        config_.pre_event_length_seconds = 2;
        config_.post_event_length_seconds = 2;
        config_.send_video_clip_cooldown = 10;
        config_.framerate = kFramerate;
    }

    void TearDown() override {
        if (recorder_)
            recorder_->stop();
    }

    void build() {
        recorder_ = std::make_unique<VideoClipRecorder>(config_);
        // Identity "mux": the clip that comes out is exactly the access units
        // that went in, so a test can read the frame indices straight off it.
        recorder_->set_muxer([](const std::vector<uint8_t> &annexb) {
            return std::optional<std::vector<uint8_t>>(annexb);
        });
        recorder_->set_uploader([this](const std::vector<uint8_t> &clip,
                                       const std::string &detection_key) {
            std::lock_guard<std::mutex> lock(mtx_);
            uploads_.push_back(Upload{clip, detection_key});
            cv_.notify_all();
            return true;
        });
        recorder_->start();
    }

    // Feeds `count` frames, continuing from wherever the last call stopped.
    void feed(int count) {
        for (int i = 0; i < count; ++i) {
            const uint8_t payload = static_cast<uint8_t>(next_frame_ & 0xff);
            recorder_->on_encoded_frame(&payload, 1, next_frame_ * kFrameIntervalUs,
                                        (next_frame_ % kGop) == 0);
            ++next_frame_;
        }
    }

    // Feeds frames one at a time until an event can be armed, so a test can get
    // past a cooldown without hard-coding how many frames that takes.
    void feed_until_event_arms(int max_frames, const std::string &detection_key = kDetectionKey) {
        for (int i = 0; i < max_frames; ++i) {
            feed(1);
            if (recorder_->on_event(detection_key))
                return;
        }
        FAIL() << "no event armed within " << max_frames << " frames";
    }

    bool wait_for_uploads(size_t count) {
        std::unique_lock<std::mutex> lock(mtx_);
        return cv_.wait_for(lock, std::chrono::seconds(5),
                            [this, count] { return uploads_.size() >= count; });
    }

    std::vector<Upload> uploads() {
        std::lock_guard<std::mutex> lock(mtx_);
        return uploads_;
    }

    VideoClipRecorder::Config config_;
    std::unique_ptr<VideoClipRecorder> recorder_;
    int64_t next_frame_ = 0;

    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<Upload> uploads_;
};

} // namespace

TEST_F(VideoClipRecorderTest, FramesBeforeTheFirstKeyframeAreDiscarded) {
    build();
    // A clip cannot start mid-GOP, so nothing is worth keeping until the first
    // keyframe arrives.
    const uint8_t payload = 0;
    for (int i = 0; i < 20; ++i)
        recorder_->on_encoded_frame(&payload, 1, i * kFrameIntervalUs, /*keyframe=*/false);
    EXPECT_EQ(recorder_->buffered_frames(), 0u);

    recorder_->on_encoded_frame(&payload, 1, 20 * kFrameIntervalUs, /*keyframe=*/true);
    EXPECT_EQ(recorder_->buffered_frames(), 1u);
}

TEST_F(VideoClipRecorderTest, BufferIsTrimmedToRoughlyThePreEventWindow) {
    build();
    feed(20 * kFramerate);  // 20s of footage into a 2s window

    // At least the window itself, and at most one extra GOP -- the buffer can
    // only be cut back as far as a keyframe.
    const size_t window_frames = static_cast<size_t>(config_.pre_event_length_seconds) * kFramerate;
    EXPECT_GE(recorder_->buffered_frames(), window_frames);
    EXPECT_LE(recorder_->buffered_frames(), window_frames + kGop);
}

TEST_F(VideoClipRecorderTest, EventWithNothingBufferedIsIgnored) {
    build();
    EXPECT_FALSE(recorder_->on_event(kDetectionKey));
    EXPECT_EQ(recorder_->clips_captured(), 0u);
}

TEST_F(VideoClipRecorderTest, ClipCoversFootageBeforeAndAfterTheEvent) {
    build();
    feed(5 * kFramerate);  // frames 0..49
    ASSERT_TRUE(recorder_->on_event(kDetectionKey));
    EXPECT_TRUE(recorder_->is_capturing());

    feed(3 * kFramerate);  // frames 50..79; the post window closes at frame 69
    ASSERT_TRUE(wait_for_uploads(1));

    const auto clip = uploads().at(0).data;
    // Each byte is a frame index, so the clip's first and last bytes say
    // exactly which footage it covers.
    const int trigger = 5 * kFramerate - 1;  // last frame fed before the event
    EXPECT_LE(clip.front(), trigger - config_.pre_event_length_seconds * kFramerate);
    EXPECT_GE(clip.back(), trigger + config_.post_event_length_seconds * kFramerate);
    // And it has to be independently decodable from its first frame.
    EXPECT_EQ(clip.front() % kGop, 0);
    EXPECT_FALSE(recorder_->is_capturing());
}

TEST_F(VideoClipRecorderTest, ClipIsNotUploadedUntilThePostEventWindowCloses) {
    build();
    feed(5 * kFramerate);
    ASSERT_TRUE(recorder_->on_event(kDetectionKey));

    feed(config_.post_event_length_seconds * kFramerate - 5);  // just short of the window
    EXPECT_TRUE(recorder_->is_capturing());
    EXPECT_TRUE(uploads().empty());

    feed(10);
    ASSERT_TRUE(wait_for_uploads(1));
}

TEST_F(VideoClipRecorderTest, EventDuringACaptureIsAbsorbed) {
    build();
    feed(5 * kFramerate);
    ASSERT_TRUE(recorder_->on_event(kDetectionKey));

    // The event 0.5s later is already inside the clip being recorded, so it
    // must not start a second, near-identical one.
    feed(5);
    EXPECT_FALSE(recorder_->on_event("detection-key-2"));
    EXPECT_EQ(recorder_->clips_captured(), 1u);
    EXPECT_EQ(recorder_->events_suppressed(), 1u);

    feed(5 * kFramerate);
    ASSERT_TRUE(wait_for_uploads(1));
    EXPECT_EQ(uploads().size(), 1u);
}

TEST_F(VideoClipRecorderTest, EventInsideTheCooldownIsAbsorbed) {
    build();
    feed(5 * kFramerate);
    ASSERT_TRUE(recorder_->on_event(kDetectionKey));
    feed(3 * kFramerate);
    ASSERT_TRUE(wait_for_uploads(1));

    // The capture is over, but the previous clip still covers this stretch of
    // footage: 10s of cooldown from a trigger only ~3s ago.
    feed(2 * kFramerate);
    EXPECT_FALSE(recorder_->on_event("detection-key-2"));
    EXPECT_EQ(uploads().size(), 1u);
}

TEST_F(VideoClipRecorderTest, EventAfterTheCooldownStartsANewClip) {
    build();
    feed(5 * kFramerate);
    ASSERT_TRUE(recorder_->on_event(kDetectionKey));
    feed(3 * kFramerate);
    ASSERT_TRUE(wait_for_uploads(1));

    // Once the cooldown has elapsed the next detection is a genuinely new
    // event, not a duplicate of the last one.
    feed_until_event_arms(config_.send_video_clip_cooldown * kFramerate + kGop);
    feed(3 * kFramerate);
    ASSERT_TRUE(wait_for_uploads(2));
    EXPECT_EQ(recorder_->clips_captured(), 2u);
}

TEST_F(VideoClipRecorderTest, ClipIsUploadedUnderTheDetectionKeyThatArmedIt) {
    build();
    feed(5 * kFramerate);
    ASSERT_TRUE(recorder_->on_event("first-detection"));
    feed(3 * kFramerate);
    ASSERT_TRUE(wait_for_uploads(1));
    EXPECT_EQ(uploads().at(0).detection_key, "first-detection");

    // The next clip carries its own key, not the previous one's.
    feed_until_event_arms(config_.send_video_clip_cooldown * kFramerate + kGop, "second-detection");
    feed(3 * kFramerate);
    ASSERT_TRUE(wait_for_uploads(2));
    EXPECT_EQ(uploads().at(1).detection_key, "second-detection");
}

TEST_F(VideoClipRecorderTest, AbsorbedEventDoesNotOverwriteTheKeyOfTheClipInFlight) {
    build();
    feed(5 * kFramerate);
    ASSERT_TRUE(recorder_->on_event("armed-the-clip"));

    // The suppressed detection is covered by the clip already being recorded,
    // so that clip stays keyed to the detection that started it.
    feed(5);
    ASSERT_FALSE(recorder_->on_event("absorbed"));

    feed(3 * kFramerate);
    ASSERT_TRUE(wait_for_uploads(1));
    EXPECT_EQ(uploads().at(0).detection_key, "armed-the-clip");
}

// --- in_cooldown(), which detection stills are gated on ----------------------

TEST_F(VideoClipRecorderTest, RecorderThatHasNeverTriggeredIsNotInCooldown) {
    build();
    EXPECT_FALSE(recorder_->in_cooldown());
    // Buffering footage is not a reason to hold anything off; the first
    // detection must still be able to arm a clip and upload its image.
    feed(5 * kFramerate);
    EXPECT_FALSE(recorder_->in_cooldown());
}

TEST_F(VideoClipRecorderTest, InCooldownWhileCapturingAndUntilTheCooldownElapses) {
    build();
    feed(5 * kFramerate);
    ASSERT_TRUE(recorder_->on_event(kDetectionKey));
    EXPECT_TRUE(recorder_->in_cooldown());

    feed(3 * kFramerate);
    ASSERT_TRUE(wait_for_uploads(1));
    ASSERT_FALSE(recorder_->is_capturing());
    // The capture is over, but the clip it produced still covers this footage.
    EXPECT_TRUE(recorder_->in_cooldown());

    // It clears exactly when a new event becomes able to arm a clip -- that
    // agreement is the whole point, since one gates the still and the other the
    // clip for what is a single detection.
    for (int i = 0; i < config_.send_video_clip_cooldown * kFramerate + kGop; ++i) {
        feed(1);
        if (!recorder_->in_cooldown())
            break;
    }
    EXPECT_FALSE(recorder_->in_cooldown());
    EXPECT_TRUE(recorder_->on_event("second-detection"));
}

TEST_F(VideoClipRecorderTest, StoppedRecorderIsNotInCooldown) {
    build();
    feed(5 * kFramerate);
    ASSERT_TRUE(recorder_->on_event(kDetectionKey));
    ASSERT_TRUE(recorder_->in_cooldown());

    // Timestamps stop advancing once it is stopped, so a cooldown left standing
    // would never expire -- and a stopped recorder has no clip to duplicate.
    recorder_->stop();
    EXPECT_FALSE(recorder_->in_cooldown());
}
