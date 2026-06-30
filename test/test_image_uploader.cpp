// Copyright © 2026 Alexander Taffe
//
// Unit tests for ImageUploader. The HTTP flow (token / presign / S3 PUT) needs a
// live backend, so these tests cover the deterministic, network-free surface:
//   - parsing of the backend + credentials JSON config files,
//   - error handling for missing / malformed / incomplete config,
//   - the empty-image guard in upload_image() (returns before any network I/O).

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "upload/image_uploader.h"

namespace fs = std::filesystem;

namespace {

// Writes config files into a per-test temp directory and cleans them up after.
class ImageUploaderConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("image_uploader_test_" + std::to_string(::getpid()) + "_" +
                ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::create_directories(dir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    // Writes `contents` to <dir>/<name> and returns the full path.
    std::string write_file(const std::string &name, const std::string &contents) {
        const fs::path path = dir_ / name;
        std::ofstream out(path);
        out << contents;
        out.close();
        return path.string();
    }

    std::string backend_path(const std::string &contents) {
        return write_file("backend.json", contents);
    }

    std::string creds_path(const std::string &contents) {
        return write_file("credentials.json", contents);
    }

    fs::path dir_;
};

constexpr char kValidBackend[] = R"({
    "scout_cam_webservice": {
        "base_url": "http://10.0.0.125:8000",
        "token_endpoint": "/v1/auth/token/",
        "presign_endpoint": "/v1/uploads/presign"
    }
})";

constexpr char kValidCredentials[] = R"({
    "user": {
        "email": "scout@example.com",
        "username": "scout",
        "password": "hunter2"
    }
})";

} // namespace

// --- load_backend_config -----------------------------------------------------

TEST_F(ImageUploaderConfigTest, ParsesBackendConfigFields) {
    const auto cfg = ImageUploader::load_backend_config(backend_path(kValidBackend));
    EXPECT_EQ(cfg.base_url, "http://10.0.0.125:8000");
    EXPECT_EQ(cfg.token_endpoint, "/v1/auth/token/");
    EXPECT_EQ(cfg.presign_endpoint, "/v1/uploads/presign");
}

TEST_F(ImageUploaderConfigTest, BackendConfigMissingFileThrows) {
    EXPECT_THROW(ImageUploader::load_backend_config((dir_ / "does_not_exist.json").string()),
                 std::runtime_error);
}

TEST_F(ImageUploaderConfigTest, BackendConfigMalformedJsonThrows) {
    EXPECT_THROW(ImageUploader::load_backend_config(backend_path("{ not valid json")),
                 std::runtime_error);
}

TEST_F(ImageUploaderConfigTest, BackendConfigMissingKeyThrows) {
    constexpr char kMissingPresign[] = R"({
        "scout_cam_webservice": {
            "base_url": "http://host",
            "token_endpoint": "/v1/auth/token/"
        }
    })";
    EXPECT_THROW(ImageUploader::load_backend_config(backend_path(kMissingPresign)),
                 std::runtime_error);
}

TEST_F(ImageUploaderConfigTest, BackendConfigMissingTopLevelKeyThrows) {
    EXPECT_THROW(ImageUploader::load_backend_config(backend_path(R"({"wrong_root": {}})")),
                 std::runtime_error);
}

// --- load_credentials --------------------------------------------------------

TEST_F(ImageUploaderConfigTest, ParsesCredentialFields) {
    const auto creds = ImageUploader::load_credentials(creds_path(kValidCredentials));
    EXPECT_EQ(creds.email, "scout@example.com");
    EXPECT_EQ(creds.username, "scout");
    EXPECT_EQ(creds.password, "hunter2");
}

TEST_F(ImageUploaderConfigTest, CredentialsMissingFileThrows) {
    EXPECT_THROW(ImageUploader::load_credentials((dir_ / "nope.json").string()),
                 std::runtime_error);
}

TEST_F(ImageUploaderConfigTest, CredentialsMissingKeyThrows) {
    constexpr char kMissingPassword[] = R"({
        "user": {
            "email": "scout@example.com",
            "username": "scout"
        }
    })";
    EXPECT_THROW(ImageUploader::load_credentials(creds_path(kMissingPassword)),
                 std::runtime_error);
}

// --- from_config -------------------------------------------------------------

TEST_F(ImageUploaderConfigTest, FromConfigSucceedsWithValidFiles) {
    EXPECT_NO_THROW({
        ImageUploader::from_config(backend_path(kValidBackend), creds_path(kValidCredentials));
    });
}

TEST_F(ImageUploaderConfigTest, FromConfigPropagatesBackendError) {
    EXPECT_THROW(
        ImageUploader::from_config((dir_ / "missing.json").string(), creds_path(kValidCredentials)),
        std::runtime_error);
}

// --- upload_image input validation -------------------------------------------

TEST_F(ImageUploaderConfigTest, UploadEmptyImageReturnsFalseWithoutNetwork) {
    auto uploader =
        ImageUploader::from_config(backend_path(kValidBackend), creds_path(kValidCredentials));
    // An empty buffer is rejected before any HTTP request is attempted, so this
    // stays hermetic even with no backend reachable.
    EXPECT_FALSE(uploader.upload_image(std::vector<uint8_t>{}));
}
