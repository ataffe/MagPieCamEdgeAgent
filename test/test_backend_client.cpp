// Copyright © 2026 Alexander Taffe
//
// Unit tests for BackendClient. The HTTP flow (registration / token / presign /
// S3 PUT) needs a live backend, so these tests cover the deterministic,
// network-free surface:
//   - parsing of the backend config JSON file,
//   - error handling for missing / malformed / incomplete config,
//   - loading cached device credentials without hitting the network,
//   - failing fast when credentials are missing and registration can't even
//     read a device serial number (still no network needed),
//   - the empty-image guard in upload_image() (returns before any network I/O).

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "backend/backend_client.h"

namespace fs = std::filesystem;

namespace {

using json = nlohmann::json;

// Writes config files into a per-test temp directory and cleans them up after.
class BackendClientConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("backend_client_test_" + std::to_string(::getpid()) + "_" +
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

    // Writes a backend_config.json whose credentials_path / serial_number_file /
    // claim_token_path point inside this test's isolated temp dir (unless
    // overridden), and returns its path.
    std::string backend_config_path(const std::string &credentials_path = "",
                                     const std::string &serial_number_file = "",
                                     const std::string &claim_token_path = "") {
        json cfg;
        cfg["scout_cam_webservice"] = {
            {"base_url", "http://10.0.0.125:8000"},
            {"token_endpoint", "/v1/auth/token/"},
            {"presign_endpoint", "/v1/uploads/presign"},
            {"registration_endpoint", "/v1/cameras/register/"},
            {"serial_number_file",
             serial_number_file.empty() ? (dir_ / "no_such_serial_file").string() : serial_number_file},
            {"claim_token_path",
             claim_token_path.empty() ? (dir_ / "no_such_claim_token").string() : claim_token_path},
            {"credentials_path",
             credentials_path.empty() ? (dir_ / "device_credentials.json").string() : credentials_path},
        };
        return write_file("backend_config.json", cfg.dump());
    }

    fs::path dir_;
};

constexpr char kValidBackend[] = R"({
    "scout_cam_webservice": {
        "base_url": "http://10.0.0.125:8000",
        "token_endpoint": "/v1/auth/token/",
        "presign_endpoint": "/v1/uploads/presign",
        "registration_endpoint": "/v1/cameras/register/",
        "serial_number_file": "/sys/firmware/devicetree/base/serial-number",
        "claim_token_path": "/tmp/backend_client_test_claim_token",
        "credentials_path": "/tmp/backend_client_test_credentials.json"
    }
})";

} // namespace

// --- load_backend_config -----------------------------------------------------

TEST_F(BackendClientConfigTest, ParsesBackendConfigFields) {
    const auto cfg = BackendClient::load_backend_config(write_file("backend_config.json", kValidBackend));
    EXPECT_EQ(cfg.base_url, "http://10.0.0.125:8000");
    EXPECT_EQ(cfg.token_endpoint, "/v1/auth/token/");
    EXPECT_EQ(cfg.presign_endpoint, "/v1/uploads/presign");
    EXPECT_EQ(cfg.registration_endpoint, "/v1/cameras/register/");
    EXPECT_EQ(cfg.serial_number_file, "/sys/firmware/devicetree/base/serial-number");
    EXPECT_EQ(cfg.claim_token_path, "/tmp/backend_client_test_claim_token");
    EXPECT_EQ(cfg.credentials_path, "/tmp/backend_client_test_credentials.json");
}

TEST_F(BackendClientConfigTest, BackendConfigMissingFileThrows) {
    EXPECT_THROW(BackendClient::load_backend_config((dir_ / "does_not_exist.json").string()),
                 std::runtime_error);
}

TEST_F(BackendClientConfigTest, BackendConfigMalformedJsonThrows) {
    EXPECT_THROW(BackendClient::load_backend_config(write_file("backend_config.json", "{ not valid json")),
                 std::runtime_error);
}

TEST_F(BackendClientConfigTest, BackendConfigMissingKeyThrows) {
    constexpr char kMissingPresign[] = R"({
        "scout_cam_webservice": {
            "base_url": "http://host",
            "token_endpoint": "/v1/auth/token/"
        }
    })";
    EXPECT_THROW(BackendClient::load_backend_config(write_file("backend_config.json", kMissingPresign)),
                 std::runtime_error);
}

TEST_F(BackendClientConfigTest, BackendConfigMissingTopLevelKeyThrows) {
    EXPECT_THROW(
        BackendClient::load_backend_config(write_file("backend_config.json", R"({"wrong_root": {}})")),
        std::runtime_error);
}

// --- construction / credential loading ---------------------------------------

TEST_F(BackendClientConfigTest, ConstructsFromCachedCredentialsWithoutNetwork) {
    const std::string creds_path = (dir_ / "device_credentials.json").string();
    write_file("device_credentials.json", R"({
        "public_camera_id": "cam-123",
        "device_token": "tok-abc"
    })");

    EXPECT_NO_THROW({ BackendClient client(backend_config_path(creds_path)); });
}

TEST_F(BackendClientConfigTest, ThrowsWhenCredentialsMissingAndSerialNumberUnreadable) {
    // No cached credentials file, and the serial-number file path doesn't
    // exist either, so registration fails before it would ever need the
    // network -- keeps this test hermetic.
    EXPECT_THROW(BackendClient client(backend_config_path()), std::runtime_error);
}

TEST_F(BackendClientConfigTest, ThrowsWhenClaimTokenFileMissing) {
    // Serial number is readable, but the claim token file (a separate,
    // gitignored file so the secret never lands in backend_config.json)
    // isn't there -- registration should fail without ever hitting the network.
    const std::string serial_path = write_file("serial", "1234567890abcdef");
    EXPECT_THROW(BackendClient client(backend_config_path("", serial_path)), std::runtime_error);
}

TEST_F(BackendClientConfigTest, ThrowsWhenCachedCredentialsAreMalformed) {
    write_file("device_credentials.json", R"({"not_the_right_keys": true})");
    EXPECT_THROW(
        BackendClient client(backend_config_path((dir_ / "device_credentials.json").string())),
        std::runtime_error);
}

// --- upload_image input validation -------------------------------------------

TEST_F(BackendClientConfigTest, UploadEmptyImageReturnsFalseWithoutNetwork) {
    write_file("device_credentials.json", R"({
        "public_camera_id": "cam-123",
        "device_token": "tok-abc"
    })");
    BackendClient client(backend_config_path((dir_ / "device_credentials.json").string()));

    // An empty buffer is rejected before any HTTP request is attempted, so this
    // stays hermetic even with no backend reachable.
    EXPECT_FALSE(client.upload_image(std::vector<uint8_t>{}));
}
