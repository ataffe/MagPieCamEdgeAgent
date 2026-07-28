//
// Created by alex on 6/26/26.
//
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Uploads an image to object storage (S3) using a presigned URL.
//
// Flow:
//   1. Load cached device credentials (public_camera_id, device_token), or
//      register the camera with the backend (using the device's serial number
//      and a claim token) if none are cached yet.
//   2. Exchange the device_token for a JWT.
//   3. Ask the webservice for a presigned upload URL.
//   4. PUT the image bytes directly to that URL.
//
// Backend endpoints are read from a (committed) JSON file; device credentials
// are cached in a separate (gitignored) file at backend.credentials_path so
// registration only needs to happen once. See config/backend/.
class BackendClient {
public:
    // Non-secret backend endpoints and paths. claim_token_path and
    // credentials_path point at separate (gitignored) files -- the secrets
    // themselves never live in this (committed) config.
    struct BackendConfig {
        std::string base_url;
        std::string token_endpoint;
        std::string presign_endpoint;
        std::string registration_endpoint;
        std::string serial_number_file;
        std::string claim_token_path;
        std::string credentials_path;
    };

    explicit BackendClient(const std::string &config_path);

    // Parse the (committed) backend endpoints file under config/backend/.
    // Throws std::runtime_error if the file is missing, malformed, or missing a
    // required key.
    static BackendConfig load_backend_config(const std::string &config_path);

    // Returns true if the image was successfully uploaded to storage.
    // content_type is the MIME type sent to S3 (defaults to JPEG). upload_type
    // is sent to the backend's presign endpoint to categorize the image (e.g.
    // "DETECTION", "CAMERA_PREVIEW").
    bool upload_image(const std::vector<uint8_t> &image,
                      const std::string &content_type = "image/jpeg",
                      const std::string &upload_type = "DETECTION");

private:
    struct PresignedUpload {
        std::string url;
        std::string key;
    };

    std::optional<std::string> get_jwt_token();
    std::optional<PresignedUpload> get_presigned_url(const std::string &jwt_token, const std::string &upload_type);
    bool put_to_storage(const PresignedUpload &target,
                        const std::vector<uint8_t> &image,
                        const std::string &content_type);
    std::optional<std::string> read_pi_serial_number();
    void save_credentials_atomic(const std::string& credentials);

    // Loads cached device credentials if present; otherwise registers the
    // camera with the backend and persists the returned credentials.
    // Throws std::runtime_error on failure.
    void load_credentials();

    // Registers the camera with the backend using its serial number and the
    // configured claim token, then caches the returned credentials. Only
    // called by load_credentials() when no cached credentials exist -- calling
    // it again after credentials are already cached would re-spend the
    // (typically single-use) claim token.
    void register_camera();

    BackendConfig backend_;
    std::string public_camera_id;
    std::string device_token;
};
