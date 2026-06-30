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
//   1. Authenticate against the Scout Cam webservice to obtain a JWT.
//   2. Ask the webservice for a presigned upload URL.
//   3. PUT the image bytes directly to that URL.
//
// Backend URLs are read from a (committed) JSON file; the user credentials are
// read from a separate (gitignored) JSON file so secrets stay out of version
// control. See config/backend/.
class ImageUploader {
public:
    // Non-secret backend endpoints.
    struct BackendConfig {
        std::string base_url;
        std::string token_endpoint;
        std::string presign_endpoint;
    };

    // Secret user credentials.
    struct Credentials {
        std::string email;
        std::string username;
        std::string password;
    };

    ImageUploader(BackendConfig backend, Credentials credentials);

    // Parse the (committed) backend endpoints file under config/backend/.
    // Throws std::runtime_error if the file is missing, malformed, or missing a
    // required key.
    static BackendConfig load_backend_config(const std::string &path);

    // Parse the (gitignored) user credentials file under config/backend/.
    // Throws std::runtime_error if the file is missing, malformed, or missing a
    // required key.
    static Credentials load_credentials(const std::string &path);

    // Convenience: load both structs from the JSON files under config/backend/.
    // Throws std::runtime_error if a file is missing or malformed.
    static ImageUploader from_config(const std::string &backend_config_path,
                                     const std::string &credentials_path);

    // Returns true if the image was successfully uploaded to storage.
    // content_type is the MIME type sent to S3 (defaults to JPEG).
    bool upload_image(const std::vector<uint8_t> &image,
                      const std::string &content_type = "image/jpeg");

private:
    struct PresignedUpload {
        std::string url;
        std::string key;
    };

    std::optional<std::string> get_jwt_token();
    std::optional<PresignedUpload> get_presigned_url(const std::string &jwt_token);
    bool put_to_storage(const PresignedUpload &target,
                        const std::vector<uint8_t> &image,
                        const std::string &content_type);

    BackendConfig backend_;
    Credentials credentials_;
};
