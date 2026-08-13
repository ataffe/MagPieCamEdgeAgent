//
// Created by alex on 6/26/26.
//
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Uploads an object -- a JPEG frame, an MP4 event clip -- to object storage
// (S3) using a presigned URL.
//
// Flow:
//   1. Load cached device credentials (public_camera_id, device_token), or
//      register the camera with the backend (using the device's serial number
//      and a claim token) if none are cached yet.
//   2. Exchange the device_token for a JWT.
//   3. Ask the webservice for a presigned upload URL for the object's
//      upload type and content type.
//   4. PUT the bytes directly to that URL, verifying the storage host's
//      certificate against backend.ca_cert_path when one is configured.
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
        std::string update_preview_time_endpoint;
        std::string serial_number_file;
        std::string claim_token_path;
        std::string credentials_path;
        // PEM bundle used to verify the storage host's TLS certificate on
        // presigned uploads. Optional: when empty, the system CA store is used.
        // Only the presigned PUT uses it -- the webservice calls still go over
        // plain HTTP.
        std::string ca_cert_path;
    };

    explicit BackendClient(const std::string &config_path);

    // Parse the (committed) backend endpoints file under config/backend/.
    // Throws std::runtime_error if the file is missing, malformed, or missing a
    // required key.
    static BackendConfig load_backend_config(const std::string &config_path);

    // Uploads an object and returns the storage key it was written to, or
    // std::nullopt if any step failed. content_type is the MIME type sent to S3
    // (defaults to JPEG). upload_type is sent to the backend's presign endpoint
    // to categorize the object (e.g. "DETECTION", "CAMERA_PREVIEW",
    // "VIDEO_CLIP"). detection_key is the storage key of the detection image
    // this object belongs to -- i.e. what an earlier "DETECTION" upload
    // returned -- and is required for "VIDEO_CLIP" uploads; it is what pairs a
    // clip with its detection.
    //
    // Safe to call from any thread, and concurrently with itself: like
    // get_jwt_token() below, it only reads state fixed at construction. The
    // clip recorder relies on that, uploading multi-megabyte MP4s from its own
    // worker thread while the camera callback keeps uploading JPEGs.
    std::optional<std::string> upload_object(const std::vector<uint8_t> &object,
                                             const std::string &content_type = "image/jpeg",
                                             const std::string &upload_type = "DETECTION",
                                             const std::string &detection_key = {});

    // Exchanges the cached device_token for a short-lived JWT via the
    // configured token endpoint. Public so callers outside upload_object() --
    // e.g. the RTSP streamer authenticating with MediaMTX -- can reuse it
    // instead of re-implementing the device-token exchange. Safe to call
    // concurrently with itself and with upload_object(): it only reads state
    // (backend_, device_token) that is fixed after construction.
    std::optional<std::string> get_jwt_token();

    // The camera's id as assigned by the backend during registration, cached
    // in backend.credentials_path alongside device_token. Public so callers
    // outside upload_object() -- e.g. the RTSP streamer, which needs it as the
    // per-camera path segment of the MediaMTX URL -- can read it.
    const std::string &get_public_camera_id() const { return public_camera_id; }

private:
    struct PresignedUpload {
        std::string url;
        std::string key;
    };

    // content_type is sent as a request header: the backend signs the URL for
    // that exact MIME type, so it must match the Content-Type that
    // put_to_storage() then sends to S3 or the signature won't verify.
    // detection_key is sent as a query parameter when non-empty.
    std::optional<PresignedUpload> get_presigned_url(const std::string &jwt_token,
                                                     const std::string &upload_type,
                                                     const std::string &content_type,
                                                     const std::string &detection_key);
    bool update_preview_time(const std::string &jwt_token);
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
