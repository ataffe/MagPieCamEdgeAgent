//
// Created by alex on 6/26/26.
//
#include "upload/image_uploader.h"

#include <fstream>
#include <stdexcept>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using json = nlohmann::json;

namespace {

json load_json_file(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("ImageUploader: could not open config file: " + path);
    }
    try {
        return json::parse(in);
    } catch (const json::parse_error &e) {
        throw std::runtime_error("ImageUploader: failed to parse " + path + ": " + e.what());
    }
}

// Writes a non-2xx response body to disk so the (often HTML) error page can be
// inspected without flooding the console.
void dump_error_body(const std::string &what, const cpr::Response &response) {
    constexpr char kErrorFile[] = "error.html";
    std::ofstream out(kErrorFile);
    out << response.text;
    spdlog::error("[ImageUploader] {} failed (HTTP {}), wrote response body to {}",
                  what, response.status_code, kErrorFile);
}

} // namespace

ImageUploader::ImageUploader(BackendConfig backend, Credentials credentials)
    : backend_(std::move(backend)), credentials_(std::move(credentials)) {}

ImageUploader::BackendConfig ImageUploader::load_backend_config(const std::string &path) {
    const json backend = load_json_file(path);
    try {
        const auto &svc = backend.at("scout_cam_webservice");
        return BackendConfig{
            svc.at("base_url").get<std::string>(),
            svc.at("token_endpoint").get<std::string>(),
            svc.at("presign_endpoint").get<std::string>(),
        };
    } catch (const json::exception &e) {
        throw std::runtime_error("[ImageUploader] Invalid backend config " + path + ": " + e.what());
    }
}

ImageUploader::Credentials ImageUploader::load_credentials(const std::string &path) {
    const json creds = load_json_file(path);
    try {
        const auto &user = creds.at("user");
        return Credentials{
            user.at("email").get<std::string>(),
            user.at("username").get<std::string>(),
            user.at("password").get<std::string>(),
        };
    } catch (const json::exception &e) {
        throw std::runtime_error("ImageUploader: invalid credentials " + path + ": " + e.what());
    }
}

ImageUploader ImageUploader::from_config(const std::string &backend_config_path,
                                         const std::string &credentials_path) {
    return ImageUploader(load_backend_config(backend_config_path),
                         load_credentials(credentials_path));
}

std::optional<std::string> ImageUploader::get_jwt_token() {
    json body;
    body["email"] = credentials_.email;
    body["username"] = credentials_.username;
    body["password"] = credentials_.password;

    cpr::Response response = cpr::Post(
        cpr::Url{backend_.base_url + backend_.token_endpoint},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{body.dump()});

    if (response.status_code != 200) {
        dump_error_body("token request", response);
        return std::nullopt;
    }
    spdlog::debug("[Image Uploader] Successfully retrieved JWT with status code: {}", response.status_code);

    try {
        return json::parse(response.text).at("access").get<std::string>();
    } catch (const json::exception &e) {
        spdlog::error("[ImageUploader] Malformed token response: {}", e.what());
        return std::nullopt;
    }
}

std::optional<ImageUploader::PresignedUpload> ImageUploader::get_presigned_url(const std::string &jwt_token) {
    cpr::Response response = cpr::Post(
        cpr::Url{backend_.base_url + backend_.presign_endpoint},
        cpr::Header{{"Authorization", "Bearer " + jwt_token}});

    if (response.status_code != 200) {
        dump_error_body("presign request", response);
        return std::nullopt;
    }
    spdlog::debug("[Image Uploader] Successfully retrieved presigned url with status code: {}", response.status_code);

    try {
        const json parsed = json::parse(response.text);
        return PresignedUpload{
            parsed.at("url").get<std::string>(),
            parsed.at("key").get<std::string>(),
        };
    } catch (const json::exception &e) {
        spdlog::error("[ImageUploader] Malformed presign response: {}", e.what());
        return std::nullopt;
    }
}

bool ImageUploader::put_to_storage(const PresignedUpload &target,
                                   const std::vector<uint8_t> &image,
                                   const std::string &content_type) {
    cpr::Response response = cpr::Put(
        cpr::Url{target.url},
        cpr::Header{{"Content-Type", content_type}},
        cpr::Body{reinterpret_cast<const char *>(image.data()), image.size()});

    // S3 returns 200 OK for a successful presigned PUT.
    if (response.status_code != 200) {
        spdlog::error("[Image Uploader] Upload with presigned url failed. Response code: {} | Body: {}",
            response.status_code, response.text);
        return false;
    }

    spdlog::info("[ImageUploader] Uploaded {} bytes to key {}", image.size(), target.key);
    return true;
}

bool ImageUploader::upload_image(const std::vector<uint8_t> &image,
                                 const std::string &content_type) {
    if (image.empty()) {
        spdlog::error("[ImageUploader] Refusing to upload empty image.");
        return false;
    }

    const auto jwt_token = get_jwt_token();
    if (!jwt_token) {
        return false;
    }

    const auto target = get_presigned_url(*jwt_token);
    if (!target) {
        return false;
    }

    return put_to_storage(*target, image, content_type);
}
