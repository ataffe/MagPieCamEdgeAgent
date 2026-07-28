//
// Created by alex on 6/26/26.
//
#include "backend/backend_client.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

using json = nlohmann::json;

namespace {

json load_json_file(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("BackendClient: could not open config file: " + path);
    }
    try {
        return json::parse(in);
    } catch (const json::parse_error &e) {
        throw std::runtime_error("BackendClient: failed to parse " + path + ": " + e.what());
    }
}

// Writes a non-2xx response body to disk so the (often HTML) error page can be
// inspected without flooding the console.
void dump_error_body(const std::string &what, const cpr::Response &response) {
    constexpr char kErrorFile[] = "error.html";
    std::ofstream out(kErrorFile);
    out << response.text;
    spdlog::error("[BackendClient] {} failed (HTTP {}), wrote response body to {}",
                  what, response.status_code, kErrorFile);
}

// Reads a file fully into a string, stripping stray NUL bytes and trailing
// whitespace -- both common in identifiers/secrets read from sysfs or
// hand-edited files. Returns std::nullopt if the file can't be opened.
std::optional<std::string> read_trimmed_file(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return std::nullopt;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string contents = buffer.str();

    contents.erase(std::remove(contents.begin(), contents.end(), '\0'), contents.end());
    while (!contents.empty() && std::isspace(static_cast<unsigned char>(contents.back()))) {
        contents.pop_back();
    }
    return contents;
}

} // namespace

BackendClient::BackendClient(const std::string &config_path) {
    backend_ = load_backend_config(config_path);
    load_credentials();
}

BackendClient::BackendConfig BackendClient::load_backend_config(const std::string &path) {
    const json backend = load_json_file(path);
    try {
        const auto &svc = backend.at("scout_cam_webservice");
        return BackendConfig{
            svc.at("base_url").get<std::string>(),
            svc.at("token_endpoint").get<std::string>(),
            svc.at("presign_endpoint").get<std::string>(),
            svc.at("registration_endpoint").get<std::string>(),
            svc.at("serial_number_file").get<std::string>(),
            svc.at("claim_token_path").get<std::string>(),
            svc.at("credentials_path").get<std::string>(),
        };
    } catch (const json::exception &e) {
        throw std::runtime_error("[BackendClient] Invalid backend config " + path + ": " + e.what());
    }
}

std::optional<std::string> BackendClient::get_jwt_token() {
    cpr::Response response = cpr::Post(
        cpr::Url{backend_.base_url + backend_.token_endpoint},
        cpr::Header{
            {"Content-Type", "application/json"},
            {"Authorization", "Device " +  device_token}});

    if (response.status_code != 200) {
        spdlog::error("Unable to retrieve JWT. Status Code: {} | Response {}", response.status_code ,response.text);
        return std::nullopt;
    }
    spdlog::debug("[BackendClient] Successfully retrieved JWT with status code: {}", response.status_code);

    try {
        return json::parse(response.text).at("access").get<std::string>();
    } catch (const json::exception &e) {
        spdlog::error("[BackendClient] Malformed token response: {}", e.what());
        return std::nullopt;
    }
}

std::optional<BackendClient::PresignedUpload> BackendClient::get_presigned_url(const std::string &jwt_token, const std::string &upload_type) {
    cpr::Response response = cpr::Post(
        cpr::Url{backend_.base_url + backend_.presign_endpoint},
        cpr::Parameters{{"upload_type", upload_type}},
        cpr::Header{{"Authorization", "Bearer " + jwt_token}});

    if (response.status_code != 200) {
        dump_error_body("presign request", response);
        return std::nullopt;
    }
    spdlog::debug("[BackendClient] Successfully retrieved presigned url with status code: {}", response.status_code);

    try {
        const json parsed = json::parse(response.text);
        return PresignedUpload{
            parsed.at("url").get<std::string>(),
            parsed.at("key").get<std::string>(),
        };
    } catch (const json::exception &e) {
        spdlog::error("[BackendClient] Malformed presign response: {}", e.what());
        return std::nullopt;
    }
}

bool BackendClient::put_to_storage(const PresignedUpload &target,
                                   const std::vector<uint8_t> &image,
                                   const std::string &content_type) {
    cpr::Response response = cpr::Put(
        cpr::Url{target.url},
        cpr::Header{{"Content-Type", content_type}},
        cpr::Body{reinterpret_cast<const char *>(image.data()), image.size()});

    // S3 returns 200 OK for a successful presigned PUT.
    if (response.status_code != 200) {
        spdlog::error("[BackendClient] Upload with presigned url failed. Response code: {} | Body: {}",
            response.status_code, response.text);
        return false;
    }

    spdlog::info("[BackendClient] Uploaded {} bytes to key {}", image.size(), target.key);
    return true;
}

bool BackendClient::upload_image(const std::vector<uint8_t> &image,
                                 const std::string &content_type,
                                 const std::string &upload_type) {
    if (image.empty()) {
        spdlog::error("[BackendClient] Refusing to upload empty image.");
        return false;
    }

    const auto jwt_token = get_jwt_token();
    if (!jwt_token) {
        return false;
    }

    const auto target = get_presigned_url(*jwt_token, upload_type);
    if (!target) {
        return false;
    }

    return put_to_storage(*target, image, content_type);
}

void BackendClient::load_credentials() {
    spdlog::info("[BackendClient] Loading credentials.");
    fs::path credentials_path = backend_.credentials_path;
    if (!fs::exists(credentials_path)) {
        spdlog::info("[BackendClient] Credentials not found locally. Registering camera.");
        register_camera();
    } else {
        spdlog::info("[BackendClient] Credentials found locally, reading in credentials.");
        std::ifstream file(backend_.credentials_path);
        try {
            json backend_credentials = json::parse(file);
            public_camera_id = backend_credentials.at("public_camera_id").get<std::string>();
            device_token = backend_credentials.at("device_token").get<std::string>();
            spdlog::info("[BackendClient] Successfully loaded local credentials.");
        } catch (const json::exception &e) {
            throw std::runtime_error("[BackendClient] Unable to read credentials: " + std::string(e.what()));
        }
    }
}

void BackendClient::register_camera() {
    spdlog::info("[BackendClient] Registering camera.");

    const auto serial_number = read_pi_serial_number();
    if (!serial_number.has_value()) {
        throw std::runtime_error("[BackendClient] Unable to read pi serial number.");
    }

    const auto claim_token = read_trimmed_file(backend_.claim_token_path);
    if (!claim_token.has_value()) {
        throw std::runtime_error("[BackendClient] Unable to read claim token from " + backend_.claim_token_path);
    }

    json body;
    body["device_id"] = *serial_number;
    body["claim_token"] = *claim_token;

    cpr::Response response = cpr::Post(
        cpr::Url{backend_.base_url + backend_.registration_endpoint},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{body.dump()});

    if (response.status_code != 201) {
        spdlog::error("Unable to register camera. Status Code: {} | Error: {}",
            response.status_code, response.text);
        throw std::runtime_error("[BackendClient] Camera registration failed.");
    }

    try {
        json parsed_credentials = json::parse(response.text);
        public_camera_id = parsed_credentials.at("public_camera_id").get<std::string>();
        device_token = parsed_credentials.at("device_token").get<std::string>();
    } catch (const json::exception &e) {
        throw std::runtime_error("[BackendClient] Unable to parse registration response: " + std::string(e.what()));
    }

    json creds_to_save;
    creds_to_save["public_camera_id"] = public_camera_id;
    creds_to_save["device_token"] = device_token;

    spdlog::info("[BackendClient] Retrieved credentials, saving to: {}", backend_.credentials_path);
    save_credentials_atomic(creds_to_save.dump(4));
    spdlog::info("[BackendClient] Saved credentials.");
}

std::optional<std::string> BackendClient::read_pi_serial_number() {
    auto serial_number = read_trimmed_file(backend_.serial_number_file);
    if (!serial_number.has_value()) {
        spdlog::error("Could not open serial-number file");
    }
    return serial_number;
}

void BackendClient::save_credentials_atomic(const std::string& credentials) {
    const std::string &cred_save_path = backend_.credentials_path;
    fs::path file_path(cred_save_path);
    fs::path dir = file_path.parent_path();

    if (!dir.empty()) {
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) {
            throw std::runtime_error("Failed to create directory " + dir.string() + ": " + ec.message());
        }
    }

    std::string tmp_path = cred_save_path + ".tmp";
    {
        std::ofstream file(tmp_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open temp file for writing: " + tmp_path);
        }
        file << credentials;
    }

    if (std::rename(tmp_path.c_str(), cred_save_path.c_str()) != 0) {
        throw std::runtime_error("Failed to atomically rename " + tmp_path + " to " + cred_save_path);
    }
}
