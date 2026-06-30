//
// Created by alex on 6/25/26.
//
#include <iostream>
#include <string>
#include <fstream>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

std::string BASE_URL = "http://10.0.0.125:8000";

std::optional<std::string> get_token() {
    json body;
    body["email"] = "ataffe37@gmail.com";
    body["username"] = "ataffe37@gmail.com";
    body["password"] = "spidey123af";

    cpr::Response token_response = cpr::Post(
        cpr::Url{BASE_URL + "/v1/auth/token/"},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{body.dump()});

    if (token_response.status_code != 200) {
        std::ofstream out("error.html");
        out << token_response.text;
        out.close();
        std::cerr << "Failed to get token, error code: " << token_response.status_code << " wrote output to error.html"
                << std::endl;
        return std::nullopt;
    }

    auto json_response = json::parse(token_response.text);
    return json_response["access"].get<std::string>();
}

void get_presigned_url(const std::string &jwt_token) {
    cpr::Response presigned_url_response = cpr::Post(
        cpr::Url{BASE_URL + "/v1/uploads/presign"},
        cpr::Header{{"Authorization", "Bearer " + jwt_token}});

    if (presigned_url_response.status_code != 200) {
        std::ofstream out("error.html");
        out << presigned_url_response.text;
        out.close();
        std::cerr << "Failed to get presigned url, error code: " << presigned_url_response.status_code
            << " wrote output to error.html"
            << std::endl;
    }
    auto json_response = json::parse(presigned_url_response.text);
    std::cout << "url: " << json_response["url"].get<std::string>() << std::endl;
    std::cout << "key: " << json_response["key"].get<std::string>() << std::endl;
    std::cout << "expires in: " << json_response["expires_in"].get<int>() << std::endl;
}

int main(int argc, char *argv[]) {
    auto jwt_token = get_token();
    if (jwt_token) {
        get_presigned_url(jwt_token.value());
    }
}