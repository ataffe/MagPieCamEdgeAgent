

#include "streaming/bbox_ws_server.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using byte_track::BboxWsServer;
using json = nlohmann::json;
using namespace std::chrono_literals;

namespace {

namespace fs = std::filesystem;

// Writes a config into a per-test temp file. Named after the running test so
// concurrent ctest jobs can't tread on each other; removed by RemoveConfig.
std::string write_config(const std::string &contents)
{
    static int counter = 0;
    const auto path = fs::temp_directory_path() /
                      ("bbox_ws_server_test_" + std::to_string(::getpid()) + "_" +
                       ::testing::UnitTest::GetInstance()->current_test_info()->name() + "_" +
                       std::to_string(counter++) + ".json");
    std::ofstream out(path);
    out << contents;
    out.close();
    return path.string();
}

void remove_config(const std::string &path)
{
    std::error_code ec;
    fs::remove(path, ec);
}

// Blocks until `predicate` holds or the deadline passes, so the socket tests
// synchronise on the server's actual state instead of a fixed sleep.
template <typename Predicate>
bool wait_until(Predicate predicate, std::chrono::milliseconds timeout = 2s)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate())
            return true;
        std::this_thread::sleep_for(5ms);
    }
    return predicate();
}

// A raw TCP client that speaks just enough WebSocket to drive the server.
class TestClient {
public:
    explicit TestClient(int port)
    {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        connected_ = ::connect(fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0;
    }

    ~TestClient() { close(); }

    TestClient(const TestClient &) = delete;
    TestClient &operator=(const TestClient &) = delete;

    bool connected() const { return connected_; }

    void send_raw(const std::string &data) { ::send(fd_, data.data(), data.size(), 0); }

    // Sends a well-formed upgrade request and returns the server's response.
    std::string handshake(const std::string &key = "dGhlIHNhbXBsZSBub25jZQ==")
    {
        send_raw("GET /bbox HTTP/1.1\r\n"
                 "Host: 127.0.0.1\r\n"
                 "Upgrade: websocket\r\n"
                 "Connection: Upgrade\r\n"
                 "Sec-WebSocket-Key: " + key + "\r\n"
                 "Sec-WebSocket-Version: 13\r\n\r\n");
        return recv_some();
    }

    std::string recv_some(std::chrono::milliseconds timeout = 2s)
    {
        timeval tv{};
        tv.tv_sec = static_cast<time_t>(timeout.count() / 1000);
        tv.tv_usec = static_cast<suseconds_t>((timeout.count() % 1000) * 1000);
        ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        char buf[8192];
        const ssize_t n = ::recv(fd_, buf, sizeof(buf), 0);
        if (n <= 0)
            return {};
        return std::string(buf, static_cast<std::size_t>(n));
    }

    void close()
    {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    int fd_ = -1;
    bool connected_ = false;
};

// Unwraps a server->client text frame, which is never masked.
std::string decode_text_frame(const std::string &frame)
{
    if (frame.size() < 2)
        return {};
    const auto first = static_cast<uint8_t>(frame[0]);
    EXPECT_EQ(first & 0x0F, 0x1) << "expected a text frame";
    EXPECT_EQ(first & 0x80, 0x80) << "expected FIN to be set";

    const auto second = static_cast<uint8_t>(frame[1]);
    EXPECT_EQ(second & 0x80, 0) << "server frames must not be masked";

    std::size_t len = second & 0x7F;
    std::size_t offset = 2;
    if (len == 126) {
        len = (static_cast<std::size_t>(static_cast<uint8_t>(frame[2])) << 8) |
              static_cast<uint8_t>(frame[3]);
        offset = 4;
    } else if (len == 127) {
        len = 0;
        for (int i = 0; i < 8; ++i)
            len = (len << 8) | static_cast<uint8_t>(frame[2 + i]);
        offset = 10;
    }
    return frame.substr(offset, len);
}

BboxWsServer::TrackedBox make_box(int id, int label, double score, double x, double y, double w,
                                  double h)
{
    BboxWsServer::TrackedBox box;
    box.track_id = id;
    box.label = label;
    box.score = score;
    box.x = x;
    box.y = y;
    box.w = w;
    box.h = h;
    return box;
}

// A server bound to an ephemeral port, so the tests never collide with a real
// one or with each other.
BboxWsServer::Config ephemeral_config()
{
    BboxWsServer::Config config;
    config.bind_address = "127.0.0.1";
    config.port = 0;
    return config;
}

}  // namespace

// --- Handshake ------------------------------------------------------------

TEST(BboxWsHandshakeTest, AcceptKeyMatchesTheRfcExample)
{
    // RFC 6455 §1.3's worked example.
    EXPECT_EQ(byte_track::websocket_accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
              "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
}

TEST(BboxWsHandshakeTest, KeyIsParsedCaseInsensitivelyAndTrimmed)
{
    const std::string request =
        "GET /bbox HTTP/1.1\r\n"
        "Host: pi.local\r\n"
        "sec-websocket-key:   abc123==   \r\n"
        "Upgrade: websocket\r\n\r\n";
    EXPECT_EQ(byte_track::parse_websocket_key(request), "abc123==");
}

TEST(BboxWsHandshakeTest, MissingKeyYieldsAnEmptyString)
{
    EXPECT_EQ(byte_track::parse_websocket_key("GET / HTTP/1.1\r\nHost: pi\r\n\r\n"), "");
}

// --- Frame encoding -------------------------------------------------------

TEST(BboxWsFrameTest, ShortPayloadUsesASingleLengthByte)
{
    const auto frame = byte_track::encode_text_frame("hi");
    ASSERT_EQ(frame.size(), 4u);
    EXPECT_EQ(static_cast<uint8_t>(frame[0]), 0x81);
    EXPECT_EQ(static_cast<uint8_t>(frame[1]), 2);  // no mask bit, length in place
    EXPECT_EQ(frame.substr(2), "hi");
}

TEST(BboxWsFrameTest, MediumPayloadUsesTheTwoByteExtendedLength)
{
    const std::string payload(200, 'x');
    const auto frame = byte_track::encode_text_frame(payload);
    ASSERT_EQ(frame.size(), payload.size() + 4);
    EXPECT_EQ(static_cast<uint8_t>(frame[1]), 126);
    EXPECT_EQ(static_cast<uint8_t>(frame[2]), 0);
    EXPECT_EQ(static_cast<uint8_t>(frame[3]), 200);
    EXPECT_EQ(decode_text_frame(frame), payload);
}

TEST(BboxWsFrameTest, LargePayloadUsesTheEightByteExtendedLength)
{
    const std::string payload(70000, 'y');
    const auto frame = byte_track::encode_text_frame(payload);
    ASSERT_EQ(frame.size(), payload.size() + 10);
    EXPECT_EQ(static_cast<uint8_t>(frame[1]), 127);
    EXPECT_EQ(decode_text_frame(frame), payload);
}

TEST(BboxWsFrameTest, BoundaryLengthsSwitchEncodingAtTheRightPoint)
{
    // 125 is the largest single-byte length; 126 is the first extended one.
    EXPECT_EQ(static_cast<uint8_t>(byte_track::encode_text_frame(std::string(125, 'a'))[1]), 125);
    EXPECT_EQ(static_cast<uint8_t>(byte_track::encode_text_frame(std::string(126, 'a'))[1]), 126);
}

// --- Payload --------------------------------------------------------------

TEST(BboxWsPayloadTest, TracksAreSerializedWithIdsScoresAndBoxes)
{
    const std::vector<BboxWsServer::TrackedBox> boxes{
        make_box(7, 0, 0.91, 0.1, 0.2, 0.3, 0.4),
        make_box(9, 15, 0.55, 0.5, 0.6, 0.05, 0.07),
    };

    const auto doc = json::parse(byte_track::serialize_frame(boxes, 42, 1723651200123LL));
    EXPECT_EQ(doc["frame_id"], 42);
    EXPECT_EQ(doc["timestamp_ms"], 1723651200123LL);
    ASSERT_EQ(doc["tracks"].size(), 2u);

    EXPECT_EQ(doc["tracks"][0]["id"], 7);
    EXPECT_EQ(doc["tracks"][0]["label"], 0);
    EXPECT_DOUBLE_EQ(doc["tracks"][0]["score"].get<double>(), 0.91);
    EXPECT_DOUBLE_EQ(doc["tracks"][0]["x"].get<double>(), 0.1);
    EXPECT_DOUBLE_EQ(doc["tracks"][0]["h"].get<double>(), 0.4);
    EXPECT_EQ(doc["tracks"][1]["id"], 9);
    EXPECT_EQ(doc["tracks"][1]["label"], 15);
}

TEST(BboxWsPayloadTest, AFrameWithNoTracksStillCarriesAnEmptyArray)
{
    const auto doc = json::parse(byte_track::serialize_frame({}, 3, 100));
    EXPECT_EQ(doc["frame_id"], 3);
    ASSERT_TRUE(doc["tracks"].is_array());
    EXPECT_TRUE(doc["tracks"].empty());
}

// --- Config ---------------------------------------------------------------

TEST(BboxWsConfigTest, AbsentSectionKeepsEveryDefault)
{
    const auto path = write_config(R"({"streaming": {"rtsp_url": "rtsp://host"}})");
    const auto config = BboxWsServer::Config::from_file(path);

    EXPECT_EQ(config.bind_address, "127.0.0.1");
    EXPECT_EQ(config.port, 8081);
    EXPECT_DOUBLE_EQ(config.inference_input_size, 640.0);
    EXPECT_EQ(config.max_clients, 4u);
    remove_config(path);
}

TEST(BboxWsConfigTest, PresentKeysOverrideTheDefaults)
{
    const auto path = write_config(R"({
        "bbox_debug": {
            "bind_address": "0.0.0.0",
            "port": 9100,
            "inference_input_size": 480,
            "max_clients": 2
        }
    })");
    const auto config = BboxWsServer::Config::from_file(path);

    EXPECT_EQ(config.bind_address, "0.0.0.0");
    EXPECT_EQ(config.port, 9100);
    EXPECT_DOUBLE_EQ(config.inference_input_size, 480.0);
    EXPECT_EQ(config.max_clients, 2u);
    remove_config(path);
}

TEST(BboxWsConfigTest, PartialSectionKeepsTheRemainingDefaults)
{
    const auto path = write_config(R"({"bbox_debug": {"port": 9999}})");
    const auto config = BboxWsServer::Config::from_file(path);

    EXPECT_EQ(config.port, 9999);
    EXPECT_EQ(config.bind_address, "127.0.0.1");
    EXPECT_DOUBLE_EQ(config.inference_input_size, 640.0);
    remove_config(path);
}

TEST(BboxWsConfigTest, MissingFileThrows)
{
    EXPECT_THROW(BboxWsServer::Config::from_file("/nonexistent/bbox.json"), std::runtime_error);
}

TEST(BboxWsConfigTest, MalformedJsonThrows)
{
    const auto path = write_config("{ not json");
    EXPECT_THROW(BboxWsServer::Config::from_file(path), std::runtime_error);
    remove_config(path);
}

TEST(BboxWsConfigTest, WrongTypeThrows)
{
    const auto path = write_config(R"({"bbox_debug": {"port": "8081"}})");
    EXPECT_THROW(BboxWsServer::Config::from_file(path), std::runtime_error);
    remove_config(path);
}

TEST(BboxWsConfigTest, OutOfRangeValuesThrow)
{
    for (const char *body : {R"({"bbox_debug": {"port": 70000}})",
                             R"({"bbox_debug": {"port": -1}})",
                             R"({"bbox_debug": {"bind_address": ""}})",
                             R"({"bbox_debug": {"inference_input_size": 0}})",
                             R"({"bbox_debug": {"max_clients": 0}})"}) {
        const auto path = write_config(body);
        EXPECT_THROW(BboxWsServer::Config::from_file(path), std::runtime_error) << body;
        remove_config(path);
    }
}

// --- Server behaviour -----------------------------------------------------

TEST(BboxWsServerTest, StartBindsAPortAndStopIsIdempotent)
{
    BboxWsServer server(ephemeral_config());
    ASSERT_TRUE(server.start());
    EXPECT_TRUE(server.is_running());
    EXPECT_GT(server.bound_port(), 0);

    server.stop();
    EXPECT_FALSE(server.is_running());
    server.stop();  // must not hang or crash
    EXPECT_FALSE(server.is_running());
}

TEST(BboxWsServerTest, StartIsIdempotent)
{
    BboxWsServer server(ephemeral_config());
    ASSERT_TRUE(server.start());
    const int port = server.bound_port();
    EXPECT_TRUE(server.start());  // second call is a no-op, not a rebind
    EXPECT_EQ(server.bound_port(), port);
    server.stop();
}

TEST(BboxWsServerTest, HandshakeCompletesAndTheClientIsCounted)
{
    BboxWsServer server(ephemeral_config());
    ASSERT_TRUE(server.start());

    TestClient client(server.bound_port());
    ASSERT_TRUE(client.connected());

    const auto response = client.handshake();
    EXPECT_NE(response.find("101 Switching Protocols"), std::string::npos);
    EXPECT_NE(response.find("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo="),
              std::string::npos);

    EXPECT_TRUE(wait_until([&] { return server.client_count() == 1; }));
    server.stop();
}

TEST(BboxWsServerTest, AnUpgradeWithoutAKeyIsRejected)
{
    BboxWsServer server(ephemeral_config());
    ASSERT_TRUE(server.start());

    TestClient client(server.bound_port());
    ASSERT_TRUE(client.connected());
    client.send_raw("GET /bbox HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");

    EXPECT_NE(client.recv_some().find("400 Bad Request"), std::string::npos);
    EXPECT_EQ(server.client_count(), 0u);
    server.stop();
}

TEST(BboxWsServerTest, BroadcastReachesAHandshakenClient)
{
    BboxWsServer server(ephemeral_config());
    ASSERT_TRUE(server.start());

    TestClient client(server.bound_port());
    ASSERT_TRUE(client.connected());
    ASSERT_NE(client.handshake().find("101"), std::string::npos);
    ASSERT_TRUE(wait_until([&] { return server.client_count() == 1; }));

    server.broadcast({make_box(3, 14, 0.77, 0.25, 0.5, 0.1, 0.2)}, 99);

    const auto payload = decode_text_frame(client.recv_some());
    ASSERT_FALSE(payload.empty());
    const auto doc = json::parse(payload);
    EXPECT_EQ(doc["frame_id"], 99);
    ASSERT_EQ(doc["tracks"].size(), 1u);
    EXPECT_EQ(doc["tracks"][0]["id"], 3);
    EXPECT_EQ(doc["tracks"][0]["label"], 14);
    EXPECT_DOUBLE_EQ(doc["tracks"][0]["x"].get<double>(), 0.25);

    server.stop();
}

TEST(BboxWsServerTest, EveryConnectedClientGetsTheFrame)
{
    BboxWsServer server(ephemeral_config());
    ASSERT_TRUE(server.start());

    TestClient a(server.bound_port());
    TestClient b(server.bound_port());
    ASSERT_TRUE(a.connected());
    ASSERT_TRUE(b.connected());
    ASSERT_NE(a.handshake().find("101"), std::string::npos);
    ASSERT_NE(b.handshake("YWJjZGVmZ2hpamtsbW5vcA==").find("101"), std::string::npos);
    ASSERT_TRUE(wait_until([&] { return server.client_count() == 2; }));

    server.broadcast({make_box(1, 0, 0.5, 0.0, 0.0, 1.0, 1.0)}, 5);

    for (auto *client : {&a, &b}) {
        const auto payload = decode_text_frame(client->recv_some());
        ASSERT_FALSE(payload.empty());
        EXPECT_EQ(json::parse(payload)["frame_id"], 5);
    }
    server.stop();
}

TEST(BboxWsServerTest, ClientsPastMaxClientsAreRefused)
{
    auto config = ephemeral_config();
    config.max_clients = 1;
    BboxWsServer server(config);
    ASSERT_TRUE(server.start());

    TestClient first(server.bound_port());
    ASSERT_TRUE(first.connected());
    ASSERT_NE(first.handshake().find("101"), std::string::npos);
    ASSERT_TRUE(wait_until([&] { return server.client_count() == 1; }));

    TestClient second(server.bound_port());
    second.send_raw("GET / HTTP/1.1\r\nSec-WebSocket-Key: abc==\r\n\r\n");
    // Refused connections are closed outright, so the read returns nothing.
    EXPECT_TRUE(second.recv_some(500ms).empty());
    EXPECT_EQ(server.client_count(), 1u);

    server.stop();
}

TEST(BboxWsServerTest, ADisconnectedClientIsDroppedFromTheCount)
{
    BboxWsServer server(ephemeral_config());
    ASSERT_TRUE(server.start());

    {
        TestClient client(server.bound_port());
        ASSERT_TRUE(client.connected());
        ASSERT_NE(client.handshake().find("101"), std::string::npos);
        ASSERT_TRUE(wait_until([&] { return server.client_count() == 1; }));
    }  // client closes here

    EXPECT_TRUE(wait_until([&] { return server.client_count() == 0; }));
    server.stop();
}

TEST(BboxWsServerTest, BroadcastingWithNoClientsIsHarmlessAndDropsNothing)
{
    BboxWsServer server(ephemeral_config());
    ASSERT_TRUE(server.start());

    for (int i = 0; i < 50; ++i)
        server.broadcast({make_box(i, 0, 0.5, 0.1, 0.1, 0.1, 0.1)}, i);

    // With nobody connected the mailbox is never filled, so nothing is
    // "dropped" -- the frames simply had nowhere to go.
    EXPECT_EQ(server.dropped_frames(), 0u);
    EXPECT_EQ(server.client_count(), 0u);
    server.stop();
}

TEST(BboxWsServerTest, BroadcastBeforeStartIsIgnored)
{
    BboxWsServer server(ephemeral_config());
    EXPECT_FALSE(server.is_running());
    server.broadcast({make_box(1, 0, 0.5, 0.1, 0.1, 0.1, 0.1)}, 1);  // must not crash
    EXPECT_EQ(server.dropped_frames(), 0u);
}

TEST(BboxWsServerTest, RestartRebindsAndServesAgain)
{
    BboxWsServer server(ephemeral_config());
    ASSERT_TRUE(server.start());
    server.stop();

    ASSERT_TRUE(server.start());
    TestClient client(server.bound_port());
    ASSERT_TRUE(client.connected());
    EXPECT_NE(client.handshake().find("101"), std::string::npos);
    EXPECT_TRUE(wait_until([&] { return server.client_count() == 1; }));
    server.stop();
}

TEST(BboxWsServerTest, StopWhileBroadcastingFromAnotherThreadIsSafe)
{
    // Mirrors the real wiring: the camera's post-processing thread broadcasts
    // while the command poller's thread stops the server. The wake eventfd is
    // shared between the two, so a stop that closed it out from under a
    // concurrent broadcast could write to a recycled descriptor.
    for (int attempt = 0; attempt < 20; ++attempt) {
        BboxWsServer server(ephemeral_config());
        ASSERT_TRUE(server.start());

        TestClient client(server.bound_port());
        ASSERT_TRUE(client.connected());
        ASSERT_NE(client.handshake().find("101"), std::string::npos);
        ASSERT_TRUE(wait_until([&] { return server.client_count() == 1; }));

        std::atomic<bool> stop_publishing{false};
        std::thread publisher([&] {
            while (!stop_publishing.load())
                server.broadcast({make_box(1, 0, 0.5, 0.1, 0.1, 0.2, 0.2)}, 0);
        });

        std::this_thread::sleep_for(5ms);
        server.stop();
        stop_publishing.store(true);
        publisher.join();

        EXPECT_FALSE(server.is_running());
    }
}

TEST(BboxWsServerTest, StopWithAClientStillConnectedDoesNotHang)
{
    BboxWsServer server(ephemeral_config());
    ASSERT_TRUE(server.start());

    TestClient client(server.bound_port());
    ASSERT_TRUE(client.connected());
    ASSERT_NE(client.handshake().find("101"), std::string::npos);
    ASSERT_TRUE(wait_until([&] { return server.client_count() == 1; }));

    const auto began = std::chrono::steady_clock::now();
    server.stop();
    EXPECT_LT(std::chrono::steady_clock::now() - began, 2s);
    EXPECT_FALSE(server.is_running());
}
