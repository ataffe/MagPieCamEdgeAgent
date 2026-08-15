// Copyright © 2026 Alexander Taffe

#include "streaming/bbox_ws_server.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace byte_track {

using json = nlohmann::json;

namespace {

constexpr char kBboxSection[] = "bbox_debug";

// RFC 6455 §1.3. Concatenated with the client's key before hashing.
constexpr char kWebSocketGuid[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

// Largest HTTP upgrade request we'll buffer before giving up on a client. A
// real handshake is a few hundred bytes; anything past this is not one.
constexpr std::size_t kMaxHandshakeBytes = 8192;

// --- SHA-1 -----------------------------------------------------------------
// Implemented here rather than pulled from OpenSSL: it is the single crypto
// primitive this file needs, and only for the handshake's fixed accept-key
// ritual -- which authenticates nothing, it just proves to the browser that the
// peer understood the upgrade. Inlining it keeps this translation unit free of
// link-time dependencies.
class Sha1 {
public:
    void update(const uint8_t *data, std::size_t len)
    {
        for (std::size_t i = 0; i < len; ++i) {
            buffer_[buffer_len_++] = data[i];
            if (buffer_len_ == 64) {
                transform(buffer_.data());
                total_bits_ += 512;
                buffer_len_ = 0;
            }
        }
    }

    std::array<uint8_t, 20> digest()
    {
        const uint64_t total = total_bits_ + static_cast<uint64_t>(buffer_len_) * 8;

        // Pad: 0x80, then zeros, then the 64-bit big-endian bit count.
        uint8_t pad = 0x80;
        update(&pad, 1);
        pad = 0x00;
        while (buffer_len_ != 56)
            update(&pad, 1);

        uint8_t len_be[8];
        for (int i = 0; i < 8; ++i)
            len_be[i] = static_cast<uint8_t>(total >> (56 - 8 * i));
        update(len_be, 8);

        std::array<uint8_t, 20> out{};
        for (int i = 0; i < 5; ++i) {
            out[i * 4 + 0] = static_cast<uint8_t>(state_[i] >> 24);
            out[i * 4 + 1] = static_cast<uint8_t>(state_[i] >> 16);
            out[i * 4 + 2] = static_cast<uint8_t>(state_[i] >> 8);
            out[i * 4 + 3] = static_cast<uint8_t>(state_[i]);
        }
        return out;
    }

private:
    static uint32_t rol(uint32_t v, int bits) { return (v << bits) | (v >> (32 - bits)); }

    void transform(const uint8_t *chunk)
    {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i)
            w[i] = (static_cast<uint32_t>(chunk[i * 4]) << 24) |
                   (static_cast<uint32_t>(chunk[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(chunk[i * 4 + 2]) << 8) |
                   static_cast<uint32_t>(chunk[i * 4 + 3]);
        for (int i = 16; i < 80; ++i)
            w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3], e = state_[4];
        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | (~b & d);          k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;                   k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else             { f = b ^ c ^ d;                   k = 0xCA62C1D6; }

            const uint32_t temp = rol(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rol(b, 30); b = a; a = temp;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d; state_[4] += e;
    }

    std::array<uint32_t, 5> state_{0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    std::array<uint8_t, 64> buffer_{};
    std::size_t buffer_len_ = 0;
    uint64_t total_bits_ = 0;
};

std::string base64_encode(const uint8_t *data, std::size_t len)
{
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (std::size_t i = 0; i < len; i += 3) {
        const uint32_t b0 = data[i];
        const uint32_t b1 = (i + 1 < len) ? data[i + 1] : 0;
        const uint32_t b2 = (i + 2 < len) ? data[i + 2] : 0;
        const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;

        out += kAlphabet[(triple >> 18) & 0x3F];
        out += kAlphabet[(triple >> 12) & 0x3F];
        out += (i + 1 < len) ? kAlphabet[(triple >> 6) & 0x3F] : '=';
        out += (i + 2 < len) ? kAlphabet[triple & 0x3F] : '=';
    }
    return out;
}

std::string trim(const std::string &value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
        return {};
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string to_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool set_nonblocking(int fd)
{
    const int flags = fcntl(fd, F_GETFL, 0);
    return flags != -1 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) != -1;
}

// Best-effort write of a whole buffer to a non-blocking socket. Returns false if
// the peer is gone; a merely-full send buffer (EAGAIN) also returns false, which
// costs that client this frame and nothing more.
enum class SendResult {
    Ok,
    // Nothing was written -- the send buffer was full. The client is still in
    // sync, so it just misses this frame.
    WouldBlock,
    // The peer is gone, or a partial write left a truncated frame on the wire.
    // Either way the connection cannot be used again: WebSocket framing has no
    // resynchronisation point, so a half-written frame desynchronises the
    // client permanently and it has to be dropped.
    Failed,
};

SendResult send_all(int fd, const char *data, std::size_t len)
{
    std::size_t sent = 0;
    while (sent < len) {
        const ssize_t n = ::send(fd, data + sent, len - sent, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n > 0) {
            sent += static_cast<std::size_t>(n);
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            return sent == 0 ? SendResult::WouldBlock : SendResult::Failed;
        return SendResult::Failed;
    }
    return SendResult::Ok;
}

// A connected client, before or after the upgrade handshake.
struct Client {
    int fd = -1;
    bool ready = false;      // handshake completed; frames may be sent
    std::string handshake;   // request bytes accumulated so far
};

}  // namespace

// --- Protocol helpers ------------------------------------------------------

std::string websocket_accept_key(const std::string &client_key)
{
    const std::string combined = client_key + kWebSocketGuid;
    Sha1 sha;
    sha.update(reinterpret_cast<const uint8_t *>(combined.data()), combined.size());
    const auto hash = sha.digest();
    return base64_encode(hash.data(), hash.size());
}

std::string parse_websocket_key(const std::string &request)
{
    std::size_t pos = 0;
    while (pos < request.size()) {
        const auto eol = request.find("\r\n", pos);
        const auto line = request.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
        const auto colon = line.find(':');
        if (colon != std::string::npos &&
            to_lower(trim(line.substr(0, colon))) == "sec-websocket-key")
            return trim(line.substr(colon + 1));

        if (eol == std::string::npos)
            break;
        pos = eol + 2;
    }
    return {};
}

std::string encode_text_frame(const std::string &payload)
{
    std::string frame;
    frame.reserve(payload.size() + 10);
    frame += static_cast<char>(0x81);  // FIN | opcode 0x1 (text)

    const std::size_t len = payload.size();
    if (len < 126) {
        frame += static_cast<char>(len);
    } else if (len <= 0xFFFF) {
        frame += static_cast<char>(126);
        frame += static_cast<char>((len >> 8) & 0xFF);
        frame += static_cast<char>(len & 0xFF);
    } else {
        frame += static_cast<char>(127);
        for (int i = 7; i >= 0; --i)
            frame += static_cast<char>((static_cast<uint64_t>(len) >> (8 * i)) & 0xFF);
    }
    // No mask: RFC 6455 §5.1 forbids servers masking outbound frames.
    frame += payload;
    return frame;
}

std::string serialize_frame(const std::vector<BboxWsServer::TrackedBox> &boxes,
                            std::int64_t frame_id, std::int64_t timestamp_ms)
{
    json tracks = json::array();
    for (const auto &box : boxes) {
        tracks.push_back({
            {"id", box.track_id},
            {"label", box.label},
            {"score", box.score},
            {"x", box.x},
            {"y", box.y},
            {"w", box.w},
            {"h", box.h},
        });
    }
    return json{{"frame_id", frame_id},
                {"timestamp_ms", timestamp_ms},
                {"tracks", std::move(tracks)}}
        .dump();
}

// --- Config ----------------------------------------------------------------

BboxWsServer::Config BboxWsServer::Config::from_file(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("BboxWsServer: could not open config file: " + path);

    json doc;
    try {
        doc = json::parse(in);
    } catch (const json::parse_error &e) {
        throw std::runtime_error("BboxWsServer: failed to parse " + path + ": " + e.what());
    }

    Config config;
    const auto section = doc.find(kBboxSection);
    // The section is optional: without it the overlay still works on defaults.
    if (section == doc.end())
        return config;

    try {
        if (const auto it = section->find("bind_address"); it != section->end())
            config.bind_address = it->get<std::string>();
        if (const auto it = section->find("port"); it != section->end())
            config.port = it->get<int>();
        if (const auto it = section->find("inference_input_size"); it != section->end())
            config.inference_input_size = it->get<double>();
        if (const auto it = section->find("max_clients"); it != section->end())
            config.max_clients = it->get<std::size_t>();
    } catch (const json::exception &e) {
        throw std::runtime_error("BboxWsServer: invalid bbox_debug settings in " + path + ": " +
                                 e.what());
    }

    const auto require = [&path](bool ok, const char *what) {
        if (!ok)
            throw std::runtime_error("BboxWsServer: invalid bbox_debug settings in " + path + ": " +
                                     what);
    };
    require(!config.bind_address.empty(), "bind_address must not be empty");
    // 0 is allowed: it asks the OS for an ephemeral port.
    require(config.port >= 0 && config.port <= 65535, "port must be between 0 and 65535");
    require(config.inference_input_size > 0.0, "inference_input_size must be positive");
    require(config.max_clients > 0, "max_clients must be positive");

    return config;
}

// --- Lifecycle -------------------------------------------------------------

BboxWsServer::BboxWsServer(Config config) : config_(std::move(config)) {}

BboxWsServer::~BboxWsServer() { stop(); }

bool BboxWsServer::start()
{
    // start()/stop() are driven by the command poller's thread alone, the same
    // single-caller assumption FfmpegStreamer makes.
    if (running_.load())
        return true;

    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        spdlog::error("[Bbox] socket() failed: {}", std::strerror(errno));
        return false;
    }

    const int reuse = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(config_.port));
    if (::inet_pton(AF_INET, config_.bind_address.c_str(), &addr.sin_addr) != 1) {
        spdlog::error("[Bbox] Invalid bind address \"{}\"", config_.bind_address);
        ::close(fd);
        return false;
    }

    if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        spdlog::error("[Bbox] bind({}:{}) failed: {}", config_.bind_address, config_.port,
                      std::strerror(errno));
        ::close(fd);
        return false;
    }

    if (::listen(fd, 8) != 0) {
        spdlog::error("[Bbox] listen() failed: {}", std::strerror(errno));
        ::close(fd);
        return false;
    }

    // Resolve the actual port, which matters when config_.port was 0.
    sockaddr_in bound{};
    socklen_t bound_len = sizeof(bound);
    if (::getsockname(fd, reinterpret_cast<sockaddr *>(&bound), &bound_len) == 0)
        bound_port_.store(ntohs(bound.sin_port));
    else
        bound_port_.store(config_.port);

    if (!set_nonblocking(fd)) {
        spdlog::error("[Bbox] Could not set the listener non-blocking: {}", std::strerror(errno));
        ::close(fd);
        return false;
    }

    const int wake = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (wake < 0) {
        spdlog::error("[Bbox] eventfd() failed: {}", std::strerror(errno));
        ::close(fd);
        return false;
    }

    listen_fd_ = fd;
    wake_fd_ = wake;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        has_pending_ = false;
        pending_.clear();
        client_count_ = 0;
    }
    dropped_frames_.store(0);
    running_.store(true);
    server_ = std::thread(&BboxWsServer::serve_loop, this);

    spdlog::info("[Bbox] Bounding-box overlay served at ws://{}:{}", config_.bind_address,
                 bound_port_.load());
    return true;
}

void BboxWsServer::stop()
{
    if (!running_.exchange(false))
        return;

    // Wakes the poll() so the thread notices running_ went false. Under the
    // lock because broadcast() may be writing to the same fd from the camera
    // thread, and the close below must not race it.
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (wake_fd_ >= 0) {
            const uint64_t one = 1;
            [[maybe_unused]] const ssize_t ignored = ::write(wake_fd_, &one, sizeof(one));
        }
    }
    if (server_.joinable())
        server_.join();

    if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }

    std::lock_guard<std::mutex> lock(mtx_);
    // Closed under the lock and cleared, so a concurrent broadcast() either
    // wrote before this or sees -1 and skips -- never writes to a recycled fd.
    if (wake_fd_ >= 0) { ::close(wake_fd_); wake_fd_ = -1; }
    client_count_ = 0;
    has_pending_ = false;
    pending_.clear();
    spdlog::debug("[Bbox] Overlay server stopped");
}

std::size_t BboxWsServer::client_count() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return client_count_;
}

void BboxWsServer::broadcast(const std::vector<TrackedBox> &boxes, std::int64_t frame_id)
{
    if (!running_.load())
        return;

    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    std::string payload = serialize_frame(boxes, frame_id, now);

    {
        std::lock_guard<std::mutex> lock(mtx_);
        // Nobody is watching: serializing was cheap, but there is nothing to
        // hand off, and a stale payload must not sit in the mailbox.
        if (client_count_ == 0) {
            has_pending_ = false;
            pending_.clear();
            return;
        }
        if (has_pending_)
            dropped_frames_.fetch_add(1);
        pending_ = std::move(payload);
        has_pending_ = true;

        // Inside the lock: stop() may be closing this fd concurrently, and an
        // 8-byte non-blocking eventfd write is not worth a second critical
        // section to avoid.
        if (wake_fd_ >= 0) {
            const uint64_t one = 1;
            [[maybe_unused]] const ssize_t ignored = ::write(wake_fd_, &one, sizeof(one));
        }
    }
}

// --- Serving thread --------------------------------------------------------

void BboxWsServer::serve_loop()
{
    std::vector<Client> clients;

    const auto drop_client = [&clients](std::size_t index) {
        ::close(clients[index].fd);
        clients.erase(clients.begin() + static_cast<std::ptrdiff_t>(index));
    };

    while (running_.load()) {
        std::vector<pollfd> fds;
        fds.reserve(clients.size() + 2);
        fds.push_back({listen_fd_, POLLIN, 0});
        fds.push_back({wake_fd_, POLLIN, 0});
        for (const auto &client : clients)
            fds.push_back({client.fd, POLLIN, 0});
        // How many clients fds actually describes. Anything accepted below is
        // appended past this point and has no revents to read, so it waits for
        // the next poll -- indexing fds by it would run off the end.
        const std::size_t polled_clients = clients.size();

        // The timeout only bounds how long shutdown waits if the eventfd write
        // is ever missed; normal wakeups are event-driven.
        if (::poll(fds.data(), fds.size(), 500) < 0) {
            if (errno == EINTR)
                continue;
            spdlog::error("[Bbox] poll() failed: {}", std::strerror(errno));
            break;
        }
        if (!running_.load())
            break;

        // --- New connections ---
        if (fds[0].revents & POLLIN) {
            while (true) {
                const int fd = ::accept(listen_fd_, nullptr, nullptr);
                if (fd < 0)
                    break;  // EAGAIN: drained the backlog
                if (clients.size() >= config_.max_clients) {
                    spdlog::warn("[Bbox] Refusing client: already at max_clients ({})",
                                 config_.max_clients);
                    ::close(fd);
                    continue;
                }
                if (!set_nonblocking(fd)) {
                    ::close(fd);
                    continue;
                }
                const int nodelay = 1;
                ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
                clients.push_back(Client{fd, false, {}});
            }
        }

        // --- Client readability: handshakes, closes, discarded inbound data ---
        // Descending so an erase only shifts entries already visited, keeping
        // clients[i] aligned with fds[i + 2].
        for (std::size_t i = polled_clients; i-- > 0;) {
            const auto &revents = fds[i + 2].revents;
            if (revents & (POLLHUP | POLLERR | POLLNVAL)) {
                drop_client(i);
                continue;
            }
            if (!(revents & POLLIN))
                continue;

            char buf[2048];
            const ssize_t n = ::recv(clients[i].fd, buf, sizeof(buf), MSG_DONTWAIT);
            if (n == 0) {  // orderly close
                drop_client(i);
                continue;
            }
            if (n < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                    continue;
                drop_client(i);
                continue;
            }

            if (clients[i].ready) {
                // Post-handshake inbound is only ever a close or a ping here.
                // The opcode is the low nibble of the first byte; 0x8 is close.
                if ((static_cast<uint8_t>(buf[0]) & 0x0F) == 0x8)
                    drop_client(i);
                continue;
            }

            clients[i].handshake.append(buf, static_cast<std::size_t>(n));
            if (clients[i].handshake.size() > kMaxHandshakeBytes) {
                spdlog::warn("[Bbox] Dropping client: handshake exceeded {} bytes",
                             kMaxHandshakeBytes);
                drop_client(i);
                continue;
            }
            if (clients[i].handshake.find("\r\n\r\n") == std::string::npos)
                continue;  // request still incomplete

            const auto key = parse_websocket_key(clients[i].handshake);
            if (key.empty()) {
                static constexpr char kBadRequest[] =
                    "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n";
                send_all(clients[i].fd, kBadRequest, sizeof(kBadRequest) - 1);
                spdlog::warn("[Bbox] Dropping client: no Sec-WebSocket-Key in upgrade request");
                drop_client(i);
                continue;
            }

            const std::string response =
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Accept: " + websocket_accept_key(key) + "\r\n\r\n";
            // A fresh socket's buffer easily holds a ~130-byte response, so
            // anything other than a clean write means the client is unusable.
            if (send_all(clients[i].fd, response.data(), response.size()) != SendResult::Ok) {
                drop_client(i);
                continue;
            }
            clients[i].ready = true;
            spdlog::info("[Bbox] Overlay client connected ({} total)", clients.size());
        }

        // --- Publish the latest frame ---
        if (fds[1].revents & POLLIN) {
            uint64_t counter = 0;
            [[maybe_unused]] const ssize_t ignored = ::read(wake_fd_, &counter, sizeof(counter));
        }

        std::string payload;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (has_pending_) {
                payload = std::move(pending_);
                pending_.clear();
                has_pending_ = false;
            }
            // Published here so broadcast() can skip work when nobody is
            // connected, and so client_count() reflects the live set.
            client_count_ = static_cast<std::size_t>(
                std::count_if(clients.begin(), clients.end(),
                              [](const Client &c) { return c.ready; }));
        }

        if (!payload.empty()) {
            const std::string frame = encode_text_frame(payload);
            for (std::size_t i = clients.size(); i-- > 0;) {
                if (!clients[i].ready)
                    continue;
                // WouldBlock costs this client the frame and nothing more; a
                // debug overlay would rather skip ahead than queue up stale
                // boxes behind a slow viewer.
                if (send_all(clients[i].fd, frame.data(), frame.size()) == SendResult::Failed) {
                    spdlog::debug("[Bbox] Dropping client: {}", std::strerror(errno));
                    drop_client(i);
                }
            }
        }
    }

    for (auto &client : clients)
        ::close(client.fd);
}

}  // namespace byte_track
