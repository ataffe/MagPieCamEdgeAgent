

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace byte_track {

// Serves the tracker's per-frame output to browser clients over a WebSocket, so
// the boxes can be drawn on top of the RTSP video while debugging.
//
// This is a debug aid, not a product surface: it is off until the backend sends
// "bbox_on", it only runs while the RTSP stream is up, and it is unauthenticated
// -- bind it on a trusted network only.
//
// Only the server->client half of RFC 6455 is implemented, which is all this
// needs: the handshake, unfragmented text frames out, and enough of the inbound
// framing to notice a client closing. Frames are never masked outbound (the RFC
// forbids it for servers) and inbound payloads are read and discarded.
//
// Threading: one poll()-driven thread owns the listening socket and every
// client. broadcast() is called from the camera's post-processing thread and
// never blocks it -- see the mailbox note on broadcast().
class BboxWsServer {
public:
    struct Config {
        // Loopback by default: the overlay is meant to be viewed through an SSH
        // tunnel rather than published to the LAN. Set "0.0.0.0" to expose it.
        std::string bind_address = "127.0.0.1";
        int port = 8081;

        // Side of the model's (square) input, used to normalize boxes to 0..1.
        // The tracker runs on raw network-input pixels -- parse_imx500_detections
        // is called with bbox_normalization=false -- and the tensor metadata does
        // not carry the input size, so it has to be stated here. It MUST match
        // the .rpk in the ML config or every box lands in the wrong place.
        double inference_input_size = 640.0;

        // Clients past this are refused. A debug overlay wants one viewer, maybe
        // two; the cap keeps a runaway reconnect loop from exhausting fds.
        std::size_t max_clients = 4;

        // Reads the optional "bbox_debug" section of the client config JSON.
        // Every key is optional, and so is the section itself -- absent, the
        // defaults above apply unchanged.
        //
        // Throws std::runtime_error if the file is missing or malformed, or if
        // a value is the wrong type or out of range.
        static Config from_file(const std::string &path);
    };

    // One tracked object, already normalized to 0..1 in the *streamed frame's*
    // coordinate space, so a client can scale by its rendered size and draw.
    struct TrackedBox {
        double x = 0.0;  // left
        double y = 0.0;  // top
        double w = 0.0;
        double h = 0.0;
        double score = 0.0;
        int label = 0;
        int track_id = 0;
    };

    explicit BboxWsServer(Config config);
    ~BboxWsServer();

    BboxWsServer(const BboxWsServer &) = delete;
    BboxWsServer &operator=(const BboxWsServer &) = delete;

    // Binds, listens and spawns the serving thread. Returns false if the socket
    // could not be bound (e.g. the port is taken), which the caller should treat
    // as "no overlay this session" rather than fatal. Idempotent.
    bool start();

    // Closes every client and the listener, then joins the thread. Idempotent,
    // and safe to call from a different thread than start().
    void stop();

    bool is_running() const { return running_.load(); }

    // Publishes one frame's tracks to every connected client.
    //
    // Called from the post-processing thread, so it must not block: the payload
    // is serialized here (cheap -- a handful of boxes) and dropped into a
    // single-slot mailbox that the serving thread drains. A frame arriving
    // before the previous one has gone out simply replaces it, which is the
    // right policy for an overlay -- a stale box is worse than a missing one --
    // and bounds memory regardless of how far behind a client falls.
    void broadcast(const std::vector<TrackedBox> &boxes, std::int64_t frame_id);

    std::size_t client_count() const;

    // Frames replaced in the mailbox before they could be sent.
    std::uint64_t dropped_frames() const { return dropped_frames_.load(); }

    // The port actually bound. Equals config().port unless that was 0, in which
    // case the OS picked one -- which is how the tests get a free port.
    int bound_port() const { return bound_port_.load(); }

    const Config &config() const { return config_; }

private:
    void serve_loop();

    Config config_;

    std::thread server_;
    std::atomic<bool> running_{false};
    std::atomic<int> bound_port_{0};
    std::atomic<std::uint64_t> dropped_frames_{0};

    int listen_fd_ = -1;
    // Wakes poll() when a frame is posted or stop() is called.
    int wake_fd_ = -1;

    mutable std::mutex mtx_;
    std::string pending_;
    bool has_pending_ = false;
    mutable std::size_t client_count_ = 0;
};

// --- Pieces of the protocol, exposed for testing ---------------------------

// The Sec-WebSocket-Accept value for a client's Sec-WebSocket-Key: the RFC 6455
// base64(SHA1(key + magic GUID)) ritual.
std::string websocket_accept_key(const std::string &client_key);

// Extracts the Sec-WebSocket-Key header from an HTTP upgrade request. Returns an
// empty string if the request has no such header.
std::string parse_websocket_key(const std::string &request);

// Wraps `payload` in an unmasked, unfragmented RFC 6455 text frame.
std::string encode_text_frame(const std::string &payload);

// Serializes one frame's tracks into the JSON sent to clients.
std::string serialize_frame(const std::vector<BboxWsServer::TrackedBox> &boxes,
                            std::int64_t frame_id, std::int64_t timestamp_ms);

}  // namespace byte_track
