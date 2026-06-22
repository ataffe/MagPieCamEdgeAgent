// Copyright © 2026 Alexander Taffe

#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace byte_track {

// C++ port of streaming.py: holds the latest JPEG frame and serves it as an
// HTTP multipart/x-mixed-replace MJPEG stream. GET / returns a viewer page,
// GET /stream.mjpg returns the stream. Each client is handled on its own thread.
class MjpegServer {
public:
    explicit MjpegServer(int port = 8000);
    ~MjpegServer();

    void start();                                      // bind/listen + background accept loop
    void stop();
    void set_frame(const uint8_t* data, size_t len);   // publish the latest JPEG
    bool has_clients() const { return clients_.load() > 0; }  // any /stream.mjpg viewers?

private:
    void accept_loop();
    void handle_client(int fd);

    int port_;
    int listen_fd_ = -1;
    std::thread accept_thread_;
    std::atomic<bool> running_{false};

    std::atomic<int> clients_{0};
    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<uint8_t> frame_;
    uint64_t seq_ = 0;
};

}  // namespace byte_track
