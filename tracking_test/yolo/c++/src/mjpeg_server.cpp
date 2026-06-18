#include "mjpeg_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace byte_track {

namespace {
constexpr char const *PAGE =
    "<html><body><img src=\"stream.mjpg\" width=\"640\" height=\"480\" /></body></html>";

bool send_all(int fd, const void *buf, size_t len) {
    const char *p = static_cast<const char *>(buf);
    size_t left = len;
    while (left) {
        ssize_t n = ::send(fd, p, left, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p += n;
        left -= static_cast<size_t>(n);
    }
    return true;
}
}  // namespace

MjpegServer::MjpegServer(int port) : port_(port) {}
MjpegServer::~MjpegServer() { stop(); }

void MjpegServer::start() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) { perror("[mjpeg] socket"); return; }
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (::bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        perror("[mjpeg] bind");
        ::close(listen_fd_);
        listen_fd_ = -1;
        return;
    }
    if (::listen(listen_fd_, 8) < 0) {
        perror("[mjpeg] listen");
        ::close(listen_fd_);
        listen_fd_ = -1;
        return;
    }
    running_ = true;
    accept_thread_ = std::thread([this] { accept_loop(); });
    std::fprintf(stderr, "[mjpeg] serving on http://0.0.0.0:%d/\n", port_);
}

void MjpegServer::stop() {
    bool was = running_.exchange(false);
    if (!was && listen_fd_ < 0) return;
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    cv_.notify_all();
    if (accept_thread_.joinable()) accept_thread_.join();
}

void MjpegServer::set_frame(const uint8_t *data, size_t len) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        frame_.assign(data, data + len);
        seq_++;
    }
    cv_.notify_all();
}

void MjpegServer::accept_loop() {
    while (running_) {
        int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) {
            if (running_) continue;
            break;
        }
        std::thread([this, fd] { handle_client(fd); }).detach();
    }
}

void MjpegServer::handle_client(int fd) {
    char req[2048];
    ssize_t n = ::recv(fd, req, sizeof(req) - 1, 0);
    if (n <= 0) { ::close(fd); return; }
    req[n] = 0;

    std::string path = "/";
    {
        char method[8], p[512];
        if (std::sscanf(req, "%7s %511s", method, p) == 2) path = p;
    }

    if (path == "/" || path == "/index.html") {
        std::string body = PAGE;
        char hdr[256];
        int hl = std::snprintf(hdr, sizeof(hdr),
                               "HTTP/1.0 200 OK\r\nContent-Type: text/html\r\nContent-Length: %zu\r\n"
                               "Connection: close\r\n\r\n",
                               body.size());
        send_all(fd, hdr, hl);
        send_all(fd, body.data(), body.size());
        ::close(fd);
        return;
    }

    if (path == "/stream.mjpg") {
        const char *hdr =
            "HTTP/1.0 200 OK\r\nConnection: close\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=FRAME\r\n\r\n";
        if (!send_all(fd, hdr, std::strlen(hdr))) { ::close(fd); return; }

        clients_++;
        uint64_t last = 0;
        while (running_) {
            std::vector<uint8_t> f;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [&] { return !running_ || seq_ != last; });
                if (!running_) break;
                last = seq_;
                f = frame_;
            }
            if (f.empty()) continue;
            char ph[128];
            int pl = std::snprintf(ph, sizeof(ph),
                                   "--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
                                   f.size());
            if (!send_all(fd, ph, pl)) break;
            if (!send_all(fd, f.data(), f.size())) break;
            if (!send_all(fd, "\r\n", 2)) break;
        }
        clients_--;
        ::close(fd);
        return;
    }

    const char *nf = "HTTP/1.0 404 Not Found\r\nConnection: close\r\n\r\n";
    send_all(fd, nf, std::strlen(nf));
    ::close(fd);
}

}  // namespace byte_track
