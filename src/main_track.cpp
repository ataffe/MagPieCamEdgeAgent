// Copyright © 2026 Alexander Taffe

  // src/main_track.cpp
  //
  // The IMX500 only produces inference (the rpi::CnnOutputTensor metadata) once
  // its network firmware (.rpk) has been uploaded to the sensor. That upload is
  // NOT something the standalone app does for free - in the post-processing
  // pipeline it was done by the chained `imx500_object_detection` stage. So here
  // we spin up our own PostProcessor, point it at imx500_only.json (which loads
  // just that stage), and let it do the firmware upload + per-frame tensor work.
  // Our own tracking then runs in the PostProcessor's result callback, where
  // CnnOutputTensor is guaranteed to be present.
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>
#include <stdexcept>
#include <unordered_map>

#include <chrono>
#include <cmath>
#include <unordered_set>

#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/formats.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>

#include "../include/tracking/byte_tracker.h"
#include "../include/backend/backend_client.h"
#include "core/buffer_sync.hpp"
#include "core/completed_request.hpp"
#include "core/rpicam_app.hpp"  // also pulls in core/post_processor.hpp
#include "core/video_options.hpp"

#include "tracking/byte_tracker.h"
#include "parsers/imx500_yolo_parser.h"
#include "backend/backend_client.h"
#include "streaming/ffmpeg_streamer.h"

using namespace std::chrono;
using byte_track::BYTETracker;
using byte_track::DetectedObject;
using byte_track::Vec4;


  namespace {
      // Paths the standalone app needs. The post-process file loads only the IMX500
      // firmware-upload stage; the libs dir is the same one run.sh stages.
      const char *kPostProcFile = "/home/alex/ScoutCamCameraClient/config/ml/imx500_config.json";
      const char *kPostProcLibs = "/home/alex/ScoutCamCameraClient/libs";
      const char *kBackendConfigPath = "/home/alex/ScoutCamCameraClient/config/backend/backend_config.json";



  // libcamera hands out YUV420 with row padding (stride >= width) and, on some
  // formats, the chroma planes packed behind the luma one. Repack it into a
  // tightly packed I420 buffer, which is what both OpenCV and ffmpeg's rawvideo
  // demuxer expect. `out` is reused across frames so this doesn't allocate.
  void pack_i420(const std::vector<libcamera::Span<uint8_t>> &planes, int w, int h, int stride,
                 std::vector<uint8_t> &out)
  {
      const uint8_t *y = planes[0].data();
      const int cstride = stride / 2;
      const uint8_t *u, *v;
      if (planes.size() >= 3) { u = planes[1].data(); v = planes[2].data(); }
      else { u = y + static_cast<size_t>(stride) * h; v = u + static_cast<size_t>(cstride) * (h / 2); }

      out.resize(static_cast<size_t>(w) * h * 3 / 2);
      uint8_t *yd = out.data();
      for (int r = 0; r < h; ++r) std::memcpy(yd + static_cast<size_t>(r) * w, y + static_cast<size_t>(r) * stride, w);
      uint8_t *ud = yd + static_cast<size_t>(w) * h;
      for (int r = 0; r < h / 2; ++r) std::memcpy(ud + static_cast<size_t>(r) * (w / 2), u + static_cast<size_t>(r) * cstride, w / 2);
      uint8_t *vd = ud + static_cast<size_t>(h / 2) * (w / 2);
      for (int r = 0; r < h / 2; ++r) std::memcpy(vd + static_cast<size_t>(r) * (w / 2), v + static_cast<size_t>(r) * cstride, w / 2);
  }

  void yuv420_to_bgr(const std::vector<libcamera::Span<uint8_t>> &planes, int w, int h, int stride,
                     std::vector<uint8_t> &scratch, cv::Mat &bgr)
  {
      pack_i420(planes, w, h, stride, scratch);
      cv::Mat i420(h + h / 2, w, CV_8UC1, scratch.data());
      cv::cvtColor(i420, bgr, cv::COLOR_YUV2BGR_I420);
  }
  }  // namespace



void add_args(argparse::ArgumentParser &parser)
{
      parser.add_argument("--debug")
      .help("Enable debug logging")
      .flag();

      parser.add_argument("--preview-interval-seconds")
      .help("Seconds between periodic camera preview uploads to S3")
      .default_value(300)
      .scan<'i', int>();

      parser.add_argument("--rtsp-url")
      .help("RTSP endpoint on the MediaMTX server to publish the video stream to")
      .default_value(std::string("rtsp://10.0.0.126:8554/cam"));

      parser.add_argument("--stream-width")
      .help("Width of the streamed video")
      .default_value(1280)
      .scan<'i', int>();

      parser.add_argument("--stream-height")
      .help("Height of the streamed video")
      .default_value(720)
      .scan<'i', int>();

      parser.add_argument("--stream-bitrate")
      .help("H.264 bitrate of the streamed video, in bits per second")
      .default_value(2000000)
      .scan<'i', int>();

      parser.add_argument("--stream-encoder")
      .help("ffmpeg H.264 encoder; h264_v4l2m2m is the Pi's hardware encoder, libx264 the CPU fallback")
      .default_value(std::string("h264_v4l2m2m"));

      parser.add_argument("--no-stream")
      .help("Disable RTSP video streaming (tracking and uploads still run)")
      .flag();
}

  static volatile bool quit = false;
  static void sig_handler(int) { quit = true; }

  int main(int argc, char *argv[])
  {
      std::signal(SIGINT,  sig_handler);
      std::signal(SIGTERM, sig_handler);

      RPiCamApp app(std::make_unique<VideoOptions>());
      Options *opts = app.GetOptions();
      argparse::ArgumentParser parser("Scout Camera Client");

      // Parse standard rpicam-apps flags (--width, --height, --framerate, etc.).
      if (char *dummy[1]; !opts->Parse(1, dummy))
          return 1;

      add_args(parser);
      try
      {
          parser.parse_args(argc, argv);
      } catch (const std::exception &err)
      {
          spdlog::error("{}", err.what());
          std::cerr << parser;
          std::exit(1);
      }

      bool debug = parser.get<bool>("--debug");
      spdlog::set_level(debug ? spdlog::level::debug : spdlog::level::info);

      const auto preview_interval = seconds(parser.get<int>("--preview-interval-seconds"));
      // Small lores stream for overlay/JPEG; main stream carries full-res frames.
      opts->Set().lores_width  = 640;
      opts->Set().lores_height = 480;
      // The main (viewfinder) stream is what gets encoded and pushed to RTSP, so
      // it is sized to the requested stream resolution rather than the display's.
      opts->Set().viewfinder_width  = parser.get<int>("--stream-width");
      opts->Set().viewfinder_height = parser.get<int>("--stream-height");

      app.OpenCamera();

      // --- IMX500 inference plumbing ---------------------------------------
      // A PostProcessor we drive ourselves. LoadModules makes the imx500 stage
      // discoverable; Read() parses imx500_only.json and registers the .rpk so
      // the firmware upload can begin. (Configure(), below, needs the camera to
      // be configured first, so it runs after ConfigureViewfinder().)
      PostProcessor post_processor(&app);
      post_processor.LoadModules(kPostProcLibs);
      post_processor.Read(kPostProcFile);

      // ConfigureViewfinder sets up a lores + main stream without recording.
      app.ConfigureViewfinder();

      // Uploads the network firmware to the IMX500. The imx500 stage prints
      // "Network Firmware Upload" progress; first run can take a while.
      post_processor.Configure();

      BYTETracker tracker(0.5, 0.8, 30, opts->Get().framerate.value_or(30.0f));

      // Constructing BackendClient registers the camera with the backend (a
      // network call) if no cached credentials exist yet. A transient network
      // outage at boot shouldn't stop local tracking/streaming, so this stays
      // best-effort: on failure we simply run without upload support.
      std::optional<BackendClient> backend_client;
      try {
          backend_client.emplace(kBackendConfigPath);
      } catch (const std::exception &err) {
          spdlog::error("[Track] Backend client unavailable, continuing without uploads: {}", err.what());
      }

      // --- RTSP video streaming --------------------------------------------
      // The main stream's frames are packed and piped to ffmpeg, which encodes
      // them to H.264 and publishes to MediaMTX. Sized from the stream libcamera
      // actually gave us, which may differ from what we asked for.
      std::optional<byte_track::FfmpegStreamer> streamer;
      libcamera::Stream *video_stream = app.GetMainStream();
      if (parser.get<bool>("--no-stream")) {
          spdlog::info("[Stream] RTSP streaming disabled by --no-stream");
      } else if (!video_stream) {
          spdlog::error("[Stream] No main stream available, continuing without video streaming");
      } else {
          const StreamInfo si = app.GetStreamInfo(video_stream);
          if (si.pixel_format != libcamera::formats::YUV420) {
              spdlog::error("[Stream] Main stream is {}, expected YUV420; continuing without video streaming",
                            si.pixel_format.toString());
          } else {
              byte_track::FfmpegStreamer::Config cfg;
              cfg.rtsp_url  = parser.get<std::string>("--rtsp-url");
              cfg.width     = static_cast<int>(si.width)  & ~1;
              cfg.height    = static_cast<int>(si.height) & ~1;
              cfg.framerate = static_cast<int>(std::lround(opts->Get().framerate.value_or(30.0f)));
              cfg.bitrate   = parser.get<int>("--stream-bitrate");
              cfg.gop       = cfg.framerate;  // one keyframe per second, as --intra 30 did at 30fps
              cfg.encoder   = parser.get<std::string>("--stream-encoder");

              streamer.emplace(cfg);
              if (!streamer->start()) {
                  spdlog::error("[Stream] Could not start ffmpeg, continuing without video streaming");
                  streamer.reset();
              }
          }
      }

      unsigned infer_frames = 0;
      bool send_frame = false;
      int sent_frames = 0;
      auto last_preview_upload = steady_clock::now();
      // Reused per-frame packing buffers; the callback is the only thread here.
      std::vector<uint8_t> video_i420;
      std::vector<uint8_t> jpeg_i420;

      // The PostProcessor delivers each request here AFTER the imx500 stage has
      // run, so CnnOutputTensor is present. This runs on the PostProcessor's
      // output thread; it is the only thread touching tracker.
      post_processor.SetCallback([&](CompletedRequestPtr &req) {
          // --- IMX500 inference output (only on frames where the NPU fired) ---
          auto out_ctrl  = req->metadata.get(libcamera::controls::rpi::CnnOutputTensor);
          auto info_ctrl = req->metadata.get(libcamera::controls::rpi::CnnOutputTensorInfo);

          if (out_ctrl && info_ctrl) {
              auto objs = byte_track::parse_imx500_detections(
                  out_ctrl->data(),  out_ctrl->size(),
                  info_ctrl->data(), info_ctrl->size(),
                  /*bbox_normalization=*/false, /*input_h=*/480.0);

              auto tracks = tracker.update(objs);

              for (auto &t : tracks)
              {
                  if (t->is_ready_to_send()) {
                      send_frame = true;
                      t->increment_send_count();
                  }
              }

              // if (++infer_frames % 30 == 0)
              //     spdlog::debug("[Track] infer_frame {} | dets={} tracks={} | cam_fps={:.1f}",
              //                   infer_frames, objs.size(), tracks.size(), req->framerate);
          }

          // --- H.264 stream to MediaMTX ---
          // Done before the JPEG work so a slow upload can't delay the stream.
          if (streamer) {
              if (auto vit = req->buffers.find(video_stream); vit != req->buffers.end()) {
                  BufferReadSync vr(&app, vit->second);
                  const auto &vplanes = vr.Get();
                  if (!vplanes.empty()) {
                      const StreamInfo vsi = app.GetStreamInfo(video_stream);
                      pack_i420(vplanes, streamer->config().width, streamer->config().height,
                                vsi.stride, video_i420);
                      streamer->write_frame(video_i420.data(), video_i420.size());
                  }
              }
          }

          // --- JPEG encode for backend upload ---
          libcamera::Stream *stream = app.LoresStream();
          if (!stream) stream = app.GetMainStream();
          if (!stream) return;

          auto it = req->buffers.find(stream);
          if (it == req->buffers.end()) return;

          StreamInfo si = app.GetStreamInfo(stream);
          const int width = static_cast<int>(si.width)  & ~1;
          const int height = static_cast<int>(si.height) & ~1;

          BufferReadSync r(&app, it->second);
          const auto &planes = r.Get();
          if (planes.empty()) return;

          cv::Mat bgr;
          yuv420_to_bgr(planes, width, height, si.stride, jpeg_i420, bgr);

          std::vector<uint8_t> jpg;
          cv::imencode(".jpg", bgr, jpg, {cv::IMWRITE_JPEG_QUALITY, 80});
          if (backend_client && send_frame && sent_frames++ < 3)
          {
              if (backend_client->upload_image(jpg)) {
                  spdlog::info("[Track] Successfully uploaded image to storage.");
              }
              send_frame = false;
          }

          const auto now = steady_clock::now();
          if (backend_client && now - last_preview_upload >= preview_interval)
          {
              if (backend_client->upload_image(jpg, "image/jpeg", "CAMERA_PREVIEW")) {
                  spdlog::info("[Track] Successfully uploaded camera preview image to storage.");
              }
              last_preview_upload = now;
          }
      });

      app.StartCamera();
      post_processor.Start();

      while (!quit)
      {
          // Wait() blocks until a frame arrives, a timeout fires, or Quit is sent.
          RPiCamApp::Msg msg = app.Wait();

          if (msg.type == RPiCamApp::MsgType::Quit)
              break;
          if (msg.type != RPiCamApp::MsgType::RequestComplete)
              continue;

          auto req = std::get<CompletedRequestPtr>(msg.payload);

          // Hand the frame to the imx500 stage; our callback runs once it's done.
          post_processor.Process(req);
      }

      post_processor.Stop();
      app.StopCamera();
      post_processor.Teardown();
      // After the callback can no longer fire, so no frame arrives mid-shutdown.
      if (streamer) {
          spdlog::info("[Stream] Stopping ffmpeg ({} frames dropped)", streamer->dropped_frames());
          streamer->stop();
      }
      return 0;
  }
