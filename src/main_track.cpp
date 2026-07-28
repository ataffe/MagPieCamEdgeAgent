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



  void yuv420_to_bgr(const std::vector<libcamera::Span<uint8_t>> &planes, int w, int h, int stride, cv::Mat &bgr)
  {
      const uint8_t *y = planes[0].data();
      const int cstride = stride / 2;
      const uint8_t *u, *v;
      if (planes.size() >= 3) { u = planes[1].data(); v = planes[2].data(); }
      else { u = y + static_cast<size_t>(stride) * h; v = u + static_cast<size_t>(cstride) * (h / 2); }

      cv::Mat i420(h + h / 2, w, CV_8UC1);
      for (int r = 0; r < h; ++r) std::memcpy(i420.ptr(r), y + static_cast<size_t>(r) * stride, w);
      uint8_t *ud = i420.ptr(h);
      for (int r = 0; r < h / 2; ++r) std::memcpy(ud + static_cast<size_t>(r) * (w / 2), u + static_cast<size_t>(r) * cstride, w / 2);
      uint8_t *vd = ud + static_cast<size_t>(h / 2) * (w / 2);
      for (int r = 0; r < h / 2; ++r) std::memcpy(vd + static_cast<size_t>(r) * (w / 2), v + static_cast<size_t>(r) * cstride, w / 2);

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

      unsigned infer_frames = 0;
      bool send_frame = false;
      int sent_frames = 0;
      auto last_preview_upload = steady_clock::now();

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

              if (++infer_frames % 30 == 0)
                  spdlog::debug("[Track] infer_frame {} | dets={} tracks={} | cam_fps={:.1f}",
                                infer_frames, objs.size(), tracks.size(), req->framerate);
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
          yuv420_to_bgr(planes, width, height, si.stride, bgr);

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
      return 0;
  }
