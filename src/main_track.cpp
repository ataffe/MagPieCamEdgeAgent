// Copyright © 2026 Alexander Taffe

  // src/main_track.cpp
  //
  // The IMX500 only produces inference (the rpi::CnnOutputTensor metadata) once
  // its network firmware (.rpk) has been uploaded to the sensor. That upload is
  // NOT something the standalone app does for free - in the post-processing
  // pipeline it was done by the chained `imx500_object_detection` stage. So here
  // we spin up our own PostProcessor, point it at imx500_only.json (which loads
  // just that stage), and let it do the firmware upload + per-frame tensor work.
  // Our own tracking / drawing / streaming then runs in the PostProcessor's
  // result callback, where CnnOutputTensor is guaranteed to be present.
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
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
#include "../include/upload/image_uploader.h"
#include "core/buffer_sync.hpp"
#include "core/completed_request.hpp"
#include "core/rpicam_app.hpp"  // also pulls in core/post_processor.hpp
#include "core/video_options.hpp"

#include "tracking/byte_tracker.h"
#include "parsers/imx500_yolo_parser.h"
#include "server/mjpeg_server.h"
#include "upload/image_uploader.h"

using namespace std::chrono;
using byte_track::BYTETracker;
using byte_track::DetectedObject;
using byte_track::Vec4;


  namespace {
      // Paths the standalone app needs. The post-process file loads only the IMX500
      // firmware-upload stage; the libs dir is the same one run.sh stages.
      const char *kPostProcFile = "/home/alex/ScoutCamCameraClient/config/ml/imx500_config.json";
      const char *kPostProcLibs = "/home/alex/ScoutCamCameraClient/libs";
      const char *kLabelsFile   = "/home/alex/ScoutCamCameraClient/labels/coco_labels_no_space.txt";
      const char *kBackendConfig = "/home/alex/ScoutCamCameraClient/config/backend/backend_config.json";
      const char *kBackendCredentials = "/home/alex/ScoutCamCameraClient/config/backend/credentials.json";



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
      .help("Enable debug including server that streams video to port 8001")
      .flag();
}

void draw_bounding_boxes(cv::Mat &bgr, std::vector<byte_track::Drawable> &tracks, int w, int h, std::vector<std::string> &labels)
{
      // ByteTrack boxes are in the 640x480 inference space; scale to the
      // lores stream's actual dimensions (matches byte_track_stage.cpp).
      const double sx = w / 640.0;
      const double sy = h / 480.0;
      for (const auto &d : tracks)
      {
          cv::Rect box(cvRound(d.tlwh[0] * sx), cvRound(d.tlwh[1] * sy),
                       cvRound(d.tlwh[2] * sx), cvRound(d.tlwh[3] * sy));
          cv::rectangle(bgr, box, cv::Scalar(0, 255, 0), 2);
          std::string cls = (d.label >= 0 && d.label < (int)labels.size())
                            ? labels[d.label] : std::to_string(d.label);
          std::string lbl = "Id:" + std::to_string(d.id) + " " + cls;
          cv::Point org(box.x + 4, box.y + 16);
          cv::putText(bgr, lbl, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,0,0},       3, cv::LINE_AA);
          cv::putText(bgr, lbl, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,206,138},   1, cv::LINE_AA);
      }
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

      byte_track::MjpegServer server(8001);
      server.start();
      spdlog::info("streaming on :8001");
      std::vector<byte_track::Drawable> last_tracks;

      std::vector<std::string> labels;
      {
          std::ifstream f(kLabelsFile);
          std::string line;
          while (std::getline(f, line))
              if (!line.empty()) labels.push_back(line);
      }

      ImageUploader::BackendConfig backend_config = ImageUploader::load_backend_config(kBackendConfig);
      ImageUploader::Credentials backend_credentials = ImageUploader::load_credentials(kBackendCredentials);
      ImageUploader uploader(backend_config, backend_credentials);

      unsigned infer_frames = 0;
      bool send_frame = false;
      int sent_frames = 0;

      // The PostProcessor delivers each request here AFTER the imx500 stage has
      // run, so CnnOutputTensor is present. This runs on the PostProcessor's
      // output thread; it is the only thread touching tracker/server/last_tracks.
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

              // TODO(project): Replace last tracks with active_track_id
              last_tracks.clear();
              for (auto &t : tracks)
              {
                  last_tracks.push_back({t->tlwh(), t->track_id, t->label});
                  if (t->is_ready_to_send()) {
                      send_frame = true;
                      t->increment_send_count();
                  }
              }

              if (++infer_frames % 30 == 0)
                  spdlog::debug("[Track] infer_frame {} | dets={} tracks={} | cam_fps={:.1f}",
                                infer_frames, objs.size(), tracks.size(), req->framerate);
          }

          // --- Draw + JPEG encode only when someone is watching ---
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
          if (send_frame && ++sent_frames < 5)
          {
              if (uploader.upload_image(jpg)) {
                  spdlog::info("[Track] Successfully uploaded image to storage.");
              }
              send_frame = false;
          }

          if (server.has_clients()) {
              // ByteTrack boxes are in the 640x480 inference space; scale to the
              // lores stream's actual dimensions (matches byte_track_stage.cpp).
              draw_bounding_boxes(bgr, last_tracks, width, height, labels);
              cv::imencode(".jpg", bgr, jpg, {cv::IMWRITE_JPEG_QUALITY, 80});
              server.set_frame(jpg.data(), jpg.size());
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
      server.stop();
      app.StopCamera();
      post_processor.Teardown();
      return 0;
  }
