// Copyright © 2026 Alexander Taffe

  // src/main_track.cpp
  //
  // Standalone (loop-driven) version of byte_track_stage.cpp. Instead of running
  // as an rpicam-apps post-processing stage, we own main() and the camera event
  // loop ourselves - the shape the larger project needs so it can, per new track,
  // push a frame onto a queue (e.g. RabbitMQ) for cloud/LLM alerting.
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

  #include <libcamera/control_ids.h>
  #include <libcamera/controls.h>
  #include <opencv2/imgcodecs.hpp>
  #include <opencv2/imgproc.hpp>

  #include "core/buffer_sync.hpp"
  #include "core/completed_request.hpp"
  #include "core/rpicam_app.hpp"  // also pulls in core/post_processor.hpp
  #include "core/video_options.hpp"

  #include "tracking/byte_tracker.h"
  #include "parsers/imx500_yolo_parser.h"
  #include "server/mjpeg_server.h"

using namespace std::chrono;
  using byte_track::BYTETracker;
  using byte_track::DetectedObject;
  using byte_track::Vec4;

  namespace {
  // Paths the standalone app needs. The post-process file loads only the IMX500
  // firmware-upload stage; the libs dir is the same one run.sh stages.
  const char *kPostProcFile = "/home/alex/ScoutCamCameraClient/config/imx500_config.json";
  const char *kPostProcLibs = "/home/alex/ScoutCamCameraClient/libs";
  const char *kLabelsFile   = "/home/alex/ScoutCamCameraClient/labels/coco_labels_no_space.txt";

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

  static volatile bool quit = false;
  static void sig_handler(int) { quit = true; }

  int main(int argc, char *argv[])
  {
      std::signal(SIGINT,  sig_handler);
      std::signal(SIGTERM, sig_handler);

      RPiCamApp app(std::make_unique<VideoOptions>());
      Options *opts = app.GetOptions();

      // Parse standard rpicam-apps flags (--width, --height, --framerate, etc.).
      if (!opts->Parse(argc, argv))
          return 1;

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

      byte_track::MjpegServer server(8000);
      server.start();
      std::fprintf(stderr, "streaming on :8000\n");

      std::vector<std::string> labels;
      {
          std::ifstream f(kLabelsFile);
          std::string line;
          while (std::getline(f, line))
              if (!line.empty()) labels.push_back(line);
      }

      struct Drawable { Vec4 tlwh; int id; int label; };
      std::vector<Drawable> last_tracks;
      unsigned infer_frames = 0;

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

              last_tracks.clear();
              for (auto &t : tracks)
                  last_tracks.push_back({t->tlwh(), t->track_id, t->label});

              if (++infer_frames % 30 == 0)
                  std::fprintf(stderr, "[track] infer_frame %u | dets=%zu tracks=%zu | cam_fps=%.1f\n",
                               infer_frames, objs.size(), tracks.size(), req->framerate);

              // TODO(project): on a newly-seen track_id, push this frame onto the
              // RabbitMQ queue for the cloud/LLM alerting service.
          }

          // --- Draw + JPEG encode only when someone is watching ---
          if (!server.has_clients())
              return;

          libcamera::Stream *stream = app.LoresStream();
          if (!stream) stream = app.GetMainStream();
          if (!stream) return;

          auto it = req->buffers.find(stream);
          if (it == req->buffers.end()) return;

          StreamInfo si = app.GetStreamInfo(stream);
          const int w = static_cast<int>(si.width)  & ~1;
          const int h = static_cast<int>(si.height) & ~1;

          BufferReadSync r(&app, it->second);
          const auto &planes = r.Get();
          if (planes.empty()) return;

          cv::Mat bgr;
          yuv420_to_bgr(planes, w, h, si.stride, bgr);

          // ByteTrack boxes are in the 640x480 inference space; scale to the
          // lores stream's actual dimensions (matches byte_track_stage.cpp).
          const double sx = w / 640.0;
          const double sy = h / 480.0;
          for (const auto &d : last_tracks) {
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

          std::vector<uint8_t> jpg;
          cv::imencode(".jpg", bgr, jpg, {cv::IMWRITE_JPEG_QUALITY, 80});
          server.set_frame(jpg.data(), jpg.size());
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
