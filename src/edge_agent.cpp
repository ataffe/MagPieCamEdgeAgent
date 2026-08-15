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
#include <memory>
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
#include "streaming/bbox_ws_server.h"
#include "streaming/ffmpeg_streamer.h"
#include "streaming/h264_tee.h"
#include "streaming/stream_command_poller.h"
#include "streaming/video_clip_recorder.h"

using namespace std::chrono;
using byte_track::BYTETracker;
using byte_track::DetectedObject;
using byte_track::Vec4;


  namespace {
      // Paths the standalone app needs. The post-process file loads only the IMX500
      // firmware-upload stage; the libs dir is the same one run.sh stages.
      const char *kPostProcFile = "/home/alex/MagPieCamEdgeAgent/config/ml/imx500_config.json";
      const char *kPostProcLibs = "/home/alex/MagPieCamEdgeAgent/libs";
      const char *kBackendConfigPath = "/home/alex/MagPieCamEdgeAgent/config/backend/backend_config.json";



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
      .default_value(600)
      .scan<'i', int>();

      // The streaming flags below have no defaults on purpose: unless one is
      // passed explicitly, the value from the config file's "streaming" section
      // wins. See the streaming setup in main().
      parser.add_argument("--rtsp-url")
      .help("Override the configured RTSP endpoint to publish the video stream to");

      parser.add_argument("--stream-width")
      .help("Override the configured width of the streamed video")
      .scan<'i', int>();

      parser.add_argument("--stream-height")
      .help("Override the configured height of the streamed video")
      .scan<'i', int>();

      parser.add_argument("--stream-framerate")
      .help("Override the configured frame rate of the streamed video")
      .scan<'i', int>();

      parser.add_argument("--stream-bitrate")
      .help("Override the configured H.264 bitrate, in bits per second. Shared by the RTSP "
            "stream and the event clips, which come off the same encoder")
      .scan<'i', int>();

      parser.add_argument("--no-stream")
      .help("Disable RTSP video streaming (tracking, uploads and clips still run)")
      .flag();

      // Like the streaming flags, these have no defaults: the "video_clips"
      // section of the config file wins unless one is passed explicitly.
      parser.add_argument("--clip-pre-seconds")
      .help("Override how many seconds of footage before a detection go into its clip")
      .scan<'i', int>();

      parser.add_argument("--clip-post-seconds")
      .help("Override how many seconds of footage after a detection go into its clip")
      .scan<'i', int>();

      parser.add_argument("--clip-cooldown-seconds")
      .help("Override the minimum spacing between uploaded clips; detections inside the "
            "cooldown are already covered by the previous clip")
      .scan<'i', int>();

      parser.add_argument("--no-video-clips")
      .help("Disable event video clips (tracking, uploads and streaming still run)")
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
      argparse::ArgumentParser parser("MagPieCam Edge Agent");

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

      // --- Streaming configuration -----------------------------------------
      // Three layers, each overriding the previous: the defaults baked into
      // FfmpegStreamer::Config, the "streaming" section of the client config
      // file, and finally any explicitly passed CLI flag. Bad config is not
      // fatal -- tracking and uploads should still run.
      auto stream_config = byte_track::FfmpegStreamer::Config();
      try {
          stream_config = byte_track::FfmpegStreamer::Config::from_file(kBackendConfigPath);
      } catch (const std::exception &err) {
          spdlog::warn("[Stream] Falling back to default streaming settings: {}", err.what());
      }
      if (parser.is_used("--rtsp-url"))         stream_config.rtsp_url  = parser.get<std::string>("--rtsp-url");
      if (parser.is_used("--stream-width"))     stream_config.width     = parser.get<int>("--stream-width");
      if (parser.is_used("--stream-height"))    stream_config.height    = parser.get<int>("--stream-height");
      if (parser.is_used("--stream-framerate")) stream_config.framerate = parser.get<int>("--stream-framerate");
      if (parser.is_used("--stream-bitrate"))   stream_config.bitrate   = parser.get<int>("--stream-bitrate");

      // --- Video clip configuration ----------------------------------------
      // Same three layers as above. Bad config is not fatal either: without a
      // clip recorder the client still tracks, streams and uploads stills.
      std::optional<byte_track::VideoClipRecorder::Config> clip_config;
      try {
          clip_config = byte_track::VideoClipRecorder::Config::from_file(kBackendConfigPath);
      } catch (const std::exception &err) {
          spdlog::error("[Clip] Video clips unavailable: {}", err.what());
      }
      if (clip_config) {
          if (parser.is_used("--clip-pre-seconds"))
              clip_config->pre_event_length_seconds = parser.get<int>("--clip-pre-seconds");
          if (parser.is_used("--clip-post-seconds"))
              clip_config->post_event_length_seconds = parser.get<int>("--clip-post-seconds");
          if (parser.is_used("--clip-cooldown-seconds"))
              clip_config->send_video_clip_cooldown = parser.get<int>("--clip-cooldown-seconds");
      }

      // Small lores stream for overlay/JPEG; main stream carries full-res frames.
      opts->Set().lores_width  = 640;
      opts->Set().lores_height = 480;
      // The main (viewfinder) stream is what gets encoded and pushed to RTSP, so
      // it is sized to the configured stream resolution rather than the display's,
      // and the camera runs at the frame rate the stream is declared to have.
      opts->Set().viewfinder_width  = stream_config.width;
      opts->Set().viewfinder_height = stream_config.height;
      opts->Set().framerate         = static_cast<float>(stream_config.framerate);

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

      BYTETracker tracker(0.5, 0.8, 30, opts->Get().framerate.value_or(static_cast<float>(stream_config.framerate)));

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

      // --- The video pipeline ------------------------------------------------
      // The main stream is encoded to H.264 once, by H264Tee, and the encoded
      // frames are handed to two consumers: the RTSP streamer and the event
      // clip recorder. The Pi has a single hardware encoder, so encoding once
      // and sharing the result is both cheaper than a session per consumer and
      // the only way the clip recorder can have a pre-event buffer at all --
      // the past cannot be encoded after the fact.
      libcamera::Stream *video_stream = app.GetMainStream();
      bool video_stream_usable = false;
      if (!video_stream) {
          spdlog::error("[Encode] No main stream available; no streaming and no video clips");
      } else {
          const StreamInfo si = app.GetStreamInfo(video_stream);
          if (si.pixel_format != libcamera::formats::YUV420) {
              spdlog::error("[Encode] Main stream is {}, expected YUV420; no streaming and no video clips",
                            si.pixel_format.toString());
          } else {
              // libcamera may not have granted exactly the size we asked for, so
              // the encoder is told what the stream actually is.
              const int actual_width  = static_cast<int>(si.width)  & ~1;
              const int actual_height = static_cast<int>(si.height) & ~1;
              if (actual_width != stream_config.width || actual_height != stream_config.height) {
                  spdlog::warn("[Encode] Camera gave {}x{}, not the configured {}x{}; encoding at the former",
                               actual_width, actual_height, stream_config.width, stream_config.height);
                  stream_config.width  = actual_width;
                  stream_config.height = actual_height;
              }
              video_stream_usable = true;
          }
      }

      // --- RTSP video streaming --------------------------------------------
      // Streaming is on demand: the streamer is built here but stays idle until
      // the backend asks for it (see the stream command poller below), so an
      // unwatched camera spends nothing on the uplink. The encoder keeps
      // running either way -- the clip recorder needs it.
      std::optional<byte_track::FfmpegStreamer> streamer;
      if (parser.get<bool>("--no-stream")) {
          spdlog::info("[Stream] RTSP streaming disabled by --no-stream");
      } else if (video_stream_usable) {
          streamer.emplace(stream_config);
          // MediaMTX now serves multiple cameras behind one host:port, each
          // at a path keyed by its public_camera_id, and requires a token;
          // reuse BackendClient's cached camera id and device-token
          // exchange rather than re-implementing either here.
          if (backend_client) {
              streamer->set_public_camera_id(backend_client->get_public_camera_id());
              streamer->set_jwt_provider([&backend_client]() { return backend_client->get_jwt_token(); });
          } else {
              spdlog::warn("[Stream] No backend client available; publishing to MediaMTX unauthenticated");
          }
      }

      // --- Event video clips -------------------------------------------------
      // Keeps a rolling buffer of encoded frames so a detection can be uploaded
      // with the seconds leading up to it, not just the ones after.
      std::optional<byte_track::VideoClipRecorder> clip_recorder;
      if (parser.get<bool>("--no-video-clips")) {
          spdlog::info("[Clip] Video clips disabled by --no-video-clips");
      } else if (video_stream_usable && clip_config) {
          if (!backend_client) {
              spdlog::error("[Clip] No backend client; video clips disabled (nowhere to upload them)");
          } else {
              // The camera's real frame rate is what gives the muxed MP4 its
              // timebase, so take it from the resolved stream settings.
              clip_config->framerate = stream_config.framerate;
              clip_recorder.emplace(*clip_config);
              clip_recorder->set_uploader(
                  [&backend_client, content_type = clip_config->content_type,
                   upload_type = clip_config->upload_type](const std::vector<uint8_t> &clip,
                                                           const std::string &detection_key) {
                      return backend_client
                          ->upload_object(clip, content_type, upload_type, detection_key)
                          .has_value();
                  });
              clip_recorder->start();
          }
      }

      // --- Shared H.264 encoder ----------------------------------------------
      // Built last, once it is known which consumers actually exist. If the
      // encoder can't be opened neither consumer can work, so both are torn
      // back down rather than left waiting for frames that will never arrive.
      std::unique_ptr<byte_track::H264Tee> tee;
      if (streamer || clip_recorder) {
          byte_track::H264Tee::Config encoder_config;
          encoder_config.bitrate   = stream_config.bitrate;
          encoder_config.gop       = stream_config.gop.value_or(stream_config.framerate);
          encoder_config.framerate = static_cast<float>(stream_config.framerate);
          try {
              tee = std::make_unique<byte_track::H264Tee>(&app, video_stream, encoder_config);
          } catch (const std::exception &err) {
              spdlog::error("[Encode] H.264 encoder unavailable, no streaming and no video clips: {}",
                            err.what());
          }

          if (tee) {
              if (streamer) {
                  tee->add_sink([&streamer](const uint8_t *data, size_t len, int64_t, bool keyframe) {
                      streamer->write_access_unit(data, len, keyframe);
                  });
              }
              if (clip_recorder) {
                  tee->add_sink([&clip_recorder](const uint8_t *data, size_t len, int64_t timestamp_us,
                                                 bool keyframe) {
                      clip_recorder->on_encoded_frame(data, len, timestamp_us, keyframe);
                  });
              }
          } else {
              streamer.reset();
              if (clip_recorder) {
                  clip_recorder->stop();
                  clip_recorder.reset();
              }
          }
      }

      // --- Bounding-box debug overlay ---------------------------------------
      // Serves the tracker's per-frame output over a WebSocket so the boxes can
      // be drawn on the video while debugging. Built alongside the streamer but
      // left stopped: it only runs between a "bbox_on" command and either
      // "bbox_off" or the stream stopping. Without a streamer there is no video
      // to overlay, so there is nothing to build.
      std::optional<byte_track::BboxWsServer> bbox_server;
      if (streamer) {
          try {
              bbox_server.emplace(byte_track::BboxWsServer::Config::from_file(kBackendConfigPath));
          } catch (const std::exception &err) {
              spdlog::error("[Bbox] Overlay unavailable: {}", err.what());
          }
      }

      // --- Stream command long poll ----------------------------------------
      // Asks the backend whether anyone wants to watch, and starts/stops the
      // streamer to match. Without a backend client there is no JWT to poll
      // with, so streaming would never be requested -- say so rather than
      // leaving a silently idle streamer.
      std::optional<byte_track::StreamCommandPoller> poller;
      if (streamer && !backend_client) {
          spdlog::error("[StreamCmd] No backend client; stream will stay off (nothing can request it)");
      } else if (streamer) {
          try {
              poller.emplace(byte_track::StreamCommandPoller::Config::from_file(kBackendConfigPath));
          } catch (const std::exception &err) {
              spdlog::error("[StreamCmd] Stream commands unavailable, stream will stay off: {}", err.what());
          }
      }
      if (poller) {
          poller->set_jwt_provider([&backend_client]() { return backend_client->get_jwt_token(); });
          poller->set_on_start([&streamer]() {
              if (streamer->is_running())
                  return;  // already streaming; the backend re-sends "start" as a nudge
              if (!streamer->start())
                  spdlog::error("[Stream] Could not start ffmpeg for the requested stream");
          });
          poller->set_on_stop([&streamer, &bbox_server]() {
              if (!streamer->is_running())
                  return;
              spdlog::info("[Stream] Stopping ffmpeg ({} frames dropped)", streamer->dropped_frames());
              streamer->stop();
              // The overlay exists to be drawn over this stream, so it goes with
              // it; watching again needs a fresh "bbox_on".
              if (bbox_server && bbox_server->is_running()) {
                  spdlog::info("[Bbox] Stream stopped, stopping the overlay too");
                  bbox_server->stop();
              }
          });
          poller->set_on_bbox_on([&streamer, &bbox_server]() {
              if (!bbox_server) {
                  spdlog::warn("[Bbox] \"bbox_on\" ignored: the overlay failed to configure");
                  return;
              }
              // Strictly follows the stream: with no video to draw on, an
              // overlay socket would just sit there serving boxes for nothing.
              if (!streamer->is_running()) {
                  spdlog::warn("[Bbox] \"bbox_on\" ignored: the stream is not running");
                  return;
              }
              if (bbox_server->is_running())
                  return;  // already on; the backend may re-send as a nudge
              if (!bbox_server->start())
                  spdlog::error("[Bbox] Could not start the overlay server");
          });
          poller->set_on_bbox_off([&bbox_server]() {
              if (!bbox_server || !bbox_server->is_running())
                  return;
              spdlog::info("[Bbox] Stopping the overlay ({} frames dropped)",
                           bbox_server->dropped_frames());
              bbox_server->stop();
          });
          poller->start();
      }

      // Arms a clip covering the seconds either side of the detection whose
      // image was just uploaded, keyed to that image's storage key.
      const auto arm_clip_for_detection = [&clip_recorder](const std::string &detection_key) {
          // Returns false when the event is already covered
          // which is what keeps overlapping detections from producing near-duplicate uploads.
          if (clip_recorder && clip_recorder->on_event(detection_key))
              spdlog::info("[Clip] Detection {} armed a video clip", detection_key);
      };

      // Detections ride the clip cooldown: once one has armed a clip, the next
      // few minutes of detections are the same animal on the same visit, and
      // the clip already covers them -- so they cost neither a clip nor a
      // still. Asked before the upload rather than after, since the upload is
      // the expense the cooldown exists to avoid. Without a clip recorder
      // there is no cooldown, and detection images go out as they always did.
      uint64_t detections_suppressed = 0;
      const auto detection_in_cooldown = [&clip_recorder]() {
          return clip_recorder && clip_recorder->in_cooldown();
      };

      bool send_frame = false;
      // Counts only frames the NPU actually ran on, which is what the overlay
      // publishes; it lets a client spot gaps rather than assume every frame.
      int64_t bbox_frame_id = 0;
      // Backdated so the first preview goes out seconds after startup instead of
      // a full interval which lets the sensor's auto-exposure and white balance settle
      constexpr auto kStartupPreviewDelay = seconds(2);
      auto last_preview_upload = steady_clock::now() - preview_interval + kStartupPreviewDelay;
      // Reused per-frame packing buffer; the callback is the only thread here.
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

              // Debug overlay: publish this frame's tracks, normalized out of
              // the network's input-pixel space (which is what the tracker runs
              // in, since the parser is called with bbox_normalization=false)
              // so a client can scale them to whatever size it renders at.
              if (bbox_server && bbox_server->is_running()) {
                  const double scale = bbox_server->config().inference_input_size;
                  std::vector<byte_track::BboxWsServer::TrackedBox> boxes;
                  boxes.reserve(tracks.size());
                  for (const auto &t : tracks) {
                      const auto tlwh = t->tlwh();
                      byte_track::BboxWsServer::TrackedBox box;
                      box.x = tlwh[0] / scale;
                      box.y = tlwh[1] / scale;
                      box.w = tlwh[2] / scale;
                      box.h = tlwh[3] / scale;
                      box.score = t->score;
                      box.label = t->label;
                      box.track_id = t->track_id;
                      boxes.push_back(box);
                  }
                  bbox_server->broadcast(boxes, bbox_frame_id++);
              }
          }

          // --- H.264 encode ---
          // Done before the JPEG work so a slow upload can't delay the stream.
          if (tee)
              tee->encode(req);

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
          if (backend_client && send_frame)
          {
              if (detection_in_cooldown()) {
                  ++detections_suppressed;
                  spdlog::debug("[Track] Detection suppressed: still inside the {}s clip cooldown",
                                clip_recorder->config().send_video_clip_cooldown);
              } else {
                  const auto object_key = backend_client->upload_object(jpg);
                  if (object_key) {
                      spdlog::info("[Track] Successfully uploaded image to storage.");
                      arm_clip_for_detection(*object_key);
                  }
              }
              send_frame = false;
          }

          const auto now = steady_clock::now();
          if (backend_client && now - last_preview_upload >= preview_interval)
          {
              if (backend_client->upload_object(jpg, "image/jpeg", "CAMERA_PREVIEW")) {
                  spdlog::debug("[Track] Successfully uploaded camera preview image to storage.");
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
      // Before StopCamera, and before the sinks below go away: destroying the
      // tee joins the encoder's threads, so no sink can be called afterwards,
      // and releases the camera buffers it was still holding.
      tee.reset();
      app.StopCamera();
      post_processor.Teardown();
      // Poller first: it is the only thing that starts the streamer, so once it
      // has joined nothing can bring ffmpeg back up underneath the stop below.
      if (poller)
          poller->stop();
      // After the callback can no longer fire, so no frame arrives mid-shutdown.
      if (bbox_server && bbox_server->is_running()) {
          spdlog::info("[Bbox] Stopping the overlay ({} frames dropped)",
                       bbox_server->dropped_frames());
          bbox_server->stop();
      }
      if (streamer && streamer->is_running()) {
          spdlog::info("[Stream] Stopping ffmpeg ({} access units dropped)", streamer->dropped_frames());
          streamer->stop();
      }
      if (clip_recorder) {
          // Detections held off by the cooldown are counted here rather than by
          // the recorder: they never reach it, which is the point.
          spdlog::info("[Clip] Stopping clip recorder ({} uploaded, {} failed, "
                       "{} detections suppressed by the cooldown)",
                       clip_recorder->clips_uploaded(), clip_recorder->clips_failed(),
                       detections_suppressed + clip_recorder->events_suppressed());
          clip_recorder->stop();
      }
      return 0;
  }
