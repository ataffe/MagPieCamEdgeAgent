// rpicam-apps post-processing stage: runs our C++ ByteTracker on detections
// parsed directly from the IMX500 raw output tensor (rpi::CnnOutputTensor),
// overlays tracked boxes/IDs with OpenCV, and serves the annotated frames as an
// MJPEG HTTP stream (port 8000) - a C++ port of the Python draw + streaming.py.
//
// The built-in imx500_object_detection stage is chained before us only to upload
// the network firmware; we read the raw (dequantized) tensor ourselves because
// that stage mis-scales this custom YOLO11n model's boxes.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <libcamera/control_ids.h>
#include <libcamera/controls.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/buffer_sync.hpp"
#include "core/completed_request.hpp"
#include "core/rpicam_app.hpp"
#include "post_processing_stages/post_processing_stage.hpp"

#include "byte_tracker.h"
#include "imx500_yolo_parser.h"
#include "mjpeg_server.h"

using namespace std::chrono;
using byte_track::BYTETracker;
using byte_track::Object;
using byte_track::Vec4;

namespace {
// YUV420 (I420) planes -> BGR, handling row stride and separate/contiguous planes.
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

class ByteTrackStage : public PostProcessingStage
{
public:
	ByteTrackStage(RPiCamApp *app) : PostProcessingStage(app) {}

	char const *Name() const override { return "byte_track"; }

	void Read(boost::property_tree::ptree const &params) override
	{
		track_thresh_ = params.get<double>("track_thresh", 0.5);
		match_thresh_ = params.get<double>("match_thresh", 0.8);
		track_buffer_ = params.get<int>("track_buffer", 30);
		frame_rate_ = params.get<double>("frame_rate", 30.0);
		bbox_normalization_ = params.get<bool>("bbox_normalization", false);
		input_w_ = params.get<double>("input_w", 640.0);
		input_h_ = params.get<double>("input_h", 480.0);
		stream_port_ = params.get<int>("stream_port", 8000);

		std::string labels_file = params.get<std::string>("labels", "");
		if (!labels_file.empty()) {
			std::ifstream f(labels_file);
			std::string line;
			while (std::getline(f, line)) {
				if (!line.empty()) labels_.push_back(line);
			}
		}
	}

	void Configure() override
	{
		tracker_ = std::make_unique<BYTETracker>(track_thresh_, match_thresh_, track_buffer_, frame_rate_);
		total_frames_ = 0;
		infer_frames_ = 0;
		first_ = true;
		min_ms_ = max_ms_ = sum_ms_ = 0.0;
		last_tracks_.clear();

		if (stream_port_ > 0)
		{
			server_ = std::make_unique<byte_track::MjpegServer>(stream_port_);
			server_->start();
		}
	}

	void Stop() override
	{
		if (server_) server_->stop();
	}

	bool Process(CompletedRequestPtr &completed_request) override
	{
		total_frames_++;

		auto out_ctrl = completed_request->metadata.get(libcamera::controls::rpi::CnnOutputTensor);
		auto info_ctrl = completed_request->metadata.get(libcamera::controls::rpi::CnnOutputTensorInfo);

		// New inference result: step the tracker and refresh the boxes we overlay.
		if (out_ctrl && info_ctrl)
		{
			auto t = *out_ctrl;
			auto inf = *info_ctrl;
			std::vector<Object> objs = byte_track::parse_imx500_detections(
				t.data(), t.size(), inf.data(), inf.size(), bbox_normalization_, input_h_);

			const auto now = steady_clock::now();
			if (infer_frames_ == 0) t_first_ = now;

			const auto t0 = steady_clock::now();
			auto tracks = tracker_->update(objs);
			const auto t1 = steady_clock::now();
			const double ms = duration<double, std::milli>(t1 - t0).count();

			if (first_) { min_ms_ = max_ms_ = ms; first_ = false; }
			else { if (ms < min_ms_) min_ms_ = ms; if (ms > max_ms_) max_ms_ = ms; }
			sum_ms_ += ms;
			infer_frames_++;

			last_tracks_.clear();
			for (const auto &tr : tracks)
				last_tracks_.push_back({tr->tlwh(), tr->track_id, tr->label});

			if (infer_frames_ % 30 == 0)
			{
				const double elapsed = duration<double>(now - t_first_).count();
				const double infer_fps = elapsed > 0 ? (infer_frames_ - 1) / elapsed : 0.0;
				std::fprintf(stderr,
							 "[byte_track] infer_frame %u | dets=%zu tracks=%zu | infer_fps=%.1f cam_fps=%.1f | "
							 "track ms: cur=%.3f min=%.3f max=%.3f avg=%.3f\n",
							 infer_frames_, objs.size(), tracks.size(), infer_fps,
							 completed_request->framerate, ms, min_ms_, max_ms_, sum_ms_ / infer_frames_);
			}
		}

		// Overlay + encode only while a client is watching, so the (software) JPEG
		// encode doesn't steal CPU from inference when nobody is connected.
		if (server_ && server_->has_clients()) draw_and_stream(completed_request);

		return false;
	}

private:
	struct Drawable { Vec4 tlwh; int id; int label; };

	void draw_and_stream(CompletedRequestPtr &completed_request)
	{
		// Prefer the small lores stream for overlay/encode; the main stream is the
		// full sensor resolution (e.g. 2026x1520) and far too heavy to JPEG-encode here.
		libcamera::Stream *stream = app_->LoresStream();
		if (!stream) stream = app_->GetMainStream();
		if (!stream) return;
		auto it = completed_request->buffers.find(stream);
		if (it == completed_request->buffers.end()) return;

		StreamInfo si = app_->GetStreamInfo(stream);
		const int w = static_cast<int>(si.width) & ~1;   // YUV420 needs even dims
		const int h = static_cast<int>(si.height) & ~1;
		if (w <= 0 || h <= 0) return;

		BufferReadSync r(app_, it->second);
		const auto &planes = r.Get();
		if (planes.empty()) return;

		cv::Mat bgr;
		yuv420_to_bgr(planes, w, h, si.stride, bgr);

		const double sx = w / input_w_;
		const double sy = h / input_h_;
		for (const auto &d : last_tracks_)
		{
			cv::Rect box(cvRound(d.tlwh[0] * sx), cvRound(d.tlwh[1] * sy),
						 cvRound(d.tlwh[2] * sx), cvRound(d.tlwh[3] * sy));
			cv::rectangle(bgr, box, cv::Scalar(0, 255, 0), 2);
			std::string cls = (d.label >= 0 && d.label < static_cast<int>(labels_.size()))
				? labels_[d.label] : std::to_string(d.label);
			std::string label = "Id: " + std::to_string(d.id) + " " + cls;
			cv::Point org(box.x + 4, box.y + 16);
			cv::putText(bgr, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
			cv::putText(bgr, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 206, 138), 1, cv::LINE_AA);
		}

		std::vector<uint8_t> jpg;
		cv::imencode(".jpg", bgr, jpg, {cv::IMWRITE_JPEG_QUALITY, 80});
		server_->set_frame(jpg.data(), jpg.size());
	}

	double track_thresh_ = 0.5;
	double match_thresh_ = 0.8;
	int track_buffer_ = 30;
	double frame_rate_ = 30.0;
	bool bbox_normalization_ = false;
	double input_w_ = 640.0;
	double input_h_ = 480.0;
	int stream_port_ = 8000;
	std::vector<std::string> labels_;

	std::unique_ptr<BYTETracker> tracker_;
	std::unique_ptr<byte_track::MjpegServer> server_;
	std::vector<Drawable> last_tracks_;

	unsigned total_frames_ = 0;
	unsigned infer_frames_ = 0;
	steady_clock::time_point t_first_;
	bool first_ = true;
	double min_ms_ = 0.0, max_ms_ = 0.0, sum_ms_ = 0.0;
};

static PostProcessingStage *Create(RPiCamApp *app) { return new ByteTrackStage(app); }
static RegisterStage reg("byte_track", &Create);
