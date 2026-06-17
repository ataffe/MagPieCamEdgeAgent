
import argparse
import sys
import time
from functools import lru_cache
import cv2
from functools import partial
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import numpy as np

from streaming import *
from bytetrack.byte_tracker import BYTETracker
from parsers.IMX500YOLOBoundingBoxParser import IMX500YOLOBoundingBoxParser

bounding_box_parser = None

# FPS tracking (smoothed over recent frames)
_fps_last_time = None
_fps_frame_count = 0
_fps_smoothed = None

@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    global bounding_box_parser
    global _fps_last_time, _fps_frame_count, _fps_smoothed
    if not bounding_box_parser:
        raise RuntimeError('Parser not initialized')

    now = time.monotonic()
    if _fps_last_time is not None:
        instant_fps = 1.0 / (now - _fps_last_time)
        # exponential moving average to smooth out jitter
        _fps_smoothed = instant_fps if _fps_smoothed is None else 0.9 * _fps_smoothed + 0.1 * instant_fps
    _fps_last_time = now
    _fps_frame_count += 1
    if _fps_smoothed is not None and _fps_frame_count % 30 == 0:
        print(f'fps: {_fps_smoothed:.1f}')

    metadata = request.get_metadata()
    detections = bounding_box_parser.parse_detections(metadata)
    labels = get_labels()
    with MappedArray(request, stream) as m:
        for detection in detections:
            x1, y1, w, h = detection.box.astype(int)
            try:
                label = f"{labels[int(detection.category)]} | id: {detection.id}"
            except Exception as e:
                print(f"Error getting label for category {detection.category}, id {detection.id}")
                raise e

            # Calculate text position
            text_x = x1 + 5
            text_y = y1 + 15

            # Draw text: black stroke underneath, green fill on top
            cv2.putText(m.array, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(m.array, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (138, 206, 0), 1, cv2.LINE_AA)

            # Draw detection box
            cv2.rectangle(m.array, (x1, y1), (x1 + w, y1 + h), (0, 255, 0, 0), thickness=2)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Path of the model",
        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
    )
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument(
        "--bbox-order", choices=["yx", "xy"], default="yx",
        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)"
    )
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"], default=None, help="Run post process of type")
    parser.add_argument(
        "-r",
        "--preserve-aspect-ratio",
        action=argparse.BooleanOptionalAction,
        help="preserve the pixel aspect ratio of the input tensor",
    )
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true", help="Print JSON network_intrinsics then exit")
    # Tracking Args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    image_width_height = (640, 480)
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": image_width_height, "format": "RGB888"},
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12)

    tracker = BYTETracker(args, intrinsics.inference_rate)
    bounding_box_parser = IMX500YOLOBoundingBoxParser(
        imx500=imx500,
        intrinsics=intrinsics,
        tracker=tracker,
        img_wh=image_width_height
    )

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    picam2.pre_callback = draw_detections

    output = StreamingOutput()
    picam2.start_recording(MJPEGEncoder(), FileOutput(output))
    streaming_handler = partial(StreamingHandler, output=output)
    server_thread = StreamingServer(('', 8000), streaming_handler)
    import threading

    print("Starting streaming server...")
    threading.Thread(target=server_thread.serve_forever, daemon=True).start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Cleaning up...")
        picam2.stop_recording()
        picam2.stop_preview()
        picam2.stop()
        print("Done")
