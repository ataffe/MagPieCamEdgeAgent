import argparse
import sys
from functools import lru_cache
import cv2
from functools import partial
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from streaming import *
from bytetrack.byte_tracker import BYTETracker, STrack

last_detections = []

printed = False

class Detection:
    def __init__(self, coords, category, id):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        # self.box = imx500.convert_inference_coords(coords, metadata, picam2)
        self.box = coords
        self.id = id


def parse_detections(metadata: dict, intrinsics: NetworkIntrinsics, tracker: BYTETracker, img_wh: tuple[int, int]):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections, printed
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0], conf=threshold, iou_thres=iou, max_out_dets=max_detections
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes

        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    # boxes = np.array([imx500.convert_inference_coords(box, metadata, picam2) for box in boxes]).astype(np.float64)
    tracks = tracker.update(scores=np.squeeze(scores), bboxes=boxes, categories=classes, img_wh=img_wh)

    last_detections = [
        Detection(coords=track.tlwh, category=track.category, id=track.track_id) for track in tracks
    ]
    # last_detections = [
    #     Detection(box, category, score, metadata) for box, score, category in zip(boxes, scores, classes) if score > threshold
    # ]

    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    detections = last_results
    if detections is None:
        return
    labels = get_labels()
    with MappedArray(request, stream) as m:
        for detection in detections:
            x, y, w, h = detection.box.astype(int)
            label = f"{labels[int(detection.category)]} | id: {detection.id}"

            # Calculate text size and position
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            global printed
            if not printed:
                print(f'Output Images Shape: {m.array.shape}')
                printed = True

            overlay = m.array.copy()

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            img = Image.fromarray(m.array)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default(20)
            draw.text(
                (text_x, text_y),
                label,
                font=font,
                fill=(138, 206, 0),
                stroke_width=1,
                stroke_fill=(0,0,0)
            )
            m.array[:] = np.array(img)

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)

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
        "--bbox-order", choices=["yx", "xy"], default="yx", help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)"
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

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections

    output = StreamingOutput()
    picam2.start_recording(MJPEGEncoder(), FileOutput(output))
    streaming_handler = partial(StreamingHandler, output=output)
    server_thread = StreamingServer(('', 8000), streaming_handler)
    import threading

    print("Starting streaming server...")
    threading.Thread(target=server_thread.serve_forever, daemon=True).start()

    last_results = None
    try:
        while True:
            last_results = parse_detections(picam2.capture_metadata(), intrinsics, tracker, image_width_height)
    except KeyboardInterrupt:
        print("Cleaning up...")
        picam2.stop_recording()
        picam2.stop_preview()
        picam2.stop()
        print("Done")
