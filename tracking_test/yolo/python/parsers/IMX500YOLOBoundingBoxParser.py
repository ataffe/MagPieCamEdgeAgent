from typing import Tuple, List

from picamera2.devices.imx500 import NetworkIntrinsics
from picamera2.devices import IMX500
from bytetrack.byte_tracker import BYTETracker
import numpy as np
import time

class Detection:
    def __init__(self, coords, category, id):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.box = coords
        self.id = id

class IMX500YOLOBoundingBoxParser:
    def __init__(self, imx500: IMX500, intrinsics: NetworkIntrinsics, tracker: BYTETracker, img_wh: Tuple[int, int] = (640, 480)):
        self.intrinsics = intrinsics
        self.tracker = tracker
        self.img_wh = img_wh
        self.imx500 = imx500
        self.prev_detections = []
        self.stale_after = 0.5
        self.last_update = None

        self.min_tracking_time = None
        self.max_tracking_time = None
        self.avg_tracking_time = 0
        self.num_iterations = 0

    def parse_detections(self, metadata: dict) -> List[Detection]:
        output_tensors = self.imx500.get_outputs(metadata, add_batch=True)
        stale = time.monotonic() - self.last_update > self.stale_after if self.last_update else True
        if not output_tensors and not stale:
            return self.prev_detections
        elif not output_tensors:
            return []

        num_boxes = int(output_tensors[3][0][0])
        boxes = output_tensors[0][0][:num_boxes]
        scores = np.squeeze(output_tensors[1][0][:num_boxes])
        categories = output_tensors[2][0][:num_boxes]
        boxes = np.maximum(boxes, 0)
        input_w, input_h = self.imx500.get_input_size()

        if self.intrinsics.bbox_normalization:
            boxes = boxes / input_h

        start = time.perf_counter()
        tracks = self.tracker.update(scores=scores, bboxes=boxes, categories=categories, img_wh=self.img_wh)
        diff = (time.perf_counter() - start) * 1000
        self.min_tracking_time = min(diff, self.min_tracking_time) if self.min_tracking_time else diff
        self.max_tracking_time = max(diff, self.max_tracking_time) if self.max_tracking_time else diff
        self.avg_tracking_time += diff if self.avg_tracking_time else diff
        self.num_iterations += 1
        if self.num_iterations % 30 == 0:
            print(f'track update took: {diff:.2f}ms | min tracking time: {self.min_tracking_time:.2f}ms | max tracking time: {self.max_tracking_time:.2f}ms | avg tracking time: {self.avg_tracking_time / self.num_iterations:.2f}ms')

        detections = [
            Detection(coords=track.tlwh, category=track.category, id=track.track_id) for track in tracks
        ]
        self.prev_detections = detections
        self.last_update = time.monotonic()
        return detections
