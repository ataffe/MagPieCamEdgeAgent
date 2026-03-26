import asyncio
import logging
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput
from libcamera import controls
import cv2
import time

from motion_detection import is_motion_detected
from client import send_video_clip
from config import app_config

logger = logging.getLogger('Guardian Cam Camera')

class Camera:
    def __init__(self):
        self.picam = Picamera2(tuning="/usr/share/libcamera/ipa/rpi/vc4/imx219_af.json")
        self.picam.configure(self.picam.create_preview_configuration(main={"format": "BGR888", "size": (1920, 1080)}))
        self.picam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        self.picam.controls.FrameRate = 30

        self.h264_encoder = H264Encoder(bitrate=1_000_000, repeat=True)
        # Keep last 10 frames in buffer for motion clip recording
        self.h264_encoder.output = CircularOutput(buffersize=10)

    async def start(self):
        self.picam.start()
        self.picam.start_encoder(self.h264_encoder)
        reference_frame = None
        reference_frame_capture_time = None
        last_trigger_time = None
        logger.info("Guardian Cam Camera started")
        while True:
            frame = await asyncio.to_thread(self.picam.capture_array)
            # Refresh reference frame every 30 seconds to adapt to stuff changing.
            if reference_frame is None or time.perf_counter() - reference_frame_capture_time > 30:
                reference_frame = frame
                reference_frame_capture_time = time.perf_counter()
                continue
            
            # Wait for cooldown before checking for motion again to avoid sending too many clips in a short time.
            if last_trigger_time is not None and time.perf_counter() - last_trigger_time < app_config.config.cooldown_seconds:
                await asyncio.sleep(0.1)
                continue

            if is_motion_detected(frame, reference_frame, app_config.config.motion_threshold):
                # send_img_for_prediction(frame, app_config.config.server_ip_address, app_config.config.server_port)
                send_video_clip(self.h264_encoder, app_config.config.server_ip_address, app_config.config.server_port)
                last_trigger_time = time.perf_counter()
                logger.info(f"Motion detected, cooling down for {app_config.config.cooldown_seconds} seconds")
            
            # Yield control back to api event loop.
            await asyncio.sleep(0.05)


    def stop(self):
        self.picam.stop()
        self.picam.stop_encoder()
        logger.info("Guardian Cam Client stopped by user")