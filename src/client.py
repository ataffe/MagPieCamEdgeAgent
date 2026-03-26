
import logging
import cv2
import requests
import time
import os
from config import app_config

logger = logging.getLogger('Guardian Cam Client')

def send_img_for_prediction(img, server_ip_address, server_port):
    img_resized = cv2.resize(img, (1280, 720))
    success, buffer = cv2.imencode(".jpg", img_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])

    if not success:
        logger.error("Error: Could not encode the image to JPEG")
        return False

    files = {'file': ('image.jpg', buffer.tobytes(), 'image/jpeg')}
    try:
        response = requests.post(f"http://{server_ip_address}:{server_port}/predict", files=files, timeout=10)

        if response.status_code == 201:
            logger.info("Image sent successfully")
            return True
        else:
            logger.error(f"Failed to send image. Status code: {response.status_code}, Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending image: {e}")
        return False
    
def send_video_clip(encoder, server_ip_address, server_port, clip_duration_seconds=5):
    tmp_video_path = 'temp_motion_clip.mp4'
    encoder.output.fileoutput = tmp_video_path
    encoder.output.start()
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < clip_duration_seconds:
        time.sleep(0.1)
    encoder.output.stop()
    logger.info("Video clip recording finished")

    logger.info("Sending video clip to server...")
    with open(tmp_video_path, 'rb') as video_file:
        files = {'file': ('motion_clip.mp4', video_file, 'video/mp4')}
        try:
            response = requests.post(f"http://{server_ip_address}:{server_port}/upload_motion_clip", files=files, timeout=30)

            if response.status_code == 201:
                logger.info("Video clip sent successfully")
                return True
            else:
                logger.error(f"Failed to send video clip. Status code: {response.status_code}, Response: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending video clip: {e}")
            return False
        finally:
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)
                logger.debug("Temporary video file removed")