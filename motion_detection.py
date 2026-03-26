import logging
import cv2

logger = logging.getLogger('Guardian Cam Motion Detection')

def is_motion_detected(img, ref_frame, min_contour_area=6000):
    if img is None:
        logger.error("Image passed to motion detection is None")
        return False
    
    if ref_frame is None:
        logger.error("Reference frame passed to motion detection is None")
        return False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    ref_frame = cv2.GaussianBlur(ref_frame, (21, 21), 0)

    frame_delta = cv2.absdiff(ref_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = len([contour for contour in contours if cv2.contourArea(contour) >= min_contour_area])
    return contours > 0