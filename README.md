# MagPieCam Edge Agent
An edge agent designed to run on a Raspberry Pi Zero 2W. The agent uses the [Raspberry Pi AI camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) to 
track moving objects. Each time an object is detected and image is sent to the MagPieCam-Core service and a video clip is uploaded
to S3. Images are uploaded with an exponential backoff while objects are in view.

### System Diagram
![MagPieCam system diagram.](https://github.com/ataffe/MagPieCam-Assets/blob/main/system_diagram/magpie-cam-system-diagram-edge-agent.png?raw=true)
[MagPieCam-Core](https://github.com/ataffe/MagPieCam-Core) - The backend for the MagPieCam system.
MagPieCam-Core Handles CRUD operations for Users, Cameras, Rules, and notifications. MagPieCam-Core also facilitates
streaming coordination between users and cameras using a MedaMTX server. The core also contains workers that handle
rules evaluation and trigger push notifications when a rule is fired.

[MagPieCam-iOS](https://github.com/ataffe/MagPieCam-iOS) - An iOS app that enables users to receive smart notifications
based on rules that they set, and video live video from a MagPieCam.


### Bounding-box debug overlay
The stream command poller understands two debug commands alongside `start`/`stop`: `bbox_on` and
`bbox_off`. While the RTSP stream is running, `bbox_on` starts a WebSocket server that publishes the
tracker's output for every inferenced frame, so the boxes can be drawn over the video.

Each message is one frame:
```json
{
  "frame_id": 42,
  "timestamp_ms": 1723651200123,
  "tracks": [
    {"id": 7, "label": 0, "score": 0.91, "x": 0.12, "y": 0.30, "w": 0.08, "h": 0.22}
  ]
}
```
Boxes are normalized to `0.0–1.0`, so a client multiplies by the size it renders the video at.
`label` is the index into the `classes` array in `config/ml/imx500_config.json`; `id` is the
ByteTrack track id, which is what makes ID switches and re-identifications visible.

The overlay strictly follows the stream: `bbox_on` is ignored while the stream is down, and stopping
the stream stops the overlay too. It is unauthenticated and binds to `127.0.0.1:8081` by default —
reach it over an SSH tunnel rather than exposing it. Settings live in the optional `bbox_debug`
section of `config/backend/backend_config.json`; `inference_input_size` there **must** match the
input size of the `.rpk` model, or every box lands in the wrong place.