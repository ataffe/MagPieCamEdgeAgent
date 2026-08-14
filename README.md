# MagPieCam Edge Agent
An edge agent designed to run on a Raspberry Pi Zero 2W. The agent uses the [Raspberry Pi AI camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) to 
track moving objects. Each time an object is detected and image is sent to the MagPieCam-Core service and a video clip is uploaded
to S3. Images are uploaded with an exponential backoff while objects are in view.

### System Diagram
![MagPieCam system diagram.](https://github.com/ataffe/MagPieCam-Assets/blob/main/system_diagram/magpie-cam-system-diagram-edge-agent.png?raw=true)
[MagPieCam-Core](https://github.com/ataffe/MagPieCam-Core) - Handles CRUD operations for Users, Cameras, Rules, Notifications and runs jobs.

[MagPieCam-iOS](https://github.com/ataffe/MagPieCam-iOS) - iOS app and main frontend for the system.
