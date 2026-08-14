# MagPieCam Edge Agent
A camera client designed to run on a Raspberry Pi Zero 2W. The client uses the Raspberry Pi AI camera to 
detect moving objects and sends images to the Scout Cam Event Processor to determine if a user rule should be
triggered.

### System Diagram
![MagPieCam system diagram.](https://github.com/ataffe/MagPieCam-Assets/blob/main/system_diagram/magpie-cam-system-diagram-edge-agent.png?raw=true)
[MagPieCam-Core](https://github.com/ataffe/MagPieCam-Core) - Handles CRUD operations for Users, Cameras, and Rules.

[MagPieCam-iOS](https://github.com/ataffe/MagPieCam-iOS) - iOS app and main frontend for the system.
