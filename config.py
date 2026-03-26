import logging

from pydantic import BaseModel, Field
import json
import asyncio
import os

CONFIG_FILE = "config.json"

logger = logging.getLogger('Config Manager')

class CameraConfig(BaseModel):
    cooldown_seconds: int = Field(default=120, ge=0, description="Seconds to wait after a trigger")
    motion_threshold: int = Field(default=25, ge=7000, le=12000, description="Sensitivity of motion detection")
    server_ip_address: str = Field(default="10.0.0.124", description="IP address of the server")
    server_port: int = Field(default=9445, ge=1, le=65535, description="Port number of the server")
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    
class ConfigManager:
    def __init__(self):
        self.config: CameraConfig = CameraConfig()
        self.lock = asyncio.Lock()
    
    async def load_from_disk(self):
        logger.debug("Loading configuration from disk...")
        if os.path.exists(CONFIG_FILE):
            async with self.lock:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    self.config = CameraConfig(**data)
        else:
            # Save default config if file doesn't exist
            await self.save_to_disk()
    

    async def save_to_disk(self):
        logger.debug("Saving configuration to disk...")
        async with self.lock:
            with open(CONFIG_FILE, 'w') as f:
                f.write(self.config.model_dump_json(indent=4))

app_config = ConfigManager()