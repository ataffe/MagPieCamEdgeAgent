import logging

from pydantic import BaseModel, Field
import json
import asyncio
import os

logger = logging.getLogger('Config Manager')

class CameraConfig(BaseModel):
    cooldown_seconds: int = Field(default=120, ge=0, description="Seconds to wait after a trigger")
    motion_threshold: int = Field(default=8000, ge=7000, le=12000, description="Sensitivity of motion detection")
    server_ip_address: str = Field(default="10.0.0.124", description="IP address of the server")
    server_port: int = Field(default=9445, ge=1, le=65535, description="Port number of the server")
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    
class ConfigManager:
    def __init__(self, config_file_path='config.json'):
        self.config: CameraConfig = CameraConfig()
        self.lock = asyncio.Lock()
        self.config_file_path = config_file_path
        logger.info(f"ConfigManager initialized with config file path: {self.config_file_path}")
        
    
    async def load_from_disk(self):
        logger.debug("Loading configuration from disk...")
        if os.path.exists(self.config_file_path):
            async with self.lock:
                with open(self.config_file_path, 'r') as f:
                    data = json.load(f)
                    self.config = CameraConfig(**data)
        else:
            # Save default config if file doesn't exist
            await self.save_to_disk()
    

    async def save_to_disk(self):
        logger.debug("Saving configuration to disk...")
        async with self.lock:
            with open(self.config_file_path, 'w') as f:
                f.write(self.config.model_dump_json(indent=4))

app_config = ConfigManager()