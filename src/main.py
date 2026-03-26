import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio

from camera import Camera
from config import app_config, CameraConfig

logger = logging.getLogger('Guardian Cam Main')

def set_logging_level(log_level_str):
    log_level_str = log_level_str.upper()
    if log_level_str == "DEBUG":
        logging_level = logging.DEBUG
    elif log_level_str == "INFO":
        logging_level = logging.INFO
    elif log_level_str == "WARNING":
        logging_level = logging.WARNING
    elif log_level_str == "ERROR":
        logging_level = logging.ERROR
    logging.basicConfig(level=logging_level)


camera_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera_task

    logger.info("Loading configuration from disk")
    await app_config.load_from_disk()
    logger.debug(f"Configuration loaded")
    set_logging_level(app_config.config.log_level)

    camera = Camera()
    camera_task = asyncio.create_task(camera.start())

    yield

    camera.stop()
    if camera_task:
        camera_task.cancel()
        try:
            await camera_task
        except asyncio.CancelledError:
            logger.info("Camera task cancelled successfully")

app = FastAPI(lifespan=lifespan)


# --- API Endpoints for configuration management ---
@app.get("/v1/config", response_model=CameraConfig)
async def get_config():
    async with app_config.lock:
        return app_config.config

@app.put("/v1/config", response_model=CameraConfig)
async def update_config(new_config: CameraConfig):
    async with app_config.lock:
        app_config.config = new_config
        await app_config.save_to_disk()
        set_logging_level(app_config.config.log_level)
        logger.info("Configuration updated by user")
        return app_config.config


    




        
    


