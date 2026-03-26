from src.config import ConfigManager
import json

async def test_save_to_disk(tmp_path):
    config_file_path = tmp_path / "test_config.json"
    config_manager = ConfigManager(str(config_file_path))
    await config_manager.save_to_disk()
    assert config_file_path.exists(), "Config file was not created on disk"

    with open(config_file_path, 'r') as f:
        data = json.load(f)
        assert data['cooldown_seconds'] == 120, "Default cooldown_seconds value is incorrect"
        assert data['motion_threshold'] == 8000, "Default motion_threshold value is incorrect"
        assert data['server_ip_address'] == "10.0.0.124", "Default server_ip_address value is incorrect"
        assert data['server_port'] == 9445, "Default server_port value is incorrect"
        assert data['log_level'] == "INFO", "Default log_level value is incorrect"

async def test_load_from_disk(tmp_path):
    config_file_path = tmp_path / "test_config.json"
    # Create a config file with custom values
    custom_config = {
        "cooldown_seconds": 50,
        "motion_threshold": 9000,
        "server_ip_address": "10.0.0.125",
        "server_port": 9446,
        "log_level": "DEBUG"
    }

    with open(config_file_path, 'w') as f:
        json.dump(custom_config, f)

    config_manager = ConfigManager(str(config_file_path))
    await config_manager.load_from_disk()

    assert config_manager.config.cooldown_seconds == 50, "Loaded cooldown_seconds value is incorrect"
    assert config_manager.config.motion_threshold == 9000, "Loaded motion_threshold value is incorrect"
    assert config_manager.config.server_ip_address == "10.0.0.125", "Loaded server_ip_address value is incorrect"
    assert config_manager.config.server_port == 9446, "Loaded server_port value is incorrect"
    assert config_manager.config.log_level == "DEBUG", "Loaded log_level value is incorrect"

