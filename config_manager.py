import os
import yaml
from typing import Any, Optional

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            self.config = {}
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            self.config = {}

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

def get_config(config_path: Optional[str] = None) -> ConfigManager:
    return ConfigManager(config_path)
