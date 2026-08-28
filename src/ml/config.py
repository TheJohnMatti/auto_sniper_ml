"""Loads the shared project configuration from config.yaml."""
import functools
import os

import yaml

CONFIG_PATH = os.environ.get("AUTO_SNIPER_CONFIG", "config.yaml")


@functools.lru_cache(maxsize=1)
def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ml_config(path: str = CONFIG_PATH) -> dict:
    return load_config(path)["ml_pipeline"]
