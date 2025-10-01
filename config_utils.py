import json
import os
from typing import Any, Dict


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
