"""Config loader — converts YAML to a dot-access object."""

import yaml
from pathlib import Path


class ConfigNode:
    """Dot-access config object. cfg.data.batch_size just works."""

    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigNode(v))
            elif isinstance(v, list):
                setattr(self, k, [
                    ConfigNode(i) if isinstance(i, dict) else i for i in v
                ])
            else:
                setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def __repr__(self):
        return f"ConfigNode({self.__dict__})"

    def __iter__(self):
        return iter(self.__dict__.keys())

    def dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigNode):
                result[k] = v.dict()
            elif isinstance(v, list):
                result[k] = [i.dict() if isinstance(i, ConfigNode) else i for i in v]
            else:
                result[k] = v
        return result


def load_config(path: str) -> ConfigNode:
    """Load YAML config and return dot-access ConfigNode."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ConfigNode(raw)


def override_config(cfg: ConfigNode, overrides: dict) -> ConfigNode:
    """
    Apply flat-key overrides to a config.
    e.g. override_config(cfg, {"distillation.enabled": False})
    """
    raw = cfg.dict()
    for key_path, value in overrides.items():
        parts = key_path.split(".")
        node = raw
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return ConfigNode(raw)
