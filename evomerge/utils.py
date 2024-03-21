import importlib
import json
import os
import yaml
import random
import numpy as np
import torch


def default(val, d):
    if val is not None:
        return val
    return d


def load_config(config_path: str) -> dict:
    ext = os.path.splitext(config_path)[-1]
    if ext in [".yaml", ".yml"]:
        with open(config_path, "r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp)
    elif ext == ".json":
        with open(config_path) as fp:
            config = json.load(fp)
    else:
        raise RuntimeError
    return config


# copied from https://github.com/Stability-AI/generative-models/blob/main/sgm/util.py#L168
def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# copied from https://github.com/Stability-AI/generative-models/blob/main/sgm/util.py#L168
def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``.

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
