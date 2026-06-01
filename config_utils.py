# coding: UTF-8

import os
import random
from typing import Optional

import numpy as np
import torch


DEFAULT_MODEL_ROOT = "/home/students/wzj_4090_2/code_server/..bert-model/"
DEFAULT_DATA_ROOT = "/home/students/wzj_4090_2/code_server/A_paper4/data/"


def resolve_model_path(model_root_path: Optional[str], model_name: str) -> str:
    """
    Prefer a local model directory. If it does not exist, return model_name so
    transformers can try its normal cache / HuggingFace resolution.
    """
    if model_root_path:
        candidate = os.path.join(model_root_path, model_name)
        if os.path.isdir(candidate):
            return candidate
    return model_name


def pick_device(gpu_id: int):
    """Return cuda:<gpu_id> when available, otherwise CPU; fallback to cuda:0 if needed."""
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n <= 0:
            return torch.device("cpu")
        real_gpu = int(gpu_id) if int(gpu_id) < n else 0
        return torch.device(f"cuda:{real_gpu}")
    return torch.device("cpu")


def set_global_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
