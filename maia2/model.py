"""Model loading utilities for MAIA2.

Provides functions to load pre-trained MAIA2 models
for blitz and rapid time controls.
"""

import os
import warnings
from typing import Final, Literal

import gdown  # type: ignore
import torch
from torch import nn

from .main import MAIA2Model
from .utils import create_elo_dict, get_all_possible_moves, parse_args

warnings.filterwarnings("ignore")

# Constants
DEFAULT_SAVE_ROOT: Final[str] = "./maia2_models"
CONFIG_URL: Final[str] = (
    "https://drive.google.com/uc?id=1GQTskYMVMubNwZH2Bi6AmevI15CS6gk0"
)
MODEL_BLITZ_URL: Final[str] = (
    "https://drive.google.com/uc?id=1X-Z4J3PX3MQFJoa8gRt3aL8CIH0PWoyt"
)
MODEL_RAPID_URL: Final[str] = (
    "https://drive.google.com/uc?id=1gbC1-c7c0EQOPPAVpGWubezeEW8grVwc"
)


def from_pretrained(
    model_type: Literal["blitz", "rapid"],
    device: Literal["gpu", "cpu"],
    save_root: str = DEFAULT_SAVE_ROOT,
) -> MAIA2Model:
    """Load pre-trained MAIA2 model.

    Args:
        model_type: Type of model ("blitz" or "rapid").
        device: Device to load on ("gpu" or "cpu").
        save_root: Directory to save model files.

    Returns:
        Loaded MAIA2 model.

    Raises:
        ValueError: If model_type is invalid.
        OSError: If download or directory creation fails.
        RuntimeError: If model loading fails.
    """
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    if model_type == "blitz":
        url = MODEL_BLITZ_URL
        output_path = os.path.join(save_root, "blitz_model.pt")

    elif model_type == "rapid":
        url = MODEL_RAPID_URL
        output_path = os.path.join(save_root, "rapid_model.pt")

    else:
        raise ValueError("Invalid model type. Choose between 'blitz' and 'rapid'.")

    if os.path.exists(output_path):
        print(f"Model for {model_type} games already downloaded.")
    else:
        print(f"Downloading model for {model_type} games.")
        gdown.download(url, output_path, quiet=False)

    cfg_path = os.path.join(save_root, "config.yaml")
    if not os.path.exists(cfg_path):
        gdown.download(CONFIG_URL, cfg_path, quiet=False)

    cfg = parse_args(cfg_path)
    all_moves = get_all_possible_moves()
    elo_dict = create_elo_dict()

    maia2_model = MAIA2Model(len(all_moves), elo_dict, cfg)
    model = nn.DataParallel(maia2_model)

    checkpoint = torch.load(output_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model_module = model.module

    if device == "gpu":
        model_module = model_module.cuda()

    print(f"Model for {model_type} games loaded to {device}.")

    return model_module
