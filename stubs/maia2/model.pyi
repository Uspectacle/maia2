"""Model loading utilities for MAIA2.

Provides functions to load pre-trained MAIA2 models
for blitz and rapid time controls.
"""

from typing import Literal

from .main import MAIA2Model


def from_pretrained(
    model_type: Literal["blitz", "rapid"],
    device: Literal["gpu", "cpu"],
    save_root: str = ...,
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
    ...
