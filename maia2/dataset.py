"""Dataset loading utilities for MAIA2.

Provides functions to download and load chess datasets containing
positions, moves, and Elo ratings for training and testing.
"""

import os
from typing import Final

import gdown  # type: ignore
import pandas as pd

# Constants
DEFAULT_SAVE_ROOT: Final[str] = "./maia2_data"
TEST_DATASET_URL: Final[str] = (
    "https://drive.google.com/uc?id=1fSu4Yp8uYj7xocbHAbjBP6DthsgiJy9X"
)
TRAIN_DATASET_URL: Final[str] = (
    "https://drive.google.com/uc?id=1XBeuhB17z50mFK4tDvPG9rQRbxLSzNqB"
)


def load_example_test_dataset(
    save_root: str = DEFAULT_SAVE_ROOT,
) -> pd.DataFrame:
    """Download and load example test dataset.

    Args:
        save_root: Directory to save dataset.

    Returns:
        DataFrame with columns [board, move, active_elo, opponent_elo].
        Filtered to positions after move 10.

    Raises:
        OSError: If download or directory creation fails.
        pd.errors.EmptyDataError: If dataset is empty or corrupted.
    """

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    output_path = os.path.join(save_root, "example_test_dataset.csv")

    if os.path.exists(output_path):
        print("Example test dataset already downloaded.")
    else:
        gdown.download(TEST_DATASET_URL, output_path, quiet=False)
        print("Example test dataset downloaded.")

    data = pd.read_csv(output_path)
    filtered_data = data[data.move_ply > 10][
        ["board", "move", "active_elo", "opponent_elo"]
    ]

    return filtered_data


def load_example_train_dataset(save_root: str = DEFAULT_SAVE_ROOT) -> str:
    """Download example training dataset.

    Args:
        save_root: Directory to save dataset.

    Returns:
        Path to downloaded training dataset CSV file.

    Raises:
        OSError: If download or directory creation fails.
    """
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    output_path = os.path.join(save_root, "example_train_dataset.csv")

    if os.path.exists(output_path):
        print("Example train dataset already downloaded.")
    else:
        gdown.download(TRAIN_DATASET_URL, output_path, quiet=False)
        print("Example train dataset downloaded.")

    return output_path
