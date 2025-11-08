"""Dataset loading utilities for MAIA2.

Provides functions to download and load chess datasets containing
positions, moves, and Elo ratings for training and testing.
"""

import pandas as pd


def load_example_test_dataset(save_root: str = ...) -> pd.DataFrame:
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
    ...


def load_example_train_dataset(save_root: str = ...) -> str:
    """Download example training dataset.

    Args:
        save_root: Directory to save dataset.

    Returns:
        Path to downloaded training dataset CSV file.

    Raises:
        OSError: If download or directory creation fails.
    """
    ...
