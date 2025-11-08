"""Inference functions for MAIA2 model.

Provides preprocessing, dataset creation, and inference utilities
for running predictions on chess positions.
"""

from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.utils.data

from .main import MAIA2Model
from .utils import (
    BoardPosition,
    ChessMove,
    EloRangeDict,
    EloRating,
    MovesDict,
    ReverseMovesDict,
)

TestDatasetItem = Tuple[BoardPosition, torch.Tensor,
                        EloRating, EloRating, torch.Tensor]
DictMoveProb = Dict[ChessMove, float]
PreparedDicts = Tuple[MovesDict, EloRangeDict, ReverseMovesDict]
DeprecatedPreparedDicts = List[MovesDict | EloRangeDict | ReverseMovesDict]


def preprocessing(
    fen: BoardPosition,
    elo_self: EloRating,
    elo_oppo: EloRating,
    elo_dict: EloRangeDict,
    all_moves_dict: MovesDict,
) -> Tuple[torch.Tensor, EloRating, EloRating, torch.Tensor]:
    """Preprocess FEN and Elo ratings into model tensors.

    Args:
        fen: FEN string of chess position.
        elo_self: Elo rating of active player.
        elo_oppo: Elo rating of opponent.
        elo_dict: Mapping of Elo ratings to categories.
        all_moves_dict: Mapping of moves to indices.

    Returns:
        Tuple of (board_tensor, elo_self_cat, elo_oppo_cat, legal_moves_mask).
    """
    ...


class TestDataset(torch.utils.data.Dataset[TestDatasetItem]):
    """PyTorch Dataset for MAIA2 test data."""

    all_moves_dict: MovesDict
    data: List[Tuple]
    elo_dict: EloRangeDict

    def __init__(
        self,
        data: pd.DataFrame,
        all_moves_dict: MovesDict,
        elo_dict: EloRangeDict,
    ) -> None:
        """Initialize dataset.

        Args:
            data: DataFrame with [fen, move, elo_self, elo_oppo].
            all_moves_dict: UCI moves to model indices.
            elo_dict: Raw Elo to binned categories.
        """
        ...

    def __len__(self) -> int:
        """Return number of samples."""
        ...

    def __getitem__(self, idx: int) -> TestDatasetItem:
        """Get preprocessed tensors for position.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (fen, board_tensor, elo_self_cat, elo_oppo_cat, legal_moves_mask).
        """
        ...


def get_preds(
    model: MAIA2Model,
    dataloader: torch.utils.data.DataLoader,
    all_moves_dict_reversed: ReverseMovesDict,
) -> Tuple[List[DictMoveProb], List[float]]:
    """Compute move and win probabilities for dataset.

    Args:
        model: Trained MAIA2 model.
        dataloader: DataLoader yielding test data.
        all_moves_dict_reversed: Move indices to UCI strings.

    Returns:
        Tuple of (move_probs_list, win_probs_list).
        move_probs_list: List of dicts mapping UCI moves to probabilities.
        win_probs_list: List of win probabilities.
    """
    ...


def inference_batch(
    data: pd.DataFrame,
    model: MAIA2Model,
    verbose: bool,
    batch_size: int,
    num_workers: int,
) -> Tuple[pd.DataFrame, float]:
    """Run inference on batch of chess positions.

    Args:
        data: DataFrame with [fen, move, elo_self, elo_oppo].
        model: Trained MAIA2 model.
        verbose: Show progress bar if True.
        batch_size: Batch size for DataLoader.
        num_workers: Number of DataLoader workers.

    Returns:
        Tuple of (updated_dataframe, accuracy).
        updated_dataframe: Input data with added win_probs and move_probs columns.
        accuracy: Move prediction accuracy.
    """
    ...


def prepare() -> DeprecatedPreparedDicts:
    """Initialize dictionaries for model inference.

    Returns:
        List of [all_moves_dict, elo_dict, all_moves_dict_reversed].
        all_moves_dict: UCI moves to indices.
        elo_dict: Raw Elo to categories.
        all_moves_dict_reversed: Indices to UCI moves.
    """
    ...


def inference_each(
    model: MAIA2Model,
    prepared: PreparedDicts | DeprecatedPreparedDicts,
    fen: BoardPosition,
    elo_self: EloRating,
    elo_oppo: EloRating,
) -> Tuple[DictMoveProb, float]:
    """Analyze single chess position with MAIA2.

    Args:
        model: Trained MAIA2 model.
        prepared: Tuple from prepare() with mapping dicts.
        fen: FEN string of position.
        elo_self: Elo of player to move.
        elo_oppo: Elo of opponent.

    Returns:
        Tuple of (move_probs, win_prob).
        move_probs: Dict mapping UCI moves to probabilities (sorted).
        win_prob: Win probability (0-1).
    """
    ...
