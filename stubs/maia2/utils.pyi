"""Utility functions for MAIA2.

Provides functions for chess position handling, Elo rating mapping,
model configuration, and data processing utilities.
"""

from typing import Dict, Generator, List, Optional, Tuple, TypeVar, Union

import chess
import torch

BoardPosition = str
ChessMove = str
EloRating = int
TimeSeconds = float
FileOffset = int
EloRangeDict = Dict[str, int]
ConfigDict = Dict[str, Union[str, int, float, bool]]
MovesDict = Dict[ChessMove, int]
ReverseMovesDict = Dict[int, ChessMove]
Chunk = Tuple[FileOffset, FileOffset]
SideInfo = Tuple[torch.Tensor, torch.Tensor]


class Config:
    """Dynamic configuration container for MAIA2."""

    input_channels: int
    elo_dim: int
    dim_cnn: int
    dim_vit: int
    num_blocks_cnn: int
    num_blocks_vit: int
    vit_length: int
    batch_size: Optional[int]
    chunk_size: int
    verbose: Optional[bool]
    start_year: int
    end_year: int
    start_month: int
    end_month: int
    first_n_moves: int
    clock_threshold: float
    max_ply: Optional[int]
    data_root: str
    num_workers: int
    side_info: Optional[int]
    side_info_coefficient: Optional[float]
    value: Optional[int]
    value_coefficient: Optional[float]
    seed: int
    num_cpu_left: int
    lr: float
    wd: float
    from_checkpoint: Optional[bool]
    checkpoint_epoch: Optional[int]
    checkpoint_year: Optional[str]
    checkpoint_month: Optional[str]
    max_epochs: int
    queue_length: int
    max_games_per_elo_range: int

    def __init__(self, config_dict: ConfigDict) -> None:
        """Initialize from dictionary.

        Args:
            config_dict: Configuration key-value pairs.
        """
        ...


def parse_args(cfg_file_path: str) -> Config:
    """Parse YAML configuration file.

    Args:
        cfg_file_path: Path to YAML config file.

    Returns:
        Config object with settings.

    Raises:
        OSError: If file cannot be read.
        yaml.YAMLError: If YAML is malformed.
    """
    ...


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Seed value for all RNGs.
    """
    ...


def delete_file(filename: str) -> None:
    """Delete file if it exists.

    Args:
        filename: Path to file.
    """
    ...


def readable_num(num: int) -> str:
    """Convert large number to readable format.

    Args:
        num: Number to format.

    Returns:
        Formatted string with suffix (K/M/B).
    """
    ...


def readable_time(elapsed_time: TimeSeconds) -> str:
    """Format elapsed time in readable format.

    Args:
        elapsed_time: Duration in seconds.

    Returns:
        Formatted time string (e.g., "1h 30m 45.50s").
    """
    ...


def count_parameters(model: torch.nn.Module) -> str:
    """Count trainable parameters in model.

    Args:
        model: PyTorch model.

    Returns:
        Formatted parameter count string.
    """
    ...


def create_elo_dict() -> EloRangeDict:
    """Create Elo rating ranges to category indices mapping.

    Returns:
        Dict mapping Elo range strings to indices.
    """
    ...


def map_to_category(elo: EloRating, elo_dict: EloRangeDict) -> int:
    """Map Elo rating to category index.

    Args:
        elo: Player's Elo rating.
        elo_dict: Elo ranges to indices mapping.

    Returns:
        Category index for the rating.

    Raises:
        TypeError: If elo is not integer.
        ValueError: If elo cannot be categorized.
    """
    ...


def get_side_info(
    board: chess.Board, move_uci: ChessMove, all_moves_dict: MovesDict
) -> SideInfo:
    """Generate feature vectors for chess move.

    Args:
        board: Current chess position.
        move_uci: Move in UCI format.
        all_moves_dict: UCI moves to indices.

    Returns:
        Tuple of (legal_moves_mask, side_info_vector).
    """
    ...


def extract_clock_time(comment: str) -> Optional[int]:
    """Extract remaining clock time from PGN comment.

    Args:
        comment: PGN comment string.

    Returns:
        Remaining time in seconds, or None if not found.
    """
    ...


def read_or_create_chunks(pgn_path: str, cfg: Config) -> List[Chunk]:
    """Load or create file offset chunks for PGN.

    Args:
        pgn_path: Path to PGN file.
        cfg: Configuration with chunk_size.

    Returns:
        List of (start_offset, end_offset) tuples.

    Raises:
        OSError: If file access fails.
    """
    ...


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Convert chess position to feature tensor.

    Args:
        board: Chess position.

    Returns:
        Tensor [18, 8, 8] with board representation.
    """
    ...


def generate_pawn_promotions() -> List[ChessMove]:
    """Generate all possible pawn promotion moves.

    Returns:
        List of UCI promotion moves.
    """
    ...


def mirror_square(square: str) -> str:
    """Mirror chess square vertically.

    Args:
        square: Square in algebraic notation.

    Returns:
        Mirrored square.
    """
    ...


def mirror_move(move_uci: ChessMove) -> ChessMove:
    """Mirror chess move vertically.

    Args:
        move_uci: Move in UCI notation.

    Returns:
        Mirrored move in UCI notation.
    """
    ...


def get_chunks(pgn_path: str, chunk_size: int) -> List[Chunk]:
    """Divide PGN file into chunks by game count.

    Args:
        pgn_path: Path to PGN file.
        chunk_size: Target games per chunk.

    Returns:
        List of (start_offset, end_offset) tuples.

    Raises:
        ValueError: If PGN format is invalid.
        OSError: If file cannot be read.
    """
    ...


def decompress_zst(file_path: str, decompressed_path: str) -> None:
    """Decompress Zstandard (.zst) file.

    Args:
        file_path: Path to .zst file.
        decompressed_path: Output path for decompressed file.

    Raises:
        OSError: If file access fails.
        pyzstd.ZstdError: If decompression fails.
    """
    ...


def get_all_possible_moves() -> List[ChessMove]:
    """Generate all possible legal chess moves.

    Returns:
        List of all moves in UCI notation.
    """
    ...


T = TypeVar("T")


def chunks(lst: List[T], n: int) -> Generator[List[T], None, None]:
    """Split list into fixed-size chunks.

    Args:
        lst: List to divide.
        n: Chunk size.

    Yields:
        Sublists of size n (or smaller for last chunk).
    """
    ...
