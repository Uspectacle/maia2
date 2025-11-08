"""Utility functions for MAIA2.

Provides functions for chess position handling, Elo rating mapping,
model configuration, and data processing utilities.
"""

import os
import pickle
import random
import re
import time
from typing import Dict, Final, Generator, List, Optional, Tuple, TypeVar, Union

import chess
import numpy as np
import pyzstd
import torch
import yaml

# Type aliases
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

# Constants
ELO_INTERVAL: Final[EloRating] = 100
ELO_START: Final[EloRating] = 1100
ELO_END: Final[EloRating] = 2000
PIECE_TYPES: Final[List[chess.PieceType]] = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


class Config:
    """Dynamic configuration container for MAIA2."""

    input_channels: int = 1
    elo_dim: int = 1
    dim_cnn: int = 1
    dim_vit: int = 1
    num_blocks_cnn: int = 1
    num_blocks_vit: int = 1
    vit_length: int = 1
    batch_size: Optional[int]
    chunk_size: int = 1
    verbose: Optional[bool]
    start_year: int = 1900
    end_year: int = 2027
    start_month: int = 1
    end_month: int = 12
    first_n_moves: int = 0
    clock_threshold: float = 0
    max_ply: Optional[int]
    data_root: str = "~"
    num_workers: int = 1
    side_info: Optional[int]
    side_info_coefficient: Optional[float]
    value: Optional[int]
    value_coefficient: Optional[float]
    seed: int = 123456789
    num_cpu_left: int = 1
    lr: float = 1e-3
    wd: float = 1e-2
    from_checkpoint: Optional[bool]
    checkpoint_epoch: Optional[int]
    checkpoint_year: Optional[str]
    checkpoint_month: Optional[str]
    max_epochs: int = 1
    queue_length: int = 1
    max_games_per_elo_range: int = 1

    def __init__(self, config_dict: ConfigDict) -> None:
        """Initialize from dictionary.

        Args:
            config_dict: Configuration key-value pairs.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)


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
    with open(cfg_file_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = Config(cfg_dict)

    return cfg


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Seed value for all RNGs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def delete_file(filename: str) -> None:
    """Delete file if it exists.

    Args:
        filename: Path to file.
    """
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Data {filename} has been deleted.")
    else:
        print(f"The file '{filename}' does not exist.")


def readable_num(num: int) -> str:
    """Convert large number to readable format.

    Args:
        num: Number to format.

    Returns:
        Formatted string with suffix (K/M/B).
    """
    if num >= 1e9:  # if parameters are in the billions
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:  # if parameters are in the millions
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:  # if parameters are in the thousands
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def readable_time(elapsed_time: TimeSeconds) -> str:
    """Format elapsed time in readable format.

    Args:
        elapsed_time: Duration in seconds.

    Returns:
        Formatted time string (e.g., "1h 30m 45.50s").
    """
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"


def count_parameters(model: torch.nn.Module) -> str:
    """Count trainable parameters in model.

    Args:
        model: PyTorch model.

    Returns:
        Formatted parameter count string.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected PyTorch Module, got {type(model)}")

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    return readable_num(total_params)


def create_elo_dict() -> EloRangeDict:
    """Create Elo rating ranges to category indices mapping.

    Returns:
        Dict mapping Elo range strings to indices.
    """
    range_dict: EloRangeDict = {f"<{ELO_START}": 0}
    range_index = 1

    for lower_bound in range(ELO_START, ELO_END - 1, ELO_INTERVAL):
        upper_bound = lower_bound + ELO_INTERVAL
        range_dict[f"{lower_bound}-{upper_bound - 1}"] = range_index
        range_index += 1

    range_dict[f">={ELO_END}"] = range_index

    # print(range_dict, flush=True)

    return range_dict


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
    if not isinstance(elo, int):
        raise TypeError(f"Elo rating must be an integer, got {type(elo)}")

    if elo < ELO_START:
        return elo_dict[f"<{ELO_START}"]
    elif elo >= ELO_END:
        return elo_dict[f">={ELO_END}"]
    else:
        for lower_bound in range(ELO_START, ELO_END - 1, ELO_INTERVAL):
            upper_bound = lower_bound + ELO_INTERVAL
            if lower_bound <= elo < upper_bound:
                return elo_dict[f"{lower_bound}-{upper_bound - 1}"]

    raise ValueError(f"Elo {elo} could not be categorized.")


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
    move = chess.Move.from_uci(move_uci)

    moving_piece = board.piece_at(move.from_square)
    captured_piece = board.piece_at(move.to_square)

    from_square_encoded = torch.zeros(64)
    from_square_encoded[move.from_square] = 1

    to_square_encoded = torch.zeros(64)
    to_square_encoded[move.to_square] = 1

    if move_uci == "e1g1":
        rook_move = chess.Move.from_uci("h1f1")
        from_square_encoded[rook_move.from_square] = 1
        to_square_encoded[rook_move.to_square] = 1

    if move_uci == "e1c1":
        rook_move = chess.Move.from_uci("a1d1")
        from_square_encoded[rook_move.from_square] = 1
        to_square_encoded[rook_move.to_square] = 1

    board.push(move)
    is_check = board.is_check()
    board.pop()

    # Order: Pawn, Knight, Bishop, Rook, Queen, King
    side_info = torch.zeros(6 + 6 + 1)
    assert moving_piece is not None
    side_info[moving_piece.piece_type - 1] = 1
    if move_uci in ["e1g1", "e1c1"]:
        side_info[3] = 1
    if captured_piece:
        side_info[6 + captured_piece.piece_type - 1] = 1
    if is_check:
        side_info[-1] = 1

    legal_moves = torch.zeros(len(all_moves_dict))
    legal_moves_idx = torch.tensor(
        [all_moves_dict[move.uci()] for move in board.legal_moves]
    )
    legal_moves[legal_moves_idx] = 1

    side_info = torch.cat(
        [side_info, from_square_encoded, to_square_encoded, legal_moves], dim=0
    )

    return legal_moves, side_info


def extract_clock_time(comment: str) -> Optional[int]:
    """Extract remaining clock time from PGN comment.

    Args:
        comment: PGN comment string.

    Returns:
        Remaining time in seconds, or None if not found.
    """
    match = re.search(r"\[%clk (\d+):(\d+):(\d+)\]", comment)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        return hours * 3600 + minutes * 60 + seconds
    return None


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
    cache_file = pgn_path.replace(".pgn", "_chunks.pkl")

    if os.path.exists(cache_file):
        print(f"Loading cached chunks from {cache_file}")
        with open(cache_file, "rb") as f:
            pgn_chunks = pickle.load(f)
    else:
        print(f"Cache not found. Creating chunks for {pgn_path}")
        start_time = time.time()
        pgn_chunks = get_chunks(pgn_path, cfg.chunk_size)
        print(
            f"Chunking took {readable_time(time.time() - start_time)}", flush=True)

        with open(cache_file, "wb") as f:
            pickle.dump(pgn_chunks, f)

    return pgn_chunks


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Convert chess position to feature tensor.

    Args:
        board: Chess position.

    Returns:
        Tensor [18, 8, 8] with board representation.
    """
    num_piece_channels: Final[int] = 12  # 6 piece types * 2 colors
    # 1 for player's turn, 4 for castling rights, 1 for en passant
    additional_channels: Final[int] = 6
    tensor = torch.zeros(
        (num_piece_channels + additional_channels, 8, 8), dtype=torch.float32
    )

    # Precompute indices for each piece type
    piece_indices = {piece: i for i, piece in enumerate(PIECE_TYPES)}

    # Fill tensor for each piece type
    for piece_type in PIECE_TYPES:
        for color in [True, False]:  # True=White, False=Black
            piece_map = board.pieces(piece_type, color)
            index = piece_indices[piece_type] + (0 if color else 6)
            for square in piece_map:
                row, col = divmod(square, 8)
                tensor[index, row, col] = 1.0

    # Player's turn channel (White = 1, Black = 0)
    turn_channel = num_piece_channels
    if board.turn == chess.WHITE:
        tensor[turn_channel, :, :] = 1.0

    # Castling rights channels
    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]
    for i, has_right in enumerate(castling_rights):
        if has_right:
            tensor[num_piece_channels + 1 + i, :, :] = 1.0

    # En passant target channel
    ep_channel = num_piece_channels + 5
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        tensor[ep_channel, row, col] = 1.0

    return tensor


def generate_pawn_promotions() -> List[ChessMove]:
    """Generate all possible pawn promotion moves.

    Returns:
        List of UCI promotion moves.
    """
    # Define the promotion rows for both colors and the promotion pieces
    # promotion_rows = {'white': '7', 'black': '2'}
    promotion_rows = {"white": "7"}
    promotion_pieces = ["q", "r", "b", "n"]
    promotions: List[ChessMove] = []

    # Iterate over each color
    for color, row in promotion_rows.items():
        # Target rows for promotion (8 for white, 1 for black)
        target_row = "8" if color == "white" else "1"

        # Each file from 'a' to 'h'
        for file in "abcdefgh":
            # Direct move to promotion
            for piece in promotion_pieces:
                promotions.append(f"{file}{row}{file}{target_row}{piece}")

            # Capturing moves to the left and right (if not on the edges of the board)
            if file != "a":
                left_file = chr(ord(file) - 1)
                for piece in promotion_pieces:
                    promotions.append(
                        f"{file}{row}{left_file}{target_row}{piece}")

            # Capture right
            if file != "h":
                right_file = chr(ord(file) + 1)
                for piece in promotion_pieces:
                    promotions.append(
                        f"{file}{row}{right_file}{target_row}{piece}")

    return promotions


def mirror_square(square: str) -> str:
    """Mirror chess square vertically.

    Args:
        square: Square in algebraic notation.

    Returns:
        Mirrored square.
    """
    file = square[0]
    rank = str(9 - int(square[1]))

    return file + rank


def mirror_move(move_uci: ChessMove) -> ChessMove:
    """Mirror chess move vertically.

    Args:
        move_uci: Move in UCI notation.

    Returns:
        Mirrored move in UCI notation.
    """
    # Check if the move is a promotion (length of UCI string will be more than 4)
    is_promotion = len(move_uci) > 4

    # Extract the start and end squares, and the promotion piece if applicable
    start_square = move_uci[:2]
    end_square = move_uci[2:4]
    promotion_piece = move_uci[4:] if is_promotion else ""

    # Mirror the start and end squares
    mirrored_start = mirror_square(start_square)
    mirrored_end = mirror_square(end_square)

    # Return the mirrored move, including the promotion piece if applicable
    return mirrored_start + mirrored_end + promotion_piece


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
    chunk_list: List[Chunk] = []
    with open(pgn_path, "r", encoding="utf-8") as pgn_file:
        while True:
            start_pos = pgn_file.tell()
            game_count = 0
            while game_count < chunk_size:
                line = pgn_file.readline()
                if not line:
                    break
                if line[-4:] == "1-0\n" or line[-4:] == "0-1\n":
                    game_count += 1
                if line[-8:] == "1/2-1/2\n":
                    game_count += 1
                if line[-2:] == "*\n":
                    game_count += 1
            line = pgn_file.readline()
            if line not in ["\n", ""]:
                raise ValueError
            end_pos = pgn_file.tell()
            chunk_list.append((start_pos, end_pos))
            if not line:
                break

    return chunk_list


def decompress_zst(file_path: str, decompressed_path: str) -> None:
    """Decompress Zstandard (.zst) file.

    Args:
        file_path: Path to .zst file.
        decompressed_path: Output path for decompressed file.

    Raises:
        OSError: If file access fails.
        pyzstd.ZstdError: If decompression fails.
    """
    with (
        open(file_path, "rb") as compressed_file,
        open(decompressed_path, "wb") as decompressed_file,
    ):
        pyzstd.decompress_stream(compressed_file, decompressed_file)


def get_all_possible_moves() -> List[ChessMove]:
    """Generate all possible legal chess moves.

    Returns:
        List of all moves in UCI notation.
    """
    all_moves: List[chess.Move] = []

    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)

            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.QUEEN, chess.WHITE))
            legal_moves = list(board.legal_moves)
            all_moves.extend(legal_moves)

            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.KNIGHT, chess.WHITE))
            legal_moves = list(board.legal_moves)
            all_moves.extend(legal_moves)

    all_moves_uci = [all_moves[i].uci() for i in range(len(all_moves))]

    pawn_promotions = generate_pawn_promotions()

    return all_moves_uci + pawn_promotions


T = TypeVar("T")


def chunks(lst: List[T], n: int) -> Generator[List[T], None, None]:
    """Split list into fixed-size chunks.

    Args:
        lst: List to divide.
        n: Chunk size.

    Yields:
        Sublists of size n (or smaller for last chunk).
    """
    for i in range(0, len(lst), n):
        yield lst[i: i + n]
