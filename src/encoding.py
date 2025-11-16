# src/encoding.py

from __future__ import annotations

import numpy as np
import torch
import chess
from typing import List

# 0–5: White pieces, 6–11: Black pieces
PIECE_INDEX = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}


def parse_fen_to_bitboards(fen: str):
    """
    Parse a *full* FEN string into 12 bitboards + metadata.

    Returns:
      bitboards: list[12] of ints (one for each piece type)
      stm: "w" or "b"
      castling: castling rights string
      ep: en passant square string or "-"
      halfmove: int
      fullmove: int
    """
    parts = fen.split()
    if len(parts) < 4:
        raise ValueError(f"Invalid FEN: {fen}")

    board_part = parts[0]
    stm = parts[1]
    castling = parts[2]
    ep = parts[3]
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1

    bitboards = [0] * 12

    rows = board_part.split("/")
    assert len(rows) == 8, f"Unexpected FEN rows: {rows}"

    for row_idx, row in enumerate(rows):
        file_idx = 0  # a..h = 0..7
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                p_idx = PIECE_INDEX[ch]
                # rank: 0 = rank1, ..., 7 = rank8
                rank = 7 - row_idx  # FEN row 0 is rank8
                sq = rank * 8 + file_idx  # 0..63
                bitboards[p_idx] |= (1 << sq)
                file_idx += 1

    return bitboards, stm, castling, ep, halfmove, fullmove


def bitboards_to_piece_planes(bitboards: List[int]) -> np.ndarray:
    """
    Convert 12 bitboards into a [12, 8, 8] planes array.
    planes[p, row, col] = 1 if piece present, else 0.
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for p_idx, bb in enumerate(bitboards):
        b = bb
        while b:
            lsb = b & -b
            sq = (lsb.bit_length() - 1)
            b ^= lsb

            rank = sq // 8       # 0..7, 0 = rank1
            file = sq % 8        # 0..7, 0 = file a
            row = 7 - rank       # row 0 = rank8, row 7 = rank1
            col = file
            planes[p_idx, row, col] = 1.0

    return planes


def context_planes_from_meta(stm: str, castling: str, ep: str) -> np.ndarray:
    """
    Build 6 context planes:
      0: side-to-move (1 for white, 0 for black)
      1: white can castle kingside
      2: white can castle queenside
      3: black can castle kingside
      4: black can castle queenside
      5: en passant file (1s on that file if ep != "-")
    """
    planes = []

    # 1) side to move
    stm_plane = np.ones((8, 8), dtype=np.float32) if stm == "w" else np.zeros((8, 8), dtype=np.float32)
    planes.append(stm_plane)

    # 2) castling rights: K, Q, k, q
    for flag in ["K", "Q", "k", "q"]:
        has = flag in castling
        plane = np.ones((8, 8), dtype=np.float32) if has else np.zeros((8, 8), dtype=np.float32)
        planes.append(plane)

    # 3) en-passant file
    ep_plane = np.zeros((8, 8), dtype=np.float32)
    if ep != "-" and len(ep) >= 2:
        file_char = ep[0]
        file_idx = ord(file_char) - ord("a")
        if 0 <= file_idx < 8:
            ep_plane[:, file_idx] = 1.0
    planes.append(ep_plane)

    return np.stack(planes, axis=0)  # [6, 8, 8]


def fen_to_board_tensor(fen: str) -> torch.Tensor:
    """
    Convert a full FEN into a torch tensor [18, 8, 8].

    12 planes for pieces, 6 for context.
    """
    bitboards, stm, castling, ep, halfmove, fullmove = parse_fen_to_bitboards(fen)
    piece_planes = bitboards_to_piece_planes(bitboards)
    ctx_planes = context_planes_from_meta(stm, castling, ep)
    planes = np.concatenate([piece_planes, ctx_planes], axis=0)  # [18, 8, 8]
    return torch.from_numpy(planes)


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convenience helper: chess.Board → [18, 8, 8] tensor.
    """
    return fen_to_board_tensor(board.fen())
