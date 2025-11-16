from __future__ import annotations

from chess import Move
import chess
import torch
import math
import os

from .utils import chess_manager, GameContext
from .encoding import fen_to_board_tensor  # we’ll wrap this for Boards
from .model import PolicyNet, ValueNet


# =========================
# Device & model loading
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(THIS_DIR, "policy_net.pth")
VALUE_PATH  = os.path.join(THIS_DIR, "value_net.pth")


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a python-chess Board to the 18x8x8 tensor
    used in training (via fen_to_board_tensor).
    """
    fen = board.fen()
    planes = fen_to_board_tensor(fen)          # [18,8,8]
    return planes                              # torch.Tensor float32


# ---- Policy net ----

print("Loading policy net from:", POLICY_PATH)
policy_ckpt = torch.load(POLICY_PATH, map_location=device)

move2idx: dict[str, int] = policy_ckpt["move2idx"]
idx2move: dict[int, str] = {v: k for k, v in move2idx.items()}

policy_model = PolicyNet(
    action_size=len(move2idx),
    channels=128,
    num_blocks=5,
).to(device)
policy_model.load_state_dict(policy_ckpt["model_state_dict"])
policy_model.eval()
print(f"Policy net loaded with {len(move2idx)} actions.")


# ---- Value net ----

print("Loading value net from:", VALUE_PATH)
value_ckpt = torch.load(VALUE_PATH, map_location=device)

value_model = ValueNet(
    channels=128,
    num_blocks=5,
).to(device)
value_model.load_state_dict(value_ckpt["model_state_dict"])
value_model.eval()
print("Value net loaded.")


# =========================
# Evaluation & search
# =========================

def eval_board(board: chess.Board) -> float:
    """
    Evaluate a position from the side-to-move perspective using ValueNet.
    Output is roughly in [-1, 1].
    """
    with torch.no_grad():
        x = board_to_tensor(board).unsqueeze(0).to(device)  # [1,18,8,8]
        v = value_model(x)                                  # [1]
    return float(v.item())


def negamax(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    """
    Simple Negamax search with alpha-beta pruning
    using ValueNet at the leaves.
    """
    if depth <= 0 or board.is_game_over():
        return eval_board(board)

    best = -1e9

    # Plain move loop for now; root ordering will be done separately.
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    return best


SEARCH_DEPTH = 2


def get_policy_probs_for_legal_moves(board: chess.Board) -> dict[Move, float]:
    """
    Run PolicyNet on the current board and turn logits into a probability
    distribution over *legal* moves.

    Returns: {chess.Move: prob}
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return {}

    with torch.no_grad():
        x = board_to_tensor(board).unsqueeze(0).to(device)  # [1,18,8,8]
        logits = policy_model(x)[0]                         # [action_size]

    scores: list[float] = []
    moves: list[Move] = []

    for mv in legal_moves:
        uci = mv.uci()
        idx = move2idx.get(uci)
        if idx is not None:
            scores.append(float(logits[idx].item()))
            moves.append(mv)

    if not scores:
        # None of the legal moves are in vocab (very rare) → uniform.
        p = 1.0 / len(legal_moves)
        return {m: p for m in legal_moves}

    # Softmax over the legal moves
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    Z = sum(exps)
    probs = [e / Z for e in exps]

    return {m: p for m, p in zip(moves, probs)}


def select_move_with_search(board: chess.Board) -> tuple[Move | None, dict[Move, float]]:
    """
    Use PolicyNet to prioritize moves, ValueNet+Negamax to score them.

    Returns:
      best_move: chosen chess.Move or None
      move_probs: dict[Move, float] – probabilities to log to UI
                  (policy-based distribution over legal moves)
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, {}

    # 1) Get policy probabilities over legal moves
    policy_probs = get_policy_probs_for_legal_moves(board)

    # 2) Order moves by policy prior (high → low)
    ordered_moves = sorted(
        legal_moves,
        key=lambda m: policy_probs.get(m, 0.0),
        reverse=True,
    )

    # 3) Run negamax search to get a value-based score for each move
    best_move: Move | None = None
    best_score = -1e9
    move_scores: dict[Move, float] = {}

    for move in ordered_moves:
        board.push(move)
        score = -negamax(board, SEARCH_DEPTH - 1, -1e9, 1e9)
        board.pop()

        move_scores[move] = score

        if best_move is None or score > best_score:
            best_move = move
            best_score = score

    # 4) Decide what to log as move probabilities.
    # Here we log *policy* probabilities restricted to legal moves.
    total_p = sum(policy_probs.get(m, 0.0) for m in ordered_moves)
    if total_p > 0:
        move_probs = {m: policy_probs.get(m, 0.0) / total_p for m in ordered_moves}
    else:
        # fallback uniform if policy had no mass
        u = 1.0 / len(ordered_moves)
        move_probs = {m: u for m in ordered_moves}

    return best_move, move_probs


# =========================
# Entrypoint for chess_manager
# =========================

@chess_manager.entrypoint
def engine_entrypoint(ctx: GameContext) -> Move:
    """
    Called every time the engine needs to make a move.

    - Uses ValueNet + negamax for the decision.
    - Uses PolicyNet for the move probability distribution.
    - Logs probabilities via ctx.logProbabilities so devtools can show them.
    """
    board = ctx.board
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        # No legal moves → game over.
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (game over)")

    best_move, move_probs = select_move_with_search(board)

    if best_move is None:
        # Safety fallback (shouldn't normally happen)
        best_move = legal_moves[0]

    # Log probabilities (dict[Move, float]) for the devtools sidebar
    ctx.logProbabilities(move_probs)

    return best_move
