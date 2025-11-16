# src/main.py

from __future__ import annotations

from chess import Move
import chess
import torch
import math
import os

from .utils import chess_manager, GameContext
from .encoding import fen_to_board_tensor
from .policy_model import PolicyNet

# =========================
#  Global config
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Depth in plies (half-moves): 2 = our move + their reply
SEARCH_DEPTH = 3

# =========================
#  Load policy net + move2idx
# =========================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(THIS_DIR, "policy_net.pth")

print("Loading policy net from:", POLICY_PATH)
ckpt = torch.load(POLICY_PATH, map_location=device)

move2idx: dict[str, int] = ckpt["move2idx"]
idx2move: dict[int, str] = {v: k for k, v in move2idx.items()}

policy_model = PolicyNet(action_size=len(move2idx))
policy_model.load_state_dict(ckpt["model_state_dict"])
policy_model.to(device)
policy_model.eval()

print(f"Policy net loaded with {len(move2idx)} actions.")


# =========================
#  Evaluation using policy net ONLY
# =========================

def eval_board(board: chess.Board) -> float:
    """
    Evaluate the position from the side-to-move's perspective
    using ONLY the policy net.

    Idea:
      - Run PolicyNet to get logits over all actions.
      - Restrict to *legal* moves and take something like
        "how good is the best legal move?"
      - Squash via tanh to keep values in a reasonable range.

    This is a hacky value function, but it's 100% NN-based.
    """
    # Handle terminal positions by rules (no heuristics like material)
    if board.is_game_over():
        if board.is_checkmate():
            # side to move is checkmated -> awful
            return -1.0
        # stalemate / draw
        return 0.0

    fen = board.fen()
    planes = fen_to_board_tensor(fen).unsqueeze(0).to(device)  # [1, 18, 8, 8]

    with torch.no_grad():
        logits = policy_model(planes)[0]  # [num_actions]

    legal_moves = list(board.legal_moves)
    legal_logits = []

    for m in legal_moves:
        uci = m.uci()
        idx = move2idx.get(uci)
        if idx is not None:
            legal_logits.append(logits[idx])

    if not legal_logits:
        # No mapped moves (should be rare) -> fall back to overall max logit
        best_logit = float(logits.max().item())
    else:
        best_logit = float(torch.stack(legal_logits).max().item())

    # Compress to something ~[-1,1]
    value = math.tanh(best_logit / 3.0)
    return float(value)


# =========================
#  Move ordering using policy net
# =========================

def order_moves_with_policy(board: chess.Board, moves: list[Move]) -> list[Move]:
    """
    Use policy logits to order legal moves for better pruning.
    """
    if not moves:
        return moves

    fen = board.fen()
    planes = fen_to_board_tensor(fen).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = policy_model(planes)[0]  # [num_actions]

    scores: dict[Move, float] = {}
    for m in moves:
        uci = m.uci()
        idx = move2idx.get(uci)
        if idx is None:
            # unseen move -> low score but still considered
            scores[m] = -1e9
        else:
            scores[m] = float(logits[idx].item())

    return sorted(moves, key=lambda mv: scores[mv], reverse=True)


# =========================
#  Negamax + alpha-beta using policy eval
# =========================

def negamax(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    """
    Negamax search with alpha-beta pruning.

    Evaluation uses ONLY the policy net (eval_board).
    """
    if depth == 0 or board.is_game_over():
        return eval_board(board)

    best = -1e9

    legal_moves = list(board.legal_moves)
    ordered_moves = order_moves_with_policy(board, legal_moves)

    for move in ordered_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break  # alpha-beta cutoff

    return best


def select_move_with_search(board: chess.Board) -> tuple[Move | None, dict[Move, float]]:
    """
    Run negamax from the current position and return:
      - the best move according to policy-based search
      - a dict {move: score} for all legal moves
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, {}

    move_scores: dict[Move, float] = {}
    best_move: Move | None = None
    best_score = -1e9

    root_moves = order_moves_with_policy(board, legal_moves)

    for move in root_moves:
        board.push(move)
        score = -negamax(board, SEARCH_DEPTH - 1, -1e9, 1e9)
        board.pop()

        move_scores[move] = score

        if best_move is None or score > best_score:
            best_move = move
            best_score = score

    return best_move, move_scores


# =========================
#  Scores → probabilities for UI
# =========================

def scores_to_probabilities(
    move_scores: dict[Move, float],
    temperature: float = 1.0,
) -> dict[Move, float]:
    """
    Convert {move: score} to {move: probability} via softmax.
    """
    if not move_scores:
        return {}

    max_score = max(move_scores.values())
    exps: dict[Move, float] = {}

    for m, s in move_scores.items():
        exps[m] = math.exp((s - max_score) / temperature)

    Z = sum(exps.values())
    if Z <= 0.0:
        # uniform fallback
        uniform = 1.0 / len(move_scores)
        return {m: uniform for m in move_scores}

    return {m: exps[m] / Z for m in move_scores}


# =========================
#  Entrypoint & reset
# =========================

@chess_manager.entrypoint
def test_func(ctx: GameContext) -> Move:
    """
    Called every time the engine needs to make a move.

    - Search: negamax + alpha-beta
    - Eval: policy net ONLY (via eval_board)
    - Move ordering: policy net
    - Probabilities: softmax over search scores
    """
    board = ctx.board
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (game over)")

    best_move, move_scores = select_move_with_search(board)

    if best_move is None:
        best_move = legal_moves[0]
        move_scores = {m: 0.0 for m in legal_moves}

    move_probs = scores_to_probabilities(move_scores, temperature=1.0)
    ctx.logProbabilities(move_probs)

    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game begins.
    No persistent state currently.
    """
    pass
