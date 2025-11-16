from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import chess
import os

from src.utils import chess_manager

app = FastAPI()

# ---------- LAZY IMPORT FLAG ----------
_model_loaded = False

def ensure_model_loaded():
    """
    Import src.main once, lazily, so the heavy policy_net
    is loaded only when we actually need to move.
    """
    global _model_loaded
    if not _model_loaded:
        print("Importing src.main to register entrypoint and load policy net...")
        # this import triggers @chess_manager.entrypoint in src/main.py
        import src.main  # noqa: F401
        _model_loaded = True


# ---------- HEALTH CHECK ENDPOINT ----------
@app.post("/")
async def root():
    # devtools uses POST / as the readiness probe
    print("Health check: POST /")
    return JSONResponse({"running": True})


# (optional, nice for manual curl testing)
@app.get("/")
async def health_get():
    print("Health check: GET /")
    return JSONResponse({"running": True})


# ---------- MOVE ENDPOINT ----------
@app.post("/move")
async def get_move(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            content={"error": "Missing pgn or timeleft"},
            status_code=400,
        )

    if ("pgn" not in data or "timeleft" not in data):
        return JSONResponse(
            content={"error": "Missing pgn or timeleft"},
            status_code=400,
        )

    pgn = data["pgn"]
    timeleft = data["timeleft"]  # in ms

    # Make sure our NN entrypoint + model are loaded
    ensure_model_loaded()

    chess_manager.set_context(pgn, timeleft)
    print("pgn", pgn)

    try:
        start_time = time.perf_counter()
        move, move_probs, logs = chess_manager.get_model_move()
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
    except Exception as e:
        time_taken = (time.perf_counter() - start_time) * 1000
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": "Bot raised an exception",
                "logs": None,
                "exception": str(e),
            },
            status_code=500,
        )

    # move_probs must be dict[Move, float]
    if not isinstance(move_probs, dict):
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "error": "Failed to get move",
                "message": "Move probabilities is not a dictionary",
            },
            status_code=500,
        )

    for m, prob in move_probs.items():
        if not isinstance(m, chess.Move) or not isinstance(prob, float):
            return JSONResponse(
                content={
                    "move": None,
                    "move_probs": None,
                    "error": "Failed to get move",
                    "message": "Move probabilities has wrong types",
                },
                status_code=500,
            )

    move_probs_dict = {move.uci(): prob for move, prob in move_probs.items()}

    return JSONResponse(
        content={
            "move": move.uci(),
            "error": None,
            "time_taken": time_taken,
            "move_probs": move_probs_dict,
            "logs": logs,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("SERVE_PORT", "5058"))
    uvicorn.run(app, host="0.0.0.0", port=port)
