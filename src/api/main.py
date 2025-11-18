# src/api/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib
import os
import pandas as pd
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

from src.api.similarity_service import (
    load_players_df,
    search_players,
    get_player_by_name_or_id,
    get_similar_players,
    get_player_stats_for_radar,
    FEATURE_DESCRIPTIONS,
    clean
)

from src.api.gemini_client import generate_comparison_report

app = FastAPI(title="Player Recommendation API")

HERE = pathlib.Path(__file__).resolve().parent
FRONTEND_ROOT = (HERE / "../../frontend").resolve()

if FRONTEND_ROOT.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_ROOT)), name="static")

CSV_PATH = os.environ.get("PLAYER_CSV_PATH")
_meta_cache = {"leagues": [], "positions": []}

@app.on_event("startup")
def startup_event():
    load_players_df()
    build_meta()

def build_meta():
    global _meta_cache
    if not CSV_PATH or not os.path.exists(CSV_PATH):
        print("⚠ CSV not found:", CSV_PATH)
        _meta_cache = {"leagues": [], "positions": []}
        return
    try:
        df = pd.read_csv(CSV_PATH, low_memory=False)
        leagues = sorted(df["Comp"].dropna().astype(str).unique().tolist()) if "Comp" in df else []
        pos_set = set()
        if "Pos" in df.columns:
            for raw in df["Pos"].dropna().astype(str).unique():
                parts = [p.strip() for p in raw.split(',')]
                for p in parts:
                    pos_set.add(p)
        _meta_cache = {
            "leagues": leagues,
            "positions": sorted(list(pos_set))
        }
    except Exception as e:
        print("⚠ META BUILD FAILED:", e)
        _meta_cache = {"leagues": [], "positions": []}

@app.get("/")
def index():
    index_file = FRONTEND_ROOT / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"ok": True}

@app.get("/api/meta")
def api_meta():
    return {"ok": True, "leagues": _meta_cache["leagues"], "positions": _meta_cache["positions"]}

@app.get("/api/feature_desc")
def api_feature_desc():
    return {"ok": True, "descriptions": FEATURE_DESCRIPTIONS}

@app.get("/api/search")
def api_search(q: str, rows: int = 20):
    try:
        res = search_players(q, rows)
        return {"ok": True, "results": res}
    except Exception as e:
        return {"ok": False, "detail": str(e)}

@app.get("/api/player")
def api_player(player_id: str):
    p = get_player_by_name_or_id(player_id)
    if not p:
        raise HTTPException(404, "Player not found")
    return {"ok": True, "player": clean(p)}

@app.get("/api/similar")
def api_similar(
    player_id: str,
    k: int = 10,
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
    leagues: Optional[List[str]] = Query(None),
    positions: Optional[List[str]] = Query(None),
    nationalities: Optional[List[str]] = Query(None),
):
    filters = {
        "min_age": min_age,
        "max_age": max_age,
        "leagues": leagues,
        "positions": positions,
        "nationalities": nationalities
    }
    try:
        sim = get_similar_players(player_id, top_k=k, filters=filters)
        rad = get_player_stats_for_radar(player_id)
        return clean({"ok": True, "results": sim, "input_radar": rad})
    except Exception as e:
        return {"ok": False, "detail": str(e)}

@app.post("/api/compare")
def api_compare(player_ids: List[str]):
    if len(player_ids) < 2:
        raise HTTPException(400, "At least 2 players required.")
    try:
        players = []
        for pid in player_ids:
            p = get_player_by_name_or_id(pid)
            if not p:
                raise HTTPException(404, f"Player not found: {pid}")
            players.append(clean(p))
        radars = [get_player_stats_for_radar(p.get("Rk") or p.get("Player")) for p in players]
        keys = ["Player", "Pos", "Squad", "Age"] + radars[0]["labels"]
        rows = []
        for p in players:
            row = {k: p.get(k, "") for k in keys}
            rows.append({"Rk": p.get("Rk"), "stats": row})
        ai_payload = {"players": players, "radar": radars}
        ai_report = generate_comparison_report(ai_payload)
        return clean({
            "ok": True,
            "players": players,
            "radar": radars,
            "compare_stats": {"keys": keys, "rows": rows},
            "ai_report": ai_report
        })
    except Exception as e:
        return {"ok": False, "detail": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
