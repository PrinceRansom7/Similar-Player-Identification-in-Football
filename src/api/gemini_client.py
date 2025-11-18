# src/api/gemini_client.py
import os
import requests
import json
from typing import Dict, Any

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-pro")

GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

AI_PROMPT_TEMPLATE = """
You are an elite European football scouting analyst.
Compare the players strictly based on the provided data (stats + radar values).

Your analysis should:
- Identify technical strengths
- Identify weaknesses
- Compare each player directly to the others
- Highlight tactical suitability
- Mention role tendencies and expected usage
- Keep the summary concise (~300 words max)
"""

def _ensure_key():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing in environment variables.")

def _build_prompt(players):
    lines = [AI_PROMPT_TEMPLATE.strip(), "\nPLAYER DATA:\n"]
    brief_keys = [
        "Performance Gls", "Performance Ast", "KP",
        "Expected xG", "Expected xA",
        "Int", "Tackles Tkl", "Aerial Duels Won"
    ]
    for p in players:
        name = p.get("Player", "Unknown")
        pos  = p.get("Pos", "")
        club = p.get("Squad", "")
        snippet = []
        for k in brief_keys:
            if k in p:
                snippet.append(f"{k}: {p[k]}")
        lines.append(f"- {name} | {pos} | {club} | " + "; ".join(snippet))
    lines.append("\nProvide a direct comparison and a summary.\n")
    return "\n".join(lines)

def generate_comparison_report(payload: Dict[str, Any]) -> str:
    try:
        _ensure_key()
    except Exception as e:
        return str(e)

    players = payload.get("players", [])
    prompt_text = _build_prompt(players)

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.25,
            "maxOutputTokens": 600
        }
    }

    url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"

    try:
        resp = requests.post(url, json=body, timeout=30)
    except Exception as e:
        return f"Gemini request failed: {e}"

    if resp.status_code != 200:
        return f"Gemini error {resp.status_code}: {resp.text}"

    try:
        data = resp.json()
    except:
        return resp.text[:1500]

    def extract_text(obj):
        if isinstance(obj, dict):
            if "candidates" in obj:
                return extract_text(obj["candidates"][0])
            if "content" in obj:
                parts = obj["content"][0].get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
            for v in obj.values():
                out = extract_text(v)
                if out:
                    return out
        elif isinstance(obj, list):
            for item in obj:
                out = extract_text(item)
                if out:
                    return out
        return None

    out = extract_text(data)
    return out if out else json.dumps(data)[:1500]
