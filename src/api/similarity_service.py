# src/api/similarity_service.py
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import unidecode
import threading

CSV_PATH = os.environ.get("PLAYER_CSV_PATH", "data/football-player-stats-2023-COMPLETE.csv")

ATTACKER_FEATURES = [
    'Performance Gls', 'Performance Ast', 'Performance G+A',
    'Standard Sh', 'Standard SoT', 'Standard SoT%',
    'Standard Sh/90', 'Standard Dist',
    'Expected xG', 'Expected npxG', 'Expected xAG', 'Expected xA',
    'KP', 'PPA', 'CrsPA',
    'Progression PrgC', 'Progression PrgP', 'Progression PrgR',
    'Aerial Duels Won', 'Aerial Duels Won%',
    'Tackles Att 3rd', 'Performance Off'
]

MIDFIELDER_FEATURES = [
    'Total Cmp', 'Total Att', 'KP', 'PPA', 'CrsPA',
    'Expected xA', 'Expected xAG',
    'Progression PrgP', 'Progression PrgC', 'Progression PrgR',
    'Total TotDist', 'Total PrgDist',
    'Int', 'Tkl+Int', 'Tackles Tkl', 'Tackles Mid 3rd',
    'Challenges Tkl%', 'Challenges Lost',
    'Performance Gls', 'Performance Ast',
    'Aerial Duels Won%', 'Performance Recov'
]

DEFENDER_FEATURES = [
    'Tackles Tkl', 'Tackles TklW', 'Int', 'Tkl+Int',
    'Challenges Tkl%', 'Challenges Lost',
    'Blocks Blocks', 'Blocks Sh', 'Blocks Pass',
    'Clr', 'Aerial Duels Won', 'Aerial Duels Lost', 'Aerial Duels Won%',
    'Progression PrgP', 'Progression PrgC', 'Progression PrgR',
    'Total Cmp', 'Total Att', 'Total PrgDist',
    'Err'
]

GK_FEATURES = [
    'Performance GA', 'Performance SoTA', 'Performance Saves',
    'Performance Save%', 'Performance CS', 'Performance CS%',
    'Penalty Kicks PKsv', 'Penalty Kicks PKatt'
]

UNIVERSAL_FEATURES = [
    "Age", "Nation", "Squad", "Pos", "Playing Time MP",
    "Performance Gls", "Performance Ast", "Expected xG", "Expected xA", "KP"
]

RADAR_CATEGORIES_DEFAULT = [
    'Performance Gls','Performance Ast','KP','GCA GCA',
    'Aerial Duels Won','Int','Tackles TklW',
    'Performance Saves','Performance CS','Performance GA','Performance SoTA'
]

ALL_FEATURES_BY_POSITION = {
    "attacker": ATTACKER_FEATURES,
    "midfielder": MIDFIELDER_FEATURES,
    "defender": DEFENDER_FEATURES,
    "goalkeeper": GK_FEATURES
}

# Feature descriptions (from the user). Use keys exactly as the CSV column headers.
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "Rk": "Index or rank of the player in the list.",
    "Player": "Full name of the player.",
    "Nation": "Player's country of origin.",
    "Pos": "The position in which the player plays, e.g., 'FW' for forward.",
    "Squad": "The team the player belongs to.",
    "Comp": "The competition in which the player participated.",
    "Age": "The player's age.",
    "Born": "The player's date of birth.",
    "Playing Time MP": "The total number of matches the player participated in.",
    "Playing Time Starts": "The number of matches the player started.",
    "Playing Time Min": "The total minutes played by the player.",
    "Playing Time 90s": "The number of 90-minute periods played by the player.",
    "Performance Gls": "The number of goals scored by the player.",
    "Performance Ast": "The number of assists provided by the player.",
    "Performance G+A": "The sum of goals and assists by the player.",
    "Performance G-PK": "The number of goals scored excluding penalty kicks.",
    "Performance PK": "The number of penalty kicks scored by the player.",
    "Performance PKatt": "The number of penalty kicks attempted by the player.",
    "Performance CrdY": "The number of yellow cards received by the player.",
    "Performance CrdR": "The number of red cards received by the player.",
    "Performance 2CrdY": "The number of second yellow cards (subsequent yellow cards).",
    "Performance Fls": "The number of fouls committed by the player.",
    "Performance Fld": "The number of fouls suffered by the player.",
    "Performance Off": "The number of offside offenses committed by the player.",
    "Performance Crs": "The number of crosses executed by the player.",
    "Performance Int": "The number of interceptions made by the player.",
    "Performance OG": "The number of own goals scored by the player.",
    "Performance Recov": "The number of ball recoveries made by the player.",
    "Expected xG": "Expected Goals, i.e., the number of expected goals based on statistics.",
    "Expected npxG": "Non-Penalty Expected Goals, expected goals excluding penalties.",
    "Expected xAG": "Expected Assisted Goals, expected assists based on statistics.",
    "Expected npxG+xAG": "The sum of Non-Penalty Expected Goals and Expected Assisted Goals.",
    "Expected xA": "Expected Assists, expected assists based on statistics.",
    "Expected A-xAG": "The difference between Expected Assists and Expected Assisted Goals.",
    "Expected npxG/Sh": "Non-Penalty Expected Goals per shot attempted.",
    "Expected G-xG": "The difference between the actual number of goals and Expected Goals.",
    "Expected np:G-xG": "The difference between the actual number of non-penalty goals and Expected Goals.",
    "Progression PrgC": "The number of progressive ball carries.",
    "Progression PrgP": "The number of progressive passes made by the player.",
    "Progression PrgR": "The number of progressive passes received by the player.",
    "Per 90 Minutes Gls": "Number of goals per 90 minutes played.",
    "Per 90 Minutes Ast": "Number of assists per 90 minutes played.",
    "Per 90 Minutes G+A": "Sum of goals and assists per 90 minutes played.",
    "Per 90 Minutes G-PK": "Number of goals excluding penalty kicks per 90 minutes played.",
    "Per 90 Minutes G+A-PK": "Sum of goals and assists excluding penalty kicks per 90 minutes played.",
    "Per 90 Minutes xG": "Expected Goals per 90 minutes played.",
    "Per 90 Minutes xAG": "Expected Assisted Goals per 90 minutes played.",
    "Per 90 Minutes xG+xAG": "Sum of Expected Goals and Expected Assisted Goals per 90 minutes played.",
    "Per 90 Minutes npxG": "Non-Penalty Expected Goals per 90 minutes played.",
    "Per 90 Minutes npxG+xAG": "Sum of Non-Penalty Expected Goals and Expected Assisted Goals per 90 minutes played.",
    "Tackles Tkl": "The number of tackle attempts made by the player.",
    "Tackles TklW": "The number of tackles won by the player.",
    "Tackles Def 3rd": "The number of tackles made in the defensive third.",
    "Tackles Mid 3rd": "The number of tackles made in the central third of the field.",
    "Tackles Att 3rd": "The number of tackles made in the attacking third.",
    "Challenges Att": "The total number of challenges faced by the player.",
    "Challenges Tkl%": "Percentage of challenges won compared to total challenges.",
    "Challenges Lost": "The number of challenges lost by the player.",
    "Blocks Blocks": "The number of times the player has blocked the ball.",
    "Blocks Sh": "The number of opponent shots blocked by the player.",
    "Blocks Pass": "The number of opponent passes blocked by the player.",
    "Int": "The number of interceptions made by the player.",
    "Tkl+Int": "The sum of tackle attempts and interceptions made by the player.",
    "Clr": "The number of balls cleared by the player.",
    "Err": "The number of errors committed by the player that led to opponent shots.",
    "Standard Gls": "The number of standard goals scored by the player.",
    "Standard Sh": "The total number of standard shots attempted by the player.",
    "Standard SoT": "The number of standard shots on target made by the player.",
    "Standard SoT%": "Percentage of standard shots on target compared to total standard shots.",
    "Standard Sh/90": "Number of standard shots per 90 minutes played.",
    "Standard SoT/90": "Number of standard shots on target per 90 minutes played.",
    "Standard G/Sh": "Number of goals scored per standard shot attempted by the player.",
    "Standard G/SoT": "Number of goals scored per standard shot on target made by the player.",
    "Standard Dist": "The average distance of shots attempted by the player.",
    "Standard FK": "The number of free-kick shots attempted by the player.",
    "Standard PK": "The number of penalty kicks scored by the player.",
    "Standard PKatt": "The number of penalty kicks attempted by the player.",
    "Performance GA": "The number of goals conceded by the team when the player is on the field. (GK)",
    "Performance GA90": "The number of goals conceded per 90 minutes played by the player. (GK)",
    "Performance SoTA": "The number of opponent shots on target faced by the team when the player is on the field. (GK)",
    "Performance Saves": "The number of shots saved by the goalkeeper when the player is on the field. (GK)",
    "Performance Save%": "Percentage of shots saved compared to shots faced. (GK)",
    "Performance W": "The number of matches won by the team when the player is on the field. (GK)",
    "Performance D": "The number of matches drawn by the team when the player is on the field. (GK)",
    "Performance L": "The number of matches lost by the team when the player is on the field. (GK)",
    "Performance CS": "The number of clean sheets (matches without conceding goals) achieved by the team when the player is on the field. (GK)",
    "Performance CS%": "Percentage of matches in which the team does not concede goals when the player is on the field. (GK)",
    "Penalty Kicks PKatt": "The number of penalty kicks attempted by the team when the player is on the field.",
    "Penalty Kicks PKA": "The number of penalty kicks conceded by the team when the player is on the field.",
    "Penalty Kicks PKsv": "The number of penalty kicks saved by the team when the player is on the field.",
    "Penalty Kicks PKm": "The number of penalty kicks missed by the team when the player is on the field.",
    "SCA SCA": "The total number of 'Shot-Creating Actions' performed by the player.",
    "SCA SCA90": "The number of 'Shot-Creating Actions' per 90 minutes played by the player.",
    "SCA Types PassLive": "Number of 'Shot-Creating Actions' through live ball passes.",
    "SCA Types PassDead": "Number of 'Shot-Creating Actions' through dead ball passes.",
    "SCA Types TO": "Number of 'Shot-Creating Actions' through successful dribbles (take-ons).",
    "SCA Types Sh": "Number of 'Shot-Creating Actions' through shots.",
    "SCA Types Fld": "Number of 'Shot-Creating Actions' through fouls drawn.",
    "SCA Types Def": "Number of 'Shot-Creating Actions' through defensive actions.",
    "GCA GCA": "The total number of 'Goal-Creating Actions' performed by the player.",
    "GCA GCA90": "The number of 'Goal-Creating Actions' per 90 minutes played by the player.",
    "GCA Types PassLive": "Number of 'Goal-Creating Actions' through live ball passes.",
    "GCA Types PassDead": "Number of 'Goal-Creating Actions' through dead ball passes.",
    "GCA Types TO": "Number of 'Goal-Creating Actions' through successful dribbles (take-ons).",
    "GCA Types Sh": "Number of 'Goal-Creating Actions' through shots.",
    "GCA Types Fld": "Number of 'Goal-Creating Actions' through fouls drawn.",
    "GCA Types Def": "Number of 'Goal-Creating Actions' through defensive actions.",
    "Aerial Duels Won": "The number of aerial duels won by the player.",
    "Aerial Duels Lost": "The number of aerial duels lost by the player.",
    "Aerial Duels Won%": "Percentage of aerial duels won compared to total aerial duels.",
    "Total Cmp": "The total number of passes completed by the player.",
    "Total Att": "The total number of passes attempted by the player.",
    "Total Cmp%": "Percentage of passes completed compared to total passes attempted.",
    "Total TotDist": "The total distance covered by completed passes by the player.",
    "Total PrgDist": "The total distance covered by completed passes that advance towards the opponent's goal.",
    "KP": "The number of 'Key Passes' (key passes) made by the player.",
    "1/3": "The number of passes completed in the final third of the opponent's field.",
    "PPA": "The number of passes completed in the opponent's penalty area.",
    "CrsPA": "The number of crosses completed in the opponent's penalty area."
}

_df_players: Optional[pd.DataFrame] = None
_lock = threading.Lock()

_pos_feature_map = {
    'attacker': [c for c in ATTACKER_FEATURES],
    'midfielder': [c for c in MIDFIELDER_FEATURES],
    'defender': [c for c in DEFENDER_FEATURES],
    'goalkeeper': [c for c in GK_FEATURES]
}

_pos_scalers: Dict[str, MinMaxScaler] = {}
_pos_feature_cols: Dict[str, List[str]] = {}
_pos_index_to_group_index: Dict[str, Dict[int, int]] = {}
_pos_group_matrices: Dict[str, np.ndarray] = {}
_pos_similarity: Dict[str, np.ndarray] = {}

def map_position_by_first(pos_raw: Any) -> str:
    if not isinstance(pos_raw, str):
        return "midfielder"
    token = pos_raw.split(",")[0].strip().upper()
    if token == "GK": return "goalkeeper"
    if token in ("FW","ST","CF","FW "): return "attacker"
    if token in ("MF","CM","CAM","AM","DM","CDM","RM","LM"): return "midfielder"
    if token in ("DF","CB","LB","RB","LWB","RWB"): return "defender"
    if "FW" in pos_raw: return "attacker"
    if "MF" in pos_raw: return "midfielder"
    if "DF" in pos_raw: return "defender"
    return "midfielder"

def _ensure_loaded():
    global _df_players, _pos_scalers, _pos_feature_cols, _pos_group_matrices, _pos_similarity, _pos_index_to_group_index
    with _lock:
        if _df_players is not None: return
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"CSV not found at {CSV_PATH}.")
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip() for c in df.columns]
        df.fillna(0, inplace=True)
        df['PlayerNormalized'] = df['Player'].astype(str).apply(lambda s: unidecode.unidecode(s).lower())
        if 'Rk' not in df.columns:
            df.insert(0, 'Rk', list(range(len(df))))
        else:
            try:
                df['Rk'] = df['Rk'].astype(int)
            except Exception:
                df['Rk'] = pd.to_numeric(df['Rk'], errors='coerce').fillna(-1).astype(int)
        df['PositionGroup'] = df['Pos'].apply(map_position_by_first)
        _df_players = df
        for group, feature_list in _pos_feature_map.items():
            existing = [f for f in feature_list if f in df.columns]
            _pos_feature_cols[group] = existing
            if len(existing) == 0:
                _pos_scalers[group] = None
                _pos_group_matrices[group] = np.zeros((0, 0))
                _pos_similarity[group] = np.zeros((0, 0))
                _pos_index_to_group_index[group] = {}
                continue
            group_indices = df.index[df['PositionGroup'] == group].tolist()
            if len(group_indices) == 0:
                _pos_scalers[group] = None
                _pos_group_matrices[group] = np.zeros((0, len(existing)))
                _pos_similarity[group] = np.zeros((0, 0))
                _pos_index_to_group_index[group] = {}
                continue
            X = df.loc[group_indices, existing].copy()
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X.values)
            _pos_scalers[group] = scaler
            _pos_group_matrices[group] = X_scaled
            if X_scaled.shape[0] > 1:
                sim = cosine_similarity(X_scaled)
            else:
                sim = np.zeros((X_scaled.shape[0], X_scaled.shape[0]))
            _pos_similarity[group] = sim
            mapping = {int(idx): i for i, idx in enumerate(group_indices)}
            _pos_index_to_group_index[group] = mapping

def clean(obj):
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean(v) for v in obj]
    if isinstance(obj, tuple): return tuple(clean(v) for v in obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def load_players_df() -> pd.DataFrame:
    _ensure_loaded()
    return _df_players.copy()

def search_players(q: str, rows: int = 20) -> List[Dict[str, Any]]:
    _ensure_loaded()
    if not q: return []
    qnorm = unidecode.unidecode(q).lower()
    df = _df_players[_df_players['PlayerNormalized'].str.contains(qnorm, na=False)]
    df = df.head(rows)
    return [{"player_id": int(r['Rk']), "player_name": r['Player']} for _, r in df.iterrows()]

def get_player_by_name_or_id(player_identifier: str) -> Optional[Dict[str, Any]]:
    _ensure_loaded()
    df = _df_players
    row = pd.DataFrame()
    try:
        pid = int(player_identifier); row = df.loc[df['Rk'] == pid]
    except Exception:
        needle = unidecode.unidecode(str(player_identifier)).lower()
        row = df[df['PlayerNormalized'] == needle]
        if row.empty:
            row = df[df['PlayerNormalized'].str.contains(needle, na=False)]
    if row.empty: return None
    r = row.iloc[0].to_dict()
    out = {}
    for k, v in r.items():
        if isinstance(v, (np.integer,)): out[k] = int(v)
        elif isinstance(v, (np.floating,)): out[k] = float(v)
        else: out[k] = v
    return out

def _build_radar_for_player_row(row_index: int, category_labels: List[str]) -> Dict[str, Any]:
    _ensure_loaded()
    df = _df_players
    if row_index not in df.index: raise ValueError(f"Row index {row_index} not found")
    row = df.loc[row_index]
    player_pos_group = row['PositionGroup']
    labels = [c for c in category_labels if c in df.columns]
    values = []
    for label in labels:
        raw = 0.0
        try: raw = float(row.get(label, 0.0))
        except Exception: raw = 0.0
        pos_cols = _pos_feature_cols.get(player_pos_group, [])
        if label in pos_cols and _pos_scalers.get(player_pos_group) is not None:
            idx = pos_cols.index(label); scaler = _pos_scalers[player_pos_group]
            try:
                minv = float(scaler.data_min_[idx]); maxv = float(scaler.data_max_[idx])
                scaled = 0.0 if maxv == minv else (raw - minv) / (maxv - minv)
            except Exception:
                colmin = float(df[label].min()) if label in df.columns else 0.0
                colmax = float(df[label].max()) if label in df.columns else 1.0
                scaled = 0.0 if colmax == colmin else (raw - colmin) / (colmax - colmin)
        else:
            if label in df.columns:
                colmin = float(df[label].min()); colmax = float(df[label].max())
                scaled = 0.0 if colmax == colmin else (raw - colmin) / (colmax - colmin)
            else:
                scaled = 0.0
        scaled = max(0.0, min(1.0, float(scaled)))
        values.append(round(scaled, 4))
    return {"labels": labels, "values": values}

def get_player_stats_for_radar(player_identifier: str) -> Dict[str, Any]:
    _ensure_loaded()
    df = _df_players
    row_index = None
    try:
        pid = int(player_identifier)
        matches = df.loc[df['Rk'] == pid]
        if not matches.empty:
            row_index = matches.index[0]
    except Exception:
        needle = unidecode.unidecode(str(player_identifier)).lower()
        matches = df[df['PlayerNormalized'] == needle]
        if matches.empty:
            matches = df[df['PlayerNormalized'].str.contains(needle, na=False)]
        if matches.empty:
            raise ValueError(f"Player not found: {player_identifier}")
        row_index = matches.index[0]
    radar = _build_radar_for_player_row(row_index, RADAR_CATEGORIES_DEFAULT)
    return clean(radar)

def _attempt_group_for_player_index(global_index: int) -> str:
    _ensure_loaded()
    df = _df_players
    if global_index not in df.index: raise ValueError(f"Index {global_index} not found in df")
    primary = df.loc[global_index, 'PositionGroup']
    sim = _pos_similarity.get(primary, None)
    if isinstance(sim, np.ndarray) and sim.size > 1: return primary
    for alt in ['midfielder','attacker','defender','goalkeeper']:
        sim = _pos_similarity.get(alt, None)
        if isinstance(sim, np.ndarray) and sim.size > 1: return alt
    for alt in _pos_feature_cols.keys():
        if _pos_feature_cols.get(alt): return alt
    return primary

def _normalize_filter_param(param):
    if param is None: return None
    if isinstance(param, (list, tuple)):
        out = []
        for item in param:
            if not item: continue
            if isinstance(item, str) and ',' in item:
                out.extend([x.strip().lower() for x in item.split(',') if x.strip()])
            else:
                out.append(str(item).strip().lower())
        return out
    if isinstance(param, str):
        if ',' in param: return [x.strip().lower() for x in param.split(',') if x.strip()]
        return [param.strip().lower()]
    return None

def get_similar_players(player_id: str, top_k: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    _ensure_loaded()
    df = _df_players
    matches = None
    try:
        pid = int(player_id)
        matches = df.loc[df['Rk'] == pid]
    except Exception:
        needle = unidecode.unidecode(str(player_id)).lower()
        matches = df[df['PlayerNormalized'] == needle]
        if matches.empty:
            matches = df[df['PlayerNormalized'].str.contains(needle, na=False)]
    if matches is None or matches.empty:
        raise ValueError(f"Player not found: {player_id}")
    global_idx = int(matches.index[0])
    primary_group = df.loc[global_idx, 'PositionGroup']
    chosen_group = _attempt_group_for_player_index(global_idx)

    candidate_indices = list(df.index)
    if filters:
        cands = []
        min_age = filters.get("min_age")
        max_age = filters.get("max_age")
        leagues = _normalize_filter_param(filters.get("leagues"))
        positions = _normalize_filter_param(filters.get("positions"))
        nationalities = _normalize_filter_param(filters.get("nationalities"))
        for idx in df.index:
            row = df.loc[idx]
            ok = True
            if min_age is not None:
                try:
                    if float(row.get("Age", 0)) < float(min_age): ok = False
                except Exception: pass
            if max_age is not None:
                try:
                    if float(row.get("Age", 0)) > float(max_age): ok = False
                except Exception: pass
            if leagues:
                comp = str(row.get("Comp", "")).lower(); squad = str(row.get("Squad", "")).lower()
                if not any(l in comp or l in squad for l in leagues): ok = False
            if positions:
                posval = str(row.get("Pos", "")).lower()
                if not any(p == posval.split(",")[0].strip() or p in posval for p in positions): ok = False
            if nationalities:
                nation = str(row.get("Nation", "")).lower()
                if not any(n in nation for n in nationalities): ok = False
            if ok: cands.append(idx)
        candidate_indices = cands

    mapping = _pos_index_to_group_index.get(chosen_group, {})
    group_candidate_pairs = []
    for idx in candidate_indices:
        if idx == global_idx: continue
        if idx in mapping:
            group_candidate_pairs.append((idx, mapping[idx]))
    if len(group_candidate_pairs) == 0:
        for grp in ['midfielder','attacker','defender','goalkeeper']:
            mp = _pos_index_to_group_index.get(grp, {})
            if not mp: continue
            for idx in candidate_indices:
                if idx == global_idx: continue
                if idx in mp:
                    group_candidate_pairs.append((idx, mp[idx]))
            if len(group_candidate_pairs) > 0: chosen_group = grp; break
    if len(group_candidate_pairs) == 0: return []

    query_group_index = _pos_index_to_group_index.get(chosen_group, {}).get(global_idx, None)
    results = []
    sim_matrix = _pos_similarity.get(chosen_group, None)
    pos_cols = _pos_feature_cols.get(chosen_group, [])

    if query_group_index is not None and isinstance(sim_matrix, np.ndarray) and sim_matrix.size > 0:
        sims = sim_matrix[query_group_index]; pairs = []
        for (gidx, grp_idx) in group_candidate_pairs:
            score = float(sims[grp_idx]); pairs.append((gidx, score))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    else:
        pairs = []
        if chosen_group in _pos_scalers and _pos_scalers[chosen_group] is not None and len(_pos_feature_cols[chosen_group])>0:
            scaler = _pos_scalers[chosen_group]; cols = _pos_feature_cols[chosen_group]
            qrow = _df_players.loc[global_idx, cols].copy()
            qarr = []
            for col in cols:
                try: qarr.append(float(qrow.get(col, 0.0)))
                except Exception: qarr.append(0.0)
            qarr = np.array(qarr).reshape(1, -1)
            try: qscaled = scaler.transform(qarr)
            except Exception: qscaled = np.zeros((1, len(cols)))
            group_matrix = _pos_group_matrices.get(chosen_group, None)
            for gidx, grp_idx in group_candidate_pairs:
                if grp_idx is not None and group_matrix is not None and grp_idx < group_matrix.shape[0]:
                    cand_vec = group_matrix[grp_idx].reshape(1,-1)
                else:
                    crow = _df_players.loc[gidx, cols]
                    carr = []
                    for col in cols:
                        try: carr.append(float(crow.get(col, 0.0)))
                        except Exception: carr.append(0.0)
                    carr = np.array(carr).reshape(1,-1)
                    try: cand_vec = scaler.transform(carr)
                    except Exception: cand_vec = np.zeros_like(qscaled)
                denom = np.linalg.norm(qscaled) * np.linalg.norm(cand_vec)
                if denom == 0: score = 0.0
                else: score = float(np.dot(qscaled, cand_vec.T)[0][0] / denom)
                pairs.append((gidx, score))
        else:
            fallback_cols = ['Per 90 Minutes Gls', 'Per 90 Minutes Ast', 'Per 90 Minutes xG', 'Per 90 Minutes xAG']
            fallback_cols = [c for c in fallback_cols if c in _df_players.columns]
            if not fallback_cols: return []
            fallback_df = _df_players.loc[[global_idx] + [g for g,_ in group_candidate_pairs], fallback_cols].copy()
            for col in fallback_df.columns:
                fallback_df[col] = pd.to_numeric(fallback_df[col], errors='coerce').fillna(0.0)
            arr = fallback_df.values.astype(float)
            minv = np.min(arr, axis=0); maxv = np.max(arr, axis=0)
            denom = (maxv - minv); denom[denom == 0] = 1.0
            arr_norm = (arr - minv) / denom
            qvec = arr_norm[0].reshape(1,-1)
            pairs = []
            for i, (gidx,_) in enumerate(group_candidate_pairs):
                cand_vec = arr_norm[1+i].reshape(1,-1)
                denom2 = np.linalg.norm(qvec) * np.linalg.norm(cand_vec)
                if denom2 == 0: score = 0.0
                else: score = float(np.dot(qvec, cand_vec.T)[0][0] / denom2)
                pairs.append((gidx, score))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

    top_pairs = pairs_sorted[:top_k]
    for gidx, score in top_pairs:
        row = df.loc[gidx]
        top_stats = {}
        for c in RADAR_CATEGORIES_DEFAULT:
            if c in df.columns:
                try: top_stats[c] = float(row.get(c, 0.0))
                except Exception: top_stats[c] = 0.0
        radar = _build_radar_for_player_row(gidx, RADAR_CATEGORIES_DEFAULT)
        results.append({
            "Rk": int(row.get("Rk", int(gidx))),
            "Player": row.get("Player", ""),
            "Pos": row.get("Pos", ""),
            "Squad": row.get("Squad", ""),
            "Age": row.get("Age", ""),
            "Nation": row.get("Nation", ""),
            "similarity_score": float(score),
            "top_stats": top_stats,
            "radar": radar
        })

    return clean(results)
