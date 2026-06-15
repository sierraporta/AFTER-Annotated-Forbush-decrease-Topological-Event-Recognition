from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Constantes ────────────────────────────────────────────────────────────────
SAME_STATION_EVENT_MERGE_GAP_HOURS  = 12.0
TRIGGER_GROUP_ONSET_WINDOW_HOURS    = 18.0
TRIGGER_GROUP_OVERLAP_TOL_HOURS     = 12.0
SUPPORT_PRE_HOURS                   = 24.0
SUPPORT_POST_HOURS                  = 36.0
SUPPORT_MIN_DROP_PERCENT            = 0.75
SUPPORT_NEGATIVE_PERCENT            = 0.25
SUPPORT_MIN_DURATION_HOURS          = 0.5
SUPPORT_MAX_ONSET_SHIFT_HOURS       = 18.0
MIN_SUPPORTED_STATIONS_CORE         = 2

# ── Funciones del detector ────────────────────────────────────────────────────
def infer_cadence_minutes(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return np.nan
    diffs = index.to_series().diff().dropna().dt.total_seconds().div(60.0)
    if diffs.empty:
        return np.nan
    return float(diffs.mode().iloc[0])


def rolling_window_points(index: pd.DatetimeIndex, hours: float) -> int:
    cadence_min = infer_cadence_minutes(index)
    if not np.isfinite(cadence_min) or cadence_min <= 0:
        raise ValueError("Cannot infer a valid cadence from the index.")
    return max(int(round(hours * 60.0 / cadence_min)), 1)


def longest_true_run(mask: pd.Series) -> int:
    if mask.empty:
        return 0
    arr = mask.fillna(False).astype(bool).to_numpy()
    best = cur = 0
    for x in arr:
        if x:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def contiguous_true_segments(mask: pd.Series) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(mask.fillna(False).to_numpy())
    if len(idx) == 0:
        return []
    segments = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segments.append((start, prev))
            start = prev = i
    segments.append((start, prev))
    return segments


def merge_nearby_station_events(
    events: pd.DataFrame,
    time_col: str = "onset_time",
    merge_gap_hours: float = SAME_STATION_EVENT_MERGE_GAP_HOURS,
) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    ev = events.sort_values(time_col).reset_index(drop=True)
    gap = pd.Timedelta(hours=merge_gap_hours)
    kept = []
    current = ev.iloc[0].copy()
    for i in range(1, len(ev)):
        row = ev.iloc[i]
        if row[time_col] - current[time_col] <= gap:
            if row["drop_percent"] > current["drop_percent"]:
                current = row.copy()
        else:
            kept.append(current)
            current = row.copy()
    kept.append(current)
    return pd.DataFrame(kept).reset_index(drop=True)


def detect_station_events(panel: pd.DataFrame, **pars) -> pd.DataFrame:
    df = panel.copy()
    cadence_min = infer_cadence_minutes(df.index)
    pts_per_hour = max(int(round(60.0 / cadence_min)), 1)

    min_a_pts = max(int(round(pars["min_a_duration_hours"] * pts_per_hour)), 1)
    pre_pts = max(int(round(pars["pre_event_hours"] * pts_per_hour)), 1)
    neg_confirm_pts = max(int(round(pars["confirm_negative_duration_hours"] * pts_per_hour)), 1)
    max_to_min_pts = max(int(round(pars["max_time_to_min_hours"] * pts_per_hour)), 1)
    max_rec_pts = max(int(round(pars["max_recovery_days"] * 24.0 * pts_per_hour)), 1)

    high_a = df["A_complex"] >= pars["a_thresh"]
    segments = contiguous_true_segments(high_a)

    events = []
    for s0, s1 in segments:
        if (s1 - s0 + 1) < min_a_pts:
            continue

        onset_idx = s0
        onset_time = df.index[onset_idx]
        segment_start = df.index[s0]
        segment_end = df.index[s1]

        pre_slice = df["deltaN"].iloc[max(0, onset_idx - pre_pts):onset_idx]
        if pre_slice.notna().sum() < max(int(0.6 * pre_pts), 3):
            continue

        pre_level = float(pre_slice.median())
        search_slice = df["deltaN"].iloc[onset_idx:min(len(df), onset_idx + max_to_min_pts + 1)]
        if search_slice.dropna().empty:
            continue

        neg_thr = pre_level - pars["confirm_negative_percent"]
        if longest_true_run(search_slice <= neg_thr) < neg_confirm_pts:
            continue

        min_time = search_slice.idxmin()
        min_val = float(search_slice.loc[min_time])
        drop = float(pre_level - min_val)
        if not np.isfinite(drop) or drop < pars["min_drop_percent"]:
            continue

        j_min = df.index.get_loc(min_time)
        rec_slice = df["deltaN"].iloc[j_min:min(len(df), j_min + max_rec_pts + 1)]
        target = pre_level - (1.0 - pars["recovery_fraction"]) * drop
        rec_candidates = rec_slice[rec_slice >= target]
        rec_time = rec_candidates.index[0] if not rec_candidates.empty else pd.NaT

        events.append(
            {
                "station": str(df["station"].iloc[0]),
                "segment_start": segment_start,
                "segment_end": segment_end,
                "onset_time": onset_time,
                "min_time": min_time,
                "rec_time": rec_time,
                "deltaN_pre": pre_level,
                "deltaN_min": min_val,
                "drop_percent": drop,
                "A_complex_onset": float(df.iloc[onset_idx]["A_complex"]),
                "A_complex_max_segment": float(df["A_complex"].iloc[s0:s1 + 1].max()),
                "segment_duration_h": (segment_end - segment_start).total_seconds() / 3600.0,
                "onset_to_min_h": (min_time - onset_time).total_seconds() / 3600.0,
            }
        )

    out = pd.DataFrame(events)
    if out.empty:
        return out
    out = merge_nearby_station_events(out, time_col="onset_time")
    return out.sort_values("onset_time").reset_index(drop=True)


def collapse_same_station(group_df: pd.DataFrame) -> pd.DataFrame:
    if group_df.empty:
        return group_df.copy()
    group_df = group_df.sort_values(
        ["station", "drop_percent", "A_complex_max_segment"],
        ascending=[True, False, False],
    )
    return group_df.drop_duplicates(subset=["station"], keep="first").copy()


def same_physical_trigger(
    row: pd.Series,
    group_df: pd.DataFrame,
    onset_window_h: float = TRIGGER_GROUP_ONSET_WINDOW_HOURS,
    overlap_tolerance_h: float = TRIGGER_GROUP_OVERLAP_TOL_HOURS,
) -> bool:
    if group_df.empty:
        return True

    onset_window = pd.Timedelta(hours=onset_window_h)
    overlap_tol = pd.Timedelta(hours=overlap_tolerance_h)

    group_anchor = pd.to_datetime(group_df["onset_time"]).min()
    row_onset = pd.to_datetime(row["onset_time"])
    cond_onset = abs(row_onset - group_anchor) <= onset_window

    row_start = pd.to_datetime(row["segment_start"])
    row_end = pd.to_datetime(row["min_time"])
    g_start = pd.to_datetime(group_df["segment_start"]).min()
    g_end = pd.to_datetime(group_df["min_time"]).max()

    latest_start = max(row_start, g_start)
    earliest_end = min(row_end, g_end)
    overlap = earliest_end - latest_start
    cond_overlap = overlap >= -overlap_tol

    return bool(cond_onset or cond_overlap)


def build_trigger_groups(events_df: pd.DataFrame, onset_window_h: float = TRIGGER_GROUP_ONSET_WINDOW_HOURS, overlap_tolerance_h: float = TRIGGER_GROUP_OVERLAP_TOL_HOURS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if events_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    ev = events_df.sort_values(["onset_time", "min_time", "station"]).reset_index(drop=True)
    assigned = np.zeros(len(ev), dtype=bool)
    groups: List[pd.DataFrame] = []

    for i in range(len(ev)):
        if assigned[i]:
            continue
        current_idx = [i]
        assigned[i] = True
        changed = True
        while changed:
            changed = False
            current_group = collapse_same_station(ev.loc[current_idx].copy())
            for j in range(len(ev)):
                if assigned[j]:
                    continue
                if same_physical_trigger(ev.loc[j], current_group, onset_window_h=onset_window_h, overlap_tolerance_h=overlap_tolerance_h):
                    current_idx.append(j)
                    assigned[j] = True
                    changed = True
        final_group = collapse_same_station(ev.loc[current_idx].copy())
        groups.append(final_group.sort_values(["onset_time", "min_time", "station"]).reset_index(drop=True))

    rows = []
    members = []
    for gid, g in enumerate(groups, start=1):
        onset_times = pd.to_datetime(g["onset_time"])
        min_times = pd.to_datetime(g["min_time"])
        # repr_time: tiempo de mínimo de la estación más profunda
        # (más coherente con el mínimo global registrado por FEID
        #  que la mediana de mínimos de todas las estaciones)
        if "deltaN_min" in g.columns and g["deltaN_min"].notna().any():
            deepest_idx = g["deltaN_min"].idxmin()   # estación más profunda
            repr_time   = pd.Timestamp(g.loc[deepest_idx, "min_time"])
        elif min_times.notna().any():
            repr_time   = min_times.min()             # fallback: mínimo temporal
        else:
            repr_time   = onset_times.median()        # fallback final
        row = {
            "group_id": gid,
            "repr_time": repr_time,
            "anchor_onset_time": onset_times.median(),
            "time_start": onset_times.min(),
            "time_end": min_times.max(),
            "group_span_h": (min_times.max() - onset_times.min()).total_seconds() / 3600.0,
            "n_trigger_events": len(g),
            "n_trigger_stations": g["station"].nunique(),
            "trigger_stations": ",".join(sorted(g["station"].unique())),
            "trigger_drop_mean": g["drop_percent"].mean(),
            "trigger_drop_max": g["drop_percent"].max(),
            "trigger_A_complex_mean": g["A_complex_max_segment"].mean(),
            "onset_min": onset_times.min(),
            "onset_max": onset_times.max(),
            "min_time_min": min_times.min(),
            "min_time_max": min_times.max(),
        }
        rows.append(row)

        gm = g.copy()
        gm["group_id"] = gid
        gm["repr_time"] = repr_time
        gm["anchor_onset_time"] = onset_times.median()
        members.append(gm)

    catalog = pd.DataFrame(rows).sort_values("repr_time").reset_index(drop=True)
    trigger_members = pd.concat(members, ignore_index=True) if members else pd.DataFrame()
    return catalog, trigger_members


def station_support_signature(
    panel: pd.DataFrame,
    anchor_onset_time: pd.Timestamp,
    pre_hours: float = SUPPORT_PRE_HOURS,
    post_hours: float = SUPPORT_POST_HOURS,
    min_drop_percent: float = SUPPORT_MIN_DROP_PERCENT,
    negative_percent: float = SUPPORT_NEGATIVE_PERCENT,
    min_duration_hours: float = SUPPORT_MIN_DURATION_HOURS,
    max_onset_shift_hours: float = SUPPORT_MAX_ONSET_SHIFT_HOURS,
) -> Dict[str, object]:
    cadence_min = infer_cadence_minutes(panel.index)
    pts_per_hour = max(int(round(60.0 / cadence_min)), 1)
    neg_confirm_pts = max(int(round(min_duration_hours * pts_per_hour)), 1)

    t0 = pd.Timestamp(anchor_onset_time)
    pre_slice = panel.loc[t0 - pd.Timedelta(hours=pre_hours): t0, "deltaN"]
    post_slice = panel.loc[t0: t0 + pd.Timedelta(hours=post_hours), "deltaN"]

    out = {
        "support_flag": False,
        "support_pre_level": np.nan,
        "support_min_time": pd.NaT,
        "support_min_deltaN": np.nan,
        "support_drop_percent": np.nan,
        "support_onset_time": pd.NaT,
        "support_onset_shift_h": np.nan,
        "support_negative_run_pts": 0,
        "support_reason": "insufficient_data",
    }

    if pre_slice.notna().sum() < max(12, int(0.5 * len(pre_slice))):
        return out
    if post_slice.notna().sum() < max(12, int(0.3 * len(post_slice))):
        return out

    pre_level = float(pre_slice.median())
    thr = pre_level - negative_percent

    below = post_slice <= thr
    best_run = longest_true_run(below)

    post_valid = post_slice.dropna()
    if post_valid.empty:
        return out

    min_time = post_valid.idxmin()
    min_val = float(post_valid.min())
    drop = float(pre_level - min_val)

    onset_candidates = post_slice[post_slice <= thr]
    onset_time = onset_candidates.index[0] if not onset_candidates.empty else pd.NaT
    onset_shift_h = (onset_time - t0).total_seconds() / 3600.0 if pd.notna(onset_time) else np.nan

    support_flag = bool(
        np.isfinite(drop)
        and drop >= min_drop_percent
        and best_run >= neg_confirm_pts
        and pd.notna(onset_time)
        and abs(onset_shift_h) <= max_onset_shift_hours
    )

    out.update(
        {
            "support_flag": support_flag,
            "support_pre_level": pre_level,
            "support_min_time": min_time,
            "support_min_deltaN": min_val,
            "support_drop_percent": drop,
            "support_onset_time": onset_time,
            "support_onset_shift_h": onset_shift_h,
            "support_negative_run_pts": int(best_run),
            "support_reason": "passed" if support_flag else "failed_thresholds",
        }
    )
    return out


def assign_quality_label(n_supported_stations: int, n_trigger_stations: int) -> str:
    if n_supported_stations >= 4:
        return "A"
    if n_supported_stations >= 2:
        return "B"
    if n_supported_stations == 1 and n_trigger_stations >= 1:
        return "C"
    return "D"


def promote_with_station_support(
    trigger_catalog: pd.DataFrame,
    trigger_members: pd.DataFrame,
    station_panels: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trigger_catalog.empty:
        return pd.DataFrame(), pd.DataFrame()

    support_rows = []
    promoted_rows = []

    for _, ev in trigger_catalog.iterrows():
        gid = int(ev["group_id"])
        anchor = pd.Timestamp(ev["anchor_onset_time"])
        trig = trigger_members[trigger_members["group_id"] == gid].copy()
        trigger_stations = set(trig["station"].astype(str))

        per_station_rows = []
        for st, panel in station_panels.items():
            sig = station_support_signature(panel, anchor)
            is_trigger = st in trigger_stations
            support_flag = bool(sig["support_flag"] or is_trigger)
            reason = "trigger" if is_trigger else sig["support_reason"]

            row = {
                "group_id": gid,
                "repr_time": ev["repr_time"],
                "anchor_onset_time": anchor,
                "station": st,
                "is_trigger_station": is_trigger,
                "support_flag": support_flag,
                "support_reason": reason,
                **sig,
            }
            if is_trigger:
                row["support_onset_time"] = trig.loc[trig["station"] == st, "onset_time"].iloc[0]
                row["support_min_time"] = trig.loc[trig["station"] == st, "min_time"].iloc[0]
                row["support_min_deltaN"] = trig.loc[trig["station"] == st, "deltaN_min"].iloc[0]
                row["support_drop_percent"] = trig.loc[trig["station"] == st, "drop_percent"].iloc[0]
                row["support_flag"] = True
            per_station_rows.append(row)
            support_rows.append(row)

        sup_df = pd.DataFrame(per_station_rows)
        sup_yes = sup_df[sup_df["support_flag"]].copy()

        n_supported = int(sup_yes["station"].nunique())
        supported_stations = ",".join(sorted(sup_yes["station"].astype(str).unique()))
        trigger_stations_str = ",".join(sorted(trigger_stations))
        n_trigger = int(len(trigger_stations))

        support_only = sup_yes[~sup_yes["is_trigger_station"]].copy()
        n_support_only = int(support_only["station"].nunique())
        support_only_stations = ",".join(sorted(support_only["station"].astype(str).unique()))

        promoted_rows.append(
            {
                **ev.to_dict(),
                "n_supported_stations": n_supported,
                "supported_stations": supported_stations,
                "n_support_only_stations": n_support_only,
                "support_only_stations": support_only_stations,
                "mean_supported_drop": sup_yes["support_drop_percent"].mean() if not sup_yes.empty else np.nan,
                "median_supported_drop": sup_yes["support_drop_percent"].median() if not sup_yes.empty else np.nan,
                "max_supported_drop": sup_yes["support_drop_percent"].max() if not sup_yes.empty else np.nan,
                "catalog_tier": "core" if n_supported >= MIN_SUPPORTED_STATIONS_CORE else "exploratory",
                "fd_scope": "supported_multi_station_candidate" if n_supported >= MIN_SUPPORTED_STATIONS_CORE else "single_station_candidate",
                "quality": assign_quality_label(n_supported, n_trigger),
                "promoted_from_single_trigger": bool(n_trigger == 1 and n_supported >= 2),
                "trigger_stations": trigger_stations_str,
                "n_trigger_stations": n_trigger,
            }
        )

    promoted_catalog = pd.DataFrame(promoted_rows).sort_values("repr_time").reset_index(drop=True)
    support_members = pd.DataFrame(support_rows).sort_values(["group_id", "station"]).reset_index(drop=True)
    promoted_catalog["year"] = pd.to_datetime(promoted_catalog["repr_time"]).dt.year
    return promoted_catalog, support_members


def one_to_one_time_match(
    ref_times: Sequence[pd.Timestamp],
    cand_times: Sequence[pd.Timestamp],
    tolerance_hours: float = 18.0,
) -> pd.DataFrame:
    tol = pd.Timedelta(hours=tolerance_hours)
    pairs = []
    for i, t_ref in enumerate(ref_times):
        for j, t_cand in enumerate(cand_times):
            dt = t_cand - t_ref
            if abs(dt) <= tol:
                pairs.append((i, j, dt))
    if not pairs:
        return pd.DataFrame(columns=["ref_pos", "cand_pos", "dt_hours"])

    pairs = sorted(pairs, key=lambda x: abs(x[2].total_seconds()))
    used_ref = set()
    used_cand = set()
    kept = []
    for i, j, dt in pairs:
        if i in used_ref or j in used_cand:
            continue
        used_ref.add(i)
        used_cand.add(j)
        kept.append({"ref_pos": i, "cand_pos": j, "dt_hours": dt.total_seconds() / 3600.0})
    return pd.DataFrame(kept)

# ── Configuración ─────────────────────────────────────────────────────────────
BASE         = Path("Results_v2_4")
PANELS_DIR   = BASE / "station_panels"
FEID_PARQUET = Path("feid_clean.parquet")
OUT          = BASE / "sensitivity_grouping_active"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_VALUES  = [6, 8, 10, 12, 18]
MATCH_TOL_H    = 18.0
ACTIVE_YEARS   = [2022, 2023, 2024]
SAVE_FIGS      = True
DPI            = 220

BASE_PARAMS = {
    "a_thresh":                        2.0,
    "min_a_duration_hours":            2.0,
    "min_drop_percent":                1.0,
    "confirm_negative_percent":        0.30,
    "confirm_negative_duration_hours": 0.5,
    "max_time_to_min_hours":           36.0,
    "pre_event_hours":                 24.0,
    "recovery_fraction":               0.50,
    "max_recovery_days":               7.0,
}

def savefig(name):
    if SAVE_FIGS:
        plt.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close()

def eval_catalog(catalog_df, feid_subset, tol_h=18.0, label=""):
    ref  = list(feid_subset["t_izmiran"])
    cand = list(pd.to_datetime(catalog_df["repr_time"]))
    m    = one_to_one_time_match(ref, cand, tolerance_hours=tol_h)
    nr, nc, nm = len(ref), len(cand), len(m)
    rec  = nm/nr  if nr else np.nan
    prec = nm/nc  if nc else np.nan
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else np.nan
    return {"label":label,"N_ref":nr,"N_cat":nc,"N_match":nm,
            "recall":rec,"precision":prec,"f1":f1}

def fn_large_recovered(catalog_df, feid_lg, tol_h=18.0):
    tol  = pd.Timedelta(hours=tol_h)
    cand = pd.to_datetime(catalog_df["repr_time"])
    return sum(
        any(abs(pd.Timestamp(c) - t) <= tol for c in cand)
        for t in feid_lg["t_izmiran"]
    )

# ── 1. Cargar station panels ──────────────────────────────────────────────────
print("[1/4] Cargando station panels...", flush=True)
station_panels = {}
for pf in sorted(PANELS_DIR.glob("*_panel.csv")):
    st = pf.stem.replace("_panel","")
    p  = pd.read_csv(pf)
    tc = next((c for c in ["DATETIME","datetime","index","time"]
                if c in p.columns), None)
    if tc:
        p[tc] = pd.to_datetime(p[tc], utc=True, errors="coerce")
        p[tc] = p[tc].dt.tz_convert(None)
        p = p.set_index(tc)
    if "A_complex" in p.columns:
        station_panels[st] = p
        print(f"  checkp {st}", flush=True)
print(f"  Total: {len(station_panels)} estaciones")

# ── 2. Cargar FEID ────────────────────────────────────────────────────────────
print("\n[2/4] Cargando FEID...", flush=True)
feid = pd.read_parquet(FEID_PARQUET)
feid.index = pd.to_datetime(feid.index)
feid["t_izmiran"] = feid.index
if feid["t_izmiran"].dt.tz is not None:
    feid["t_izmiran"] = feid["t_izmiran"].dt.tz_convert(None)
feid["year"] = feid["t_izmiran"].dt.year

feid_active = feid[feid["year"].isin(ACTIVE_YEARS)].copy()
feid_2019   = feid[feid["year"] == 2019].copy()
feid_large  = feid[feid["magnitude"] >= 1.0].copy()
feid_large_active = feid_large[feid_large["year"].isin(ACTIVE_YEARS)].copy()

print(f"  FEID activo (2022-2024): {len(feid_active)}")
print(f"  FEID 2019:               {len(feid_2019)}")
print(f"  FEID mag>=1%:            {len(feid_large)}")
print(f"  FEID mag>=1% activo:     {len(feid_large_active)}")

# ── 3. Loop de sensibilidad ───────────────────────────────────────────────────
print("\n[3/4] Corriendo sensibilidad...", flush=True)
results    = []
yearly_all = {}

for window_h in WINDOW_VALUES:
    print(f"\n  window = {window_h}h ...", flush=True)

    all_events = []
    for st, panel in station_panels.items():
        ev = detect_station_events(panel, **BASE_PARAMS)
        if not ev.empty:
            all_events.append(ev)

    trigger_events = pd.concat(all_events, ignore_index=True) \
                     if all_events else pd.DataFrame()

    trigger_catalog, trigger_members = build_trigger_groups(
        trigger_events,
        onset_window_h=window_h,
        overlap_tolerance_h=TRIGGER_GROUP_OVERLAP_TOL_HOURS,
    )

    after_catalog, _ = promote_with_station_support(
        trigger_catalog, trigger_members, station_panels)

    after_catalog["year"] = pd.to_datetime(after_catalog["repr_time"]).dt.year
    core = after_catalog[after_catalog["n_supported_stations"] >= 2].copy()

    cat_2019  = after_catalog[after_catalog["year"]==2019]
    core_2019 = core[core["year"]==2019]
    cat_act   = after_catalog[after_catalog["year"].isin(ACTIVE_YEARS)]
    core_act  = core[core["year"].isin(ACTIVE_YEARS)]

    m19_all   = eval_catalog(cat_2019,  feid_2019,   MATCH_TOL_H, "all_2019")
    m19_core  = eval_catalog(core_2019, feid_2019,   MATCH_TOL_H, "core_2019")
    m_act_all = eval_catalog(cat_act,   feid_active, MATCH_TOL_H, "all_active")
    m_act_core= eval_catalog(core_act,  feid_active, MATCH_TOL_H, "core_active")

    fn_rec_act = fn_large_recovered(after_catalog, feid_large_active, MATCH_TOL_H)
    fn_rec_all = fn_large_recovered(after_catalog, feid_large, MATCH_TOL_H)

    yr_counts = after_catalog.groupby("year").size().rename(f"W_{window_h}h")
    yearly_all[window_h] = yr_counts

    n_prom = int(after_catalog["promoted_from_single_trigger"].sum()) \
             if "promoted_from_single_trigger" in after_catalog.columns else 0

    results.append({
        "window_h":       window_h,
        "N_total":        len(after_catalog),
        "N_core":         len(core),
        "N_promoted":     n_prom,
        "f1_all_2019":    round(m19_all["f1"],   3),
        "f1_core_2019":   round(m19_core["f1"],  3),
        "rec_all_2019":   round(m19_all["recall"],3),
        "prec_all_2019":  round(m19_all["precision"],3),
        "f1_all_active":  round(m_act_all["f1"],  3),
        "f1_core_active": round(m_act_core["f1"], 3),
        "rec_all_active": round(m_act_all["recall"],3),
        "prec_all_active":round(m_act_all["precision"],3),
        "N_active_match": m_act_all["N_match"],
        "fn_large_active":fn_rec_act,
        "fn_large_total": fn_rec_all,
    })

    print(f"    N={len(after_catalog)}  "
          f"F1_2019={m19_all['f1']:.3f}  "
          f"F1_active={m_act_all['f1']:.3f}  "
          f"FN_large_active={fn_rec_act}/{len(feid_large_active)}")

# ── 4. Tablas y figuras ───────────────────────────────────────────────────────
print("\n[4/4] Tablas y figuras...", flush=True)

sens = pd.DataFrame(results)
print("\nTabla de sensibilidad:")
print(sens.to_string(index=False))
sens.to_csv(OUT / "sensitivity_grouping_active_summary.csv", index=False)

yr_df = pd.concat(yearly_all.values(), axis=1).fillna(0).astype(int)
yr_df.index.name = "year"
yr_df.to_csv(OUT / "sensitivity_grouping_active_annual.csv")

# Fig 1: F1 vs ventana
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(sens["window_h"], sens["f1_all_2019"],
             marker="o", color="#2166ac", label="All (2019)", lw=1.8)
axes[0].plot(sens["window_h"], sens["f1_core_2019"],
             marker="s", color="#2166ac", ls="--", label="Core (2019)", lw=1.2)
axes[0].plot(sens["window_h"], sens["f1_all_active"],
             marker="o", color="#d7191c", label="All (2022-2024)", lw=1.8)
axes[0].plot(sens["window_h"], sens["f1_core_active"],
             marker="s", color="#d7191c", ls="--", label="Core (2022-2024)", lw=1.2)
axes[0].axvline(18, color="gray", ls=":", lw=1, label="Current (18h)")
axes[0].set_xlabel("Grouping window (h)"); axes[0].set_ylabel("F1 score")
axes[0].set_title("Validation F1: solar minimum vs active years")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].plot(sens["window_h"], sens["fn_large_active"],
             marker="o", color="#d7191c", lw=1.8,
             label=f"Recovered 2022-2024 (of {len(feid_large_active)})")
axes[1].plot(sens["window_h"], sens["fn_large_total"],
             marker="s", color="#4d4d4d", lw=1.2, ls="--",
             label=f"Recovered all years (of {len(feid_large)})")
axes[1].axvline(18, color="gray", ls=":", lw=1, label="Current (18h)")
axes[1].set_xlabel("Grouping window (h)")
axes[1].set_ylabel("FN large events recovered")
axes[1].set_title(r"FEID mag$\geq$1% recovered vs grouping window")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
savefig("fig_sensitivity_grouping_active.png")

# Fig 2: conteos anuales
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#d73027","#fc8d59","#2c7bb6","#31a354","#636363"]
for wh, color in zip(WINDOW_VALUES, colors):
    col = f"W_{wh}h"
    if col in yr_df.columns:
        lw = 2.0 if wh==18 else 1.2
        ls = "-"  if wh==18 else "--"
        ax.plot(yr_df.index, yr_df[col], marker="o",
                color=color, lw=lw, ls=ls,
                label=f"Window={wh}h" + (" (current)" if wh==18 else ""))
ax.axvspan(2021.5, 2024.5, alpha=0.06, color="red")
ax.set_xlabel("Year"); ax.set_ylabel("Number of events")
ax.set_title("Annual counts by grouping window")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("fig_sensitivity_grouping_active_annual.png")

print(f"\n✓ Outputs en: {OUT.resolve()}")
