#!/usr/bin/env python3
from __future__ import annotations
"""
AFTER_detector.py  —  v2.4
===========================
Script unificado del detector AFTER (Annotated Forbush-decrease
Topological Event Recognition).

Integra en un único flujo:
  1. Construcción de paneles de complejidad por estación (con checkpoint)
  2. Detección de triggers y grouping
  3. Support-based promotion
  4. Enriquecimiento FEID/OMNI (contexto interplanetario)
  5. Confirmación de huella FD-like (fd_like_confirmed)
  6. Validación contra FEID 2019

CHECKPOINTING:
  Los paneles de estación se guardan en OUTPUT_DIR/station_panels/.
  Si ya existen, se cargan directamente sin recalcular.
  Para forzar recálculo completo: --force-rebuild

Uso:
  python AFTER_detector.py              # usa checkpoints si existen
  python AFTER_detector.py --force-rebuild  # recalcula todo
  python AFTER_detector.py --skip-panels    # solo re-corre desde triggers

Inputs (mismo directorio):
  alldata_integrated_2min.parquet
  feid_clean.parquet
  dataset_meta.json

Outputs en OUTPUT_DIR (por defecto Results_v2_4/):
  station_panels/          paneles de complejidad por estación
  AFTER_catalog_detector.csv          catálogo principal enriquecido
  AFTER_catalog_detector_core.csv
  AFTER_catalog_detector_exploratory.csv
  trigger_events_detector.csv
  trigger_catalog_detector.csv
  trigger_members_detector.csv
  support_members_detector.csv
  validation_summary_detector.csv
  promotion_summary_detector.csv
  station_coverage_report.csv
  feid_catalog_parsed.csv
  yearly_physical_context.csv
  fd_like_confirmation_summary.csv
  match_table_detector_*.csv
"""
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="AFTER detector v2.4")
parser.add_argument("--force-rebuild", action="store_true",
                    help="Recalcular todos los paneles aunque existan en disco")
parser.add_argument("--skip-panels",  action="store_true",
                    help="Saltar construcción de paneles y cargar desde disco")
parser.add_argument("--output-dir", default="Results_v2_4",
                    help="Directorio de salida (default: Results_v2_4)")
args, _ = parser.parse_known_args()


import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:

# ── Funciones del detector ───────────────────────────────────────────────────
    def display(x):
        print(x)


# ============================================================
# Paths
# ============================================================
ALLDATA_PARQUET   = "alldata_integrated_2min.parquet"  # NMDB + OMNI integrado
FEID_PARQUET       = "feid_clean.parquet"               # catálogo FEID limpio
DATASET_META_JSON  = "dataset_meta.json"                # columnas por bloque

# OUTPUT_DIR se define en __main__ via --output-dir (default: Results_v2_4)

# ============================================================
# Stations
# ============================================================
# Stations se cargan desde dataset_meta.json en el bloque Main run
# STATIONS se define dinámicamente tras leer el parquet
TIME_COL = "DATETIME"  # no usado con parquet, pero se conserva por compatibilidad

# ============================================================
# Preprocessing parameters
# ============================================================
SPIKE_MAD_FACTOR = 6.0
BACKGROUND_HOURS = 72
BACKGROUND_MIN_FRAC = 0.70
COMPLEXITY_WINDOW_HOURS = 3
COMPLEXITY_MIN_FRAC = 0.70
PE_ORDER = 3
PE_DELAY = 1
ROBUST_REF_HOURS = 24 * 30
ROBUST_REF_MIN_FRAC = 0.50
ROBUST_Z_EPS = 1e-9

# ============================================================
# Single detector (former loose)
# ============================================================
DETECTOR_PARAMS = {
    "a_thresh": 2.0,
    "min_a_duration_hours": 2.0,
    "min_drop_percent": 1.0,
    "confirm_negative_percent": 0.30,
    "confirm_negative_duration_hours": 0.5,
    "max_time_to_min_hours": 36.0,
    "pre_event_hours": 24.0,
    "recovery_fraction": 0.50,
    "max_recovery_days": 7.0,
}

# ============================================================
# Trigger grouping + support-based promotion
# ============================================================
TRIGGER_GROUP_ONSET_WINDOW_HOURS = 18.0
TRIGGER_GROUP_OVERLAP_TOL_HOURS = 12.0
SAME_STATION_EVENT_MERGE_GAP_HOURS = 12.0

SUPPORT_PRE_HOURS = 24.0
SUPPORT_POST_HOURS = 36.0
SUPPORT_MIN_DROP_PERCENT = 0.75
SUPPORT_NEGATIVE_PERCENT = 0.25
SUPPORT_MIN_DURATION_HOURS = 0.5
SUPPORT_MAX_ONSET_SHIFT_HOURS = 18.0
MIN_SUPPORTED_STATIONS_CORE = 2

MATCH_TOLERANCES_H = [6, 12, 18, 24]
DEFAULT_MATCH_TOLERANCE_H = 18


# ============================================================
# Generic utilities
# ============================================================
def load_nmdb_matrix(
    parquet_path: str | os.PathLike,
    meta_json: str | os.PathLike,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Carga el dataset integrado (NMDB + OMNI) desde parquet.
    Devuelve: df_nmdb, df_omni, nmdb_cols, omni_cols
    """
    import json
    with open(meta_json) as f:
        meta = json.load(f)
    nmdb_cols = meta["nmdb_cols"]
    omni_cols = meta["omni_cols"]

    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Separar bloques
    available_nmdb = [c for c in nmdb_cols if c in df.columns]
    available_omni = [c for c in omni_cols if c in df.columns]

    df_nmdb = df[available_nmdb].copy()
    df_omni = df[available_omni].copy()

    print(f"  NMDB: {len(available_nmdb)} estaciones, {len(df_nmdb):,} filas")
    print(f"  OMNI: {len(available_omni)} variables, {len(df_omni):,} filas")
    print(f"  Rango: {df_nmdb.index.min()} → {df_nmdb.index.max()}")

    return df_nmdb, df_omni, available_nmdb, available_omni


def infer_cadence_minutes(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return np.nan
    diffs = index.to_series().diff().dropna().dt.total_seconds().div(60.0)
    if diffs.empty:
        return np.nan
    return float(diffs.mode().iloc[0])


def regularity_report(index: pd.DatetimeIndex) -> pd.Series:
    diffs = index.to_series().diff().dropna().dt.total_seconds().div(60.0)
    out = {
        "n_rows": len(index),
        "start": index.min(),
        "end": index.max(),
        "cadence_mode_min": diffs.mode().iloc[0] if not diffs.empty else np.nan,
        "cadence_median_min": diffs.median() if not diffs.empty else np.nan,
        "n_irregular_steps": int((diffs != diffs.mode().iloc[0]).sum()) if not diffs.empty else 0,
    }
    return pd.Series(out, dtype="object")


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


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ============================================================
# Complexity helpers
# ============================================================
def causal_rolling_median(series: pd.Series, window_pts: int, min_frac: float = 0.7) -> pd.Series:
    min_periods = max(int(math.ceil(window_pts * min_frac)), 1)
    return series.rolling(window_pts, min_periods=min_periods).median().shift(1)


def robust_spike_mask(series: pd.Series, mad_factor: float = 6.0) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=series.index)
    return (series - med).abs() > (mad_factor * mad)


def remove_spikes(series: pd.Series, mad_factor: float = 6.0) -> Tuple[pd.Series, pd.Series]:
    mask = robust_spike_mask(series, mad_factor=mad_factor)
    return series.mask(mask), mask


def ordinal_pattern(window: np.ndarray) -> Tuple[int, ...]:
    window = np.asarray(window, dtype=float)
    if np.isnan(window).any():
        raise ValueError("ordinal_pattern received NaNs.")
    eps = 1e-12 * np.arange(len(window), dtype=float)
    return tuple(np.argsort(window + eps, kind="mergesort"))


def permutation_entropy(x: Sequence[float], m: int = 3, delay: int = 1) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < (m - 1) * delay + 1:
        return np.nan

    patterns: List[Tuple[int, ...]] = []
    for i in range(n - (m - 1) * delay):
        win = x[i : i + m * delay : delay]
        if len(win) == m and np.isfinite(win).all():
            patterns.append(ordinal_pattern(win))

    if not patterns:
        return np.nan

    counts = pd.Series(patterns).value_counts().to_numpy(dtype=float)
    probs = counts / counts.sum()
    pe = -(probs * np.log(probs)).sum()
    return float(pe / math.log(math.factorial(m)))


def katz_fd(x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan

    diffs = np.diff(x)
    L = np.sum(np.sqrt(1.0 + diffs**2))
    t = np.arange(n, dtype=float)
    d = np.max(np.sqrt((t - t[0])**2 + (x - x[0])**2))
    if not np.isfinite(L) or not np.isfinite(d) or L <= 0 or d <= 0:
        return np.nan
    return float(math.log10(n) / (math.log10(n) + math.log10(d / L)))


def rolling_complexity(
    series: pd.Series,
    window_pts: int,
    min_frac: float = 0.7,
    m: int = 3,
    delay: int = 1,
) -> pd.DataFrame:
    min_periods = max(int(math.ceil(window_pts * min_frac)), 1)
    pe = series.rolling(window_pts, min_periods=min_periods).apply(
        lambda x: permutation_entropy(x, m=m, delay=delay), raw=True
    )
    kfd = series.rolling(window_pts, min_periods=min_periods).apply(katz_fd, raw=True)
    return pd.DataFrame({"PE": pe, "KFD": kfd}, index=series.index)


def trailing_robust_z(
    series: pd.Series,
    ref_pts: int,
    min_frac: float = 0.5,
    eps: float = 1e-9,
) -> pd.Series:
    min_periods = max(int(math.ceil(ref_pts * min_frac)), 1)
    med = series.rolling(ref_pts, min_periods=min_periods).median().shift(1)
    mad = (series - med).abs().rolling(ref_pts, min_periods=min_periods).median().shift(1)
    scale = 1.4826 * mad
    z = (series - med) / (scale + eps)
    z[(scale.isna()) | (scale <= eps)] = np.nan
    return z


def euclidean_complexity_amplitude(z1: pd.Series, z2: pd.Series) -> pd.Series:
    n_valid = z1.notna().astype(int) + z2.notna().astype(int)
    amp = np.sqrt(np.square(z1.fillna(0.0)) + np.square(z2.fillna(0.0)))
    amp[n_valid == 0] = np.nan
    return amp


# ============================================================
# Station panels and detector
# ============================================================
def build_station_panel(
    df: pd.DataFrame,
    station: str,
    spike_mad_factor: float = SPIKE_MAD_FACTOR,
    background_hours: float = BACKGROUND_HOURS,
    background_min_frac: float = BACKGROUND_MIN_FRAC,
    complexity_window_hours: float = COMPLEXITY_WINDOW_HOURS,
    complexity_min_frac: float = COMPLEXITY_MIN_FRAC,
    pe_order: int = PE_ORDER,
    pe_delay: int = PE_DELAY,
    robust_ref_hours: float = ROBUST_REF_HOURS,
    robust_ref_min_frac: float = ROBUST_REF_MIN_FRAC,
) -> pd.DataFrame:
    if station not in df.columns:
        raise KeyError(f"Station '{station}' is missing from the NMDB matrix.")

    series = pd.to_numeric(df[station], errors="coerce").astype(float)
    counts_clean, spike_mask = remove_spikes(series, mad_factor=spike_mad_factor)

    bg_pts = rolling_window_points(df.index, background_hours)
    bg = causal_rolling_median(counts_clean, bg_pts, min_frac=background_min_frac)
    deltaN = 100.0 * (counts_clean - bg) / bg

    comp_pts = rolling_window_points(df.index, complexity_window_hours)
    comp = rolling_complexity(
        counts_clean,
        window_pts=comp_pts,
        min_frac=complexity_min_frac,
        m=pe_order,
        delay=pe_delay,
    )

    ref_pts = rolling_window_points(df.index, robust_ref_hours)
    z_pe = trailing_robust_z(comp["PE"], ref_pts=ref_pts, min_frac=robust_ref_min_frac, eps=ROBUST_Z_EPS)
    z_kfd = trailing_robust_z(comp["KFD"], ref_pts=ref_pts, min_frac=robust_ref_min_frac, eps=ROBUST_Z_EPS)
    a_complex = euclidean_complexity_amplitude(z_pe, z_kfd)

    panel = pd.DataFrame(
        {
            "counts": series,
            "counts_clean": counts_clean,
            "spike_flag": spike_mask.astype(int),
            "bg": bg,
            "deltaN": deltaN,
            "PE": comp["PE"],
            "KFD": comp["KFD"],
            "z_PE": z_pe,
            "z_KFD": z_kfd,
            "A_complex": a_complex,
        },
        index=df.index,
    )
    panel["station"] = station
    return panel


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


# ============================================================
# Trigger grouping
# ============================================================
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


def build_trigger_groups(events_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                if same_physical_trigger(ev.loc[j], current_group):
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
        # repr_time: mediana de los onsets de las estaciones del grupo
        # (anchor_onset_time) — más cercano al tiempo FEID que el mínimo
        # de la estación más profunda, especialmente en eventos multi-fase.
        # El mínimo más profundo queda disponible en min_time_min/max
        # para análisis posteriores de δt y amplitud.
        repr_time = onset_times.median() if onset_times.notna().any()                     else min_times.median()
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


# ============================================================
# Support-based promotion
# ============================================================
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


# ============================================================
# IZMIRAN validation
# ============================================================
def load_feid_catalog(parquet_path: str | os.PathLike) -> pd.DataFrame:
    """
    Carga el catálogo FEID limpio desde feid_clean.parquet.
    Estandariza a columna t_izmiran para compatibilidad con el resto del pipeline.
    """
    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Alias para compatibilidad con funciones de matching existentes
    df["t_izmiran"] = df.index
    return df.reset_index(drop=False).rename(columns={"index": "datetime"})


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


def evaluate_catalog_vs_izmiran(
    izm_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    tolerances_h: Sequence[float] = (6, 12, 18, 24),
    year: int = 2019,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, float], pd.DataFrame]]:
    if catalog_df.empty:
        return pd.DataFrame(), {}
    cat = catalog_df[pd.to_datetime(catalog_df["repr_time"]).dt.year == year].copy()
    if cat.empty:
        return pd.DataFrame(), {}

    subsets = {
        "all": cat,
        "core": cat[cat["catalog_tier"] == "core"].copy(),
        "exploratory": cat[cat["catalog_tier"] == "exploratory"].copy(),
    }

    rows = []
    matched = {}
    for subset_name, subset_df in subsets.items():
        ref_times = list(izm_df["t_izmiran"])
        cand_times = list(pd.to_datetime(subset_df["repr_time"]))
        for tol_h in tolerances_h:
            match_df = one_to_one_time_match(ref_times, cand_times, tolerance_hours=tol_h)
            n_ref = len(ref_times)
            n_cand = len(cand_times)
            n_match = len(match_df)
            recall = n_match / n_ref if n_ref else np.nan
            precision = n_match / n_cand if n_cand else np.nan
            f1 = (2 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else np.nan
            rows.append(
                {
                    "subset": subset_name,
                    "tolerance_h": tol_h,
                    "n_izmiran": n_ref,
                    "n_catalog": n_cand,
                    "n_matched": n_match,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "median_abs_dt_h": match_df["dt_hours"].abs().median() if not match_df.empty else np.nan,
                    "mean_abs_dt_h": match_df["dt_hours"].abs().mean() if not match_df.empty else np.nan,
                }
            )
            matched[(subset_name, tol_h)] = match_df
    return pd.DataFrame(rows).sort_values(["subset", "tolerance_h"]).reset_index(drop=True), matched


def enrich_match_table(match_df: pd.DataFrame, izm_df: pd.DataFrame, subset_df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    if match_df.empty:
        return pd.DataFrame(columns=["subset", "idx_izmiran", "idx_catalog", "dt_hours"])
    ref_tbl = izm_df.reset_index(drop=True).reset_index().rename(columns={"index": "idx_izmiran"})
    cand_tbl = subset_df.reset_index(drop=True).reset_index().rename(columns={"index": "idx_catalog"})
    out = match_df.rename(columns={"ref_pos": "idx_izmiran", "cand_pos": "idx_catalog"}).copy()
    out["subset"] = subset_name
    out = out.merge(ref_tbl, on="idx_izmiran", how="left")
    out = out.merge(cand_tbl, on="idx_catalog", how="left", suffixes=("_izmiran", "_catalog"))
    return out




# ============================================================
# Contexto físico OMNI por evento AFTER
# ============================================================
def extract_omni_context(
    after_catalog: pd.DataFrame,
    df_omni: pd.DataFrame,
    feid_catalog: pd.DataFrame,
    window_pre_h: float = 48.0,
    window_post_h: float = 48.0,
    feid_match_tol_h: float = 18.0,
) -> pd.DataFrame:
    """
    Para cada evento AFTER, extrae el contexto físico OMNI en ±window h
    y el evento FEID más cercano (si existe dentro de feid_match_tol_h).

    Columnas resultantes por evento:
      omni_Bz_min       : Bz mínimo en la ventana (nT)
      omni_Speed_max     : velocidad máxima del viento solar (km/s)
      omni_IMF_max       : IMF máximo (nT)
      omni_Density_mean  : densidad media (cm⁻³)
      feid_dt_h          : diferencia temporal al FEID más cercano (h)
      feid_magnitude     : magnitud del FD FEID más cercano (%)
      feid_ons_label     : tipo de onset (SSC/SI/NaN) del FEID más cercano
      feid_Vmax          : V max del FEID más cercano
      feid_Bmax          : B max del FEID más cercano
      feid_has_shock      : bool, True si SSC o SI
      feid_matched       : bool, hay FEID dentro del tolerance
    """
    tol_feid = pd.Timedelta(hours=feid_match_tol_h)
    feid_times = pd.to_datetime(feid_catalog["t_izmiran"])

    rows = []
    for _, ev in after_catalog.iterrows():
        t0 = pd.Timestamp(ev["repr_time"])
        t_pre  = t0 - pd.Timedelta(hours=window_pre_h)
        t_post = t0 + pd.Timedelta(hours=window_post_h)

        # Ventana OMNI
        win = df_omni.loc[t_pre:t_post]

        omni_row = {
            "omni_Bz_min":      win["BZ"].min()      if "BZ"          in win and not win["BZ"].isna().all()      else np.nan,
            "omni_Speed_max":   win["Speed"].max()   if "Speed"       in win and not win["Speed"].isna().all()   else np.nan,
            "omni_IMF_max":     win["IMF_avg"].max() if "IMF_avg"     in win and not win["IMF_avg"].isna().all() else np.nan,
            "omni_Density_mean":win["Density"].mean()if "Density"     in win and not win["Density"].isna().all() else np.nan,
        }

        # FEID más cercano
        dt_feid = (feid_times - t0).abs()
        if dt_feid.min() <= tol_feid:
            idx_best = dt_feid.idxmin()
            best = feid_catalog.iloc[idx_best] if isinstance(idx_best, int) else feid_catalog.loc[idx_best]
            ons_col = "ons type" if "ons type" in best.index else "ons"
            ons_val = best.get(ons_col, np.nan)
            feid_row = {
                "feid_matched":    True,
                "feid_dt_h":       (feid_times.iloc[dt_feid.argmin()] - t0).total_seconds() / 3600.0,
                "feid_magnitude":  best.get("magnitude", np.nan),
                "feid_ons_label":  ons_val,
                "feid_Vmax":       best.get("V max", np.nan),
                "feid_Bmax":       best.get("B max", np.nan),
                "feid_has_shock":  ons_val in ["SSC", "SI"],
            }
        else:
            feid_row = {
                "feid_matched":   False,
                "feid_dt_h":      np.nan,
                "feid_magnitude": np.nan,
                "feid_ons_label": np.nan,
                "feid_Vmax":      np.nan,
                "feid_Bmax":      np.nan,
                "feid_has_shock": False,
            }

        rows.append({**omni_row, **feid_row})

    ctx = pd.DataFrame(rows, index=after_catalog.index)
    return pd.concat([after_catalog, ctx], axis=1)


# ── Funciones de confirmación FD-like ────────────────────────────────────────

# ── Constantes de confirmación FD-like ───────────────────────────────────────
BZ_THRESH_NT    = -3.0   # nT — Bz southward threshold
BZ_MIN_DUR_MIN  = 60     # minutos consecutivos mínimos para Cond A
DV_THRESH_KMS   = 30.0   # km/s — incremento de velocidad para Cond B
DV_WINDOW_H     =  6.0   # horas para calcular ΔV
IMF_THRESH_NT   =  8.0   # nT — IMF elevado para Cond C
IMF_MIN_DUR_MIN = 60     # minutos consecutivos mínimos para Cond C

def max_consecutive_below(series: pd.Series, thresh: float,
                           cadence_min: float = 2.0) -> float:
    """
    Máximo número de minutos consecutivos donde series < thresh.
    cadence_min: resolución temporal en minutos.
    """
    below = series < thresh
    if not below.any():
        return 0.0
    # Grupos consecutivos
    groups = (below != below.shift()).cumsum()
    max_run = below.groupby(groups).sum().max()
    return float(max_run) * cadence_min


def max_consecutive_above(series: pd.Series, thresh: float,
                           cadence_min: float = 2.0) -> float:
    """Máximo minutos consecutivos donde series > thresh."""
    return max_consecutive_below(-series, -thresh, cadence_min)


def max_speed_increment(series: pd.Series,
                         window_h: float = 6.0,
                         cadence_min: float = 2.0) -> float:
    """
    Máximo incremento de velocidad en cualquier ventana de window_h horas.
    ΔV = max(V) - min(V) dentro de la ventana deslizante.
    """
    if series.isna().all():
        return np.nan
    window_pts = int(window_h * 60 / cadence_min)
    if window_pts < 2:
        return np.nan
    roll_max = series.rolling(window_pts, min_periods=window_pts//2).max()
    roll_min = series.rolling(window_pts, min_periods=window_pts//2).min()
    return float((roll_max - roll_min).max())


def check_fd_like(win: pd.DataFrame,
                   bz_col: str, speed_col: str, imf_col: str,
                   cadence_min: float = 2.0) -> dict:
    """
    Evalúa las tres condiciones heliofísicas en la ventana dada.
    Devuelve dict con booleanos individuales y resultado combinado.
    """
    result = {
        "cond_A_bz":    False,
        "cond_B_speed": False,
        "cond_C_imf":   False,
        "fd_like_confirmed": False,
        "bz_min":       np.nan,
        "speed_max":    np.nan,
        "imf_max":      np.nan,
        "dv_max":       np.nan,
        "bz_consec_min":   0.0,
        "imf_consec_min":  0.0,
    }

    if win.empty:
        return result

    # Condición A — Bz southward sostenido
    if bz_col in win.columns and not win[bz_col].isna().all():
        bz = win[bz_col].dropna()
        result["bz_min"] = float(bz.min())
        consec = max_consecutive_below(win[bz_col], BZ_THRESH_NT, cadence_min)
        result["bz_consec_min"] = consec
        result["cond_A_bz"] = consec >= BZ_MIN_DUR_MIN

    # Condición B — incremento de velocidad
    if speed_col in win.columns and not win[speed_col].isna().all():
        v = win[speed_col].dropna()
        result["speed_max"] = float(v.max())
        dv = max_speed_increment(win[speed_col], DV_WINDOW_H, cadence_min)
        result["dv_max"] = dv if not np.isnan(dv) else np.nan
        result["cond_B_speed"] = (not np.isnan(dv)) and (dv >= DV_THRESH_KMS)

    # Condición C — IMF elevado sostenido
    if imf_col in win.columns and not win[imf_col].isna().all():
        imf = win[imf_col].dropna()
        result["imf_max"] = float(imf.max())
        consec = max_consecutive_above(win[imf_col], IMF_THRESH_NT, cadence_min)
        result["imf_consec_min"] = consec
        result["cond_C_imf"] = consec >= IMF_MIN_DUR_MIN

    # Resultado combinado — OR de las tres condiciones
    result["fd_like_confirmed"] = (
        result["cond_A_bz"] or
        result["cond_B_speed"] or
        result["cond_C_imf"]
    )
    return result


# ============================================================
# Main run
# ============================================================
if __name__ == "__main__":

    OUTPUT_DIR = Path(args.output_dir)
    (OUTPUT_DIR / "station_panels").mkdir(parents=True, exist_ok=True)

    # ── Paths ────────────────────────────────────────────────────────────────
    ALLDATA_PARQUET   = Path("alldata_integrated_2min.parquet")
    FEID_PARQUET      = Path("feid_clean.parquet")
    DATASET_META_JSON = Path("dataset_meta.json")

    # ── Cargar datos ──────────────────────────────────────────────────────────
    print("[1/7] Cargando dataset integrado (NMDB + OMNI)...", flush=True)
    df_nmdb, df_omni, NMDB_COLS, OMNI_COLS = load_nmdb_matrix(
        ALLDATA_PARQUET, DATASET_META_JSON)
    STATIONS = NMDB_COLS
    print(f"  NMDB: {df_nmdb.shape}  OMNI: {df_omni.shape}", flush=True)

    # ── CHECKPOINT: Paneles de estación ───────────────────────────────────────
    print("\n[2/7] Paneles de complejidad (con checkpoint)...", flush=True)
    station_panels: Dict[str, pd.DataFrame] = {}

    for st in STATIONS:
        if st not in df_nmdb.columns:
            continue
        panel_path = OUTPUT_DIR / "station_panels" / f"{st}_panel.csv"

        if panel_path.exists() and not args.force_rebuild:
            # Cargar desde disco
            print(f"  [checkpoint] {st} — cargando desde disco", flush=True)
            p = pd.read_csv(panel_path)
            tc = next((c for c in ["DATETIME","datetime","index","time"]
                        if c in p.columns), None)
            if tc:
                p[tc] = pd.to_datetime(p[tc], utc=True, errors="coerce")
                p[tc] = p[tc].dt.tz_convert(None)
                p = p.set_index(tc)
            station_panels[st] = p
        else:
            # Calcular y guardar
            print(f"  [building]    {st}", flush=True)
            panel = build_station_panel(df_nmdb, st)
            station_panels[st] = panel
            safe_write_csv(panel.reset_index(), panel_path)
            print(f"  [saved]       {st}", flush=True)

    print(f"  Total estaciones: {len(station_panels)}", flush=True)

    # ── Coverage report ───────────────────────────────────────────────────────
    coverage_rows = []
    for st, panel in station_panels.items():
        coverage_rows.append({
            "station": st,
            "n_rows":               len(panel),
            "counts_missing_pct":   100 * panel["counts"].isna().mean(),
            "deltaN_missing_pct":   100 * panel["deltaN"].isna().mean(),
            "A_complex_missing_pct":100 * panel["A_complex"].isna().mean(),
            "n_spikes_flagged":     int(panel["spike_flag"].sum()),
        })
    coverage_df = pd.DataFrame(coverage_rows).sort_values("station").reset_index(drop=True)
    display(coverage_df)

    # ── Triggers ──────────────────────────────────────────────────────────────
    print("\n[3/7] Detectando triggers y construyendo grupos...", flush=True)
    trigger_parts = []
    for st, panel in station_panels.items():
        ev = detect_station_events(panel, **DETECTOR_PARAMS)
        if not ev.empty:
            ev["detector"] = "detector"
            trigger_parts.append(ev)

    trigger_events = (
        pd.concat(trigger_parts, ignore_index=True)
        .sort_values(["onset_time","station"])
        .reset_index(drop=True)
        if trigger_parts else pd.DataFrame()
    )
    print(f"  Triggers crudos: {len(trigger_events)}", flush=True)

    trigger_catalog, trigger_members = build_trigger_groups(trigger_events)
    print(f"  Grupos:          {len(trigger_catalog)}", flush=True)

    # ── Support stage ─────────────────────────────────────────────────────────
    print("\n[4/7] Support-based promotion...", flush=True)
    after_catalog, support_members = promote_with_station_support(
        trigger_catalog, trigger_members, station_panels)
    print(f"  Catálogo final:  {len(after_catalog)}", flush=True)
    display(after_catalog["quality"].value_counts(dropna=False)
            .rename("count").to_frame())

    promotion_summary = pd.DataFrame({
        "metric": ["n_total","n_core","n_exploratory","n_promoted"],
        "value": [
            len(after_catalog),
            int((after_catalog["n_supported_stations"] >= 2).sum()),
            int((after_catalog["n_supported_stations"] < 2).sum()),
            int(after_catalog["promoted_from_single_trigger"].sum()),
        ],
    })
    display(promotion_summary)

    # ── FEID + OMNI enrichment ────────────────────────────────────────────────
    print("\n[5/7] Enriquecimiento FEID/OMNI...", flush=True)
    feid_catalog = load_feid_catalog(FEID_PARQUET)
    feid_2019    = feid_catalog[
        pd.to_datetime(feid_catalog["t_izmiran"]).dt.year == 2019].copy()
    print(f"  FEID total: {len(feid_catalog):,} | 2019: {len(feid_2019)}")
    # Nota: eventos AFTER de 2025 no tienen contraparte FEID disponible
    # (FEID cubre hasta 2024). feid_matched=False para esos eventos es
    # esperado y se declara como limitación del benchmark, no del detector.
    max_feid_year = pd.to_datetime(feid_catalog["t_izmiran"]).dt.year.max()
    print(f"  FEID cubre hasta: {max_feid_year} "
          f"(eventos post-{max_feid_year} son predicciones pendientes de validación)")

    # Normalizar timezone FEID
    if feid_catalog["t_izmiran"].dt.tz is not None:
        feid_catalog["t_izmiran"] = feid_catalog["t_izmiran"].dt.tz_convert(None)
    if feid_2019["t_izmiran"].dt.tz is not None:
        feid_2019["t_izmiran"] = feid_2019["t_izmiran"].dt.tz_convert(None)

    # OMNI 1h para contexto (más rápido que 2min para ventanas largas)
    alldata_1h_path = Path("alldata_integrated_1h.parquet")
    if alldata_1h_path.exists():
        df_omni_ctx = pd.read_parquet(alldata_1h_path)
        df_omni_ctx.index = pd.to_datetime(df_omni_ctx.index)
        if df_omni_ctx.index.tz is not None:
            df_omni_ctx.index = df_omni_ctx.index.tz_convert(None)
        omni_vars = ["IMF_avg","BX","BY","BZ","Speed","Density","Temperature"]
        df_omni_ctx = df_omni_ctx[[c for c in omni_vars if c in df_omni_ctx.columns]]
    else:
        df_omni_ctx = df_omni

    after_catalog_enriched = extract_omni_context(
        after_catalog, df_omni_ctx, feid_catalog,
        window_pre_h=48.0, window_post_h=48.0,
        feid_match_tol_h=DEFAULT_MATCH_TOLERANCE_H,
    )

    # Yearly physical context
    yearly_physical = (
        after_catalog_enriched
        .assign(year=pd.to_datetime(
            after_catalog_enriched["repr_time"]).dt.year)
        .groupby("year")
        .agg(
            n_events      =("repr_time","count"),
            n_feid_matched=("feid_matched","sum"),
            n_with_shock  =("feid_has_shock","sum"),
            mean_Bz_min   =("omni_Bz_min","mean"),
            mean_Speed_max=("omni_Speed_max","mean"),
        )
    )
    yearly_physical["pct_feid_matched"] = (
        yearly_physical["n_feid_matched"]/yearly_physical["n_events"]*100).round(1)
    yearly_physical["pct_with_shock"] = (
        yearly_physical["n_with_shock"]/yearly_physical["n_events"]*100).round(1)
    display(yearly_physical.round(2))

    # ── FD-like confirmation ──────────────────────────────────────────────────
    print("\n[6/7] Confirmación de huella FD-like (OMNI 2-min)...", flush=True)

    # Cargar OMNI 2-min para la ventana precisa de onset
    alldata_2m = pd.read_parquet(ALLDATA_PARQUET)
    alldata_2m.index = pd.to_datetime(alldata_2m.index)
    if alldata_2m.index.tz is not None:
        alldata_2m.index = alldata_2m.index.tz_convert(None)

    BZ_COL    = next((c for c in ["BZ","Bz","bz"] if c in alldata_2m.columns), None)
    SPEED_COL = next((c for c in ["Speed","speed","V"] if c in alldata_2m.columns), None)
    IMF_COL   = next((c for c in ["IMF_avg","IMF","imf"] if c in alldata_2m.columns), None)
    print(f"  Columnas OMNI: Bz={BZ_COL}  Speed={SPEED_COL}  IMF={IMF_COL}")

    # Onset time para la ventana de confirmación
    onset_col = "anchor_onset_time" if "anchor_onset_time" in after_catalog_enriched.columns                 else "repr_time"
    after_catalog_enriched[onset_col] = pd.to_datetime(
        after_catalog_enriched[onset_col])
    if after_catalog_enriched[onset_col].dt.tz is not None:
        after_catalog_enriched[onset_col] =             after_catalog_enriched[onset_col].dt.tz_convert(None)

    conf_rows = []
    n_total   = len(after_catalog_enriched)
    for idx, (_, ev) in enumerate(after_catalog_enriched.iterrows()):
        if idx % 100 == 0:
            print(f"  Evento {idx+1}/{n_total}...", end="\r", flush=True)
        t_onset = pd.Timestamp(ev[onset_col])
        t_pre   = t_onset - pd.Timedelta(hours=12.0)
        t_post  = t_onset + pd.Timedelta(hours=6.0)
        win     = alldata_2m.loc[t_pre:t_post]
        conf_rows.append(check_fd_like(
            win,
            bz_col    = BZ_COL    or "BZ",
            speed_col = SPEED_COL or "Speed",
            imf_col   = IMF_COL   or "IMF_avg",
            cadence_min = 2.0,
        ))

    conf_df = pd.DataFrame(conf_rows, index=after_catalog_enriched.index)
    after_catalog_enriched = pd.concat([after_catalog_enriched, conf_df], axis=1)
    print(f"\n  fd_like_confirmed=True: "
          f"{after_catalog_enriched['fd_like_confirmed'].sum()} "
          f"({after_catalog_enriched['fd_like_confirmed'].mean()*100:.1f}%)")

    # Tabla tres niveles de confianza
    core_e = after_catalog_enriched[
        after_catalog_enriched["n_supported_stations"] >= 2]
    expl_e = after_catalog_enriched[
        after_catalog_enriched["n_supported_stations"] < 2]
    n_top = int(core_e["fd_like_confirmed"].sum())
    n_mid = int((~core_e["fd_like_confirmed"]).sum())
    n_low = len(expl_e)
    print(f"\n  Tres niveles de confianza:")
    print(f"  Core + confirmed:   {n_top:>4} ({n_top/n_total*100:.1f}%)")
    print(f"  Core + unconfirmed: {n_mid:>4} ({n_mid/n_total*100:.1f}%)")
    print(f"  Exploratory:        {n_low:>4} ({n_low/n_total*100:.1f}%)")

    # FD-like summary por año
    fd_yr_summary = after_catalog_enriched.groupby(
        pd.to_datetime(after_catalog_enriched["repr_time"]).dt.year
    ).agg(
        N=("fd_like_confirmed","count"),
        N_confirmed=("fd_like_confirmed","sum"),
        pct_A=("cond_A_bz","mean"),
        pct_B=("cond_B_speed","mean"),
        pct_C=("cond_C_imf","mean"),
    ).round(3)
    fd_yr_summary["pct_confirmed"] = (
        fd_yr_summary["N_confirmed"]/fd_yr_summary["N"]*100).round(1)
    display(fd_yr_summary)

    # ── Validación FEID 2019 ──────────────────────────────────────────────────
    print("\n[7/7] Validación FEID 2019 y guardado...", flush=True)
    validation_summary, validation_matches = evaluate_catalog_vs_izmiran(
        feid_2019, after_catalog_enriched,
        tolerances_h=MATCH_TOLERANCES_H, year=2019,
    )
    display(validation_summary)

    cat_2019 = after_catalog_enriched[
        pd.to_datetime(after_catalog_enriched["repr_time"]).dt.year == 2019].copy()
    DEFAULT_MATCHES = {}
    for subset_name in ["all","core","exploratory"]:
        sub = cat_2019 if subset_name == "all"               else cat_2019[cat_2019["n_supported_stations"] >= 2].copy()                    if subset_name == "core"                    else cat_2019[cat_2019["n_supported_stations"] < 2].copy()
        match_df = one_to_one_time_match(
            list(feid_2019["t_izmiran"]),
            list(pd.to_datetime(sub["repr_time"])) if not sub.empty else [],
            tolerance_hours=DEFAULT_MATCH_TOLERANCE_H,
        )
        DEFAULT_MATCHES[subset_name] = enrich_match_table(
            match_df, feid_2019, sub, subset_name)

    # ── Guardar todos los outputs ─────────────────────────────────────────────
    safe_write_csv(trigger_events,   OUTPUT_DIR / "trigger_events_detector.csv")
    safe_write_csv(trigger_catalog,  OUTPUT_DIR / "trigger_catalog_detector.csv")
    safe_write_csv(trigger_members,  OUTPUT_DIR / "trigger_members_detector.csv")

    safe_write_csv(after_catalog_enriched,
                   OUTPUT_DIR / "AFTER_catalog_detector.csv")
    safe_write_csv(after_catalog_enriched[
                       after_catalog_enriched["n_supported_stations"] >= 2],
                   OUTPUT_DIR / "AFTER_catalog_detector_core.csv")
    safe_write_csv(after_catalog_enriched[
                       after_catalog_enriched["n_supported_stations"] < 2],
                   OUTPUT_DIR / "AFTER_catalog_detector_exploratory.csv")

    safe_write_csv(support_members,    OUTPUT_DIR / "support_members_detector.csv")
    safe_write_csv(validation_summary, OUTPUT_DIR / "validation_summary_detector.csv")
    safe_write_csv(promotion_summary,  OUTPUT_DIR / "promotion_summary_detector.csv")
    safe_write_csv(coverage_df,        OUTPUT_DIR / "station_coverage_report.csv")
    safe_write_csv(feid_catalog,       OUTPUT_DIR / "feid_catalog_parsed.csv")
    safe_write_csv(yearly_physical.reset_index(),
                   OUTPUT_DIR / "yearly_physical_context.csv")
    safe_write_csv(fd_yr_summary.reset_index(),
                   OUTPUT_DIR / "fd_like_confirmation_summary.csv")

    for subset_name, table in DEFAULT_MATCHES.items():
        safe_write_csv(table, OUTPUT_DIR /
                       f"match_table_detector_{subset_name}_{DEFAULT_MATCH_TOLERANCE_H}h.csv")

    print(f"\n[INFO] Outputs en: {OUTPUT_DIR.resolve()}", flush=True)
    # Resumen por año — incluyendo 2025 sin FEID
    cat_yr = after_catalog_enriched.copy()
    cat_yr["year"] = pd.to_datetime(cat_yr["repr_time"]).dt.year
    yr_counts = cat_yr.groupby("year").agg(
        N=("repr_time","count"),
        N_feid=("feid_matched","sum"),
        N_fdlike=("fd_like_confirmed","sum"),
    )
    yr_counts["note"] = yr_counts.index.map(
        lambda y: "no FEID" if y > max_feid_year else "")
    print(f"\n[INFO] Conteos anuales:")
    print(yr_counts.to_string())

    print(f"\n[INFO] Resumen global:")
    print(f"  Total:             {len(after_catalog_enriched):>6}")
    print(f"  Core:              {(after_catalog_enriched['n_supported_stations']>=2).sum():>6}")
    print(f"  Exploratory:       {(after_catalog_enriched['n_supported_stations']<2).sum():>6}")
    print(f"  FEID matched:      {after_catalog_enriched['feid_matched'].sum():>6}  "
          f"(solo hasta {max_feid_year})")
    print(f"  FD-like confirmed: {after_catalog_enriched['fd_like_confirmed'].sum():>6}")
    print(f"  Core + confirmed:  {n_top:>6}")