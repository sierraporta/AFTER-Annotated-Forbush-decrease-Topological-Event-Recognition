from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)


# -----------------------------
# Paths
# -----------------------------
NMDB_CSV_PATH = "../Results/DataStudy.csv"      # main NMDB matrix
IZMIRAN_TXT_PATH = "../izmiran2019.txt"
OUTPUT_DIR = Path("Results_v2_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Stations
# -----------------------------
STATIONS = ["MXCO", "JUNG1", "LMKS", "NEWK", "KERG", "OULU", "DOMC", "INVK", "APTY", "AATB"]

# -----------------------------
# Processing parameters
# -----------------------------
TIME_COL = "DATETIME"
EXPECTED_CADENCE_MIN = 2

SPIKE_MAD_FACTOR = 6.0
BACKGROUND_HOURS = 72                # causal rolling median
BACKGROUND_MIN_FRAC = 0.70

COMPLEXITY_WINDOW_HOURS = 3
COMPLEXITY_MIN_FRAC = 0.70
PE_ORDER = 3
PE_DELAY = 1

ROBUST_REF_HOURS = 24 * 30           # 30-day trailing robust reference
ROBUST_REF_MIN_FRAC = 0.50
ROBUST_Z_EPS = 1e-9

# -----------------------------
# Detection levels
# -----------------------------
DETECTION_LEVELS = {
    "strict": {
        "a_thresh": 3.0,
        "min_a_duration_hours": 3.0,
        "min_drop_percent": 2.0,
        "confirm_negative_percent": 0.75,
        "confirm_negative_duration_hours": 1.0,
        "max_time_to_min_hours": 36.0,
        "pre_event_hours": 24.0,
        "recovery_fraction": 0.50,
        "max_recovery_days": 7.0,
    },
    "medium": {
        "a_thresh": 2.5,
        "min_a_duration_hours": 3.0,
        "min_drop_percent": 1.5,
        "confirm_negative_percent": 0.50,
        "confirm_negative_duration_hours": 1.0,
        "max_time_to_min_hours": 36.0,
        "pre_event_hours": 24.0,
        "recovery_fraction": 0.50,
        "max_recovery_days": 7.0,
    },
    "loose": {
        "a_thresh": 2.0,
        "min_a_duration_hours": 2.0,
        "min_drop_percent": 1.0,
        "confirm_negative_percent": 0.30,
        "confirm_negative_duration_hours": 0.5,
        "max_time_to_min_hours": 36.0,
        "pre_event_hours": 24.0,
        "recovery_fraction": 0.50,
        "max_recovery_days": 7.0,
    },
}

# IMPORTANT CHANGE IN v2.1:
# group multi-station events by onset/segment timing, not by min_time.
COINCIDENCE_WINDOW_HOURS = 18
MIN_STATIONS_CORE = 2

MATCH_TOLERANCES_H = [6, 12, 18, 24]
DEFAULT_MATCH_TOLERANCE_H = 18


# ============================================================
# I/O and utility helpers
# ============================================================
def load_nmdb_matrix(
    csv_path: str | os.PathLike,
    time_col: str = "DATETIME",
    station_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in {csv_path}")

    dt = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.loc[dt.notna()].copy()
    df[time_col] = dt[dt.notna()]
    df = df.sort_values(time_col).drop_duplicates(subset=[time_col]).set_index(time_col)

    if station_cols is None:
        station_cols = [c for c in df.columns]

    for c in station_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = [c for c in station_cols if c in df.columns]
    if not keep:
        raise ValueError("No requested station columns were found in the NMDB file.")

    return df[keep].copy()


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
        lambda x: permutation_entropy(x, m=m, delay=delay),
        raw=True,
    )
    kfd = series.rolling(window_pts, min_periods=min_periods).apply(
        katz_fd,
        raw=True,
    )
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
# Station-level processing and event detection
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
    merge_gap_hours: float = 12.0,
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


def detect_station_events(
    panel: pd.DataFrame,
    a_thresh: float,
    min_a_duration_hours: float,
    min_drop_percent: float,
    confirm_negative_percent: float,
    confirm_negative_duration_hours: float,
    max_time_to_min_hours: float,
    pre_event_hours: float,
    recovery_fraction: float,
    max_recovery_days: float,
) -> pd.DataFrame:
    df = panel.copy()
    required = {"deltaN", "A_complex", "station"}
    if not required.issubset(df.columns):
        raise ValueError(f"Panel is missing required columns: {required - set(df.columns)}")

    cadence_min = infer_cadence_minutes(df.index)
    pts_per_hour = max(int(round(60.0 / cadence_min)), 1)

    min_a_pts = max(int(round(min_a_duration_hours * pts_per_hour)), 1)
    pre_pts = max(int(round(pre_event_hours * pts_per_hour)), 1)
    neg_confirm_pts = max(int(round(confirm_negative_duration_hours * pts_per_hour)), 1)
    max_to_min_pts = max(int(round(max_time_to_min_hours * pts_per_hour)), 1)
    max_rec_pts = max(int(round(max_recovery_days * 24.0 * pts_per_hour)), 1)

    high_a = df["A_complex"] >= a_thresh
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

        neg_thr = pre_level - confirm_negative_percent
        if longest_true_run(search_slice <= neg_thr) < neg_confirm_pts:
            continue

        min_time = search_slice.idxmin()
        min_val = float(search_slice.loc[min_time])
        drop = float(pre_level - min_val)
        if not np.isfinite(drop) or drop < min_drop_percent:
            continue

        j_min = df.index.get_loc(min_time)
        rec_slice = df["deltaN"].iloc[j_min:min(len(df), j_min + max_rec_pts + 1)]
        target = pre_level - (1.0 - recovery_fraction) * drop
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

    out = merge_nearby_station_events(out, time_col="onset_time", merge_gap_hours=12.0)
    return out.sort_values("onset_time").reset_index(drop=True)


# ============================================================
# Multi-station coincidence logic (v2.1 main change)
# ============================================================
def collapse_same_station(group_df: pd.DataFrame) -> pd.DataFrame:
    if group_df.empty:
        return group_df.copy()
    group_df = group_df.sort_values(
        ["station", "drop_percent", "A_complex_max_segment"],
        ascending=[True, False, False],
    )
    return group_df.drop_duplicates(subset=["station"], keep="first").copy()


def same_physical_candidate(
    row: pd.Series,
    group_df: pd.DataFrame,
    onset_window_h: float = 18.0,
    overlap_tolerance_h: float = 12.0,
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


def build_coincidence_catalog_v2(
    events_df: pd.DataFrame,
    onset_window_hours: float = 18.0,
    overlap_tolerance_hours: float = 12.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                row = ev.loc[j]
                if same_physical_candidate(
                    row,
                    current_group,
                    onset_window_h=onset_window_hours,
                    overlap_tolerance_h=overlap_tolerance_hours,
                ):
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
        repr_time = min_times.median() if min_times.notna().any() else onset_times.median()

        row = {
            "group_id": gid,
            "repr_time": repr_time,
            "anchor_onset_time": onset_times.median(),
            "time_start": onset_times.min(),
            "time_end": min_times.max(),
            "group_span_h": (min_times.max() - onset_times.min()).total_seconds() / 3600.0,
            "n_station_events": len(g),
            "n_stations": g["station"].nunique(),
            "stations": ",".join(sorted(g["station"].unique())),
            "drop_mean": g["drop_percent"].mean(),
            "drop_median": g["drop_percent"].median(),
            "drop_max": g["drop_percent"].max(),
            "A_complex_mean": g["A_complex_max_segment"].mean(),
            "A_complex_max": g["A_complex_max_segment"].max(),
            "onset_min": onset_times.min(),
            "onset_max": onset_times.max(),
            "min_time_min": min_times.min(),
            "min_time_max": min_times.max(),
            "median_onset_to_min_h": g["onset_to_min_h"].median(),
        }
        rows.append(row)

        gm = g.copy()
        gm["group_id"] = gid
        gm["repr_time"] = repr_time
        gm["anchor_onset_time"] = onset_times.median()
        members.append(gm)

    catalog = pd.DataFrame(rows).sort_values("repr_time").reset_index(drop=True)
    group_members = pd.concat(members, ignore_index=True) if members else pd.DataFrame()
    return catalog, group_members


def assign_quality_label(n_stations: int, drop_mean: float) -> str:
    if n_stations >= 3:
        return "A"
    if n_stations == 2:
        return "B"
    if n_stations == 1 and drop_mean >= 2.0:
        return "C"
    return "D"


# ============================================================
# Validation helpers
# ============================================================
def load_izmiran_txt(txt_path: str | os.PathLike) -> pd.DataFrame:
    df = pd.read_csv(
        txt_path,
        sep="\t",
        na_values=["None", "none", "-999.0", "-99.9", "999.0", "99.99"],
    )
    if "Date of event" not in df.columns:
        raise ValueError("IZMIRAN file must contain 'Date of event'.")

    df["t_izmiran"] = pd.to_datetime(df["Date of event"], utc=True, errors="coerce")
    df = df.loc[df["t_izmiran"].notna()].copy()

    numeric_cols = [c for c in df.columns if c not in ["Date of event", "t_izmiran"]]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("t_izmiran").reset_index(drop=True)


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

    cat = catalog_df.copy()
    cat = cat[pd.to_datetime(cat["repr_time"]).dt.year == year].copy()
    if cat.empty:
        return pd.DataFrame(), {}

    subsets = {
        "all": cat,
        "core": cat[cat["catalog_tier"] == "core"].copy(),
        "exploratory": cat[cat["catalog_tier"] == "exploratory"].copy(),
    }

    summary_rows = []
    matched_tables = {}

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

            if not match_df.empty:
                dt_abs = match_df["dt_hours"].abs()
                median_abs_dt = dt_abs.median()
                mean_abs_dt = dt_abs.mean()
            else:
                median_abs_dt = np.nan
                mean_abs_dt = np.nan

            summary_rows.append(
                {
                    "subset": subset_name,
                    "tolerance_h": tol_h,
                    "n_izmiran": n_ref,
                    "n_catalog": n_cand,
                    "n_matched": n_match,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "median_abs_dt_h": median_abs_dt,
                    "mean_abs_dt_h": mean_abs_dt,
                }
            )
            matched_tables[(subset_name, tol_h)] = match_df

    summary = pd.DataFrame(summary_rows).sort_values(["subset", "tolerance_h"]).reset_index(drop=True)
    return summary, matched_tables


def enrich_match_table(
    match_df: pd.DataFrame,
    izm_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    subset_name: str,
) -> pd.DataFrame:
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
# Main run
# ============================================================
print("[INFO] Loading NMDB matrix...", flush=True)
df_nmdb = load_nmdb_matrix(NMDB_CSV_PATH, time_col=TIME_COL, station_cols=STATIONS)
display(regularity_report(df_nmdb.index))
display(df_nmdb.head())
print("[INFO] Matrix shape:", df_nmdb.shape, flush=True)

print("[INFO] Building station panels...", flush=True)
station_panels: Dict[str, pd.DataFrame] = {}
for st in STATIONS:
    if st in df_nmdb.columns:
        print(f"[INFO] Processing station {st} ...", flush=True)
        station_panels[st] = build_station_panel(df_nmdb, st)
        print(f"[INFO] Finished station {st}", flush=True)

coverage_rows = []
for st, panel in station_panels.items():
    coverage_rows.append(
        {
            "station": st,
            "n_rows": len(panel),
            "counts_missing_pct": 100 * panel["counts"].isna().mean(),
            "bg_missing_pct": 100 * panel["bg"].isna().mean(),
            "deltaN_missing_pct": 100 * panel["deltaN"].isna().mean(),
            "A_complex_missing_pct": 100 * panel["A_complex"].isna().mean(),
            "n_spikes_flagged": int(panel["spike_flag"].sum()),
        }
    )
coverage_df = pd.DataFrame(coverage_rows).sort_values("station").reset_index(drop=True)
display(coverage_df)

print("[INFO] Detecting station events...", flush=True)
events_by_level: Dict[str, pd.DataFrame] = {}
for level_name, pars in DETECTION_LEVELS.items():
    parts = []
    for st, panel in station_panels.items():
        ev = detect_station_events(panel, **pars)
        if not ev.empty:
            ev["level"] = level_name
            parts.append(ev)
    events_by_level[level_name] = (
        pd.concat(parts, ignore_index=True).sort_values(["onset_time", "station"]).reset_index(drop=True)
        if parts else pd.DataFrame()
    )
    print(f"[INFO] Level {level_name}: {len(events_by_level[level_name])} station events", flush=True)

print("[INFO] Building coincidence catalogs v2.1...", flush=True)
coincidence_by_level: Dict[str, pd.DataFrame] = {}
group_members_by_level: Dict[str, pd.DataFrame] = {}
catalogs_by_level: Dict[str, pd.DataFrame] = {}

for level_name, ev in events_by_level.items():
    coinc, members = build_coincidence_catalog_v2(
        ev,
        onset_window_hours=COINCIDENCE_WINDOW_HOURS,
        overlap_tolerance_hours=12.0,
    )
    coincidence_by_level[level_name] = coinc
    group_members_by_level[level_name] = members

    if coinc.empty:
        catalogs_by_level[level_name] = coinc.copy()
        continue

    cat = coinc.copy()
    cat["level"] = level_name
    cat["year"] = pd.to_datetime(cat["repr_time"]).dt.year
    cat["quality"] = [assign_quality_label(n, d) for n, d in zip(cat["n_stations"], cat["drop_mean"])]
    cat["catalog_tier"] = np.where(cat["n_stations"] >= MIN_STATIONS_CORE, "core", "exploratory")
    cat["fd_scope"] = np.where(cat["n_stations"] >= MIN_STATIONS_CORE, "multi_station_candidate", "single_station_candidate")
    catalogs_by_level[level_name] = cat

    print(f"[INFO] Level {level_name}: {len(cat)} coincidence groups", flush=True)
    display(cat["catalog_tier"].value_counts(dropna=False).rename("count").to_frame())
    display(cat["quality"].value_counts(dropna=False).rename("count").to_frame())

print("[INFO] Loading IZMIRAN...", flush=True)
izmiran_2019 = load_izmiran_txt(IZMIRAN_TXT_PATH)
display(izmiran_2019.head())
print(f"[INFO] Loaded {len(izmiran_2019)} IZMIRAN 2019 events.", flush=True)

print("[INFO] Running validation...", flush=True)
validation_summary_by_level = {}
validation_matches_by_level = {}
DEFAULT_MATCHES = {}

for level_name, cat in catalogs_by_level.items():
    summary, matches = evaluate_catalog_vs_izmiran(
        izmiran_2019,
        cat,
        tolerances_h=MATCH_TOLERANCES_H,
        year=2019,
    )
    validation_summary_by_level[level_name] = summary
    validation_matches_by_level[level_name] = matches

    print(f"\n[INFO] Validation summary for level = {level_name}", flush=True)
    display(summary)

    cat_2019 = cat[pd.to_datetime(cat["repr_time"]).dt.year == 2019].copy() if not cat.empty else pd.DataFrame()
    for subset_name in ["all", "core", "exploratory"]:
        if cat_2019.empty:
            DEFAULT_MATCHES[(level_name, subset_name)] = pd.DataFrame()
            continue
        sub = cat_2019 if subset_name == "all" else cat_2019[cat_2019["catalog_tier"] == subset_name].copy()
        match_df = one_to_one_time_match(
            list(izmiran_2019["t_izmiran"]),
            list(pd.to_datetime(sub["repr_time"])) if not sub.empty else [],
            tolerance_hours=DEFAULT_MATCH_TOLERANCE_H,
        )
        DEFAULT_MATCHES[(level_name, subset_name)] = enrich_match_table(match_df, izmiran_2019, sub, subset_name)

print("[INFO] Writing outputs...", flush=True)
for st, panel in station_panels.items():
    safe_write_csv(panel.reset_index(), OUTPUT_DIR / "station_panels" / f"{st}_panel.csv")

for level_name, ev in events_by_level.items():
    safe_write_csv(ev, OUTPUT_DIR / f"single_station_events_{level_name}.csv")

for level_name, coinc in coincidence_by_level.items():
    safe_write_csv(coinc, OUTPUT_DIR / f"coincidence_catalog_{level_name}.csv")

for level_name, members in group_members_by_level.items():
    safe_write_csv(members, OUTPUT_DIR / f"coincidence_members_{level_name}.csv")

for level_name, cat in catalogs_by_level.items():
    safe_write_csv(cat, OUTPUT_DIR / f"AFTER_catalog_{level_name}.csv")
    safe_write_csv(cat[cat["catalog_tier"] == "core"], OUTPUT_DIR / f"AFTER_catalog_{level_name}_core.csv")
    safe_write_csv(cat[cat["catalog_tier"] == "exploratory"], OUTPUT_DIR / f"AFTER_catalog_{level_name}_exploratory.csv")

for level_name, summary in validation_summary_by_level.items():
    safe_write_csv(summary, OUTPUT_DIR / f"validation_summary_{level_name}.csv")

for (level_name, subset_name), match_df in DEFAULT_MATCHES.items():
    safe_write_csv(match_df, OUTPUT_DIR / f"match_table_{level_name}_{subset_name}_{DEFAULT_MATCH_TOLERANCE_H}h.csv")

safe_write_csv(coverage_df, OUTPUT_DIR / "station_coverage_report.csv")
safe_write_csv(izmiran_2019, OUTPUT_DIR / "izmiran_2019_parsed.csv")

print(f"[INFO] All outputs written under: {OUTPUT_DIR.resolve()}", flush=True)
