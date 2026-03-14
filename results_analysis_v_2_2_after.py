from __future__ import annotations

# ============================================================
# Results analysis for AFTER v2.2
# Final methodological basis:
#   - detector-based trigger generation
#   - support-based multi-station promotion
#   - main core: n_supported_stations >= 2
#   - conservative core: n_supported_stations >= 2 & n_support_only_stations >= 1
# ============================================================

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)


# ============================================================
# Paths
# ============================================================
BASE = Path("Results_v2_2")
OUT = BASE / "ResultsAnalysis_v2_2"
OUT.mkdir(parents=True, exist_ok=True)

NMDB_CSV_PATH = Path("Results/DataStudy.csv")

CAT_FILE = BASE / "AFTER_catalog_detector.csv"
SUPPORT_FILE = BASE / "support_members_detector.csv"
TRIGGER_FILE = BASE / "trigger_catalog_detector.csv"
VALIDATION_FILE = BASE / "validation_summary_detector.csv"
IZM_FILE = BASE / "izmiran_2019_parsed.csv"

SAVE_FIGS = True
DPI = 220


# ============================================================
# Utility functions
# ============================================================
def savefig(name: str):
    if SAVE_FIGS:
        plt.savefig(OUT / name, dpi=DPI, bbox_inches="tight")


def one_to_one_time_match(ref_times, cand_times, tolerance_hours=18.0):
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
        kept.append({
            "ref_pos": i,
            "cand_pos": j,
            "dt_hours": dt.total_seconds() / 3600.0,
        })
    return pd.DataFrame(kept)


def evaluate_subset(subset_df: pd.DataFrame, izm_df: pd.DataFrame, tolerances=(12, 18, 24)) -> pd.DataFrame:
    ref_times = list(izm_df["t_izmiran"])
    cand_times = list(subset_df["repr_time"])
    rows = []
    for tol in tolerances:
        m = one_to_one_time_match(ref_times, cand_times, tolerance_hours=tol)
        n_ref = len(ref_times)
        n_cand = len(cand_times)
        n_match = len(m)
        recall = n_match / n_ref if n_ref else np.nan
        precision = n_match / n_cand if n_cand else np.nan
        f1 = (2 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else np.nan
        rows.append(
            {
                "tolerance_h": tol,
                "n_catalog": n_cand,
                "n_matched": n_match,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "median_abs_dt_h": m["dt_hours"].abs().median() if not m.empty else np.nan,
                "mean_abs_dt_h": m["dt_hours"].abs().mean() if not m.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_case_table(subset_df: pd.DataFrame, izm_df: pd.DataFrame, tolerance_hours=18.0) -> pd.DataFrame:
    match_df = one_to_one_time_match(
        list(izm_df["t_izmiran"]),
        list(subset_df["repr_time"]),
        tolerance_hours=tolerance_hours,
    )
    if match_df.empty:
        return pd.DataFrame()

    ref_tbl = izm_df.reset_index(drop=True).reset_index().rename(columns={"index": "idx_izmiran"})
    cand_tbl = subset_df.reset_index(drop=True).reset_index().rename(columns={"index": "idx_catalog"})
    out = match_df.rename(columns={"ref_pos": "idx_izmiran", "cand_pos": "idx_catalog"}).copy()
    out = out.merge(ref_tbl, on="idx_izmiran", how="left")
    out = out.merge(cand_tbl, on="idx_catalog", how="left", suffixes=("_izmiran", "_catalog"))
    return out.sort_values("dt_hours", key=lambda s: s.abs()).reset_index(drop=True)


def read_nmdb_subset(start: pd.Timestamp, end: pd.Timestamp, stations: list[str]) -> pd.DataFrame:
    usecols = ["DATETIME"] + stations
    df = pd.read_csv(NMDB_CSV_PATH, usecols=usecols)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], utc=True, errors="coerce")
    df = df.dropna(subset=["DATETIME"]).sort_values("DATETIME")
    df = df[(df["DATETIME"] >= start) & (df["DATETIME"] <= end)].copy()
    for c in stations:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def robust_relative_deviation(df: pd.DataFrame, stations: list[str], baseline_hours=72.0) -> pd.DataFrame:
    out = df.copy()
    if len(out) < 2:
        return out
    cadence_min = out["DATETIME"].diff().dt.total_seconds().dropna().median() / 60.0
    win = max(int(round(baseline_hours * 60.0 / cadence_min)), 1)
    for st in stations:
        med = out[st].rolling(win, min_periods=max(3, int(0.7 * win))).median().shift(1)
        out[f"deltaN_{st}"] = 100.0 * (out[st] - med) / med
    return out


# ============================================================
# Load data
# ============================================================
cat = pd.read_csv(CAT_FILE)
support = pd.read_csv(SUPPORT_FILE)
trigger = pd.read_csv(TRIGGER_FILE)
val0 = pd.read_csv(VALIDATION_FILE)
izm = pd.read_csv(IZM_FILE)

for c in ["repr_time", "anchor_onset_time", "time_start", "time_end", "onset_min", "onset_max", "min_time_min", "min_time_max"]:
    if c in cat.columns:
        cat[c] = pd.to_datetime(cat[c], utc=True, errors="coerce")
for c in ["repr_time", "anchor_onset_time", "support_min_time", "support_onset_time"]:
    if c in support.columns:
        support[c] = pd.to_datetime(support[c], utc=True, errors="coerce")
if "repr_time" in trigger.columns:
    trigger["repr_time"] = pd.to_datetime(trigger["repr_time"], utc=True, errors="coerce")
if "t_izmiran" in izm.columns:
    izm["t_izmiran"] = pd.to_datetime(izm["t_izmiran"], utc=True, errors="coerce")

cat2019 = cat.loc[cat["year"] == 2019].copy()
izm2019 = izm.loc[izm["t_izmiran"].dt.year == 2019].copy()

# Final subsets
core_main = cat.loc[cat["n_supported_stations"] >= 2].copy()
core_conservative = cat.loc[(cat["n_supported_stations"] >= 2) & (cat["n_support_only_stations"] >= 1)].copy()
exploratory_final = cat.loc[cat["n_supported_stations"] < 2].copy()

core_main_2019 = core_main.loc[core_main["year"] == 2019].copy()
core_cons_2019 = core_conservative.loc[core_conservative["year"] == 2019].copy()
expl_2019 = exploratory_final.loc[exploratory_final["year"] == 2019].copy()

print("Loaded catalog shape:", cat.shape)
print("Core main:", core_main.shape)
print("Core conservative:", core_conservative.shape)
print("Exploratory:", exploratory_final.shape)


# ============================================================
# 1) Global descriptive tables
# ============================================================
summary_main = pd.DataFrame(
    {
        "subset": ["all", "core_main", "core_conservative", "exploratory"],
        "n_total": [len(cat), len(core_main), len(core_conservative), len(exploratory_final)],
        "n_2019": [len(cat2019), len(core_main_2019), len(core_cons_2019), len(expl_2019)],
        "mean_supported_stations": [cat["n_supported_stations"].mean(), core_main["n_supported_stations"].mean(), core_conservative["n_supported_stations"].mean(), exploratory_final["n_supported_stations"].mean()],
        "median_supported_stations": [cat["n_supported_stations"].median(), core_main["n_supported_stations"].median(), core_conservative["n_supported_stations"].median(), exploratory_final["n_supported_stations"].median()],
        "mean_supported_drop": [cat["mean_supported_drop"].mean(), core_main["mean_supported_drop"].mean(), core_conservative["mean_supported_drop"].mean(), exploratory_final["mean_supported_drop"].mean()],
        "n_promoted_from_single_trigger": [cat["promoted_from_single_trigger"].sum(), core_main["promoted_from_single_trigger"].sum(), core_conservative["promoted_from_single_trigger"].sum(), exploratory_final["promoted_from_single_trigger"].sum()],
    }
)

display(summary_main)
summary_main.to_csv(OUT / "table_catalog_summary.csv", index=False)

quality_table = pd.concat(
    {
        "all": cat["quality"].value_counts(dropna=False),
        "core_main": core_main["quality"].value_counts(dropna=False),
        "core_conservative": core_conservative["quality"].value_counts(dropna=False),
        "exploratory": exploratory_final["quality"].value_counts(dropna=False),
    },
    axis=1,
).fillna(0).astype(int)

display(quality_table)
quality_table.to_csv(OUT / "table_quality_counts.csv")

counts_by_year = pd.concat(
    {
        "all": cat["year"].value_counts().sort_index(),
        "core_main": core_main["year"].value_counts().sort_index(),
        "core_conservative": core_conservative["year"].value_counts().sort_index(),
        "exploratory": exploratory_final["year"].value_counts().sort_index(),
    },
    axis=1,
).fillna(0).astype(int)

display(counts_by_year)
counts_by_year.to_csv(OUT / "table_counts_by_year.csv")


# ============================================================
# 2) Validation tables
# ============================================================
val_all = evaluate_subset(cat2019, izm2019)
val_all["subset"] = "all"
val_core_main = evaluate_subset(core_main_2019, izm2019)
val_core_main["subset"] = "core_main"
val_core_cons = evaluate_subset(core_cons_2019, izm2019)
val_core_cons["subset"] = "core_conservative"
val_expl = evaluate_subset(expl_2019, izm2019)
val_expl["subset"] = "exploratory"

validation_table = pd.concat([val_all, val_core_main, val_core_cons, val_expl], ignore_index=True)
validation_table = validation_table[["subset", "tolerance_h", "n_catalog", "n_matched", "recall", "precision", "f1", "median_abs_dt_h", "mean_abs_dt_h"]]

display(validation_table)
validation_table.to_csv(OUT / "table_validation_summary.csv", index=False)


# ============================================================
# 3) Case-study candidate tables
# ============================================================
match_all_18 = build_case_table(cat2019, izm2019, tolerance_hours=18)
match_core_main_18 = build_case_table(core_main_2019, izm2019, tolerance_hours=18)
match_core_cons_18 = build_case_table(core_cons_2019, izm2019, tolerance_hours=18)

# Best-aligned cases
best_aligned = match_core_main_18.sort_values("dt_hours", key=lambda s: s.abs()).copy()
keep_cols = [c for c in [
    "Date of event", "t_izmiran", "repr_time", "dt_hours",
    "n_supported_stations", "n_support_only_stations",
    "supported_stations", "support_only_stations",
    "quality", "mean_supported_drop", "promoted_from_single_trigger"
] if c in best_aligned.columns]

best_aligned = best_aligned[keep_cols].head(20)
display(best_aligned)
best_aligned.to_csv(OUT / "table_best_aligned_matches_core_main_18h.csv", index=False)

# Promoted successful cases
promoted_good = match_core_main_18.loc[match_core_main_18["promoted_from_single_trigger"] == True].copy()
promoted_good = promoted_good.sort_values(["dt_hours"], key=lambda s: s.abs())
promoted_good = promoted_good[keep_cols].head(20)
display(promoted_good)
promoted_good.to_csv(OUT / "table_promoted_matches_core_main_18h.csv", index=False)

# Conservative matched cases
cons_cases = match_core_cons_18.sort_values("dt_hours", key=lambda s: s.abs())[keep_cols].head(20)
display(cons_cases)
cons_cases.to_csv(OUT / "table_matches_core_conservative_18h.csv", index=False)


# ============================================================
# 4) Support-members summary
# ============================================================
support_summary = support.groupby("group_id").agg(
    n_support_flag=("support_flag", "sum"),
    n_trigger_station=("is_trigger_station", "sum"),
    mean_support_drop=("support_drop_percent", "mean"),
    median_support_shift_h=("support_onset_shift_h", "median"),
).reset_index()

display(support_summary.head())
support_summary.to_csv(OUT / "table_support_summary_by_group.csv", index=False)


# ============================================================
# 5) Figures
# ============================================================
# Fig 1: counts by year
plt.figure(figsize=(9, 5))
plt.plot(counts_by_year.index, counts_by_year["all"], marker="o", label="All")
plt.plot(counts_by_year.index, counts_by_year["core_main"], marker="o", label="Core main")
plt.plot(counts_by_year.index, counts_by_year["core_conservative"], marker="o", label="Core conservative")
plt.plot(counts_by_year.index, counts_by_year["exploratory"], marker="o", label="Exploratory")
plt.xlabel("Year")
plt.ylabel("Number of events")
plt.title("AFTER v2.2 catalog composition by year")
plt.legend()
plt.grid(True, alpha=0.3)
savefig("fig_counts_by_year.png")
plt.show()

# Fig 2: quality composition
quality_pct = quality_table.div(quality_table.sum(axis=0), axis=1)
quality_pct.T.plot(kind="bar", stacked=True, figsize=(9, 5))
plt.ylabel("Fraction")
plt.xlabel("Subset")
plt.title("Quality-class composition across subsets")
plt.legend(title="Quality")
plt.grid(True, axis="y", alpha=0.3)
savefig("fig_quality_composition.png")
plt.show()

# Fig 3: validation curves
plt.figure(figsize=(8, 5))
for subset_name, grp in validation_table.groupby("subset"):
    plt.plot(grp["tolerance_h"], grp["f1"], marker="o", label=subset_name)
plt.xlabel("Matching tolerance (hours)")
plt.ylabel("F1 score")
plt.title("Validation against IZMIRAN 2019")
plt.legend()
plt.grid(True, alpha=0.3)
savefig("fig_validation_f1_curves.png")
plt.show()

# Fig 4: precision-recall at 18h
v18 = validation_table.loc[validation_table["tolerance_h"] == 18].copy()
plt.figure(figsize=(6.5, 5))
for _, r in v18.iterrows():
    plt.scatter(r["recall"], r["precision"], s=80)
    plt.text(r["recall"] + 0.005, r["precision"] + 0.005, r["subset"], fontsize=9)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–recall trade-off at ±18 h")
plt.grid(True, alpha=0.3)
savefig("fig_precision_recall_18h.png")
plt.show()

# Fig 5: supported stations distribution
plt.figure(figsize=(8, 5))
cat["n_supported_stations"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Number of supported stations")
plt.ylabel("Count")
plt.title("Distribution of supported stations per event")
plt.grid(True, axis="y", alpha=0.3)
savefig("fig_supported_stations_distribution.png")
plt.show()

# Fig 6: support-only stations distribution
plt.figure(figsize=(8, 5))
cat["n_support_only_stations"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Number of support-only stations")
plt.ylabel("Count")
plt.title("Distribution of support-only stations per event")
plt.grid(True, axis="y", alpha=0.3)
savefig("fig_support_only_distribution.png")
plt.show()

# Fig 7: supported drop vs supported stations
plt.figure(figsize=(7, 5))
plt.scatter(cat["n_supported_stations"], cat["mean_supported_drop"], alpha=0.6)
plt.xlabel("Number of supported stations")
plt.ylabel("Mean supported drop (%)")
plt.title("Support extent vs mean supported drop")
plt.grid(True, alpha=0.3)
savefig("fig_supported_drop_vs_nstations.png")
plt.show()

# Fig 8: promotion effect
promotion_counts = pd.Series(
    {
        "promoted_from_single_trigger": int(cat["promoted_from_single_trigger"].sum()),
        "not_promoted_from_single_trigger": int((~cat["promoted_from_single_trigger"]).sum()),
    }
)
plt.figure(figsize=(6, 4.5))
promotion_counts.plot(kind="bar")
plt.ylabel("Count")
plt.title("Promotion from single-trigger events")
plt.grid(True, axis="y", alpha=0.3)
savefig("fig_promotion_counts.png")
plt.show()


# ============================================================
# 6) Select a few case studies automatically
# ============================================================
# a) best aligned matches in core_main
cases_best = match_core_main_18.sort_values("dt_hours", key=lambda s: s.abs()).head(3).copy()

# b) best promoted-from-single-trigger matches
cases_promoted = match_core_main_18.loc[match_core_main_18["promoted_from_single_trigger"] == True].sort_values("dt_hours", key=lambda s: s.abs()).head(2).copy()

case_pool = pd.concat([cases_best, cases_promoted], ignore_index=True)
case_pool = case_pool.drop_duplicates(subset=["repr_time"]).reset_index(drop=True)

case_selection = case_pool[[c for c in [
    "t_izmiran", "repr_time", "dt_hours", "n_supported_stations",
    "n_support_only_stations", "supported_stations", "support_only_stations",
    "quality", "mean_supported_drop", "promoted_from_single_trigger"
] if c in case_pool.columns]]

display(case_selection)
case_selection.to_csv(OUT / "table_selected_case_studies.csv", index=False)


# ============================================================
# 7) Optional quick-look plots for selected cases
# ============================================================
# These plots use raw NMDB counts transformed to a causal relative deviation
# for a compact reviewer-facing visualization.
for i, row in case_pool.iterrows():
    event_time = pd.Timestamp(row["repr_time"])
    stations = sorted(str(row["supported_stations"]).split(",")) if pd.notna(row.get("supported_stations", np.nan)) else []
    stations = [s for s in stations if s in ["MXCO", "JUNG1", "LMKS", "NEWK", "KERG", "OULU", "DOMC", "INVK", "APTY", "AATB"]]
    if not stations:
        continue

    start = event_time - pd.Timedelta(hours=48)
    end = event_time + pd.Timedelta(hours=48)
    df_case = read_nmdb_subset(start, end, stations)
    if df_case.empty:
        continue
    df_case = robust_relative_deviation(df_case, stations, baseline_hours=72.0)

    plt.figure(figsize=(10, 5.5))
    for st in stations:
        col = f"deltaN_{st}"
        if col in df_case.columns:
            plt.plot(df_case["DATETIME"], df_case[col], label=st)
    plt.axvline(event_time, linestyle="--")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Relative deviation (%)")
    plt.title(f"Selected case {i+1}: {event_time.strftime('%Y-%m-%d %H:%M UTC')}")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    savefig(f"fig_case_{i+1}_{event_time.strftime('%Y%m%d_%H%M')}.png")
    plt.show()


# ============================================================
# 8) Narrative-ready key numbers
# ============================================================
key_18 = validation_table.loc[validation_table["tolerance_h"] == 18].copy().sort_values("subset")
key_24 = validation_table.loc[validation_table["tolerance_h"] == 24].copy().sort_values("subset")

key_numbers = {
    "n_total": int(len(cat)),
    "n_core_main": int(len(core_main)),
    "n_core_conservative": int(len(core_conservative)),
    "n_exploratory": int(len(exploratory_final)),
    "n_promoted_from_single_trigger": int(cat["promoted_from_single_trigger"].sum()),
    "core_main_2019_n": int(len(core_main_2019)),
    "core_conservative_2019_n": int(len(core_cons_2019)),
    "all_18h_f1": float(validation_table.query("subset == 'all' and tolerance_h == 18")["f1"].iloc[0]),
    "core_main_18h_f1": float(validation_table.query("subset == 'core_main' and tolerance_h == 18")["f1"].iloc[0]),
    "core_conservative_18h_f1": float(validation_table.query("subset == 'core_conservative' and tolerance_h == 18")["f1"].iloc[0]),
    "exploratory_18h_f1": float(validation_table.query("subset == 'exploratory' and tolerance_h == 18")["f1"].iloc[0]),
}

key_numbers_df = pd.DataFrame([key_numbers])
display(key_numbers_df)
key_numbers_df.to_csv(OUT / "table_key_numbers.csv", index=False)

print("\nResults notebook outputs written to:", OUT.resolve())
