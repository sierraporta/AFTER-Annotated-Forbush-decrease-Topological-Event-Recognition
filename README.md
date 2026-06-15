# AFTER — Annotated Forbush-decrease Topological Event Recognition

**Pipeline v2.4 | Sierra-Porta (2026)**

This repository contains the complete research compendium for:

> D. Sierra-Porta, *Complexity-enhanced detection and interplanetary
> characterization of Forbush decreases in neutron monitor networks
> (2016–2025)*, submitted to *Journal of Geophysical Research: Space Physics*.

---

## Overview

AFTER is a multi-station neutron-monitor detection framework that combines
a causal amplitude channel with a complexity-amplitude channel built from
**permutation entropy** (PE) and **Katz fractal dimension** (KFD), then
evaluates network-level coherence through an explicit support stage.
Applied to 19 NMDB stations at 2-minute cadence over 2016–2025, AFTER
detects 3.3× more gradual/unclassified-onset Forbush decreases than a
pure amplitude detector, and characterises the network signature by
interplanetary driver type.

---

## Repository structure

```
.
├── 001_GetData.ipynb              # Download NMDB + OMNI data
├── 002_PrepareData.ipynb          # Preprocessing and integration
├── 003_Analysis_v2.ipynb          # Catalogue analysis and validation
├── 004_FD_Characterization.ipynb  # Driver characterisation and figures
├── AFTER_detector.py              # Main detection pipeline (v2.4)
├── test_amplitude_only.py         # Amplitude-only baseline comparison
├── test_grouping_window_active.py # Grouping window sensitivity test
├── alldata_integrated_1h.parquet  # Integrated NMDB+OMNI at 1-h cadence
├── alldata_integrated_2min.parquet# Integrated NMDB+OMNI at 2-min cadence
├── dataset_meta.json              # Dataset metadata (stations, variables)
├── feid_clean.parquet             # FEID/IZMIRAN reference catalogue
├── feid-3.csv                     # FEID working copy
└── Results_v2_4/                  # All outputs from pipeline v2.4
    ├── AFTER_catalog_detector.csv         # Main event catalogue (908 events)
    ├── trigger_catalog_detector.csv       # Station-level trigger catalogue
    ├── amplitude_vs_after_comparison.csv  # AFTER vs amplitude comparison
    ├── station_panels/                    # Per-station complexity panels (checkpoint)
    ├── Analysis/                          # Figures from 003_Analysis_v2
    └── Characterization/                  # Figures and tables from 004
```

---

## Data

| File | Description | Cadence | Period |
|------|-------------|---------|--------|
| `alldata_integrated_2min.parquet` | 19 NMDB stations + 7 OMNI variables | 2 min | 2016–2025 |
| `alldata_integrated_1h.parquet` | Same, resampled to 1 h | 1 h | 2016–2025 |
| `feid_clean.parquet` | FEID/IZMIRAN catalogue | — | 1978–2024 |

**NMDB stations included:** NEWK, KIEL2, KERG, OULU, APTY, FSMT, JUNG,
JUNG1, LMKS, DRBS, ATHN, MXCO, NANM, CALM, ROME, AATB, BKSN, NRLK, IRKT.

Raw neutron-monitor data are available from the NMDB portal
([www.nmdb.eu](https://www.nmdb.eu)). OMNI 2-minute data were obtained
from NASA/GSFC CDAWeb ([cdaweb.gsfc.nasa.gov](https://cdaweb.gsfc.nasa.gov)).
The FEID catalogue is maintained at IZMIRAN
([spaceweather.izmiran.ru](http://spaceweather.izmiran.ru/eng/dbs.html)).

---

## Reproducing the results

### Requirements

```bash
Python >= 3.10
pandas >= 1.5
numpy >= 1.23
scipy >= 1.9
matplotlib >= 3.6
pyarrow >= 10.0        # for parquet files
```

Install all dependencies:

```bash
pip install pandas numpy scipy matplotlib pyarrow
```

### Step-by-step

**Step 1 — (optional) Re-download raw data**
Run `001_GetData.ipynb` to fetch fresh NMDB and OMNI data.
Skip this step if you use the provided parquet files.

**Step 2 — (optional) Re-run preprocessing**
Run `002_PrepareData.ipynb` to rebuild the integrated dataset.
Skip this step if you use the provided parquet files.

**Step 3 — Run the detector**

```bash
python AFTER_detector.py
```

This will:
- Load station panels from `Results_v2_4/station_panels/` (checkpoint — fast)
- Detect triggers and build the event catalogue
- Enrich with FEID/OMNI context
- Apply FD-like confirmation criteria
- Save all outputs to `Results_v2_4/`

Force full rebuild (recomputes all station panels from scratch, slow):

```bash
python AFTER_detector.py --force-rebuild
```

**Step 4 — Catalogue analysis (Section 4.1–4.2 of the paper)**
Run `003_Analysis_v2.ipynb` in order. All figures are saved to
`Results_v2_4/Analysis/`.

**Step 5 — Driver characterisation (Sections 4.3–4.5)**
Run `004_FD_Characterization.ipynb` in order. All figures are saved to
`Results_v2_4/Characterization/`.

**Step 6 — AFTER vs amplitude comparison (Table 3)**

```bash
python test_amplitude_only.py
```

Output saved to `Results_v2_4/amplitude_vs_after_comparison.csv`.

---

## Key pipeline parameters

| Stage | Parameter | Value |
|-------|-----------|-------|
| Background | *L*_bg | 72 h causal rolling median |
| Complexity window | *L*_win | 3 h (90 pts at 2-min cadence) |
| PE embedding | *m*, τ | m = 3, τ = 1 |
| Standardisation ref. | *L*_ref | 30-day trailing window |
| Trigger threshold | *A*_th | 2.0 |
| Min. complexity duration | *T*_A,min | 2 h |
| Min. amplitude drop | ΔN_min | 1.0 % |
| Support window | — | [−24, +36] h |
| Min. support drop | — | 0.75 % for ≥ 0.5 h |
| FEID matching tolerance | Δ*t* | ±36 h |

---

## Main results

| Result | Value |
|--------|-------|
| Total events (2016–2025) | 908 |
| Core subset (*n*_sup ≥ 2) | 827 (91.1 %) |
| FD-like confirmed | 853 (93.9 %) |
| Detection gain vs amplitude-only (CIR/NaN) | ×3.3 |
| Recall vs FEID 2019 at ±18 h | 0.432 |
| Precision vs FEID 2019 at ±18 h | 0.571 |
| KS (SSC vs unclassified, *n*_sup) | 0.484, *p* < 0.001 |
| δt median SSC / unclassified | 6.0 h / 11.2 h |

---

## Citation

If you use this code or data, please cite:

> Sierra-Porta, D. (2026). Complexity-enhanced detection and interplanetary
> characterization of Forbush decreases in neutron monitor networks (2016–2025).
> *Journal of Geophysical Research: Space Physics*. [submitted]

---

## Licence

Code and derived data products: **Creative Commons Attribution 4.0
International (CC BY 4.0)**.
See [LICENSE](LICENSE) for details.

Raw NMDB and OMNI data are subject to their respective data provider
policies (NMDB data policy: [www.nmdb.eu/use-of-data](https://www.nmdb.eu/use-of-data)).

---

## Contact

D. Sierra-Porta — dporta@utb.edu.co  
Universidad Tecnológica de Bolívar, Cartagena de Indias, Colombia  
ORCID: [0000-0003-3461-1347](https://orcid.org/0000-0003-3461-1347)
