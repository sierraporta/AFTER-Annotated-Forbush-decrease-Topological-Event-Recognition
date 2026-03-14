# AFTER — Annotated Forbush-decrease Topological Event Recognition

AFTER is a support-based, complexity-aware framework for identifying and annotating **FD-like depressions** in multi-station neutron-monitor data.

https://sierraporta.github.io/AFTER-Annotated-Forbush-decrease-Topological-Event-Recognition/

The revised version of the project uses a **single station-level detector** combined with an explicit **network-support stage**. Instead of building separate strict / medium / loose catalogues, the final workflow produces:

- a full detector-based catalogue,
- a **main core** subset,
- a **conservative core** subset,
- and an **exploratory** subset.

This repository contains the processed catalogues, validation summaries, representative figures, and analysis scripts associated with the revised AFTER workflow applied to ten NMDB stations over **2018–2025**.

**Suggested manuscript citation**
`Sierra-Porta, D. AFTER: A support-based, complexity-aware framework for identifying FD-like depressions in multi-station neutron-monitor data (2018–2025).
[manuscript in revision / update with final journal reference]`

---

## What AFTER does

AFTER detects candidate FD-like depressions in neutron-monitor count-rate series by combining:

- a **causal background-referenced amplitude channel**,
- a **complexity-amplitude channel** built from permutation entropy and Katz fractal dimension,
- and a **support-based network promotion step**.

The workflow is:

1. compute station-level trigger candidates from joint amplitude and complexity excursions;
2. group temporally compatible station triggers into network episodes;
3. evaluate support across all stations, including stations that did not independently trigger;
4. classify events into support-based subsets and quality classes.

This design allows the catalogue to retain clearly network-wide events while also recovering physically plausible events whose coherence is unevenly expressed across stations.

---

## Data sources

The neutron-monitor data used here come from the [Neutron Monitor Database (NMDB)](https://www.nmdb.eu), using pressure- and efficiency-corrected 2-minute count rates for the stations:

- MXCO
- JUNG1
- LMKS
- KERG
- OULU
- NEWK
- DOMC
- INVK
- APTY
- AATB

The external benchmark used for validation is the 2019 Forbush-effect list from the IZMIRAN / FEID database:

- http://spaceweather.izmiran.ru/eng/dbs.html

---

## Final catalogue products

The revised pipeline produces the following principal outputs.

### 1. Full detector-based catalogue
`AFTER_catalog_detector.csv`

This is the complete network-level catalogue produced by the revised AFTER detector after grouping and support evaluation.

### 2. Main core subset
`AFTER_catalog_detector_core_main.csv`

This is the default working catalogue.  
It contains all events with:

- `n_supported_stations >= 2`

### 3. Conservative core subset
`AFTER_catalog_detector_core_conservative.csv`

This is a stricter subset intended for sensitivity analysis.  
It contains all events with:

- `n_supported_stations >= 2`
- `n_support_only_stations >= 1`

### 4. Exploratory subset
`AFTER_catalog_detector_exploratory_final.csv`

This contains lower-confidence candidates with:

- `n_supported_stations < 2`

---

## Validation products

The repository also includes validation outputs against the 2019 IZMIRAN reference list, including:

- `validation_summary_detector.csv`
- `match_table_detector_all_18h.csv`
- `match_table_detector_core_18h.csv`
- `match_table_detector_exploratory_18h.csv`

These tables report recall, precision, F1, and representative event-level matches for multiple temporal tolerances.

---

## Support diagnostics

The revised workflow explicitly tracks support across stations. Relevant outputs include:

- `support_members_detector.csv`
- `promotion_summary_detector.csv`
- `trigger_catalog_detector.csv`

These files allow inspection of:

- which stations triggered an event,
- which stations supported an event without independently triggering,
- and how many events were promoted from an initially single-trigger state.

---

## Main results in brief

Using ten NMDB stations over 2018–2025, the revised AFTER catalogue contains:

- **700** total events,
- **605** events in the main core subset,
- **552** events in the conservative core subset,
- **95** exploratory events,
- and **190** events promoted from an initially single-trigger state by additional network support.

Validation against the IZMIRAN 2019 reference list shows that the most useful agreement is concentrated in the support-based core population. At a matching tolerance of ±18 h:

- the **full catalogue** reaches recall **0.379**, precision **0.522**, and F1 **0.439**;
- the **main core subset** reaches recall **0.347**, precision **0.573**, and F1 **0.432**.

This means the main core subset preserves most informative matches while improving precision relative to the full catalogue.

---

## Repository structure

Suggested structure for the revised repository:

```text
.
├── README.md
├── index.md
├── data/
│   └── izmiran_2019_parsed.csv
├── results/
│   ├── AFTER_catalog_detector.csv
│   ├── AFTER_catalog_detector_core_main.csv
│   ├── AFTER_catalog_detector_core_conservative.csv
│   ├── AFTER_catalog_detector_exploratory_final.csv
│   ├── validation_summary_detector.csv
│   ├── support_members_detector.csv
│   ├── promotion_summary_detector.csv
│   ├── trigger_catalog_detector.csv
│   ├── match_table_detector_all_18h.csv
│   ├── match_table_detector_core_18h.csv
│   └── match_table_detector_exploratory_18h.csv
├── figures/
│   ├── fig_counts_by_year.png
│   ├── fig_quality_composition.png
│   ├── fig_promotion_counts.png
│   ├── fig_validation_f1_curves.png
│   ├── fig_precision_recall_121824h.png
│   ├── fig_case_1_20190805_0100.png
│   ├── fig_case_2_20190906_0206.png
│   └── fig_case_3_20191007_1138.png
├── scripts/
│   ├── Processing_Catalogs_v2_2_AFTER.py
│   └── ResultsAnalysis_v2_2_AFTER.py
└── manuscript/
    └── Paper
