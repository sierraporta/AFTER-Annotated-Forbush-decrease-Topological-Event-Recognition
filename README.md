# AFTER – Annotated Forbush–decrease Topological Event Recognition

This repository contains the code, notebooks and catalogues associated with the **AFTER** framework  
(**A**nnotated **F**orbush–decrease **T**opological **E**vent **R**ecognition), a network–level method to detect Forbush–decrease–like events from multi–station neutron–monitor data.

AFTER combines background–referenced amplitude with sliding–window **complexity markers**  
(normalised permutation entropy and Katz fractal dimension) and merges station–level candidates into network–level coincidences with quality classes (A/B/C/D).  
The method was applied to 2-minute data from ten NMDB stations over 2019–2025 and benchmarked against the IZMIRAN/FEID Forbush catalogue for 2019.

---

## Main features

- **Complexity–aware FD detection**  
  - Uses permutation entropy and Katz fractal dimension computed on sliding windows.  
  - Robust standardisation (median + MAD) and aggregation into a single complexity amplitude channel.

- **Amplitude + complexity criteria**  
  - Events are required to cross thresholds in *both* background–referenced fractional decrease and complexity amplitude.  
  - Three configurations: **strict**, **medium** (AFTER-M, default) and **loose**.

- **Network–level coincidence catalogue**  
  - Station–level detections are grouped into time coincidences across a 10-station subset of NMDB.  
  - Each network event is annotated with mean/max fractional decrease, mean/max complexity amplitude and station multiplicity.

- **Quality classes (A/B/C/D)**  
  - A: multi-station events with \( n_{\rm st} \ge 3 \).  
  - B: two–station coincidences.  
  - C: strong single-station events.  
  - D: marginal events passing minimum cuts.

- **Benchmarking against IZMIRAN (2019)**  
  - One–to–one matching procedure with a ±24 h window.  
  - Recall statistics and comparison of amplitude/complexity distributions for matched vs AFTER–only events.

## Data sources
- Neutron monitor data: downloaded from the Neutron Monitor Database (NMDB), https://www.nmdb.eu
- Reference FD catalogue (2019): Forbush Effects and Interplanetary Disturbances (FEID) catalogue provided by IZMIRAN, http://spaceweather.izmiran.ru/eng/dbs.html

Please cite NMDB / contributing stations and the IZMIRAN FEID catalogue if you reuse the data or catalogues.

## How to cite
If you use the AFTER method, catalogues or scripts in a publication, please cite:

D. Sierra-Porta, et al. Annotated Forbush–decrease Topological Event Recognition (AFTER): a complexity-aware catalogue of neutron-monitor Forbush decreases
(journal and DOI to be added when available).

You can also cite this repository directly, for example as:
Sierra-Porta, D. (2025). AFTER: Annotated Forbush–decrease Topological Event Recognition. GitHub repository. [https://github.com/<your-user>/<your-repo>](https://github.com/sierraporta/AFTER-Annotated-Forbush-decrease-Topological-Event-Recognition/tree/main)

## License
This project is licensed under the MIT License.

---

## Repository structure

```text
.
├── notebooks/
│   ├── 01_after_detection_from_DataStudy.ipynb
│   ├── 02_after_catalogs_strict_medium_loose.ipynb
│   └── 03_after_vs_izmiran2019_analysis.ipynb
│
├── data/
│   ├── DataStudy.csv                        # multi-station 2-min counts (NMDB subset)
│   ├── izmiran2019.txt                      # digitised IZMIRAN FD list for 2019
│   └── FD20240510.txt, FD20211104.txt       # multi-station segments for case studies
│
├── results/
│   ├── sierra_events_strict.csv            # AFTER station-level events (strict)
│   ├── sierra_events_medium.csv            # AFTER station-level events (medium = AFTER-M)
│   ├── sierra_events_loose.csv             # AFTER station-level events (loose)
│   ├── sierra_coincidences_strict.csv      # AFTER network events (strict)
│   ├── sierra_coincidences_medium.csv      # AFTER network events (medium)
│   ├── sierra_coincidences_loose.csv       # AFTER network events (loose)
│   ├── sierra_M_2019_catalog.csv           # AFTER-M 2019 full annotated catalogue
│   └── izmiran_2019_with_sierra_flags.csv  # IZMIRAN 2019 + AFTER match flags
│
├── src/                                    # helper functions (optional)
│   └── after_utils.py
│
├── figures/
│   ├── after_year_level_summary.png
│   ├── after_vs_izmiran_recall.png
│   └── after_cases_2021_2024.png
│
├── requirements.txt
└── README.md

