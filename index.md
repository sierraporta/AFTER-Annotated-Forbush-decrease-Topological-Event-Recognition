---
layout: default
title: "AFTER – Annotated Forbush–decrease Topological Event Recognition"
---

# AFTER: A Complexity–Aware Catalogue of Forbush Decreases

**AFTER** (Annotated Forbush–decrease Topological Event Recognition) is a framework to detect and characterise **Forbush–decrease–like events** in multi–station neutron–monitor data.

<div class="link-box">

<p>
This site documents the public AFTER catalogues, explains how they were constructed,
and provides simple examples of how to use them in your own work.
</p>

<ul>
  <li>
    <strong>Results directory:</strong><br>
    <a href="https://github.com/sierraporta/AFTER-Annotated-Forbush-decrease-Topological-Event-Recognition/tree/main/Results">
      GitHub repo – <code>Results/</code> folder
    </a>
  </li>
  <li>
    <strong>Full AFTER–M catalogue (2019–2025):</strong><br>
    <a href="https://github.com/sierraporta/AFTER-Annotated-Forbush-decrease-Topological-Event-Recognition/blob/main/Results/AFTER_M_full_catalog.csv">
      AFTER_M_full_catalog.csv
    </a>
  </li>
  <li>
    <strong>AFTER–M 2019 catalogue:</strong><br>
    <a href="https://github.com/sierraporta/AFTER-Annotated-Forbush-decrease-Topological-Event-Recognition/blob/main/Results/sierra_M_2019_catalog.csv">
      AFTER_M_2019_catalog.csv
    </a>
  </li>
</ul>

</div>


#### How to cite
If you use the AFTER method, catalogues or scripts in a publication, please cite:

_*D. Sierra-Porta (2026). Annotated Forbush–decrease Topological Event Recognition (AFTER): a complexity-aware catalogue of neutron-monitor Forbush decreases (journal and DOI to be added when available).*_

You can also cite this repository directly, for example as:
_*Sierra-Porta, D. (2026). AFTER: Annotated Forbush–decrease Topological Event Recognition. GitHub repository. [https://github.com/<your-user>/<your-repo>](https://github.com/sierraporta/AFTER-Annotated-Forbush-decrease-Topological-Event-Recognition/tree/main)*_

#### License
This project is licensed under the MIT License.

#### For questions, comments or suggestions, please contact:
_*David Sierra-Porta, Universidad Tecnológica de Bolívar (UTB). Email: dporta@utb.edu.co.*_

Contributions and constructive feedback are very welcome — feel free to open an issue or submit a pull request.

---

## 1. What is AFTER?
Forbush decreases (FDs) are transient depressions in the galactic cosmic–ray flux associated with interplanetary coronal mass ejections (ICMEs), shocks and stream interaction regions. Most existing FD catalogues are **amplitude–driven**: an event is selected if the cosmic–ray intensity drops below some threshold at one or a few neutron monitors.

**AFTER takes a different route.**
Instead of using only the depth of the depression, AFTER:
- works on a **network** of neutron monitors;
- tracks not only the **fractional decrease** but also the **temporal organisation** of the signal;
- combines geometric and ordinal **complexity measures** (permutation entropy and Katz fractal dimension);
- groups station–level detections into **network–level coincidences** with quality labels.

The goal is to obtain a **reproducible, complexity–aware catalogue** that complements existing lists such as the IZMIRAN Forbush–effect catalogue and the Lockwood FD list.

---

## 2. Data and time span
The current version of AFTER uses:

- 2-minute, pressure–corrected neutron–monitor counts from **ten NMDB stations** (MXCO, JUNG1, LMKS, KERG, OULU, NEWK, DOMC, INVK, APTY, AATB),
- for the period **2019–2025** (inclusive),
- plus the IZMIRAN Forbush–effect list for **2019** as an external benchmark.

The raw count rates are taken from the [Neutron Monitor Database (NMDB)](https://www.nmdb.eu).  AFTER uses only publicly available data; no station–specific corrections beyond those provided  by NMDB are applied.

---

## 3. How AFTER works (high–level)
At a very high level, the AFTER pipeline has four stages:

1. **Station–level preprocessing**  
   Each station’s 2-minute count rate is smoothed, and a slowly varying background is estimated by a multi–hour moving average.  The main amplitude channel is the **fractional change relative to background**, $\Delta N_s(t)$ (in percent).

2. **Sliding–window complexity**  
   - On sliding windows $(\sim$3 h) for each station, AFTER computes two markers:
     - *Normalised permutation entropy* (ordinal complexity of the time series),
     - *Katz fractal dimension* (geometric roughness of the trajectory).  
   - These are standardised (median + MAD) over a multi–year reference period and combined into a single **complexity amplitude** $A_{\text{complex},s}(t)$.

3. **Station–level event detection**  
   - Candidate events are defined around local minima of the count rate where **both**:
     - the mean fractional decrease over the event, and
     - the maximum complexity amplitude exceed prescribed thresholds.  
   - This step is run in three configurations:
     - **strict** (few but very strong candidates),
     - **medium** (balanced, used as default catalogue),
     - **loose** (many candidates, high recall).

4. **Network–level coincidences and quality classes**  
   Station–level events with similar times are grouped into **network events**.  Each network event is annotated with:
   - station multiplicity $n_{\text{st}}$,
   - mean/max fractional decrease across stations,
   - mean/max complexity amplitude across stations.  

   A **quality class** summarises strength and coherence:

   - **A** – multi–station events $(n_{\text{st}} \ge 3$),  
   - **B** – bi–station events $(n_{\text{st}} = 2$),  
   - **C** – strong single–station events,  
   - **D** – marginal events.

The **medium configuration** is referred to as **AFTER–M** and is the default catalogue used in the paper.

---

## 4. Catalogue files
All catalogue files live in the [https://github.com/sierraporta/AFTER-Annotated-Forbush-decrease-Topological-Event-Recognition/tree/main/Results](https://github.com/sierraporta/AFTER-Annotated-Forbush-decrease-Topological-Event-Recognition/tree/main/Results) directory of this repository.

### 4.1 Station–level event lists
- `sierra_events_strict.csv`
- `sierra_events_medium.csv`
- `sierra_events_loose.csv`

Each row contains one **station–level** candidate with columns such as:
- `station` – NMDB station code (e.g. `JUNG1`),
- `onset_time`, `min_time`, `rec_time` – onset, minimum and recovery timestamps,
- `drop_mean`, `drop_max` – mean and maximum fractional decrease (%),
- `A_complex_mean`, `A_complex_max` – mean and maximum complexity amplitude,
- additional flags and diagnostics.

These are mostly useful if you want to build your own grouping criteria  
or explore station–specific behaviour.

### 4.2 Network–level coincidence catalogues
- `sierra_coincidences_strict.csv`
- `sierra_coincidences_medium.csv`
- `sierra_coincidences_loose.csv`

Each row is a **network–level event** (coincidence across stations) with:
- `repr_time` – representative time (median of station minima),
- `year` – year of the event,
- `n_stations` – number of stations contributing,
- `stations` – comma–separated list of station codes,
- `drop_mean`, `drop_max` – network–mean and network–maximum fractional decrease,
- `A_complex_mean`, `A_complex_max` – network–mean and network–maximum complexity amplitude,
- `quality` – A/B/C/D class (see above),
- configuration tags (strict / medium / loose).

### 4.3 AFTER–M catalogue for 2019
- `sierra_M_2019_catalog.csv`

This is the main **AFTER–M network catalogue for 2019**,  
annotated with quality classes and matched to IZMIRAN where possible.  
Additional columns indicate:

- whether the event matches an IZMIRAN entry within ±24 h,
- the index of the matched IZMIRAN event (if any),
- and cross–quality labels summarising the joint classification  
  in AFTER and IZMIRAN.

### 4.4 IZMIRAN list with AFTER flags
- `izmiran_2019_with_sierra_flags.csv`

The 2019 IZMIRAN Forbush–effect list, augmented with:
- the nearest AFTER–M event (if any),
- time difference in hours,
- and basic AFTER properties at the match (drop, complexity, multiplicity).

This file is useful if you want to start from IZMIRAN  
and ask “what does AFTER say about this event?”.

---

## 5. How to use the catalogue (example in Python)
Here is a minimal example of how to load and explore the AFTER–M 2019 catalogue in Python:

```python
import pandas as pd

# Load AFTER–M 2019 network catalogue
after_2019 = pd.read_csv("results/sierra_M_2019_catalog.csv", parse_dates=["repr_time"])

# Quick overview
print(after_2019.head())
print(after_2019["quality"].value_counts())

# Select A-class (multi-station) events only
A_events = after_2019[after_2019["quality"] == "A"]

# Plot the distribution of mean fractional decrease
A_events["drop_mean"].hist(bins=20)
```
