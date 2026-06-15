"""
test_amplitude_only.py
----------------------
Compara AFTER (amplitude + complexity) vs un detector puramente de amplitud.

Pregunta: ¿añade A_complex valor real, o los mismos eventos
se capturarían con solo deltaN <= -1%?

Estrategia:
  1. Para cada evento FEID, verificar si AFTER lo captura (ya sabemos esto)
  2. Verificar si un detector puramente de amplitud (deltaN <= umbral
     durante >= duración mínima) también lo capturaría
  3. Comparar especialmente en la población CIR/difusa donde
     la complejidad debería añadir más valor

Detector de amplitud puro:
  - deltaN <= -1.0% sostenido >= 2h en al menos 2 estaciones dentro de ±18h
  - Sin ningún criterio de complejidad (sin A_complex)
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE         = Path("Results_v2_4")
PANELS_DIR   = BASE / "station_panels"
FEID_PARQUET = Path("feid_clean.parquet")
CAT_CSV      = BASE / "AFTER_catalog_detector.csv"

# Parámetros del detector de amplitud puro
AMP_THRESH_PCT  = -1.0   # deltaN umbral (%)
AMP_MIN_DUR_MIN = 120    # minutos mínimos sostenidos (2h)
AMP_MIN_STATIONS = 2     # mínimo de estaciones que lo detectan
CADENCE_MIN      = 2.0   # resolución en minutos
TOL_MATCH        = pd.Timedelta(hours=18)
WIN_H            = 48    # ventana de búsqueda alrededor del evento FEID

# ── Cargar datos ──────────────────────────────────────────────────────────────
print("Cargando datos...", flush=True)

# FEID
feid = pd.read_parquet(FEID_PARQUET)
feid.index = pd.to_datetime(feid.index)
feid["t_izmiran"] = feid.index
if feid["t_izmiran"].dt.tz is not None:
    feid["t_izmiran"] = feid["t_izmiran"].dt.tz_convert(None)
feid["year"] = feid["t_izmiran"].dt.year
ons_col = "ons type" if "ons type" in feid.columns else "ons"

# AFTER catalogue
cat = pd.read_csv(CAT_CSV)
cat["repr_time"] = pd.to_datetime(cat["repr_time"])
if cat["repr_time"].dt.tz is not None:
    cat["repr_time"] = cat["repr_time"].dt.tz_convert(None)

# FEID matched por AFTER
feid["after_matched"] = [
    (cat["repr_time"] - t).abs().min() <= TOL_MATCH
    for t in feid["t_izmiran"]
]

# Clasificar drivers
def driver(ons):
    if pd.isna(ons): return "CIR"
    if ons == "SSC":  return "SSC"
    if ons == "SI":   return "SI"
    return "CIR"

feid["driver"] = feid[ons_col].apply(driver)

# Cargar paneles — solo deltaN
print("Cargando paneles (solo deltaN)...", flush=True)
panels_dN = {}
for pf in sorted(PANELS_DIR.glob("*_panel.csv")):
    st = pf.stem.replace("_panel","")
    p  = pd.read_csv(pf, usecols=lambda c: c in
                     ["DATETIME","datetime","index","time","deltaN"])
    tc = next((c for c in ["DATETIME","datetime","index","time"]
                if c in p.columns), None)
    if tc and "deltaN" in p.columns:
        p[tc] = pd.to_datetime(p[tc], utc=True, errors="coerce")
        p[tc] = p[tc].dt.tz_convert(None)
        p = p.set_index(tc)
        panels_dN[st] = p["deltaN"]
        print(f"  ✓ {st}", flush=True)

STATIONS = list(panels_dN.keys())
print(f"  Total: {len(STATIONS)} estaciones")

# ── Detector de amplitud puro ─────────────────────────────────────────────────
def amplitude_detector(t0, panels, thresh=AMP_THRESH_PCT,
                        min_dur_min=AMP_MIN_DUR_MIN,
                        win_h=WIN_H, cadence=CADENCE_MIN):
    """
    Devuelve True si al menos AMP_MIN_STATIONS estaciones muestran
    deltaN <= thresh sostenido >= min_dur_min minutos en ventana ±win_h.
    """
    t_pre = t0 - pd.Timedelta(hours=win_h)
    t_pos = t0 + pd.Timedelta(hours=win_h)
    min_steps = int(min_dur_min / cadence)

    n_detected = 0
    for st, dN in panels.items():
        win = dN.loc[t_pre:t_pos]
        if win.empty:
            continue
        below = (win <= thresh)
        # Máxima racha consecutiva por debajo del umbral
        groups    = (below != below.shift()).cumsum()
        max_steps = below.groupby(groups).sum().max()
        if max_steps >= min_steps:
            n_detected += 1
        if n_detected >= AMP_MIN_STATIONS:
            return True
    return False

# ── Evaluar para todos los eventos FEID ──────────────────────────────────────
print(f"\nEvaluando detector de amplitud puro...", flush=True)
print(f"  Umbral: deltaN <= {AMP_THRESH_PCT}%  "
      f"sostenido >= {AMP_MIN_DUR_MIN} min  "
      f"en >= {AMP_MIN_STATIONS} estaciones", flush=True)

n_total = len(feid)
amp_detected = []
for idx, (_, ev) in enumerate(feid.iterrows()):
    if idx % 100 == 0:
        print(f"  {idx+1}/{n_total}...", end="\r", flush=True)
    t0 = ev["t_izmiran"]
    amp_detected.append(amplitude_detector(t0, panels_dN))

feid["amp_detected"] = amp_detected
print(f"\n  Completado: {sum(amp_detected)} / {n_total} detectados")

# ── Comparación por driver ────────────────────────────────────────────────────
print("\n" + "="*65)
print("COMPARACIÓN: AFTER vs DETECTOR DE AMPLITUD PURO")
print("="*65)

print(f"\nTolerancia matching: ±{TOL_MATCH.total_seconds()/3600:.0f}h")
print(f"\n{'Driver':8s}  {'N_FEID':>7}  "
      f"{'AFTER':>8}  {'Amp_only':>9}  "
      f"{'AFTER_only':>11}  {'Amp_only_only':>14}  {'Both':>6}")
print("-"*75)

results = []
for drv in ["SSC","SI","CIR"]:
    sub = feid[feid["driver"]==drv].copy()
    n   = len(sub)
    n_after  = sub["after_matched"].sum()
    n_amp    = sub["amp_detected"].sum()
    n_both   = (sub["after_matched"] & sub["amp_detected"]).sum()
    n_after_only = (sub["after_matched"] & ~sub["amp_detected"]).sum()
    n_amp_only   = (~sub["after_matched"] & sub["amp_detected"]).sum()
    n_neither    = (~sub["after_matched"] & ~sub["amp_detected"]).sum()

    print(f"{drv:8s}  {n:>7}  "
          f"{n_after:>7} ({n_after/n*100:.1f}%)  "
          f"{n_amp:>7} ({n_amp/n*100:.1f}%)  "
          f"{n_after_only:>9} ({n_after_only/n*100:.1f}%)  "
          f"{n_amp_only:>12} ({n_amp_only/n*100:.1f}%)  "
          f"{n_both:>4} ({n_both/n*100:.1f}%)")

    results.append({
        "driver":        drv,
        "N_feid":        n,
        "N_after":       n_after,
        "N_amp":         n_amp,
        "N_both":        n_both,
        "N_after_only":  n_after_only,
        "N_amp_only":    n_amp_only,
        "N_neither":     n_neither,
        "pct_after":     n_after/n*100,
        "pct_amp":       n_amp/n*100,
        "pct_after_only":n_after_only/n*100,
    })

# ── Resumen por magnitud (CIR only — donde más importa) ───────────────────────
print(f"\n{'='*65}")
print("DETALLE CIR: por rango de magnitud")
print("="*65)

cir = feid[feid["driver"]=="CIR"].copy()
bins   = [0, 0.5, 1.0, 2.0, 999]
labels = ["<0.5%","0.5-1%","1-2%",">2%"]
cir["mag_bin"] = pd.cut(cir["magnitude"], bins=bins, labels=labels)

print(f"\n{'Mag range':10s}  {'N':>5}  "
      f"{'AFTER':>8}  {'Amp_only':>9}  {'AFTER_only':>11}")
print("-"*55)
for lb in labels:
    sub = cir[cir["mag_bin"]==lb]
    if len(sub)==0: continue
    n   = len(sub)
    na  = sub["after_matched"].sum()
    namp= sub["amp_detected"].sum()
    nao = (sub["after_matched"] & ~sub["amp_detected"]).sum()
    print(f"{lb:10s}  {n:>5}  "
          f"{na:>6} ({na/n*100:.0f}%)  "
          f"{namp:>7} ({namp/n*100:.0f}%)  "
          f"{nao:>9} ({nao/n*100:.0f}%)")

# ── Resumen ejecutivo ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("RESUMEN EJECUTIVO")
print("="*65)

total_after_only = sum(r["N_after_only"] for r in results)
total_amp_only   = sum(r["N_amp_only"]   for r in results)
total_both       = sum(r["N_both"]       for r in results)

print(f"\n  Eventos FEID capturados SOLO por AFTER:      {total_after_only}")
print(f"  Eventos FEID capturados SOLO por amplitud:   {total_amp_only}")
print(f"  Capturados por ambos:                        {total_both}")
print(f"\n  → AFTER añade {total_after_only} eventos que amplitud pura no detecta")
print(f"  → Amplitud añade {total_amp_only} eventos que AFTER no detecta")

cir_after_only = results[2]["N_after_only"]
cir_amp_only   = results[2]["N_amp_only"]
print(f"\n  En población CIR (graduales):")
print(f"  AFTER captura {cir_after_only} eventos que amplitud pura pierde")
print(f"  → Esto cuantifica el valor añadido de A_complex para CIRs")

pd.DataFrame(results).round(2).to_csv(
    BASE / "amplitude_vs_after_comparison.csv", index=False)
print(f"\n✓ Guardado en {BASE}/amplitude_vs_after_comparison.csv")
