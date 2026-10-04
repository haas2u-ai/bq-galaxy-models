# Video scripts: constant-Lagrangian galaxy model and PG jets

Python scripts that produce the animations of E.P.J. de Haas's constant-Lagrangian (CL)
metric-inflow galaxy model and the three-rapidity Painlevé–Gullstrand (PG) jet geometry.
Every frame is computed from the model equations; nothing is drawn by hand.

* Videos and interactive 3D models: https://haas2u-ai.github.io/bq-galaxy-models/landing.html
* Interactive models only: https://haas2u-ai.github.io/bq-galaxy-models/

## Requirements

Python 3.9 or later, the packages in `requirements.txt`, and **ffmpeg** on the system path.

```
pip install -r requirements.txt
```

## Quick start

All scripts live in `scripts/` and write their MP4 to `videos/`. They can be started from any folder.

```
cd scripts
python animate.py                      # one full video (a few minutes)
BQ_MAX_FRAMES=60 python animate.py     # quick test: only the first 60 frames
```

The four fitting videos use precomputed solver paths and profile scans. These are included in
`scripts/cache/`; to regenerate them, run the matching `prep_*.py` script first (see the table).

## Which script makes which video

| Video | Run | Model / data modules |
|---|---|---|
| Metric-inflow spiral across cosmic time (redshift sequence) | `animate.py` | `model.py` |
| Double-Lagrangian nested spiral | `animate2L.py` | `model2L.py` |
| Bulge reset: birth of a nested spiral | `animate_reset.py` | `model_reset.py` |
| Spiral galaxy with polar jets (3D) | `animate_jet3d.py` | `model.py` |
| Polar jets: face-on to edge-on (θ sweep, X shape) | `animate_jet_theta.py` | `model.py` |
| Hubble sequence from cone occupation | `hubble_seq.py` | `model.py`, `model2L.py` |
| CL vs MOND vs DM: AIC/BIC comparison | `aic_movie.py` | `data/curated.json` |
| UGC 8286 single-L fit, step by step | `prep_ugc8286.py`, then `fit_movie.py` | `clfit.py` |
| UGC 1281 single-L fit, step by step | `prep_ugc1281.py`, then `fit_movie_1281.py` | `clfit.py`, `data1281.py` |
| IC 2574 two-Lagrangian fit | `prep_ic2574.py`, then `movie2L.py` | `model2574.py`, `data2574.py` |
| ESO079–G014: does Φ_BH improve the fit? | `prep_eso.py`, then `movie_eso.py` | `model_eso.py`, `data_eso.py` |
| The M87 jet in one PG geometry | `m87movie.py` | `pgjet.py` |
| The riverbed of space: Hubble, absorption in matter, CL spiral | `movie2.py` | `model2.py`, `riverbed.py` |

Shared helpers: `paths.py` (folders and the quick-test switch), `lm.py` (transparent
Levenberg–Marquardt that records each accepted step for the fitting videos).

## Model equations and display choices

Each video separates the model from presentation choices, and states both on screen.

* **Galaxy films** use the single- and two-Lagrangian CL velocity fields
  (`v_rad,eff = √(2GM/r) − H r`, `v_orb² = 3/2·X² − v_rad,eff²` outside the bulge) and spiral arms as
  streamlines of that flow. Stage parameters in the Hubble-sequence film and the tracer clocks are display choices.
* **Jet films** use helical streamlines on cones of fixed polar angle. The jet cones are drawn wider than
  their true angle so the helices are visible; jet speeds and helix pitch are display choices.
* **The M87 film** follows the revised Sec. 2.2: `v_φ/v_esc = tan θ₀ (1+ε)`, `ε = ε∞(1 − r₀/r)`, with
  θ₀ = 6°, r₀ = 15 r_g and ε∞ = 0.05 (calibrated); the camera moves in log radius.
* **The riverbed film** uses schematic units (r_c = 1, time in 1/H). The absorption rate Q is a display
  choice, and the speed of light is slowed so the influence front is visible.

## Fitting videos: conventions

The fits work in v² with σ(v²) = 2V·σ_V, H_z fixed per galaxy as in the paper, and report masses with
M⊙ = 1.989×10³⁰ kg (`clfit.py` uses the workbook's mass unit of 2×10⁴⁰ kg for 10¹⁰ M⊙ internally).

| Galaxy | Result reproduced by the prep script |
|---|---|
| UGC 8286 | R = 1.117 kpc, M = 6.673×10⁸ M⊙, χ² = 2.299 |
| UGC 1281 | R = 1.964 kpc, M = 6.912×10⁸ M⊙, χ² = 0.617 |
| IC 2574 (two-L) | ‘<’ convention (first curve for r < R₂): global minimum R₂ = 5.97 kpc (χ² = 38.89); the paper's solution at R₂ = 6.26 kpc (χ² = 39.65) lies in the neighbouring step of the R₂ profile |
| ESO079–G014 | single-L χ² = 14.52; with Φ_BH = 1590 (km/s)²: χ² = 4.41 |

## Reproducing the AIC/BIC comparison from the paper

`data/curated.json` is included. To rebuild it from the text of the SPARC fits paper:

```
pdftotext -layout paper.pdf paper_text.txt
python parse_fits.py paper_text.txt     # -> data/fit_rows.json
python curate_fits.py                   # -> data/curated.json and two CSV files
```

The curation rules (later table version for NGC 3741 and UGC 6446; F563–V2 and UGC 2953 excluded;
UGC 1281 NFW row entered by hand) are listed in `curate_fits.py` and shown in the video.

## Data and references

* Rotation curves: SPARC — F. Lelli, S.S. McGaugh, J.M. Schombert, *AJ* 152, 157 (2016).
* E.P.J. de Haas, *J. High Energy Phys. Gravit. Cosmol.* 11 (2025), doi:10.4236/jhepgc.2025.114097
* E.P.J. de Haas, *J. High Energy Phys. Gravit. Cosmol.* 12 (2026), doi:10.4236/jhepgc.2026.121022
* E.P.J. de Haas (2025), Jet collimation, helical geometry and UHECR acceleration from a single PG metric, doi:10.5281/zenodo.17681340
* M87 observational context: K. Hada et al., *Astron. Astrophys. Rev.* 32, 5 (2024), doi:10.1007/s00159-024-00155-y

## Licence

MIT — see `LICENSE`.
