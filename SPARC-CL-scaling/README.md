# SPARC-CL-scaling

Companion material for **"The central velocity gradient–surface brightness relation and the baryonic Tully–Fisher relation in two-parameter constant-Lagrangian fits to SPARC"** (E.P.J. de Haas, 2026).

Archived release: https://doi.org/10.5281/zenodo.23339753. Project overview, CL fit atlases and interactive models: https://haas2u-ai.github.io/bq-galaxy-models/

The paper refits all 175 SPARC galaxies with its own code (single CL curve fitted in v²); its per-galaxy results are in `data/galaxies.csv`, not the earlier atlas fits shown on the project site.

| Folder | Contents |
|---|---|
| `paper/` | LaTeX source, PDF and figures |
| `code/` | `cl_lelli_analysis.py` (CL + arctan fits of all 175 SPARC galaxies, gradients, surface-brightness proxies, all statistics), `cl_photometric_prediction.py` (leave-one-out rotation curves from baryonic quantities), `cl_vspace_robustness.py` (CL refit in velocity space; robustness check), `make_figures.py` |
| `data/` | Original SPARC tables (Lelli, McGaugh & Schombert 2016), per-galaxy results `galaxies.csv`, summary statistics `results.json`, `prediction.json` |

## Reproduce

```bash
cd code
python cl_lelli_analysis.py          # writes data/galaxies.csv and data/results.json
python cl_photometric_prediction.py  # writes data/prediction.json
python cl_vspace_robustness.py       # writes data/robustness_vspace.json
python make_figures.py               # writes paper/fig/*.pdf
```

Requirements: Python 3, NumPy, SciPy, pandas, Matplotlib.

## Main columns of `galaxies.csv`

`logM, e_logM, logR, e_logR, rho_MR` single-L CL fit (M in Msun, R in kpc, H_z = 2.2e-18 s^-1) · `vL` CL plateau speed · `s_CL = vL/(sqrt3 R)` · `Vc, rt, s_atan` arctan fit · `s_data` gradient from the first three points · `SB_inner, SB_extrap, SBd0` 3.6 µm surface-brightness proxies (Lsun/pc²) · `Mbar` baryonic mass (Υd = 0.5, Υb = 0.7) · `r13` radius where V reaches Vflat/√3 · `unconstrained` CL radius at its lower bound.
