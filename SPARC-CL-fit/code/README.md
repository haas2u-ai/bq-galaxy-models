# Code

Python 3 scripts that reproduce every fit, figure and report. They need `numpy`, `pandas`, `scipy` and `matplotlib`; building the PDFs also needs `pdflatex`.

The SPARC input files must be available, either in `../data/` or in the folder named by the environment variable `SPARC_DATA`:

- `MassModels_Lelli2016c.mrt`
- `Table1wbulge.mrt`

Every step writes its results (`*.csv`, `*.pkl`) into this folder, and later steps read them. The cached results of the published run are included, so any step can be rerun on its own.

## Run order

```bash
cd SPARC-CL-fit/code
python pipeline.py && python pipeline_plots.py && python make_report.py
python catA.py && python catA_plots.py && python catA_report.py
python catB.py && python catB_profile.py && python catB_fix.py
python catB.py && python catB_profile.py
python catB_plots.py && python catB_report.py
python catC.py && python catC_post.py
python catC_plots.py && python catC_report.py
python catD.py && python catD_post.py && python catD_status.py
python catD_plots.py && python catD_report.py
python catE.py && python catE_post.py
python catE_plots.py && python catE_report.py
python overall_build.py && python overall_plots.py
python overall_report.py
```

The `*_report.py` scripts write LaTeX sources; run `pdflatex` twice on each. Figures and atlases are written to this folder; the published versions are in the report folders.

`catB_fix.py` adopts better two-L optima found by the profile scan, so B's fit and profile steps are run twice.

## Modules

| File | Content |
|---|---|
| `cl_fit.py` | Single-L CL model, H_z = 2.2e-18 s⁻¹, SPARC readers |
| `cl2_fit.py` | Two-Lagrangian model and fitter |
| `wave_detect.py` | Runs test and +−+ / −+− template scores |
| `descent.py` | Virial-window model (continuous gauge) |
| `catC_model.py` | Virial window in the (M, R, r_v, M_N) parametrisation |
| `catD.py` | Model set S, VW, 2L, VW2, 2LVW with a common fitter |
