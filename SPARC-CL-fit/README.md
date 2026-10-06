# SPARC-CL-fit

Constant-Lagrangian (CL) inflow fits of all 175 SPARC rotation curves.

The work proceeds in three stages:

- A pipeline sorts the galaxies into five categories (A–E) by the residuals of a single-Lagrangian fit.
- Each category is analysed in its own report.
- An overall report combines the five reports into one classification.

The entrance page is [`../SPARC.html`](../SPARC.html). It is best viewed through GitHub Pages; on github.com the HTML is shown as source.

| Folder | Content |
|---|---|
| [`01_pipeline`](01_pipeline) | Pipeline report, single-L atlas (175 galaxies), first two-L atlas, summary table |
| [`02_category_A_single`](02_category_A_single) | A: single-L adequate (54) |
| [`03_category_B_double`](03_category_B_double) | B: + − + residual wave, two Lagrangians tested (45) |
| [`04_category_C_descent`](04_category_C_descent) | C: − + − residual wave, virial (descent) window tested (41) |
| [`05_category_D_structured`](05_category_D_structured) | D: other residual structure, multi-region models tested (18) |
| [`06_category_E_poor`](06_category_E_poor) | E: poor fit without detected structure (17) |
| [`07_overall`](07_overall) | Overall classification of all 175 galaxies |
| [`code`](code) | Python scripts and cached intermediate results |
| [`data`](data) | Place for the SPARC input files (not included) |
| [`assets`](assets) | Images used by `SPARC.html` |

Each category folder holds four kinds of file:

- the report (PDF) and its LaTeX source;
- the figures the LaTeX source includes;
- an atlas with a residual panel for every galaxy;
- the results table (CSV).

## Overall classification

| Class | Galaxies | Median type |
|---|---|---|
| Single Lagrangian | 100 | Sm |
| Two Lagrangians (reset / nested spiral) | 25 | Sm |
| Single + Keplerian descent window | 24 | Sbc |
| Descent only | 7 | Sb |
| Multi-region | 6 | Sc |
| Flattening window | 2 | Sbc |
| Unresolved (bulge-dominated / non-nested) | 11 | Sab |

## Model references

- E.P.J. de Haas (2026), *J. High Energy Phys. Gravit. Cosmol.* 12, 334–367, [doi:10.4236/jhepgc.2026.121022](https://doi.org/10.4236/jhepgc.2026.121022). This paper gives the virial term and the multi-region model.
- E.P.J. de Haas, *Galactic Rotation Curves and the Constant–Lagrangian Field: Empirical Tests within the Q_g Rotor Framework* (manuscript). This gives the single-L, virial-term and two-L equations.
- Data: F. Lelli, S.S. McGaugh, J.M. Schombert (2016), *AJ* 152, 157 (SPARC).
