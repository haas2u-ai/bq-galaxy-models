# CL-vacuum-flow

Companion code for the interactive model **CL vacuum flow: Bernoulli and Rayleigh**
(https://haas2u-ai.github.io/bq-galaxy-models/models/CL_Vacuum_Flow.html) and its motion model
(`models/CL_Vacuum_Flow.mp4`, also on YouTube: https://youtu.be/t7D2nkuFlwY).

A rotating bulge of radius R drags space with it. At its border the flow of space is continuous; beyond it the
vacuum continues on its own terms. Three continuations of the same border value, all carried by the radial river
v_r = sqrt(2GM/r), are compared:

| Continuation | Outward law | Bernoulli: L kept | Rayleigh: r v_phi rises | GR vacuum: (r^4 Omega')' = 0 |
|---|---|---|---|---|
| CL | v_r^2 + v_phi^2 = L, so v_phi = sqrt(L - 2GM/r) -> v_L = sqrt(L) | yes | yes | no |
| GR, Lense-Thirring | Omega = Omega(R) (R/r)^3 | no | no | yes |
| Kepler-like | v_phi = v_phi(R) sqrt(R/r) | no | yes | no |

L is the Bernoulli value along a streamline, the constant Lagrangian of the CL papers (L = 3GM/R for v_phi(R) = sqrt(GM/R)).

**The postulate.** The argument rests on one postulate: the vacuum behaves as an ideal, barotropic, non-dissipative
flow, the river and analogue-gravity picture taken as physics rather than as an analogy. For dissipation, matter is
needed, as the source of turbulence: inside R the matter rotates rigidly; outside R the vacuum cannot dissipate the
energy it receives at the border, so the energy is carried along the spiral streamlines. Which column of the table
decides is the postulate.

| Folder | Contents |
|---|---|
| `code/` | `cl_vacuum_flow.py`: exact SymPy checks of every claim (Bernoulli, uniqueness, Rayleigh, GR's momentum constraint, the second-order energy term, continuity at the border) and the four charts of the page |
| `figures/` | the charts written by the script |

The motion model is made by `bq-video-scripts/scripts/cl_vacuum_flow_video.py`.

## Reproduce

```bash
cd code
pip install -r ../requirements.txt
python cl_vacuum_flow.py       # prints the checks and writes ../figures/*.png
```

Slow-flow model; the radial river is taken unchanged by the rotation it carries (no back-reaction). The GR
continuation matches Omega at R but not Omega' (a surface current at R in GR).
