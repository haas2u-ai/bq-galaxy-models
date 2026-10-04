"""Solver path for the UGC 1281 single-L fit video (fit_movie_1281.py). Run once before fit_movie_1281.py."""
from paths import cache
import json, numpy as np
import clfit
from data1281 import DATA
from lm import lm_path
r, V, eV = DATA.T; v2, e2 = V**2, 2*V*eV; HZ = 2.30e-18
res = lambda p: (v2 - clfit.model_v2(r, p[0], p[1], HZ))/e2
path = lm_path(lambda q: res([q[0], 10**q[1]]), [0.7, np.log10(0.25)], tol=1e-10)
json.dump([dict(R=p[0], M10=p[1], chi2=p[2]) for p in path], open(cache("lm_path_ugc1281.json"), "w"), indent=1)
print(f"{len(path)-1} accepted steps; final R = {path[-1][0]:.4f} kpc, chi2 = {path[-1][2]:.4f}")
