"""Solver path for the UGC 8286 single-L fit video (fit_movie.py). Run once before fit_movie.py."""
from paths import cache
import json, numpy as np
from clfit import resid
from lm import lm_path
path = lm_path(lambda q: resid([q[0], 10**q[1]]), [2.6, np.log10(0.25)])
json.dump([dict(R=p[0], M10=p[1], chi2=p[2]) for p in path], open(cache("lm_path_ugc8286.json"), "w"), indent=1)
print(f"{len(path)-1} accepted steps; final R = {path[-1][0]:.4f} kpc, chi2 = {path[-1][2]:.4f}")
