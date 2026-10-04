"""Single-L solver path, R2 profile scan and best fits for the IC 2574 two-L video (movie2L.py).
Uses the '<' convention: first CL curve for r < R2, second from R2 on. Takes about a minute."""
from paths import cache
import json, numpy as np
from scipy.optimize import least_squares
from model2574 import *
from lm import lm_path
c = MUNIT/1.989e30/1e9
json.dump(lm_path(lambda q: res1([q[0], 10**q[1]]), [1.6, np.log10(0.02)], tol=1e-8), open(cache("ic2574_lm_single.json"), "w"))
prof = []; prev = None
for R2 in np.round(np.arange(4.50, 7.501, 0.01), 3):
    best = None
    for s0 in [(2.5, 0.05, 0.3), (1.8, 0.03, 0.2), (3.2, 0.07, 0.4)] + ([tuple(prev)] if prev is not None else []):
        try: s = least_squares(lambda q: res2lt([q[0], q[1], R2, q[2]]), s0, bounds=([0.2, 1e-4, 1e-3], [R2, 1.0, 5.0]))
        except Exception: continue
        if best is None or s.cost < best.cost: best = s
    prev = best.x; prof.append([R2, 2*best.cost, *best.x])
prof = np.array(prof); np.save(cache("ic2574_profile.npy"), prof)
i = np.argmin(prof[:, 1]); j = np.argmin(np.abs(prof[:, 0] - 6.26))
def refine(k, R2):
    s = least_squares(lambda q: res2lt([q[0], q[1], R2, q[2]]), prof[k, 2:], method="lm"); return s
s = refine(i, prof[i, 0]); n = len(r); k4 = 4; chi2 = 2*s.cost
cov = np.linalg.inv(s.jac.T@s.jac); sd = np.sqrt(np.diag(cov))
rms = np.sqrt(np.mean(((v2 - two_lt(r, s.x[0], s.x[1], prof[i, 0], s.x[2]))/v2)**2))
sp = refine(j, 6.26)
out = dict(R1=s.x[0], sR1=sd[0], M1=s.x[1]*c, sM1=sd[1]*c, R2=prof[i, 0], M2=s.x[2]*c, sM2=sd[2]*c, chi2=chi2,
           chi2nu=chi2/(n-k4), AIC=chi2+2*k4, BIC=chi2+k4*np.log(n), RMS=rms, M1_model=s.x[1], M2_model=s.x[2],
           paper_valley=dict(R2=6.26, R1=sp.x[0], M1=sp.x[1]*c, M2=sp.x[2]*c, chi2=2*sp.cost, M1_model=sp.x[1], M2_model=sp.x[2]))
json.dump(out, open(cache("ic2574_best.json"), "w"), indent=1)
print(f"global minimum R2 = {out['R2']:.2f} kpc, chi2 = {chi2:.3f};  paper valley R2 = 6.26, chi2 = {out['paper_valley']['chi2']:.3f}")
