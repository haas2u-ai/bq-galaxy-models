"""Single-L solver path, Phi_BH profile, best fits and (R, Phi_BH) landscape for the ESO079-G014 video (movie_eso.py)."""
from paths import cache
import json, numpy as np
from scipy.optimize import least_squares
from model_eso import *
from lm import lm_path
json.dump(lm_path(lambda q: res([q[0], 10**q[1]]), [1.5, np.log10(0.4)]), open(cache("eso_lm_single.json"), "w"))
prof = []; x0 = [5.1, 1.58]
for phi in np.arange(0, 3601, 20.0):
    s = least_squares(lambda p: res([p[0], p[1], phi]), x0, method="lm"); x0 = s.x
    prof.append([phi, float(np.sum(s.fun**2)), *s.x])
np.save(cache("eso_phi_profile.npy"), np.array(prof))
def pack(s, k):
    chi = float(np.sum(s.fun**2)); cov = np.linalg.inv(s.jac.T@s.jac); sd = np.sqrt(np.diag(cov))
    rms = float(np.sqrt(np.mean(((v2 - model(r, *s.x))/v2)**2)))
    return dict(x=list(s.x), sd=list(sd), chi2=chi, chi2nu=chi/(n-k), AIC=chi+2*k, BIC=chi+k*np.log(n), RMS=rms,
                corr=list(map(list, cov/np.outer(sd, sd))))
best = dict(single=pack(least_squares(res, [5, 1.5], method="lm"), 2), phi=pack(least_squares(res, [5, 1.5, 1000], method="lm"), 3))
json.dump(best, open(cache("eso_best.json"), "w"), indent=1)
Rg = np.linspace(4.2, 7.2, 61); Pg = np.linspace(-500, 3700, 61); L = np.zeros((len(Pg), len(Rg)))
for a, phi in enumerate(Pg):
    m0 = 1.6
    for b, R in enumerate(Rg):
        s = least_squares(lambda p: res([R, p[0], phi]), [m0], method="lm"); m0 = s.x[0]; L[a, b] = np.sum(s.fun**2)
np.savez(cache("eso_R_phi_landscape.npz"), Rg=Rg, Pg=Pg, L=L)
print(f"single-L chi2 = {best['single']['chi2']:.3f};  with Phi_BH: Phi = {best['phi']['x'][2]:.0f}, chi2 = {best['phi']['chi2']:.3f}")
