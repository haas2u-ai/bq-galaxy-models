"""Leave-one-out prediction of rotation curves from photometry alone, via the two scaling relations.
CL:     log vL = a1 + b1 log Mbar ;  log s = a2 + b2 log SB_inner ;  R = vL/(sqrt3 s) ; then M = vL^2 R/(3G) (H_z term kept exact)
arctan: log Vc = ...          ;  log s = ...                ;  rt = 2Vc/(pi s)
Writes ../data/prediction.json and per-galaxy rms."""
import json, numpy as np, pandas as pd
from scipy.optimize import brentq
from cl_lelli_analysis import D, G, HZ, v2_cl, v_atan, load_massmodels, bisector
df = pd.read_csv(D/'galaxies.csv'); MM = load_massmodels()
S = df[(df.Q <= 2) & (df.Inc >= 30) & (~df.unconstrained)].reset_index(drop=True)
L = np.log10
def fitline(x, y):            # bisector line through the medians (symmetric, as in the paper)
    b = bisector(x, y); return np.median(y) - b*np.median(x), b
def M_from(vL, R):            # invert vL = sqrt(1.5)(sqrt(2GM/R) - H R) exactly
    return (vL/np.sqrt(1.5) + HZ*R)**2*R/(2*G)
rows = []
for i, g in S.iterrows():
    tr = S.drop(i)
    a1, b1 = fitline(L(tr.Mbar), L(tr.vL)); a2, b2 = fitline(L(tr.SB_inner), L(tr.s_CL))
    c1, d1 = fitline(L(tr.Mbar), L(tr.Vc)); c2, d2 = fitline(L(tr.SB_inner), L(tr.s_atan))
    vL = 10**(a1 + b1*L(g.Mbar)); s = 10**(a2 + b2*L(g.SB_inner)); R = vL/(np.sqrt(3)*s); M = M_from(vL, R)
    Vc = 10**(c1 + d1*L(g.Mbar)); sa = 10**(c2 + d2*L(g.SB_inner)); rt = 2*Vc/(np.pi*sa)
    d = MM[MM.ID == g.Galaxy].sort_values('R'); r, v = d.R.values, d.Vobs.values
    vcl = np.sqrt(np.maximum(v2_cl(r, M, R), 0)); vat = v_atan(r, Vc, rt)
    vfit_cl = np.sqrt(np.maximum(v2_cl(r, 10**g.logM, 10**g.logR), 0)); vfit_at = v_atan(r, g.Vc, g.rt)
    rows.append(dict(Galaxy=g.Galaxy, rmsfit_CL=float(np.sqrt(np.mean(L(np.maximum(vfit_cl, 1e-3)/v)**2))), rmsfit_atan=float(np.sqrt(np.mean(L(np.maximum(vfit_at, 1e-3)/v)**2))), rms_CL=float(np.sqrt(np.mean(L(vcl/v)**2))), rms_atan=float(np.sqrt(np.mean(L(np.maximum(vat, 1e-3)/v)**2))),
                     dlogR=float(L(R) - g.logR), dlogvL=float(L(vL/g.vL)), dlogrt=float(L(rt/g.rt))))
P = pd.DataFrame(rows); P.to_csv(D/'prediction_per_galaxy.csv', index=False, float_format='%.5g')
out = dict(n=len(P), median_rms_CL_dex=float(P.rms_CL.median()), median_rms_atan_dex=float(P.rms_atan.median()),
           frac_CL_better=float((P.rms_CL < P.rms_atan).mean()), scatter_dlogR=float(P.dlogR.std()), scatter_dlogrt=float(P.dlogrt.std()),
           scatter_dlogvL=float(P.dlogvL.std()), median_rmsfit_CL_dex=float(P.rmsfit_CL.median()), median_rmsfit_atan_dex=float(P.rmsfit_atan.median()), median_frac_err_CL=float(10**P.rms_CL.median()-1), median_frac_err_atan=float(10**P.rms_atan.median()-1))
json.dump(out, open(D/'prediction.json', 'w'), indent=1); print(json.dumps(out, indent=1))
