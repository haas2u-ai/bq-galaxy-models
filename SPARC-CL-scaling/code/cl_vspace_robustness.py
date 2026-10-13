"""Robustness check: refit the CL model in velocity space (V, dV) instead of
v^2 space (V^2, 2V dV), and repeat the main gradient and BTFR statistics.
Run after cl_lelli_analysis.py. Writes data/robustness_vspace.json."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from cl_lelli_analysis import load_massmodels, v2_cl, G, HZ, slopes, cov_errors, D


def fit_cl_v(r, v, ev):
    res = lambda p: (np.sqrt(np.maximum(v2_cl(r, 10**p[0], 10**p[1]), 0)) - v) / ev
    best = None
    for R0 in np.geomspace(max(r.min(), 0.05), r.max()*1.5, 12):
        M0 = max(v.max()**2*R0/(3*G), 1e5)
        try:
            sol = least_squares(res, [np.log10(M0), np.log10(R0)],
                                bounds=([4, -2], [13.5, np.log10(200)]), x_scale='jac')
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    return best


def main():
    MM = load_massmodels()
    df = pd.read_csv(D / 'galaxies.csv')
    rows = []
    for _, g in df.iterrows():
        d = MM[MM.ID == g.Galaxy].sort_values('R')
        r, v, ev = d.R.values, d.Vobs.values, d.eV.values
        s = fit_cl_v(r, v, ev)
        lM, lR = s.x
        slM, slR, _ = cov_errors(s, len(r))
        M, R = 10**lM, 10**lR
        X = np.sqrt(2*G*M/R) - HZ*R
        vL = np.sqrt(1.5)*X
        chi = np.sum(((np.sqrt(np.maximum(v2_cl(r, M, R), 0)) - v)/ev)**2)/max(len(r)-2, 1)
        rows.append(dict(Galaxy=g.Galaxy, logM_v=lM, logR_v=lR,
                         unc_v=bool(slM > 0.5 or lR < -1.5),
                         vL_v=vL, s_v=vL/(np.sqrt(3)*R), chi_v=chi))
    V = pd.DataFrame(rows)
    A = df.merge(V, on='Galaxy')
    S = A[(A.Q <= 2) & (A.Inc >= 30) & (~A.unconstrained) & (~A.unc_v)]
    L = np.log10
    out = dict(n=int(len(S)), n_unconstrained_vfit=int(A.unc_v.sum()),
               dlogR_median=float(np.median(S.logR_v - S.logR)),
               dlogR_scatter=float(np.std(S.logR_v - S.logR)),
               s_ratio_median=float(np.median(S.s_v/S.s_CL)),
               s_ratio_scatter_dex=float(np.std(L(S.s_v/S.s_CL))))
    for k, col in [('v2fit', 's_CL'), ('Vfit', 's_v')]:
        st = slopes(L(S.SB_inner), L(S[col]))
        out[k] = dict(rho=st['rho'], ols=st['ols'], inverse=st['inverse'],
                      bisector=st['bisector'], bis_lo=st['bis_lo'], bis_hi=st['bis_hi'],
                      scatter=st['scatter_bis'],
                      s_over_sdata=float(np.median(S[col]/S.s_data)),
                      rho_with_sdata=float(spearmanr(S[col], S.s_data)[0]))
    out['chi2nu_median'] = dict(CL_v2fit=float(S.chi2nu_CL.median()),
                                CL_Vfit=float(S.chi_v.median()),
                                atan=float(S.chi2nu_atan.median()),
                                frac_CLV_better=float((S.chi_v < S.chi2nu_atan).mean()))
    f = S[S.Vflat > 0]
    for k, col in [('btfr_v2fit', f.vL), ('btfr_Vfit', f.vL_v)]:
        b, a = np.polyfit(L(col), L(f.Mbar), 1)
        out[k] = dict(n=int(len(f)), slope=float(b),
                      scatter=float(np.std(L(f.Mbar) - (a + b*L(col)), ddof=2)),
                      vL_over_Vf=float(np.median(col/f.Vflat)))
    (D / 'robustness_vspace.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
