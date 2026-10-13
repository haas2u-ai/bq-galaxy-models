"""CL rotation-curve parameters and two SPARC scaling relations.
Reproduces every number and figure input of the paper.
Usage: python cl_lelli_analysis.py   (reads ../data/*.mrt, writes ../data/galaxies.csv and ../data/results.json)
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from scipy.stats import spearmanr

G   = 4.30091e-6                    # kpc (km/s)^2 / Msun
HZ  = 2.2e-18 * 3.085677581e16      # 2.2e-18 s^-1 in km/s/kpc (0.0679)
UPS_D, UPS_B, X_HE = 0.5, 0.7, 1.33 # SPARC 3.6 um mass-to-light ratios and helium factor
D = Path(__file__).resolve().parent.parent / 'data'

# ---------------- data ----------------
def load_table1():
    names = ['Galaxy','T','D','eD','fD','Inc','eInc','L36','eL','Reff','SBeff','Rd','SBd','MHI','RHI','Vf','eVf','Q','Ref','Lb']
    rows = []
    for L in open(D/'Table1wbulge.mrt'):
        t = L.split()
        if len(t) != len(names): continue
        try: int(t[1]); float(t[2])
        except ValueError: continue
        rows.append(t)
    df = pd.DataFrame(rows, columns=names)
    for c in names:
        if c not in ('Galaxy','Ref'): df[c] = pd.to_numeric(df[c])
    return df.set_index('Galaxy')

def load_massmodels():
    rows = []
    for L in open(D/'MassModels_Lelli2016c.mrt'):
        t = L.split()
        if len(t) != 10: continue
        try: [float(x) for x in t[1:]]
        except ValueError: continue
        rows.append(t)
    df = pd.DataFrame(rows, columns=['ID','D','R','Vobs','eV','Vgas','Vdisk','Vbul','SBd','SBb'])
    for c in df.columns[1:]: df[c] = pd.to_numeric(df[c])
    return df

# ---------------- models ----------------
def v2_cl(r, M, R, Hz=HZ):
    """single-Lagrangian CL model (de Haas 2018; 2025b): v^2 inside and outside R"""
    X = np.sqrt(2*G*M/R) - Hz*R
    return np.where(r <= R, 0.5*X**2*r**2/R**2, 1.5*X**2 - (np.sqrt(2*G*M/r) - Hz*r)**2)

def v_atan(r, Vc, rt):
    """Courteau (1997) arctan rotation curve"""
    return 2/np.pi*Vc*np.arctan(r/rt)

def fit_cl(r, v, ev):
    y, s = v**2, 2*v*ev
    res = lambda p: (v2_cl(r, 10**p[0], 10**p[1]) - y)/s
    best = None
    for R0 in np.geomspace(max(r.min(), 0.05), r.max()*1.5, 12):
        M0 = max(v.max()**2*R0/(3*G), 1e5)
        try: sol = least_squares(res, [np.log10(M0), np.log10(R0)], bounds=([4, -2], [13.5, np.log10(200)]), x_scale='jac')
        except Exception: continue
        if best is None or sol.cost < best.cost: best = sol
    return best

def fit_atan(r, v, ev):
    res = lambda p: (v_atan(r, 10**p[0], 10**p[1]) - v)/ev
    best = None
    for rt0 in np.geomspace(max(r.min(), 0.02), r.max(), 10):
        try: sol = least_squares(res, [np.log10(v.max()*1.1), np.log10(rt0)], bounds=([0, -3], [3.5, 3]), x_scale='jac')
        except Exception: continue
        if best is None or sol.cost < best.cost: best = sol
    return best

def cov_errors(sol, n, k=2):
    J = sol.jac; chi2 = 2*sol.cost; dof = max(n-k, 1)
    try:
        cov = np.linalg.inv(J.T@J)
        if chi2/dof > 1: cov *= chi2/dof
        e = np.sqrt(np.diag(cov)); return e[0], e[1], cov[0,1]/(e[0]*e[1])
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

# ---------------- statistics ----------------
def bisector(x, y):
    b1 = np.polyfit(x, y, 1)[0]; b2 = 1/np.polyfit(y, x, 1)[0]
    return (b1*b2 - 1 + np.sqrt((1+b1**2)*(1+b2**2)))/(b1+b2)

def slopes(x, y, nboot=1000, seed=1):
    x, y = np.asarray(x), np.asarray(y); rng = np.random.default_rng(seed)
    ols = np.polyfit(x, y, 1)[0]; inv = 1/np.polyfit(y, x, 1)[0]; bis = bisector(x, y)
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, len(x), len(x)); bs.append(bisector(x[i], y[i]))
    a_bis = np.median(y) - bis*np.median(x)
    sc = np.std(y - (a_bis + bis*x), ddof=2)
    return dict(n=len(x), ols=ols, inverse=inv, bisector=bis, bis_lo=np.percentile(bs, 2.5), bis_hi=np.percentile(bs, 97.5),
                rho=spearmanr(x, y)[0], scatter_bis=sc, intercept_bis=a_bis)

def partial_spearman(x, y, z):
    rx, ry, rz = [pd.Series(v).rank().values for v in (x, y, z)]
    a, b, c = np.corrcoef(rx, ry)[0,1], np.corrcoef(rx, rz)[0,1], np.corrcoef(ry, rz)[0,1]
    return (a - b*c)/np.sqrt((1-b*b)*(1-c*c))

# ---------------- main ----------------
def main():
    T1, MM = load_table1(), load_massmodels()
    rows = []
    for g, t in T1.iterrows():
        d = MM[MM.ID == g].sort_values('R'); r, v, ev = d.R.values, d.Vobs.values, d.eV.values
        n = len(r)
        c = fit_cl(r, v, ev); lM, lR = c.x; slM, slR, rho = cov_errors(c, n)
        a = fit_atan(r, v, ev); lVc, lrt = a.x; slVc, slrt, _ = cov_errors(a, n)
        M, R, Vc, rt = 10**lM, 10**lR, 10**lVc, 10**lrt
        X = np.sqrt(2*G*M/R) - HZ*R; vL = np.sqrt(1.5)*X
        chi_cl = np.sum(((np.sqrt(np.maximum(v2_cl(r, M, R), 0)) - v)/ev)**2)/max(n-2, 1)
        chi_at = np.sum(((v_atan(r, Vc, rt) - v)/ev)**2)/max(n-2, 1)
        # data gradient: least-squares slope through the origin of the first three points
        s_dat = np.sum(v[:3]*r[:3])/np.sum(r[:3]**2)
        # central surface brightness proxies (3.6 um, Lsun/pc^2)
        SB = d.SBd.values + d.SBb.values
        SB_in = SB[0]
        if n >= 2 and SB[0] > 0 and SB[1] > 0:
            slope = (np.log10(SB[1]) - np.log10(SB[0]))/(r[1]-r[0]); SB_ex = 10**(np.log10(SB[0]) - slope*r[0])
            SB_ex = max(SB_ex, SB_in)     # never below the innermost observed value
        else: SB_ex = SB_in
        # model-free one-third radius: first r where v >= Vflat/sqrt(3)
        r13 = np.nan
        if t.Vf > 0:
            i = np.where(v >= t.Vf/np.sqrt(3))[0]
            if len(i) and i[0] > 0:
                j = i[0]; r13 = r[j-1] + (t.Vf/np.sqrt(3) - v[j-1])*(r[j]-r[j-1])/(v[j]-v[j-1])
        Ms = (UPS_D*(t.L36 - t.Lb) + UPS_B*t.Lb)*1e9; Mb = Ms + X_HE*t.MHI*1e9
        rows.append(dict(Galaxy=g, T=t['T'], Q=t.Q, D=t.D, Inc=t.Inc, n=n, rmin=r[0], rmax=r[-1],
            logM=lM, e_logM=slM, logR=lR, e_logR=slR, rho_MR=rho, unconstrained=bool(slM > 0.5 or lR < -1.5),
            vL=vL, s_CL=vL/(np.sqrt(3)*R), chi2nu_CL=chi_cl,
            Vc=Vc, rt=rt, e_logrt=slrt, s_atan=2*Vc/(np.pi*rt), chi2nu_atan=chi_at,
            s_data=s_dat, SB_inner=SB_in, SB_extrap=SB_ex, SBd0=t.SBd, Rd=t.Rd, Reff=t.Reff, Vflat=t.Vf,
            Lb=t.Lb, Mstar=Ms, Mbar=Mb, r13=r13))
    df = pd.DataFrame(rows); df.to_csv(D/'galaxies.csv', index=False, float_format='%.6g')

    # ---------------- sample and statistics ----------------
    S = df[(df.Q <= 2) & (df.Inc >= 30) & (~df.unconstrained)].copy()
    A = df[(df.Q <= 2) & (df.Inc >= 30)].copy()                    # incl. unconstrained (arctan / data only)
    out = dict(N_total=len(df), N_main=len(S), N_unconstrained=int(df.unconstrained.sum()),
               N_main_with_bulge=int((S.Lb > 0).sum()), N_all_with_bulge=int((df.Lb > 0).sum()))
    L = np.log10
    out['identity_s2_over_GM_R3'] = float(np.median(S.s_CL**2/(G*10**S.logM/10**(3*S.logR))))
    out['identity_M_over_vL2R_3G'] = float(np.median(10**S.logM/(S.vL**2*10**S.logR/(3*G))))
    rel = {}
    for gname, col in [('s_CL', 's_CL'), ('s_atan', 's_atan'), ('s_data', 's_data')]:
        for sbname in ['SB_inner', 'SB_extrap', 'SBd0']:
            rel[f'{gname}|{sbname}'] = slopes(L(S[sbname]), L(S[col]))
    out['gradient_SB'] = rel
    out['gradient_SB_withUnconstrained_atan'] = slopes(L(A.SB_inner), L(A.s_atan))
    out['gradient_SB_withUnconstrained_data'] = slopes(L(A.SB_inner), L(A.s_data))
    m = S.Vflat > 0
    out['partial_given_Vflat'] = {k: partial_spearman(L(S.SB_inner[m]), L(S[k][m]), L(S.Vflat[m])) for k in ['s_CL', 's_atan', 's_data']}
    out['grad_agreement'] = {
        'CL_vs_data_rho': spearmanr(S.s_CL, S.s_data)[0], 'CL_vs_data_median_ratio': float(np.median(S.s_CL/S.s_data)),
        'atan_vs_data_rho': spearmanr(S.s_atan, S.s_data)[0], 'atan_vs_data_median_ratio': float(np.median(S.s_atan/S.s_data)),
        'CL_vs_atan_rho': spearmanr(S.s_CL, S.s_atan)[0], 'CL_vs_atan_median_ratio': float(np.median(S.s_CL/S.s_atan))}
    # residuals of the gradient-SB relation: does the CL radius carry extra information?
    fitc = rel['s_CL|SB_inner']; res_cl = L(S.s_CL) - (fitc['intercept_bis'] + fitc['bisector']*L(S.SB_inner))
    fita = rel['s_atan|SB_inner']; res_at = L(S.s_atan) - (fita['intercept_bis'] + fita['bisector']*L(S.SB_inner))
    out['residual_rho_CL_vs_atan'] = spearmanr(res_cl, res_at)[0]
    # fit quality
    out['chi2nu'] = {'CL_median': float(S.chi2nu_CL.median()), 'atan_median': float(S.chi2nu_atan.median()),
                     'frac_CL_better': float((S.chi2nu_CL < S.chi2nu_atan).mean())}
    # BTFR
    f = S[S.Vflat > 0]
    def btfr(v):
        b, a = np.polyfit(L(v), L(f.Mbar), 1); return dict(slope=b, scatter=float(np.std(L(f.Mbar) - (a + b*L(v)), ddof=2)))
    out['BTFR'] = {'Vflat': btfr(f.Vflat), 'vL': btfr(f.vL), 'Vc_atan': btfr(f.Vc), 'n': len(f)}
    out['vL_over_Vflat'] = {'median': float(np.median(f.vL/f.Vflat)), 'sd_dex': float(np.std(L(f.vL/f.Vflat)))}
    out['Vc_over_Vflat'] = {'median': float(np.median(f.Vc/f.Vflat)), 'sd_dex': float(np.std(L(f.Vc/f.Vflat)))}
    # R as structural scale
    k = S.r13.notna()
    out['r13'] = {'n': int(k.sum()), 'rho': spearmanr(S.r13[k], 10**S.logR[k])[0], 'median_ratio': float(np.median(S.r13[k]/10**S.logR[k]))}
    out['R_Rd'] = {'median': float(np.median(10**S.logR/S.Rd)), 'rho_R_Rd': spearmanr(10**S.logR, S.Rd)[0],
                   'rho_ratio_T': spearmanr(S['T'], 10**S.logR/S.Rd)[0], 'rho_ratio_SB': spearmanr(S.SB_inner, 10**S.logR/S.Rd)[0]}
    out['rt_Rd'] = {'median': float(np.median(S.rt/S.Rd)), 'rho_rt_Rd': spearmanr(S.rt, S.Rd)[0], 'rho_ratio_T': spearmanr(S['T'], S.rt/S.Rd)[0]}
    # bulge galaxies
    b = S[S.Lb > 0]
    rms_about = lambda key, sb: float(np.sqrt(np.mean((L(b.s_CL) - (rel[key]['intercept_bis'] + rel[key]['bisector']*L(b[sb])))**2)))
    out['bulge'] = {'n': len(b), 'median_dex_SBinner_over_SBd0': float(np.median(L(b.SB_inner/b.SBd0))),
                    'rms_about_SBinner_fit': rms_about('s_CL|SB_inner', 'SB_inner'), 'rms_about_SBd0_fit': rms_about('s_CL|SBd0', 'SBd0')}
    out['unconstrained_at_bound'] = int((df.logR < -1.5).sum())
    f2 = S[S.Vflat > 0]; resid = L(f2.Mbar) - np.polyval(np.polyfit(L(f2.Vflat), L(f2.Mbar), 1), L(f2.Vflat))
    out['rho_BTFRresid_vs_vLoverVf'] = spearmanr(resid, L(f2.vL/f2.Vflat))[0]
    # embedding-condition consequence (reported, not used): g = GM/R^2 vs cH
    cH = 299792.458*HZ
    out['g_over_cH_median_dex'] = float(np.median(L(G*10**S.logM/10**(2*S.logR)/cH)))
    json.dump(out, open(D/'results.json', 'w'), indent=1, default=float)
    return df, S, out

if __name__ == '__main__':
    df, S, out = main()
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4)))
