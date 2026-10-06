"""Two-Lagrangian (nested bar/inner-spiral + disk) CL inflow fits, de Haas SPARC Hz paper Eqs. (24)-(26).
  r <= R1      : v^2 = 1/2 X1^2 r^2/R1^2                        (bulge, L1)
  R1 < r <= R2 : v^2 = 3/2 X1^2 - (sqrt(2GM1/r) - Hz r)^2        (bar zone, L1)
  r > R2       : v^2 = 3/2 X2^2 - (sqrt(2GM2/r) - Hz r)^2        (disk, L2)
  X_i = sqrt(2GM_i/R_i) - Hz R_i ;  optional shared offset Phi_BH added everywhere."""
import numpy as np
from scipy.optimize import least_squares
from cl_fit import G, HZ, v2_model, load_massmodels, load_table1

def v2_two(r, M1, R1, M2, R2, phi=0.0, Hz=HZ):
    r = np.asarray(r, float); rs = np.maximum(r, 1e-9)
    X1 = np.sqrt(2*G*M1/R1) - Hz*R1; X2 = np.sqrt(2*G*M2/R2) - Hz*R2
    return np.where(r <= R1, 0.5*X1**2*r**2/R1**2,
           np.where(r <= R2, 1.5*X1**2 - (np.sqrt(2*G*M1/rs) - Hz*rs)**2,
                             1.5*X2**2 - (np.sqrt(2*G*M2/rs) - Hz*rs)**2)) + phi

def _metrics(y, s, mod, k):
    n = len(y); chi2 = float(np.sum(((mod-y)/s)**2))
    return dict(n=n, k=k, chi2=chi2, chi2nu=chi2/(n-k), AIC=chi2+2*k, BIC=chi2+k*np.log(n),
                RMSrel=float(np.sqrt(np.mean(((y-mod)/y)**2))))

def _errors(sol, k, dof_scale=False):
    J = sol.jac
    try: cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError: cov = np.full((k,k), np.nan)
    return np.sqrt(np.clip(np.diag(cov), 0, None))

def fit_single(r, v, ev, phi=False):
    y, s = v**2, 2*v*ev
    def mod(p): return v2_model(r, p[0]*1e9, p[1], p[2] if phi else 0.0)
    best = None
    for R0 in np.geomspace(max(r.min(), .05), r.max()*1.2, 14):
        M0 = max(v.max()**2*R0/(3*G), 1e5)/1e9
        p0 = [M0, R0] + ([0.] if phi else [])
        lo = [1e-5, 0.02] + ([-np.inf] if phi else []); hi = [500, 100] + ([np.inf] if phi else [])
        try: sol = least_squares(lambda p: (mod(p)-y)/s, p0, bounds=(lo, hi), x_scale='jac')
        except Exception: continue
        if best is None or sol.cost < best.cost: best = sol
    k = len(best.x); out = _metrics(y, s, mod(best.x), k)
    out.update(params=best.x, errors=_errors(best, k)); return out

def fit_two(r, v, ev, phi=False):
    y, s = v**2, 2*v*ev
    # p = [M1 (1e9), R1, M2 (1e9), R2 (, phi)];  R2 = R1 + d enforced via d>0 reparametrisation
    def unpack(q): return q[0], q[1], q[2], q[1]+q[3], (q[4] if phi else 0.0)
    def mod(q):
        M1, R1, M2, R2, ph = unpack(q); return v2_two(r, M1*1e9, R1, M2*1e9, R2, ph)
    best = None
    grid = np.unique(np.r_[r[:-1], r.max()*0.9])
    cand_R1 = np.geomspace(max(r.min(), .05), r.max()*0.7, 8)
    for R1 in cand_R1:
        for R2 in grid[grid > R1*1.05][::max(1, len(grid)//10)]:
            inner = v[r <= R2].max() if np.any(r <= R2) else v.min()
            M1 = max(inner**2*R1/(3*G), 1e5)/1e9; M2 = max(v.max()**2*R2/(3*G), 1e5)/1e9
            q0 = [M1, R1, M2, R2-R1] + ([0.] if phi else [])
            lo = [1e-5, 0.02, 1e-5, 0.01] + ([-np.inf] if phi else [])
            hi = [500, 100, 5000, 200] + ([np.inf] if phi else [])
            try: sol = least_squares(lambda q: (mod(q)-y)/s, q0, bounds=(lo, hi), x_scale='jac')
            except Exception: continue
            if best is None or sol.cost < best.cost: best = sol
    q = best.x; k = len(q); out = _metrics(y, s, mod(q), k)
    e = _errors(best, k)
    # error on R2 = R1 + d from covariance
    J = best.jac
    try:
        cov = np.linalg.inv(J.T @ J); eR2 = np.sqrt(max(cov[1,1]+cov[3,3]+2*cov[1,3], 0))
    except np.linalg.LinAlgError: eR2 = np.nan
    M1, R1, M2, R2, ph = unpack(q)
    out.update(params=np.array([M1, R1, M2, R2] + ([ph] if phi else [])),
               errors=np.array([e[0], e[1], e[2], eR2] + ([e[4]] if phi else [])))
    return out
