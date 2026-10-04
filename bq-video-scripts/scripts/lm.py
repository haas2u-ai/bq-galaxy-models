"""Transparent Levenberg-Marquardt in (R, log10 M) that records every accepted step (used by the prep scripts)."""
import numpy as np
def lm_path(res_q, q0, lam=1.0, tol=1e-9, max_it=100):
    q = np.array(q0, float); n = len(res_q(q))
    path = [[q[0], 10**q[1], float(np.sum(res_q(q)**2))]]
    for _ in range(max_it):
        f0 = res_q(q); J = np.zeros((n, len(q)))
        for j in range(len(q)):
            dq = np.zeros(len(q)); dq[j] = 1e-6; J[:, j] = (res_q(q + dq) - f0)/1e-6
        A = J.T@J; g = J.T@f0; chi = np.sum(f0**2)
        while True:
            qn = q - np.linalg.solve(A + lam*np.diag(np.diag(A)), g); chin = np.sum(res_q(qn)**2)
            if np.isfinite(chin) and chin < chi: q = qn; lam = max(lam/3, 1e-7); break
            lam *= 4
            if lam > 1e10: return path
        path.append([q[0], 10**q[1], float(chin)])
        if abs(chi - chin) < tol: break
    return path
