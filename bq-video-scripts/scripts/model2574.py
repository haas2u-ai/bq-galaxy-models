import numpy as np
from clfit import G_SI, KPC, MUNIT
from data2574 import DATA
HZ = 2.27e-18
r, V, eV = DATA.T; v2, e2 = V**2, 2*V*eV
def X_(R, M10): return np.sqrt(2*G_SI*M10*MUNIT/(R*KPC)) - HZ*R*KPC
def vr_(rr, M10): return np.sqrt(2*G_SI*M10*MUNIT/(rr*KPC)) - HZ*rr*KPC
def single(rr, R, M10):
    X = X_(R, M10)
    return 1e-6*np.where(rr <= R, 0.5*X**2*(rr/R)**2, 1.5*X**2 - vr_(rr, M10)**2)
def two(rr, R1, M1, R2, M2):
    """paper: inner (R1,M1) for r <= R2 (bulge inside R1), outer (R2,M2) disk branch for r > R2"""
    return np.where(rr <= R2, single(rr, R1, M1), 1e-6*(1.5*X_(R2, M2)**2 - vr_(rr, M2)**2))
def res1(p): return (v2 - single(r, *p))/e2
def res2(p): return (v2 - two(r, *p))/e2

def two_lt(rr, R1, M1, R2, M2):
    """'<' convention: first CL curve (R1,M1) for r < R2; reset at R2 -> second CL curve (bulge R2, mass M2) for r >= R2"""
    return np.where(rr < R2, single(rr, R1, M1), 1e-6*(1.5*X_(R2, M2)**2 - vr_(rr, M2)**2))
def res2lt(p): return (v2 - two_lt(r, *p))/e2
