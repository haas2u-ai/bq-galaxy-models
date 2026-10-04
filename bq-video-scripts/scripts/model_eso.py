import numpy as np
import clfit
from data_eso import DATA
HZ = 2.276e-18
r, V, eV = DATA.T; v2, e2 = V**2, 2*V*eV; n = len(r)
C10 = clfit.MUNIT/1.989e30/1e10          # model mass units -> 1e10 Msun (paper)
def model(rr, R, M, phi=0.0): return clfit.model_v2(rr, R, M, HZ, phi)
def res(p): return (v2 - model(r, *p))/e2
