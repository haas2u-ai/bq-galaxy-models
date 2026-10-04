import numpy as np
from scipy.optimize import least_squares
# raw SPARC data from the workbook (columns C, D, E, rows 32-48)
DATA = np.array([(0.47,22.2,2.81),(0.95,43.4,3.47),(1.42,60.2,3.04),(1.9,69.6,2.05),(2.37,72.5,2),(2.84,74.3,2.02),
 (3.3,75.8,2.01),(3.78,77.2,2.05),(4.25,79.3,2.08),(4.73,81.3,2.08),(5.2,81.6,2.03),(5.67,82.1,2.02),(6.15,82.9,2),
 (6.62,83.2,2),(7.1,83.5,2.02),(7.57,83.8,2),(8.04,84.3,2)])
r, V, eV = DATA.T
v2, e2 = V**2, 2*V*eV                      # workbook: G = V^2, H = 2 V err
# workbook constants
G_SI = 6.674e-11; KPC = 3.086e19; MUNIT = 2e40   # 1e10 Msun with Msun = 2e30 kg (workbook D11)
HZ_PAPER = 2.1856e-18
def model_v2(rr, R, M10, Hz=HZ_PAPER, phi=0.0):
    """single-L CL model in (km/s)^2, R in kpc, M in 1e10 Msun (workbook I-column, no virial window)"""
    X = np.sqrt(2*G_SI*M10*MUNIT/(R*KPC)) - Hz*R*KPC              # m/s
    inside = 0.5*X**2*(rr/R)**2
    vr = np.sqrt(2*G_SI*M10*MUNIT/(rr*KPC)) - Hz*rr*KPC
    outside = 1.5*X**2 - vr**2
    return phi + 1e-6*np.where(rr <= R, inside, outside)
def resid(p, Hz=HZ_PAPER): return (v2 - model_v2(r, p[0], p[1], Hz))/e2
