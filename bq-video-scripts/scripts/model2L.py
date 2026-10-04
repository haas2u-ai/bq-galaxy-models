import numpy as np
from model import G, KMS_TO_KPC_PER_MYR

KM_PER_KPC = 3.0857e16
def Hz_from_si(h_si): return h_si*KM_PER_KPC          # s^-1 -> km/s/kpc

def rc(M, H): return (2*G*M/H**2)**(1/3)

def velocities_2L(r, R1, M1, R2, M2, H):
    """Paper Eqs. (24)-(26): r<=R1 bulge (L1), R1<r<=R2 bar zone (L1), r>R2 disk (L2).
       Returns v_orb, v_rad_eff (inward positive), km/s."""
    r = np.asarray(r, float); rs = np.maximum(r, 1e-9)
    X1 = np.sqrt(2*G*M1/R1) - H*R1
    X2 = np.sqrt(2*G*M2/R2) - H*R2
    vorb2 = np.where(r <= R1, 0.5*X1**2*r**2/R1**2,
             np.where(r <= R2, 1.5*X1**2 - (np.sqrt(2*G*M1/rs) - H*rs)**2,
                               1.5*X2**2 - (np.sqrt(2*G*M2/rs) - H*rs)**2))
    vrad = np.where(r <= R1, np.sqrt(G*M1/R1*(3 - np.minimum(r,R1)**2/R1**2)) - H*r,
            np.where(r <= R2, np.sqrt(2*G*M1/rs) - H*rs,
                              np.sqrt(2*G*M2/rs) - H*rs))
    return np.sqrt(np.maximum(vorb2, 0)), vrad

def streamline(R1, M1, R2, M2, H, rmax, n=3000):
    """Continuous streamline r dphi/dr = -v_orb/v_rad, phi=0 at r=R1, integrated per segment
       so the jump at R2 is not smeared."""
    def seg(a, b, m):
        rr = np.linspace(a, b, m)
        vo, vr = velocities_2L(rr, R1, M1, R2, M2, H)
        f = -vo/(rr*vr)
        return rr, np.concatenate([[0], np.cumsum(0.5*(f[1:]+f[:-1])*np.diff(rr))])
    r0, p0 = seg(R1, 0.01, 400)                 # inside bulge (going inward)
    r1, p1 = seg(R1, R2*(1-1e-9), 800)          # bar zone, L1
    r2, p2 = seg(R2*(1+1e-9), rmax, n)          # disk, L2
    p2 = p2 + p1[-1]
    r = np.concatenate([r0[::-1], r1[1:], r2])
    p = np.concatenate([p0[::-1], p1[1:], p2])
    return r, p
