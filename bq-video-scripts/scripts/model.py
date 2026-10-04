import numpy as np
G = 4.30091e-6          # kpc (km/s)^2 / Msun
H0 = 0.07               # km/s/kpc  (70 km/s/Mpc)
Om, OL = 0.3, 0.7
KMS_TO_KPC_PER_MYR = 1.02271e-3

def Hz(z):  return H0*np.sqrt(Om*(1+z)**3 + OL)

# fiducial mass-growth track: points quoted in the paper (Sec. V.B, V.C)
_zt = np.array([0, 2, 3, 6, 10, 15, 20.])
_Mt = np.array([1.0e10, 4.2e9, 2.6e9, 8.4e8, 3.0e8, 1.0e8, 5.0e7])
def Mz(z):  return 10**np.interp(np.log10(1+z), np.log10(1+_zt), np.log10(_Mt))

def rc(M, H): return (2*G*M/H**2)**(1/3)

def age_Gyr(z):
    H0s = H0/ (3.0857e16) * 1.0      # 1/s  (km/s/kpc -> 1/s)
    t = 2/(3*H0s*np.sqrt(OL))*np.arcsinh(np.sqrt(OL/Om)*(1+z)**-1.5)
    return t/3.156e16

def velocities(r, M, R, H):
    """Paper Eqs. (inside / outside bulge): returns v_orb, v_rad_eff (inward, >0), both km/s"""
    r = np.asarray(r, float)
    X = np.sqrt(2*G*M/R) - H*R
    vorb2 = np.where(r <= R, 0.5*X**2*r**2/R**2,
                     1.5*X**2 - (np.sqrt(2*G*M/np.maximum(r,1e-9)) - H*r)**2)
    vrad  = np.where(r <= R, np.sqrt(G*M/R*(3 - np.minimum(r,R)**2/R**2)) - H*r,
                     np.sqrt(2*G*M/np.maximum(r,1e-9)) - H*r)
    return np.sqrt(np.maximum(vorb2, 0)), vrad

def arm_locus(M, R, H, rmax=20.0, n=1500):
    """streamline of v_L: r dphi/dr = -v_orb/v_rad (trailing, CCW rotation), phi=0 at r=R"""
    rcrit = rc(M, H)
    rout = min(rmax, 0.995*rcrit)
    ro = np.linspace(R, rout, n)
    vo, vr = velocities(ro, M, R, H)
    f = -vo/(ro*vr)
    phio = np.concatenate([[0], np.cumsum(0.5*(f[1:]+f[:-1])*np.diff(ro))])
    ri = np.linspace(R, 0.02, 400)
    vo, vr = velocities(ri, M, R, H)
    f = -vo/(ri*vr)
    phii = np.concatenate([[0], np.cumsum(0.5*(f[1:]+f[:-1])*np.diff(ri))])
    return (ro, phio), (ri, phii)
