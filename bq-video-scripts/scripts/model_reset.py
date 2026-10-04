import numpy as np
from scipy.optimize import brentq
from model import G, KMS_TO_KPC_PER_MYR
from model2L import velocities_2L

H0, Om, OL = 0.07, 0.3, 0.7                 # km/s/kpc, flat LCDM
H0_GYR = H0*1.02271                         # 1/Gyr
_k = 1.5*H0_GYR*np.sqrt(OL)
def H_of_t(t):  return H0*np.sqrt(OL)/np.tanh(_k*t)                        # km/s/kpc
def z_of_t(t):  return 1/((Om/OL)**(1/3)*np.sinh(_k*t)**(2/3)) - 1
T0 = brentq(lambda t: z_of_t(t), 1, 20)

# NGC 3741 nested-spiral fit (jhepgc 2026, Sec. 3): bulge 7.36e37 kg @1.543e19 m, bar 1.09e39 kg @7.129e19 m
MSUN, KPC = 1.989e30, 3.0857e19
R1, M1_0 = 1.543e19/KPC, 7.36e37/MSUN
R2, M2_0 = 7.129e19/KPC, 1.09e39/MSUN
ALPHA = 1.5                                   # Eq. (20)
def M1(t): return M1_0*(t/T0)**ALPHA          # anchored today => mass closure exact
def M2(t): return M2_0*(t/T0)**ALPHA
def rc(M, H): return (2*G*M/H**2)**(1/3)
def rc1(t): return rc(M1(t), H_of_t(t))
def rc2(t): return rc(M2(t), H_of_t(t))

T_ON = brentq(lambda t: rc1(t) - R2, 0.05, T0)          # Eq. (19): r_c,1(t_on) = R2
T_START = brentq(lambda t: rc1(t) - 1.0, 0.01, T_ON)    # movie starts when proto-spiral reaches 1 kpc

def velocities_t(r, t):
    """Single L1 before reset (Eqs. 24-25 everywhere beyond R1); two-L (Eqs. 24-26) after."""
    H = H_of_t(t)
    if t < T_ON:
        return velocities_2L(r, R1, M1(t), 1e9, 1.0, H)   # R2 -> infinity: L1 only
    return velocities_2L(r, R1, M1(t), R2, M2(t), H)

if __name__ == "__main__":
    print(f"R1={R1:.3f} kpc M1_0={M1_0:.3e}  R2={R2:.3f} kpc M2_0={M2_0:.3e}  T0={T0:.2f} Gyr")
    print(f"onset: t_on={T_ON:.3f} Gyr z_on={z_of_t(T_ON):.2f} H_on={H_of_t(T_ON):.4f} km/s/kpc"
          f"  M1(t_on)={M1(T_ON):.3e} (f={M1(T_ON)/M1_0:.4f})  M2(t_on)={M2(T_ON):.3e} rc2(t_on)={rc2(T_ON):.2f}")
    print(f"start: t={T_START:.3f} Gyr z={z_of_t(T_START):.2f}")
    print("today: rc1=%.1f rc2=%.1f H=%.4f"%(rc1(T0), rc2(T0), H_of_t(T0)))
