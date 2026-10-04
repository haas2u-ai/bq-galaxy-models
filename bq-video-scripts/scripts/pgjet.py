import numpy as np
# M87 PG jet: revised Sec. 2.2 kinematics (all radii in r_g)
TH0 = np.radians(6.0); KAP0 = np.tan(TH0)          # jet cone, kappa = tan(theta0) = 0.1051
THB = np.radians(35.0)                               # broad inflow cone (Fig. 1)
R0, EPSI = 15.0, 0.05                                 # stagnation radius, asymptotic offset
THS = 0.3*TH0                                         # inner spine cone
RG_PC = 3.1e-4                                        # r_g in pc (M = 6.5e9 Msun)
def eps(r): return EPSI*(1 - R0/np.asarray(r, float))
def kappa(r): return KAP0*(1 + eps(r))
# inflow cone: same kappa reused, w/v_esc = kappa cot(theta_b) - 1
W_IN = KAP0/np.tan(THB) - 1                           # = -0.85
K_IN = KAP0/(W_IN*np.sin(THB))                        # dphi/dln r on the inflow cone
# jet sheath (theta0): dphi/dr = 1/(eps_inf (r - r0) cos) + 1/(r cos)  -> closed form
def phi_sheath(r): r = np.asarray(r, float); return (np.log(np.abs(r - R0))/EPSI + np.log(r))/np.cos(TH0)
# spine (theta_s): w/v_esc = kappa(r) cot(theta_s) - 1 (>0 everywhere), dphi/dln r = kappa/((w/v_esc) sin theta_s)
_u = np.linspace(np.log(2.0), np.log(5e7), 6000); _r = np.exp(_u)
_ws = kappa(_r)/np.tan(THS) - 1
_dphi = kappa(_r)/(_ws*np.sin(THS))
_phis = np.concatenate([[0], np.cumsum(0.5*(_dphi[1:] + _dphi[:-1])*np.diff(_u))])
def phi_spine(r): return np.interp(np.log(r), _u, _phis)
if __name__ == "__main__":
    print(f"kappa0 = {KAP0:.5f}, cot(theta0) = {1/KAP0:.4f}, product = {KAP0/np.tan(TH0):.6f}")
    print(f"inflow cone 35 deg: w/v_esc = {W_IN:.3f}, dphi/dln r = {K_IN:.3f} rad  ({abs(K_IN)/(2*np.pi):.3f} turns per e-fold)")
    for r in (10, 14, 15.5, 20, 30, 100, 1e3, 1e5, 1e7):
        e = float(eps(r)); Ne = (1+e)/(2*np.pi*e*np.cos(TH0)); lam = 2*np.pi*r*e*np.cos(TH0)/(1+e)
        print(f"r = {r:9.4g} r_g: eps = {e:+.4f}  w/v_esc = {e:+.4f}  N_e = {Ne:8.2f}  lambda/r = {lam/r:+.3f}")
    print("spine w/v_esc at r = 3, 15, 1e4:", np.round(kappa(np.array([3, 15, 1e4]))/np.tan(THS) - 1, 3))
    print("theta_jet drift at large r: %.3f deg" % np.degrees(np.arctan(KAP0*(1+EPSI)) - TH0))
