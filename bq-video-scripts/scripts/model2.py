import numpy as np, warnings; warnings.filterwarnings("ignore")
from riverbed import H, GM2, GM, RB, X, v_in_cl, v_phi, vorticity
RU = 0.25                                  # display choice: space-drift balance radius (Q/4pi H)^(1/3)
Q4 = H*RU**3                               # Q/(4 pi)
A_ABS = 3*Q4/RB**3                         # per-cell absorption rate inside the matter (uniform in the bulge)
def v_space(r):                            # slow drift of space cells, outward positive (incompressible)
    r = np.asarray(r, float)
    return np.where(r <= RB, H*r - Q4*r/RB**3, H*r - Q4/np.maximum(r, 1e-9)**2)
def v_river(r):                            # riverbed acting on mass, outward positive (CL radial profile)
    return -v_in_cl(r)
def absorption(r): return np.where(np.asarray(r) <= RB, A_ABS, 0.0)
def influx(r):                             # net inward space flux through a sphere / 4pi  (diminishes outward)
    r = np.asarray(r, float); return np.where(r <= RB, Q4*(r/RB)**3, Q4) - H*r**3
if __name__ == "__main__":
    r = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 1.5])
    div = lambda f, x: ((x+1e-6)**2*f(x+1e-6) - (x-1e-6)**2*f(x-1e-6))/(2e-6)/x**2
    print("div v_space - (3H - absorption):", np.round(div(v_space, r) - (3*H - absorption(r)), 8))
    print(f"Q/4pi = {Q4:.4f}, per-cell absorption inside R = {A_ABS:.2f} (units of H)")
    print("drift at R vs riverbed at R:", round(float(-v_space(RB)), 3), round(float(-v_river(RB)), 3))
    print("v_space:", np.round(v_space(r), 3)); print("v_river:", np.round(v_river(r), 3)); print("influx:", np.round(influx(r), 4))
