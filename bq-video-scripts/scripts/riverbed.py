import numpy as np
import warnings; warnings.filterwarnings("ignore")
# schematic units: H = 1, r_c = (2GM/H^2)^(1/3) = 1  ->  2GM = 1
H = 1.0; GM2 = 1.0; GM = 0.5; RB = 0.12          # RB: bulge radius for the CL azimuthal flow
def v_hub(r): return H*np.asarray(r, float)                       # outward
def v_new(r): return -np.sqrt(GM2/np.asarray(r, float))           # inward (outward-positive convention)
def v_rad(r, h=1.0, m=1.0): return h*v_hub(r) + m*v_new(r)         # combined riverbed, outward positive
def creation(r, h=1.0): return 3*h*H*np.ones_like(np.asarray(r, float))          # per-cell rate (= div of H r)
def absorption(r, m=1.0): return m*1.5*np.sqrt(GM2/np.asarray(r, float)**3)        # per-cell rate (= -div of inflow)
def sigma(r, h=1.0, m=1.0): return creation(r, h) - absorption(r, m)
# CL azimuthal flow (single Lagrangian, bulge RB)
X = np.sqrt(GM2/RB) - H*RB
def v_in_cl(r):                         # inward-positive radial speed of the CL model
    r = np.asarray(r, float)
    return np.where(r <= RB, np.sqrt(GM/RB*(3 - (r/RB)**2)) - H*r, np.sqrt(GM2/r) - H*r)
def v_phi(r):
    r = np.asarray(r, float)
    q = np.where(r <= RB, 0.5*X**2*(r/RB)**2, 1.5*X**2 - v_in_cl(r)**2)
    return np.sqrt(np.maximum(q, 0))
def vorticity(r, h=1e-5): return ((r+h)*v_phi(r+h) - (r-h)*v_phi(r-h))/(2*h)/r
if __name__ == "__main__":
    r = np.linspace(0.05, 1.6, 2000)
    rc = r[np.argmin(np.abs(v_rad(r)))]; rs = r[np.argmin(np.abs(sigma(r)))]
    print(f"velocity balance r_c = {rc:.4f}   cell balance r_sigma = {rs:.4f}   (GM/2H^2)^(1/3) = {(GM/(2*H**2))**(1/3):.4f}")
    # divergence check: (1/r^2) d(r^2 v)/dr == sigma
    rr = np.linspace(0.1, 1.5, 7); dv = ((rr+1e-6)**2*v_rad(rr+1e-6) - (rr-1e-6)**2*v_rad(rr-1e-6))/(2e-6)/rr**2
    print("divergence check, max |div v - sigma| =", np.max(np.abs(dv - sigma(rr))))
    # convective acceleration check vs paper Eq. (96) with a''/a = H^2
    vv = v_rad(rr); dvr = (v_rad(rr+1e-6) - v_rad(rr-1e-6))/2e-6
    print("v dv/dr - [H^2 r - GM/r^2 - 1/2 H sqrt(2GM/r)] max =", np.max(np.abs(vv*dvr - (H**2*rr - GM/rr**2 - 0.5*H*np.sqrt(GM2/rr)))))
    print(f"X = {X:.3f}, v_L = sqrt(1.5) X = {np.sqrt(1.5)*X:.3f}; v_phi at 0.3, 1, 1.6:", np.round(v_phi(np.array([0.3, 1, 1.6])), 3))
    print("vorticity at 0.3, 1, 1.6:", np.round(vorticity(np.array([0.3, 1.0, 1.6])), 3))
