from paths import video, cache, MAX_FRAMES   # sets the working directory; see paths.py
import numpy as np, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from model import velocities, arm_locus, rc, KMS_TO_KPC_PER_MYR, G

# ---- disk: single-Lagrangian metric inflow (z = 0 example of the first movie) --------
M, R, H = 1.0e10, 2.0, 0.07
RD = 15.0                              # disk extent shown (kpc)
import os
DELTA = np.radians(float(os.environ.get('DELTA_DEG','4')))                # broad inflow cones at theta = pi/2 -/+ delta
DT_DISK = 3.0                          # Myr per frame (disk tracer clock)
# ---- jet: PG four-cone geometry ------------------------------------------------------
THETA0 = np.radians(6.0)               # M87 opening angle -> v_phi = 0.105 v_esc
VPHI_RATIO = np.tan(THETA0)            # 0.105
A_W = 0.10                             # display choice: w_out = A_W * v_esc  (sets helix pitch)
K_PHI = VPHI_RATIO/(A_W*np.sin(THETA0))  # dphi/dln r on the cone
R_LAUNCH, RJ = 0.3, 19.0               # jet from launch region to 26 kpc (launch region not to scale)
F_SURPLUS = 0.35                       # display choice: fraction of central inflow redirected into jets
FPS, NF = 30, int(float(os.environ.get('T_TOTAL','34'))*30)
rng = np.random.default_rng(3)

# disk tracers, steady-state seeded along the travel-time coordinate
tt, rt = [0.0], [RD]
while rt[-1] > 0.25:
    rt.append(rt[-1] - 0.2*velocities(rt[-1], M, R, H)[1]*KMS_TO_KPC_PER_MYR); tt.append(tt[-1]+0.2)
tt, rt = np.array(tt), np.array(rt)
ND = 7000
d_r = np.interp(rng.random(ND)*tt[-1], tt, rt); d_p = 2*np.pi*rng.random(ND)
d_s = rng.choice([-1, 1], ND); d_n = 0.012*rng.standard_normal(ND)
# arms (in the plane)
(ro, po), (ri, pi_) = arm_locus(M, R, H, RD)
arm_r = np.concatenate([ri[::-1], ro[1:]]); arm_p = np.concatenate([pi_[::-1], po[1:]])
NA = 900
a_r = np.interp(rng.random(NA)*tt[-1], tt, rt); a_id = rng.integers(0, 2, NA); a_side = rng.choice([-1, 1], NA)
# jet tracers
NJ = 2600
def jet_speed(r):                      # kpc/frame, qualitative: slow start, acceleration, flattening
    return 0.02 + 0.30*(1 - np.exp(-r/4.0))
j_r = np.exp(rng.uniform(np.log(R_LAUNCH), np.log(RJ), NJ))
def new_phase(n): return rng.integers(0, 2, n)*np.pi + 0.18*rng.standard_normal(n)   # double helix
def new_theta(n):                                                                    # sheath on theta0, thin spine
    return np.where(rng.random(n) < 0.75, THETA0*(0.93 + 0.07*rng.random(n)), THETA0*0.25*rng.random(n))
j_p0 = new_phase(NJ); j_s = rng.choice([-1, 1], NJ); j_thn = new_theta(NJ)
# bulge points
NB = 1500
b = rng.standard_normal((NB, 3)); b /= np.linalg.norm(b, axis=1)[:, None]
b *= (R*rng.random(NB)**(1/2.2))[:, None]; b[:, 2] *= 0.7

def disk_xyz():
    z = d_s*d_r*np.tan(DELTA) + d_n*np.minimum(d_r, 3)/3
    return d_r*np.cos(d_p), d_r*np.sin(d_p), z
def jet_xyz():
    phi = j_p0 + K_PHI*np.log(j_r/R_LAUNCH)
    rho = j_r*np.sin(j_thn)
    return rho*np.cos(phi), rho*np.sin(phi), j_s*j_r*np.cos(j_thn)

def step(frame_i):
    global d_r, d_p, j_r, j_p0, j_s, j_thn
    for _ in range(3):
        h = DT_DISK/3
        vo, vr = velocities(d_r, M, R, H)
        d_r = d_r - h*vr*KMS_TO_KPC_PER_MYR
        d_p = d_p + h*vo/np.maximum(d_r, 1e-3)*KMS_TO_KPC_PER_MYR
    global a_r
    for _ in range(3):
        a_r = a_r - DT_DISK/3*velocities(a_r, M, R, H)[1]*KMS_TO_KPC_PER_MYR
    a_done = a_r < 0.1; a_r[a_done] = RD*(0.97 + 0.03*rng.random(a_done.sum()))
    arrived = d_r < 0.25
    n_arr = arrived.sum()
    # surplus into the polar cones: recycle the oldest jet tracers (those beyond RJ) first
    j_r = j_r + jet_speed(j_r)
    gone = np.where(j_r > RJ)[0]
    n_new = min(len(gone), rng.binomial(n_arr, F_SURPLUS) if n_arr else 0) if len(gone) else 0
    # tracers beyond RJ that are not yet refilled are parked (hidden) until central inflow feeds them
    j_r[gone] = RJ + 1
    if n_new:
        idx = gone[:n_new]
        j_r[idx] = R_LAUNCH*(1 + 0.2*rng.random(n_new)); j_p0[idx] = new_phase(n_new)
        j_s[idx] = rng.choice([-1, 1], n_new); j_thn[idx] = new_theta(n_new)
    d_r[arrived] = RD*(0.97 + 0.03*rng.random(n_arr)); d_p[arrived] = 2*np.pi*rng.random(n_arr)
    d_s[arrived] = rng.choice([-1, 1], n_arr)

# balance: fraction of jet tracers needed; pre-run to steady state
for i in range(400): step(i)

# ---- figure ------------------------------------------------------------------------
BG, FG, MUT = "#05070d", "#e6e9ef", "#8a93a6"
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
ax = fig.add_axes([-0.10, -0.10, 0.84, 1.20], projection="3d", facecolor=BG)
ax.set_axis_off()
L = 14.5
ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-19.5, 19.5)
ax.set_box_aspect((1, 1, 1.3))

sc_b = ax.scatter(b[:, 0], b[:, 1], b[:, 2], s=3, c="#ffe6b0", alpha=0.25, lw=0, depthshade=False)
sc_d = ax.scatter([], [], [], s=1.6, lw=0, depthshade=False)
sc_j = ax.scatter([], [], [], s=3.0, lw=0, depthshade=False)
sc_a = ax.scatter([], [], [], s=6, c='#ffb86b', lw=0, alpha=0.95, depthshade=False)
for off in (0, np.pi):
    for sg in (1, -1):
        ax.plot(arm_r*np.cos(DELTA)*np.cos(arm_p+off), arm_r*np.cos(DELTA)*np.sin(arm_p+off), sg*arm_r*np.sin(DELTA),
                color="#ff9f43", lw=1.1, alpha=0.55)
# cone guides
th = np.linspace(0, 2*np.pi, 120)
for sgn in (1, -1):
    for rr in (6, 12, 18):
        ax.plot(rr*np.sin(THETA0)*np.cos(th), rr*np.sin(THETA0)*np.sin(th), sgn*rr*np.cos(THETA0)*np.ones_like(th),
                color="#48dbfb", lw=0.6, alpha=0.35)
    for a in np.linspace(0, 2*np.pi, 8, endpoint=False):
        ax.plot([0, RJ*np.sin(THETA0)*np.cos(a)], [0, RJ*np.sin(THETA0)*np.sin(a)], [0, sgn*RJ*np.cos(THETA0)],
                color="#48dbfb", lw=0.5, alpha=0.25)
    ax.plot(RD*np.cos(th), RD*np.sin(th), sgn*RD*np.tan(DELTA)*np.ones_like(th), color="#ff9f43", lw=0.6, alpha=0.3)
    for a in np.linspace(0, 2*np.pi, 16, endpoint=False):
        ax.plot([-RD*np.cos(a), RD*np.cos(a)], [-RD*np.sin(a), RD*np.sin(a)], [-sgn*RD*np.tan(DELTA), sgn*RD*np.tan(DELTA)],
                color="#ffc98a", lw=0.5, alpha=0.18)
ax.plot(R*np.cos(th), R*np.sin(th), 0*th, color=MUT, lw=0.8, ls="--", alpha=0.6)
rh = np.geomspace(R_LAUNCH, RJ, 800)
for sgn in (1, -1):
    for ph0 in (0, np.pi):
        ph = ph0 + K_PHI*np.log(rh/R_LAUNCH)
        ax.plot(rh*np.sin(THETA0)*np.cos(ph), rh*np.sin(THETA0)*np.sin(ph), sgn*rh*np.cos(THETA0),
                color="#c7ecff", lw=0.7, alpha=0.35)

# colours
cmap_d = plt.get_cmap("YlOrBr_r"); cmap_j = plt.get_cmap("cool")

fig.text(0.655, 0.93, "Spiral galaxy with polar jets", fontsize=22, weight="bold", color=FG)
fig.text(0.655, 0.895, "de Haas metric-inflow disk + three-rapidity PG four-cone jets", fontsize=11.5, color=MUT)
txt = ("Disk (single Lagrangian)\n"
       f"  M = 1.0×10¹⁰ M⊙, R = 2 kpc, H = 0.070 km/s/kpc\n"
       f"  r_c = {rc(M, H):.0f} kpc,  v_L = v_orb φ̂ − v_rad,eff r̂\n"
       f"  broad inflow cones θ = π/2 ∓ {np.degrees(DELTA):.0f}°\n\n"
       "Jet (PG cone flow)\n"
       "  w_θ₀(r) = −√(2GM/r) + v_φ cot θ₀\n"
       "  launch where w_θ₀ > 0\n"
       "  θ_jet = arctan(v_φ / v_esc) = 6°\n"
       "  v_φ = 0.105 v_esc  (M87 constraint)\n"
       "  double-helix streamlines, spine–sheath\n"
       "  nested cones, bipolar θ → π − θ\n\n"
       "Bernoulli funnel: surplus central inflow\n"
       "  → narrow polar cones, A_narrow << A_broad")
fig.text(0.655, 0.84, txt, fontsize=10.5, va="top", family="DejaVu Sans Mono", color=FG, linespacing=1.45,
         bbox=dict(boxstyle="round,pad=0.7", fc="#10151f", ec="#2c3445"))
fig.text(0.655, 0.20, "Schematic, not to scale: launch region (≲ 10³ r_g)\n"
         "magnified; jet speed and helix pitch are display\n"
         f"choices (w_out = {A_W}·v_esc, surplus fraction {F_SURPLUS:.0%}).\n"
         "Disk flow at 3 Myr/frame from the model equations.",
         fontsize=9.5, va="top", color=MUT, linespacing=1.4)
caption = fig.text(0.33, 0.035, "", ha="center", fontsize=13.5, color=FG,
                   bbox=dict(boxstyle="round,pad=0.45", fc="#10151f", ec="#2c3445"))

AZ0 = -60
T_TOT = NF/FPS; HOLD = 1.5
_th = np.linspace(0.5, 179.5, 4000)
_speed = 0.10 + 0.90*(1 - np.exp(-((_th-90)/11.0)**2))       # slow motion near 90 deg
_tcum = np.concatenate([[0], np.cumsum(np.diff(_th)/(0.5*(_speed[1:]+_speed[:-1])))]); _tcum /= _tcum[-1]
def theta_of_s(s):
    u = np.clip((s-HOLD)/(T_TOT-2*HOLD), 0, 1)
    return float(np.interp(u, _tcum, _th))
theta_txt = fig.text(0.34, 0.95, "", ha="center", fontsize=15, color="#e6e9ef", family="DejaVu Sans Mono")

def frame(i):
    step(i)
    s = i/FPS
    x, y, z = disk_xyz()
    sc_d._offsets3d = (x, y, z)
    sc_d.set_color(cmap_d(np.clip(d_r/RD, 0, 1)*0.85)); sc_d.set_alpha(0.55)
    ap = np.interp(a_r, arm_r, arm_p) + np.pi*a_id
    sc_a._offsets3d = (a_r*np.cos(DELTA)*np.cos(ap), a_r*np.cos(DELTA)*np.sin(ap), a_side*a_r*np.sin(DELTA))
    vis = j_r <= RJ
    jx, jy, jz = jet_xyz()
    sc_j._offsets3d = (jx[vis], jy[vis], jz[vis])
    sc_j.set_color(cmap_j(np.clip(np.log(j_r[vis]/R_LAUNCH)/np.log(RJ/R_LAUNCH), 0, 1)*0.8)); sc_j.set_alpha(0.85)
    th_view = theta_of_s(s)
    ax.view_init(elev=90 - th_view, azim=AZ0)
    theta_txt.set_text(f"viewing angle θ = {th_view:5.1f}°" + ("   ·  slow motion" if abs(th_view-90) < 14 else ""))
    if th_view < 40:    caption.set_text("Face-on from the north: the disk spiral and the jet base")
    elif th_view < 76:  caption.set_text("Tilting down: the jets rise out of the disk plane")
    elif th_view < 104: caption.set_text("Edge-on (θ = 90°): the two broad inflow cones θ = π/2 ∓ δ form an X")
    elif th_view < 140: caption.set_text("Past the plane: the southern cone and the counter-jet")
    else:               caption.set_text("Face-on from the south: the same PG flow, mirrored θ → π − θ")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        for i in range(0, NF, 1):
            if i in (0, 300, 600): frame(i); fig.savefig(f"j_{i}.png", facecolor=BG)
            else: step(i)
    else:
        w = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1,
                         extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"])
        with w.saving(fig, video("pg_jets_theta_sweep_xshape.mp4"), dpi=120):
            for i in range(min(NF, MAX_FRAMES)):
                frame(i); w.grab_frame()
                if i % 100 == 0: print(i, flush=True)

def still_at(theta_deg, name):
    for k in range(60): step(k)
    s_target = None
    for i in range(NF):
        if abs(theta_of_s(i/FPS) - theta_deg) < 0.3: s_target = i; break
    frame(s_target); fig.savefig(name, facecolor=BG)
