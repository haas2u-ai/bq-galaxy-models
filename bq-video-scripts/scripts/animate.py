from paths import video, cache, MAX_FRAMES   # sets the working directory; see paths.py
import numpy as np, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from model import *

FPS = 30
R = 2.0; RMAX = 20.0
HOLD0, SWEEP, HOLD1 = 3.0, 20.0, 5.0
NF = int((HOLD0+SWEEP+HOLD1)*FPS)
Z0, Z1 = 6.0, 0.0
DT_MYR = 2.5                      # tracer clock per frame
rng = np.random.default_rng(1)

def z_of_frame(i):
    t = i/FPS
    if t < HOLD0: return Z0
    if t > HOLD0+SWEEP: return Z1
    s = (t-HOLD0)/SWEEP
    s = 0.5-0.5*np.cos(np.pi*s)                   # ease in/out
    return np.exp(np.log(1+Z0)*(1-s) + np.log(1+Z1)*s) - 1   # uniform in ln(1+z)

# ---- tracers -----------------------------------------------------------
N = 3000
tr_r = RMAX*np.sqrt(rng.random(N)); tr_p = 2*np.pi*rng.random(N)
arm_r = np.empty(0); arm_p = np.empty(0); arm_id = np.empty(0, int)

def step(r, p, M, H, dt):
    for _ in range(3):                           # RK2 substeps
        h = dt/3
        vo, vr = velocities(r, M, R, H)
        rm = np.maximum(r - 0.5*h*vr*KMS_TO_KPC_PER_MYR, 1e-3)
        vo2, vr2 = velocities(rm, M, R, H)
        r = r - h*vr2*KMS_TO_KPC_PER_MYR
        p = p + h*vo2/np.maximum(rm,1e-3)*KMS_TO_KPC_PER_MYR
        r = np.maximum(r, 1e-3)
    return r, p

# ---- figure ------------------------------------------------------------
BG, FG, MUT = "#0b0f19", "#e6e9ef", "#8a93a6"
C1, C2, CB, CT = "#ff9f43", "#48dbfb", "#c8d6e5", "#feca57"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":FG,"axes.labelcolor":FG,
                     "xtick.color":MUT,"ytick.color":MUT,"axes.edgecolor":"#2c3445"})
fig = plt.figure(figsize=(16,9), dpi=120, facecolor=BG)
axG = fig.add_axes([0.035,0.105,0.47,0.775], facecolor=BG)
axV = fig.add_axes([0.585,0.455,0.35,0.255], facecolor=BG)
axC = fig.add_axes([0.585,0.105,0.35,0.235], facecolor=BG)

fig.text(0.03,0.945,"Metric inflow spiral — redshift sequence", fontsize=22, weight="bold")
fig.text(0.03,0.905,"de Haas metric-inflow model · bulge R = 2 kpc · M(z) on fiducial growth track · flat ΛCDM H(z)",
         fontsize=12.5, color=MUT)

axG.set_xlim(-RMAX,RMAX); axG.set_ylim(-RMAX,RMAX); axG.set_aspect("equal")
axG.set_xlabel("x (kpc)"); axG.set_ylabel("y (kpc)")
th = np.linspace(0,2*np.pi,300)
axG.plot(R*np.cos(th), R*np.sin(th), ls="--", color=MUT, lw=1)
axG.text(R*0.72, -R*1.05, "R", color=MUT, fontsize=11)
sc_bg = axG.scatter([],[], s=2.2, c=CB, alpha=0.35, lw=0)
sc_a1 = axG.scatter([],[], s=9, c=C1, alpha=0.95, lw=0)
sc_a2 = axG.scatter([],[], s=9, c=C2, alpha=0.95, lw=0)
la1, = axG.plot([],[], color=C1, lw=1.4, alpha=0.55)
la2, = axG.plot([],[], color=C2, lw=1.4, alpha=0.55)
li1, = axG.plot([],[], color=C1, lw=1.1, alpha=0.45, ls="--")
li2, = axG.plot([],[], color=C2, lw=1.1, alpha=0.45, ls="--")
lrc, = axG.plot([],[], color=CT, lw=1.3, ls="-.")
info = fig.text(0.585, 0.875, "", fontsize=12, va="top", family="DejaVu Sans Mono", linespacing=1.45,
                bbox=dict(boxstyle="round,pad=0.6", fc="#121829", ec="#2c3445"))

# velocity panel
rr = np.linspace(0.01, RMAX, 600)
axV.set_xlim(0,RMAX); axV.set_xlabel("r (kpc)"); axV.set_ylabel("v²  (10³ km² s⁻²)")
axV.set_title("Velocity channels of v_L = v_orb φ̂ − v_rad,eff r̂", fontsize=13, color=FG, loc="left")
lvo, = axV.plot([],[], color=C1, lw=2.2, label="v²_orb")
lvr, = axV.plot([],[], color=C2, lw=2.2, label="v²_rad,eff")
lvl, = axV.plot([],[], color=FG, lw=1.4, ls=":", label="v²_L = v²_orb + v²_rad,eff")
axV.axvline(R, color=MUT, ls="--", lw=1)
axV.legend(loc="center right", frameon=False, fontsize=10.5)
axV.grid(alpha=0.12)

# cosmic evolution panel
zg = np.linspace(0,6,400)
rcg = rc(Mz(zg), Hz(zg))
axC.set_xlim(6.2,-0.2); axC.set_yscale("log"); axC.set_ylim(15,400)
axC.set_yticks([20,50,100,200]); axC.set_yticklabels(["20","50","100","200"]); axC.minorticks_off()
axC.set_xlabel("redshift z"); axC.set_ylabel("r_c (kpc)", color=CT)
axC.plot(zg, rcg, color=CT, lw=2)
axC.axhline(RMAX, color=MUT, lw=0.8, ls=":")
axC.text(5.9, RMAX*1.07, "frame edge 20 kpc", color=MUT, fontsize=9.5)
axC.set_title("Critical radius r_c = (2GM/H²)^{1/3}  and bulge mass M(z)", fontsize=13, loc="left")
axC2 = axC.twinx(); axC2.set_yscale("log"); axC2.set_ylim(5e8, 2e10)
axC2.plot(zg, Mz(zg), color=C1, lw=1.6, ls="--"); axC2.set_ylabel("M (M⊙)", color=C1)
axC2.tick_params(colors=MUT); axC2.minorticks_off()
mk_rc, = axC.plot([],[], "o", color=CT, ms=9)
mk_M,  = axC2.plot([],[], "o", color=C1, ms=7)
axC.grid(alpha=0.12, which="both")

fig.text(0.035,0.012,"Arms: streamlines of the metric inflow (outer solid, inner r<R dashed, 2nd arm rotated 180°). "
         "Dots: flow tracers in the instantaneous field (tracer clock 2.5 Myr/frame); z-sweep is quasi-static.",
         fontsize=10, color=MUT)

def pol(r,p): return r*np.cos(p), r*np.sin(p)

def frame(i):
    global tr_r, tr_p, arm_r, arm_p, arm_id
    z = z_of_frame(i); H = Hz(z); M = Mz(z); rcr = rc(M,H)
    (ro,po),(ri,pi_) = arm_locus(M,R,H,RMAX)
    for ln, off in ((la1,0),(la2,np.pi)): ln.set_data(*pol(ro,po+off))
    for ln, off in ((li1,0),(li2,np.pi)): ln.set_data(*pol(ri,pi_+off))
    lrc.set_data(*pol(np.full_like(th,rcr), th)) if rcr < RMAX*1.45 else lrc.set_data([],[])

    # background tracers
    tr_r, tr_p = step(tr_r, tr_p, M, H, DT_MYR)
    dead = tr_r < 0.08
    tr_r[dead] = RMAX*(0.97+0.03*rng.random(dead.sum())); tr_p[dead] = 2*np.pi*rng.random(dead.sum())
    sc_bg.set_offsets(np.c_[pol(tr_r,tr_p)])

    # arm tracers: move along the current streamline, dr/dt = -v_rad,eff (exact for a stationary field)
    if i % 2 == 0:
        k = 2
        arm_r = np.r_[arm_r, ro[-1] - 0.3*rng.random(2*k)]
        arm_id = np.r_[arm_id, np.r_[np.zeros(k,int), np.ones(k,int)]]
    for _ in range(3):
        _, vr = velocities(arm_r, M, R, H)
        arm_r = arm_r - DT_MYR/3*vr*KMS_TO_KPC_PER_MYR
    keep = arm_r > 0.05
    arm_r, arm_id = arm_r[keep], arm_id[keep]
    arm_p = np.where(arm_r >= R, np.interp(arm_r, ro, po), np.interp(arm_r, ri[::-1], pi_[::-1]))
    arm_p = arm_p + np.pi*arm_id
    jit = 0.12*np.sin(7.3*np.arange(arm_r.size))
    xa, ya = pol(arm_r + jit, arm_p)
    sc_a1.set_offsets(np.c_[xa[arm_id==0], ya[arm_id==0]])
    sc_a2.set_offsets(np.c_[xa[arm_id==1], ya[arm_id==1]])

    vo, vr = velocities(rr, M, R, H)
    lvo.set_data(rr, vo**2/1e3); lvr.set_data(rr, vr**2/1e3); lvl.set_data(rr, (vo**2+vr**2)/1e3)
    axV.set_ylim(0, 1.12*np.max(vo**2+vr**2)/1e3)

    vo10, vr10 = velocities(10.0, M, R, H)
    alpha = np.degrees(np.arctan(vr10/vo10))
    info.set_text(f"z = {z:5.2f}     t = {age_Gyr(z):5.2f} Gyr     H(z) = {H:6.4f} km/s/kpc\n"
                  f"M = {M:8.2e} M⊙     r_c = {rcr:6.1f} kpc     α(10 kpc) = {alpha:4.1f}°")
    mk_rc.set_data([z],[rcr]); mk_M.set_data([z],[M])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        for i in range(min(NF, MAX_FRAMES)):
            frame(i)
            if i in (60, 330, 820):
                fig.savefig(f"still_{i}.png", facecolor=BG)
    else:
        w = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1,
                         extra_args=["-pix_fmt","yuv420p","-crf","18","-preset","medium"])
        with w.saving(fig, video("metric_inflow_redshift_sequence.mp4"), dpi=120):
            for i in range(min(NF, MAX_FRAMES)):
                frame(i); w.grab_frame()
                if i % 100 == 0: print(i, flush=True)
