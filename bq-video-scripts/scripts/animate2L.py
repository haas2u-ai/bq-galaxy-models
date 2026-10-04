from paths import video, cache, MAX_FRAMES   # sets the working directory; see paths.py
import numpy as np, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from model2L import *

R1, M1, R2, M2 = 0.500, 4.09e7, 2.30, 5.81e8        # NGC 3741 two-L fit, Table X
H = Hz_from_si(2.2e-18)
RMAX = 7.5
FPS = 30; T_TOTAL = 30.0; NF = int(T_TOTAL*FPS)
DT = 3.0                                            # Myr per frame (tracer clock)
TA0, TA1 = 0.5, 3.5      # grow inner (L1) arms
TR0, TR1 = 3.5, 5.0      # ring at R2 appears
TB0, TB1 = 5.0, 8.5      # grow outer (L2) arms
rng = np.random.default_rng(7)
vel = lambda r: velocities_2L(r, R1, M1, R2, M2, H)
rs, ps = streamline(R1, M1, R2, M2, H, RMAX)

def ease(t, a, b): s = np.clip((t-a)/(b-a), 0, 1); return 0.5-0.5*np.cos(np.pi*s)

# ---- tracers (burned in to steady state before recording) ----------------
N = 4500
# steady-state seeding: uniform in travel time along the inflow from RMAX
_tt, _rt = [0.0], [RMAX]
while _rt[-1] > 0.03:
    _rt.append(_rt[-1] - 0.05*vel(_rt[-1])[1]*KMS_TO_KPC_PER_MYR); _tt.append(_tt[-1] + 0.05)
_tt, _rt = np.array(_tt), np.array(_rt)
tr_r = np.interp(rng.random(N)*_tt[-1], _tt, _rt); tr_p = 2*np.pi*rng.random(N)
arm_r = np.empty(0); arm_id = np.empty(0, int)

def step_bg(r, p, dt):
    for _ in range(4):
        h = dt/4
        vo, vr = vel(r)
        rm = np.maximum(r - 0.5*h*vr*KMS_TO_KPC_PER_MYR, 1e-3)
        vo2, vr2 = vel(rm)
        r = np.maximum(r - h*vr2*KMS_TO_KPC_PER_MYR, 1e-3)
        p = p + h*vo2/rm*KMS_TO_KPC_PER_MYR
    return r, p

def advance(k_inject=True):
    global tr_r, tr_p, arm_r, arm_id
    tr_r, tr_p = step_bg(tr_r, tr_p, DT)
    dead = tr_r < 0.03
    tr_r[dead] = RMAX*(0.985+0.015*rng.random(dead.sum())); tr_p[dead] = 2*np.pi*rng.random(dead.sum())
    if k_inject:
        arm_r = np.r_[arm_r, RMAX - 0.15*rng.random(4)]
        arm_id = np.r_[arm_id, [0,0,1,1]]
    for _ in range(4):
        _, vr = vel(arm_r); arm_r = arm_r - DT/4*vr*KMS_TO_KPC_PER_MYR
    keep = arm_r > 0.03; arm_r, arm_id = arm_r[keep], arm_id[keep]

for _ in range(120): advance()            # fill the arms (~360 Myr)

# ---- figure -------------------------------------------------------------
BG, FG, MUT = "#0b0f19", "#e6e9ef", "#8a93a6"
CL1, CL2, CRING, CB = "#ff9f43", "#48dbfb", "#feca57", "#c8d6e5"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":FG,"axes.labelcolor":FG,
                     "xtick.color":MUT,"ytick.color":MUT,"axes.edgecolor":"#2c3445"})
fig = plt.figure(figsize=(16,9), dpi=120, facecolor=BG)
axG = fig.add_axes([0.035,0.105,0.47,0.775], facecolor=BG)
axV = fig.add_axes([0.585,0.455,0.35,0.255], facecolor=BG)
axA = fig.add_axes([0.585,0.105,0.35,0.235], facecolor=BG)
fig.text(0.03,0.945,"Double-Lagrangian galaxy — nested spiral (bulge · bar · ring · disk)", fontsize=22, weight="bold")
fig.text(0.03,0.905,"de Haas metric-inflow model, Eqs. (24)–(26) · parameters: NGC 3741 two-L fit (Table X) · "
         "H_z = 2.2×10⁻¹⁸ s⁻¹", fontsize=12.5, color=MUT)

axG.set_xlim(-RMAX,RMAX); axG.set_ylim(-RMAX,RMAX); axG.set_aspect("equal")
axG.set_xlabel("x (kpc)"); axG.set_ylabel("y (kpc)")
th = np.linspace(0,2*np.pi,400)
axG.plot(R1*np.cos(th), R1*np.sin(th), ls="--", color=MUT, lw=1)
ring, = axG.plot(R2*np.cos(th), R2*np.sin(th), color=CRING, lw=1.6, ls="-.", alpha=0)
sc_bg = axG.scatter([],[], s=2.4, c=CB, alpha=0.33, lw=0)
sc_1 = axG.scatter([],[], s=10, c=CL1, alpha=0.95, lw=0)
sc_2 = axG.scatter([],[], s=10, c=CL2, alpha=0.95, lw=0)
arms = [axG.plot([],[], color=c, lw=1.6, alpha=0.6)[0] for c in (CL1, CL1, CL2, CL2)]
caption = axG.text(0, -RMAX*0.93, "", ha="center", fontsize=13.5, color=FG,
                   bbox=dict(boxstyle="round,pad=0.45", fc="#121829", ec="#2c3445"))
lab_R1 = axG.text(R1*0.75, -R1*1.45, "R₁", color=MUT, fontsize=11)
lab_R2 = axG.text(R2*0.72, -R2*0.86, "R₂  ring", color=CRING, fontsize=11, alpha=0)

X1 = np.sqrt(2*G*M1/R1) - H*R1; X2 = np.sqrt(2*G*M2/R2) - H*R2
fig.text(0.585, 0.875,
    f"L₁ bulge–bar : R₁={R1:.2f} kpc  M₁={M1:.2e} M⊙  r_c={rc(M1,H):5.1f} kpc  v_L={np.sqrt(1.5)*X1:4.1f} km/s\n"
    f"L₂ bar→disk  : R₂={R2:.2f} kpc  M₂={M2:.2e} M⊙  r_c={rc(M2,H):5.1f} kpc  v_L={np.sqrt(1.5)*X2:4.1f} km/s",
    fontsize=10.5, va="top", family="DejaVu Sans Mono", linespacing=1.6,
    bbox=dict(boxstyle="round,pad=0.6", fc="#121829", ec="#2c3445"))

# static right panels, split at R2 so the jump is shown as a jump
def split_curve(ax, f, **kw):
    a = np.linspace(0.005, R2, 400); b = np.linspace(R2*(1+1e-9), RMAX, 600)
    l, = ax.plot(a, f(a), **kw); kw.pop("label", None); ax.plot(b, f(b), **kw); return l
vo2 = lambda r: vel(r)[0]**2; vr2 = lambda r: vel(r)[1]**2
for ax in (axV, axA):
    ax.axvspan(0, R1, color="#ffffff", alpha=0.035); ax.axvspan(R1, R2, color=CL1, alpha=0.06)
    ax.axvspan(R2, RMAX, color=CL2, alpha=0.05)
    ax.axvline(R2, color=CRING, lw=1.2, ls="-."); ax.axvline(R1, color=MUT, lw=1, ls="--")
    ax.set_xlim(0, RMAX); ax.grid(alpha=0.12)
split_curve(axV, vo2, color=FG, lw=2.2, label="v²_orb")
split_curve(axV, vr2, color="#a29bfe", lw=2.2, label="v²_rad,eff")
split_curve(axV, lambda r: vo2(r)+vr2(r), color=MUT, lw=1.3, ls=":", label="v²_L")
axV.set_ylabel("v²  (km² s⁻²)"); axV.set_xlabel("r (kpc)")
axV.set_title("Velocity channels — piecewise Lagrangians", fontsize=13, loc="left")
axV.legend(loc="upper right", frameon=False, fontsize=10, ncol=3)
axV.set_ylim(0, 1.25*max(np.sqrt(1.5)*X2, np.sqrt(1.5)*X1)**2)
for x, s, c in ((R1/2, "bulge", MUT), ((R1+R2)/2, "bar (L₁)", CL1), ((R2+RMAX)/2, "disk (L₂)", CL2)):
    axA.text(x, 84 if s != "bulge" else 6, s, ha="center", color=c, fontsize=10.5)
alpha = lambda r: np.degrees(np.arctan(vel(r)[1]/np.maximum(vel(r)[0],1e-9)))
split_curve(axA, alpha, color=CRING, lw=2.2)
axA.set_ylim(0, 92); axA.set_ylabel("pitch α (deg)"); axA.set_xlabel("r (kpc)")
axA.set_title("Spiral pitch angle  tan α = v_rad,eff / v_orb", fontsize=13, loc="left")
mkV, = axV.plot([],[], "o", color=FG, ms=6); mkA, = axA.plot([],[], "o", color=CRING, ms=7)

fig.text(0.035,0.012,"Arms: one continuous streamline of the metric inflow (+ copy rotated 180°), colored by the Lagrangian that governs it. "
         "Dots: flow tracers (clock 3 Myr/frame). Global Φ_BH offset of the fit not included (v² gauge only).",
         fontsize=9.5, color=MUT)

pol = lambda r, p: (r*np.cos(p), r*np.sin(p))

def frame(i):
    t = i/FPS
    advance()
    # arm growth: inner part reveals outward to R2, then outer part to RMAX
    rin = R2*ease(t, TA0, TA1); rout = R2 + (RMAX-R2)*ease(t, TB0, TB1)
    m1 = rs <= min(rin, R2); m2 = (rs > R2) & (rs <= rout)
    for k, off in enumerate((0, np.pi)):
        arms[k].set_data(*pol(rs[m1], ps[m1]+off)); arms[k+2].set_data(*pol(rs[m2], ps[m2]+off))
    ra = ease(t, TR0, TR1); ring.set_alpha(0.85*ra); lab_R2.set_alpha(ra)

    sc_bg.set_offsets(np.c_[pol(tr_r, tr_p)])
    ap = np.interp(arm_r, rs, ps) + np.pi*arm_id
    jit = 0.035*np.sin(7.3*np.arange(arm_r.size))
    x, y = pol(arm_r+jit, ap)
    vis = ((arm_r <= R2) & (arm_r <= rin)) | ((arm_r > R2) & (arm_r <= rout))
    inner = arm_r <= R2
    sc_1.set_offsets(np.c_[x[vis & inner], y[vis & inner]])
    sc_2.set_offsets(np.c_[x[vis & ~inner], y[vis & ~inner]])

    rp = 0.05 + (RMAX-0.1)*(0.5-0.5*np.cos(2*np.pi*max(t-9, 0)/14)) if t > 9 else None
    if rp is not None:
        mkV.set_data([rp],[vo2(rp)]); mkA.set_data([rp],[alpha(rp)])
    if t < TA1:   caption.set_text("Lagrangian 1: compact bulge (R₁) and its spiral — the relic 'bar'")
    elif t < TB0: caption.set_text("Critical transition at R₂: the bulge–bar system acts as a new bulge")
    elif t < TB1: caption.set_text("Lagrangian 2: bar-as-new-bulge drives the outer disk spiral")
    elif t < 20:  caption.set_text("Inflow slows across R₂ (46 → 12 km/s): tracer density jumps ~4× — sharp ring edge")
    else:         caption.set_text("Nested spiral: pitch kinks from 55° to 22° at the ring")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        for i in range(min(NF, MAX_FRAMES)):
            frame(i)
            if i in (60, 160, 700): fig.savefig(f"s2L_{i}.png", facecolor=BG)
    else:
        w = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1,
                         extra_args=["-pix_fmt","yuv420p","-crf","18","-preset","medium"])
        with w.saving(fig, video("double_lagrangian_nested_spiral.mp4"), dpi=120):
            for i in range(min(NF, MAX_FRAMES)):
                frame(i); w.grab_frame()
