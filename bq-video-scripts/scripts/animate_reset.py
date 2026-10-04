from paths import video, cache, MAX_FRAMES   # sets the working directory; see paths.py
import numpy as np, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from model_reset import *

FPS = 30; NF = 900
S_A0, S_A1 = 1.0, 11.0          # proto-galaxy sweep t_start -> t_on
S_R0, S_R1 = 11.0, 15.0         # reset (cosmic time frozen at t_on)
S_B0, S_B1 = 15.0, 26.0         # evolution t_on -> today
RV_PRE, RV_POST = 3.3, 7.5
rng = np.random.default_rng(11)
ease = lambda s, a, b: 0.5-0.5*np.cos(np.pi*np.clip((s-a)/(b-a), 0, 1))
def logsweep(a, b, u): return np.exp(np.log(a)*(1-u) + np.log(b)*u)

def state(s):
    """video second -> (cosmic t, reset flag, view radius, L2 growth fraction)"""
    if s < S_R0: t = logsweep(T_START, T_ON, ease(s, S_A0, S_A1)); reset = False
    elif s < S_B0: t = T_ON; reset = s >= S_R0 + 0.8
    else: t = logsweep(T_ON, T0, ease(s, S_B0, S_B1)); reset = True
    rv = RV_PRE + (RV_POST-RV_PRE)*ease(s, S_R0+0.8, S_R1)
    g2 = ease(s, S_R0+1.2, S_R1-0.3)
    return t, reset, rv, g2

def vel(r, t, reset):
    H = H_of_t(t)
    if not reset: return velocities_2L(r, R1, M1(t), 1e9, 1.0, H)
    return velocities_2L(r, R1, M1(t), R2, M2(t), H)

def seg(a, b, t, reset, n=900):
    u = np.linspace(0, 1, n); rr = a + (b-a)*(1-(1-u)**3)     # cluster points near b
    vo, vr = vel(rr, t, reset)
    f = -vo/(rr*np.where(np.abs(vr) < 1e-6, 1e-6, vr))
    return rr, np.concatenate([[0], np.cumsum(0.5*(f[1:]+f[:-1])*np.diff(rr))])

def streamline(t, reset, rmax):
    r0, p0 = seg(R1, 0.01, t, reset, 300)
    end1 = min(R2*(1-1e-6), 0.995*rc1(t)) if reset else min(0.995*rc1(t), rmax)
    r1, p1 = seg(R1, end1, t, reset)
    rs = [r0[::-1], r1[1:]]; ps = [p0[::-1], p1[1:]]
    if reset:
        end2 = min(rmax, 0.995*rc2(t))
        r2, p2 = seg(R2*(1+1e-6), end2, t, reset)
        rs.append(r2); ps.append(p2 + p1[-1])
    return np.concatenate(rs), np.concatenate(ps), end1

# ---- tracers --------------------------------------------------------------
N = 6000
def spawn(n, rv): return rv*1.05*np.sqrt(rng.random(n)), 2*np.pi*rng.random(n)
tr_r, tr_p = spawn(N, RV_PRE)
arm_r = np.empty(0); arm_id = np.empty(0, int)

def tracer_dt(t, reset):
    vo, _ = vel(1.0, t, reset)
    return 0.045*1.0/(max(float(vo), 0.2)*KMS_TO_KPC_PER_MYR)     # Myr/frame, ~0.045 rad/frame at 1 kpc

def advance(t, reset, rv, r_arm_end, inject=True):
    global tr_r, tr_p, arm_r, arm_id
    dt = tracer_dt(t, reset)
    for _ in range(4):
        h = dt/4
        vo, vr = vel(tr_r, t, reset)
        rm = np.maximum(tr_r - 0.5*h*vr*KMS_TO_KPC_PER_MYR, 1e-3)
        vo2, vr2 = vel(rm, t, reset)
        tr_r = np.maximum(tr_r - h*vr2*KMS_TO_KPC_PER_MYR, 1e-3)
        tr_p = tr_p + h*vo2/rm*KMS_TO_KPC_PER_MYR
    bad = (tr_r < 0.03) | (tr_r > rv*1.05)
    tr_r[bad], tr_p[bad] = spawn(bad.sum(), rv)
    if inject:
        arm_r = np.r_[arm_r, r_arm_end*(0.96 - 0.03*rng.random(4))]
        arm_id = np.r_[arm_id, [0, 0, 1, 1]]
    for _ in range(4):
        _, vr = vel(arm_r, t, reset); arm_r = arm_r - dt/4*vr*KMS_TO_KPC_PER_MYR
    keep = (arm_r > 0.03) & (arm_r < r_arm_end); arm_r, arm_id = arm_r[keep], arm_id[keep]
    return dt

for _ in range(150):                                   # pre-fill proto-arms at t_start
    _, _, e = streamline(T_START, False, RV_PRE); advance(T_START, False, RV_PRE, e)

# ---- figure -------------------------------------------------------------
BG, FG, MUT = "#0b0f19", "#e6e9ef", "#8a93a6"
CL1, CL2, CRING, CB, CRAD = "#ff9f43", "#48dbfb", "#feca57", "#c8d6e5", "#a29bfe"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":FG,"axes.labelcolor":FG,
                     "xtick.color":MUT,"ytick.color":MUT,"axes.edgecolor":"#2c3445"})
fig = plt.figure(figsize=(16,9), dpi=120, facecolor=BG)
axG = fig.add_axes([0.035,0.105,0.47,0.775], facecolor=BG)
axV = fig.add_axes([0.585,0.455,0.35,0.235], facecolor=BG)
axT = fig.add_axes([0.585,0.105,0.35,0.235], facecolor=BG)
fig.text(0.03,0.945,"Bulge reset — birth of a nested spiral", fontsize=22, weight="bold")
fig.text(0.03,0.905,"de Haas metric-inflow model · onset r_c,1 = R₂ (v_esc = v_H), Eq. (19) · M ∝ t^1.5, Eq. (20) · "
         "NGC 3741 structure · flat ΛCDM", fontsize=12.5, color=MUT)
axG.set_aspect("equal"); axG.set_xlabel("x (kpc)"); axG.set_ylabel("y (kpc)")
th = np.linspace(0, 2*np.pi, 400)
axG.plot(R1*np.cos(th), R1*np.sin(th), ls="--", color=MUT, lw=1)
lrc1, = axG.plot([], [], color=CL1, lw=1.2, ls=":", alpha=0.9)
ring, = axG.plot(R2*np.cos(th), R2*np.sin(th), color=CRING, lw=1.5, ls="-.", alpha=0.25)
lrc2, = axG.plot([], [], color=CL2, lw=1.2, ls=":", alpha=0.9)
sc_bg = axG.scatter([], [], s=2.4, c=CB, alpha=0.32, lw=0)
sc_1 = axG.scatter([], [], s=10, c=CL1, alpha=0.95, lw=0)
sc_2 = axG.scatter([], [], s=10, c=CL2, alpha=0.95, lw=0)
arms = [axG.plot([], [], color=c, lw=1.6, alpha=0.6)[0] for c in (CL1, CL1, CL2, CL2)]
flash = axG.scatter([0], [0], s=0, facecolors="none", edgecolors=CRING, lw=2.5)
caption = axG.text(0.5, 0.035, "", transform=axG.transAxes, ha="center", fontsize=13, color=FG,
                   bbox=dict(boxstyle="round,pad=0.45", fc="#121829", ec="#2c3445"))
info = fig.text(0.585, 0.875, "", fontsize=10.5, va="top", family="DejaVu Sans Mono", linespacing=1.55,
                bbox=dict(boxstyle="round,pad=0.6", fc="#121829", ec="#2c3445"))

axV.set_title("Velocity channels  (v_rad,eff > 0: inflow · < 0: Hubble outflow)", fontsize=12.5, loc="left")
axV.set_xlabel("r (kpc)"); axV.set_ylabel("v (km/s)"); axV.grid(alpha=0.12)
axV.axhline(0, color=MUT, lw=0.8)
lvo, = axV.plot([], [], color=FG, lw=2.1, label="v_orb")
lvr, = axV.plot([], [], color=CRAD, lw=2.1, label="v_rad,eff")
lvo2, = axV.plot([], [], color=FG, lw=2.1); lvr2, = axV.plot([], [], color=CRAD, lw=2.1)
vR2 = axV.axvline(R2, color=CRING, lw=1.2, ls="-.", alpha=0.25)
vrc1 = axV.axvline(0, color=CL1, lw=1, ls=":")
axV.legend(loc="upper right", frameon=False, fontsize=10, ncol=2)

tg = np.geomspace(T_START, T0, 400)
axT.set_xscale("log"); axT.set_yscale("log"); axT.set_xlim(T_START*0.95, T0*1.05); axT.set_ylim(0.8, 150)
axT.plot(tg, rc1(tg), color=CL1, lw=1.2, alpha=0.35)
axT.plot(tg[tg >= T_ON], rc2(tg[tg >= T_ON]), color=CL2, lw=1.2, alpha=0.35)
axT.axhline(R2, color=CRING, lw=1.2, ls="-.")
axT.text(T_START*1.02, R2*1.12, "R₂ (bar–ring–disk)", color=CRING, fontsize=9.5)
axT.axvline(T_ON, color=CRING, lw=1, ls=":")
axT.text(T_ON*1.05, 90, f"onset\nt = {T_ON:.2f} Gyr\nz = {z_of_t(T_ON):.2f}", color=CRING, fontsize=9.5, va="top")
tr1, = axT.plot([], [], color=CL1, lw=2.4, label="r_c,1 (L₁)"); tr2, = axT.plot([], [], color=CL2, lw=2.4, label="r_c,2 (L₂)")
mk1, = axT.plot([], [], "o", color=CL1, ms=7); mk2, = axT.plot([], [], "o", color=CL2, ms=7)
axT.set_xlabel("cosmic time t (Gyr)"); axT.set_ylabel("r_c (kpc)")
axT.set_xticks([0.5, 1, 2, 5, 10]); axT.set_xticklabels(["0.5", "1", "2", "5", "10"])
axT.set_yticks([1, 2, 5, 10, 20, 50, 100]); axT.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
axT.minorticks_off(); axT.grid(alpha=0.12)
axT.set_title("Critical radii r_c = (2GM/H²)^{1/3}", fontsize=12.5, loc="left")
axT.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=10)
fig.text(0.035, 0.012, "Arms: continuous inflow streamline (+ 180° copy), orange = L₁, blue = L₂; dotted circles: current r_c. "
         "Dots: tracers in the instantaneous field, adaptive tracer clock (quasi-static sequence).", fontsize=9.5, color=MUT)
pol = lambda r, p: (r*np.cos(p), r*np.sin(p))

def frame(i):
    s = i/FPS
    t, reset, rv, g2 = state(s)
    rs, ps, end1 = streamline(t, reset, rv*1.05)
    arm_end = rs[-1]
    dt = advance(t, reset, rv, arm_end)
    axG.set_xlim(-rv, rv); axG.set_ylim(-rv, rv)
    # arms
    m1 = rs <= min(R2, end1) if reset else rs <= end1
    r2lim = R2 + (arm_end - R2)*g2
    m2 = (rs > R2) & (rs <= r2lim) if reset else np.zeros_like(rs, bool)
    for k, off in enumerate((0, np.pi)):
        arms[k].set_data(*pol(rs[m1], ps[m1]+off)); arms[k+2].set_data(*pol(rs[m2], ps[m2]+off))
    r_c1, r_c2 = rc1(t), rc2(t)
    lrc1.set_data(*pol(np.full_like(th, r_c1), th)) if r_c1 < rv*1.5 else lrc1.set_data([], [])
    lrc2.set_data(*pol(np.full_like(th, r_c2), th)) if (reset and r_c2 < rv*1.5) else lrc2.set_data([], [])
    ring_a = 0.25 + 0.65*ease(s, S_R0, S_R0+0.8)
    ring.set_alpha(ring_a); vR2.set_alpha(ring_a)
    fl = ease(s, S_R0, S_R0+1.6)
    flash.set_sizes([0 if (fl == 0 or fl == 1) else (R2*(1+1.5*fl)/rv*300)**2])
    flash.set_alpha(1-fl)
    # tracers
    sc_bg.set_offsets(np.c_[pol(tr_r, tr_p)])
    ap = np.interp(arm_r, rs, ps) + np.pi*arm_id
    x, y = pol(arm_r + 0.03*np.sin(7.3*np.arange(arm_r.size)), ap)
    inner = arm_r <= R2
    vis = inner | (arm_r <= r2lim)
    sc_1.set_offsets(np.c_[x[vis & inner], y[vis & inner]])
    sc_2.set_offsets(np.c_[x[vis & ~inner], y[vis & ~inner]])
    # velocity panel
    axV.set_xlim(0, rv)
    if reset:
        a = np.linspace(0.005, R2, 300); b = np.linspace(R2*(1+1e-6), rv, 400)
        va, ra = vel(a, t, True); vb, rb = vel(b, t, True)
        lvo.set_data(a, va); lvr.set_data(a, ra); lvo2.set_data(b, vb); lvr2.set_data(b, rb)
        top = max(va.max(), vb.max(), ra.max(), rb.max()); bot = min(ra.min(), rb.min(), 0)
    else:
        a = np.linspace(0.005, rv, 600); va, ra = vel(a, t, False)
        lvo.set_data(a, va); lvr.set_data(a, ra); lvo2.set_data([], []); lvr2.set_data([], [])
        top = max(va.max(), ra.max()); bot = min(ra.min(), 0)
    axV.set_ylim(bot - 0.08*(top-bot), top + 0.25*(top-bot))
    vrc1.set_xdata([r_c1, r_c1]); vrc1.set_alpha(0 if reset else 1)
    # timeline
    tt = tg[tg <= t]; tr1.set_data(tt, rc1(tt)); mk1.set_data([t], [rc1(t)])
    if reset:
        tt2 = tg[(tg >= T_ON) & (tg <= t)]; tr2.set_data(np.r_[T_ON, tt2], rc2(np.r_[T_ON, tt2])); mk2.set_data([t], [r_c2])
    else: tr2.set_data([], []); mk2.set_data([], [])
    # text
    l2 = f"M₂ = {M2(t):.2e} M⊙   r_c,2 = {r_c2:6.1f} kpc" if reset else "M₂ —  (L₂ not yet active)"
    info.set_text(f"t = {t:6.3f} Gyr   z = {abs(z_of_t(t)):5.2f}   H = {H_of_t(t):6.4f} km/s/kpc\n"
                  f"M₁ = {M1(t):.2e} M⊙   r_c,1 = {r_c1:6.2f} kpc   (R₁={R1:.2f}, R₂={R2:.2f} kpc)\n"
                  f"{l2}    tracer clock {dt:4.1f} Myr/frame")
    if s < 5:      caption.set_text("Proto-galaxy: one Lagrangian L₁ — inflow spiral ends at its critical circle r_c,1")
    elif s < S_R0: caption.set_text("M₁ grows ∝ t^1.5 while H(t) falls: r_c,1 expands toward R₂")
    elif s < 13:   caption.set_text("Onset, Eq. (19): r_c,1 = R₂ — v_esc = v_H at the ring")
    elif s < S_B0: caption.set_text("Reset: bulge–bar becomes the new bulge — L₂ drives a disk spiral from R₂")
    elif s < S_B1: caption.set_text("Nested spiral: fossil L₁ bar + ring + L₂ disk evolve together")
    else:          caption.set_text("Today (z = 0): the double-Lagrangian galaxy")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        for i in range(NF):
            frame(i)
            if i in (90, 320, 380, 440, 880): fig.savefig(f"rs_{i}.png", facecolor=BG)
    else:
        w = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1,
                         extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"])
        with w.saving(fig, video("bulge_reset_nested_spiral.mp4"), dpi=120):
            for i in range(min(NF, MAX_FRAMES)): frame(i); w.grab_frame()
