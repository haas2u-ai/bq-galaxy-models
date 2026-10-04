from paths import video, cache, MAX_FRAMES   # sets the working directory; see paths.py
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from model2574 import *

c9 = MUNIT/1.989e30/1e9
B = json.load(open(cache("ic2574_best.json"))); PV = B["paper_valley"]
PROF = np.load(cache("ic2574_profile.npy"))                 # R2, chi2, R1, M1, M2 (model units)
SL = np.array(json.load(open(cache("ic2574_lm_single.json"))))
R1b, M1b, R2b, M2b = B["R1"], B["M1_model"], B["R2"], B["M2_model"]
n = len(r)
chi_single = SL[-1, 2]
FPS = 30
T = dict(title=(0, 7), data=(7, 14), single=(14, 27), idea=(27, 40), scan=(40, 62), valley=(62, 72), result=(72, 84), note=(84, 96))
NF = int(96*FPS)
ease = lambda u: 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0, 1))
def prog(s, a, b): return ease((s - a)/(b - a))

BG, FG, MUT, EDGE = "#070a14", "#e7ebf3", "#8f9ab2", "#2c3445"
C1, C2, CS, CJ = "#f2a24a", "#48dbfb", "#c8a2ff", "#feca57"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.edgecolor": EDGE})
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
axV = fig.add_axes([0.06, 0.36, 0.53, 0.49], facecolor=BG)
axR = fig.add_axes([0.06, 0.09, 0.53, 0.18], facecolor=BG)
axP = fig.add_axes([0.665, 0.45, 0.31, 0.40], facecolor=BG)
info = fig.text(0.665, 0.37, "", va="top", fontsize=11.5, family="DejaVu Sans Mono", linespacing=1.5)
title = fig.text(0.03, 0.94, "", fontsize=22, weight="bold")
sub = fig.text(0.03, 0.912, "", fontsize=12.5, color=MUT)
eq = fig.text(0.06, 0.872, "", fontsize=12.3)
card = fig.text(0.5, 0.5, "", ha="center", va="center", multialignment="left", fontsize=14.5, linespacing=1.6)
fig.text(0.03, 0.015, "Data: SPARC (Lelli, McGaugh & Schombert 2016), IC 2574, n = 34. Fit in v² with σ(v²) = 2V·σ_V, H_z = 2.27×10⁻¹⁸ s⁻¹ fixed. "
         "Two-L ‘<’ convention: first CL curve for r < R₂, second from R₂ on. M⊙ = 1.989×10³⁰ kg.", fontsize=9.3, color=MUT)

XM = 10.8; rr = np.linspace(0.02, XM, 900)
axV.set_xlim(0, XM); axV.set_ylim(0, 5600); axV.set_xlabel("r (kpc)"); axV.grid(alpha=0.1)
def errbars(y, e, col):
    return axV.errorbar(r, y, yerr=e, fmt="none", ecolor=col, elinewidth=1, capsize=2.2, zorder=3)
eb2 = errbars(v2, e2, "#9aa3b5"); ebV = errbars(V, eV, "#9aa3b5")
pts = axV.scatter(r, v2, s=26, color=FG, zorder=5)
lS, = axV.plot([], [], color=CS, lw=2.3, zorder=4)
l1, = axV.plot([], [], color=C1, lw=2.6, zorder=4); l1d, = axV.plot([], [], color=C1, lw=1.2, ls="--", alpha=0.55, zorder=3)
l2, = axV.plot([], [], color=C2, lw=2.6, zorder=4); l2d, = axV.plot([], [], color=C2, lw=1.2, ls="--", alpha=0.55, zorder=3)
jump, = axV.plot([], [], color=CJ, lw=2, zorder=6)
vR2 = axV.axvline(0, color=CJ, ls="-.", lw=1.3); vR2.set_visible(False)
vR1 = axV.axvline(0, color=MUT, ls="--", lw=1); vR1.set_visible(False)
labR2 = axV.text(0, 5350, "", color=CJ, fontsize=11); labR1 = axV.text(0, 5350, "", color=MUT, fontsize=10.5)
shade = axV.axvspan(0, 1, color=C2, alpha=0.0, zorder=0)
leg = axV.text(0.015, 0.97, "", transform=axV.transAxes, va="top", fontsize=11, linespacing=1.5)
axR.set_xlim(0, XM); axR.axhline(0, color=MUT, lw=0.8); axR.set_xlabel("r (kpc)"); axR.set_ylabel("(data − model)/σ", fontsize=10)
axR.grid(alpha=0.1); axR.set_ylim(-9, 9)
bars = axR.bar(r, np.zeros(n), width=0.2, color=MUT)
# profile panel
axP.set_xlim(4.5, 7.5); axP.set_yscale("log"); axP.set_ylim(30, 400)
axP.set_yticks([40, 60, 100, 200, 300]); axP.set_yticklabels(["40", "60", "100", "200", "300"]); axP.minorticks_off()
axP.set_xlabel("R₂ (kpc)"); axP.set_ylabel("χ² (masses and R₁ refitted)")
axP.set_title("Profile scan over the reset radius R₂", fontsize=12, loc="left")
for rd in r[(r > 4.5) & (r < 7.5)]: axP.axvline(rd, color=EDGE, lw=0.8)
axP.text(4.55, 33, "grey lines: data radii", color=MUT, fontsize=9)
lp, = axP.plot([], [], color=CJ, lw=1.8, drawstyle="steps-post"); mp, = axP.plot([], [], "o", color="#ffffff", ms=8, mec=CJ, mew=2)
v1m, = axP.plot([], [], "o", color=C2, ms=10); v2m, = axP.plot([], [], "o", color=C1, ms=10, mfc="none", mew=2)
vtxt = axP.text(0, 0, "", fontsize=10, color=FG)

def curves(R1, M1, R2, M2):
    a = single(rr, R1, M1); b = 1e-6*(1.5*X_(R2, M2)**2 - vr_(rr, M2)**2)
    # L2's own bulge branch inside R2 (unused): the 'new bulge' of mass M2
    b_in = single(rr, R2, M2)
    return a, b, b_in
def set_two(R1, M1, R2, M2, ghosts=True):
    a, b, b_in = curves(R1, M1, R2, M2); m = rr < R2
    l1.set_data(rr[m], a[m]); l2.set_data(rr[~m], b[~m])
    if ghosts: l1d.set_data(rr[~m], a[~m]); l2d.set_data(rr[m], b_in[m])
    ya = single(np.array([R2]), R1, M1)[0]; yb = 1e-6*(1.5*X_(R2, M2)**2 - vr_(np.array([R2]), M2)[0]**2)
    jump.set_data([R2, R2], [ya, yb])
    vR2.set_visible(True); vR2.set_xdata([R2, R2]); labR2.set_position((R2+0.07, 5350)); labR2.set_text("R₂ (reset)")
    vR1.set_visible(True); vR1.set_xdata([R1, R1]); labR1.set_position((R1+0.07, 120)); labR1.set_text("R₁")
    inner = r < R2; pts.set_color([C1 if x else C2 for x in inner])
    res = res2lt([R1, M1, R2, M2])
    return res, inner
def set_bars(res, cols):
    for bb, h, col in zip(bars, np.clip(res, -8.8, 8.8), cols): bb.set_height(h); bb.set_color(col)
def clear():
    for o in (lS, l1, l1d, l2, l2d, jump, lp, mp, v1m, v2m): o.set_data([], [])
    vR2.set_visible(False); vR1.set_visible(False); labR2.set_text(""); labR1.set_text(""); vtxt.set_text("")
    leg.set_text(""); eq.set_text(""); info.set_text(""); card.set_text(""); shade.set_alpha(0)
    set_bars(np.zeros(n), [MUT]*n); pts.set_color(FG); pts.set_offsets(np.c_[r, v2])
def vis(V_=True, R_=True, P_=True):
    axV.set_visible(V_); axR.set_visible(R_); axP.set_visible(P_)
def ebv(e, on):
    for a in list(e[1]) + list(e[2]): a.set_visible(on)

def frame(i):
    s = i/FPS; clear(); vis(); ebv(eb2, True); ebv(ebV, False)
    axV.set_ylim(0, 5600); axV.set_ylabel("v² (km² s⁻²)")
    if s < T["title"][1]:
        vis(False, False, False)
        title.set_text("Fitting a two-Lagrangian CL model: IC 2574"); sub.set_text("Coddington’s nebula (SABm), step by step")
        card.set_text("1  The data, in v²\n2  One Lagrangian is not enough\n3  The idea: a discontinuous reset at R₂\n"
                      "4  Profile scan over R₂ — a staircase, not a smooth valley\n5  Two near-equal valleys\n6  Result: two curves, one χ²\n7  Reading the fit")
        card.set_fontsize(16); return
    if s < T["data"][1]:
        vis(True, False, False); title.set_text("1 · The data, written in v²"); sub.set_text("34 SPARC points; σ(v²) = 2V·σ_V")
        if s < 10.5:
            ebv(eb2, False); ebv(ebV, True); pts.set_offsets(np.c_[r, V]); axV.set_ylim(0, 80); axV.set_ylabel("V (km/s)")
            eq.set_text("V(r) with 1σ errors, 0.85 – 10.23 kpc")
        else: eq.set_text("v² = V²,  σ(v²) = 2V σ_V  —  the CL model is written for v²")
        return
    if s < T["single"][1]:
        vis(True, True, False); title.set_text("2 · One Lagrangian is not enough"); sub.set_text("Single-L (R, M) fit by Levenberg–Marquardt")
        u = prog(s, T["single"][0]+0.5, T["single"][1]-3.5); k = int(round(u*(len(SL)-1)))
        R, M, chi = SL[k]; lS.set_data(rr, single(rr, R, M))
        res = res1([R, M]); set_bars(res, [CS]*n)
        vR1.set_visible(True); vR1.set_xdata([R, R]); labR1.set_position((R+0.07, 120)); labR1.set_text("R")
        info.set_text(f"single-L, step {k:2d}/{len(SL)-1}\nR = {R:.3f} kpc\nM = {M*c9:.3f}×10⁹ M⊙\nχ² = {chi:8.2f}")
        if s > T["single"][1]-3.5:
            eq.set_text(f"χ² = {chi_single:.0f} for 34 points (χ²ν ≈ 13.8): coherent residuals across 5–8 kpc, where the curve steepens again")
        return
    if s < T["idea"][1]:
        vis(True, False, False); title.set_text("3 · The idea: a discontinuous reset at R₂")
        u = s - T["idea"][0]
        if u < 4:
            sub.set_text("First CL curve (R₁, M₁): it serves the data up to R₂ and ends there")
            a, _, _ = curves(R1b, M1b, R2b, M2b); m = rr < R2b
            l1.set_data(rr[m], a[m]); l1d.set_data(rr[~m], a[~m])
            vR2.set_visible(True); vR2.set_xdata([R2b, R2b]); labR2.set_position((R2b+0.07, 5350)); labR2.set_text("R₂")
            vR1.set_visible(True); vR1.set_xdata([R1b, R1b]); labR1.set_position((R1b+0.07, 120)); labR1.set_text("R₁")
            leg.set_text("orange: first CL curve (r < R₂)\ndashed: its continuation, not used")
        elif u < 8:
            sub.set_text("Reset at R₂: everything inside R₂ becomes the new bulge, of mass M₂")
            res, inner = set_two(R1b, M1b, R2b, M2b); l2.set_data([], []); jump.set_data([], [])
            shade.set_alpha(0.10*prog(s, T["idea"][0]+4, T["idea"][0]+5.5)); shade.set_x(0); shade.set_width(R2b)
            leg.set_text("blue dashed: the new bulge of the second\nLagrangian (M₂ within R₂) — defines it, not fitted to data")
        else:
            sub.set_text("Second CL curve (R₂, M₂) starts at R₂; the jump at R₂ is the reset")
            res, inner = set_two(R1b, M1b, R2b, M2b)
            shade.set_alpha(0.10); shade.set_x(0); shade.set_width(R2b)
            leg.set_text("orange: first CL curve, r < R₂\nblue: second CL curve, r ≥ R₂\nyellow: discontinuity at R₂")
        eq.set_text("v²(r) = v²_CL(r; R₁, M₁)  for r < R₂        v²(r) = v²_CL,disk(r; R₂, M₂)  for r ≥ R₂")
        return
    if s < T["valley"][0]:
        title.set_text("4 · Profile scan over R₂"); sub.set_text("At each R₂, R₁, M₁, M₂ are refitted; χ² jumps whenever R₂ passes a data radius")
        u = prog(s, T["scan"][0]+0.5, T["scan"][1]-1); R2 = 4.5 + 3.0*u
        k = int(np.clip(np.searchsorted(PROF[:, 0], R2), 0, len(PROF)-1)); row = PROF[k]
        res, inner = set_two(row[2], row[3], row[0], row[4])
        set_bars(res, [C1 if x else C2 for x in inner])
        lp.set_data(PROF[:k+1, 0], PROF[:k+1, 1]); mp.set_data([row[0]], [row[1]])
        info.set_text(f"R₂ = {row[0]:.2f} kpc\nR₁ = {row[2]:.3f} kpc\nM₁ = {row[3]*c9:.3f}×10⁹ M⊙\nM₂ = {row[4]*c9:.3f}×10⁹ M⊙\nχ² = {row[1]:8.2f}\n"
                      f"points: {int(inner.sum())} first curve, {int((~inner).sum())} second")
        eq.set_text("orange points/bars: served by the first CL curve   ·   blue: by the second")
        return
    if s < T["valley"][1]:
        title.set_text("5 · Two near-equal valleys"); sub.set_text("Both minima put R₂ exactly on a data radius, with that point given to the second curve")
        lp.set_data(PROF[:, 0], PROF[:, 1])
        v1m.set_data([R2b], [B["chi2"]]); v2m.set_data([PV["R2"]], [PV["chi2"]])
        vtxt.set_position((4.6, 260)); vtxt.set_text(f"● global min  R₂ = {R2b:.2f},  χ² = {B['chi2']:.2f}\n○ paper       R₂ = {PV['R2']:.2f},  χ² = {PV['chi2']:.2f}\nΔχ² = {PV['chi2']-B['chi2']:.2f}: statistically equivalent")
        u = prog(s, T["valley"][0]+3, T["valley"][0]+6)
        p = np.array([R1b, M1b, R2b, M2b])*(1-u) + np.array([PV["R1"], PV["M1_model"], PV["R2"], PV["M2_model"]])*u
        if s > T["valley"][0]+7.5: p = np.array([R1b, M1b, R2b, M2b])
        res, inner = set_two(*p); set_bars(res, [C1 if x else C2 for x in inner])
        info.set_text(f"shown: R₂ = {p[2]:.2f} kpc\nM₂ = {p[3]*c9:.3f}×10⁹ M⊙\n\nM₂ shifts by 12% between the valleys:\nits uncertainty is set by R₂")
        return
    if s < T["result"][1]:
        title.set_text("6 · Result: two curves, one χ²"); sub.set_text("The first curve ends and the second starts at R₂; together they give the residuals and χ²")
        res, inner = set_two(R1b, M1b, R2b, M2b); set_bars(res, [C1 if x else C2 for x in inner])
        lp.set_data(PROF[:, 0], PROF[:, 1]); v1m.set_data([R2b], [B["chi2"]])
        chi1 = float(np.sum(res[inner]**2)); chi2_ = float(np.sum(res[~inner]**2))
        info.set_text(f"R₁ = {R1b:.3f} ± {B['sR1']:.3f} kpc\nM₁ = ({B['M1']:.3f} ± {B['sM1']:.3f})×10⁹ M⊙\n"
                      f"R₂ = {R2b:.2f} kpc  (on a data radius)\nM₂ = {B['M2']:.2f}×10⁹ M⊙  (at fixed R₂)\n"
                      f"χ² = {chi1:.1f} (first) + {chi2_:.1f} (second) = {B['chi2']:.2f}\n"
                      f"χ²ν = {B['chi2nu']:.2f}   AIC = {B['AIC']:.2f}   BIC = {B['BIC']:.2f}\n"
                      f"RMS_rel = {B['RMS']:.3f}\n\nsingle-L: χ² = {chi_single:.0f}, AIC = {chi_single+4:.0f}")
        eq.set_text("χ² = Σ [(v²_data − v²_model)/σ(v²)]²  over all 34 points, each served by exactly one CL curve")
        return
    vis(False, False, False); title.set_text("7 · Reading the fit"); sub.set_text("")
    card.set_text(
        "•  Two Lagrangians reduce χ² from 440 to 38.9 (ΔAIC ≈ 397, ΔBIC ≈ 394), and χ²ν = 1.30 —\n"
        "    unlike many SPARC fits, the error bars here match the scatter, so the improvement is meaningful.\n\n"
        "•  The reset makes χ²(R₂) a staircase: R₂ can only be located between data radii, and a local solver\n"
        "    alone can miss the best step. A profile scan over R₂ is required.\n\n"
        f"•  Global minimum R₂ = 5.97 kpc (χ² = 38.89); the paper’s solution R₂ = 6.26 kpc (χ² = 39.65)\n"
        "    lies in the neighbouring valley — equivalent at Δχ² = 0.77. M₂ moves from 3.16 to 3.55×10⁹ M⊙.\n\n"
        "•  The ‘<’ convention matters: assigning the point at 6.26 kpc to the first curve instead raises\n"
        "    χ² at the paper’s parameters to 69.8.\n\n"
        "•  IC 2574 has strong non-circular motions and HI holes (THINGS); part of what the second\n"
        "    Lagrangian absorbs may be kinematic disturbance rather than structure.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        for t in (4, 12, 25, 30, 35, 39, 48, 58, 67, 78, 90): frame(int(t*FPS)); fig.savefig(f"m_{t}.png", facecolor=BG)
    else:
        w = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1, extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"])
        with w.saving(fig, video("ic2574_two_lagrangian_fit.mp4"), dpi=120):
            for i in range(min(NF, MAX_FRAMES)): frame(i); w.grab_frame()
