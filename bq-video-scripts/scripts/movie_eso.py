from paths import video, cache, MAX_FRAMES   # sets the working directory; see paths.py
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from model_eso import *

B = json.load(open(cache("eso_best.json"))); S1, S3 = B["single"], B["phi"]
PATH = np.array(json.load(open(cache("eso_lm_single.json"))))
PROF = np.load(cache("eso_phi_profile.npy"))
LS = np.load(cache("eso_R_phi_landscape.npz")); Rg, Pg, L = LS["Rg"], LS["Pg"], LS["L"]
R1, M1 = S1["x"]; R3, M3, P3 = S3["x"]
FPS = 30
T = dict(title=(0, 7), data=(7, 14), single=(14, 29), idea=(29, 40), scan=(40, 56), result=(56, 70), corr=(70, 80), note=(80, 92))
NF = int(92*FPS)
ease = lambda u: 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0, 1))
def prog(s, a, b): return ease((s - a)/(b - a))

BG, FG, MUT, EDGE = "#070a14", "#e7ebf3", "#8f9ab2", "#2c3445"
CS, CP, CPHI = "#c8a2ff", "#f2a24a", "#feca57"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.edgecolor": EDGE})
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
axV = fig.add_axes([0.06, 0.36, 0.53, 0.49], facecolor=BG)
axR = fig.add_axes([0.06, 0.09, 0.53, 0.18], facecolor=BG)
axP = fig.add_axes([0.665, 0.47, 0.31, 0.38], facecolor=BG)
axC = fig.add_axes([0.665, 0.47, 0.31, 0.38], facecolor=BG)
info = fig.text(0.665, 0.39, "", va="top", fontsize=11.5, family="DejaVu Sans Mono", linespacing=1.5)
title = fig.text(0.03, 0.94, "", fontsize=22, weight="bold")
sub = fig.text(0.03, 0.912, "", fontsize=12.5, color=MUT)
eq = fig.text(0.06, 0.872, "", fontsize=12.3)
card = fig.text(0.5, 0.5, "", ha="center", va="center", multialignment="left", fontsize=14.5, linespacing=1.6)
fig.text(0.03, 0.015, "Data: SPARC (Lelli, McGaugh & Schombert 2016), ESO079–G014 (NGC 360), n = 15. Fit in v² with σ(v²) = 2V·σ_V, "
         "H_z = 2.276×10⁻¹⁸ s⁻¹ fixed. M⊙ = 1.989×10³⁰ kg. Reproduces the paper’s Tables XIV–XV.", fontsize=9.3, color=MUT)

XM = 17.5; rr = np.linspace(0.02, XM, 800)
axV.set_xlim(0, XM); axV.set_ylim(-500, 36000); axV.set_xlabel("r (kpc)"); axV.grid(alpha=0.1)
axV.axhline(0, color=EDGE, lw=0.8)
ebv2 = axV.errorbar(r, v2, yerr=e2, fmt="none", ecolor="#9aa3b5", elinewidth=1, capsize=2.5, zorder=3)
ebV = axV.errorbar(r, V, yerr=eV, fmt="none", ecolor="#9aa3b5", elinewidth=1, capsize=2.5, zorder=3)
pts = axV.scatter(r, v2, s=30, color=FG, zorder=5)
lS, = axV.plot([], [], color=CS, lw=2.4, zorder=4); lSd, = axV.plot([], [], color=CS, lw=1.4, ls="--", alpha=0.8, zorder=4)
lP, = axV.plot([], [], color=CP, lw=2.6, zorder=4)
lphi, = axV.plot([], [], color=CPHI, lw=1.3, ls=":", zorder=3)
arr = axV.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=CPHI, lw=2))
phitxt = axV.text(0, 0, "", color=CPHI, fontsize=11)
vR = axV.axvline(0, color=MUT, ls="--", lw=1); vR.set_visible(False); labR = axV.text(0, 34200, "", color=MUT, fontsize=10.5)
leg = axV.text(0.015, 0.97, "", transform=axV.transAxes, ha="left", va="top", fontsize=11, linespacing=1.5)
axR.set_xlim(0, XM); axR.axhline(0, color=MUT, lw=0.8); axR.set_xlabel("r (kpc)"); axR.set_ylabel("(data − model)/σ", fontsize=10)
axR.grid(alpha=0.1); axR.set_ylim(-3.2, 3.2)
W = 0.32
barsS = axR.bar(r - W/2, np.zeros(n), width=W, color=CS); barsP = axR.bar(r + W/2, np.zeros(n), width=W, color=CP)
# profile panel chi2(Phi)
axP.set_xlim(0, 3600); axP.set_ylim(0, 16); axP.set_xlabel("Φ_BH ((km/s)²)"); axP.set_ylabel("χ² (R, M refitted)")
axP.set_title("Profile scan over the offset Φ_BH", fontsize=12, loc="left"); axP.grid(alpha=0.1)
lprof, = axP.plot([], [], color=CPHI, lw=2); mprof, = axP.plot([], [], "o", color="#ffffff", ms=8, mec=CPHI, mew=2)
band = axP.axhspan(0, 0, color=CP, alpha=0); ptxt = axP.text(0, 0, "", fontsize=10)
lmin, = axP.plot([], [], color=CP, lw=1, ls="--")
# (R, Phi) landscape
cf = axC.contourf(Rg, Pg, np.log10(L), levels=22, cmap="magma")
axC.contour(Rg, Pg, L - L.min(), levels=[2.30, 6.18], colors="#ffffff", linewidths=[1.0, 0.6], alpha=0.7)
axC.axhline(0, color=CS, lw=1.2, ls="--"); axC.text(4.25, 80, "Φ_BH = 0 (single-L)", color=CS, fontsize=9.5)
axC.plot([R3], [P3], "o", color=CP, ms=9); axC.plot([R1], [0], "o", color=CS, ms=8)
axC.set_xlabel("R (kpc)"); axC.set_ylabel("Φ_BH ((km/s)²)"); axC.set_title("χ²(R, Φ_BH) with M profiled; 1σ, 2σ contours", fontsize=11.5, loc="left")

def ebvis(e, on):
    for a in list(e[1]) + list(e[2]): a.set_visible(on)
def clear():
    for o in (lS, lSd, lP, lphi, lprof, mprof, lmin): o.set_data([], [])
    vR.set_visible(False); labR.set_text(""); phitxt.set_text(""); arr.set_visible(False); leg.set_text("")
    eq.set_text(""); info.set_text(""); card.set_text(""); ptxt.set_text(""); band.set_alpha(0)
    for b in list(barsS) + list(barsP): b.set_height(0)
    pts.set_offsets(np.c_[r, v2])
def vis(V_=True, R_=True, P_=False, C_=False):
    axV.set_visible(V_); axR.set_visible(R_); axP.set_visible(P_); axC.set_visible(C_)
def setR(R):
    vR.set_visible(True); vR.set_xdata([R, R]); labR.set_position((R+0.12, 34200)); labR.set_text("R")
def bars(bs, resv):
    for b, h in zip(bs, np.clip(resv, -3.1, 3.1)): b.set_height(h)
def show_phi(phi, x=0.6):
    lphi.set_data([0, XM], [phi, phi]); arr.set_visible(True)
    arr.xy = (x, phi); arr.set_position((x, 0)); phitxt.set_position((11.2, phi+700)); phitxt.set_text(f"Φ_BH = {phi:.0f} (km/s)²\n→ v(0) = {np.sqrt(max(phi,0)):.0f} km/s")

def frame(i):
    s = i/FPS; clear(); vis(); ebvis(ebv2, True); ebvis(ebV, False); info.set_position((0.665, 0.39))
    axV.set_ylim(-500, 36000); axV.set_ylabel("v² (km² s⁻²)")
    if s < T["title"][1]:
        vis(False, False)
        title.set_text("When a constant offset earns its place: ESO079–G014 (NGC 360)"); sub.set_text("Single-Lagrangian CL fit, with and without Φ_BH")
        card.set_text("1  The data, in v²\n2  Single-L fit (R, M)\n3  The idea: a constant offset Φ_BH added to v²\n"
                      "4  Profile scan over Φ_BH\n5  Result: does the extra parameter pay for itself?\n6  How Φ_BH trades off against R\n7  Reading the fit")
        card.set_fontsize(16); return
    if s < T["data"][1]:
        vis(True, False); title.set_text("1 · The data, written in v²"); sub.set_text("15 SPARC points; σ(v²) = 2V·σ_V")
        if s < 10.5:
            ebvis(ebv2, False); ebvis(ebV, True); pts.set_offsets(np.c_[r, V]); axV.set_ylim(0, 200); axV.set_ylabel("V (km/s)")
            eq.set_text("V(r) with 1σ errors, 0.41 – 16.67 kpc")
        else: eq.set_text("v² = V²,  σ(v²) = 2V σ_V — the CL model is written for v²")
        return
    if s < T["single"][1]:
        vis(True, True); title.set_text("2 · Single-L fit (R, M)"); sub.set_text("Levenberg–Marquardt, 14 real accepted steps")
        eq.set_text("r ≤ R:  v² = ½X²(r/R)²      r > R:  v² = 3/2·X² − (√(2GM/r) − H_z r)²      X = √(2GM/R) − H_z R")
        u = prog(s, T["single"][0]+0.5, T["single"][1]-4); k = int(round(u*(len(PATH)-1))); R, M, chi = PATH[k]
        lS.set_data(rr, model(rr, R, M)); setR(R); bars(barsS, res([R, M]))
        info.set_text(f"single-L, step {k:2d}/{len(PATH)-1}\nR = {R:.3f} kpc\nM = {M*C10:.3f}×10¹⁰ M⊙\nχ² = {chi:8.2f}")
        if s > T["single"][1]-4:
            info.set_text(info.get_text() + f"\n\nχ²ν = {S1['chi2nu']:.2f}: errors realistic\nRMS_rel = {S1['RMS']*100:.1f}%")
            leg.set_text("inner points sit above the curve,\nmid-disk points below: a coherent pattern")
        return
    if s < T["idea"][1]:
        vis(True, False); title.set_text("3 · The idea: a constant offset Φ_BH added to v²")
        eq.set_text("v²(r) = v²_CL(r; R, M) + Φ_BH        one extra parameter (k = 3)")
        u = s - T["idea"][0]
        if u < 5.5:
            sub.set_text("Lifting the single-L curve by Φ_BH alone (R, M unchanged) fixes the centre but overshoots the outer disk")
            phi = P3*prog(s, T["idea"][0]+0.5, T["idea"][0]+3.5)
            lSd.set_data(rr, model(rr, R1, M1)); lS.set_data(rr, model(rr, R1, M1, phi)); show_phi(phi); setR(R1)
        else:
            sub.set_text("So R and M readjust together with Φ_BH: the curve bends differently, not just upward")
            w = prog(s, T["idea"][0]+5.8, T["idea"][0]+9.5)
            R = R1*(1-w) + R3*w; M = M1*(1-w) + M3*w
            lSd.set_data(rr, model(rr, R1, M1)); lP.set_data(rr, model(rr, R, M, P3)); show_phi(P3); setR(R)
            leg.set_text("dashed purple: single-L\norange: R, M readjusting with Φ_BH")
        return
    if s < T["scan"][1]:
        vis(True, True, True); title.set_text("4 · Profile scan over Φ_BH"); sub.set_text("At each Φ_BH, R and M are refitted; χ² traces a smooth parabola")
        kmin = int(np.argmin(PROF[:, 1]))
        if s < T["scan"][1]-5: u = prog(s, T["scan"][0]+0.5, T["scan"][1]-5); k = int(round(u*(len(PROF)-1)))
        else: w = prog(s, T["scan"][1]-5, T["scan"][1]-3); k = int(round((len(PROF)-1)*(1-w) + kmin*w))
        phi, chi, R, M = PROF[k]
        lSd.set_data(rr, model(rr, R1, M1)); lP.set_data(rr, model(rr, R, M, phi)); show_phi(phi); setR(R)
        bars(barsS, res([R1, M1])); bars(barsP, res([R, M, phi]))
        lprof.set_data(PROF[:max(k, int(round(prog(s, T["scan"][0]+0.5, T["scan"][1]-5)*(len(PROF)-1))))+1, 0], PROF[:max(k, int(round(prog(s, T["scan"][0]+0.5, T["scan"][1]-5)*(len(PROF)-1))))+1, 1]); mprof.set_data([phi], [chi])
        if s > T["scan"][1]-3:
            cm = S3["chi2"]; band.set_alpha(0.15); band.set_xy([[0, cm], [0, cm+1], [1, cm+1], [1, cm], [0, cm]]) if False else None
            lmin.set_data([0, 3600], [cm+1, cm+1])
            ptxt.set_position((150, 12.6)); ptxt.set_text(f"minimum Φ_BH = {P3:.0f}\nΔχ² = 1 → ±{S3['sd'][2]:.0f}\nχ²(0) − χ²min = {S1['chi2']-cm:.1f}  (≈ 3.2σ)")
        info.set_text(f"Φ_BH = {phi:6.0f} (km/s)²\nR = {R:.3f} kpc\nM = {M*C10:.3f}×10¹⁰ M⊙\nχ² = {chi:7.3f}")
        leg.set_text("purple bars: single-L residuals\norange bars: with Φ_BH")
        return
    if s < T["result"][1]:
        vis(True, True, False); info.set_position((0.665, 0.84)); title.set_text("5 · Result: does the extra parameter pay for itself?")
        sub.set_text("Yes — χ² drops by 10.1 for one parameter; AIC and BIC both prefer Φ_BH")
        lSd.set_data(rr, model(rr, R1, M1)); lP.set_data(rr, model(rr, R3, M3, P3)); show_phi(P3); setR(R3)
        bars(barsS, res([R1, M1])); bars(barsP, res([R3, M3, P3]))
        info.set_text(f"                single-L    +Φ_BH\n"
                      f"R (kpc)          {R1:6.3f}    {R3:6.3f}\nM (10¹⁰ M⊙)      {M1*C10:6.3f}    {M3*C10:6.3f}\n"
                      f"Φ_BH ((km/s)²)      —     {P3:5.0f}±{S3['sd'][2]:.0f}\n"
                      f"χ²               {S1['chi2']:6.2f}    {S3['chi2']:6.2f}\nχ²ν              {S1['chi2nu']:6.2f}    {S3['chi2nu']:6.2f}\n"
                      f"AIC              {S1['AIC']:6.2f}    {S3['AIC']:6.2f}\nBIC              {S1['BIC']:6.2f}    {S3['BIC']:6.2f}\n"
                      f"RMS_rel          {S1['RMS']*100:5.1f}%    {S3['RMS']*100:5.1f}%\n\n"
                      f"ΔAIC = {S1['AIC']-S3['AIC']:.1f}   ΔBIC = {S1['BIC']-S3['BIC']:.1f}  (strong)\n"
                      f"proxies: MOND AIC 15.03 · ISO AIC 23.09")
        leg.set_text("dashed purple: single-L (R, M)\norange: single-L + Φ_BH")
        return
    if s < T["corr"][1]:
        vis(True, False, False, True); title.set_text("6 · How Φ_BH trades off against R")
        sub.set_text("The valley in (R, Φ_BH) is tilted: a larger offset goes with a larger bulge radius (correlation +0.57)")
        lSd.set_data(rr, model(rr, R1, M1)); lP.set_data(rr, model(rr, R3, M3, P3)); setR(R3)
        info.set_text(f"correlations of the 3-parameter fit\n  R–M      {S3['corr'][0][1]:+.2f}\n  R–Φ_BH   {S3['corr'][0][2]:+.2f}\n  M–Φ_BH   {S3['corr'][1][2]:+.2f}\n\n"
                      f"the single-L point (Φ_BH = 0)\nlies outside the 2σ contour")
        leg.set_text("dashed purple: single-L\norange: single-L + Φ_BH")
        return
    vis(False, False); title.set_text("7 · Reading the fit"); sub.set_text("")
    card.set_text(
        "•  One constant added to v² lowers χ² from 14.5 to 4.4 and RMS_rel from 34.5% to 14.8%;\n"
        "    ΔAIC = 8.1 and ΔBIC = 7.4 count as strong evidence despite the extra parameter.\n\n"
        "•  Φ_BH = 1590 ± 476 (km/s)² is 3.3σ from zero, and the profile scan shows a clean parabola —\n"
        "    the offset is constrained by the data, not by the prior.\n\n"
        "•  The single-L baseline has χ²ν = 1.12, so its misfit is real; with Φ_BH, χ²ν = 0.37 — close to\n"
        "    fitting the noise, so further extensions would not be justified here.\n\n"
        "•  ESO079–G014 is the exception: in four of the six galaxies tested with a pure Φ_BH variant,\n"
        "    the offset does not pay for its extra parameter.\n\n"
        "•  Physically, a constant in v² sets a central velocity floor v(0) ≈ 40 km/s; a compact point mass\n"
        "    would instead add GM/r. Φ_BH is best read as an effective inner energy term, as in the paper.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        for t in (4, 12, 27, 32, 37, 54, 63, 75, 86): frame(int(t*FPS)); fig.savefig(f"e_{t}.png", facecolor=BG)
    else:
        w = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1, extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"])
        with w.saving(fig, video("eso079g014_phiBH_fit.mp4"), dpi=120):
            for i in range(min(NF, MAX_FRAMES)): frame(i); w.grab_frame()
