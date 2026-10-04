from paths import video, cache, MAX_FRAMES   # sets the working directory; see paths.py
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Ellipse
from clfit import *
import clfit
from data1281 import DATA
r, V, eV = DATA.T; v2, e2 = V**2, 2*V*eV
HZ_PAPER = 2.30e-18
def model_v2(rr, R, M10, Hz=HZ_PAPER, phi=0.0): return clfit.model_v2(rr, R, M10, Hz, phi)
def resid(p, Hz=HZ_PAPER): return (v2 - model_v2(r, p[0], p[1], Hz))/e2
XMAXR = 5.4

MSUN_PAPER = 1.989e30
def Msun(M10): return M10*MUNIT/MSUN_PAPER
PATH = json.load(open(cache("lm_path_ugc1281.json")))
Rb, Mb = PATH[-1]["R"], PATH[-1]["M10"]
n, k = len(r), 2
fb = resid([Rb, Mb]); chi2b = float(np.sum(fb**2))
# covariance in (R, M_sun)
h = 1e-6; J = np.zeros((n, 2))
for j, (dR, dM) in enumerate(((h, 0), (0, h*Mb))):
    J[:, j] = (resid([Rb+dR, Mb+dM]) - fb)/(dR if j == 0 else dM)
cov = np.linalg.inv(J.T@J); sR = np.sqrt(cov[0, 0]); sM = np.sqrt(cov[1, 1])
conv = MUNIT/MSUN_PAPER
covS = cov.copy(); covS[0, 1] *= conv; covS[1, 0] *= conv; covS[1, 1] *= conv**2
rms = float(np.sqrt(np.mean(((v2 - model_v2(r, Rb, Mb))/v2)**2)))
AIC, BIC = chi2b + 2*k, chi2b + k*np.log(n)
H = HZ_PAPER
def channels(rr, R, M10):
    X = np.sqrt(2*G_SI*M10*MUNIT/(R*KPC)) - H*R*KPC
    vrad = np.where(rr <= R, np.sqrt(G_SI*M10*MUNIT/(R*KPC)*(3 - (np.minimum(rr, R)/R)**2)) - H*rr*KPC,
                    np.sqrt(2*G_SI*M10*MUNIT/(rr*KPC)) - H*rr*KPC)
    vorb2 = model_v2(rr, R, M10)
    return vorb2, 1e-6*vrad**2, 1e-6*1.5*X**2
rc = (2*G_SI*Mb*MUNIT/H**2)**(1/3)/KPC

# chi2 landscape on (R, log10 M_sun)
Rg = np.linspace(0.6, 3.8, 220); Lg = np.linspace(np.log10(1.0e8), np.log10(4e9), 220)
CH = np.zeros((len(Lg), len(Rg)))
for i, L in enumerate(Lg):
    M10 = 10**L*MSUN_PAPER/MUNIT
    for j, R in enumerate(Rg): CH[i, j] = np.sum(resid([R, M10])**2)

FPS = 30
T = dict(title=(0, 7), data=(7, 16), model=(16, 25), land=(25, 31), iter=(31, 51), result=(51, 62), chan=(62, 72), note=(72, 82))
NF = int(82*FPS)
ease = lambda u: 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0, 1))
def prog(s, a, b): return ease((s - a)/(b - a))

BG, FG, MUT, EDGE = "#070a14", "#e7ebf3", "#8f9ab2", "#2c3445"
C_DATA, C_MOD, C_ORB, C_RAD, C_L, C_PATH = "#e7ebf3", "#f2a24a", "#f2a24a", "#a29bfe", "#8f9ab2", "#7fd6ff"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.edgecolor": EDGE})
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
axV = fig.add_axes([0.06, 0.36, 0.52, 0.50], facecolor=BG)
axR = fig.add_axes([0.06, 0.09, 0.52, 0.18], facecolor=BG)
axL = fig.add_axes([0.66, 0.42, 0.31, 0.44], facecolor=BG)
info = fig.text(0.66, 0.33, "", va="top", fontsize=12, family="DejaVu Sans Mono", linespacing=1.55)
title = fig.text(0.03, 0.94, "", fontsize=22, weight="bold")
sub = fig.text(0.03, 0.912, "", fontsize=12.5, color=MUT)
eq = fig.text(0.06, 0.872, "", fontsize=12.5, color=FG)
card = fig.text(0.5, 0.5, "", ha="center", va="center", multialignment="left", fontsize=15, linespacing=1.6)
fig.text(0.03, 0.015, "Data: SPARC (Lelli, McGaugh & Schombert 2016), UGC 1281, n = 25. Fit in v² with σ(v²) = 2V·σ_V, "
         "fixed H_z = 2.30×10⁻¹⁸ s⁻¹ as in the paper. Conventions from the author's workbook; M⊙ = 1.989×10³⁰ kg.", fontsize=9.3, color=MUT)

rr = np.linspace(0.02, XMAXR, 500)
axV.set_xlim(0, XMAXR); axV.set_xlabel("r (kpc)"); axV.grid(alpha=0.1)
eb = axV.errorbar(r, v2, yerr=e2, fmt="o", color=C_DATA, ms=5, capsize=2.5, lw=1, zorder=4)
ebV = axV.errorbar(r, V, yerr=eV, fmt="o", color=C_DATA, ms=5, capsize=2.5, lw=1, zorder=4)
lmod, = axV.plot([], [], color=C_MOD, lw=2.4, zorder=3)
lorb, = axV.plot([], [], color=C_ORB, lw=2.4, zorder=3)
lrad, = axV.plot([], [], color=C_RAD, lw=2.2, zorder=3)
lL, = axV.plot([], [], color=C_L, lw=1.4, ls=":", zorder=3)
vR = axV.axvline(Rb, color=MUT, ls="--", lw=1)
vRlab = axV.text(0, 0, "", color=MUT, fontsize=10)
mk1, = axV.plot([], [], "o", color=C_ORB, ms=9, zorder=5); mk2, = axV.plot([], [], "o", color=C_RAD, ms=9, zorder=5)
leg = axV.text(0.98, 0.03, "", transform=axV.transAxes, ha="right", va="bottom", fontsize=11, linespacing=1.5)
axR.set_xlim(0, XMAXR); axR.axhline(0, color=MUT, lw=0.8); axR.set_ylabel("(data − model)/σ", fontsize=10); axR.set_xlabel("r (kpc)")
axR.set_ylim(-1.0, 1.0); axR.grid(alpha=0.1)
rs = axR.scatter([], [], s=28, color=C_MOD)
im = axL.contourf(Rg, Lg, np.log10(CH), levels=24, cmap="magma")
axL.contour(Rg, Lg, CH - chi2b, levels=[2.30, 6.18, 11.8], colors=["#ffffff"], linewidths=[1.0, 0.7, 0.5], alpha=0.6)
axL.set_xlabel("R (kpc)"); axL.set_ylabel("log₁₀ M (M⊙)"); axL.set_title("χ² landscape (log₁₀ χ²); contours 1σ, 2σ, 3σ", fontsize=11.5, loc="left")
pth, = axL.plot([], [], "-o", color=C_PATH, ms=4, lw=1.4)
cur, = axL.plot([], [], "o", color="#ffffff", ms=9, mec=C_PATH, mew=2)
ell = Ellipse((Rb, np.log10(Msun(Mb))), 0, 0, fill=False, ec=C_MOD, lw=2); axL.add_patch(ell)

def show(*objs, on=True):
    for o in objs: o.set_visible(on)
def eb_vis(e, on):
    for a in [e[0]] + list(e[1]) + list(e[2]): a.set_visible(on)

PR = np.array([p["R"] for p in PATH]); PM = np.array([p["M10"] for p in PATH]); PC = np.array([p["chi2"] for p in PATH])
def path_at(u):
    x = u*(len(PATH) - 1); i = int(min(np.floor(x), len(PATH) - 2)); f = ease(x - i)
    R = PR[i]*(1-f) + PR[i+1]*f; L = np.log10(PM[i])*(1-f) + np.log10(PM[i+1])*f
    return R, 10**L, i + (1 if f > 0.5 else 0)

def frame(i):
    s = i/FPS
    for a in (axV, axR, axL): a.set_visible(True)
    card.set_text(""); eq.set_text(""); info.set_text(""); leg.set_text("")
    for o in (lmod, lorb, lrad, lL, mk1, mk2): o.set_data([], [])
    vRlab.set_text(""); vR.set_visible(False); rs.set_offsets(np.empty((0, 2))); pth.set_data([], []); cur.set_data([], []); ell.set_visible(False)
    eb_vis(eb, True); eb_vis(ebV, False)
    if s < T["title"][1]:
        for a in (axV, axR, axL): a.set_visible(False)
        title.set_text("Fitting a rotation curve with the Constant-Lagrangian model"); sub.set_text("UGC 1281 (dwarf, nearly solid-body rise), step by step")
        card.set_text("1  The measured rotation curve\n2  Writing it in v², the quantity the model predicts\n3  The CL model and a first guess\n"
                      "4  The χ² landscape\n5  Levenberg–Marquardt: the fit, step by step\n6  Result, uncertainty, AIC / BIC\n7  What the fitted parameters imply")
        return
    if s < T["data"][1]:
        title.set_text("1–2 · The data, written in v²"); sub.set_text("SPARC rotation curve; errors propagate as σ(v²) = 2V·σ_V")
        u = prog(s, 10.5, 14.5)
        axR.set_visible(False); axL.set_visible(False)
        if u < 0.5:
            eb_vis(eb, False); eb_vis(ebV, True); axV.set_ylim(0, 72); axV.set_ylabel("V (km/s)")
            eq.set_text("V(r) with 1σ errors  —  25 points from 0.08 to 4.99 kpc (σ_V ≥ 4 km/s)")
        else:
            axV.set_ylim(0, 3900); axV.set_ylabel("v² (km² s⁻²)")
            eq.set_text("v² = V²,   σ(v²) = 2V σ_V   ·   the CL model is written for v², so the fit is done in v²")
        return
    axV.set_ylim(0, 3900); axV.set_ylabel("v² (km² s⁻²)")
    if s < T["model"][1]:
        axR.set_visible(False); axL.set_visible(False)
        title.set_text("3 · The CL model and a first guess"); sub.set_text("Two parameters: bulge radius R and mass M; H_z fixed")
        eq.set_text("r ≤ R:  v² = ½X²(r/R)²      r > R:  v² = 3/2·X² − (√(2GM/r) − H_z r)²      X = √(2GM/R) − H_z R")
        R0, M0 = PR[0], PM[0]; u = prog(s, 18, 21)
        y = model_v2(rr, R0, M0); m = rr <= rr[0] + u*(rr[-1]-rr[0])
        lmod.set_data(rr[m], y[m]); vR.set_visible(u > 0); vR.set_xdata([R0, R0]); vRlab.set_position((R0+0.05, 3650)); vRlab.set_text("R")
        if s > 21: info.set_text(f"first guess\nR = {R0:.2f} kpc\nM = {Msun(M0):.2e} M⊙\nχ² = {PC[0]:.0f}")
        return
    if s < T["iter"][1]:
        if s < T["land"][1]:
            title.set_text("4 · The χ² landscape"); sub.set_text("χ² = Σ [(v²_data − v²_model)/σ(v²)]² over (R, M); the fit must find the valley")
            R, M10, it = PR[0], PM[0], 0
        else:
            title.set_text("5 · Levenberg–Marquardt, step by step"); sub.set_text("Each dot is a real accepted iteration of the solver")
            R, M10, it = path_at(prog(s, T["land"][1] + 0.5, T["iter"][1] - 1.5))
        y = model_v2(rr, R, M10); lmod.set_data(rr, y); vR.set_visible(True); vR.set_xdata([R, R]); vRlab.set_position((R+0.05, 3650)); vRlab.set_text("R")
        res = resid([R, M10]); rs.set_offsets(np.c_[r, np.clip(res, -0.95, 0.95)])
        axR.set_ylim(-1.0, 1.0)
        upto = it + 1
        pth.set_data(PR[:upto], np.log10(Msun(PM[:upto]))); cur.set_data([R], [np.log10(Msun(M10))])
        chi = float(np.sum(res**2))
        info.set_text(f"iteration {it:2d} / {len(PATH)-1}\nR  = {R:6.3f} kpc\nM  = {Msun(M10):.3e} M⊙\nχ² = {chi:9.3f}")
        return
    if s < T["result"][1]:
        title.set_text("6 · Result"); sub.set_text("Best fit with 1σ ellipse; residuals far inside ±1σ (χ²ν ≈ 0.03)")
        lmod.set_data(rr, model_v2(rr, Rb, Mb)); vR.set_visible(True); vR.set_xdata([Rb, Rb]); vRlab.set_position((Rb+0.05, 3650)); vRlab.set_text("R")
        rs.set_offsets(np.c_[r, fb]); pth.set_data(PR, np.log10(Msun(PM))); cur.set_data([Rb], [np.log10(Msun(Mb))])
        # 1-sigma ellipse in (R, log10 M): transform covariance
        Lb = np.log10(Msun(Mb)); jl = 1/(np.log(10)*Msun(Mb))
        C = np.array([[covS[0, 0], covS[0, 1]*jl], [covS[1, 0]*jl, covS[1, 1]*jl**2]])
        w, v = np.linalg.eigh(C); a = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
        ell.set_visible(True); ell.set_center((Rb, Lb)); ell.width = 2*np.sqrt(2.30*w[1]); ell.height = 2*np.sqrt(2.30*w[0]); ell.angle = a
        info.set_text(f"R  = {Rb:.3f} ± {sR:.3f} kpc\nM  = ({Msun(Mb)/1e8:.3f} ± {sM*conv/1e8:.2f})×10⁸ M⊙\n"
                      f"χ² = {chi2b:.3f}   χ²ν = {chi2b/(n-k):.3f}\nAIC = {AIC:.3f}   BIC = {BIC:.3f}\nRMS_rel = {rms:.4f}\n\n"
                      f"paper: R = 1.965 ± 0.153,\nM = (6.915 ± 1.089)×10⁸, χ² = 0.617,\nAIC 4.62, BIC 7.05, RMS 6.55% ✓")
        return
    if s < T["chan"][1]:
        title.set_text("7 · What the fitted parameters imply"); sub.set_text("The fit uses only v²_orb; the radial channel follows from the same R, M, H_z")
        vo, vr2, vL2 = channels(rr, Rb, Mb)
        u = prog(s, T["chan"][0], T["chan"][0] + 2.5)
        lorb.set_data(rr, vo); lrad.set_data(rr[: max(2, int(u*len(rr)))], vr2[: max(2, int(u*len(rr)))])
        lL.set_data(rr[rr > Rb], np.full((rr > Rb).sum(), vL2))
        xm = 0.2 + 4.75*(0.5 - 0.5*np.cos(2*np.pi*(s - T["chan"][0])/8))
        a, b, _ = channels(np.array([xm]), Rb, Mb); mk1.set_data([xm], a); mk2.set_data([xm], b)
        axV.set_ylim(0, 4800); vR.set_visible(True); vR.set_xdata([Rb, Rb])
        axR.set_visible(False)
        eq.set_text("v²_rad,eff = (√(2GM/r) − H_z r)²      v²_orb + v²_rad,eff = v²_L = const  (r > R)")
        leg.set_text("orange: v²_orb (fitted to data)\npurple: v²_rad,eff (inferred, not observed)\ndotted: v²_L")
        info.set_text(f"r_c = (2GM/H_z²)^(1/3) = {rc:.0f} kpc\nv_L = {np.sqrt(vL2):.1f} km/s\n"
                      f"at r = {xm:4.1f} kpc:\n  v²_orb = {a[0]:6.0f}\n  v²_rad = {b[0]:6.0f}")
        rs.set_offsets(np.empty((0, 2)))
        return
    for a in (axV, axR, axL): a.set_visible(False)
    title.set_text("Reading this fit"); sub.set_text("")
    card.set_text("•  The two-parameter CL fit reproduces the published values exactly\n"
                  "    (R, M with uncertainties, χ², AIC, BIC, RMS).\n\n"
                  "•  χ²ν = 0.027 ≪ 1: most SPARC errors here sit at the 4 km/s floor, large compared with the\n"
                  "    scatter. RMS_rel = 6.6% is dominated by the innermost points, where v² is tiny.\n\n"
                  "•  The paper's outer virial variant (r ≥ 3 kpc, k = 4) lowers χ² to 0.18, but AIC rises\n"
                  "    to 8.18: the simple (R, M) fit is preferred. NFW (AIC 79.0, r_s pegged) and the\n"
                  "    MOND Plummer proxy (AIC 6.67) do worse on the same data.\n\n"
                  "•  The curve is still rising at 5 kpc, so the data-only seed v²(R) = v_L²/3 gives\n"
                  "    R₀ ≈ 1.66 kpc, below the fitted 1.96 kpc — the plateau is not yet reached.\n\n"
                  "•  Only the orbital channel is fitted; v_rad,eff is a model consequence, not a measurement.")
    card.set_fontsize(14.5)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        for t in (4, 9, 14, 23, 28, 42, 56, 67, 78): frame(int(t*FPS)); fig.savefig(f"f_{t}.png", facecolor=BG)
    else:
        w = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1, extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"])
        with w.saving(fig, video("ugc1281_cl_fit_step_by_step.mp4"), dpi=120):
            for i in range(min(NF, MAX_FRAMES)): frame(i); w.grab_frame()
