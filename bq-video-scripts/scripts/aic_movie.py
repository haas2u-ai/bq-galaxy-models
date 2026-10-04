from paths import video, cache, MAX_FRAMES   # sets the working directory; see paths.py
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyBboxPatch

D = json.load(open("data/curated.json"))
D.sort(key=lambda o: o["dAIC_k2"])          # fixed row order: like-for-like result
N = len(D)
FPS = 30
BG, FG, MUT, EDGE = "#070a14", "#e7ebf3", "#8f9ab2", "#2c3445"
C_CL, C_COMP, C_TIE = "#f2a24a", "#7fb8ff", "#8f9ab2"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.edgecolor": EDGE})

# ---------- timeline (seconds) ----------
T_TITLE, T_RULES, T_SETUP = (0, 7), (7, 15), (15, 29)
T_RACE, T_HOLD1 = (29, 41), (41, 45)
T_SLIDE, T_HOLD2 = (45, 53), (53, 57)
T_BIC, T_HOLD3 = (57, 62), (62, 66)
T_SUM, T_CAVEAT = (66, 76), (76, 88)
T_END = 88
NF = int(T_END*FPS)
ease = lambda u: 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0, 1))
def prog(s, a, b): return ease((s-a)/(b-a))

# ---------- symlog-style x mapping with clipping ----------
LT, CLIP = 10.0, 80.0
def xmap(d):
    d = np.clip(d, -CLIP, CLIP)
    return np.where(np.abs(d) <= LT, d/LT, np.sign(d)*(1 + np.log10(np.abs(d)/LT)))
XMAX = float(xmap(np.array([CLIP]))[0])

fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
ax = fig.add_axes([0.13, 0.11, 0.56, 0.72], facecolor=BG)
side = fig.add_axes([0.72, 0.11, 0.26, 0.72], facecolor=BG); side.set_axis_off()
title = fig.text(0.03, 0.935, "", fontsize=22, weight="bold")
subtitle = fig.text(0.03, 0.895, "", fontsize=12.5, color=MUT)
card = fig.text(0.5, 0.5, "", ha="center", va="center", multialignment="left", fontsize=15, linespacing=1.6, color=FG)
foot = fig.text(0.03, 0.02, "Source: de Haas, “Galactic Rotation Curves and the Constant–Lagrangian Field” (SPARC fit tables). "
                "Later table version used for NGC 3741 and UGC 6446; UGC 2953 and F563–V2 excluded.", fontsize=9.3, color=MUT)

# static axes decoration
ax.set_ylim(-1, N); ax.set_xlim(-XMAX-0.05, XMAX+0.05)
ticks = [-80, -30, -10, -6, -2, 0, 2, 6, 10, 30, 80]
ax.set_xticks(xmap(np.array(ticks, float))); ax.set_xticklabels(["≤−80", "−30", "−10", "−6", "−2", "0", "2", "6", "10", "30", "≥80"], fontsize=10)
ax.set_yticks(range(N)); ax.set_yticklabels([o["galaxy"] for o in D], fontsize=9)
ax.axvspan(xmap(np.array([-2.]))[0], xmap(np.array([2.]))[0], color=C_TIE, alpha=0.12)
for v in (-10, -6, 6, 10):
    ax.axvline(xmap(np.array([float(v)]))[0], color=EDGE, lw=0.8, ls=":")
ax.axvline(0, color=MUT, lw=1)
ax.grid(axis="y", alpha=0.06)
xl = ax.set_xlabel("", fontsize=12)
ax.text(xmap(np.array([-40.]))[0], N-0.2, "← CL preferred", ha="center", color=C_CL, fontsize=12, va="bottom")
ax.text(xmap(np.array([40.]))[0], N-0.2, "MOND / DM preferred →", ha="center", color=C_COMP, fontsize=12, va="bottom")
ax.text(0, -0.9, "|Δ| ≤ 2: no clear preference", ha="center", color=MUT, fontsize=9.5)
sc = ax.scatter([], [], s=70, zorder=3, edgecolors="none")
ktxt = [ax.text(0, i, "", fontsize=8.5, va="center", color=FG, zorder=4) for i in range(N)]
clipt = [ax.text(0, i, "", fontsize=8.5, va="center", color=MUT, zorder=4) for i in range(N)]
axes_objs = [ax] 
tally_txt = side.text(0.0, 0.98, "", va="top", fontsize=13, family="DejaVu Sans Mono", linespacing=1.7, transform=side.transAxes)
note_txt = side.text(0.0, 0.40, "", va="top", fontsize=11, color=MUT, linespacing=1.5, transform=side.transAxes, wrap=True)

def colors(d):
    return [C_CL if x < -2 else (C_COMP if x > 2 else C_TIE) for x in d]
def tally(d):
    d = np.asarray(d); return int((d < -2).sum()), int((np.abs(d) <= 2).sum()), int((d > 2).sum())

RULES = ("How the models are scored\n\n"
         "AIC = χ² + 2k          BIC = χ² + k ln n\n"
         "k = number of free parameters,  n = number of data points\n\n"
         "Δ = score(CL) − score(best of MOND / DM)   ·   lower is better\n"
         "|Δ| ≤ 2: no clear preference   ·   |Δ| > 6: strong   ·   |Δ| > 10: very strong")
SETUP = ("What CL is compared with — the restrictions of this study\n\n"
         "MOND:  “simple” µ-function, a₀ = 1.2×10⁻¹⁰ m s⁻² fixed;\n"
         "baryons modelled as ONE Plummer sphere with free mass M_b and scale a  (k = 2).\n"
         "No SPARC stellar/gas decomposition, no mass-to-light ratios, no external-field effect\n"
         "(one k = 5 variant with EFE for NGC 247 only).\n\n"
         "Dark matter:  pseudo-isothermal (ISO) core halo ALONE, free ρ₀ and r_c  (k = 2);\n"
         "no baryonic disk or gas.  NFW instead for UGC 1281, IC 2233, NGC 3917;  ISO + Plummer baryon for NGC 3972;\n"
         "one k = 5 gNFW + baryons variant for NGC 247.\n\n"
         "CL:  per galaxy the best of single-L (k = 2) and its extensions —\n"
         "two-Lagrangian, Φ_BH offset, virial windows  (k = 3 … 6).\n\n"
         "These MOND and DM fits are two-parameter proxies, not the standard SPARC\n"
         "mass-model fits of the MOND and ΛCDM literature, which use the measured baryons.")
CAVEAT = ("Reading these results\n\n"
          "1.  Like-for-like (k = 2 each), MOND or DM is preferred more often than CL.\n"
          "2.  CL's advantage comes from its extensions, which pay the AIC/BIC penalty and still win in about half the galaxies.\n"
          "3.  Choosing the best CL variant per galaxy after seeing the data is not itself penalised by AIC or BIC.\n"
          "4.  MOND and DM are restricted proxies here (single Plummer sphere; halo without baryons).\n"
          "     Standard literature fits with measured baryons could change individual verdicts.\n"
          "5.  Many fits have χ²ν ≪ 1, which suggests overestimated v² errors; this shrinks χ² differences\n"
          "     and lets the 2k or k ln n penalty dominate.\n\n"
          "A fair next step: refit MOND and DM with SPARC baryonic mass models on the same v² data.")

def set_plot_visible(v):
    ax.set_visible(v); side.set_visible(v)

def frame(i):
    s = i/FPS
    card.set_text(""); set_plot_visible(False)
    if s < T_TITLE[1]:
        title.set_text("CL vs MOND vs dark matter on 33 SPARC rotation curves")
        subtitle.set_text("What the information criteria in the fit tables actually say")
        card.set_text("Each galaxy was fitted in v²(r) with the Constant-Lagrangian inflow model,\n"
                      "a MOND proxy and a dark-matter proxy.\n\nWho wins — and on what terms?")
        card.set_fontsize(17); return
    if s < T_RULES[1]:
        title.set_text("Scoring rules"); subtitle.set_text("Akaike and Bayesian information criteria")
        card.set_text(RULES); card.set_fontsize(15.5); return
    if s < T_SETUP[1]:
        title.set_text("The comparison is only as good as the competitors"); subtitle.set_text("Model choices made in the paper — stated explicitly")
        card.set_text(SETUP); card.set_fontsize(13.2); return
    if s >= T_CAVEAT[0]:
        title.set_text("Caveats before any conclusion"); subtitle.set_text("")
        card.set_text(CAVEAT); card.set_fontsize(13.5); return

    set_plot_visible(True)
    d_k2 = np.array([o["dAIC_k2"] for o in D]); d_best = np.array([o["dAIC_best"] for o in D]); d_bic = np.array([o["dBIC_best"] for o in D])
    stage = "k2"
    if s < T_HOLD1[1]:
        u = np.clip((s - T_RACE[0])/(T_RACE[1] - T_RACE[0] - 1.5), 0, 1); nshow = int(np.ceil(u*N))
        d = d_k2; shown = np.arange(N) < nshow
        title.set_text("Round 1 · like-for-like: CL (k = 2) vs best of MOND / DM (k = 2)")
        subtitle.set_text("ΔAIC per galaxy; negative = CL preferred")
        xl.set_text("ΔAIC = AIC(CL, k=2) − min[AIC(MOND), AIC(DM)]")
    elif s < T_HOLD2[1]:
        u = prog(s, *T_SLIDE); d = d_k2*(1-u) + d_best*u; shown = np.ones(N, bool); stage = "best"
        title.set_text("Round 2 · CL may use its extensions (k = 3 … 6), MOND / DM keep theirs")
        subtitle.set_text("Dots slide to the best CL variant; the number is its parameter count k")
        xl.set_text("ΔAIC = AIC(best CL variant) − min[AIC(MOND), AIC(DM)]")
    elif s < T_HOLD3[1]:
        u = prog(s, *T_BIC); d = d_best*(1-u) + d_bic*u; shown = np.ones(N, bool); stage = "bic"
        title.set_text("Round 3 · the same comparison with BIC")
        subtitle.set_text("BIC penalises extra parameters more strongly (k ln n)")
        xl.set_text("ΔBIC = BIC(best CL variant) − min[BIC(MOND), BIC(DM)]")
    else:
        d = d_bic; shown = np.ones(N, bool); stage = "sum"
        title.set_text("Summary of the three rounds"); subtitle.set_text("")
    x = xmap(d); y = np.arange(N)
    sc.set_offsets(np.c_[x[shown], y[shown]]); sc.set_color(np.array(colors(d))[shown])
    for k_ in range(N):
        clipt[k_].set_text(""); ktxt[k_].set_text("")
        if not shown[k_]: continue
        clipped = abs(d[k_]) > CLIP
        lab = []
        if clipped: lab.append(f"Δ = {d[k_]:.0f}")
        if stage in ("best", "bic", "sum"): lab.append(f"k={D[k_]['cl_best_k']}")
        if lab:
            inward = -1 if (clipped and d[k_] > 0) else 1
            ktxt[k_].set_ha("left" if inward > 0 else "right")
            ktxt[k_].set_position((x[k_] + 0.08*inward, y[k_] + 0.32)); ktxt[k_].set_text("  ".join(lab))
            ktxt[k_].set_alpha(1.0 if (clipped or stage != "best") else prog(s, T_SLIDE[0], T_SLIDE[0]+1.5))
    a1 = tally(d_k2[shown] if stage == "k2" else d_k2); a2 = tally(d_best); b2 = tally(d_bic)
    lines = ["              CL  tie  MOND/DM",
             f"AIC, k=2    {a1[0]:4d} {a1[1]:4d} {a1[2]:6d}"]
    if stage in ("best", "bic", "sum"): lines.append(f"AIC, best   {a2[0]:4d} {a2[1]:4d} {a2[2]:6d}")
    if stage in ("bic", "sum"):         lines.append(f"BIC, best   {b2[0]:4d} {b2[1]:4d} {b2[2]:6d}")
    tally_txt.set_text("\n".join(lines))
    notes = {"k2": "Among the 16 k=2 losses the winner is\nMOND in 8 and DM in 8.\n\nSame number of parameters on both sides.",
             "best": "CL's chosen variant per galaxy:\nk=2 in 13, k=3 in 5, k=4 in 8,\nk=5 in 6, k=6 in 1.\n\nAIC charges 2 per extra parameter.",
             "bic": "BIC charges ln n ≈ 2.3–4.2 per\nparameter here, so several CL wins\nbecome ties.",
             "sum": "Lowest AIC of all fitted models:\nCL 20 · DM 8 · MOND 5 galaxies.\n\nThe outcome depends on whether\nCL's extensions are allowed —\nand on how restricted the\ncompetitors are (next card)."}
    note_txt.set_text(notes[stage])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        for t in (4, 11, 22, 40, 50, 60, 70, 82):
            frame(int(t*FPS)); fig.savefig(f"aic_{t}.png", facecolor=BG)
    else:
        w = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1, extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"])
        with w.saving(fig, video("cl_mond_dm_aic_bic_comparison.mp4"), dpi=120):
            for i in range(min(NF, MAX_FRAMES)): frame(i); w.grab_frame()
