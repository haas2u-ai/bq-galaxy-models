from paths import video, cache, MAX_FRAMES   # sets the working directory; see paths.py
import numpy as np, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from pgjet import *

FPS = 30; T_END = 100.0; NF = int(T_END*FPS)
S_FAR, S_LAUNCH, S_END = 4.0e6, 30.0, 6.0e6
RMAX = 4e7
ease = lambda u: 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0, 1))
def lerp_log(a, b, u): return np.exp(np.log(a)*(1-u) + np.log(b)*u)
def scale(s):
    if s < 8: return S_FAR
    if s < 34: return lerp_log(S_FAR, S_LAUNCH, ease((s-8)/26))
    if s < 56: return S_LAUNCH
    if s < 90: return lerp_log(S_LAUNCH, S_END, ease((s-56)/34))
    return S_END

rng = np.random.default_rng(87)
A = 0.012                                    # display rate in ln r per frame
# inflow tracers (35 deg cones)
NI = 7000; iu = rng.uniform(np.log(20), np.log(RMAX), NI); ip = 2*np.pi*rng.random(NI); isd = rng.choice([-1, 1], NI)
# jet sheath on theta0 (exact cone), double helix strands, above r0
NJ = 12000; ju = rng.uniform(np.log(R0*1.001), np.log(RMAX), NJ); jp = rng.integers(0, 2, NJ)*np.pi + 0.15*rng.standard_normal(NJ)
jsd = rng.choice([-1, 1], NJ); jjit = 1 + 0.03*rng.standard_normal(NJ)
# infall on the jet cone below r0
NB = 700; bu = rng.uniform(np.log(2.2), np.log(R0*0.999), NB); bp = rng.integers(0, 2, NB)*np.pi + 0.15*rng.standard_normal(NB); bsd = rng.choice([-1, 1], NB)
# spine
NS = 1800; su = rng.uniform(np.log(3), np.log(RMAX), NS); sp = 2*np.pi*rng.random(NS); ssd = rng.choice([-1, 1], NS)

def step():
    global iu, ju, bu, su
    iu -= A; m = iu < np.log(20); iu[m] = np.log(RMAX) - rng.random(m.sum())*0.05
    r = np.exp(ju); g = np.clip(eps(r)/EPSI, 0.15, 1.0); ju += A*g
    m = ju > np.log(RMAX); ju[m] = np.log(R0*(1.001 + 0.002*rng.random(m.sum())))
    r = np.exp(bu); g = np.clip(np.abs(eps(r))/EPSI, 0.05, 1.0); bu -= A*g
    m = bu < np.log(2.2); bu[m] = np.log(R0*(0.999 - 0.002*rng.random(m.sum())))
    su += 1.6*A; m = su > np.log(RMAX); su[m] = np.log(3.0)
for _ in range(1500): step()

FLAT = [3.0]
def lateral(s):   # transverse display widening of the jet cones (labelled on screen)
    if s < 30: return 3.0
    if s < 36: return 3.0 + 2.0*ease((s-30)/6)
    if s < 56: return 5.0
    if s < 62: return 5.0 - 2.0*ease((s-56)/6)
    return 3.0
def cone_xyz(r, th, phi, side, f=1.0):
    rho = f*r*np.sin(th); return rho*np.cos(phi), rho*np.sin(phi), side*r*np.cos(th)

BG, FG, MUT = "#03050b", "#e7ebf3", "#8f9ab2"
C_IN, C_RING, C_SPINE = "#f2a24a", "#feca57", "#e8f4ff"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.edgecolor": "#2c3445"})
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
ax = fig.add_axes([-0.05, -0.06, 0.78, 1.10], projection="3d", facecolor=BG); ax.set_axis_off()
LIM = 1.25; ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM); ax.set_box_aspect((1, 1, 1))
sc_in = ax.scatter([], [], [], s=2.0, c=C_IN, alpha=0.55, lw=0, depthshade=False)
sc_ju = ax.scatter([], [], [], s=2.6, lw=0, depthshade=False)
sc_jl = ax.scatter([], [], [], s=2.6, lw=0, depthshade=False)
sc_b = ax.scatter([], [], [], s=3.0, c="#ff6b6b", alpha=0.85, lw=0, depthshade=False)
sc_s = ax.scatter([], [], [], s=2.2, c=C_SPINE, alpha=0.7, lw=0, depthshade=False)
parcel, = ax.plot([], [], [], "o", color="#ffffff", ms=9, mec=C_RING, mew=2, zorder=10)
trail, = ax.plot([], [], [], color="#ffffff", lw=2.0, alpha=0.9)
hand, = ax.plot([], [], [], color=C_RING, lw=1.6, ls=":")
stag = [ax.plot([], [], [], color=C_RING, lw=2.0)[0] for _ in range(2)]
pring, = ax.plot([], [], [], color="#ffffff", lw=1.2, ls="--", alpha=0.8)
hst = [ax.plot([], [], [], color="#ff9ff3", lw=2.2)[0] for _ in range(2)]
bh = [None]
cm_j = plt.get_cmap("cool")

title = fig.text(0.03, 0.94, "The M 87 jet in a single PG geometry", fontsize=22, weight="bold")
sub = fig.text(0.03, 0.905, "Following the flow of space: in along the broad cone, through the stagnation surface, out along the 6° jet",
               fontsize=12.5, color=MUT)
caption = fig.text(0.36, 0.035, "", ha="center", fontsize=13.5, color=FG,
                   bbox=dict(boxstyle="round,pad=0.45", fc="#10151f", ec="#2c3445"))
scale_txt = fig.text(0.03, 0.85, "", fontsize=13, family="DejaVu Sans Mono", color=FG)
card = fig.text(0.5, 0.5, "", ha="center", va="center", multialignment="left", fontsize=14, linespacing=1.6)
foot = fig.text(0.03, 0.012, "Kinematics: revised Sec. 2.2 (θ₀ = 6°, κ = tan θ₀ (1+ε), ε = 0.05 (1 − 15 r_g/r)); broad cone 35° with κ reused. "
                "Log-radius camera, display tracer speeds. M = 6.5×10⁹ M⊙, r_g = 3.1×10⁻⁴ pc. Markers after the paper and Hada et al. (2024).",
                fontsize=9, color=MUT)
# ruler panel
axr = fig.add_axes([0.80, 0.12, 0.035, 0.72], facecolor=BG)
axr.set_yscale("log"); axr.set_ylim(1, 1e8); axr.set_xlim(0, 1); axr.set_xticks([])
axr.set_ylabel("radius (r_g)", fontsize=11)
for y0, y1, col in ((30, 1e3, "#2e86de"), (1e3, 6.5e5, "#5f27cd"), (6.5e5, 1e8, "#341f97")):
    axr.axhspan(y0, y1, color=col, alpha=0.35)
MARK = [(2, "horizon ~2 r_g"), (5.2, "photon ring √27 r_g (EHT)"), (15, "stagnation r₀ = 15 r_g"),
        (1e3, "inner funnel → parabolic zone"), (6.5e5, "HST-1 / collimation break ≈ 200 pc"),
        (3.2e6, "1 kpc"), (9.7e6, "3 kpc")]
for y, lab in MARK:
    axr.axhline(y, color=FG, lw=0.6, alpha=0.6)
    fig.text(0.842, 0.12 + 0.72*np.log10(y)/8, lab, fontsize=9.5, va="center", color=FG)
win = axr.axhspan(1, 2, color=C_RING, alpha=0.35); mark_s, = axr.plot([0.5], [10], ">", color=C_RING, ms=11)
# epsilon inset
axe = fig.add_axes([0.80, 0.62, 0.17, 0.0001], facecolor=BG)   # placeholder, resized when visible
axe.set_visible(False)
axe2 = fig.add_axes([0.075, 0.58, 0.2, 0.2], facecolor="#0b0f1a")
rr_e = np.geomspace(2.2, 1e3, 400)
axe2.semilogx(rr_e, eps(rr_e), color="#48dbfb", lw=2); axe2.axhline(0, color=MUT, lw=0.8); axe2.axvline(R0, color=C_RING, lw=1, ls="--")
axe2.set_ylim(-0.32, 0.08); axe2.set_xlabel("r (r_g)", fontsize=9); axe2.set_ylabel("ε = w_θ / v_esc", fontsize=9)
axe2.tick_params(labelsize=8); axe2.set_title("6° cone: infall below r₀, jet above", fontsize=9.5, loc="left")
axe2.text(30, -0.25, "ε < 0: falls in\nε > 0: jet", fontsize=8.5, color=FG)
mk_e, = axe2.plot([], [], "o", color="#ffffff", ms=6, mec=C_RING)

def parcel_state(s, S):
    """position of the followed parcel and its trail (in r_g)"""
    if s < 8: return None
    if s < 34:
        r = 0.55*S; seg = np.geomspace(r, min(1.9*r, RMAX), 120)
        return cone_xyz(np.array([r]), THB, K_IN*np.log([r]), 1), cone_xyz(seg, THB, K_IN*np.log(seg), 1), "in"
    if s < 38:
        r = lerp_log(0.55*S_LAUNCH, 17.0, ease((s-34)/4)); seg = np.geomspace(r, 2.0*r, 120)
        return cone_xyz(np.array([r]), THB, K_IN*np.log([r]), 1), cone_xyz(seg, THB, K_IN*np.log(seg), 1), "in"
    if s < 40: return "hand"
    if s < 56:
        x = R0*(1 + 0.003*np.exp(np.log(0.6/0.003)*ease((s-40)/12))) if s < 52 else lerp_log(R0*1.6, 0.55*S_LAUNCH, ease((s-52)/4))
        seg = np.linspace(max(R0*1.0005, x/1.4), x, 300)
        return cone_xyz(np.array([x]), TH0, phi_sheath([x]), 1, FLAT[0]), cone_xyz(seg, TH0, phi_sheath(seg), 1, FLAT[0]), "jet"
    if s < 90:
        r = 0.55*S; seg = np.geomspace(max(R0*1.002, r/1.6), r, 300)
        return cone_xyz(np.array([r]), TH0, phi_sheath([r]), 1, FLAT[0]), cone_xyz(seg, TH0, phi_sheath(seg), 1, FLAT[0]), "jet"
    return None

def sphere(Rs, S):
    u, v = np.mgrid[0:2*np.pi:24j, 0:np.pi:12j]
    return ax.plot_surface(Rs/S*np.cos(u)*np.sin(v), Rs/S*np.sin(u)*np.sin(v), Rs/S*np.cos(v), color="#000000", shade=False, zorder=9)

def frame(i):
    s = i/FPS; step(); S = scale(s); F = lateral(s); FLAT[0] = F
    if bh[0] is not None: bh[0].remove(); bh[0] = None
    if s >= 90:
        ax.set_visible(False); axr.set_visible(False); axe2.set_visible(False); caption.set_text(""); scale_txt.set_text("")
        for t in fig.texts:
            if t not in (title, sub, card, foot): t.set_visible(False)
        title.set_text("What this film shows — and what it assumes"); sub.set_text("")
        card.set_text(
            "Imposed by observation:  the 6° cone and v_φ / v_esc = tan θ₀ = 0.105 (to leading order).\n\n"
            "Calibrated:  stagnation radius r₀ = 15 r_g (VLBI inflow–outflow transition) and ε∞ = 0.05\n"
            "      (within the bound |ε∞| ≲ 0.17 from the opening angle; to be fixed by measured helix wavelengths).\n\n"
            "Derived from the PG geometry:  helical streamlines on constant-θ cones; diverging winding at r₀;\n"
            "      pitch λ ≈ 6.25 r ε/(1+ε) → 0.3 r far out; 0.3° opening-angle drift; bipolar symmetry; nested cones.\n\n"
            "Interpretive:  the hand-over from the 35° inflow cone to the 6° jet cone (Bernoulli redirection — PG streamlines\n"
            "      keep θ fixed); the parabolic-to-conical break and HST-1 as recollimation; the jet’s relativistic speed.\n\n"
            "Display conventions:  log-radius camera; tracer speeds are not physical; launch region and markers schematic.\n"
            "Sources: de Haas, PG jet of M 87 (2025) with revised Sec. 2.2; Hada et al., A&A Rev. 32, 5 (2024).")
        return
    ax.set_visible(True); axr.set_visible(True)
    ax.view_init(elev=16 + 8*np.sin(2*np.pi*s/60), azim=-70 + 0.9*s*3.6/3)
    lo, hi = 0.004*S, 1.3*S
    # inflow
    r = np.exp(iu); m = (r > lo) & (r < hi)
    x, y, z = cone_xyz(r[m], THB, ip[m] + K_IN*np.log(r[m]), isd[m]); sc_in._offsets3d = (x/S, y/S, z/S)
    # jet sheath: upper / lower (counter-jet dimmer)
    r = np.exp(ju); m = (r > lo) & (r < hi)
    for scat, sd, alpha in ((sc_ju, 1, 0.9), (sc_jl, -1, 0.45)):
        mm = m & (jsd == sd); rr = r[mm]
        x, y, z = cone_xyz(rr, TH0*jjit[mm], jp[mm] + phi_sheath(rr), sd, F); scat._offsets3d = (x/S, y/S, z/S)
        scat.set_color(cm_j(np.clip(np.log10(rr)/7.5, 0, 1)*0.85)); scat.set_alpha(alpha)
    r = np.exp(bu); m = (r > lo) & (r < hi) & (S < 2000)
    x, y, z = cone_xyz(r[m], TH0, bp[m] + phi_sheath(r[m]), bsd[m], F); sc_b._offsets3d = (x/S, y/S, z/S)
    r = np.exp(su); m = (r > lo) & (r < hi)
    x, y, z = cone_xyz(r[m], THS, sp[m] + phi_spine(r[m]), ssd[m], F); sc_s._offsets3d = (x/S, y/S, z/S)
    # launch-region markers
    th = np.linspace(0, 2*np.pi, 120)
    if S < 3000:
        for k, sd in enumerate((1, -1)):
            stag[k].set_data_3d(F*R0*np.sin(TH0)*np.cos(th)/S, F*R0*np.sin(TH0)*np.sin(th)/S, np.full_like(th, sd*R0*np.cos(TH0)/S))
        pring.set_data_3d(np.sqrt(27)*np.cos(th)/S, np.sqrt(27)*np.sin(th)/S, 0*th)
        bh[0] = sphere(2.0, S)
    else:
        for l in stag: l.set_data_3d([], [], []);
        pring.set_data_3d([], [], [])
    zh = 6.5e5
    for k, sd in enumerate((1, -1)):
        if 0.003*S < zh < 1.3*S:
            rad = F*zh*np.tan(TH0)*1.08; hst[k].set_data_3d(rad*np.cos(th)/S, rad*np.sin(th)/S, np.full_like(th, sd*zh/S)); hst[k].set_alpha(1.0 if sd > 0 else 0.4)
        else: hst[k].set_data_3d([], [], [])
    # parcel
    ps = parcel_state(s, S); hand.set_data_3d([], [], [])
    if ps is None or (isinstance(ps, str) and ps != "hand"): parcel.set_data_3d([], [], []); trail.set_data_3d([], [], [])
    elif isinstance(ps, str):
        a = cone_xyz(np.array([17.0]), THB, K_IN*np.log([17.0]), 1); b = cone_xyz(np.array([R0*1.003]), TH0, phi_sheath([R0*1.003]), 1, F)
        u = ease((s-38)/2); p = [a[j]*(1-u) + b[j]*u for j in range(3)]
        parcel.set_data_3d(p[0]/S, p[1]/S, p[2]/S); trail.set_data_3d([], [], [])
        hand.set_data_3d(np.array([a[0][0], b[0][0]])/S, np.array([a[1][0], b[1][0]])/S, np.array([a[2][0], b[2][0]])/S)
    else:
        p, seg, kind = ps; parcel.set_data_3d(p[0]/S, p[1]/S, p[2]/S); trail.set_data_3d(seg[0]/S, seg[1]/S, seg[2]/S)
    # ruler & scale readout
    win.set_y(S*0.004); win.set_height(1.3*S - 0.004*S); mark_s.set_data([0.5], [S])
    fov = 2*1.25*S; pc = fov*RG_PC
    pcs = f"{pc*1e3:.3g} mpc" if pc < 1 else (f"{pc:.3g} pc" if pc < 1000 else f"{pc/1e3:.3g} kpc")
    e10 = int(np.floor(np.log10(fov))); fs = f"{fov:.0f}" if fov < 1e4 else f"{fov/10**e10:.1f}×10^{e10}"
    scale_txt.set_text(f"field of view ≈ {fs} r_g ≈ {pcs}\njet cones drawn {F:.0f}× wider than 6° (display)")
    show_e = 34 <= s < 56; axe2.set_visible(show_e)
    if show_e and isinstance(ps, tuple) and ps[2] == "jet":
        rp = float(np.hypot(ps[0][0][0], ps[0][1][0])/(F*np.sin(TH0))); mk_e.set_data([rp], [float(eps(rp))])
    else: mk_e.set_data([], [])
    # captions
    if s < 8: cap = "M 87 at kiloparsec scale: two broad inflow cones (35°) and the bipolar 6° jet cones"
    elif s < 20: cap = "Zooming in with a parcel of inflowing space on the 35° cone: w_θ = −0.85 v_esc, a nearly radial helix"
    elif s < 34: cap = "Parsec scales: the inflow keeps falling in; the jet cone is populated by the outflow above r₀"
    elif s < 38: cap = "Launch region (~50 r_g): black hole, photon ring (√27 r_g), stagnation rings at r₀ = 15 r_g on the jet cones"
    elif s < 40: cap = "Hand-over to the 6° cone — Bernoulli redirection (interpretive: PG streamlines keep θ fixed)"
    elif s < 52: cap = "Just above r₀, ε → 0: the helix winds ~100 turns per e-fold and hovers before moving out (red: infall below r₀)"
    elif s < 56: cap = "ε grows toward 0.05: the outward helix opens up"
    elif s < 68: cap = "Inner funnel (30–10³ r_g): tightly wound helices, pitch λ ≈ 6.25 r ε/(1+ε); nested spine and sheath"
    elif s < 80: cap = "Parabolic zone and the collimation break ≈ 200 pc: HST-1 (pink ring, after the paper and Hada et al. 2024)"
    else: cap = "Kiloparsec scale: λ → 0.3 r — the strands read as a double helix; the counter-jet is shown dimmed (de-boosted)"
    caption.set_text(cap)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        want = {int(t*FPS) for t in (4, 22, 36, 39, 46, 62, 76, 86)}
        for i in range(max(want)+1):
            if i in want: frame(i); fig.savefig(f"j_{i//FPS:02d}.png", facecolor=BG)
            else: step()
    else:
        w = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1, extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"])
        with w.saving(fig, video("m87_pg_jet_zoom.mp4"), dpi=120):
            for i in range(min(NF, MAX_FRAMES)):
                frame(i); w.grab_frame()
                if i % 200 == 0: print(i, NF, flush=True)
