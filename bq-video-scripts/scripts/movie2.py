from paths import video, cache, MAX_FRAMES   # sets the working directory; see paths.py
import numpy as np, sys, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from model2 import *

FPS = 30; T_END = 100.0; NF = int(T_END*FPS); DT = 0.004
LAT = np.radians(12.0); NT = 6500; NM = 700; C_DISP = 4.5
rng = np.random.default_rng(3)
ease = lambda u: 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0, 1))
A1, A2, A3, A4, A5 = (5, 22), (22, 34), (34, 56), (56, 72), (72, 92)
T_ON = A2[0] + 1.0
def state(s):
    """m: mass on, sp: spiral weight, Rd: view radius"""
    if s < A2[0]: return 0, 0, 1.6, 1
    if s < A3[0]: return 1, 0, 1.6, 2
    if s < A4[0]:
        u = ease((s-A3[0])/3); return 1, 0, 1.6 + (0.5-1.6)*u, 3
    if s < A5[0]:
        u = ease((s-A4[0])/3); return 1, 0, 0.5 + (1.6-0.5)*u, 4
    if s < A5[1]: return 1, ease((s-A5[0])/5), 1.6, 5
    return 1, 1, 1.6, 6

def sample(n, R0, R1, lat=LAT):
    u = rng.random(n); r = (R0**3 + u*(R1**3 - R0**3))**(1/3)
    return r, 2*np.pi*rng.random(n), np.arcsin(np.sin(lat)*(2*rng.random(n) - 1))
def xyz(r, p, l): c = r*np.cos(l); return c*np.cos(p), c*np.sin(p), r*np.sin(l)
cr, cp, cl = sample(NT, 0.01, 1.6)
chain = np.zeros(NT, bool)
mr = 1.6*np.sqrt(rng.random(NM)); mp = 2*np.pi*rng.random(NM)       # matter tracers (disk plane)
fb, fd = [], []; rate_c = rate_a = 0.0

def step(s, m, sp, Rd):
    global cr, cp, cl, chain, mr, mp, fb, fd, rate_c, rate_a
    # space cells: slow incompressible drift (Hubble part always, absorption part once the mass is on)
    for _ in range(2):
        v = v_space(cr) if m else H*cr
        cr = np.maximum(cr + 0.5*DT*v, 1e-3)
    # creation: random isotropic insertion, 3H per cell (no parents, no centre)
    nc = rng.poisson(3*H*DT*cr.size)
    if nc:
        r, p, l = sample(nc, 0.01, Rd); cr, cp, cl = np.r_[cr, r], np.r_[cp, p], np.r_[cl, l]; chain = np.r_[chain, np.zeros(nc, bool)]
        fb += list(zip(*xyz(r, p, l), [0]*nc))
    # absorption: only inside the matter (M, R)
    na = 0
    if m:
        dead = (rng.random(cr.size) < 1 - np.exp(-absorption(cr)*DT))
        na = int(dead.sum())
        if na: fd += list(zip(*xyz(cr[dead], cp[dead], cl[dead]), [0]*na))
        keep = ~dead; cr, cp, cl, chain = cr[keep], cp[keep], cl[keep], chain[keep]
    out = cr > Rd; cr, cp, cl, chain = cr[~out], cp[~out], cl[~out], chain[~out]
    # display normalisation during zoom only (not physical creation)
    if cr.size < 0.93*NT:
        k = NT - cr.size; r, p, l = sample(k, 0.01, Rd); cr, cp, cl = np.r_[cr, r], np.r_[cp, p], np.r_[cl, l]; chain = np.r_[chain, np.zeros(k, bool)]
    elif cr.size > 1.07*NT:
        keep = np.ones(cr.size, bool); idx = np.where(~chain)[0]; drop = rng.choice(idx, cr.size - NT, replace=False); keep[drop] = False
        cr, cp, cl, chain = cr[keep], cp[keep], cl[keep], chain[keep]
    rate_c = 0.9*rate_c + 0.1*nc/DT/max(cr.size, 1); rate_a = 0.9*rate_a + 0.1*na/DT/max(cr.size, 1)
    fb = [(x, y, z, a+1) for x, y, z, a in fb if a < 6][-900:]; fd = [(x, y, z, a+1) for x, y, z, a in fd if a < 10][-900:]
    # matter: moves with the riverbed behind the influence front; with the Hubble flow ahead of it
    front = C_DISP*(s - T_ON)*FPS*DT if (m and s >= T_ON) else -1
    for _ in range(3):
        h = DT/3
        behind = mr < front
        vr = np.where(behind, v_river(mr), H*mr)
        mr = np.maximum(mr + h*vr, 1e-3)
        if sp > 0: mp = mp + h*sp*np.where(behind, v_phi(mr), 0)/mr
    gone = (mr < 0.015) | (mr > 1.65)
    mr[gone] = np.where(rng.random(gone.sum()) < 0.5, 1.6*np.sqrt(rng.random(gone.sum())), 1.55 + 0.05*rng.random(gone.sum())); mp[gone] = 2*np.pi*rng.random(gone.sum())
    return front

def seed_chain():
    global cr, cp, cl, chain
    r = np.linspace(0.135, 0.245, 14); p = np.full(14, 0.6 + 1.1*rng.random()); l = np.zeros(14)
    cr, cp, cl, chain = np.r_[cr, r], np.r_[cp, p], np.r_[cl, l], np.r_[chain, np.ones(14, bool)]

BG, FG, MUT = "#05070d", "#e7ebf3", "#8f9ab2"
CC, CB, CD, CM, CS, CH = "#9aa7bd", "#2ecc71", "#ff6b6b", "#feca57", "#feca57", "#48dbfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.edgecolor": "#2c3445"})
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
ax = fig.add_axes([-0.04, -0.02, 0.66, 0.98], projection="3d", facecolor=BG); ax.set_axis_off()
sc = ax.scatter([], [], [], s=2.2, c=CC, alpha=0.45, lw=0, depthshade=False)
sbx = ax.scatter([], [], [], s=9, c=CB, alpha=0.9, lw=0, depthshade=False)
sdx = ax.scatter([], [], [], s=16, c=CD, alpha=0.95, lw=0, depthshade=False)
sch = ax.scatter([], [], [], s=22, c=CH, alpha=1.0, lw=0, depthshade=False)
smt = ax.scatter([], [], [], s=9, c="#ffb86b", alpha=0.95, lw=0, depthshade=False)
bulge, = ax.plot([], [], [], color="#fff0cf", lw=1.6)
frontL, = ax.plot([], [], [], color="#ffffff", lw=1.4, ls="--")
rcL, = ax.plot([], [], [], color=CS, lw=1.8, ls="-.")
ruL, = ax.plot([], [], [], color=MUT, lw=1.2, ls=":")
mark, = ax.plot([], [], [], "o", ms=12, mfc="none", mec=CH, mew=2)
arms = [ax.plot([], [], [], color="#ffb86b", lw=1.8)[0] for _ in range(4)]
title = fig.text(0.03, 0.94, "", fontsize=21, weight="bold"); sub = fig.text(0.03, 0.905, "", fontsize=12.2, color=MUT)
eqt = fig.text(0.03, 0.865, "", fontsize=13); leg = fig.text(0.03, 0.80, "", fontsize=11, linespacing=1.5, va="top")
card = fig.text(0.5, 0.5, "", ha="center", va="center", multialignment="left", fontsize=14, linespacing=1.6)
cnt = fig.text(0.03, 0.10, "", fontsize=11.5, family="DejaVu Sans Mono", linespacing=1.5, bbox=dict(boxstyle="round,pad=0.5", fc="#10151f", ec="#2c3445"))
fig.text(0.03, 0.015, "Schematic units: r_c = (2GM/H²)^{1/3} = 1, time in 1/H, bulge R = 0.12 r_c. Space cells in a ±12° equatorial wedge; matter in the disk plane. "
         "Display choices: absorption rate Q (drift balance at 0.25 r_c), c slowed for visibility.", fontsize=9, color=MUT)
pv = fig.add_axes([0.665, 0.69, 0.31, 0.20], facecolor=BG); pf = fig.add_axes([0.665, 0.40, 0.31, 0.20], facecolor=BG); pa = fig.add_axes([0.665, 0.11, 0.31, 0.20], facecolor=BG)
for p in (pv, pf, pa): p.grid(alpha=0.12); p.tick_params(labelsize=9); p.set_xlim(0, 1.6); p.axhline(0, color=MUT, lw=0.8); p.set_xlabel("r / r_c", fontsize=9.5)
rr = np.linspace(0.02, 1.6, 600)
lvr, = pv.plot([], [], color="#ffb86b", lw=2.2); lvs, = pv.plot([], [], color=CC, lw=1.8, ls="--"); lvp, = pv.plot([], [], color=CH, lw=1.8)
pv.set_ylim(-3.8, 3.8); pv.set_title("Velocities (outward positive)", fontsize=11.5, loc="left")
pvt = pv.text(0.98, 0.06, "", transform=pv.transAxes, ha="right", va="bottom", fontsize=9.2, linespacing=1.4)
lf1, = pf.plot([], [], color=FG, lw=2); lf2, = pf.plot([], [], color=CB, lw=1.6); lf3, = pf.plot([], [], color=CD, lw=1.6)
pft = pf.text(0.98, 0.94, "", transform=pf.transAxes, ha="right", va="top", fontsize=9.2, linespacing=1.4)
la1, = pa.plot([], [], color=FG, lw=2.4); la2, = pa.plot([], [], color=CS, lw=1.4, ls="--")
pat = pa.text(0.98, 0.06, "", transform=pa.transAxes, ha="right", va="bottom", fontsize=9.2)
vls = [pv.axvline(np.nan, color=CS, lw=1, ls="-."), pv.axvline(np.nan, color=MUT, lw=1, ls=":")]
th = np.linspace(0, 2*np.pi, 160)
def circ(line, rad, Rd, off=(0, 0)):
    if 0 < rad < Rd*1.05: line.set_data_3d((rad*np.cos(th) - off[0])/Rd, (rad*np.sin(th) - off[1])/Rd, 0*th)
    else: line.set_data_3d([], [], [])
def arm(r0, r1, n=700):
    r = np.linspace(r0, r1, n); f = -v_phi(r)/(r*np.where(np.abs(v_river(r)) < 1e-6, 1e-6, -v_river(r)))
    return r, np.concatenate([[0], np.cumsum(0.5*(f[1:] + f[:-1])*np.diff(r))])
AI = arm(RB, 0.995); AO = arm(1.6, 1.005)
OFFS = {"A": None, "B": None}

def frame(i):
    s = i/FPS; m, sp, Rd, act = state(s)
    if A3[0] + 2 <= s < A3[1] and (i - int((A3[0]+2)*FPS)) % 120 == 0: seed_chain()
    front = step(s, m, sp, Rd)
    for t in fig.texts: pass
    if s < A1[0] or act == 6:
        ax.set_visible(False); [p.set_visible(False) for p in (pv, pf, pa)]; cnt.set_text(""); leg.set_text(""); eqt.set_text("")
        if s < A1[0]:
            title.set_text("The riverbed of space: cell creation, absorption in matter, and the CL spiral"); sub.set_text(""); card.set_fontsize(15.5)
            card.set_text("1  Creation of space cells → Hubble’s law, with no preferred centre\n2  A mass appears: its influence spreads at c and activates the riverbed\n"
                          "3  Absorption only in matter; a slow replacement chain drifts toward M\n4  The riverbed moves matter: Newton’s law\n5  The CL spiral riverbed and its limit circle at r_c")
        else:
            title.set_text("What this animation shows — and what it assumes"); sub.set_text(""); card.set_fontsize(13.2)
            card.set_text(
                "Space cells (exact bookkeeping):  created at random at 3H per cell; absorbed only inside the matter (M, R); incompressible, so each absorbed cell is\n"
                "      replaced by its neighbours. The slow drift u = Q/(4πr²) − Hr carries a net inward flux that diminishes outward.\n\n"
                "Riverbed (acts on mass):  v_r = √(2GM/r) − Hr — the active fraction of space moving matter, as inertia couples a body to the space it moves through.\n"
                "      It is not the cell drift: it reaches r_c, while the drift turns outward at 0.25 r_c here. Its convective acceleration gives −GM/r².\n\n"
                "Einsteinian locality:  local space acts on local mass; the influence of M on space spreads at c (slowed here for visibility).\n\n"
                "Spiral riverbed:  with v_φ² = 3/2 X² − v_r², matter follows CL spirals; r_c is a limit circle — inside, spirals wind toward it; outside, they unwind away.\n\n"
                "Open inputs:  the absorption rate Q (here a display choice) and the microscopic law linking the riverbed fraction to M.\n"
                "Display: schematic units (r_c = 1, 1/H = 1); r_c lies far outside real disks (≈ 100–260 kpc in the fits).")
        return
    card.set_text(""); ax.set_visible(True); [p.set_visible(True) for p in (pv, pf, pa)]
    lim = 1.02; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim); ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=(38 if act < 5 else 38 + 35*ease((s-A5[0])/5)), azim=-60 + 4*s)
    # re-centring on chosen cells during act 1 (Hubble flow looks the same from any cell)
    off = np.zeros(3)
    if act == 1 and s > 11:
        key = "A" if s < 16.5 else "B"
        if OFFS[key] is None:
            j = int(np.argmin(np.abs(cr - (0.55 if key == "A" else 0.9)))); OFFS[key] = j
        j = min(OFFS[key], cr.size-1); P = np.array(xyz(cr[j], cp[j], cl[j]))
        w = ease((s - (11 if key == "A" else 16.5))/1.0); off = P*w
        mark.set_data_3d([(P[0]-off[0])/Rd], [(P[1]-off[1])/Rd], [(P[2]-off[2])/Rd])
    else: mark.set_data_3d([], [], [])
    x, y, z = xyz(cr, cp, cl); sc._offsets3d = ((x-off[0])/Rd, (y-off[1])/Rd, (z-off[2])/Rd)
    ch = chain & (cr < Rd); sch._offsets3d = (x[ch]/Rd, y[ch]/Rd, z[ch]/Rd) if act >= 3 else ([], [], [])
    if fb: a_ = np.array(fb); sbx._offsets3d = ((a_[:, 0]-off[0])/Rd, (a_[:, 1]-off[1])/Rd, (a_[:, 2]-off[2])/Rd)
    else: sbx._offsets3d = ([], [], [])
    if fd: a_ = np.array(fd); sdx._offsets3d = (a_[:, 0]/Rd, a_[:, 1]/Rd, a_[:, 2]/Rd)
    else: sdx._offsets3d = ([], [], [])
    if act >= 2: vm = mr < 1.0*Rd; smt._offsets3d = (mr[vm]*np.cos(mp[vm])/Rd, mr[vm]*np.sin(mp[vm])/Rd, 0*mr[vm])
    else: smt._offsets3d = ([], [], [])
    circ(bulge, RB, Rd) if m else bulge.set_data_3d([], [], [])
    circ(frontL, front, Rd) if (act == 2 and 0 < front < 1.7) else frontL.set_data_3d([], [], [])
    circ(rcL, 1.0, Rd) if act >= 4 else rcL.set_data_3d([], [], [])
    circ(ruL, RU, Rd) if act >= 3 else ruL.set_data_3d([], [], [])
    for k in range(4):
        A_, off_a = (AI if k < 2 else AO), (0 if k % 2 == 0 else np.pi)
        if sp > 0.05: arms[k].set_data_3d(A_[0]*np.cos(A_[1]+off_a)/Rd, A_[0]*np.sin(A_[1]+off_a)/Rd, 0*A_[0]); arms[k].set_alpha(sp)
        else: arms[k].set_data_3d([], [], [])
    # panels
    lvs.set_data(rr, v_space(rr) if m else H*rr)
    lvr.set_data(rr, v_river(rr)) if act >= 2 else lvr.set_data([], [])
    lvp.set_data(rr, sp*v_phi(rr)) if sp > 0.05 else lvp.set_data([], [])
    vls[0].set_xdata([1, 1] if act >= 4 else [np.nan]*2); vls[1].set_xdata([RU, RU] if act >= 3 else [np.nan]*2)
    pvt.set_text("grey dashed: space-cell drift" + ("\norange: riverbed √(2GM/r) − Hr (acts on mass)" if act >= 2 else "") + ("\nblue: CL azimuthal v_φ" if sp > 0.05 else ""))
    pf.set_xlim(0, 1.6)
    if act <= 2:
        pf.set_title("Cell budget per cell and unit time", fontsize=11.5, loc="left"); pf.set_ylim(-5, 35)
        lf2.set_data(rr, 3*H*np.ones_like(rr)); lf3.set_data(rr, absorption(rr) if m else 0*rr); lf1.set_data([], [])
        pft.set_text("green: creation 3H, everywhere" + ("\nred: absorption, only inside R" if m else ""))
    else:
        pf.set_title("Net inward flux of space through a sphere (×1/4π)", fontsize=11.5, loc="left"); pf.set_ylim(-0.03, 0.022); pf.set_xlim(0, 0.5)
        lf1.set_data(rr, influx(rr)); lf2.set_data([], []); lf3.set_data([], [])
        pft.set_text("absorbed inside R, minus space created\nwithin r: diminishes outward")
    if act == 1:
        pa.set_title("Hubble’s law: relative velocity = H × separation", fontsize=11.5, loc="left"); pa.set_ylim(-0.2, 1.8); la1.set_data(rr, H*rr); la2.set_data([], []); pat.set_text("the same from every cell")
    elif act in (2, 3, 4):
        pa.set_title("Riverbed convective acceleration v dv/dr", fontsize=11.5, loc="left"); pa.set_ylim(-14, 2)
        vv = v_river(rr); la1.set_data(rr[rr > RB], (vv*np.gradient(vv, rr))[rr > RB]); la2.set_data(rr, H**2*rr - GM/rr**2 - 0.5*H*np.sqrt(GM2/rr))
        pat.set_text("dashed: H²r − GM/r² − ½H√(2GM/r)  → Newton")
    else:
        pa.set_title("Spiral pitch of the CL riverbed  tan α = |v_r| / v_φ", fontsize=11.5, loc="left"); pa.set_ylim(0, 60)
        al = np.degrees(np.arctan(np.abs(v_river(rr))/np.maximum(v_phi(rr), 1e-6))); la1.set_data(rr, al); la2.set_data([], [])
        pat.set_text("α → 0 at r_c: the spiral becomes a circle"); pat.set_position((0.98, 0.85))
    cnt.set_text(f"space cells in view: {cr.size:5d}\ncreated:   {rate_c:5.2f} per cell per 1/H\nabsorbed:  {rate_a:5.2f} per cell per 1/H\nview: r ≤ {Rd:.2f} r_c")
    texts = {
        1: ("1 · Creation of space cells → Hubble’s law", "Cells appear at random everywhere, 3H per cell; every cell recedes from every other — no centre, no perspective",
            "∇·v = 3H   ⟹   v = H × separation", "green flash: a new cell" + ("\nblue ring: the view re-centred on this cell — the expansion looks the same" if s > 11 else "")),
        2: ("2 · A mass appears: its influence spreads at c", "Behind the front, the riverbed is active and matter (orange) is drawn in; ahead of it, matter only follows the Hubble flow",
            "riverbed: v_r = √(2GM/r) − Hr", "white dashed: influence front (c, slowed for display)\norange dots: matter"),
        3: ("3 · Absorption only in matter; a slow replacement chain", "Cells vanish only inside the bulge R; neighbours replace them, so a chain (blue) drifts slowly toward M",
            "u = Q/(4πr²) − Hr   (incompressible)", "red flash: a cell absorbed inside R\nblue: a radial chain of cells\ngrey dotted: drift balance 0.25 r_c"),
        4: ("4 · The riverbed moves matter: Newton’s law", "The active riverbed, not the slow cell drift, carries matter inward to r_c; its convective acceleration is −GM/r²",
            "v_r dv_r/dr = H²r − GM/r² − ½H√(2GM/r)", "orange: matter on the riverbed\ngold dash-dot: r_c"),
        5: ("5 · The CL spiral riverbed and its limit circle", "With v_φ² = 3/2 X² − v_r², matter spirals; at r_c the radial riverbed vanishes and the spiral turns into a circle",
            "v_φ² + v_r² = v_L²     α → 0 at r_c", "inner arms wind toward r_c; outer arms unwind away\nmatter near r_c circles"),
    }
    t_, s_, e_, l_ = texts[act]; title.set_text(t_); sub.set_text(s_); eqt.set_text(e_); leg.set_text(l_)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        want = {int(t*FPS) for t in (3, 14, 28, 50, 66, 86, 96)}
        for i in range(max(want)+1):
            if i in want: frame(i); fig.savefig(f"t_{i//FPS:02d}.png", facecolor=BG)
            else:
                s = i/FPS; m, sp, Rd, act = state(s)
                if A3[0] + 2 <= s < A3[1] and (i - int((A3[0]+2)*FPS)) % 120 == 0: seed_chain()
                step(s, m, sp, Rd)
    else:
        w_ = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1, extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"])
        with w_.saving(fig, video("riverbed_cells_matter_cl_spiral.mp4"), dpi=120):
            for i in range(min(NF, MAX_FRAMES)):
                frame(i); w_.grab_frame()
                if i % 300 == 0: print(i, NF, flush=True)
