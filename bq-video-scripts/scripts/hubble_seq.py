from paths import video, cache, MAX_FRAMES   # sets the working directory; see paths.py
import numpy as np, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.special import erfinv
from model import G, KMS_TO_KPC_PER_MYR
from model2L import velocities_2L, streamline

H = 0.07                       # z = 0
RD = 15.0
FPS = 30
rng = np.random.default_rng(5)

# ---------------- stages of the tuning fork --------------------------------------------
# wiso: isotropic cone occupation (E0); sig: width of cone occupation about the equator (deg)
# kap: azimuthal-flow strength (Eq. 49: |v_phi| <~ |v_r| for spheroids); arms: streamline highlight;
# bar: 0 -> single Lagrangian, 1 -> two-Lagrangian nested spiral (R1 = 0.22 R2, M1 = 0.07 M2, NGC 3741 ratios)
S = dict(
 E0 =dict(M=1e10,R=3.0,wiso=1,sig=35,kap=0.20,arms=0,bar=0, pos=(0,0)),
 E5 =dict(M=1e10,R=3.0,wiso=0,sig=30,kap=0.35,arms=0,bar=0, pos=(1,0)),
 S0 =dict(M=1e10,R=3.0,wiso=0,sig=8, kap=1.0, arms=0,bar=0, pos=(2,0)),
 Sa =dict(M=1e7, R=1.2,wiso=0,sig=2.5,kap=1.0,arms=1,bar=0, pos=(3,0.7)),
 Sb =dict(M=1e9, R=2.5,wiso=0,sig=2.5,kap=1.0,arms=1,bar=0, pos=(4,0.7)),
 Sc =dict(M=1e10,R=4.0,wiso=0,sig=2.5,kap=1.0,arms=1,bar=0, pos=(5,0.7)),
 SBa=dict(M=1e7, R=1.2,wiso=0,sig=2.5,kap=1.0,arms=1,bar=1, pos=(3,-0.7)),
 SBb=dict(M=1e9, R=2.5,wiso=0,sig=2.5,kap=1.0,arms=1,bar=1, pos=(4,-0.7)),
 SBc=dict(M=1e10,R=4.0,wiso=0,sig=2.5,kap=1.0,arms=1,bar=1, pos=(5,-0.7)),
)
ORDER = ["E0","E5","S0","Sa","Sb","Sc","S0","SBa","SBb","SBc"]
HOLD, MORPH, RETURN = 2.4, 2.0, 2.6
segs = []; t = 0.0
for k, name in enumerate(ORDER):
    segs.append((t, t+HOLD, name, name)); t += HOLD
    if k < len(ORDER)-1:
        d = RETURN if (name == "Sc") else MORPH
        segs.append((t, t+d, name, ORDER[k+1])); t += d
segs.append((t, t+1.5, "SBc", "SBc")); T_TOTAL = t+1.5
NF = int(T_TOTAL*FPS)

def ease(u): return 0.5-0.5*np.cos(np.pi*np.clip(u,0,1))
def params(s):
    for a,b,A,B in segs:
        if a <= s < b or (s >= b and (a,b,A,B) == segs[-1]): u = ease((s-a)/(b-a)); break
    PA, PB = S[A], S[B]
    lin = lambda k: PA[k]*(1-u) + PB[k]*u
    p = {k: lin(k) for k in ("R","wiso","sig","kap","arms","bar")}
    p["M"] = np.exp(np.log(PA["M"])*(1-u) + np.log(PB["M"])*u)
    p["pos"] = tuple(np.array(PA["pos"])*(1-u) + np.array(PB["pos"])*u)
    p["label"] = A if u < 0.5 else B
    p["R2"], p["M2"] = p["R"], p["M"]
    p["R1"], p["M1"] = p["R"]*0.22**p["bar"], p["M"]*0.07**p["bar"]
    p["rc"] = (2*G*p["M"]/H**2)**(1/3)
    p["rout"] = min(RD, 0.985*p["rc"])
    return p

def vel_cone(r, sinth, p):
    """v_r (theta-independent) and conic v_phi (MC paper Eqs. 69-71), two-L piecewise, scaled by kap."""
    R1,M1,R2,M2 = p["R1"],p["M1"],p["R2"],p["M2"]
    _, vr = velocities_2L(r, R1, M1, R2, M2, H)
    X1 = np.sqrt(2*G*M1/R1) - H*R1; X2 = np.sqrt(2*G*M2/R2) - H*R2
    c = 0.5*(2 + sinth)
    vphi2 = np.where(r <= R1, 0.5*sinth*X1**2*r**2/R1**2,
             np.where(r <= R2, c*X1**2 - vr**2, c*X2**2 - vr**2))
    return vr, p["kap"]*np.sqrt(np.maximum(vphi2, 0))

# ---------------- tracers -------------------------------------------------------------
ND, NBG = 9000, 1800
def travel_table(p):
    tt, rt = [0.0], [p["rout"]]
    while rt[-1] > 0.05:
        v = max(float(velocities_2L(rt[-1], p["R1"],p["M1"],p["R2"],p["M2"],H)[1]), 0.3)
        rt.append(rt[-1]-0.02); tt.append(tt[-1] + 0.02/(v*KMS_TO_KPC_PER_MYR))
    return np.array(tt), np.array(rt)
p0 = params(0.0); TT, RT = travel_table(p0)
def steady(n): return np.interp(rng.random(n)*TT[-1], TT, RT)
# disk population: quantile u -> cone latitude via current occupation; side s = +-1
d_r = steady(ND); d_p = 2*np.pi*rng.random(ND); d_u = rng.random(ND); d_s = rng.choice([-1,1], ND)
# bulge population: isotropic cones, fed at r = R1
b_r = p0["R1"]*rng.random(NBG)**0.6; b_p = 2*np.pi*rng.random(NBG); b_u = rng.random(NBG); b_s = rng.choice([-1,1], NBG)
NA = 1100; a_r = steady(NA); a_id = rng.integers(0,2,NA)

def latitude(u, p):
    half = np.minimum(np.radians(p["sig"])*np.sqrt(2)*erfinv(np.minimum(u, 0.995)), np.pi/2*0.98)
    return (1-p["wiso"])*half + p["wiso"]*np.arcsin(u)

def clock(p):
    vL = np.sqrt(1.5)*(np.sqrt(2*G*p["M"]/p["R"]) - H*p["R"])
    return float(np.clip(3.0*250/max(vL, 1), 3.0, 45.0))     # Myr/frame

def step(p):
    global d_r, d_p, b_r, b_p, a_r
    dt = clock(p)
    for arr_r, arr_p, lat_u, sgn, which in ((d_r, d_p, d_u, d_s, "d"), (b_r, b_p, b_u, b_s, "b")):
        lat = latitude(lat_u, p) if which == "d" else np.arcsin(lat_u)
        sinth = np.cos(lat)                                  # sin(theta0), theta0 = pi/2 - lat
        for _ in range(3):
            h = dt/3
            vr, vp = vel_cone(arr_r, sinth, p)
            arr_r -= h*vr*KMS_TO_KPC_PER_MYR
            np.maximum(arr_r, 1e-3, out=arr_r)
            arr_p += h*vp/(arr_r*np.maximum(sinth, 0.08))*KMS_TO_KPC_PER_MYR
    dead = (d_r < 0.05) | (d_r > p["rout"]*1.001)
    d_r[dead] = p["rout"]*(0.97+0.03*rng.random(dead.sum())); d_p[dead] = 2*np.pi*rng.random(dead.sum())
    d_u[dead] = rng.random(dead.sum()); d_s[dead] = rng.choice([-1,1], dead.sum())
    bd = (b_r < 0.03) | (b_r > p["R1"]*1.05)
    b_r[bd] = p["R1"]*(0.9+0.1*rng.random(bd.sum())); b_p[bd] = 2*np.pi*rng.random(bd.sum())
    for _ in range(3):
        _, vr = velocities_2L(a_r, p["R1"],p["M1"],p["R2"],p["M2"],H); a_r -= dt/3*vr*KMS_TO_KPC_PER_MYR
    ad = (a_r < 0.05) | (a_r > p["rout"]); a_r[ad] = p["rout"]*(0.93+0.05*rng.random(ad.sum()))
    return dt

def xyz(r, ph, lat, s):
    cyl = r*np.cos(lat); return cyl*np.cos(ph), cyl*np.sin(ph), s*r*np.sin(lat)

for _ in range(150): step(p0)

# ---------------- figure ------------------------------------------------------------------
BG, FG, MUT = "#05070d", "#e6e9ef", "#8a93a6"
CD, CB, CA, CR = "#f2a24a", "#fff0cf", "#ffb86b", "#feca57"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":FG,"axes.labelcolor":FG,
                     "xtick.color":MUT,"ytick.color":MUT,"axes.edgecolor":"#2c3445"})
fig = plt.figure(figsize=(16,9), dpi=120, facecolor=BG)
ax = fig.add_axes([-0.07,-0.05,0.74,1.05], projection="3d", facecolor=BG); ax.set_axis_off()
L = 13.5; ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_zlim(-L*0.75,L*0.75); ax.set_box_aspect((1,1,0.75))
sc_d = ax.scatter([],[],[], s=1.7, lw=0, depthshade=False)
sc_b = ax.scatter([],[],[], s=2.2, c=CB, alpha=0.5, lw=0, depthshade=False)
sc_a = ax.scatter([],[],[], s=6, c=CA, lw=0, depthshade=False)
armL = [ax.plot([],[],[], color=CA, lw=1.3)[0] for _ in range(2)]
ringL, = ax.plot([],[],[], color=CR, lw=1.2, ls="-.")

fig.text(0.03, 0.935, "The Hubble sequence from cone occupation of the PG flow", fontsize=21, weight="bold")
fig.text(0.03, 0.897, "de Haas metric-inflow model: dθ/dt = 0 streamlines on cones θ₀, v_r = √(2GM/r) − Hr, "
         "v_φ² = ½(2+sin θ₀) v_r²(R) − v_r²(r) · z = 0", fontsize=11.5, color=MUT)
stage_txt = fig.text(0.03, 0.83, "", fontsize=30, weight="bold", color=CD)
desc_txt = fig.text(0.03, 0.79, "", fontsize=13, color=FG)

# tuning fork
axF = fig.add_axes([0.665, 0.64, 0.31, 0.25], facecolor=BG); axF.set_axis_off()
axF.set_xlim(-0.5, 5.6); axF.set_ylim(-1.25, 1.25)
axF.plot([0,2],[0,0], color=MUT, lw=1.5); axF.plot([2,3,5],[0,0.7,0.7], color=MUT, lw=1.5); axF.plot([2,3,5],[0,-0.7,-0.7], color=MUT, lw=1.5)
for n in ("E0","E5","S0","Sa","Sb","Sc","SBa","SBb","SBc"):
    x,y = S[n]["pos"]; axF.plot([x],[y],"o", color="#2c3445", ms=9)
    axF.text(x, y+(0.28 if y >= 0 else -0.4), n, ha="center", color=FG, fontsize=11)
axF.text(4, 1.18, "arms: tight  →  open   (M, R ↑)", ha="center", color=MUT, fontsize=9.5)
mk, = axF.plot([],[],"o", color=CD, ms=13, alpha=0.9)
# occupation
axO = fig.add_axes([0.69, 0.37, 0.28, 0.19], facecolor=BG)
axO.set_title("Cone occupation  (latitude 90° − θ₀)", fontsize=12, loc="left")
axO.set_xlim(-90, 90); axO.set_ylim(0, 1.1); axO.set_yticks([]); axO.set_xlabel("latitude (deg)")
axO.set_xticks([-90,-45,0,45,90]); axO.grid(alpha=0.12)
occ, = axO.plot([],[], color=CD, lw=2); occf = [None]
# pitch
axP = fig.add_axes([0.69, 0.09, 0.28, 0.19], facecolor=BG)
axP.set_title("Equatorial pitch  tan α = v_r / v_φ", fontsize=12, loc="left")
axP.set_xlim(0, RD); axP.set_ylim(0, 90); axP.set_xlabel("r (kpc)"); axP.set_ylabel("α (deg)"); axP.grid(alpha=0.12)
pit, = axP.plot([],[], color=CA, lw=2); pit2, = axP.plot([],[], color=CA, lw=2)
info = fig.text(0.03, 0.08, "", fontsize=10.5, family="DejaVu Sans Mono", color=FG, linespacing=1.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.6", fc="#10151f", ec="#2c3445"))
fig.text(0.03, 0.02, "Each dot stays on its cone θ₀. Bars: two-Lagrangian nested spiral (R₁ = 0.22 R₂, M₁ = 0.07 M₂) ending in a ring. "
         "Display choices: stage parameters, occupation widths, κ for ellipticals; adaptive tracer clock.", fontsize=9.3, color=MUT)

DESC = {"E0":"isotropic cone occupation, weak azimuthal flow (|v_φ| ≲ |v_r|)",
        "E5":"broad but flattened occupation, weak rotation",
        "S0":"several adjacent cones: thick rotating disk, no arm pattern",
        "Sa":"thin equatorial cone; small M, R: r_c near the disk, arms wind tight",
        "Sb":"thin equatorial cone; intermediate M, R",
        "Sc":"thin equatorial cone; large M, R: r_c far out, arms stay open",
        "SBa":"nested spiral (bar zone) + ring at R₂ + tight outer arms",
        "SBb":"nested spiral (bar zone) + ring at R₂ + outer disk spiral",
        "SBc":"nested spiral (bar zone) + ring at R₂ + open outer arms"}
cm = plt.get_cmap("YlOrBr_r")
th = np.linspace(0, 2*np.pi, 200)

def frame(i):
    s = i/FPS; p = params(s)
    global TT, RT
    if i % 6 == 0: TT, RT = travel_table(p)
    dt = step(p)
    lat_d = latitude(d_u, p); x,y,z = xyz(d_r, d_p, lat_d, d_s)
    sc_d._offsets3d = (x,y,z); sc_d.set_color(cm(0.15 + 0.6*np.clip(d_r/RD,0,1))); sc_d.set_alpha(0.6)
    lat_b = np.arcsin(b_u); bx,by,bz = xyz(b_r, b_p, lat_b, b_s); sc_b._offsets3d = (bx,by,bz)
    # arms (equatorial streamline, two-L)
    rs, ps = streamline(p["R1"],p["M1"],p["R2"],p["M2"],H, p["rout"])
    wA = p["arms"]
    for k, off in enumerate((0, np.pi)):
        armL[k].set_data_3d(rs*np.cos(ps+off), rs*np.sin(ps+off), 0*rs); armL[k].set_alpha(0.7*wA)
    ap = np.interp(a_r, rs, ps) + np.pi*a_id
    sc_a._offsets3d = (a_r*np.cos(ap), a_r*np.sin(ap), 0*a_r); sc_a.set_alpha(0.95*wA)
    ringL.set_data_3d(p["R2"]*np.cos(th), p["R2"]*np.sin(th), 0*th); ringL.set_alpha(0.9*p["bar"])
    ax.view_init(elev=28 + 6*np.sin(2*np.pi*s/30), azim=-60 + 360*s/45)
    # panels
    mk.set_data([p["pos"][0]], [p["pos"][1]])
    allat = np.degrees(np.r_[lat_d*d_s, lat_b*b_s])
    h, e = np.histogram(allat, bins=72, range=(-90,90)); h = h/ max(h.max(),1)
    occ.set_data(0.5*(e[1:]+e[:-1]), h)
    if occf[0] is not None: occf[0].remove()
    occf[0] = axO.fill_between(0.5*(e[1:]+e[:-1]), h, color=CD, alpha=0.25, lw=0)
    ra = np.linspace(0.05, min(p["R2"], p["rout"]), 200); rb = np.linspace(p["R2"]*(1+1e-6), p["rout"], 400)
    for ln, rr in ((pit, ra), (pit2, rb)):
        vr, vp = vel_cone(rr, 1.0, p); ln.set_data(rr, np.degrees(np.arctan2(vr, vp)))
    stage_txt.set_text(p["label"]); desc_txt.set_text(DESC[p["label"]])
    vr10, vp10 = vel_cone(np.array([10.0]), 1.0, p)
    a10 = np.degrees(np.arctan2(vr10, vp10))[0] if 10 < p["rout"] else float("nan")
    info.set_text(f"M = {p['M']:.2e} M⊙   R = {p['R']:.2f} kpc   r_c = {p['rc']:6.1f} kpc\n"
                  f"cone width σ = {p['sig']:4.1f}°   isotropic = {p['wiso']:.2f}   κ = {p['kap']:.2f}\n"
                  f"α(10 kpc) = {a10:4.1f}°   bar (two-L) = {p['bar']:.2f}   clock {dt:4.1f} Myr/frame")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        want = {int(FPS*(a+0.5*(b-a))): A for a,b,A,B in segs if A == B}
        for i in range(NF):
            if i in want: frame(i); fig.savefig(f"hs_{want[i]}_{i}.png", facecolor=BG)
            else: step(params(i/FPS))
    else:
        w = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1, extra_args=["-pix_fmt","yuv420p","-crf","18","-preset","medium"])
        with w.saving(fig, video("hubble_sequence_pg_cones.mp4"), dpi=120):
            for i in range(min(NF, MAX_FRAMES)):
                frame(i); w.grab_frame()
                if i % 100 == 0: print(i, NF, flush=True)
