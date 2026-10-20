"""
cl_vacuum_flow_video.py -- the motion model of the page models/CL_Vacuum_Flow.html (27 s, 1280x720, 30 fps).

The vacuum flow around a rotating bulge: from the same border value, three outward laws carried by the same radial
river v_r = sqrt(2GM/r): CL (Bernoulli, v_phi^2 = L - 2GM/r), GR's Lense-Thirring frame dragging (Omega ~ r^-3) and a
Kepler-like rotation. Tracers follow the space flow; their colour is the Bernoulli value L along the streamline
(green = the border value kept). The argument rests on one postulate: the vacuum behaves as an ideal, barotropic,
non-dissipative flow; dissipation needs matter.

Writes ../videos/CL_Vacuum_Flow.mp4. Quick test: BQ_MAX_FRAMES=60 python cl_vacuum_flow_video.py
"""
import paths
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

OUT = paths.video('CL_Vacuum_Flow.mp4')
FPS, W, H_ = 30, 12.8, 7.2
BG, INK, SOFT, EDGE = '#05070d', '#e7ebf3', '#97a2b9', '#222c42'
C = {'CL': '#5fd38a', 'GR': '#a29bfe', 'KE': '#48dbfb'}
ST, BAD, GOOD = '#f2a24a', '#ff7b7b', '#5fd38a'
vR, H0, XMAX = 1.0, 3.0, 6.0
vr = lambda x: np.sqrt(2 / x)
VP = {'CL': lambda x: np.sqrt(np.maximum(0, H0 - 2 / x)), 'GR': lambda x: vR / x**2, 'KE': lambda x: vR / np.sqrt(x)}
NAME = {'CL': 'CL: Bernoulli', 'GR': 'GR: Lense–Thirring', 'KE': 'Kepler-like'}
VERD = {'CL': ('L kept', 'laminar (r $v_\\varphi$ rises)', 'not a GR vacuum'), 'GR': ('L lost', 'Rayleigh-unstable (r $v_\\varphi$ falls)', 'GR vacuum'),
        'KE': ('L lost', 'laminar', 'not a GR vacuum')}
OKS = {'CL': (1, 1, 0), 'GR': (0, 0, 1), 'KE': (0, 1, 0)}

T_INTRO, T_MAIN, T_END = 5.0, 16.0, 6.0
NT = min(int((T_INTRO + T_MAIN + T_END) * FPS), paths.MAX_FRAMES)
rng = np.random.default_rng(3)
N = 320
P = {k: [1 + rng.random(N) * (XMAX - 1), rng.random(N) * 2 * np.pi] for k in C}
turb = [np.sqrt(rng.random(70)) * 0.95, rng.random(70) * 2 * np.pi, rng.random(70) * 6.28]

def col(rel):
    a = np.clip(np.abs(rel) / 0.5, 0, 1)
    g, b = np.array(matplotlib.colors.to_rgb(GOOD)), np.array(matplotlib.colors.to_rgb(BAD))
    return g[None, :] * (1 - a[:, None]) + b[None, :] * a[:, None]

fig = plt.figure(figsize=(W, H_), dpi=100, facecolor=BG)
writer = FFMpegWriter(fps=FPS, bitrate=4500, extra_args=['-pix_fmt', 'yuv420p'])
dt = 1 / FPS
with writer.saving(fig, OUT, dpi=100):
    t, ang = 0.0, 0.0
    for n in range(NT):
        fig.clf(); fig.set_facecolor(BG)
        t = n * dt
        if t < T_INTRO:
            a = min(1, t / 1.0, (T_INTRO - t) / 0.6)
            fig.text(0.5, 0.74, 'The vacuum flow around a rotating bulge', ha='center', color=INK, fontsize=30, family='serif', alpha=a)
            fig.text(0.5, 0.62, 'Bernoulli, Rayleigh and the CL boundary condition', ha='center', color=SOFT, fontsize=18, alpha=a)
            fig.text(0.5, 0.44, 'The postulate', ha='center', color=ST, fontsize=15, weight='bold', alpha=a)
            fig.text(0.5, 0.34, 'The vacuum behaves as an ideal, barotropic, non-dissipative flow:\nthe river and analogue-gravity picture taken as physics, not as an analogy.',
                     ha='center', va='center', color=INK, fontsize=18, family='serif', alpha=a, linespacing=1.5)
            fig.text(0.5, 0.18, 'For dissipation, matter is needed, as the source of turbulence.', ha='center', color=SOFT, fontsize=15, alpha=a)
        elif t < T_INTRO + T_MAIN:
            tm = t - T_INTRO
            fig.text(0.5, 0.94, 'The same border value, three outward laws', ha='center', color=INK, fontsize=20, family='serif')
            fig.text(0.5, 0.895, 'tracers: the flow of space (inflow $v_r$ = √(2GM/r), rotation $v_\\varphi$);  colour: Bernoulli value along the streamline, green = border value kept',
                     ha='center', color=SOFT, fontsize=11)
            ang += vR * dt * 0.9
            for i, k in enumerate(['CL', 'GR', 'KE']):
                ax = fig.add_axes([0.02 + i * 0.33, 0.2, 0.30, 0.64]); ax.set_facecolor(BG)
                ax.set_xlim(-XMAX * 1.03, XMAX * 1.03); ax.set_ylim(-XMAX * 1.03, XMAX * 1.03); ax.set_aspect('equal'); ax.axis('off')
                for rr in (2, 4, 6): ax.add_patch(Circle((0, 0), rr, fill=False, ec=EDGE, lw=1))
                x, a_ = P[k]
                x -= vr(x) * dt * 0.9
                a_ += VP[k](np.maximum(1, x)) / x * dt * 0.9
                dead = x <= 1
                x[dead] = XMAX; a_[dead] = rng.random(dead.sum()) * 2 * np.pi
                rel = ((vr(x)**2 + VP[k](x)**2) - H0) / H0
                ax.scatter(x * np.cos(a_), x * np.sin(a_), s=6, c=col(rel), linewidths=0)
                ax.add_patch(Circle((0, 0), 1, color=ST, alpha=0.2)); ax.add_patch(Circle((0, 0), 1, fill=False, ec=ST, lw=2))
                for j in range(4):
                    aa = ang + j * np.pi / 2; ax.plot([0, 0.9 * np.cos(aa)], [0, 0.9 * np.sin(aa)], color=ST, lw=1.6)
                tr, ta, tp = turb
                rr_ = tr + 0.05 * np.sin(5 * tm + 2 * tp); aa_ = ta + ang + 0.25 * np.sin(3 * tm + tp)
                ax.scatter(rr_ * np.cos(aa_), rr_ * np.sin(aa_), s=3, c=ST, linewidths=0)
                ax.set_title(NAME[k], color=C[k], fontsize=16, family='serif', pad=4)
                ok = OKS[k]
                for j, txt in enumerate(VERD[k]):
                    fig.text(0.17 + i * 0.33, 0.145 - j * 0.04, ('✓ ' if ok[j] else '✗ ') + txt, ha='center', color=GOOD if ok[j] else BAD, fontsize=12.5)
            fig.text(0.5, 0.02, 'inside R: matter of the bulge, rigid and turbulent (dissipation possible)   ·   outside R: pure vacuum (no dissipation)', ha='center', color=ST, fontsize=11.5)
        else:
            te = t - T_INTRO - T_MAIN
            a = min(1, te / 0.8, (T_END - te) / 0.5)
            fig.text(0.5, 0.78, 'Outside the bulge there is no matter, so nothing can dissipate:', ha='center', color=INK, fontsize=19, family='serif', alpha=a)
            fig.text(0.5, 0.70, 'the border energy is carried along the spiral streamlines.', ha='center', color=INK, fontsize=19, family='serif', alpha=a)
            fig.text(0.5, 0.56, 'Bernoulli fixes  $v_\\varphi^2$ = 3GM/R − 2GM/r :  the CL flow,  $v_\\varphi$ → √(3GM/R)', ha='center', color=GOOD, fontsize=18, alpha=a)
            fig.text(0.5, 0.47, 'Rayleigh: r $v_\\varphi$ rises, so the flow stays laminar', ha='center', color=GOOD, fontsize=16, alpha=a)
            fig.text(0.5, 0.35, 'GR’s vacuum law (r⁴Ω′)′ = 0 gives Lense–Thirring, Ω ∝ r⁻³:\nno Bernoulli, and read as a flow, Rayleigh-unstable.', ha='center', color=C['GR'], fontsize=15, alpha=a, linespacing=1.5)
            fig.text(0.5, 0.18, 'Which law holds is the postulate: the vacuum as an ideal, barotropic, non-dissipative flow.', ha='center', color=ST, fontsize=15, alpha=a)
        writer.grab_frame()
print('written', OUT, NT, 'frames')
