"""
cl_vacuum_flow.py -- the vacuum flow at the border of a rotating bulge, as on the page models/CL_Vacuum_Flow.html.

Setting (slow flow, units GM = R = 1, x = r/R): a bulge of radius R rotates rigidly and drags space with it. All three
continuations are carried by the same radial river v_r = sqrt(2GM/r) and start from the same border value v_phi(R):
    CL (Bernoulli)       v_r^2 + v_phi^2 = L = 2GM/R + v_phi(R)^2   =>  v_phi = sqrt(L - 2GM/r) -> sqrt(L) = v_L
    GR (Lense-Thirring)  (r^4 Omega')' = 0, Omega(R) continuous, Omega -> 0  =>  Omega = Omega(R)(R/r)^3, v_phi = v_phi(R) R^2/r^2
    Kepler-like          v_phi = v_phi(R) sqrt(R/r)
L is the Bernoulli value along a streamline, the constant Lagrangian of the CL papers (L = 3GM/R for v_phi(R) = sqrt(GM/R)).

The postulate: the vacuum behaves as an ideal, barotropic, non-dissipative flow (the river and analogue-gravity picture
taken as physics rather than as an analogy). Dissipation needs matter, the source of turbulence: inside R the matter
rotates rigidly; outside R the pure vacuum cannot dissipate the energy it receives at the border.

Checked exactly with SymPy:
  1  Bernoulli: L = v_r^2 + v_phi^2 is constant for CL and equals the border value; it is not constant for GR or Kepler
  2  uniqueness: with the river v_r fixed and L constant along the (spiral) streamlines, v_phi = sqrt(L - 2GM/r) is the
     only nonnegative profile (for v_r = 0 the streamlines are circles and Bernoulli fixes no profile; printed as a note)
  3  Rayleigh: the specific angular momentum l = r v_phi rises outward for CL and Kepler (stable), falls for GR (unstable)
  4  GR's vacuum momentum constraint (r^4 Omega')' = 0 holds for Lense-Thirring only; CL and Kepler need a source
  5  the second-order energy term -r^2 Omega'^2/4 is nonzero for all three, so it does not separate the branches
  6  continuity at the border: v_phi(R) is the same in all three and equals the rigid interior value
and writes the four charts of the page to ../figures/.

Usage
  python cl_vacuum_flow.py
"""
import os
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.normpath(os.path.join(HERE, '..', 'figures'))


def report(name, ok):
    print(f'  {name:<88} {"OK (exact)" if ok else "FAILS"}')
    return ok


def main():
    ok = True
    x, w = sp.symbols('x w', positive=True)           # x = r/R, w = v_phi(R) in units sqrt(GM/R)
    vr = sp.sqrt(2 / x)
    L = 2 + w**2
    vp = {'CL': sp.sqrt(L - 2 / x), 'GR': w / x**2, 'KE': w / sp.sqrt(x)}
    Bern = {k: sp.simplify(vr**2 + v**2) for k, v in vp.items()}
    ell = {k: x * v for k, v in vp.items()}
    Om = {k: v / x for k, v in vp.items()}

    print('1  Bernoulli along the streamline')
    ok &= report('CL: v_r^2 + v_phi^2 = L = 2 + v_phi(R)^2 at every radius', sp.simplify(Bern['CL'] - L) == 0)
    ok &= report('GR and Kepler: v_r^2 + v_phi^2 changes with radius', all(sp.simplify(sp.diff(Bern[k], x)) != 0 for k in ('GR', 'KE')))
    print('2  Uniqueness')
    f = sp.Function('f')
    sol = sp.solve(sp.Eq(vr**2 + f(x)**2, L), f(x))
    ok &= report('v_r fixed, L constant: v_phi = +sqrt(L - 2/x) is the only nonnegative solution',
                 any(sp.simplify(s - vp['CL']) == 0 for s in sol) and len(sol) == 2)
    print('  note: with v_r = 0 the streamlines are circles, Bernoulli holds along each circle for any v_phi(r)')
    print('        and fixes no profile; the radial river is what carries L from the border outward')
    print('3  Rayleigh stability: d(r v_phi)/dr')
    dl = {k: sp.simplify(sp.diff(ell[k], x)) for k in ell}
    ok &= report('CL: d(l^2)/dx = 2 L x - 2 > 0 for x >= 1', sp.simplify(sp.diff(ell['CL']**2, x) - (2 * L * x - 2)) == 0)
    ok &= report('Kepler: l = w sqrt(x) rises', sp.simplify(dl['KE'] - w / (2 * sp.sqrt(x))) == 0)
    ok &= report('GR: l = w/x falls (Rayleigh-unstable read as a flow)', sp.simplify(dl['GR'] + w / x**2) == 0)
    print("4  GR's vacuum momentum constraint (r^4 Omega')' = 0")
    res = {k: sp.simplify(sp.diff(x**4 * sp.diff(Om[k], x), x)) for k in Om}
    ok &= report('Lense-Thirring: (x^4 Omega\')\' = 0', res['GR'] == 0)
    ok &= report('CL and Kepler: (x^4 Omega\')\' != 0 (GR would need a mass current outside R)', res['CL'] != 0 and res['KE'] != 0)
    print('5  Second-order energy term')
    e2 = {k: sp.simplify(-x**2 * sp.diff(Om[k], x)**2 / 4) for k in Om}
    ok &= report('-x^2 Omega\'^2/4 nonzero for all three: it does not separate the branches', all(e != 0 for e in e2.values()))
    print('6  Continuity at the border')
    ok &= report('v_phi(R) = w in all three, equal to the rigid interior w x at x = 1', all(sp.simplify(v.subs(x, 1) - w) == 0 for v in vp.values()))
    print('\nAll checks:', 'OK' if ok else 'SOME FAIL')
    figures(vp, vr, Bern, ell, res, x, w)
    return ok


def figures(vp, vr, Bern, ell, res, x, w):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available: figures skipped')
        return
    os.makedirs(FIG, exist_ok=True)
    col = {'CL': '#2f8a4f', 'GR': '#5b4fd6', 'KE': '#0f7fb0'}
    lab = {'CL': 'CL (Bernoulli)', 'GR': 'GR, Lense-Thirring', 'KE': 'Kepler-like'}
    X = np.linspace(1, 12, 400)
    Xi = np.linspace(0, 1, 50)
    sub = {w: 1}
    panels = [('v_phi', vp, 'rotation of space v_phi(r)', 'vphi.png', lambda t: t),
              ('L', Bern, 'Bernoulli value L = v_r^2 + v_phi^2', 'bernoulli.png', None),
              ('r v_phi', ell, 'Rayleigh: specific angular momentum r v_phi', 'rayleigh.png', lambda t: t * t),
              ('(r^4 Omega\')\'/(2r^3)', {k: res[k] / (2 * x**3) for k in res}, "GR source needed (r^4 Omega')'/(2 r^3)", 'gr_source.png', None)]
    for ylab, fam, title, fn, inside in panels:
        fig, ax = plt.subplots(figsize=(6, 3.8), dpi=150)
        ax.axvspan(0, 1, color='#b8620f', alpha=0.1)
        for k in ('CL', 'GR', 'KE'):
            fnum = sp.lambdify(x, sp.sympify(fam[k]).subs(sub), 'numpy')
            ax.plot(X, np.broadcast_to(fnum(X), X.shape), color=col[k], lw=2, label=lab[k])
        if inside is not None:
            ax.plot(Xi, inside(Xi), color='#b8620f', lw=2, label='bulge, rigid matter')
        if fn == 'bernoulli.png':
            ax.axhline(3, color='#2f8a4f', ls='--', lw=1, alpha=0.7)
        ax.set_xlabel('r / R'); ax.set_ylabel(ylab); ax.set_title(title, fontsize=10); ax.set_xlim(0, 12)
        ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(FIG, fn)); plt.close(fig)
    print('figures written to', FIG)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
