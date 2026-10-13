"""Figures for the paper (PDF, in ../fig)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from cl_lelli_analysis import D, G, HZ, v2_cl, v_atan, load_massmodels, bisector
plt.rcParams.update({'font.family': 'serif', 'font.size': 9.5, 'axes.linewidth': .8, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'legend.frameon': False, 'savefig.bbox': 'tight', 'mathtext.fontset': 'dejavuserif'})
C1, C2, C3, GR = '#2a78d6', '#eb6834', '#1baf7a', '#6b6a66'
F = D.parent/'fig' if (D.parent/'fig').exists() else D.parent/'paper'/'fig'; F.mkdir(exist_ok=True)
df = pd.read_csv(D/'galaxies.csv'); MM = load_massmodels(); R = json.load(open(D/'results.json'))
S = df[(df.Q <= 2) & (df.Inc >= 30) & (~df.unconstrained)].copy(); L = np.log10
bul = S.Lb > 0

# ---- Fig 1: example fits
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0))
for a, g in zip(ax.flat, ['NGC6503', 'NGC2976', 'F568-3', 'UGC07524']):
    row = df[df.Galaxy == g].iloc[0]; d = MM[MM.ID == g].sort_values('R')
    rr = np.linspace(1e-3, d.R.max()*1.05, 400); M, Rr = 10**row.logM, 10**row.logR
    a.errorbar(d.R, d.Vobs, d.eV, fmt='o', ms=3, color='k', lw=.8, capsize=0, label='SPARC')
    a.plot(rr, np.sqrt(np.maximum(v2_cl(rr, M, Rr), 0)), color=C1, lw=1.6, label='CL fit')
    a.plot(rr, v_atan(rr, row.Vc, row.rt), color=C2, lw=1.3, ls='--', label='arctan fit')
    a.axvline(Rr, color=C1, lw=.8, ls=':'); a.axhline(row.vL/np.sqrt(3), color=C1, lw=.6, ls=':')
    a.plot(rr[rr < Rr*1.3], row.s_CL*rr[rr < Rr*1.3], color=C3, lw=1.1, label=r'$s_{\rm CL}\,r$')
    a.set_title(f"{g}  ($T={int(row['T'])}$)", fontsize=9.5); a.set_xlim(0, d.R.max()*1.05); a.set_ylim(0, max(d.Vobs+d.eV)*1.15)
    a.text(Rr, a.get_ylim()[1]*0.04, r' $R$', color=C1, fontsize=9)
for a in ax[1]: a.set_xlabel(r'$r$ [kpc]')
for a in ax[:, 0]: a.set_ylabel(r'$V$ [km s$^{-1}$]')
ax[0, 0].legend(loc='lower right', fontsize=8)
fig.tight_layout(); fig.savefig(F/'fig1_examples.pdf'); plt.close(fig)

# ---- Fig 2: gradient-SB relation for data, CL, arctan
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for a, (col, lab, key) in zip(ax, [('s_data', r'$s_{\rm data}$ (first 3 points)', 's_data|SB_inner'), ('s_CL', r'$s_{\rm CL}=v_L/\sqrt{3}R$', 's_CL|SB_inner'), ('s_atan', r'$s_{\rm atan}=2V_c/\pi r_t$', 's_atan|SB_inner')]):
    x, y = L(S.SB_inner), L(S[col]); st = R['gradient_SB'][key]
    a.scatter(x[~bul], y[~bul], s=10, color=C1, lw=0, alpha=.85, label='no bulge'); a.scatter(x[bul], y[bul], s=14, color=C2, marker='^', lw=0, label='bulge')
    xx = np.linspace(x.min()-.1, x.max()+.1, 10); xm, ym = np.median(x), np.median(y)
    a.plot(xx, st['intercept_bis'] + st['bisector']*xx, color='k', lw=1.3, label=f"bisector {st['bisector']:.2f}")
    a.plot(xx, ym + 0.55*(xx-xm), color=C3, lw=1.1, ls='--', label='Lelli+13 slope 0.55')
    a.plot(xx, ym + 0.5*(xx-xm), color=GR, lw=.9, ls=':', label='theory 0.5')
    a.set_title(lab, fontsize=9); a.set_xlabel(r'$\log\,\Sigma_{\rm in}$ [L$_\odot$ pc$^{-2}$, 3.6 $\mu$m]')
    a.text(.04, .93, rf"$\rho={st['rho']:.2f}$, $\sigma={st['scatter_bis']:.2f}$", transform=a.transAxes, fontsize=8.5)
ax[0].set_ylabel(r'$\log\,s$ [km s$^{-1}$ kpc$^{-1}$]')
h, l = ax[0].get_legend_handles_labels(); l[2] = 'bisector fit'
fig.legend(h, l, loc='lower center', ncol=5, fontsize=8, bbox_to_anchor=(.5, -.02))
fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(F/'fig2_gradient_SB.pdf'); plt.close(fig)

# ---- Fig 3: gradient estimators against the data gradient
fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.9), sharey=True)
for a, (col, lab, c, k) in zip(ax, [('s_CL', r'$s_{\rm CL}$', C1, 'CL'), ('s_atan', r'$s_{\rm atan}$', C2, 'atan')]):
    a.scatter(L(S.s_data), L(S[col]), s=10, color=c, lw=0); lim = [L(S.s_data).min()-.2, L(S.s_data).max()+.2]
    a.plot(lim, lim, color='k', lw=.8); a.set_xlabel(r'$\log\,s_{\rm data}$'); a.set_title(lab, fontsize=9.5)
    g = R['grad_agreement']; a.text(.04, .9, rf"$\rho={g[k+'_vs_data_rho']:.2f}$, median ratio {g[k+'_vs_data_median_ratio']:.2f}", transform=a.transAxes, fontsize=8.5)
ax[0].set_ylabel(r'$\log\,s_{\rm model}$ [km s$^{-1}$ kpc$^{-1}$]')
fig.tight_layout(); fig.savefig(F/'fig3_gradient_agreement.pdf'); plt.close(fig)

# ---- Fig 4: BTFR with three velocities
f = S[S.Vflat > 0]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for a, (col, lab, key) in zip(ax, [('Vflat', r'$V_{\rm flat}$ (SPARC)', 'Vflat'), ('vL', r'$v_L$ (CL)', 'vL'), ('Vc', r'$V_c$ (arctan)', 'Vc_atan')]):
    x, y = L(f[col]), L(f.Mbar); b, c0 = np.polyfit(x, y, 1)
    a.scatter(x, y, s=9, color=C1, lw=0); xx = np.linspace(x.min(), x.max(), 5); a.plot(xx, c0 + b*xx, color='k', lw=1.2)
    a.set_xlabel(r'$\log\,V$ [km s$^{-1}$]'); a.set_title(lab, fontsize=9.5)
    st = R['BTFR'][key]; a.text(.05, .9, f"slope {st['slope']:.2f}, scatter {st['scatter']:.3f} dex", transform=a.transAxes, fontsize=8)
ax[0].set_ylabel(r'$\log\,M_{\rm bar}$ [M$_\odot$]')
fig.tight_layout(); fig.savefig(F/'fig4_btfr.pdf'); plt.close(fig)

# ---- Fig 5: R as a structural scale
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8))
k = S.r13.notna(); a = ax[0]
a.scatter(S.logR[k], L(S.r13[k]), s=10, color=C1, lw=0); lim = [S.logR[k].min()-.1, S.logR[k].max()+.1]; a.plot(lim, lim, 'k', lw=.8)
a.set_xlabel(r'$\log R$ (fit) [kpc]'); a.set_ylabel(r'$\log r_{1/3}$ (data) [kpc]')
a.text(.04, .9, rf"$\rho={R['r13']['rho']:.2f}$, median ratio {R['r13']['median_ratio']:.2f}", transform=a.transAxes, fontsize=8.5)
a = ax[1]; groups = [(0, 2, 'S0–Sab'), (3, 4, 'Sb–Sbc'), (5, 6, 'Sc–Scd'), (7, 9, 'Sd–Sm'), (10, 11, 'Im–BCD')]
data = [L(10**S.logR/S.Rd)[(S['T'] >= lo) & (S['T'] <= hi)] for lo, hi, _ in groups]
bp = a.boxplot(data, widths=.5, patch_artist=True, medianprops=dict(color='k'), flierprops=dict(ms=3))
for p in bp['boxes']: p.set_facecolor('#cfe0f6'); p.set_edgecolor(C1)
a.set_xticks(range(1, 6)); a.set_xticklabels([f"{g[2]}\n({len(x)})" for g, x in zip(groups, data)], fontsize=7.5); a.axhline(0, color=GR, lw=.6, ls=':')
a.set_ylabel(r'$\log(R/R_d)$'); a.text(.04, .9, rf"$\rho(T,R/R_d)={R['R_Rd']['rho_ratio_T']:.2f}$", transform=a.transAxes, fontsize=8.5)
fig.tight_layout(); fig.savefig(F/'fig5_R_structure.pdf'); plt.close(fig)

# ---- Fig 6: photometric prediction
P = pd.read_csv(D/'prediction_per_galaxy.csv'); pj = json.load(open(D/'prediction.json'))
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
bins = np.linspace(0, .45, 19)
ax[0].hist(P.rms_CL, bins, color=C1, alpha=.75, label='CL'); ax[0].hist(P.rms_atan, bins, histtype='step', color=C2, lw=1.4, label='arctan')
ax[0].set_xlabel('rms log($V_{\\rm pred}/V_{\\rm obs}$) per galaxy [dex]'); ax[0].set_ylabel('galaxies'); ax[0].legend(fontsize=8)
m = S.set_index('Galaxy').loc[P.Galaxy]
ax[1].scatter(m.logR.values, m.logR.values + P.dlogR.values, s=10, color=C1, lw=0); lim = [m.logR.min()-.1, m.logR.max()+.1]; ax[1].plot(lim, lim, 'k', lw=.8)
ax[1].set_xlabel(r'$\log R$ (fit)'); ax[1].set_ylabel(r'$\log R$ (from photometry)')
ax[1].text(.04, .9, rf"scatter {pj['scatter_dlogR']:.2f} dex", transform=ax[1].transAxes, fontsize=8.5)
fig.tight_layout(); fig.savefig(F/'fig6_prediction.pdf'); plt.close(fig)
print('figures written')
