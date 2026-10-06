import pandas as pd,numpy as np
from scipy import stats
df=pd.read_csv('catC_results.csv'); d=df[df.outcome=='descent window']; k=df[df.outcome=='single-L kept']; acc=df[df.outcome!='single-L kept']
TY={0:'S0',1:'Sa',2:'Sab',3:'Sb',4:'Sbc',5:'Sc',6:'Scd',7:'Sd',8:'Sdm',9:'Sm',10:'Im',11:'BCD'}
gn=lambda g:g.replace('-','--'); lst=lambda s:', '.join(gn(x) for x in s); med=lambda s,c: np.median(s[c])
def asym(x,lo,hi,dd=3): return f"${x:.{dd}g}^{{+{max(hi-x,0):.2g}}}_{{-{max(x-lo,0):.2g}}}$"
e=lambda x: r'\mathrm{n/a}' if pd.isna(x) else f"{x:.2g}"
gI=d[d.grade.str.match(r'^IM?$')]; gII=d[d.grade.str.match(r'^IIM?$')]; gIII=d[d.grade.str.startswith('III')]
L=[]; A=L.append
A(r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs,longtable,amsmath,amssymb,graphicx,pdfpages,caption,lscape}
\captionsetup{font=small,labelfont=bf}
\title{\bfseries CL inflow fits with a virial window for the SPARC category-C galaxies:\\ detection, model selection, fit quality and robustness}
\author{Automated analysis for E.P.J.\ de Haas}\date{}
\begin{document}\maketitle
\section*{Summary}""")
A(f"""Category C contains the {len(df)} SPARC galaxies whose single-Lagrangian (single-L) residuals show a significant $-+-$ wave: the data rise above the CL curve at intermediate radii and fall below it further out. For all of them we fitted the CL inflow model extended with a virial window, as introduced in de Haas (2026) [1] and in the CL field paper [2]. Beyond a radius $r_v$ the orbital speed gains a Newtonian component of strength $p$. The window is accepted in {len(acc)} galaxies ($\\Delta\\mathrm{{BIC}}<-6$ relative to single-L); in {len(k)} the wave is too weak and single-L is kept. In {len(d)} of the accepted galaxies the window is a \\emph{{descent}} ($p>2$): beyond $r_v$, $v^2$ falls as $GM_N/r$ onto a floor $A$. For these, median $\\chi^2_\\nu$ falls from {med(d,'chi2nu_S'):.2f} to {med(d,'chi2nu_VW'):.2f}, RMS$_\\mathrm{{rel}}$ from {med(d,'RMS_S'):.3f} to {med(d,'RMS_VW'):.3f}, and the window radius, floor and descent mass are determined to {100*med(d,'relrv'):.0f}\\%, {100*med(d,'relA'):.0f}\\% and {100*med(d,'relM_N') if 'relM_N' in d else 100*med(d,'relMN'):.0f}\\% (medians). The window is preferred over the two-Lagrangian alternative in every accepted galaxy, and a free gauge offset is never required. The descent galaxies are early and intermediate spirals (Sab--Sc), complementary to the late-type nested-spiral galaxies of category B. Residual structure persists in {int((d.pruns_VW<0.05).sum())} of {len(d)}, and {int(d.flagM.sum())} remain at $\\chi^2_\\nu>2$; these are bulge-dominated curves with very small errors, where the inner rise and peak need further regions.""")
A(r"""
\section{Models}
\paragraph{Single-L.} The CL inflow curve of [2] (Eqs.~17--18; see also [1, Eqs.~4--5] and [3]), with $X=\sqrt{2GM/R}-H_zR$ and $Y(r)=\sqrt{2GM/r}-H_zr$:
\begin{equation*}v^2=\tfrac12X^2r^2/R^2\ (r\le R),\qquad v^2=\tfrac32X^2-Y(r)^2\ (r>R),\qquad H_z=2.2\times10^{-18}\,\mathrm{s^{-1}}.\end{equation*}
\paragraph{CL + virial window (VW).} Following [1, Eq.~6] and [2, Eq.~19], beyond $r_v$:
\begin{equation*}v^2(r)=\tfrac32X^2-Y(r)^2+p\Bigl[\tfrac12Y(r)^2-\Phi_p\Bigr],\qquad r\ge r_v.\end{equation*}
Two choices differ from [1], where $r_v$ is set by inspection and $\Phi_p$ is free. Here $r_v$ is fitted, and $\Phi_p=\tfrac12Y(r_v)^2$ makes $v^2$ continuous at $r_v$; leaving $\Phi_p$ free (model VWf, $k=5$) is tested separately. For $H_z\to0$ the window reads
\begin{equation*}v^2=A+\frac{GM_N}{r},\qquad A=\frac{3GM}{R}-\frac{pGM}{r_v},\qquad M_N=(p-2)\,M,\end{equation*}
so $p>2$ gives a Keplerian descent of mass $M_N$ onto the floor $A$, $0<p<2$ a flattening, $p<0$ an additional rise, and $p=3$ the pure Kepler term of the inner mass $M$. Because $p$ and $M$ become degenerate when the descent dominates, the fit is parametrised by $(M,R,r_v,M_N)$, which are also the measured quantities. This descent behaviour was first noticed in earlier spreadsheet fits as solutions with formally negative $(M,R)$, which have exactly the $A+B/r$ form; the window replaces them with positive parameters valid for $H_z\neq0$.
\paragraph{Alternatives.} Two-L ([2, Eqs.~24--26]; $k=4$, the same parameter count as VW), and VWf ($k=5$).
\paragraph{Selection and uncertainties.} The window is accepted when $\Delta\mathrm{BIC}=\mathrm{BIC_{VW}}-\mathrm{BIC_S}<-6$, and assigned to two-L instead if two-L has lower BIC by more than 2. Uncertainties come from five sources. (1) Statistical: local covariance; $A$ by linear propagation. (2) $r_v$: since $\partial v^2/\partial r$ is discontinuous at $r_v$, $\chi^2(r_v)$ changes slope at each data radius, so a profile-likelihood interval ($\Delta\chi^2\le1$, 80 grid values, other parameters refitted) is used. (3) Jackknife: leave-one-out. (4) Systematic: distance $+\delta D$ and inclination $+\delta i$ refits. (5) $H_z=0$ refit. The profile scan also served as a global check of the optimum; it found a better solution for NGC 6674, which was adopted.
\paragraph{Grades (descent windows).} \textbf{I}: $R$, $r_v$, $M_N$ and $A$ all within 30\%, at least 3 points on each side of $r_v$, jackknife scatter $\le30\%$. \textbf{II}: $A$ within 10\% and $M_N$ within 50\%. \textbf{III}: otherwise. \textbf{M} flag: $\chi^2_\nu>2$ after the window.
""")
A(r"\section{Model selection outcome}")
A(r"\begin{table}[h]\centering\small\caption{Category-C outcome.}\begin{tabular}{lrl}\toprule Outcome & $N$ & Galaxies\\\midrule")
for nm,s in [('Descent window, grade I',gI),('Descent window, grade II',gII),('Descent window, grade III',gIII),('Flattening window ($0<p<2$)',df[df.outcome=='flattening window']),('Rising window ($p<0$)',df[df.outcome=='rising window']),('Single-L kept',k)]:
    A(f"{nm} & {len(s)} & \\parbox[t]{{10.5cm}}{{\\raggedright {lst(s.Galaxy)}}}\\\\")
A(r"\bottomrule\end{tabular}\\[2pt]{\footnotesize Galaxies with $\chi^2_\nu>2$ after the window (flag M): "+lst(df[df.flagM].Galaxy)+".}\end{table}")
A(f"""\\textbf{{Window versus two-L.}} With the same number of parameters, the window beats the two-Lagrangian model in every accepted galaxy (smallest $\\mathrm{{BIC_{{2L}}-BIC_{{VW}}}}$ = {acc.dBIC_2L_VW.min():+.1f}, median {acc.dBIC_2L_VW.median():+.0f}). A two-L reset produces an upward step and cannot follow a $1/r$ descent, so category C is genuinely different from category B. \\textbf{{Free gauge.}} Leaving $\\Phi_p$ free never helps ($\\Delta\\mathrm{{BIC_{{VWf-VW}}}}\\ge{df.dBIC_VWf.min():.1f}$), so the data show no step at $r_v$ and the continuous gauge is adequate. \\textbf{{Single-L kept.}} The {len(k)} rejected galaxies already have small single-L $\\chi^2_\\nu$ (median {med(k,'chi2nu_S'):.2f}) and are mostly late types; their $-+-$ wave lies within the SPARC errors. \\textbf{{Other windows.}} ESO563-G021 and IC 4202 need an outer \\emph{{flattening}} ($p\\approx1.7$), and NGC 6946 needs an extra \\emph{{rise}} ($p<0$, with two-L only $+2.2$ behind). That makes NGC 6946 a borderline case between categories B and C.""")
A(r"\begin{figure}[h]\centering\includegraphics[width=\textwidth]{catC_fig_quality.pdf}\caption{Left: Durbin--Watson statistic before and after the window. Middle: relative RMS in $v^2$. Right: $\Delta$BIC (VW $-$ single-L) versus $n$; acceptance threshold $-6$. Colours: descent (purple), flattening (cyan), rising (olive), single-L kept (grey).}\end{figure}")
A(r"\section{Fit quality of the descent windows}")
A(r"\begin{table}[h]\centering\small\caption{Descent-window galaxies: fit-quality statistics (medians, single-L $\to$ CL + virial window).}\begin{tabular}{lrrr}\toprule & Grade I & Grade II & Grade III\\\midrule")
for lab,c1,c2,dd in [('$\\chi^2_\\nu$','chi2nu_S','chi2nu_VW',2),('RMS$_\\mathrm{rel}$','RMS_S','RMS_VW',3),('Durbin--Watson','DW_S','DW_VW',2)]:
    A(lab+' & '+' & '.join(f"{med(s,c1):.{dd}f} $\\to$ {med(s,c2):.{dd}f}" for s in (gI,gII,gIII))+'\\\\')
A('$p_\\mathrm{runs}<0.05$ after window & '+' & '.join(f"{int((s.pruns_VW<0.05).sum())}/{len(s)}" for s in (gI,gII,gIII))+'\\\\')
A('$\\Delta$BIC (median) & '+' & '.join(f"{med(s,'dBIC_VW'):.0f}" for s in (gI,gII,gIII))+'\\\\')
A('$n$ (median) & '+' & '.join(f"{int(med(s,'n'))}" for s in (gI,gII,gIII))+'\\\\')
A(r"\bottomrule\end{tabular}\end{table}")
A(f"""The window removes the dominant $-+-$ pattern in all descent galaxies. The median Durbin--Watson statistic rises from {med(d,'DW_S'):.2f} to {med(d,'DW_VW'):.2f}, and the residual sign pattern loses its long runs (atlas). Residual structure is not fully removed in {int((d.pruns_VW<0.05).sum())} galaxies. Unlike categories A and B, category C contains many high-precision, bulge-dominated curves (median $n={int(med(d,'n'))}$, {int((d.Q==1).sum())} of {len(d)} Q=1). There small, coherent deviations near the inner peak become significant: {int((d.maxz_VW>3).sum())} galaxies have at least one residual beyond $3\\sigma$, and Shapiro--Wilk rejects normality in {int((d.pSW_VW<0.05).sum())}. The SPARC errors are again conservative on average ({int((d.Plow_VW<0.01).sum())} galaxies with $P(\\chi^2\\le\\chi^2_\\mathrm{{obs}})<0.01$), so $\\chi^2_\\nu<1$ is not unusual. The flag-M galaxies ({lst(df[df.flagM].Galaxy)}) are improved by factors of 2 to 30 in $\\chi^2$, but their inner bulge region and peak are not described by a single CL bulge. They are the natural targets for a bulge Lagrangian plus disk Lagrangian with a window, or for the two-window scheme of [1, Eqs.~5--7].""")
A(r"\section{Parameter determination}")
A(f"""Median relative uncertainties for the descent windows are: $R$ {100*med(d,'relR'):.0f}\\%, $r_v$ {100*med(d,'relrv'):.0f}\\% (profile), $M_N$ {100*med(d,'relMN'):.0f}\\%, $A$ {100*med(d,'relA'):.1f}\\%, and $M$ {100*med(d,'relM'):.0f}\\%. The floor $A$ and the window radius are the best-determined quantities. The parametrisation by $M_N$ removes the $p$--$M$ degeneracy (median $\\rho(M,M_N)={med(d,'rho_M_MN'):.2f}$). The virial strength $p=2+M_N/M$ is then a derived quantity, and it becomes very large where $M$ is small: in NGC 891, NGC 5033, NGC 6674, UGC 2916 and UGC 5253 the inner CL bulge shrinks and the curve is a compact central mass plus Keplerian descent. $p$ ranges over {np.percentile(d.p,16):.1f}--{np.percentile(d.p,84):.0f} (16--84\\%, median {d.p.median():.1f}), and {int((d.p<=5).sum())} galaxies have $p\\le5$. The pure Kepler case $p=3$ is therefore one end of a broad distribution, not a universal value.

\\paragraph{{The $r_v$ likelihood.}} Fig.~2 shows that $\\chi^2(r_v)$ has a well-defined minimum in most galaxies, kinked at the data radii. Because $v^2$ is continuous at $r_v$, the profile is smoother than the $R_2$ profile of the two-L model, and $r_v$ is localised to within one or two point spacings.

\\paragraph{{Systematics, jackknife, $H_z$.}} Distance and inclination systematics are comparable to or larger than the statistical errors (median sys/stat {np.median(d.srv/d.erv_p):.1f} for $r_v$, {np.median(d.sA/d.eA):.1f} for $A$, {np.median(d.sMN/d.eMN):.1f} for $M_N$). Leave-one-out scatter stays below 30\\% for grade I. Setting $H_z=0$ changes the parameters by at most about 12\\% except in NGC 5033 and UGC 2916, where the inner $R$ is poorly constrained and the solution moves along the $M$--$R$ degeneracy.""")
A(r"\begin{figure}[h]\centering\includegraphics[width=\textwidth]{catC_fig_profiles.pdf}\caption{Profile $\Delta\chi^2(r_v)$ for the descent-window galaxies; grey lines are the data radii.}\end{figure}")
A(r"\section{Physical consistency}")
ok=d.Vflat>0; lb=d.Lb>0
A(f"""\\begin{{itemize}}\\itemsep0pt
\\item \\textbf{{Window radius.}} $r_v$ = {np.percentile(d.rv,16):.1f}--{np.percentile(d.rv,84):.1f}\\,kpc (median {d.rv.median():.1f}), $r_v/R_\\mathrm{{disk}}$ = {np.median(d.rv/d.Rdisk):.1f} (16--84\\%: {np.percentile(d.rv/d.Rdisk,16):.1f}--{np.percentile(d.rv/d.Rdisk,84):.1f}; Spearman $\\rho$ = {stats.spearmanr(d.rv,d.Rdisk)[0]:.2f}). The descent sets in at 2--4 disk scale lengths, beyond the rotation-curve peak.
\\item \\textbf{{Floor.}} $\\sqrt{{A}}$ tracks SPARC $V_\\mathrm{{flat}}$ ($r$ = {stats.pearsonr(np.sqrt(d.A[ok]),d.Vflat[ok])[0]:.2f}), with $\\sqrt{{A}}/V_\\mathrm{{flat}}$ = {np.median(np.sqrt(d.A[ok])/d.Vflat[ok]):.2f}. The floor is the asymptotic speed that the descent approaches; $V_\\mathrm{{flat}}$, measured over the last points, lies slightly above it because the descent is not yet complete.
\\item \\textbf{{Descent mass.}} $M_N/M_\\mathrm{{bar}}$ = {np.median(d.MN/d.Mbar):.2f} (16--84\\%: {np.percentile(d.MN/d.Mbar,16):.2f}--{np.percentile(d.MN/d.Mbar,84):.2f}; $r$ = {stats.pearsonr(np.log10(d.MN),np.log10(d.Mbar))[0]:.2f} in log). For the {int(lb.sum())} galaxies with a photometric bulge, $M_N$ is {np.median((d.MN/(0.7*d.Lb))[lb]):.1f} times the bulge stellar mass ($\\Upsilon_\\mathrm{{bulge}}=0.7$; 16--84\\%: {np.percentile((d.MN/(0.7*d.Lb))[lb],16):.1f}--{np.percentile((d.MN/(0.7*d.Lb))[lb],84):.1f}). The Keplerian descent mass is therefore of the order of the baryonic mass inside the peak, consistent with the reading in [1] that the window matter orbits Newtonially with respect to the inflowing metric.
\\item \\textbf{{Morphology.}} The descent windows are early and intermediate spirals ({', '.join(f"{TY[kk]} {v}" for kk,v in d['T'].value_counts().sort_index().items())}), with {int((d.Q==1).sum())} of {len(d)} Q=1. The single-L-kept galaxies are mostly Scd--BCD. Together with category B (late-type nested spirals), this gives a clean morphological split. Bulge-dominated galaxies show a Keplerian descent window; bulgeless late types show a Lagrangian reset.
\\end{{itemize}}""")
A(r"\begin{figure}[h]\centering\includegraphics[width=\textwidth]{catC_fig_physics.pdf}\caption{Descent windows. Left to right: $r_v$ versus $R_\mathrm{disk}$; floor $\sqrt A$ versus SPARC $V_\mathrm{flat}$; descent mass $M_N$ versus $M_\mathrm{bar}$; distribution of the virial strength $p$ (dashed: pure Kepler $p=3$). Colours: grade I green, II orange, III red.}\end{figure}")
A(r"\section{Assessment}")
A(f"""\\textbf{{The descent is a robust, recurring feature.}} The CL + virial window of [1,2], with $r_v$ fitted and a continuous gauge, describes {len(d)} of the {len(df)} category-C galaxies far better than single-L ($\\Delta$BIC from {d.dBIC_VW.max():.0f} to {d.dBIC_VW.min():.0f}). It is always preferred over a two-Lagrangian reset and needs no free gauge. Its measured quantities, the window radius $r_v$, the floor $A$ and the Keplerian descent mass $M_N$, are well determined and match independent photometric and kinematic scales.

\\textbf{{Grades.}} Grade I ({lst(gI.Galaxy)}) is the robust core. Grade II ({lst(gII.Galaxy)}) has a well-determined floor and window but a less certain descent mass. Grade III ({lst(gIII.Galaxy)}) has a weakly determined descent mass ($>50$\\%) or floor ($>10$\\%), mostly because only a few points lie in the window, the descent is shallow ($p$ close to 2), or the inner radius is unconstrained.

\\textbf{{Limits.}} (i) The inner bulge region of the most massive, best-measured spirals is not captured by a single CL bulge, which leaves residual structure in about two thirds of the descent galaxies and $\\chi^2_\\nu>2$ in eight. (ii) $p$ is not a robust quantity when the inner mass is small; $M_N$ and $A$ should be reported instead. (iii) The window has no outer end here. The two-window scheme of [1] (window strengths $p$, $q$) is the natural next refinement for the flag-M galaxies, together with an explicit bulge component.

\\section*{{References}}
\\small
[1] E.P.J.\\ de Haas (2026), From Rotation Curves to Cosmic Time: Probing High-Redshift Expansion through Nested Spiral Dynamics, \\emph{{J.\\ High Energy Phys.\\ Grav.\\ Cosmol.}} 12, 334--367, doi:10.4236/jhepgc.2026.121022 (virial term Eq.~3; multi-region virial model Eqs.~5--7).\\par
[2] E.P.J.\\ de Haas, Galactic Rotation Curves and the Constant--Lagrangian Field: Empirical Tests within the $Q_g$ Rotor Framework (manuscript; single-L Eqs.~17--18, virial term Eq.~19, two-L Eqs.~24--26).\\par
[3] E.P.J.\\ de Haas, The Role of the Hubble Parameter in Galactic Rotation Curves and Spiral Morphology (manuscript).\\par
[4] F.\\ Lelli, S.S.\\ McGaugh, J.M.\\ Schombert (2016), SPARC, \\emph{{AJ}} 152, 157.
""")
A(r"\clearpage\begin{landscape}{\scriptsize\setlength{\tabcolsep}{3pt}")
A(r"""\begin{longtable}{lllrrrrrrrrrll}
\caption{All category-C galaxies: single-L, CL + virial window (VW), two-L and VWf comparison.}\\\toprule
Galaxy & Type & Q & $n$ & $\chi^2_{\nu,S}$ & RMS$_S$ & $p_\mathrm{runs,S}$ & $\chi^2_{\nu,VW}$ & $\Delta$BIC$_{VW}$ & BIC$_{2L}-$BIC$_{VW}$ & BIC$_{VWf}-$BIC$_{VW}$ & $p$ & Outcome & Grade\\\midrule\endfirsthead
\toprule Galaxy & Type & Q & $n$ & $\chi^2_{\nu,S}$ & RMS$_S$ & $p_\mathrm{runs,S}$ & $\chi^2_{\nu,VW}$ & $\Delta$BIC$_{VW}$ & BIC$_{2L}-$BIC$_{VW}$ & BIC$_{VWf}-$BIC$_{VW}$ & $p$ & Outcome & Grade\\\midrule\endhead\bottomrule\endfoot""")
for _,r in df.sort_values('dBIC_VW').iterrows():
    A(f"{gn(r.Galaxy)} & {TY[int(r['T'])]} & {int(r.Q)} & {int(r.n)} & {r.chi2nu_S:.2f} & {r.RMS_S:.3f} & {r.p_runs1:.1e} & {r.chi2nu_VW:.2f} & {r.dBIC_VW:.1f} & {r.dBIC_2L_VW:+.1f} & {r.dBIC_VWf:+.1f} & {r.p:.3g} & {r.outcome} & {r.grade}\\\\")
A(r"\end{longtable}")
A(r"""\begin{longtable}{llrlllllllr}
\caption{Accepted windows: parameters. $R,r_v$ in kpc; $M,M_N$ in $10^9M_\odot$; $A$ in km$^2$s$^{-2}$. $R$, $M_N$, $A$: statistical $\pm$ systematic; $r_v$: profile $\Delta\chi^2\le1$ interval [systematic]. $n_\mathrm{in}/n_\mathrm{win}$: points before/after $r_v$.}\\\toprule
Galaxy & Grade & $n$ & $R$ & $M$ & $r_v$ & $M_N$ & $A$ & $p$ & $n_\mathrm{in}/n_\mathrm{win}$ & jk\\\midrule\endfirsthead
\toprule Galaxy & Grade & $n$ & $R$ & $M$ & $r_v$ & $M_N$ & $A$ & $p$ & $n_\mathrm{in}/n_\mathrm{win}$ & jk\\\midrule\endhead\bottomrule\endfoot""")
o=acc.assign(o=acc.outcome.map({'descent window':0,'flattening window':1,'rising window':2})).sort_values(['o','grade','dBIC_VW'])
for _,r in o.iterrows():
    A(f"{gn(r.Galaxy)} & {r.grade if r.grade!='--' else r.outcome.split()[0]} & {int(r.n)} & ${r.R:.3g}\\pm{e(r.eR)}\\pm{e(r.sR)}$ & ${r.M:.3g}\\pm{e(r.eM)}$ & {asym(r.rv,r.rv_lo,r.rv_hi)} [{e(r.srv)}] & ${r.MN:.3g}\\pm{e(r.eMN)}\\pm{e(r.sMN)}$ & ${r.A:.0f}\\pm{e(r.eA)}\\pm{e(r.sA)}$ & {r.p:.3g} & {int(r.n_in)}/{int(r.n_win)} & {r.jk_rel:.2f}\\\\")
A(r"\end{longtable}")
A(r"""\begin{longtable}{lrrrrrrrrrrr}
\caption{Accepted windows: goodness of fit. $P_<=P(\chi^2\le\chi^2_\mathrm{obs})$ for $n-4$ dof; $p_\mathrm{SW}$ Shapiro--Wilk; max$|z|$ largest weighted residual.}\\\toprule
Galaxy & $\chi^2_{\nu,S}$ & $\chi^2_{\nu,VW}$ & BIC$_S$ & BIC$_{VW}$ & RMS$_S$ & RMS$_{VW}$ & $p_\mathrm{runs,VW}$ & DW$_S\to$DW$_{VW}$ & $p_\mathrm{SW}$ & $P_<$ & max$|z|$\\\midrule\endfirsthead
\toprule Galaxy & $\chi^2_{\nu,S}$ & $\chi^2_{\nu,VW}$ & BIC$_S$ & BIC$_{VW}$ & RMS$_S$ & RMS$_{VW}$ & $p_\mathrm{runs,VW}$ & DW$_S\to$DW$_{VW}$ & $p_\mathrm{SW}$ & $P_<$ & max$|z|$\\\midrule\endhead\bottomrule\endfoot""")
for _,r in o.iterrows():
    A(f"{gn(r.Galaxy)} & {r.chi2nu_S:.2f} & {r.chi2nu_VW:.2f} & {r.BIC_S:.1f} & {r.BIC_VW:.1f} & {r.RMS_S:.3f} & {r.RMS_VW:.3f} & {r.pruns_VW:.3f} & {r.DW_S:.2f}$\\to${r.DW_VW:.2f} & {r.pSW_VW:.3f} & {r.Plow_VW:.3f} & {r.maxz_VW:.2f}\\\\")
A(r"\end{longtable}}\end{landscape}")
A(r"\section*{Atlas: category-C fits with residuals}\includepdf[pages=-,landscape=true,scale=0.95]{catC_atlas.pdf}\end{document}")
open('CL_categoryC_report.tex','w').write('\n'.join(L))
