import pandas as pd,numpy as np
df=pd.read_csv('catE_results.csv')
TY={0:'S0',1:'Sa',2:'Sab',3:'Sb',4:'Sbc',5:'Sc',6:'Scd',7:'Sd',8:'Sdm',9:'Sm',10:'Im',11:'BCD'}
gn=lambda g:g.replace('-','--'); med=lambda s,c: np.median(s[c])
kinds=['rising window (B-like)','descent window (C-like)','two windows (D-like)','single-L kept']
K=lambda k: df[df.kind==k]; acc=df[df.best!='S']; sk=K('single-L kept')
NM={'S':'single-L','VW':'one window','2L':'two-L','VW2':'two windows'}
L=[]; A=L.append
A(r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs,longtable,amsmath,amssymb,graphicx,pdfpages,caption,lscape}
\captionsetup{font=small,labelfont=bf}
\title{\bfseries CL inflow fits of the SPARC category-E galaxies:\\ poor single-L fits without detectable structure}
\author{Automated analysis for E.P.J.\ de Haas}\date{}
\begin{document}\maketitle
\section*{Summary}""")
A(f"""Category E contains the {len(df)} SPARC galaxies whose single-Lagrangian (single-L) fit has $\\chi^2_\\nu>1$ but whose residual signs pass the runs test ($p\\ge0.05$): the fit is poor, yet no coherent pattern is detected. We show that this combination arises mainly because the curves are short (median $n={int(med(df,'n'))}$; {int((df.n<=13).sum())} of {len(df)} have $n\\le13$). For $n=4$--5 the runs test cannot reach $p<0.05$ for \\emph{{any}} sign sequence, and for $n\\le13$ it detects only very clean patterns. Testing the model extensions directly by BIC, as for categories C and D ([1,2]), gives {len(acc)} galaxies in which an extension is strongly preferred ($\\Delta\\mathrm{{BIC}}<-6$). Of these, {len(K(kinds[0]))} show a \\emph{{rising window}}, closely related to a Lagrangian reset (B-like; statistically equivalent in three), all in Im/Sm galaxies. {len(K(kinds[1]))} show a \\emph{{descent window}} (C-like), in Sb--Sc spirals. {len(K(kinds[2]))} need two windows (D-like). The remaining {len(sk)} galaxies are adequately described by single-L with a modest intrinsic velocity scatter (median {med(sk,'sigma_int'):.1f}\\,km\\,s$^{{-1}}$), their excess $\\chi^2$ dominated by a single point. Category E is therefore not a physically distinct class. It consists of B-, C- and D-type galaxies that the sign test could not recognise with few points, plus single-L galaxies with slightly underestimated errors. The morphological split of B (late types, reset) versus C (bulged spirals, descent) reappears here.""")
A(r"""
\section{Models and tests}
The same model set as for category D was used, with $X=\sqrt{2GM/R}-H_zR$, $Y(r)=\sqrt{2GM/r}-H_zr$, $H_z=2.2\times10^{-18}$\,s$^{-1}$, fits on $v^2$ with $\sigma_{v^2}=2V_\mathrm{obs}\delta V_\mathrm{obs}$:
\textbf{S} single-L [2, Eqs.~17--18] ($k=2$); \textbf{VW} one virial window [1, Eq.~6; 2, Eq.~19] with continuous gauge and descent-mass parametrisation $M_N=(p-2)M$ ($k=4$; $n\ge6$); \textbf{2L} two Lagrangians [2, Eqs.~24--26] ($k=4$; $n\ge6$); \textbf{VW2} two windows [1, Eqs.~5--7] ($k=6$; $n\ge10$). The preferred model has the lowest BIC (a simpler model within $\Delta\mathrm{BIC}<2$ is preferred), and an extension must beat S by $\Delta\mathrm{BIC}<-6$. Status as in the D report: \textbf{clean} (all parameters within 60\%, jackknife scatter within 60\%, at least 3 points on each side of the first window radius, no bound or $|p|>100$, $\chi^2_\nu\le2$), \textbf{weakly constrained} (one of these fails), \textbf{phenomenological} (bulge radius at bound or $|p|>100$).

\paragraph{Power of the runs test.} For $n$ residuals the smallest attainable one-sided runs-test $p$ is reached by two equal runs. It is 0.110 ($n=4$), 0.063 ($n=5$), 0.020 ($n=7$), 0.0036 ($n=10$) and $7\times10^{-4}$ ($n=13$). With 4--5 points no pattern can ever be significant. With 7--10 points only the cleanest two- or three-run sequences reach $p<0.05$, and a single misplaced sign is enough to fail.
\paragraph{Scatter and outlier tests on single-L.} (a) Intrinsic velocity scatter $\sigma_\mathrm{int}$, added in quadrature to $\delta V_\mathrm{obs}$ and refitted, such that $\chi^2_\nu=1$. (b) Influence of the worst point: $\chi^2_\nu$ after removing it and refitting. (c) Robust (soft-$\ell_1$) refit. (d) Upper-tail probability $P(\chi^2\ge\chi^2_\mathrm{obs})$. Distance and inclination systematics as in the A--D reports.
""")
A(r"\section{Outcome}")
A(r"\begin{table}[h]\centering\small\caption{Category-E outcome by preferred description.}\begin{tabular}{llp{10cm}}\toprule Preferred description & $N$ & Galaxies (type, $n$, status)\\\midrule")
for k in kinds:
    s=K(k); A(f"{k} & {len(s)} & "+', '.join(f"{gn(r.Galaxy)} ({TY[int(r['T'])]}, {int(r.n)}, {r.status})" for _,r in s.sort_values('dBIC_best').iterrows())+"\\\\")
A(r"\bottomrule\end{tabular}\end{table}")
A(r"\begin{figure}[h]\centering\includegraphics[width=\textwidth]{catE_fig_models.pdf}\caption{Left: $\Delta$BIC of each extension relative to single-L (red: $-6$); label colour = preferred description (green rising window, purple descent, blue two windows, grey single-L). Middle: runs-test $p$ of the single-L residuals versus $n$, with the best attainable $p$ (black) and the 0.05 level; for small $n$ no or only very clean patterns can be detected. Right: $\chi^2_\nu$ single-L versus preferred model.}\end{figure}")
r_=K(kinds[0]); d_=K(kinds[1]); w_=K(kinds[2])
A(f"""\\paragraph{{Rising windows, B-like ({len(r_)}).}} {', '.join(gn(g) for g in r_.Galaxy)}, all Im/Sm with 7--12 points. Beyond $r_v\\approx1.1$--6.3\\,kpc the curve rises above the single-L continuation ($p<0$). In ESO444-G084, UGC 191 and UGC 11820 two-L is statistically equivalent ($|\\mathrm{{BIC_{{2L}}-BIC_{{VW}}}}|<0.3$); DDO 168 and UGC 5716 prefer the window by 2.2 and 5.8. They are the short-curve counterparts of the category-B nested-spiral galaxies. UGC 5716 and UGC 11820 are clean (single-L $\\chi^2_\\nu$ 9.3 and 6.7 $\\to$ 1.85 and 0.14). The other three have too few points before $r_v$ for stable parameters.

\\paragraph{{Descent windows, C-like ({len(d_)}).}} NGC 801 (Sc), NGC 2683 (Sb) and NGC 2998 (Sc): $\\chi^2_\\nu$ 10.8, 10.8 and 2.0 $\\to$ 1.5, 1.1 and 0.9, with $\\Delta\\mathrm{{BIC}}=-100$, $-85$ and $-8$. NGC 801 is clean ($r_v=4.2$\\,kpc, $p=3.5$, close to the pure Kepler case). In NGC 2683 only two points precede the window, and in NGC 2998 the descent rests on the last two points, so both are weakly constrained. The two edge-on systems (NGC 801, NGC 2683; $i=80^\\circ$) have the largest single-L $\\chi^2_\\nu$ of the category.

\\paragraph{{Two windows, D-like ({len(w_)}).}} NGC 6195 (Sb) is well fitted by a near-Kepler descent ($p\\approx3$) followed by a flattening ($q\\approx0.6$), but has only two points inside the first window radius. UGC 5764 (Im) is improved but its parameters are unstable. NGC 2955 (Sb) is phenomenological ($|q|>100$, parameter uncertainties $\\gg100$\\%), like the bulge-dominated D galaxies.

\\paragraph{{Single-L kept ({len(sk)}).}} {', '.join(gn(g) for g in sk.Galaxy)}. No extension is justified. Their excess $\\chi^2$ is small (single-L $\\chi^2_\\nu$ {sk.chi2nu_S.min():.2f}--{sk.chi2nu_S.max():.2f}; upper-tail $P(\\chi^2\\ge\\chi^2_\\mathrm{{obs}})$ from {sk.P_high.min():.3f} to {sk.P_high.max():.2f}) and is dominated by one point: removing the worst point gives $\\chi^2_\\nu$ = {sk.chi2nu_drop1.min():.2f}--{sk.chi2nu_drop1.max():.2f}. An intrinsic scatter of {sk.sigma_int.min():.1f}--{sk.sigma_int.max():.1f}\\,km\\,s$^{{-1}}$ (median {med(sk,'sigma_int'):.1f}, i.e.\\ {100*med(sk,'sigma_int_rel'):.0f}\\% of the median speed) brings $\\chi^2_\\nu$ to 1. That level is typical of non-circular motions and is consistent with single-L describing these galaxies as well as it describes category A.""")
A(r"\section{Fit quality and robustness}")
A(r"\begin{table}[h]\centering\small\caption{Fit-quality statistics (medians, single-L $\to$ preferred).}\begin{tabular}{lrrrr}\toprule & Rising (5) & Descent (3) & Two windows (3) & Single-L kept (6)\\\midrule")
for lab,c1,c2,dd in [('$\\chi^2_\\nu$','chi2nu_S','chi2nu_best',2),('RMS$_\\mathrm{rel}$','RMS_S','RMS_best',3)]:
    A(lab+' & '+' & '.join(f"{med(s,c1):.{dd}f} $\\to$ {med(s,c2):.{dd}f}" for s in (r_,d_,w_,sk))+'\\\\')
A('$n$ & '+' & '.join(f"{int(med(s,'n'))}" for s in (r_,d_,w_,sk))+'\\\\')
A('$\\sigma_\\mathrm{int}$ (km/s, single-L) & '+' & '.join(f"{med(s,'sigma_int'):.1f}" for s in (r_,d_,w_,sk))+'\\\\')
A(r"\bottomrule\end{tabular}\end{table}")
A(f"""None of the preferred fits leaves residual sign structure ($p_\\mathrm{{runs}}\\ge0.05$ in all cases), but with this few points that criterion is weak, as shown above. The single-L parameters are robust against the treatment of the excess scatter: the robust (soft-$\\ell_1$) refit changes $M$ and $R$ by a median {100*np.median(np.abs(df.M_rob/df.M-1)):.0f}\\% and {100*np.median(np.abs(df.R_rob/df.R-1)):.0f}\\%, and inflating the errors by $\\sigma_\\mathrm{{int}}$ changes them by {100*np.median(np.abs(df.M_sint/df.M-1)):.0f}\\% and {100*np.median(np.abs(df.R_sint/df.R-1)):.0f}\\%. Hence the single-L scales of category E can be used, provided their errors are scaled by $\\sqrt{{\\chi^2_\\nu}}$ or $\\sigma_\\mathrm{{int}}$ is included. For the accepted extensions the parameter precision is limited by the number of points on either side of the window radius. Only NGC 801, UGC 5716 and UGC 11820 meet the clean criteria.""")
A(r"\section{Assessment}")
A(f"""\\begin{{enumerate}}\\itemsep1pt
\\item \\textbf{{Category E is a low-power class, not a physical one.}} Poor fits without detected structure occur because the runs test has little power for $n\\lesssim13$. When the extensions of [1,2] are tested directly by BIC, {len(acc)} of {len(df)} galaxies are described by the same features as categories B, C and D.
\\item \\textbf{{The B/C morphological split holds.}} The rising (reset-like) windows occur in Im/Sm galaxies, and the descents in Sb--Sc spirals, as in categories B and C.
\\item \\textbf{{The rest are single-L galaxies with a few km/s of extra scatter.}} {len(sk)} galaxies need no extension. Their excess $\\chi^2$ is carried by single points and corresponds to $\\sigma_\\mathrm{{int}}\\approx{med(sk,'sigma_int'):.0f}$\\,km\\,s$^{{-1}}$.
\\item \\textbf{{Pipeline consequence.}} For short curves the sign-based screen should be replaced or complemented by the direct BIC comparison of the extensions, with $n\\le13$ results flagged as tentative. Applied to all 175 galaxies, this would merge category E into A--D.
\\end{{enumerate}}

\\section*{{References}}\\small
[1] E.P.J.\\ de Haas (2026), From Rotation Curves to Cosmic Time: Probing High-Redshift Expansion through Nested Spiral Dynamics, \\emph{{J.\\ High Energy Phys.\\ Grav.\\ Cosmol.}} 12, 334--367, doi:10.4236/jhepgc.2026.121022 (virial term Eq.~3; multi-region virial model Eqs.~5--7).\\par
[2] E.P.J.\\ de Haas, Galactic Rotation Curves and the Constant--Lagrangian Field: Empirical Tests within the $Q_g$ Rotor Framework (manuscript; single-L Eqs.~17--18, virial term Eq.~19, two-L Eqs.~24--26).\\par
[3] F.\\ Lelli, S.S.\\ McGaugh, J.M.\\ Schombert (2016), SPARC, \\emph{{AJ}} 152, 157.
""")
A(r"\clearpage\begin{landscape}{\scriptsize\setlength{\tabcolsep}{3pt}")
A(r"""\begin{longtable}{lllrrlrrrrrrrrl}
\caption{All category-E galaxies: single-L fit, scatter diagnostics and model comparison ($\Delta$BIC relative to single-L; -- = not fitted).}\\\toprule
Galaxy & Type & Q & $i$ & $n$ & signs & $p_\mathrm{runs}$ & $p_\mathrm{min}$ & $\chi^2_{\nu,S}$ & $\sigma_\mathrm{int}$ & $\chi^2_{\nu,-1}$ & VW & 2L & VW2 & preferred / status\\\midrule\endfirsthead
\toprule Galaxy & Type & Q & $i$ & $n$ & signs & $p_\mathrm{runs}$ & $p_\mathrm{min}$ & $\chi^2_{\nu,S}$ & $\sigma_\mathrm{int}$ & $\chi^2_{\nu,-1}$ & VW & 2L & VW2 & preferred / status\\\midrule\endhead\bottomrule\endfoot""")
o=df.assign(o=df.kind.map({k:i for i,k in enumerate(kinds)})).sort_values(['o','dBIC_best'])
f=lambda x: '--' if pd.isna(x) else f"{x:.1f}"
for _,r in o.iterrows():
    A(f"{gn(r.Galaxy)} & {TY[int(r['T'])]} & {int(r.Q)} & {r.Inc:.0f} & {int(r.n)} & \\texttt{{{r.signs}}} & {r.p_runs1:.3f} & {r.min_runs_p:.3f} & {r.chi2nu_S:.2f} & {r.sigma_int:.1f} & {r.chi2nu_drop1:.2f} & {f(r.BIC_VW-r.BIC_S)} & {f(r.BIC_2L-r.BIC_S)} & {f(r.BIC_VW2-r.BIC_S)} & {NM[r.best]} / {r.status}\\\\")
A(r"\end{longtable}")
A(r"""\begin{longtable}{lllrrrll}
\caption{Preferred-model parameters. Single-L: $(M\,[10^9M_\odot],R\,[\mathrm{kpc}])$ $\pm$ statistical $\pm$ systematic; extensions: radii (kpc), window strengths, $\chi^2_\nu$ and runs $p$ of the preferred fit.}\\\toprule
Galaxy & Preferred & Status & $\chi^2_\nu$ & RMS$_\mathrm{rel}$ & $p_\mathrm{runs}$ & Radii & Strengths / single-L $(M,R)$\\\midrule\endfirsthead
\toprule Galaxy & Preferred & Status & $\chi^2_\nu$ & RMS$_\mathrm{rel}$ & $p_\mathrm{runs}$ & Radii & Strengths / single-L $(M,R)$\\\midrule\endhead\bottomrule\endfoot""")
for _,r in o.iterrows():
    if r.best=='S': extra=f"$M={r.M:.3g}\\pm{r.eM:.2g}\\pm{r.sM:.2g}$, $R={r.R:.3g}\\pm{r.eR:.2g}\\pm{r.sR:.2g}$"
    else: extra=str(r.pvals)
    A(f"{gn(r.Galaxy)} & {NM[r.best]} & {r.status} & {r.chi2nu_best:.2f} & {r.RMS_best:.3f} & {r.pruns_best:.3f} & {str(r.radii).replace('_','\\_')} & {extra}\\\\")
A(r"\end{longtable}}\end{landscape}")
A(r"\section*{Atlas: category-E fits with residuals}\includepdf[pages=-,landscape=true,scale=0.95]{catE_atlas.pdf}\end{document}")
open('CL_categoryE_report.tex','w').write('\n'.join(L))
