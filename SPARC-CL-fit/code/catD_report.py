import pandas as pd,numpy as np,pickle
df=pd.read_csv('catD_results.csv'); B=pickle.load(open('catD_best.pkl','rb'))
TY={0:'S0',1:'Sa',2:'Sab',3:'Sb',4:'Sbc',5:'Sc',6:'Scd',7:'Sd',8:'Sdm',9:'Sm',10:'Im',11:'BCD'}
gn=lambda g:g.replace('-','--'); lst=lambda s:', '.join(gn(x) for x in s); med=lambda s,c: np.median(s[c])
NM={'S':'single-L','VW':'one window','2L':'two-L','VW2':'two windows','2LVW':'two-L + disk window'}
st=lambda k: df[df.status==k]
acc=df[df.best!='S']
L=[]; A=L.append
A(r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs,longtable,amsmath,amssymb,graphicx,pdfpages,caption,lscape}
\captionsetup{font=small,labelfont=bf}
\title{\bfseries Multi-region CL inflow fits of the SPARC category-D galaxies:\\ can the structured residuals be absorbed by the existing model extensions?}
\author{Automated analysis for E.P.J.\ de Haas}\date{}
\begin{document}\maketitle
\section*{Summary}""")
A(f"""Category D contains the {len(df)} SPARC galaxies whose single-Lagrangian (single-L) residuals are significantly structured (runs test $p<0.05$) but match neither the $+-+$ template of category B nor the $-+-$ template of category C. Before attributing their kinematics to the history of the galaxy (mergers, accretion), we test whether the existing CL model extensions absorb the structure: one virial window [1,2], two Lagrangians [2], two virial windows [1], and two Lagrangians with a disk window. The result is a split into four groups, not a single class. (i) In {len(st('single-L adequate'))} galaxies no extension is justified ($\\Delta\\mathrm{{BIC}}\\ge-6$): the sign pattern is real, but its amplitude lies within the SPARC errors (single-L $\\chi^2_\\nu=0.03$--$0.26$). (ii) {len(st('clean'))} galaxies are described cleanly by one of the extensions with well-determined parameters. (iii) {len(st('weakly constrained'))} are improved, but their parameters are poorly constrained. (iv) {len(st('phenomenological'))} bulge-dominated S0--Sbc galaxies are improved enormously ($\\Delta\\mathrm{{BIC}}$ down to $-3916$), but only with unphysical parameters: the inner CL bulge collapses ($R\\to0$) and window strengths reach $|p|>100$. For this group the structure points to a missing bulge component rather than to history. Category D therefore does not support the general conclusion that historical dynamics dominate. It is mainly a mixture of residuals within the errors, curves needing more regions, and massive bulges not represented by the CL bulge.""")
A(r"""
\section{Models}
All models use the CL inflow quantities $X=\sqrt{2GM/R}-H_zR$ and $Y(r)=\sqrt{2GM/r}-H_zr$ with $H_z=2.2\times10^{-18}\,\mathrm{s^{-1}}$ and are fitted to $v^2$ with $\sigma_{v^2}=2V_\mathrm{obs}\delta V_\mathrm{obs}$.
\begin{itemize}\itemsep1pt
\item \textbf{S} single-L [2, Eqs.~17--18]: $v^2=\tfrac12X^2r^2/R^2$ ($r\le R$), $\tfrac32X^2-Y^2$ ($r>R$); $k=2$.
\item \textbf{VW} one virial window [1, Eq.~6; 2, Eq.~19]: $+\,p[\tfrac12Y(r)^2-\Phi_p]$ for $r\ge r_v$; $k=4$.
\item \textbf{2L} two Lagrangians [2, Eqs.~24--26]: $L_1(M_1,R_1)$ up to $R_2$, $L_2(M_2,R_2)$ beyond; $k=4$.
\item \textbf{VW2} two virial windows [1, Eqs.~5--7]: strength $p$ on $[r_{v1},r_{v2})$, $q$ beyond $r_{v2}$; $k=6$.
\item \textbf{2LVW} two Lagrangians with a window of strength $p$ in the $L_2$ disk beyond $r_v>R_2$; $k=6$.
\end{itemize}
Window gauges $\Phi$ are fixed by continuity of $v^2$ at each window radius, and window strengths are fitted as descent masses $M_N=(p-2)M$ (see the category-C report). For $H_z\to0$ a window adds $GM_N/r$ plus a constant, so $p>2$ is a Keplerian descent, $0<p<2$ a flattening and $p<0$ an extra rise.

\paragraph{Selection.} The preferred model has the lowest BIC, with a simpler model preferred when it lies within $\Delta\mathrm{BIC}<2$. An extension is accepted only if it beats single-L by $\Delta\mathrm{BIC}<-6$. For galaxies with $n\le7$ the six-parameter models are not fitted.
\paragraph{Status of the preferred fit.} \textbf{Clean}: all parameters within 60\%, no parameter at a bound, $|p|,|q|\le100$, $\chi^2_\nu\le2$. \textbf{Weakly constrained}: some parameter uncertainty above 60\% (degenerate radii or masses), or $\chi^2_\nu>2$. \textbf{Phenomenological}: the bulge radius at its lower bound ($R\le0.025$\,kpc) or a window strength $|p|>100$; the fit is statistically excellent but the parameters have no physical CL reading. Uncertainties: covariance, leave-one-out jackknife, distance/inclination refits and an $H_z=0$ refit, as in the A--C reports.
""")
A(r"\section{Outcome}")
A(r"\begin{table}[h]\centering\small\caption{Category-D outcome by status of the preferred fit.}\begin{tabular}{llp{9.5cm}}\toprule Status & $N$ & Galaxies (type, preferred model)\\\midrule")
for k in ['single-L adequate','clean','weakly constrained','phenomenological']:
    s=st(k); A(f"{k} & {len(s)} & "+', '.join(f"{gn(r.Galaxy)} ({TY[int(r['T'])]}, {NM[r.best]})" for _,r in s.iterrows())+"\\\\")
A(r"\bottomrule\end{tabular}\end{table}")
A(r"\begin{figure}[h]\centering\includegraphics[width=\textwidth]{catD_fig_models.pdf}\caption{Left: $\Delta$BIC of each extension relative to single-L (symlog scale; red: $-6$). Galaxy labels are coloured by the status of the preferred fit: clean (green), weakly constrained (orange), phenomenological (red), single-L adequate (grey). Right: Durbin--Watson statistic, single-L versus preferred model.}\end{figure}")
c=st('clean'); w=st('weakly constrained'); ph=st('phenomenological'); sa=st('single-L adequate')
A(f"""\\paragraph{{Single-L adequate ({len(sa)}).}} {lst(sa.Galaxy)}. These are mostly late types and dwarfs ({', '.join(TY[int(x)] for x in sa['T'])}) with very small single-L $\\chi^2_\\nu$ (median {med(sa,'chi2nu_S'):.2f}). The runs test detects a coherent sign pattern, but no extension improves BIC, because the pattern lies well inside the error bars. UGC 1281 ($\\chi^2_\\nu=0.03$) is the galaxy for which [1] introduced the virial term on the basis of its outer six points. With the full error bars and BIC this refinement is not statistically required; [1] reported the same, a reduction of RMS$_\\mathrm{{rel}}$ from 6.5\\% to 6.3\\%.

\\paragraph{{Clean ({len(c)}).}} NGC 2841 (Sb) is described by two windows: a Keplerian descent ($p\\approx14$) beyond $r_{{v1}}\\approx13$\\,kpc, followed by a renewed rise ($q<0$) beyond $r_{{v2}}\\approx44$\\,kpc. $\\chi^2_\\nu$ falls from 9.3 to 0.40 and $\\Delta\\mathrm{{BIC}}=-413$. This is a category-C descent with an outer rise, exactly the two-window configuration of [1]. NGC 6015 (Scd) is described by a Lagrangian reset at $R_2\\approx2.3$\\,kpc followed by a flattening window ($p\\approx1.8$) at $r_v\\approx3.5$\\,kpc, i.e.\\ a B-type and a C-type feature combined ($\\chi^2_\\nu$ 8.7$\\to$1.7). NGC 300 (Sd) needs one rising window ($p<0$) at $r_v\\approx2.5$\\,kpc. Two-L is equally good here ($\\Delta\\mathrm{{BIC}}$ +0.05), so NGC 300 is a B/C borderline case.

\\paragraph{{Weakly constrained ({len(w)}).}} NGC 7814 (Sab) is a descent from the first point; the window reproduces it ($\\chi^2_\\nu$ 2.96$\\to$0.67), but $R$ and $r_v$ (0.4 and 0.9\\,kpc) are degenerate with the inner masses. NGC 4138 (S0) has only 7 points. NGC 1090 (Sbc) prefers a two-L + window fit in which $R_2$ collapses onto $R_1$, i.e.\\ effectively one window with a mass change. UGC 128 (Sdm) is improved by two windows ($\\chi^2_\\nu$ 6.1$\\to$2.9) but remains structured. These galaxies are compatible with the C or B pictures, but the data do not fix the parameters.

\\paragraph{{Phenomenological ({len(ph)}).}} {lst(ph.Galaxy)} ({', '.join(TY[int(x)] for x in ph['T'])}). Two virial windows lower $\\chi^2_\\nu$ dramatically (UGC 9133: 64$\\to$4.7; UGC 2487: 38$\\to$0.66), but in every case the parameters become extreme: the inner CL bulge shrinks to $R\\le0.06$\\,kpc (four of five) and window strengths reach $|p|$ or $|q|\\approx10^2$--$3\\times10^3$. The windows then act as a flexible spline, not as virial regions. All five are early and intermediate types (S0--Sbc) with prominent bulges: they have a sharp inner peak that the parabolic CL bulge ($v^2\\propto r^2$ inside $R$) cannot follow. The structure in their residuals is therefore a model limitation of the bulge description and should not be read as evidence of history.""")
A(r"\section{Fit quality of the accepted extensions}")
A(r"\begin{table}[h]\centering\small\caption{Fit-quality statistics (medians, single-L $\to$ preferred model).}\begin{tabular}{lrrr}\toprule & Clean (3) & Weakly constrained (4) & Phenomenological (5)\\\midrule")
for lab,c1,c2,dd in [('$\\chi^2_\\nu$','chi2nu_S','chi2nu_best',2),('RMS$_\\mathrm{rel}$','RMS_S','RMS_best',3),('Durbin--Watson','DW_S','DW_best',2)]:
    A(lab+' & '+' & '.join(f"{med(s,c1):.{dd}f} $\\to$ {med(s,c2):.{dd}f}" for s in (c,w,ph))+'\\\\')
A('$p_\\mathrm{runs}<0.05$ after & '+' & '.join(f"{int((s.pruns_best<0.05).sum())}/{len(s)}" for s in (c,w,ph))+'\\\\')
A('$\\Delta$BIC (median) & '+' & '.join(f"{med(s,'dBIC_best'):.0f}" for s in (c,w,ph))+'\\\\')
A(r"\bottomrule\end{tabular}\end{table}")
A(f"""Residual structure remains (runs $p<0.05$) in {int((acc.pruns_best<0.05).sum())} of the {len(acc)} accepted fits, including NGC 2841 and NGC 6015. These are high-precision curves (Q=1--2, $n=44$--50) in which small coherent deviations, e.g.\\ from spiral-arm streaming, become significant at the few-per-cent level (RMS$_\\mathrm{{rel}}$ 0.02--0.06). As in categories A--C, the SPARC errors are conservative for most galaxies. Distance and inclination systematics are of the same order as the statistical errors. Setting $H_z=0$ changes the clean fits by at most 7\\%; the phenomenological fits move by up to about 110\\%, consistent with their degeneracy.""")
A(r"\section{Assessment: history versus model structure}")
A(f"""\\textbf{{Can the dynamics of category D be attributed to the history of the galaxies?}} Not on the basis of these fits. Step 1 shows that the class is heterogeneous:
\\begin{{enumerate}}\\itemsep1pt
\\item \\textbf{{{len(sa)} galaxies}} have structure within the errors; single-L is statistically sufficient.
\\item \\textbf{{{len(c)+len(w)} galaxies}} are absorbed, cleanly or with weak constraints, by the existing extensions of [1,2]: descent and rising windows, a Lagrangian reset, or combinations of these. They are B- and C-type features that occur together or close to the centre, which is why the single-template classification missed them.
\\item \\textbf{{{len(ph)} bulge-dominated S0--Sbc galaxies}} require unphysical parameters. Their residuals trace a massive, compact bulge that the CL bulge parabola cannot represent, which is a limitation of the inner model.
\\end{{enumerate}}
Galaxy history (accretion, mergers, settled virial matter) remains the interpretation proposed in [1] for virial windows in general. That applies equally to categories C and D, and is not established for D specifically. A history-specific conclusion would require step 2: comparing the galaxies that still need windows with independent indicators of disturbance, such as lopsided or tidal morphology, approaching/receding asymmetry of the rotation curve, HI warps and environment. For the phenomenological group, a separate bulge component is needed first: either a Newtonian or CL bulge with its own scale, or the inner bulge region of [1] with a free inner profile. Only then can its residual structure be interpreted.

\\section*{{References}}\\small
[1] E.P.J.\\ de Haas (2026), From Rotation Curves to Cosmic Time: Probing High-Redshift Expansion through Nested Spiral Dynamics, \\emph{{J.\\ High Energy Phys.\\ Grav.\\ Cosmol.}} 12, 334--367, doi:10.4236/jhepgc.2026.121022 (virial term Eq.~3; multi-region virial model Eqs.~5--7).\\par
[2] E.P.J.\\ de Haas, Galactic Rotation Curves and the Constant--Lagrangian Field: Empirical Tests within the $Q_g$ Rotor Framework (manuscript; single-L Eqs.~17--18, virial term Eq.~19, two-L Eqs.~24--26).\\par
[3] F.\\ Lelli, S.S.\\ McGaugh, J.M.\\ Schombert (2016), SPARC, \\emph{{AJ}} 152, 157.
""")
A(r"\clearpage\begin{landscape}{\scriptsize\setlength{\tabcolsep}{3pt}")
A(r"""\begin{longtable}{lllrlrrrrrrrrll}
\caption{All category-D galaxies: residual signs of the single-L fit and model comparison ($\Delta$BIC relative to single-L; -- = not fitted, $n\le7$).}\\\toprule
Galaxy & Type & Q & $n$ & single-L residual signs & $\chi^2_{\nu,S}$ & $p_\mathrm{runs}$ & VW & 2L & VW2 & 2LVW & best & $\chi^2_{\nu,best}$ & $p_\mathrm{runs,best}$ & Status\\\midrule\endfirsthead
\toprule Galaxy & Type & Q & $n$ & signs & $\chi^2_{\nu,S}$ & $p_\mathrm{runs}$ & VW & 2L & VW2 & 2LVW & best & $\chi^2_{\nu,best}$ & $p_\mathrm{runs,best}$ & Status\\\midrule\endhead\bottomrule\endfoot""")
o=df.assign(o=df.status.map({'clean':0,'weakly constrained':1,'phenomenological':2,'single-L adequate':3})).sort_values(['o','dBIC_best'])
f=lambda x: '--' if pd.isna(x) else f"{x:.1f}"
for _,r in o.iterrows():
    sg=r.signs if len(r.signs)<=40 else r.signs[:38]+'..'
    A(f"{gn(r.Galaxy)} & {TY[int(r['T'])]} & {int(r.Q)} & {int(r.n)} & \\texttt{{{sg}}} & {r.chi2nu_S:.2f} & {r.p_runs1:.1e} & {f(r.BIC_VW-r.BIC_S)} & {f(r.BIC_2L-r.BIC_S)} & {f(r.BIC_VW2-r.BIC_S)} & {f(r.BIC_2LVW-r.BIC_S)} & {r.best} & {r.chi2nu_best:.2f} & {r.pruns_best:.3f} & {r.status}\\\\")
A(r"\end{longtable}")
A(r"""\begin{longtable}{llrlll}
\caption{Accepted extensions: preferred-model parameters. Radii in kpc; $p,q$ window strengths ($p=2+M_N/M$). Parameter vector order: S $(M,R)$; VW $(M,R,r_v-R,M_N)$; 2L $(M_1,R_1,M_2,R_2-R_1)$; VW2 $(M,R,r_{v1}-R,M_{N1},r_{v2}-r_{v1},M_{N2})$; 2LVW $(M_1,R_1,M_2,R_2-R_1,r_v-R_2,M_N)$; masses in $10^9M_\odot$.}\\\toprule
Galaxy & Model & $k$ & Radii & Strengths & Parameters $\pm$ statistical error\\\midrule\endfirsthead
\toprule Galaxy & Model & $k$ & Radii & Strengths & Parameters $\pm$ statistical error\\\midrule\endhead\bottomrule\endfoot""")
for _,r in o[o.best!='S'].iterrows():
    b=B[r.Galaxy]; pe=', '.join(f"${a:.3g}\\pm{e:.2g}$" if np.isfinite(e) else f"${a:.3g}$" for a,e in zip(b['x'],b['e']))
    A(f"{gn(r.Galaxy)} & {r.best} & {int(r.k)} & {r.radii.replace('_','\\_')} & {r.pvals if isinstance(r.pvals,str) else '--'} & {pe}\\\\")
A(r"\end{longtable}}\end{landscape}")
A(r"\section*{Atlas: category-D fits with residuals}\includepdf[pages=-,landscape=true,scale=0.95]{catD_atlas.pdf}\end{document}")
open('CL_categoryD_report.tex','w').write('\n'.join(L))
