import pandas as pd,numpy as np
from scipy import stats
O=pd.read_csv('overall_classes.csv')
TY={0:'S0',1:'Sa',2:'Sab',3:'Sb',4:'Sbc',5:'Sc',6:'Scd',7:'Sd',8:'Sdm',9:'Sm',10:'Im',11:'BCD'}
CLS=['Single-L','Double (two-L reset)','Single-L + descent','Descent only','Multi-region','Flattening window','Unresolved (bulge / non-nested)']
gn=lambda g:g.replace('-','--'); lst=lambda s:', '.join(gn(x) for x in s)
C=lambda c:O[O.cls==c]; med=lambda s,c:np.median(s[c])
def vrat(s):
    ok=(s.Vflat>0)&s.v_asym.notna(); return (np.median(s.v_asym[ok]/s.Vflat[ok]), stats.pearsonr(s.v_asym[ok],s.Vflat[ok])[0], int(ok.sum())) if ok.sum()>3 else (np.nan,np.nan,0)
kw=stats.kruskal(*[C(c)['T'] for c in CLS[:4]])
L=[]; A=L.append
A(r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs,longtable,amsmath,amssymb,graphicx,caption,lscape}
\captionsetup{font=small,labelfont=bf}
\title{\bfseries Constant-Lagrangian inflow fits of the full SPARC sample:\\ overall classification of 175 rotation curves}
\author{Synthesis of the category A--E analyses, prepared for E.P.J.\ de Haas}\date{}
\begin{document}\maketitle
\section*{Summary}""")
s1,s2,s3,s4=C(CLS[0]),C(CLS[1]),C(CLS[2]),C(CLS[3])
A(f"""This report combines the five category analyses (A--E) of the SPARC rotation curves (175 galaxies) into one physical classification. It is based only on the stored results of those analyses; nothing was refitted. Every galaxy is assigned to the simplest CL description that the respective analysis accepted, using the CL inflow model and its extensions from [1,2]. The result is:
\\begin{{itemize}}\\itemsep0pt
\\item \\textbf{{Single-L}} ({len(s1)}): one constant Lagrangian $(M,R)$ describes the whole curve.
\\item \\textbf{{Double}} ({len(s2)}): a Lagrangian reset (two-L, nested spiral) or the equivalent rising window.
\\item \\textbf{{Single-L + descent}} ({len(s3)}): a CL curve followed by a Keplerian descent window.
\\item \\textbf{{Descent only}} ({len(s4)}): the inner CL bulge is unresolved and the curve is a compact central mass plus a Keplerian descent.
\\item \\textbf{{Multi-region}} ({len(C(CLS[4]))}): two windows, or a reset plus a window.
\\item \\textbf{{Flattening window}} ({len(C(CLS[5]))}).
\\item \\textbf{{Unresolved}} ({len(C(CLS[6]))}): bulge-dominated or non-nested galaxies that the present extensions describe only with unphysical parameters.
\\end{{itemize}}
The classes form a clear morphological sequence (Kruskal--Wallis on Hubble type for the four main classes, $p={kw.pvalue:.0e}$). Single-L and double galaxies are late types (median Sm), descent galaxies are intermediate spirals (median Sbc), descent-only and unresolved galaxies are early types (median Sb and Sab). Over the full sample the median $\\chi^2_\\nu$ goes from {O.chi2nu_S.median():.2f} (single-L) to {O.chi2nu_final.median():.2f} (final description) and the median RMS$_\\mathrm{{rel}}$ in $v^2$ from {O.RMS_S.median():.3f} to {O.RMS_final.median():.3f}. {int((O.chi2nu_final>2).sum())} galaxies remain at $\\chi^2_\\nu>2$; apart from NGC 4051 (single-L, $n=7$) all are high-precision curves, either massive bulged spirals with a sharp inner peak or well-sampled nearby disks that need more than one reset.""")
A(r"""
\section{Basis of the synthesis}
\paragraph{Models.} All descriptions use the CL inflow quantities $X=\sqrt{2GM/R}-H_zR$ and $Y(r)=\sqrt{2GM/r}-H_zr$ with $H_z=2.2\times10^{-18}\,\mathrm{s^{-1}}$, fitted to $v^2$ with $\sigma_{v^2}=2V_\mathrm{obs}\delta V_\mathrm{obs}$ (SPARC [3]).
\begin{itemize}\itemsep0pt
\item \textbf{Single-L} [2, Eqs.~17--18]: $v^2=\tfrac12X^2r^2/R^2$ for $r\le R$ and $\tfrac32X^2-Y^2$ beyond; asymptotic speed $v_L=\sqrt{3/2}X$.
\item \textbf{Two-L} [2, Eqs.~24--26]: $L_1(M_1,R_1)$ up to $R_2$, then $L_2(M_2,R_2)$.
\item \textbf{Virial window} [1, Eq.~6; 2, Eq.~19]: $+\,p[\tfrac12Y^2-\Phi_p]$ beyond $r_v$. With the continuous gauge and $H_z\to0$ this is $v^2=A+GM_N/r$, with $M_N=(p-2)M$. So $p>2$ is a Keplerian descent, $0<p<2$ a flattening and $p<0$ an additional rise.
\item \textbf{Two windows} [1, Eqs.~5--7].
\end{itemize}
\paragraph{Input analyses.} \textbf{A} (54): single-L adequate, with grades I--III. \textbf{B} (45): $+-+$ wave, two-L tested; accepted at $\Delta\mathrm{BIC}<-6$, with physical screening (nested / needs more regions / not nested). \textbf{C} (41): $-+-$ wave, virial window tested against single-L and two-L. \textbf{D} (18): other structure, five models tested. \textbf{E} (17): poor fit without detected structure (low-power runs test), extensions tested directly by BIC.
\paragraph{Assignment rules.} Each galaxy keeps the outcome of its own analysis, mapped as follows.
""")
A(r"\begin{table}[h]\centering\small\caption{Mapping of the category outcomes to the overall classes, and counts.}\begin{tabular}{llrrrrrr}\toprule Overall class & Sub-type & A & B & C & D & E & Total\\\midrule")
for c in CLS:
    s=C(c)
    for i,(sub,ss) in enumerate(s.groupby('sub')):
        cnt=[int((ss.src==k).sum()) for k in 'ABCDE']
        A(f"{c if i==0 else ''} & {sub.replace('+-+','$+-+$').replace('-+-','$-+-$').replace('<=','$\\le$').replace('<','$<$').replace('sigma_int','$\\sigma_\\mathrm{int}$')} & "+' & '.join(str(x) if x else '' for x in cnt)+f" & {len(ss)}\\\\")
    A(r"\midrule")
A(f"Total & & 54 & 45 & 41 & 18 & 17 & {len(O)}\\\\\\bottomrule\\end{{tabular}}\\end{{table}}")
A(r"""\noindent Descent galaxies from C are labelled \emph{descent only} when no data point lies inside the fitted CL bulge radius and $p\ge30$ (inner Lagrangian mass negligible against $M_N$), or when at most two points precede $r_v$. NGC 7814 (D) is assigned to this class because its descent starts at the first data points.
""")
A(r"\begin{figure}[h]\centering\includegraphics[width=\textwidth]{overall_fig_gallery.pdf}\caption{One representative per class, drawn from the stored best fits (dotted: single-L; solid: final description).}\end{figure}")
A(r"\section{The classes}")
def stats_line(s):
    v,rr,nn=vrat(s)
    out=f"$n$ median {int(med(s,'n'))}; $\\chi^2_\\nu$ {med(s,'chi2nu_S'):.2f}$\\to${med(s,'chi2nu_final'):.2f}; RMS$_\\mathrm{{rel}}$ {med(s,'RMS_S'):.3f}$\\to${med(s,'RMS_final'):.3f}; Q=1: {int((s.Q==1).sum())}/{len(s)}; final $\\chi^2_\\nu>2$: {int((s.chi2nu_final>2).sum())}; $n\\le13$ (tentative): {int(s.tentative.sum())}"
    if nn: out+=f"; asymptotic speed/$V_\\mathrm{{flat}}$ = {v:.2f} ($r$={rr:.2f}, {nn} galaxies)"
    return out
A(f"""\\subsection{{Single-L ({len(s1)})}}
{stats_line(s1)}.

One constant Lagrangian describes the full rotation curve. The class consists of the 54 category-A galaxies plus {len(s1)-54} galaxies from B--E whose residual structure, although detectable in sign, does not justify extra parameters ($\\Delta\\mathrm{{BIC}}\\ge-6$). These comprise {int((s1['sub']=='undecided (-6<=dBIC<-2)').sum())} B galaxies with positive but not strong evidence for a reset (NGC 3109, NGC 3972, NGC 4010, UGC 6446, UGC 7125, UGC 7323, UGC 7399), weak waves in B and C, structure within the errors in D, and E galaxies whose excess $\\chi^2$ corresponds to an intrinsic scatter of a few km\\,s$^{{-1}}$. The class is dominated by late types and dwarfs (median {TY[int(s1['T'].median())]}; {int((s1['T']>=9).sum())} Sm/Im/BCD). The curves are short: {int(s1.tentative.sum())} of {len(s1)} have $n\\le13$, so ``single-L adequate'' often means that the data cannot demand more. The best-determined quantity is $v_L$, which tracks $V_\\mathrm{{flat}}$ closely. $M$ and $R$ are strongly correlated (category-A median $\\rho_{{MR}}=0.97$), and only 17 category-A galaxies have both determined to better than 30\\% (grade I).

\\subsection{{Double: Lagrangian reset ({len(s2)})}}
{stats_line(s2)}.

The rotation curve switches from an inner Lagrangian (bulge/bar zone, $L_1$) to an outer one (disk, $L_2$) at $R_2$, the nested-spiral configuration. The class contains the {int((s2['sub']=='nested two-L').sum())} clean nested two-L galaxies of B, the {int((s2['sub']=='two-L, needs more regions').sum())} B galaxies improved by two-L but needing further regions ({lst(s2[s2['sub']=='two-L, needs more regions'].Galaxy)}), and {int((s2['sub']=='rising window (p<0)').sum())} galaxies from C, D and E where a rising window ($p<0$) is preferred; in most of these the two-L fit is statistically equivalent. It is the late-type class (median {TY[int(s2['T'].median())]}, {int((s2['T']>=7).sum())} of {len(s2)} Sd--Im; photometric bulge in {int((s2.Lb>0).sum())}). Typical scales: $R_1/R_2\\approx0.3$, $M_2/M_1\\approx8$, $v_{{L2}}/v_{{L1}}\\approx1.6$, $R_2$ at 1--4 disk scale lengths. Caveat: $R_2$ is a discontinuity location and is localised only to the gap between data points; the size of the $v^2$ jump is not constrained by the data.

\\subsection{{Single-L + descent ({len(s3)})}}
{stats_line(s3)}.

A CL rise and peak followed, beyond $r_v$, by a Keplerian descent $v^2=A+GM_N/r$ onto a floor $A$. These are intermediate spirals (median {TY[int(s3['T'].median())]}; photometric bulge in {int((s3.Lb>0).sum())} of {len(s3)}). The descent starts at $r_v\\approx$ 3--4 disk scale lengths, just beyond the peak. The floor and window radius are well determined (category C: 6\\% and 9\\%), the descent mass less so (23\\%), and the descent mass is of order of the baryonic mass. The virial strength $p$ ranges widely (category C: $p\\approx3.8$--60, 16--84\\%); the pure Kepler case $p=3$ is one end of this range, not the rule. Residual structure remains in {int((s3.pruns_final<0.05).sum())} galaxies, located at the inner peak of high-precision curves.

\\subsection{{Descent only ({len(s4)})}}
{stats_line(s4)}.

{lst(s4.Galaxy)}. All are Sab--Sc spirals ({', '.join(TY[int(x)] for x in s4['T'])}), with a photometric bulge in {int((s4.Lb>0).sum())} of {len(s4)}. The inner CL Lagrangian is not resolved: its radius lies inside the first data point, or the descent begins at the first points. The curve is described by a compact central mass with a Keplerian descent onto a floor. The quantities $r_v$, $A$ and $M_N$ are well defined; $M$, $R$ and $p$ are not. The floor lies well below $V_\\mathrm{{flat}}$ (median ratio {vrat(s4)[0]:.2f}) because these curves are still descending at the last point. This class is the direct successor of the earlier descent-only fits; with the window formulation the parameters are positive and remain defined for $H_z\\neq0$.

\\subsection{{Multi-region ({len(C(CLS[4]))})}}
{stats_line(C(CLS[4]))}.

{lst(C(CLS[4]).Galaxy)}. These need two windows or a reset plus a window. NGC 2841 (descent at 13\\,kpc, renewed rise at 44\\,kpc) and NGC 6015 (reset at 2.3\\,kpc, flattening at 3.5\\,kpc) are clean examples of the multi-region configurations of [1]. The others are improved but weakly constrained.

\\subsection{{Flattening window ({len(C(CLS[5]))})}}
{stats_line(C(CLS[5]))}.

ESO563-G021 and IC 4202 (Sbc, edge-on): beyond $r_v\\approx10$\\,kpc the curve rises more slowly than single-L ($0<p<2$), with no descent.

\\subsection{{Unresolved ({len(C(CLS[6]))})}}
{stats_line(C(CLS[6]))}.

{lst(C(CLS[6]).Galaxy)}. All are early or intermediate types (S0--Sc, median {TY[int(C(CLS[6])['T'].median())]}), with a photometric bulge in {int((C(CLS[6]).Lb>0).sum())} of {len(C(CLS[6]))}. Statistically the extensions improve them strongly, but only with non-nested resets (a $v^2$ drop at $R_2$, or $M_2<M_1$) or extreme window strengths with the CL bulge radius at its lower bound. Their residuals trace a sharp inner peak that the parabolic CL bulge ($v^2\\propto r^2$ inside $R$) cannot follow. This class marks the present limit of the model, not a separate dynamical regime.""")
A(r"\section{Morphological sequence}")
A(r"\begin{figure}[h]\centering\includegraphics[width=\textwidth]{overall_fig_morphology.pdf}\caption{Hubble-type composition of the classes (left) and Hubble type per galaxy (right; bar = median).}\end{figure}")
A(f"""The classes are ordered in morphology (Fig.~2). Single-L and double galaxies are late types and dwarfs (medians {TY[int(s1['T'].median())]} and {TY[int(s2['T'].median())]}). Single-L + descent galaxies are intermediate spirals (median {TY[int(s3['T'].median())]}). Descent-only and unresolved galaxies are the earliest types (medians {TY[int(s4['T'].median())]} and {TY[int(C(CLS[6])['T'].median())]}). The difference between the four main classes is highly significant (Kruskal--Wallis $H={kw.statistic:.1f}$, $p={kw.pvalue:.0e}$). The fraction of galaxies with a photometric bulge rises along the same sequence: {(s1.Lb>0).mean():.0%} (single-L), {(s2.Lb>0).mean():.0%} (double), {(s3.Lb>0).mean():.0%} (single + descent), {(s4.Lb>0).mean():.0%} (descent only), {(C(CLS[6]).Lb>0).mean():.0%} (unresolved). In CL terms: bulgeless disks either keep one Lagrangian or reset to a second one, bulged spirals develop a Keplerian descent beyond the peak, and bulge-dominated systems are descent-dominated or exceed the present bulge description.""")
A(r"\section{Global fit quality}")
A(r"\begin{figure}[h]\centering\includegraphics[width=\textwidth]{overall_fig_quality.pdf}\caption{Left: $\chi^2_\nu$ of single-L versus the final description. Middle: final RMS$_\mathrm{rel}$ per class. Right: model asymptotic speed ($v_L$ for single-L, $v_{L2}$ for nested two-L, $\sqrt A$ for descents) versus SPARC $V_\mathrm{flat}$.}\end{figure}")
A(r"\begin{table}[h]\centering\small\caption{Fit quality per class (medians).}\begin{tabular}{lrrrrrrr}\toprule Class & $N$ & $n$ & $\chi^2_{\nu}$ single-L & $\chi^2_\nu$ final & RMS$_\mathrm{rel}$ final & final $\chi^2_\nu>2$ & $n\le13$\\\midrule")
for c in CLS:
    s=C(c); A(f"{c} & {len(s)} & {int(med(s,'n'))} & {med(s,'chi2nu_S'):.2f} & {med(s,'chi2nu_final'):.2f} & {med(s,'RMS_final'):.3f} & {int((s.chi2nu_final>2).sum())} & {int(s.tentative.sum())}\\\\")
A(f"All & {len(O)} & {int(med(O,'n'))} & {med(O,'chi2nu_S'):.2f} & {med(O,'chi2nu_final'):.2f} & {med(O,'RMS_final'):.3f} & {int((O.chi2nu_final>2).sum())} & {int(O.tentative.sum())}\\\\\\bottomrule\\end{{tabular}}\\end{{table}}")
hi=O[O.chi2nu_final>2]
A(f"""With the final descriptions, {len(O)-len(hi)} of {len(O)} galaxies have $\\chi^2_\\nu\\le2$, and the median RMS$_\\mathrm{{rel}}$ in $v^2$ is {O.RMS_final.median():.3f}. The {len(hi)} galaxies above $\\chi^2_\\nu=2$ are {lst(hi.Galaxy)}. Apart from NGC 4051 ($n=7$, single-L with one deviant point), they are high-precision curves of two kinds: massive bulged spirals with a sharp inner peak (e.g.\\ NGC 5055, UGC 2953, UGC 6787), and well-sampled nearby disks whose curves need more than one reset (NGC 2403, NGC 5585, NGC 1003, DDO 154). SPARC errors are conservative on average (median $\\chi^2_\\nu<1$ in all well-fitting classes), so $\\chi^2_\\nu$ is a weak discriminator within classes; RMS$_\\mathrm{{rel}}$, residual structure and $\\Delta\\mathrm{{BIC}}$ are the more informative measures. The model asymptotic speed tracks $V_\\mathrm{{flat}}$ in every class (right panel). The systematic offsets reflect whether the curve is still rising (single-L, double: model asymptote above $V_\\mathrm{{flat}}$) or still descending (descent classes: floor below $V_\\mathrm{{flat}}$) at the last measured point.""")
A(r"\section{Limitations}")
A(r"""\begin{enumerate}\itemsep1pt
\item \textbf{Thresholds.} Class boundaries depend on $\Delta\mathrm{BIC}<-6$ and on the physical screening rules. The seven undecided B galaxies and the borderline rising-window galaxies (NGC 300, NGC 6946) could move between Single-L and Double with more data.
\item \textbf{Short curves.} 86 galaxies have $n\le13$. For these the sign-based screen has little power, and a ``single-L'' classification often reflects data limitations.
\item \textbf{Heterogeneous selection.} The five analyses tested slightly different model sets: B tested two-L only, while C--E compared windows and two-L. A uniform re-run with all models on all galaxies would remove this dependence; by design it was not done here.
\item \textbf{Bulge description.} The unresolved class and most remaining residual structure point to the inner CL bulge. An explicit bulge component is the main open refinement.
\item \textbf{Model features.} The $v^2$ jump at a Lagrangian reset is not constrained by data. Window strengths $p$ are degenerate with $M$ when the descent dominates, so $M_N$ and $A$ should be quoted instead.
\item \textbf{Interpretation.} Associating windows with galaxy history, as proposed in [1], requires independent indicators (morphology, kinematic asymmetry, HI warps, environment); these were not part of the analyses.
\end{enumerate}""")
A(r"\section{Conclusions}")
A(f"""Starting from a two-parameter constant Lagrangian, {len(s1)} of 175 SPARC galaxies need nothing more. A further {len(s2)+len(s3)+len(s4)} are described by one additional, physically interpretable feature: a Lagrangian reset (late types) or a Keplerian descent (bulged spirals). {len(C(CLS[4]))+len(C(CLS[5]))} need two features or a flattening, and {len(C(CLS[6]))} bulge-dominated galaxies remain beyond the present model. The ordering of these classes along the Hubble sequence is the main empirical result. It ties the CL description to galaxy structure: the number and kind of Lagrangian regions follow the prominence of the bulge.

\\section*{{References}}\\small
[1] E.P.J.\\ de Haas (2026), From Rotation Curves to Cosmic Time: Probing High-Redshift Expansion through Nested Spiral Dynamics, \\emph{{J.\\ High Energy Phys.\\ Grav.\\ Cosmol.}} 12, 334--367, doi:10.4236/jhepgc.2026.121022.\\par
[2] E.P.J.\\ de Haas, Galactic Rotation Curves and the Constant--Lagrangian Field: Empirical Tests within the $Q_g$ Rotor Framework (manuscript; single-L Eqs.~17--18, virial term Eq.~19, two-L Eqs.~24--26).\\par
[3] F.\\ Lelli, S.S.\\ McGaugh, J.M.\\ Schombert (2016), SPARC, \\emph{{AJ}} 152, 157.\\par
Category reports A--E (this project): CL\\_categoryA\\_report, CL\\_categoryB\\_report, CL\\_categoryC\\_report, CL\\_categoryD\\_report, CL\\_categoryE\\_report.
""")
A(r"\clearpage\begin{landscape}{\scriptsize\setlength{\tabcolsep}{3pt}")
A(r"""\begin{longtable}{llllrrrrrrrl}
\caption{All 175 SPARC galaxies: overall class, origin, sub-type and fit quality (single-L $\to$ final). t = tentative ($n\le13$).}\\\toprule
Galaxy & Class & Sub-type & Cat. & Type & Q & $n$ & $\chi^2_{\nu}$ S & $\chi^2_\nu$ final & RMS final & $\Delta$BIC & Quality / grade\\\midrule\endfirsthead
\toprule Galaxy & Class & Sub-type & Cat. & Type & Q & $n$ & $\chi^2_{\nu}$ S & $\chi^2_\nu$ final & RMS final & $\Delta$BIC & Quality / grade\\\midrule\endhead\bottomrule\endfoot""")
o=O.assign(o=O.cls.map({c:i for i,c in enumerate(CLS)})).sort_values(['o','sub','Galaxy'])
prev=None
for _,r in o.iterrows():
    if prev is not None and r.cls!=prev: A(r"\midrule")
    prev=r.cls
    sub=r['sub'].replace('+-+','$+-+$').replace('-+-','$-+-$').replace('<=','$\\le$').replace('<','$<$').replace('sigma_int','$\\sigma_\\mathrm{int}$')
    q=str(r.quality).replace('_','\\_')
    A(f"{gn(r.Galaxy)}{' t' if r.tentative else ''} & {r.cls} & {sub} & {r.src} & {TY[int(r['T'])]} & {int(r.Q)} & {int(r.n)} & {r.chi2nu_S:.2f} & {r.chi2nu_final:.2f} & {r.RMS_final:.3f} & {r.dBIC:.1f} & {q}\\\\")
A(r"\end{longtable}}\end{landscape}\end{document}")
open('CL_SPARC_overall_report.tex','w').write('\n'.join(L))
