import pandas as pd, numpy as np
from scipy import stats
df=pd.read_csv('catB_results.csv'); acc=df[df.accepted]; cl=df[df.outcome=='clean nested two-L']; rej=df[~df.accepted]
MM=df[df.outcome=='two-L, needs more regions']; NN=df[df.outcome=='not nested']
TY={0:'S0',1:'Sa',2:'Sab',3:'Sb',4:'Sbc',5:'Sc',6:'Scd',7:'Sd',8:'Sdm',9:'Sm',10:'Im',11:'BCD'}
gn=lambda g: g.replace('-','--'); med=lambda s,c: np.median(s[c])
def lst(s): return ', '.join(gn(g) for g in s)
def asym(x,lo,hi,d=3): return f"${x:.{d}g}^{{+{max(hi-x,0):.2g}}}_{{-{max(x-lo,0):.2g}}}$"
L=[]; A=L.append
A(r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs,longtable,amsmath,amssymb,graphicx,pdfpages,caption,lscape}
\captionsetup{font=small,labelfont=bf}
\title{\bfseries Two-Lagrangian CL inflow fits of the SPARC category-B galaxies:\\ detection, model selection, fit quality and robustness}
\author{Automated analysis for E.P.J.\ de Haas}\date{}
\begin{document}\maketitle
\section*{Summary}""")
A(f"""Category B contains the {len(df)} SPARC galaxies whose single-Lagrangian (single-L) CL residuals show a significant $+-+$ wave. For all of them the two-Lagrangian (two-L) model of Eqs.~(24)--(26) was fitted and compared with the single-L fit by BIC. In {len(acc)} galaxies the two-L description is accepted ($\\Delta\\mathrm{{BIC}}<-6$); in {len(rej)} the wave is real in sign but too weak to justify two extra parameters, and the single-L fit is kept. The accepted fits split into three groups. {len(cl)} are clean nested fits: physically ordered ($M_2>M_1$, no large drop at $R_2$) and statistically adequate. {len(MM)} are improved but still need more regions ($\\chi^2_\\nu>2$). {len(NN)} are not nested, meaning the fit exploits the discontinuity at $R_2$. For the clean nested fits the median RMS$_\\mathrm{{rel}}$ in $v^2$ falls from {med(cl,'RMS1'):.3f} to {med(cl,'RMS2'):.3f} and the median Durbin--Watson statistic rises from {med(cl,'DW1'):.2f} to {med(cl,'DW2'):.2f}. The outer scale is well determined ($\\sigma_{{v_{{L2}}}}/v_{{L2}}$ = {100*med(cl,'relvL2'):.1f}\\%, $\\sigma_{{R_2}}/R_2$ = {100*med(cl,'relR2'):.0f}\\% from profile likelihood). The inner scale is less well determined ($\\sigma_{{M_1}}/M_1$ = {100*med(cl,'relM1'):.0f}\\%). Eight of the 13 galaxies identified as double fits in 2018 are recovered as clean nested fits. The main caveats are that $R_2$ is a discontinuity location, whose likelihood is non-smooth and in some galaxies multimodal, and that residual structure persists in {int((cl.p_runs2<0.05).sum())} of the {len(cl)} clean fits, pointing to further (virial) regions.""")
A(r"""
\section{Sample and procedure}
\paragraph{Selection.} Category B was defined in the full-SPARC pipeline. After the single-L CL fit ($H_z=2.2\times10^{-18}\,\mathrm{s^{-1}}$, fits on $v^2$ with $\sigma_{v^2}=2V_\mathrm{obs}\delta V_\mathrm{obs}$), the weighted residuals must fail the one-sided runs test ($p_\mathrm{runs}<0.05$), agree to at least 80\% with a three-segment $+-+$ sign template, and agree better with it than with the reverse $-+-$ template.
\paragraph{Two-L model.} Eqs.~(24)--(26): bulge $r\le R_1$ and bar zone $R_1<r\le R_2$ governed by $L_1$ $(M_1,R_1)$, disk $r>R_2$ by $L_2$ $(M_2,R_2)$, with asymptotic speeds $v_{Li}=\sqrt{3/2}\,X_i$, $X_i=\sqrt{2GM_i/R_i}-H_zR_i$. A shared offset $\Phi_\mathrm{BH}$ is tested as a third variant.
\paragraph{Fitting and selection.} The two-L fit is seeded at $R_2=r_{\times1}/0.8$ (first $+\to-$ residual crossover) and cross-checked by a global grid and by the $R_2$ profile scan below; the best optimum is kept. A two-L fit is accepted if $\Delta\mathrm{BIC}=\mathrm{BIC_{2L}}-\mathrm{BIC_{1L}}<-6$ (strong evidence), and marked tentative when $n\le13$.
\paragraph{Uncertainties.} $M_1$, $R_1$, $M_2$: local covariance. $R_2$: because $v^2$ is discontinuous at $R_2$, $\chi^2(R_2)$ is piecewise smooth, changing slope whenever $R_2$ crosses a data radius, so the covariance error is unreliable. $R_2$ and $M_2$ are therefore given as profile-likelihood intervals ($\Delta\chi^2\le1$, other parameters refitted on a grid of 140 $R_2$ values). Jackknife: leave-one-out refits. Systematic: refits with distance $+\delta D$ and inclination $+\delta i$, added in quadrature. $H_z$ sensitivity: refit with $H_z=0$.
\paragraph{Post-hoc classification of accepted fits.} \textbf{Not nested (N)}: $M_2<M_1$ or $v^2$ drops by more than 40\% at $R_2$. \textbf{Needs more regions (M)}: $\chi^2_\nu>2$ after two-L. \textbf{Clean nested}: neither. Grades: \textbf{I} all four parameters within 30\% (profile errors for $R_2,M_2$), at least 2 points in bulge and bar zones and 3 in the disk, jackknife scatter $\le30\%$; \textbf{II} both $v_{L1}$ and $v_{L2}$ within 10\%; \textbf{III} otherwise.
""")
A(r"\section{Model selection outcome}")
A(r"\begin{table}[h]\centering\small\caption{Category-B outcome.}\begin{tabular}{lrl}\toprule Outcome & $N$ & Galaxies\\\midrule")
for nm,s in [('Clean nested two-L, grade I',cl[cl.grade=='I']),('Clean nested two-L, grade II',cl[cl.grade=='II']),('Two-L, needs more regions (M)',MM),('Not nested (N)',NN),
             ('Single-L kept, $-6\\le\\Delta$BIC$<-2$',rej[rej.dBIC<-2]),('Single-L kept, $-2\\le\\Delta$BIC$<0$',rej[(rej.dBIC>=-2)&(rej.dBIC<0)]),('Single-L kept, $\\Delta$BIC$\\ge0$',rej[rej.dBIC>=0])]:
    A(f"{nm} & {len(s)} & \\parbox[t]{{10.5cm}}{{\\raggedright {lst(s.Galaxy)}}}\\\\")
A(r"\bottomrule\end{tabular}\end{table}")
A(f"""All {len(df)} galaxies have significant single-L residual structure ($p_\\mathrm{{runs}}<0.05$ by selection). The rejected galaxies differ from the accepted ones mainly in amplitude, not in shape. Their single-L fits are already good (median $\\chi^2_\\nu$ = {med(rej,'chi2nu1'):.2f}, RMS$_\\mathrm{{rel}}$ = {med(rej,'RMS1'):.3f}, against {med(acc,'chi2nu1'):.2f} and {med(acc,'RMS1'):.3f} for the accepted ones), so the wave is coherent but lies within the SPARC errors. {int((rej.dBIC>=0).sum())} of them have $\\Delta\\mathrm{{BIC}}\\ge0$. Among these are F568-1, F574-1, F579-V1 and UGC 8286, which were single fits in 2018 as well. The seven galaxies with $-6\\le\\Delta\\mathrm{{BIC}}<-2$ (positive but not strong evidence) include the 2018 double fits NGC 3109, NGC 3972 and UGC 6446. With $n=10$--25 points, a threshold of $-6$ is conservative for them, and they are the natural candidates for confirmation with more data. Their status is therefore \\emph{{undecided}} rather than single-L.""")
A(r"\begin{figure}[h]\centering\includegraphics[width=\textwidth]{catB_fig_quality.pdf}\caption{Left: Durbin--Watson statistic before and after the two-L fit (2 = uncorrelated residuals). Middle: relative RMS in $v^2$, single-L versus two-L. Right: $\Delta$BIC versus number of points; acceptance threshold $-6$ (red) and tentative limit $n\le13$ (dotted). Colours: clean nested (green), needs more regions (orange), not nested (red), single-L kept (grey).}\end{figure}")
A(r"\section{Fit quality of the accepted fits}")
A(r"\begin{table}[h]\centering\small\caption{Fit-quality statistics (medians), single-L $\to$ two-L.}\begin{tabular}{lrrr}\toprule & Clean nested (13) & Needs more regions (5) & Not nested (5)\\\midrule")
for lab,c1,c2,d in [('$\\chi^2_\\nu$','chi2nu1','chi2nu2',2),('RMS$_\\mathrm{rel}$','RMS1','RMS2',3),('Durbin--Watson','DW1','DW2',2)]:
    A(lab+' & '+' & '.join(f"{med(s,c1):.{d}f} $\\to$ {med(s,c2):.{d}f}" for s in (cl,MM,NN))+'\\\\')
A('$p_\\mathrm{runs}<0.05$ after two-L & '+' & '.join(f"{int((s.p_runs2<0.05).sum())}/{len(s)}" for s in (cl,MM,NN))+'\\\\')
A('$p_\\mathrm{runs}<0.05$ after two-L$+\\Phi_\\mathrm{BH}$ & '+' & '.join(f"{int((s.p_runs3<0.05).sum())}/{len(s)}" for s in (cl,MM,NN))+'\\\\')
A('$\\Delta$BIC (median) & '+' & '.join(f"{med(s,'dBIC'):.1f}" for s in (cl,MM,NN))+'\\\\')
A(r"\bottomrule\end{tabular}\end{table}")
A(f"""\\paragraph{{Clean nested fits.}} The two-L model removes most of the coherent residual. The median RMS$_\\mathrm{{rel}}$ halves, the median $\\chi^2_\\nu$ drops from {med(cl,'chi2nu1'):.2f} to {med(cl,'chi2nu2'):.2f}, and the residual sign pattern is broken in all cases (atlas). Residual structure is not fully removed in {int((cl.p_runs2<0.05).sum())} galaxies ({lst(cl[cl.p_runs2<0.05].Galaxy)}). Adding the shared offset $\\Phi_\\mathrm{{BH}}$ brings $p_\\mathrm{{runs}}$ above 0.05 in DDO 161 and UGC 12732. In the others the remaining pattern is consistent with a further region (virial window) as introduced in the $H_z$ paper; in NGC 7793 it is the declining outer curve. As in category A, $\\chi^2_\\nu<1$ is common (median {med(cl,'chi2nu2'):.2f}; {int((cl.P_low2<0.01).sum())} galaxies with $P(\\chi^2\\le\\chi^2_\\mathrm{{obs}})<0.01$) because SPARC errors are conservative. Shapiro--Wilk rejects normality of the two-L residuals in {int((cl.p_SW2<0.05).sum())} of {len(cl)}.

\\paragraph{{Offset $\\Phi_\\mathrm{{BH}}$.}} Among the clean fits only IC 2574 strongly prefers the offset ($\\Delta\\mathrm{{BIC}}=-7.9$, $\\Phi_\\mathrm{{BH}}=23\\pm6$\\,km$^2$s$^{{-2}}$). ESO116-G012, NGC 247 and UGC 12732 show positive evidence ($-6<\\Delta\\mathrm{{BIC}}<-4.7$; $\\Phi_\\mathrm{{BH}}\\approx430$, 360 and 950\\,km$^2$s$^{{-2}}$). NGC 247 and UGC 12732 are the two galaxies where the 2018 fit also needed $V_0^2$ (359 and 949), in remarkable agreement.

\\paragraph{{Needs more regions.}} {lst(MM.Galaxy)}. The two-L fit improves these enormously (NGC 2403: $\\Delta\\mathrm{{BIC}}=-1719$), but $\\chi^2_\\nu$ stays at 2.8--7.9 and the atlas shows coherent outer residuals. These are three-region (bulge--bar--disk--virial) or three-Lagrangian systems.

\\paragraph{{Not nested.}} {lst(NN.Galaxy)}. Here the optimiser places $R_2$ so that $v^2$ drops by 45--85\\% or $M_2<M_1$. In NGC 4013 the inner radius also runs to its lower bound. These are mostly early types or edge-on systems (Sa--Sc; NGC 4013 is edge-on) whose rising--peaked--declining curves are not a bar-in-disk configuration. The statistical gain is real but the two-L parameters have no nested-spiral interpretation, so these galaxies should be moved to category C or D.""")
A(r"\section{Parameter determination}")
A(f"""For the clean nested fits the outer Lagrangian is much better determined than the inner one: median relative errors are {100*med(cl,'relM2'):.0f}\\% ($M_2$), {100*med(cl,'relR2'):.0f}\\% ($R_2$, profile) and {100*med(cl,'relvL2'):.1f}\\% ($v_{{L2}}$), against {100*med(cl,'relM1'):.0f}\\%, {100*med(cl,'relR1'):.0f}\\% and {100*med(cl,'relvL1'):.1f}\\% for $M_1$, $R_1$ and $v_{{L1}}$. As in category A, each pair is strongly correlated (median $\\rho_{{M_1R_1}}$ = {med(cl,'rho_M1R1'):.2f}), and the well-determined quantities are the two asymptotic speeds.

\\paragraph{{The $R_2$ likelihood.}} Fig.~2 shows $\\Delta\\chi^2(R_2)$. In most galaxies the minimum is sharp and lies between two data radii, so the $\\Delta\\chi^2\\le1$ interval is effectively the gap between neighbouring points and is often asymmetric. In NGC 247 and especially UGC 12732 the profile has several near-equal minima (UGC 12732: four $\\Delta\\chi^2\\le1$ islands between 9.6 and 10.1\\,kpc, plus shallow minima down to 5\\,kpc). Its $R_2$ is therefore set by the sampling rather than measured, and its grade-II status should be read with that in mind. This sampling dependence is intrinsic to a model with a discontinuity; a smooth transition of width $\\sim$ the point spacing would remove it.

\\paragraph{{Systematics, jackknife, $H_z$.}} Distance and inclination systematics exceed the statistical errors (median sys/stat {np.median(cl.sM1/cl.eM1):.1f} for $M_1$, {np.median(cl.sM2/cl.eM2p):.1f} for $M_2$, {np.median(cl.sR2/cl.eR2p):.1f} for $R_2$) and should be quoted alongside them. Leave-one-out scatter is within 30\\% for grade I but larger for several grade II galaxies (ESO116-G012: one point controls $M_1$). Setting $H_z=0$ changes no parameter by more than {100*np.max(np.abs(np.r_[cl.M1_Hz0/cl.M1,cl.R1_Hz0/cl.R1,cl.M2_Hz0/cl.M2,cl.R2_Hz0/cl.R2]-1)):.1f}\\%.""")
A(r"\begin{figure}[h]\centering\includegraphics[width=\textwidth]{catB_fig_profiles.pdf}\caption{Profile $\Delta\chi^2(R_2)$ for the 13 clean nested fits. Vertical grey lines are the data radii; the likelihood changes slope at each, so $R_2$ is localised between data points. Multiple minima (NGC 247, UGC 12732) indicate sampling-limited $R_2$.}\end{figure}")
A(r"\section{Physical consistency}")
ok=cl.Vflat>0
A(f"""\\begin{{itemize}}\\itemsep0pt
\\item \\textbf{{Seed versus fit.}} $R_2/r_{{\\times1}}$ = {np.median(cl.R2/cl.r_cross1):.2f} (16--84\\%: {np.percentile(cl.R2/cl.r_cross1,16):.2f}--{np.percentile(cl.R2/cl.r_cross1,84):.2f}), consistent with the seed factor $1/0.8=1.25$. The residual crossover is a good locator of the Lagrangian reset.
\\item \\textbf{{Nesting.}} $R_1/R_2$ = {np.median(cl.R1/cl.R2):.2f}, $M_2/M_1$ = {np.median(cl.M2/cl.M1):.1f} (16--84\\%: {np.percentile(cl.M2/cl.M1,16):.1f}--{np.percentile(cl.M2/cl.M1,84):.1f}), $v_{{L2}}/v_{{L1}}$ = {np.median(cl.vL2/cl.vL1):.2f}. The bar zone lies at about $0.3$--$1\\,R_2$, and the outer Lagrangian carries about eight times the inner mass.
\\item \\textbf{{Disk scale.}} $R_2/R_\\mathrm{{disk}}$ = {np.median(cl.R2/cl.Rdisk):.2f} (Spearman $\\rho$ = {stats.spearmanr(cl.R2,cl.Rdisk)[0]:.2f}), so the reset lies about 1--4 disk scale lengths out, where bars and inner rings are found.
\\item \\textbf{{Asymptotic speed.}} $v_{{L2}}$ tracks SPARC $V_\\mathrm{{flat}}$ ($r$ = {stats.pearsonr(cl.vL2[ok],cl.Vflat[ok])[0]:.3f}, {int(ok.sum())} galaxies), with $v_{{L2}}/V_\\mathrm{{flat}}$ = {np.median(cl.vL2[ok]/cl.Vflat[ok]):.2f}. As in category A, the CL curve is still rising at the last measured point.
\\item \\textbf{{Mass.}} $M_2/M_\\mathrm{{bar}}$ = {np.median(cl.M2/cl.Mbar):.2f} (16--84\\%: {np.percentile(cl.M2/cl.Mbar,16):.2f}--{np.percentile(cl.M2/cl.Mbar,84):.2f}), $r(\\log M_2,\\log M_\\mathrm{{bar}})$ = {stats.pearsonr(np.log10(cl.M2),np.log10(cl.Mbar))[0]:.2f}. The outer Lagrangian mass is comparable to the SPARC baryonic mass.
\\item \\textbf{{Jump at $R_2$.}} For the clean fits $v^2$ changes by {cl.jump.min():+.2f} to {cl.jump.max():+.2f} (median {cl.jump.median():+.2f}) across $R_2$. No data point lies at $R_2$, so the jump size is a model feature that the data do not constrain directly.
\\item \\textbf{{Morphology.}} The clean fits are all late types (Sc--Im: {', '.join(f"{TY[k]} {v}" for k,v in cl['T'].value_counts().sort_index().items())}), and {int((cl.Q==1).sum())} of {len(cl)} are Q=1. The non-nested and multi-region cases include all the early types.
\\end{{itemize}}""")
A(r"\begin{figure}[h]\centering\includegraphics[width=\textwidth]{catB_fig_physics.pdf}\caption{Accepted fits. Left to right: $R_2$ versus first residual crossover (dashed: seed relation); $R_2$ versus $R_\mathrm{disk}$; $v_{L2}$ versus SPARC $V_\mathrm{flat}$; $M_2$ versus $M_\mathrm{bar}$. Colours as in Fig.~1.}\end{figure}")
A(r"\section{Assessment}")
A(f"""\\textbf{{Detection works.}} The wave screen plus BIC selects {len(cl)} clean nested two-L galaxies. All are late types with Q mostly 1. They include 8 of the 13 galaxies identified as double fits by hand in 2018, and the automated $R_2$ agrees with the crossover radius. In these galaxies the two-L model is preferred by $\\Delta\\mathrm{{BIC}}$ from $-8$ to $-374$, removes the coherent residual wave, and yields an outer Lagrangian ($v_{{L2}}$, $M_2$) that matches independent photometric and kinematic scales.

\\textbf{{Grades.}} Grade I ({lst(cl[cl.grade=='I'].Galaxy)}) is the robust core, with every parameter determined to within 30\\% and each zone sampled. Grade II ({lst(cl[cl.grade=='II'].Galaxy)}) has well-determined asymptotic speeds, but one parameter is weakly constrained or influenced by one point. UGC 12732 and NGC 247 have sampling-limited $R_2$.

\\textbf{{Limits.}} (i) $\\Delta$BIC accepts any statistically better fit, so physical screening (N, M) is essential. 10 of the 23 accepted fits are not clean nested spirals. (ii) $R_2$ is localised only to the gap between data points, and the size of the $v^2$ jump is unconstrained. (iii) Residual structure remains in about half of the clean fits, mostly in the outer disk, which motivates the virial-window extension. (iv) For $n\\lesssim 15$ the $-6$ threshold is conservative, so the seven galaxies with $-6\\le\\Delta\\mathrm{{BIC}}<-2$ remain undecided.
""")
# tables
A(r"\clearpage\begin{landscape}{\scriptsize\setlength{\tabcolsep}{3pt}")
A(r"""\begin{longtable}{lllrrrrrrrrll}
\caption{All category-B galaxies: single-L fit, wave statistics and model-selection outcome. $\dagger$ = 2018 double fit.}\\\toprule
Galaxy & Type & Q & $n$ & $\chi^2_{\nu,1L}$ & RMS$_{1L}$ & $p_\mathrm{runs}$ & $A_{+-+}$ & $r_{\times1}$ & $\chi^2_{\nu,2L}$ & $\Delta$BIC & Outcome & Grade\\\midrule\endfirsthead
\toprule Galaxy & Type & Q & $n$ & $\chi^2_{\nu,1L}$ & RMS$_{1L}$ & $p_\mathrm{runs}$ & $A_{+-+}$ & $r_{\times1}$ & $\chi^2_{\nu,2L}$ & $\Delta$BIC & Outcome & Grade\\\midrule\endhead\bottomrule\endfoot""")
for _,r in df.sort_values('dBIC').iterrows():
    A(f"{gn(r.Galaxy)}{'$^\\dagger$' if r.double_2018 else ''} & {TY[int(r['T'])]} & {int(r.Q)} & {int(r.n)} & {r.chi2nu1:.2f} & {r.RMS1:.3f} & {r.p_runs1:.1e} & {r.A_pmp:.2f} & {r.r_cross1:.2f} & {r.chi2nu2:.2f} & {r.dBIC:.1f} & {r.outcome}{' (tent.)' if r.tentative else ''} & {r.grade}\\\\")
A(r"\end{longtable}")
A(r"""\begin{longtable}{llrlllllrrrr}
\caption{Accepted two-L fits: parameters. $R$ in kpc, $M$ in $10^9M_\odot$, $v_L$ in km\,s$^{-1}$. $R_1,M_1$: statistical $\pm$ systematic; $R_2,M_2$: profile $\Delta\chi^2\le1$ interval, and systematic in brackets. Jump = fractional $v^2$ change at $R_2$. $\Phi_\mathrm{BH}$ from the two-L$+\Phi_\mathrm{BH}$ fit.}\\\toprule
Galaxy & Grade & $n$ & $R_1$ & $M_1$ & $R_2$ & $M_2$ & $v_{L1}$ & $v_{L2}$ & zones & jump & $\Phi_\mathrm{BH}$\\\midrule\endfirsthead
\toprule Galaxy & Grade & $n$ & $R_1$ & $M_1$ & $R_2$ & $M_2$ & $v_{L1}$ & $v_{L2}$ & zones & jump & $\Phi_\mathrm{BH}$\\\midrule\endhead\bottomrule\endfoot""")
o=acc.assign(o=acc.outcome.map({'clean nested two-L':0,'two-L, needs more regions':1,'not nested':2})).sort_values(['o','grade','dBIC'])
prev=None
for _,r in o.iterrows():
    if prev is not None and r.o!=prev: A(r"\midrule")
    prev=r.o
    e=lambda x: '\\mathrm{n/a}' if pd.isna(x) else f"{x:.2g}"
    A(f"{gn(r.Galaxy)} & {r.grade}{' t' if r.tentative else ''} & {int(r.n)} & ${r.R1:.3g}\\pm{e(r.eR1)}\\pm{e(r.sR1)}$ & ${r.M1:.3g}\\pm{e(r.eM1)}\\pm{e(r.sM1)}$ & {asym(r.R2,r.R2_lo,r.R2_hi)} [{e(r.sR2)}] & {asym(r.M2,r.M2_lo,r.M2_hi)} [{e(r.sM2)}] & ${r.vL1:.1f}\\pm{e(r.evL1)}$ & ${r.vL2:.1f}\\pm{e(r.evL2)}$ & {int(r.n_bulge)}/{int(r.n_bar)}/{int(r.n_disk)} & {r.jump:+.2f} & ${r.Phi:.0f}\\pm{e(r.ePhi)}$\\\\")
A(r"\end{longtable}")
A(r"""\begin{longtable}{lrrrrrrrrrrrr}
\caption{Accepted two-L fits: goodness of fit. Subscripts 1, 2, 3 = single-L, two-L, two-L$+\Phi_\mathrm{BH}$. $P_<=P(\chi^2_2\le\chi^2_\mathrm{obs})$; jk = largest relative leave-one-out scatter of the four parameters.}\\\toprule
Galaxy & $\chi^2_{\nu1}$ & $\chi^2_{\nu2}$ & $\chi^2_{\nu3}$ & BIC$_1$ & BIC$_2$ & BIC$_3$ & RMS$_2$ & $p_\mathrm{runs,2}$ & $p_\mathrm{runs,3}$ & DW$_2$ & $P_<$ & jk\\\midrule\endfirsthead
\toprule Galaxy & $\chi^2_{\nu1}$ & $\chi^2_{\nu2}$ & $\chi^2_{\nu3}$ & BIC$_1$ & BIC$_2$ & BIC$_3$ & RMS$_2$ & $p_\mathrm{runs,2}$ & $p_\mathrm{runs,3}$ & DW$_2$ & $P_<$ & jk\\\midrule\endhead\bottomrule\endfoot""")
prev=None
for _,r in o.iterrows():
    if prev is not None and r.o!=prev: A(r"\midrule")
    prev=r.o
    A(f"{gn(r.Galaxy)} & {r.chi2nu1:.2f} & {r.chi2nu2:.2f} & {r.chi2nu3:.2f} & {r.BIC1:.1f} & {r.BIC2:.1f} & {r.BIC3:.1f} & {r.RMS2:.3f} & {r.p_runs2:.3f} & {r.p_runs3:.3f} & {r.DW2:.2f} & {r.P_low2:.3f} & {r.jk_rel_max:.2f}\\\\")
A(r"\end{longtable}}\end{landscape}")
A(r"\section*{Atlas 1: accepted two-L fits with residuals}\includepdf[pages=-,landscape=true,scale=0.95]{catB_atlas_accepted.pdf}")
A(r"\section*{Atlas 2: category-B galaxies with single-L kept}\includepdf[pages=-,landscape=true,scale=0.95]{catB_atlas_rejected.pdf}")
A(r"\end{document}")
open('CL_categoryB_report.tex','w').write('\n'.join(L))
