import sys,pickle,numpy as np,pandas as pd; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from cl2_fit import v2_two
F=pickle.load(open('pipeline_fits.pkl','rb')); df=pd.read_csv('pipeline_summary.csv')
TY={0:'S0',1:'Sa',2:'Sab',3:'Sb',4:'Sbc',5:'Sc',6:'Scd',7:'Sd',8:'Sdm',9:'Sm',10:'Im',11:'BCD'}
def gn(g): return g.replace('-','--').replace('_','\\_')
def pm(x,e,d=3):
    if x is None or np.isnan(x): return '--'
    if e is None or np.isnan(e): return f"${x:.{d}g}$ (unc.)"
    return f"${x:.{d}g}\\pm{e:.2g}$"
# physical / adequacy flags for accepted fits
flags={}
for g in df[df.accepted==True].Galaxy:
    q=F[g]['two']['params']; lo=v2_two(np.array([q[3]*(1-1e-7)]),q[0]*1e9,q[1],q[2]*1e9,q[3])[0]; hi=v2_two(np.array([q[3]*(1+1e-7)]),q[0]*1e9,q[1],q[2]*1e9,q[3])[0]
    j=(hi-lo)/lo; fl=[]
    if q[2]<q[0] or j<-0.4: fl.append('N')
    if F[g]['two']['chi2nu']>2: fl.append('M')
    flags[g]=(j,''.join(fl))
df['jump']=df.Galaxy.map(lambda g: flags.get(g,(np.nan,''))[0]); df['flag']=df.Galaxy.map(lambda g: flags.get(g,(np.nan,''))[1])
def outcome(r):
    if not r.category.startswith('B'): return ''
    if r.accepted: return 'two-L'+(' (tent.)' if r.tentative else '')+(f' [{r.flag}]' if r.flag else '')
    return 'single-L kept'
df['outcome']=df.apply(outcome,axis=1)
df.to_csv('pipeline_summary_SPARC175.csv',index=False,float_format='%.5g')
L=[]; A=L.append
A(r"""\documentclass[9pt]{extarticle}
\usepackage[a4paper,landscape,margin=1.2cm]{geometry}
\usepackage{booktabs,longtable,multirow,amsmath,graphicx,pdfpages,xcolor,caption}
\captionsetup{font=small,labelfont=bf}
\renewcommand{\arraystretch}{1.05}
\begin{document}
\begin{center}{\Large\bfseries Constant-Lagrangian inflow pipeline applied to SPARC (175 galaxies)}\\[4pt]
{\small Single-L and two-L models of de Haas (SPARC $H_z$ paper, Eqs.\ 17--18 and 24--26), $H_z=2.2\times10^{-18}\,\mathrm{s^{-1}}$ fixed; fits on $v^2(r)$ with $\sigma_{v^2}=2V_\mathrm{obs}\delta V_\mathrm{obs}$; data Lelli, McGaugh \& Schombert (2016).}\end{center}
\section*{Pipeline}
\begin{enumerate}\itemsep0pt
\item Single-L fit $(R,M)$ to every galaxy (14 starting radii, bounded least squares).
\item Wave screen on the weighted residuals $\mathrm{WR}_i=(v^2_\mathrm{obs}-v^2_\mathrm{mod})/\sigma_{v^2}$: one-sided Wald--Wolfowitz runs test ($p_\mathrm{runs}$, too few sign runs) and the best agreement $A_{+-+}$ ($A_{-+-}$) of the residual signs with a three-segment template (each segment $\geq2$ points). Categories: \textbf{A} adequate ($p_\mathrm{runs}\ge0.05$, $\chi^2_\nu\le1$); \textbf{B} $+-+$ wave ($p_\mathrm{runs}<0.05$, $A_{+-+}\ge0.8$, $A_{+-+}>A_{-+-}$); \textbf{C} $-+-$ wave (same with signs reversed); \textbf{D} structured, other ($p_\mathrm{runs}<0.05$); \textbf{E} poor, unstructured ($p_\mathrm{runs}\ge0.05$, $\chi^2_\nu>1$).
\item Category B: two-L fit seeded at $R_2\simeq r_{\times1}/0.8$ ($r_{\times1}$ = first $+\to-$ crossover), cross-checked with a global grid search; also two-L\,+\,$\Phi_\mathrm{BH}$.
\item Accept the two-L description when $\Delta\mathrm{BIC}=\mathrm{BIC}_\mathrm{2L}-\mathrm{BIC}_\mathrm{1L}<-6$.
\item Accepted fits with $n\le13$ are marked tentative.
\end{enumerate}
Post-hoc flags on accepted fits (not part of the acceptance rule): \textbf{N} = not a nested configuration ($M_2<M_1$, or $v^2$ drops by more than 40\% at $R_2$); \textbf{M} = still inadequate after two-L ($\chi^2_\nu>2$), i.e.\ more regions (virial windows, a third Lagrangian) needed.
""")
cnt=df.category.value_counts().sort_index(); acc=df[df.accepted==True]
A(r"\begin{table}[h]\centering\small\caption{Pipeline outcome.}\begin{tabular}{lr}\toprule Category after single-L fit & $N$\\\midrule")
for k,v in cnt.items(): A(f"{k.replace('+-+','$+-+$').replace('-+-','$-+-$')} & {v}\\\\")
A(r"\midrule")
A(f"B: two-L accepted ($\\Delta$BIC$<-6$) & {len(acc)}\\\\ \\quad of which tentative ($n\\le13$) & {int(acc.tentative.sum())}\\\\ \\quad of which flagged N (not nested) & {int(acc.flag.str.contains('N').sum())}\\\\ \\quad of which flagged M (still $\\chi^2_\\nu>2$) & {int(acc.flag.str.contains('M').sum())}\\\\")
A(f"B: single-L kept ($\\Delta$BIC$\\ge-6$) & {int((df.category.str.startswith('B')&(df.accepted!=True)).sum())}\\\\")
A(f"2018 RMWRSS double-fit galaxies recovered as accepted two-L & {int(acc.double_2018.sum())} / 13\\\\")
A(f"Seeded two-L start reached the global-grid optimum & {int(df.seed_found_best.sum())} / {int(df.seed_found_best.notna().sum())}\\\\")
A(r"\bottomrule\end{tabular}\end{table}\clearpage")
# Table: categories for all 175
A(r"""{\scriptsize\setlength{\tabcolsep}{3.5pt}
\begin{longtable}{lllrrrrrrrrcrl}
\caption{Single-L fit and wave screen for all 175 SPARC galaxies, sorted by category and $\chi^2_\nu$. $R$ in kpc, $M$ in $10^9M_\odot$; $\dagger$ = 2018 RMWRSS double-fit galaxy.}\\\toprule
Galaxy & Type & Q & $n$ & $R$ & $M$ & $\chi^2_\nu$ & RMS$_\mathrm{rel}$ & $p_\mathrm{runs}$ & $A_{+-+}$ & $A_{-+-}$ & Cat & $\Delta$BIC & Outcome\\\midrule\endfirsthead
\toprule Galaxy & Type & Q & $n$ & $R$ & $M$ & $\chi^2_\nu$ & RMS$_\mathrm{rel}$ & $p_\mathrm{runs}$ & $A_{+-+}$ & $A_{-+-}$ & Cat & $\Delta$BIC & Outcome\\\midrule\endhead
\bottomrule\endfoot""")
prev=None
for _,r in df.sort_values(['category','chi2nu1']).iterrows():
    c=r.category[0]
    if prev and c!=prev: A(r"\midrule")
    prev=c
    dag='$^\\dagger$' if r.double_2018 else ''
    f=lambda x,d=2: '--' if pd.isna(x) else f"{x:.{d}f}"
    A(f"{gn(r.Galaxy)}{dag} & {TY[r['T']]} & {r.Q} & {r.n} & {pm(r.R,r.eR)} & {pm(r.M,r.eM)} & {r.chi2nu1:.2f} & {r.RMSrel1:.3f} & {('%.1e'%r.p_runs) if r.p_runs<0.001 else f(r.p_runs,3)} & {f(r.A_pmp)} & {f(r.A_mpm)} & {c} & {f(r.dBIC,1)} & {r.outcome}\\\\")
A(r"\end{longtable}}\clearpage")
# Double-fit tables
order=acc.sort_values('dBIC').Galaxy.tolist()
def pref(g):
    o=F[g]; ms=[('Single-L',o['single']),('Two-L',o['two']),('Two-L + $\\Phi_\\mathrm{BH}$',o['twophi'])]
    b=min(m['BIC'] for _,m in ms)
    for name,m in ms:
        if m['BIC']<=b+2: return name
A(r"""{\scriptsize\setlength{\tabcolsep}{4pt}
\begin{longtable}{llrlcccccc}
\caption{Accepted two-L galaxies: best-fit parameters ($1\sigma$, local covariance). Radii in kpc, masses in $10^9M_\odot$, $\Phi_\mathrm{BH}$ in (km\,s$^{-1}$)$^2$; $^\star$ preferred model (lowest BIC, simpler model preferred within $\Delta$BIC$<2$). Jump = fractional change of $v^2$ across $R_2$ in the two-L fit.}\\\toprule
Galaxy & Type & $n$ & Model & $R_1$ & $M_1$ & $R_2$ & $M_2$ & $\Phi_\mathrm{BH}$ & Jump / flags\\\midrule\endfirsthead
\toprule Galaxy & Type & $n$ & Model & $R_1$ & $M_1$ & $R_2$ & $M_2$ & $\Phi_\mathrm{BH}$ & Jump / flags\\\midrule\endhead\bottomrule\endfoot""")
for g in order:
    o=F[g]; r=df[df.Galaxy==g].iloc[0]; P=pref(g)
    nm=gn(g)+('$^\\dagger$' if r.double_2018 else '')+(' (tent.)' if r.tentative else '')
    for i,(name,m) in enumerate([('Single-L',o['single']),('Two-L',o['two']),('Two-L + $\\Phi_\\mathrm{BH}$',o['twophi'])]):
        p,e=m['params'],m['errors']; star='$^\\star$' if name==P else ''
        if name=='Single-L': cells=[pm(p[1],e[1]),pm(p[0],e[0]),'--','--','--']
        else: cells=[pm(p[1],e[1]),pm(p[0],e[0]),pm(p[3],e[3]),pm(p[2],e[2]),((f"${p[4]:.0f}\\pm{e[4]:.0f}$" if not np.isnan(e[4]) else f"${p[4]:.0f}$ (unc.)") if len(p)==5 else '--')]
        last=(f"{r.jump:+.2f} {r.flag}" if name=='Two-L' else '')
        head=(f"{nm} & {TY[r['T']]} & {r.n}" if i==0 else " & & ")
        A(f"{head} & {name}{star} & "+' & '.join(cells)+f" & {last}\\\\")
    A(r"\midrule" if g!=order[-1] else '')
A(r"\end{longtable}}\clearpage")
A(r"""{\scriptsize\setlength{\tabcolsep}{4pt}
\begin{longtable}{llrrrrrrrrr}
\caption{Accepted two-L galaxies: goodness of fit. AIC $=\chi^2+2k$, BIC $=\chi^2+k\ln n$; $\Delta$ relative to the best model of the galaxy; $p_\mathrm{runs}$ = runs test on the residuals of that model (small = residual structure remains).}\\\toprule
Galaxy & Model & $k$ & $\chi^2$ & $\chi^2_\nu$ & AIC & BIC & $\Delta$AIC & $\Delta$BIC & RMS$_\mathrm{rel}$ & $p_\mathrm{runs}$\\\midrule\endfirsthead
\toprule Galaxy & Model & $k$ & $\chi^2$ & $\chi^2_\nu$ & AIC & BIC & $\Delta$AIC & $\Delta$BIC & RMS$_\mathrm{rel}$ & $p_\mathrm{runs}$\\\midrule\endhead\bottomrule\endfoot""")
from wave_detect import runs_p
from cl_fit import v2_model, load_massmodels
mm=load_massmodels()
for g in order:
    o=F[g]; d=mm[mm.ID==g]; rr,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev
    ms=[('Single-L',o['single'],o['z1'])]
    q=o['two']['params']; ms.append(('Two-L',o['two'],(y-v2_two(rr,q[0]*1e9,q[1],q[2]*1e9,q[3]))/s))
    q=o['twophi']['params']; ms.append(('Two-L + $\\Phi_\\mathrm{BH}$',o['twophi'],(y-v2_two(rr,q[0]*1e9,q[1],q[2]*1e9,q[3],q[4]))/s))
    aic=min(m['AIC'] for _,m,_ in ms); bic=min(m['BIC'] for _,m,_ in ms); P=pref(g)
    for i,(name,m,z) in enumerate(ms):
        p=runs_p(z); star='$^\\star$' if name==P else ''
        A(f"{gn(g) if i==0 else ''} & {name}{star} & {m['k']} & {m['chi2']:.2f} & {m['chi2nu']:.3f} & {m['AIC']:.2f} & {m['BIC']:.2f} & {m['AIC']-aic:.2f} & {m['BIC']-bic:.2f} & {m['RMSrel']:.3f} & {('%.1e'%p) if p<0.001 else '%.3f'%p}\\\\")
    A(r"\midrule" if g!=order[-1] else '')
A(r"\end{longtable}}")
A(r"\clearpage\section*{Atlas 1: single-L fits with residuals (all 175, grouped by category)}")
A(r"\includepdf[pages=-,fitpaper=false,scale=0.95]{single_L_atlas.pdf}")
A(r"\section*{Atlas 2: accepted two-L fits with residuals}")
A(r"\includepdf[pages=-,scale=0.95]{double_L_atlas.pdf}")
A(r"\end{document}")
open('CL_pipeline_SPARC.tex','w').write('\n'.join(L))
