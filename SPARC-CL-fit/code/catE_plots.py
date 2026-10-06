import sys,pickle,numpy as np,pandas as pd; sys.path.insert(0,'.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from catD import MODELS
from cl_fit import load_massmodels
from wave_detect import runs_p
mm=load_massmodels(); F=pickle.load(open('catE_fits.pkl','rb')); df=pd.read_csv('catE_results.csv'); D=df.set_index('Galaxy')
KC={'rising window (B-like)':'tab:green','descent window (C-like)':'tab:purple','two windows (D-like)':'tab:blue','single-L kept':'0.45'}
plt.rcParams.update({'font.size':8})
order=df.assign(o=df.kind.map({'rising window (B-like)':0,'descent window (C-like)':1,'two windows (D-like)':2,'single-L kept':3})).sort_values(['o','dBIC_best']).Galaxy.tolist()
fig,ax=plt.subplots(1,3,figsize=(15,4.2),gridspec_kw={'width_ratios':[2,1,1]})
ks=['VW','2L','VW2']; cols=['tab:purple','tab:green','tab:blue']; w=0.27; xx=np.arange(len(order))
for i,(k,c) in enumerate(zip(ks,cols)):
    vals=[D.loc[g,f'BIC_{k}']-D.loc[g,'BIC_S'] for g in order]; ax[0].bar(xx+(i-1)*w,np.nan_to_num(vals),w,color=c,label=k)
ax[0].set_yscale('symlog',linthresh=10); ax[0].axhline(-6,color='r',ls='--',lw=.8); ax[0].axhline(0,color='k',lw=.6)
ax[0].set_xticks(xx); ax[0].set_xticklabels(order,rotation=60,ha='right',fontsize=7)
for lab,g in zip(ax[0].get_xticklabels(),order): lab.set_color(KC[D.loc[g,'kind']])
ax[0].set_ylabel('$\\Delta$BIC relative to single-L'); ax[0].legend(fontsize=7,ncol=3); ax[0].set_title('Model comparison (label colour = preferred description)')
n=np.arange(4,31); mp=[min(runs_p(np.r_[np.ones(k),-np.ones(m-k)]) for k in range(1,m)) for m in n]
ax[1].semilogy(n,mp,'k-',label='best attainable $p_{runs}$'); ax[1].axhline(0.05,color='r',ls='--',lw=.8,label='0.05')
ax[1].hist(df.n,bins=np.arange(3.5,26,1),weights=np.full(len(df),1e-6),alpha=0)  # keep axis
for g in df.Galaxy: ax[1].plot(D.loc[g,'n'],D.loc[g,'p_runs1'],'o',color=KC[D.loc[g,'kind']],ms=4)
ax[1].set_xlabel('n points'); ax[1].set_ylabel('$p_{runs}$ of single-L residuals'); ax[1].legend(fontsize=6.5); ax[1].set_title('Power of the runs test')
for k,c in KC.items():
    s=df[df.kind==k]; ax[2].scatter(s.chi2nu_S,s.chi2nu_best,c=c,s=25,label=k)
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].plot([.1,20],[.1,20],'k:',lw=.8); ax[2].axhline(1,color='0.7',lw=.6)
ax[2].set_xlabel('$\\chi^2_\\nu$ single-L'); ax[2].set_ylabel('$\\chi^2_\\nu$ preferred'); ax[2].legend(fontsize=6)
fig.tight_layout(); fig.savefig('catE_fig_models.pdf'); plt.close(fig)
NM={'S':'single-L','VW':'CL + virial window','2L':'two-L','VW2':'CL + two windows'}
with PdfPages('catE_atlas.pdf') as pdf:
    for p in range(0,len(order),9):
        fig=plt.figure(figsize=(16,12)); gs=GridSpec(3,3,figure=fig,hspace=0.32,wspace=0.25)
        for k,g in enumerate(order[p:p+9]):
            sub=GridSpecFromSubplotSpec(2,1,gs[k],height_ratios=[3,1.3],hspace=0.05); a=fig.add_subplot(sub[0]); b=fig.add_subplot(sub[1],sharex=a)
            d=mm[mm.ID==g]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev; row=D.loc[g]; rr=np.linspace(1e-2,r.max()*1.04,2000)
            a.errorbar(r,y,yerr=s,fmt='o',ms=3,color='crimson',ecolor='0.6',lw=.7,zorder=3)
            a.plot(rr,MODELS['S'][0](rr,F[g]['S']['x']),':',color='0.45',lw=1.2,label=f"single-L $\\chi^2_\\nu$={row.chi2nu_S:.2f}")
            if row.best!='S':
                m=row.best; a.plot(rr,MODELS[m][0](rr,F[g][m]['x']),'k-',lw=1.4,label=f"{NM[m]} $\\chi^2_\\nu$={row.chi2nu_best:.2f}")
                if '2L' in F[g] and m!='2L':
                    yy=MODELS['2L'][0](rr,F[g]['2L']['x']); a.plot(rr,yy,'--',color='tab:green',lw=.9,alpha=.8,label=f"two-L $\\chi^2_\\nu$={row.chi2nu_2L:.2f}")
                for kv in row.radii.split(';')[1:]:
                    nm_,val=kv.split('='); 
                    for ax_ in (a,b): ax_.axvline(float(val),color='tab:blue',ls='--',lw=.7)
            a.set_ylim(0,y.max()*1.3)
            a.set_title(f"{g} [{row.kind}; {row.status}]  n={row.n}  $\\Delta$BIC={row.dBIC_best:.1f}",fontsize=7.5,weight='bold',color=KC[row.kind])
            a.legend(fontsize=5.8,loc='lower right',frameon=False); a.tick_params(labelbottom=False); a.set_ylabel('$v^2$ (km$^2$s$^{-2}$)',fontsize=6.5)
            zS=F[g]['S']['z']; zb=F[g][row.best]['z']
            b.axhline(0,color='k',lw=.7); b.axhline(1,color='0.6',ls='--',lw=.6); b.axhline(-1,color='0.6',ls='--',lw=.6)
            b.plot(r,zS,'o',mfc='none',mec='0.55',ms=3); b.scatter(r,zb,c=np.where(zb>0,'crimson','royalblue'),s=10,zorder=3)
            lim=max(1.5,np.abs(np.r_[zS,zb]).max()*1.1); b.set_ylim(-lim,lim); b.set_ylabel('WR',fontsize=6.5); b.set_xlabel('r (kpc)',fontsize=6.5)
            b.text(.02,.05,f"$p_{{runs}}$: {row.p_runs1:.3f} (min. attainable {row.min_runs_p:.3f}); $\\sigma_{{int}}$={row.sigma_int:.1f} km/s",transform=b.transAxes,fontsize=6)
        fig.suptitle("Category E: single-L (dotted), preferred model (solid; blue dashed = window radii), two-L (green dashed). Residuals: single-L open, preferred filled.",fontsize=9.5)
        pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
