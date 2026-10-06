import sys,pickle,numpy as np,pandas as pd; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from cl_fit import v2_model, load_massmodels
from cl2_fit import v2_two
mm=load_massmodels(); PF=pickle.load(open('pipeline_fits.pkl','rb')); df=pd.read_csv('catB_results.csv'); prof=pickle.load(open('catB_profiles.pkl','rb'))
D=df.set_index('Galaxy')
OC={'clean nested two-L':'tab:green','two-L, needs more regions':'tab:orange','not nested':'tab:red','single-L kept':'0.4'}
plt.rcParams.update({'font.size':8})
# Fig 1: residual structure before/after
fig,ax=plt.subplots(1,3,figsize=(13,3.9))
for oc,c in OC.items():
    s=df[df.outcome==oc]
    ax[0].scatter(s.DW1,s.DW2,c=c,s=20,label=f'{oc} ({len(s)})')
    ax[1].scatter(s.RMS1,s.RMS2,c=c,s=20)
    ax[2].scatter(s.n,s.dBIC.clip(upper=10),c=c,s=20)
ax[0].plot([0,3],[0,3],'k:',lw=0.8); ax[0].axhline(2,color='0.7',lw=0.6); ax[0].set_xlabel('Durbin-Watson, single-L'); ax[0].set_ylabel('Durbin-Watson, two-L'); ax[0].legend(fontsize=6.5); ax[0].set_title('Residual autocorrelation removed')
ax[1].plot([0,0.7],[0,0.7],'k:',lw=0.8); ax[1].set_xlabel('RMS$_{rel}$ single-L'); ax[1].set_ylabel('RMS$_{rel}$ two-L'); ax[1].set_title('Relative scatter in $v^2$')
ax[2].set_yscale('symlog',linthresh=10); ax[2].axhline(-6,color='r',ls='--',lw=0.8); ax[2].axvline(13.5,color='0.6',ls=':',lw=0.8)
ax[2].set_xlabel('n points'); ax[2].set_ylabel('$\\Delta$BIC (two-L - single-L)'); ax[2].set_title('Acceptance: $\\Delta$BIC<-6; n$\\leq$13 tentative'); ax[2].set_xscale('log')
fig.tight_layout(); fig.savefig('catB_fig_quality.pdf'); plt.close(fig)
# Fig 2: R2 profiles for clean galaxies
cl=df[df.outcome=='clean nested two-L'].sort_values('grade').Galaxy.tolist()
fig,axs=plt.subplots(3,5,figsize=(14,7.5)); axs=axs.ravel()
for a,g in zip(axs,cl):
    P=prof[g]; d=P[:,1]-P[:,1].min(); a.plot(P[:,0],d,'k-',lw=1); a.axhline(1,color='r',ls='--',lw=0.7); a.axhline(4,color='orange',ls=':',lw=0.7)
    r=mm[mm.ID==g].R.values
    for x in r: a.axvline(x,color='0.85',lw=0.5,zorder=0)
    a.set_ylim(0,min(60,d.max())); a.set_title(f"{g} [{D.loc[g,'grade']}]",fontsize=8); a.set_xlabel('$R_2$ (kpc)',fontsize=7)
for a in axs[len(cl):]: a.axis('off')
fig.suptitle('Profile $\\Delta\\chi^2(R_2)$ for the clean nested galaxies (other parameters refitted); red: $\\Delta\\chi^2=1$, orange: 4; grey: data radii',fontsize=10)
fig.tight_layout(); fig.savefig('catB_fig_profiles.pdf'); plt.close(fig)
# Fig 3: physics
c=df[df.outcome=='clean nested two-L']; a2=df[df.accepted]
fig,ax=plt.subplots(1,4,figsize=(15,3.8))
for oc,col in list(OC.items())[:3]:
    s=a2[a2.outcome==oc]
    ax[0].scatter(s.r_cross1,s.R2,c=col,s=20,label=oc); ax[1].scatter(s.Rdisk,s.R2,c=col,s=20)
    ok=s.Vflat>0; ax[2].scatter(s.Vflat[ok],s.vL2[ok],c=col,s=20); ax[3].scatter(s.Mbar,s.M2,c=col,s=20)
ax[0].plot([0.5,25],[0.5,25],'k:',lw=0.8); ax[0].plot([0.5,25],[0.5/0.8,25/0.8],'b--',lw=0.7,label='$R_2=r_{\\times1}/0.8$ (seed)'); ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('first crossover $r_{\\times1}$ (kpc)'); ax[0].set_ylabel('$R_2$ (kpc)'); ax[0].legend(fontsize=6)
ax[1].plot([0.3,10],[0.3,10],'k:',lw=0.8); ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('$R_{disk}$ (kpc)'); ax[1].set_ylabel('$R_2$ (kpc)')
ax[2].plot([20,250],[20,250],'k:',lw=0.8); ax[2].set_xlabel('SPARC $V_{flat}$'); ax[2].set_ylabel('$v_{L2}$ (km/s)')
ax[3].plot([0.05,100],[0.05,100],'k:',lw=0.8); ax[3].set_xscale('log'); ax[3].set_yscale('log'); ax[3].set_xlabel('$M_{bar}$ (10$^9$M$_\\odot$)'); ax[3].set_ylabel('$M_2$')
fig.tight_layout(); fig.savefig('catB_fig_physics.pdf'); plt.close(fig)

def resid(ax,r,z,ms=10): ax.scatter(r,z,c=np.where(z>0,'crimson','royalblue'),s=ms,zorder=3)
# Atlas accepted
order=a2.assign(o=a2.outcome.map({'clean nested two-L':0,'two-L, needs more regions':1,'not nested':2})).sort_values(['o','grade','dBIC']).Galaxy.tolist()
with PdfPages('catB_atlas_accepted.pdf') as pdf:
    for p in range(0,len(order),8):
        fig=plt.figure(figsize=(16,11)); gs=GridSpec(2,4,figure=fig,hspace=0.3,wspace=0.28)
        for k,g in enumerate(order[p:p+8]):
            sub=GridSpecFromSubplotSpec(2,1,gs[k],height_ratios=[3,1.4],hspace=0.05); a=fig.add_subplot(sub[0]); b=fig.add_subplot(sub[1],sharex=a)
            d=mm[mm.ID==g]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev; f=PF[g]; row=D.loc[g]
            rr=np.linspace(1e-3,r.max()*1.04,1500)
            a.errorbar(r,y,yerr=s,fmt='o',ms=2.8,color='crimson',ecolor='0.6',lw=0.7,zorder=3)
            a.plot(rr,v2_model(rr,f['single']['params'][0]*1e9,f['single']['params'][1]),':',color='0.45',lw=1.2,label=f"single-L $\\chi^2_\\nu$={row.chi2nu1:.2f}")
            q=f['two']['params']; yy=v2_two(rr,q[0]*1e9,q[1],q[2]*1e9,q[3]); yy[np.abs(rr-q[3])<(rr[1]-rr[0])]=np.nan
            a.plot(rr,yy,'k-',lw=1.4,label=f"two-L $\\chi^2_\\nu$={row.chi2nu2:.2f}")
            q5=f['twophi']['params']; a.plot(rr,v2_two(rr,q5[0]*1e9,q5[1],q5[2]*1e9,q5[3],q5[4]),'--',color='tab:blue',lw=1,label=f"two-L+$\\Phi_{{BH}}$ $\\chi^2_\\nu$={row.chi2nu3:.2f}")
            a.axvspan(row.R2_lo,row.R2_hi,color='0.85',zorder=0)
            for ax_ in (a,b): ax_.axvline(q[1],color='k',ls=':',lw=0.7); ax_.axvline(q[3],color='k',ls='--',lw=0.7)
            tag=' tent.' if row.tentative else ''
            a.set_title(f"{g} [{row.grade}{tag}]  n={row.n}  $\\Delta$BIC={row.dBIC:.1f}",fontsize=8.5,weight='bold',color=OC[row.outcome])
            a.legend(fontsize=6,loc='lower right',frameon=False); a.set_ylim(bottom=0); a.tick_params(labelbottom=False); a.set_ylabel('$v^2$ (km$^2$s$^{-2}$)',fontsize=6.5)
            z1,z2,_=pickle.load(open('catB_resid.pkl','rb'))[g]
            b.axhline(0,color='k',lw=0.7); b.axhline(1,color='0.6',ls='--',lw=0.6); b.axhline(-1,color='0.6',ls='--',lw=0.6)
            b.plot(r,z1,'o',mfc='none',mec='0.55',ms=3.2,label='single-L'); resid(b,r,z2,9)
            lim=max(1.5,np.abs(np.r_[z1,z2]).max()*1.15); b.set_ylim(-lim,lim); b.set_ylabel('WR',fontsize=6.5); b.set_xlabel('r (kpc)',fontsize=6.5)
            b.text(0.02,0.05,f"$p_{{runs}}$: {row.p_runs1:.3f} $\\to$ {row.p_runs2:.3f}",transform=b.transAxes,fontsize=6)
        fig.suptitle("Category B, accepted two-L fits (green clean nested, orange needs more regions, red not nested). Shaded: profile $\\Delta\\chi^2\\leq1$ interval of $R_2$. Residuals: single-L open grey, two-L filled.",fontsize=9.5)
        pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
# Atlas rejected
rj=df[~df.accepted].sort_values('dBIC').Galaxy.tolist()
with PdfPages('catB_atlas_rejected.pdf') as pdf:
    for p in range(0,len(rj),12):
        fig=plt.figure(figsize=(16,11)); gs=GridSpec(3,4,figure=fig,hspace=0.42,wspace=0.28)
        for k,g in enumerate(rj[p:p+12]):
            sub=GridSpecFromSubplotSpec(2,1,gs[k],height_ratios=[3,1.3],hspace=0.05); a=fig.add_subplot(sub[0]); b=fig.add_subplot(sub[1],sharex=a)
            d=mm[mm.ID==g]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev; f=PF[g]; row=D.loc[g]
            rr=np.linspace(1e-3,r.max()*1.04,1500)
            a.errorbar(r,y,yerr=s,fmt='o',ms=2.8,color='crimson',ecolor='0.6',lw=0.7,zorder=3)
            a.plot(rr,v2_model(rr,f['single']['params'][0]*1e9,f['single']['params'][1]),'k-',lw=1.3,label=f"single-L $\\chi^2_\\nu$={row.chi2nu1:.2f}")
            q=f['two']['params']; yy=v2_two(rr,q[0]*1e9,q[1],q[2]*1e9,q[3]); yy[np.abs(rr-q[3])<(rr[1]-rr[0])]=np.nan
            a.plot(rr,yy,'--',color='tab:green',lw=1,label=f"two-L (rejected) $\\chi^2_\\nu$={row.chi2nu2:.2f}")
            a.set_title(f"{g}  n={row.n}  $\\Delta$BIC={row.dBIC:+.1f}"+(" $\\dagger$" if row.double_2018 else ''),fontsize=8.5,weight='bold')
            a.legend(fontsize=6,loc='lower right',frameon=False); a.set_ylim(bottom=0); a.tick_params(labelbottom=False); a.set_ylabel('$v^2$',fontsize=6.5)
            z1=pickle.load(open('catB_resid.pkl','rb'))[g][0]
            b.axhline(0,color='k',lw=0.7); b.axhline(1,color='0.6',ls='--',lw=0.6); b.axhline(-1,color='0.6',ls='--',lw=0.6); resid(b,r,z1)
            b.axvline(row.r_cross1,color='tab:orange',lw=0.8); lim=max(1.5,np.abs(z1).max()*1.15); b.set_ylim(-lim,lim); b.set_ylabel('WR',fontsize=6.5); b.set_xlabel('r (kpc)',fontsize=6.5)
            b.text(0.02,0.05,f"$p_{{runs}}$={row.p_runs1:.3f}  A$_{{+-+}}$={row.A_pmp:.2f}",transform=b.transAxes,fontsize=6)
        fig.suptitle("Category B, two-L not accepted ($\\Delta$BIC$\\geq-6$): single-L fit and its residuals (orange: first crossover); dashed: best two-L for comparison. $\\dagger$ 2018 double fit.",fontsize=9.5)
        pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
