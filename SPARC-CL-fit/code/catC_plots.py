import sys,pickle,numpy as np,pandas as pd; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from catC_model import mVW
from cl_fit import v2_model, load_massmodels
from cl2_fit import v2_two
mm=load_massmodels(); F=pickle.load(open('catC_fits.pkl','rb')); PF=pickle.load(open('pipeline_fits.pkl','rb')); df=pd.read_csv('catC_results.csv'); D=df.set_index('Galaxy')
OC={'descent window':'tab:purple','flattening window':'tab:cyan','rising window':'tab:olive','single-L kept':'0.45'}
plt.rcParams.update({'font.size':8})
fig,ax=plt.subplots(1,3,figsize=(13,3.9))
for oc,c in OC.items():
    s=df[df.outcome==oc]
    ax[0].scatter(s.DW_S,s.DW_VW,c=c,s=20,label=f'{oc} ({len(s)})'); ax[1].scatter(s.RMS_S,s.RMS_VW,c=c,s=20); ax[2].scatter(s.n,s.dBIC_VW.clip(upper=10),c=c,s=20)
ax[0].plot([0,2.5],[0,2.5],'k:',lw=.8); ax[0].axhline(2,color='0.7',lw=.6); ax[0].set_xlabel('Durbin-Watson, single-L'); ax[0].set_ylabel('Durbin-Watson, CL + virial window'); ax[0].legend(fontsize=6.5)
ax[1].plot([0,.5],[0,.5],'k:',lw=.8); ax[1].set_xlabel('RMS$_{rel}$ single-L'); ax[1].set_ylabel('RMS$_{rel}$ CL + VW')
ax[2].set_yscale('symlog',linthresh=10); ax[2].set_xscale('log'); ax[2].axhline(-6,color='r',ls='--',lw=.8); ax[2].set_xlabel('n points'); ax[2].set_ylabel('$\\Delta$BIC (VW - single-L)')
fig.tight_layout(); fig.savefig('catC_fig_quality.pdf'); plt.close(fig)
d=df[df.outcome=='descent window'].sort_values('grade'); gl=d.Galaxy.tolist()
fig,axs=plt.subplots(4,7,figsize=(16,9)); axs=axs.ravel()
for a,g in zip(axs,gl):
    P=F[g]['prof']; dd=P[:,1]-P[:,1].min(); a.plot(P[:,0],dd,'k-',lw=1); a.axhline(1,color='r',ls='--',lw=.7); a.axhline(4,color='orange',ls=':',lw=.7)
    for x in mm[mm.ID==g].R.values: a.axvline(x,color='0.88',lw=.4,zorder=0)
    a.set_ylim(0,min(50,max(dd.max(),5))); a.set_title(f"{g} [{D.loc[g,'grade']}]",fontsize=7.5); a.tick_params(labelsize=6)
for a in axs[len(gl):]: a.axis('off')
fig.suptitle('Profile $\\Delta\\chi^2(r_v)$ for the descent-window galaxies (other parameters refitted); red $\\Delta\\chi^2=1$, orange 4; grey: data radii',fontsize=10)
fig.tight_layout(); fig.savefig('catC_fig_profiles.pdf'); plt.close(fig)
fig,ax=plt.subplots(1,4,figsize=(15,3.8)); c=[{'I':'tab:green','I':'tab:green'}.get(x[0] if x!='--' else '','tab:orange') if x[0]=='I' and not x.startswith('II') else ('tab:orange' if x.startswith('II') and not x.startswith('III') else 'tab:red') for x in d.grade]
ax[0].scatter(d.Rdisk,d.rv,c=c,s=20); ax[0].plot([0.5,10],[0.5,10],'k:',lw=.8); ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('$R_{disk}$ (kpc)'); ax[0].set_ylabel('$r_v$ (kpc)')
ok=d.Vflat>0; ax[1].errorbar(d.Vflat[ok],np.sqrt(d.A[ok]),yerr=0.5*d.eA[ok]/np.sqrt(d.A[ok]),fmt='o',ms=3.5,color='k',lw=.7); ax[1].plot([50,320],[50,320],'r--',lw=.8)
ax[1].set_xlabel('SPARC $V_{flat}$ (km/s)'); ax[1].set_ylabel('$\\sqrt{A}$ (km/s), floor of descent')
ax[2].scatter(d.Mbar,d.MN,c=c,s=20); ax[2].plot([1,500],[1,500],'k:',lw=.8); ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel('$M_{bar}$ (10$^9$M$_\\odot$)'); ax[2].set_ylabel('$M_N=(p-2)M$')
ax[3].hist(np.log10(d.p),bins=15,color='tab:purple'); ax[3].axvline(np.log10(3),color='k',ls='--',lw=.8,label='p=3 (Kepler)'); ax[3].set_xlabel('log$_{10}$ p'); ax[3].legend(fontsize=7)
fig.tight_layout(); fig.savefig('catC_fig_physics.pdf'); plt.close(fig)
order=df.assign(o=df.outcome.map({'descent window':0,'flattening window':1,'rising window':2,'single-L kept':3})).sort_values(['o','grade','dBIC_VW']).Galaxy.tolist()
with PdfPages('catC_atlas.pdf') as pdf:
    for p in range(0,len(order),9):
        fig=plt.figure(figsize=(16,12)); gs=GridSpec(3,3,figure=fig,hspace=0.32,wspace=0.25)
        for k,g in enumerate(order[p:p+9]):
            sub=GridSpecFromSubplotSpec(2,1,gs[k],height_ratios=[3,1.3],hspace=0.05); a=fig.add_subplot(sub[0]); b=fig.add_subplot(sub[1],sharex=a)
            dd=mm[mm.ID==g]; r,v,ev=dd.R.values,dd.Vobs.values,dd.eV.values; y,s=v**2,2*v*ev; row=D.loc[g]; f=F[g]
            rr=np.linspace(1e-2,r.max()*1.04,1500)
            a.errorbar(r,y,yerr=s,fmt='o',ms=2.8,color='crimson',ecolor='0.6',lw=.7,zorder=3)
            x=PF[g]['single']['params']; a.plot(rr,v2_model(rr,x[0]*1e9,x[1]),':',color='0.45',lw=1.2,label=f"single-L $\\chi^2_\\nu$={row.chi2nu_S:.2f}")
            a.plot(rr,mVW(rr,f['x']),'k-',lw=1.4,label=f"CL + virial window $\\chi^2_\\nu$={row.chi2nu_VW:.2f}")
            q=f['two']['params']; yy=v2_two(rr,q[0]*1e9,q[1],q[2]*1e9,q[3]); yy[np.abs(rr-q[3])<(rr[1]-rr[0])]=np.nan
            a.plot(rr,yy,'--',color='tab:green',lw=.9,alpha=.8,label=f"two-L $\\chi^2_\\nu$={row.chi2nu_2L:.2f}")
            a.axvspan(row.rv_lo,row.rv_hi,color=OC[row.outcome],alpha=.15,lw=0)
            for ax_ in (a,b): ax_.axvline(row.rv,color=OC[row.outcome],ls='--',lw=.8)
            a.set_ylim(0,y.max()*1.3)
            a.set_title(f"{g} [{row.grade if row.grade!='--' else row.outcome}]  n={row.n}  $r_v$={row.rv:.1f}, p={row.p:.3g}, $\\Delta$BIC={row.dBIC_VW:.1f}",fontsize=7.8,weight='bold',color=OC[row.outcome])
            a.legend(fontsize=5.8,loc='lower right',frameon=False); a.tick_params(labelbottom=False); a.set_ylabel('$v^2$ (km$^2$s$^{-2}$)',fontsize=6.5)
            b.axhline(0,color='k',lw=.7); b.axhline(1,color='0.6',ls='--',lw=.6); b.axhline(-1,color='0.6',ls='--',lw=.6)
            b.plot(r,f['zS'],'o',mfc='none',mec='0.55',ms=3); z=f['zV']; b.scatter(r,z,c=np.where(z>0,'crimson','royalblue'),s=9,zorder=3)
            lim=max(1.5,np.abs(np.r_[f['zS'],z]).max()*1.1); b.set_ylim(-lim,lim); b.set_ylabel('WR',fontsize=6.5); b.set_xlabel('r (kpc)',fontsize=6.5)
            b.text(.02,.05,f"$p_{{runs}}$: {row.p_runs1:.3f} $\\to$ {row.pruns_VW:.3f}",transform=b.transAxes,fontsize=6)
        fig.suptitle("Category C: single-L (dotted), CL + virial window (solid; dashed line $r_v$, band = profile $\\Delta\\chi^2\\leq1$), two-L (green dashed). Residuals: single-L open, CL + VW filled.",fontsize=9.5)
        pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
