import sys, pickle, numpy as np, pandas as pd; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from cl_fit import v2_model, load_massmodels
from cl2_fit import v2_two
mm=load_massmodels(); F=pickle.load(open('pipeline_fits.pkl','rb')); df=pd.read_csv('pipeline_summary.csv').set_index('Galaxy')
CAT={'A':'tab:green','B':'tab:orange','C':'tab:purple','D':'tab:brown','E':'tab:red'}
plt.rcParams.update({'font.size':7})
def resid_axis(ax, r, z, col_pos='crimson', col_neg='royalblue', mk='o', ms=3, alpha=1, label=None, filled=True):
    c=np.where(z>0,col_pos,col_neg)
    ax.scatter(r,z,c=c if filled else 'none',edgecolors=c,s=ms**2*2,marker=mk,alpha=alpha,lw=0.8,label=label,zorder=3)
def single_atlas(order, fname):
    with PdfPages(fname) as pdf:
        for p in range(0,len(order),12):
            fig=plt.figure(figsize=(16,11)); gs=GridSpec(3,4,figure=fig,hspace=0.42,wspace=0.28)
            for k,g in enumerate(order[p:p+12]):
                sub=GridSpecFromSubplotSpec(2,1,gs[k],height_ratios=[3,1.25],hspace=0.05)
                a=fig.add_subplot(sub[0]); b=fig.add_subplot(sub[1],sharex=a)
                d=mm[mm.ID==g]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values; s1=F[g]['single']; z=F[g]['z1']; row=df.loc[g]
                rr=np.linspace(1e-3,r.max()*1.04,600)
                a.errorbar(r,v**2,yerr=2*v*ev,fmt='o',ms=2.8,color='crimson',ecolor='0.6',lw=0.7,zorder=3)
                a.plot(rr,v2_model(rr,s1['params'][0]*1e9,s1['params'][1]),'k-',lw=1.2)
                a.axvline(s1['params'][1],color='k',ls=':',lw=0.7)
                cat=row.category[0]
                a.set_title(f"{g}   [{cat}]   n={row.n}",fontsize=8.5,weight='bold',color=CAT[cat])
                a.text(0.03,0.95,f"M={s1['params'][0]:.3g}e9 M$_\\odot$, R={s1['params'][1]:.2f} kpc\n$\\chi^2_\\nu$={s1['chi2nu']:.2f}, RMS$_{{rel}}$={s1['RMSrel']:.2f}",
                       transform=a.transAxes,va='top',fontsize=6.5)
                a.set_ylim(bottom=0); a.tick_params(labelbottom=False); a.set_ylabel('$v^2$ (km$^2$s$^{-2}$)',fontsize=6.5)
                b.axhline(0,color='k',lw=0.7); b.axhline(1,color='0.6',ls='--',lw=0.6); b.axhline(-1,color='0.6',ls='--',lw=0.6)
                resid_axis(b,r,z); b.set_ylabel('WR',fontsize=6.5); b.set_xlabel('r (kpc)',fontsize=6.5)
                lim=max(1.5,np.abs(z).max()*1.15); b.set_ylim(-lim,lim)
                if row.category[0]=='B' and not np.isnan(row.r_cross1): b.axvline(row.r_cross1,color='tab:orange',lw=0.8)
            fig.suptitle("Step 1-2: single-L CL fits and weighted residuals WR=(v$^2_{obs}$-v$^2_{mod}$)/$\\sigma_{v^2}$  "
                         "[A adequate, B +-+ wave, C -+- wave, D structured other, E poor unstructured]; orange line: first crossover",fontsize=10)
            pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
def double_atlas(order, fname):
    with PdfPages(fname) as pdf:
        for p in range(0,len(order),8):
            fig=plt.figure(figsize=(16,11)); gs=GridSpec(2,4,figure=fig,hspace=0.3,wspace=0.28)
            for k,g in enumerate(order[p:p+8]):
                sub=GridSpecFromSubplotSpec(2,1,gs[k],height_ratios=[3,1.4],hspace=0.05)
                a=fig.add_subplot(sub[0]); b=fig.add_subplot(sub[1],sharex=a)
                d=mm[mm.ID==g]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev
                f=F[g]; s1,t2,t3=f['single'],f['two'],f['twophi']; row=df.loc[g]
                rr=np.linspace(1e-3,r.max()*1.04,1500)
                a.errorbar(r,y,yerr=s,fmt='o',ms=2.8,color='crimson',ecolor='0.6',lw=0.7,zorder=3)
                a.plot(rr,v2_model(rr,s1['params'][0]*1e9,s1['params'][1]),':',color='0.45',lw=1.2,label=f"single-L $\\chi^2_\\nu$={s1['chi2nu']:.2f}")
                q=t2['params']; yy=v2_two(rr,q[0]*1e9,q[1],q[2]*1e9,q[3]); yy[np.abs(rr-q[3])<(rr[1]-rr[0])]=np.nan
                a.plot(rr,yy,'k-',lw=1.4,label=f"two-L $\\chi^2_\\nu$={t2['chi2nu']:.2f}")
                q5=t3['params']; a.plot(rr,v2_two(rr,q5[0]*1e9,q5[1],q5[2]*1e9,q5[3],q5[4]),'--',color='tab:blue',lw=1,label=f"two-L+$\\Phi_{{BH}}$ $\\chi^2_\\nu$={t3['chi2nu']:.2f}")
                for ax in (a,b): ax.axvline(q[1],color='k',ls=':',lw=0.7); ax.axvline(q[3],color='k',ls='--',lw=0.7)
                tag=' (tentative, n$\\leq$13)' if row.tentative else ''
                a.set_title(f"{g}   n={row.n}   $\\Delta$BIC={row.dBIC:.1f}{tag}",fontsize=8.5,weight='bold')
                a.legend(fontsize=6,loc='lower right',frameon=False); a.set_ylim(bottom=0); a.tick_params(labelbottom=False)
                a.set_ylabel('$v^2$ (km$^2$s$^{-2}$)',fontsize=6.5)
                z2=(y-v2_two(r,q[0]*1e9,q[1],q[2]*1e9,q[3]))/s
                b.axhline(0,color='k',lw=0.7); b.axhline(1,color='0.6',ls='--',lw=0.6); b.axhline(-1,color='0.6',ls='--',lw=0.6)
                b.plot(r,f['z1'],'o',mfc='none',mec='0.55',ms=3.2,lw=0,label='single-L')
                resid_axis(b,r,z2,ms=2.6,label=None); b.plot([],[],'o',color='k',ms=3,label='two-L (red +, blue -)')
                lim=max(1.5,np.abs(np.r_[f['z1'],z2]).max()*1.15); b.set_ylim(-lim,lim)
                b.legend(fontsize=5.5,loc='lower right',frameon=False,ncol=2); b.set_ylabel('WR',fontsize=6.5); b.set_xlabel('r (kpc)',fontsize=6.5)
            fig.suptitle("Steps 3-5: accepted two-Lagrangian fits (Eqs. 24-26), R$_1$ dotted, R$_2$ dashed; lower panels: weighted residuals of single-L (open grey) and two-L (filled)",fontsize=10)
            pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
if __name__=='__main__':
    order=df.sort_values(['category','chi2nu1']).index.tolist()
    single_atlas(order,'single_L_atlas.pdf')
    acc=df[df.accepted==True].sort_values('dBIC').index.tolist()
    double_atlas(acc,'double_L_atlas.pdf')
