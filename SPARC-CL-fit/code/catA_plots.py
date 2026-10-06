import sys,pickle,numpy as np,pandas as pd; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats
from cl_fit import v2_model, load_massmodels, G, HZ
mm=load_massmodels(); F=pickle.load(open('catA_fits.pkl','rb')); df=pd.read_csv('catA_results.csv')
def grade(r):
    if r.n>=8 and r.n_in>=2 and r.n_out>=3 and r.relM<=0.3 and r.relR<=0.3 and max(r.jk_shift_M,r.jk_shift_R)<=2: return 'I'
    if r.relV<=0.10: return 'II'
    return 'III'
df['grade']=df.apply(grade,axis=1); df['err_over']=df.p_chi2_low<0.01
df.to_csv('catA_results.csv',index=False,float_format='%.5g')
GC={'I':'tab:green','II':'tab:orange','III':'tab:red'}
plt.rcParams.update({'font.size':8})
# ---- diagnostics figures
Z=np.concatenate([F[g]['z'] for g in df.Galaxy]); Zs=np.concatenate([F[g]['z']/np.sqrt(max(F[g]['chi2nu'],1e-3)) for g in df.Galaxy])
fig,ax=plt.subplots(1,3,figsize=(13,3.8))
x=np.linspace(-4,4,300)
ax[0].hist(Z,bins=30,range=(-3,3),density=True,color='0.6',label=f'WR (raw), sd={Z.std():.2f}')
ax[0].hist(Zs,bins=30,range=(-3,3),density=True,histtype='step',color='k',lw=1.3,label=f'WR/$\\sqrt{{\\chi^2_\\nu}}$, sd={Zs.std():.2f}')
ax[0].plot(x,stats.norm.pdf(x),'r-',label='N(0,1)'); ax[0].legend(fontsize=7); ax[0].set_xlabel('weighted residual'); ax[0].set_title('Pooled residuals (%d points)'%len(Z))
stats.probplot(Zs,dist='norm',plot=ax[1]); ax[1].set_title('Q-Q plot, rescaled residuals'); ax[1].get_lines()[0].set_markersize(3)
ax[2].hist(df.p_chi2_low,bins=20,range=(0,1),color='steelblue'); ax[2].axhline(len(df)/20,color='r',ls='--',label='expected if errors correct')
ax[2].set_xlabel('P($\\chi^2\\leq\\chi^2_{obs}$ | n-2 dof)'); ax[2].set_title('Calibration of SPARC errors'); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig('catA_fig_residuals.pdf'); plt.close(fig)

fig,ax=plt.subplots(1,3,figsize=(13,3.9))
for gr in 'I','II','III':
    s=df[df.grade==gr]
    ax[0].errorbar(s.R,s.M,xerr=s.eR,yerr=s.eM,fmt='o',ms=4,color=GC[gr],lw=0.7,label=f'grade {gr} ({len(s)})')
Rg=np.geomspace(0.1,8,50)
for vL in [30,60,100,150,200]:
    M=(vL/np.sqrt(1.5)+HZ*Rg)**2*Rg/(2*G)/1e9; ax[0].plot(Rg,M,':',color='0.6',lw=0.8); ax[0].text(Rg[-1],M[-1],f'{vL}',fontsize=6,color='0.4')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('R (kpc)'); ax[0].set_ylabel('M (10$^9$ M$_\\odot$)'); ax[0].legend(fontsize=7)
ax[0].set_title('M-R plane; dotted: constant $v_L$ (km/s)')
ax[1].scatter(df.n,df.rho_MR,c=[GC[g] for g in df.grade],s=18); ax[1].set_xlabel('n points'); ax[1].set_ylabel('corr(M,R)'); ax[1].set_title('M-R degeneracy')
ax[2].hist([df.relM,df.relR,df.relV],bins=np.linspace(0,1,21),label=['$\\sigma_M/M$','$\\sigma_R/R$','$\\sigma_{v_L}/v_L$'],color=['tab:blue','tab:orange','tab:green'])
ax[2].set_xlabel('relative statistical uncertainty'); ax[2].legend(fontsize=7); ax[2].set_title('Parameter precision')
fig.tight_layout(); fig.savefig('catA_fig_params.pdf'); plt.close(fig)

fig,ax=plt.subplots(1,4,figsize=(15,3.8))
ok=df.Vflat_obs>0
ax[0].errorbar(df.Vflat_obs[ok],df.vflat_CL[ok],yerr=df.e_vflat_CL[ok],fmt='o',ms=3.5,color='k',lw=0.7); l=[20,250]; ax[0].plot(l,l,'r--',lw=0.8)
ax[0].set_xlabel('SPARC $V_{flat}$ (km/s)'); ax[0].set_ylabel('CL $v_L=\\sqrt{3/2}X$ (km/s)'); ax[0].set_title(f'Asymptotic speed ({ok.sum()} with $V_{{flat}}$)')
ax[1].scatter(df.Rdisk,df.R,c=[GC[g] for g in df.grade],s=18); ax[1].plot([0.1,6],[0.1,6],'r--',lw=0.8); ax[1].set_xlabel('SPARC $R_{disk}$ (kpc)'); ax[1].set_ylabel('CL R (kpc)'); ax[1].set_title('Scale radius')
ax[2].scatter(df.Mbar,df.M,c=[GC[g] for g in df.grade],s=18); ax[2].plot([0.03,80],[0.03,80],'r--',lw=0.8); ax[2].set_xscale('log'); ax[2].set_yscale('log')
ax[2].set_xlabel('SPARC $M_{bar}$ (10$^9$M$_\\odot$)'); ax[2].set_ylabel('CL M'); ax[2].set_title('Mass')
ax[3].scatter(df.eM/df.M,df.sysM/df.M,s=14,label='M',color='tab:blue'); ax[3].scatter(df.eR/df.R,df.sysR/df.R,s=14,label='R',color='tab:orange',marker='s')
ax[3].plot([0,1.3],[0,1.3],'k:',lw=0.8); ax[3].set_xlabel('statistical (rel.)'); ax[3].set_ylabel('systematic D, i (rel.)'); ax[3].legend(fontsize=7); ax[3].set_title('Stat. vs syst. uncertainty')
ax[3].set_xlim(0,1); ax[3].set_ylim(0,1)
fig.tight_layout(); fig.savefig('catA_fig_physics.pdf'); plt.close(fig)

# ---- atlas
order=df.sort_values(['grade','chi2nu']).Galaxy.tolist(); D=df.set_index('Galaxy')
with PdfPages('catA_atlas.pdf') as pdf:
    for p in range(0,len(order),12):
        fig=plt.figure(figsize=(16,11)); gs=GridSpec(3,4,figure=fig,hspace=0.42,wspace=0.28)
        for k,g in enumerate(order[p:p+12]):
            sub=GridSpecFromSubplotSpec(2,1,gs[k],height_ratios=[3,1.25],hspace=0.05)
            a=fig.add_subplot(sub[0]); b=fig.add_subplot(sub[1],sharex=a)
            d=mm[mm.ID==g]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values; f=F[g]; row=D.loc[g]; M,R=f['p']
            rr=np.linspace(1e-3,r.max()*1.05,600)
            # 1-sigma band from covariance (sampled)
            smp=np.random.default_rng(0).multivariate_normal(f['p'],f['cov'],300); smp=smp[(smp>0).all(1)]
            band=np.array([v2_model(rr,m*1e9,rad) for m,rad in smp]); lo,hi=np.percentile(band,[16,84],axis=0)
            a.fill_between(rr,lo,hi,color=GC[row.grade],alpha=0.18,lw=0)
            a.errorbar(r,v**2,yerr=2*v*ev,fmt='o',ms=2.8,color='crimson',ecolor='0.6',lw=0.7,zorder=3)
            a.plot(rr,v2_model(rr,M*1e9,R),'k-',lw=1.2); a.axvline(R,color='k',ls=':',lw=0.7)
            a.set_title(f"{g}   grade {row.grade}   n={row.n}",fontsize=8.5,weight='bold',color=GC[row.grade])
            a.text(0.03,0.95,f"M=({M:.3g}$\\pm${row.eM:.2g})e9, R={R:.2f}$\\pm${row.eR:.2f}\n$v_L$={row.vflat_CL:.1f}$\\pm${row.e_vflat_CL:.1f} km/s\n$\\chi^2_\\nu$={row.chi2nu:.2f}  RMS$_{{rel}}$={row.RMSrel:.2f}  $p_{{runs}}$={row.p_runs:.2f}",
                   transform=a.transAxes,va='top',fontsize=6.2)
            a.set_ylim(bottom=0); a.tick_params(labelbottom=False); a.set_ylabel('$v^2$ (km$^2$s$^{-2}$)',fontsize=6.5)
            z=f['z']; b.axhline(0,color='k',lw=0.7); b.axhline(1,color='0.6',ls='--',lw=0.6); b.axhline(-1,color='0.6',ls='--',lw=0.6)
            b.scatter(r,z,c=np.where(z>0,'crimson','royalblue'),s=10,zorder=3); lim=max(1.5,np.abs(z).max()*1.15); b.set_ylim(-lim,lim)
            b.set_ylabel('WR',fontsize=6.5); b.set_xlabel('r (kpc)',fontsize=6.5)
        fig.suptitle("Category A single-L CL fits with 1$\\sigma$ parameter band and weighted residuals (sorted by grade, then $\\chi^2_\\nu$)",fontsize=10)
        pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
print(df.grade.value_counts().to_dict(), "err_over:",df.err_over.sum())
print(df.groupby('grade')[['n','chi2nu','RMSrel','relM','relR','relV']].median().round(3))
