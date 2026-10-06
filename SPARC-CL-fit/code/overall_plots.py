import sys,pickle,numpy as np,pandas as pd; sys.path.insert(0,'.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from cl_fit import v2_model,load_massmodels
from cl2_fit import v2_two
from catC_model import mVW
from catD import MODELS
mm=load_massmodels(); O=pd.read_csv('overall_classes.csv')
PF=pickle.load(open('pipeline_fits.pkl','rb')); CF=pickle.load(open('catC_fits.pkl','rb')); DF=pickle.load(open('catD_fits.pkl','rb'))
CLS=['Single-L','Double (two-L reset)','Single-L + descent','Descent only','Multi-region','Flattening window','Unresolved (bulge / non-nested)']
COL={'Single-L':'0.5','Double (two-L reset)':'tab:green','Single-L + descent':'tab:purple','Descent only':'tab:red','Multi-region':'tab:blue','Flattening window':'tab:cyan','Unresolved (bulge / non-nested)':'tab:brown'}
TY=['S0','Sa','Sab','Sb','Sbc','Sc','Scd','Sd','Sdm','Sm','Im','BCD']
plt.rcParams.update({'font.size':8})
# Fig 1 morphology
fig,ax=plt.subplots(1,2,figsize=(14,4.3),gridspec_kw={'width_ratios':[1.4,1]})
ct=pd.crosstab(O.cls,O['T']).reindex(CLS).reindex(columns=range(12),fill_value=0)
frac=ct.div(ct.sum(1),axis=0); left=np.zeros(len(CLS)); cmap=plt.get_cmap('viridis',12)
for tt in range(12):
    ax[0].barh(range(len(CLS)),frac[tt],left=left,color=cmap(tt),label=TY[tt]); left+=frac[tt]
ax[0].set_yticks(range(len(CLS))); ax[0].set_yticklabels([f"{c} ({ct.loc[c].sum()})" for c in CLS]); ax[0].invert_yaxis()
ax[0].set_xlabel('fraction of class'); ax[0].legend(ncol=6,fontsize=6.5,loc='upper center',bbox_to_anchor=(0.5,-0.13)); ax[0].set_title('Hubble-type composition of each class')
for i,c in enumerate(CLS):
    s=O[O.cls==c]; ax[1].scatter(s['T']+np.random.default_rng(i).uniform(-.25,.25,len(s)),np.full(len(s),i)+np.random.default_rng(i+9).uniform(-.2,.2,len(s)),c=COL[c],s=14)
    ax[1].plot(s['T'].median(),i,'k|',ms=16,mew=2)
ax[1].set_yticks(range(len(CLS))); ax[1].set_yticklabels(['']*len(CLS)); ax[1].invert_yaxis(); ax[1].set_xticks(range(12)); ax[1].set_xticklabels(TY,rotation=45)
ax[1].set_title('Hubble type per galaxy (bar: median)')
fig.tight_layout(); fig.savefig('overall_fig_morphology.pdf',bbox_inches='tight'); plt.close(fig)
# Fig 2 quality
fig,ax=plt.subplots(1,3,figsize=(15,4.2))
for c in CLS:
    s=O[O.cls==c]; ax[0].scatter(s.chi2nu_S,s.chi2nu_final,c=COL[c],s=16,label=c)
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].plot([.01,200],[.01,200],'k:',lw=.8); ax[0].axhline(1,color='0.7',lw=.6); ax[0].axhline(2,color='r',ls='--',lw=.6)
ax[0].set_xlabel('$\\chi^2_\\nu$ single-L'); ax[0].set_ylabel('$\\chi^2_\\nu$ final description'); ax[0].legend(fontsize=6)
data=[O[O.cls==c].RMS_final for c in CLS]; ax[1].boxplot(data,vert=True,widths=.6); ax[1].set_xticks(range(1,8)); ax[1].set_xticklabels([c.split(' (')[0] for c in CLS],rotation=40,ha='right')
ax[1].set_ylabel('RMS$_{rel}$ in $v^2$, final'); ax[1].set_yscale('log')
for c in CLS:
    s=O[(O.cls==c)&(O.Vflat>0)&O.v_asym.notna()]; ax[2].scatter(s.Vflat,s.v_asym,c=COL[c],s=16)
ax[2].plot([15,320],[15,320],'k:',lw=.8); ax[2].set_xlabel('SPARC $V_{flat}$ (km/s)'); ax[2].set_ylabel('model asymptotic speed (km/s)')
ax[2].set_title('$v_L$ (single), $v_{L2}$ (double), $\\sqrt{A}$ (descent)',fontsize=8)
fig.tight_layout(); fig.savefig('overall_fig_quality.pdf',bbox_inches='tight'); plt.close(fig)
# Fig 3 gallery
EX=[('Single-L','UGC05005'),('Double (two-L reset)','IC2574'),('Single-L + descent','NGC3198'),('Descent only','UGC05253'),('Multi-region','NGC2841'),('Flattening window','ESO563-G021'),('Unresolved (bulge / non-nested)','UGC02487')]
fig,axs=plt.subplots(2,4,figsize=(16,7.5)); axs=axs.ravel()
for a,(c,g) in zip(axs,EX):
    d=mm[mm.ID==g]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values; rr=np.linspace(1e-2,r.max()*1.04,1500)
    a.errorbar(r,v**2,yerr=2*v*ev,fmt='o',ms=2.5,color='crimson',ecolor='0.6',lw=.6,zorder=3)
    x=PF[g]['single']['params']; a.plot(rr,v2_model(rr,x[0]*1e9,x[1]),'-' if c=='Single-L' else ':',color='k' if c=='Single-L' else '0.4',lw=1.4 if c=='Single-L' else 1.2,label='single-L')
    if c=='Double (two-L reset)':
        q=PF[g]['two']['params']; yy=v2_two(rr,q[0]*1e9,q[1],q[2]*1e9,q[3]); yy[np.abs(rr-q[3])<(rr[1]-rr[0])]=np.nan; a.plot(rr,yy,'-',color=COL[c],lw=1.6,label='two-L')
    elif c in ('Single-L + descent','Descent only','Flattening window'): a.plot(rr,mVW(rr,CF[g]['x']),'-',color=COL[c],lw=1.6,label='CL + window')
    elif c in ('Multi-region','Unresolved (bulge / non-nested)'): a.plot(rr,MODELS['VW2'][0](rr,DF[g]['VW2']['x']),'-',color=COL[c],lw=1.6,label='two windows')
    a.set_title(f"{c}\n{g}",fontsize=8.5,color=COL[c] if c!='Single-L' else 'k',weight='bold'); a.set_ylim(0,(v**2).max()*1.25); a.legend(fontsize=6.5,loc='lower right',frameon=False)
    a.set_xlabel('r (kpc)',fontsize=7); a.set_ylabel('$v^2$ (km$^2$s$^{-2}$)',fontsize=7)
axs[-1].axis('off')
fig.tight_layout(); fig.savefig('overall_fig_gallery.pdf',bbox_inches='tight'); plt.close(fig)
