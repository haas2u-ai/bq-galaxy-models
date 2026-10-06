import sys,numpy as np,pandas as pd,pickle,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from catD import MODELS,lsq,G,HZ
from cl_fit import load_massmodels,load_table1
from cl2_fit import _metrics
from scipy import stats
from wave_detect import runs_p
mm=load_massmodels(); t=load_table1(); df=pd.read_csv('catD_results.csv'); F=pickle.load(open('catD_fits.pkl','rb'))
K={'S':2,'VW':4,'2L':4,'VW2':6,'2LVW':6}
LAB={'S':'single-L kept','VW':'one virial window (C-like)','2L':'two-L (B-like)','VW2':'two virial windows','2LVW':'two-L + disk window'}
best=[];acc=[]
for _,r in df.iterrows():
    B={k:r[f'BIC_{k}'] for k in K if np.isfinite(r[f'BIC_{k}'])}; b=min(B.values())
    cand=sorted([k for k in B if B[k]<=b+2],key=lambda k:(K[k],B[k])); m=cand[0]
    if m!='S' and B[m]-B['S']>=-6: m='S'
    best.append(m)
df['best']=best; df['outcome']=df.best.map(LAB)
df['dBIC_best']=[r[f'BIC_{r.best}']-r.BIC_S for _,r in df.iterrows()]
df['ambiguous']=[ (sum(1 for k in K if np.isfinite(r[f'BIC_{k}']) and k!=r.best and r[f'BIC_{k}']<=r[f'BIC_{r.best}']+2)>0) for _,r in df.iterrows()]
# diagnostics on best model
rows=[]
for _,r in df.iterrows():
    g=r.Galaxy; m=r.best; f=MODELS[m][0]; x=F[g][m]['x']; d=mm[mm.ID==g]; rr,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev; n=len(rr); row=t.loc[g]
    z=(y-f(rr,x))/s; J=F[g][m]['jac']
    try: cov=np.linalg.inv(J.T@J); e=np.sqrt(np.clip(np.diag(cov),0,None))
    except Exception: e=np.full(len(x),np.nan)
    jk=[]
    for i in range(n):
        mk=np.ones(n,bool); mk[i]=False; jk.append(lsq(m,rr[mk],y[mk],s[mk],[x]).x)
    jk=np.array(jk); jke=np.sqrt((n-1)/n*np.sum((jk-jk.mean(0))**2,axis=0))
    fD=1+row.eD/row.D; xD=lsq(m,rr*fD,y,s,[x]).x
    inc=np.radians(row.Inc); i2=min(inc+np.radians(row.eInc),np.radians(89.9)); fi=(np.sin(inc)/np.sin(i2))**2
    xI=lsq(m,rr,y*fi,s*fi,[x]).x; x0=lsq(m,rr,y,s,[x],Hz=0.0).x
    # derived physical quantities
    if m=='S': rad=[x[1]]; names=['R']
    elif m=='VW': rad=[x[1],x[1]+x[2]]; names=['R','r_v']
    elif m=='2L': rad=[x[1],x[1]+x[3]]; names=['R1','R2']
    elif m=='VW2': rad=[x[1],x[1]+x[2],x[1]+x[2]+x[4]]; names=['R','r_v1','r_v2']
    else: rad=[x[1],x[1]+x[3],x[1]+x[3]+x[4]]; names=['R1','R2','r_v']
    pv={}
    if m=='VW': pv['p']=2+x[3]/x[0]
    if m=='VW2': pv['p']=2+x[3]/x[0]; pv['q']=2+x[5]/x[0]
    if m=='2LVW': pv['p']=2+x[5]/x[2]
    relmax=np.nanmax(np.abs(e/x)) if np.all(np.isfinite(e)) else np.nan
    rows.append(dict(Galaxy=g,k=len(x),params=';'.join(f'{a:.4g}' for a in x),errors=';'.join(f'{a:.3g}' for a in e),jk=';'.join(f'{a:.3g}' for a in jke),
        sysD=';'.join(f'{a:.3g}' for a in np.abs(xD-x)),sysI=';'.join(f'{a:.3g}' for a in np.abs(xI-x)),hz_max=np.nanmax(np.abs((x0-x)/np.where(x!=0,x,np.nan))),
        radii=';'.join(f'{n_}={a:.2f}' for n_,a in zip(names,rad)),pvals=';'.join(f'{k}={v_:.3g}' for k,v_ in pv.items()),
        relerr_max=relmax,jk_rel_max=np.nanmax(np.abs(jke/np.where(x!=0,x,np.nan))),pruns_best=runs_p(z),DW_best=np.sum(np.diff(z)**2)/np.sum(z**2),
        pSW_best=stats.shapiro(z).pvalue,Plow_best=stats.chi2.cdf(np.sum(z**2),n-len(x)),maxz_best=np.abs(z).max(),chi2nu_best=np.sum(z**2)/(n-len(x)),RMS_best=np.sqrt(np.mean(((y-f(rr,x))/y)**2)),
        x=x,e=e,names=names,rad=rad,pv=pv))
P=pd.DataFrame(rows); df=df.merge(P.drop(columns=['x','e','names','rad','pv']),on='Galaxy')
df['flagM']=df.chi2nu_best>2
pickle.dump({r.Galaxy:r for _,r in P.iterrows()},open('catD_best.pkl','wb'))
df.to_csv('catD_results.csv',index=False,float_format='%.5g')
pd.set_option('display.width',250)
print(df[['Galaxy','T','Q','n','best','ambiguous','dBIC_best','chi2nu_S','chi2nu_best','RMS_S','RMS_best','p_runs1','pruns_best','DW_S','DW_best','radii','pvals','relerr_max','jk_rel_max','hz_max']].round(3).to_string())
print(df.outcome.value_counts())
