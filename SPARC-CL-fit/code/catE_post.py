import sys,numpy as np,pandas as pd,pickle,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from catD import MODELS,lsq
from cl_fit import load_massmodels
from scipy import stats
from wave_detect import runs_p
mm=load_massmodels(); df=pd.read_csv('catE_results.csv'); F=pickle.load(open('catE_fits.pkl','rb'))
K={'S':2,'VW':4,'2L':4,'VW2':6}
out=[]
for _,r in df.iterrows():
    B={k:r[f'BIC_{k}'] for k in K if np.isfinite(r[f'BIC_{k}'])}; b=min(B.values())
    m=sorted([k for k in B if B[k]<=b+2],key=lambda k:(K[k],B[k]))[0]
    if m!='S' and B[m]-B['S']>=-6: m='S'
    g=r.Galaxy; x=F[g][m]['x']; J=F[g][m]['jac']
    try: e=np.sqrt(np.clip(np.diag(np.linalg.inv(J.T@J)),0,None))
    except Exception: e=np.full(len(x),np.nan)
    d=mm[mm.ID==g]; rr,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev; n=len(rr)
    jk=[]
    for i in range(n):
        mk=np.ones(n,bool); mk[i]=False; jk.append(lsq(m,rr[mk],y[mk],s[mk],[x]).x)
    jk=np.array(jk); jke=np.sqrt((n-1)/n*np.sum((jk-jk.mean(0))**2,axis=0))
    pv={}; rad={'R':x[1]}
    if m=='VW': rad['r_v']=x[1]+x[2]; pv['p']=2+x[3]/x[0]
    if m=='2L': rad['R2']=x[1]+x[3]
    if m=='VW2': rad['r_v1']=x[1]+x[2]; rad['r_v2']=x[1]+x[2]+x[4]; pv['p']=2+x[3]/x[0]; pv['q']=2+x[5]/x[0]
    z=F[g][m]['z']; chi=np.sum(z**2)
    relmax=np.nanmax(np.abs(e/x)) if np.all(np.isfinite(e)) else np.inf
    if m=='S': status='single-L kept'
    elif x[1]<=0.025 or any(abs(v_)>100 for v_ in pv.values()): status='phenomenological'
    elif relmax>0.6 or np.nanmax(np.abs(jke/x))>0.6: status='weakly constrained'
    elif chi/(n-len(x))>2: status='improved, still chi2nu>2'
    else: status='clean'
    out.append(dict(Galaxy=g,best=m,dBIC_best=B[m]-B['S'],chi2nu_best=chi/(n-len(x)),RMS_best=np.sqrt(np.mean(((y-MODELS[m][0](rr,x))/y)**2)),
                    pruns_best=runs_p(z),DW_best=np.sum(np.diff(z)**2)/np.sum(z**2),status=status,relerr_max=relmax,jk_rel_max=np.nanmax(np.abs(jke/x)),
                    radii=';'.join(f'{k}={v_:.2f}' for k,v_ in rad.items()),pvals=';'.join(f'{k}={v_:.3g}' for k,v_ in pv.items()),
                    params=';'.join(f'{a:.4g}' for a in x),errors=';'.join(f'{a:.3g}' for a in e)))
    # VW quick info regardless of best
    if np.isfinite(r.BIC_VW):
        xv=F[g]['VW']['x']; out[-1].update(VW_rv=xv[1]+xv[2],VW_p=2+xv[3]/xv[0],VW_R=xv[1],dBIC_VW=r.BIC_VW-r.BIC_S)
O=pd.DataFrame(out); df=df.drop(columns=[c for c in O.columns if c in df.columns and c!='Galaxy']).merge(O,on='Galaxy')
df.to_csv('catE_results.csv',index=False,float_format='%.5g')
pd.set_option('display.width',260)
print(df[['Galaxy','T','Q','n','Inc','best','status','dBIC_best','chi2nu_S','chi2nu_best','pruns_best','radii','pvals','relerr_max','jk_rel_max','sigma_int','n_out3','VW_p','VW_rv','dBIC_VW']].round(3).to_string())
