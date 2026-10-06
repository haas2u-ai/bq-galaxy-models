import sys, numpy as np, pandas as pd, pickle, warnings; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))); warnings.filterwarnings('ignore')
from cl_fit import v2_model, G, HZ, load_massmodels, load_table1
from scipy.optimize import least_squares
from scipy import stats
from wave_detect import runs_p
mm=load_massmodels(); t=load_table1(); S=pd.read_csv('pipeline_summary.csv').set_index('Galaxy')
A=S[S.category.str.startswith('A')].index.tolist()

def fit1(r,v,ev,Hz=HZ,offset=False):
    y,s=v**2,2*v*ev
    def mod(p): return v2_model(r,p[0]*1e9,p[1],p[2] if offset else 0.0,Hz)
    best=None
    for R0 in np.geomspace(max(r.min(),.05),r.max()*1.2,14):
        p0=[max(v.max()**2*R0/(3*G),1e5)/1e9,R0]+([0.] if offset else [])
        lo=[1e-5,0.02]+([-np.inf] if offset else []); hi=[500,100]+([np.inf] if offset else [])
        try: sol=least_squares(lambda p:(mod(p)-y)/s,p0,bounds=(lo,hi),x_scale='jac')
        except Exception: continue
        if best is None or sol.cost<best.cost: best=sol
    p=best.x; z=(y-mod(p))/s; k=len(p); n=len(r); chi2=float(np.sum(z**2))
    try: cov=np.linalg.inv(best.jac.T@best.jac)
    except np.linalg.LinAlgError: cov=np.full((k,k),np.nan)
    e=np.sqrt(np.clip(np.diag(cov),0,None))
    return dict(p=p,e=e,cov=cov,z=z,chi2=chi2,n=n,k=k,chi2nu=chi2/(n-k),AIC=chi2+2*k,BIC=chi2+k*np.log(n),
                RMSrel=float(np.sqrt(np.mean(((y-mod(p))/y)**2))),bound=bool(np.any(np.isclose(p[:2],[1e-5,0.02],rtol=1e-3))|np.any(np.isclose(p[:2],[500,100],rtol=1e-3))))
rows=[]; F={}
for g in A:
    d=mm[mm.ID==g]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values
    f=fit1(r,v,ev); f0=fit1(r,v,ev,Hz=0.0); f3=fit1(r,v,ev,offset=True)
    M,R=f['p']; eM,eR=f['e']; rho=f['cov'][0,1]/(eM*eR) if eM>0 and eR>0 else np.nan
    # jackknife
    jk=[]
    for i in range(len(r)):
        m=np.ones(len(r),bool); m[i]=False
        if m.sum()>=3: jk.append(fit1(r[m],v[m],ev[m])['p'])
    jk=np.array(jk); nJ=len(jk)
    jkM=np.sqrt((nJ-1)/nJ*np.sum((jk[:,0]-jk[:,0].mean())**2)); jkR=np.sqrt((nJ-1)/nJ*np.sum((jk[:,1]-jk[:,1].mean())**2))
    shift=np.max(np.abs(jk-f['p'][:2])/np.array([eM,eR]),axis=0)
    # systematics (distance, inclination) -- numerically: rescale data and refit
    row=t.loc[g]; inc=np.radians(row.Inc); dinc=np.radians(row.eInc)
    fD=fit1(r*(1+row.eD/row.D),v,ev); fi=fit1(r,v*np.sin(inc)/np.sin(min(inc+dinc,np.radians(89.9))),ev*np.sin(inc)/np.sin(min(inc+dinc,np.radians(89.9))))
    sysM=np.hypot(fD['p'][0]-M,fi['p'][0]-M); sysR=np.hypot(fD['p'][1]-R,fi['p'][1]-R)
    z=f['z']; n=len(r)
    sw=stats.shapiro(z).pvalue if n>=3 else np.nan
    pchi_low=stats.chi2.cdf(f['chi2'],n-2)          # P(chi2 <= observed)
    nin=int(np.sum(r<=R)); nout=n-nin
    Mbar=(0.5*(row.L36-row.Lb)+0.7*row.Lb+1.33*row.MHI)*1e9
    rows.append(dict(Galaxy=g,T=int(row['T']),Q=int(row.Q),Inc=row.Inc,D=row.D,fD=int(row.fD),n=n,n_in=nin,n_out=nout,rmax_R=r.max()/R,
        M=M,eM=eM,sysM=sysM,jkM=jkM,R=R,eR=eR,sysR=sysR,jkR=jkR,rho_MR=rho,jk_shift_M=shift[0],jk_shift_R=shift[1],
        chi2=f['chi2'],chi2nu=f['chi2nu'],p_chi2_low=pchi_low,AIC=f['AIC'],BIC=f['BIC'],RMSrel=f['RMSrel'],
        p_runs=runs_p(z),DW=np.sum(np.diff(z)**2)/np.sum(z**2),p_SW=sw,frac1s=np.mean(np.abs(z)<=1),maxabsz=np.abs(z).max(),
        bound=f['bound'],M_Hz0=f0['p'][0],R_Hz0=f0['p'][1],dBIC_offset=f3['BIC']-f['BIC'],V02=f3['p'][2],eV02=f3['e'][2],
        vflat_CL=np.sqrt(1.5)*(np.sqrt(2*G*M*1e9/R)-HZ*R),Vflat_obs=row.Vf,Mbar=Mbar/1e9,Rdisk=row.Rd,Reff=row.Reff))
    F[g]=f
df=pd.DataFrame(rows); df.to_csv('catA_results.csv',index=False,float_format='%.5g'); pickle.dump(F,open('catA_fits.pkl','wb'))
pd.set_option('display.width',260)
print(df.describe().T[['min','25%','50%','75%','max']].round(3).to_string())
print("Q:",df.Q.value_counts().to_dict(),"T:",df['T'].value_counts().sort_index().to_dict())
