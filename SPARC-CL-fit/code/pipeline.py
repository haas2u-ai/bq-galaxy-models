"""CL pipeline on SPARC (175):  1 single-L fit  2 wave screen  3 two-L fit seeded at R2 ~ r_cross1/0.8
4 accept if dBIC<-6  5 tentative if n<=13.  Hz = 2.2e-18 s^-1 fixed; fits on v^2."""
import sys, numpy as np, pandas as pd, pickle, warnings; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))); warnings.filterwarnings('ignore')
from cl_fit import v2_model, G, HZ, load_massmodels, load_table1
from cl2_fit import v2_two, fit_single, fit_two, _metrics, _errors
from wave_detect import detect
from scipy.optimize import least_squares

def fit_two_seeded(r, v, ev, R2seed, phi=False):
    y, s = v**2, 2*v*ev
    def unpack(q): return q[0], q[1], q[2], q[1]+q[3], (q[4] if phi else 0.0)
    def mod(q):
        M1,R1,M2,R2,ph = unpack(q); return v2_two(r, M1*1e9, R1, M2*1e9, R2, ph)
    best=None
    R2s = np.clip(R2seed*np.array([0.7,0.85,1.0,1.2,1.45]), r.min()*1.5, r.max()*0.97)
    for R2 in np.unique(R2s):
        for R1 in np.geomspace(max(r.min(),0.05), 0.8*R2, 6):
            inner = v[r<=R2].max() if np.any(r<=R2) else v.min()
            q0=[max(inner**2*R1/(3*G),1e5)/1e9, R1, max(v.max()**2*R2/(3*G),1e5)/1e9, R2-R1]+([0.] if phi else [])
            lo=[1e-5,0.02,1e-5,0.01]+([-np.inf] if phi else []); hi=[500,100,5000,200]+([np.inf] if phi else [])
            try: sol=least_squares(lambda q:(mod(q)-y)/s, q0, bounds=(lo,hi), x_scale='jac')
            except Exception: continue
            if best is None or sol.cost<best.cost: best=sol
    q=best.x; k=len(q); out=_metrics(y,s,mod(q),k); e=_errors(best,k)
    try: cov=np.linalg.inv(best.jac.T@best.jac); eR2=np.sqrt(max(cov[1,1]+cov[3,3]+2*cov[1,3],0))
    except np.linalg.LinAlgError: eR2=np.nan
    M1,R1,M2,R2,ph=unpack(q)
    out.update(params=np.array([M1,R1,M2,R2]+([ph] if phi else [])), errors=np.array([e[0],e[1],e[2],eR2]+([e[4]] if phi else [])))
    return out

if __name__=='__main__':
    mm=load_massmodels(); t=load_table1()
    D13=['D631-7','DDO161','F571-8','F583-4','IC2574','NGC0247','NGC3109','NGC3741','NGC3972','UGC04278','UGC05829','UGC06446','UGC12732']
    rows=[]; fits={}
    for g,d in mm.groupby('ID',sort=False):
        r,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev
        s1=fit_single(r,v,ev)                                              # step 1
        z=(y-v2_model(r,s1['params'][0]*1e9,s1['params'][1]))/s
        w=detect(z) if len(r)>=6 else dict(A_pmp=np.nan,A_mpm=np.nan,p_runs=1.0,i=0,j=0,wave=False)  # step 2
        rx1=r[w['i']] if w['wave'] else np.nan
        if w['wave']: pat='B: +-+ wave (double-L candidate)'
        elif w['p_runs']<0.05 and w['A_mpm']>=0.8 and w['A_mpm']>w['A_pmp']: pat='C: -+- wave (declining / virial)'
        elif w['p_runs']<0.05: pat='D: structured, other'
        elif s1['chi2nu']<=1: pat='A: single-L adequate'
        else: pat='E: poor, unstructured'
        rec=dict(Galaxy=g,T=int(t.loc[g,'T']),Q=int(t.loc[g,'Q']),n=len(r),R=s1['params'][1],eR=s1['errors'][1],
                 M=s1['params'][0],eM=s1['errors'][0],chi2nu1=s1['chi2nu'],RMSrel1=s1['RMSrel'],BIC1=s1['BIC'],
                 p_runs=w['p_runs'],A_pmp=w['A_pmp'],A_mpm=w['A_mpm'],category=pat,r_cross1=rx1,
                 signs=''.join('+' if x>0 else '-' for x in z),double_2018=g in D13)
        f=dict(single=s1,z1=z)
        if w['wave']:                                                      # step 3
            two=fit_two_seeded(r,v,ev,rx1/0.8); twp=fit_two_seeded(r,v,ev,rx1/0.8,phi=True)
            g2=fit_two(r,v,ev); g3=fit_two(r,v,ev,phi=True)            # global-grid check
            seed_ok = two['chi2']<=g2['chi2']*1.001
            if g2['chi2']<two['chi2']: two=g2
            if g3['chi2']<twp['chi2']: twp=g3
            rec.update(seed_found_best=seed_ok)
            dB=two['BIC']-s1['BIC']
            rec.update(dBIC=dB, accepted=dB<-6, tentative=(dB<-6) and len(r)<=13)   # steps 4,5
            f.update(two=two,twophi=twp)
        fits[g]=f; rows.append(rec)
    df=pd.DataFrame(rows); df.to_csv('pipeline_summary.csv',index=False,float_format='%.5g')
    pickle.dump(fits,open('pipeline_fits.pkl','wb'))
    print(df.category.value_counts())
    acc=df[df.get('accepted',False)==True]
    print("accepted:",len(acc)," tentative:",int(acc.tentative.sum())," 2018 among accepted:",int(acc.double_2018.sum()))
    print("seed best in",int(df.seed_found_best.sum()),"of",int(df.seed_found_best.notna().sum()))
    print(acc[['Galaxy','n','dBIC','tentative','double_2018','seed_found_best']].sort_values('dBIC').to_string())
