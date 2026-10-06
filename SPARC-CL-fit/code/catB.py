import sys, numpy as np, pandas as pd, pickle, warnings; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))); warnings.filterwarnings('ignore')
from cl_fit import v2_model, G, HZ, load_massmodels, load_table1
from cl2_fit import v2_two
from scipy.optimize import least_squares
from scipy import stats
from wave_detect import runs_p
mm=load_massmodels(); t=load_table1(); S=pd.read_csv('pipeline_summary.csv').set_index('Galaxy'); PF=pickle.load(open('pipeline_fits.pkl','rb'))
B=S[S.category.str.startswith('B')].index.tolist()

def warm_two(r,v,ev,q,Hz=HZ):
    """refit two-L from q=[M1,R1,M2,R2] (M in 1e9) with small perturbations"""
    y,s=v**2,2*v*ev
    def mod(x): return v2_two(r,x[0]*1e9,x[1],x[2]*1e9,x[1]+x[3],0.0,Hz)
    best=None; rng=np.random.default_rng(1)
    for k in range(4):
        f=1+0.15*rng.standard_normal(4)*(k>0)
        x0=[q[0]*f[0],q[1]*f[1],q[2]*f[2],max(q[3]-q[1],0.02)*f[3]]
        try: sol=least_squares(lambda x:(mod(x)-y)/s,x0,bounds=([1e-5,0.02,1e-5,0.01],[500,100,5000,200]),x_scale='jac')
        except Exception: continue
        if best is None or sol.cost<best.cost: best=sol
    x=best.x; return np.array([x[0],x[1],x[2],x[1]+x[3]]), 2*best.cost
def warm_one(r,v,ev,p,Hz=HZ):
    y,s=v**2,2*v*ev
    sol=least_squares(lambda x:(v2_model(r,x[0]*1e9,x[1],0,Hz)-y)/s,p,bounds=([1e-5,0.02],[500,100]),x_scale='jac'); return sol.x
def vL(M,R): return np.sqrt(1.5)*(np.sqrt(2*G*M*1e9/R)-HZ*R)
def cov_two(r,v,ev,q):
    y,s=v**2,2*v*ev
    f=lambda x:(v2_two(r,x[0]*1e9,x[1],x[2]*1e9,x[3])-y)/s
    J=np.zeros((len(r),4)); h=1e-5
    for i in range(4):
        dx=np.zeros(4); dx[i]=h*max(abs(q[i]),1e-3); J[:,i]=(f(q+dx)-f(q-dx))/(2*dx[i])
    try: return np.linalg.inv(J.T@J)
    except np.linalg.LinAlgError: return np.full((4,4),np.nan)
rows=[]; ZZ={}
for g in B:
    d=mm[mm.ID==g]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev; n=len(r); row=t.loc[g]; sm=S.loc[g]
    f1,f2,f3=PF[g]['single'],PF[g]['two'],PF[g]['twophi']
    q=f2['params'].copy()
    z1=PF[g]['z1']; z2=(y-v2_two(r,q[0]*1e9,q[1],q[2]*1e9,q[3]))/s
    q5=f3['params']; z3=(y-v2_two(r,q5[0]*1e9,q5[1],q5[2]*1e9,q5[3],q5[4]))/s
    ZZ[g]=(z1,z2,z3)
    cov=cov_two(r,v,ev,q); e=np.sqrt(np.clip(np.diag(cov),0,None))
    corr=cov/np.outer(e,e)
    # derived v_L1, v_L2 with errors
    def gradv(fun):
        gr=np.zeros(4); h=1e-6
        for i in range(4):
            dq=np.zeros(4); dq[i]=h*q[i]; gr[i]=(fun(q+dq)-fun(q-dq))/(2*h*q[i])
        return np.sqrt(gr@cov@gr)
    vL1=vL(q[0],q[1]); vL2=vL(q[2],q[3]); evL1=gradv(lambda x:vL(x[0],x[1])); evL2=gradv(lambda x:vL(x[2],x[3]))
    # jump at R2
    lo=v2_two(np.array([q[3]*(1-1e-7)]),q[0]*1e9,q[1],q[2]*1e9,q[3])[0]; hi=v2_two(np.array([q[3]*(1+1e-7)]),q[0]*1e9,q[1],q[2]*1e9,q[3])[0]
    jump=(hi-lo)/lo
    # jackknife (warm start)
    jk=[]
    for i in range(n):
        m=np.ones(n,bool); m[i]=False; jk.append(warm_two(r[m],v[m],ev[m],q)[0])
    jk=np.array(jk); jke=np.sqrt((n-1)/n*np.sum((jk-jk.mean(0))**2,axis=0)); jkshift=np.max(np.abs(jk-q)/np.where(e>0,e,np.inf))
    # systematics
    qD,_=warm_two(r*(1+row.eD/row.D),v,ev,q*np.array([1+row.eD/row.D,1+row.eD/row.D,1+row.eD/row.D,1+row.eD/row.D]))
    inc=np.radians(row.Inc); i2=min(inc+np.radians(row.eInc),np.radians(89.9)); fac=np.sin(inc)/np.sin(i2)
    qI,_=warm_two(r,v*fac,ev*fac,q*np.array([fac**2,1,fac**2,1]))
    sysq=np.hypot(qD-q,qI-q)
    qH,_=warm_two(r,v,ev,q,Hz=0.0)
    nb=int(np.sum(r<=q[1])); nbar=int(np.sum((r>q[1])&(r<=q[3]))); nd=n-nb-nbar
    Mbar=(0.5*(row.L36-row.Lb)+0.7*row.Lb+1.33*row.MHI)
    rows.append(dict(Galaxy=g,T=int(row['T']),Q=int(row.Q),Inc=row.Inc,n=n,double_2018=sm.double_2018,r_cross1=sm.r_cross1,A_pmp=sm.A_pmp,p_runs1=sm.p_runs,
        M=f1['params'][0],eM=f1['errors'][0],R=f1['params'][1],eR=f1['errors'][1],chi2_1=f1['chi2'],chi2nu1=f1['chi2nu'],BIC1=f1['BIC'],AIC1=f1['AIC'],RMS1=f1['RMSrel'],DW1=np.sum(np.diff(z1)**2)/np.sum(z1**2),
        M1=q[0],eM1=e[0],R1=q[1],eR1=e[1],M2=q[2],eM2=e[2],R2=q[3],eR2=e[3],sM1=sysq[0],sR1=sysq[1],sM2=sysq[2],sR2=sysq[3],
        jM1=jke[0],jR1=jke[1],jM2=jke[2],jR2=jke[3],jk_shift=jkshift,rho_M1R1=corr[0,1],rho_M2R2=corr[2,3],rho_R1R2=corr[1,3],
        vL1=vL1,evL1=evL1,vL2=vL2,evL2=evL2,jump=jump,n_bulge=nb,n_bar=nbar,n_disk=nd,
        chi2_2=f2['chi2'],chi2nu2=f2['chi2nu'],BIC2=f2['BIC'],AIC2=f2['AIC'],RMS2=f2['RMSrel'],p_runs2=runs_p(z2),DW2=np.sum(np.diff(z2)**2)/np.sum(z2**2),
        p_SW2=stats.shapiro(z2).pvalue,P_low2=stats.chi2.cdf(f2['chi2'],n-4),maxz2=np.abs(z2).max(),
        Phi=q5[4],ePhi=f3['errors'][4],chi2_3=f3['chi2'],chi2nu3=f3['chi2nu'],BIC3=f3['BIC'],AIC3=f3['AIC'],RMS3=f3['RMSrel'],p_runs3=runs_p(z3),
        dBIC=f2['BIC']-f1['BIC'],dAIC=f2['AIC']-f1['AIC'],dBIC_phi=f3['BIC']-f2['BIC'],
        M1_Hz0=qH[0],R1_Hz0=qH[1],M2_Hz0=qH[2],R2_Hz0=qH[3],
        Vflat=row.Vf,Mbar=Mbar,Rdisk=row.Rd,Reff=row.Reff,RHI=row.RHI))
    print(g,end=' ',flush=True)
df=pd.DataFrame(rows)
df['accepted']=df.dBIC<-6; df['tentative']=df.accepted&(df.n<=13)
df['flagN']=df.accepted&((df.M2<df.M1)|(df.jump<-0.4)); df['flagM']=df.accepted&(df.chi2nu2>2)
for c in ['M1','R1','M2','R2']: df['rel'+c]=df['e'+c]/df[c]
df['relvL1']=df.evL1/df.vL1; df['relvL2']=df.evL2/df.vL2
def grade(r):
    if not r.accepted: return '--'
    if r.flagN: return 'N'
    if r.relM1<=0.3 and r.relR1<=0.3 and r.relM2<=0.3 and r.relR2<=0.3 and r.n_bulge>=2 and r.n_bar>=2 and r.n_disk>=3 and r.jk_shift<=2: g='I'
    elif r.relvL1<=0.10 and r.relvL2<=0.10: g='II'
    else: g='III'
    return g+('M' if r.flagM else '')
df['grade']=df.apply(grade,axis=1)
def outcome(r):
    if not r.accepted: return 'single-L kept'
    if r.flagN: return 'not nested'
    if r.flagM: return 'two-L, needs more regions'
    return 'clean nested two-L'
df['outcome']=df.apply(outcome,axis=1)
df.to_csv('catB_results.csv',index=False,float_format='%.5g'); pickle.dump(ZZ,open('catB_resid.pkl','wb'))
print(); print(df.outcome.value_counts()); print(df.grade.value_counts())
