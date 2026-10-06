"""Category C analysis: single-L (S), CL + virial window (VW, continuous gauge), VW with free gauge Phi (VWf), two-L (2L).
Virial window as in de Haas (2026) JHEPGC 12:334, Eqs.(5)-(6):  v2 = 3/2 X^2 - Y^2 + p(1/2 Y^2 - Phi_p), r >= r_v.
Parametrisation used here: (M, R, r_v, M_N) with M_N = (p-2) M  (Keplerian descent mass); continuous gauge Phi_p = Y(r_v)^2/2."""
import sys,numpy as np,pandas as pd,pickle,warnings; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))); warnings.filterwarnings('ignore')
from cl_fit import G,HZ,v2_model,load_massmodels,load_table1
from cl2_fit import fit_two,_metrics
from descent import v2_VW,Y
from scipy.optimize import least_squares
from scipy import stats
from wave_detect import runs_p
mm=load_massmodels(); t=load_table1(); S=pd.read_csv('pipeline_summary.csv').set_index('Galaxy'); PF=pickle.load(open('pipeline_fits.pkl','rb'))
C=S[S.category.str.startswith('C')].index.tolist()
def mVW(r,x,Hz=HZ,phi=None):
    M,R,rv,MN=x[0],x[1],x[1]+x[2],x[3]; p=2+MN/M
    out=v2_VW(r,M*1e9,R,rv,p,Hz)
    if phi is not None: out=out+np.where(r>=rv,-p*phi+0.5*p*Y(M*1e9,rv,Hz)**2,0.0)
    return out
def jac_cov(f,x,y,s):
    x=np.asarray(x,float); J=np.zeros((len(y),len(x)))
    for i in range(len(x)):
        h=1e-6*max(abs(x[i]),1e-4); dx=np.zeros(len(x)); dx[i]=h; J[:,i]=((f(x+dx)-y)/s-(f(x-dx)-y)/s)/(2*h)
    try: return np.linalg.inv(J.T@J)
    except np.linalg.LinAlgError: return np.full((len(x),len(x)),np.nan)
LO=[1e-5,.02,0.0,-1e4]; HI=[5000,100,200,1e5]
def fitVW(r,v,ev,starts,Hz=HZ):
    y,s=v**2,2*v*ev; best=None
    for x0 in starts:
        x0=np.clip(x0,np.array(LO)+1e-9,np.array(HI)-1e-9)
        try: sol=least_squares(lambda x:(mVW(r,x,Hz)-y)/s,x0,bounds=(LO,HI),x_scale='jac')
        except Exception: continue
        if best is None or sol.cost<best.cost: best=sol
    return best.x,2*best.cost
def starts_for(r,v):
    st=[]; rmax=r.max(); vmax=v.max()
    for R0 in np.geomspace(max(r.min()*0.5,.03),rmax*0.6,7):
        for rv0 in np.quantile(r,[0.15,0.3,0.45,0.6,0.75]):
            M0=max(vmax**2*R0/(3*G),1e5)/1e9
            for pp in [1.0,2.5,3.5,6,15]:
                st.append([M0,R0,max(rv0-R0,0.02),(pp-2)*M0])
    return st
rows=[]; FITS={}
for g in C:
    d=mm[mm.ID==g]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev; n=len(r); row=t.loc[g]
    s1=PF[g]['single']; z1=PF[g]['z1']
    x,chi=fitVW(r,v,ev,starts_for(r,v)); f=lambda xx: mVW(r,xx)
    mV=_metrics(y,s,f(x),4); zV=(y-f(x))/s
    cov=jac_cov(f,x,y,s); e=np.sqrt(np.clip(np.diag(cov),0,None))
    M,Rb,rv,MN=x[0],x[1],x[1]+x[2],x[3]; p=2+MN/M
    # floor A (Hz->0 expression) and its error, v_L
    def Afun(xx): return 3*G*xx[0]*1e9/xx[1]-(2+xx[3]/xx[0])*G*xx[0]*1e9/(xx[1]+xx[2])
    gA=np.array([(Afun(x+np.eye(4)[i]*1e-6*max(abs(x[i]),1e-4))-Afun(x-np.eye(4)[i]*1e-6*max(abs(x[i]),1e-4)))/(2e-6*max(abs(x[i]),1e-4)) for i in range(4)])
    A=Afun(x); eA=np.sqrt(max(gA@cov@gA,0)) if np.all(np.isfinite(cov)) else np.nan
    erv=np.sqrt(max(cov[1,1]+cov[2,2]+2*cov[1,2],0)) if np.all(np.isfinite(cov)) else np.nan
    # free gauge variant (k=5): fit Phi as extra parameter starting at continuity
    phi0=0.5*Y(M*1e9,rv)**2
    try:
        sol=least_squares(lambda xx:(mVW(r,xx[:4],phi=xx[4])-y)/s,list(x)+[phi0],bounds=(LO+[-1e6],HI+[1e6]),x_scale='jac'); xf=sol.x
        mF=_metrics(y,s,mVW(r,xf[:4],phi=xf[4]),5)
    except Exception: mF=dict(BIC=np.nan,chi2nu=np.nan)
    # two-L comparison
    t2=fit_two(r,v,ev)
    # profile in r_v
    prof=[]; xx=x.copy()
    for rvg in np.linspace(max(Rb*1.01,r.min()),r.max()*0.98,80):
        bestc=None
        for x0 in (xx,x):
            q0=[x0[0],min(x0[1],rvg*0.99),x0[3]]
            try: so=least_squares(lambda q:(mVW(r,[q[0],q[1],rvg-q[1],q[2]])-y)/s,q0,bounds=([1e-5,.02,-1e4],[5000,rvg*0.999,1e5]),x_scale='jac')
            except Exception: continue
            if bestc is None or so.cost<bestc.cost: bestc=so
        xx=np.array([bestc.x[0],bestc.x[1],rvg-bestc.x[1],bestc.x[2]]); prof.append((rvg,2*bestc.cost,bestc.x[2]))
    P=np.array(prof); better=P[:,1].min()<chi-0.5
    if better:   # adopt profile optimum
        b=P[np.argmin(P[:,1])]; i=np.argmin(P[:,1]); 
        x,chi=fitVW(r,v,ev,[np.array([xx[0],xx[1],b[0]-xx[1],b[2]])]+[x])
        mV=_metrics(y,s,f(x),4); zV=(y-f(x))/s; cov=jac_cov(f,x,y,s); e=np.sqrt(np.clip(np.diag(cov),0,None))
        M,Rb,rv,MN=x[0],x[1],x[1]+x[2],x[3]; p=2+MN/M; A=Afun(x)
    ok=(P[:,1]-min(P[:,1].min(),chi))<=1; ib=np.argmin(np.abs(P[:,0]-rv)); lo=hi=ib
    while lo>0 and ok[lo-1]: lo-=1
    while hi<len(P)-1 and ok[hi+1]: hi+=1
    rv_lo,rv_hi=min(P[lo,0],rv),max(P[hi,0],rv); MN_lo,MN_hi=P[lo:hi+1,2].min(),P[lo:hi+1,2].max()
    # jackknife (warm)
    jk=[]
    for i in range(n):
        m=np.ones(n,bool); m[i]=False
        xj,_=fitVW(r[m],v[m],ev[m],[x]); jk.append([xj[0],xj[1],xj[1]+xj[2],xj[3],Afun(xj)])
    jk=np.array(jk); jke=np.sqrt((n-1)/n*np.sum((jk-jk.mean(0))**2,axis=0))
    # systematics: distance and inclination
    fD=1+row.eD/row.D; xD,_=fitVW(r*fD,v,ev,[x*np.array([fD,fD,fD,fD])])
    inc=np.radians(row.Inc); i2=min(inc+np.radians(row.eInc),np.radians(89.9)); fi=np.sin(inc)/np.sin(i2)
    xI,_=fitVW(r,v*fi,ev*fi,[x*np.array([fi**2,1,1,fi**2])])
    vec=lambda xx: np.array([xx[0],xx[1],xx[1]+xx[2],xx[3],Afun(xx)])
    sysv=np.hypot(vec(xD)-vec(x),vec(xI)-vec(x))
    x0,_=fitVW(r,v,ev,[x],Hz=0.0); hz=np.abs(vec(x0)/vec(x)-1)
    nin=int(np.sum(r<rv)); nwin=n-nin
    Mbar=(0.5*(row.L36-row.Lb)+0.7*row.Lb+1.33*row.MHI)
    rows.append(dict(Galaxy=g,T=int(row['T']),Q=int(row.Q),Inc=row.Inc,n=n,A_mpm=S.loc[g,'A_mpm'],p_runs1=S.loc[g,'p_runs'],
        chi2_S=s1['chi2'],chi2nu_S=s1['chi2nu'],BIC_S=s1['BIC'],RMS_S=s1['RMSrel'],DW_S=np.sum(np.diff(z1)**2)/np.sum(z1**2),
        M=M,eM=e[0],R=Rb,eR=e[1],rv=rv,erv=erv,rv_lo=rv_lo,rv_hi=rv_hi,MN=MN,eMN=e[3],MN_lo=MN_lo,MN_hi=MN_hi,p=p,A=A,eA=eA,
        rho_M_MN=cov[0,3]/(e[0]*e[3]) if e[0]>0 and e[3]>0 else np.nan,
        jM=jke[0],jR=jke[1],jrv=jke[2],jMN=jke[3],jA=jke[4],sM=sysv[0],sR=sysv[1],srv=sysv[2],sMN=sysv[3],sA=sysv[4],hz_max=hz.max(),
        n_in=nin,n_win=nwin,chi2_VW=mV['chi2'],chi2nu_VW=mV['chi2nu'],BIC_VW=mV['BIC'],AIC_VW=mV['AIC'],RMS_VW=mV['RMSrel'],
        pruns_VW=runs_p(zV),DW_VW=np.sum(np.diff(zV)**2)/np.sum(zV**2),pSW_VW=stats.shapiro(zV).pvalue,Plow_VW=stats.chi2.cdf(mV['chi2'],n-4),maxz_VW=np.abs(zV).max(),
        BIC_VWf=mF['BIC'],chi2nu_VWf=mF['chi2nu'],BIC_2L=t2['BIC'],chi2nu_2L=t2['chi2nu'],profile_better=better,
        Vflat=row.Vf,Mbar=Mbar,Rdisk=row.Rd,RHI=row.RHI,Lb=row.Lb,L36=row.L36))
    FITS[g]=dict(x=x,cov=cov,zS=z1,zV=zV,prof=P,two=t2)
    print(g,end=' ',flush=True)
df=pd.DataFrame(rows)
df['dBIC_VW']=df.BIC_VW-df.BIC_S; df['dBIC_VWf']=df.BIC_VWf-df.BIC_VW; df['dBIC_2L_VW']=df.BIC_2L-df.BIC_VW
df.to_csv('catC_results.csv',index=False,float_format='%.5g'); pickle.dump(FITS,open('catC_fits.pkl','wb'))
print(); print(df[['Galaxy','n','chi2nu_S','chi2nu_VW','dBIC_VW','p','rv','MN','A','dBIC_2L_VW','dBIC_VWf','profile_better']].round(2).to_string())
