import sys,numpy as np,pandas as pd,pickle,warnings,itertools; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))); warnings.filterwarnings('ignore')
from catD import MODELS,lsq,G,HZ
from cl_fit import load_massmodels,load_table1
from cl2_fit import fit_two,_metrics
from scipy.optimize import least_squares,brentq
from scipy import stats
from wave_detect import runs_p,detect
mm=load_massmodels(); t=load_table1(); S=pd.read_csv('pipeline_summary.csv').set_index('Galaxy'); PF=pickle.load(open('pipeline_fits.pkl','rb'))
Eg=S[S.category.str.startswith('E')].index.tolist()
def min_runs_p(n):
    """smallest attainable one-sided runs-test p for n residuals (best case: 2 runs, balanced split)"""
    best=1
    for n1 in range(1,n):
        s=np.r_[np.ones(n1),-np.ones(n-n1)]; best=min(best,runs_p(s))
    return best
rows=[]; FITS={}
for g in Eg:
    d=mm[mm.ID==g]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev; n=len(r); rmax=r.max(); vmax=v.max(); row=t.loc[g]
    res={}; xs=PF[g]['single']['params']; res['S']=lsq('S',r,y,s,[xs])
    if n>=6:
        st=[[max(vmax**2*R0/(3*G),1e5)/1e9,R0,max(np.quantile(r,rq)-R0,.02),(pp-2)*max(vmax**2*R0/(3*G),1e5)/1e9]
            for R0 in np.geomspace(max(r.min()*0.5,.03),rmax*0.6,6) for rq in [.2,.4,.6,.8] for pp in [1,2.5,4,8,20]]
        res['VW']=lsq('VW',r,y,s,st)
        t2=fit_two(r,v,ev); q=t2['params']; xS=res['S'].x
        res['2L']=lsq('2L',r,y,s,[[q[0],q[1],q[2],q[3]-q[1]],[xS[0],xS[1],xS[0],rmax+1-xS[1]]])
    if n>=10:
        xv=res['VW'].x; st=[np.r_[xv,rmax+1,0]]
        for rq1,rq2,p1,p2 in itertools.product([.2,.4],[.6,.8],[-5,1,4,10],[-5,1,4,10]):
            rv1=max(np.quantile(r,rq1),xv[1]*1.05); rv2=max(np.quantile(r,rq2),rv1*1.05)
            st.append([xv[0],xv[1],rv1-xv[1],(p1-2)*xv[0],rv2-rv1,(p2-2)*xv[0]])
        res['VW2']=lsq('VW2',r,y,s,st)
    rec=dict(Galaxy=g,T=int(row['T']),Q=int(row.Q),Inc=row.Inc,eInc=row.eInc,D=row.D,eD=row.eD,fD=int(row.fD),n=n,signs=S.loc[g,'signs'],p_runs1=S.loc[g,'p_runs'],
             min_runs_p=min_runs_p(n),Vflat=row.Vf,Mbar=(0.5*(row.L36-row.Lb)+0.7*row.Lb+1.33*row.MHI),Rdisk=row.Rd,
             mean_relerr=np.mean(ev/v),med_eV=np.median(ev))
    fit={}
    for k,sol in res.items():
        mod=MODELS[k][0](r,sol.x); m=_metrics(y,s,mod,len(sol.x)); z=(y-mod)/s
        rec.update({f'chi2_{k}':m['chi2'],f'chi2nu_{k}':m['chi2nu'],f'BIC_{k}':m['BIC'],f'RMS_{k}':m['RMSrel'],f'pruns_{k}':runs_p(z)})
        fit[k]=dict(x=sol.x,z=z,jac=sol.jac)
    for k in ['VW','2L','VW2']:
        if k not in res: rec.update({f'chi2_{k}':np.nan,f'chi2nu_{k}':np.nan,f'BIC_{k}':np.nan,f'RMS_{k}':np.nan,f'pruns_{k}':np.nan})
    # ---- scatter / outlier analysis on single-L
    x=res['S'].x; z=fit['S']['z']; mod=MODELS['S'][0](r,x)
    rec.update(DW_S=np.sum(np.diff(z)**2)/np.sum(z**2),pSW_S=stats.shapiro(z).pvalue if n>=3 else np.nan,maxz_S=np.abs(z).max(),n_out3=int(np.sum(np.abs(z)>3)),
               n_out2=int(np.sum(np.abs(z)>2)),frac_chi2_worst=np.max(z**2)/np.sum(z**2),
               P_high=1-stats.chi2.cdf(np.sum(z**2),n-2))
    # intrinsic velocity scatter sigma_int (km/s) added in quadrature to eV, refitting
    def chi2nu_with(si):
        s2=2*v*np.sqrt(ev**2+si**2); so=lsq('S',r,y,s2,[x]); return np.sum(((y-MODELS['S'][0](r,so.x))/s2)**2)/(n-2)-1, so.x
    try:
        sint=brentq(lambda si: chi2nu_with(si)[0],0,200); xint=chi2nu_with(sint)[1]
    except Exception: sint=np.nan; xint=x
    rec.update(sigma_int=sint,sigma_int_rel=sint/np.median(v),M_sint=xint[0],R_sint=xint[1])
    # drop worst point
    i=np.argmax(np.abs(z)); mk=np.ones(n,bool); mk[i]=False
    so=lsq('S',r[mk],y[mk],s[mk],[x]); zz=(y[mk]-MODELS['S'][0](r[mk],so.x))/s[mk]
    rec.update(chi2nu_drop1=np.sum(zz**2)/(n-3),r_worst=r[i],z_worst=z[i],M_drop1=so.x[0],R_drop1=so.x[1])
    # robust (soft_l1) fit
    sr=least_squares(lambda xx:(MODELS['S'][0](r,xx)-y)/s,x,bounds=([1e-5,.02],[5000,100]),loss='soft_l1',f_scale=1.0)
    rec.update(M_rob=sr.x[0],R_rob=sr.x[1])
    # covariance on S
    try: cov=np.linalg.inv(res['S'].jac.T@res['S'].jac); e=np.sqrt(np.diag(cov))
    except Exception: e=[np.nan,np.nan]
    rec.update(M=x[0],eM=e[0],R=x[1],eR=e[1])
    # systematics for S
    fDf=1+row.eD/row.D; xD=lsq('S',r*fDf,y,s,[x]).x
    inc=np.radians(row.Inc); i2=min(inc+np.radians(row.eInc),np.radians(89.9)); fi=(np.sin(inc)/np.sin(i2))**2; xI=lsq('S',r,y*fi,s*fi,[x]).x
    rec.update(sM=np.hypot(xD[0]-x[0],xI[0]-x[0]),sR=np.hypot(xD[1]-x[1],xI[1]-x[1]))
    rows.append(rec); FITS[g]=fit; print(g,end=' ',flush=True)
df=pd.DataFrame(rows); df.to_csv('catE_results.csv',index=False,float_format='%.5g'); pickle.dump(FITS,open('catE_fits.pkl','wb'))
pd.set_option('display.width',260); print()
print(df[['Galaxy','n','min_runs_p','chi2nu_S','chi2nu_VW','chi2nu_2L','chi2nu_VW2','BIC_S','BIC_VW','BIC_2L','BIC_VW2','sigma_int','sigma_int_rel','n_out3','n_out2','frac_chi2_worst','chi2nu_drop1','P_high','pSW_S']].round(3).to_string())
