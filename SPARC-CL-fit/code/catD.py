"""Category D: refit with single-L (S), CL+virial window (VW), two-L (2L), two virial windows (VW2, [1] Eqs.5-7),
two-L with a virial window in the disk (2LVW). Windows use continuous gauges and the descent-mass parametrisation M_N=(p-2)M."""
import sys,numpy as np,pandas as pd,pickle,warnings; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))); warnings.filterwarnings('ignore')
from cl_fit import G,HZ,v2_model,load_massmodels,load_table1
from cl2_fit import v2_two,fit_two,_metrics
from descent import Y
from scipy.optimize import least_squares
from scipy import stats
from wave_detect import runs_p
def base(r,M,R,Hz=HZ): return v2_model(r,M*1e9,R,0,Hz)
def win(r,M,rv,p,Hz=HZ):  # continuous-gauge window term
    return 0.5*p*(Y(M*1e9,np.maximum(r,1e-9),Hz)**2-Y(M*1e9,rv,Hz)**2)
def m_S(r,x,Hz=HZ): return base(r,x[0],x[1],Hz)
def m_VW(r,x,Hz=HZ):
    M,R,rv,p=x[0],x[1],x[1]+x[2],2+x[3]/x[0]; return base(r,M,R,Hz)+np.where(r>=rv,win(r,M,rv,p,Hz),0)
def m_2L(r,x,Hz=HZ): return v2_two(r,x[0]*1e9,x[1],x[2]*1e9,x[1]+x[3],0,Hz)
def m_VW2(r,x,Hz=HZ):
    M,R=x[0],x[1]; rv1=R+x[2]; p=2+x[3]/M; rv2=rv1+x[4]; q=2+x[5]/M
    w1=win(r,M,rv1,p,Hz); off=win(np.array([rv2]),M,rv1,p,Hz)[0]      # value of window 1 at rv2
    w2=win(r,M,rv2,q,Hz)+off
    return base(r,M,R,Hz)+np.where(r>=rv2,w2,np.where(r>=rv1,w1,0))
def m_2LVW(r,x,Hz=HZ):
    M1,R1,M2,R2=x[0],x[1],x[2],x[1]+x[3]; rv=R2+x[4]; p=2+x[5]/M2
    v=v2_two(r,M1*1e9,R1,M2*1e9,R2,0,Hz); return v+np.where(r>=rv,win(r,M2,rv,p,Hz),0)
MODELS={'S':(m_S,[1e-5,.02],[5000,100]),'VW':(m_VW,[1e-5,.02,0,-1e4],[5000,100,200,1e5]),'2L':(m_2L,[1e-5,.02,1e-5,.01],[5000,100,5000,200]),
        'VW2':(m_VW2,[1e-5,.02,0,-1e4,.01,-1e4],[5000,100,200,1e5,200,1e5]),'2LVW':(m_2LVW,[1e-5,.02,1e-5,.01,0,-1e4],[5000,100,5000,200,200,1e5])}
def lsq(name,r,y,s,starts,Hz=HZ):
    f,lo,hi=MODELS[name]; best=None
    for x0 in starts:
        x0=np.clip(np.asarray(x0,float),np.array(lo)+1e-9,np.array(hi)-1e-9)
        try: sol=least_squares(lambda x:(f(r,x,Hz)-y)/s,x0,bounds=(lo,hi),x_scale='jac')
        except Exception: continue
        if best is None or sol.cost<best.cost: best=sol
    return best
if __name__=='__main__':
    mm=load_massmodels(); t=load_table1(); S=pd.read_csv('pipeline_summary.csv').set_index('Galaxy'); PF=pickle.load(open('pipeline_fits.pkl','rb'))
    Dg=S[S.category.str.startswith('D')].index.tolist()
    rows=[]; FITS={}
    for g in Dg:
        d=mm[mm.ID==g]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev; n=len(r); rmax=r.max(); vmax=v.max()
        res={}
        x=PF[g]['single']['params']; sol=lsq('S',r,y,s,[x]); res['S']=sol
        st=[]
        for R0 in np.geomspace(max(r.min()*0.5,.03),rmax*0.6,6):
            M0=max(vmax**2*R0/(3*G),1e5)/1e9
            for rq in [.15,.3,.45,.6,.75]:
                for pp in [1,2.5,4,8,20]: st.append([M0,R0,max(np.quantile(r,rq)-R0,.02),(pp-2)*M0])
        res['VW']=lsq('VW',r,y,s,st)
        t2=fit_two(r,v,ev); q=t2['params']; xs=res['S'].x; res['2L']=lsq('2L',r,y,s,[[q[0],q[1],q[2],q[3]-q[1]],[xs[0],xs[1],xs[0],rmax+1-xs[1]]])
        xv=res['VW'].x; st=[np.r_[xv,rmax+1,0]]
        for rq1 in [.15,.3,.45]:
            for rq2 in [.55,.7,.85]:
                for p1 in [-5,1,4,10]:
                    for p2 in [-5,1,4,10]:
                        M0,R0=xv[0],xv[1]; rv1=max(np.quantile(r,rq1),R0*1.05); rv2=max(np.quantile(r,rq2),rv1*1.05)
                        st.append([M0,R0,rv1-R0,(p1-2)*M0,rv2-rv1,(p2-2)*M0])
        res['VW2']=lsq('VW2',r,y,s,st)
        x2=res['2L'].x; R2=x2[1]+x2[3]; st=[np.r_[x2,rmax+1,0],[xv[0],xv[1],xv[0],0.0101,max(xv[2]-0.0101,0),xv[3]]]
        for rq in [.4,.55,.7,.85]:
            for pp in [-5,1,4,10,30]:
                rv=max(np.quantile(r,rq),R2*1.02); st.append(np.r_[x2,rv-R2,(pp-2)*x2[2]])
        res['2LVW']=lsq('2LVW',r,y,s,st)
        rec=dict(Galaxy=g,T=int(t.loc[g,'T']),Q=int(t.loc[g,'Q']),Inc=t.loc[g,'Inc'],n=n,signs=S.loc[g,'signs'],A_pmp=S.loc[g,'A_pmp'],A_mpm=S.loc[g,'A_mpm'],p_runs1=S.loc[g,'p_runs'],
                 Vflat=t.loc[g,'Vf'],Rdisk=t.loc[g,'Rd'],Lb=t.loc[g,'Lb'],Mbar=(0.5*(t.loc[g,'L36']-t.loc[g,'Lb'])+0.7*t.loc[g,'Lb']+1.33*t.loc[g,'MHI']))
        fit={}
        for k_,sol in res.items():
            if len(sol.x)>=n-1:      # not enough points
                rec.update({f'chi2_{k_}':np.nan,f'chi2nu_{k_}':np.nan,f'BIC_{k_}':np.nan,f'RMS_{k_}':np.nan,f'pruns_{k_}':np.nan,f'DW_{k_}':np.nan}); continue
            f=MODELS[k_][0]; mod=f(r,sol.x); m=_metrics(y,s,mod,len(sol.x)); z=(y-mod)/s
            rec.update({f'chi2_{k_}':m['chi2'],f'chi2nu_{k_}':m['chi2nu'] if n>len(sol.x) else np.nan,f'BIC_{k_}':m['BIC'],f'RMS_{k_}':m['RMSrel'],f'pruns_{k_}':runs_p(z),f'DW_{k_}':np.sum(np.diff(z)**2)/np.sum(z**2)})
            fit[k_]=dict(x=sol.x,z=z,jac=sol.jac)
        rows.append(rec); FITS[g]=fit; print(g,end=' ',flush=True)
    df=pd.DataFrame(rows); df.to_csv('catD_results.csv',index=False,float_format='%.5g'); pickle.dump(FITS,open('catD_fits.pkl','wb'))
    pd.set_option('display.width',250)
    print(); print(df[['Galaxy','n']+[f'chi2nu_{k}' for k in MODELS]+[f'BIC_{k}' for k in MODELS]].round(2).to_string())
