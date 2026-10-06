"""Semi-Newtonian descent (2018, negative M,R) vs present CL framework with a virial window (Hz paper Eq. 19).
Models (Hz = 2.2e-18 s^-1 unless stated; Y(r)=sqrt(2GM/r)-Hz r, X=sqrt(2GM/R)-Hz R):
 S  : single-L                                     k=2
 VW : single-L + virial window for r>=r_v,  v2 = v2_L + p[Y(r)^2-Y(r_v)^2]/2  (Phi_D fixed by continuity)  k=4
 VWL: L1 + virial window [r_v,R2] + L2 reset beyond R2 (two-L Eq.26 outer form)  k=6   (2018 'triple fits')
 N18: 2018 negative-(M,R) form, outer branch everywhere, Hz=0: v2 = 3G|M|/|R| + 2G|M|/r   k=2"""
import sys,numpy as np,pandas as pd,warnings; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))); warnings.filterwarnings('ignore')
from cl_fit import G,HZ,v2_model,load_massmodels,load_table1
from cl2_fit import _metrics
from scipy.optimize import least_squares
from wave_detect import runs_p
Y=lambda M,r,Hz=HZ: np.sqrt(2*G*M/r)-Hz*r
def v2_VW(r,M,R,rv,p,Hz=HZ):
    base=v2_model(r,M,R,0,Hz); add=0.5*p*(Y(M,np.maximum(r,1e-9),Hz)**2-Y(M,rv,Hz)**2)
    return base+np.where(r>=rv,add,0.0)
def v2_VWL(r,M1,R1,rv,p,M2,R2,Hz=HZ):
    inner=v2_VW(r,M1,R1,rv,p,Hz); X2=np.sqrt(2*G*M2/R2)-Hz*R2
    outer=1.5*X2**2-Y(M2,np.maximum(r,1e-9),Hz)**2
    return np.where(r<=R2,inner,outer)
def v2_N18(r,M,R): return 3*G*M/R+2*G*M/r
def lsq(fun,y,s,starts,lo,hi):
    best=None
    for x0 in starts:
        x0=np.clip(x0,np.array(lo)*1.0001+1e-12,np.array(hi)*0.9999)
        try: sol=least_squares(lambda x:(fun(x)-y)/s,x0,bounds=(lo,hi),x_scale='jac')
        except Exception: continue
        if best is None or sol.cost<best.cost: best=sol
    J=best.jac
    try: e=np.sqrt(np.clip(np.diag(np.linalg.inv(J.T@J)),0,None))
    except np.linalg.LinAlgError: e=np.full(len(best.x),np.nan)
    return best.x,e
def fit_all(r,v,ev,triple=False):
    y,s=v**2,2*v*ev; out={}; rmax=r.max(); vmax=v.max()
    # S
    st=[[max(vmax**2*R0/(3*G),1e5)/1e9,R0] for R0 in np.geomspace(max(r.min(),.05),rmax*1.2,14)]
    x,e=lsq(lambda x:v2_model(r,x[0]*1e9,x[1]),y,s,st,[1e-5,.02],[500,100])
    m=_metrics(y,s,v2_model(r,x[0]*1e9,x[1]),2); m.update(x=x,e=e,z=(y-v2_model(r,x[0]*1e9,x[1]))/s); out['S']=m
    # N18 (Hz=0, as in 2018)
    st=[[M0,R0] for R0 in np.geomspace(.05,rmax,8) for M0 in [0.01,0.1,1,10]]
    x,e=lsq(lambda x:v2_N18(r,x[0]*1e9,x[1]),y,s,st,[1e-5,.01],[5000,500])
    m=_metrics(y,s,v2_N18(r,x[0]*1e9,x[1]),2); m.update(x=x,e=e,z=(y-v2_N18(r,x[0]*1e9,x[1]))/s); out['N18']=m
    # VW: params M,R,d=r_v-R,p
    st=[]
    for R0 in np.geomspace(max(r.min()*0.5,.03),rmax*0.6,7):
        for rv0 in np.quantile(r,[0.15,0.3,0.45,0.6,0.75]):
            for p0 in [2.5,3.5,6]:
                st.append([max(vmax**2*R0/(3*G),1e5)/1e9,R0,max(rv0-R0,0.02),p0])
    f=lambda x:v2_VW(r,x[0]*1e9,x[1],x[1]+x[2],x[3])
    x,e=lsq(f,y,s,st,[1e-5,.02,.0,-10],[5000,100,200,60])
    m=_metrics(y,s,f(x),4); m.update(x=np.array([x[0],x[1],x[1]+x[2],x[3]]),e=e,z=(y-f(x))/s); out['VW']=m
    if triple:
        f=lambda x:v2_VWL(r,x[0]*1e9,x[1],x[1]+x[2],x[3],x[4]*1e9,x[1]+x[2]+x[5])
        st=[]
        M1,R1,rv,p=out['VW']['x']
        st.append([M1,R1,max(rv-R1,1e-3),p,1e-3,rmax-rv+1.0])        # VW nested inside VWL
        for R2f in np.quantile(r,[0.4,0.55,0.7,0.85]):
            for p0 in [p,3,6]:
                for rvf in [rv, np.quantile(r,0.25)]:
                    R2=max(R2f,rvf*1.1)
                    st.append([M1,R1,max(rvf-R1,0.02),p0,max(vmax**2*R2/(3*G),1e5)/1e9,max(R2-rvf,0.05)])
        x,e=lsq(f,y,s,st,[1e-5,.02,0,-10,1e-5,.01],[5000,100,200,60,5000,300])
        m=_metrics(y,s,f(x),6); m.update(x=np.array([x[0],x[1],x[1]+x[2],x[3],x[4],x[1]+x[2]+x[5]]),e=e,z=(y-f(x))/s); out['VWL']=m
    for k in out: out[k]['p_runs']=runs_p(out[k]['z'])
    return out
