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
