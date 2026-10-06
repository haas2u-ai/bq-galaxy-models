"""Constant-Lagrangian (CL) inflow model fits to SPARC rotation curves.
Model (de Haas, SPARC Hz fits V1, Eqs. 17-18; reduces to the 2018 RMWRSS CL model for Hz->0):
  r<=R : v^2 = 1/2 (sqrt(2GM/R) - Hz R)^2 r^2/R^2
  r> R : v^2 = 3/2 (sqrt(2GM/R) - Hz R)^2 - (sqrt(2GM/r) - Hz r)^2
  optional constant offset V0^2 (2018 paper) added to both regions.
Fit is done on v^2 with sigma(v^2) = 2 Vobs eVobs, as in the papers."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
G  = 4.30091e-6                 # kpc (km/s)^2 / Msun
HZ = 2.2e-18 * 3.085677581e16   # s^-1 -> km/s/kpc  (=0.0679)
P  = __import__('os').environ.get('SPARC_DATA', __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..','data')) + '/'

def load_table1(f=P+'Table1wbulge.mrt'):
    names=['Galaxy','T','D','eD','fD','Inc','eInc','L36','eL','Reff','SBeff','Rd','SBd',
           'MHI','RHI','Vf','eVf','Q','Ref','Lb']
    rows=[]
    for L in open(f):
        t=L.split()
        if len(t)!=len(names): continue
        try: int(t[1]); float(t[2])
        except ValueError: continue
        rows.append(t)
    df=pd.DataFrame(rows,columns=names)
    for c in names:
        if c not in('Galaxy','Ref'): df[c]=pd.to_numeric(df[c])
    return df.set_index('Galaxy')

def load_massmodels(f=P+'MassModels_Lelli2016c.mrt'):
    rows=[]
    for L in open(f):
        t=L.split()
        if len(t)!=10: continue
        try: [float(x) for x in t[1:]]
        except ValueError: continue
        rows.append(t)
    df=pd.DataFrame(rows,columns=['ID','D','R','Vobs','eV','Vgas','Vdisk','Vbul','SBd','SBb'])
    for c in df.columns[1:]: df[c]=pd.to_numeric(df[c])
    return df

def v2_model(r, M, R, V02=0.0, Hz=HZ):
    a = np.sqrt(2*G*M/R) - Hz*R
    inner = 0.5*a**2 * r**2/R**2
    outer = 1.5*a**2 - (np.sqrt(2*G*M/r) - Hz*r)**2
    return np.where(r<=R, inner, outer) + V02

def fit_galaxy(r, v, ev, offset=False, Hz=HZ):
    y, s = v**2, 2*v*ev
    def res(p):
        M, R = 10**p[0], 10**p[1]
        return (v2_model(r, M, R, p[2] if offset else 0.0, Hz) - y)/s
    best=None
    vmax=np.max(v)
    for R0 in np.geomspace(max(r.min(),0.05), r.max()*1.5, 12):
        M0 = max(vmax**2*R0/(3*G), 1e5)          # outer plateau 3GM/R ~ vmax^2
        p0=[np.log10(M0), np.log10(R0)] + ([0.0] if offset else [])
        lo=[4, np.log10(0.01)] + ([-np.inf] if offset else [])
        hi=[13.5, np.log10(200)] + ([np.inf] if offset else [])
        try: sol=least_squares(res,p0,bounds=(lo,hi),x_scale='jac')
        except Exception: continue
        if best is None or sol.cost<best.cost: best=sol
    p=best.x; k=len(p); N=len(r)
    mod=v2_model(r,10**p[0],10**p[1],p[2] if offset else 0.0,Hz)
    chi2=np.sum(((mod-y)/s)**2)
    return dict(M=10**p[0], R=10**p[1], V02=(p[2] if offset else 0.0), N=N, k=k,
                chi2=chi2, chi2nu=chi2/max(N-k,1),
                RMSrel=np.sqrt(np.mean(((y-mod)/y)**2)),
                vflat_CL=np.sqrt(max(1.5*(np.sqrt(2*G*10**p[0]/10**p[1])-Hz*10**p[1])**2,0)))
