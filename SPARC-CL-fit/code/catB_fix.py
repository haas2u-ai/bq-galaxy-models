import sys,numpy as np,pandas as pd,pickle,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from cl_fit import load_massmodels
from cl2_fit import v2_two,_metrics,_errors
from scipy.optimize import least_squares
mm=load_massmodels(); PF=pickle.load(open('pipeline_fits.pkl','rb')); prof=pickle.load(open('catB_profiles.pkl','rb'))
df=pd.read_csv('catB_results.csv')
for g in df[df.prof_min_better==True].Galaxy:
    d=mm[mm.ID==g]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev
    P=prof[g]; b=P[np.argmin(P[:,1])]; R2,M1,R1,M2=b[0],b[2],b[3],b[4]
    mod=lambda x: v2_two(r,x[0]*1e9,x[1],x[2]*1e9,x[1]+x[3])
    sol=least_squares(lambda x:(mod(x)-y)/s,[M1,R1,M2,R2-R1],bounds=([1e-5,.02,1e-5,.01],[500,100,5000,200]),x_scale='jac')
    x=sol.x; out=_metrics(y,s,mod(x),4); e=_errors(sol,4)
    out.update(params=np.array([x[0],x[1],x[2],x[1]+x[3]]),errors=np.array([e[0],e[1],e[2],np.nan]))
    print(g,"old chi2",PF[g]['two']['chi2'],"new",out['chi2'],out['params'])
    if out['chi2']<PF[g]['two']['chi2']: PF[g]['two']=out
pickle.dump(PF,open('pipeline_fits.pkl','wb'))
