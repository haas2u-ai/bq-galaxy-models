import sys,numpy as np,pandas as pd,pickle,warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from cl_fit import load_massmodels
from cl2_fit import v2_two
from scipy.optimize import least_squares
mm=load_massmodels(); df=pd.read_csv('catB_results.csv')
prof={}
for i,row in df[df.accepted].iterrows():
    d=mm[mm.ID==row.Galaxy]; r,v,ev=d.R.values,d.Vobs.values,d.eV.values; y,s=v**2,2*v*ev
    q=np.array([row.M1,row.R1,row.M2]); chi0=row.chi2_2
    grid=np.unique(np.r_[np.linspace(max(row.R1*1.05,r.min()),r.max()*0.98,140),row.R2])
    out=[]
    x=q.copy()
    for R2 in grid:
        best=None
        for x0 in (x,q):
            if x0[1]>=R2: x0=np.array([x0[0],0.8*R2,x0[2]])
            try: sol=least_squares(lambda p:(v2_two(r,p[0]*1e9,p[1],p[2]*1e9,R2)-y)/s,x0,bounds=([1e-5,0.02,1e-5],[500,R2*0.999,5000]),x_scale='jac')
            except Exception: continue
            if best is None or sol.cost<best.cost: best=sol
        x=best.x; out.append((R2,2*best.cost,*best.x))
    P=np.array(out); dchi=P[:,1]-min(P[:,1].min(),chi0)
    ok=dchi<=1.0
    # contiguous interval containing best
    ib=np.argmin(np.abs(P[:,0]-row.R2)); lo=ib; hi=ib
    while lo>0 and ok[lo-1]: lo-=1
    while hi<len(P)-1 and ok[hi+1]: hi+=1
    prof[row.Galaxy]=P
    df.loc[i,'R2_lo']=P[lo,0]; df.loc[i,'R2_hi']=P[hi,0]; df.loc[i,'M2_lo']=P[lo:hi+1,4].min(); df.loc[i,'M2_hi']=P[lo:hi+1,4].max()
    df.loc[i,'prof_min_better']=P[:,1].min()<chi0-0.5
    df.loc[i,'n_islands']=int(np.sum(np.diff(ok.astype(int))==1)+(ok[0]))
df['eR2p']=np.maximum(df.eR2,(df.R2_hi-df.R2_lo)/2); df['eM2p']=np.maximum(df.eM2,(df.M2_hi-df.M2_lo)/2)
df['relR2']=df.eR2p/df.R2; df['relM2']=df.eM2p/df.M2
# jackknife shift relative to profile-aware errors: use jackknife errors instead
df['jk_rel_max']=np.nanmax(np.c_[df.jM1/df.M1,df.jR1/df.R1,df.jM2/df.M2,df.jR2/df.R2],axis=1)
def grade(r):
    if not r.accepted: return '--'
    if r.flagN: return 'N'
    if max(r.relM1,r.relR1,r.relM2,r.relR2)<=0.3 and r.n_bulge>=2 and r.n_bar>=2 and r.n_disk>=3 and r.jk_rel_max<=0.3: g='I'
    elif r.relvL1<=0.10 and r.relvL2<=0.10: g='II'
    else: g='III'
    return g+('M' if r.flagM else '')
df['grade']=df.apply(grade,axis=1)
df.to_csv('catB_results.csv',index=False,float_format='%.5g'); pickle.dump(prof,open('catB_profiles.pkl','wb'))
print(df[df.accepted][['Galaxy','R2','eR2','R2_lo','R2_hi','M2','eM2','M2_lo','M2_hi','n_islands','prof_min_better','jk_rel_max','grade']].round(3).to_string())
print(df.grade.value_counts())
