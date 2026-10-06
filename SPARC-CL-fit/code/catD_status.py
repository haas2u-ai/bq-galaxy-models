import pandas as pd,numpy as np,pickle
df=pd.read_csv('catD_results.csv'); B=pickle.load(open('catD_best.pkl','rb'))
def status(r):
    if r.best=='S': return 'single-L adequate'
    b=B[r.Galaxy]; x=b['x']; pv=b['pv']; R=b['rad'][0]
    if R<=0.025 or any(abs(v)>100 for v in pv.values()): return 'phenomenological'
    if not np.isfinite(r.relerr_max) or r.relerr_max>0.6: return 'weakly constrained'
    if r.flagM: return 'improved, still chi2nu>2'
    return 'clean'
df['status']=df.apply(status,axis=1)
df.to_csv('catD_results.csv',index=False,float_format='%.5g')
TY={0:'S0',1:'Sa',2:'Sab',3:'Sb',4:'Sbc',5:'Sc',6:'Scd',7:'Sd',8:'Sdm',9:'Sm',10:'Im',11:'BCD'}
for s,gg in df.groupby('status'): print(s, [(g,TY[t],b,round(rx,2)) for g,t,b,rx in zip(gg.Galaxy,gg['T'],gg.best,gg.chi2nu_best)])
