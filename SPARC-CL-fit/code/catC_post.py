import pandas as pd,numpy as np
from scipy import stats
df=pd.read_csv('catC_results.csv')
df['erv_p']=np.maximum(df.erv.fillna(0),(df.rv_hi-df.rv_lo)/2)
def cls(r):
    if r.dBIC_VW>=-6: return 'single-L kept'
    if r.dBIC_2L_VW<-2: return 'two-L preferred'
    if r.p>2: return 'descent window'
    if r.p>=0: return 'flattening window'
    return 'rising window'
df['outcome']=df.apply(cls,axis=1); df['flagM']=(df.outcome!='single-L kept')&(df.chi2nu_VW>2)
for c,e in [('R','eR'),('rv','erv_p'),('MN','eMN'),('A','eA'),('M','eM')]: df['rel'+c]=np.abs(df[e]/df[c])
df['jk_rel']=np.nanmax(np.c_[df.jR/df.R,df.jrv/df.rv,np.abs(df.jMN/df.MN),df.jA/df.A],axis=1)
def grade(r):
    if r.outcome!='descent window': return '--'
    if max(r.relR,r.relrv,r.relMN,r.relA)<=0.3 and r.n_in>=3 and r.n_win>=3 and r.jk_rel<=0.3: g='I'
    elif r.relA<=0.10 and r.relMN<=0.5: g='II'
    else: g='III'
    return g+('M' if r.flagM else '')
df['grade']=df.apply(grade,axis=1)
df.to_csv('catC_results.csv',index=False,float_format='%.5g')
pd.set_option('display.width',250)
print(df.outcome.value_counts()); print(df.grade.value_counts())
d=df[df.outcome=='descent window']; k=df[df.outcome=='single-L kept']
print(d[['Galaxy','T','n','grade','relR','relrv','relMN','relA','relM','jk_rel','n_in','n_win','chi2nu_VW','pruns_VW','rho_M_MN','hz_max']].round(2).to_string())
print("desc: chi2nu S,VW",d.chi2nu_S.median().round(2),d.chi2nu_VW.median().round(2)," RMS",d.RMS_S.median().round(3),d.RMS_VW.median().round(3)," DW",d.DW_S.median().round(2),d.DW_VW.median().round(2),
      " pruns<.05",(d.pruns_VW<.05).sum(),len(d)," Plow<.01",(d.Plow_VW<.01).sum()," SW<.05",(d.pSW_VW<.05).sum()," maxz>3",(d.maxz_VW>3).sum())
print("kept: chi2nu",k.chi2nu_S.median().round(2)," n",k.n.median()," T",k['T'].value_counts().sort_index().to_dict())
print("desc T",d['T'].value_counts().sort_index().to_dict(),"Q",d.Q.value_counts().to_dict())
print("rel med R rv MN A M",[round(np.median(d[c]),3) for c in ['relR','relrv','relMN','relA','relM']], " rho_M_MN med",d.rho_M_MN.median().round(3))
print("sys/stat: rv",np.median(d.srv/d.erv_p).round(2)," MN",np.median(d.sMN/d.eMN).round(2)," A",np.median(d.sA/d.eA).round(2), " hz max",d.hz_max.max().round(3))
print("rv",np.percentile(d.rv,[16,50,84]).round(1)," rv/R",np.percentile(d.rv/d.R,[16,50,84]).round(1)," rv/Rdisk",np.percentile(d.rv/d.Rdisk,[16,50,84]).round(2), stats.spearmanr(d.rv,d.Rdisk)[0].round(2),
      " rv/RHI",np.percentile(d.rv/d.RHI.replace(0,np.nan),[16,50,84]).round(2))
ok=d.Vflat>0; print("sqrt(A)/Vflat",np.percentile(np.sqrt(d.A[ok])/d.Vflat[ok],[16,50,84]).round(3),stats.pearsonr(np.sqrt(d.A[ok]),d.Vflat[ok])[0].round(3),ok.sum())
print("MN/Mbar",np.percentile(d.MN/d.Mbar,[16,50,84]).round(2),stats.pearsonr(np.log10(d.MN),np.log10(d.Mbar))[0].round(2), " MN/Mbulge(0.7Lb) for Lb>0:", np.percentile((d.MN/(0.7*d.Lb))[d.Lb>0],[16,50,84]).round(2),(d.Lb>0).sum())
print("p",np.percentile(d.p,[16,50,84]).round(1)," p<=5:",(d.p<=5).sum())
print("VWf better (<-6):",(df.dBIC_VWf<-6).sum(), "min",df.dBIC_VWf.min())
print("2L comparison accepted: dBIC_2L_VW min",df[df.outcome!='single-L kept'].dBIC_2L_VW.min())
print(df[df.outcome.isin(['flattening window','rising window','two-L preferred'])][['Galaxy','p','dBIC_VW','dBIC_2L_VW']])
print(df[df.flagM][['Galaxy','chi2nu_VW','T']])
