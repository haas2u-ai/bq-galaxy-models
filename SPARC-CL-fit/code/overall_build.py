"""Overall synthesis of the A-E reports. No refitting: only the stored results (CSV/pickles) are combined."""
import sys,numpy as np,pandas as pd; sys.path.insert(0,'.')
from cl_fit import G,HZ,load_massmodels,load_table1
mm=load_massmodels(); t=load_table1(); rmin=mm.groupby('ID').R.min()
P=pd.read_csv('pipeline_summary.csv').set_index('Galaxy')
A=pd.read_csv('catA_results.csv'); B=pd.read_csv('catB_results.csv'); C=pd.read_csv('catC_results.csv'); D=pd.read_csv('catD_results.csv'); E=pd.read_csv('catE_results.csv')
vL=lambda M,R: np.sqrt(1.5)*(np.sqrt(2*G*M*1e9/R)-HZ*R)
rows=[]
def add(g,src,cls,sub,chiS,chiB,rmsS,rmsB,prB,dBIC,qual,**kw):
    rows.append(dict(Galaxy=g,src=src,cls=cls,sub=sub,chi2nu_S=chiS,chi2nu_final=chiB,RMS_S=rmsS,RMS_final=rmsB,pruns_final=prB,dBIC=dBIC,quality=qual,**kw))
for _,r in A.iterrows():
    add(r.Galaxy,'A','Single-L','adequate',r.chi2nu,r.chi2nu,r.RMSrel,r.RMSrel,r.p_runs,0,'grade '+r.grade,R=r.R,M=r.M,v_asym=r.vflat_CL)
for _,r in B.iterrows():
    if r.outcome=='single-L kept':
        add(r.Galaxy,'B','Single-L','weak +-+ wave' if r.dBIC>=-2 else 'undecided (-6<=dBIC<-2)',r.chi2nu1,r.chi2nu1,r.RMS1,r.RMS1,r.p_runs1,0,'structured residuals',R=r.R,M=r.M,v_asym=vL(r.M,r.R))
    elif r.outcome=='not nested':
        add(r.Galaxy,'B','Unresolved (bulge / non-nested)','two-L non-nested',r.chi2nu1,r.chi2nu2,r.RMS1,r.RMS2,r.p_runs2,r.dBIC,'flag N')
    else:
        add(r.Galaxy,'B','Double (two-L reset)','nested two-L' if r.outcome=='clean nested two-L' else 'two-L, needs more regions',r.chi2nu1,r.chi2nu2,r.RMS1,r.RMS2,r.p_runs2,r.dBIC,'grade '+r.grade,
            R=r.R1,M=r.M1,R2=r.R2,M2=r.M2,v_asym=r.vL2)
for _,r in C.iterrows():
    if r.outcome=='single-L kept':
        add(r.Galaxy,'C','Single-L','weak -+- wave',r.chi2nu_S,r.chi2nu_S,r.RMS_S,r.RMS_S,r.p_runs1,0,'structured residuals',R=P.loc[r.Galaxy,'R'],M=P.loc[r.Galaxy,'M'],v_asym=vL(P.loc[r.Galaxy,'M'],P.loc[r.Galaxy,'R']))
    elif r.outcome=='descent window':
        only=(r.R<rmin[r.Galaxy] and r.p>=30) or r.n_in<=2
        add(r.Galaxy,'C','Descent only' if only else 'Single-L + descent','descent-dominated' if only else 'CL + Kepler descent',r.chi2nu_S,r.chi2nu_VW,r.RMS_S,r.RMS_VW,r.pruns_VW,r.dBIC_VW,'grade '+r.grade,
            R=r.R,M=r.M,rv=r.rv,MN=r.MN,p=r.p,Afloor=r.A,v_asym=np.sqrt(r.A))
    elif r.outcome=='flattening window':
        add(r.Galaxy,'C','Flattening window','0<p<2',r.chi2nu_S,r.chi2nu_VW,r.RMS_S,r.RMS_VW,r.pruns_VW,r.dBIC_VW,'--',R=r.R,M=r.M,rv=r.rv,p=r.p)
    else:
        add(r.Galaxy,'C','Double (two-L reset)','rising window (p<0)',r.chi2nu_S,r.chi2nu_VW,r.RMS_S,r.RMS_VW,r.pruns_VW,r.dBIC_VW,'rising window',R=r.R,M=r.M,rv=r.rv,p=r.p)
for _,r in D.iterrows():
    pv=dict(kv.split('=') for kv in r.pvals.split(';')) if isinstance(r.pvals,str) else {}
    rad=dict(kv.split('=') for kv in r.radii.split(';'))
    base=dict(R=float(list(rad.values())[0]))
    if r.status=='single-L adequate': cls,sub='Single-L','structure within errors'
    elif r.status=='phenomenological': cls,sub='Unresolved (bulge / non-nested)','extreme window parameters'
    elif r.best=='VW':
        p=float(pv['p'])
        if p<0: cls,sub='Double (two-L reset)','rising window (p<0)'
        else:
            cls,sub=('Descent only','descent from first points') if r.Galaxy=='NGC7814' else ('Single-L + descent','CL + Kepler descent')
        base.update(p=p,rv=float(rad['r_v']))
    else: cls,sub='Multi-region',{'VW2':'two windows','2LVW':'reset + disk window'}[r.best]
    add(r.Galaxy,'D',cls,sub,r.chi2nu_S,r.chi2nu_best,r.RMS_S,r.RMS_best,r.pruns_best,r.dBIC_best,r.status,**base)
for _,r in E.iterrows():
    k=r.kind
    if k=='single-L kept': cls,sub='Single-L','extra scatter (sigma_int)'
    elif k.startswith('rising'): cls,sub='Double (two-L reset)','rising window (p<0)'
    elif k.startswith('descent'): cls,sub='Single-L + descent','CL + Kepler descent'
    else: cls,sub=('Unresolved (bulge / non-nested)','extreme window parameters') if r.status=='phenomenological' else ('Multi-region','two windows')
    add(r.Galaxy,'E',cls,sub,r.chi2nu_S,r.chi2nu_best,r.RMS_S,r.RMS_best,r.pruns_best if np.isfinite(r.pruns_best) else r.p_runs1,r.dBIC_best,r.status,
        R=r.R,M=r.M,sigma_int=r.sigma_int)
O=pd.DataFrame(rows)
assert len(O)==175 and O.Galaxy.is_unique, (len(O),O.Galaxy.duplicated().sum())
O['T']=O.Galaxy.map(t['T']); O['Q']=O.Galaxy.map(t.Q); O['n']=O.Galaxy.map(P.n); O['Inc']=O.Galaxy.map(t.Inc)
O['Vflat']=O.Galaxy.map(t.Vf); O['Rdisk']=O.Galaxy.map(t.Rd); O['Lb']=O.Galaxy.map(t.Lb)
O['Mbar']=O.Galaxy.map((0.5*(t.L36-t.Lb)+0.7*t.Lb+1.33*t.MHI))
O['tentative']=O.n<=13
O.to_csv('overall_classes.csv',index=False,float_format='%.5g')
print(O.cls.value_counts()); print(pd.crosstab(O.cls,O.src,margins=True))
print(O.groupby(['cls','sub']).size())
