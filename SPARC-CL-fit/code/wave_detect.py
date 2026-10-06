"""Automated detector for the double-Lagrangian 'wave' in single-L CL residuals.
Inputs: weighted residuals z_i = (v2_obs - v2_singleL)/sigma_v2, ordered by radius.
Statistics (all sign-based, so independent of error-bar scale):
  A_pmp : max over splits (i<j, each segment >= mmin points) of the fraction of residual signs
          matching the template (+...+, -...-, +...+)
  A_mpm : same for the opposite template (-, +, -)
  p_runs: Wald-Wolfowitz runs test, one-sided p for too FEW sign runs (coherent structure)
Flag as 'wave' if p_runs < p_max, A_pmp >= a_min and A_pmp > A_mpm."""
import numpy as np
from scipy import stats
def template_agreement(z, sgn=(1,-1,1), mmin=2):
    s=np.sign(z); n=len(s); best=(np.nan,0,0)
    for i in range(mmin,n-2*mmin+1):
        for j in range(i+mmin,n-mmin+1):
            ref=np.r_[np.full(i,sgn[0]),np.full(j-i,sgn[1]),np.full(n-j,sgn[2])]
            a=np.mean(s==ref)
            if not (a<=best[0]): best=(a,i,j)
    return best
def runs_p(z):
    s=np.sign(z); s=s[s!=0]; n1=(s>0).sum(); n2=(s<0).sum(); n=n1+n2
    if n1==0 or n2==0: return 1.0
    R=1+np.sum(s[1:]!=s[:-1]); mu=2*n1*n2/n+1; var=2*n1*n2*(2*n1*n2-n)/(n**2*(n-1))
    return float(stats.norm.cdf((R-mu)/np.sqrt(var))) if var>0 else 1.0
def detect(z, a_min=0.80, p_max=0.05, mmin=2):
    a,i,j=template_agreement(z,(1,-1,1),mmin); b,_,_=template_agreement(z,(-1,1,-1),mmin)
    p=runs_p(z)
    return dict(A_pmp=a,A_mpm=b,p_runs=p,i=i,j=j,wave=bool(p<p_max and a>=a_min and a>b))
