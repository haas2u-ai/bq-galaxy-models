"""Extract every goodness-of-fit row (k, chi2, AIC, BIC, RMS) from the text of
"Galactic Rotation Curves and the Constant-Lagrangian Field" (SPARC fits paper).
Usage:  python parse_fits.py paper_text.txt   ->  data/fit_rows.json
The paper text can be obtained with  pdftotext -layout paper.pdf paper_text.txt"""
import re, json, sys
from paths import HERE
L = [l.replace('\r','').rstrip() for l in open(sys.argv[1] if len(sys.argv) > 1 else 'sparc.txt', encoding='utf-8', errors='ignore')]
num = r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'
rows = []
hdr_re = re.compile(r'^TABLE ([LXVIC]+)\. (.*)')
for i, l in enumerate(L):
    m = hdr_re.match(l)
    if not m: continue
    title = m.group(2)
    if not re.search(r'metric|comparison|quality|goodness|criteria|complexity', title, re.I): continue
    gal = re.split(r'[:—]| fit| model| goodness| metrics', title)[0].strip()
    if title.startswith('Fit quality metrics for UGC 2953'): gal = 'UGC 2953'
    if title.startswith('Fit quality metrics for 2-'): gal = 'NGC 55 (segments)'
    if title.startswith('Goodness–of–fit metrics for the (R,M)'): gal = 'NGC 3877'
    for l2 in L[i+1:i+60]:
        if hdr_re.match(l2): break
        toks = l2.split()
        nums = [t for t in toks if re.fullmatch(num, t)]
        if len(nums) < 4: continue
        # name = tokens before first integer k (1..8) that is followed by numbers
        k_idx = None
        for j, t in enumerate(toks):
            if re.fullmatch(r'[1-8]', t) and j > 0 and all(re.fullmatch(num, x) for x in toks[j+1:j+4] if x):
                k_idx = j; break
        if k_idx is None: continue
        name = ' '.join(toks[:k_idx]); vals = [float(x) for x in toks[k_idx:] if re.fullmatch(num, x)]
        if len(vals) < 5 or not re.search(r'[A-Za-z]', name): continue
        k = int(vals[0]); aic, bic, rms = vals[-3], vals[-2], vals[-1]; chi2 = vals[1]
        if abs(aic - (chi2 + 2*k)) > 0.6: continue          # sanity: AIC = chi2 + 2k
        rows.append(dict(table=m.group(1), galaxy=gal, model=name, k=k, chi2=chi2, AIC=aic, BIC=bic, RMS=rms))
def fam(n):
    n2 = n.lower()
    if 'mond' in n2: return 'MOND'
    if any(s in n2 for s in ('dm', 'iso', 'nfw', 'burkert')): return 'DM'
    return 'CL'
for r in rows: r['family'] = fam(r['model'])
json.dump(rows, open('data/fit_rows.json','w'), indent=1)
from collections import defaultdict
G = defaultdict(list)
for r in rows: G[r['galaxy']].append(r)
print(len(rows), 'rows,', len(G), 'galaxies')
for g, rs in G.items():
    fams = sorted({r['family'] for r in rs})
    print(f"{g:22s} {','.join(fams):12s} " + ' | '.join(f"{r['model'][:22]}(k{r['k']}) AIC {r['AIC']}" for r in rs))
