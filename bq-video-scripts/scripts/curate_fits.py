"""Curate the parsed fit rows into the per-galaxy comparison used by aic_movie.py.
Rules (as stated in the video): the later table version is used for NGC 3741 and UGC 6446;
F563-V2 is excluded (no DM fit) and UGC 2953 is excluded (its AIC does not follow chi2 + 2k);
the UGC 1281 NFW row is entered by hand (its RMS column is not numeric in the paper).
Usage:  python curate_fits.py   (reads data/fit_rows.json, writes data/curated.json and two CSV files)"""
from paths import HERE
import json, csv
from collections import defaultdict, Counter
rows = json.load(open("data/fit_rows.json"))
LATER = {"NGC 3741": {"LXXVIII", "LXXX"}, "UGC 6446": {"CIII"}}
rows = [r for r in rows if not (r["galaxy"] in LATER and r["table"] not in LATER[r["galaxy"]])]
rows = [r for r in rows if r["galaxy"] not in ("F563–V2 (n = 10)",)]
rows.append(dict(table="V", galaxy="UGC 1281", model="DM: NFW halo", k=2, chi2=75.02, AIC=79.02, BIC=81.46, RMS=None, family="DM"))
seen, R = set(), []
for r in rows:
    key = (r["galaxy"], r["model"].replace(" ", ""), r["k"], r["AIC"])
    if key not in seen: seen.add(key); R.append(r)
G = defaultdict(list)
for r in R: G[r["galaxy"]].append(r)
out = []
for g, rs in G.items():
    f = lambda fam, k=None: [r for r in rs if r["family"] == fam and (k is None or r["k"] == k)]
    if not (f("CL") and f("MOND") and f("DM")): continue
    m = lambda Lst, c: min(Lst, key=lambda r: r[c])
    cl2, comp2 = m(f("CL", 2), "AIC"), m(f("MOND", 2) + f("DM", 2), "AIC")
    clb, compb = m(f("CL"), "AIC"), m(f("MOND") + f("DM"), "AIC")
    clbB, compbB = m(f("CL"), "BIC"), m(f("MOND") + f("DM"), "BIC")
    dmtype = "NFW" if any("NFW" in r["model"] and "gNFW" not in r["model"] for r in f("DM")) else \
             ("ISO+baryon" if any("baryon" in r["model"] for r in f("DM", 2)) else "ISO")
    out.append(dict(galaxy=g.replace(" (NGC 360)", "").replace(" (IC 2233)", ""),
                    dAIC_k2=round(cl2["AIC"] - comp2["AIC"], 2), comp_k2=comp2["family"],
                    dAIC_best=round(clb["AIC"] - compb["AIC"], 2), cl_best_k=clb["k"], cl_best_model=clb["model"],
                    comp_best=compb["family"], comp_best_k=compb["k"], dBIC_best=round(clbB["BIC"] - compbB["BIC"], 2),
                    lowest_family=min(rs, key=lambda r: r["AIC"])["family"], dm_type=dmtype))
json.dump(out, open("data/curated.json", "w"), indent=1)
with open("data/cl_mond_dm_aic_bic_curated.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
with open("data/cl_mond_dm_all_rows.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["galaxy", "table", "family", "model", "k", "chi2", "AIC", "BIC", "RMS"]); w.writeheader()
    for r in R: w.writerow({k: r.get(k) for k in w.fieldnames})
t = lambda k: (sum(o[k] < -2 for o in out), sum(abs(o[k]) <= 2 for o in out), sum(o[k] > 2 for o in out))
print(len(out), "galaxies; CL/tie/competitor  AIC k=2:", t("dAIC_k2"), " AIC best:", t("dAIC_best"), " BIC best:", t("dBIC_best"))
print("lowest-AIC family:", dict(Counter(o["lowest_family"] for o in out)))
