import pandas as pd, numpy as np, glob
FAM=["lm","svm","mars","rf","rpart"]
d=pd.concat(pd.read_csv(f) for f in glob.glob("raw_sera_by_dataset/DS*.csv")); d=d[d.model_family.isin(FAM)]
key=["dataset_id","model_family"]
def pooled(agg, drop=None):
    c=d.groupby(key+["strategy"]).sera.agg(agg).unstack("strategy")
    if drop is not None: c=c.drop(index=drop, errors="ignore")
    return (np.log(c).sub(np.log(c["baseline"]),axis=0).drop(columns="baseline")).mean()
# (a) cells containing any of the 12 dominated workflows (whole cell dropped to keep pairing)
dom=[("DS22","mars"),("DS23","lm")]  # cells hosting all dominated SERA/RMSE workflows, plus DS23 mars
dom3=dom+[("DS23","mars")]
res=pd.DataFrame({"mean":pooled("mean"),"mean_drop3cells":pooled("mean",dom3),"median":pooled("median")})
res["share_of_shrink_explained"]=((res["mean"]-res["mean_drop3cells"])/(res["mean"]-res["median"])).round(2)
print(res.round(3))
# (b) blow-up fold rate per strategy: fold value > 100x its cell's median
g=d.groupby(key+["strategy"]).sera
d["ratio"]=d.sera/g.transform("median")
for t in [10,100]:
    r=d.assign(b=d.ratio>t).groupby("strategy").b.mean().mul(100).round(2)
    print(f"% folds > {t}x cell median:", r.to_dict())
# (c) general skew: mean/median per cell, baseline vs resampled
mm=(g.mean()/g.median()).unstack("strategy")
print("median over cells of mean/median ratio:", mm.median().round(3).to_dict())
# (d) threshold sensitivity for 'dominated' count
for share,mult in [(0.5,100),(0.3,10),(0.5,10),(0.8,1000)]:
    n=0
    for _,gg in d.groupby(key+["strategy"]):
        v=gg.sera.values
        if v.max()>share*v.sum() and v.max()>mult*np.median(v): n+=1
    print(f"dominated cells (share>{share}, >{mult}x median): {n}")
