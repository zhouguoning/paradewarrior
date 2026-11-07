"""
Does a building's recent rent trend, relative to the city, predict its rent over the next few months?

Walk-forward backtest:
  at each anchor date t (quarterly), using only data <= t:
    - rebuild the Task 4 rent model
    - measure each building's 12-month rent growth from repeat adverts, minus the city's
  then for adverts in (t, t+h]:
    - residual = log(actual asking rent / model prediction)
  test: residual ~ beta * relative building trend
    beta > 0  -> momentum (laggards keep lagging)
    beta < 0  -> mean reversion (laggards catch up)
"""
import json, numpy as np, pandas as pd
import rents as R

P = json.load(open("rent_metrics.json"))["validation"]["params"]
L, I = R.L, R.I

def trends(cut, min_pairs=5):
    s = L[L.listing_date <= cut].sort_values(["gnaf_id", "listing_date"])
    g = s.groupby("gnaf_id")
    p = pd.DataFrame({"bld": s.bld, "suburb": s.suburb, "d1": g.listing_date.shift(), "d2": s.listing_date,
                      "y": s.ly - g.ly.shift(), "c1": g.car.shift(), "c2": s.car}).dropna(subset=["d1"])
    p["hold"] = (p.d2 - p.d1).dt.days / 365.25
    p = p[p.hold.between(0.5, 3) & (p.y.abs() < 0.8) & ((p.c1 == p.c2) | p.c1.isna()) & (p.d2 > cut - pd.DateOffset(years=1))]
    p["ann"] = p.y / p.hold
    city = p.ann.median()
    b = p.groupby("bld").ann.agg(["median", "count"]); b = b[b["count"] >= min_pairs]
    sb = p.groupby("suburb").ann.median()
    return city, b["median"], b["count"], sb

rows = []
anchors = pd.date_range("2018-12-31", "2025-12-31", freq="QE")
for t in anchors:
    m = R.build(t)
    city, bt, bn, st = trends(t)
    for h in (3, 6, 12):
        end = t + pd.DateOffset(months=h)
        if end > R.TASK_DATE: continue
        ad = L[(L.listing_date > t) & (L.listing_date <= end) & L.gnaf_id.isin(I.gnaf_id)]
        ad = ad.sort_values("listing_date").groupby("gnaf_id").tail(1)
        x = ad[["gnaf_id", "weekly_rent", "car_space"]].merge(
            I[["gnaf_id", "bld", "suburb", "bedrooms", "fl", "car_space"]].rename(columns={"car_space": "car_in"}), on="gnaf_id")
        x["beds"] = x.bedrooms
        lo, hi, _ = R.predict(m, x, **P)
        car = x.car_space.fillna(x.car_in).fillna(0)
        x["resid"] = np.log(x.weekly_rent) - np.where(car >= 1, hi, lo)
        x["rel_bld"] = x.bld.map(bt) - city
        x["n_pairs"] = x.bld.map(bn)
        x["rel_sub"] = x.suburb.map(st) - city
        x["anchor"] = t; x["h"] = h
        rows.append(x.dropna(subset=["rel_bld"]))
    print(t.date(), "city 12m trend %.1f%%" % (100 * city), "buildings with trend:", len(bt))
D = pd.concat(rows, ignore_index=True)

def cluster_ols(y, X, groups):
    X = np.column_stack([np.ones(len(X)), X]); XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    meat = np.zeros((X.shape[1],) * 2)
    for g in np.unique(groups):
        u = (X[groups == g] * e[groups == g, None]).sum(0); meat += np.outer(u, u)
    return b, np.sqrt(np.diag(XtXi @ meat @ XtXi))

out = {"anchors": [str(anchors[0].date()), str(anchors[-1].date()), len(anchors)]}
for h in (3, 6, 12):
    d = D[(D.h == h)]
    d = d[d.rel_bld.abs() < 0.3]                          # drop absurd building trends
    grp = (d.bld + "|" + d.anchor.astype(str)).factorize()[0]
    b, se = cluster_ols(d.resid.values, d[["rel_bld"]].values, grp)
    # buckets: laggards vs leaders
    d = d.assign(bucket=pd.cut(d.rel_bld, [-1, -0.03, -0.01, 0.01, 0.03, 1],
                               labels=["lag >3pts", "lag 1-3", "in line", "lead 1-3", "lead >3pts"]))
    bk = d.groupby("bucket", observed=True).resid.agg(["mean", "count"])
    out[f"h{h}"] = dict(n=len(d), beta=round(b[1], 3), t=round(b[1] / se[1], 2),
                        buckets={k: [round(100 * v["mean"], 2), int(v["count"])] for k, v in bk.iterrows()})
    print(f"\nh={h}m  n={len(d)}  beta={b[1]:.3f}  t={b[1]/se[1]:.2f}   (resid in %, by building trend vs city)")
    print((bk.assign(mean=100 * bk["mean"])).round(2))

# out-of-sample: learn beta on earlier anchors only, apply to later ones
d6 = D[(D.h == 6) & (D.rel_bld.abs() < 0.3)].sort_values("anchor")
res = []
for t in sorted(d6.anchor.unique())[8:]:
    tr = d6[d6.anchor < t - pd.DateOffset(months=6)]; te = d6[d6.anchor == t]
    if len(tr) < 200 or te.empty: continue
    beta = np.polyfit(tr.rel_bld, tr.resid, 1)[0]
    base = np.abs(np.exp(te.resid) - 1); adj = np.abs(np.exp(te.resid - beta * te.rel_bld) - 1)
    rent = te.weekly_rent.values
    res.append(dict(anchor=str(t.date()), beta=beta, n=len(te),
                    mae_base=np.median(np.abs(rent - rent / np.exp(te.resid))),
                    mae_adj=np.median(np.abs(rent - rent / np.exp(te.resid) * np.exp(beta * te.rel_bld)))))
oos = pd.DataFrame(res)
out["oos_6m"] = dict(windows=len(oos), median_miss_base=round(oos.mae_base.mean(), 2),
                     median_miss_with_trend=round(oos.mae_adj.mean(), 2), beta_range=[round(oos.beta.min(), 3), round(oos.beta.max(), 3)])
print("\nOUT-OF-SAMPLE 6m:", out["oos_6m"])
json.dump(out, open("rent_forecast_backtest.json", "w"), indent=2, default=str)
D.to_parquet("rent_forecast_panel.parquet") if False else None

# ---------------- forecast model + scenario simulation ----------------
# expected % change in a building's asking rents over h months, relative to today's fair rent:
#   resid = a + g * city_trend + beta * (building_trend - city_trend)
cityT = {t: trends(t)[0] for t in anchors}
D["city"] = D.anchor.map(cityT)
fc = {}
for h in (3, 6, 12):
    d = D[(D.h == h) & (D.rel_bld.abs() < 0.3)]
    gb = d.groupby(["anchor", "bld"]).agg(resid=("resid", "mean"), rel=("rel_bld", "first"), city=("city", "first"), n=("resid", "size"))
    gb = gb[gb.n >= 3].reset_index()
    X = gb[["city", "rel"]].values
    b, se = cluster_ols(gb.resid.values, X, gb.anchor.factorize()[0])
    e = gb.resid.values - np.column_stack([np.ones(len(X)), X]) @ b
    fc[h] = dict(a=b[0], g=b[1], beta=b[2], t_g=b[1] / se[1], t_beta=b[2] / se[2], resid=e, n=len(gb))
    print(f"FORECAST h={h}m: a={b[0]:+.3f}  city-trend coef={b[1]:.2f} (t={b[1]/se[1]:.1f})  building-gap coef={b[2]:.2f} (t={b[2]/se[2]:.1f})  n={len(gb)} building-quarters")

def simulate(city_trend, building_trend, h, n_sims=20000, seed=0):
    """Monte Carlo: model expectation + historical building-level forecast errors (bootstrap)."""
    f = fc[h]; rng = np.random.default_rng(seed)
    mu = f["a"] + f["g"] * city_trend + f["beta"] * (building_trend - city_trend)
    sims = mu + rng.choice(f["resid"], n_sims)
    ch = np.exp(sims) - 1
    return dict(expected=float(np.exp(mu) - 1), p10=float(np.percentile(ch, 10)), p50=float(np.percentile(ch, 50)),
                p90=float(np.percentile(ch, 90)), prob_decrease=float((ch < 0).mean()))

scen = {}
for name, ct, bt in [("building in line: city 3.5%, building 3.5%", 0.035, 0.035),
                     ("building lagging: city 3.5%, building 0%", 0.035, 0.0),
                     ("building falling: city 3.5%, building -3%", 0.035, -0.03),
                     ("whole market flat: city 0%, building 0%", 0.0, 0.0)]:
    scen[name] = {f"{h}m": {k: round(100 * v, 1) for k, v in simulate(ct, bt, h).items()} for h in (3, 6, 12)}
    print("\nSCENARIO", name)
    for h in (3, 6, 12): print(f"   {h:>2}m: expected {scen[name][f'{h}m']['expected']:+.1f}%  range p10..p90 {scen[name][f'{h}m']['p10']:+.1f}% .. {scen[name][f'{h}m']['p90']:+.1f}%  P(decrease) {scen[name][f'{h}m']['prob_decrease']:.0f}%")
out["forecast_coefs"] = {h: {k: round(float(v), 3) for k, v in f.items() if k != "resid"} for h, f in fc.items()}
out["scenarios_pct"] = scen
json.dump(out, open("rent_forecast_backtest.json", "w"), indent=2, default=str)
