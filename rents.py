"""
paradewarrior - Microburbs Hackathon Task 4: what will this flat rent for?

Same engine as the valuation model, applied to weekly asking rents:
  1. repeat-rent index (quarterly, smoothed) from flats advertised more than once
  2. parking premium measured INSIDE buildings: same building, same bedrooms,
     with vs without a car space; every advert is normalised to "no car space"
  3. value ladder: own last advert (time-adjusted) blended with same-building,
     same-bedroom comparables, then suburb x bedrooms, suburb, city
  4. a short trend carry from 31 July to the Aug-Sep scoring window
Tuned on two earlier folds (incl. Aug-Sep 2025, the same season as scoring);
the Jun-Jul 2026 test is scored once.
"""
import re, json, itertools
import numpy as np, pandas as pd

D = "data/"; TEAM = "paradewarrior"
TASK_DATE = pd.Timestamp("2026-07-31")
FOLDS = [(pd.Timestamp("2025-07-31"), pd.Timestamp("2025-09-30")),   # same season as judging
         (pd.Timestamp("2026-01-31"), pd.Timestamp("2026-03-31"))]
TEST = (pd.Timestamp("2026-05-31"), pd.Timestamp("2026-07-31"))

def qidx(d): return (d.dt.year - 2012) * 4 + (d.dt.month - 1) // 3
UNITNUM_RE = re.compile(r"^\s*(?:unit|apartment|apt|flat|penthouse)\s+([A-Z]?\d+[A-Z]?)\s*,", re.I)
def floor_of(u):
    s = str(u).strip().upper()
    if re.fullmatch(r"G\d{1,2}", s): return 0.0
    m = re.fullmatch(r"(\d{3,4})", s)
    return float(int(m.group(1)[:-2])) if m else np.nan

L = pd.read_csv(D + "rents-past-leases-city-of-sydney.csv", parse_dates=["listing_date"])
I = pd.read_csv(D + "rents-input-flats-city-of-sydney.csv")
L = L[(L.weekly_rent >= 150) & (L.weekly_rent <= 5000) & (L.listing_date <= TASK_DATE)].copy()
L["bld"] = L.building_address.fillna(L.address.str.replace(r"^\s*UNIT\s+[^,]+,\s*", "", regex=True))
L["q"] = qidx(L.listing_date); L["ly"] = np.log(L.weekly_rent)
L["fl"] = L.address.str.extract(UNITNUM_RE, expand=False).map(floor_of)
# advert parking, else what we know about the flat
L["car"] = L.car_space.fillna(L.gnaf_id.map(I.set_index("gnaf_id").car_space))
L["beds"] = L.bedrooms.fillna(L.gnaf_id.map(I.set_index("gnaf_id").bedrooms))
I["bld"] = I.building_address
I["fl"] = I.floor_level

def fit_index(p, Q, lam=4.0):
    for _ in range(2):
        X = np.zeros((len(p), Q)); r = np.arange(len(p))
        X[r, p.q2.values] += 1; X[r, p.q1.values] -= 1
        w = 1 / np.sqrt(1 + 0.2 * p.hold.values)
        P = np.zeros((Q - 2, Q))
        for i in range(Q - 2): P[i, i:i + 3] = [1, -2, 1]
        A = np.vstack([X * w[:, None], np.sqrt(lam) * P, np.eye(1, Q)])
        b = np.concatenate([p.y.values * w, np.zeros(Q - 2), [0]])
        idx = np.linalg.lstsq(A, b, rcond=None)[0]
        res = p.y.values - (idx[p.q2.values] - idx[p.q1.values])
        mad = np.median(np.abs(res - np.median(res))) * 1.4826
        p = p[np.abs(res) < 4 * mad]
    return idx

def build(cutoff):
    s = L[L.listing_date <= cutoff].sort_values(["gnaf_id", "listing_date"]).copy()
    Q = int(qidx(pd.Series([cutoff])).iloc[0]) + 1
    g = s.groupby("gnaf_id")
    p = pd.DataFrame({"q1": g.q.shift(), "q2": s.q, "d1": g.listing_date.shift(), "d2": s.listing_date,
                      "y": s.ly - g.ly.shift(), "c1": g.car.shift(), "c2": s.car}).dropna(subset=["q1", "y"])
    p["hold"] = (p.d2 - p.d1).dt.days / 365.25
    p = p[(p.hold >= 0.25) & (p.y.abs() < 0.8) & ((p.c1 == p.c2) | p.c1.isna())]   # same parking status
    p[["q1", "q2"]] = p[["q1", "q2"]].astype(int)
    idx = fit_index(p, Q)
    trend_q = (idx[-1] - idx[-5]) / 4          # average quarterly growth over the last year
    s["adj"] = s.ly + idx[-1] - idx[s.q.values]
    s["yrs"] = (cutoff - s.listing_date).dt.days / 365.25

    # parking premium within building x bedrooms, by bedroom band
    c = s[s.yrs <= 4].dropna(subset=["car", "beds"]).copy()
    c["band"] = c.beds.clip(0, 3)
    grp = c.groupby(["bld", "beds"])
    both = grp.car.transform("nunique") == 2
    k = c[both]
    dx = k.car - k.groupby(["bld", "beds"]).car.transform("mean")
    dy = k.adj - k.groupby(["bld", "beds"]).adj.transform("mean")
    prem = {}
    for band in range(4):
        m_ = (k.band == band)
        prem[band] = float(np.clip((dx[m_] * dy[m_]).sum() / max((dx[m_] ** 2).sum(), 1e-9), 0, 0.4)) if m_.sum() > 30 else 0.06
    pi_car = float(s.car.mean())
    band_s = s.beds.clip(0, 3).fillna(1).astype(int)
    s["adj0"] = s.adj - band_s.map(prem) * s.car.fillna(pi_car)       # normalised to "no car space"

    # floor premium within building x beds (from unit numbers)
    f = s[(s.yrs <= 4)].dropna(subset=["fl", "beds"])
    f = f[f.groupby(["bld", "beds"]).adj0.transform("count") >= 3]
    fx = f.fl - f.groupby(["bld", "beds"]).fl.transform("mean")
    fy = f.adj0 - f.groupby(["bld", "beds"]).adj0.transform("mean")
    fslope = float(np.clip((fx * fy).sum() / max((fx ** 2).sum(), 1e-9), 0, 0.02))
    bmf = s.groupby("bld").fl.median()
    s["adj0f"] = s.adj0 - fslope * (s.fl - s.bld.map(bmf)).fillna(0)

    last = s.groupby("gnaf_id").tail(1).set_index("gnaf_id")
    c2 = s[s.yrs <= 3]; cr = s[s.yrs <= 1.5]
    def med(gr, n, col):
        a = gr[col].agg(["median", "count"]); return a[a["count"] >= n]
    tabs = dict(bb=med(c2.groupby(["bld", "beds"]), 2, "adj0f"), b=med(c2.groupby("bld"), 3, "adj0f"),
                sb=med(cr.groupby(["suburb", "beds"]), 5, "adj0")["median"],
                s=med(cr.groupby("suburb"), 5, "adj0")["median"], cb=med(cr.groupby("beds"), 5, "adj0")["median"])
    return dict(idx=idx, trend_q=trend_q, prem=prem, fslope=fslope, bmf=bmf, last=last, tabs=tabs,
                city=float(cr.adj0.median()), pairs=len(p))

def predict(m, t, w_b=0.6, w_s=0.85, half=2.0, k=2, carry=0.5, shift=0.0, use_floor=True):
    """returns (log rent without car space, log rent with car space, tier)"""
    T = m["tabs"]
    kb = pd.Series(list(zip(t.bld, t.beds)), index=t.index)
    bb = kb.map(T["bb"]["median"]).to_numpy(dtype=float, copy=True); nbb = kb.map(T["bb"]["count"]).to_numpy(dtype=float, copy=True)
    b = t.bld.map(T["b"]["median"]).to_numpy(dtype=float, copy=True)
    sb = pd.Series(list(zip(t.suburb, t.beds))).map(T["sb"]).to_numpy(dtype=float, copy=True)
    s = t.suburb.map(T["s"]).to_numpy(dtype=float, copy=True)
    cb = t.beds.map(T["cb"]).to_numpy(dtype=float, copy=True)
    if use_floor:
        fadj = (m["fslope"] * (t.fl - t.bld.map(m["bmf"]))).fillna(0).values
        bb, b = bb + fadj, b + fadj
    prior = np.where(np.isnan(sb), np.where(np.isnan(cb), s, cb), sb)
    ok = ~np.isnan(bb) & ~np.isnan(prior)
    bb[ok] = (nbb[ok] * bb[ok] + k * prior[ok]) / (nbb[ok] + k)
    comp, tier = np.full(len(t), m["city"]), np.full(len(t), "city", object)
    for name, v in [("suburb", s), ("beds", cb), ("suburb_beds", sb), ("building", b), ("building_beds", bb)]:
        o = ~np.isnan(v); comp[o] = v[o]; tier[o] = name
    own = t.gnaf_id.map(m["last"].adj0).to_numpy(dtype=float, copy=True)
    oy = t.gnaf_id.map(m["last"].yrs).to_numpy(dtype=float, copy=True)
    has = ~np.isnan(own)
    w = np.where(np.isin(tier, ["building", "building_beds"]), w_b, w_s) / (1 + np.nan_to_num(oy) / half)
    base = np.where(has, w * np.nan_to_num(own) + (1 - w) * comp, comp) + carry * m["trend_q"] + shift
    band = t.beds.clip(0, 3).fillna(1).astype(int).map(m["prem"]).values
    return base, base + band, np.where(has, "own_lease+" + tier, tier)

def holdout(a, b):
    h = L[(L.listing_date > a) & (L.listing_date <= b) & L.gnaf_id.isin(I.gnaf_id)]
    h = h.sort_values("listing_date").groupby("gnaf_id").tail(1)[["gnaf_id", "weekly_rent", "car_space"]]
    h = h.rename(columns={"car_space": "advert_car"}).merge(
        I[["gnaf_id", "bld", "suburb", "bedrooms", "fl", "car_space"]], on="gnaf_id")
    h["beds"] = h.bedrooms
    h["scored_car"] = h.advert_car.fillna(h.car_space)
    return h.dropna(subset=["scored_car"]).reset_index(drop=True)

def score(h, lo, hi):
    pred = np.where(h.scored_car.values >= 1, np.exp(hi), np.exp(lo))
    err = pred - h.weekly_rent.values; ae = np.abs(err)
    a = h.weekly_rent.values
    return dict(n=int(len(h)), median_abs_err=float(np.median(ae)), within100=float((ae <= 100).mean()),
                median_bias=float(np.median(err)),
                r2=float(1 - (err ** 2).sum() / ((a - a.mean()) ** 2).sum()))

V1 = dict(w_b=0.6, w_s=0.85, half=2.0, k=0, carry=0.0, shift=0.0, use_floor=False)

if __name__ == "__main__":
    out = {}
    folds = [(build(a), holdout(a, b)) for a, b in FOLDS]
    names = ["w_b", "w_s", "half", "k", "carry", "shift", "use_floor"]
    grid = itertools.product([0.4, 0.6, 0.8], [0.7, 0.85, 0.95], [1.0, 2.0, 4.0], [0, 2, 5],
                             [0.0, 0.5, 1.0], [-0.02, 0.0, 0.02], [False, True])
    best = None
    for vals in grid:
        kw = dict(zip(names, vals))
        scs = [score(h, *predict(m, h, **kw)[:2]) for m, h in folds]
        key = (-np.mean([x["median_abs_err"] for x in scs]), np.mean([x["within100"] for x in scs]))
        if best is None or key > best[0]: best = (key, kw, scs)
    P = best[1]
    out["validation"] = dict(folds=[[str(a.date()), str(b.date())] for a, b in FOLDS], params=P, per_fold=best[2])
    print("VALIDATION", P, [(round(x["median_abs_err"]), round(x["within100"], 3)) for x in best[2]])

    mt = build(TEST[0]); ht = holdout(*TEST)
    lo, hi, tier = predict(mt, ht, **P)
    out["test"] = dict(window=[str(TEST[0].date()), str(TEST[1].date())], **score(ht, lo, hi))
    print("TEST", {k: round(v, 3) for k, v in out["test"].items() if isinstance(v, float)}, "n", out["test"]["n"])
    abl = {}
    for name, kw in [("v1 ladder (own advert + building + suburb)", V1),
                     ("+ shrink thin building cells", {**V1, "k": P["k"]}),
                     ("+ floor premium", {**V1, "k": P["k"], "use_floor": True}),
                     ("+ trend carry to scoring window (final)", P)]:
        abl[name] = score(ht, *predict(mt, ht, **kw)[:2])
        print("ABLATION", name, round(abl[name]["median_abs_err"]), round(abl[name]["within100"], 3))
    out["ablation"] = abl
    # baseline: suburb x bedrooms median (as on the task page)
    sb = pd.Series(list(zip(ht.suburb, ht.beds))).map(mt["tabs"]["sb"]).to_numpy(dtype=float, copy=True)
    sb = np.where(np.isnan(sb), mt["city"], sb)
    out["baseline_suburb_beds"] = score(ht, sb, sb)
    print("BASELINE suburb x beds", round(out["baseline_suburb_beds"]["median_abs_err"]), round(out["baseline_suburb_beds"]["within100"], 3))
    ht2 = ht.assign(tier=tier, lo=lo, hi=hi)
    out["test_by_tier"] = {k: score(g, g.lo.values, g.hi.values) for k, g in ht2.groupby("tier") if len(g) >= 10}
    out["parking_premium_pct_by_bedrooms"] = {str(k): round(100 * (np.exp(v) - 1), 1) for k, v in mt["prem"].items()}

    mf = build(TASK_DATE)
    t = I.copy(); t["beds"] = t.bedrooms
    lo, hi, tf = predict(mf, t, **P)
    sub = pd.DataFrame({"gnaf_id": I.gnaf_id,
                        "rent_with_car_space": np.round(np.exp(hi) * 2) / 2,
                        "rent_without_car_space": np.round(np.exp(lo) * 2) / 2})
    assert len(sub) == 65_777 and sub.gnaf_id.is_unique and sub.notna().all().all()
    sub.to_csv(f"rents_{TEAM}.csv", index=False)
    prem_d = sub.rent_with_car_space - sub.rent_without_car_space
    out["final"] = dict(tiers=pd.Series(tf).value_counts().to_dict(), floor_slope=mf["fslope"],
                        parking_premium_pct_by_bedrooms={str(k): round(100 * (np.exp(v) - 1), 1) for k, v in mf["prem"].items()},
                        median_weekly_parking_premium=float(prem_d.median()),
                        trend_q=mf["trend_q"], repeat_pairs=mf["pairs"])
    json.dump(out, open("rent_metrics.json", "w"), indent=2, default=str)
    print("FINAL", out["final"])
