"""
paradewarrior - Microburbs Hackathon Task 5: value every flat.

Value ladder:
  1. own last sale x repeat-sales index, plus shrunk suburb/building drift
  2. same-building comparables (bedroom-matched, floor-adjusted), shrunk toward suburb x bedrooms
  3. suburb x bedrooms, suburb, city medians as fallbacks
Own-sale and comparable estimates are blended in log space.
Parameters are tuned on two validation folds; the Jun-Aug 2026 test is scored once.

Usage:  python model.py
"""
import re, json, itertools
import numpy as np, pandas as pd

D = "data/"
TEAM = "paradewarrior"
FINAL_DATE = pd.Timestamp("2026-08-31")
FOLDS = [(pd.Timestamp("2025-12-01"), pd.Timestamp("2026-03-01")),
         (pd.Timestamp("2026-03-01"), pd.Timestamp("2026-06-01"))]
TEST_SPLIT = pd.Timestamp("2026-06-01")
VAL_SPLIT = FOLDS[-1][0]   # kept for make_report.py

UNIT_RE = re.compile(r"^\s*(unit|apartment|apt|flat|suite|shop|office|villa|townhouse|car\s*space|lot|penthouse|studio)\b[^,]*,\s*", re.I)
NONRES_RE = re.compile(r"^\s*(shop|office|suite|car\s*space|storage|lot)\b", re.I)
UNITNUM_RE = re.compile(r"^\s*(?:unit|apartment|apt|flat|penthouse)\s+([A-Z]?\d+[A-Z]?)\s*,", re.I)

def bkey(a):  return UNIT_RE.sub("", str(a)).strip().upper()
def qidx(d):  return (d.dt.year - 1995) * 4 + (d.dt.month - 1) // 3

def floor_of(u):
    """Unit 1204 -> 12, Unit 305 -> 3, G08 -> 0; unknown otherwise"""
    s = str(u).strip().upper()
    if re.fullmatch(r"G\d{1,2}", s): return 0.0
    m = re.fullmatch(r"(\d{3,4})", s)
    return float(int(m.group(1)[:-2])) if m else np.nan

def unit_pos(u):
    """Unit 1204 -> '04': flats stacked vertically usually share a layout"""
    m = re.fullmatch(r"\d{3,4}", str(u).strip())
    return m.group(0)[-2:] if m else None

# ---------- load ----------
S = pd.read_csv(D + "flat-sales-city-of-sydney.csv", parse_dates=["sold_date"])
F = pd.read_csv(D + "flats-city-of-sydney.csv", low_memory=False)
S = S[(S.price > 50_000)].copy()
S["bld"] = S.address.map(bkey)
S["nonres"] = S.address.str.match(NONRES_RE)
S["q"] = qidx(S.sold_date)
S["ly"] = np.log(S.price)
S["fl"] = S.address.str.extract(UNITNUM_RE, expand=False).map(floor_of)
F["bld"] = F.address.map(bkey)
F["nonres"] = F.address.str.match(NONRES_RE)
F["pos"] = F.unit_number.map(unit_pos)
F["fl"] = F.unit_number.map(floor_of)

# ---------- repeat-sales index ----------
def pairs_of(s):
    s = s[~s.nonres].sort_values(["gnaf_id", "sold_date"])
    g = s.groupby("gnaf_id")
    p = pd.DataFrame({
        "gnaf_id": s.gnaf_id, "suburb": s.suburb, "bld": s.bld,
        "q1": g.q.shift(), "q2": s.q, "d1": g.sold_date.shift(), "d2": s.sold_date,
        "y": s.ly - g.ly.shift()}).dropna()
    p["hold"] = (p.d2 - p.d1).dt.days / 365.25
    p = p[(p.hold >= 0.5) & (p.y.abs() < 1.5)]
    p[["q1", "q2"]] = p[["q1", "q2"]].astype(int)
    return p

def fit_index(p, Q, lam=4.0):
    for _ in range(2):
        X = np.zeros((len(p), Q)); r = np.arange(len(p))
        X[r, p.q2.values] += 1; X[r, p.q1.values] -= 1
        w = 1 / np.sqrt(1 + 0.1 * p.hold.values)
        P = np.zeros((Q - 2, Q))
        for i in range(Q - 2): P[i, i:i + 3] = [1, -2, 1]
        A = np.vstack([X * w[:, None], np.sqrt(lam) * P, np.eye(1, Q)])
        b = np.concatenate([p.y.values * w, np.zeros(Q - 2), [0]])
        idx = np.linalg.lstsq(A, b, rcond=None)[0]
        res = p.y.values - (idx[p.q2.values] - idx[p.q1.values])
        mad = np.median(np.abs(res - np.median(res))) * 1.4826
        p = p[np.abs(res) < 4 * mad]
    return idx

def drift(p, idx, key, k, cutoff):
    rec = p[p.d2 >= cutoff - pd.DateOffset(years=4)]
    res = rec.y - (idx[rec.q2.values] - idx[rec.q1.values])
    g = pd.DataFrame({key: rec[key], "r": res, "h": rec.hold}).groupby(key)
    return (g.r.sum() / (g.h.sum() + k)).clip(-0.05, 0.05)

# ---------- model ----------
def build_model(cutoff):
    s = S[S.sold_date < cutoff]
    Q = int(qidx(pd.Series([cutoff])).iloc[0]) + 1
    p = pairs_of(s)
    idx = fit_index(p, Q)
    qT = Q - 1
    ds, db = drift(p, idx, "suburb", 15, cutoff), drift(p, idx, "bld", 6, cutoff)

    s = s.copy()
    yrs = (cutoff - s.sold_date).dt.days / 365.25
    s["adj"] = (s.ly + idx[qT] - idx[s.q.values]
                + s.suburb.map(ds).fillna(0) * yrs + s.bld.map(db).fillna(0) * yrs)
    s["yrs"] = yrs
    last = s.sort_values("sold_date").groupby("gnaf_id").tail(1).set_index("gnaf_id")
    beds_hist = s.dropna(subset=["bedrooms"]).sort_values("sold_date").groupby("gnaf_id").bedrooms.last()

    c = s[(~s.nonres) & (s.yrs <= 8)].copy()
    # floor premium: within building x bedrooms, log price vs floor
    fk = c.dropna(subset=["fl", "bedrooms"])
    grp = fk.groupby(["bld", "bedrooms"])
    fk = fk[grp.adj.transform("count") >= 3]
    dx = fk.fl - fk.groupby(["bld", "bedrooms"]).fl.transform("mean")
    dy = fk.adj - fk.groupby(["bld", "bedrooms"]).adj.transform("mean")
    slope = float(np.clip((dx * dy).sum() / max((dx * dx).sum(), 1e-9), 0, 0.03))
    bmf = c.groupby("bld").fl.median()
    c["adjf"] = c.adj - slope * (c.fl - c.bld.map(bmf)).fillna(0)

    cr = c[c.yrs <= 3]
    def med(g, n, col="adj"):
        a = g[col].agg(["median", "count"]); return a[a["count"] >= n]
    tabs = {}
    for col in ("adj", "adjf"):
        tabs[f"bb_{col}"] = med(c.groupby(["bld", "bedrooms"]), 2, col)
        gb = c.groupby("bld")[col]; cnt = gb.count()
        for qq in (0.3, 0.4, 0.5):
            tabs[f"b{int(qq*100)}_{col}"] = gb.quantile(qq)[cnt >= 3]
    tabs["sb"] = med(cr.groupby(["suburb", "bedrooms"]), 5)["median"]
    tabs["s"] = med(cr.groupby("suburb"), 5)["median"]
    city = cr.adj.median()
    kb = F.assign(b2=F.bedrooms.fillna(F.gnaf_id.map(beds_hist))).dropna(subset=["b2", "pos"])
    stack = kb.groupby(["bld", "pos"]).b2.agg(lambda x: x.mode().iloc[0])
    return dict(last=last, beds_hist=beds_hist, tabs=tabs, city=city, idx=idx,
                stack=stack, slope=slope, bmf=bmf)

def predict(m, targets, w_b=0.5, w_s=0.8, half=5, bq=0.3, shift=0.0,
            use_stack=True, use_floor=True, k=0):
    t = targets.copy()
    t["beds"] = t.bedrooms.fillna(t.gnaf_id.map(m["beds_hist"]))
    if use_stack:
        st = pd.Series(list(zip(t.bld, t.pos))).map(m["stack"]).values
        t["beds"] = t.beds.fillna(pd.Series(st, index=t.index))
    T = m["tabs"]; col = "adjf" if use_floor else "adj"
    keys_bb = pd.Series(list(zip(t.bld, t.beds)))
    bb = keys_bb.map(T[f"bb_{col}"]["median"]).values.astype(float)
    nbb = keys_bb.map(T[f"bb_{col}"]["count"]).values.astype(float)
    sb = pd.Series(list(zip(t.suburb, t.beds))).map(T["sb"]).values.astype(float)
    b = t.bld.map(T[f"b{int(bq*100)}_{col}"]).values.astype(float)
    s = t.suburb.map(T["s"]).values.astype(float)
    if use_floor:
        fadj = (m["slope"] * (t.fl - t.bld.map(m["bmf"]))).fillna(0).values
        bb, b = bb + fadj, b + fadj
    if k > 0:   # shrink thin building x beds cells toward suburb x beds (or suburb)
        prior = np.where(np.isnan(sb), s, sb)
        ok = ~np.isnan(bb) & ~np.isnan(prior)
        bb[ok] = (nbb[ok] * bb[ok] + k * prior[ok]) / (nbb[ok] + k)
    comp, tier = np.full(len(t), m["city"]), np.full(len(t), "city", object)
    for name, v in [("suburb", s), ("suburb_beds", sb), ("building", b), ("building_beds", bb)]:
        ok = ~np.isnan(v); comp[ok] = v[ok]; tier[ok] = name
    own = t.gnaf_id.map(m["last"].adj).values.astype(float)
    oy = t.gnaf_id.map(m["last"].yrs).values.astype(float)
    has = ~np.isnan(own)
    w = np.where(np.isin(tier, ["building", "building_beds"]), w_b, w_s) / (1 + np.nan_to_num(oy) / half)
    pred = np.where(has, w * np.nan_to_num(own) + (1 - w) * comp, comp) + shift
    tier = np.where(has, "own_sale+" + tier, tier)
    return np.exp(pred), tier

def score(pred, actual):
    e = pred / actual - 1
    ss_res = ((actual - pred) ** 2).sum(); ss_tot = ((actual - actual.mean()) ** 2).sum()
    return dict(n=int(len(actual)), within10=float((np.abs(e) <= 0.10).mean()),
                median_abs_err=float(np.median(np.abs(pred - actual))),
                median_pct_err=float(np.median(e)), r2=float(1 - ss_res / ss_tot))

def holdout(split, end):
    t = S[(S.sold_date >= split) & (S.sold_date < end) & (S.price < 1.2e6) & (~S.nonres)]
    t = t.sort_values("sold_date").groupby("gnaf_id").tail(1).drop(columns=["bedrooms", "fl"])
    return t.merge(F[["gnaf_id", "bedrooms", "pos", "fl"]], on="gnaf_id", how="left")

def baseline_citywide(m, t):
    last = m["last"]; idx = m["idx"]; qT = len(idx) - 1
    lq = t.gnaf_id.map(last.q)
    return np.exp(t.gnaf_id.map(last.ly) + idx[qT] - idx[lq.fillna(qT).astype(int).values]).values

V1 = dict(w_b=0.5, w_s=0.8, half=5, bq=0.3, shift=0.0, use_stack=True, use_floor=False, k=0)

if __name__ == "__main__":
    out = {}
    folds = [(build_model(a), holdout(a, b)) for a, b in FOLDS]
    names = ["w_b", "w_s", "half", "bq", "shift", "use_floor", "k"]
    grid = itertools.product([0.3, 0.5, 0.7], [0.6, 0.8, 0.95], [5, 10, 1e9], [0.3, 0.4, 0.5],
                             [-0.04, -0.02, 0.0], [False, True], [0, 2, 5])
    best = None
    for vals in grid:
        kw = dict(zip(names, vals))
        scs = [score(predict(m, t, **kw)[0], t.price.values) for m, t in folds]
        key = (np.mean([x["within10"] for x in scs]), -np.mean([x["median_abs_err"] for x in scs]))
        if best is None or key > best[0]: best = (key, kw, scs)
    P = best[1]
    out["validation"] = dict(folds=[[str(a.date()), str(b.date())] for a, b in FOLDS], params=P,
                             within10=best[0][0], per_fold=best[2],
                             r2=float(np.mean([x["r2"] for x in best[2]])),
                             n=int(sum(x["n"] for x in best[2])))
    print("VALIDATION", P, [round(x["within10"], 3) for x in best[2]])

    mt = build_model(TEST_SPLIT); tt = holdout(TEST_SPLIT, FINAL_DATE + pd.Timedelta(days=1))
    pr, tier = predict(mt, tt, **P)
    out["test"] = dict(split=str(TEST_SPLIT.date()), floor_slope=mt["slope"], **score(pr, tt.price.values))
    abl = {}
    steps = [("v1 model (repeat-sales ladder + stacked beds)", V1),
             ("+ floor premium", {**V1, "use_floor": True}),
             ("+ shrink thin building cells", {**V1, "use_floor": True, "k": P["k"]}),
             ("final (all params from 2-fold validation)", P)]
    for name, kw in steps:
        p_, _ = predict(mt, tt, **kw); abl[name] = score(p_, tt.price.values)
        print("ABLATION", name, round(abl[name]["within10"], 3), round(abl[name]["median_abs_err"]))
    out["ablation"] = abl
    tt = tt.assign(pred=pr, tier=tier)
    out["test_by_tier"] = {k_: score(g.pred.values, g.price.values) for k_, g in tt.groupby("tier") if len(g) >= 5}
    b1 = baseline_citywide(mt, tt); okb = ~np.isnan(b1)
    out["baseline_citywide_last_sale"] = score(b1[okb], tt.price.values[okb])
    out["model_on_same_subset"] = score(pr[okb], tt.price.values[okb])
    print("TEST", {k_: round(v, 3) if isinstance(v, float) else v for k_, v in out["test"].items()})
    print("BASELINE", round(out["baseline_citywide_last_sale"]["within10"], 3), "MODEL same subset", round(out["model_on_same_subset"]["within10"], 3))
    tt[["gnaf_id", "suburb", "sold_date", "price", "pred", "tier"]].to_csv("test_predictions.csv", index=False)

    mf = build_model(FINAL_DATE + pd.Timedelta(days=1))
    pf, tf = predict(mf, F, **P)
    sub = pd.DataFrame({"gnaf_id": F.gnaf_id, "value": np.round(pf).astype(np.int64)})
    assert len(sub) == 181_711 and sub.gnaf_id.is_unique and sub.value.notna().all()
    sub.to_csv(f"valuation_{TEAM}.csv", index=False)
    out["final_tiers"] = pd.Series(tf).value_counts().to_dict()
    out["final_floor_slope"] = mf["slope"]
    json.dump(out, open("metrics.json", "w"), indent=2, default=str)
    print("FINAL slope", round(mf["slope"], 4), "tiers", out["final_tiers"])
