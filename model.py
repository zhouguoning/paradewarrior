"""
paradewarrior - Microburbs Hackathon Task 5: value every flat.

Value ladder:
  1. own last sale x repeat-sales index, plus shrunk suburb/building drift
  2. same-building comparables (bedroom-matched when possible), time-adjusted
  3. suburb x bedrooms, suburb, city medians as fallbacks
Own-sale and comparable estimates are blended in log space.

Usage:  python model.py            -> validation + held-out test + final CSV
"""
import re, json, itertools
import numpy as np, pandas as pd

D = "data/"
TEAM = "paradewarrior"
FINAL_DATE = pd.Timestamp("2026-08-31")
VAL_SPLIT, TEST_SPLIT = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-06-01")

UNIT_RE = re.compile(r"^\s*(unit|apartment|apt|flat|suite|shop|office|villa|townhouse|car\s*space|lot|penthouse|studio)\b[^,]*,\s*", re.I)
NONRES_RE = re.compile(r"^\s*(shop|office|suite|car\s*space|storage|lot)\b", re.I)

def bkey(a):  return UNIT_RE.sub("", str(a)).strip().upper()
def qidx(d):  return (d.dt.year - 1995) * 4 + (d.dt.month - 1) // 3

# ---------- load ----------
S = pd.read_csv(D + "flat-sales-city-of-sydney.csv", parse_dates=["sold_date"])
F = pd.read_csv(D + "flats-city-of-sydney.csv", low_memory=False)
S = S[(S.price > 50_000)].copy()
S["bld"] = S.address.map(bkey)
S["nonres"] = S.address.str.match(NONRES_RE)
S["q"] = qidx(S.sold_date)
S["ly"] = np.log(S.price)
F["bld"] = F.address.map(bkey)
F["nonres"] = F.address.str.match(NONRES_RE)

def unit_pos(u):
    """Unit 1204 -> '04': flats stacked vertically usually share a layout"""
    m = re.fullmatch(r"\d{3,4}", str(u).strip())
    return m.group(0)[-2:] if m else None
F["pos"] = F.unit_number.map(unit_pos)

# ---------- 1. repeat-sales index (smoothed, all quarters) ----------
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
    for _ in range(2):  # fit, trim outliers, refit
        X = np.zeros((len(p), Q)); r = np.arange(len(p))
        X[r, p.q2.values] += 1; X[r, p.q1.values] -= 1
        w = 1 / np.sqrt(1 + 0.1 * p.hold.values)
        Xw, yw = X * w[:, None], p.y.values * w
        P = np.zeros((Q - 2, Q))                      # 2nd-difference smoothing
        for i in range(Q - 2): P[i, i:i + 3] = [1, -2, 1]
        A = np.vstack([Xw, np.sqrt(lam) * P, np.eye(1, Q)])   # anchor idx[0]=0
        b = np.concatenate([yw, np.zeros(Q - 2), [0]])
        idx = np.linalg.lstsq(A, b, rcond=None)[0]
        res = p.y.values - (idx[p.q2.values] - idx[p.q1.values])
        mad = np.median(np.abs(res - np.median(res))) * 1.4826
        p = p[np.abs(res) < 4 * mad]
    return idx

def drift(p, idx, key, k, cutoff):
    """annual growth above/below the city index, shrunk toward 0"""
    rec = p[p.d2 >= cutoff - pd.DateOffset(years=4)]
    res = rec.y - (idx[rec.q2.values] - idx[rec.q1.values])
    g = pd.DataFrame({key: rec[key], "r": res, "h": rec.hold}).groupby(key)
    return (g.r.sum() / (g.h.sum() + k)).clip(-0.05, 0.05)

# ---------- prediction ----------
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

    c = s[(~s.nonres) & (s.yrs <= 8)]
    cr = c[c.yrs <= 3]
    med = lambda g, n: g.adj.agg(["median", "count"]).query(f"count >= {n}")["median"]
    tabs = dict(
        bb=med(c.groupby(["bld", "bedrooms"]), 2), b=med(c.groupby("bld"), 3),
        sb=med(cr.groupby(["suburb", "bedrooms"]), 5), s=med(cr.groupby("suburb"), 5))
    gb = c.groupby("bld").adj
    cnt = gb.count()
    for qq in (0.3, 0.4, 0.5):
        tabs[f"b{int(qq*100)}"] = gb.quantile(qq)[cnt >= 3]
    kb = F.assign(b2=F.bedrooms.fillna(F.gnaf_id.map(beds_hist))).dropna(subset=["b2", "pos"])
    stack = kb.groupby(["bld", "pos"]).b2.agg(lambda x: x.mode().iloc[0])
    city = cr.adj.median()
    return dict(last=last, beds_hist=beds_hist, tabs=tabs, city=city, idx=idx, stack=stack)

def predict(m, targets, w_b, w_s, half, bq=0.5, shift=0.0, use_stack=True):
    t = targets.copy()
    t["beds"] = t.bedrooms.fillna(t.gnaf_id.map(m["beds_hist"]))
    if use_stack:
        st = pd.Series(list(zip(t.bld, t.pos))).map(m["stack"]).values
        t["beds"] = t.beds.fillna(pd.Series(st, index=t.index))
    T = m["tabs"]
    bb = pd.Series(list(zip(t.bld, t.beds))).map(T["bb"]).values
    sb = pd.Series(list(zip(t.suburb, t.beds))).map(T["sb"]).values
    b, s = t.bld.map(T[f"b{int(bq*100)}"]).values, t.suburb.map(T["s"]).values
    comp, tier = np.full(len(t), m["city"]), np.full(len(t), "city", object)
    for name, v in [("suburb", s), ("suburb_beds", sb), ("building", b), ("building_beds", bb)]:
        ok = ~pd.isna(v); comp[ok] = v[ok]; tier[ok] = name
    own = t.gnaf_id.map(m["last"].adj).values
    oy = t.gnaf_id.map(m["last"].yrs).values
    has = ~pd.isna(own)
    bld_tier = np.isin(tier, ["building", "building_beds"])
    w = np.where(bld_tier, w_b, w_s) / (1 + np.nan_to_num(oy) / half)
    pred = np.where(has, w * np.nan_to_num(own) + (1 - w) * comp, comp) + shift
    tier = np.where(has, "own_sale+" + tier, tier)
    return np.exp(pred), tier

def score(pred, actual):
    e = pred / actual - 1
    ss_res = ((actual - pred) ** 2).sum(); ss_tot = ((actual - actual.mean()) ** 2).sum()
    return dict(n=len(actual), within10=float((np.abs(e) <= 0.10).mean()),
                median_abs_err=float(np.median(np.abs(pred - actual))),
                median_pct_err=float(np.median(e)), r2=float(1 - ss_res / ss_tot))

def holdout(split, end):
    t = S[(S.sold_date >= split) & (S.sold_date < end) & (S.price < 1.2e6) & (~S.nonres)]
    t = t.sort_values("sold_date").groupby("gnaf_id").tail(1)
    t = t.merge(F[["gnaf_id", "bedrooms", "pos"]].rename(columns={"bedrooms": "fb"}), on="gnaf_id", how="left")
    t["bedrooms"] = t.fb
    return t

def baseline_citywide(m, t):
    last = m["last"]; idx = m["idx"]; qT = len(idx) - 1
    lp = t.gnaf_id.map(last.ly); lq = t.gnaf_id.map(last.q)
    v = np.exp(lp + idx[qT] - idx[lq.fillna(qT).astype(int).values])
    return v

if __name__ == "__main__":
    out = {}
    # --- tune on validation fold (Mar-May 2026), never touching the test fold
    mv = build_model(VAL_SPLIT); tv = holdout(VAL_SPLIT, TEST_SPLIT)
    best = None
    grid = itertools.product([0.3, 0.5, 0.7], [0.6, 0.8, 0.95], [5, 10, 1e9],
                             [0.3, 0.4, 0.5], [-0.06, -0.04, -0.02, 0.0])
    for w_b, w_s, half, bq, sh in grid:
        pr, _ = predict(mv, tv, w_b, w_s, half, bq, sh)
        sc = score(pr, tv.price.values)
        key = (sc["within10"], -sc["median_abs_err"])
        if best is None or key > best[2]: best = (sc, (w_b, w_s, half, bq, sh), key)
    out["validation"] = dict(split=str(VAL_SPLIT.date()), params=best[1], **best[0])
    w_b, w_s, half, bq, sh = best[1]
    print("VALIDATION", out["validation"])

    # --- held-out test (Jun-Aug 2026)
    mt = build_model(TEST_SPLIT); tt = holdout(TEST_SPLIT, FINAL_DATE + pd.Timedelta(days=1))
    pr, tier = predict(mt, tt, w_b, w_s, half, bq, sh)
    abl = {}
    for name, kw in [("v1: no stacking, building median, no shift", dict(bq=0.5, shift=0.0, use_stack=False)),
                     ("+ stacked-unit bedroom imputation", dict(bq=0.5, shift=0.0, use_stack=True)),
                     ("+ building quantile", dict(bq=bq, shift=0.0, use_stack=True)),
                     ("+ selection-bias shift (final)", dict(bq=bq, shift=sh, use_stack=True))]:
        p_, _ = predict(mt, tt, w_b, w_s, half, **kw); abl[name] = score(p_, tt.price.values)
        print("ABLATION", name, round(abl[name]["within10"], 3), round(abl[name]["median_abs_err"]))
    out["ablation"] = abl
    out["test"] = dict(split=str(TEST_SPLIT.date()), **score(pr, tt.price.values))
    tt = tt.assign(pred=pr, tier=tier)
    out["test_by_tier"] = {k: score(g.pred.values, g.price.values) for k, g in tt.groupby("tier") if len(g) >= 5}
    # baselines on the same test set
    sbm = tt.groupby(["suburb"]).price.transform("median")  # NB: oracle-ish, for reference only
    b1 = baseline_citywide(mt, tt); okb = ~np.isnan(b1)
    out["baseline_citywide_last_sale"] = score(b1[okb], tt.price.values[okb])
    out["model_on_same_subset"] = score(pr[okb], tt.price.values[okb])
    print("TEST", out["test"])
    for k, v in out["test_by_tier"].items(): print("  ", k, v)
    print("BASELINE citywide-index last sale", out["baseline_citywide_last_sale"])
    print("MODEL on same subset", out["model_on_same_subset"])
    tt[["gnaf_id", "suburb", "sold_date", "price", "pred", "tier"]].to_csv("test_predictions.csv", index=False)

    # --- final: everything to 31 Aug 2026
    mf = build_model(FINAL_DATE + pd.Timedelta(days=1))
    pf, tf = predict(mf, F, w_b, w_s, half, bq, sh)
    sub = pd.DataFrame({"gnaf_id": F.gnaf_id, "value": np.round(pf).astype(np.int64)})
    assert len(sub) == 181_711 and sub.gnaf_id.is_unique and sub.value.notna().all()
    sub.to_csv(f"valuation_{TEAM}.csv", index=False)
    out["final_tiers"] = pd.Series(tf).value_counts().to_dict()
    print("FINAL tiers", out["final_tiers"])
    json.dump(out, open("metrics.json", "w"), indent=2, default=str)
