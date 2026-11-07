"""
Hypothesis (KZ): falling flat prices and rising interest rates eventually lead to falling rents in the City of Sydney.

Tests
  A. Time series: do past price growth and past cash-rate changes predict NEXT-year rent growth?
     (quarterly, overlapping 4-quarter windows, Newey-West standard errors)
  B. Panel: within the same year, do suburbs whose flat prices fell more see weaker rent growth next year?
     (year fixed effects remove the cash rate and anything else citywide)
  C. Out of sample: does adding price and rate information beat a rent-only forecast?
Data: supplied Microburbs sales and rent adverts (to Aug/Jul 2026), RBA cash rate target (rba.gov.au).
"""
import json, numpy as np, pandas as pd
import model as MS          # sales: S, pairs_of, fit_index
import rents as MR          # rents: L

def qindex(pairs, start_year, Q, lam=4.0):
    return MS.fit_index(pairs, Q, lam)

# ---------- city indices, quarterly from 2012Q1 ----------
END_Q = pd.Period("2026Q2", "Q")
qs = pd.period_range("2012Q1", END_Q, freq="Q"); Q = len(qs)

def pairs_generic(df, date, logv, key="gnaf_id", min_hold=0.25, keep=None):
    d = df.sort_values([key, date]).copy()
    d["qq"] = (d[date].dt.year - 2012) * 4 + (d[date].dt.month - 1) // 3
    d = d[(d.qq >= 0) & (d.qq < Q)]
    g = d.groupby(key)
    p = pd.DataFrame({"q1": g.qq.shift(), "q2": d.qq, "d1": g[date].shift(), "d2": d[date],
                      "y": d[logv] - g[logv].shift(), "suburb": d.suburb}).dropna(subset=["q1", "y"])
    p["hold"] = (p.d2 - p.d1).dt.days / 365.25
    p = p[(p.hold >= min_hold) & (p.y.abs() < 1.2)]
    p[["q1", "q2"]] = p[["q1", "q2"]].astype(int)
    return p

S = MS.S[~MS.S.nonres]
L = MR.L
ps = pairs_generic(S, "sold_date", "ly", min_hold=0.5)
pr = pairs_generic(L, "listing_date", "ly")
pr = pr[pr.index.isin(pr.index)]  # all
price = pd.Series(MS.fit_index(ps, Q), index=qs)
rent = pd.Series(MS.fit_index(pr, Q), index=qs)

rba = pd.read_csv("data/rba_cash_rate_changes.csv", parse_dates=["date"]).set_index("date").cash_rate
daily = rba.reindex(pd.date_range("2010-01-01", "2026-11-30")).ffill()
rate = daily.groupby(daily.index.to_period("Q")).mean().reindex(qs)

df = pd.DataFrame({"price": price, "rent": rent, "rate": rate})
df["yield_idx"] = df.rent - df.price                                   # log rent/price (gross yield, up to a constant)
df["g_rent_next4"] = df.rent.shift(-4) - df.rent                       # rent growth over the NEXT year
df["g_rent_past4"] = df.rent - df.rent.shift(4)
df["g_price_past4"] = df.price - df.price.shift(4)
df["d_rate_past4"] = df.rate - df.rate.shift(4)
df["d_rate_past8"] = df.rate - df.rate.shift(8)
df["g_price_past8"] = df.price - df.price.shift(8)
df["covid"] = ((df.index >= pd.Period("2020Q2", "Q")) & (df.index <= pd.Period("2021Q4", "Q"))).astype(float)

def ols_hac(y, X, lags=4):
    X = np.column_stack([np.ones(len(X)), X]); n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    u = X * e[:, None]; Sm = u.T @ u
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1); G = u[l:].T @ u[:-l]; Sm += w * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ Sm @ XtXi))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, r2, n

out = {}
specs = {
    "A1: next-year rent growth ~ past-year price growth": ["g_price_past4"],
    "A2: next-year rent growth ~ past-year rate change": ["d_rate_past4"],
    "A3: ~ price + rate + past rent growth": ["g_price_past4", "d_rate_past4", "g_rent_past4"],
    "A4: ~ 2-year price + 2-year rate + past rent + COVID": ["g_price_past8", "d_rate_past8", "g_rent_past4", "covid"],
}
for name, cols in specs.items():
    d = df.dropna(subset=["g_rent_next4"] + cols)
    b, se, r2, n = ols_hac(d.g_rent_next4.values, d[cols].values)
    out[name] = dict(n=n, r2=round(r2, 3), coef={c: [round(b[i+1], 4), round(b[i+1] / se[i+1], 2)] for i, c in enumerate(cols)})
    print(name, out[name])

# correlation at several lead/lag horizons
corr = {}
for h in (0, 2, 4, 6, 8):
    fr = df.rent.shift(-4 - h) - df.rent.shift(-h)
    corr[h] = dict(price=round(fr.corr(df.g_price_past4), 2), rate=round(fr.corr(df.d_rate_past4), 2))
out["corr_future_rent_growth_vs_past_year_by_lead_quarters"] = corr
print("lead correlations", corr)

# ---------- B. suburb panel with year fixed effects ----------
def suburb_annual(pairs, min_pairs=40):
    pairs = pairs.copy(); pairs["y1"] = pairs.q1 // 4; pairs["y2"] = pairs.q2 // 4
    res = {}
    for sub, g in pairs.groupby("suburb"):
        if len(g) < min_pairs: continue
        Y = 15
        X = np.zeros((len(g), Y)); r = np.arange(len(g))
        X[r, g.y2.values] += 1; X[r, g.y1.values] -= 1
        P = np.zeros((Y - 2, Y))
        for i in range(Y - 2): P[i, i:i+3] = [1, -2, 1]
        A = np.vstack([X, 2.0 * P, np.eye(1, Y)]); bb = np.concatenate([g.y.values, np.zeros(Y - 2), [0]])
        res[sub] = np.linalg.lstsq(A, bb, rcond=None)[0]
    return pd.DataFrame(res, index=range(2012, 2027))
sp, sr = suburb_annual(ps), suburb_annual(pr)
common = sp.columns.intersection(sr.columns)
panel = []
for sub in common:
    for y in range(2014, 2025):
        panel.append(dict(suburb=sub, year=y, g_price_past=sp.loc[y, sub] - sp.loc[y-1, sub],
                          g_rent_next=sr.loc[y+1, sub] - sr.loc[y, sub], g_rent_past=sr.loc[y, sub] - sr.loc[y-1, sub]))
pn = pd.DataFrame(panel)
for c in ("g_price_past", "g_rent_next", "g_rent_past"):
    pn[c + "_w"] = pn[c] - pn.groupby("year")[c].transform("mean")     # year fixed effects
b, se, r2, n = ols_hac(pn.g_rent_next_w.values, pn[["g_price_past_w", "g_rent_past_w"]].values, lags=0)
out["B: suburb panel, year FE"] = dict(n=n, suburbs=len(common), r2=round(r2, 3),
    coef_price=[round(b[1], 4), round(b[1] / se[1], 2)], coef_past_rent=[round(b[2], 4), round(b[2] / se[2], 2)])
print("B", out["B: suburb panel, year FE"])

# ---------- C. out-of-sample forecast of next-year rent growth ----------
d = df.dropna(subset=["g_rent_next4", "g_price_past4", "d_rate_past4", "g_rent_past4"]).copy()
errs = {"rent_only": [], "rent+price+rate": [], "naive_last_year": []}
idxs = list(d.index)
for i in range(16, len(idxs)):
    tr = d.iloc[:i - 4]                    # only windows fully observed before the forecast date
    te = d.iloc[i]
    for name, cols in [("rent_only", ["g_rent_past4"]), ("rent+price+rate", ["g_rent_past4", "g_price_past4", "d_rate_past4"])]:
        X = np.column_stack([np.ones(len(tr)), tr[cols].values]); bb = np.linalg.lstsq(X, tr.g_rent_next4.values, rcond=None)[0]
        errs[name].append(te.g_rent_next4 - (bb[0] + te[cols].values @ bb[1:]))
    errs["naive_last_year"].append(te.g_rent_next4 - te.g_rent_past4)
out["C: out-of-sample RMSE of next-year rent growth (pct pts)"] = {k: round(100 * float(np.sqrt(np.mean(np.square(v)))), 2) for k, v in errs.items()}
out["C: forecast window"] = [str(idxs[16]), str(idxs[-1]), len(idxs) - 16]
print("C", out["C: out-of-sample RMSE of next-year rent growth (pct pts)"], out["C: forecast window"])

# episodes
ep = df.loc[["2016Q2", "2017Q4", "2019Q2", "2020Q1", "2021Q4", "2022Q2", "2023Q4", "2024Q4", "2025Q4", "2026Q2"], ["price", "rent", "rate"]]
ep[["price", "rent"]] = np.exp(ep[["price", "rent"]] - df.loc["2022Q1", ["price", "rent"]])
ep.index = ep.index.astype(str); out["episodes_index_2022Q1_eq_1"] = ep.round(3).to_dict()
print(ep.round(3))
df[["price", "rent", "rate", "yield_idx"]].to_csv("rent_vs_rates_series.csv")
json.dump(out, open("rent_vs_rates.json", "w"), indent=2, default=str)
