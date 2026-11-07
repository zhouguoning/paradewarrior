"""
Fair Rent Check - core logic (no UI). Used by app.py.

For any flat in the City of Sydney input file:
  - fair weekly rent with and without a car space (Task 4 model, as at 31 July 2026)
  - a realistic range, from the model's held-out error for that kind of evidence
  - suburb and building rent trend
  - how an asking rent compares, and how often flats priced like that actually lease
  - negotiation pointers
Everything runs locally from the supplied files; nothing is published.
"""
import json, numpy as np, pandas as pd
import rents as R

PARAMS = json.load(open("rent_metrics.json"))["validation"]["params"]
TIER_ERR = {k: v["median_abs_err"] for k, v in json.load(open("rent_metrics.json"))["test_by_tier"].items()}

class RentCheck:
    def __init__(self):
        self.L, self.I = R.L, R.I.copy()
        self.m = R.build(R.TASK_DATE)
        t = self.I.copy(); t["beds"] = t.bedrooms
        lo, hi, tier = R.predict(self.m, t, **PARAMS)
        # app-level fix: where the building has no same-bedroom comparables, the building median reflects
        # its typical flat; shift it by the citywide bedroom gap (e.g. a 4-bed in a mostly 1-bed tower)
        cb = self.m["tabs"]["cb"]
        typ = self.I.groupby("building_address").bedrooms.transform("median").round()
        shift = (self.I.bedrooms.clip(0, 4).map(cb) - typ.clip(0, 4).map(cb)).fillna(0).values
        fix = np.char.find(tier.astype(str), "building") >= 0
        fix &= np.char.find(tier.astype(str), "building_beds") < 0
        lo = lo + np.where(fix, shift, 0); hi = hi + np.where(fix, shift, 0)
        self.I["fair_without"] = np.exp(lo); self.I["fair_with"] = np.exp(hi); self.I["tier"] = tier
        self.I["key"] = self.I.address.str.upper()
        self._trends(); self._leasing_curve()

    # ---------- trends: annual rent growth from repeat adverts in the last 12 months ----------
    def _trends(self):
        L = self.L.sort_values(["gnaf_id", "listing_date"])
        g = L.groupby("gnaf_id")
        p = pd.DataFrame({"suburb": L.suburb, "bld": L.bld, "d1": g.listing_date.shift(), "d2": L.listing_date,
                          "y": L.ly - g.ly.shift(), "c1": g.car.shift(), "c2": L.car}).dropna(subset=["d1"])
        p["hold"] = (p.d2 - p.d1).dt.days / 365.25
        p = p[(p.hold.between(0.5, 3)) & (p.y.abs() < 0.8) & ((p.c1 == p.c2) | p.c1.isna())]
        rec = p[p.d2 > R.TASK_DATE - pd.DateOffset(years=1)].copy()
        rec["ann"] = np.exp(rec.y / rec.hold) - 1
        self.city_trend = float(rec.ann.median())
        s = rec.groupby("suburb").ann.agg(["median", "count"]); self.sub_trend = s[s["count"] >= 15]
        b = rec.groupby("bld").ann.agg(["median", "count"]); self.bld_trend = b[b["count"] >= 5]

    # ---------- how often flats lease, by how far the asking rent sits above fair ----------
    def _leasing_curve(self):
        cut = pd.Timestamp("2025-12-31")
        m = R.build(cut)                        # out of sample: model knows nothing after 2025
        ad = self.L[self.L.listing_date > cut].sort_values("listing_date").groupby("gnaf_id").tail(1)
        t = ad[["gnaf_id", "weekly_rent", "car_space", "was_leased"]].merge(
            R.I[["gnaf_id", "bld", "suburb", "bedrooms", "fl", "car_space"]].rename(columns={"car_space": "car_in"}), on="gnaf_id")
        t["beds"] = t.bedrooms
        lo, hi, _ = R.predict(m, t, **PARAMS)
        car = t.car_space.fillna(t.car_in).fillna(0)
        t["gap"] = t.weekly_rent / np.where(car >= 1, np.exp(hi), np.exp(lo)) - 1
        bins = [-1, -0.05, 0.05, 0.15, 10]; labels = ["more than 5% below", "within 5%", "5-15% above", "over 15% above"]
        t["band"] = pd.cut(t.gap, bins, labels=labels)
        self.curve = t.groupby("band", observed=True).was_leased.agg(["mean", "count"])
        self.curve_n = len(t)

    # ---------- public API ----------
    def search(self, text, limit=15):
        q = " ".join(text.upper().replace(",", " ").split())
        if len(q) < 3: return self.I.iloc[0:0]
        import re
        mask = np.ones(len(self.I), bool)
        for w in q.split():   # numbers must match whole tokens: "101" is not "1101"
            pat = rf"(?<![0-9A-Z]){re.escape(w)}(?![0-9])" if w[0].isdigit() else re.escape(w)
            mask &= self.I.key.str.contains(pat, regex=True).values
        hits = self.I[mask].copy()
        hits["_rank"] = ~hits.building_address.str.upper().str.startswith(q.split()[0])   # building number first
        return hits.sort_values(["_rank", "address"]).head(limit).drop(columns="_rank")

    def check(self, gnaf_id, asking=None, has_parking=None):
        f = self.I[self.I.gnaf_id == gnaf_id].iloc[0]
        car = f.car_space if has_parking is None else float(has_parking)
        parking_known = not pd.isna(car)
        fair = f.fair_with if (parking_known and car >= 1) else f.fair_without
        err = TIER_ERR.get(f.tier, 60)
        sub = self.sub_trend["median"].get(f.suburb, np.nan)
        bld = self.bld_trend["median"].get(f.building_address, np.nan)
        out = dict(address=f.address, suburb=f.suburb, bedrooms=int(f.bedrooms), bathrooms=int(f.bathrooms),
                   parking=("yes" if car >= 1 else "no") if parking_known else "unknown",
                   fair=round(fair / 5) * 5, low=round((fair - err) / 5) * 5, high=round((fair + err) / 5) * 5,
                   fair_with=round(f.fair_with / 5) * 5, fair_without=round(f.fair_without / 5) * 5,
                   parking_premium=round((f.fair_with - f.fair_without) / 5) * 5,
                   evidence=f.tier.replace("_", " ").replace("own lease+", "own past adverts + "),
                   city_trend=self.city_trend, suburb_trend=None if np.isnan(sub) else float(sub),
                   building_trend=None if np.isnan(bld) else float(bld),
                   building_trend_n=int(self.bld_trend["count"].get(f.building_address, 0)),
                   building_recent=self.building_recent(f.building_address))
        if asking:
            gap = asking / fair - 1
            band = ("more than 5% below" if gap < -0.05 else "within 5%" if gap <= 0.05
                    else "5-15% above" if gap <= 0.15 else "over 15% above")
            out.update(asking=asking, gap=gap, band=band,
                       leased_share=float(self.curve["mean"].get(band, np.nan)),
                       advice=self.advice(asking, out, gap))
        return out

    def building_recent(self, bld):
        """aggregates only: median asking rent by bedrooms for the last 12 months in this building"""
        b = self.L[(self.L.bld == bld) & (self.L.listing_date > R.TASK_DATE - pd.DateOffset(years=1))]
        if b.empty: return None
        a = b.groupby("beds").weekly_rent.agg(["median", "count"]).reset_index()
        a = a[a["count"] >= 2]
        return a.rename(columns={"beds": "bedrooms", "median": "median_asking", "count": "adverts"}) if len(a) else None

    def advice(self, asking, o, gap):
        tips = []
        trend = o["suburb_trend"] if o["suburb_trend"] is not None else o["city_trend"]
        if gap > 0.05:
            tips.append(f"The asking rent is {gap:.0%} above our fair estimate of ${o['fair']:,}. "
                        f"A counter-offer near ${o['fair']:,} to ${o['high']:,} is backed by the data.")
            br = o.get("building_recent")
            if br is not None and (br.bedrooms == o["bedrooms"]).any():
                med = float(br.loc[br.bedrooms == o["bedrooms"], "median_asking"].iloc[0])
                tips.append(f"Other {o['bedrooms']}-bedroom flats in this building were advertised at a median ${med:,.0f}/week "
                            "over the last year. Use that as your comparable.")
            tips.append("Point to specifics the price should reflect (floor, view, condition, furnishing). "
                        "If the landlord won't move on rent, ask for a longer fixed term or a later start date instead.")
        elif gap < -0.05:
            tips.append(f"The asking rent is {-gap:.0%} below our fair estimate. It looks good value; "
                        "check condition and inclusions, and apply quickly.")
        else:
            tips.append("The asking rent is in line with the market. Negotiate on terms rather than price: "
                        "lease length, start date, furnishing, or a fixed rent for a longer lease.")
        if o["parking"] == "yes" and o["parking_premium"] > 0:
            tips.append(f"Parking is worth about ${o['parking_premium']}/week here. If you don't need the car space, "
                        f"ask whether it can be excluded for about that amount off (fair without parking: ${o['fair_without']:,}).")
        if trend is not None:
            tips.append(f"Rents in {o['suburb']} rose about {trend:.1%} over the last year. "
                        + ("Locking in a longer lease at today's rent protects you from the next increase."
                           if trend > 0.02 else "With rents flat, there's less reason to accept a premium."))
        return tips

if __name__ == "__main__":
    rc = RentCheck()
    print("city trend", round(rc.city_trend, 3), "| leasing curve (n=%d)" % rc.curve_n); print(rc.curve.round(2))
    hit = rc.search("101 bathurst")
    print(hit[["gnaf_id", "address", "bedrooms", "car_space"]].head(3))
    r = rc.check(hit.gnaf_id.iloc[0], asking=1150)
    for k, v in r.items():
        if k not in ("advice", "building_recent"): print(f"  {k}: {v}")
    print(r["building_recent"]); [print(" -", t) for t in r["advice"]]


# ---------------- live API features (used by app.py) ----------------
def score_listings(rc, listings, suburb, months_since_task=None):
    """Rate live listings against fair rent. Fair rent is as at 31 July 2026, so it is carried forward
    by the city trend for the months since. Returns a DataFrame sorted best value first."""
    import pandas as pd, numpy as np
    if not listings:
        return pd.DataFrame()
    if months_since_task is None:
        months_since_task = max(0.0, (pd.Timestamp.today() - pd.Timestamp("2026-07-31")).days / 30.44)
    carry = (1 + rc.city_trend) ** (months_since_task / 12)
    df = pd.DataFrame(listings)
    I = rc.I.set_index("gnaf_id")
    fair, basis = [], []
    for r in df.itertuples():
        if r.gnaf_id in I.index:
            f = I.loc[r.gnaf_id]
            park = (r.parking or 0) >= 1 if r.parking is not None else (f.car_space >= 1 if not pd.isna(f.car_space) else False)
            fair.append((f.fair_with if park else f.fair_without) * carry); basis.append("this flat")
        else:
            sub = rc.I[(rc.I.suburb == suburb) & (rc.I.bedrooms == r.bedrooms)]
            fair.append(sub.fair_without.median() * carry if len(sub) >= 5 else np.nan); basis.append("suburb, same beds")
    df["fair"] = np.round(np.array(fair, dtype=float) / 5) * 5
    df["vs_fair"] = df.price / df.fair - 1
    df["basis"] = basis
    return df.dropna(subset=["fair"]).sort_values("vs_fair")
