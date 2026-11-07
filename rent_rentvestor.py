"""
Rentvestor squeeze theory (KZ):
  rate rises -> owners who also rent face higher mortgage costs -> can't pay higher rent for their own home,
  can't fully pass costs to their tenants -> trade down to cheaper rentals -> rents fall -> spiral?

Test 1 (data): trading down predicts CHEAP flats' rents rise faster than EXPENSIVE flats' during the hiking cycle.
  Repeat adverts of the same flat; growth by price quintile; hiking cycle vs a calm period.
Test 2 (data): did landlords' re-listing increases jump during the hikes (cost pass-through attempts)?
Test 3 (simulation): feedback loop of a household population with rentvestors; when does it spiral?
"""
import json, numpy as np, pandas as pd
import rents as R
L = R.L.sort_values(["gnaf_id", "listing_date"])
g = L.groupby("gnaf_id")
p = pd.DataFrame({"d1": g.listing_date.shift(), "d2": L.listing_date, "r1": g.weekly_rent.shift(), "r2": L.weekly_rent,
                  "c1": g.car.shift(), "c2": L.car, "beds": L.beds}).dropna(subset=["d1"])
p["hold"] = (p.d2 - p.d1).dt.days / 365.25
p = p[p.hold.between(0.75, 3) & ((p.c1 == p.c2) | p.c1.isna())]
p["ann"] = np.log(p.r2 / p.r1) / p.hold
p = p[p.ann.abs() < 0.5]
out = {}
periods = {"calm 2016-2019 (rates flat/low)": ("2016-01-01", "2019-12-31"),
           "hiking cycle 2022-05 to 2024-12": ("2022-05-01", "2024-12-31")}
print("Test 1: annualised rent growth of the same flat, by its starting rent quintile")
for name, (a, b) in periods.items():
    q = p[(p.d1 >= a) & (p.d2 <= b)].copy()
    q["quintile"] = pd.qcut(q.r1.rank(method="first"), 5, labels=["Q1 cheapest", "Q2", "Q3", "Q4", "Q5 dearest"])
    t = q.groupby("quintile", observed=True).ann.agg(median="mean", count="count")
    gap = t["median"].iloc[0] - t["median"].iloc[-1]
    out[name] = dict(growth_pct={k: round(100 * v, 1) for k, v in t["median"].items()}, n=int(t["count"].sum()),
                     cheap_minus_dear_pts=round(100 * gap, 1))
    print(f"  {name}: " + ", ".join(f"{k} {100*v:+.1f}%" for k, v in t["median"].items()) + f"   cheap-dear gap {100*gap:+.1f} pts  (n={int(t['count'].sum())})")
# same test by bedrooms (trading down from 2-bed to 1-bed / studio)
for name, (a, b) in periods.items():
    q = p[(p.d1 >= a) & (p.d2 <= b)]
    t = q.groupby(q.beds.clip(0, 3)).ann.mean()
    out[name]["by_bedrooms_pct"] = {str(int(k)): round(100 * v, 1) for k, v in t.items()}
    print(f"  {name} by bedrooms: " + ", ".join(f"{int(k)}bd {100*v:+.1f}%" for k, v in t.items()))
# Test 2: share of re-listings that ask >10%/yr more, by year
p["year"] = p.d2.dt.year
t2 = p[p.year.between(2016, 2026)].groupby("year").ann.agg(median="median", big_rise=lambda s: (s > 0.10).mean())
out["relist_by_year"] = {int(k): dict(median_pct=round(100 * v["median"], 1), share_over_10pct=round(100 * v["big_rise"], 0)) for k, v in t2.iterrows()}
print("\nTest 2: re-listing increases by year (median annualised, share asking >10%/yr more)")
print((t2 * 100).round(1).T)

# ---------------- Test 3: feedback simulation ----------------
# Parameters (assumptions, sampled over ranges; no City-level rentvestor data in this dataset):
#   phi   share of renter households who also own an investment property      5%..20%
#   lev   mortgage cost increase as share of their own rent budget for a +4.25pt rate rise   20%..60%
#   s     share of that squeeze taken out of their own rent (rest: savings, spending)    10%..50%
#   h     rent received on their investment / rent paid for their home         0.6..1.4
#   pass  share of the cost increase they manage to pass to their own tenants   0%..50%
#   e     rent elasticity to effective renter demand (from the border-closure calibration)  1.3..2.7
rng = np.random.default_rng(3); N = 20000
phi = rng.uniform(0.05, 0.20, N); lev = rng.uniform(0.2, 0.6, N); s = rng.uniform(0.1, 0.5, N)
h = rng.uniform(0.6, 1.4, N); pas = rng.uniform(0.0, 0.5, N); e = rng.uniform(1.3, 2.7, N)
# round 0: direct demand cut from the squeeze (net of what they pass on to tenants)
d0 = phi * s * lev * (1 - pas)
# each round: a rent change dR changes their budget by (h - 1) * dR (they receive h, pay 1, per unit of rent)
#   -> extra demand change phi * s * (h - 1) * dR -> extra rent change e * that
gain = e * phi * s * (h - 1)                 # loop gain; |gain| >= 1 would be a spiral
direct = -e * d0                             # first-round rent change
total = direct / (1 - gain)                  # geometric sum of the feedback rounds (valid while |gain| < 1)
out["simulation"] = dict(
    first_round_rent_change_pct=[round(100 * float(np.percentile(direct, q)), 1) for q in (10, 50, 90)],
    loop_gain=[round(float(np.percentile(gain, q)), 3) for q in (1, 50, 99)],
    share_runs_spiral=round(float((gain >= 1).mean()) * 100, 2),
    total_rent_change_pct=[round(100 * float(np.percentile(total, q)), 1) for q in (10, 50, 90)])
print("\nTest 3 simulation (20,000 runs):", out["simulation"])
# what would it take to spiral?  gain = e*phi*s*(h-1) >= 1
need_phi = 1 / (2.7 * 0.5 * (1.4 - 1))
print(f"  spiral needs e*phi*s*(h-1) >= 1, e.g. with e=2.7, s=0.5, h=1.4 it needs phi >= {need_phi:.0%} of renters to be rentvestors")
out["spiral_requirement_phi_at_extreme_params"] = round(need_phi, 2)
json.dump(out, open("rent_rentvestor.json", "w"), indent=2, default=str)
