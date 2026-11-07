"""
Population scenarios for City of Sydney rents.

Calibration 1 (cross-section, supplied data): across suburbs, how did rent growth 2021->2026 relate to
  growth in people and in homes (Microburbs block estimates, 2021 Census base)?
Calibration 2 (history): the 2020-21 border-closure episode in our own repeat-rent index.
Simulation: Monte Carlo over the estimated elasticities (bootstrap across suburbs) for user-defined
  population and supply shocks.
"""
import json, numpy as np, pandas as pd
import rent_vs_rates as RV          # suburb annual rent indices (sr), city quarterly rent index

B = pd.read_csv("data/blocks-homes-and-people-2021-2026.csv")
s = B.groupby("suburb")[["homes_2021", "homes_2026", "people_2021", "people_2026"]].sum()
s = s[(s.homes_2021 > 300) & (s.people_2021 > 500)]
s["d_people"] = np.log(s.people_2026 / s.people_2021)
s["d_homes"] = np.log(s.homes_2026 / s.homes_2021)
s["d_crowd"] = s.d_people - s.d_homes                         # people per home
sr = RV.suburb_annual(RV.pr, min_pairs=40)                     # suburb annual repeat-rent index
s["rent_g"] = (sr.loc[2026] - sr.loc[2021]).reindex(s.index)   # log rent growth 2021 -> 2026
d = s.dropna(subset=["rent_g"])
print(f"{len(d)} suburbs.  people {d.d_people.mean():+.1%} avg, homes {d.d_homes.mean():+.1%} avg, rents {d.rent_g.mean():+.1%} avg (log)")

def fit(df, cols):
    X = np.column_stack([np.ones(len(df)), df[cols].values]); b = np.linalg.lstsq(X, df.rent_g.values, rcond=None)[0]
    e = df.rent_g.values - X @ b; r2 = 1 - e @ e / ((df.rent_g - df.rent_g.mean()) ** 2).sum()
    return b, r2
rng = np.random.default_rng(1)
res = {}
for name, cols in [("crowding (people per home)", ["d_crowd"]), ("people and homes separately", ["d_people", "d_homes"])]:
    b, r2 = fit(d, cols)
    boots = np.array([fit(d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))), cols)[0] for _ in range(2000)])
    res[name] = dict(coef={c: round(float(b[i+1]), 3) for i, c in enumerate(cols)}, r2=round(float(r2), 2),
                     ci90={c: [round(float(np.percentile(boots[:, i+1], 5)), 3), round(float(np.percentile(boots[:, i+1], 95)), 3)] for i, c in enumerate(cols)})
    print(name, res[name])
corr = d[["d_people", "d_homes", "d_crowd", "rent_g"]].corr().round(2)
print(corr)

# calibration 2: border closure in the city rent index
ri = RV.df.rent
covid = dict(pre_peak=str(ri.loc["2019Q1":"2020Q1"].idxmax()), trough=str(ri.loc["2020Q1":"2022Q1"].idxmin()))
covid["fall_pct"] = round(100 * (np.exp(ri.loc["2020Q1":"2022Q1"].min() - ri.loc["2019Q1":"2020Q1"].max()) - 1), 1)
covid["rebound_to_2023Q4_pct"] = round(100 * (np.exp(ri.loc["2023Q4"] - ri.loc["2020Q1":"2022Q1"].min()) - 1), 1)
print("border-closure episode:", covid)

# Monte Carlo scenarios using the crowding elasticity (bootstrap distribution)
b_cr = np.array([fit(d.sample(len(d), replace=True, random_state=i), ["d_crowd"])[0][1] for i in range(4000)])
def scenario(people_pct, homes_pct):
    shock = np.log1p(people_pct) - np.log1p(homes_pct)
    ch = np.exp(b_cr * shock) - 1
    return dict(median=round(100 * float(np.median(ch)), 1), p10=round(100 * float(np.percentile(ch, 10)), 1),
                p90=round(100 * float(np.percentile(ch, 90)), 1))
scen = {}
for name, p, h in [("fewer students/temporary migrants: renters -2%, supply +1%", -0.02, 0.01),
                   ("stronger cut: renters -5%, supply +1%", -0.05, 0.01),
                   ("COVID-scale exodus: renters -10%, supply +1%", -0.10, 0.01),
                   ("migration rebound: renters +3%, supply +1%", 0.03, 0.01),
                   ("supply surge only: renters 0%, supply +4%", 0.0, 0.04)]:
    scen[name] = scenario(p, h); print(f"SCENARIO {name}: rent change {scen[name]['median']:+.1f}% (p10 {scen[name]['p10']:+.1f}%, p90 {scen[name]['p90']:+.1f}%)")
json.dump(dict(n_suburbs=len(d), fits=res, covid_episode=covid, scenarios_pct=scen,
               elasticity_crowding_median=round(float(np.median(b_cr)), 2)), open("rent_population.json", "w"), indent=2)

# ---- Calibration 2 drives the simulation: short-run elasticity from the border closure ----
# Rents fell covid["fall_pct"] while inner-city renter populations fell. The City of Sydney's own drop is not in our
# data; the closest official figure is Melbourne LGA, -7.5% in 2020-21 (ABS / BCARR). We treat the City's renter-
# population drop as uncertain, uniform 5%..12%, and the rent response as 70-100% due to demand (rest: lockdown discounts).
rng = np.random.default_rng(7)
N = 20000
pop_drop = rng.uniform(0.05, 0.12, N)
demand_share = rng.uniform(0.7, 1.0, N)
elast = (np.log1p(covid["fall_pct"] / 100) * demand_share) / np.log1p(-pop_drop)     # % rent per % renters
print(f"\nshort-run elasticity of rent to renter population: median {np.median(elast):.2f} (p10 {np.percentile(elast,10):.2f}, p90 {np.percentile(elast,90):.2f})")
def scenario2(renters_pct, supply_pct=0.0):
    ch = np.exp(elast * (np.log1p(renters_pct) - np.log1p(supply_pct))) - 1
    return dict(median=round(100 * float(np.median(ch)), 1), p10=round(100 * float(np.percentile(ch, 10)), 1),
                p90=round(100 * float(np.percentile(ch, 90)), 1))
scen2 = {}
for name, p, h in [("renters -1% (mild student/visa tightening)", -0.01, 0.0),
                   ("renters -3% (strong tightening)", -0.03, 0.0),
                   ("renters -3%, new supply +2%", -0.03, 0.02),
                   ("renters -8% (COVID-scale exodus)", -0.08, 0.0),
                   ("renters +2% (migration rebound)", 0.02, 0.0)]:
    scen2[name] = scenario2(p, h)
    print(f"SCENARIO2 {name}: rent change {scen2[name]['median']:+.1f}% (p10 {scen2[name]['p10']:+.1f}%, p90 {scen2[name]['p90']:+.1f}%)")
json.dump(dict(n_suburbs=len(d), cross_section=res, covid_episode=covid,
               elasticity_short_run=dict(median=round(float(np.median(elast)), 2), p10=round(float(np.percentile(elast, 10)), 2), p90=round(float(np.percentile(elast, 90)), 2)),
               scenarios_pct=scen2), open("rent_population.json", "w"), indent=2)
