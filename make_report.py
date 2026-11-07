"""Builds report.html from model.py outputs (metrics.json, test_predictions.csv). Only aggregates are published."""
import json, html
import numpy as np, pandas as pd
import model as M
from make_rent_section import section as rent_section

m = json.load(open("metrics.json"))
tp = pd.read_csv("test_predictions.csv", parse_dates=["sold_date"])
P = m["validation"]["params"]; half = P["half"]

# --- extra insights from the final model (data to 31 Aug 2026)
mf = M.build_model(M.FINAL_DATE + pd.Timedelta(days=1))
idx = mf["idx"]; Q = len(idx)
qdates = pd.period_range("1995Q1", periods=Q, freq="Q")
city = pd.Series(np.exp(idx - idx[qdates.get_loc(pd.Period("2017Q1"))]), index=qdates)
yoy = float(np.exp(idx[-1] - idx[-5]) - 1)
since17 = float(city.iloc[-1] - 1)
p = M.pairs_of(M.S[M.S.sold_date <= M.FINAL_DATE])
ds = M.drift(p, idx, "suburb", 15, M.FINAL_DATE).sort_values()
_, t_nostack = M.predict(mf, M.F, **{**P, "use_stack": False})
_, t_stack = M.predict(mf, M.F, **P)
bb_before = int(np.isin(t_nostack, ["building_beds", "own_sale+building_beds"]).sum())
bb_after = int(np.isin(t_stack, ["building_beds", "own_sale+building_beds"]).sum())

T = m["test"]; V = m["validation"]; B = m["baseline_citywide_last_sale"]; MS = m["model_on_same_subset"]
e = np.abs(tp.pred / tp.price - 1)
buckets = [("within 5%", (e <= .05).sum()), ("5–10%", ((e > .05) & (e <= .10)).sum()),
           ("10–20%", ((e > .10) & (e <= .20)).sum()), ("over 20%", (e > .20).sum())]
pct = lambda x: f"{100*x:.1f}%"
usd = lambda x: f"${x:,.0f}"

def bar_svg(items, color="#2b6cb0", w=640, unit=""):
    h = 34 * len(items) + 10; mx = max(v for _, v in items) or 1
    rows = []
    for i, (k, v) in enumerate(items):
        bw = 380 * v / mx; y = 8 + 34 * i
        rows.append(f'<text x="0" y="{y+18}" class="lab">{html.escape(str(k))}</text>'
                    f'<rect x="200" y="{y+4}" width="{bw:.1f}" height="20" rx="3" fill="{color}"/>'
                    f'<text x="{206+bw:.1f}" y="{y+19}" class="val">{v}{unit}</text>')
    return f'<svg viewBox="0 0 {w} {h}" role="img">{"".join(rows)}</svg>'

def line_svg(s, w=640, h=220):
    s = s[s.index >= pd.Period("2005Q1")]
    x = np.linspace(40, w - 10, len(s)); lo, hi = s.min(), s.max()
    y = h - 30 - (s.values - lo) / (hi - lo) * (h - 50)
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    ticks = "".join(f'<text x="{x[i]:.0f}" y="{h-8}" class="lab" text-anchor="middle">{s.index[i].year}</text>'
                    for i in range(0, len(s), 12))
    return (f'<svg viewBox="0 0 {w} {h}" role="img"><polyline points="{pts}" fill="none" stroke="#2b6cb0" stroke-width="2.5"/>'
            f'{ticks}<text x="40" y="14" class="lab">City of Sydney flat price index (2017 Q1 = 1.00); now {s.iloc[-1]:.2f}</text></svg>')

ablation = "".join(f"<tr><td>{html.escape(k)}</td><td>{pct(v['within10'])}</td><td>{usd(v['median_abs_err'])}</td></tr>"
                   for k, v in m["ablation"].items())
tiers = "".join(f"<tr><td>{k.replace('_',' ')}</td><td>{v['n']}</td><td>{pct(v['within10'])}</td><td>{pct(v['median_pct_err'])}</td></tr>"
                for k, v in sorted(m["test_by_tier"].items(), key=lambda kv: -kv[1]["n"]))
final_tiers = "".join(f"<tr><td>{k.replace('_',' ')}</td><td>{v:,}</td></tr>" for k, v in m["final_tiers"].items())
drift_items = [(k, round(100 * v, 1)) for k, v in pd.concat([ds.head(4), ds.tail(4)]).items()]

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>paradewarrior: Sydney flats</title>
<style>
:root{{--fg:#1a202c;--muted:#4a5568;--bg:#fff;--card:#f7fafc;--acc:#2b6cb0;--line:#e2e8f0}}
@media (prefers-color-scheme:dark){{:root{{--fg:#e2e8f0;--muted:#a0aec0;--bg:#1a202c;--card:#2d3748;--acc:#63b3ed;--line:#4a5568}}}}
body{{font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--fg);background:var(--bg);margin:0;padding:0 16px}}
main{{max-width:760px;margin:40px auto}} h1{{font-size:1.9rem;line-height:1.25}} h2{{margin-top:2.2em;border-top:1px solid var(--line);padding-top:1em}}
.big{{display:flex;gap:16px;flex-wrap:wrap}} .big div{{background:var(--card);border-radius:8px;padding:14px 18px;flex:1;min-width:150px}}
.big b{{display:block;font-size:1.7rem;color:var(--acc)}} .muted{{color:var(--muted)}}
table{{border-collapse:collapse;width:100%;font-size:.93rem}} td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}}
svg{{width:100%;height:auto;margin:10px 0}} svg .lab{{font-size:12px;fill:var(--muted)}} svg .val{{font-size:12px;fill:var(--fg)}}
code{{background:var(--card);padding:1px 5px;border-radius:4px}}
</style></head><body><main>
<p class="muted">Microburbs Hackathon · Task 5 · team paradewarrior · 7 November 2026</p>
<p><a href="#task5">Task 5: sale values</a> · <a href="#task4">Task 4: weekly rents</a></p>
<h1 id="task5">Task 5: value every flat, with a repeat-sales ladder that knows which building you're in</h1>
<p>We valued all 181,711 City of Sydney addresses as at 31 August 2026. On {T['n']} later flat sales we held out
(1 June to 31 August 2026, under $1.2M), <b>{pct(T['within10'])}</b> of our estimates landed within 10% of the sale price.
On the same flats where the task page's best baseline applies (last sale updated by a citywide index), that baseline scored
{pct(B['within10'])} and we scored {pct(MS['within10'])}, with typical dollar error down from {usd(B['median_abs_err'])} to {usd(MS['median_abs_err'])}.</p>

<h2>What we found along the way</h2>
<div class="big">
<div><b>{since17*100:+.0f}%</b>City of Sydney flat prices since 2017 Q1, from our repeat-sales index</div>
<div><b>{yoy*100:+.1f}%</b>change over the 12 months to Q3 2026</div>
<div><b>{bb_before:,} → {bb_after:,}</b>flats with a bedroom-matched building comparable after stacked-unit imputation</div>
</div>
{line_svg(city)}
<p><b>Flats are a slow market.</b> The index shows unit prices in the City of Sydney have barely moved in real terms over the last decade,
so a flat's own last sale, even an old one, is a strong anchor once it's moved forward by the index.</p>
<p><b>Suburbs drift apart from the city.</b> We measured each suburb's annual growth above or below the citywide index over the last four years,
shrunk toward zero where evidence is thin. Biggest gaps, % a year versus the city:</p>
{bar_svg(drift_items, unit="%")}
<p><b>Stacked units share layouts.</b> 73% of addresses have no bedroom count. But Unit 1204 usually sits directly above Unit 1104 with the same floor plan,
so we borrowed bedrooms from the same position on other floors of the same building. That raised by about 45% the number of flats we can match to a
same-building, same-bedroom comparable.</p>
<p><b>Height pays, a little.</b> Within the same building and bedroom count, each extra floor added about
{100*m['final_floor_slope']:.1f}% to the sale price, so floor 15 versus floor 2 is worth roughly {100*(np.exp(13*m['final_floor_slope'])-1):.0f}%.
We read the floor straight out of the unit number (Unit 1204 is on floor 12).</p>
<p><b>The scoring window is selective.</b> Only sales under $1.2M are scored. On our test our estimates ran a median {pct(T['median_pct_err'])} high,
part of which is this truncation: expensive flats that happen to sell cheaply stay in, cheap flats that sell dear drop out.</p>

<h2>What our held-out check showed</h2>
<div class="big">
<div><b>{pct(T['within10'])}</b>within 10% ({T['n']} test sales)</div>
<div><b>{usd(T['median_abs_err'])}</b>median absolute error</div>
<div><b>{T['r2']:.2f}</b>R² (dollar scale)</div>
</div>
{bar_svg([(k, int(v)) for k, v in buckets])}
<p>We picked every tuning parameter on two earlier validation folds (December 2025 to February 2026, and March to May 2026;
average {pct(V['within10'])} within 10% over {V['n']} sales). Only then did we rebuild on sales before <b>1 June 2026</b> and score June to August once.
No test-period price was used to build or tune anything. All {T['n']} test flats received a prediction.</p>
<p>R² is low because a handful of large misses dominate a squared-error metric; the judged metric (within 10%) is about the typical flat, where we do well.</p>

<h3>Which steps helped (test fold)</h3>
<table><tr><th>Model</th><th>Within 10%</th><th>Median $ error</th></tr>{ablation}</table>

<h3>Accuracy by evidence tier (test fold)</h3>
<table><tr><th>Evidence used</th><th>Sales</th><th>Within 10%</th><th>Median bias</th></tr>{tiers}</table>

<h2>Data</h2>
<ul>
<li><code>flat-sales-city-of-sydney.csv</code>: 81,078 sales from 1995 to 31 August 2026 (Microburbs, supplied for the day; not republished).</li>
<li><code>flats-city-of-sydney.csv</code>: the 181,711 target addresses with unit number, building size and last-advert bedrooms.</li>
<li>We built our own repeat-sales pairs from the sales file (it runs to 31 August, the supplied pair file stops in April 2026).</li>
<li>Addresses: Open G-NAF © Geoscape Australia. Nothing dated after 31 August 2026 was used.</li>
</ul>

<h2>Method</h2>
<ol>
<li><b>Repeat-sales index.</b> Quarterly Case-Shiller-style regression of log price change on time dummies, weighted down for long holds,
smoothed with a second-difference penalty, outliers trimmed at 4 MAD.</li>
<li><b>Local drift.</b> Each suburb's and building's growth above or below the city, as % per year, shrunk toward zero (Bayesian-style, by holding years).</li>
<li><b>Value ladder.</b> Every past sale is moved to 31 August 2026. For each flat we take the best evidence available:
same building and bedrooms → same building (lower quantile when bedrooms unknown) → suburb and bedrooms → suburb → city.</li>
<li><b>Floor and thin cells.</b> Building comparables are moved to the target flat's floor, and building-and-bedroom groups with few sales are shrunk toward the suburb-and-bedroom median.</li>
<li><b>Blend.</b> If the flat sold before, blend its own index-adjusted sale with the building comparable in log space,
trusting old sales less (weight halves after {half:g} years).</li>
</ol>
<p class="muted">Final evidence mix across all 181,711 addresses:</p>
<table><tr><th>Evidence</th><th>Addresses</th></tr>{final_tiers}</table>

<h2>What did not work, and what remains uncertain</h2>
<ul>
<li>Flats with an old sale but no known bedrooms are weak (few test cases, large misses): the building may mix studios and 3-bedders.</li>
<li>A global downward shift for the $1.2M truncation did not help on validation, so we left it out rather than tune on the test.</li>
<li>Tuning on two folds raised validation accuracy from 55.9% to 57.5%, but on the final test it scored 53.9% against 54.9% for our simpler first version.
With 421 test sales, one percentage point is about four flats, well inside the noise, so we kept the validation-chosen settings rather than pick by test score.
Like-for-like against the baseline it improved (56.0% vs 53.4% for v1), and R² rose from 0.12 to 0.20.</li>
<li>About 6,700 shop and office titles get a value but are probably poorly valued; they are not scored.</li>
<li>Not yet used: aspect and views. Both need outside data (building footprints and orientation) and a new dated check.</li>
</ul>
{rent_section(bar_svg)}
<p class="muted">Estimates are automated and indicative only. Data © Microburbs.</p>
</main></body></html>"""
open("report.html", "w", encoding="utf-8").write(page)
print("report.html written", len(page), "bytes;", "yoy", round(yoy, 4), "since17", round(since17, 3), "stack", bb_before, bb_after)
print(ds.head(4).round(3).to_dict(), ds.tail(4).round(3).to_dict())
