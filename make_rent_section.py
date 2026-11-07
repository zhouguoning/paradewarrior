"""Returns the Task 4 HTML section for report.html (aggregates only)."""
import json, html
import numpy as np, pandas as pd

def section(bar_svg):
    m = json.load(open("rent_metrics.json"))
    T, B, F = m["test"], m["baseline_suburb_beds"], m["final"]
    P = m["validation"]["params"]; folds = m["validation"]["per_fold"]
    usd = lambda x: f"${x:,.0f}"; pct = lambda x: f"{100*x:.1f}%"
    prem = F["parking_premium_pct_by_bedrooms"]
    prem_items = [(f"{'studio' if k=='0' else k+' bed' if k!='3' else '3+ bed'}", float(v)) for k, v in prem.items()]
    abl = "".join(f"<tr><td>{html.escape(k)}</td><td>{usd(v['median_abs_err'])}</td><td>{pct(v['within100'])}</td></tr>"
                  for k, v in m["ablation"].items())
    abl += f"<tr><td>Baseline: suburb × bedrooms median</td><td>{usd(B['median_abs_err'])}</td><td>{pct(B['within100'])}</td></tr>"
    tiers = "".join(f"<tr><td>{k.replace('_',' ')}</td><td>{v['n']:,}</td><td>{usd(v['median_abs_err'])}</td><td>{pct(v['within100'])}</td></tr>"
                    for k, v in sorted(m["test_by_tier"].items(), key=lambda kv: -kv[1]["n"]))
    final_t = "".join(f"<tr><td>{k.replace('_',' ')}</td><td>{v:,}</td></tr>" for k, v in F["tiers"].items())
    return f"""
<h1 id="task4">Task 4: what will this flat rent for?</h1>
<p>We estimated two weekly asking rents, with and without a car space, for all 65,777 flats as at 31 July 2026 (<code>rents_paradewarrior.csv</code>).
On {T['n']:,} later adverts we held out (1 June to 31 July 2026), our typical miss was <b>{usd(T['median_abs_err'])} a week</b> and
<b>{pct(T['within100'])}</b> landed within $100. A suburb-and-bedrooms median on the same adverts missed by {usd(B['median_abs_err'])}
({pct(B['within100'])} within $100). The task page reports $100 for its simple estimate and $86 for the Microburbs rental valuation on the judging set.</p>

<h2>What we found along the way</h2>
<div class="big">
<div><b>{usd(F['median_weekly_parking_premium'])}/wk</b>typical parking premium, measured inside the same building and bedroom count</div>
<div><b>{100*F['trend_q']:+.1f}%</b>rent growth per quarter over the year to July 2026, from {F['repeat_pairs']:,} repeat adverts</div>
<div><b>{usd(T['median_abs_err'])}</b>typical miss on {T['n']:,} held-out adverts</div>
</div>
<p><b>Parking is worth most to one-bedders.</b> Comparing adverts in the same building with the same bedroom count, a car space added (%):</p>
{bar_svg(prem_items, unit="%")}
<p>Studios rarely come with parking, and three-bedders usually do, so a car space separates one-bedders more than anyone else.
Comparing within the building strips out location and building quality, which a suburb-wide comparison would wrongly credit to parking.</p>
<p><b>Rents keep moving after the cutoff.</b> Rents rose about {100*F['trend_q']:.1f}% a quarter over the last year. Carrying half a quarter of that trend
forward to the August to September scoring window was the single biggest improvement in our check. The task page notes the Microburbs valuation ran
$61 a week low, which is consistent with this lag.</p>
<p><b>The flat's own last advert is the best clue.</b> 75% of flats had been advertised before. Moved forward with our repeat-rent index and blended with
same-building, same-bedroom adverts, those flats are where we are most accurate.</p>

<h2>What our held-out check showed</h2>
<div class="big"><div><b>{usd(T['median_abs_err'])}</b>median absolute miss</div><div><b>{pct(T['within100'])}</b>within $100 a week</div><div><b>{T['r2']:.2f}</b>R²</div></div>
<p>We tuned every setting on two earlier windows: August to September 2025 (the same season as the judging window) and February to March 2026
(typical misses {usd(folds[0]['median_abs_err'])} and {usd(folds[1]['median_abs_err'])}). Only then did we rebuild on adverts dated by
<b>31 May 2026</b> and score 1 June to 31 July once. No later advert was used to build or tune anything. The scored rent uses the column
matching the advert's parking, or the flat's recorded parking where the advert is silent, as the judges do.</p>
<h3>Which steps helped (test window)</h3>
<table><tr><th>Model</th><th>Median miss</th><th>Within $100</th></tr>{abl}</table>
<h3>Accuracy by evidence (test window)</h3>
<table><tr><th>Evidence used</th><th>Adverts</th><th>Median miss</th><th>Within $100</th></tr>{tiers}</table>

<h2>Data and method</h2>
<ul>
<li><code>rents-past-leases-city-of-sydney.csv</code>: 112,457 adverts from 2012 to 31 July 2026, and <code>rents-input-flats-city-of-sydney.csv</code>
(Microburbs, supplied for the day; not republished). Nothing after 31 July 2026 was used, and no live listings.</li>
<li><b>Repeat-rent index:</b> quarterly regression on {F['repeat_pairs']:,} pairs of adverts for the same flat with the same parking status, smoothed and outlier-trimmed.</li>
<li><b>Parking normalisation:</b> every advert is converted to its "no car space" equivalent using the within-building premium for its bedroom band,
so flats with and without parking can be compared directly. The with-parking column adds the premium back.</li>
<li><b>Ladder:</b> own last advert (time-adjusted, trusted less as it ages) blended with same-building, same-bedroom adverts (shrunk toward the suburb
when the building has few), then suburb × bedrooms, suburb, bedrooms, city. All 65,777 flats get both rents, including those outside the City block file.</li>
</ul>
<p class="muted">Final evidence mix:</p><table><tr><th>Evidence</th><th>Flats</th></tr>{final_t}</table>

<h2>What remains uncertain</h2>
<ul>
<li>Our check covers June to July; the judges score August to September, which is spring leasing season. We tuned on last year's August to September to cover this.</li>
<li>A floor premium (about {100*F['floor_slope']:.1f}% a floor) did not improve rents on validation, so it is off, unlike in our sale-price model.</li>
<li>Flats never advertised before (about a quarter) rely on building and suburb comparables and are less accurate.</li>
</ul>
"""
