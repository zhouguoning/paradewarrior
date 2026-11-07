# paradewarrior: Sydney flat values and rents (Microburbs Hackathon 2026, Tasks 4 and 5)

We entered two data tasks with one engine: a repeat-sales (or repeat-rent) index plus a value ladder that always uses the most local evidence available, the flat's own history first, then its own building, then its suburb.

## Task 4: what will this flat rent for?

**What we built:** two weekly asking rents, with and without a car space, for all 65,777 flats as at 31 July 2026 (`rents_paradewarrior.csv`).

**Who it helps:** tenants checking whether an asking rent is too high, landlords setting a fair rent, and investors comparing likely rent before buying.

**Result:** on 2,592 held-out adverts (1 June to 31 July 2026) our typical miss was **$44 a week** and 77.7% were within $100, against $94 and 51.4% for a suburb x bedrooms median on the same adverts. R² 0.85.

**The clever step:** parking is priced *inside* the same building and bedroom count (about $70 a week, highest for one-bedders), every past advert is normalised to its no-parking equivalent, and the rent trend is carried forward from the cutoff to the scoring window.

### Fair Rent Check app

A renter's tool built on the Task 4 model: type any City of Sydney address and optionally the asking rent, and see the fair rent
(with and without parking), a realistic range, the suburb and building rent trend, how the asking rent compares, and negotiation pointers.
It runs locally from the supplied files, so no data is published.

```bash
pip install streamlit pandas numpy
streamlit run app.py
```

## Task 5: value every flat

**What we built:** a sale-value estimate as at 31 August 2026 for all 181,711 unit-style addresses in the City of Sydney (`valuation_paradewarrior.csv`).

**Who it helps:** buyers and sellers who want a price to check before negotiating, and landlords and everyday investors sizing up a flat.

**Result:** on 421 held-out flat sales (1 June to 31 August 2026, under $1.2M), 53.9% of estimates were within 10% of the sale price; on the 234 flats the baseline can value, 56.0% against 48.3% for the task page's best baseline (last sale x citywide index). Details in [`report.html`](report.html).

## The clever step

A value ladder that always uses the most local evidence available:

1. **Own past sale**, moved to 31 Aug 2026 with our own repeat-sales index (quarterly, smoothed, outlier-trimmed), plus each suburb's and building's growth above or below the city, shrunk toward zero where evidence is thin.
2. **Same building, same bedrooms.** 73% of addresses have no bedroom count, so we borrow it from the flats stacked directly above and below (Unit 1204 sits over Unit 1104 with the same plan). Comparables are also adjusted for floor (about 0.5% a floor, read from the unit number). Stacking lifted bedroom-matched building comparables by about 45%.
3. Suburb and bedrooms, suburb, city as fallbacks, so no row is ever blank.

Own-sale and building estimates are blended in log space, trusting old sales less. Every parameter was tuned on a separate March to May 2026 validation fold before the June to August test was scored once.

## Data

- `rents-past-leases-city-of-sydney.csv`, `rents-input-flats-city-of-sydney.csv`, `flat-sales-city-of-sydney.csv` and `flats-city-of-sydney.csv`, supplied by Microburbs for the day. **Not included in this repo**: place them in `data/`.
- Addresses: Open G-NAF © Geoscape Australia, licensed by the Commonwealth of Australia.
- Nothing dated after 31 July 2026 (rents) or 31 August 2026 (valuation) is used. No API calls are needed to reproduce the result.

## How to run

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   macOS/Linux: source .venv/bin/activate
pip install pandas numpy
# put the supplied CSVs in data/
python model.py        # validation, held-out test, writes valuation_paradewarrior.csv + metrics.json (~30 s)
python rents.py        # Task 4: tuning, held-out test, writes rents_paradewarrior.csv + rent_metrics.json (~1 min)
python make_report.py  # writes report.html (both tasks) from the metrics
```

## Files

| File | What it is |
|---|---|
| `model.py` | index, drift, value ladder, validation/test, final CSV |
| `make_report.py` | builds `report.html` (aggregates only) |
| `rents.py` | Task 4: repeat-rent index, parking premium, ladder, validation/test, final CSV |
| `make_rent_section.py` | Task 4 part of `report.html` |
| `rents_paradewarrior.csv` | Task 4 submission: `gnaf_id,rent_with_car_space,rent_without_car_space` |
| `rent_metrics.json` | Task 4 scores |
| `valuation_paradewarrior.csv` | the submission: `gnaf_id,value` for all 181,711 addresses |
| `metrics.json` | validation, test, ablation and tier scores |
| `report.html` | findings, method, held-out check |
| `pitch.mp4`, `demo.mp4` | videos |

Data © Microburbs. Estimates are automated and indicative only.
