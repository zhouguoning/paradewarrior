# paradewarrior: Value every flat (Microburbs Hackathon 2026, Task 5)

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

- `flat-sales-city-of-sydney.csv` and `flats-city-of-sydney.csv`, supplied by Microburbs for the day. **Not included in this repo**: place them in `data/`.
- Addresses: Open G-NAF © Geoscape Australia, licensed by the Commonwealth of Australia.
- Nothing dated after 31 August 2026 is used. No API calls are needed to reproduce the result.

## How to run

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   macOS/Linux: source .venv/bin/activate
pip install pandas numpy
# put the supplied CSVs in data/
python model.py        # validation, held-out test, writes valuation_paradewarrior.csv + metrics.json (~30 s)
python make_report.py  # writes report.html from the metrics
```

## Files

| File | What it is |
|---|---|
| `model.py` | index, drift, value ladder, validation/test, final CSV |
| `make_report.py` | builds `report.html` (aggregates only) |
| `valuation_paradewarrior.csv` | the submission: `gnaf_id,value` for all 181,711 addresses |
| `metrics.json` | validation, test, ablation and tier scores |
| `report.html` | findings, method, held-out check |
| `pitch.mp4`, `demo.mp4` | videos |

Data © Microburbs. Estimates are automated and indicative only.
