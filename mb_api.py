"""Small, safe Microburbs API client for the app. Key comes from .env and stays server-side."""
import os, time, urllib.parse
import requests
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

BASE = "https://api-ht.microburbs.com.au"

class MBError(Exception):
    pass

def _get(path, params=None, retries=3):
    key = os.environ.get("MICROBURBS_API_KEY")
    if not key:
        raise MBError("No MICROBURBS_API_KEY in .env")
    for i in range(retries):
        r = requests.get(BASE + path, params=params, headers={"Authorization": f"Bearer {key}"}, timeout=15)
        if r.status_code == 429:                 # ~10 requests/second per key: back off and retry
            time.sleep(1.0 * (i + 1)); continue
        if r.status_code == 403:
            raise MBError("Outside the 29 City of Sydney suburbs the key covers")
        if r.status_code == 401:
            raise MBError("API key rejected - check .env")
        r.raise_for_status()
        return r.json()
    raise MBError("Rate limited, try again in a moment")

def _payload(j):
    """'No data' is still a 200: {"data": null, "available": false, ...}"""
    if isinstance(j, dict) and "data" in j:
        return j["data"]
    return j

def rent_history(gnaf_id):
    d = _payload(_get(f"/v1/properties/{gnaf_id}/rent-history"))
    out = []
    for row in d or []:
        try:
            out.append(dict(date=row["date"], price=float(row["price"]), type=row.get("type", "")))
        except (KeyError, TypeError, ValueError):
            continue
    return out

def listings_for_rent(suburb):
    j = _get(f"/v1/suburbs/{urllib.parse.quote(suburb)}/listings/for-rent")
    d = _payload(j)
    rows = d.get("results", []) if isinstance(d, dict) else (d or [])
    out = []
    for x in rows:
        a = x.get("attributes") or {}
        try:
            price = float(str(x.get("price")).replace("$", "").replace(",", "").split()[0])
        except (TypeError, ValueError, IndexError):
            continue
        if x.get("off_market"):
            continue
        out.append(dict(gnaf_id=x.get("gnaf_pid"), address=x.get("area_name", ""), price=price,
                        bedrooms=a.get("bedrooms"), bathrooms=a.get("bathrooms"),
                        parking=a.get("garage_spaces"), listing_date=x.get("listing_date")))
    return out
