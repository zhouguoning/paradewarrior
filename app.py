"""
Fair Rent Check - a renter's tool for any City of Sydney flat.
Run:  streamlit run app.py      (needs the supplied CSVs in data/, runs locally only)
"""
import streamlit as st
from rent_check import RentCheck, score_listings
import mb_api, pandas as pd

st.set_page_config(page_title="Fair Rent Check", page_icon="🏠", layout="centered")

@st.cache_resource(show_spinner="Building the rent model from past adverts (about 10 seconds)...")
def load():
    return RentCheck()

rc = load()
st.title("🏠 Fair Rent Check")
st.caption("Is this asking rent fair? Weekly rent estimates for every flat in the City of Sydney, "
           "built from past adverts to 31 July 2026. Estimates are indicative only.")

q = st.text_input("Address", placeholder="e.g. 101 Bathurst Street, or 1506 101 Bathurst")
hits = rc.search(q) if q else None
if q and (hits is None or hits.empty):
    st.info("No flat found. Try the street number and street name, e.g. '20 Dunkerley Place'.")
if hits is not None and not hits.empty:
    labels = {r.gnaf_id: f"{r.address.title()}  ·  {int(r.bedrooms)} bed" for r in hits.itertuples()}
    gid = st.selectbox("Pick the flat", list(labels), format_func=labels.get)
    c1, c2 = st.columns(2)
    asking = c1.number_input("Asking rent ($/week, optional)", min_value=0, step=10, value=0)
    park = c2.radio("Car space", ["As recorded", "With", "Without"], horizontal=True)
    has_p = None if park == "As recorded" else (park == "With")
    r = rc.check(gid, asking=asking or None, has_parking=has_p)

    st.subheader(r["address"].title())
    st.write(f"{r['bedrooms']} bed · {r['bathrooms']} bath · car space: {r['parking'] if has_p is None else park.lower()}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Fair rent", f"${r['fair']:,}/wk")
    m2.metric("Likely range", f"${r['low']:,} – ${r['high']:,}")
    m3.metric("Parking worth", f"${r['parking_premium']}/wk")
    st.caption(f"Based on: {r['evidence']}. The range is our typical miss for this kind of evidence on adverts we held out.")

    if r.get("asking"):
        gap = r["gap"]
        if gap > 0.05:   st.error(f"Asking ${r['asking']:,} is **{gap:.0%} above** fair rent.")
        elif gap < -0.05: st.success(f"Asking ${r['asking']:,} is **{-gap:.0%} below** fair rent. Good value.")
        else:            st.info(f"Asking ${r['asking']:,} is **in line** with fair rent ({gap:+.0%}).")
        st.markdown("**Negotiation pointers**")
        for t in r["advice"]: st.markdown(f"- {t}")

    st.markdown("**Rent trend (last 12 months)**")
    t1, t2, t3 = st.columns(3)
    t1.metric("City of Sydney", f"{r['city_trend']:+.1%}")
    t2.metric(r["suburb"], f"{r['suburb_trend']:+.1%}" if r["suburb_trend"] is not None else "n/a")
    t3.metric("This building", f"{r['building_trend']:+.1%}" if r["building_trend"] is not None else "too few adverts",
              help=f"From {r['building_trend_n']} pairs of repeat adverts" if r["building_trend_n"] else None)
    if r["building_recent"] is not None:
        st.markdown("**This building, last 12 months** (median asking rent)")
        br = r["building_recent"].copy()
        br["bedrooms"] = br.bedrooms.astype(int); br["median_asking"] = br.median_asking.map(lambda v: f"${v:,.0f}")
        st.dataframe(br, hide_index=True, use_container_width=True)

if hits is not None and not hits.empty:
    st.divider()
    st.subheader("Live from the Microburbs API")
    live = st.toggle("Fetch live data for this flat and suburb", value=False,
                     help="Uses your team key from .env, called from this computer only.")
    if live:
        @st.cache_data(ttl=600, show_spinner="Fetching live listings...")
        def _listings(suburb): return mb_api.listings_for_rent(suburb)
        @st.cache_data(ttl=600, show_spinner=False)
        def _history(gid): return mb_api.rent_history(gid)
        try:
            hist = _history(gid)
            if hist:
                h = pd.DataFrame(hist); h["date"] = pd.to_datetime(h.date)
                h = h.sort_values("date").set_index("date")[["price"]].rename(columns={"price": "advertised rent"})
                h["fair rent today"] = r["fair"]
                st.markdown("**This flat's past adverts vs today's fair rent**")
                st.line_chart(h)
            else:
                st.caption("No past rent adverts for this flat in the API.")
        except Exception as ex:
            st.warning(f"Rent history unavailable: {ex}")
        try:
            sub = rc.I.loc[rc.I.gnaf_id == gid, "suburb"].iloc[0]
            ls = _listings(sub)
            mine = [x for x in ls if x["gnaf_id"] == gid]
            if mine:
                st.info(f"This flat is on the market now at **${mine[0]['price']:,.0f}/week** (listed {mine[0]['listing_date']}).")
            df = score_listings(rc, ls, sub)
            if df.empty:
                st.caption("No live listings returned for this suburb.")
            else:
                beds = st.multiselect("Bedrooms", sorted(df.bedrooms.dropna().astype(int).unique()),
                                      default=[r["bedrooms"]] if r["bedrooms"] in df.bedrooms.values else None)
                v = df[df.bedrooms.isin(beds)] if beds else df
                st.markdown(f"**Live listings in {sub}, best value first** ({len(v)} of {len(df)})")
                show = v.assign(**{"asking": v.price.map(lambda x: f"${x:,.0f}"), "fair": v.fair.map(lambda x: f"${x:,.0f}"),
                                   "vs fair": v.vs_fair.map(lambda x: f"{x:+.0%}")})
                st.dataframe(show[["address", "bedrooms", "asking", "fair", "vs fair", "basis", "listing_date"]],
                             hide_index=True, use_container_width=True)
                st.caption("Fair rent is our 31 July 2026 estimate carried forward by the city rent trend. "
                           "'this flat' = flat-level estimate; 'suburb, same beds' = suburb median for that bedroom count.")
        except mb_api.MBError as ex:
            st.warning(str(ex))
        except Exception as ex:
            st.warning(f"Live listings unavailable: {ex}")

with st.expander("How it works"):
    st.markdown("""
- **Fair rent**: the flat's own past adverts moved forward with a repeat-rent index, blended with same-building,
  same-bedroom adverts, then suburb and bedroom medians. Every advert is converted to a no-parking equivalent using
  the parking premium measured inside the same building.
- **Accuracy**: on 2,592 later adverts we held out, the typical miss was $44/week and 78% were within $100.
- **Trend**: median annual change across flats advertised more than once.
- Data: Microburbs supplied adverts (used on the day, not republished). Indicative only, not financial advice.
""")
