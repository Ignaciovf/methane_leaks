import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import json
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pydeck as pdk

# ---- Project modules ----
from mapbox_facility_finder import MapboxFacilityFinder
import contact_scraper as cs

# =========================
# Config & Utilities
# =========================
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "methane_leaks_db")
DB_USER = os.getenv("DB_USER", "methane_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "methane_password")

CSV_FALLBACK = "/mnt/data/unep_methanedata_detected_plumes.csv"

ACCENT_COLOR = "#87C062"
st.set_page_config(
    page_title="Methane Leaks",
    layout="wide",
    page_icon="static/images/methane_leaks_without_background_with_text.png"
)



def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )


@st.cache_data(show_spinner=False)
def load_emitters(country_filter: str, source_filter: str):
    """Try DB first; fall back to CSV."""
    try:
        with get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            where = []
            params = []
            if country_filter:
                where.append("LOWER(country) LIKE LOWER(%s)")
                params.append(f"%{country_filter}%")
            if source_filter:
                where.append("LOWER(source_name) LIKE LOWER(%s)")
                params.append(f"%{source_filter}%")
            where_sql = ("WHERE " + " AND ".join(where)) if where else ""

            cur.execute("SELECT COUNT(*) AS n FROM methane_leaks")
            total = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM methane_leaks WHERE contacted_by_methane_leaks = true")
            contacted = cur.fetchone()["n"]

            cur.execute(f"""
                SELECT id, source_name, country, lat, lon, ch4_fluxrate, contacted_by_methane_leaks
                FROM methane_leaks
                {where_sql}
                ORDER BY id DESC
                LIMIT 1000
            """, params)
            rows = cur.fetchall()

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["id","source_name","country","lat","lon","ch4_fluxrate","contacted_by_methane_leaks"])
        stats = {"total": total, "contacted": contacted, "remaining": total - contacted}
        ui_df = df.rename(columns={
            "id": "ID",
            "source_name": "Source",
            "country": "Country",
            "lat": "Lat",
            "lon": "Lon",
            "ch4_fluxrate": "CH4 Rate",
            "contacted_by_methane_leaks": "Contacted"
        })
        ui_df["Contacted"] = ui_df["Contacted"].map({True: "Yes", False: "No"})
        return stats, ui_df

    except Exception as e_db:
        if not os.path.exists(CSV_FALLBACK):
            raise RuntimeError(f"No DB and no CSV at {CSV_FALLBACK}. DB error: {e_db}")
        raw = pd.read_csv(CSV_FALLBACK)
        ui_df = pd.DataFrame()
        colmap = {"id": "ID","id_plume":"ID","source_name":"Source","country":"Country","lat":"Lat","lon":"Lon","ch4_fluxrate":"CH4 Rate"}
        for s, d in colmap.items():
            if s in raw.columns:
                ui_df[d] = raw[s]
        if "ID" not in ui_df.columns:
            ui_df["ID"] = range(1, len(ui_df) + 1)
        if "Country" in ui_df.columns and country_filter:
            ui_df = ui_df[ui_df["Country"].str.contains(country_filter, case=False, na=False)]
        if "Source" in ui_df.columns and source_filter:
            ui_df = ui_df[ui_df["Source"].str.contains(source_filter, case=False, na=False)]
        ui_df["Contacted"] = "No"
        stats = {"total": len(ui_df), "contacted": 0, "remaining": len(ui_df)}
        return stats, ui_df


def _serialize_candidates(cands, country_hint: str = ""):
    out = []
    for c in cands:
        out.append({
            "name": getattr(c, "name", None),
            "lat": getattr(c, "lat", None),
            "lon": getattr(c, "lon", None),
            "distance_km": getattr(c, "distance_km", None),
            "category_id": getattr(c, "category_id", None),
            "category_name": getattr(c, "category_name", None),
            "address": getattr(c, "address", None),
            "phone": getattr(c, "phone", None),
            "website": getattr(c, "website", None),
            "source": getattr(c, "source", None),
            "country": country_hint or "",
            "raw": getattr(c, "raw", None),
        })
    return out


def _fmt_candidates_df(rows_dicts):
    if not rows_dicts:
        return pd.DataFrame(columns=["Company", "Distance_km"])
    data = []
    for d in rows_dicts:
        data.append({
            "Company": d.get("name") or "(unnamed)",
            "Distance_km": round(d.get("distance_km"), 2) if d.get("distance_km") is not None else None
        })
    return pd.DataFrame(data).sort_values("Distance_km", na_position="last").reset_index(drop=True)


def run_facility_finder(lat, lon, bearing, radius_km, leak_type, engine_choice, country_hint: str = ""):
    try:
        debug = []
        bearing_val = None if str(bearing).strip() == "" else float(bearing)
        finder = MapboxFacilityFinder(verbose=True)
        cands = finder.find_likely_culprits(
            lat=float(lat), lon=float(lon),
            plume_bearing_deg=bearing_val,
            radius_km=float(radius_km),
            leak_type_hint=(leak_type or None),
            limit=25,
            engine=engine_choice,
            debug=debug,
        )
        cands_dict = _serialize_candidates(cands, country_hint=country_hint)
        df = _fmt_candidates_df(cands_dict)
        logs = "\n".join(debug)
        return df, json.dumps(cands_dict), logs
    except Exception as e:
        logs = f"Exception: {e}"
        err = [{"Company": f"Error: {e}", "Distance_km": None}]
        return pd.DataFrame(err), "[]", logs


def simplify_contacts_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Company", "Website", "EmailOrContact", "Notes"])
    cols = {c.lower(): c for c in df.columns}

    def col_like(*names):
        for n in names:
            for lc, orig in cols.items():
                if n in lc:
                    return orig
        return None

    name_col = col_like("candidate", "company", "name") or next(iter(df.columns))
    email_col = col_like("best email", "email")
    web_col = col_like("website", "url", "domain", "site")
    contact_col = col_like("contact", "contact_url", "contact page", "page")

    out_rows = []
    for _, r in df.iterrows():
        comp = str(r.get(name_col, "")).strip() if name_col else ""
        web = str(r.get(web_col, "")).strip() if web_col else ""
        email = str(r.get(email_col, "")).strip() if email_col else ""
        contact_url = str(r.get(contact_col, "")).strip() if contact_col else ""
        email_or_contact = email or contact_url or web
        notes = "Email found" if email else ("Contact page" if contact_url else ("Website only" if web else ""))
        out_rows.append({"Company": comp or "(unknown)", "Website": web, "EmailOrContact": email_or_contact, "Notes": notes})
    return pd.DataFrame(out_rows)


def pick_candidate(cands_json: str, selected_idx: int) -> dict:
    try:
        arr = json.loads(cands_json or "[]")
        if not isinstance(arr, list) or not arr:
            return {}
        if isinstance(selected_idx, int) and 0 <= selected_idx < len(arr):
            return arr[selected_idx]
        return arr[0]
    except Exception:
        return {}


def generate_outreach_email(cands_json, selected_idx, leak_type, radius_km, bearing):
    cand = pick_candidate(cands_json, selected_idx)
    if not cand:
        return "", "No candidate selected and no results to use."
    company = cand.get("name") or "Facility Operator"
    website = cand.get("website") or ""
    dist = cand.get("distance_km")
    lat = cand.get("lat"); lon = cand.get("lon")

    subj = f"Urgent: Possible methane emission near your facility ({company})"
    bearing_note = f" • Plume bearing considered: {bearing}°" if str(bearing).strip() != "" else ""
    body = f"""Hello {company},

We're contacting you because our satellite-based monitoring detected a methane plume in the vicinity of your facility.

• Approx. coordinates: {lat if lat is not None else "N/A"}, {lon if lon is not None else "N/A"}
• Approx. distance from plume center: {dist:.2f} km
• Leak type hint: {leak_type or "unspecified"}
• Search radius used: {radius_km} km{bearing_note}

Why this matters:
Methane is a potent greenhouse gas. Rapid mitigation reduces climate impact and can improve safety and operational efficiency (lost gas is lost product).

What helps:
1) A quick internal check (flaring, valves, compressors, digesters, landfill cells, WWTP units).
2) A point of contact for your environmental or operations team.
3) If you prefer, we can share a short report with detection time, map, and suggested ground checks.

If {website or "your preferred channel"} or another contact route is better, please let us know the right recipient.
We’re happy to coordinate remediation support and verification.

Thanks,
Methane Leaks Project
contact@methaneleaks.org
"""
    return subj, body


# =========================
# UI
# =========================
st.markdown(
    f"""
    <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        :root {{
            --primary-color: {ACCENT_COLOR};
        }}
        .stButton>button {{
            background-color: {ACCENT_COLOR};
            color: white;
            border: none;
        }}
        .stButton>button:hover {{
            background-color: #76a153;
            color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("static/images/methane_leaks_without_background_with_text.png", use_column_width=True)


# Sidebar
st.sidebar.header("Facility Finder defaults")
bearing_default = st.sidebar.text_input("Plume bearing (deg, optional)", value="")
radius_default = st.sidebar.slider("Search radius (km)", min_value=1, max_value=50, value=10, step=1)
leak_default = st.sidebar.selectbox("Leak type (hint)", ["", "landfill", "oil_gas", "wastewater", "agriculture", "coal_mine"], index=1)
engine_default = st.sidebar.selectbox("Engine", ["auto", "searchbox", "tilequery"], index=0)

st.sidebar.header("Scraper")
debug_scraper = st.sidebar.checkbox("Verbose logs", value=True)
request_timeout = st.sidebar.slider("HTTP timeout (s)", 5, 30, 12)
max_pages = st.sidebar.slider("Max pages per site", 1, 30, 10)
retry_count = st.sidebar.slider("Retries per request", 0, 3, 1)
enable_enrichment = st.sidebar.checkbox("Enrich missing websites (OSM + Search)", value=True)

# State
for key, default in {
    "cands_json": "[]",
    "cands_table": pd.DataFrame(columns=["Company", "Distance_km"]),
    "finder_logs": "",
    "selected_cand_idx": -1,
    "scrape_logs": "",
    "current_country": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Filters
with st.expander("Filters", expanded=True):
    colf1, colf2, colf3 = st.columns([1, 1, 1.2])
    with colf1:
        country_filter = st.text_input("Filter by Country", value="")
    with colf2:
        source_filter = st.text_input("Filter by Source", value="")
    with colf3:
        run_id = st.text_input("Run by ID (fallback)", value="")

# Data
try:
    stats, table_df = load_emitters(country_filter.strip(), source_filter.strip())
    st.caption(f"**Total:** {stats['total']}  |  **Contacted:** {stats['contacted']}  |  **Remaining:** {stats['remaining']}")
except Exception as e:
    st.error(str(e))
    st.stop()

# DASHBOARD (full-width)
st.subheader("Active Methane Plume")
if table_df.empty:
    st.info("No rows.")
else:
    # Rename the column
    if "Contacted" in table_df.columns:
        table_df = table_df.rename(columns={"Contacted": "Contacted by Methane Leaks"})

    total_count = len(table_df)

    # Page size selector (default 30)
    page_size = st.selectbox("Rows per page", [30, 50, 100, 200], index=0)

    # Summary line
    st.caption(f"Showing up to {page_size} per page • {total_count} active plumes total")

    # Configure grid with pagination
    gob = GridOptionsBuilder.from_dataframe(table_df)
    gob.configure_selection(selection_mode="single", use_checkbox=False)
    gob.configure_pagination(paginationAutoPageSize=False, paginationPageSize=page_size)
    # Ensure the pagination toolbar is visible under the rows
    gob.configure_grid_options(domLayout="autoHeight")
    grid_options = gob.build()

    grid_return = AgGrid(
        table_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        # Remove fixed height so autoHeight can reveal the pagination bar
        # height argument intentionally omitted
    )

    selected_rows = grid_return.get("selected_rows", [])
    if selected_rows:
        r = selected_rows[0]
        try:
            lat_val = float(r["Lat"])
            lon_val = float(r["Lon"])
            flux_val = float(r.get("CH4 Rate") or 0)
            country_val = str(r.get("Country") or "").strip()
            st.session_state["current_country"] = country_val
            st.session_state["selected_lat"] = lat_val
            st.session_state["selected_lon"] = lon_val
            st.session_state["selected_flux"] = flux_val

            df, cands_json, logs = run_facility_finder(
                lat=lat_val,
                lon=lon_val,
                bearing=bearing_default,
                radius_km=radius_default,
                leak_type=leak_default,
                engine_choice=engine_default,
                country_hint=country_val,
            )
            st.session_state["cands_table"] = df
            st.session_state["cands_json"] = cands_json
            st.session_state["finder_logs"] = logs
            st.session_state["selected_cand_idx"] = -1
            st.success(
                f"Facility Finder run for Lat {lat_val:.5f}, Lon {lon_val:.5f} (Country: {country_val or 'N/A'})"
            )
        except Exception as e:
            st.warning(f"Could not run finder for selected row: {e}")

    if "selected_lat" in st.session_state and "selected_lon" in st.session_state:
        st.subheader("Plume Map")
        flux = st.session_state.get("selected_flux", 0.0)
        token = os.getenv("MAPBOX_ACCESS_TOKEN")
        if token:
            map_df = pd.DataFrame([
                {
                    "lat": st.session_state["selected_lat"],
                    "lon": st.session_state["selected_lon"],
                    "flux": flux,
                    "radius": max(flux, 0) * 100,
                }
            ])
            # Build a color column based on CH4 magnitude (replace 'flux' with your CH4 field if different)
            ch4_col = "flux"  # e.g., "flux", "ch4", "ch4_ppm", etc.

            if not map_df.empty:
                if ch4_col in map_df:
                    min_v = float(map_df[ch4_col].min())
                    max_v = float(map_df[ch4_col].max())
                    rng = max(max_v - min_v, 1e-9)  # avoid division by zero

                    def lerp(a, b, t):
                        return int(a + (b - a) * t)

                    colors = []
                    for v in map_df[ch4_col]:
                        t = (float(v) - min_v) / rng  # normalize 0..1
                        # Blue (low) -> Orange/Red (high), with constant alpha
                        r = lerp(0, 255, t)
                        g = lerp(160, 64, t)
                        b = lerp(255, 0, t)
                        colors.append([r, g, b, 180])
                    map_df["color"] = colors
                else:
                    # Fallback color if CH4 column missing
                    map_df["color"] = [[0, 160, 255, 180]] * len(map_df)

            # Render circle with constant pixel size; color scales with CH4
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_fill_color='color',
                # Force a fixed on-screen circle size by clamping pixel radius
                get_radius=1,
                radius_scale=1,
                radius_min_pixels=8,   # change this number to adjust the on-screen size
                radius_max_pixels=8,   # same as min to keep it constant
                pickable=True,
            )
            view_state = pdk.ViewState(
                latitude=st.session_state["selected_lat"],
                longitude=st.session_state["selected_lon"],
                zoom=8,
            )
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/satellite-streets-v12",
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={"text": "CH4 Rate: {flux}"},
                    api_keys={"mapbox": token},
                )
            )
        else:
            st.warning("MAPBOX_ACCESS_TOKEN not set. Map cannot be displayed.")

# Fallback by ID
if st.button("Run Facility Finder for ID"):
    try:
        leak_id = int(run_id)
        with get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT lat, lon, country FROM methane_leaks WHERE id=%s LIMIT 1", (leak_id,))
            row = cur.fetchone()
        if not row:
            st.warning(f"ID {leak_id} not found in DB.")
        else:
            lat_val, lon_val = float(row["lat"]), float(row["lon"])
            country_val = str(row.get("country") or "").strip()
            st.session_state["current_country"] = country_val

            df, cands_json, logs = run_facility_finder(
                lat=lat_val, lon=lon_val,
                bearing=bearing_default, radius_km=radius_default,
                leak_type=leak_default, engine_choice=engine_default,
                country_hint=country_val,
            )
            st.session_state["cands_table"] = df
            st.session_state["cands_json"] = cands_json
            st.session_state["finder_logs"] = logs
            st.session_state["selected_cand_idx"] = -1
            st.success(f"Facility Finder run for ID {leak_id} (Lat {lat_val:.5f}, Lon {lon_val:.5f}, Country: {country_val or 'N/A'})")
    except Exception as e:
        st.error(f"Error: {e}")

# FACILITY FINDER under the dashboard
st.markdown("---")
st.subheader("Facility Finder")

with st.expander("Finder logs", expanded=False):
    st.code(st.session_state["finder_logs"] or "(no logs yet)")

cand_df = st.session_state["cands_table"]
if cand_df is None or cand_df.empty:
    st.info("No candidates yet. Select a row above to run the finder.")
else:
    gob2 = GridOptionsBuilder.from_dataframe(cand_df)
    gob2.configure_selection(selection_mode="single", use_checkbox=False)
    gob2.configure_pagination(paginationAutoPageSize=False, paginationPageSize=12)
    gob2.configure_grid_options(domLayout="normal")
    grid_options2 = gob2.build()

    grid_return2 = AgGrid(
        cand_df,
        gridOptions=grid_options2,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        height=320,
        fit_columns_on_grid_load=True
    )

    selected2 = grid_return2.get("selected_rows", [])
    if selected2:
        try:
            sel_company = selected2[0]["Company"]
            idx = cand_df.index[cand_df["Company"] == sel_company].tolist()
            st.session_state["selected_cand_idx"] = idx[0] if idx else 0
        except Exception:
            st.session_state["selected_cand_idx"] = 0

    st.markdown("---")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        if st.button("Scrape contacts (all listed)"):
            st.session_state["scrape_logs"] = ""
            with st.status("Scraping contacts…", expanded=True) as status:
                log_area = st.empty()
                progress = st.progress(0.0)
                try:
                    df_contacts, logs = cs.scrape_contacts_bulk(
                        json.loads(st.session_state["cands_json"]),
                        debug=debug_scraper,
                        timeout=request_timeout,
                        max_pages=max_pages,
                        retries=retry_count,
                        enrich_missing_website=enable_enrichment,
                        progress_cb=lambda p: progress.progress(min(max(p, 0.0), 1.0)),
                        log_cb=lambda line: log_area.code(line, language="text")
                    )
                    st.session_state["scrape_logs"] = "\n".join(logs[-2000:])
                    status.update(label="Scraping finished", state="complete")
                except Exception as e:
                    status.update(label=f"Scraping failed: {e}", state="error")
                    df_contacts = pd.DataFrame([{"Company": "Error", "Website": "", "EmailOrContact": "", "Notes": str(e)}])

            st.dataframe(simplify_contacts_table(df_contacts), use_container_width=True)

    with c2:
        if st.button("Generate email (selected or first)"):
            subj, body = generate_outreach_email(
                st.session_state["cands_json"],
                st.session_state["selected_cand_idx"],
                leak_default, radius_default, bearing_default
            )
            st.text_input("Email subject", value=subj, key="email_subject")
            st.text_area("Email body", value=body, height=280, key="email_body")

    with st.expander("Scraper logs", expanded=False):
        st.code(st.session_state["scrape_logs"] or "(no logs yet)")