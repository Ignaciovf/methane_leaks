import os
import re
import json
import math
import pandas as pd
import gradio as gr
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env (no shell export needed)
load_dotenv()

# ---- Database config (can be overridden in .env) ----
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "methane_leaks_db")
DB_USER = os.getenv("DB_USER", "methane_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "methane_password")

def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )

# =========================
# Dashboard data & actions
# =========================
def get_data_filtered(country_filter, source_filter, contacted_filter):
    """
    Return stats + filtered table (up to 500 rows).
    Filters are case-insensitive substring matches.
    """
    where = []
    params = []
    if country_filter:
        where.append("LOWER(country) LIKE LOWER(%s)")
        params.append(f"%{country_filter}%")
    if source_filter:
        where.append("LOWER(source_name) LIKE LOWER(%s)")
        params.append(f"%{source_filter}%")
    if contacted_filter in ("Yes", "No"):
        where.append("contacted_by_methane_leaks = %s")
        params.append(True if contacted_filter == "Yes" else False)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM methane_leaks")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM methane_leaks WHERE contacted_by_methane_leaks = true")
    contacted = cur.fetchone()[0]

    cur.execute(f"""
        SELECT id, source_name, country, lat, lon, ch4_fluxrate, contacted_by_methane_leaks
        FROM methane_leaks
        {where_sql}
        ORDER BY id DESC
        LIMIT 500
    """, params)
    rows = cur.fetchall()
    cur.close(); conn.close()

    df = pd.DataFrame(rows, columns=['ID', 'Source', 'Country', 'Lat', 'Lon', 'CH4 Rate', 'Contacted'])
    df['Contacted'] = df['Contacted'].map({True: 'Yes', False: 'No'})
    stats = f"Total: {total} | Contacted: {contacted} | Remaining: {total - contacted}"
    return stats, df

def mark_contacted(leak_id):
    if not leak_id:
        return "Enter a leak ID"
    try:
        conn = get_connection(); cur = conn.cursor()
        cur.execute("UPDATE methane_leaks SET contacted_by_methane_leaks = true WHERE id = %s", (int(leak_id),))
        if cur.rowcount > 0:
            conn.commit(); msg = f"✅ Marked leak {leak_id} as contacted"
        else:
            msg = f"❌ Leak ID {leak_id} not found"
        cur.close(); conn.close()
        return msg
    except Exception as e:
        return f"Error: {e}"

# =========================
# Facility Finder
# =========================
from mapbox_facility_finder import MapboxFacilityFinder

def _serialize_candidates(cands):
    """
    Turn FacilityCandidate objects into JSONable dicts for UI state & scraper.
    """
    out = []
    for c in cands:
        out.append({
            "name": c.name,
            "lat": c.lat, "lon": c.lon,
            "distance_km": c.distance_km,
            "category_id": c.category_id, "category_name": c.category_name,
            "address": c.address, "phone": c.phone, "website": c.website,
            "source": c.source,
            "raw": c.raw,
        })
    return out

def _fmt_candidates_dict(rows_dicts):
    if not rows_dicts:
        return pd.DataFrame(columns=["Company", "Distance_km"])
    data = []
    for d in rows_dicts:
        nm = d.get("name") or "(unnamed)"
        dist = d.get("distance_km")
        data.append({"Company": nm, "Distance_km": round(dist, 2) if dist is not None else None})
    df = pd.DataFrame(data).sort_values("Distance_km", na_position="last").reset_index(drop=True)
    return df

def run_facility_finder(lat, lon, bearing, radius_km, leak_type, use_search):
    """
    Run the finder with verbose logs. Normal search (Tilequery) when use_search=False.
    """
    try:
        debug = []
        bearing_val = None if str(bearing).strip() == "" else float(bearing)
        finder = MapboxFacilityFinder(verbose=True)  # verbose on
        cands = finder.find_likely_culprits(
            lat=float(lat), lon=float(lon),
            plume_bearing_deg=bearing_val,
            radius_km=float(radius_km),
            leak_type_hint=(leak_type or None),
            limit=25,
            use_search_box=bool(use_search),  # True => Search Box; False => Tilequery fallback
            debug=debug,                      # collect step-by-step logs
        )
        cands_dict = _serialize_candidates(cands)
        df = _fmt_candidates_dict(cands_dict)
        logs = "\n".join(debug)
        return df, json.dumps(cands_dict), logs
    except Exception as e:
        logs = f"Exception: {e}"
        err = [{"Company": f"Error: {e}", "Distance_km": None}]
        return pd.DataFrame(err), "[]", logs

# Row click in Dashboard → prefill Facility Finder lat/lon
def on_dashboard_select(evt, table_df):
    """
    evt carries the selected cell coordinates. We just need the row to get Lat/Lon.
    """
    try:
        row_idx = evt["index"][0]
        row = table_df.iloc[row_idx]
        lat = float(row["Lat"]); lon = float(row["Lon"])
        return gr.update(value=lat), gr.update(value=lon)
    except Exception:
        return gr.update(), gr.update()

# Run from Dashboard selection and try to switch tab programmatically (best-effort)
def run_from_selection(lat, lon, bearing, radius_km, leak_type, use_search):
    df, cands_json, logs = run_facility_finder(lat, lon, bearing, radius_km, leak_type, use_search)
    try:
        tabs_update = gr.Tabs.update(selected="Facility Finder")  # might be ignored on older Gradio
    except Exception:
        tabs_update = ""
    return df, cands_json, tabs_update, logs

# Diagnostics: raw Tilequery features (to see what's actually there)
def run_tilequery_diagnose(lat, lon, radius_km):
    try:
        finder = MapboxFacilityFinder(verbose=True)
        rows = finder.debug_tilequery(float(lat), float(lon), float(radius_km), limit=100)
        if not rows:
            return pd.DataFrame(columns=["layer", "class", "type", "maki", "name", "lon", "lat"])
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame([{"layer": "error", "class": "", "type": "", "maki": "", "name": str(e), "lon": None, "lat": None}])

# =========================
# Contact Scraper
# =========================
from contact_scraper import scrape_contacts_bulk

def run_contact_scraper(cands_json):
    """
    Expects a JSON array of candidate dicts, e.g.:
    [
      {
        "name": "Consorcio Santa Marta",
        "lat": -33.68, "lon": -70.75,
        "distance_km": 3.42,
        "category_id": "landfill",
        "category_name": "landfill",
        "address": "Camino X 123, Región Metropolitana",
        "phone": "+56 2 2345 6789",
        "website": "https://www.csmarta.cl/",
        "source": "tilequery",
        "raw": {"properties": {"website": "https://www.csmarta.cl/"}}
      }
    ]
    Minimum keys used by the scraper:
      - name (str), website (str or empty)
      - lat, lon (floats) [optional; just shown in output]
      - distance_km (float) [optional; shown in output]
      - source (str) [optional; shown in output]
      - raw (dict) [optional; may contain fallback website]
    """
    try:
        cands = json.loads(cands_json or "[]")
        if not isinstance(cands, list):
            raise ValueError("Input must be a JSON array of candidate objects.")
        results = scrape_contacts_bulk(cands)   # pandas DataFrame
        return results
    except Exception as e:
        return pd.DataFrame([{"Candidate": "Error", "Detail": str(e)}])

# =========================
# UI Layout
# =========================
with gr.Blocks(title="Methane Leaks") as app:
    tabs_container = gr.Tabs()

    with tabs_container:
        # -------- Dashboard --------
        with gr.Tab("Dashboard"):
            gr.Markdown("### 🛰️ Methane Leak Tracker — Dashboard (click a row to prefill Facility Finder)")
            with gr.Row():
                country_filter = gr.Textbox(label="Filter by Country", placeholder="e.g., Chile")
                source_filter = gr.Textbox(label="Filter by Source", placeholder="e.g., Santa Marta")
                contacted_filter = gr.Dropdown(label="Contacted", choices=["", "Yes", "No"], value="")
            stats_text = gr.Textbox(label="Statistics", interactive=False)
            data_table = gr.Dataframe(wrap=True, interactive=False)
            with gr.Row():
                refresh_btn = gr.Button("Refresh / Apply Filters", variant="primary")
                go_btn = gr.Button("Run culprits from selection → Facility Finder", variant="secondary")

            # Hidden inputs used to pass state to the finder
            lat_in = gr.Number(label="Plume latitude", value=0.0, visible=False)
            lon_in = gr.Number(label="Plume longitude", value=0.0, visible=False)
            bearing_in = gr.Textbox(label="Plume bearing (deg, optional)", value="", visible=False)
            radius_in = gr.Slider(label="Search radius (km)", minimum=1, maximum=50, step=1, value=10, visible=False)
            leak_in = gr.Dropdown(choices=["", "landfill", "oil_gas", "wastewater", "agriculture", "coal_mine"],
                                  label="Leak type (hint)", value="landfill", visible=False)
            use_search_in = gr.Checkbox(label="Use Search Box (US/CA/EU only)", value=False, visible=False)

            # Placeholder outputs for the dashboard-triggered run
            find_table_placeholder = gr.Dataframe(visible=False)
            cands_state_from_dashboard = gr.Textbox(visible=False)
            tabs_switch_sink = gr.Textbox(visible=False)  # just to consume the tab-update return

        # -------- Mark Contacted --------
        with gr.Tab("Mark Contacted"):
            gr.Markdown("### Mark a leak as contacted")
            leak_id_input = gr.Textbox(label="Leak ID")
            mark_btn = gr.Button("Mark as Contacted")
            result_text = gr.Textbox(label="Result", interactive=False)

        # -------- Facility Finder --------
        with gr.Tab("Facility Finder"):
            gr.Markdown("### Find likely culprits (keys loaded from `.env`).  \n"
                        "- Unchecked: uses global Tilequery fallback (recommended outside US/CA/EU).  \n"
                        "- Checked: uses Mapbox Search Box (US/CA/EU).")
            with gr.Row():
                lat = gr.Number(label="Plume latitude", value=0.0)
                lon = gr.Number(label="Plume longitude", value=0.0)
                bearing = gr.Textbox(label="Plume bearing (deg, optional)", value="")
                radius = gr.Slider(label="Search radius (km)", minimum=1, maximum=50, step=1, value=10)
                leak = gr.Dropdown(
                    label="Leak type (hint)",
                    choices=["", "landfill", "oil_gas", "wastewater", "agriculture", "coal_mine"],
                    value="landfill",
                )
            use_search = gr.Checkbox(label="Use Search Box (US/CA/EU only)", value=False)
            run = gr.Button("Find candidates", variant="primary")
            out_table = gr.Dataframe(headers=["Company", "Distance_km"], interactive=False)
            cands_json = gr.Textbox(label="Candidates (internal JSON)", visible=False)
            finder_logs = gr.Textbox(label="Finder logs", interactive=False, lines=12)

            # Diagnostics
            diag_btn = gr.Button("Diagnose: raw Tilequery features")
            diag_table = gr.Dataframe(label="Raw features (layer/class/type/maki/name)", wrap=True)

        # -------- Contact Scraper --------
        with gr.Tab("Contact Scraper"):
            gr.Markdown("### Scrape contact info for candidates")
            src_candidates = gr.Textbox(
                label="Candidates JSON",
                placeholder="Click 'Use last Facility Finder result' to auto-fill (expects a JSON array of candidate objects).",
                lines=4
            )
            use_last = gr.Button("Use last Facility Finder result")
            scrape_btn = gr.Button("Scrape contacts", variant="primary")
            contacts_table = gr.Dataframe(wrap=True)

    # =========================
    # Event wiring (after components exist)
    # =========================

    # Dashboard load + filtering
    refresh_btn.click(get_data_filtered, inputs=[country_filter, source_filter, contacted_filter],
                      outputs=[stats_text, data_table])
    app.load(get_data_filtered, inputs=[country_filter, source_filter, contacted_filter],
             outputs=[stats_text, data_table])

    # Row select → prefill hidden lat/lon inputs
    data_table.select(on_dashboard_select, inputs=[data_table], outputs=[lat_in, lon_in])

    # Run finder from Dashboard selection (and attempt tab switch)
    go_btn.click(
        run_from_selection,
        inputs=[lat_in, lon_in, bearing_in, radius_in, leak_in, use_search_in],
        outputs=[find_table_placeholder, cands_state_from_dashboard, tabs_switch_sink, finder_logs],
    )

    # Mark contacted
    mark_btn.click(mark_contacted, inputs=leak_id_input, outputs=result_text)

    # Finder (manual run)
    run.click(run_facility_finder,
              inputs=[lat, lon, bearing, radius, leak, use_search],
              outputs=[out_table, cands_json, finder_logs])

    # Diagnostics button
    diag_btn.click(run_tilequery_diagnose, inputs=[lat, lon, radius], outputs=[diag_table])

    # Contact Scraper
    use_last.click(lambda x: gr.update(value=x), inputs=[cands_json], outputs=[src_candidates])
    scrape_btn.click(run_contact_scraper, inputs=[src_candidates], outputs=[contacts_table])

if __name__ == "__main__":
    app.launch()