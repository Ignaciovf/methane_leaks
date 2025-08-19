import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os


# Database connection
def get_connection():
    return psycopg2.connect(
        host="localhost",
        port="5433",
        database="methane_leaks_db",
        user="methane_user",
        password="methane_password"
    )


def create_table_if_not_exists(cur):
    """Create the methane_leaks table if it doesn't exist - completely unrestricted"""
    create_table_query = """
                         CREATE \
                         EXTENSION IF NOT EXISTS postgis;

                         CREATE TABLE IF NOT EXISTS methane_leaks \
                         ( \
                             id \
                             SERIAL \
                             PRIMARY \
                             KEY, \
                             id_plume \
                             TEXT, \
                             source_name \
                             TEXT, \
                             satellite \
                             TEXT, \
                             tile_date \
                             TIMESTAMP, \
                             lat \
                             NUMERIC, \
                             lon \
                             NUMERIC, \
                             actionable \
                             TEXT, \
                             notified \
                             BOOLEAN, \
                             country \
                             TEXT, \
                             sector \
                             TEXT, \
                             detection_institution \
                             TEXT, \
                             quantification_institution \
                             TEXT, \
                             tile \
                             TEXT, \
                             ch4_fluxrate \
                             NUMERIC, \
                             ch4_fluxrate_std \
                             NUMERIC, \
                             wind_u \
                             NUMERIC, \
                             wind_v \
                             NUMERIC, \
                             total_emission \
                             NUMERIC, \
                             total_emission_std \
                             NUMERIC, \
                             wind_speed \
                             NUMERIC, \
                             last_update \
                             TIMESTAMP, \
                             insert_date \
                             TIMESTAMP, \
                             feedback_operator \
                             TEXT, \
                             feedback_government \
                             TEXT, \
                             contacted_by_methane_leaks \
                             BOOLEAN \
                             DEFAULT \
                             FALSE
                         ); \
                         """
    cur.execute(create_table_query)
    print("✅ Table created or already exists")


def load_csv_data():
    # Read CSV
    df = pd.read_csv("data/unep_methanedata_detected_plumes.csv")
    print(f"Loaded {len(df)} rows from CSV")

    # Connect to database
    conn = get_connection()
    cur = conn.cursor()

    # Drop and recreate table to ensure no restrictions
    cur.execute("DROP TABLE IF EXISTS methane_leaks CASCADE;")
    print("🗑️ Dropped existing table")

    create_table_if_not_exists(cur)
    conn.commit()

    # Prepare ALL data for insertion with minimal processing
    data_to_insert = []
    for _, row in df.iterrows():
        # Convert everything to string first, then to appropriate type or None
        def safe_convert(val, convert_func):
            if pd.isna(val) or val == '' or str(val).lower() in ['nan', 'null', 'none']:
                return None
            try:
                return convert_func(val)
            except:
                return None

        def safe_datetime(val):
            if pd.isna(val):
                return None
            try:
                return pd.to_datetime(val)
            except:
                return None

        data_to_insert.append((
            str(row['id_plume']) if not pd.isna(row['id_plume']) else None,
            str(row['source_name']) if not pd.isna(row['source_name']) else None,
            str(row['satellite']) if not pd.isna(row['satellite']) else None,
            safe_datetime(row['tile_date']),
            safe_convert(row['lat'], float),
            safe_convert(row['lon'], float),
            str(row['actionable']) if not pd.isna(row['actionable']) else None,
            safe_convert(row['notified'], lambda x: str(x).lower() == 'true'),
            str(row['country']) if not pd.isna(row['country']) else None,
            str(row['sector']) if not pd.isna(row['sector']) else None,
            str(row['detection_institution']) if not pd.isna(row['detection_institution']) else None,
            str(row['quantification_institution']) if not pd.isna(row['quantification_institution']) else None,
            str(row['tile']) if not pd.isna(row['tile']) else None,
            safe_convert(row['ch4_fluxrate'], float),
            safe_convert(row['ch4_fluxrate_std'], float),
            safe_convert(row['wind_u'], float),
            safe_convert(row['wind_v'], float),
            safe_convert(row['total_emission'], float),
            safe_convert(row['total_emission_std'], float),
            safe_convert(row['wind_speed'], float),
            safe_datetime(row['last_update']),
            safe_datetime(row['insert_date']),
            str(row['feedback_operator']) if not pd.isna(row['feedback_operator']) else None,
            str(row['feedback_government']) if not pd.isna(row['feedback_government']) else None,
            False
        ))

    # Insert ALL data with no restrictions
    insert_query = """
                   INSERT INTO methane_leaks (id_plume, source_name, satellite, tile_date, lat, lon, actionable,
                                              notified,
                                              country, sector, detection_institution, quantification_institution, tile,
                                              ch4_fluxrate, ch4_fluxrate_std, wind_u, wind_v, total_emission,
                                              total_emission_std,
                                              wind_speed, last_update, insert_date, feedback_operator,
                                              feedback_government,
                                              contacted_by_methane_leaks)
                   VALUES %s
                   """

    execute_values(cur, insert_query, data_to_insert, page_size=1000)
    conn.commit()

    print(f"✅ Inserted {cur.rowcount} records into database")

    # Verify the data
    cur.execute("SELECT COUNT(*) FROM methane_leaks")
    total_count = cur.fetchone()[0]
    print(f"📊 Total records in database: {total_count}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    load_csv_data()