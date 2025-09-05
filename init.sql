CREATE TABLE IF NOT EXISTS methane_leaks (
    id SERIAL PRIMARY KEY,
    source_name TEXT,
    country TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    ch4_fluxrate DOUBLE PRECISION,
    contacted_by_methane_leaks BOOLEAN DEFAULT FALSE
);

