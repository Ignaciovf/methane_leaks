-- init.sql
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create table matching your CSV structure exactly
CREATE TABLE IF NOT EXISTS methane_leaks (
    id SERIAL PRIMARY KEY,
    id_plume VARCHAR(255) UNIQUE NOT NULL,
    source_name VARCHAR(255),
    satellite VARCHAR(255),
    tile_date TIMESTAMP,
    lat DECIMAL(10, 8),
    lon DECIMAL(11, 8),
    actionable VARCHAR(50),
    notified BOOLEAN,
    country VARCHAR(100),
    sector VARCHAR(100),
    detection_institution VARCHAR(255),
    quantification_institution VARCHAR(255),
    tile VARCHAR(255),
    ch4_fluxrate DECIMAL(15, 2),
    ch4_fluxrate_std DECIMAL(15, 2),
    wind_u DECIMAL(10, 2),
    wind_v DECIMAL(10, 2),
    total_emission DECIMAL(15, 2),
    total_emission_std DECIMAL(15, 2),
    wind_speed DECIMAL(10, 2),
    last_update TIMESTAMP,
    insert_date TIMESTAMP,
    feedback_operator VARCHAR(255),
    feedback_government VARCHAR(255),

    -- Our additional tracking columns
    contacted_by_methane_leaks BOOLEAN DEFAULT FALSE,
    contact_date TIMESTAMP,
    contact_notes TEXT,

    -- Spatial column
    geom GEOMETRY(POINT, 4326),

    -- Tracking columns
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_methane_leaks_geom ON methane_leaks USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_methane_leaks_country ON methane_leaks (country);
CREATE INDEX IF NOT EXISTS idx_methane_leaks_contacted ON methane_leaks (contacted_by_methane_leaks);
CREATE INDEX IF NOT EXISTS idx_methane_leaks_tile_date ON methane_leaks (tile_date);
CREATE INDEX IF NOT EXISTS idx_methane_leaks_id_plume ON methane_leaks (id_plume);

-- Update trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_methane_leaks_updated_at
    BEFORE UPDATE ON methane_leaks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();