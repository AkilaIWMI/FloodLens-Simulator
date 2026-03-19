"""
preprocess_data.py
==================
Transforms the raw OCR-scraped CSV (kelani_flood_guages.csv) into the
flat per-day format that the GA fitness function expects.

Raw CSV structure (one row per station per report):
  date, River Basin, Tributory/River, Gauging Station, Unit,
  Alert Level, Minor Flood Level, Major Flood Level,
  Water Level 1, Water Level 2, Remarks, Water Level Rising or Falling,
  Rainfall, report_time

Target CSV structure (one row per day, Glencourse-centric):
  date, report_time, rainfall_24hr, rainfall_3hr,
  water_level_9am, water_level_noon, rate_of_change,
  upstream_kithulgala_wl, upstream_norwood_wl, upstream_deraniyagala_wl,
  antecedent_rainfall_7day, duration_hours_above_alert,
  remarks, water_trend

Usage:
    python preprocess_data.py

Output:
    ../data/glencourse_data.csv
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


# =============================================================
# CONFIGURATION
# =============================================================

RAW_CSV_PATH = "../data/kelani_flood_guages.csv"
OUTPUT_CSV_PATH = "../data/glencourse_data.csv"

# Station names as they appear in the raw CSV (from OCR)
# Adjust these if your OCR produces slightly different spellings
STATION_NAMES = {
    "glencourse": ["Glencourse", "glencourse", "GLENCOURSE"],
    "kithulgala": ["Kithulgala", "kithulgala", "KITHULGALA"],
    "hanwella": ["Hanwella", "hanwella", "HANWELLA"],
    "nagalagam": ["Nagalagam Street", "Nagalagam street", "NAGALAGAM STREET"],
    "norwood": ["Norwood", "norwood", "NORWOOD"],
    "deraniyagala": ["Deraniyagala", "deraniyagala", "DERANIYAGALA"],
    "holombuwa": ["Holombuwa", "holombuwa", "HOLOMBUWA"],
}

# Glencourse thresholds (for computing duration)
GLENCOURSE_ALERT = 15.0


# =============================================================
# HELPER FUNCTIONS
# =============================================================

def clean_numeric(val):
    """Safely convert a value to float, handling OCR artifacts."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    # Handle common OCR issues
    if val_str in ["-", "NA", "N/A", "na", "", "—", "–", "nil"]:
        return np.nan
    # Remove any non-numeric characters except . and -
    cleaned = "".join(c for c in val_str if c in "0123456789.-")
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def match_station(gauging_station, station_key):
    """Check if a gauging station name matches a station key."""
    if pd.isna(gauging_station):
        return False
    gs = str(gauging_station).strip()
    return gs in STATION_NAMES.get(station_key, [])


def determine_report_period(report_time_str):
    """
    Classify report time into morning or noon.
    Morning reports: typically 06:00-10:00
    Noon reports: typically 12:00-15:00
    """
    if pd.isna(report_time_str):
        return "unknown"
    t = str(report_time_str).strip().replace(".", ":")
    try:
        parts = t.split(":")
        hour = int(parts[0])
        if hour < 11:
            return "morning"
        else:
            return "noon"
    except (ValueError, IndexError):
        return "unknown"


# =============================================================
# MAIN PREPROCESSING
# =============================================================

def preprocess():
    print("=" * 60)
    print("  PREPROCESSING RAW OCR DATA")
    print("=" * 60)

    # --- 1. Load raw CSV ---
    if not os.path.exists(RAW_CSV_PATH):
        print(f"\n  ERROR: Raw CSV not found at {RAW_CSV_PATH}")
        print(f"  Please place your kelani_flood_guages.csv in ../data/")
        return None

    raw = pd.read_csv(RAW_CSV_PATH)
    print(f"\n  Loaded {len(raw)} rows from {RAW_CSV_PATH}")
    print(f"  Columns: {list(raw.columns)}")
    print(f"  Date range: {raw['date'].min()} to {raw['date'].max()}")

    # --- 2. Clean column names (strip whitespace) ---
    raw.columns = raw.columns.str.strip()

    # --- 3. Parse dates ---
    raw["date"] = pd.to_datetime(raw["date"], format="mixed", dayfirst=True)

    # --- 4. Clean numeric columns ---
    for col in ["Alert Level", "Minor Flood Level", "Major Flood Level",
                 "Water Level 1", "Water Level 2", "Rainfall"]:
        if col in raw.columns:
            raw[col] = raw[col].apply(clean_numeric)

    # --- 5. Add report period classification ---
    raw["report_period"] = raw["report_time"].apply(determine_report_period)

    # --- 6. Extract Glencourse rows ---
    glencourse_mask = raw["Gauging Station"].apply(
        lambda x: match_station(x, "glencourse")
    )
    glencourse = raw[glencourse_mask].copy()
    print(f"\n  Glencourse rows found: {len(glencourse)}")

    if len(glencourse) == 0:
        print("  ERROR: No Glencourse data found!")
        print(f"  Available stations: {raw['Gauging Station'].unique()}")
        return None

    # --- 7. Extract upstream station data ---
    upstream_stations = {}
    for key in ["kithulgala", "norwood", "deraniyagala", "holombuwa", "hanwella"]:
        mask = raw["Gauging Station"].apply(lambda x: match_station(x, key))
        station_df = raw[mask][["date", "report_time", "Water Level 1",
                                 "Water Level 2", "Rainfall"]].copy()
        station_df = station_df.rename(columns={
            "Water Level 1": f"{key}_wl1",
            "Water Level 2": f"{key}_wl2",
            "Rainfall": f"{key}_rainfall",
        })
        upstream_stations[key] = station_df
        print(f"  {key.capitalize()} rows found: {len(station_df)}")

    # --- 8. Build daily Glencourse records ---
    # Group by date, taking the best available readings
    daily_records = []

    for date, group in glencourse.groupby("date"):
        record = {"date": date}

        # Get the latest report time for this day
        report_times = group["report_time"].tolist()
        record["report_time"] = report_times[-1] if report_times else "unknown"

        # Water levels: use Water Level 1 as the earlier reading,
        # Water Level 2 as the later reading within the same report
        # If multiple reports exist for the same day (morning + noon),
        # use morning WL2 as 9am and noon WL2 as noon
        morning = group[group["report_period"] == "morning"]
        noon = group[group["report_period"] == "noon"]

        if len(noon) > 0 and len(morning) > 0:
            # Best case: we have both morning and noon reports
            record["water_level_9am"] = clean_numeric(
                morning.iloc[-1]["Water Level 2"]
            )
            record["water_level_noon"] = clean_numeric(
                noon.iloc[-1]["Water Level 2"]
            )
        elif len(noon) > 0:
            # Only noon report: use WL1 as earlier, WL2 as later
            record["water_level_9am"] = clean_numeric(
                noon.iloc[-1]["Water Level 1"]
            )
            record["water_level_noon"] = clean_numeric(
                noon.iloc[-1]["Water Level 2"]
            )
        elif len(morning) > 0:
            # Only morning report: use WL1 and WL2
            record["water_level_9am"] = clean_numeric(
                morning.iloc[-1]["Water Level 1"]
            )
            record["water_level_noon"] = clean_numeric(
                morning.iloc[-1]["Water Level 2"]
            )
        else:
            # Unknown period: use last row
            record["water_level_9am"] = clean_numeric(
                group.iloc[-1]["Water Level 1"]
            )
            record["water_level_noon"] = clean_numeric(
                group.iloc[-1]["Water Level 2"]
            )

        # Rainfall: take the maximum reported for the day
        rainfall_vals = group["Rainfall"].apply(clean_numeric).dropna()
        record["rainfall_24hr"] = rainfall_vals.max() if len(rainfall_vals) > 0 else 0.0

        # Estimate 3hr rainfall as a fraction of 24hr
        # (we don't have actual 3hr data from the reports)
        if record["rainfall_24hr"] > 0:
            record["rainfall_3hr"] = record["rainfall_24hr"] * 0.5  # rough estimate
        else:
            record["rainfall_3hr"] = 0.0

        # Rate of change (m/hr)
        # Approximate: difference between two readings / time gap
        wl1 = record.get("water_level_9am", np.nan)
        wl2 = record.get("water_level_noon", np.nan)
        if not np.isnan(wl1) and not np.isnan(wl2):
            # Assume ~3 hour gap between readings
            record["rate_of_change"] = round((wl2 - wl1) / 3.0, 4)
        else:
            record["rate_of_change"] = 0.0

        # Remarks and trend from last report
        record["remarks"] = group.iloc[-1].get("Remarks", "Normal")
        trend = group.iloc[-1].get("Water Level Rising or Falling", "")
        record["water_trend"] = str(trend).strip() if pd.notna(trend) else ""

        # --- Upstream stations for this date ---
        for key in ["kithulgala", "norwood", "deraniyagala"]:
            station_data = upstream_stations[key]
            day_data = station_data[station_data["date"] == date]
            if len(day_data) > 0:
                # Take the latest Water Level 2 reading
                record[f"upstream_{key}_wl"] = clean_numeric(
                    day_data.iloc[-1][f"{key}_wl2"]
                )
            else:
                record[f"upstream_{key}_wl"] = np.nan

        daily_records.append(record)

    df = pd.DataFrame(daily_records)
    df = df.sort_values("date").reset_index(drop=True)

    # --- 9. Compute rolling/derived columns ---

    # Antecedent 7-day rainfall
    df["antecedent_rainfall_7day"] = (
        df["rainfall_24hr"]
        .rolling(window=7, min_periods=1)
        .sum()
        .shift(1)  # exclude current day
        .fillna(0)
    )

    # Duration above alert (estimated from consecutive days)
    # Simple approach: count consecutive days where water level >= alert
    df["above_alert"] = df["water_level_noon"] >= GLENCOURSE_ALERT
    duration_hours = []
    consecutive = 0
    for _, row in df.iterrows():
        if row["above_alert"]:
            consecutive += 1
            # Assume ~12 hours per day above alert (rough estimate)
            duration_hours.append(consecutive * 12)
        else:
            consecutive = 0
            duration_hours.append(0)
    df["duration_hours_above_alert"] = duration_hours
    df = df.drop(columns=["above_alert"])

    # --- 10. Save output ---
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    # --- 11. Print summary ---
    print(f"\n  {'='*50}")
    print(f"  OUTPUT SUMMARY")
    print(f"  {'='*50}")
    print(f"  Saved to: {OUTPUT_CSV_PATH}")
    print(f"  Total days: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\n  Water Level (noon):")
    wl_valid = df["water_level_noon"].dropna()
    print(f"    Valid readings: {len(wl_valid)}/{len(df)}")
    print(f"    Mean:  {wl_valid.mean():.2f} m")
    print(f"    Std:   {wl_valid.std():.2f} m")
    print(f"    Min:   {wl_valid.min():.2f} m")
    print(f"    Max:   {wl_valid.max():.2f} m")
    print(f"\n  Rainfall (24hr):")
    rf_valid = df["rainfall_24hr"].dropna()
    print(f"    Mean:  {rf_valid.mean():.1f} mm")
    print(f"    Max:   {rf_valid.max():.1f} mm")
    print(f"    Rainy days: {(rf_valid > 0).sum()}/{len(rf_valid)}")
    print(f"\n  Flood conditions:")
    if "remarks" in df.columns:
        print(f"    Major Flood: {(df['remarks'].str.contains('Major', na=False)).sum()}")
        print(f"    Minor Flood: {(df['remarks'].str.contains('Minor', na=False)).sum()}")
        print(f"    Alert:       {(df['remarks'].str.contains('Alert', na=False)).sum()}")
    print(f"\n  Upstream coverage:")
    for key in ["kithulgala", "norwood", "deraniyagala"]:
        col = f"upstream_{key}_wl"
        valid = df[col].dropna()
        print(f"    {key.capitalize()}: {len(valid)}/{len(df)} days with data")

    # --- 12. Show sample rows ---
    print(f"\n  {'='*50}")
    print(f"  SAMPLE ROWS (first 3):")
    print(f"  {'='*50}")
    for i, row in df.head(3).iterrows():
        print(f"\n  Row {i}:")
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                print(f"    {col:<30} {val:.2f}")
            else:
                print(f"    {col:<30} {val}")

    # Show the max water level day (likely Ditwah)
    if len(wl_valid) > 0:
        max_idx = df["water_level_noon"].idxmax()
        max_row = df.loc[max_idx]
        print(f"\n  {'='*50}")
        print(f"  PEAK FLOOD DAY:")
        print(f"  {'='*50}")
        for col in df.columns:
            val = max_row[col]
            if isinstance(val, float):
                print(f"    {col:<30} {val:.2f}")
            else:
                print(f"    {col:<30} {val}")

    return df


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    df = preprocess()
    if df is not None:
        print(f"\n  Preprocessing complete!")
        print(f"  Next step: python ga_flood.py")