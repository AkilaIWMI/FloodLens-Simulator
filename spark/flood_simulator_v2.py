"""
flood_simulator_v2.py
=====================
Generates a FULL TIME SERIES flood scenario, not just a single snapshot.

Given a target PEAK water level, the GA evolves an entire flood
hydrograph — the hour-by-hour rise, peak, and recession of the
river over 72 hours. This is what NVIDIA actually needs to animate
the flood in Unreal Engine.

The GA genome encodes the SHAPE of the flood, not just one moment:
  - When does the flood start rising?
  - How fast does it rise?
  - How long does the peak last?
  - How fast does it recede?
  - What's the rainfall pattern that drives it?
  - How do upstream stations behave relative to Glencourse?

Output: A JSON array of hourly snapshots, each containing water level,
rainfall, upstream conditions — ready for frame-by-frame NVIDIA rendering.

Usage:
    python flood_simulator_v2.py --wl 21.7
    python flood_simulator_v2.py --wl 30.0
    python flood_simulator_v2.py --wl 8.43
"""

import numpy as np
import json
import os
import sys
import argparse
from datetime import datetime

from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import AlgorithmParams

from params import GLENCOURSE as GC
from params import UPSTREAM
from params import export_config as ec

from fitness import (
    DITWAH,
    check_upstream_coherence,
)


# =============================================================
# FLOOD HYDROGRAPH MODEL
# =============================================================
# A flood hydrograph has a characteristic shape:
#
#  Water Level
#       ^
#       |          ___
#       |         /   \        <- peak
#       |        /     \
#       |       /       \___   <- recession tail
#       |      /
#       | ____/                <- base level before flood
#       +-------------------------> Time (hours)
#       0    T_rise  T_peak  T_end
#
# The GA evolves these shape parameters:
#   - base_level:     water level before flood starts
#   - rise_start:     hour when rising begins
#   - rise_duration:  hours from start of rise to peak
#   - peak_duration:  hours the water stays near peak
#   - recession_rate: how fast it falls after peak (m/hr)
#   - rainfall_peak_hour: when the heaviest rain falls
#   - rainfall_intensity: peak rainfall rate (mm/hr)
#   - rainfall_duration:  hours of significant rainfall
#   - upstream_lag:   hours before Glencourse that upstream peaks
#   - upstream_intensity: how severe upstream flooding is (0-1 scale)


TOTAL_HOURS = 72  # simulate 3 days


def decode_hydrograph_genome(X):
    """Decode GA genome into flood shape parameters."""
    return {
        "base_level": X[0],
        "rise_start_hour": X[1],
        "rise_duration_hours": X[2],
        "peak_duration_hours": X[3],
        "recession_rate": X[4],
        "rainfall_peak_hour": X[5],
        "rainfall_intensity": X[6],
        "rainfall_duration": X[7],
        "upstream_lag_hours": X[8],
        "upstream_intensity": X[9],
    }


def generate_hydrograph(params, target_peak):
    """
    Generate hour-by-hour water level from shape parameters.

    Returns a list of 72 dicts, one per hour, each containing:
      - hour
      - water_level_m
      - rainfall_mm (for that hour)
      - upstream_kithulgala_m
      - upstream_norwood_m
      - upstream_deraniyagala_m
      - flood_status
    """
    p = params
    hours = []

    # --- Water level curve ---
    rise_start = p["rise_start_hour"]
    rise_end = rise_start + p["rise_duration_hours"]
    peak_end = rise_end + p["peak_duration_hours"]

    # Amplitude: difference between base and peak
    amplitude = target_peak - p["base_level"]

    water_levels = []
    for h in range(TOTAL_HOURS):
        if h < rise_start:
            # Before flood: at base level with small noise
            wl = p["base_level"] + np.random.uniform(-0.05, 0.05)

        elif h < rise_end:
            # Rising limb: smooth curve (sine-based for natural shape)
            progress = (h - rise_start) / max(1, p["rise_duration_hours"])
            # Sine curve gives natural S-shape rise
            rise_factor = 0.5 * (1 - np.cos(np.pi * progress))
            wl = p["base_level"] + amplitude * rise_factor

        elif h < peak_end:
            # At peak: slight variation around target
            peak_noise = np.random.uniform(-0.1, 0.1)
            wl = target_peak + peak_noise

        else:
            # Recession: exponential decay
            hours_past_peak = h - peak_end
            decay = amplitude * np.exp(-p["recession_rate"] * hours_past_peak)
            wl = p["base_level"] + decay

        water_levels.append(max(0.5, wl))

    # --- Rainfall curve ---
    # Rain typically peaks BEFORE the water level peak (rain causes the rise)
    rain_peak = p["rainfall_peak_hour"]
    rain_dur = p["rainfall_duration"]
    rain_intensity = p["rainfall_intensity"]

    rainfalls = []
    for h in range(TOTAL_HOURS):
        dist_from_peak = abs(h - rain_peak)
        if dist_from_peak < rain_dur / 2:
            # Gaussian-ish rainfall pattern
            rain_factor = np.exp(-0.5 * (dist_from_peak / max(1, rain_dur / 4)) ** 2)
            rf = rain_intensity * rain_factor
            rf += np.random.uniform(0, rain_intensity * 0.1)  # noise
        else:
            rf = np.random.uniform(0, 2)  # trace rainfall
        rainfalls.append(max(0, rf))

    # --- Upstream stations ---
    # Upstream peaks BEFORE Glencourse by upstream_lag hours
    upstream_lag = p["upstream_lag_hours"]
    upstream_scale = p["upstream_intensity"]

    kithulgala_levels = []
    norwood_levels = []
    deraniyagala_levels = []

    for h in range(TOTAL_HOURS):
        # Shift the Glencourse hydrograph backward in time for upstream
        upstream_hour = min(TOTAL_HOURS - 1, max(0, int(h + upstream_lag)))
        upstream_wl_ratio = (water_levels[upstream_hour] - p["base_level"]) / max(1, amplitude)

        # Each station has its own scale
        kith_base = 1.5
        kith_amplitude = (UPSTREAM["Kithulgala"]["major"] - kith_base) * upstream_scale
        kithulgala_levels.append(round(kith_base + kith_amplitude * upstream_wl_ratio, 2))

        norw_base = 0.4
        norw_amplitude = (UPSTREAM["Norwood"]["major"] - norw_base) * upstream_scale
        norwood_levels.append(round(norw_base + norw_amplitude * upstream_wl_ratio, 2))

        dera_base = 0.5
        dera_amplitude = (UPSTREAM["Deraniyagala"]["major"] - dera_base) * upstream_scale
        deraniyagala_levels.append(round(dera_base + dera_amplitude * upstream_wl_ratio, 2))

    # --- Build hourly snapshots ---
    for h in range(TOTAL_HOURS):
        wl = round(water_levels[h], 2)

        if wl >= GC["major_flood_level"]:
            status = "MAJOR_FLOOD"
        elif wl >= GC["minor_flood_level"]:
            status = "MINOR_FLOOD"
        elif wl >= GC["alert_level"]:
            status = "ALERT"
        else:
            status = "NORMAL"

        hours.append({
            "hour": h,
            "water_level_m": wl,
            "rainfall_mm": round(rainfalls[h], 1),
            "upstream_kithulgala_m": kithulgala_levels[h],
            "upstream_norwood_m": norwood_levels[h],
            "upstream_deraniyagala_m": deraniyagala_levels[h],
            "flood_status": status,
            "color_code": "RED" if wl >= GC["major_flood_level"]
                else "ORANGE" if wl >= GC["minor_flood_level"]
                else "YELLOW" if wl >= GC["alert_level"]
                else "GREEN",
        })

    return hours


# =============================================================
# FITNESS FUNCTION FOR HYDROGRAPH SHAPE
# =============================================================

def hydrograph_fitness(X, target_peak):
    """
    Evaluate how physically coherent and realistic the
    flood hydrograph shape is.
    """
    p = decode_hydrograph_genome(X)
    score = 0.0

    # Generate the hydrograph
    try:
        hydro = generate_hydrograph(p, target_peak)
    except Exception:
        return 1000  # invalid, worst score

    # --- 1. Peak must actually reach target ---
    actual_peak = max(h["water_level_m"] for h in hydro)
    peak_diff = abs(actual_peak - target_peak)
    if peak_diff < 0.5:
        score += 20.0
    elif peak_diff < 1.0:
        score += 10.0
    else:
        score -= peak_diff * 5.0

    # --- 2. Base level should be reasonable ---
    if 5.0 <= p["base_level"] <= 12.0:
        score += 10.0
    elif p["base_level"] < 3.0 or p["base_level"] > 15.0:
        score -= 10.0

    # --- 3. Rise duration should match amplitude ---
    amplitude = target_peak - p["base_level"]
    if amplitude > 0:
        implied_rate = amplitude / max(1, p["rise_duration_hours"])
        # Max realistic rise: ~3 m/hr
        if implied_rate <= 3.0:
            score += 10.0
        elif implied_rate <= 5.0:
            score += 3.0
        else:
            score -= 10.0

    # --- 4. Rainfall should precede or coincide with rising water ---
    rain_peak = p["rainfall_peak_hour"]
    rise_start = p["rise_start_hour"]
    # Rain should peak during or slightly before the rising limb
    if rise_start - 6 <= rain_peak <= rise_start + p["rise_duration_hours"] * 0.7:
        score += 15.0
    elif abs(rain_peak - rise_start) < 12:
        score += 5.0
    else:
        score -= 10.0

    # --- 5. Total rainfall should justify the water level ---
    total_rain = sum(h["rainfall_mm"] for h in hydro)
    if target_peak >= GC["major_flood_level"]:
        if total_rain >= 80:
            score += 10.0
        elif total_rain < 30:
            # Check if upstream can explain it
            if p["upstream_intensity"] > 0.6:
                score += 5.0
            else:
                score -= 15.0
    elif target_peak < GC["alert_level"]:
        if total_rain < 50:
            score += 10.0

    # --- 6. Upstream lag should be positive (upstream peaks first) ---
    if p["upstream_lag_hours"] >= 2:
        score += 10.0
    elif p["upstream_lag_hours"] >= 0:
        score += 3.0
    else:
        score -= 10.0

    # --- 7. Upstream intensity scales with flood severity ---
    if target_peak >= GC["major_flood_level"]:
        if p["upstream_intensity"] >= 0.5:
            score += 10.0
        else:
            score -= 5.0
    elif target_peak < GC["alert_level"]:
        if p["upstream_intensity"] < 0.3:
            score += 8.0

    # --- 8. Peak duration should be reasonable ---
    if 1 <= p["peak_duration_hours"] <= 12:
        score += 8.0
    elif p["peak_duration_hours"] > 24:
        score -= 5.0

    # --- 9. Recession rate should be realistic ---
    # Typical: 0.05-0.3 m/hr
    if 0.03 <= p["recession_rate"] <= 0.4:
        score += 8.0
    else:
        score -= 5.0

    # --- 10. Hydrograph should have clear structure ---
    # Count hours in each phase
    hours_rising = sum(1 for h in hydro
                       if h["water_level_m"] > p["base_level"] + amplitude * 0.1
                       and h["water_level_m"] < target_peak * 0.95)
    hours_at_peak = sum(1 for h in hydro
                        if h["water_level_m"] >= target_peak * 0.95)
    hours_normal = sum(1 for h in hydro
                       if h["water_level_m"] < GC["alert_level"])

    if hours_rising > 3 and hours_at_peak >= 1 and hours_normal > 10:
        score += 10.0  # clear rise-peak-fall structure

    return -score


# =============================================================
# MAIN SIMULATOR
# =============================================================

class FloodSimulatorV2:

    def __init__(self):
        print("=" * 60)
        print("  GLENCOURSE FLOOD SIMULATOR v2")
        print("  Time Series Mode — 72-hour Hydrograph")
        print("=" * 60)

    def simulate(self, target_peak_wl):
        """
        Generate a full 72-hour flood time series for the
        given peak water level.
        """
        target = float(target_peak_wl)
        print(f"\n  Target peak: {target:.2f}m")

        # Dynamic bounds based on target
        if target < GC["alert_level"]:
            base_range = [max(0.5, target - 3), target - 0.5]
        else:
            base_range = [5.0, min(12.0, target - 4)]

        bounds = np.array([
            base_range,              # [0] base_level (m)
            [6.0, 36.0],            # [1] rise_start_hour
            [3.0, 24.0],            # [2] rise_duration_hours
            [1.0, 12.0],            # [3] peak_duration_hours
            [0.03, 0.4],            # [4] recession_rate (1/hr)
            [4.0, 30.0],            # [5] rainfall_peak_hour
            [2.0, 50.0],            # [6] rainfall_intensity (mm/hr)
            [4.0, 24.0],            # [7] rainfall_duration (hours)
            [2.0, 12.0],            # [8] upstream_lag_hours
            [0.1, 1.0],             # [9] upstream_intensity (0-1)
        ], dtype=object)

        # Fitness wrapper with fixed target
        def fitness(X):
            return hydrograph_fitness(X, target)

        # Run GA
        print(f"  Evolving hydrograph shape...")
        model = ga(
            function=fitness,
            dimension=10,
            variable_type="real",
            variable_boundaries=bounds,
            algorithm_parameters=AlgorithmParams(
                max_num_iteration=120,
                population_size=200,
                mutation_probability=0.15,
                elit_ratio=0.05,
                parents_portion=0.3,
                crossover_type="uniform",
                max_iteration_without_improv=30,
            )
        )

        result = model.run(no_plot=True, seed=None)
        best_params = decode_hydrograph_genome(result.variable)

        # Generate the time series
        time_series = generate_hydrograph(best_params, target)

        # Compute summary stats
        peak_wl = max(h["water_level_m"] for h in time_series)
        peak_hour = next(h["hour"] for h in time_series
                         if h["water_level_m"] == peak_wl)
        total_rain = sum(h["rainfall_mm"] for h in time_series)
        hours_above_alert = sum(1 for h in time_series
                                if h["water_level_m"] >= GC["alert_level"])
        hours_above_major = sum(1 for h in time_series
                                if h["water_level_m"] >= GC["major_flood_level"])

        # Build output
        output = {
            "generated_at": datetime.now().isoformat(),
            "station": "Glencourse",
            "target_peak_m": target,
            "actual_peak_m": peak_wl,
            "peak_hour": peak_hour,
            "total_rainfall_mm": round(total_rain, 1),
            "hours_above_alert": hours_above_alert,
            "hours_above_major_flood": hours_above_major,
            "simulation_hours": TOTAL_HOURS,

            "hydrograph_params": {
                "base_level_m": round(best_params["base_level"], 2),
                "rise_start_hour": round(best_params["rise_start_hour"], 1),
                "rise_duration_hours": round(best_params["rise_duration_hours"], 1),
                "peak_duration_hours": round(best_params["peak_duration_hours"], 1),
                "recession_rate": round(best_params["recession_rate"], 4),
                "upstream_lag_hours": round(best_params["upstream_lag_hours"], 1),
            },

            "thresholds": {
                "alert_level_m": GC["alert_level"],
                "minor_flood_level_m": GC["minor_flood_level"],
                "major_flood_level_m": GC["major_flood_level"],
            },

            # The actual time series — this is what NVIDIA renders
            "time_series": time_series,
        }

        self._print_summary(output)
        return output

    def _print_summary(self, output):
        ts = output["time_series"]
        print(f"\n  {'─'*55}")
        print(f"  72-HOUR FLOOD HYDROGRAPH")
        print(f"  {'─'*55}")
        print(f"  Peak:              {output['actual_peak_m']}m at hour {output['peak_hour']}")
        print(f"  Base level:        {output['hydrograph_params']['base_level_m']}m")
        print(f"  Total rainfall:    {output['total_rainfall_mm']}mm")
        print(f"  Hours above alert: {output['hours_above_alert']}")
        print(f"  Hours above major: {output['hours_above_major_flood']}")
        print(f"\n  Hour-by-hour (key moments):")
        print(f"  {'Hour':<6} {'WL(m)':<8} {'Rain(mm)':<10} {'Kith(m)':<9} {'Status'}")
        print(f"  {'─'*55}")

        # Show every 6 hours plus peak
        peak_hour = output["peak_hour"]
        for h in ts:
            if h["hour"] % 6 == 0 or h["hour"] == peak_hour:
                marker = " <<<" if h["hour"] == peak_hour else ""
                print(f"  {h['hour']:<6} {h['water_level_m']:<8} "
                      f"{h['rainfall_mm']:<10} {h['upstream_kithulgala_m']:<9} "
                      f"{h['flood_status']}{marker}")

        print(f"  {'─'*55}")

    def export(self, result, filename=None, gcs_output_path=None):
        """Export to GCS if gcs_output_path provided, otherwise local."""
        if filename is None:
            filename = "hydrograph_scenario.json"

        json_bytes = json.dumps(result, indent=2, default=str).encode("utf-8")

        if gcs_output_path:
            from google.cloud import storage

            # If gcs_output_path ends with /, treat as directory and append filename
            if gcs_output_path.endswith("/"):
                full_path = gcs_output_path + filename
            else:
                full_path = gcs_output_path

            path_parts = full_path[len("gs://"):].split("/", 1)
            bucket_name = path_parts[0]
            blob_name = path_parts[1] if len(path_parts) > 1 else filename

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_string(json_bytes, content_type="application/json")
            print(f"\n  Exported to GCS: {full_path}")
            return full_path
        else:
            output_dir = ec["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename)
            with open(path, 'wb') as f:
                f.write(json_bytes)
            print(f"\n  Exported locally: {path}")
            return path


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    sim = FloodSimulatorV2()

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--wl", nargs="*", type=float, default=[21.7])
    parser.add_argument("--water_levels", type=str, default=None,
                        help="Space-separated water levels as string")
    parser.add_argument("--gcs_output_path", type=str, default=None,
                        help="GCS path e.g. gs://bucket/path/")
    parser.add_argument("--gcs_input_path", type=str, default=None)
    parser.add_argument("--partition_value", type=str, default=None)
    known_args, _ = parser.parse_known_args()

    # Determine water levels
    if known_args.water_levels:
        targets = [float(x) for x in known_args.water_levels.split()]
    else:
        targets = known_args.wl

    for target in targets:
        result = sim.simulate(target)
        safe_name = f"hydrograph_{str(target).replace('.', '_')}m.json"
        sim.export(result, filename=safe_name, gcs_output_path=known_args.gcs_output_path)

    print(f"\n  DONE — {len(targets)} hydrograph(s) generated")