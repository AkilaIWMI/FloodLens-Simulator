import numpy as np
import pandas as pd
import os
from pathlib import Path
from params import params as p
from params import GLENCOURSE as GC
from params import UPSTREAM
from params import genome_config as gc
from params import fitness_weights as fw
from params import export_config as ec

DITWAH = {
    "water_level": 21.70,
    "rainfall_24hr": 91.8,
    "rainfall_3hr": 45.0,
    "antecedent_7day": 450.0,
    "duration_above_alert": 36.0,
    "kithulgala_wl": 3.52,
    "upstream_max_rainfall": 159.8,  # Kithulgala's rainfall
}


class LoadData():
    """
    Loads historical Glencourse OCR data for reference distributions.
    Used to validate that GA-generated scenarios are within plausible
    ranges based on observed historical patterns.

    The CSV is expected to have these columns (matching generate_test_data.py):
        date, report_time, rainfall_24hr, rainfall_3hr,
        water_level_9am, water_level_noon, rate_of_change,
        upstream_kithulgala_wl, upstream_norwood_wl, upstream_deraniyagala_wl,
        antecedent_rainfall_7day, duration_hours_above_alert,
        remarks, water_trend
    """

    def __init__(self, inputfile=p["inputfile"]):
        self.data = None
        self.stats = None

        if os.path.exists(inputfile):
            self.data = pd.read_csv(inputfile)
            self.numeric_data = self.data.select_dtypes(include=[np.number])
            self._compute_stats()
            print(f"[INFO] Loaded {len(self.data)} rows from {inputfile}")
        else:
            print(f"[WARNING] Historical data not found at {inputfile}")
            print("          GA will run with threshold-based fitness only.")
            self._set_defaults()

    def _safe_col(self, col):
        """Safely get a column, return None if missing."""
        if self.data is not None and col in self.data.columns:
            return self.data[col]
        return None

    def _compute_stats(self):
        """Compute statistical summaries from historical data."""
        wl_col = self._safe_col("water_level_noon")
        rf_col = self._safe_col("rainfall_24hr")

        self.stats = {
            "mean_wl": float(wl_col.mean()) if wl_col is not None else 8.0,
            "std_wl": float(wl_col.std()) if wl_col is not None else 3.0,
            "max_wl": float(wl_col.max()) if wl_col is not None else DITWAH["water_level"],
            "min_wl": float(wl_col.min()) if wl_col is not None else GC["historical_min_wl"],
            "p95_wl": float(wl_col.quantile(0.95)) if wl_col is not None else 14.0,
            "p99_wl": float(wl_col.quantile(0.99)) if wl_col is not None else 18.0,
            "median_wl": float(wl_col.median()) if wl_col is not None else 7.0,
            "mean_rf": float(rf_col.mean()) if rf_col is not None else 50.0,
            "max_rf": float(rf_col.max()) if rf_col is not None else 350.0,
            "p95_rf": float(rf_col.quantile(0.95)) if rf_col is not None else 120.0,
            "rainy_day_pct": float((rf_col > 0).mean() * 100) if rf_col is not None else 35.0,
            "correlation_rf_wl": self._calc_correlation(),
            "n_rows": len(self.data),
        }

        if "remarks" in self.data.columns:
            self.stats["n_major_floods"] = int((self.data["remarks"] == "Major Flood").sum())
            self.stats["n_minor_floods"] = int((self.data["remarks"] == "Minor Flood").sum())
            self.stats["n_alerts"] = int((self.data["remarks"] == "Alert").sum())

    def _set_defaults(self):
        """Set default stats when no CSV is available."""
        self.stats = {
            "mean_wl": 8.0, "std_wl": 3.0,
            "max_wl": DITWAH["water_level"], "min_wl": GC["historical_min_wl"],
            "p95_wl": 14.0, "p99_wl": 18.0, "median_wl": 7.0,
            "mean_rf": 50.0, "max_rf": 350.0, "p95_rf": 120.0,
            "rainy_day_pct": 35.0, "correlation_rf_wl": 0.6,
            "n_rows": 0,
            "n_major_floods": 0, "n_minor_floods": 0, "n_alerts": 0,
        }

    def _calc_correlation(self):
        if self.data is not None and \
                "rainfall_24hr" in self.data.columns and \
                "water_level_noon" in self.data.columns:
            corr = self.data["rainfall_24hr"].corr(self.data["water_level_noon"])
            return round(float(corr), 3) if not np.isnan(corr) else 0.6
        return 0.6

    def get_data(self):
        return self.data

    def get_stats(self):
        return self.stats

    def print_stats(self):
        """Pretty print the loaded statistics."""
        if self.stats:
            print(f"\n  {'=' * 45}")
            print(f"  HISTORICAL DATA STATISTICS")
            print(f"  {'=' * 45}")
            for k, v in self.stats.items():
                print(f"    {k:<25} {v}")


# Load historical stats once at module level
_historical = LoadData()
_stats = _historical.get_stats()


# =============================================================
# GENOME DECODER
# =============================================================

def decode_genome(X):
    """
    Decode a GA genome vector into named scenario variables.

    Genome layout (10 genes):
        [0] rainfall_24hr            (mm)
        [1] rainfall_3hr             (mm)
        [2] water_level_9am          (m)
        [3] water_level_noon         (m)
        [4] water_level_rate_of_change (m/hr)
        [5] upstream_kithulgala_wl   (m)
        [6] upstream_norwood_wl      (m)
        [7] upstream_deraniyagala_wl (m)
        [8] antecedent_rainfall_7day (mm)
        [9] duration_hours_above_alert (hrs)
    """
    return {
        "rainfall_24hr": X[0],
        "rainfall_3hr": X[1],
        "water_level_9am": X[2],
        "water_level_noon": X[3],
        "rate_of_change": X[4],
        "upstream_kithulgala": X[5],
        "upstream_norwood": X[6],
        "upstream_deraniyagala": X[7],
        "antecedent_rainfall_7day": X[8],
        "duration_above_alert": X[9],
    }


# =============================================================
# UTILITY: DIMINISHING RETURNS FUNCTION
# =============================================================

def diminishing_score(value, threshold, observed_max, weight):
    """
    Score a value with full reward up to the observed historical max,
    then diminishing returns beyond it.

    - Below threshold: minimal score (proportional)
    - threshold to observed_max: full linear reward (this is the
      "known extreme" zone where we have real evidence)
    - Above observed_max: square root diminishing returns (possible
      but increasingly unlikely, so less reward per unit increase)

    This prevents the GA from endlessly pushing values to their
    upper bounds, because the marginal fitness gain decreases.

    Example with water level (threshold=19, observed_max=21.7):
        18.0m -> small score (below major flood)
        19.0m -> threshold bonus kicks in
        21.7m -> full reward (matches Ditwah)
        23.0m -> some additional reward, but less per meter
        25.0m -> very little additional reward vs 23.0m
    """
    if value <= threshold:
        # Below threshold: proportional score, no bonus
        return (value / threshold) * weight * 0.3

    elif value <= observed_max:
        # Between threshold and observed max: full linear reward
        # This is the "sweet spot" — extreme but historically precedented
        base = weight * 0.3  # credit for reaching threshold
        range_size = observed_max - threshold
        if range_size > 0:
            progress = (value - threshold) / range_size
            bonus = progress * weight * 0.7  # remaining 70% of weight
        else:
            bonus = 0
        return base + bonus

    else:
        # Beyond observed max: diminishing returns via square root
        # Full score for reaching observed_max, then sqrt bonus
        full_score = weight  # 100% of weight for reaching observed_max
        excess = value - observed_max
        range_size = observed_max - threshold
        if range_size > 0:
            # Normalize excess relative to the known range
            normalized_excess = excess / range_size
            # Square root gives diminishing returns
            diminishing_bonus = np.sqrt(normalized_excess) * weight * 0.2
        else:
            diminishing_bonus = 0
        return full_score + diminishing_bonus


# =============================================================
# COHERENCE CHECKS (physical plausibility)
# =============================================================

def check_rainfall_waterlevel_coherence(s):
    """
    Penalize scenarios where water level is extreme but rainfall
    is negligible (or vice versa).

    Calibrated against Ditwah: 21.70m with 91.8mm local rain is valid
    because upstream Kithulgala had 159.8mm driving flow downstream.
    """
    penalty = 0.0

    if s["water_level_noon"] > GC["minor_flood_level"]:
        upstream_any_elevated = (
                s["upstream_kithulgala"] >= UPSTREAM["Kithulgala"]["alert"] or
                s["upstream_norwood"] >= UPSTREAM["Norwood"]["alert"] or
                s["upstream_deraniyagala"] >= UPSTREAM["Deraniyagala"]["alert"]
        )

        if s["rainfall_24hr"] < 20.0 and s["antecedent_rainfall_7day"] < 50.0:
            if upstream_any_elevated:
                penalty -= 5.0
            else:
                penalty -= 30.0
        elif s["rainfall_24hr"] < 50.0 and s["antecedent_rainfall_7day"] < 100.0:
            if not upstream_any_elevated:
                penalty -= 15.0

    # 3hr rainfall exceeding 24hr rainfall — physically impossible
    if s["rainfall_3hr"] > s["rainfall_24hr"]:
        penalty -= 20.0

    return penalty


def check_temporal_coherence(s):
    """
    Check that 9am -> noon water level change matches the rate of change.
    Time gap = 3 hours.
    """
    penalty = 0.0
    expected_change = s["rate_of_change"] * 3.0
    actual_change = s["water_level_noon"] - s["water_level_9am"]

    deviation = abs(actual_change - expected_change)
    if deviation > 3.0:
        penalty -= 20.0
    elif deviation > 1.5:
        penalty -= 10.0
    elif deviation > 0.5:
        penalty -= 3.0

    if actual_change < -1.0 and s["rate_of_change"] > 1.0:
        penalty -= 15.0

    return penalty


def check_upstream_coherence(s):
    """
    If Glencourse is at major flood, upstream stations should be elevated.

    Calibrated from 29-Nov-2025 Ditwah event:
    - Glencourse hit 21.70m (Major Flood)
    - Kithulgala at 3.52m (Alert) — already falling
    - Deraniyagala at 2.97m (Normal) — tributary at tail end
    - Upstream may be receding when Glencourse peaks (travel time lag)
    """
    bonus = 0.0
    penalty = 0.0

    if s["water_level_noon"] >= GC["major_flood_level"]:
        upstream_elevated = 0
        if s["upstream_kithulgala"] >= UPSTREAM["Kithulgala"]["alert"]:
            upstream_elevated += 1
            bonus += 5.0
        if s["upstream_norwood"] >= UPSTREAM["Norwood"]["alert"]:
            upstream_elevated += 1
            bonus += 5.0
        if s["upstream_deraniyagala"] >= UPSTREAM["Deraniyagala"]["alert"]:
            upstream_elevated += 1
            bonus += 5.0

        if upstream_elevated == 0:
            penalty -= 15.0

        if s["upstream_kithulgala"] >= UPSTREAM["Kithulgala"]["major"]:
            bonus += 12.0
        if s["upstream_norwood"] >= UPSTREAM["Norwood"]["major"]:
            bonus += 10.0
        if s["upstream_deraniyagala"] >= UPSTREAM["Deraniyagala"]["major"]:
            bonus += 10.0

    elif s["water_level_noon"] >= GC["minor_flood_level"]:
        if s["upstream_kithulgala"] >= UPSTREAM["Kithulgala"]["alert"]:
            bonus += 3.0
        if s["upstream_norwood"] >= UPSTREAM["Norwood"]["alert"]:
            bonus += 3.0

    return bonus + penalty


def check_statistical_plausibility(s):
    """
    NEW in v2: Penalize scenarios that are statistically implausible
    based on historical data distributions.

    The idea: if the historical max water level is 21.70m (Ditwah),
    a scenario at 22.5m is plausible (slightly beyond observed),
    but 24.9m is extremely unlikely and should be penalized.

    Uses the concept of "standard deviations beyond the mean" —
    more extreme = exponentially increasing penalty.
    """
    penalty = 0.0

    # --- Water level plausibility ---
    # How many standard deviations above mean?
    if _stats["std_wl"] > 0:
        wl_z_score = (s["water_level_noon"] - _stats["mean_wl"]) / _stats["std_wl"]
    else:
        wl_z_score = 0

    # Beyond the historical max: increasing penalty
    if s["water_level_noon"] > _stats["max_wl"]:
        excess_beyond_max = s["water_level_noon"] - _stats["max_wl"]
        # Quadratic penalty: 1m beyond max = small, 3m beyond = 9x worse
        penalty -= (excess_beyond_max ** 2) * 2.0

    # --- Rainfall plausibility ---
    if s["rainfall_24hr"] > _stats["max_rf"]:
        excess_rf = s["rainfall_24hr"] - _stats["max_rf"]
        penalty -= (excess_rf / 50.0) ** 2 * 3.0

    # --- Combined implausibility ---
    # If BOTH water level AND rainfall are simultaneously at their
    # extreme upper bounds, that's exponentially more unlikely
    wl_ratio = s["water_level_noon"] / DITWAH["water_level"]
    rf_ratio = s["rainfall_24hr"] / DITWAH["rainfall_24hr"]

    if wl_ratio > 1.0 and rf_ratio > 2.0:
        # Both significantly beyond Ditwah — compound penalty
        combined_excess = (wl_ratio - 1.0) * (rf_ratio - 1.0)
        penalty -= combined_excess * 10.0

    # --- Antecedent rainfall plausibility ---
    # 1000mm in 7 days = ~143mm/day average for a full week
    # Even in extreme monsoons, this is at the outer edge
    if s["antecedent_rainfall_7day"] > 600:
        excess_ant = (s["antecedent_rainfall_7day"] - 600) / 100.0
        penalty -= excess_ant ** 2 * 1.5

    # --- Duration plausibility ---
    # The Ditwah event lasted ~36hrs above alert
    # 96 hours (4 days) continuously above alert is extremely rare
    if s["duration_above_alert"] > DITWAH["duration_above_alert"]:
        excess_dur = (s["duration_above_alert"] - DITWAH["duration_above_alert"]) / 12.0
        penalty -= excess_dur ** 2 * 1.0

    return penalty


def check_joint_consistency(s):
    """
    NEW in v2: Check that the COMBINATION of variables tells a
    coherent physical story, not just individual plausibility.

    Scenarios should follow one of these realistic patterns:

    Pattern A - "Flash Flood": Very high short-duration rainfall,
    rapid rise, moderate duration, moderate antecedent

    Pattern B - "Sustained Monsoon Flood": Moderate daily rainfall,
    very high antecedent (ground saturated), long duration,
    gradual rise — like an extended Ditwah

    Pattern C - "Upstream Cascade": Moderate local rainfall,
    very high upstream levels, rapid rise at Glencourse
    as flood wave arrives — this IS the Ditwah pattern

    Scenarios that mix incompatible elements get penalized.
    """
    bonus = 0.0
    penalty = 0.0

    # --- Pattern A: Flash Flood signature ---
    is_flash = (
            s["rainfall_3hr"] > 80 and
            s["rate_of_change"] > 1.5 and
            s["duration_above_alert"] < 24
    )
    if is_flash:
        bonus += 8.0  # coherent flash flood story

    # --- Pattern B: Sustained Monsoon signature ---
    is_sustained = (
            s["antecedent_rainfall_7day"] > 300 and
            s["duration_above_alert"] > 24 and
            abs(s["rate_of_change"]) < 1.0
    )
    if is_sustained:
        bonus += 8.0  # coherent sustained flood story

    # --- Pattern C: Upstream Cascade signature (Ditwah-like) ---
    upstream_max = max(
        s["upstream_kithulgala"] / UPSTREAM["Kithulgala"]["major"],
        s["upstream_norwood"] / UPSTREAM["Norwood"]["major"],
        s["upstream_deraniyagala"] / UPSTREAM["Deraniyagala"]["major"]
    )
    is_cascade = (
            upstream_max > 0.6 and
            s["water_level_noon"] >= GC["major_flood_level"] and
            s["rainfall_24hr"] < 200  # local rain is moderate
    )
    if is_cascade:
        bonus += 10.0  # most realistic extreme pattern for Glencourse

    # --- Incoherent combinations ---
    # Very high rate of rise + very long duration = contradictory
    # (if it's rising fast, it hasn't been above alert for days)
    if s["rate_of_change"] > 2.0 and s["duration_above_alert"] > 48:
        penalty -= 10.0

    # Extremely high local rainfall + very high upstream + very high antecedent
    # = everything simultaneously at maximum, which is the boundary-hugging problem
    all_high_count = 0
    if s["rainfall_24hr"] > 300: all_high_count += 1
    if s["antecedent_rainfall_7day"] > 700: all_high_count += 1
    if s["upstream_kithulgala"] > UPSTREAM["Kithulgala"]["major"]: all_high_count += 1
    if s["duration_above_alert"] > 60: all_high_count += 1
    if s["water_level_noon"] > 23: all_high_count += 1

    if all_high_count >= 4:
        # Penalize "everything at max" — real floods have a dominant driver,
        # not every variable simultaneously at extreme
        penalty -= (all_high_count - 3) * 15.0

    return bonus + penalty


# =============================================================
# MAIN FITNESS FUNCTION
# =============================================================

def fitness_function(X):
    """
    Evaluate how extreme and physically plausible a flood scenario is.

    Version 2 changes:
    - Uses diminishing_score() instead of linear scaling
    - Adds statistical plausibility checks
    - Adds joint consistency checks
    - Anchored against Ditwah reference event

    The GA MINIMIZES this function, so more extreme -> more negative.
    """
    s = decode_genome(X)
    score = 0.0

    # ---- 1. WATER LEVEL SEVERITY (diminishing returns) ----
    score += diminishing_score(
        value=s["water_level_noon"],
        threshold=GC["major_flood_level"],  # 19.0m
        observed_max=DITWAH["water_level"],  # 21.7m (real event)
        weight=fw["w_water_level_severity"]  # 15.0
    )

    # Threshold breach bonuses (kept but reduced from v1)
    if s["water_level_noon"] >= GC["major_flood_level"]:
        score += fw["w_threshold_breach_bonus"] * 0.5  # was 1.0, now 0.5
        excess = s["water_level_noon"] - GC["major_flood_level"]
        # Diminishing excess bonus: sqrt instead of linear
        score += np.sqrt(max(0, excess)) * 3.0  # was excess * 5.0
    elif s["water_level_noon"] >= GC["minor_flood_level"]:
        score += fw["w_threshold_breach_bonus"] * 0.2
    elif s["water_level_noon"] >= GC["alert_level"]:
        score += fw["w_threshold_breach_bonus"] * 0.05

    # ---- 2. RAINFALL INTENSITY (diminishing returns) ----
    score += diminishing_score(
        value=s["rainfall_24hr"],
        threshold=100.0,  # significant rain
        observed_max=DITWAH["rainfall_24hr"],  # 91.8mm at Glencourse
        weight=fw["w_rainfall_intensity"]  # 5.0
    )

    # 3hr rainfall bonus (smaller weight)
    if s["rainfall_3hr"] > 0:
        rf3_ratio = min(1.0, s["rainfall_3hr"] / GC["historical_max_3hr_rf"])
        score += rf3_ratio * fw["w_rainfall_intensity"] * 0.3

    # ---- 3. RATE OF RISE (diminishing returns) ----
    if s["rate_of_change"] > 0:
        # Reward rising, but diminishing after 1.5 m/hr
        if s["rate_of_change"] <= 1.5:
            score += (s["rate_of_change"] / 1.5) * fw["w_rate_of_rise"]
        else:
            base = fw["w_rate_of_rise"]
            excess = s["rate_of_change"] - 1.5
            score += base + np.sqrt(excess) * fw["w_rate_of_rise"] * 0.2

    # ---- 4. DURATION ABOVE ALERT (diminishing returns) ----
    score += diminishing_score(
        value=s["duration_above_alert"],
        threshold=6.0,  # meaningful duration
        observed_max=DITWAH["duration_above_alert"],  # 36hrs
        weight=fw["w_duration"]  # 6.0
    )

    # ---- 5. ANTECEDENT SATURATION (diminishing returns) ----
    score += diminishing_score(
        value=s["antecedent_rainfall_7day"],
        threshold=200.0,  # ground starts saturating
        observed_max=DITWAH["antecedent_7day"],  # 450mm
        weight=fw["w_antecedent_saturation"]  # 4.0
    )

    # ---- 6. UPSTREAM COHERENCE ----
    score += check_upstream_coherence(s) * (fw["w_upstream_coherence"] / 10.0)

    # ---- 7. PHYSICAL PLAUSIBILITY PENALTIES ----
    score += check_rainfall_waterlevel_coherence(s)
    score += check_temporal_coherence(s)

    # ---- 8. NEW: STATISTICAL PLAUSIBILITY PENALTIES ----
    score += check_statistical_plausibility(s)

    # ---- 9. NEW: JOINT CONSISTENCY BONUS/PENALTY ----
    score += check_joint_consistency(s)

    # GA minimizes -> negate to maximize extremity
    return -score


# =============================================================
# SCENARIO CLASSIFICATION & EXPORT HELPERS
# =============================================================

def classify_scenario(s):
    """Classify the flood severity of a decoded scenario."""
    wl = s["water_level_noon"]
    if wl >= GC["major_flood_level"]:
        level = "MAJOR_FLOOD"
    elif wl >= GC["minor_flood_level"]:
        level = "MINOR_FLOOD"
    elif wl >= GC["alert_level"]:
        level = "ALERT"
    else:
        level = "NORMAL"

    return {
        "flood_level": level,
        "severity_pct": round((wl / GC["major_flood_level"]) * 100, 1),
        "excess_above_major": round(max(0, wl - GC["major_flood_level"]), 2),
        "comparison_to_ditwah": round((wl / DITWAH["water_level"]) * 100, 1),
    }


def genome_to_nvidia_format(X, scenario_id=1):
    """
    Convert genome to JSON format expected by NVIDIA simulation pipeline.
    """
    s = decode_genome(X)
    classification = classify_scenario(s)

    return {
        "scenario_id": scenario_id,
        "station": "Glencourse",
        "river_basin": "Kelani Ganga (RB 01)",
        "flood_classification": classification["flood_level"],
        "severity_percent": classification["severity_pct"],
        "comparison_to_ditwah_pct": classification["comparison_to_ditwah"],
        "parameters": {
            "rainfall_24hr_mm": round(s["rainfall_24hr"], 2),
            "rainfall_3hr_mm": round(s["rainfall_3hr"], 2),
            "water_level_9am_m": round(s["water_level_9am"], 2),
            "water_level_noon_m": round(s["water_level_noon"], 2),
            "rate_of_change_m_per_hr": round(s["rate_of_change"], 3),
            "duration_above_alert_hrs": round(s["duration_above_alert"], 1),
            "antecedent_rainfall_7day_mm": round(s["antecedent_rainfall_7day"], 1),
        },
        "upstream_conditions": {
            "kithulgala_wl_m": round(s["upstream_kithulgala"], 2),
            "norwood_wl_m": round(s["upstream_norwood"], 2),
            "deraniyagala_wl_m": round(s["upstream_deraniyagala"], 2),
        },
        "thresholds": {
            "alert_level_m": GC["alert_level"],
            "minor_flood_level_m": GC["minor_flood_level"],
            "major_flood_level_m": GC["major_flood_level"],
        },
        "reference_event": {
            "name": "Ditwah 29-Nov-2025",
            "water_level_m": DITWAH["water_level"],
        }
    }