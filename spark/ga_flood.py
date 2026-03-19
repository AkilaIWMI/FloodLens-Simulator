# ----- Glencourse Station Thresholds (from Irrigation Dept) -----
GLENCOURSE = {
    "unit": "m",
    "alert_level": 15.00,
    "minor_flood_level": 16.50,
    "major_flood_level": 19.00,
    "historical_max_wl": 25.00,       # highest plausible water level (m)
    "historical_min_wl": 0.05,        # lowest recorded water level (m)
    "historical_max_rainfall": 400.0, # max 24hr rainfall (mm) - Kithulgala hit 159.8mm in 21hrs
    "historical_max_3hr_rf": 180.0,   # max 3hr rainfall (mm)
}

# ----- General Params -----
params = {
    "inputfile": "data/glencourse_data.csv",  # preprocessed from kelani_flood_guages.csv
    "input_name": "Glencourse_Flood",
    "normalize_data": True,
    "test_size": 0.3,
    "output_colname": "water_level_noon",  # target: 12:00 noon water level
    "fitness_metric": "flood_severity",
    "var_type": "real",
    "timeout": 5000
}

# ----- GA Params -----
# Tuned for scenario exploration (larger population, more generations)
ga_params = {
    "max_num_iteration": 150,
    "population_size": 300,
    "mutation_probability": 0.15,
    "elit_ratio": 0.05,
    "parents_portion": 0.3,
    "crossover_type": "uniform",
    "max_iteration_without_improv": 40
}

# ----- Genome Structure -----
# Each individual (scenario) is a vector of these variables:
#
# Gene Index | Variable                    | Unit | Range
# -----------|-----------------------------|------|---------------------------
# 0          | rainfall_24hr               | mm   | [0, historical_max_rainfall]
# 1          | rainfall_3hr                | mm   | [0, historical_max_3hr_rf]
# 2          | water_level_9am             | m    | [historical_min, historical_max]
# 3          | water_level_noon            | m    | [historical_min, historical_max]
# 4          | water_level_rate_of_change  | m/hr | [-1.0, 3.0] (neg=falling, pos=rising)
# 5          | upstream_kithulgala_wl      | m    | [0.5, 7.0]
# 6          | upstream_norwood_wl         | m    | [0.2, 5.0]
# 7          | upstream_deraniyagala_wl    | m    | [0.05, 7.0]
# 8          | antecedent_rainfall_7day    | mm   | [0, 800]
# 9          | duration_hours_above_alert  | hrs  | [0, 72]

genome_config = {
    "n_genes": 10,
    "gene_names": [
        "rainfall_24hr",
        "rainfall_3hr",
        "water_level_9am",
        "water_level_noon",
        "water_level_rate_of_change",
        "upstream_kithulgala_wl",
        "upstream_norwood_wl",
        "upstream_deraniyagala_wl",
        "antecedent_rainfall_7day",
        "duration_hours_above_alert"
    ],
    "bounds": [
        [0.0, 400.0],      # rainfall_24hr (mm) - Kithulgala saw ~183mm/24hr equivalent
        [0.0, 180.0],      # rainfall_3hr (mm)
        [0.05, 25.0],      # water_level_9am (m) - allow beyond 21.7m observed
        [0.05, 25.0],      # water_level_noon (m) - allow beyond 21.7m observed
        [-1.5, 3.5],       # rate of change (m/hr) - widened for extreme flash scenarios
        [0.5, 8.0],        # Kithulgala upstream WL - was at 3.52m during Ditwah event
        [0.2, 6.0],        # Norwood upstream WL - gauge went NA during event
        [0.05, 8.0],       # Deraniyagala upstream WL - was at 2.97m (tail end)
        [0.0, 1000.0],     # antecedent 7-day rainfall - monsoon can be relentless
        [0.0, 96.0],       # duration above alert (hrs) - extended to 4 days
    ]
}

# ----- Upstream Station Thresholds (for coherence checks) -----
# Hanwella added - was at MAJOR FLOOD (10.47m) during 29-Nov-2025 event
UPSTREAM = {
    "Kithulgala":  {"alert": 3.00, "minor": 4.00, "major": 6.00},
    "Norwood":     {"alert": 1.50, "minor": 3.00, "major": 4.50},
    "Deraniyagala": {"alert": 4.80, "minor": 5.80, "major": 6.40},
    "Hanwella":    {"alert": 7.00, "minor": 8.00, "major": 10.00},
}

# ----- 29-Nov-2025 Ditwah Reference Event -----
# Use this to validate GA outputs - any generated scenario should
# be comparable to or more extreme than this real event
DITWAH_REFERENCE = {
    "date": "2025-11-29",
    "glencourse_wl_peak": 21.70,
    "glencourse_rainfall_21hr": 91.8,
    "kithulgala_wl": 3.52,
    "kithulgala_rainfall_21hr": 159.8,
    "hanwella_wl": 10.47,
    "hanwella_rainfall_21hr": 41.6,
    "deraniyagala_wl": 2.97,
    "deraniyagala_rainfall_21hr": 92.4,
    "holombuwa_rainfall_21hr": 135.2,
    "glencourse_status": "Major Flood",
    "hanwella_status": "Major Flood",
    "nagalagam_status": "Minor Flood",
    "kithulgala_status": "Alert",
    "water_trend": "Falling",
}

# ----- Fitness Weights (tune these to shape what "extreme" means) -----
fitness_weights = {
    "w_water_level_severity": 15.0,   # how close/above major flood
    "w_rainfall_intensity": 5.0,      # heavier rain = more extreme
    "w_rate_of_rise": 8.0,            # rapid rise is more dangerous
    "w_upstream_coherence": 10.0,     # upstream flooding should support downstream
    "w_duration": 6.0,                # prolonged flooding is worse
    "w_antecedent_saturation": 4.0,   # saturated ground = worse flooding
    "w_threshold_breach_bonus": 20.0, # bonus for crossing major flood level
    "w_incoherence_penalty": -25.0,   # penalty for physically impossible combos
}

# ----- Export Config -----
export_config = {
    "output_dir": "../results/Glencourse_Flood/",
    "json_filename": "extreme_scenarios.json",
    "csv_filename": "extreme_scenarios.csv",
    "nvidia_json_filename": "nvidia_sim_input.json",  # formatted for NVIDIA pipeline
}