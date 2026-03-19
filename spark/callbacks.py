import numpy as np
import os

from params import params as p
from params import GLENCOURSE as GC
from params import genome_config as gc
from fitness import decode_genome, classify_scenario


class Callbacks:

    @staticmethod
    def SavePopulation_bests(
        folder="../results/" + p["input_name"] + "/tmp/",
        save_gen_step=1,
        file_prefix="population_bests"
    ):
        """
        Callback that runs after each GA generation.
        Saves the best (most extreme & plausible) flood scenario
        from each generation to a .npz file.
        """

        def func(generation_number, report_list, last_population, last_scores):
            if generation_number % save_gen_step != 0:
                return

            # Ensure output directory exists
            os.makedirs(folder, exist_ok=True)

            # --- Find the best individual (lowest score = most extreme) ---
            best_idx = np.argmin(last_scores)
            best_genome = last_population[best_idx]
            best_fitness = last_scores[best_idx]

            # --- Decode and classify the scenario ---
            scenario = decode_genome(best_genome)
            classification = classify_scenario(scenario)

            # --- Build the data dict to save ---
            save_data = {
                "generation": [generation_number],
                "individual": [best_genome],
                "fitness_score": [abs(best_fitness)],

                # Decoded scenario values
                "rainfall_24hr": [round(scenario["rainfall_24hr"], 2)],
                "rainfall_3hr": [round(scenario["rainfall_3hr"], 2)],
                "water_level_9am": [round(scenario["water_level_9am"], 2)],
                "water_level_noon": [round(scenario["water_level_noon"], 2)],
                "rate_of_change": [round(scenario["rate_of_change"], 3)],
                "upstream_kithulgala": [round(scenario["upstream_kithulgala"], 2)],
                "upstream_norwood": [round(scenario["upstream_norwood"], 2)],
                "upstream_deraniyagala": [round(scenario["upstream_deraniyagala"], 2)],
                "antecedent_rainfall_7day": [round(scenario["antecedent_rainfall_7day"], 1)],
                "duration_above_alert": [round(scenario["duration_above_alert"], 1)],

                # Classification
                "flood_level": [classification["flood_level"]],
                "severity_pct": [classification["severity_pct"]],
                "excess_above_major": [classification["excess_above_major"]],
            }

            # --- Population statistics for this generation ---
            all_fitnesses = np.abs(last_scores)
            save_data["pop_mean_fitness"] = [np.mean(all_fitnesses)]
            save_data["pop_best_fitness"] = [np.max(all_fitnesses)]
            save_data["pop_worst_fitness"] = [np.min(all_fitnesses)]
            save_data["pop_std_fitness"] = [np.std(all_fitnesses)]

            # Count how many individuals in population are at each flood level
            major_count = 0
            minor_count = 0
            alert_count = 0
            for individual in last_population:
                wl = individual[3]  # water_level_noon is gene index 3
                if wl >= GC["major_flood_level"]:
                    major_count += 1
                elif wl >= GC["minor_flood_level"]:
                    minor_count += 1
                elif wl >= GC["alert_level"]:
                    alert_count += 1

            save_data["pop_major_flood_count"] = [major_count]
            save_data["pop_minor_flood_count"] = [minor_count]
            save_data["pop_alert_count"] = [alert_count]

            # Save to npz
            filepath = os.path.join(
                folder,
                f"{file_prefix}_{generation_number}.npz"
            )
            np.savez(filepath, **save_data)

            # Print progress
            print(f"  Gen {generation_number:>4d} | "
                  f"Best WL: {scenario['water_level_noon']:.2f}m | "
                  f"Rain: {scenario['rainfall_24hr']:.0f}mm | "
                  f"Status: {classification['flood_level']} | "
                  f"Score: {abs(best_fitness):.2f} | "
                  f"Pop Major: {major_count}/{len(last_population)}")

        return func