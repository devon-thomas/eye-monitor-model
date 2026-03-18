import copy
import json
import sys

from eye_mon.engine import load_config, validate_config, run_model


HEIGHT_METRIC = "eye_height_center_safe"
AREA_METRIC = "eye_area_safe"


def load_sweep_config(path):
    with open(path, "r") as f:
        return json.load(f)

def load_tx_settings(sweep_cfg):
    def _filter_enabled(settings, source_name):
        if not isinstance(settings, list) or len(settings) == 0:
            raise ValueError("%s must contain a non-empty JSON list" % source_name)

        enabled_settings = [s for s in settings if bool(s.get("enabled", True))]

        print(
            "Loaded %d TX settings (%d enabled) from %s"
            % (len(settings), len(enabled_settings), source_name)
        )

        if len(enabled_settings) == 0:
            raise ValueError("No enabled TX settings found in %s" % source_name)

        return enabled_settings

    # ----------------------------
    # Case 1: explicit tx_settings_file
    # ----------------------------
    if "tx_settings_file" in sweep_cfg:
        tx_settings_file = sweep_cfg["tx_settings_file"]
        with open(tx_settings_file, "r") as f:
            tx_settings = json.load(f)

        return _filter_enabled(tx_settings, tx_settings_file)

    # ----------------------------
    # Case 2: tx_settings (string or inline)
    # ----------------------------
    if "tx_settings" in sweep_cfg:
        tx_settings = sweep_cfg["tx_settings"]

        # Case 2a: string → treat as file path
        if isinstance(tx_settings, str):
            with open(tx_settings, "r") as f:
                file_settings = json.load(f)

            return _filter_enabled(file_settings, tx_settings)

        # Case 2b: inline list
        return _filter_enabled(tx_settings, "inline config")

    # ----------------------------
    # Error
    # ----------------------------
    raise ValueError(
        "Sweep config must contain either 'tx_settings_file' or 'tx_settings'"
    )


def apply_tx_setting(cfg, tx_setting):
    model = tx_setting.get("model", "fir_3tap")
    enabled = bool(tx_setting.get("enabled", True))

    cfg["tx_eq"]["enabled"] = enabled
    cfg["tx_eq"]["model"] = model

    if model == "fir_3tap":
        cfg["tx_eq"]["fir_3tap"]["c_pre"] = float(tx_setting["c_pre"])
        cfg["tx_eq"]["fir_3tap"]["c_main"] = float(tx_setting["c_main"])
        cfg["tx_eq"]["fir_3tap"]["c_post"] = float(tx_setting["c_post"])

    elif model == "fir_3tap_causal":
        cfg["tx_eq"]["fir_3tap_causal"]["c_main"] = float(tx_setting["c_main"])
        cfg["tx_eq"]["fir_3tap_causal"]["c_post1"] = float(tx_setting["c_post1"])
        cfg["tx_eq"]["fir_3tap_causal"]["c_post2"] = float(tx_setting["c_post2"])

    else:
        raise ValueError("Unsupported TX setting model: %s" % model)


def apply_sampling_override(cfg, sampling_override):
    if "sigma_j" in sampling_override:
        cfg["sampling"]["sigma_j"] = float(sampling_override["sigma_j"])
    if "n_ui_collect" in sampling_override:
        cfg["sampling"]["n_ui_collect"] = int(sampling_override["n_ui_collect"])
    if "n_phases" in sampling_override:
        cfg["sampling"]["n_phases"] = int(sampling_override["n_phases"])
    if "skip_initial_ui" in sampling_override:
        cfg["sampling"]["skip_initial_ui"] = int(sampling_override["skip_initial_ui"])


def apply_jitter_override(cfg, sigma_j):
    cfg["sampling"]["sigma_j"] = float(sigma_j)


def _result_row_from_tx_setting(tx_setting):
    model = tx_setting.get("model", "fir_3tap")

    if model == "fir_3tap":
        return {
            "model": model,
            "c1": float(tx_setting["c_pre"]),
            "c2": float(tx_setting["c_main"]),
            "c3": float(tx_setting["c_post"]),
        }

    if model == "fir_3tap_causal":
        return {
            "model": model,
            "c1": float(tx_setting["c_main"]),
            "c2": float(tx_setting["c_post1"]),
            "c3": float(tx_setting["c_post2"]),
        }

    raise ValueError("Unsupported TX setting model: %s" % model)


def evaluate_one_setting(base_cfg, tx_setting):
    cfg = copy.deepcopy(base_cfg)
    apply_tx_setting(cfg, tx_setting)
    validate_config(cfg)
    sim = run_model(cfg)

    row = _result_row_from_tx_setting(tx_setting)
    metrics = sim["metrics"]

    return {
        "name": tx_setting["name"],
        "enabled": bool(tx_setting.get("enabled", True)),
        "model": row["model"],
        "c1": row["c1"],
        "c2": row["c2"],
        "c3": row["c3"],
        HEIGHT_METRIC: float(metrics[HEIGHT_METRIC]),
        AREA_METRIC: float(metrics[AREA_METRIC]),
        "best_phase_ui": float(metrics["best_phase_ui"]),
        "eye_width_ui": float(metrics["eye_width_ui"]),
        "eye_height_center_raw": float(metrics["eye_height_center_raw"]),
        "eye_area_raw": float(metrics["eye_area_raw"]),
    }


def rank_results(results, metric_name):
    return sorted(results, key=lambda r: r[metric_name], reverse=True)


def print_results_table(title, results, metric_name):
    print("")
    print("%s [%s]" % (title, metric_name))
    print("-" * (len(title) + len(metric_name) + 3))
    print(
        "%-12s %-18s %-10s %-10s %-10s %-14s %-14s %-14s"
        % ("name", "model", "c1", "c2", "c3", "score", "best_phase_ui", "eye_width_ui")
    )
    for r in results:
        print(
            "%-12s %-18s %-10.4f %-10.4f %-10.4f %-14.6f %-14.6f %-14.6f"
            % (
                r["name"],
                r["model"],
                r["c1"],
                r["c2"],
                r["c3"],
                r[metric_name],
                r["best_phase_ui"],
                r["eye_width_ui"],
            )
        )
    print("")


def save_results_json(path, tbit, oracle_by_metric, jitter_runs_by_metric):
    output_data = {
        "tbit": tbit,
        "oracle": oracle_by_metric,
        "jitter_runs": jitter_runs_by_metric,
    }

    with open(path, "w") as f:
        json.dump(output_data, f, indent=2)
        f.write("\n")


def save_csv_pretty(path, oracle_by_metric, jitter_runs_by_metric):
    import csv

    metrics_to_write = [HEIGHT_METRIC, AREA_METRIC]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        for metric_name in metrics_to_write:
            oracle_ranked = oracle_by_metric[metric_name]["ranking"]
            writer.writerow([f"Oracle ranking [{metric_name}]"])
            writer.writerow(["name", "model", "c1", "c2", "c3", "score", "best_phase_ui", "eye_width_ui"])

            for r in oracle_ranked:
                writer.writerow([
                    r["name"],
                    r["model"],
                    r["c1"],
                    r["c2"],
                    r["c3"],
                    r[metric_name],
                    r["best_phase_ui"],
                    r["eye_width_ui"],
                ])

            writer.writerow([])

            for run in jitter_runs_by_metric[metric_name]:
                sigma_j_fs = run["sigma_j"] * 1e15

                writer.writerow([f"Jitter ranking [{metric_name}]: sigma_j = {sigma_j_fs:.3f} fs"])
                writer.writerow(["name", "model", "c1", "c2", "c3", "score", "best_phase_ui", "eye_width_ui"])

                for r in run["ranked_results"]:
                    writer.writerow([
                        r["name"],
                        r["model"],
                        r["c1"],
                        r["c2"],
                        r["c3"],
                        r[metric_name],
                        r["best_phase_ui"],
                        r["eye_width_ui"],
                    ])

                writer.writerow([])


def build_oracle_cfg(base_cfg, sweep_cfg):
    cfg = copy.deepcopy(base_cfg)
    oracle_cfg = sweep_cfg.get("oracle", {})
    sampling_override = oracle_cfg.get("sampling", {})
    apply_sampling_override(cfg, sampling_override)
    return cfg


def run_single_sweep(base_cfg, tx_settings):
    results = []
    for tx_setting in tx_settings:
        result = evaluate_one_setting(base_cfg, tx_setting)
        results.append(result)
    return results


def run_oracle_sweep(base_cfg, sweep_cfg, tx_settings):
    oracle_cfg = build_oracle_cfg(base_cfg, sweep_cfg)
    results = run_single_sweep(oracle_cfg, tx_settings)

    return {
        HEIGHT_METRIC: rank_results(results, HEIGHT_METRIC),
        AREA_METRIC: rank_results(results, AREA_METRIC),
    }


def run_jitter_sweeps(base_cfg, sweep_cfg, tx_settings):
    jitter_values = sweep_cfg["jitter_values"]

    by_metric = {
        HEIGHT_METRIC: [],
        AREA_METRIC: [],
    }

    for sigma_j in jitter_values:
        cfg = copy.deepcopy(base_cfg)
        apply_jitter_override(cfg, sigma_j)
        results = run_single_sweep(cfg, tx_settings)

        by_metric[HEIGHT_METRIC].append({
            "sigma_j": float(sigma_j),
            "ranked_results": rank_results(results, HEIGHT_METRIC),
            "winner": rank_results(results, HEIGHT_METRIC)[0],
        })

        by_metric[AREA_METRIC].append({
            "sigma_j": float(sigma_j),
            "ranked_results": rank_results(results, AREA_METRIC),
            "winner": rank_results(results, AREA_METRIC)[0],
        })

    return by_metric


def print_summary_comparison(oracle_by_metric, jitter_runs_by_metric, tbit):
    for metric_name in [HEIGHT_METRIC, AREA_METRIC]:
        oracle_winner = oracle_by_metric[metric_name][0]

        print("")
        print("Oracle winner [%s]" % metric_name)
        print("-" * (17 + len(metric_name)))
        print(
            "name=%s  model=%s  score=%.6f  best_phase=%.6f UI (%.3f ps)"
            % (
                oracle_winner["name"],
                oracle_winner["model"],
                oracle_winner[metric_name],
                oracle_winner["best_phase_ui"],
                oracle_winner["best_phase_ui"] * tbit * 1e12,
            )
        )

        print("")
        print("Jitter sweep winner comparison [%s]" % metric_name)
        print("-" * (31 + len(metric_name)))
        print(
            "%-14s %-12s %-18s %-12s %-14s %-14s"
            % ("sigma_j [fs]", "winner", "model", "matches?", "score", "best_phase_ui")
        )

        for run in jitter_runs_by_metric[metric_name]:
            sigma_j_fs = run["sigma_j"] * 1e15
            winner = run["winner"]
            matches = "yes" if winner["name"] == oracle_winner["name"] else "no"
            print(
                "%-14.3f %-12s %-18s %-12s %-14.6f %-14.6f"
                % (
                    sigma_j_fs,
                    winner["name"],
                    winner["model"],
                    matches,
                    winner[metric_name],
                    winner["best_phase_ui"],
                )
            )
        print("")


def main():
    sweep_config_path = "config_sweep.json"
    if len(sys.argv) > 1:
        sweep_config_path = sys.argv[1]

    sweep_cfg = load_sweep_config(sweep_config_path)
    base_config_path = sweep_cfg["base_config"]
    base_cfg = load_config(base_config_path)

    validate_config(base_cfg)

    tbit = float(base_cfg["data"]["tbit"])
    tx_settings = load_tx_settings(sweep_cfg)
    oracle_rankings = run_oracle_sweep(base_cfg, sweep_cfg, tx_settings)
    jitter_runs = run_jitter_sweeps(base_cfg, sweep_cfg, tx_settings)

    for metric_name in [HEIGHT_METRIC, AREA_METRIC]:
        print_results_table("Oracle ranking", oracle_rankings[metric_name], metric_name)
        for run in jitter_runs[metric_name]:
            title = "Jitter ranking: sigma_j = %.3f fs" % (run["sigma_j"] * 1e15)
            print_results_table(title, run["ranked_results"], metric_name)

    print_summary_comparison(oracle_rankings, jitter_runs, tbit)

    if "output_file" in sweep_cfg:
        oracle_payload = {
            HEIGHT_METRIC: {
                "metric": HEIGHT_METRIC,
                "winner": oracle_rankings[HEIGHT_METRIC][0],
                "ranking": oracle_rankings[HEIGHT_METRIC],
            },
            AREA_METRIC: {
                "metric": AREA_METRIC,
                "winner": oracle_rankings[AREA_METRIC][0],
                "ranking": oracle_rankings[AREA_METRIC],
            },
        }

        jitter_payload = {
            HEIGHT_METRIC: jitter_runs[HEIGHT_METRIC],
            AREA_METRIC: jitter_runs[AREA_METRIC],
        }

        save_results_json(
            sweep_cfg["output_file"],
            tbit,
            oracle_payload,
            jitter_payload
        )
        print("Saved results to %s" % sweep_cfg["output_file"])

    if "output_csv" in sweep_cfg:
        oracle_payload = {
            HEIGHT_METRIC: {"ranking": oracle_rankings[HEIGHT_METRIC]},
            AREA_METRIC: {"ranking": oracle_rankings[AREA_METRIC]},
        }

        save_csv_pretty(
            sweep_cfg["output_csv"],
            oracle_payload,
            jitter_runs
        )
        print("Saved CSV (pretty format) to %s" % sweep_cfg["output_csv"])


if __name__ == "__main__":
    main()
