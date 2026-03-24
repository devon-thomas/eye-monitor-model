import copy
import csv
import json
import sys


HEIGHT_METRIC = "eye_height_center_safe"
AREA_METRIC = "eye_area_safe"
DEFAULT_METRICS = [HEIGHT_METRIC, AREA_METRIC]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def save_csv(path, rows):
    if len(rows) == 0:
        with open(path, "w") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_adc_bits(cfg, adc_bits, adc_fullscale):
    if adc_bits is None:
        cfg["adc"] = {
            "enabled": False,
            "n_bits": 8,
            "vmin": float(adc_fullscale["vmin"]),
            "vmax": float(adc_fullscale["vmax"]),
        }
    else:
        cfg["adc"] = {
            "enabled": True,
            "n_bits": int(adc_bits),
            "vmin": float(adc_fullscale["vmin"]),
            "vmax": float(adc_fullscale["vmax"]),
        }


def apply_tx_setting(cfg, tx_setting):
    cfg["tx_eq"]["enabled"] = bool(tx_setting.get("enabled", True))
    cfg["tx_eq"]["model"] = tx_setting["model"]

    if tx_setting["model"] == "fir_3tap":
        cfg["tx_eq"]["fir_3tap"]["c_pre"] = float(tx_setting["c_pre"])
        cfg["tx_eq"]["fir_3tap"]["c_main"] = float(tx_setting["c_main"])
        cfg["tx_eq"]["fir_3tap"]["c_post"] = float(tx_setting["c_post"])
        return

    if tx_setting["model"] == "fir_3tap_causal":
        cfg["tx_eq"]["fir_3tap_causal"]["c_main"] = float(tx_setting["c_main"])
        cfg["tx_eq"]["fir_3tap_causal"]["c_post1"] = float(tx_setting["c_post1"])
        cfg["tx_eq"]["fir_3tap_causal"]["c_post2"] = float(tx_setting["c_post2"])
        return

    raise ValueError("Unsupported TX model: %s" % tx_setting["model"])


def load_tx_settings(sweep_cfg):
    def _filter_settings(settings, source_name):
        if not isinstance(settings, list) or len(settings) == 0:
            raise ValueError("%s must contain a non-empty JSON list" % source_name)

        model_filter = sweep_cfg.get("tx_model_filter", None)
        if model_filter is not None:
            if not isinstance(model_filter, list) or len(model_filter) == 0:
                raise ValueError("tx_model_filter must be a non-empty list if provided")

            model_filter = set(model_filter)
            settings = [s for s in settings if s.get("model") in model_filter]

            print(
                "Filtered TX settings by model %s: %d remain"
                % (sorted(model_filter), len(settings))
            )

        if len(settings) == 0:
            raise ValueError("No TX settings remain after model filtering in %s" % source_name)

        enabled_settings = [s for s in settings if bool(s.get("enabled", True))]
        print(
            "Loaded %d TX settings (%d enabled) from %s"
            % (len(settings), len(enabled_settings), source_name)
        )

        if len(enabled_settings) == 0:
            raise ValueError("No enabled TX settings found in %s" % source_name)

        return enabled_settings

    if "tx_settings_file" in sweep_cfg:
        tx_settings_file = sweep_cfg["tx_settings_file"]
        with open(tx_settings_file, "r") as f:
            tx_settings = json.load(f)
        return _filter_settings(tx_settings, tx_settings_file)

    if "tx_settings" in sweep_cfg:
        tx_settings = sweep_cfg["tx_settings"]

        if isinstance(tx_settings, str):
            with open(tx_settings, "r") as f:
                file_settings = json.load(f)
            return _filter_settings(file_settings, tx_settings)

        return _filter_settings(tx_settings, "inline config")

    raise ValueError("Sweep config must contain either 'tx_settings_file' or 'tx_settings'")


def metric_value(sim, metric_name):
    return float(sim["metrics"][metric_name])


def rank_tx_settings(base_cfg, tx_settings, run_model, validate_config, metric_name):
    results = []

    for tx_setting in tx_settings:
        cfg = copy.deepcopy(base_cfg)
        apply_tx_setting(cfg, tx_setting)
        validate_config(cfg)
        sim = run_model(cfg)

        results.append({
            "tx_name": tx_setting["name"],
            "tx_model": tx_setting["model"],
            "metric_name": metric_name,
            "metric_value": metric_value(sim, metric_name),
            "best_phase_ui": float(sim["metrics"]["best_phase_ui"]),
            "eye_width_ui": float(sim["metrics"]["eye_width_ui"]),
            "eye_area_safe": float(sim["metrics"]["eye_area_safe"]),
            "eye_height_center_safe": float(sim["metrics"]["eye_height_center_safe"]),
        })

    ranked = sorted(results, key=lambda r: r["metric_value"], reverse=True)
    return ranked


def find_rank(ranked, tx_name):
    for idx, row in enumerate(ranked, start=1):
        if row["tx_name"] == tx_name:
            return idx
    return None


def build_oracle_cfg(base_cfg, sweep_cfg):
    cfg = copy.deepcopy(base_cfg)

    oracle = sweep_cfg.get("oracle", {})
    for section_name, section_updates in oracle.items():
        if section_name not in cfg:
            cfg[section_name] = {}
        for key, value in section_updates.items():
            cfg[section_name][key] = value

    if "adc" not in oracle:
        cfg["adc"] = {
            "enabled": False,
            "n_bits": 8,
            "vmin": -0.5,
            "vmax": 0.5,
        }

    return cfg


def build_impaired_cfg(base_cfg, jitter_value, adc_bits, adc_fullscale):
    cfg = copy.deepcopy(base_cfg)
    cfg["sampling"]["sigma_j"] = float(jitter_value)
    configure_adc_bits(cfg, adc_bits, adc_fullscale)
    return cfg


def normalize_metric_names(sweep_cfg):
    metric_names = sweep_cfg.get("metrics", DEFAULT_METRICS)
    if not isinstance(metric_names, list) or len(metric_names) == 0:
        raise ValueError("metrics must be a non-empty list if provided")

    allowed = set(DEFAULT_METRICS)
    for m in metric_names:
        if m not in allowed:
            raise ValueError("Unsupported metric: %s" % m)

    return metric_names


def format_adc_label(adc_bits):
    return "ideal" if adc_bits is None else ("%db" % int(adc_bits))


def main():
    default_cfg = "config/config_jitter_adc.json"
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else default_cfg
    print("[INFO] Using sweep config:", cfg_path)

    sweep_cfg = load_json(cfg_path)

    from eye_mon.engine import load_config, validate_config, run_model

    base_cfg = load_config(sweep_cfg["base_config"])
    validate_config(base_cfg)

    tx_settings = load_tx_settings(sweep_cfg)

    metric_names = normalize_metric_names(sweep_cfg)
    jitter_values = list(sweep_cfg["jitter_values"])
    adc_bits_list = list(sweep_cfg["adc_bits"])
    adc_fullscale = sweep_cfg.get("adc_fullscale", {"vmin": -0.5, "vmax": 0.5})

    oracle_cfg = build_oracle_cfg(base_cfg, sweep_cfg)
    validate_config(oracle_cfg)

    oracle_ranked_by_metric = {}
    oracle_best_by_metric = {}

    for metric_name in metric_names:
        oracle_ranked = rank_tx_settings(
            oracle_cfg, tx_settings, run_model, validate_config, metric_name
        )
        oracle_ranked_by_metric[metric_name] = oracle_ranked
        oracle_best_by_metric[metric_name] = oracle_ranked[0]

    print("")
    print("=== Oracle Winners ===")
    for metric_name in metric_names:
        oracle_best = oracle_best_by_metric[metric_name]
        print("Metric: %s" % metric_name)
        print("Winner: %s (%s)" % (oracle_best["tx_name"], oracle_best["tx_model"]))
        print("Metric value: %.6f" % oracle_best["metric_value"])
        print("")

    out_rows = []

    for jitter_value in jitter_values:
        for adc_bits in adc_bits_list:
            cfg = build_impaired_cfg(base_cfg, jitter_value, adc_bits, adc_fullscale)
            validate_config(cfg)

            adc_label = format_adc_label(adc_bits)

            for metric_name in metric_names:
                ranked = rank_tx_settings(
                    cfg, tx_settings, run_model, validate_config, metric_name
                )
                chosen = ranked[0]
                oracle_best = oracle_best_by_metric[metric_name]
                oracle_best_name = oracle_best["tx_name"]
                oracle_rank = find_rank(ranked, oracle_best_name)
                match = (chosen["tx_name"] == oracle_best_name)

                row = {
                    "metric": metric_name,
                    "sigma_j_s": float(jitter_value),
                    "sigma_j_fs": float(jitter_value) * 1e15,
                    "adc_bits": "" if adc_bits is None else int(adc_bits),
                    "adc_label": adc_label,
                    "adc_enabled": False if adc_bits is None else True,
                    "oracle_tx_name": oracle_best["tx_name"],
                    "oracle_tx_model": oracle_best["tx_model"],
                    "oracle_metric_value": oracle_best["metric_value"],
                    "chosen_tx_name": chosen["tx_name"],
                    "chosen_tx_model": chosen["tx_model"],
                    "chosen_metric_value": chosen["metric_value"],
                    "match_oracle": bool(match),
                    "oracle_rank_under_impaired": oracle_rank,
                    "chosen_best_phase_ui": chosen["best_phase_ui"],
                    "chosen_eye_width_ui": chosen["eye_width_ui"],
                    "chosen_eye_area_safe": chosen["eye_area_safe"],
                    "chosen_eye_height_center_safe": chosen["eye_height_center_safe"],
                }
                out_rows.append(row)

                print(
                    "metric = %-24s | sigma_j = %8.3f fs | adc = %-5s | chosen = %-30s | oracle_rank = %-3s | match = %s"
                    % (
                        metric_name,
                        float(jitter_value) * 1e15,
                        adc_label,
                        chosen["tx_name"],
                        str(oracle_rank),
                        str(match),
                    )
                )

    output_json = sweep_cfg.get("output_file", "./results/jitter_adc_sweep.json")
    output_csv = sweep_cfg.get("output_csv", "./results/jitter_adc_sweep.csv")

    save_json(output_json, {
        "metrics": metric_names,
        "oracle_best_by_metric": oracle_best_by_metric,
        "results": out_rows,
    })
    save_csv(output_csv, out_rows)

    print("")
    print("Saved JSON to %s" % output_json)
    print("Saved CSV  to %s" % output_csv)


if __name__ == "__main__":
    main()
