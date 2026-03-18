import json
import sys


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def expand_range(start, step, stop):
    """
    Expand an inclusive numeric range:
        start, start+step, ..., stop

    Special case:
        if start == stop -> return [start]
    """
    start = float(start)
    step = float(step)
    stop = float(stop)

    if abs(stop - start) < 1e-15:
        return [round(start, 12)]

    if step == 0:
        raise ValueError("Range step must be nonzero when start != stop")

    values = []
    x = start

    if step > 0:
        if stop < start:
            raise ValueError("Positive step requires stop >= start")
        while x <= stop + 1e-12:
            values.append(round(x, 12))
            x += step
    else:
        if stop > start:
            raise ValueError("Negative step requires stop <= start")
        while x >= stop - 1e-12:
            values.append(round(x, 12))
            x += step

    return values


def build_settings_centered(range_cfg):
    pre_vals = expand_range(
        range_cfg["pre"]["start"],
        range_cfg["pre"]["step"],
        range_cfg["pre"]["stop"],
    )

    main_vals = expand_range(
        range_cfg["main"]["start"],
        range_cfg["main"]["step"],
        range_cfg["main"]["stop"],
    )

    post_vals = expand_range(
        range_cfg["post"]["start"],
        range_cfg["post"]["step"],
        range_cfg["post"]["stop"],
    )

    l1_max = float(range_cfg["constraints"]["l1_max"])
    prefix = str(range_cfg["naming"]["prefix"])

    if l1_max <= 0:
        raise ValueError("constraints.l1_max must be > 0")

    settings = []
    idx = 0

    for c_pre in pre_vals:
        for c_main in main_vals:
            for c_post in post_vals:
                l1 = abs(c_pre) + abs(c_main) + abs(c_post)

                if l1 <= l1_max + 1e-15:
                    settings.append({
                        "name": "%s%d" % (prefix, idx),
                        "model": "fir_3tap",
                        "enabled": True,
                        "c_pre": float(c_pre),
                        "c_main": float(c_main),
                        "c_post": float(c_post),
                    })
                    idx += 1

    return settings


def build_settings_causal(range_cfg):
    main_vals = expand_range(
        range_cfg["main"]["start"],
        range_cfg["main"]["step"],
        range_cfg["main"]["stop"],
    )

    post1_vals = expand_range(
        range_cfg["post1"]["start"],
        range_cfg["post1"]["step"],
        range_cfg["post1"]["stop"],
    )

    post2_vals = expand_range(
        range_cfg["post2"]["start"],
        range_cfg["post2"]["step"],
        range_cfg["post2"]["stop"],
    )

    l1_max = float(range_cfg["constraints"]["l1_max"])
    prefix = str(range_cfg["naming"]["prefix"])

    if l1_max <= 0:
        raise ValueError("constraints.l1_max must be > 0")

    settings = []
    idx = 0

    for c_main in main_vals:
        for c_post1 in post1_vals:
            for c_post2 in post2_vals:
                l1 = abs(c_main) + abs(c_post1) + abs(c_post2)

                if l1 <= l1_max + 1e-15:
                    settings.append({
                        "name": "%s%d" % (prefix, idx),
                        "model": "fir_3tap_causal",
                        "enabled": True,
                        "c_main": float(c_main),
                        "c_post1": float(c_post1),
                        "c_post2": float(c_post2),
                    })
                    idx += 1

    return settings


def build_settings(range_cfg):
    model = range_cfg["model"]

    if model == "fir_3tap":
        return build_settings_centered(range_cfg)

    if model == "fir_3tap_causal":
        return build_settings_causal(range_cfg)

    raise ValueError("Unsupported model in tx_ranges.json: %s" % model)


def print_summary(settings):
    print("")
    print("Generated %d valid TX settings" % len(settings))
    print("")

    if len(settings) > 0:
        print("First few settings:")
        for s in settings[:5]:
            if s["model"] == "fir_3tap":
                print(
                    "  %s: model=%s pre=%+.4f main=%+.4f post=%+.4f"
                    % (s["name"], s["model"], s["c_pre"], s["c_main"], s["c_post"])
                )
            elif s["model"] == "fir_3tap_causal":
                print(
                    "  %s: model=%s main=%+.4f post1=%+.4f post2=%+.4f"
                    % (s["name"], s["model"], s["c_main"], s["c_post1"], s["c_post2"])
                )
        print("")


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python3 generate_tx_settings.py tx_ranges.json tx_settings.json")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    range_cfg = load_json(in_path)
    settings = build_settings(range_cfg)
    save_json(out_path, settings)
    print_summary(settings)
    print("Saved TX settings to %s" % out_path)


if __name__ == "__main__":
    main()
