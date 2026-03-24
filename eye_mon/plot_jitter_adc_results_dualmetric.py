import csv
import sys
from collections import defaultdict

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def parse_adc_label(val, enabled, adc_label=None):
    if adc_label not in [None, ""]:
        return str(adc_label)
    if str(enabled).lower() in ["false", "0", ""]:
        return "ideal"
    try:
        return str(int(float(val)))
    except Exception:
        return str(val)

def parse_adc_numeric(label):
    if label == "ideal":
        return None
    if label.endswith("b"):
        return int(label[:-1])
    return int(label)

def adc_sort_key(label):
    if label == "ideal":
        return 10**9
    try:
        return int(label)
    except Exception:
        return -1


def as_float(x):
    return float(x)


def group_rows_by_metric(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["metric"]].append(r)
    return out


def build_grids(rows):
    jitter_fs_vals = sorted(set(as_float(r["sigma_j_fs"]) for r in rows))

    present_adc_labels = set(
        parse_adc_label(r.get("adc_bits"), r.get("adc_enabled"), r.get("adc_label"))
        for r in rows
    )

    adc_labels = []
    if "ideal" in present_adc_labels:
        adc_labels.append("ideal")

    numeric_adc_labels = []
    other_adc_labels = []
    
    for label in present_adc_labels:
        if label == "ideal":
            continue
        try:
            numeric_adc_labels.append(parse_adc_numeric(label))
        except Exception:
            other_adc_labels.append(label)
    
    numeric_adc_labels = sorted(set(numeric_adc_labels), reverse=True)
    
    # convert back to labels (preserve "b" format)
    numeric_adc_labels = [f"{b}b" for b in numeric_adc_labels]
    
    other_adc_labels = sorted(set(other_adc_labels), reverse=True)
    
    adc_labels.extend(numeric_adc_labels)
    adc_labels.extend(other_adc_labels)

    match_grid = np.full((len(jitter_fs_vals), len(adc_labels)), np.nan)
    rank_grid = np.full((len(jitter_fs_vals), len(adc_labels)), np.nan)
    metric_grid = np.full((len(jitter_fs_vals), len(adc_labels)), np.nan)

    jitter_index = {v: i for i, v in enumerate(jitter_fs_vals)}
    adc_index = {v: i for i, v in enumerate(adc_labels)}

    for r in rows:
        jf = as_float(r["sigma_j_fs"])
        al = parse_adc_label(r.get("adc_bits"), r.get("adc_enabled"), r.get("adc_label"))
        i = jitter_index[jf]
        j = adc_index[al]

        match_grid[i, j] = 1.0 if str(r["match_oracle"]).lower() == "true" else 0.0

        rank = r["oracle_rank_under_impaired"]
        rank_grid[i, j] = np.nan if rank in ["", "None", None] else float(rank)

        metric_grid[i, j] = float(r["chosen_metric_value"])

    return jitter_fs_vals, adc_labels, match_grid, rank_grid, metric_grid

def draw_heatmap(ax, grid, xlabels, ylabels, title, cbar_label, fmt=None, vmin=None, vmax=None):
    im = ax.imshow(grid, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(["%.1f" % y for y in ylabels])

    ax.set_xlabel("ADC resolution [bits]")
    ax.set_ylabel("Sampling jitter RMS [fs]")
    ax.set_title(title)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if np.isnan(val):
                txt = ""
            elif fmt is None:
                txt = str(val)
            else:
                txt = fmt % val
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)


def draw_min_bits_curve(ax, rows, jitter_fs_vals, adc_labels, title):
    per_jitter = defaultdict(list)
    for r in rows:
        jf = as_float(r["sigma_j_fs"])
        adc_label = parse_adc_label(r.get("adc_bits"), r.get("adc_enabled"), r.get("adc_label"))
        if adc_label == "ideal":
            continue
        match = str(r["match_oracle"]).lower() == "true"
        try:
            bits = int(adc_label)
        except Exception:
            continue
        per_jitter[jf].append((bits, match))

    y_bits = []
    for jf in jitter_fs_vals:
        req = np.nan
        vals = sorted(per_jitter.get(jf, []), key=lambda t: t[0])
        for bits, match in vals:
            if match:
                req = bits
                break
        y_bits.append(req)

    ax.plot(jitter_fs_vals, y_bits, marker="o")
    ax.set_xlabel("Sampling jitter RMS [fs]")
    ax.set_ylabel("Minimum ADC bits for oracle match")
    ax.set_title(title)
    ax.grid(True)


def safe_metric_filename(metric_name):
    return metric_name.replace("/", "_").replace(" ", "_")


def plot_metric_set(metric_name, rows, outdir):
    jitter_fs_vals, adc_labels, match_grid, rank_grid, metric_grid = build_grids(rows)

    # Figure 1: three heatmaps
    fig1, axes = plt.subplots(3, 1, figsize=(8.5, 15))

    draw_heatmap(
        axes[0],
        match_grid,
        adc_labels,
        jitter_fs_vals,
        title="Winner Match Heatmap\n%s" % metric_name,
        cbar_label="Oracle match (1=yes, 0=no)",
        fmt="%.0f",
        vmin=0.0,
        vmax=1.0,
    )

    finite_rank = rank_grid[np.isfinite(rank_grid)]
    rank_vmax = np.max(finite_rank) if finite_rank.size > 0 else 1.0
    draw_heatmap(
        axes[1],
        rank_grid,
        adc_labels,
        jitter_fs_vals,
        title="Oracle Rank Under Impairment\n%s" % metric_name,
        cbar_label="Rank (1 = best)",
        fmt="%.0f",
        vmin=1.0,
        vmax=rank_vmax,
    )

    finite_metric = metric_grid[np.isfinite(metric_grid)]
    metric_vmin = np.min(finite_metric) if finite_metric.size > 0 else 0.0
    metric_vmax = np.max(finite_metric) if finite_metric.size > 0 else 1.0
    draw_heatmap(
        axes[2],
        metric_grid,
        adc_labels,
        jitter_fs_vals,
        title="Chosen Metric Value Heatmap\n%s" % metric_name,
        cbar_label="Chosen metric value",
        fmt="%.3f",
        vmin=metric_vmin,
        vmax=metric_vmax,
    )

    fig1.tight_layout()
    fig1_path = outdir / ("jitter_adc_heatmaps_%s.png" % safe_metric_filename(metric_name))
    fig1.savefig(fig1_path, dpi=200)

    # Figure 2: minimum bits vs jitter
    fig2, ax = plt.subplots(figsize=(8.0, 4.8))
    draw_min_bits_curve(
        ax,
        rows,
        jitter_fs_vals,
        adc_labels,
        title="Minimum ADC Resolution Needed vs Jitter\n%s" % metric_name,
    )
    fig2.tight_layout()
    fig2_path = outdir / ("min_bits_vs_jitter_%s.png" % safe_metric_filename(metric_name))
    fig2.savefig(fig2_path, dpi=200)

    return [str(fig1_path), str(fig2_path)]


def main():
    default_csv = "results/jitter_adc_sweep.csv"
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv
    print("[INFO] Using CSV:", csv_path)

    rows = load_rows(csv_path)
    if len(rows) == 0:
        raise ValueError("CSV contains no rows")

    by_metric = group_rows_by_metric(rows)

    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    saved = []
    for metric_name, metric_rows in by_metric.items():
        saved.extend(plot_metric_set(metric_name, metric_rows, outdir))

    print("Saved:")
    for path in saved:
        print("  %s" % path)
    plt.show()

if __name__ == "__main__":
    main()
