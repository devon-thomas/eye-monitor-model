#!/usr/bin/env python3
"""
Interpolate a coarse UI-spaced pulse response into a dense pulse response,
plot coarse vs dense, and optionally save the dense response to a file.

Examples:
  python make_dense_pulse.py --coarse 0,0.04,0.35,0.1,0.01,0 --samples-per-bit 512 --outfile pulse_dense.csv
  python make_dense_pulse.py --coarse-file coarse_pulse.csv --samples-per-bit 512 --method linear --outfile pulse_dense.csv
  python3 make_dense_pulse.py --coarse 0,0.04,0.5,0.05,0.01,0 --samples-per-bit 512 --method pchip --outfile pulse_response_dense_interp.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# === _PCHIP_SLOPES ===
# Compute monotone cubic Hermite slopes for 1D interpolation.
def _pchip_slopes(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)
    if n < 2:
        raise ValueError("Need at least two points for interpolation")

    h = np.diff(x)
    d = np.diff(y) / h

    m = np.zeros(n, dtype=float)

    if n == 2:
        m[0] = d[0]
        m[1] = d[0]
        return m

    for k in range(1, n - 1):
        if d[k - 1] == 0.0 or d[k] == 0.0 or np.sign(d[k - 1]) != np.sign(d[k]):
            m[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    m0 = ((2.0 * h[0] + h[1]) * d[0] - h[0] * d[1]) / (h[0] + h[1])
    if np.sign(m0) != np.sign(d[0]):
        m0 = 0.0
    elif np.sign(d[0]) != np.sign(d[1]) and abs(m0) > 3.0 * abs(d[0]):
        m0 = 3.0 * d[0]
    m[0] = m0

    mn = ((2.0 * h[-1] + h[-2]) * d[-1] - h[-1] * d[-2]) / (h[-1] + h[-2])
    if np.sign(mn) != np.sign(d[-1]):
        mn = 0.0
    elif np.sign(d[-1]) != np.sign(d[-2]) and abs(mn) > 3.0 * abs(d[-1]):
        mn = 3.0 * d[-1]
    m[-1] = mn

    return m


# === INTERPOLATE_PULSE_RESPONSE_UI ===
# Interpolate a coarse UI-spaced pulse response into a dense pulse response.
def interpolate_pulse_response_ui(coarse_y, samples_per_bit, coarse_x_ui=None, method="pchip"):
    coarse_y = np.asarray(coarse_y, dtype=float)

    if coarse_x_ui is None:
        coarse_x_ui = np.arange(len(coarse_y), dtype=float)
    else:
        coarse_x_ui = np.asarray(coarse_x_ui, dtype=float)

    if len(coarse_x_ui) != len(coarse_y):
        raise ValueError("coarse_x_ui and coarse_y must have the same length")
    if len(coarse_y) < 2:
        raise ValueError("Need at least two coarse pulse points")
    if samples_per_bit < 2:
        raise ValueError("samples_per_bit must be >= 2")

    x0 = coarse_x_ui[0]
    x1 = coarse_x_ui[-1]
    n_dense = int(round((x1 - x0) * samples_per_bit)) + 1
    dense_x_ui = np.linspace(x0, x1, n_dense)

    if method == "linear":
        dense_y = np.interp(dense_x_ui, coarse_x_ui, coarse_y)
        return dense_x_ui, dense_y

    if method != "pchip":
        raise ValueError("method must be 'pchip' or 'linear'")

    m = _pchip_slopes(coarse_x_ui, coarse_y)
    dense_y = np.zeros_like(dense_x_ui)

    for k in range(len(coarse_x_ui) - 1):
        xa = coarse_x_ui[k]
        xb = coarse_x_ui[k + 1]
        ya = coarse_y[k]
        yb = coarse_y[k + 1]
        ma = m[k]
        mb = m[k + 1]

        seg = (dense_x_ui >= xa) & (dense_x_ui <= xb if k == len(coarse_x_ui) - 2 else dense_x_ui < xb)
        xseg = dense_x_ui[seg]

        h = xb - xa
        t = (xseg - xa) / h

        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2

        dense_y[seg] = h00 * ya + h10 * h * ma + h01 * yb + h11 * h * mb
    return dense_x_ui, dense_y


# === PLOT_PULSE_RESPONSE_COARSE_VS_DENSE ===
# Plot a coarse UI-spaced pulse response against its dense interpolated version.
def plot_pulse_response_coarse_vs_dense(coarse_y, dense_x_ui, dense_y, coarse_x_ui=None):
    coarse_y = np.asarray(coarse_y, dtype=float)

    if coarse_x_ui is None:
        coarse_x_ui = np.arange(len(coarse_y), dtype=float)
    else:
        coarse_x_ui = np.asarray(coarse_x_ui, dtype=float)

    plt.figure(figsize=(9, 4.5))
    plt.plot(dense_x_ui, dense_y, label="Dense interpolated pulse")
    plt.plot(coarse_x_ui, coarse_y, "o", markersize=7, label="Coarse UI points")
    plt.xlabel("Time [UI]")
    plt.ylabel("Amplitude")
    plt.title("Coarse vs Dense Pulse Response")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


# === _PARSE_FLOAT_LIST ===
# Parse a comma-separated list of floats.
def _parse_float_list(s):
    vals = []
    for item in s.split(","):
        item = item.strip()
        if item == "":
            continue
        vals.append(float(item))
    if len(vals) == 0:
        raise ValueError("Expected at least one numeric value")
    return vals


# === _LOAD_VECTOR_FILE ===
# Load a 1D numeric vector from a text or CSV file.
def _load_vector_file(path):
    arr = np.loadtxt(path, delimiter=",", ndmin=1)
    arr = np.asarray(arr, dtype=float).flatten()
    if arr.size == 0:
        raise ValueError("File contains no numeric data")
    return arr.tolist()


# === MAIN ===
# Parse arguments, interpolate, plot, and save the dense pulse response.
def main():
    parser = argparse.ArgumentParser(description="Create a dense pulse response from coarse UI-spaced values.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--coarse", type=str, help="Comma-separated coarse pulse values, e.g. 0,0.04,0.35,0.1,0.01,0")
    src.add_argument("--coarse-file", type=str, help="Text/CSV file containing coarse pulse values")
    parser.add_argument("--coarse-x-ui", type=str, default=None,
                        help="Optional comma-separated UI positions for coarse values, e.g. 0,1,2,3,4,5")
    parser.add_argument("--samples-per-bit", type=int, required=True, help="Dense samples per UI in output")
    parser.add_argument("--method", choices=["pchip", "linear"], default="pchip", help="Interpolation method")
    parser.add_argument("--outfile", type=str, required=True, help="Output CSV file for dense pulse response")
    parser.add_argument("--plotfile", type=str, default=None, help="Optional image file to save the plot")
    args = parser.parse_args()

    if args.coarse is not None:
        coarse_y = _parse_float_list(args.coarse)
    else:
        coarse_y = _load_vector_file(args.coarse_file)

    coarse_x_ui = None
    if args.coarse_x_ui is not None:
        coarse_x_ui = _parse_float_list(args.coarse_x_ui)

    dense_x_ui, dense_y = interpolate_pulse_response_ui(
        coarse_y=coarse_y,
        samples_per_bit=args.samples_per_bit,
        coarse_x_ui=coarse_x_ui,
        method=args.method,
    )

    dense_y = dense_y * args.samples_per_bit / np.sum(dense_y)
#    coarse_y = coarse_y / np.sum(coarse_y)
    outpath = Path(args.outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(outpath, dense_y, fmt="%.12e")

    plot_pulse_response_coarse_vs_dense(
        coarse_y=coarse_y,
        dense_x_ui=dense_x_ui,
        dense_y=dense_y,
        coarse_x_ui=coarse_x_ui,
    )

    if args.plotfile:
        plotpath = Path(args.plotfile)
        plotpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plotpath, dpi=200)

    print("Saved dense pulse response to:", outpath)
    if args.plotfile:
        print("Saved plot to:", args.plotfile)

    plt.show()


if __name__ == "__main__":
    main()
