import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import mplcursors



# ================================
# CONFIG / VALIDATION
# ================================

# === LOAD_CONFIG ===
# Load the JSON configuration file into a Python dictionary.
def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


# === VALIDATE_CONFIG ===
# Validate required configuration fields and basic parameter consistency.
def validate_config(cfg):
    required_sections = [
        "data",
        "tx_eq",
        "channel",
        "frontend",
        "sampling",
        "nonlinearity",
        "noise",
        "eye_metrics",
        "output",
    ]
    for section in required_sections:
        if section not in cfg:
            raise ValueError("Missing config section: %s" % section)

    # ----------------------------
    # Data
    # ----------------------------
    tbit = float(cfg["data"]["tbit"])
    n_bits = int(cfg["data"]["n_bits"])
    samples_per_bit = int(cfg["data"]["samples_per_bit"])
    vpp = float(cfg["data"]["waveform"]["vpp"])
    float(cfg["data"]["waveform"]["vcm"])
    int(cfg["data"]["prbs_seed"])

    if tbit <= 0:
        raise ValueError("data.tbit must be > 0")
    if n_bits < 10:
        raise ValueError("data.n_bits must be >= 10")
    if samples_per_bit < 16:
        raise ValueError("data.samples_per_bit should be >= 16")
    if vpp <= 0:
        raise ValueError("data.waveform.vpp must be > 0")

    # ----------------------------
    # Input waveform (optional)
    # ----------------------------
    input_waveform_cfg = cfg.get("input_waveform", {"model": "nrz_channel_fir"})
    input_waveform_model = input_waveform_cfg.get("model", "nrz_channel_fir")

    if input_waveform_model not in ["nrz_channel_fir", "pulse_response_bits", "pulse_response_dense"]:
        raise ValueError(
            "input_waveform.model must be 'nrz_channel_fir', 'pulse_response_bits', or 'pulse_response_dense'"
        )

    if input_waveform_model == "pulse_response_bits":
        taps = input_waveform_cfg.get("taps", None)
        if not isinstance(taps, list) or len(taps) < 1:
            raise ValueError("input_waveform.taps must be a non-empty list in pulse_response_bits mode")
        for x in taps:
            float(x)
        if bool(cfg["channel"]["enabled"]):
            raise ValueError(
                "input_waveform.model='pulse_response_bits' requires channel.enabled=false to avoid double-counting ISI"
            )

    if input_waveform_model == "pulse_response_dense":
        pulse_file = input_waveform_cfg.get("file", "")
        if not isinstance(pulse_file, str) or pulse_file.strip() == "":
            raise ValueError("input_waveform.file must be a non-empty string in pulse_response_dense mode")
        if bool(cfg["channel"]["enabled"]):
            raise ValueError(
                "input_waveform.model='pulse_response_dense' requires channel.enabled=false to avoid double-counting ISI"
            )

    # ----------------------------
    # TX EQ
    # ----------------------------
    tx_eq_enabled = bool(cfg["tx_eq"]["enabled"])
    tx_model = cfg["tx_eq"]["model"]

    if tx_model not in ["fir_3tap", "fir_3tap_causal"]:
        raise ValueError("tx_eq.model must be 'fir_3tap' or 'fir_3tap_causal'")

    if tx_eq_enabled:
        if tx_model == "fir_3tap":
            c_pre = float(cfg["tx_eq"]["fir_3tap"]["c_pre"])
            c_main = float(cfg["tx_eq"]["fir_3tap"]["c_main"])
            c_post = float(cfg["tx_eq"]["fir_3tap"]["c_post"])
            tap_l1 = abs(c_pre) + abs(c_main) + abs(c_post)

        else:
            c_main = float(cfg["tx_eq"]["fir_3tap_causal"]["c_main"])
            c_post1 = float(cfg["tx_eq"]["fir_3tap_causal"]["c_post1"])
            c_post2 = float(cfg["tx_eq"]["fir_3tap_causal"]["c_post2"])
            tap_l1 = abs(c_main) + abs(c_post1) + abs(c_post2)

        if tap_l1 > 1.0 + 1e-15:
            raise ValueError(
                "TX EQ constraint violated: sum of absolute tap values = %.6f > 1.0"
                % tap_l1
            )

    # ----------------------------
    # Channel
    # ----------------------------
    channel_enabled = bool(cfg["channel"]["enabled"])
    channel_model = cfg["channel"]["model"]

    if channel_model not in ["fir_ui", "one_pole"]:
        raise ValueError("channel.model must be 'fir_ui' or 'one_pole'")

    if channel_enabled:
        if channel_model == "fir_ui":
            taps = cfg["channel"]["fir_ui"]["taps"]
            if not isinstance(taps, list) or len(taps) < 1:
                raise ValueError("channel.fir_ui.taps must be a non-empty list")
            for x in taps:
                float(x)
        else:
            fp_hz = float(cfg["channel"]["one_pole"]["fp_hz"])
            float(cfg["channel"]["one_pole"]["gain"])
            if fp_hz <= 0:
                raise ValueError("channel.one_pole.fp_hz must be > 0")

    # ----------------------------
    # Front-end
    # ----------------------------
    frontend_enabled = bool(cfg["frontend"]["enabled"])
    fe_model = cfg["frontend"]["model"]

    if fe_model not in ["one_pole", "measured_tf"]:
        raise ValueError("frontend.model must be 'one_pole' or 'measured_tf'")

    if frontend_enabled:
        if fe_model == "one_pole":
            fp_hz = float(cfg["frontend"]["one_pole"]["fp_hz"])
            float(cfg["frontend"]["one_pole"]["gain"])
            if fp_hz <= 0:
                raise ValueError("frontend.one_pole.fp_hz must be > 0")
        else:
            mtf_cfg = cfg["frontend"]["measured_tf"]
            if not bool(mtf_cfg["enabled"]):
                raise ValueError(
                    "frontend.model is 'measured_tf' but frontend.measured_tf.enabled is false"
                )
            mtf_file = mtf_cfg["file"]
            if not isinstance(mtf_file, str) or mtf_file.strip() == "":
                raise ValueError(
                    "frontend.measured_tf.file must be a non-empty string when measured_tf is enabled"
                )

    # ----------------------------
    # Sampling
    # ----------------------------
    sigma_j = float(cfg["sampling"]["sigma_j"])
    n_phases = int(cfg["sampling"]["n_phases"])
    n_ui_collect = int(cfg["sampling"]["n_ui_collect"])
    skip_initial_ui = int(cfg["sampling"]["skip_initial_ui"])

    if sigma_j < 0:
        raise ValueError("sampling.sigma_j must be >= 0")
    if n_phases < 4:
        raise ValueError("sampling.n_phases must be >= 4")
    if n_ui_collect < 1:
        raise ValueError("sampling.n_ui_collect must be >= 1")
    if n_bits < skip_initial_ui + n_ui_collect:
        raise ValueError(
            "data.n_bits (%d) must be >= sampling.skip_initial_ui (%d) + sampling.n_ui_collect (%d)"
            % (n_bits, skip_initial_ui, n_ui_collect)
        )

    # ----------------------------
    # Nonlinearity
    # ----------------------------
    nl_enabled = bool(cfg["nonlinearity"]["enabled"])
    nl_model = cfg["nonlinearity"]["model"]

    if nl_model not in ["polynomial"]:
        raise ValueError("nonlinearity.model must be 'polynomial'")

    if nl_enabled:
        float(cfg["nonlinearity"]["polynomial"]["a1"])
        float(cfg["nonlinearity"]["polynomial"]["a2"])
        float(cfg["nonlinearity"]["polynomial"]["a3"])

    # ----------------------------
    # Noise
    # ----------------------------
    noise_enabled = bool(cfg["noise"]["enabled"])
    sigma_v = float(cfg["noise"]["sigma_v"])

    if noise_enabled and sigma_v < 0:
        raise ValueError("noise.sigma_v must be >= 0 when noise is enabled")

    # ----------------------------
    # Eye metrics
    # ----------------------------
    k_sigma = float(cfg["eye_metrics"]["k_sigma"])
    eye_threshold_mode = cfg["eye_metrics"]["eye_width"]["mode"]
    eye_threshold_value = float(cfg["eye_metrics"]["eye_width"]["value"])

    if k_sigma < 0:
        raise ValueError("eye_metrics.k_sigma must be >= 0")
    if eye_threshold_mode not in ["safe_positive", "absolute"]:
        raise ValueError("eye_metrics.eye_width.mode must be 'safe_positive' or 'absolute'")
    if eye_threshold_mode == "absolute" and eye_threshold_value < 0:
        raise ValueError("eye_metrics.eye_width.value must be >= 0 for absolute mode")

    # ----------------------------
    # Output
    # ----------------------------
    output_enabled = bool(cfg["output"]["enabled"])
    bool(cfg["output"]["show_plots"])
    bool(cfg["output"]["save_plots"])
    str(cfg["output"]["save_prefix"])

    if output_enabled:
        waveform_ui_start = float(cfg["output"]["waveform_plot"]["ui_start"])
        waveform_ui_stop = float(cfg["output"]["waveform_plot"]["ui_stop"])
        if waveform_ui_stop <= waveform_ui_start:
            raise ValueError("output.waveform_plot.ui_stop must be > ui_start")



# ================================
# DATA / WAVEFORM GENERATION
# ================================

# === PRBS7 ===
# Generate a PRBS7 bit sequence mapped to +/-1 symbols.
def prbs7(n_bits, seed=0x5A):
    state = seed & 0x7F
    if state == 0:
        state = 0x5A

    bits = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        newbit = ((state >> 6) ^ (state >> 5)) & 0x1
        outbit = state & 0x1
        bits[i] = outbit
        state = ((state << 1) & 0x7E) | newbit

    return 2 * bits - 1


# === APPLY_TX_EQ ===
# Apply UI-spaced TX equalization in symbol domain using configured FIR taps.
def apply_tx_eq(bits, cfg_tx_eq):
    """
    Supported TX models:
      fir_3tap:
        eq[n] = c_pre * b[n+1] + c_main * b[n] + c_post * b[n-1]

      fir_3tap_causal:
        eq[n] = c_main * b[n] + c_post1 * b[n-1] + c_post2 * b[n-2]
    """
    if not bool(cfg_tx_eq["enabled"]):
        return bits.astype(float)

    model = cfg_tx_eq["model"]
    b = bits.astype(float)

    if model == "fir_3tap":
        c_pre = float(cfg_tx_eq["fir_3tap"]["c_pre"])
        c_main = float(cfg_tx_eq["fir_3tap"]["c_main"])
        c_post = float(cfg_tx_eq["fir_3tap"]["c_post"])

        tap_l1 = abs(c_pre) + abs(c_main) + abs(c_post)
        if tap_l1 > 1.0 + 1e-15:
            raise ValueError(
                "TX EQ constraint violated: abs(c_pre)+abs(c_main)+abs(c_post)=%.6f > 1.0"
                % tap_l1
            )

        b_next = np.zeros_like(b)
        b_prev = np.zeros_like(b)

        b_next[:-1] = b[1:]
        b_prev[1:] = b[:-1]

        return c_pre * b_next + c_main * b + c_post * b_prev

    elif model == "fir_3tap_causal":
        c_main = float(cfg_tx_eq["fir_3tap_causal"]["c_main"])
        c_post1 = float(cfg_tx_eq["fir_3tap_causal"]["c_post1"])
        c_post2 = float(cfg_tx_eq["fir_3tap_causal"]["c_post2"])

        tap_l1 = abs(c_main) + abs(c_post1) + abs(c_post2)
        if tap_l1 > 1.0 + 1e-15:
            raise ValueError(
                "TX EQ constraint violated: abs(c_main)+abs(c_post1)+abs(c_post2)=%.6f > 1.0"
                % tap_l1
            )

        b_prev1 = np.zeros_like(b)
        b_prev2 = np.zeros_like(b)

        b_prev1[1:] = b[:-1]
        b_prev2[2:] = b[:-2]

        return c_main * b + c_post1 * b_prev1 + c_post2 * b_prev2



# === APPLY_BIT_PULSE_RESPONSE ===
# Convolve the TX-equalized bit stream with an M-bit pulse response to create symbol amplitudes.
def apply_bit_pulse_response(symbols, pulse_taps):
    symbols = np.asarray(symbols, dtype=float)
    pulse_taps = np.asarray(pulse_taps, dtype=float)
    return np.convolve(symbols, pulse_taps, mode="same")


# === LOAD_DENSE_PULSE_RESPONSE ===
# Load a dense pulse response from a text or CSV file with one numeric sample per row.
def load_dense_pulse_response(path):
    pulse = np.loadtxt(path, delimiter=",", ndmin=1)
    pulse = np.asarray(pulse, dtype=float).flatten()
    if pulse.size < 2:
        raise ValueError("Dense pulse-response file must contain at least two numeric samples")
    return pulse


# === APPLY_DENSE_PULSE_SUPERPOSITION ===
# Build the dense waveform by UI-spaced superposition of a dense bit pulse response.
def apply_dense_pulse_superposition(symbols, pulse_dense, samples_per_bit):
    symbols = np.asarray(symbols, dtype=float)
    pulse_dense = np.asarray(pulse_dense, dtype=float).flatten()

    n_out = len(symbols) * samples_per_bit + len(pulse_dense) - samples_per_bit
    y = np.zeros(n_out, dtype=float)

    for k, a in enumerate(symbols):
        i0 = k * samples_per_bit
        i1 = i0 + len(pulse_dense)
        y[i0:i1] += a * pulse_dense

    return y


# === BUILD_NRZ_WAVEFORM ===
# Convert symbol sequence into a dense-time NRZ waveform.
def build_nrz_waveform(symbols, t, tbit, vpp, vcm):
    idx = np.floor(t / tbit).astype(int)
    idx = np.clip(idx, 0, len(symbols) - 1)
    sym = symbols[idx].astype(float)
    return vcm + 0.5 * vpp * sym



# ================================
# CHANNEL / FRONTEND / NONLINEARITY
# ================================

# === ONE_POLE_LOWPASS_FFT ===
# Apply a one-pole low-pass response in frequency domain to a waveform.
def one_pole_lowpass_fft(x, dt, fp_hz, gain=1.0):
    n = len(x)
    n_pad = 2 * n

    x_pad = np.zeros(n_pad, dtype=float)
    x_pad[:n] = x

    f = np.fft.fftfreq(n_pad, d=dt)
    H = gain / (1.0 + 1j * (f / fp_hz))
    X = np.fft.fft(x_pad)
    y_pad = np.fft.ifft(X * H)

    return np.real(y_pad[:n])


# === FIR_UI_CHANNEL_WAVEFORM ===
# Apply a UI-spaced FIR directly to the dense-time waveform.
def fir_ui_channel_waveform(x, samples_per_bit, taps):
    taps = np.asarray(taps, dtype=float)
    y = np.zeros_like(x, dtype=float)

    for k, hk in enumerate(taps):
        delay = k * samples_per_bit
        if delay == 0:
            y += hk * x
        elif delay < len(x):
            y[delay:] += hk * x[:-delay]

    return y


# === APPLY_CHANNEL ===
# Apply the configured channel model to the waveform.
def apply_channel(x, dt, samples_per_bit, cfg_channel):
    if not bool(cfg_channel["enabled"]):
        return x

    model = cfg_channel["model"]

    if model == "fir_ui":
        taps = cfg_channel["fir_ui"]["taps"]
        return fir_ui_channel_waveform(x, samples_per_bit, taps)

    elif model == "one_pole":
        fp_hz = float(cfg_channel["one_pole"]["fp_hz"])
        gain = float(cfg_channel["one_pole"]["gain"])
        return one_pole_lowpass_fft(x, dt, fp_hz, gain=gain)

    raise ValueError("Unsupported channel.model: %s" % model)


# === APPLY_FRONTEND ===
# Apply the configured front-end transfer function to the waveform.
def apply_frontend(x, dt, cfg_frontend):
    if not bool(cfg_frontend["enabled"]):
        return x

    model = cfg_frontend["model"]

    if model == "one_pole":
        fp_hz = float(cfg_frontend["one_pole"]["fp_hz"])
        gain = float(cfg_frontend["one_pole"]["gain"])
        return one_pole_lowpass_fft(x, dt, fp_hz, gain=gain)

    elif model == "measured_tf":
        mtf_cfg = cfg_frontend["measured_tf"]
        if not bool(mtf_cfg["enabled"]):
            raise ValueError(
                "frontend.model is 'measured_tf' but frontend.measured_tf.enabled is false"
            )
        raise NotImplementedError("measured_tf path is configured but not implemented yet")

    raise ValueError("Unsupported frontend.model: %s" % model)


# === APPLY_NONLINEARITY ===
# Apply the configured static polynomial nonlinearity to the waveform.
def apply_nonlinearity(x, cfg_nl):
    if not bool(cfg_nl["enabled"]):
        return x

    model = cfg_nl["model"]
    if model == "polynomial":
        a1 = float(cfg_nl["polynomial"]["a1"])
        a2 = float(cfg_nl["polynomial"]["a2"])
        a3 = float(cfg_nl["polynomial"]["a3"])
        return a1 * x + a2 * x ** 2 + a3 * x ** 3

    raise ValueError("Unsupported nonlinearity.model: %s" % model)



# ================================
# SAMPLING / METRICS
# ================================

# === SAMPLE_EYE ===
# Collect eye-monitor samples across commanded phases with optional jitter and noise.
def sample_eye(t, y, tbit, n_phases, n_ui_collect, sigma_j, sigma_v, skip_initial_ui):
    phase_offsets = np.arange(n_phases, dtype=float) * tbit / float(n_phases)

    sample_times = []
    sample_values = []
    sample_phase_cmd_ui = []

    k0 = int(skip_initial_ui)

    for phi in phase_offsets:
        tk_nom = (np.arange(n_ui_collect) + k0) * tbit + phi
        tk = tk_nom + np.random.randn(n_ui_collect) * sigma_j

        mask = (tk >= t[0]) & (tk <= t[-1])
        tk = tk[mask]

        vk = np.interp(tk, t, y)

        if sigma_v > 0.0:
            vk = vk + np.random.randn(len(vk)) * sigma_v

        phi_cmd_ui = phi / tbit
        phase_cmd = np.full(len(vk), phi_cmd_ui)

        sample_times.append(tk)
        sample_values.append(vk)
        sample_phase_cmd_ui.append(phase_cmd)

    return {
        "phase_offsets": phase_offsets,
        "sample_times": sample_times,
        "sample_values": sample_values,
        "sample_phase_cmd_ui": sample_phase_cmd_ui,
    }


# === _SAFE_MEAN ===
# Return the mean of an array or NaN if the array is empty.
def _safe_mean(x):
    if len(x) == 0:
        return np.nan
    return np.mean(x)


# === _SAFE_STD ===
# Return the standard deviation of an array or NaN if the array is empty.
def _safe_std(x):
    if len(x) == 0:
        return np.nan
    return np.std(x)


# === COMPUTE_EYE_METRICS ===
# Compute rail statistics, eye opening, width, and area metrics from samples.
def compute_eye_metrics(results, tbit, cfg_eye_metrics, vcm):
    k_sigma = float(cfg_eye_metrics["k_sigma"])
    phase_cmd_ui = results["phase_offsets"] / tbit
    sample_values = results["sample_values"]

    mu_upper = np.zeros(len(sample_values), dtype=float)
    mu_lower = np.zeros(len(sample_values), dtype=float)
    std_upper = np.zeros(len(sample_values), dtype=float)
    std_lower = np.zeros(len(sample_values), dtype=float)
    n_upper = np.zeros(len(sample_values), dtype=int)
    n_lower = np.zeros(len(sample_values), dtype=int)

    for i, vals in enumerate(sample_values):
        upper = vals[vals >= vcm]
        lower = vals[vals < vcm]

        n_upper[i] = len(upper)
        n_lower[i] = len(lower)

        mu_upper[i] = _safe_mean(upper)
        mu_lower[i] = _safe_mean(lower)
        std_upper[i] = _safe_std(upper)
        std_lower[i] = _safe_std(lower)

    opening_raw = mu_upper - mu_lower
    opening_safe = (mu_upper - k_sigma * std_upper) - (mu_lower + k_sigma * std_lower)

    if np.all(np.isnan(opening_safe)):
        best_idx = int(np.nanargmax(opening_raw))
    else:
        best_idx = int(np.nanargmax(opening_safe))

    best_phase_ui = phase_cmd_ui[best_idx]

    eye_height_center_raw = opening_raw[best_idx]
    eye_height_center_safe = opening_safe[best_idx]
    eye_height_max_raw = np.nanmax(opening_raw)
    eye_height_max_safe = np.nanmax(opening_safe)

    dphi_ui = 1.0 / len(phase_cmd_ui)
    opening_raw_clipped = np.maximum(opening_raw, 0.0)
    opening_safe_clipped = np.maximum(opening_safe, 0.0)
    eye_area_raw = np.sum(opening_raw_clipped) * dphi_ui
    eye_area_safe = np.sum(opening_safe_clipped) * dphi_ui

    width_mode = cfg_eye_metrics["eye_width"]["mode"]
    width_value = float(cfg_eye_metrics["eye_width"]["value"])

    if width_mode == "safe_positive":
        open_mask = opening_safe > 0.0
    elif width_mode == "absolute":
        open_mask = opening_safe > width_value
    else:
        raise ValueError("Unsupported eye width mode: %s" % width_mode)

    left_idx = best_idx
    while left_idx > 0 and open_mask[left_idx - 1]:
        left_idx -= 1

    right_idx = best_idx
    while right_idx < len(open_mask) - 1 and open_mask[right_idx + 1]:
        right_idx += 1

    left_edge_ui = phase_cmd_ui[left_idx]
    right_edge_ui = phase_cmd_ui[right_idx]
    eye_width_ui = right_edge_ui - left_edge_ui
    return {
        "phase_cmd_ui": phase_cmd_ui,
        "mu_upper": mu_upper,
        "mu_lower": mu_lower,
        "std_upper": std_upper,
        "std_lower": std_lower,
        "n_upper": n_upper,
        "n_lower": n_lower,
        "opening_raw": opening_raw,
        "opening_safe": opening_safe,
        "best_idx": best_idx,
        "best_phase_ui": best_phase_ui,
        "best_phase_height_ui": best_phase_ui,
        "eye_height_center_raw": eye_height_center_raw,
        "eye_height_center_safe": eye_height_center_safe,
        "eye_height_max_raw": eye_height_max_raw,
        "eye_height_max_safe": eye_height_max_safe,
        "eye_area_raw": eye_area_raw,
        "eye_area_safe": eye_area_safe,
        "left_edge_ui": left_edge_ui,
        "right_edge_ui": right_edge_ui,
        "eye_width_ui": eye_width_ui,
    }


# === BUILD_EYE_PLOT_COORDS ===
# Build eye-diagram x-coordinates for sampled reconstruction plots.
def build_eye_plot_coords(results, best_phase_ui):
    eye_x_ui = []

    for phase_cmd, vals in zip(results["sample_phase_cmd_ui"], results["sample_values"]):
        x1 = np.mod(phase_cmd - best_phase_ui + 1.0, 2.0)
        x2 = np.mod(x1 + 1.0, 2.0)
        eye_x_ui.append((x1, x2, vals))

    return eye_x_ui


# === PRINT_EYE_SUMMARY ===
# Print the key eye metrics in a compact console summary.
def print_eye_summary(metrics, tbit):
    print("")
    print("=== Eye Metrics Summary ===")
    print("Best phase (commanded):       %.6f UI  (%.3f ps)" %
          (metrics["best_phase_ui"], metrics["best_phase_ui"] * tbit * 1e12))
    print("Eye height @ best phase raw:  %.6f V" % metrics["eye_height_center_raw"])
    print("Eye height @ best phase safe: %.6f V" % metrics["eye_height_center_safe"])
    print("Eye area raw:                 %.6f V*UI" % metrics["eye_area_raw"])
    print("Eye area safe:                %.6f V*UI" % metrics["eye_area_safe"])
    print("Max eye height raw:           %.6f V" % metrics["eye_height_max_raw"])
    print("Max eye height safe:          %.6f V" % metrics["eye_height_max_safe"])
    print("Left eye edge:                %.6f UI  (%.3f ps)" %
          (metrics["left_edge_ui"], metrics["left_edge_ui"] * tbit * 1e12))
    print("Right eye edge:               %.6f UI  (%.3f ps)" %
          (metrics["right_edge_ui"], metrics["right_edge_ui"] * tbit * 1e12))
    print("Eye width:                    %.6f UI  (%.3f ps)" %
          (metrics["eye_width_ui"], metrics["eye_width_ui"] * tbit * 1e12))
    print("")

    # -- main plot --

# ================================
# PLOTTING / TOP LEVEL
# ================================



# === _SAVE_FIGURE_IF_REQUESTED ===
# Save the current matplotlib figure when plot saving is enabled.
def _save_figure_if_requested(save_plots, save_prefix, suffix):
    if save_plots:
        plt.savefig(save_prefix + suffix, dpi=200)


# === _PLOT_ACTUAL_WAVEFORM_EYE ===
# Overlay limited 2-UI waveform eye segments centered at the best phase.
def _plot_actual_waveform_eye(t, y, tbit, phi_center, max_eye_segments):
    t_ui = t / tbit
    dt_ui = t_ui[1] - t_ui[0]
    samples_per_ui = int(round(1.0 / dt_ui))
    seg_len = 2 * samples_per_ui
    n_segments_total = max(0, len(t_ui) // samples_per_ui - 1)
    n_segments_plot = min(n_segments_total, max_eye_segments)

    waveform_label_added = False
    for k in range(n_segments_plot):
        i0 = k * samples_per_ui
        i1 = min(i0 + seg_len, len(t_ui))
        if i1 - i0 < 2:
            continue

        x_seg = np.mod(t_ui[i0:i1] - phi_center + 1.0, 2.0)
        if not waveform_label_added:
            plt.plot(
                x_seg,
                y[i0:i1],
                alpha=0.08,
                linewidth=0.8,
                label="Actual waveform eye"
            )
            waveform_label_added = True
        else:
            plt.plot(
                x_seg,
                y[i0:i1],
                alpha=0.08,
                linewidth=0.8
            )


# === _PLOT_COMMANDED_PHASE_SAMPLE_EYE ===
# Overlay sampled-eye points using commanded phase across a 2-UI frame.
def _plot_commanded_phase_sample_eye(results, phi_center):
    sample_label_added = False
    for phase_cmd, vals in zip(results["sample_phase_cmd_ui"], results["sample_values"]):
        x1 = np.mod(phase_cmd - phi_center + 1.0, 2.0)
        x2 = np.mod(x1 + 1.0, 2.0)

        if not sample_label_added:
            plt.scatter(
                x1,
                vals,
                s=4,
                alpha=0.22,
                label="Sampled eye reconstruction"
            )
            plt.scatter(
                x2,
                vals,
                s=4,
                alpha=0.22
            )
            sample_label_added = True
        else:
            plt.scatter(
                x1,
                vals,
                s=4,
                alpha=0.22
            )
            plt.scatter(
                x2,
                vals,
                s=4,
                alpha=0.22
            )


# === _PLOT_NL_DEBUG ===
# Plot pre/post nonlinearity waveforms and their error over a short time window.


# === _GET_TX_FIR_UI_TAPS ===
# Return TX equalizer taps as a UI-spaced FIR list for frequency-response plotting.
def _get_tx_fir_ui_taps(cfg_tx_eq):
    if not bool(cfg_tx_eq["enabled"]):
        return [1.0]

    model = cfg_tx_eq["model"]

    if model == "fir_3tap":
        c_pre = float(cfg_tx_eq["fir_3tap"]["c_pre"])
        c_main = float(cfg_tx_eq["fir_3tap"]["c_main"])
        c_post = float(cfg_tx_eq["fir_3tap"]["c_post"])
        return [c_main, c_post, c_pre]

    if model == "fir_3tap_causal":
        c_main = float(cfg_tx_eq["fir_3tap_causal"]["c_main"])
        c_post1 = float(cfg_tx_eq["fir_3tap_causal"]["c_post1"])
        c_post2 = float(cfg_tx_eq["fir_3tap_causal"]["c_post2"])
        return [c_main, c_post1, c_post2]

    raise ValueError("Unsupported tx_eq.model: %s" % model)


# === _EVAL_FIR_RESPONSE_UI ===
# Evaluate a UI-spaced FIR response on a linear frequency grid from 0 to 1/tbit.
def _eval_fir_response_ui(taps, f_hz, tbit):
    taps = np.asarray(taps, dtype=float)
    w = 2.0 * np.pi * f_hz * tbit
    resp = np.zeros_like(f_hz, dtype=complex)

    for k, hk in enumerate(taps):
        resp += hk * np.exp(-1j * w * k)

    return resp




# === _GET_INPUT_WAVEFORM_RESPONSE ===
# Return the selected input-waveform response on the requested frequency grid.
def _get_input_waveform_response(cfg, f_hz, tbit):
    input_waveform_cfg = cfg.get("input_waveform", {"model": "nrz_channel_fir"})
    model = input_waveform_cfg.get("model", "nrz_channel_fir")

    if model == "nrz_channel_fir":
        return _get_channel_response(cfg["channel"], f_hz, tbit), "H(f) Channel"

    if model == "pulse_response_bits":
        taps = [float(x) for x in input_waveform_cfg["taps"]]
        return _eval_fir_response_ui(taps, f_hz, tbit), "P(f) Pulse response (UI)"

#    if model == "pulse_response_dense":
#        pulse_dense = load_dense_pulse_response(input_waveform_cfg["file"])
#        dt = tbit / float(int(cfg["data"]["samples_per_bit"]))
#        resp = np.zeros_like(f_hz, dtype=complex)
#        for k, hk in enumerate(pulse_dense):
#            resp += hk * np.exp(-1j * 2.0 * np.pi * f_hz * (k * dt))
#        resp = resp / samples_per_bit
    if model == "pulse_response_dense":
        pulse_dense = load_dense_pulse_response(input_waveform_cfg["file"])
        samples_per_bit = float(int(cfg["data"]["samples_per_bit"]))
        dt = tbit / samples_per_bit
        resp = np.zeros_like(f_hz, dtype=complex)
        for k, hk in enumerate(pulse_dense):
            resp += hk * np.exp(-1j * 2.0 * np.pi * f_hz * (k * dt))
         
        resp =  resp / samples_per_bit
        return resp, "P(f) Pulse response (dense)"

    raise ValueError("Unsupported input_waveform.model: %s" % model)


# === _GET_CHANNEL_RESPONSE ===
# Return the channel frequency response on the requested frequency grid.
def _get_channel_response(cfg_channel, f_hz, tbit):
    if not bool(cfg_channel["enabled"]):
        return np.ones_like(f_hz, dtype=complex)

    model = cfg_channel["model"]

    if model == "fir_ui":
        taps = [float(x) for x in cfg_channel["fir_ui"]["taps"]]
        return _eval_fir_response_ui(taps, f_hz, tbit)

    if model == "one_pole":
        fp_hz = float(cfg_channel["one_pole"]["fp_hz"])
        gain = float(cfg_channel["one_pole"]["gain"])
        return gain / (1.0 + 1j * (f_hz / fp_hz))

    raise ValueError("Unsupported channel.model: %s" % model)


# === _GET_FRONTEND_RESPONSE ===
# Return the front-end frequency response on the requested frequency grid.
def _get_frontend_response(cfg_frontend, f_hz):
    if not bool(cfg_frontend["enabled"]):
        return np.ones_like(f_hz, dtype=complex)

    model = cfg_frontend["model"]

    if model == "one_pole":
        fp_hz = float(cfg_frontend["one_pole"]["fp_hz"])
        gain = float(cfg_frontend["one_pole"]["gain"])
        return gain / (1.0 + 1j * (f_hz / fp_hz))

    if model == "measured_tf":
        raise NotImplementedError("Frequency plot for frontend.measured_tf is not implemented yet")

    raise ValueError("Unsupported frontend.model: %s" % model)


# === _PLOT_FREQUENCY_RESPONSE ===
# Plot TX, channel, frontend, and combined responses on a linear frequency axis in dB.
def _plot_frequency_response(cfg):
    if not bool(cfg.get("debug", {}).get("plot_frequency_response", False)):
        return

    tbit = float(cfg["data"]["tbit"])
    f_max = 1.0 / tbit
    n_points = int(cfg.get("debug", {}).get("frequency_plot_points", 1000))
    f_hz = np.linspace(0.0, f_max, n_points)

    tx_taps = _get_tx_fir_ui_taps(cfg["tx_eq"])
    G = _eval_fir_response_ui(tx_taps, f_hz, tbit)
    W, w_label = _get_input_waveform_response(cfg, f_hz, tbit)
    Q = _get_frontend_response(cfg["frontend"], f_hz)
    GWQ = G * W * Q

    eps = 1e-15
    plt.figure(figsize=(9, 5))
    plt.plot(f_hz, 20.0 * np.log10(np.maximum(np.abs(G), eps)), label="G(f) TX")
    plt.plot(f_hz, 20.0 * np.log10(np.maximum(np.abs(W), eps)), label=w_label)
    plt.plot(f_hz, 20.0 * np.log10(np.maximum(np.abs(Q), eps)), label="Q(f) Front-end")
    plt.plot(f_hz, 20.0 * np.log10(np.maximum(np.abs(GWQ), eps)), label="Total")
    plt.xlim([0.0, f_max])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.title("Frequency Response of TX / Waveform / Front-end")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
def _plot_nl_debug(sim, cfg):
    t = sim["t"]
    y_before = sim["y_before_nl"]
    y_after = sim["y_after_nl"]
    err = y_after - y_before

    tbit = float(cfg["data"]["tbit"])
    mask = t < (5 * tbit)

    plt.figure()
    plt.plot(t[mask], y_before[mask], label="Before NL", alpha=0.6)
    plt.plot(t[mask], y_after[mask], label="After NL", alpha=0.6)
    plt.plot(t[mask], err[mask], label="NL Error", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Nonlinearity Debug (Zoomed)")
    plt.legend()
    plt.grid(True)


# === PLOT_RESULTS ===
# Plot waveform, reconstructed eye, and eye-metric diagnostic figures.
def plot_results(sim, cfg):
    output_cfg = cfg["output"]
    if not bool(output_cfg["enabled"]):
        return

    t = sim["t"]
    x_tx = sim["x_tx"]
    x_ch = sim["x_ch"]
    y = sim["y"]
    tbit = sim["tbit"]
    vcm = sim["vcm"]
    results = sim["results"]
    metrics = sim["metrics"]
    phase_cmd_ui = metrics["phase_cmd_ui"]

    show_plots = bool(output_cfg["show_plots"])
    save_plots = bool(output_cfg["save_plots"])
    save_prefix = output_cfg["save_prefix"]

    waveform_ui_start = float(output_cfg["waveform_plot"]["ui_start"])
    waveform_ui_stop = float(output_cfg["waveform_plot"]["ui_stop"])

    plt.figure(figsize=(10, 4.5))
    waveform_mask = (t >= waveform_ui_start * tbit) & (t <= waveform_ui_stop * tbit)
    plt.plot(t[waveform_mask] / tbit, x_tx[waveform_mask], label="TX waveform")
    plt.plot(t[waveform_mask] / tbit, x_ch[waveform_mask], label="After channel")
    plt.plot(t[waveform_mask] / tbit, y[waveform_mask], label="Front-end output")
    plt.axhline(vcm, linestyle="--", linewidth=1.0, alpha=0.8, label="VCM")
    plt.xlabel("Time [UI]")
    plt.ylabel("Amplitude [V]")
    plt.title("Waveform Segment")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    _save_figure_if_requested(save_plots, save_prefix, "_waveform_segment.png")

    plt.figure(figsize=(8, 5))
    phi_center = metrics["best_phase_ui"]
    max_eye_segments = int(cfg.get("debug", {}).get("max_eye_segments", 200))
    _plot_actual_waveform_eye(t, y, tbit, phi_center, max_eye_segments)
    _plot_commanded_phase_sample_eye(results, phi_center)
    plt.axvline(1.0, linestyle="--", linewidth=1.0, alpha=0.8, label="Best phase center")
    plt.axhline(vcm, linestyle="--", linewidth=1.0, alpha=0.8, label="VCM")
    plt.xlim([0.0, 2.0])
    plt.xlabel("Time [UI]")
    plt.ylabel("Voltage [V]")
    plt.title("Actual Waveform Eye + Reconstructed Sample Eye")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    _save_figure_if_requested(save_plots, save_prefix, "_eye_scatter.png")

    plt.figure(figsize=(9, 5))
    plt.plot(phase_cmd_ui, metrics["mu_upper"], marker="o", label="Upper mean")
    plt.plot(phase_cmd_ui, metrics["mu_lower"], marker="o", label="Lower mean")
    plt.plot(phase_cmd_ui, metrics["opening_raw"], marker="o", label="Opening raw")
    plt.plot(phase_cmd_ui, metrics["opening_safe"], marker="o", label="Opening safe")
    plt.fill_between(
        phase_cmd_ui,
        0.0,
        np.maximum(metrics["opening_safe"], 0.0),
        alpha=0.2,
        label="Safe area"
    )
    plt.axvline(metrics["best_phase_ui"], linestyle="--", linewidth=1.0, alpha=0.8, label="Best phase")
    plt.axvline(metrics["left_edge_ui"], linestyle=":", linewidth=1.0, alpha=0.8, label="Left edge")
    plt.axvline(metrics["right_edge_ui"], linestyle=":", linewidth=1.0, alpha=0.8, label="Right edge")
    plt.xlim([0.0, 1.0])
    plt.xlabel("Commanded sample phase [UI]")
    plt.ylabel("Voltage [V]")
    plt.title("Rail Statistics and Eye Opening vs Commanded Phase")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    _save_figure_if_requested(save_plots, save_prefix, "_eye_metrics.png")

    if cfg.get("debug", {}).get("plot_frequency_response", False):
        _plot_frequency_response(cfg)
        _save_figure_if_requested(save_plots, save_prefix, "_frequency_response.png")

    if cfg.get("nonlinearity", {}).get("enabled", False) and cfg.get("debug", {}).get("plot_nl_error", False):
        _plot_nl_debug(sim, cfg)

    if show_plots:
        plt.show()
    else:
        plt.close("all")


# === RUN_MODEL ===
# Run the full signal-chain simulation and return all waveforms and metrics.
def run_model(cfg):
    data_cfg = cfg["data"]
    waveform_cfg = data_cfg["waveform"]
    sampling_cfg = cfg["sampling"]
    noise_cfg = cfg["noise"]
    nl_cfg = cfg["nonlinearity"]

    tbit = float(data_cfg["tbit"])
    n_bits = int(data_cfg["n_bits"])
    samples_per_bit = int(data_cfg["samples_per_bit"])
    prbs_seed = int(data_cfg["prbs_seed"])

    vpp = float(waveform_cfg["vpp"])
    vcm = float(waveform_cfg["vcm"])

    sigma_j = float(sampling_cfg["sigma_j"])
    n_phases = int(sampling_cfg["n_phases"])
    n_ui_collect = int(sampling_cfg["n_ui_collect"])
    skip_initial_ui = int(sampling_cfg["skip_initial_ui"])

    sigma_v = 0.0
    if bool(noise_cfg["enabled"]):
        sigma_v = float(noise_cfg["sigma_v"])

    dt = tbit / float(samples_per_bit)
    t_end = n_bits * tbit
    t = np.arange(0.0, t_end, dt)

    bits = prbs7(n_bits, seed=prbs_seed)
    eq_symbols = apply_tx_eq(bits, cfg["tx_eq"])
    x_tx = build_nrz_waveform(eq_symbols, t, tbit, vpp=vpp, vcm=vcm)

    input_waveform_cfg = cfg.get("input_waveform", {"model": "nrz_channel_fir"})
    input_waveform_model = input_waveform_cfg.get("model", "nrz_channel_fir")

    pulse_symbols = None
    pulse_dense = None

    if input_waveform_model == "pulse_response_bits":
        pulse_symbols = apply_bit_pulse_response(eq_symbols, input_waveform_cfg["taps"])
        x_ch = build_nrz_waveform(pulse_symbols, t, tbit, vpp=vpp, vcm=vcm)
    elif input_waveform_model == "pulse_response_dense":
        pulse_dense = load_dense_pulse_response(input_waveform_cfg["file"])
        pulse_drive_symbols = 0.5 * vpp * eq_symbols
        x_ch_full = apply_dense_pulse_superposition(pulse_drive_symbols, pulse_dense, samples_per_bit)
        x_ch = x_ch_full[:len(t)]
    else:
        x_ch = apply_channel(x_tx, dt, samples_per_bit, cfg["channel"])

    y = apply_frontend(x_ch, dt, cfg["frontend"])

    y_before_nl = y.copy()
    if nl_cfg.get("enabled", False):
        y_after_nl = apply_nonlinearity(y, nl_cfg)
    else:
        y_after_nl = y.copy()
    y = y_after_nl

    results = sample_eye(
        t=t,
        y=y,
        tbit=tbit,
        n_phases=n_phases,
        n_ui_collect=n_ui_collect,
        sigma_j=sigma_j,
        sigma_v=sigma_v,
        skip_initial_ui=skip_initial_ui
    )

    metrics = compute_eye_metrics(results, tbit, cfg["eye_metrics"], vcm=vcm)

    return {
        "t": t,
        "x_tx": x_tx,
        "x_ch": x_ch,
        "y": y,
        "y_before_nl": y_before_nl,
        "y_after_nl": y_after_nl,
        "tbit": tbit,
        "dt": dt,
        "bits": bits,
        "eq_symbols": eq_symbols,
        "pulse_symbols": pulse_symbols,
        "pulse_dense": pulse_dense,
        "vpp": vpp,
        "vcm": vcm,
        "results": results,
        "metrics": metrics,
    }



# === MAIN ===
# Load config, run the model, print summary, and optionally show plots.
def main():
    default_cfg = "config/config_nominal.json"

    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    else:
        cfg_path = default_cfg
        print(f"[INFO] No config provided, using default: {cfg_path}")

    cfg = load_config(cfg_path)
    validate_config(cfg)

    sim = run_model(cfg)
    print_eye_summary(sim["metrics"], sim["tbit"])
    plot_results(sim, cfg)


if __name__ == "__main__":
    main()
