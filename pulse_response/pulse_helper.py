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

    # interior slopes
    for k in range(1, n - 1):
        if d[k - 1] == 0.0 or d[k] == 0.0 or np.sign(d[k - 1]) != np.sign(d[k]):
            m[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    # endpoint slope m[0]
    m0 = ((2.0 * h[0] + h[1]) * d[0] - h[0] * d[1]) / (h[0] + h[1])
    if np.sign(m0) != np.sign(d[0]):
        m0 = 0.0
    elif np.sign(d[0]) != np.sign(d[1]) and abs(m0) > 3.0 * abs(d[0]):
        m0 = 3.0 * d[0]
    m[0] = m0

    # endpoint slope m[-1]
    mn = ((2.0 * h[-1] + h[-2]) * d[-1] - h[-1] * d[-2]) / (h[-1] + h[-2])
    if np.sign(mn) != np.sign(d[-1]):
        mn = 0.0
    elif np.sign(d[-1]) != np.sign(d[-2]) and abs(mn) > 3.0 * abs(d[-1]):
        mn = 3.0 * d[-1]
    m[-1] = mn

    return m


# === INTERPOLATE_PULSE_RESPONSE_UI ===
# Interpolate a coarse UI-spaced pulse response into a dense pulse response.
#
# Inputs:
#   coarse_y         : pulse amplitudes at coarse UI locations
#   samples_per_bit  : dense samples per UI for output
#   coarse_x_ui      : optional UI locations for coarse_y; defaults to [0, 1, 2, ...]
#   method           : "pchip" (recommended) or "linear"
#
# Returns:
#   dense_x_ui       : dense UI time axis
#   dense_y          : dense interpolated pulse response
def interpolate_pulse_response_ui(
    coarse_y,
    samples_per_bit,
    coarse_x_ui=None,
    method="pchip",
):
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

    # Dense grid includes endpoint
    n_dense = int(round((x1 - x0) * samples_per_bit)) + 1
    dense_x_ui = np.linspace(x0, x1, n_dense)

    if method == "linear":
        dense_y = np.interp(dense_x_ui, coarse_x_ui, coarse_y)
        return dense_x_ui, dense_y

    if method != "pchip":
        raise ValueError("method must be 'pchip' or 'linear'")

    m = _pchip_slopes(coarse_x_ui, coarse_y)
    dense_y = np.zeros_like(dense_x_ui)

    # piecewise cubic Hermite
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
