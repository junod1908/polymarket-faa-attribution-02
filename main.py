"""Forecast Polymarket Yes prices 240 minutes ahead as five quantiles.

Per market, the distribution of h-minute price changes over the recent
history gives empirical quantiles on a grid of horizons, demeaned and
interpolated in sqrt(h), around a persistence median (the last price):
prediction-market prices sit still for long stretches and then jump, and
the market's own recent changes say how wide each horizon's band must be.
Deterministic, standard library only, no network.
"""

import json
import math
import os

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
ZS = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
HGRID = (1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240)
WINDOW_SHORT = 300     # recent regime: minutes feeding the tight band
WINDOW_LONG = 1800     # fallback window for very young markets
BASE_SIGMA = 0.0025    # house baseline sigma, used when history is thin
BAND_SCALE = 0.85      # slight narrowing of the empirical band


def _quantile(sorted_vals, tau):
    """Linear-interpolated quantile of an already sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _median(vals):
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _clamp(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _baseline_rows(x, horizon):
    rows = []
    for h in range(1, horizon + 1):
        s = BASE_SIGMA * math.sqrt(h)
        rows.append([_clamp(round(x + z * s, 6)) for z in ZS])
    return rows


def _grid_quantiles(prices, start):
    """Per grid horizon, the five quantiles of demeaned h-minute changes
    over prices[start:], with gaps filled from the nearest neighbour.
    Returns None when no grid point has enough samples."""
    n = len(prices)
    grid = []
    for h in HGRID:
        i0 = max(start, h)
        if n - i0 < 40:
            grid.append(None)
            continue
        changes = [prices[i] - prices[i - h] for i in range(i0, n)]
        med = _median(changes)
        centered = sorted(c - med for c in changes)
        grid.append([_quantile(centered, t) for t in TAUS])
    if all(g is None for g in grid):
        return None
    for idx, g in enumerate(grid):
        if g is None:
            for d in range(1, len(grid)):
                if idx - d >= 0 and grid[idx - d] is not None:
                    grid[idx] = grid[idx - d]
                    break
                if idx + d < len(grid) and grid[idx + d] is not None:
                    grid[idx] = grid[idx + d]
                    break
    return grid


def forecast_market(prices, horizon=240):
    """Return `horizon` rows of five quantiles for one market's prices."""
    n = len(prices)
    x = prices[-1]
    if n < 30:
        return _baseline_rows(x, horizon)

    # Empirical quantiles of demeaned h-minute changes on the grid, from
    # the market's recent history: the last few hours carry the current
    # volatility regime, and prediction-market prices sit still for long
    # stretches, so a short window keeps the band honest about both the
    # stillness and the current size of a move. The long window remains
    # as a fallback for markets too young to fill the short one.
    grid_q = _grid_quantiles(prices, max(0, n - WINDOW_SHORT))
    if grid_q is None:
        grid_q = _grid_quantiles(prices, max(0, n - WINDOW_LONG))
    if grid_q is None:
        return _baseline_rows(x, horizon)

    grid_h = [min(h, horizon) for h in HGRID if h <= HGRID[-1]]
    rows = []
    for h in range(1, horizon + 1):
        med_path = x
        # Interpolate the five quantiles in sqrt(h) across the grid.
        if h <= grid_h[0]:
            qs = grid_q[0]
        elif h >= grid_h[-1]:
            qs = grid_q[-1]
        else:
            j = 0
            while grid_h[j + 1] < h:
                j += 1
            h0, h1 = grid_h[j], grid_h[j + 1]
            w = (math.sqrt(h) - math.sqrt(h0)) / (math.sqrt(h1) - math.sqrt(h0))
            qs = [grid_q[j][k] * (1.0 - w) + grid_q[j + 1][k] * w
                  for k in range(5)]
        qs = [q * BAND_SCALE for q in qs]
        row = []
        prev = 0.0
        for k in range(5):
            v = _clamp(round(med_path + qs[k], 6))
            if v < prev:
                v = prev
            row.append(v)
            prev = v
        rows.append(row)
    return rows


def forecast_case(case):
    horizon = case.get("horizon_minutes", 240)
    out = []
    for m in case["markets"]:
        out.append({"id": m["id"], "q": forecast_market(m["prices"], horizon)})
    return {"forecasts": out}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
