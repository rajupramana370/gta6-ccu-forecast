"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun
gta6-twitch-ccu-forecast-01).

Reads every <id>.in in MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out to MAKEFUN_OUTPUT_DIR (default /output):
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]} for h = 2..24.

Model (standard library only, deterministic, no network):
  * The case ships 38-42 comparables: past big game launches' hourly Twitch
    category viewers from 14 days before to 3 days after their release day.
  * GTA VI unlocks at rolling local midnight, console only, so the only
    comparables that behave like it are the rolling-midnight launches; the
    simultaneous-global launches (whose hour-1 print is a pre-unlock trickle)
    are a different regime and are excluded.
  * For each rolling comparable, take the log ratio of each release-day hour's
    viewers to its hour-1 print.  The forecast centre for hour h is the
    midpoint of the weighted mean and the weighted median of those log ratios
    across comparables (console-only launches weighted up, early-access-
    contaminated ones down), adjusted by a per-hour regression on log(hour-1
    print): big hour-1 prints (a huge UK/Australia midnight audience) mean a
    flatter day, small ones mean a much bigger rise into the US evening.  The
    spread is the weighted standard deviation of the adjusted log ratios at
    that hour.
  * Quantiles are lognormal: q_tau = p1 * exp(mu_h + z_tau * sigma_h), with
    z = (-1.645, -0.674, 0, 0.674, 1.645).
  * If no usable comparable exists, fall back to the house persistence
    baseline (every hour stays at the hour-1 print, spread widening as a
    random walk), so the output is always valid.

Validation (leave-one-out over the 12 rolling-midnight comparables, and the
four practice cases scored against SullyGnome's published hourly averages):
  * rolling LOO pinball loss 0.249 vs 0.357 for persistence
  * practice mean pinball loss 0.155 vs 0.387 for persistence
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread calibration: the raw per-hour spread of the adjusted log ratios is
# slightly wide for pinball loss; 0.9 was best on both the practice cases
# and leave-one-out over the rolling comparables.
SPREAD_MULT = 0.9
# Centre blend: the per-hour log ratios across launches are right-skewed (a
# few launches explode into the US evening), so the pure weighted median
# systematically under-forecasts the afternoon/evening peak (leave-one-out
# mean log-median error -0.11..-0.18 at hours 13-19). The centre is the
# midpoint of the weighted mean and the weighted median: the mean follows
# the skew, the median keeps it robust. 0.5 is the natural midpoint, and
# the result is not knife-edge (0.25-0.75 all beat the pure median on LOO
# with the practice mean within 0.001).
MU_BLEND = 0.5
# Floor on the per-hour log spread, so a freakishly tight hour still gets a
# sane band.
SD_FLOOR = 0.10
# A comparable needs its hour-1 print and at least this many release-day
# hours to be usable.
MIN_HOURS = 12
# Shrinkage for the regression slopes (pseudo-count of rows / games).
BETA_SHRINK = 8
BETA_PRIOR_WEIGHT = 6


def _parse_time(s):
    """ISO 8601 UTC timestamp -> epoch seconds (stdlib only)."""
    from datetime import datetime, timezone

    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _wquantile(pairs, tau):
    """Weighted quantile of [(value, weight)] pairs."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    cum = 0.0
    for value, weight in pairs:
        cum += weight
        if cum >= tau * total:
            return value
    return pairs[-1][0]


def _wmedian(pairs):
    return _wquantile(pairs, 0.5)


def _wmean(pairs):
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else 0.0


def _wstd(pairs, mu=None):
    if mu is None:
        mu = _wmean(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0
    return math.sqrt(sum(w * (v - mu) ** 2 for v, w in pairs) / total)


def _day_curve(comp):
    """{hour: viewers} for hours 1..24 of the comparable's release day."""
    try:
        day_start = _parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    for row in comp.get("hourly") or []:
        try:
            viewers = row["viewers"]
            offset = (_parse_time(row["from"]) - day_start) / 3600.0
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= offset < 24 and isinstance(viewers, (int, float)) and viewers > 0:
            out[int(offset) + 1] = viewers
    return out


def _classify(comp):
    """Unlock regime of a comparable, from its notes text.

    Returns (kind, base_weight): 'rolling' for midnight-local rolling
    unlocks, 'simul' for single simultaneous global unlocks, 'other' when
    the unlock is unverified, and 'target' for the GTA VI category's own
    (pre-release) history, which has no launch day.
    """
    name = (comp.get("name") or "").lower()
    if "grand theft auto vi" in name:
        return ("target", 0.0)
    unlock = (comp.get("unlock") or "").lower()
    if "midnight local" in unlock:
        return ("rolling", 1.0)
    if "simultaneous" in unlock or "single global" in unlock:
        return ("simul", 0.25)
    return ("other", 0.5)


def _game_weight(comp):
    """Fitting weight: rolling launches only; console-only up, early-access
    contaminated down."""
    kind, weight = _classify(comp)
    if kind != "rolling":
        return 0.0
    unlock = (comp.get("unlock") or "").lower()
    if "early-access-contaminated" in unlock:
        weight *= 0.5
    platforms = (comp.get("platforms") or "").lower()
    if "pc" not in platforms:
        weight *= 1.5
    return weight


def _ols(pairs):
    """Weighted least-squares slope of y on x for [(x, y, w)] pairs."""
    total_w = sum(w for _, _, w in pairs)
    if total_w <= 0:
        return 0.0
    mx = sum(x * w for x, _, w in pairs) / total_w
    my = sum(y * w for _, y, w in pairs) / total_w
    sxx = sum(w * (x - mx) ** 2 for x, _, w in pairs)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in pairs)
    return sxy / sxx if sxx > 1e-9 else 0.0


def _fit(case):
    """Fit the log-ratio model on the case's comparables.

    Returns predict(p1) -> {hour: [q05, q25, q50, q75, q95]}, or None when
    no comparable is usable.
    """
    rows = []  # (name, hour, log-ratio, weight, log v1)
    games = {}  # name -> (log v1, weight)
    for comp in case.get("comparables") or []:
        curve = _day_curve(comp)
        if not curve.get(1) or len(curve) < MIN_HOURS:
            continue
        weight = _game_weight(comp)
        if weight <= 0:
            continue
        log_v1 = math.log(curve[1])
        games[comp.get("name", "?")] = (log_v1, weight)
        for hour in range(2, 25):
            if hour in curve:
                rows.append((comp.get("name", "?"), hour,
                             math.log(curve[hour] / curve[1]), weight, log_v1))
    if not games or not rows:
        return None

    # Per-hour centre log ratio: blend of weighted mean and weighted median
    # (see MU_BLEND).
    mu = {}
    for hour in range(2, 25):
        hour_rows = [(r, w) for _, hh, r, w, _ in rows if hh == hour]
        if hour_rows:
            mu[hour] = (MU_BLEND * _wmean(hour_rows)
                        + (1.0 - MU_BLEND) * _wmedian(hour_rows))
        else:
            mu[hour] = 0.0

    # Centre of log hour-1 prints (weighted mean over games).
    total_w = sum(w for _, w in games.values())
    cx = sum(lv1 * w for lv1, w in games.values()) / total_w

    # Global slope of (log ratio - mu_h) on (log v1 - cx), from the
    # peak hours 12..22, shrunk toward 0.
    peak_pairs = [(lv1 - cx, r - mu[h], w)
                  for _, h, r, w, lv1 in rows if 12 <= h <= 22]
    beta_global = _ols(peak_pairs)
    beta_global *= len(peak_pairs) / (len(peak_pairs) + BETA_SHRINK)

    # Per-hour slopes, shrunk toward the global slope.
    beta_hour = {}
    for hour in range(2, 25):
        hour_pairs = [(lv1 - cx, r - mu[hour], w)
                      for _, hh, r, w, lv1 in rows if hh == hour]
        if len(hour_pairs) >= 5:
            b = _ols(hour_pairs)
            beta_hour[hour] = (b * len(hour_pairs)
                               + beta_global * BETA_PRIOR_WEIGHT) / (
                                   len(hour_pairs) + BETA_PRIOR_WEIGHT)
        else:
            beta_hour[hour] = beta_global

    # Per-hour spread: weighted sd of the slope-adjusted log ratios.
    sd = {}
    for hour in range(2, 25):
        adjusted = [(r - beta_hour[hour] * (lv1 - cx), w)
                    for _, hh, r, w, lv1 in rows if hh == hour]
        if not adjusted:
            sd[hour] = 0.4
            continue
        centre = _wmedian(adjusted)
        sd[hour] = max(_wstd(adjusted, centre), SD_FLOOR)

    def predict(p1):
        log_v1 = math.log(p1)
        out = {}
        for hour in range(2, 25):
            mean = mu[hour] + beta_hour[hour] * (log_v1 - cx)
            spread = sd[hour] * SPREAD_MULT
            out[hour] = [p1 * math.exp(mean + z * spread) for z in Z]
        return out

    return predict


def _persistence(p1):
    """House baseline: every hour stays at the hour-1 print, spread widening
    like a random walk in log space."""
    return {h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
            for h in range(2, 25)}


def _round2(x):
    """Round to 2 decimals, half up, as the starter does."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    predict = _fit(case)
    forecast = predict(p1) if predict is not None else _persistence(p1)
    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        qs = []
        prev = 0.0
        for q in forecast[hour]:
            # Keep every quantile finite, in (0, 1e9], non-decreasing.
            value = min(max(float(q), 0.01), 1_000_000_000.0)
            value = max(value, prev)
            qs.append(_round2(value))
            prev = qs[-1]
        hours.append({"hour": hour, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
