"""gta6-trophy-rarity-forecast-01: forecast every GTA VI trophy's Exophase earn
rate at a case's target, as five quantiles (5/25/50/75/95%).

Python 3.13, standard library only. Run by the grader as
`python3 -P -s /bot/main.py` (PYTHONHASHSEED=0), no network, 8 s per case.
Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for
/input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Input: every <id>.in in the input directory, JSON
{"schema": "makefun/gta6-trophies-case/v1", "id", "at", "horizon_minutes",
 "target", "levels", "game", "trophies", "history", "notes"}, where `trophies`
lists {id, name, description, tier, points, secret} and `history` every read
before `at`, oldest first, each {slot, at, players_tracked, trophies_earned,
rates} with `rates` in trophy order (null where a read lacked the trophy).

Output: <id>.out, {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]},
one row per trophy, quantiles in percent, non-decreasing, at most 65,536 bytes.

Method -- a cohort model, because an earn rate is a ratio with a mechanism
-------------------------------------------------------------------------
rate_i(t) = E_i(t) / N(t): the players who earned trophy i over the players
Exophase tracks. N(t) surges at launch as new players arrive, and every one
of them starts with zero trophies: a surge DILUTES every rate first, and a
rate rises only as cohorts of players mature into the trophy. A direct
extrapolation of the ratio hides that, and a straight line fitted to
logit(rate) against log time overshoots catastrophically for a saturating
adoption curve (on synthetic launch weeks it forecast 87% for a trophy whose
true day-7 rate was 28%). So this program models the mechanism:

1. Arrivals. The reads give N(t), the players tracked. Distinct reads give
   arrival buckets: the players who arrived between one distinct read and
   the next, plus the first read's whole count as the opening cohort.
   Consecutive identical reads are one observation -- Exophase refreshes
   its page on its own schedule, so a run of identical reads is one
   refresh, not many data points.

2. Players tracked at the target. Fit ln N = c + d ln(1 + t/24h) by least
   squares over the distinct reads, d clamped to [0, 2]; read the line at
   the target. N never forecasts below its last value, never more than 8x
   it. The fitted curve also spreads the forecast arrivals between the
   last read and the target.

3. Per trophy, a delay distribution. A player who joins at time s earns
   trophy i after a delay D_i ~ LogNormal(mu_i, sigma), with sigma fixed at
   0.7, and only a share p_i of players ever earn it at all. (Sigma was
   chosen on synthetic cohort worlds with true delay spreads from 0.6 to
   1.2: 0.7 had the lowest mean pinball loss across all of them, so it is
   a robustness choice, not a fit to any one world.) So
       E_i(t) = p_i * sum over arrival buckets of dN * Phi((ln(t - s) - ln mu_i) / sigma)
   For each mu on a log-spaced grid (1.5 h to 1500 h) the model is linear
   in p_i, so p_i is a weighted least-squares estimate from the case's
   distinct snapshots, and mu_i is the grid point with the smallest
   weighted squared error. The forecast is the same sum carried to the
   target, including the forecast arrival buckets. By construction the
   forecast saturates at p_i: it cannot run away the way a log-time line
   does, and a trophy whose earners are slow shows up as a rate that
   FALLS while the player base surges, which is what dilution really does.

4. Spread (logit standard deviation s, quantiles logistic(logit(m) + z s)):
   s combines, in quadrature,
   - the residual scatter of the trophy's cohort fit in log space (floor
     0.10, cap 0.80; 0.35 when the trophy has too few distinct points),
   - a random-walk term 0.40 * sqrt(horizon in days),
   - a flat-history penalty: +1.10 when the trophy's rate never moved in
     the history, +0.50 when it moved exactly once. A flat history means
     Exophase has not refreshed since the trophy was first seen, and the
     one refresh the house recorded on launch night cut every rate by
     roughly the ratio of the player counts (638 -> 3,868 tracked, a logit
     shift near 1.8). A forecast that has never seen a refresh must carry
     that risk in its width, not in its median.
   - a small-count term 0.50 / sqrt(E_last + 1), capped at 0.50: rates
     printed to 2 decimals from a few hundred tracked players quantise
     the earned count, so a rare trophy's level is genuinely uncertain,
   - a dilution term 0.25 * |ln(N_target / N_last)|: the wider the
     forecast player growth, the less certain the dilution.
   s is floored at 0.25 and capped at 2.0.

Edge cases, each decided, not defaulted:
- A trophy never earned in the history (rate 0 throughout): median 0.02%
  with s = 1.8. By day seven a few players may reach it, but with tens of
  thousands tracked a handful of earners is still a tiny rate; the wide
  upper tail (95% near 0.4%) carries the chance it turns out easy.
- A trophy at 100% in the last read needs no special case: the cohort fit
  finds a short delay and p near 1, and the forecast arrivals that have
  not yet matured pull the rate just below 100 on their own.
- A trophy whose last read lacks a rate (null): fall back to its last
  non-null rate; if it never had one, treat it as never earned.
- Two safety nets on the cohort fit itself. When the snapshots span less
  than 2 hours the delay distribution is unidentified (two reads minutes
  apart say nothing about a delay measured in hours), so the program says
  the rate stays where it is, with a wide spread, instead of fitting.
  And when the fitted delay curve extrapolates with p_hat near 1 -- the
  fit claiming (nearly) every player will eventually earn a trophy
  barely anyone has -- the forecast median may not move more than 10x
  from the last read (30x for a trophy with fewer than 10 earners). A
  well identified fit may legitimately forecast a 20x move from a low
  base (backtests: every such move was real), so it is not clipped; the
  cap exists only for the unidentified edge, which once forecast 94%
  for a trophy last seen at 0.38%.
- Rates are clamped to [0.01, 99.99] before any logit, as the scorer does.

Everything is deterministic: no randomness, no wall-clock reads, no network.
The same case file always produces the same output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a sibling
# module shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at LEVELS.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# Release midnight, the origin of the log-time axis (game.release_date).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# Launch dynamics run on a day scale: the log-time regressor is ln(1 + t/24h).
TIME_SCALE_HOURS = 24.0

# --- tuning constants (round numbers, chosen on synthetic cohort backtests;
# see README.md). None of them is fit to the practice cases: the practice
# histories are perfectly flat, so they carry no trend signal at all. ---
N_SLOPE_MAX = 2.0          # clamp on ln N vs ln(1+t/24h) slope
N_GROWTH_CAP = 8.0         # forecast players tracked at most 8x the last read
SIGMA_DELAY = 0.7          # lognormal delay spread, shared by all trophies
MU_GRID = [math.exp(x) for x in
           [i * (math.log(1500.0) - math.log(1.5)) / 19.0 + math.log(1.5)
            for i in range(20)]]
RESID_FLOOR = 0.10
RESID_CAP = 0.80
RESID_DEFAULT = 0.35       # when a trophy has too few distinct points
RW_SIGMA = 0.40            # logit random-walk spread per sqrt(day)
FLAT_PENALTY_0 = 1.10      # rate never moved in the history
FLAT_PENALTY_1 = 0.50      # rate moved exactly once
SMALL_COUNT_SIGMA = 0.50
DILUTION_SIGMA = 0.25
SPREAD_FLOOR = 0.25
SPREAD_CAP = 2.0
NEVER_EARNED_MEDIAN = 0.02
NEVER_EARNED_SPREAD = 1.8
MIN_SNAPSHOT_SPAN_HOURS = 2.0  # below this the delay fit is unidentified
MEDIAN_MOVE_CAP = 10.0         # forecast median move cap (large counts)
MEDIAN_MOVE_CAP_RARE = 30.0    # ... and for trophies with < 10 earners


def parse_time(s):
    """ISO-8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt):
    return (dt - RELEASE).total_seconds() / 3600.0


def log_time(t_hours):
    """Log-time regressor: ln(1 + t / 24h)."""
    return math.log1p(max(t_hours, 0.0) / TIME_SCALE_HOURS)


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up (as the starter rounds it)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def ols(xs, ys):
    """Least-squares line y = a + b x. Returns (a, b). Degenerate -> (mean y, 0)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return ys[0], 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def norm_quantile(p):
    """Acklam's approximation to the standard normal quantile (fallback for
    levels outside the LEVELS table)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def z_for(level):
    for lv, z in zip(LEVELS, Z):
        if level == lv:
            return z
    return norm_quantile(level)


def distinct_points(series):
    """Collapse consecutive duplicates of a (time, value) series, keeping the
    first time of each plateau. The last point is always kept (its time
    matters for the fit even when its value repeats the plateau)."""
    out = []
    for t, v in series:
        if out and out[-1][1] == v:
            continue
        out.append((t, v))
    return out


def phi(z):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def matured(buckets, t, mu):
    """Players from `buckets` [(arrival_time, count)] who have earned the
    trophy by time t, per unit of p: sum dN * Phi((ln(t - s) - ln mu)/sigma).
    Buckets arriving after t, or less than 0.05 h before it, contribute 0."""
    ln_mu = math.log(mu)
    total = 0.0
    for s, dn in buckets:
        dt = t - s
        if dt <= 0.05 or dn <= 0:
            continue
        total += dn * phi((math.log(dt) - ln_mu) / SIGMA_DELAY)
    return total


def forecast_case(case, name):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    levels = case.get("levels", LEVELS)

    t_target = hours_since_release(parse_time(case["target"]))
    t_last = hours_since_release(parse_time(history[-1]["at"]))
    horizon_days = max(case["horizon_minutes"], 0) / 1440.0

    # ---- players tracked: distinct series, log-log fit on a day scale. ----
    n_series = []
    for read in history:
        n = read.get("players_tracked")
        if n is None or n <= 0:
            continue
        n_series.append((hours_since_release(parse_time(read["at"])), float(n)))
    n_distinct = distinct_points(n_series)
    n_last = n_series[-1][1] if n_series else 1.0
    if len(n_distinct) >= 2:
        xs = [log_time(t) for t, _ in n_distinct]
        ys = [math.log(v) for _, v in n_distinct]
        a_n, b_n = ols(xs, ys)
        b_n = min(max(b_n, 0.0), N_SLOPE_MAX)
        a_n = sum(y - b_n * x for x, y in zip(xs, ys)) / len(xs)
        n_target = math.exp(a_n + b_n * log_time(t_target))
    else:
        a_n, b_n = math.log(n_last), 0.0
        n_target = n_last
    n_target = min(max(n_target, n_last), n_last * N_GROWTH_CAP)
    dilution = abs(math.log(n_target / n_last)) if n_last > 0 else 0.0

    # ---- arrival buckets: opening cohort + increments at distinct reads. ----
    buckets = []
    prev_n = 0.0
    for t, n in n_distinct:
        if n > prev_n:
            buckets.append((t, n - prev_n))
        prev_n = max(prev_n, n)
    # Forecast arrivals between the last read and the target, spread over
    # 16 buckets following the fitted N curve.
    if n_target > n_last and t_target > t_last:
        steps = 16
        prev = n_last
        for k in range(1, steps + 1):
            tk = t_last + (t_target - t_last) * k / steps
            nk = math.exp(a_n + b_n * log_time(tk))
            nk = min(max(nk, prev), n_target)
            if nk > prev:
                buckets.append((tk, nk - prev))
            prev = nk

    # ---- per-trophy earned-count snapshots (distinct in E). ----
    rows = []
    for i, trophy in enumerate(trophies):
        pts = []  # (t, rate) with rate not None
        for read in history:
            rate = read["rates"][i] if i < len(read["rates"]) else None
            if rate is None:
                continue
            pts.append((hours_since_release(parse_time(read["at"])), float(rate)))
        dpts = distinct_points(pts)
        last_rate = pts[-1][1] if pts else 0.0
        n_distinct_rates = len({r for _, r in dpts})
        e_last = (last_rate / 100.0 * n_last) if pts else 0.0

        if not pts or all(r == 0.0 for _, r in pts):
            med, spread = NEVER_EARNED_MEDIAN, NEVER_EARNED_SPREAD
        else:
            # Earned-count snapshots, aligned with N at each read.
            snaps = []
            for read in history:
                rate = read["rates"][i] if i < len(read["rates"]) else None
                n = read.get("players_tracked")
                if rate is None or n is None or n <= 0:
                    continue
                snaps.append((hours_since_release(parse_time(read["at"])),
                              float(rate) / 100.0 * float(n)))
            snaps = distinct_points(snaps)
            span = snaps[-1][0] - snaps[0][0] if len(snaps) >= 2 else 0.0
            if len(snaps) < 2 or span < MIN_SNAPSHOT_SPAN_HOURS:
                # Too little time has passed to identify a delay distribution
                # at all (two reads minutes apart say nothing about a delay
                # measured in hours): say the rate stays where it is, with
                # a wide spread. Fitting a delay to such data picks the
                # grid's fastest mu and a capped p, and then extrapolates a
                # near-total maturation that the data never showed.
                med, spread = last_rate, max(
                    math.sqrt(RESID_DEFAULT ** 2
                              + (RW_SIGMA * math.sqrt(horizon_days)) ** 2
                              + FLAT_PENALTY_0 ** 2),
                    SPREAD_FLOOR)
            else:
                # Grid over mu; p by weighted least squares at each mu.
                weights = [1.0 / (e + 2.0) ** 2 for _, e in snaps]
                best = None  # (sse, mu, p, resid)
                for mu in MU_GRID:
                    a_vals = [matured(buckets, t, mu) for t, _ in snaps]
                    num = sum(w * e * a for w, (_, e), a in zip(weights, snaps, a_vals))
                    den = sum(w * a * a for w, a in zip(weights, a_vals))
                    if den <= 0:
                        continue
                    p = min(max(num / den, 0.0), 1.0)
                    sse = sum(w * (e - p * a) ** 2
                              for w, (_, e), a in zip(weights, snaps, a_vals))
                    if best is None or sse < best[0]:
                        best = (sse, mu, p, a_vals)
                _, mu_hat, p_hat, a_vals = best
                # Residual scatter in log space (levels, not just shape).
                log_res = [math.log((e + 0.5) / (p_hat * a + 0.5))
                           for (_, e), a in zip(snaps, a_vals)]
                if len(log_res) >= 3:
                    mean_r = sum(log_res) / len(log_res)
                    var = sum((r - mean_r) ** 2 for r in log_res) / max(len(log_res) - 2, 1)
                    resid = min(max(math.sqrt(var), RESID_FLOOR), RESID_CAP)
                else:
                    resid = RESID_DEFAULT
                a_target = matured(buckets, t_target, mu_hat)
                e_target = p_hat * a_target
                med = 100.0 * e_target / n_target if n_target > 0 else last_rate
                # Safety net for a fit the data never identified: however the
                # delay curve extrapolates, the rate may not move more than
                # 10x from its last read (launch-week story trophies move
                # up to ~6x in backtests; nothing in the data justifies more).
                # The cap bites only when the fit itself is at its edge
                # (p_hat near 1: the fit claims (nearly) every player will
                # eventually earn a trophy barely anyone has). A well
                # identified fit may legitimately forecast a 20x move from
                # a low base -- backtests show all such moves are real --
                # so it is not clipped.
                if p_hat > 0.9 and last_rate > 0:
                    cap = MEDIAN_MOVE_CAP if e_last >= 10.0 else MEDIAN_MOVE_CAP_RARE
                    med = min(max(med, last_rate / cap), last_rate * cap)
                med = clamp_rate(med)
                flat = FLAT_PENALTY_0 if n_distinct_rates <= 1 else (
                    FLAT_PENALTY_1 if n_distinct_rates == 2 else 0.0)
                small = min(SMALL_COUNT_SIGMA / math.sqrt(e_last + 1.0), SMALL_COUNT_SIGMA)
                spread = math.sqrt(resid ** 2
                                   + (RW_SIGMA * math.sqrt(horizon_days)) ** 2
                                   + flat ** 2 + small ** 2
                                   + (DILUTION_SIGMA * dilution) ** 2)
                spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)

        lg = logit(med)
        qs = []
        prev = None
        for lv in levels:
            q = round4(logistic(lg + z_for(lv) * spread))
            q = min(max(q, 0.0), 100.0)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
