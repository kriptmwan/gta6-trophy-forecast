# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kriptmwan/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kriptmwan/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A cohort-model forecaster for Grand Theft Auto VI's Exophase trophy earn
rates, built by an AI agent for make.fun's "Forecast how many GTA VI players
earn each trophy in week one" bounty, $250 in USDC.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `<id>.in` case in
the input directory and writes `<id>.out` with five quantiles (5/25/50/75/95%)
of each trophy's earn rate at the case's target. It runs offline, with no
network, in ~0.3 s per case, and is deterministic: the same case file always
produces the same output bytes.

## The idea

An earn rate is a ratio with a mechanism: `rate_i(t) = E_i(t) / N(t)`, the
players who earned trophy `i` over the players Exophase tracks. At launch
`N(t)` surges as new players arrive, and every one of them starts with zero
trophies: a surge **dilutes** every rate first, and a rate rises only as
cohorts of players mature into the trophy. Extrapolating the ratio directly
hides that, and a straight line fitted to `logit(rate)` against log time
overshoots catastrophically for a saturating adoption curve (on synthetic
launch weeks it forecast 87% for a trophy whose true day-7 rate was 28%).

So `main.py` models the mechanism:

1. **Arrivals.** The reads give `N(t)`. Distinct reads give arrival buckets:
   the players who arrived between one distinct read and the next, plus the
   first read's whole count as the opening cohort. Consecutive identical
   reads are one observation — Exophase refreshes its page on its own
   schedule, so a run of identical reads is one refresh, not many data
   points.
2. **Players at the target.** `ln N = c + d ln(1 + t/24h)` by least squares
   over the distinct reads, `d` clamped to [0, 2]; read at the target, never
   below the last value, never more than 8x it. The fitted curve also spreads
   the forecast arrivals between the last read and the target.
3. **Per trophy, a delay distribution.** A player who joins at time `s`
   earns trophy `i` after a delay `D_i ~ LogNormal(mu_i, sigma)` with sigma
   fixed at 0.7, and only a share `p_i` of players ever earn it. So
   `E_i(t) = p_i * sum dN * Phi((ln(t - s) - ln mu_i) / sigma)` over the
   arrival buckets. For each `mu` on a log-spaced grid (1.5 h to 1500 h) the
   model is linear in `p_i`, so `p_i` is a weighted least-squares estimate
   from the case's distinct snapshots, and `mu_i` is the grid point with the
   smallest weighted squared error. The forecast carries the same sum to the
   target, including the forecast arrival buckets. By construction it
   saturates at `p_i`: it cannot run away the way a log-time line does, and
   a trophy whose earners are slow shows up as a rate that **falls** while
   the player base surges, which is what dilution really does.
4. **Spread.** Quantiles are `logistic(logit(median) + z * s)`, with `s`
   combining in quadrature: the residual scatter of the trophy's cohort fit
   in log space, a random-walk term `0.40 * sqrt(horizon in days)`, a
   flat-history penalty (+1.10 when the rate never moved in the history,
   +0.50 when it moved once — a flat history means Exophase has not
   refreshed since the trophy was first seen, and launch night's one
   refresh cut every rate by roughly the ratio of the player counts,
   638 -> 3,868, a logit shift near 1.8), a small-count term
   `0.50 / sqrt(E_last + 1)`, and a dilution term
   `0.25 * |ln(N_target / N_last)|`. `s` is floored at 0.25, capped at 2.0.

Edge cases: a trophy never earned gets median 0.02% with `s = 1.8`; a trophy
at 100% needs no special case (the fit finds a short delay and `p` near 1,
and the forecast arrivals that have not yet matured pull the rate just below
100 on their own). Two safety nets: when the snapshots span less than 2
hours the delay distribution is unidentified, so the program says the rate
stays where it is with a wide spread; and when the fit extrapolates with
`p` near 1, the median may not move more than 10x from the last read (30x
for a trophy with fewer than 10 earners).

## Validation (honest version)

The kit's practice cases cannot validate a trend model: their histories are
perfectly flat (7–8 identical pre-refresh reads), so every trend fit is a
flat line and the only thing the practice measures is whether the spread
survives the refresh. On the practice cases this program is valid, fast
(0.1 s), and much wider than the starter baseline — e.g. for a trophy last
at 1.10% it forecasts [0.17, 0.51, 1.10, 2.37, 6.96] against the starter's
[0.90, 1.01, 1.10, 1.20, 1.35]. Against a synthetic refresh answer (earned
counts frozen, player counts 638 -> 3,868, as the house recorded launch
night's refresh) it scores a pinball loss of 0.99 vs the starter's 1.69,
with 98% 90%-coverage.

The trend model was validated on a synthetic cohort simulator instead:
players arrive on a power-law `N(t)`, each trophy has a lognormal
time-to-earn by tier (first mission hours, story tens of hours, grinds
hundreds), Exophase-style refresh freezes and 2-decimal rounding, reads
every 10 minutes, checkpoint at 59.5 h, target at 168 h. Mean pinball loss
on the logit scale over 15 simulations (5 seeds x 3 worlds: base, faster
player growth, slower adoption): **this program 0.34, house curve baseline
0.49, starter persistence baseline 1.41**. The delay spread sigma = 0.7 was
chosen by checking model sigmas 0.6–1.1 against simulated worlds with true
delay spreads 0.6–1.2: 0.7 had the lowest mean loss across all of them, so
it is a robustness choice, not a fit to any one world. The simulator is not
the real world — its adoption curves are my own guesses — so these numbers
are evidence the structure is right, not a promise about the final score.
