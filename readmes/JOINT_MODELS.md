# Joint matchup models

Two separate ensembles, trained on actual away-minus-home margin and total points.
Both see both sets of 15 offense-minus-opposing-defense differences.
The same network F evaluates both orientations: margin = F(A,B) - F(B,A);
total = (F(A,B) + F(B,A))/2 plus the training mean total.
Context travels with each side, so swapping the complete side blocks negates the
margin and preserves the total. Context enters the network and can interact with
matchup strength, rather than being an identical additive term that cancels.

## Preview

    python joint_scoring.py --season 2026 --week 1 --jobs 8

Default groups: stadium, field, referee, weather, importance. Use --groups with
no names for the no-extra-context baseline. Each market has 100 members and
100 epochs by default; this is two ensembles, not one.

Reports use the existing Headline / Spread / Totals / Feature importance tabs.
The original headline table is unchanged and remains the production model,
not the joint challenger. Packet folders are separate. Scores are *implied*
from the two models, not directly trained. SDs are computed independently for
margin and total. No model is automatically approved for bets.

## Context and data availability

- Home-site flag and signed rest difference are always present.
- Stadium ID, roof, and surface use training-only categories. Unknown categories
  are zero encoded. Stadium indicators belong to the home side.
- Referee uses the existing prior scoring-average deviation, shrunk with 20
  league-average games, excluding current-week and future results. It appears
  in both side blocks and can affect spread through matchup interactions.
- Importance uses paired remaining-season scenarios (details below), not distance
  from a standings cutline. Old must-win/clinch and generic rivalry bonuses are
  excluded. It measures what this result could change, not player motivation.
- Weather: temperature F, wind mph, precipitation probability, indoor flag and
  missing flags. Closed/domed roofs use a nominal 72 F, zero wind/precipitation.
  Roof metadata and referee assignments are not archived decision-time records.

Forecast mode is the default. It NEVER silently substitutes recorded game-day
temperature/wind. Without a forecast file, outdoor values remain missing and are
imputed from training-only medians with explicit missing indicators. Thus no
claim of outdoor-weather predictive value is possible until data is supplied.

Supply --weather-file PATH.csv (or parquet), with columns:

    game_id, issued_at, valid_at, temperature_f, wind_mph, precip_probability

Timestamps must carry UTC Z or an explicit offset; precipitation is a probability
between 0 and 1. For each game, use the latest forecast issued at least
--decision-hours before kickoff (default 24), valid within one hour of kickoff.
The schedule's game time is interpreted as US Eastern.

For exploratory historical weather only:

    python joint_scoring.py --season 2025 --week 22 --weather-source recorded

Recorded schedule weather is not an archived forecast. Do not treat its backtest
as evidence of executable pregame weather-based returns. Precipitation remains
missing in recorded mode because the schedule does not supply it here.

## Walk-forward tests (run separately from previews)

    python backtester.py --model joint --groups stadium field referee weather importance --start-season 2017 --validation-season 2025 --season 2025 --week 22

This compares joint baseline vs all requested context, predicting both markets
for each historical week. Use --individual for baseline plus individual groups;
--combined tests every subset (32 variants with all five groups).
Baseline versus all groups together is now the default (--full-only still works).
Defaults: 100 ensemble members, 8 training workers, 60 minimum calibration bets.
Feature preparation defaults to one worker unless NFL_WORKERS is explicitly set.
This worker default does not resolve the separate native loader crash.
Use the same --weather-file/--weather-source options as the preview.

Cutoffs are selected on pre-2025 predictions, then frozen for 2025. Existing
cutoff/settlement logic is reused; training does not see market lines or odds.
Each week trains on the prior 20 regular weeks plus intervening playoff games.
History coverage and earlier feature data are derived from the start/end seasons
and the training window. This runner no longer accepts --history-weeks.
Saved results: data/optimize_picks/joint_context_forecast (or _recorded).
Ensemble caches are separate from shared-scoring caches. Preserve old research
as a different model version: the QB-allowed change also changes new inputs.

Attribution is integrated gradients relative to a zero transformed-input
reference; features can interact and contributions are not causal.
The importance tab is a training-sample permutation diagnostic, not validation.
Model and feature selection remain exploratory; no best-return guarantee.

## Playoff leverage

For each historical week, keep earlier results fixed and simulate 512 paired
remaining-season scenarios. For a target game, force an away win in one branch
and a home win in the other, leaving every other simulated result unchanged.
Seed conferences with division champions first, then wild cards. Six teams/two
byes before 2020; seven teams/one bye from 2020.

Each side supplies these model features:

- playoff_swing: playoff-entry share with a win minus share with a loss;
- division_swing: corresponding division-title swing;
- bye_swing and top_seed_swing: respective bye/No. 1 seed swings;
- seed_swing: fraction of paired scenarios with a changed seed while qualifying
  in both branches;
- advancement_swing: 1 for actual playoff games, 0 in the regular season;
- postseason flag;
- importance: maximum of those swings (not a weighted sum or calibrated odds).

Conditional win/loss scenario shares are also saved for reporting. The maximum
avoids double-counting overlapping division/bye/seed implications; separate
components let the model learn whether those stakes have different value.
Each week's features are cached separately. Only requested feature weeks are
built, while the full remaining regular-season schedule informs scenarios.

Assumptions: results available before the week, not kickoff-by-kickoff updates;
future games are 50/50 with no simulated ties; historical ties count half.
No market lines, future scores, or actual eventual playoff participants enter
the regular-season calculation. Common-game, head-to-head, division/conference,
strength-of-victory and strength-of-schedule tiebreaks are included. Remaining
points-based tiebreaks use shared random lots. This is still an approximate
scenario heuristic, not an official clinch/elimination engine. A zero in 512
scenarios does not prove an event mathematically impossible. Special 2022
postseason venue contingencies are not represented by the normal seed rules.

Inspect a week without neural-net training:

    python playoff_importance.py --season 2023 --week 18 --samples 2048

Reference rules:
https://www.nfl.com/standings/tie-breaking-procedures
https://www.nfl.com/news/new-cba-includes-playoff-expansion-to-14-teams-0ap3000001106259
