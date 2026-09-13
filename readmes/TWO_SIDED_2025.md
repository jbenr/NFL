# Fixed 2025 retrospective experiment

    python backtester.py --model two-sided --season 2025

Add --plan to print settings without building features or training.
Defaults: weeks 1–22, 20 regular-week feature lookback, 100 regular-week
training window, 100 ensemble members, 100 epochs, seed 1337, 8 model workers.
Pre-start training games and their feature histories are loaded automatically.
Feature preparation defaults to one worker. Set --prep-jobs independently of
--jobs (ensemble training); the CLI overrides any inherited NFL_WORKERS value.
For four preparation workers and eight training workers:

    python backtester.py --model two-sided --season 2025 --prep-jobs 4 --jobs 8

More preparation workers use more memory; one is the safer default.
The runner also accepts --lookback, --train-window, --iterations and --epochs.
No parameter sweep, feature selection or betting-cutoff optimization occurs.

For each game the shared neural function is applied twice:

- Away points: away offense vs home defense; away defense vs home offense.
- Home points: home offense vs away defense; home defense vs away offense.

Each metric has an own-offense and an own-defense slot, with the same weights
across home/away scoring passes. The function can learn interactions between
the two matchups and weather. Home field and rest remain additive learned terms.
Spread (away minus home margin) and total are derived from paired member scores;
ensemble SD is calculated from those paired margins/totals, not independent SDs.

Team stats reuse the current crunchski_2 calculations read-only, requesting
steep decay and UNSCALED league-snapshot z-score differences. No production
source is modified. Steep uses the existing 160-day scale, steepness 10 and
1% floor; QB Elo retains its existing separate decay. Rank-direction conventions
also remain the existing ones; this experiment does not redefine stat quality.
crunchski_3 applies pass/run usage multipliers AFTER league normalization, using
the attacking team's usage on both matchup perspectives, not just away offense.
Final numeric conditioning and missing-stat imputation are fit on training only.

Weather comes from data/weather/historical_features.parquet, joined by game ID.
Fields: feels-like temperature, wind, precipitation amount, rain, snowfall,
snow depth, plus an indoor indicator. Closed/domed games use 72 F, calm/dry.
Unknown/open roofs are not treated as closed. Missing weather is an error.
This is reconstructed historical weather, NOT a pregame forecast. Results are
explicitly retrospective and must not be treated as pregame betting validation.
Use --weather-file to supply another file with the same historical schema.

Outputs: data/bt/two_sided/2025/<configuration-and-source-fingerprint>/

- config.json: exact setup and retrospective-weather label.
- Per-week parquet files: game inputs, predictions and per-team attributions.
- predictions.csv: combined game results after completion.
- summary.json: spread/total MAE, RMSE and unfiltered ATS/O-U pick results.

Fitted results use a distinct data/cache/two_sided_scores namespace keyed by
actual model arrays, seed, ensemble settings and implementation sources.
Rerun the same command to resume; unchanged completed fits load from cache.
Old caches/results are not deleted. No production picks or policy is promoted.
