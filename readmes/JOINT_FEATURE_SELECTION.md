# Joint feature selection

Run from the NFL project after the current backtest finishes:

    python joint_feature_selection.py

This prints a compute plan only. Execute:

    python joint_feature_selection.py --run

Spread and total are searched separately. To run just one:

    python joint_feature_selection.py --run --markets spread

Defaults: 2017–2024 development, 2025 final evaluation; every historical week;
20 preceding regular training weeks plus intervening playoffs; 100 epochs;
8 training workers; 10-member screening ensembles; 100-member final comparisons.
Raw features/context reuse existing caches. A context-enriched panel is also cached.
No live odds/data pulls are added. Weather forecasts remain missing unless an
archived forecast file is supplied (--weather-file); recorded mode is explicitly
retrospective. Selected features do not automatically change production or bets.

## Method

- Screen the 15 matchup metric differences, home field, rest, and five context
  groups. Categorical encodings and each context group stay together.
- Remove a feature from BOTH team orientations. Inputs are zero-masked after
  preprocessing to hold network width, architecture and random initialization
  fixed across candidates. A zero-masked feature provides no varying information.
  Deployment must retain the full-width mask; shrinking the network is a separate
  experiment. Weather/context preprocessing uses training data only.
- Retrain the actual joint neural model, not a Ridge proxy or attribution ranking.
  Spread targets actual away-minus-home margin; totals targets actual score sum.
  Objective is game-weighted development MAE. RMSE is reported too.
- Backward beam search evaluates full model, all single removals, then removal
  combinations around the best two candidates. Default budget: 64 subsets per
  market. Complete removal layers only: unused budget is possible.
- Search is bounded, not exhaustive and NOT a global optimum claim. With 22
  candidates there are over four million subsets. Increase --max-subsets for a
  broader search; --beam widens branches. All subsets use identical weekly folds.
- Refit the screening winner and full baseline with 100 members on EVERY
  development week. Keep the candidate only if its development MAE is lower.
- Freeze selection BEFORE evaluating 2025. Report both baseline and selected
  model. Do not use final-period outcomes to revise the selected feature set.
  2025 was inspected in earlier research, so it is not a pristine holdout.
- Season-block bootstrap intervals are descriptive, not search-adjusted evidence.
  A single holdout season cannot estimate uncertainty across seasons.
- SD is ensemble disagreement, not calibrated outcome uncertainty. No betting
  thresholds, profitability claims, or automatic bet approvals are produced.

For a cheaper exploratory screen, --week-stride 4 samples every fourth available
week within each development season. Finalist comparisons still use every week.
Do not interpret a sparse screen as a full weekly validation.

## Outputs and restart

Under data/optimize_picks/joint_feature_selection/<run fingerprint>/:

- config.json: exact settings.
- spread_search.csv / total_search.csv: all tested subsets, MAE/RMSE, development
  MAE gain versus full model and number of seasons improved.
- spread_selection.json / total_selection.json: frozen feature lists and final
  comparison, including retained/removed features.
- *_holdout_*.parquet: per-game predictions, actuals and ensemble SD.
- *_holdout_by_season.csv: yearly accuracy.

Per-week subset predictions are cached with input arrays, names, model source,
seeds and fit settings. Interrupted runs replay search using these caches instead
of repeating completed fits. Results are exploratory; no best features have been
identified until the run completes. Changing source/data/settings invalidates
the applicable caches.

## Repository cleanup review

Nothing was deleted. Untracked does NOT mean unused.

Archive/remove candidates:
- joint_backtest.log, joint_backtest_fixed_20260908.log, joint_crash.log:
  old diagnostics; archive if you want the failure history.
- joint_backtest_latest.log: active run output; do NOT remove while it runs.
- benchmark_modelo.py and MODELO_PERFORMANCE.md: historical performance benchmark
  and notes; optional archive, not needed by the joint training path.
- feature_scan.py, feature_scan_data.py, feature_scan_report.py, tests/test_feature_scan.py:
  a separate older feature-research workflow. Archive as a set only if retiring
  that workflow. Keep FEATURE_SCAN.md until its mixed research notes are reviewed.

Keep:
- joint_scoring.py, shared_scoring.py, backtester.py, playoff_importance.py,
  modelo_workers.py, optimize_picks.py, edge_scan.py, weekly_packet.py, utils.py:
  current research/report/model dependencies. In particular, optimize_picks
  imports PRODUCTION_FEATURES from edge_scan, so edge_scan is NOT disposable.
- tests/test_*.py: regression coverage, not temporary run outputs. Tests now live
  in tests/, with shared-fixture imports updated. Run from the project root with
  python -m unittest discover -s tests -t .
- Modified tracked files: existing user/project work; do not discard.
- data/cache and saved research: reusable work, especially after these crashes.

This new runner deliberately leaves those existing files and the running
backtest unchanged.
