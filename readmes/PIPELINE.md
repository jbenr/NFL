# Pipeline responsibilities

Shared optimus input options: --input-modes separate differential percentile zscore.
Separate preserves offensive and opposing-defensive raw levels. Differential
subtracts raw defense allowed from raw offense. Percentile and zscore are
parallel representations, both produced by data_crunchski_2.comp_stats from
the same per-week snapshot: every team's rolling stat is normalized against
every OTHER team in the league that same week (current-week cross-sectional
rank or z-score, never against its own history and never across concatenated
training rows), then differenced, then comp_stats' usage-scaling multiplier
is applied post-normalization. Percentile uses rank_it/rev_rank_it; zscore
uses z_it/rev_z_it; both share the same off/def/exceptions sign conventions.
home/away share the same role references. These feature transforms are
distinct from ensemble SD betting cutoffs. Single-config calls (prepare,
fit_panel, etc.) still default to plain separate inputs; run_shared_track's
sweep now defaults to all four modes. Each extra mode multiplies the number
of shared configurations; 4 modes with the default other axes means 96
configurations, before subset/confirmation fits.

- data_pullson.py / pull_weather.py: source downloads and caches.
- data_crunchski_2.py: existing historical team-stat calculations and caches.
- data_crunchski_3.py: extracted referee tendencies, matchup rows and
  training-only normalization/category preparation, shared by both models.
- model_shredski.py: original model; shared_scoring.py and joint_scoring.py:
  alternative model implementations (their existing entry points still work).
- main.py: original weekly entry point; weekly_packet.py: report rendering.
- backtester.py: chronological shared/joint model evaluation.
- optimus_prime.py: alternative feature-panel preparation, currently Phase 0.

This is an incremental extraction, not a second copy of crunchski_2.
Weather and playoff context remain in their existing modules for now.
Weekly CLI consolidation is still pending; original defaults are unchanged.

Matchup construction accepts pandas or Polars frames. Existing pandas callers
use batched NumPy arithmetic; native Polars callers use expressions. Both return
pandas for the existing normalization/model interface. Converting pandas to
Polars solely for this operation was slower in a local timing check, so it is
not forced. No end-to-end backtest speedup is claimed.

Caches are preserved. Source-aware model/selection caches may invalidate when
their preparation changes. The active optimus pipeline still uses crunchski_2.
