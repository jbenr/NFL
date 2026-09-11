# Current workflow

### Referee scoring tendency and venue display

The referee group now uses a numeric pregame tendency instead of identity dummies:
the referee's mean game total over the preceding two seasons plus earlier current-season
weeks, shrunk toward the corresponding league mean with 20 games of prior weight.
The input is that shrunk mean minus the league mean. Unknown referees receive zero
adjustment. Current-week/future outcomes are excluded for every historical training row.
This is an unadjusted scoring association, not an isolated causal referee effect.
Its additive term affects both scores equally, so it contributes to totals and cancels
directly from spreads. Retrain to test the new encoding; old referee forecasts retain
their original encoding until rerun.

Home-site, stadium and field remain distinct model features and importance entries.
Their attribution components are summed into one Home field row in the packet only.

## Differential shared scoring (current default)

```sh
python shared_scoring.py --season 2026 --week 1 --inputs differential
python shared_scoring.py --season 2026 --week 1 --inputs differential --groups stadium field referee
```

There are 15 matchup inputs: raw offense stat minus raw opposing-defense stat,
then standardized using earlier training games only. This is raw-rate subtraction,
not the original production model's percentile-rank subtraction. Both game directions
use exactly the same construction and shared scoring network, trained on team points.
The existing architecture sizes hidden layers from input count (15 here versus 30 in
separate-input mode). Context inputs remain separate. Differential QB Elo retains the
existing absolute offensive rating and relative defensive proxy; that metric definition
has not been changed.

Outputs use `packet_shared_differential/` (plus context-group suffixes), retaining
both spread and total tabs. Importance lists one `diff_*` input per metric. Use
`--inputs separate` to reproduce the separate offense/defense representation.
Historical group tests accept the same flag; their differential-mode default writes
to `data/optimize_picks/shared_context_differential/`. Previous packets/results
remain untouched.

## Linked spread / total report and context groups

```sh
python shared_scoring.py --season 2026 --week 1
python shared_scoring.py --season 2026 --week 1 --groups stadium field referee
NFL_MODEL_JOBS=8 NFL_WORKERS=4 OPENBLAS_NUM_THREADS=1 python shared_research.py
```

The first command refreshes both markets from one paired ensemble. Each report has
Headline, Spread, Totals, and Feature importance tabs. The first tab embeds the
unchanged **production** headline, explicitly labeled; challenger predictions remain
in the analysis tabs. Training permutation importance is not held-out evidence.
Old previews need one rerun for total SD/attribution. New runs retain member-level
scores and explanations so presentation changes need not retrain.

Context previews have separate packet folders. Stadium indicators modify home-site
points only; field (surface/roof) and referee indicators affect both team scores.
Extra coefficients use L2 regularization (.1), categories are fitted on prior games
only, and unknown categories default to zero adjustment. These additive field/referee
effects cancel directly in the margin but add in the total; fitting them can also
change the shared matchup function. Interactions are not implemented yet.

The longer runner compares baseline, stadium, field, and referee separately using
identical weeks, seeds, 100-member ensembles, and 20-week training windows. It writes
checkpoints, MAE/MSE, separately selected spread/total rules, validation PnL/ROI
intervals, and the challenger's result under the baseline's frozen rule to
`data/optimize_picks/shared_context/`. Add `--combined` to evaluate all combinations
and leave-one-group-out comparisons. Calibration defaults to 2024, evaluation to
2025; those seasons have already been inspected, so this remains exploratory.

No actual game weather is used. Archived decision-time forecasts are still needed
for a weather group. Historical roof states and referee assignments are not
timestamp-verified, so these group results cannot establish executable bet returns.
No automatic production model or highlight changes.

## Shared scoring preview

Attribution schema 2 fixes the original export's interleaved-versus-grouped input
column mismatch. Earlier score predictions remain valid experimental outputs, but
feature attribution and importance labels do not. Rerun the preview to regenerate
explanations; old shared packets hide these charts until regenerated.

```sh
python shared_scoring.py --season 2026 --week 1
```

This is a separate, unvalidated team-points model, not the multiplier-only experiment.
One shared network receives offense and opposing-defense raw metrics, standardized
using both sides of prior training games together. Training uses each team's actual
points (MSE). Home-site status and rest differential enter through a separate additive
linear adjustment. The same scoring function predicts both teams; subtract for margin.
Neutral, identical matchups produce identical scores. Common baseline scoring cancels
in the margin. This changes the target and input representation as well as weight sharing;
it is not a controlled multiplier-only comparison.

The ensemble uses 100 members/100 epochs by default, seed 1337, and the existing CPU
worker limits. Margin variance uses paired away-minus-home predictions within each
member. Integrated gradients group offense and opponent-defense effects by metric;
CSV retains full precision. Importance is a training-sample diagnostic, not validation.
No production rules/highlights are inherited. Scores are continuous regression outputs,
not probabilities or guaranteed nonnegative score distributions.

Output: `data/results/2026_1_20/packet_shared/2026_01/`. Cached reruns do not train.
Production, symmetric packets, and the original headline table are not overwritten.

## Symmetric matchup experiment

Production is unchanged. The challenger removes the usage multiplier from away-offense
passing/rushing rank differences, matching the unweighted construction of the other
matchup. Rates, features, home-field input, training window, architecture, seed, and
ensemble size remain unchanged. This does not force equal learned contributions or
replace the network with a shared scoring model.

Preview one week (trains 100 members if not cached):

```sh
python weekly_packet.py --season 2026 --week 1 --symmetric
```

The preview goes to `data/results/2026_1_20/packet_symmetric/2026_01/`.
The original headline table and packet are not overwritten.

Longer walk-forward comparison (run locally):

```sh
NFL_MODEL_JOBS=16 NFL_WORKERS=4 OPENBLAS_NUM_THREADS=1 python main.py research --rule-scan --symmetric
```

Spread-only results go to `data/optimize_picks/symmetric_rules/`; completed weekly
forecasts are cached. Compare with `production_rules/`, including the same fixed
edge 4 / SD 2.5 rule. Any newly selected cutoffs on already-reviewed seasons are
exploratory, not fresh validation. Stadium and rest annotations are display metadata,
not new trained stadium effects.

## Existing commands

```sh
python main.py research
python main.py research --confirm-neural --neural-iterations 20
python main.py research --rescore
python main.py research --rule-scan
python main.py --packet --season 2025 --week 20
python -m unittest tests.test_optimize_picks tests.test_modelo tests.test_feature_scan
```

The normal `python main.py` spread workflow remains. Its seed is fixed; downloads
are explicit with `--refresh`. Schedule refreshes preserve other seasons.
Research now reuses `data_crunchski_2.py` through `optimize_picks.py` rather than
the separate experimental pipeline described below.

Results: `data/optimize_picks/current/report.html`, feature CSVs, cutoff grid,
per-game predictions and packets for every validation week. Use `--output` to
retain separate experiments. The weekly `--packet` command uses the real
100-member neural models for both spreads and totals, with embedded logos,
team summaries, feature importance and point attributions. Output lives under
`data/results/<season>_<week>_<lookback>/packet`.

## Research logic and limits

`--rule-scan` is the direct fixed-model test: the actual 100-member packet models,
original features, legacy 20-week calculations and seed 1337. It does not select
features. Defaults use 2024 to select an edge/SD rule and 2025 to evaluate it,
refitting on the preceding 20 regular weeks before every forecast. It compares
100 literal edge/SD combinations, including no SD ceiling, with actual stored
odds. Forecasts are checkpointed once per week and shared by every rule.
Results go to `data/optimize_picks/production_rules`, leaving earlier studies and
live highlighting policies unchanged. All rules' later-period results are
reported for exploration; the chosen rule is determined only by earlier PnL.
Do not choose a different winner from the validation table and call it validated.

- Test the existing 32 spread features plus game importance. Totals use symmetric
  levels from both offenses/defenses, combined rest/importance and home venue.
- Compare 10/20-regular-week feature windows and original versus consistently
  weighted rate denominators. Research history is separately set to 150 weeks.
- After training burn-in, use the first half of scoring weeks for paired feature
  drop tests, the next quarter for feature/calculation/cutoff selection, and the
  last quarter for frozen validation. Fit every forecast on earlier weeks only.
- Compare full sets, positive drop-test shortlists capped at 8/16 plus venue,
  and venue alone. This is a bounded search, not an exhaustive global optimum.
- Test minimum edges 0, .5, 1, 1.5, 2, 3, 4, 5 points; SD quartile cutoffs and
  **no SD filter**. Require 60 bets, without a small-sample fallback.
- Use stored side-specific American odds; explicitly count missing-price −110
  fallbacks. Risk one unit. Pushes return zero and are excluded from win rate,
  but remain in bet counts. Exact zero edges are passes.
- Require a positive validation week-bootstrap 95% ROI lower bound to label a
  candidate `PAPER QUALIFIED`; otherwise `PASS`. Never automatically promote it.

The fast model is a 15-member week-block-bootstrap Ridge challenger predicting
market residuals. Neural confirmation tests its shortlist with actual neural
fits, 100 epochs, a 20-regular-week training window and the requested ensemble
size. It recalibrates its own cutoffs and saves `neural_summary.json`. Neither
Ridge nor 20-member thresholds apply to the unchanged 100-member model.
Confirmation does not exhaustively optimize neural feature combinations.

The seasons have already been explored: this is retrospective validation, not
an untouched proof of future profit. Archived closing lines and final starting
QBs are not decision-time-verified snapshots. Actual game weather is excluded
from new total features because it is not an archived forecast.

Selection/evaluation separation follows [scikit-learn's evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).
Feature intervals are adjusted within each scan, not across every research
decision. Correlated features may substitute for each other; weak individual
importance is not an automatic deletion rule ([importance guidance](https://scikit-learn.org/stable/modules/permutation_importance.html)).
Price signs and fields follow the [nflverse schedule dictionary](https://nflreadr.nflverse.com/articles/dictionary_schedules.html).

## Home field, context and explanations

Spread target = away score minus home score; edge = target prediction + away
handicap. Total edge = predicted points minus market total. Positive edges favor
away/over. Do not add a fixed home bonus to a market that already prices venue.
The model intercept can absorb ordinary home bias; the home/neutral indicator
estimates a separate contrast from relatively few neutral games.

Game importance remains an approximate standings/tiebreak proxy, not exact
clinch probability. Records now reset per team/season, bye teams retain their
records, remaining games count fixtures, and head-to-head results stop before
the prediction week. Historical feature windows also stop before that week.

Raw files stay local. Weekly features, full panels, context and predictions are
cached in `data/cache`, keyed by settings, code identity and source metadata.
Writes are atomic; neural confirmation resumes completed weeks. The packet
command reuses saved neural predictions and explanations on identical reruns.

Headlines and packets display SD in points. Cached variance is retained for
compatibility; SD = sqrt(variance), so an exactly converted cutoff selects the
same bets. `--rescore` updates saved policies, grids and reports without fitting
models (team-feature snapshots may be rebuilt if their data cache changed).

Yellow highlights no longer mean model/market favorite disagreement or the old
hard-coded edge/variance combination. They require a matching neural model
specification (features, calculation, window, seed, ensemble size and code),
positive calibration PnL, at least 60 calibration/validation bets, positive
validation PnL and its 95% ROI lower bound, and weekly walk-forward evaluation.
Validation must predate the displayed week. Learned edge/SD limits use unrounded
values. Missing, failed or mismatched evidence means no yellow highlight. These
are research-qualified candidates, not a guarantee of profit; SD measures model
disagreement, not game-outcome risk.

Ridge contributions add to a market-plus-intercept baseline. Neural integrated
gradients use mean training features and trapezoid integration; the numerical
residual is shown explicitly. These are model explanations, not causal points.
Neural permutation importance is labeled training-sample diagnostic; use the
separate walk-forward drop tests for predictive feature evidence.

Verification note: earlier runs encountered intermittent native-process failures
(a research-process crash and a TensorFlow allocator error). Their root cause
was not established. The completed neural confirmation, cache-only reruns and
final full regression run succeeded; per-week checkpoints permit resumption.

---

# Earlier standalone experiment (retained, not the current workflow)

Run `python feature_scan.py` in the project environment. It runs offline and
creates `data/feature_scan/<timestamp>/report.html`, two importance charts,
CSV tables and reproducible per-game predictions. It does not change `modelo`
or select its live features automatically.

Defaults: 2020 warm-up data, 2021 first training season, evaluation on each of
2022/2023/2024 using only preceding seasons, and 2025 reserved for confirmation.
Team summaries and game-stakes tables are for 2025 week 20. Configure these with
`--start-season`, `--holdout-season`, `--asof-season`, `--asof-week`,
`--lookback-games` and `--jobs`. New output directories are never overwritten.

The 54 candidates cover passing/rushing efficiency, EPA, QB-adjusted EPA, CPOE,
conversions, turnovers, penalties, sacks/hits, explosive plays, play selection,
rest, standings and game stakes. The lookback is the preceding 20 **team games**,
with pooled numerators/denominators. Matchup components combine league-centered
offensive performance with opposing defensive allowances. These are consistent
new research representations, not an exact replication of production's rank
differences, weighting, or custom QB Elo. Injuries, actual future starting QBs,
weather forecasts and exact playoff leverage are not included.

Each individual feature and related group gets two controlled comparisons:

- Add it to the baseline and retrain: does it help on its own?
- Remove it from the full set and retrain: does it add beyond the others?

Positive gains mean lower held-out MAE from retaining/adding the feature. The
same games are scored in every comparison. Preprocessing is fitted only on
training games. Fixed-parameter ridge and boosted-tree models provide a fast
linear/nonlinear screen. Candidate removal changes regularization effects too;
these are predictive comparisons, not causal coefficients or proof of value
for the production neural network. Model-specific winners need confirmation.

Two targets are reported: away-minus-home margin, and the remaining margin error
beyond the stored market spread. Home venue is a protected baseline control;
the market track also controls for the spread. No fixed home boost is added to
the market line. The home indicator's coefficient/contrast is a home-versus-
neutral comparison with few neutral observations. The ordinary-home bias can
also live in the intercept; a zero/unstable contrast does not mean no home edge.

Game stakes is a provisional 0–1 proxy built from late-season proximity to the
conference seventh-place win percentage. Standings freeze before each whole
week, include bye teams, reset by season, and count ties as half-wins. Remaining
games come from scheduled regular-season fixtures. It does not implement full
NFL division/wild-card tiebreakers, exact clinch/elimination or playoff odds.
Postseason stakes equal 1 for both teams; differences are therefore zero then.
The existing `matchup_importance` function is not enabled or reused.

Week-bootstrap intervals are exploratory and not multiple-testing corrected.
Do not turn a first-pass rank or zero-crossing into an automatic keep/drop rule.
Historical schedule spreads lack decision-time snapshots and odds/transaction
costs; ATS rates are descriptive, not a profitability claim. Marginal add value
and conditional drop value often differ because correlated features substitute.

Verification: `python -m unittest -v tests.test_feature_scan`.
