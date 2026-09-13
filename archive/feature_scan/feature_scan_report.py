"""Readable tables and graphics for the standalone additive-feature scan."""
import html
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_scan_data import METRICS


def team_summary(snapshots, season, week):
    snap = snapshots[(snapshots.season == season) & (snapshots.week == week)]
    result = []
    for side, data in snap.groupby('side'):
        for metric, (label, group, higher) in METRICS.items():
            higher_better = higher if side == 'off' else not higher
            ranks = data[metric].rank(ascending=not higher_better, method='min')
            if metric in ['pass_rate', 'pass_oe']:
                ranks[:] = np.nan  # Tendency is not intrinsically better or worse.
            for idx, row in data.iterrows():
                result.append(dict(team=row.team, unit='Offense' if side == 'off' else 'Defense (opponent outcomes)',
                    metric=metric, label=label, group=group, value=row[metric], league_rank=ranks.loc[idx],
                    history_games=row.history_games, opportunities=row[f'{metric}_opportunities']))
    return pd.DataFrame(result)


def write_report(outdir, summary, scores, home, teams, context, catalog, metadata):
    for filename, table in [('additive_importance', summary), ('fold_scores', scores),
                            ('home_field_diagnostics', home), ('team_summary', teams),
                            ('game_stakes', context), ('feature_catalog', catalog)]:
        table.to_csv(outdir / f'{filename}.csv', index=False)
    (outdir / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    focus = summary[(summary.track == 'market_residual') & (summary.learner == 'ridge')]
    for level in ['group', 'feature']:
        plot = focus[focus.level == level].sort_values('drop_gain', ascending=True)
        fig, ax = plt.subplots(figsize=(13, max(5, len(plot) * .29)))
        ax.barh(plot.label, plot.drop_gain, color=np.where(plot.drop_gain > 0, '#247878', '#b9564e'))
        ax.hlines(np.arange(len(plot)), plot.drop_low, plot.drop_high, color='#333333', lw=1)
        ax.axvline(0, color='#555555', lw=1)
        ax.set_xlabel('Extra MAE after removal (points); positive supports retaining the feature/group')
        ax.set_title('Value beyond the market — regularized linear screen\n95% week-bootstrap intervals; exploratory, not adjusted for multiple tests')
        ax.grid(axis='x', alpha=.15)
        fig.tight_layout()
        fig.savefig(outdir / f'{level}_importance.png', dpi=150)
        plt.close(fig)
    tables = []
    columns = ['label', 'learner', 'add_gain', 'drop_gain', 'drop_low', 'drop_high', 'drop_positive_seasons', 'assessment']
    for track, title in [('market_residual', 'Value beyond the market spread'), ('margin', 'Score-margin prediction')]:
        for level in ['group', 'feature']:
            table = summary[(summary.track == track) & (summary.level == level)].sort_values('drop_gain', ascending=False)
            tables.append(f'<details open><summary>{title}: {level}s</summary>' + table[columns].round(4).to_html(index=False) + '</details>')
    compact = teams[teams.unit == 'Offense'].pivot(index='team', columns='label', values='value')
    details = []
    for team, stats in teams.groupby('team', sort=True):
        details.append(f'<details><summary>{html.escape(team)} — stats, ranks, sample sizes</summary>' +
            stats[['unit', 'label', 'value', 'league_rank', 'history_games', 'opportunities']].round(3).to_html(index=False) + '</details>')
    body = f'''<!doctype html><html><head><meta charset="utf-8"><title>Additive NFL feature scan</title>
<style>body{{font:15px system-ui;margin:32px;color:#203038;background:#f7f9f9}}h1,h2{{color:#174e54}}
table{{border-collapse:collapse;background:white;margin:15px 0;font-size:12px}}td,th{{padding:7px;border-bottom:1px solid #dde3e3;text-align:right}}
td:first-child,th:first-child{{text-align:left}}th{{position:sticky;top:0;background:#e6efef}}img{{max-width:100%}}details{{margin:16px 0}}summary{{cursor:pointer;font-weight:600}}.wide{{overflow:auto}}</style></head><body>
<h1>Which features add predictive value?</h1>
<p>{metadata['scan_games']} research games; evaluation seasons {metadata['validation_seasons']}.
{metadata['holdout_season']} is reserved and was not used to rank features. Team summaries: {metadata['asof_season']} week {metadata['asof_week']}.</p>
<p><b>Add gain:</b> MAE improvement from adding to the home-field baseline (also spread in the market track).
<b>Drop gain:</b> extra error after removing from the full model and retraining. Positive supports retention.
Every comparison uses identical games, training-only preprocessing, and fixed model settings.</p>
<p>This is a predictive screen using ridge and small boosted trees, not a causal effect, production-neural-net validation,
or proof of profitability. Correlated inputs can substitute for one another. Intervals resample whole weeks; they are exploratory,
not corrected for multiple tests and do not capture all season dependence. Nothing is automatically removed.</p>
<p>Target: away minus home points. The market track learns the remaining error after the market's home-margin spread.
Home field is protected as a control; a constant away/home bias can also live in the intercept.
Home-minus-neutral contrasts are model associations with limited neutral-game support, not causal estimates.
With very few neutral games, contrasts can have implausible signs or be zero because a tree cannot split such a small sample.
That does not imply home advantage is absent. Compare ordinary-home fitted predictions and training home-margin averages too.
The fitted-home column is away margin for the margin track, but a market correction for the market-residual track.</p>
<h2>Home field and baseline performance</h2><div class="wide">{home.round(4).to_html(index=False)}</div>
<div class="wide">{scores[scores.variant.isin(['baseline','full','without_home_control','market_only'])].round(4).to_html(index=False)}</div>
<h2>Group scan</h2><img src="group_importance.png" alt="Group removal effects and uncertainty">
<p>Chart: ridge market-residual screen. Both learners and both targets are tabulated below.</p>
<div class="wide">{''.join(tables)}</div>
<h2>Team overview</h2><p>Prior {metadata['lookback_games']} team games, pooled event rates; excludes the target week.
Rates are fractions; CPOE and pass-over-expected are percentage points. Defense rows describe opponent outcomes.
These are descriptive rankings, not overall power ratings. Tendency fields intentionally have no good/bad rank.</p>
<div class="wide">{compact.round(3).to_html()}</div>{''.join(details)}
<h2>Game stakes</h2><p>0–1 heuristic: late-season proximity to the conference's seventh-ranked win percentage;
postseason games receive 1 for both teams. Individual components and differences are tested.
This is NOT a playoff probability, official seeding, clinch, elimination or must-win determination.
All records, including bye teams, are frozen before the week. Ties count as half a win.</p>
<div class="wide">{context.round(3).to_html(index=False)}</div>
<h2>Scope and reproducibility</h2><p>Candidates overlap existing concepts but use rebuilt raw rates,
not production percentile-rank transformations or custom QB Elo. No actual future starter identities are used.
Weather, injuries, starter changes and playoff simulations remain future candidates. Schedule spreads are historical
stored lines, not timestamped tradable quotes. ATS rates exclude pushes and zero-edge selections; no returns after odds/costs are asserted.</p>
<p><a href="https://scikit-learn.org/stable/modules/permutation_importance.html">Importance and correlated inputs</a> ·
<a href="https://www.nfl.com/standings/tie-breaking-procedures">Official playoff tiebreaking rules</a></p>
<p>Configuration and provenance: manifest.json. Per-game comparisons: scan_predictions.parquet.</p></body></html>'''
    (outdir / 'report.html').write_text(body, encoding='utf-8')
