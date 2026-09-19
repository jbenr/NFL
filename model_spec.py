"""Model 2.0 -- the weekly model's name, version, results folder, and the
spec written next to every run so a packet always says exactly which model
made it.

Each run folder, data/results/model_2.0/{season}_{week}_{lookback}/, gets:
    packet_{yy}w{week}.html  the packet
    model.json               this spec: version, code commit, every setting
                             and method, and this run's training window
and data/results/model_2.0/README.md is the same spec in plain English
(rewritten on every run, so it always matches the latest code).

The values come from the code and the run itself (feature lists, decay
presets, cutoffs, layer sizes, training span), not from a hand-kept copy,
so the spec can't drift from what actually ran.
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

NAME = 'Model'
VERSION = '2.0'
LABEL = f'{NAME} {VERSION}'
ID = f'model-{VERSION}'
RESULTS = Path(f'data/results/model_{VERSION}')

CHANGES = [
    'Weather inputs cut from 7 (feels-like, wind, precipitation, rain, snowfall, snow depth, indoor flag) '
    'to 3: feels-like, wind, precipitation.',
    'Renamed from "two-sided-team-points-v1"; results moved from data/results/packet_shared/ to '
    'data/results/model_2.0/.',
]


def run_folder(season, week, lookback):
    return RESULTS / f'{season}_{week}_{lookback}'


def code_version():
    """Short git commit, flagged when tracked files have uncommitted edits."""
    repo = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo, capture_output=True,
                                text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo,
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    return (commit or 'unknown') + (' + uncommitted changes' if dirty else '')


def spec(*, season, week, lookback, train_window, iterations, epochs, seed, calculation, features,
         training, weather_file, forecast_file):
    """Everything about this run. training: the games the model was fit on
    (one row per game, with season/week)."""
    import data_crunchski_2 as dc2
    import data_crunchski_3 as dc3
    from weekly_packet import HIGH_CONFIDENCE_CUTOFFS, pretty

    network = [f for f in features]
    n = len(network)
    layers = [n, (n + 1) // 2, max(1, (n + 1) // 3)]
    weeks = training[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])
    first, last = weeks.iloc[0], weeks.iloc[-1]
    decay = dc2.DECAY_PRESETS.get(calculation, {})
    cutoffs = HIGH_CONFIDENCE_CUTOFFS

    def modified(path):
        path = Path(path) if path else None
        return (f'{path} (updated {datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M})'
                if path and path.exists() else str(path) if path else 'none')

    return {
        'model': {'name': NAME, 'version': VERSION, 'id': ID, 'code': code_version(),
                  'generated': f'{datetime.now():%Y-%m-%d %H:%M}'},
        'run': {'season': int(season), 'week': int(week), 'ensemble_members': int(iterations),
                'epochs': int(epochs), 'seed': int(seed),
                'training_games': int(len(training)),
                'training_span': f'{int(first.season)} week {int(first.week)} to {int(last.season)} week {int(last.week)}',
                'training_window': f'the last {train_window} regular-season weeks before this one (playoff weeks in '
                                   'between included); games without a final score are left out'},
        'predicts': ('Each team\'s points, with one network shared by both teams (one input row per team per '
                     'game). Spread = away points minus home points; total = their sum. The published numbers are '
                     'averages across the ensemble; SD is the standard deviation of the members\' spreads (or totals).'),
        'inputs': {
            'team_stats': {
                'metrics': [pretty(m) for m in dc3.METRICS],
                'per_team_row': (f'{len(dc3.METRICS)} offense inputs (this team\'s offense vs the opponent\'s defense) '
                                 f'and {len(dc3.METRICS)} defense inputs (this team\'s defense vs the opponent\'s offense)'),
                'lookback': f'the previous {lookback} regular-season weeks of play-by-play',
                'recency_weighting': {'preset': calculation, **decay},
                'opponent_adjustment': ('each stat is turned into a league z-score within the window; an input is the '
                                        'team\'s z-score minus the opponent\'s'),
                'usage_scaling': ('pass and run inputs are multiplied by the offense\'s pass or run rate + 0.5, the '
                                  'same way for both teams'),
                'qb_elo': ('the scheduled starter\'s recency-weighted rating from his own plays in the window '
                           '(decay: 160 days, steepness 4, floor 0.4)'),
            },
            'weather': {
                'inputs': list(dc3.MODEL_WEATHER),
                'sources': {'played games': 'Open-Meteo historical reanalysis at kickoff', 'upcoming games':
                            'latest Open-Meteo forecast for kickoff'},
                'indoor_games': 'dome or closed roof: fixed 72°F feels-like, 0 mph wind, 0 in precipitation',
                'retractable_roof_not_listed': 'treated as outdoors',
            },
            'context': {
                'home_field': '1 for the home team, 0 for the away team and at neutral sites; learned weight x value',
                'rest_advantage': 'this team\'s rest days minus the opponent\'s; learned weight x value',
            },
            'preparation': ('network inputs are standardized with the training games\' mean and standard deviation; '
                            'missing values are filled with the training median. Home field and rest are not standardized.'),
            'network_input_count': n,
        },
        'network': {
            'layers': f'dropout 0.10, dense {layers[0]} (ELU), dense {layers[1]} (ELU), dense {layers[2]} (ELU), dense 1',
            'context_layer': 'home field and rest go through a separate linear layer with no bias, added to the output',
            'target': 'team points minus the training average',
            'loss': 'mean squared error',
            'optimizer': 'Adam (amsgrad), learning rate halved after 5 epochs without improvement',
            'batch_size': 32,
            'members': f'{iterations} independently trained networks (seeds {seed} to {seed + iterations - 1})',
        },
        'picks': {
            'spread': f'edge >= {cutoffs["spread"]["diff_cutoff"]:g} points and SD <= {cutoffs["spread"]["sd_cutoff"]:.2f}',
            'total': f'edge >= {cutoffs["total"]["diff_cutoff"]:g} points and SD <= {cutoffs["total"]["sd_cutoff"]:.2f}',
            'provenance': ('cutoffs were calibrated on the 2024 two-sided backtest and checked on 2025 '
                           '(data/bt/two_sided). Those backtests used the "steep" recency preset and 7 weather inputs, '
                           f'so they have not been re-validated for {LABEL}.'),
        },
        'attributions': {
            'method': 'integrated gradients (64 steps), per member, averaged across the ensemble',
            'reference': ('an all-zero standardized input, i.e. the average training game (every input at its training '
                          'mean); home field and rest are measured from a neutral site with equal rest'),
            'spread': 'away team\'s row minus home team\'s row; totals add the two rows',
            'offense_defense_bars': ('grouped by input slot, not by matchup: the "offense" bar nets each team\'s '
                                     'offense-vs-opposing-defense inputs to its own score, so it also carries the home '
                                     'offense against the away defense'),
            'weather': ('both teams get the same weather, so weather moves the spread only through how it combines with '
                        'each team\'s stats; these spread effects vary a lot between members'),
        },
        'feature_importance': ('permutation test per member: shuffle one input across 128 random training rows and '
                               'record how much the squared error rises; averaged across members. A training-sample '
                               'diagnostic, not held-out evidence.'),
        'display_only': ('stat values and ranks on the page (plain averages over the lookback), stadium, surface and '
                         'kickoff time are shown for reference and are not model inputs'),
        'data': {'play_by_play': 'nflverse (nfl_data_py), data/pbp/pbp_{season}.parquet',
                 'schedule_and_lines': modified('data/sched.parquet'),
                 'historical_weather': modified(weather_file), 'forecast_weather': modified(forecast_file)},
        'changes_from_previous_version': CHANGES,
    }


def readme(model):
    """The spec as plain-English Markdown."""
    lines = [f'# {model["model"]["name"]} {model["model"]["version"]}', '',
             f'Latest run: {model["run"]["season"]} week {model["run"]["week"]}, generated '
             f'{model["model"]["generated"]}, code {model["model"]["code"]}. Every run folder has its own '
             '`model.json` with the exact settings used for that week.', '']

    def section(title, value, depth=0):
        pad = '  ' * depth
        if isinstance(value, dict):
            if title:
                lines.append(f'{pad}- **{title}:**')
            for key, item in value.items():
                section(key.replace('_', ' ').capitalize(), item, depth + (1 if title else 0))
        elif isinstance(value, list):
            lines.append(f'{pad}- **{title}:** ' + ', '.join(str(v) for v in value) if all(len(str(v)) < 40 for v in value)
                         else f'{pad}- **{title}:**')
            if not all(len(str(v)) < 40 for v in value):
                lines.extend(f'{pad}  - {v}' for v in value)
        else:
            lines.append(f'{pad}- **{title}:** {value}')

    for key in ['run', 'predicts', 'inputs', 'network', 'picks', 'attributions', 'feature_importance',
                'display_only', 'data', 'changes_from_previous_version']:
        lines += [f'## {key.replace("_", " ").capitalize()}', '']
        value = model[key]
        if isinstance(value, str):
            lines.append(value)
        else:
            section('', value)
        lines.append('')
    return '\n'.join(lines)


def write(folder, model):
    """model.json into the run folder; README.md into the version folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'model.json').write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding='utf-8')
    (RESULTS / 'README.md').write_text(readme(model), encoding='utf-8')
