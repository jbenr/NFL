"""Model 2.0 -- the weekly model's name, version, results folder, and the
spec written at the version root so packets say exactly which model made them.

Each run folder, data/results/model_2.0/{season}_{week}_{lookback}/, gets:
    packet_{yy}w{week}.html  the packet
and data/results/model_2.0/model.json gets this spec: version, code commit,
every setting and method, and the latest run's training window. README.md is
the same spec in plain English (rewritten on every run, so it always matches
the latest code).

The values come from the code and the run itself (feature lists, decay
presets, cutoffs, layer sizes, training span), not from a hand-kept copy,
so the spec can't drift from what actually ran.
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

NAME = 'Model'
# Every version the code can build, newest last. 'epa' is the only structural
# difference so far: 2.1 adds data_crunchski_3.EPA_METRICS to the input set.
# select() switches between them; nothing else in the pipeline hard-codes a
# version, so both stay runnable side by side and write to separate folders.
VERSIONS = {
    '2.0': dict(epa=False, changes=[
        'Weather inputs cut from 7 (feels-like, wind, precipitation, rain, snowfall, snow depth, indoor flag) '
        'to 3: feels-like, wind, precipitation.',
        'Renamed from "two-sided-team-points-v1"; results moved from data/results/packet_shared/ to '
        'data/results/model_2.0/.']),
    '2.1': dict(epa=True, changes=[
        'Adds four EPA inputs per team, split by play type: pass and run EPA per play, and pass and run '
        'success rate (the share of plays with positive EPA). EPA comes from nflverse\'s expected-points '
        'model, which credits down, distance and field position rather than raw yards.',
        'Split rather than combined because overall EPA per play correlates about 0.97 with its passing half '
        'alone, so a single number would bury the run signal (pass and run EPA correlate about 0.67).',
        'Everything else matches Model 2.0: same network, weather inputs, training window and lookback.']),
}
VERSION = '2.0'
LABEL = f'{NAME} {VERSION}'
ID = f'model-{VERSION}'
RESULTS = Path(f'data/results/model_{VERSION}')
CHANGES = VERSIONS[VERSION]['changes']


def select(version):
    """Switch the whole process to a model version: its label, results folder
    and input set. Call once, before building a panel -- the feature lists are
    shared module state (data_crunchski_3.use_epa)."""
    import data_crunchski_3 as dc3
    version = str(version).replace('model_', '')
    if version not in VERSIONS:
        raise ValueError(f'Unknown model version {version!r}; have {", ".join(VERSIONS)}')
    global VERSION, LABEL, ID, RESULTS, CHANGES
    VERSION = version
    LABEL, ID = f'{NAME} {VERSION}', f'model-{VERSION}'
    RESULTS = Path(f'data/results/model_{VERSION}')
    CHANGES = VERSIONS[VERSION]['changes']
    dc3.use_epa(VERSIONS[VERSION]['epa'])
    return VERSION


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
            **{market: f'edge >= {rule["diff_cutoff"]:g} points'
                       + (f' and SD <= {rule["sd_cutoff"]:.2f}' if rule.get('sd_cutoff') else ' (no SD condition)')
               for market, rule in cutoffs.items()},
            'provenance': (f'checked on {LABEL}\'s 16-season backtest (data/bt/model_2.0/2010-2025: 4363 '
                           'games). Totals: no edge in weeks 1-4 (49.1%), 55.2% from week 5 on (n=888, four of '
                           'four eras above break-even) and 60.5% in weeks 13-14. Spreads: 58.0% in weeks 13-14 '
                           'and the playoffs (n=335, four of four eras), 49.9% otherwise -- no edge. A '
                           'walk-forward check over 2014-2025 (rule chosen on prior seasons only) returns +6.3% '
                           'roi on totals and -2.2% on spreads. Break-even is 52.4%; the pick tiers on the sheet '
                           'carry these rates.'),
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
             f'{model["model"]["generated"]}, code {model["model"]["code"]}. The version-level '
             '`model.json` in this folder is rewritten on each run with the exact current settings.', '']

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
    """model.json and README.md into the version folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'model.json').write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding='utf-8')
    (RESULTS / 'README.md').write_text(readme(model), encoding='utf-8')
