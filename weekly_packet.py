"""Self-contained, printable weekly model packets using existing team assets."""
import base64
import json
from functools import lru_cache
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


STYLE = '''
body{font:15px/1.5 system-ui,sans-serif;color:#203039;background:#f1f3f1;margin:0}
main{max-width:1050px;margin:auto;padding:40px 24px}h1,h2,h3{font-weight:600;line-height:1.2}
h1{font-size:34px;margin:8px 0}h2{font-size:24px}h3{font-size:17px}.eyebrow{letter-spacing:2px;text-transform:uppercase;font-size:12px;color:#64796f}
.muted,small{color:#63716d}.card{background:white;border:1px solid #dce3dd;border-radius:10px;padding:25px;margin:22px 0;break-inside:avoid}
.teams{display:flex;align-items:center;gap:15px}.logo{width:52px;height:52px;object-fit:contain}.metrics{display:flex;flex-wrap:wrap;gap:30px;margin:20px 0}.metric strong{display:block;font-size:25px;font-weight:550}
.pill{display:inline-block;background:#edf1ec;border-radius:4px;padding:4px 9px;font-size:12px;font-weight:600}.warn{border-left:3px solid #b18b4f;padding:10px 16px;background:#faf7f0}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px 10px;text-align:right;border-bottom:1px solid #edf0ed}td:first-child,th:first-child{text-align:left}a{color:#236950}
.columns{display:grid;grid-template-columns:1.2fr 1fr;gap:28px}.barrow{display:grid;grid-template-columns:180px 1fr 55px;align-items:center;gap:8px;font-size:12px;margin:7px 0}.track{position:relative;height:12px;background:linear-gradient(90deg,#f7f4ef 50%,#eef4f0 50%)}.bar{position:absolute;height:12px;background:#30765d}.negative{background:#b17f51}.value{text-align:right;font-variant-numeric:tabular-nums}
@media(max-width:760px){.columns{grid-template-columns:1fr}.barrow{grid-template-columns:145px 1fr 45px}main{padding:20px 12px}}
@media print{body{background:white}main{padding:0}.card{border-radius:0;page-break-inside:avoid}a{color:inherit}.no-print{display:none}}
'''

STYLE += '''
body{font-family:Arial,sans-serif;background:white;color:#222}
main{max-width:1150px;padding:24px}h1{font-size:26px}h2{font-size:21px}
.card{border:2px solid #aaa;border-radius:0;padding:20px;margin:20px 0}
td,th{border:1px solid #bbb;padding:9px 12px}th{background:#eee}
.pill{border-radius:0}.eyebrow{letter-spacing:0}.columns{display:block}
.rank{color:#666;font-size:12px;margin-left:8px}.contribution{position:relative;height:26px;background:linear-gradient(90deg,#f5eee6 50%,#edf4ef 50%)}
.contribution:after{content:"";position:absolute;left:50%;height:100%;border-left:1px solid #aaa}
.contribution .bar{height:26px;opacity:.45}.contribution b{position:relative;z-index:1;display:block;text-align:center;line-height:26px;font-size:12px}
.reconcile{border-top:2px solid #999;padding-top:10px;text-align:right}
details{margin:12px 0;font-size:13px;color:#666}summary{cursor:pointer}
.packet{max-width:800px;background:#111;color:#eee}
body:has(.packet){background:#111}
.packet .card{border:0;border-top:1px solid #353535;background:transparent;padding:24px 0}
.packet .muted,.packet small,.packet details,.packet .rank{color:#aaa}
.packet a{color:#8bb9ff}.packet .pill{background:#292929;color:#ccc}
.packet .warn{background:#28231b;color:#eedbb9}
.packet-tabs{display:flex;gap:6px;border-bottom:1px solid #444;margin:0 0 20px;position:sticky;top:0;background:#111;z-index:5;padding:10px 0}
.packet-tabs a{padding:8px 12px;text-decoration:none;color:#aaa;font-size:14px}
.packet-tabs a[aria-current="page"]{color:white;border-bottom:3px solid #8bb9ff}
.headline-frame{width:100%;height:80vh;border:0;background:white}
main.headline-shell{max-width:1500px}
.packet th{background:#222}.packet td,.packet th{border-color:#333}
.matchup{margin:14px 0 18px}
.matchup-head,.stat-line{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,2fr) minmax(0,1fr);align-items:center;gap:10px}
.matchup-head{text-align:center;margin-bottom:12px;font-size:18px;font-weight:600}
.matchup-head .side{font-size:14px;text-align:left}.matchup-head .side:last-child{text-align:right}
.matchup-head .logo{width:36px;height:36px;display:block;margin-bottom:6px}
.matchup-head .side:last-child .logo{margin-left:auto}
.stat-row{padding:6px 0;border-bottom:1px solid #ffffff09}.stat-line{font-size:14px;grid-template-columns:170px 90px minmax(0,1fr) 90px}
.stat-name{text-align:left;font-size:12px;margin:0}.stat-number{font-variant-numeric:tabular-nums;white-space:nowrap}
.stat-number:last-child{text-align:right}.stat-number .rank{display:inline;margin-left:6px;font-size:11px}
.packet .contribution{height:20px;background:#ffffff05}
.packet .contribution:after{top:-5px;height:calc(100% + 10px);border-color:#ddd}
.matchup-head .net-bar{margin:0}
.packet .contribution .bar{height:20px;opacity:1;box-shadow:inset 0 0 0 1px #ffffff35}
.packet .contribution b{font-size:11px;line-height:20px;color:white;text-shadow:0 1px 2px black,0 0 3px black}
.packet .card{padding:14px 0}.packet .metrics{margin:12px 0;gap:24px}
.stat-bars{display:flex;gap:8px;height:5px;margin-top:12px}
.stat-bars span{border-radius:4px;min-width:0}
.stat-bars .left{background:var(--left-color,#888)}.stat-bars .right{background:var(--right-color,#888)}
.stat-bars span{box-shadow:0 0 0 1px #ffffff40}
.stat-impact{text-align:center;font-size:12px;color:#aaa;margin-top:8px}
.stat-impact.left,.stat-impact.right{color:#ddd}
.stat-impact:before{content:"";display:inline-block;width:9px;height:9px;margin-right:6px;border-radius:50%;box-shadow:0 0 0 1px #ffffff60}
.stat-impact.left:before{background:var(--left-color,#888)}.stat-impact.right:before{background:var(--right-color,#888)}
.packet .reconcile{font-size:14px;border-color:#444}
.game-heading{display:flex;align-items:center;gap:9px;margin:0 0 10px}
.game-heading .logo{width:34px;height:34px}.game-heading h2{margin:0;font-size:20px}
.game-heading .pill{margin-left:auto;font-size:10px}
.game-line{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:14px;padding-bottom:12px;border-bottom:1px solid #333}
.game-line span{white-space:nowrap}.game-line small{margin-right:5px}
.scoreboard{display:flex;justify-content:center;align-items:center;gap:0;margin:5px 0;font-variant-numeric:tabular-nums}
.scoreboard .score-team{padding:5px 9px;border-radius:2px;font-size:16px;font-weight:700;color:white;text-shadow:0 1px 2px #000}
.scoreboard .score{padding:0 9px;font-size:27px;font-weight:800;line-height:1.15}
.scoreboard .score-dash{color:#aaa;font-size:15px;padding:0 2px}
.qb-line{display:flex;flex-wrap:wrap;gap:8px 24px;color:#bbb;font-size:12px;margin:-2px 0 9px}
.match-banner{display:grid;grid-template-columns:150px minmax(0,1fr) 150px;gap:14px;align-items:center;padding:8px 0 16px;border-bottom:1px solid #333}
.banner-team .identity{display:flex;align-items:center;gap:8px;font-size:25px;font-weight:700}
.banner-team .logo{width:62px;height:62px}.banner-team:last-child{text-align:right}
.banner-team:last-child .identity{justify-content:flex-end}
.banner-qb{font-size:12px;color:#bbb;line-height:1.5}
.banner-center{text-align:center}.banner-center .game-line{justify-content:center;border:0;padding:7px 0;gap:8px 14px;font-size:13px}
.banner-total{font-size:12px;color:#ccc;margin-top:6px}
@media(max-width:650px){.match-banner{grid-template-columns:1fr 1fr}.banner-center{grid-column:1/-1;grid-row:2}.banner-team .identity{font-size:22px}.banner-team .logo{width:48px;height:48px}}
.context-row{margin:12px 0}.context-value{font-size:12px;line-height:1.35;overflow-wrap:anywhere;color:#ccc}
.context-value:last-child{text-align:right}.matchup-head{grid-template-columns:170px 90px minmax(0,1fr) 90px;font-size:15px}
@media(max-width:650px){.stat-line,.matchup-head{grid-template-columns:minmax(90px,1.3fr) 65px minmax(75px,1fr) 65px;gap:5px}.stat-line{font-size:12px}.stat-name{font-size:11px}.stat-number .rank{display:block;margin:0}.matchup-head .side{font-size:11px}}
@media(max-width:600px){main.packet{padding:18px 16px}.stat-line{font-size:16px}.stat-name{font-size:14px}.packet h1{font-size:23px}.packet .metrics{gap:18px}.packet .metric strong{font-size:22px}}
@media print{body:has(.packet),.packet{background:white;color:#222}.packet .card{border-color:#aaa}.stat-impact.left,.stat-impact.right{color:#333}}
'''


@lru_cache(maxsize=1)
def team_colors():
    path = Path('data/logos/team_colors.json')
    return json.loads(path.read_text()) if path.exists() else {}


def team_color(team):
    aliases = {'LA': 'LAR', 'STL': 'LAR', 'SD': 'LAC', 'OAK': 'LV'}
    color = team_colors().get(aliases.get(team, team), '#888888')
    return color if len(color) == 7 and color[0] == '#' and all(c in '0123456789abcdefABCDEF' for c in color[1:]) else '#888888'


def display_stats(season, week, lookback):
    """Unweighted observed rates, strictly pregame; never model inputs."""
    import data_crunchski_2 as dc
    import utils
    sources = [__file__, 'data_crunchski_2.py', 'data/sched.parquet']
    sources += list(Path('data/pbp').glob('pbp_*.parquet'))
    cached = utils.cache_path('packet_stats', [int(season), int(week), lookback], sources)
    if cached.exists():
        return pd.read_parquet(cached)
    sched = pd.read_parquet('data/sched.parquet').replace(
        {'away_team': dc.RELOCATED_TEAMS, 'home_team': dc.RELOCATED_TEAMS})
    prior = sched[(sched.season < season) | ((sched.season == season) & (sched.week < week))]
    weeks = prior.groupby(['season', 'week']).game_type.apply(lambda x: x.eq('REG').any()).sort_index()
    regular = weeks[weeks].index
    if not len(regular):
        return pd.DataFrame()
    start = regular[-min(lookback, len(regular))]
    selected = weeks.loc[start:].index
    frames = []
    for year in selected.get_level_values(0).unique():
        path = Path(f'data/pbp/pbp_{year}.parquet')
        if not path.exists():
            return pd.DataFrame()  # Do not present partial-history league ranks.
        data = pd.read_parquet(path)
        frames.append(data[data.week.isin([w for s, w in selected if s == year])])
    plays = pd.concat(frames)
    result = dc.calc_stats(plays.drop(columns='game_date', errors='ignore')).reset_index()
    qb, defense = dc.calc_qb_elo(plays, sched)
    # Current scheduled starters; most recent scheduled starter for teams on bye.
    known = sched[(sched.season < season) | ((sched.season == season) & (sched.week <= week))]
    starters = pd.concat([known[['season', 'week', f'{side}_team', f'{side}_qb_name']].rename(
        columns={f'{side}_team': 'team', f'{side}_qb_name': 'name'}) for side in ['away', 'home']])
    starters = starters.dropna(subset=['name']).sort_values(['season', 'week']).drop_duplicates('team', keep='last')
    starters['team'] = starters.team.replace(dc.RELOCATED_TEAMS)
    starters['name'] = starters.name.map(lambda n: utils.strip_suffix(f'{n.split()[0][0]}.{n.split()[1]}'))
    qb['name'] = qb.name.map(utils.strip_suffix)
    ratings = starters.merge(qb, on='name', how='left')[['team', 'weighted_qb_elo']].rename(
        columns={'weighted_qb_elo': 'off_qb_elo'})
    result = result.merge(ratings, on='team', how='left', validate='one_to_one').merge(
        defense, on='team', how='left', validate='one_to_one')
    utils.save_parquet(result, cached)
    return result


def stat_cell(stats, team, unit, metric, rank_before=False):
    column = f'{unit}_{metric}'
    if stats.empty or column not in stats or team not in stats.team.values:
        return '—'
    values = stats.set_index('team')[column].replace([np.inf, -np.inf], np.nan)
    value = values.loc[team]
    if pd.isna(value):
        return '—'
    formatted = f'{value:.1%}' if '%' in metric or metric.endswith('_pp') else f'{value:.1f}'
    lower_off = metric in ['turnovers_pp', 'stuff_%', 'sack_%', 'qb_hit_%', 'penalties_pp']
    ascending = lower_off if unit == 'off' else not lower_off
    if metric == 'penalties_pp':
        ascending = True  # Fewer flags, for either possession-based unit.
    if metric == 'qb_elo':
        ascending = unit == 'def'  # Offense higher first; defense lower first.
    rank = values.rank(method='min', ascending=ascending).loc[team]
    badge = f'<span class="rank">(#{int(rank)})</span>'
    return badge + ' ' + formatted if rank_before else formatted + badge


def point_bar(value, scale, row, left_team=None):
    if row.get('market') == 'total':
        width = 44 * abs(value) / max(scale, .01)
        edge = 50 + width if value >= 0 else 50 - width
        start = 50 if value >= 0 else edge
        color = '#3d9275' if value >= 0 else '#b67f48'
        anchor = 'none' if value >= 0 else 'translateX(-100%)'
        signed = value if abs(value) >= .05 else 0
        return (f'<div class="contribution"><span class="bar" style="background:{color};left:{start:.2f}%;width:{width:.2f}%"></span>'
                f'<b style="position:absolute;left:{edge:.2f}%;transform:{anchor};padding:0 3px">{signed:+.1f}</b></div>')
    if row.get('market') == 'spread':
        width = 44 * abs(value) / max(scale, .01)
        favored = row.away_team if value > 0 else row.home_team
        toward_left = favored == (left_team or row.away_team)
        edge = 50 - width if toward_left else 50 + width
        start = 50 - width if toward_left else 50
        signed = -value if abs(value) >= .05 else 0
        anchor = 'translateX(-100%)' if toward_left else 'none'
        return (f'<div class="contribution"><span class="bar" style="background:{team_color(favored)};left:{start:.2f}%;width:{width:.2f}%"></span>'
                f'<b style="position:absolute;left:{edge:.2f}%;transform:{anchor};padding:0 3px">{signed:+.1f}</b></div>')
    if row.get('market') == 'spread':
        label = row.away_team if value > 0 else row.home_team
    else:
        label = 'Over' if value > 0 else 'Under'
    label = f'{label} {abs(value):.2f}' if value else '0.00'
    width = 49 * abs(value) / max(scale, .01)
    left = 50 if value >= 0 else 50 - width
    if left_team is not None:
        favored = row.away_team if value > 0 else row.home_team
        left = 50 - width if favored == left_team else 50
    color = f'background:{team_color(row.away_team if value > 0 else row.home_team)};' if row.get('market') == 'spread' else ''
    return (f'<div class="contribution"><span class="bar {"negative" if value < 0 else ""}" '
            f'style="{color}left:{left:.2f}%;width:{width:.2f}%"></span><b>{escape(label)} pts</b></div>')


def comparison_bars(stats, offense, defense, metric):
    """Bar lengths compare observed rates, not ranks or model attribution."""
    if metric == 'qb_elo' or stats.empty:
        return ''
    data = stats.set_index('team')
    left = data.loc[offense].get(f'off_{metric}', np.nan) if offense in data.index else np.nan
    right = data.loc[defense].get(f'def_{metric}', np.nan) if defense in data.index else np.nan
    if not np.isfinite([left, right]).all() or min(left, right) < 0:
        return ''
    share = left / (left + right) if left + right else .5
    return (f'<div class="stat-bars" aria-label="Observed rate comparison">'
            f'<span class="left" style="flex:{share:.6f}"></span>'
            f'<span class="right" style="flex:{1-share:.6f}"></span></div>')


@lru_cache(maxsize=1)
def packet_schedule():
    import data_crunchski_2 as dc
    return pd.read_parquet('data/sched.parquet').replace(
        {'away_team': dc.RELOCATED_TEAMS, 'home_team': dc.RELOCATED_TEAMS})


def context_cells(feature, row):
    label = pretty(feature)
    if feature == 'context_importance':
        cells = [f'{row[side + "_importance"]:.0%}' if pd.notna(row.get(side + '_importance')) else '—'
                 for side in ['away', 'home']]
        return 'Playoff leverage' if 'importance_method' in row else 'Game importance', cells[0], cells[1]
    if feature == 'context_weather':
        temp, wind = row.get('weather_temperature_f'), row.get('weather_wind_mph')
        return ('Weather', f'{temp:.0f} °F' if pd.notna(temp) else 'Temp unknown',
                f'{wind:.0f} mph' if pd.notna(wind) else 'Wind unknown')
    if feature not in ['away_rest_adv', 'home_field_adv', 'context_referee'] or 'season' not in row:
        return label, '', ''
    sched = packet_schedule()
    match = sched[(sched.season == row.season) & (sched.week == row.week) &
                  (sched.away_team == row.away_team) & (sched.home_team == row.home_team)]
    if match.empty:
        return label, '', ''
    game = match.iloc[0]
    if feature == 'context_referee':
        name = game.get('referee')
        name = str(name) if pd.notna(name) else 'Unassigned'
        average, count = row.get('referee_avg_total'), row.get('referee_prior_games')
        detail = f'Avg {average:.1f} · n={int(count)}' if pd.notna(average) and pd.notna(count) else ''
        return 'Referee', name, detail
    if feature == 'home_field_adv':
        venue = game.get('stadium')
        site = ' · '.join(str(v) for v in [game.get('location'), game.get('roof'), game.get('surface')] if pd.notna(v))
        return label, str(venue) if pd.notna(venue) else '—', site
    away, home = game.get('away_rest'), game.get('home_rest')
    if pd.isna(away) or pd.isna(home):
        return label, '', ''
    return f'Rest (Δ {away-home:+g}d)', f'{row.away_team} {away:g}d', f'{row.home_team} {home:g}d'


def matchup_attribution(row, stats, panel, shared=False, differential=False):
    direction = 1 if row.get('market') == 'total' else -1
    values = {c[5:]: float(row[c]) for c in row.index if c.startswith('attr_') and pd.notna(row[c])}
    # Display-only grouping; model inputs, CSV components, and importance stay separate.
    home_keys = ['home_field_adv', 'context_stadium', 'context_field']
    if any(key in values for key in home_keys):
        home_total = sum(values.pop(key, 0) for key in home_keys)
        values['home_field_adv'] = home_total
    scale = max([abs(v) for v in values.values()] + [.01])
    net_scale = max([abs(sum(v for f, v in values.items() if f.startswith(prefix)))
                     for prefix in ['away_off_', 'away_def_']] + [.01])
    sections, used = [], set()
    order = ['qb_elo', 'pass_ypp', 'pass_completion_%', 'explosive_pass_%', 'sack_%', 'qb_hit_%',
             'run_ypp', 'explosive_run_%', 'stuff_%', 'first_down_pp', 'series_success_%',
             'third_down_%', 'fourth_down_%', 'turnovers_pp', 'penalties_pp']
    for prefix, offense, defense in [('away_off_', row.away_team, row.home_team),
                                      ('away_def_', row.home_team, row.away_team)]:
        features = sorted([f for f in values if f.startswith(prefix)],
                          key=lambda f: (order.index(f[len(prefix):]) if f[len(prefix):] in order else len(order), f))
        if not features:
            continue
        rows = []
        away_unit, home_unit = ('off', 'def') if prefix == 'away_off_' else ('def', 'off')
        for feature in features:
            metric = feature[len(prefix):]
            cells = [stat_cell(stats, row.away_team, away_unit, metric),
                     stat_cell(stats, row.home_team, home_unit, metric, rank_before=True)]
            label = 'QB Elo / allowed' if metric == 'qb_elo' else pretty(metric)
            value = values[feature]
            rows.append(f'<div class="stat-row"><div class="stat-line"><div class="stat-name">{escape(label)}</div><div class="stat-number">{cells[0]}</div>'
                        f'{point_bar(value, scale, row, left_team=row.away_team)}'
                        f'<div class="stat-number">{cells[1]}</div></div></div>')
            used.add(feature)
        net = direction * sum(values[f] for f in features)
        left_label = 'offense' if away_unit == 'off' else 'defense'
        right_label = 'offense' if home_unit == 'off' else 'defense'
        sections.append(f'<section class="matchup"><div class="matchup-head"><div></div><div class="side">{logo(row.away_team)}{escape(row.away_team)} {left_label}</div>'
                        f'<div class="net-bar" aria-label="Net matchup contribution">{point_bar(direction * net, net_scale, row)}</div>'
                        f'<div class="side">{logo(row.home_team)}{escape(row.home_team)} {right_label}</div></div>{"".join(rows)}</section>')
    other = []
    context_order = ['home_field_adv', 'context_referee', 'away_rest_adv']
    context_features = sorted((f for f in values if f not in used),
                              key=lambda f: context_order.index(f) if f in context_order else len(context_order))
    for feature in context_features:
        value = values[feature]
        label, left, right = map(escape, context_cells(feature, row))
        other.append(f'<div class="stat-line context-row"><div class="stat-name">{label}</div>'
                     f'<div class="context-value">{left}</div>{point_bar(value, scale, row)}'
                     f'<div class="context-value">{right}</div></div>')
    total = sum(values.values())
    residual = row.prediction - row.baseline - total
    input_note = ('The shared model uses raw role-specific inputs with training-only standardization. Each displayed metric sums offense and opposing-defense effects. '
                  if shared else 'Display rates are separate from the model’s recency-weighted, league-ranked inputs. ')
    if differential:
        input_note = ('Each model input is raw offense minus opposing-defense stat, standardized using prior training data. '
                      'Displayed team rates are unweighted; model rate calculations retain the legacy recency weighting. ')
    if row.get('model_family') == 'joint':
        input_note += 'Both matchups and context interact through shared weights; margin and total are independently trained targets. '
    sign_note = ('Positive raises the total; negative lowers it. ' if direction == 1 else
                 'Negative contributions favor the away team; positive favor the home team, matching the away-team spread above. ')
    return ''.join(sections + other) + (
        f'<details><summary>Calculation notes</summary>Observed, unweighted rates over the pregame feature window; #1 is best, ties share rank. '
        f'Ranks include teams on bye, using their most recent scheduled QB when needed. QB ranks: offense higher first, defense lower first. Penalty ranks: fewer possession-based flags first, not necessarily flags committed by that unit. '
        f'Defensive QB Elo measures opposing QB game production allowed, lower is better. New feature builds include relief-QB production without league-average subtraction; older cached runs retain the previous centered metric. '
        f'{input_note}'
        f'{sign_note}'
        f'Section nets sum feature contributions, not predicted team scores. Net bars share a scale with each other; feature bars share their own scale. Display rounding can affect visible sums; calculations retain full precision. '
        f'The Home field row combines home-site, stadium and field contributions for display only. Referee average totals are shrunk toward the earlier league mean with 20 prior games of weight; same-week and future results are excluded. '
        f'Contributions explain the fitted prediction, not causal effects. Numerical residual {direction * residual:+.4f} points.</details>')


def page(title, content, theme=''):
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{escape(title)}</title><style>{STYLE}</style><main class="{escape(theme)}">{content}</main></html>'


def logo(team):
    path = Path('data/logos') / f'{team}.png'
    if not path.exists():
        return ''
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f'<img class="logo" alt="{escape(team)} logo" src="data:image/png;base64,{encoded}">'


def pretty(feature):
    if feature.startswith('diff_'):
        return pretty(feature[len('diff_'):]) + ' · off − def'
    labels = {'fourth_down_%': '4th-down conversion', 'third_down_%': '3rd-down conversion',
              'pass_completion_%': 'Pass completion', 'series_success_%': 'Series success',
              'stuff_%': 'Runs stopped at / behind line', 'sack_%': 'Sacks / pass play',
              'qb_hit_%': 'QB hits / pass play', 'penalties_pp': 'Penalty flags / play',
              'first_down_pp': 'First downs / play', 'turnovers_pp': 'Turnovers / play',
              'explosive_run_%': 'Runs of 10+ yards', 'explosive_pass_%': 'Passes of 20+ yards',
              # two_sided_packet's per-dimension weather features -- context_weather
              # (below) is a different, single combined feature from an older model.
              'context_weather_feels_like_f': 'Feels like (°F)', 'context_weather_wind_mph': 'Wind (mph)',
              'context_weather_precip_inches': 'Precipitation (in)', 'context_weather_rain_inches': 'Rain (in)',
              'context_weather_snowfall_inches': 'Snowfall (in)', 'context_weather_snow_depth_inches': 'Snow depth (in)',
              'context_weather_indoor': 'Indoor'}
    if feature in labels:
        return labels[feature]
    label = feature.replace('away_off_', 'Away offense · ').replace('away_def_', 'Away defense · ')
    label = label.replace('total_', 'Combined · ').replace('_', ' ')
    for old, new in [('ypp', 'yards/play'), (' pp', '/play'), ('qb elo', 'QB Elo'),
                     ('home field adv', 'Home field'), ('away rest adv', 'Away rest advantage'),
                     ('away game importance', 'Game-importance difference')]:
        label = label.replace(old, new)
    return label[:1].upper() + label[1:]


def attribution(row):
    values = pd.Series({pretty(c[5:]): float(row[c]) for c in row.index if c.startswith('attr_')})
    order = values.abs().sort_values(ascending=False).index
    shown = values.loc[order[:8]].copy()
    if len(order) > 8:
        shown['Other features (net)'] = values.loc[order[8:]].sum()
    scale = max(shown.abs().max(), .01)
    bars = []
    for name, value in shown.items():
        width = 49 * abs(value) / scale
        left = 50 if value >= 0 else 50 - width
        bars.append(f'<div class="barrow"><span>{escape(name)}</span><div class="track"><span class="bar {"negative" if value < 0 else ""}" style="left:{left:.2f}%;width:{width:.2f}%"></span></div><span class="value">{value:+.2f}</span></div>')
    residual = float(row.prediction - row.baseline - values.sum())
    target = 'away − home margin' if row.get('market') == 'spread' else 'prediction'
    return ''.join(bars) + (f'<p class="muted">Baseline {row.baseline:+.2f} + features {values.sum():+.2f}'
                            f' + numerical residual {residual:+.4f} = {target} {row.prediction:+.2f} points.</p>')


def team_table(row, stats):
    if stats.empty:
        return '<p class="muted">Team summaries unavailable.</p>'
    stats = stats.iloc[0]
    rows = []
    for unit, metric in [('off', 'run_ypp'), ('off', 'pass_ypp'), ('off', 'series_success_%'),
                         ('off', 'qb_elo'), ('def', 'pass_ypp'), ('def', 'run_ypp')]:
        values = [stats.get(f'{side}_raw_{unit}_{metric}', np.nan) for side in ['away', 'home']]
        formatted = [f'{v:.1%}' if '%' in metric and pd.notna(v) else f'{v:.2f}' if pd.notna(v) else '—' for v in values]
        name = ('Offense · ' if unit == 'off' else 'Defense · ') + pretty(metric)
        rows.append(f'<tr><td>{escape(name)}</td><td>{formatted[0]}</td><td>{formatted[1]}</td></tr>')
    return f'<table><tr><th>Pregame team rates</th><th>{escape(row.away_team)}</th><th>{escape(row.home_team)}</th></tr>{"".join(rows)}</table>'


def game_header(row, market, action):
    away, home = escape(row.away_team), escape(row.home_team)
    lean = (away if row.edge > 0 else home) if market == 'spread' else ('Over' if row.edge > 0 else 'Under')
    market_value = f'{away} {-row.market_base:+.1f}' if market == 'spread' else f'{row.market_base:.1f}'
    model_value = f'{away} {-row.prediction:+.1f}' if market == 'spread' else f'{row.prediction:.1f}'
    items = [('Market', market_value), ('Model', model_value),
             ('Edge', f'{lean} {abs(row.edge):.1f}'), ('SD', f'{np.sqrt(row.variance):.1f}')]
    summary = ''.join(f'<span><small>{label}</small><strong>{value}</strong></span>' for label, value in items)
    scores = ''
    if pd.notna(row.get('away_points')) and pd.notna(row.get('home_points')):
        scores = (f'<div class="scoreboard" role="img" aria-label="Projected score: {away} {row.away_points:.1f} — {home} {row.home_points:.1f}">'
                  f'<span class="score-team" style="background:{team_color(row.away_team)}">{away}</span>'
                  f'<span class="score">{row.away_points:.1f}</span><span class="score-dash">-</span>'
                  f'<span class="score">{row.home_points:.1f}</span>'
                  f'<span class="score-team" style="background:{team_color(row.home_team)}">{home}</span></div>')
        if row.get('scores_implied', False) == True:
            scores = '<div class="banner-total">Implied score</div>' + scores
    total = ''
    if market == 'spread':
        market_total = row.get('total_line', np.nan)
        total_parts = [f'Market {market_total:.1f}'] if pd.notna(market_total) else []
        if pd.notna(row.get('away_points')) and pd.notna(row.get('home_points')):
            model_total = row.away_points + row.home_points
            total_parts.append(f'Model {model_total:.1f}')
            if pd.notna(market_total):
                gap = model_total - market_total
                total_parts.append(f'{"Over" if gap > 0 else "Under" if gap < 0 else "Even"} {abs(gap):.1f}')
        if total_parts:
            total = '<div class="banner-total">O/U · ' + ' · '.join(total_parts) + '</div>'
    return (f'<header class="match-banner"><div class="banner-team"><div class="identity">{logo(row.away_team)}{away}</div></div>'
            f'<div class="banner-center"><span class="pill">{escape(action)}</span><div class="game-line">{summary}</div>{scores}{total}</div>'
            f'<div class="banner-team"><div class="identity">{home}{logo(row.home_team)}</div></div></header>')


def packet_tabs(active):
    return '<nav class="packet-tabs" aria-label="Weekly report">' + ''.join(
        f'<a href="{path}"{""" aria-current="page" """ if key == active else ""}>{label}</a>'
        for key, path, label in [('headline', 'index.html', 'Headline'), ('spread', 'spread.html', 'Spread'),
                                ('total', 'total.html', 'Totals'), ('importance', 'importance.html', 'Feature importance')]) + '</nav>'


# Derived from your own two-sided backtests, not assumed: calibrated on
# 2024 (data/bt/two_sided/2024), validated out-of-sample on 2025
# (data/bt/two_sided/2025) via backtester.cutoff_grid on 2024, then
# backtester.score with that exact cutoff replayed against 2025 --
# see two_sided_diagnostics.py for the reusable version of this check.
#   spread: diff>=5.0, sd<=4.83 -- 2024 calib 56.1% n=67 +5.16u ->
#           2025 valid 57.1% n=57 +4.94u (held up, roi_95 still crosses 0)
#   total:  diff>=4.0, sd<=4.43 -- 2024 calib 57.4% n=61 +5.74u ->
#           2025 valid 61.2% n=49 +7.94u (held up better than calibration)
# Two seasons is not a lot of validation data -- rebuild this as more
# two-sided backtest seasons accumulate, don't treat it as permanent.
HIGH_CONFIDENCE_CUTOFFS = {
    'spread': dict(diff_cutoff=5.0, sd_cutoff=4.826863267174656),
    'total': dict(diff_cutoff=4.0, sd_cutoff=4.429386074841524),
}


def headline_table(folder):
    """Per-game summary across both markets -- same mechanism and column
    order as main.py's original h_to_the_tml (pandas Styler, Greens/Reds
    background_gradient on diff/sd, #ffe590 yellow highlight only on cells
    that ARE the qualifying pick), minus QB/Elo, plus O/U appended in the
    same flat style. Built from whichever {market}_details.csv this folder
    already has (each write_packets(..., market=...) call saves its own).
    'Pick' is a real qualify/pass call from HIGH_CONFIDENCE_CUTOFFS, not
    "always show a lean" -- most games should say PASS, unhighlighted."""
    from backtester import settle
    frames = {}
    for market in ['spread', 'total']:
        path = folder / f'{market}_details.csv'
        if path.exists():
            cutoffs = HIGH_CONFIDENCE_CUTOFFS[market]
            frames[market] = settle(pd.read_csv(path), cutoffs['diff_cutoff'], cutoffs['sd_cutoff'])
    if not frames:
        return ''
    base = next(iter(frames.values()))
    sched = packet_schedule()[['season', 'week', 'away_team', 'home_team', 'gameday', 'gametime']]
    games = base[['season', 'week', 'away_team', 'home_team']].merge(
        sched, on=['season', 'week', 'away_team', 'home_team'], how='left')

    def market_fields(frame, g, market):
        match = frame[(frame.away_team == g.away_team) & (frame.home_team == g.home_team)] if frame is not None else None
        if match is None or match.empty:
            return dict(line=np.nan, model=np.nan, diff=np.nan, sd=np.nan, pick=''), None
        r = match.iloc[0]
        pick = (r.away_team if r.edge > 0 else r.home_team) if market == 'spread' else ('OVER' if r.edge > 0 else 'UNDER')
        pick = pick if bool(r.qualifies) else 'PASS'
        return dict(line=r.market_base, model=r.prediction, diff=abs(r.edge), sd=r.sd, pick=pick), (pick if pick != 'PASS' else None)

    rows = []
    picks = set()
    for _, g in games.iterrows():
        spread, spread_pick = market_fields(frames.get('spread'), g, 'spread')
        total, total_pick = market_fields(frames.get('total'), g, 'total')
        for pick in [spread_pick, total_pick]:
            if pick:
                picks.add(pick)
        # Same shape as h_to_the_tml (minus qb/qb_elo): away, then this
        # market's line/model, then home, then diff/sd/pick -- O/U's own
        # line/model/diff/sd/pick block appended after, team names not repeated.
        rows.append(dict(
            gameday=g.gameday, gametime=g.gametime,
            away_logo=g.away_team, away_team=g.away_team,
            line=spread['line'], model=spread['model'],
            home_team=g.home_team, home_logo=g.home_team,
            diff=spread['diff'], sd=spread['sd'], pick=spread['pick'],
            total_line=total['line'], total_model=total['model'],
            total_diff=total['diff'], total_sd=total['sd'], total_pick=total['pick']))
    table = pd.DataFrame(rows)
    table['gameday'] = pd.to_datetime(table.gameday)
    table['gametime'] = pd.to_datetime(table.gametime, format='%H:%M', errors='coerce').dt.time
    table = table.sort_values(['gameday', 'gametime', 'away_team']).reset_index(drop=True)

    def signed(value, precision=1):
        return '—' if pd.isna(value) else f'{value:+.{precision}f}'

    def plain(value, precision=1):
        return '—' if pd.isna(value) else f'{value:.{precision}f}'

    def highlight_picks(value):
        return 'background-color: #ffe590' if value in picks else ''

    styled = (table.style.hide(axis='index')
             .background_gradient(subset=['diff', 'total_diff'], cmap='Greens')
             .background_gradient(subset=['sd', 'total_sd'], cmap='Reds')
             .map(lambda _: 'font-size: 14px; font-family: Arial; border: 1px solid gray')
             .format({
                 'gameday': lambda x: x.strftime('%a %m/%d'),
                 'gametime': lambda x: x.strftime('%I:%M %p').lstrip('0') if x else '—',
                 'line': signed, 'model': signed, 'diff': plain, 'sd': plain,
                 'total_line': signed, 'total_model': signed, 'total_diff': plain, 'total_sd': plain,
                 'away_logo': lambda x: logo(x), 'home_logo': lambda x: logo(x),
             })
             .map(highlight_picks, subset=['away_team', 'home_team', 'pick', 'total_pick'])
             .relabel_index(['Date', 'Time', 'Away logo', 'Away', 'Home', 'Home logo',
                            'Line', 'Model', 'Diff', 'SD', 'Pick',
                            'Total', 'Model', 'Diff', 'SD', 'O/U pick'], axis=1))
    return styled.to_html()


def write_packets(predictions, panel, importance, config, root):
    root = Path(root)
    market = config['market']
    for (season, week), games in predictions.groupby(['season', 'week']):
        shared = config['calculation'] in ['shared-scoring-v1', 'joint-matchup-v1', 'two-sided-team-points-v1']
        snapshot = display_stats(season, week, config['lookback']) if market == 'spread' or shared else pd.DataFrame()
        folder = root / f'{int(season)}_{int(week):02d}'
        folder.mkdir(parents=True, exist_ok=True)
        title = f'{int(season)} · Week {int(week)} · {"Against the spread" if market == "spread" else "Over / under"}'
        header = packet_tabs(market) + f'<h1>{title}</h1>'
        header += f'<p class="muted">{escape(config["model"])} · {config["lookback"]} regular-week feature window · {escape(config["calculation"])} rates</p>'
        if config.get('headline_href'):
            header += f'<p><a href="{escape(config["headline_href"], quote=True)}">Weekly headline table</a></p>'
        header += '<details><summary>Data & model notes</summary>Stored-line analysis, not executable bets. Starting-QB availability is not timestamp-verified. SD measures model disagreement, not game-outcome risk.</details>'
        if config.get('context_note'):
            header += '<details><summary>Context data</summary>' + escape(config['context_note']) + '</details>'
        cards = []
        for _, row in games.iterrows():
            lean = (row.away_team if row.edge > 0 else row.home_team) if market == 'spread' else ('OVER' if row.edge > 0 else 'UNDER')
            qualifies = bool(row.qualifies) and config['status'] == 'PAPER QUALIFIED'
            action = f'PAPER WATCH · {lean}' if qualifies else 'PASS'
            reason = config['reason'] if config['status'] != 'PAPER QUALIFIED' else ('Cutoffs met; paper tracking only' if qualifies else 'Edge or SD cutoff not met')
            stats = panel[(panel.season == season) & (panel.week == week) & (panel.away_team == row.away_team) & (panel.home_team == row.home_team)]
            direction = f'Displayed positive contributions favor {row.home_team}; negative favor {row.away_team}.' if market == 'spread' else 'Positive contributions raise the total; negative lower it.'
            outcome = f'Historical {"away − home margin" if market == "spread" else "total"}: {row.actual:.0f} points; hypothetical rule PnL {row.pnl:+.2f} units.' if pd.notna(row.actual) else 'Result pending.'
            baseline_note = config.get('baseline_note', 'Baseline includes the market and fitted intercept.')
            chart = matchup_attribution(row, snapshot, stats, shared=shared,
                    differential=config.get('input_mode') == 'differential') if market == 'spread' or shared else attribution(row)
            if market == 'total':
                chart = '<p class="muted">Total contributions · + raises scoring · − lowers scoring</p>' + chart
            if config['calculation'] == 'shared-scoring-v1' and config.get('attribution_schema', 0) < 2:
                chart = '<p class="warn">Feature labels in this saved preview need regeneration after an attribution-column correction. Scores remain usable as experimental predictions. Rerun shared_scoring.py for corrected explanations.</p>'
            cards.append(f'<section class="card">{game_header(row, market, action)}{chart}<details><summary>Pick details</summary>{escape(reason)}<p>{escape(baseline_note)} {direction}</p>{outcome} Odds: {row.odds:+.0f}{" (assumed)" if row.assumed_odds else " (stored)"}.</details></section>')
        table = importance.head(12).copy()
        table['feature'] = table.feature.map(pretty)
        note = config.get('importance_note', 'Discovery-only paired refit/drop tests. Positive MSE contribution means removing the feature increased error. Intervals are week-block bootstrap, multiplicity-adjusted across this feature scan. Correlated features can substitute for one another.')
        fi = f'<section class="card"><h2>Feature importance</h2><details><summary>Method</summary>{escape(note)}</details>' + table.to_html(index=False, float_format=lambda x: f'{x:.3f}', border=0) + f'<p><a href="{market}_importance.csv">All features</a> · <a href="{market}_details.csv">Predictions and contributions</a></p></section>'
        if config['calculation'] == 'shared-scoring-v1' and config.get('attribution_schema', 0) < 2:
            fi = '<p class="warn">Saved feature-importance labels also require regeneration. They are hidden until the shared-scoring preview is rerun.</p>'
        games.to_csv(folder / f'{market}_details.csv', index=False)
        importance.to_csv(folder / f'{market}_importance.csv', index=False)
        (folder / f'{market}_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        (folder / f'{market}.html').write_text(page(title, header + ''.join(cards), 'packet'), encoding='utf-8')
        (folder / f'{market}_importance.html').write_text(fi, encoding='utf-8')
        evidence = ''.join(f'<h2>{name.title()}</h2>' + (folder / f'{name}_importance.html').read_text()
                           for name in ['spread', 'total'] if (folder / f'{name}_importance.html').exists())
        (folder / 'importance.html').write_text(page(title, packet_tabs('importance') + evidence, 'packet'), encoding='utf-8')
        links = ''.join(f'<li><a href="{name}.html">{name.title()} packet</a></li>' for name in ['spread', 'total'] if (folder / f'{name}.html').exists())
        if config.get('headline_href'):
            links = f'<li><a href="{escape(config["headline_href"], quote=True)}">Weekly headline table</a></li>' + links
        headline = f'<h1>{int(season)} · Week {int(week)}</h1>'
        if config.get('headline_href'):
            headline += (f'<p class="muted">Original production headline table. Analysis tabs: {escape(config["model"])}.</p>'
                         f'<iframe class="headline-frame" title="Original headline table" src="{escape(config["headline_href"], quote=True)}"></iframe>')
        else:
            headline += headline_table(folder) + f'<ul>{links}</ul>'
        (folder / 'index.html').write_text(page(title, packet_tabs('headline') + headline, 'packet headline-shell'), encoding='utf-8')
        for missing in ['spread', 'total']:
            if not (folder / f'{missing}.html').exists():
                (folder / f'{missing}.html').write_text(page(title, packet_tabs(missing) +
                    '<p>This model has no saved analysis for this market yet. Regenerate the preview to add it.</p>', 'packet'), encoding='utf-8')


def write_research_report(summaries, grid, output):
    from optimize_picks import policy_sd_cutoff
    cards = []
    for market, config in summaries.items():
        cards.append(f'<section class="card"><h2>{market.title()} · {config["status"]}</h2><p>{escape(config["reason"])}</p>')
        if 'validation' in config:
            cards.append(pd.DataFrame([config['calibration'], config['validation']], index=['Calibration (selected)', 'Validation (fixed)']).to_html(float_format=lambda x: f'{x:.3f}', border=0))
            sd = policy_sd_cutoff(config)
            cards.append(f'<p>Minimum edge: {config["diff_cutoff"]} points. Maximum SD: {sd if sd is not None else "none"} points. Validation ROI 95% interval: {config["validation_roi_95"]}.</p><p>Features: {escape(", ".join(map(pretty, config["features"])))}</p>')
        cards.append('</section>')
    neural = output / 'neural_summary.json'
    if neural.exists():
        cards.append('<section class="card"><h2>Actual neural confirmation</h2><p class="muted">Separate model-specific calibration; these are not the unchanged 100-member production ensemble.</p>')
        rows = []
        for market, config in json.loads(neural.read_text()).items():
            if 'validation' in config:
                rows.append(dict(market=market, model=config['model'], status=config['status'],
                                 edge_cutoff=config['diff_cutoff'], sd_cutoff=policy_sd_cutoff(config),
                                 **config['validation']))
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table[['market', 'model', 'status', 'n', 'win_rate', 'pnl_units', 'roi']]
        cards.append(table.to_html(index=False, float_format=lambda x: f'{x:.3f}', border=0))
        cards.append('<p><a href="neural_summary.json">Neural configuration and ROI intervals</a></p></section>')
    links = ''.join(f'<li><a href="weeks/{p.parent.name}/index.html">{p.parent.name.replace("_", " · Week ")}</a></li>' for p in sorted((output / 'weeks').glob('*/index.html')))
    header = '<div class="eyebrow">NFL / Model research</div><h1>Spread & total evidence</h1><p class="warn">Retrospective validation, not guaranteed future profit. These seasons have already been explored. Keep production unchanged; paper-test any challenger prospectively. Ridge variance cutoffs do not apply to the neural ensemble.</p><p>All fits use earlier weeks only. Discovery ranks features; calibration selects feature/calculation/cutoff combinations; the final period is evaluated after choices are fixed. Prices fall back to −110 only when missing. Exact zero edges are passes; pushes return zero. Flat one-unit risk, no Kelly sizing.</p>'
    (output / 'report.html').write_text(page('NFL research', header + ''.join(cards) + f'<h2>Weekly packets</h2><ul>{links}</ul><p><a href="cutoff_grid.csv">Full calibration grid</a> · <a href="summary.json">Model-specific configuration</a></p>'), encoding='utf-8')


def neural_packet(season, week, lookback=20, iterations=100, seed=1337, symmetric=False):
    """The existing ensemble, both markets; never borrow a Ridge bet cutoff."""
    import model_shredski as ms
    import optimize_picks as op
    import utils
    panel = op.build_panel(season, week, lookback, lookback, 'mean')
    if symmetric:
        panel = op.symmetric_features(panel)
    calculation = 'mean-symmetric-v1' if symmetric else 'mean'
    output = Path(f'data/results/{season}_{week}_{lookback}/{"packet_symmetric" if symmetric else "packet"}')
    headline = None
    for market in (['spread'] if symmetric else ['spread', 'total']):
        features = [f for f in op.feature_names(panel, market) if f != 'away_game_importance']
        data = op.market_panel(panel, market)
        target = data[(data.season == season) & (data.week == week)]
        if target.empty:
            print(f'{market}: no stored market line; packet skipped')
            continue
        fingerprint = pd.util.hash_pandas_object(panel[op.KEY + features + ['away_score', 'home_score']], index=False).values.tobytes().hex()
        cached = utils.cache_path('neural_predictions', [fingerprint, market, features, iterations, seed, 100],
                                  ['model_shredski.py', 'modelo_workers.py'])
        artifacts = cached.with_suffix('')
        if not cached.exists():
            ms.modelo(panel, season, week, artifacts, bt=False, features=features, market=market,
                      iterations=iterations, random_state=seed, round_predictions=False)
            utils.save_parquet(pd.read_csv(artifacts / 'explanations.csv'), cached)
        details = pd.read_parquet(cached)
        if market == 'spread':
            headline = details[['away_team', 'home_team', 'prediction', 'variance']].copy()
            headline.attrs['model_spec'] = op.neural_spec(features, lookback, calculation, iterations, seed)
            headline.attrs['model_spec']['train_weeks'] = lookback
        predictions = target.merge(details, on=['away_team', 'home_team'], validate='one_to_one')
        predictions['edge'] = predictions.prediction - predictions.market_base
        config = dict(model=f'production-architecture-neural / {iterations} members / seed {seed}',
                      market=market, lookback=lookback, calculation=calculation, status='PASS',
                      headline_href=f'../../html_{season}_{week}_{lookback}.html',
                      reason='Neural-specific profitable cutoffs not established',
                      baseline_note='Baseline is the ensemble prediction at mean training features.',
                      importance_note='Training-sample permutation diagnostic, not held-out predictive evidence. Error bars in the CSV measure variation across ensemble members, not confidence in betting profitability.')
        importance = pd.read_csv(artifacts / 'importance.csv').sort_values('importance', ascending=False)
        write_packets(op.settle(predictions), panel, importance, config, output)
    print(f'Weekly packet: {output / f"{season}_{week:02d}" / "index.html"}')
    return headline


def refresh_packet(season, week, lookback=20, symmetric=False):
    """Render saved forecasts only: no model fits, data pulls, or headline edits."""
    root = Path(f'data/results/{season}_{week}_{lookback}/{"packet_symmetric" if symmetric else "packet"}')
    folder = root / f'{season}_{week:02d}'
    for market in ['spread', 'total']:
        details = folder / f'{market}_details.csv'
        if not details.exists():
            continue
        predictions = pd.read_csv(details)
        saved_config = folder / f'{market}_config.json'
        config = json.loads(saved_config.read_text()) if saved_config.exists() else dict(
            model='Production ensemble', market=market, lookback=lookback,
            calculation='mean', status='PASS', reason='Validated cutoffs not established',
            headline_href=f'../../html_{season}_{week}_{lookback}.html',
            baseline_note='Baseline: ensemble prediction at mean training features.',
            importance_note='Training-sample permutation diagnostic; not held-out feature evidence.')
        importance = pd.read_csv(folder / f'{market}_importance.csv')
        write_packets(predictions, predictions, importance, config, root)
    print(f'Packet refreshed: {folder / "index.html"}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build a weekly packet by fitting the chosen model '
                                     '-- writes to data/results/{season}_{week}_{lookback}/packet_shared/.')
    parser.add_argument('--model', choices=['two-sided', 'neural'], default='two-sided',
                        help="'two-sided': league z-scores, symmetric usage scaling, historical weather "
                             "(backtester.py --model two-sided's model). 'neural': the original single-network "
                             "packet (main.py --packet's model). Default: two-sided.")
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--week', type=int, required=True)
    parser.add_argument('--lookback', type=int, default=20)
    parser.add_argument('--train-window', type=int, default=100, help='Two-sided only: training REG weeks')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--jobs', type=int)
    parser.add_argument('--weather-file', help='Two-sided only: historical/reanalysis weather')
    parser.add_argument('--forecast-file', help='Two-sided only: fallback for a target week with no historical '
                        'weather yet (i.e. not played yet) -- pull it first with pull_weather.py --season ... '
                        '--week ... --mode live. Defaults to data/weather/forecasts.parquet if it exists.')
    args = parser.parse_args()
    if args.model == 'two-sided':
        from shared_scoring import two_sided_packet
        two_sided_packet(args.season, args.week, args.lookback, args.train_window, args.iterations,
                         args.epochs, args.seed, args.jobs, args.weather_file, args.forecast_file)
    else:
        neural_packet(args.season, args.week, args.lookback, args.iterations, args.seed)
