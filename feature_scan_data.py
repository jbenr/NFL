"""Pregame-only candidate data for the standalone feature scan.

Uses consistent, pooled event rates over previous team games. These are research
candidates, not the percentile-rank features used by the production ensemble.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# label, football group, whether more is better for the offense
METRICS = {
    'pass_epa': ('Passing EPA / dropback', 'Passing efficiency', True),
    'rush_epa': ('Rushing EPA / attempt', 'Rushing efficiency', True),
    'pass_ypp': ('Net passing yards / dropback', 'Passing efficiency', True),
    'rush_ypp': ('Rushing yards / attempt', 'Rushing efficiency', True),
    'completion_rate': ('Completion rate', 'Passing efficiency', True),
    'success_rate': ('Play success rate', 'Down-to-down efficiency', True),
    'first_down_rate': ('First downs / play', 'Down-to-down efficiency', True),
    'third_down_rate': ('Third-down conversion rate', 'Situational conversion', True),
    'fourth_down_rate': ('Fourth-down conversion rate', 'Situational conversion', True),
    'turnover_rate': ('Turnovers / play', 'Turnovers and penalties', False),
    'penalty_rate': ('Offensive penalties / play', 'Turnovers and penalties', False),
    'explosive_pass_rate': ('20+ yard gains / dropback', 'Explosive plays', True),
    'explosive_run_rate': ('10+ yard gains / rush', 'Explosive plays', True),
    'stuff_rate': ('Rushes gaining zero or fewer yards', 'Rushing efficiency', False),
    'sack_rate': ('Sacks / dropback', 'Pass protection and disruption', False),
    'qb_hit_rate': ('QB hits / dropback', 'Pass protection and disruption', False),
    'qb_epa': ('QB-adjusted EPA / dropback', 'Quarterback efficiency', True),
    'cpoe': ('Completion percentage over expected', 'Quarterback efficiency', True),
    'pass_rate': ('Dropbacks / offensive play', 'Play selection', True),
    'pass_oe': ('Pass tendency over expected', 'Play selection', True),
}

# Modern alignment, restricted to 2020 onward by the CLI.
DIVISIONS = ['BUF MIA NE NYJ', 'BAL CIN CLE PIT', 'HOU IND JAX TEN', 'DEN KC LV LAC',
             'DAL NYG PHI WAS', 'CHI DET GB MIN', 'ATL CAR NO TB', 'ARI LA SEA SF']
TEAM_INFO = {t: (i // 4, i) for i, teams in enumerate(DIVISIONS) for t in teams.split()}
KEYS = ['season', 'week', 'away_team', 'home_team']


def normalize_schedule(schedule):
    s = schedule.copy()
    for col in ['away_team', 'home_team']:
        s[col] = s[col].replace({'LAR': 'LA', 'OAK': 'LV', 'SD': 'LAC'})
    s = s[s.game_type.isin(['REG', 'WC', 'DIV', 'CON', 'CONF', 'SB'])]
    if s.duplicated(KEYS).any():
        raise ValueError('Duplicate schedule game keys')
    return s.sort_values(KEYS).reset_index(drop=True)


def pregame_context(schedule):
    """Freeze every standings snapshot before a whole week, including bye teams.

    Race proximity is a transparent proxy, not an official playoff probability,
    clinch, elimination, or must-win flag. No current/future scores are read.
    """
    s = normalize_schedule(schedule)
    rows = []
    for season, season_games in s.groupby('season', sort=True):
        record = {t: [0, 0, 0] for t in TEAM_INFO}  # wins, ties, played
        reg = season_games[season_games.game_type == 'REG']
        last_week = 18 if season >= 2021 else 17
        for week, games in season_games.groupby('week', sort=True):
            snapshot = {}
            for team, (conf, division) in TEAM_INFO.items():
                wins, ties, played = record[team]
                pct = (wins + 0.5 * ties) / played if played else 0.5
                remaining = int(((reg.week >= week) & ((reg.away_team == team) | (reg.home_team == team))).sum())
                snapshot[team] = dict(wins=wins, ties=ties, played=played, win_pct=pct,
                                      remaining=remaining, conf=conf, division=division)
            # All conference members are included, even on a bye.
            cutoffs = {conf: sorted([v['win_pct'] for v in snapshot.values() if v['conf'] == conf],
                                   reverse=True)[6] for conf in (0, 1)}
            leaders = {div: max(v['win_pct'] for v in snapshot.values() if v['division'] == div)
                       for div in range(8)}
            late = float(np.clip((week - last_week / 2) / (last_week / 2), 0, 1))
            for game in games.itertuples():
                row = {k: getattr(game, k) for k in KEYS}
                postseason = int(game.game_type != 'REG')
                row.update(home_field_adv=float(game.location == 'Home') if pd.notna(game.location) else np.nan,
                           rest_days_diff=float(game.away_rest - game.home_rest),
                           postseason=postseason, late_season=late,
                           division_game=int(TEAM_INFO[game.away_team][1] == TEAM_INFO[game.home_team][1]))
                for side in ['away', 'home']:
                    v = snapshot[getattr(game, f'{side}_team')]
                    cutoff_gap = v['win_pct'] - cutoffs[v['conf']]
                    division_gap = v['win_pct'] - leaders[v['division']]
                    # Historical-only, symmetric proximity score; no assumed motivation benefit.
                    race = 1.0 if postseason else late * np.exp(-abs(cutoff_gap) / 0.125)
                    row.update({f'{side}_wins': v['wins'], f'{side}_ties': v['ties'],
                                f'{side}_played': v['played'], f'{side}_win_pct': v['win_pct'],
                                f'{side}_remaining': v['remaining'], f'{side}_cutoff_gap': cutoff_gap,
                                f'{side}_division_gap': division_gap, f'{side}_stakes_proxy': race})
                for term in ['win_pct', 'remaining', 'cutoff_gap', 'division_gap', 'stakes_proxy']:
                    row[f'{term}_diff'] = row[f'away_{term}'] - row[f'home_{term}']
                row['late_cutoff_gap_diff'] = late * row['cutoff_gap_diff']
                row['late_division_gap_diff'] = late * row['division_gap_diff']
                rows.append(row)
            # Update only after constructing ALL games' features for this week.
            for game in games.itertuples():
                if game.game_type != 'REG' or pd.isna(game.away_score) or pd.isna(game.home_score):
                    continue
                for team, score, other in [(game.away_team, game.away_score, game.home_score),
                                            (game.home_team, game.home_score, game.away_score)]:
                    record[team][0] += int(score > other)
                    record[team][1] += int(score == other)
                    record[team][2] += 1
    return pd.DataFrame(rows)


def aggregate_plays(pbp):
    """Numerator and denominator per metric/team/game, before rolling history."""
    p = pbp.copy()
    for col in ['posteam', 'defteam']:
        p[col] = p[col].replace({'LAR': 'LA', 'OAK': 'LV', 'SD': 'LAC'})
    p = p[p.posteam.isin(TEAM_INFO) & p.defteam.isin(TEAM_INFO)]
    p = p[(p.qb_kneel.fillna(0) == 0) & (p.qb_spike.fillna(0) == 0)]
    dropback = p.qb_dropback.eq(1)
    rush = p.rush_attempt.eq(1) & ~dropback
    play = dropback | rush
    attempts = p.pass_attempt.eq(1) & ~p.sack.eq(1)
    third = p.third_down_converted.eq(1) | p.third_down_failed.eq(1)
    fourth = p.fourth_down_converted.eq(1) | p.fourth_down_failed.eq(1)
    specs = {
        'pass_epa': (p.epa, dropback), 'rush_epa': (p.epa, rush),
        'pass_ypp': (p.yards_gained, dropback), 'rush_ypp': (p.yards_gained, rush),
        'completion_rate': (p.complete_pass, attempts), 'success_rate': (p.success, play),
        'first_down_rate': (p.first_down, play), 'third_down_rate': (p.third_down_converted, third),
        'fourth_down_rate': (p.fourth_down_converted, fourth),
        'turnover_rate': (p.interception.fillna(0) + p.fumble_lost.fillna(0), play),
        'penalty_rate': (p.penalty_team.eq(p.posteam).astype(float), play),
        'explosive_pass_rate': (p.yards_gained.ge(20).astype(float), dropback),
        'explosive_run_rate': (p.yards_gained.ge(10).astype(float), rush),
        'stuff_rate': (p.yards_gained.le(0).astype(float), rush),
        'sack_rate': (p.sack, dropback), 'qb_hit_rate': (p.qb_hit, dropback),
        'qb_epa': (p.qb_epa, dropback), 'cpoe': (p.cpoe, attempts),
        'pass_rate': (dropback.astype(float), play), 'pass_oe': (p.pass_oe, play),
    }
    keys = ['season', 'week', 'game_id']
    values = p[keys + ['posteam', 'defteam']].copy()
    for name, (amount, mask) in specs.items():
        valid = mask & amount.notna() & np.isfinite(amount)
        values[f'{name}_num'] = amount.where(valid, 0).astype(float)
        values[f'{name}_den'] = valid.astype(float)
    parts = []
    for side, team_col in [('off', 'posteam'), ('def', 'defteam')]:
        g = values.groupby(keys + [team_col], sort=False).sum(numeric_only=True).reset_index()
        g = g.rename(columns={team_col: 'team'})
        g['side'] = side
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def rolling_snapshots(game_stats, week_keys, lookback_games=20):
    """Snapshot each team using previous weeks only; retain true measurement units."""
    history = defaultdict(list)
    groups = {k: g for k, g in game_stats.groupby(['season', 'week'])}
    out = []
    for season, week in sorted(set(week_keys) | set(groups)):
        for (team, side), games in history.items():
            recent = games[-lookback_games:]
            row = dict(season=season, week=week, team=team, side=side, history_games=len(recent))
            for metric in METRICS:
                num = sum(g[f'{metric}_num'] for g in recent)
                den = sum(g[f'{metric}_den'] for g in recent)
                row[metric] = num / den if den else np.nan
                row[f'{metric}_opportunities'] = den
            out.append(row)
        for game in groups.get((season, week), pd.DataFrame()).to_dict('records'):
            history[(game['team'], game['side'])].append(game)
    return pd.DataFrame(out)


def candidate_catalog():
    catalog = []
    for metric, (label, group, _) in METRICS.items():
        for side, matchup in [('off', 'Away offense vs home defense'), ('def', 'Away defense vs home offense')]:
            catalog.append(dict(feature=f'{side}_{metric}_edge', label=f'{matchup}: {label}',
                                group=group, description='League-centered offense plus opponent allowance; positive is oriented toward away strength, except play-selection tendencies.'))
    context = {
        'home_field_adv': ('Home venue (non-neutral)', 'Home field'),
        'rest_days_diff': ('Rest advantage: away minus home, days', 'Rest and schedule'),
        'postseason': ('Postseason game', 'Game setting'),
        'late_season': ('Late-season progress', 'Game setting'),
        'division_game': ('Division matchup', 'Game setting'),
        'win_pct_diff': ('Pregame win percentage: away minus home', 'Standings strength'),
        'remaining_diff': ('Remaining regular-season games: away minus home', 'Rest and schedule'),
        'cutoff_gap_diff': ('Conference seventh-place proximity: away minus home', 'Standings strength'),
        'division_gap_diff': ('Division-lead proximity: away minus home', 'Standings strength'),
        'stakes_proxy_diff': ('Game-stakes proxy: away minus home', 'Game stakes'),
        'away_stakes_proxy': ('Away game-stakes proxy', 'Game stakes'),
        'home_stakes_proxy': ('Home game-stakes proxy', 'Game stakes'),
        'late_cutoff_gap_diff': ('Late-season conference-race proximity difference', 'Game stakes'),
        'late_division_gap_diff': ('Late-season division-race proximity difference', 'Game stakes'),
    }
    for feature, (label, group) in context.items():
        catalog.append(dict(feature=feature, label=label, group=group,
                            description='Known before the week. Stakes are heuristic proximity scores, not playoff probabilities or official clinch flags.'))
    return pd.DataFrame(catalog)


def build_candidates(schedule, game_stats, lookback_games=20, min_history=4):
    s = normalize_schedule(schedule)
    snapshots = rolling_snapshots(game_stats, list(s[['season', 'week']].itertuples(index=False, name=None)), lookback_games)
    data = s.merge(pregame_context(s), on=KEYS, validate='one_to_one')
    # The schedule contains no engineered context names; labels refer to both teams.
    for location in ['away', 'home']:
        for side in ['off', 'def']:
            snap = snapshots[snapshots.side == side].drop(columns='side')
            snap = snap.rename(columns={c: f'{location}_{side}_{c}' for c in snap if c not in ['season', 'week', 'team']})
            data = data.merge(snap, left_on=['season', 'week', f'{location}_team'],
                              right_on=['season', 'week', 'team'], how='left', validate='many_to_one').drop(columns='team')
    for metric, (_, _, higher) in METRICS.items():
        sign = 1 if higher else -1
        league = snapshots[snapshots.side == 'off'].groupby(['season', 'week'])[metric].mean()
        center = pd.MultiIndex.from_frame(data[['season', 'week']]).map(league).to_numpy(dtype=float)
        # A high allowance is weak defense; swapping teams negates/exchanges the components.
        data[f'off_{metric}_edge'] = sign * (data[f'away_off_{metric}'] + data[f'home_def_{metric}'] - 2 * center)
        data[f'def_{metric}_edge'] = -sign * (data[f'home_off_{metric}'] + data[f'away_def_{metric}'] - 2 * center)
    data['margin'] = data.away_score - data.home_score
    # nflverse spread_line is the market's expected HOME margin.
    data['market_residual'] = data.margin + data.spread_line
    histories = [f'{loc}_{side}_history_games' for loc in ['away', 'home'] for side in ['off', 'def']]
    data['eligible_history'] = data[histories].ge(min_history).all(axis=1)
    return data, snapshots, candidate_catalog()


def load_game_stats(root, seasons):
    columns = ['game_id', 'season', 'week', 'posteam', 'defteam', 'qb_dropback', 'rush_attempt',
               'pass_attempt', 'sack', 'qb_hit', 'complete_pass', 'first_down', 'third_down_converted',
               'third_down_failed', 'fourth_down_converted', 'fourth_down_failed', 'interception',
               'fumble_lost', 'penalty_team', 'yards_gained', 'epa', 'success', 'qb_epa', 'cpoe',
               'pass_oe', 'qb_kneel', 'qb_spike']
    parts = []
    for season in seasons:
        path = Path(root) / 'data' / 'pbp' / f'pbp_{season}.parquet'
        if not path.exists():
            raise FileNotFoundError(f'Offline scan requires {path}')
        print(f'Preparing prior-game rates: {season}', flush=True)
        parts.append(aggregate_plays(pd.read_parquet(path, columns=columns)))
    return pd.concat(parts, ignore_index=True)
