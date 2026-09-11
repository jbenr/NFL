"""Pregame win-versus-loss playoff leverage, using paired remaining-season scenarios.

This is a heuristic, not an exact NFL clinching engine. Future games are fair
coin flips without ties. Historical ties count half. Tiebreaks use head-to-head,
division/conference/common-game records and strength of victory/schedule.
Unresolved points-based tiebreaks use paired random lots, not invented scores.
"""
import argparse
import hashlib

import numpy as np
import pandas as pd


FEATURES = ['importance', 'playoff_swing', 'division_swing', 'bye_swing',
            'top_seed_swing', 'seed_swing', 'advancement_swing', 'postseason']


def seeds(points, played, conferences, divisions, lots, season, victories=None):
    """Seed division champions first, then wild cards (division priority retained)."""
    samples, teams, _ = points.shape
    record = points.sum(2) / np.maximum(played.sum(1), 1)
    division_mask = (divisions[:, None] == divisions) & (conferences[:, None] == conferences)
    conference_mask = conferences[:, None] == conferences
    division_record = (points * division_mask).sum(2) / np.maximum((played * division_mask).sum(1), 1)
    conference_record = (points * conference_mask).sum(2) / np.maximum((played * conference_mask).sum(1), 1)
    wins = np.floor(points) if victories is None else victories
    opponent_points, opponent_games = points.sum(2), played.sum(1)
    sov = (wins * opponent_points[:, None, :]).sum(2) / np.maximum((wins * opponent_games).sum(2), 1)
    sos = (played * opponent_points[:, None, :]).sum(2) / np.maximum((played * opponent_games).sum(1), 1)

    def choose(eligible, within_division=False):
        tied = eligible.copy()
        best = np.where(tied, record, -np.inf).max(1, keepdims=True)
        tied &= record == best
        # Restart criteria when the tied group shrinks (e.g. three clubs to two).
        for _ in range(teams):
            if np.all(tied.sum(1) <= 1):
                break
            counts = tied.sum(1)
            opponents = tied[:, None, :]
            h_games = (played[None, :, :] * opponents).sum(2)
            h_points = (points * opponents).sum(2)
            common = ((played > 0)[None, :, :] | ~tied[:, :, None]).all(1)
            common_games = (played * common[:, None, :]).sum(2)
            common_record = (points * common[:, None, :]).sum(2) / np.maximum(common_games, 1)
            minimum = 1 if within_division else 4
            usable_common = np.all(~tied | (common_games >= minimum), axis=1)
            common_record = np.where(usable_common[:, None], common_record, 0)
            if within_division:
                head = h_points / np.maximum(h_games, 1)
                usable = np.all(~tied | (h_games > 0), axis=1)
                metrics = [np.where(usable[:, None], head, 0), division_record, common_record, conference_record]
            else:
                faced = ((played > 0)[None, :, :] & opponents).sum(2)
                all_faced = (faced == counts[:, None] - 1) & (h_games > 0)
                sweep = np.where((h_points == h_games) & all_faced, 1.,
                                 np.where((h_points == 0) & all_faced, -1., 0.))
                head = np.where(counts[:, None] == 2, h_points / np.maximum(h_games, 1), sweep)
                metrics = [head, conference_record, common_record]
            narrowed = np.zeros(samples, dtype=bool)
            for metric in metrics + [sov, sos, lots]:
                best = np.where(tied, metric, -np.inf).max(1, keepdims=True)
                next_tie = tied & (metric == best)
                change = ~narrowed & (next_tie.sum(1) < tied.sum(1))
                tied[change] = next_tie[change]
                narrowed |= change
        winner = tied.argmax(1)
        return winner, eligible.any(1)

    positions = np.zeros((samples, teams), dtype=int)
    rows = np.arange(samples)
    for conf in sorted(set(conferences)):
        eligible = np.broadcast_to(conferences == conf, (samples, teams)).copy()
        champions = np.zeros_like(eligible)
        for division in sorted(set(divisions[conferences == conf])):
            winner, active = choose(eligible & (divisions == division), True)
            champions[rows[active], winner[active]] = True
        remaining_champions = champions.copy()
        for rank in range(1, int(champions.sum(1).max()) + 1):
            winner, active = choose(remaining_champions)
            positions[rows[active], winner[active]] = rank
            remaining_champions[rows[active], winner[active]] = False
        eligible &= ~champions
        # NFL has four division champions; smaller fixtures are also supported.
        first_wild = int(champions.sum(1).max()) + 1
        for rank in range(first_wild, (7 if season >= 2020 else 6) + 1):
            representatives = np.zeros_like(eligible)
            for division in sorted(set(divisions[conferences == conf])):
                winner, active = choose(eligible & (divisions == division), True)
                representatives[rows[active], winner[active]] = True
            winner, active = choose(representatives)
            positions[rows[active], winner[active]] = rank
            eligible[rows[active], winner[active]] = False
    return positions


def week_importance(schedule, season, week, samples=512, seed=1337):
    from data_crunchski_2 import TEAM_META
    if samples < 2:
        raise ValueError('At least two scenarios required')
    slate = schedule[schedule.season.eq(season)].copy()
    target = slate[slate.week.eq(week)]
    regular = slate[slate.game_type.eq('REG')]
    teams = sorted(pd.unique(regular[['away_team', 'home_team']].values.ravel()))
    unknown = set(teams) - set(TEAM_META)
    if unknown:
        raise ValueError(f'Unknown conference/division for {sorted(unknown)}')
    lookup = {t: i for i, t in enumerate(teams)}
    conferences = np.array([TEAM_META[t][0] for t in teams])
    divisions = np.array([TEAM_META[t][1] for t in teams])
    base = np.zeros((len(teams), len(teams)), dtype=float)
    tie_points = np.zeros_like(base)
    played = np.zeros_like(base)
    past = regular[regular.week.lt(week)].dropna(subset=['away_score', 'home_score'])
    for game in past.itertuples():
        a, h = lookup[game.away_team], lookup[game.home_team]
        outcome = float(game.away_score > game.home_score) + .5 * (game.away_score == game.home_score)
        base[a, h] += outcome
        base[h, a] += 1 - outcome
        if outcome == .5:
            tie_points[a, h] += .5
            tie_points[h, a] += .5
        played[a, h] += 1
        played[h, a] += 1
    future = regular[regular.week.ge(week)].sort_values(['week', 'away_team', 'home_team'])
    rng = np.random.default_rng([seed, int(season), int(week)])
    draws = rng.integers(0, 2, (samples, len(future))).astype(float)
    worlds = np.broadcast_to(base, (samples, *base.shape)).copy()
    for j, game in enumerate(future.itertuples()):
        a, h = lookup[game.away_team], lookup[game.home_team]
        worlds[:, a, h] += draws[:, j]
        worlds[:, h, a] += 1 - draws[:, j]
        played[a, h] += 1
        played[h, a] += 1
    lots = rng.random((samples, len(teams)))
    game_columns = {(g.week, g.away_team, g.home_team): j for j, g in enumerate(future.itertuples())}
    result = target.copy()
    for index, game in target.iterrows():
        if game.game_type != 'REG':
            for side in ['away', 'home']:
                for feature in FEATURES:
                    result.loc[index, side + '_' + feature] = float(feature in ['importance', 'advancement_swing', 'postseason'])
                result.loc[index, side + '_playoff_if_win'] = 1.
                result.loc[index, side + '_playoff_if_loss'] = 1.
            continue
        a, h = lookup[game.away_team], lookup[game.home_team]
        j = game_columns[(game.week, game.away_team, game.home_team)]
        branches = []
        for outcome in [1., 0.]:
            branch = worlds.copy()
            change = outcome - draws[:, j]
            branch[:, a, h] += change
            branch[:, h, a] -= change
            branches.append(seeds(branch, played, conferences, divisions, lots, season,
                                  victories=branch - tie_points))
        for side, team, win, lose in [('away', a, branches[0], branches[1]),
                                      ('home', h, branches[1], branches[0])]:
            win, lose = win[:, team], lose[:, team]
            swings = {}
            for event, predicate in [
                ('playoff', lambda s: s > 0),
                ('division', lambda s: (s > 0) & (s <= 4)),
                ('bye', lambda s: (s > 0) & (s <= (1 if season >= 2020 else 2))),
                ('top_seed', lambda s: s == 1),
            ]:
                p_win, p_loss = predicate(win).mean(), predicate(lose).mean()
                result.loc[index, f'{side}_{event}_if_win'] = p_win
                result.loc[index, f'{side}_{event}_if_loss'] = p_loss
                swings[event + '_swing'] = max(0., p_win - p_loss)
            # Paired scenarios isolate seeding changes when qualified either way.
            swings['seed_swing'] = float(((win > 0) & (lose > 0) & (win != lose)).mean())
            swings.update(importance=max(swings.values()), advancement_swing=0., postseason=0.)
            for feature, value in swings.items():
                result.loc[index, side + '_' + feature] = value
    result['importance_method'] = 'paired-playoff-scenarios-v1'
    result['importance_samples'] = samples
    return result


def game_importance(schedule, samples=512, weeks=None):
    import utils
    from tqdm import tqdm
    weeks = schedule[['season', 'week']] if weeks is None else weeks[['season', 'week']]
    weeks = weeks.drop_duplicates().sort_values(['season', 'week'])
    results = []
    for season, week in tqdm(list(weeks.itertuples(index=False, name=None)), desc='Playoff leverage'):
        slate = schedule[schedule.season.eq(season)]
        digest = hashlib.sha256(pd.util.hash_pandas_object(
            slate.sort_values(['week', 'away_team', 'home_team']), index=False).values.tobytes()).hexdigest()
        path = utils.cache_path('playoff_leverage', [int(season), int(week), samples, digest],
                                [__file__, 'data_crunchski_2.py'])
        if path.exists():
            results.append(pd.read_parquet(path))
        else:
            result = week_importance(slate, int(season), int(week), samples)
            utils.save_parquet(result, path)
            results.append(result)
    return pd.concat(results)


if __name__ == '__main__':
    from data_crunchski_2 import RELOCATED_TEAMS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--season', type=int, required=True)
    p.add_argument('--week', type=int, required=True)
    p.add_argument('--samples', type=int, default=2048)
    args = p.parse_args()
    schedule = pd.read_parquet('data/sched.parquet').replace(
        {'away_team': RELOCATED_TEAMS, 'home_team': RELOCATED_TEAMS})
    data = week_importance(schedule, args.season, args.week, args.samples)
    columns = ['away_team', 'home_team'] + [side + '_' + name for side in ['away', 'home']
               for name in ['playoff_if_win', 'playoff_if_loss', 'division_swing', 'bye_swing', 'importance']]
    print(data.reindex(columns=columns).to_string(index=False, float_format=lambda v: f'{v:.1%}'))
