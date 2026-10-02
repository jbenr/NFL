"""Opponent-adjusted team ratings: what a team's numbers would look like
against an average opponent.

The problem this exists for. A team's stat today is a raw pooled rate over
the lookback window, with no memory of who it was compiled against. An
elite rushing team whose first five opponents are the five best run
defenses posts mid-table numbers and the model reads it as a mid-table
rushing team. The matchup input differences that team's offence against
the next opponent's defence -- but both of those numbers carry their own
schedule noise, and the errors are independent, so differencing them adds
noise rather than cancelling it.

The fix is to stop asking "what did they average" and start asking "how
good are they", which is a two-way problem:

    metric(offence o vs defence d)  =  mu + offence_o + defence_d + home

Every team-game constrains an offence and a defence at once, so the whole
set is estimated simultaneously and the circularity resolves itself: a run
defence does not get credit for facing bad rushing teams, because the
rushing teams it faced are being estimated from the same games.

Two details that matter more than they look.

RIDGE DOES TWO JOBS. The penalty is what makes this usable in September --
in week 2 there are roughly 64 team-games for 65 parameters, and the
unpenalised fit is nonsense. Shrinking the team effects toward zero is
exactly "pull the estimate toward league average when the evidence is
thin", which taper_solver measured independently as the right thing to do
(a week-2 estimate wants to be ~60% league average). It also fixes the
additive degeneracy for free: mu + offence_o + defence_d is unchanged if
you add a constant to every offence and subtract it from every defence, so
the unpenalised problem has no unique answer. Penalising the team effects
picks the centred one.

VOLUME IS THE WEIGHT. A 40-carry game says more about a rushing offence
than a 12-carry game, and an unweighted fit treats them alike.
"""
import numpy as np
import pandas as pd

# How many games of evidence it takes to earn half your measured effect.
# A team with `PRIOR_GAMES` games of data sits halfway between the league
# average and what it actually did; with four times that, three quarters of
# the way. Chosen to be tuned (see `tune`), not believed.
PRIOR_GAMES = 6.0


def design(frame, teams=None):
    """The model matrix for `mu + offence_o + defence_d + home`.

    One row per (game, team): that team's offence against the opponent's
    defence. Columns are [intercept, home, offence per team, defence per
    team]. Dense on purpose -- this is 65 columns and a few hundred rows
    even at a full season, and a sparse solve would cost more in overhead
    than it saves."""
    teams = list(teams if teams is not None else
                 sorted(set(frame.team) | set(frame.opponent)))
    position = {team: i for i, team in enumerate(teams)}
    rows = len(frame)
    matrix = np.zeros((rows, 2 + 2 * len(teams)), dtype=float)
    matrix[:, 0] = 1.0
    matrix[:, 1] = frame.is_home.to_numpy(dtype=float)
    offence = np.array([position[t] for t in frame.team], dtype=int)
    defence = np.array([position[t] for t in frame.opponent], dtype=int)
    matrix[np.arange(rows), 2 + offence] = 1.0
    matrix[np.arange(rows), 2 + len(teams) + defence] = 1.0
    return matrix, teams


def solve(matrix, values, weights, teams, prior_games=PRIOR_GAMES):
    """Weighted ridge, penalising the team effects but not mu or home.

    `values` may be a single column or many: the design matrix is shared
    across metrics, so every metric is solved from one decomposition."""
    values = np.asarray(values, dtype=float)
    single = values.ndim == 1
    if single:
        values = values[:, None]
    weights = np.asarray(weights, dtype=float)
    # Weights in units of "an average game", so the penalty can be stated
    # in games rather than in whatever the volume happens to be.
    scale = weights.mean()
    weights = weights / scale if scale > 0 else np.ones_like(weights)

    penalty = np.zeros(matrix.shape[1])
    penalty[2:] = float(prior_games)     # mu and home are unpenalised
    weighted = matrix * weights[:, None]
    normal = matrix.T @ weighted + np.diag(penalty)
    targets = weighted.T @ values
    # Missing values are per-metric (a team with no rushing attempts in a
    # game has no run_ypp), so each column gets the rows it actually has.
    coefficients = np.zeros((matrix.shape[1], values.shape[1]))
    for column in range(values.shape[1]):
        present = np.isfinite(values[:, column]) & (weights > 0)
        if present.sum() < 2:
            coefficients[:, column] = np.nan
            continue
        if present.all():
            coefficients[:, column] = np.linalg.solve(normal, targets[:, column])
            continue
        sub_weighted = matrix[present] * weights[present, None]
        sub_normal = matrix[present].T @ sub_weighted + np.diag(penalty)
        coefficients[:, column] = np.linalg.solve(
            sub_normal, sub_weighted.T @ values[present, column])
    count = len(teams)
    return dict(intercept=coefficients[0], home=coefficients[1],
                offence=coefficients[2:2 + count], defence=coefficients[2 + count:])


# Which volume each metric is a rate over. A rushing number weighted by
# total plays over-counts a game the team happened to throw a lot in, so
# every metric is weighted by its own denominator -- the same one
# _ratios_from_ingredients divides by. Anything unlisted falls back to
# scrimmage plays.
METRIC_VOLUME = {
    'run_ypp': '_run_plays', 'explosive_run_%': '_run_plays', 'stuff_%': '_run_plays',
    'run_epa_pp': '_run_plays', 'run_success_%': '_run_plays',
    'pass_ypp': '_pass_plays', 'explosive_pass_%': '_pass_plays', 'sack_%': '_pass_plays',
    'qb_hit_%': '_pass_plays', 'pass_epa_pp': '_pass_plays', 'pass_success_%': '_pass_plays',
    'pass_completion_%': '_pass_plays',
    'third_down_%': '_third_downs', 'fourth_down_%': '_fourth_downs',
    'turnovers_pp': '_turnover_opportunities', 'penalties_pp': '_penalty_plays',
    'series_success_%': '_series_count',
}
DEFAULT_VOLUME = '_scrimmage_plays'


def volume_for(metric, available):
    """The weight column for one metric, or None to fall back to flat.

    `metric` may carry an off_/def_ prefix; the volume is the same either
    way, because a defensive rushing number is a rate over the rushes the
    opponent ran."""
    bare = metric.split('_', 1)[1] if metric.startswith(('off_', 'def_')) else metric
    wanted = METRIC_VOLUME.get(bare, DEFAULT_VOLUME)
    if wanted in available:
        return wanted
    return DEFAULT_VOLUME if DEFAULT_VOLUME in available else None


def ratings(frame, metrics, weight='plays', prior_games=PRIOR_GAMES, teams=None):
    """Opponent-adjusted ratings, in the metric's own units.

    `frame`: one row per (game, team) with `team`, `opponent`, `is_home`,
    a column per metric, and a volume column.

    Returns (offence, defence), each a DataFrame indexed by team with one
    column per metric. A value is what that team would be expected to post
    against an average opponent -- the same units as the raw pooled rate it
    replaces, so everything downstream (z-scoring, differencing) is
    unchanged."""
    metrics = [m for m in metrics if m in frame.columns]
    if not metrics or frame.empty:
        empty = pd.DataFrame(index=pd.Index([], name='team'), columns=metrics, dtype=float)
        return empty, empty.copy()
    matrix, teams = design(frame, teams)
    index = pd.Index(teams, name='team')
    # Metrics that share a volume column share a solve; the design matrix is
    # the same and only the weights differ, so this is a handful of
    # factorisations rather than one per metric.
    by_volume = {}
    for metric in metrics:
        column = weight if weight in frame else volume_for(metric, frame.columns)
        by_volume.setdefault(column, []).append(metric)
    offence = pd.DataFrame(index=index, columns=metrics, dtype=float)
    defence = pd.DataFrame(index=index, columns=metrics, dtype=float)
    for column, group in by_volume.items():
        weights = (frame[column].to_numpy(dtype=float) if column in frame
                   else np.ones(len(frame)))
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.)
        fitted = solve(matrix, frame[group].to_numpy(dtype=float), weights, teams, prior_games)
        offence[group] = fitted['intercept'] + fitted['offence']
        defence[group] = fitted['intercept'] + fitted['defence']
    return offence, defence


def strength_of_schedule(frame, metrics, prior_games=PRIOR_GAMES):
    """How hard each team's opponents have been, per metric, in the units
    of that metric -- the size of the correction this whole module exists
    to apply. Positive means the defences faced were tougher than average.

    Diagnostic rather than an input: it is what you show someone who asks
    why a team's adjusted rushing number is a point above its raw one."""
    offence, defence = ratings(frame, metrics, prior_games=prior_games)
    league = defence.mean()
    faced = frame.groupby('team').apply(
        lambda g: defence.reindex(g.opponent)[[m for m in metrics if m in defence]].mean(),
        include_groups=False)
    return (league - faced).rename_axis('team')
