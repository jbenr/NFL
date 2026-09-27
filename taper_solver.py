"""Solve for the recency taper instead of guessing it.

The model pools a team's recent games into a stat estimate using a decay
curve that was chosen by hand (data_crunchski_2.DECAY_PRESETS). This asks
the curve to justify itself: given the games a team has actually played,
what weighting best predicts what that team does NEXT? That is precisely
the job the pooled stat has to do, so the weights that answer it are the
right weights.

The estimate is a regression, one per "how many games into the season we
are", pooled across every team, season and metric on a common z-scale:

    next game's stat  ~  w1*(last game) + w2*(game before) + ... + prior terms

Two deliberate choices:

  * Lags are counted in GAMES PLAYED, not weeks. A team coming off a bye
    has the same amount of evidence as one that played, and treating the
    bye as a missing week would misalign every team against each other.

  * The weights are constrained non-negative but NOT to sum to one. On a
    z-scale the league average is exactly zero, so a weight vector summing
    to less than one is shrinking toward the league average -- and how much
    to shrink is one of the things being asked. Forcing the sum to one
    would assume the answer is "never" and quietly throw away the result.

So the shortfall 1 - sum(w) is the answer to "how much of this is just
league average", read off the data rather than set by hand.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import nnls

CACHE = Path('data/roster')
GAME_STATS = CACHE / 'game_stats.parquet'
# Metrics the production model consumes, offense and defense alike.
import data_crunchski_3 as dc3
METRICS = dc3.METRICS


def game_panel(metrics=None, first_season=2011):
    """Per-game stats on a common z-scale, with each team-season's games
    numbered in the order they were played.

    z-scored within (season, metric) so that league average is zero and a
    weight vector can be read as a blend of team signal and league average.
    Scoring within season -- rather than pooling every year -- keeps rule
    changes and era drift out of the weights."""
    table = pd.read_parquet(GAME_STATS)
    table = table[table.season >= first_season].copy()
    metrics = list(metrics or METRICS)
    columns = [f'{side}_{metric}' for side in ('off', 'def') for metric in metrics
               if f'{side}_{metric}' in table.columns]
    for column in columns:
        grouped = table.groupby('season')[column]
        table[column] = (table[column] - grouped.transform('mean')) / grouped.transform('std')
    table = table.sort_values(['season', 'team', 'week'])
    table['game_no'] = table.groupby(['season', 'team']).cumcount() + 1
    return table, columns


def lag_design(table, columns, target_game, max_lag=None, prior_games=8):
    """Stack one row per (season, team, metric) predicting `target_game`.

    Regressors, in order: the in-season games already played, most recent
    first, then the prior season's closing form (its last `prior_games`)
    and the prior season as a whole. Pooling the metrics into one stack
    treats the taper as a property of football rather than of a stat --
    checked separately in `per_metric`."""
    in_season = min(target_game - 1, max_lag or target_game - 1)
    rows, targets = [], []
    prior = table.groupby(['season', 'team'])
    closing, whole = {}, {}
    for (season, team), group in prior:
        ordered = group.sort_values('game_no')
        closing[(season, team)] = ordered.tail(prior_games)[columns].mean()
        whole[(season, team)] = ordered[columns].mean()
    for (season, team), group in table.groupby(['season', 'team']):
        ordered = group.sort_values('game_no').set_index('game_no')
        if target_game not in ordered.index:
            continue
        back = closing.get((season - 1, team))
        full = whole.get((season - 1, team))
        if back is None:
            continue
        for column in columns:
            y = ordered.at[target_game, column]
            lags = [ordered.at[target_game - k, column] if target_game - k in ordered.index
                    else np.nan for k in range(1, in_season + 1)]
            row = [*lags, back[column], full[column]]
            if not np.isfinite(y) or not np.all(np.isfinite(row)):
                continue
            rows.append(row)
            targets.append(y)
    names = [f'game-{k}' for k in range(1, in_season + 1)] + ['prior_close', 'prior_all']
    return np.array(rows, dtype=float), np.array(targets, dtype=float), names


def solve(X, y):
    """Non-negative least squares plus the honest accounting of what it did.

    r2 is in-sample; `holdout` below is the number to believe."""
    weights, _ = nnls(X, y)
    fitted = X @ weights
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(weights=weights, r2=1 - ss_res / ss_tot if ss_tot else np.nan,
                total=float(weights.sum()), n=len(y))


def holdout(table, columns, target_game, folds=5, **kwargs):
    """Leave-seasons-out: fit the weights on some seasons, score on others.

    A taper fit and scored on the same games will always look better than a
    hand-chosen one, because it had the answers. This is the comparison
    that counts."""
    X, y, names = lag_design(table, columns, target_game, **kwargs)
    if len(y) < 200:
        return None
    seasons = sorted(table.season.unique())
    blocks = np.array_split(np.array(seasons), folds)
    # Rebuild per fold so no season's rows leak between fit and score.
    scores = []
    for block in blocks:
        train_mask = ~table.season.isin(block)
        Xtr, ytr, _ = lag_design(table[train_mask], columns, target_game, **kwargs)
        Xte, yte, _ = lag_design(table[~train_mask], columns, target_game, **kwargs)
        if len(ytr) < 100 or len(yte) < 50:
            continue
        weights, _ = nnls(Xtr, ytr)
        resid = yte - Xte @ weights
        scores.append(1 - float((resid ** 2).sum()) / float(((yte - yte.mean()) ** 2).sum()))
    fit = solve(X, y)
    fit.update(names=names, holdout_r2=float(np.mean(scores)) if scores else np.nan)
    return fit


def preset_weights(target_game, in_season, preset, prior_games=8, prior_span=16):
    """The weight a DECAY_PRESETS curve would put on each regressor in
    `lag_design`, so a hand-tuned curve can be scored against a solved one.

    Days are reconstructed from the schedule's shape rather than looked up:
    games are a week apart, and a season opens about eight months after the
    last one closed. The decay curve is smooth over months, so a few days
    either way changes nothing."""
    import data_crunchski_2 as dc2
    OFFSEASON = 245.
    lags = np.arange(1, in_season + 1) * 7.
    # prior_close is the mean of that season's last `prior_games`; charge it
    # the age of its midpoint, likewise prior_all over `prior_span`.
    close_age = OFFSEASON + (target_game - 1) * 7. + (prior_games / 2.) * 7.
    all_age = OFFSEASON + (target_game - 1) * 7. + (prior_span / 2.) * 7.
    ages = np.r_[lags, close_age, all_age]
    if preset == 'mean':
        weight = np.ones_like(ages)
    else:
        weight = dc2.gradual_acceleration_with_floor(ages, **dc2.DECAY_PRESETS[preset])
        prior = dc2.PRIOR_SEASON_WEIGHT.get(preset, 1.)
        if prior != 1.:
            weight = np.r_[weight[:in_season], weight[in_season:] * prior]
    # A pooled average: the curve says relative weight, the sum says it is
    # an average and nothing more. Counts stand in for how many games each
    # regressor averages, so the two prior terms don't each count as one.
    counts = np.r_[np.ones(in_season), prior_games, prior_span]
    mass = weight * counts
    return mass / mass.sum()


def score_weights(X, y, weights):
    resid = y - X @ np.asarray(weights, dtype=float)
    return 1 - float((resid ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def retention_map(fade=None):
    """(season, team, unit) -> share of last year's production still here.

    Built by roster_study.unit_retention: for each unit, the fraction of
    the prior season's output (receiving yards, rushing yards, snaps) that
    belongs to players still on the roster."""
    table = pd.read_parquet(CACHE / 'unit_retention.parquet')
    units = [c for c in table.columns if c not in ('season', 'team')]
    return {(int(r.season), r.team): {u: float(getattr(r, u)) for u in units}
            for r in table.itertuples()}, units


def cap_map():
    """(season, team) -> per-unit z-score of cap brought in on new players."""
    table = pd.read_parquet(CACHE / 'incoming_cap.parquet')
    targets = [c for c in table.columns if c.endswith('_target')]
    return {(int(r.season), r.team): {t[:-len('_target')]: float(getattr(r, t))
                                      for t in targets} for r in table.itertuples()}


def unit_of(column):
    """Which group of players owns a metric, and on which side of the ball.

    An offensive pass stat belongs to the receivers; the same stat on
    defense belongs to the pass defense. dc3 already draws these lines for
    the shrinkage experiment -- reused here so both answer the same
    question."""
    side, metric = column.split('_', 1)
    unit = dc3.METRIC_UNITS.get(metric)
    if unit is None:
        return None
    return unit if side == 'off' else dc3.DEFENSIVE_UNITS.get(unit)


def roster_design(table, columns, target_game, mode='plain', prior_games=8):
    """`lag_design` with the prior season re-expressed through the roster.

    mode:
      plain     -- prior season as it stands (the baseline)
      retention -- prior season scaled by how much of that unit is still
                   here, so a gutted unit is pulled toward league average
      cap       -- retention, plus what the replacements cost, so a unit
                   rebuilt expensively is pulled toward better than average

    Each mode is a strictly larger set of regressors than the last, so
    holdout R2 -- not in-sample fit -- is the only honest comparison."""
    retention, _ = retention_map()
    caps = cap_map() if mode == 'cap' else {}
    in_season = target_game - 1
    closing, whole = {}, {}
    for (season, team), group in table.groupby(['season', 'team']):
        ordered = group.sort_values('game_no')
        closing[(season, team)] = ordered.tail(prior_games)[columns].mean()
        whole[(season, team)] = ordered[columns].mean()
    rows, targets = [], []
    for (season, team), group in table.groupby(['season', 'team']):
        ordered = group.sort_values('game_no').set_index('game_no')
        if target_game not in ordered.index:
            continue
        back, full = closing.get((season - 1, team)), whole.get((season - 1, team))
        kept = retention.get((season, team))
        held = caps.get((season, team), {})
        # Every mode is held to the same rows -- a metric with no unit, or
        # a team with no retention figure, is dropped even from the plain
        # baseline. Otherwise the modes would be scored on different games
        # and the R2 gap would be a change of sample, not of information.
        if back is None or kept is None:
            continue
        for column in columns:
            unit = unit_of(column)
            if unit is None:
                continue
            y = ordered.at[target_game, column]
            lags = [ordered.at[target_game - k, column] if target_game - k in ordered.index
                    else np.nan for k in range(1, in_season + 1)]
            row = [*lags, back[column], full[column]]
            if mode != 'plain':
                share = kept.get(unit, np.nan)
                row.append(full[column] * share)
                if mode == 'cap':
                    # What was bought, weighted by how much was lost: a team
                    # that kept everyone gets nothing from this term.
                    row.append(held.get(unit, 0.) * (1. - share))
            if not np.isfinite(y) or not np.all(np.isfinite(row)):
                continue
            rows.append(row)
            targets.append(y)
    names = [f'game-{k}' for k in range(1, in_season + 1)] + ['prior_close', 'prior_all']
    if mode != 'plain':
        names.append('prior_x_retention')
        if mode == 'cap':
            names.append('incoming_cap')
    return np.array(rows, dtype=float), np.array(targets, dtype=float), names


def roster_holdout(table, columns, target_game, mode='plain', folds=5):
    """Leave-seasons-out R2 for a roster mode -- the number that decides
    whether retention or cap earns its place."""
    seasons = np.array(sorted(table.season.unique()))
    scores = []
    for block in np.array_split(seasons, folds):
        train = table[~table.season.isin(block)]
        test = table[table.season.isin(block)]
        Xtr, ytr, _ = roster_design(train, columns, target_game, mode)
        Xte, yte, _ = roster_design(test, columns, target_game, mode)
        if len(ytr) < 100 or len(yte) < 50:
            continue
        weights, _ = nnls(Xtr, ytr)
        resid = yte - Xte @ weights
        scores.append(1 - float((resid ** 2).sum()) / float(((yte - yte.mean()) ** 2).sum()))
    return float(np.mean(scores)) if scores else np.nan


def curve_holdout(table, columns, games, params, folds=5):
    """Out-of-sample R2 for one candidate set of DECAY_PRESETS parameters,
    averaged over the target games given.

    The curve supplies the shape; a single multiplier per target game is
    fitted alongside it, because the network downstream learns the scale
    of its own inputs and a curve should not be judged on something that
    is not its job."""
    import data_crunchski_2 as dc2
    seasons = np.array(sorted(table.season.unique()))
    OFFSEASON, prior_games, prior_span = 245., 8, 16
    out = []
    for target_game in games:
        in_season = target_game - 1
        ages = np.r_[np.arange(1, in_season + 1) * 7.,
                     OFFSEASON + (target_game - 1) * 7. + (prior_games / 2.) * 7.,
                     OFFSEASON + (target_game - 1) * 7. + (prior_span / 2.) * 7.]
        weight = dc2.gradual_acceleration_with_floor(
            ages, total_season_days=params['total_season_days'],
            steepness=params['steepness'], floor_weight=params['floor_weight'])
        weight = np.r_[weight[:in_season], weight[in_season:] * params['prior_season_weight']]
        mass = weight * np.r_[np.ones(in_season), prior_games, prior_span]
        if mass.sum() <= 0:
            return -np.inf
        w = mass / mass.sum()
        scores = []
        for block in np.array_split(seasons, folds):
            tr, te = table[~table.season.isin(block)], table[table.season.isin(block)]
            Xtr, ytr, _ = lag_design(tr, columns, target_game)
            Xte, yte, _ = lag_design(te, columns, target_game)
            if len(ytr) < 100 or len(yte) < 50:
                continue
            ptr, pte = Xtr @ w, Xte @ w
            alpha = float((ptr * ytr).sum() / (ptr * ptr).sum())
            resid = yte - alpha * pte
            scores.append(1 - float((resid ** 2).sum()) / float(((yte - yte.mean()) ** 2).sum()))
        if scores:
            out.append(float(np.mean(scores)))
    return float(np.mean(out)) if out else -np.inf


def blend_table(table, columns, games=range(2, 18), folds=5, prior_games=16):
    """The headline answer: at each point in the season, how much of a
    team's stat estimate should be this year's games and how much last
    year's -- solved, not chosen.

    Deliberately only two knobs per week. The free per-lag fit above found
    the in-season weights to be flat, so pooling them costs nothing and
    buys an answer simple enough to install and to state in a sentence.
    A third number, 1 - in_season - prior, is the pull toward league
    average, which on a z-scale is what is left when neither source is
    trusted."""
    seasons = np.array(sorted(table.season.unique()))
    out = []
    for target_game in games:
        def design(sub):
            rows, targets = [], []
            whole = {k: g.sort_values('game_no').tail(prior_games)[columns].mean()
                     for k, g in sub.groupby(['season', 'team'])}
            for (season, team), group in sub.groupby(['season', 'team']):
                ordered = group.sort_values('game_no').set_index('game_no')
                if target_game not in ordered.index:
                    continue
                prior = whole.get((season - 1, team))
                if prior is None:
                    continue
                played = [i for i in range(1, target_game) if i in ordered.index]
                if not played:
                    continue
                for column in columns:
                    y = ordered.at[target_game, column]
                    this_year = float(np.mean([ordered.at[i, column] for i in played]))
                    row = [this_year, prior[column]]
                    if not np.isfinite(y) or not np.all(np.isfinite(row)):
                        continue
                    rows.append(row)
                    targets.append(y)
            return np.array(rows, dtype=float), np.array(targets, dtype=float)

        scores = []
        for block in np.array_split(seasons, folds):
            Xtr, ytr = design(table[~table.season.isin(block)])
            Xte, yte = design(table[table.season.isin(block)])
            if len(ytr) < 100 or len(yte) < 50:
                continue
            w, _ = nnls(Xtr, ytr)
            resid = yte - Xte @ w
            scores.append(1 - float((resid ** 2).sum()) / float(((yte - yte.mean()) ** 2).sum()))
        X, y = design(table)
        if len(y) < 100:
            continue
        w, _ = nnls(X, y)
        out.append(dict(game=target_game, in_season=float(w[0]), prior=float(w[1]),
                        league_avg=float(1 - w.sum()), n=len(y),
                        holdout_r2=float(np.mean(scores)) if scores else np.nan))
    return pd.DataFrame(out)


def constant_ratio_sweep(table, columns, games, ratios, folds=5, prior_games=16):
    """Score a range of FIXED prior-season discounts: every game this
    season counts one, every game last season counts `ratio`, with a
    single scale fitted out of sample on top.

    The per-week solved ratios wobble (0.67, 0.34, 0.44, 0.40 ...) on
    samples where the in-season block carries only a few percent of the
    weight -- exactly where a fitted number is mostly noise. If one
    constant matches the per-week table out of sample, the constant is the
    honest thing to install.

    Every ratio is scored against the SAME prebuilt designs: rebuilding
    them per candidate is both slow and, with a backtest running beside
    it, enough to run the box out of memory."""
    seasons = np.array(sorted(table.season.unique()))
    built = {}
    for target_game in games:
        for fold, block in enumerate(np.array_split(seasons, folds)):
            Xtr, ytr, _ = lag_design(table[~table.season.isin(block)], columns, target_game)
            Xte, yte, _ = lag_design(table[table.season.isin(block)], columns, target_game)
            if len(ytr) >= 100 and len(yte) >= 50:
                built[(target_game, fold)] = (Xtr, ytr, Xte, yte)
    out = {}
    for ratio in ratios:
        per_game = []
        for target_game in games:
            in_season = target_game - 1
            w = np.r_[np.ones(in_season), 0., ratio * prior_games]
            w = w / w.sum()
            scores = []
            for fold in range(folds):
                if (target_game, fold) not in built:
                    continue
                Xtr, ytr, Xte, yte = built[(target_game, fold)]
                ptr, pte = Xtr @ w, Xte @ w
                alpha = float((ptr * ytr).sum() / (ptr * ptr).sum())
                resid = yte - alpha * pte
                scores.append(1 - float((resid ** 2).sum()) / float(((yte - yte.mean()) ** 2).sum()))
            if scores:
                per_game.append(float(np.mean(scores)))
        out[ratio] = float(np.mean(per_game)) if per_game else np.nan
    return out
