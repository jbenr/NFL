import unittest

import numpy as np
import pandas as pd

import opponent_adjust as oa


def season(offence, defence, schedule, home_edge=0.3, noise=0.0, plays=30, seed=0):
    """Games generated from known ratings, so the fit can be checked
    against the truth rather than against itself."""
    rng = np.random.default_rng(seed)
    rows = []
    for away, home in schedule:
        for team, opponent, is_home in ((away, home, 0), (home, away, 1)):
            value = (offence[team] + defence[opponent] + home_edge * is_home
                     + (rng.normal(0, noise) if noise else 0.0))
            rows.append(dict(team=team, opponent=opponent, is_home=is_home,
                             metric=value, plays=plays))
    return pd.DataFrame(rows)


class RecoversWhatItWasGiven(unittest.TestCase):
    """Synthetic seasons where the true ratings are known."""

    TEAMS = [f'T{i:02d}' for i in range(10)]

    def round_robin(self, repeats=2):
        return [(a, b) for _ in range(repeats)
                for i, a in enumerate(self.TEAMS) for b in self.TEAMS[i + 1:]]

    def test_a_balanced_schedule_recovers_the_ordering(self):
        offence = {t: v for t, v in zip(self.TEAMS, np.linspace(-1.5, 1.5, 10))}
        defence = {t: 0.0 for t in self.TEAMS}
        games = season(offence, defence, self.round_robin())
        rated, _ = oa.ratings(games, ['metric'], prior_games=0.5)
        order = rated.metric.sort_values().index.tolist()
        self.assertEqual(order, sorted(offence, key=offence.get))

    def test_the_hard_schedule_case(self):
        """The case this module exists for: the best rushing offence plays
        only the best run defences, so its RAW average is mid-table. The
        adjusted rating should still put it top; the raw one does not."""
        teams = self.TEAMS
        offence = {t: 0.0 for t in teams}
        offence['T00'] = 2.0                      # the best offence, by a street
        defence = {t: 0.0 for t in teams}
        for t in teams[1:5]:
            # Hard enough that the schedule outweighs the talent: a +2.0
            # offence facing -5.0 defences posts -3.0, below a average
            # offence on an ordinary slate. That is the whole scenario --
            # a weaker handicap leaves the raw average ranking it first
            # anyway and the test proves nothing.
            defence[t] = -5.0                     # the four best defences
        # T00 plays only those four; everyone else plays a normal slate.
        schedule = [('T00', t) for t in teams[1:5]] * 2
        schedule += [(a, b) for i, a in enumerate(teams[1:], 1) for b in teams[i + 1:]]
        games = season(offence, defence, schedule)

        raw = games.groupby('team').metric.mean()
        self.assertLess(raw.rank(ascending=False)['T00'], 11)
        self.assertGreater(raw.rank(ascending=False)['T00'], 1,
                           'the raw average should NOT already rank it first')

        adjusted, _ = oa.ratings(games, ['metric'], prior_games=0.5)
        self.assertEqual(adjusted.metric.idxmax(), 'T00',
                         'the adjusted rating should see through the schedule')

    def test_defence_is_estimated_too(self):
        offence = {t: 0.0 for t in self.TEAMS}
        defence = {t: v for t, v in zip(self.TEAMS, np.linspace(-1.5, 1.5, 10))}
        games = season(offence, defence, self.round_robin())
        _, rated = oa.ratings(games, ['metric'], prior_games=0.5)
        self.assertEqual(rated.metric.idxmin(), min(defence, key=defence.get))

    def test_home_advantage_is_held_out_of_the_ratings(self):
        """A team is not a better offence for having played at home."""
        offence = {t: 0.0 for t in self.TEAMS}
        defence = {t: 0.0 for t in self.TEAMS}
        games = season(offence, defence, self.round_robin(), home_edge=3.0)
        rated, _ = oa.ratings(games, ['metric'], prior_games=0.5)
        self.assertLess(rated.metric.std(), 0.2, 'home edge leaked into the offence ratings')


class ThePenalty(unittest.TestCase):
    """Ridge does two jobs here: it shrinks toward league average when the
    evidence is thin, and it picks the centred solution out of a family
    that is otherwise indistinguishable."""

    TEAMS = [f'T{i:02d}' for i in range(10)]

    def games(self, **kwargs):
        offence = {t: v for t, v in zip(self.TEAMS, np.linspace(-2, 2, 10))}
        defence = {t: 0.0 for t in self.TEAMS}
        schedule = [(a, b) for i, a in enumerate(self.TEAMS) for b in self.TEAMS[i + 1:]]
        return season(offence, defence, schedule, **kwargs)

    def test_a_heavier_prior_pulls_everything_toward_the_middle(self):
        games = self.games()
        loose, _ = oa.ratings(games, ['metric'], prior_games=0.5)
        tight, _ = oa.ratings(games, ['metric'], prior_games=50.0)
        self.assertLess(tight.metric.std(), loose.metric.std())

    def test_the_additive_degeneracy_is_resolved(self):
        """mu + offence + defence is unchanged by adding a constant to every
        offence and subtracting it from every defence. Unpenalised there is
        no unique answer; the penalty picks the centred one."""
        games = self.games()
        offence, defence = oa.ratings(games, ['metric'], prior_games=1.0)
        centre = (offence.metric.mean() + defence.metric.mean()) / 2
        self.assertAlmostEqual(offence.metric.mean(), centre, places=6)

    def test_ratings_are_in_the_metric_s_own_units(self):
        """Not z-scores: a rating is what the team would post against an
        average opponent, so everything downstream is unchanged."""
        games = self.games()
        rated, _ = oa.ratings(games, ['metric'], prior_games=0.5)
        self.assertAlmostEqual(float(rated.metric.mean()), float(games.metric.mean()), delta=0.5)


class Volume(unittest.TestCase):
    def test_a_bigger_sample_counts_for_more(self):
        """A 40-carry game says more than a 12-carry one."""
        rows = []
        for opponent, value, plays in (('B', 5.0, 40), ('C', 1.0, 4)):
            rows.append(dict(team='A', opponent=opponent, is_home=0, metric=value, plays=plays))
            rows.append(dict(team=opponent, opponent='A', is_home=1, metric=3.0, plays=20))
        frame = pd.DataFrame(rows)
        weighted, _ = oa.ratings(frame, ['metric'], prior_games=0.1)
        flat, _ = oa.ratings(frame.assign(plays=1), ['metric'], prior_games=0.1)
        self.assertGreater(weighted.metric['A'], flat.metric['A'],
                           'the 40-carry game should pull A up further than the 4-carry one')


class Robustness(unittest.TestCase):
    def test_a_metric_missing_for_some_games_still_fits(self):
        teams = [f'T{i:02d}' for i in range(6)]
        offence = {t: v for t, v in zip(teams, np.linspace(-1, 1, 6))}
        games = season(offence, {t: 0.0 for t in teams},
                       [(a, b) for i, a in enumerate(teams) for b in teams[i + 1:]])
        games.loc[games.index[:4], 'metric'] = np.nan
        rated, _ = oa.ratings(games, ['metric'], prior_games=0.5)
        self.assertTrue(np.isfinite(rated.metric).all())

    def test_an_empty_window_returns_empty_rather_than_raising(self):
        frame = pd.DataFrame(columns=['team', 'opponent', 'is_home', 'metric', 'plays'])
        offence, defence = oa.ratings(frame, ['metric'])
        self.assertTrue(offence.empty and defence.empty)


if __name__ == '__main__':
    unittest.main()


class PerMetricVolume(unittest.TestCase):
    """A rushing rate is a rate over rushes. Weighting it by total plays
    over-counts games the team happened to throw a lot in."""

    def test_each_metric_maps_to_its_own_denominator(self):
        for metric, expected in (('run_ypp', '_run_plays'), ('off_run_ypp', '_run_plays'),
                                 ('def_pass_ypp', '_pass_plays'), ('sack_%', '_pass_plays'),
                                 ('third_down_%', '_third_downs')):
            self.assertEqual(oa.volume_for(metric, [expected]), expected, metric)

    def test_an_unlisted_metric_falls_back_to_scrimmage_plays(self):
        self.assertEqual(oa.volume_for('first_down_pp', ['_scrimmage_plays']), '_scrimmage_plays')

    def test_a_missing_volume_column_does_not_crash_the_fit(self):
        self.assertIsNone(oa.volume_for('run_ypp', []))

    def test_rushing_and_passing_are_weighted_differently(self):
        """The whole point: two games, one run-heavy and one pass-heavy,
        should pull the rushing and passing ratings toward different
        games."""
        rows = [
            dict(team='A', opponent='B', is_home=0, run_ypp=6.0, pass_ypp=4.0,
                 _run_plays=40, _pass_plays=5, _scrimmage_plays=45),
            dict(team='A', opponent='C', is_home=0, run_ypp=2.0, pass_ypp=9.0,
                 _run_plays=5, _pass_plays=40, _scrimmage_plays=45),
            dict(team='B', opponent='A', is_home=1, run_ypp=4.0, pass_ypp=6.5,
                 _run_plays=20, _pass_plays=20, _scrimmage_plays=40),
            dict(team='C', opponent='A', is_home=1, run_ypp=4.0, pass_ypp=6.5,
                 _run_plays=20, _pass_plays=20, _scrimmage_plays=40),
        ]
        frame = pd.DataFrame(rows)
        offence, _ = oa.ratings(frame, ['run_ypp', 'pass_ypp'], weight=None, prior_games=0.1)
        # A's rushing should follow the 40-carry game (6.0), not the 5-carry
        # one (2.0); its passing should follow the 40-dropback game (9.0).
        self.assertGreater(offence.run_ypp['A'], 4.0, 'rushing was not weighted by carries')
        self.assertGreater(offence.pass_ypp['A'], 6.5, 'passing was not weighted by dropbacks')


class WiredIntoCalcStats(unittest.TestCase):
    """The preset has to be a drop-in: same columns, same index, same
    units as the pooled path, or everything downstream breaks."""

    @classmethod
    def setUpClass(cls):
        import data_crunchski_2 as dc2
        cls.dc2 = dc2
        pbp = pd.read_parquet('data/pbp/pbp_2023.parquet')
        cls.pbp = pbp[pbp.week <= 6]

    def stats(self, mode):
        self.dc2.RATE_MODE = mode
        try:
            return self.dc2.calc_stats(self.pbp)
        finally:
            self.dc2.RATE_MODE = 'mean'

    def test_it_is_a_registered_calculation(self):
        self.assertIn('adjusted_1.0', self.dc2.ADJUSTED_PRESETS)
        self.assertNotIn('adjusted_1.0', self.dc2.DECAY_PRESETS,
                         'it is not a decay curve and must not be treated as one')

    def test_same_columns_as_the_pooled_path(self):
        adjusted, pooled = self.stats('adjusted_1.0'), self.stats('mean')
        self.assertEqual(sorted(adjusted.columns), sorted(pooled.columns))
        self.assertEqual(adjusted.index.name, pooled.index.name)
        self.assertEqual(set(adjusted.index), set(pooled.index))

    def test_no_defensive_usage_rate_is_invented(self):
        """Usage rate and time of possession belong to an offence. The
        defence side is fitted on the same offensive metrics, so these
        have to be dropped rather than mirrored."""
        adjusted = self.stats('adjusted_1.0')
        for metric in self.dc2.OFFENCE_ONLY:
            self.assertNotIn(f'def_{metric}', adjusted.columns)
            self.assertIn(f'off_{metric}', adjusted.columns)

    def test_ratings_stay_in_the_metric_s_units(self):
        """Not z-scores -- the adjusted rushing number should still look
        like yards per carry, or every threshold downstream moves."""
        adjusted, pooled = self.stats('adjusted_1.0'), self.stats('mean')
        self.assertAlmostEqual(adjusted.off_run_ypp.mean(), pooled.off_run_ypp.mean(), delta=0.5)
        self.assertTrue(adjusted.notna().all().all())

    def test_it_actually_changes_something(self):
        """A preset that returns the pooled numbers back is not doing its
        job -- but it should not reshuffle the league either."""
        adjusted, pooled = self.stats('adjusted_1.0'), self.stats('mean')
        shift = (adjusted.off_run_ypp - pooled.off_run_ypp).abs()
        self.assertGreater(shift.mean(), 0.05, 'the adjustment did nothing')
        self.assertGreater(adjusted.off_run_ypp.corr(pooled.off_run_ypp, method='spearman'), 0.8,
                           'the adjustment reordered the league beyond recognition')
