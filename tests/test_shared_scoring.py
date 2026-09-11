import unittest
import numpy as np
import pandas as pd
import shared_scoring as ss


def games():
    rng = np.random.default_rng(7)
    rows = pd.DataFrame(dict(season=[2025] * 8 + [2026], week=list(range(1, 9)) + [1],
                             away_team=['ARI'] * 9, home_team=['LAC'] * 9,
                             away_score=np.arange(9) + 17., home_score=np.arange(9) + 20.,
                             home_field_adv=[1.] * 9, away_rest=[7.] * 9, home_rest=[7.] * 9))
    for side in ['away', 'home']:
        for unit in ['off', 'def']:
            for metric in ss.METRICS:
                rows[f'{side}_raw_{unit}_{metric}'] = rng.normal(size=9)
    return rows


class SharedScoringTests(unittest.TestCase):
    def test_referee_tendency_excludes_current_week_and_shrinks(self):
        history = pd.DataFrame(dict(season=[2025]*4, week=[1, 1, 2, 3],
                                    referee=['A', 'B', 'A', 'A'],
                                    away_score=[10, 30, 999, 888], home_score=[10, 30, 999, 888]))
        target = pd.DataFrame(dict(season=[2025, 2025], week=[2, 2], referee=['A', 'new']))
        a = ss.referee_tendencies(target, history)
        self.assertAlmostEqual(a.referee_avg_total.iloc[0], (20 + 20 * 40) / 21)
        self.assertEqual(a.referee_prior_games.iloc[0], 1)
        self.assertEqual(a.referee_total_delta.iloc[1], 0)
        history.loc[history.week.ge(2), ['away_score', 'home_score']] = 0
        pd.testing.assert_frame_equal(a, ss.referee_tendencies(target, history))

    def test_differentials_and_explanations(self):
        data = games()
        sides = ss.scoring_rows(data, 'differential')
        for i, (off, defense) in enumerate([('away', 'home'), ('home', 'away')]):
            self.assertAlmostEqual(sides.iloc[i * len(data)]['diff_stuff_%'],
                                   data.iloc[0][f'{off}_raw_off_stuff_%'] - data.iloc[0][f'{defense}_raw_def_stuff_%'])
        target, x, y, xp, offset = ss.prepare(data, 2026, 1, inputs='differential')
        self.assertEqual(x.shape[1], len(ss.METRICS) + 2)
        raw_train = sides.iloc[np.r_[np.arange(8), np.arange(9, 17)]]
        column = 'diff_stuff_%'
        j = ss.feature_names('differential').index(column)
        expected = (sides.iloc[8][column] - raw_train[column].mean()) / raw_train[column].std(ddof=0)
        self.assertAlmostEqual(xp[0, j], expected, places=5)
        run = ss.fit_member(0, x, y, xp, 13, 1, len(ss.METRICS))
        details, importance = ss.summarize(target, [run, run], offset)
        self.assertIn('diff_stuff_%', importance.feature.values)
        self.assertNotIn('off_stuff_%', importance.feature.values)
        self.assertLess(abs(details.integration_residual.iloc[0]), .05)
        self.assertLess(abs(details.total_integration_residual.iloc[0]), .05)

    def test_context_categories_use_training_only(self):
        data = games().assign(stadium_id='old', surface='grass', roof='outdoors', referee='known')
        data.loc[8, ['stadium_id', 'referee']] = ['new', 'unseen']
        target, x, y, xp, _ = ss.prepare(data, 2026, 1, ['stadium', 'referee'])
        names = target.attrs['context_names']
        self.assertNotIn('stadium:stadium_id=new', names)
        self.assertNotIn('referee:referee=unseen', names)
        np.testing.assert_array_equal(xp[:, len(ss.FEATURES)+2:], 0)
        stadium = len(ss.FEATURES) + names.index('stadium:stadium_id=old')
        np.testing.assert_array_equal(x[:len(y)//2, stadium], 0)
        np.testing.assert_array_equal(x[len(y)//2:, stadium], 1)

    def test_attribution_uses_the_actual_input_column_order(self):
        target = games().iloc[[-1]]
        attrs = np.zeros((2, len(ss.FEATURES) + 2))
        attrs[0, ss.FEATURES.index('off_explosive_pass_%')] = 1.
        attrs[0, ss.FEATURES.index('def_explosive_pass_%')] = 2.
        attrs[1, ss.FEATURES.index('off_qb_elo')] = 4.
        run = (np.array([3., 4.]), attrs, np.zeros(2), np.zeros(attrs.shape[1]))
        details, _ = ss.summarize(target, [run, run], 0)
        self.assertEqual(details['attr_away_off_explosive_pass_%'].iloc[0], 3.)
        self.assertEqual(details['attr_away_def_qb_elo'].iloc[0], -4.)
        self.assertEqual(details['attr_away_off_qb_elo'].iloc[0], 0.)
        self.assertEqual(details.integration_residual.iloc[0], 0.)
        self.assertEqual(ss.scoring_rows(games()).columns.tolist(),
                         ss.FEATURES + ['home_field', 'rest_advantage'])

    def test_same_roles_for_both_teams_and_no_future_leakage(self):
        data = games()
        sides = ss.scoring_rows(data)
        n = len(data)
        self.assertEqual(sides.iloc[0].off_qb_elo, data.iloc[0].away_raw_off_qb_elo)
        self.assertEqual(sides.iloc[n].off_qb_elo, data.iloc[0].home_raw_off_qb_elo)
        self.assertEqual(sides.iloc[n].def_qb_elo, data.iloc[0].away_raw_def_qb_elo)
        _, x, y, xp, offset = ss.prepare(data, 2026, 1)
        data.loc[8, ['away_score', 'home_score']] = 9999
        data = pd.concat([data, data.iloc[[-1]].assign(week=2)])
        _, x2, y2, xp2, offset2 = ss.prepare(data, 2026, 1)
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(y, y2)
        np.testing.assert_array_equal(xp, xp2)
        self.assertEqual(offset, offset2)

    def test_model_swap_symmetry_context_and_explanations(self):
        from modelo_workers import initialize_worker
        initialize_worker()
        import tensorflow as tf
        tf.keras.utils.set_random_seed(11)
        model = ss.build_model()
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, len(ss.FEATURES) + 2)).astype('float32')
        x[:, -2:] = 0
        scores = np.asarray(model(x, training=False)).ravel()
        swapped = np.asarray(model(x[::-1], training=False)).ravel()
        np.testing.assert_allclose(scores, swapped[::-1], atol=1e-6)
        self.assertAlmostEqual(float(scores[0] - scores[1]), -float(swapped[0] - swapped[1]), places=6)
        model.get_layer('venue_and_rest').set_weights([np.array([[2.], [.5]], dtype='float32')])
        context = x.copy()
        context[0, -2:] = [1, 2]
        changed = np.asarray(model(context, training=False)).ravel()
        self.assertAlmostEqual(float(changed[0] - scores[0]), 3., places=5)
        _, train_x, y, xp, offset = ss.prepare(games(), 2026, 1)
        run = ss.fit_member(0, train_x, y, xp, 13, 2)
        target = games().iloc[[-1]]
        # Identical members test accounting; this is not a production ensemble run.
        details, importance = ss.summarize(target, [run, run], offset)
        self.assertAlmostEqual(details.prediction.iloc[0],
                               details.away_points.iloc[0] - details.home_points.iloc[0], places=5)
        self.assertAlmostEqual(details.baseline.iloc[0], 0., places=5)
        self.assertLess(abs(details.integration_residual.iloc[0]), .05)
        self.assertEqual(details.variance.iloc[0], 0.)
        self.assertEqual(len(importance), len(ss.FEATURES) + 2)
        context_model = ss.build_model(3)
        self.assertIsNotNone(context_model.get_layer('context_groups').kernel_regularizer)
        context_data = games().assign(stadium_id='test')
        ct, cx, cy, cp, co = ss.prepare(context_data, 2026, 1, ['stadium'])
        cr = ss.fit_member(0, cx, cy, cp, 13, 1)
        cd, ci = ss.summarize(ct, [cr, cr], co)
        self.assertIn('attr_context_stadium', cd)
        self.assertIn('total_attr_context_stadium', cd)
        self.assertLess(abs(cd.total_integration_residual.iloc[0]), .05)

    def test_margin_variance_uses_paired_scores(self):
        target = games().iloc[[-1]]
        attrs = np.zeros((2, len(ss.FEATURES) + 2))
        runs = [(np.array(scores), attrs, np.zeros(2), np.zeros(attrs.shape[1]))
                for scores in [[20, 17], [30, 27]]]
        details, _ = ss.summarize(target, runs, 0)
        self.assertEqual(details.prediction.iloc[0], 3)
        self.assertEqual(details.variance.iloc[0], 0)
        self.assertEqual(details.total_prediction.iloc[0], 47)
        self.assertEqual(details.total_variance.iloc[0], 200)
        total = ss.market_details(details, 'total')
        self.assertEqual(total.prediction.iloc[0], 47)
        self.assertEqual(total.variance.iloc[0], 200)


if __name__ == '__main__':
    unittest.main()
