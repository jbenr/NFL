"""Regression checks for inference, CPU workers, output charts, and bad caches."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import model_shredski as ms
import tensorflow as tf


def sample_data():
    metrics = ['run_ypp', 'pass_ypp', 'pass_completion_%', 'series_success_%',
               'first_down_pp', 'third_down_%', 'fourth_down_%', 'turnovers_pp',
               'penalties_pp', 'qb_elo', 'explosive_run_%', 'explosive_pass_%',
               'stuff_%', 'sack_%', 'qb_hit_%']
    features = [f'away_{side}_{metric}' for metric in metrics for side in ('off', 'def')]
    features += ['away_rest_adv', 'home_field_adv']
    rng = np.random.default_rng(23)
    data = pd.DataFrame(rng.normal(0, 0.2, (50, 32)), columns=features)
    data['season'] = 2025
    data['week'] = [19] * 48 + [20] * 2
    data['away_team'] = ['SEA'] * 49 + ['BUF']
    data['home_team'] = ['NE'] * 49 + ['DEN']
    data['away_score'] = rng.integers(3, 35, 50).astype(float)
    data['home_score'] = rng.integers(3, 35, 50).astype(float)
    data.loc[data.week == 20, ['away_score', 'home_score']] = np.nan
    return data, features


class ModeloTests(unittest.TestCase):
    def test_totals_and_exported_point_accounting(self):
        data, features = sample_data()
        with tempfile.TemporaryDirectory(prefix='modelo-total-') as folder:
            with patch.object(ms, 'save_feature_importance_hbar'), patch.object(ms, 'save_matchup_edges_hbar'):
                result = ms.modelo(data, 2025, 20, folder, n_jobs=1, iterations=2,
                                   epochs=2, random_state=1337, market='total',
                                   features=features[:3], round_predictions=False)
            details = pd.read_csv(Path(folder) / 'explanations.csv')
            np.testing.assert_allclose(result.prediction, details.prediction)
            np.testing.assert_allclose(details.prediction, details.baseline + details.filter(like='attr_').sum(axis=1) + details.integration_residual)
            self.assertTrue((result.prediction > 0).all())

    def test_future_training_rows_are_excluded(self):
        data, _ = sample_data()
        future = data.iloc[:2].copy().assign(season=2026, away_score=np.nan)
        with tempfile.TemporaryDirectory() as folder:
            first = ms.modelo(data, 2025, 20, folder, bt=True, n_jobs=1, iterations=1, epochs=1, random_state=7)
            second = ms.modelo(pd.concat([data, future]), 2025, 20, folder, bt=True,
                               n_jobs=1, iterations=1, epochs=1, random_state=7)
            pd.testing.assert_frame_equal(first, second)

    def test_batched_importance_matches_original_predict_loop(self):
        data, features = sample_data()
        x = data.loc[:47, features].to_numpy(dtype=np.float32)
        y = np.linspace(-2, 2, len(x), dtype=np.float32)
        tf.keras.utils.set_random_seed(123)
        model = ms.create_model(len(features))
        model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True), loss=ms.sign_penalty)
        model.fit(x, y, epochs=2, verbose=0)
        baseline = model.predict(x, verbose=0).reshape(-1)
        np.testing.assert_allclose(np.asarray(model(x, training=False)).reshape(-1),
                                   baseline, rtol=1e-5, atol=1e-6)
        base_loss = float(ms.sign_penalty(tf.constant(y), tf.constant(baseline)))
        rng = np.random.default_rng(42)
        expected = []
        for j in range(x.shape[1]):
            perm = x.copy()
            rng.shuffle(perm[:, j])
            prediction = model.predict(perm, verbose=0).reshape(-1)
            expected.append(float(ms.sign_penalty(tf.constant(y), tf.constant(prediction))) - base_loss)
        actual = ms.permutation_importance(model, x, y)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-6)
        tf.keras.backend.clear_session()

    def test_seeded_serial_and_parallel_results_and_explanations(self):
        data, _ = sample_data()
        original = data.copy(deep=True)
        with tempfile.TemporaryDirectory(prefix='modelo-test-') as folder:
            with patch.object(ms, 'save_feature_importance_hbar') as fi, \
                 patch.object(ms, 'save_matchup_edges_hbar') as edge:
                serial = ms.modelo(data, 2025, 20, folder, n_jobs=1,
                                   iterations=2, epochs=3, random_state=100)
                serial_fi = fi.call_args.kwargs['mean_imp'].copy()
                serial_edges = [call.kwargs['edge_series'].copy() for call in edge.call_args_list]
            with patch.object(ms, 'save_feature_importance_hbar') as fi, \
                 patch.object(ms, 'save_matchup_edges_hbar') as edge:
                parallel = ms.modelo(data, 2025, 20, folder, n_jobs=2,
                                     iterations=2, epochs=3, random_state=100)
                np.testing.assert_allclose(fi.call_args.kwargs['mean_imp'], serial_fi, atol=1e-4)
                self.assertEqual(edge.call_count, 2)
                for call, expected in zip(edge.call_args_list, serial_edges):
                    np.testing.assert_allclose(call.kwargs['edge_series'], expected, atol=1e-5)
            bt_path = Path(folder) / 'backtest'
            bt = ms.modelo(data, 2025, 20, bt_path, bt=True, n_jobs=2,
                           iterations=2, epochs=3, random_state=100)
            self.assertFalse(bt_path.exists())
        self.assertEqual(list(serial.columns), ['away_team', 'home_team', 'prediction', 'variance'])
        self.assertEqual(serial.away_team.tolist(), ['SEA', 'BUF'])
        self.assertTrue(np.isfinite(serial[['prediction', 'variance']]).all().all())
        pd.testing.assert_frame_equal(serial, parallel, atol=1e-5, rtol=1e-5)
        pd.testing.assert_frame_equal(parallel, bt, atol=1e-5, rtol=1e-5)
        pd.testing.assert_frame_equal(data, original)

    def test_charts_and_single_iteration_variance(self):
        data, _ = sample_data()
        with tempfile.TemporaryDirectory(prefix='modelo-test-') as folder:
            result = ms.modelo(data, 2025, 20, folder, n_jobs=1, iterations=1, epochs=1)
            self.assertTrue((result.variance == 0).all())
            for filename in ['feature_importance.png', 'edge_SEA_@_NE.png', 'edge_BUF_@_DEN.png']:
                self.assertGreater((Path(folder) / filename).stat().st_size, 1000)

    def test_bad_data_fails_before_workers_start(self):
        data, features = sample_data()
        with patch.object(ms, 'Parallel') as pool:
            with self.assertRaisesRegex(ValueError, 'No prediction rows'):
                ms.modelo(data, 2025, 22, 'unused')
            for bad in (np.nan, np.inf):
                broken = data.copy()
                broken.loc[0, 'away_score'] = bad
                with self.assertRaisesRegex(ValueError, 'Non-finite training targets'):
                    ms.modelo(broken, 2025, 20, 'unused')
            broken = data.copy()
            broken.loc[49, features[0]] = np.nan
            with self.assertRaisesRegex(ValueError, 'Non-finite prediction features'):
                ms.modelo(broken, 2025, 20, 'unused')
            pool.assert_not_called()


if __name__ == '__main__':
    unittest.main()
