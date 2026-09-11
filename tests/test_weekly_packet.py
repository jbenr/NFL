import unittest
from unittest.mock import patch
import pandas as pd
import weekly_packet as wp


class PacketTests(unittest.TestCase):
    def test_context_row_order(self):
        row = pd.Series(dict(away_team='ARI', home_team='LAC', market='spread',
                             baseline=0., prediction=0., attr_away_rest_adv=0.,
                             attr_context_referee=0., attr_home_field_adv=0.))
        html = wp.matchup_attribution(row, pd.DataFrame(), pd.DataFrame(), shared=True)
        labels = [wp.pretty(f) for f in ['home_field_adv', 'context_referee', 'away_rest_adv']]
        positions = [html.index(f'class="stat-name">{label}</div>') for label in labels]
        self.assertEqual(positions, sorted(positions))

    def test_home_field_grouping_preserves_sum_and_inputs(self):
        row = pd.Series(dict(away_team='ARI', home_team='LAC', market='spread',
                             baseline=0., prediction=6., attr_home_field_adv=1.,
                             attr_context_stadium=2., attr_context_field=3.))
        before = row.copy()
        html = wp.matchup_attribution(row, pd.DataFrame(), pd.DataFrame(), shared=True)
        self.assertIn('>-6.0</b>', html)
        self.assertNotIn('class="reconcile"', html)
        self.assertNotIn('Baseline', html)
        self.assertIn('stat-line context-row', html)
        self.assertNotIn('Context stadium', html)
        self.assertNotIn('Context field', html)
        pd.testing.assert_series_equal(row, before)

    @patch.object(wp, 'logo', return_value='')
    def test_team_framed_header_includes_total(self, _):
        row = pd.Series(dict(away_team='DEN', home_team='KC', market_base=-3.,
                             prediction=6., edge=9., variance=9., away_points=22.,
                             home_points=16., total_line=43.5))
        html = wp.game_header(row, 'spread', 'PASS')
        self.assertIn('match-banner', html)
        self.assertIn('O/U · Market 43.5 · Model 38.0 · Under 5.5', html)
        self.assertIn('DEN 22.0 — KC 16.0', html)
        self.assertIn('class="scoreboard"', html)
        self.assertIn('class="score-dash">-</span>', html)
        self.assertIn(f'background:{wp.team_color("DEN")}', html)
        self.assertIn(f'background:{wp.team_color("KC")}', html)

    @patch.object(wp, 'logo', return_value='')
    def test_qbs_and_elo_in_header(self, _):
        row = pd.Series(dict(away_team='ARI', home_team='LAC', market_base=-10.,
                             prediction=-12.7, edge=-2.7, variance=3.24,
                             away_qb_name='Away Starter', home_qb_name='Home Starter',
                             away_raw_off_qb_elo=46.86, home_raw_off_qb_elo=42.80))
        html = wp.game_header(row, 'spread', 'PASS')
        self.assertIn('Away Starter', html)
        self.assertIn('(Elo 46.9)', html)
        self.assertIn('Home Starter', html)
        self.assertIn('(Elo 42.8)', html)

    @patch.object(wp, 'logo', return_value='')
    def test_compact_header_names_spread_team(self, _):
        row = pd.Series(dict(away_team='ARI', home_team='LAC', market_base=-10.,
                             prediction=-12.7, edge=-2.7, variance=3.24))
        html = wp.game_header(row, 'spread', 'PASS')
        self.assertIn('ARI +12.7', html)
        self.assertIn('LAC 2.7', html)
        self.assertIn('game-line', html)
        self.assertNotIn('class="metrics"', html)

    def test_qb_and_penalties_are_ranked(self):
        stats = pd.DataFrame({'team': ['ARI', 'LAC'], 'off_qb_elo': [40, 60],
                              'def_qb_elo': [-10, 5], 'off_penalties_pp': [.05, .1],
                              'def_penalties_pp': [.08, .04]})
        self.assertIn('(#1)', wp.stat_cell(stats, 'LAC', 'off', 'qb_elo'))
        self.assertIn('(#1)', wp.stat_cell(stats, 'ARI', 'def', 'qb_elo'))
        self.assertIn('(#1)', wp.stat_cell(stats, 'ARI', 'off', 'penalties_pp'))
        self.assertIn('(#1)', wp.stat_cell(stats, 'LAC', 'def', 'penalties_pp'))

    @patch.object(wp, 'packet_schedule')
    def test_stadium_and_rest_context(self, schedule):
        schedule.return_value = pd.DataFrame([dict(season=2026, week=1, away_team='ARI',
            home_team='LAC', away_rest=10, home_rest=7, stadium='SoFi Stadium', location='Home')])
        row = schedule.return_value.iloc[0].copy()
        self.assertEqual(('Home field', 'SoFi Stadium', 'Home'), wp.context_cells('home_field_adv', row))
        self.assertEqual(('Rest (Δ +3d)', 'ARI 10d', 'LAC 7d'), wp.context_cells('away_rest_adv', row))
        row['referee_avg_total'], row['referee_prior_games'] = 46., 0
        self.assertEqual(('Referee', 'Unassigned', 'Avg 46.0 · n=0'), wp.context_cells('context_referee', row))
    @patch.object(wp, 'team_colors', return_value={'ARI': '#97233F', 'LAC': '#0080C6', 'LAR': '#003594'})
    def test_team_colors_follow_team_not_side(self, _):
        row = pd.Series(dict(away_team='ARI', home_team='LAC', market='spread',
                             prediction=0, baseline=0, attr_away_off_sack_pct=1,
                             attr_away_def_sack_pct=-1))
        row.index = row.index.str.replace('pct', '%')
        html = wp.matchup_attribution(row, pd.DataFrame(), pd.DataFrame())
        self.assertIn('background:#97233F;left:6.00%', html)
        self.assertIn('background:#0080C6;left:50.00%', html)
        self.assertEqual('#003594', wp.team_color('LA'))
        self.assertEqual('#888888', wp.team_color('UNKNOWN'))

    def test_ranks_use_metric_direction_and_keep_missing(self):
        stats = pd.DataFrame(dict(team=['ARI', 'LAC', 'BUF'],
                                  off_stuff_pct=[.1, .2, None],
                                  def_stuff_pct=[.1, .2, None])).rename(
                                      columns=lambda c: c.replace('pct', '%'))
        self.assertIn('#1', wp.stat_cell(stats, 'ARI', 'off', 'stuff_%'))
        self.assertIn('#1', wp.stat_cell(stats, 'LAC', 'def', 'stuff_%'))
        self.assertEqual('—', wp.stat_cell(stats, 'BUF', 'off', 'stuff_%'))

    @patch.object(wp, 'logo', return_value='')
    def test_defense_feature_keeps_away_team_on_left(self, _):
        row = pd.Series(dict(away_team='ARI', home_team='LAC', market='spread',
                             prediction=-2, baseline=1, attr_away_def_fourth_down_pct=-3))
        row.index = row.index.str.replace('pct', '%')
        stats = pd.DataFrame({'team': ['ARI', 'LAC'], 'off_fourth_down_%': [.2, .6],
                              'def_fourth_down_%': [.7, .4]})
        html = wp.matchup_attribution(row, stats, pd.DataFrame())
        self.assertIn('LAC offense', html)
        self.assertIn('ARI defense', html)
        self.assertIn('60.0%', html)
        self.assertIn('70.0%', html)
        self.assertNotIn('Net +3.0 pts', html)
        self.assertIn('>+3.0</b>', html)
        self.assertIn('class="net-bar"', html)
        self.assertNotIn('Matchup net', html)
        self.assertLess(html.index('ARI defense'), html.index('LAC offense'))
        self.assertIn('<span class="rank">(#1)</span> 60.0%', html)
        self.assertIn('<details>', html)
        self.assertNotIn('<table', html)
        self.assertIn('stat-line', html)
        self.assertIn('contribution', html)
        self.assertNotIn('class="stat-bars"', html)

    def test_contribution_points_toward_favored_team(self):
        row = pd.Series(dict(market='spread', away_team='ARI', home_team='LAC'))
        self.assertIn('left:6.00%', wp.point_bar(2, 2, row, left_team='ARI'))
        self.assertIn('left:50.00%', wp.point_bar(2, 2, row, left_team='LAC'))
        self.assertIn('left:6.00%', wp.point_bar(-2, 2, row, left_team='LAC'))
        self.assertIn('>-2.0</b>', wp.point_bar(2, 2, row))
        self.assertIn('>+2.0</b>', wp.point_bar(-2, 2, row))
        self.assertIn('width:0.00%', wp.point_bar(0, 2, row, left_team='LAC'))

    def test_comparison_bars_show_rates_not_attribution(self):
        stats = pd.DataFrame({'team': ['ARI', 'LAC'], 'off_fourth_down_%': [.25, .6],
                              'def_fourth_down_%': [.7, .75]})
        html = wp.comparison_bars(stats, 'ARI', 'LAC', 'fourth_down_%')
        self.assertIn('flex:0.250000', html)
        self.assertIn('flex:0.750000', html)
        self.assertEqual('', wp.comparison_bars(stats, 'ARI', 'LAC', 'qb_elo'))
        self.assertEqual('', wp.comparison_bars(stats, 'BUF', 'LAC', 'fourth_down_%'))

    def test_total_direction(self):
        self.assertIn('>-2.0</b>', wp.point_bar(-2, 3, pd.Series({'market': 'total'})))
        self.assertIn('>+2.0</b>', wp.point_bar(2, 3, pd.Series({'market': 'total'})))

    def test_tabs_keep_headline_first(self):
        html = wp.packet_tabs('total')
        self.assertLess(html.index('Headline'), html.index('Spread'))
        self.assertIn('href="importance.html"', html)
        self.assertIn('href="total.html" aria-current="page"', html)


if __name__ == '__main__':
    unittest.main()
