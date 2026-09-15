import re
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import weekly_packet as wp


class HeadlineTableTests(unittest.TestCase):
    def _details(self, market, edge, sd):
        # headline_table reads market_base/prediction/edge/variance directly
        # regardless of market -- by the time two_sided_packet saves
        # {market}_details.csv, market_details() has already made these
        # columns market-appropriate (e.g. 'prediction' is total_prediction
        # for the total market's own saved file).
        return pd.DataFrame(dict(
            season=[2026], week=[1], away_team=['ARI'], home_team=['LAC'],
            market_base=[-9.5], prediction=[-9.5 + edge], edge=[edge],
            variance=[sd ** 2], positive_odds=[-110.], negative_odds=[-110.],
            residual=[float('nan')]))

    @patch.object(wp, 'logo', return_value='')
    @patch.object(wp, 'packet_schedule', return_value=pd.DataFrame(dict(
        season=[2026], week=[1], away_team=['ARI'], home_team=['LAC'], gameday=['2026-09-13'], gametime=['16:25'])))
    def test_pick_only_shown_when_cutoffs_are_actually_met(self, *_):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            cutoffs = wp.HIGH_CONFIDENCE_CUTOFFS['spread']
            # Edge clears diff_cutoff but SD is too wide -- should still PASS.
            self._details('spread', cutoffs['diff_cutoff'] + 1, cutoffs['sd_cutoff'] + 1).to_csv(
                folder / 'spread_details.csv', index=False)
            html = wp.headline_table(folder)
            self.assertIn('>PASS</td>', html)
            self.assertNotIn('#ffe590', html)  # nothing qualifies -- no yellow highlight anywhere

    @patch.object(wp, 'logo', return_value='')
    @patch.object(wp, 'packet_schedule', return_value=pd.DataFrame(dict(
        season=[2026], week=[1], away_team=['ARI'], home_team=['LAC'], gameday=['2026-09-13'], gametime=['16:25'])))
    def test_pick_shown_when_both_cutoffs_are_met(self, *_):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            cutoffs = wp.HIGH_CONFIDENCE_CUTOFFS['spread']
            # Positive edge -> away team (ARI) qualifies as the pick.
            self._details('spread', cutoffs['diff_cutoff'] + 1, cutoffs['sd_cutoff'] - 1).to_csv(
                folder / 'spread_details.csv', index=False)
            html = wp.headline_table(folder)
            self.assertIn('>ARI</td>', html)
            self.assertNotIn('>PASS</td>', html)
            # Real pick -> both the pick cell and the matching away_team
            # cell get the yellow highlight (pandas consolidates identical
            # per-cell styles into one shared CSS rule with 2+ selectors).
            self.assertIn('#ffe590', html)
            rule = html.split('background-color: #ffe590')[0].rsplit('{', 1)[0]
            self.assertGreaterEqual(rule.count('#T_'), 2)
            self.assertIn('>Picks</th>', html)
            self.assertNotIn('>Calls</th>', html)

    @patch.object(wp, 'logo', return_value='')
    @patch.object(wp, 'packet_schedule', return_value=pd.DataFrame(dict(
        season=[2026], week=[1], away_team=['ARI'], home_team=['LAC'], gameday=['2026-09-13'], gametime=['16:25'])))
    def test_light_theme_table_is_white_for_copying(self, *_):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            cutoffs = wp.HIGH_CONFIDENCE_CUTOFFS['spread']
            self._details('spread', cutoffs['diff_cutoff'] + 1, cutoffs['sd_cutoff'] - 1).to_csv(
                folder / 'spread_details.csv', index=False)
            dark = wp.headline_table(folder)
            light = wp.headline_table(folder, light=True)
            self.assertIn('class="headline-table"', dark)
            self.assertIn('class="headline-table-light"', light)
            self.assertNotIn('headline-table-light', dark)
            self.assertIn('>ARI</td>', dark)
            self.assertIn('>ARI</td>', light)  # same real data, just a different theme

    def test_copy_picks_widget_has_title_subtitle_and_white_source(self):
        html = wp.copy_picks_widget(2026, 1, '<table class="headline-table-light"><tr><td>x</td></tr></table>')
        self.assertIn('>Copy<', html)
        self.assertIn('class="copy-title">Model<', html)
        self.assertIn('class="copy-subtitle">2026 Week 1<', html)
        self.assertIn('id="picks-copy-source"', html)
        self.assertIn('headline-table-light', html)  # embeds the white-themed table, not the dark on-page one
        self.assertIn('copyPicksTable', html)
        self.assertIn("document.execCommand(\"copy\")", html)
        self.assertIn('id="copy-feedback" class="copy-feedback">Headline table copied</span>', html)
        # Restarts the fade animation on a second click before the first
        # one finishes -- classList.remove + a forced reflow + re-add.
        self.assertIn('feedback.classList.remove("show")', html)
        self.assertIn('void feedback.offsetWidth', html)
        self.assertIn('feedback.classList.add("show")', html)

    def test_copy_subtitle_uses_nfl_font(self):
        self.assertIn('.copy-subtitle{font-size:13px;margin-bottom:10px;color:#444;font-family:Graduate,Georgia,serif}', wp.STYLE)

    @patch.object(wp, 'logo', return_value='')
    @patch.object(wp, 'packet_schedule', return_value=pd.DataFrame(dict(
        season=[2026], week=[1], away_team=['ARI'], home_team=['LAC'], gameday=['2026-09-13'], gametime=['16:25'])))
    def test_missing_market_shows_placeholder_not_a_crash(self, *_):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            cutoffs = wp.HIGH_CONFIDENCE_CUTOFFS['spread']
            self._details('spread', cutoffs['diff_cutoff'] + 1, cutoffs['sd_cutoff'] - 1).to_csv(
                folder / 'spread_details.csv', index=False)
            with warnings.catch_warnings():
                # write_packets() hits exactly this shape on its first (spread)
                # call each run, before total_details.csv exists -- an all-NaN
                # total_diff/total_sd column used to trip a real numpy
                # RuntimeWarning (nanmin/nanmax of nothing) on every run.
                warnings.simplefilter('error', RuntimeWarning)
                html = wp.headline_table(folder)  # no total_details.csv at all
            self.assertIn('>—</td>', html)  # blank O/U cells, not a crash

    def test_no_details_files_returns_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(wp.headline_table(Path(folder)), '')


class PacketTests(unittest.TestCase):
    def _schedule(self, seasons=(2025, 2026), weeks=range(1, 19), playoff_weeks=(19, 20, 21, 22)):
        rows = [dict(season=s, week=w, game_type='REG') for s in seasons for w in weeks]
        rows += [dict(season=s, week=w, game_type='POST') for s in seasons for w in playoff_weeks]
        return pd.DataFrame(rows)

    def test_lookback_span_includes_playoffs_without_counting_them(self):
        # 15-week lookback from 2026 wk1 reaches back to 2025 wk4 (15 REG
        # weeks: 4..18) and keeps going through the 2025 postseason -- those
        # 4 playoff weeks land inside the span (their stats count) but don't
        # themselves consume one of the 15 backward steps (only REG weeks do).
        with patch.object(wp.pd, 'read_parquet', return_value=self._schedule()):
            span = wp.lookback_span(2026, 1, 15)
        self.assertEqual(span.index[0], (2025, 4))
        self.assertEqual(span.index[-1], (2025, 22))  # includes the 2025 postseason
        self.assertEqual(int(span.sum()), 15)  # only 15 of the included weeks are REG
        self.assertEqual(int((~span).sum()), 4)  # the 4 playoff weeks

    def test_lookback_description_names_the_real_week_range(self):
        with patch.object(wp.pd, 'read_parquet', return_value=self._schedule()):
            html = wp.lookback_description(2026, 1, 15)
        self.assertIn('<details><summary>How these stats are calculated</summary>', html)
        self.assertIn('2025 Weeks 4–22', html)
        self.assertIn('4 of those are playoff weeks', html)
        self.assertIn('QB Elo', html)
        # Raw/Model explanation lives in this same dropdown now, not as its
        # own separate always-visible paragraph next to the toggle.
        self.assertIn('Raw: flat average over the lookback window', html)
        self.assertTrue(html.rstrip().endswith('</details>'))

    def test_lookback_description_spans_a_season_boundary(self):
        # 20-week lookback from 2026 wk1 needs more REG weeks than 2025
        # alone has (18) -- reaches back into 2024 too, so the description
        # should name both seasons, not just one.
        schedule = self._schedule(seasons=(2024, 2025, 2026))
        with patch.object(wp.pd, 'read_parquet', return_value=schedule):
            html = wp.lookback_description(2026, 1, 20)
        self.assertIn('2024 Week 17 through 2025 Week 22', html)

    def test_lookback_description_empty_when_no_history(self):
        with patch.object(wp.pd, 'read_parquet', return_value=self._schedule(seasons=[2026])):
            self.assertEqual(wp.lookback_description(2026, 1, 20), '')

    def test_stats_include_qb_and_both_units_with_directional_ranks(self):
        stats = pd.DataFrame(dict(team=['ARI', 'LAC'], off_qb_elo=[40., 60.], def_qb_elo=[30., 50.]))
        games = pd.DataFrame(dict(away_team=['ARI'], home_team=['LAC'], away_qb_short=['A.QB'], home_qb_short=['B.QB']))
        html = wp.stats_tables(stats, games)
        for text in ['QB Elo', 'Offense', 'Defense', 'A.QB', 'B.QB', '(#1)', 'sortStats(this)']:
            self.assertIn(text, html)
        self.assertLess(html.index('B.QB'), html.index('A.QB'))

    def test_raw_model_toggle_only_shown_when_weighted_stats_given(self):
        raw = pd.DataFrame(dict(team=['ARI', 'LAC'], off_pass_ypp=[6., 7.], def_pass_ypp=[6.5, 7.5]))
        weighted = pd.DataFrame(dict(team=['ARI', 'LAC'], off_pass_ypp=[6.2, 6.8], def_pass_ypp=[6.6, 7.4]))
        games = pd.DataFrame({'away_team': [], 'home_team': []})
        without_toggle = wp.stats_tables(raw, games)
        self.assertNotIn('stats-mode-radio', without_toggle)
        self.assertNotIn('id="stats-view-raw"', without_toggle)
        with_toggle = wp.stats_tables(raw, games, weighted_stats=weighted)
        self.assertIn('id="stats-mode-raw" class="stats-mode-radio" checked', with_toggle)  # raw shown by default
        self.assertIn('id="stats-mode-model" class="stats-mode-radio"', with_toggle)
        self.assertIn('id="stats-view-raw"', with_toggle)
        self.assertIn('id="stats-view-model"', with_toggle)
        # Same table (Passing), different ids so both copies can coexist
        # in one document -- and each side shows its own numbers.
        self.assertIn('id="off-passing-raw"', with_toggle)
        self.assertIn('id="off-passing-model"', with_toggle)
        raw_section = with_toggle[with_toggle.index('id="stats-view-raw"'):with_toggle.index('id="stats-view-model"')]
        model_section = with_toggle[with_toggle.index('id="stats-view-model"'):]
        self.assertIn('6.0', raw_section)  # ARI's raw off_pass_ypp
        self.assertIn('6.2', model_section)  # ARI's weighted off_pass_ypp
        self.assertNotIn('6.0', model_section)

    def test_sorting_carries_over_between_raw_and_model_twins(self):
        # Clicking a column header on the Raw table shouldn't leave the
        # Model table (a separate <table> under the hood) back at its
        # original unsorted order once you flip the toggle.
        raw = pd.DataFrame(dict(team=['ARI', 'LAC'], off_pass_ypp=[6., 7.]))
        weighted = pd.DataFrame(dict(team=['ARI', 'LAC'], off_pass_ypp=[6.2, 6.8]))
        html = wp.stats_tables(raw, pd.DataFrame({'away_team': [], 'home_team': []}), weighted_stats=weighted)
        script = html[html.index('<script>'):]
        self.assertIn('function sortTable(table,i,direction)', script)
        self.assertIn("table.id.endsWith('-raw')", script)
        self.assertIn("table.id.slice(0,-4)+'-model'", script)
        self.assertIn("table.id.slice(0,-6)+'-raw'", script)
        self.assertIn('if(twin)sortTable(twin,i,direction)', script)

    def test_display_stats_calculation_is_part_of_the_cache_key(self):
        # Without this, calling display_stats with a different calculation
        # (e.g. the Raw/Model toggle's two calls) but the same season/week/
        # lookback would silently reuse whichever one got cached first.
        with patch('utils.cache_path') as cache_path, \
             patch.object(wp.pd, 'read_parquet', side_effect=FileNotFoundError):
            cache_path.return_value = Path('/nonexistent/does-not-exist.parquet')
            for calculation in ['mean', 'steep']:
                try:
                    wp.display_stats(2026, 1, 20, calculation=calculation)
                except Exception:
                    pass
        configs = [call.args[1] for call in cache_path.call_args_list]
        self.assertEqual(len(configs), 2)
        self.assertIn('mean', configs[0])
        self.assertIn('steep', configs[1])
        self.assertNotEqual(configs[0], configs[1])

    def test_qb_elo_table_grouped_scrollable_with_rank_column(self):
        stats = pd.DataFrame(dict(team=['ARI', 'LAC', 'BUF'], off_qb_elo=[40., 60., 50.],
                                  off_pass_ypp=[6., 7., 5.], def_pass_ypp=[6.5, 7.5, 5.5]))
        html = wp.stats_tables(stats, pd.DataFrame({'away_team': [], 'home_team': []}))
        self.assertIn('id="qb-elo"', html)
        # Every table scrolls past ~10 rows now, not just QB Elo.
        self.assertEqual(html.count('scroll-tall'), html.count('<table class="stats-table'))
        qb_elo_table = html[html.index('id="qb-elo"'):html.index('</table>')]
        # Rank leads (before Team), as a plain "1" -- not "(#1)" and not
        # repeated as the old trailing inline (#N) badge on the value cell.
        heads = re.findall(r'<th><button[^>]*>([^<]*)</button></th>', qb_elo_table)
        self.assertEqual(heads[:2], ['Rank', 'Team'])
        first_row = re.search(r'<tr>(.*?)</tr>', qb_elo_table[qb_elo_table.index('<tbody>'):], re.DOTALL).group(1)
        rank_cell = re.search(r'<td[^>]*>([^<]*)</td>', first_row).group(1)
        self.assertEqual(rank_cell, '1')  # LAC (elo 60) sorts first -- plain number, no parens/hash
        self.assertNotIn('<span class="rank">', qb_elo_table)
        # QB Elo stands alone above "Offense" now (own <h3>, not part of the
        # grid), and Offense's grid is just Passing/Rushing/Misc.
        self.assertLess(html.index('<h3>QB Elo</h3>'), html.index('<h2>Offense</h2>'))
        self.assertNotIn('class="stats-cell"><h3>QB Elo</h3>', html)
        # Offense/Defense each broken into Passing/Rushing/Misc sub-tables
        # instead of one wide table -- Passing present here since the
        # fixture has pass_ypp; Rushing/Misc correctly absent (no data).
        self.assertIn('<h3>Passing</h3>', html)
        self.assertNotIn('<h3>Rushing</h3>', html)
        self.assertIn('class="stats-grid"', html)

    def test_portable_downloads_fonts_and_stats_tab(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'spread_details.csv').write_text('team,points\nARI,21\n')
            content = '<a href="spread_details.csv">Download</a><a href="missing.html">Old page</a>'
            (folder / 'index.html').write_text(wp.page('Model', content, 'packet'))
            (folder / 'stats.html').write_text(wp.page('Stats', '<h1>Stats</h1>', 'packet'))
            html = wp.bundle_single_file(folder).read_text()
            self.assertIn('data:text/csv;base64,', html)
            self.assertIn('data:font/ttf;base64,', html)
            self.assertIn('SIL OPEN FONT LICENSE', html)
            self.assertIn('id="pt-stats"', html)
            self.assertNotIn('href="missing.html"', html)
            self.assertNotIn('href="spread_details.csv"', html)

    def test_context_row_order(self):
        # home_field_adv gets its own descriptive site-row (not a plain
        # stat-name bar); weather has no differential bar on the spread
        # page at all (see test_weather_bar_hidden_on_spread_but_still_
        # reconciles below); away_rest_adv/context_referee still render as
        # ordinary bar rows, in order, after the home-field site-row.
        row = pd.Series(dict(away_team='ARI', home_team='LAC', market='spread',
                             baseline=0., prediction=0., attr_away_rest_adv=0.,
                             attr_context_referee=0., attr_home_field_adv=0.,
                             attr_context_weather_feels_like_f=0.))
        html = wp.matchup_attribution(row, pd.DataFrame(), pd.DataFrame(), shared=True)
        self.assertNotIn('class="stat-name">Weather</div>', html)
        order = ['away_rest_adv', 'context_referee']
        positions = [html.index(f'class="stat-name">{wp.pretty(f)}</div>') for f in order]
        self.assertEqual(positions, sorted(positions))
        self.assertLess(html.index('class="site-label"'), positions[0])  # home field comes first

    def test_two_sided_weather_features_get_clean_labels(self):
        # Without these, pretty() falls through to its generic underscore-
        # replacement path and produces "Context weather feels like f" --
        # functional but redundant/clunky (context_weather is a different,
        # already-labeled single combined feature from an older model).
        self.assertEqual(wp.pretty('context_weather_feels_like_f'), 'Feels like (°F)')
        self.assertEqual(wp.pretty('context_weather_wind_mph'), 'Wind (mph)')
        self.assertNotIn('Context weather', wp.pretty('context_weather_snow_depth_inches'))

    def test_home_field_grouping_preserves_sum_and_inputs(self):
        # home_field_adv/context_stadium/context_field still get grouped
        # into one combined bar row and counted in the residual reconciliation.
        row = pd.Series(dict(away_team='ARI', home_team='LAC', market='spread',
                             baseline=0., prediction=6., attr_home_field_adv=1.,
                             attr_context_stadium=2., attr_context_field=3.))
        before = row.copy()
        html = wp.matchup_attribution(row, pd.DataFrame(), pd.DataFrame(), shared=True)
        self.assertIn('Numerical residual -0.0000 points', html)  # 6 - 0 - (1+2+3) == 0 -- grouped sum still counted
        self.assertNotIn('class="reconcile"', html)
        self.assertNotIn('Baseline', html)
        self.assertIn('class="site-label"', html)  # one combined home-field row (descriptive, not a bare stat-name)
        self.assertIn(f'<strong>{wp.pretty("home_field_adv")}</strong>', html)
        self.assertNotIn('Context stadium', html)
        self.assertNotIn('Context field', html)
        pd.testing.assert_series_equal(row, before)

    def test_weather_dimensions_grouped_into_one_combined_bar(self):
        # two_sided_packet's 7 separate weather attr_context_weather_* keys
        # collapse into a single "Weather" bar row -- on the total page,
        # where it's still shown as a differential (spread hides it; see
        # test_weather_bar_hidden_on_spread_but_still_reconciles below).
        row = pd.Series(dict(away_team='ARI', home_team='LAC', market='total',
                             baseline=0., prediction=5., roof='outdoors',
                             feels_like_f=61., wind_mph=8., rain_inches=0., snowfall_inches=0.,
                             attr_context_weather_feels_like_f=2., attr_context_weather_wind_mph=1.,
                             attr_context_weather_rain_inches=1., attr_context_weather_snowfall_inches=1.))
        html = wp.matchup_attribution(row, pd.DataFrame(), pd.DataFrame(), shared=True)
        self.assertIn('Numerical residual +0.0000 points', html)  # 5 - 0 - (2+1+1+1) == 0, direction=+1 for total
        self.assertEqual(html.count('class="stat-name">Weather</div>'), 1)  # combined into ONE row
        self.assertIn('61°F, wind 8 mph', html)
        self.assertIn('No precipitation', html)

    def test_weather_bar_hidden_on_spread_but_still_reconciles(self):
        # The spread page skips weather as its own differential bar (not
        # very meaningful there) but the value still has to survive in the
        # residual reconciliation and the calculation notes.
        row = pd.Series(dict(away_team='ARI', home_team='LAC', market='spread',
                             baseline=0., prediction=5., roof='outdoors',
                             feels_like_f=61., wind_mph=8., rain_inches=0., snowfall_inches=0.,
                             attr_context_weather_feels_like_f=2., attr_context_weather_wind_mph=1.,
                             attr_context_weather_rain_inches=1., attr_context_weather_snowfall_inches=1.))
        html = wp.matchup_attribution(row, pd.DataFrame(), pd.DataFrame(), shared=True)
        self.assertNotIn('class="stat-name">Weather</div>', html)
        self.assertIn('Weather contribution: -5.00 points (included in the model)', html)
        self.assertIn('Numerical residual -0.0000 points', html)  # 5 - 0 - 5 == 0, still reconciles

    def test_qb_elo_not_repeated_in_matchup_labels(self):
        # Starting QB + Elo shown right in the collapsed matchup-head, on
        # whichever side is offense that section -- visible without
        # expanding the per-metric breakdown.
        row = pd.Series(dict(away_team='ARI', home_team='LAC', market='spread',
                             baseline=0., prediction=0., attr_away_off_qb_elo=1., attr_away_def_qb_elo=-1.,
                             away_qb_short='K.Murray', home_qb_short='J.Herbert',
                             away_raw_off_qb_elo=45.3, home_raw_off_qb_elo=61.7))
        html = wp.matchup_attribution(row, pd.DataFrame(), pd.DataFrame(), shared=True)
        self.assertIn('ARI offense', html)
        self.assertIn('LAC offense', html)
        self.assertNotIn('K.Murray', html)
        self.assertNotIn('J.Herbert', html)
        self.assertNotIn('ARI defense · ', html)  # defense side never gets a QB tag
        self.assertNotIn('LAC defense · ', html)

    @patch.object(wp, 'logo', return_value='')
    def test_team_framed_header_keeps_it_to_the_spread(self, _):
        # Header shows only its own market -- no O/U line cluttering the
        # spread page (that info lives in the headline table instead).
        row = pd.Series(dict(away_team='DEN', home_team='KC', market_base=-3.,
                             prediction=6., edge=9., variance=9., away_points=22.,
                             home_points=16., total_line=43.5))
        html = wp.game_header(row, 'spread', 'PASS')
        self.assertIn('pick-header', html)
        self.assertNotIn('O/U', html)
        self.assertIn('Prediction: DEN 22.0 - 16.0 KC', html)

    @patch.object(wp, 'logo', return_value='')
    def test_compact_header_names_spread_team(self, _):
        row = pd.Series(dict(away_team='ARI', home_team='LAC', market_base=-10.,
                             prediction=-12.7, edge=-2.7, variance=3.24))
        html = wp.game_header(row, 'spread', 'PASS')
        self.assertIn('<div class="pick-team"><span class="venue-label">Away</span>ARI</div>', html)
        self.assertIn('<div class="pick-value">+10.0</div>', html)  # MarketLine, away perspective
        self.assertIn('<div class="pick-value">+12.7</div>', html)  # ModelLine
        self.assertIn('<div class="pick-value">2.7</div>', html)  # Edge
        self.assertIn('<div class="pick-value">1.8</div>', html)  # SD (sqrt of variance)

    @patch.object(wp, 'logo', return_value='')
    def test_qb_names_and_elo_under_team_names(self, _):
        # Removed once the headline table's O/U column made the game
        # header feel redundant -- QB/Elo detail still lives in the rich
        # matchup-attribution chart below the header, just not up top too.
        row = pd.Series(dict(away_team='ARI', home_team='LAC', market_base=-10.,
                             prediction=-12.7, edge=-2.7, variance=3.24,
                             away_qb_name='Away Starter', home_qb_name='Home Starter',
                             away_raw_off_qb_elo=46.86, home_raw_off_qb_elo=42.80))
        html = wp.game_header(row, 'spread', 'PASS')
        self.assertIn('Away Starter · 47 Elo', html)
        self.assertIn('Home Starter · 43 Elo', html)
        self.assertIn('class="venue-label">Away</span>', html)
        self.assertIn('class="venue-label">Home</span>', html)

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
        # 'location' is a Home/Neutral site-type flag, not a city -- only
        # worth surfacing in the unusual case (a neutral-site game), so the
        # ordinary 'Home' value shouldn't show up as if it were meaningful.
        schedule.return_value = pd.DataFrame([dict(season=2026, week=1, away_team='ARI',
            home_team='LAC', away_rest=10, home_rest=7, stadium='SoFi Stadium', location='Home')])
        row = schedule.return_value.iloc[0].copy()
        self.assertEqual(('Home field', 'SoFi Stadium', ''), wp.context_cells('home_field_adv', row))
        schedule.return_value.loc[0, 'location'] = 'Neutral'
        row = schedule.return_value.iloc[0].copy()
        self.assertEqual(('Home field', 'SoFi Stadium', 'Neutral site'), wp.context_cells('home_field_adv', row))
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

    def test_bundle_single_file_is_self_contained(self):
        # The four separately-written pages -> one file with CSS-only tabs,
        # no leftover cross-file nav/links that would break once detached.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'index.html').write_text(wp.page('2026 · Week 1', wp.packet_tabs('headline') + '<h1>Headline</h1>', 'packet headline-shell'), encoding='utf-8')
            (folder / 'spread.html').write_text(wp.page('2026 · Week 1', wp.packet_tabs('spread') + '<p>Spread card</p>', 'packet'), encoding='utf-8')
            (folder / 'total.html').write_text(wp.page('2026 · Week 1', wp.packet_tabs('total') + '<p>Total card</p>', 'packet'), encoding='utf-8')
            (folder / 'importance.html').write_text(wp.page('2026 · Week 1', wp.packet_tabs('importance') + '<p>Feature bars</p>', 'packet'), encoding='utf-8')
            out = wp.bundle_single_file(folder)
            self.assertEqual(out, folder / 'packet.html')
            html = out.read_text(encoding='utf-8')
            self.assertEqual(html.count('<style>'), 1)  # one shared style block, not four
            self.assertNotIn('<nav class="packet-tabs"', html)  # stale multi-file nav stripped
            for marker in ['Headline', 'Spread card', 'Total card', 'Feature bars']:
                self.assertIn(marker, html)
            self.assertEqual(html.count('class="pkgtab-radio"'), 4)
            self.assertEqual(html.count('class="pkgtab-label"'), 4)
            self.assertIn('id="pt-headline" class="pkgtab-radio" checked', html)  # headline opens by default

    def test_bundle_single_file_skips_missing_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'index.html').write_text(wp.page('t', '<h1>Only page</h1>', 'packet'), encoding='utf-8')
            out = wp.bundle_single_file(folder)
            self.assertEqual(out.read_text(encoding='utf-8').count('class="pkgtab-radio"'), 1)
            self.assertIsNone(wp.bundle_single_file(Path(directory) / 'nonexistent'))

    def test_feature_importance_shows_every_feature_scrollable(self):
        # Used to truncate to importance.head(12) -- now the full set
        # renders inside a scrolling container instead of getting cut off.
        row = pd.Series(dict(season=2026, week=1, away_team='ARI', home_team='LAC', market='spread',
                             market_base=-3., baseline=0., prediction=0., edge=1., variance=4.,
                             odds=-110., assumed_odds=True, actual=float('nan'), qualifies=False, pnl=0.,
                             # headline_table() re-settles the saved {market}_details.csv -- needs these.
                             positive_odds=-110., negative_odds=-110., residual=float('nan')))
        data = pd.DataFrame([row])
        n = 20
        importance = pd.DataFrame(dict(feature=[f'away_off_metric_{i}' for i in range(n)],
                                       importance=[float(n - i) for i in range(n)], std=[.1] * n))
        config = dict(model='test', calculation='two-sided-team-points-v1', market='spread',
                      lookback=20, status='PASS', reason='test')
        with tempfile.TemporaryDirectory() as directory, patch.object(wp, 'display_stats', return_value=pd.DataFrame()):
            wp.write_packets(data, data, importance, config, Path(directory))
            html = (Path(directory) / '2026_01' / 'importance.html').read_text()
            self.assertIn('class="importance-scroll"', html)
            self.assertEqual(html.count('class="barrow"'), n)  # every feature, not just the top 12
            self.assertIn('paired refit/drop test', html)  # the improved, plainer-English description
            spread = (Path(directory) / '2026_01' / 'spread.html').read_text()
            cards, notes = spread.split('<footer class="sheet-notes">')
            for label in ['Calculation notes', 'Pick details', 'Data & model notes']:
                self.assertNotIn(label, cards)
                self.assertIn(label, notes)
            self.assertNotIn('regular-week feature window', cards)

    def test_notes_are_generic_and_shown_once_not_per_matchup(self):
        # Calculation notes/Pick details used to repeat as a <details>
        # dropdown once per game -- same boilerplate every time. Now it's
        # one always-visible (no dropdown) statement in the footer.
        rows = [dict(season=2026, week=1, away_team=a, home_team=h, market='spread',
                     market_base=-3., baseline=0., prediction=0., edge=1., variance=4.,
                     odds=-110., assumed_odds=True, actual=float('nan'), qualifies=False, pnl=0.,
                     positive_odds=-110., negative_odds=-110., residual=float('nan'))
                for a, h in [('ARI', 'LAC'), ('DAL', 'NYG')]]
        data = pd.DataFrame(rows)
        importance = pd.DataFrame(dict(feature=['away_off_run_ypp'], importance=[1.], std=[.1]))
        config = dict(model='test', calculation='two-sided-team-points-v1', market='spread',
                      lookback=20, status='PASS', reason='test')
        with tempfile.TemporaryDirectory() as directory, patch.object(wp, 'display_stats', return_value=pd.DataFrame()):
            wp.write_packets(data, data, importance, config, Path(directory))
            html = (Path(directory) / '2026_01' / 'spread.html').read_text()
            self.assertNotIn('<details><summary>Calculation notes', html)
            self.assertNotIn('<details><summary>Pick details', html)
            self.assertEqual(html.count('<h3>Calculation notes</h3>'), 1)  # once, not once per game
            self.assertEqual(html.count('<h3>Pick details</h3>'), 1)

    @patch.object(wp, 'packet_schedule', return_value=pd.DataFrame(dict(
        season=[2026], week=[1], away_team=['ARI'], home_team=['LAC'], gameday=['2026-09-13'], gametime=['16:25'])))
    def test_headline_page_has_a_copy_button(self, _):
        row = pd.Series(dict(season=2026, week=1, away_team='ARI', home_team='LAC', market='spread',
                             market_base=-3., baseline=0., prediction=0., edge=1., variance=4.,
                             odds=-110., assumed_odds=True, actual=float('nan'), qualifies=False, pnl=0.,
                             positive_odds=-110., negative_odds=-110., residual=float('nan')))
        data = pd.DataFrame([row])
        importance = pd.DataFrame(dict(feature=['away_off_run_ypp'], importance=[1.], std=[.1]))
        config = dict(model='test', calculation='two-sided-team-points-v1', market='spread',
                      lookback=20, status='PASS', reason='test')
        with tempfile.TemporaryDirectory() as directory, patch.object(wp, 'display_stats', return_value=pd.DataFrame()):
            wp.write_packets(data, data, importance, config, Path(directory))
            html = (Path(directory) / '2026_01' / 'index.html').read_text()
            self.assertIn('class="copy-btn"', html)
            self.assertIn('id="picks-copy-source"', html)
            self.assertIn('class="copy-title">Model<', html)
            self.assertIn('class="copy-subtitle">2026 Week 1<', html)
            self.assertIn('class="headline-table-light"', html)  # the copy payload, not the on-page dark table


if __name__ == '__main__':
    unittest.main()
