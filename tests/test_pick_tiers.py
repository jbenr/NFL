import unittest
import re

import numpy as np
import pandas as pd

import weekly_packet as wp


def settled(**overrides):
    """One settled row, as apply_tiers receives it."""
    row = dict(season=2025, week=3, market_base=-3.5, prediction=-8., edge=-4.5, variance=16.,
               total_game_importance=.5, qualifies=True)
    row.update(overrides)
    return pd.DataFrame([row])


class TierTests(unittest.TestCase):
    def test_no_tier_outside_the_validated_windows(self):
        self.assertIsNone(wp.pick_tier('spread', week=3, importance=.9))    # leverage means nothing early
        self.assertIsNone(wp.pick_tier('spread', week=12, importance=.9))
        self.assertIsNone(wp.pick_tier('spread', week=14, importance=.4))   # late, but nothing riding on it
        self.assertIsNone(wp.pick_tier('spread', week=14, importance=None))
        self.assertIsNone(wp.pick_tier('total', week=2, line=48.))
        self.assertIsNone(wp.pick_tier('total', week=10, line=44.))   # the 42-46 dead band

    def test_the_windows_that_do_earn_a_tier(self):
        self.assertEqual(wp.pick_tier('spread', week=13, importance=.75), 'S')
        self.assertEqual(wp.pick_tier('spread', week=17, importance=.9), 'S')   # covers weeks 15-18
        self.assertEqual(wp.pick_tier('spread', week=20, importance=1.), 'S')
        self.assertEqual(wp.pick_tier('total', week=13, line=50.), 'S')   # 64.6%
        self.assertEqual(wp.pick_tier('total', week=8, line=50.), 'S')    # 60.0%
        self.assertEqual(wp.pick_tier('total', week=10, sd=3.5, line=50.), 'S')   # 68.3%, ensemble agrees
        self.assertEqual(wp.pick_tier('total', week=10, sd=5., line=50.), 'B')    # 55.2%, it does not
        self.assertEqual(wp.pick_tier('total', week=16, sd=5., line=50.), 'B')

    def test_a_tierless_game_is_not_a_pick_however_big_the_edge(self):
        """A catch-all tier used to make every qualifying disagreement a
        graded pick; those measured 49.1%, so they are no bet at all now."""
        early = wp.apply_tiers(settled(week=3, edge=-9., total_game_importance=.95), 'spread')
        self.assertIsNone(early.tier.iloc[0])
        self.assertFalse(bool(early.qualifies.iloc[0]))
        late = wp.apply_tiers(settled(week=13, edge=-9., total_game_importance=.95), 'spread')
        self.assertEqual(late.tier.iloc[0], 'S')
        self.assertTrue(bool(late.qualifies.iloc[0]))

    def test_a_late_game_nobody_needs_to_win_is_not_a_pick(self):
        dead = wp.apply_tiers(settled(week=17, edge=-9., total_game_importance=.2), 'spread')
        self.assertIsNone(dead.tier.iloc[0])
        self.assertFalse(bool(dead.qualifies.iloc[0]))

    def test_a_tier_cannot_rescue_a_pick_that_failed_the_edge_test(self):
        losing = wp.apply_tiers(settled(week=13, qualifies=False, total_game_importance=.95), 'spread')
        self.assertFalse(bool(losing.qualifies.iloc[0]))

    def test_every_bucket_is_measured_documented_and_bettable(self):
        for market, spec in wp.PICK_BUCKETS.items():
            for bucket in spec['buckets']:
                tier = wp.tier_band(bucket['rate'])
                self.assertIsNotNone(tier, f'{market}: {bucket["rule"]} is below the bottom band')
                self.assertIn(tier, wp.TIER_COLORS)
                self.assertGreaterEqual(bucket['rate'], .524)   # never ship a losing bucket
                self.assertGreaterEqual(bucket['n'], 100)       # nor one measured on a handful of games
                for field in ['rule', 'model', 'worth', 'eras', 'siblings', 'note']:
                    self.assertTrue(bucket[field])
            self.assertIn(market, wp.NO_PICK)          # the dash row is documented too

    def test_the_guide_warns_when_the_packet_ran_a_different_model(self):
        """Rates are model-specific -- the spread S bucket alone runs 60.0% on
        the source model and 49.4% on 2.1-importance -- so a packet built with
        something else must not present them as its own."""
        self.assertNotIn('tier-warn', wp.tier_guide(wp.TIER_MODEL))
        self.assertNotIn('tier-warn', wp.tier_guide(None))      # unknown: no claim either way
        warned = wp.tier_guide('Model 2.1 · importance')
        self.assertIn('tier-warn', warned)
        self.assertIn('Model 2.1 · importance', warned)

    def test_every_bucket_names_the_run_it_was_measured_on(self):
        for market, spec in wp.PICK_BUCKETS.items():
            for bucket in spec['buckets']:
                self.assertIn(bucket['model'], wp.tier_guide())
            # One list, one model: the no-pick row has to describe the same
            # run its buckets were measured on, or the guide contradicts itself.
            owners = {bucket['model'] for bucket in spec['buckets']}
            self.assertEqual(len(owners), 1, f'{market} mixes models: {owners}')
            self.assertEqual(wp.NO_PICK[market]['model'], owners.pop())

    def test_a_bucket_never_fires_on_another_model(self):
        """Rates are model-specific: the late-season leverage bucket is 60.7%
        on the model it was measured on, and describes nothing on a different
        architecture. A model with no buckets of its own gets no picks."""
        late = dict(market='spread', week=14, importance=.8, sd=4., line=-3., edge=4.)
        self.assertEqual(wp.pick_tier(**late), 'S')
        self.assertEqual(wp.pick_tier(**late, running=wp.TIER_MODEL), 'S')
        self.assertIsNone(wp.pick_tier(**late, running=wp.SHARED_MODEL))
        for week in [3, 8, 14]:
            self.assertIsNone(wp.pick_tier('spread', week, importance=.8, sd=1.5, line=-3., edge=4.,
                                           running=wp.SHARED_MODEL))

    def test_the_letter_follows_the_measured_rate(self):
        self.assertEqual(wp.tier_band(.64), 'S')
        self.assertEqual(wp.tier_band(.60), 'S')
        self.assertEqual(wp.tier_band(.58), 'A')
        self.assertEqual(wp.tier_band(.549), 'B')
        self.assertIsNone(wp.tier_band(.539))
        self.assertIsNone(wp.tier_band(.50))


class StatFormatTests(unittest.TestCase):
    def test_per_play_points_are_not_shown_as_percentages(self):
        """EPA per play is points (0.153), not a share of plays. The '_pp'
        suffix it shares with first downs and turnovers per play -- which
        really are rates -- once turned it into '15.3%'."""
        import re
        stats = pd.DataFrame({'team': ['SF'], 'off_pass_epa_pp': [.153], 'off_run_epa_pp': [-.020],
                              'off_first_down_pp': [.333], 'off_turnovers_pp': [.016],
                              'off_pass_ypp': [6.6]})
        cell = lambda metric: re.sub(r'<[^>]+>', '', wp.stat_cell(stats, 'SF', 'off', metric))
        self.assertEqual(cell('pass_epa_pp'), '+0.153(#1)')
        self.assertEqual(cell('run_epa_pp'), '-0.020(#1)')
        self.assertEqual(cell('first_down_pp'), '33.3%(#1)')   # a real per-play rate
        self.assertEqual(cell('turnovers_pp'), '1.6%(#1)')
        self.assertEqual(cell('pass_ypp'), '6.6(#1)')

    def test_every_points_metric_is_declared(self):
        for metric in wp.POINTS_PER_PLAY:
            self.assertTrue(metric.endswith('_pp'), metric)
            self.assertNotIn('%', metric)


class ProfileTests(unittest.TestCase):
    """Lookback, training window and stat preset belong to the model, not to
    the command line -- a bucket's hit rate was measured with specific ones."""

    def test_every_bucket_names_a_model_the_packet_knows_how_to_run(self):
        import model_spec
        for market, spec in wp.PICK_BUCKETS.items():
            for bucket in spec['buckets']:
                self.assertIn(bucket['model'], model_spec.PROFILES,
                              f'{market}: no profile for {bucket["model"]}, so the packet cannot reproduce it')

    def test_a_profile_carries_the_whole_recipe(self):
        import model_spec
        for name in model_spec.PROFILES:
            settings = model_spec.profile(name)
            for field in ['version', 'architecture', 'lookback', 'train_window', 'calculation']:
                self.assertIsNotNone(settings[field], f'{name} is missing {field}')

    def test_a_week_is_only_fit_by_profiles_that_own_something(self):
        """No profile is run for nothing: the shared architecture owned weeks
        1-12 spreads until its finished backtest came in at 53.3%, and the
        packet stopped fitting it the moment its bucket went.

        Stated as the invariant rather than as today's answer -- which
        profile owns which week is expected to change as buckets are found
        and lost, but fitting a model no market will use never is."""
        import model_spec
        for week in [1, 3, 5, 6, 12, 13, 22]:
            picks, order = model_spec.lineup(week)
            self.assertEqual(sorted(order), sorted(set(picks.values())),
                             f'week {week} fits {set(order) - set(picks.values())} for nothing')
            self.assertEqual(order[0], model_spec.DEFAULT_PROFILE,
                             'the default profile owns the shared panel and must be fit first')

    def test_the_solved_preset_owns_exactly_the_weeks_it_was_measured_on(self):
        """Early-season spreads go to the fixed-taper model, and only for the
        weeks its backtest actually covers -- weeks 6-12 were still running
        when the bucket went in, so the window stops at 5."""
        import model_spec
        for week in [1, 3, 5]:
            self.assertEqual(model_spec.owner('spread', week), model_spec.SOLVED_PROFILE)
            self.assertEqual(model_spec.owner('total', week), model_spec.DEFAULT_PROFILE)
        for week in [6, 12, 13, 22]:
            self.assertEqual(model_spec.owner('spread', week), model_spec.DEFAULT_PROFILE)

    def test_every_owning_profile_has_buckets_to_justify_it(self):
        import model_spec
        owners = {model_spec.owner(market, week) for week in range(1, 23) for market in ['spread', 'total']}
        with_buckets = {bucket['model'] for spec in wp.PICK_BUCKETS.values() for bucket in spec['buckets']}
        self.assertTrue(owners <= with_buckets,
                        f'{owners - with_buckets} would be fit every week without a bucket to show for it')


class PlumbingTests(unittest.TestCase):
    """The packet runs two models now, so the tier table has to follow the
    predictions, not the folder."""

    def test_write_packets_tiers_against_the_model_that_made_the_predictions(self):
        """Regression: the cards used to read {market}_config.json for the
        model, but that file is written AFTER the pages render -- so a
        shared-model packet silently graded its picks against the two-sided
        table and threw every pick away."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        seen = []
        real = wp.apply_tiers

        def spy(frame, market, running=None):
            seen.append(running)
            return real(frame, market, running)

        data = pd.DataFrame(dict(
            season=[2026], week=[3], away_team=['ATL'], home_team=['GB'], market='spread',
            market_base=[-3.], prediction=[-7.5], edge=[-4.5], variance=[4.], actual=[np.nan],
            residual=[np.nan], away_points=[20.], home_points=[27.5], baseline=[0.],
            integration_residual=[0.], qualifies=[True], win=[False], push=[False], pnl=[np.nan],
            odds=[-110.], assumed_odds=[False], positive_odds=[-110.], negative_odds=[-110.],
            attr_away_off_run_ypp=[1.], total_game_importance=[.5], sd=[2.]))
        config = dict(model=f'{wp.SHARED_MODEL} · 100 members', tier_model=wp.SHARED_MODEL,
                      calculation='model-2.1', feature_calculation='weighted', lookback=20,
                      market='spread', status='PASS', reason='test')
        importance = pd.DataFrame(dict(feature=['own_off_run_ypp'], importance=[1.], std=[.1]))
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(wp, 'display_stats', return_value=pd.DataFrame()), \
                patch.object(wp, 'apply_tiers', side_effect=spy):
            wp.write_packets(data, data, importance, config, Path(folder))
        self.assertIn(wp.SHARED_MODEL, seen)
        self.assertNotIn(wp.TIER_MODEL, seen)


if __name__ == '__main__':
    unittest.main()


class HeadlineSheetLayout(unittest.TestCase):
    """The sheet's column order lives in one list; three places used to
    hardcode it separately and had already drifted apart."""

    def right_aligned(self, css_class):
        """Which columns end up right-aligned once the whole stylesheet has
        had its say.

        Resolved across every rule, not just the first one that matches: an
        older right-align list once sat *below* the current one and quietly
        won, right-aligning the team and quarterback columns that the
        picture centres. A test that stopped at the first match passed
        through all of it."""
        aligned = set()
        for position in range(1, len(wp.HEADLINE_COLUMNS) + 1):
            if self.declarations_for(css_class, position).get('text-align') == 'right':
                aligned.add(position)
        return sorted(aligned)

    def test_css_right_alignment_matches_the_column_list(self):
        expected = sorted(wp.headline_indices(wp.HEADLINE_RIGHT))
        for css_class in ('headline-table-light', 'headline-table'):
            self.assertEqual(self.right_aligned(css_class), expected,
                             f'.{css_class} right-aligns different columns than HEADLINE_RIGHT')

    def test_labels_line_up_with_columns(self):
        self.assertEqual(len(wp.HEADLINE_LABELS), len(wp.HEADLINE_COLUMNS))

    def test_every_named_group_is_a_real_column(self):
        known = set(wp.HEADLINE_COLUMNS)
        for group in (wp.HEADLINE_RIGHT, wp.HEADLINE_SMALL, wp.HEADLINE_TIGHT):
            self.assertLessEqual(group, known, f'{group - known} is not a column')

    def test_the_split_fields_are_labelled(self):
        """Kickoff is three columns and each QB two. The date needs no
        heading of its own -- it reads under 'Kickoff' -- but the rating
        does, or a bare number beside a name is anyone's guess."""
        label = dict(zip(wp.HEADLINE_COLUMNS, wp.HEADLINE_LABELS))
        self.assertEqual(label['kick_day'], 'Kickoff')
        self.assertEqual(label['kick_time'], 'ET', 'the sheet has to say which timezone it means')
        self.assertEqual(label['away_elo'], 'Elo')
        self.assertEqual(label['home_elo'], 'Elo', 'both sides, not just one')
        for blank in ('kick_date', 'away_logo', 'home_logo'):
            self.assertEqual(label[blank], '')

    def declarations_for(self, css_class, position):
        """Everything the stylesheet ends up saying about one column."""
        merged = {}
        for selectors, body in re.findall(r'([^{}]+)\{([^}]*)\}', wp.STYLE):
            positions = {int(n) for sel in selectors.split(',') if css_class in sel
                         for n in re.findall(r'nth-child\((\d+)\)', sel)}
            # A bare ".headline-table td{...}" applies to every column.
            blanket = any(css_class in sel and 'nth-child' not in sel and ' td' in sel
                          for sel in selectors.split(','))
            if position in positions or blanket:
                for declaration in body.split(';'):
                    if ':' in declaration:
                        key, value = declaration.split(':', 1)
                        merged[key.strip()] = value.strip()
        return merged

    def test_both_rating_columns_are_styled_alike(self):
        """The two ratings are the same field on opposite sides of the
        sheet; a rule that catches one and not the other is exactly how one
        column ends up in the wrong typeface."""
        away, home = wp.headline_indices({'away_elo'})[0], wp.headline_indices({'home_elo'})[0]
        for css_class in ('headline-table-light', 'headline-table'):
            left = self.declarations_for(css_class, away)
            right = self.declarations_for(css_class, home)
            self.assertEqual(left, right, f'.{css_class} styles the two ratings differently')
            self.assertIn('Graduate', left.get('font-family', ''),
                          f'.{css_class} ratings are not pinned to the NFL font')


class SheetMatchesThePicture(unittest.TestCase):
    """The page and the emailed picture are the same sheet drawn twice --
    once by the browser from CSS, once by Pillow in picks_png. Anything
    keyed to a column position has to say the same thing in both."""

    def test_no_rule_points_past_the_end_of_the_sheet(self):
        """A leftover rule from an older column order shows up here first:
        it either overshoots the table or lands on a column that has since
        become something else."""
        last = len(wp.HEADLINE_COLUMNS)
        for selectors, _ in re.findall(r'([^{}]+)\{([^}]*)\}', wp.STYLE):
            for selector in selectors.split(','):
                if 'headline-table' not in selector:
                    continue
                for position in re.findall(r'nth-child\((\d+)\)', selector):
                    self.assertLessEqual(int(position), last,
                                         f'{selector.strip()} points past the last column ({last})')

    def test_neither_theme_leans_on_the_global_table_defaults(self):
        """The stylesheet has a global `td,th{text-align:right}` for every
        other table on the page. Both sheets must override it explicitly:
        the dark one did not, so the page right-aligned the team codes and
        quarterbacks while the picture centred them."""
        for css_class in ('headline-table', 'headline-table-light'):
            base = {}
            for selectors, body in re.findall(r'([^{}]+)\{([^}]*)\}', wp.STYLE):
                for selector in selectors.split(','):
                    trimmed = selector.strip().removeprefix('.packet ')
                    if trimmed in (f'.{css_class} td', f'.{css_class} th',
                                   f'.{css_class} th,.{css_class} td'):
                        for declaration in body.split(';'):
                            if ':' in declaration:
                                key, value = declaration.split(':', 1)
                                base[key.strip()] = value.strip()
            self.assertEqual(base.get('text-align'), 'center',
                             f'.{css_class} inherits its alignment from the global td rule')
            self.assertEqual(base.get('line-height'), '1.15', f'.{css_class} row height differs')

    def test_the_two_themes_align_their_columns_identically(self):
        """Dark page and light picture differ in colour, never in which
        column is a figure and which is a label."""
        for position in range(1, len(wp.HEADLINE_COLUMNS) + 1):
            dark = HeadlineSheetLayout().declarations_for('headline-table', position)
            light = HeadlineSheetLayout().declarations_for('headline-table-light', position)
            self.assertEqual(dark.get('text-align'), light.get('text-align'),
                             f'column {position} ({wp.HEADLINE_COLUMNS[position - 1]}) '
                             'is aligned differently on the page than in the picture')
            self.assertEqual(dark.get('font-size'), light.get('font-size'),
                             f'column {position} ({wp.HEADLINE_COLUMNS[position - 1]}) '
                             'is a different size on the page than in the picture')


class DiffShadingStaysOutOfTheTiersWay(unittest.TestCase):
    """The Diff column is shaded by how big the disagreement is; the Picks
    column is painted by tier. They sit two columns apart, so the two
    scales must not be confusable -- a deep green Diff cell beside a green
    A pick read as though the row had been picked when it had not."""

    @staticmethod
    def hue(colour):
        """Degrees around the colour wheel. Channel comparisons are too
        blunt here: #e3c4ff is a purple whose largest channel is blue."""
        import colorsys
        r, g, b = (int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return colorsys.rgb_to_hsv(r, g, b)[0] * 360

    def gradient_stops(self):
        source = open('weekly_packet.py', encoding='utf-8').read()
        found = re.search(r"headline_blues_light', \[([^\]]*)\]", source)
        self.assertIsNotNone(found, 'the Diff gradient is no longer the blue one')
        stops = re.findall(r'#[0-9a-fA-F]{6}', found.group(1))
        self.assertTrue(stops)
        return stops

    def test_the_diff_gradient_is_blue(self):
        for stop in self.gradient_stops():
            self.assertTrue(190 <= self.hue(stop) <= 250, f'{stop} is not a blue')

    def test_no_tier_colour_is_near_that_blue(self):
        """40 degrees is the margin: purple (S) sits ~56 away, and that is
        the closest of the three."""
        for tier, colour in wp.TIER_COLORS.items():
            for stop in self.gradient_stops():
                apart = abs(self.hue(colour) - self.hue(stop))
                apart = min(apart, 360 - apart)
                self.assertGreater(apart, 40, f'tier {tier} ({colour}) is too close to the Diff shading')


class GuideHeadings(unittest.TestCase):
    """A bucket set is keyed '{market}_{model}', which is plumbing. The
    reader should see the market."""

    def test_headings_name_the_market_not_the_key(self):
        guide = wp.tier_guide()
        for market in wp.PICK_BUCKETS:
            if '_' not in market:
                continue
            self.assertNotIn(market.title(), guide,
                             f'the raw key {market!r} is showing in a heading')
        for market in {m.split('_')[0] for m in wp.PICK_BUCKETS}:
            self.assertIn(f'{market.title()} picks', guide)


class SheetPageIsNoWiderThanTheSheet(unittest.TestCase):
    """The sheet sizes itself to its columns, so the page around it should
    not be several hundred pixels wider than the thing it holds."""

    @staticmethod
    def cap(selector):
        found = re.search(r'%s\{[^}]*max-width:(\d+)px' % re.escape(selector), wp.STYLE)
        return int(found.group(1)) if found else None

    def test_the_headline_tab_is_pulled_in_from_the_shared_shell(self):
        sheet, shell = self.cap('main.sheet-page'), self.cap('main.headline-shell')
        self.assertIsNotNone(sheet, 'the headline tab has no width of its own')
        self.assertLess(sheet, shell, 'the headline tab is no narrower than the wide stats shell')

    def test_overflow_still_scrolls_rather_than_clipping(self):
        """A narrower page is only safe because anything wider scrolls."""
        self.assertRegex(wp.STYLE, r'main\.headline-shell\{[^}]*overflow-x:auto')

    def test_the_headline_page_actually_asks_for_it(self):
        """The narrow width only reaches the sheet if index.html is built
        with the class; the stats tab must not pick it up."""
        source = open('weekly_packet.py', encoding='utf-8').read().splitlines()
        for number, line in enumerate(source):
            if "'index.html'" in line and 'write_text' in line:
                statement = ' '.join(source[number:number + 3])
                self.assertIn('sheet-page', statement,
                              'index.html is not built with the sheet-page class')
                break
        else:
            self.fail('could not find where index.html is written')
        for number, line in enumerate(source):
            if "'stats.html'" in line and 'write_text' in line:
                statement = ' '.join(source[number:number + 3])
                self.assertNotIn('sheet-page', statement,
                                 'the stats tab needs the wide shell, not the sheet width')
                break
