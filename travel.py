"""Travel distance: how far each team came to play.

Model 2.2's context input, built the same way as rest days -- one number per
team per game, differenced between the two teams inside the model. A team's
travel is the great-circle distance from the venue it calls home that season
to the venue the game is played in, so the home team is normally 0, the away
team carries the trip, and at a neutral site (London, Mexico City, Melbourne)
both teams travel and only the difference between them is an edge.

Distances come from the stadium coordinates pull_weather.py already keeps in
data/weather/stadium_coordinates.parquet, joined to the schedule's stadium_id.
Everything is keyed on game_id, so this never has to know about team
abbreviations changing (SD/STL/OAK) between the schedule and the panel.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SCHEDULE = Path('data/sched.parquet')
COORDINATES = Path('data/weather/stadium_coordinates.parquet')  # written by pull_weather.py
EARTH_RADIUS_MILES = 3958.7613
COLUMNS = ['away_travel_miles', 'home_travel_miles', 'away_travel_adv']


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle miles between two points -- the flight, not the drive."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    inner = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(inner, 0., 1.)))


def home_venues(schedule):
    """Series of stadium_id indexed by (season, team): the venue a team hosted
    the most home games in that season.

    The most, rather than every venue it hosted in, because of displaced
    seasons -- New Orleans 2005 (Katrina: the Alamodome, LSU and Giants
    Stadium), Buffalo's yearly Toronto game, Minnesota 2010 after the
    Metrodome roof collapsed. Travel is measured from where the team actually
    based its season, and a rare home game somewhere else then shows up as
    travel for the home team too, which is what it was.

    Neutral-site games are excluded outright: a designated "home" team in
    London is not hosting anything. A team with no home game on the books
    yet (a partly released future schedule) inherits its last known venue.
    """
    hosted = schedule.loc[schedule.location.astype(str).str.lower().eq('home'),
                          ['season', 'home_team', 'stadium_id']].dropna()
    hosted = hosted.rename(columns={'home_team': 'team'})
    # Ties (a season split evenly between two venues) break on stadium_id, so
    # the choice is at least stable from run to run.
    venues = (hosted.groupby(['season', 'team', 'stadium_id']).size().rename('games').reset_index()
                    .sort_values(['games', 'stadium_id']).drop_duplicates(['season', 'team'], keep='last'))
    teams = pd.unique(pd.concat([schedule.away_team, schedule.home_team]).dropna())
    seasons = range(int(schedule.season.min()), int(schedule.season.max()) + 1)
    grid = pd.MultiIndex.from_product([seasons, sorted(teams)], names=['season', 'team'])
    known = venues.set_index(['season', 'team']).stadium_id.reindex(grid)
    return known.groupby(level='team').ffill()


def _frame(value, default):
    """A DataFrame, a path to one, or None for the default path."""
    if value is None:
        return pd.read_parquet(default)
    return value if isinstance(value, pd.DataFrame) else pd.read_parquet(value)


def game_travel(schedule=None, coordinates=None):
    """One row per game: each team's travel in miles, and the away team's
    travel advantage (away minus home -- positive means the away team came
    further, which is the away team's disadvantage)."""
    schedule = _frame(schedule, SCHEDULE)
    venues = _frame(coordinates, COORDINATES).set_index('stadium_id')
    homes = home_venues(schedule)

    def coords(stadium_ids):
        located = venues.reindex(pd.Index(stadium_ids))
        return located.latitude.to_numpy(), located.longitude.to_numpy()

    games = schedule[['game_id', 'season', 'week', 'away_team', 'home_team', 'stadium_id']].copy()
    venue_lat, venue_lon = coords(games.stadium_id)
    for side in ['away', 'home']:
        base = pd.MultiIndex.from_arrays([games.season, games[f'{side}_team']]).map(homes)
        lat, lon = coords(base)
        games[f'{side}_travel_miles'] = np.round(haversine_miles(lat, lon, venue_lat, venue_lon), 1)
    games['away_travel_adv'] = games.away_travel_miles - games.home_travel_miles
    return games[['game_id'] + COLUMNS]


def attach_travel(panel, schedule=None, coordinates=None):
    """Add the travel columns to a prepared panel, by game_id. Raises rather
    than filling a gap silently -- a missing distance is an unknown stadium or
    an unknown home venue, and both are worth hearing about."""
    result = panel.merge(game_travel(schedule, coordinates), on='game_id', how='left', validate='one_to_one')
    missing = result[COLUMNS].isna().any(axis=1)
    if missing.any():
        bad = result.loc[missing, 'game_id']
        raise ValueError(f'Travel distance missing for {len(bad)} games, including {bad.iloc[0]}')
    return result
