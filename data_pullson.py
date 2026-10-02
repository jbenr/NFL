import pandas as pd
import os
import nfl_data_py as nfl
import requests
import json
from io import StringIO
from bs4 import BeautifulSoup
from tabulate import tabulate
import utils

def pull_sched(szns):
    if not os.path.exists('data'): os.makedirs('data')
    sched = nfl.import_schedules(szns)
    if os.path.exists('data/sched.parquet'):
        previous = pd.read_parquet('data/sched.parquet')
        sched = pd.concat([previous[~previous.season.isin(sched.season.unique())], sched], ignore_index=True)
    sched.to_parquet('data/sched.parquet')


def pull_pbp(szns):
    if not os.path.exists('data/pbp'): os.makedirs('data/pbp')
    for szn in szns:
        try:
            dat = nfl.import_pbp_data([szn], cache=False, alt_path=None)
            dat.to_parquet(f'data/pbp/pbp_{szn}.parquet')
        except Exception as e:
            print(e)
            try:
                url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{szn}.parquet"
                file_path = f"data/pbp/pbp_{szn}.parquet"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(file_path, 'wb') as file:
                        file.write(response.content)
                else:
                    print(f"Failed to download file. Status code: {response.status_code}")
            except Exception as e:
                print(e)

    # df = pd.read_parquet(f'data/pbp/pbp_{szns.max()}.parquet')
    # print(f'Latest data from {szns.max()}: week {df[df.season==szns.max()].week.max()}\n'
    #       f'{df[(df.season==szns.max())&(df.week==df.week.max())].groupby(["away_team","home_team"]).agg("count").index.tolist()}')


def pull_ngs(szns):
    if not os.path.exists('data'): os.makedirs('data')
    df = nfl.import_ngs_data('passing', szns)
    df = df.replace({'LAR':'LA'})
    df.to_parquet('data/ngs_passing.parquet')

def get_abbr():
    # get the response in the form of html
    wikiurl = "https://en.wikipedia.org/w/index.php?title=Wikipedia:WikiProject_National_Football_League/National_Football_League_team_abbreviations&oldid=1200558873"
    table_class = "wikitable sortable jquery-tablesorter"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(wikiurl, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    indiatable = soup.find('table',{'class':"wikitable"})
    df = pd.read_html(StringIO(str(indiatable)))
    df = pd.DataFrame(df[0])
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True).rename(columns={'Franchise':'team_name','Commonly Used Abbreviations':'abbr'})[['team_name','abbr']]
    df.abbr = df.abbr.replace(['JAC','LAR'],['JAX','LA'])
    return dict(zip(list(df.team_name),list(df.abbr)))

def pull_odds():
    # An api key is emailed to you when you sign up to a plan
    # Get a free API key at https://api.the-odds-api.com/
    API_KEY = '7f0d888986edaf32491f95580e31a0dd'
    SPORT = 'americanfootball_nfl' # use the sport_key from the /sports endpoint below, or use 'upcoming' to see the next 8 games across all sports
    REGIONS = 'us' # uk | us | eu | au. Multiple can be specified if comma delimited
    MARKETS = 'spreads,totals' # h2h | spreads | totals. Multiple can be specified if comma delimited
    ODDS_FORMAT = 'decimal' # decimal | american
    DATE_FORMAT = 'iso' # iso | unix

    odds_response = requests.get(
        f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds',
        params={
            'api_key': API_KEY,
            'regions': REGIONS,
            'markets': MARKETS,
            'oddsFormat': ODDS_FORMAT,
            'dateFormat': DATE_FORMAT,
        }
    )

    if odds_response.status_code != 200:
        print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

    else:
        odds_json = odds_response.json()
        with open('data/book.json', 'w') as f:
            json.dump(odds_json, f)
        print('Number of events pulled from odds-api:', len(odds_json))

        df = pd.json_normalize(odds_json, ['bookmakers', 'markets', 'outcomes'],
                               ['commence_time', 'away_team', 'home_team', ['bookmakers', 'last_update']]).drop(
            columns='price')
        # print(tabulate(df,headers='keys'))
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        df.commence_time = pd.to_datetime(df.commence_time).dt.date
        # print(tabulate(df,headers='keys'))
        df['bookmakers.last_update'] = pd.to_datetime(df['bookmakers.last_update'])

        total = df.loc[df.name.isin(['Over'])].drop_duplicates()
        total = total.groupby(['commence_time', 'away_team', 'home_team']).agg(
            {'bookmakers.last_update': 'max', 'point': 'median'}) \
            .reset_index().rename(columns={'point': 'total'})

        spread = df.loc[df.name == df.away_team]
        spread = spread.groupby(['commence_time', 'away_team', 'home_team']).agg(
            {'bookmakers.last_update': 'max', 'point': 'median'}) \
            .reset_index().rename(columns={'point': 'spread'})

        df = total.merge(spread, on=['commence_time', 'away_team', 'home_team'])
        df = df[['commence_time', 'away_team', 'spread', 'home_team', 'total']].rename(columns={'commence_time':'date'})

        abbr = get_abbr()
        df = df.replace({'away_team': abbr, 'home_team': abbr})

        # df['date'] = pd.to_datetime(df['date']).dt.date
        # print(df.columns)
        # df.to_parquet('data/book.parquet')

        # Check the usage quota
        print('odds-api Remaining requests', odds_response.headers['x-requests-remaining'])
        print('odds-api Used requests', odds_response.headers['x-requests-used'])

        return df


# Report statuses that mean a quarterback will not start. 'Questionable'
# is deliberately absent: most questionable quarterbacks play.
OUT_STATUSES = {'Out', 'Doubtful'}
# Missing practice entirely is the strongest live signal before the official
# designation is published. It is a proxy, not a ruling -- a veteran can be
# rested on a Wednesday -- but on the Thursday of a game week it is the only
# thing the feed carries, and it is right far more often than it is wrong.
OUT_PRACTICE = {'Did Not Participate In Practice'}


def quarterback_status(seasons):
    """(team, week) -> ordered list of (name, why_not) for that team's
    quarterbacks, best first.

    `why_not` is None for an available quarterback and a short reason for
    one who will not start, so a caller can walk down the depth chart and
    say which names it skipped.

    Two sources, because neither is sufficient alone. The depth chart says
    who the team designates, which is stale the moment somebody is ruled
    out. The injury report says who is unavailable, which is published
    through the week -- the official report_status lands on the Friday, so
    a Thursday build has only practice participation to go on."""
    import nfl_data_py as nfl
    seasons = list(seasons)
    charts = nfl.import_depth_charts(seasons)
    charts['dt'] = pd.to_datetime(charts.dt, errors='coerce')
    ones = charts[(charts.pos_abb == 'QB')].dropna(subset=['dt'])
    if ones.empty:
        return {}
    # The newest chart per team, then that chart's own ranking.
    newest_dt = ones.groupby('team').dt.transform('max')
    current = ones[ones.dt == newest_dt].sort_values(['team', 'pos_rank'])

    try:
        injuries = nfl.import_injuries(seasons)
    except Exception:
        injuries = pd.DataFrame(columns=['team', 'week', 'full_name', 'report_status', 'practice_status'])

    def reason(team, week, name):
        rows = injuries[(injuries.team == team) & (injuries.week == week)
                        & (injuries.full_name == name)]
        if rows.empty:
            return None
        row = rows.iloc[-1]
        if row.get('report_status') in OUT_STATUSES:
            return str(row['report_status']).lower()
        if pd.isna(row.get('report_status')) and row.get('practice_status') in OUT_PRACTICE:
            return 'did not practise'
        return None

    weeks = sorted(injuries.week.dropna().unique().tolist()) or [None]
    out = {}
    for team, group in current.groupby('team'):
        names = group.player_name.tolist()
        for week in weeks:
            out[(team, int(week))] = [(n, reason(team, int(week), n)) for n in names]
    return out


def refresh_starters(seasons, schedule_path='data/sched.parquet'):
    """Update the schedule's designated starter for unplayed games, using
    the depth chart filtered by the injury report.

    The depth chart alone is NOT enough, and getting this wrong is worse
    than leaving it alone. nflverse's own away_qb_name/home_qb_name for an
    upcoming game already accounts for injuries -- in 2026 week 4 it had
    Chicago starting Case Keenum and Tampa starting Jalon Daniels because
    Caleb Williams and Baker Mayfield were out. The raw depth chart still
    listed both as QB1, so overwriting the schedule with it replaced two
    correct names with two wrong ones.

    What the schedule does get wrong is the other direction: a starter
    returning from injury. It carries the last man to actually start, so
    Seattle showed Drew Lock for a week 4 that Sam Darnold was back for.

    So: walk the depth chart in order and take the first quarterback the
    injury report does not rule out, and only overwrite when that
    disagrees with the schedule. A finished game is never touched -- it
    records who actually played."""
    sched = pd.read_parquet(schedule_path)
    try:
        rooms = quarterback_status(seasons)
    except Exception as error:
        print(f'  could not read depth charts/injuries ({error}); leaving starters alone', flush=True)
        return sched
    if not rooms:
        return sched
    unplayed = sched.away_score.isna()
    changes = []
    for side in ('away', 'home'):
        column = f'{side}_qb_name'
        if column not in sched.columns:
            continue
        for row in sched.index[unplayed]:
            team, week = sched.at[row, f'{side}_team'], int(sched.at[row, 'week'])
            room = rooms.get((team, week))
            if not room:
                continue
            available = next((n for n, why in room if not why), None)
            ruled_out = [f'{n} ({why})' for n, why in room if why]
            if available and available != sched.at[row, column]:
                changes.append((sched.at[row, 'game_id'], team, sched.at[row, column],
                                available, '; '.join(ruled_out)))
                sched.at[row, column] = available
    if changes:
        print(f'  {len(changes)} starter(s) updated from depth chart + injury report:', flush=True)
        for game, team, was, now, out in changes[:12]:
            tail = f'   [out: {out}]' if out else ''
            print(f'    {game} {team}: {was} -> {now}{tail}', flush=True)
        utils.save_parquet(sched, schedule_path)
    else:
        print('  starters agree with the depth chart and injury report', flush=True)
    return sched
