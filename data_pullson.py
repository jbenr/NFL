import pandas as pd
import os
import nfl_data_py as nfl
import requests
import json
from bs4 import BeautifulSoup
from tabulate import tabulate

def pull_sched(szns):
    if not os.path.exists('data'): os.makedirs('data')
    sched = nfl.import_schedules(szns)
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
    df = pd.read_html(str(indiatable))
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