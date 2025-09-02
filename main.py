# Go Commies
import numpy as np
import pandas as pd
from datetime import datetime

import utils
from opt_einsum.blas import tensor_blas
from tabulate import tabulate, tabulate_formats
import os
import glob
import model_shredski
import data_crunchski
import data_crunchski_2
import data_pullson
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import requests
pd.set_option('display.max_columns', None)


def download_team_logos(teams, logo_dir='data/logos'):
    """
    Given a list/array of team abbreviations, download the NFL team logos
    from ESPN if they are not already present in logo_dir.
    """
    if not os.path.exists(logo_dir):
        os.makedirs(logo_dir)
    for team in teams:
        # Construct the local file path.
        logo_path = os.path.join(logo_dir, f'{team}.png')
        if not os.path.exists(logo_path):
            # Construct the ESPN URL for the team logo.
            url = f'https://a.espncdn.com/i/teamlogos/nfl/500/{team}.png'
            response = requests.get(url)
            if response.status_code == 200:
                with open(logo_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded logo for {team}")
            else:
                print(f"Failed to download logo for {team} from {url}")


def h_to_the_tml(pred, season, week, lookback):
    qb = pd.read_parquet(f'data/qb/qb_{season}_{week}_{lookback}.parquet')

    lines = data_pullson.pull_odds()
    sched = pd.read_parquet('data/sched.parquet')
    sched = sched[(sched.season == season) & (sched.week == week)]

    sched.loc[:, 'away_qb'] = sched['away_qb_name'].apply(lambda x: f"{x.split()[0][0]}.{x.split()[1]}")
    sched.loc[:, 'home_qb'] = sched['home_qb_name'].apply(lambda x: f"{x.split()[0][0]}.{x.split()[1]}")
    sched['away_qb'] = sched['away_qb'].apply(utils.strip_suffix)
    sched['home_qb'] = sched['home_qb'].apply(utils.strip_suffix)

    sched = pd.merge(sched, qb[['name', 'weighted_qb_elo']], left_on='away_qb', right_on='name', how='left').rename(
        columns={'weighted_qb_elo': 'away_qb_elo'}
    ).drop(columns='name')
    sched = pd.merge(sched, qb[['name', 'weighted_qb_elo']], left_on='home_qb', right_on='name', how='left').rename(
        columns={'weighted_qb_elo': 'home_qb_elo'}
    ).drop(columns='name')

    sched = sched[[
        'away_team', 'home_team', 'gameday', 'gametime', 'away_qb_elo', 'home_qb_elo', 'away_qb', 'home_qb'
    ]]
    pred.columns = pred.columns.get_level_values(0)

    result = pd.merge(lines, pred, on=['away_team', 'home_team'], how='left')
    result = pd.merge(result, sched, on=['away_team', 'home_team'], how='left')

    def rename_col_by_index(dataframe, index_mapping):
        dataframe.columns = [index_mapping.get(i, col) for i, col in enumerate(dataframe.columns)]
        return dataframe

    # Renaming columns using the function.
    new_column_mapping = {6: 'var'}
    result = rename_col_by_index(result, new_column_mapping)

    result['diff'] = result['prediction'] + result['spread']
    result['diff_abs'] = abs(result['prediction'] + result['spread'])

    def pick_func(x):
        if x['diff'] < 0:
            return x['home_team']
        elif x['diff'] > 0:
            return x['away_team']
        else:
            return None

    result['pick'] = result.apply(pick_func, axis=1)

    # Adding logos
    all_teams = pd.concat([result['away_team'], result['home_team']]).unique()
    download_team_logos(all_teams, logo_dir='data/logos')

    result['away_logo'] = result['away_team'].apply(lambda team: f'../logos/{team}.png')
    result['home_logo'] = result['home_team'].apply(lambda team: f'../logos/{team}.png')

    result = result[['gameday', 'gametime',
                     'away_qb', 'away_qb_elo', 'away_logo', 'away_team',
                     'spread', 'prediction',
                     'home_team', 'home_logo', 'home_qb', 'home_qb_elo',
                     'diff_abs', 'var', 'pick']]

    result['gameday'] = pd.to_datetime(result['gameday'])
    result['gametime'] = pd.to_datetime(result['gametime']).dt.time

    result = result.sort_values(by=['gameday', 'gametime', 'away_team'])
    result = result.dropna(subset=['prediction'])
    result['prediction'] = result['prediction'] * -1
    result = result.round(1).rename(columns={'diff_abs': 'diff'})

    if not os.path.exists('data/results'):
        os.makedirs('data/results')
    result.to_csv(f'data/results/results_{season}_{week}_{lookback}.csv')

    def style_fonts_and_borders(val):
        return 'font-size: 16px; font-family: Arial; border: 2px solid gray'

    def format_bold(x):
        return 'font-weight: bold'

    def set_precision(val, precision):
        return f'{val:.{precision}f}'

    def style_spread(val, precision):
        if val > 0:
            return f'+{val:.{precision}f}'
        else:
            return f'{val:.{precision}f}'

    mapper = {1: '#ffca1e', 2: '#ffe590', 3: '#fef2c9'}
    top_indexes = result.nlargest(3, 'diff').index

    def highlight_cells(x):
        if x == result.at[top_indexes[0], 'pick']:
            return f'background-color: {mapper[1]}'
        elif x == result.at[top_indexes[1], 'pick']:
            return f'background-color: {mapper[2]}'
        elif x == result.at[top_indexes[2], 'pick']:
            return f'background-color: {mapper[3]}'
        else:
            return ''

    picks = result.copy()
    picks['abs_pred'] = abs(picks.prediction)
    # picks = picks[picks['diff']>picks['var']]
    picks = picks[picks['abs_pred'] > 1]
    picks = picks[picks['var'] <= 0.5]
    picks = picks[picks['diff'] > 3]['pick'].to_list()

    ud = result.copy()
    ud['objection'] = ((ud.spread * ud.prediction) < 0).astype(int)
    ud = ud[ud.objection == 1]['pick'].to_list()

    def highlight_picks(x):
        return f'background-color: {mapper[2]}' if x in picks else ''

    def highlight_ud(x):
        return f'background-color: {mapper[1]}' if x in ud else ''

    result = result.reset_index(drop=True)

    # Create the HTML representation with styling.
    # Note the format functions for the logo columns:
    # we wrap the local file path in an <img> tag.
    html = (result.style
            .background_gradient(subset=['diff'], cmap='Greens')
            .background_gradient(subset=['var'], cmap='Reds')
            .applymap(style_fonts_and_borders)
            .format({
        'away_qb_elo': lambda x: set_precision(x, precision=1),
        'home_qb_elo': lambda x: set_precision(x, precision=1),
        'gameday': lambda x: x.strftime('%a %m/%d'),
        'gametime': lambda x: x.strftime("%I:%M %p").lstrip('0'),
        'spread': lambda x: style_spread(x, precision=1),
        'prediction': lambda x: style_spread(x, precision=1),
        'diff': lambda x: set_precision(x, precision=1),
        'var': lambda x: set_precision(x, precision=1),
        # Wrap the logo file path in an <img> tag.
        'away_logo': lambda x: f'<img src="{x}" alt="Away Logo" height="50">' if pd.notnull(x) else '',
        'home_logo': lambda x: f'<img src="{x}" alt="Home Logo" height="50">' if pd.notnull(x) else ''
    })
            .applymap(highlight_picks, subset=['away_team', 'home_team', 'pick'])
            .applymap(highlight_ud, subset=['away_team', 'home_team', 'pick'])
            )
    # IMPORTANT: disable escaping so that the <img> tags render as images.
    html.to_html(f'data/results/html_{season}_{week}_{lookback}.html', escape=False)

    print(tabulate(result, headers='keys'))

def pull_bt(lookback):
    sched = pd.read_parquet('data/sched.parquet')
    sched.dropna(subset='result',inplace=True)
    print(tabulate(sched.tail(5), headers='keys'))
    sched = sched[sched['game_type']=='REG'].groupby(['season','week']).agg('count').index.tolist()
    sched.reverse()
    for s, w in sched[:-lookback]:
        season = s
        week = w
        lookback = lookback

        try:
            # if os.path.exists(f'data/stats/dat_{season}_{week}_{lookback}.parquet'):
            #     df = pd.read_parquet(f'data/stats/dat_{season}_{week}_{lookback}.parquet')
            # else:
            df = data_crunchski_2.prep_test_train(season, week, lookback)
            df.to_parquet(f'data/stats/dat_{season}_{week}_{lookback}.parquet')

            if not os.path.exists(f'data/bt/{lookback}'): os.makedirs(f'data/bt/{lookback}')
            try:
                pred = model_shredski.modelo(df, season, week).round(1)
                pred.columns = pred.columns.get_level_values(0)
                pred.to_csv(f'data/bt/{lookback}/bt_{season}_{week}_{lookback}.csv')
                print(tabulate(pred,headers='keys'))
            except Exception as e:
                print(e)
        except Exception as e:
            print(f'Error running season: {season}, week: {week}, lookback: {lookback} - {e}')

def back_test(bt):
    lst = []
    for i in os.listdir(f'data/bt/{bt}'):
        name = i.split('_')
        try:
            if name[3][:2] == str(bt):
                temp = pd.read_csv(f'data/bt/{bt}/{i}')
                temp['week'] = int(name[2])
                temp['season'] = int(name[1])
                lst.append(temp)
            else: None
        except Exception as e: print(e)
    bt = pd.concat(lst)
    sched = pd.read_parquet('data/sched.parquet')
    bt = pd.merge(sched, bt, how='left', on=['week','season','away_team','home_team'])
    bt = bt.dropna(subset=['prediction'])
    bt = bt.dropna(subset=['result'])
    bt = bt[['season','week','away_team','away_score','home_team','home_score','result','spread_line','prediction','prediction.1']]
    bt['prediction'] = bt['prediction']*-1
    bt['abs_pred'] = abs(bt['prediction'])
    bt['diff'] = bt.spread_line - bt.prediction
    bt['diff_abs'] = abs(bt['diff'])

    def pick_func(x):
        if x['diff'] < 0: return x['home_team']
        elif x['diff'] > 0: return x['away_team']
        else: return None
    bt['pick'] = bt.apply(pick_func, axis=1)

    def winner_func(x):
        if x['result'] < x['spread_line']: return x['away_team']
        elif x['result'] > x['spread_line']: return x['home_team']
        else: return None
    bt['winner'] = bt.apply(winner_func, axis=1)
    bt['dinner'] = (bt['pick'] == bt['winner']).astype(int)
    bt['switcherooni'] = ((bt['prediction']==abs(bt['prediction'])) != (bt['spread_line']==abs(bt['spread_line']))).astype(int)
    bt['abs_spread'] = abs(bt['spread_line'])
    bt['prediction.1'] = bt['prediction.1']/bt['abs_spread']

    bt = bt.dropna(subset=['winner'])
    # bt_s = bt[bt.switcherooni==1]
    # print(bt_s['dinner'].sum()/len(bt_s))
    g = bt.groupby('pick').agg({'dinner':'sum','winner':'count'})
    g['%'] = g['dinner']/g['winner']
    # print(tabulate(g,headers='keys'))

    bins = np.linspace(0,10,101)
    var_bins = np.linspace(0,1,51)

    big = bt[bt.diff_abs<=100]
    # big = big[big.abs_pred < big.abs_spread]
    # big = big[big['diff_abs']>big['prediction.1']]
    # big = big[(big['abs_pred']>1)&(big['abs_pred']<=2)]
    # big = big[big['switcherooni']==1]
    big = big[big['prediction.1']<=0.1]
    # big = big[big.diff_abs>6]

    # big = big[big.diff_abs>=2]
    # big = big[big.season == 2024]
    print(tabulate(big,headers='keys'))
    print(len(big))

    piv = big.pivot_table(columns='dinner', index=pd.cut(bt['diff_abs'], bins), aggfunc='size')
    print(tabulate(piv,headers='keys',tablefmt=tabulate_formats[4]))
    piv = big.pivot_table(columns='dinner', index=pd.cut(bt['prediction.1'], var_bins), aggfunc='size')
    print(tabulate(piv,headers='keys',tablefmt=tabulate_formats[4]))
    piv = big.pivot_table(columns='dinner', index=pd.cut(bt['abs_pred'], bins), aggfunc='size')
    print(tabulate(piv,headers='keys',tablefmt=tabulate_formats[4]))

    # print(tabulate(big,headers='keys'))
    print(len(big[big.dinner==1])/len(big))

def run(season, week, lookback, bt=False):
    if bt and os.path.exists(f'data/stats/dat_{season}_{week}_{lookback}.parquet'):
            df = pd.read_parquet(f'data/stats/dat_{season}_{week}_{lookback}.parquet')
    else: df = data_crunchski_2.prep_test_train(season, week, lookback)

    if not os.path.exists('data/stats'): os.makedirs('data/stats')
    df.to_parquet(f'data/stats/dat_{season}_{week}_{lookback}.parquet')

    utils.pdf(df.tail(40))

    pred = model_shredski.modelo(df, season, week)
    return pred


if __name__ == '__main__':
    data_pullson.pull_sched(range(1999, 2026))
    data_pullson.pull_pbp([2025])
    # data_pullson.pull_ngs(range(1999, 2025))

    season = 2025
    week = 1
    lookback = 20

    sched = pd.read_parquet('data/sched.parquet')
    utils.pdf(sched[(sched['season']==season) & (sched['week']==week)])

    # pull_bt(20)
    # back_test(20)

    pred = run(season, week, lookback, bt=True).round(1)

    h_to_the_tml(pred, season, week, lookback)

