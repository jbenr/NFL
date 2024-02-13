# Go Commies
import numpy as np
import pandas as pd
from tabulate import tabulate, tabulate_formats
import os
import glob
import model_shredski
import data_crunchski
import data_crunchski_2
import data_pullson
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def h_to_the_tml(pred, season, week, lookback):
    lines = data_pullson.pull_odds()
    sched = pd.read_parquet('data/sched.parquet')
    sched = sched[(sched.season==season)&(sched.week==week)][['away_team','home_team','gameday','gametime']]
    pred.columns = pred.columns.get_level_values(0)

    result = pd.merge(lines, pred, on=['away_team','home_team'], how='left')
    result = pd.merge(result, sched, on=['away_team','home_team'], how='left')

    def rename_col_by_index(dataframe, index_mapping):
        dataframe.columns = [index_mapping.get(i, col) for i, col in enumerate(dataframe.columns)]
        return dataframe

    # Renaming columns using the function
    new_column_mapping = {6: 'var'}
    result = rename_col_by_index(result, new_column_mapping)

    result['diff'] = result['prediction']+result['spread']
    result['diff_abs'] = abs(result['prediction']+result['spread'])
    def pick_func(x):
        if x['diff'] < 0: return x['home_team']
        elif x['diff'] > 0: return x['away_team']
        else: return None
    result['pick'] = result.apply(pick_func, axis=1)

    result = result[['gameday','gametime','away_team','spread','prediction','home_team','diff_abs','var','pick']]
    result['gameday'] = pd.to_datetime(result['gameday'])
    result['gametime'] = pd.to_datetime(result['gametime']).dt.time

    result = result.sort_values(by=['gameday','gametime','away_team'])
    result = result.dropna(subset=['prediction'])
    result['prediction'] = result['prediction']*-1
    result = result.round(1).rename(columns={'diff_abs':'diff'})

    if not os.path.exists('data/results'): os.makedirs('data/results')
    result.to_csv(f'data/results/results_{season}_{week}_{lookback}.csv')

    def style_fonts_and_borders(val):
        return 'font-size: 16px; font-family: Arial; border: 2px solid gray'
    def format_bold(x):
        return f'font-weight: bold'
    def set_precision(val, precision):
        return f'{val:.{precision}f}'
    def style_spread(val, precision):
        if val > 0: return f'+{val:.{precision}f}'
        else: return f'{val:.{precision}f}'

    mapper = {1: '#ffca1e', 2: '#ffe590', 3: '#fef2c9'}
    top_indexes = result.nlargest(3, 'diff').index
    def highlight_cells(x):
        if x == result.at[top_indexes[0], 'pick']:
            return f'background-color: {mapper[1]}'
        elif x == result.at[top_indexes[1], 'pick']:
            return f'background-color: {mapper[2]}'
        elif x == result.at[top_indexes[2], 'pick']:
            return f'background-color: {mapper[3]}'
        else: return ''

    picks = result[(result['var']<=0.4)&(result['diff']>4)]['pick'].tolist()
    def highlight_picks(x):
        return f'background-color: {mapper[2]}' if x in picks else ''
    # result = result.drop(columns=['pick'])
    result = result.reset_index(drop=True)

    html = result.style \
        .background_gradient(subset=['diff'],cmap='Greens') \
        .background_gradient(subset=['var'],cmap='Reds') \
        .applymap(style_fonts_and_borders) \
        .format({
        'gameday': lambda x: x.strftime(f'%a %m/%d'),
        'gametime': lambda x: x.strftime("%I:%M %p").lstrip('0'),
        'spread': lambda x: style_spread(x, precision=1),
        'prediction': lambda x: style_spread(x, precision=1),
        'diff': lambda x: set_precision(x, precision=1),
        'var': lambda x: set_precision(x, precision=1)
    }).applymap(highlight_picks, subset=['away_team','home_team','pick'])

    html.to_html(f'data/results/html_{season}_{week}_{lookback}.html')

    print(tabulate(result,headers='keys'))

def pull_bt(lookback):
    sched = pd.read_parquet('data/sched.parquet')
    sched = sched[sched['game_type']=='REG'].groupby(['season','week']).agg('count').index.tolist()
    for s, w in sched[-lookback:]:
        season = s
        week = w
        lookback = lookback

        # print(tabulate(data_pullson.pull_odds(),headers='keys'))

        if os.path.exists(f'data/stats/dat_{season}_{week}_{lookback}.parquet'):
            df = pd.read_parquet(f'data/stats/dat_{season}_{week}_{lookback}.parquet')
        else:
            df = data_crunchski_2.prep_test_train(season, week, lookback)
            df.to_parquet(f'data/stats/dat_{season}_{week}_{lookback}.parquet')

        if not os.path.exists(f'data/bt/{lookback}'): os.makedirs(f'data/bt/{lookback}')
        pred = model_shredski.modelo(df, season, week).round(1)
        pred.columns = pred.columns.get_level_values(0)
        pred.to_csv(f'data/bt/{lookback}/bt_{season}_{week}_{lookback}.csv')
        print(tabulate(pred,headers='keys'))

def back_test(bt):
    lst = []
    for i in os.listdir(f'data/bt/{bt}'):
        name = i.split('_')
        if name[3][:2] == str(bt):
            temp = pd.read_csv(f'data/bt/{bt}/{i}')
            temp['week'] = int(name[2])
            temp['season'] = int(name[1])
            lst.append(temp)
        else: None
    bt = pd.concat(lst)
    sched = pd.read_parquet('data/sched.parquet')
    bt = pd.merge(sched, bt, how='left', on=['week','season','away_team','home_team'])
    bt = bt.dropna(subset=['prediction'])
    bt = bt.dropna(subset=['result'])
    bt = bt[['season','week','away_team','away_score','home_team','home_score','result','spread_line','prediction','prediction.1']]
    bt['prediction'] = bt['prediction']*-1
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

    bt = bt.dropna(subset=['winner'])
    bt_s = bt[bt.switcherooni==1]
    # print(tabulate(bt[bt.switcherooni==1],headers='keys'))

    bins = np.linspace(-20,20,21)
    # big = bt[bt.diff_abs>4]
    # big = big[big['prediction.1']<=0.4]
    # piv = bt[(bt['prediction.1']<0.6)&(bt['prediction.1']>0.2)].pivot_table(columns='dinner', index=pd.cut(bt['diff'], bins), aggfunc='size')
    # piv = bt.pivot_table(columns='dinner', index=pd.cut(bt['prediction.1'], bins), aggfunc='size')
    piv = bt.pivot_table(columns='dinner', index=pd.cut(bt['diff'], bins), aggfunc='size')

    # piv = bt.pivot_table(columns='dinner', index='switcherooni', aggfunc='size')
    print(piv)
    # print(tabulate(big,headers='keys'))
    # print(len(big[big.dinner==1])/len(big))

def run(season, week, lookback):
    if os.path.exists(f'data/stats/dat_{season}_{week}_{lookback}.parquet'):
        df = pd.read_parquet(f'data/stats/dat_{season}_{week}_{lookback}.parquet')
    else:
        df = data_crunchski_2.prep_test_train(season, week, lookback)
        df.to_parquet(f'data/stats/dat_{season}_{week}_{lookback}.parquet')

    pred = model_shredski.modelo(df, season, week)
    return pred


if __name__ == '__main__':
    data_pullson.pull_sched(range(1999, 2024))
    data_pullson.pull_pbp([2023])

    season = 2023
    week = 22
    lookback = 20

    # pull_bt(30)
    # back_test(30)

    pred = run(season, week, lookback).round(1)

    h_to_the_tml(pred, season, week, lookback)

