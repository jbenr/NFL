import pandas as pd
import numpy as np
import utils
# from opt_einsum.blas import tensor_blas
from tabulate import tabulate, tabulate_formats
from scipy import stats
from datetime import datetime

from tqdm import tqdm
import concurrent.futures
import os
import re


def slicer1(df, play_type, group, stat, agg):
    df1 = df.copy()

    if type(play_type)==type(['doo','doo']): df1 = df.loc[df['play_type'].isin(play_type)]
    elif type(play_type) == type('derp'): df1 = df.loc[df['play_type']==play_type]

    if type(group)==type(['dumbo','']):
        if agg == 'count': df1 = df1.groupby(group).agg(agg)
        else: df1 = df1.groupby(group).agg(agg, numeric_only=True)
    else:
        if agg == 'count': df1 = df1.groupby([group]).agg(agg)
        else: df1 = df1.groupby([group]).agg(agg, numeric_only=True)

    df1.index.name = 'team'
    return df1[stat]


def slicer2(df, play_type, group, stat, agg):
    df1 = df.copy()

    # Filter by play_type
    if isinstance(play_type, list):
        df1 = df1.loc[df1['play_type'].isin(play_type)]
    elif isinstance(play_type, str):
        df1 = df1.loc[df1['play_type'] == play_type]

    # Calculate weights based on game_date
    if 'game_date' in df1.columns:
        max_date = pd.to_datetime(df1['game_date']).max()
        df1['weight'] = df1['game_date'].apply(lambda x: (max_date - pd.to_datetime(x)).days)
        df1['weight'] = df1['weight'].rank(ascending=False)  # Rank for higher weights to recent dates
        df1['weight'] /= df1['weight'].sum()  # Normalize weights to sum to 1
    else:
        df1['weight'] = 1  # Default to equal weights if game_date is missing

    # Grouping and aggregation
    if isinstance(group, list):
        if agg == 'count':
            df1 = df1.groupby(group).size().rename(stat)
        elif agg == 'mean':
            df1 = df1.groupby(group).apply(
                lambda x: (x[stat] * x['weight']).sum() / x['weight'].sum()
            )
        elif agg == 'sum':
            df1 = df1.groupby(group).apply(lambda x: (x[stat] * x['weight']).sum())
    else:
        if agg == 'count':
            df1 = df1.groupby([group]).size().rename(stat)
        elif agg == 'mean':
            df1 = df1.groupby([group]).apply(
                lambda x: (x[stat] * x['weight']).sum() / x['weight'].sum()
            )
        elif agg == 'sum':
            df1 = df1.groupby([group]).apply(lambda x: (x[stat] * x['weight']).sum())

    df1.index.name = 'team'
    return df1

def gradual_acceleration_with_floor(days_from_max, total_season_days=130, steepness=5, floor_weight=0.05):
    # Normalize days_from_max to a 0–1 scale relative to total_season_days
    normalized_days = days_from_max / total_season_days

    # Apply a polynomial decay with a floor for older games
    weights = np.exp(-steepness * normalized_days**3)
    weights = np.maximum(weights, floor_weight)  # Ensure weights don't drop below the floor

    # Normalize weights to sum to 1
    normalized_weights = weights / weights.sum()
    return normalized_weights


def slicer(df, play_type, group, stat, agg,
           total_season_days=160, steepness=3, floor_weight=0.05):
    df1 = df.copy()

    # Filter by play_type
    if isinstance(play_type, list):
        df1 = df1.loc[df1['play_type'].isin(play_type)]
    elif isinstance(play_type, str):
        df1 = df1.loc[df1['play_type'] == play_type]

    # Calculate days from max_date and apply weights
    if 'game_date' in df1.columns:
        max_date = pd.to_datetime(df1['game_date']).max()
        df1['days_from_max'] = (max_date - pd.to_datetime(df1['game_date'])).dt.days

        # Compute weights using gradual acceleration decay with a floor
        df1['weight'] = gradual_acceleration_with_floor(
            df1['days_from_max'].values,
            total_season_days=total_season_days,
            steepness=steepness,
            floor_weight=floor_weight
        )
    else:
        df1['weight'] = 1  # Default to equal weights if game_date is missing

    # Grouping and weighted aggregation
    group_cols = group if isinstance(group, list) else [group]
    keys = [df1[c] for c in group_cols]  # array-like keys to group the Series

    if agg == 'count':
        df1 = df1.groupby(group_cols).size().rename(stat)

    elif agg == 'sum':
        wx = df1[stat] * df1['weight']  # Σ(x * w)
        df1 = wx.groupby(keys).sum()

    elif agg == 'mean':
        wx = (df1[stat] * df1['weight'])  # Σ(x * w) / Σ(w)
        num = wx.groupby(keys).sum()
        den = df1['weight'].groupby(keys).sum()
        df1 = num / den

    df1.index.name = 'team'
    return df1


def calc_stats(df):
    guy = df[df['play_type']=='run'].groupby('posteam').agg(
        'mean',numeric_only=True)[['yards_gained']]                                         # Off run yards per play
    guy.columns = ['off_run_ypp']
    guy.index.name = 'team'

    guy['def_run_ypp'] = slicer(df, 'run', 'defteam', 'yards_gained', 'mean')               # Def run yards per play

    guy['off_pass_ypp'] = slicer(df, 'pass', 'posteam', 'yards_gained', 'mean')             # Off pass yards per play
    guy['def_pass_ypp'] = slicer(df, 'pass', 'defteam', 'yards_gained', 'mean')             # Def pass yards per play

    guy['off_run_%'] = slicer(df,'run', 'posteam', 'yards_gained', 'count')/\
                      slicer(df,['run','pass'], 'posteam', 'yards_gained', 'count')            # Off run %
    guy['off_pass_%'] = slicer(df,'pass', 'posteam', 'yards_gained', 'count')/\
                      slicer(df,['run','pass'], 'posteam', 'yards_gained', 'count')            # Off pass %

    guy['off_pass_completion_%'] = slicer(df,'pass', 'posteam', 'complete_pass', 'sum')/\
                                   slicer(df,'pass', 'posteam', 'complete_pass', 'count')      # Off pass completion %
    guy['def_pass_completion_%'] = slicer(df,'pass', 'defteam', 'complete_pass', 'sum')/\
                                   slicer(df,'pass', 'defteam', 'complete_pass', 'count')      # Def pass completion %

    temp = slicer(df,None, ['game_id','series','posteam'], 'series_success', 'mean').reset_index()
    temp = temp.rename(columns={0: 'series_success'})
    guy['off_series_success_%'] = \
        slicer(temp,None, 'posteam', 'series_success', 'sum')/\
        slicer(temp,None, 'posteam', 'series_success', 'count')             # Off series sucess %
    temp = slicer(df,None, ['game_id','series','defteam'], 'series_success', 'mean').reset_index()
    temp = temp.rename(columns={0: 'series_success'})
    guy['def_series_success_%'] = \
        slicer(temp,None, 'defteam', 'series_success', 'sum')/\
        slicer(temp,None, 'defteam', 'series_success', 'count')             # Def series sucess %

    guy['off_first_down_pp'] = slicer(df,['run','pass'], 'posteam', 'first_down', 'sum')/\
                               slicer(df,['run','pass'], 'posteam', 'first_down', 'count')     # Off first downs per play
    guy['def_first_down_pp'] = slicer(df,['run','pass'], 'defteam', 'first_down', 'sum')/\
                               slicer(df,['run','pass'], 'defteam', 'first_down', 'count')     # Def first downs per play

    guy['off_third_down_%'] = slicer(df,['run','pass'], 'posteam', 'third_down_converted', 'sum')/\
                          (slicer(df,['run','pass'], 'posteam', 'third_down_converted', 'sum')+\
                           slicer(df,['run','pass'], 'posteam', 'third_down_failed', 'sum'))   # Off 3rd down %
    guy['def_third_down_%'] = slicer(df,['run','pass'], 'defteam', 'third_down_converted', 'sum')/\
                          (slicer(df,['run','pass'], 'defteam', 'third_down_converted', 'sum')+\
                           slicer(df,['run','pass'], 'defteam', 'third_down_failed', 'sum'))   # Def 3rd down %

    guy['off_fourth_down_%'] = slicer(df,['run','pass'], 'posteam', 'fourth_down_converted', 'sum')/\
                          (slicer(df,['run','pass'], 'posteam', 'fourth_down_converted', 'sum')+\
                           slicer(df,['run','pass'], 'posteam', 'fourth_down_failed', 'sum'))   # Off 3rd down %
    guy['def_fourth_down_%'] = slicer(df,['run','pass'], 'defteam', 'fourth_down_converted', 'sum')/\
                          (slicer(df,['run','pass'], 'defteam', 'fourth_down_converted', 'sum')+\
                           slicer(df,['run','pass'], 'defteam', 'fourth_down_failed', 'sum'))   # Def 3rd down %

    p_types = ['kickoff', 'run', 'pass', 'punt','field_goal','extra_point']
    guy['off_turnovers_pp'] = (slicer(df,p_types,'posteam', 'interception', 'sum')+\
                              slicer(df,p_types, 'posteam', 'fumble_lost', 'sum'))/\
                              slicer(df,p_types, 'posteam', 'interception', 'count')           # Off turnovers per play
    guy['def_turnovers_pp'] = (slicer(df,p_types, 'defteam', 'interception', 'sum')+\
                              slicer(df,p_types, 'defteam', 'fumble_lost', 'sum'))/\
                              slicer(df,p_types, 'defteam', 'interception', 'count')           # Def turnovers per play

    guy['off_penalties_pp'] = slicer(df,None, 'posteam','penalty','sum')/\
                              slicer(df,None, 'posteam', 'penalty', 'count')                   # Off penalties per play
    guy['def_penalties_pp'] = slicer(df,None, 'defteam','penalty','sum')/\
                              slicer(df,None, 'defteam', 'penalty', 'count')                   # Def penalties per play

    df1 = df.copy()
    df1['drive_sec_of_possession'] = (pd.to_datetime(df1['drive_time_of_possession'], format='%M:%S').dt.second) + (
                pd.to_datetime(df1['drive_time_of_possession'], format='%M:%S').dt.minute * 60)

    temp = slicer(df1, None, ['game_id', 'drive', 'posteam'], 'drive_sec_of_possession', 'mean').reset_index()
    temp = temp.rename(columns={0: 'drive_sec_of_possession'})
    temp = slicer(temp, None, ['game_id', 'posteam'], 'drive_sec_of_possession', 'sum').reset_index()
    temp = temp.rename(columns={0: 'drive_sec_of_possession'})
    temp['drive_sec_of_possession'] = temp['drive_sec_of_possession'] / 3600
    guy['off_possession_%'] = slicer(temp, None, 'posteam', 'drive_sec_of_possession', 'mean')  # Possession %

    # # special teams
    # guy['punt_avg'] = df.loc[df['play_type']=='punt'].groupby(['posteam']).agg(
    # 'mean', numeric_only=True)['kick_distance']
    # guy['return_avg'] = df.loc[df['play_type']=='punt'].groupby(['return_team']).agg(
    # 'mean', numeric_only=True)['return_yards']
    # # field goals
    # # PAT 1, FG Missed -1, 0-39 3, 40-49 4, 50-59 5, 60+ 6
    # fg = np.where(df[''])
    # guy['field_goals'] = df.loc[df['play_type']=='field_goal'].groupby(['posteam']).agg('mean',numeric_only=True)['']
    # print(tabulate(guy.tail(10),headers='keys',tablefmt=tabulate_formats[2]))

    # New features (same math)
    guy['off_explosive_run_%'] = slicer(df[(df['play_type'] == 'run') & (df['yards_gained'] >= 10)],
                                        'run', 'posteam', 'yards_gained', 'count') / \
                                 slicer(df, 'run', 'posteam', 'yards_gained', 'count')
    guy['def_explosive_run_%'] = slicer(df[(df['play_type'] == 'run') & (df['yards_gained'] >= 10)],
                                        'run', 'defteam', 'yards_gained', 'count') / \
                                 slicer(df, 'run', 'defteam', 'yards_gained', 'count')

    guy['off_explosive_pass_%'] = slicer(df[(df['play_type'] == 'pass') & (df['yards_gained'] >= 20)],
                                         'pass', 'posteam', 'yards_gained', 'count') / \
                                  slicer(df, 'pass', 'posteam', 'yards_gained', 'count')
    guy['def_explosive_pass_%'] = slicer(df[(df['play_type'] == 'pass') & (df['yards_gained'] >= 20)],
                                         'pass', 'defteam', 'yards_gained', 'count') / \
                                  slicer(df, 'pass', 'defteam', 'yards_gained', 'count')

    guy['off_stuff_%'] = slicer(df[(df['play_type'] == 'run') & (df['yards_gained'] <= 0)],
                                'run', 'posteam', 'yards_gained', 'count') / \
                         slicer(df, 'run', 'posteam', 'yards_gained', 'count')
    guy['def_stuff_%'] = slicer(df[(df['play_type'] == 'run') & (df['yards_gained'] <= 0)],
                                'run', 'defteam', 'yards_gained', 'count') / \
                         slicer(df, 'run', 'defteam', 'yards_gained', 'count')

    guy['off_sack_%'] = slicer(df, 'pass', 'posteam', 'sack', 'sum') / \
                       slicer(df, 'pass', 'posteam', 'sack', 'count')
    guy['def_sack_%'] = slicer(df, 'pass', 'defteam', 'sack', 'sum') / \
                       slicer(df, 'pass', 'defteam', 'sack', 'count')

    guy['off_qb_hit_%'] = slicer(df, 'pass', 'posteam', 'qb_hit', 'sum') / \
                         slicer(df, 'pass', 'posteam', 'qb_hit', 'count')
    guy['def_qb_hit_%'] = slicer(df, 'pass', 'defteam', 'qb_hit', 'sum') / \
                         slicer(df, 'pass', 'defteam', 'qb_hit', 'count')

    return guy


def calc_ngs(qb, sched):
    # qb_cols = ['season','week','team_abbr','avg_time_to_throw','passer_rating','avg_air_distance']
    qb_cols = ['season','week','team_abbr','passer_rating']

    away = pd.merge(sched, qb[qb_cols], how='right',
                    left_on=['season','week','away_team'],
                    right_on=['season','week','team_abbr'])
    away = away.loc[:, ~away.columns.duplicated()].copy().dropna(subset=['away_team']) # rid duplicate col names
    away = away.rename(columns={i:f'off_{i}' for i in qb.columns if i not in ['season','week']})

    home = pd.merge(sched, qb[qb_cols], how='right',
                   left_on=['season','week','home_team'],
                   right_on=['season','week','team_abbr'])
    home = home.loc[:, ~home.columns.duplicated()].copy().dropna(subset=['away_team']) # rid duplicate col names
    home = home.rename(columns={i:f'def_{i}' for i in qb.columns if i not in ['season','week']})

    guy = pd.merge(away, home, on=sched.columns.tolist(), how='left')
    guy = guy[
        ['away_team']+
        [f'off_{i}' for i in qb_cols if i not in ['season','week','team_abbr']]+
        [f'def_{i}' for i in qb_cols if i not in ['season','week','team_abbr']]
    ].groupby(['away_team']).agg('mean')
    guy.index.name = 'team'

    return guy


def calc_qb_elo(df_, sched_, total_season_days=160, steepness=3, floor_weight=0.05):
    sched = sched_.merge(df_[['season', 'week']], on=['season', 'week']).drop_duplicates()
    sched = pd.merge(sched, df_[['season', 'week', 'game_date', 'home_team']], on=['season', 'week', 'home_team'], how='left').drop_duplicates()
    sched['away_qb_short'] = sched.away_qb_name.apply(lambda x: f"{x.split()[0][0]}.{x.split()[1]}")
    sched['home_qb_short'] = sched.home_qb_name.apply(lambda x: f"{x.split()[0][0]}.{x.split()[1]}")

    p = df_.groupby(['season', 'week', 'game_date', 'passer']).agg({
        'qb_scramble': 'sum',
        'rushing_yards': 'sum',
        'incomplete_pass': 'sum',
        'complete_pass': 'sum',
        'passing_yards': 'sum',
        'pass_touchdown': 'sum',
        'interception': 'sum',
        'sack': 'sum'
    }).reset_index().rename(columns={'passer': 'name', 'rushing_yards': 'scramble_yards'})

    r = df_.groupby(['season', 'week', 'game_date', 'rusher']).agg({
        'rush_attempt': 'sum',
        'rushing_yards': 'sum',
        'rush_touchdown': 'sum'
    }).reset_index().rename(columns={'rusher': 'name'})

    guy = pd.merge(p, r, how='left', on=['season', 'week', 'game_date', 'name'])
    guy['pass_attempt'] = guy.incomplete_pass + guy.complete_pass
    guy['rush_attempt'] = guy.rush_attempt + guy.qb_scramble
    guy['rushing_yards'] = guy.rushing_yards + guy.scramble_yards
    guy.drop(columns=['qb_scramble', 'scramble_yards', 'incomplete_pass'], inplace=True)
    guy.fillna(0, inplace=True)

    # QB ELO formula
    guy['qb_elo'] = (-2.2 * guy.pass_attempt + 3.7 * guy.complete_pass + guy.passing_yards / 5 +
                     11.3 * guy.pass_touchdown - 14.1 * guy.interception - 8 * guy.sack -
                     1.1 * guy.rush_attempt + 0.6 * guy.rushing_yards + 15.9 * guy.rush_touchdown)

    sched = pd.merge(sched, guy[['season', 'week', 'name', 'qb_elo']], how='left',
                     left_on=['season', 'week', 'away_qb_short'], right_on=['season', 'week', 'name']
                     ).rename(columns={'qb_elo': 'away_qb_elo'}).drop(columns='name')
    sched = pd.merge(sched, guy[['season', 'week', 'name', 'qb_elo']], how='left',
                     left_on=['season', 'week', 'home_qb_short'], right_on=['season', 'week', 'name']
                     ).rename(columns={'qb_elo': 'home_qb_elo'}).drop(columns='name')

    away = sched[['season', 'week', 'away_team', 'home_qb_elo', 'game_date']].rename(
        columns={'away_team': 'team', 'home_qb_elo': 'def_qb_elo'}
    )
    home = sched[['season', 'week', 'home_team', 'away_qb_elo', 'game_date']].rename(
        columns={'home_team': 'team', 'away_qb_elo': 'def_qb_elo'}
    )
    defense = pd.concat([away, home]).sort_values(by=['season', 'week', 'team'])

    max_date = pd.to_datetime(defense['game_date']).max()
    defense['days_from_max'] = (max_date - pd.to_datetime(defense['game_date'])).dt.days
    defense['weight'] = gradual_acceleration_with_floor(
        defense['days_from_max'].values,
        total_season_days=total_season_days,
        steepness=steepness,
        floor_weight=floor_weight
    )
    # print(tabulate(defense,headers='keys',tablefmt=tabulate_formats[4]))

    def_mean = (defense['def_qb_elo'] * defense['weight']).sum() / defense['weight'].sum()
    wx = (defense['def_qb_elo'] * defense['weight'])
    num = wx.groupby(defense['team']).sum()
    den = defense['weight'].groupby(defense['team']).sum()
    defense = (num / den)
    defense -= def_mean
    defense = defense.reset_index(name='def_qb_elo')

    max_date = pd.to_datetime(guy['game_date']).max()
    guy['days_from_max'] = (max_date - pd.to_datetime(guy['game_date'])).dt.days
    guy['weight'] = gradual_acceleration_with_floor(
        guy['days_from_max'].values,
        total_season_days=total_season_days,
        steepness=4,
        floor_weight=0.4
    )
    # print(tabulate(guy,headers='keys',tablefmt=tabulate_formats[4]))

    wx = guy['qb_elo'] * guy['weight']
    num = wx.groupby(guy['name']).sum()
    den = guy['weight'].groupby(guy['name']).sum()
    guy_weighted = (num / den).reset_index(name='weighted_qb_elo')

    # sched = sched.merge(sched,guy,left_on=['away_qb_short'])

    # print(tabulate(guy_weighted, headers='keys', tablefmt=tabulate_formats[2]))
    # print(tabulate(defense.reset_index(), headers='keys', tablefmt=tabulate_formats[2]))
    # print(tabulate(sched.tail(10),headers='keys',tablefmt=tabulate_formats[2]))

    return guy_weighted, defense


# helpers for comp_stats
def rank_it(x):
    x = x.fillna(x.median())
    return stats.rankdata(x,'average')/len(x)
def rev_rank_it(x):
    x = x.fillna(x.median())
    return (len(x) - stats.rankdata(x, "average") + 1)/len(x)

def comp_stats(stats, sched):
    # take calc'd stats and create a metric that can be dabbled upon
    stats_ = stats.copy()

    skips = ['run_%','pass_%']
    exceptions = ['turnovers','penalties']
    for col in stats_.columns:
        if any(skip in col for skip in skips): pass
        else:
            if "off" in col:
                if any(exc in col for exc in exceptions): stats_[col] = rev_rank_it(stats_[col])
                else: stats_[col] = rank_it(stats_[col])
            elif "def" in col:
                if any(exc in col for exc in exceptions): stats_[col] = rank_it(stats_[col])
                else: stats_[col] = rev_rank_it(stats_[col])

    df_ = []
    for away, home in sched.groupby(['away_team','home_team']).agg('count').index:
        away = stats_[stats_.index==away]
        home = stats_[stats_.index==home]
        away.columns = [f'away_{col}' for col in away.columns]
        home.columns = [f'home_{col}' for col in home.columns]

        df = pd.DataFrame(None)
        df['away_team'] = away.index
        df['home_team'] = home.index
        for col in away.columns.tolist():
            col_ = col[9:]
            if 'off' in col:
                if 'pass' in col:
                    try: df[col] = (away[f'away_off_{col_}'].iloc[0] - home[f'home_def_{col_}'].iloc[0])*\
                                   (away['away_off_pass_%'].iloc[0]+0.5)
                    except Exception as e: pass
                elif 'run' in col:
                    try: df[col] = (away[f'away_off_{col_}'].iloc[0] - home[f'home_def_{col_}'].iloc[0])*\
                                   (away['away_off_run_%'].iloc[0]+0.5)
                    except Exception as e: pass
                else:
                    try: df[col] = away[f'away_off_{col_}'].iloc[0] - home[f'home_def_{col_}'].iloc[0]
                    except Exception as e: pass
            elif 'def' in col:
                try: df[col] = away[f'away_def_{col_}'].iloc[0] - home[f'home_off_{col_}'].iloc[0]
                except Exception as e: print(e)
            else: print(f'no off or def in {col}')
        df_.append(df)
    df = pd.concat(df_)
    sched = pd.merge(sched, df, how='left', on=['away_team','home_team'])

    return sched


def prep_test_train(szn, week, lookback):
    sched = pd.read_parquet('data/sched.parquet')
    sched = sched[['season','week','game_type','away_team','away_score','home_team','home_score','away_rest','home_rest',
                   'roof','surface','temp','wind','away_qb_name','home_qb_name','away_coach','home_coach',
                   'referee','location']]
    sched = sched.loc[~((sched['season'] == szn) & (sched['week'] > week))].copy()

    df = []
    szn_, week_, lookback_ = szn, week, lookback
    while lookback_ >= 0:
        temp = sched.query(f'season=={szn_} & week=={week_}')
        df.append(temp)
        week_ -= 1
        if week_ <= 0: szn_ -= 1; week_ = sched[sched.season==szn_].week.max()
        if sched.query(f'season=={szn_} & week=={week_}')['game_type'].unique()[0] == 'REG': lookback_ -= 1

    df = pd.concat(df)
    df_ = []
    szn_, week_, lookback_ = df.season.min(), df.week.min(), lookback
    while lookback_ >= 0:
        temp = sched.query(f'season=={szn_} & week=={week_}')
        df_.append(temp)
        week_ -= 1
        if week_ <= 0: szn_ -= 1; week_ = sched[sched.season == szn_].week.max()
        if sched.query(f'season=={szn_} & week=={week_}')['game_type'].unique()[0] == 'REG': lookback_ -= 1

    df_ = pd.concat([df]+df_)

    pbp = []
    for szn in df_.season.unique().tolist():
        try:
            df_pbp = pd.read_parquet(f'data/pbp/pbp_{szn}.parquet')
            pbp.append(df_pbp)
        except FileNotFoundError:
            print(f"⚠️ File not found for season {szn}. Skipping.")
            continue
    pbp = pd.concat(pbp)

    # ngs = pd.read_parquet(f'data/ngs_passing.parquet')

    tings = df.groupby(['season', 'week']).agg('count').index.tolist()
    print(tings)

    def calculate_stats(args):
        s, w, lookback, pbp, sched, df = args
        # print(f'Calculating stats for szn:{s} week:{w}')

        pbp_, qbr_ = [], []
        s_, w_, lb_ = s, w - 1, lookback

        while lb_ > 0:
            temp = pbp.query(f'season=={s_} & week=={w_}')
            pbp_.append(temp)
            qbr_.append(temp)

            w_ -= 1
            if w_ <= 0: s_ -= 1; w_ = sched[sched.season == s_].week.max()
            if sched.query(f'season=={s_} & week=={w_}')['game_type'].unique()[0] == 'REG': lb_ -= 1

        pbp_ = pd.concat(pbp_)
        calc = calc_stats(pbp_)

        qbr_ = pd.concat(qbr_)
        qb, dee = calc_qb_elo(qbr_, sched)

        sched_ = df.query(f'season=={s} & week=={w}').copy()
        sched_.loc[:, 'away_qb_short'] = sched_['away_qb_name'].apply(lambda x: f"{x.split()[0][0]}.{x.split()[1]}")
        sched_.loc[:, 'home_qb_short'] = sched_['home_qb_name'].apply(lambda x: f"{x.split()[0][0]}.{x.split()[1]}")

        qb_team_map = pd.concat([
            sched_[['away_qb_short', 'away_team']].rename(columns={'away_qb_short': 'qb', 'away_team': 'team'}),
            sched_[['home_qb_short', 'home_team']].rename(columns={'home_qb_short': 'qb', 'home_team': 'team'})
        ])

        qb['name'] = qb['name'].apply(utils.strip_suffix)

        qb = pd.merge(qb, qb_team_map, left_on='name', right_on='qb', how='left').drop(columns='qb').sort_values(by='team').reset_index(drop=True)
        utils.make_dir('data/qb')
        qb.to_parquet(f'data/qb/qb_{s}_{w}_{lookback}.parquet')

        calc = pd.merge(calc, qb, on='team', how='left').rename(columns={'weighted_qb_elo':'off_qb_elo'}).drop(columns='name')
        calc = pd.merge(calc, dee, on='team', how='left').set_index('team')

        comp = comp_stats(calc, sched_)
        return comp

    tings = df.groupby(['season', 'week']).agg('count').index.tolist()
    num_cores = os.cpu_count()
    num_workers = max(1, num_cores // 4)
    # num_workers = 1
    print(f'Num workers: {num_workers} from {num_cores} cores!')
    args_list = [(s, w, lookback, pbp, sched, df) for s, w in tings]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(executor.map(calculate_stats, args_list), total=len(args_list), desc="Crunching the numbers"))

    data = pd.concat(results).reset_index(drop=True)
    print(tabulate(data.tail(3),headers='keys',tablefmt=tabulate_formats[4]))
    return data


import numpy as np
import pandas as pd

# --------------------------
# Config / metadata
# --------------------------
TEAM_META = {  # team: (conference, division)
    "BUF": ("AFC","EAST"), "MIA": ("AFC","EAST"), "NE":  ("AFC","EAST"), "NYJ": ("AFC","EAST"),
    "BAL": ("AFC","NORTH"),"CIN": ("AFC","NORTH"),"CLE": ("AFC","NORTH"),"PIT": ("AFC","NORTH"),
    "HOU": ("AFC","SOUTH"),"IND": ("AFC","SOUTH"),"JAX": ("AFC","SOUTH"),"TEN": ("AFC","SOUTH"),
    "DEN": ("AFC","WEST"), "KC":  ("AFC","WEST"), "LV":  ("AFC","WEST"), "LAC": ("AFC","WEST"),
    "DAL": ("NFC","EAST"), "NYG": ("NFC","EAST"), "PHI": ("NFC","EAST"), "WAS": ("NFC","EAST"),
    "CHI": ("NFC","NORTH"),"DET": ("NFC","NORTH"),"GB":  ("NFC","NORTH"),"MIN": ("NFC","NORTH"),
    "ATL": ("NFC","SOUTH"),"CAR": ("NFC","SOUTH"),"NO":  ("NFC","SOUTH"),"TB":  ("NFC","SOUTH"),
    "ARI": ("NFC","WEST"), "LAR": ("NFC","WEST"), "SEA": ("NFC","WEST"), "SF":  ("NFC","WEST"),
    # tolerate "LA" if it appears in historical data as Rams
    "LA":  ("NFC","WEST"),
}

# Any playoff game is max-importance
NON_REG_MAX = {"WC", "DIV", "CON", "CONF", "SB"}

# --------------------------
# Helpers
# --------------------------
def _records_pregame(df):
    """
    Per-game pregame record strings/wins/losses/ties for home & away.
    Only counts games with both scores present; ties allowed.
    """
    d = df.copy()
    have_score = d["home_score"].notna() & d["away_score"].notna()
    home_win = (d["home_score"] > d["away_score"]) & have_score
    tie      = (d["home_score"] == d["away_score"]) & have_score

    rows = []
    for side, opp, win_expr in [("home","away",home_win), ("away","home",~home_win & ~tie & have_score)]:
        r = d.loc[:, ["season","week",f"{side}_team"]].rename(columns={f"{side}_team":"team"})
        r["win"] = win_expr.astype(int)
        r["tie"] = tie.astype(int)
        r["played"] = have_score.astype(int)
        rows.append(r)
    long = pd.concat(rows, ignore_index=True)

    long = long.sort_values(["season","team","week"])
    grp = long.groupby(["season","team"], sort=False)
    long["W_pg"] = grp["win"].cumsum().shift(fill_value=0)
    long["T_pg"] = grp["tie"].cumsum().shift(fill_value=0)
    long["G_pg"] = grp["played"].cumsum().shift(fill_value=0)
    long["L_pg"] = (long["G_pg"] - long["W_pg"] - long["T_pg"]).clip(lower=0)

    rec = long[["season","week","team","W_pg","L_pg","T_pg"]].copy()
    rec["record_str"] = rec["W_pg"].astype(int).astype(str) + "-" + rec["L_pg"].astype(int).astype(str)
    has_t = rec["T_pg"] > 0
    rec.loc[has_t, "record_str"] += "-" + rec.loc[has_t, "T_pg"].astype(int).astype(str)
    return rec

def _conf_div(df):
    safe = lambda t: TEAM_META.get(t, ("UNK","UNK"))
    c_home, d_home = zip(*df["home_team"].map(safe))
    c_away, d_away = zip(*df["away_team"].map(safe))
    df["conf_home"], df["div_home"] = c_home, d_home
    df["conf_away"], df["div_away"] = c_away, d_away
    df["is_div"] = (df["div_home"] == df["div_away"]).astype(int)
    df["same_conf"] = (df["conf_home"] == df["conf_away"]).astype(int)
    return df

def _reg_weeks(df):
    """Dynamic number of REG weeks per season (handles 16/17/18...)."""
    return df[df["game_type"]=="REG"].groupby("season")["week"].max()

def _nth_largest(arr, n):
    """n is 1-based (7-> seventh largest). Returns -inf if not enough elements."""
    if len(arr) < n:
        return -np.inf
    idx = np.argpartition(arr, -n)[-n:]
    return arr[idx].min()

def _build_h2h_proxy(df):
    """
    Build head-to-head proxy for current season, pregames only.
    Returns a dict keyed by (season, team, opp) -> {+1: team has H2H edge, -1: opp edge, 0: none}.
    We use completed earlier meeting(s) between the same two teams; 1 win edge => +1, -1 => -1, else 0.
    """
    d = df.copy()
    played = d["home_score"].notna() & d["away_score"].notna()
    d = d[played]

    # Expand to long pairs for counting H2H wins
    a = d[["season","week","home_team","away_team","home_score","away_score"]].copy()
    a["home_win"] = (a["home_score"] > a["away_score"]).astype(int)
    a["away_win"] = (a["away_score"] > a["home_score"]).astype(int)

    # For each direction
    ha = a.groupby(["season","home_team","away_team"], as_index=False)["home_win"].sum().rename(
        columns={"home_team":"team","away_team":"opp","home_win":"wins_as_home"})
    aw = a.groupby(["season","away_team","home_team"], as_index=False)["away_win"].sum().rename(
        columns={"away_team":"team","home_team":"opp","away_win":"wins_as_away"})

    comb = pd.merge(ha, aw, on=["season","team","opp"], how="outer").fillna(0)
    comb["h2h_wins"] = comb["wins_as_home"] + comb["wins_as_away"]

    # Also compute opponent's h2h wins quickly by flipping keys
    comb_flip = comb.rename(columns={"team":"opp","opp":"team","h2h_wins":"opp_h2h_wins"})[["season","team","opp","opp_h2h_wins"]]
    comb = comb.merge(comb_flip, on=["season","team","opp"], how="left").fillna(0)

    # Edge: sign of (my wins - opp wins). Map to -1/0/+1
    comb["edge"] = np.sign(comb["h2h_wins"] - comb["opp_h2h_wins"]).astype(int)

    # Dict
    return {(int(r.season), str(r.team), str(r.opp)): int(r.edge) for r in comb.itertuples(index=False)}

def _early_floor(row, side):
    """
    Early/mid-season baseline so games don't collapse to 0.10 when hope exists.
    Scales with division > conference > inter-conference + a 'hope' term
    based on the optimistic cutline (win+help feasible).
    """
    is_div = float(row["is_div"])
    same_conf = float(row["same_conf"])

    wins   = float(row.get(f"{side}_wins", 0.0) or 0.0)
    cutopt = float(row.get(f"{side}_cutoff7_opt", 0.0) or 0.0)
    left   = float(row.get(f"{side}_games_left", 0.0) or 0.0)

    hope_raw = (wins + left) - cutopt          # positive => there exists a path to be ≥ cutline
    hope_clip = float(np.clip((hope_raw + 1.0)/3.0, 0.0, 1.0))  # smooth 3-win band

    late = float(row["late_w"])
    early_weight = 1.0 - late

    base = 0.12 + 0.10*same_conf + 0.20*is_div    # 0.12 IC, 0.22 conf, 0.32 div
    floor_val = base + 0.18*hope_clip

    return float(floor_val * early_weight)

# --------------------------
# Main engine
# --------------------------
def matchup_importance(sched: pd.DataFrame) -> pd.DataFrame:
    df = sched.copy()
    df = _conf_div(df)

    # Pregame records for all relevant games (REG+PO)
    rec = _records_pregame(df[df["game_type"].isin(["REG","WC","DIV","CON","CONF","SB"])])
    for side in ("home","away"):
        m = rec.rename(columns={
            "team":f"{side}_team","W_pg":f"{side}_wins","L_pg":f"{side}_losses","T_pg":f"{side}_ties","record_str":f"{side}_record"
        })
        df = df.merge(m, on=["season","week",f"{side}_team"], how="left")

    # Dynamic season lengths and remaining REG games (inclusive of current week)
    reg_last = _reg_weeks(df)
    df["reg_last_week"] = df["season"].map(reg_last).fillna(df["week"])
    df["home_games_left"] = (df["reg_last_week"] - df["week"] + 1).clip(lower=0).astype(int)
    df["away_games_left"] = df["home_games_left"].astype(int)

    # Late-season scaler
    with np.errstate(divide='ignore', invalid='ignore'):
        df["late_w"] = (df["week"] / df["reg_last_week"].replace(0, np.nan)).clip(0,1).fillna(0)

    # Head-to-head proxy (this season)
    h2h = _build_h2h_proxy(df)

    def side_block(g: pd.DataFrame, side: str) -> pd.DataFrame:
        # Conference name for this side
        conf_name = g[f"conf_{side}"].iloc[0]

        # All rows in this (season, week) that belong to this conference on either side
        all_rows = df[
            (df["season"] == g["season"].iloc[0]) &
            (df["week"]   == g["week"].iloc[0]) &
            ((df["conf_home"] == conf_name) | (df["conf_away"] == conf_name))
        ]

        # Snapshot per team in conference: current wins and remaining games
        hh = all_rows[["home_team","home_wins","home_ties","home_games_left"]].rename(
            columns={"home_team":"team","home_wins":"wins","home_ties":"ties","home_games_left":"left"})
        aa = all_rows[["away_team","away_wins","away_ties","away_games_left"]].rename(
            columns={"away_team":"team","away_wins":"wins","away_ties":"ties","away_games_left":"left"})
        snap = pd.concat([hh,aa], ignore_index=True).groupby("team", as_index=False).max(numeric_only=True)

        # Ensure full conference membership (16 teams historically; tolerate variations)
        conf_teams = [t for t,(c,_) in TEAM_META.items() if c == conf_name]
        if len(snap) < len(conf_teams):
            miss = sorted(set(conf_teams) - set(snap["team"]))
            if miss:
                # choose modal remaining among present
                gr_mode = int(pd.Series(snap["left"]).mode(dropna=True).iloc[0]) if len(snap)>0 else 0
                snap = pd.concat([snap, pd.DataFrame({"team":miss,"wins":0,"ties":0,"left":gr_mode})], ignore_index=True)

        # Compute others' max_final = wins + left (ties ignored for simplicity)
        snap["max_final"] = snap["wins"].astype(int) + snap["left"].astype(int)

        # For volatility: where is the current conf-best?
        best_now = int(snap["wins"].max().item()) if len(snap) else 0

        # For each row in g, compute cutlines and flags
        out_rows = []
        for idx, row in g.iterrows():
            me    = row[f"{side}_team"]
            w     = int(row.get(f"{side}_wins", 0) or 0)
            left  = int(row.get(f"{side}_games_left", 0) or 0)
            opp   = row["away_team"] if side=="home" else row["home_team"]

            # Others' max_final excluding me
            arr_others = snap.set_index("team")["max_final"].drop(index=me, errors="ignore").values

            base_cut = _nth_largest(arr_others, 7)  # 7th largest among others

            # Tiebreak proxy:
            #   - Pessimistic: assume ties break against you (+0.5)
            #   - Optimistic:  assume ties break for you (-0.5)
            #   - H2H edge: small nudge (-0.25 if you have edge; +0.25 if you are behind)
            h2h_edge = h2h.get((int(row["season"]), str(me), str(opp)), 0)
            cut_pess = base_cut + 0.5 + ( -0.25 * h2h_edge )  # if edge=+1, reduce pess cut a bit
            cut_opt  = base_cut - 0.5 + ( -0.25 * h2h_edge )

            # Alive assessments
            alive_pess = (w + left >= cut_pess)     # can still tie/exceed pess cut? (strict-ish due to +0.5)
            alive_opt  = (w + left >= cut_opt)      # optimistic board

            # Strict elimination only if even optimistic path fails
            elim_strict = (not alive_opt)

            # Clinched under pessimistic (already above the pessimistic 7-seed cut)
            clinch_pess = (w > cut_pess)

            # Must-win under pessimistic (to keep pessimistic path open)
            mustwin_pess = (not elim_strict) and (not clinch_pess) and (left > 0) and (w + (left - 1) < cut_pess)

            # Help paths
            help_path_live = (not alive_pess) and alive_opt

            # Crude division-title-with-help proxy: still in division race but need help; weight only for in-conf/div slates
            div_title_help_live = bool(help_path_live) and bool(row["same_conf"])

            # Volatility & “elasticity” proxies
            seed_vol = float(np.clip((best_now - w) / max(best_now, 1), 0, 1))
            # normalized rank width proxy (bigger possible swing -> more elastic)
            better = int((snap["wins"] > (w + left)).sum())
            worse  = int((snap["wins"] < w).sum())
            best_rank  = 1 + better
            worst_rank = len(snap) - worse
            seed_elastic_n = float(np.clip((worst_rank - best_rank) / max(len(snap)-1, 1), 0, 1))

            # Top-1 & seed-locked proxies
            top1_locked = (best_rank == 1 and worst_rank == 1)
            seed_locked = (best_rank == worst_rank)

            out_rows.append({
                "idx": idx,
                f"{side}_cutoff7_pess": float(cut_pess),
                f"{side}_cutoff7_opt":  float(cut_opt),
                f"{side}_elim":         bool(elim_strict),
                f"{side}_clinched":     bool(clinch_pess),
                f"{side}_mustwin":      bool(mustwin_pess),
                f"{side}_help_path_live": bool(help_path_live),
                f"{side}_div_title_help_live": bool(div_title_help_live),
                f"{side}_seed_vol":     seed_vol,
                f"{side}_seed_elastic_n": seed_elastic_n,
                f"{side}_best_rank":    int(best_rank),
                f"{side}_worst_rank":   int(worst_rank),
                f"{side}_top1_locked":  bool(top1_locked),
                f"{side}_seed_locked":  bool(seed_locked),
            })

        out = pd.DataFrame(out_rows).set_index("idx").sort_index()
        return out

    # Apply side logic per (season, week, conference_of_side)
    side_frames = []
    for side in ("home","away"):
        key = ["season","week",f"conf_{side}"]
        side_out = df.groupby(key, group_keys=False).apply(lambda g: side_block(g, side))
        side_frames.append(side_out)

    extra = pd.concat(side_frames, axis=1)
    df = pd.concat([df, extra], axis=1)

    # Division/conf weight (for importance blend)
    df["home_div_w"] = np.where(df["is_div"].astype(bool), 1.0, np.where(df["same_conf"].astype(bool), 0.6, 0.3))
    df["away_div_w"] = df["home_div_w"]

    # --------------------------
    # Importance formula (updated)
    # --------------------------
    def _importance(row, side):
        if row["game_type"] in NON_REG_MAX:
            return 1.0

        wins    = int(row.get(f"{side}_wins", 0) or 0)
        cut_p   = float(row.get(f"{side}_cutoff7_pess", 0.0) or 0.0)
        me_left = int(row.get(f"{side}_games_left", 0) or 0)

        lev = float(np.clip(1 - max(0.0, (cut_p - wins)) / max(cut_p, 1.0), 0, 1))

        help_bonus     = 1.0 if bool(row.get(f"{side}_help_path_live", False)) else 0.0
        div_help_bonus = 1.0 if bool(row.get(f"{side}_div_title_help_live", False)) else 0.0
        help_term = 0.25 * max(help_bonus, div_help_bonus) * float(row["late_w"])  # stronger late

        base = (0.30*lev
                + 0.20*float(row[f"{side}_div_w"])
                + 0.15*float(row["late_w"])
                + 0.15*float(row.get(f"{side}_seed_vol", 0.0))
                + 0.15*float(row.get(f"{side}_seed_elastic_n", 0.0))
                + help_term)

        # Clamps & floors
        if bool(row.get(f"{side}_elim", False)):
            # If truly eliminated (even optimistic path fails)
            if row["late_w"] > 0.7:
                base = min(base, 0.12)
            else:
                base = min(base, 0.20)

        if bool(row.get(f"{side}_top1_locked", False)):
            base = 0.05

        if bool(row.get(f"{side}_seed_locked", False)):
            base = min(base, 0.06)

        if bool(row.get(f"{side}_mustwin", False)):
            base = max(base, 0.97)

        if bool(row.get(f"{side}_help_path_live", False)) and (row["late_w"] > 0.7) and (not bool(row.get(f"{side}_elim", False))):
            base = max(base, 0.82)

        if bool(row.get(f"{side}_div_title_help_live", False)) and row["late_w"] > 0.7:
            base = max(base, 0.82)

        # Early/mid-season floor to prevent 0.10 spam when hope exists
        base = max(base, _early_floor(row, side))

        return float(np.clip(base, 0, 1))

    df["home_importance"] = df.apply(lambda r: _importance(r, "home"), axis=1)
    df["away_importance"] = df.apply(lambda r: _importance(r, "away"), axis=1)
    df["importance_diff"] = df["home_importance"] - df["away_importance"]

    # Output columns
    keep = [
        "game_id","season","game_type","week",
        "home_team","home_record","away_team","away_record",
        "home_importance","away_importance","importance_diff",

        # elimination / clinch / must-win + late flags
        "home_elim","home_clinched","home_mustwin","away_elim","away_clinched","away_mustwin",
        "home_help_path_live","away_help_path_live","home_div_title_help_live","away_div_title_help_live",

        # dynamic season context
        "home_games_left","away_games_left","late_w",

        # cutlines (pess/opt) for transparency
        "home_cutoff7_pess","home_cutoff7_opt","away_cutoff7_pess","away_cutoff7_opt",

        # rank/seed diagnostics
        "home_best_rank","home_worst_rank","home_top1_locked","home_seed_locked",
        "away_best_rank","away_worst_rank","away_top1_locked","away_seed_locked",
    ]

    # Some columns may be missing for years with incomplete data; fill if needed
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan if c.endswith(("_pess","_opt","_rank","_left","late_w")) else False

    return df[keep].sort_values(["season","week","game_id"]).reset_index(drop=True)


def additional_features(df_):
    df = df_.copy()
    # imp = matchup_importance(df)
    # df = df.merge(
    #     imp[[
    #         "game_id","home_record","away_record",
    #         "home_importance","away_importance","importance_diff"
    #     ]],
    #     on="game_id", how="left"
    # )

    df["away_rest_adv"] = df["away_rest"] - df["home_rest"]
    df["home_field_adv"] = (df["location"] == "Home").astype(int) if "location" in df.columns else 0
    # df["away_game_importance"] = df["away_importance"] - df["home_importance"]
    return df


# sched = pd.read_parquet("data/sched.parquet")
# imp = matchup_importance(sched)
# utils.pdf(imp[(imp["season"]==2025)&(imp["week"]==11)][[
#     "home_team","home_record","away_team","away_record",
#     "home_importance","away_importance"
# ]])
# utils.pdf(imp[(imp["season"]==2024)&(imp["week"]>=18)][[
#     "home_team","home_record","away_team","away_record",
#     "home_importance","away_importance"
# ]])
# utils.pdf(imp[(imp["season"]==2024)&(imp["week"]>=18)])
