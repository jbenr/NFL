import pandas as pd
import numpy as np
import utils
from opt_einsum.blas import tensor_blas
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
    defense = defense.groupby('team').apply(
        lambda x: (x['def_qb_elo'] * x['weight']).sum() / x['weight'].sum()
    )
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

    guy_weighted = guy.groupby('name').apply(
        lambda x: (x['qb_elo'] * x['weight']).sum() / x['weight'].sum()
    ).reset_index(name='weighted_qb_elo')

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
                    try: df[col] = (away[f'away_off_{col_}'][0] - home[f'home_def_{col_}'][0])*\
                                   (away['away_off_pass_%'][0]+0.5)
                    except Exception as e: pass
                elif 'run' in col:
                    try: df[col] = (away[f'away_off_{col_}'][0] - home[f'home_def_{col_}'][0])*\
                                   (away['away_off_run_%'][0]+0.5)
                    except Exception as e: pass
                else:
                    try: df[col] = away[f'away_off_{col_}'][0] - home[f'home_def_{col_}'][0]
                    except Exception as e: pass
            elif 'def' in col:
                try: df[col] = away[f'away_def_{col_}'][0] - home[f'home_off_{col_}'][0]
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
                   'referee']]
    sched[~((sched.season==szn) & (sched.week>week))]

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

    pbp = [pd.read_parquet(f'data/pbp/pbp_{szn}.parquet') for szn in df_.season.unique().tolist()]
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

        sched_ = df.query(f'season=={s} & week=={w}')
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
    # num_workers = max(1, num_cores // 2)
    num_workers = 1
    print(f'Num workers: {num_workers} from {num_cores} cores!')
    args_list = [(s, w, lookback, pbp, sched, df) for s, w in tings]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(executor.map(calculate_stats, args_list), total=len(args_list), desc="Crunching the numbers"))

    data = pd.concat(results).reset_index(drop=True)
    print(tabulate(data.tail(3),headers='keys',tablefmt=tabulate_formats[4]))
    return data


# print(tabulate(prep_test_train(2024, 14, 10).tail(5),headers='keys',tablefmt=tabulate_formats[1]))