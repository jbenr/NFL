import pandas as pd
import numpy as np
from tabulate import tabulate
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
import keras.losses

import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

import tensorflow as tf
import logging

# Set TensorFlow logging level to suppress warnings
tf.get_logger().setLevel(logging.ERROR)

def modelo(data, season, week):
    dat = data.copy()
    dat.loc[:,'result'] = dat['away_score']-dat['home_score']

    target = 'result'
    features = [
        "away_off_run_ypp","away_def_run_ypp",
        "away_off_pass_ypp","away_def_pass_ypp",
        "away_off_pass_completion_%","away_def_pass_completion_%",
        "away_off_series_success_%","away_def_series_success_%",
        "away_off_first_down_pp","away_def_first_down_pp",
        "away_off_third_down_%","away_def_third_down_%",
        "away_off_fourth_down_%","away_def_fourth_down_%",
        "away_off_turnovers_pp","away_def_turnovers_pp",
        "away_off_penalties_pp","away_def_penalties_pp",
        "away_off_passer_rating","away_def_passer_rating"
    ]

    preds = dat[(dat.season==season)&(dat.week==week)]
    train = dat[~((dat.season==season)&(dat.week==week))]

    X,Y = train[features], train[target]

    # Create loss function that penalizes wrong winner
    def sign_penalty(y_true, y_pred):
        penalty = 1.3
        loss = tf.where(tf.less(y_true * y_pred, 0), \
                        penalty * tf.square(y_true - y_pred), \
                        tf.square(y_true - y_pred))
        return tf.reduce_mean(loss, axis=-1)
    keras.losses.sign_penalty = sign_penalty  # enable use of loss with keras

    def create_model():
        model = Sequential()
        model.add(Dropout(0.1))
        model.add(Dense(X.shape[1], input_dim=X.shape[1], activation='elu'))
        # model.add(Dropout(0.1))
        model.add(Dense((X.shape[1] + 1) // 2, activation='elu'))
        model.add(Dense((X.shape[1] + 1) // 3, activation='elu'))
        model.add(Dense(1, activation='linear'))
        return model

    iterations = 100
    lst = []
    tf.keras.backend.clear_session()
    for i in range(iterations):
        model = create_model()
        opt = keras.optimizers.Adam(amsgrad=True)
        # model.compile(optimizer=opt, loss=sign_penalty, metrics=['accuracy'])
        model.compile(optimizer=opt, loss=sign_penalty)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, mode='auto')
        model.fit(X, Y, epochs=100, callbacks=[reduce_lr])

        train_preds = model.predict(X)
        test_preds = model.predict(preds[features])

        r2 = sklearn.metrics.r2_score(Y, train_preds)
        mae = sklearn.metrics.mean_absolute_error(Y, train_preds)
        mse = sklearn.metrics.mean_squared_error(Y, train_preds)
        # acc = sklearn.metrics.accuracy_score(Y, train_preds)
        print(f'\nR2: {r2}\nMAE:{mae}\nMSE:{mse}\n')

        if r2 > 0:
            preds.loc[:,'prediction'] = np.round(test_preds, 1)
            lst.append(preds[['away_team','home_team','prediction']])
        else: None

    results = pd.concat(lst)
    results = results.groupby(['away_team','home_team']).agg(['mean','var']).reset_index()

    results = results[['home_team', 'prediction', 'away_team']]
    # breakpoint()
    return results
