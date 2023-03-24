#%%
import pandas as pd
from pandas import DataFrame
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from keras.models import Sequential
from keras.layers import Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_data(file_path):
    df = pd.read_csv(file_path)
    df.set_index('time', inplace=True)
    return df

def split_data(df, target_col, test_size=0.4, random_state=42):
    feature_columns = list(df.columns.difference([target_col]))
    X = df[feature_columns].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def optimize_lstm(trial, X_train, y_train):
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
    units = trial.suggest_int("units", 32, 128)
    dropout = trial.suggest_float("dropout", 0, 0.5)
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])

    #recurrent_dropout = trial.suggest_float("recurrent_dropout", 0, 0.5)

    # LSTM model 생성
    model = Sequential()
    model.add(LSTM(recurrent_dropout=0, return_sequences=True, units=units, dropout=dropout, input_shape=((X_train.shape[1],1))))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, recurrent_dropout=0))
    model.add(Dropout(dropout))
    model.add(Reshape((units, 1)))
    model.add(LSTM(units=units, recurrent_dropout=0))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# 모델 트레이닝 및 성능 평가
def objective(trial):
    X_train, X_test, y_train, y_test = split_data(df, target_col)
    model = optimize_lstm(trial, X_train, y_train)
    # ModelCheckpoint 콜백 추가
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
    # EarlyStopping 콜백 추가
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=0, validation_data=(X_test, y_test),
              callbacks=[checkpoint, early_stop])
    # 저장된 가장 좋은 모델 불러오기
    model.load_weights('best_model.h5')
    score = model.evaluate(X_test, y_test, verbose=2)
    return score


# %%


