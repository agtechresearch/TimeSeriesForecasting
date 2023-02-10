#%%
import pandas as pd
from pandas import DataFrame
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Dense, GRU, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import joblib
from bayes_opt import BayesianOptimization

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from keras.models import Sequential
from keras.layers import Reshape


'''
Load dataset

'''
df = pd.read_csv('/home/jy/TimeSeriesForecasting/1. AGIC/df_AGIC_final.csv')
df.info()
df.describe()
df.columns # 'CO2air', 'EC_drain_PC', 'Tot_PAR', 'pH_drain_PC', 'water_sup','T_out', 'I_glob', 'RadSum', 'Windsp', 'Stem_elong', 'Cum_trusses']
df.set_index('time', inplace = True)


# train/test split
feature_columns = list(df.columns.difference(['Stem_elong', 'Cum_trusses']))
X = df[feature_columns]
y = df['Stem_elong']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

# reshape input data
X_train = np.array(X_train.values, dtype=np.float32)
X_test = np.array(X_test.values, dtype=np.float32)
y_train = np.array(y_train.values, dtype=np.float32)
y_test = np.array(y_test.values, dtype=np.float32)


'''
Hyperparameter tuning

'''

# Optuna 사용
sampler = TPESampler(seed=10)

def optimize_lstm(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    units = trial.suggest_int("units", 32, 128)
    dropout = trial.suggest_float("dropout", 0, 0.5)
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    # Model
    model = Sequential()
    model.add(GRU(recurrent_dropout=0, return_sequences=True, units=units, dropout=dropout, input_shape=((X_train.shape[1],1))))
    model.add(Dropout(dropout))
    model.add(GRU(units=units, recurrent_dropout=0))
    model.add(Dropout(dropout))
    model.add(Reshape((units, 1)))
    model.add(GRU(units=units, recurrent_dropout=0))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error') 
    return model

def objective(trial):
    model = optimize_lstm(trial)
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    score = model.evaluate(X_test, y_test, verbose=2)
    return score

# create a new study and run the optimization
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100)

# print the best set of hyperparameters
gru_trial = study.best_trial
gru_trial_params = gru_trial.params
print('Best Trial: score {},\nparams {}'.format(gru_trial.value, gru_trial_params))

best_model = optimize_lstm(study.best_trial)
best_model.fit(X_train, y_train)
best_model.evaluate(X_test, y_test)
predictions = best_model.predict(X_test)

# results
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))  

mae = mean_absolute_error(y_test, predictions)
print('MAE: %f' %(mae)) 



'''
# Cum_trusses
Best Trial: score 0.005595719441771507,
params {'learning_rate': 0.0006914221141427193, 'units': 93, 'dropout': 0.001999350966253366, 'optimizer': 'SGD'}
RMSE: 0.167135
MAE: 0.125389

# Stem_elong
Best Trial: 
RMSE: 0.218812
mae: 0.171415
'''



#%%
# model save
#joblib.dump(best_model,'/home/jy/TimeSeriesForecasting/1. AGIC/agic_gru_truss.pkl')

# 모델 불러옴
# file -> model load
model_from_joblib = joblib.load('/home/jy/TimeSeriesForecasting/1. AGIC/agic_gru_truss.pkl')
model_from_joblib.evaluate(X_test, y_test)

