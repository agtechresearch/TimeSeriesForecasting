#%%
import pandas as pd
from pandas import DataFrame
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 42)

# reshape input data
X_train = np.array(X_train.values, dtype=np.float32)
#X_train = X_train.reshape(int(len(X_train)/3), 3, 10) #(samples, timestep, feature)

X_test = np.array(X_test.values, dtype=np.float32)
#X_test = X_test.reshape(int(len(X_test)/3), 3, 10)

y_train = np.array(y_train.values, dtype=np.float32)
y_test = np.array(y_test.values, dtype=np.float32)

#%%
'''
Hyperparameter tuning

'''
# https://inside-machinelearning.com/en/optuna-tutorial/
# https://dacon.io/en/codeshare/4646

# Optuna 사용
sampler = TPESampler(seed=10)

def optimize_lstm(trial):
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
    model = optimize_lstm(trial)
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    score = model.evaluate(X_test, y_test, verbose=2)
    return score

# create a new study and run the optimization
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100)

# print the best set of hyperparameters
lstm_trial = study.best_trial
lstm_trial_params = lstm_trial.params
print('Best Trial: score {},\nparams {}'.format(lstm_trial.value, lstm_trial_params))
'''
# Cum_trusses
Best Trial: score 0.00923875905573368,
params {'learning_rate': 0.01340053110487026, 'units': 112, 'dropout': 0.01623560670538488, 'optimizer': 'SGD'}


# Stem_elong
Best Trial: score 0.008794150315225124,
params {'learning_rate': 0.09522026435797089, 'units': 102, 'dropout': 0.017851993438105437, 'optimizer': 'SGD'}
'''


#%%
'''
Model
: Cum_trusses
'''
best_model = optimize_lstm(study.best_trial)
best_model.fit(X_train, y_train)
best_model.evaluate(X_test, y_test)
pred_truss = best_model.predict(X_test)


# results
rmse = np.sqrt(mean_squared_error(y_test, pred_truss))
print("RMSE: %f" % (rmse))  #RMSE: 0.221339

mae = mean_absolute_error(y_test, pred_truss)
print('mae: %f' %(mae))  #mae: 0.176222

from sklearn.metrics import r2_score
r2 = r2_score(y_test, pred_truss)
print('R2: %f' %(r2))  #R2: 0.372126



# visualization 예측값과 실제값 비교 그래프
import matplotlib.pyplot as plt
fit = np.polyfit(y_test, pred_truss.ravel(), 1)  #predictions를 1D array로 바꿈
fit_fn = np.poly1d(fit)

plt.scatter(y_test, pred_truss)
plt.plot(y_test, fit_fn(y_test), '--k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values - Cum_trusses(LSTM)")
plt.show() 
slope = fit[0]
print(slope)


# model save
# joblib.dump(best_model,'/home/jy/TimeSeriesForecasting/1. AGIC/agic_lstm_cum_truss.pkl')

# 모델 불러옴
#file -> model load
import joblib
model_from_joblib = joblib.load('/home/jy/TimeSeriesForecasting/1. AGIC/agic_lstm_cum_truss.pkl')
model_from_joblib.evaluate(X_test, y_test)






# %%
'''
Model
: Stem_elong
'''
best_model = optimize_lstm(study.best_trial)
best_model.fit(X_train, y_train)
best_model.evaluate(X_test, y_test)
predictions = best_model.predict(X_test)


# results
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))  #RMSE: 0.235915

mae = mean_absolute_error(y_test, predictions)
print('mae: %f' %(mae))  #mae: 0.188903

from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions)
print('R2: %f' %(r2))  #R2: 0.039779




#%%
# visualization 예측값과 실제값 비교 그래프
import matplotlib.pyplot as plt
fit = np.polyfit(y_test, predictions.ravel(), 1)  #predictions를 1D array로 바꿈
fit_fn = np.poly1d(fit)

plt.scatter(y_test, predictions)
plt.plot(y_test, fit_fn(y_test), '--k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.ylim(0, 1)  #y축 범위
plt.title("Actual vs Predicted Values - Stem_elong(LSTM)")
plt.show() 
slope = fit[0]
print(slope)
# %%
'''
# model save
import joblib
joblib.dump(best_model,'/home/jy/TimeSeriesForecasting/1. AGIC/agic_lstm_elong.pkl')
'''
# 모델 불러옴
#file -> model load
import joblib
lstm_elong = joblib.load('/home/jy/TimeSeriesForecasting/1. AGIC/agic_lstm_elong.pkl')
lstm_elong.evaluate(X_test, y_test)

