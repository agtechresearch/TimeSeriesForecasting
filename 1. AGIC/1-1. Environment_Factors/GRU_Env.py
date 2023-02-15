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
df = pd.read_csv('/home/jy/TimeSeriesForecasting/1. AGIC/df_AGIC_final_forEnv.csv')
df.info()
df.describe()
df.columns 
df.set_index('time', inplace = True)


# train/test split
feature_columns = list(df.columns.difference(['CO2air']))
X = df[feature_columns]
y = df['CO2air']

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

'''
# Rhair
Best Trial: score 0.029248211532831192,
params {'learning_rate': 0.05564687764340289, 
'units': 109, 'dropout': 0.052414907412672955, 'optimizer': 'Adam'}

# CO2air
Best Trial: score 0.02516219951212406,
params {'learning_rate': 0.0018169367080179285, 
'units': 128, 'dropout': 0.0011608096644537696, 'optimizer': 'RMSprop'}

'''

#%%
'''
Model
: Rhair
'''
best_model = optimize_lstm(study.best_trial)
best_model.fit(X_train, y_train)
best_model.evaluate(X_test, y_test)
predictions = best_model.predict(X_test)

# results
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))  #RMSE: 0.190659

mae = mean_absolute_error(y_test, predictions)
print('MAE: %f' %(mae)) #MAE: 0.153887

from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions)
print('R2: %f' %(r2))  #R2: 0.088711


# visualization 예측값과 실제값 비교 그래프
import matplotlib.pyplot as plt
fit = np.polyfit(y_test, predictions.ravel(), 1)  #predictions를 1D array로 바꿈
fit_fn = np.poly1d(fit)

plt.scatter(y_test, predictions)
plt.plot(y_test, fit_fn(y_test), '--k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.ylim(0, 1)
plt.title("Actual vs Predicted Values - RH(GRU)")
plt.show() 
slope = fit[0]
print(slope)

# model save
#joblib.dump(best_model,'/home/jy/TimeSeriesForecasting/1. AGIC/1-1. Environment_Factors/agic_gru_rh.pkl')

# 모델 불러옴
# file -> model load
model_from_joblib = joblib.load('/home/jy/TimeSeriesForecasting/1. AGIC/1-1. Environment_Factors/agic_gru_rh.pkl')
model_from_joblib.evaluate(X_test, y_test)

abc = model_from_joblib.predict(X_test)

rmse2 = np.sqrt(mean_squared_error(y_test, abc))
print("RMSE: %f" % (rmse2))




# %%
'''
Model
: CO2
'''
best_model = optimize_lstm(study.best_trial)
best_model.fit(X_train, y_train)
best_model.evaluate(X_test, y_test)
predictions = best_model.predict(X_test)

# results
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))  #RMSE: 0.171034

mae = mean_absolute_error(y_test, predictions)
print('MAE: %f' %(mae)) #MAE: 0.133952

from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions)
print('R2: %f' %(r2))  #R2: 0.116877

# visualization 예측값과 실제값 비교 그래프
import matplotlib.pyplot as plt
fit = np.polyfit(y_test, predictions.ravel(), 1)  #predictions를 1D array로 바꿈
fit_fn = np.poly1d(fit)

plt.scatter(y_test, predictions)
plt.plot(y_test, fit_fn(y_test), '--k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.ylim(0, 1)
plt.title("Actual vs Predicted Values - CO2(GRU)")
plt.show() 
slope = fit[0]
print(slope)


# model save
joblib.dump(best_model,'/home/jy/TimeSeriesForecasting/1. AGIC/1-1. Environment_Factors/agic_gru_co2.pkl')

# 모델 불러옴
# file -> model load
model_from_joblib = joblib.load('/home/jy/TimeSeriesForecasting/1. AGIC/1-1. Environment_Factors/agic_gru_co2.pkl')
model_from_joblib.evaluate(X_test, y_test)

abc = model_from_joblib.predict(X_test)

rmse2 = np.sqrt(mean_squared_error(y_test, abc))
print("RMSE: %f" % (rmse2))

# %%
