import statistics as st
import matplotlib.dates as mdates
import pandas as pd
from pandas import DataFrame
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# Optuna Libraries
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
# XGBRegressor
import xgboost as xgb
from xgboost import XGBRegressor

'''
Load dataset

'''
df = pd.read_csv('/home/jy/AGIC/df_AGIC_final.csv')
df.info()
df.describe()
df.columns # 'CO2air', 'EC_drain_PC', 'Tot_PAR', 'pH_drain_PC', 'water_sup','T_out', 'I_glob', 'RadSum', 'Windsp', 'Stem_elong', 'Cum_trusses']
df.set_index('time', inplace = True)



'''
Hyperparameter tunning

'''
# train/test split
feature_columns = list(df.columns.difference(['Stem_elong', 'Cum_trusses'])) # target을 제외한 모든 행
X = df[feature_columns] # 설명변수
y = df['Stem_elong']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state = 42)


# optuna
# random sampler
sampler = TPESampler(seed=10)

# define function
def objective(trial):

    xgb_param = {
        'max_depth': trial.suggest_int('max_depth',5, 15),
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.5),
        'gamma': trial.suggest_float("gamma", 0, 2),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 5),
        'subsample': trial.suggest_float("subsample", 0.3, 1),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1)
    }

    # Generate model
    model_xgb = xgb.XGBRegressor(**xgb_param)
    model_xgb = model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                           verbose=0, early_stopping_rounds=25)
                           
    # 평가 지표
                        
    MSE = mean_squared_error(y_val, model_xgb.predict(X_val))
    return MSE

optuna_xgb = optuna.create_study(direction='minimize', sampler=sampler)


# * n_trials : optuna를 몇번 실행하여 hyperparameter를 찾을 것인지를 정한다.
optuna_xgb.optimize(objective, n_trials=100)


# best trial
xgb_trial = optuna_xgb.best_trial
xgb_trial_params = xgb_trial.params
print('Best Trial: score {},\nparams {}'.format(xgb_trial.value, xgb_trial_params))



'''
Model

'''
# fit
xgb = xgb.XGBRegressor(**xgb_trial_params)
xgb_study = xgb.fit(X_train, y_train)

# Predict the y_test
xgb_elong = xgb_study.predict(X_test)

# results
rmse = np.sqrt(mean_squared_error(y_test, xgb_elong))
print("RMSE: %f" % (rmse))  #RMSE: 0.022494

mae = mean_absolute_error(y_test, xgb_elong)
print('mae: %f' %(mae))  #mae: 0.011306
'''
Best Trial: score 0.0006779644881853906,
params {'max_depth': 11, 'n_estimators': 1628, 'learning_rate': 0.12571746929137662, 'gamma': 0.0038410403138717914, 'min_child_weight': 1, 'subsample': 0.3914614017029183, 'colsample_bytree': 0.8890387206211186}
RMSE: 0.022494
mae: 0.011306
'''

'''
# model save
import joblib
joblib.dump(xgb_study,'/home/jy/AGIC/agic_xgb_elong.pkl')

# 모델 불러옴
#file -> model load
model_from_joblib = joblib.load('/home/jy/AGIC/agic_xgb_elong.pkl')
model_from_joblib.score(X_test, y_test)
'''