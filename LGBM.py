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
# LGBM Regressor
import lightgbm as lgb
from lightgbm import LGBMRegressor


'''
Load dataset

'''
df = pd.read_csv('/home/jy/AGIC/df_AGIC_final.csv')
df.info()
df.describe()
df.columns # 'CO2air', 'EC_drain_PC', 'Tot_PAR', 'pH_drain_PC', 'water_sup','T_out', 'I_glob', 'RadSum', 'Windsp', 'Stem_elong', 'Cum_trusses']
df.set_index('time', inplace = True)

# train/test split
feature_columns = list(df.columns.difference(['Stem_elong', 'Cum_trusses'])) # target을 제외한 모든 행
X = df[feature_columns] # 설명변수
y = df['Stem_elong']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state = 42)


'''
Hyperparameter tunning

'''
# optuna
# random sampler
sampler = TPESampler(seed=10)

# define function
def objective(trial):

    lgbm_param = {
        'objective': 'regression',
        'verbose': -1,
        'metric': 'mse', 
        'num_leaves': trial.suggest_int('num_leaves', 2, 1024, step=1, log=True), 
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        'reg_alpha': trial.suggest_uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_uniform('reg_lambda', 0.0, 10.0),
        'max_depth': trial.suggest_int('max_depth',3, 15),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_loguniform('subsample', 0.3, 1),
    }

    # Generate model
    model_lgbm = LGBMRegressor(**lgbm_param)
    model_lgbm = model_lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                           verbose=0, early_stopping_rounds=25)
                           
    # 평가 지표
                        
    MSE = mean_squared_error(y_val, model_lgbm.predict(X_val))
    return MSE

optuna_lgbm = optuna.create_study(direction='minimize', sampler=sampler)

# * n_trials : optuna를 몇번 실행하여 hyperparameter를 찾을 것인지를 정한다.

optuna_lgbm.optimize(objective, n_trials=100)


# best trial
lgbm_trial = optuna_lgbm.best_trial
lgbm_trial_params = lgbm_trial.params
print('Best Trial: score {},\nparams {}'.format(lgbm_trial.value, lgbm_trial_params))


'''
Model

'''
# Modeling
# LGBM Regressor fit

lgbm = LGBMRegressor(**lgbm_trial_params)
lgbm_study = lgbm.fit(X_train, y_train)

# Predict the y_test
lgbm_ = lgbm_study.predict(X_test)

# results
rmse = np.sqrt(mean_squared_error(y_test, lgbm_))
print("RMSE: %f" % (rmse))  #RMSE: 0.019575

mae = mean_absolute_error(y_test, lgbm_)
print('mae: %f' %(mae))  # mae: 0.008041

'''
Best Trial: score 0.00046327132239537615,
params {'num_leaves': 875, 'colsample_bytree': 0.8102915024752978, 'reg_alpha': 0.0773548492375728, 'reg_lambda': 9.691018789040738, 'max_depth': 15, 'learning_rate': 0.0311986781539818, 'n_estimators': 1721, 'min_child_samples': 7, 'subsample': 0.5990412909930394}
RMSE: 0.019575
mae: 0.008041
'''

'''
# model save
import joblib
joblib.dump(lgbm_study,'/home/jy/AGIC/agic_lgbm_elong.pkl')

# 모델 불러옴
#file -> model load
model_from_joblib = joblib.load('/home/jy/AGIC/agic_lgbm_elong.pkl')
model_from_joblib.score(X_test, y_test)
'''
