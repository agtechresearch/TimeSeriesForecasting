#%%
import pandas as pd
from pandas import DataFrame
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optuna Libraries
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
# XGBRegressor
import xgboost as xgb
from xgboost import XGBRegressor

#import graphviz
import matplotlib.pyplot as plt
import joblib
'''
Load dataset

'''
df = pd.read_csv('/home/jy/TimeSeriesForecasting/1. AGIC/df_AGIC_final.csv')
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
# Cum_trusses
Best Trial: score 0.00023683533514177354,
params {'max_depth': 9, 'n_estimators': 777, 'learning_rate': 0.1179188904994195, 'gamma': 2.6855360821175997e-05, 'min_child_weight': 1, 'subsample': 0.6750353047311403, 'colsample_bytree': 0.9182179002588413}
RMSE: 0.015415
mae: 0.005981

# stem_elong
Best Trial: score 0.0004260357895389751,
params {'max_depth': 11, 'n_estimators': 1786, 'learning_rate': 0.3375046483776262, 'gamma': 0.0001475455948565045, 'min_child_weight': 1, 'subsample': 0.941386985239264, 'colsample_bytree': 0.907770485785147}
RMSE: 0.017814
mae: 0.006787

'''



#%%
'''
Model
: Cum_truss

'''
# fit
xgboost = xgb.XGBRegressor(**xgb_trial_params)
xgb_truss = xgboost.fit(X_train, y_train)

# Predict the y_test
pred_truss = xgb_truss.predict(X_test)

# results
rmse = np.sqrt(mean_squared_error(y_test, pred_truss))
print("RMSE: %f" % (rmse))  #RMSE: 0.015415

mae = mean_absolute_error(y_test, pred_truss)
print('mae: %f' %(mae))  #mae: 0.005981

r2 = r2_score(y_test, pred_truss)
print('R2: %f' %(r2))  #R2: 0.996927


# model save
# joblib.dump(xgb_truss,'/home/jy/TimeSeriesForecasting/1. AGIC/1-1.Growth_Factors/agic_xgb_truss.pkl')


# model load
#file -> model load
agic_xgb_truss = joblib.load('/home/jy/TimeSeriesForecasting/1. AGIC/1-1.Growth_Factors/agic_xgb_truss.pkl')
abc = agic_xgb_truss.predict(X_test)
mae2 = mean_absolute_error(y_test, abc)
print('mae: %f' %(mae2))


# visualization 예측값과 실제값 비교 그래프
fit = np.polyfit(y_test, abc, 1)
fit_fn = np.poly1d(fit)

plt.scatter(y_test, abc)
plt.plot(y_test, fit_fn(y_test), '--k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values - Cum_trusses(XGB)")
plt.show() 
slope = fit[0]
print(slope)


# Feature importance
fig, ax = plt.subplots(figsize=(10, 10))
xgb.plot_importance(agic_xgb_truss, ax = ax, title='Feature importance - Cum_trusses(XGB)')
plt.show()





#%%
'''
Model
: Stem_elong

'''
# fit
xgboost = xgb.XGBRegressor(**xgb_trial_params)
xgb_elong = xgboost.fit(X_train, y_train)

# Predict the y_test
pred_elong = xgb_elong.predict(X_test)

# results
rmse = np.sqrt(mean_squared_error(y_test, pred_elong))
print("RMSE: %f" % (rmse))  #RMSE: 0.017814

mae = mean_absolute_error(y_test, pred_elong)
print('mae: %f' %(mae))  #mae: 0.006787

r2 = r2_score(y_test, pred_elong)
print('R2: %f' %(r2))  #R2: 0.994411


# model save
# joblib.dump(xgb_elong,'/home/jy/TimeSeriesForecasting/1. AGIC/agic_xgb_elong.pkl')

# model load
#file -> model load
agic_xgb_truss = joblib.load('/home/jy/TimeSeriesForecasting/1. AGIC/agic_xgb_elong.pkl')
abc = agic_xgb_truss.predict(X_test)
mae2 = mean_absolute_error(y_test, abc)
print('mae: %f' %(mae2))


# visualization 예측값과 실제값 비교 그래프
fit = np.polyfit(y_test, abc, 1)
fit_fn = np.poly1d(fit)

plt.scatter(y_test, abc)
plt.plot(y_test, fit_fn(y_test), '--k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values - Stem_elong(XGB)")
plt.show() 
slope = fit[0]
print(slope)


# Feature importance
fig, ax = plt.subplots(figsize=(10, 10))
xgb.plot_importance(agic_xgb_truss, ax = ax, title='Feature importance - Stem_elong(XGB)')
plt.show()


'''
안됨
# Plot metrics
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_importance.html
lgb.plot_metric(agic_lgbm_truss)


# Tree graph 
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_tree.html
lgb.plot_tree(agic_lgbm_truss, tree_index=0, figsize=(20, 10),
orientation='vertical',
show_info=['internal_count', 'leaf_count'])
plt.show()
'''

