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
# LGBM Regressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

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
# Cum_trusses
Best Trial: score 0.00022382297334423235,
params {'num_leaves': 49, 'colsample_bytree': 0.8071915973989296, 'reg_alpha': 0.03860369689946567, 'reg_lambda': 6.221371632703895, 'max_depth': 14, 'learning_rate': 0.14958984196867808, 'n_estimators': 1866, 'min_child_samples': 7, 'subsample': 0.30075845637158566}
RMSE: 0.014794
mae: 0.007148

# stem_elong
Best Trial: score 0.0002360076267223417,
params {'num_leaves': 117, 'colsample_bytree': 0.7815178863536182, 'reg_alpha': 0.06293205178847364, 'reg_lambda': 0.610006371062484, 'max_depth': 13, 'learning_rate': 0.05185433766242212, 'n_estimators': 1026, 'min_child_samples': 30, 'subsample': 0.5185349928407017}
RMSE: 0.014939
mae: 0.007517
'''



#%%
'''
Model
: Cum_truss

'''
# Modeling
# LGBM Regressor fit
lgbm = LGBMRegressor(**lgbm_trial_params)
lgbm_truss = lgbm.fit(X_train, y_train)

# Predict the y_test
pred_truss = lgbm_truss.predict(X_test)

# results
rmse = np.sqrt(mean_squared_error(y_test, pred_truss))
print("RMSE: %f" % (rmse))  #RMSE: 0.014794

mae = mean_absolute_error(y_test, pred_truss)
print('mae: %f' %(mae))  #mae: 0.007148

r2 = r2_score(y_test, pred_truss)
print('R2: %f' %(r2))

# model save
# joblib.dump(lgbm_truss,'/home/jy/TimeSeriesForecasting/1. AGIC/1-1.Growth_Factors/agic_lgbm_truss.pkl')


# model load
#file -> model load
agic_lgbm_truss = joblib.load('/home/jy/TimeSeriesForecasting/1. AGIC/1-1.Growth_Factors/agic_lgbm_truss.pkl')
abc = agic_lgbm_truss.predict(X_test)
mae2 = mean_absolute_error(y_test, abc)
print('mae: %f' %(mae2))



# visualization 예측값과 실제값 비교 그래프
fit = np.polyfit(y_test, abc, 1)
fit_fn = np.poly1d(fit)

plt.scatter(y_test, abc)
plt.plot(y_test, fit_fn(y_test), '--k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values - Cum_trusses (LGBM)")
plt.show() 
slope = fit[0]
print(slope)# 직선의 기울기가 1에 가깝다.



# Feature importance
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_importance.html
lgb.plot_importance(agic_lgbm_truss, figsize=(10, 10), title='Feature importance - Cum_trusses(LGBM)')
plt.show()






#%%
'''
Model
: Stem_elong

'''
# Modeling
# LGBM Regressor fit
lgbm = LGBMRegressor(**lgbm_trial_params)
lgbm_elong = lgbm.fit(X_train, y_train)

# Predict the y_test
pred_elong = lgbm_elong.predict(X_test)

# results
rmse = np.sqrt(mean_squared_error(y_test, pred_elong))
print("RMSE: %f" % (rmse))   #RMSE: 0.014939

mae = mean_absolute_error(y_test, pred_elong)
print('mae: %f' %(mae))  #mae: 0.007517

r2 = r2_score(y_test, pred_elong)
print('R2: %f' %(r2))  #R2: 0.996070

# model save
# joblib.dump(lgbm_elong,'/home/jy/TimeSeriesForecasting/1. AGIC/1-1.Growth_Factors/agic_lgbm_elong.pkl')


# model load
#file -> model load
agic_lgbm_elong = joblib.load('/home/jy/TimeSeriesForecasting/1. AGIC/1-1.Growth_Factors/agic_lgbm_elong.pkl')
abc = agic_lgbm_elong.predict(X_test)
mae2 = mean_absolute_error(y_test, abc)
print('mae: %f' %(mae2))



# visualization 예측값과 실제값 비교 그래프
fit = np.polyfit(y_test, abc, 1)
fit_fn = np.poly1d(fit)

plt.scatter(y_test, abc)
plt.plot(y_test, fit_fn(y_test), '--k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values - Stem_elong (LGBM)")
plt.show() 
slope = fit[0]
print(slope)# 직선의 기울기가 1에 가깝다.



# Feature importance
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_importance.html
lgb.plot_importance(agic_lgbm_elong, figsize=(10, 10), title='Feature importance - Stem_elong(LGBM)')
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

