## 온실 내 환경 변수 +  외부 환경 변수들을 활용하여 온실 내 환경을 예측한다. 
# Target parameters : CO2air, Rhair

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
1. Load dataset

'''
df = pd.read_csv('/home/jy/TimeSeriesForecasting/1. AGIC/df_AGIC_final_forEnv.csv')
df.info()
df.describe()
df.columns # ['Rhair', 'CO2air', 'T_out', 'PARout', 'Pyrgeo', 'RadSum', 'Winddir','Windsp']
df.set_index('time', inplace = True)

# train/test split
feature_columns = list(df.columns.difference(['Rhair'])) # target을 제외한 모든 행
X = df[feature_columns] # 설명변수
y = df['Rhair']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state = 42)

#%%
'''
2. Hyperparameter tunning

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
# CO2air
Best Trial: score 0.013381675176130052,
params {'num_leaves': 1016, 'colsample_bytree': 0.98148655810414, 'reg_alpha': 0.7580558383303997, 'reg_lambda': 7.276219698582098, 'max_depth': 13, 
'learning_rate': 0.03908497308349739, 'n_estimators': 1393, 'min_child_samples': 5, 'subsample': 0.44593430192371725}

# Rhair
Best Trial: score 0.011996210914935513,
params {'num_leaves': 327, 'colsample_bytree': 0.8227030050554357, 'reg_alpha': 0.4938093039728857, 'reg_lambda': 4.3596540050407455, 'max_depth': 12, 
'learning_rate': 0.05950961132268364, 'n_estimators': 1211, 'min_child_samples': 17, 'subsample': 0.5254201790943778}
'''



#%%
'''
3. Model
: CO2air

'''
# Modeling
# LGBM Regressor fit
lgbm = LGBMRegressor(**lgbm_trial_params)
lgbm_co2 = lgbm.fit(X_train, y_train)

# Predict the y_test
pred_co2 = lgbm_co2.predict(X_test)

# results
rmse = np.sqrt(mean_squared_error(y_test, pred_co2))
print("RMSE: %f" % (rmse))  #RMSE: 0.115160

mae = mean_absolute_error(y_test, pred_co2)
print('mae: %f' %(mae))  #mae: 0.080502

r2 = r2_score(y_test, pred_co2)
print('R2: %f' %(r2))  #R2: 0.596387


# model save
#joblib.dump(lgbm_co2,'/home/jy/TimeSeriesForecasting/1. AGIC/1-1. Environment_Factors/agic_lgbm_co2.pkl')


# model load
#file -> model load
agic_lgbm_co2 = joblib.load('/home/jy/TimeSeriesForecasting/1. AGIC/1-1. Environment_Factors/agic_lgbm_co2.pkl')
abc = agic_lgbm_co2.predict(X_test)
mae2 = mean_absolute_error(y_test, abc)
print('mae: %f' %(mae2))



# visualization 예측값과 실제값 비교 그래프
fit = np.polyfit(y_test, abc, 1)
fit_fn = np.poly1d(fit)

plt.scatter(y_test, abc)
plt.plot(y_test, fit_fn(y_test), '--k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values - CO2 (LGBM)")
plt.show() 
slope = fit[0]
print(slope)# 직선의 기울기가 1에 가깝다.



# Feature importance
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_importance.html
lgb.plot_importance(agic_lgbm_co2, figsize=(10, 10), title='Feature importance - CO2(LGBM)')
plt.show()



# %%
'''
4. Model
: Rhair

'''
# Modeling
# LGBM Regressor fit
lgbm = LGBMRegressor(**lgbm_trial_params)
lgbm_rh = lgbm.fit(X_train, y_train)

# Predict the y_test
pred_rh = lgbm_rh.predict(X_test)

# results
rmse = np.sqrt(mean_squared_error(y_test, pred_rh))
print("RMSE: %f" % (rmse))  #RMSE: 0.113277

mae = mean_absolute_error(y_test, pred_rh)
print('mae: %f' %(mae))  #mae: 0.074911

r2 = r2_score(y_test, pred_rh)
print('R2: %f' %(r2))  #R2: 0.685186



# model save
#joblib.dump(lgbm_rh,'/home/jy/TimeSeriesForecasting/1. AGIC/1-1. Environment_Factors/agic_lgbm_rh.pkl')


# model load
#file -> model load
agic_lgbm_rh = joblib.load('/home/jy/TimeSeriesForecasting/1. AGIC/1-1. Environment_Factors/agic_lgbm_rh.pkl')
abc = agic_lgbm_rh.predict(X_test)
mae2 = mean_absolute_error(y_test, abc)
print('mae: %f' %(mae2))



# visualization 예측값과 실제값 비교 그래프
fit = np.polyfit(y_test, abc, 1)
fit_fn = np.poly1d(fit)

plt.scatter(y_test, abc)
plt.plot(y_test, fit_fn(y_test), '--k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values - RH(LGBM)")
plt.show() 
slope = fit[0]
print(slope)# 직선의 기울기가 1에 가깝다.



# Feature importance
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_importance.html
lgb.plot_importance(agic_lgbm_rh, figsize=(10, 10), title='Feature importance - RH(LGBM)')
plt.show()


# %%
