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
# XGBRegressor
import xgboost as xgb
from xgboost import XGBRegressor

#import graphviz
import matplotlib.pyplot as plt
import joblib


'''
1. Load dataset

'''
df = pd.read_csv('/home/jy/TimeSeriesForecasting/1. AGIC/df_AGIC_final_forEnv.csv')
df.info()
df.describe()
df.columns
df.set_index('time', inplace = True)



'''
2. Hyperparameter tunning

'''
# train/test split
feature_columns = list(df.columns.difference(['Rhair'])) # target을 제외한 모든 행
X = df[feature_columns] # 설명변수
y = df['Rhair']

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
# Rhair
Best Trial: score 0.012476242854286678,
params {'max_depth': 8, 'n_estimators': 1848, 'learning_rate': 0.07958020388151651, 'gamma': 0.0035410309667650575, 'min_child_weight': 4, 
'subsample': 0.9781219337374943, 'colsample_bytree': 0.7394104721379644}

# CO2air
Best Trial: score 0.013814644221717236,
params {'max_depth': 10, 'n_estimators': 1205, 'learning_rate': 0.04086850654820532, 'gamma': 0.003282849690253046, 'min_child_weight': 5, 
'subsample': 0.8465202233877132, 'colsample_bytree': 0.9346589713732588}
'''



#%%
'''
3. Model
: Rhair

'''
# fit
xgboost = xgb.XGBRegressor(**xgb_trial_params)
xgb_rh = xgboost.fit(X_train, y_train)

# Predict the y_test
pred_rh = xgb_rh.predict(X_test)

# results
rmse = np.sqrt(mean_squared_error(y_test, pred_rh))
print("RMSE: %f" % (rmse)) #RMSE: 0.119050

mae = mean_absolute_error(y_test, pred_rh)
print('mae: %f' %(mae))  #mae: 0.078660

r2 = r2_score(y_test, pred_rh)
print('R2: %f' %(r2))  #R2: 0.652276


# model save
# joblib.dump(xgb_rh,'/home/jy/TimeSeriesForecasting/1. AGIC/1-1. Environment_Factors/agic_xgb_rh.pkl')


# model load
#file -> model load
agic_xgb_rh = joblib.load('/home/jy/TimeSeriesForecasting/1. AGIC/1-1. Environment_Factors/agic_xgb_rh.pkl')
abc = agic_xgb_rh.predict(X_test)
mae2 = mean_absolute_error(y_test, abc)
print('mae: %f' %(mae2))


# visualization 예측값과 실제값 비교 그래프
fit = np.polyfit(y_test, abc, 1)
fit_fn = np.poly1d(fit)

plt.scatter(y_test, abc)
plt.plot(y_test, fit_fn(y_test), '--k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values - RH(XGB)")
plt.show() 
slope = fit[0]
print(slope)


# Feature importance
fig, ax = plt.subplots(figsize=(10, 10))
xgb.plot_importance(agic_xgb_rh, ax = ax, title='Feature importance - RH(XGB)')
plt.show()




# %%
'''
4. Model
: CO2air

'''
# fit
xgboost = xgb.XGBRegressor(**xgb_trial_params)
xgb_co2 = xgboost.fit(X_train, y_train)

# Predict the y_test
pred_co2 = xgb_co2.predict(X_test)

# results
rmse = np.sqrt(mean_squared_error(y_test, pred_co2))
print("RMSE: %f" % (rmse)) #RMSE: 0.118433

mae = mean_absolute_error(y_test, pred_co2)
print('mae: %f' %(mae))  #mae: 0.081071

r2 = r2_score(y_test, pred_co2)
print('R2: %f' %(r2))  #R2: 0.573118


# model save
# joblib.dump(xgb_co2,'/home/jy/TimeSeriesForecasting/1. AGIC/1-1. Environment_Factors/agic_xgb_co2.pkl')


# model load
#file -> model load
agic_xgb_co2 = joblib.load('/home/jy/TimeSeriesForecasting/1. AGIC/1-1. Environment_Factors/agic_xgb_co2.pkl')
abc = agic_xgb_co2.predict(X_test)
mae2 = mean_absolute_error(y_test, abc)
print('mae: %f' %(mae2))


# visualization 예측값과 실제값 비교 그래프
fit = np.polyfit(y_test, abc, 1)
fit_fn = np.poly1d(fit)

plt.scatter(y_test, abc)
plt.plot(y_test, fit_fn(y_test), '--k')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values - CO2(XGB)")
plt.show() 
slope = fit[0]
print(slope)


# Feature importance
fig, ax = plt.subplots(figsize=(10, 10))
xgb.plot_importance(agic_xgb_co2, ax = ax, title='Feature importance - CO2(XGB)')
plt.show()
